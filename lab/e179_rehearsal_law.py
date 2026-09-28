"""E179 — THE REHEARSAL-FREQUENCY LAW: how little practice suffices? (the
maintenance threshold r* — the quantitative completion of the capacity story).

WHY (T107's maintenance direction + T119's rate-law neighbor): e174 showed
1:1 interleaved rehearsal MAINTAINS the fact (g0 0.92+ alongside F2's formed
graft) while its absence kills on the FIRST gradient — and e176n/e180 priced
the wash itself (the basin is ~5e-3 wide; t* ~ lr^-1.16; the neutral stream
dies at +2 under lr 1e-3). Capacity = the MAINTENANCE BUDGET. The one number
still missing is the budget's DENSITY: the rehearsal fraction r* below which
the wash wins — T107's surviving direction, now as a curve. QUEUE row e179
(verbatim): "THE REHEARSAL-FREQUENCY LAW (how little practice suffices? the
maintenance threshold) | READY (4-5 CPU trainings ~8 min each) | e176's
stream with F1-replay interleaved at r in {0, 1/32, 1/8, 1/4, 1/2}; F1
trajectory per rate. THRESHOLD (a sharp r*) vs GRADED (proportional
slowing) vs PUMP-FIT (w and g extracted)". The dispatch realizes "e176's
stream" as e176N arm A's NEUTRAL wash (the wash with the contradiction
channel discharged — the correct control; the dispatch's DESIGN line names
"the neutral wash stream" and READ names e176n as the wash to counter).

REGISTERED PREDICTION (the dispatch's registration VERBATIM; no bar
shopping — adjudicate against exactly this):
  - THRESHOLD fires if: a sharp r* exists (below it the fact dies by +50;
    at or above it maintains >= 0.5) — a maintenance phase transition.
  - GRADED fires if: any rehearsal slows the wash proportionally (no
    threshold; F1's +300 level rises smoothly with r).
  - PUMP-FIT fires if: both a wash rate w and a rehearsal gain g are
    extractable (the maintenance law's two constants).
  - No bar shopping; texture => TEXTURE with the curve.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * r = the fraction of a cell's 300 optimizer steps that are F1-REPLAY
    batches, scheduled as one replay batch per k wash-side steps (replay at
    every step with step % k == 0; the step REPLACES that step's wash
    batch). Nominal r = 1/k; the REALIZED r = n_replay/300 is co-reported
    (k=32 -> 9 events -> 0.030; k=8 -> 37 -> 0.123; k=4 -> 75 -> 0.250).
  * F1 ruler = g-12, the ABSOLUTE install-60 battery mean p(Z) at ctx
    offset -12 (e176/e176n/e180's convention verbatim — the same batteries,
    the same corpus rebuild, the same ruler). Die bar 0.27 (e158/e161/
    e176/e176n/e180); maintain bar 0.5 (e176n's NEUTRAL-SURVIVES constant).
  * dies(r) := g-12 AT THE +50 CHECKPOINT <= 0.27 — e176n's NEUTRAL-
    DISSOLVES convention VERBATIM (the dispatch's "the fact dies by +50"
    read as the +50 STATE, exactly as e176n operationalized its dispatch's
    identical phrase). SMOKE CORRECTION (pre-main; nothing adjudicated in
    smoke): the first frozen draft read "some checkpoint in (0,50] under
    0.27", which makes the PRE-REPLAY eviction dip (steps 1..k-1 are wash
    steps for every k >= 4, and the wash's two-step clock dips g-12 to
    ~0.03 at +2 before the first replay event) count as death for EVERY
    rate — structurally unfireable and contrary to the ancestor's
    state-based reading; re-frozen to the +50 cell BEFORE the main compute.
  * maintains(r) := g-12 >= 0.5 at BOTH registered horizons, +50 AND +300
    (the state-based reading of "at or above it maintains >= 0.5": the
    fact is present at both horizon states; a transient dip between
    replays does not un-maintain, and phase-luck at one horizon does not
    alone maintain).
  * THRESHOLD := the three-rate ladder {1/32 < 1/8 < 1/4} splits as a
    SUFFIX: some r* in the ladder maintains while EVERY measured rate below
    r* dies by +50 (r* = the smallest maintaining rate; the bracket
    (previous rate, r*] co-reported). All-rates-maintain does NOT fire it
    (no dying side exists) — recorded as texture with r* <= 1/32 noted.
  * GRADED := THRESHOLD did not fire AND the +300 g-12 levels are strictly
    increasing over {1/32, 1/8, 1/4} AND the smallest rehearsal already
    moves the curve (level(1/32, +300) > the wash's stored +300 level
    0.00382; the t*-based disjunct of the first draft is vacuous under
    re-teach dynamics — every rate's first-under sits at +2 in the
    pre-replay eviction dip — and is dropped by the same smoke correction).
  * t*(r) := the FIRST measured checkpoint with g-12 <= 0.27 (None if
    never); the wash's stored t* = 2 (e176n arm A, grid {1,2,4,...}) —
    CO-REPORTED timing texture only (see the smoke correction above).
  * CYCLE STATS (co-reported, not bar-adjudicated; phase-honesty): the mean
    and min g-12 over the late checkpoints {100, 200, 300} per rate — the
    fixed-grid horizons sit at different replay-cycle phases by schedule
    (k=32's +300 is 12 wash steps after its last replay; k=8's is 4; k=4's
    IS a replay step), so the late-grid mean/min carry the sustained level
    the endpoint alone cannot.
  * PUMP-FIT := a pooled regression-through-origin over ALL consecutive-
    checkpoint segments of the three rate cells: d(g-12) = w*(wash steps in
    segment) + g*(replay steps in segment); fires iff w < 0 AND g > 0 AND
    uncentered R^2 >= 0.80 (R^2 = 1 - SSres/SSy with SSy = sum y^2, the
    standard origin-regression convention — frozen here). A log-domain
    variant (d log10(g-12 + 0.01)) is CO-REPORTED as annotation only.
  * Adjudication order: THRESHOLD -> GRADED -> TEXTURE (the shape); PUMP-FIT
    is a separate co-verdict (e174's "ladder + rehearsal" two-verdict
    precedent); final verdict = "<SHAPE> + <PUMP-FIT|NO-PUMP>". Every
    sub-boolean reported regardless.
  * The stored endpoints: r=0 IS e176n arm A's stored trace (embedded,
    verified vs file); r=1/2 IS e174 arm B's dose-300 dial on the same
    g-12 ruler (0.7100) — STREAM-MISMATCHED (arm B's interleaved partner
    is the F2-INSTALL batch, a STRONGER interference than the neutral wash;
    the 1/2 point is therefore a conservative floor for the curve's end).
    A fourth training — the r=0 wash REPLICATE — is a RIDER (rig validation
    + same-run anchor; G_REPRO vs e176n arm A's stored cells); the bars
    anchor on the STORED trace as registered either way.

DESIGN: e176N arm A's NEUTRAL wash stream (e170's neutral bank: 16 plain-
corpus windows, RNG seed 170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL
in [s, s+257); 0/16 host content, 0/16 junctions; batch 32 = 16 neutral-
anchor draws + 16 random corpus windows, full-token CE, NO fact windows in
the wash steps) with F1-replay interleaved at rates r in {1/32, 1/8, 1/4}
(one replay batch per k neutral batches; 300 steps each). A replay step's
batch is e174 arm B's F1-replay batch VERBATIM: 16 windows drawn from e113's
jitter pool (60 install hosts x offsets {-8,-4,0,+4,+8}, ZEPHYRA spliced at
home, 7 name-char masked CE) + 16 anchors (8 paired neutral-bank + 8 random
corpus, full CE), union CE — the ONLY delta vs a wash step is the replay
channel. Optimizer AdamW (0.9,0.95) wd 0.1 CONSTANT lr 1e-3 clip 1.0 (the
wash's own lr; e174 arm B's lr), seed 10902 (the locked lineage), one fresh
CPU generator per cell (device-independent draws). Checkpoints {1, 2, 4,
10, 25, 50, 100, 200, 300} with no-RNG snapshots + light CPU evals (g-12,
g0, CE_R + the in-batch CE); FULL dial (e180's measure) at +50 and +300 per
cell — the two bar-relevant points (e180's rider convention).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/measure dial/
flat_cells/gate_vs/verify_ref are lab/e180_wash_rate.py VERBATIM (= e176n/
e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043); pick_dev/
migrate_to_cpu are lab/e184_seed_replicates.py VERBATIM via e180 (the
park-once policy + mid-run guard); the F1 jitter pool builder + G_JIT +
make_batch (the replay batch's union-CE arithmetic) are lab/
e174_dose_rehearsal.py VERBATIM (= e113's recipe); the wash batch + neutral
bank + junction accounting are e176n arm A VERBATIM via e180. Copied, not
imported, to own the device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n/e180's G_ROOT set). The r=0 stored
trajectory = e176n arm A's trace (embedded, verified vs file at plot time);
the r=1/2 endpoint = e174 arm B's stored dose-300 dial (embedded, verified
vs file). New checkpoints: runs/checkpoints/e179_r{0,1_32,1_8,1_4}.pt
(+ _s50 intermediates).

COMPUTE ENVELOPE (dispatch): GPU ALLOWED (idle 0%/67 C at dispatch) —
strict pre-training quick check per training (gpu_status()/gpu_ok(): util
<= 85 AND temp <= 80 C, plus the mem-headroom guard <= 85% of total;
double-poll 5 s apart), PARK-ONCE to CPU on any failure, MID-RUN
contention guard every 25 steps (mem/temp breach -> migrate net + optimizer
state to CPU and finish there; e184/e180's precedent; any device mixing
recorded per cell). cooldown(90 s) before and after EACH training
(dispatch: 60-120 s); caps 1800 s per training (dispatch); ALL readouts
CPU-side, sequential; NO concurrent GPU. Torch threads 8 (e152R/e143/
e184/e180 convention). Nets are the mandated 2.7M e131_consolidated line
(the dispatch's '<=1M family' note is an envelope statement — e174/e180's
precedent note; every gate reference and lineage number of this cell lives
on the 2.7M line, and the mandated root IS e131_consolidated_e113.pt).

Outputs: runs/e179/{metrics.json, rehearsal_law.png}; checkpoints
runs/checkpoints/e179_r*.pt (gitignored by lab policy). No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e179_rehearsal_law.py    (E179_SMOKE=1 shakedown)
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
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184/e180

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    gpu_ok, gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E179_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"                   # F1 — the tenant (the consolidated fact)
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176N_METRICS = E43.REPO / "runs" / "e176n" / "metrics.json"
E174_METRICS = E43.REPO / "runs" / "e174" / "metrics.json"

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

# ---- F1 replay pool (e113 arm-(a) jitter recipe, e174 VERBATIM) -----------------
JITTERS = (-8, -4, 0, 4, 8)       # e109/e113's registered jitter set

# ---- the rehearsal-rate cells (dispatch order 1/32 -> 1/8 -> 1/4) ---------------
# (k, tag): replay fires at every step with step % k == 0. k=0 -> never (wash
# replicate rider). Nominal r = 1/k; realized r co-reported per cell.
RATES: tuple[tuple[int, str], ...] = ((0, "r0_wash"), (32, "r1_32"),
                                      (8, "r1_8"), (4, "r1_4"))
NOMINAL_R = {"r0_wash": 0.0, "r1_32": 1.0 / 32, "r1_8": 1.0 / 8,
             "r1_4": 1.0 / 4}
LADDER = ("r1_32", "r1_8", "r1_4")          # the three registered rates

# ---- checkpoint grid + full-dial points ------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 10, 25, 50, 100, 200, 300) if not SMOKE \
    else (1, 2, 4, 36)
FULL_DIAL_AT: tuple[int, ...] = (50, 300) if not SMOKE else ()

# ---- fine-tune envelope (e176N arm A's wash + e174 arm B's replay, both lr 1e-3) -
FT_LR = 1e-3
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e174)
TRAIN_CAP_S = 1800.0              # dispatch: <=1800 s caps (per training, any device)
WASH_ANCH_BS, WASH_RAND_BS = 16, 16     # e176n arm A's wash batch: 16 + 16 full CE
RP_NAME_BS, RP_ANCH_BS = 16, 16         # e174 arm B's replay batch: 16 + (8 + 8)
COOLDOWN_S = 90.0                 # dispatch: 60-120 s; e152R/e184/e180's 90
MIDRUN_POLL_EVERY = 25            # mid-run GPU contention poll cadence (steps)

# ---- e170's neutral anchor bank (the wash's stream, VERBATIM) --------------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) ----------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176n/e180's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
}

# e176n arm A's stored trajectory (r=0, the NEUTRAL wash, lr 1e-3, seed 10902;
# runs/e176n/metrics.json trace_armA VERBATIM as embedded in e180, re-verified
# vs file at plot time) — the REGISTERED r=0 point: t* = 2 on {1,2,4,...},
# +300 g-12 0.00382.
E176N_ARMA = {
    "label": "e176n armA: neutral wash, lr 1e-3, seed 10902 (r=0 stored)",
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.6780440807342529,
             0.027077054604887962, 0.010940761305391788,
             0.022122304886579514, 0.014922752045904967,
             0.00204725144430995, 0.0038193254731595516],
    "g0": [0.7850371599197388, 0.4619811177253723,
           0.1147073358297348, 0.06042749062180519,
           0.16835635900497437, 0.05249874293804169,
           0.011475668287767338, 0.016238771378993988],
    "ce_r": [1.663516640663147, 2.2113420963287354,
             2.032074451446533, 1.8213403224945068,
             1.708834171595166, 1.6700899609382666,
             1.6468615531921387, 1.642844796180725],
}

# e174 arm B's stored 1:1 rehearsal endpoint (r=1/2; dose 300 = step 600) on the
# SAME g-12 ruler (e174's ladder_B base dials; runs/e174/metrics.json, verified
# vs file at plot time) — the dispatch's r=1/2 point. STREAM-MISMATCHED: arm B
# interleaves the F2-INSTALL batch (stronger interference than the neutral wash).
E174B_REF = {
    "label": "e174 armB: 1:1 F1-replay interleaved with F2-INSTALL (not wash), "
             "lr 1e-3, seed 10902 (r=1/2 stored, stream-mismatched)",
    "step": 600, "f2_dose": 300,
    "gm12": 0.7099656462669373, "g0": 0.9652615189552307,
    "held30_g0": 0.716054379940233, "ce_r": 1.6393135786056519,
    "ladder_gm12_by_step": {50: 0.7296594977378845, 100: 0.5977289080619812,
                            150: 0.4990350604057312, 300: 0.7532988786697388,
                            600: 0.7099656462669373},
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SHUT_BAR = 0.27                   # die bar (e158/e161/e176/e176n/e180)
MAINTAIN_BAR = 0.50               # maintain bar (e176n's NEUTRAL-SURVIVES)
WASH_T_STAR = 2                   # e176n arm A's stored clock (grid {1,2,4,...})
WASH_LEVEL_300 = 0.0038193254731595516   # e176n arm A's stored +300 g-12
PUMP_R2_BAR = 0.80                # PUMP-FIT's fit-quality bar (frozen here)
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "threshold": "THRESHOLD fires if: a sharp r* exists (below it the fact "
        "dies by +50; at or above it maintains >= 0.5) — a maintenance phase "
        "transition.",
    "graded": "GRADED fires if: any rehearsal slows the wash proportionally "
        "(no threshold; F1's +300 level rises smoothly with r).",
    "pump_fit": "PUMP-FIT fires if: both a wash rate w and a rehearsal gain g "
        "are extractable (the maintenance law's two constants).",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with the curve.",
    "operationalizations": "r = fraction of the 300 optimizer steps that are "
        "F1-replay batches (replay at every step % k == 0, replacing that "
        "step's wash batch; realized r co-reported); F1 ruler = g-12 "
        "(install-60 battery, offset -12); dies(r) = g-12 AT THE +50 "
        "CHECKPOINT <= 0.27 (e176n's NEUTRAL-DISSOLVES convention VERBATIM "
        "— the dispatch's 'dies by +50' as the +50 STATE; smoke-corrected "
        "pre-main: the first draft's 'any checkpoint under 0.27 in (0,50]' "
        "made the pre-replay eviction dip (steps 1..k-1 are wash steps) "
        "count as death for every rate, structurally unfireable); "
        "maintains(r) = g-12 >= 0.5 at BOTH horizons +50 AND +300; "
        "THRESHOLD = the ladder {1/32<1/8<1/4} splits as a suffix (r* "
        "maintains, every rate below dies at +50; all-maintain does NOT "
        "fire it); GRADED = not THRESHOLD AND +300 levels strictly "
        "increasing over the three rates AND level(1/32,+300) > the wash's "
        "stored 0.00382; t*(r) = first checkpoint with g-12 <= 0.27 "
        "(co-reported timing only); cycle stats (late-grid {100,200,300} "
        "mean/min g-12) co-reported for phase honesty, not bar-adjudicated; "
        "PUMP-FIT = pooled origin-OLS over consecutive-checkpoint segments "
        "of the three rate cells, d(g-12) = w*wash_steps + g*replay_steps, "
        "fires iff w<0 AND g>0 AND uncentered R^2 >= 0.80 (log-domain "
        "variant annotation only); order THRESHOLD -> GRADED -> TEXTURE for "
        "the shape; PUMP-FIT a separate co-verdict; final verdict "
        "'<SHAPE> + <PUMP-FIT|NO-PUMP>'; r=0 = e176n arm A's stored trace, "
        "r=1/2 = e174 arm B's stored dose-300 dial (stream-mismatched, "
        "conservative); the r=0 replicate is a rider (rig validation, "
        "G_REPRO), bars anchor on the stored trace.",
    "registration": "QUEUE row e179 (verbatim above) + the dispatch's "
                    "registered prediction VERBATIM; frozen in this docstring "
                    "before compute. Adjudicate against exactly this; no bar "
                    "shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "GPU ALLOWED by dispatch (idle 0%/67C at dispatch; e184/e180's policy "
    "verbatim): strict pre-training quick check per training (gpu_ok() gates: "
    "util <= 85, temp <= 80 C, mem <= 85% of total; double-poll 5 s apart), "
    "PARK-ONCE to CPU on failure; mid-run contention guard polls every "
    f"{MIDRUN_POLL_EVERY} steps and migrates net + optimizer state to CPU on "
    "mem/temp breach, finishing there (any mixing recorded per cell in "
    "device_events).",
    "The r=1/2 endpoint is STREAM-MISMATCHED by the dispatch's own design "
    "(e174 arm B interleaves the F2-INSTALL batch — a STRONGER interference "
    "than the neutral wash — alongside the same F1-replay batches): the 1/2 "
    "point is a conservative floor for the curve's end, flagged at every use; "
    "the threshold/graded bars adjudicate on the three NEW rates only.",
    "A FOURTH training beyond the dispatch's three — the r=0 wash REPLICATE "
    "(QUEUE's '4-5 trainings' allowance) — is a RIDER: it validates the rig "
    "(G_REPRO vs e176n arm A's stored cells; its RNG draw sequence is "
    "arm-A-identical at seed 10902) and anchors the curve same-run. The bars "
    "anchor on the STORED e176n trace as registered either way; a G_REPRO "
    "fail on a device-mixed replicate is recorded and flagged, not raised "
    "(pure-CPU fail still raises).",
    "Full dials at +50 and +300 only (light in-run evals elsewhere, e180's "
    "rider convention): the maintenance question is answered by the g-12 "
    "trajectory; the full dial is the anatomy co-report at the two "
    "bar-relevant points.",
    "SMOKE CORRECTION (pre-main; nothing adjudicated in smoke): the first "
    "frozen draft read dies(r) as 'any checkpoint in (0,50] under 0.27' and "
    "maintains(r) as 'never under 0.27 through +300' — both structurally "
    "unfireable under re-teach dynamics (steps 1..k-1 are wash steps for "
    "every k >= 4, so the two-step eviction dip at +2 counts as death for "
    "EVERY rate, 1:1 included by extension). Re-frozen BEFORE the main "
    "compute to e176n's state-based ancestor convention (dies = the +50 "
    "CELL <= 0.27, exactly e176n's NEUTRAL-DISSOLVES; maintains = >= 0.5 "
    "at BOTH registered horizons +50 and +300). This correction makes the "
    "registered THRESHOLD clause fireable; it does not favor any outcome.",
    "The fixed checkpoint grid is NOT aligned to replay events (no per-rate "
    "phase shopping): an oscillating rate reads differently at +300 depending "
    "on where its last replay sits relative to 300 (k=32: 12 wash steps "
    "after the last replay; k=8: 4; k=4: step 300 IS a replay step) — the "
    "per-checkpoint trajectory AND the co-reported late-grid cycle stats "
    "(mean/min g-12 over {100,200,300}) carry the sustained level; flagged "
    "in the honesty reflex.",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's '<=1M "
    "family' note is an envelope statement; e174/e180 precedent — every gate "
    "reference lives on this line, and the mandated root IS "
    "e131_consolidated_e113.pt).",
    "GPU float nondeterminism: the stored r=0 reference ran pure CPU; if the "
    "replicate or rate cells run GPU (or park mid-run), devices mix ACROSS "
    "curve points — recorded per cell. The bars live at order-of-magnitude "
    "level separations (0.27/0.5 vs floors ~0.004), which float noise cannot "
    "move; any read within 0.05 of either bar is FLAGGED, not smoothed.",
    "Eval thread count is 8 (e152R/e143/e184/e180 convention) vs e151's "
    "stored cells — CPU reduction order can drift low-order bits; the G_ROOT "
    "gate reports both the 5e-6 bit flag and the 0.05 fallback tolerance.",
    "The replay steps draw RNG shapes e174-arm-B-style (ix(16,)+aj(8,)+"
    "rj(8,)) while wash steps draw e176n-arm-A-style (aj(16,)+rj(16,)): the rate "
    "cells are their own streams at seed 10902 by design (only the r=0 "
    "replicate is arm-A-identical); recorded so no one reads cross-cell RNG "
    "identity into the design.",
    "Single seed (10902), one trajectory per rate cell, one root lineage, "
    "n=1 per cell — point estimates until replicated (e184's seed lottery "
    "lives in the tail; the maintenance threshold is unreplicated).",
    "Smoke mode trims: 36-step trainings, checkpoints {1,2,4,36}, lean "
    "measures, no cooldowns, no full dials, nothing adjudicated.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM via e180 (= e152r's pick_dev
# adapted to the lab's gpu_ok(); plus the dispatch's mid-run migration guard).

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
# PROVENANCE: lab/e180_wash_rate.py VERBATIM (see the module docstring).
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
    while len(out_y) < n and tries < 500 * n:
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
    e176n/e178/e184/e180 copy): p(true name char) at positions addr_row..addr_row+6."""
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


# ------------------------------------------------------------------ batches

def wash_loss(net, anchor, aj, train_ids, rj, dev):
    """e176n arm A's WASH batch VERBATIM arithmetic (device-parameterized as
    in e180): 16 neutral-bank draws + 16 random corpus windows, full-token CE."""
    anc = anchor[aj]
    rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
    x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
    y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0).to(dev)
    logits, _ = net(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def replay_loss(net, jit_x, jit_mask, ix, anchor, aj, train_ids, rj, dev):
    """e174 arm B's F1-REPLAY batch VERBATIM (make_batch): 16 F1 jitter windows
    (7 name-char masked CE) + 16 anchors (8 paired neutral + 8 random, full
    CE), union CE — the ONLY delta vs a wash step is the replay channel."""
    nw = jit_x[ix]
    anc = torch.cat([anchor[aj],
                     torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0)
    x = torch.cat([nw[:, :-1], anc[:, :-1]], 0).to(dev)
    y = torch.cat([nw[:, 1:], anc[:, 1:]], 0).to(dev)
    m = torch.zeros(RP_NAME_BS + RP_ANCH_BS, x.shape[1], dtype=torch.bool,
                    device=dev)
    m[:RP_NAME_BS] = jit_mask[ix].to(dev)
    logits, _ = net(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                          reduction="none").view(x.shape[0], x.shape[1])
    nm = nll[:RP_NAME_BS][m[:RP_NAME_BS]]
    cm = nll[RP_NAME_BS:]
    return (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())


# ------------------------------------------------------------------ fine-tune

def finetune_rate(tag: str, net0: TinyGPT, k: int,
                  jit_x: torch.Tensor, jit_mask: torch.Tensor,
                  anchor: torch.Tensor, train_ids: torch.Tensor, itos,
                  r_eval_xy, gm12_ids, g0_ids, zid: int, seed: int,
                  ckpt_steps: tuple[int, ...]):
    """THE REHEARSAL-RATE CELL. `ckpt_steps[-1]` optimizer steps; every step
    with (k >= 1 and step % k == 0) is an F1-REPLAY batch (e174 arm B's
    batch VERBATIM; draws ix(16,)+aj(8,)+rj(8,)); every other step is
    e176n arm A's WASH batch VERBATIM (draws aj(16,)+rj(16,)). k=0 -> never
    replay (the pure-wash replicate; its RNG stream is arm-A-identical at
    seed 10902). AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0. No-RNG
    snapshots + light CPU evals (g-12, g0, CE_R + the in-batch CE) at the
    checkpoint steps. Mid-run GPU contention guard every MIDRUN_POLL_EVERY
    steps (e184/e180 verbatim)."""
    dev = pick_dev(tag)
    cap = TRAIN_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    n_jit = jit_x.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    n_replay = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, n_steps + 1):
        replay = (k >= 1 and step % k == 0)
        if replay:
            ix = torch.randint(n_jit, (RP_NAME_BS,), generator=gen)
            aj = torch.randint(n_anc, (RP_ANCH_BS // 2,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1,
                               (RP_ANCH_BS // 2,), generator=gen)
            # name-free VERIFY on the random half (no-op by construction)
            for s in rj:
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            loss = replay_loss(net, jit_x, jit_mask, ix, anchor, aj,
                               train_ids, rj, dev)
            n_replay += 1
        else:
            aj = torch.randint(n_anc, (WASH_ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (WASH_RAND_BS,),
                               generator=gen)
            for s in rj:
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            loss = wash_loss(net, anchor, aj, train_ids, rj, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set or step % 50 == 0:
            log(f"  [{tag}] s{step:4d} ({'REPLAY' if replay else 'wash'}) "
                f"in-batch CE {float(loss.item()):.4f} "
                f"({time.time() - t_start:.0f}s)")
        if step in ckpt_set:
            sd_cpu = {kk: v.detach().cpu().clone()
                      for kk, v in net.state_dict().items()}
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
                         "replay_steps_so_far": n_replay,
                         "was_replay_step": bool(replay),
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"(replays so far {n_replay})")
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
            "k": k, "n_replay": n_replay,
            "nominal_r": (1.0 / k) if k >= 1 else 0.0,
            "realized_r": n_replay / max(step, 1),
            "zeph_violations": zeph_checks,
            "initial_device": "cuda" if not GPU_PARKED else "cpu",
            "final_device": str(dev), "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e179", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, src_name: str,
               layout: str = "auto") -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (no silent divergence; e176n/e180's verify_ref convention)."""
    src = {"source": f"embedded verbatim copy ({src_name})",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        if "trace_armA" in mm:                  # runs/e176n layout
            rows = mm["trace_armA"]
            ts = {"freeze_steps": [r["freeze_steps"] for r in rows],
                  "gm12": [r["gm12"] for r in rows],
                  "g0": [r["g0"] for r in rows],
                  "ce_r": [r["ce_r"] for r in rows]}
            ref_kind = "trace_armA"
        else:
            src["verified_vs_embedded"] = False
            src["note"] = "unrecognized metrics layout"
            return src
        diffs = [abs(a - b) for kk in ("gm12", "g0", "ce_r")
                 for a, b in zip(ts[kk], embedded[kk])]
        steps_ok = list(ts["freeze_steps"]) == embedded["freeze_steps"]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = (f"{src_name} ({ref_kind}; embedded copy "
                             f"verified, max|diff| {max(diffs):.1e})")
    return src


def verify_e174b(path: Path) -> dict:
    """Verify the embedded e174 arm-B endpoint against runs/e174/metrics.json
    (the r=1/2 stored point: dose-300 dial + the per-dose g-12 overlay)."""
    src = {"source": "embedded verbatim copy (runs/e174/metrics.json ladder_B)",
           "file_present": path.exists(), "verified_vs_embedded": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        lb = mm["ladder_B"]["300"]
        b = lb["base"]
        cells = {"gm12": b["-12"]["mean_pz"], "g0": b["0"]["mean_pz"],
                 "held30_g0": lb["base_held30_g0"]["mean_pz"],
                 "ce_r": lb["ce_r"]}
        diffs = {k: abs(cells[k] - E174B_REF[k])
                 for k in ("gm12", "g0", "held30_g0", "ce_r")}
        steps = {"25": 50, "50": 100, "75": 150, "150": 300, "300": 600}
        ladder = {steps[d]: mm["ladder_B"][d]["base"]["-12"]["mean_pz"]
                  for d in steps}
        ldiff = max(abs(ladder[s] - E174B_REF["ladder_gm12_by_step"][s])
                    for s in steps.values())
        src["max_abs_diff"] = max(list(diffs.values()) + [ldiff])
        src["verified_vs_embedded"] = bool(src["max_abs_diff"] < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = (f"runs/e174/metrics.json ladder_B (embedded copy "
                             f"verified, max|diff| {src['max_abs_diff']:.1e})")
    return src


# ------------------------------------------------------------------ pump fit

def pump_fit(cells: dict, ladder: tuple) -> dict:
    """PUMP-FIT (registered): pooled regression-through-origin over ALL
    consecutive-checkpoint segments of the three rate cells:
    d(g-12) = w*(wash steps in segment) + g*(replay steps in segment).
    Fires iff w < 0 AND g > 0 AND uncentered R^2 >= PUMP_R2_BAR. The
    log-domain variant is annotation only."""
    rows = []
    for tag in ladder:
        k = cells[tag]["k"]
        for a, b in zip(cells[tag]["traj"][:-1], cells[tag]["traj"][1:]):
            s0, s1 = a["step"], b["step"]
            nr = sum(1 for s in range(s0 + 1, s1 + 1)
                     if k >= 1 and s % k == 0)
            nw = (s1 - s0) - nr
            rows.append((nw, nr, b["g_m12_mean_pz"] - a["g_m12_mean_pz"],
                         tag, s0, s1))
    A = np.array([[r[0], r[1]] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=float)

    def _fit(A, y):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_y = float((y ** 2).sum())
        r2 = 1.0 - ss_res / ss_y if ss_y > 0 else float("nan")
        return {"w": float(coef[0]), "g": float(coef[1]),
                "r2_uncentered": float(r2), "n_segments": int(len(y)),
                "per_step_lr_displacement_note":
                    "w and g are per-STEP effects on the g-12 level; "
                    "w's magnitude x the wash clock prices the basin ride "
                    "(T119's lr x t* ~ 2-6e-3 lives on the same axis)"}

    fit = _fit(A, y)
    fit["fires"] = bool(fit["w"] < 0 and fit["g"] > 0
                        and fit["r2_uncentered"] >= PUMP_R2_BAR)
    # log-domain variant (annotation only), computed per-cell so no segment
    # spans a cell boundary; the predictor rows rebuild in the SAME order.
    dlog_rows, Alog = [], []
    for tag in ladder:
        k = cells[tag]["k"]
        tr = cells[tag]["traj"]
        lv = [np.log10(t["g_m12_mean_pz"] + 0.01) for t in tr]
        dlog_rows += [lv[i + 1] - lv[i] for i in range(len(lv) - 1)]
        for a, b in zip(tr[:-1], tr[1:]):
            s0, s1 = a["step"], b["step"]
            nr = sum(1 for s in range(s0 + 1, s1 + 1)
                     if k >= 1 and s % k == 0)
            Alog.append([(s1 - s0) - nr, nr])
    Alog = np.array(Alog, dtype=float)
    assert Alog.shape == A.shape, "log-fit segment mismatch"
    fit_log = _fit(Alog, np.array(dlog_rows, dtype=float))
    fit_log["fires"] = None           # annotation only (registered bar is linear)
    fit_log["note"] = ("log-domain variant (d log10(g-12 + 0.01)) — "
                       "ANNOTATION ONLY; the registered PUMP-FIT bar lives on "
                       "the linear form")
    return {"linear": fit, "log_domain": fit_log, "segments": [
        {"tag": r[3], "from": r[4], "to": r[5], "wash_steps": r[0],
         "replay_steps": r[1], "d_gm12": r[2]} for r in rows]}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e179_smoke" if SMOKE else "e179")
    log(f"E179 THE REHEARSAL-FREQUENCY LAW (smoke={SMOKE}) -> {rd}")
    log(f"compute: GPU allowed (park-once policy), cooldown {COOLDOWN_S:.0f}s "
        f"around each training, per-training cap {TRAIN_CAP_S:.0f}s")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e180)
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

    # ---------------- F1 replay pool: e113's jitter recipe (e174 VERBATIM)
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"F1 jitter window len {len(w)} != {BLOCK} "
                                   f"at offset {j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = m
    jit_pool_x = torch.cat([jit_x[j] for j in JITTERS])       # (300, 256)
    jit_pool_mask = torch.cat([jit_mask[j] for j in JITTERS])
    G_JIT = {
        "jitters": list(JITTERS), "pool_shape": list(jit_pool_x.shape),
        "name_in_place_all": bool(all(
            torch.equal(w[PRE + j: PRE + j + len(NAME)], name_ids)
            for j in JITTERS for w in jit_x[j])),
        "mask_targets_per_window": int(jit_pool_mask[0].sum()),
        "masks_vary_with_jitter": bool(
            len({int(jit_mask[j][0].nonzero()[0]) for j in JITTERS}) ==
            len(JITTERS)),
        "stays_in_f1_band": bool(all(
            121 <= PRE - 1 + j <= 137 for j in JITTERS)),   # e113's onset-row
                                                            # span 121..137
        "note": ("e113's jitter set reads F1 at onset rows 129+j in "
                 "[121,137] (the grown-address band); the +8 span reaches "
                 "row 143 — e113/e174 VERBATIM arithmetic"),
    }
    G_JIT["pass"] = bool(G_JIT["name_in_place_all"]
                         and G_JIT["mask_targets_per_window"] == len(NAME)
                         and G_JIT["masks_vary_with_jitter"]
                         and G_JIT["stays_in_f1_band"]
                         and jit_pool_x.shape[0] == 300)
    assert G_JIT["pass"], f"F1 jitter pool gate FAILED: {G_JIT}"
    log(f"F1 replay pool (e113 recipe, e174 verbatim): "
        f"{tuple(jit_pool_x.shape)} (offsets {list(JITTERS)}), name masks at "
        f"y-cols {[int(jit_mask[j][0].nonzero()[0]) for j in JITTERS]}, all "
        f"inside the F1 band 121..137")

    # ---------------- e170's NEUTRAL anchor bank (the wash's stream VERBATIM)
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
                             "e180's bank)"),
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
        "rng_note": ("wash steps draw aj(16,)+rj(16,) — e176n arm A's exact "
                     "shapes/moduli at seed 10902 (the r=0 replicate is "
                     "arm-A-identical); replay steps draw e174 arm B's "
                     "ix(16,)+aj(8,)+rj(8,)"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the random corpus windows are e161/e176 VERBATIM and "
                     "unfiltered — identical background, not part of the "
                     "delta"),
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
        """e180's measure() VERBATIM (= e176n's, already minus the 183-span
        census — e178's recorded deviation): base 3-geos + held30 + CE_R +
        site read + old-band census (row-0 sink / A129 brake) + deletion
        table; lean=True keeps only a quick A(129)."""
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
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_JIT, G_ANCHOR, G_ROOT all PASS")

    # reference provenance (embedded copies verified vs the stored files)
    src_arma = verify_ref(E176N_ARMA, E176N_METRICS,
                          "runs/e176n/metrics.json trace_armA")
    src_e174b = verify_e174b(E174_METRICS)

    # =====================================================================
    # THE FOUR TRAININGS (r=0 replicate rider first — it validates the rig
    # against e176n arm A BEFORE the rate cells burn compute; then the
    # dispatch's order 1/32 -> 1/8 -> 1/4)
    # =====================================================================
    cells: dict = {}
    for k, tag in RATES:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        sched = (f"replay every {k}-th step (nominal r = 1/{k})"
                 if k >= 1 else "NO replay (pure wash replicate, rider)")
        log(f"CELL {tag} — {sched}: {CK_MAIN[-1]} steps, e176n arm A's wash "
            f"stream (batch {WASH_ANCH_BS} neutral + {WASH_RAND_BS} random, "
            f"full-token CE) with e174 arm B's replay batches, lr {FT_LR}, "
            f"seed {FREEZE_SEED}, checkpoints +{list(CK_MAIN)}")
        arm = finetune_rate(tag, net0, k, jit_pool_x, jit_pool_mask,
                            anchor_neutral, train_ids, itos, r_eval_xy,
                            gm12_ids, g0_ids, zid, FREEZE_SEED, CK_MAIN)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        g_drawfree = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert g_drawfree["pass"], f"{tag}: name token leaked into a window"

        smax = max(arm["sds"])
        save_ckpt(f"e179_{tag}", arm["sds"][smax],
                  {"desc": f"e131_consolidated_e113 + {smax}-step neutral-wash "
                           f"stream with F1-replay at nominal r="
                           f"{arm['nominal_r']:.6g} ({arm['n_replay']} replay "
                           f"batches of {smax}), lr {FT_LR}, seed {FREEZE_SEED}"
                           + (" — the r=0 wash REPLICATE (rider)"
                              if k == 0 else ""),
                   "steps": int(smax), "seed": FREEZE_SEED, "lr": FT_LR,
                   "k": int(k), "nominal_r": arm["nominal_r"],
                   "n_replay": int(arm["n_replay"]),
                   "realized_r": arm["realized_r"],
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "base": f"runs/checkpoints/{ROOT_CK}"})
        if 50 in arm["sds"] and 50 != smax:
            save_ckpt(f"e179_{tag}_s50", arm["sds"][50],
                      {"desc": f"e131_consolidated_e113 + 50-step neutral-wash "
                               f"r={arm['nominal_r']:.6g} cell (the die-by-50 "
                               f"bar point), seed {FREEZE_SEED}",
                       "steps": 50, "seed": FREEZE_SEED, "lr": FT_LR,
                       "k": int(k), "nominal_r": arm["nominal_r"],
                       "n_replay_at_50": sum(
                           1 for s in range(1, 51) if k >= 1 and s % k == 0),
                       "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                       "base": f"runs/checkpoints/{ROOT_CK}"})

        # FULL dials at the bar-relevant points (+50, +300)
        batteries: dict = {}
        for s in FULL_DIAL_AT:
            if s in arm["sds"]:
                log(f"CELL {tag} +{s} full dial")
                batteries[str(s)] = measure(arm["sds"][s], f"{tag}_s{s}",
                                            lean=False)

        cells[tag] = {
            "desc": ("e176n arm A's NEUTRAL wash stream with e174 arm B's "
                     f"F1-replay batches interleaved at nominal r="
                     f"{arm['nominal_r']:.6g} (replay at every step % "
                     f"{k} == 0)"
                     if k >= 1 else
                     "the r=0 wash REPLICATE (rider): e176n arm A's NEUTRAL "
                     "protocol VERBATIM at seed 10902 (arm-A-identical RNG "
                     "stream)"),
            "k": int(k), "ckpt_steps": list(CK_MAIN),
            "steps_ran": arm["steps_ran"], "seed": arm["seed"],
            "lr": FT_LR, "n_replay": arm["n_replay"],
            "nominal_r": arm["nominal_r"], "realized_r": arm["realized_r"],
            "traj": arm["traj"],
            "zeph_violations": arm["zeph_violations"],
            "missing_checkpoints": sorted(set(CK_MAIN) - set(arm["sds"])),
            "initial_device": arm["initial_device"],
            "final_device": arm["final_device"],
            "time_cap_s": arm["time_cap_s"],
            "batteries_full": batteries,
            "cells_full": {s: flat_cells(b) for s, b in batteries.items()},
        }
        del arm

    # G_REPRO: the r=0 replicate vs e176n arm A's stored trace
    stored = {s: g for s, g in zip(E176N_ARMA["freeze_steps"],
                                   E176N_ARMA["gm12"]) if s > 0}
    repro_cells = {}
    for t in cells["r0_wash"]["traj"]:
        if t["step"] in stored:
            repro_cells[t["step"]] = {
                "this_run": t["g_m12_mean_pz"], "e176n_armA": stored[t["step"]],
                "diff": t["g_m12_mean_pz"] - stored[t["step"]]}
    maxdiff = max((abs(c["diff"]) for c in repro_cells.values()), default=0.0)
    pure_cpu = (cells["r0_wash"]["initial_device"] == "cpu"
                and cells["r0_wash"]["final_device"] == "cpu")
    G_REPRO = {"cells": repro_cells, "n_cells": len(repro_cells),
               "max_abs_diff": maxdiff, "tol": G_FALLBACK_TOL,
               "bit_reproducible": bool(0 < len(repro_cells)
                                        and maxdiff < G_BIT_TOL),
               "replicate_pure_cpu": bool(pure_cpu),
               "pass": bool(len(repro_cells) > 0 and maxdiff < G_FALLBACK_TOL),
               "note": ("the rider's wash replicate vs the stored r=0 trace "
                        "(common steps); a device-mixed replicate can drift "
                        "beyond 0.05 through chaotic amplification — recorded "
                        "and flagged, not raised (the bars anchor on the "
                        "stored trace either way); pure-CPU fail raises")}
    log(f"G_REPRO r0-replicate vs e176n armA ({len(repro_cells)} common "
        f"steps): max|diff| {maxdiff:.2e} (tol {G_FALLBACK_TOL}): "
        f"{'PASS' if G_REPRO['pass'] else 'FAIL'}"
        f"{' [bit-exact]' if G_REPRO['bit_reproducible'] else ''} "
        f"(device {cells['r0_wash']['initial_device']}->"
        f"{cells['r0_wash']['final_device']})")
    if G_REPRO["pass"] is False and pure_cpu and not SMOKE:
        raise RuntimeError(f"the r=0 replicate failed to reproduce e176n "
                           f"arm A's wash trajectory: {maxdiff}")

    # =====================================================================
    # ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    def t_star(traj):
        return next((t["step"] for t in traj
                     if t["g_m12_mean_pz"] <= SHUT_BAR), None)

    def level_at(traj, step):
        return next((t["g_m12_mean_pz"] for t in traj if t["step"] == step),
                    traj[-1]["g_m12_mean_pz"])

    def g0_at(traj, step):
        return next((t["g0_mean_pz"] for t in traj if t["step"] == step),
                    traj[-1]["g0_mean_pz"])

    lvl50 = {tag: level_at(cells[tag]["traj"], 50) for tag in LADDER}
    lvl300 = {tag: level_at(cells[tag]["traj"], 300) for tag in LADDER}
    ts = {tag: t_star(cells[tag]["traj"]) for tag in LADDER}
    dies = {tag: bool(lvl50[tag] <= SHUT_BAR) for tag in LADDER}
    never_dead = {tag: all(t["g_m12_mean_pz"] > SHUT_BAR
                           for t in cells[tag]["traj"]) for tag in LADDER}
    maintains = {tag: bool(lvl50[tag] >= MAINTAIN_BAR
                           and lvl300[tag] >= MAINTAIN_BAR)
                 for tag in LADDER}
    cycle = {tag: {"late_grid": [100, 200, 300],
                   "mean_gm12_late": float(np.mean(
                       [level_at(cells[tag]["traj"], s)
                        for s in (100, 200, 300)])),
                   "min_gm12_late": float(min(
                       level_at(cells[tag]["traj"], s)
                       for s in (100, 200, 300)))}
             for tag in LADDER}

    threshold_r = None
    for i in range(1, len(LADDER)):
        if maintains[LADDER[i]] and all(dies[LADDER[j]] for j in range(i)):
            threshold_r = LADDER[i]
            break
    all_maintain = all(maintains[t] for t in LADDER)
    levels_ladder = [lvl300[t] for t in LADDER]
    strictly_rising = all(levels_ladder[i] < levels_ladder[i + 1]
                          for i in range(len(levels_ladder) - 1))
    smallest_moves = bool(levels_ladder[0] > WASH_LEVEL_300)
    graded = bool(threshold_r is None and not all_maintain
                  and strictly_rising and smallest_moves)

    if threshold_r is not None:
        shape = "THRESHOLD"
        idx = LADDER.index(threshold_r)
        prev = LADDER[idx - 1] if idx > 0 else None
        shape_clause = (
            f"a maintenance phase transition: r* = {NOMINAL_R[threshold_r]:.6g} "
            f"maintains (g-12 {lvl50[threshold_r]:.4f} at +50 AND "
            f"{lvl300[threshold_r]:.4f} at +300, both >= {MAINTAIN_BAR}) while "
            f"every measured rate below it is DEAD at +50 ("
            + ", ".join(f"r={NOMINAL_R[t]:.6g}: +50 {lvl50[t]:.4f}"
                        for t in LADDER[:idx]) +
            f" <= {SHUT_BAR}); the bracket (r={NOMINAL_R[prev]:.6g}, "
            f"{NOMINAL_R[threshold_r]:.6g}] — resolution bounded by the "
            f"registered ladder).")
    elif graded:
        shape = "GRADED"
        shape_clause = (
            "no threshold: the ladder does not split as a suffix, and the "
            "+300 level rises smoothly with r ("
            + ", ".join(f"r={NOMINAL_R[t]:.6g}: {lvl300[t]:.4f}"
                        for t in LADDER) +
            f"), all above the wash's stored +300 ({WASH_LEVEL_300:.4f}) — "
            "any rehearsal slows the wash proportionally.")
    elif all_maintain:
        shape = "TEXTURE"
        shape_clause = (
            f"every measured rate down to r=1/32 maintains (g-12 >= "
            f"{MAINTAIN_BAR} at both horizons: "
            + ", ".join(f"{NOMINAL_R[t]:.6g}: +50 {lvl50[t]:.4f}, +300 "
                        f"{lvl300[t]:.4f}" for t in LADDER) +
            ") — no dying side exists inside the registered ladder, so no "
            "threshold is observable here; r* <= 1/32 (the bracket's "
            "resolution is the ladder's floor).")
    else:
        shape = "TEXTURE"
        shape_clause = (
            "neither clause fired (ladder (+50 state, +300 state, "
            "maintains): "
            + ", ".join(f"r={NOMINAL_R[t]:.6g}: ({lvl50[t]:.4f}, "
                        f"{lvl300[t]:.4f}, {maintains[t]})"
                        for t in LADDER)
            + f"; late-grid cycle mean/min "
            + ", ".join(f"1/{cells[t]['k']}: {cycle[t]['mean_gm12_late']:.3f}"
                        f"/{cycle[t]['min_gm12_late']:.3f}" for t in LADDER)
            + ") — texture with the curve.")

    pump = pump_fit(cells, LADDER)
    pump_verdict = "PUMP-FIT" if pump["linear"]["fires"] else "NO-PUMP"
    if pump["linear"]["fires"]:
        pump_clause = (f"the maintenance law's two constants extractable: "
                       f"w = {pump['linear']['w']:+.6f} per wash step, "
                       f"g = {pump['linear']['g']:+.6f} per replay step "
                       f"(pooled origin-OLS, {pump['linear']['n_segments']} "
                       f"segments, uncentered R^2 = "
                       f"{pump['linear']['r2_uncentered']:.3f} >= "
                       f"{PUMP_R2_BAR}).")
    else:
        pump_clause = (f"the two-constant pump did NOT fit (w = "
                       f"{pump['linear']['w']:+.6f}, g = "
                       f"{pump['linear']['g']:+.6f}, R^2 = "
                       f"{pump['linear']['r2_uncentered']:.3f} < "
                       f"{PUMP_R2_BAR} or wrong signs) — the linear pump is "
                       f"not the curve's form; log-domain variant "
                       f"w={pump['log_domain']['w']:+.6f}, g="
                       f"{pump['log_domain']['g']:+.6f}, R^2="
                       f"{pump['log_domain']['r2_uncentered']:.3f} "
                       f"(annotation).")

    verdict = f"{shape} + {pump_verdict}"

    # ---- THE R-CURVE (five points: stored wash, three new, stored 1:1)
    r_curve = {
        "ruler": "g-12 (install-60 battery, offset -12; e176/e176n/e180's)",
        "bars": {"die": SHUT_BAR, "maintain": MAINTAIN_BAR},
        "points": {
            "0.0_wash_stored_e176n_armA": {
                "r": 0.0, "provenance": src_arma["source"],
                "t_star": WASH_T_STAR, "gm12_plus300": WASH_LEVEL_300,
                "note": "the REGISTERED r=0 anchor (stored trace)"},
            "0.0_wash_replicate_this_run": {
                "r": 0.0, "provenance": "this run (rider)",
                "t_star": t_star(cells["r0_wash"]["traj"]),
                "gm12_plus50": level_at(cells["r0_wash"]["traj"], 50),
                "gm12_plus300": level_at(cells["r0_wash"]["traj"], 300),
                "G_REPRO_max_abs_diff": maxdiff,
                "note": "rider validation; bars anchor on the stored trace"},
            "1_32_this_run": {
                "r": cells["r1_32"]["nominal_r"],
                "realized_r": cells["r1_32"]["realized_r"],
                "n_replay": cells["r1_32"]["n_replay"],
                "t_star": ts["r1_32"], "dead_at_50": dies["r1_32"],
                "maintains": maintains["r1_32"],
                "gm12_plus50": level_at(cells["r1_32"]["traj"], 50),
                "gm12_plus300": lvl300["r1_32"],
                "g0_plus300": g0_at(cells["r1_32"]["traj"], 300),
                "provenance": "this run"},
            "1_8_this_run": {
                "r": cells["r1_8"]["nominal_r"],
                "realized_r": cells["r1_8"]["realized_r"],
                "n_replay": cells["r1_8"]["n_replay"],
                "t_star": ts["r1_8"], "dead_at_50": dies["r1_8"],
                "maintains": maintains["r1_8"],
                "gm12_plus50": level_at(cells["r1_8"]["traj"], 50),
                "gm12_plus300": lvl300["r1_8"],
                "g0_plus300": g0_at(cells["r1_8"]["traj"], 300),
                "provenance": "this run"},
            "1_4_this_run": {
                "r": cells["r1_4"]["nominal_r"],
                "realized_r": cells["r1_4"]["realized_r"],
                "n_replay": cells["r1_4"]["n_replay"],
                "t_star": ts["r1_4"], "dead_at_50": dies["r1_4"],
                "maintains": maintains["r1_4"],
                "gm12_plus50": level_at(cells["r1_4"]["traj"], 50),
                "gm12_plus300": lvl300["r1_4"],
                "g0_plus300": g0_at(cells["r1_4"]["traj"], 300),
                "provenance": "this run"},
            "0.5_rehearsal1to1_stored_e174_armB": {
                "r": 0.5, "provenance": src_e174b["source"],
                "gm12_plus300": E174B_REF["gm12"],
                "g0_plus300": E174B_REF["g0"],
                "held30_g0_plus300": E174B_REF["held30_g0"],
                "ce_r": E174B_REF["ce_r"],
                "note": ("STREAM-MISMATCHED (dispatch's design): arm B's "
                         "interference is the F2-INSTALL batch, STRONGER "
                         "than the neutral wash — a conservative floor for "
                         "the curve's end")},
        },
        "threshold": {"r_star": (NOMINAL_R[threshold_r]
                                 if threshold_r is not None else None),
                      "tag": threshold_r,
                      "bracket": ((f"({NOMINAL_R[LADDER[LADDER.index(threshold_r) - 1]]:.6g}, "
                                   f"{NOMINAL_R[threshold_r]:.6g}]")
                                  if threshold_r is not None else None)},
        "adjudication_order": "THRESHOLD -> GRADED -> TEXTURE (shape); "
                              "PUMP-FIT separate",
    }

    adjudication = {
        "bars": {"die": SHUT_BAR, "maintain": MAINTAIN_BAR,
                 "pump_r2": PUMP_R2_BAR},
        "per_rate": {t: {"nominal_r": NOMINAL_R[t],
                         "realized_r": cells[t]["realized_r"],
                         "n_replay": cells[t]["n_replay"],
                         "t_star_timing_only": ts[t],
                         "dead_at_50": dies[t],
                         "never_under_die_bar": never_dead[t],
                         "gm12_plus50": lvl50[t],
                         "gm12_plus300": lvl300[t],
                         "cycle_stats_late_grid": cycle[t],
                         "maintains": maintains[t]} for t in LADDER},
        "all_rates_maintain": bool(all_maintain),
        "levels_strictly_rising": bool(strictly_rising),
        "smallest_rehearsal_moves_curve": smallest_moves,
        "THRESHOLD_fires": bool(threshold_r is not None),
        "GRADED_fires": graded,
        "PUMP_FIT_fires": pump["linear"]["fires"],
        "shape": shape, "shape_clause": shape_clause,
        "pump_verdict": pump_verdict, "pump_clause": pump_clause,
        "verdict": verdict,
        "order": "THRESHOLD -> GRADED -> TEXTURE (shape); PUMP-FIT a "
                 "separate co-verdict; no bar shopping",
    }

    log("=" * 78)
    log(f"E179 VERDICT: {verdict}")
    log(f"  shape: {shape_clause}")
    log(f"  pump: {pump_clause}")
    log("  r-curve (+50 g-12 | +300 g-12 | late-grid cycle mean): wash "
        f"{level_at(cells['r0_wash']['traj'], 50):.4f} | {WASH_LEVEL_300:.4f} "
        f"(stored; replicate {level_at(cells['r0_wash']['traj'], 300):.4f}) | "
        + " | ".join(f"1/{cells[t]['k']} {lvl50[t]:.4f} | {lvl300[t]:.4f} | "
                     f"{cycle[t]['mean_gm12_late']:.4f}" for t in LADDER)
        + f" || 1/2 {E174B_REF['gm12']:.4f} (e174 armB, stream-mismatched)")
    log("=" * 78)

    # =====================================================================
    # PLOT (the r-curve deliverable + trajectories + pump fit + verdict)
    # =====================================================================
    png = make_plot(rd, cells, r_curve, adjudication, pump, root_cells,
                    E174B_REF, E176N_ARMA)
    log(f"[plot] saved {png}")

    # =====================================================================
    # METRICS
    # =====================================================================
    metrics = {
        "experiment": "e179_rehearsal_law",
        "date": common.now_iso(),
        "registration": ("QUEUE row e179 (verbatim in the module docstring) + "
                         "the dispatch's registered prediction VERBATIM; "
                         "frozen before compute; adjudicated against exactly "
                         "that — no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("how little rehearsal suffices against the neutral wash? "
                     "— the maintenance threshold r* (a phase transition) vs "
                     "a graded response vs a two-constant pump law (w, g); "
                     "the quantitative completion of the capacity story "
                     "(capacity = the maintenance budget; T107's surviving "
                     "direction, beside T119's rate law)"),
        "root": f"runs/checkpoints/{ROOT_CK} "
                f"(gated vs e151 before-cells, max|diff| "
                f"{G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cells": cells,
        "protocol": {
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
            "geometries_measured": {"novel": [-12, 12], "trained_g0": 0},
            "wash_stream": ("e176n arm A's NEUTRAL protocol VERBATIM: e170's "
                            "neutral bank (seed 170), 16 neutral-anchor draws "
                            "+ 16 random corpus windows, full-token CE, no "
                            "fact windows in wash steps"),
            "replay_batch": ("e174 arm B's F1-replay batch VERBATIM: 16 e113 "
                             "jitter windows (7 name-char masked CE) + 16 "
                             "anchors (8 paired neutral + 8 random), union "
                             "CE; replay at every step % k == 0, replacing "
                             "that step's wash batch"),
            "optimizer": f"AdamW (0.9,0.95) wd 0.1 constant lr {FT_LR} clip "
                         "1.0; seed 10902 (locked lineage); 300 steps/cell",
            "ckpt_grid": list(CK_MAIN), "full_dial_at": list(FULL_DIAL_AT),
            "measure_dial": "e180's measure() (minus the 183-span census — "
                            "e178's recorded deviation)",
        },
        "gates": {
            "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
            "G_POOL": G_POOL, "G_JIT": G_JIT, "G_ANCHOR": G_ANCHOR,
            "G_ROOT": G_ROOT, "G_REPRO": G_REPRO,
            "G_DRAWFREE": {t: {"zeph_violations": cells[t]["zeph_violations"],
                               "pass": cells[t]["zeph_violations"] == 0}
                           for t in cells},
            "G_SURG": gates_surg,
        },
        "stored_refs": {"e176n_armA_r0": src_arma,
                        "e174_armB_r_half": src_e174b},
        "r_curve": r_curve,
        "pump_fit": pump,
        "adjudication": adjudication,
        "honesty_reflex": {
            "rate_accounting": ("r=1/32 is NINE replay events in 300 steps "
                                "(realized r 0.030): a threshold between 1/32 "
                                "and 1/8 is a threshold between ~9 and ~37 "
                                "interventions — the ladder's resolution, "
                                "not a continuum; r=1/4's +300 read sits ON a "
                                "replay step (300 % 4 == 0) while 1/32's sits "
                                "12 wash steps after its last replay: the "
                                "endpoint phase differs by schedule, and the "
                                "per-checkpoint trajectory (not the endpoint "
                                "alone) carries the shape"),
            "single_seed_single_lineage": ("one trajectory per rate, seed "
                                           "10902, one root (e131 "
                                           "consolidated) — n=1 per cell; "
                                           "e152R/e184 showed timing texture "
                                           "is seed-dependent while outcomes "
                                           "held; the threshold bracket is "
                                           "unreplicated"),
            "stream_mismatch_at_r_half": ("the 1/2 endpoint is e174 arm B, "
                                          "whose interleaved partner is the "
                                          "F2-INSTALL batch (stronger "
                                          "interference than the neutral "
                                          "wash) — the curve's end is a "
                                          "conservative floor, and its "
                                          "non-monotonicity (if any) vs the "
                                          "new cells is the stream delta, "
                                          "not a rate effect"),
            "device_mixing": ("stored r=0 ran pure CPU; this run's cells may "
                              "launch GPU and thermal-migrate mid-run "
                              "(e180's precedent) — recorded per cell; the "
                              "bars live at 0.27/0.5 vs floors ~0.004, beyond "
                              "float noise; the replicate's G_REPRO reports "
                              "the drift"),
            "pump_fit_scope": ("the linear pump is the SIMPLEST two-constant "
                               "summary; dead-flat floor segments attenuate w "
                               "and sawtooth segments violate the linear "
                               "form — the R^2 bar (0.80) is the honesty "
                               "dial, and a NO-PUMP is a finding about the "
                               "curve's form, not a failure of measurement"),
            "logits_alone": ("the g-12 ruler is a readout probability; the "
                             "e174 arm-B precedent (genuine cohabitation "
                             "with the graft formed) and e176n's whole-"
                             "anatomy wash argue the readout tracks the "
                             "memory here, but a maintained readout at tiny r "
                             "could in principle ride a re-teach cycle rather "
                             "than true maintenance — the +50 full dials "
                             "(held30, site read, census) co-report the "
                             "difference"),
        },
        "compute": {
            "train_device_policy": ("GPU allowed (park-once, quick check + "
                                    "mid-run guard); cooldown "
                                    f"{COOLDOWN_S:.0f}s around each training; "
                                    f"per-training cap {TRAIN_CAP_S:.0f}s"),
            "gpu_parked": GPU_PARKED, "park_reason": PARK_REASON,
            "device_events": device_events,
            "torch_threads": torch.get_num_threads(),
            "cells_devices": {t: {"initial": cells[t]["initial_device"],
                                  "final": cells[t]["final_device"]}
                              for t in cells},
        },
        "trims": trims,
        "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "elapsed_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log(f"outputs: {rd / 'metrics.json'}, {png}, ckpts "
        f"{len(CKPT_INVENTORY)} (e179_r*)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def make_plot(rd, cells, r_curve, adjudication, pump, root_cells,
              e174b, e176n_arma) -> Path:
    """THE PLOT (deliverable): F1-at-+300 vs r (the r-curve, with the wash
    and 1:1 endpoints), the trajectories, the pump fit, the verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    gm12_0 = root_cells["gm12"]

    # (0,0) trajectories: g-12 vs steps (log x)
    ax = axes[0, 0]
    ax.plot(e176n_arma["freeze_steps"][1:], e176n_arma["gm12"][1:], "o-",
            ms=5, lw=1.6, color="tab:red",
            label="r=0 wash (e176n armA, STORED)")
    colors = {"r0_wash": "darkred", "r1_32": "tab:orange", "r1_8": "tab:green",
              "r1_4": "tab:blue"}
    labels = {"r0_wash": "r=0 replicate (this run, rider)",
              "r1_32": "r=1/32 (9 replays)", "r1_8": "r=1/8 (37 replays)",
              "r1_4": "r=1/4 (75 replays)"}
    for tag in ("r0_wash", "r1_32", "r1_8", "r1_4"):
        tr = cells[tag]["traj"]
        ax.plot([t["step"] for t in tr], [t["g_m12_mean_pz"] for t in tr],
                "^--" if tag == "r0_wash" else "o-", ms=4, lw=1.5,
                color=colors[tag], alpha=0.55 if tag == "r0_wash" else 1.0,
                label=labels[tag])
    steps_b = sorted(e174b["ladder_gm12_by_step"])
    ax.plot(steps_b, [e174b["ladder_gm12_by_step"][s] for s in steps_b], "s:",
            ms=6, lw=1.4, color="tab:purple",
            label="r=1/2 (e174 armB, F2-install stream)")
    ax.axhline(SHUT_BAR, color="firebrick", ls="--", lw=1.0)
    ax.text(0.9, SHUT_BAR + 0.02, f"die bar {SHUT_BAR}", fontsize=7.5,
            color="firebrick")
    ax.axhline(MAINTAIN_BAR, color="seagreen", ls="--", lw=1.0)
    ax.text(0.9, MAINTAIN_BAR + 0.02, f"maintain bar {MAINTAIN_BAR}",
            fontsize=7.5, color="seagreen")
    ax.axhline(gm12_0, color="dimgray", ls=":", lw=1.0)
    ax.text(0.9, gm12_0 - 0.06, f"root {gm12_0:.3f}", fontsize=7,
            color="dimgray")
    ax.set_xscale("log")
    ax.set_xlabel("optimizer step (neutral wash + replay schedule)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("E179 — F1 trajectories per rehearsal rate (seed 10902)",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE R-CURVE: g-12 at +50 and +300 vs r (log x)
    ax = axes[0, 1]
    pts = r_curve["points"]
    r_new = [pts[f"{t}_this_run"]["r"] for t in ("1_32", "1_8", "1_4")]
    y50 = [pts[f"{t}_this_run"]["gm12_plus50"] for t in ("1_32", "1_8", "1_4")]
    y300 = [pts[f"{t}_this_run"]["gm12_plus300"] for t in ("1_32", "1_8", "1_4")]
    ax.plot(r_new, y300, "o-", ms=8, lw=1.8, color="black",
            label="F1 g-12 at +300 (this run)")
    for t, r_, y_ in zip(("r1_32", "r1_8", "r1_4"), r_new, y300):
        ax.annotate(f"1/{cells[t]['k']}\n{y_:.3f}", (r_, y_),
                    textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.plot(r_new, y50, "o--", ms=5, lw=1.2, color="gray", alpha=0.8,
            label="F1 g-12 at +50 (the dies-by-50 bar point)")
    cyc = [adjudication["per_rate"][t]["cycle_stats_late_grid"]
           ["mean_gm12_late"] for t in ("r1_32", "r1_8", "r1_4")]
    ax.plot(r_new, cyc, "^:", ms=6, lw=1.2, color="teal",
            label="late-grid cycle mean g-12 ({100,200,300}; phase-honest)")
    ax.plot([0.0], [pts["0.0_wash_stored_e176n_armA"]["gm12_plus300"]],
            "*", ms=16, color="tab:red", zorder=4,
            label="r=0 wash +300 (e176n armA, stored)")
    ax.plot([0.0], [pts["0.0_wash_replicate_this_run"]["gm12_plus300"]],
            "*", ms=10, color="darkred", markerfacecolor="none",
            markeredgewidth=1.4, zorder=4, label="r=0 replicate (this run)")
    ax.plot([0.5], [pts["0.5_rehearsal1to1_stored_e174_armB"]["gm12_plus300"]],
            "D", ms=8, color="tab:purple", zorder=4,
            label="r=1/2 (e174 armB; F2-install stream — conservative)")
    ax.axhline(MAINTAIN_BAR, color="seagreen", ls="--", lw=1.0)
    ax.axhline(SHUT_BAR, color="firebrick", ls="--", lw=1.0)
    if r_curve["threshold"]["r_star"] is not None:
        rstar = r_curve["threshold"]["r_star"]
        ax.axvline(rstar, color="black", ls=":", lw=1.6)
        ax.text(rstar, 0.06, f"r* = 1/{int(1 / rstar)}", fontsize=9,
                rotation=90, va="bottom", ha="right")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.015, 0.62)
    ax.set_xticks([0.0, 0.03125, 0.125, 0.25, 0.5])
    ax.set_xticklabels(["0 (wash)", "1/32", "1/8", "1/4", "1/2"])
    ax.set_xlabel("rehearsal fraction r (replay batches / total steps)")
    ax.set_ylabel("F1 g-12")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE REHEARSAL-FREQUENCY LAW — F1 level vs r "
                 f"({adjudication['shape']})", fontsize=9.5)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.25, which="both")

    # (1,0) pump fit: measured vs predicted segment deltas
    ax = axes[1, 0]
    fl = pump["linear"]
    segs = pump["segments"]
    nw = np.array([s["wash_steps"] for s in segs], dtype=float)
    nr = np.array([s["replay_steps"] for s in segs], dtype=float)
    dy = np.array([s["d_gm12"] for s in segs], dtype=float)
    pred = fl["w"] * nw + fl["g"] * nr
    ax.scatter(pred, dy, s=22, alpha=0.75,
               c=["tab:orange" if s["tag"] == "r1_32"
                  else "tab:green" if s["tag"] == "r1_8" else "tab:blue"
                  for s in segs])
    lim = max(abs(dy).max(), abs(pred).max()) * 1.15 + 1e-6
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"predicted d(g-12) = w*n_wash + g*n_replay   "
                  f"(w={fl['w']:+.2e}, g={fl['g']:+.2e})")
    ax.set_ylabel("measured d(g-12) per segment")
    ax.set_title(f"PUMP FIT — pooled two-constant model "
                 f"(R\u00b2={fl['r2_uncentered']:.3f} vs bar "
                 f"{PUMP_R2_BAR}; {fl['n_segments']} segments; "
                 f"{adjudication['pump_verdict']})", fontsize=9.5)
    ax.grid(alpha=0.25)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    vlines = [
        "REGISTERED (dispatch verbatim; no bar shopping):",
        "  THRESHOLD: a sharp r* (below it the fact dies by +50;",
        "   at or above it maintains >= 0.5) — a phase transition",
        "  GRADED: any rehearsal slows the wash proportionally",
        "   (no threshold; +300 level rises smoothly with r)",
        "  PUMP-FIT: both w (wash rate) and g (rehearsal gain)",
        "   extractable — the maintenance law's two constants",
        "",
        f"SHAPE: {adjudication['shape']}",
    ] + [f"  {wd}" for wd in
         [adjudication["shape_clause"][i:i + 64]
          for i in range(0, len(adjudication["shape_clause"]), 64)]]
    vlines += ["", f"PUMP: {adjudication['pump_verdict']}"]
    vlines += [f"  {wd}" for wd in
               [adjudication["pump_clause"][i:i + 64]
                for i in range(0, len(adjudication["pump_clause"]), 64)]]
    vlines += ["", f"VERDICT: {adjudication['verdict']}",
               "", "r-curve (+300 g-12): "
               f"r=0 {r_curve['points']['0.0_wash_stored_e176n_armA']['gm12_plus300']:.4f}"
               f" (stored; repl "
               f"{r_curve['points']['0.0_wash_replicate_this_run']['gm12_plus300']:.4f})"
               + " | " + " | ".join(
                   f"1/{cells[t]['k']} "
                   f"{r_curve['points'][f'{t[1:]}_this_run']['gm12_plus300']:.4f}"
                   for t in ("r1_32", "r1_8", "r1_4"))
               + f" | 1/2 {e174b['gm12']:.4f} (e174 armB)"]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.0265, tx, fontsize=6.8, va="top",
                family="monospace")

    fig.suptitle("E179 — THE REHEARSAL-FREQUENCY LAW: the maintenance "
                 "threshold r* (root e131_consolidated; neutral wash + "
                 "interleaved F1-replay, seed 10902) -> "
                 f"{adjudication['verdict']}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "rehearsal_law.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


if __name__ == "__main__":
    sys.exit(main())
