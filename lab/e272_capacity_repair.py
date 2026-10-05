"""E272 — THE CAPACITY REPAIR LADDER (the R65 frontier review's top repair
bill: the critic found the capacity number's threshold only BRACKETED
(1k,10k] and its rank/dose confound LIVE — kept-fraction varies 3.7x across
the cliff pair by construction, and ZERO dose-at-matched-rank arms exist in
the record). Design dispatched 2026-10-05; this docstring carries the
registered bars VERBATIM, committed at birth BEFORE any compute. Adjudicate
against exactly this; no bar shopping.

THE QUESTION (frozen, verbatim from the dispatch): Is the expression cliff
written by RANK or by DOSE? Where exactly in (1k, 10k] does the edge sit?
And how big is the room lottery at the cliff?

THE FOUR ARMS (2.74M organism, the family's standard fact protocol, all
serial/quiet-water — NO concurrency in this cell; every arm the SAME fresh
root (e001, fact-free-gated), the SAME dose/stream (Dmix install s400 gen
24314 + e113 cons s300 seed 10901 HELD), the hook VERBATIM (backward ->
clip 1.0 -> project (CPU fp64, write fp32) -> opt.step; norm NOT
rescaled)):
  (a) K2K — the k=2,000 rung, e261's machinery verbatim (fresh room seeds
      27211/27212, e260's RANDOM construction convention).
  (b) K5K — the k=5,000 rung, same (fresh seeds 27213/27214).
  (c) K1KM (KEPT-MATCHED-1K) — e261's committed k=1,000 room VERBATIM
      (seeds 26111/26112, bit-gated vs e261_rooms.pt) with the INSTALL lr
      COMPENSATED so the APPLIED in-room dose matches the 10k arm's:
      scaling factor = stored_kept(10k)/stored_kept(1k) = 0.060045162390265784
      / 0.016095496225535792 = 3.7305567687315575 (computed from the
      COMMITTED kept_frac_curve, md5-bound; the exact stored values are
      stated in metrics['dose_control']). The projection kills out-of-room
      mass by construction, so lr-compensation matches the applied in-room
      step norm (first-order; the empirical match is MEASURED in the
      dose_control ledger — never nominal). CONS stays NATURAL (e113
      VERBATIM, seed 10901 HELD, lr 1e-3 const — only the INSTALL lr is
      compensated, per the dispatch).
  (d) K10KR (K10K-REPLICATE) — the k=10,000 rung with FRESH room seeds
      (27215/27216; same protocol, same everything else) — prices the room
      lottery at the cliff rung vs the committed rung (post g0 0.2646476328,
      room 26113/26114).

REGISTERED BARS (frozen in the dispatch letter, VERBATIM in this docstring
BEFORE any compute; adjudicate against exactly this; no bar shopping).
Adjudication is on the WRITE read — post g0 (the g0 battery on the
install-final net), the bit-faithful one; the landing/root read is CARRIED
under the family's rehearsal-lane caveat (T246; e269/e270/e271 precedent:
the cons re-teaches — the root g0 measures the rehearsal lane, not the
write), never adjudicated:
  - RANK-WRITES-THE-CURVE — "the kept-matched 1k arm STAYS DEAD (post g0 <
    0.01) AND the {2k,5k} edge lands inside (1k,10k] (at least one adjacent
    pair among 1k->2k->5k->10k jumps >=10x in post g0, with the dead side
    < 0.01) — dose acquitted at matched rank; the capacity story stands
    with a located edge"
  - DOSE-WRITES-THE-CURVE — "the kept-matched 1k arm EXPRESSES (post g0 >=
    0.05) — the 609x jump was the 3.7x kept-dose change in disguise; the
    capacity number falls to a dose artifact; the critic's embarrassment
    scenario, caught in time"
  - ROOM-LOTTERY — "the 10k replicate's post g0 leaves [0.15, 0.45] (the
    committed rung was 0.2646) — the cliff pair untrusted; demand
    replicates before any further capacity claim"
  - MIXED — "anything else — the trajectories verbatim, both readouts, no
    inflation"

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * THE EDGE LADDER := the ascending posts over {1k, 2k, 5k, 10k} — 1k =
    e261's committed K1K rung (post g0 0.00043458465370349586, CITED,
    md5-bound < 1e-12), 2k/5k = THIS cell's fresh rungs, 10k = e264's
    committed K10K rung (post g0 0.26464763283729553, CITED, md5-bound <
    1e-12; the room 26113/26114). The K10KR replicate is the room-lottery
    PROBE at the 10k rung, never a ladder rung; a SENSITIVITY ladder with
    the replicate's post substituted at 10k is CO-REPORTED (if the verdict
    flips under it, that is disclosed in the clause — the committed cite
    stays primary).
  * "at least one adjacent pair ... jumps >= 10x ... with the dead side <
    0.01" := exists an adjacent pair (k_lo -> k_hi) among (1k->2k),
    (2k->5k), (5k->10k) with post(k_hi)/max(post(k_lo), 1e-6) >= 10 AND
    post(k_lo) < 0.01; the edge bracket := the firing pair with the
    largest k_lo (the tightest); ALL firing pairs reported; the family
    expression floor 0.05 crossing co-reported as context.
  * THE DOSE PROBE := K1KM's post g0: dead < 0.01 (the RANK clause's dead
    bar); expresses >= 0.05 (the DOSE clause's expression floor — the
    family's frozen 0.05); the gap [0.01, 0.05) is the named MIXED branch
    "PARTIAL EXPRESSION AT MATCHED DOSE" (P-272a's 'below the expression
    floor' would still hold at 0.05, but the frozen letter governs).
  * THE ROOM LOTTERY := K10KR's post g0 vs the window [0.15, 0.45]; a
    leave fires ROOM-LOTTERY. Composite order (frozen): hard-gate failure
    (TEXTURE; nothing adjudicated) -> [non-halting texture disclosures
    carried] -> RANK-WRITES-THE-CURVE -> DOSE-WRITES-THE-CURVE ->
    ROOM-LOTTERY -> MIXED. If ROOM-LOTTERY co-fires with RANK/DOSE, the
    verdict stands and the clause carries the rider ("the cliff pair
    untrusted; demand replicates") verbatim.
  * THE COMPENSATION ARITHMETIC (exact, gate-able): LR_SCALE =
    kept_frac_curve_committed['10000'] / kept_frac_curve_committed['1000']
    re-derived at runtime from runs/e264/metrics.json (md5-bound
    a42ff4786784b04cb9819a69b545e343) and asserted == the frozen literal
    3.7305567687315575; the compensated install's lr(s) = 1e-3 x LR_SCALE x
    cosine_lr(s-1, 1000) — the house schedule scaled UNIFORMLY; implemented
    by the disclosed module-global rebind of e043_install.LR around K1KM's
    install call ONLY (e264's E261.CONS_SEED rebind precedent), restored in
    a finally block; the cons driver reads G1.FT_LR and is untouched.
    DISCLOSED COSTS of the frozen form, measured never nominal: (i) Adam's
    per-coordinate normalizer partially cancels uniform gradient scales, so
    the in-room step-norm match is first-order — the per-step ledger
    (lr_s x ||g'|| medians) and the end-to-end in-room displacement norms
    ||P_room(theta_final - theta_0)|| are measured and reported; (ii) the
    rebind scales AdamW's decoupled weight decay with the same lr (the
    honest price of the one-knob compensation; disclosed).
  * HARD GATES (a failure HALTS; the family's 12-gate style, adapted):
    {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR(bank), G_INSTMASK, G_PARENTS,
    G_DOSE_ARITH, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOM1K}.
    NO FREE ARM and NO anchor re-run (disclosed: no arm in this cell is
    bit-identical to a committed rung BY CONSTRUCTION — the 2k/5k rungs are
    new ks, K1KM compensates lr, K10KR draws a fresh room; the instrument
    is the 5-generation lineage: e264's G_FREE PASS at install L2 6.2e-4 +
    e268/e269/e270/e271's bit-identical serial anchors at |d post g0|
    2.4e-7/9.5e-7/1.0e-6/9.5e-7; K10KR itself is the live fresh-room
    calibration at the cliff rung).
  * THE LANDING READ per arm (root g0 after natural cons) is CARRIED as
    context with the rehearsal-lane caveat, never adjudicated; root g-12
    reported with the lottery note, never gated.
  * MEASURED, NEVER NOMINAL: per-arm kept-fraction ledger (median over the
    install ledger), v-excess pre/post (vs e258's LOADED committed v-map),
    in-span pre AND applied (vs e246's committed LATE span), displacement
    cos-to-span + in-own-room at post-install and root, and the
    dose_control ledger (the applied-dose proxies + the end-to-end in-room
    displacement norms).

REGISTERED PREDICTIONS already on the record (cited, NOT re-registered):
P-272a (T249): "the kept-matched 1k arm STILL DIES below the expression
floor — dose does not buy expression." The creep/edge expectations carried
verbatim: "if the edge is sharp at ~10k, K2K and K5K both land dead
(<0.01); if the bracket slides, K5K may express partially — that is
exactly the location information the cell buys."

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); the
owner's max-priority window — bursts <= 175 s, per-step thermal polls at a
78 C margin, 40 s cooldowns (the 30-60 s window), the 84 C never-past line
recorded; CPU fp64 dense projections (pocketfft workers 2); CPU probing
threads 4; SERIAL arms only (no concurrency in this cell).

Outputs: runs/e272/{metrics.json (PROGRESSIVE), e272_capacity_repair.png,
e272_instrument.png, REPORT.md, run.log (gitignored)}; checkpoints
runs/checkpoints/e272_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e272_capacity_repair.py    (E272_SMOKE=1 shakedown)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # CORP_BS, MIX_RANDOM,
                                                      # LR, jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # LADDER MACHINERY,
                                                      # PORTED WHOLE BY
                                                      # IMPORT (the serial
                                                      # install + cons
                                                      # drivers; the
                                                      # committed file is
                                                      # NOT modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E272_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e272_smoke" if SMOKE else "e272"
assert torch.cuda.is_available(), "e272 owns the GPU lane (dispatch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


G1.log = log                                          # unify the timeline

# the shared instrument ledgers (defined BEFORE the rebinding below so
# e261's drivers land their rows in THIS cell's ledger — e270's
# instrumentation convention, inherited from e271)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e264/e268/e271's disclosed convention, in force before
# ANY machinery call): e261's ported drivers resolve their module globals
# (log / NAME / SMOKE / LADDER / RUNG_NAMES / thermal_log / device_events)
# AT CALL TIME through e261's module namespace — rebound HERE so they write
# THIS cell's log, label THIS cell's envelope polls, run THIS cell's room
# set, and land their thermal rows in THIS cell's ledger. The committed
# lab/e261_rank_ladder.py itself is untouched.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.thermal_log = thermal_log
E261.device_events = device_events
if SMOKE:
    E261.INST_STEPS = 8
    E261.CONS_STEPS = 8

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
ROOMS261_CK = "e261_rooms.pt"     # e261's committed rooms (the K1K bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE FOUR ARMS' ROOMS. Full mode: {K1KM: k=1000 at e261's committed
# K1K seeds (26111/26112 — the SHARED room, bit-gated), K2K: k=2000 fresh
# (27211/27212), K5K: k=5000 fresh (27213/27214), K10KR: k=10000 FRESH
# (27215/27216 — the room-lottery probe; NOT the committed 26113/26114)}.
# Smoke: the same four names at smoke ks (the bit-gate vacuous, disclosed).
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (1_000, 26111, 26112),        # K1KM — e261's committed K1K room VERBATIM
    (2_000, 27211, 27212),        # K2K — fresh
    (5_000, 27213, 27214),        # K5K — fresh
    (10_000, 27215, 27216),       # K10KR — fresh (the room-lottery probe)
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (256, 26111, 26112),          # K1KM analog
    (64, 27211, 27212),           # K2K analog
    (128, 27213, 27214),          # K5K analog
    (512, 27215, 27216),          # K10KR analog
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG_NAMES = {1000: "K1KM", 2000: "K2K", 5000: "K5K", 10000: "K10KR"}
if SMOKE:
    RUNG_NAMES = {k: nm for k, (_, _, _), nm in
                  zip((256, 64, 128, 512), LADDER_SMOKE,
                      ("K1KM", "K2K", "K5K", "K10KR"))}
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG_NAMES
ARMS = ("K10KR", "K2K", "K5K", "K1KM")      # EXECUTION order: the replicate
                                            # first (the live fresh-room
                                            # calibration at the cliff rung),
                                            # then the rungs ascending, the
                                            # dose probe LAST
K1KM_MODE = RUNG_NAMES[LADDER_FULL[0][0]] if not SMOKE else "K1KM"
DOSE_ARM = "K1KM"

# ---- THE DOSE COMPENSATION (frozen): the exact stored kept values from
# e264's COMMITTED kept_frac_curve (md5-bound; re-derived + asserted at
# runtime in G_DOSE_ARITH), and their quotient.
KEPT_1K_STORED = 0.016095496225535792            # kept_frac_curve['1000']
KEPT_10K_STORED = 0.060045162390265784           # kept_frac_curve['10000']
LR_SCALE_FROZEN = KEPT_10K_STORED / KEPT_1K_STORED   # 3.7305567687315575
LR_BASE = None            # read at bind from E43.LR (must be 1e-3)

# ---- the committed records, HARD-BOUND (Rule 12; md5-gated at runtime)
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "f460475d8e6b76f0719e91c1e9c6041b"
E261_K1K_POST = 0.00043458465370349586            # THE cited 1k rung (dead)
E261_K1K_ROOT = 0.5768678784370422
E261_ROOMS_MD5 = "f81571c40bb6009d4ed1cfab66e73f6c"
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_K10K_POST = 0.26464763283729553              # THE cited 10k rung (alive)
E264_K10K_ROOT = 0.7707884907722473
E264_VERDICT = "SHARP-THRESHOLD"
E264_GFREE_L2 = 0.0006242410945250543             # the instrument lineage's
                                                  # cited validation point
# the committed 5-rung curve (context co-plots; from e264's adjudication)
E264_CURVE_POST = {1000: 0.00043458465370349586, 10000: 0.26464763283729553,
                   40000: 0.346476286649704, 100000: 0.43598243594169617,
                   237123: 0.38436421751976013}
E264_CURVE_KEPT = {1000: 0.016095496225535792, 10000: 0.060045162390265784,
                   40000: 0.12105602142254653, 100000: 0.19149141218938606,
                   237123: 0.2945979051104054}
E264_CURVE_ROOT = {1000: 0.5768678784370422, 10000: 0.7707884907722473,
                   40000: 0.7519555687904358, 100000: 0.7276116609573364,
                   237123: 0.7105798125267029}
# the serial-anchor lineage (the instrument validation, cited not re-run)
ANCHOR_LINEAGE = {"e264_G_FREE_l2": 6.24e-4, "e268_post_d": 2.4e-7,
                  "e269_post_d": 9.5e-7, "e270_post_d": 1.0e-6,
                  "e271_post_d": 9.5e-7}

# the frozen bars' numbers
DEAD_BAR = 0.01                   # the RANK clause's dead side (< 0.01)
EXPRESS_FLOOR = 0.05              # the DOSE clause's expression floor (the
                                  # family's frozen 0.05)
JUMP_BAR = 10.0                   # the >= 10x adjacent-jump bar
RATIO_DEN_FLOOR = 1e-6            # the ladder's floor-guard convention
ROOM_WINDOW = (0.15, 0.45)        # the ROOM-LOTTERY window on the replicate
G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 (context co-report)
MATCH_BAND = E261.MATCH_BAND              # the landing band (context only)
G_READ_TOL = E261.G_READ_TOL             # 5e-3

REGISTERED = {
    "question_verbatim": "Is the expression cliff written by RANK or by "
        "DOSE? Where exactly in (1k, 10k] does the edge sit? And how big is "
        "the room lottery at the cliff?",
    "bars_verbatim": {
        "RANK-WRITES-THE-CURVE": "the kept-matched 1k arm STAYS DEAD (post "
            "g0 < 0.01) AND the {2k,5k} edge lands inside (1k,10k] (at least "
            "one adjacent pair among 1k->2k->5k->10k jumps >=10x in post g0, "
            "with the dead side < 0.01) — dose acquitted at matched rank; "
            "the capacity story stands with a located edge",
        "DOSE-WRITES-THE-CURVE": "the kept-matched 1k arm EXPRESSES (post "
            "g0 >= 0.05) — the 609x jump was the 3.7x kept-dose change in "
            "disguise; the capacity number falls to a dose artifact; the "
            "critic's embarrassment scenario, caught in time",
        "ROOM-LOTTERY": "the 10k replicate's post g0 leaves [0.15, 0.45] "
            "(the committed rung was 0.2646) — the cliff pair untrusted; "
            "demand replicates before any further capacity claim",
        "MIXED": "anything else — the trajectories verbatim, both readouts, "
            "no inflation",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE CELL := four serial arms at the family's "
        "standard fact protocol (e001 fresh fact-free base; Dmix install "
        "s400 gen 24314, bit-identical streams; hook VERBATIM (backward -> "
        "clip 1.0 -> project CPU fp64 -> opt.step, norm NOT rescaled); e113 "
        "cons s300 seed 10901 HELD NATURAL on all arms): K2K (k=2000, fresh "
        "seeds 27211/27212), K5K (k=5000, 27213/27214), K1KM (k=1000 at "
        "e261's committed K1K room seeds 26111/26112, bit-gated vs "
        "e261_rooms.pt, with the INSTALL lr COMPENSATED by LR_SCALE = "
        f"stored_kept(10k)/stored_kept(1k) = {KEPT_10K_STORED!r}/"
        f"{KEPT_1K_STORED!r} = {LR_SCALE_FROZEN!r}, the house cosine scaled "
        "UNIFORMLY via the disclosed e043_install.LR module rebind, restored "
        "in a finally block; cons untouched), K10KR (k=10000 with FRESH room "
        "seeds 27215/27216 — the room-lottery probe); THE EDGE LADDER := "
        "posts over {1k cited (e261, md5-bound), 2k, 5k, 10k cited (e264, "
        "md5-bound)}; 'jumps >=10x with the dead side < 0.01' := exists "
        "adjacent pair (k_lo->k_hi) with post(k_hi)/max(post(k_lo), 1e-6) >= "
        f"10 AND post(k_lo) < {DEAD_BAR}; edge bracket := the firing pair "
        "with the largest k_lo; all firing pairs reported; the floor 0.05 "
        "crossing co-reported as context; the K10KR replicate is NEVER a "
        "ladder rung — a sensitivity ladder with its post substituted at 10k "
        "is CO-REPORTED (the committed cite stays primary); THE DOSE PROBE "
        f":= K1KM post g0 — dead < {DEAD_BAR} (RANK clause), expresses >= "
        f"{EXPRESS_FLOOR} (DOSE clause), the gap [{DEAD_BAR}, {EXPRESS_FLOOR})"
        " is the named MIXED branch 'PARTIAL EXPRESSION AT MATCHED DOSE'; "
        f"THE ROOM LOTTERY := K10KR post g0 vs the window "
        f"[{ROOM_WINDOW[0]}, {ROOM_WINDOW[1]}]; composite order: hard-gate "
        "failure (TEXTURE; nothing adjudicated) -> [non-halting texture "
        "disclosures carried] -> RANK-WRITES-THE-CURVE -> DOSE-WRITES-THE-"
        "CURVE -> ROOM-LOTTERY -> MIXED (a co-firing ROOM-LOTTERY rides the "
        "clause verbatim); the ADJUDICATION READ is the WRITE read (post g0 "
        "on the install-final net); the landing read (root g0 after natural "
        "cons) is CARRIED under T246's rehearsal-lane caveat, never "
        "adjudicated; root g-12 never gated; HARD GATES := {G_NAMEFREE, "
        "G_SPLICE, G_BATTERY, G_ANCHOR(bank), G_INSTMASK, G_PARENTS, "
        "G_DOSE_ARITH, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, "
        "G_ROOM1K} — a failure HALTS; NO FREE arm and NO anchor re-run "
        "(disclosed: no arm is bit-identical to a committed rung BY "
        "CONSTRUCTION; the instrument is the 5-generation lineage — e264's "
        "G_FREE PASS at install L2 6.2e-4 + e268/e269/e270/e271's serial "
        "anchors at |d post g0| 2.4e-7/9.5e-7/1.0e-6/9.5e-7); MEASURED "
        "NEVER NOMINAL: kept ledger, v-excess pre/post, in-span pre/applied, "
        "displacement loads, the lr_s x ||g'|| applied-dose proxies + the "
        "end-to-end in-room displacement norms (the compensation's empirical "
        "match); P-272a (T249) CITED, never re-registered."),
    "registration": "bars + question frozen VERBATIM from the R65 repair "
        "dispatch (the capacity repair ladder, the frontier review's top "
        "repair bill); this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping.",
    "predictions_cited": {
        "P-272a_T249": "the kept-matched 1k arm STILL DIES below the "
                       "expression floor — dose does not buy expression "
                       "(CITED from the record, not re-registered)",
        "edge_expectation": "if the edge is sharp at ~10k, K2K and K5K both "
                            "land dead (<0.01); if the bracket slides, K5K "
                            "may express partially — that is exactly the "
                            "location information the cell buys",
    },
}

deviations: list[str] = [
    "NO FREE ARM, NO ANCHOR RE-RUN (disclosed at birth): no arm in this "
    "cell is bit-identical to a committed rung BY CONSTRUCTION — K2K/K5K "
    "are new ks, K1KM compensates the install lr, K10KR draws a fresh room "
    "(the room lottery IS its object). The instrument is the 5-generation "
    "lineage: e264's G_FREE PASS (install L2 6.2e-4, root in band) + "
    "e268/e269/e270/e271's bit-identical serial anchors (|d post g0| "
    "2.4e-7/9.5e-7/1.0e-6/9.5e-7 — the WRITE read's cross-session "
    "determinism law); K10KR is itself the live fresh-room calibration at "
    "the cliff rung. No bar or gate-form change follows.",
    "THE LR COMPENSATION IS A DISCLOSED MODULE-GLOBAL REBIND: e261's "
    "chunked_install reads e043_install.LR at CALL TIME, so K1KM's install "
    "runs under E43.LR = 1e-3 x LR_SCALE (restored in a finally block; the "
    "cons driver reads G1.FT_LR and is untouched) — e264's E261.CONS_SEED "
    "rebind precedent for the one-arm draw difference. The compensation "
    "scales the house cosine UNIFORMLY (lr(s) = 1e-3 x LR_SCALE x "
    "cosine_lr(s-1, 1000)) and, disclosed as the frozen form's honest "
    "costs: (i) Adam's per-coordinate normalizer partially cancels uniform "
    "gradient scales, so the in-room step-norm match is FIRST-ORDER — "
    "measured in the dose_control ledger (lr_s x ||g'|| medians + the "
    "end-to-end in-room displacement norms), never assumed; (ii) the rebind "
    "scales AdamW's decoupled weight decay with the same lr.",
    "THE 10k RUNG IS CITED, THE REPLICATE IS THE PROBE: the edge ladder's "
    "10k post is e264's committed K10K rung (room 26113/26114) — K10KR "
    "(room 27215/27216) prices the room lottery and is never a rung; a "
    "sensitivity ladder with the replicate substituted is co-reported, and "
    "if the verdict flips under it the clause discloses it (the committed "
    "cite stays primary — no bar shopping).",
    "THE LANDING READ IS NOT A BAR (T246's rehearsal-lane finding, carried; "
    "e269/e270/e271 triply-quadruply confirmed): the frozen bars adjudicate "
    "the WRITE read (post g0) only; the root g0 is reported per arm as "
    "context (dead writes have landed 0.70-0.81 across e268-e271 — the cons "
    "re-teaches; the landing read measures the rehearsal lane); root g-12 "
    "is the wild lottery, reported never gated.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked chunked install/cons drivers (bit-identical arithmetic + draw "
    "order), the thermal envelope (per-step polls, 78C margin, 175s bursts, "
    "40s cooldowns, the 84C line), the progressive-metrics + resume-ckpt "
    "conventions — the module-global rebinding (log/NAME/LADDER/RUNG_NAMES/"
    "thermal ledgers, disclosed in-code) retargets the drivers' I/O to this "
    "cell; the committed lab/e261_rank_ladder.py is NOT modified.",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger; no new "
    "history is run. The installs' own fresh Adam state remains the "
    "mechanism's channel (the rungs restrict the GRADIENT; the in-own-room "
    "ledger says how much of the state stayed).",
    "NO WASH IN THIS CELL (the ladder's inherited registration): the "
    "readouts are post g0 (adjudicated) + root g0 (carried) + kept per arm; "
    "the retention/occupancy question stays T237's separately registered "
    "follow-up.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat — "
    "the critic's lottery note carried verbatim); K10KR exists precisely to "
    "price the n=1 room lottery at the cliff rung; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
    "Smoke mode (E272_SMOKE=1): 8-step installs/cons, the four room names "
    "at smoke ks {256(K1KM-analog), 64, 128, 512}, G_ROOM1K vacuous (no "
    "committed record at smoke k; disclosed), the compensation exercised "
    "with the SAME committed LR_SCALE, all paths smoke_-prefixed, own smoke "
    "dir; NOTHING adjudicated or gated (SMOKE stamp on every read).",
]


# ------------------------------------------------------------------ envelope
def gpu_poll(tag: str) -> dict:
    s = common.gpu_status()
    ok = s["util"] <= common.GPU_UTIL_CEIL and s["temp"] <= common.GPU_TEMP_CEIL
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"], ok)
    log(f"  [gpu:{tag}] util {s['util']:.0f}% temp {s['temp']:.0f}C "
        f"mem {s['mem_used']:.0f}/{s['mem_total']:.0f}MB "
        f"power {s['power']:.1f}W -> {'OK' if ok else 'HOLD'}")
    return {"poll": s, "ok": bool(ok)}


def wait_gpu_free(tag: str, max_wait_s: float = 1800.0) -> list[dict]:
    polls = [gpu_poll(f"{tag}#1")]
    time.sleep(E261.LAUNCH_POLL_GAP_S)
    polls.append(gpu_poll(f"{tag}#2"))
    t0w = time.time()
    while not (polls[-2]["ok"] and polls[-1]["ok"]):
        if time.time() - t0w > max_wait_s:
            raise RuntimeError(f"GPU window never opened for {tag}")
        log(f"  [gpu:{tag}] waiting 20s for the envelope")
        time.sleep(20.0)
        polls.append(gpu_poll(f"{tag}#w"))
    return polls


def burst_cooldown(tag: str) -> None:
    log(f"[thermal] cooldown {E261.COOLDOWN_S:.0f}s ({tag})")
    time.sleep(E261.COOLDOWN_S)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def save_ckpt(name: str, sd: dict, meta: dict) -> str:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": NAME, **meta}}, path)
    log(f"[ckpt] saved {path.name}")
    return str(path.relative_to(E43.REPO)).replace("\\", "/")


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def inst_resume_path(arm: str) -> Path:
    return CKPT_DIR / (f"smoke_e272_{arm}_inst_resume.pt" if SMOKE
                       else f"e272_{arm}_inst_resume.pt")


def cons_resume_path(arm: str) -> Path:
    return CKPT_DIR / (f"smoke_e272_{arm}_cons_resume.pt" if SMOKE
                       else f"e272_{arm}_cons_resume.pt")


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def main():
    global LR_BASE
    dev = torch.device("cuda")
    LR_BASE = float(E43.LR)
    assert abs(LR_BASE - 1e-3) < 1e-15, f"E43.LR unexpected: {LR_BASE}"
    metrics.update({
        "experiment": "e272_capacity_repair",
        "phase": "THE CAPACITY REPAIR LADDER — the R65 review's top repair "
                 "bill: is the expression cliff written by RANK or by DOSE? "
                 "{K2K, K5K} locate the edge inside (1k,10k]; K1KM (the "
                 "committed 1k room, install lr x 3.7306 = kept(10k)/kept(1k)"
                 ") is the dose-at-matched-rank probe; K10KR (fresh room at "
                 "k=10k) prices the room lottery at the cliff — "
                 "RANK-WRITES vs DOSE-WRITES vs ROOM-LOTTERY vs MIXED, "
                 "adjudicated on the WRITE read (post g0)",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (the owner's max-priority window; "
                      "this cell owns the GPU lane) + CPU fp64 dense "
                      "projections (pocketfft workers 2), CPU probing "
                      "threads 4; SERIAL ARMS ONLY (no concurrency)",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s, per-step thermal polls "
                      f"at a {E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line recorded",
            "trainings": "4 installs s400 (K10KR/K2K/K5K hooked at lr 1e-3; "
                         "K1KM hooked at lr 1e-3 x LR_SCALE) + 4 cons s300 "
                         "(seed 10901 HELD, natural); NO washes; NO FREE "
                         "(the 5-generation instrument lineage cited)",
        },
        "deviations": deviations,
        "builds_on": [
            "R65 / the frontier review (THE repair bill: the capacity "
            "number's threshold only bracketed (1k,10k], the rank/dose "
            "confound live — kept varies 3.7x across the cliff pair by "
            "construction, ZERO dose-at-matched-rank arms in the record)",
            "T249 / e271 (the dispatch source: P-272a registered there — "
            "the kept-matched 1k arm still dies below the expression floor; "
            "the serial-anchor lineage's freshest point |d post g0| 9.5e-7)",
            "T242 / e264 (THE cliff record: the completed 5-rung ladder — "
            "the committed kept_frac_curve {1k 0.0161, 10k 0.0600, 40k "
            "0.1211, 100k 0.1915, 237k 0.2946} that freezes this cell's "
            "compensation factor; the cited 10k rung post 0.2646; the G_FREE "
            "instrument validation)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT: the "
            "SRCT projector, the hooked drivers, the thermal envelope; the "
            "cited 1k rung post 0.000435; the committed K1K room this "
            "cell's K1KM arm bit-binds)",
            "T246 / e268 + T247 / e269 + T248 / e270 (the rehearsal-lane "
            "finding — the landing read carried, never adjudicated; the "
            "serial-anchor re-run discipline this cell cites in lieu of a "
            "FREE arm)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314 "
            "+ e113 cons 10901; the committed root + install records are "
            "the controls)",
        ],
        "whats_new": [
            "THE DOSE-AT-MATCHED-RANK ARM (the record's first): the "
            "committed k=1k room with the install lr compensated by "
            "kept(10k)/kept(1k) = 3.7306 — the 3.7x kept-dose change across "
            "the cliff pair, applied at matched rank; if the cliff is rank-"
            "written this arm stays dead, if dose-written it expresses",
            "THE EDGE LOCATED: fresh rungs at k=2k and k=5k walk inside the "
            "(1k,10k] bracket for the first time — the tightest firing pair "
            "is the located edge bracket",
            "THE ROOM LOTTERY PRICED: a fresh-room replicate at the cliff "
            "rung k=10k vs the committed rung (0.2646) — the n=1 room-draw "
            "variance at the capacity number's own rung, with the "
            "[0.15,0.45] trust window pre-registered",
        ],
        "gates": {},
    })
    log(f"E272 — THE CAPACITY REPAIR LADDER (smoke={SMOKE}) -> {RD}")
    log(f"arms (serial, execution order): {' -> '.join(ARMS)}; the edge "
        f"ladder {{1k cited {E261_K1K_POST:.6f}, 2k, 5k, 10k cited "
        f"{E264_K10K_POST:.6f}}}; the dose probe K1KM at LR_SCALE "
        f"{LR_SCALE_FROZEN:.10f} = {KEPT_10K_STORED!r}/{KEPT_1K_STORED!r}; "
        f"the room window {ROOM_WINDOW}; dead bar {DEAD_BAR}, expression "
        f"floor {EXPRESS_FLOOR}, jump bar >={JUMP_BAR:.0f}x")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(27201)                 # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (g1c's gates VERBATIM) ==
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
    name_ids = corpus.encode(G1.NAME)

    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "expected": {"g-12": [60, G1.PRE - 12], "g0": [60, G1.PRE],
                              "g+12": [60, G1.PRE + 12]},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} at j={j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]

    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries = [], 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            tries += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    G_ANCHOR = {"neutral_bank": {"seed": G1.E170_ANCHOR_SEED,
                                 "n_windows": 16, "starts": n_starts,
                                 "note": "built for protocol identity; NO "
                                         "wash runs in this cell"},
                "pass": bool(len(n_starts) == 16)}
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                             "G_INSTMASK": G_INSTMASK})
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank / install mask)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e261m = json.loads(E261_METRICS.read_text(encoding="utf-8"))
    e261_k1k_post = e261m["arms"]["K1K"]["install"]["post_cells"]["g0"]
    e261_k1k_root = e261m["arms"]["K1K"]["root"]["g0"]
    e261_k1k_kept = e261m["arms"]["K1K"]["install"][
        "ledger_kept_frac_median"]
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_adj_reads = e264m["adjudication"]["reads"]
    kept_curve_file = {int(k): v for k, v in
                       e264_adj_reads["kept_frac_curve"].items()}
    post_curve_file = {int(k): v for k, v in
                       e264_adj_reads["post_g0_curve"].items()}
    e264_free_pass = bool(e264m["gates"]["G_FREE"]["pass"])
    G_PARENTS = {
        "e261_metrics": {"path": str(E261_METRICS),
                         "md5": md5of(E261_METRICS), "bound_md5": E261_MD5,
                         "verdict": e261m["adjudication"]["verdict"],
                         "K1K_post_g0": e261_k1k_post,
                         "K1K_root_g0": e261_k1k_root,
                         "K1K_kept_median": e261_k1k_kept,
                         "note": "the cited 1k rung (dead) + the committed "
                                 "K1K room's source record"},
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "K10K_post_g0": post_curve_file[10000],
                         "kept_frac_curve": {str(k): v for k, v in
                                             sorted(kept_curve_file.items())},
                         "G_FREE_pass": e264_free_pass,
                         "note": "THE cliff record: the completed 5-rung "
                                 "ladder whose committed kept_frac_curve "
                                 "freezes this cell's compensation factor; "
                                 "the cited 10k rung (alive)"},
        "e261_rooms": {"path": f"runs/checkpoints/{ROOMS261_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS261_CK),
                       "bound_md5": E261_ROOMS_MD5,
                       "note": "the stored K1K room (the K1KM arm's "
                               "bit-bind source)"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "anchor_lineage_cited": ANCHOR_LINEAGE,
        "hardbound": {
            "e261_md5": E261_MD5, "e264_md5": E264_MD5,
            "e264_verdict": E264_VERDICT,
            "e261_K1K_post_g0": E261_K1K_POST,
            "e261_K1K_root_g0": E261_K1K_ROOT,
            "e264_K10K_post_g0": E264_K10K_POST,
            "e264_K10K_root_g0": E264_K10K_ROOT,
            "kept_frac_curve": {str(k): E264_CURVE_KEPT[k]
                                for k in sorted(E264_CURVE_KEPT)},
            "e261_rooms_md5": E261_ROOMS_MD5},
        "pass": bool(
            md5of(E261_METRICS) == E261_MD5
            and md5of(E264_METRICS) == E264_MD5
            and abs(e261_k1k_post - E261_K1K_POST) < 1e-12
            and abs(e261_k1k_root - E261_K1K_ROOT) < 1e-12
            and abs(e261_k1k_kept - KEPT_1K_STORED) < 1e-12
            and abs(post_curve_file[1000] - E261_K1K_POST) < 1e-12
            and abs(post_curve_file[10000] - E264_K10K_POST) < 1e-12
            and abs(kept_curve_file[1000] - KEPT_1K_STORED) < 1e-12
            and abs(kept_curve_file[10000] - KEPT_10K_STORED) < 1e-12
            and e264m["adjudication"]["verdict"] == E264_VERDICT
            and e264_free_pass
            and md5of(CKPT_DIR / ROOMS261_CK) == E261_ROOMS_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e261 (md5 {E261_MD5[:8]}) K1K post "
        f"{e261_k1k_post:.6f} kept {e261_k1k_kept:.6f}; e264 (md5 "
        f"{E264_MD5[:8]}) {E264_VERDICT}, K10K post "
        f"{post_curve_file[10000]:.6f} kept {kept_curve_file[10000]:.6f}; "
        f"the committed kept_frac_curve bound")
    write_partial("P0b parents hard-bound (the cliff record + the kept curve)")

    # ---- G_DOSE_ARITH (the compensation arithmetic, exact) ---------------
    lr_scale_rt = kept_curve_file[10000] / kept_curve_file[1000]
    G_DOSE_ARITH = {
        "form": "the compensation factor re-derived at runtime from the "
                "md5-bound committed kept_frac_curve and asserted == the "
                "frozen literal; the compensated install's lr(s) = 1e-3 x "
                "LR_SCALE x cosine_lr(s-1, 1000) via the disclosed "
                "e043_install.LR module rebind (restored in a finally "
                "block); the cons reads G1.FT_LR and is untouched",
        "stored_kept_1k": kept_curve_file[1000],
        "stored_kept_10k": kept_curve_file[10000],
        "lr_scale_runtime": lr_scale_rt,
        "lr_scale_frozen": LR_SCALE_FROZEN,
        "lr_base": LR_BASE,
        "lr_compensated": LR_BASE * LR_SCALE_FROZEN,
        "formula": "LR_SCALE = stored_kept(10k)/stored_kept(1k); "
                   "lr_K1KM(s) = lr_base x LR_SCALE x cosine_lr(s-1, 1000)",
        "pass": bool(abs(lr_scale_rt - LR_SCALE_FROZEN) < 1e-12
                     and abs(kept_curve_file[1000] - KEPT_1K_STORED) < 1e-15
                     and abs(kept_curve_file[10000] - KEPT_10K_STORED)
                     < 1e-15
                     and abs(LR_BASE - 1e-3) < 1e-15),
    }
    assert G_DOSE_ARITH["pass"], f"dose arithmetic gate FAILED: {G_DOSE_ARITH}"
    metrics["gates"]["G_DOSE_ARITH"] = G_DOSE_ARITH
    log(f"P0c G_DOSE_ARITH: LR_SCALE = {kept_curve_file[10000]!r}/"
        f"{kept_curve_file[1000]!r} = {lr_scale_rt:.12f} (frozen "
        f"{LR_SCALE_FROZEN:.12f}); install lr {LR_BASE} -> "
        f"{LR_BASE * LR_SCALE_FROZEN:.6f} for K1KM only: PASS")
    write_partial("P0c G_DOSE_ARITH PASSED (the compensation exact)")

    # ---- G-BASE ----------------------------------------------------------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, CE_R {base_ce_r:.4f}): PASS")
    write_partial("P0d G-BASE PASSED")

    # ================= P1: THE ROOMS (v-map + span + cert + bit-bind) ====
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    root_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "battery_read_measured": root_read,
        "battery_read_committed": E261.G1C_ROOT_GM12,
        "abs_diff": abs(root_read - E261.G1C_ROOT_GM12), "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12) < G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"P1 G_ROOT: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {E261.G1C_ROOT_GM12:.10f}: PASS")
    write_partial("P1 G_ROOT PASSED")
    del root_net

    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))   # the params basis

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "meta_k": vmap_art.get("meta", {}).get("k"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N
                     and (SMOKE or int(vmap_art["meta"]["k"])
                          == E261.E258_K_HARD)),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND

    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND

    # ---- THE FOUR ROOMS BUILT + CERTIFIED (the machinery's own gates) ----
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the four rooms certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the DCT "
                "roundtrip identity; each room's IDEMPOTENCY and kept^2 "
                "rank probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N); each room's span-overlap (expect "
                "~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N) per room"},
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    for nm, r in cert["per_rung"].items():
        log(f"  room {nm}: k {r['k']} (seeds {r['seeds']}) idem "
            f"{r['idempotency_max']:.1e} kept2 {r['kept2_mean']:.6f} vs "
            f"{r['kept2_expect']:.6f} (bar {r['kept2_bar_10sig']:.1e}) "
            f"span-ovl {r['span_overlap_mean']:.4f} "
            f"(expect ~{r['span_overlap_expect']:.4f})")

    # ---- G_ROOM1K: the K1KM arm's room bit-identical to e261's committed
    # K1K room (the SHARED room — the only shared one in this cell)
    rooms261 = torch.load(CKPT_DIR / ROOMS261_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    if not SMOKE:
        D261 = _to_np(rooms261["model"]["K1K"]["D_int8"]).astype(np.float64)
        S261 = _to_np(rooms261["model"]["K1K"]["S"])
        D_mine = rooms.rooms[K1KM_MODE].D
        S_mine = rooms.rooms[K1KM_MODE].S
        G_ROOM1K = {
            "form": "the K1KM arm's room == e261's committed K1K room (seeds "
                    "26111/26112 at k=1,000): the +-1 diagonal and the "
                    "index set bit-identical to e261_rooms.pt's stored K1K "
                    "D/S (exact equality) — the SHARED room; the "
                    "compensation is the arm's ONLY delta vs the committed "
                    "rung",
            "D_bit_equal": bool(np.array_equal(D_mine, D261)),
            "S_bit_equal": bool(np.array_equal(S_mine, S261)),
            "e261_rooms_md5": md5of(CKPT_DIR / ROOMS261_CK),
            "pass": bool(np.array_equal(D_mine, D261)
                         and np.array_equal(S_mine, S261)
                         and int(rooms261["model"]["K1K"]["k"]) == 1000
                         and list(rooms261["model"]["K1K"]["seeds"])
                         == [26111, 26112]),
        }
    else:
        G_ROOM1K = {
            "form": "SMOKE: the K1KM analog room shares the seed pair "
                    "(26111/26112) at smoke k — no committed record at this "
                    "k; the bit-bind is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
    del rooms261
    assert G_ROOM1K["pass"], f"K1K room bind failed: {G_ROOM1K}"
    metrics["gates"]["G_ROOM1K"] = G_ROOM1K
    log(f"P1 G_ROOM1K: the K1KM arm's room "
        f"{('bit-identical to e261_rooms.pt K1K (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e272_rooms",
        {RUNG_NAMES[k]: {"D_int8": rooms.rooms[RUNG_NAMES[k]].D
                         .astype(np.int8),
                         "S": rooms.rooms[RUNG_NAMES[k]].S,
                         "k": k, "seeds": [sd, ss]}
         for k, sd, ss in LADDER},
        {"desc": "e272's four rooms (the flat basis is net.parameters() "
                 "order): K1KM = e261's committed K1K room VERBATIM "
                 "(bit-gated); K2K/K5K/K10KR = fresh SRCT rooms (the "
                 "registered seeds; K10KR is the room-lottery probe at the "
                 "cliff rung)",
         "ladder": [k for k, _, _ in LADDER], "n": N,
         "span_rank": rooms.r_span,
         "cert": {nm: {kk: vv for kk, vv in r.items()
                       if not isinstance(vv, list)}
                  for nm, r in cert["per_rung"].items()}})
    metrics["rooms"] = {
        "ladder": {"rule": "four rooms: K1KM (k=1000, e261's committed K1K "
                           "seeds — SHARED, bit-gated), K2K (k=2000 fresh), "
                           "K5K (k=5000 fresh), K10KR (k=10000 FRESH — the "
                           "room-lottery probe vs the committed rung's room "
                           "26113/26114)",
                   "rungs": [{"k": k, "name": RUNG_NAMES[k],
                              "seeds": [sd, ss], "k_fraction_of_N": k / N}
                             for k, sd, ss in LADDER]},
        "seeds": {RUNG_NAMES[k]: [sd, ss] for k, sd, ss in LADDER},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE FOUR ROOMS: {' + '.join(f'{RUNG_NAMES[k]}(k={k})' for k, _, _ in LADDER)}"
        f": BUILT + CERTIFIED + (K1KM) BIT-BOUND")
    write_partial("P1 the four rooms built (parents bound + v-map loaded + "
                  "span loaded + certification + the K1K bit-bind)")

    # ================= P2-P5: THE ARMS (serial; K10KR first) ============
    arms_rec: dict = {}
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None  # noqa: E731

    def run_arm(arm: str) -> None:
        log("=" * 78)
        k_arm = rooms.room_k[arm]
        comp = (arm == DOSE_ARM)
        log(f"ARM-{arm} — k={k_arm} room (seeds "
            f"{rooms.rooms[arm].seed_d}/{rooms.rooms[arm].seed_s})"
            + (" + the INSTALL lr COMPENSATED x "
               f"{LR_SCALE_FROZEN:.10f} (kept(10k)/kept(1k); the cons "
               "untouched)" if comp else " at the family dose (lr 1e-3)"))
        if comp:
            # ---- THE ONE-KNOB COMPENSATION (the disclosed module rebind) --
            E43.LR = LR_BASE * LR_SCALE_FROZEN
            log(f"[dose] E43.LR rebinned {LR_BASE} -> {E43.LR} "
                f"(x{LR_SCALE_FROZEN:.10f}) for {arm}'s install ONLY")
        try:
            inst = E261.chunked_install(
                f"{arm}-inst", arm, G1.evl_load(base_sd), rooms,
                inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
                r_eval_xy, zid, inst_resume_path(arm), dev)
        finally:
            if comp:
                E43.LR = LR_BASE
                log(f"[dose] E43.LR restored {E43.LR}")
        sd_install = inst["sd"]
        inst_net = G1.evl_load(sd_install)
        inst_cells = {"gm12": G1.battery_cell(inst_net, gm12_ids,
                                              zid)["mean_pz"],
                      "g0": G1.battery_cell(inst_net, g0_ids, zid)["mean_pz"],
                      "gp12": G1.battery_cell(inst_net, bat_ids[12],
                                              zid)["mean_pz"],
                      "ce_r": G1.ce_fixed_cpu(inst_net, *r_eval_xy)}
        d_inst = flat_params_cpu(inst_net) - base_flat
        load_inst = rooms.displacement_loads(d_inst, arm)
        # the end-to-end in-room write size (the measured in-room dose)
        d64 = d_inst.double().numpy().astype(np.float64)
        p_d = rooms.proj_of(arm, d64)
        in_room_norm = float(np.linalg.norm(p_d))
        del inst_net
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        led_vpre = [v["v_excess_pre"] for v in inst["ledger"].values()]
        led_vpost = [v["v_excess_post"] for v in inst["ledger"].values()]
        led_span = [v["in_span_frac"] for v in inst["ledger"].values()]
        led_aspan = [v["applied_in_span_frac"]
                     for v in inst["ledger"].values()]
        # the applied-dose proxy: lr_s x ||g'|| at every ledgered step
        lr_mult = LR_SCALE_FROZEN if comp else 1.0
        dose_proxy = [LR_BASE * lr_mult * cosine_lr(s - 1, E261.INST_TOTAL)
                      * v["gpn"] for s, v in inst["ledger"].items()]
        arms_rec[arm] = {
            "desc": (f"k={k_arm} SRCT room (seeds "
                     f"{rooms.rooms[arm].seed_d}/{rooms.rooms[arm].seed_s})"
                     + ("; the INSTALL lr compensated x "
                        f"{LR_SCALE_FROZEN:.10f} = kept(10k)/kept(1k) — the "
                        "dose-at-matched-rank probe" if comp else
                        (" — the room-lottery probe at the cliff rung"
                         if arm == "K10KR" else " — the edge rung"))),
            "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "ledger_v_excess_pre_median": med(led_vpre),
                "ledger_v_excess_post_median": med(led_vpost),
                "ledger_in_span_frac_median": med(led_span),
                "ledger_applied_in_span_frac_median": med(led_aspan),
                "chunk_table": inst["chunk_table"], "steps": E261.INST_STEPS,
                "post_cells": inst_cells,
                "displacement_loads": load_inst,
                "in_room_disp_norm": in_room_norm,
                "disp_norm": float(np.linalg.norm(d64)),
                "lr_scale_applied": lr_mult,
                "applied_dose_proxy_median": med(dose_proxy),
                "resumed_final": bool(inst.get("resumed_final", False)),
            },
        }
        log(f"ARM-{arm} install done: post g0 {inst_cells['g0']:.6f} g-12 "
            f"{inst_cells['gm12']:.6f} CE_R {inst_cells['ce_r']:.4f} | d "
            f"v-excess {load_inst['v_excess']:.2f} cos-to-span "
            f"{load_inst['cos_to_span']:.4f} in-own-room "
            f"{load_inst['in_own_room']:.4f} ||P d|| {in_room_norm:.5f} "
            f"||d|| {arms_rec[arm]['install']['disp_norm']:.5f} | ledger "
            f"kept med {med(led_kept):.4f} v-exc pre {med(led_vpre):.2f} "
            f"post {med(led_vpost):.2f} in-span {med(led_span):.3f}->applied "
            f"{med(led_aspan):.3f} | lr x ||g'|| med "
            f"{med(dose_proxy):.6f}")
        write_partial(f"P2 ARM-{arm} install + post dial + measured loads")

        if not inst.get("resumed_final", False):
            burst_cooldown(f"{arm} inst->cons")
        cons = E261.chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            cons_resume_path(arm), dev)
        theta0 = cons["sd"]
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"]
                 for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - base_flat
        load_root = rooms.displacement_loads(d_root, arm)
        root_ck = save_ckpt(
            f"e272_{arm}_root", theta0,
            {"desc": f"e272 ARM-{arm} root: e001 + Dmix s{E261.INST_STEPS} "
                     f"(gen {E261.FRESH_GEN}; k={k_arm} room"
                     + (f", lr x{LR_SCALE_FROZEN:.6f}" if comp else "")
                     + f") + e113 cons s{E261.CONS_STEPS} (seed "
                     f"{E261.CONS_SEED} HELD, natural)",
             "arm": arm, "k": k_arm, "install_seed": E261.FRESH_GEN,
             "lr_scale": lr_mult,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e272_rooms.pt"})
        arms_rec[arm]["consolidation"] = {"traj": cons["traj"],
                                          "chunk_table": cons["chunk_table"]}
        arms_rec[arm]["root"] = {"cells": cells,
                                 "gm12": cells["g-12"],
                                 "g0": cells["g+0"],
                                 "displacement_loads": load_root,
                                 "checkpoint": root_ck}
        landed = (E261.G1C_ROOT_G0 * (1 - MATCH_BAND)
                  <= cells["g+0"] <= E261.G1C_ROOT_G0 * (1 + MATCH_BAND))
        arms_rec[arm]["root"]["landed"] = bool(landed)
        arms_rec[arm]["root"]["expresses"] = bool(
            inst_cells["g0"] >= G0_ZERO_FLOOR)
        log(f"ARM-{arm} ROOT (carried, the rehearsal lane): g0 "
            f"{cells['g+0']:.4f} (band context "
            f"[{E261.G1C_ROOT_G0 * (1 - MATCH_BAND):.4f}, "
            f"{E261.G1C_ROOT_G0 * (1 + MATCH_BAND):.4f}]) landed={landed} "
            f"(WRITE read: post-install g0 {inst_cells['g0']:.6f} "
            f"expresses={inst_cells['g0'] >= G0_ZERO_FLOOR}) | g-12 "
            f"{cells['g-12']:.4f} (lottery) | held30 "
            f"{cells['held30_gm12']:.4f} CE_R {cells['ce_r']:.4f} | d-root "
            f"v-excess {load_root['v_excess']:.2f} in-own-room "
            f"{load_root['in_own_room']:.4f}")
        del root_net_arm
        metrics["arms"] = arms_rec
        write_partial(f"P3/P4 ARM-{arm} root built + landing read (carried)")

    for i, arm in enumerate(ARMS):
        run_arm(arm)
        if arm != ARMS[-1]:
            burst_cooldown(f"{arm} -> next arm")

    # ---- THE DOSE CONTROL LEDGER (measured, never nominal) ---------------
    k1km, k10kr = arms_rec[DOSE_ARM], arms_rec["K10KR"]
    dose_control = {
        "note": "the compensation's exact arithmetic + its EMPIRICAL match, "
                "measured never nominal: (i) the per-step applied-dose "
                "proxy lr_s x ||g'|| (median over the install ledger); (ii) "
                "the end-to-end in-room displacement norm ||P_room(theta_s400"
                " - theta_0)|| (the in-room write size the install actually "
                "produced); DISCLOSED first-order caveats: Adam's "
                "per-coordinate normalizer partially cancels uniform "
                "gradient scales (the proxy is the gradient-norm channel; "
                "the displacement norms are the state channel), and the "
                "rebind scales AdamW's decoupled weight decay with the same "
                "lr",
        "stored_kept_1k": KEPT_1K_STORED,
        "stored_kept_10k": KEPT_10K_STORED,
        "lr_scale": LR_SCALE_FROZEN,
        "lr_base": LR_BASE,
        "lr_compensated": LR_BASE * LR_SCALE_FROZEN,
        "formula": "LR_SCALE = stored_kept(10k)/stored_kept(1k) = "
                   f"{KEPT_10K_STORED!r}/{KEPT_1K_STORED!r} = "
                   f"{LR_SCALE_FROZEN!r}; lr_K1KM(s) = {LR_BASE} x LR_SCALE "
                   "x cosine_lr(s-1, 1000)",
        "applied_dose_proxy": {
            "K1KM_lr_x_gp_median": k1km["install"]["applied_dose_proxy_median"],
            "K10KR_lr_x_gp_median": k10kr["install"]["applied_dose_proxy_median"],
            "ratio_K1KM_over_K10KR": (
                k1km["install"]["applied_dose_proxy_median"]
                / max(k10kr["install"]["applied_dose_proxy_median"], 1e-30)),
            "expected_first_order": 1.0,
        },
        "end_to_end_in_room": {
            "K1KM_disp_norm": k1km["install"]["disp_norm"],
            "K1KM_in_room_norm": k1km["install"]["in_room_disp_norm"],
            "K10KR_disp_norm": k10kr["install"]["disp_norm"],
            "K10KR_in_room_norm": k10kr["install"]["in_room_disp_norm"],
            "ratio_in_room_norm_K1KM_over_K10KR": (
                k1km["install"]["in_room_disp_norm"]
                / max(k10kr["install"]["in_room_disp_norm"], 1e-30)),
        },
        "kept_ledger": {a: arms_rec[a]["install"]["ledger_kept_frac_median"]
                        for a in ARMS},
    }
    metrics["dose_control"] = dose_control
    log(f"DOSE CONTROL: lr x ||g'|| med — K1KM "
        f"{k1km['install']['applied_dose_proxy_median']:.6f} vs K10KR "
        f"{k10kr['install']['applied_dose_proxy_median']:.6f} (ratio "
        f"{dose_control['applied_dose_proxy']['ratio_K1KM_over_K10KR']:.3f},"
        f" first-order 1.0); ||P d|| — K1KM "
        f"{k1km['install']['in_room_disp_norm']:.5f} vs K10KR "
        f"{k10kr['install']['in_room_disp_norm']:.5f}")
    write_partial("P5 the dose-control ledger (measured)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    post = {a: arms_rec[a]["install"]["post_cells"]["g0"] for a in ARMS}
    root = {a: arms_rec[a]["root"]["g0"] for a in ARMS}
    kept = {a: arms_rec[a]["install"]["ledger_kept_frac_median"]
            for a in ARMS}

    # ---- THE EDGE LADDER (the committed cites + this cell's rungs)
    edge_ladder = [
        {"k": 1000, "post": E261_K1K_POST, "source": "e261 committed (cited)"},
        {"k": 2000, "post": post["K2K"], "source": "this cell (fresh room)"},
        {"k": 5000, "post": post["K5K"], "source": "this cell (fresh room)"},
        {"k": 10000, "post": E264_K10K_POST,
         "source": "e264 committed (cited)"},
    ]
    pairs = []
    for i in range(len(edge_ladder) - 1):
        lo, hi = edge_ladder[i], edge_ladder[i + 1]
        ratio = hi["post"] / max(lo["post"], RATIO_DEN_FLOOR)
        pairs.append({"from_k": lo["k"], "to_k": hi["k"],
                      "post_lo": lo["post"], "post_hi": hi["post"],
                      "ratio": ratio, "dead_side": bool(lo["post"] < DEAD_BAR),
                      "fires": bool(ratio >= JUMP_BAR
                                    and lo["post"] < DEAD_BAR)})
    firing = [p for p in pairs if p["fires"]]
    edge_located = bool(firing)
    edge_bracket = (max(firing, key=lambda p: p["from_k"])["from_k"],
                    max(firing, key=lambda p: p["from_k"])["to_k"]) if firing \
        else None
    floor_cross = next((r["k"] for r in edge_ladder
                        if r["post"] >= EXPRESS_FLOOR), None)

    # ---- the sensitivity ladder (the replicate substituted at 10k)
    sens_ladder = [dict(r) for r in edge_ladder]
    sens_ladder[-1] = {"k": 10000, "post": post["K10KR"],
                       "source": "this cell's replicate (SENSITIVITY)"}
    sens_pairs = []
    for i in range(len(sens_ladder) - 1):
        lo, hi = sens_ladder[i], sens_ladder[i + 1]
        ratio = hi["post"] / max(lo["post"], RATIO_DEN_FLOOR)
        sens_pairs.append({"from_k": lo["k"], "to_k": hi["k"],
                           "ratio": ratio,
                           "fires": bool(ratio >= JUMP_BAR
                                         and lo["post"] < DEAD_BAR)})
    sens_edge_located = any(p["fires"] for p in sens_pairs)

    # ---- the bars
    dose_probe = post[DOSE_ARM]
    dose_dead = bool(dose_probe < DEAD_BAR)
    dose_expresses = bool(dose_probe >= EXPRESS_FLOOR)
    rank_fires = bool(dose_dead and edge_located)
    dose_fires = dose_expresses
    rep_post = post["K10KR"]
    lottery_fires = bool(not (ROOM_WINDOW[0] <= rep_post <= ROOM_WINDOW[1]))
    rep_abs_diff = abs(rep_post - E264_K10K_POST)

    hard = {k: v for k, v in metrics["gates"].items()}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    else:
        rider = ""
        if lottery_fires:
            rider = (" [RIDER — ROOM-LOTTERY co-fired: the 10k replicate's "
                     f"post g0 {rep_post:.6f} leaves [{ROOM_WINDOW[0]}, "
                     f"{ROOM_WINDOW[1]}] (the committed rung was "
                     f"{E264_K10K_POST:.6f}) — the cliff pair untrusted; "
                     "demand replicates before any further capacity claim]")
        if rank_fires:
            verdict = "RANK-WRITES-THE-CURVE"
            edge_txt = ", ".join(f"{p['from_k']}->{p['to_k']} "
                                 f"({p['ratio']:.1f}x, dead side "
                                 f"{p['post_lo']:.6f})" for p in firing)
            clause = (f"the kept-matched 1k arm STAYS DEAD (post g0 "
                      f"{dose_probe:.6f} < {DEAD_BAR}) AND the edge lands "
                      f"inside (1k,10k] — firing pairs: {edge_txt}; the "
                      f"tightest bracket {edge_bracket[0]}->{edge_bracket[1]}"
                      + (f"; the floor {EXPRESS_FLOOR} first crossed at "
                         f"k={floor_cross}" if floor_cross else
                         "; the floor 0.05 not yet crossed inside the "
                         "bracket (a dead-side jump located the edge)")
                      + f" — dose acquitted at matched rank (the applied "
                      f"in-room dose proxy matched at "
                      f"{dose_control['applied_dose_proxy']['ratio_K1KM_over_K10KR']:.2f}"
                      " first-order); the capacity story stands with a "
                      "located edge. P-272a (T249) CONFIRMED: dose does not "
                      "buy expression."
                      + rider
                      + ("" if sens_edge_located == edge_located else
                         " [SENSITIVITY: substituting the replicate's post "
                         "at 10k flips the edge read — disclosed]"))
        elif dose_fires:
            verdict = "DOSE-WRITES-THE-CURVE"
            clause = (f"the kept-matched 1k arm EXPRESSES (post g0 "
                      f"{dose_probe:.6f} >= {EXPRESS_FLOOR}) — the 609x "
                      "jump was the 3.7x kept-dose change in disguise; the "
                      "capacity number falls to a dose artifact; the "
                      "critic's embarrassment scenario, caught in time. "
                      "P-272a (T249) REFUTED."
                      + rider)
        elif lottery_fires:
            verdict = "ROOM-LOTTERY"
            clause = (f"the 10k replicate's post g0 {rep_post:.6f} leaves "
                      f"[{ROOM_WINDOW[0]}, {ROOM_WINDOW[1]}] (the committed "
                      f"rung was {E264_K10K_POST:.6f}; |d| {rep_abs_diff:.4f}"
                      f" — the room lottery at the cliff rung is LARGE) — "
                      "the cliff pair untrusted; demand replicates before "
                      "any further capacity claim; the rank/dose probe "
                      f"read verbatim: K1KM post g0 {dose_probe:.6f}"
                      + (" (dead)" if dose_dead else
                         " (below the expression floor)"
                         if dose_probe < EXPRESS_FLOOR else " (expresses)"))
        else:
            why = []
            if DEAD_BAR <= dose_probe < EXPRESS_FLOOR:
                why.append(f"PARTIAL EXPRESSION AT MATCHED DOSE: the "
                           f"kept-matched 1k arm's post g0 {dose_probe:.6f} "
                           f"sits in the gap [{DEAD_BAR}, {EXPRESS_FLOOR}) — "
                           "above the dead bar, below the expression floor "
                           "(P-272a's 'below the expression floor' holds at "
                           "0.05, but the frozen letter's clauses govern)")
            elif dose_dead and not edge_located:
                why.append(f"the probe stayed dead (post g0 {dose_probe:.6f}"
                           f" < {DEAD_BAR}) but NO adjacent pair jumps "
                           f">={JUMP_BAR:.0f}x with a dead side inside "
                           "(1k,10k] — the edge not located at this "
                           "resolution")
            else:
                why.append("between the bars")
            verdict = "MIXED"
            clause = ("; ".join(why)
                      + " — the trajectories verbatim, both readouts, no "
                      "inflation. P-272a (T249): "
                      + ("confirmed at the expression floor"
                         if dose_probe < EXPRESS_FLOOR else "refuted")
                      + rider)

    log("=" * 78)
    log(f"E272 VERDICT: {verdict}")
    log("  THE FOUR ARMS (post g0 -> root g0 [carried] | kept | room):")
    for a in ARMS:
        log(f"    {a:6s} (k={rooms.room_k[a]:6d}): post g0 {post[a]:.6f} -> "
            f"root g0 {root[a]:.4f} | kept {kept[a]:.4f} | lr x"
            f"{arms_rec[a]['install']['lr_scale_applied']:.4f}")
    log(f"  the edge ladder: " + " -> ".join(
        f"{r['k']}:{r['post']:.6f}" for r in edge_ladder))
    log(f"  firing pairs: {[(p['from_k'], p['to_k'], round(p['ratio'], 1)) for p in firing]}"
        f"; bracket {edge_bracket}; floor crossing k={floor_cross}")
    log(f"  the dose probe: K1KM post g0 {dose_probe:.6f} "
        f"(dead bar {DEAD_BAR}, expression floor {EXPRESS_FLOOR})")
    log(f"  the room lottery: K10KR post g0 {rep_post:.6f} vs committed "
        f"{E264_K10K_POST:.6f} (|d| {rep_abs_diff:.4f}; window "
        f"{ROOM_WINDOW}) -> {'FIRES' if lottery_fires else 'holds'}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "hard-gate failure (TEXTURE) -> [non-halting "
                           "texture disclosures] -> RANK-WRITES-THE-CURVE ->"
                           " DOSE-WRITES-THE-CURVE -> ROOM-LOTTERY -> MIXED "
                           "(frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "post_g0": post, "root_g0_carried": root, "kept_frac": kept,
            "post_gm12": {a: arms_rec[a]["install"]["post_cells"]["gm12"]
                          for a in ARMS},
            "root_gm12_lottery": {a: arms_rec[a]["root"]["gm12"]
                                  for a in ARMS},
            "edge_ladder": edge_ladder,
            "adjacent_pairs": pairs,
            "firing_pairs": firing,
            "edge_located": edge_located,
            "edge_bracket": list(edge_bracket) if edge_bracket else None,
            "floor_cross_k": floor_cross,
            "sensitivity_ladder_replicate_at_10k": {
                "ladder": sens_ladder, "pairs": sens_pairs,
                "edge_located": sens_edge_located,
                "agrees_with_primary": bool(sens_edge_located
                                            == edge_located)},
            "dose_probe": {"arm": DOSE_ARM, "post_g0": dose_probe,
                           "dead": dose_dead,
                           "expresses": dose_expresses,
                           "dead_bar": DEAD_BAR,
                           "expression_floor": EXPRESS_FLOOR},
            "room_lottery": {"arm": "K10KR", "post_g0": rep_post,
                             "committed_post_g0": E264_K10K_POST,
                             "abs_diff": rep_abs_diff,
                             "window": list(ROOM_WINDOW),
                             "fires": lottery_fires},
            "root_read_status": "CARRIED, NEVER ADJUDICATED — T246's "
                                "rehearsal-lane caveat: the cons re-teaches "
                                "the fact; the root g0 measures the "
                                "rehearsal lane, not the write (e268-e271's "
                                "dead writes landed 0.70-0.81)",
            "cited_rungs": {"k1k_e261": {"post_g0": E261_K1K_POST,
                                          "root_g0": E261_K1K_ROOT},
                            "k10k_e264": {"post_g0": E264_K10K_POST,
                                          "root_g0": E264_K10K_ROOT}},
            "edge_expectation_context": REGISTERED[
                "predictions_cited"]["edge_expectation"],
        },
        "RANK_WRITES_THE_CURVE": rank_fires,
        "DOSE_WRITES_THE_CURVE": dose_fires,
        "ROOM_LOTTERY": lottery_fires,
        "MIXED": bool(gates_pass and not rank_fires and not dose_fires
                      and not lottery_fires),
        "P272a_T249": ("confirmed (the kept-matched arm below the "
                       "expression floor)" if dose_probe < EXPRESS_FLOOR
                       else "refuted (the kept-matched arm expresses)"),
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars adjudicated")

    # ================= P9: honesty + provenance ==========================
    metrics["honesty"] = {
        "intervention_not_logits": ("all four arms share bit-identical "
            "install streams (one generator, seed 24314, one draw order), "
            "the same fresh fact-free base, the same cons (seed 10901 "
            "HELD, natural); the K2K/K5K/K10KR arms differ from the "
            "committed rungs ONLY in the room (rank k + the draw); the "
            "K1KM arm differs from the committed K1K rung ONLY in the "
            "install lr (x the md5-bound kept quotient — its room is "
            "BIT-IDENTICAL, gated) — fate differences are rank- or dose-"
            "caused or nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
            "g-series standing caveat); K10KR exists to price the room "
            "lottery at the cliff rung; the sensitivity ladder discloses "
            "the cite choice's reach"),
        "dose_measured_not_nominal": ("the compensation's empirical match "
            "is MEASURED (the lr x ||g'|| proxies + the end-to-end in-room "
            "displacement norms in dose_control), with the Adam "
            "scale-invariance and weight-decay-coupling caveats disclosed "
            "at birth — the bars read the fate, the dose_control ledger "
            "carries the mechanism's honesty"),
        "the_landing_read_carried": ("the bars adjudicate the WRITE read "
            "(post g0) only; the root g0 is context under T246's "
            "rehearsal-lane caveat; root g-12 never gated"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all four branches and "
            "the trajectories are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e261_rooms": f"runs/checkpoints/{ROOMS261_CK}",
            "rooms": rooms_ck,
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS},
            "arm_inst_resumes": {a: str(inst_resume_path(a).relative_to(
                E43.REPO)).replace("\\", "/") for a in ARMS},
        },
        "machinery": {
            "install_cons": "e261's chunked drivers VERBATIM BY IMPORT "
                            "(E43.exposure Dmix / e113 consolidate), the "
                            "dense-room hook, the thermal envelope; the "
                            "K1KM install under the disclosed E43.LR "
                            "module rebind (restored in a finally block)",
            "hook": "e237's pre-Adam projection (backward -> clip 1.0 -> "
                    "project CPU fp64 SRCT -> write fp32 -> step; norm "
                    "not rescaled) on every install step",
            "rooms": "one SRCT room per arm (K1KM = e261's committed K1K "
                     "room VERBATIM, bit-gated; K2K/K5K/K10KR fresh at "
                     "the registered seeds 27211-27216)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": {
            "burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
            "per_step_polls": True,
            "early_end_margin_c": E261.TEMP_EARLY_END,
            "hard_line_c": E261.TEMP_HARD,
            "max_temp_seen_c": max((r["temp"] for r in thermal_log),
                                   default=None),
            "violations_ge_84c": sum(1 for r in thermal_log
                                     if r["temp"] >= E261.TEMP_HARD),
            "n_polls": len(thermal_log),
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }

    # ================= P10: figures ======================================
    make_main_plot(RD, arms_rec, post, root, kept, pairs, firing,
                   edge_bracket, floor_cross, dose_probe, rep_post,
                   lottery_fires, dose_control, verdict, clause)
    make_instrument_plot(RD, arms_rec, cert, thermal_log, dose_control)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e272_capacity_repair.png"),
                          str(RD / "e272_instrument.png"),
                          str(RD / "REPORT.md")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_main_plot(rd, arms_rec, post, root, kept, pairs, firing,
                   edge_bracket, floor_cross, dose_probe, rep_post,
                   lottery_fires, dose_control, verdict, clause):
    """THE CELL'S HEADLINE FIGURE: the rung curve (the committed ladder +
    the new rungs + the kept-matched arm + the replicate overlaid), the
    dose control, the room lottery, the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    N = GB.G1B_PARAMS

    # (0,0) THE EXPRESSION CURVE (post g0 vs k)
    ax = axes[0, 0]
    cks = sorted(E264_CURVE_POST)
    ax.plot(cks, [E264_CURVE_POST[k] for k in cks], "o-", ms=5, lw=1.4,
            color="dimgray", alpha=0.8,
            label="the committed 5-rung ladder (e261/e264, cited)")
    ax.plot([2000], [post["K2K"]], "o", ms=11, color="#1a6faf",
            label=f"K2K (this cell): {post['K2K']:.6f}")
    ax.plot([5000], [post["K5K"]], "o", ms=11, color="#1a6faf",
            label=f"K5K (this cell): {post['K5K']:.6f}")
    ax.plot([1000], [dose_probe], "D", ms=11, color="crimson",
            label=f"K1KM kept-matched (lr x{LR_SCALE_FROZEN:.3f}): "
                  f"{dose_probe:.6f}")
    ax.plot([10000], [rep_post], "s", ms=11, color="#e67e22",
            label=f"K10KR fresh-room replicate: {rep_post:.6f} "
                  f"(committed {E264_K10K_POST:.4f})")
    for r in ({"k": 2000, "v": post["K2K"]}, {"k": 5000, "v": post["K5K"]}):
        ax.annotate(f"{r['v']:.5f}", (r["k"], r["v"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#1a6faf")
    ax.axhline(EXPRESS_FLOOR, ls=":", lw=1.4, color="crimson",
               label=f"the expression floor ({EXPRESS_FLOOR})")
    ax.axhline(DEAD_BAR, ls=":", lw=1.2, color="darkred", alpha=0.7,
               label=f"the dead bar ({DEAD_BAR})")
    if edge_bracket:
        ax.axvspan(edge_bracket[0], edge_bracket[1], color="seagreen",
                   alpha=0.12, zorder=0)
        ax.annotate(f"EDGE BRACKET\n({edge_bracket[0]}, {edge_bracket[1]}]",
                    (math.sqrt(edge_bracket[0] * edge_bracket[1]), 0.9),
                    fontsize=8, color="seagreen", ha="center",
                    weight="bold")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("post g0 (the WRITE read)")
    ax.legend(fontsize=7.0, loc="center right")
    ax.grid(alpha=0.25, which="both")
    fires_txt = ", ".join(f"{p['from_k']}->{p['to_k']} {p['ratio']:.0f}x"
                          for p in firing) if firing else "NONE"
    ax.set_title(f"THE EXPRESSION CURVE — firing pairs (>= 10x, dead side "
                 f"< {DEAD_BAR}): {fires_txt}", fontsize=9.5)

    # (0,1) THE DOSE CONTROL (kept curve + the compensation)
    ax = axes[0, 1]
    kpts = [E264_CURVE_KEPT[k] for k in cks]
    ax.plot(cks, kpts, "o-", ms=5, lw=1.4, color="#8e44ad",
            label="kept ||g'||/||g|| (committed curve)")
    for a, kk in (("K2K", 2000), ("K5K", 5000)):
        ax.plot([kk], [kept[a]], "o", ms=10, color="#1a6faf",
                label=f"{a}: {kept[a]:.4f}")
    ax.plot([1000], [kept["K1KM"]], "D", ms=10, color="crimson",
            label=f"K1KM: {kept['K1KM']:.4f} (room kept UNCHANGED)")
    ax.plot([10000], [kept["K10KR"]], "s", ms=10, color="#e67e22",
            label=f"K10KR: {kept['K10KR']:.4f}")
    ax.annotate(f"LR_SCALE = {KEPT_10K_STORED:.5f}/{KEPT_1K_STORED:.5f}\n"
                f"= {LR_SCALE_FROZEN:.4f} (K1KM's install lr)",
                (0.03, 0.72), xycoords="axes fraction", fontsize=8.5,
                color="crimson")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("kept fraction (the delivered-dose covariate)",
                  color="#8e44ad")
    ax.legend(fontsize=7.0, loc="center right")
    ax.grid(alpha=0.25, which="both")
    adp = dose_control["applied_dose_proxy"]
    ax.set_title(f"THE DOSE CONTROL — lr x ||g'|| med: K1KM "
                 f"{adp['K1KM_lr_x_gp_median']:.5f} vs K10KR "
                 f"{adp['K10KR_lr_x_gp_median']:.5f} (ratio "
                 f"{adp['ratio_K1KM_over_K10KR']:.2f}, first-order 1.0)",
                 fontsize=9.0)

    # (1,0) THE ROOM LOTTERY (the 10k rung's reads)
    ax = axes[1, 0]
    ax.axhspan(ROOM_WINDOW[0], ROOM_WINDOW[1], color="#b8e0b8",
               alpha=0.45, zorder=0,
               label=f"the trust window [{ROOM_WINDOW[0]}, {ROOM_WINDOW[1]}]")
    vals = [E264_K10K_POST, rep_post]
    xs = [0, 1]
    ax.bar([xs[0]], [vals[0]], width=0.5, color="dimgray", alpha=0.75)
    ax.bar([xs[1]], [vals[1]], width=0.5, color="#e67e22", alpha=0.9)
    ax.axhline(E264_K10K_POST, color="k", ls=":", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"committed K10K (room 26113/26114)\n"
                        f"{E264_K10K_POST:.6f}",
                        f"K10KR fresh room (27215/27216)\n"
                        f"{rep_post:.6f}"], fontsize=8.5)
    ax.set_ylabel("post g0 at k=10,000")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=7.2, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"THE ROOM LOTTERY at the cliff rung — |d| "
                 f"{abs(rep_post - E264_K10K_POST):.4f}: "
                 f"{'LEAVES the window (FIRES)' if lottery_fires else 'holds'}",
                 fontsize=9.5)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E272 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=94,
                            break_long_words=False)[:13]:
        ax.text(0.02, y, wd, fontsize=6.6, va="top", family="monospace")
        y -= 0.023
    y -= 0.012
    gates_txt = "  ".join(
        f"{g}={'PASS' if v.get('pass') else 'FAIL'}"
        for g, v in metrics["gates"].items() if isinstance(v, dict))
    for wd in textwrap.wrap("GATES: " + gates_txt, width=96)[:2]:
        ax.text(0.02, y, wd, fontsize=6.2, va="top", family="monospace")
        y -= 0.02
    fig.suptitle("E272 — THE CAPACITY REPAIR LADDER: rank or dose? the edge "
                 f"located; the room lottery priced -> {verdict}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "e272_capacity_repair.png", dpi=130)
    plt.close(fig)


def make_instrument_plot(rd, arms_rec, cert, thermal_log, dose_control):
    """THE INSTRUMENT PAGE: the rooms' certification, the kept ledgers, the
    measured per-arm loads, the thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    N = GB.G1B_PARAMS
    names = list(ARMS)

    # (0,0) THE ROOMS' CERTIFICATION
    ax = axes[0, 0]
    cert_names = [RUNG_NAMES[k] for k, _, _ in LADDER]
    idem = [cert["per_rung"][nm]["idempotency_max"] for nm in cert_names]
    ax.bar([f"{nm}\nk={cert['per_rung'][nm]['k']}" for nm in cert_names],
           [max(v, 1e-18) for v in idem], color="#1a6faf", alpha=0.85)
    for b, v in zip(ax.patches, idem):
        ax.annotate(f"{v:.0e}", (b.get_x() + b.get_width() / 2,
                                 max(v, 1e-18)), ha="center", va="bottom",
                    fontsize=6.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-2)
    ax.axhline(1e-8, ls=":", color="k", lw=1.0, label="the idem bar 1e-8")
    kt = " | ".join(
        f"{nm}: kept2 {cert['per_rung'][nm]['kept2_mean']:.2e} vs "
        f"{cert['per_rung'][nm]['kept2_expect']:.2e}"
        for nm in cert_names)
    ax.set_ylabel("max relative deviation (log)")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.set_title("THE FOUR ROOMS' CERTIFICATION (idem shown; " + kt + ")",
                 fontsize=7.4)

    # (0,1) THE KEPT-FRACTION LEDGERS
    ax = axes[0, 1]
    cols = {"K10KR": "#e67e22", "K2K": "#1a6faf", "K5K": "#2e86ab",
            "K1KM": "crimson"}
    for a in ARMS:
        led = arms_rec[a]["install"]["ledger"]
        xs_l = sorted(int(s) for s in led)
        get = lambda s: led[s] if s in led else led[str(s)]  # noqa: E731
        med_a = arms_rec[a]["install"]["ledger_kept_frac_median"]
        ax.plot(xs_l, [get(s)["kept_frac"] for s in xs_l], "o-", ms=3.0,
                lw=1.2, alpha=0.9, color=cols[a],
                label=f"{a} (med {med_a:.4f})")
    for k, _, _ in LADDER:
        ax.axhline(math.sqrt(k / N), ls=":", lw=0.8, color="gray",
                   alpha=0.5)
    ax.set_xlabel("install step")
    ax.set_ylabel("the kept-gradient fraction (dose honesty)")
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)
    ax.set_title("THE KEPT-FRACTION LEDGERS (dotted: each room's "
                 "sqrt(k/N) expectation)", fontsize=9.5)

    # (1,0) THE MEASURED PER-ARM LOADS + the in-room write sizes
    ax = axes[1, 0]
    w = 0.22
    xs = np.arange(len(names))
    ior_post = [arms_rec[a]["install"]["displacement_loads"]["in_own_room"]
                or 0 for a in names]
    ior_root = [arms_rec[a]["root"]["displacement_loads"]["in_own_room"]
                or 0 for a in names]
    irn = [arms_rec[a]["install"]["in_room_disp_norm"] for a in names]
    ax.bar(xs - w, ior_post, width=w, color="#1a6faf", alpha=0.85,
           label="post-install displacement in-own-room (frac)")
    ax.bar(xs, ior_root, width=w, color="#8e44ad", alpha=0.8,
           label="root displacement in-own-room (frac)")
    ax.bar(xs + w, [v / max(max(irn), 1e-9) for v in irn], width=w,
           color="#c0392b", alpha=0.65,
           label="||P_room d_post|| (normalized to the max)")
    for x, v in zip(xs + w, irn):
        ax.annotate(f"{v:.4f}", (x, v / max(max(irn), 1e-9)),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8.5, rotation=12)
    ax.set_ylabel("measured load / fraction")
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE MEASURED PER-ARM LOADS (never nominal; the annotated "
                 "numbers are the raw in-room displacement norms)",
                 fontsize=9.0)

    # (1,1) THE THERMAL ENVELOPE
    ax = axes[1, 1]
    if thermal_log:
        ax.plot([r["t"] for r in thermal_log],
                [r["temp"] for r in thermal_log],
                "-", lw=0.8, color="dimgray", alpha=0.7)
    ax.axhline(E261.TEMP_EARLY_END, color="crimson", ls=":", lw=1.0,
               label=f"burst-end margin {E261.TEMP_EARLY_END:.0f}C")
    ax.axhline(E261.TEMP_HARD, color="crimson", ls="--", lw=1.2,
               label=f"never-past line {E261.TEMP_HARD:.0f}C")
    ax.set_xlabel("run seconds")
    ax.set_ylabel("GPU temp (C) per-step polls")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in thermal_log if r['temp'] >= E261.TEMP_HARD)}; "
                 f"{len(thermal_log)} polls)", fontsize=9.5)

    fig.suptitle("E272 — the instrument page: the rooms, the kept ledgers, "
                 "the loads, the thermal envelope", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "e272_instrument.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
