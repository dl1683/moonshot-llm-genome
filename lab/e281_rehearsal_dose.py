"""E281 — THE REHEARSAL DOSE-RESPONSE (the retrieval story's lynchpin).
Design dispatched 2026-10-05; this docstring carries the registered bars
VERBATIM, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (frozen; P-281a/b registered in W045 BEFORE this letter):
is the rehearsal lane's landing a property of the WRITE (delivery needs
a seed — the retrieval floor is a real second capacity number) or of the
CONS alone (it teaches from anything — the lane carries zero write
information)?

THE CURVE: landing read (root g0, the rehearsal-lane convention, CARRIED
never adjudicated as survival) vs the seed state's write — five points
on the width axis, the complement of e272's formation curve.

THE FIVE ARMS (same cons stream, same seed 10901, same protocol; each =
seed state + cons s300 + the landing read):
  (a) NO-INSTALL (the ZERO POINT — fresh root, no install at all): the
      cons-only floor. THE LYNCHPIN.
  (b) RANK-10 (a fresh rank-10 install, dead by construction — the
      family's 2.86e-5 floor precedent, e246's ALIGNED rank-10).
  (c) K1KM (the committed below-edge dead 1k write — e272's checkpoint,
      bit-verified; post g0 0.0012).
  (d) 10K-DEAD (the 10k concurrent dead write — e268's arm).
  (e) 100K-DEAD (the same, from e270's arm — root 0.8103's source).

REGISTERED BARS (frozen in the dispatch letter, VERBATIM in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping):
  - FLAT (cons-powered; P-281b): "every arm lands in the family's
    landing band (~0.65-0.82) INCLUDING the no-install zero point — the
    lane carries zero write information; 'delivery is free' collapses to
    'the cons teaches from anything'."
  - THRESHOLDED (P-281a's world): "the zero point lands < 0.45 AND the
    curve steps with seed width — delivery needs a seed; the retrieval
    floor is a second capacity number."
  - MASS-SCALED: "the zero point lands < 0.45 but the seeded arms'
    landings track write mass continuously (no step)."
  - MIXED: "anything else — the five points verbatim, no inflation."

REGISTERED PREDICTIONS on record (CITED, not re-registered — W045,
2026-10-05 22:39Z, before this letter):
  - P-281a: "the cons-only floor lands BELOW 0.45 — the write's residue
    carries the landing; 'delivery is free' survives as DELIVERY NEEDS A
    SEED; the retrieval floor is a real second capacity number; the
    mirror-operations unification (A3) keeps its cons-side leg."
  - P-281b: "the floor lands >= 0.65 — the cons teaches from ANYTHING;
    the rehearsal lane carries zero write information; 'delivery is
    free' collapses; T246's rehearsal-lane reading and W042's bridge both
    lose their object; the landing read becomes a property of the CONS
    alone."
  The mid-band (0.45-0.65) is MIXED with the five-point curve verbatim.

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE CONS := e261's chunked_consolidate VERBATIM BY IMPORT (e113:
    batch 32 = 16 jittered-pool install windows + 16 anchors; token-
    level union CE; AdamW (0.9,0.95) wd 0.1 const lr 1e-3; clip 1.0; s300
    seed 10901; NATURAL — no hook) run FRESH from each arm's seed state,
    all five in ONE session: the cons draws are bit-identical across arms
    by construction (one generator seed, one draw order) — the ONLY
    delta across arms is the seed state.
  * THE SEED STATES: (a) e001 (the 2.74M corpus base, fact-free, NO
    install — the zero point); (b) a FRESH rank-10 install (e261's
    chunked_install VERBATIM: Dmix s400 gen 24314, the hook = clip 1.0 ->
    project onto the fresh R10 room (CPU fp64, write fp32) -> opt.step;
    room k=10 at FRESH REGISTERED seeds 28111/28112 — dead by
    construction, gated G_R10DEAD post g0 < 0.01, e246's ALIGNED 2.86e-5
    the family precedent); (c) runs/checkpoints/e272_K1KM_inst_resume.pt
    (step 400); (d) runs/checkpoints/e268_CONCURRENT_inst_resume.pt
    (step 400); (e) runs/checkpoints/e270_CONCURRENT_inst_resume.pt
    (step 400). ALL THREE loaded checkpoints are COMPLETE final states —
    NO REGENERATION NEEDED (disclosed: the dispatch letter anticipated
    regenerating e268's/e270's final states if not checkpointed; the
    resume checkpoints ARE the committed final states — step 400, traj
    last step 400 — bound by FRESH md5 (recorded here for future cells,
    x10's no-committed-md5 convention) + the behavioral content match
    |d post g0| == 0 measured live in G_SEEDSTATE (bit-exact CPU re-read
    of the committed numbers).
  * THE LANDING BAND := [0.65, 0.82] HARD (the letter's own numbers);
    the +-10%-of-g1c window [0.6703, 0.8192] co-reported as CONTEXT; any
    adjudication-relevant point inside (0.65, 0.6703] or [0.8192, 0.82)
    straddling the two forms is DISCLOSED as a named band-edge branch
    (never silently adjudicated); the near-miss cushion 0.02 around the
    band names the band-edge-MIXED branch when all five points sit
    within (0.63, 0.84) but not all inside [0.65, 0.82].
  * THE DISCRIMINATOR := the five landings L(w), width-ordered
    w in {0 (NOINST), 10 (R10), 1000 (K1KM), 10000 (K10KD), 100000
    (K100KD)}; floor := L(0):
      FLAT iff every L(w) in [0.65, 0.82] (P-281b's clause floor >= 0.65
      plus all seeded arms in-band);
      THRESHOLDED iff floor < 0.45 AND a clean step exists: an adjacent
      width pair (w_lo -> w_hi) with L(w_lo) < 0.45, L(w_hi) in-band, and
      every arm above w_hi in-band (entry + stay) — the step's location
      reported verbatim (the (0,10] case is named "delivery needs a
      seed, any seed — the retrieval floor at-or-below rank-10");
      MASS-SCALED iff floor < 0.45 AND no clean step AND the seeded
      landings are monotonically non-decreasing in width (Spearman 1.0)
      with at least one seeded arm below 0.65 (continuous tracking that
      never fully enters the band);
      MIXED otherwise (named branches: the floor in the mid-band
      [0.45, 0.65); the floor in-band but a seeded arm out; a SMEARED
      step — an arm in [0.45, 0.65) adjacent to an in-band arm with the
      floor below 0.45; a non-monotone seeded curve; the band-edge
      straddle). Composite order TEXTURE (hard-gate failure) -> MIXED ->
      FLAT -> THRESHOLDED -> MASS-SCALED. The P-281a/P-281b clause
      scorings (floor < 0.45; floor >= 0.65) are reported verbatim
      beside the verdict.
  * THE FORMATION OVERLAY (context, hard-bound from the committed
    records): serial post g0 across {1k (e261) 0.000435, K1KM (e272)
    0.001245, 2k (e272) 0.0266, 5k (e272) 0.1271, 10k (e264) 0.2646,
    40k (e264) 0.3465, 100k (e264) 0.4360, 237k (e264) 0.3844} + e246's
    rank-10 2.86e-5 + the zero point 0 — the curve this cell is the
    COMPLEMENT of (e272's edge bracketed at (1k, 2k]; where the landing
    curve steps, if it steps at all, is the retrieval floor's own
    bracket).
  * THE COMMITTED LANDING ANCHORS (non-halting cons-lane texture,
    G_CONS_ANCHOR): arms (c)/(d)/(e) carry committed landings from their
    own sessions (e272's K1KM root 0.6879; e268's CONCURRENT root
    0.7119; e270's CONCURRENT root 0.8103) measured with the SAME cons
    seed on the SAME seed states — this cell's fresh landings cross-check
    them; |d| <= 0.02 expected to MISS about half the time (the known
    cross-session cons lottery family ~0.0419; e269/e270/e271's
    anchor-contamination precedents carried openly) — NEVER a bar input;
    the five fresh within-session points are the registered primary
    (one session, one cons stream — internally consistent by
    construction).
  * HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK,
    G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_SEEDSTATE,
    G_R10DEAD} — a failure HALTS (nothing adjudicated). G_CONS_ANCHOR is
    NON-halting texture. NO wash, NO serial re-runs, NO fresh FREE arm
    (the loaded seed states ARE the committed arms, bit-bound; the cons
    lane's instrument lineage is the family's own: e264's G_FREE PASS +
    e268/e269/e270/e271's serial anchors at |d post g0| <= 1.0e-6).
  * MEASURED, NEVER NOMINAL: per-arm seed displacement ||d|| vs base
    (the write-mass proxy), v-excess + cos-to-span of the seed and root
    displacements (vs e258's LOADED v-map and e246's committed LATE
    span), the cons trajectories (the rehearsal lane made visible), and
    the full cell reads (g-12/g0/g+12, held30, CE_R) at seed and root.

CHECKS (the dispatch's, in force): the machinery smoke FIRST; the
checkpoints bit-bound; the cons stream registered; n=1 per arm; the
anchor conventions; nothing guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175 s, per-step thermal polls at a 78 C margin, 40 s cooldowns, the
84 C never-past line recorded; CPU fp64 dense projections; CPU probing
threads 4. Estimated 10-15 GPU min (no regenerations needed).

Outputs: runs/e281/{metrics.json (PROGRESSIVE),
e281_rehearsal_dose.png, e281_instrument.png, REPORT.md, run.log
(gitignored)}; checkpoints runs/checkpoints/e281_*.pt. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Commit + push
per phase.

Run:  cd lab && python e281_rehearsal_dose.py    (E281_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
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
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

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

SMOKE = os.environ.get("E281_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e281_smoke" if SMOKE else "e281"
assert torch.cuda.is_available(), "e281 owns the GPU lane (dispatch)"

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
# the ported drivers can append into them — e270/e271's convention)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e264/e268/e271's disclosed convention, in force before
# ANY machinery call): e261's ported drivers resolve their module globals
# (log / NAME / SMOKE / LADDER / RUNG_NAMES / INST_STEPS / CONS_STEPS / T0 /
# thermal_log / device_events) AT CALL TIME through e261's module namespace —
# rebound HERE so they write THIS cell's log, label THIS cell's envelope
# polls (tag e281:ARM:phase, the dispatch's own form), run THIS cell's
# single-rung ladder, and land their thermal rows in THIS cell's ledger.
# The committed lab/e261_rank_ladder.py itself is untouched.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
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
ROOMS261_CK = "e261_rooms.pt"     # e261's committed rooms (context bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE R10 ROOM (arm b's vehicle): a FRESH rank-10 random room, seeds
# REGISTERED FRESH THIS CELL (28111/28112 — the family's per-cell seed
# convention: e260's 26xxx, e272's 272xx, THIS cell's 281xx). k=10 at full
# AND at smoke (a rank-10 room is cheap; the bit-gate is vacuous in smoke —
# no committed record at this room — disclosed).
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10, 28111, 28112),           # R10 — fresh, dead by construction
)
LADDER = LADDER_FULL
RUNG = {10: "R10"}
ROOM_MODE = "R10"
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# ---- THE SEED-STATE CHECKPOINTS (arms c/d/e): the committed final states,
# LOADED not regenerated (disclosed at birth). No committed md5 exists
# anywhere in the record for the resume files (x10's searched-and-disclosed
# finding) — FRESH md5s computed at birth and hard-bound HERE; the behavioral
# content match (|d post g0| vs the committed metrics) is measured live in
# G_SEEDSTATE (the desk pre-check read 0.0 bit-exact on all three).
K1KM_CK = "e272_K1KM_inst_resume.pt"
K1KM_MD5 = "2306a448b450f192ef2d7d1f9aa90e2e"
K10KD_CK = "e268_CONCURRENT_inst_resume.pt"
K10KD_MD5 = "9b66f01010283d5c775071c3c27e1686"
K100KD_CK = "e270_CONCURRENT_inst_resume.pt"
K100KD_MD5 = "59eeed2bb0f8eb067dd2aa36a958d0f7"

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "eb9f624708bcb6576c4115161dfd7042"
E272_VERDICT = "RANK-WRITES-THE-CURVE"
E272_K1KM_POST = 0.0012453667586669326          # THE below-edge dead 1k write
E272_K1KM_ROOT = 0.6879481673240662             # its committed landing (anchor)
E272_K1KM_KEPT = 0.016130520975733354
E272_K2K_POST = 0.026616254821419716            # the formation curve's 2k rung
E272_K5K_POST = 0.12709596753120422             # the 5k rung

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"
E268_VERDICT = "DYNAMICAL-CARRIER"
E268_K10KD_POST = 4.004325455753133e-05         # the 10k concurrent dead write
E268_K10KD_ROOT = 0.7118977308273315            # its committed landing (anchor)
E268_SERIAL_POST = 0.26464739441871643          # the 10k serial (formation)

E270_METRICS = E43.REPO / "runs" / "e270" / "metrics.json"
E270_MD5 = "006ae12e5ee7ff262db2d1c19d95c0f1"
E270_VERDICT = "MIXED"                 # the anchor-contamination letter; the
                                       # content read: TURBULENCE-TOTAL
E270_K100KD_POST = 0.010897441767156124        # the 100k concurrent dead write
E270_K100KD_ROOT = 0.8102989196777344          # root 0.8103's source (anchor)
E270_SERIAL_POST = 0.4359813928604126          # the 100k serial (formation)

# the formation curve's committed rungs (e264's 5-rung ladder, hard-bound in
# e272's own G_PARENTS — itself md5-bound here; context co-plots)
E264_CURVE_POST = {1000: 0.00043458465370349586, 10000: 0.26464763283729553,
                   40000: 0.346476286649704, 100000: 0.43598243594169617,
                   237123: 0.38436421751976013}
E264_CURVE_ROOT = {1000: 0.5768678784370422, 10000: 0.7707884907722473,
                   40000: 0.7519555687904358, 100000: 0.7276116609573364,
                   237123: 0.7105798125267029}
E246_R10_POST = E261.E246_ALIGNED_POST_G0      # 2.8590e-05 — the rank-10 dead
                                               # precedent (e261's literal)

# the base/root/artifact binds (committed where the record carries one —
# x6's e267 base bind, e261's span bind, e271's rooms bind; fresh otherwise,
# recorded here for future cells)
BASE_MD5 = "d114536d1c0983ab3be67f67ff0667c8"      # x6/e267 committed bind
ROOTREF_MD5 = "9c7d4ca1b60c8a1158d080f932e2c95f"   # fresh bind (this cell)
VMAP_MD5 = "baafd327e9e35cffa2d41ea4ecf5fc5c"      # fresh bind (this cell)
SPAN_MD5 = E261.E246_SPAN_MD5                      # e261's committed bind
ROOMS261_MD5 = "f81571c40bb6009d4ed1cfab66e73f6c"  # e271's committed bind

# the frozen bars' numbers
BAND = (0.65, 0.82)               # the family's landing band (the letter's
                                  # own numbers, HARD)
BAND_CTX = ((1 - E261.MATCH_BAND) * E261.G1C_ROOT_G0,
            (1 + E261.MATCH_BAND) * E261.G1C_ROOT_G0)   # [0.6703, 0.8192]
BELOW_BAR = 0.45                  # the letter's below-band split
STRADDLE_CUSHION = 0.02           # the band-edge near-miss naming window
DEAD_BAR = 0.01                   # the family's dead bar (G_R10DEAD)
SEEDSTATE_TOL = 5e-3              # the behavioral content-match bar (G_READ_TOL
                                  # family; the desk pre-check read 0.0)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
CONS_ANCHOR_TOL = 0.02            # the cons-lane anchor texture bar (known to
                                  # miss ~half the time — the lottery family)

# the five arms: name -> (width, seed source, committed landing anchor)
ARM_ORDER = ("NOINST", "K1KM", "K10KD", "K100KD", "R10")   # execution order
WIDTH_OF = {"NOINST": 0, "R10": 10, "K1KM": 1000, "K10KD": 10000,
            "K100KD": 100000}
SEEDED = ("R10", "K1KM", "K10KD", "K100KD")        # width order
LANDING_ANCHOR = {"NOINST": None, "R10": None,
                  "K1KM": E272_K1KM_ROOT, "K10KD": E268_K10KD_ROOT,
                  "K100KD": E270_K100KD_ROOT}
POST_ANCHOR = {"NOINST": None, "R10": None,
               "K1KM": E272_K1KM_POST, "K10KD": E268_K10KD_POST,
               "K100KD": E270_K100KD_POST}
ARM_DESC = {
    "NOINST": "THE ZERO POINT (the lynchpin): the fresh e001 root, NO "
              "install at all — the cons-only floor. The rehearsal lane "
              "asked to teach from nothing.",
    "K1KM": "e272's committed below-edge dead 1k write (the kept-matched "
            "dose-compensated arm; post g0 0.001245, dead at every "
            "measured condition; checkpoint LOADED bit-verified)",
    "K10KD": "e268's committed concurrent dead 10k write (post g0 "
             "4.004e-05, the DYNAMICAL-CARRIER cell's killed arm; "
             "checkpoint LOADED bit-verified)",
    "K100KD": "e270's committed concurrent dead 100k write (post g0 "
              "0.010897, the turbulence-total rung; root 0.8103's source; "
              "checkpoint LOADED bit-verified)",
    "R10": "a FRESH rank-10 room install (k=10, seeds 28111/28112 "
           "registered fresh this cell; Dmix s400 gen 24314, the family's "
           "standard hook), dead by construction — the family's 2.86e-5 "
           "floor precedent (e246's ALIGNED)",
}

REGISTERED = {
    "question_verbatim": "is the rehearsal lane's landing a property of "
        "the WRITE (delivery needs a seed — the retrieval floor is a real "
        "second capacity number) or of the CONS alone (it teaches from "
        "anything — the lane carries zero write information)?",
    "bars_verbatim": {
        "FLAT": "every arm lands in the family's landing band (~0.65-0.82) "
            "INCLUDING the no-install zero point — the lane carries zero "
            "write information; 'delivery is free' collapses to 'the cons "
            "teaches from anything'.",
        "THRESHOLDED": "the zero point lands < 0.45 AND the curve steps "
            "with seed width — delivery needs a seed; the retrieval floor "
            "is a second capacity number.",
        "MASS-SCALED": "the zero point lands < 0.45 but the seeded arms' "
            "landings track write mass continuously (no step).",
        "MIXED": "anything else — the five points verbatim, no inflation.",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE CONS := e261's chunked_consolidate "
        "VERBATIM (e113; s300 seed 10901; NATURAL) run FRESH from each "
        f"arm's seed state, all five in ONE session — the cons draws are "
        "bit-identical across arms by construction; the ONLY delta is the "
        "seed state; THE SEED STATES := (a) e001 (no install — the zero "
        "point), (b) a FRESH rank-10 install (room k=10 seeds 28111/28112 "
        "registered fresh; Dmix s400 gen 24314; dead gated < "
        f"{DEAD_BAR}), (c) e272_K1KM_inst_resume.pt, (d) "
        "e268_CONCURRENT_inst_resume.pt, (e) e270_CONCURRENT_inst_resume.pt "
        "— ALL THREE COMPLETE final states (step 400), LOADED not "
        "regenerated (disclosed), bound by FRESH md5 + the behavioral "
        "content match measured live in G_SEEDSTATE; THE LANDING BAND := "
        f"[{BAND[0]}, {BAND[1]}] HARD (the letter's numbers), the "
        f"+-10%-of-g1c window [{BAND_CTX[0]:.4f}, {BAND_CTX[1]:.4f}] "
        "co-reported as context, the near-miss cushion "
        f"+-{STRADDLE_CUSHION} naming the band-edge-MIXED branch; THE "
        f"DISCRIMINATOR := the five landings width-ordered, floor := L(0): "
        f"FLAT iff all five in [{BAND[0]}, {BAND[1]}]; THRESHOLDED iff "
        f"floor < {BELOW_BAR} AND a clean adjacent step (L(lo) < "
        f"{BELOW_BAR} -> L(hi) in-band, all above in-band, location "
        "reported verbatim); MASS-SCALED iff floor < "
        f"{BELOW_BAR} AND no clean step AND the seeded landings "
        "monotone non-decreasing in width with at least one below "
        f"{BAND[0]}; MIXED otherwise (named branches: the floor mid-band "
        f"[{BELOW_BAR}, {BAND[0]}); the floor in-band but a seeded arm "
        f"out; a smeared step; a non-monotone seeded curve; the band-edge "
        "straddle); composite order TEXTURE -> MIXED -> FLAT -> "
        "THRESHOLDED -> MASS-SCALED; the P-281a/P-281b clause scorings "
        "reported verbatim beside the verdict; THE COMMITTED LANDING "
        "ANCHORS (arms c/d/e: 0.6879/0.7119/0.8103) cross-checked under "
        "G_CONS_ANCHOR (NON-halting — the cross-session cons lottery "
        "~0.0419 family, e269/e270/e271's precedent), never bar inputs; "
        "HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, "
        "G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, "
        "G_PROJ, G_SEEDSTATE, G_R10DEAD} (a failure HALTS); no wash, no "
        "serial re-runs, no fresh FREE arm (the loaded seed states ARE "
        "the committed arms; the cons instrument lineage is the "
        "family's own).",
    ),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (W045's registered fork: P-281a/P-281b, 2026-10-05 "
        "22:39Z, BEFORE the letter); this script committed at birth "
        "BEFORE any compute; adjudicate against exactly this; no bar "
        "shopping.",
    "predictions_cited": {
        "P-281a_W045": "the cons-only floor lands BELOW 0.45 — the "
            "write's residue carries the landing; 'delivery is free' "
            "survives as DELIVERY NEEDS A SEED; the retrieval floor is a "
            "real second capacity number; the mirror-operations "
            "unification (A3) keeps its cons-side leg.",
        "P-281b_W045": "the floor lands >= 0.65 — the cons teaches from "
            "ANYTHING; the rehearsal lane carries zero write "
            "information; 'delivery is free' collapses; T246's "
            "rehearsal-lane reading and W042's bridge both lose their "
            "object; the landing read becomes a property of the CONS "
            "alone.",
        "note": "CITED from W045 (registered before the letter), not "
                "re-registered here; the mid-band (0.45-0.65) is MIXED "
                "with the curve verbatim.",
    },
}

deviations: list[str] = [
    "THE SEED STATES ARE ALL CHECKPOINTED — NO REGENERATION (disclosed at "
    "birth): the dispatch letter anticipated regenerating e268's/e270's "
    "10k/100k dead writes if their final states were not checkpointed; "
    "the resume checkpoints ARE the committed final states (step 400, "
    "traj last step 400, verified at bind) — e268_CONCURRENT_inst_resume.pt, "
    "e270_CONCURRENT_inst_resume.pt, and e272_K1KM_inst_resume.pt are "
    "LOADED, not re-run. Bind form (x10's convention — no committed md5 "
    "exists anywhere for the resume files, searched): FRESH md5 computed "
    "at birth and hard-bound in THIS script + the behavioral content "
    "match |d post g0| vs the committed metrics measured live in "
    "G_SEEDSTATE (bar "
    f"{SEEDSTATE_TOL}; the birth desk-check read 0.0 bit-exact on all "
    "three). The ~5 GPU min the letter budgeted for regenerations is "
    "therefore NOT spent.",
    "THE SMOKE RECORD (the e260-family discipline): the machinery smoke "
    "ran FIRST (E281_SMOKE=1, runs/e281_smoke/) — pass 1's findings "
    "recorded in the smoke metrics verbatim; NOTHING adjudicated in "
    "smoke (SMOKE stamp on every read).",
    "THE CONS RUNS FRESH FROM LOADED STATES, ONE SESSION (the design's "
    "own strength, stated): every prior landing read (e268's 0.7119, "
    "e270's 0.8103, e272's 0.6879) was measured in ITS OWN session; this "
    "cell re-measures all five under ONE cons stream in ONE session — "
    "the five points are internally consistent by construction (the "
    "cross-session lottery ~0.0419 cannot bend the CURVE's shape, only "
    "its level), and the committed anchors live in G_CONS_ANCHOR as "
    "non-halting texture.",
    "THE R10 ROOM IS FRESH-SEEDED, NOT e246's ALIGNED ROOM (stated at "
    "birth): the letter's arm (b) asks for 'a fresh rank-10 install, "
    "dead by construction — the family's 2.86e-5 floor precedent'; "
    "e246's 2.86e-5 was a SPAN-ALIGNED rank-10 room; THIS cell's R10 is "
    "a RANDOM rank-10 room (seeds 28111/28112, fresh registered) at the "
    "family's standard install — dead by construction is the premise "
    "(gated), and the 2.86e-5 is the PRECEDENT CONTEXT (e260 already "
    "showed random-vs-aligned room choice does not matter at these "
    "widths: both die), not a bit-bind.",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed 2.74M v-map + e246's committed LATE span feed the "
    "measured-loads ledger and the R10 room's hook machinery; no new "
    "history is run. The rooms' displacement 'in_own_room' reads are vs "
    "the R10 room only (the one room this session builds) — disclosed "
    "as context (the loaded arms' own rooms are their parents' objects; "
    "rebuilding all four rooms would spend GPU-adjacent time on "
    "non-registered reads).",
    "THE LANDING BAND'S TWO FORMS (stated): the letter's ~0.65-0.82 is "
    f"frozen HARD as [{BAND[0]}, {BAND[1]}]; the family's own "
    f"+-10%-of-g1c band [{BAND_CTX[0]:.4f}, {BAND_CTX[1]:.4f}] is "
    "co-reported as context; the two differ by <= 0.0035 at the edges — "
    "any adjudication-relevant point landing between them is disclosed "
    "as a named band-edge branch, never silently adjudicated.",
    "NO WASH, NO SERIAL RE-RUNS, NO FRESH FREE ARM (disclosed; e268's "
    "skip carried): the loaded seed states ARE the committed arms "
    "(bit-bound), the fresh R10 install is the only fresh write, and "
    "the cons lane's instrument lineage is the family's own record "
    "(e264's G_FREE PASS install L2 6.2e-4; e268/e269/e270/e271's "
    "serial anchors at |d post g0| 2.4e-7/9.5e-7/1.0e-6/9.5e-7).",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked chunked_install driver + the chunked_consolidate cons driver "
    "(bit-identical arithmetic + draw order), the thermal envelope "
    "(per-step polls, 78C margin, 175s bursts, 40s cooldowns, the 84C "
    "line), the progressive-metrics + resume-ckpt conventions — the "
    "module-global rebinding (log/NAME/T0/LADDER/RUNG_NAMES/SMOKE/INST_"
    "STEPS/CONS_STEPS/thermal ledgers, disclosed in-code) retargets the "
    "drivers' I/O to this cell; the committed lab/e261_rank_ladder.py is "
    "NOT modified.",
    "n=1 per arm, one lineage, one session (the g-series standing "
    "caveat — the critic's lottery note carried verbatim); the CURVE's "
    "shape is the registered object, not any single point; nothing "
    "guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E281_SMOKE=1): 8 install + 8 cons steps, the same R10 "
    "k=10 room (the room bit-gate VACUOUS in smoke — no committed record "
    "at this room; disclosed), the loaded seed states loaded as in full "
    "(their binds exercised), all paths smoke_-prefixed, own smoke dir; "
    "NOTHING adjudicated or gated (SMOKE stamp on every read).",
]

# (device_events / thermal_log defined above the e261 rebinding — the
# shared instrument ledgers)


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


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e281_rehearsal_dose",
        "phase": "THE REHEARSAL DOSE-RESPONSE — the retrieval story's "
                 "lynchpin (W045's registered fork): the landing read "
                 "(root g0, the rehearsal-lane convention, carried never "
                 "adjudicated as survival) vs the seed state's write width "
                 "— five points {0 no-install, 10 rank-10 dead, 1k K1KM "
                 "dead, 10k concurrent dead, 100k concurrent dead} under "
                 "ONE cons stream (e113 s300 seed 10901) in ONE session — "
                 "the complement of e272's formation curve: FLAT "
                 "(cons-powered; P-281b) vs THRESHOLDED (delivery needs "
                 "a seed; P-281a's world) vs MASS-SCALED vs MIXED",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (the owner's max-priority window; "
                      "this cell owns the GPU lane) + CPU fp64 dense "
                      "projections, CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s, per-step thermal polls "
                      f"at a {E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line recorded (tags e281:ARM:phase)",
            "trainings": f"5 cons s{E261.CONS_STEPS} (seed 10901 HELD, "
                         "bit-identical draws across arms) + 1 fresh R10 "
                         f"install s{E261.INST_STEPS} (k=10 room, seeds "
                         "28111/28112); the three dead seed states LOADED "
                         "from committed checkpoints (no regeneration); "
                         "NO washes; NO serial re-runs; NO fresh FREE",
        },
        "deviations": deviations,
        "builds_on": [
            "W045 (THE registration this cell adjudicates: P-281a/P-281b "
            "registered 2026-10-05 22:39Z BEFORE the letter — the "
            "cons-only floor is the lynchpin; e272's K1KM root 0.6879 "
            "already proved a below-edge write IS cons-retrievable, so "
            "the retrieval floor sits at-or-below 1k-of-write ALREADY "
            "and the zero point is the only genuinely open arm)",
            "T246 / e268 (the rehearsal lane's ORIGINAL reading: the cons "
            "re-teaches a dead write to landing strength — THE LANDING "
            "READ AND THE WRITE READ ARE DIFFERENT OBJECTS; the 10k dead "
            "write landed 0.7119; the interleave + this cell's 10K-DEAD "
            "seed state originate here)",
            "T255 / e272 (the formation curve this cell is the complement "
            "of: the edge bracketed at (1k, 2k], 1k 0.000435 -> 2k 0.0266 "
            "-> 5k 0.1271 -> 10k 0.2646; the K1KM dose-compensated arm "
            "STILL DEAD 0.001245 with its rehearsal lane 5-for-5 in-band "
            "— the (c) seed state + the formation overlay's committed "
            "numbers)",
            "T249 / e271 (the cons machinery + landing-read conventions "
            "ported by import; the anchor-contamination precedent; the "
            "100K-DEAD seed state's sibling record)",
            "T248 / e270 (the 100k concurrent dead write — the (e) seed "
            "state; root 0.8103's source, ABOVE its serial's 0.6884 — the "
            "rehearsal lane's sharpest prior form)",
            "x10 (the loaded-checkpoint bind convention: fresh md5 + "
            "behavioral content match, no committed md5 anywhere — "
            "searched and disclosed)",
        ],
        "whats_new": [
            "THE ZERO POINT, MEASURED FOR THE FIRST TIME: no prior cell "
            "ever ran the cons from a fresh un-installed root — the "
            "cons-only floor is the one number that decides whether the "
            "rehearsal lane carries ANY write information (W045: 'the "
            "retrieval floor sits at-or-below 1k-of-write ALREADY, and "
            "e281's only genuinely open arm is the ZERO POINT')",
            "THE FIVE-POINT DOSE-RESPONSE IN ONE SESSION: the landing "
            "read vs seed width {0, 10, 1k, 10k, 100k} under ONE "
            "bit-identical cons stream — the curve's SHAPE is immune to "
            "the cross-session cons lottery that scattered the prior "
            "landing reads (0.577-0.810) across sessions",
            "THE COMPLEMENT CURVE: e272 located the FORMATION edge at "
            "(1k, 2k]; this cell asks where (whether) the RETRIEVAL "
            "curve steps on the same axis — the retrieval floor's own "
            "bracket, the second capacity number if P-281a's world is "
            "real",
        ],
        "gates": {},
    })
    log("E281 — THE REHEARSAL DOSE-RESPONSE (the retrieval story's "
        f"lynchpin; W045's fork) (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARM_ORDER)}; cons = e113 s{E261.CONS_STEPS} seed "
        f"{E261.CONS_SEED} HELD (bit-identical across arms); seed states: "
        "e001 (no install) / fresh R10 (k=10, 28111/28112) / LOADED "
        "e272_K1KM + e268_CONCURRENT + e270_CONCURRENT (all step-400 "
        "complete, no regeneration); discriminator: the five landings vs "
        f"band [{BAND[0]}, {BAND[1]}] with floor bar {BELOW_BAR}")
    write_partial("startup (bars registered, committed at birth)")

    set_seed(10901)       # global init only; every RNG is its own

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

    # ================= P0b: the parents + seed states hard-bound =========
    e272m = json.loads(E272_METRICS.read_text(encoding="utf-8"))
    e272_k1km_post = e272m["adjudication"]["reads"]["post_g0"]["K1KM"]
    e272_k1km_root = e272m["adjudication"]["reads"]["root_g0_carried"]["K1KM"]
    e268m = json.loads(E268_METRICS.read_text(encoding="utf-8"))
    e268_post = e268m["adjudication"]["reads"]["CONCURRENT"]["post_g0"]
    e268_root = e268m["adjudication"]["reads"]["CONCURRENT"]["root_g0"]
    e268_serial = e268m["adjudication"]["reads"]["SERIAL"]["post_g0"]
    e270m = json.loads(E270_METRICS.read_text(encoding="utf-8"))
    e270_post = e270m["adjudication"]["reads"]["CONCURRENT"]["post_g0"]
    e270_root = e270m["adjudication"]["reads"]["CONCURRENT"]["root_g0"]
    e270_serial = e270m["adjudication"]["reads"]["SERIAL"]["post_g0"]

    seed_files = {"K1KM": (K1KM_CK, K1KM_MD5, E272_K1KM_POST),
                  "K10KD": (K10KD_CK, K10KD_MD5, E268_K10KD_POST),
                  "K100KD": (K100KD_CK, K100KD_MD5, E270_K100KD_POST)}
    seed_binds = {}
    for arm, (fn, fmd5, committed_post) in seed_files.items():
        p = CKPT_DIR / fn
        art = torch.load(p, map_location="cpu", weights_only=False)
        step = int(art.get("step", -1))
        traj_steps = ([int(t["step"]) for t in art["traj"]]
                      if isinstance(art.get("traj"), list) else [])
        seed_binds[arm] = {
            "path": f"runs/checkpoints/{fn}", "md5": md5of(p),
            "bound_md5": fmd5, "step": step,
            "traj_last_step": traj_steps[-1] if traj_steps else None,
            "committed_post_g0": committed_post,
            "note": "the committed FINAL state (resume ckpt complete at "
                    "s400) — LOADED, not regenerated (x10's bind "
                    "convention: no committed md5 anywhere, fresh-bound "
                    "here)",
        }
        del art

    G_PARENTS = {
        "e272_metrics": {"path": str(E272_METRICS),
                         "md5": md5of(E272_METRICS), "bound_md5": E272_MD5,
                         "verdict": e272m["adjudication"]["verdict"],
                         "K1KM_post_g0": e272_k1km_post,
                         "K1KM_root_g0": e272_k1km_root,
                         "K2K_post_g0": e272m["adjudication"]["reads"]
                                        ["post_g0"]["K2K"],
                         "K5K_post_g0": e272m["adjudication"]["reads"]
                                        ["post_g0"]["K5K"],
                         "note": "the formation curve's (1k, 10k] rungs + "
                                 "the (c) seed state's own record"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of(E268_METRICS), "bound_md5": E268_MD5,
                         "verdict": e268m["adjudication"]["verdict"],
                         "CONCURRENT_post_g0": e268_post,
                         "CONCURRENT_root_g0": e268_root,
                         "SERIAL_post_g0": e268_serial,
                         "note": "the (d) seed state's own record (the "
                                 "DYNAMICAL-CARRIER cell's killed arm)"},
        "e270_metrics": {"path": str(E270_METRICS),
                         "md5": md5of(E270_METRICS), "bound_md5": E270_MD5,
                         "verdict": e270m["adjudication"]["verdict"],
                         "CONCURRENT_post_g0": e270_post,
                         "CONCURRENT_root_g0": e270_root,
                         "SERIAL_post_g0": e270_serial,
                         "note": "the (e) seed state's own record (root "
                                 "0.8103's source)"},
        "seed_checkpoints": seed_binds,
        "artifacts": {
            "base_e001": {"md5": md5of(CKPT_DIR / BASE_CK),
                          "bound_md5": BASE_MD5,
                          "source": "x6's e267 committed bind"},
            "root_ref_g1c": {"md5": md5of(CKPT_DIR / ROOT_CK),
                             "bound_md5": ROOTREF_MD5,
                             "source": "fresh bind (this cell; recorded "
                                       "for future cells)"},
            "vmap_e258": {"md5": md5of(CKPT_DIR / VMAP_CK),
                          "bound_md5": VMAP_MD5,
                          "source": "fresh bind (this cell)"},
            "span_e246": {"md5": md5of(CKPT_DIR / SPAN_CK),
                          "bound_md5": SPAN_MD5,
                          "source": "e261's committed bind"},
            "rooms_e261": {"md5": md5of(CKPT_DIR / ROOMS261_CK),
                           "bound_md5": ROOMS261_MD5,
                           "source": "e271's committed bind (context: the "
                                     "K1KM seed's room provenance)"},
        },
        "hardbound": {
            "e272_verdict": E272_VERDICT,
            "e272_K1KM_post_g0": E272_K1KM_POST,
            "e272_K1KM_root_g0": E272_K1KM_ROOT,
            "e272_K2K_post_g0": E272_K2K_POST,
            "e272_K5K_post_g0": E272_K5K_POST,
            "e268_verdict": E268_VERDICT,
            "e268_K10KD_post_g0": E268_K10KD_POST,
            "e268_K10KD_root_g0": E268_K10KD_ROOT,
            "e268_serial_post_g0": E268_SERIAL_POST,
            "e270_verdict": E270_VERDICT,
            "e270_K100KD_post_g0": E270_K100KD_POST,
            "e270_K100KD_root_g0": E270_K100KD_ROOT,
            "e270_serial_post_g0": E270_SERIAL_POST,
            "e264_curve_post": E264_CURVE_POST,
            "e264_curve_root": E264_CURVE_ROOT,
            "e246_rank10_precedent_post_g0": E246_R10_POST,
        },
        "pass": bool(
            e272m["adjudication"]["verdict"] == E272_VERDICT
            and abs(e272_k1km_post - E272_K1KM_POST) < 1e-12
            and abs(e272_k1km_root - E272_K1KM_ROOT) < 1e-12
            and abs(e272m["adjudication"]["reads"]["post_g0"]["K2K"]
                    - E272_K2K_POST) < 1e-12
            and abs(e272m["adjudication"]["reads"]["post_g0"]["K5K"]
                    - E272_K5K_POST) < 1e-12
            and e268m["adjudication"]["verdict"] == E268_VERDICT
            and abs(e268_post - E268_K10KD_POST) < 1e-12
            and abs(e268_root - E268_K10KD_ROOT) < 1e-12
            and abs(e268_serial - E268_SERIAL_POST) < 1e-12
            and e270m["adjudication"]["verdict"] == E270_VERDICT
            and abs(e270_post - E270_K100KD_POST) < 1e-12
            and abs(e270_root - E270_K100KD_ROOT) < 1e-12
            and abs(e270_serial - E270_SERIAL_POST) < 1e-12
            and md5of(E272_METRICS) == E272_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E270_METRICS) == E270_MD5
            and all(b["md5"] == b["bound_md5"] and b["step"] == 400
                    and b["traj_last_step"] == 400
                    for b in seed_binds.values())
            and md5of(CKPT_DIR / BASE_CK) == BASE_MD5
            and md5of(CKPT_DIR / ROOT_CK) == ROOTREF_MD5
            and md5of(CKPT_DIR / VMAP_CK) == VMAP_MD5
            and md5of(CKPT_DIR / SPAN_CK) == SPAN_MD5
            and md5of(CKPT_DIR / ROOMS261_CK) == ROOMS261_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e272 {E272_VERDICT} (K1KM post "
        f"{e272_k1km_post:.6f} / root {e272_k1km_root:.6f}), e268 "
        f"{E268_VERDICT} (CONC post {e268_post:.2e} / root "
        f"{e268_root:.6f}), e270 {E270_VERDICT} (CONC post "
        f"{e270_post:.6f} / root {e270_root:.6f}); the three seed "
        f"checkpoints md5+step-400 bound (COMPLETE final states)")
    write_partial("P0b parents + seed-state files hard-bound")
    del e272m, e268m, e270m

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_g0 = G1.battery_cell(base_net, g0_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "fact_free_g0": base_g0,
              "ce_r": base_ce_r,
              "fact_free": bool(base_gm12 <= 0.05 and base_g0 <= 0.05),
              "pass": bool(base_gm12 <= 0.05 and base_g0 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.2e}, g0 {base_g0:.2e}, CE_R "
        f"{base_ce_r:.4f}): PASS — THE ZERO POINT'S SEED STATE")
    write_partial("P0c G-BASE PASSED (the no-install seed state)")

    # ================= P1: the reference root + artifacts + R10 room =====
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
        "abs_diff": abs(root_read - E261.G1C_ROOT_GM12),
        "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12) < G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"P1 G_ROOT: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {E261.G1C_ROOT_GM12:.10f} "
        f"(|d| {abs(root_read - E261.G1C_ROOT_GM12):.1e}): PASS")
    write_partial("P1 G_ROOT PASSED")
    del root_net

    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "mean_v": float(v64_np.mean()),
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    log(f"P1 G_VMBIND: {VMAP_CK} (md5 {G_VMBIND['md5'][:8]}..., "
        f"{G_VMBIND['size']} coords): PASS")

    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND
    log(f"P1 G_SPANBIND: {SPAN_CK} (rank {G_SPANBIND['rank']}): PASS")

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the fresh R10 room certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY and the kept^2 rank "
                "probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N — at k=10 the bar (8.2e-6) sits ABOVE the "
                "expectation (3.65e-6): the probe still catches size bugs "
                "(a wrong k shifts the mean), disclosed); the span-overlap "
                "(expect ~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N)"},
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

    rooms_ck = save_ckpt(
        "e281_rooms",
        {ROOM_MODE: {"D_int8": rooms.rooms[ROOM_MODE].D.astype(np.int8),
                     "S": rooms.rooms[ROOM_MODE].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e281's R10 room (the flat basis is net.parameters() "
                 "order): a FRESH rank-10 random room (seeds 28111/28112, "
                 "registered this cell) — arm (b)'s install vehicle, dead "
                 "by construction (the 2.86e-5 precedent)",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][ROOM_MODE].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "r10": {"k": LADDER[0][0], "name": ROOM_MODE,
                "seeds": [LADDER[0][1], LADDER[0][2]],
                "k_fraction_of_N": LADDER[0][0] / N,
                "note": "FRESH registered this cell (the family's per-cell "
                        "seed convention); dead by construction, gated "
                        "G_R10DEAD; e246's ALIGNED 2.86e-5 the precedent "
                        "context"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE R10 ROOM: {ROOM_MODE} (k={LADDER[0][0]}): BUILT + "
        f"CERTIFIED")
    write_partial("P1 the R10 room built (parents bound + v-map loaded "
                  "+ span loaded + certification)")

    # ================= P2: THE FIVE ARMS =================================
    arms_rec: dict = {}
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None

    def read_cells(sd: dict) -> dict:
        net_ = G1.evl_load(sd)
        cells_ = {"gm12": G1.battery_cell(net_, gm12_ids, zid)["mean_pz"],
                  "g0": G1.battery_cell(net_, g0_ids, zid)["mean_pz"],
                  "gp12": G1.battery_cell(net_, bat_ids[12], zid)["mean_pz"],
                  "ce_r": G1.ce_fixed_cpu(net_, *r_eval_xy)}
        return cells_

    def seed_record(arm: str, sd: dict, note: str) -> dict:
        cells = read_cells(sd)
        net_ = G1.evl_load(sd)
        d = flat_params_cpu(net_) - base_flat
        load = rooms.displacement_loads(d, ROOM_MODE)
        del net_
        rec = {"cells": cells, "post_g0": cells["g0"],
               "displacement_l2": float(np.linalg.norm(
                   d.double().numpy())), "displacement_loads": load,
               "note": note}
        if POST_ANCHOR[arm] is not None:
            rec["committed_post_g0"] = POST_ANCHOR[arm]
            rec["abs_diff_vs_committed"] = abs(cells["g0"]
                                               - POST_ANCHOR[arm])
        return rec

    def run_cons(arm: str, sd_seed: dict) -> dict:
        cons = E261.chunked_consolidate(
            f"{arm}:cons", G1.evl_load(sd_seed), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e281_{arm}_cons_resume.pt" if SMOKE
                        else f"e281_{arm}_cons_resume.pt"), dev)
        return cons

    def read_root(arm: str, theta0: dict) -> dict:
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"]
                 for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - base_flat
        load_root = rooms.displacement_loads(d_root, ROOM_MODE)
        root_ck = save_ckpt(
            f"e281_{arm}_root", theta0,
            {"desc": f"e281 ARM-{arm} (width {WIDTH_OF[arm]}): "
                     f"{ARM_DESC[arm][:110]}... + e113 cons "
                     f"s{E261.CONS_STEPS} (seed {E261.CONS_SEED} HELD — "
                     "the shared stream)",
             "arm": arm, "width": WIDTH_OF[arm],
             "cons_seed": E261.CONS_SEED,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e281_rooms.pt"})
        del root_net_arm
        landed_band = bool(BAND[0] <= cells["g+0"] <= BAND[1])
        landed_ctx = bool(BAND_CTX[0] <= cells["g+0"] <= BAND_CTX[1])
        return {"cells": cells, "gm12": cells["g-12"], "g0": cells["g+0"],
                "displacement_l2": float(np.linalg.norm(
                    d_root.double().numpy())),
                "displacement_loads": load_root, "checkpoint": root_ck,
                "landed_band": landed_band, "landed_ctx": landed_ctx}

    # ---- the loaded seed states' behavioral bind (G_SEEDSTATE) ----------
    log("=" * 78)
    log("G_SEEDSTATE: the three loaded seed states' behavioral content "
        "match (the committed arms' post g0 re-read on CPU)")
    g_seedstate = {"form": "each loaded seed checkpoint's model re-read on "
                   "the g0 battery vs its committed post g0 (bar "
                   f"{SEEDSTATE_TOL}; the resume file IS the final state "
                   "the parent cell evaluated)",
                   "arms": {}, "bar": SEEDSTATE_TOL}
    loaded_sd = {}
    for arm, (fn, fmd5, committed_post) in seed_files.items():
        art = torch.load(CKPT_DIR / fn, map_location="cpu",
                         weights_only=False)
        loaded_sd[arm] = {k: v.detach().clone()
                          for k, v in art["model"].items()}
        cells = read_cells(loaded_sd[arm])
        g_seedstate["arms"][arm] = {
            "post_g0_measured": cells["g0"],
            "post_g0_committed": committed_post,
            "abs_diff": abs(cells["g0"] - committed_post),
            "gm12": cells["gm12"], "ce_r": cells["ce_r"]}
        del art
    g_seedstate["pass"] = bool(all(
        v["abs_diff"] <= SEEDSTATE_TOL for v in g_seedstate["arms"].values()))
    assert g_seedstate["pass"], f"seed-state bind FAILED: {g_seedstate}"
    metrics["gates"]["G_SEEDSTATE"] = g_seedstate
    for arm, v in g_seedstate["arms"].items():
        log(f"  {arm}: post g0 {v['post_g0_measured']:.12f} vs committed "
            f"{v['post_g0_committed']:.12f} (|d| {v['abs_diff']:.2e}): "
            f"{'BIT-EXACT' if v['abs_diff'] == 0.0 else 'match'}")
    write_partial("P2a G_SEEDSTATE PASSED (the three loaded arms "
                  "behaviorally bound)")

    # ---- ARM-NOINST (the zero point) + the three loaded arms ------------
    for arm in ("NOINST", "K1KM", "K10KD", "K100KD"):
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        if arm == "NOINST":
            sd_seed = base_sd
            seed_rec = seed_record(
                arm, sd_seed, "the fresh e001 root, NO install — the zero "
                              "point's seed state (post g0 = the base's "
                              "fact-free read)")
        else:
            sd_seed = loaded_sd[arm]
            seed_rec = seed_record(arm, sd_seed,
                                   f"the committed {arm} seed state "
                                   "(LOADED, G_SEEDSTATE-bound)")
        arms_rec[arm] = {"desc": ARM_DESC[arm], "width": WIDTH_OF[arm],
                         "seed": seed_rec}
        log(f"  seed: post g0 {seed_rec['post_g0']:.6f} | ||d|| "
            f"{seed_rec['displacement_l2']:.4f} | v-excess "
            f"{seed_rec['displacement_loads']['v_excess']:.2f} cos-to-span "
            f"{seed_rec['displacement_loads']['cos_to_span']:.4f}")
        burst_cooldown(f"{arm} seed->cons")
        cons = run_cons(arm, sd_seed)
        arms_rec[arm]["consolidation"] = {"traj": cons["traj"],
                                          "chunk_table": cons["chunk_table"]}
        arms_rec[arm]["root"] = read_root(arm, cons["sd"])
        r_ = arms_rec[arm]["root"]
        log(f"ARM-{arm} ROOT: g0 {r_['g0']:.4f} "
            f"landed_band={r_['landed_band']} | g-12 {r_['gm12']:.4f} "
            f"(lottery) | CE_R {r_['cells']['ce_r']:.4f} | ||d_root|| "
            f"{r_['displacement_l2']:.4f}")
        write_partial(f"P2 ARM-{arm} cons + root (landing "
                      f"{r_['g0']:.4f})")

    # ---- ARM-R10 (the fresh dead-by-construction install) ---------------
    log("=" * 78)
    log(f"ARM-R10 — {ARM_DESC['R10']}")
    inst_r10 = E261.chunked_install(
        "R10:inst", ROOM_MODE, G1.evl_load(base_sd), rooms,
        inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e281_R10_inst_resume.pt" if SMOKE
                    else "e281_R10_inst_resume.pt"), dev)
    sd_r10 = inst_r10["sd"]
    seed_rec = seed_record("R10", sd_r10,
                           "the fresh rank-10 install's post state (dead "
                           "by construction — gated below)")
    led_kept = [v["kept_frac"] for v in inst_r10["ledger"].values()]
    arms_rec["R10"] = {"desc": ARM_DESC["R10"], "width": WIDTH_OF["R10"],
                       "install": {"traj": inst_r10["traj"],
                                   "ledger": inst_r10["ledger"],
                                   "ledger_kept_frac_median": med(led_kept),
                                   "chunk_table": inst_r10["chunk_table"],
                                   "steps": E261.INST_STEPS,
                                   "resumed_final":
                                       bool(inst_r10.get("resumed_final",
                                                          False)),
                                   "opt_steps": E261.INST_STEPS},
                       "seed": seed_rec}
    G_R10DEAD = {
        "form": "the fresh rank-10 install is DEAD by construction (the "
                "arm's premise): post g0 < the family's dead bar "
                f"{DEAD_BAR}; context — e246's ALIGNED rank-10 precedent "
                f"{E246_R10_POST:.3e}",
        "post_g0": seed_rec["post_g0"], "bar": DEAD_BAR,
        "e246_precedent": E246_R10_POST,
        "pass": bool(seed_rec["post_g0"] < DEAD_BAR),
        "vacuous_note": "SMOKE: the 8-step install's read carries no "
                        "formation claim (SMOKE stamp)" if SMOKE else None,
    }
    if not SMOKE:
        assert G_R10DEAD["pass"], f"R10 not dead: {G_R10DEAD}"
    else:
        G_R10DEAD["pass"] = True
        G_R10DEAD["vacuous"] = True
    metrics["gates"]["G_R10DEAD"] = G_R10DEAD
    log(f"  seed: post g0 {seed_rec['post_g0']:.6f} (dead bar "
        f"{DEAD_BAR}; e246 precedent {E246_R10_POST:.3e}) | ||d|| "
        f"{seed_rec['displacement_l2']:.4f} | kept med {med(led_kept):.4f}")
    log(f"ARM-R10 G_R10DEAD: "
        f"{'PASS' if G_R10DEAD['pass'] else 'FAIL'}"
        + (" (SMOKE-vacuous)" if SMOKE else ""))
    write_partial("P2 ARM-R10 install + G_R10DEAD")
    if not inst_r10.get("resumed_final", False):
        burst_cooldown("R10 inst->cons")
    cons_r10 = run_cons("R10", sd_r10)
    arms_rec["R10"]["consolidation"] = {"traj": cons_r10["traj"],
                                        "chunk_table":
                                            cons_r10["chunk_table"]}
    arms_rec["R10"]["root"] = read_root("R10", cons_r10["sd"])
    r_ = arms_rec["R10"]["root"]
    log(f"ARM-R10 ROOT: g0 {r_['g0']:.4f} landed_band={r_['landed_band']} | "
        f"g-12 {r_['gm12']:.4f} (lottery) | CE_R "
        f"{r_['cells']['ce_r']:.4f} | ||d_root|| "
        f"{r_['displacement_l2']:.4f}")
    write_partial(f"P2 ARM-R10 cons + root (landing {r_['g0']:.4f})")

    # ---- G_CONS_ANCHOR (non-halting cons-lane texture) ------------------
    g_cons_anchor = {"form": "the fresh landings for the three loaded arms "
                     "vs their committed landings (their own sessions' "
                     "cons on the SAME bit-identical seed states + seed): "
                     "|d| <= "
                     f"{CONS_ANCHOR_TOL} expected to MISS about half the "
                     "time (the cross-session cons lottery family "
                     "~0.0419; e269/e270/e271's anchor-contamination "
                     "precedents) — NON-HALTING texture, never a bar "
                     "input; the five within-session points are the "
                     "registered primary",
                     "arms": {}, "bar": CONS_ANCHOR_TOL, "pass": None}
    for arm in ("K1KM", "K10KD", "K100KD"):
        mine = arms_rec[arm]["root"]["g0"]
        g_cons_anchor["arms"][arm] = {
            "mine": mine, "committed": LANDING_ANCHOR[arm],
            "abs_diff": abs(mine - LANDING_ANCHOR[arm])}
    g_cons_anchor["all_within_bar"] = bool(all(
        v["abs_diff"] <= CONS_ANCHOR_TOL
        for v in g_cons_anchor["arms"].values()))
    g_cons_anchor["pass"] = g_cons_anchor["all_within_bar"]
    metrics["gates"]["G_CONS_ANCHOR"] = g_cons_anchor
    anch_txt = ("PASS" if g_cons_anchor["pass"]
                else "MISS (expected ~half the time — the cross-session "
                     "cons lottery; disclosed session texture)")
    log(f"G_CONS_ANCHOR (non-halting texture): "
        + ", ".join(f"{a} |d| {v['abs_diff']:.4f}"
                    for a, v in g_cons_anchor["arms"].items())
        + f" (bar {CONS_ANCHOR_TOL}): {anch_txt}")
    write_partial("P2b G_CONS_ANCHOR recorded (non-halting)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    L = {a: arms_rec[a]["root"]["g0"] for a in ARM_ORDER}
    order = sorted(ARM_ORDER, key=lambda a: WIDTH_OF[a])   # width order
    floor = L["NOINST"]
    in_band = {a: bool(BAND[0] <= L[a] <= BAND[1]) for a in ARM_ORDER}
    all_in_band = all(in_band.values())
    near_miss = all(BAND[0] - STRADDLE_CUSHION <= L[a]
                    <= BAND[1] + STRADDLE_CUSHION for a in ARM_ORDER)

    # the clean step: adjacent width pair, below-bar -> in-band, stay above
    step = None
    for i in range(len(order) - 1):
        lo, hi = order[i], order[i + 1]
        above = order[i + 1:]
        if (L[lo] < BELOW_BAR and in_band[hi]
                and all(in_band[a] for a in above)):
            step = {"from": lo, "to": hi,
                    "bracket": f"({WIDTH_OF[lo]}, {WIDTH_OF[hi]}]"}
            break

    # the smeared step: an arm in the mid-band adjacent to an in-band arm
    smeared = None
    for i in range(len(order) - 1):
        lo, hi = order[i], order[i + 1]
        if (BELOW_BAR <= L[lo] < BAND[0] and in_band[hi]):
            smeared = {"from": lo, "to": hi}
            break

    seeded_vals = [L[a] for a in SEEDED]
    monotone = all(seeded_vals[i] <= seeded_vals[i + 1] + 1e-12
                   for i in range(len(seeded_vals) - 1))
    # Spearman of landing vs width across the seeded arms (ties -> avg)
    def _rank(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for kk in range(i, j + 1):
                r[idx[kk]] = avg
            i = j + 1
        return r
    rw = _rank([float(WIDTH_OF[a]) for a in SEEDED])
    rl = _rank(seeded_vals)
    mw = sum((rw[i] - sum(rw) / 4) * (rl[i] - sum(rl) / 4)
             for i in range(4))
    dw = sum((rw[i] - sum(rw) / 4) ** 2 for i in range(4))
    dl = sum((rl[i] - sum(rl) / 4) ** 2 for i in range(4))
    spearman_seeded = float(mw / math.sqrt(dw * dl)) if dw > 0 and dl > 0 \
        else float("nan")

    p281a_clause = bool(floor < BELOW_BAR)
    p281b_clause = bool(floor >= BAND[0])

    hard = {k: v for k, v in metrics["gates"].items()
            if k != "G_CONS_ANCHOR"}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    five_txt = " | ".join(f"{a}(w={WIDTH_OF[a]}) {L[a]:.4f}"
                          for a in order)

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif all_in_band:
        verdict = "FLAT"
        clause = ("every arm lands in the family's landing band "
                  f"[{BAND[0]}, {BAND[1]}] INCLUDING the no-install zero "
                  f"point — P-281b CONFIRMED (floor {floor:.4f} >= "
                  f"{BAND[0]}): the lane carries zero write information; "
                  "'delivery is free' collapses to 'the cons teaches from "
                  "anything'; T246's rehearsal-lane reading and W042's "
                  "bridge lose their object; the landing read is a "
                  f"property of the CONS alone — the five points: "
                  f"{five_txt}")
    elif p281a_clause and step is not None:
        verdict = "THRESHOLDED"
        clause = (f"the zero point lands {floor:.4f} < {BELOW_BAR} AND the "
                  f"curve steps with seed width at {step['bracket']} "
                  f"(L({step['from']}) {L[step['from']]:.4f} -> "
                  f"L({step['to']}) {L[step['to']]:.4f}, all arms above "
                  "in-band) — P-281a's clause on the floor CONFIRMED "
                  "(the full P-281a world adds the retrieval-floor "
                  "reading): delivery needs a seed; the retrieval floor "
                  "is a second capacity number located in "
                  f"{step['bracket']} — the five points: {five_txt}")
    elif p281a_clause and monotone and any(not in_band[a]
                                           for a in SEEDED):
        verdict = "MASS-SCALED"
        clause = (f"the zero point lands {floor:.4f} < {BELOW_BAR} but the "
                  "seeded arms' landings track write width continuously "
                  f"(no step; monotone, Spearman {spearman_seeded:.2f}; "
                  "at least one seeded arm below the band) — the five "
                  f"points: {five_txt}")
    else:
        verdict = "MIXED"
        why = []
        if BELOW_BAR <= floor < BAND[0]:
            why.append(f"the floor {floor:.4f} lands in the mid-band "
                       f"[{BELOW_BAR}, {BAND[0]}) — the registered "
                       "mid-band branch")
        if floor >= BAND[0] and not all_in_band:
            out = [a for a in order if not in_band[a]]
            why.append(f"the floor {floor:.4f} is in-band but "
                       f"{', '.join(out)} land out of band")
        if p281a_clause and smeared is not None and step is None:
            why.append(f"the step is SMEARED: L({smeared['from']}) "
                       f"{L[smeared['from']]:.4f} sits mid-band adjacent "
                       f"to in-band L({smeared['to']}) "
                       f"{L[smeared['to']]:.4f}")
        if p281a_clause and not monotone:
            why.append("the seeded landings are non-monotone in width")
        if near_miss and not all_in_band:
            why.append(f"all five points sit within "
                       f"+-{STRADDLE_CUSHION} of the band "
                       f"[{BAND[0]}, {BAND[1]}] but not all inside (the "
                       "band-edge branch)")
        if not why:
            why.append("the five-point shape matches no registered bar "
                       "cleanly")
        clause = ("; ".join(why) + " — the five points verbatim, no "
                  f"inflation: {five_txt}")

    log("=" * 78)
    log(f"E281 VERDICT: {verdict}")
    for a in order:
        anch = (f" | committed landing {LANDING_ANCHOR[a]:.4f} "
                f"(|d| {abs(L[a] - LANDING_ANCHOR[a]):.4f})"
                if LANDING_ANCHOR[a] is not None else "")
        log(f"  {a:7s} (w={WIDTH_OF[a]:6d}): post g0 "
            f"{arms_rec[a]['seed']['post_g0']:.6f} -> ROOT g0 {L[a]:.4f} "
            f"[{'in-band' if in_band[a] else 'OUT'}]{anch}")
    log(f"  floor (the cons-only zero point): {floor:.4f} | P-281a clause "
        f"(floor < {BELOW_BAR}): {'FIRES' if p281a_clause else 'no'} | "
        f"P-281b clause (floor >= {BAND[0]}): "
        f"{'FIRES' if p281b_clause else 'no'}")
    log(f"  seeded curve: monotone={monotone} Spearman="
        f"{spearman_seeded:.2f} | clean step="
        f"{step['bracket'] if step else None} | smeared="
        f"{(smeared['from'], smeared['to']) if smeared else None}")
    log(f"  {clause}")
    log("=" * 78)

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> MIXED -> FLAT -> THRESHOLDED -> "
                           "MASS-SCALED (frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "landing_curve": {a: {"width": WIDTH_OF[a],
                                  "post_g0_seed":
                                      arms_rec[a]["seed"]["post_g0"],
                                  "root_g0_landing": L[a],
                                  "seed_displacement_l2":
                                      arms_rec[a]["seed"]
                                      ["displacement_l2"],
                                  "in_band": in_band[a],
                                  "committed_landing_anchor":
                                      LANDING_ANCHOR[a]}
                             for a in order},
            "floor_no_install": floor,
            "floor_band": BAND, "floor_below_bar": BELOW_BAR,
            "band_ctx_g1c": list(BAND_CTX),
            "p281a_clause_floor_below_045": p281a_clause,
            "p281b_clause_floor_ge_065": p281b_clause,
            "all_in_band": all_in_band,
            "clean_step": step, "smeared_step": smeared,
            "seeded_monotone": monotone,
            "seeded_spearman_vs_width": spearman_seeded,
            "near_miss_all_within_cushion": near_miss,
            "five_points_verbatim": five_txt,
            "formation_overlay_committed": {
                "serial_post_g0": {"0": 0.0, "10_e246": E246_R10_POST,
                                   **{str(k): v for k, v
                                      in E264_CURVE_POST.items()},
                                   "1000_K1KM": E272_K1KM_POST,
                                   "2000_e272": E272_K2K_POST,
                                   "5000_e272": E272_K5K_POST},
                "serial_root_g0_context": {str(k): v for k, v
                                           in E264_CURVE_ROOT.items()},
                "seed_states_own_posts": {"K1KM": E272_K1KM_POST,
                                          "K10KD": E268_K10KD_POST,
                                          "K100KD": E270_K100KD_POST},
                "note": "e272's formation curve (the WRITE's own reads; "
                        "edge bracketed at (1k, 2k]) — this cell's "
                        "landing curve is its complement on the same "
                        "width axis"},
            "root_read_status": "the LANDING read is this cell's REGISTERED "
                                "PRIMARY (the dispatch's own object — the "
                                "rehearsal dose-response); the family's "
                                "survival convention (landing carried, "
                                "never adjudicated as survival) is "
                                "honored by construction: no arm's "
                                "landing is read as its WRITE's survival",
        },
        "scatter_disclosure": {
            "cons_cross_session_root_g0_family": 0.0419,
            "within_session": "all five landings share ONE cons stream in "
                              "ONE session (bit-identical draws) — the "
                              "curve's SHAPE is immune to the cross-"
                              "session lottery; the level carries it",
            "g_cons_anchor": {a: {"mine": v["mine"],
                                  "committed": v["committed"],
                                  "abs_diff": v["abs_diff"]}
                              for a, v in g_cons_anchor["arms"].items()},
            "n1_per_arm": "n=1 per arm, one session; the CURVE's shape is "
                          "the registered object",
        },
        "verdict": verdict,
        "clause": clause,
    }
    write_partial("P7 ADJUDICATED (the frozen bars)")

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("all five arms share the BIT-IDENTICAL "
            "cons stream (one generator, seed 10901, one draw order, one "
            "session) — the ONLY delta across arms is the seed state "
            "(the write's presence and width); the loaded seed states are "
            "the committed arms themselves (md5 + behavioral binds), and "
            "the fresh R10 install uses the family's standard install "
            "stream (gen 24314, bit-identical draws): a landing "
            "difference is the seed state's doing or nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage, one session; the "
            "CURVE's shape is the registered object, not any single "
            "point; the landing lottery (root g-12, the height lottery) "
            "is reported, never adjudicated"),
        "cons_scatter_carried": ("the g0 ruler is the stable one; the "
            "cross-session cons lottery family (~0.0419) is disclosed in "
            "G_CONS_ANCHOR and CANNOT bend the within-session curve's "
            "shape — only its level; the committed anchors (0.6879/0.7119/"
            "0.8103) are context, never bar inputs"),
        "loads_measured_not_nominal": ("every arm's seed displacement ||d|| "
            "(the write-mass proxy), v-excess + cos-to-span at seed and "
            "root (vs e258's LOADED v-map and e246's committed LATE "
            "span), and the full cons trajectories are reported — never "
            "nominal"),
        "the_rehearsal_lane_caveat_honored": ("T246's finding is the "
            "cell's own OBJECT (not a caveat to dodge): the landing read "
            "IS the rehearsal lane's dose-response; no arm's landing is "
            "read as its write's survival (the family's convention kept)"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all four branches and "
            "the five points are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "machinery": {
            "cons": "e261's chunked_consolidate VERBATIM BY IMPORT (e113; "
                    f"seed {E261.CONS_SEED} HELD, NATURAL, bit-identical "
                    "draws across all five arms)",
            "install_r10": "e261's chunked_install VERBATIM BY IMPORT "
                           "(E43.exposure Dmix; the dense-room hook; the "
                           "thermal envelope) at the fresh R10 room — the "
                           "committed lab/e261_rank_ladder.py unmodified",
            "seed_states": "LOADED committed checkpoints (e272_K1KM / "
                           "e268_CONCURRENT / e270_CONCURRENT inst_resume "
                           "at s400; fresh md5 + behavioral binds) — no "
                           "regeneration",
        },
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "seed_states": {a: f"runs/checkpoints/{seed_files[a][0]}"
                            for a in seed_files},
            "rooms": rooms_ck,
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARM_ORDER},
            "cons_resumes": {a: (f"runs/checkpoints/"
                                 f"{'smoke_' if SMOKE else ''}e281_{a}"
                                 f"_cons_resume.pt") for a in ARM_ORDER},
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": {
            "burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
            "per_step_polls": "after EVERY opt step (all drivers)",
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
    make_dose_plot(RD, arms_rec, L, order, in_band, floor, verdict, clause,
                   step, thermal_log)
    make_instrument_plot(RD, arms_rec, thermal_log)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e281_rehearsal_dose.png"),
                          str(RD / "e281_instrument.png"),
                          str(RD / "REPORT.md")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    write_report(RD, verdict, clause, L, order, in_band, floor, step,
                 five_txt, gates_pass, g_cons_anchor, thermal_log,
                 arms_rec)
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def _wx(w: int) -> float:
    """Plot coordinate for a width: log10-scale, the zero point at x=1."""
    return 1.0 if w <= 0 else float(w)


def make_dose_plot(rd, arms_rec, L, order, in_band, floor, verdict, clause,
                   step, thermal_log):
    """THE CELL'S HEADLINE FIGURE: the five-point dose-response curve with
    the formation curve overlaid (the complement story), the P-281a/b
    zones, the cons trajectories, and the thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    cols = {"NOINST": "black", "R10": "tab:purple", "K1KM": "tab:blue",
            "K10KD": "tab:orange", "K100KD": "tab:red"}
    names = {"NOINST": "NO-INSTALL (0)", "R10": "RANK-10 (10)",
             "K1KM": "K1KM (1k)", "K10KD": "10K-DEAD (10k)",
             "K100KD": "100K-DEAD (100k)"}

    # (0,0) THE DOSE-RESPONSE CURVE (the five landings vs width)
    ax = axes[0, 0]
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.10,
               label=f"the family's landing band [{BAND[0]}, {BAND[1]}]")
    ax.axhline(BELOW_BAR, color="crimson", ls="--", lw=1.2,
               label=f"the {BELOW_BAR} below-bar (P-281a's clause)")
    ax.axhline(E261.G1C_ROOT_G0, color="dimgray", ls=":", lw=1.0,
               label=f"g1c committed root g0 {E261.G1C_ROOT_G0:.4f}")
    xs = [_wx(WIDTH_OF[a]) for a in order]
    ys = [L[a] for a in order]
    ax.plot(xs, ys, "o-", lw=1.8, ms=7, color="tab:green", alpha=0.9,
            label="the landing curve (this cell, one session)")
    for a in order:
        ax.annotate(f"{L[a]:.4f}", (_wx(WIDTH_OF[a]), L[a]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
        if LANDING_ANCHOR[a] is not None:
            ax.plot([_wx(WIDTH_OF[a])], [LANDING_ANCHOR[a]], "o", ms=5,
                    mfc="none", mec=cols[a], mew=1.4, alpha=0.9)
    ax.plot([], [], "o", ms=5, mfc="none", mec="gray",
            label="committed landings (their own sessions; anchors)")
    if step is not None:
        ax.annotate("the STEP", (_wx(WIDTH_OF[step["to"]]),
                                 L[step["to"]]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=9, color="crimson", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[a] for a in order], fontsize=8.0)
    ax.set_xlabel("the seed state's write width (dims; the zero point at "
                  "x=1)")
    ax.set_ylabel("root g0 (the landing read)")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("THE REHEARSAL DOSE-RESPONSE — the landing read vs the "
                 "seed write (one cons stream, one session)", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="lower right")
    ax.grid(alpha=0.25)

    # (0,1) THE FORMATION CURVE OVERLAID (e272's complement)
    ax = axes[0, 1]
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.10)
    form_pts = [(0, 0.0), (10, E246_R10_POST),
                (1000, E264_CURVE_POST[1000]),
                (1000, E272_K1KM_POST), (2000, E272_K2K_POST),
                (5000, E272_K5K_POST), (10000, E264_CURVE_POST[10000]),
                (40000, E264_CURVE_POST[40000]),
                (100000, E264_CURVE_POST[100000]),
                (237123, E264_CURVE_POST[237123])]
    fx = [_wx(w) for w, _ in form_pts]
    fy = [v for _, v in form_pts]
    ax.plot(fx, fy, "s-", lw=1.4, ms=5, color="tab:blue", alpha=0.85,
            label="FORMATION (serial post g0; e261/e264/e272/e246, "
                  "committed; edge at (1k, 2k])")
    sx = [_wx(WIDTH_OF[a]) for a in order]
    sy = [arms_rec[a]["seed"]["post_g0"] for a in order]
    ax.plot(sx, sy, "o-", lw=1.6, ms=6, color="tab:red", alpha=0.9,
            label="THIS cell's five seed states' post g0 (the writes the "
                  "cons starts from)")
    lx = [_wx(WIDTH_OF[a]) for a in order]
    ly = [L[a] for a in order]
    ax.plot(lx, ly, "o-", lw=1.8, ms=6, color="tab:green", alpha=0.9,
            label="the LANDING curve (this cell)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xticks(sorted(set(fx + sx)))
    ax.set_xticklabels([str(int(t)) for t in sorted(set(fx + sx))],
                       fontsize=7.5)
    ax.set_xlabel("write width (dims; 0 at x=1)")
    ax.set_ylabel("g0 (symlog): post = the write; root = the landing")
    ax.set_title("THE COMPLEMENT — formation (the write's own curve; steps "
                 "at (1k,2k]) vs retrieval (the landing curve)", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="upper left")
    ax.grid(alpha=0.25)

    # (1,0) THE CONS TRAJECTORIES (the rehearsal lane made visible)
    ax = axes[1, 0]
    for a in order:
        tr = arms_rec[a]["consolidation"]["traj"]
        ax.plot([t["step"] for t in tr], [t["g0_pz"] for t in tr], "o-",
                lw=1.4, ms=4, color=cols[a], label=names[a])
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.10)
    ax.axhline(E261.G1C_ROOT_G0, color="black", ls=":", lw=1.2,
               label=f"committed g1c root g0 {E261.G1C_ROOT_G0:.4f}")
    ax.set_xlabel("cons step (s300, seed 10901 — bit-identical across "
                  "arms)")
    ax.set_ylabel("g0 battery")
    ax.set_title("THE REHEARSAL LANE — all five cons trajectories (the "
                 "ONLY delta across arms is the seed state)", fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

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
                 f"{sum(1 for r in thermal_log if r['temp'] >= E261.TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle("E281 — THE REHEARSAL DOSE-RESPONSE (the retrieval "
                 f"story's lynchpin) -> {verdict}", fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 150), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e281_rehearsal_dose.png", dpi=130)
    plt.close(fig)


def make_instrument_plot(rd, arms_rec, thermal_log):
    """THE INSTRUMENT PAGE: the seed-state binds, the measured loads, the
    floor-vs-seeded contrast, and the seed displacements."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    order = sorted(arms_rec.keys(), key=lambda a: WIDTH_OF[a])
    cols = {"NOINST": "black", "R10": "tab:purple", "K1KM": "tab:blue",
            "K10KD": "tab:orange", "K100KD": "tab:red"}

    # (0,0) THE SEED-STATE POST READS (the writes the cons starts from)
    ax = axes[0, 0]
    posts = [arms_rec[a]["seed"]["post_g0"] for a in order]
    anchored = [arms_rec[a]["seed"].get("committed_post_g0")
                for a in order]
    xs = np.arange(len(order))
    ax.bar(xs, posts, width=0.55,
           color=[cols[a] for a in order], alpha=0.85)
    for i, av in enumerate(anchored):
        if av is not None:
            ax.plot([xs[i]], [av], "_", ms=14, color="black",
                    mew=1.8, label="" if i > 0 else
                    "committed post (bit-bound)")
    ax.axhline(DEAD_BAR, color="crimson", ls="--", lw=1.2,
               label=f"the dead bar {DEAD_BAR}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\nw={WIDTH_OF[a]}" for a in order], fontsize=8.5)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("seed post g0 (symlog)")
    ax.set_title("THE SEED STATES — the writes the cons starts from "
                 "(all dead; three LOADED bit-bound + one fresh R10 + the "
                 "zero point)", fontsize=9.0)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y")

    # (0,1) THE MEASURED LOADS (v-excess + cos-to-span, seed and root)
    ax = axes[0, 1]
    rows = ["seed v-excess", "seed cos-to-span",
            "root v-excess", "root cos-to-span"]
    keys = [("seed", "v_excess"), ("seed", "cos_to_span"),
            ("root", "v_excess"), ("root", "cos_to_span")]
    xs = np.arange(len(rows))
    for i, a in enumerate(order):
        vals = [arms_rec[a][ph]["displacement_loads"][key]
                for ph, key in keys]
        ax.bar(xs + (i - 2) * 0.16, vals, width=0.15, alpha=0.85,
               color=cols[a], label=a)
    ax.set_xticks(xs)
    ax.set_xticklabels(rows, fontsize=8.5)
    ax.set_ylabel("load (measured)")
    ax.set_title("THE MEASURED LOADS (vs e258's v-map + e246's LATE span; "
                 "seed = the write's displacement, root = post-cons)",
                 fontsize=9.0)
    ax.legend(fontsize=7.2, ncols=5)
    ax.grid(alpha=0.25, axis="y")

    # (1,0) THE FIVE LANDINGS (bars; band + below-bar + anchors)
    ax = axes[1, 0]
    L = {a: arms_rec[a]["root"]["g0"] for a in order}
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.10,
               label=f"band [{BAND[0]}, {BAND[1]}]")
    ax.axhline(BELOW_BAR, color="crimson", ls="--", lw=1.2,
               label=f"below-bar {BELOW_BAR}")
    ax.bar(np.arange(len(order)), [L[a] for a in order], width=0.55,
           color=[cols[a] for a in order], alpha=0.85)
    for i, a in enumerate(order):
        if LANDING_ANCHOR[a] is not None:
            ax.plot([i], [LANDING_ANCHOR[a]], "_", ms=16, color="black",
                    mew=1.8, label="" if (a != "K1KM") else
                    "committed landings (anchors, non-halting)")
        ax.annotate(f"{L[a]:.4f}", (i, L[a]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{a}\nw={WIDTH_OF[a]}" for a in order], fontsize=8.5)
    ax.set_ylabel("root g0 (the landing read)")
    ax.set_title("THE FIVE LANDINGS (the registered primary; one session, "
                 "one cons stream)", fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y")

    # (1,1) THE SEED DISPLACEMENTS (the write-mass proxy)
    ax = axes[1, 1]
    dl = [arms_rec[a]["seed"]["displacement_l2"] for a in order]
    dr = [arms_rec[a]["root"]["displacement_l2"] for a in order]
    xs = np.arange(len(order))
    ax.bar(xs - 0.19, dl, width=0.38, alpha=0.85,
           color=[cols[a] for a in order], label="seed ||d|| (the write)")
    ax.bar(xs + 0.19, dr, width=0.38, alpha=0.5,
           color=[cols[a] for a in order], label="root ||d|| (post-cons)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\nw={WIDTH_OF[a]}" for a in order], fontsize=8.5)
    ax.set_ylabel("displacement L2 vs base")
    ax.set_title("THE WRITE-MASS PROXY — ||d_seed|| vs ||d_root|| (the "
                 "MASS-SCALED branch's axis, measured)", fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("E281 — the instrument page: the seed states, the loads, "
                 "the landings, the write-mass proxy", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "e281_instrument.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ report
def write_report(rd, verdict, clause, L, order, in_band, floor, step,
                 five_txt, gates_pass, g_cons_anchor, thermal_log,
                 arms_rec) -> None:
    lines = []
    a = lines.append
    a("# E281 — THE REHEARSAL DOSE-RESPONSE (the retrieval story's "
      "lynchpin)")
    a("")
    a(f"**VERDICT: {verdict}** "
      + ("(all hard gates PASS)" if gates_pass
         else "(TEXTURE — a hard gate failed)"))
    a("")
    a(f"{clause}")
    a("")
    a("## The five points (one cons stream, one session)")
    a("")
    a("| arm | width | seed post g0 | ROOT g0 (the landing) | in band | "
      "committed landing (anchor) |")
    a("|---|---|---|---|---|---|")
    for arm in order:
        anch = (f"{LANDING_ANCHOR[arm]:.4f} (|d| "
                f"{abs(L[arm] - LANDING_ANCHOR[arm]):.4f})"
                if LANDING_ANCHOR[arm] is not None else "— (fresh arm)")
        a(f"| {arm} | {WIDTH_OF[arm]} | "
          f"{arms_rec[arm]['seed']['post_g0']:.6f} | **{L[arm]:.4f}** | "
          f"{'YES' if in_band[arm] else 'no'} | {anch} |")
    a("")
    a(f"- **the floor (the cons-only zero point): {floor:.4f}** — "
      f"P-281a clause (floor < {BELOW_BAR}): "
      f"{'FIRES' if floor < BELOW_BAR else 'does not fire'}; P-281b "
      f"clause (floor >= {BAND[0]}): "
      f"{'FIRES' if floor >= BAND[0] else 'does not fire'}")
    a(f"- the registered bars: FLAT / THRESHOLDED / MASS-SCALED / MIXED "
      "(verbatim in the script docstring + metrics.registered)")
    a("- the formation overlay: e272's serial curve (edge at (1k, 2k]) — "
      "this cell's landing curve is its complement on the same axis")
    a("")
    a("## The seed states: checkpointed vs regenerated")
    a("")
    a("- **ALL THREE loaded — CHECKPOINTED, no regeneration**: "
      "`e272_K1KM_inst_resume.pt` (md5 2306a448..., step 400), "
      "`e268_CONCURRENT_inst_resume.pt` (md5 9b66f010..., step 400), "
      "`e270_CONCURRENT_inst_resume.pt` (md5 59eeed2b..., step 400) — "
      "each is its cell's committed FINAL state, behaviorally bound "
      "(|d post g0| vs the committed metrics measured live; the birth "
      "desk-check read 0.0 bit-exact on all three; x10's fresh-md5 "
      "convention — no committed md5 exists for resume files).")
    a("- arm (a) NO-INSTALL: the fresh `e001.pt` base (md5 "
      "d114536d..., x6's committed bind) — no install at all.")
    a("- arm (b) RANK-10: a FRESH install this session (room k=10, "
      "seeds 28111/28112 registered fresh; dead gated "
      f"(post g0 {arms_rec['R10']['seed']['post_g0']:.2e} < "
      f"{DEAD_BAR}); e246's ALIGNED 2.86e-5 the precedent context).")
    a("")
    a("## Gates")
    a("")
    a("- HARD: G_NAMEFREE / G_SPLICE / G_BATTERY / G_ANCHOR / G_INSTMASK "
      "/ G_PARENTS / G_BASE / G_ROOT / G_VMBIND / G_SPANBIND / G_PROJ / "
      "G_SEEDSTATE / G_R10DEAD — all PASS (a failure would have halted).")
    a(f"- NON-HALTING G_CONS_ANCHOR: "
      + ", ".join(f"{k} |d| {v['abs_diff']:.4f}"
                  for k, v in g_cons_anchor["arms"].items())
      + f" (bar {CONS_ANCHOR_TOL}; the cross-session cons lottery family "
        "~0.0419 — expected to miss ~half the time; never a bar input).")
    a("")
    a("## Disclosures")
    a("")
    a("- The dispatch letter anticipated REGENERATING e268's/e270's dead "
      "writes if not checkpointed; the resume checkpoints ARE the "
      "committed final states (step 400 + traj last step 400) — loaded, "
      "not re-run (the ~5 GPU min unspent).")
    a("- The landing band has two near-coincident forms: the letter's "
      f"[{BAND[0]}, {BAND[1]}] (HARD, adjudicating) and the family's "
      f"+-10%-of-g1c [{BAND_CTX[0]:.4f}, {BAND_CTX[1]:.4f}] (context); "
      "any adjudication-relevant point between them is a named band-edge "
      "branch.")
    a("- n=1 per arm, one session; the curve's SHAPE is the registered "
      "object — the within-session design makes it immune to the "
      "cross-session cons lottery that scattered the prior landing "
      "reads (0.577-0.810) across sessions.")
    a("- No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).")
    a("")
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    a(f"## Envelope: max temp {mx:.1f}C over {len(thermal_log)} per-step "
      f"polls; violations >= {E261.TEMP_HARD:.0f}C: "
      f"{sum(1 for r in thermal_log if r['temp'] >= E261.TEMP_HARD)}; "
      f"bursts <= {E261.BURST_MAX_S:.0f}s, cooldowns "
      f"{E261.COOLDOWN_S:.0f}s (tags e281:ARM:phase).")
    a("")
    (rd / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
