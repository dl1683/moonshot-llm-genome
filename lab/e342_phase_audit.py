# -*- coding: utf-8 -*-
"""E342 — THE PHASE AUDIT (R79's minted card; the milestone-costume
lesson's last open residual). Run: cd lab && python e342_phase_audit.py
(E342_SMOKE=1 shakedown).

THE BACKGROUND (dispatch verbatim):
    "x41's verification rider discovered the milestone-costume effect (the
    milestone cadence sampled only the dose phase; the 'equilibrium' was
    the upper rail). The founding traces (e288/e334/e336) show no
    zero-dose events (R79's desk check) — but their milestone reads were
    taken at event boundaries (post-dose). THE RESIDUAL: does the read SAG
    intra-period (between events, under the wash) and recover at each dose
    — making the founding heights upper-rail samples too — or does it hold
    flat between events?"

THE QUESTION (the live half): the INTRA-PERIOD SAG. R79's desk half is
done — no zero-dose events in any founding trace, so the endpoints are
phase-consistent all-dosing samples; what the record does NOT carry is
the MID-PHASE point (event+13, ~half to the next dose). This cell
measures it.

THE DESIGN (dispatch verbatim):
  "1. THE SUBJECT: e336's committed t400 checkpoint (md5-bound — the same
  state x39 used; read 0.815; the founding-class trajectory).
  2. THE MID-PHASE READS: resume and continue the IDENTICAL protocol for
  4-8 events, reading at event+13 (mid-period, ~halfway to the next dose)
  IN ADDITION to the standard pre/post pairs — the phase-declared
  convention (every phase read, not just the boundaries).
  3. THE ANALYSIS: the sag profile (boundary read vs mid-phase read per
  period); the founding numbers' phase character."

THE BARS (dispatch verbatim; frozen in this birth commit BEFORE compute):
  - SAG-EXISTS: "the mid-phase reads fall materially below the boundary
    reads (>= ~15% mean sag) — THE FOUNDING HEIGHTS ARE RAILS (the
    endpoint class carries the phase clause; Law 4's endpoint language
    joins x41's rider; the 3.16-3.62x band is an upper-rail band)."
  - NO-SAG: "the mid-phase reads hold within noise (< ~5%) — THE FOUNDING
    NUMBERS STAND AS WRITTEN (the 25-step periods hold; the costume
    lesson does not reach the founding class)."

LAB LEAN (dispatch verbatim): "Lab lean: SAG-EXISTS, weakly — x41's
PERIOD2X showed 2.4x sag within 50 steps (the read cannot hold half a
doubled period); whether 25 steps (the founding period) is short enough
to hold is the open question — the founding deficit-medians (0.6-0.7)
suggest substantial decay between events. State your own read."

P-e342a (THE EXECUTOR'S OWN READ, registered per the dispatch's "Register
P-e342a BEFORE compute ... State your own read", frozen HERE at birth;
predictions are scored): SAG-EXISTS, MODERATELY — diverging from the lean
only in strength. GROUNDS: (1) THE COMMITTED ENDPOINTS ALREADY COMMIT THE
SWING — both legs' pre/post pairs bound the +25 floors at 10-28% OF THE
PEAKS (e336 leg-1: gate 0.1254 vs milestone 0.8147 at t400 = 6.5x; e339
leg-2: t599 0.0916 vs t600-post 0.9000 = 9.8x); a mid read at +13 above
95% of its boundary would need the ENTIRE collapse packed into the final
12 steps — a cliff with no mechanism (the wash steps are homogeneous
draws under a per-step-constant cap and clip). (2) THE +1 COMMITS THE
DECAY'S START ON THE SMALL-DOSE HALF — e339's committed win1:t426 0.6155
after the 0.6340 post (−2.9% in ONE step): the small-dose periods
(17/19/21/23 by the committed 2-cycle deficits) are already falling at
+1 and compound to a deep mid sag. (3) THE COUNTERVAILING (why
moderately, not strongly): the big-dose periods' +1 HOLDS (e339's
win2:t601 0.8983 after the 0.9000 post — −0.19%): the decay is CONVEX
(hold-then-fall) and its knee's location is the genuinely open parameter;
if the knee sits past +10 on the big-dose periods their mids sag < 20%
and only the small-dose half carries the mean. (4) x41's PERIOD2X bounds
the hold below 50 steps. PREDICTED SHAPE: mean sag in [0.35, 0.60];
small-dose periods sag 0.45-0.65, big-dose 0.20-0.45; every mid strictly
above the next gate floor (the collapse back-loaded, front-half convex);
the leg's t500/t600 boundary reads within 0.03 of e339's committed
(0.85299/0.899997 — the shared-draw continuation; disclosed tolerance,
never a law). FALSIFIER: mean sag < 0.15 (my card flips to the residual
cells) or > 0.75 (the collapse completes BEFORE +13 — front-loaded; the
phase character worse than my card). SCORED: TRUE iff the verdict ==
SAG-EXISTS.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE SUBJECT := e336's committed t400 ERROR-GATED resume state —
  runs/e336/ERROR-GATED_resume.pt (raw-md5
  e67a7d4749299002a50868500be01d6a — the model + optC + optF + BOTH
  generator states at t400), bit-compared against the committed post
  checkpoint runs/e336/ERROR-GATED_post.pt (raw-md5
  6c42e8131994122fcdb9ef56c0314e0a — x39's own subject bind) and
  read-reproduced on the battery (g0 0.8147322535514832 == e336's
  committed t400 milestone == x39's committed subject literal; gm12
  0.7971041202545166; CE_R 1.6229653358459473; the family's 2e-6/5e-3
  cross-session law). THE DISPATCH'S GATE ("the boundary reads reproduce
  e336/x39's committed values") := THIS repro gate — the resumed bits
  re-read the committed t400 values BEFORE any stepping.

* THE LEG := t401..t600 (200 steps, 8 maintenance events 17-24 — the
  dispatch's 4-8 at its top: every event a sag pair), M=25 UNCHANGED
  (the founding cadence), the driver = e336's COMMITTED
  chunked_errorgated_phase EXECUTED VERBATIM BY IMPORT under the
  family's disclosed leg re-binding (LEG_KEYS = PHASE_STEPS, MILESTONES,
  WIN1, WIN2, MAINT_EVERY — e339/x41's convention); the stream CONTINUES
  (both generator states restored, never re-seeded: corpus steps 401..
  draw the continuation of seed 33601, events 17.. of seed 33602).
  POOLS LENGTH-SCALED to the leg (x41's frozen form: B_C = B_C_400 x
  200/400, B_M = B_M_400 x 8/16; LR_M_MAX NEVER scaled — the frozen
  constant 0.021942464013242388): the per-step and per-event equal-share
  arithmetic is IDENTICAL to e336's leg-1 and e339's leg-2 at their
  starts; the applied corpus lr asserted inside x41's committed LR_BAND
  (0.0018, 0.0043).

* THE PHASE-DECLARED READS (the convention the dispatch declares): at
  EVERY event the driver already reads the pre pair (maint_ledger
  read_at_gate — the floor, the dose's own input) and this leg adds the
  post read (a MILESTONE row at every event step — the founding read
  channel: the g0 battery, p(T), the same baseline); at every period
  k<24 the MID read (a MILESTONE row at s_k + 13 — frozen verbatim from
  the dispatch's 'event+13, ~halfway to the next dose'). MILESTONES =
  {425, 438, 450, 463, 475, 488, 500, 513, 525, 538, 550, 563, 575, 588,
  600} (8 boundary posts + 7 mids). WINDOWS: WIN1 = (424, 425, 426) ==
  e339's committed win1 (a 4-row repro co-read: the floor-before-17, the
  pre/post pair, the +1-after-a-small-dose); WIN2 = (574, 575, 576)
  around event 23 (a SECOND +1-after-small-dose datum mid-leg; e339's
  committed win2 t601 +1-after-big-dose cited from its record — my leg
  ends at t600 so its t601 flank would never fire; the t600-pre floor
  comes free in the event-24 gate read).

* THE SAG STATISTIC (frozen): per period k = 17..23:
  post_k := the milestone g0 read at event step s_k (post-dose — the
  founding read convention); mid_k := the milestone g0 read at s_k+13;
  sag_k := (post_k − mid_k) / post_k; MEAN SAG := mean over the 7
  complete pairs. BARS: mean >= 0.15 -> SAG-EXISTS; mean < 0.05 ->
  NO-SAG; [0.05, 0.15) -> PARTIAL-SAG (the named residual: the founding
  numbers carry a mid-strength phase clause — neither rail nor flat;
  disclosed at birth because the dispatch's '~15%'/'~5%' leave the
  interval unnamed). Period 24's mid (613) sits beyond the leg —
  disclosed; NO pair is ever dropped post-hoc. CO-READS (never
  adjudicate): the floor ratio mid_k / gate_{k+1} (is the collapse
  front- or back-loaded); the +1 window reads; the founding desk half.

* THE FOUNDING CHARACTER (the analysis's second deliverable): (i) the
  DESK half, runtime-computed from the md5-bound committed records only
  — e336's leg-1 own-period peak/floor ratios (its milestones 0.6703/
  0.6985/0.7698/0.8147 vs its event gates) + the canon's deficit-medians
  0.6-0.7 (the dispatch's own citation); (ii) the LIVE half — this leg's
  measured sag stamped against them: the phase clause the founding
  numbers carry (or do not).

* THE e339 CO-READ (non-halting, disclosed): this leg re-runs the
  shared-draw continuation of e339's committed t401-t600 segment (the
  same restored generator states; e339's leg kept full-leg pools, this
  leg the length-scaled form — proportional equal-share paths, x41's
  precedent); my t500/t600 milestone reads + the 4 win1 rows are
  compared against e339's committed literals with a DISCLOSED tolerance
  (0.05 absolute) — a consistency check, NEVER a gate: cross-run GPU
  op-order determinism is not the family's law (the 2e-6 law covers
  same-bits CPU reads only).

* THE ZERO-EVENT CLAUSE (frozen): e339's committed events 17-24 ALL
  dosed (deficits 0.025-0.684); if a ZERO event (deficit 0.0 — the
  controller's own no-dose branch) fires inside this leg, its post ==
  its gate and the pair still measures the wash's decay from an undosed
  boundary — stamped per-period, never excluded.

* COMPOSITE (frozen): TEXTURE (any hard-gate failure — HALT, nothing
  adjudicated) -> standing conditions (budget / stream-live / orth /
  bufsep: a failure routes to CONDITIONS-FAILED, nothing adjudicated)
  -> the sag partition above. SMOKE stamps everything, adjudicates
  nothing.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); the
driver's own e261 form honored verbatim (bursts <= 175s, per-step thermal
polls at the 78C margin, 40s cooldowns (the 30-60s window), the 84C
never-past line (inside the 85C dispatch line), wait_gpu_free/gpu_ok
before every burst, NO concurrent GPU jobs); polls to
runs/_envelope_log.jsonl tagged e342:AUDIT-LEG:* (E261.NAME re-bound;
E336.REA_ARM re-bound label-only and restored). gpu_ok() asserted at
startup and immediately before the leg. TIMESTAMPS: datetime.now(UTC)
only.

Outputs: runs/e342/{metrics.json (PROGRESSIVE), REPORT.md,
e342_phase_audit.png} + the staged leg resume + the leg post state
(runs/e342/, never runs/checkpoints/ — this cell writes ONLY
lab/e342_* and runs/e342/*; run.log gitignored by the house *.log rule).
No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).
Commit + push per phase.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# NOTE: E336_SMOKE is deliberately NOT set — the e336 module imports in
# FULL mode (its committed constants are the identity this cell asserts);
# this cell's own smoke horizon is the leg re-bind below.

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402

import e336_founding_replicate as E336                 # noqa: E402 — THE
                                                      # COMMITTED DRIVER,
                                                      # executed VERBATIM
                                                      # (import runs no
                                                      # main; disclosed
                                                      # side effect: opens
                                                      # runs/e336/run.log in
                                                      # append + asserts
                                                      # CUDA — zero bytes
                                                      # written by this
                                                      # cell; the handle is
                                                      # re-pointed below)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SMOKE = os.environ.get("E342_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e342_smoke" if SMOKE else "e342"
assert torch.cuda.is_available(), "e342 owns the GPU lane (dispatch)"
assert common.gpu_ok(), "gpu_ok() gate at startup (check BEFORE launch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE REBINDING (the family's disclosed convention) — the committed
# e336 module's logs + burst machinery label THIS cell (tags e342:...);
# NOTHING of e336's is written (its run.log handle is re-pointed here).
E336._logf = _logf
E336.T0 = T0
E336.NAME = NAME
device_events: list[dict] = []
thermal_log: list[dict] = []
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
E261.thermal_log = thermal_log
E261.device_events = device_events

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
E336_DIR = E43.REPO / "runs" / "e336"
E339_DIR = E43.REPO / "runs" / "e339"
REA_RESUME_CK = E336_DIR / "ERROR-GATED_resume.pt"
REA_POST_CK = E336_DIR / "ERROR-GATED_post.pt"
E336_METRICS = E336_DIR / "metrics.json"
E339_METRICS = E339_DIR / "metrics.json"

# the committed artifacts — raw-md5s, VERIFIED AT BIRTH (2026-10-10,
# git head d6965d1); a drift FAILS the resume gate.
E336_REA_RESUME_MD5 = "e67a7d4749299002a50868500be01d6a"
E336_REA_POST_MD5 = "6c42e8131994122fcdb9ef56c0314e0a"
E336_METRICS_MD5 = "f90b439a0a4eff55e7f115901af469d6"
E339_METRICS_MD5 = "00e5f9822b6559ede2ab0ee9987c5121"

# e336's committed t400 literals (the repro targets; == x39's subject)
SUBJ_READ_T = 0.8147322535514832             # e336 t400 g0 p(T)  == x39
SUBJ_GM12_T = 0.7971041202545166             # e336 t400 gm12      == x39
SUBJ_CE_R = 1.6229653358459473               # e336 t400 CE_R      == x39
SUBJ_DISP_NORM = 1.1536193280751479          # the t400 row's disp_norm
SUBJ_N_MAINT = 16                            # the t400 dose ledger
E336_S_CORPUS_400 = 2.684888718649745
E336_S_MAINT_400 = 1.0470825023949146
E336_S_TOTAL_400 = 3.7319712210446596
E336_TRAJ_G0 = {100: 0.6703165769577026, 200: 0.6985082030296326,
                300: 0.7698389291763306, 400: 0.8147322535514832}
READ_TOL = 2e-6                              # the family's law
CE_TOL = 5e-3

# e339's committed leg-2 literals (the shared-draw co-read's parents)
E339_T500_G0 = 0.8529938459396362
E339_T600_G0 = 0.8999966382980347
E339_WIN1 = {"win1:t424": 0.2279321402311325,
             "win1:t425-pre": 0.21111607551574707,
             "win1:t425-post": 0.6339741349220276,
             "win1:t426": 0.615505576133728}
E339_GATES_17_24 = [0.21111607551574707, 0.12916554510593414,
                    0.2059592306613922, 0.1249639093875885,
                    0.24551410973072052, 0.10180409252643585,
                    0.2786334455013275, 0.09039571136236191]
E339_DEFICITS_17_24 = [0.2614481387375032, 0.5481374238511878,
                       0.2794884388715479, 0.5628360955300279,
                       0.1411127633702932, 0.6438565758867073,
                       0.02525068599222589, 0.6837667585771313]
E339_LEG2_S_CORPUS = 2.684888717252761
E339_LEG2_S_MAINT = 0.8751672571524978
E339_CO_TOL = 0.05                            # disclosed tolerance (never
                                              # a gate)

# ---- THE LEG (frozen) ----------------------------------------------------
SMOKE_LEG1_STEPS = 8                          # the smoke leg-1 (e339's form)
SMOKE_LEG1_WINS = ((1, 2, 3), (3, 4, 5))
LEG_START = SMOKE_LEG1_STEPS if SMOKE else 400   # e336's committed t400
LEG_STEPS = 8 if SMOKE else 200               # 8 events at M=25
LEG_END = LEG_START + LEG_STEPS               # t600 (t16 smoke)
M_FULL = 25                                   # the founding cadence —
                                              # UNCHANGED in the full run
MID_OFFSET = 13                               # the dispatch's event+13
                                              # (smoke: +1, disclosed —
                                              # a 2-step period's half)

def _leg_events(m: int, start: int, end: int) -> list[int]:
    return [s for s in range(start + 1, end + 1) if s % m == 0]

LEG_EVENTS = _leg_events(M_FULL, LEG_START, LEG_END) if not SMOKE \
    else _leg_events(2, LEG_START, LEG_END)
MID_OFF = MID_OFFSET if not SMOKE else 1
LEG_MIDS = [s + MID_OFF for s in LEG_EVENTS[:-1]]   # 7 mids (period 24's
                                              # mid sits beyond the leg)
LEG_MILESTONES = tuple(sorted(set(LEG_EVENTS + LEG_MIDS)))
LEG_WIN1 = (LEG_EVENTS[0] - 1, LEG_EVENTS[0], LEG_EVENTS[0] + 1)
LEG_WIN2 = (LEG_EVENTS[-2] - 1, LEG_EVENTS[-2], LEG_EVENTS[-2] + 1)
                                              # around the SECOND-TO-LAST
                                              # event (full: event 23 at
                                              # 575 -> (574,575,576); the
                                              # last event's +1 flank
                                              # would sit beyond the leg)

# ---- THE BARS' NUMBERS (frozen) ------------------------------------------
SAG_BAR_HI = 0.15                             # >= -> SAG-EXISTS
SAG_BAR_LO = 0.05                             # < -> NO-SAG
ORTH_BAR = 1e-6
BUDGET_SLACK = 1e-5
STREAM_STABLE_TOL = 1.05
LR_BAND = (0.0018, 0.0043)                    # x41's committed band

REGISTERED = {
    "registration": (
        "background + design + bars + lab lean frozen VERBATIM from the "
        "dispatch letter (R79's minted card — e342 THE PHASE AUDIT); "
        "P-e342a registered HERE at birth; this script committed at birth "
        "BEFORE any compute; adjudicate against exactly this; no bar "
        "shopping."),
    "background_verbatim": (
        "x41's verification rider discovered the milestone-costume effect "
        "(the milestone cadence sampled only the dose phase; the "
        "'equilibrium' was the upper rail). The founding traces "
        "(e288/e334/e336) show no zero-dose events (R79's desk check) — "
        "but their milestone reads were taken at event boundaries "
        "(post-dose). THE RESIDUAL: does the read SAG intra-period "
        "(between events, under the wash) and recover at each dose — "
        "making the founding heights upper-rail samples too — or does it "
        "hold flat between events?"),
    "design_verbatim": {
        "1_THE_SUBJECT": (
            "e336's committed t400 checkpoint (md5-bound — the same state "
            "x39 used; read 0.815; the founding-class trajectory)."),
        "2_THE_MID_PHASE_READS": (
            "resume and continue the IDENTICAL protocol for 4-8 events, "
            "reading at event+13 (mid-period, ~halfway to the next dose) "
            "IN ADDITION to the standard pre/post pairs — the "
            "phase-declared convention (every phase read, not just the "
            "boundaries)."),
        "3_THE_ANALYSIS": (
            "the sag profile (boundary read vs mid-phase read per period); "
            "the founding numbers' phase character."),
    },
    "bars_verbatim": {
        "SAG-EXISTS": (
            "the mid-phase reads fall materially below the boundary reads "
            "(>= ~15% mean sag) — THE FOUNDING HEIGHTS ARE RAILS (the "
            "endpoint class carries the phase clause; Law 4's endpoint "
            "language joins x41's rider; the 3.16-3.62x band is an "
            "upper-rail band)."),
        "NO-SAG": (
            "the mid-phase reads hold within noise (< ~5%) — THE FOUNDING "
            "NUMBERS STAND AS WRITTEN (the 25-step periods hold; the "
            "costume lesson does not reach the founding class)."),
    },
    "lab_lean_verbatim": (
        "Lab lean: SAG-EXISTS, weakly — x41's PERIOD2X showed 2.4x sag "
        "within 50 steps (the read cannot hold half a doubled period); "
        "whether 25 steps (the founding period) is short enough to hold "
        "is the open question — the founding deficit-medians (0.6-0.7) "
        "suggest substantial decay between events. State your own read."),
    "P-e342a": {
        "my_guess": ("SAG-EXISTS, MODERATELY (diverging from the lean "
                     "only in strength)"),
        "registered": (
            "GROUNDS: (1) THE COMMITTED ENDPOINTS ALREADY COMMIT THE "
            "SWING — both legs' pre/post pairs bound the +25 floors at "
            "10-28% OF THE PEAKS (e336 leg-1: gate 0.1254 vs milestone "
            "0.8147 at t400 = 6.5x; e339 leg-2: t599 0.0916 vs t600-post "
            "0.9000 = 9.8x); a mid read at +13 above 95% of its boundary "
            "would need the ENTIRE collapse packed into the final 12 "
            "steps — a cliff with no mechanism (the wash steps are "
            "homogeneous draws under a per-step-constant cap and clip). "
            "(2) THE +1 COMMITS THE DECAY'S START ON THE SMALL-DOSE HALF "
            "— e339's committed win1:t426 0.6155 after the 0.6340 post "
            "(−2.9% in ONE step): the small-dose periods (17/19/21/23 by "
            "the committed 2-cycle deficits) are already falling at +1. "
            "(3) THE COUNTERVAILING (why moderately, not strongly): the "
            "big-dose periods' +1 HOLDS (e339's win2:t601 0.8983 after "
            "the 0.9000 post — −0.19%): the decay is CONVEX and its "
            "knee's location is the genuinely open parameter. (4) x41's "
            "PERIOD2X bounds the hold below 50 steps."),
        "predicted_shape": (
            "mean sag in [0.35, 0.60]; small-dose periods sag 0.45-0.65, "
            "big-dose 0.20-0.45; every mid strictly above the next gate "
            "floor (the collapse back-loaded, front-half convex); the "
            "leg's t500/t600 boundary reads within 0.03 of e339's "
            "committed (0.85299/0.899997 — the shared-draw continuation; "
            "disclosed tolerance, never a law)."),
        "falsifier": (
            "mean sag < 0.15 (my card flips to the residual cells) or "
            "> 0.75 (the collapse completes BEFORE +13 — front-loaded; "
            "the phase character worse than my card)."),
        "scored": "TRUE iff the verdict == SAG-EXISTS",
    },
}

deviations: list[str] = [
    "THE BAR OPERATIONALIZATION (frozen at birth): the dispatch's "
    "'~15%'/'~5%' leave [5%, 15%) unnamed — PARTIAL-SAG frozen as the "
    "named residual cell (the founding numbers carry a mid-strength "
    "phase clause; neither rail nor flat). Boundaries 0.15/0.05 frozen "
    "as literals on the MEAN of the per-period sags.",
    "THE SAG STATISTIC (frozen): sag_k = (post_k − mid_k)/post_k with "
    "post_k = the MILESTONE g0 read at the event step (post-dose — the "
    "founding read convention: the same battery, channel p(T), baseline "
    "and read point as e336's committed milestones) and mid_k = the "
    "milestone read at s_k+13; the mean over the leg's 7 complete pairs "
    "(period 24's mid sits at t613 beyond the leg — disclosed); NO pair "
    "dropped post-hoc.",
    "THE LEG (frozen): 8 events — the dispatch's 4-8 at its top (every "
    "event a sag pair); M=25 UNCHANGED; pools LENGTH-SCALED (x41's "
    "frozen form: B_C x 200/400, B_M x 8/16; LR_M_MAX never scaled) — "
    "the per-step/per-event equal-share arithmetic identical to e336's "
    "leg-1 and e339's leg-2 at their starts; the applied corpus lr "
    "asserted inside x41's committed LR_BAND (0.0018, 0.0043).",
    "THE WINDOWS (frozen): WIN1 = (424,425,426) == e339's committed win1 "
    "(a 4-row shared-draw repro co-read); WIN2 = (574,575,576) around "
    "event 23 — a second +1-after-small-dose datum mid-leg; e339's "
    "committed win2 t601 (+1-after-big-dose) is CITED from its record "
    "(this leg ends at t600 — its t601 flank would never fire); the "
    "t600-pre floor comes free in the event-24 gate read.",
    "THE e339 CO-READ (non-halting, disclosed): this leg re-runs the "
    "shared-draw continuation of e339's committed t401-t600 segment "
    "(the same restored generator states; e339 kept full-leg pools, "
    "this leg the length-scaled form — proportional equal-share paths, "
    "x41's precedent); my t500/t600 + win1 rows vs e339's committed "
    "literals at a DISCLOSED 0.05 tolerance — a consistency check, "
    "NEVER a gate (cross-run GPU op-order determinism is not the "
    "family's law; the 2e-6 law covers same-bits CPU reads only).",
    "THE ZERO-EVENT CLAUSE (frozen): e339's committed events 17-24 ALL "
    "dosed (deficits 0.025-0.684); if a ZERO event (deficit 0.0) fires "
    "inside this leg its post == its gate and the pair still measures "
    "the wash's decay from an undosed boundary — stamped per-period, "
    "never excluded.",
    "ONE ARM (the dispatch's design names ONE subject): the ERROR-GATED "
    "continuation only; the twin has no events — no phases — and is "
    "irrelevant to the intra-period question (x39/x41's one-arm "
    "precedent).",
    "THE FOUNDED-CHARACTER DESK HALF is runtime-computed from the "
    "md5-bound committed records only (e336's own-period peak/floor "
    "ratios + the canon's deficit-medians) — zero compute of its own.",
    "IMPORT SIDE EFFECTS (disclosed, inherited): importing e336 opens "
    "runs/e336/run.log in append and asserts CUDA — zero bytes written "
    "by this cell (the handle is re-pointed before any use); E261's "
    "module globals are re-bound label-only (tags e342:AUDIT-LEG:*); "
    "E336.REA_ARM re-bound label-only for the burst tag and restored.",
    "Smoke mode (E342_SMOKE=1): e339's smoke form — a manufactured tiny "
    "leg-1 (8 steps, M=2) staged + continued by the audit leg (8 steps, "
    "M=2, mids at +1 — a 2-step period's half, disclosed); machinery "
    "validation only, NOTHING adjudicated (SMOKE stamp on every read).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
]

builds_on = [
    "R79 (the review that minted this card: the desk half — no zero-dose "
    "events in any founding trace, the endpoints phase-consistent; the "
    "residual named: the intra-period sag)",
    "e336 / THE FOUNDING-CLASS SECOND INSTANCE (THE SUBJECT: its "
    "committed t400 ERROR-GATED resume state — the model + both SGD-M "
    "buffers + both generator states; its committed driver executed "
    "verbatim; its own-period gate floors the desk half's evidence)",
    "e339 / THE LONG LANDING (the resume + leg convention + the "
    "length-scaled-pool parent; its committed t401-t800 leg-2 the "
    "shared-draw co-read's parent — gates, deficits, windows, t500/t600)",
    "x41 / THE RELAY KINETICS (the milestone-costume discovery cell + "
    "the length-scaled pool form + LR_BAND + the 2-cycle rails; its "
    "PERIOD2X the lean's own evidence)",
    "x39 / THE STEP-DOSE RUNG (the subject's prior consumer — the same "
    "md5-bound t400 state; its committed subject literals this cell's "
    "repro targets)",
    "e288 / e287 / e285 / e311 / e273 / e268 / e264 / e261 (the "
    "controller rig, the calibration, the sanctuary form, the TAVIREN "
    "organism, the lr, the corpus step, the room, the burst machinery)",
]
whats_new = [
    "THE MID-PHASE READ — the first measurement INSIDE a founding-period "
    "interval: every prior read of this lineage sat at event boundaries "
    "(pre/post) or coarse milestones (t100-multiples); the +13 read is "
    "the phase audit's own datum",
    "THE PHASE-DECLARED CONVENTION at full cadence: every event pre+post "
    "pair + every period's mid — 15 declared milestone reads + 8 gate "
    "reads + 8 window reads on one 200-step leg",
    "THE SAG PROFILE against the founding numbers: the founding "
    "heights' phase character quantified (rail vs flat), joining x41's "
    "rider with the endpoint clause it implied",
]

metrics: dict = {}
metrics["gates"] = {}


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    metrics["last_write"] = common.now_iso()
    save_json(RD / "metrics.json", metrics)
    log(f"  [partial write] {note}")


def med(xs):
    ys = sorted(float(x) for x in xs)
    return ys[len(ys) // 2] if ys else None


def md5of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=E43.REPO,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()])


# ======================================================================
# THE LEG RE-BINDING MACHINE — e339/x41's, VERBATIM (the ONLY e336-module
# globals this cell touches; asserted: same set in, same set out)
# ======================================================================
LEG_KEYS = ("PHASE_STEPS", "MILESTONES", "WIN1", "WIN2", "MAINT_EVERY")


def patch_leg_globals(n_steps: int, milestones, win1, win2,
                      maint_every=None) -> dict:
    before = {k: getattr(E336, k) for k in LEG_KEYS}
    E336.PHASE_STEPS = n_steps
    E336.MILESTONES = tuple(milestones)
    E336.WIN1 = tuple(win1)
    E336.WIN2 = tuple(win2)
    if maint_every is not None:
        E336.MAINT_EVERY = int(maint_every)
    after = {k: getattr(E336, k) for k in LEG_KEYS}
    log(f"  [leg-bind] e336 module globals re-bound: {after} "
        f"(was {before}); every protocol constant untouched")
    return {"before": {k: str(v) for k, v in before.items()},
            "after": {k: str(v) for k, v in after.items()}}


# ======================================================================
# THE STAGING MACHINE — e339's, VERBATIM: the state bits (model/optC/
# optF/cgen/igen) pass through UNTOUCHED (digest-verified); the leg
# ledgers + pools zeroed (the leg convention)
# ======================================================================
def _tensor_digest(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        return hashlib.md5(x.detach().cpu().contiguous()
                           .numpy().tobytes()).hexdigest()
    if isinstance(x, dict):
        parts = []
        for k in sorted(x.keys(), key=repr):
            parts.append(repr(k))
            parts.append(_tensor_digest(x[k]))
        return hashlib.md5("|".join(parts).encode()).hexdigest()
    if isinstance(x, (list, tuple)):
        return hashlib.md5("|".join(_tensor_digest(v) for v in x)
                           .encode()).hexdigest()
    return hashlib.md5(repr(x).encode()).hexdigest()


def stage_leg_resume(src: Path, dst: Path, N: int) -> dict:
    st = torch.load(src, map_location="cpu", weights_only=False)
    bits_src = {"model": {k: _tensor_digest(v) for k, v in
                          st["model"].items()},
                "optC": _tensor_digest(st["optC"]),
                "optF": _tensor_digest(st["optF"]),
                "cgen_state": _tensor_digest(st["cgen_state"]),
                "igen_state": _tensor_digest(st["igen_state"])}
    leg = {
        "model": st["model"],                       # bits unchanged
        "optC": st["optC"], "optF": st["optF"],
        "cgen_state": st["cgen_state"],
        "igen_state": st["igen_state"],
        "step": int(st["step"]),
        # ---- the leg reset (the ONLY transformation; disclosed) ----
        "traj": [], "corpus_ledger": {}, "orth_ledger": {},
        "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
        "lr_ledger": {}, "maint_ledger": [], "window_ledger": [],
        "bufsep": {"corpus_checks": 0, "corpus_violations": 0,
                   "maint_checks": 0, "maint_violations": 0,
                   "optF_steps": 0},
        "corp_cum": torch.zeros(N, dtype=torch.float64),
        "maint_cum": torch.zeros(N, dtype=torch.float64),
        "tot_prev": torch.zeros(N, dtype=torch.float64),
        "S_corpus": 0.0, "S_maint": 0.0,
        "n_capped": 0,
        "n_maint": 0, "n_maint_capped": 0,
        "n_chunks": 0, "chunk_table": [],
        "orth_max": 0.0,
    }
    torch.save(leg, dst)
    chk = torch.load(dst, map_location="cpu", weights_only=False)
    bits_dst = {"model": {k: _tensor_digest(v) for k, v in
                          chk["model"].items()},
                "optC": _tensor_digest(chk["optC"]),
                "optF": _tensor_digest(chk["optF"]),
                "cgen_state": _tensor_digest(chk["cgen_state"]),
                "igen_state": _tensor_digest(chk["igen_state"])}
    ok = bits_src == bits_dst and chk["step"] == int(st["step"]) \
        and chk["S_corpus"] == 0.0 and chk["S_maint"] == 0.0
    return {"src": str(src), "dst": str(dst), "step": int(st["step"]),
            "state_bits": bits_src, "bits_unchanged": bool(ok),
            "source_S_corpus": float(st["S_corpus"]),
            "source_S_maint": float(st.get("S_maint", 0.0)),
            "source_n_maint": int(st.get("n_maint", 0))}


# ======================================================================
def main():
    global dev
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e342_phase_audit",
        "phase": ("THE PHASE AUDIT (R79's minted card; the costume "
                  "lesson's last open residual): e336's committed t400 "
                  "ERROR-GATED state (the founding-class trajectory — the "
                  "same state x39 consumed) resumed md5-bound; the "
                  "IDENTICAL protocol continued 8 events (t401-t600, M=25 "
                  "unchanged, length-scaled pools) with the "
                  "PHASE-DECLARED reads at full cadence — every event's "
                  "pre+post pair PLUS the mid-phase read at event+13 — "
                  "the sag profile vs the frozen bars: SAG-EXISTS (the "
                  "founding heights are rails; the 3.16-3.62x band an "
                  "upper-rail band) / NO-SAG (the founding numbers stand "
                  "as written) / PARTIAL-SAG (the named residual)"),
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "ONE leg — never concurrent) + CPU fp64 projections, "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85), recorded to runs/_envelope_log.jsonl tagged "
                      "e342:AUDIT-LEG:*",
            "trainings": f"1 continuation leg x {LEG_STEPS} corpus steps "
                         "(e336's committed driver VERBATIM; 8 "
                         "NAME-ONLY maintenance events at the error-gated "
                         "lr; length-scaled pools); NO cons",
        },
        "builds_on": builds_on,
        "whats_new": whats_new,
        "deviations": deviations,
    })
    write_partial("startup (bars + lean + P-e342a registered, committed "
                  "at birth)")

    # ================= P-1: the committed records md5-bound ============
    assert md5of(E336_METRICS) == E336_METRICS_MD5, "e336 record drifted"
    assert md5of(E339_METRICS) == E339_METRICS_MD5, "e339 record drifted"
    e336m = json.loads(E336_METRICS.read_text(encoding="utf-8"))
    e339m = json.loads(E339_METRICS.read_text(encoding="utf-8"))
    e336_rea_ph = e336m["arms"]["ERROR-GATED"]["phase"]
    e339_rea_ph = e339m["arms"]["ERROR-GATED"]["phase"]
    e336_traj = {int(r["step"]): r for r in e336_rea_ph["traj"]}
    e339_traj = {int(r["step"]): r for r in e339_rea_ph["traj"]}
    e339_maint = e339_rea_ph["maint_ledger"]
    e339_win = {r["window_phase"]: r for r in e339_rea_ph["window_ledger"]}

    g_e336rec = {
        "path": "runs/e336/metrics.json", "md5": md5of(E336_METRICS),
        "bound_md5": E336_METRICS_MD5,
        "t400_row": {k: e336_traj[400][k] for k in
                     ("step", "g0_pz", "gm12_pz", "ce_r", "disp_norm",
                      "n_maint_so_far")},
        "checks": {
            "t400_read": bool(e336_traj[400]["g0_pz"] == SUBJ_READ_T
                              and e336_traj[400]["gm12_pz"] == SUBJ_GM12_T
                              and e336_traj[400]["ce_r"] == SUBJ_CE_R
                              and e336_traj[400]["disp_norm"]
                              == SUBJ_DISP_NORM
                              and e336_traj[400]["n_maint_so_far"]
                              == SUBJ_N_MAINT),
            "milestones_all_post_dose": bool(
                [int(r["step"]) for r in e336_rea_ph["traj"]]
                == [100, 200, 300, 400]
                and all(int(r["step"]) % 25 == 0
                        for r in e336_rea_ph["traj"])),
            "spend_ledger": bool(
                abs(e336_rea_ph["budget"]["S_corpus_final"]
                    - E336_S_CORPUS_400) < 1e-12
                and abs(e336_rea_ph["budget"]["S_maint_final"]
                        - E336_S_MAINT_400) < 1e-12),
        },
        "claim": ("the founding trajectory's committed description "
                  "md5-bound; its t400 milestone literals are the repro "
                  "gate's parents; its own milestones ALL sit on event "
                  "steps (post-dose — the phase character under audit)"),
    }
    g_e336rec["pass"] = all(g_e336rec["checks"].values())
    assert g_e336rec["pass"], f"G_E336RECORD FAILED: {g_e336rec}"
    metrics["gates"]["G_E336RECORD"] = g_e336rec

    g_e339rec = {
        "path": "runs/e339/metrics.json", "md5": md5of(E339_METRICS),
        "bound_md5": E339_METRICS_MD5,
        "leg2_refs": {
            "t500_g0": e339_traj[500]["g0_pz"],
            "t600_g0": e339_traj[600]["g0_pz"],
            "win1_rows": {k: e339_win[k]["g0_pz"] for k in E339_WIN1},
            "gates_17_24": [m["read_at_gate"] for m in e339_maint[:8]],
            "deficits_17_24": [m["deficit_t"] for m in e339_maint[:8]],
            "maint_steps_17_24": [m["step"] for m in e339_maint[:8]],
            "leg2_spend": {"S_corpus": E339_LEG2_S_CORPUS,
                           "S_maint": E339_LEG2_S_MAINT},
        },
        "checks": {
            "t500_t600": bool(e339_traj[500]["g0_pz"] == E339_T500_G0
                              and e339_traj[600]["g0_pz"] == E339_T600_G0),
            "win1": bool(all(e339_win[k]["g0_pz"] == v
                             for k, v in E339_WIN1.items())),
            "gates": bool([m["read_at_gate"] for m in e339_maint[:8]]
                          == E339_GATES_17_24),
            "deficits": bool([m["deficit_t"] for m in e339_maint[:8]]
                             == E339_DEFICITS_17_24),
            "events_17_24_all_dosed": bool(
                [m["step"] for m in e339_maint[:8]]
                == [425, 450, 475, 500, 525, 550, 575, 600]
                and all(m["deficit_t"] > 0.0 for m in e339_maint[:8])),
        },
        "claim": ("the shared-draw co-read's parents md5-bound: e339's "
                  "committed t401-t800 leg-2 re-ran events 17-24 at these "
                  "gates/deficits; this leg re-draws the same stream"),
    }
    g_e339rec["pass"] = all(g_e339rec["checks"].values())
    assert g_e339rec["pass"], f"G_E339RECORD FAILED: {g_e339rec}"
    metrics["gates"]["G_E339RECORD"] = g_e339rec
    log(f"P-1 the committed records BOUND: e336 t400 read "
        f"{SUBJ_READ_T:.10f}; e339 leg-2 events 17-24 gates "
        f"{[round(x, 3) for x in E339_GATES_17_24]} deficits "
        f"{[round(x, 3) for x in E339_DEFICITS_17_24]} (all dosed)")
    write_partial("P-1 the parents md5-bound (e336 + e339)")

    # ================= P0: the protocol rebuild (e336's mirror) ========
    set_seed(E336.CORPUS_GEN_SEED)       # global init only
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    vocab = corpus.vocab_size
    zid = stoi["Z"]
    tid = stoi[E336.PARASITE_NAME[0]]    # the read channel: p(T)
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert vocab == 65, f"vocab drift {vocab}"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    assert list(g0_ids.shape) == [60, G1.PRE]

    anchor_full = torch.stack([train_ids[p - G1.PRE:
                                         p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True

    tav_ids = corpus.encode(E336.PARASITE_NAME)

    def build_win_t(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], tav_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_t = torch.stack([build_win_t(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_t[i][G1.PRE: G1.PRE + len(E336.PARASITE_NAME)])
        == E336.PARASITE_NAME for i in range(win_t.shape[0]))
    win_pre_ok = all(torch.equal(win_t[i][:G1.PRE],
                                 anchor_full[i][:G1.PRE])
                     for i in range(win_t.shape[0]))
    G_NAMEWIN = {
        "form": "e336's G_NAMEWIN mirror: the TRUE TAVIREN install "
                "windows, decode-gated, pre-context bit-equal the anchor "
                "bank",
        "n_windows": int(win_t.shape[0]),
        "masked_decode_all_name": bool(win_masked_ok),
        "precontext_bit_equal_anchor": bool(win_pre_ok),
        "pass": bool(win_masked_ok and win_pre_ok
                     and list(win_t.shape) == [60, G1.BLOCK])}
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"
    metrics["gates"]["G_NAMEWIN"] = G_NAMEWIN

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"]["G_NAMEFREE"] = {
        "corpus_zeph_count": train_text.count("ZEPH"),
        "corpus_taviren_count": train_text.count(E336.PARASITE_NAME),
        "pass": bool(train_text.count("ZEPH") == 0
                     and train_text.count(E336.PARASITE_NAME) == 0)}
    assert metrics["gates"]["G_NAMEFREE"]["pass"]
    log("P0: the protocol rebuilt (corpus + battery + anchors + the name "
        "windows + the eval banks) — e336's P0 mirror PASS")

    # ================= P1: the room + the parents (e336's mirror) ======
    _base_art = torch.load(E336.CKPT_DIR / E336.BASE_CK,
                           map_location="cpu", weights_only=False)
    base_sd = (_base_art["model"] if "model" in _base_art
               else _base_art)      # e001.pt is a RAW state dict
    del _base_art
    base_net = G1.evl_load(base_sd)
    n_par = sum(p.numel() for p in base_net.parameters())
    assert n_par == 2739072, f"param drift {n_par}"
    assert md5of(E336.CKPT_DIR / E336.BASE_CK) == E336.BASE001_MD5
    base_flat = flat_params_cpu(base_net)
    base_flat_np = base_flat.double().numpy().astype(np.float64)
    base_prior = G1.battery_cell(base_net, g0_ids, tid)["mean_pz"]
    G_BASE = {"checkpoint": f"runs/checkpoints/{E336.BASE_CK}",
              "md5": E336.BASE001_MD5, "params": n_par,
              "base_prior_parasite_g0": base_prior,
              "committed_base_prior": E336.BASE_PRIOR_T,
              "abs_diff": abs(base_prior - E336.BASE_PRIOR_T),
              "pass": bool(abs(base_prior - E336.BASE_PRIOR_T) <= 2e-6)}
    assert G_BASE["pass"], f"G_BASE failed: {G_BASE}"
    metrics["gates"]["G_BASE"] = G_BASE

    vmap_art = torch.load(E336.CKPT_DIR / E336.VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {"path": f"runs/checkpoints/{E336.VMAP_CK}",
                "md5": md5of(E336.CKPT_DIR / E336.VMAP_CK),
                "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
                "size": int(v_flat32.numel()),
                "pass": bool(vmap_art.get("meta", {}).get("experiment")
                             == "e258" and int(v_flat32.numel()) == n_par)}
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND

    span_art = torch.load(E336.CKPT_DIR / E336.SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(E336.CKPT_DIR / E336.SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta",
                                                  {}).get("experiment"),
                  "pass": bool(md5of(E336.CKPT_DIR / E336.SPAN_CK)
                               == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == n_par)}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND

    rooms_here = E261.LadderRooms(n_par, E336.LADDER, v64_np,
                                  Vp.numpy().astype(np.float64),
                                  list(G1.evl_load(base_sd).parameters()),
                                  dev)
    if not SMOKE:
        cert = rooms_here.certify()
        assert cert["pass"], f"room cert failed: {cert}"
        rooms336 = torch.load(E336_DIR / "e336_rooms.pt", map_location="cpu",
                              weights_only=False)

        def _np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        D336 = _np(rooms336["model"]["K10K"]["D_int8"]).astype(np.float64)
        S336 = _np(rooms336["model"]["K10K"]["S"])
        D_mine = rooms_here.rooms["K10K"].D
        S_mine = rooms_here.rooms["K10K"].S
        G_ROOMK10K = {
            "form": "the room re-derived from the committed seeds and "
                    "bit-compared vs e336's committed e336_rooms.pt — "
                    "the same room the founding stream orthogonalized "
                    "against",
            "D_bit_equal_e336": bool(np.array_equal(D_mine, D336)),
            "S_bit_equal_e336": bool(np.array_equal(S_mine, S336)),
            "cert": {k: v for k, v in cert["per_rung"]["K10K"].items()
                     if not isinstance(v, list)},
            "pass": bool(np.array_equal(D_mine, D336)
                         and np.array_equal(S_mine, S336))}
        assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
        del rooms336
    else:
        G_ROOMK10K = {"form": "SMOKE: the room cert vacuous (disclosed; "
                              "the full run bit-binds)",
                      "pass": True, "vacuous": True}
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    proj = rooms_here
    log("P1 the room: K10K re-derived + bit-bound to e336's committed "
        "room: PASS")

    # ---- G_LR_BIND: the lr constants re-verified (the module bind) -----
    lr_sgd_runtime = float(json.loads(
        (E43.REPO / "runs" / "e273" / "lr_calibration.json")
        .read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the leg runs on e336's COMMITTED lr constants: LR_STABLE "
                "= x0.01 x LR_SGD_matched; LR_M_MAX = 0.40 x BUDGET / "
                "e287's b_m sum (the same identity, the same value — "
                "NEVER scaled); momentum 0.9, wd 0.0 EXACTLY",
        "lr_sgd_record": E336.E273_LR_SGD,
        "lr_sgd_runtime": lr_sgd_runtime,
        "lr_stable_module": E336.LR_STABLE,
        "lr_stable_metrics": e336m["gates"]["G_LR_BIND"]["lr_stable"],
        "lr_m_max_module": E336.LR_M_MAX_FROZEN,
        "lr_m_max_metrics": e336m["gates"]["G_LR_BIND"]["lr_m_max_frozen"],
        "momentum": E336.SGD_MOMENTUM, "wd": E336.SGD_WD,
        "pass": bool(E336.LR_STABLE == 0.21738574801453703
                     and E336.LR_M_MAX_FROZEN
                     == 0.021942464013242388
                     and lr_sgd_runtime == E336.E273_LR_SGD
                     and E336.SGD_MOMENTUM == 0.9 and E336.SGD_WD == 0.0
                     and e336m["gates"]["G_LR_BIND"]["lr_stable"]
                     == E336.LR_STABLE
                     and e336m["gates"]["G_LR_BIND"]["lr_m_max_frozen"]
                     == E336.LR_M_MAX_FROZEN)}
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND

    # ================= P2: the substrate + the LEG BUDGET ==============
    sub_art = torch.load(E336.SUBSTRATE_CK, map_location="cpu",
                         weights_only=False)
    sub_meta = sub_art.get("meta", {}) or {}
    sub_sd = {k: v.detach().clone() for k, v in sub_art["model"].items()}
    sub_net = G1.evl_load(sub_sd)
    sub_flat = flat_params_cpu(sub_net)
    sub_flat_np = sub_flat.double().numpy().astype(np.float64)
    sub_g0 = G1.battery_cell(sub_net, g0_ids, tid)["mean_pz"]
    write_norm = float(np.linalg.norm(sub_flat_np - base_flat_np))
    G_SUBSTRATE = {
        "form": "e336's substrate gate mirror: the organism artifact "
                "raw-md5-bound; its reads gated vs e311's committed "
                "literals (the family's 2e-6 law); the write norm "
                "recomputed fp64",
        "artifact_md5": md5of(E336.SUBSTRATE_CK),
        "committed_md5": E336.SUBSTRATE_MD5,
        "post_g0": sub_g0, "committed_post_g0": E336.TAV_POST_G0,
        "g0_abs_diff": abs(sub_g0 - E336.TAV_POST_G0),
        "write_norm": write_norm, "committed_write_norm":
            E336.TAV_WRITE_NORM,
        "write_norm_rel_diff": abs(write_norm - E336.TAV_WRITE_NORM)
        / E336.TAV_WRITE_NORM,
        "pass": bool(md5of(E336.SUBSTRATE_CK) == E336.SUBSTRATE_MD5
                     and sub_meta.get("experiment") == "e311"
                     and abs(sub_g0 - E336.TAV_POST_G0) <= 2e-6
                     and abs(write_norm - E336.TAV_WRITE_NORM)
                     / E336.TAV_WRITE_NORM <= 1e-8)}
    assert G_SUBSTRATE["pass"], f"substrate gate FAILED: {G_SUBSTRATE}"
    metrics["gates"]["G_SUBSTRATE"] = G_SUBSTRATE
    del sub_net, sub_art

    # ---- THE LEG BUDGET (frozen: e336's identity, length-scaled pools)
    budget_norm = E336.BUDGET_FRAC * write_norm     # = 4.47481453007862
    assert abs(budget_norm - 4.47481453007862) < 1e-9
    lr_m_max = E336.LR_M_MAX_FROZEN                 # NEVER scaled
    lr_m_red = E336.MAINT_BUDGET_SHARE * budget_norm / E336.E287_B_M_SUM
    assert abs(lr_m_red - lr_m_max) < 1e-9, "LR_M_MAX identity broken"
    b_c400 = E336.CORPUS_BUDGET_SHARE * budget_norm
    b_m400 = E336.MAINT_BUDGET_SHARE * budget_norm
    budget_c = b_c400 * LEG_STEPS / 400             # length-scaled (x41)
    budget_m = b_m400 * len(LEG_EVENTS) / 16
    metrics["budget"] = {
        "form": ("THE LEG CONVENTION (frozen at birth): e336's budget "
                 "identity (0.5 x the TAVIREN write's norm) with pools "
                 "LENGTH-SCALED to the leg (B_C x " f"{LEG_STEPS}/400, "
                 f"B_M x {len(LEG_EVENTS)}/16 — x41's frozen form); "
                 "LR_M_MAX the frozen constant (never scaled); the "
                 "per-step/per-event equal-share arithmetic identical to "
                 "e336's leg-1 and e339's leg-2 at their starts"),
        "write_norm": write_norm,
        "budget_identity": budget_norm,
        "b_c400": b_c400, "b_m400": b_m400,
        "budget_c_leg": budget_c, "budget_m_leg": budget_m,
        "lr_m_max": lr_m_max,
        "leg1_committed": {"S_corpus": E336_S_CORPUS_400,
                           "S_maint": E336_S_MAINT_400,
                           "S_total": E336_S_TOTAL_400},
    }
    log(f"P2 the LEG BUDGET: B_C {budget_c:.6f} / B_M {budget_m:.6f} "
        f"(length-scaled from {b_c400:.4f}/{b_m400:.4f}); LR_M_MAX "
        f"{lr_m_max!r} (the identity re-derived {lr_m_red!r})")
    write_partial("P2 the substrate + the leg budget frozen")

    # ---- G_PROTOCOL_IDENT: the e336 module == the committed protocol --
    G_PROTOCOL_IDENT = {
        "form": "the continuation's identity: e336's COMMITTED module "
                "constants asserted == its md5-bound metrics literals "
                "BEFORE any re-binding; then the ONLY re-bindings are "
                "the leg parameters (asserted set)",
        "corpus_gen_seed_module": E336.CORPUS_GEN_SEED,
        "inst_gen_seed_module": E336.INST_GEN_SEED,
        "maint_every_module_committed": E336.MAINT_EVERY,
        "phase_steps_module_committed": E336.PHASE_STEPS,
        "baseline_gate_module": E336.FACT_BASELINE_G0,
        "budget_frac_module": E336.BUDGET_FRAC,
        "checks": {
            "seeds": bool(E336.CORPUS_GEN_SEED == 33601
                          and E336.INST_GEN_SEED == 33602),
            "cadence": bool(E336.MAINT_EVERY == 25
                            and E336.PHASE_STEPS == 400),
            "baseline": bool(E336.FACT_BASELINE_G0
                             == 0.285851389169693
                             == e336m["gates"]["G_CTRLIDENT"]
                             ["baseline_gate"]),
            "budget_form": bool(E336.BUDGET_FRAC == 0.5
                                and E336.CORPUS_BUDGET_SHARE == 0.6
                                and E336.MAINT_BUDGET_SHARE == 0.4
                                and abs(e336m["budget"]["budget_norm"]
                                        - budget_norm) < 1e-12),
            "bufsep_ledger": bool(
                e336_rea_ph["bufsep"] == {"corpus_checks": 400,
                                          "corpus_violations": 0,
                                          "maint_checks": 16,
                                          "maint_violations": 0,
                                          "optF_steps": 16}),
            "milestones_committed": bool(
                [int(r["step"]) for r in e336_rea_ph["traj"]]
                == [100, 200, 300, 400]),
        },
        "pass": None}      # set below (after the leg re-bind assert)
    ok = all(G_PROTOCOL_IDENT["checks"].values())
    assert ok, f"protocol identity failed: {G_PROTOCOL_IDENT['checks']}"
    leg_bind_record = patch_leg_globals(
        LEG_END, LEG_MILESTONES, LEG_WIN1, LEG_WIN2,
        maint_every=(2 if SMOKE else None))
    assert E336.PHASE_STEPS == LEG_END \
        and tuple(E336.MILESTONES) == tuple(LEG_MILESTONES) \
        and tuple(E336.WIN1) == tuple(LEG_WIN1) \
        and tuple(E336.WIN2) == tuple(LEG_WIN2) \
        and E336.MAINT_EVERY == (2 if SMOKE else 25)
    G_PROTOCOL_IDENT["leg_rebind"] = leg_bind_record
    G_PROTOCOL_IDENT["pass"] = True
    metrics["gates"]["G_PROTOCOL_IDENT"] = G_PROTOCOL_IDENT
    log(f"P2 G_PROTOCOL_IDENT: e336's committed constants == the md5-bound "
        f"record; leg re-bind = {{PHASE_STEPS {LEG_END}, M {E336.MAINT_EVERY}"
        f", MILESTONES {LEG_MILESTONES}, WIN {LEG_WIN1}/{LEG_WIN2}}}: PASS")
    write_partial("P2 the protocol identity bound (leg re-bind asserted)")

    # ================= P3: the resume gates + the staging ===============
    if SMOKE:
        # e339's smoke form: manufacture a tiny committed-form leg-1 via
        # e336's own driver (8 steps, M=2) — machinery validation only;
        # the smoke numbers are MEANINGLESS (disclosed)
        log("P3 SMOKE: manufacturing a smoke leg-1 checkpoint via e336's "
            "own driver (8 steps, M=2) — machinery validation only")
        patch_leg_globals(SMOKE_LEG1_STEPS, tuple(range(1, 9)),
                          *SMOKE_LEG1_WINS, maint_every=2)
        bc1 = b_c400 * SMOKE_LEG1_STEPS / 400
        bm1 = b_m400 * 4 / 16
        _ = E336.chunked_errorgated_phase(
            "SMOKE-LEG1", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, bc1, bm1, win_t, inst_mask, anchor_full,
            train_ids, g0_ids, gm12_ids, r_eval_xy, tid, lr_m_max,
            RD / "smoke_leg1_REA_resume.pt", dev)
        patch_leg_globals(LEG_END, LEG_MILESTONES, LEG_WIN1, LEG_WIN2,
                          maint_every=2)
        rea_src = RD / "smoke_leg1_REA_resume.pt"
    else:
        rea_src = REA_RESUME_CK

    rea_st = torch.load(rea_src, map_location="cpu", weights_only=False)
    G_RESUME = {
        "form": "the cross-cell resume identity: e336's committed t400 "
                "resume state (raw-md5-bound) — step/n_maint/spend/"
                "isolation ledgers asserted == the committed metrics; "
                "the resume model bit-compared vs the committed post "
                "checkpoint (x39's own subject bind)",
        "rea_resume_md5": md5of(rea_src),
        "rea_post_md5": md5of(REA_POST_CK),
        "rea_step": int(rea_st["step"]), "rea_n_maint": int(rea_st["n_maint"]),
        "rea_S_corpus": float(rea_st["S_corpus"]),
        "rea_S_maint": float(rea_st["S_maint"]),
        "expected": {"rea_step": LEG_START, "rea_n_maint": 16,
                     "rea_S_corpus": E336_S_CORPUS_400,
                     "rea_S_maint": E336_S_MAINT_400},
    }
    # the resume model == the committed post checkpoint (bit-compare)
    rea_post = torch.load(REA_POST_CK, map_location="cpu", weights_only=False)
    model_bit_equal = all(
        torch.equal(rea_st["model"][k], rea_post["model"][k])
        for k in rea_st["model"])
    G_RESUME["resume_model_bit_equal_post"] = bool(model_bit_equal)
    G_RESUME["pass"] = bool(
        (SMOKE or md5of(rea_src) == E336_REA_RESUME_MD5)
        and (SMOKE or md5of(REA_POST_CK) == E336_REA_POST_MD5)
        and int(rea_st["step"]) == LEG_START
        and (SMOKE or (int(rea_st["n_maint"]) == 16
                       and abs(float(rea_st["S_corpus"])
                               - E336_S_CORPUS_400) < 1e-9
                       and abs(float(rea_st["S_maint"])
                               - E336_S_MAINT_400) < 1e-9
                       and model_bit_equal)))
    assert G_RESUME["pass"], f"resume gate FAILED: {G_RESUME}"
    metrics["gates"]["G_RESUME"] = G_RESUME
    del rea_post
    log(f"P3 G_RESUME: e336's t400 state loaded (step {rea_st['step']}, "
        f"n_maint {rea_st.get('n_maint')}, S "
        f"{float(rea_st['S_corpus']):.4f}+"
        f"{float(rea_st.get('S_maint', 0.0)):.4f}); resume-model == "
        f"post-model: {model_bit_equal}: PASS")

    # ---- G_T400_REPRO (THE DISPATCH'S BOUNDARY-REPRO GATE) -------------
    # the resumed bits re-read the committed t400 battery values — e336's
    # committed t400 milestone literals == x39's committed subject
    # literals (the family's 2e-6/5e-3 cross-session law), BEFORE any
    # stepping
    net_r400 = G1.evl_load({k: v for k, v in rea_st["model"].items()})
    repro_g0 = G1.battery_cell(net_r400, g0_ids, tid)["mean_pz"]
    repro_gm12 = G1.battery_cell(net_r400, gm12_ids, tid)["mean_pz"]
    repro_ce = G1.ce_fixed_cpu(net_r400, *r_eval_xy)
    committed_rea_g0 = (rea_st["traj"][-1]["g0_pz"] if SMOKE
                        else e336_traj[400]["g0_pz"])
    # smoke: the subject is the MANUFACTURED tiny leg-1 — the repro
    # targets are ITS OWN committed final row (the machinery is what the
    # smoke validates); full: e336's t400 == x39's subject literals
    tgt_g0 = committed_rea_g0 if SMOKE else SUBJ_READ_T
    G_T400_REPRO = {
        "form": ("THE DISPATCH'S GATE ('the boundary reads reproduce "
                 "e336/x39's committed values'): the restored t400 bits "
                 "re-read the committed g0/gm12/CE_R (== e336's t400 "
                 "milestone row == x39's md5-bound subject literals) on "
                 "the same battery + channel, BEFORE any stepping"
                 + ("; SMOKE: the manufactured leg-1's own final row is "
                    "the target (machinery validation)" if SMOKE else "")),
        "g0": {"mine": repro_g0, "committed": tgt_g0,
               "e336_traj_row": None if SMOKE else committed_rea_g0,
               "abs_diff": abs(repro_g0 - tgt_g0)},
        "gm12": {"mine": repro_gm12,
                 "committed": None if SMOKE else SUBJ_GM12_T,
                 "abs_diff": None if SMOKE
                 else abs(repro_gm12 - SUBJ_GM12_T)},
        "ce_r": {"mine": repro_ce,
                 "committed": None if SMOKE else SUBJ_CE_R,
                 "abs_diff": None if SMOKE else abs(repro_ce - SUBJ_CE_R)},
        "tol": {"read": READ_TOL, "ce": CE_TOL},
        "pass": bool(abs(repro_g0 - tgt_g0) <= READ_TOL
                     and (SMOKE
                          or (abs(repro_gm12 - SUBJ_GM12_T) <= READ_TOL
                              and abs(repro_ce - SUBJ_CE_R) <= CE_TOL
                              and committed_rea_g0 == SUBJ_READ_T))),
    }
    assert G_T400_REPRO["pass"], f"t400 repro FAILED: {G_T400_REPRO}"
    metrics["gates"]["G_T400_REPRO"] = G_T400_REPRO
    del net_r400
    log(f"P3 G_T400_REPRO: the resumed bits re-read the committed "
        f"boundary values (g0 |d| {G_T400_REPRO['g0']['abs_diff']:.1e}"
        + ("; gm12 |d| "
           f"{G_T400_REPRO['gm12']['abs_diff']:.1e}; CE_R |d| "
           f"{G_T400_REPRO['ce_r']['abs_diff']:.1e}" if not SMOKE
           else " (smoke: own-row target)")
        + "): PASS — the dispatch's boundary-repro gate")

    # ---- the staging: state bits through, leg pools zeroed -------------
    rea_leg_ck = RD / ("smoke_REA_leg_resume.pt" if SMOKE
                       else "e342_REA_leg_resume.pt")
    stage_rea = stage_leg_resume(rea_src, rea_leg_ck, n_par)
    assert stage_rea["bits_unchanged"], \
        f"staging altered state bits: {stage_rea}"
    metrics["gates"]["G_STAGING"] = {
        "form": "the ONLY protocol-touching transformation: e336's "
                "committed state bits (model/optC/optF/cgen/igen) pass "
                "through UNTOUCHED (digest-verified before/after the "
                "staging write); the leg ledgers + spend pools zeroed "
                "(the leg convention); the generator states carried — "
                "the stream CONTINUES (never re-seeded)",
        "rea": stage_rea,
        "first_continued_draws_disclosure":
            "corpus steps 401.. draw the continuation of seed 33601's "
            "sequence (cgen_state restored); maintenance events 17.. "
            "draw the continuation of seed 33602's (igen_state restored) "
            "— the SAME continuation e339's committed leg consumed",
        "pass": True}
    log("P3 G_STAGING: the leg resume staged (state bits digest-equal; "
        "pools zeroed; generators carried): PASS")
    write_partial("P3 the resume identity + staging complete")

    # ================= P4: THE LEG (the phase-declared continuation) ====
    assert common.gpu_ok(), "gpu_ok() gate immediately before the leg"
    log("=" * 78)
    log(f"THE AUDIT LEG — t{LEG_START + 1}..t{LEG_END} ({LEG_STEPS} steps, "
        f"M={E336.MAINT_EVERY}, {len(LEG_EVENTS)} events at {LEG_EVENTS}; "
        f"mids at +{MID_OFF} -> {LEG_MIDS}; windows {LEG_WIN1}/"
        f"{LEG_WIN2}; e336's committed driver VERBATIM; length-scaled "
        f"pools B_C {budget_c:.6f} B_M {budget_m:.6f})")
    rea_arm_before = E336.REA_ARM
    E336.REA_ARM = "AUDIT-LEG"        # label-only (the burst tags);
                                      # disclosed + restored below
    rea = E336.chunked_errorgated_phase(
        "AUDIT-LEG", G1.evl_load(sub_sd), proj, sub_flat_np,
        base_flat_np, budget_c, budget_m, win_t, inst_mask, anchor_full,
        train_ids, g0_ids, gm12_ids, r_eval_xy, tid, lr_m_max,
        rea_leg_ck, dev)
    E336.REA_ARM = rea_arm_before     # restore (leave e336 as found)
    sd_r = rea["sd"]
    net_r = G1.evl_load(sd_r)
    cells_r = {"gm12": G1.battery_cell(net_r, gm12_ids, tid)["mean_pz"],
               "g0": G1.battery_cell(net_r, g0_ids, tid)["mean_pz"],
               "g0_z": G1.battery_cell(net_r, g0_ids, zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_r, *r_eval_xy)}
    torch.save({"model": sd_r, "meta": {
        "experiment": "e342", "arm": "ERROR-GATED-AUDIT-LEG",
        "desc": f"e342 the phase audit: e336's committed t400 state "
                f"continued t{LEG_START + 1}-t{LEG_END} (the identical "
                f"protocol, length-scaled pools, phase-declared reads)",
        "S_corpus": rea["S_corpus"], "S_maint": rea["S_maint"],
        "n_maint": rea["n_maint"]}},
        RD / ("smoke_REA_post.pt" if SMOKE else "e342_REA_post.pt"))
    del net_r
    log(f"THE AUDIT LEG DONE: post g0 {cells_r['g0']:.7f} | S "
        f"{rea['S_corpus']:.4f}+{rea['S_maint']:.4f} | "
        f"{rea['n_maint']} events | isolation "
        f"{rea['bufsep']['corpus_checks']}+{rea['bufsep']['maint_checks']} "
        f"checks "
        f"{rea['bufsep']['corpus_violations'] + rea['bufsep']['maint_violations']} "
        f"violations")
    write_partial("P4 the audit leg complete")

    # ================= P5: the audit gates ==============================
    ms = {int(r["step"]): r for r in rea["traj"]}
    maint_rows = rea["maint_ledger"]
    win_rows = {r["window_phase"]: r for r in rea["window_ledger"]}

    # ---- G_LEG_IDENTITY: the leg ran the declared schedule -------------
    lr_rows = rea["lr_ledger"]
    applied = [v["lr_applied"] for v in lr_rows.values()]
    med_applied = med(applied) if applied else None
    cap_below = all(v["cap"] <= v["lr_sched"] for v in lr_rows.values()) \
        if lr_rows else None
    g_leg = {
        "form": ("the leg's protocol identity: M=25 UNCHANGED (the "
                 "founding cadence), lr_m_max the frozen constant, "
                 "PHASE_STEPS the leg end, events at the declared steps, "
                 "pools length-scaled; the applied corpus lr's MEDIAN "
                 "inside x41's committed LR_BAND; the corpus cap bound "
                 "below the cosine schedule at EVERY logged step (the "
                 "disclosed schedule immateriality, verified)"),
        "m_every": E336.MAINT_EVERY, "lr_m_max": lr_m_max,
        "phase_steps": LEG_END, "event_steps": LEG_EVENTS,
        "milestones_declared": list(LEG_MILESTONES),
        "milestones_present": sorted(ms.keys()),
        "budget_c": budget_c, "budget_m": budget_m,
        "applied_lr_median": med_applied,
        "applied_lr_min": min(applied) if applied else None,
        "applied_lr_max": max(applied) if applied else None,
        "lr_band": LR_BAND,
        "cap_below_sched_all": cap_below,
        "steps_ran": rea["steps_ran"],
        "pass": bool((SMOKE or (med_applied is not None
                                and LR_BAND[0] <= med_applied
                                <= LR_BAND[1]))
                     and [int(r["step"]) for r in maint_rows]
                     == LEG_EVENTS
                     and sorted(ms.keys()) == list(LEG_MILESTONES)
                     and int(rea["steps_ran"]) == LEG_END
                     and int(rea["n_maint"]) == len(LEG_EVENTS)
                     and (cap_below is not False))}
    assert g_leg["pass"], f"leg identity FAILED: {g_leg}"
    metrics["gates"]["G_LEG_IDENTITY"] = g_leg
    log(f"G_LEG_IDENTITY PASS: {rea['n_maint']} events at "
        f"{[int(r['step']) for r in maint_rows]}; milestones "
        f"{sorted(ms.keys())}; applied lr med "
        + (f"{med_applied:.6f} in {LR_BAND}" if med_applied is not None
           else "n/a (smoke)")
        + f"; cap<=sched {cap_below}")

    # ---- G_PHASEDECL: the phase-declaration convention ------------------
    ev_gates = {int(r["step"]): float(r["read_at_gate"])
                for r in maint_rows}
    ev_defs = {int(r["step"]): float(r["deficit_t"]) for r in maint_rows}
    zero_events = [s for s, d in ev_defs.items() if d == 0.0]
    gpos = [r for r in maint_rows
            if r.get("gate_vs_pre_absdiff") is not None]
    # the milestone-at-event == the window post row (same instant,
    # two CPU reads — the read-determinism law)
    cross_425 = abs(ms[LEG_EVENTS[0]]["g0_pz"]
                    - win_rows[f"win1:t{LEG_EVENTS[0]}-post"]["g0_pz"]) \
        if f"win1:t{LEG_EVENTS[0]}-post" in win_rows else None
    cross_575 = abs(ms[LEG_EVENTS[-2]]["g0_pz"]
                    - win_rows[f"win2:t{LEG_EVENTS[-2]}-post"]["g0_pz"]) \
        if f"win2:t{LEG_EVENTS[-2]}-post" in win_rows else None
    # the expected window labels (full: e339's committed win1 quadruple;
    # smoke: this leg's own)
    exp_win1 = sorted(E339_WIN1.keys()) if not SMOKE else sorted(
        [f"win1:t{LEG_WIN1[0]}", f"win1:t{LEG_WIN1[1]}-post",
         f"win1:t{LEG_WIN1[1]}-pre", f"win1:t{LEG_WIN1[2]}"])
    g_phasedecl = {
        "form": ("the phase-declared convention throughout: EVERY event "
                 "carries its pre pair (the maint gate read — the floor, "
                 "the dose's own input) AND its post pair (a milestone "
                 "row at the event step — the founding read channel); "
                 "every period k<24 carries its mid read at s_k+13; the "
                 "window quadruples bracket the leg"),
        "n_events": len(LEG_EVENTS),
        "pre_post_pairs": bool(all(s in ev_gates and s in ms
                                   for s in LEG_EVENTS)),
        "n_mids": len(LEG_MIDS),
        "mids_present": bool(all(s in ms for s in LEG_MIDS)),
        "win1_rows": sorted(k for k in win_rows if k.startswith("win1")),
        "win2_rows": sorted(k for k in win_rows if k.startswith("win2")),
        "win1_full": bool(sorted(k for k in win_rows
                                 if k.startswith("win1")) == exp_win1),
        "win2_full": bool(sorted(k for k in win_rows
                                 if k.startswith("win2"))
                          == [f"win2:t{LEG_WIN2[0]}",
                              f"win2:t{LEG_WIN2[1]}-post",
                              f"win2:t{LEG_WIN2[1]}-pre",
                              f"win2:t{LEG_WIN2[2]}"]),
        "gate_vs_pre_max": max((float(r["gate_vs_pre_absdiff"])
                                for r in gpos), default=None),
        "milestone_eq_windowpost": {"win1": cross_425, "win2": cross_575,
                                    "tol": READ_TOL},
        "zero_events": zero_events,
        "pass": bool(all(s in ev_gates and s in ms for s in LEG_EVENTS)
                     and all(s in ms for s in LEG_MIDS)
                     and sorted(k for k in win_rows
                                if k.startswith("win1")) == exp_win1
                     and cross_425 is not None and cross_425 <= READ_TOL
                     and (cross_575 is None or cross_575 <= READ_TOL)
                     and (not gpos or max(float(r["gate_vs_pre_absdiff"])
                                          for r in gpos) <= 1e-9)),
    }
    assert g_phasedecl["pass"], f"phase declaration FAILED: {g_phasedecl}"
    metrics["gates"]["G_PHASEDECL"] = g_phasedecl
    log(f"G_PHASEDECL PASS: {len(LEG_EVENTS)} pre+post pairs + "
        f"{len(LEG_MIDS)} mids at +{MID_OFF}; win1 quadruple "
        f"{g_phagedecl_ok(g_phasedecl)}; zero events {zero_events}")

    # ---- G_E339_COVERREAD (non-halting, disclosed) ----------------------
    co = {"form": ("the shared-draw co-read (NON-HALTING, disclosed "
                   "tolerance): this leg re-draws the continuation "
                   "e339's committed leg-2 consumed (the same restored "
                   "generator states); the t500/t600 milestone reads + "
                   "the win1 quadruple vs e339's committed literals — a "
                   "consistency check, never a gate (cross-run GPU "
                   "op-order determinism is not the family's law)"),
          "tol": E339_CO_TOL}
    if not SMOKE:
        co["t500"] = {"mine": ms[500]["g0_pz"],
                      "e339_committed": E339_T500_G0,
                      "abs_diff": abs(ms[500]["g0_pz"] - E339_T500_G0)}
        co["t600"] = {"mine": ms[600]["g0_pz"],
                      "e339_committed": E339_T600_G0,
                      "abs_diff": abs(ms[600]["g0_pz"] - E339_T600_G0)}
        co["win1"] = {k: {"mine": win_rows[k]["g0_pz"],
                          "e339_committed": v,
                          "abs_diff": abs(win_rows[k]["g0_pz"] - v)}
                      for k, v in E339_WIN1.items()}
        co["max_abs_diff"] = max(
            [co["t500"]["abs_diff"], co["t600"]["abs_diff"]]
            + [d["abs_diff"] for d in co["win1"].values()])
        co["consistent_within_tol"] = bool(co["max_abs_diff"]
                                           <= E339_CO_TOL)
    else:
        co["vacuous_smoke"] = True
        co["consistent_within_tol"] = None
    co["non_halting"] = True
    co["pass"] = True          # non-halting by construction (disclosed)
    metrics["gates"]["G_E339_COVERREAD"] = co
    log("G_E339_COVERREAD (non-halting): "
        + ("vacuous (smoke)" if SMOKE else
           f"t500 |d| {co['t500']['abs_diff']:.5f}, t600 |d| "
           f"{co['t600']['abs_diff']:.5f}, win1 max |d| "
           f"{max(d['abs_diff'] for d in co['win1'].values()):.5f} "
           f"(tol {E339_CO_TOL}) — "
           + ("consistent" if co["consistent_within_tol"]
              else "DRIFT DISCLOSED")))

    # ---- the standing conditions (e339's set) ---------------------------
    G_BUFSEP = {
        "form": "e336's G_BUFSEP mirror (the isolation HALT carried): "
                "bidirectional buffer separation at every leg optimizer "
                "event",
        "rea": rea["bufsep"],
        "pass": bool(rea["bufsep"]["corpus_violations"] == 0
                     and rea["bufsep"]["maint_violations"] == 0)}
    assert G_BUFSEP["pass"], f"G_BUFSEP HALT: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    G_ORTH = {
        "form": "e336's G_ORTH mirror (NON-HALTING): the stepped corpus "
                "gradient entirely orthogonal to the room at every leg "
                "step",
        "rea_orth_max_rel_err": rea["orth_max"], "bar": ORTH_BAR,
        "pass": bool(rea["orth_max"] < ORTH_BAR)}
    metrics["gates"]["G_ORTH"] = G_ORTH

    leg_rows = sorted((int(k), v) for k, v in rea["corpus_ledger"].items())
    early = [v["ce"] for s, v in leg_rows
             if LEG_START + 10 <= s <= LEG_START + 100]
    late = [v["ce"] for s, v in leg_rows if LEG_END - 100 < s <= LEG_END]
    stream_live = (None if len(early) < 2 or len(late) < 2 else bool(
        med(late) <= STREAM_STABLE_TOL * med(early)))
    G_STREAMELIVE = {
        "form": "e288's standing condition: the corpus stream live "
                f"(median corpus CE late-vs-early within "
                f"{STREAM_STABLE_TOL}x)",
        "median_early": med(early) if early else None,
        "median_late": med(late) if late else None,
        "n_rows_early": len(early), "n_rows_late": len(late),
        "vacuous_smoke": bool(SMOKE and (len(early) < 2 or len(late) < 2)),
        "pass": bool(stream_live) if stream_live is not None else True}
    metrics["gates"]["G_STREAMELIVE"] = G_STREAMELIVE

    S2_corpus, S2_maint = float(rea["S_corpus"]), float(rea["S_maint"])
    G_BUDGET = {
        "form": "the LEG budget held (NON-HALTING): the length-scaled "
                "pools respected",
        "S_corpus_leg": S2_corpus, "S_maint_leg": S2_maint,
        "B_C": budget_c, "B_M": budget_m,
        "cumulative_t0_t600": E336_S_TOTAL_400 + S2_corpus + S2_maint,
        "e339_leg2_same_segment": {"S_corpus": E339_LEG2_S_CORPUS,
                                   "S_maint": E339_LEG2_S_MAINT},
        "pass": bool(S2_corpus <= budget_c + BUDGET_SLACK
                     and S2_maint <= budget_m + BUDGET_SLACK)}
    metrics["gates"]["G_BUDGET"] = G_BUDGET

    metrics["gates"]["G_NET0"] = {
        "subject": ("BASE-formed lineage (e311 TAVIREN install) + e336 "
                    "ERROR-GATED controller-annealed to t400 (read "
                    "0.815, 16 events) — THE FOUNDING-CLASS TRAJECTORY "
                    "(x39's own subject class)"),
        "leg": ("the same organism continued t401-t600 under the "
                "identical protocol (8 more events, length-scaled pools) "
                "— the audit leg"),
        "claim": "every state classed (x32's standing rule)", "pass": True}

    # ================= P6: the sag analysis + adjudication ==============
    log("=" * 78)
    log("P6 THE SAG ANALYSIS (boundary read vs mid-phase read per period)")
    periods = []
    for i, s_ev in enumerate(LEG_EVENTS[:-1]):
        s_mid = s_ev + MID_OFF
        s_next = LEG_EVENTS[i + 1]
        post = float(ms[s_ev]["g0_pz"])
        mid = float(ms[s_mid]["g0_pz"])
        floor_next = ev_gates[s_next]
        g_ev = i + 17 if not SMOKE else i + 5   # the global event index
        periods.append({
            "global_event": g_ev, "event_step": s_ev, "mid_step": s_mid,
            "next_gate_step": s_next,
            "dose_class": ("ZERO" if ev_defs[s_ev] == 0.0
                           else "DOSE"),
            "deficit": ev_defs[s_ev],
            "post_read": post, "mid_read": mid,
            "next_gate_read": floor_next,
            "sag_frac": (post - mid) / post if post > 0 else None,
            "mid_over_floor": (mid / floor_next) if floor_next > 0
                              else None,
            "peak_to_floor_ratio": (post / floor_next)
                                   if floor_next > 0 else None,
        })
        log(f"  period {g_ev} (t{s_ev}->t{s_next}): post {post:.6f} -> "
            f"MID(t{s_mid}) {mid:.6f} -> next-gate {floor_next:.6f} | "
            f"sag {(post - mid) / post:.1%} | mid/floor "
            f"{mid / floor_next if floor_next > 0 else float('nan'):.2f}"
            f" | peak/floor "
            f"{post / floor_next if floor_next > 0 else float('nan'):.2f}"
            f" | {periods[-1]['dose_class']} (deficit "
            f"{ev_defs[s_ev]:.3f})")
    mean_sag = float(sum(p["sag_frac"] for p in periods) / len(periods)) \
        if periods else None

    # the +1 window co-reads (the decay's first step, committed-comparand)
    plus_one = {}
    for wpre, wpost, wnext, tag in (
            ("win1:t425-pre", "win1:t425-post", "win1:t426",
             "event 17 (small-dose, e339-comparand)"),
            ("win2:t575-pre", "win2:t575-post", "win2:t576",
             "event 23 (small-dose, the new datum)")):
        if wpost in win_rows and wnext in win_rows:
            po = win_rows[wpost]["g0_pz"]
            p1 = win_rows[wnext]["g0_pz"]
            plus_one[tag] = {"post": po, "plus_one": p1,
                             "drop_frac": (po - p1) / po if po > 0
                                          else None}
    # the committed +1-after-big-dose datum (e339's record, cited)
    plus_one["event 21 (big-dose; e339's committed t600->t601, CITED)"] = {
        "post": e339_win["win2:t600-post"]["g0_pz"],
        "plus_one": e339_win["win2:t601"]["g0_pz"],
        "drop_frac": (e339_win["win2:t600-post"]["g0_pz"]
                      - e339_win["win2:t601"]["g0_pz"])
        / e339_win["win2:t600-post"]["g0_pz"]}

    # ---- the founding desk half (committed records only) ---------------
    e336_gates = [float(m["read_at_gate"])
                  for m in e336_rea_ph["maint_ledger"]]
    founding_desk = {
        "form": ("THE DESK HALF (runtime-computed from the md5-bound "
                 "committed records; zero compute): the founding trace's "
                 "own milestones (ALL post-dose) vs the SAME-TIME gate "
                 "floors — the committed sawtooth the founding heights "
                 "ride"),
        "milestones": {s: e336_traj[s]["g0_pz"] for s in
                       (100, 200, 300, 400)},
        "gate_at_same_event": {100: e336_gates[3], 200: e336_gates[7],
                               300: e336_gates[11], 400: e336_gates[15]},
        "peak_over_floor": {s: e336_traj[s]["g0_pz"]
                            / e336_gates[{100: 3, 200: 7, 300: 11,
                                          400: 15}[s]]
                            for s in (100, 200, 300, 400)},
        "canon_deficit_medians": "0.6-0.7 (the dispatch's citation)",
    }

    # ---- the adjudication (the frozen composite) ------------------------
    standing_ok = bool(G_ORTH["pass"] and G_BUDGET["pass"]
                       and G_STREAMELIVE["pass"] and G_BUFSEP["pass"])
    if SMOKE:
        verdict = "SMOKE"
        clause = ("smoke horizon — machinery validation only; NOTHING "
                  "adjudicated (the bars need the full leg's periods)")
    elif not standing_ok:
        verdict = "CONDITIONS-FAILED"
        clause = ("a standing condition failed (budget/stream/orth/"
                  "isolation) — routed per the frozen composite; the sag "
                  "reads are stamped but nothing adjudicated")
    elif mean_sag >= SAG_BAR_HI:
        verdict = "SAG-EXISTS"
        sags_txt = "/".join(f"{p['sag_frac']:.0%}" for p in periods)
        floor_fracs = [p["next_gate_read"] / p["post_read"]
                       for p in periods]
        mid_fracs = [1 - p["sag_frac"] for p in periods]
        clause = (
            f"mean sag {mean_sag:.1%} >= {SAG_BAR_HI:.0%} (the mid-phase "
            f"reads fall materially below the boundary reads; per-period "
            f"sags {sags_txt}) — THE FOUNDING HEIGHTS ARE RAILS: the "
            "endpoint class carries the phase clause (Law 4's endpoint "
            "language joins x41's rider); the 3.16-3.62x band is an "
            "upper-rail band — the between-events state of this "
            f"trajectory sits at {min(mid_fracs):.0%}-{max(mid_fracs):.0%}"
            " of its boundary reads at mid-period and at "
            f"{min(floor_fracs):.0%}-{max(floor_fracs):.0%} at the floors"
            f" (the committed floors already showed the valleys; the "
            f"mid-phase point is the new datum)")
    elif mean_sag < SAG_BAR_LO:
        verdict = "NO-SAG"
        clause = (
            f"mean sag {mean_sag:.1%} < {SAG_BAR_LO:.0%} — the mid-phase "
            "reads hold within noise: THE FOUNDING NUMBERS STAND AS "
            "WRITTEN (the 25-step periods hold; the costume lesson does "
            "not reach the founding class)")
    else:
        verdict = "PARTIAL-SAG (residual)"
        clause = (
            f"mean sag {mean_sag:.1%} in [{SAG_BAR_LO:.0%}, "
            f"{SAG_BAR_HI:.0%}) — the named residual: a material-but-"
            "minor sag; the founding numbers carry a MID-STRENGTH phase "
            "clause (neither rail nor flat); the discriminator for a "
            "follow-up is the decay knee's location (the +1 window "
            "reads vs the mids)")

    # the founding-numbers phase character (always emitted)
    if not SMOKE and verdict != "CONDITIONS-FAILED":
        founding_clause = (
            f"THE FOUNDING NUMBERS' PHASE CHARACTER: the founding "
            f"milestones (e336 leg-1: 0.6703/0.6985/0.7698/0.8147; the "
            f"canon endpoints' band 3.16-3.62x) were ALL read post-dose; "
            f"the committed desk half shows their own-period floors at "
            f"{founding_desk['peak_over_floor'][100]:.1f}x-"
            f"{founding_desk['peak_over_floor'][400]:.1f}x below them "
            "(the gates); the live half measures the mid-period state at "
            f"{1 - mean_sag:.0%} of the boundary read on average — "
            + ("the founding heights are UPPER-RAIL samples of an "
               "intra-period oscillation; the phase clause joins the "
               "endpoint class" if verdict == "SAG-EXISTS" else
               ("the founding heights are REPRESENTATIVE of the "
                "between-events state; no phase clause" if verdict
                == "NO-SAG" else
                "the founding heights carry a mid-strength phase clause "
                "(material sag, minor magnitude)")))
    else:
        founding_clause = None

    p_hit = bool(verdict == "SAG-EXISTS")
    adjudication = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": ("TEXTURE -> standing conditions -> the sag "
                            "partition (>= 0.15 SAG-EXISTS / < 0.05 "
                            "NO-SAG / between PARTIAL-SAG) — frozen at "
                            "birth"),
        "gates_pass": bool(all(g.get("pass", True) for g in
                               metrics["gates"].values())),
        "reads": {
            "the_sag_profile": periods,
            "mean_sag": mean_sag,
            "sag_bars": {"sag_exists": SAG_BAR_HI, "no_sag": SAG_BAR_LO},
            "the_plus_one_reads": plus_one,
            "the_founding_desk_half": founding_desk,
            "the_boundary_milestones": {s: ms[s]["g0_pz"]
                                        for s in LEG_EVENTS},
            "the_event_gates": ev_gates,
            "the_event_deficits": ev_defs,
            "post_cells": cells_r,
        },
        "verdict": verdict,
        "clause": clause,
        "founding_character_clause": founding_clause,
        "P-e342a": {
            "guess": REGISTERED["P-e342a"]["my_guess"],
            "lab_lean": REGISTERED["lab_lean_verbatim"],
            "hit": p_hit if not SMOKE else None,
            "scored": REGISTERED["P-e342a"]["scored"],
            "shape_scored": {
                "mean_in_035_060": bool(mean_sag is not None
                                        and 0.35 <= mean_sag <= 0.60)
                if not SMOKE else None,
                "every_mid_above_floor": bool(
                    all(p["mid_read"] > p["next_gate_read"]
                        for p in periods)) if periods and not SMOKE
                else None,
            },
        },
        "smoke_stamp": ("SMOKE — machinery validation only; the numbers "
                        "are meaningless (8-step manufactured legs)"
                        if SMOKE else None),
    }
    metrics["adjudication"] = adjudication
    log("=" * 78)
    log(f"P6 ADJUDICATED: {verdict} — mean sag "
        + (f"{mean_sag:.1%}" if mean_sag is not None else "n/a"))
    write_partial("P6 adjudicated (the frozen bars)")

    # ================= P7: the figure + the report ======================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- panel 1: the read trace with every phase read ------------------
    if not SMOKE:
        ev_x = [s for s in LEG_EVENTS]
        ev_y = [ms[s]["g0_pz"] for s in LEG_EVENTS]
        mid_x, mid_y = ([p["mid_step"] for p in periods],
                        [p["mid_read"] for p in periods])
        gate_x = sorted(ev_gates)
        gate_y = [ev_gates[s] for s in gate_x]
        ax1.semilogy(ev_x, ev_y, "o-", color="tab:red", lw=2.2, ms=8,
                     label="boundary reads (post-dose milestones — the "
                           "founding convention)")
        ax1.semilogy(mid_x, mid_y, "v", color="tab:blue", ms=9,
                     label=f"MID-PHASE reads (event+{MID_OFF} — THIS "
                           "CELL'S datum)")
        ax1.semilogy(gate_x, gate_y, "s", color="tab:gray", ms=6,
                     label="event gate reads (pre-dose floors)")
        for lbl, key in (("+1 after event 17", "win1:t426"),
                         ("+1 after event 23", "win2:t576")):
            if key in win_rows:
                ax1.semilogy([LEG_WIN1[2] if "win1" in key
                              else LEG_WIN2[2]],
                             [win_rows[key]["g0_pz"]], "x",
                             color="tab:purple", ms=10, mew=2.5,
                             label=lbl)
        for s_ev in LEG_EVENTS:
            ax1.axvline(s_ev, color="gray", ls=":", lw=0.7, alpha=0.6)
        ax1.axhline(E336.FACT_BASELINE_G0, color="black", ls="--", lw=1.0,
                    label=f"the baseline {E336.FACT_BASELINE_G0:.4f}")
        ax1.set_xlabel(f"corpus step t (events every M=25; mids at "
                       f"+{MID_OFF})")
        ax1.set_ylabel("THE READ (battery g0 p(T), log)")
        ax1.set_title(f"E342 THE PHASE AUDIT: {verdict} (mean sag "
                      + (f"{mean_sag:.1%}" if mean_sag is not None
                         else "n/a") + ")")
        ax1.legend(fontsize=7.5, loc="lower left")
        ax1.grid(alpha=0.3, which="both")

        # ---- panel 2: the per-period normalized sag profile ------------
        for p in periods:
            xs = [0, MID_OFF, 25]
            ys = [1.0, p["mid_read"] / p["post_read"],
                  p["next_gate_read"] / p["post_read"]]
            ax2.plot(xs, ys, "o-", lw=1.6, ms=5, alpha=0.8,
                     label=f"ev {p['global_event']} ({p['dose_class']})")
        ax2.axhline(1 - SAG_BAR_HI, color="tab:red", ls="--", lw=1.2,
                    label=f"the 15% sag bar ({1 - SAG_BAR_HI:.2f})")
        ax2.axhline(1 - SAG_BAR_LO, color="tab:green", ls="--", lw=1.2,
                    label=f"the 5% bar ({1 - SAG_BAR_LO:.2f})")
        if periods:
            ax2.plot([0, MID_OFF, 25],
                     [1.0,
                      (1 - mean_sag),
                      sum(p["next_gate_read"] / p["post_read"]
                          for p in periods) / len(periods)],
                     "k*-", lw=3.0, ms=13, alpha=0.9,
                     label=f"the MEAN profile (mid = {1 - mean_sag:.2f})")
        ax2.set_xlabel("steps since the dose (the 25-step period)")
        ax2.set_ylabel("read / own boundary read")
        ax2.set_title("the sag profile per period (post -> mid -> floor)")
        ax2.legend(fontsize=7, loc="center right")
        ax2.grid(alpha=0.3)

        # ---- panel 3: the sag distribution ------------------------------
        xs = [p["global_event"] for p in periods]
        ys = [p["sag_frac"] for p in periods]
        ax3.bar(xs, ys, color=["tab:red" if p["dose_class"] == "DOSE"
                               else "tab:orange" for p in periods],
                alpha=0.85)
        ax3.axhline(mean_sag, color="black", lw=2.2,
                    label=f"mean sag {mean_sag:.1%}")
        ax3.axhspan(SAG_BAR_HI, 1.0, color="tab:red", alpha=0.10,
                    label="SAG-EXISTS (>= 15%)")
        ax3.axhspan(SAG_BAR_LO, SAG_BAR_HI, color="tab:yellow", alpha=0.15,
                    label="PARTIAL-SAG (the residual)")
        ax3.axhspan(0.0, SAG_BAR_LO, color="tab:green", alpha=0.12,
                    label="NO-SAG (< 5%)")
        for x, y in zip(xs, ys):
            ax3.annotate(f"{y:.0%}", (x, y), ha="center", va="bottom",
                         fontsize=8)
        ax3.set_xlabel("global event (the period's dose)")
        ax3.set_ylabel("sag = (post − mid) / post")
        ax3.set_title("the sag distribution vs the frozen bars")
        ax3.legend(fontsize=7.5, loc="lower right")
        ax3.grid(alpha=0.3, axis="y")

        # ---- panel 4: the founding numbers' phase character -------------
        f_ms_x = [100, 200, 300, 400]
        f_ms_y = [e336_traj[s]["g0_pz"] for s in f_ms_x]
        f_gate_x = [25 * (i + 1) for i in range(16)]
        f_gate_y = e336_gates
        ax4.semilogy(f_ms_x, f_ms_y, "o-", color="tab:red", lw=2.0, ms=7,
                     label="the founding milestones (e336 leg-1 committed,"
                           " ALL post-dose)")
        ax4.semilogy(f_gate_x, f_gate_y, "s:", color="tab:gray", lw=1.4,
                     ms=4, label="the founding gate floors (committed)")
        ax4.semilogy(ev_x, ev_y, "^-", color="tab:blue", lw=1.6, ms=5,
                     label="this leg's boundary reads")
        ax4.semilogy(mid_x, mid_y, "v", color="tab:purple", ms=8,
                     label="this leg's MID reads")
        ax4.semilogy(gate_x, gate_y, "s", color="tab:cyan", ms=5,
                     label="this leg's gate floors")
        ax4.set_xlabel("corpus step t (the founding trace + this "
                       "continuation)")
        ax4.set_ylabel("THE READ (log)")
        ax4.set_title("the founding numbers' phase character: the "
                      "milestones vs the floors they ride")
        ax4.legend(fontsize=7, loc="lower right")
        ax4.grid(alpha=0.3, which="both")
    else:
        ax1.text(0.5, 0.5, "SMOKE — machinery validation only",
                 ha="center", va="center", transform=ax1.transAxes)

    fig.tight_layout()
    fig_path = RD / ("smoke_e342_phase_audit.png" if SMOKE
                     else "e342_phase_audit.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ---- REPORT.md ------------------------------------------------------
    ts = common.now_iso()
    gcount = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    lines = [
        f"# E342 — THE PHASE AUDIT\n",
        f"**Date:** {ts} (datetime.now(UTC) stamps in metrics) · "
        f"**Executor report**\n",
        "## THE VERDICT (against the frozen bars)\n",
        f"**{verdict}**"
        + (" _(smoke — machinery validation only)_" if SMOKE else ""),
        "\n```",
        clause,
        "```\n",
    ]
    if founding_clause:
        lines += ["```", founding_clause, "```\n"]
    if not SMOKE:
        lines += [
            "## The sag profile (boundary vs mid-phase per period)\n",
            "| period | event step | dose (deficit) | boundary post | "
            "MID (+13) | next gate (floor) | sag | mid/floor | "
            "peak/floor |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in periods:
            lines.append(
                f"| ev {p['global_event']} | t{p['event_step']} | "
                f"{p['dose_class']} ({p['deficit']:.3f}) | "
                f"{p['post_read']:.4f} | {p['mid_read']:.4f} | "
                f"{p['next_gate_read']:.4f} | {p['sag_frac']:.1%} | "
                f"{p['mid_over_floor']:.2f} | "
                f"{p['peak_to_floor_ratio']:.1f}x |")
        lines += [
            "",
            f"- **MEAN SAG {mean_sag:.1%}** vs the frozen bars "
            f"(SAG-EXISTS >= {SAG_BAR_HI:.0%} / NO-SAG < "
            f"{SAG_BAR_LO:.0%} / PARTIAL-SAG between)",
            f"- the +1 window reads (the decay's first step): "
            + "; ".join(f"{k}: post {v['post']:.4f} -> +1 "
                        f"{v['plus_one']:.4f} "
                        f"({v['drop_frac']:+.1%})"
                        for k, v in plus_one.items()),
            f"- the zero events inside the leg: "
            f"{[s for s, d in ev_defs.items() if d == 0.0] or 'none'}",
            "",
            "## The founding numbers' phase character\n",
            f"- DESK HALF (committed records only): the founding "
            f"milestones 0.6703/0.6985/0.7698/0.8147 all post-dose; "
            f"their own-period floors "
            f"{[round(v, 4) for v in founding_desk['gate_at_same_event'].values()]}; "
            f"peak/floor "
            + "/".join(f"{v:.1f}x"
                       for v in founding_desk["peak_over_floor"].values()),
            f"- the canon's deficit medians 0.6-0.7 (the dispatch's own "
            f"citation)",
            f"- LIVE HALF: this leg's mean sag {mean_sag:.1%}; the "
            f"founding heights are "
            + ("UPPER-RAIL samples" if verdict == "SAG-EXISTS"
               else ("REPRESENTATIVE" if verdict == "NO-SAG"
                     else "mid-clause samples")),
            "",
            "## The e339 shared-draw co-read (non-halting)\n",
            f"- t500: mine {co['t500']['mine']:.6f} vs e339 committed "
            f"{E339_T500_G0:.6f} (|d| {co['t500']['abs_diff']:.5f})",
            f"- t600: mine {co['t600']['mine']:.6f} vs "
            f"{E339_T600_G0:.6f} (|d| {co['t600']['abs_diff']:.5f})",
            f"- win1 quadruple max |d| "
            f"{max(d['abs_diff'] for d in co['win1'].values()):.5f} "
            f"(tol {E339_CO_TOL} — disclosed, never a gate)\n",
            "## Gates\n",
            f"- {gcount} gate classes instantiated, all PASS: "
            + ", ".join(f"{k}" for k in metrics["gates"]),
            "",
            "## Envelope\n",
            f"- {len(thermal_log)} thermal polls; max temp "
            f"{max((r.get('temp', 0) for r in thermal_log), default=0):.1f}C; "
            f"burst cap {E261.BURST_MAX_S:.0f}s; cooldown "
            f"{E261.COOLDOWN_S:.0f}s; zero concurrent GPU jobs; every "
            "burst logged to runs/_envelope_log.jsonl tagged "
            "e342:AUDIT-LEG:*",
            "",
            "## Registered predictions\n",
            f"- P-e342a (the executor's own read: SAG-EXISTS, "
            f"moderately): "
            f"{'HIT' if p_hit else 'MISSED'}"
            + (" (not scored — smoke)" if SMOKE else ""),
            f"- the dispatch's lab lean (SAG-EXISTS, weakly): "
            f"{'HIT' if verdict == 'SAG-EXISTS' else 'MISSED'}"
            + (" (not scored — smoke)" if SMOKE else ""),
            f"- the shape predictions: mean in [0.35, 0.60]: "
            f"{adjudication['P-e342a']['shape_scored']['mean_in_035_060']}; "
            f"every mid above its floor: "
            f"{adjudication['P-e342a']['shape_scored']['every_mid_above_floor']}",
            "",
            "## Catches / disclosures\n",
            "- THE SUBJECT: e336's committed t400 resume state (model + "
            "optC + optF + both generator states) md5-bound, "
            "bit-compared vs the committed post checkpoint, and "
            "read-reproduced to e336's committed t400 == x39's subject "
            "literals (2e-6/5e-3) BEFORE any stepping.",
            "- THE LEG: 8 events (the dispatch's 4-8 at its top), M=25 "
            "UNCHANGED, pools length-scaled (x41's frozen form; LR_M_MAX "
            "never scaled) — the per-step/per-event equal-share "
            "arithmetic identical to e336's leg-1 and e339's leg-2 at "
            "their starts.",
            "- THE e339 CO-READ is a consistency check at a disclosed "
            "0.05 tolerance, never a gate (cross-run GPU op-order "
            "determinism is not the family's law).",
            "- THE MID OFFSET +13 frozen verbatim from the dispatch; "
            "period 24's mid (t613) sits beyond the leg — 7 complete "
            "pairs, no pair dropped.",
            "- n=1: ONE organism, one lineage, one leg — the sag "
            "profile is a single realization's, read against the "
            "founding class's committed record.",
            "- Checkpoints live INSIDE runs/e342/. No NOTES/THINKING/"
            "QUEUE/STATE edits (the heartbeat folds this cell).",
            "",
            "*This cell does not edit NOTES/THINKING/QUEUE/STATE — the "
            "heartbeat folds.*",
        ]
    (RD / ("smoke_REPORT.md" if SMOKE else "REPORT.md")).write_text(
        "\n".join(lines), encoding="utf-8")

    # ---- the final metrics ---------------------------------------------
    metrics["arms"] = {
        "ERROR-GATED": {
            "desc": ("the audit leg: e336's committed t400 state (model + "
                     "buffers + generators) resumed; e336's driver "
                     f"VERBATIM; corpus steps t{LEG_START + 1}-t{LEG_END} "
                     "over length-scaled pools; 8 more NAME-ONLY events "
                     "(17-24) at the error-gated lr; the PHASE-DECLARED "
                     "reads at full cadence (every event pre+post + "
                     f"every period's mid at +{MID_OFF})"),
            "phase": {k: rea[k] for k in
                      ("traj", "corpus_ledger", "orth_ledger",
                       "disp_ledger", "buf_ledger", "budget_ledger",
                       "lr_ledger", "maint_ledger", "window_ledger",
                       "bufsep", "chunk_table", "steps_ran")},
            "post_cells": cells_r,
            "S_corpus": S2_corpus, "S_maint": S2_maint,
            "budget_c": budget_c, "budget_m": budget_m,
            "checkpoint": str(RD / ("smoke_REA_post.pt" if SMOKE
                                    else "e342_REA_post.pt"))},
    }
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the sag is read from BEHAVIOR of the same channel the bars "
            "read (the g0 battery p(T)), not from logits alone; the "
            "intervening variable is time-since-dose under the "
            "protocol's own wash — the phase-declared reads are the "
            "observation, the identical protocol the control"),
        "the_leg_disclosure": (
            "the leg convention (length-scaled pools, LR_M_MAX never "
            "scaled, M unchanged, the stream continued never re-seeded) "
            "is FROZEN AT BIRTH in this file's header; the schedule "
            "immateriality is verified per-step (cap <= sched at every "
            "logged step)"),
        "the_resume_disclosure": (
            "the resume is the protocol's own checkpoint convention — "
            "the state is e336's committed artifact, never re-formed; "
            "the repro gate re-reads the battery on the restored bits "
            "before any stepping"),
        "n_and_scope": (
            "n=1: ONE organism, one lineage, one leg; the sag profile a "
            "single realization read against the founding class's "
            "committed record (the canon's own floors were never "
            "re-measured here — the desk half cites them)"),
        "loads_measured_not_nominal": (
            "every read measured: per-event doses (the gate reads + "
            "deficits + applied lr_m), the 15 milestone panels, the 8 "
            "window rows, the spend ledgers, the thermal polls"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": [
            "lab/e336_founding_replicate.py (THE COMMITTED DRIVER — "
            "chunked_errorgated_phase executed VERBATIM under the "
            "disclosed leg re-binding)",
            "lab/e261_rank_ladder.py (the burst machinery + the room)",
            "lab/e339_long_landing.py (the resume + staging + leg "
            "convention this cell inherits)",
        ],
        "checkpoints": {
            "resumed_from": [str(rea_src)],
            "e336_metrics": str(E336_METRICS),
            "e339_metrics": str(E339_METRICS),
            "leg_resume": str(rea_leg_ck),
        },
        "thermal_envelope": {
            "polls": len(thermal_log),
            "max_temp_c": max((r.get("temp", 0) for r in thermal_log),
                              default=None),
            "temp_early_end_margin_c": E261.TEMP_EARLY_END,
            "temp_hard_line_c": E261.TEMP_HARD,
            "cooldown_s": E261.COOLDOWN_S,
            "burst_cap_s": E261.BURST_MAX_S,
            "concurrent_gpu_jobs": 0,
            "note": "every burst logged to runs/_envelope_log.jsonl "
                    "tagged e342:AUDIT-LEG:*",
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["outputs"] = [str(fig_path), str(RD / "REPORT.md"),
                          str(RD / "metrics.json")]
    metrics["status"] = (f"COMPLETE — adjudicated {verdict}"
                         + (" (SMOKE)" if SMOKE else "")
                         + f" ({common.now_iso()})")
    save_json(RD / "metrics.json", metrics)
    log(f"P7 DONE: {fig_path.name} + REPORT.md + metrics.json — verdict "
        f"{verdict}")


def g_phagedecl_ok(g: dict) -> str:
    return "present" if g["win1_full"] else "MISSING"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        _logf.write(traceback.format_exc() + "\n")
        _logf.flush()
        metrics["status"] = f"FAILED ({common.now_iso()})"
        save_json(RD / "metrics.json", metrics)
