"""E296 — THE PROSTHETIC GRAFT (maintenance with NO teaching signal) — the
wild queue's #1. This docstring carries the registered question + THE FROZEN
BARS VERBATIM from the dispatch letter, committed at birth BEFORE any
compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION: the controller works by error-gated re-teaching (a signal).
The graft asks: can a fact be maintained by REFRESHING ITS GEOMETRY ALONE —
periodically transplanting the write's in-room projection from a frozen
external copy (theta <- theta + alpha*(P_room w_frozen - P_room theta),
under the same budget language), with NO gradient, NO loss, NO signal? If
the graft holds: memory maintenance reduces to consolidation-from-checkpoint
— the anesthesia metaphor, the cheapest endgame. If it fails where the
controller holds: the signal is load-bearing and maintenance is genuinely
re-teaching.

THE RIG (e288's, ported whole — READ FIRST per the dispatch: the
established fact, the traffic, the budget): the committed quiet-formed 10k
fact (e261_K10K_inst_resume) loaded BIT-EXACT + gated three ways; the fact's
OWN committed K10K room (seeds 26113/26114) rebuilt + bit-gated vs
e264_rooms.pt; the e268 corpus step VERBATIM on THIS cell's ONE fresh
registered generator seed 29601 (BOTH arms draw the IDENTICAL sequence);
the orthogonalized corpus stream (g_perp, verified EVERY step) under the
DISPLACEMENT BUDGET (BUDGET = 0.5 x ||write||, S_total = S_corpus +
S_actuator BOTH counted) with e288's SPLIT pools (B_C = 60% corpus /
B_act = 40% actuator, each capped by the symmetric equal-share form);
M=25 cadence, 400 corpus steps, milestones t=100/200/300/400 read
POST-actuator; windows t24/t25pre/t25post/t26 AND t199/t200pre/t200post/
t201; SGD-M at LR_STABLE on the corpus side; thermal envelope 175/40/78/84
(the dispatch's 180/30-60/85 all inside).

THE ARMS (the actuator is the arms' ONLY delta — same draws, same budget,
same gating LAW FORM: dose on the read's deficit, capped at the stream's
40% share):
  (a) GRAFT (the build): every M=25 steps, ONE GRAFT STEP — the frozen-copy
      transplant, a DIRECT PARAMETER EDIT through NO optimizer:
          theta <- theta + alpha_t * P_room(fact - theta)
      (in displacement language this IS theta <- theta + alpha*(P_room
      w_frozen - P_room theta): P_room linear, the base cancels). The dose
      gated THE SAME WAY as the controller — deficit_t = max(0, BASELINE -
      read)/BASELINE measured on the SAME g0 battery the bars read; alpha_t
      = min(ALPHA_MAX * deficit_t, cap_g) with ALPHA_MAX = 1.0 (at full
      deficit the FULL transplant: the in-room gap closed entirely — the
      anesthesia ideal; there is no momentum ramp to calibrate against —
      NO optimizer — so the equal-share cap does the budget-fitting: cap_g
      = share_g/||gap||, share_g = (B_G - S_graf)/rem_graf). NO gradient,
      NO loss, NO optimizer, NO install draws — the graft is deterministic
      geometry. The read's RESPONSE per graft is the registered read.
  (b) CONTROLLER-TWIN (e288's form VERBATIM, same session/draws — the
      signal-borne reference): e288's ERROR-GATED arm — the NAME-ONLY CE
      on win_i's 112 masked name positions (install stream seed 29602),
      clip 1.0, UNPROJECTED, through opt_F (the fact-side SGD-M) at
      lr_m_t = LR_M_MAX * deficit_t (LR_M_MAX = 0.40 x BUDGET / 81.5736...
      = 0.02250444534525284, re-derived + asserted), capped at the B_M
      share. Its committed record (e288 ERROR-GATED-HOLDS x3.4792) is this
      cell's hard-bound cite.

FROZEN BARS (survival at t400; the budget <= 100%; the stream live) —
VERBATIM from the dispatch:
  - GRAFT-HOLDS: ">= 0.5x with the budget held and the stream live —
    maintenance needs NO signal: memory preservation is geometry refresh;
    the anesthesia endgame."
  - SIGNAL-NEEDED: "the graft < 0.5x while the controller twin holds — the
    teaching signal is load-bearing; maintenance is genuinely re-teaching
    (and the era's economics deepen)."
  - MIXED: "partial (the graft slows the decay but does not stop it —
    geometry buys time, signal buys survival; a two-channel account of
    maintenance)."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * "survival at t400" := post-graft/step g0 battery read at t400 / the
    committed baseline 0.26464763283729553 (PRIMARY; the session
    denominator co-reported).
  * "the budget held" := S_total = S_corpus + S_act <= BUDGET + 1e-5 slack
    AND the realized cumulative displacement <= BUDGET + slack, BOTH arms
    (per-pool: S_corpus <= B_C, S_act <= B_act on both — both arms use
    e288's split pools; non-halting — a break routes the composite to
    MIXED, everything verbatim).
  * "the stream live" := e287/e288's form VERBATIM per arm: median of the
    arm's logged corpus-batch CE rows t in (300,400] vs rows t in [1,100];
    improving := late < early; stable := late <= 1.05 x early; live :=
    improving OR stable.
  * "the controller twin holds" := twin ratio_400 >= 0.5 (the HOLDS bar
    applied to the twin) with the twin's budget held and the twin's stream
    live — the reference must itself be sound.
  * MIXED-partial (the "slows the decay but does not stop it" clause,
    frozen): graft ratio_400 < 0.5 AND the twin holds AND BOTH engagement
    conditions: (i) S_graf >= 0.01 x B_G (the graft SPENT at least 1% of
    its pool — it engaged) AND (ii) graft ratio_400 >= 10 x e285's
    committed sanctuary ratio 0.00303243599263398 (the read sits >= 10x the
    orthogonal-passive death class — the decay demonstrably slowed, not
    just noise). Engagement without slowing -> SIGNAL-NEEDED; slowing
    without engagement is not implementable (a no-spend graft moved
    nothing).
  * THE COMPOSITE := TEXTURE (any bind/isolation hard-gate failure —
    nothing adjudicated, HALT) -> MIXED-by-conditions (G_BUDGET blown OR
    G_ORTH failed, any ratio — no bar covers a broken standing condition;
    everything verbatim) -> GRAFT-HOLDS (graft ratio_400 >= 0.5 AND budget
    held AND the graft arm's stream live) -> [twin holds] MIXED-partial
    (both engagement conditions) / SIGNAL-NEEDED (else) -> MIXED (the twin
    itself failed to hold — no clean reference; everything verbatim).
  * THE TRANSPLANT ARITHMETIC (the dispatch's smoke demand, machine-live
    at EVERY graft event): one graft step must move P_room theta toward
    P_room w_frozen by EXACTLY alpha times the gap — asserted as
    ||P_room(fact - theta_new)|| == (1-alpha) x ||P_room(fact - theta_old)||
    (fp32-write tolerances: abs 1e-4 / rel 1e-3; the fp64-exact unit test
    is G_TRANSPLANT, run at startup in BOTH smoke and full) AND the edit
    displaces by ||alpha x gap|| (rel 1e-3) AND the out-of-room component
    is untouched within the same fp32 floor AND alpha <= ALPHA_MAX always,
    alpha == 0 at deficit 0.
  * G_GRAFISOL (the graft arm's no-signal isolation, HARD): the graft edit
    touches NO optimizer state (opt_C's buffers snapshotted bitwise around
    EVERY graft event) and NO gradient EVER exists on the graft path
    (asserted: the grads bitwise-unchanged through every edit — the
    graft path computes none; the corpus step's leftovers pass through
    untouched).
  * HARD GATES (a failure HALTs): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_PARENTS, G_BASE, G_ROOT, G_VMBIND,
    G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND,
    G_MAINTBIND, G_TRANSPLANT, G_GRAFTBIND, G_GRAFISOL, G_BUFSEP-isolation
    (the twin)}; NON-HALTING (route the composite to MIXED): {G_ORTH,
    G_BUDGET, bufC-composition}.
  * NO CONS (T259/e281; the family's committed form): the frozen bars read
    the WRITE and the DISPLACEMENT only; both arms' post states
    CHECKPOINTED for any later landing pass.

REGISTERED READS (the dispatch): post g0 at t100/200/300/400 (both arms);
the graft trace (alpha, displacement spent, the read's response per graft);
the budget; the corpus CE. PLUS the in-room gap ledger (the engagement
datum — see P-e296c) and the windows (both directions, both arms).

REGISTERED PREDICTIONS (the executor's, frozen at birth BEFORE compute):
  - P-e296a (the anesthesia hypothesis): GRAFT-HOLDS — the periodic
    in-room refresh alone preserves the read under traffic; maintenance
    reduces to consolidation-from-checkpoint.
  - P-e296b (the signal hypothesis): SIGNAL-NEEDED — the graft < 0.5x
    while the twin holds (the founding success replicates in-session);
    the teaching signal is load-bearing.
  - P-e296c (the engagement bound — the executor's structural read,
    registered BEFORE compute): under the era's ORTHOGONALIZED corpus
    stream the in-room gap ||P_room(fact - theta)|| is pinned at the fp
    floor (e285's committed sanctuary law: drift 1.8445 with in-room
    1.23e-6 — the orthogonalized stream cannot move the room-frame), so
    the graft's displacement is bounded by ~alpha x fp-floor and the graft
    CANNOT ENGAGE: the expected landing is SIGNAL-NEEDED with S_graf ~ 0
    and the graft curve riding the orthogonal-passive death class, and the
    finding then names the MECHANISM: the read's death is OUT-of-room (the
    room-frame is not the operative frame — T267's law, now at the
    maintenance actuator), the anesthesia endgame is structurally
    unavailable in this rig, and only the signal's out-of-room brush
    (e288: maint in-room ~0.0603; e313's brush class band [0.055, 0.065])
    reaches the wound. DISCRIMINATOR: the graft trace's per-event gap_norm
    column — gap >= 1e-4 anywhere (any in-room leakage) means the graft
    engages and P-e296a/P-e296b adjudicate on the bars with the trace as
    the datum; gap ~ fp-floor everywhere confirms the structural reading.
    This prediction is registered to keep the null honest — not to move
    the bars.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (inside the dispatch's 180s), per-step thermal polls at a 78C
margin, 40s cooldowns (the 30-60s window), the 84C never-past line (inside
the dispatch's 85C), polls persisted to runs/_envelope_log.jsonl tagged
e296:<ARM>:<phase>; CPU fp64 dense projections; CPU probing threads 4; NO
concurrent GPU jobs (the arms run sequentially with cooldowns between).
TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/e296/{metrics.json (PROGRESSIVE), e296_prosthetic_graft.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e296_*.pt (gitignored; md5s in metrics). No NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit +
push per phase.

Run:  cd lab && python e296_prosthetic_graft.py    (E296_SMOKE=1 shakedown)
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
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # CORP_BS, MIX_RANDOM,
                                                      # LR, jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # MACHINERY, PORTED
                                                      # WHOLE BY IMPORT (the
                                                      # committed file is
                                                      # NOT modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E296_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e296_smoke" if SMOKE else "e296"
assert torch.cuda.is_available(), "e296 owns the GPU lane (dispatch)"

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
# e261's drivers land their rows in THIS cell's ledger)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e268/e273/e278/e283/e284/e285/e288's disclosed
# convention): e261's drivers resolve their module globals AT CALL TIME
# through e261's module namespace — rebound HERE so the room build +
# envelope polls label THIS cell. The committed lab/e261_rank_ladder.py is
# untouched.
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
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (the K10K bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE ROOM: the committed K10K room (k=10k, seeds 26113/26114) — the
# room the established fact was WRITTEN IN (the vehicle's own room)
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — e261's registered seed pair
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (512, 26113, 26114),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG = {k: ("K10K" if not SMOKE else f"K{k}") for k, _, _ in LADDER}
ROOM_MODE = RUNG[LADDER[0][0]]       # the room's mode key (smoke names it K512)
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# the arms (execution order: the GRAFT first — the question's arm; then the
# CONTROLLER-TWIN — the signal-borne reference)
ARMS = ("GRAFT", "CONTROLLER-TWIN")
GRF_ARM, TWN_ARM = ARMS
ARM_DESC = {
    GRF_ARM: "(a) THE GRAFT (the build, maintenance with NO teaching "
             "signal): e288's rig VERBATIM (the loaded established 10k "
             "fact; the orthogonalized budgeted corpus stream over B_C = "
             "60%; M=25 cadence; the same milestones/windows/reads) with "
             "the maintenance actuator REPLACED by the frozen-copy "
             "transplant — every M=25 corpus steps ONE GRAFT STEP: a "
             "DIRECT PARAMETER EDIT theta <- theta + alpha_t * P_room"
             "(fact - theta), through NO optimizer, with NO gradient, NO "
             "loss, NO install draws. The dose gated THE SAME WAY as the "
             "controller (deficit_t on the SAME g0 battery; alpha_t = min("
             "ALPHA_MAX x deficit_t, cap_g); ALPHA_MAX = 1.0 — at full "
             "deficit the FULL transplant) under the SAME budget language "
             "(cap_g = share_g/||gap||, share_g = (B_G - S_graf)/"
             "rem_graf, B_G = 40% of budget). 400 corpus steps, 16 graft "
             "steps.",
    TWN_ARM: "(b) THE CONTROLLER-TWIN (e288's ERROR-GATED form VERBATIM, "
             "same session/draws — the signal-borne reference): the "
             "NAME-ONLY CE on win_i's 112 masked name positions (the true "
             "signal; install generator seed 29602), clip 1.0, "
             "UNPROJECTED, through opt_F (the fact-side SGD-M, private "
             "persistent buffer) at the error-gated dose lr_m_t = LR_M_MAX "
             "x deficit_t (LR_M_MAX = 0.40 x BUDGET / 81.57360134901982, "
             "re-derived + asserted), capped at the B_M = 40% share; the "
             "corpus stream seed 29601 + the split pools B_C = 60%/"
             "B_M = 40% — e288's arm VERBATIM on THIS cell's stream. The "
             "arms' ONLY delta: the actuator (geometry transplant vs "
             "signal gradient). e288's committed record "
             "(ERROR-GATED-HOLDS x3.4792) is the hard-bound cite.",
}

# THIS cell's ONE fresh registered corpus stream (the e268-family
# convention: .../28801/.../HERE 29601). BOTH arms draw the IDENTICAL
# sequence — bit-identical corpus batches across arms, the actuator the
# arms' ONLY delta. The INSTALL stream (the TWIN's maintenance draws) has
# its OWN registered generator, seed 29602 (disclosed; the corpus stream
# untouched). THE GRAFT ARM HAS NO INSTALL DRAWS (the transplant is
# deterministic geometry).
CORPUS_GEN_SEED = 29601
INST_GEN_SEED = 29602

# THE CONVENTION FREEZES (e283/e285/e288's, carried)
PHASE_STEPS = 8 if SMOKE else 400               # CORPUS steps per arm
MILESTONES = tuple(range(1, 9)) if SMOKE else (100, 200, 300, 400)

# ---- THE ACTUATOR SCHEDULE (frozen) --------------------------------------
ACT_EVERY = 2 if SMOKE else 25       # one actuator step after every M corpus steps
N_ACT_EXPECTED = PHASE_STEPS // ACT_EVERY    # 16 full / 4 smoke

# THE BETWEEN-STEPS WINDOW READS (the family's two windows — the SAME two
# as e287/e288, direct comparability with the parent ledgers; smoke scales)
WIN1 = (24, 25, 26) if not SMOKE else (1, 2, 3)
WIN2 = (199, 200, 201) if not SMOKE else (3, 4, 5)

# THE ESTABLISHED FACT: the committed quiet-formed 10k write — e261's serial
# cut, completed by e264 (the committed threshold rung's final state)
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_STEP = 400
FACT_TRAJ_STEPS = [1, 100, 200, 300, 400]
FACT_LEDGER_MAX = 400

# ---- THE SGD-M CONFIG (frozen; e280/e284/e285/e288's committed lr class)
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
E273_LR_SGD = 21.7385748014537         # the calibration record's literal
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.21738574801453703

# ---- THE TWIN'S CONTROLLER LR (e288's calibration, frozen + asserted) ----
MAINT_BUDGET_SHARE = 0.40          # the actuator stream's allocation (BOTH arms)
CORPUS_BUDGET_SHARE = 0.60          # the corpus stream's allocation (BOTH arms)
assert abs(MAINT_BUDGET_SHARE + CORPUS_BUDGET_SHARE - 1.0) < 1e-12
# e287's committed per-event pending-momentum norms (the twin's
# calibration reference trace; md5-bound in G_PARENTS and re-read at runtime)
E287_B_M_TRACE = (
    0.999999850988377, 1.8681899695586686, 2.6312390614914225,
    3.3060561067374516, 3.88561848925063, 4.429405028471657,
    4.8863053913153305, 5.328364481536122, 5.756600944763833,
    6.115254960441239, 6.437143852974447, 6.718666537453984,
    6.968961840775396, 7.2139469698486876, 7.42026452510681,
    7.607583338305771,
)
E287_B_M_SUM = 81.57360134901982    # sum(E287_B_M_TRACE), asserted at import
assert abs(sum(E287_B_M_TRACE) - E287_B_M_SUM) < 1e-9
# the frozen literal (0.40 x 4.58942163293615 / 81.57360134901982); the
# runtime re-derivation (after the write_norm load) must match to 1e-9
LR_M_MAX_FROZEN = 0.02250444534525284

# ---- THE GRAFT'S OWN DIAL (frozen HERE at birth, BEFORE compute) --------
# ALPHA_MAX = 1.0: at full deficit the FULL transplant — the in-room gap
# closed ENTIRELY (P_room theta := P_room w_frozen at alpha = 1 — the
# consolidation-from-checkpoint ideal). There is NO momentum ramp to
# calibrate a ceiling against (NO optimizer on the graft path — e287's
# b_m trace is inapplicable VERBATIM; disclosed), so the BUDGET-FITTING is
# the cap's job alone: cap_g = share_g/||gap|| with share_g = (B_G -
# S_graf)/rem_graf the symmetric equal-share reservation (the same form as
# the corpus cap and the twin's maintenance cap) — at 16 events x full
# deficit the graft spends min(sum of gaps, B_G): S_graf <= B_G by
# construction, S_total <= BUDGET by the triangle bound.
ALPHA_MAX = 1.0

# ---- THE DISPLACEMENT BUDGET (frozen; e285's DIAL 3 / e288's, verbatim)
BUDGET_FRAC = 0.5                 # "0.5 x the write's own norm"
BUDGET_SLACK = 1e-5               # fp32 accumulation slack on S + cum (abs)

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_VERDICT = "SHARP-THRESHOLD"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"
E268_VERDICT = "DYNAMICAL-CARRIER"
E268_CONCURRENT_POST = 4.004325455753133e-05  # the FORMING write under fire
E268_RATIO = 0.00015130794937726159          # ~0.0002x — the forming death

E283_METRICS = E43.REPO / "runs" / "e283" / "metrics.json"
E283_MD5 = "cf5be012f636ede7b953eb9a615c60a5"
E283_VERDICT = "ESTABLISHED-DIES"
E283_POST = 9.04614535102155e-06             # the unprotected reference
E283_RATIO = 3.4181848724802654e-05          # 0.0000342x — the cite
E283_DRIFT = 14.453935847208887              # the transport read (||d||)
E283_WRITE_NORM = 9.1788432658723            # the write's own norm

E284_METRICS = E43.REPO / "runs" / "e284" / "metrics.json"
E284_MD5 = "18ad7e334c4bb9b52e494f1739842ac6"
E284_VERDICT = "MOMENTUM-OWNED"
E284_SEP_PRIMARY = 2.319773558897833e-07     # the separate-buffer in-room share
E284_SHA_PRIMARY = 0.45027988873803715       # the shared twin's funnel

X14_METRICS = E43.REPO / "runs" / "x14" / "metrics.json"
X14_MD5 = "4970e27ae8c315df8762e5c3499a2be5"
X14_VERDICT = "MIXED/INCONCLUSIVE"
X14_ARM_A = 0.03915366902947426              # the orthogonal-subtraction read
X14_ARM_B = 5.864363629370928e-05            # the in-room-subtraction read
X14_RESURRECTION_X = 4328.215777016314       # arm A / the dead state

# e285's committed record (the orthogonal-passive sanctuary — THE GRAFT
# ARM'S STRUCTURAL TWIN UNDER P-e296c: the orthogonalized stream pins the
# in-room geometry, the read dies anyway)
E285_METRICS = E43.REPO / "runs" / "e285" / "metrics.json"
E285_MD5 = "3f71c7b209fe9b6e9645d7504ddbb49c"
E285_VERDICT = "AIM-ONLY-KILLS"
E285_SAN_POST = 0.0008025270071811974        # the sanctuary's dead read
E285_SAN_RATIO = 0.00303243599263398         # 0.0030x
E285_SAN_DRIFT = 1.8445091953053947           # 20.1% of the write (the
                                               # CURRENT committed record;
                                               # e288's bind read
                                               # 1.844509195284748 — the
                                               # e285 record was rewritten
                                               # after e288's fold at a
                                               # 2e-8 fp discrepancy;
                                               # disclosed in deviations)
E285_SAN_DRIFT_INROOM = 1.2260267392361931e-06   # fp floor (out-of-room)
E285_S_FINAL = 4.589421633863822             # 100.00000002% of budget
E285_BUDGET_NORM = 4.58942163293615
E285_N_CAPPED = 397
E285_ORTH_MAX = 4.555131955207115e-17
E285_T100_DRIFT = 0.6609871202355514         # 7.2% of the write — the kill
E285_T100_RATIO = 0.005076788048356657       # ~200x down at t100
E285_TWN_POST = 3.895893769367831e-06        # the same-session twin
E285_TWN_RATIO = 1.472106033067378e-05
E285_TWN_DRIFT = 14.432213219853237
E285_COUPLING_BAR = 0.07                     # the threshold datum (<=0.07x)

# e286's committed record (the misbind parent — G_NAMEWIN's provenance)
E286_METRICS = E43.REPO / "runs" / "e286" / "metrics.json"
E286_MD5 = "9c5a3cefc806879cb6b6025c81c6a92c"
E286_VERDICT = "MAINTENANCE-FAILS"
E286_REA_POST = 0.004766650963574648
E286_REA_RATIO = 0.01801131154082594
E286_S_CORPUS = 1.8678520986577496
E286_S_MAINT = 4.106666415929794
E286_S_TOTAL_USAGE = 1.301802055342049
E286_MAINT_BUDGET_SHARE = 0.8948113170640402  # S_maint alone / budget
E286_WIN1_LIFT = 2.2028232105442878
E286_WIN2_LIFT = 0.37694117340634276
E286_MAINT_INROOM_FIRST = 0.059743382851433
E286_MAINT_INROOM_LAST = 0.060004314619887644

# e287's committed record (the third form — the calibration trace's home)
E287_METRICS = E43.REPO / "runs" / "e287" / "metrics.json"
E287_MD5 = "b7d18b8b286352732087b2ea99af3d2a"
E287_VERDICT = "SAWTOOTH-CONFIRMED"
E287_REA_POST = 0.024276327341794968         # the third form's endpoint read
E287_REA_RATIO = 0.09173075565244133         # THE BAR BOUNDARY ("e287's 0.0917")
E287_TRAJ_G0 = {100: 0.011196622624993324, 200: 0.02056380733847618,
                300: 0.03232550993561745, 400: 0.024276327341794968}
E287_S_CORPUS = 4.412244869628921            # the 96.1/3.9 split (the dial)
E287_S_MAINT = 0.17717676144093275
E287_S_SPLIT_CORPUS = 0.9613945338468215
E287_S_SPLIT_MAINT = 0.03860546615317855
E287_TWN_POST = 0.0004871897690463811        # e287's same-session sanctuary twin
E287_WIN1_LIFT = 2.8191793888537053
E287_WIN2_LIFT = 3.9155105780674777
E287_MAINT_INROOM_MEDIAN = 0.06095682399802789   # T267: AT chance (~0.06)
E287_MAINT_LR_MEDIAN = 0.0020782263794567883
E287_CORPUS_CE_EARLY = 0.9261485934257507    # the stream-live record
E287_CORPUS_CE_LATE = 0.8192217350006104

# e288's committed record (THIS cell's DIRECT PARENT — the twin's form +
# THE FOUNDING SUCCESS: the controller's curve is the signal-borne
# reference this cell's twin replicates in-session)
E288_METRICS = E43.REPO / "runs" / "e288" / "metrics.json"
E288_MD5 = "31bf8df55b51c8a19c48a155388050be"
E288_VERDICT = "ERROR-GATED-HOLDS"
E288_REA_POST = 0.9207638502120972           # the founding success's read
E288_REA_RATIO = 3.4792068243367953          # x3.48 — the 3.5x overshoot
E288_TRAJ_G0 = {100: 0.7112284302711487, 200: 0.9221979975700378,
                300: 0.9006620645523071, 400: 0.9207638502120972}
E288_S_CORPUS = 2.7536529786884785           # the realized 60/40 split
E288_S_MAINT = 1.2461527194827795
E288_S_TOTAL_USAGE = 0.8715271809995638      # 87.2% of budget
E288_TWN_POST = 0.04296277463436127          # its flat-dose twin (x0.1623)
E288_TWN_RATIO = 0.16233953870569715
E288_WIN1_LIFT = 7.490785666221227
E288_WIN2_LIFT = 22.013619863872503
E288_MAINT_INROOM_MEDIAN = 0.06026178670986923   # the brush at chance
E288_LR_M_MEDIAN = 0.0189421246658655
E288_DEFICIT_TRACE = (
    0.964767267990533, 0.9638790707448035, 0.9823719276954169,
    0.982105892221834, 0.9654872168683667, 0.9334813097846523,
    0.8769808755553569, 0.8417059107773661, 0.8283771631651486,
    0.7779996393059043, 0.7411397685679476, 0.7004941441544414,
    0.6600448294450705, 0.6695609374262748, 0.6534653434685034,
    0.6580809159715638,
)
E288_CE_EARLY = 0.9369652271270752
E288_CE_LATE = 0.8805128335952759

# e313's committed record (the brush data the dispatch cites: the
# controller's stroke OUT-of-room — the graft's IN-room stroke the clean
# contrast; the bearer axis formation-only)
E313_METRICS = E43.REPO / "runs" / "e313" / "metrics.json"
E313_MD5 = "4894788c7f3a56082ce7cb41acd64b3f"
E313_VERDICT_A = "THIN-HELD-UNCHANGED"
E313_VERDICT_B = "SEQ-LEAKY"
E313_BRUSH_BAND = (0.055, 0.065)             # the maintenance gradient's
                                              # in-room fraction class band

E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268's G_PARENTS bind

# the frozen bars' numbers ------------------------------------------------
SURVIVE_FRAC = 0.5               # GRAFT-HOLDS / twin-holds: >= 0.5x at t400
STREAM_STABLE_TOL = 1.05         # "stable": late CE <= 1.05 x early (frozen)
ORTH_BAR = 1e-6                  # G_ORTH: the missile's orthogonality gate
BUFSEP_ORTH_BAR = 1e-4           # G_BUFSEP: ||P_room buf_C||/||buf_C|| below
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
RATIO_DEN_FLOOR = 1e-6           # the ladder's floor-guard convention
VOCAB_EXPECT = 65
# MIXED-partial engagement floors (frozen): the graft SPENT >= 1% of B_G AND
# the read sits >= 10x e285's orthogonal-passive death class
GRAFT_ENGAGE_S_FLOOR = 0.01      # x B_G
GRAFT_SLOW_FLOOR_X = 10.0        # x E285_SAN_RATIO -> 0.0303x

REGISTERED = {
    "law_verbatim": "the controller works by error-gated re-teaching (a "
        "signal). The graft asks: can a fact be maintained by REFRESHING "
        "ITS GEOMETRY ALONE — periodically transplanting the write's "
        "in-room projection from a frozen external copy (theta <- theta + "
        "alpha*(P_room w_frozen - P_room theta), under the same budget "
        "language), with NO gradient, NO loss, NO signal? If the graft "
        "holds: memory maintenance reduces to consolidation-from-checkpoint "
        "— the anesthesia metaphor, the cheapest endgame. If it fails where "
        "the controller holds: the signal is load-bearing and maintenance "
        "is genuinely re-teaching.",
    "arms_verbatim": ARM_DESC,
    "bars_verbatim": {
        "GRAFT-HOLDS": ">= 0.5x with the budget held and the stream live — "
            "maintenance needs NO signal: memory preservation is geometry "
            "refresh; the anesthesia endgame.",
        "SIGNAL-NEEDED": "the graft < 0.5x while the controller twin holds "
            "— the teaching signal is load-bearing; maintenance is "
            "genuinely re-teaching (and the era's economics deepen).",
        "MIXED": "partial (the graft slows the decay but does not stop it "
            "— geometry buys time, signal buys survival; a two-channel "
            "account of maintenance).",
    },
    "reads_verbatim": "post g0 at t100/200/300/400 (both arms); the graft "
        "trace (alpha, displacement spent, the read's response per graft); "
        "the budget; the corpus CE.",
    "operationalizations": (
        "frozen BEFORE compute: THE PHASE := 400 corpus steps + 16 actuator "
        "steps interleaved on BOTH arms; milestones t=100/200/300/400 read "
        "POST-actuator (t400 includes the 16th step); THE VEHICLE := the "
        "committed quiet-formed 10k fact loaded BIT-EXACT (three-way "
        "G_FACTLOAD); the room := the fact's OWN K10K room (bit-gated vs "
        "e264_rooms.pt); THE CORPUS STEP := e268's registered form VERBATIM "
        "on THIS cell's ONE fresh registered stream seed 29601 (BOTH arms "
        "draw the identical sequence — the actuator is the arms' ONLY "
        f"delta); THE GRAFT := at every M={ACT_EVERY} corpus steps ONE "
        "DIRECT PARAMETER EDIT theta <- theta + alpha_t * P_room(fact - "
        "theta) (== alpha*(P_room w_frozen - P_room theta) in displacement "
        "language; the frozen external copy := the LOADED fact itself, "
        "never updated — an external checkpoint, not a state of the net); "
        "the dose gated THE SAME WAY as the controller: deficit_t = max(0, "
        f"{FACT_BASELINE_G0} - read)/{FACT_BASELINE_G0} on the SAME g0 "
        "battery the bars read (at t25/t200 cross-checked against the "
        "window pre-read); alpha_t = min(ALPHA_MAX x deficit_t, cap_g) "
        f"with ALPHA_MAX = {ALPHA_MAX!r} (full deficit -> the FULL "
        "transplant — the anesthesia ideal; no momentum ramp exists to "
        "calibrate against, NO optimizer) and cap_g = share_g/||gap||, "
        "share_g = (B_G - S_graf)/rem_graf the symmetric equal-share "
        "reservation over the graft stream's OWN pool B_G = 40% of budget "
        "(the corpus stream's B_C = 60%, e288's split, BOTH arms); "
        "S_graf := the realized ||edit|| each event; THE TWIN := e288's "
        "ERROR-GATED arm VERBATIM on this session's shared stream 29601 "
        "(the name-only CE, install stream 29602, LR_M_MAX re-derived + "
        "asserted, the B_M share cap); 'survival at t400' := post-actuator "
        "g0 / the committed baseline (PRIMARY; the session denominator "
        "co-reported); 'the budget <= 100%' := S_total = S_corpus + S_act "
        "<= BUDGET + 1e-5 AND the realized cumulative displacement <= "
        "BUDGET + slack, BOTH arms, per-pool (non-halting — a break routes "
        "the composite to MIXED, everything verbatim); 'the stream live' "
        ":= per arm, median corpus-batch CE rows t in (300,400] vs "
        "[1,100]: improving (late < early) OR stable (late <= 1.05 x "
        "early); 'the controller twin holds' := twin ratio_400 >= 0.5 with "
        "its budget held and its stream live; MIXED-partial := graft "
        "ratio_400 < 0.5 AND the twin holds AND (i) S_graf >= 0.01 x B_G "
        "AND (ii) graft ratio_400 >= 10 x e285's sanctuary ratio "
        f"{E285_SAN_RATIO!r} (the engagement + slowed-decay floors); "
        "COMPOSITE := TEXTURE (bind/isolation failure) -> MIXED-by-"
        "conditions (G_BUDGET blown OR G_ORTH failed, any ratio) -> "
        "GRAFT-HOLDS (graft ratio_400 >= 0.5 AND budget held AND the "
        "graft arm's stream live) -> [twin holds] MIXED-partial (both "
        "engagement floors) / SIGNAL-NEEDED (else) -> MIXED (the twin "
        "itself failed to hold); THE TRANSPLANT ARITHMETIC := machine-live "
        "at EVERY graft event (gap_new == (1-alpha) x gap_old within "
        "fp32-write tolerances abs 1e-4 / rel 1e-3; ||edit|| == alpha x "
        "gap rel 1e-3; alpha <= ALPHA_MAX always; alpha == 0 at deficit 0) "
        "+ the fp64-exact unit test G_TRANSPLANT at startup; G_GRAFISOL "
        "(HARD) := the graft edit touches NO optimizer state (opt_C "
        "buffers bitwise around every event) and NO gradient exists on "
        "the path (the grads bitwise-unchanged through every edit); "
        "HARD GATES (HALT) := "
        "{G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, "
        "G_NAMEWIN, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, "
        "G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND, "
        "G_MAINTBIND, G_TRANSPLANT, G_GRAFTBIND, G_GRAFISOL, G_BUFSEP-"
        "isolation (the twin)}; NON-HALTING (route to MIXED) := {G_ORTH, "
        "G_BUDGET, bufC-composition}; NO CONS (T259/e281; both post states "
        "checkpointed)."),
    "registration": "question + bars + arms + predictions frozen VERBATIM "
        "from the dispatch letter (the wild queue's #1: THE PROSTHETIC "
        "GRAFT — maintenance with NO teaching signal); this script "
        "committed at birth BEFORE any compute; adjudicate against exactly "
        "this; no bar shopping.",
    "predictions": {
        "P-e296a_anesthesia": "GRAFT-HOLDS — the periodic in-room refresh "
            "alone preserves the read under traffic; maintenance reduces "
            "to consolidation-from-checkpoint (the anesthesia endgame).",
        "P-e296b_signal": "SIGNAL-NEEDED — the graft < 0.5x while the "
            "twin holds (the founding success replicates in-session); the "
            "teaching signal is load-bearing; maintenance is genuinely "
            "re-teaching.",
        "P-e296c_engagement_bound": "under the era's ORTHOGONALIZED corpus "
            "stream the in-room gap ||P_room(fact - theta)|| is pinned at "
            "the fp floor (e285's committed sanctuary law: drift 1.8445, "
            "in-room 1.23e-6), so the graft CANNOT ENGAGE: expected "
            "SIGNAL-NEEDED with S_graf ~ 0, the graft curve riding the "
            "orthogonal-passive death class, and the mechanism named: the "
            "read's death is OUT-of-room, the anesthesia endgame "
            "structurally unavailable in this rig, only the signal's "
            "out-of-room brush reaches the wound. DISCRIMINATOR: the graft "
            "trace's per-event gap_norm column (>= 1e-4 anywhere = the "
            "graft engages; fp-floor everywhere = the structural reading "
            "confirmed). Registered to keep the null honest, not to move "
            "the bars.",
    },
}

deviations: list[str] = [
    "THE E285 DRIFT LITERAL (disclosed, discovered at the parent bind): "
    "runs/e285/metrics.json's drift_from_fact_final_norm now reads "
    "1.8445091953053947 while e288's committed bind read "
    "1.844509195284748 — the e285 record was rewritten after e288's fold "
    "(the 2e-8 fp discrepancy of a re-computed ledger). THIS cell binds "
    "the CURRENT committed record (md5 3f71c7b209fe9b6e9645d7504ddbb49c, "
    "verified at runtime) and its current literal; nothing scientific "
    "hangs on the 8th digit (the sanctuary's story: drift ~1.84, in-room "
    "~1.2e-6, read x0.0030 — unchanged).",
    "THE GRAFT (the ONE new mechanism, frozen at birth): the maintenance "
    "actuator REPLACED by the frozen-copy transplant — a DIRECT PARAMETER "
    "EDIT theta <- theta + alpha_t * P_room(fact - theta) through NO "
    "optimizer, with NO gradient, NO loss, NO install draws (the graft is "
    "deterministic geometry; the twin keeps the install stream 29602). In "
    "displacement language the edit IS alpha*(P_room w_frozen - P_room "
    "theta): P_room is linear and the base cancels — the frozen external "
    "copy is the LOADED fact (the committed write w = fact - base), held "
    "fixed, never updated. The dose gated THE SAME WAY as the controller "
    "(deficit on the same g0 battery; the read is the ONLY thing measured) "
    "and under the SAME budget language (the equal-share cap over B_G = "
    "40%, S_graf the realized edit norm). ALPHA_MAX = 1.0 (full deficit -> "
    "the FULL transplant): with no optimizer there is no momentum ramp to "
    "calibrate a ceiling against (e287's b_m trace inapplicable), so the "
    "cap alone does the budget-fitting — disclosed.",
    "THE STRUCTURAL ENGAGEMENT BOUND (P-e296c, registered BEFORE compute — "
    "the honest null): the corpus stream is e288's ORTHOGONALIZED stream "
    "(the era's rig; the twin runs it too, so the arms' only delta stays "
    "the actuator), and an orthogonalized stream cannot move the "
    "room-frame: the in-room gap is bounded at the fp floor (e285's "
    "committed sanctuary law). The graft's displacement is therefore "
    "bounded by ~alpha x fp-floor — the graft may be structurally UNABLE "
    "to spend its pool. The bars adjudicate VERBATIM regardless; the graft "
    "trace's gap_norm column is the registered discriminator between "
    "'geometry cannot maintain' (engaged and failed) and 'geometry was "
    "never the thing that broke' (never engaged — the room-frame not the "
    "operative frame, T267's law at the actuator).",
    "THE CONTROLLER-TWIN IS e288's ERROR-GATED ARM VERBATIM (the dispatch: "
    "'the signal-borne reference'), NOT e287's flat form: the name-only "
    "CE at lr_m_t = LR_M_MAX x deficit_t over the split pools B_C/B_M = "
    "60/40, on THIS session's shared stream 29601 + install 29602 — the "
    "family's own-draw-stream convention: it replicates e288's FORM not "
    "its bits; its expected landing is the HOLDS class (e288 committed "
    "x3.4792, hard-bound in G_PARENTS). The arms' ONLY delta: the actuator "
    "(geometry transplant vs signal gradient).",
    "THE BUDGET'S ACCOUNTING (e288's, carried): G_BUDGET (and G_ORTH) are "
    "computed, recorded, and NON-halting — a failure routes the composite "
    "to MIXED with every ledger intact; the bind gates and the isolation "
    "probes remain HARD HALTs. BOTH arms use the split pools (B_C = 60% / "
    "B_act = 40%): the graft arm's B_G and the twin's B_M are the SAME "
    "allocation given to different actuators — the budget language "
    "identical, the actuator the only difference.",
    "G_GRAFISOL (the graft arm's no-signal isolation, HARD): the graft "
    "edit touches NO optimizer state (opt_C's momentum buffers snapshotted "
    "bitwise around EVERY graft event; any mismatch HALTs) and NO gradient "
    "is ever computed on the graft path (no loss, no backward — "
    "structural; machine-checked: the grads BITWISE-UNCHANGED through "
    "every edit — the SMOKE caught the naive None-check: the corpus "
    "step's grads persist after opt_C.step() until the next iteration's "
    "zero_grad, so None was the wrong invariant; amended at smoke, "
    "disclosed). "
    "The twin keeps e288's bidirectional G_BUFSEP machinery VERBATIM "
    "(opt_C/opt_F). The corpus buffer's composition bar (< 1e-4 in-room) "
    "carried on BOTH arms.",
    "NO CONS (T259/e281; the family's committed form): the frozen bars "
    "read the WRITE and the DISPLACEMENT only; both arms' post-phase "
    "states are checkpointed (e296_GRAFT_post.pt / "
    "e296_CONTROLLER-TWIN_post.pt) for any later landing pass.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT (the family's convention, "
    "carried): the SRCT projector + LadderRooms (the room rebuild, "
    "certification, displacement loads), the thermal envelope (per-step "
    "polls, 78C margin, 175s bursts inside the dispatch's 180s, 40s "
    "cooldowns, the 84C line inside the dispatch's 85C), the "
    "progressive-metrics + resume-ckpt conventions — the module-global "
    "rebinding (log/NAME/LADDER/RUNG_NAMES/T0/thermal ledgers, disclosed "
    "in-code) retargets the machinery's I/O to this cell (envelope tags "
    "e296:<ARM>:<phase> per the dispatch); the committed "
    "lab/e261_rank_ladder.py is NOT modified. The TWO drivers are THIS "
    "file's: chunked_graf_phase (the graft — the corpus driver with the "
    "actuator block replaced by the transplant) and "
    "chunked_controller_twin_phase (e288's chunked_errorgated_phase "
    "VERBATIM in body — tags and checkpoint names adapted).",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the measured "
    "displacement loads; no new history is run.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat — "
    "the critic's lottery note carried verbatim); the arms' DIFFERENCE is "
    "the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E296_SMOKE=1): 8 corpus steps per arm, milestones at "
    "every step 1..8, room k=512 at the same seed pair (G_ROOMK10K "
    "vacuous — no committed record at smoke k; disclosed), the REAL "
    "committed fact loaded and read (G_FACTLOAD live), G_NAMEWIN live (the "
    "decode gate), G_CORPUSGEN live, G_GRAFTBIND live at the smoke cadence "
    "(M=2 -> exactly 4 graft steps — the graft's full path exercised: the "
    "gate read, the deficit arithmetic, the gap, the cap, THE TRANSPLANT "
    "ARITHMETIC asserts live), G_TRANSPLANT live (the fp64 unit test), "
    "G_GRAFISOL live, the twin's G_MAINTBIND live at M=2 (4 error-gated "
    "steps), the smoke budgets SCALED per-stream to the full run's "
    "per-event shares (B_C x 8/400, B_G/B_M x 4/16; LR_M_MAX unscaled — "
    "the full-run constant, disclosed), windows at t1/t2/t3 and t3/t4/t5, "
    "G_ORTH live (every step), G_BUDGET live (total + split + "
    "per-allocation); the adjudication form + figure exercised; all paths "
    "smoke_-prefixed, own smoke dir; NOTHING adjudicated or gated for the "
    "record (SMOKE stamp on every read).",
]


# ------------------------------------------------------------------ envelope
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


# --------------------------------------------- THE MISSILE'S PROJECTION (port)
def orthogonalize_grads(proj: "E261.LadderRooms", params, mode: str) -> dict:
    """e278/e288's orthogonalize_grads VERBATIM: replace the (clipped)
    corpus gradient g by g_perp = g - P_room(g) — the component ENTIRELY
    ORTHOGONAL to the room (CPU fp64, write fp32, norm NOT rescaled). The
    verification read ||P_room g_perp|| / ||g_perp|| is returned for the
    gate."""
    room = proj.rooms[mode]
    params = list(params)
    g = torch.cat([p.grad.detach().reshape(-1) for p in params]) \
        .to(CPU).double().numpy().astype(np.float64)
    gn2 = float(g @ g)
    gp_in = room.project(g)                 # P_room(g) — the in-room part
    gperp = g - gp_in                       # the orthogonal complement
    gpn2 = float(gperp @ gperp)
    resid = room.project(gperp)             # must be ~0 (the verification)
    rel = (float(np.sqrt(resid @ resid)) / float(np.sqrt(gpn2))
           if gpn2 > 0 else 0.0)
    gp32 = torch.from_numpy(gperp.astype(np.float32))
    with torch.no_grad():
        for p, (a, b), shp in zip(params, proj.offsets, proj.shapes):
            p.grad.copy_(gp32[a:b].to(proj.dev).reshape(shp))
    return {"gn": math.sqrt(gn2), "gperp_norm": math.sqrt(gpn2),
            "norm_ratio": (math.sqrt(gpn2 / gn2) if gn2 > 0 else 0.0),
            "in_room_frac": (math.sqrt(max(0.0, 1.0 - gpn2 / gn2))
                             if gn2 > 0 else 0.0),
            "orth_rel_err": rel}


# --------------------------------------------- THE BUFFER MACHINES (port)
def snap_buffers(opt, params) -> list:
    """Snapshot an optimizer's per-param momentum buffers (None where the
    state has none yet) — the isolation probes' eyes."""
    st = opt.state
    return [st[p].get("momentum_buffer").detach().clone()
            if isinstance(st.get(p), dict)
            and "momentum_buffer" in st[p] else None
            for p in params]


def buffers_bitwise_equal(sn_a: list, sn_b: list) -> bool:
    if len(sn_a) != len(sn_b):
        return False
    for a, b in zip(sn_a, sn_b):
        if (a is None) != (b is None):
            return False
        if a is not None and not torch.equal(a, b):
            return False
    return True


def buffer_inroom_frac(opt, params, room) -> tuple:
    """(||P_room buf|| / ||buf||, ||buf||) of an optimizer's momentum
    buffer state (CPU fp64 projection)."""
    parts = []
    st = opt.state
    for p in params:
        d = st.get(p)
        if isinstance(d, dict) and "momentum_buffer" in d:
            parts.append(d["momentum_buffer"].detach().reshape(-1).cpu())
    if not parts:
        return None, 0.0
    b = torch.cat(parts).double().numpy().astype(np.float64)
    bn = float(np.linalg.norm(b))
    if bn == 0.0:
        return None, 0.0
    return float(np.linalg.norm(room.project(b)) / bn), bn


def pending_buffer_sqnorm(opt_C, params, momentum: float) -> float:
    """||b_t||^2 for PyTorch SGD-M's exact in-step update b_t = mu * buf_{t-1}
    + g (dampening 0): computed on-GPU fp32 from the optimizer's stored
    buffers + the (orthogonalized) grads now in p.grad — the budget cap's
    per-step price of the applied displacement (theta -= lr_t * b_t)."""
    sq = torch.zeros((), device=next(iter(params)).device)
    st = opt_C.state
    for p in params:
        d = st.get(p)
        buf_prev = d.get("momentum_buffer") if isinstance(d, dict) else None
        if buf_prev is None:
            v = p.grad.detach()
        else:
            v = momentum * buf_prev + p.grad.detach()
        sq = sq + v.square().sum()
    return float(sq.item())


# ====================================================== THE GRAFT (the core)
def graft_gap(room, fact_np: np.ndarray, theta_np: np.ndarray) -> np.ndarray:
    """THE TRANSPLANT VECTOR: P_room(fact - theta) — in displacement
    language exactly (P_room w_frozen - P_room theta): P_room linear, the
    base cancels. fp64 CPU."""
    return room.project(fact_np - theta_np)


def apply_graft_edit(net, proj, edit64: np.ndarray, dev) -> None:
    """THE DIRECT PARAMETER EDIT: theta += edit (fp32 cast, per-parameter
    slicing through the room machinery's offsets/shapes — the family's
    write convention). NO optimizer, NO gradient."""
    e32 = torch.from_numpy(edit64.astype(np.float32))
    with torch.no_grad():
        for p, (a, b), shp in zip(net.parameters(), proj.offsets,
                                  proj.shapes):
            p.data.add_(e32[a:b].to(dev).reshape(shp))


def transplant_unit_test(room, n: int) -> dict:
    """G_TRANSPLANT — the dispatch's smoke demand, fp64-EXACT (run at
    startup in BOTH smoke and full, before any arm): one graft step must
    move P_room theta toward P_room w_frozen by EXACTLY alpha times the
    gap. Synthetic: theta0 := fact - v with v a KNOWN in-room vector (the
    gap); one step at alpha; the checks: (i) the new gap == (1-alpha) x
    old (to 1e-12), (ii) the edit == alpha x v EXACTLY (the out-of-room
    component untouched, to 1e-12), (iii) alpha=1 closes the gap to the
    fp floor, (iv) alpha=0 is the identity."""
    rng = np.random.default_rng(29611)
    fact = rng.standard_normal(n)
    # a known IN-ROOM gap vector: project a random vector into the room
    v = room.project(rng.standard_normal(n))
    vn = float(np.linalg.norm(v))
    theta0 = fact - v
    out = {"n": n, "gap_norm": vn, "checks": {}}
    for alpha in (0.37, 1.0, 0.0):
        gap_pre = room.project(fact - theta0)
        gn_pre = float(np.linalg.norm(gap_pre))
        theta1 = theta0 + alpha * graft_gap(room, fact, theta0)
        gap_post = room.project(fact - theta1)
        gn_post = float(np.linalg.norm(gap_post))
        edit = theta1 - theta0
        en = float(np.linalg.norm(edit))
        # (i) the gap contracted by exactly (1-alpha)
        c1 = abs(gn_post - (1.0 - alpha) * gn_pre)
        # (ii) the edit == alpha x v exactly (in-room, nothing else moved)
        c2 = float(np.linalg.norm(edit - alpha * v))
        # the out-of-room component untouched: edit's in-room fraction 1
        c3 = abs(float(np.linalg.norm(room.project(edit)) / max(en, 1e-300))
                 - 1.0) if en > 0 else 0.0
        out["checks"][f"alpha={alpha}"] = {
            "gap_pre": gn_pre, "gap_post": gn_post,
            "gap_contraction_err": c1, "edit_vs_alpha_v_err": c2,
            "edit_inroom_frac_dev": c3, "edit_norm": en}
    ok = all(ch["gap_contraction_err"] < 1e-12
             and ch["edit_vs_alpha_v_err"] < 1e-12
             and ch["edit_inroom_frac_dev"] < 1e-9
             for ch in out["checks"].values())
    out["pass"] = bool(ok and out["checks"]["alpha=1.0"]["gap_post"] < 1e-12
                       and abs(out["checks"]["alpha=0.0"]["gap_post"]
                               - out["checks"]["alpha=0.0"]["gap_pre"])
                       < 1e-15)
    out["form"] = ("the transplant arithmetic, fp64-exact on a synthetic "
                   "in-room gap: theta <- theta + alpha*P_room(fact - "
                   "theta) contracts the in-room gap by exactly (1-alpha), "
                   "moves NOTHING out-of-room, alpha=1 closes the gap to "
                   "the fp floor, alpha=0 is the identity — the dispatch's "
                   "smoke demand, machine-verified")
    return out


# ======================================================================
# THE GRAFT DRIVER — the prosthetic arm: e288's corpus rig with the
# actuator REPLACED by the direct parameter edit (NO optimizer, NO
# gradient, NO loss, NO install draws)
# ======================================================================
def chunked_graf_phase(tag: str, net0, proj: "E261.LadderRooms",
                       fact_flat_np: np.ndarray,
                       base_flat_np: np.ndarray,
                       budget_c: float, budget_g: float,
                       anchor_full, train_ids, g0_ids, gm12_ids,
                       r_eval_xy, zid, resume_ck: Path,
                       dev: torch.device) -> dict:
    """THE GRAFT ARM'S DRIVER. The net starts at the LOADED formed fact;
    per corpus step t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 29601 — bit-identical to
      the twin's draws); corpus batch = 16 original-host anchors + 32
      random corpus windows; full-window CE; lr_sched = LR_STABLE x
      cosine_lr(t-1, 1000); backward -> clip 1.0 -> the orthogonalized
      stream (g_perp, verified EVERY step) -> the per-step lr cap
      (equal-share reservation over the CORPUS stream's OWN remaining
      pool B_C — e288's cap form verbatim) -> opt_C.step() (the ONLY
      optimizer on this arm; its buffers snapshotted bitwise around every
      graft event too — the edit must not touch them).

      AND THE GRAFT INJECTION (the cell's build): after every M=25 corpus
      steps, ONE GRAFT STEP — the frozen-copy transplant at the
      deficit-gated alpha: the read's current deficit measured on the g0
      battery (the SAME battery the bars read; at t25/t200 cross-checked
      against the window pre-read); the in-room gap g_in = P_room(fact -
      theta) (CPU fp64); alpha = min(ALPHA_MAX x deficit, cap_g) with
      cap_g = share_g/||gap|| the equal-share reservation over B_G; THE
      EDIT theta += alpha x g_in (fp32 write, NO optimizer); the
      transplant arithmetic verified live (the gap contracted by exactly
      (1-alpha); the edit's norm == alpha x gap); the realized edit norm
      enters S_graf (the budget's total) and the graft ledger (the full
      per-event trace: gate read, deficit, gap, alpha, cap, binder, edit
      norm, the post-graft read response via the window rows).

    Ledgers: the family's full set (orth / corpus / disp / budget / lr) +
    the graft ledger (every graft step, the geometry trace) + the window
    ledger. Thermal: a poll after EVERY optimizer event AND every graft
    event."""
    budget_norm = budget_c + budget_g       # display/ledger denominator
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    n_steps = PHASE_STEPS
    M = ACT_EVERY
    n_anc = anchor_full.shape[0]
    N = int(fact_flat_np.size)
    # the window reads' map: plain rows at the non-actuator flank steps
    # (t24/t26, t199/t201) + PRE/POST rows around the actuator step itself
    win_mid = {WIN1[0]: f"win1:t{WIN1[0]}", WIN1[2]: f"win1:t{WIN1[2]}",
               WIN2[0]: f"win2:t{WIN2[0]}", WIN2[2]: f"win2:t{WIN2[2]}"}
    win_act = {WIN1[1]: "win1", WIN2[1]: "win2"}
    state = {"step": 0, "traj": [], "corpus_ledger": {}, "orth_ledger": {},
             "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
             "lr_ledger": {}, "graf_ledger": [], "window_ledger": [],
             "graisol": {"corpus_checks": 0, "corpus_violations": 0,
                         "graf_checks": 0, "graf_violations": 0,
                         "graf_grad_none_violations": 0, "n_graf": 0},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "graf_cum": torch.zeros(N, dtype=torch.float64),
             "tot_prev": torch.zeros(N, dtype=torch.float64),
             "S_corpus": 0.0, "S_graf": 0.0, "n_capped": 0,
             "n_graf": 0, "n_graf_capped": 0, "orth_max": 0.0}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at corpus step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        lrs_c = [v["lr_applied"] for v in state.get("lr_ledger",
                                                    {}).values()]
        return {"sd": state.get("model"), "traj": state.get("traj", []),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "buf_ledger": state.get("buf_ledger", []),
                "budget_ledger": state.get("budget_ledger", []),
                "lr_ledger": state.get("lr_ledger", {}),
                "graf_ledger": state.get("graf_ledger", []),
                "window_ledger": state.get("window_ledger", []),
                "graisol": state.get("graisol", {}),
                "S_corpus": state.get("S_corpus"),
                "S_graf": state.get("S_graf"),
                "n_capped": state.get("n_capped"),
                "n_graf": state.get("n_graf"),
                "n_graf_capped": state.get("n_graf_capped", 0),
                "orth_max": state.get("orth_max"),
                "lr_applied_min": min(lrs_c) if lrs_c else None,
                "lr_applied_median": (float(sorted(lrs_c)[len(lrs_c) // 2])
                                      if lrs_c else None),
                "lr_sched_median": None, "cap_min": None,
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt_C = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[ROOM_MODE]
    corp_cum = state["corp_cum"].to(dev)
    graf_cum = state["graf_cum"].to(dev)
    tot_prev = state["tot_prev"].to(dev)
    S_corpus = float(state["S_corpus"])
    S_graf = float(state["S_graf"])
    S_total = S_corpus + S_graf
    n_capped = int(state["n_capped"])
    n_graf = int(state["n_graf"])
    n_graf_capped = int(state.get("n_graf_capped", 0))
    orth_max = 0.0

    def _read_window(phase_label: str, at_step: int) -> None:
        """the between-steps read (the response's eyes): battery g0 on the
        CPU eval net + the drift + the in-room gap + the S split."""
        sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        bz0 = G1.battery_cell(evl, g0_ids, zid)
        theta_t = flat_params_cpu(net).double().numpy().astype(np.float64)
        dn = float(np.linalg.norm(theta_t - fact_flat_np))
        gapn = float(np.linalg.norm(
            graft_gap(room, fact_flat_np, theta_t)))
        state["window_ledger"].append({
            "window_phase": phase_label, "step": at_step,
            "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
            "survival_ratio_vs_committed": bz0["mean_pz"] / FACT_BASELINE_G0,
            "drift_from_fact_norm": dn, "in_room_gap_norm": gapn,
            "n_graf_so_far": n_graf,
            "S_corpus": S_corpus, "S_graf": S_graf,
            "elapsed_s": round(time.time() - T0, 1)})
        log(f"  [{tag}] WINDOW {phase_label} (after t{at_step}): g0 "
            f"{bz0['mean_pz']:.6f} "
            f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) |d| {dn:.3f} "
            f"in-room-gap {gapn:.3e} | S_corp {S_corpus:.4f} S_graf "
            f"{S_graf:.6f} (graf #{n_graf})")

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{GRF_ARM}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            # ONE optimizer ONLY: the corpus side's SGD-M. There is NO
            # fact-side optimizer on this arm — the graft is a direct
            # parameter edit (the no-signal arm).
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt_C.load_state_dict(state["optC"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
                S_corpus = float(state["S_corpus"])
                S_graf = float(state["S_graf"])
                S_total = S_corpus + S_graf
                n_capped = int(state["n_capped"])
                n_graf = int(state["n_graf"])
                n_graf_capped = int(state.get("n_graf_capped", 0))
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            lr_sched = LR_STABLE * cosine_lr(step - 1, E261.INST_TOTAL)
            # ---- THE CORPUS STEP (the family's standard, bit-identical
            # draws to the twin)
            aj_c = torch.randint(n_anc, (corp_bs - mix_random,),
                                 generator=cgen)
            rj_c = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                                 generator=cgen)
            corp_c = torch.cat([anchor_full[aj_c],
                                torch.stack([train_ids[s: s + G1.BLOCK]
                                             for s in rj_c])], 0)
            xc = corp_c[:, :-1].to(dev)
            yc = corp_c[:, 1:].to(dev)
            logits_c, _ = net(xc)
            nll_c = F.cross_entropy(
                logits_c.reshape(-1, logits_c.shape[-1]),
                yc.reshape(-1), reduction="none").view(xc.shape[0],
                                                       xc.shape[1])
            loss_c = nll_c.mean()
            net.zero_grad(set_to_none=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gn_c = float(torch.cat([p.grad.detach().reshape(-1)
                                    for p in net.parameters()
                                    if p.grad is not None]).norm().item())
            # the RAW gradient's in-room fraction (the rig identity read)
            g_in_room = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                pg = room.project(g64)
                g_in_room = float(np.linalg.norm(pg)
                                  / max(np.linalg.norm(g64), 1e-30))
            # ---- the orthogonalized stream (verified EVERY step)
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- the per-step lr cap — equal-share reservation over the
            # CORPUS stream's OWN remaining pool B_C (e288's cap form)
            b_sq = pending_buffer_sqnorm(opt_C, params_live, SGD_MOMENTUM)
            b_norm = math.sqrt(max(b_sq, 0.0))
            rem_corpus = n_steps - step + 1
            remaining_c = max(budget_c - S_corpus, 0.0)
            share = remaining_c / rem_corpus
            cap = share / max(b_norm, 1e-12)
            lr_t = min(lr_sched, cap)
            capped = bool(cap < lr_sched)
            if capped:
                n_capped += 1
            for g_ in opt_C.param_groups:
                g_["lr"] = lr_t
            # ---- the isolated corpus step (NO other optimizer exists)
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()
            with torch.no_grad():
                d_step = torch.cat([p.detach().reshape(-1)
                                    for p in net.parameters()]) - theta_b
                corp_cum += d_step
            realized = float(d_step.norm().item())
            S_corpus += realized
            S_total = S_corpus + S_graf
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c,
                                                "g_in_room_frac": g_in_room}
                state["orth_ledger"][step] = {
                    "gn_clipped": gn_c,
                    "gperp_norm": orth_row["gperp_norm"],
                    "norm_ratio": orth_row["norm_ratio"],
                    "in_room_frac": orth_row["in_room_frac"],
                    "orth_rel_err": orth_row["orth_rel_err"]}
                state["lr_ledger"][step] = {
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped, "b_norm": b_norm,
                    "share": share, "realized_step_norm": realized,
                    "rem_corpus": rem_corpus, "budget_c": budget_c,
                    "S_corpus": S_corpus}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(
                f"{GRF_ARM}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)

            # ---- the plain window rows (the flanks: t24/t26, t199/t201)
            if step in win_mid:
                _read_window(win_mid[step], step)

            # ---- the PRE-actuator window row (t25/t200) ----------------
            if step in win_act:
                _read_window(f"{win_act[step]}:t{step}-pre", step)

            # ---- THE GRAFT INJECTION (the cell's build: the frozen-copy
            # transplant — NO optimizer, NO gradient, NO loss) ------------
            if step % M == 0:
                n_graf += 1
                # (i) THE GATE (the same read the controller gates on): the
                # read's current deficit on the SAME g0 battery the bars
                # read (the CPU eval net at the event instant; at t25/t200
                # cross-checked against the window pre-read)
                sd_gate = {k: v.detach().cpu().clone()
                           for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_gate)
                evl.eval()
                read_gate = G1.battery_cell(evl, g0_ids, zid)["mean_pz"]
                deficit_t = min(1.0, max(
                    0.0, FACT_BASELINE_G0 - read_gate) / FACT_BASELINE_G0)
                gate_vs_pre = None
                if step in win_act and state["window_ledger"] \
                        and state["window_ledger"][-1]["window_phase"] \
                        == f"{win_act[step]}:t{step}-pre":
                    gate_vs_pre = abs(
                        read_gate - state["window_ledger"][-1]["g0_pz"])
                # (ii) THE GEOMETRY: the in-room gap to the frozen copy
                theta64 = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                g_in = graft_gap(room, fact_flat_np, theta64)
                gap_norm = float(np.linalg.norm(g_in))
                # (iii) THE DOSE: alpha gated THE SAME WAY as the
                # controller (deficit-proportional, capped at the graft
                # stream's OWN equal-share reservation of B_G)
                alpha_target = ALPHA_MAX * deficit_t
                assert alpha_target <= ALPHA_MAX * (1.0 + 1e-12), \
                    f"graft law broken: alpha_target {alpha_target} > " \
                    f"ALPHA_MAX {ALPHA_MAX}"
                if deficit_t <= 0.0:
                    assert alpha_target == 0.0
                rem_graf_now = 1 + sum(1 for m in range(step + 1,
                                                        n_steps + 1)
                                       if m % M == 0)
                remaining_g = max(budget_g - S_graf, 0.0)
                share_g = remaining_g / rem_graf_now
                cap_g = share_g / max(gap_norm, 1e-300)
                alpha = min(alpha_target, cap_g, ALPHA_MAX)
                assert alpha <= ALPHA_MAX * (1.0 + 1e-9), \
                    f"applied alpha {alpha} exceeds ALPHA_MAX {ALPHA_MAX}"
                capped_g = bool(cap_g < alpha_target)
                binder_g = "cap" if capped_g else "gate"
                if capped_g:
                    n_graf_capped += 1
                # (iv) THE NO-SIGNAL ISOLATION: the graft path computes NO
                # gradient (no loss, no backward — structural) and the
                # edit must touch NO optimizer state AND NO gradient: the
                # corpus step's grads persist after opt_C.step() (cleared
                # only at the next iteration's zero_grad), so the honest
                # machine check is BITWISE UNCHANGED across the edit
                # (the graft never reads, writes, or needs them)
                grads_before = [None if p.grad is None
                                else p.grad.detach().clone()
                                for p in params_live]
                sn_c_before = snap_buffers(opt_C, params_live)
                # (v) THE TRANSPLANT (the direct parameter edit)
                theta_b2 = torch.cat([p.detach().reshape(-1)
                                      for p in net.parameters()])
                apply_graft_edit(net, proj, alpha * g_in, dev)
                with torch.no_grad():
                    d_g = torch.cat([p.detach().reshape(-1)
                                     for p in net.parameters()]) - theta_b2
                    graf_cum += d_g
                realized_g = float(d_g.norm().item())
                S_graf += realized_g
                S_total = S_corpus + S_graf
                state["graisol"]["graf_checks"] += 1
                grads_unchanged = all(
                    (a is None and p.grad is None)
                    or (a is not None and p.grad is not None
                        and torch.equal(a, p.grad.detach()))
                    for a, p in zip(grads_before, params_live))
                if not grads_unchanged:
                    state["graisol"]["graf_grad_none_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_GRAFISOL VIOLATION at t{step}: the "
                        "graft edit changed a gradient — HALT")
                if not buffers_bitwise_equal(
                        sn_c_before, snap_buffers(opt_C, params_live)):
                    state["graisol"]["graf_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_GRAFISOL VIOLATION at t{step}'s graft "
                        "edit: it touched the corpus optimizer's state — "
                        "HALT")
                state["graisol"]["n_graf"] = n_graf
                # (vi) THE TRANSPLANT ARITHMETIC, LIVE (the dispatch's
                # smoke demand at EVERY event; the fp32-WRITE tolerances —
                # an edit below the fp32 rounding floor may round away
                # entirely, so the exact-arithmetic guarantee lives in the
                # fp64 unit test G_TRANSPLANT; these live checks verify to
                # write-noise scale): the gap contracted by exactly
                # (1-alpha); the edit's norm == alpha x gap
                theta64_post = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                gap_post = float(np.linalg.norm(
                    graft_gap(room, fact_flat_np, theta64_post)))
                tol_gap = max(1e-4, 1e-3 * gap_norm)
                assert abs(gap_post - (1.0 - alpha) * gap_norm) <= tol_gap, \
                    (f"graft arithmetic broken at t{step}: gap_post "
                     f"{gap_post!r} != (1-alpha)*gap "
                     f"{(1.0 - alpha) * gap_norm!r} (alpha {alpha!r})")
                assert abs(realized_g - alpha * gap_norm) <= max(
                    1e-4, 1e-3 * alpha * gap_norm), \
                    (f"graft edit norm {realized_g!r} != alpha*gap "
                     f"{alpha * gap_norm!r} at t{step}")
                state["graf_ledger"].append({
                    "step": step, "graf_index": n_graf,
                    "read_at_gate": read_gate, "deficit_t": deficit_t,
                    "gate_vs_pre_absdiff": gate_vs_pre,
                    "gap_norm_pre": gap_norm, "gap_norm_post": gap_post,
                    "gap_contraction_check": abs(
                        gap_post - (1.0 - alpha) * gap_norm),
                    "alpha_max": ALPHA_MAX, "alpha_target": alpha_target,
                    "cap_g": cap_g, "alpha": alpha, "capped_g": capped_g,
                    "binder": binder_g, "share_g": share_g,
                    "edit_norm": realized_g,
                    "S_graf": S_graf, "S_total": S_total,
                    "budget_g": budget_g,
                    "grads_bitwise_unchanged": grads_unchanged,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] GRAFT #{n_graf:2d} @t{step:3d}: GATE read "
                    f"{read_gate:.6f} deficit {deficit_t:.4f} -> alpha "
                    f"{alpha:.6f} {binder_g.upper()} (target "
                    f"{alpha_target:.6f} cap {cap_g:.3e}; gap {gap_norm:.3e}"
                    f" share {share_g:.5f}) -> edit {realized_g:.3e} | gap "
                    f"{gap_norm:.3e} -> {gap_post:.3e} (== (1-a) x "
                    f"{(1.0 - alpha) * gap_norm:.3e}) | S_graf "
                    f"{S_graf:.6f}/{budget_g:.4f} S_total {S_total:.4f}/"
                    f"{budget_norm:.4f}")
                n_burst += 1
                ok_t, temp = E261.burst_temp_check(
                    f"{tag}:graf:c{n_chunks}.x")
                chunk_temps.append(temp)
            # ---- the POST-actuator window row (t25/t200) ---------------
            if step in win_act:
                _read_window(f"{win_act[step]}:t{step}-post", step)

            if step in MILESTONES or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                theta_t = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                d_fact = theta_t - fact_flat_np          # THE drift read
                dn = float(np.linalg.norm(d_fact))
                pdf = room.project(d_fact)
                d_fact_in_room = float(np.linalg.norm(pdf) / dn) \
                    if dn > 0 else None
                gap_at_mile = float(np.linalg.norm(
                    graft_gap(room, fact_flat_np, theta_t)))
                rem = theta_t - base_flat_np             # the write's own
                rn = float(np.linalg.norm(rem))          # remaining disp
                pr = room.project(rem)
                rem_in_room = float(np.linalg.norm(pr) / rn) if rn > 0 \
                    else None
                tot_cum = corp_cum + graf_cum
                v_int_np = (tot_cum - tot_prev).double().cpu().numpy()
                vn = float(np.linalg.norm(v_int_np))
                if vn > 0:
                    pv = room.project(v_int_np)
                    in_room_frac_c = float(np.linalg.norm(pv) / vn)
                else:
                    in_room_frac_c = None
                tot_prev = tot_cum.clone()
                corp_np = corp_cum.double().cpu().numpy()
                corp_n = float(np.linalg.norm(corp_np))
                pcum = room.project(corp_np)
                corp_ir = (float(np.linalg.norm(pcum) / corp_n)
                           if corp_n > 0 else None)
                grf_np = graf_cum.double().cpu().numpy()
                grf_n = float(np.linalg.norm(grf_np))
                pgrf = room.project(grf_np)
                grf_ir = (float(np.linalg.norm(pgrf) / grf_n)
                          if grf_n > 0 else None)
                cum_n = float(np.linalg.norm(
                    (corp_cum + graf_cum).double().cpu().numpy()))
                bufC_ir, bufC_n = buffer_inroom_frac(opt_C, params_live,
                                                     room)
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_fact_norm": dn,
                    "drift_from_fact_in_room_frac": d_fact_in_room,
                    "in_room_gap_norm": gap_at_mile,
                    "remaining_from_base_norm": rn,
                    "remaining_from_base_in_room_frac": rem_in_room,
                    "corpus_disp_cum_norm": corp_n,
                    "corpus_disp_cum_in_room_frac": corp_ir,
                    "graf_disp_cum_norm": grf_n,
                    "graf_disp_cum_in_room_frac": grf_ir,
                    "total_disp_interval_norm": vn,
                    "total_disp_interval_in_room_frac": in_room_frac_c,
                    "total_disp_cum_norm": cum_n})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_inroom_frac": bufC_ir, "bufC_norm": bufC_n,
                    "bufF_inroom_frac": None, "bufF_norm": None,
                    "note": "NO fact-side optimizer on the graft arm (the "
                            "graft is a direct parameter edit; G_GRAFISOL)",
                    "n_graf": n_graf})
                state["budget_ledger"].append({
                    "step": step, "S_corpus": S_corpus, "S_graf": S_graf,
                    "S_total": S_total,
                    "S_total_usage_frac": S_total / budget_norm,
                    "cum_corpus_norm": corp_n, "cum_graf_norm": grf_n,
                    "cum_total_norm": cum_n,
                    "cum_total_usage_frac": cum_n / budget_norm,
                    "n_capped_so_far": n_capped, "n_graf_so_far": n_graf,
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped})
                state["traj"].append({
                    "step": step,
                    "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
                    "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                    "ce_corpus": float(loss_c.item()),
                    "survival_ratio_vs_committed":
                        bz0["mean_pz"] / FACT_BASELINE_G0,
                    "disp_norm": dn,
                    "in_room_frac": d_fact_in_room,
                    "in_room_gap_norm": gap_at_mile,
                    "remaining_in_room_frac": rem_in_room,
                    "corpus_disp_cum_in_room_frac": corp_ir,
                    "graf_disp_cum_in_room_frac": grf_ir,
                    "budget_S_total_usage_frac": S_total / budget_norm,
                    "budget_S_corpus": S_corpus, "budget_S_graf": S_graf,
                    "lr_applied": lr_t, "lr_sched": lr_sched,
                    "capped": capped, "n_graf_so_far": n_graf,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.6f} "
                    f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d| {dn:.3f} drift-in-room "
                    f"{('%.2e' % d_fact_in_room) if d_fact_in_room is not None else 'n/a'} "
                    f"GAP {gap_at_mile:.3e} "
                    f"| BUDGET S_corp {S_corpus:.4f} + S_graf "
                    f"{S_graf:.6f} = {S_total:.4f}/{budget_norm:.4f} "
                    f"({S_total / budget_norm:.1%}) cum {cum_n:.4f} "
                    f"({cum_n / budget_norm:.1%}) lr {lr_t:.5f} "
                    f"{'CAP' if capped else 'sched'} bufC "
                    f"{('%.*e' % (2, bufC_ir)) if bufC_ir is not None else 'n/a'}")
            if not ok_t:
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at t{step} — resume ckpt saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "optC": opt_C.state_dict(),
                    "cgen_state": cgen.get_state(),
                    "step": step, "traj": state["traj"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "buf_ledger": state["buf_ledger"],
                    "budget_ledger": state["budget_ledger"],
                    "lr_ledger": state["lr_ledger"],
                    "graf_ledger": state["graf_ledger"],
                    "window_ledger": state["window_ledger"],
                    "graisol": state["graisol"],
                    "corp_cum": corp_cum.cpu(),
                    "graf_cum": graf_cum.cpu(),
                    "tot_prev": tot_prev.cpu(),
                    "S_corpus": S_corpus, "S_graf": S_graf,
                    "n_capped": n_capped, "n_graf": n_graf,
                    "n_graf_capped": n_graf_capped,
                    "orth_max": orth_max,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    lrs = [v["lr_applied"] for v in state["lr_ledger"].values()]
    caps = [v["cap"] for v in state["lr_ledger"].values()]
    scheds = [v["lr_sched"] for v in state["lr_ledger"].values()]
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "buf_ledger": state["buf_ledger"],
            "budget_ledger": state["budget_ledger"],
            "lr_ledger": state["lr_ledger"],
            "graf_ledger": state["graf_ledger"],
            "window_ledger": state["window_ledger"],
            "graisol": state["graisol"],
            "S_corpus": S_corpus, "S_graf": S_graf,
            "n_capped": n_capped, "n_graf": n_graf,
            "n_graf_capped": n_graf_capped,
            "orth_max": orth_max,
            "lr_applied_min": min(lrs) if lrs else None,
            "lr_applied_median": float(sorted(lrs)[len(lrs) // 2])
            if lrs else None,
            "lr_sched_median": float(sorted(scheds)[len(scheds) // 2])
            if scheds else None,
            "cap_min": min(caps) if caps else None,
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}


# ======================================================================
# THE CONTROLLER-TWIN DRIVER — e288's chunked_errorgated_phase VERBATIM in
# body (the signal-borne reference: the NAME-ONLY CE at the error-gated
# lr over the split pools; tags + checkpoint names adapted, the mechanism
# UNTOUCHED)
# ======================================================================
def chunked_controller_twin_phase(tag: str, net0, proj: "E261.LadderRooms",
                                  fact_flat_np: np.ndarray,
                                  base_flat_np: np.ndarray,
                                  budget_c: float, budget_m: float,
                                  inst_x, inst_mask, anchor_full,
                                  train_ids, g0_ids, gm12_ids, r_eval_xy,
                                  zid, lr_m_max: float,
                                  resume_ck: Path, dev: torch.device) -> dict:
    """THE CONTROLLER-TWIN'S DRIVER (e288's ERROR-GATED arm VERBATIM). The
    net starts at the LOADED formed fact; per corpus step t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 29601 — bit-identical to
      the graft arm's draws); corpus batch = 16 original-host anchors + 32
      random corpus windows; full-window CE; lr_sched = LR_STABLE x
      cosine_lr(t-1, 1000); backward -> clip 1.0 -> the orthogonalized
      stream (g_perp, verified EVERY step) -> the per-step lr cap
      (equal-share reservation over the CORPUS stream's OWN remaining pool
      B_C) -> opt_C.step() (the corpus side's OWN buffer; BOTH optimizers
      snapshotted bitwise around the step).

      AND THE MAINTENANCE INJECTION (the signal): after every M=25 corpus
      steps, ONE NAME-ONLY gradient step — ix(16) TRUE install windows
      (win_i, G_NAMEWIN-gated) from igen seed 29602; token-level CE on the
      masked NAME positions ONLY (112; the pure name signal); backward ->
      clip 1.0 -> UNPROJECTED -> through opt_F (the FACT-side SGD-M) at
      the ERROR-GATED lr: deficit_t measured on the g0 battery, lr_m_t =
      LR_M_MAX * deficit_t, capped at the maintenance stream's share of
      B_M — e288's controller VERBATIM."""
    budget_norm = budget_c + budget_m       # display/ledger denominator
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    name_bs = G1.NAME_BS
    n_steps = PHASE_STEPS
    M = ACT_EVERY
    n_anc = anchor_full.shape[0]
    n_inst = inst_x.shape[0]
    N = int(fact_flat_np.size)
    win_mid = {WIN1[0]: f"win1:t{WIN1[0]}", WIN1[2]: f"win1:t{WIN1[2]}",
               WIN2[0]: f"win2:t{WIN2[0]}", WIN2[2]: f"win2:t{WIN2[2]}"}
    win_maint = {WIN1[1]: "win1", WIN2[1]: "win2"}
    state = {"step": 0, "traj": [], "corpus_ledger": {}, "orth_ledger": {},
             "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
             "lr_ledger": {}, "maint_ledger": [], "window_ledger": [],
             "bufsep": {"corpus_checks": 0, "corpus_violations": 0,
                        "maint_checks": 0, "maint_violations": 0,
                        "optF_steps": 0},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "maint_cum": torch.zeros(N, dtype=torch.float64),
             "tot_prev": torch.zeros(N, dtype=torch.float64),
             "S_corpus": 0.0, "S_maint": 0.0, "n_capped": 0,
             "n_maint": 0, "n_maint_capped": 0, "orth_max": 0.0}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at corpus step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        lrs_c = [v["lr_applied"] for v in state.get("lr_ledger",
                                                    {}).values()]
        return {"sd": state.get("model"), "traj": state.get("traj", []),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "buf_ledger": state.get("buf_ledger", []),
                "budget_ledger": state.get("budget_ledger", []),
                "lr_ledger": state.get("lr_ledger", {}),
                "maint_ledger": state.get("maint_ledger", []),
                "window_ledger": state.get("window_ledger", []),
                "bufsep": state.get("bufsep", {}),
                "S_corpus": state.get("S_corpus"),
                "S_maint": state.get("S_maint"),
                "n_capped": state.get("n_capped"),
                "n_maint": state.get("n_maint"),
                "n_maint_capped": state.get("n_maint_capped", 0),
                "orth_max": state.get("orth_max"),
                "lr_applied_min": min(lrs_c) if lrs_c else None,
                "lr_applied_median": (float(sorted(lrs_c)[len(lrs_c) // 2])
                                      if lrs_c else None),
                "lr_sched_median": None, "cap_min": None,
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt_C = opt_F = cgen = igen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[ROOM_MODE]
    corp_cum = state["corp_cum"].to(dev)
    maint_cum = state["maint_cum"].to(dev)
    tot_prev = state["tot_prev"].to(dev)
    S_corpus = float(state["S_corpus"])
    S_maint = float(state["S_maint"])
    S_total = S_corpus + S_maint
    n_capped = int(state["n_capped"])
    n_maint = int(state["n_maint"])
    n_maint_capped = int(state.get("n_maint_capped", 0))
    orth_max = 0.0

    def _read_window(phase_label: str, at_step: int) -> None:
        """the between-steps read (the sawtooth's eyes): battery g0 on the
        CPU eval net + the drift + the S split at this instant."""
        sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        bz0 = G1.battery_cell(evl, g0_ids, zid)
        theta_t = flat_params_cpu(net).double().numpy().astype(np.float64)
        dn = float(np.linalg.norm(theta_t - fact_flat_np))
        state["window_ledger"].append({
            "window_phase": phase_label, "step": at_step,
            "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
            "survival_ratio_vs_committed": bz0["mean_pz"] / FACT_BASELINE_G0,
            "drift_from_fact_norm": dn,
            "n_maint_so_far": n_maint,
            "S_corpus": S_corpus, "S_maint": S_maint,
            "elapsed_s": round(time.time() - T0, 1)})
        log(f"  [{tag}] WINDOW {phase_label} (after t{at_step}): g0 "
            f"{bz0['mean_pz']:.6f} "
            f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) |d| {dn:.3f} "
            f"| S_corp {S_corpus:.4f} S_maint {S_maint:.4f} "
            f"(maint #{n_maint})")

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{tag}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            # e285's DIAL 1 topology: TWO SGD instances over the SAME
            # parameters — the corpus side's buffer PRIVATE; the fact side
            # the maintenance instrument (stepped at maintenance steps
            # only, at the error-gated lr, unprojected).
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_F = torch.optim.SGD(net.parameters(), lr=lr_m_max,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            igen = torch.Generator().manual_seed(INST_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt_C.load_state_dict(state["optC"])
                opt_F.load_state_dict(state["optF"])
                cgen.set_state(state["cgen_state"])
                igen.set_state(state["igen_state"])
                step = state["step"]
                S_corpus = float(state["S_corpus"])
                S_maint = float(state["S_maint"])
                S_total = S_corpus + S_maint
                n_capped = int(state["n_capped"])
                n_maint = int(state["n_maint"])
                n_maint_capped = int(state.get("n_maint_capped", 0))
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            lr_sched = LR_STABLE * cosine_lr(step - 1, E261.INST_TOTAL)
            # ---- THE CORPUS STEP (the family's standard, bit-identical
            # draws to the graft arm)
            aj_c = torch.randint(n_anc, (corp_bs - mix_random,),
                                 generator=cgen)
            rj_c = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                                 generator=cgen)
            corp_c = torch.cat([anchor_full[aj_c],
                                torch.stack([train_ids[s: s + G1.BLOCK]
                                             for s in rj_c])], 0)
            xc = corp_c[:, :-1].to(dev)
            yc = corp_c[:, 1:].to(dev)
            logits_c, _ = net(xc)
            nll_c = F.cross_entropy(
                logits_c.reshape(-1, logits_c.shape[-1]),
                yc.reshape(-1), reduction="none").view(xc.shape[0],
                                                       xc.shape[1])
            loss_c = nll_c.mean()
            net.zero_grad(set_to_none=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gn_c = float(torch.cat([p.grad.detach().reshape(-1)
                                    for p in net.parameters()
                                    if p.grad is not None]).norm().item())
            # the RAW gradient's in-room fraction (the rig identity read)
            g_in_room = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                pg = room.project(g64)
                g_in_room = float(np.linalg.norm(pg)
                                  / max(np.linalg.norm(g64), 1e-30))
            # ---- the orthogonalized stream (verified EVERY step)
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- the per-step lr cap — equal-share reservation over the
            # CORPUS stream's OWN remaining pool B_C (e288's cap form)
            b_sq = pending_buffer_sqnorm(opt_C, params_live, SGD_MOMENTUM)
            b_norm = math.sqrt(max(b_sq, 0.0))
            rem_corpus = n_steps - step + 1
            remaining_c = max(budget_c - S_corpus, 0.0)
            share = remaining_c / rem_corpus
            cap = share / max(b_norm, 1e-12)
            lr_t = min(lr_sched, cap)
            capped = bool(cap < lr_sched)
            if capped:
                n_capped += 1
            for g_ in opt_C.param_groups:
                g_["lr"] = lr_t
            # ---- DIAL 1: the isolated corpus step (the fact side untouched)
            sn_f_before = snap_buffers(opt_F, params_live)
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()
            with torch.no_grad():
                d_step = torch.cat([p.detach().reshape(-1)
                                    for p in net.parameters()]) - theta_b
                corp_cum += d_step
            realized = float(d_step.norm().item())
            S_corpus += realized
            S_total = S_corpus + S_maint
            state["bufsep"]["corpus_checks"] += 1
            if not buffers_bitwise_equal(sn_f_before,
                                         snap_buffers(opt_F, params_live)):
                state["bufsep"]["corpus_violations"] += 1
                raise RuntimeError(
                    f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}: the "
                    "corpus step touched the fact side's state — HALT")
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c,
                                                "g_in_room_frac": g_in_room}
                state["orth_ledger"][step] = {
                    "gn_clipped": gn_c,
                    "gperp_norm": orth_row["gperp_norm"],
                    "norm_ratio": orth_row["norm_ratio"],
                    "in_room_frac": orth_row["in_room_frac"],
                    "orth_rel_err": orth_row["orth_rel_err"]}
                state["lr_ledger"][step] = {
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped, "b_norm": b_norm,
                    "share": share, "realized_step_norm": realized,
                    "rem_corpus": rem_corpus, "budget_c": budget_c,
                    "S_corpus": S_corpus}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(
                f"{tag}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)

            # ---- the plain window rows (the flanks: t24/t26, t199/t201)
            if step in win_mid:
                _read_window(win_mid[step], step)

            # ---- the PRE-maintenance window row (t25/t200) --------------
            if step in win_maint:
                _read_window(f"{win_maint[step]}:t{step}-pre", step)

            # ---- THE MAINTENANCE INJECTION (the signal, e288 VERBATIM) ---
            if step % M == 0:
                n_maint += 1
                # DIAL (i) — THE NAME-ONLY BATCH: ix(16) TRUE install
                # windows (win_i — ZEPHYRA spliced, e261's build_win
                # VERBATIM, G_NAMEWIN-gated); e261's Dmix corpus half
                # DROPPED (the pure name signal, no corpus dilution)
                ix = torch.randint(n_inst, (name_bs,), generator=igen)
                nw = inst_x[ix]
                x_i = nw[:, :-1].to(dev)
                y_i = nw[:, 1:].to(dev)
                m_i = inst_mask[ix].to(dev)
                logits_i, _ = net(x_i)
                nll_i = F.cross_entropy(
                    logits_i.reshape(-1, logits_i.shape[-1]),
                    y_i.reshape(-1), reduction="none"
                ).view(x_i.shape[0], x_i.shape[1])
                nm_i = nll_i[m_i]                    # the 112 name positions
                loss_i = nm_i.mean()                 # THE NAME-ONLY CE
                net.zero_grad(set_to_none=True)
                loss_i.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                gn_i = float(torch.cat(
                    [p.grad.detach().reshape(-1) for p in net.parameters()
                     if p.grad is not None]).norm().item())
                # the name-only gradient's in-room fraction (the brush
                # read: expect ~0.06 — the out-of-room stroke)
                g64i = torch.cat([p.grad.detach().reshape(-1)
                                  for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                gin_i = float(np.linalg.norm(room.project(g64i))
                              / max(np.linalg.norm(g64i), 1e-30))
                # DIAL (ii) — THE ERROR-GATED CONTROLLER (e288 VERBATIM):
                # measure the read's current deficit on the SAME g0
                # battery the bars read (the CPU eval net at the event
                # instant — at t25/t200 cross-checked against the window
                # pre-read), then lr_target = LR_M_MAX * deficit_t, capped
                # at the maintenance stream's OWN share of B_M
                sd_gate = {k: v.detach().cpu().clone()
                           for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_gate)
                evl.eval()
                read_gate = G1.battery_cell(evl, g0_ids, zid)["mean_pz"]
                deficit_t = min(1.0, max(
                    0.0, FACT_BASELINE_G0 - read_gate) / FACT_BASELINE_G0)
                gate_vs_pre = None
                if step in win_maint and state["window_ledger"] \
                        and state["window_ledger"][-1]["window_phase"] \
                        == f"{win_maint[step]}:t{step}-pre":
                    gate_vs_pre = abs(
                        read_gate - state["window_ledger"][-1]["g0_pz"])
                lr_target = lr_m_max * deficit_t
                assert lr_target <= lr_m_max * (1.0 + 1e-12), \
                    f"controller law broken: lr_target {lr_target} > " \
                    f"LR_M_MAX {lr_m_max}"
                if deficit_t <= 0.0:
                    assert lr_target == 0.0
                bm_sq = pending_buffer_sqnorm(opt_F, params_live,
                                              SGD_MOMENTUM)
                bm_norm = math.sqrt(max(bm_sq, 0.0))
                rem_maint_now = 1 + sum(1 for m in range(step + 1,
                                                         n_steps + 1)
                                        if m % M == 0)
                remaining_m = max(budget_m - S_maint, 0.0)
                share_m = remaining_m / rem_maint_now
                cap_m = share_m / max(bm_norm, 1e-12)
                lr_m = min(lr_target, cap_m)
                assert lr_m <= lr_m_max * (1.0 + 1e-9), \
                    f"applied lr_m {lr_m} exceeds LR_M_MAX {lr_m_max}"
                capped_m = bool(cap_m < lr_target)
                binder_m = "cap" if capped_m else "controller"
                if capped_m:
                    n_maint_capped += 1
                for g_ in opt_F.param_groups:
                    g_["lr"] = lr_m
                sn_c_before = snap_buffers(opt_C, params_live)
                theta_b2 = torch.cat([p.detach().reshape(-1)
                                      for p in net.parameters()])
                opt_F.step()
                with torch.no_grad():
                    d_m = torch.cat([p.detach().reshape(-1)
                                     for p in net.parameters()]) - theta_b2
                    maint_cum += d_m
                realized_m = float(d_m.norm().item())
                S_maint += realized_m
                S_total = S_corpus + S_maint
                state["bufsep"]["maint_checks"] += 1
                state["bufsep"]["optF_steps"] += 1
                if not buffers_bitwise_equal(
                        sn_c_before, snap_buffers(opt_C, params_live)):
                    state["bufsep"]["maint_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}'s "
                        "maintenance step: it touched the corpus side's "
                        "state — HALT")
                bufF_ir, bufF_n = buffer_inroom_frac(opt_F, params_live,
                                                     room)
                state["maint_ledger"].append({
                    "step": step, "maint_index": n_maint,
                    "name_ce": float(loss_i.item()),
                    "n_name_tokens": int(nm_i.numel()),
                    "gn_clipped": gn_i, "g_in_room_frac": gin_i,
                    "read_at_gate": read_gate, "deficit_t": deficit_t,
                    "gate_vs_pre_absdiff": gate_vs_pre,
                    "lr_m_max": lr_m_max, "lr_target": lr_target,
                    "cap_m": cap_m, "lr_maint": lr_m, "capped_m": capped_m,
                    "binder": binder_m,
                    "b_m_norm": bm_norm, "share_m": share_m,
                    "realized_step_norm": realized_m,
                    "realized_vs_share": (realized_m / share_m
                                          if share_m > 0 else None),
                    "bufF_inroom_frac": bufF_ir, "bufF_norm": bufF_n,
                    "S_maint": S_maint, "S_total": S_total,
                    "budget_m": budget_m,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] MAINT #{n_maint:2d} @t{step:3d}: name_ce "
                    f"{float(loss_i.item()):.4f} ({int(nm_i.numel())} toks) "
                    f"|g| {gn_i:.3f} in-room {gin_i:.4f} | GATE read "
                    f"{read_gate:.6f} deficit {deficit_t:.4f} -> lr "
                    f"{lr_m:.6f} {binder_m.upper()} (target "
                    f"{lr_target:.6f} cap {cap_m:.6f}; b_m {bm_norm:.3f} "
                    f"share {share_m:.5f}) -> step {realized_m:.5f} | "
                    f"S_maint {S_maint:.4f}/{budget_m:.4f} "
                    f"({S_maint / budget_m:.1%} of B_M) S_total "
                    f"{S_total:.4f}/{budget_norm:.4f} bufF "
                    f"{('%.*e' % (2, bufF_ir)) if bufF_ir is not None else 'n/a'}")
                n_burst += 1
                ok_t, temp = E261.burst_temp_check(
                    f"{tag}:maint:c{n_chunks}.x")
                chunk_temps.append(temp)
            # ---- the POST-maintenance window row (t25/t200) ------------
            if step in win_maint:
                _read_window(f"{win_maint[step]}:t{step}-post", step)

            if step in MILESTONES or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                theta_t = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                d_fact = theta_t - fact_flat_np          # THE drift read
                dn = float(np.linalg.norm(d_fact))
                pdf = room.project(d_fact)
                d_fact_in_room = float(np.linalg.norm(pdf) / dn) \
                    if dn > 0 else None
                rem = theta_t - base_flat_np             # the write's own
                rn = float(np.linalg.norm(rem))          # remaining disp
                pr = room.project(rem)
                rem_in_room = float(np.linalg.norm(pr) / rn) if rn > 0 \
                    else None
                tot_cum = corp_cum + maint_cum
                v_int_np = (tot_cum - tot_prev).double().cpu().numpy()
                vn = float(np.linalg.norm(v_int_np))
                if vn > 0:
                    pv = room.project(v_int_np)
                    in_room_frac_c = float(np.linalg.norm(pv) / vn)
                else:
                    in_room_frac_c = None
                tot_prev = tot_cum.clone()
                corp_np = corp_cum.double().cpu().numpy()
                corp_n = float(np.linalg.norm(corp_np))
                pcum = room.project(corp_np)
                corp_ir = (float(np.linalg.norm(pcum) / corp_n)
                           if corp_n > 0 else None)
                mnt_np = maint_cum.double().cpu().numpy()
                mnt_n = float(np.linalg.norm(mnt_np))
                pmnt = room.project(mnt_np)
                mnt_ir = (float(np.linalg.norm(pmnt) / mnt_n)
                          if mnt_n > 0 else None)
                cum_n = float(np.linalg.norm(
                    (corp_cum + maint_cum).double().cpu().numpy()))
                bufC_ir, bufC_n = buffer_inroom_frac(opt_C, params_live,
                                                     room)
                bufF_ir2, bufF_n2 = buffer_inroom_frac(opt_F, params_live,
                                                       room)
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_fact_norm": dn,
                    "drift_from_fact_in_room_frac": d_fact_in_room,
                    "remaining_from_base_norm": rn,
                    "remaining_from_base_in_room_frac": rem_in_room,
                    "corpus_disp_cum_norm": corp_n,
                    "corpus_disp_cum_in_room_frac": corp_ir,
                    "maint_disp_cum_norm": mnt_n,
                    "maint_disp_cum_in_room_frac": mnt_ir,
                    "total_disp_interval_norm": vn,
                    "total_disp_interval_in_room_frac": in_room_frac_c,
                    "total_disp_cum_norm": cum_n})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_inroom_frac": bufC_ir, "bufC_norm": bufC_n,
                    "bufF_inroom_frac": bufF_ir2, "bufF_norm": bufF_n2,
                    "optF_steps": state["bufsep"]["optF_steps"]})
                state["budget_ledger"].append({
                    "step": step, "S_corpus": S_corpus, "S_maint": S_maint,
                    "S_total": S_total,
                    "S_total_usage_frac": S_total / budget_norm,
                    "cum_corpus_norm": corp_n, "cum_maint_norm": mnt_n,
                    "cum_total_norm": cum_n,
                    "cum_total_usage_frac": cum_n / budget_norm,
                    "n_capped_so_far": n_capped, "n_maint_so_far": n_maint,
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped})
                state["traj"].append({
                    "step": step,
                    "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
                    "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                    "ce_corpus": float(loss_c.item()),
                    "survival_ratio_vs_committed":
                        bz0["mean_pz"] / FACT_BASELINE_G0,
                    "disp_norm": dn,
                    "in_room_frac": d_fact_in_room,
                    "remaining_in_room_frac": rem_in_room,
                    "corpus_disp_cum_in_room_frac": corp_ir,
                    "maint_disp_cum_in_room_frac": mnt_ir,
                    "budget_S_total_usage_frac": S_total / budget_norm,
                    "budget_S_corpus": S_corpus, "budget_S_maint": S_maint,
                    "lr_applied": lr_t, "lr_sched": lr_sched,
                    "capped": capped, "n_maint_so_far": n_maint,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.6f} "
                    f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d| {dn:.3f} drift-in-room "
                    f"{('%.2e' % d_fact_in_room) if d_fact_in_room is not None else 'n/a'} "
                    f"rem-in-room "
                    f"{('%.4f' % rem_in_room) if rem_in_room is not None else 'n/a'} "
                    f"| BUDGET S_corp {S_corpus:.4f} + S_maint "
                    f"{S_maint:.4f} = {S_total:.4f}/{budget_norm:.4f} "
                    f"({S_total / budget_norm:.1%}) cum {cum_n:.4f} "
                    f"({cum_n / budget_norm:.1%}) lr {lr_t:.5f} "
                    f"{'CAP' if capped else 'sched'} bufC "
                    f"{('%.*e' % (2, bufC_ir)) if bufC_ir is not None else 'n/a'}")
            if not ok_t:
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at t{step} — resume ckpt saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "optC": opt_C.state_dict(),
                    "optF": opt_F.state_dict(),
                    "cgen_state": cgen.get_state(),
                    "igen_state": igen.get_state(),
                    "step": step, "traj": state["traj"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "buf_ledger": state["buf_ledger"],
                    "budget_ledger": state["budget_ledger"],
                    "lr_ledger": state["lr_ledger"],
                    "maint_ledger": state["maint_ledger"],
                    "window_ledger": state["window_ledger"],
                    "bufsep": state["bufsep"],
                    "corp_cum": corp_cum.cpu(),
                    "maint_cum": maint_cum.cpu(),
                    "tot_prev": tot_prev.cpu(),
                    "S_corpus": S_corpus, "S_maint": S_maint,
                    "n_capped": n_capped, "n_maint": n_maint,
                    "n_maint_capped": n_maint_capped,
                    "orth_max": orth_max,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    lrs = [v["lr_applied"] for v in state["lr_ledger"].values()]
    caps = [v["cap"] for v in state["lr_ledger"].values()]
    scheds = [v["lr_sched"] for v in state["lr_ledger"].values()]
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "buf_ledger": state["buf_ledger"],
            "budget_ledger": state["budget_ledger"],
            "lr_ledger": state["lr_ledger"],
            "maint_ledger": state["maint_ledger"],
            "window_ledger": state["window_ledger"],
            "bufsep": state["bufsep"],
            "S_corpus": S_corpus, "S_maint": S_maint,
            "n_capped": n_capped, "n_maint": n_maint,
            "n_maint_capped": n_maint_capped,
            "orth_max": orth_max,
            "lr_applied_min": min(lrs) if lrs else None,
            "lr_applied_median": float(sorted(lrs)[len(lrs) // 2])
            if lrs else None,
            "lr_sched_median": float(sorted(scheds)[len(scheds) // 2])
            if scheds else None,
            "cap_min": min(caps) if caps else None,
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}

# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def _envelope_summary() -> dict:
    out = {"burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
           "per_step_polls": "after EVERY optimizer event AND every graft "
                             "event, both arms — aggregated from "
                             "runs/_envelope_log.jsonl (the persisted "
                             "ledger; survives resume passes; tags "
                             "e296:<ARM>:<phase> per the dispatch)",
           "early_end_margin_c": E261.TEMP_EARLY_END,
           "hard_line_c": E261.TEMP_HARD,
           "dispatch_envelope": "bursts <= 180s, cooldowns 30-60s, never "
                                "past 85C — this cell runs 175/40/84 (all "
                                "inside)"}
    temps = []
    try:
        with open(E43.REPO / "runs" / "_envelope_log.jsonl",
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tag = str(row.get("tag", ""))
                if tag.startswith(f"{NAME}:") and row.get("temp") is not None:
                    temps.append(float(row["temp"]))
    except FileNotFoundError:
        pass
    out["n_polls"] = len(temps)
    out["max_temp_seen_c"] = max(temps) if temps else (
        max((r["temp"] for r in thermal_log), default=None))
    out["violations_ge_84c"] = sum(1 for t in temps if t >= E261.TEMP_HARD)
    out["note"] = ("aggregated across ALL passes of this cell; the smoke's "
                   f"polls are tagged e296_smoke: and excluded")
    return out


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e296_prosthetic_graft",
        "phase": "THE PROSTHETIC GRAFT (the wild queue's #1) — maintenance "
                 "with NO teaching signal: is memory preservation a SIGNAL "
                 "problem or a GEOMETRY problem? (a) THE GRAFT: every M=25 "
                 "steps the frozen-copy transplant theta <- theta + alpha*"
                 "P_room(fact - theta) — a DIRECT parameter edit, no "
                 "gradient, no loss, no optimizer, the dose gated the same "
                 "way (deficit on the g0 battery) under the same budget "
                 "language (B_G = 40% equal-share cap); (b) THE "
                 "CONTROLLER-TWIN: e288's ERROR-GATED form VERBATIM on the "
                 "same session/draws (the signal-borne reference). "
                 "GRAFT-HOLDS vs SIGNAL-NEEDED vs MIXED, adjudicated on "
                 "the WRITE read at t400 with the gates' verdicts",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property; "
                           "the frozen bars read the WRITE and the "
                           "DISPLACEMENT only; the family's committed "
                           "NO-CONS form; both arms' states are "
                           "checkpointed"},
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; the "
                      "arms run SEQUENTIALLY with cooldowns between — never "
                      "concurrent) + CPU fp64 dense projections (pocketfft "
                      "workers 2), CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e296:<ARM>:<phase>",
            "trainings": "2 phases x 400 corpus opt steps (GRAFT: SGD-M "
                         "separate corpus buffer + orthogonal projection + "
                         "the corpus cap over B_C=60% + 16 GRAFT STEPS "
                         "(direct parameter edits at the deficit-gated "
                         "alpha, capped at the B_G=40% share; NO optimizer "
                         "on the graft path); CONTROLLER-TWIN: e288's "
                         "error-gated form VERBATIM (the name-only CE "
                         "through opt_F at lr_m_t = LR_M_MAX x deficit_t, "
                         "capped at the B_M=40% share); NO cons",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} corpus steps per arm; BOTH arms "
                     f"interleave {N_ACT_EXPECTED} actuator steps (one "
                     f"after every M={ACT_EVERY} corpus steps)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps (read POST-actuator)",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors (the same "
                           "60-window bank) + 32 random corpus windows; "
                           "full-window CE; THIS cell's ONE fresh registered "
                           f"generator seed {CORPUS_GEN_SEED}; BOTH arms "
                           "draw the IDENTICAL sequence (the actuator is "
                           "the arms' ONLY delta)",
            "graft_step": "THE PROSTHETIC FORM: after every "
                          f"M={ACT_EVERY} corpus steps ONE GRAFT STEP — the "
                          "frozen-copy transplant: theta <- theta + alpha_t "
                          "* P_room(fact - theta) (== alpha*(P_room "
                          "w_frozen - P_room theta); a DIRECT PARAMETER "
                          "EDIT, fp32 write, NO optimizer, NO gradient, NO "
                          "loss, NO draws); the dose gated THE SAME WAY as "
                          "the controller: deficit_t = max(0, "
                          f"{FACT_BASELINE_G0} - read)/"
                          f"{FACT_BASELINE_G0} on the SAME g0 battery; "
                          "alpha_t = min(ALPHA_MAX x deficit_t, cap_g), "
                          f"ALPHA_MAX = {ALPHA_MAX!r} (full deficit -> the "
                          "FULL transplant), cap_g = share_g/||gap||, "
                          "share_g = (B_G - S_graf)/rem_graf (the "
                          "symmetric equal-share reservation); the "
                          "transplant arithmetic machine-live at EVERY "
                          "event (the gap contracts by exactly (1-alpha)); "
                          "S_graf := the realized edit norm",
            "survival_ratio": f"post g0(t) / {FACT_BASELINE_G0} (the "
                              "committed loaded baseline, PRIMARY; the "
                              "session's loaded read co-reported)",
            "adjudication_read": "the t=400 endpoints (post the 16th "
                                 "actuator step) with the gates' verdicts "
                                 "at the same endpoints; trajectories + the "
                                 "cited curves (e288's controller, e285's "
                                 "sanctuary, e287's third form) co-reported",
        },
        "the_graft_convention": {
            "the_step": "ONE GRAFT STEP after every M="
                        f"{ACT_EVERY} corpus steps — exactly "
                        f"{N_ACT_EXPECTED} on the graft arm; the edit: "
                        "theta += alpha * P_room(fact - theta) (CPU fp64 "
                        "projection, fp32 parameter write through the room "
                        "machinery's offsets/shapes)",
            "the_gate": "THE SAME READ GATE as the controller: the read's "
                        "current deficit on the SAME g0 battery the bars "
                        "read (the CPU eval net at the event instant; at "
                        "t25/t200 cross-checked against the window "
                        "pre-read); alpha_t = ALPHA_MAX x deficit_t — the "
                        "sicker the read the stronger the transplant; "
                        "deficit 0 -> alpha 0 -> the edit is a no-op (the "
                        "event still machine-counted, the arithmetic "
                        "checks still run)",
            "the_dose": f"ALPHA_MAX = {ALPHA_MAX!r} — at full deficit the "
                        "FULL transplant (the in-room gap closed entirely: "
                        "consolidation-from-checkpoint, the anesthesia "
                        "ideal). NO optimizer -> NO momentum ramp to "
                        "calibrate against (e287's b_m trace inapplicable; "
                        "disclosed) -> the equal-share cap alone does the "
                        "budget-fitting: cap_g = ((B_G - S_graf)/"
                        "rem_graf)/||gap||, so S_graf <= B_G by "
                        "construction",
            "the_no_signal_claims": "the graft path uses NO loss, NO "
                                    "gradient, NO optimizer, NO data draws "
                                    "— machine-checked: grads bitwise-"
                                    "unchanged at "
                                    "every event (asserted), opt_C's "
                                    "buffers bitwise-unchanged around every "
                                    "edit (asserted), the graft arm has no "
                                    "install generator at all; the ONLY "
                                    "things the dose sees: the read (the "
                                    "gate) and the frozen geometry",
            "the_budget": f"BUDGET = {BUDGET_FRAC} x ||fact - base|| "
                          "(~4.59); S_total = S_corpus + S_graf BOTH "
                          "counted; the pools SPLIT (B_C = 0.60, B_G = "
                          "0.40) with S_total <= BUDGET by the triangle "
                          "bound BY CONSTRUCTION (e288's machinery "
                          "verbatim, the pools split on BOTH arms); a blown "
                          "G_BUDGET routes to MIXED (never a HALT)",
            "the_windows": f"t{WIN1[0]}/t{WIN1[1]}-pre/t{WIN1[1]}-post/"
                           f"t{WIN1[2]} AND t{WIN2[0]}/t{WIN2[1]}-pre/"
                           f"t{WIN2[1]}-post/t{WIN2[2]} (the SAME two "
                           "windows as e287/e288 — direct comparability "
                           "with the parent ledgers); the graft's "
                           "per-event read response read through the "
                           "pre/post rows",
            "the_twin": "e288's ERROR-GATED arm VERBATIM on this session's "
                        "shared stream (the name-only CE, install stream "
                        "29602, LR_M_MAX re-derived + asserted, the B_M "
                        "share cap, the split pools) — the live "
                        "signal-borne reference; e288's committed curve "
                        "cited (md5-bound)",
        },
        "deviations": deviations,
        "builds_on": [
            "the dispatch letter (the wild queue's #1: THE PROSTHETIC "
            "GRAFT — the question + the bars + the design frozen VERBATIM)",
            "e288 / THE ERROR-GATED MAINTENANCE CELL (THE DIRECT PARENT: "
            "the rig — the established fact, the orthogonalized traffic, "
            "the budget's 60/40 split pools, the controller's calibration "
            "LR_M_MAX, the milestone/window/battery conventions — ALL port "
            "from here; its ERROR-GATED arm IS this cell's twin; its "
            "committed founding success x3.4792 the hard-bound reference)",
            "T268 (e288's fold: the controller's law — the organism's own "
            "error a sufficient preservation signal; maintenance "
            "read-directed) + T275 (e313's fold: maintenance bearer-size-"
            "blind; the graft tests whether it is also SIGNAL-free)",
            "e313 / THE BEARER SESSION (the brush data the dispatch cites: "
            "the controller's stroke OUT-of-room — the maintenance "
            "gradient's in-room fraction at the ~0.06 chance class; the "
            "graft's stroke IN-room by construction: the clean contrast)",
            "T265 / e285 (THE SANCTUARY: the orthogonalized stream pins "
            "the in-room geometry and the read dies anyway — the graft "
            "arm's structural twin under P-e296c; the projection/cap "
            "machinery's provenance)",
            "T267 / e287 (the third form's SAWTOOTH-CONFIRMED + the "
            "room-frame-not-the-operative-frame law; the LR_M_MAX "
            "calibration trace's home)",
            "T263 / x14 (the directional-transport contrast: arm A's "
            "orthogonal subtraction resurrects x4328, arm B's in-room "
            "subtraction does not — the in-room/out-room actuation "
            "asymmetry this cell's graft tests at the maintenance seat)",
            "T261 / e283 + T264 / e284 + T266 / e286 (the established-"
            "write rig, the buffer separation, the misbind corrigendum)",
            "T239 / e261 + T242 / e264 (the fact itself + the ladder "
            "machinery PORTED WHOLE BY IMPORT)",
            "T259 / e281 (the NO-CONS form); T258 / e273 (the lr "
            "calibration, md5-bound)",
        ],
        "whats_new": [
            "THE PROSTHETIC GRAFT (the record's first signal-free "
            "maintenance actuator): a periodic DIRECT PARAMETER EDIT "
            "transplanting the write's in-room projection from a frozen "
            "external copy — no gradient, no loss, no optimizer, no data — "
            "gated by the same read deficit and held to the same budget "
            "language as the controller",
            "THE CLEANest STROKE CONTRAST IN THE RECORD: both arms share "
            "ONE stream, ONE budget split, ONE gating law; the arms' ONLY "
            "delta is the actuator's nature — the twin's gradient brush "
            "(out-of-room, ~0.06 in-room) vs the graft's geometric "
            "transplant (IN-room by construction)",
            "THE ENGAGEMENT LEDGER (P-e296c's registered discriminator): "
            "the per-event in-room gap ||P_room(fact - theta)|| — the "
            "first direct measurement of whether the era's protected "
            "traffic ever damages the room-frame at all",
        ],
        "gates": {},
    })
    log(f"E296 — THE PROSTHETIC GRAFT (the wild queue's #1; smoke={SMOKE}) "
        f"-> {RD}")
    log(f"arms: {' / '.join(ARMS)}; the fact = {FACT_CK} (md5-bound, post "
        f"g0 {FACT_BASELINE_G0:.8f}); the room = the fact's own committed "
        f"K10K (seeds {LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs "
        f"{ROOMS264_CK}); the phase = {PHASE_STEPS} corpus steps + "
        f"{N_ACT_EXPECTED} actuator steps/arm (M={ACT_EVERY}); THE GRAFT: "
        f"theta += alpha*P_room(fact - theta), alpha = min({ALPHA_MAX!r} x "
        f"deficit, cap_g) under B_G = 40%; THE TWIN: e288's error-gated "
        f"controller (LR_M_MAX {LR_M_MAX_FROZEN!r}); the corpus stream = "
        f"seed {CORPUS_GEN_SEED} (bit-identical across arms); the bars: "
        f"GRAFT-HOLDS >= {SURVIVE_FRAC:.0%}x at t400 (budget <= 100% + "
        f"stream live); SIGNAL-NEEDED (graft < 0.5x while the twin holds); "
        f"MIXED (partial)")
    write_partial("startup (bars + arms + the graft registered, committed "
                  "at birth)")
    set_seed(CORPUS_GEN_SEED)       # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (g1c's gates VERBATIM) ==
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    vocab = corpus.vocab_size
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    G_VOCAB = {"vocab_size": vocab, "expected": VOCAB_EXPECT,
               "pass": bool(vocab == VOCAB_EXPECT)}
    assert G_VOCAB["pass"], f"vocab drift: {vocab} != {VOCAB_EXPECT}"
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

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "note": "the install windows are NOT run as a phase in "
                          "this cell (the fact is loaded); the mask gate "
                          "carries the splice/mask convention's identity — "
                          "and the mask IS the twin's own teaching mask",
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    # ---- G_NAMEWIN: the TRUE install windows (win_i) — the twin's
    # name-only CE vehicle bind + THE E286 MISBIND DISCLOSURE carried
    name_ids = corpus.encode(G1.NAME)

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_i[i][G1.PRE: G1.PRE + len(G1.NAME)]) == G1.NAME
        for i in range(win_i.shape[0]))
    win_pre_ok = all(torch.equal(win_i[i][:G1.PRE],
                                 anchor_full[i][:G1.PRE])
                     for i in range(win_i.shape[0]))
    G_NAMEWIN = {
        "form": "the twin's maintenance batch's windows := e261's build_win "
                "VERBATIM (the TRUE install windows: 130 pre-context "
                "tokens | ZEPHYRA | 119 post tokens); the masked "
                "y-positions (x-cols PRE-1..PRE+5) hold the NAME in ALL "
                "60 windows (decode-verified) and the pre-context is "
                "bit-identical to the anchor bank's — THE VEHICLE OF THE "
                "TWIN'S NAME-ONLY CE (the e286 misbind's correction, "
                "carried)",
        "n_windows": int(win_i.shape[0]),
        "masked_decode_all_name": bool(win_masked_ok),
        "precontext_bit_equal_anchor": bool(win_pre_ok),
        "the_e286_misbind": {
            "finding": "e286's driver call bound anchor_full into the "
                       "inst_x slot: its maintenance batch's masked "
                       "positions held the HOST OPENINGS, NOT ZEPHYRA",
            "verified": "decode check at this cell's birth: win_i masked "
                        "positions == ZEPHYRA 60/60; anchor_full masked "
                        "positions == ZEPHYRA 0/60; win_i[:, :PRE] "
                        "bit-equal anchor_full[:, :PRE]",
        },
        "pass": bool(win_masked_ok and win_pre_ok
                     and list(win_i.shape) == [60, G1.BLOCK]),
    }
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"
    metrics["gates"]["G_NAMEWIN"] = G_NAMEWIN

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
                                         "wash/cons runs in this cell"},
                "pass": bool(len(n_starts) == 16)}
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                             "G_INSTMASK": G_INSTMASK})
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes "
        "/ e170 bank / install mask identity / THE NAME-WINDOW BIND — "
        "win_i: ZEPHYRA at the masked positions 60/60 / vocab 65)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_post = e264m["arms"]["K10K"]["install"]["post_cells"]["g0"]
    e268m = json.loads(E268_METRICS.read_text(encoding="utf-8"))
    e268_post_c = e268m["arms"]["CONCURRENT"]["install"]["post_cells"]["g0"]
    e283m = json.loads(E283_METRICS.read_text(encoding="utf-8"))
    e283_arm = e283m["arms"]["ESTABLISHED-CONCURRENT"]["phase"]
    e283_post = e283_arm["post_cells"]["g0"]
    e283_drift = e283_arm["disp_ledger"][-1]["drift_from_fact_norm"]
    e283m_ctrl = e283m["arms"]["ESTABLISHED-SERIAL-CONTROL"]["phase"]
    e284m = json.loads(E284_METRICS.read_text(encoding="utf-8"))
    e284_sep = e284m["arms"]["SEP"]["install"]["primary_median_t100_400"]
    e284_sha = e284m["arms"]["SHA"]["install"]["primary_median_t100_400"]
    x14m = json.loads(X14_METRICS.read_text(encoding="utf-8"))
    x14_reads = x14m["adjudication"]["reads"]
    e285m = json.loads(E285_METRICS.read_text(encoding="utf-8"))
    e285_verdict = e285m["adjudication"]["verdict"]
    e285_san = e285m["arms"]["SANCTUARY"]["phase"]["post_cells"]["g0"]
    e285_san_S = e285m["gates"]["G_BUDGET"]["S_final"]
    e285_san_drift = e285m["gates"]["G_BUDGET"]["drift_from_fact_final_norm"]
    e285_twn = e285m["arms"]["UNPROTECTED-TWIN"]["phase"]["post_cells"]["g0"]
    e286m = json.loads(E286_METRICS.read_text(encoding="utf-8"))
    e286_verdict = e286m["adjudication"]["verdict"]
    e286_rea_post = e286m["arms"]["RE-ANCHORED"]["phase"]["post_cells"]["g0"]
    e286_S_m = e286m["gates"]["G_BUDGET"]["re_anchored"]["S_maint_final"]
    e287m = json.loads(E287_METRICS.read_text(encoding="utf-8"))
    e287_verdict = e287m["adjudication"]["verdict"]
    e287_rea = e287m["arms"]["NAME-MAINTAINED"]["phase"]
    e287_post = e287_rea["post_cells"]["g0"]
    e287_ratio = e287m["adjudication"]["reads"]["NAME-MAINTAINED"][
        "survival_ratio_committed_denom"]
    e287_b_m = [r["b_m_norm"] for r in e287_rea["maint_ledger"]]
    e287_S_c = e287_rea["budget"]["S_corpus_final"]
    e287_S_m = e287_rea["budget"]["S_maint_final"]
    e288m = json.loads(E288_METRICS.read_text(encoding="utf-8"))
    e288_verdict = e288m["adjudication"]["verdict"]
    e288_rea = e288m["arms"]["ERROR-GATED"]["phase"]
    e288_post = e288_rea["post_cells"]["g0"]
    e288_ratio = e288m["adjudication"]["reads"]["ERROR-GATED"][
        "survival_ratio_committed_denom"]
    e288_traj = {t["step"]: t["g0_pz"] for t in e288_rea["traj"]}
    e288_S_c = e288_rea["budget"]["S_corpus_final"]
    e288_S_m = e288_rea["budget"]["S_maint_final"]
    e288_twn = e288m["arms"]["NAME-FIXED-TWIN"]["phase"]["post_cells"]["g0"]
    e288_def = e288_rea["controller_trace"]["per_event_deficit"]
    e313m = json.loads(E313_METRICS.read_text(encoding="utf-8"))
    e313_vA = e313m["adjudication_A"]["verdict"]
    e313_vB = e313m["adjudication_B"]["verdict"]
    vehicle = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    fact_sd = {k: v.detach().clone() for k, v in vehicle["model"].items()}
    del vehicle
    G_PARENTS = {
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "K10K_post_g0": e264_post,
                         "note": "THE loaded fact's committed record (the "
                                 "quiet-formed threshold rung)"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of(E268_METRICS), "bound_md5": E268_MD5,
                         "verdict": e268m["adjudication"]["verdict"],
                         "concurrent_post_g0": e268_post_c,
                         "ratio": E268_RATIO,
                         "note": "the FORMING-concurrent death (~0.0002x)"},
        "e283_metrics": {"path": str(E283_METRICS),
                         "md5": md5of(E283_METRICS), "bound_md5": E283_MD5,
                         "verdict": e283m["adjudication"]["verdict"],
                         "concurrent_post_g0": e283_post,
                         "ratio": E283_RATIO,
                         "drift_from_fact_norm": e283_drift,
                         "serial_control_post_g0":
                             e283m_ctrl["post_cells"]["g0"],
                         "note": "the unprotected transport cite"},
        "e284_metrics": {"path": str(E284_METRICS),
                         "md5": md5of(E284_METRICS), "bound_md5": E284_MD5,
                         "verdict": e284m["adjudication"]["verdict"],
                         "sep_primary": e284_sep, "sha_primary": e284_sha,
                         "note": "THE SEPARATION RECORD (DIAL 1+2's "
                                 "machinery + provenance)"},
        "x14_metrics": {"path": str(X14_METRICS),
                        "md5": md5of(X14_METRICS), "bound_md5": X14_MD5,
                        "verdict": x14m["adjudication"]["verdict"],
                        "arm_A_orthogonal_subtraction_g0":
                            x14_reads["arm_A_orthogonal_subtraction_g0"],
                        "arm_B_in_room_subtraction_g0":
                            x14_reads["arm_B_in_room_subtraction_g0"],
                        "resurrection_x_over_dead_state":
                            x14_reads["resurrection_x_over_dead_state"],
                        "note": "THE DIRECTIONAL-TRANSPORT RECORD (the "
                                "out-of-room subtraction resurrects x4328, "
                                "the in-room does not — the graft's "
                                "actuation class contrast, cited)"},
        "e285_metrics": {"path": str(E285_METRICS),
                         "md5": md5of(E285_METRICS), "bound_md5": E285_MD5,
                         "verdict": e285_verdict,
                         "sanctuary_post_g0": e285_san,
                         "sanctuary_ratio": E285_SAN_RATIO,
                         "sanctuary_S_final": e285_san_S,
                         "sanctuary_drift": e285_san_drift,
                         "sanctuary_drift_in_room": E285_SAN_DRIFT_INROOM,
                         "twin_post_g0": e285_twn,
                         "t100_drift": E285_T100_DRIFT,
                         "t100_ratio": E285_T100_RATIO,
                         "note": "THE GRAFT ARM'S STRUCTURAL TWIN "
                                 "(P-e296c): the orthogonalized stream "
                                 "pins the in-room geometry (drift 1.8445, "
                                 "in-room 1.23e-6) and the read dies "
                                 "anyway (x0.0030) — the room-frame is not "
                                 "where the death lives"},
        "e286_metrics": {"path": str(E286_METRICS),
                         "md5": md5of(E286_METRICS), "bound_md5": E286_MD5,
                         "verdict": e286_verdict,
                         "re_anchored_post_g0": e286_rea_post,
                         "ratio": E286_REA_RATIO,
                         "S_maint_final": e286_S_m,
                         "note": "the misbind parent (G_NAMEWIN's "
                                 "provenance)"},
        "e287_metrics": {"path": str(E287_METRICS),
                         "md5": md5of(E287_METRICS), "bound_md5": E287_MD5,
                         "verdict": e287_verdict,
                         "rea_post_g0": e287_post,
                         "ratio": e287_ratio,
                         "S_corpus_final": e287_S_c,
                         "S_maint_final": e287_S_m,
                         "b_m_trace": e287_b_m,
                         "b_m_sum": E287_B_M_SUM,
                         "lr_m_max_calibration": LR_M_MAX_FROZEN,
                         "note": "the third form + the twin's calibration "
                                 "trace (the b_m ramp LR_M_MAX divides)"},
        "e288_metrics": {"path": str(E288_METRICS),
                         "md5": md5of(E288_METRICS), "bound_md5": E288_MD5,
                         "verdict": e288_verdict,
                         "error_gated_post_g0": e288_post,
                         "error_gated_ratio": e288_ratio,
                         "error_gated_traj_g0": e288_traj,
                         "S_corpus_final": e288_S_c,
                         "S_maint_final": e288_S_m,
                         "flat_dose_twin_post_g0": e288_twn,
                         "deficit_trace": e288_def,
                         "maint_in_room_frac_median": E288_MAINT_INROOM_MEDIAN,
                         "note": "THIS CELL'S DIRECT PARENT (the twin's "
                                 "form VERBATIM + the founding success "
                                 "x3.4792 — the signal-borne reference; "
                                 "its controller trace + brush class the "
                                 "hard-bound cites)"},
        "e313_metrics": {"path": str(E313_METRICS),
                         "md5": md5of(E313_METRICS), "bound_md5": E313_MD5,
                         "verdict_A": e313_vA, "verdict_B": e313_vB,
                         "brush_class_band": list(E313_BRUSH_BAND),
                         "note": "the brush data the dispatch cites: the "
                                 "controller's stroke OUT-of-room (the "
                                 "maintenance gradient's in-room fraction "
                                 "at the ~0.06 chance class; T275: "
                                 "maintenance read-directed, bearer-size-"
                                 "blind) — the graft's IN-room stroke the "
                                 "clean contrast"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                "md5": md5of(E273_LRCAL),
                                "bound_md5": E273_LRCAL_MD5,
                                "lr_sgd": E273_LR_SGD,
                                "note": "LR_STABLE's provenance record"},
        "the_fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(CKPT_DIR / FACT_CK),
                     "bound_md5": FACT_MD5,
                     "size": (CKPT_DIR / FACT_CK).stat().st_size,
                     "bound_size": FACT_SIZE, "state": vehicle_state,
                     "note": "the established fact itself = THE FROZEN "
                             "EXTERNAL COPY the graft transplants from"},
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS264_CK),
                       "bound_md5": ROOMS264_MD5,
                       "note": "THE ROOM FILE (the fact's own room; the D/S "
                               "bit-bind itself is G_ROOMK10K)"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "e288_verdict": E288_VERDICT, "e288_post": E288_REA_POST,
            "e288_ratio": E288_REA_RATIO, "e288_S_corpus": E288_S_CORPUS,
            "e288_S_maint": E288_S_MAINT, "e288_twin_post": E288_TWN_POST,
            "e288_maint_inroom_median": E288_MAINT_INROOM_MEDIAN,
            "e313_verdicts": [E313_VERDICT_A, E313_VERDICT_B],
            "e283_verdict": E283_VERDICT,
            "e283_post": E283_POST, "e283_ratio": E283_RATIO,
            "e283_drift": E283_DRIFT, "e283_write_norm": E283_WRITE_NORM,
            "e285_verdict": E285_VERDICT, "e285_san_post": E285_SAN_POST,
            "e285_san_ratio": E285_SAN_RATIO, "e285_san_drift":
                E285_SAN_DRIFT,
            "e285_san_drift_in_room": E285_SAN_DRIFT_INROOM,
            "e285_S_final": E285_S_FINAL, "e285_budget_norm":
                E285_BUDGET_NORM,
            "x14_arm_A": X14_ARM_A, "x14_arm_B": X14_ARM_B,
            "x14_resurrection_x": X14_RESURRECTION_X,
            "e286_verdict": E286_VERDICT, "e286_rea_post": E286_REA_POST,
            "e286_rea_ratio": E286_REA_RATIO, "e286_S_maint": E286_S_MAINT,
            "e287_verdict": E287_VERDICT, "e287_rea_post": E287_REA_POST,
            "e287_rea_ratio": E287_REA_RATIO,
            "fact_md5": FACT_MD5, "fact_step": FACT_STEP,
            "fact_traj_steps": FACT_TRAJ_STEPS,
            "fact_ledger_max": FACT_LEDGER_MAX,
        },
        "pass": bool(
            e264m["adjudication"]["verdict"] == "SHARP-THRESHOLD"
            and abs(e264_post - FACT_BASELINE_G0) < 1e-12
            and e268m["adjudication"]["verdict"] == E268_VERDICT
            and abs(e268_post_c - E268_CONCURRENT_POST) < 1e-12
            and e283m["adjudication"]["verdict"] == E283_VERDICT
            and abs(e283_post - E283_POST) < 1e-12
            and abs(e283_drift - E283_DRIFT) < 1e-9
            and abs(e283m_ctrl["post_cells"]["g0"]
                    - FACT_BASELINE_G0) < 1e-9
            and e284m["adjudication"]["verdict"] == E284_VERDICT
            and abs(e284_sep - E284_SEP_PRIMARY) < 1e-12
            and abs(e284_sha - E284_SHA_PRIMARY) < 1e-12
            and x14m["adjudication"]["verdict"] == X14_VERDICT
            and e285_verdict == E285_VERDICT
            and abs(e285_san - E285_SAN_POST) < 1e-12
            and abs(e285_san_S - E285_S_FINAL) < 1e-9
            and abs(e285_san_drift - E285_SAN_DRIFT) < 1e-9
            and abs(e285_twn - E285_TWN_POST) < 1e-15
            and e286_verdict == E286_VERDICT
            and abs(e286_rea_post - E286_REA_POST) < 1e-12
            and abs(e286_S_m - E286_S_MAINT) < 1e-9
            and e287_verdict == E287_VERDICT
            and abs(e287_post - E287_REA_POST) < 1e-12
            and abs(e287_ratio - E287_REA_RATIO) < 1e-12
            and abs(e287_S_c - E287_S_CORPUS) < 1e-9
            and abs(e287_S_m - E287_S_MAINT) < 1e-9
            and len(e287_b_m) == 16
            and all(abs(a - b) < 1e-9 for a, b in zip(e287_b_m,
                                                      E287_B_M_TRACE))
            and abs(sum(e287_b_m) - E287_B_M_SUM) < 1e-9
            and e288_verdict == E288_VERDICT
            and abs(e288_post - E288_REA_POST) < 1e-12
            and abs(e288_ratio - E288_REA_RATIO) < 1e-12
            and abs(e288_S_c - E288_S_CORPUS) < 1e-9
            and abs(e288_S_m - E288_S_MAINT) < 1e-9
            and e288_traj == E288_TRAJ_G0
            and len(e288_def) == 16
            and all(abs(a - b) < 1e-12 for a, b in zip(e288_def,
                                                       E288_DEFICIT_TRACE))
            and abs(e288_twn - E288_TWN_POST) < 1e-12
            and e313_vA == E313_VERDICT_A and e313_vB == E313_VERDICT_B
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E283_METRICS) == E283_MD5
            and md5of(E284_METRICS) == E284_MD5
            and md5of(X14_METRICS) == X14_MD5
            and md5of(E285_METRICS) == E285_MD5
            and md5of(E286_METRICS) == E286_MD5
            and md5of(E287_METRICS) == E287_MD5
            and md5of(E288_METRICS) == E288_MD5
            and md5of(E313_METRICS) == E313_MD5
            and md5of(E273_LRCAL) == E273_LRCAL_MD5
            and md5of(CKPT_DIR / FACT_CK) == FACT_MD5
            and (CKPT_DIR / FACT_CK).stat().st_size == FACT_SIZE
            and vehicle_state["step"] == FACT_STEP
            and vehicle_state["traj_steps"] == FACT_TRAJ_STEPS
            and vehicle_state["ledger_max"] == FACT_LEDGER_MAX
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
            and (SMOKE or LADDER[0][0] == 10_000)),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e288 {E288_VERDICT} hard-bound (THE DIRECT "
        f"PARENT: the controller x{E288_REA_RATIO:.4f}, S {E288_S_CORPUS:.4f}"
        f"+{E288_S_MAINT:.4f}, its flat twin x{E288_TWN_RATIO:.4f}); e285 "
        f"{E285_VERDICT} (the sanctuary x{E285_SAN_RATIO:.4f} with the "
        f"in-room pinned at {E285_SAN_DRIFT_INROOM:.1e} — P-e296c's prior); "
        f"e313 {E313_VERDICT_A}/{E313_VERDICT_B} (the brush cite); the fact "
        f"ckpt md5/size/step-bound")
    write_partial("P0b parents hard-bound (the fact ckpt loaded + e288 + "
                  "e313 bound)")
    del e264m, e268m, e283m, e284m, x14m, e285m, e286m, e287m, e288m, e313m

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
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
    write_partial("P0c G-BASE PASSED")

    # ================= P1: THE ROOM (v-map + span + cert + bit-bind) ====
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
    base_flat_np = base_flat.double().numpy().astype(np.float64)

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "meta_k": vmap_art.get("meta", {}).get("k"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "mean_v": float(v64_np.mean()),
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

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the K10K room certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY and the kept^2 rank "
                "probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N); the span-overlap (expect ~sqrt(k/N))",
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

    # ---- G_ROOMK10K: bit-identity vs e264's committed K10K room ---------
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    k10k_name = RUNG[LADDER[0][0]]
    if not SMOKE:
        D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
        S264 = _to_np(rooms264["model"]["K10K"]["S"])
        D_mine = rooms.rooms[k10k_name].D
        S_mine = rooms.rooms[k10k_name].S
        G_ROOMK10K = {
            "form": "the room == the fact's own committed K10K room (seeds "
                    "26113/26114 at k=10,000): the +-1 diagonal and the "
                    "index set bit-identical to e264_rooms.pt's stored "
                    "K10K D/S (exact equality)",
            "D_bit_equal": bool(np.array_equal(D_mine, D264)),
            "S_bit_equal": bool(np.array_equal(S_mine, S264)),
            "e264_rooms_md5": md5of(CKPT_DIR / ROOMS264_CK),
            "pass": bool(np.array_equal(D_mine, D264)
                         and np.array_equal(S_mine, S264)
                         and int(rooms264["model"]["K10K"]["k"])
                         == LADDER[0][0]
                         and list(rooms264["model"]["K10K"]["seeds"])
                         == [LADDER[0][1], LADDER[0][2]]),
        }
        del rooms264
    else:
        G_ROOMK10K = {
            "form": "SMOKE: the room shares the seed pair (26113/26114) at "
                    "smoke k — no committed record at this k; the bit-bind "
                    "is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
        del rooms264
    assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    log(f"P1 G_ROOMK10K: the fact's room "
        f"{('bit-identical to e264_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e296_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e296's room (the fact's own room): the committed K10K "
                 "room (seeds 26113/26114), rebuilt + bit-gated vs "
                 "e264_rooms.pt",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][k10k_name].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": k10k_name,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS264_CK} "
                                    "(e264's committed K10K room — the "
                                    "fact's own room)"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE ROOM: {k10k_name} (k={LADDER[0][0]}): BUILT + CERTIFIED + "
        f"BIT-BOUND (the fact's own writing room)")
    write_partial("P1 the room built (parents bound + v-map loaded + span "
                  "loaded + certification + bit-bind)")

    # ---- G_TRANSPLANT: the graft arithmetic's fp64-exact unit test ------
    gt = transplant_unit_test(rooms.rooms[k10k_name], N)
    G_TRANSPLANT = gt
    assert G_TRANSPLANT["pass"], \
        f"transplant unit test FAILED: {G_TRANSPLANT}"
    metrics["gates"]["G_TRANSPLANT"] = G_TRANSPLANT
    log(f"P1b G_TRANSPLANT: the transplant arithmetic fp64-EXACT on a "
        f"synthetic in-room gap (gap {gt['gap_norm']:.4f}; alpha 0.37/1.0/0: "
        f"contraction errors "
        + "/".join(f"{c['gap_contraction_err']:.1e}"
                   for c in gt["checks"].values())
        + ", edit-vs-alpha*v errors "
        + "/".join(f"{c['edit_vs_alpha_v_err']:.1e}"
                   for c in gt["checks"].values())
        + "): PASS")
    write_partial("P1b G_TRANSPLANT PASSED (the dispatch's smoke demand, "
                  "fp64-exact)")

    # ---- G_LR_BIND: the lr/momentum provenance bind ----------------------
    lr_sgd_runtime = float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    lr_stable_runtime = SGD_STABLE_FACTOR * lr_sgd_runtime
    G_LR_BIND = {
        "form": "the corpus lr := e273's STABLE RIDER POINT (the family's "
                "committed class) — LR_STABLE = x0.01 x LR_SGD_matched, "
                "re-derived at runtime from the md5-bound runs/e273/"
                "lr_calibration.json and asserted == the frozen literal; "
                "momentum 0.9, wd 0.0 EXACTLY. THE TWIN'S controller "
                "ceiling: LR_M_MAX = 0.40 x BUDGET / sum(e287's 16 b_m) = "
                f"{LR_M_MAX_FROZEN!r} (re-derived from the loaded "
                "write_norm at P2b and asserted). THE GRAFT's alpha: "
                "ALPHA_MAX = 1.0 (a FRACTION, not an lr — the transplant "
                "closes the (1-alpha) share of the in-room gap; no "
                "optimizer, no lr class applies; disclosed)",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "alpha_max": ALPHA_MAX,
        "alpha_max_form": "ALPHA_MAX = 1.0: at full deficit the FULL "
                          "transplant (P_room theta := P_room w_frozen); "
                          "the budget-fitting is the equal-share cap's "
                          "job alone (cap_g = share_g/||gap||)",
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
        "pass": bool(abs(lr_stable_runtime - LR_STABLE) < 1e-15
                     and float(LR_STABLE) == 0.21738574801453703
                     and abs(sum(E287_B_M_TRACE) - E287_B_M_SUM) < 1e-9
                     and SGD_MOMENTUM == 0.9 and SGD_WD == 0.0
                     and lr_sgd_runtime == E273_LR_SGD
                     and ALPHA_MAX == 1.0),
    }
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P1c G_LR_BIND: LR_STABLE = {SGD_STABLE_FACTOR} x {E273_LR_SGD} = "
        f"{LR_STABLE!r}; the twin's LR_M_MAX = 0.40 x BUDGET / "
        f"{E287_B_M_SUM:.4f} = {LR_M_MAX_FROZEN!r} (re-derived + asserted "
        f"at P2b); the graft's ALPHA_MAX = {ALPHA_MAX!r}: PASS")
    write_partial("P1c G_LR_BIND PASSED")

    # ================= P2: THE FACT (loaded bit-exact + read + gated) ===
    log("=" * 78)
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_g0 = G1.battery_cell(fact_net, g0_ids, zid)["mean_pz"]
    fact_gm12 = G1.battery_cell(fact_net, gm12_ids, zid)["mean_pz"]
    fact_gp12 = G1.battery_cell(fact_net, bat_ids[12], zid)["mean_pz"]
    fact_ce_r = G1.ce_fixed_cpu(fact_net, *r_eval_xy)
    d_fact_base = fact_flat - base_flat
    loads_fact = rooms.displacement_loads(d_fact_base, ROOM_MODE)
    write_norm = float(np.linalg.norm(fact_flat_np - base_flat_np))
    G_FACTLOAD = {
        "form": "the established fact, loaded BIT-EXACT and gated THREE "
                "ways: (1) the artifact (md5/size/step/traj/ledger — "
                "G_PARENTS), (2) the loaded state's flat-md5, (3) the "
                "behavioral read (post g0 within "
                f"{FACT_READ_TOL_G0:.0e} / gm12 within "
                f"{FACT_READ_TOL_GM12:.0e} of e264's committed literals — "
                "the family's cross-session read-determinism law). THIS "
                "LOADED STATE IS ALSO THE GRAFT'S FROZEN EXTERNAL COPY "
                "(the transplant's target geometry; never updated)",
        "flat_md5": fact_flat_md5,
        "read_g0": {"mine": fact_g0, "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_g0 - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_gm12, "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_gm12 - FACT_BASELINE_GM12)},
        "read_gp12": fact_gp12, "read_ce_r": fact_ce_r,
        "write_norm": write_norm,
        "survival_ratio_denominator": "the COMMITTED literal (PRIMARY); "
                                      "the session read co-reported",
        "displacement_from_base": {
            "norm": write_norm,
            **loads_fact,
            "note": "the write's own standing displacement (the budget = "
                    f"{BUDGET_FRAC} x this)"},
        "pass": bool(abs(fact_g0 - FACT_BASELINE_G0) <= FACT_READ_TOL_G0
                     and abs(fact_gm12 - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P2 G_FACTLOAD: the formed fact LOADS — read g0 {fact_g0:.10f} vs "
        f"committed {FACT_BASELINE_G0:.10f} (|d| "
        f"{abs(fact_g0 - FACT_BASELINE_G0):.1e}); gm12 |d| "
        f"{abs(fact_gm12 - FACT_BASELINE_GM12):.1e}; flat md5 "
        f"{fact_flat_md5[:10]}...; the write stands "
        f"{loads_fact['in_own_room']:.4f} in-room, norm {write_norm:.4f} "
        f"(the budget's denominator): PASS")
    metrics["the_fact"] = {
        "checkpoint": f"runs/checkpoints/{FACT_CK}", "md5": FACT_MD5,
        "committed_post_g0": FACT_BASELINE_G0, "session_read_g0": fact_g0,
        "session_read_gm12": fact_gm12, "session_read_ce_r": fact_ce_r,
        "write_norm": write_norm,
        "displacement_from_base": G_FACTLOAD["displacement_from_base"],
        "baseline_used_for_ratios": "committed (PRIMARY)",
        "role_in_this_cell": "the vehicle (loaded bit-exact) AND the "
                             "graft's frozen external copy (the "
                             "transplant's target geometry)",
    }
    write_partial("P2 the fact loaded bit-exact + gated (the baseline read)")
    del fact_net

    # ---- the BUDGET (frozen at load time; the TOTAL + the SPLIT pools) --
    budget_norm = BUDGET_FRAC * write_norm      # the TOTAL (never smoke-scaled)
    # THE TWIN'S CALIBRATION, re-derived from the LOADED write_norm and
    # asserted == the frozen birth literal
    lr_m_max = MAINT_BUDGET_SHARE * budget_norm / E287_B_M_SUM
    assert abs(lr_m_max - LR_M_MAX_FROZEN) < 1e-9, \
        f"LR_M_MAX runtime {lr_m_max!r} != frozen {LR_M_MAX_FROZEN!r}"
    budget_c = CORPUS_BUDGET_SHARE * budget_norm    # the corpus pool (BOTH arms)
    budget_g = MAINT_BUDGET_SHARE * budget_norm     # the GRAFT arm's pool
    budget_m = MAINT_BUDGET_SHARE * budget_norm     # the TWIN's pool (same 40%)
    if SMOKE:
        # the smoke budgets are SCALED PER-STREAM to the full run's
        # per-EVENT shares (corpus 400 events / actuator 16 events);
        # LR_M_MAX + ALPHA_MAX stay the full-run constants (disclosed)
        budget_c = budget_c * PHASE_STEPS / 400
        budget_g = budget_g * (PHASE_STEPS // ACT_EVERY) / 16
        budget_m = budget_g
    metrics["budget"] = {
        "form": f"BUDGET := {BUDGET_FRAC} x ||fact - base|| (the write's "
                "own norm, fp64 from the loaded fact); S_total = S_corpus "
                "+ S_act BOTH counted; the pools SPLIT (B_C = "
                f"{CORPUS_BUDGET_SHARE:.0%} / B_act = "
                f"{MAINT_BUDGET_SHARE:.0%}) on BOTH arms — B_G (the graft) "
                "and B_M (the twin's maintenance) are the SAME allocation "
                "given to different actuators",
        "write_norm": write_norm,
        "budget_norm": budget_norm,
        "the_calibration": {
            "lr_m_max": lr_m_max,
            "lr_m_max_frozen": LR_M_MAX_FROZEN,
            "formula": "LR_M_MAX = 0.40 x BUDGET / sum(e287's 16 b_m "
                       f"norms = {E287_B_M_SUM!r})",
            "alpha_max": ALPHA_MAX,
            "alpha_form": "the graft's dial is a FRACTION: at full deficit "
                          "the FULL transplant; cap_g = ((B_G - S_graf)/"
                          "rem_graf)/||gap|| does the budget-fitting",
        },
        "pools": {
            "graft": {"budget_c": budget_c, "budget_g": budget_g,
                      "reservation": "PER-STREAM equal-share: corpus "
                                     "cap_t = ((B_C - S_corpus)/"
                                     "rem_corpus)/||b_t||; graft cap_g = "
                                     "((B_G - S_graf)/rem_graf)/||gap|| — "
                                     "every event's realized displacement "
                                     "fits its share, so S_total <= B_C + "
                                     "B_G = BUDGET by the triangle bound "
                                     "BY CONSTRUCTION"},
            "twin": {"budget_c": budget_c, "budget_m": budget_m,
                     "reservation": "e288's split-pool form VERBATIM: "
                                    "cap_t over B_C; cap_m = ((B_M - "
                                    "S_maint)/rem_maint)/||b_m|| with the "
                                    "controller's target lr = LR_M_MAX x "
                                    "deficit_t"},
        },
        "slack": BUDGET_SLACK,
        "smoke_scaling": (f"SMOKE: B_C x {PHASE_STEPS}/400, B_G/B_M x "
                          f"{PHASE_STEPS // ACT_EVERY}/16 — the full run's "
                          "per-event shares preserved; LR_M_MAX + ALPHA_MAX "
                          "unscaled (disclosed)"
                          if SMOKE else None),
    }
    log(f"P2b THE BUDGET: {BUDGET_FRAC} x {write_norm:.4f} = "
        f"{budget_norm:.4f}; the pools: B_C {budget_c:.4f} / B_G "
        f"{budget_g:.4f} (graft) = B_M {budget_m:.4f} (twin — the SAME 40% "
        f"given to different actuators); the twin's calibration LR_M_MAX = "
        f"{lr_m_max!r} (frozen {LR_M_MAX_FROZEN!r}, asserted); the graft's "
        f"ALPHA_MAX = {ALPHA_MAX!r}")
    write_partial("P2b the budget computed (the split pools + the twin's "
                  "calibration asserted)")

    # ---- G_CORPUSGEN: the corpus-generator registration ------------------
    scratch = torch.Generator().manual_seed(CORPUS_GEN_SEED)
    aj_probe = torch.randint(anchor_full.shape[0],
                             (E43.CORP_BS - E43.MIX_RANDOM,),
                             generator=scratch)
    rj_probe = torch.randint(len(train_ids) - G1.BLOCK - 1,
                             (E43.MIX_RANDOM,), generator=scratch)
    G_CORPUSGEN = {
        "form": "the REGISTERED fresh corpus stream: seed "
                f"{CORPUS_GEN_SEED} (the family's per-cell rule — "
                ".../28801/29101/.../HERE 29601), its own generator; per "
                "corpus step the draws (aj_c(16), rj_c(32)) in that order; "
                "the first step's draws logged here from a scratch "
                "generator (the drivers' cgens start identically by "
                "construction); BOTH arms draw the IDENTICAL sequence",
        "seed": CORPUS_GEN_SEED,
        "first_step_aj": [int(x) for x in aj_probe.tolist()],
        "first_step_rj": [int(x) for x in rj_probe.tolist()],
        "composition": "16 original-host anchors (the 60-window bank) + 32 "
                       "random corpus windows (contiguous train_ids "
                       "slices); full-window CE; clip 1.0 -> the arm's own "
                       "step (GRAFT: g_perp + capped lr + opt_C + the "
                       "graft injection; TWIN: + the name-only maintenance "
                       "through opt_F)",
        "pass": True,
    }
    metrics["gates"]["G_CORPUSGEN"] = G_CORPUSGEN
    log(f"P2c G_CORPUSGEN: the corpus stream REGISTERED (seed "
        f"{CORPUS_GEN_SEED}; first draws aj[:4]="
        f"{G_CORPUSGEN['first_step_aj'][:4]} "
        f"rj[:4]={G_CORPUSGEN['first_step_rj'][:4]})")

    # ---- G_GRAFTBIND: the graft mechanism's own bind ----------------------
    G_GRAFTBIND = {
        "form": "the graft mechanism's bind (THE PROSTHETIC FORM): ONE "
                "graft step after every M="f"{ACT_EVERY} corpus steps — "
                f"exactly {N_ACT_EXPECTED} on the graft arm; the edit: "
                "theta += alpha * P_room(fact - theta) (== alpha*(P_room "
                "w_frozen - P_room theta) in displacement language; the "
                "frozen external copy := the LOADED fact, never updated); "
                "a DIRECT PARAMETER EDIT through NO optimizer — fp64 CPU "
                "projection, fp32 parameter write; NO gradient, NO loss, "
                "NO draws (the graft arm has NO install generator). THE "
                "GATE (the same read): deficit_t = max(0, "
                f"{FACT_BASELINE_G0} - read)/"
                f"{FACT_BASELINE_G0} on the SAME g0 battery the bars read; "
                "THE DOSE: alpha_t = min(ALPHA_MAX x deficit_t, cap_g) "
                f"with ALPHA_MAX = {ALPHA_MAX!r}; THE CAP: cap_g = "
                "((B_G - S_graf)/rem_graf)/||gap|| (the symmetric "
                "equal-share form); THE ARITHMETIC: machine-live at EVERY "
                "event (the gap contracts by exactly (1-alpha); the edit's "
                "norm == alpha x gap) + the fp64-exact unit test "
                "G_TRANSPLANT; alpha == 0 at deficit 0 (a no-op edit, "
                "still machine-counted)",
        "act_every": ACT_EVERY,
        "n_graf_expected": N_ACT_EXPECTED,
        "alpha_max": ALPHA_MAX,
        "graft_law": "deficit_t = max(0, 0.26464763283729553 - read_t)/"
                     "0.26464763283729553 (clipped to [0,1]); alpha_t = "
                     "min(ALPHA_MAX x deficit_t, cap_g); cap_g = ((B_G - "
                     "S_graf)/rem_graf)/||gap||",
        "no_signal_claims": {"no_loss": True, "no_gradient": True,
                             "no_optimizer": True, "no_draws": True,
                             "machine_checks": "grads bitwise-unchanged "
                                               "at every "
                                               "event (asserted); opt_C "
                                               "buffers bitwise-unchanged "
                                               "around every edit "
                                               "(asserted)"},
        "pass": bool(ACT_EVERY == (2 if SMOKE else 25)
                     and N_ACT_EXPECTED == PHASE_STEPS // ACT_EVERY
                     and ALPHA_MAX == 1.0
                     and G_TRANSPLANT["pass"]),
    }
    assert G_GRAFTBIND["pass"], f"graft bind failed: {G_GRAFTBIND}"
    metrics["gates"]["G_GRAFTBIND"] = G_GRAFTBIND
    log(f"P2c G_GRAFTBIND: the graft REGISTERED (M={ACT_EVERY} -> "
        f"{N_ACT_EXPECTED} steps; theta += alpha*P_room(fact - theta), "
        f"alpha = min({ALPHA_MAX!r} x deficit, cap_g); NO loss / gradient "
        f"/ optimizer / draws — all machine-checked): PASS")

    # ---- G_MAINTBIND: the twin's maintenance mechanism bind ----------------
    isc = torch.Generator().manual_seed(INST_GEN_SEED)
    ix_probe = torch.randint(win_i.shape[0], (G1.NAME_BS,), generator=isc)
    n_name_tokens = int(inst_mask[:G1.NAME_BS].sum())
    G_MAINTBIND = {
        "form": "the twin's maintenance bind (e288's ERROR-GATED arm "
                "VERBATIM): ONE name-only gradient step after every M="
                f"{ACT_EVERY} corpus steps — exactly {N_ACT_EXPECTED}; the "
                "batch := ix(16) TRUE install windows (win_i, G_NAMEWIN-"
                "gated) from the registered install generator seed "
                f"{INST_GEN_SEED}; THE NAME-ONLY CE := mean token-level CE "
                "over the masked NAME positions ONLY (7 x 16 = 112); clip "
                "1.0; UNPROJECTED, through opt_F (the fact-side SGD-M "
                "momentum 0.9 wd 0, private buffer) at the error-gated lr "
                "lr_m_t = LR_M_MAX * deficit_t, capped at the B_M share",
        "act_every": ACT_EVERY,
        "n_maint_expected": N_ACT_EXPECTED,
        "lr_m_max": LR_M_MAX_FROZEN,
        "controller_law": "deficit_t = max(0, 0.26464763283729553 - "
                          "read_t)/0.26464763283729553 (clipped to [0,1]); "
                          "lr_m_t = LR_M_MAX * deficit_t; applied = min("
                          "lr_m_t, cap_m); cap_m = ((B_M - S_maint)/"
                          "rem_maint)/||b_m||",
        "inst_gen_seed": INST_GEN_SEED,
        "first_maint_draws": {"ix": [int(x) for x in ix_probe.tolist()]},
        "name_token_disclosure": {
            "name_masked_tokens_per_step": n_name_tokens,
            "corpus_tokens_per_step": 0,
            "note": "the PURE NAME SIGNAL: 112 masked name positions per "
                    "step, NO corpus tokens; the brush read: this "
                    "gradient's in-room fraction per step (expect ~0.06 — "
                    "the OUT-of-room stroke, e288/e313's class)"},
        "pass": bool(ACT_EVERY == (2 if SMOKE else 25)
                     and N_ACT_EXPECTED == PHASE_STEPS // ACT_EVERY
                     and abs(LR_M_MAX_FROZEN * E287_B_M_SUM
                             - MAINT_BUDGET_SHARE * BUDGET_FRAC
                             * E283_WRITE_NORM) < 1e-6
                     and inst_mask.dtype == torch.bool
                     and n_name_tokens == G1.NAME_BS * len(G1.NAME)
                     and G_NAMEWIN["pass"]),
    }
    assert G_MAINTBIND["pass"], f"maintenance bind failed: {G_MAINTBIND}"
    metrics["gates"]["G_MAINTBIND"] = G_MAINTBIND
    log(f"P2c G_MAINTBIND: the twin's maintenance REGISTERED (M="
        f"{ACT_EVERY} -> {N_ACT_EXPECTED} steps; lr_m_t = "
        f"{LR_M_MAX_FROZEN!r} x deficit_t capped at the B_M share; "
        f"install stream {INST_GEN_SEED}; the name tokens "
        f"{n_name_tokens}/step): PASS")
    write_partial("P2c the generators + the graft + the twin's maintenance "
                  "bound")

    # ================= P3: ARM GRAFT (the prosthetic build) ==============
    log("=" * 78)
    log(f"ARM-{GRF_ARM} — {ARM_DESC[GRF_ARM]}")
    grf = chunked_graf_phase(
        f"{GRF_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, budget_c, budget_g, anchor_full,
        train_ids, g0_ids, gm12_ids, r_eval_xy, zid,
        CKPT_DIR / ("smoke_e296_GRAFT_resume.pt" if SMOKE
                    else "e296_GRAFT_resume.pt"), dev)
    sd_g = grf["sd"]
    net_g = G1.evl_load(sd_g)
    cells_g = {"gm12": G1.battery_cell(net_g, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_g, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_g, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_g, *r_eval_xy)}
    d_final_g = flat_params_cpu(net_g) - fact_flat
    loads_final_g = rooms.displacement_loads(d_final_g, ROOM_MODE)
    gap_final_g = float(np.linalg.norm(
        graft_gap(rooms.rooms[k10k_name], fact_flat_np,
                  flat_params_cpu(net_g).double().numpy().astype(np.float64))))
    del net_g
    corp_ce_g = [v["ce"] for v in grf["corpus_ledger"].values()]
    corp_gn_g = [v["gn_clipped"] for v in grf["corpus_ledger"].values()]
    corp_gin_g = [v["g_in_room_frac"] for v in grf["corpus_ledger"].values()
                  if v.get("g_in_room_frac") is not None]
    graf_alphas = [m["alpha"] for m in grf["graf_ledger"]]
    graf_gaps = [m["gap_norm_pre"] for m in grf["graf_ledger"]]
    graf_edits = [m["edit_norm"] for m in grf["graf_ledger"]]
    grf_ck = save_ckpt(
        "e296_GRAFT_post", sd_g,
        {"desc": "e296 ARM-GRAFT post-phase state: the committed "
                 "quiet-formed 10k fact + 400 budgeted orthogonal SGD-M "
                 "corpus steps over B_C (separate buffer, corpus stream "
                 f"{CORPUS_GEN_SEED}) + {grf['n_graf']} GRAFT STEPS (the "
                 "frozen-copy transplants at the deficit-gated alpha, "
                 "capped at the B_G share; NO optimizer on the graft "
                 "path) — NO cons",
         "arm": GRF_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "alpha_max": ALPHA_MAX,
         "act_every": ACT_EVERY,
         "budget_norm": budget_norm, "budget_c": budget_c,
         "budget_g": budget_g,
         "S_corpus": grf["S_corpus"], "S_graf": grf["S_graf"],
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"] = {GRF_ARM: {
        "desc": ARM_DESC[GRF_ARM], "phase": {
            "traj": grf["traj"], "corpus_ledger": grf["corpus_ledger"],
            "orth_ledger": grf["orth_ledger"],
            "disp_ledger": grf["disp_ledger"],
            "buf_ledger": grf["buf_ledger"],
            "budget_ledger": grf["budget_ledger"],
            "lr_ledger": grf["lr_ledger"],
            "graf_ledger": grf["graf_ledger"],
            "window_ledger": grf["window_ledger"],
            "graisol": grf["graisol"],
            "corpus_ce_median": med(corp_ce_g),
            "corpus_gn_clipped_median": med(corp_gn_g),
            "corpus_g_in_room_frac_median": med(corp_gin_g),
            "orth_max_rel_err": grf["orth_max"],
            "graft_trace": {
                "alpha_max": ALPHA_MAX,
                "per_event_alpha": graf_alphas,
                "per_event_gap_norm": graf_gaps,
                "per_event_edit_norm": graf_edits,
                "per_event_deficit": [m["deficit_t"]
                                      for m in grf["graf_ledger"]],
                "per_event_binder": [m["binder"] for m in grf["graf_ledger"]],
                "n_binder_cap": sum(1 for b in
                                    (m["binder"] for m in grf["graf_ledger"])
                                    if b == "cap"),
                "gap_max": max(graf_gaps) if graf_gaps else None,
                "gap_final": graf_gaps[-1] if graf_gaps else None,
                "S_graf_final": grf["S_graf"],
                "S_graf_of_B_G": grf["S_graf"] / budget_g,
                "note": "THE REGISTERED READ (the engagement ledger): the "
                        "per-event in-room gap ||P_room(fact - theta)|| + "
                        "alpha + the spent edit — P-e296c's discriminator "
                        "(gap ~ fp-floor everywhere = the graft never "
                        "engaged; gap >= 1e-4 anywhere = it engaged)"},
            "budget": {"budget_norm": budget_norm,
                       "budget_c": budget_c, "budget_g": budget_g,
                       "S_corpus_final": grf["S_corpus"],
                       "S_graf_final": grf["S_graf"],
                       "S_total_final": grf["S_corpus"] + grf["S_graf"],
                       "S_total_usage_frac": (grf["S_corpus"]
                                              + grf["S_graf"])
                       / budget_norm,
                       "S_corpus_usage_of_B_C": grf["S_corpus"] / budget_c,
                       "S_graf_usage_of_B_G": grf["S_graf"] / budget_g,
                       "cum_total_final_norm":
                           grf["disp_ledger"][-1]["total_disp_cum_norm"]
                           if grf["disp_ledger"] else None,
                       "cum_total_usage_frac": (
                           grf["disp_ledger"][-1]["total_disp_cum_norm"]
                           / budget_norm
                           if grf["disp_ledger"] else None),
                       "n_capped": grf["n_capped"],
                       "n_steps": PHASE_STEPS,
                       "n_graf": grf["n_graf"],
                       "capped_frac": grf["n_capped"] / PHASE_STEPS,
                       "n_graf_capped": grf.get("n_graf_capped"),
                       "lr_applied_min": grf["lr_applied_min"],
                       "lr_applied_median": grf["lr_applied_median"],
                       "lr_sched_median": grf["lr_sched_median"],
                       "cap_min": grf["cap_min"]},
            "chunk_table": grf["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_g,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_g.double().numpy())), **loads_final_g,
                "in_room_gap_norm": gap_final_g},
            "checkpoint": grf_ck,
            "resumed_final": bool(grf.get("resumed_final", False)),
        }}}
    log(f"ARM-{GRF_ARM} DONE: post g0 {cells_g['g0']:.7f} "
        f"(x{cells_g['g0'] / FACT_BASELINE_G0:.4f}) g-12 {cells_g['gm12']:.7f} "
        f"CE_R {cells_g['ce_r']:.4f} | BUDGET S_corp {grf['S_corpus']:.4f}"
        f"/{budget_c:.4f} ({grf['S_corpus'] / budget_c:.1%} of B_C) + "
        f"S_graf {grf['S_graf']:.6f}/{budget_g:.4f} "
        f"({grf['S_graf'] / budget_g:.1%} of B_G) = "
        f"{grf['S_corpus'] + grf['S_graf']:.4f}/{budget_norm:.4f} "
        f"({(grf['S_corpus'] + grf['S_graf']) / budget_norm:.1%}) | graft "
        f"{grf['n_graf']} steps (gap max "
        f"{(max(graf_gaps) if graf_gaps else float('nan')):.3e}, final "
        f"{(graf_gaps[-1] if graf_gaps else float('nan')):.3e}; alpha med "
        f"{(med(graf_alphas) if graf_alphas else float('nan')):.4f}) | "
        f"capped {grf['n_capped']}/{PHASE_STEPS} | ORTH max "
        f"{grf['orth_max']:.2e} | isolation "
        f"{grf['graisol']['graf_checks']} checks "
        f"{grf['graisol']['graf_violations'] + grf['graisol']['graf_grad_none_violations']} "
        f"violations | corpus CE med {med(corp_ce_g):.4f}")
    write_partial(f"ARM-{GRF_ARM} complete (the prosthetic build's ledgers)")
    # ================= P4: ARM CONTROLLER-TWIN (e288's form) =============
    E261.burst_cooldown(f"{GRF_ARM} -> {TWN_ARM}")
    log("=" * 78)
    log(f"ARM-{TWN_ARM} — {ARM_DESC[TWN_ARM]}")
    twn = chunked_controller_twin_phase(
        f"{TWN_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, budget_c, budget_m, win_i, inst_mask, anchor_full,
        train_ids, g0_ids, gm12_ids, r_eval_xy, zid, lr_m_max,
        CKPT_DIR / ("smoke_e296_CONTROLLER-TWIN_resume.pt" if SMOKE
                    else "e296_CONTROLLER-TWIN_resume.pt"), dev)
    sd_t = twn["sd"]
    net_t = G1.evl_load(sd_t)
    cells_t = {"gm12": G1.battery_cell(net_t, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_t, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_t, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_t, *r_eval_xy)}
    d_final_t = flat_params_cpu(net_t) - fact_flat
    loads_final_t = rooms.displacement_loads(d_final_t, ROOM_MODE)
    del net_t
    corp_ce_t = [v["ce"] for v in twn["corpus_ledger"].values()]
    corp_gn_t = [v["gn_clipped"] for v in twn["corpus_ledger"].values()]
    corp_gin_t = [v["g_in_room_frac"] for v in twn["corpus_ledger"].values()
                  if v.get("g_in_room_frac") is not None]
    maint_lrs_t = [m["lr_maint"] for m in twn["maint_ledger"]]
    maint_def_t = [m["deficit_t"] for m in twn["maint_ledger"]]
    maint_bind_t = [m["binder"] for m in twn["maint_ledger"]]
    twn_ck = save_ckpt(
        "e296_CONTROLLER-TWIN_post", sd_t,
        {"desc": "e296 ARM-CONTROLLER-TWIN post-phase state: the committed "
                 "quiet-formed 10k fact + 400 budgeted orthogonal SGD-M "
                 "corpus steps + 16 NAME-ONLY maintenance steps at e288's "
                 "EXACT error-gated settings (lr_m_t = LR_M_MAX x "
                 f"deficit_t over the split pools; corpus stream "
                 f"{CORPUS_GEN_SEED}, install stream {INST_GEN_SEED}) — "
                 "the signal-borne reference",
         "arm": TWN_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "inst_gen_seed": INST_GEN_SEED,
         "lr_m_max": lr_m_max, "act_every": ACT_EVERY,
         "budget_norm": budget_norm, "budget_c": budget_c,
         "budget_m": budget_m,
         "S_corpus": twn["S_corpus"], "S_maint": twn["S_maint"],
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"][TWN_ARM] = {
        "desc": ARM_DESC[TWN_ARM], "phase": {
            "traj": twn["traj"], "corpus_ledger": twn["corpus_ledger"],
            "orth_ledger": twn["orth_ledger"],
            "disp_ledger": twn["disp_ledger"],
            "buf_ledger": twn["buf_ledger"],
            "budget_ledger": twn["budget_ledger"],
            "lr_ledger": twn["lr_ledger"],
            "maint_ledger": twn["maint_ledger"],
            "window_ledger": twn["window_ledger"],
            "corpus_ce_median": med(corp_ce_t),
            "corpus_gn_clipped_median": med(corp_gn_t),
            "corpus_g_in_room_frac_median": med(corp_gin_t),
            "orth_max_rel_err": twn["orth_max"],
            "bufsep": twn["bufsep"],
            "controller_trace": {
                "lr_m_max": lr_m_max,
                "per_event_deficit": maint_def_t,
                "per_event_lr_applied": maint_lrs_t,
                "per_event_binder": maint_bind_t,
                "n_binder_cap": sum(1 for b in maint_bind_t if b == "cap"),
                "n_binder_controller": sum(1 for b in maint_bind_t
                                            if b == "controller"),
                "deficit_min": min(maint_def_t) if maint_def_t else None,
                "deficit_max": max(maint_def_t) if maint_def_t else None,
                "deficit_median": med(maint_def_t),
                "gate_vs_pre_absdiff_max": max(
                    (m["gate_vs_pre_absdiff"] for m in twn["maint_ledger"]
                     if m.get("gate_vs_pre_absdiff") is not None),
                    default=None),
                "note": "e288's controller trace on this session's stream "
                        "(the reference's own behavior)",
            },
            "budget": {"budget_norm": budget_norm,
                       "budget_c": budget_c, "budget_m": budget_m,
                       "S_corpus_final": twn["S_corpus"],
                       "S_maint_final": twn["S_maint"],
                       "S_total_final": twn["S_corpus"] + twn["S_maint"],
                       "S_total_usage_frac": (twn["S_corpus"]
                                              + twn["S_maint"])
                       / budget_norm,
                       "S_corpus_usage_of_B_C": twn["S_corpus"] / budget_c,
                       "S_maint_usage_of_B_M": twn["S_maint"] / budget_m,
                       "cum_total_final_norm":
                           twn["disp_ledger"][-1]["total_disp_cum_norm"]
                           if twn["disp_ledger"] else None,
                       "n_capped": twn["n_capped"],
                       "n_steps": PHASE_STEPS,
                       "n_maint": twn["n_maint"],
                       "capped_frac": twn["n_capped"] / PHASE_STEPS,
                       "lr_applied_min": twn["lr_applied_min"],
                       "lr_applied_median": twn["lr_applied_median"],
                       "lr_sched_median": twn["lr_sched_median"],
                       "cap_min": twn["cap_min"],
                       "n_maint_capped": twn.get("n_maint_capped"),
                       "maint_lr_applied_median": (med(maint_lrs_t)
                                                   if maint_lrs_t else None),
                       "maint_lr_applied_min": (min(maint_lrs_t)
                                                if maint_lrs_t else None)},
            "chunk_table": twn["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_t,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_t.double().numpy())), **loads_final_t},
            "checkpoint": twn_ck,
            "resumed_final": bool(twn.get("resumed_final", False)),
        }}
    log(f"ARM-{TWN_ARM} DONE: post g0 {cells_t['g0']:.7f} "
        f"(x{cells_t['g0'] / FACT_BASELINE_G0:.4f}; e288's committed "
        f"ERROR-GATED x{E288_REA_RATIO:.4f}) g-12 {cells_t['gm12']:.7f} "
        f"CE_R {cells_t['ce_r']:.4f} | BUDGET S_corp {twn['S_corpus']:.4f} "
        f"+ S_maint {twn['S_maint']:.4f} = "
        f"{twn['S_corpus'] + twn['S_maint']:.4f}/{budget_norm:.4f} "
        f"({(twn['S_corpus'] + twn['S_maint']) / budget_norm:.1%}) | "
        f"maint {twn['n_maint']} steps (deficit med {med(maint_def_t):.4f}, "
        f"lr_m med {(med(maint_lrs_t) if maint_lrs_t else float('nan')):.6f}) "
        f"| corpus CE med {med(corp_ce_t):.4f}")
    write_partial(f"ARM-{TWN_ARM} complete (the signal-borne reference)")

    # ---- the draw-integrity texture check (non-halting) ----------------
    first_ce = {a: metrics["arms"][a]["phase"]["corpus_ledger"].get(
        "1", metrics["arms"][a]["phase"]["corpus_ledger"].get(1, {})
    ).get("ce") for a in ARMS}
    first_gn = {a: metrics["arms"][a]["phase"]["corpus_ledger"].get(
        "1", metrics["arms"][a]["phase"]["corpus_ledger"].get(1, {})
    ).get("gn_clipped") for a in ARMS}
    draw_ok = all(v is not None and abs(v - first_ce[ARMS[0]]) < 1e-9
                  for v in first_ce.values())
    log(f"draw-integrity (non-halting): the t1 corpus CE + clipped ||g|| "
        f"identical across arms = {draw_ok} (ce {first_ce}; gn {first_gn}; "
        f"bit-identical draws — the arms' ONLY delta is the actuator)")

    # ================= P5: the instantiated gates ========================
    # G_ORTH — the corpus orthogonality gate (NON-HALTING)
    orth_max_g = grf["orth_max"]
    G_ORTH = {
        "form": "the stepped CORPUS gradient is ENTIRELY ORTHOGONAL to the "
                "room on BOTH arms: max over ALL corpus steps of "
                "||P_room g_perp|| / ||g_perp|| < 1e-6 (checked EVERY "
                "corpus step). This gate is also P-e296c's premise: an "
                "orthogonalized stream cannot move the room-frame — the "
                "graft's engagement ledger measures whether anything else "
                "does. NON-HALTING (a failure routes the composite to "
                "MIXED)",
        "graft_orth_max_rel_err": orth_max_g,
        "twin_orth_max_rel_err": twn["orth_max"],
        "bar": ORTH_BAR,
        "n_steps_checked": PHASE_STEPS,
        "pass": bool(orth_max_g is not None and orth_max_g < ORTH_BAR
                     and twn["orth_max"] < ORTH_BAR),
    }
    metrics["gates"]["G_ORTH"] = G_ORTH
    if not G_ORTH["pass"]:
        log(f"  G_ORTH FAILED (grf {orth_max_g:.2e} / twn "
            f"{twn['orth_max']:.2e} >= {ORTH_BAR:.0e}) — NON-HALTING: the "
            f"composite routes to MIXED (the gates clause)")

    # G_GRAFISOL — the graft arm's no-signal isolation (HARD)
    G_GRAFISOL = {
        "form": "the graft arm's NO-SIGNAL isolation, machine-verified: "
                "around EVERY graft event (a) opt_C's momentum buffers are "
                "snapshotted bitwise and compared after (the edit must "
                "touch NO optimizer state) and (b) the parameter grads are "
                "snapshotted bitwise and compared after — the corpus "
                "step's grads persist after opt_C.step() (cleared at the "
                "next iteration's zero_grad), so the honest check is "
                "BITWISE UNCHANGED across the edit: the graft path "
                "computes NO gradient, NO loss, NO backward (structural — "
                "the driver's graft block contains neither). Any violation "
                "RAISES (a hard HALT)",
        "graf_checks": grf["graisol"]["graf_checks"],
        "graf_violations": grf["graisol"]["graf_violations"],
        "grad_violations": grf["graisol"]["graf_grad_none_violations"],
        "n_graf": grf["n_graf"],
        "checks_equal_events": bool(
            grf["graisol"]["graf_checks"] == grf["n_graf"]),
        "the_no_signal_ledger": {
            "loss_calls_on_graft_path": 0, "backward_calls_on_graft_path": 0,
            "optimizer_steps_on_graft_path": 0, "data_draws": 0,
            "note": "by construction + machine-checked (grads bitwise "
                    "unchanged through every edit; opt_C buffers "
                    "bitwise-unchanged); the graft arm has NO install "
                    "generator"},
        "pass": bool(grf["graisol"]["graf_violations"] == 0
                     and grf["graisol"]["graf_grad_none_violations"] == 0
                     and grf["graisol"]["graf_checks"] == grf["n_graf"]
                     and grf["n_graf"] == N_ACT_EXPECTED),
    }
    assert G_GRAFISOL["pass"], f"graft isolation gate FAILED: {G_GRAFISOL}"
    metrics["gates"]["G_GRAFISOL"] = G_GRAFISOL

    # G_BUFSEP — the twin's bidirectional separation (HARD, e288 verbatim)
    bufC_fr = [b["bufC_inroom_frac"] for b in grf["buf_ledger"]
               if b.get("bufC_inroom_frac") is not None]
    bufC_fr_t = [b["bufC_inroom_frac"] for b in twn["buf_ledger"]
                 if b.get("bufC_inroom_frac") is not None]
    bufF_fr_t = [b["bufF_inroom_frac"] for b in twn["buf_ledger"]
                 if b.get("bufF_inroom_frac") is not None]
    G_BUFSEP = {
        "form": "the twin's buffer separation, BIDIRECTIONAL (e288 "
                "VERBATIM): around EVERY optimizer event BOTH optimizers' "
                "buffer states are snapshotted bitwise and compared after "
                "(any mismatch RAISES — a hard HALT); COMPOSITION — per "
                "milestone, ||P_room buf_C||/||buf_C|| < 1e-4 on both arms "
                "(NON-HALTING — a failure routes the composite to MIXED); "
                "buf_F's composition DISCLOSED (never a bar). The GRAFT "
                "arm's isolation is G_GRAFISOL's (no fact-side optimizer "
                "exists there)",
        "controller_twin": {
            "corpus_checks": twn["bufsep"]["corpus_checks"],
            "corpus_violations": twn["bufsep"]["corpus_violations"],
            "maint_checks": twn["bufsep"]["maint_checks"],
            "maint_violations": twn["bufsep"]["maint_violations"],
            "optF_steps": twn["bufsep"]["optF_steps"],
            "n_maint": twn["n_maint"],
            "optF_steps_equals_n_maint":
                bool(twn["bufsep"]["optF_steps"] == twn["n_maint"]),
            "bufC_inroom_frac_max": max(bufC_fr_t) if bufC_fr_t else None,
            "bufF_inroom_frac_last": bufF_fr_t[-1] if bufF_fr_t else None,
        },
        "graft_arm_bufC_inroom_frac_max": max(bufC_fr) if bufC_fr else None,
        "bufC_bar": BUFSEP_ORTH_BAR,
        "e284_sep_cite": E284_SEP_PRIMARY,
        "pass": None,           # computed explicitly below (a hard gate)
    }
    G_BUFSEP["pass"] = bool(
        twn["bufsep"]["corpus_violations"] == 0
        and twn["bufsep"]["maint_violations"] == 0
        and twn["bufsep"]["corpus_checks"] == PHASE_STEPS
        and twn["bufsep"]["maint_checks"] == twn["n_maint"]
        and twn["bufsep"]["optF_steps"] == twn["n_maint"]
        and twn["n_maint"] == N_ACT_EXPECTED
        and bufC_fr and max(bufC_fr) < BUFSEP_ORTH_BAR
        and bufC_fr_t and max(bufC_fr_t) < BUFSEP_ORTH_BAR)
    assert G_BUFSEP["pass"], f"buffer separation gate FAILED: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    # G_BUDGET — the TOTAL budget's machine verification (NON-HALTING)
    S_cor_final_g = float(grf["S_corpus"])
    S_g_final = float(grf["S_graf"])
    S_tot_final_g = S_cor_final_g + S_g_final
    cum_total_final_g = (grf["disp_ledger"][-1]["total_disp_cum_norm"]
                         if grf["disp_ledger"] else None)
    S_cor_final_t = float(twn["S_corpus"])
    S_m_final_t = float(twn["S_maint"])
    S_tot_final_t = S_cor_final_t + S_m_final_t
    cum_total_final_t = (twn["disp_ledger"][-1]["total_disp_cum_norm"]
                         if twn["disp_ledger"] else None)
    G_BUDGET = {
        "form": "the TOTAL displacement budget's machine verification "
                "(the bars' standing condition 'the budget <= 100%'): the "
                "GRAFT's S_total = S_corpus + S_graf <= BUDGET + slack "
                "with the per-stream allocations held (S_corpus <= B_C, "
                "S_graf <= B_G) AND the realized total cumulative "
                "displacement <= BUDGET + slack; THE TWIN under the same "
                "split-pool form (S_corpus <= B_C, S_maint <= B_M); every "
                "event's realized displacement fits its reserved share, so "
                "both hold by the triangle bound BY CONSTRUCTION — a "
                "failure here is an implementation outcome to autopsy "
                "(NON-HALTING, routed to MIXED, everything verbatim)",
        "budget_norm": budget_norm,
        "budget_frac_of_write_norm": BUDGET_FRAC,
        "write_norm": write_norm,
        "graft": {
            "budget_c": budget_c, "budget_g": budget_g,
            "S_corpus_final": S_cor_final_g,
            "S_graf_final": S_g_final,
            "S_total_final": S_tot_final_g,
            "S_total_usage_frac": S_tot_final_g / budget_norm,
            "S_corpus_usage_of_B_C": S_cor_final_g / budget_c,
            "S_graf_usage_of_B_G": S_g_final / budget_g,
            "cum_disp_total_final_norm": cum_total_final_g,
            "cum_total_usage_frac": (cum_total_final_g / budget_norm
                                     if cum_total_final_g is not None
                                     else None),
            "n_capped": grf["n_capped"], "n_steps": PHASE_STEPS,
            "capped_frac": grf["n_capped"] / PHASE_STEPS,
            "n_graf_capped": grf.get("n_graf_capped"),
            "lr_price": {"lr_applied_median": grf["lr_applied_median"],
                         "lr_applied_min": grf["lr_applied_min"],
                         "lr_sched_median": grf["lr_sched_median"]}},
        "controller_twin": {
            "budget_c": budget_c, "budget_m": budget_m,
            "S_corpus_final": S_cor_final_t,
            "S_maint_final": S_m_final_t,
            "S_total_final": S_tot_final_t,
            "S_total_usage_frac": S_tot_final_t / budget_norm,
            "S_corpus_usage_of_B_C": S_cor_final_t / budget_c,
            "S_maint_usage_of_B_M": S_m_final_t / budget_m,
            "cum_final_norm": cum_total_final_t,
            "n_capped": twn["n_capped"], "n_maint_capped":
                twn.get("n_maint_capped"),
            "maint_lr_applied_median": (med(maint_lrs_t)
                                        if maint_lrs_t else None)},
        "e288_contrast": {
            "S_corpus_final": E288_S_CORPUS,
            "S_maint_final": E288_S_MAINT,
            "note": "the direct parent's realized 60/40 split (87.2% of "
                    "budget total)"},
        "slack": BUDGET_SLACK,
        "pass": bool(S_tot_final_g <= budget_norm + BUDGET_SLACK
                     and S_cor_final_g <= budget_c + BUDGET_SLACK
                     and S_g_final <= budget_g + BUDGET_SLACK
                     and cum_total_final_g is not None
                     and cum_total_final_g <= budget_norm + BUDGET_SLACK
                     and S_tot_final_t <= budget_norm + BUDGET_SLACK
                     and S_cor_final_t <= budget_c + BUDGET_SLACK
                     and S_m_final_t <= budget_m + BUDGET_SLACK
                     and cum_total_final_t is not None
                     and cum_total_final_t <= budget_norm + BUDGET_SLACK),
    }
    metrics["gates"]["G_BUDGET"] = G_BUDGET
    if not G_BUDGET["pass"]:
        log(f"  G_BUDGET FAILED (graft S_total {S_tot_final_g:.4f} / cum "
            f"{cum_total_final_g} or twin S {S_tot_final_t:.4f} / cum "
            f"{cum_total_final_t} over budget) — NON-HALTING: the "
            f"composite routes to MIXED (the standing conditions broken)")
    log(f"P5 GATES: G_ORTH {'PASS' if G_ORTH['pass'] else 'FAIL'} (graft max "
        f"{orth_max_g:.2e}, twn max {twn['orth_max']:.2e} vs "
        f"{ORTH_BAR:.0e} over {PHASE_STEPS} steps); G_GRAFISOL PASS ("
        f"{G_GRAFISOL['graf_checks']} graft events checked, 0 violations, "
        f"grads bitwise-unchanged throughout); G_BUFSEP PASS (twin "
        f"{twn['bufsep']['corpus_checks']}+{twn['bufsep']['maint_checks']} "
        f"checks, 0 violations; opt_F stepped exactly "
        f"{twn['bufsep']['optF_steps']}x; bufC in-room max "
        f"{(max(bufC_fr) if bufC_fr else float('nan')):.2e}/"
        f"{(max(bufC_fr_t) if bufC_fr_t else float('nan')):.2e}); "
        f"G_BUDGET {'PASS' if G_BUDGET['pass'] else 'FAIL'} (graft S_corp "
        f"{S_cor_final_g:.4f}/{budget_c:.4f} B_C + S_graf {S_g_final:.6f}/"
        f"{budget_g:.4f} B_G = {S_tot_final_g:.4f}/{budget_norm:.4f}; twin "
        f"{S_tot_final_t:.4f}/{budget_norm:.4f})")
    write_partial("P5 gates complete (ORTH + GRAFISOL + BUFSEP + BUDGET)")

    # ================= P6: ADJUDICATION (the frozen bars) ================
    ratio_400 = cells_g["g0"] / FACT_BASELINE_G0
    ratio_session_denom = cells_g["g0"] / max(fact_g0, RATIO_DEN_FLOOR)
    twin_ratio = cells_t["g0"] / FACT_BASELINE_G0
    ratio_mile = {t["step"]: t["g0_pz"] / FACT_BASELINE_G0
                  for t in grf["traj"]}
    twin_ratio_mile = {t["step"]: t["g0_pz"] / FACT_BASELINE_G0
                       for t in twn["traj"]}

    def _stream_live(corpus_ledger):
        ce_steps = sorted(int(k) for k in corpus_ledger.keys())
        early = [corpus_ledger[str(k)]["ce"] if str(k) in
                 corpus_ledger else corpus_ledger[k]["ce"]
                 for k in ce_steps if k <= (8 if SMOKE else 100)]
        late = [corpus_ledger[str(k)]["ce"] if str(k) in
                corpus_ledger else corpus_ledger[k]["ce"]
                for k in ce_steps if k > (3 if SMOKE else 300)]
        e_med, l_med = med(early) if early else None, med(late) if late else None
        improving = bool(e_med is not None and l_med is not None
                         and l_med < e_med)
        stable = bool(e_med is not None and l_med is not None
                      and l_med <= STREAM_STABLE_TOL * e_med)
        return {"early_median": e_med, "late_median": l_med,
                "improving": improving, "stable_1p05": stable,
                "live": bool(improving or stable)}

    stream_g = _stream_live(grf["corpus_ledger"])
    stream_t = _stream_live(twn["corpus_ledger"])

    # the read RESPONSE per graft (the registered read): post-graft read /
    # pre-graft read at the two windows
    graft_response = {}
    for (a, b) in ((WIN1, f"win1_t{WIN1[1]}"), (WIN2, f"win2_t{WIN2[1]}")):
        pre = next((w for w in grf["window_ledger"]
                    if w["step"] == a[1] and "pre" in w["window_phase"]),
                   None)
        post = next((w for w in grf["window_ledger"]
                     if w["step"] == a[1] and "post" in w["window_phase"]),
                    None)
        if pre is not None and post is not None:
            graft_response[b] = {
                "pre_g0": pre["g0_pz"], "post_g0": post["g0_pz"],
                "response_ratio": (post["g0_pz"] / pre["g0_pz"]
                                   if pre["g0_pz"] > 0 else None)}

    def _resp_txt(rr: dict) -> str:
        if not rr:
            return "n/a"
        return "; ".join(
            (f"{k} x{v['response_ratio']:.4f}"
             if v["response_ratio"] is not None else f"{k} n/a")
            for k, v in rr.items())

    # the twin's sawtooth (its own reference read)
    twin_saw = {}
    for (a, b) in ((WIN1, f"win1_t{WIN1[1]}"), (WIN2, f"win2_t{WIN2[1]}")):
        pre = next((w for w in twn["window_ledger"]
                    if w["step"] == a[1] and "pre" in w["window_phase"]),
                   None)
        post = next((w for w in twn["window_ledger"]
                     if w["step"] == a[1] and "post" in w["window_phase"]),
                    None)
        if pre is not None and post is not None:
            twin_saw[b] = {
                "pre_g0": pre["g0_pz"], "post_g0": post["g0_pz"],
                "lift_ratio": (post["g0_pz"] / pre["g0_pz"]
                               if pre["g0_pz"] > 0 else None)}

    hard = dict(metrics["gates"])
    halt_gates_pass = bool(all(
        g.get("pass") for k, g in hard.items()
        if k not in ("G_ORTH", "G_BUDGET")))
    orth_held = bool(G_ORTH["pass"])
    budget_held = bool(G_BUDGET["pass"])
    twin_holds = bool(twin_ratio >= SURVIVE_FRAC and budget_held
                      and stream_t["live"])
    # MIXED-partial engagement floors (frozen at birth)
    engage_s = S_g_final >= GRAFT_ENGAGE_S_FLOOR * budget_g
    slowed = ratio_400 >= GRAFT_SLOW_FLOOR_X * E285_SAN_RATIO
    graf_gap_max = max(graf_gaps) if graf_gaps else None
    # the engagement sentence (P-e296c's reading, precomputed for the
    # clause builders below)
    if engage_s:
        engagement_sentence = (
            f"the graft ENGAGED (S_graf {S_g_final:.6f} >= "
            f"{GRAFT_ENGAGE_S_FLOOR:.0%} x B_G) but did not slow the decay "
            f"to the {GRAFT_SLOW_FLOOR_X:.0f}x-sanctuary floor")
    else:
        engagement_sentence = (
            f"THE GRAFT NEVER ENGAGED: S_graf {S_g_final:.6f} of B_G "
            f"{budget_g:.4f} ({S_g_final / budget_g:.1%}), the in-room gap "
            f"max {graf_gap_max:.3e} over all events — the orthogonalized "
            f"stream pins the room-frame at the fp floor (P-e296c "
            f"confirmed; e285's sanctuary law), so the death is "
            f"OUT-of-room and only the signal's out-of-room brush "
            f"(e288's ~0.06 in-room) reaches it")

    if not halt_gates_pass:
        failed = [k for k, g in hard.items()
                  if k not in ("G_ORTH", "G_BUDGET") and not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard bind/isolation gate failed — nothing adjudicated; "
                  "the record is complete for the autopsy")
    elif not budget_held or not orth_held:
        verdict = "MIXED"
        which = []
        if not budget_held:
            which.append(
                f"a standing condition broken: the budget NOT held (graft "
                f"S_total {S_tot_final_g:.4f} vs BUDGET {budget_norm:.4f} "
                f"+ {BUDGET_SLACK:.0e} slack; realized cum "
                f"{cum_total_final_g}; twin S {S_tot_final_t:.4f} — an "
                f"implementation outcome to autopsy)")
        if not orth_held:
            which.append(f"the ORTHOGONALITY gate broken (graft "
                         f"{orth_max_g:.2e} / twn {twn['orth_max']:.2e} vs "
                         f"{ORTH_BAR:.0e})")
        clause = (f"ratio_400 {ratio_400:.4f}x — MIXED by conditions "
                  f"(no bar covers a broken standing condition): "
                  f"{' AND '.join(which)}; everything verbatim")
    elif ratio_400 >= SURVIVE_FRAC and stream_g["live"]:
        verdict = "GRAFT-HOLDS"
        clause = (f">= {SURVIVE_FRAC:.0%}x survival at t400 (ratio "
                  f"{ratio_400:.4f}x vs the twin's {twin_ratio:.4f}x) with "
                  f"the budget HELD (S_total {S_tot_final_g:.4f} = "
                  f"{S_tot_final_g / budget_norm:.1%} of {budget_norm:.4f}: "
                  f"corpus {S_cor_final_g:.4f} = {S_cor_final_g / budget_c:.1%}"
                  f" of B_C + graft {S_g_final:.6f} = {S_g_final / budget_g:.1%}"
                  f" of B_G) and the stream live (corpus CE "
                  f"{stream_g['early_median']:.4f} -> "
                  f"{stream_g['late_median']:.4f}) — maintenance needs NO "
                  f"signal: memory preservation is geometry refresh; the "
                  f"anesthesia endgame. THE GRAFT TRACE: alpha median "
                  f"{(med(graf_alphas) if graf_alphas else float('nan')):.4f}, "
                  f"gap max {graf_gap_max:.3e}, S_graf {S_g_final:.6f}; the "
                  f"read response per graft {_resp_txt(graft_response)}")
    elif twin_holds:
        if engage_s and slowed:
            verdict = "MIXED"
            clause = (f"partial: the graft ratio_400 {ratio_400:.4f}x < "
                      f"{SURVIVE_FRAC:.0%}x while the twin holds "
                      f"({twin_ratio:.4f}x), and the graft ENGAGED + "
                      f"slowed the decay (S_graf {S_g_final:.6f} >= "
                      f"{GRAFT_ENGAGE_S_FLOOR:.0%} x B_G; ratio >= "
                      f"{GRAFT_SLOW_FLOOR_X:.0f}x e285's sanctuary "
                      f"{E285_SAN_RATIO:.4f}) — geometry buys time, signal "
                      f"buys survival; a two-channel account of "
                      f"maintenance. THE GRAFT TRACE: gap max "
                      f"{graf_gap_max:.3e}, alpha median "
                      f"{(med(graf_alphas) if graf_alphas else float('nan')):.4f}, "
                      f"the read response per graft {_resp_txt(graft_response)}; "
                      f"the milestones "
                      + "; ".join(f"t{t} x{r:.4f}"
                                  for t, r in ratio_mile.items()))
        else:
            verdict = "SIGNAL-NEEDED"
            clause = (f"the graft ratio_400 {ratio_400:.4f}x < "
                      f"{SURVIVE_FRAC:.0%}x while the controller twin "
                      f"HOLDS ({twin_ratio:.4f}x >= 0.5; e288's committed "
                      f"x{E288_REA_RATIO:.4f} — the founding success "
                      f"replicates in-session, n=2) with the budget held "
                      f"(graft {S_tot_final_g:.4f}/{budget_norm:.4f}; twin "
                      f"{S_tot_final_t:.4f}/{budget_norm:.4f}) and both "
                      f"streams live (graft CE {stream_g['early_median']:.4f}"
                      f" -> {stream_g['late_median']:.4f}; twin CE "
                      f"{stream_t['early_median']:.4f} -> "
                      f"{stream_t['late_median']:.4f}) — the teaching "
                      f"signal is load-bearing; maintenance is genuinely "
                      f"re-teaching (and the era's economics deepen). THE "
                      f"GRAFT TRACE (the datum): {engagement_sentence}; "
                      f"the read response per graft "
                      f"{_resp_txt(graft_response)} (the twin's lifts: "
                      + "; ".join(
                          f"x{v['lift_ratio']:.3f}" for v in
                          twin_saw.values()) + "); the milestones "
                      + "; ".join(f"t{t} x{r:.4f}"
                                  for t, r in ratio_mile.items()))
    else:
        verdict = "MIXED"
        clause = (f"ratio_400 {ratio_400:.4f}x (graft) and twin "
                  f"{twin_ratio:.4f}x — the CONTROLLER TWIN ITSELF FAILED "
                  f"TO HOLD (below {SURVIVE_FRAC:.0%}x or its conditions "
                  f"broken: stream live {stream_t['live']}, budget "
                  f"{budget_held}): no clean signal-borne reference; "
                  f"neither GRAFT-HOLDS nor SIGNAL-NEEDED applies; "
                  f"everything verbatim, every ledger, no inflation. THE "
                  f"GRAFT TRACE: gap max {graf_gap_max:.3e}, S_graf "
                  f"{S_g_final:.6f}; the milestones "
                  + "; ".join(f"t{t} x{r:.4f}" for t, r in ratio_mile.items()))

    log("=" * 78)
    log(f"E296 VERDICT: {verdict}")
    log(f"  GRAFT: post g0 {cells_g['g0']:.8f} (x{ratio_400:.4f} "
        f"committed-denom; x{ratio_session_denom:.4f} session-denom)")
    log(f"  CONTROLLER-TWIN: post g0 {cells_t['g0']:.8f} (x{twin_ratio:.4f}; "
        f"e288's committed ERROR-GATED x{E288_REA_RATIO:.4f} — the "
        f"founding success's same-session replicate)")
    log(f"  milestones (GRAFT): "
        + "; ".join(f"t{t} x{r:.4f}" for t, r in ratio_mile.items()))
    log(f"  milestones (TWIN): "
        + "; ".join(f"t{t} x{r:.4f}" for t, r in twin_ratio_mile.items()))
    log(f"  the graft trace: per-event gap "
        + ", ".join(f"{g:.2e}" for g in graf_gaps)
        + f" | alpha median {(med(graf_alphas) if graf_alphas else float('nan')):.4f}"
        f" | S_graf {S_g_final:.6f} of B_G {budget_g:.4f} "
        f"({S_g_final / budget_g:.1%})")
    log(f"  the budget: graft S_corp {S_cor_final_g:.4f}/{budget_c:.4f} B_C "
        f"+ S_graf {S_g_final:.6f}/{budget_g:.4f} B_G = {S_tot_final_g:.4f}/"
        f"{budget_norm:.4f}; twin S_corp {S_cor_final_t:.4f} + S_maint "
        f"{S_m_final_t:.4f} = {S_tot_final_t:.4f}/{budget_norm:.4f}")
    log(f"  the read response per graft: {_resp_txt(graft_response)} (the "
        f"twin's lifts: "
        + "; ".join(f"{k}: x{v['lift_ratio']:.3f}"
                    for k, v in twin_saw.items()) + ")")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any bind/isolation hard-gate "
                           "failure — HALT) -> MIXED-by-conditions "
                           "(G_BUDGET blown OR G_ORTH failed, any ratio) "
                           "-> GRAFT-HOLDS (graft ratio_400 >= 0.5 AND "
                           "budget held AND the graft arm's stream live) "
                           "-> [twin holds] MIXED-partial (S_graf >= 0.01 x "
                           "B_G AND ratio >= 10 x e285's sanctuary) / "
                           "SIGNAL-NEEDED (else) -> MIXED (the twin itself "
                           "failed to hold) — frozen at birth",
        "gates_pass": bool(all(g.get("pass") for g in hard.values())),
        "halt_gates_pass": halt_gates_pass,
        "reads": {
            GRF_ARM: {"post_g0": cells_g["g0"],
                      "survival_ratio_committed_denom": ratio_400,
                      "survival_ratio_session_denom": ratio_session_denom,
                      "post_gm12": cells_g["gm12"],
                      "post_ce_r": cells_g["ce_r"],
                      "traj_g0": {t["step"]: t["g0_pz"]
                                  for t in grf["traj"]},
                      "survival_ratio_milestones": ratio_mile,
                      "corpus_ce_median": med(corp_ce_g),
                      "corpus_ce_early_median": stream_g["early_median"],
                      "corpus_ce_late_median": stream_g["late_median"],
                      "ce_improving": stream_g["improving"],
                      "ce_stable_1p05": stream_g["stable_1p05"],
                      "stream_live": stream_g["live"],
                      "corpus_g_in_room_frac_median": med(corp_gin_g),
                      "graft_trace": metrics["arms"][GRF_ARM][
                          "phase"]["graft_trace"],
                      "graft_read_response": graft_response,
                      "drift_ledger": grf["disp_ledger"],
                      "budget_ledger": grf["budget_ledger"],
                      "buf_ledger": grf["buf_ledger"],
                      "graf_ledger": grf["graf_ledger"],
                      "window_ledger": grf["window_ledger"],
                      "orth_ledger_max": orth_max_g,
                      "drift_from_fact_final_in_room":
                          loads_final_g["in_own_room"],
                      "in_room_gap_final": gap_final_g,
                      "engagement": {"S_graf_of_B_G": S_g_final / budget_g,
                                     "engage_floor_hit": bool(engage_s),
                                     "gap_max": graf_gap_max,
                                     "slow_floor_hit": bool(slowed)},
                      },
            TWN_ARM: {"post_g0": cells_t["g0"],
                      "survival_ratio": twin_ratio,
                      "post_gm12": cells_t["gm12"],
                      "post_ce_r": cells_t["ce_r"],
                      "traj_g0": {t["step"]: t["g0_pz"]
                                  for t in twn["traj"]},
                      "survival_ratio_milestones": twin_ratio_mile,
                      "corpus_ce_median": med(corp_ce_t),
                      "stream_live": stream_t["live"],
                      "maint_ledger": twn["maint_ledger"],
                      "window_ledger": twn["window_ledger"],
                      "budget_ledger": twn["budget_ledger"],
                      "maint_lr_applied_median": (med(maint_lrs_t)
                                                  if maint_lrs_t else None),
                      "maint_g_in_room_frac_median": med(
                          [m["g_in_room_frac"] for m in twn["maint_ledger"]
                           if m.get("g_in_room_frac") is not None]),
                      "drift_ledger": twn["disp_ledger"],
                      "drift_from_fact_final_norm":
                          float(np.linalg.norm(d_final_t.double().numpy())),
                      "drift_from_fact_final_in_room":
                          loads_final_t["in_own_room"],
                      "twin_holds": twin_holds,
                      "note": "e288's ERROR-GATED form on this session's "
                              "shared stream — the live signal-borne "
                              "reference; e288's committed curve cited "
                              "(md5-bound)"},
            "the_cited_references": {
                "e288_error_gated": {
                    "post_g0": E288_REA_POST, "ratio": E288_REA_RATIO,
                    "traj_g0": E288_TRAJ_G0,
                    "S_corpus_final": E288_S_CORPUS,
                    "S_maint_final": E288_S_MAINT,
                    "win1_lift": E288_WIN1_LIFT, "win2_lift": E288_WIN2_LIFT,
                    "maint_in_room_frac_median": E288_MAINT_INROOM_MEDIAN,
                    "note": "the twin's committed record (md5-bound in "
                            "G_PARENTS): the founding success"},
                "e285_sanctuary": {
                    "post_g0": E285_SAN_POST, "ratio": E285_SAN_RATIO,
                    "drift": E285_SAN_DRIFT,
                    "drift_in_room": E285_SAN_DRIFT_INROOM,
                    "S_final": E285_S_FINAL,
                    "note": "the orthogonal-passive death (the graft arm's "
                            "structural twin under P-e296c; the "
                            "MIXED-partial slow floor's reference)"},
                "e287_name_maintained": {
                    "post_g0": E287_REA_POST, "ratio": E287_REA_RATIO,
                    "note": "the third form's curve (the family's "
                            "mid-reference)"},
            },
            "the_two_gates": {
                "orthogonality": {
                    "graft_max": orth_max_g,
                    "twin_max": twn["orth_max"], "bar": ORTH_BAR,
                    "verdict": ("HELD" if orth_held else "BROKEN")},
                "budget": {
                    "budget_norm": budget_norm,
                    "graft_S": S_tot_final_g, "twin_S": S_tot_final_t,
                    "verdict": ("HELD" if budget_held else "BROKEN")}},
        },
        "scatter_disclosure": {
            "read_determinism": "G_FACTLOAD's measured delta co-reported "
                                "(cross-session |d post g0| ~ 5e-7-1e-6 on a "
                                "bit-identical artifact — the family law)",
            "the_two_denominators": {"committed": FACT_BASELINE_G0,
                                     "session_loaded": fact_g0,
                                     "ratios_agree_to":
                                         abs(ratio_400
                                             - ratio_session_denom)},
            "twin_stream_note": "the twin shares THIS cell's stream 29601 "
                                "(the family's own-draw-stream convention): "
                                "it replicates e288's FORM not its bits; "
                                "e288's committed curve is the cross-session "
                                "reference (cited, md5-bound)",
            "n1_caveat": "n=1 per arm, one lineage, one session (the "
                         "g-series standing lottery note carried verbatim)",
        },
        "draw_integrity_first_batch": {"pass": bool(draw_ok),
                                       "corpus_ce": first_ce,
                                       "corpus_gn": first_gn},
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P6 ADJUDICATED (the frozen bars)")
    # ================= P7: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the two arms share ONE loaded artifact (the committed "
            "quiet-formed 10k write, md5/flat-md5/behavior-gated), the "
            "same milestone cadence, the same reads, and BIT-IDENTICAL "
            "corpus draws (one generator, seed 29601, one draw order) — "
            "the ONLY delta is the ACTUATOR: the graft's direct geometric "
            "transplant vs the twin's name-signal gradient step, both "
            "deficit-gated on the same battery and held to the same 40% "
            "pool. The rooms are identical across arms BY CONSTRUCTION "
            "(one bit-gated room)."),
        "the_graft_disclosure": (
            "the graft path uses NO loss, NO gradient, NO optimizer, NO "
            "data draws — machine-checked (grads bitwise-unchanged "
            "through every edit; "
            "opt_C's buffers bitwise-unchanged around every edit; the "
            "graft arm has no install generator); the ONLY things the "
            "dose sees are the read (the gate — the same gate the "
            "controller uses) and the frozen geometry (the loaded fact, "
            "never updated). The transplant arithmetic is verified "
            "machine-live at every event + fp64-exact in G_TRANSPLANT."),
        "the_engagement_disclosure": (
            "P-e296c, registered BEFORE compute: under the era's "
            "orthogonalized corpus stream the in-room gap is pinned at "
            "the fp floor (e285's sanctuary law), so the graft's "
            "displacement is bounded by ~alpha x fp-floor and the graft "
            "may be structurally UNABLE to engage. The bars adjudicate "
            "VERBATIM regardless; the graft trace's gap column is the "
            "registered discriminator between 'geometry cannot maintain' "
            "and 'geometry was never the thing that broke' — the verdict "
            "clause carries whichever reading the trace supports, and the "
            "MIXED-partial branch requires measured engagement + "
            "measured slowing (never assumed)."),
        "the_budget_disclosure": (
            "B_G and B_M are the SAME 40% allocation given to different "
            "actuators (e288's split-pool machinery verbatim on both "
            "arms); every event's realized displacement fits its reserved "
            "share, so S_total <= BUDGET by the triangle bound BY "
            "CONSTRUCTION; a blown G_BUDGET would be an implementation "
            "outcome to autopsy, routed to MIXED, never a HALT"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; nothing guaranteed"),
        "loads_measured_not_nominal": (
            "every read is measured: the per-event in-room gap + edit "
            "norm (the graft ledger), the per-event realized displacement "
            "(corpus and actuator separately), the S split, the "
            "per-milestone interval + cumulative projections (fp64), the "
            "buffer-composition ledgers, the orthogonality ledger (every "
            "step), the corpus CE + clipped-grad ledgers, the "
            "between-steps window reads — never nominal"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "branches and the trajectories are "
                               "reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "the_fact": {"file": f"runs/checkpoints/{FACT_CK}",
                         "md5": FACT_MD5,
                         "loaded_flat_md5": fact_flat_md5,
                         "note": "read-only here — the established fact "
                                 "AND the graft's frozen external copy"},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
            "rooms": rooms_ck,
            "post_states": {GRF_ARM: grf_ck, TWN_ARM: twn_ck},
        },
        "machinery": {
            "graf_phase": "THIS file's chunked_graf_phase (the "
                          "prosthetic arm): e288's corpus driver with the "
                          "actuator block REPLACED by the direct parameter "
                          "edit (the frozen-copy transplant at the "
                          "deficit-gated alpha) + the live transplant-"
                          "arithmetic asserts + the no-signal isolation + "
                          "the total-budget accounting + the window reads "
                          "+ the per-event graft trace",
            "twin_phase": "THIS file's chunked_controller_twin_phase: "
                          "e288's chunked_errorgated_phase VERBATIM in "
                          "body (the signal-borne reference)",
            "cons": "NONE (registered deviation — the bars read the WRITE "
                    "and the DISPLACEMENT only; T259/e281)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": _envelope_summary(),
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }

    # ================= P8: figures ======================================
    make_graf_plot(RD, metrics, verdict, clause, thermal_log,
                   budget_norm, budget_c, budget_g, budget_m,
                   write_norm)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e296_prosthetic_graft.png"),
                          str(RD / "REPORT.md")]
    write_partial("P8 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_graf_plot(rd, metrics, verdict, clause, thermal_log,
                   budget_norm, budget_c, budget_g, budget_m,
                   write_norm):
    """THE CELL'S HEADLINE FIGURE: the survival curves (the graft vs the
    controller twin vs e288's CITED committed curve + the bars), THE GRAFT
    TRACE (the registered read: the per-event in-room gap + alpha + the
    spent edit), the budget split, the two actuators' in-room ledgers (the
    engagement picture), the corpus CE + the twin's lr, the thermal
    envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    grf = metrics["arms"][GRF_ARM]["phase"]
    twn = metrics["arms"][TWN_ARM]["phase"]
    gt_, tt_ = grf["traj"], twn["traj"]

    # (0,0) THE SURVIVAL CURVES (+ the cited references)
    ax = axes[0, 0]
    ax.plot([t["step"] for t in gt_], [max(t["g0_pz"], 1e-7) for t in gt_],
            "o-", lw=2.0, ms=5.5, color="tab:red",
            label=f"{GRF_ARM} (the transplant: alpha x P_room(fact - theta))")
    ax.plot([t["step"] for t in tt_], [max(t["g0_pz"], 1e-7) for t in tt_],
            "s--", lw=1.6, ms=4.5, color="tab:green",
            label=f"{TWN_ARM} (e288's error-gated controller, this session)")
    e288_steps = sorted(int(k) for k in E288_TRAJ_G0.keys())
    ax.plot(e288_steps,
            [max(E288_TRAJ_G0[k], 1e-7) for k in e288_steps],
            "^:", lw=1.5, ms=5.5, color="tab:purple",
            label=f"e288's committed ERROR-GATED (CITED, md5-bound; "
                  f"x{E288_REA_RATIO:.4f})")
    ax.axhline(FACT_BASELINE_G0, color="black", ls=":", lw=1.3,
               label=f"the loaded fact {FACT_BASELINE_G0:.4f}")
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.4, label=f"the {SURVIVE_FRAC:.0%}x HOLDS bar")
    ax.axhline(E285_SAN_RATIO * FACT_BASELINE_G0, color="darkorange",
               ls="--", lw=1.1,
               label=f"e285's sanctuary x{E285_SAN_RATIO:.4f} (the "
                     f"orthogonal-passive class)")
    ax.set_yscale("log")
    ax.set_xlabel(f"corpus step t (actuator fires every {ACT_EVERY}; "
                  "reads POST-actuator)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE SURVIVAL CURVES — the graft vs the signal (bit-"
                 "identical draws, same budget, same gate)", fontsize=9.0)
    ax.legend(fontsize=6.4)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE GRAFT TRACE (the registered read) ----------------------
    ax = axes[0, 1]
    gl = grf["graf_ledger"]
    if gl:
        xs = [m["step"] for m in gl]
        ax.plot(xs, [m["deficit_t"] for m in gl], "o-", lw=1.8, ms=6,
                color="tab:red", label="deficit_t (the gate's reading)")
        ax.plot(xs, [m["alpha"] for m in gl], "D-", lw=1.4, ms=5,
                color="tab:blue",
                label="applied alpha (the transplant fraction)")
        for m in gl:
            if m.get("binder") == "cap":
                ax.annotate("CAP", (m["step"], m["alpha"]), fontsize=6.0,
                            xytext=(2, -11), textcoords="offset points",
                            color="tab:gray")
        ax.axhline(1.0, color="black", ls=":", lw=0.9)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel(f"corpus step t (one graft event every {ACT_EVERY})")
        ax.set_ylabel("deficit | alpha (dimensionless)")
        ax2 = ax.twinx()
        ax2.semilogy(xs, [max(m["gap_norm_pre"], 1e-18) for m in gl],
                     "v-", lw=1.2, ms=4.5, color="tab:orange",
                     label="in-room gap ||P_room(fact-theta)||")
        ax2.semilogy(xs, [max(m["edit_norm"], 1e-18) for m in gl],
                     "^:", lw=1.2, ms=4.5, color="tab:brown",
                     label="edit spent (alpha x gap)")
        ax2.set_ylabel("in-room gap | edit norm (log)", fontsize=8.5)
        ax2.axhline(1e-4, color="crimson", ls="--", lw=0.9)
        ax2.annotate("engagement floor 1e-4", (xs[0], 1e-4), fontsize=6.0,
                     color="crimson", xytext=(3, 3),
                     textcoords="offset points")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.4)
    ax.set_title("THE GRAFT TRACE — the dose tracks the deficit; the gap "
                 "ledger is P-e296c's discriminator", fontsize=9.0)
    ax.grid(alpha=0.25)

    # (0,2) THE BUDGET SPLIT (S_corpus + S_graf vs B_C/B_G/BUDGET)
    ax = axes[0, 2]
    bl = grf["budget_ledger"]
    if bl:
        ax.plot([b["step"] for b in bl], [b["S_corpus"] for b in bl], "o-",
                lw=1.7, ms=4.5, color="tab:cyan",
                label="S_corpus (the B_C pool)")
        ax.plot([b["step"] for b in bl], [b["S_graf"] for b in bl], "^-",
                lw=1.9, ms=5, color="tab:orange",
                label="S_graf (the B_G pool — the graft's stream)")
        ax.plot([b["step"] for b in bl], [b["S_total"] for b in bl], "s-",
                lw=1.6, ms=4, color="tab:red", label="S_total (graft arm)")
    tl = twn["budget_ledger"]
    if tl:
        ax.plot([b["step"] for b in tl],
                [b["S_corpus"] + b["S_maint"] for b in tl], "v--",
                lw=1.2, ms=3.5, color="tab:green", alpha=0.7,
                label="the twin's S_total (S_corpus + S_maint)")
    ax.axhline(budget_norm, color="crimson", ls="--", lw=1.5,
               label=f"THE BUDGET {BUDGET_FRAC}x write = {budget_norm:.3f}")
    ax.axhline(budget_c, color="tab:cyan", ls=":", lw=1.0,
               label=f"B_C = 60% = {budget_c:.3f}")
    ax.axhline(budget_g, color="tab:orange", ls=":", lw=1.0,
               label=f"B_G = B_M = 40% = {budget_g:.3f}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("cumulative displacement (L2)")
    ax.set_title("THE BUDGET — the SAME 40% given to two actuators",
                 fontsize=9.0)
    ax.legend(fontsize=6.3)
    ax.grid(alpha=0.25)

    # (1,0) THE TWO ACTUATORS' DIRECTIONS (in-room shares + the gap)
    ax = axes[1, 0]
    dl_g = {int(d["step"]): d for d in grf["disp_ledger"]}
    dl_t = {int(d["step"]): d for d in twn["disp_ledger"]}

    def _fl(v, floor=1e-12):
        return max(v, floor) if v is not None else floor
    steps_g = sorted(dl_g)
    ax.semilogy(steps_g,
                [_fl(dl_g[s]["in_room_gap_norm"]) for s in steps_g], "*-",
                lw=1.8, ms=8, color="tab:red",
                label="GRAFT: the in-room gap (P-e296c's ledger)")
    ax.semilogy(steps_g,
                [_fl(dl_g[s]["graf_disp_cum_in_room_frac"])
                 for s in steps_g], "^-", lw=1.4, ms=5,
                color="tab:orange", label="GRAFT cum edit in-room (=1 by "
                "construction)")
    steps_t = sorted(dl_t)
    ax.semilogy(steps_t,
                [_fl(dl_t[s]["maint_disp_cum_in_room_frac"])
                 for s in steps_t], "s--", lw=1.4, ms=4.5,
                color="tab:green", alpha=0.85,
                label="TWIN maint cum in-room (the brush, ~0.06)")
    ax.axhline(BUFSEP_ORTH_BAR, color="crimson", ls="--", lw=1.0,
               label=f"the fp-floor bar {BUFSEP_ORTH_BAR:.0e}")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.1, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(LADDER[0][0] / 2739072):.4f}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("in-room share / gap norm (log)")
    ax.set_title("THE TWO STROKES — the graft's IN-room vs the signal's "
                 "OUT-of-room (e313's brush contrast)", fontsize=9.0)
    ax.legend(fontsize=6.4)
    ax.grid(alpha=0.25, which="both")

    # (1,1) THE CORPUS CE + THE TWIN'S DOSE
    ax = axes[1, 1]
    for a, col, mk in ((GRF_ARM, "tab:red", "o"), (TWN_ARM, "tab:green",
                                                   "s")):
        cl = metrics["arms"][a]["phase"]["corpus_ledger"]
        steps = sorted(int(k) for k in cl.keys())

        def _ce(k):
            return cl[str(k)]["ce"] if str(k) in cl else cl[k]["ce"]
        ax.plot(steps, [_ce(k) for k in steps], mk + "-", lw=1.3, ms=3.5,
                color=col, label=f"{a} corpus CE")
    ax.set_xlabel("corpus step t")
    ax.set_ylabel("corpus batch CE (nats)")
    ax3 = ax.twinx()
    ml_steps = [m["step"] for m in twn["maint_ledger"]]
    if ml_steps:
        ax3.plot(ml_steps, [m["lr_maint"] for m in twn["maint_ledger"]],
                 "D-", ms=5, lw=1.2, color="tab:blue",
                 label="TWIN lr_m (the signal's dose)")
        ax3.axhline(LR_M_MAX_FROZEN, color="tab:blue", ls=":", lw=1.1,
                    label=f"LR_M_MAX {LR_M_MAX_FROZEN:.5f} (deficit 1)")
    ax3.set_ylabel("the twin's lr_m", fontsize=8.5)
    ax.set_title("THE ECONOMICS — both streams live; the twin's dose "
                 "tracks its deficit", fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.3)
    ax.grid(alpha=0.25)

    # (1,2) THE THERMAL ENVELOPE + THE ENDPOINT READS
    ax = axes[1, 2]
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
    ax.legend(fontsize=7.0, loc="center left")
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C) + the endpoint "
                 f"reads: GRAFT x"
                 f"{metrics['arms'][GRF_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f} "
                 f"vs TWIN x"
                 f"{metrics['arms'][TWN_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f} "
                 f"vs e288 x{E288_REA_RATIO:.4f}", fontsize=8.2)

    fig.suptitle(f"E296 — THE PROSTHETIC GRAFT (maintenance with NO "
                 f"teaching signal; the wild queue's #1) -> {verdict}",
                 fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e296_prosthetic_graft.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
