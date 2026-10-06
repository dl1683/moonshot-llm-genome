"""E287 — THE NAME-WEIGHTED MAINTENANCE CELL — the build lane's third
iteration: the lab's first injection of the NAME SIGNAL ITSELF as the
maintenance mechanism. This docstring carries the registered question +
arms + bars VERBATIM from the dispatch letter + BOTH changed conventions
(the name-only CE's exact construction + the budget-fitting maintenance
lr), committed at birth BEFORE any compute. Adjudicate against exactly
this; no bar shopping.

THE CONTEXT (e286's committed record, hard-bound below): the first
active-maintenance form FAILED honestly — the budget blown 130.2% (the
maintenance momentum ALONE ate 89.5%), the corpus lr frozen to 0, the
read dead at 0.0180x — BUT the mechanism's signature was machine-
measured: win1's maintenance step LIFTED the read x2.203 (t25-pre
0.003178 -> t25-post 0.007001, held at t26), win2's CRASHED x0.377 (the
effect inverts with drift), and the maintenance gradient's in-room
fraction sat flat 0.0597-0.0612 == the corpus gradient's ~0.060 — the
obstacle was DIRECTIONAL.

THE BIRTH-DAY CORRECTION (material, disclosed at the top): porting the
rig exposed a MISBIND in e286's maintenance batch — its driver call
bound anchor_full (the original-host anchor bank) into the inst_x slot,
so the masked positions of e286's "install windows" held the HOST
OPENINGS ('ELIZABE'/'FLORIZE'), NOT ZEPHYRA — 0/60 windows carried the
name (decode-verified this session: win_i's masked positions hold
ZEPHYRA 60/60; win_i[:, :PRE] is bit-equal to anchor_full[:, :PRE]).
The consequences, stated honestly: e286's "name_ce" 0.0615 was the CE
on memorized host-opening text; its maintenance gradient carried ZERO
name content (the in-room fraction matched the corpus gradient because
the step WAS a corpus-text step); and win1's x2.203 lift came from a
host-text batch. THE NAME SIGNAL HAS NEVER BEEN INJECTED BEFORE THIS
CELL. The dispatch's premise ("the name signal rides ~1% of the mixed
install gradient's mass") is superseded by the stronger measured fact
(0%) — and e287's design is unchanged but MORE first: the pure name
signal's first injection, its geometry (in-room fraction) and its
effect on the read both genuinely unmeasured until now.

THE ARMS (the established quiet-formed 10k fact, loaded bit-exact; the
corpus stream 1:1; SGD-M at the stable lr):
  (a) NAME-MAINTAINED (the build, the third form): the e286 rig with
      TWO changed dials — (i) the maintenance gradient is a NAME-ONLY/
      name-weighted CE: the maintenance loss computed on the NAME
      TOKENS ONLY (the 112 name positions; the pure name signal, no
      corpus dilution); (ii) the maintenance lr is BUDGET-FITTING: the
      maintenance momentum's accumulated displacement is capped to fit
      WITHIN the remaining budget — everything else kept: M=25 cadence,
      the fact-side buffer, the orthogonal corpus projection, the
      corpus-side cap. 400 corpus steps, 16 maintenance steps.
  (b) SANCTUARY-TWIN (the control): the same-session passive control,
      e285's form.

CONVENTION (i) — THE NAME-ONLY CE (frozen HERE at birth): the
maintenance batch := ix(16) TRUE install windows drawn from THIS cell's
registered install generator (win_i — e261's build_win VERBATIM: 130
pre-context tokens | ZEPHYRA | 119 post tokens; G_NAMEWIN-gated: the
masked positions decode to ZEPHYRA in ALL 60 windows, the pre-context
bit-equal to the anchor bank's); forward the 16 windows; token-level
CE; select the masked NAME positions (7 x 16 = 112); loss := their
MEAN. The corpus half of e261's Dmix batch (aj anchors + rj random
windows) is DROPPED entirely — no corpus tokens anywhere in the
maintenance loss (the pure name signal). backward -> clip 1.0 ->
UNPROJECTED (the fact's own stream may write in-room — that IS the
re-anchoring; the corpus orthogonality applies to corpus steps only,
disclosed) -> through opt_F (the FACT-side SGD-M, private buffer).

CONVENTION (ii) — THE BUDGET-FITTING MAINTENANCE LR (frozen HERE at
birth, the momentum-capped form): the SCHEDULE ceiling stays e286's
denomination VERBATIM — MAINT_LR_BASE = LR_SGD_matched/100 =
0.21738574801453703 (== LR_STABLE; e273's md5-bound SGD-denominated
install formation lr; the applying optimizer's own class; the
AdamW-literal 1e-5 fork stays named + rejected as a guaranteed-null).
The APPLIED lr at every maintenance event is lr_m = min(MAINT_LR_BASE,
cap_m) with cap_m = share_m / ||b_m||, where b_m = 0.9*buf_F + g_name
is the EXACT pending in-step momentum (pending_buffer_sqnorm — the
event's displacement price is lr_m x ||b_m||) and share_m = (BUDGET -
S_total)/(1 + remaining corpus steps + later maintenance events) — the
SAME equal-share reservation the corpus cap uses (e286's corpus-cap
convention extended to the maintenance side, symmetric). THE TRIANGLE
BOUND: every optimizer event's realized displacement fits its reserved
share of the remaining budget, so S_total = S_corpus + S_maint <=
BUDGET (+ fp slack) BY CONSTRUCTION — e286's 89.5% blowout (16 steps
moved 4.107) is structurally excluded (16 steps x ~share ~0.011 =
~0.18 ~= 3.9% of budget). The arithmetic consequence, disclosed BEFORE
compute: the third form's dose is ~1/15 of e286's first step (0.162)
per maintenance step — the bars cover the too-small-dose outcome
honestly (SAWTOOTH-CONFIRMED / FAILS report it; the direction read —
the name-only gradient's in-room fraction — is dose-independent).

READS: post g0 at t100/200/300/400; the between-steps windows (>= 2,
both directions — HERE the same two as e286: t24/t25-pre/t25-post/t26
AND t199/t200-pre/t200-post/t201, direct comparability with the
parent's window ledger); the maintenance gradient's in-room fraction
PER STEP (expect >> 0.06 now — the pure name signal's geometry; e286's
misbound steps sat AT the corpus ~0.06); the budget split (S_corpus +
S_maint per milestone); the corpus CE.

FROZEN BARS (survival ratio = post g0 / the loaded baseline
0.26464763283729553):
  - NAME-MAINTAINED-HOLDS: ">= 0.5x at t400 with the budget HELD (<=
    100%) and the stream live (the corpus CE improving or stable) —
    THE FIRST ACTIVE SURVIVAL: a memory maintained by its own name
    signal under traffic."
  - SAWTOOTH-CONFIRMED: "0.05x-0.5x with the windows showing
    consistent LIFTS (every sampled window x>1) — the mechanism real,
    the dosage still short; the trajectory + windows verbatim."
  - FAILS: "< 0.05x or the budget blown again or the lifts absent —
    the trajectories verbatim."
  - MIXED: "anything else."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE PHASE := 400 CORPUS steps + 16 maintenance steps interleaved
    (NAME-MAINTAINED) / 400 corpus steps (SANCTUARY-TWIN); milestones
    t = 100/200/300/400 corpus steps, read POST-maintenance (every
    milestone a multiple of 25; t400 includes the 16th step).
  * THE VEHICLE := the committed QUIET-FORMED 10k fact (e261's serial
    cut completed by e264), loaded BIT-EXACT + gated THREE ways
    (G_FACTLOAD); the room := the fact's OWN committed K10K room
    (seeds 26113/26114), rebuilt + bit-gated vs e264_rooms.pt
    (G_ROOMK10K); THE MAINTENANCE WINDOWS := win_i (G_NAMEWIN — the
    e286 misbind's correction, decode-gated).
  * THE CORPUS STEP := e268's registered form VERBATIM on THIS cell's
    ONE fresh registered generator seed 28701 (the family's per-cell
    rule: .../28501/28601/HERE 28701); BOTH arms draw the IDENTICAL
    sequence — the maintenance mechanism the arms' ONLY delta.
  * "THE STREAM LIVE (the corpus CE improving or stable)" := median of
    the NAME-MAINTAINED arm's logged corpus-batch CE rows t in
    (300,400] vs median of rows t in [1,100]; improving := late <
    early; stable := late <= 1.05 x early (the 5% tolerance frozen
    here); live := improving OR stable.
  * "EVERY SAMPLED WINDOW x>1" := BOTH window ledgers' lift ratios
    (post/pre at win1 AND win2) > 1 — "consistent lifts".
  * THE COMPOSITE := TEXTURE (any bind/isolation hard-gate failure —
    nothing adjudicated, HALT) -> FAILS-by-gates (G_BUDGET blown OR
    G_ORTH failed, any ratio — "the budget blown again") ->
    NAME-MAINTAINED-HOLDS (ratio_400 >= 0.5 AND budget held AND stream
    live) -> MIXED-if-stream-dead (ratio_400 >= 0.5 AND budget held
    AND NOT stream live) -> SAWTOOTH-CONFIRMED (0.05 <= ratio_400 <
    0.5 AND budget held AND every window lift > 1) -> FAILS-by-lifts
    (0.05 <= ratio_400 < 0.5 AND budget held AND a lift absent) ->
    FAILS-by-ratio (ratio_400 < 0.05 AND gates held) -> MIXED
    (everything else).
  * HARD GATES (a failure HALTS): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_PARENTS, G_BASE, G_ROOT,
    G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN,
    G_LR_BIND, G_MAINTBIND, G_BUFSEP-isolation}; NON-HALTING (route to
    FAILS): {G_ORTH, G_BUDGET, bufC-composition} — "the budget blown
    again" adjudicates, never HALTs.
  * NO CONS (T259/e281; e278-e286's committed form): the frozen bars
    read the WRITE and the DISPLACEMENT only; both arms' post-phase
    states CHECKPOINTED for any later landing pass.

REGISTERED PREDICTIONS (from the dispatch letter's framing + the
birth-day misbind finding, cited not new):
  - P-e287a (the name-geometry reading): the pure name gradient points
    substantially IN-ROOM (>> 0.06 — the signal that WROTE the fact,
    which itself stands 94.4% in-room): the maintenance steps lift the
    read at the windows and the third form holds >= 0.05x with the
    budget held — SAWTOOTH-CONFIRMED or better.
  - P-e287b (the saturation reading): the fact already KNOWS the name
    (name CE ~0.2-0.3 nats, concentrated in the first char where the
    battery reads p(Z) ~ 0.26): the name-only gradient may be tiny and
    first-char-local — under the budget-fitting cap the dose (~0.011/
    step) may sit below the read's response threshold; the windows
    flat -> FAILS with the direction read still the datum.
  - P-e287c (the corrected-skeptic reading): e286's lift came from a
    HOST-TEXT batch (the misbind); the name signal's effect on the
    read is genuinely unmeasured — ANY outcome is informative, and the
    in-room fraction + the window shape are the discriminating reads.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier);
bursts <= 175s (inside the dispatch's 180s), per-step thermal polls at
a 78C margin, 40s cooldowns (the 30-60s window), the 84C never-past
line (inside the dispatch's 85C), polls persisted to
runs/_envelope_log.jsonl tagged e287:<ARM>:<phase>; CPU fp64 dense
projections; CPU probing threads 4; NO concurrent GPU jobs (the arms
run sequentially with cooldowns between). TIMESTAMPS: datetime.now(UTC)
only.

Outputs: runs/e287/{metrics.json (PROGRESSIVE), e287_name_maintained.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e287_*.pt (gitignored; md5s in metrics). No NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit +
push per phase.

Run:  cd lab && python e287_name_maintained.py    (E287_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E287_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e287_smoke" if SMOKE else "e287"
assert torch.cuda.is_available(), "e287 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268/e273/e278/e283/e284/e285's disclosed convention):
# e261's drivers resolve their module globals AT CALL TIME through e261's
# module namespace — rebound HERE so the room build + envelope polls label
# THIS cell. The committed lab/e261_rank_ladder.py is untouched.
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

# the arms (execution order: the build first — the question's arm; then the
# sanctuary twin — the same-session passive control)
ARMS = ("NAME-MAINTAINED", "SANCTUARY-TWIN")
REA_ARM, TWN_ARM = ARMS
ARM_DESC = {
    REA_ARM: "(a) NAME-MAINTAINED (the build, the third form): the e286 "
             "rig with TWO changed dials — (i) the maintenance gradient "
             "is a NAME-ONLY/name-weighted CE — the maintenance loss "
             "computed on the NAME TOKENS ONLY (the 112 name positions; "
             "the pure name signal, no corpus dilution); (ii) the "
             "maintenance lr is BUDGET-FITTING — the maintenance "
             "momentum's accumulated displacement capped to fit WITHIN "
             "the remaining budget (the momentum-capped form); everything "
             "else kept: M=25 cadence, the fact-side buffer, the "
             "orthogonal corpus projection, the corpus-side cap. 400 "
             "corpus steps, 16 maintenance steps.",
    TWN_ARM: "(b) SANCTUARY-TWIN (the same-session passive control, "
             "e285's form).",
}

# THIS cell's ONE fresh registered corpus stream (the e268-family
# convention: .../28401/28501/28601/HERE 28701). BOTH arms draw the
# IDENTICAL sequence — bit-identical corpus batches across arms, the
# maintenance mechanism the arms' ONLY delta. The INSTALL (maintenance)
# stream has its OWN registered generator, seed 28702 (disclosed; the
# corpus stream untouched).
CORPUS_GEN_SEED = 28701
INST_GEN_SEED = 28702

# THE CONVENTION FREEZES (e283/e285's, carried)
PHASE_STEPS = 8 if SMOKE else 400               # CORPUS steps per arm
MILESTONES = tuple(range(1, 9)) if SMOKE else (100, 200, 300, 400)

# ---- THE MAINTENANCE SCHEDULE (frozen) ---------------------------------
MAINT_EVERY = 2 if SMOKE else 25     # maintenance after every M corpus steps
N_MAINT_EXPECTED = PHASE_STEPS // MAINT_EVERY    # 16 full / 4 smoke

# THE BETWEEN-STEPS WINDOW READS (the dispatch: "sample t24/t26 around one
# maintenance window at least once, disclosed" — HERE twice: the FIRST
# window and the MID-PHASE window; smoke scales to the smoke cadence)
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

# ---- THE SGD-M CONFIG (frozen; e280/e284/e285's committed lr class) ----
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
E273_LR_SGD = 21.7385748014537         # the calibration record's literal
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.21738574801453703

# ---- THE MAINTENANCE LR (frozen; see the docstring's convention (ii)) --
# MAINT_LR_BASE is the SCHEDULE ceiling (e286's denomination VERBATIM: the
# applying optimizer's SGD class, LR_SGD_matched/100 == LR_STABLE); the
# BUDGET-FITTING CAP (applied per maintenance event on the pending momentum
# b_m = 0.9*buf_F + g_name) reduces the APPLIED lr so the event's realized
# displacement fits the stream's remaining equal share of the budget — the
# e286 blowout (89.5% on maintenance alone) is structurally excluded.
MAINT_LR_DIV = 100                        # "1/100 of the install's lr" (the schedule ceiling's divisor)
MAINT_LR_BASE = LR_SGD_MATCHED / MAINT_LR_DIV   # 0.21738574801453703 == LR_STABLE
assert MAINT_LR_BASE == LR_STABLE              # the denominations coincide (bound)
MAINT_LR = MAINT_LR_BASE                  # legacy alias for the schedule ceiling (e286's symbol)

# ---- THE DISPLACEMENT BUDGET (frozen; e285's DIAL 3 verbatim) ----------
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

E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"
E278_VERDICT = "UNDERTOW-REGARDLESS"         # the roach-motel record

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

# e285's committed record (THIS cell's direct parent — the passive
# sanctuary's numbers; the twin's form + the coupling cite)
E285_METRICS = E43.REPO / "runs" / "e285" / "metrics.json"
E285_MD5 = "3f71c7b209fe9b6e9645d7504ddbb49c"
E285_VERDICT = "AIM-ONLY-KILLS"
E285_SAN_POST = 0.0008025270071811974        # the sanctuary's dead read
E285_SAN_RATIO = 0.00303243599263398         # 0.0030x
E285_SAN_DRIFT = 1.844509195284748           # 20.1% of the write
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

# e286's committed record (THIS cell's direct parent — the active first
# form's honest failure + win1's existence proof + THE MISBIND datum
# discovered at this cell's birth: its maintenance batch bound anchor_full
# into the inst_x slot — host openings at the masked positions, 0/60
# windows carried the name; see G_NAMEWIN)
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

E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268's G_PARENTS bind

# the frozen bars' numbers ------------------------------------------------
SURVIVE_FRAC = 0.5               # NAME-MAINTAINED-HOLDS: >= 0.5x at t400
PARTIAL_FRAC = 0.05              # SAWTOOTH-CONFIRMED: >= 0.05x
STREAM_STABLE_TOL = 1.05         # "stable": late CE <= 1.05 x early (frozen)
ORTH_BAR = 1e-6                  # G_ORTH: the missile's orthogonality gate
BUFSEP_ORTH_BAR = 1e-4           # G_BUFSEP: ||P_room buf_C||/||buf_C|| below
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
RATIO_DEN_FLOOR = 1e-6           # the ladder's floor-guard convention
VOCAB_EXPECT = 65

REGISTERED = {
    "law_verbatim": "e286 (the active first form) failed honestly: the "
        "budget blown 130.2% (the maintenance momentum ALONE ate 89.5%), "
        "the corpus lr frozen, the read dead at 0.0180x — but the "
        "mechanism's signature was machine-measured: win1's maintenance "
        "step LIFTED the read x2.203 (the existence proof the read "
        "responds to the name component) and the obstacle was DIRECTIONAL "
        "(the maintenance gradient's in-room fraction flat at the corpus "
        "~0.06). e287's birth-day correction: e286's maintenance batch "
        "was MISBOUND (anchor_full in the inst_x slot — host openings at "
        "the masked positions, 0/60 windows carried the name) — the name "
        "signal was NEVER injected before this cell. The third form "
        "injects it: the name-only CE at a budget-fitting lr.",
    "arms_verbatim": ARM_DESC,
    "bars_verbatim": {
        "NAME-MAINTAINED-HOLDS": ">= 0.5x at t400 with the budget HELD "
            "(<= 100%) and the stream live (the corpus CE improving or "
            "stable) — THE FIRST ACTIVE SURVIVAL: a memory maintained by "
            "its own name signal under traffic.",
        "SAWTOOTH-CONFIRMED": "0.05x-0.5x with the windows showing "
            "consistent LIFTS (every sampled window x>1) — the mechanism "
            "real, the dosage still short; the trajectory + windows "
            "verbatim.",
        "FAILS": "< 0.05x or the budget blown again or the lifts absent — "
            "the trajectories verbatim.",
        "MIXED": "anything else.",
    },
    "reads_verbatim": "the between-steps windows (sample >= 2, both "
        "directions); the maintenance gradient's in-room fraction per step "
        "(expect >> 0.06 now — the pure name signal's geometry); the "
        "budget split; the corpus CE.",
    "operationalizations": (
        "frozen BEFORE compute: THE PHASE := 400 corpus steps + 16 "
        "maintenance steps interleaved (NAME-MAINTAINED) / 400 corpus "
        "steps (twin); milestones t=100/200/300/400 read POST-maintenance "
        "(t400 includes the 16th step); THE VEHICLE := the committed "
        "quiet-formed 10k fact loaded BIT-EXACT (three-way G_FACTLOAD); "
        "the room := the fact's OWN K10K room (bit-gated vs "
        "e264_rooms.pt); THE MAINTENANCE WINDOWS := win_i (e261's "
        "build_win VERBATIM, G_NAMEWIN decode-gated — THE E286 MISBIND'S "
        "CORRECTION: e286 bound anchor_full into the inst_x slot, host "
        "openings at the masked positions, 0/60 name — the name signal "
        "was NEVER injected before this cell); THE CORPUS STEP := "
        "e268's registered form VERBATIM on THIS cell's ONE fresh "
        f"registered stream seed {CORPUS_GEN_SEED} (BOTH arms draw the "
        "identical sequence — the maintenance mechanism is the arms' "
        "ONLY delta); THE NAME-ONLY CE := ix(16) TRUE install windows "
        f"from the registered install generator seed {INST_GEN_SEED}; "
        "token-level CE; select the masked NAME positions (7 x 16 = "
        "112); loss := their MEAN; the Dmix corpus half DROPPED (the "
        "pure name signal, no corpus dilution); backward -> clip 1.0 -> "
        "UNPROJECTED -> through opt_F (the FACT-side SGD-M, private "
        "persistent buffer); THE BUDGET-FITTING LR := lr_m = min("
        "MAINT_LR_BASE, cap_m), cap_m = share_m/||b_m||, b_m = "
        "0.9*buf_F + g_name the EXACT pending in-step momentum, share_m "
        "= (BUDGET - S_total)/(1 + rem corpus + rem maint) — the SAME "
        "equal-share reservation the corpus cap uses (e286's corpus cap "
        "VERBATIM + the symmetric maintenance cap): every optimizer "
        "event's realized displacement fits its share, S_total <= BUDGET "
        "by the triangle bound, e286's 89.5% blowout structurally "
        "excluded (16 x ~0.011 = ~0.18 = ~3.9% of budget; the dose ~1/15 "
        "of e286's first step — disclosed BEFORE compute); THE BUDGET "
        f":= {BUDGET_FRAC} x ||fact - base|| (~4.59) with S_total = "
        "S_corpus + S_maint BOTH counted (the split disclosed); a blown "
        "G_BUDGET routes to FAILS ('the budget blown again'), never a "
        "HALT; G_BUFSEP v2 := BIDIRECTIONAL bitwise isolation (any "
        "mismatch HALTs) + buf_C < 1e-4 in-room per milestone + opt_F "
        "stepped EXACTLY at the maintenance steps (counter-checked) + "
        "buf_F composition DISCLOSED (never a bar); THE WINDOWS := "
        f"t{WIN1[0]}/t{WIN1[1]}-pre/t{WIN1[1]}-post/t{WIN1[2]} AND "
        f"t{WIN2[0]}/t{WIN2[1]}-pre/t{WIN2[1]}-post/t{WIN2[2]} (sample "
        ">= 2, both directions — the SAME two as e286); 'stream live' "
        ":= median corpus CE rows t in (300,400] vs [1,100]: improving "
        "(late < early) OR stable (late <= 1.05 x early); 'every sampled "
        "window x>1' := BOTH window lift ratios > 1; COMPOSITE := "
        "TEXTURE (bind/isolation failure) -> FAILS-by-gates (G_BUDGET "
        "blown OR G_ORTH failed, any ratio) -> NAME-MAINTAINED-HOLDS "
        "(ratio_400 >= 0.5 AND budget held AND stream live) -> MIXED-if-"
        "stream-dead (ratio_400 >= 0.5 AND held AND NOT live) -> "
        "SAWTOOTH-CONFIRMED (0.05 <= ratio_400 < 0.5 AND held AND every "
        "window lift > 1) -> FAILS-by-lifts (0.05 <= ratio_400 < 0.5 AND "
        "held AND a lift absent) -> FAILS-by-ratio (ratio_400 < 0.05 AND "
        "gates held) -> MIXED (everything else); HARD GATES (HALT) := "
        "{G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, "
        "G_NAMEWIN, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, "
        "G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND, "
        "G_MAINTBIND, G_BUFSEP-isolation}; NON-HALTING (route to FAILS) "
        ":= {G_ORTH, G_BUDGET, bufC-composition}; NO CONS (T259/e281; "
        "both post states checkpointed)."),
    "registration": "bars + question + arms + BOTH changed conventions "
        "(the name-only CE's construction + the budget-fitting lr) frozen "
        "VERBATIM from the dispatch letter (the e286 fold's dispatch; the "
        "build lane's third iteration — the name signal's FIRST "
        "injection); this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping.",
    "predictions": {
        "P-e287a_name_geometry": "the pure name gradient points "
            "substantially IN-ROOM (>> 0.06 — the signal that WROTE the "
            "fact, which stands 94.4% in-room): the maintenance steps "
            "lift the read at the windows and the third form holds >= "
            "0.05x with the budget held — SAWTOOTH-CONFIRMED or better.",
        "P-e287b_saturation": "the fact already KNOWS the name (name CE "
            "~0.2-0.3 nats, concentrated in the first char where the "
            "battery reads p(Z) ~ 0.26): the name-only gradient may be "
            "tiny and first-char-local — under the budget-fitting cap "
            "the dose (~0.011/step) may sit below the read's response "
            "threshold; the windows flat -> FAILS with the direction "
            "read still the datum.",
        "P-e287c_corrected_skeptic": "e286's win1 lift came from a "
            "HOST-TEXT batch (the birth-day misbind: 0/60 windows "
            "carried the name); the name signal's effect on the read is "
            "genuinely unmeasured — ANY outcome is informative, and the "
            "in-room fraction + the window shape are the discriminating "
            "reads.",
    },
}

deviations: list[str] = [
    "THE BIRTH-DAY MISBIND DISCOVERY (the record's material correction, "
    "disclosed at the top of the docstring + hard-gated HERE): porting "
    "the rig exposed that e286's chunked_reanchor_phase call bound "
    "anchor_full (the original-host anchor bank) into the driver's "
    "inst_x slot — the masked positions of its maintenance batch held "
    "the HOST OPENINGS ('ELIZABE'/'FLORIZE'), NOT ZEPHYRA: 0/60 windows "
    "carried the name (decode-verified this session: win_i masked == "
    "ZEPHYRA 60/60; anchor masked == ZEPHYRA 0/60; win_i[:, :PRE] "
    "bit-equal anchor_full[:, :PRE]). Consequences stated honestly: "
    "e286's 'name_ce' 0.0615 was the CE on memorized host-opening text; "
    "its maintenance gradient carried ZERO name content (the flat "
    "in-room 0.0597-0.0612 == the corpus ~0.06 because the step WAS a "
    "corpus-text step); win1's lift x2.203 came from a host-text batch. "
    "THE NAME SIGNAL HAS NEVER BEEN INJECTED BEFORE THIS CELL — the "
    "dispatch's '~1% of the mixed mass' premise is superseded by the "
    "stronger measured fact (0%). e287's design is UNCHANGED (the "
    "dispatch's two dials run exactly) but its reading is MORE first: "
    "the pure name signal's first injection. G_NAMEWIN (new, HARD) "
    "carries the decode gate so the misbind class can never recur "
    "silently.",
    "THE NAME-ONLY CE CONSTRUCTION (convention (i), frozen at birth): "
    "the maintenance batch := ix(16) TRUE install windows (win_i — "
    "e261's build_win VERBATIM: 130 pre-context tokens | ZEPHYRA | 119 "
    "post tokens; G_NAMEWIN-gated) drawn from the registered install "
    "generator seed 28702; forward the 16 windows; token-level CE; "
    "select the masked NAME positions (7 x 16 = 112); loss := their "
    "MEAN. The corpus half of e261's Dmix batch (aj anchors + rj random "
    "windows) is DROPPED entirely — no corpus tokens anywhere in the "
    "maintenance loss (the pure name signal, no corpus dilution); clip "
    "1.0; UNPROJECTED (the corpus orthogonality applies to corpus steps "
    "only — the re-anchoring may write in-room; disclosed); through "
    "opt_F (the FACT-side SGD-M, private persistent buffer).",
    "THE BUDGET-FITTING LR CONVENTION (convention (ii), frozen at "
    "birth, the momentum-capped form): the SCHEDULE ceiling stays "
    "e286's denomination VERBATIM (MAINT_LR_BASE = LR_SGD_matched/100 "
    "== LR_STABLE = 0.21738574801453703; the applying optimizer's SGD "
    "class; the AdamW-literal 1e-5 fork named + rejected as a "
    "guaranteed-null — carried from e286); the APPLIED lr at every "
    "maintenance event is lr_m = min(MAINT_LR_BASE, cap_m) with cap_m = "
    "share_m/||b_m||, b_m = 0.9*buf_F + g_name the EXACT pending "
    "in-step momentum (the event's displacement price lr_m x ||b_m||; "
    "pending_buffer_sqnorm), share_m = (BUDGET - S_total)/(1 + "
    "remaining corpus steps + later maintenance events) — the SAME "
    "equal-share reservation the corpus cap uses, SYMMETRIC: every "
    "optimizer event's realized displacement fits its reserved share, "
    "so S_total <= BUDGET by the triangle bound; e286's 89.5% blowout "
    "is structurally excluded. THE DOSE DISCLOSURE (BEFORE compute): 16 "
    "steps x ~share (~0.011) ~= 0.18 ~= 3.9% of budget — ~1/15 of "
    "e286's first step (0.162); the too-small-dose outcome is covered "
    "honestly by the bars (SAWTOOTH-CONFIRMED / FAILS report it); the "
    "direction read (the name-only gradient's in-room fraction) is "
    "dose-independent.",
    "THE BUDGET'S AMENDED ACCOUNTING v2 (e286's, carried + both sides "
    "capped): G_BUDGET (and G_ORTH) are computed, recorded, and "
    "NON-halting — a failure routes the verdict to FAILS ('the budget "
    "blown again' / the gates clause) with every ledger intact; the "
    "bind gates and the bidirectional buffer-isolation probe remain "
    "HARD HALTs. BOTH streams now enter the equal-share cap (e286's "
    "corpus cap VERBATIM + the new symmetric maintenance cap); a blown "
    "G_BUDGET would be an implementation outcome to autopsy, never an "
    "expected reading.",
    "THE OPT_F DISCLOSURE v2 (e286's, carried): opt_F IS the "
    "maintenance instrument — stepped EXACTLY at the 16 maintenance "
    "steps, at the budget-fitted lr_m, on the UNPROJECTED name-only "
    "gradient, its momentum buffer PRIVATE and persistent across "
    "maintenance steps. The isolation probe is BIDIRECTIONAL and "
    "machine-checked bitwise around EVERY optimizer event (any mismatch "
    "HALTS); buf_C's composition bar (< 1e-4 in-room) carried; buf_F's "
    "composition DISCLOSED per milestone (never a bar).",
    "THE TWIN IS THIS SESSION'S OWN DRAW STREAM (seed 28701, shared "
    "bit-identically with the NAME-MAINTAINED arm): e285's committed "
    "sanctuary ran seed 28501 and e286's twin seed 28601, so this twin "
    "replicates the FORM not the bits; its expected landing is the "
    "e285/e286 class (ratio ~0.002-0.003x, S ~100% of budget, cap bound "
    "~99% of steps), cross-checked against the hard-bound committed "
    "records. The class replication is the reference; bit-replication "
    "was never the family's rule.",
    "NO CONS (T259/e281; e278-e286's committed form): the frozen bars "
    "read the WRITE and the DISPLACEMENT only; both arms' post-phase "
    "states are checkpointed (e287_NAME-MAINTAINED_post.pt / "
    "e287_SANCTUARY-TWIN_post.pt) for any later landing pass.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT (e286's convention, "
    "carried): the SRCT projector + LadderRooms (the room rebuild, "
    "certification, displacement loads), the thermal envelope (per-step "
    "polls, 78C margin, 175s bursts inside the dispatch's 180s, 40s "
    "cooldowns, the 84C line inside the dispatch's 85C), the "
    "progressive-metrics + resume-ckpt conventions — the module-global "
    "rebinding (log/NAME/LADDER/RUNG_NAMES/T0/thermal ledgers, "
    "disclosed in-code) retargets the machinery's I/O to this cell; the "
    "committed lab/e261_rank_ladder.py is NOT modified. The TWO drivers "
    "are THIS file's: chunked_namemaint_phase (the third form — e286's "
    "reanchor driver with the TWO changed dials) and "
    "chunked_sanctuary_twin_phase (e285's chunked_sanctuary_phase "
    "VERBATIM in body — tags and checkpoint names adapted).",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the "
    "measured displacement loads; no new history is run.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E287_SMOKE=1): 8 corpus steps per arm, milestones at "
    "every step 1..8, room k=512 at the same seed pair (G_ROOMK10K "
    "vacuous — no committed record at smoke k; disclosed), the REAL "
    "committed fact loaded and read (G_FACTLOAD live), G_NAMEWIN live "
    "(the decode gate), G_CORPUSGEN live, G_MAINTBIND live at the smoke "
    "cadence (M=2 -> exactly 4 NAME-ONLY maintenance steps — the "
    "injection path + the budget-fitting cap + the total accounting "
    "exercised at share scale: the smoke budget is SCALED to the full "
    "run's per-event share, BUDGET x (12 events)/(416 events)), windows "
    "at t1/t2/t3 and t3/t4/t5, G_ORTH live (every step), G_BUFSEP live "
    "(bidirectional isolation + composition), G_BUDGET live (total + "
    "split); the adjudication form + figure exercised; all paths "
    "smoke_-prefixed, own smoke dir; NOTHING adjudicated or gated for "
    "the record (SMOKE stamp on every read).",
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
    """e278/e280/e284/e285's orthogonalize_grads VERBATIM: replace the
    (clipped) corpus gradient g by g_perp = g - P_room(g) — the component
    ENTIRELY ORTHOGONAL to the room (CPU fp64, write fp32, norm NOT
    rescaled). The verification read ||P_room g_perp|| / ||g_perp|| is
    returned for the gate."""
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
    state has none yet) — the G_BUFSEP isolation probe's eyes."""
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
    buffer state — the G_BUFSEP composition probe (CPU fp64 projection)."""
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


# ======================================================================
# THE NAME-MAINTAINED DRIVER — the third form: the name-only CE at the
# budget-fitting lr (e286's reanchor driver + the TWO changed dials)
# ======================================================================
def chunked_namemaint_phase(tag: str, net0, proj: "E261.LadderRooms",
                           fact_flat_np: np.ndarray,
                           base_flat_np: np.ndarray, budget_norm: float,
                           inst_x, inst_mask, anchor_full, train_ids,
                           g0_ids, gm12_ids, r_eval_xy, zid,
                           resume_ck: Path, dev: torch.device) -> dict:
    """THE NAME-MAINTAINED ARM'S DRIVER (the third form). The net starts
    at the LOADED formed fact; per corpus step t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 28701 — bit-identical to
      the twin's draws); corpus batch = 16 original-host anchors + 32
      random corpus windows; full-window CE; lr_sched = LR_STABLE x
      cosine_lr(t-1, 1000); backward -> clip 1.0 -> the orthogonalized
      stream (g_perp, verified EVERY step) -> the per-step lr cap
      (equal-share reservation across ALL remaining optimizer events —
      corpus + maintenance; e286's corpus cap VERBATIM) -> opt_C.step()
      (the corpus side's OWN buffer; BOTH optimizers snapshotted bitwise
      around the step).

      AND THE MAINTENANCE INJECTION (the cell's build, TWO changed dials):
      after every M=25 corpus steps, ONE NAME-ONLY gradient step — ix(16)
      TRUE install windows (win_i, G_NAMEWIN-gated) from igen seed 28702;
      token-level CE on the masked NAME positions ONLY (112; the pure
      name signal, no corpus dilution); clip 1.0; UNPROJECTED, applied
      through opt_F (the FACT-side SGD-M) at the BUDGET-FITTED lr_m =
      min(MAINT_LR_BASE, cap_m) with cap_m = share_m/||b_m|| on the
      pending momentum b_m = 0.9*buf_F + g_name (the event's exact
      displacement price) — bidirectional bitwise isolation; the realized
      displacement enters S_maint (the budget's total) and the
      maintenance ledger (with the name-only gradient's in-room fraction
      — THE registered read).

    Ledgers: e285/e286's full set (orth / corpus / disp / buf / budget /
    lr) + the maintenance ledger (every maintenance step) + the WINDOW
    ledger (the between-steps read at t24/t25pre/t25post/t26 and
    t199/t200pre/t200post/t201). Thermal: a poll after EVERY optimizer
    event."""
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    name_bs = G1.NAME_BS
    n_steps = PHASE_STEPS
    M = MAINT_EVERY
    n_anc = anchor_full.shape[0]
    n_inst = inst_x.shape[0]
    N = int(fact_flat_np.size)
    # the window reads' map: plain rows at the non-maintenance flank steps
    # (t24/t26, t199/t201) + PRE/POST rows around the maintenance step
    # itself (t25/t200); labels carry the window name (smoke overlaps)
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
                f"{REA_ARM}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            # e285's DIAL 1 topology: TWO SGD instances over the SAME
            # parameters — the corpus side's buffer PRIVATE; the fact side
            # NOW the maintenance instrument (stepped at maintenance steps
            # only, at the budget-fitted lr_m, unprojected).
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_F = torch.optim.SGD(net.parameters(), lr=MAINT_LR_BASE,
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
            # the RAW gradient's in-room fraction (the re-aiming contrast)
            g_in_room = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                pg = room.project(g64)
                g_in_room = float(np.linalg.norm(pg)
                                  / max(np.linalg.norm(g64), 1e-30))
            # ---- DIAL 2: the orthogonalized stream (verified EVERY step)
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- DIAL 3 (amended): the per-step lr cap — equal-share
            # reservation across ALL remaining optimizer events (the
            # remaining corpus steps + the remaining maintenance events)
            b_sq = pending_buffer_sqnorm(opt_C, params_live, SGD_MOMENTUM)
            b_norm = math.sqrt(max(b_sq, 0.0))
            rem_corpus = n_steps - step + 1
            rem_maint = sum(1 for m in range(step, n_steps + 1)
                            if m % M == 0)
            remaining = max(budget_norm - S_total, 0.0)
            share = remaining / (rem_corpus + rem_maint)
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
                    "rem_corpus": rem_corpus, "rem_maint": rem_maint}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(
                f"{REA_ARM}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)

            # ---- the plain window rows (the flanks: t24/t26, t199/t201)
            if step in win_mid:
                _read_window(win_mid[step], step)

            # ---- the PRE-maintenance window row (t25/t200) --------------
            if step in win_maint:
                _read_window(f"{win_maint[step]}:t{step}-pre", step)

            # ---- THE MAINTENANCE INJECTION (the cell's build, the third
            # form: THE NAME-ONLY CE at the BUDGET-FITTING lr) -----------
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
                # the name-only gradient's in-room fraction (THE registered
                # read: expect >> the corpus ~0.06 — the pure name signal's
                # geometry; e286's misbound steps sat AT the corpus value)
                g64i = torch.cat([p.grad.detach().reshape(-1)
                                  for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                gin_i = float(np.linalg.norm(room.project(g64i))
                              / max(np.linalg.norm(g64i), 1e-30))
                # DIAL (ii) — THE BUDGET-FITTING CAP: b_m = 0.9*buf_F +
                # g_name prices THIS event's displacement (lr_m x ||b_m||);
                # cap the applied lr so the event fits the stream's
                # remaining equal share (this event + every later event)
                bm_sq = pending_buffer_sqnorm(opt_F, params_live,
                                              SGD_MOMENTUM)
                bm_norm = math.sqrt(max(bm_sq, 0.0))
                rem_corpus_after = n_steps - step
                rem_maint_after = sum(1 for m in range(step + 1,
                                                       n_steps + 1)
                                      if m % M == 0)
                remaining_m = max(budget_norm - S_total, 0.0)
                share_m = remaining_m / (1 + rem_corpus_after
                                         + rem_maint_after)
                cap_m = share_m / max(bm_norm, 1e-12)
                lr_m = min(MAINT_LR_BASE, cap_m)
                capped_m = bool(cap_m < MAINT_LR_BASE)
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
                    "lr_sched": MAINT_LR_BASE, "cap_m": cap_m,
                    "lr_maint": lr_m, "capped_m": capped_m,
                    "b_m_norm": bm_norm, "share_m": share_m,
                    "realized_step_norm": realized_m,
                    "realized_vs_share": (realized_m / share_m
                                          if share_m > 0 else None),
                    "bufF_inroom_frac": bufF_ir, "bufF_norm": bufF_n,
                    "S_maint": S_maint, "S_total": S_total,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] MAINT #{n_maint:2d} @t{step:3d}: name_ce "
                    f"{float(loss_i.item()):.4f} ({int(nm_i.numel())} toks) "
                    f"|g| {gn_i:.3f} in-room {gin_i:.4f} | lr_m {lr_m:.6f} "
                    f"{'CAP' if capped_m else 'sched'} (b_m {bm_norm:.3f} "
                    f"share {share_m:.5f}) -> step {realized_m:.5f} "
                    f"{(realized_m / share_m if share_m > 0 else float('nan')):.2f}x "
                    f"share | S_maint {S_maint:.4f} S_total "
                    f"{S_total:.4f}/{budget_norm:.4f} "
                    f"({S_total / budget_norm:.1%}) bufF "
                    f"{('%.*e' % (2, bufF_ir)) if bufF_ir is not None else 'n/a'}")
                n_burst += 1
                ok_t, temp = E261.burst_temp_check(
                    f"{REA_ARM}:maint:c{n_chunks}.x")
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


# ======================================================================
# THE SANCTUARY-TWIN DRIVER — e285's chunked_sanctuary_phase VERBATIM in
# body (the passive stack: separate buffers + orthogonal projection + the
# budget cap, NO maintenance — the same-session control)
# ======================================================================
def chunked_sanctuary_twin_phase(tag: str, net0, proj: "E261.LadderRooms",
                                 fact_flat_np: np.ndarray,
                                 base_flat_np: np.ndarray, budget_norm: float,
                                 anchor_full, train_ids, g0_ids, gm12_ids,
                                 r_eval_xy, zid, resume_ck: Path,
                                 dev: torch.device) -> dict:
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    n_steps = PHASE_STEPS
    n_anc = anchor_full.shape[0]
    N = int(fact_flat_np.size)
    state = {"step": 0, "traj": [], "corpus_ledger": {}, "orth_ledger": {},
             "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
             "lr_ledger": {},
             "bufsep": {"isolation_checks": 0, "isolation_violations": 0,
                        "optF_ever_stepped": False},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "corp_prev": torch.zeros(N, dtype=torch.float64),
             "S": 0.0, "n_capped": 0}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at corpus step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        lrs_c = [v["lr_applied"] for v in state.get("lr_ledger",
                                                    {}).values()]
        scheds_c = [v["lr_sched"] for v in state.get("lr_ledger",
                                                     {}).values()]
        caps_c = [v["cap"] for v in state.get("lr_ledger", {}).values()]
        return {"sd": state["model"], "traj": state.get("traj", []),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "buf_ledger": state.get("buf_ledger", []),
                "budget_ledger": state.get("budget_ledger", []),
                "lr_ledger": state.get("lr_ledger", {}),
                "bufsep": state.get("bufsep", {}),
                "S": state.get("S"), "n_capped": state.get("n_capped"),
                "orth_max": state.get("orth_max"),
                "lr_applied_min": min(lrs_c) if lrs_c else None,
                "lr_applied_median": (float(sorted(lrs_c)[len(lrs_c) // 2])
                                      if lrs_c else None),
                "lr_sched_median": (float(sorted(scheds_c)[len(scheds_c)
                                                           // 2])
                                    if scheds_c else None),
                "cap_min": min(caps_c) if caps_c else None,
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt_C = opt_F = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[ROOM_MODE]
    corp_cum = state["corp_cum"].to(dev)
    corp_prev = state["corp_prev"].to(dev)
    S = float(state["S"])
    n_capped = int(state["n_capped"])
    orth_max = 0.0

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{TWN_ARM}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_F = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
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
                S = float(state["S"])
                n_capped = int(state["n_capped"])
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            lr_sched = LR_STABLE * cosine_lr(step - 1, E261.INST_TOTAL)
            # ---- THE CORPUS STEP (bit-identical draws to the build's)
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
            g_in_room = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                pg = room.project(g64)
                g_in_room = float(np.linalg.norm(pg)
                                  / max(np.linalg.norm(g64), 1e-30))
            # ---- DIAL 2: the orthogonalized stream (verified EVERY step)
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- DIAL 3: the per-step lr cap (equal-share, corpus only —
            # e285's form VERBATIM; no maintenance events exist here)
            b_sq = pending_buffer_sqnorm(opt_C, params_live, SGD_MOMENTUM)
            b_norm = math.sqrt(max(b_sq, 0.0))
            remaining = max(budget_norm - S, 0.0)
            share = remaining / (n_steps - step + 1)
            cap = share / max(b_norm, 1e-12)
            lr_t = min(lr_sched, cap)
            capped = bool(cap < lr_sched)
            if capped:
                n_capped += 1
            for g_ in opt_C.param_groups:
                g_["lr"] = lr_t
            # ---- DIAL 1: the isolated corpus step (the fact side EMPTY)
            sn_f_before = snap_buffers(opt_F, params_live)
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()
            with torch.no_grad():
                d_step = torch.cat([p.detach().reshape(-1)
                                    for p in net.parameters()]) - theta_b
                corp_cum += d_step
            realized = float(d_step.norm().item())
            S += realized
            state["bufsep"]["isolation_checks"] += 1
            if not buffers_bitwise_equal(sn_f_before,
                                         snap_buffers(opt_F, params_live)):
                state["bufsep"]["isolation_violations"] += 1
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
                    "share": share, "realized_step_norm": realized}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(
                f"{TWN_ARM}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)
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
                v_int_np = (corp_cum - corp_prev).double().cpu().numpy()
                vn = float(np.linalg.norm(v_int_np))
                if vn > 0:
                    pv = room.project(v_int_np)
                    in_room_frac_c = float(np.linalg.norm(pv) / vn)
                else:
                    in_room_frac_c = None
                corp_prev = corp_cum.clone()
                cum_np = corp_cum.double().cpu().numpy()
                cum_n = float(np.linalg.norm(cum_np))
                pcum = room.project(cum_np)
                in_room_frac_cum = (float(np.linalg.norm(pcum) / cum_n)
                                    if cum_n > 0 else None)
                bufC_ir, bufC_n = buffer_inroom_frac(opt_C, params_live,
                                                     room)
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_fact_norm": dn,
                    "drift_from_fact_in_room_frac": d_fact_in_room,
                    "remaining_from_base_norm": rn,
                    "remaining_from_base_in_room_frac": rem_in_room,
                    "corpus_disp_interval_norm": vn,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_norm": cum_n,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_inroom_frac": bufC_ir, "bufC_norm": bufC_n,
                    "optF_state_entries": len(opt_F.state),
                    "optF_ever_stepped": False})
                state["budget_ledger"].append({
                    "step": step, "S": S,
                    "S_usage_frac": S / budget_norm,
                    "cum_norm": cum_n,
                    "cum_usage_frac": cum_n / budget_norm,
                    "n_capped_so_far": n_capped,
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
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum,
                    "budget_S_usage_frac": S / budget_norm,
                    "budget_cum_usage_frac": cum_n / budget_norm,
                    "lr_applied": lr_t, "lr_sched": lr_sched,
                    "capped": capped,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.6f} "
                    f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d| {dn:.3f} drift-in-room "
                    f"{('%.2e' % d_fact_in_room) if d_fact_in_room is not None else 'n/a'} "
                    f"rem-in-room "
                    f"{('%.4f' % rem_in_room) if rem_in_room is not None else 'n/a'} "
                    f"| BUDGET S {S:.4f}/{budget_norm:.4f} "
                    f"({S / budget_norm:.1%}) cum {cum_n:.4f} "
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
                    "bufsep": state["bufsep"],
                    "corp_cum": corp_cum.cpu(),
                    "corp_prev": corp_prev.cpu(),
                    "S": S, "n_capped": n_capped, "orth_max": orth_max,
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
            "bufsep": state["bufsep"],
            "S": S, "n_capped": n_capped, "orth_max": orth_max,
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
           "per_step_polls": "after EVERY optimizer event (corpus + "
                             "maintenance, both arms) — aggregated from "
                             "runs/_envelope_log.jsonl (the persisted "
                             "ledger; survives resume passes; tags "
                             "e287:<ARM>:<phase> per the dispatch)",
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
                   f"polls are tagged e287_smoke: and excluded")
    return out


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e287_name_maintained",
        "phase": "THE NAME-WEIGHTED MAINTENANCE CELL — the build lane's "
                 "third iteration: the lab's FIRST INJECTION OF THE NAME "
                 "SIGNAL ITSELF as the maintenance mechanism. e286 (the "
                 "active first form) failed honestly — the budget blown "
                 "130.2% (the maintenance momentum alone 89.5%), the read "
                 "0.0180x — but measured its signature: win1's lift "
                 "x2.203 and a maintenance gradient whose in-room "
                 "fraction sat AT the corpus ~0.06. THE BIRTH-DAY "
                 "CORRECTION: e286's maintenance batch was MISBOUND "
                 "(anchor_full in the inst_x slot — host openings at the "
                 "masked positions, 0/60 windows carried the name) — the "
                 "name signal was NEVER injected before this cell. e287 = "
                 "the e286 rig with TWO changed dials: (i) the maintenance "
                 "gradient is a NAME-ONLY CE (the 112 name positions, the "
                 "pure name signal, no corpus dilution); (ii) the "
                 "maintenance lr is BUDGET-FITTING (the pending momentum's "
                 "displacement capped to the stream's remaining equal "
                 "share) — vs the same-session SANCTUARY-TWIN (the "
                 "passive stack): NAME-MAINTAINED-HOLDS vs SAWTOOTH-"
                 "CONFIRMED vs FAILS vs MIXED, adjudicated on the WRITE "
                 "read at t400 with the gates' verdicts",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property; "
                           "the frozen bars read the WRITE and the "
                           "DISPLACEMENT only; e278-e286's committed "
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
                      "e287:<ARM>:<phase>",
            "trainings": "2 phases x 400 corpus opt steps (NAME-"
                         "MAINTAINED: SGD-M separate buffers + orthogonal "
                         "projection + budget-capped corpus lr + 16 NAME-"
                         "ONLY maintenance steps through opt_F at the "
                         "budget-fitted lr_m; SANCTUARY-TWIN: e285's "
                         "passive stack verbatim); NO cons",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} corpus steps per arm; the RE-ANCHORED "
                     f"arm interleaves {N_MAINT_EXPECTED} maintenance steps "
                     f"(one after every M={MAINT_EVERY} corpus steps)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps (read POST-maintenance on the "
                          "build arm)",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors (the same "
                           "60-window bank) + 32 random corpus windows; "
                           "full-window CE; THIS cell's ONE fresh registered "
                           f"generator seed {CORPUS_GEN_SEED}; BOTH arms "
                           "draw the IDENTICAL sequence (the maintenance "
                           "mechanism is the arms' ONLY delta)",
            "maintenance_step": "THE THIRD FORM: ix(16) TRUE install "
                                "windows (win_i, G_NAMEWIN-gated) from the "
                                f"registered install generator seed "
                                f"{INST_GEN_SEED}; THE NAME-ONLY CE := mean "
                                "token-level CE over the masked name "
                                "positions ONLY (7 x 16 = 112; the Dmix "
                                "corpus half DROPPED — the pure name "
                                "signal); clip 1.0; UNPROJECTED, through "
                                "opt_F (the FACT-side SGD-M, private "
                                "buffer) at the APPLIED lr_m = min("
                                "MAINT_LR_BASE, cap_m) — the BUDGET-FITTING "
                                "cap on the pending momentum (see "
                                "the_maintenance_convention)",
            "survival_ratio": f"post g0(t) / {FACT_BASELINE_G0} (the "
                              "committed loaded baseline, PRIMARY; the "
                              "session's loaded read co-reported)",
            "adjudication_read": "the t=400 endpoint (post the 16th "
                                 "maintenance step) with the gates' "
                                 "verdicts at the same endpoint; "
                                 "trajectories + the twin co-reported",
        },
        "the_maintenance_convention": {
            "the_step": "ONE NAME-ONLY gradient step after every "
                        f"M={MAINT_EVERY} corpus steps — exactly "
                        f"{N_MAINT_EXPECTED}: batch := ix(16) TRUE install "
                        "windows (win_i — e261's build_win VERBATIM, "
                        "ZEPHYRA spliced, G_NAMEWIN-gated); loss := mean "
                        "token-level CE over the masked NAME positions "
                        "only (112; the corpus half of e261's Dmix batch "
                        "DROPPED — the pure name signal, no corpus "
                        "dilution); backward -> clip 1.0 -> UNPROJECTED -> "
                        "through opt_F (the fact-side SGD-M, private "
                        "persistent buffer)",
            "the_lr": f"MAINT_LR_BASE = LR_SGD_matched/{MAINT_LR_DIV} = "
                      f"{MAINT_LR_BASE!r} (== LR_STABLE; e286's "
                      "denomination VERBATIM as the SCHEDULE ceiling; the "
                      "AdamW-literal 1e-5 fork stays rejected as a "
                      "guaranteed-null); THE BUDGET-FITTING CAP (the "
                      "momentum-capped form, frozen at birth): at every "
                      "maintenance event lr_m = min(MAINT_LR_BASE, cap_m), "
                      "cap_m = share_m/||b_m||, b_m = 0.9*buf_F + g_name "
                      "the EXACT pending in-step momentum (the event's "
                      "displacement price lr_m x ||b_m||), share_m = "
                      "(BUDGET - S_total)/(1 + remaining corpus steps + "
                      "later maintenance events) — the SAME equal-share "
                      "reservation the corpus cap uses, SYMMETRIC: every "
                      "optimizer event fits its share, S_total <= BUDGET "
                      "by the triangle bound; e286's 89.5% blowout "
                      "structurally excluded (16 steps x ~0.011 = ~0.18 "
                      "~ 3.9% of budget)",
            "the_budget": f"BUDGET = {BUDGET_FRAC} x ||fact - base|| "
                          "(~4.59); S_total = S_corpus + S_maint BOTH "
                          "counted (the split disclosed); BOTH sides now "
                          "enter the equal-share cap (e286's corpus cap "
                          "VERBATIM + the new symmetric maintenance cap) — "
                          "a blown G_BUDGET would be an implementation "
                          "outcome to autopsy, routed to FAILS (never a "
                          "HALT)",
            "the_windows": f"t{WIN1[0]}/t{WIN1[1]}-pre/t{WIN1[1]}-post/"
                           f"t{WIN1[2]} AND t{WIN2[0]}/t{WIN2[1]}-pre/"
                           f"t{WIN2[1]}-post/t{WIN2[2]} (sample >= 2, both "
                           "directions — the SAME two windows as e286, "
                           "direct comparability with the parent's "
                           "ledger)",
            "the_name_signal": "the maintenance gradient's in-room "
                               "fraction is read PER STEP (expect >> 0.06 "
                               "— the pure name signal's geometry; e286's "
                               "misbound steps sat AT the corpus ~0.06 "
                               "because they WERE corpus-text steps)",
        },
        "deviations": deviations,
        "builds_on": [
            "the e286 fold's dispatch (the coordinator's third-form "
            "letter: the name-only CE + the budget-fitting lr; the bars "
            "frozen VERBATIM; 'entering with its existence proof already "
            "measured' — corrected at birth to the misbind's stronger "
            "fact)",
            "e286 / THE RE-ANCHORING CELL (the direct parent: the active "
            "first form's honest failure — the budget blown 130.2% on "
            "uncapped maintenance momentum, win1's lift x2.203, win2's "
            "crash x0.377, the in-room fraction AT the corpus ~0.06; THE "
            "MISBIND discovered at THIS cell's birth: its maintenance "
            "batch never carried the name — G_NAMEWIN corrects + "
            "hard-gates it; the rig + the budget's total accounting + the "
            "window reads port from here)",
            "T265 / e285 (THE SANCTUARY — the passive stack's honest "
            "failure; all dials verified, the read dead at 0.0030x, the "
            "write's mass 93% intact — THE READ IS A PROPERTY OF THE "
            "WHOLE STATE; the twin's form + the projection/cap machinery "
            "port from here)",
            "the e285 fold's dispatch (the coordinator's threshold datum: "
            "7.2% drift killed ~200x at t100 — the coupling constant "
            "<= 0.07x the write's norm; the fork tilted to the ACTIVE "
            "design)",
            "T264 / e284 (MOMENTUM-OWNED: buffer separation kills the "
            "re-aiming — DIAL 1 + DIAL 2's machinery)",
            "T261 / e283 (the established-write rig: the loaded fact, the "
            "400-corpus-step phase, the conventions)",
            "T263 / x14 (DIRECTIONAL TRANSPORT: the out-of-room context "
            "carries the kill — e285's AIM-ONLY clause + this cell's "
            "motivation)",
            "T239 / e261 + T242 / e264 (the fact itself + the ladder "
            "machinery PORTED WHOLE BY IMPORT; chunked_install's step form "
            "is THE maintenance step, verbatim)",
            "T259 / e281 (the NO-CONS form; the cons's teaching power — "
            "the dispatch's 'the organism's own dynamics' cite)",
            "T258 / e273 (the lr calibration: LR_SGD_matched — the "
            "maintenance lr's provenance, md5-bound)",
        ],
        "whats_new": [
            "THE NAME SIGNAL'S FIRST INJECTION (the record's first — the "
            "birth-day correction's fruit): e286's maintenance batch was "
            "misbound (host openings at the masked positions, 0/60 name) "
            "and every prior cell only WROTE the fact passively-loaded; "
            "HERE the pure name gradient itself is the maintenance "
            "mechanism, its geometry (in-room fraction per step) and its "
            "effect on the read both measured for the first time",
            "THE BUDGET-FITTING MAINTENANCE LR (the record's first): the "
            "pending momentum's displacement capped to the stream's "
            "remaining equal share — the same reservation as the corpus "
            "cap, symmetric, making S_total <= BUDGET a construction not "
            "a hope (e286's 89.5% blowout structurally excluded)",
            "THE E286 MISBIND DISCOVERY + THE DECODE GATE (the record's "
            "material correction): the parent's 'name' term never carried "
            "the name — disclosed here, and G_NAMEWIN (masked positions "
            "decode to ZEPHYRA in ALL windows) makes the window vehicle "
            "bit-checkable forever after",
        ],
        "gates": {},
    })
    log(f"E287 — THE NAME-WEIGHTED MAINTENANCE CELL (the build lane's "
        f"third iteration; smoke={SMOKE}) -> {RD}")
    log(f"arms: {' / '.join(ARMS)}; the fact = {FACT_CK} (md5-bound, post "
        f"g0 {FACT_BASELINE_G0:.8f}); the room = the fact's own committed "
        f"K10K (seeds {LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs "
        f"{ROOMS264_CK}); the phase = {PHASE_STEPS} corpus steps + "
        f"{N_MAINT_EXPECTED} NAME-ONLY maintenance steps (M={MAINT_EVERY}, "
        f"MAINT_LR_BASE {MAINT_LR_BASE!r} schedule + THE BUDGET-FITTING "
        f"cap on the pending momentum); the corpus stream = seed "
        f"{CORPUS_GEN_SEED} (bit-identical across arms); the install "
        f"stream = seed {INST_GEN_SEED}; the bar: HOLDS >= "
        f"{SURVIVE_FRAC:.0%}x at t400 with the budget held + the stream "
        f"live; SAWTOOTH-CONFIRMED {PARTIAL_FRAC:.0%}-{SURVIVE_FRAC:.0%}x "
        f"with every window lift x>1; FAILS < {PARTIAL_FRAC:.0%}x / the "
        f"budget blown again / the lifts absent")
    write_partial("startup (bars + arms + the maintenance convention "
                  "registered, committed at birth)")
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
                          "and the mask IS the maintenance step's own "
                          "teaching mask (the Dmix CE's nm term)",
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    # ---- G_NAMEWIN: the TRUE install windows (win_i) — the name-only
    # CE's own vehicle bind + THE E286 MISBIND DISCLOSURE (the record's
    # material correction, decode-verified at this cell's birth: e286's
    # driver call bound anchor_full into the inst_x slot, so the masked
    # positions of its maintenance batch held the HOST OPENINGS — 0/60
    # windows carried the name; the name signal was never injected)
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
        "form": "the maintenance batch's windows := e261's build_win "
                "VERBATIM (the TRUE install windows: 130 pre-context "
                "tokens | ZEPHYRA | 119 post tokens); the masked "
                "y-positions (x-cols PRE-1..PRE+5) hold the NAME in ALL "
                "60 windows (decode-verified) and the pre-context is "
                "bit-identical to the anchor bank's — THE VEHICLE OF THE "
                "NAME-ONLY CE (the e286 misbind's correction)",
        "n_windows": int(win_i.shape[0]),
        "masked_decode_all_name": bool(win_masked_ok),
        "precontext_bit_equal_anchor": bool(win_pre_ok),
        "the_e286_misbind": {
            "finding": "e286's chunked_reanchor_phase call bound "
                       "anchor_full (the original-host anchor bank) into "
                       "the driver's inst_x slot: its maintenance batch's "
                       "masked positions held the HOST OPENINGS "
                       "('ELIZABE'/'FLORIZE'), NOT ZEPHYRA — 0/60 "
                       "windows carried the name",
            "verified": "decode check at this cell's birth: win_i masked "
                        "positions == ZEPHYRA 60/60; anchor_full masked "
                        "positions == ZEPHYRA 0/60 (host openings); "
                        "win_i[:, :PRE] bit-equal anchor_full[:, :PRE]",
            "consequence": "e286's 'name_ce' 0.0615 was the CE on "
                           "memorized host-opening text and its "
                           "maintenance gradient carried ZERO name "
                           "content — the flat in-room 0.0597-0.0612 == "
                           "the corpus ~0.06 because the step WAS a "
                           "corpus-text step; win1's lift x2.203 came "
                           "from a host-text batch. THE NAME SIGNAL HAS "
                           "NEVER BEEN INJECTED BEFORE THIS CELL",
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
        "win_i: ZEPHYRA at the masked positions 60/60, the e286 misbind "
        "corrected + hard-gated / vocab 65)")
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
                         "note": "the unprotected transport cite (post "
                                 "9.05e-06, drift 14.45 vs the write's "
                                 "9.18)"},
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
                                "out-of-room context carries the kill)"},
        "e285_metrics": {"path": str(E285_METRICS),
                         "md5": md5of(E285_METRICS), "bound_md5": E285_MD5,
                         "verdict": e285_verdict,
                         "sanctuary_post_g0": e285_san,
                         "sanctuary_ratio": E285_SAN_RATIO,
                         "sanctuary_S_final": e285_san_S,
                         "sanctuary_drift": e285_san_drift,
                         "twin_post_g0": e285_twn,
                         "t100_drift": E285_T100_DRIFT,
                         "t100_ratio": E285_T100_RATIO,
                         "coupling_bar": E285_COUPLING_BAR,
                         "note": "THIS CELL'S DIRECT PARENT (the passive "
                                 "sanctuary's honest failure: ratio "
                                 "0.0030x with the budget held + the write's "
                                 "mass 93% intact — the <= 0.07x coupling "
                                 "constant; the twin's form + the "
                                 "projection/cap machinery port from here)"},
        "e286_metrics": {"path": str(E286_METRICS),
                         "md5": md5of(E286_METRICS), "bound_md5": E286_MD5,
                         "verdict": e286_verdict,
                         "re_anchored_post_g0": e286_rea_post,
                         "ratio": E286_REA_RATIO,
                         "S_maint_final": e286_S_m,
                         "S_total_usage_frac": E286_S_TOTAL_USAGE,
                         "win1_lift": E286_WIN1_LIFT,
                         "win2_lift": E286_WIN2_LIFT,
                         "maint_in_room_frac_range":
                             [E286_MAINT_INROOM_FIRST,
                              E286_MAINT_INROOM_LAST],
                         "the_misbind": "e286's maintenance batch bound "
                                        "anchor_full into the inst_x slot "
                                        "(host openings at the masked "
                                        "positions, 0/60 windows carried "
                                        "the name — disclosed + corrected "
                                        "HERE; see G_NAMEWIN)",
                         "note": "THIS CELL'S DIRECT PARENT (the active "
                                 "first form: MAINTENANCE-FAILS with the "
                                 "budget blown 130.2%, the maintenance "
                                 "alone 89.5% — but win1's lift x2.203 the "
                                 "existence proof + the in-room fraction "
                                 "AT the corpus ~0.06; the third form's "
                                 "premises)"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                "md5": md5of(E273_LRCAL),
                                "bound_md5": E273_LRCAL_MD5,
                                "lr_sgd": E273_LR_SGD,
                                "note": "LR_STABLE + MAINT_LR_BASE's "
                                        "provenance record"},
        "the_fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(CKPT_DIR / FACT_CK),
                     "bound_md5": FACT_MD5,
                     "size": (CKPT_DIR / FACT_CK).stat().st_size,
                     "bound_size": FACT_SIZE, "state": vehicle_state,
                     "note": "the established fact itself (the committed "
                             "serial 10k rung's final state)"},
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
            "e283_verdict": E283_VERDICT,
            "e283_post": E283_POST, "e283_ratio": E283_RATIO,
            "e283_drift": E283_DRIFT, "e283_write_norm": E283_WRITE_NORM,
            "e284_verdict": E284_VERDICT,
            "e284_sep_primary": E284_SEP_PRIMARY,
            "e284_sha_primary": E284_SHA_PRIMARY,
            "x14_verdict": X14_VERDICT, "x14_arm_A": X14_ARM_A,
            "x14_arm_B": X14_ARM_B, "x14_resurrection_x": X14_RESURRECTION_X,
            "e285_verdict": E285_VERDICT, "e285_san_post": E285_SAN_POST,
            "e285_san_ratio": E285_SAN_RATIO, "e285_san_drift":
                E285_SAN_DRIFT,
            "e285_S_final": E285_S_FINAL, "e285_budget_norm":
                E285_BUDGET_NORM,
            "e285_twin_post": E285_TWN_POST, "e285_twin_drift":
                E285_TWN_DRIFT,
            "e285_t100_drift": E285_T100_DRIFT, "e285_coupling_bar":
                E285_COUPLING_BAR,
            "e286_verdict": E286_VERDICT, "e286_rea_post": E286_REA_POST,
            "e286_rea_ratio": E286_REA_RATIO, "e286_S_maint": E286_S_MAINT,
            "e286_S_total_usage": E286_S_TOTAL_USAGE,
            "e286_maint_budget_share": E286_MAINT_BUDGET_SHARE,
            "e286_win1_lift": E286_WIN1_LIFT,
            "e286_win2_lift": E286_WIN2_LIFT,
            "e286_maint_inroom_range":
                [E286_MAINT_INROOM_FIRST, E286_MAINT_INROOM_LAST],
            "fact_md5": FACT_MD5, "fact_step": FACT_STEP,
            "fact_traj_steps": FACT_TRAJ_STEPS,
            "fact_ledger_max": FACT_LEDGER_MAX},
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
            and e286_verdict == E286_VERDICT
            and abs(e286_rea_post - E286_REA_POST) < 1e-12
            and abs(e286_S_m - E286_S_MAINT) < 1e-9
            and e285_verdict == E285_VERDICT
            and abs(e285_san - E285_SAN_POST) < 1e-12
            and abs(e285_san_S - E285_S_FINAL) < 1e-9
            and abs(e285_san_drift - E285_SAN_DRIFT) < 1e-9
            and abs(e285_twn - E285_TWN_POST) < 1e-15
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E283_METRICS) == E283_MD5
            and md5of(E284_METRICS) == E284_MD5
            and md5of(X14_METRICS) == X14_MD5
            and md5of(E285_METRICS) == E285_MD5
            and md5of(E286_METRICS) == E286_MD5
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
    log(f"P0b: G_PARENTS PASS — e285 {E285_VERDICT} (sanctuary "
        f"{e285_san:.2e} = x{E285_SAN_RATIO:.4f}, S {e285_san_S:.4f}, drift "
        f"{e285_san_drift:.3f}; twin {e285_twn:.2e}); e283 {E283_VERDICT} "
        f"(post {e283_post:.2e}, drift {e283_drift:.2f}); e284 "
        f"{E284_VERDICT}; x14 {X14_VERDICT}; the fact ckpt md5/size/step-"
        "bound")
    log(f"P0b: e286 {E286_VERDICT} hard-bound (the active parent: ratio "
        f"{E286_REA_RATIO:.4f}x, S_maint {E286_S_MAINT:.4f} = "
        f"{E286_MAINT_BUDGET_SHARE:.1%} of budget alone; win1 x"
        f"{E286_WIN1_LIFT:.3f} / win2 x{E286_WIN2_LIFT:.3f}; THE MISBIND: "
        f"its maintenance batch never carried the name — corrected HERE, "
        f"G_NAMEWIN hard-gates the true windows)")
    write_partial("P0b parents hard-bound (the fact ckpt loaded)")
    del e264m, e268m, e283m, e284m, x14m, e285m, e286m

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
    log(f"P1 G_VMBIND: {VMAP_CK} (md5 {G_VMBIND['md5'][:8]}..., "
        f"{G_VMBIND['size']} coords): PASS")

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
    log(f"P1 G_SPANBIND: {SPAN_CK} (rank {G_SPANBIND['rank']}): PASS")

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
        "e286_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e286's room (the fact's own room): the committed K10K "
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

    # ---- G_LR_BIND: the lr/momentum provenance bind (e280/e284/e285's
    # record + the maintenance lr's own bind)
    lr_sgd_runtime = float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    lr_stable_runtime = SGD_STABLE_FACTOR * lr_sgd_runtime
    maint_lr_runtime = lr_sgd_runtime / MAINT_LR_DIV
    G_LR_BIND = {
        "form": "the corpus lr := e273's STABLE RIDER POINT (e280/e284/"
                "e285/e286's committed class) — LR_STABLE = x0.01 x "
                "LR_SGD_matched, re-derived at runtime from the md5-bound "
                "runs/e273/lr_calibration.json and asserted == the frozen "
                "literal; momentum 0.9, wd 0.0 EXACTLY; THE MAINTENANCE "
                f"LR := LR_SGD_matched/{MAINT_LR_DIV} = MAINT_LR_BASE "
                "(likewise re-derived and asserted == LR_STABLE — e286's "
                "denomination VERBATIM) as the SCHEDULE CEILING; the "
                "APPLIED maintenance lr is BUDGET-FITTED per event (cap_m "
                "= share_m/||b_m|| on the pending momentum — the cell's "
                "frozen convention (ii)); both budget caps may reduce the "
                "APPLIED lrs below their schedules (disclosed per step)",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "maint_lr_div": MAINT_LR_DIV,
        "maint_lr_base": MAINT_LR_BASE,
        "maint_lr_base_runtime": maint_lr_runtime,
        "maint_lr_form": "lr_m = min(MAINT_LR_BASE, cap_m); cap_m = "
                         "((BUDGET - S_total)/(1 + rem_corpus + "
                         "rem_maint))/||b_m||; b_m = 0.9*buf_F + g_name",
        "denomination_disclosure": "MAINT_LR_BASE is read in the APPLYING "
                                   "optimizer's denomination (SGD-M, the "
                                   "fact-side buffer's class); e273's "
                                   "LR_SGD_matched is the lab's committed "
                                   "SGD-denominated value of the install's "
                                   "formation lr. The AdamW-literal reading "
                                   "(1e-5) named + rejected at birth (a "
                                   "guaranteed-null; see deviations)",
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
        "pass": bool(abs(lr_stable_runtime - LR_STABLE) < 1e-15
                     and float(LR_STABLE) == 0.21738574801453703
                     and abs(maint_lr_runtime - MAINT_LR_BASE) < 1e-15
                     and MAINT_LR_BASE == LR_STABLE
                     and SGD_MOMENTUM == 0.9 and SGD_WD == 0.0
                     and lr_sgd_runtime == E273_LR_SGD),
    }
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P1 G_LR_BIND: LR_STABLE = {SGD_STABLE_FACTOR} x {E273_LR_SGD} = "
        f"{LR_STABLE!r}; MAINT_LR_BASE = {E273_LR_SGD}/{MAINT_LR_DIV} = "
        f"{MAINT_LR_BASE!r} (== LR_STABLE, bound; the applied lr_m is "
        f"budget-fitted per event); momentum {SGD_MOMENTUM}, wd "
        f"{SGD_WD}: PASS")
    write_partial("P1b G_LR_BIND PASSED")

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
                "the family's cross-session read-determinism law)",
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
            "note": "the write's own standing displacement (the transport "
                    "channel's reference quantity; the budget = "
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
    }
    write_partial("P2 the fact loaded bit-exact + gated (the baseline read)")
    del fact_net

    # ---- the BUDGET (frozen at load time; the TOTAL accounting) ---------
    budget_norm = BUDGET_FRAC * write_norm
    if SMOKE:
        # the smoke budget is SCALED to the full run's per-EVENT share so
        # the cap's binding path + the maintenance accounting run at the
        # full-run share scale (full: 400 corpus + 16 maint = 416 events)
        n_events_smoke = PHASE_STEPS + PHASE_STEPS // MAINT_EVERY
        budget_norm = budget_norm * n_events_smoke / 416
    metrics["budget"] = {
        "form": f"BUDGET := {BUDGET_FRAC} x ||fact - base|| (the write's "
                "own norm, fp64 from the loaded fact); S_total = S_corpus "
                "+ S_maint BOTH counted (the dispatch: the maintenance "
                "steps' displacement counts against it)",
        "write_norm": write_norm,
        "budget_norm": budget_norm,
        "reservation": "EQUAL-SHARE over ALL remaining optimizer events, "
                       "BOTH streams (e286's corpus cap VERBATIM + the new "
                       "symmetric maintenance cap): corpus cap_t = "
                       "((BUDGET - S_total)/(rem_corpus + rem_maint)) / "
                       "||b_t||, lr_t = min(LR_STABLE x cosine, cap_t); "
                       "maintenance cap_m = ((BUDGET - S_total)/(1 + "
                       "rem_corpus_after + rem_maint_after)) / ||b_m||, "
                       "lr_m = min(MAINT_LR_BASE, cap_m) — EVERY event's "
                       "realized displacement fits its share: S_total <= "
                       "BUDGET by the triangle bound (e286's 89.5% blowout "
                       "structurally excluded); G_BUDGET checks the TOTAL "
                       "(non-halting: a failure would be an implementation "
                       "outcome to autopsy, routed to FAILS)",
        "slack": BUDGET_SLACK,
        "smoke_scaling": (f"SMOKE: budget x {PHASE_STEPS + PHASE_STEPS // MAINT_EVERY}"
                          "/416 — the full run's per-event share preserved"
                          if SMOKE else None),
    }
    log(f"P2b THE BUDGET: {BUDGET_FRAC} x {write_norm:.4f} = "
        f"{budget_norm:.4f} (the transport channel's cap; per-event share "
        f"~{budget_norm / (PHASE_STEPS + N_MAINT_EXPECTED):.5f})")
    write_partial("P2b the budget computed")

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
                ".../28501/28601/28701), its own generator; per corpus "
                "step the draws (aj_c(16), rj_c(32)) in that order; the "
                "first step's draws logged here from a scratch generator "
                "(the drivers' cgens start identically by construction); "
                "BOTH arms draw the IDENTICAL sequence",
        "seed": CORPUS_GEN_SEED,
        "first_step_aj": [int(x) for x in aj_probe.tolist()],
        "first_step_rj": [int(x) for x in rj_probe.tolist()],
        "composition": "16 original-host anchors (the 60-window bank) + 32 "
                       "random corpus windows (contiguous train_ids "
                       "slices); full-window CE; clip 1.0 -> the arm's own "
                       "step (RE-ANCHORED: g_perp + capped lr + opt_C + "
                       "the maintenance injection; TWIN: opt_C only)",
        "pass": True,
    }
    metrics["gates"]["G_CORPUSGEN"] = G_CORPUSGEN
    log(f"P2c G_CORPUSGEN: the corpus stream REGISTERED (seed "
        f"{CORPUS_GEN_SEED}; first draws aj[:4]="
        f"{G_CORPUSGEN['first_step_aj'][:4]} "
        f"rj[:4]={G_CORPUSGEN['first_step_rj'][:4]})")

    # ---- G_MAINTBIND: the maintenance mechanism's own bind ----------------
    isc = torch.Generator().manual_seed(INST_GEN_SEED)
    ix_probe = torch.randint(win_i.shape[0], (G1.NAME_BS,), generator=isc)
    n_name_tokens = int(inst_mask[:G1.NAME_BS].sum())
    G_MAINTBIND = {
        "form": "the maintenance mechanism's bind (THE THIRD FORM): ONE "
                "name-only gradient step after every M="
                f"{MAINT_EVERY} corpus steps — exactly "
                f"{N_MAINT_EXPECTED}; the batch := ix(16) TRUE install "
                "windows (win_i, G_NAMEWIN-gated) drawn from the "
                f"registered install generator seed {INST_GEN_SEED} (its "
                "own stream; the corpus stream untouched); THE NAME-ONLY "
                "CE := mean token-level CE over the masked NAME positions "
                "ONLY (7 x 16 = 112; the corpus half of e261's Dmix batch "
                "DROPPED — the pure name signal, no corpus dilution); "
                "clip 1.0; UNPROJECTED, applied through opt_F (the "
                "FACT-side SGD-M momentum 0.9 wd 0, private buffer) at "
                "the APPLIED lr_m = min(MAINT_LR_BASE, cap_m) — THE "
                "BUDGET-FITTING CAP: cap_m = share_m/||b_m|| with b_m = "
                "0.9*buf_F + g_name the EXACT pending in-step momentum "
                "(the event's displacement price lr_m x ||b_m||) and "
                "share_m = (BUDGET - S_total)/(1 + remaining corpus steps "
                "+ later maintenance events) — the SAME equal-share "
                "reservation the corpus cap uses, symmetric: every "
                "optimizer event's realized displacement fits its share "
                "and S_total <= BUDGET by the triangle bound (e286's "
                "89.5% blowout structurally excluded; the arithmetic: 16 "
                "steps x ~share ~0.011 = ~0.18 ~= 3.9% of budget)",
        "maint_every": MAINT_EVERY,
        "n_maint_expected": N_MAINT_EXPECTED,
        "maint_lr_base": MAINT_LR_BASE,
        "maint_lr_derivation": f"LR_SGD_matched/{MAINT_LR_DIV} (= "
                               f"{E273_LR_SGD}/{MAINT_LR_DIV}; == "
                               "LR_STABLE) as the SCHEDULE ceiling; the "
                               "applied lr is cap-fitted per event",
        "budget_fitting_cap": "cap_m = ((BUDGET - S_total)/(1 + rem_corpus "
                              "+ rem_maint))/||b_m||; lr_m = min("
                              "MAINT_LR_BASE, cap_m); b_m = 0.9*buf_F + "
                              "g_name (pending_buffer_sqnorm — the exact "
                              "SGD-M in-step buffer); the realized step "
                              "norm enters S_maint (the total accounting)",
        "inst_gen_seed": INST_GEN_SEED,
        "first_maint_draws": {"ix": [int(x) for x in ix_probe.tolist()]},
        "name_token_disclosure": {
            "name_masked_tokens_per_step": n_name_tokens,
            "corpus_tokens_per_step": 0,
            "note": "the PURE NAME SIGNAL: 112 masked name positions per "
                    "step, NO corpus tokens (e286's union CE carried 112 "
                    "vs 12240 — and its batch's masked positions were "
                    "host openings, not the name: the misbind disclosed "
                    "in G_NAMEWIN); the registered read: this gradient's "
                    "in-room fraction per step (expect >> the corpus "
                    "~0.06 — the pure name signal's geometry)"},
        "pass": bool(MAINT_EVERY == (2 if SMOKE else 25)
                     and N_MAINT_EXPECTED == PHASE_STEPS // MAINT_EVERY
                     and MAINT_LR_BASE == LR_SGD_MATCHED / 100
                     and MAINT_LR_BASE == LR_STABLE
                     and inst_mask.dtype == torch.bool
                     and n_name_tokens == G1.NAME_BS * len(G1.NAME)
                     and G_NAMEWIN["pass"]),
    }
    assert G_MAINTBIND["pass"], f"maintenance bind failed: {G_MAINTBIND}"
    metrics["gates"]["G_MAINTBIND"] = G_MAINTBIND
    log(f"P2c G_MAINTBIND: the maintenance REGISTERED (M={MAINT_EVERY} -> "
        f"{N_MAINT_EXPECTED} NAME-ONLY steps; MAINT_LR_BASE "
        f"{MAINT_LR_BASE!r} schedule + the budget-fitting cap; install "
        f"stream seed {INST_GEN_SEED}; the name tokens "
        f"{n_name_tokens}/step, corpus tokens 0/step — the pure name "
        f"signal; win_i G_NAMEWIN-bound): PASS")
    write_partial("P2c the generators + the maintenance bound")

    # ================= P3: ARM RE-ANCHORED (the active build) ===========
    log("=" * 78)
    log(f"ARM-{REA_ARM} — {ARM_DESC[REA_ARM]}")
    rea = chunked_namemaint_phase(
        f"{REA_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, budget_norm, win_i, inst_mask, anchor_full,
        train_ids, g0_ids, gm12_ids, r_eval_xy, zid,
        CKPT_DIR / ("smoke_e287_NAME-MAINTAINED_resume.pt" if SMOKE
                    else "e287_NAME-MAINTAINED_resume.pt"), dev)
    sd_r = rea["sd"]
    net_r = G1.evl_load(sd_r)
    cells_r = {"gm12": G1.battery_cell(net_r, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_r, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_r, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_r, *r_eval_xy)}
    d_final_r = flat_params_cpu(net_r) - fact_flat
    loads_final_r = rooms.displacement_loads(d_final_r, ROOM_MODE)
    del net_r
    corp_ce_r = [v["ce"] for v in rea["corpus_ledger"].values()]
    corp_gn_r = [v["gn_clipped"] for v in rea["corpus_ledger"].values()]
    corp_gin_r = [v["g_in_room_frac"] for v in rea["corpus_ledger"].values()
                  if v.get("g_in_room_frac") is not None]
    maint_lrs_r = [m["lr_maint"] for m in rea["maint_ledger"]]
    rea_ck = save_ckpt(
        "e287_NAME-MAINTAINED_post", sd_r,
        {"desc": "e287 ARM-NAME-MAINTAINED post-phase state: the "
                 "committed quiet-formed 10k fact + 400 budgeted "
                 "orthogonal SGD-M corpus steps (separate buffers, "
                 f"corpus stream {CORPUS_GEN_SEED}) + "
                 f"{rea['n_maint']} NAME-ONLY maintenance steps through "
                 f"opt_F at the budget-fitted lr_m (install stream "
                 f"{INST_GEN_SEED}; MAINT_LR_BASE {MAINT_LR_BASE!r} "
                 "schedule) — NO cons",
         "arm": REA_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "inst_gen_seed": INST_GEN_SEED,
         "maint_lr_base": MAINT_LR_BASE, "maint_every": MAINT_EVERY,
         "budget_norm": budget_norm, "S_corpus": rea["S_corpus"],
         "S_maint": rea["S_maint"],
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"] = {REA_ARM: {
        "desc": ARM_DESC[REA_ARM], "phase": {
            "traj": rea["traj"], "corpus_ledger": rea["corpus_ledger"],
            "orth_ledger": rea["orth_ledger"],
            "disp_ledger": rea["disp_ledger"],
            "buf_ledger": rea["buf_ledger"],
            "budget_ledger": rea["budget_ledger"],
            "lr_ledger": rea["lr_ledger"],
            "maint_ledger": rea["maint_ledger"],
            "window_ledger": rea["window_ledger"],
            "corpus_ce_median": med(corp_ce_r),
            "corpus_gn_clipped_median": med(corp_gn_r),
            "corpus_g_in_room_frac_median": med(corp_gin_r),
            "orth_max_rel_err": rea["orth_max"],
            "bufsep": rea["bufsep"],
            "budget": {"budget_norm": budget_norm,
                       "S_corpus_final": rea["S_corpus"],
                       "S_maint_final": rea["S_maint"],
                       "S_total_final": rea["S_corpus"] + rea["S_maint"],
                       "S_total_usage_frac": (rea["S_corpus"]
                                              + rea["S_maint"])
                       / budget_norm,
                       "cum_total_final_norm":
                           rea["disp_ledger"][-1]["total_disp_cum_norm"]
                           if rea["disp_ledger"] else None,
                       "n_capped": rea["n_capped"],
                       "n_steps": PHASE_STEPS,
                       "n_maint": rea["n_maint"],
                       "capped_frac": rea["n_capped"] / PHASE_STEPS,
                       "lr_applied_min": rea["lr_applied_min"],
                       "lr_applied_median": rea["lr_applied_median"],
                       "lr_sched_median": rea["lr_sched_median"],
                       "cap_min": rea["cap_min"],
                       "n_maint_capped": rea.get("n_maint_capped"),
                       "maint_lr_applied_median": (med(maint_lrs_r)
                                                   if maint_lrs_r else None),
                       "maint_lr_applied_min": (min(maint_lrs_r)
                                                if maint_lrs_r else None)},
            "chunk_table": rea["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_r,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_r.double().numpy())), **loads_final_r},
            "checkpoint": rea_ck,
            "resumed_final": bool(rea.get("resumed_final", False)),
        }}}
    log(f"ARM-{REA_ARM} DONE: post g0 {cells_r['g0']:.7f} "
        f"(x{cells_r['g0'] / FACT_BASELINE_G0:.4f}) g-12 {cells_r['gm12']:.7f} "
        f"CE_R {cells_r['ce_r']:.4f} | BUDGET S_corp {rea['S_corpus']:.4f} "
        f"+ S_maint {rea['S_maint']:.4f} = "
        f"{rea['S_corpus'] + rea['S_maint']:.4f}/{budget_norm:.4f} "
        f"({(rea['S_corpus'] + rea['S_maint']) / budget_norm:.1%}) | "
        f"maint {rea['n_maint']} steps | capped {rea['n_capped']}/"
        f"{PHASE_STEPS} | ORTH max {rea['orth_max']:.2e} | isolation "
        f"{rea['bufsep']['corpus_checks']}+{rea['bufsep']['maint_checks']} "
        f"checks {rea['bufsep']['corpus_violations'] + rea['bufsep']['maint_violations']} "
        f"violations | corpus CE med {med(corp_ce_r):.4f}")
    write_partial(f"ARM-{REA_ARM} complete (the active build's ledgers)")

    # ================= P4: ARM SANCTUARY-TWIN (e285's form) =============
    E261.burst_cooldown(f"{REA_ARM} -> {TWN_ARM}")
    log("=" * 78)
    log(f"ARM-{TWN_ARM} — {ARM_DESC[TWN_ARM]}")
    twn = chunked_sanctuary_twin_phase(
        f"{TWN_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, budget_norm, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e287_SANCTUARY-TWIN_resume.pt" if SMOKE
                    else "e287_SANCTUARY-TWIN_resume.pt"), dev)
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
    twn_ck = save_ckpt(
        "e287_SANCTUARY-TWIN_post", sd_t,
        {"desc": "e287 ARM-SANCTUARY-TWIN post-phase state: the committed "
                 "quiet-formed 10k fact + 400 budgeted orthogonal SGD-M "
                 "corpus steps (e285's passive stack VERBATIM, corpus "
                 f"stream {CORPUS_GEN_SEED}) — the same-session control",
         "arm": TWN_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "budget_norm": budget_norm, "S": twn["S"],
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
            "corpus_ce_median": med(corp_ce_t),
            "corpus_gn_clipped_median": med(corp_gn_t),
            "corpus_g_in_room_frac_median": med(corp_gin_t),
            "orth_max_rel_err": twn["orth_max"],
            "bufsep": twn["bufsep"],
            "budget": {"budget_norm": budget_norm, "S_final": twn["S"],
                       "S_usage_frac": twn["S"] / budget_norm,
                       "cum_final_norm":
                           twn["disp_ledger"][-1]["corpus_disp_cum_norm"]
                           if twn["disp_ledger"] else None,
                       "n_capped": twn["n_capped"],
                       "n_steps": PHASE_STEPS,
                       "capped_frac": twn["n_capped"] / PHASE_STEPS,
                       "lr_applied_min": twn["lr_applied_min"],
                       "lr_applied_median": twn["lr_applied_median"],
                       "lr_sched_median": twn["lr_sched_median"],
                       "cap_min": twn["cap_min"]},
            "chunk_table": twn["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_t,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_t.double().numpy())), **loads_final_t},
            "checkpoint": twn_ck,
            "resumed_final": bool(twn.get("resumed_final", False)),
        }}
    log(f"ARM-{TWN_ARM} DONE: post g0 {cells_t['g0']:.7f} "
        f"(x{cells_t['g0'] / FACT_BASELINE_G0:.4f}) g-12 {cells_t['gm12']:.7f} "
        f"CE_R {cells_t['ce_r']:.4f} | BUDGET S {twn['S']:.4f}/"
        f"{budget_norm:.4f} ({twn['S'] / budget_norm:.1%}) | capped "
        f"{twn['n_capped']}/{PHASE_STEPS} | corpus CE med {med(corp_ce_t):.4f} "
        f"(e285's committed sanctuary: post {E285_SAN_POST:.2e} = x"
        f"{E285_SAN_RATIO:.4f}, S {E285_S_FINAL:.4f})")
    write_partial(f"ARM-{TWN_ARM} complete (the same-session passive pair)")

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
        f"bit-identical draws — the arms' ONLY delta is the maintenance "
        f"mechanism)")

    # ================= P5: the instantiated gates ========================
    # G_ORTH — the corpus orthogonality gate (NON-HALTING: a failure
    # routes to MAINTENANCE-FAILS, per the dispatch's gates clause)
    orth_max_r = rea["orth_max"]
    G_ORTH = {
        "form": "the stepped CORPUS gradient is ENTIRELY ORTHOGONAL to the "
                "room on BOTH arms: max over ALL corpus steps of "
                "||P_room g_perp|| / ||g_perp|| < 1e-6 (checked EVERY "
                "corpus step; the SRCT projector is exact, so any drift "
                "above fp64 roundoff is an implementation bug). The "
                "MAINTENANCE gradient is deliberately UNPROJECTED (the "
                "fact's own stream — the re-anchoring; disclosed at "
                "birth). NON-HALTING here (a failure routes the verdict "
                "to FAILS — the dispatch's gates clause)",
        "re_anchored_orth_max_rel_err": orth_max_r,
        "twin_orth_max_rel_err": twn["orth_max"],
        "bar": ORTH_BAR,
        "n_steps_checked": PHASE_STEPS,
        "pass": bool(orth_max_r is not None and orth_max_r < ORTH_BAR
                     and twn["orth_max"] < ORTH_BAR),
    }
    metrics["gates"]["G_ORTH"] = G_ORTH
    if not G_ORTH["pass"]:
        log(f"  G_ORTH FAILED (rea {orth_max_r:.2e} / twn "
            f"{twn['orth_max']:.2e} >= {ORTH_BAR:.0e}) — NON-HALTING: the "
            f"verdict routes to FAILS (the gates clause)")

    # G_BUFSEP — the bidirectional separation's machine verification
    bufC_fr = [b["bufC_inroom_frac"] for b in rea["buf_ledger"]
               if b.get("bufC_inroom_frac") is not None]
    bufF_fr = [b["bufF_inroom_frac"] for b in rea["buf_ledger"]
               if b.get("bufF_inroom_frac") is not None]
    optF_steps_ledger = [b.get("optF_steps") for b in rea["buf_ledger"]]
    bufC_fr_t = [b["bufC_inroom_frac"] for b in twn["buf_ledger"]
                 if b.get("bufC_inroom_frac") is not None]
    optF_entries_t = [b.get("optF_state_entries") for b in twn["buf_ledger"]]
    G_BUFSEP = {
        "form": "the buffer separation v2 (the dispatch's 'the maintenance "
                "steps enter through the fact side ONLY'): (i) "
                "BIDIRECTIONAL ISOLATION on the RE-ANCHORED arm — around "
                "EVERY optimizer event BOTH optimizers' buffer states are "
                "snapshotted bitwise and compared after; the stepped one "
                "may change, the other must be bitwise untouched (any "
                "mismatch RAISES — a hard HALT); the twin keeps e285's "
                "unidirectional probe (opt_F must stay EMPTY); (ii) "
                "COMPOSITION — per milestone, ||P_room buf_C||/||buf_C|| "
                "< 1e-4 on both arms (the fp-floor bar; NON-HALTING — a "
                "failure routes to FAILS); buf_F's composition "
                "DISCLOSED (the fact side carries the install stream's "
                "momentum — its in-room fraction is the mechanism's own "
                "direction read, never a bar); (iii) opt_F stepped "
                "EXACTLY at the maintenance steps (machine-counted)",
        "re_anchored": {
            "corpus_checks": rea["bufsep"]["corpus_checks"],
            "corpus_violations": rea["bufsep"]["corpus_violations"],
            "maint_checks": rea["bufsep"]["maint_checks"],
            "maint_violations": rea["bufsep"]["maint_violations"],
            "optF_steps": rea["bufsep"]["optF_steps"],
            "n_maint": rea["n_maint"],
            "optF_steps_equals_n_maint":
                bool(rea["bufsep"]["optF_steps"] == rea["n_maint"]),
            "bufC_inroom_frac_max": max(bufC_fr) if bufC_fr else None,
            "bufF_inroom_frac_first": bufF_fr[0] if bufF_fr else None,
            "bufF_inroom_frac_last": bufF_fr[-1] if bufF_fr else None,
        },
        "twin": {
            "isolation_checks": twn["bufsep"]["isolation_checks"],
            "isolation_violations": twn["bufsep"]["isolation_violations"],
            "optF_state_entries_max": max(optF_entries_t)
            if optF_entries_t else 0,
            "optF_ever_stepped": bool(
                twn["bufsep"].get("optF_ever_stepped", False)),
            "bufC_inroom_frac_max": max(bufC_fr_t) if bufC_fr_t else None,
        },
        "bufC_bar": BUFSEP_ORTH_BAR,
        "e284_sep_cite": E284_SEP_PRIMARY,
        "pass": None,           # computed explicitly below (a hard gate)
    }
    G_BUFSEP["pass"] = bool(
        rea["bufsep"]["corpus_violations"] == 0
        and rea["bufsep"]["maint_violations"] == 0
        and rea["bufsep"]["corpus_checks"] == PHASE_STEPS
        and rea["bufsep"]["maint_checks"] == rea["n_maint"]
        and rea["bufsep"]["optF_steps"] == rea["n_maint"]
        and rea["n_maint"] == N_MAINT_EXPECTED
        and twn["bufsep"]["isolation_violations"] == 0
        and twn["bufsep"]["isolation_checks"] == PHASE_STEPS
        and max(optF_entries_t if optF_entries_t else [0]) == 0
        and not twn["bufsep"].get("optF_ever_stepped", False)
        and bufC_fr and max(bufC_fr) < BUFSEP_ORTH_BAR
        and bufC_fr_t and max(bufC_fr_t) < BUFSEP_ORTH_BAR)
    assert G_BUFSEP["pass"], f"buffer separation gate FAILED: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    # G_BUDGET — the TOTAL budget's machine verification (NON-HALTING)
    S_cor_final = float(rea["S_corpus"])
    S_m_final = float(rea["S_maint"])
    S_tot_final = S_cor_final + S_m_final
    cum_total_final = (rea["disp_ledger"][-1]["total_disp_cum_norm"]
                       if rea["disp_ledger"] else None)
    drift_final_r = float(np.linalg.norm(d_final_r.double().numpy()))
    S_twn_final = float(twn["S"])
    cum_twn_final = (twn["disp_ledger"][-1]["corpus_disp_cum_norm"]
                     if twn["disp_ledger"] else None)
    G_BUDGET = {
        "form": "the TOTAL displacement budget's machine verification "
                "(the dispatch: 'the budget blown again' adjudicates): "
                "S_total = S_corpus + S_maint <= BUDGET + slack AND the "
                "realized total cumulative displacement ||theta_400 - "
                "theta_fact|| <= S_total; BOTH streams enter the "
                "equal-share cap (the corpus cap e286-verbatim + the "
                "maintenance cap on the pending momentum b_m) so every "
                "event's realized displacement fits its reserved share — "
                "S_total <= BUDGET by the triangle bound BY CONSTRUCTION; "
                "a failure here is an implementation outcome to autopsy "
                "(routed to FAILS, never a HALT); the SPLIT disclosed",
        "budget_norm": budget_norm,
        "budget_frac_of_write_norm": BUDGET_FRAC,
        "write_norm": write_norm,
        "re_anchored": {
            "S_corpus_final": S_cor_final,
            "S_maint_final": S_m_final,
            "S_total_final": S_tot_final,
            "S_total_usage_frac": S_tot_final / budget_norm,
            "S_split_frac_corpus": (S_cor_final / S_tot_final
                                    if S_tot_final > 0 else None),
            "S_split_frac_maint": (S_m_final / S_tot_final
                                   if S_tot_final > 0 else None),
            "cum_disp_total_final_norm": cum_total_final,
            "cum_total_usage_frac": (cum_total_final / budget_norm
                                     if cum_total_final is not None
                                     else None),
            "drift_from_fact_final_norm": drift_final_r,
            "n_capped": rea["n_capped"], "n_steps": PHASE_STEPS,
            "capped_frac": rea["n_capped"] / PHASE_STEPS,
            "n_maint_capped": rea.get("n_maint_capped"),
            "maint_lr_applied_median": (med(maint_lrs_r)
                                        if maint_lrs_r else None),
            "lr_price": {"lr_applied_median": rea["lr_applied_median"],
                         "lr_applied_min": rea["lr_applied_min"],
                         "lr_sched_median": rea["lr_sched_median"]}},
        "twin": {"S_final": S_twn_final,
                 "S_usage_frac": S_twn_final / budget_norm,
                 "cum_final_norm": cum_twn_final,
                 "n_capped": twn["n_capped"]},
        "e285_contrast": {
            "sanctuary_S_final": E285_S_FINAL,
            "sanctuary_budget_norm": E285_BUDGET_NORM,
            "sanctuary_drift": E285_SAN_DRIFT,
            "note": "the passive parent: S 100.00000002% of budget, cum "
                    "drift 1.845, ratio 0.0030x — the coupling-constant "
                    "cite"},
        "e286_contrast": {
            "S_maint_final": E286_S_MAINT,
            "maint_budget_share": E286_MAINT_BUDGET_SHARE,
            "note": "the active parent's UNCAPPED first form: the "
                    "maintenance momentum alone ate 89.5% of budget — the "
                    "budget-fitting cap is THIS cell's answer (and that "
                    "parent's batch was misbound: host openings, no name "
                    "— see G_NAMEWIN)"},
        "slack": BUDGET_SLACK,
        "pass": bool(S_tot_final <= budget_norm + BUDGET_SLACK
                     and cum_total_final is not None
                     and cum_total_final <= budget_norm + BUDGET_SLACK
                     and S_twn_final <= budget_norm + BUDGET_SLACK),
    }
    metrics["gates"]["G_BUDGET"] = G_BUDGET
    if not G_BUDGET["pass"]:
        log(f"  G_BUDGET FAILED (S_total {S_tot_final:.4f} or cum "
            f"{cum_total_final} > {budget_norm:.4f} + slack) — NON-HALTING:"
            f" the verdict routes to FAILS (the budget-blown-again clause)")
    log(f"P5 GATES: G_ORTH {'PASS' if G_ORTH['pass'] else 'FAIL'} (rea max "
        f"{orth_max_r:.2e}, twn max {twn['orth_max']:.2e} vs "
        f"{ORTH_BAR:.0e} over {PHASE_STEPS} steps); G_BUFSEP PASS "
        f"(bidirectional: {rea['bufsep']['corpus_checks']} corpus + "
        f"{rea['bufsep']['maint_checks']} maint checks, 0 violations; "
        f"opt_F stepped exactly {rea['bufsep']['optF_steps']}x == "
        f"{rea['n_maint']} maintenance steps; bufC in-room max "
        f"{(max(bufC_fr) if bufC_fr else float('nan')):.2e}; bufF in-room "
        f"disclosed {(bufF_fr[-1] if bufF_fr else float('nan')):.4f}); "
        f"G_BUDGET {'PASS' if G_BUDGET['pass'] else 'FAIL'} (S_corp "
        f"{S_cor_final:.4f} + S_maint {S_m_final:.4f} = {S_tot_final:.4f} "
        f"vs {budget_norm:.4f}; twin S {S_twn_final:.4f})")
    write_partial("P5 gates complete (ORTH + BUFSEP v2 + BUDGET total)")

    # ================= P6: ADJUDICATION (the frozen bars) ================
    ratio_400 = cells_r["g0"] / FACT_BASELINE_G0
    ratio_session_denom = cells_r["g0"] / max(fact_g0, RATIO_DEN_FLOOR)
    twin_ratio = cells_t["g0"] / FACT_BASELINE_G0
    ratio_mile = {t["step"]: t["g0_pz"] / FACT_BASELINE_G0
                  for t in rea["traj"]}
    # "the stream live (the corpus CE improving or stable)": medians of the
    # logged corpus-batch CE rows t in (300,400] vs t in [1,100]; improving
    # := late < early; stable := late <= 1.05 x early (frozen at birth)
    ce_steps = sorted(int(k) for k in rea["corpus_ledger"].keys())
    ce_early = [rea["corpus_ledger"][str(k)]["ce"] if str(k) in
                rea["corpus_ledger"] else rea["corpus_ledger"][k]["ce"]
                for k in ce_steps if k <= (8 if SMOKE else 100)]
    ce_late = [rea["corpus_ledger"][str(k)]["ce"] if str(k) in
               rea["corpus_ledger"] else rea["corpus_ledger"][k]["ce"]
               for k in ce_steps if k > (3 if SMOKE else 300)]
    ce_early_med = med(ce_early) if ce_early else None
    ce_late_med = med(ce_late) if ce_late else None
    ce_improving = bool(ce_early_med is not None and ce_late_med is not None
                        and ce_late_med < ce_early_med)
    ce_stable = bool(ce_early_med is not None and ce_late_med is not None
                     and ce_late_med <= STREAM_STABLE_TOL * ce_early_med)
    stream_live = bool(ce_improving or ce_stable)
    # the LIFTS read: every sampled window's post-maintenance read exceeds
    # its pre-maintenance read (x > 1 at BOTH windows = "consistent lifts")
    sawtooth = {}
    for (a, b) in ((WIN1, f"win1_t{WIN1[1]}"), (WIN2, f"win2_t{WIN2[1]}")):
        pre = next((w for w in rea["window_ledger"]
                    if w["step"] == a[1] and "pre" in w["window_phase"]),
                   None)
        post = next((w for w in rea["window_ledger"]
                     if w["step"] == a[1] and "post" in w["window_phase"]),
                    None)
        if pre is not None and post is not None:
            sawtooth[b] = {
                "pre_g0": pre["g0_pz"], "post_g0": post["g0_pz"],
                "lift_ratio": (post["g0_pz"] / pre["g0_pz"]
                               if pre["g0_pz"] > 0 else None)}

    def _saw_txt(saw: dict) -> str:
        if not saw:
            return "n/a"
        return "; ".join(
            (f"{k} x{v['lift_ratio']:.3f}"
             if v["lift_ratio"] is not None else f"{k} n/a")
            for k, v in saw.items())

    lifts_consistent = bool(sawtooth and all(
        v["lift_ratio"] is not None and v["lift_ratio"] > 1.0
        for v in sawtooth.values()))
    m_ir_steps = [mm["g_in_room_frac"] for mm in rea["maint_ledger"]
                  if mm.get("g_in_room_frac") is not None]
    m_ir_med = med(m_ir_steps) if m_ir_steps else None
    lr_m_steps = [mm["lr_maint"] for mm in rea["maint_ledger"]]
    lr_m_med = med(lr_m_steps) if lr_m_steps else None

    hard = dict(metrics["gates"])
    halt_gates_pass = bool(all(
        g.get("pass") for k, g in hard.items()
        if k not in ("G_ORTH", "G_BUDGET")))
    orth_held = bool(G_ORTH["pass"])
    budget_held = bool(G_BUDGET["pass"])

    if not halt_gates_pass:
        failed = [k for k, g in hard.items()
                  if k not in ("G_ORTH", "G_BUDGET") and not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard bind/isolation gate failed — nothing adjudicated; "
                  "the record is complete for the autopsy")
    elif not budget_held or not orth_held:
        verdict = "FAILS"
        which = []
        if not budget_held:
            which.append(
                f"the budget blown again: S_total {S_tot_final:.4f} "
                f"(S_corpus {S_cor_final:.4f} + S_maint {S_m_final:.4f}) vs "
                f"BUDGET {budget_norm:.4f} + {BUDGET_SLACK:.0e} slack; "
                f"realized cum {cum_total_final} — the budget-fitting cap "
                f"failed to contain the momentum (an implementation "
                f"outcome to autopsy)")
        if not orth_held:
            which.append(f"the ORTHOGONALITY gate broken (rea "
                         f"{orth_max_r:.2e} / twn {twn['orth_max']:.2e} vs "
                         f"{ORTH_BAR:.0e})")
        clause = (f"ratio_400 {ratio_400:.4f}x — the dispatch's FAILS "
                  f"clause: {' AND '.join(which)}; the trajectories "
                  f"verbatim")
    elif ratio_400 >= SURVIVE_FRAC and stream_live:
        verdict = "NAME-MAINTAINED-HOLDS"
        clause = (f">= {SURVIVE_FRAC:.0%}x survival at t400 (ratio "
                  f"{ratio_400:.4f}x) with the budget HELD (S_total "
                  f"{S_tot_final:.4f} = {S_tot_final / budget_norm:.1%} of "
                  f"{budget_norm:.4f}: corpus {S_cor_final:.4f} + "
                  f"maintenance {S_m_final:.4f}) and the stream live "
                  f"(corpus CE {ce_early_med:.4f} -> {ce_late_med:.4f}, "
                  f"{'improving' if ce_improving else 'stable'}) — THE "
                  f"FIRST ACTIVE SURVIVAL: a memory maintained by its own "
                  f"name signal under traffic. The name-only gradient's "
                  f"in-room fraction median {m_ir_med} (vs the corpus "
                  f"~0.06 and e286's misbound 0.0597-0.0600); the lifts "
                  f"{_saw_txt(sawtooth)}; the same-session passive twin "
                  f"died at {twin_ratio:.4f}x on bit-identical corpus "
                  f"draws; e286's misbound first form x{E286_REA_RATIO:.4f}")
    elif ratio_400 >= SURVIVE_FRAC and not stream_live:
        verdict = "MIXED"
        clause = (f"ratio_400 {ratio_400:.4f}x >= the 0.5x bar with the "
                  f"budget held BUT the stream NOT live (corpus CE "
                  f"{ce_early_med:.4f} -> {ce_late_med:.4f}; neither "
                  f"improving nor stable within {STREAM_STABLE_TOL:.2f}x) — "
                  f"the HOLDS clause's stream-live condition unmet; the "
                  f"trajectories verbatim, every ledger, no inflation")
    elif ratio_400 >= PARTIAL_FRAC and lifts_consistent:
        verdict = "SAWTOOTH-CONFIRMED"
        clause = (f"0.05x-0.5x survival at t400 (ratio {ratio_400:.4f}x) "
                  f"with the windows showing consistent LIFTS "
                  f"({_saw_txt(sawtooth)} — every sampled window x>1), the "
                  f"budget held (S_total {S_tot_final:.4f} = "
                  f"{S_tot_final / budget_norm:.1%}: corpus "
                  f"{S_cor_final:.4f} + maint {S_m_final:.4f}) and the "
                  f"stream live — the mechanism real (the name signal "
                  f"lifts the read under traffic), the dosage still short "
                  f"(applied lr_m median {lr_m_med:.6f} vs the "
                  f"{MAINT_LR_BASE:.4f} schedule; per-step displacement ~"
                  f"{S_m_final / max(rea['n_maint'], 1):.4f}; S_maint "
                  f"{S_m_final / budget_norm:.1%} of budget); the "
                  f"name-only gradient's in-room fraction median "
                  f"{m_ir_med} vs the corpus ~0.06 — THE PURE NAME "
                  f"SIGNAL'S GEOMETRY; the twin {twin_ratio:.4f}x, e286's "
                  f"misbound first form x{E286_REA_RATIO:.4f}; the "
                  f"trajectory + windows verbatim")
    elif ratio_400 >= PARTIAL_FRAC:
        verdict = "FAILS"
        clause = (f"0.05x-0.5x survival at t400 (ratio {ratio_400:.4f}x) "
                  f"with the budget held BUT the lifts absent "
                  f"({_saw_txt(sawtooth)} — NOT every sampled window x>1) "
                  f"— the dispatch's FAILS clause (the lifts absent); the "
                  f"trajectory + windows verbatim. The name-only "
                  f"gradient's in-room fraction median {m_ir_med} (vs the "
                  f"corpus ~0.06); the dose: lr_m median {lr_m_med:.6f}, "
                  f"S_maint {S_m_final:.4f} = {S_m_final / budget_norm:.1%} "
                  f"of budget across {rea['n_maint']} steps")
    else:
        verdict = "FAILS"
        m_ir_cum = (rea["disp_ledger"][-1]["maint_disp_cum_in_room_frac"]
                    if rea["disp_ledger"] else None)
        if m_ir_cum is not None and m_ir_cum >= 0.2:
            direction = (f"substantially IN-ROOM (cum {m_ir_cum:.4f}, "
                         f"per-step median {m_ir_med}) — the name signal's "
                         f"geometry is real (>> the corpus ~0.06): the "
                         f"re-anchoring direction is right, the "
                         f"budget-fitted dose too small or too rare")
        else:
            direction = (f"still mostly OUT-OF-ROOM (cum {m_ir_cum}, "
                         f"per-step median {m_ir_med}) — even the PURE "
                         f"name gradient at the established state points "
                         f"out-of-room: the obstacle is deeper than token "
                         f"mass")
        clause = (f"< {PARTIAL_FRAC:.0%}x survival at t400 (ratio "
                  f"{ratio_400:.4f}x) with the gates held (orth "
                  f"{orth_max_r:.1e}; budget S_total {S_tot_final:.4f} = "
                  f"{S_tot_final / budget_norm:.1%} of {budget_norm:.4f}) — "
                  f"the trajectories verbatim. THE SHAPE: the windows' "
                  f"lifts {_saw_txt(sawtooth)}; the maintenance "
                  f"displacement {direction}; the applied lr_m median "
                  f"{lr_m_med:.6f} (cap-bound), per-step displacement ~"
                  f"{S_m_final / max(rea['n_maint'], 1):.4f}; the twin "
                  f"{twin_ratio:.4f}x, e286's misbound first form "
                  f"x{E286_REA_RATIO:.4f}, the passive parent "
                  f"x{E285_SAN_RATIO:.4f}")

    log("=" * 78)
    log(f"E287 VERDICT: {verdict}")
    log(f"  NAME-MAINTAINED: post g0 {cells_r['g0']:.8f} (x{ratio_400:.4f} "
        f"committed-denom; x{ratio_session_denom:.4f} session-denom)")
    log(f"  SANCTUARY-TWIN: post g0 {cells_t['g0']:.8f} (x{twin_ratio:.4f}; "
        f"e285's committed sanctuary x{E285_SAN_RATIO:.4f})")
    log(f"  milestones (NAME-MAINTAINED): "
        + "; ".join(f"t{t} x{r:.4f}" for t, r in ratio_mile.items()))
    log(f"  the budget: S_corp {S_cor_final:.4f} + S_maint {S_m_final:.4f} "
        f"= {S_tot_final:.4f}/{budget_norm:.4f} "
        f"({S_tot_final / budget_norm:.1%}); the twin's S "
        f"{S_twn_final:.4f}; the maint lr_m median {lr_m_med}")
    log(f"  the windows: "
        + "; ".join(f"{k}: pre {v['pre_g0']:.6f} -> post {v['post_g0']:.6f} "
                    f"(x{v['lift_ratio']:.3f})"
                    for k, v in sawtooth.items()))
    log(f"  the name-only gradient's in-room fraction: per-step median "
        f"{m_ir_med} (the corpus ~0.06; e286's misbound 0.0597-0.0600)")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any bind/isolation hard-gate "
                           "failure — HALT) -> FAILS-by-gates (G_BUDGET "
                           "blown OR G_ORTH failed, any ratio — 'the "
                           "budget blown again') -> NAME-MAINTAINED-HOLDS "
                           "(ratio_400 >= 0.5 AND budget held AND stream "
                           "live) -> MIXED-if-stream-dead (ratio_400 >= "
                           "0.5 AND budget held AND NOT live) -> SAWTOOTH-"
                           "CONFIRMED (0.05 <= ratio_400 < 0.5 AND budget "
                           "held AND every window lift > 1) -> FAILS-by-"
                           "lifts (0.05 <= ratio_400 < 0.5 AND budget held "
                           "AND a lift absent) -> FAILS-by-ratio "
                           "(ratio_400 < 0.05 AND gates held) -> MIXED "
                           "(everything else) — frozen at birth",
        "gates_pass": bool(all(g.get("pass") for g in hard.values())),
        "halt_gates_pass": halt_gates_pass,
        "reads": {
            REA_ARM: {"post_g0": cells_r["g0"],
                      "survival_ratio_committed_denom": ratio_400,
                      "survival_ratio_session_denom": ratio_session_denom,
                      "post_gm12": cells_r["gm12"],
                      "post_ce_r": cells_r["ce_r"],
                      "traj_g0": {t["step"]: t["g0_pz"]
                                  for t in rea["traj"]},
                      "survival_ratio_milestones": ratio_mile,
                      "corpus_ce_median": med(corp_ce_r),
                      "corpus_ce_early_median": ce_early_med,
                      "corpus_ce_late_median": ce_late_med,
                      "ce_improving": ce_improving,
                      "ce_stable_1p05": ce_stable,
                      "stream_live": stream_live,
                      "corpus_g_in_room_frac_median": med(corp_gin_r),
                      "maint_g_in_room_frac_per_step": m_ir_steps,
                      "maint_g_in_room_frac_median": m_ir_med,
                      "maint_lr_applied": {"median": lr_m_med,
                                           "min": (min(lr_m_steps)
                                                   if lr_m_steps else None),
                                           "schedule": MAINT_LR_BASE,
                                           "n_capped":
                                               rea.get("n_maint_capped")},
                      "drift_ledger": rea["disp_ledger"],
                      "budget_ledger": rea["budget_ledger"],
                      "buf_ledger": rea["buf_ledger"],
                      "maint_ledger": rea["maint_ledger"],
                      "window_ledger": rea["window_ledger"],
                      "sawtooth": sawtooth,
                      "lifts_consistent": lifts_consistent,
                      "orth_ledger_max": orth_max_r,
                      "drift_from_fact_final_in_room":
                          loads_final_r["in_own_room"]},
            TWN_ARM: {"post_g0": cells_t["g0"],
                      "survival_ratio": twin_ratio,
                      "post_gm12": cells_t["gm12"],
                      "post_ce_r": cells_t["ce_r"],
                      "traj_g0": {t["step"]: t["g0_pz"]
                                  for t in twn["traj"]},
                      "corpus_ce_median": med(corp_ce_t),
                      "drift_ledger": twn["disp_ledger"],
                      "drift_from_fact_final_norm":
                          float(np.linalg.norm(d_final_t.double().numpy())),
                      "drift_from_fact_final_in_room":
                          loads_final_t["in_own_room"]},
            "the_passive_reference_e285": {
                "post_g0": E285_SAN_POST, "ratio": E285_SAN_RATIO,
                "drift": E285_SAN_DRIFT, "S_final": E285_S_FINAL,
                "budget_norm": E285_BUDGET_NORM,
                "note": "the passive parent (hard-bound in G_PARENTS); "
                        "this session's twin replicates the FORM on its "
                        "own draw stream"},
            "the_active_reference_e286": {
                "post_g0": E286_REA_POST, "ratio": E286_REA_RATIO,
                "S_maint_final": E286_S_MAINT,
                "maint_budget_share": E286_MAINT_BUDGET_SHARE,
                "win1_lift": E286_WIN1_LIFT, "win2_lift": E286_WIN2_LIFT,
                "maint_in_room_frac_range":
                    [E286_MAINT_INROOM_FIRST, E286_MAINT_INROOM_LAST],
                "note": "the active parent's MISBOUND first form (host "
                        "openings at the masked positions — the name "
                        "signal was never injected; see G_NAMEWIN): its "
                        "numbers are the corpus-text step's numbers"},
            "the_two_gates": {
                "orthogonality": {
                    "re_anchored_max": orth_max_r,
                    "twin_max": twn["orth_max"], "bar": ORTH_BAR,
                    "verdict": ("HELD" if orth_held else "BROKEN")},
                "budget": {
                    "budget_norm": budget_norm, "S_corpus": S_cor_final,
                    "S_maint": S_m_final, "S_total": S_tot_final,
                    "cum_total": cum_total_final,
                    "twin_S": S_twn_final,
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
            "corpus draws (one generator, seed 28701, one draw order) — "
            "the ONLY delta is the maintenance mechanism (16 NAME-ONLY "
            "gradient steps through the fact-side SGD-M at the budget-"
            "fitted lr_m). The rooms are identical across arms BY "
            "CONSTRUCTION (one bit-gated room)."),
        "the_maintenance_disclosure": (
            "the maintenance step is the PURE NAME SIGNAL: the mean CE on "
            "the 112 masked ZEPHYRA positions of TRUE install windows "
            "(win_i, decode-gated), unprojected, at the budget-fitted "
            "lr_m — THE NAME SIGNAL'S FIRST INJECTION (the birth-day "
            "correction: e286's maintenance batch was misbound to host "
            "openings, so its 'name' term carried no name; its flat "
            "in-room ~0.06 was a corpus-text step's reading). THIS "
            "gradient's per-step in-room fraction + the window shape are "
            "the mechanism's own direction read, measured never assumed"),
        "the_budget_disclosure": (
            "the maintenance displacement counts against the same 0.5x "
            "budget (the dispatch) and BOTH streams now enter the "
            "equal-share cap: the corpus cap (e286-verbatim) + the "
            "maintenance cap on the pending momentum b_m — every event's "
            "realized displacement fits its reserved share, so the budget "
            "holds by construction (e286's 89.5% blowout structurally "
            "excluded); the split ledger discloses who consumed what; a "
            "blown budget would be an implementation outcome to autopsy, "
            "routed to FAILS, never a HALT"),
        "the_separation_disclosure": (
            "the fact side's buffer is the maintenance instrument (its "
            "own momentum, persistent across maintenance steps); the "
            "isolation is BIDIRECTIONAL and machine-checked bitwise "
            "around EVERY optimizer event; buf_F's composition is "
            "disclosed per milestone (never a bar) — the corpus buffer "
            "stays at the fp floor (< 1e-4)"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; nothing guaranteed"),
        "loads_measured_not_nominal": (
            "every read is measured: the per-event realized displacement "
            "(corpus and maintenance separately), the S split, the "
            "per-milestone interval + cumulative projections (fp64), the "
            "buffer-composition ledgers, the orthogonality ledger (every "
            "step), the maintenance ledger (every step), the corpus CE + "
            "clipped-grad ledgers, the between-steps window reads — "
            "never nominal"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "branches and the trajectories are reported "
                               "verbatim regardless"),
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
                         "note": "read-only here — the established fact"},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
            "rooms": rooms_ck,
            "post_states": {REA_ARM: rea_ck, TWN_ARM: twn_ck},
        },
        "machinery": {
            "namemaint_phase": "THIS file's chunked_namemaint_phase (the "
                              "third form): e286's reanchor driver with "
                              "the TWO changed dials (the NAME-ONLY CE on "
                              "win_i's 112 name positions + the "
                              "BUDGET-FITTING cap on the pending momentum)"
                              " + bidirectional isolation + the "
                              "total-budget accounting + the window reads",
            "twin_phase": "THIS file's chunked_sanctuary_twin_phase: "
                          "e285's chunked_sanctuary_phase VERBATIM in body "
                          "(the same-session passive control)",
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
    make_namemaint_plot(RD, metrics, verdict, clause, thermal_log,
                       budget_norm, write_norm)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e287_name_maintained.png"),
                          str(RD / "REPORT.md")]
    write_partial("P8 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_namemaint_plot(rd, metrics, verdict, clause, thermal_log,
                       budget_norm, write_norm):
    """THE CELL'S HEADLINE FIGURE: the survival curves (the name-"
    "maintained build vs the passive twin vs the references + bars), THE "
    "BETWEEN-STEPS WINDOW (the sawtooth), the budget split (S_corpus + "
    "S_maint vs the budget), the channels' in-room ledgers (corpus vs the "
    "name-only maintenance), the corpus CE + lr price, the thermal "
    "envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    rea = metrics["arms"][REA_ARM]["phase"]
    twn = metrics["arms"][TWN_ARM]["phase"]
    rt, tt = rea["traj"], twn["traj"]

    # (0,0) THE SURVIVAL CURVES
    ax = axes[0, 0]
    ax.plot([t["step"] for t in rt], [max(t["g0_pz"], 1e-7) for t in rt],
            "o-", lw=1.9, ms=5, color="tab:green",
            label=f"{REA_ARM} (sanctuary + NAME-ONLY maintenance)")
    ax.plot([t["step"] for t in tt], [max(t["g0_pz"], 1e-7) for t in tt],
            "s--", lw=1.6, ms=4.5, color="tab:blue",
            label=f"{TWN_ARM} (e285's passive stack, this session)")
    ax.axhline(FACT_BASELINE_G0, color="black", ls=":", lw=1.3,
               label=f"the loaded fact {FACT_BASELINE_G0:.4f}")
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.4, label=f"the {SURVIVE_FRAC:.0%}x HOLDS bar")
    ax.axhline(PARTIAL_FRAC * FACT_BASELINE_G0, color="darkorange",
               ls="--", lw=1.1, label=f"the {PARTIAL_FRAC:.0%}x PARTIAL bar")
    ax.axhline(E285_SAN_POST, color="tab:purple", ls="-.", lw=1.0,
               label=f"e285's committed sanctuary {E285_SAN_POST:.1e} "
                     f"(x{E285_SAN_RATIO:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("corpus step t (maintenance fires every "
                  f"{MAINT_EVERY}; reads POST-maintenance)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE SURVIVAL CURVES — the name-maintained build vs "
                 "the passive twin (bit-identical draws)", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE BETWEEN-STEPS WINDOW (the sawtooth) — the cell's own read
    ax = axes[0, 1]
    wl = rea["window_ledger"]
    if wl:
        xs = list(range(len(wl)))
        ys = [max(w["g0_pz"], 1e-7) for w in wl]
        labs = [w["window_phase"] for w in wl]
        cols = ["tab:green" if "post" in l else "tab:gray" for l in labs]
        ax.semilogy(xs, ys, "o-", lw=1.6, ms=6)
        for x, y, c in zip(xs, ys, cols):
            ax.plot(x, y, "o", ms=7, color=c)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=7.5, rotation=20)
        for i, w in enumerate(wl):
            if "post" in w["window_phase"]:
                ax.annotate(f"x{w['survival_ratio_vs_committed']:.4f}",
                            (i, w["g0_pz"]), fontsize=6.5,
                            xytext=(3, 6), textcoords="offset points",
                            color="tab:green")
        maint_ticks = [i for i, w in enumerate(wl)
                       if w["n_maint_so_far"] > (wl[i - 1]["n_maint_so_far"]
                                                 if i else 0)]
        for i in maint_ticks:
            ax.axvline(i - 0.5 if i else 0.5, color="crimson", ls=":",
                       lw=1.2)
    ax.set_xlabel("the window phases (red dotted = a maintenance step "
                  "between)")
    ax.set_ylabel("g0 battery (log)")
    ax.set_title("THE BETWEEN-STEPS WINDOW — the read around the "
                 "maintenance steps (the sawtooth or recovery shape)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, which="both")

    # (0,2) THE BUDGET SPLIT (S_corpus + S_maint vs the budget)
    ax = axes[0, 2]
    bl = rea["budget_ledger"]
    if bl:
        ax.plot([b["step"] for b in bl], [b["S_corpus"] for b in bl], "o-",
                lw=1.7, ms=4.5, color="tab:cyan",
                label="S_corpus (the budget-capped stream)")
        ax.plot([b["step"] for b in bl], [b["S_maint"] for b in bl], "^-",
                lw=1.7, ms=4.5, color="tab:orange",
                label="S_maint (the maintenance stream, budget-fitted lr)")
        ax.plot([b["step"] for b in bl], [b["S_total"] for b in bl], "s-",
                lw=1.9, ms=4, color="tab:green", label="S_total")
    tl = twn["budget_ledger"]
    if tl:
        ax.plot([b["step"] for b in tl], [b["S"] for b in tl], "v--",
                lw=1.2, ms=3.5, color="tab:blue", alpha=0.7,
                label="the twin's S (passive)")
    ax.axhline(budget_norm, color="crimson", ls="--", lw=1.5,
               label=f"THE BUDGET {BUDGET_FRAC}x write norm = "
                     f"{budget_norm:.3f}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("cumulative displacement (L2)")
    ax.set_title("THE BUDGET'S SPLIT — corpus + maintenance BOTH counted",
                 fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)

    # (1,0) THE TWO STREAMS' DIRECTION (in-room shares)
    ax = axes[1, 0]
    dl_r = {int(d["step"]): d for d in rea["disp_ledger"]}
    dl_t = {int(d["step"]): d for d in twn["disp_ledger"]}

    def _fl(v, floor=1e-12):
        return max(v, floor) if v is not None else floor
    steps_s = sorted(dl_r)
    ax.semilogy(steps_s,
                [_fl(dl_r[s]["corpus_disp_cum_in_room_frac"])
                 for s in steps_s], "o-", lw=1.6, ms=5,
                color="tab:cyan", label="RE-ANCHORED corpus cum in-room")
    ax.semilogy(steps_s,
                [_fl(dl_r[s]["maint_disp_cum_in_room_frac"])
                 for s in steps_s], "^-", lw=1.6, ms=5,
                color="tab:orange", label="RE-ANCHORED maint cum in-room "
                "(the direction read; floor = none yet)")
    steps_t = sorted(dl_t)
    ax.semilogy(steps_t,
                [_fl(dl_t[s]["corpus_disp_cum_in_room_frac"])
                 for s in steps_t], "s--", lw=1.4, ms=4.5,
                color="tab:blue", alpha=0.8, label="TWIN corpus cum "
                "in-room")
    ax.axhline(BUFSEP_ORTH_BAR, color="crimson", ls="--", lw=1.0,
               label=f"the fp-floor bar {BUFSEP_ORTH_BAR:.0e}")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.1, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(LADDER[0][0] / 2739072):.4f}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("in-room share of the realized displacement (log)")
    ax.set_title("THE TWO STREAMS' DIRECTIONS — orthogonal corpus vs the "
                 "maintenance's own course", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (1,1) THE CORPUS CE + THE BUDGET'S LR PRICE
    ax = axes[1, 1]
    for a, col, mk in ((REA_ARM, "tab:green", "o"), (TWN_ARM, "tab:blue",
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
    lr_rows = [(int(k), v) for k, v in rea["lr_ledger"].items()]
    ax3.plot([k for k, _ in lr_rows],
             [v["lr_applied"] for _, v in lr_rows], "^-", lw=1.0, ms=3,
             color="tab:red", alpha=0.8, label="RE-ANCHORED corpus lr")
    ml_steps = [m["step"] for m in rea["maint_ledger"]]
    if ml_steps:
        ax3.plot(ml_steps, [m["lr_maint"] for m in rea["maint_ledger"]],
                 "D-", ms=5, lw=1.0, color="tab:orange",
                 label="applied lr_m (budget-fitted)")
        ax3.axhline(MAINT_LR_BASE, color="tab:orange", ls=":", lw=1.0,
                    label=f"MAINT_LR_BASE {MAINT_LR_BASE:.4f} (schedule)")
    ax3.set_ylabel("lr (corpus applied vs the budget-fitted maintenance)",
                   fontsize=8.5)
    ax.set_title("THE ECONOMICS — corpus CE + the lr price (the "
                 f"maintenance moves "
                 f"{(rea['budget']['S_maint_final']) if rea['budget'].get('S_maint_final') is not None else float('nan'):.3f})",
                 fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6)
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
                 f"reads: RE-ANCHORED x"
                 f"{metrics['arms'][REA_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f} "
                 f"vs TWIN x"
                 f"{metrics['arms'][TWN_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f}",
                 fontsize=8.6)

    fig.suptitle(f"E287 — THE NAME-WEIGHTED MAINTENANCE CELL (the third "
                 f"form: the name-only CE at a budget-fitting lr) -> "
                 f"{verdict}", fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e287_name_maintained.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
