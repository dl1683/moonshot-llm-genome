"""E283 — THE ESTABLISHED-FACT COLLISION TEST (T244's never-run AFTER
arm, in its purest form). Design dispatched 2026-10-05 (the e278 fold's
dispatch; the queue's e283 row); this docstring carries the registered
question + bars VERBATIM + the convention freezes, committed at birth
BEFORE any compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): does an ALREADY-ESTABLISHED
fact collide as hard as a forming one? Every concurrent arm so far
FORMED under fire (the install ran while the corpus interleaved — e268/
e269/e270/e271's rungs, e273's barrels, e278's three nulls). T244's
registered discriminator said "corpus CONCURRENTLY vs AFTER"; the AFTER
arm never ran. Tonight it runs: the established fact = the committed
QUIET-FORMED 10k write (e261's serial cut, completed by e264 — the
committed threshold rung's final state, post g0 0.26464763, md5/size/
step-bound in e268's G_PARENTS record), loaded BIT-EXACT, and then the
corpus stream alone.

THE ARMS (verbatim intent from the dispatch):
  (a) ESTABLISHED-CONCURRENT: load the formed 10k fact, then run the
      1:1 interleave's corpus stream (the family's standard corpus
      step, the rig verbatim) for the full 400 corpus steps with NO
      install steps (the write already stands); read post g0 at
      milestones s100/200/300/400 of corpus-only-plus-matching-cadence.
  (b) ESTABLISHED-SERIAL-CONTROL (the same-session twin): load the same
      formed fact, run 400 corpus steps with the stream PAUSED (quiet
      drift only — the family's serial convention adapted; FROZEN:
      "quiet" = NO steps at all, the pure storage-decay control — the
      dispatch's recommended pick; the rejected alternative, the
      install-stream's zero gradient, disclosed below).

THE CONVENTION FREEZES (frozen HERE before compute; the family's step
accounting pairs install+corpus and HERE THERE IS NO INSTALL — the
smoke's centerpiece):
  * THE CONCURRENT PHASE := 400 CORPUS steps (t = 1..400 counts corpus
    steps only; NO install steps exist in the phase).
  * THE MILESTONES := t = 100/200/300/400 corpus steps (the dispatch's
    frozen list; no t=1 read — the t=0 read is the loaded fact's own
    G_FACTLOAD read).
  * THE MATCHING CADENCE := the corpus step's paired-lr schedule
    VERBATIM: lr(t) = 1e-3 x cosine_lr(t-1, 1000) (the lr the corpus
    steps carried in e268's 1:1 interleave — the same house cosine,
    warmup 100 included).
  * THE CORPUS STEP := e268's registered corpus step VERBATIM: 48
    windows = 16 original-host anchors (the same 60-window anchor bank)
    + 32 random corpus windows (contiguous train_ids slices);
    full-window CE; draws (aj_c(16), rj_c(32)) from the REGISTERED
    fresh corpus generator (seed 28301 — the family's per-cell rule:
    26801/26901/27001/27101/27301/27801/HERE 28301; its own generator,
    so no cell's verdict rides another's draw stream); backward ->
    clip 1.0 -> opt.step FREE (NO projection: the corpus stream is the
    natural stream, the rig verbatim).
  * THE OPTIMIZER := ONE AdamW (0.9, 0.95) wd 0.1, FRESH at phase start
    (the rig's optimizer, carried by the corpus stream alone — "shared
    AdamW" with the install stream that no longer exists; e278's
    datum: the optimizer IS the undertow, so it must be present,
    fresh, verbatim — AdamW's decoupled wd rides the corpus steps at
    the paired lr exactly as it did in e268's concurrent arm;
    disclosed as part of the intervention's body).
  * THE SERIAL CONTROL := the same loaded fact, the stream PAUSED, NO
    optimizer steps AT ALL (the pure storage-decay control; the reads
    are the arm's whole body, at the SAME milestone cadence
    t = 100/200/300/400 of the matched virtual timeline — by
    construction they must return the loaded read bit-identically;
    G_CONTROL gates exactly that). Matched in corpus exposure := ZERO
    corpus steps vs the concurrent arm's 400 — the arms' exposure
    delta IS the intervention, as in e268.
  * THE SURVIVAL RATIO := established-concurrent post g0(t) / the
    loaded fact's baseline 0.26464763283729553 (the dispatch's
    "0.2646" — e264's committed K10K post g0, PRIMARY; the session's
    own loaded read co-reported — the two agree to the read's
    determinism floor ~5e-7).
  * THE ADJUDICATION READ := the t=400 endpoint (the matched-corpus-
    exposure read); "through 400 corpus steps" = the endpoint
    adjudicates with the full milestone trajectory + the min milestone
    ratio co-reported verbatim.

FROZEN BARS (verbatim from the dispatch letter):
  - ESTABLISHED-SPARED: "the established write SURVIVES (>= 0.5x
    through 400 corpus steps) while the forming write died at 0.0002x
    — the collision kills FORMATION, not the formed; the barrier is
    formation-time only; T244's AFTER arm lands as 'the corpus is an
    additive abrasive only for forming writes.'"
  - ESTABLISHED-DIES: "the established write also dies (< 0.5x) — the
    collision is content-agnostic about age; retention and formation
    die together; the 237k retention barrier (e271) generalizes down
    to 10k for established facts."
  - MIXED: "partial decay (0.5x-0.9x) or non-monotone — the
    trajectories verbatim."

REGISTERED EXPECTATION (CITED, not re-registered — the dispatch's own
citation): under the roach-motel/two-body account, ESTABLISHED-DIES is
the natural prediction (the corpus traffic funnels through the room
regardless); under an additive-abrasive account, ESTABLISHED-SPARED.

THE COMPOSITE (the bars' letters overlap — MIXED's "(0.5x-0.9x)"
partial-decay carve-out sits inside SPARED's ">= 0.5x" floor; named
HERE at birth, no bar moved): TEXTURE (any hard-gate failure, OR the
loaded fact fails its G_FACTLOAD behavioral bind — nothing
adjudicated) -> ESTABLISHED-DIES (ratio_400 < 0.5x) -> MIXED (0.5x <=
ratio_400 < 0.9x [the partial-decay carve-out, named PARTIAL-DECAY] OR
non-monotone [frozen: a later milestone exceeding an earlier one by >
0.01 absolute g0 — a real upward swing, four orders above the ~5e-7
read determinism — OR any interior milestone ratio < 0.5x while the
endpoint >= 0.5x; named NON-MONOTONE]) -> ESTABLISHED-SPARED
(ratio_400 >= 0.9x and not non-monotone — SPARED's own >= 0.5x floor
satisfied a fortiori).

READS (the dispatch's list): post g0 per milestone (the WRITE read) +
post gm12 + CE_R; THE DISPLACEMENT LEDGER per milestone — (i) the
state's drift from the LOADED FACT: ||d|| + in-room share ||P d||/||d||
(THE roach-motel continuation read: does the established write's own
state drift in-room under the corpus?), (ii) the write's REMAINING
displacement from base: ||theta_t - base|| + its in-room share (the
write's room-occupancy trajectory; e268's SERIAL post read 0.944),
(iii) the corpus stream's realized displacement per milestone interval
+ cumulative, with in-room fractions (e278's ledger form), (iv) the
corpus gradient's own in-room fraction every 10 steps (the re-aiming
contrast datum: ||P g||/||g|| before the step vs the realized
displacement's in-room share after); the corpus CE + clipped-grad-norm
ledger; the survival ratio per milestone.

HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK,
G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K,
G_FACTLOAD, G_CORPUSGEN, G_CONTROL} — a failure HALTS (nothing
adjudicated). G_FACTLOAD binds the loaded artifact THREE ways: the
checkpoint's md5/size/step/traj/ledger (the dispatch's "checkpoint's
md5 bind"), the loaded state's flat-md5, and the behavioral read (post
g0 within 2e-6 / gm12 within 1e-5 of the committed literals — the
family's cross-session read-determinism law, headroom disclosed). The
room bit-gate (G_ROOMK10K) binds the rebuilt K10K room (seeds
26113/26114) to e264_rooms.pt's stored D/S (exact equality).

CHECKS (the dispatch's, in force): the machinery smoke FIRST (the
no-install accounting is the smoke's centerpiece — the phase counter
counts CORPUS steps, the milestones are corpus-step milestones, and
the control's reads are the t=0 read four times); the room certified
AND bit-bound; the corpus-generator registration (G_CORPUSGEN — the
seed, the draw order, the first step's draws logged at startup from a
scratch generator); n=1 per arm (the lottery note); nothing guaranteed.

NO CONS in this cell (the e278 precedent, T259/e281: the rehearsal
lane carries zero write information — a cons run cannot inform a
retention bar; the post-phase state is checkpointed so a later cons
can be run on exactly this state if the question ever returns).

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier);
bursts <= 175s (inside the dispatch's 180s), per-step thermal polls at
a 78C margin, 40s cooldowns (the 30-60s window), the 84C never-past
line (inside the dispatch's 85C), polls to runs/_envelope_log.jsonl
tagged e283:ARM:phase; CPU fp64 dense projections (pocketfft workers
2); CPU probing threads 4; NO concurrent GPU jobs.

Outputs: runs/e283/{metrics.json (PROGRESSIVE),
e283_established_collision.png, REPORT.md, run.log (gitignored)};
checkpoints runs/checkpoints/e283_*.pt (gitignored; md5s in metrics).
No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).
Commit + push per phase.

Run:  cd lab && python e283_established_collision.py  (E283_SMOKE=1
shakedown)
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
                                                      # MACHINERY, PORTED
                                                      # WHOLE BY IMPORT
                                                      # (the committed file
                                                      # is NOT modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E283_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e283_smoke" if SMOKE else "e283"
assert torch.cuda.is_available(), "e283 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268/e273/e278's disclosed convention): e261's ported
# drivers resolve their module globals (log / NAME / LADDER / RUNG_NAMES /
# SMOKE / INST_STEPS / T0 / thermal ledgers) AT CALL TIME through e261's
# module namespace — rebound HERE so the room build + envelope polls label
# THIS cell. The committed lab/e261_rank_ladder.py is untouched.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
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

# the arms
ARMS = ("ESTABLISHED-SERIAL-CONTROL", "ESTABLISHED-CONCURRENT")
CTRL_ARM, CONC_ARM = ARMS

# THIS cell's OWN REGISTERED FRESH corpus stream (the e268-family rule: one
# fresh registered stream per concurrent cell — e268 26801, e269 26901,
# e270 27001, e271 27101, e273 27301, e278 27801, HERE 28301)
CORPUS_GEN_SEED = 28301

# THE CONVENTION FREEZES (the docstring's, as numbers)
PHASE_STEPS = 8 if SMOKE else 400               # CORPUS steps, no install
MILESTONES = tuple(range(1, 9)) if SMOKE else (100, 200, 300, 400)

# THE ESTABLISHED FACT: the committed quiet-formed 10k write — e261's serial
# cut, completed by e264 (the committed threshold rung's final state)
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_STEP = 400
FACT_TRAJ_STEPS = [1, 100, 200, 300, 400]
FACT_LEDGER_MAX = 400

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_VERDICT = "SHARP-THRESHOLD"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12
E264_K10K_ROOT_G0 = 0.7707884907722473
E264_K10K_KEPT_MED = 0.060045162390265784

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"
E268_VERDICT = "DYNAMICAL-CARRIER"
E268_SERIAL_POST = 0.26464739441871643       # the forming-phase serial twin
E268_CONCURRENT_POST = 4.004325455753133e-05  # the FORMING write under fire
E268_RATIO = 0.00015130794937726159          # ~0.0002x — the SPARED bar's cite

E271_METRICS = E43.REPO / "runs" / "e271" / "metrics.json"
E271_MD5 = "16c29a3167d8523a64f66f03c717f67b"
E271_VERDICT = "MIXED"
E271_CONCURRENT_POST = 0.034998517483472824  # the 237k retention datum
E271_CITED_SERIAL = 0.38436421751976013
E271_RATIO = 0.09105560790573215             # the 237k retention barrier

E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"
E278_VERDICT = "UNDERTOW-REGARDLESS"         # the roach-motel record
E278_MISSILE_GRAD_IN_ROOM = 4.6e-17          # the exactly-orthogonal gradient
E278_MISSILE_DISP_IN_ROOM = (0.48, 0.60)     # ...and where it WALKED anyway

ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268's G_PARENTS bind

# the discriminator's frozen numbers
SURVIVE_FRAC = 0.5               # "< 0.5x" (DIES) / ">= 0.5x" (SPARED floor)
PARTIAL_FRAC = 0.9               # MIXED's partial-decay carve-out's ceiling
NONMONO_SWING = 0.01             # a later milestone g0 exceeding an earlier
                                 # one by > this = NON-MONOTONE (frozen)
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law, with
                                 # headroom; disclosed)
CONTROL_READ_TOL = 5e-7          # G_CONTROL: the repeated t=0 read's scatter
RATIO_DEN_FLOOR = 1e-6           # the ladder's floor-guard convention
G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 — the expression floor (ctx)
MATCH_BAND = E261.MATCH_BAND              # the landing band (context only)
G_READ_TOL = E261.G_READ_TOL             # 5e-3

ARM_DESC = {
    CTRL_ARM: "(b) the same-session twin: the SAME loaded formed fact, the "
              "corpus stream PAUSED — NO optimizer steps AT ALL (the pure "
              "storage-decay control, the dispatch's recommended freeze; "
              "the rejected alternative 'the install stream's zero "
              "gradient' disclosed in the docstring) — with the milestone "
              "reads at the SAME cadence t=100/200/300/400 of the matched "
              "virtual timeline; by construction every read must return "
              "the loaded fact's own read (G_CONTROL gates the null)",
    CONC_ARM: "(a) the established fact under fire: the loaded formed 10k "
              "write + the family's standard corpus stream ALONE (the 1:1 "
              "interleave's corpus steps, the rig verbatim — 48 windows = "
              "16 anchors + 32 random, full-window CE, the matching "
              "cadence lr 1e-3 x cosine_lr(t-1,1000), clip 1.0 -> "
              "opt.step FREE) through ONE fresh AdamW (0.9,0.95) wd 0.1 "
              "for 400 CORPUS steps — NO install steps (the write already "
              "stands); T244's never-run AFTER arm",
}

REGISTERED = {
    "question_verbatim": "does an ALREADY-ESTABLISHED fact collide as hard "
        "as a forming one? Every concurrent arm so far FORMED under fire "
        "(the install ran while the corpus interleaved). T244's registered "
        "discriminator said 'corpus CONCURRENTLY vs AFTER'; the AFTER arm "
        "never ran. Tonight it runs.",
    "arms_verbatim": {
        "a_ESTABLISHED-CONCURRENT": "load the formed 10k fact, then run the "
            "1:1 interleave (the family's standard corpus stream, shared "
            "AdamW — the rig verbatim) for the full 400 corpus steps with "
            "NO install steps needed (the write already stands); read post "
            "g0 at milestones s100/200/300/400 of "
            "corpus-only-plus-matching-cadence",
        "b_ESTABLISHED-SERIAL-CONTROL": "load the same formed fact, run 400 "
            "corpus steps with the stream PAUSED (quiet drift only — the "
            "family's serial convention adapted; PICK and freeze: NO steps "
            "at all, the pure storage-decay control)",
    },
    "bars_verbatim": {
        "ESTABLISHED-SPARED": "the established write SURVIVES (>= 0.5x "
            "through 400 corpus steps) while the forming write died at "
            "0.0002x — the collision kills FORMATION, not the formed; the "
            "barrier is formation-time only; T244's AFTER arm lands as "
            "'the corpus is an additive abrasive only for forming writes.'",
        "ESTABLISHED-DIES": "the established write also dies (< 0.5x) — the "
            "collision is content-agnostic about age; retention and "
            "formation die together; the 237k retention barrier (e271) "
            "generalizes down to 10k for established facts.",
        "MIXED": "partial decay (0.5x-0.9x) or non-monotone — the "
            "trajectories verbatim.",
    },
    "expectation_cited": "under the roach-motel/two-body account, "
        "ESTABLISHED-DIES is the natural prediction (the corpus traffic "
        "funnels through the room regardless); under an additive-abrasive "
        "account, ESTABLISHED-SPARED (CITED from the dispatch, not "
        "re-registered)",
    "operationalizations": (
        "frozen BEFORE compute: THE CONCURRENT PHASE := 400 CORPUS steps "
        "(t=1..400, no install); THE MILESTONES := t=100/200/300/400 "
        f"corpus steps; THE MATCHING CADENCE := lr(t) = 1e-3 x "
        f"cosine_lr(t-1, {E261.INST_TOTAL}) (the corpus steps' paired-lr "
        "schedule VERBATIM); THE CORPUS STEP := e268's registered form "
        "(16 anchors + 32 random windows, full-window CE, the REGISTERED "
        f"fresh corpus generator seed {CORPUS_GEN_SEED}, clip 1.0 -> "
        "opt.step FREE); THE OPTIMIZER := ONE fresh AdamW (0.9,0.95) wd "
        "0.1 at phase start (the rig's optimizer; the wd channel rides "
        "the corpus steps exactly as in e268's concurrent arm — "
        "disclosed); THE SERIAL CONTROL := NO steps at all (the pure "
        "storage-decay control; the reads at the same milestone cadence; "
        "G_CONTROL gates the null); THE SURVIVAL RATIO := post g0(t) / "
        f"{FACT_BASELINE_G0} (the committed baseline, PRIMARY; the "
        "session's loaded read co-reported); THE ADJUDICATION READ := the "
        "t=400 endpoint, trajectory + min-milestone co-reported; "
        "COMPOSITE := TEXTURE (hard-gate failure OR G_FACTLOAD "
        "behavioral miss) -> ESTABLISHED-DIES (ratio_400 < 0.5x) -> MIXED "
        "(0.5x <= ratio_400 < 0.9x [PARTIAL-DECAY] OR non-monotone [a "
        "later milestone > an earlier one + 0.01 absolute, OR an interior "
        "milestone < 0.5x while the endpoint >= 0.5x; NON-MONOTONE]) -> "
        "ESTABLISHED-SPARED (ratio_400 >= 0.9x, monotone — MIXED's "
        "partial-decay carve-out takes precedence over SPARED's floor, "
        "named at birth, no bar moved); NO CONS (T259/e281; the post-phase "
        "state checkpointed); HARD GATES := {G_NAMEFREE, G_SPLICE, "
        "G_BATTERY, G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, "
        "G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD, "
        "G_CORPUSGEN, G_CONTROL} (a failure HALTS)."),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (the e278 fold's dispatch; the queue's e283 row — T244's "
        "never-run AFTER arm); this script committed at birth BEFORE any "
        "compute; adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE CONVENTION FREEZE IS THE CELL'S CENTERPIECE (the dispatch's "
    "CAREFUL): the family's step accounting pairs install+corpus; here "
    "there is no install. FROZEN: the phase counts CORPUS steps only "
    "(400), the milestones are corpus-step milestones (100/200/300/400), "
    "the cadence is the corpus steps' own paired-lr schedule "
    "cosine_lr(t-1,1000), and the control is matched by CADENCE (the "
    "same milestone reads on a paused stream) with corpus exposure ZERO "
    "by construction — the arms' exposure delta is the intervention, as "
    "in e268.",
    "THE CONTROL IS VACUOUS BY CONSTRUCTION AND THAT IS ITS JOB "
    "(registered at birth): with NO steps at all the state cannot move; "
    "the control's four milestone reads must return the loaded fact's "
    "read bit-identically (G_CONTROL, tol 5e-7 — same-session repeated "
    "CPU reads are exact to fp32 determinism). Its role is the null "
    "anchor: any decay the concurrent arm shows is attributable to the "
    "corpus traffic (optimizer + wd + gradients), never to time, storage "
    "or the read pipeline itself. The rejected alternative — the "
    "install stream's zero gradient (forward/backward with zeroed "
    "grads, opt.step on nothing) — would ADD optimizer-state noise "
    "(Adam's eps-floor drift on zero gradients) that is NOT storage "
    "decay; the dispatch's recommendation (NO steps) is taken.",
    "THE OPTIMIZER IS FRESH AT PHASE START (the disclosed form): in the "
    "family's concurrent arms 'shared AdamW' meant one optimizer shared "
    "between the install and corpus streams from install step 1. Here "
    "the install is DONE — the phase's AdamW starts fresh (zero "
    "moments) and is carried by the corpus stream alone. The e278 "
    "roach-motel datum (Adam re-aims even orthogonal gradients into the "
    "room; the optimizer IS the undertow) makes the optimizer's presence "
    "the mechanism's carrier, so the rig's optimizer runs VERBATIM "
    "(AdamW 0.9/0.95, wd 0.1): its decoupled wd shrinks every weight by "
    "~3.4% over the 400-step cosine — a uniform, mostly in-room-"
    "direction shrink of the fact's own displacement (the fact is ~94% "
    "in-room), disclosed as part of the intervention's body (the same "
    "channel acted on e268's forming arm), measured in the drift ledger "
    "never assumed.",
    "NO CONS (the e278 precedent, T259/e281): the rehearsal lane carries "
    "zero write information — a cons run cannot inform a retention bar; "
    "the post-phase state is checkpointed (e283_ESTABLISHED-CONCURRENT_"
    "post.pt) so a later cons can be run on exactly this state if the "
    "question ever returns.",
    "THE VEHICLE FACT IS LOADED, NEVER RE-FORMED (extend, don't repeat): "
    "the established fact is e261's committed serial cut completed by "
    "e264 (e261_K10K_inst_resume.pt), md5/size/step/traj/ledger-bound + "
    "flat-md5'd + behaviorally gated (G_FACTLOAD) — the same artifact "
    "e268's G_PARENTS bound and e268's serial re-run reproduced to "
    "|d post g0| 2.4e-7. No install runs in this cell.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector + "
    "LadderRooms (the room rebuild, certification, displacement loads), "
    "the thermal envelope (per-step polls, 78C margin, 175s bursts "
    "inside the dispatch's 180s, 40s cooldowns, the 84C line inside the "
    "dispatch's 85C), the progressive-metrics + resume-ckpt conventions "
    "— the module-global rebinding (log/NAME/LADDER/RUNG_NAMES/T0/"
    "thermal ledgers, disclosed in-code) retargets the machinery's I/O "
    "to this cell; the committed lab/e261_rank_ladder.py is NOT "
    "modified. The ONE new driver is chunked_corpus_phase (this file) — "
    "the corpus-only phase with the drift ledger.",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the "
    "measured displacement loads; no new history is run.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E283_SMOKE=1): 8 corpus steps, milestones at every step "
    "1..8, room k=512 at the same seed pair (G_ROOMK10K vacuous — no "
    "committed record at smoke k; disclosed), the REAL committed fact "
    "loaded and read (G_FACTLOAD live — the artifact binds are the smoke "
    "centerpiece alongside the no-install accounting), G_CORPUSGEN live, "
    "G_CONTROL live, the adjudication + figure exercised, all paths "
    "smoke_-prefixed, own smoke dir; NOTHING adjudicated or gated "
    "(SMOKE stamp on every read).",
]

device_events: list[dict] = []
thermal_log: list[dict] = []
E261.thermal_log = thermal_log          # ONE ledger (e278's rebinding)
E261.device_events = device_events


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


def burst_temp_check(tag: str) -> tuple[bool, float]:
    s = common.gpu_status()
    common._log_envelope_poll(f"{NAME}:{tag}:mid", s["util"], s["temp"],
                              s["temp"] < E261.TEMP_EARLY_END)
    row = {"tag": tag, "temp": s["temp"], "t": round(time.time() - T0, 1)}
    thermal_log.append(row)
    if s["temp"] >= E261.TEMP_HARD:
        device_events.append({"tag": tag, "event": "THERMAL VIOLATION "
                              f"(>= {E261.TEMP_HARD:.0f}C)", "status": s})
        log(f"  [gpu:{tag}:mid] TEMP {s['temp']:.0f}C >= "
            f"{E261.TEMP_HARD:.0f}C — HARD VIOLATION recorded; ending burst")
        return False, s["temp"]
    if s["temp"] >= E261.TEMP_EARLY_END:
        log(f"  [gpu:{tag}:mid] temp {s['temp']:.0f}C >= "
            f"{E261.TEMP_EARLY_END:.0f}C margin — ending burst "
            f"(84C line protected)")
        return False, s["temp"]
    return True, s["temp"]


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


# ---------------------------------------- THE CORPUS-ONLY PHASE (the new driver)
def chunked_corpus_phase(tag, net0, proj: "E261.LadderRooms",
                         fact_flat_np: np.ndarray, base_flat_np: np.ndarray,
                         anchor_full, train_ids, g0_ids, gm12_ids,
                         r_eval_xy, zid, resume_ck: Path,
                         dev: torch.device) -> dict:
    """THE ESTABLISHED-CONCURRENT ARM'S DRIVER (this cell's only new
    machinery). The net starts at the LOADED formed fact (net0 carries it);
    ONE fresh AdamW (0.9,0.95) wd 0.1; per corpus step t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 28301, the registered
      fresh stream); corpus batch = 16 original-host anchors + 32 random
      corpus windows (the g1c Dmix corpus convention VERBATIM);
      full-window CE; lr = 1e-3 x cosine_lr(t-1, 1000) (the matching
      cadence); backward -> clip 1.0 -> opt.step FREE (no projection).

    Thermal: a poll after EVERY corpus opt step (tag <arm>:corpus:c<n>.x).
    Milestone reads at t in MILESTONES: the WRITE read (g0/gm12 battery +
    CE_R on the CPU eval net) + THE DRIFT LEDGER (fp64 CPU): the state's
    drift from the loaded fact (||d||, in-room share), the remaining
    displacement from base (||.||, in-room share), the corpus stream's
    realized interval/cumulative displacement with in-room fractions, and
    every 10 steps the corpus gradient's own in-room fraction (the
    re-aiming contrast datum)."""
    corp_bs, mix_random, lr = E43.CORP_BS, E43.MIX_RANDOM, E43.LR
    n_steps = PHASE_STEPS
    n_anc = anchor_full.shape[0]
    N = int(fact_flat_np.size)
    state = {"step": 0, "traj": [], "corpus_ledger": {},
             "disp_ledger": [],
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "corp_prev": torch.zeros(N, dtype=torch.float64)}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at corpus step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[ROOM_MODE]
    corp_cum = state["corp_cum"].to(dev)
    corp_prev = state["corp_prev"].to(dev)

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{CONC_ARM}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = common.cosine_lr(step - 1, E261.INST_TOTAL)
            for g in opt.param_groups:
                g["lr"] = lr * f
            # ---- THE CORPUS STEP (the family's standard, the rig verbatim)
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
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt.step()                          # FREE — the natural stream
            with torch.no_grad():
                corp_cum += torch.cat([p.detach().reshape(-1)
                                       for p in net.parameters()]) - theta_b
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c,
                                                "g_in_room_frac": g_in_room}
            n_burst += 1
            ok_t, temp = burst_temp_check(f"{CONC_ARM}:corpus:c{n_chunks}.x")
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
                ld_fact = proj.displacement_loads(
                    torch.from_numpy(d_fact), ROOM_MODE)
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
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_fact_norm": dn,
                    "drift_from_fact_in_room_frac": d_fact_in_room,
                    "drift_from_fact_v_excess": ld_fact["v_excess"],
                    "drift_from_fact_cos_to_span": ld_fact["cos_to_span"],
                    "remaining_from_base_norm": rn,
                    "remaining_from_base_in_room_frac": rem_in_room,
                    "corpus_disp_interval_norm": vn,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_norm": cum_n,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum})
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
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.6f} "
                    f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d| {dn:.3f} "
                    f"drift-in-room "
                    f"{('%.4f' % d_fact_in_room) if d_fact_in_room is not None else 'n/a'} "
                    f"rem-in-room "
                    f"{('%.4f' % rem_in_room) if rem_in_room is not None else 'n/a'} "
                    f"corp|v| {vn if vn else 0.0:.3f}")
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
                    "opt": opt.state_dict(), "cgen_state": cgen.get_state(),
                    "step": step, "traj": state["traj"],
                    "corpus_ledger": state["corpus_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "corp_cum": corp_cum.cpu(),
                    "corp_prev": corp_prev.cpu(),
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "disp_ledger": state["disp_ledger"],
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}


def serial_control_reads(fact_sd: dict, fact_g0: float) -> dict:
    """THE ESTABLISHED-SERIAL-CONTROL ARM: the same loaded fact, the stream
    PAUSED, NO optimizer steps AT ALL (the pure storage-decay control — the
    frozen pick). The reads are the arm's whole body: the milestone
    battery/CE_R reads at the SAME cadence (t = 100/200/300/400 of the
    matched virtual timeline). By construction every read must return the
    loaded fact's own read — G_CONTROL gates exactly that (the null)."""
    net = G1.evl_load(fact_sd)
    net.eval()
    rows = []
    for t in MILESTONES:
        bz0 = G1.battery_cell(net, g0_ids_ctl[0], zid_ctl)
        bz12 = G1.battery_cell(net, g0_ids_ctl[1], zid_ctl)
        ce_r = G1.ce_fixed_cpu(net, *r_eval_xy_ctl)
        rows.append({"step": t, "g0_pz": bz0["mean_pz"],
                     "g0_argmax": bz0["frac_argmax_z"],
                     "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                     "survival_ratio_vs_committed":
                         bz0["mean_pz"] / FACT_BASELINE_G0,
                     "elapsed_s": round(time.time() - T0, 1)})
        log(f"  [{CTRL_ARM}] t{t:4d} g0 {bz0['mean_pz']:.8f} "
            f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
            f"{bz12['mean_pz']:.6f} CE_R {ce_r:.4f} — stream PAUSED "
            f"(no steps; the storage-decay null)")
    return {"traj": rows, "steps": 0,
            "max_abs_g0_drift": max(abs(r["g0_pz"] - fact_g0)
                                    for r in rows)}


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def _envelope_summary() -> dict:
    """The thermal-envelope summary aggregated from runs/_envelope_log.jsonl
    (the PERSISTED per-poll ledger — survives resume passes)."""
    out = {"burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
           "per_step_polls": "after EVERY corpus opt step — aggregated "
                             "from runs/_envelope_log.jsonl (the persisted "
                             "ledger; survives resume passes; tags "
                             "e283:ARM:phase per the dispatch)",
           "early_end_margin_c": E261.TEMP_EARLY_END,
           "hard_line_c": E261.TEMP_HARD}
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
                   f"polls are tagged e283_smoke: and excluded")
    return out


# the control's read dependencies (set in main before the control runs)
g0_ids_ctl: list = []
zid_ctl: int = -1
r_eval_xy_ctl: tuple = ()


def main():
    global g0_ids_ctl, zid_ctl, r_eval_xy_ctl
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e283_established_collision",
        "phase": "THE ESTABLISHED-FACT COLLISION TEST — T244's never-run "
                 "AFTER arm in its purest form: the committed QUIET-FORMED "
                 "10k write (post g0 0.2646, loaded bit-exact) under the "
                 "family's standard corpus stream ALONE (400 corpus steps, "
                 "the matching cadence, one fresh AdamW — no install) vs "
                 "the same-session paused-stream storage-decay control — "
                 "ESTABLISHED-SPARED vs ESTABLISHED-DIES vs MIXED, "
                 "adjudicated on the WRITE read at matched corpus exposure",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; the "
                      "control arm is CPU-read-only) + CPU fp64 dense "
                      "projections (pocketfft workers 2), CPU probing "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (inside the dispatch's "
                      f"180s), per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (the 30-60s window), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (inside the "
                      "dispatch's 85C), polls to runs/_envelope_log.jsonl "
                      "tagged e283:ARM:phase",
            "trainings": "1 corpus-only phase s400 (the established-"
                         "concurrent arm; NO install — the fact is loaded; "
                         "NO cons — T259/e281, the post state checkpointed); "
                         "1 paused-stream control (reads only, no steps)",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} CORPUS steps (no install steps — the "
                     "family's install+corpus accounting does not apply; "
                     "the phase's counter counts corpus steps only)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps (the dispatch's frozen list)",
            "cadence": f"lr(t) = 1e-3 x cosine_lr(t-1, {E261.INST_TOTAL}) "
                       "(the corpus steps' paired-lr schedule VERBATIM)",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors (the same "
                           "60-window bank) + 32 random corpus windows; "
                           "full-window CE; the REGISTERED fresh corpus "
                           f"generator seed {CORPUS_GEN_SEED}; clip 1.0 -> "
                           "opt.step FREE (unprojected — the natural stream)",
            "optimizer": "ONE fresh AdamW (0.9, 0.95) wd 0.1 at phase start "
                         "(the rig's optimizer, carried by the corpus "
                         "stream alone; the wd channel disclosed)",
            "control": "NO steps at all — the pure storage-decay control "
                       "(the dispatch's recommended pick; the zero-gradient "
                       "alternative rejected, disclosed); the reads at the "
                       "same milestone cadence",
            "survival_ratio": f"post g0(t) / {FACT_BASELINE_G0} (the "
                              "committed baseline, PRIMARY; the session's "
                              "loaded read co-reported)",
            "adjudication_read": "the t=400 endpoint (matched corpus "
                                 "exposure); trajectory + min milestone "
                                 "co-reported",
        },
        "deviations": deviations,
        "builds_on": [
            "T244 / e268 (THE registration whose AFTER arm never ran: "
            "DYNAMICAL-CARRIER at ~0.0002x — the forming 10k write died "
            "under the 1:1 interleave through the one shared AdamW; this "
            "cell runs the complementary arm — the SAME corpus stream "
            "against the ALREADY-FORMED write)",
            "T258 / e273 (TRAJECTORY-TWO-BODY: the coupling is in the "
            "parameters — separate-AdamW died harder 0.008x)",
            "T260 / e278 (UNDERTOW-REGARDLESS + THE ROACH MOTEL: the "
            "missile's exactly-orthogonal gradient still walked 48-60% "
            "in-room — Adam re-aims orthogonal gradients into the room; "
            "the optimizer IS the undertow; the registered expectation "
            "under this account: ESTABLISHED-DIES)",
            "T248 / e271 (the 237k retention barrier: the widest forming "
            "write retained only 0.091x under concurrent — the DIES bar's "
            "cite)",
            "T242 / e264 (the committed threshold rung: post g0 "
            f"{FACT_BASELINE_G0:.8f} — the loaded fact's baseline) + "
            "T239 / e261 (the cut itself; the vehicle ckpt)",
            "T259 / e281 (the cons teaches from anything — the NO-CONS "
            "registration's basis)",
            "T239 / e261 (the machinery PORTED WHOLE BY IMPORT: the SRCT "
            "projector, LadderRooms, the thermal envelope)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314 "
            "+ e113 cons 10901)",
        ],
        "whats_new": [
            "THE AFTER ARM ITSELF: the record's first collision test "
            "against an ALREADY-ESTABLISHED fact — every prior concurrent "
            "arm formed under fire; here the write stands (loaded "
            "bit-exact from the committed rung) and ONLY the corpus "
            "stream runs — formation and retention finally separated at "
            "the same rung, the same room, the same stream",
            "THE STORAGE-DECAY CONTROL: the family's first pure "
            "paused-stream twin (no steps at all) — the null that makes "
            "any concurrent-arm decay attributable to the corpus traffic "
            "alone",
            "THE DRIFT LEDGER ON AN ESTABLISHED WRITE: the state's drift "
            "from the loaded fact with its in-room share per milestone "
            "(the roach-motel continuation: does the ESTABLISHED write's "
            "own state drift in-room under the corpus?) + the remaining-"
            "from-base room occupancy + the corpus gradient-vs-displacement "
            "re-aiming contrast",
        ],
        "gates": {},
    })
    log(f"E283 — THE ESTABLISHED-FACT COLLISION TEST (smoke={SMOKE}) -> {RD}")
    log(f"arms: {' / '.join(ARMS)}; the fact = {FACT_CK} (md5-bound, post "
        f"g0 {FACT_BASELINE_G0:.8f}); the room = the committed K10K (seeds "
        f"{LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs {ROOMS264_CK}); the "
        f"phase = {PHASE_STEPS} corpus steps at the matching cadence; "
        f"milestones {'/'.join(str(m) for m in MILESTONES)}; bars: "
        f"DIES < {SURVIVE_FRAC:.0%}x < MIXED < {PARTIAL_FRAC:.0%}x <= "
        f"SPARED (non-monotone -> MIXED)")
    write_partial("startup (bars + conventions registered, committed at "
                  "birth)")
    set_seed(CORPUS_GEN_SEED)       # global init only; every RNG is its own

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

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "note": "the install windows are NOT run in this cell "
                          "(the fact is loaded); the mask gate carries the "
                          "splice/mask convention's identity",
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

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
        "/ e170 bank / install mask identity)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_post = e264m["arms"]["K10K"]["install"]["post_cells"]["g0"]
    e264_gm12 = e264m["arms"]["K10K"]["install"]["post_cells"]["gm12"]
    e268m = json.loads(E268_METRICS.read_text(encoding="utf-8"))
    e268_post_c = e268m["arms"]["CONCURRENT"]["install"]["post_cells"]["g0"]
    e271m = json.loads(E271_METRICS.read_text(encoding="utf-8"))
    e271_conc = e271m["adjudication"]["reads"]["CONCURRENT"]["post_g0"]
    e278m = json.loads(E278_METRICS.read_text(encoding="utf-8"))
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
                         "K10K_post_g0": e264_post, "K10K_post_gm12": e264_gm12,
                         "note": "THE loaded fact's committed record (the "
                                 "quiet-formed threshold rung: e261's cut, "
                                 "completed by e264)"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of(E268_METRICS), "bound_md5": E268_MD5,
                         "verdict": e268m["adjudication"]["verdict"],
                         "serial_post_g0": E268_SERIAL_POST,
                         "concurrent_post_g0": e268_post_c,
                         "ratio": E268_RATIO,
                         "note": "the FORMING-concurrent reference (~0.0002x "
                                 "— the SPARED bar's cite); this cell's "
                                 "corpus step + cadence are its registered "
                                 "forms"},
        "e271_metrics": {"path": str(E271_METRICS),
                         "md5": md5of(E271_METRICS), "bound_md5": E271_MD5,
                         "verdict": e271m["adjudication"]["verdict"],
                         "concurrent_post_g0": e271_conc,
                         "ratio": E271_RATIO,
                         "note": "the 237k retention barrier (the DIES "
                                 "bar's cite)"},
        "e278_metrics": {"path": str(E278_METRICS),
                         "md5": md5of(E278_METRICS), "bound_md5": E278_MD5,
                         "verdict": e278m["adjudication"]["verdict"],
                         "missile_grad_in_room": E278_MISSILE_GRAD_IN_ROOM,
                         "missile_disp_in_room": list(E278_MISSILE_DISP_IN_ROOM),
                         "note": "THE ROACH-MOTEL RECORD (the registered "
                                 "expectation's source: ESTABLISHED-DIES "
                                 "under the undertow account)"},
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
            "e264_verdict": E264_VERDICT,
            "fact_baseline_g0": FACT_BASELINE_G0,
            "fact_baseline_gm12": FACT_BASELINE_GM12,
            "e264_K10K_root_g0": E264_K10K_ROOT_G0,
            "e264_K10K_kept_med": E264_K10K_KEPT_MED,
            "e268_verdict": E268_VERDICT,
            "e268_concurrent_post_g0": E268_CONCURRENT_POST,
            "e268_ratio": E268_RATIO,
            "e271_verdict": E271_VERDICT,
            "e271_concurrent_post_g0": E271_CONCURRENT_POST,
            "e271_ratio": E271_RATIO,
            "e278_verdict": E278_VERDICT,
            "fact_md5": FACT_MD5, "fact_step": FACT_STEP,
            "fact_traj_steps": FACT_TRAJ_STEPS,
            "fact_ledger_max": FACT_LEDGER_MAX},
        "pass": bool(
            e264m["adjudication"]["verdict"] == E264_VERDICT
            and abs(e264_post - FACT_BASELINE_G0) < 1e-12
            and abs(e264_gm12 - FACT_BASELINE_GM12) < 1e-12
            and e268m["adjudication"]["verdict"] == E268_VERDICT
            and abs(e268_post_c - E268_CONCURRENT_POST) < 1e-12
            and e271m["adjudication"]["verdict"] == E271_VERDICT
            and abs(e271_conc - E271_CONCURRENT_POST) < 1e-12
            and e278m["adjudication"]["verdict"] == E278_VERDICT
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E271_METRICS) == E271_MD5
            and md5of(E278_METRICS) == E278_MD5
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
    log(f"P0b: G_PARENTS PASS — e264 {E264_VERDICT} (the fact's post g0 "
        f"{e264_post:.8f}); e268 forming-concurrent {E268_CONCURRENT_POST:.2e} "
        f"(x{E268_RATIO:.6f}); e271 237k retention x{E271_RATIO:.4f}; e278 "
        f"{E278_VERDICT}; the fact ckpt md5/size/step-bound")
    write_partial("P0b parents hard-bound (the fact ckpt loaded)")
    del e264m, e268m, e271m, e278m

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
        "e283_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e283's room (the fact's own room): the committed K10K "
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
    G_FACTLOAD = {
        "form": "the established fact, loaded BIT-EXACT and gated THREE "
                "ways: (1) the artifact (md5/size/step/traj/ledger — "
                "G_PARENTS), (2) the loaded state's flat-md5, (3) the "
                "behavioral read (post g0 within "
                f"{FACT_READ_TOL_G0:.0e} / gm12 within "
                f"{FACT_READ_TOL_GM12:.0e} of e264's committed literals — "
                "the family's cross-session read-determinism law "
                "|d post g0| ~ 5e-7-1e-6 with headroom, disclosed)",
        "flat_md5": fact_flat_md5,
        "read_g0": {"mine": fact_g0, "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_g0 - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_gm12, "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_gm12 - FACT_BASELINE_GM12)},
        "read_gp12": fact_gp12, "read_ce_r": fact_ce_r,
        "survival_ratio_denominator": "the COMMITTED literal (PRIMARY); "
                                      "the session read co-reported",
        "displacement_from_base": {
            "norm": float(np.linalg.norm(fact_flat_np - base_flat_np)),
            **loads_fact,
            "note": "the write's own standing displacement (e268's SERIAL "
                    "post read in_own_room 0.944 — the continuation "
                    "ledger's t=0 reference)"},
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
        f"{loads_fact['in_own_room']:.4f} in-room: PASS")
    metrics["the_fact"] = {
        "checkpoint": f"runs/checkpoints/{FACT_CK}", "md5": FACT_MD5,
        "committed_post_g0": FACT_BASELINE_G0, "session_read_g0": fact_g0,
        "session_read_gm12": fact_gm12, "session_read_ce_r": fact_ce_r,
        "displacement_from_base": G_FACTLOAD["displacement_from_base"],
        "baseline_used_for_ratios": "committed (PRIMARY)",
    }
    write_partial("P2 the fact loaded bit-exact + gated (the baseline read)")
    del fact_net

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
                "26801/26901/27001/27101/27301/27801/28301), its own "
                "generator; per corpus step the draws (aj_c(16), rj_c(32)) "
                "in that order; the first step's draws logged here from a "
                "scratch generator (the driver's cgen starts identically "
                "by construction)",
        "seed": CORPUS_GEN_SEED,
        "first_step_aj": [int(x) for x in aj_probe.tolist()],
        "first_step_rj": [int(x) for x in rj_probe.tolist()],
        "composition": "16 original-host anchors (the 60-window bank) + 32 "
                       "random corpus windows (contiguous train_ids "
                       "slices); full-window CE; clip 1.0 -> opt.step FREE",
        "pass": True,
    }
    metrics["gates"]["G_CORPUSGEN"] = G_CORPUSGEN
    log(f"P2b G_CORPUSGEN: the corpus stream REGISTERED (seed "
        f"{CORPUS_GEN_SEED}; first draws aj[:4]={G_CORPUSGEN['first_step_aj'][:4]}"
        f" rj[:4]={G_CORPUSGEN['first_step_rj'][:4]})")
    write_partial("P2b the corpus generator registered")

    # ================= P3: ARM CONTROL (the storage-decay null) =========
    log("=" * 78)
    log(f"ARM-{CTRL_ARM} — {ARM_DESC[CTRL_ARM]}")
    g0_ids_ctl = [g0_ids, gm12_ids]
    zid_ctl = zid
    r_eval_xy_ctl = r_eval_xy
    ctrl = serial_control_reads(fact_sd, fact_g0)
    G_CONTROL = {
        "form": "the storage-decay null: with NO steps the state cannot "
                "move; all four milestone reads must return the loaded "
                "fact's own read (the read pipeline's determinism gate — "
                f"tol {CONTROL_READ_TOL:.0e}, same-session repeated CPU "
                "reads are exact to fp32 determinism)",
        "max_abs_g0_drift": ctrl["max_abs_g0_drift"],
        "bar": CONTROL_READ_TOL,
        "pass": bool(ctrl["max_abs_g0_drift"] <= CONTROL_READ_TOL),
    }
    assert G_CONTROL["pass"], f"G_CONTROL FAILED: {G_CONTROL}"
    metrics["gates"]["G_CONTROL"] = G_CONTROL
    metrics["arms"] = {CTRL_ARM: {
        "desc": ARM_DESC[CTRL_ARM], "phase": {
            "traj": ctrl["traj"], "steps": 0,
            "post_cells": {"g0": ctrl["traj"][-1]["g0_pz"],
                           "gm12": ctrl["traj"][-1]["gm12_pz"],
                           "ce_r": ctrl["traj"][-1]["ce_r"]},
            "survival_ratio_final":
                ctrl["traj"][-1]["survival_ratio_vs_committed"],
            "max_abs_g0_drift": ctrl["max_abs_g0_drift"],
            "note": "NO steps — the reads are the arm's whole body; the "
                    "control is the null anchor, vacuous by construction "
                    "and gated to be exactly that",
        }}}
    log(f"ARM-{CTRL_ARM}: the null HOLDS — max |g0 drift| "
        f"{ctrl['max_abs_g0_drift']:.1e} over the milestone cadence "
        f"(bar {CONTROL_READ_TOL:.0e}): PASS")
    write_partial("P3 the storage-decay control read (the null anchor)")

    # ================= P4: ARM CONCURRENT (the corpus phase) ============
    burst_cooldown(f"{CTRL_ARM} -> {CONC_ARM}")
    log("=" * 78)
    log(f"ARM-{CONC_ARM} — {ARM_DESC[CONC_ARM]}")
    conc = chunked_corpus_phase(
        f"{CONC_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, anchor_full, train_ids, g0_ids, gm12_ids, r_eval_xy,
        zid, CKPT_DIR / ("smoke_e283_CONCURRENT_corpus_resume.pt" if SMOKE
                         else "e283_CONCURRENT_corpus_resume.pt"), dev)
    sd_c = conc["sd"]
    net_c = G1.evl_load(sd_c)
    cells_c = {"gm12": G1.battery_cell(net_c, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_c, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_c, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_c, *r_eval_xy)}
    d_final = flat_params_cpu(net_c) - fact_flat
    loads_final = rooms.displacement_loads(d_final, ROOM_MODE)
    del net_c
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None
    corp_ce = [v["ce"] for v in conc["corpus_ledger"].values()]
    corp_gn = [v["gn_clipped"] for v in conc["corpus_ledger"].values()]
    corp_gin = [v["g_in_room_frac"] for v in conc["corpus_ledger"].values()
                if v.get("g_in_room_frac") is not None]
    post_ck = save_ckpt(
        "e283_ESTABLISHED-CONCURRENT_post", sd_c,
        {"desc": "e283 ARM-ESTABLISHED-CONCURRENT post-phase state: the "
                 "committed quiet-formed 10k fact + 400 corpus steps (the "
                 "matching cadence, fresh AdamW, generator "
                 f"{CORPUS_GEN_SEED}) — NO install, NO cons",
         "arm": CONC_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"][CONC_ARM] = {
        "desc": ARM_DESC[CONC_ARM], "phase": {
            "traj": conc["traj"], "corpus_ledger": conc["corpus_ledger"],
            "disp_ledger": conc["disp_ledger"],
            "corpus_ce_median": med(corp_ce),
            "corpus_gn_clipped_median": med(corp_gn),
            "corpus_g_in_room_frac_median": med(corp_gin),
            "chunk_table": conc["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_c,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final.double().numpy())), **loads_final},
            "checkpoint": post_ck,
            "resumed_final": bool(conc.get("resumed_final", False)),
        }}
    log(f"ARM-{CONC_ARM} phase done: post g0 {cells_c['g0']:.6f} "
        f"(x{cells_c['g0'] / FACT_BASELINE_G0:.4f}) g-12 "
        f"{cells_c['gm12']:.6f} CE_R {cells_c['ce_r']:.4f} | drift-from-fact "
        f"|d| {float(np.linalg.norm(d_final.double().numpy())):.3f} "
        f"in-room {loads_final['in_own_room']:.4f} | corpus CE med "
        f"{med(corp_ce):.4f}")
    write_partial("P4 the established-concurrent phase done (the full "
                  "trajectory + drift ledger)")

    # ================= P5: ADJUDICATION (the frozen bars) ================
    traj = conc["traj"]
    g0_mile = {int(t["step"]): t["g0_pz"] for t in traj}
    ratio_mile = {t: g0_mile[t] / FACT_BASELINE_G0 for t in sorted(g0_mile)}
    ratio_400 = cells_c["g0"] / FACT_BASELINE_G0
    ratio_session_denom = cells_c["g0"] / max(fact_g0, RATIO_DEN_FLOOR)
    ratio_min = min(ratio_mile.values())
    ctrl_ratio = ctrl["traj"][-1]["survival_ratio_vs_committed"]
    steps_sorted = sorted(g0_mile)
    max_up_swing = max((g0_mile[b] - g0_mile[a]
                        for i, a in enumerate(steps_sorted)
                        for b in steps_sorted[i + 1:]), default=0.0)
    interior_below = any(r < SURVIVE_FRAC for t, r in ratio_mile.items()
                         if t < max(steps_sorted)) and ratio_400 >= SURVIVE_FRAC
    non_mono = bool(max_up_swing > NONMONO_SWING or interior_below)

    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif ratio_400 < SURVIVE_FRAC:
        verdict = "ESTABLISHED-DIES"
        clause = (f"the established write also dies (ratio_400 "
                  f"{ratio_400:.4f}x < {SURVIVE_FRAC:.0%}x) — the "
                  "collision is content-agnostic about age; retention and "
                  "formation die together; the 237k retention barrier "
                  "(e271) generalizes down to 10k for established facts "
                  "(the trajectory verbatim: "
                  + "; ".join(f"t{t} x{r:.4f}"
                              for t, r in ratio_mile.items()) + ")")
    elif (SURVIVE_FRAC <= ratio_400 < PARTIAL_FRAC) or non_mono:
        why = []
        if SURVIVE_FRAC <= ratio_400 < PARTIAL_FRAC:
            why.append(f"PARTIAL-DECAY (ratio_400 {ratio_400:.4f}x inside "
                       f"the [{SURVIVE_FRAC:.0%}x, {PARTIAL_FRAC:.0%}x) "
                       "partial-decay band)")
        if non_mono:
            why.append(f"NON-MONOTONE (max upward swing {max_up_swing:.4f} "
                       f"> {NONMONO_SWING}; interior-below-bar "
                       f"{bool(interior_below)})")
        verdict = "MIXED"
        clause = ("; ".join(why) + f" — ratio_400 {ratio_400:.4f}x, "
                  f"min milestone x{ratio_min:.4f}; the trajectories "
                  "verbatim")
    else:
        verdict = "ESTABLISHED-SPARED"
        clause = (f"the established write SURVIVES (ratio_400 "
                  f"{ratio_400:.4f}x >= {SURVIVE_FRAC:.0%}x through 400 "
                  f"corpus steps, monotone within the {NONMONO_SWING} "
                  "swing freeze) while the forming write died at "
                  f"{E268_RATIO:.6f}x — the collision kills FORMATION, not "
                  "the formed; the barrier is formation-time only; T244's "
                  "AFTER arm lands as 'the corpus is an additive abrasive "
                  "only for forming writes.'")

    log("=" * 78)
    log(f"E283 VERDICT: {verdict}")
    log(f"  the fact (loaded):     post g0 {fact_g0:.8f} (committed "
        f"{FACT_BASELINE_G0:.8f})")
    log(f"  {CTRL_ARM}: post g0 "
        f"{ctrl['traj'][-1]['g0_pz']:.8f} (x{ctrl_ratio:.4f}; the null)")
    log(f"  {CONC_ARM}: post g0 {cells_c['g0']:.8f} "
        f"(x{ratio_400:.4f} committed-denom; x{ratio_session_denom:.4f} "
        f"session-denom)")
    log(f"  milestones: "
        + "; ".join(f"t{t} x{r:.4f}" for t, r in ratio_mile.items()))
    log(f"  the forming reference (e268): x{E268_RATIO:.6f}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> ESTABLISHED-DIES -> MIXED "
                           "(PARTIAL-DECAY / NON-MONOTONE) -> "
                           "ESTABLISHED-SPARED (frozen; MIXED's "
                           "partial-decay carve-out takes precedence over "
                           "SPARED's floor — named at birth, no bar moved)",
        "gates_pass": gates_pass,
        "reads": {
            "the_fact_loaded": {"g0": fact_g0, "committed": FACT_BASELINE_G0,
                                "gm12": fact_gm12, "ce_r": fact_ce_r},
            CTRL_ARM: {"post_g0": ctrl["traj"][-1]["g0_pz"],
                       "survival_ratio": ctrl_ratio,
                       "max_abs_g0_drift": ctrl["max_abs_g0_drift"],
                       "note": "the storage-decay null (no steps)"},
            CONC_ARM: {"post_g0": cells_c["g0"],
                       "survival_ratio_committed_denom": ratio_400,
                       "survival_ratio_session_denom": ratio_session_denom,
                       "post_gm12": cells_c["gm12"],
                       "post_ce_r": cells_c["ce_r"],
                       "traj_g0": {t["step"]: t["g0_pz"] for t in traj},
                       "survival_ratio_milestones": ratio_mile,
                       "min_milestone_ratio": ratio_min,
                       "corpus_ce_median": med(corp_ce),
                       "corpus_gn_clipped_median": med(corp_gn),
                       "corpus_g_in_room_frac_median": med(corp_gin),
                       "drift_ledger": conc["disp_ledger"],
                       "drift_from_fact_final_in_room":
                           loads_final["in_own_room"]},
            "the_forming_reference_e268": {
                "serial_post_g0": E268_SERIAL_POST,
                "concurrent_post_g0": E268_CONCURRENT_POST,
                "ratio": E268_RATIO,
                "note": "the SPARED bar's '0.0002x' cite (hard-bound in "
                        "G_PARENTS)"},
            "the_237k_retention_barrier_e271": {
                "concurrent_post_g0": E271_CONCURRENT_POST,
                "ratio": E271_RATIO},
            "non_monotone_reads": {
                "max_upward_swing": max_up_swing,
                "bar": NONMONO_SWING,
                "interior_below_bar": bool(interior_below),
                "non_monotone": non_mono},
        },
        "scatter_disclosure": {
            "read_determinism": "same-session repeated CPU reads are exact "
                                "(G_CONTROL measured "
                                f"{ctrl['max_abs_g0_drift']:.1e}); "
                                "cross-session |d post g0| ~ 5e-7-1e-6 on a "
                                "bit-identical arm (the family law; "
                                "G_FACTLOAD's measured delta co-reported)",
            "the_two_denominators": {"committed": FACT_BASELINE_G0,
                                     "session_loaded": fact_g0,
                                     "ratios_agree_to": abs(ratio_400
                                                            - ratio_session_denom)},
            "n1_caveat": "n=1 per arm, one lineage, one session (the "
                         "g-series standing lottery note carried verbatim)",
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P5 ADJUDICATED (the frozen bars)")

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the two arms share ONE loaded artifact (the committed "
            "quiet-formed 10k write, md5/flat-md5/behavior-gated), the "
            "same milestone cadence, the same reads — the ONLY delta is "
            "the corpus stream RUNNING (400 steps of the family's "
            "standard corpus step through the rig's fresh AdamW) vs "
            "PAUSED (no steps). Any decay in the concurrent arm is the "
            "corpus traffic's doing (gradients + clip + Adam + wd) or "
            "nothing is; the control nulls time, storage and the read "
            "pipeline itself."),
        "the_wd_disclosure": (
            "the rig's AdamW wd 0.1 rides the corpus steps at the paired "
            "lr — a ~3.4% uniform weight shrink over the 400-step cosine "
            "(sum lr ~ 0.344), itself mostly an in-room-direction shrink "
            "of the ~94%-in-room fact. This channel is part of the "
            "intervention's body (the same channel acted on e268's "
            "forming arm) — measured in the drift ledger, never assumed"),
        "n_and_scope": ("n=1 per arm, one lineage, one session; the arms' "
                        "DIFFERENCE is the registered object; the corpus "
                        "stream is the intervention's own body, ledgered "
                        "per step"),
        "loads_measured_not_nominal": (
            "the drift ledger is measured per milestone in fp64: the "
            "state's drift from the loaded fact with its in-room share "
            "(the roach-motel continuation), the remaining-from-base "
            "room occupancy, the corpus stream's realized interval/"
            "cumulative displacement with in-room fractions, and the "
            "corpus gradient's own in-room fraction every 10 steps — "
            "never nominal"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "three branches and the trajectories are "
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
                         "note": "read-only here — the established fact"},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
            "rooms": rooms_ck,
            "post_state_concurrent": post_ck,
        },
        "machinery": {
            "corpus_phase": "THIS file's chunked_corpus_phase (the cell's "
                            "one new driver): the family's standard corpus "
                            "step VERBATIM (e268's registered form, fresh "
                            "generator 28301) at the matching cadence "
                            "through the rig's fresh AdamW — no install, "
                            "no projection, the drift ledger at the "
                            "milestones",
            "control": "THIS file's serial_control_reads: the paused-stream "
                       "storage-decay null (no steps; the reads are the "
                       "arm)",
            "cons": "NONE (the e278 precedent, T259/e281; the post state "
                    "checkpointed)",
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

    # ================= P10: figures ======================================
    make_collision_plot(RD, metrics, verdict, clause, thermal_log)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e283_established_collision.png")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_collision_plot(rd, metrics, verdict, clause, thermal_log):
    """THE CELL'S HEADLINE FIGURE: the survival trajectories (the "
    established write under the corpus vs the paused null vs the forming "
    reference), the discriminator bars, the roach-motel drift ledger, the
    corpus stream's realized displacement geometry, the corpus ledgers, and
    the thermal envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    ctrl = metrics["arms"][CTRL_ARM]["phase"]
    conc = metrics["arms"][CONC_ARM]["phase"]
    ct = ctrl["traj"]
    kt = conc["traj"]
    dl = {int(d["step"]): d for d in conc["disp_ledger"]}
    cl = conc["corpus_ledger"]

    # (0,0) THE SURVIVAL TRAJECTORIES
    ax = axes[0, 0]
    ax.plot([t["step"] for t in ct],
            [max(t["g0_pz"], 1e-7) for t in ct], "s--", lw=1.4, ms=4,
            color="tab:blue", label=f"{CTRL_ARM} (stream PAUSED)")
    ax.plot([t["step"] for t in kt],
            [max(t["g0_pz"], 1e-7) for t in kt], "o-", lw=1.8, ms=5,
            color="tab:red", label=f"{CONC_ARM} (the corpus alone)")
    ax.axhline(FACT_BASELINE_G0, color="black", ls=":", lw=1.3,
               label=f"the loaded fact {FACT_BASELINE_G0:.4f}")
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.2, label=f"the {SURVIVE_FRAC:.0%}x DIES bar "
               f"({SURVIVE_FRAC * FACT_BASELINE_G0:.4f})")
    ax.axhline(PARTIAL_FRAC * FACT_BASELINE_G0, color="darkorange", ls="--",
               lw=1.0, label=f"the {PARTIAL_FRAC:.0%}x partial-decay ceiling")
    ax.axhline(E268_CONCURRENT_POST, color="gray", ls="-.", lw=1.0,
               label=f"the forming write's death (e268) "
               f"{E268_CONCURRENT_POST:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("corpus step t (NO install — the phase's own counter)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE SURVIVAL TRAJECTORIES (the established write under "
                 "the corpus)", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE DISCRIMINATOR READS
    ax = axes[0, 1]
    names = ["loaded fact\n(t=0)", CTRL_ARM, CONC_ARM]
    posts = [FACT_BASELINE_G0, ctrl["post_cells"]["g0"],
             conc["post_cells"]["g0"]]
    bars = ax.bar(names, posts,
                  color=["black", "tab:blue", "tab:red"], alpha=0.88)
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.4, label=f"DIES bar {SURVIVE_FRAC:.0%}x")
    ax.axhline(PARTIAL_FRAC * FACT_BASELINE_G0, color="darkorange", ls="--",
               lw=1.2, label=f"partial ceiling {PARTIAL_FRAC:.0%}x")
    for b, v in zip(bars, posts):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.5f}\n(x{v / FACT_BASELINE_G0:.3f})", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_ylabel("post g0 at t=400 (the WRITE read)")
    ax.set_title(f"THE DISCRIMINATOR READS — the forming reference died at "
                 f"x{E268_RATIO:.6f} (e268)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, axis="y")

    # (0,2) THE ROACH-MOTEL DRIFT LEDGER (the dispatch's named read)
    ax = axes[0, 2]
    steps = sorted(dl)
    ax.plot(steps, [dl[s]["drift_from_fact_in_room_frac"] for s in steps],
            "o-", lw=1.6, ms=5, color="tab:red",
            label="drift from the fact: ||P d||/||d||")
    ax.plot(steps, [dl[s]["remaining_from_base_in_room_frac"]
                    for s in steps], "s--", lw=1.4, ms=4, color="tab:purple",
            label="the write's remaining room occupancy "
                  "(theta_t - base)")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.2, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(LADDER[0][0] / 2739072):.4f}")
    ax.axhspan(E278_MISSILE_DISP_IN_ROOM[0], E278_MISSILE_DISP_IN_ROOM[1],
               color="gray", alpha=0.12,
               label="e278's roach-motel band (missile disp 0.48-0.60)")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("in-room share of the displacement")
    ax.set_ylim(0, 1.05)
    ax.set_title("THE ROACH-MOTEL CONTINUATION (does the established "
                 "write's own drift stay in-room?)", fontsize=9.0)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)

    # (1,0) THE CORPUS STREAM'S REALIZED DISPLACEMENT
    ax = axes[1, 0]
    steps = sorted(dl)
    ax.plot(steps, [dl[s]["corpus_disp_interval_in_room_frac"]
                    for s in steps], "o-", lw=1.5, ms=5, color="tab:red",
            label="interval displacement in-room frac")
    ax.plot(steps, [dl[s]["corpus_disp_cum_in_room_frac"] for s in steps],
            "s--", lw=1.3, ms=4, color="tab:green",
            label="cumulative displacement in-room frac")
    gsteps = sorted(int(k) for k in cl)
    ax.plot(gsteps,
            [cl[str(k)].get("g_in_room_frac") if str(k) in cl
             else cl[k].get("g_in_room_frac") for k in gsteps], ".", ms=3,
            color="tab:orange", alpha=0.6,
            label="corpus GRADIENT in-room frac (the re-aiming contrast)")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.2, label="volume overlap sqrt(k/N)")
    ax.set_xlabel("corpus step t")
    ax.set_ylabel("in-room fraction")
    ax.set_title("WHERE THE CORPUS STREAM ACTUALLY WALKED (gradient vs "
                 "realized displacement)", fontsize=9.0)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)

    # (1,1) THE CORPUS LEDGERS
    ax = axes[1, 1]
    steps = sorted(int(k) for k in cl)
    ax.plot(steps, [cl[str(k)]["ce"] if str(k) in cl else cl[k]["ce"]
                    for k in steps], "o-", lw=1.3, ms=3, color="tab:red",
            label="corpus step CE")
    ax.set_xlabel("corpus step t")
    ax.set_ylabel("corpus CE (nats)")
    ax2 = ax.twinx()
    ax2.plot(steps, [cl[str(k)]["gn_clipped"] if str(k) in cl
                     else cl[k]["gn_clipped"] for k in steps], "s--",
             lw=0.9, ms=2.5, color="dimgray", alpha=0.7,
             label="clipped ||g||")
    ax2.set_ylabel("clipped grad norm", fontsize=8.5)
    ax.set_title(f"THE CORPUS LEDGERS (CE med "
                 f"{conc['corpus_ce_median']:.4f}; |g| med "
                 f"{conc['corpus_gn_clipped_median']:.3f})", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0)
    ax.grid(alpha=0.25)

    # (1,2) THE THERMAL ENVELOPE + THE CONTROL NULL
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
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; the control's null: "
                 f"max |g0 drift| "
                 f"{ctrl['max_abs_g0_drift']:.1e})", fontsize=9.0)

    fig.suptitle(f"E283 — THE ESTABLISHED-FACT COLLISION TEST -> {verdict}",
                 fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e283_established_collision.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
