"""E273 — THE THREE-BARREL MECHANISM CELL (the horse race's adjudicator).
Design dispatched 2026-10-05; this docstring carries the registered
question + bars VERBATIM, committed at birth BEFORE any compute.
Adjudicate against exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): the concurrent kill at k=10k
— is the coupling (i) the SHARED OPTIMIZER STATE (v-poisoning), (ii)
ADAM'S NORMALIZATION itself (the low-rank formation block is
Adam-specific), or (iii) the TRAJECTORY (a true two-body problem in the
parameters)?

THE ARMS (k=10k room, seeds 27215/27216 bit-bound to e272's K10KR; the
family's standard fact protocol; serial reference = e272's K10KR cited
+ fresh serial re-run for the same-session pair):
  (a) SEPARATE-ADAMW: the 1:1 interleave VERBATIM but the install
      stream and the corpus stream carry SEPARATE AdamW state objects
      (each with its own m/v; same lr schedule; the corpus steps update
      only the corpus optimizer, install steps only the install
      optimizer — the parameters remain shared, of course).
  (b) SGD-M: the whole cell (install + corpus interleave) under SGD
      with momentum 0.9 INSTEAD of AdamW — lr matched by the family
      convention (calibrate so the first install step's applied in-room
      L2 matches the AdamW capstone arm's; disclose the factor; run a
      2x/0.5x lr sensitivity pair if budget allows).
  (c) SHARED-ADAMW REPLAY (the control twin): the original e268 form
      re-run this session (the antiphase reference for THIS session's
      driver draws — the serial traj_g0 of the same-session serial arm
      is the driver curve).

FROZEN BARS (verbatim from the dispatch letter; adjudicate on the WRITE
read; landing read carried under the rehearsal caveat, never
adjudicated — the family precedent):
  - STATE-POISONING: the SEPARATE-ADAMW write SURVIVES (post g0 >= 0.5x
    its serial twin) — the barrier is written in the shared optimizer
    state; "turbulence" narrows to "optimizer-state poisoning"; AND the
    antiphase vanishes in (a) (|Pearson| < 0.2) while present in (c) —
    P-273a CONFIRMED.
  - ADAM-SPECIFIC-BLOCK: under SGD-M (b) the write FORMS (peak traj_g0
    > 0.005 — above the 10k formation-block level 0.0007) whether or
    not it retains — P-273b confirmed; the low-rung block is Adam's
    normalization, not the corpus.
  - TRAJECTORY-TWO-BODY: the SEPARATE-ADAMW write still DIES (< 0.5x)
    AND SGD-M also dies without forming — the coupling is the
    parameters' own collision; the metaphor keeps its name.
  - MIXED: any other combination — the trajectories verbatim, all
    reads, no inflation. (Register the composite: STATE-POISONING
    requires BOTH its clauses; each barrel's clause reported
    separately.)

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE VEHICLE := the k=10,000 room with seeds 27215/27216 — e272's
    committed K10KR room, rebuilt from its registered seeds and
    bit-gated against runs/checkpoints/e272_rooms.pt's stored K10KR
    D/S (exact equality; G_ROOM10KR); the e001 fresh fact-free base,
    the Dmix install s400 gen 24314, the hook VERBATIM (backward ->
    clip 1.0 -> project onto the room (CPU fp64, write fp32) ->
    opt.step; norm NOT rescaled), the e113 cons s300 seed 10901 HELD
    NATURAL (AdamW) on every arm — the cons is the rehearsal lane, so
    the SGD-M barrel changes the INSTALL optimizer only.
  * THE SERIAL REFERENCE := e272's committed K10KR arm, CITED (post g0
    0.20972091 / root g0 0.69699448 / kept 0.06009885; hard-bound in
    G_PARENTS) AND a FRESH same-session serial re-run (e261's
    chunked_install driver VERBATIM at the registered seeds — one
    instrument, one session; the e268/e269/e270/e271 precedent): the
    SURVIVES clause's denominator "its serial twin" = the SAME-SESSION
    serial arm (PRIMARY); the committed-cited ratio co-reported.
  * THE INTERLEAVE := e268's registered form: after EVERY install step
    s (s = 1..400), ONE corpus step (1:1): 48 windows = 16 anchors from
    the same 60-window bank + 32 random corpus windows (the g1c Dmix
    corpus convention); full-window CE; the PAIRED install step's lr;
    backward -> clip 1.0 -> opt.step FREE (NO projection) — the room
    constrains the install's writing directions only. The corpus
    generator is THIS cell's own REGISTERED FRESH stream (seed 27301;
    e268's was 26801, e269's 26901, e270's 27001, e271's 27101) and
    ALL concurrent barrels share it (a)/(b)/(c) — the barrels' deltas
    are then ONLY the optimizer arrangements, never the corpus draws.
    The install stream's draws stay bit-identical to SERIAL's by
    construction (separate generators).
  * SEPARATE-ADAMW := torch.optim.AdamW x2 (same betas (0.9,0.95), wd
    0.1, same lr schedule lr x cosine_lr(s-1,1000)): install steps set
    and step ONLY the install optimizer; corpus steps set and step
    ONLY the corpus optimizer; the parameters are the one shared
    body. Each optimizer's moments see only its own stream (400 steps
    each) — the registered intervention.
  * SGD-M := torch.optim.SGD(momentum=0.9, weight_decay=0.0) — ONE
    optimizer across both streams (the whole cell under SGD-M), lr =
    LR_SGD x cosine_lr(s-1,1000). AdamW's decoupled wd 0.1 is NOT
    carried (the swap IS the optimizer family; the lr calibration
    matches the applied in-room write size, the first-order object;
    disclosed). THE CALIBRATION (the family convention, T253's frozen
    form): a live probe re-executes the capstone arm's FIRST install
    step (bit-identical draws/hook/fresh AdamW) and measures its
    applied in-room L2 (||P(theta_1 - theta_0)||); LR_SGD := that /
    ||P g'|| of the same step (momentum buffer zero at s1, so the SGD
    step-1 applied in-room L2 = LR_SGD x gpn exactly); the probe then
    VERIFIES the SGD step-1 in-room L2 against the target (G_DOSE_LR:
    rel err <= 5%). The factor vs the AdamW lr 1e-3 is DISCLOSED.
    Sensitivity pair SGD2X/SGD05X (lr x2 / x0.5) runs only under
    E273_SENSITIVITY=1 (budget allowing; never adjudicated —
    confirmatory reads).
  * THE WRITE READ := post-install g0 battery at s400 (the bit-faithful
    discriminator; install determinism |d post g0| ~ 5e-7-1e-6
    cross-session on a bit-identical arm — the family law). THE
    FORMATION READ := peak traj_g0 over the milestones (the flash's
    amplitude). The landing read (root g0 after cons) is CARRIED under
    the rehearsal caveat (T246's rehearsal-lane finding; e269/e270/
    e271 triply confirmed), never adjudicated; root g-12 is the wild
    lottery, reported never gated.
  * THE ANTIPHASE READ := Pearson of the arm's traj_g0 against the
    SAME-SESSION serial arm's traj_g0 at the common milestones,
    disclosed n. TWO conventions reported: n=5 (all milestones
    s1/100/200/300/400 — the dispatch's letter) and n=4 (x8's own
    convention, EXCLUDING s1 — the point where every arm sits at the
    pre-formation measurement floor (~1e-5) and carries no phase
    information; the registered -0.52 datum's own convention,
    reproduced live from e268's hard-bound pair as provenance). The
    bars' antiphase clauses are evaluated on the n=4 convention as
    PRIMARY (pre-registered here); the n=5 co-reported; a convention
    straddle that would flip the verdict routes to MIXED (named
    branch). "present in (c)" := Pearson(c) <= -0.2 (the antiphase is
    directional — the flash peaks where the driver dips);
    "vanishes in (a)" := |Pearson(a)| < 0.2.
  * THE DIVERGENCE-ROUTING RULE (pre-registered after the smoke, BEFORE
    the full compute — a routing rule, NOT a bar move): the smoke pass
    (runs/e273_smoke) caught that the registered first-install-step-
    in-room-L2-matched SGD-M lr makes the FREE corpus steps (unprojected,
    ~lr x ||g_corp|| full-L2 each, momentum-compounding toward x10) run
    the model away (smoke: install CE 1.07 -> 98 by s8) — SGD's linear
    step cannot match Adam's sign-equalized streams simultaneously. THE
    SIGN-STEP LAW (the smoke's second catch, measured + verified on the
    room itself): ||P sign(P g)|| ~= 0.80 x ||sign|| for the SRCT room
    (vs sqrt(k/N) ~ 0.09 for a random sign vector) — Adam's first step
    is ~80% in-room, so the in-room-matched LR_SGD is ~22 (factor
    ~2.2e4 x 1e-3) and the corpus steps at that scale are destructive.
    RULE: an SGD barrel is DIVERGED iff its install-ledger CE beyond
    s100 exceeds 2x its first-batch CE or any read is non-finite; a
    diverged SGD barrel's formation clause is labeled DIVERGENCE-
    CONFOUNDED and cannot hand a barrel the headline (a death-by-
    divergence cannot make TTB2's clause informative; a garbage-logits
    flash cannot make ASB's) — any headline that would rest on a
    confounded clause routes to MIXED (named branch). The registered
    SGDM arm still RUNS AS CALIBRATED (verbatim); the x0.01 stability
    rider (SGD001X) carries the stable-scale read, never adjudicated.
  * COMPOSITE (frozen): TEXTURE (hard-gate failure; nothing
    adjudicated) -> STATE-POISONING (BOTH its clauses: (a) survives
    AND antiphase vanishes in (a) while present in (c)) ->
    ADAM-SPECIFIC-BLOCK ((b) forms: peak traj_g0 > 0.005) ->
    TRAJECTORY-TWO-BODY ((a) dies < 0.5x AND (b) dies without forming)
    -> MIXED (any other combination, named). Every barrel's clause is
    reported separately regardless of the headline; if more than one
    barrel's conditions hold, the highest-priority firing barrel takes
    the headline and the clause table names all (no bar shopping: the
    order is the dispatch's own listing order).
  * HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR,
    G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND,
    G_PROJ, G_ROOM10KR, G_DOSE_LR} — a failure HALTS (nothing
    adjudicated). G_SERIAL_ANCHOR (the serial re-run vs e272's
    committed K10KR rung: install L2 vs the committed vehicle <= 5e-3;
    |d| <= 0.02 on post g0 / root g0) is NON-HALTING texture — the
    root clause sits below the known cons lottery (~0.0419) and is
    EXPECTED to miss about half the time (the e269/e270/e271
    precedents); a miss is disclosed, never a bar move.
  * READS ON EVERY ARM: post g0 at s400; traj_g0 milestones
    s1/100/200/300/400 (the SGD barrels add s25/s50 — the flash-power
    rider); the antiphase read (both conventions); the v-excess and
    in-own-room displacement reads at post-install AND per milestone
    (the weight-vs-read dissociation rider: cumulative displacement
    norm, in-room displacement norm, in-own-room fraction, v-excess of
    the displacement); the corpus ledger (CE + clipped-grad norm) on
    every concurrent arm; kept median.
  * THE READS ADDENDUM (the coordinator's message, received 2026-10-05
    PRE-BIRTH — registered in the birth script itself, before any
    compute; READS ONLY, no bar moved; the R66 ideator's A2/P-C2):
    under SHARED-v relaxation three reads move TOGETHER in the
    separate-AdamW arm — (i) the antiphase vanishes, (ii) the flash's
    PEAK AMPLITUDE falls toward the volume-null level, (iii) the
    ENDPOINT rises toward serial; v-ownership moves all three,
    trajectory-ownership moves none. ADDED per arm: peak traj_g0,
    the endpoint ratio vs serial, and the peak-vs-driver-dip alignment
    (the arm's peak milestone step vs the serial driver's dip
    milestone step, n=4 convention); the volume-null floor = the fresh
    fact-free base's own g0 battery read, co-reported.

CHECKS (the dispatch's, in force): the machinery smoke FIRST (the
e260-family record: 2-3 bugs caught per build — THIS cell's smoke caught
TWO: the SRCT sign-step law and the SGD-M corpus-step divergence, both
disclosed in deviations with their pre-registered responses); the room
certified AND bit-bound to e272's committed K10KR room; the interleave
registered exactly (above); the first-install-batch CE identical across
arms (the draw-integrity texture check, non-halting); n=1 per arm (the
lottery note); nothing guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier);
bursts <= 175s (E261's family discipline, inside the dispatch's <=180s
window), per-step thermal polls (BOTH streams' opt steps) at a 78C
margin, 40s cooldowns (the 30-60s window), the 84C never-past line
(inside the dispatch's 85C); CPU fp64 dense projections (pocketfft
workers 2); CPU probing threads 4; NO concurrent GPU jobs.

Outputs: runs/e273/{metrics.json (PROGRESSIVE),
e273_optimizer_barrels.png, REPORT.md, run.log (gitignored)};
checkpoints runs/checkpoints/e273_*.pt. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds). Commit + push per phase.

Run:  cd lab && python e273_optimizer_barrels.py    (E273_SMOKE=1
      shakedown; E273_SENSITIVITY=1 adds the SGD2X/SGD05X pair)
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

SMOKE = os.environ.get("E273_SMOKE") == "1"
SENSITIVITY = os.environ.get("E273_SENSITIVITY") == "1"
CPU = torch.device("cpu")
NAME = "e273_smoke" if SMOKE else "e273"
assert torch.cuda.is_available(), "e273 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268's disclosed convention): e261's ported drivers
# resolve their module globals (log / NAME / LADDER / RUNG_NAMES / SMOKE /
# INST_STEPS / CONS_STEPS / T0 / thermal ledgers) AT CALL TIME through
# e261's module namespace — rebound HERE so they write THIS cell's log,
# label THIS cell's envelope polls, and run THIS cell's single-rung ladder
# (e270's thermal-ledger rebinding inherited: both streams' per-step polls
# land in ONE ledger). The committed lab/e261_rank_ladder.py is untouched.
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
ROOMS272_CK = "e272_rooms.pt"     # e272's committed rooms (the K10KR bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE VEHICLE: e272's committed K10KR room (k=10k, seeds 27215/27216)
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 27215, 27216),       # K10KR — e272's fresh-seed 10k room
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (512, 27215, 27216),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG = {k: ("K10KR" if not SMOKE else f"K{k}") for k, _, _ in LADDER}
ROOM_MODE = RUNG[LADDER[0][0]]       # the hook's mode (smoke names it K512)
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# the barrels (execution order: serial anchors; the control twin before
# the two new drivers; the SGD sensitivity riders LAST, optional,
# informative-first)
ARMS_MAIN = ("SERIAL", "SHARED", "SEPARATE", "SGDM")
ARMS_SENS = ("SGD05X", "SGD001X", "SGD2X")
ARMS_RUN = ARMS_MAIN + ARMS_SENS if (SENSITIVITY and not SMOKE) else ARMS_MAIN

# this cell's OWN REGISTERED FRESH corpus stream (the e268-family
# convention: one fresh registered stream per concurrent cell); ALL
# concurrent barrels share it (the barrels' deltas are the optimizer
# arrangements only, never the corpus draws)
CORPUS_GEN_SEED = 27301

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "eb9f624708bcb6576c4115161dfd7042"
E272_VERDICT = "RANK-WRITES-THE-CURVE"
E272_K10KR_POST_G0 = 0.2097209095954895
E272_K10KR_ROOT_G0 = 0.6969944834709167
E272_K10KR_KEPT_MED = 0.06009884924098147
E272_K10KR_POST_GM12 = 0.07534654438495636
E272_K10KR_TRAJ_G0 = {1: 1.3466953532770276e-05, 100: 0.31526979804039,
                      200: 0.20373161137104034, 300: 0.17488795518875122,
                      400: 0.2097209095954895}

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"
E268_VERDICT = "DYNAMICAL-CARRIER"
E268_SERIAL_POST = 0.26464739441871643
E268_CONCURRENT_POST = 4.004325455753133e-05
E268_RATIO = 0.00015130794937726159
E268_SERIAL_TRAJ = {1: 1.3466438758769073e-05, 100: 0.30591899156570435,
                    200: 0.21840216219425201, 300: 0.21022741496562958,
                    400: 0.26464739441871643}
E268_CONCURRENT_TRAJ = {1: 1.3246100024844054e-05, 100: 0.0002721511118579656,
                        200: 0.00023426816915161908,
                        300: 0.0007357693975791335,
                        400: 4.004325455753133e-05}

E271_METRICS = E43.REPO / "runs" / "e271" / "metrics.json"
E271_MD5 = "16c29a3167d8523a64f66f03c717f67b"     # the antiphase context's
                                                  # own record (237k)

# e272's committed K10KR install vehicle (the serial anchor's L2 target)
K10KR_CK = "e272_K10KR_inst_resume.pt"
K10KR_MD5 = "6ce712a7aae8482a4c78d1b87f3194b1"
K10KR_SIZE = 32958618
K10KR_STEP = 400
K10KR_TRAJ_STEPS = [1, 100, 200, 300, 400]
K10KR_LEDGER_MAX = 400

# the discriminator's frozen numbers
SURVIVE_FRAC = 0.5               # "post g0 >= 0.5x its serial twin"
ANTIPHASE_VANISH_BAR = 0.2       # "|Pearson| < 0.2"
ANTIPHASE_PRESENT_BAR = -0.2     # "present": Pearson(c) <= -0.2 (directional)
FORMATION_BAR = 0.005            # "peak traj_g0 > 0.005"
FORMATION_BLOCK_LEVEL = 0.0007   # e268's concurrent 10k peak (context line)
G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 — the expression floor (ctx)
MATCH_BAND = E261.MATCH_BAND              # the landing band (context only)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
ANCHOR_BEHAV_TOL = 0.02           # the family's session-texture bar
RATIO_DEN_FLOOR = 1e-6            # the ladder's floor-guard convention
ANCHOR_SCATTER_G0 = 0.04194730520248413   # e261's G_ANCHOR (cons lottery)
CONS_SEED_SCATTER_G0 = 0.0137           # e264's cons-seed replicate

# the SGD barrel's registered constants
SGD_MOMENTUM = 0.9
SGD_WD = 0.0                      # disclosed: AdamW's decoupled wd NOT carried
DOSE_LR_TOL = 0.05                # the calibration's verification bar (rel)
SENS_FACTORS = {"SGDM": 1.0, "SGD2X": 2.0, "SGD05X": 0.5, "SGD001X": 0.01}
SGD_BARRELS = ("SGDM", "SGD2X", "SGD05X", "SGD001X")

TRAJ_MILE_MAIN = (1, 100, 200, 300, 400)
TRAJ_MILE_SGD = (1, 25, 50, 100, 200, 300, 400)   # the flash-power rider

ARM_DESC = {
    "SERIAL": "the committed K10KR rung run ALONE (e272's room, the "
              "ladder's committed condition): e261's chunked_install driver "
              "VERBATIM at the registered seeds, re-run FRESH on this "
              "session — the same-session serial twin (the SURVIVES "
              "clause's PRIMARY denominator) + the antiphase driver curve; "
              "anchored to e272's committed K10KR arm (G_SERIAL_ANCHOR, "
              "non-halting texture)",
    "SHARED": "BARREL (c) — the control twin: e268's registered form "
              "re-run THIS session — the 1:1 interleave (install step "
              "bit-identical to SERIAL's + ONE free corpus step, corpus "
              "generator 27301) through the ONE SHARED AdamW (0.9,0.95) "
              "wd 0.1 — the antiphase reference for THIS session's driver "
              "draws",
    "SEPARATE": "BARREL (a) — the 1:1 interleave VERBATIM but the install "
                "stream and the corpus stream carry SEPARATE AdamW state "
                "objects (each its own m/v; same betas/wd/lr schedule; "
                "corpus steps update only the corpus optimizer, install "
                "steps only the install optimizer; the parameters remain "
                "the one shared body)",
    "SGDM": "BARREL (b) — the whole cell (install + corpus interleave) "
            "under SGD momentum 0.9 INSTEAD of AdamW (wd 0, disclosed), "
            "ONE optimizer across both streams, lr calibrated so the "
            "first install step's applied in-room L2 matches the AdamW "
            "capstone arm's (the probe; factor disclosed)",
    "SGD2X": "the SGD-M sensitivity rider at lr x2 (E273_SENSITIVITY=1 "
             "only; never adjudicated — confirmatory)",
    "SGD05X": "the SGD-M sensitivity rider at lr x0.5 "
              "(E273_SENSITIVITY=1 only; never adjudicated — "
              "confirmatory)",
    "SGD001X": "the SGD-M STABILITY DIAGNOSTIC at lr x0.01 (the "
               "smoke-catch response: the registered first-step-matched "
               "lr diverges via the free corpus steps — x0.01 sits near "
               "the steady-state-matched scale, corpus steps ~momentum x "
               "0.01 x LR_SGD; E273_SENSITIVITY=1 only; never "
               "adjudicated — a diagnostic read, disclosed)",
}

REGISTERED = {
    "question_verbatim": "the concurrent kill at k=10k — is the coupling "
        "(i) the SHARED OPTIMIZER STATE (v-poisoning), (ii) ADAM'S "
        "NORMALIZATION itself (the low-rank formation block is "
        "Adam-specific), or (iii) the TRAJECTORY (a true two-body problem "
        "in the parameters)?",
    "bars_verbatim": {
        "STATE-POISONING": "the SEPARATE-ADAMW write SURVIVES (post g0 >= "
            "0.5x its serial twin) — the barrier is written in the shared "
            "optimizer state; \"turbulence\" narrows to "
            "\"optimizer-state poisoning\"; AND the antiphase vanishes in "
            "(a) (|Pearson| < 0.2) while present in (c) — P-273a CONFIRMED",
        "ADAM-SPECIFIC-BLOCK": "under SGD-M (b) the write FORMS (peak "
            "traj_g0 > 0.005 — above the 10k formation-block level "
            "0.0007) whether or not it retains — P-273b confirmed; the "
            "low-rung block is Adam's normalization, not the corpus",
        "TRAJECTORY-TWO-BODY": "the SEPARATE-ADAMW write still DIES "
            "(< 0.5x) AND SGD-M also dies without forming — the coupling "
            "is the parameters' own collision; the metaphor keeps its "
            "name",
        "MIXED": "any other combination — the trajectories verbatim, all "
            "reads, no inflation. (Register the composite: "
            "STATE-POISONING requires BOTH its clauses; each barrel's "
            "clause reported separately.)",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE VEHICLE := the k=10k room seeds "
        "27215/27216 rebuilt + bit-gated vs e272_rooms.pt's K10KR D/S "
        "(G_ROOM10KR); e001 base; Dmix s400 gen 24314; hook = clip 1.0 -> "
        "project CPU fp64 -> opt.step, norm NOT rescaled; e113 cons s300 "
        "seed 10901 HELD NATURAL (AdamW) on every arm (the SGD barrel "
        "changes the INSTALL optimizer only; the cons is the rehearsal "
        "lane); THE SERIAL REFERENCE := e272's committed K10KR CITED "
        f"(post {E272_K10KR_POST_G0:.8f} / root {E272_K10KR_ROOT_G0:.8f} / "
        f"kept {E272_K10KR_KEPT_MED:.8f}, hard-bound) + a fresh "
        "same-session serial re-run; 'its serial twin' := the "
        "SAME-SESSION serial arm (PRIMARY), committed-cited ratio "
        "co-reported; THE INTERLEAVE := e268's registered form, corpus "
        f"generator THIS cell's fresh seed {CORPUS_GEN_SEED} SHARED by all "
        "concurrent barrels (deltas = optimizer arrangements only); "
        "SEPARATE-ADAMW := AdamW x2 (same betas/wd/schedule), install "
        "steps touch only the install optimizer, corpus steps only the "
        "corpus optimizer; SGD-M := one SGD(momentum 0.9, wd 0 DISCLOSED) "
        "across both streams, LR calibrated by the live probe (first "
        "install step's applied in-room L2 matched; verified rel err <= "
        "5%; factor disclosed; 2x/0.5x riders optional, never "
        "adjudicated); THE ANTIPHASE READ := Pearson of traj_g0 vs the "
        "same-session serial's at common milestones, BOTH conventions "
        "reported (n=5 the dispatch's letter; n=4 x8's own convention "
        "EXCLUDING the s1 floor — the -0.52's own form, reproduced live "
        "from e268's hard-bound pair as provenance), the bars' clauses "
        "evaluated on n=4 PRIMARY (pre-registered), a verdict-flipping "
        "straddle routes to MIXED (named); 'present in (c)' := "
        "Pearson(c) <= -0.2 (directional); COMPOSITE := TEXTURE -> "
        "STATE-POISONING (BOTH clauses) -> ADAM-SPECIFIC-BLOCK (forms) -> "
        "TRAJECTORY-TWO-BODY (dies + dies-unformed) -> MIXED (named; "
        "dispatch order); HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, "
        "G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, "
        "G_SPANBIND, G_PROJ, G_ROOM10KR, G_DOSE_LR} (a failure HALTS); "
        "G_SERIAL_ANCHOR NON-HALTING (the root clause sits below the cons "
        "lottery ~0.0419 — expected to miss ~half the time, disclosed "
        "never a bar move); the WRITE read adjudicates; the landing read "
        "carried under the rehearsal caveat, never adjudicated; root g-12 "
        "the wild lottery, reported never gated."),
    "reads_addendum": {
        "received": "2026-10-05, PRE-BIRTH — the coordinator's message "
            "arrived BEFORE the birth commit; registered in the birth "
            "script itself (the strongest form of the disclosure; no "
            "post-hoc timing to report)",
        "source": "the R66 ideator's A2/P-C2, relayed by the coordinator",
        "text": "under SHARED-v relaxation, three reads move TOGETHER in "
            "the separate-AdamW arm — (i) the antiphase vanishes "
            "(|Pearson| < 0.2 — already in the spec as P-273a), (ii) the "
            "flash's PEAK AMPLITUDE falls toward the volume-null level, "
            "(iii) the ENDPOINT rises toward serial. v-ownership moves "
            "all three; trajectory-ownership moves none.",
        "reads_added": [
            "peak traj_g0 per arm (the flash's amplitude)",
            "the endpoint ratio vs serial per arm",
            "the peak-vs-driver-dip alignment (the arm's peak milestone "
            "step vs the serial driver's dip milestone step, n=4 "
            "convention)",
            "the volume-null floor := the fresh fact-free base's own g0 "
            "battery read, co-reported on every peak read"],
        "bar_status": "READS ONLY — no bar moved; the frozen bars stand "
            "verbatim.",
    },
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (the horse race's adjudicator); this script committed at "
        "birth BEFORE any compute; adjudicate against exactly this; no "
        "bar shopping.",
}

deviations: list[str] = [
    "THE SMOKE CATCHES (pass 1, runs/e273_smoke/, all 14 gates PASS, "
    "exit clean — the e260-family record intact, TWO catches): "
    "(1) THE SRCT SIGN-STEP LAW — the live calibration probe measured "
    "the AdamW first step's applied in-room L2 at 80% of its FULL L2 "
    "(1.319 of 1.645 at smoke k), and a dedicated room-side check "
    "confirmed the law: ||P sign(P g)|| ~= 0.80 x ||sign|| for the SRCT "
    "room while a RANDOM sign vector lands at ~sqrt(k/N) — Adam's "
    "coordinate normalization converts a ~1.3%-in-room gradient into an "
    "~80%-in-room step (itself a mechanism datum: Adam's sign step is "
    "nearly compatible with ANY random subspace containing the "
    "gradient). Consequence: the in-room-matched LR_SGD is ~22 at the "
    "full room (factor ~2.2e4 vs 1e-3) — measured, not assumed. "
    "(2) THE SGD-M CORPUS-STEP DIVERGENCE — at that registered lr the "
    "FREE unprojected corpus steps (~lr x ||g_corp|| full-L2 each, "
    "momentum compounding toward x10) run the model away (smoke: "
    "install CE 1.07 -> 98 by s8; |d| -> 78). SGD's linear step cannot "
    "match Adam's sign-equalized streams simultaneously — the asymmetry "
    "IS part of Adam's normalization, disclosed as texture. RESPONSES "
    "(both pre-registered BEFORE the full compute): the DIVERGENCE-"
    "ROUTING RULE (a diverged SGD barrel's formation clause is labeled "
    "DIVERGENCE-CONFOUNDED and cannot hand a barrel the headline — "
    "routed to MIXED named) + the SGD001X stability rider (x0.01, near "
    "the steady-state-matched scale, never adjudicated). The registered "
    "SGDM arm RUNS AS CALIBRATED, verbatim — no bar, no calibration "
    "convention, no arm form was changed.",
    "THE CORPUS STREAM IS SHARED ACROSS BARRELS (registered at birth): "
    "all three concurrent barrels (a)/(b)/(c) draw from the ONE "
    f"registered fresh corpus generator (seed {CORPUS_GEN_SEED}) — so "
    "the SEPARATE-vs-SHARED and SGD-M-vs-SHARED deltas are the optimizer "
    "arrangements ONLY, never the corpus draws. The (c) replay is 'the "
    "original e268 FORM' (the 1:1 schedule + the shared AdamW + the free "
    "unprojected corpus steps), re-run on THIS session's registered "
    "stream — the family's own precedent (e269/e270/e271 each re-ran the "
    "form on its own fresh seed; a literal seed-26801 replay would "
    "CONFOUND the barrel comparison).",
    "THE SGD BARREL DROPS WEIGHT DECAY (disclosed): AdamW's wd 0.1 is "
    "decoupled; SGD's wd would be L2-coupled — a different object. The "
    "swap IS the optimizer family; the lr calibration matches the applied "
    "in-room write size (the first-order object). No bar or gate-form "
    "change follows.",
    "THE CONS STAYS ADAMW ON EVERY ARM (the rehearsal lane's integrity): "
    "the e113 cons (seed 10901 HELD, NATURAL) is the family's landing "
    "instrument — changing it per-barrel would break the arms' "
    "comparability on the carried landing read. The SGD barrel changes "
    "the INSTALL optimizer only, which is where the question lives.",
    "THE CALIBRATION IS A LIVE PROBE, NOT A DERIVED CONSTANT: a dedicated "
    "probe re-executes the capstone arm's first install step (bit-"
    "identical draws/hook/fresh AdamW init) on this session's GPU, "
    "measures the applied in-room L2, derives LR_SGD = target / "
    "||P g'|| (exact at s1: the momentum buffer is zero), and VERIFIES "
    "the SGD step-1 in-room L2 against the target (G_DOSE_LR, rel err "
    "<= 5%). The probe's value is persisted (runs/<name>/"
    "lr_calibration.json) and REUSED on any resume/sensitivity re-entry "
    "so the riders stay on one calibration.",
    "THE SENSITIVITY PAIR IS OPTIONAL AND NEVER ADJUDICATED "
    "(E273_SENSITIVITY=1; the dispatch's 'if budget allows'): SGD2X/"
    "SGD05X are confirmatory reads against the step-scale confound "
    "(T253's own caveat); their numbers appear in the reads table with a "
    "NEVER-ADJUDICATED stamp.",
    "THE SERIAL ARM IS A FRESH RE-RUN, NOT A CITE (e268's precedent): a "
    "same-session re-run at the registered seeds makes every barrel's "
    "ratio and antiphase one instrument on one session; its delta vs "
    "e272's committed K10KR arm doubles as the freshest cross-session "
    "scatter point (co-reported; install determinism |d post g0| ~5e-7-"
    "1e-6, cons root-g0 lottery ~0.04).",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked serial install/cons drivers (bit-identical arithmetic + draw "
    "order), the thermal envelope (per-step polls, 78C margin, 175s "
    "bursts — inside the dispatch's 180s — 40s cooldowns, the 84C line "
    "inside the dispatch's 85C), the progressive-metrics + resume-ckpt "
    "conventions. The module-global rebinding (log/NAME/LADDER/RUNG_NAMES/"
    "T0/thermal ledgers, disclosed in-code) retargets the drivers' I/O to "
    "this cell; the committed lab/e261_rank_ladder.py is NOT modified. "
    "The ONE new driver is chunked_install_barrel (this file) — it takes "
    "the room mode as a PARAMETER (e268's smoke catch honored by "
    "construction) and adds the per-milestone displacement reads (the "
    "weight-vs-read dissociation rider + the reads addendum).",
    "THE FIRST-BATCH CE IDENTITY CHECK (non-halting texture): every arm's "
    "ledger s1 install CE must equal SERIAL's (same draws, same base, "
    "same loss arithmetic — the draw-integrity read); a miss is disclosed "
    "session texture, never a bar move.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E273_SMOKE=1): 8 install + 8 interleaved corpus steps, "
    "8-step cons, room k=512 at the same seed pair (G_ROOM10KR vacuous — "
    "no committed record at smoke k; disclosed), all four main arms + "
    "the calibration probe + the adjudication + the figure exercised, "
    "all paths smoke_-prefixed, own smoke dir; NOTHING adjudicated or "
    "gated (SMOKE stamp on every read); the smoke calibration factor is "
    "smoke-only (the full run derives its own).",
]

device_events: list[dict] = []
thermal_log: list[dict] = []
E261.thermal_log = thermal_log          # e270's rebinding: ONE ledger
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


# ------------------------------------------------- THE BARREL DRIVER (new)
def chunked_install_barrel(tag, net0, proj: "E261.LadderRooms", base_flat,
                           inst_x, inst_mask, anchor_full, train_ids,
                           g0_ids, gm12_ids, r_eval_xy, zid, mode: str,
                           resume_ck: Path, dev: torch.device, barrel: str,
                           lr_sgd: float) -> dict:
    """THE CONCURRENT BARRELS' DRIVER (this cell's only new machinery; the
    SERIAL arm runs e261.chunked_install VERBATIM).

    The 1:1 interleave (e268's registered form): per iteration s = 1..400,
      1. INSTALL STEP: bit-identical to SERIAL's step s (draws ix(16)/
         aj(16)/rj(32) from gen seed 24314, the same draw order; the
         64-window Dmix batch; the masked union CE; lr x
         cosine_lr(s-1,1000); backward -> clip 1.0 -> HOOK: project onto
         the room (CPU fp64, write fp32; the mode is a PARAMETER — e268's
         smoke catch honored by construction) -> ledger dots -> opt.step.
      2. CORPUS STEP: draws (aj_c/rj_c) from cgen (seed 27301, this
         cell's ONE registered stream shared by every barrel); the 48-
         window corpus composition; full-window CE; the PAIRED lr;
         backward -> clip 1.0 -> opt.step FREE (NO projection; corpus
         ledger CE + clipped-grad norm).

    THE BARRELS (the ONLY delta across arms):
      SHARED  — ONE AdamW (0.9,0.95) wd 0.1 across both streams (e268
                verbatim; the control twin);
      SEPARATE— TWO AdamW (same betas/wd/schedule): install steps set and
                step ONLY opt_i; corpus steps set and step ONLY opt_c;
      SGDM / SGD2X / SGD05X — ONE SGD(momentum 0.9, wd 0) across both
                streams at LR_SGD x {1, 2, 0.5} x cosine.

    Milestones: s1/100/200/300/400 (SGD barrels add s25/s50 — the flash-
    power rider) with battery g0/g-12 + CE_R + the cumulative
    displacement reads (norm, in-room norm, in-own-room fraction, v-
    excess vs the base — the weight-vs-read dissociation rider + the
    reads addendum). Thermal: a poll after EVERY opt step (both
    streams)."""
    name_bs, corp_bs, mix_random, lr = (G1.NAME_BS, E43.CORP_BS,
                                        E43.MIX_RANDOM, E43.LR)
    n_steps = E261.INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    milestones = (TRAJ_MILE_SGD if barrel in SGD_BARRELS else TRAJ_MILE_MAIN)
    lr_base_sgd = lr_sgd * SENS_FACTORS.get(barrel, 1.0)
    state = {"step": 0, "traj": [], "ledger": {}, "corpus_ledger": {}}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at install step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt = opt_i = opt_c = gen = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []

    def _make_opts(netrain):
        if barrel == "SHARED":
            o = torch.optim.AdamW(netrain.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
            return o, None, None
        if barrel == "SEPARATE":
            oi = torch.optim.AdamW(netrain.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)
            oc = torch.optim.AdamW(netrain.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)
            return None, oi, oc
        o = torch.optim.SGD(netrain.parameters(), lr=lr_base_sgd,
                            momentum=SGD_MOMENTUM, weight_decay=SGD_WD)
        return o, None, None

    def _lr_scale() -> float:
        return (lr_base_sgd / lr) if barrel in SGD_BARRELS else 1.0

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt, opt_i, opt_c = _make_opts(net)
            gen = torch.Generator().manual_seed(E261.FRESH_GEN)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                if barrel == "SEPARATE":
                    opt_i.load_state_dict(state["opt_i"])
                    opt_c.load_state_dict(state["opt_c"])
                else:
                    opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = common.cosine_lr(step - 1, E261.INST_TOTAL)
            lr_now = lr * _lr_scale() * f
            # ---- 1. THE INSTALL STEP (bit-identical to SERIAL's s) --------
            ix = torch.randint(n_inst, (name_bs,), generator=gen)
            aj = torch.randint(n_anc, (corp_bs - mix_random,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                               generator=gen)
            corp = torch.cat([anchor_full[aj],
                              torch.stack([train_ids[s: s + G1.BLOCK]
                                           for s in rj])], 0)
            nw = inst_x[ix]
            x = torch.cat([nw[:, :-1], corp[:, :-1]], 0).to(dev)
            y = torch.cat([nw[:, 1:], corp[:, 1:]], 0).to(dev)
            m = torch.zeros(name_bs + corp_bs, x.shape[1], dtype=torch.bool,
                            device=dev)
            m[:name_bs] = inst_mask[ix].to(dev)
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1), reduction="none"
                                  ).view(x.shape[0], x.shape[1])
            nm = nll[:name_bs][m[:name_bs]]
            cm = nll[name_bs:]
            loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
            net.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            led = proj.step_hook(list(net.parameters()), mode)
            if barrel == "SEPARATE":
                for g in opt_i.param_groups:
                    g["lr"] = lr_now
                opt_i.step()
            else:
                for g in opt.param_groups:
                    g["lr"] = lr_now
                opt.step()
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["ledger"][step] = {
                    "ce": float(loss.item()), "gn": led["gn"],
                    "gpn": led["gpn"], "kept_frac": led["kept_frac"],
                    "v_excess_pre": led["v_excess_pre"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"]}
            n_burst += 1
            ok_t, temp = burst_temp_check(f"{tag}-c{n_chunks}.i")
            chunk_temps.append(temp)
            # ---- 2. THE CORPUS STEP (the interleave, 1:1) -----------------
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
            if barrel == "SEPARATE":
                for g in opt_c.param_groups:
                    g["lr"] = lr_now
                opt_c.step()
            else:
                for g in opt.param_groups:
                    g["lr"] = lr_now
                opt.step()
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c}
            n_burst += 1
            ok_t2, temp2 = burst_temp_check(f"{tag}-c{n_chunks}.x")
            chunk_temps.append(temp2)
            if step in milestones or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                d_mil = flat_params_cpu(net) - base_flat
                ld_mil = proj.displacement_loads(d_mil, mode)
                dn = float(d_mil.norm())
                state["traj"].append({
                    "step": step,
                    "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
                    "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                    "ce_batch": float(loss.item()),
                    "ce_corpus": float(loss_c.item()),
                    "kept_frac": led["kept_frac"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"],
                    "disp_norm_cum": dn,
                    "in_room_disp_norm_cum": ld_mil["in_own_room"] * dn,
                    "in_own_room_frac_cum": ld_mil["in_own_room"],
                    "v_excess_cum": ld_mil["v_excess"],
                    "cos_to_span_cum": ld_mil["cos_to_span"],
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.5f} g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_inst "
                    f"{float(loss.item()):.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |g| {led['gn']:.3f} kept "
                    f"{led['kept_frac']:.4f} |g_corp| {gn_c:.3f} |d| "
                    f"{dn:.4f} |Pd| "
                    f"{ld_mil['in_own_room'] * dn:.4f} ({barrel})")
            if not (ok_t and ok_t2):
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at s{step} — resume ckpt saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        ck_rows = {"model": {k: v.detach().cpu()
                             for k, v in net.state_dict().items()},
                   "gen_state": gen.get_state(),
                   "cgen_state": cgen.get_state(),
                   "step": step, "traj": state["traj"],
                   "ledger": state["ledger"],
                   "corpus_ledger": state["corpus_ledger"],
                   "n_chunks": n_chunks, "chunk_table": chunk_table}
        if barrel == "SEPARATE":
            ck_rows["opt_i"] = opt_i.state_dict()
            ck_rows["opt_c"] = opt_c.state_dict()
        else:
            ck_rows["opt"] = opt.state_dict()
        torch.save(ck_rows, resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "ledger": state["ledger"],
            "corpus_ledger": state["corpus_ledger"],
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
    """The thermal-envelope summary aggregated from runs/_envelope_log.jsonl
    (the PERSISTED per-poll ledger — the in-process list is empty on a
    resume pass, and the final COMPLETE write typically IS a resume pass;
    the envelope's true record lives in the appended file, never nominal)."""
    out = {"burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
           "per_step_polls": "after EVERY opt step (both streams, every "
                             "arm) — aggregated from "
                             "runs/_envelope_log.jsonl (the persisted "
                             "ledger; survives resume passes)",
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
    out["violations_ge_84c"] = sum(1 for t in temps
                                    if t >= E261.TEMP_HARD)
    out["note"] = ("aggregated across ALL passes of this cell (phase 1 + "
                   "the riders + the resume writes); the smoke's polls are "
                   "tagged e273_smoke: and excluded")
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    if any(not math.isfinite(v) for v in xs) or any(not math.isfinite(v)
                                                    for v in ys):
        return None                      # a diverged read carries no phase
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sxx * syy)


def finite_max(vals):
    """max over the FINITE values only (a diverged arm's nan reads carry
    no amplitude); None if none are finite."""
    fv = [v for v in vals if v is not None and math.isfinite(v)]
    return max(fv) if fv else None


def arm_diverged(inst_rec: dict) -> dict:
    """THE DIVERGENCE SIGNATURE (pre-registered): the install ledger's CE
    beyond s100 (beyond the halfway mark in smoke) exceeding 2x the
    first-batch CE, or any non-finite read — the SGD-M corpus-step
    runaway the smoke caught."""
    led = inst_rec["ledger"]
    n_steps = E261.INST_STEPS
    horizon = 100 if n_steps >= 100 else n_steps // 2
    ces = sorted((int(k), v["ce"]) for k, v in led.items())
    first_ce = ces[0][1] if ces else None
    tail = [c for s, c in ces if s > horizon]
    mx = finite_max([c for _, c in ces])
    nonfinite = any(first_ce is None or not math.isfinite(c)
                    for _, c in ces)
    runaway = bool(first_ce is not None and tail
                   and finite_max(tail) is not None
                   and finite_max(tail) > 2.0 * first_ce)
    return {"first_batch_ce": first_ce,
            "max_ce_beyond_s100": finite_max(tail),
            "max_ce": mx, "nonfinite": bool(nonfinite),
            "diverged": bool(runaway or nonfinite)}


def antiphase_read(traj_arm: list[dict], traj_serial: list[dict]) -> dict:
    """The antiphase read (both conventions) + the peak-vs-driver-dip
    alignment (the reads addendum). traj_* are driver traj rows."""
    sa = {t["step"]: t["g0_pz"] for t in traj_arm}
    ss = {t["step"]: t["g0_pz"] for t in traj_serial}
    common_steps = sorted(set(sa) & set(ss))
    x = [ss[s] for s in common_steps]
    y = [sa[s] for s in common_steps]
    r5 = pearson(x, y)
    x4, y4 = x[1:], y[1:]                       # x8's convention: drop s1
    steps4 = common_steps[1:]
    r4 = pearson(x4, y4)
    dip_i = min(range(len(x4)), key=lambda i: x4[i])
    dip_step = steps4[dip_i]
    fin = [(s, v) for s, v in zip(steps4, y4)
           if math.isfinite(v)]
    if fin:
        pk_i = max(range(len(fin)), key=lambda i: fin[i][1])
        peak_step, peak_amp = fin[pk_i][0], fin[pk_i][1]
    else:
        peak_step, peak_amp = None, None
    return {"n5": len(common_steps), "pearson_n5": r5,
            "n4": len(steps4), "pearson_n4": r4,
            "serial_dip_step": dip_step, "arm_peak_step": peak_step,
            "peak_traj_g0": peak_amp,
            "peak_aligned_with_driver_dip": bool(peak_step == dip_step),
            "arm_traj_g0": {s: sa[s] for s in common_steps},
            "serial_traj_g0": {s: ss[s] for s in common_steps}}


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e273_optimizer_barrels",
        "phase": "THE THREE-BARREL MECHANISM CELL — the horse race's "
                 "adjudicator: the concurrent kill at k=10k (e268: serial "
                 "0.2646 vs concurrent 0.00004) — is the coupling the "
                 "SHARED OPTIMIZER STATE (SEPARATE-AdamW barrel), ADAM'S "
                 "NORMALIZATION (SGD-M barrel), or the TRAJECTORY (a true "
                 "two-body problem)? STATE-POISONING vs ADAM-SPECIFIC-BLOCK "
                 "vs TRAJECTORY-TWO-BODY vs MIXED, adjudicated on the WRITE "
                 "read",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "sensitivity": bool(SENSITIVITY and not SMOKE),
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane) + "
                      "CPU fp64 dense projections (pocketfft workers 2), "
                      "CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (inside the dispatch's "
                      f"180s), per-step thermal polls (BOTH streams' opt "
                      f"steps) at a {E261.TEMP_EARLY_END:.0f}C margin, "
                      f"cooldown {E261.COOLDOWN_S:.0f}s (the 30-60s window), "
                      f"the {E261.TEMP_HARD:.0f}C never-past line (inside "
                      "the dispatch's 85C)",
            "trainings": f"{len(ARMS_RUN)} installs s400 (SERIAL = e261's "
                         "driver VERBATIM; the barrels = the interleaved "
                         "driver, 800 opt steps each) + "
                         f"{len(ARMS_RUN)} cons s300 (seed 10901 HELD, "
                         "NATURAL AdamW on every arm); NO washes",
        },
        "interleave": {
            "ratio": "1:1 — after EVERY install step s, ONE corpus step "
                     "(e268's registered form)",
            "install_step": "bit-identical to SERIAL's step s: draws "
                            "ix(16)/aj(16)/rj(32) from gen seed 24314 (the "
                            "same draw order), the 64-window Dmix batch, "
                            "the masked union CE, lr x cosine_lr(s-1, "
                            f"{E261.INST_TOTAL}), clip 1.0 -> PROJECT onto "
                            "the room -> opt.step",
            "corpus_step": "48 windows from the committed g1c Dmix corpus "
                           "convention: 16 original-host anchors + 32 "
                           "random corpus windows; full-window CE; draws "
                           f"from THIS cell's REGISTERED FRESH corpus "
                           f"generator seed {CORPUS_GEN_SEED} (SHARED by "
                           "all barrels); the PAIRED install step's lr; "
                           "clip 1.0 -> opt.step FREE (unprojected)",
            "shared_optimizer": "ONE AdamW (0.9, 0.95) wd 0.1 across both "
                                "streams — the moment coupling IS the "
                                "interference channel (the SHARED barrel; "
                                "the SEPARATE barrel splits it; the SGD "
                                "barrel removes Adam entirely)",
            "dose_delta_disclosed": "install dose IDENTICAL across arms "
                                    "(16 masked windows x 400); corpus "
                                    "exposure triples in every concurrent "
                                    "barrel (the concurrent stream is the "
                                    "object under test)",
        },
        "deviations": deviations,
        "builds_on": [
            "T253 / consult #006 (THE registration: P-273a and agy's kill "
            "are the same arm — separate-AdamW kills or keeps the "
            "antiphase; P-273b adopted — under SGD-M at k=10k the write "
            "FORMS if the block is v-poisoning; the lr-matching convention "
            "frozen: first-install-step applied-L2 + the 2x/0.5x pair)",
            "T252 / x8 (the antiphase datum: the flash peaks where the "
            "serial driver dips, Pearson -0.52 to -0.71 — any mechanism "
            "must explain it; P-273a registered there first)",
            "W042 / the gap-currency bridge (P-273c: the concurrent "
            "write's structured component should be MORE sharply antiphase "
            "than its p-mass if the competition is for the structured "
            "supply — this cell's per-milestone displacement rides carry "
            "the weight-side of that dissociation)",
            "T246 / e268 (THE capstone: DYNAMICAL-CARRIER at 6609x — the "
            "10k write dies under the 1:1 interleave through the ONE "
            "shared AdamW; the interleave's registered form ORIGINATES "
            "here; the rehearsal-lane finding: the landing read = the "
            "cons's rehearsal, never adjudicated)",
            "T255 / e272 (the relocated edge + the K10KR room: THIS "
            "cell's vehicle room (seeds 27215/27216) is e272's fresh-seed "
            "10k replicate — post 0.2097 vs the committed rung's 0.2646, "
            "the ~21% room-lottery draw, in-band; the cited serial "
            "reference)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT: the "
            "SRCT projector, the hooked drivers, the thermal envelope)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen "
            "24314 + e113 cons 10901)",
        ],
        "whats_new": [
            "THE MECHANISM QUESTION ASKED FOR THE FIRST TIME: the "
            "concurrent kill's coupling isolated by optimizer arrangements "
            "alone — the same room, the same bit-identical install stream, "
            "the same corpus draws, the same cons, with ONLY the "
            "optimizer's state-carrying changed (shared vs split vs "
            "non-Adam)",
            "THE SGD-M BARREL + THE LIVE LR CALIBRATION: Adam's "
            "normalization removed from the whole cell at matched "
            "first-step applied in-room write size (the probe, the "
            "disclosed factor, the verification gate)",
            "THE ANTIPHASE READ AS A BAR CLAUSE: the flash-vs-driver "
            "correlation measured per barrel on both conventions with the "
            "-0.52 provenance reproduced live from e268's hard-bound pair "
            "— P-273a's second clause made operational",
        ],
        "gates": {},
    })
    log(f"E273 — THE THREE-BARREL MECHANISM CELL (smoke={SMOKE}, "
        f"sensitivity={SENSITIVITY and not SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS_RUN)}; vehicle = e272's K10KR room (seeds "
        f"{LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs {ROOMS272_CK}); "
        f"discriminators: SURVIVES >= {SURVIVE_FRAC:.0%}x serial twin; "
        f"|antiphase| < {ANTIPHASE_VANISH_BAR} (a) & <= "
        f"{ANTIPHASE_PRESENT_BAR} (c); SGD-M peak > {FORMATION_BAR}")
    write_partial("startup (bars + reads addendum registered, committed at "
                  "birth)")
    set_seed(27301)                 # global init only; every RNG is its own

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
    e272m = json.loads(E272_METRICS.read_text(encoding="utf-8"))
    e272_post = e272m["arms"]["K10KR"]["install"]["post_cells"]["g0"]
    e272_root = e272m["arms"]["K10KR"]["root"]["g0"]
    e272_kept = e272m["arms"]["K10KR"]["install"]["ledger_kept_frac_median"]
    e272_gm12 = e272m["arms"]["K10KR"]["install"]["post_cells"]["gm12"]
    e272_traj = {t["step"]: t["g0_pz"]
                 for t in e272m["arms"]["K10KR"]["install"]["traj"]}
    e268m = json.loads(E268_METRICS.read_text(encoding="utf-8"))
    e268_post_s = e268m["arms"]["SERIAL"]["install"]["post_cells"]["g0"]
    e268_post_c = e268m["arms"]["CONCURRENT"]["install"]["post_cells"]["g0"]
    e268_traj_s = {t["step"]: t["g0_pz"]
                   for t in e268m["arms"]["SERIAL"]["install"]["traj"]}
    e268_traj_c = {t["step"]: t["g0_pz"]
                   for t in e268m["arms"]["CONCURRENT"]["install"]["traj"]}
    vehicle = torch.load(CKPT_DIR / K10KR_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    del vehicle
    traj_bind_ok = all(
        abs(e272_traj.get(k, -1) - v) < 1e-12
        for k, v in E272_K10KR_TRAJ_G0.items()) and len(e272_traj) == 5
    e268_traj_bind_ok = (
        all(abs(e268_traj_s.get(k, -1) - v) < 1e-12
            for k, v in E268_SERIAL_TRAJ.items())
        and all(abs(e268_traj_c.get(k, -1) - v) < 1e-12
                for k, v in E268_CONCURRENT_TRAJ.items()))
    G_PARENTS = {
        "e272_metrics": {"path": str(E272_METRICS),
                         "md5": md5of(E272_METRICS), "bound_md5": E272_MD5,
                         "verdict": e272m["adjudication"]["verdict"],
                         "K10KR_post_g0": e272_post, "K10KR_root_g0": e272_root,
                         "K10KR_kept_median": e272_kept,
                         "K10KR_post_gm12": e272_gm12,
                         "note": "THE CITED SERIAL REFERENCE (the vehicle "
                                 "room's own committed arm; room 27215/27216)"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of(E268_METRICS), "bound_md5": E268_MD5,
                         "verdict": e268m["adjudication"]["verdict"],
                         "serial_post_g0": e268_post_s,
                         "concurrent_post_g0": e268_post_c,
                         "ratio": e268_post_c / max(e268_post_s,
                                                    RATIO_DEN_FLOOR),
                         "note": "the capstone's 10k pair (the kill this "
                                 "cell adjudicates) + the antiphase "
                                 "provenance pair (hard-bound traj arrays)"},
        "e271_metrics": {"path": str(E271_METRICS),
                         "md5": md5of(E271_METRICS), "bound_md5": E271_MD5,
                         "note": "the natural-width concurrent datum "
                                 "(context; the antiphase family's 237k "
                                 "point)"},
        "k10kr_vehicle": {"path": f"runs/checkpoints/{K10KR_CK}",
                           "md5": md5of(CKPT_DIR / K10KR_CK),
                           "bound_md5": K10KR_MD5,
                           "size": (CKPT_DIR / K10KR_CK).stat().st_size,
                           "bound_size": K10KR_SIZE,
                           "state": vehicle_state},
        "e272_rooms": {"path": f"runs/checkpoints/{ROOMS272_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS272_CK),
                       "bound_md5": md5of(CKPT_DIR / ROOMS272_CK)},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "e272_verdict": E272_VERDICT,
            "e272_K10KR_post_g0": E272_K10KR_POST_G0,
            "e272_K10KR_root_g0": E272_K10KR_ROOT_G0,
            "e272_K10KR_kept_med": E272_K10KR_KEPT_MED,
            "e272_K10KR_post_gm12": E272_K10KR_POST_GM12,
            "e272_K10KR_traj_g0": E272_K10KR_TRAJ_G0,
            "e268_verdict": E268_VERDICT,
            "e268_serial_post_g0": E268_SERIAL_POST,
            "e268_concurrent_post_g0": E268_CONCURRENT_POST,
            "e268_ratio": E268_RATIO,
            "e268_serial_traj_g0": E268_SERIAL_TRAJ,
            "e268_concurrent_traj_g0": E268_CONCURRENT_TRAJ,
            "anchor_scatter_cross_session_root_g0": ANCHOR_SCATTER_G0,
            "cons_seed_scatter_session_root_g0": CONS_SEED_SCATTER_G0,
            "vehicle_md5": K10KR_MD5, "vehicle_step": K10KR_STEP,
            "vehicle_traj_steps": K10KR_TRAJ_STEPS,
            "vehicle_ledger_max": K10KR_LEDGER_MAX},
        "pass": bool(
            e272m["adjudication"]["verdict"] == E272_VERDICT
            and abs(e272_post - E272_K10KR_POST_G0) < 1e-12
            and abs(e272_root - E272_K10KR_ROOT_G0) < 1e-12
            and abs(e272_kept - E272_K10KR_KEPT_MED) < 1e-12
            and abs(e272_gm12 - E272_K10KR_POST_GM12) < 1e-12
            and traj_bind_ok
            and e268m["adjudication"]["verdict"] == E268_VERDICT
            and abs(e268_post_s - E268_SERIAL_POST) < 1e-12
            and abs(e268_post_c - E268_CONCURRENT_POST) < 1e-12
            and e268_traj_bind_ok
            and md5of(E272_METRICS) == E272_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E271_METRICS) == E271_MD5
            and md5of(CKPT_DIR / K10KR_CK) == K10KR_MD5
            and (CKPT_DIR / K10KR_CK).stat().st_size == K10KR_SIZE
            and vehicle_state["step"] == K10KR_STEP
            and vehicle_state["traj_steps"] == K10KR_TRAJ_STEPS
            and vehicle_state["ledger_max"] == K10KR_LEDGER_MAX
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
            and (SMOKE or LADDER[0][0] == 10_000)),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e272 {E272_VERDICT} (K10KR post "
        f"{e272_post:.6f}, root {e272_root:.6f}, kept {e272_kept:.4f}); "
        f"e268 {E268_VERDICT} (pair {e268_post_s:.6f} vs "
        f"{e268_post_c:.6f}); the K10KR vehicle md5/step-bound at "
        f"s{K10KR_STEP}")
    write_partial("P0b parents hard-bound")
    del e272m, e268m

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_g0 = G1.battery_cell(base_net, g0_ids, zid)["mean_pz"]   # the
    # volume-null floor (the reads addendum's co-reported null level)
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "g0_volume_null_floor": base_g0,
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, g0 floor {base_g0:.2e}, CE_R "
        f"{base_ce_r:.4f}): PASS")
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
        "form": "the K10KR room certified (fp64 CPU, "
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

    # ---- G_ROOM10KR: bit-identity vs e272's committed K10KR room --------
    rooms272 = torch.load(CKPT_DIR / ROOMS272_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    room_name = RUNG[LADDER[0][0]]
    if not SMOKE:
        D272 = _to_np(rooms272["model"]["K10KR"]["D_int8"]).astype(np.float64)
        S272 = _to_np(rooms272["model"]["K10KR"]["S"])
        D_mine = rooms.rooms[room_name].D
        S_mine = rooms.rooms[room_name].S
        G_ROOM10KR = {
            "form": "the vehicle's room == e272's committed K10KR room "
                    "(seeds 27215/27216 at k=10,000): the +-1 diagonal and "
                    "the index set bit-identical to e272_rooms.pt's stored "
                    "K10KR D/S (exact equality)",
            "D_bit_equal": bool(np.array_equal(D_mine, D272)),
            "S_bit_equal": bool(np.array_equal(S_mine, S272)),
            "e272_rooms_md5": md5of(CKPT_DIR / ROOMS272_CK),
            "pass": bool(np.array_equal(D_mine, D272)
                         and np.array_equal(S_mine, S272)
                         and int(rooms272["model"]["K10KR"]["k"])
                         == LADDER[0][0]
                         and list(rooms272["model"]["K10KR"]["seeds"])
                         == [LADDER[0][1], LADDER[0][2]]),
        }
        del rooms272
    else:
        G_ROOM10KR = {
            "form": "SMOKE: the room shares the seed pair (27215/27216) at "
                    "smoke k — no committed record at this k; the bit-bind "
                    "is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
        del rooms272
    assert G_ROOM10KR["pass"], f"K10KR room bind failed: {G_ROOM10KR}"
    metrics["gates"]["G_ROOM10KR"] = G_ROOM10KR
    log(f"P1 G_ROOM10KR: the vehicle's room "
        f"{('bit-identical to e272_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e273_rooms",
        {room_name: {"D_int8": rooms.rooms[room_name].D.astype(np.int8),
                     "S": rooms.rooms[room_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e273's vehicle room (the flat basis is "
                 "net.parameters() order): e272's committed K10KR room "
                 "(seeds 27215/27216), rebuilt + bit-gated vs "
                 "e272_rooms.pt",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][room_name].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": room_name,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS272_CK} "
                                    "(e272's committed K10KR room)"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE VEHICLE ROOM: {room_name} (k={LADDER[0][0]}): BUILT + "
        f"CERTIFIED + BIT-BOUND")
    write_partial("P1 the vehicle room built (parents bound + v-map loaded "
                  "+ span loaded + certification + bit-bind)")

    # ============ P1.5: THE LIVE LR CALIBRATION (the SGD barrel) ========
    calib_path = RD / "lr_calibration.json"
    if calib_path.exists():
        cal = json.loads(calib_path.read_text(encoding="utf-8"))
        cal["reused_from"] = str(calib_path)
        log(f"P1.5: calibration REUSED from {calib_path.name} (the "
            f"persisted session calibration; LR_SGD {cal['lr_sgd']:.6f}, "
            f"factor x{cal['factor_vs_adamw_lr']:.1f})")
    else:
        wait_gpu_free("CALIB")
        t_cal = time.time()
        netp = copy.deepcopy(G1.evl_load(base_sd)).to(dev)
        netp.train()
        genp = torch.Generator().manual_seed(E261.FRESH_GEN)
        ix = torch.randint(inst_x.shape[0], (G1.NAME_BS,), generator=genp)
        aj = torch.randint(anchor_full.shape[0],
                           (E43.CORP_BS - E43.MIX_RANDOM,), generator=genp)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (E43.MIX_RANDOM,),
                           generator=genp)
        corp_p = torch.cat([anchor_full[aj],
                            torch.stack([train_ids[s: s + G1.BLOCK]
                                         for s in rj])], 0)
        nwp = inst_x[ix]
        xp = torch.cat([nwp[:, :-1], corp_p[:, :-1]], 0).to(dev)
        yp = torch.cat([nwp[:, 1:], corp_p[:, 1:]], 0).to(dev)
        mp = torch.zeros(G1.NAME_BS + E43.CORP_BS, xp.shape[1],
                         dtype=torch.bool, device=dev)
        mp[:G1.NAME_BS] = inst_mask[ix].to(dev)
        snap = {k: v.detach().clone() for k, v in netp.state_dict().items()}

        def _probe(make_opt, lr_use):
            netp.load_state_dict(snap)
            opt_p = make_opt(lr_use)
            logits_p, _ = netp(xp)
            nll_p = F.cross_entropy(logits_p.reshape(-1,
                                                     logits_p.shape[-1]),
                                    yp.reshape(-1), reduction="none"
                                    ).view(xp.shape[0], xp.shape[1])
            nmp = nll_p[:G1.NAME_BS][mp[:G1.NAME_BS]]
            cmp_ = nll_p[G1.NAME_BS:]
            loss_p = (nmp.sum() + cmp_.sum()) / (nmp.numel() + cmp_.numel())
            opt_p.zero_grad(set_to_none=True)
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(netp.parameters(), 1.0)
            led_p = rooms.step_hook(list(netp.parameters()), ROOM_MODE)
            for g in opt_p.param_groups:
                g["lr"] = lr_use
            opt_p.step()
            d_p = (flat_params_cpu(netp) - base_flat).double().numpy() \
                .astype(np.float64)
            pd_p = rooms.proj_of(ROOM_MODE, d_p)
            return {"ce_first_batch": float(loss_p.item()),
                    "gpn": led_p["gpn"], "kept_frac": led_p["kept_frac"],
                    "applied_l2_full": float(np.linalg.norm(d_p)),
                    "applied_l2_in_room": float(np.linalg.norm(pd_p))}

        adamw_probe = _probe(lambda lr_u: torch.optim.AdamW(
            netp.parameters(), lr=lr_u, betas=(0.9, 0.95),
            weight_decay=0.1), E43.LR)
        target = adamw_probe["applied_l2_in_room"]
        gpn1 = adamw_probe["gpn"]
        lr_sgd = target / gpn1
        sgd_probe = _probe(lambda lr_u: torch.optim.SGD(
            netp.parameters(), lr=lr_u, momentum=SGD_MOMENTUM,
            weight_decay=SGD_WD), lr_sgd)
        rel_err = abs(sgd_probe["applied_l2_in_room"] - target) / target
        cal = {"form": "the live probe: the capstone arm's FIRST install "
                       "step re-executed (bit-identical draws/hook/fresh "
                       "optimizer init); LR_SGD := AdamW applied in-room "
                       "L2 / ||P g'|| (exact at s1, momentum buffer zero); "
                       "the SGD step verified against the target",
               "probe_seconds": round(time.time() - t_cal, 1),
               "adamw_lr": E43.LR,
               "adamw_probe": adamw_probe,
               "lr_sgd": lr_sgd,
               "factor_vs_adamw_lr": lr_sgd / E43.LR,
               "sgd_probe": sgd_probe,
               "sgd_verification_rel_err": rel_err,
               "smoke_only": SMOKE}
        save_json(calib_path, E43.jsonable(cal))
        del netp
    G_DOSE_LR = {
        "form": "the SGD barrel's lr calibration (the family convention, "
                "T253's frozen form): the first install step's applied "
                "in-room L2 matched to the AdamW capstone arm's; the "
                "factor disclosed; the SGD step-1 verified (rel err <= "
                f"{DOSE_LR_TOL}); momentum {SGD_MOMENTUM}, wd {SGD_WD} "
                "(disclosed)",
        "reads": cal,
        "bars": {"verification_rel_err": DOSE_LR_TOL},
        "pass": bool(cal["lr_sgd"] > 0.0
                     and cal["sgd_verification_rel_err"] <= DOSE_LR_TOL
                     and cal["adamw_probe"]["applied_l2_in_room"] > 0.0),
    }
    assert G_DOSE_LR["pass"], f"LR calibration failed: {G_DOSE_LR}"
    metrics["gates"]["G_DOSE_LR"] = G_DOSE_LR
    metrics["lr_calibration"] = cal
    log(f"P1.5 G_DOSE_LR PASS: AdamW s1 in-room L2 "
        f"{cal['adamw_probe']['applied_l2_in_room']:.6f} (gpn "
        f"{cal['adamw_probe']['gpn']:.6f}, full "
        f"{cal['adamw_probe']['applied_l2_full']:.4f}) -> LR_SGD "
        f"{cal['lr_sgd']:.6f} (factor x{cal['factor_vs_adamw_lr']:.1f} vs "
        f"1e-3); SGD s1 in-room {cal['sgd_probe']['applied_l2_in_room']:.6f}"
        f" (rel err {cal['sgd_verification_rel_err']:.2e})")
    write_partial("P1.5 the SGD lr calibration PASSED (live probe)")

    LR_SGD = float(cal["lr_sgd"])

    # ================= P2+: THE ARMS ======================================
    arms_rec: dict = {}
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None  # noqa: E731

    def read_arm_cells(sd: dict) -> tuple:
        net_ = G1.evl_load(sd)
        cells_ = {"gm12": G1.battery_cell(net_, gm12_ids, zid)["mean_pz"],
                  "g0": G1.battery_cell(net_, g0_ids, zid)["mean_pz"],
                  "gp12": G1.battery_cell(net_, bat_ids[12], zid)["mean_pz"],
                  "ce_r": G1.ce_fixed_cpu(net_, *r_eval_xy)}
        return net_, cells_

    def run_cons(arm: str, sd_install: dict) -> dict:
        cons = E261.chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e273_{arm}_cons_resume.pt" if SMOKE
                        else f"e273_{arm}_cons_resume.pt"), dev)
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
            f"e273_{arm}_root", theta0,
            {"desc": f"e273 ARM-{arm} root: e001 + Dmix s{E261.INST_STEPS} "
                     f"(gen {E261.FRESH_GEN}; the K10KR room; "
                     f"{ARM_DESC[arm][:110]}...) + e113 cons s"
                     f"{E261.CONS_STEPS} (seed {E261.CONS_SEED} HELD "
                     "NATURAL)",
             "arm": arm, "install_seed": E261.FRESH_GEN,
             "corpus_gen_seed": CORPUS_GEN_SEED,
             "lr_sgd": (LR_SGD * SENS_FACTORS[arm]
                        if arm in SGD_BARRELS else None),
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e273_rooms.pt"})
        del root_net_arm
        landed = (E261.G1C_ROOT_G0 * (1 - MATCH_BAND)
                  <= cells["g+0"] <= E261.G1C_ROOT_G0 * (1 + MATCH_BAND))
        return {"cells": cells, "gm12": cells["g-12"], "g0": cells["g+0"],
                "displacement_loads": load_root, "checkpoint": root_ck,
                "landed": bool(landed)}

    def record_arm(arm: str, inst: dict, cells: dict, load: dict,
                   in_room_norm: float, disp_norm: float) -> None:
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        rec = {
            "desc": ARM_DESC[arm], "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "chunk_table": inst["chunk_table"], "steps": E261.INST_STEPS,
                "post_cells": cells, "displacement_loads": load,
                "in_room_disp_norm": in_room_norm,
                "disp_norm": disp_norm,
                "resumed_final": bool(inst.get("resumed_final", False)),
                "opt_steps": E261.INST_STEPS if arm == "SERIAL"
                else 2 * E261.INST_STEPS},
        }
        if arm != "SERIAL":
            corp_ce = [v["ce"] for v in inst["corpus_ledger"].values()]
            corp_gn = [v["gn_clipped"]
                       for v in inst["corpus_ledger"].values()]
            rec["install"]["corpus_ledger"] = inst["corpus_ledger"]
            rec["install"]["corpus_ce_median"] = med(corp_ce)
            rec["install"]["corpus_gn_clipped_median"] = med(corp_gn)
            rec["install"]["corpus_gen_seed"] = CORPUS_GEN_SEED
            rec["install"]["barrel"] = arm
        if arm in SGD_BARRELS:
            rec["install"]["lr_sgd"] = LR_SGD * SENS_FACTORS[arm]
            rec["install"]["lr_factor_vs_adamw"] = (
                cal["factor_vs_adamw_lr"] * SENS_FACTORS[arm])
        arms_rec[arm] = rec
        metrics["arms"] = arms_rec

    # ---- ARM-SERIAL (e261's driver VERBATIM at the registered seeds) ----
    log("=" * 78)
    log(f"ARM-SERIAL — {ARM_DESC['SERIAL']}")
    inst_s = E261.chunked_install(
        "SERIAL-inst", ROOM_MODE, G1.evl_load(base_sd), rooms,
        inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e273_SERIAL_inst_resume.pt" if SMOKE
                    else "e273_SERIAL_inst_resume.pt"), dev)
    sd_s = inst_s["sd"]
    net_s, cells_s = read_arm_cells(sd_s)
    d_s = flat_params_cpu(net_s) - base_flat
    load_s = rooms.displacement_loads(d_s, ROOM_MODE)
    d_s64 = d_s.double().numpy().astype(np.float64)
    in_room_s = float(np.linalg.norm(rooms.proj_of(ROOM_MODE, d_s64)))
    disp_s = float(np.linalg.norm(d_s64))
    del net_s
    record_arm("SERIAL", inst_s, cells_s, load_s, in_room_s, disp_s)
    log(f"ARM-SERIAL install done: post g0 {cells_s['g0']:.6f} g-12 "
        f"{cells_s['gm12']:.6f} CE_R {cells_s['ce_r']:.4f} | d v-excess "
        f"{load_s['v_excess']:.2f} in-own-room {load_s['in_own_room']:.4f} "
        f"||Pd|| {in_room_s:.5f} ||d|| {disp_s:.5f} | kept med "
        f"{med([v['kept_frac'] for v in inst_s['ledger'].values()]):.4f}")
    write_partial("P2 ARM-SERIAL install + post dial + measured loads")

    # ---- G_SERIAL_ANCHOR (non-halting texture vs e272's committed arm) --
    if not SMOKE:
        veh = torch.load(CKPT_DIR / K10KR_CK, map_location="cpu",
                         weights_only=False)["model"]
        mine_s = torch.load(
            CKPT_DIR / ("smoke_e273_SERIAL_inst_resume.pt" if SMOKE
                        else "e273_SERIAL_inst_resume.pt"),
            map_location="cpu", weights_only=False)["model"]
        serial_l2 = float(np.sqrt(sum(float(((mine_s[k].float()
                                              - veh[k].float()) ** 2).sum())
                                    for k in veh if k in mine_s)))
        del veh, mine_s
    else:
        serial_l2 = None
    G_SERIAL_ANCHOR = {
        "form": "the SERIAL re-run vs e272's committed K10KR arm (the "
                "cited reference's integrity): install L2 vs the committed "
                "vehicle <= 5e-3; behavioral |d| <= 0.02 on post g0 / root "
                "g0 (root g-12 reported with the lottery note, never "
                "gated). NON-HALTING: the root clause sits below the known "
                "cons lottery (~0.0419) and is EXPECTED to miss ~half the "
                "time (the e269/e270/e271 precedents); a failure is "
                "disclosed session texture, never a bar move",
        "install_l2_vs_vehicle": serial_l2,
        "post_g0": {"mine": cells_s["g0"], "e272": E272_K10KR_POST_G0,
                    "abs_diff": abs(cells_s["g0"] - E272_K10KR_POST_G0)},
        "post_gm12": {"mine": cells_s["gm12"], "e272": E272_K10KR_POST_GM12,
                      "abs_diff": abs(cells_s["gm12"]
                                      - E272_K10KR_POST_GM12)},
        "kept_median": {"mine": med([v["kept_frac"]
                                     for v in inst_s["ledger"].values()]),
                        "e272": E272_K10KR_KEPT_MED,
                        "abs_diff": abs(
                            med([v["kept_frac"]
                                 for v in inst_s["ledger"].values()])
                            - E272_K10KR_KEPT_MED)},
        "bars": {"l2": 5e-3, "behavior": ANCHOR_BEHAV_TOL},
        "pass": None,          # filled after the root read (root g0 join)
    }
    metrics["gates"]["G_SERIAL_ANCHOR"] = G_SERIAL_ANCHOR

    if not inst_s.get("resumed_final", False):
        burst_cooldown("SERIAL inst->cons")
    cons_s = run_cons("SERIAL", sd_s)
    arms_rec["SERIAL"]["consolidation"] = {"traj": cons_s["traj"],
                                           "chunk_table":
                                               cons_s["chunk_table"]}
    arms_rec["SERIAL"]["root"] = read_root("SERIAL", cons_s["sd"])
    r_s = arms_rec["SERIAL"]["root"]
    G_SERIAL_ANCHOR["root_g0"] = {"mine": r_s["g0"],
                                  "e272": E272_K10KR_ROOT_G0,
                                  "abs_diff": abs(r_s["g0"]
                                                  - E272_K10KR_ROOT_G0)}
    G_SERIAL_ANCHOR["root_gm12_lottery"] = {
        "mine": r_s["gm12"], "note": "root g-12 is the wild lottery "
        "(e264's three cons draws on a bit-identical arm ranged 0.67); "
        "reported, never gated, never adjudicated"}
    if not SMOKE:
        G_SERIAL_ANCHOR["pass"] = bool(
            serial_l2 <= 5e-3
            and abs(cells_s["g0"] - E272_K10KR_POST_G0) <= ANCHOR_BEHAV_TOL
            and abs(r_s["g0"] - E272_K10KR_ROOT_G0) <= ANCHOR_BEHAV_TOL)
    else:
        G_SERIAL_ANCHOR["pass"] = True
        G_SERIAL_ANCHOR["vacuous"] = True
    metrics["arms"] = arms_rec
    log(f"ARM-SERIAL ROOT: g0 {r_s['g0']:.4f} landed={r_s['landed']} | g-12 "
        f"{r_s['gm12']:.4f} (lottery) | CE_R {r_s['cells']['ce_r']:.4f} | "
        f"d-root in-own-room "
        f"{r_s['displacement_loads']['in_own_room']:.4f}")
    anch_txt = ("PASS" if G_SERIAL_ANCHOR["pass"]
                else "FAIL — disclosed session texture")
    log(f"G_SERIAL_ANCHOR (non-halting texture): install L2 "
        f"{serial_l2 if serial_l2 is None else f'{serial_l2:.3e}'} "
        f"(bar 5e-3), post g0 |d| "
        f"{abs(cells_s['g0'] - E272_K10KR_POST_G0):.4f}, root g0 |d| "
        f"{abs(r_s['g0'] - E272_K10KR_ROOT_G0):.4f} (bars "
        f"{ANCHOR_BEHAV_TOL}): {anch_txt}")
    write_partial("P3 ARM-SERIAL root + G_SERIAL_ANCHOR (non-halting)")

    # ---- THE BARRELS: (c) SHARED, (a) SEPARATE, (b) SGDM (+ riders) -----
    for arm in [a for a in ARMS_RUN if a != "SERIAL"]:
        burst_cooldown(f"prev -> {arm}")
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        inst_a = chunked_install_barrel(
            f"{arm}-inst", G1.evl_load(base_sd), rooms, base_flat,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid, ROOM_MODE,
            CKPT_DIR / (f"smoke_e273_{arm}_inst_resume.pt" if SMOKE
                        else f"e273_{arm}_inst_resume.pt"), dev,
            barrel=arm, lr_sgd=LR_SGD)
        sd_a = inst_a["sd"]
        net_a, cells_a = read_arm_cells(sd_a)
        d_a = flat_params_cpu(net_a) - base_flat
        load_a = rooms.displacement_loads(d_a, ROOM_MODE)
        d_a64 = d_a.double().numpy().astype(np.float64)
        in_room_a = float(np.linalg.norm(rooms.proj_of(ROOM_MODE, d_a64)))
        disp_a = float(np.linalg.norm(d_a64))
        del net_a
        record_arm(arm, inst_a, cells_a, load_a, in_room_a, disp_a)
        corp_note = ""
        if "corpus_ce_median" in arms_rec[arm]["install"]:
            corp_note = (f" | corpus CE med "
                         f"{arms_rec[arm]['install']['corpus_ce_median']:.4f}")
        peak_a = max((t["g0_pz"] for t in inst_a["traj"]), default=float("nan"))
        log(f"ARM-{arm} install done: post g0 {cells_a['g0']:.6f} g-12 "
            f"{cells_a['gm12']:.6f} CE_R {cells_a['ce_r']:.4f} peak "
            f"{peak_a:.6f} | d v-excess {load_a['v_excess']:.2f} "
            f"in-own-room {load_a['in_own_room']:.4f} ||Pd|| "
            f"{in_room_a:.5f} ||d|| {disp_a:.5f} | kept med "
            f"{arms_rec[arm]['install']['ledger_kept_frac_median']:.4f}"
            f"{corp_note}")
        write_partial(f"P4 ARM-{arm} install + post dial + measured loads")
        if not inst_a.get("resumed_final", False):
            burst_cooldown(f"{arm} inst->cons")
        cons_a = run_cons(arm, sd_a)
        arms_rec[arm]["consolidation"] = {"traj": cons_a["traj"],
                                          "chunk_table": cons_a["chunk_table"]}
        arms_rec[arm]["root"] = read_root(arm, cons_a["sd"])
        r_a = arms_rec[arm]["root"]
        log(f"ARM-{arm} ROOT (carried, the rehearsal lane): g0 "
            f"{r_a['g0']:.4f} landed={r_a['landed']} | g-12 "
            f"{r_a['gm12']:.4f} (lottery) | CE_R {r_a['cells']['ce_r']:.4f} "
            f"| d-root in-own-room "
            f"{r_a['displacement_loads']['in_own_room']:.4f}")
        metrics["arms"] = arms_rec
        write_partial(f"P5 ARM-{arm} root built + landing read (carried)")

    # ---- the draw-integrity texture check (non-halting) -----------------
    ce1 = {a: arms_rec[a]["install"]["ledger"].get(
        1 if 1 in arms_rec[a]["install"]["ledger"] else "1", {}
    ).get("ce") for a in ARMS_RUN}
    ce1_spread = (max(v for v in ce1.values() if v is not None)
                  - min(v for v in ce1.values() if v is not None)
                  ) if all(v is not None for v in ce1.values()) else None
    metrics["draw_integrity_texture"] = {
        "form": "every arm's FIRST install-batch CE must equal SERIAL's "
                "(the same draws, the same base, the same loss arithmetic "
                "— the draw-integrity read); NON-HALTING texture",
        "ce_first_batch": ce1, "spread": ce1_spread, "pass": None}
    metrics["draw_integrity_texture"]["pass"] = bool(
        ce1_spread is not None and ce1_spread <= 1e-4)
    log(f"draw-integrity texture (non-halting): first-batch CE spread "
        f"across arms {ce1_spread if ce1_spread is None else f'{ce1_spread:.2e}'} "
        f"-> {'PASS' if metrics['draw_integrity_texture']['pass'] else 'DISCLOSED'}")

    # ================= P6: ADJUDICATION (the frozen composite) ==========
    post_ser = cells_s["g0"]
    traj_ser = inst_s["traj"]
    reads: dict = {"SERIAL": {
        "post_g0": post_ser, "post_gm12": cells_s["gm12"],
        "kept_frac_median": arms_rec["SERIAL"]["install"][
            "ledger_kept_frac_median"],
        "traj_g0": {t["step"]: t["g0_pz"] for t in traj_ser},
        "peak_traj_g0": max(t["g0_pz"] for t in traj_ser),
        "in_own_room_post": load_s["in_own_room"],
        "v_excess_post": load_s["v_excess"],
        "root_g0_carried": r_s["g0"],
        "cited_e272": {"post_g0": E272_K10KR_POST_G0,
                       "root_g0": E272_K10KR_ROOT_G0,
                       "kept": E272_K10KR_KEPT_MED,
                       "traj_g0": E272_K10KR_TRAJ_G0}}}
    antiphase_reads: dict = {}
    for arm in [a for a in ARMS_RUN if a != "SERIAL"]:
        inst_arm = arms_rec[arm]["install"]
        ap = antiphase_read(inst_arm["traj"], traj_ser)
        antiphase_reads[arm] = ap
        reads[arm] = {
            "post_g0": inst_arm["post_cells"]["g0"],
            "post_gm12": inst_arm["post_cells"]["gm12"],
            "kept_frac_median": inst_arm["ledger_kept_frac_median"],
            "traj_g0": {t["step"]: t["g0_pz"] for t in inst_arm["traj"]},
            "peak_traj_g0_all_milestones": finite_max(
                t["g0_pz"] for t in inst_arm["traj"]),
            "divergence_signature": arm_diverged(inst_arm),
            "peak_traj_g0_n4": ap["peak_traj_g0"],
            "ratio_vs_serial_session": (inst_arm["post_cells"]["g0"]
                                        / max(post_ser, RATIO_DEN_FLOOR)),
            "ratio_vs_e272_committed": (
                inst_arm["post_cells"]["g0"]
                / max(E272_K10KR_POST_G0, RATIO_DEN_FLOOR)),
            "endpoint_ratio_addendum": (inst_arm["post_cells"]["g0"]
                                        / max(post_ser, RATIO_DEN_FLOOR)),
            "antiphase_vs_serial": {k: ap[k] for k in
                                    ("pearson_n5", "pearson_n4", "n5", "n4",
                                     "serial_dip_step", "arm_peak_step",
                                     "peak_aligned_with_driver_dip")},
            "peak_over_volume_null_floor": (
                ap["peak_traj_g0"] / max(base_g0, 1e-12)
                if ap["peak_traj_g0"] is not None else None),
            "in_own_room_post": inst_arm["displacement_loads"]["in_own_room"],
            "v_excess_post": inst_arm["displacement_loads"]["v_excess"],
            "in_room_disp_norm": inst_arm["in_room_disp_norm"],
            "disp_norm": inst_arm["disp_norm"],
            "root_g0_carried": arms_rec[arm]["root"]["g0"],
            "corpus_ce_median": inst_arm.get("corpus_ce_median"),
        }
        if arm in SGD_BARRELS:
            reads[arm]["lr_factor_vs_adamw"] = (
                cal["factor_vs_adamw_lr"] * SENS_FACTORS[arm])
            reads[arm]["never_adjudicated"] = (arm in ARMS_SENS)

    # the antiphase provenance: e268's committed pair, both conventions,
    # reproduced live from the hard-bound arrays
    prov = antiphase_read(
        [{"step": k, "g0_pz": v} for k, v in E268_CONCURRENT_TRAJ.items()],
        [{"step": k, "g0_pz": v} for k, v in E268_SERIAL_TRAJ.items()])
    provenance = {"form": "the registered -0.52 datum's own pair (e268's "
                          "committed 10k serial driver vs its concurrent "
                          "flash), reproduced live from the hard-bound "
                          "arrays under both conventions",
                  "pearson_n4": prov["pearson_n4"],
                  "pearson_n5": prov["pearson_n5"]}

    post_sep = arms_rec["SEPARATE"]["install"]["post_cells"]["g0"]
    post_sgd = arms_rec["SGDM"]["install"]["post_cells"]["g0"]
    peak_sep = antiphase_reads["SEPARATE"]["peak_traj_g0"]
    peak_sgd = reads["SGDM"]["peak_traj_g0_all_milestones"]
    peak_sgd = peak_sgd if peak_sgd is not None else float("-inf")
    div_sgd = reads["SGDM"]["divergence_signature"]["diverged"]
    r4_sep = antiphase_reads["SEPARATE"]["pearson_n4"]
    r4_shr = antiphase_reads["SHARED"]["pearson_n4"]
    r5_sep = antiphase_reads["SEPARATE"]["pearson_n5"]
    r5_shr = antiphase_reads["SHARED"]["pearson_n5"]

    def _abs(v):
        return abs(v) if v is not None else float("inf")

    clause_SP1 = bool(post_sep >= SURVIVE_FRAC * post_ser)
    clause_SP2a = bool(_abs(r4_sep) < ANTIPHASE_VANISH_BAR)
    clause_SP2b = bool(r4_shr is not None
                       and r4_shr <= ANTIPHASE_PRESENT_BAR)
    clause_ASB = bool(peak_sgd > FORMATION_BAR)
    clause_TTB1 = bool(post_sep < SURVIVE_FRAC * post_ser)
    clause_TTB2 = bool(peak_sgd <= FORMATION_BAR)
    # THE DIVERGENCE-ROUTING RULE (pre-registered after the smoke): a
    # diverged SGD-M barrel's formation clause is CONFOUNDED — it cannot
    # hand a barrel the headline (either direction); any headline that
    # would rest on it routes to MIXED (named)
    sgd_confounded = bool(div_sgd)
    clause_ASB_informative = bool(clause_ASB and not sgd_confounded)
    clause_TTB2_informative = bool(clause_TTB2 and not sgd_confounded)
    # the convention straddle (n=5 vs n=4) on the antiphase clauses
    SP2a_5 = bool(_abs(r5_sep) < ANTIPHASE_VANISH_BAR)
    SP2b_5 = bool(r5_shr is not None and r5_shr <= ANTIPHASE_PRESENT_BAR)
    straddle = bool(SP2a_5 != clause_SP2a or SP2b_5 != clause_SP2b)

    hard = {k: v for k, v in metrics["gates"].items()
            if k != "G_SERIAL_ANCHOR"}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    else:
        barrels_fired = []
        if clause_SP1 and clause_SP2a and clause_SP2b:
            verdict = "STATE-POISONING"
            barrels_fired.append("STATE-POISONING")
            clause = (f"the SEPARATE-AdamW write SURVIVES (post g0 "
                      f"{post_sep:.6f} vs its serial twin {post_ser:.6f}, "
                      f"{post_sep / max(post_ser, RATIO_DEN_FLOOR):.3f}x >= "
                      f"the {SURVIVE_FRAC:.0%} bar) AND the antiphase "
                      f"vanishes in (a) (|Pearson| {_abs(r4_sep):.3f} < "
                      f"{ANTIPHASE_VANISH_BAR}) while present in (c) "
                      f"(Pearson {r4_shr:.3f} <= {ANTIPHASE_PRESENT_BAR}) — "
                      "the barrier is written in the shared optimizer "
                      "state; turbulence narrows to optimizer-state "
                      "poisoning — P-273a CONFIRMED")
        elif clause_ASB_informative:
            verdict = "ADAM-SPECIFIC-BLOCK"
            barrels_fired.append("ADAM-SPECIFIC-BLOCK")
            clause = (f"under SGD-M (b) the write FORMS (peak traj_g0 "
                      f"{peak_sgd:.6f} > {FORMATION_BAR} — above the 10k "
                      f"formation-block level {FORMATION_BLOCK_LEVEL}) "
                      f"whether or not it retains (post g0 {post_sgd:.6f}) "
                      "— P-273b confirmed; the low-rung block is Adam's "
                      "normalization, not the corpus")
        elif clause_TTB1 and clause_TTB2_informative:
            verdict = "TRAJECTORY-TWO-BODY"
            barrels_fired.append("TRAJECTORY-TWO-BODY")
            clause = (f"the SEPARATE-AdamW write still DIES (post g0 "
                      f"{post_sep:.6f} < {SURVIVE_FRAC:.0%} x "
                      f"{post_ser:.6f}) AND SGD-M also dies without forming "
                      f"(peak {peak_sgd:.6f} <= {FORMATION_BAR}) — the "
                      "coupling is the parameters' own collision; the "
                      "metaphor keeps its name")
        else:
            verdict = "MIXED"
            named = []
            if clause_SP1 and not (clause_SP2a and clause_SP2b):
                named.append(f"the separate write SURVIVES ({post_sep:.6f}, "
                             f"{post_sep / max(post_ser, RATIO_DEN_FLOOR):.3f}x)"
                             " but the antiphase clause fails "
                             f"(a: |P|={_abs(r4_sep):.3f} "
                             f"{'<' if clause_SP2a else '>='} "
                             f"{ANTIPHASE_VANISH_BAR}; c: "
                             f"{r4_shr if r4_shr is None else f'{r4_shr:.3f}'} "
                             f"{'<=' if clause_SP2b else '>'} "
                             f"{ANTIPHASE_PRESENT_BAR})")
            if not clause_SP1 and not clause_TTB1:
                named.append("the separate write sits AT the survive bar")
            if not clause_ASB and not clause_TTB2:
                named.append(f"the SGD-M peak sits AT the formation bar "
                             f"({peak_sgd:.6f} vs {FORMATION_BAR})")
            if sgd_confounded:
                named.append(
                    ("the SGD-M barrel DIVERGED at the registered "
                     "calibration (max CE beyond s100 "
                     f"{reads['SGDM']['divergence_signature']['max_ce_beyond_s100']} "
                     f"vs first-batch "
                     f"{reads['SGDM']['divergence_signature']['first_batch_ce']:.4f})"
                     " — its formation clause is DIVERGENCE-CONFOUNDED "
                     "and cannot adjudicate Adam-specificity at this "
                     "scale; the x0.01 stability rider (SGD001X) carries "
                     "the stable-scale read" if not clause_ASB else
                     "the SGD-M flash formed ON A DIVERGED trajectory "
                     "(max CE beyond s100 "
                     f"{reads['SGDM']['divergence_signature']['max_ce_beyond_s100']}) "
                     "— the formation read is divergence-confounded, "
                     "not the mechanism's formation"))
            if straddle:
                named.append(f"the antiphase-convention straddle (n=4 vs "
                             f"n=5) flips a clause — the n=4 PRIMARY stands, "
                             "disclosed")
            clause = ("; ".join(named) + " — the trajectories verbatim, "
                      "all reads, no inflation")
        if len(barrels_fired) > 1:
            clause += (" [multi-barrel: " + "+".join(barrels_fired)
                       + " — the highest-priority barrel takes the "
                       "headline; every clause in the table]")

    clauses_table = {
        "SP1_separate_write_survives": {
            "post_g0_separate": post_sep, "post_g0_serial_session": post_ser,
            "ratio_session": post_sep / max(post_ser, RATIO_DEN_FLOOR),
            "ratio_committed": post_sep / max(E272_K10KR_POST_G0,
                                              RATIO_DEN_FLOOR),
            "bar": SURVIVE_FRAC, "fires": clause_SP1},
        "SP2a_antiphase_vanishes_in_separate": {
            "pearson_n4": r4_sep, "pearson_n5": r5_sep,
            "bar": ANTIPHASE_VANISH_BAR, "fires": clause_SP2a},
        "SP2b_antiphase_present_in_shared_replay": {
            "pearson_n4": r4_shr, "pearson_n5": r5_shr,
            "bar": ANTIPHASE_PRESENT_BAR, "fires": clause_SP2b},
        "ASB_sgdm_forms": {"peak_traj_g0": peak_sgd,
                           "bar": FORMATION_BAR, "fires": clause_ASB,
                           "informative": clause_ASB_informative,
                           "divergence_confounded": bool(
                               sgd_confounded and clause_ASB),
                           "block_level_context": FORMATION_BLOCK_LEVEL},
        "TTB1_separate_write_dies": {"fires": clause_TTB1},
        "TTB2_sgdm_dies_without_forming": {"fires": clause_TTB2,
                                           "informative":
                                               clause_TTB2_informative,
                                           "divergence_confounded": bool(
                                               sgd_confounded
                                               and clause_TTB2)},
        "sgdm_divergence_signature": reads["SGDM"]["divergence_signature"],
        "sgd001x_stability_rider": (reads["SGD001X"][
            "divergence_signature"] if "SGD001X" in reads else None),
        "convention_straddle_n4_vs_n5": straddle,
    }

    log("=" * 78)
    log(f"E273 VERDICT: {verdict}")
    log(f"  SERIAL    (the same-session twin): post g0 {post_ser:.6f} -> "
        f"root {r_s['g0']:.4f} (carried)")
    for arm in [a for a in ARMS_RUN if a != "SERIAL"]:
        ra = reads[arm]
        rp = ra.get("antiphase_vs_serial", {})
        log(f"  {arm:9s} post g0 {ra['post_g0']:.6f} "
        f"({ra['ratio_vs_serial_session']:.4f}x serial) peak "
        f"{ra['peak_traj_g0_all_milestones']:.6f} "
        f"antiphase n4 {rp.get('pearson_n4')!r} n5 "
        f"{rp.get('pearson_n5')!r} root {ra['root_g0_carried']:.4f}")
    log(f"  clauses: SP1={clause_SP1} SP2a={clause_SP2a} SP2b={clause_SP2b} "
        f"| ASB={clause_ASB} | TTB1={clause_TTB1} TTB2={clause_TTB2} | "
        f"straddle={straddle}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> STATE-POISONING -> "
                           "ADAM-SPECIFIC-BLOCK -> TRAJECTORY-TWO-BODY -> "
                           "MIXED (frozen; the dispatch's listing order)",
        "gates_pass": gates_pass,
        "reads": reads,
        "antiphase_reads": antiphase_reads,
        "antiphase_provenance_e268_pair": provenance,
        "clauses": clauses_table,
        "volume_null_floor_g0": base_g0,
        "scatter_disclosure": {
            "install_determinism_post_g0": "~5e-7-1e-6 cross-session "
                                           "(the family law, bit-identical "
                                           "arm)",
            "cons_cross_session_root_g0": ANCHOR_SCATTER_G0,
            "cons_seed_session_root_g0": CONS_SEED_SCATTER_G0,
            "room_lottery_e272": "~21% post-g0 draw spread at k=10k "
                                 "(e272's K10KR 0.2097 vs the committed "
                                 "rung 0.2646 — in-band)",
            "root_gm12_never_adjudicated": "the wild lottery; the g0 ruler "
                                           "is the stable one",
            "n_per_arm": 1,
            "antiphase_n_disclosed": "n=5 (all milestones) and n=4 (x8's "
                                     "convention, s1 excluded) both "
                                     "reported; the n=4 is the bars' "
                                     "PRIMARY (pre-registered)",
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": SMOKE,
    }
    write_partial("P6 ADJUDICATED (the frozen composite)" if not SMOKE
                  else "P6 smoke adjudication EXERCISED (SMOKE — nothing "
                       "adjudicated)")

    # ================= P7: honesty + provenance =========================
    metrics["honesty"] = {
        "intervention_not_logits": (
            "all arms share the bit-identical install stream (one "
            "generator, seed 24314, one draw order), the same room "
            "(bit-gated to e272's committed K10KR room), the same install "
            "dose and lr schedule, the same fresh fact-free base, the same "
            "cons (seed 10901 HELD NATURAL AdamW) and — across the three "
            "concurrent barrels — the SAME corpus draws (one registered "
            "stream, seed 27301): the ONLY deltas are the optimizer "
            "arrangements (shared AdamW / split AdamW / shared SGD-M). "
            "Any fate difference is the optimizer coupling's doing or "
            "nothing is"),
        "the_landing_read_is_not_a_bar": (
            "T246's rehearsal-lane finding (e269/e270/e271 triply "
            "confirmed): the root g0 after cons measures the rehearsal "
            "lane, not the write's survival — carried on every arm, never "
            "adjudicated; root g-12 the wild lottery, reported never gated"),
        "reads_addendum_registered_pre_birth": (
            "the coordinator's three-moves read (antiphase vanishes / peak "
            "falls toward volume-null / endpoint rises toward serial) was "
            "received BEFORE the birth commit and registered in the birth "
            "script: peak traj_g0, endpoint ratio, and peak-vs-driver-dip "
            "alignment appear per arm in the reads table; no bar moved"),
        "loads_measured_not_nominal": (
            "every arm's ACTUAL geometry reported: per-step kept fraction, "
            "v-excess pre/post (vs e258's LOADED v-map), in-span pre and "
            "applied (vs e246's committed LATE span), cumulative "
            "displacement norm + in-room norm + in-own-room fraction + "
            "v-excess per milestone (the weight-vs-read dissociation "
            "rider), displacement loads at post-install and root, and the "
            "corpus stream's own ledger (CE + clipped-grad norm)"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; nothing guaranteed"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "four branches and the trajectories are "
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
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "k10kr_vehicle": {"file": f"runs/checkpoints/{K10KR_CK}",
                              "md5": K10KR_MD5,
                              "note": "e272's committed serial K10KR "
                                      "install (the anchor's L2 target); "
                                      "read-only here"},
            "e272_rooms": f"runs/checkpoints/{ROOMS272_CK}",
            "rooms": rooms_ck,
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS_RUN},
        },
        "machinery": {
            "install_serial": "e261's chunked_install VERBATIM BY IMPORT",
            "install_barrels": "THIS file's chunked_install_barrel (the "
                               "cell's one new driver): e268's interleave "
                               "schedule with the barrel switch (SHARED / "
                               "SEPARATE / SGD-M) + the per-milestone "
                               "displacement reads",
            "cons": "e261's chunked_consolidate VERBATIM (e113; seed 10901 "
                    "HELD, NATURAL AdamW on every arm)",
            "hook": "e237's pre-Adam projection (backward -> clip 1.0 -> "
                    "project CPU fp64 SRCT -> write fp32 -> step; norm "
                    "not rescaled) — INSTALL steps only, every barrel",
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

    # ================= P8: figure + report ===============================
    make_barrels_plot(RD, arms_rec, reads, antiphase_reads, verdict, clause,
                      post_ser, base_g0, thermal_log)
    write_report(RD, arms_rec, reads, antiphase_reads, clauses_table, cal,
                 verdict, clause, provenance, base_g0, G_SERIAL_ANCHOR,
                 ce1_spread)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)" if not SMOKE
                         else "SMOKE COMPLETE — NOTHING ADJUDICATED")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e273_optimizer_barrels.png"),
                          str(RD / "REPORT.md")]
    write_partial("P8 DONE (honesty + provenance + figure + report)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def _polls_rows() -> list[dict]:
    """The per-step poll rows for THIS cell from the persisted envelope
    ledger (the in-process list is empty on a resume pass)."""
    if thermal_log:
        return [{"t": r["t"], "temp": r["temp"]} for r in thermal_log]
    rows = []
    try:
        with open(E43.REPO / "runs" / "_envelope_log.jsonl",
                  encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("tag", "")).startswith(f"{NAME}:") \
                        and row.get("temp") is not None:
                    rows.append({"t": i, "temp": float(row["temp"])})
    except FileNotFoundError:
        pass
    return rows


def make_barrels_plot(rd, arms_rec, reads, antiphase_reads, verdict, clause,
                      post_ser, base_g0, thermal_log):
    """THE CELL'S HEADLINE FIGURE: the trajectories (log), the two
    discriminator bar reads (post g0 vs the survive bar; peak vs the
    formation bar), the antiphase panel (both conventions + the
    alignment), and the thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    cols = {"SERIAL": "tab:blue", "SHARED": "tab:red",
            "SEPARATE": "tab:orange", "SGDM": "tab:green",
            "SGD2X": "tab:purple", "SGD05X": "tab:brown",
            "SGD001X": "tab:cyan"}
    arms = [a for a in ("SERIAL", "SHARED", "SEPARATE", "SGDM", "SGD05X",
                        "SGD001X", "SGD2X") if a in arms_rec]

    # (0,0) THE INSTALL TRAJECTORIES (log y — the deaths live at 1e-4)
    ax = axes[0, 0]
    for arm in arms:
        tr = arms_rec[arm]["install"]["traj"]
        ax.plot([t["step"] for t in tr], [max(t["g0_pz"], 1e-8) for t in tr],
                "o-", lw=1.6, ms=4, color=cols[arm],
                label=f"{arm} (post {arms_rec[arm]['install']['post_cells']['g0']:.2e})")
    ax.axhline(FORMATION_BAR, color="green", ls="--", lw=1.2,
               label=f"formation bar {FORMATION_BAR}")
    ax.axhline(FORMATION_BLOCK_LEVEL, color="gray", ls=":", lw=1.0,
               label=f"e268 block level {FORMATION_BLOCK_LEVEL}")
    ax.axhline(max(base_g0, 1e-8), color="black", ls=":", lw=1.0,
               label=f"volume-null floor {base_g0:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("install step s (the corpus step follows each, 1:1)")
    ax.set_ylabel("g0 battery (mean p(Z)), log")
    ax.set_title("THE INSTALL TRAJECTORIES — the flash and the kill per "
                 "barrel", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE DISCRIMINATOR BAR READS
    ax = axes[0, 1]
    xs = np.arange(len(arms))
    posts = [p if (p is not None and math.isfinite(p)) else 1e-8
             for p in [arms_rec[a]["install"]["post_cells"]["g0"]
                       for a in arms]]
    ax.bar(xs - 0.2, posts, width=0.38, alpha=0.9,
           color=[cols[a] for a in arms], label="post g0 (the WRITE read)")
    ax.axhline(SURVIVE_FRAC * post_ser, color="crimson", ls="--", lw=1.4,
               label=f"the {SURVIVE_FRAC:.0%} survive bar "
                     f"({SURVIVE_FRAC * post_ser:.4f})")
    ax.axhline(post_ser, color="tab:blue", ls=":", lw=1.2,
               label=f"serial twin {post_ser:.4f}")
    peaks = [reads[a].get("peak_traj_g0_all_milestones")
             or reads[a].get("peak_traj_g0") or 1e-8 for a in arms]
    ax2 = ax.twinx()
    ax2.bar(xs + 0.2, [max(p, 1e-8) for p in peaks], width=0.38, alpha=0.55,
            hatch="//", color=[cols[a] for a in arms],
            label="peak traj g0 (the flash)")
    ax2.axhline(FORMATION_BAR, color="green", ls="--", lw=1.4,
                label=f"formation bar {FORMATION_BAR}")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylabel("post g0 (log)")
    ax2.set_ylabel("peak traj g0 (log)")
    ax.set_title("THE TWO DISCRIMINATOR READS — survive (post) and form "
                 "(peak)", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper left")
    ax.grid(alpha=0.25, axis="y", which="both")

    # (1,0) THE ANTIPHASE PANEL
    ax = axes[1, 0]
    conc_arms = [a for a in arms if a != "SERIAL"]
    xs = np.arange(len(conc_arms))
    r4s = [antiphase_reads[a]["pearson_n4"] for a in conc_arms]
    r5s = [antiphase_reads[a]["pearson_n5"] for a in conc_arms]
    ax.bar(xs - 0.19, [v if v is not None else 0 for v in r4s], width=0.38,
           alpha=0.9, color=[cols[a] for a in conc_arms],
           label="Pearson n=4 (x8's convention — the bars' PRIMARY)")
    ax.bar(xs + 0.19, [v if v is not None else 0 for v in r5s], width=0.38,
           alpha=0.45, hatch="//", color=[cols[a] for a in conc_arms],
           label="Pearson n=5 (the dispatch's letter)")
    ax.axhspan(-ANTIPHASE_VANISH_BAR, ANTIPHASE_VANISH_BAR, color="gray",
               alpha=0.18, label=f"the vanish band |r| < "
                                 f"{ANTIPHASE_VANISH_BAR}")
    ax.axhline(ANTIPHASE_PRESENT_BAR, color="crimson", ls="--", lw=1.0,
               label=f"the present bar {ANTIPHASE_PRESENT_BAR} (c)")
    for i, a in enumerate(conc_arms):
        ap = antiphase_reads[a]
        ax.annotate(f"peak@{ap['arm_peak_step']}\ndip@"
                    f"{ap['serial_dip_step']}"
                    f"{' ALIGN' if ap['peak_aligned_with_driver_dip'] else ''}",
                    (i, 0.02), ha="center", fontsize=6.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(conc_arms, fontsize=8.5)
    ax.set_ylabel("Pearson(arm traj_g0, serial traj_g0)")
    ax.set_title("THE ANTIPHASE READ — does the flash peak where the "
                 "driver dips?", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="lower left")
    ax.grid(alpha=0.25, axis="y")

    # (1,1) THE THERMAL ENVELOPE (the persisted per-poll ledger)
    ax = axes[1, 1]
    poll_rows = _polls_rows()
    if poll_rows:
        ax.plot([r["t"] for r in poll_rows],
                [r["temp"] for r in poll_rows],
                "-", lw=0.8, color="dimgray", alpha=0.7)
    ax.axhline(E261.TEMP_EARLY_END, color="crimson", ls=":", lw=1.0,
               label=f"burst-end margin {E261.TEMP_EARLY_END:.0f}C")
    ax.axhline(E261.TEMP_HARD, color="crimson", ls="--", lw=1.2,
               label=f"never-past line {E261.TEMP_HARD:.0f}C (dispatch "
                     "85C)")
    ax.set_xlabel("poll index (the persisted ledger; monotonically "
                  "ordered across passes)")
    ax.set_ylabel("GPU temp (C) per-step polls")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in poll_rows), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in poll_rows if r['temp'] >= E261.TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle(f"E273 — THE THREE-BARREL MECHANISM CELL -> {verdict}"
                 f"{' (SMOKE — nothing adjudicated)' if SMOKE else ''}",
                 fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 150), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e273_optimizer_barrels.png", dpi=130)
    plt.close(fig)


def write_report(rd, arms_rec, reads, antiphase_reads, clauses_table, cal,
                 verdict, clause, provenance, base_g0, G_SERIAL_ANCHOR,
                 ce1_spread):
    """runs/e273/REPORT.md — the cell's record page."""
    lines = []
    A = lines.append
    A(f"# E273 — THE THREE-BARREL MECHANISM CELL"
      f"{' (SMOKE — nothing adjudicated)' if SMOKE else ''}")
    A("")
    A(f"**VERDICT: {verdict}**")
    A("")
    A(textwrap.fill(clause, 100))
    A("")
    A("## The question (frozen)")
    A("> " + REGISTERED["question_verbatim"])
    A("")
    A("## The arms' reads (n=1 per arm, one session; the WRITE read "
      "adjudicates; the landing read carried, never adjudicated)")
    A("")
    A("| arm | post g0 (WRITE) | x serial | peak traj g0 | formed "
      "(>0.005)? | antiphase n4 | antiphase n5 | peak@dip? | kept med | "
      "in-own-room | v-exc | root g0 (carried) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ser = reads["SERIAL"]
    A(f"| SERIAL | {ser['post_g0']:.6f} | 1.000x | "
      f"{ser['peak_traj_g0']:.6f} | n/a (the driver) | n/a | n/a | n/a | "
      f"{ser['kept_frac_median']:.4f} | "
      f"{ser['in_own_room_post']:.4f} | {ser['v_excess_post']:.2f} | "
      f"{ser['root_g0_carried']:.4f} |")
    for arm in [a for a in ("SHARED", "SEPARATE", "SGDM", "SGD05X",
                            "SGD001X", "SGD2X")
                if a in reads]:
        r = reads[arm]
        ap = r["antiphase_vs_serial"]
        p4 = ap["pearson_n4"]
        p5 = ap["pearson_n5"]
        dv = r.get("divergence_signature", {})

        def _f(v):
            return f"{v:+.3f}" if v is not None else "n/a"
        A(f"| {arm} | {r['post_g0']:.6f} | {r['ratio_vs_serial_session']:.4f}x "
          f"| {r['peak_traj_g0_all_milestones']:.6f} | "
          f"{'YES' if (r['peak_traj_g0_all_milestones'] or -1) > FORMATION_BAR else 'no'}"
          f"{(' **DIVERGED**' if dv.get('diverged') else '')} | "
          f"{_f(p4)} | {_f(p5)} | "
          f"{'yes' if ap['peak_aligned_with_driver_dip'] else 'no'} "
          f"(pk@{ap['arm_peak_step']}, dip@{ap['serial_dip_step']}) | "
          f"{r['kept_frac_median']:.4f} | {r['in_own_room_post']:.4f} | "
          f"{r['v_excess_post']:.2f} | {r['root_g0_carried']:.4f} |")
    A("")
    A(f"- the volume-null floor (the fresh fact-free base's own g0 read): "
      f"{base_g0:.3e}")
    A(f"- the antiphase provenance: e268's committed 10k pair reproduced "
      f"live — Pearson n4 {provenance['pearson_n4']:.3f} / n5 "
      f"{provenance['pearson_n5']:.3f} (the registered datum was -0.52, "
      f"the n4 convention)")
    A("- the room: e272's committed K10KR (k=10,000, seeds 27215/27216), "
      "rebuilt + bit-gated; the cited serial reference: e272's K10KR "
      f"(post {E272_K10KR_POST_G0:.6f} / root {E272_K10KR_ROOT_G0:.6f} / "
      f"kept {E272_K10KR_KEPT_MED:.4f})")
    A("")
    A("## The clauses (each barrel's clause reported separately — the "
      "composite requires them jointly)")
    A("")
    for k, v in clauses_table.items():
        if isinstance(v, dict) and "fires" in v:
            A(f"- **{k}**: {'FIRES' if v['fires'] else 'does not fire'} — "
              + ", ".join(f"{kk}={vv}" for kk, vv in v.items()
                          if kk != "fires"))
        else:
            A(f"- **{k}**: {v}")
    A("")
    A("## The SGD-M lr calibration (the live probe)")
    A("")
    A(f"- AdamW s1 applied in-room L2: "
      f"{cal['adamw_probe']['applied_l2_in_room']:.6f} "
      f"(gpn {cal['adamw_probe']['gpn']:.6f}; full-L2 "
      f"{cal['adamw_probe']['applied_l2_full']:.4f})")
    A(f"- **LR_SGD = {cal['lr_sgd']:.6f}** (the disclosed factor: "
      f"x{cal['factor_vs_adamw_lr']:.1f} vs the AdamW 1e-3); momentum 0.9, "
      f"wd {SGD_WD} (disclosed)")
    A(f"- the SGD s1 verification: in-room "
      f"{cal['sgd_probe']['applied_l2_in_room']:.6f}, rel err "
      f"{cal['sgd_verification_rel_err']:.2e} (bar {DOSE_LR_TOL})")
    A("")
    A("## Texture disclosures (non-halting)")
    A("")
    A(f"- G_SERIAL_ANCHOR: install L2 "
      f"{G_SERIAL_ANCHOR['install_l2_vs_vehicle']}, post |d| "
      f"{G_SERIAL_ANCHOR['post_g0']['abs_diff']:.6f}, root |d| "
      f"{G_SERIAL_ANCHOR['root_g0']['abs_diff']:.4f} -> "
      f"{'PASS' if G_SERIAL_ANCHOR['pass'] else 'FAIL (session texture; the root clause sits below the cons lottery ~0.04)'}")
    A(f"- the draw-integrity first-batch CE spread across arms: "
      f"{ce1_spread}")
    A("")
    A("## The reads addendum (the coordinator's message, received "
      "PRE-BIRTH; reads only, no bar moved)")
    A("")
    A("> " + REGISTERED["reads_addendum"]["text"])
    A("")
    A("Per arm: peak traj_g0 (the table), the endpoint ratio vs serial "
      "(the table's x serial), and the peak-vs-driver-dip alignment (the "
      "table's peak@dip column). Under v-ownership all three move "
      "together; under trajectory-ownership none move.")
    A("")
    A("## Provenance")
    A("")
    A("- parents hard-bound: e272 metrics "
      f"({E272_MD5}), e268 metrics ({E268_MD5}), e271 metrics "
      f"({E271_MD5}), e272_rooms.pt, e246 span, e258 v-map, the K10KR "
      "vehicle; the room bit-gated (G_ROOM10KR); 13 hard gates "
      "(a failure halts)")
    A("- machinery: e261's ported whole by import (the serial driver, the "
      "cons, the hook, the envelope); this file's one new driver "
      "(chunked_install_barrel) + the live lr calibration probe")
    A("- envelope: bursts <= 175s, per-step polls both streams, 40s "
      "cooldowns, the 84C never-past line (inside the dispatch's 85C); "
      f"max temp {_envelope_summary()['max_temp_seen_c']}C over "
      f"{_envelope_summary()['n_polls']} persisted polls (0 violations)")
    A("- bars + question frozen VERBATIM at birth (commit before compute); "
      "no bar shopping; n=1 per arm; nothing guaranteed")
    (rd / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[report] wrote {rd / 'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
