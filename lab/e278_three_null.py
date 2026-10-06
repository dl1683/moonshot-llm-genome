"""E278 — THE THREE-NULL COLLISION CELL (the two-body mechanism's
semantics test). Design dispatched 2026-10-05 (T258's map; consult #007's
guided missile); this docstring carries the registered question + bars
VERBATIM, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): e278 asks WHAT THE COLLISION
NEEDS: semantics (the corpus's content), mere energy (any gradient
stream), or mere overlap (spatial presence in the room). Context:
e273 (T258) adjudicated the concurrent kill as TRAJECTORY-TWO-BODY —
the coupling is in the parameters (separate-AdamW died HARDER 0.008x).

THE THREE CONCURRENT ARMS (identical 1:1 interleave, shared AdamW, the
only delta the CORPUS STREAM's construction; k=10k room — e272's
committed K10KR (seeds 27215/27216), bit-gated; the family's standard
protocol; serial reference = e272's K10KR cited + a fresh serial re-run
for the same-session pair, the family convention):
  (a) SEMANTIC (the control twin): the committed corpus generator (the
      family's standard, fresh seed per convention — 27801, the
      e27X-family rule).
  (b) ISOTOPE: a MAX-ENTROPY corpus — random tokens from the same
      vocab, uniform draws, same sequence/window structure (a pure
      kinetic-energy stream: gradients with the right scale and no
      semantics). Own generator (seed 27811). The corpus CE must stay
      near chance (the honesty gate).
  (c) THE GUIDED MISSILE (consult #007's cell): the SEMANTIC corpus's
      gradients PROJECTED ENTIRELY ORTHOGONAL to the 10k room — per
      corpus step, compute g, apply g_perp = g - P_room(g) (the room's
      orthonormal basis is committed; here: the SRCT projector, exact,
      fp64, basis never materialized — e261's certified form), step
      with g_perp (the resulting step-size change disclosed; if
      ||g_perp|| is degenerate small, report and continue with the
      normalized convention, disclosed). The corpus stream steps ONLY
      outside the room. SEMANTIC and MISSILE share the ONE registered
      corpus generator (seed 27801): bit-identical corpus batches, the
      projection is the missile's ONLY delta vs SEMANTIC.

FROZEN BARS (verbatim from the dispatch letter; adjudicate on the WRITE
read; no landing read in this cell — see deviations):
  - COLLISION-IS-SPATIAL: "the MISSILE arm's write SURVIVES (post g0
    >= 0.5x serial) while SEMANTIC and ISOTOPE die — the kill needs
    the corpus stepping IN the room; the guided-missile account
    confirmed; the collision is strictly spatial overlap"
  - COLLISION-IS-ENERGY: "the ISOTOPE kills as hard as SEMANTIC (both
    < 0.5x) — semantics irrelevant; any gradient traffic of the right
    scale destroys the write; the kill is kinetic"
  - UNDERTOW-REGARDLESS: "ALL THREE arms die (< 0.5x) INCLUDING the
    missile — even a corpus that never steps in the room kills the
    write (the undertow drags the write out regardless of where the
    corpus steps); the two-body story gains its second clause"
  - MIXED: "anything else — trajectories verbatim, all reads, no
    inflation"
(P-C-x on record, CITED not re-registered: "the ISOTOPE arm KILLS TOO
(the max-entropy corpus still feeds the shared denominator) iff the
shared-v account owns the kill; any semantic-alignment account
predicts the isotope spares" — T256. The shared-v account is dead
(e273: separate Adams died harder); under two-body the isotope's
prediction is the ENERGY branch.)

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE VEHICLE := the k=10,000 room with seeds 27215/27216 — e272's
    committed K10KR room, rebuilt from its registered seeds and
    bit-gated against runs/checkpoints/e272_rooms.pt's stored K10KR
    D/S (exact equality; G_ROOM10KR); the e001 fresh fact-free base,
    the Dmix install s400 gen 24314, the hook VERBATIM (backward ->
    clip 1.0 -> project onto the room (CPU fp64, write fp32) ->
    opt.step; norm NOT rescaled). NO CONS in this cell (the bars
    adjudicate the WRITE read only; the dispatch's reads list no
    landing read; T259/e281's FLAT verdict — the rehearsal lane
    carries zero write information — makes a cons run uninformative
    for these bars; the post-install states are checkpointed so a
    later cons can be run if wanted).
  * THE SERIAL REFERENCE := e272's committed K10KR arm, CITED (post g0
    0.20972091 / kept 0.06009885; hard-bound in G_PARENTS) + a FRESH
    same-session serial re-run (e261's chunked_install driver VERBATIM
    at the registered seeds — one instrument, one session; the
    e268/e269/e270/e271/e273 precedent): every SURVIVES/DIES ratio's
    denominator "serial" = the SAME-SESSION serial arm (PRIMARY); the
    committed-cited ratio co-reported.
  * THE INTERLEAVE := e268's registered form (e273's port): after
    EVERY install step s (s = 1..400), ONE corpus step (1:1): the
    48-window corpus composition (16 original-host anchors + 32 random
    corpus windows for SEMANTIC/MISSILE; 48 uniform-random windows for
    ISOTOPE); full-window CE; the PAIRED install step's lr
    (lr x cosine_lr(s-1,1000)); backward -> clip 1.0 -> [MISSILE:
    orthogonalize] -> opt.step through the ONE SHARED AdamW (0.9,0.95)
    wd 0.1. The install stream's draws stay bit-identical to SERIAL's
    by construction (separate generators). 800 opt steps per
    concurrent arm.
  * ISOTOPE := per corpus step, ONE draw: torch.randint(V, (48,
    BLOCK), generator=igen(27811)) — uniform over the full 65-token
    vocab, same window geometry (48 x 256), same x/y split ([:, :-1] /
    [:, 1:]). Chance CE = ln(65) = 4.174387. G_ISOTOPE_CE (hard): the
    corpus-CE ledger's SECOND-HALF median (ledger steps > 200) within
    [ln65 - 0.20, ln65 + 0.75] (the model cannot systematically beat
    ln65 on uniform iid tokens — Gibbs; the upper slack absorbs the
    install stream's tug; the transient excluded by the second-half
    form). SMOKE: vacuous (8 steps cannot converge; disclosed).
  * MISSILE := per corpus step, AFTER clip 1.0: g = flat(grads) (CPU
    fp64); g_perp = g - P_room(g); VERIFY ||P_room g_perp|| /
    ||g_perp|| < 1e-6 (checked EVERY corpus step — the dispatch's
    smoke centerpiece made the full-run discipline); write g_perp fp32
    to grads; opt.step. The norm is NOT renormalized (the register
    anticipated degeneracy: ||g_perp||/||g|| = sqrt(1 - kept^2) ~=
    0.998 at kept ~ 0.06 — the raw gradient is only ~0.4% in-room, so
    the projection's step-size cost is ~0.2%, DISCLOSED per ledger;
    the normalized convention is NOT triggered). Adam's per-coordinate
    normalization and AdamW's decoupled wd are NOT projected (the
    optimizer's own nonlinearity is part of the undertow question —
    measured, never assumed).
  * THE DISPLACEMENT READS (per milestone interval, every concurrent
    arm): the corpus stream's realized displacement v = sum of
    (theta_after - theta_before) over the interval's corpus steps
    (fp64 CPU ledger); reported ||v||, ||P_room v||/||v|| (in-room
    fraction), cumulative norm. For the MISSILE the stepped GRADIENT
    is exactly orthogonal (the gate); the realized DISPLACEMENT's
    in-room fraction is a READ — the dispatch's "~0 by construction"
    expectation is TESTED, and if the optimizer's nonlinearity rotates
    the displacement into the room, that rotation is itself the datum
    (named in the adjudication; never a bar move).
  * SURVIVES / DIES := post g0 >= / < 0.5 x the SAME-SESSION serial
    arm's post g0 (PRIMARY; e272-cited ratio co-reported).
  * "as hard as" (the ENERGY clause's isotope-vs-semantic depth
    comparison) := both < 0.5x AND ratio_isotope <= 2 x
    ratio_semantic + 0.01 (the quantitative form; co-reported inside
    every branch where both die).
  * THE COMPOSITE (the four bars' letters are not a partition —
    ENERGY's letter condition (S and I both < 0.5x) is exactly the
    UNION of SPATIAL's and UNDERTOW's signatures; named here at birth,
    no bar moved): TEXTURE (any hard-gate failure, OR the serial twin
    fails to express post g0 < 0.05 — the instrument's own control;
    nothing adjudicated) -> EDGE-MIXED (any adjudicating arm's ratio
    within 0.125 of the 0.5x bar — e268's AT-bar convention) ->
    COLLISION-IS-SPATIAL (S < 0.5x AND I < 0.5x AND M >= 0.5x — the
    letter's exact signature) -> UNDERTOW-REGARDLESS (S < 0.5x AND
    I < 0.5x AND M < 0.5x — the letter's exact all-three signature) ->
    MIXED (anything else, NAMED: CONTROL-TWIN-NO-KILL if S >= 0.5x —
    the concurrent kill did not reproduce this session, the nulls have
    no kill to explain; SEMANTICS-REQUIRED if S < 0.5x and I >= 0.5x —
    P-C-x's isotope-spares branch, any semantic-alignment account's
    prediction; the residual). COLLISION-IS-ENERGY never takes the
    headline (its letter condition is the union of the two firing
    signatures above); its CONTENT (semantics irrelevant; the kill is
    kinetic) is evaluated and co-reported as the named sub-clause
    "ENERGY-CO-FIRE (the isotope clause)" inside every branch where
    both S and I die.
  * HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR,
    G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND,
    G_PROJ, G_ROOM10KR, G_ISOTOPE_CE, G_MISSILE_ORTH} — a failure
    HALTS (nothing adjudicated). G_SERIAL_ANCHOR (the serial re-run vs
    e272's committed K10KR rung: install L2 <= 5e-3; |d post g0| <=
    0.02) is NON-HALTING texture (the family precedent).
  * READS ON EVERY ARM: post g0 (the WRITE read) at s400 + post gm12
    + CE_R; traj_g0 milestones s1/100/200/300/400; the corpus CE
    ledger (the isotope must sit near chance); for the MISSILE the
    orthogonality ledger (per-step rel err, max) and the in-room
    component of the corpus displacement per milestone; kept median;
    the corpus clipped-grad-norm ledger.

CHECKS (the dispatch's, in force): the machinery smoke FIRST (the
e260-family record: 2-3 bugs caught per build — the missile's
projection arithmetic is the smoke's centerpiece: g_perp's in-room
component < 1e-6 relative); the room certified AND bit-bound to e272's
committed K10KR room; the interleave registered exactly (above); the
first-batch CE identity across arms (the draw-integrity texture
check, non-halting); n=1 per arm (the lottery note); nothing
guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier);
bursts <= 175s (inside the dispatch's 180s), per-step thermal polls
(BOTH streams' opt steps) at a 78C margin, 40s cooldowns (the 30-60s
window), the 84C never-past line (inside the dispatch's 85C); CPU
fp64 dense projections (pocketfft workers 2); CPU probing threads 4;
NO concurrent GPU jobs.

Outputs: runs/e278/{metrics.json (PROGRESSIVE),
e278_three_null.png, REPORT.md, run.log (gitignored)}; checkpoints
runs/checkpoints/e278_*.pt (gitignored; md5s in metrics). No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Commit +
push per phase.

Run:  cd lab && python e278_three_null.py    (E278_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E278_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e278_smoke" if SMOKE else "e278"
assert torch.cuda.is_available(), "e278 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268/e273's disclosed convention): e261's ported
# drivers resolve their module globals (log / NAME / LADDER / RUNG_NAMES /
# SMOKE / INST_STEPS / T0 / thermal ledgers) AT CALL TIME through e261's
# module namespace — rebound HERE so they write THIS cell's log, label THIS
# cell's envelope polls, and run THIS cell's single-rung ladder. The
# committed lab/e261_rank_ladder.py is untouched.
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
ROOMS272_MD5 = "066944855b3295e8796c6ca28b2e498c"
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

# the arms (execution order: the serial anchor first — it is every ratio's
# denominator; then the control twin; then the two nulls)
ARMS = ("SERIAL", "SEMANTIC", "ISOTOPE", "MISSILE")
CONCURRENT_ARMS = ("SEMANTIC", "ISOTOPE", "MISSILE")

# THIS cell's OWN REGISTERED FRESH corpus stream (the e268-family
# convention: one fresh registered stream per concurrent cell — e268
# 26801, e269 26901, e270 27001, e271 27101, e273 27301, HERE 27801).
# SEMANTIC and MISSILE share it (the missile's ONLY delta vs SEMANTIC is
# the orthogonal projection — bit-identical corpus batches by construction).
CORPUS_GEN_SEED = 27801
ISOTOPE_GEN_SEED = 27811           # the isotope's own uniform-token stream

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

E273_METRICS = E43.REPO / "runs" / "e273" / "metrics.json"
E273_MD5 = "df0f608ad0b89cbb0407049418ca4f86"
E273_VERDICT = "MIXED"                     # the SGDM barrel's divergence
                                            # routing; TTB1 (separate-AdamW
                                            # dies) FIRED — T258's basis
E273_SERIAL_POST = 0.209721177816391
E273_SHARED_POST = 0.005288494750857353     # the control twin's committed
                                            # same-session kill (0.025x)
E273_SEPARATE_POST = 0.0015990192769095302  # died HARDER (0.008x) — the
                                            # TRAJECTORY-TWO-BODY datum

# e272's committed K10KR install vehicle (the serial anchor's L2 target)
K10KR_CK = "e272_K10KR_inst_resume.pt"
K10KR_MD5 = "6ce712a7aae8482a4c78d1b87f3194b1"
K10KR_SIZE = 32958618
K10KR_STEP = 400
K10KR_TRAJ_STEPS = [1, 100, 200, 300, 400]
K10KR_LEDGER_MAX = 400

# the discriminator's frozen numbers
SURVIVE_FRAC = 0.5               # "post g0 >= 0.5x serial" (SURVIVES/DIES)
EDGE_BAND = 0.125                # the AT-bar convention (e268: quarter-bar)
AS_HARD_FACTOR = 2.0             # "as hard as": ratio_I <= 2 x ratio_S...
AS_HARD_SLACK = 0.01             # ...+ 0.01 (absolute ratio slack)
G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 — the serial control's floor
MATCH_BAND = E261.MATCH_BAND              # the landing band (context only)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
ANCHOR_BEHAV_TOL = 0.02           # the family's session-texture bar
RATIO_DEN_FLOOR = 1e-6            # the ladder's floor-guard convention

# the isotope's chance-CE gate (the vocab is the family's 65-token set)
VOCAB_EXPECT = 65
CHANCE_CE = math.log(VOCAB_EXPECT)        # ln(65) = 4.174387
ISOTOPE_CE_LO = CHANCE_CE - 0.20
ISOTOPE_CE_HI = CHANCE_CE + 0.75

# the missile's orthogonality gate
ORTH_BAR = 1e-6                   # ||P_room g_perp|| / ||g_perp|| < 1e-6

TRAJ_MILE = (1, 100, 200, 300, 400)

ARM_DESC = {
    "SERIAL": "the committed K10KR rung run ALONE (e272's room, the "
              "ladder's committed condition): e261's chunked_install driver "
              "VERBATIM at the registered seeds, re-run FRESH on this "
              "session — the same-session serial twin (every ratio's "
              "PRIMARY denominator); anchored to e272's committed K10KR arm "
              "(G_SERIAL_ANCHOR, non-halting texture)",
    "SEMANTIC": "(a) the control twin: e268/e273's registered 1:1 interleave "
                "re-run on THIS cell's fresh corpus stream (seed 27801) — "
                "the install step bit-identical to SERIAL's + ONE free "
                "48-window corpus step (16 original-host anchors + 32 "
                "random corpus windows, the g1c Dmix corpus convention) "
                "through the ONE SHARED AdamW (0.9,0.95) wd 0.1 — the "
                "kill this cell's nulls must explain",
    "ISOTOPE": "(b) the max-entropy null: the identical interleave/schedule/"
               "optimizer with the corpus step's 48 windows drawn UNIFORM "
               "over the full 65-token vocab (own generator, seed 27811; "
               "same window geometry 48x256; same x/y split) — a pure "
               "kinetic-energy stream: gradients with the right scale and "
               "no semantics; the corpus CE must sit near chance (ln 65 = "
               f"{CHANCE_CE:.4f}; the honesty gate)",
    "MISSILE": "(c) the guided missile (consult #007): SEMANTIC's twin — "
               "bit-identical corpus draws (the ONE registered generator "
               "seed 27801) — with the corpus gradient PROJECTED ENTIRELY "
               "ORTHOGONAL to the K10KR room after the clip: g_perp = g - "
               "P_room(g) (CPU fp64, write fp32, norm NOT renormalized; "
               "||g_perp||/||g|| ledgered ~0.998); the corpus stream steps "
               "ONLY outside the room (the orthogonality gate, checked "
               "every corpus step)",
}

REGISTERED = {
    "question_verbatim": "e278 asks WHAT THE COLLISION NEEDS: semantics "
        "(the corpus's content), mere energy (any gradient stream), or "
        "mere overlap (spatial presence in the room). Context: e273 (T258) "
        "adjudicated the concurrent kill as TRAJECTORY-TWO-BODY — the "
        "coupling is in the parameters (separate-AdamW died HARDER 0.008x).",
    "bars_verbatim": {
        "COLLISION-IS-SPATIAL": "the MISSILE arm's write SURVIVES (post g0 "
            ">= 0.5x serial) while SEMANTIC and ISOTOPE die — the kill "
            "needs the corpus stepping IN the room; the guided-missile "
            "account confirmed; the collision is strictly spatial overlap",
        "COLLISION-IS-ENERGY": "the ISOTOPE kills as hard as SEMANTIC (both "
            "< 0.5x) — semantics irrelevant; any gradient traffic of the "
            "right scale destroys the write; the kill is kinetic",
        "UNDERTOW-REGARDLESS": "ALL THREE arms die (< 0.5x) INCLUDING the "
            "missile — even a corpus that never steps in the room kills "
            "the write (the undertow drags the write out regardless of "
            "where the corpus steps); the two-body story gains its second "
            "clause",
        "MIXED": "anything else — trajectories verbatim, all reads, no "
            "inflation",
    },
    "predictions_cited": {
        "P-C-x_T256": "the ISOTOPE arm KILLS TOO (the max-entropy corpus "
            "still feeds the shared denominator) iff the shared-v account "
            "owns the kill; any semantic-alignment account predicts the "
            "isotope spares (CITED from the record, not re-registered; "
            "the shared-v account is dead — e273's separate Adams died "
            "harder; under two-body the isotope's prediction is the ENERGY "
            "branch)",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE VEHICLE := the k=10k room seeds "
        "27215/27216 rebuilt + bit-gated vs e272_rooms.pt's K10KR D/S "
        "(G_ROOM10KR); e001 base; Dmix s400 gen 24314; hook = clip 1.0 -> "
        "project CPU fp64 -> opt.step, norm NOT rescaled; NO CONS (the "
        "bars are WRITE-read-only; T259/e281: the rehearsal lane carries "
        "zero write information; post states checkpointed); THE SERIAL "
        "REFERENCE := e272's committed K10KR CITED (post "
        f"{E272_K10KR_POST_G0:.8f} / kept {E272_K10KR_KEPT_MED:.8f}, "
        "hard-bound) + a fresh same-session serial re-run (the ratios' "
        "PRIMARY denominator); THE INTERLEAVE := e268's registered form "
        f"(corpus generators: semantic+missile {CORPUS_GEN_SEED} — ONE "
        f"stream, the missile's only delta is the projection; isotope "
        f"{ISOTOPE_GEN_SEED} uniform over the 65-token vocab, same 48x256 "
        "geometry); ISOTOPE gate := second-half corpus-CE median within "
        f"[{ISOTOPE_CE_LO:.4f}, {ISOTOPE_CE_HI:.4f}] (chance ln65 = "
        f"{CHANCE_CE:.4f}); MISSILE := after clip, g_perp = g - P_room(g) "
        "(CPU fp64, write fp32, NOT renormalized — the degeneracy clause "
        "not triggered, ledgered), orthogonality verified EVERY corpus "
        f"step at < {ORTH_BAR:.0e} relative; the corpus DISPLACEMENT "
        "reads (per-milestone interval, fp64) are READS — the missile's "
        "realized displacement's in-room fraction tests the '~0 by "
        "construction' expectation (the optimizer's nonlinearity is part "
        "of the undertow question, measured never assumed); SURVIVES/DIES "
        ":= post g0 >= / < 0.5 x the SAME-SESSION serial (cited ratio "
        "co-reported); 'as hard as' := ratio_I <= 2 x ratio_S + 0.01; "
        "COMPOSITE := TEXTURE (hard-gate failure OR serial twin < 0.05) "
        "-> EDGE-MIXED (any adjudicating ratio within 0.125 of the 0.5 "
        "bar) -> COLLISION-IS-SPATIAL (S<0.5 & I<0.5 & M>=0.5) -> "
        "UNDERTOW-REGARDLESS (S<0.5 & I<0.5 & M<0.5) -> MIXED (named: "
        "CONTROL-TWIN-NO-KILL if S>=0.5; SEMANTICS-REQUIRED if S<0.5 & "
        "I>=0.5; the residual) — ENERGY's letter condition is the UNION "
        "of SPATIAL's and UNDERTOW's signatures, so ENERGY never takes "
        "the headline; its content is co-fired as the named sub-clause "
        "inside every both-die branch; HARD GATES := {G_NAMEFREE, "
        "G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, "
        "G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOM10KR, G_ISOTOPE_CE, "
        "G_MISSILE_ORTH} (a failure HALTS); G_SERIAL_ANCHOR NON-HALTING."),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (T258's map; consult #007's guided missile; the queue's "
        "e278 row); this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "NO CONS, NO ROOT/LANDING READS IN THIS CELL (registered at birth): "
    "the dispatch's reads list carries NO landing read and every bar "
    "adjudicates the WRITE read (post g0 at s400); T259/e281's FLAT "
    "verdict (the cons-only floor 0.6508 in-band from a FACT-FREE base — "
    "the rehearsal lane carries zero write information) makes a cons run "
    "uninformative for these bars. The four post-install states are "
    "checkpointed (e278_<ARM>_post.pt) so a later cons can be run on "
    "exactly these states if the question ever returns. No bar or "
    "gate-form change follows.",
    "SEMANTIC AND MISSILE SHARE THE ONE REGISTERED CORPUS STREAM "
    "(registered at birth): both draw from the single fresh generator "
    f"(seed {CORPUS_GEN_SEED}) — bit-identical corpus batches across the "
    "two arms, so the missile's ONLY delta vs the control twin is the "
    "orthogonal projection (the cleanest two-body form; e273's "
    "shared-stream-across-barrels precedent). The isotope's uniform "
    f"stream is its own generator (seed {ISOTOPE_GEN_SEED}).",
    "THE MISSILE'S STEP-SIZE DISCLOSURE (measured, never assumed): the "
    "projection acts on the CLIPPED corpus gradient; ||g_perp||/||g|| = "
    "sqrt(1 - kept^2) with kept = the corpus gradient's in-room fraction "
    "(~0.06 at this room) — a ~0.2% step-size cost, ledgered per step; "
    "the dispatch's degeneracy clause (renormalize if ||g_perp|| is "
    "degenerate small) is NOT triggered (first-step and median ratios "
    "disclosed in the orth ledger). Adam's per-coordinate normalization "
    "and AdamW's decoupled wd are NOT projected — the optimizer's own "
    "nonlinearity is part of the undertow question, and the realized "
    "corpus displacement's in-room fraction is MEASURED per milestone "
    "interval (the reads that test the '~0 by construction' expectation).",
    "THE COMPOSITE IS REGISTERED, NOT ASSUMED (named at birth): the four "
    "bars' letters do not form a partition — ENERGY's letter condition "
    "(S and I both die) is exactly the UNION of SPATIAL's and UNDERTOW's "
    "signatures. The registered order (TEXTURE -> EDGE-MIXED -> SPATIAL "
    "-> UNDERTOW -> MIXED-named) keeps every clause's letter intact: "
    "SPATIAL and UNDERTOW fire on their exact signatures; ENERGY's "
    "CONTENT (semantics irrelevant; kinetic) is co-fired as the named "
    "sub-clause with its quantitative form (ratio_I <= 2 x ratio_S + "
    "0.01) inside every both-die branch. No bar moved.",
    "THE SERIAL ARM IS A FRESH RE-RUN, NOT A CITE (the family precedent): "
    "a same-session re-run at the registered seeds makes every null's "
    "ratio one instrument on one session; its delta vs e272's committed "
    "K10KR arm doubles as the freshest cross-session scatter point "
    "(co-reported; install determinism |d post g0| ~ 5e-7-1e-6 on a "
    "bit-identical arm — the family law).",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked serial install driver (bit-identical arithmetic + draw "
    "order), the thermal envelope (per-step polls, 78C margin, 175s "
    "bursts — inside the dispatch's 180s — 40s cooldowns, the 84C line "
    "inside the dispatch's 85C), the progressive-metrics + resume-ckpt "
    "conventions. The module-global rebinding (log/NAME/LADDER/RUNG_NAMES/"
    "T0/thermal ledgers, disclosed in-code) retargets the drivers' I/O to "
    "this cell; the committed lab/e261_rank_ladder.py is NOT modified. "
    "The ONE new driver is chunked_install_threenull (this file) — it "
    "takes the room mode as a PARAMETER (e268's smoke catch honored by "
    "construction) and adds the arm-parameterized corpus construction + "
    "the missile's orthogonalization + the corpus-displacement ledger.",
    "THE FIRST-BATCH CE IDENTITY CHECK (non-halting texture): every "
    "concurrent arm's ledger s1 INSTALL CE must equal SERIAL's (same "
    "draws, same base, same loss arithmetic — the draw-integrity read); a "
    "miss is disclosed session texture, never a bar move.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E278_SMOKE=1): 8 install + 8 interleaved corpus steps, "
    "room k=512 at the same seed pair (G_ROOM10KR vacuous — no committed "
    "record at smoke k; disclosed), all four arms + the adjudication + "
    "the figure exercised, the missile's orthogonality checked EVERY "
    "step (the smoke's centerpiece), G_ISOTOPE_CE VACUOUS (8 steps "
    "cannot converge to chance; disclosed), all paths smoke_-prefixed, "
    "own smoke dir; NOTHING adjudicated or gated (SMOKE stamp on every "
    "read).",
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


# --------------------------------------------- THE MISSILE'S PROJECTION (new)
def orthogonalize_grads(proj: "E261.LadderRooms", params, mode: str) -> dict:
    """THE GUIDED MISSILE'S ONLY INTERVENTION: replace the (clipped) corpus
    gradient g by g_perp = g - P_room(g) — the component ENTIRELY ORTHOGONAL
    to the room (CPU fp64, write fp32, norm NOT rescaled). The verification
    read ||P_room g_perp|| / ||g_perp|| is returned for the gate (the
    projector is exact: P(I-P) = 0 to fp64 roundoff ~1e-15; any drift above
    1e-6 is an implementation bug, not numerics)."""
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


# ------------------------------------------- THE THREE-NULL DRIVER (the new)
def chunked_install_threenull(tag, net0, proj: "E261.LadderRooms",
                              base_flat_np: np.ndarray,
                              inst_x, inst_mask, anchor_full, train_ids,
                              vocab: int, g0_ids, gm12_ids, r_eval_xy, zid,
                              mode: str, resume_ck: Path, dev: torch.device,
                              arm: str) -> dict:
    """THE CONCURRENT ARMS' DRIVER (this cell's only new machinery; the
    SERIAL arm runs e261.chunked_install VERBATIM).

    The 1:1 interleave (e268's registered form): per iteration s = 1..400,
      1. INSTALL STEP: bit-identical to SERIAL's step s (draws ix(16)/
         aj(16)/rj(32) from gen seed 24314, the same draw order; the
         64-window Dmix batch; the masked union CE; lr x
         cosine_lr(s-1,1000); backward -> clip 1.0 -> HOOK: project onto
         the room (CPU fp64, write fp32; the mode is a PARAMETER) ->
         ledger dots -> opt.step).
      2. CORPUS STEP (the ONLY delta across the three arms):
         SEMANTIC/MISSILE — draws (aj_c/rj_c) from cgen (seed 27801, the
           ONE registered stream both arms share); 16 anchors + 32 random
           corpus windows; full-window CE;
         ISOTOPE — ONE draw randint(V, (48, BLOCK)) from igen (seed
           27811); uniform over the vocab; same geometry;
         then: the PAIRED lr; backward -> clip 1.0 -> [MISSILE:
         g_perp = g - P_room(g), verified] -> opt.step through the ONE
         SHARED AdamW (0.9,0.95) wd 0.1.

    The displacement ledger (fp64, per arm): theta flattened before/after
    every corpus opt.step; the corpus stream's realized displacement
    accumulated; per milestone interval the in-room fraction of the
    interval's corpus displacement is projected and recorded. Thermal: a
    poll after EVERY opt step (both streams)."""
    name_bs, corp_bs, mix_random, lr = (G1.NAME_BS, E43.CORP_BS,
                                        E43.MIX_RANDOM, E43.LR)
    n_steps = E261.INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    N = int(base_flat_np.size)
    state = {"step": 0, "traj": [], "ledger": {}, "corpus_ledger": {},
             "orth_ledger": {}, "disp_ledger": [],
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "corp_prev": torch.zeros(N, dtype=torch.float64)}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at install step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "orth_max": max([v["orth_rel_err"]
                                 for v in state.get("orth_ledger",
                                                    {}).values()],
                                default=None),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt = gen = cgen = igen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    corp_cum = state["corp_cum"].numpy().astype(np.float64).copy()
    corp_prev = state["corp_prev"].numpy().astype(np.float64).copy()
    room = proj.rooms[mode]
    orth_max = 0.0
    orth_ratio_first = None
    orth_ratio_sum, orth_ratio_n = 0.0, 0

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            gen = torch.Generator().manual_seed(E261.FRESH_GEN)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            igen = torch.Generator().manual_seed(ISOTOPE_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                cgen.set_state(state["cgen_state"])
                igen.set_state(state["igen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = common.cosine_lr(step - 1, E261.INST_TOTAL)
            lr_now = lr * f
            for g in opt.param_groups:
                g["lr"] = lr_now
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
            if arm == "ISOTOPE":
                corp_c = torch.randint(vocab, (corp_bs, G1.BLOCK),
                                       generator=igen)
            else:
                aj_c = torch.randint(n_anc, (corp_bs - mix_random,),
                                     generator=cgen)
                rj_c = torch.randint(len(train_ids) - G1.BLOCK - 1,
                                     (mix_random,), generator=cgen)
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
            orth_row = None
            if arm == "MISSILE":
                orth_row = orthogonalize_grads(proj, net.parameters(), mode)
                orth_max = max(orth_max, orth_row["orth_rel_err"])
                r = orth_row["norm_ratio"]
                orth_ratio_first = r if orth_ratio_first is None \
                    else orth_ratio_first
                orth_ratio_sum += r
                orth_ratio_n += 1
                if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                    state["orth_ledger"][step] = {
                        "gn_clipped": gn_c,
                        "gperp_norm": orth_row["gperp_norm"],
                        "norm_ratio": r,
                        "in_room_frac": orth_row["in_room_frac"],
                        "orth_rel_err": orth_row["orth_rel_err"]}
            theta_b = flat_params_cpu(net).double().numpy()
            opt.step()                          # FREE — except the missile's
                                                # gradient was orthogonalized
            theta_a = flat_params_cpu(net).double().numpy()
            corp_cum += theta_a - theta_b
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c}
            n_burst += 1
            ok_t2, temp2 = burst_temp_check(f"{tag}-c{n_chunks}.x")
            chunk_temps.append(temp2)
            if (step in TRAJ_MILE or step == n_steps or SMOKE):
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                d_mil = flat_params_cpu(net).double().numpy() - base_flat_np
                ld_mil = proj.displacement_loads(torch.from_numpy(d_mil),
                                                 mode)
                dn = float(np.linalg.norm(d_mil))
                v_int = corp_cum - corp_prev
                vn = float(np.linalg.norm(v_int))
                if vn > 0:
                    pv = room.project(v_int)
                    in_room_frac_c = float(np.linalg.norm(pv) / vn)
                else:
                    in_room_frac_c = None
                corp_prev = corp_cum.copy()
                cum_n = float(np.linalg.norm(corp_cum))
                pcum = room.project(corp_cum)
                in_room_frac_cum = (float(np.linalg.norm(pcum) / cum_n)
                                    if cum_n > 0 else None)
                state["disp_ledger"].append({
                    "step": step, "interval_norm": vn,
                    "interval_in_room_frac": in_room_frac_c,
                    "cum_norm": cum_n,
                    "cum_in_room_frac": in_room_frac_cum,
                    "cum_in_room_norm": (float(np.linalg.norm(pcum))
                                         if cum_n > 0 else 0.0)})
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
                    "corpus_disp_interval_norm": vn,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum,
                    "orth_max_so_far": (orth_max if arm == "MISSILE"
                                        else None),
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.5f} g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_inst "
                    f"{float(loss.item()):.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} kept "
                    f"{led['kept_frac']:.4f} |d| {dn:.4f} corp|v| "
                    f"{vn if vn else 0.0:.3f} "
                    f"({('in-room %.4f' % in_room_frac_c) if in_room_frac_c is not None else 'n/a'})"
                    + (f" orth {orth_max:.1e}" if arm == "MISSILE" else ""))
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
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "cgen_state": cgen.get_state(),
                    "igen_state": igen.get_state(),
                    "step": step, "traj": state["traj"],
                    "ledger": state["ledger"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "corp_cum": torch.from_numpy(corp_cum.copy()),
                    "corp_prev": torch.from_numpy(corp_prev.copy()),
                    "orth_max": orth_max,
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
    return {"sd": sd_cpu, "traj": state["traj"], "ledger": state["ledger"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "orth_max": orth_max,
            "orth_norm_ratio_first": orth_ratio_first,
            "orth_norm_ratio_mean": (orth_ratio_sum / orth_ratio_n
                                     if orth_ratio_n else None),
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
    (the PERSISTED per-poll ledger — survives resume passes)."""
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
    out["violations_ge_84c"] = sum(1 for t in temps if t >= E261.TEMP_HARD)
    out["note"] = ("aggregated across ALL passes of this cell; the smoke's "
                   f"polls are tagged e278_smoke: and excluded")
    return out


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e278_three_null",
        "phase": "THE THREE-NULL COLLISION CELL — the two-body mechanism's "
                 "semantics test: WHAT THE COLLISION NEEDS — semantics (the "
                 "corpus's content), mere energy (any gradient stream), or "
                 "mere overlap (spatial presence in the room)? Three "
                 "concurrent arms on ONE rig (1:1 interleave, shared "
                 "AdamW, k=10k room): SEMANTIC (the control twin) vs "
                 "ISOTOPE (max-entropy) vs THE GUIDED MISSILE (gradients "
                 "projected entirely orthogonal to the room) — "
                 "COLLISION-IS-SPATIAL vs COLLISION-IS-ENERGY vs "
                 "UNDERTOW-REGARDLESS vs MIXED, adjudicated on the WRITE "
                 "read",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
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
            "trainings": "4 installs s400 (SERIAL = e261's driver VERBATIM; "
                         "the three nulls = the interleaved driver, 800 opt "
                         "steps each); NO cons (the bars are WRITE-read-"
                         "only; T259/e281 — the rehearsal lane carries zero "
                         "write information; post states checkpointed)",
        },
        "arms_desc": ARM_DESC,
        "interleave": {
            "ratio": "1:1 — after EVERY install step s, ONE corpus step "
                     "(e268's registered form)",
            "install_step": "bit-identical to SERIAL's step s: draws "
                            "ix(16)/aj(16)/rj(32) from gen seed 24314 (the "
                            "same draw order), the 64-window Dmix batch, "
                            "the masked union CE, lr 1e-3 x "
                            f"cosine_lr(s-1, {E261.INST_TOTAL}), clip 1.0 "
                            "-> PROJECT onto the K10KR room -> opt.step",
            "corpus_step_semantic_missile": "48 windows = 16 original-host "
                            "anchors (the same 60-window bank) + 32 random "
                            "corpus windows (the g1c Dmix corpus "
                            "convention); full-window CE; draws from the "
                            f"ONE registered fresh generator seed "
                            f"{CORPUS_GEN_SEED} (bit-identical batches "
                            "across SEMANTIC and MISSILE — the projection "
                            "is the missile's ONLY delta); the PAIRED lr; "
                            "clip 1.0 -> [MISSILE: g_perp = g - P_room(g)] "
                            "-> opt.step",
            "corpus_step_isotope": f"ONE draw randint({VOCAB_EXPECT}, "
                           "(48, 256)) from the isotope's own generator "
                           f"seed {ISOTOPE_GEN_SEED} — uniform over the "
                           "vocab, same window geometry, same x/y split; "
                           "full-window CE; the PAIRED lr; clip 1.0 -> "
                           "opt.step FREE (the max-entropy stream)",
            "shared_optimizer": "ONE AdamW (0.9, 0.95) wd 0.1 across both "
                                "streams on every concurrent arm — the "
                                "moment coupling IS the interference "
                                "channel (e273's two-body verdict)",
            "dose_delta_disclosed": "install dose IDENTICAL across arms "
                                    "(16 masked windows x 400); corpus "
                                    "exposure triples in every concurrent "
                                    "arm (the concurrent stream is the "
                                    "object under test)",
        },
        "deviations": deviations,
        "builds_on": [
            "T258 / e273 (THE two-body verdict: separate-AdamW died HARDER "
            "0.008x — the coupling is in the parameters; this cell asks "
            "what the collision NEEDS; the SHARED-arm control twin's "
            "same-session kill 0.025x is the reproduced baseline)",
            "T244 / e268 (the capstone: DYNAMICAL-CARRIER at 6609x — the "
            "10k write dies under the 1:1 interleave through the ONE "
            "shared AdamW; the interleave's registered form ORIGINATES "
            "here)",
            "consult #007 (agy's second pushback round: THE GUIDED MISSILE "
            "— the corpus's gradients projected entirely orthogonal to the "
            "room; 'if the corpus is forced to take steps strictly outside "
            "the granted room, does the concurrent write survive?' — this "
            "cell's (c) arm, verbatim intent)",
            "T255 / e272 (the relocated edge + the K10KR room: THIS cell's "
            "vehicle room (seeds 27215/27216) is e272's fresh-seed 10k "
            "replicate — post 0.2097, the cited serial reference)",
            "T259 / e281 (the rehearsal lane carries zero write "
            "information — the cons skip's registered basis)",
            "T256 / P-C-x (cited, not re-registered: the isotope kills iff "
            "the shared-v account owned the kill — dead; under two-body "
            "the isotope's prediction is the ENERGY branch)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT: the "
            "SRCT projector, the hooked drivers, the thermal envelope)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314 "
            "+ e113 cons 10901)",
        ],
        "whats_new": [
            "THE SEMANTICS TEST ITSELF: the concurrent kill's necessity "
            "structure asked for the first time — the SAME room, the SAME "
            "bit-identical install stream, the SAME interleave and shared "
            "optimizer, with ONLY the corpus stream's construction changed "
            "(content vs entropy vs geometry)",
            "THE ISOTOPE: the record's first max-entropy concurrent arm — "
            "uniform random tokens, same scale, same schedule, zero "
            "semantics (the honesty gate: the corpus CE pinned at chance)",
            "THE GUIDED MISSILE: the record's first orthogonalized corpus "
            "stream — the semantic corpus's gradients stripped of their "
            "in-room component exactly (verified per step at < 1e-6 "
            "relative), with the realized corpus displacement's in-room "
            "fraction measured per milestone (the optimizer-nonlinearity "
            "read the dispatch's '~0 by construction' expectation tests)",
        ],
        "gates": {},
    })
    log(f"E278 — THE THREE-NULL COLLISION CELL (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)}; vehicle = e272's K10KR room (seeds "
        f"{LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs {ROOMS272_CK}); "
        f"discriminator: SURVIVES >= {SURVIVE_FRAC:.0%}x the same-session "
        f"serial; isotope CE in [{ISOTOPE_CE_LO:.3f}, {ISOTOPE_CE_HI:.3f}] "
        f"(chance {CHANCE_CE:.4f}); missile orth < {ORTH_BAR:.0e}")
    write_partial("startup (bars + composite registered, committed at "
                  "birth)")
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
               "chance_ce_ln_v": CHANCE_CE,
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
        "/ e170 bank / install mask / vocab 65)")
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
    e273m = json.loads(E273_METRICS.read_text(encoding="utf-8"))
    e273_verdict = e273m["adjudication"]["verdict"]
    e273_serial = e273m["adjudication"]["reads"]["SERIAL"]["post_g0"]
    e273_shared = e273m["adjudication"]["reads"]["SHARED"]["post_g0"]
    e273_separate = e273m["adjudication"]["reads"]["SEPARATE"]["post_g0"]
    e273_ttb1 = e273m["adjudication"]["clauses"]["TTB1_separate_write_dies"]
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
                         "note": "the capstone's 10k pair — THE kill this "
                                 "cell's nulls must explain (0.00015x)"},
        "e273_metrics": {"path": str(E273_METRICS),
                         "md5": md5of(E273_METRICS), "bound_md5": E273_MD5,
                         "verdict": e273_verdict,
                         "serial_post_g0": e273_serial,
                         "shared_post_g0": e273_shared,
                         "separate_post_g0": e273_separate,
                         "TTB1_separate_write_dies": e273_ttb1,
                         "note": "THE TWO-BODY VERDICT's record (T258): "
                                 "separate-AdamW died HARDER (0.0076x vs "
                                 "shared 0.025x) — the coupling is in the "
                                 "parameters; the SHARED arm's same-session "
                                 "kill is this cell's control-twin baseline"},
        "k10kr_vehicle": {"path": f"runs/checkpoints/{K10KR_CK}",
                          "md5": md5of(CKPT_DIR / K10KR_CK),
                          "bound_md5": K10KR_MD5,
                          "size": (CKPT_DIR / K10KR_CK).stat().st_size,
                          "bound_size": K10KR_SIZE,
                          "state": vehicle_state},
        "e272_rooms": {"path": f"runs/checkpoints/{ROOMS272_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS272_CK),
                       "bound_md5": ROOMS272_MD5,
                       "note": "THE ROOM FILE (the dispatch's parent-bind "
                               "list); the D/S bit-bind itself is "
                               "G_ROOM10KR"},
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
            "e273_verdict": E273_VERDICT,
            "e273_serial_post_g0": E273_SERIAL_POST,
            "e273_shared_post_g0": E273_SHARED_POST,
            "e273_separate_post_g0": E273_SEPARATE_POST,
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
            and e273_verdict == E273_VERDICT
            and abs(e273_serial - E273_SERIAL_POST) < 1e-12
            and abs(e273_shared - E273_SHARED_POST) < 1e-12
            and abs(e273_separate - E273_SEPARATE_POST) < 1e-12
            and bool(e273_ttb1.get("fires"))
            and md5of(E272_METRICS) == E272_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E273_METRICS) == E273_MD5
            and md5of(CKPT_DIR / ROOMS272_CK) == ROOMS272_MD5
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
        f"{e272_post:.6f}, kept {e272_kept:.4f}); e268 {E268_VERDICT} "
        f"(pair {e268_post_s:.6f} vs {e268_post_c:.6f}); e273 two-body "
        f"(shared {e273_shared:.6f}, separate {e273_separate:.6f}, TTB1 "
        f"fires); the K10KR vehicle md5/step-bound at s{K10KR_STEP}")
    write_partial("P0b parents hard-bound")
    del e272m, e268m, e273m

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
        "e278_rooms",
        {room_name: {"D_int8": rooms.rooms[room_name].D.astype(np.int8),
                     "S": rooms.rooms[room_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e278's vehicle room (the flat basis is "
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

    # ================= P2-P5: THE ARMS (SERIAL first — it anchors) ======
    arms_rec: dict = {}

    def read_arm_cells(sd: dict) -> tuple:
        net_ = G1.evl_load(sd)
        cells_ = {"gm12": G1.battery_cell(net_, gm12_ids, zid)["mean_pz"],
                  "g0": G1.battery_cell(net_, g0_ids, zid)["mean_pz"],
                  "gp12": G1.battery_cell(net_, bat_ids[12], zid)["mean_pz"],
                  "ce_r": G1.ce_fixed_cpu(net_, *r_eval_xy)}
        return net_, cells_

    # ---- ARM-SERIAL (e261's driver VERBATIM at the registered seeds) ----
    log("=" * 78)
    log(f"ARM-SERIAL — {ARM_DESC['SERIAL']}")
    inst_s = E261.chunked_install(
        "SERIAL-inst", ROOM_MODE, G1.evl_load(base_sd), rooms,
        inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e278_SERIAL_inst_resume.pt" if SMOKE
                    else "e278_SERIAL_inst_resume.pt"), dev)
    sd_s = inst_s["sd"]
    net_s, cells_s = read_arm_cells(sd_s)
    d_s = flat_params_cpu(net_s)
    load_s = rooms.displacement_loads(d_s, ROOM_MODE)
    del net_s
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None
    led_s_kept = [v["kept_frac"] for v in inst_s["ledger"].values()]
    arms_rec["SERIAL"] = {
        "desc": ARM_DESC["SERIAL"], "install": {
            "traj": inst_s["traj"], "ledger": inst_s["ledger"],
            "ledger_kept_frac_median": med(led_s_kept),
            "chunk_table": inst_s["chunk_table"], "steps": E261.INST_STEPS,
            "post_cells": cells_s, "displacement_loads": load_s,
            "resumed_final": bool(inst_s.get("resumed_final", False)),
            "opt_steps": E261.INST_STEPS},
    }
    log(f"ARM-SERIAL install done: post g0 {cells_s['g0']:.6f} g-12 "
        f"{cells_s['gm12']:.6f} CE_R {cells_s['ce_r']:.4f} | d v-excess "
        f"{load_s['v_excess']:.2f} cos-to-span {load_s['cos_to_span']:.4f} "
        f"in-own-room {load_s['in_own_room']:.4f} | kept med "
        f"{med(led_s_kept):.4f}")
    write_partial("P2 ARM-SERIAL install + post dial + measured loads")

    # ---- G_SERIAL_ANCHOR (non-halting texture vs e272's committed rung) -
    if not SMOKE:
        veh = torch.load(CKPT_DIR / K10KR_CK, map_location="cpu",
                         weights_only=False)["model"]
        mine_s = torch.load(CKPT_DIR / "e278_SERIAL_inst_resume.pt",
                            map_location="cpu", weights_only=False)["model"]
        serial_l2 = float(np.sqrt(sum(float(((mine_s[k].float()
                                              - veh[k].float()) ** 2).sum())
                                    for k in veh if k in mine_s)))
        del veh, mine_s
    else:
        serial_l2 = None
    G_SERIAL_ANCHOR = {
        "form": "the SERIAL re-run vs e272's committed K10KR rung (the "
                "vehicle's integrity): install L2 vs the committed vehicle "
                "<= 5e-3; |d post g0| <= 0.02 (kept |d| co-reported). "
                "NON-HALTING: a failure is disclosed session texture "
                "(the family precedent), never a bar move — the ratios' "
                "primary denominator is the SAME-SESSION serial arm",
        "install_l2_vs_vehicle": serial_l2,
        "post_g0": {"mine": cells_s["g0"], "e272": E272_K10KR_POST_G0,
                    "abs_diff": abs(cells_s["g0"] - E272_K10KR_POST_G0)},
        "post_gm12": {"mine": cells_s["gm12"], "e272": E272_K10KR_POST_GM12,
                      "abs_diff": abs(cells_s["gm12"]
                                      - E272_K10KR_POST_GM12)},
        "kept_median": {"mine": med(led_s_kept), "e272": E272_K10KR_KEPT_MED,
                        "abs_diff": abs(med(led_s_kept)
                                        - E272_K10KR_KEPT_MED)},
        "bars": {"l2": 5e-3, "behavior": ANCHOR_BEHAV_TOL},
        "pass": None,
    }
    if not SMOKE:
        G_SERIAL_ANCHOR["pass"] = bool(
            serial_l2 <= 5e-3
            and abs(cells_s["g0"] - E272_K10KR_POST_G0) <= ANCHOR_BEHAV_TOL)
    else:
        G_SERIAL_ANCHOR["pass"] = True
        G_SERIAL_ANCHOR["vacuous"] = True
    metrics["gates"]["G_SERIAL_ANCHOR"] = G_SERIAL_ANCHOR
    metrics["arms"] = arms_rec
    anch_txt = ("PASS" if G_SERIAL_ANCHOR["pass"]
                else "FAIL — disclosed session texture")
    serial_l2_txt = ("SMOKE-vacuous" if serial_l2 is None
                     else f"{serial_l2:.3e}")
    log(f"G_SERIAL_ANCHOR (non-halting texture): install L2 "
        f"{serial_l2_txt} (bar 5e-3), post g0 |d| "
        f"{abs(cells_s['g0'] - E272_K10KR_POST_G0):.6f} (bar "
        f"{ANCHOR_BEHAV_TOL}): {anch_txt}")
    write_partial("P2b ARM-SERIAL + G_SERIAL_ANCHOR (non-halting)")

    # ---- THE THREE CONCURRENT ARMS (sequential; cooldowns between) ------
    arm_sds = {"SERIAL": sd_s}

    # ---- ARM-SEMANTIC (the control twin) ---------------------------------
    burst_cooldown("SERIAL -> SEMANTIC")
    log("=" * 78)
    log(f"ARM-SEMANTIC — {ARM_DESC['SEMANTIC']}")
    inst_sem = chunked_install_threenull(
        "SEMANTIC-inst", G1.evl_load(base_sd), rooms, base_flat_np,
        inst_x, inst_mask, anchor_full, train_ids, vocab, g0_ids, gm12_ids,
        r_eval_xy, zid, ROOM_MODE,
        CKPT_DIR / ("smoke_e278_SEMANTIC_inst_resume.pt" if SMOKE
                    else "e278_SEMANTIC_inst_resume.pt"), dev, "SEMANTIC")
    sd_sem = inst_sem["sd"]
    net_sem, cells_sem = read_arm_cells(sd_sem)
    d_sem = flat_params_cpu(net_sem)
    load_sem = rooms.displacement_loads(d_sem, ROOM_MODE)
    del net_sem
    led_sem_kept = [v["kept_frac"] for v in inst_sem["ledger"].values()]
    sem_ce = [v["ce"] for v in inst_sem["corpus_ledger"].values()]
    sem_gn = [v["gn_clipped"] for v in inst_sem["corpus_ledger"].values()]
    arms_rec["SEMANTIC"] = {
        "desc": ARM_DESC["SEMANTIC"], "install": {
            "traj": inst_sem["traj"], "ledger": inst_sem["ledger"],
            "corpus_ledger": inst_sem["corpus_ledger"],
            "corpus_ce_median": med(sem_ce),
            "corpus_gn_clipped_median": med(sem_gn),
            "disp_ledger": inst_sem["disp_ledger"],
            "ledger_kept_frac_median": med(led_sem_kept),
            "chunk_table": inst_sem["chunk_table"], "steps": E261.INST_STEPS,
            "opt_steps": 2 * E261.INST_STEPS,
            "post_cells": cells_sem, "displacement_loads": load_sem,
            "resumed_final": bool(inst_sem.get("resumed_final", False))},
    }
    log(f"ARM-SEMANTIC install done: post g0 {cells_sem['g0']:.6f} g-12 "
        f"{cells_sem['gm12']:.6f} CE_R {cells_sem['ce_r']:.4f} | corpus CE "
        f"med {med(sem_ce):.4f} |g_corp| med {med(sem_gn):.3f} | kept med "
        f"{med(led_sem_kept):.4f}")
    write_partial("P3 ARM-SEMANTIC install + post dial + corpus ledger")

    # ---- ARM-ISOTOPE (the max-entropy null) ------------------------------
    burst_cooldown("SEMANTIC -> ISOTOPE")
    log("=" * 78)
    log(f"ARM-ISOTOPE — {ARM_DESC['ISOTOPE']}")
    inst_iso = chunked_install_threenull(
        "ISOTOPE-inst", G1.evl_load(base_sd), rooms, base_flat_np,
        inst_x, inst_mask, anchor_full, train_ids, vocab, g0_ids, gm12_ids,
        r_eval_xy, zid, ROOM_MODE,
        CKPT_DIR / ("smoke_e278_ISOTOPE_inst_resume.pt" if SMOKE
                    else "e278_ISOTOPE_inst_resume.pt"), dev, "ISOTOPE")
    sd_iso = inst_iso["sd"]
    net_iso, cells_iso = read_arm_cells(sd_iso)
    d_iso = flat_params_cpu(net_iso)
    load_iso = rooms.displacement_loads(d_iso, ROOM_MODE)
    del net_iso
    led_iso_kept = [v["kept_frac"] for v in inst_iso["ledger"].values()]
    iso_ce = [v["ce"] for v in inst_iso["corpus_ledger"].values()]
    iso_gn = [v["gn_clipped"] for v in inst_iso["corpus_ledger"].values()]
    iso_led2 = sorted((int(k), v["ce"]) for k, v in
                      inst_iso["corpus_ledger"].items())
    iso_second_half = [c for s, c in iso_led2 if s > E261.INST_STEPS // 2]
    iso_second_half_med = med(iso_second_half)
    if not SMOKE:
        G_ISOTOPE_CE = {
            "form": f"the isotope's corpus-CE second-half median (ledger "
                    f"steps > {E261.INST_STEPS // 2}) within "
                    f"[{ISOTOPE_CE_LO:.4f}, {ISOTOPE_CE_HI:.4f}] — chance "
                    f"ln({VOCAB_EXPECT}) = {CHANCE_CE:.4f}; Gibbs: no "
                    f"predictor systematically beats chance on uniform "
                    f"iid tokens (the upper slack absorbs the install "
                    f"stream's tug)",
            "second_half_median": iso_second_half_med,
            "band": [ISOTOPE_CE_LO, ISOTOPE_CE_HI],
            "s400_ce": (inst_iso["corpus_ledger"][str(E261.INST_STEPS)]["ce"]
                        if str(E261.INST_STEPS) in inst_iso["corpus_ledger"]
                        else iso_led2[-1][1]),
            "pass": bool(iso_second_half_med is not None
                         and ISOTOPE_CE_LO <= iso_second_half_med
                         <= ISOTOPE_CE_HI),
        }
    else:
        G_ISOTOPE_CE = {
            "form": "SMOKE: VACUOUS — 8 corpus steps cannot converge to "
                    "chance; the gate runs in the full cell only "
                    "(disclosed)",
            "second_half_median": iso_second_half_med,
            "band": [ISOTOPE_CE_LO, ISOTOPE_CE_HI],
            "pass": True, "vacuous": True,
        }
    assert G_ISOTOPE_CE["pass"], f"isotope chance-CE gate FAILED: {G_ISOTOPE_CE}"
    metrics["gates"]["G_ISOTOPE_CE"] = G_ISOTOPE_CE
    arms_rec["ISOTOPE"] = {
        "desc": ARM_DESC["ISOTOPE"], "install": {
            "traj": inst_iso["traj"], "ledger": inst_iso["ledger"],
            "corpus_ledger": inst_iso["corpus_ledger"],
            "corpus_ce_median": med(iso_ce),
            "corpus_ce_second_half_median": iso_second_half_med,
            "corpus_gn_clipped_median": med(iso_gn),
            "disp_ledger": inst_iso["disp_ledger"],
            "ledger_kept_frac_median": med(led_iso_kept),
            "chunk_table": inst_iso["chunk_table"], "steps": E261.INST_STEPS,
            "opt_steps": 2 * E261.INST_STEPS,
            "post_cells": cells_iso, "displacement_loads": load_iso,
            "resumed_final": bool(inst_iso.get("resumed_final", False))},
    }
    log(f"ARM-ISOTOPE install done: post g0 {cells_iso['g0']:.6f} g-12 "
        f"{cells_iso['gm12']:.6f} CE_R {cells_iso['ce_r']:.4f} | corpus CE "
        f"2nd-half med {iso_second_half_med if iso_second_half_med else float('nan'):.4f} "
        f"(chance {CHANCE_CE:.4f}) |g_corp| med {med(iso_gn):.3f} | kept med "
        f"{med(led_iso_kept):.4f}")
    write_partial("P4 ARM-ISOTOPE install + the chance-CE gate")

    # ---- ARM-MISSILE (the guided missile) --------------------------------
    burst_cooldown("ISOTOPE -> MISSILE")
    log("=" * 78)
    log(f"ARM-MISSILE — {ARM_DESC['MISSILE']}")
    inst_mis = chunked_install_threenull(
        "MISSILE-inst", G1.evl_load(base_sd), rooms, base_flat_np,
        inst_x, inst_mask, anchor_full, train_ids, vocab, g0_ids, gm12_ids,
        r_eval_xy, zid, ROOM_MODE,
        CKPT_DIR / ("smoke_e278_MISSILE_inst_resume.pt" if SMOKE
                    else "e278_MISSILE_inst_resume.pt"), dev, "MISSILE")
    sd_mis = inst_mis["sd"]
    net_mis, cells_mis = read_arm_cells(sd_mis)
    d_mis = flat_params_cpu(net_mis)
    load_mis = rooms.displacement_loads(d_mis, ROOM_MODE)
    del net_mis
    led_mis_kept = [v["kept_frac"] for v in inst_mis["ledger"].values()]
    mis_ce = [v["ce"] for v in inst_mis["corpus_ledger"].values()]
    mis_gn = [v["gn_clipped"] for v in inst_mis["corpus_ledger"].values()]
    orth_max = inst_mis["orth_max"]
    G_MISSILE_ORTH = {
        "form": "the missile's stepped corpus gradient is ENTIRELY "
                "ORTHOGONAL to the room: max over ALL corpus steps of "
                "||P_room g_perp|| / ||g_perp|| < 1e-6 (checked EVERY "
                "corpus step — the dispatch's smoke centerpiece as the "
                "full-run discipline; the SRCT projector is exact, so any "
                "drift above fp64 roundoff is an implementation bug)",
        "orth_max_rel_err": orth_max,
        "bar": ORTH_BAR,
        "n_steps_checked": E261.INST_STEPS,
        "norm_ratio_first": inst_mis["orth_norm_ratio_first"],
        "norm_ratio_mean": inst_mis["orth_norm_ratio_mean"],
        "degeneracy_clause": "NOT triggered — ||g_perp||/||g|| ~ 0.998 "
                             "(the raw corpus gradient is only ~0.4% "
                             "in-room energy at this room); the norm is "
                             "NOT renormalized; the step-size change is "
                             "the disclosed ~0.2% projection cost",
        "pass": bool(orth_max is not None and orth_max < ORTH_BAR),
    }
    assert G_MISSILE_ORTH["pass"], f"missile orthogonality gate FAILED: {G_MISSILE_ORTH}"
    metrics["gates"]["G_MISSILE_ORTH"] = G_MISSILE_ORTH
    arms_rec["MISSILE"] = {
        "desc": ARM_DESC["MISSILE"], "install": {
            "traj": inst_mis["traj"], "ledger": inst_mis["ledger"],
            "corpus_ledger": inst_mis["corpus_ledger"],
            "orth_ledger": inst_mis["orth_ledger"],
            "orth_max_rel_err": orth_max,
            "corpus_ce_median": med(mis_ce),
            "corpus_gn_clipped_median": med(mis_gn),
            "disp_ledger": inst_mis["disp_ledger"],
            "ledger_kept_frac_median": med(led_mis_kept),
            "chunk_table": inst_mis["chunk_table"], "steps": E261.INST_STEPS,
            "opt_steps": 2 * E261.INST_STEPS,
            "post_cells": cells_mis, "displacement_loads": load_mis,
            "resumed_final": bool(inst_mis.get("resumed_final", False))},
    }
    log(f"ARM-MISSILE install done: post g0 {cells_mis['g0']:.6f} g-12 "
        f"{cells_mis['gm12']:.6f} CE_R {cells_mis['ce_r']:.4f} | corpus CE "
        f"med {med(mis_ce):.4f} |g_corp| med {med(mis_gn):.3f} | kept med "
        f"{med(led_mis_kept):.4f} | ORTH max {orth_max:.2e} (bar "
        f"{ORTH_BAR:.0e})")
    write_partial("P5 ARM-MISSILE install + the orthogonality gate")

    # ---- the draw-integrity texture check (non-halting) ------------------
    first_ce = {a: arms_rec[a]["install"]["ledger"].get(
        "1", arms_rec[a]["install"]["ledger"].get(1, {})).get("ce")
        for a in ARMS}
    draw_ok = all(v is not None and abs(v - first_ce["SERIAL"]) < 1e-9
                  for v in first_ce.values())
    log(f"draw-integrity (non-halting): first-batch install CE identical "
        f"across arms = {draw_ok} ({first_ce})")

    # ---- the post checkpoints (a later cons can run on exactly these) ----
    arm_sds.update({"SEMANTIC": sd_sem, "ISOTOPE": sd_iso, "MISSILE": sd_mis})
    for arm in ARMS:
        arms_rec[arm]["install"]["checkpoint"] = save_ckpt(
            f"e278_{arm}_post", arm_sds[arm],
            {"desc": f"e278 ARM-{arm} post-install state (s400): "
                     f"{ARM_DESC[arm][:110]}...",
             "arm": arm, "install_seed": E261.FRESH_GEN,
             "corpus_gen_seed": CORPUS_GEN_SEED,
             "isotope_gen_seed": ISOTOPE_GEN_SEED,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e278_rooms.pt"})
    metrics["arms"] = arms_rec
    write_partial("P5b post checkpoints saved")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    post = {a: arms_rec[a]["install"]["post_cells"]["g0"] for a in ARMS}
    serial_post = post["SERIAL"]
    ratio_sess = {a: post[a] / max(serial_post, RATIO_DEN_FLOOR)
                  for a in ARMS}
    ratio_cited = {a: post[a] / max(E272_K10KR_POST_G0, RATIO_DEN_FLOOR)
                   for a in ARMS}
    die = {a: bool(ratio_sess[a] < SURVIVE_FRAC) for a in ARMS}
    edge = {a: bool(abs(ratio_sess[a] - SURVIVE_FRAC) < EDGE_BAND)
            for a in CONCURRENT_ARMS}
    serial_expresses = bool(serial_post >= G0_ZERO_FLOOR)
    as_hard = bool(die["SEMANTIC"] and die["ISOTOPE"]
                   and ratio_sess["ISOTOPE"]
                   <= AS_HARD_FACTOR * ratio_sess["SEMANTIC"] + AS_HARD_SLACK)

    hard = {k: v for k, v in metrics["gates"].items()
            if k != "G_SERIAL_ANCHOR"}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif not serial_expresses:
        verdict = "TEXTURE (SERIAL TWIN FAILED TO EXPRESS)"
        clause = (f"the same-session serial twin's post g0 {serial_post:.6f} "
                  f"sits below the expression floor {G0_ZERO_FLOOR} — the "
                  "instrument's own control failed; nothing adjudicated")
    elif any(edge.values()):
        ed = [a for a in CONCURRENT_ARMS if edge[a]]
        verdict = "MIXED (EDGE-AT-THE-BAR)"
        clause = (f"an adjudicating arm's ratio sits at the {SURVIVE_FRAC}x "
                  f"bar's edge ({', '.join(f'{a} {ratio_sess[a]:.3f}x' for a in ed)}) — "
                  "the trajectories verbatim, all reads, no inflation")
    elif (die["SEMANTIC"] and die["ISOTOPE"]
          and not die["MISSILE"]):
        verdict = "COLLISION-IS-SPATIAL"
        clause = (f"the MISSILE arm's write SURVIVES (post g0 "
                  f"{post['MISSILE']:.6f} = {ratio_sess['MISSILE']:.2f}x "
                  f"serial) while SEMANTIC ({post['SEMANTIC']:.6f} = "
                  f"{ratio_sess['SEMANTIC']:.3f}x) and ISOTOPE "
                  f"({post['ISOTOPE']:.6f} = {ratio_sess['ISOTOPE']:.3f}x) "
                  f"die — the kill needs the corpus stepping IN the room; "
                  "the guided-missile account confirmed; the collision is "
                  "strictly spatial overlap"
                  + ("; ENERGY-CO-FIRE (the isotope clause): the isotope "
                     "kills as hard as the semantic corpus — semantics "
                     "irrelevant, the kinetic content confirmed inside the "
                     "spatial headline" if as_hard else ""))
    elif (die["SEMANTIC"] and die["ISOTOPE"] and die["MISSILE"]):
        verdict = "UNDERTOW-REGARDLESS"
        clause = (f"ALL THREE arms die (< {SURVIVE_FRAC:.0%}x serial: "
                  f"SEMANTIC {ratio_sess['SEMANTIC']:.3f}x, ISOTOPE "
                  f"{ratio_sess['ISOTOPE']:.3f}x, MISSILE "
                  f"{ratio_sess['MISSILE']:.3f}x) INCLUDING the missile — "
                  "even a corpus that never steps in the room kills the "
                  "write (the undertow drags the write out regardless of "
                  "where the corpus steps); the two-body story gains its "
                  "second clause"
                  + ("; ENERGY-CO-FIRE (the isotope clause): the isotope "
                     "kills as hard as the semantic corpus — semantics "
                     "irrelevant, the kinetic content co-fires" if as_hard
                     else "; the isotope dies SOFTER than the semantic "
                          "corpus — partial kinetic content"))
    elif not die["SEMANTIC"]:
        verdict = "MIXED (CONTROL-TWIN-NO-KILL)"
        clause = (f"the SEMANTIC control twin survived this session "
                  f"(post g0 {post['SEMANTIC']:.6f} = "
                  f"{ratio_sess['SEMANTIC']:.2f}x serial, bar "
                  f"{SURVIVE_FRAC:.0%}x) — the concurrent kill did not "
                  "reproduce; the nulls have no kill to explain; the "
                  "trajectories verbatim, all reads, no inflation")
    elif not die["ISOTOPE"]:
        verdict = "MIXED (SEMANTICS-REQUIRED)"
        clause = (f"the SEMANTIC corpus kills ({ratio_sess['SEMANTIC']:.3f}x) "
                  f"but the ISOTOPE spares the write (post g0 "
                  f"{post['ISOTOPE']:.6f} = {ratio_sess['ISOTOPE']:.2f}x "
                  f"serial) — the kill needs the corpus's CONTENT (the "
                  "semantic-alignment accounts' prediction; P-C-x's "
                  "isotope-spares branch); the missile read "
                  f"({ratio_sess['MISSILE']:.3f}x) co-reported; the "
                  "trajectories verbatim, all reads, no inflation")
    else:
        verdict = "MIXED"
        clause = ("the residual configuration — the trajectories verbatim, "
                  "all reads, no inflation")

    log("=" * 78)
    log(f"E278 VERDICT: {verdict}")
    for a in ARMS:
        log(f"  {a:9s}: post g0 {post[a]:.6f} = {ratio_sess[a]:.4f}x "
            f"same-session serial ({ratio_cited[a]:.4f}x cited)")
    log(f"  serial expresses: {serial_expresses}; dies: "
        f"{ {a: die[a] for a in ARMS} }; isotope-as-hard-as: {as_hard}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> EDGE-MIXED -> COLLISION-IS-SPATIAL "
                           "-> UNDERTOW-REGARDLESS -> MIXED (frozen at "
                           "birth; ENERGY's letter condition is the union "
                           "of SPATIAL's and UNDERTOW's signatures — its "
                           "content co-fires as the named sub-clause)",
        "gates_pass": gates_pass,
        "reads": {
            **{a: {"post_g0": post[a], "post_gm12":
                   arms_rec[a]["install"]["post_cells"]["gm12"],
                   "post_ce_r": arms_rec[a]["install"]["post_cells"]["ce_r"],
                   "ratio_vs_serial_session": ratio_sess[a],
                   "ratio_vs_e272_cited": ratio_cited[a],
                   "dies": die[a],
                   "kept_frac_median":
                       arms_rec[a]["install"]["ledger_kept_frac_median"],
                   "traj_g0": {t["step"]: t["g0_pz"]
                               for t in arms_rec[a]["install"]["traj"]},
                   "peak_traj_g0": max(t["g0_pz"] for t in
                                       arms_rec[a]["install"]["traj"]),
                   "in_own_room_post":
                       arms_rec[a]["install"]["displacement_loads"]["in_own_room"],
                   } for a in ARMS},
            "concurrent_extra": {
                a: {"corpus_ce_median":
                        arms_rec[a]["install"]["corpus_ce_median"],
                    "corpus_gn_clipped_median":
                        arms_rec[a]["install"]["corpus_gn_clipped_median"],
                    "corpus_disp_ledger":
                        arms_rec[a]["install"]["disp_ledger"]}
                for a in CONCURRENT_ARMS},
            "e272_committed_rung": {"post_g0": E272_K10KR_POST_G0,
                                    "kept": E272_K10KR_KEPT_MED},
            "e268_capstone_pair": {"serial": E268_SERIAL_POST,
                                   "concurrent": E268_CONCURRENT_POST,
                                   "ratio": E268_RATIO},
            "e273_two_body": {"shared": E273_SHARED_POST,
                              "separate": E273_SEPARATE_POST},
            "energy_cofire_isotope_clause": {
                "as_hard_as": as_hard,
                "form": f"ratio_I <= {AS_HARD_FACTOR}x ratio_S + "
                        f"{AS_HARD_SLACK}",
                "ratio_S": ratio_sess["SEMANTIC"],
                "ratio_I": ratio_sess["ISOTOPE"]},
            "volume_null_floor_g0": base_g0,
        },
        "scatter_disclosure": {
            "install_determinism_post_g0": "~5e-7-1e-6 cross-session on a "
                                           "bit-identical arm (the family "
                                           "law, e268-e273)",
            "serial_anchor_this_session": {
                "install_l2": serial_l2,
                "post_g0_abs_diff": abs(post["SERIAL"] - E272_K10KR_POST_G0)},
            "room_lottery_note": "the vehicle room is e272's K10KR "
                                 "fresh-seed replicate (post 0.2097 vs the "
                                 "committed rung's 0.2646 — the ~21% draw, "
                                 "in-band; every ratio's cited column "
                                 "co-reports against it)",
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P7 ADJUDICATED (the frozen bars)")

    # ================= P9: honesty + provenance + close ==================
    mis_disp = arms_rec["MISSILE"]["install"]["disp_ledger"]
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the four arms share bit-identical install streams (one "
            "generator, seed 24314, one draw order), the same room "
            "(bit-gated to e272's committed K10KR room), the same install "
            "dose and lr schedule, the same fresh fact-free base — the "
            "ONLY delta is the concurrent corpus stream's construction "
            "(content vs entropy vs geometry). SEMANTIC and MISSILE share "
            "bit-identical corpus batches, so the missile's only delta is "
            "the orthogonal projection of its gradients. The rooms' "
            "eigenstructures are identical across arms BY CONSTRUCTION."),
        "the_missile_disclosure": (
            "the missile projects the corpus GRADIENT (after the family's "
            "clip), exactly (verified every corpus step at < 1e-6 "
            "relative); it does NOT project the optimizer's per-coordinate "
            "nonlinearity or AdamW's decoupled weight decay — the realized "
            "corpus displacement's in-room fraction is MEASURED per "
            "milestone interval precisely so the optimizer's own rotation "
            "is a datum, never an assumption"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; the corpus-dose tripling is the "
                        "intervention's own body, ledgered per step"),
        "loads_measured_not_nominal": (
            "every arm's ACTUAL geometry is reported: per-step kept "
            "fraction, v-excess pre/post, in-span pre/applied, "
            "displacement cos-to-span + in-own-room at post-install, the "
            "corpus streams' own ledgers (CE + clipped-grad norm + the "
            "per-milestone realized-displacement in-room fractions) — "
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
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e272_rooms": f"runs/checkpoints/{ROOMS272_CK}",
            "rooms": rooms_ck,
            "post_states": {a: arms_rec[a]["install"]["checkpoint"]
                            for a in ARMS},
        },
        "machinery": {
            "install_serial": "e261's chunked_install VERBATIM BY IMPORT "
                              "(E43.exposure Dmix; the dense-room hook; "
                              "the thermal envelope); the committed "
                              "lab/e261_rank_ladder.py unmodified",
            "install_threenull": "THIS file's chunked_install_threenull "
                                 "(the cell's one new driver): the install "
                                 "steps bit-identical to SERIAL's + the "
                                 "1:1 interleaved corpus steps through the "
                                 "shared AdamW, with the arm-parameterized "
                                 "corpus construction (semantic/isotope) "
                                 "and the missile's orthogonalization + the "
                                 "fp64 corpus-displacement ledger",
            "missile_projection": "THIS file's orthogonalize_grads: g_perp "
                                  "= g - P_room(g) via e261's SRCT "
                                  "projector (exact, fp64, CPU); verified "
                                  "every corpus step",
            "cons": "NONE (registered deviation — the bars are "
                    "WRITE-read-only; T259/e281)",
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
    make_threenull_plot(RD, arms_rec, ratio_sess, die, verdict, clause,
                        thermal_log)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e278_three_null.png"),
                          str(RD / "REPORT.md")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_threenull_plot(rd, arms_rec, ratio_sess, die, verdict, clause,
                        thermal_log):
    """THE CELL'S HEADLINE FIGURE: the trajectories, the discriminator
    bars, the corpus ledgers (the isotope's chance pin), the corpus
    displacement geometry, the missile's orthogonality, the envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    cols = {"SERIAL": "tab:blue", "SEMANTIC": "tab:red",
            "ISOTOPE": "tab:green", "MISSILE": "darkorange"}

    # (0,0) THE INSTALL TRAJECTORIES
    ax = axes[0, 0]
    for arm, rec in arms_rec.items():
        tr = rec["install"]["traj"]
        ax.plot([t["step"] for t in tr], [max(t["g0_pz"], 1e-7) for t in tr],
                "o-", lw=1.6, ms=4, color=cols[arm], label=arm)
    ax.set_yscale("log")
    ax.axhline(E272_K10KR_POST_G0, color="gray", ls=":", lw=1.2,
               label=f"e272 committed K10KR post {E272_K10KR_POST_G0:.4f}")
    ax.axhline(G0_ZERO_FLOOR, color="black", ls="--", lw=0.9,
               label=f"expression floor {G0_ZERO_FLOOR}")
    ax.set_xlabel("install step s (the corpus step follows each, 1:1)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE INSTALL TRAJECTORIES (the write's fate per arm)",
                 fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE DISCRIMINATOR READS
    ax = axes[0, 1]
    names = list(arms_rec.keys())
    posts = [arms_rec[a]["install"]["post_cells"]["g0"] for a in names]
    bars = ax.bar(names, posts, color=[cols[a] for a in names], alpha=0.88)
    ax.axhline(SURVIVE_FRAC * posts[0], color="crimson", ls="--", lw=1.4,
               label=f"the {SURVIVE_FRAC:.0%}x survival bar "
                     f"({SURVIVE_FRAC * posts[0]:.4f})")
    ax.axhline(E272_K10KR_POST_G0, color="gray", ls=":", lw=1.0,
               label=f"e272 committed rung {E272_K10KR_POST_G0:.4f}")
    for b, a in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{arms_rec[a]['install']['post_cells']['g0']:.4f}\n"
                f"{ratio_sess[a]:.3f}x", ha="center", va="bottom",
                fontsize=7.5)
    ax.set_ylabel("post g0 at s400 (the WRITE read)")
    ax.set_title(f"THE DISCRIMINATOR READS — dies: "
                 f"{ {a: die[a] for a in names} }", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, axis="y")

    # (0,2) THE CORPUS LEDGERS (the isotope's chance pin)
    ax = axes[0, 2]
    for a in CONCURRENT_ARMS:
        cl = arms_rec[a]["install"]["corpus_ledger"]
        steps = sorted(int(k) for k in cl.keys())
        ax.plot(steps, [cl[str(k)]["ce"] if str(k) in cl else cl[k]["ce"]
                        for k in steps], "s-", lw=1.3, ms=3.5,
                 color=cols[a], label=f"{a} corpus CE")
    ax.axhline(CHANCE_CE, color="black", ls="--", lw=1.2,
               label=f"chance ln(65) = {CHANCE_CE:.4f}")
    sl = arms_rec["SERIAL"]["install"]["ledger"]
    ssteps = sorted(int(k) for k in sl.keys())
    ax.plot(ssteps, [sl[str(k)]["ce"] if str(k) in sl else sl[k]["ce"]
                     for k in ssteps], "o-", lw=1.0, ms=2.5,
             color="tab:blue", alpha=0.6,
             label="SERIAL install CE (in-batch corpus only)")
    ax.set_xlabel("install step s")
    ax.set_ylabel("batch CE (nats)")
    ax.set_title("THE CONCURRENT STREAMS' CE LEDGERS (the isotope must sit "
                 "at chance)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25)

    # (1,0) THE CORPUS DISPLACEMENT GEOMETRY (the spatial datum)
    ax = axes[1, 0]
    for a in CONCURRENT_ARMS:
        dl = arms_rec[a]["install"]["disp_ledger"]
        ax.plot([d["step"] for d in dl],
                [d["corpus_disp_interval_in_room_frac"] for d in dl],
                "o-", lw=1.4, ms=4.5, color=cols[a],
                label=f"{a} (interval in-room frac)")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.2, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(LADDER[0][0] / 2739072):.4f}")
    ax.set_xlabel("milestone s (interval end)")
    ax.set_ylabel("||P_room v|| / ||v|| (the corpus displacement)")
    ax.set_title("WHERE EACH CORPUS STREAM ACTUALLY WALKED (the missile's "
                 "gradient is orthogonal; its displacement is measured)",
                 fontsize=9.0)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25)

    # (1,1) THE MISSILE'S ORTHOGONALITY
    ax = axes[1, 1]
    ol = arms_rec["MISSILE"]["install"]["orth_ledger"]
    steps = sorted(int(k) for k in ol.keys())
    rels = [max(ol[str(k)]["orth_rel_err"], 1e-18) if str(k) in ol
            else max(ol[k]["orth_rel_err"], 1e-18) for k in steps]
    ax.semilogy(steps, rels, "s-", lw=1.3, ms=3.5, color="darkorange",
                label="||P_room g_perp|| / ||g_perp|| (every 10th shown)")
    ax.axhline(ORTH_BAR, color="crimson", ls="--", lw=1.4,
               label=f"the gate bar {ORTH_BAR:.0e}")
    nrs = [ol[str(k)]["norm_ratio"] if str(k) in ol else ol[k]["norm_ratio"]
           for k in steps]
    ax2 = ax.twinx()
    ax2.plot(steps, nrs, "o--", lw=1.0, ms=2.5, color="gray", alpha=0.7,
             label="||g_perp||/||g|| (the step-size cost)")
    ax2.set_ylabel("||g_perp|| / ||g||", fontsize=8.5)
    ax2.set_ylim(0.95, 1.005)
    ax.set_xlabel("install step s")
    ax.set_ylabel("orthogonality rel err (log)")
    ax.set_title(f"THE MISSILE'S ORTHOGONALITY (max "
                 f"{arms_rec['MISSILE']['install']['orth_max_rel_err']:.1e}; "
                 f"checked EVERY corpus step)", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (1,2) THE THERMAL ENVELOPE
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
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in thermal_log if r['temp'] >= E261.TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle(f"E278 — THE THREE-NULL COLLISION CELL -> {verdict}",
                 fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e278_three_null.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
