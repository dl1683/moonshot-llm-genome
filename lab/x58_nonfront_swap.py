"""X58 — THE NON-FRONT SWAP + THE MATCHED-JITTER RIDER (P-T338a riding;
R84's critic control carried). This docstring carries the question +
background + design + bars + the parity block + P-x58a + THE
PAGE-POSITION SCHEME VERBATIM from the dispatch letter + every frozen
operationalization, committed at birth BEFORE any compute. Adjudicate
against exactly this; no bar shopping.

THE BACKGROUND (dispatch verbatim): "(T338): x36 found EARLY-BLOCK as a
STEP — swapping the first 4 pages (1.3%) destroys the sign consistency
at walk-strength magnitudes, but k4/k16/k64 are indistinguishable, so
front-concentration vs any-perturbation fragility is unresolved. YOUR
CELL splits them, with the critic's rider."

THE DESIGN (dispatch verbatim): "1. THE NON-FRONT SWAP: the LAST 4
pages swapped for the matched-dose neutral (the mirrored design — the
stream's tail replaced instead of its head); x45's instrument at
matched reads, the frozen composite. 2. THE MATCHED-JITTER RIDER (the
content-vs-change control): (a) FRONT-RESHUFFLE — the first 4 pages'
order permuted (same content, changed positions); (b) FRONT-RE-JITTER
— the first 4 pages re-jittered (same content and order, changed
jitter realizations). Three-way fork: content carries it (swap kills,
reshuffle/re-jitter don't) / change carries it (all three kill) /
position carries it (swap + reshuffle kill, re-jitter doesn't)."

BARS (dispatch VERBATIM, frozen in this birth commit BEFORE compute):
  - AT-STRENGTH: "the non-front arm decorrelates — the front
    specifically carries the consistency."
  - DESTROYED-EQUALLY: "the non-front swap kills it too — any
    perturbation."
  - RIDER FORK (verbatim): "content carries it (swap kills,
    reshuffle/re-jitter don't) / change carries it (all three kill) /
    position carries it (swap + reshuffle kill, re-jitter doesn't)."

THE PARITY BLOCK (dispatch verbatim; the lab lean AND the counter side
by side; P-T338a is the lab's registered prediction FOR THIS CELL):
  LAB LEAN (P-T338a, THINKING.md T338 verbatim): "the lab leans
  AT-STRENGTH weakly — the front specifically carries it" (the
  consistency-critical authored content is front-concentrated; a
  matched-dose tail swap leaves the walk-class decorrelation intact;
  on the rider this lean maps to CONTENT — the rider arms keep the
  front's authored content, so they survive).
  COUNTER (P-T338a verbatim): "DESTROYED-EQUALLY — the step shape
  hints any perturbation" (the k4/k16/k64 flatness at walk-strength
  magnitudes is the any-perturbation signature: a knife-edge does not
  care where the cut lands; on the rider this maps to CHANGE — every
  perturbation of the front block, however gentle, kills).
  POSITION (the fork's third leg) has NO committed lean — named as
  the uncovered third.

P-x58a (THE EXECUTOR'S OWN READ, registered per the dispatch's
"Register P-x58a (your own read, both) BEFORE compute", frozen HERE at
birth; predictions are scored): **NON-FRONT: DESTROYED-EQUALLY,
moderately** — the tail4 arm lands the k4 death class (primary-side
COHERENT: sign consistencies 0.50-0.65 vs the walk's 0.78-0.83, geo_mad
at-or-above walk strength), and **RIDER: CHANGE** — the reshuffle and
re-jitter arms BOTH kill. GROUNDS (the derivation, named): (1) THE
TRAJECTORY-CHAOS MECHANISM IS THE MOST CONCRETE READING OF THE STEP —
every arm shares the committed draw stream (ix/aj/rj bit-equal), so
swapping/reshuffling/re-jittering ANY 4 pool rows changes only WHICH
window content lands in the ~16%-of-slots that draw those rows; the
first divergent gradient arrives within a handful of steps (P(a row
0-3 or 296-299 drawn in one 16-slot step) ~ 19-20%) and AdamW at lr
1e-3 compounds it; under this reading the canon walk's sign
consistency is a property of the EXACT committed realization, and ANY
pool perturbation re-rolls the early-run correlation structure to
coin-flip — exactly the measured k4/k16/k64 shape (consistency
0.50-0.63 ~ coin; geo_mad 0.28-0.33 >= walk's 0.26). (2) THE STEP'S
FLATNESS CUTS AGAINST LOCATION — a front-concentration axis should
grade somewhere between 1.3% and 21.3%; none did. (3) THE BIMODAL
PROFILE IS CHAOS-COMPATIBLE — every x36 arm kept the late-phase
negative separation (the robust consolidated face) while only the
early rungs flipped (the fragile emergence face re-rolled); a tail
swap and both rider arms should show the same two-face split.
READ-FATE CO-PREDICTION (scored): all three fresh arms HOLD the host
read (max read >= 0.40; T338's two-currency finding — maintenance is
dose-like, 4/300 pages cannot touch it). RUNNER-UP (priced): the lab
lean — AT-STRENGTH + CONTENT (the seat-grammar reading: the front's
authored content specifically); if tail4 lands AT-STRENGTH I fall to
the runner-up on the non-front bar. SCORED: four booleans —
nonfront==DESTROYED-EQUALLY; rider==CHANGE; reads_hold (all three
fresh arms max read >= 0.40); k4_ref_coherent (the committed k4
reference re-derives COHERENT in-cell — the fork's frozen reference
kill). FALSIFIER (the house clause): any hard-gate failure ->
TEXTURE (nothing adjudicated; P-x58a UNSCORED).

==== THE PAGE-POSITION SCHEME (pre-registered; the disclosure IS ====
==== part of the bar; x36's rig verbatim, one substrate, the     ====
==== shared draw stream)                                        ====

* THE PAGE (x36's definition, verbatim): one cons-pool window; page
  order := the committed jitter-major construction order [j=-8 hosts
  0-59 | j=-4 hosts 0-59 | j=0 | j=+4 | j=+8]; the canon pool = the
  ZEPHYRA lineage's 60 install host occurrences (FLORIZEL 19 +
  ELIZABETH 41, SPLICE_RNG 24301 shuffle order, gated) each at the
  five committed jitters {-8,-4,0,+4,+8}; the neutral pool = x53's
  pool_v VERBATIM (seed 53001, POOL_LOW_S 280, the same jitter-major
  block structure, ZEPHYRA spliced at the same per-jitter column,
  contexts deduped and disjoint from the cons hosts' prefixes).

* THE FOUR ARMS (all from ONE committed substrate —
  runs/checkpoints/g1c_install_resume.pt, md5-bound — ONE committed
  draw stream — cons seed 10901, draw records bit-equal across every
  arm, certified in-cell — and the cons leg arithmetic VERBATIM:
  batch 32 = 16 pool + 8 anchors + 8 random; masked union CE; AdamW
  (0.9,0.95) wd 0.1 const lr 1e-3; clip 1.0; s300; rungs
  {1,2,4,8,25,50,...,300}; anchors + random corpus HELD CANON every
  arm):
    k4 (THE FRONT SWAP — the reference kill): pool = cat([pool_v[:4],
         pool_z[4:]]) — x36's committed k4 leg, RUNTIME-READ from
         runs/x36/leg_k4_resume.pt, NEVER rerun (extend-don't-repeat);
         certified in-cell by G_K4REF (draw-record bit-equality + an
         8-step weight-level bit-proof on my rebuilt k4 pool + my
         pairing re-derives x36's committed k4 scores).
    tail4 (THE NON-FRONT SWAP — the mirrored design): pool =
         cat([pool_z[:296], pool_v[296:]]) — the LAST 4 pages (hosts
         56-59 at j=+8) replaced by pool_v's corresponding tail
         windows; head bit-equal canon, tail bit-equal pool_v, masks
         bit-equal canon; the name-token dose bit-matched per window
         and per step (7 name tokens in EVERY row of every pool).
    fshuf (FRONT-RESHUFFLE — the position perturbation): pool = the
         canon with rows 0-3 replaced by canon rows [3,2,1,0] — THE
         COMMITTED REVERSAL DERANGEMENT (maximal displacement, every
         page moves, no seed ambiguity); every window bit-equal a
         canon window (the content multiset is EXACTLY canon); the
         only delta is WHICH pool index each original page occupies
         (=> which draw slot delivers it).
    frjit (FRONT-RE-JITTER — the realization perturbation): pool =
         the canon with rows 0-3 rebuilt at REDRAWN jitters — one
         seeded draw per page (gen seed 58002) uniform over the
         committed grid MINUS the original {-4, 0, +4, +8} (a changed
         realization by construction); same content (hosts 0-3) and
         same order (host i at row i) — the only delta is each page's
         realized context span and name column. DISCLOSED
         CONSEQUENCE (in-grid redraw): each re-jittered row i is
         bit-equal the canon row of host i at its new jitter
         (duplication), and the original host i@-8 windows leave the
         pool entirely — the exposure multiset shifts by exactly
         these four swaps (the map recorded in G_X58POOLS); the name
         column of rows 0-3 shifts with the jitter (name-token COUNT
         per row unchanged at 7 — dose count bit-matched, column
         shifted; disclosed, inherent to a changed realization).

* DISCLOSED GEOMETRY (x36's, carried): the pool's front (rows 0-3) is
  hosts 0-3 at j=-8 — x36's k4 swap site; the pool's tail (rows
  296-299) is hosts 56-59 at j=+8 — the mirrored site; the rider's
  both arms act on the SAME front block the committed k4 swap
  replaced, so the fork reads content-vs-change-vs-position at
  matched dose, matched location, matched draws.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ==

* COMPUTE ROUTE (chosen + disclosed per the dispatch's "the certified
  CPU-replay route preferred and legal"): THE CERTIFIED CPU-REPLAY
  ROUTE — CPU desk ONLY, torch threads 4, ZERO CUDA calls (the
  sibling desk cell x57 shares the desk; never concurrent CUDA by
  construction). GROUNDS: (i) this family's replays are all
  CPU-certified (x45/x50/x52/x53/x36; the bit-identity gates this
  cell leans on — G_STREAMIDENT_CONS, G_K4REF, G_WALKREPRO — require
  the deterministic CPU realization); (ii) the GPU lane stays free
  for the fleet's next GPU cell; (iii) the phase-2 thermal standing
  is NOT APPLICABLE on this route (no bursts; the envelope record
  carries the route + a gpu_status snapshot at start for the record).
  1800s slice caps with full-state resume (the parked-CPU convention,
  x45/x53/x36's form). The three-proof restart standard if any
  restart: the relaunch re-executes deterministically from the resume
  state; pre-kill row identity IS the proof; disclosed in deviations.

* THE INSTRUMENT (x45's, ported VERBATIM via x53's port — single Z
  channel; NO dual channel: no foreign name exists in any x58 stream,
  the T channel has no referent — disclosed): per-state per-context
  p(Z) at g-12/g0/g+12 (battery_percell bs=30, softmax at the final
  position over the 60-context host battery), alive fractions
  (READ_BAR 0.05), paired-context Pearson r for the three geometry
  pairs, CE_R, |W| vs e001 (fp64, canonical parameters() order) and
  the K10K in-room fraction (the e264 bit-bound room).

* THE RULERS + THE FROZEN SEPARATION RULE (x45's instrument at
  matched reads, x52-certified — x36's form VERBATIM): PRIMARY RULER
  := THE ANNEAL RULER (x45's 40 committed panel rows, runtime-read +
  md5-bound); every arm pairs its rows to the ruler row at minimal
  |read difference| (many-to-one allowed), MATCH WINDOW 0.10,
  out-of-window rows named never adjudicating. THE RULE: an arm
  SEPARATES DECORRELATED iff BOTH primaries r(g-12|g0) and r(g0|g+12)
  carry dominant sign - with sign consistency >= 0.75 AND mean|d| >=
  0.15 over the in-window pairs, with n_pairs >= 3 (else
  UNDERPOWERED-SIDE, named; no bar fires on that arm; every-read-dead
  arms with 0 in-window pairs are named EXECUTED-DEAD). CO-REPORT
  SIDE := THE POOLED OWN-SIDE Z-RULER (x52's 34 committed
  walk-disjoint rows) with T335's CLASS CAVEAT carried verbatim
  ("part of the walk-vs-Z separation rides the WARM-VS-COLD class
  axis" — a cold-side-only separation is named CLASS-CARRIED and
  never adjudicates).

* THE VERDICT COMPOSITES (the bars made countable; frozen HERE).
  Per arm: arm_state := AT-STRENGTH if the primary-side verdict is
  DECORRELATED and geo_mad > 0.6 x walk_geo_mad; REDUCED if
  DECORRELATED with geo_mad <= 0.6 x walk_geo_mad (the 0.6 factor
  frozen by x36: 0.6 x 0.2597 = 0.1558 ~ the 0.15 floor);
  otherwise NOT:<verdict>. killed(arm) := the primary-side verdict is
  NOT DECORRELATED with power ok (the k4 death class).
  THE NON-FRONT FATE (bar 1):
    AT-STRENGTH          iff arm_state(tail4) == "AT-STRENGTH".
    DESTROYED-EQUALLY    iff killed(tail4) (the non-front swap kills
                          it too — any perturbation).
    ANOMALY              any uncovered shape, named: DECORRELATED-
                          but-REDUCED (consistency alive at reduced
                          magnitude — neither pole; tables verbatim)
                          or UNDERPOWERED/EXECUTED-DEAD (not
                          expected: the read face is dose-like).
    TEXTURE              any hard-gate failure (nothing adjudicated).
  THE RIDER FORK (bar 2; on the front's four pages; the reference
  kill = the committed k4 swap, frozen COHERENT — its in-cell
  re-derivation certified by G_K4REF, never re-adjudicated):
    CONTENT    iff (not killed(fshuf)) and (not killed(frjit))
               — "swap kills, reshuffle/re-jitter don't".
    CHANGE     iff killed(fshuf) and killed(frjit)
               — "all three kill".
    POSITION   iff killed(fshuf) and (not killed(frjit))
               — "swap + reshuffle kill, re-jitter doesn't".
    REALIZATION (named anomaly branch) iff (not killed(fshuf)) and
               killed(frjit) — positions changed harmlessly while the
               realizations kill; tables verbatim.
    ANOMALY    either rider arm UNDERPOWERED (the fork needs both
               rider arms powered; not expected — matched dose,
               matched reads).
    TEXTURE    any hard-gate failure.
  THE JOINT READING (co-reported, never adjudicating): the two
  composites' combination table (AT-STRENGTH+CONTENT = the seat
  grammar's clean front-content reading; DESTROYED+CHANGE = the
  knife-edge/trajectory-chaos reading, location-free; the mixed
  combos named honestly — including AT-STRENGTH+CHANGE, which would
  locate the fragile face in the FRONT while any change to it kills).

* THE FREE RIDER (|W|/in-room per arm; e346's cuts): at matched rungs
  {25, 125, 300}, the write norm and in-room fraction across ALL FIVE
  rows of the four-arm table + the walk pole (same substrate, same
  draws; the only delta is pool content) — WRITES-BLIND iff the max
  |W| spread <= 1.0 and the max in-room spread <= 0.05 at every
  matched rung (e346's W_SPREAD_MAX / ROOM_SPREAD_MAX verbatim); any
  breach named.

* NET0 CLASSES (x32's standing rule): every fresh state is
  "ZEPHYRA/install-end WARM/x58-<TAIL4|FRONT-RESHUFFLE|FRONT-REJITTER
  >-FORMING"; the committed rows carry their committed classes.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
family's own reason: CONTINUITY on the committed x45/x52/x53/x36
artifacts — the canon's own organism). CPU desk ONLY (torch threads
4; zero CUDA calls); 1800s slice caps with resume; timestamps
datetime.now(UTC) only.

Outputs: runs/x58/{metrics.json (PROGRESSIVE), REPORT.md,
x58_nonfront_swap.png}. The replay resumes live under runs/x58/ (*.pt
gitignored by the house convention, regenerable deterministically). No
NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).
Commit + push per phase (birth -> smoke -> run). Run: cd lab &&
python x58_nonfront_swap.py (X58_SMOKE=1 shakedown).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                              # noqa: E402
import scipy.fft as sf                          # noqa: E402
import torch                                    # noqa: E402
import torch.nn.functional as F                  # noqa: E402

import common                                   # noqa: E402
from common import CharCorpus, run_dir, save_json  # noqa: E402

import e043_install as E43                      # noqa: E402 (REPO,
                                                 # find_occ, SPLICE_RNG)
import g1b_continuity as GB                     # noqa: E402 — MUST be
                                                 # imported BEFORE G1
import g1_anchored_ball as G1                   # noqa: E402

torch.set_num_threads(4)           # the CPU desk lane (shared machine)

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

SMOKE = os.environ.get("X58_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "x58_smoke" if SMOKE else "x58"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


G1.log = log                                      # unify the timeline
REPO = common.REPO
CKPT_DIR = GB.CKPT_DIR

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
MENU_STEPS = 8 if SMOKE else 300           # the legs' span
CONS_SEED = G1.CONS_SEED                   # 10901, HELD (the canon stream)
FT_LR = G1.FT_LR                           # 1e-3 const (the canon's cons lr)
TRAIN_CAP_CPU = 1800.0                     # the parked-CPU slice cap

CAPTURE_STEPS = (1, 2, 4, 8) if SMOKE else \
    (1, 2, 4, 8, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300)

# ---- the page-position scheme (frozen at birth) -----------------------------
FRONT_ROWS = 4                             # the rider's front block (x36's
                                            # k4 swap site: hosts 0-3 @ j=-8)
TAIL_K = 4                                 # the non-front swap's dose
TAIL_START = 300 - TAIL_K                  # rows 296-299 = hosts 56-59 @+8
FRONT_PERM = (3, 2, 1, 0)                  # the committed reversal
                                            # derangement (max displacement)
REJIT_SEED = 58002                         # the re-jitter generator
REJIT_CHOICES = (-4, 0, 4, 8)              # the committed grid minus -8

# ---- the match + separation constants (x45/x52/x53/x36 verbatim) ------------
MATCH_WINDOW = 0.10
SIGN_FRAC = 0.75
FLOOR_CORR = 0.15
POWER_FLOOR_N = 3                          # the in-window pair floor
READ_BAR = 0.05                            # x40's aliveness bar (verbatim)
READ_TOL = 2e-6                            # the CPU read-determinism law
CORR_TOL = 1e-6                            # x53's corr tolerance
DEAD_CUT = 0.10                            # e346's fate cut (verbatim)

# ---- the verdict composite constants (frozen at birth) ---------------------
REDUCED_FRAC = 0.6                         # x36's "at-the-floor" clause
READ_HOLD = 0.40                           # P-x58a read-fate bar (all fresh)

# ---- the free rider's cuts (e346 verbatim) ---------------------------------
W_SPREAD_MAX = 1.0
ROOM_SPREAD_MAX = 0.05
RIDER_RUNGS = (25, 125, 300) if not SMOKE else (2, 4, 8)

# ---- the room (the census ruler; e311's conventions) -----------------------
ROOM_K = 10_000
ROOM_SEED_D = 26113
ROOM_SEED_S = 26114

# ---- the committed records + artifacts (md5-bound at birth; Rule 12) --------
MD5_BINDS = {
    "x45_metrics": ("runs/x45/metrics.json",
                    "1dfcfeb3840c9307dbbc48cd71859cc5"),
    "x52_metrics": ("runs/x52/metrics.json",
                    "bde14b44664b597520c2ca366e49b789"),
    "x53_metrics": ("runs/x53/metrics.json",
                    "77df6b087c86446bd379cfcb6def0aba"),
    "e346_metrics": ("runs/e346/metrics.json",
                     "7c6dc4051d46ece196486c47b764d4fa"),
    "e335_metrics": ("runs/e335/metrics.json",
                     "5d92359bc9b2e943c01bf521487cf177"),
    "x36_metrics": ("runs/x36/metrics.json",
                    "c44d5f366c5de0b6112870207ea65e46"),
    "substrate": ("runs/checkpoints/g1c_install_resume.pt",
                  "1f5a6e2b327a0731459cbb805fdc2505"),
    "e001": ("runs/checkpoints/e001.pt",
             "d114536d1c0983ab3be67f67ff0667c8"),
    "rooms264": ("runs/checkpoints/e264_rooms.pt",
                 "2d524655575cce00a3bc1c8770f4b211"),
    "rig_x53": ("lab/x53_own_name.py",
                "c9dbe6d5ab8f8ddca78e6cf334d2fcd5"),
    "rig_x45": ("lab/x45_path_difference.py",
                "88a22f2d8b2832538c93ba3d49570c23"),
    "rig_x52": ("lab/x52_ruler_symmetry.py",
                "4508ddb8ded0427c7f9249fd3f16566f"),
    "rig_x36": ("lab/x36_page_ladder.py",
                "9f3d7a9a905d63212dfdb80e7456a3c2"),
    "rig_g1c": ("lab/g1c_root_redraw.py",
                "b094295fd1d3a9e796de602d3b628344"),
    "rig_g1": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01"),
    "rig_e043": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa"),
    "rig_g1b": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a"),
    "rig_e346": ("lab/e346_name_key_ladder.py",
                 "808f89c94347d3d9d194cc268555dc2f"),
}

# the port's exact source lines (quoted; substring-gated at runtime)
PORT_QUOTES = [
    ("lab/g1c_root_redraw.py",
     "pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])",
     "the canon cons pool's committed construction order (jitter-major)"),
    ("lab/g1_anchored_ball.py",
     "ix = torch.randint(n_pool, (16,), generator=gen)",
     "the cons leg's pool draw (uniform; order-free by construction)"),
    ("lab/g1_anchored_ball.py",
     "gen = torch.Generator().manual_seed(CONS_SEED)",
     "the cons leg's generator (seed 10901)"),
    ("lab/g1c_root_redraw.py",
     "cons_anchor = anchor_full[:16]",
     "the cons anchor bank (first-16 originals; held canon every arm)"),
    ("lab/x53_own_name.py",
     "gpool = torch.Generator().manual_seed(VARIED_POOL_SEED)",
     "the neutral pool's generator (x53's pool_v, seed 53001)"),
    ("lab/x53_own_name.py",
     "s = int(torch.randint(POOL_LOW_S, hi, (1,), generator=gpool))",
     "the neutral pool's start draw (x53's line)"),
    ("lab/x53_own_name.py",
     "def paired_corr(x: list[float], y: list[float]):",
     "x45's paired-context correlation (x53's port)"),
    ("lab/x53_own_name.py",
     "ix = torch.randint(n_pool, (16,), generator=gen)",
     "the menu draw (x53's single-pool collapse of e346's per-slot form)"),
    ("lab/x45_path_difference.py",
     "alive_frac := frac of the 60 contexts with",
     "x40's aliveness bar convention (in x45's docstring)"),
]

# ======================================================================
# THE METRICS SHELL
# ======================================================================
metrics: dict = {
    "experiment": "x58 THE NON-FRONT SWAP + THE MATCHED-JITTER RIDER "
                  "(T338's named successor; P-T338a riding; R84's critic "
                  "control; P-x58a registered at birth)",
    "phase": "P0",
    "date": utcnow(),
    "status": "BIRTH (registration frozen; no compute yet)",
    "registered": {
        "bars_verbatim": {
            "AT-STRENGTH": "the non-front arm decorrelates — the front "
                           "specifically carries the consistency.",
            "DESTROYED-EQUALLY": "the non-front swap kills it too — any "
                                 "perturbation.",
            "RIDER_CONTENT": "content carries it (swap kills, "
                             "reshuffle/re-jitter don't)",
            "RIDER_CHANGE": "change carries it (all three kill)",
            "RIDER_POSITION": "position carries it (swap + reshuffle "
                              "kill, re-jitter doesn't)",
        },
        "parity_block": {
            "lab_lean_P-T338a": "AT-STRENGTH weakly — the front "
                                "specifically carries it (maps to RIDER: "
                                "CONTENT on the fork)",
            "counter_P-T338a": "DESTROYED-EQUALLY — the step shape hints "
                                "any perturbation (maps to RIDER: CHANGE "
                                "on the fork)",
            "position_leg": "no committed lean — named the uncovered "
                            "third",
        },
        "P_x58a": {
            "nonfront_call": "DESTROYED-EQUALLY, moderately (tail4 "
                             "lands the k4 death class: COHERENT at "
                             "walk-strength magnitudes)",
            "rider_call": "CHANGE (reshuffle AND re-jitter both kill — "
                          "the trajectory-chaos reading)",
            "runner_up": "the lab lean — AT-STRENGTH + CONTENT (the "
                         "seat-grammar reading)",
            "read_fate": "all three fresh arms HOLD (max read >= 0.40; "
                         "maintenance is dose-like)",
            "scored": ["nonfront == DESTROYED-EQUALLY",
                       "rider == CHANGE",
                       "max_read(tail4/fshuf/frjit) all >= 0.40",
                       "the committed k4 reference re-derives COHERENT "
                       "in-cell"],
        },
        "page_position_scheme_digest": "pages = the cons pool's 300 "
                                       "windows in the committed "
                                       "jitter-major order (x36's "
                                       "definition); arms: k4 = x36's "
                                       "committed front swap (runtime-"
                                       "read, certified, never rerun); "
                                       "tail4 = cat([pool_z[:296], "
                                       "pool_v[296:]]) (the mirrored "
                                       "non-front swap); fshuf = the "
                                       "canon with rows 0-3 reversed "
                                       "[3,2,1,0] (the committed "
                                       "derangement; content multiset "
                                       "EXACTLY canon); frjit = the "
                                       "canon with rows 0-3 re-jittered "
                                       "(seed 58002, uniform over "
                                       "{-4,0,+4,+8}; same hosts, same "
                                       "order, changed realizations; "
                                       "in-grid duplication disclosed); "
                                       "name-token count 7/window in "
                                       "every row of every pool; "
                                       "anchors + random corpus + draw "
                                       "stream HELD CANON every arm",
    },
    "smoke": SMOKE,
    "envelope": {
        "route": "CERTIFIED CPU-REPLAY (chosen + disclosed at birth): "
                 "CPU desk only, torch threads 4, zero CUDA calls; the "
                 "bit-identity gates (G_STREAMIDENT_CONS, G_K4REF, "
                 "G_WALKREPRO) require the deterministic CPU "
                 "realization; the sibling desk cell x57 shares the "
                 "desk; the GPU lane left free; phase-2 thermal "
                 "standing N/A on this route (no bursts; snapshot "
                 "recorded)",
        "slice_cap_s": TRAIN_CAP_CPU,
        "threads": 4,
        "slice_caps": [],
        "restarts": [],
    },
    "deviations": [],
    "builds_on": [
        "T338/P-T338a (x36's step-shaped early block; this cell's "
        "question + the riding prediction)",
        "R84's critic (the matched-jitter rider — the content-vs-change "
        "control this cell carries)",
        "x36 (the rig VERBATIM: the pools, the arms convention, the "
        "frozen separation rule, the verdict-composite machinery, the "
        "committed k4 leg this cell runtime-reads)",
        "x53 (pool_v, the frozen separation rule, the CPU-replay route)",
        "x45 (the instrument + the anneal ruler + the walk rungs)",
        "x52 (the pooled own-side Z-ruler + the class caveat)",
        "e346 (the fate cuts + the free rider's spread cuts)",
        "g1c_root_redraw (the canon: install seed 24314 + cons seed "
        "10901; the committed pool construction this cell pages)",
    ],
    "whats_new": [
        "THE FIRST NON-FRONT MANIPULATION OF THE AUTHORED STREAM: x36 "
        "only ever perturbed the pool's head; the mirrored tail swap "
        "at matched dose/draws/substrate splits front-concentration "
        "from any-perturbation fragility",
        "THE FIRST CONTENT-VS-CHANGE-VS-POSITION FORK: no committed "
        "cell ever held the content multiset fixed while varying "
        "positions (fshuf) or held content+positions while varying "
        "realizations (frjit) — the critic's control, now run",
        "the draw-identical four-arm table: one generator stream, four "
        "perturbation types (swap-front / swap-tail / reshuffle / "
        "re-jitter) + the committed walk pole, the ONLY delta pool "
        "content at the touched rows",
    ],
    "gates": {},
    "birth_commit": None,
    "git_head_final": None,
    "date_finished": None,
}

GATE_NAMES = ("G_MD5", "G_QUOTES", "G_NAMEFREE", "G_SPLICE", "G_BATTERY",
              "G_X58POOLS", "G_ROOM", "G_INSTRUMENT",
              "G_STREAMIDENT_CONS", "G_K4REF", "G_WALKREPRO")


# ======================================================================
# HELPERS (x53's forms, verbatim)
# ======================================================================
def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def load_model_sd(path: Path) -> dict:
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    return {k: v.detach().clone() for k, v in sd.items()}


# ---- x45's per-context reader (ported VERBATIM via x53) --------------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30
                    ) -> torch.Tensor:
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
    return torch.cat(pzs)


# ---- x45's paired elicitation correlations (ported VERBATIM) ----------------
def paired_corr(x: list[float], y: list[float]):
    mx, my = sum(x) / 60, sum(y) / 60
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(60))
    vx = math.sqrt(sum((v - mx) ** 2 for v in x))
    vy = math.sqrt(sum((v - my) ** 2 for v in y))
    return cov / (vx * vy) if vx > 0 and vy > 0 else None


# ---- e261's SRCT room (ported VERBATIM; x53's copy) ------------------------
class SRCT:
    def __init__(self, n: int, k: int, seed_d: int, seed_s: int):
        assert 0 < k <= n
        self.n, self.k = int(n), int(k)
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        self.S = torch.randperm(n, generator=g2)[:self.k].sort().values \
            .numpy()
        self.mask = np.zeros(n, dtype=np.float64)
        self.mask[self.S] = 1.0
        self.seed_d, self.seed_s = seed_d, seed_s

    def project(self, x64: np.ndarray) -> np.ndarray:
        c = sf.dct(self.D * x64, type=2, norm="ortho", workers=4)
        c *= self.mask
        return self.D * sf.idct(c, type=2, norm="ortho", workers=4)


# ======================================================================
# THE INSTRUMENT (x45's row form; single Z channel — disclosed)
# ======================================================================
BATTERIES = (("g-12", "gm12_ids"), ("g0", "g0_ids"), ("g+12", "gp12_ids"))
PRIMARY = ("r_g-12|g0", "r_g0|g+12")


def instrument(p0: dict, theta_sd: dict, tag: str, name_id: int,
               base_flat64: np.ndarray) -> dict:
    net = G1.evl_load(theta_sd)
    net.eval()
    row: dict = {"tag": tag}
    percell = {}
    alive, means = {}, {}
    for bn, bk in BATTERIES:
        pv = battery_percell(net, p0[bk], name_id)
        percell[bn] = [float(v) for v in pv]
        alive[bn] = float((pv >= READ_BAR).float().mean())
        means[bn] = float(pv.mean())
        ref = G1.battery_cell(net, p0[bk], name_id)["mean_pz"]
        assert abs(means[bn] - ref) <= 1e-9, \
            f"percell reader drift {bn} {tag}: {means[bn]} vs {ref}"
    row["read"] = means["g0"]
    row["means"] = means
    row["alive"] = alive
    row["corrs"] = {
        f"{b1}|{b2}": paired_corr(percell[b1], percell[b2])
        for b1, b2 in (("g-12", "g0"), ("g0", "g+12"), ("g-12", "g+12"))}
    row["ce_r"] = G1.ce_fixed_cpu(net, *p0["r_eval_xy"])
    flat64 = flat_params_cpu(net).double().numpy()
    d = flat64 - base_flat64
    dn = float(np.linalg.norm(d))
    row["write_norm"] = dn
    row["in_room"] = (float(np.linalg.norm(p0["room"].project(d)) / dn)
                      if dn > 0 else None)
    del net
    return row


def compare_row(mine: dict, ref: dict) -> tuple[dict, bool]:
    """x45's row-comparison form (x53's copy)."""
    cell, ok = {}, True
    for bn in ("g-12", "g0", "g+12"):
        cell[f"mean:{bn}"] = {
            "mine": mine["means"][bn], "committed": ref["means"][bn],
            "abs_diff": abs(mine["means"][bn] - ref["means"][bn])}
        ok &= cell[f"mean:{bn}"]["abs_diff"] <= READ_TOL
        cell[f"alive:{bn}"] = {
            "mine": mine["alive"][bn], "committed": ref["alive"][bn],
            "abs_diff": abs(mine["alive"][bn] - ref["alive"][bn])}
        ok &= cell[f"alive:{bn}"]["abs_diff"] == 0.0
    for pair in ("g-12|g0", "g0|g+12", "g-12|g+12"):
        cell[f"corr:{pair}"] = {
            "mine": mine["corrs"][pair], "committed": ref["corrs"][pair],
            "abs_diff": abs(mine["corrs"][pair] - ref["corrs"][pair])}
        ok &= cell[f"corr:{pair}"]["abs_diff"] <= CORR_TOL
    cell["pass"] = bool(ok)
    return cell, bool(ok)


def norm_row(tag: str, read: float, corrs: dict, alive: dict = None,
             extra: dict = None) -> dict:
    r = {"tag": tag, "read": float(read),
         "corrs": {f"r_{k}": (None if corrs.get(k) is None
                              else float(corrs[k]))
                   for k in ("g-12|g0", "g0|g+12", "g-12|g+12")}}
    if alive is not None:
        r["alive"] = {k: float(v) for k, v in alive.items()}
    if extra:
        r.update(extra)
    return r


# ======================================================================
# THE FROZEN RULE (x45's match + separation, x52-certified, x36's copy)
# ======================================================================
def match_pairs(arm_rows: list[dict], ref_rows: list[dict]):
    in_win, oow = [], []
    for ar in arm_rows:
        best = min(ref_rows, key=lambda rr: abs(rr["read"] - ar["read"]))
        d_read = abs(best["read"] - ar["read"])
        pair = {"arm_tag": ar["tag"], "arm_read": ar["read"],
                "ref_tag": best["tag"], "ref_read": best["read"],
                "abs_d_read": d_read, "in_window": bool(d_read <=
                                                        MATCH_WINDOW)}
        for key in PRIMARY:
            av, rv = ar["corrs"].get(key), best["corrs"].get(key)
            pair[key] = {"arm": av, "ref": rv,
                         "d": (None if av is None or rv is None
                               else av - rv)}
        (in_win if pair["in_window"] else oow).append(pair)
    return in_win, oow


def score_side(in_win: list[dict]) -> dict:
    scored = {}
    for key in PRIMARY:
        ds = [p[key]["d"] for p in in_win if p[key]["d"] is not None]
        n_none = sum(1 for p in in_win if p[key]["d"] is None)
        if not ds:
            scored[key] = {"n_pairs": 0, "n_none": n_none,
                           "separates": False, "note": "no valid pairs"}
            continue
        n_pos = sum(1 for d in ds if d > 0)
        n_neg = sum(1 for d in ds if d < 0)
        dominant = "+" if n_pos >= n_neg else "-"
        sign = max(n_pos, n_neg) / len(ds)
        mad = sum(abs(d) for d in ds) / len(ds)
        scored[key] = {
            "n_pairs": len(ds), "n_none": n_none,
            "dominant_sign": dominant,
            "sign_consistency": round(sign, 4),
            "mean_abs_d": round(mad, 6),
            "floor": FLOOR_CORR, "floor_kind": "absolute 0.15 (1/sqrt(59))",
            "separates": bool(sign >= SIGN_FRAC and mad >= FLOOR_CORR),
        }
    n_pairs = max((scored[k]["n_pairs"] for k in PRIMARY), default=0)
    both = (all(scored[k].get("separates") and
                scored[k]["dominant_sign"] == "-"
                for k in PRIMARY) and n_pairs >= POWER_FLOOR_N)
    return {"scored": scored, "n_in_window": len(in_win),
            "decorrelates": bool(both),
            "power": ("ok" if n_pairs >= POWER_FLOOR_N else
                      f"UNDERPOWERED (n={n_pairs}<{POWER_FLOOR_N})")}


def score_arm(arm_rows: list[dict], rulers: dict) -> dict:
    sides = {}
    for rname, rrows in rulers.items():
        in_win, oow = match_pairs(arm_rows, rrows)
        sc = score_side(in_win)
        sc["out_of_window"] = [{"arm_tag": p["arm_tag"],
                                "arm_read": p["arm_read"],
                                "ref_tag": p["ref_tag"],
                                "abs_d_read": p["abs_d_read"]}
                               for p in oow]
        sides[rname] = sc
    primary = sides.get("anneal")
    reads = [r["read"] for r in arm_rows]
    dead_all = bool(reads) and max(reads) <= DEAD_CUT
    if primary is None:
        verdict = "UNDERPOWERED (no primary side)"
    elif primary["power"] != "ok":
        if primary["n_in_window"] == 0 and dead_all:
            verdict = ("EXECUTED-DEAD (every arm read <= 0.10; no "
                       "in-window pairs — the menu failed to even "
                       "maintain the read; decorrelation vacuous)")
        else:
            verdict = primary["power"]
    else:
        verdict = "DECORRELATED" if primary["decorrelates"] else "COHERENT"
    cold_sep = any(s["decorrelates"] for k, s in sides.items()
                   if k.startswith("pooled"))
    return {"sides": sides, "verdict": verdict,
            "max_read": max(reads) if reads else None,
            "last_read": reads[-1] if reads else None,
            "dead_all": dead_all,
            "class_carried_cold_side": bool(cold_sep and
                                            verdict != "DECORRELATED")}


def geo_mad(side: dict) -> float | None:
    m1 = side["scored"][PRIMARY[0]].get("mean_abs_d")
    m2 = side["scored"][PRIMARY[1]].get("mean_abs_d")
    if not m1 or not m2:
        return None
    return math.sqrt(m1 * m2)


# ======================================================================
# THE REPLAY CORE (CPU; the committed arithmetic; instrumented rungs)
# ======================================================================
def menu_draw(p0: dict, pool_x, pool_mask):
    """The cons stream's per-step draw on ONE pool (x53's form)."""
    n_pool = pool_x.shape[0]
    cons_anchor = p0["cons_anchor"]
    train_ids = p0["train_ids"]

    def draw(gen, dev):
        ix = torch.randint(n_pool, (16,), generator=gen)
        aj = torch.randint(cons_anchor.shape[0], (8,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                           generator=gen)
        nw = pool_x[ix]
        nm = pool_mask[ix]
        anc = torch.cat([cons_anchor[aj],
                         torch.stack([train_ids[s: s + G1.BLOCK]
                                      for s in rj])], 0)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0).to(dev)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0).to(dev)
        m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
        m[:16] = nm.to(dev)
        return x, y, m, (ix.tolist(), aj.tolist(), rj.tolist())
    return draw


def replay_menu_leg(tag: str, start_sd: dict, steps: int, seed: int,
                    p0: dict, pool_x, pool_mask, rungs: set,
                    resume_ck: Path, host_id: int,
                    net0_class: str, e001_flat64: np.ndarray) -> dict:
    """One deterministic CPU menu leg (the cons arithmetic on one pool)
    with instrumented rungs (single Z channel)."""
    state = {"step": 0, "draws": [], "rows": {}}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED at step {state['step']}/{steps} "
            f"({len(state['rows'])} rows restored)")
        metrics["envelope"]["restarts"].append(
            {"tag": tag, "resumed_at_step": state["step"], "t": utcnow()})
    if int(state.get("step", 0)) >= steps:
        log(f"  [{tag}] resume ckpt already finished at s{state['step']}")
        return {"rows": state["rows"], "draws": state["draws"],
                "steps_ran": steps}

    net = G1.evl_load(start_sd).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    if resume_ck.exists():
        net.load_state_dict(state["model"])
        net.train()
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen_state"])
    draw = menu_draw(p0, pool_x, pool_mask)
    step = state["step"]
    rows: dict = state["rows"]
    draws: list = state["draws"]
    t_slice = time.time()
    n_caps = 0

    def rung_row(step_: int) -> None:
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
        row = instrument(p0, sd_cpu, f"{tag}_s{step_}", host_id,
                         e001_flat64)
        row.update({"step": step_, "provenance": "replay (CPU desk)",
                    "net0_class": net0_class, "menu": tag})
        rows[f"{tag}_s{step_}"] = row
        log(f"  [{tag}] s{step_:4d} READ {row['read']:.6f} | r1 "
            f"{row['corrs']['g-12|g0']} r2 {row['corrs']['g0|g+12']} | "
            f"|W| {row['write_norm']:.2f} inroom {row['in_room']:.3f} | "
            f"CE_R {row['ce_r']:.4f}")

    def save_resume() -> None:
        torch.save({"model": {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "draws": draws, "rows": rows},
                   resume_ck)

    while step < steps:
        step += 1
        x, y, m, dr = draw(gen, CPU)
        draws.append({"step": step, "ix": dr[0], "aj": dr[1], "rj": dr[2]})
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in rungs:
            rung_row(step)
        if (time.time() - t_slice) > TRAIN_CAP_CPU:
            save_resume()
            n_caps += 1
            metrics["envelope"]["slice_caps"].append(
                {"tag": tag, "step": step, "t": utcnow()})
            log(f"  [{tag}] slice cap at s{step} — resume saved, "
                "continuing (CPU park convention)")
            t_slice = time.time()
    save_resume()
    log(f"  [{tag}] leg finished: {steps} steps, {n_caps} slice caps, "
        f"{len(rows)} instrument rows")
    del net, opt
    return {"rows": rows, "draws": draws, "steps_ran": step}


# ---- the pure-generator cons draw sequence (no training needed) -----------
def cons_draw_sequence(n_steps: int, n_train: int) -> list[dict]:
    gen = torch.Generator().manual_seed(CONS_SEED)
    n_pool, n_anc = 300, 16
    seq = []
    for _ in range(n_steps):
        ix = torch.randint(n_pool, (16,), generator=gen)
        aj = torch.randint(n_anc, (8,), generator=gen)
        rj = torch.randint(n_train - G1.BLOCK - 1, (8,), generator=gen)
        seq.append({"ix": ix.tolist(), "aj": aj.tolist(),
                    "rj": rj.tolist()})
    return seq


# ======================================================================
# P0 — THE BINDS + THE BANK + THE POOLS + THE FOUR ARMS + THE ROOM + RECORDS
# ======================================================================
def phase_P0() -> dict:
    log("P0 — THE BINDS + the bank + the canon pool + the neutral pool + "
        "the four arms + the room + the committed records")
    metrics["envelope"]["gpu_snapshot_at_start"] = common.gpu_status()

    # ---- G_MD5 --------------------------------------------------------
    binds = {}
    for key, (rel, bound) in MD5_BINDS.items():
        p = REPO / rel
        ok = p.exists() and md5of(p) == bound
        binds[key] = {"path": rel,
                      "md5": (md5of(p) if p.exists() else None),
                      "bound_md5": bound, "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": "the canon stream records (x45/x52/x53/e346/e335/"
                      "x36), the substrate, the ruler sources, the "
                      "instrument parents, x36's rig and every port-"
                      "source rig md5-bound at run time (frozen at "
                      "birth)",
             "pass": bool(all(b["pass"] for b in binds.values()))}
    assert g_md5["pass"], \
        f"G_MD5 FAILED: {[k for k, b in binds.items() if not b['pass']]}"
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)} binds")

    # ---- G_QUOTES -----------------------------------------------------
    quotes = []
    for src, frag, why in PORT_QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": "every ported line is a verbatim substring of its "
                    "committed source rig",
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    assert g_q["pass"], f"G_QUOTES FAILED: {g_q}"
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim")

    # ---- the corpus + the bank -----------------------------------------
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    name_ids_z = corpus.encode(G1.NAME)
    assert len(name_ids_z) == 7

    g_namefree = {"zeph": train_text.count("ZEPH"),
                  "pass": bool(train_text.count("ZEPH") == 0)}
    assert g_namefree["pass"], f"G_NAMEFREE FAILED: {g_namefree}"
    metrics["gates"]["G_NAMEFREE"] = g_namefree

    import random
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
    g_splice = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert g_splice["pass"], f"G_SPLICE FAILED: {g_splice}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    g_battery = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert g_battery["pass"], f"G_BATTERY FAILED: {g_battery}"
    metrics["gates"].update({"G_SPLICE": g_splice, "G_BATTERY": g_battery})

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])   # (60, 256)
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)

    # ---- the single-window builder (the canon's committed construction,
    #      x36's jittered_pool refactored one-window-at-a-time; bit-faithful
    #      by G_X58POOLS's bit-equality vs the committed block form) ------
    def host_window(p: int, host: str, j: int):
        pre = train_ids[p - G1.PRE - j: p]
        post = train_ids[p + len(host): p + len(host) + G1.POST_CAP - j]
        w = torch.cat([pre, name_ids_z, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"window len {len(w)} at j={j}")
        m = torch.zeros(G1.BLOCK - 1, dtype=torch.bool)
        m[G1.PRE - 1 + j: G1.PRE - 1 + j + len(name_ids_z)] = True
        return w, m

    # ---- the canon pool (the committed jittered pool) -------------------
    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins, jmasks = [], []
        for p, h in install_occ:
            w, m = host_window(p, h, j)
            jwins.append(w)
            jmasks.append(m)
        jit_x[j] = torch.stack(jwins)
        jit_mask[j] = torch.stack(jmasks)
    pool_z_x = torch.cat([jit_x[j] for j in G1.JITTERS])
    pool_z_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]       # e113: first-16 originals

    # ---- the neutral pool (x53's pool_v VERBATIM) ------------------------
    gpool = torch.Generator().manual_seed(53001)   # x53's VARIED_POOL_SEED
    hi = len(train_ids) - G1.BLOCK - 1
    used_starts, vw, vm = set(), [], []
    for j in G1.JITTERS:
        for _ in range(60):
            while True:
                s = int(torch.randint(280, hi, (1,), generator=gpool))
                if s not in used_starts:
                    used_starts.add(s)
                    break
            pre = train_ids[s: s + G1.PRE + j]
            post = train_ids[s + G1.PRE + j + 7:
                             s + G1.PRE + j + 7 + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids_z, post])
            assert len(w) == G1.BLOCK
            vw.append(w)
            m = torch.zeros(G1.BLOCK - 1, dtype=torch.bool)
            m[G1.PRE - 1 + j: G1.PRE - 1 + j + 7] = True
            vm.append(m)
    pool_v_x, pool_v_mask = torch.stack(vw), torch.stack(vm)

    # ---- THE FOUR-ARM POOLS (the page-position scheme, frozen) -----------
    # k4 (the committed front swap — rebuilt for the 8-step bit-proof;
    #      its adjudication rows are runtime-read from x36's ckpt)
    k4_px = torch.cat([pool_v_x[:FRONT_ROWS], pool_z_x[FRONT_ROWS:]])
    k4_pm = torch.cat([pool_v_mask[:FRONT_ROWS], pool_z_mask[FRONT_ROWS:]])
    # tail4 (THE NON-FRONT SWAP — the mirrored design)
    tail4_px = torch.cat([pool_z_x[:TAIL_START], pool_v_x[TAIL_START:]])
    tail4_pm = torch.cat([pool_z_mask[:TAIL_START],
                          pool_v_mask[TAIL_START:]])
    # fshuf (FRONT-RESHUFFLE — the committed reversal derangement)
    perm = torch.tensor(FRONT_PERM, dtype=torch.long)
    fshuf_px = torch.cat([pool_z_x[perm], pool_z_x[FRONT_ROWS:]])
    fshuf_pm = torch.cat([pool_z_mask[perm], pool_z_mask[FRONT_ROWS:]])
    # frjit (FRONT-RE-JITTER — redrawn realizations, seed 58002)
    rj_gen = torch.Generator().manual_seed(REJIT_SEED)
    rejit_draws = [REJIT_CHOICES[int(torch.randint(len(REJIT_CHOICES),
                                                  (1,), generator=rj_gen))]
                   for _ in range(FRONT_ROWS)]
    frjit_rows = [host_window(*install_occ[i], j)
                  for i, j in enumerate(rejit_draws)]
    frjit_px = torch.cat([torch.stack([w for w, _ in frjit_rows]),
                          pool_z_x[FRONT_ROWS:]])
    frjit_pm = torch.cat([torch.stack([m for _, m in frjit_rows]),
                          pool_z_mask[FRONT_ROWS:]])

    arms_pools = {
        "k4": (k4_px, k4_pm), "tail4": (tail4_px, tail4_pm),
        "fshuf": (fshuf_px, fshuf_pm), "frjit": (frjit_px, frjit_pm),
    }

    # ---- G_X58POOLS ---------------------------------------------------
    z_prefixes = set()
    for i in range(pool_z_x.shape[0]):
        j = G1.JITTERS[i // 60]
        z_prefixes.add(pool_z_x[i, : G1.PRE + j].numpy().tobytes())
    v_disjoint = sum(
        1 for i in range(pool_v_x.shape[0])
        if pool_v_x[i, : G1.PRE + G1.JITTERS[i // 60]].numpy().tobytes()
        not in z_prefixes)
    masks_v_eq = bool(torch.equal(pool_v_mask, pool_z_mask))
    # the in-grid re-jitter duplication map (disclosed consequence)
    block_of = {j: bi for bi, j in enumerate(G1.JITTERS)}
    rejit_dup = [{"page": i, "host": install_occ[i][1],
                  "pos": install_occ[i][0],
                  "old_j": -8, "new_j": rejit_draws[i],
                  "duplicated_canon_row": block_of[rejit_draws[i]] * 60 + i,
                  "unrehearsed_canon_row": i,
                  "row_bit_equal_dup": bool(torch.equal(
                      frjit_px[i],
                      pool_z_x[block_of[rejit_draws[i]] * 60 + i]))}
                 for i in range(FRONT_ROWS)]
    arm_checks = {
        "k4": {
            "head_bit_equal_pool_v": bool(torch.equal(
                k4_px[:FRONT_ROWS], pool_v_x[:FRONT_ROWS])),
            "tail_bit_equal_canon": bool(torch.equal(
                k4_px[FRONT_ROWS:], pool_z_x[FRONT_ROWS:])),
            "mask_bit_equal_canon": bool(torch.equal(k4_pm, pool_z_mask)),
            "name_token_count": int(k4_pm.sum())},
        "tail4": {
            "head_bit_equal_canon": bool(torch.equal(
                tail4_px[:TAIL_START], pool_z_x[:TAIL_START])),
            "tail_bit_equal_pool_v": bool(torch.equal(
                tail4_px[TAIL_START:], pool_v_x[TAIL_START:])),
            "mask_bit_equal_canon": bool(torch.equal(tail4_pm,
                                                     pool_z_mask)),
            "name_token_count": int(tail4_pm.sum())},
        "fshuf": {
            "perm": list(FRONT_PERM),
            "derangement_no_fixed_point": bool(all(
                FRONT_PERM[i] != i for i in range(FRONT_ROWS))),
            "front_rows_bit_equal_canon_perm": bool(torch.equal(
                fshuf_px[:FRONT_ROWS], pool_z_x[list(FRONT_PERM)])),
            "front_is_perm_of_canon_front": bool(sorted(
                FRONT_PERM) == list(range(FRONT_ROWS))),
            "tail_bit_equal_canon": bool(torch.equal(
                fshuf_px[FRONT_ROWS:], pool_z_x[FRONT_ROWS:])),
            "mask_bit_equal_canon": bool(torch.equal(fshuf_pm,
                                                     pool_z_mask)),
            "name_token_count": int(fshuf_pm.sum()),
            "content_multiset_exactly_canon": bool(
                torch.equal(fshuf_px[:FRONT_ROWS],
                            pool_z_x[sorted(FRONT_PERM)])
                and torch.equal(fshuf_px[FRONT_ROWS:],
                                pool_z_x[FRONT_ROWS:]))},
        "frjit": {
            "seed": REJIT_SEED, "choices": list(REJIT_CHOICES),
            "realized_jitters": rejit_draws,
            "all_changed_from_-8": bool(all(j != -8 for j in rejit_draws)),
            # content preservation proven BY the duplication identity:
            # row i == the canon row of host i at its new jitter
            "row_differs_from_canon": [bool(not torch.equal(
                frjit_px[i], pool_z_x[i])) for i in range(FRONT_ROWS)],
            "front_masks_match_new_jitter": [bool(torch.equal(
                frjit_pm[i], pool_z_mask[d["duplicated_canon_row"]]))
                for d in rejit_dup],
            "duplication_map": rejit_dup,
            "tail_bit_equal_canon": bool(torch.equal(
                frjit_px[FRONT_ROWS:], pool_z_x[FRONT_ROWS:])),
            "name_token_count": int(frjit_pm.sum())},
    }
    # the name sits right after the first mask column in every sampled
    # row of every pool (jitter-matched column per row)
    def name_in_place(px, pm, idxs) -> bool:
        return all(
            bool(torch.equal(px[i, int(pm[i].nonzero()[0]) + 1:
                                int(pm[i].nonzero()[0]) + 8],
                             name_ids_z))
            for i in idxs)
    sampled_idx = [0, 1, 2, 3, 60, 150, 240, 296, 297, 298, 299]
    name_ok = all(name_in_place(px, pm, sampled_idx)
                  for px, pm in [(pool_z_x, pool_z_mask),
                                 (pool_v_x, pool_v_mask),
                                 (k4_px, k4_pm), (tail4_px, tail4_pm),
                                 (fshuf_px, fshuf_pm),
                                 (frjit_px, frjit_pm)])
    g_lp = {
        "pool_z": {"shape": list(pool_z_x.shape),
                   "mask_sum": int(pool_z_mask.sum()),
                   "order": "jitter-major [j=-8 h0-59 | j=-4 h0-59 | "
                            "j=0 | j=+4 | j=+8] (the committed "
                            "construction order)"},
        "pool_v": {"shape": list(pool_v_x.shape),
                   "mask_sum": int(pool_v_mask.sum()),
                   "seed": 53001,
                   "n_unique_starts": len(used_starts),
                   "n_context_disjoint_from_cons": v_disjoint},
        "arms": arm_checks,
        "masks_v_bit_equal_z": masks_v_eq,
        "name_in_place_sampled": name_ok,
        "claim": ("the canon pool rebuilt bit-faithfully (300 windows, "
                  "2100 name tokens, the committed jitter-major order); "
                  "pool_v = x53's committed neutral pool class (300 "
                  "unique starts, masks bit-equal to the canon's, "
                  "contexts disjoint); k4 == x36's committed front swap "
                  "pool (head bit-equal pool_v); tail4 = the mirrored "
                  "tail swap (rows 296-299 = hosts 56-59@+8 -> pool_v's "
                  "tail; head bit-equal canon); fshuf = the canon with "
                  "rows 0-3 reversed [3,2,1,0] (every window bit-equal "
                  "a canon window; content multiset EXACTLY canon; "
                  "derangement verified); frjit = the canon with rows "
                  "0-3 re-jittered (seed 58002 over {-4,0,+4,+8}; hosts "
                  "0-3 at rows 0-3, changed realizations; each re-"
                  "jittered row bit-equal the canon row of that host at "
                  "its new jitter — the in-grid duplication DISCLOSED "
                  "and mapped; name-token count 7/window in every row "
                  "of every pool => the name dose bit-matched per "
                  "window/step in COUNT; the name COLUMN of frjit rows "
                  "0-3 shifts with the jitter — inherent to a changed "
                  "realization, disclosed)"),
        "pass": bool(list(pool_z_x.shape) == [300, G1.BLOCK]
                     and int(pool_z_mask.sum()) == 2100
                     and list(pool_v_x.shape) == [300, G1.BLOCK]
                     and int(pool_v_mask.sum()) == 2100
                     and len(used_starts) == 300
                     and v_disjoint >= 290 and masks_v_eq and name_ok
                     and arm_checks["k4"]["head_bit_equal_pool_v"]
                     and arm_checks["k4"]["tail_bit_equal_canon"]
                     and arm_checks["k4"]["mask_bit_equal_canon"]
                     and arm_checks["k4"]["name_token_count"] == 2100
                     and arm_checks["tail4"]["head_bit_equal_canon"]
                     and arm_checks["tail4"]["tail_bit_equal_pool_v"]
                     and arm_checks["tail4"]["mask_bit_equal_canon"]
                     and arm_checks["tail4"]["name_token_count"] == 2100
                     and arm_checks["fshuf"]["derangement_no_fixed_point"]
                     and arm_checks["fshuf"]["front_rows_bit_equal_"
                                             "canon_perm"]
                     and arm_checks["fshuf"]["tail_bit_equal_canon"]
                     and arm_checks["fshuf"]["mask_bit_equal_canon"]
                     and arm_checks["fshuf"]["name_token_count"] == 2100
                     and arm_checks["frjit"]["all_changed_from_-8"]
                     and all(arm_checks["frjit"]["row_differs_from_"
                                               "canon"])
                     and all(arm_checks["frjit"]["front_masks_match_"
                                               "new_jitter"])
                     and all(d["row_bit_equal_dup"] for d in rejit_dup)
                     and arm_checks["frjit"]["tail_bit_equal_canon"]
                     and arm_checks["frjit"]["name_token_count"] == 2100),
    }
    assert g_lp["pass"], f"G_X58POOLS FAILED: {g_lp}"
    metrics["gates"]["G_X58POOLS"] = g_lp
    log(f"G_X58POOLS PASS: z/v 300x{G1.BLOCK}, v {v_disjoint}/300 "
        f"disjoint; rejit jitters {rejit_draws}; four arms bit-exact")

    # ---- the room ------------------------------------------------------
    room = SRCT(GB.G1B_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / "e264_rooms.pt", map_location="cpu",
                          weights_only=False)
    D264 = np.asarray(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = np.asarray(rooms264["model"]["K10K"]["S"])
    x = np.random.default_rng(26115).standard_normal(GB.G1B_PARAMS)
    px_ = room.project(x)
    idem = float(np.linalg.norm(room.project(px_) - px_)
                 / np.linalg.norm(px_))
    g_room = {"k": room.k, "seeds": [room.seed_d, room.seed_s],
              "D_bit_equal": bool(np.array_equal(room.D, D264)),
              "S_bit_equal": bool(np.array_equal(room.S, S264)),
              "idempotence_rel": idem,
              "pass": bool(np.array_equal(room.D, D264)
                           and np.array_equal(room.S, S264)
                           and idem < 1e-10)}
    assert g_room["pass"], f"G_ROOM FAILED: {g_room}"
    metrics["gates"]["G_ROOM"] = g_room
    log(f"G_ROOM PASS: K10K bit-bound (idempotence {idem:.1e})")
    del rooms264, D264, S264, x, px_

    # ---- the committed records (runtime-read; never retyped) -----------
    x45m = json.loads((REPO / "runs" / "x45" / "metrics.json")
                      .read_text(encoding="utf-8"))
    x52m = json.loads((REPO / "runs" / "x52" / "metrics.json")
                      .read_text(encoding="utf-8"))
    x53m = json.loads((REPO / "runs" / "x53" / "metrics.json")
                      .read_text(encoding="utf-8"))
    e346m = json.loads((REPO / "runs" / "e346" / "metrics.json")
                       .read_text(encoding="utf-8"))
    e335m = json.loads((REPO / "runs" / "e335" / "metrics.json")
                       .read_text(encoding="utf-8"))
    x36m = json.loads((REPO / "runs" / "x36" / "metrics.json")
                      .read_text(encoding="utf-8"))
    ws = {s["key"]: s for s in e335m["cut1_walk"]["states"]}
    committed = {
        "x45_walk_rungs": x45m["walk"]["rungs"],
        "x45_anneal_panels": x45m["anneal"]["panels"],
        "x45_separations": x45m["adjudication"]["separations"],
        "x52_pool_rows": x52m["pool"]["rows"],
        "x52_rescore": x52m["rescore"],
        "x52_rider_verdict": x52m["rider"]["adjudication"],
        "x53_varied_rows": {t: r for t, r in x53m["arm_rows"].items()
                            if t.startswith("varied_s")},
        "e346_base_row": e346m["arm_rows"]["base"],
        "e346_own_install_row": e346m["arm_rows"]["own_install"],
        "e335_anchors": {
            "base_read": ws["base"]["read"],
            "own_install_read": ws["own_install"]["read"],
            "root_read": ws["root"]["read"],
            "root_gm12": ws["root"]["battery"]["g-12"]["mean_pz"],
        },
        "x36_k4_arm_scores": x36m["arm_scores"]["k4"],
        "x36_k4_arm_state": x36m["arm_states"]["k4"],
        "x36_k4_ktable_row": [r for r in x36m["ktable"]
                              if r["arm"] == "k4"][0],
    }
    log("P0: the committed records runtime-read (x45 rungs "
        f"{len(committed['x45_walk_rungs'])} + panels "
        f"{len(committed['x45_anneal_panels'])}; x52 pool "
        f"{len(committed['x52_pool_rows'])}; x53 varied "
        f"{len(committed['x53_varied_rows'])}; x36 k4 state "
        "runtime-read)")
    write_partial("P0 the binds + the bank + the pools + the four arms "
                  "+ the room")
    return {
        "corpus": corpus, "stoi": stoi, "itos": itos, "zid": zid,
        "train_ids": train_ids,
        "bat_ids": bat_ids, "g0_ids": bat_ids[0], "gm12_ids": bat_ids[-12],
        "gp12_ids": bat_ids[12], "anchor_full": anchor_full,
        "r_eval_xy": (r_eval_x, r_eval_y), "room": room,
        "pool_z_x": pool_z_x, "pool_z_mask": pool_z_mask,
        "pool_v_x": pool_v_x, "pool_v_mask": pool_v_mask,
        "arms_pools": arms_pools, "rejit_draws": rejit_draws,
        "cons_anchor": cons_anchor, "committed": committed,
    }


# ======================================================================
# P1 — THE CERTIFICATIONS (instrument + canon stream + the k4 reference
#      + walk repro)
# ======================================================================
def phase_P1(p0: dict) -> dict:
    log("P1 — THE CERTIFICATIONS (G_INSTRUMENT + G_STREAMIDENT_CONS + "
        "G_K4REF + G_WALKREPRO)")
    C = p0["committed"]
    e001_sd = load_model_sd(CKPT_DIR / "e001.pt")
    e001_flat64 = flat_params_cpu(G1.evl_load(e001_sd)).double().numpy()
    assert e001_flat64.shape[0] == GB.G1B_PARAMS
    sub_sd = load_model_sd(CKPT_DIR / "g1c_install_resume.pt")

    # ---- G_INSTRUMENT (Z-channel cells; the T channel has no referent --
    #      in this cell — disclosed) ----------------------------------------
    cells = {}
    r = instrument(p0, e001_sd, "e001_z", p0["zid"], e001_flat64)
    c, ok_z = compare_row(r, C["x45_walk_rungs"]["base"])
    cells["e001_on_Z_vs_x45_base"] = c
    r = instrument(p0, sub_sd, "own_install", p0["zid"], e001_flat64)
    c2, ok_o = compare_row(r, C["e346_own_install_row"])
    cells["own_install_on_Z_vs_e346"] = c2
    root_sd = load_model_sd(CKPT_DIR / "g1c_root.pt")
    r = instrument(p0, root_sd, "root", p0["zid"], e001_flat64)
    cells["root_on_Z"] = {
        "mine": r["read"], "committed": C["e335_anchors"]["root_read"],
        "abs_diff": abs(r["read"] - C["e335_anchors"]["root_read"])}
    cells["root_gm12"] = {
        "mine": r["means"]["g-12"],
        "committed": C["e335_anchors"]["root_gm12"],
        "abs_diff": abs(r["means"]["g-12"]
                        - C["e335_anchors"]["root_gm12"])}
    ok_r = cells["root_on_Z"]["abs_diff"] <= READ_TOL \
        and cells["root_gm12"]["abs_diff"] <= READ_TOL
    g_instr = {"cells": cells,
               "disclosure": "single-channel (Z) cells only — no foreign "
                             "name exists in any x58 stream; x53's "
                             "dual-channel certification stands for the "
                             "ported reader itself",
               "claim": ("the ported instrument reproduces the committed "
                         "rows on the Z channel (e001 vs x45's base row; "
                         "own_install vs e346's committed row; root vs "
                         "the e335 literals) — tol 2e-6 means, 1e-6 "
                         "corrs, alive exact"),
               "pass": bool(ok_z and ok_o and ok_r)}
    assert g_instr["pass"], f"G_INSTRUMENT FAILED: {g_instr}"
    metrics["gates"]["G_INSTRUMENT"] = g_instr
    log("G_INSTRUMENT PASS: Z channel certified (3 cells)")
    del root_sd

    # ---- G_STREAMIDENT_CONS (the canon stream certification) ------------
    probe_steps = 25
    net = G1.evl_load(sub_sd).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(CONS_SEED)
    draw = menu_draw(p0, p0["pool_z_x"], p0["pool_z_mask"])
    for _ in range(probe_steps):
        x, y, m, _ = draw(gen, CPU)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    sd25 = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net, opt
    my_row = instrument(p0, sd25, "probe_cons25", p0["zid"], e001_flat64)
    cell, ok = compare_row(my_row, C["x45_walk_rungs"]["walk_cons_s25"])
    g_sc = {"probe": {"steps": probe_steps,
                      "start": "g1c_install_resume.pt", "seed": CONS_SEED,
                      "pool": "z (the canon's own)"},
            "row": cell,
            "claim": ("my menu draw path on the canon pool reproduces "
                      "x45's committed walk_cons_s25 instrument row "
                      "BIT-EXACTLY — this rig == the committed walk's "
                      "stream; the shared draw stream certified for "
                      "every arm"),
            "pass": bool(ok)}
    assert g_sc["pass"], f"G_STREAMIDENT_CONS FAILED: {g_sc}"
    metrics["gates"]["G_STREAMIDENT_CONS"] = g_sc
    log(f"G_STREAMIDENT_CONS PASS: s25 read |d| "
        f"{cell['mean:g0']['abs_diff']:.1e}")
    del sd25, my_row

    # ---- G_K4REF (the committed front-swap reference, never rerun) -----
    # (i) load x36's committed k4 leg record (rows + draws)
    x36_resume = torch.load(REPO / "runs" / "x36" / "leg_k4_resume.pt",
                            map_location="cpu", weights_only=False)
    x36_k4_rows = x36_resume["rows"]
    x36_k4_draws = x36_resume["draws"]
    # (ii) the 300-step draw record == my canon draw sequence (the
    #      shared stream: every arm in this cell draws identically)
    my_draws = cons_draw_sequence(300, len(p0["train_ids"]))

    def draw_fields_eq(a: dict, b: dict) -> bool:
        return all(a[k] == b[k] for k in ("ix", "aj", "rj"))
    draws_eq = (len(x36_k4_draws) == 300 and len(my_draws) == 300
                and all(draw_fields_eq(a, b)
                        for a, b in zip(my_draws, x36_k4_draws)))
    # (iii) the 8-step weight-level bit proof on MY rebuilt k4 pool
    net = G1.evl_load(sub_sd).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(CONS_SEED)
    draw = menu_draw(p0, *p0["arms_pools"]["k4"])
    for _ in range(8):
        x, y, m, _ = draw(gen, CPU)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    sd8 = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net, opt
    my_row = instrument(p0, sd8, "probe_k4_8", p0["zid"], e001_flat64)
    cell8, ok8 = compare_row(my_row, x36_k4_rows["k4_s8"])
    bit_exact = all(cell8[f"mean:{bn}"]["abs_diff"] == 0.0
                    for bn in ("g-12", "g0", "g+12"))
    g_k4 = {
        "draws": {"mine_steps": len(my_draws),
                  "x36_steps": len(x36_k4_draws),
                  "bit_equal_all_300": bool(draws_eq)},
        "probe8": {"row": cell8, "bit_exact_means": bool(bit_exact)},
        "claim": ("the front-swap reference IS x36's committed k4 leg: "
                  "its 300-step draw record is bit-equal to my canon "
                  "sequence (the shared stream) AND the 8-step "
                  "weight-level replay on my rebuilt k4 pool reproduces "
                  "x36's committed k4_s8 row BIT-EXACTLY — the reference "
                  "kill is reused, never rerun (extend-don't-repeat); "
                  "its in-cell re-derivation below must reproduce x36's "
                  "committed scores"),
        "pass": bool(draws_eq and ok8 and bit_exact),
    }
    assert g_k4["pass"], f"G_K4REF FAILED: {g_k4}"
    metrics["gates"]["G_K4REF"] = g_k4
    log(f"G_K4REF PASS: k4 draws bit-equal; s8 bit-exact")
    del sd8, my_row, x36_resume
    write_partial("P1 the certifications (instrument + canon + k4 ref)")

    # ---- the rulers ------------------------------------------------------
    anneal_ruler = [norm_row(tag, r["read"], r["corrs"], r.get("alive"),
                             {"src": "x45 panels"})
                    for tag, r in C["x45_anneal_panels"].items()]
    pooled_ruler = [norm_row(r["tag"], r["read"], r["corrs"],
                             r.get("alive"), {"src": r.get("src"),
                                              "strata": r.get("strata")})
                    for r in C["x52_pool_rows"]]
    log(f"  rulers: anneal {len(anneal_ruler)} rows (reads "
        f"{min(r['read'] for r in anneal_ruler):.3f}.."
        f"{max(r['read'] for r in anneal_ruler):.3f}); pooled-Z "
        f"{len(pooled_ruler)} rows")

    # ---- G_K4REF's re-derivation half (my pairing on the committed k4
    #      rows must reproduce x36's committed scores — the reference
    #      kill's in-cell re-derivation, never re-adjudicated) -----------
    rulers_ = {"anneal": anneal_ruler, "pooled_Z": pooled_ruler}
    k4_rows = [norm_row(t, r["read"], r["corrs"], r.get("alive"),
                        {"net0_class": r.get("net0_class"),
                         "write_norm": r.get("write_norm"),
                         "in_room": r.get("in_room")})
               for t, r in sorted(x36_k4_rows.items(),
                                  key=lambda kv: kv[1]["step"])]
    k4_score = score_arm(k4_rows, rulers_)
    committed_k4 = C["x36_k4_arm_scores"]["sides"]["anneal"]["scored"]
    k4_cross = {"verdict_committed": C["x36_k4_arm_scores"]["verdict"],
                "verdict_mine": k4_score["verdict"],
                "geo_mad_committed": C["x36_k4_arm_state"]["geo_mad"],
                "geo_mad_mine": geo_mad(k4_score["sides"]["anneal"]),
                "r1": {"mine_consistency":
                       k4_score["sides"]["anneal"]["scored"]
                       ["r_g-12|g0"]["sign_consistency"],
                       "committed": committed_k4["r_g-12|g0"][
                           "sign_consistency"],
                       "mine_mad": k4_score["sides"]["anneal"]["scored"]
                       ["r_g-12|g0"]["mean_abs_d"],
                       "committed_mad": committed_k4["r_g-12|g0"][
                           "mean_abs_d"]},
                "r2": {"mine_consistency":
                       k4_score["sides"]["anneal"]["scored"]
                       ["r_g0|g+12"]["sign_consistency"],
                       "committed": committed_k4["r_g0|g+12"][
                           "sign_consistency"],
                       "mine_mad": k4_score["sides"]["anneal"]["scored"]
                       ["r_g0|g+12"]["mean_abs_d"],
                       "committed_mad": committed_k4["r_g0|g+12"][
                           "mean_abs_d"]}}
    k4_cross["pass"] = bool(
        k4_cross["verdict_mine"] == k4_cross["verdict_committed"]
        and k4_cross["geo_mad_mine"] is not None
        and abs(k4_cross["geo_mad_mine"] - k4_cross["geo_mad_committed"])
        <= 0.02
        and abs(k4_cross["r1"]["mine_consistency"]
                - k4_cross["r1"]["committed"]) <= 0.02
        and abs(k4_cross["r2"]["mine_consistency"]
                - k4_cross["r2"]["committed"]) <= 0.02)
    metrics["gates"]["G_K4REF"]["re_derivation"] = k4_cross
    assert k4_cross["pass"], f"G_K4REF re-derivation FAILED: {k4_cross}"
    log("G_K4REF re-derivation PASS: my pairing reproduces x36's "
        "committed k4 scores (verdict + geo + consistencies)")

    # ---- G_WALKREPRO (the committed walk under this cell's pairing) -----
    walk_rows = [norm_row(tag, r["read"], r["corrs"])
                 for tag, r in C["x45_walk_rungs"].items()
                 if tag != "base"]
    ann_side = score_side(match_pairs(walk_rows, anneal_ruler)[0])
    wr_ = C["x45_walk_rungs"]
    own_rows = [norm_row(f"walk_cons_s{s}", wr_[f"walk_cons_s{s}"]["read"],
                         wr_[f"walk_cons_s{s}"]["corrs"])
                for s in range(25, 300, 25)]
    own_rows.append(norm_row("walk_s700", wr_["walk_s700"]["read"],
                             wr_["walk_s700"]["corrs"]))
    pooled_side = score_side(match_pairs(own_rows, pooled_ruler)[0])
    x45c = C["x45_separations"]
    x52c = C["x52_rescore"]["walk"]["scored"]
    repro = {
        "anneal_side": ann_side["scored"],
        "pooled_side": pooled_side["scored"],
        "committed": {
            "x45_anneal": {k: {"sign_consistency":
                               x45c[k]["sign_consistency"],
                               "mean_abs_d": x45c[k]["mean_abs_d"]}
                           for k in ("r_g-12|g0", "r_g0|g+12")},
            "x52_pooled": {k: {"sign_consistency":
                               x52c[k]["sign_consistency"],
                               "mean_abs_d": x52c[k]["mean_abs_d"]}
                           for k in ("r_g-12|g0", "r_g0|g+12")},
        },
    }
    checks = []
    for k in ("r_g-12|g0", "r_g0|g+12"):
        for side, ref in (("anneal_side", repro["committed"]
                           ["x45_anneal"]),
                          ("pooled_side", repro["committed"]
                           ["x52_pooled"])):
            m_ = repro[side][k]
            checks.append({
                "side": side, "obs": k,
                "mine_consistency": m_["sign_consistency"],
                "committed": ref[k]["sign_consistency"],
                "mine_mean_abs_d": m_["mean_abs_d"],
                "committed_mad": ref[k]["mean_abs_d"],
                "pass": bool(abs(m_["sign_consistency"]
                                 - ref[k]["sign_consistency"]) <= 0.02
                             and abs(m_["mean_abs_d"]
                                     - ref[k]["mean_abs_d"]) <= 0.02
                             and m_["separates"])})
    g_wr = {"repro": repro, "checks": checks,
            "claim": ("this cell's own pairing machinery re-derives the "
                      "committed walk separation on BOTH rulers (x45's "
                      "founding anneal-side + x52's own-side pooled "
                      "rescore): consistencies and mean|d| within 0.02, "
                      "separation booleans exact — the K=0 pole's "
                      "decorrelation, the reference every arm is read "
                      "against"),
            "pass": bool(all(c["pass"] for c in checks))}
    assert g_wr["pass"], f"G_WALKREPRO FAILED: {g_wr}"
    metrics["gates"]["G_WALKREPRO"] = g_wr
    log("G_WALKREPRO PASS: the walk's separation reproduces on both "
        "rulers inside this cell's pairing")
    return {"anneal_ruler": anneal_ruler, "pooled_ruler": pooled_ruler,
            "walk_rows": walk_rows, "own_rows": own_rows,
            "e001_flat64": e001_flat64, "sub_sd": sub_sd,
            "x36_k4_rows": x36_k4_rows}


# ======================================================================
# P2 — THE FRESH LEGS (tail4 / fshuf / frjit; CPU)
# ======================================================================
def phase_P2(p0: dict, p1: dict) -> dict:
    log("P2 — THE FRESH LEGS (tail4 the non-front swap; fshuf the front "
        "reshuffle; frjit the front re-jitter; the cons arithmetic on "
        "the arm pools; single Z channel)")
    e001_flat64 = p1["e001_flat64"]
    sub_sd = p1["sub_sd"]
    legs = {}
    for tag, cls in (("tail4", "TAIL4"), ("fshuf", "FRONT-RESHUFFLE"),
                     ("frjit", "FRONT-REJITTER")):
        px, pm = p0["arms_pools"][tag]
        legs[tag] = replay_menu_leg(
            tag, sub_sd, MENU_STEPS, CONS_SEED, p0, px, pm,
            set(CAPTURE_STEPS), RD / f"leg_{tag}_resume.pt", p0["zid"],
            f"ZEPHYRA/install-end WARM/x58-{cls}-FORMING", e001_flat64)
        write_partial(f"P2 the {tag} leg finished")
    return {"legs": legs}


# ======================================================================
# P3 — THE READS + THE TWO COMPOSITES + THE FREE RIDER + THE REPORT
# ======================================================================
def phase_P3(p0: dict, p1: dict, p2: dict) -> dict:
    log("P3 — THE READS + the two verdict composites + the free rider + "
        "the report")
    C = p0["committed"]
    rulers = {"anneal": p1["anneal_ruler"], "pooled_Z": p1["pooled_ruler"]}

    def fresh_rows(tag: str) -> list[dict]:
        return [norm_row(t, r["read"], r["corrs"], r.get("alive"),
                         {"net0_class": r.get("net0_class"),
                          "write_norm": r.get("write_norm"),
                          "in_room": r.get("in_room")})
                for t, r in sorted(p2["legs"][tag]["rows"].items(),
                                   key=lambda kv: kv[1]["step"])]

    def committed_rows(rows: dict) -> list[dict]:
        return [norm_row(t, r["read"], r["corrs"], r.get("alive"),
                         {"net0_class": r.get("net0_class"),
                          "write_norm": r.get("write_norm"),
                          "in_room": r.get("in_room")})
                for t, r in sorted(rows.items(),
                                   key=lambda kv: kv[1]["step"])]

    k4_rows = committed_rows(p1["x36_k4_rows"])
    k0_rows = p1["walk_rows"]

    arms = {}
    for tag in ("tail4", "fshuf", "frjit"):
        arms[tag] = score_arm(fresh_rows(tag), rulers)
    arms["k4_ref"] = score_arm(k4_rows, rulers)
    arms["k0_walk"] = score_arm(k0_rows, rulers)

    # ---- the strength profile + arm_state --------------------------------
    x45c = C["x45_separations"]
    m1 = x45c["r_g-12|g0"]["mean_abs_d"]
    m2 = x45c["r_g0|g+12"]["mean_abs_d"]
    walk_geo = math.sqrt(m1 * m2)
    reduced_cut = REDUCED_FRAC * walk_geo

    def arm_state(score: dict) -> dict:
        side = score["sides"]["anneal"]
        g = geo_mad(side)
        rho = (g / walk_geo) if g else None
        st = {"verdict": score["verdict"], "geo_mad": g,
              "rho_vs_walk": rho, "reduced_cut": reduced_cut,
              "state": None,
              "r1": side["scored"][PRIMARY[0]],
              "r2": side["scored"][PRIMARY[1]]}
        if score["verdict"] == "DECORRELATED":
            st["state"] = ("AT-STRENGTH" if (g is not None
                                             and g > reduced_cut)
                           else "REDUCED")
        else:
            st["state"] = f"NOT:{score['verdict']}"
        return st

    states = {tag: arm_state(arms[tag])
              for tag in ("tail4", "fshuf", "frjit", "k4_ref", "k0_walk")}

    def killed(tag: str) -> bool:
        """The k4 death class: NOT-DECORRELATED with power ok."""
        sc, st = arms[tag], states[tag]
        return (sc["verdict"] != "DECORRELATED"
                and sc["sides"]["anneal"]["power"] == "ok")

    def powered(tag: str) -> bool:
        return arms[tag]["sides"]["anneal"]["power"] == "ok"

    # ---- BAR 1: THE NON-FRONT FATE (frozen at birth) ----------------------
    t4v, t4s = arms["tail4"]["verdict"], states["tail4"]["state"]
    if t4s == "AT-STRENGTH":
        verdict_nonfront = (
            "AT-STRENGTH (the non-front arm decorrelates — the front "
            "specifically carries the consistency; the tail swap at "
            "matched dose/draws leaves the walk-class decorrelation "
            "intact)")
    elif killed("tail4"):
        verdict_nonfront = (
            "DESTROYED-EQUALLY (the non-front swap kills it too — any "
            f"perturbation; the tail4 arm lands the k4 death class: "
            f"{t4v}, state {t4s})")
    elif t4v == "DECORRELATED":            # REDUCED — the named middle
        verdict_nonfront = (
            "ANOMALY (DECORRELATED-but-REDUCED: the tail arm's "
            "consistency survives at reduced magnitude — neither the "
            "front-carries nor the any-perturbation pole; tables "
            "verbatim; nothing adjudicated)")
    else:                                   # UNDERPOWERED / EXECUTED-DEAD
        verdict_nonfront = (
            f"ANOMALY (tail4 {t4v} — not expected: the read face is "
            "dose-like; named, not adjudicated)")

    # ---- BAR 2: THE RIDER FORK (frozen at birth) --------------------------
    k4_ref_coherent = arms["k4_ref"]["verdict"] == "COHERENT"
    rider_anom = None
    if not (powered("fshuf") and powered("frjit")):
        verdict_rider = ("ANOMALY (a rider arm UNDERPOWERED — the fork "
                         "needs both rider arms powered; named, not "
                         "adjudicated)")
    elif k4_ref_coherent is False:
        verdict_rider = ("ANOMALY (the committed k4 reference failed to "
                         "re-derive COHERENT in-cell — G_K4REF's "
                         "re-derivation cross-check below; nothing "
                         "adjudicated)")
        rider_anom = "k4-ref"
    else:
        ks_f, ks_j = killed("fshuf"), killed("frjit")
        if not ks_f and not ks_j:
            verdict_rider = ("CONTENT (the authored content carries it — "
                             "the swap kills, reshuffle/re-jitter don't; "
                             "the fork's first leg)")
        elif ks_f and ks_j:
            verdict_rider = ("CHANGE (any change kills — all three "
                             "kill: swap + reshuffle + re-jitter; the "
                             "fork's second leg)")
        elif ks_f and not ks_j:
            verdict_rider = ("POSITION (the positions carry it — swap + "
                             "reshuffle kill, re-jitter doesn't; the "
                             "fork's third leg)")
        else:
            verdict_rider = ("REALIZATION (named anomaly branch: the "
                             "reshuffle survives while the re-jitter "
                             "kills — positions changed harmlessly, the "
                             "realizations carry it; tables verbatim)")
            rider_anom = "realization"

    # ---- THE JOINT READING (co-reported, never adjudicating) -------------
    nf_key = ("AT-STRENGTH" if verdict_nonfront.startswith("AT-STRENGTH")
              else ("DESTROYED-EQUALLY"
                    if verdict_nonfront.startswith("DESTROYED")
                    else "ANOMALY"))
    rd_key = verdict_rider.split(" (")[0]
    joint_map = {
        ("AT-STRENGTH", "CONTENT"): "front-concentration, content-"
            "carried — the seat grammar's clean reading",
        ("AT-STRENGTH", "POSITION"): "front-concentration, position-"
            "carried — the front's page PLACEMENT feeds the consistency",
        ("AT-STRENGTH", "CHANGE"): "the fragile face is front-LOCALIZED "
            "while any change to it kills — a located knife-edge",
        ("AT-STRENGTH", "REALIZATION"): "front-localized, realization-"
            "carried — named honest (the tables decide)",
        ("DESTROYED-EQUALLY", "CHANGE"): "location-free knife-edge — "
            "any perturbation anywhere kills (the trajectory-chaos "
            "reading)",
        ("DESTROYED-EQUALLY", "CONTENT"): "MIXED: the tail swap kills "
            "while front reshuffle/re-jitter survive — neither reading "
            "clean; the tail block's specificity named for a successor",
        ("DESTROYED-EQUALLY", "POSITION"): "MIXED: the tail swap kills "
            "while the front reshuffle+re-jitter pattern is positional "
            "— the four tables verbatim",
        ("DESTROYED-EQUALLY", "REALIZATION"): "MIXED — the tables "
            "verbatim; named for a successor",
    }
    joint_reading = joint_map.get((nf_key, rd_key),
                                  "uncovered combo — tables verbatim")

    # ---- P-x58a scoring ---------------------------------------------------
    max_read = {tag: arms[tag]["max_read"]
                for tag in ("tail4", "fshuf", "frjit")}
    p_score = {
        "nonfront_DESTROYED": verdict_nonfront.startswith(
            "DESTROYED-EQUALLY"),
        "rider_CHANGE": verdict_rider.startswith("CHANGE"),
        "reads_hold": bool(all(v is not None and v >= READ_HOLD
                               for v in max_read.values())),
        "k4_ref_coherent": k4_ref_coherent,
    }
    p_score["n_hit"] = sum(bool(v) for v in p_score.values())

    # ---- THE FREE RIDER (|W| / in-room; e346's cuts) ----------------------
    rider_rows = {}
    for tag in ("tail4", "fshuf", "frjit"):
        for t, r in p2["legs"][tag]["rows"].items():
            rider_rows[(tag, r["step"])] = (r["write_norm"], r["in_room"])
    for t, r in p1["x36_k4_rows"].items():
        rider_rows[("k4", r["step"])] = (r["write_norm"], r["in_room"])
    for t, r in C["x45_walk_rungs"].items():
        if t.startswith("walk_cons_s") or t == "walk_s700":
            step = r["step"] if "step" in r else int(t.split("s")[-1])
            rider_rows[("k0", step)] = (r["write_norm"], r["in_room"])
    rider_table, spreads = {}, {}
    for rung in RIDER_RUNGS:
        entry = {}
        for arm in ("k0", "k4", "tail4", "fshuf", "frjit"):
            v = rider_rows.get((arm, rung))
            if v is not None:
                entry[arm] = {"write_norm": v[0], "in_room": v[1]}
        rider_table[str(rung)] = entry
        all_vals = list(entry.values())
        if len(all_vals) >= 2:
            ws = [v["write_norm"] for v in all_vals]
            rs = [v["in_room"] for v in all_vals]
            spreads[str(rung)] = {
                "w_spread": max(ws) - min(ws),
                "room_spread": max(rs) - min(rs),
                "w_ok": (max(ws) - min(ws)) <= W_SPREAD_MAX,
                "room_ok": (max(rs) - min(rs)) <= ROOM_SPREAD_MAX}
    writes_blind = bool(spreads) and all(
        s["w_ok"] and s["room_ok"] for s in spreads.values())

    # ---- the four-arm table ----------------------------------------------
    arm_labels = (("k0_walk", "k0 (the walk pole)", 0,
                   "committed (x45/x52 runtime-read)"),
                  ("k4_ref", "k4 (the front swap)", 4,
                   "committed (x36 runtime-read; G_K4REF-certified)"),
                  ("tail4", "tail4 (the non-front swap)", 4,
                   "fresh CPU replay"),
                  ("fshuf", "fshuf (front reshuffle)", 4,
                   "fresh CPU replay"),
                  ("frjit", "frjit (front re-jitter)", 4,
                   "fresh CPU replay"))
    ktable = []
    for key, label, dose, prov in arm_labels:
        sc, st = arms[key], states[key]
        ktable.append({
            "arm": label, "key": key, "dose_pages": dose,
            "provenance": prov,
            "max_read": sc["max_read"], "last_read": sc["last_read"],
            "dead_all": sc["dead_all"],
            "verdict": sc["verdict"], "state": st["state"],
            "geo_mad": st["geo_mad"], "rho_vs_walk": st["rho_vs_walk"],
            "n_in_window_anneal": sc["sides"]["anneal"]["n_in_window"],
            "r1": {k: st["r1"].get(k) for k in ("sign_consistency",
                                               "mean_abs_d",
                                               "dominant_sign")},
            "r2": {k: st["r2"].get(k) for k in ("sign_consistency",
                                               "mean_abs_d",
                                               "dominant_sign")},
            "pooled_side_decorrelates":
                sc["sides"]["pooled_Z"]["decorrelates"],
            "class_carried_cold_side": sc["class_carried_cold_side"],
        })

    metrics["rulers"] = {
        "anneal_rows": len(p1["anneal_ruler"]),
        "pooled_rows": len(p1["pooled_ruler"]),
        "walk_geo_mad_committed": walk_geo,
        "reduced_cut": reduced_cut,
        "reduced_cut_rationale": "0.6 x walk_geo_mad ~ the 0.15 floor's "
                                 "edge (x36's frozen clause, carried)",
    }
    metrics["arm_scores"] = arms
    metrics["arm_states"] = states
    metrics["ktable"] = ktable
    metrics["verdict_nonfront"] = verdict_nonfront
    metrics["verdict_rider"] = verdict_rider
    metrics["joint_reading"] = joint_reading
    metrics["p_x58a_score"] = p_score
    metrics["free_rider"] = {
        "table": rider_table, "spreads": spreads,
        "cuts": {"W_SPREAD_MAX": W_SPREAD_MAX,
                 "ROOM_SPREAD_MAX": ROOM_SPREAD_MAX},
        "writes_blind": writes_blind,
        "claim": "the write census (|W|, in-room) across the walk pole "
                 "and the four perturbation arms at matched rungs — "
                 "same substrate, same draws; the name-blindness "
                 "control standing (the census tracks stream "
                 "arithmetic, not page content)",
    }
    metrics["rejit_realized"] = {
        "seed": REJIT_SEED, "choices": list(REJIT_CHOICES),
        "jitters": p0["rejit_draws"],
        "front_perm": list(FRONT_PERM),
    }
    metrics["net0_classes"] = (
        "fresh states: ZEPHYRA/install-end WARM/x58-TAIL4-FORMING / "
        "x58-FRONT-RESHUFFLE-FORMING / x58-FRONT-REJITTER-FORMING; "
        "committed rows carry their committed classes")
    metrics["status"] = ("COMPLETE: nonfront=" + verdict_nonfront
                         + " | rider=" + verdict_rider)
    metrics["git_head_final"] = git_head()
    metrics["date_finished"] = utcnow()
    save_json(RD / "metrics.json", metrics)

    # ---- the PNG ----------------------------------------------------------
    try:
        make_png(p2, p1, ktable, verdict_nonfront, verdict_rider,
                 walk_geo, reduced_cut, rider_table, p_score)
        log("PNG written: runs/x58/x58_nonfront_swap.png")
    except Exception as e:                                  # noqa: BLE001
        metrics["deviations"].append(f"PNG writer error (caught, "
                                     f"post-verdict): {e!r}")
        save_json(RD / "metrics.json", metrics)
        log(f"PNG writer error (caught): {e!r}")

    # ---- the REPORT -------------------------------------------------------
    make_report(ktable, verdict_nonfront, verdict_rider, joint_reading,
                p_score, writes_blind, walk_geo, reduced_cut,
                rider_table, spreads, p0["rejit_draws"])
    log("REPORT written: runs/x58/REPORT.md")
    return {"verdict_nonfront": verdict_nonfront,
            "verdict_rider": verdict_rider}


# ======================================================================
# THE PNG + THE REPORT
# ======================================================================
def make_png(p2: dict, p1: dict, ktable: list, verdict_nonfront: str,
             verdict_rider: str, walk_geo: float, reduced_cut: float,
             rider_table: dict, p_score: dict) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("x58 — THE NON-FRONT SWAP + THE MATCHED-JITTER RIDER "
                 "(front-concentration vs any-perturbation; P-T338a)",
                 fontsize=11)

    # (1) the read trajectories per arm
    ax = axs[0][0]
    for tag in ("tail4", "fshuf", "frjit"):
        rows = sorted(p2["legs"][tag]["rows"].values(),
                      key=lambda r: r["step"])
        ax.plot([r["step"] for r in rows], [r["read"] for r in rows],
                marker="o", ms=3, label=tag)
    k4rows = sorted(p1["x36_k4_rows"].values(), key=lambda r: r["step"])
    ax.plot([r["step"] for r in k4rows], [r["read"] for r in k4rows],
            marker="x", ms=4, ls="--", lw=1, color="gray",
            label="k4 (committed)")
    ax.axhline(DEAD_CUT, color="gray", ls=":", lw=1)
    ax.axhline(READ_HOLD, color="green", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("cons step"); ax.set_ylabel("host read p(Z) g0 (log)")
    ax.set_title("THE READS: maintenance per arm (dose-like face)")
    ax.legend(fontsize=8)

    # (2) the consistency-vs-magnitude profile per arm
    ax = axs[0][1]
    labels = [row["arm"].split(" (")[0] for row in ktable]
    geos = [row["geo_mad"] if row["geo_mad"] is not None else 0.0
            for row in ktable]
    cols = ["tab:blue" if row["state"] == "AT-STRENGTH"
            else ("tab:orange" if row["state"] == "REDUCED" else "tab:red")
            for row in ktable]
    ax.bar(labels, geos, color=cols)
    ax.axhline(FLOOR_CORR, color="k", ls="--", lw=1,
               label=f"floor {FLOOR_CORR}")
    ax.axhline(walk_geo, color="tab:blue", ls=":", lw=1,
               label=f"walk geo {walk_geo:.3f}")
    ax.axhline(reduced_cut, color="tab:orange", ls=":", lw=1,
               label=f"reduced cut {reduced_cut:.3f}")
    for i, row in enumerate(ktable):
        r1 = row["r1"].get("sign_consistency")
        r2 = row["r2"].get("sign_consistency")
        ax.text(i, 0.02, f"r1 {r1}\nr2 {r2}", ha="center", fontsize=7)
    ax.set_xlabel("arm"); ax.set_ylabel("geo_mad (anneal)")
    ax.set_title("THE DECORRELATION PROFILE (magnitude) + the "
                 "consistencies (text)")
    ax.legend(fontsize=7)

    # (3) the free rider: |W| + in_room at the last rider rung
    ax = axs[1][0]
    last_rung = rider_table.get(str(max(int(k) for k in rider_table))
                                ) if rider_table else {}
    arms_order = [row["key"] for row in ktable]
    wvals = [last_rung.get({"k0_walk": "k0", "k4_ref": "k4"}.get(a, a),
                           {}).get("write_norm", float("nan"))
             for a in arms_order]
    rvals = [last_rung.get({"k0_walk": "k0", "k4_ref": "k4"}.get(a, a),
                           {}).get("in_room", float("nan"))
             for a in arms_order]
    ax2 = ax.twinx()
    ax.bar([row["arm"].split(" (")[0] for row in ktable], wvals,
           color="tab:gray", alpha=0.7, label="|W|")
    ax2.plot([row["arm"].split(" (")[0] for row in ktable], rvals,
             color="tab:red", marker="s", label="in_room")
    ax.set_ylabel("|W| (fp64 vs e001)"); ax2.set_ylabel("K10K in-room")
    ax.set_title("THE FREE RIDER: the name-blind census per arm "
                 "(last matched rung)")

    # (4) the verdicts + the four-arm table
    ax = axs[1][1]
    ax.axis("off")
    lines = [f"NON-FRONT: {verdict_nonfront}", "",
             f"RIDER: {verdict_rider}", ""]
    for row in ktable:
        lines.append(f"{row['arm']:<28} {row['state']:<24} "
                     f"max_read {row['max_read'] if row['max_read'] is None else round(row['max_read'], 3)}")
    lines += ["", f"walk_geo_mad {walk_geo:.4f} | reduced_cut "
              f"{reduced_cut:.4f} (0.6x, x36's clause)",
              f"P-x58a: {p_score.get('n_hit', '?')}/4 scored calls hit"]
    ax.text(0.02, 0.98, "\n".join(lines), fontsize=8, va="top",
            family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RD / "x58_nonfront_swap.png", dpi=140)
    plt.close(fig)


def make_report(ktable, verdict_nonfront, verdict_rider, joint_reading,
                p_score, writes_blind, walk_geo, reduced_cut,
                rider_table, spreads, rejit_draws) -> None:
    L = []
    L.append("# x58 — THE NON-FRONT SWAP + THE MATCHED-JITTER RIDER "
             "(T338's named successor; P-T338a riding; R84's critic "
             "control)")
    L.append("")
    L.append(f"Verdict (bar 1, the non-front fate): **{verdict_nonfront}**")
    L.append("")
    L.append(f"Verdict (bar 2, the rider fork): **{verdict_rider}**")
    L.append("")
    L.append(f"JOINT READING (co-reported, never adjudicating): "
             f"{joint_reading}.")
    L.append("")
    L.append("x36 found the early block as a STEP (k4/k16/k64 "
             "indistinguishable: swapping the front 1.3% destroys the "
             "sign consistency at walk-strength magnitudes). This cell "
             "splits front-concentration from any-perturbation "
             "fragility: the MIRRORED tail swap (the last 4 pages, "
             "hosts 56-59@+8, matched dose/draws/substrate) + the "
             "critic's matched-jitter rider on the same front block "
             "x36's k4 swap replaced (reshuffle: same content, changed "
             "positions; re-jitter: same content and order, changed "
             "realizations — seed 58002 over {-4,0,+4,+8}, realized "
             f"{rejit_draws}).")
    L.append("")
    L.append("## The four-arm table (+ the walk pole; the host fate + "
             "the decorrelation profile per arm)")
    L.append("")
    L.append("| arm | pages touched | max read | verdict | state | "
             "geo_mad | rho | n_in_win | r1 cons/mad | r2 cons/mad | "
             "pooled side |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in ktable:
        L.append(
            f"| {row['arm']} | {row['dose_pages']} "
            f"| {row['max_read'] if row['max_read'] is None else round(row['max_read'], 4)} "
            f"| {row['verdict']} | {row['state']} "
            f"| {row['geo_mad'] if row['geo_mad'] is None else round(row['geo_mad'], 4)} "
            f"| {row['rho_vs_walk'] if row['rho_vs_walk'] is None else round(row['rho_vs_walk'], 3)} "
            f"| {row['n_in_window_anneal']} "
            f"| {row['r1'].get('sign_consistency')}/{row['r1'].get('mean_abs_d')} "
            f"| {row['r2'].get('sign_consistency')}/{row['r2'].get('mean_abs_d')} "
            f"| {'SEP' if row['pooled_side_decorrelates'] else '-'}"
            + (" (CLASS-CARRIED)" if row["class_carried_cold_side"]
               else "") + " |")
    L.append("")
    L.append(f"walk_geo_mad (committed, runtime-read) = {walk_geo:.4f}; "
             f"REDUCED cut = 0.6 x walk = {reduced_cut:.4f} (x36's "
             "at-the-floor clause, carried frozen).")
    L.append("")
    L.append("## P-x58a (registered at birth, BEFORE compute)")
    L.append("")
    L.append(f"- non-front call DESTROYED-EQUALLY: "
             f"**{p_score['nonfront_DESTROYED']}**")
    L.append(f"- rider call CHANGE: **{p_score['rider_CHANGE']}**")
    L.append(f"- read-fate all three fresh arms HOLD (>= {READ_HOLD}): "
             f"**{p_score['reads_hold']}**")
    L.append(f"- the committed k4 reference re-derives COHERENT: "
             f"**{p_score['k4_ref_coherent']}**")
    L.append(f"- scored {p_score['n_hit']}/4")
    L.append("")
    L.append("Parity: LAB LEAN (P-T338a) AT-STRENGTH weakly — the front "
             "specifically carries it (maps to CONTENT on the rider); "
             "COUNTER DESTROYED-EQUALLY — the step shape hints any "
             "perturbation (maps to CHANGE); POSITION the uncovered "
             "third.")
    L.append("")
    L.append("## The free rider (|W| / in-room; the name-blindness "
             "control standing)")
    L.append("")
    L.append(f"writes_blind: **{writes_blind}** "
             f"(cuts |W| spread <= {W_SPREAD_MAX}, in-room spread <= "
             f"{ROOM_SPREAD_MAX}; e346's conventions)")
    L.append("")
    for rung, entry in rider_table.items():
        cells = ", ".join(f"{a}: |W| {v['write_norm']:.2f} / ir "
                          f"{v['in_room']:.3f}"
                          for a, v in entry.items())
        L.append(f"- s{rung}: {cells}")
    for rung, s in spreads.items():
        L.append(f"  - s{rung} spreads: |W| {s['w_spread']:.3f} "
                 f"({'ok' if s['w_ok'] else 'BREACH'}), room "
                 f"{s['room_spread']:.4f} "
                 f"({'ok' if s['room_ok'] else 'BREACH'})")
    L.append("")
    L.append("## Gates")
    L.append("")
    for g in GATE_NAMES:
        if g in metrics["gates"]:
            L.append(f"- {g}: PASS")
    L.append("")
    L.append("## Disclosures")
    L.append("")
    L.append("- Route: the certified CPU-replay lane (chosen + "
             "disclosed at birth; zero CUDA; the bit-identity gates "
             "require it; phase-2 thermal standing N/A — no bursts; "
             "gpu snapshot recorded in metrics.envelope; the sibling "
             "desk cell x57 shares the desk; never concurrent CUDA).")
    L.append("- Page order = the canon pool's committed jitter-major "
             "construction order (x36's definition): the front (rows "
             "0-3) = hosts 0-3 @ j=-8 — x36's k4 swap site; the tail "
             "(rows 296-299) = hosts 56-59 @ j=+8 — the mirrored site.")
    L.append("- k4 (the reference kill) and k0 (the walk pole) are "
             "committed, runtime-read, never rerun "
             "(extend-don't-repeat); k4 re-certified in-cell "
             "(G_K4REF: draw-record bit-equality + an 8-step "
             "weight-level bit-proof + my pairing re-derives x36's "
             "committed scores).")
    L.append("- fshuf = the committed reversal derangement [3,2,1,0] "
             "(maximal displacement; the content multiset EXACTLY "
             "canon — only the pool-index placement changes).")
    L.append(f"- frjit = redrawn jitters (seed 58002 over "
             f"{{-4,0,+4,+8}}), realized {rejit_draws}: hosts 0-3 stay "
             "at rows 0-3 with changed context spans and name columns; "
             "IN-GRID DUPLICATION disclosed — each re-jittered row is "
             "bit-equal the canon row of that host at its new jitter, "
             "and the original host@-8 windows leave the pool (the map "
             "in G_X58POOLS); name-token COUNT per row unchanged (7; "
             "2100 per pool), the name COLUMN of rows 0-3 shifts with "
             "the jitter — inherent to a changed realization.")
    L.append("- Single Z channel (no dual): no foreign name exists in "
             "any x58 stream.")
    L.append("- n=1 seed per arm (the family's standing caveat); one "
             "lineage; the verdict binaries ride the frozen rule, the "
             "graded profile co-reports.")
    if metrics["deviations"]:
        L.append("")
        L.append("## Deviations / catches")
        L.append("")
        for d in metrics["deviations"]:
            L.append(f"- {d}")
    (RD / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X58 THE NON-FRONT SWAP + RIDER — "
        f"birth={metrics['status']} smoke={SMOKE}")
    metrics["birth_commit"] = git_head()
    p0 = phase_P0()
    p1 = phase_P1(p0)
    p2 = phase_P2(p0, p1)
    phase_P3(p0, p1, p2)
    n_pass = sum(1 for g in GATE_NAMES if g in metrics["gates"])
    log(f"X58 DONE: {n_pass}/{len(GATE_NAMES)} gates recorded; "
        f"nonfront={metrics.get('verdict_nonfront')}; "
        f"rider={metrics.get('verdict_rider')}")


if __name__ == "__main__":
    main()
