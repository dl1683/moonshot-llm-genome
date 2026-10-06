"""E280 — THE CAPACITY LADDER UNDER SGD-M (the day-twelve headline cell).
The question, frozen verbatim from the dispatch: is the expression edge at
(1k,2k] Adam's preconditioned geometry or the parameter space's own physics?

The instrument: THE SERIAL QUIET-WATER LADDER re-run at rungs {1k, 2k, 5k,
10k} under SGD-M instead of AdamW — the SAME rooms (bit-bound to
e272_rooms.pt: S1K = e261's committed K1K room via e272's K1KM bind; S2K/
S5K/S10K = e272's committed K2K/K5K/K10KR rooms), the SAME protocol (e001
fact-free base; Dmix install s400 gen 24314, bit-identical streams; hook
VERBATIM: backward -> clip 1.0 -> project CPU fp64 -> opt.step, norm NOT
rescaled), the SGD-M lr := e273's STABLE RIDER POINT (see the lr convention
below). The 40k rung SKIPPED (budget; e264's committed AdamW 40k cited in
the overlay/fit only) and 237k SKIPPED (cited: e264's 0.3844).

THE LR CONVENTION (frozen): LR_SGD_matched = 21.7385748014537 (e273's live
calibration: the first install step's applied in-room L2 matched to
AdamW's) DIVERGES — e273's smoke catch: SGD's linear step cannot carry the
matched scale (the free corpus steps run the model away; the family's own
stability boundary). CONSULT #007'S LICENSE (adopted): run at the
STABLE-BUT-UNDERMATCHED point — e272 proved dose does not buy expression
(P-272a; the kept-matched 1k arm stayed dead at 0.00125), so matched dose
is unnecessary to test the FORMATION floor. THIS cell's lr:
    LR_STABLE = x0.01 x LR_SGD_matched = 0.01 x 21.7385748014537
              = 0.2173857480145370
with the house cosine (lr(s) = LR_STABLE x cosine_lr(s-1, 1000)) — exactly
e273's SGD001X stable rider's scale, and the optimizer hyperparameters
match e273's stable rider EXACTLY: momentum 0.9, wd 0.0 (e273's disclosed
convention: SGD drops wd because AdamW's decoupled 0.1 has no L2-coupled
equivalent). DISCLOSED UNDERMATCHED: the s1 applied in-room L2 is ~x0.01
of AdamW's — the per-rung FIRST-STEP APPLIED-L2 LEDGER is recorded anyway
(full + in-room), never nominal.

REGISTERED BARS (frozen VERBATIM from the dispatch letter, in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping). Adjudication is on the WRITE read — post g0 on the
install-final net:
  - SPACE-INTRINSIC — "under SGD-M the edge HOLDS at (1k,2k] (1k below
    0.01 AND 2k >= 10x above 1k) — the capacity floor is the space's
    physics."
  - ADAM-CREATED — "1k EXPRESSES under SGD-M (>= 0.05) or the edge moves
    by >= 2x — the floor is Adam's preconditioned geometry."
  - MIXED — "the edge moves but < 2x, or the SGD-M arms fail to form at
    all rungs (report which; the undermatched-lr caveat then binds and the
    cell's honest verdict is INCONCLUSIVE-AT-THIS-LR with the trajectory
    verbatim)."
THE RIDERS' BARS (same letter): REPLICATE-IN-BAND ("both fresh rooms
within ~2x of committed"), 0.5X-DEAD ("post < 0.01"), SGD-MISSILE-ESCAPES
("in-room < 0.20") / MOTEL-IS-SPACE (">= 0.35") / MIXED between.

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE SGD-M LADDER := posts over {1k: S1K, 2k: S2K, 5k: S5K, 10k: S10K};
    adjacent pairs (1k->2k), (2k->5k), (5k->10k); a pair FIRES iff
    post(hi)/max(post(lo), 1e-6) >= 10 AND post(lo) < 0.01; the EDGE
    BRACKET := the firing pair with the largest k_lo.
  * "the edge moves by >= 2x" := the located bracket's dead-side k_lo >=
    2 x 1000 (the AdamW bracket is (1000, 2000], committed e272). With
    rungs {1k,2k,5k,10k} any bracket move is >= 2.5x in k — the "< 2x"
    MIXED branch is carried for the letter's completeness (disclosed).
  * "the SGD-M arms fail to form at all rungs" := max over the four SGD
    posts < 0.01 (nothing above the dead bar anywhere on the ladder) —
    then the verdict is the named MIXED branch INCONCLUSIVE-AT-THIS-LR and
    the per-rung trajectories + the first-step applied-L2 ledger are
    reported verbatim (the undermatched-lr caveat binds).
  * NON-FINITE READS: a diverged arm's non-finite posts are carried
    verbatim in the trajectories and treated as did-not-form (0.0) in the
    ratio arithmetic ONLY, with the divergence disclosed in the clause
    (e273's DIVERGENCE-CONFOUNDED routing precedent).
  * RIDER (a) THE FIRING-PAIR REPLICATE (the R66-critic debt): one fresh
    1k room (seeds 28011/28012) + one fresh 2k room (seeds 28013/28014)
    under ADAMW at the family dose (lr 1e-3) — REPLICATE-IN-BAND :=
    R1KR post in [0.5x, 2x] x 0.00043458465370349586 (e261's committed
    K1K) AND R2KR post in [0.5x, 2x] x 0.026616254821419716 (e272's
    committed K2K) — this prices the room lottery at the edge pair itself.
  * RIDER (b) THE 0.5x-COMPENSATED 1K ARM under AdamW: e272's committed
    K1K room (bit-bound) with install lr x LR_SCALE_HALF = 0.5 x
    kept(10k)/kept(1k) = 0.5 x 3.7305567687315575 = 1.8652783843657788
    (e272's disclosed E43.LR module rebind, restored in a finally block);
    P REGISTERED HERE (P-280b): DEAD (post < 0.01) — dose acquitted at
    three lr points (1x natural e261; 1.865x here; 3.731x e272's K1KM
    which landed 0.00125).
  * RIDER (c) THE SGD-MISSILE (P-x283b, T260 — the roach motel's escape
    test): ONE concurrent arm at k=10k — e278's MISSILE construction
    VERBATIM (the 1:1 interleave: install steps projected onto the room +
    corpus steps orthogonalized g_perp = g - P_room(g), verified
    ||P g_perp||/||g_perp|| < 1e-6 per step) but under ONE SGD(momentum
    0.9, wd 0) at LR_STABLE x cosine for BOTH streams (the paired lr);
    corpus generator THIS cell's fresh seed 28001 (the family's fresh-seed
    precedent; a literal seed-27801 replay would add nothing — the read
    is the displacement ledger, not the draw identity). THE MISSILE READ
    := the median over milestones {100, 200, 300, 400} of the corpus
    stream's realized-displacement INTERVAL in-room fraction (e278's
    AdamW missile intervals ran 0.4814-0.5988 — the cited 48-60%);
    cumulative fracs co-reported. PREDICTION registered (P-x283b, cited
    T260): in-room share < 0.20 (vs Adam's 0.48-0.60) IFF the motel is
    Adam's architecture. G_MISSILE_ORTH hard-gates the construction
    (max orth rel err < 1e-6).
  * THE W046 COORDINATE (T258/W046's frozen read): fit the two-parameter
    power law post(k) ~= A x k^beta to EACH optimizer's ladder over the
    rungs at-or-above that optimizer's own floor-bracket high side
    (posts > 0); report (floor-bracket, exponent) per optimizer as the
    cell's coordinate. AdamW's fit domain: the committed rungs {2k: e272
    K2K, 5k: e272 K5K, 10k: e264 K10K} (bracket (1k,2k]); the 40k-included
    variant co-reported as context (W046's own caveat carried: the curve
    is convex-to-power early — the 2k-10k local slope is steeper than the
    118x-width slope ~0.63). SGD-M's fit domain: its own bracket's rungs
    (or, if no bracket forms, the rungs with post > 0.01 — disclosed).
    Same-ks comparability: the primary fits share the domain {2k, 5k,
    10k} whenever the SGD bracket is (1k,2k].
  * HARD GATES (a failure HALTS): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR(bank), G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND,
    G_SPANBIND, G_PROJ, G_ROOMS, G_LR_BIND, G_DOSE_ARITH, G_MISSILE_ORTH}.
    G_ROOMS bit-binds ALL FOUR ladder rooms to e272_rooms.pt's stored
    K1KM/K2K/K5K/K10KR D/S (exact equality) — the SGD arms' ONLY delta vs
    e272's committed AdamW ladder is the OPTIMIZER (the riders' fresh
    rooms excepted by construction).
  * NO FREE ARM and NO anchor re-run (disclosed: no SGD arm is
    bit-identical to any committed rung BY CONSTRUCTION — the optimizer is
    the intervention; the AdamW riders exist to price the room lottery at
    the pair (a) and the lr dose (b)); the instrument is the lineage: the
    bit-bound rooms + e264/e268-e271's serial-anchor determinism law
    (|d post g0| 2.4e-7-1e-6) + THIS cell's G_ROOMS bit-gates.
  * NO CONS (T259/e281; e278/e283's committed form): the landing read is
    a cons property (0.6508 from a fact-free base — the cons teaches from
    anything); the frozen bars read the WRITE (post g0) only; every arm's
    install-final state is CHECKPOINTED (runs/checkpoints/e280_<arm>_post.pt)
    for any later landing pass.
  * MEASURED, NEVER NOMINAL: per-rung kept ledgers, v-excess pre/post,
    in-span pre/applied, displacement loads, the per-rung FIRST-STEP
    APPLIED-L2 ledger (full + in-room + the momentum-exactness self-check
    lr x ||g'||), the missile's per-milestone displacement ledger.

REGISTERED PREDICTIONS (cited where on the record, registered here where
new):
  - P-x283b (T260, CITED verbatim): "under SGD-M at the stable lr, an
    orthogonal corpus gradient's REALIZED displacement stays
    substantially out-of-room (in-room share < 20% vs the missile-under-
    Adam's 48-60%) — IF this holds, the roach motel is Adam's
    ARCHITECTURE, not the space's."
  - P-280b (REGISTERED HERE, rider b): the 0.5x-compensated 1k arm lands
    DEAD (post g0 < 0.01) — dose acquitted at three lr points.
  - P-280m (REGISTERED HERE, the main ladder): the SGD-M ladder KEEPS the
    (1k,2k] bracket with 1k dead (the floor is rank-written in the space,
    not the optimizer's normalizer) — the null this cell exists to test.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175 s (inside the dispatch's 180), cooldowns 40 s (inside 30-60), the
84 C never-past line (inside the dispatch's 85), per-step thermal polls
persisted to runs/_envelope_log.jsonl tagged e280:<ARM>:<phase>; CPU fp64
dense projections (pocketfft workers 2); SERIAL ARMS ONLY — the missile
is "concurrent" in ITS streams (the 1:1 interleave) but owns the lane
alone; NO two arms ever share the GPU.

Outputs: runs/e280/{metrics.json (PROGRESSIVE), e280_sgd_ladder.png,
REPORT.md, run.log (gitignored)}; checkpoints
runs/checkpoints/e280_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e280_sgd_ladder.py    (E280_SMOKE=1 shakedown)
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
                                                      # LADDER MACHINERY,
                                                      # PORTED WHOLE BY
                                                      # IMPORT (the AdamW
                                                      # serial driver runs
                                                      # VERBATIM; the
                                                      # committed file is
                                                      # NOT modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E280_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e280_smoke" if SMOKE else "e280"
assert torch.cuda.is_available(), "e280 owns the GPU lane (dispatch)"

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
# e261's drivers land their rows in THIS cell's ledger — e270/e272's
# instrumentation convention)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e264/e272's disclosed convention, in force before ANY
# machinery call): e261's drivers resolve their module globals at CALL TIME
# through e261's namespace — rebound HERE so they write THIS cell's log,
# label THIS cell's envelope polls, and land their thermal rows in THIS
# cell's ledger. The committed lab/e261_rank_ladder.py is untouched.
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
ROOMS272_CK = "e272_rooms.pt"     # e272's committed rooms (the bit-bind source)
CKPT_DIR = GB.CKPT_DIR

# ---- THE ROOMS. spec: (name, k, seed_d, seed_s). The four ladder rooms are
# BIT-BOUND to e272_rooms.pt (S1K is e261's committed K1K room by way of
# e272's K1KM bind); the replicate rooms are FRESH (the lottery probe); the
# missile shares S10K's room. Smoke: same names at smoke ks (bit-gate
# vacuous, disclosed).
SPEC_FULL: tuple[tuple[str, int, int, int], ...] = (
    ("S1K", 1_000, 26111, 26112),     # == e272's K1KM room (== e261's K1K)
    ("S2K", 2_000, 27211, 27212),     # == e272's K2K room
    ("S5K", 5_000, 27213, 27214),     # == e272's K5K room
    ("S10K", 10_000, 27215, 27216),   # == e272's K10KR room (missile's too)
    ("R1KR", 1_000, 28011, 28012),    # FRESH 1k room (rider a)
    ("R2KR", 2_000, 28013, 28014),    # FRESH 2k room (rider a)
)
SPEC_SMOKE: tuple[tuple[str, int, int, int], ...] = (
    ("S1K", 256, 26111, 26112),
    ("S2K", 64, 27211, 27212),
    ("S5K", 128, 27213, 27214),
    ("S10K", 512, 27215, 27216),
    ("R1KR", 128, 28011, 28012),
    ("R2KR", 256, 28013, 28014),
)
SPEC = SPEC_SMOKE if SMOKE else SPEC_FULL

# the e272_rooms.pt room-name bind for the four ladder rooms
ROOMS272_BIND = {"S1K": "K1KM", "S2K": "K2K", "S5K": "K5K", "S10K": "K10KR"}

# ---- THE SGD-M CONFIG (frozen; see the docstring's lr convention) --------
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.2173857480145370

# ---- THE 0.5x RIDER'S COMPENSATION (frozen; e272's arithmetic halved) ----
KEPT_1K_STORED = 0.016095496225535792            # kept_frac_curve['1000']
KEPT_10K_STORED = 0.060045162390265784           # kept_frac_curve['10000']
LR_SCALE_FULL = KEPT_10K_STORED / KEPT_1K_STORED     # 3.7305567687315575
LR_SCALE_HALF = 0.5 * LR_SCALE_FULL                  # 1.8652783843657788
LR_BASE = None            # read at bind from E43.LR (must be 1e-3)

# ---- the missile's stream seeds (this cell's registrations)
CORPUS_GEN_SEED = 28001            # the missile's corpus stream (fresh)
MISSILE_ROOM = "S10K"

# ---- EXECUTION order: the AdamW riders first (the pair debt + the dose
# rider — banked while the machine is cold), then the SGD ladder ascending,
# then the missile (the concurrent lane's sole occupant).
ARMS_ADAMW = ("R1KR", "R2KR", "K1KM05")
ARMS_SGD = ("S1K", "S2K", "S5K", "S10K")
ARMS = ARMS_ADAMW + ARMS_SGD + ("MISSILE_SGD",)

# ---- the committed records, HARD-BOUND (Rule 12; md5-gated at runtime)
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "f460475d8e6b76f0719e91c1e9c6041b"
E261_K1K_POST = 0.00043458465370349586            # THE cited 1k rung (dead)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_K10K_POST = 0.26464763283729553              # THE cited 10k rung (alive)
E264_K40K_POST = 0.346476286649704                # cited context (the fit variant)
E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "eb9f624708bcb6576c4115161dfd7042"
E272_K2K_POST = 0.026616254821419716              # the committed 2k rung
E272_K5K_POST = 0.12709596753120422               # the committed 5k rung
E272_K1KM_POST = 0.0012453667586669326            # the full-dose probe (dead)
E272_VERDICT = "RANK-WRITES-THE-CURVE"
E272_ROOMS_MD5 = "066944855b3295e8796c6ca28b2e498c"
E273_METRICS = E43.REPO / "runs" / "e273" / "metrics.json"
E273_MD5 = "df0f608ad0b89cbb0407049418ca4f86"
E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"
E273_LR_SGD = 21.7385748014537                    # the calibration record's value
E278_MISSILE_BAND = (0.4814, 0.5988)              # e278's AdamW missile intervals

# the committed AdamW ladder (the overlay + the W046 fit's AdamW side)
ADAMW_LADDER = {1000: E261_K1K_POST, 2000: E272_K2K_POST,
                5000: E272_K5K_POST, 10000: E264_K10K_POST}
ADAMW_BRACKET = (1000, 2000)                      # e272's committed edge

# the frozen bars' numbers
DEAD_BAR = 0.01                   # the dead side / the formation bar
EXPRESS_FLOOR = 0.05              # the family's frozen expression floor
JUMP_BAR = 10.0                   # the >= 10x adjacent-jump bar
RATIO_DEN_FLOOR = 1e-6            # the ladder's floor-guard convention
REPL_BAND = (0.5, 2.0)            # the replicate rider's "~2x" window
MISSILE_ESCAPE_BAR = 0.20         # P-x283b's < 20%
MISSILE_MOTEL_BAR = 0.35          # the motel-is-space bar
G_READ_TOL = E261.G_READ_TOL      # 5e-3
VOCAB_EXPECT = 65

TRAJ_MILE = (1, 100, 200, 300, 400)

REGISTERED = {
    "question_verbatim": "Is the expression edge at (1k,2k] Adam's "
        "preconditioned geometry or the parameter space's own physics?",
    "bars_verbatim": {
        "SPACE-INTRINSIC": "under SGD-M the edge HOLDS at (1k,2k] (1k below "
            "0.01 AND 2k >= 10x above 1k) — the capacity floor is the "
            "space's physics.",
        "ADAM-CREATED": "1k EXPRESSES under SGD-M (>= 0.05) or the edge "
            "moves by >= 2x — the floor is Adam's preconditioned geometry.",
        "MIXED": "the edge moves but < 2x, or the SGD-M arms fail to form "
            "at all rungs (report which; the undermatched-lr caveat then "
            "binds and the cell's honest verdict is INCONCLUSIVE-AT-THIS-LR "
            "with the trajectory verbatim).",
    },
    "riders_bars_verbatim": {
        "REPLICATE-IN-BAND": "both fresh rooms within ~2x of committed",
        "0.5X-DEAD": "post < 0.01",
        "SGD-MISSILE-ESCAPES": "in-room < 0.20",
        "MOTEL-IS-SPACE": ">= 0.35",
        "MIXED": "between",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE LADDER := four serial SGD-M arms at "
        f"{{1k, 2k, 5k, 10k}} (rooms bit-bound to e272_rooms.pt; e001 "
        "fact-free base; Dmix install s400 gen 24314 bit-identical streams; "
        "hook VERBATIM backward -> clip 1.0 -> project CPU fp64 -> opt.step, "
        "norm NOT rescaled; optimizer = SGD(momentum 0.9, wd 0.0) at "
        f"LR_STABLE = {SGD_STABLE_FACTOR} x {LR_SGD_MATCHED!r} = "
        f"{LR_STABLE!r} x cosine_lr(s-1,1000) — e273's SGD001X stable "
        "rider's scale + hyperparameters EXACTLY; DISCLOSED UNDERMATCHED "
        "(~x0.01 of the matched s1 applied in-room L2; the matched lr "
        "DIVERGES — e273's smoke catch; consult #007's license adopted: "
        "e272 acquitted dose, matched dose unnecessary for the formation "
        "floor); the per-rung FIRST-STEP APPLIED-L2 LEDGER recorded (full "
        "+ in-room + the momentum-exactness self-check); 'edge moves >= 2x' "
        ":= the located bracket's dead-side k_lo >= 2000 (AdamW's committed "
        "bracket is (1000,2000]; with rungs {1k,2k,5k,10k} any move is "
        ">= 2.5x — the <2x branch carried for completeness); 'fail to form "
        "at all rungs' := max SGD post < 0.01 -> the named MIXED branch "
        "INCONCLUSIVE-AT-THIS-LR; non-finite posts carried verbatim + "
        "treated as did-not-form in ratio arithmetic only (divergence "
        "disclosed in the clause); RIDER (a) := fresh 1k room 28011/28012 "
        "+ fresh 2k room 28013/28014 under AdamW lr 1e-3 (e261's driver "
        "VERBATIM by import) — REPLICATE-IN-BAND := R1KR in "
        f"[{REPL_BAND[0]}x, {REPL_BAND[1]}x] x {E261_K1K_POST!r} AND R2KR "
        f"in [{REPL_BAND[0]}x, {REPL_BAND[1]}x] x {E272_K2K_POST!r}; RIDER "
        "(b) := the committed K1K room at install lr x "
        f"{LR_SCALE_HALF!r} (= 0.5 x kept(10k)/kept(1k); e272's disclosed "
        "E43.LR rebind) — 0.5X-DEAD := post < 0.01 (P-280b registered "
        "HERE: dead); RIDER (c) := e278's MISSILE construction VERBATIM "
        "(1:1 interleave; install projected onto the room; corpus "
        "orthogonalized g_perp = g - P_room(g), verified < 1e-6/step) at "
        f"k=10k under ONE SGD(momentum 0.9, wd 0) at LR_STABLE x cosine, "
        f"corpus gen seed {CORPUS_GEN_SEED} (fresh; the family's "
        "fresh-seed precedent) — THE MISSILE READ := the median over "
        "milestones {100,200,300,400} of the corpus stream's realized-"
        "displacement INTERVAL in-room fraction; cumulative co-reported; "
        "bars < 0.20 ESCAPES / >= 0.35 MOTEL-IS-SPACE / else MIXED; THE "
        "W046 COORDINATE := the 2-parameter power fit post ~ A k^beta over "
        "each optimizer's own bracket-top rungs (posts > 0), reported as "
        "(floor-bracket, exponent) per optimizer; the AdamW primary domain "
        "{2k, 5k, 10k} (committed cites; the 40k-included variant "
        "co-reported); composite order: hard-gate failure (TEXTURE) -> "
        "SPACE-INTRINSIC -> ADAM-CREATED -> MIXED (named); the ADJUDICATION "
        "READ is the WRITE read (post g0); NO CONS (T259/e281; e278/e283 "
        "precedent — the landing read is a cons property; post states "
        "checkpointed); HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, "
        "G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, "
        "G_SPANBIND, G_PROJ, G_ROOMS, G_LR_BIND, G_DOSE_ARITH, "
        "G_MISSILE_ORTH} — a failure HALTS; 40k/237k rungs SKIPPED "
        "(budget/cited; e264's committed posts appear in the overlay and "
        "the fit variant only)."),
    "registration": "bars + question + riders' bars frozen VERBATIM from "
        "the day-twelve dispatch letter (the capacity ladder under SGD-M); "
        "this script committed at birth BEFORE any compute; adjudicate "
        "against exactly this; no bar shopping.",
    "predictions": {
        "P-x283b_T260_cited": "under SGD-M at the stable lr, an orthogonal "
            "corpus gradient's REALIZED displacement stays substantially "
            "out-of-room (in-room share < 20% vs the missile-under-Adam's "
            "48-60%) — IF this holds, the roach motel is Adam's "
            "ARCHITECTURE, not the space's (CITED from T260, not "
            "re-registered).",
        "P-280b_registered": "the 0.5x-compensated 1k arm lands DEAD (post "
            "g0 < 0.01) — dose acquitted at three lr points (1x e261; "
            "1.865x here; 3.731x e272's K1KM -> 0.00125).",
        "P-280m_registered": "the SGD-M ladder KEEPS the (1k,2k] bracket "
            "with 1k dead — the floor is rank-written in the space, not "
            "the optimizer's normalizer (the null this cell exists to "
            "test).",
    },
}

deviations: list[str] = [
    "THE SGD-M LR IS STABLE-BUT-UNDERMATCHED (the cell's central "
    "disclosure, frozen at birth): the matched lr (x21,738.6 of 1e-3) "
    "DIVERGES (e273's committed smoke catch — SGD's linear step cannot "
    "carry Adam's sign-equalized scale; the free corpus steps ran the "
    "model away, and e273's own SGDM barrel at the matched lr collapsed "
    "to 0/nan). Consult #007's license ADOPTED: e272 acquitted dose a "
    "fortiori (the kept-matched 1k arm stayed dead at 0.00125), so "
    "matched dose is unnecessary to test the formation floor. THIS cell "
    "runs at x0.01 of matched — e273's SGD001X stable rider's exact scale "
    "and hyperparameters (momentum 0.9, wd 0.0) — and MEASURES the "
    "per-rung first-step applied L2 (full + in-room) so the undermatch is "
    "quantified, never nominal. If nothing forms anywhere the verdict is "
    "the named INCONCLUSIVE-AT-THIS-LR branch, never a silent pass.",
    "NO CONS (T259/e281; e278/e283's committed form): the landing read is "
    "a cons property (0.6508 from a fact-free base) — the frozen bars read "
    "the WRITE (post g0) only; every arm's install-final state is "
    "checkpointed for any later landing pass. The rehearsal-lane caveat "
    "is thereby moot in this cell.",
    "THE 40k AND 237k RUNGS ARE SKIPPED (budget; disclosed): the SGD "
    "ladder tops at 10k; e264's committed AdamW 40k (0.3465) and 237k "
    "(0.3844) posts appear in the overlay and the W046 fit's co-reported "
    "variant only — the primary fits share the domain {2k, 5k, 10k} so "
    "the two optimizers are compared on the same rungs.",
    "NO FREE ARM, NO ANCHOR RE-RUN (disclosed): the SGD arms' ONLY delta "
    "vs e272's committed AdamW ladder is the optimizer (rooms bit-bound, "
    "G_ROOMS); the AdamW riders exist to price the room lottery at the "
    "pair (a) and the lr dose (b). The instrument is the bit-bound rooms "
    "+ the lineage's serial-anchor determinism law (e264/e268-e271: |d "
    "post g0| 2.4e-7-1e-6 on bit-identical re-runs).",
    "THE MISSILE'S CORPUS STREAM IS FRESH-SEEDED (seed 28001, this cell's "
    "registration): the family's fresh-seed precedent (e269/e270/e271/"
    "e273 each re-ran the form on its own stream); the read is the "
    "displacement ledger (a geometry ratio), not the draw identity — a "
    "literal seed-27801 replay would add nothing.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked serial AdamW install driver (runs the three AdamW arms "
    "VERBATIM — the 0.5x rider under the disclosed E43.LR module rebind, "
    "restored in a finally block), the thermal envelope, the progressive-"
    "metrics + resume-ckpt conventions. The ONE new serial driver "
    "(chunked_install_opt) is e261's chunked_install with the optimizer "
    "constructor + lr base parameterized and the first-step applied-L2 "
    "ledger added — the draw order, batching, hook, and arithmetic are "
    "line-identical; the ONE new concurrent driver (chunked_missile_sgd) "
    "is e278's chunked_install_threenull with the optimizer swapped to "
    "the ONE SGD and the isotope paths removed. The committed "
    "lab/e261_rank_ladder.py and lab/e278_three_null.py are NOT modified.",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the firing-pair "
    "replicate exists precisely to price the n=1 room lottery at the "
    "edge; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E280_SMOKE=1): 8-step installs + 8-step missile, the "
    "six room names at smoke ks {256(S1K), 64(S2K), 128(S5K), 512(S10K), "
    "128(R1KR), 256(R2KR)}, G_ROOMS vacuous (no committed record at smoke "
    "k; disclosed), the SAME committed LR_STABLE + LR_SCALE_HALF "
    "exercised, all paths smoke_-prefixed, own smoke dir; NOTHING "
    "adjudicated or gated (SMOKE stamp on every read).",
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
    return CKPT_DIR / (f"smoke_e280_{arm}_inst_resume.pt" if SMOKE
                       else f"e280_{arm}_inst_resume.pt")


# ======================================================================
# THE ROOMS (e261's LadderRooms with NAME-KEYED specs — this cell has two
# rooms at k=1000 and two at k=2000, which RUNG_NAMES's k->name map cannot
# express; the projector/hook/loads arithmetic is INHERITED VERBATIM)
# ======================================================================
class Rooms280(E261.LadderRooms):
    def __init__(self, n: int, spec, v_flat64: np.ndarray,
                 Vp64: np.ndarray, params_ref, dev: torch.device):
        self.n = int(n)
        self.dev = dev
        self.r_span = int(Vp64.shape[0])
        self.v64 = v_flat64.astype(np.float64)
        self.mean_v = float(self.v64.mean())
        self.Vp = Vp64.astype(np.float64)
        self.rooms: dict[str, E261.SRCT] = {}
        self.room_k: dict[str, int] = {}
        for nm, k, sd, ss in spec:
            assert nm not in self.rooms, f"duplicate room name {nm}"
            self.rooms[nm] = E261.SRCT(n, k, sd, ss)
            self.room_k[nm] = int(k)
        self.offsets, self.shapes = [], []
        off = 0
        for p in params_ref:
            self.offsets.append((off, off + p.numel()))
            self.shapes.append(tuple(p.shape))
            off += p.numel()
        assert off == self.n, f"flat size {off} != {self.n}"

    def certify280(self) -> dict:
        """e261's certify arithmetic over the NAME-KEYED room set (the
        module-global LADDER/RUNG_NAMES form cannot express duplicate ks;
        the probes, bars, and reads are identical)."""
        rng = np.random.default_rng(E261.CERT_SEED)
        out: dict = {"probes": E261.CERT_PROBES, "seed": E261.CERT_SEED}
        first = self.rooms[next(iter(self.rooms))]
        x = rng.standard_normal(self.n)
        xr = first.recon(first.coeffs(x))
        out["dct_roundtrip_rel"] = float(np.linalg.norm(xr - x)
                                         / np.linalg.norm(x))
        per_room = {}
        for nm, room in self.rooms.items():
            k = room.k
            idem, kept2 = [], []
            for _ in range(E261.CERT_PROBES):
                x = rng.standard_normal(self.n)
                px = room.project(x)
                ppx = room.project(px)
                idem.append(float(np.linalg.norm(ppx - px)
                                  / np.linalg.norm(px)))
                kept2.append(float((px @ px) / (x @ x)))
            ovr = [float(np.linalg.norm(room.project(self.Vp[j])))
                   for j in range(self.r_span)]
            bar = max(5.0 * math.sqrt(2.0 * k) / self.n, 1e-9)
            per_room[nm] = {
                "k": k, "seeds": [room.seed_d, room.seed_s],
                "idempotency_max": max(idem),
                "kept2_mean": float(np.mean(kept2)),
                "kept2_expect": k / self.n,
                "kept2_bar_10sig": bar,
                "kept2_pass": bool(abs(float(np.mean(kept2)) - k / self.n)
                                   <= bar),
                "idempotency_pass": bool(max(idem) <= 1e-8),
                "span_overlap_mean": float(np.mean(ovr)),
                "span_overlap_expect": math.sqrt(k / self.n),
            }
        out["per_room"] = per_room
        out["pass"] = bool(out["dct_roundtrip_rel"] <= 1e-8
                           and all(r["kept2_pass"] and r["idempotency_pass"]
                                   for r in per_room.values()))
        return out


# ======================================================================
# THE NEW SERIAL DRIVER — e261's chunked_install with the optimizer
# parameterized (the ONLY deltas: the optimizer constructor + lr base + the
# first-step applied-L2 ledger; draw order/batching/hook/arithmetic
# line-identical)
# ======================================================================
def chunked_install_opt(tag, mode, net0, proj: Rooms280,
                        inst_x, inst_mask, anchor_full, train_ids, g0_ids,
                        gm12_ids, r_eval_xy, zid, resume_ck: Path,
                        dev: torch.device, optimizer: str) -> dict:
    assert optimizer in ("sgd",), "the AdamW arms run e261.chunked_install"
    name_bs, corp_bs, mix_random = (G1.NAME_BS, E43.CORP_BS, E43.MIX_RANDOM)
    lr = LR_STABLE                     # the frozen stable-rider lr (sgd)
    n_steps = E261.INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    state = {"step": 0, "traj": [], "ledger": {}, "first_step": {},
             "diverged": False}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "first_step": state.get("first_step", {}),
                "diverged": state.get("diverged", False),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net, opt, gen, evl = None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[mode]
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=SGD_MOMENTUM,
                                  weight_decay=SGD_WD)
            gen = torch.Generator().manual_seed(E261.FRESH_GEN)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = cosine_lr(step - 1, E261.INST_TOTAL)          # house schedule
            lr_now = lr * f
            for g in opt.param_groups:
                g["lr"] = lr_now
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            # ---- THE HOOK (verbatim — the room projection) ----------------
            led = proj.step_hook(list(net.parameters()), mode)
            if step == 1 and not state["first_step"]:
                th0 = flat_params_cpu(net).double()
                opt.step()
                th1 = flat_params_cpu(net).double()
                d1 = (th1 - th0).numpy().astype(np.float64)
                d1_full = float(np.linalg.norm(d1))
                d1_in = float(np.linalg.norm(room.project(d1)))
                state["first_step"] = {
                    "lr_s1": lr_now,
                    "full_l2": d1_full, "in_room_l2": d1_in,
                    "gpn_s1": led["gpn"],
                    "momentum_exact_pred": lr_now * led["gpn"],
                    "momentum_exact_rel_err": (
                        abs(d1_full - lr_now * led["gpn"])
                        / max(lr_now * led["gpn"], 1e-30)),
                    "undermatch_vs_matched_lr": SGD_STABLE_FACTOR,
                }
            else:
                opt.step()
            if not math.isfinite(float(loss.item())):
                if not state.get("diverged", False):
                    log(f"  [{tag}] NON-FINITE batch CE at s{step} — "
                        "DIVERGENCE flagged (disclosed; the MIXED branch "
                        "routes it)")
                state["diverged"] = True
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["ledger"][step] = {
                    "ce": float(loss.item()), "gn": led["gn"],
                    "gpn": led["gpn"], "kept_frac": led["kept_frac"],
                    "v_excess_pre": led["v_excess_pre"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"]}
            if step % 100 == 0 or step == n_steps or step == 1 or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step,
                                      "g0_pz": bz0["mean_pz"],
                                      "g0_argmax": bz0["frac_argmax_z"],
                                      "gm12_pz": bz12["mean_pz"],
                                      "ce_r": ce_r,
                                      "ce_batch": float(loss.item()),
                                      "kept_frac": led["kept_frac"],
                                      "v_excess_post": led["v_excess_post"],
                                      "in_span_frac": led["in_span_frac"],
                                      "applied_in_span_frac":
                                          led["applied_in_span_frac"],
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.4f} g-12 "
                    f"{bz12['mean_pz']:.4f} CE_R {ce_r:.4f} CE "
                    f"{float(loss.item()):.4f} |g| {led['gn']:.3f} kept "
                    f"{led['kept_frac']:.4f} vexc pre "
                    f"{led['v_excess_pre']:.2f} post "
                    f"{led['v_excess_post']:.2f} inspan "
                    f"{led['in_span_frac']:.3f}->"
                    f"{led['applied_in_span_frac']:.3f}")
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(f"{tag}-c{n_chunks}")
            chunk_temps.append(temp)
            if not ok_t:
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
                    "step": step, "traj": state["traj"],
                    "ledger": state["ledger"],
                    "first_step": state["first_step"],
                    "diverged": state["diverged"],
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
            "first_step": state["first_step"],
            "diverged": state["diverged"],
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}


# ======================================================================
# THE MISSILE'S PROJECTION (e278's orthogonalize_grads VERBATIM by port)
# ======================================================================
def orthogonalize_grads(proj: Rooms280, params, mode: str) -> dict:
    """THE GUIDED MISSILE'S ONLY INTERVENTION (e278 verbatim): replace the
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


# ======================================================================
# THE SGD-MISSILE DRIVER — e278's chunked_install_threenull (MISSILE form)
# with the ONE SHARED optimizer swapped to SGD-M at the stable lr
# ======================================================================
def chunked_missile_sgd(tag, net0, proj: Rooms280, base_flat_np: np.ndarray,
                        inst_x, inst_mask, anchor_full, train_ids,
                        g0_ids, gm12_ids, r_eval_xy, zid, mode: str,
                        resume_ck: Path, dev: torch.device) -> dict:
    """e278's MISSILE construction VERBATIM (the 1:1 interleave: install
    step bit-identical to the serial driver's s — projected onto the room;
    corpus step = 16 anchors + 32 random corpus windows from cgen, full-
    window CE, the PAIRED lr, backward -> clip 1.0 -> g_perp = g - P_room(g)
    (verified) -> opt.step) with the ONE SHARED optimizer = SGD(momentum
    0.9, wd 0) at LR_STABLE x cosine_lr(s-1,1000) for BOTH streams. The
    displacement ledger (fp64 at milestones): the corpus stream's realized
    displacement accumulated; per milestone the interval's and the
    cumulative in-room fraction projected and recorded — THE CELL'S READ
    (P-x283b). Thermal: a poll after EVERY opt step (both streams)."""
    name_bs, corp_bs, mix_random = (G1.NAME_BS, E43.CORP_BS, E43.MIX_RANDOM)
    lr = LR_STABLE
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
    net = opt = gen = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    # the corpus-displacement accumulators (GPU fp32 per step — e278's
    # smoke performance catch; the projections stay CPU fp64 at milestones)
    corp_cum = state["corp_cum"].to(dev)
    corp_prev = state["corp_prev"].to(dev)
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
            opt = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=SGD_MOMENTUM,
                                  weight_decay=SGD_WD)
            gen = torch.Generator().manual_seed(E261.FRESH_GEN)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = cosine_lr(step - 1, E261.INST_TOTAL)
            lr_now = lr * f
            for g in opt.param_groups:
                g["lr"] = lr_now
            # ---- 1. THE INSTALL STEP (bit-identical to the serial s) ------
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
            ok_t, temp = E261.burst_temp_check(f"{tag}-c{n_chunks}.i")
            chunk_temps.append(temp)
            # ---- 2. THE CORPUS STEP (the missile's orthogonalized stream) -
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
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt.step()                          # the orthogonalized step
            with torch.no_grad():
                corp_cum += torch.cat([p.detach().reshape(-1)
                                       for p in net.parameters()]) - theta_b
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c}
            n_burst += 1
            ok_t2, temp2 = E261.burst_temp_check(f"{tag}-c{n_chunks}.x")
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
                    "orth_max_so_far": orth_max,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.5f} g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_inst "
                    f"{float(loss.item()):.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} kept "
                    f"{led['kept_frac']:.4f} |d| {dn:.4f} corp|v| "
                    f"{vn if vn else 0.0:.3f} "
                    f"({('in-room %.4f' % in_room_frac_c) if in_room_frac_c is not None else 'n/a'})"
                    f" orth {orth_max:.1e}")
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
                    "step": step, "traj": state["traj"],
                    "ledger": state["ledger"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "corp_cum": corp_cum.cpu(),
                    "corp_prev": corp_prev.cpu(),
                    "orth_max": orth_max,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 16:
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
    out = {"burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
           "per_step_polls": "after EVERY opt step (both missile streams) —"
                             " aggregated from runs/_envelope_log.jsonl "
                             "(the persisted ledger; survives resume passes)",
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
                   f"polls are tagged e280_smoke: and excluded")
    return out


def fit_power(pts: dict) -> dict:
    """THE W046 2-PARAMETER COORDINATE'S EXPONENT: least-squares log-log
    slope of post(k) ~ A k^beta over the domain's positive posts."""
    ks = sorted(k for k, v in pts.items()
                if v is not None and math.isfinite(v) and v > 0.0)
    if len(ks) < 2:
        return {"n_points": len(ks),
                "ks": [int(k) for k in sorted(pts)],
                "exponent": None, "A": None, "r2": None,
                "note": "fewer than 2 positive posts — no fit (disclosed)"}
    x = np.log(np.array(ks, dtype=np.float64))
    y = np.log(np.array([float(pts[k]) for k in ks], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"n_points": len(ks), "ks": [int(k) for k in ks],
            "exponent": float(slope), "A": float(math.exp(intercept)),
            "r2": r2}


def main():
    global LR_BASE
    dev = torch.device("cuda")
    LR_BASE = float(E43.LR)
    assert abs(LR_BASE - 1e-3) < 1e-15, f"E43.LR unexpected: {LR_BASE}"
    metrics.update({
        "experiment": "e280_sgd_ladder",
        "phase": "THE CAPACITY LADDER UNDER SGD-M — the day-twelve headline "
                 "cell: is the expression edge at (1k,2k] Adam's "
                 "preconditioned geometry or the parameter space's own "
                 "physics? The serial quiet-water ladder {1k,2k,5k,10k} "
                 "re-run under SGD(momentum 0.9, wd 0) at e273's stable "
                 "rider lr (x0.01 of matched — DISCLOSED UNDERMATCHED) on "
                 "e272's bit-bound rooms; + the firing-pair replicate "
                 "(R66 debt) + the 0.5x-compensated 1k arm + the "
                 "SGD-missile (P-x283b) — SPACE-INTRINSIC vs ADAM-CREATED "
                 "vs MIXED, adjudicated on the WRITE read (post g0)",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property "
                           "(0.6508 from a fact-free base; the cons teaches "
                           "from anything); the frozen bars read the WRITE "
                           "only; e278/e283's committed NO-CONS form; every "
                           "arm's install-final state is checkpointed"},
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "SERIAL arms only — the missile's two streams are its "
                      "own) + CPU fp64 dense projections (pocketfft workers "
                      "2), CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), per-step "
                      f"thermal polls at a {E261.TEMP_EARLY_END:.0f}C "
                      f"margin, cooldown {E261.COOLDOWN_S:.0f}s (dispatch "
                      f"30-60), the {E261.TEMP_HARD:.0f}C never-past line "
                      "(dispatch 85) recorded to "
                      "runs/_envelope_log.jsonl tagged e280:<ARM>:<phase>",
            "trainings": "7 serial installs s400 (R1KR/R2KR/K1KM05 AdamW "
                         "(e261's driver VERBATIM; K1KM05 under the lr "
                         "rebind x1.8653) + S1K/S2K/S5K/S10K SGD-M at "
                         f"LR_STABLE {LR_STABLE:.13f}) + 1 missile "
                         "interleave (800 SGD-M opt steps); NO cons; NO "
                         "FREE",
        },
        "deviations": deviations,
        "builds_on": [
            "T255 / e272 (THE relocated edge: the committed AdamW ladder "
            "{1k 0.000435, 2k 0.026616, 5k 0.127096, 10k 0.264648} — "
            "RANK-WRITES-THE-CURVE with the bracket (1k,2k]; THIS cell's "
            "rooms bit-bound to e272_rooms.pt; the dose acquittal "
            "(K1KM 0.00125) that licenses the undermatched lr)",
            "T253/e273 + consult #006 (THE SGD-M calibration: LR_SGD_matched "
            "21.7386 (first-step applied in-room L2) DIVERGES; the x0.01 "
            "stable rider (SGD001X: concurrent, post 0.00253 rising at "
            "s400) is THIS cell's lr class — stable, disclosed undermatched; "
            "momentum 0.9 / wd 0.0 matched exactly)",
            "T254/e278 + T260 (THE ROACH MOTEL: Adam re-aims orthogonal "
            "gradients into the room — the missile's realized corpus "
            "displacement ran 48-60% in-room under AdamW with the gradient "
            "exactly orthogonal; P-x283b registered: < 20% under SGD-M iff "
            "the motel is Adam's architecture)",
            "T258 + W046 (the 2-parameter coordinate: fit (floor-bracket, "
            "exponent) per optimizer — a pure intercept shift says Adam "
            "sets the floor but the width-scaling is the space's; an "
            "exponent change says the optimizer shapes the whole climb)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT: the "
            "SRCT projector, the hooked drivers, the thermal envelope)",
            "T259 / e281 (the NO-CONS form: the landing read is a cons "
            "property; the bars read the WRITE only)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314)",
        ],
        "whats_new": [
            "THE OPTIMIZER SWAP ON THE WHOLE LADDER (the record's first): "
            "the committed expression ladder re-run under a non-Adam "
            "optimizer at the same rooms, same streams, same hook — the "
            "floor's provenance (space vs normalizer) made a one-delta "
            "experiment",
            "THE W046 COORDINATE MEASURED ON BOTH SIDES: (floor-bracket, "
            "exponent) fitted per optimizer on the same rungs — the verdict "
            "a point in a 2-parameter space instead of a binary",
            "THE ESCAPE TEST RUN (P-x283b): the missile's orthogonal corpus "
            "stream under SGD-M with the per-milestone displacement ledger "
            "— the roach motel's architecture question asked directly",
            "THE EDGE PAIR'S ROOM LOTTERY PRICED (the R66-critic debt): "
            "fresh 1k + 2k rooms under AdamW at the committed condition",
        ],
        "gates": {},
    })
    log(f"E280 — THE CAPACITY LADDER UNDER SGD-M (smoke={SMOKE}) -> {RD}")
    log(f"arms (serial, execution order): {' -> '.join(ARMS)}; LR_STABLE "
        f"{LR_STABLE:.13f} = {SGD_STABLE_FACTOR} x {LR_SGD_MATCHED} "
        f"(momentum {SGD_MOMENTUM}, wd {SGD_WD}); the 0.5x rider at lr x"
        f"{LR_SCALE_HALF:.13f}; missile at k=10k room {MISSILE_ROOM} "
        f"(corpus gen {CORPUS_GEN_SEED})")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(28000)                 # global init only; every RNG is its own

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
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank / install mask)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e261m = json.loads(E261_METRICS.read_text(encoding="utf-8"))
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e272m = json.loads(E272_METRICS.read_text(encoding="utf-8"))
    e273m = json.loads(E273_METRICS.read_text(encoding="utf-8"))
    lrcal = json.loads(E273_LRCAL.read_text(encoding="utf-8"))
    e261_k1k = e261m["arms"]["K1K"]["install"]["post_cells"]["g0"]
    e272_posts = {a: e272m["adjudication"]["reads"]["post_g0"][a]
                  for a in ("K2K", "K5K", "K10KR", "K1KM")}
    kept_curve_file = {int(k): v for k, v in
                       e264m["adjudication"]["reads"]["kept_frac_curve"]
                       .items()}
    post_curve_file = {int(k): v for k, v in
                       e264m["adjudication"]["reads"]["post_g0_curve"].items()}
    e273_sgd001x = e273m["arms"]["SGD001X"]["install"]["post_cells"]["g0"]
    G_PARENTS = {
        "e261_metrics": {"path": str(E261_METRICS),
                         "md5": md5of(E261_METRICS), "bound_md5": E261_MD5,
                         "K1K_post_g0": e261_k1k,
                         "note": "THE cited 1k rung (dead) — the AdamW "
                                 "ladder's bottom + the replicate rider's "
                                 "band center"},
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "K10K_post_g0": post_curve_file[10000],
                         "K40K_post_g0": post_curve_file[40000],
                         "kept_frac_curve": {str(k): v for k, v in
                                             sorted(kept_curve_file.items())},
                         "note": "the cited 10k rung + the kept curve that "
                                 "freezes the 0.5x compensation"},
        "e272_metrics": {"path": str(E272_METRICS),
                         "md5": md5of(E272_METRICS), "bound_md5": E272_MD5,
                         "verdict": e272m["adjudication"]["verdict"],
                         "posts": e272_posts,
                         "edge_bracket": e272m["adjudication"]["reads"]
                         ["edge_bracket"],
                         "note": "THE relocated edge (1k,2k] + the bit-bound "
                                 "rooms' source record"},
        "e273_metrics": {"path": str(E273_METRICS),
                         "md5": md5of(E273_METRICS), "bound_md5": E273_MD5,
                         "verdict": e273m["adjudication"]["verdict"],
                         "SGD001X_post_g0": e273_sgd001x,
                         "note": "the SGD-M lr calibration cell — the "
                                 "matched lr's divergence + the x0.01 "
                                 "stable rider (THIS cell's lr class)"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                 "md5": md5of(E273_LRCAL),
                                 "bound_md5": E273_LRCAL_MD5,
                                 "lr_sgd": lrcal["lr_sgd"],
                                 "note": "LR_STABLE's provenance record"},
        "e272_rooms": {"path": f"runs/checkpoints/{ROOMS272_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS272_CK),
                       "bound_md5": E272_ROOMS_MD5,
                       "note": "the four bit-bound ladder rooms' source"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "pass": bool(
            md5of(E261_METRICS) == E261_MD5
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E272_METRICS) == E272_MD5
            and md5of(E273_METRICS) == E273_MD5
            and md5of(E273_LRCAL) == E273_LRCAL_MD5
            and abs(e261_k1k - E261_K1K_POST) < 1e-12
            and abs(post_curve_file[10000] - E264_K10K_POST) < 1e-12
            and abs(post_curve_file[40000] - E264_K40K_POST) < 1e-12
            and abs(e272_posts["K2K"] - E272_K2K_POST) < 1e-12
            and abs(e272_posts["K5K"] - E272_K5K_POST) < 1e-12
            and abs(e272_posts["K1KM"] - E272_K1KM_POST) < 1e-12
            and e272m["adjudication"]["verdict"] == E272_VERDICT
            and md5of(CKPT_DIR / ROOMS272_CK) == E272_ROOMS_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e261 K1K {e261_k1k:.6f}; e272 "
        f"[{E272_VERDICT}] K2K {e272_posts['K2K']:.6f} K5K "
        f"{e272_posts['K5K']:.6f} K1KM {e272_posts['K1KM']:.6f}; e264 K10K "
        f"{post_curve_file[10000]:.6f}; e273 lrcal "
        f"{lrcal['lr_sgd']:.10f} (SGD001X post {e273_sgd001x:.6f})")
    write_partial("P0b parents hard-bound")

    # ---- G_LR_BIND (the SGD lr's provenance, exact) ----------------------
    G_LR_BIND = {
        "form": "the SGD-M lr := e273's STABLE RIDER POINT — LR_STABLE = "
                f"x{SGD_STABLE_FACTOR} x LR_SGD_matched, re-derived at "
                "runtime from the md5-bound runs/e273/lr_calibration.json "
                "and asserted == the frozen literal; the optimizer "
                "hyperparameters match e273's stable rider EXACTLY "
                f"(momentum {SGD_MOMENTUM}, wd {SGD_WD}); schedule "
                "LR_STABLE x cosine_lr(s-1, 1000)",
        "lr_cal_md5": md5of(E273_LRCAL), "bound_md5": E273_LRCAL_MD5,
        "lr_sgd_record": lrcal["lr_sgd"], "lr_sgd_frozen": LR_SGD_MATCHED,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "matched_lr_diverges": ("e273's committed record: the SGDM barrel "
                                "at LR_SGD_matched collapsed (0/nan); the "
                                "free corpus steps at the matched scale "
                                "ran the model away"),
        "undermatch_disclosed": ("the s1 applied in-room L2 is ~x0.01 of "
                                 "the matched target; e272 acquitted dose "
                                 "(K1KM 0.00125 dead) so the formation "
                                 "floor is testable at this scale; the "
                                 "per-rung first-step applied-L2 ledger is "
                                 "MEASURED"),
        "e273_SGD001X_post_g0": e273_sgd001x,
        "pass": bool(abs(lrcal["lr_sgd"] - LR_SGD_MATCHED) < 1e-9
                     and abs(LR_STABLE
                             - SGD_STABLE_FACTOR * LR_SGD_MATCHED) < 1e-15
                     and md5of(E273_LRCAL) == E273_LRCAL_MD5
                     and abs(float(lrcal["adamw_lr"]) - LR_BASE) < 1e-15),
    }
    assert G_LR_BIND["pass"], f"lr bind gate FAILED: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P0c G_LR_BIND: LR_STABLE = {SGD_STABLE_FACTOR} x "
        f"{lrcal['lr_sgd']!r} = {LR_STABLE!r}: PASS")
    write_partial("P0c G_LR_BIND PASSED (the SGD lr's provenance exact)")

    # ---- G_DOSE_ARITH (the 0.5x rider's compensation, exact) -------------
    lr_scale_rt = kept_curve_file[10000] / kept_curve_file[1000]
    G_DOSE_ARITH = {
        "form": "the 0.5x rider's compensation re-derived at runtime from "
                "the md5-bound committed kept_frac_curve and asserted == "
                "the frozen literal: LR_SCALE_HALF = 0.5 x kept(10k)/"
                "kept(1k); the compensated install's lr(s) = 1e-3 x "
                "LR_SCALE_HALF x cosine_lr(s-1, 1000) via e272's disclosed "
                "E43.LR module rebind (restored in a finally block)",
        "stored_kept_1k": kept_curve_file[1000],
        "stored_kept_10k": kept_curve_file[10000],
        "lr_scale_full_runtime": lr_scale_rt,
        "lr_scale_full_frozen": LR_SCALE_FULL,
        "lr_scale_half": LR_SCALE_HALF,
        "lr_compensated": LR_BASE * LR_SCALE_HALF,
        "pass": bool(abs(lr_scale_rt - LR_SCALE_FULL) < 1e-12
                     and abs(kept_curve_file[1000] - KEPT_1K_STORED) < 1e-15
                     and abs(kept_curve_file[10000] - KEPT_10K_STORED)
                     < 1e-15
                     and abs(LR_BASE - 1e-3) < 1e-15),
    }
    assert G_DOSE_ARITH["pass"], f"dose arithmetic gate FAILED: {G_DOSE_ARITH}"
    metrics["gates"]["G_DOSE_ARITH"] = G_DOSE_ARITH
    log(f"P0d G_DOSE_ARITH: LR_SCALE_HALF = 0.5 x {lr_scale_rt:.12f} = "
        f"{LR_SCALE_HALF:.12f}; the 0.5x rider's install lr "
        f"{LR_BASE * LR_SCALE_HALF:.9f}: PASS")
    write_partial("P0d G_DOSE_ARITH PASSED")

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
              "g0_volume_null_floor": G1.battery_cell(base_net, g0_ids,
                                                      zid)["mean_pz"],
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, CE_R {base_ce_r:.4f}): PASS")
    write_partial("P0e G-BASE PASSED")

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

    # ---- THE SIX ROOMS BUILT + CERTIFIED (the machinery's own gates) -----
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = Rooms280(N, SPEC, v64_np, Vp.numpy().astype(np.float64),
                     params_ref, dev)
    cert = rooms.certify280()
    G_PROJ = {
        "form": "the six rooms certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; each room's IDEMPOTENCY and "
                "kept^2 rank probe (||P x||^2/||x||^2 vs k/N, the 10-sigma "
                "bar 5*sqrt(2k)/N); each room's span-overlap (expect "
                "~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N) per room"},
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    for nm, r in cert["per_room"].items():
        log(f"  room {nm}: k {r['k']} (seeds {r['seeds']}) idem "
            f"{r['idempotency_max']:.1e} kept2 {r['kept2_mean']:.6f} vs "
            f"{r['kept2_expect']:.6f} (bar {r['kept2_bar_10sig']:.1e}) "
            f"span-ovl {r['span_overlap_mean']:.4f} "
            f"(expect ~{r['span_overlap_expect']:.4f})")

    # ---- G_ROOMS: the four ladder rooms bit-identical to e272's committed
    rooms272 = torch.load(CKPT_DIR / ROOMS272_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    bit_rows = {}
    if not SMOKE:
        for mine, theirs in ROOMS272_BIND.items():
            D272 = _to_np(rooms272["model"][theirs]["D_int8"]) \
                .astype(np.float64)
            S272 = _to_np(rooms272["model"][theirs]["S"])
            D_mine = rooms.rooms[mine].D
            S_mine = rooms.rooms[mine].S
            bit_rows[mine] = {
                "bound_to": f"e272_rooms.pt[{theirs}]",
                "k_committed": int(rooms272["model"][theirs]["k"]),
                "k_mine": rooms.room_k[mine],
                "D_bit_equal": bool(np.array_equal(D_mine, D272)),
                "S_bit_equal": bool(np.array_equal(S_mine, S272)),
            }
        G_ROOMS = {
            "form": "the four SGD ladder rooms == e272's committed rooms "
                    "(D/S exact equality; S1K is thereby also e261's "
                    "committed K1K room — e272's own bit-gate): the "
                    "optimizer is the SGD arms' ONLY delta vs the "
                    "committed AdamW ladder",
            "rows": bit_rows,
            "e272_rooms_md5": md5of(CKPT_DIR / ROOMS272_CK),
            "pass": bool(all(r["D_bit_equal"] and r["S_bit_equal"]
                             and r["k_committed"] == r["k_mine"]
                             for r in bit_rows.values())),
        }
    else:
        G_ROOMS = {
            "form": "SMOKE: the ladder rooms share e272's seed pairs at "
                    "smoke ks — no committed record at these ks; the "
                    "bit-bind is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
    del rooms272
    assert G_ROOMS["pass"], f"rooms bind failed: {G_ROOMS}"
    metrics["gates"]["G_ROOMS"] = G_ROOMS
    log(f"P1 G_ROOMS: the four ladder rooms "
        f"{('bit-identical to e272_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e280_rooms",
        {nm: {"D_int8": rooms.rooms[nm].D.astype(np.int8),
              "S": rooms.rooms[nm].S, "k": rooms.room_k[nm],
              "seeds": [rooms.rooms[nm].seed_d, rooms.rooms[nm].seed_s]}
         for nm, _, _, _ in SPEC},
        {"desc": "e280's six rooms (the flat basis is net.parameters() "
                 "order): S1K/S2K/S5K/S10K = e272's committed K1KM/K2K/"
                 "K5K/K10KR rooms VERBATIM (bit-gated); R1KR/R2KR = fresh "
                 "replicate rooms (the firing-pair lottery probe)",
         "spec": [[nm, k, sd, ss] for nm, k, sd, ss in SPEC], "n": N,
         "span_rank": rooms.r_span,
         "cert": {nm: {kk: vv for kk, vv in r.items()
                       if not isinstance(vv, list)}
                  for nm, r in cert["per_room"].items()}})
    metrics["rooms"] = {
        "spec": {"rule": "six rooms: S1K/S2K/S5K/S10K bit-bound to "
                         "e272_rooms.pt (the committed ladder's own rooms); "
                         "R1KR/R2KR fresh (the R66 firing-pair replicate); "
                         "the missile shares S10K",
                 "rooms": [{"name": nm, "k": k, "seeds": [sd, ss],
                            "k_fraction_of_N": k / N} for nm, k, sd, ss
                           in SPEC]},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE SIX ROOMS: {' + '.join(f'{nm}(k={rooms.room_k[nm]})' for nm, _, _, _ in SPEC)}"
        f": BUILT + CERTIFIED + (the four ladder rooms) BIT-BOUND")
    write_partial("P1 the six rooms built (parents bound + v-map loaded + "
                  "span loaded + certification + the four-room bit-bind)")

    # ================= P2-P4: THE ARMS (serial) =========================
    arms_rec: dict = {}
    med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None  # noqa: E731

    def post_reads(sd_install: dict, arm: str, mode: str,
                   extra: dict | None = None) -> dict:
        inst_net = G1.evl_load(sd_install)
        inst_cells = {"gm12": G1.battery_cell(inst_net, gm12_ids,
                                              zid)["mean_pz"],
                      "g0": G1.battery_cell(inst_net, g0_ids, zid)["mean_pz"],
                      "gp12": G1.battery_cell(inst_net, bat_ids[12],
                                              zid)["mean_pz"],
                      "ce_r": G1.ce_fixed_cpu(inst_net, *r_eval_xy)}
        d_inst = flat_params_cpu(inst_net) - base_flat
        load_inst = rooms.displacement_loads(d_inst, mode)
        d64 = d_inst.double().numpy().astype(np.float64)
        in_room_norm = float(np.linalg.norm(rooms.proj_of(mode, d64)))
        del inst_net
        return {"post_cells": inst_cells, "displacement_loads": load_inst,
                "in_room_disp_norm": in_room_norm,
                "disp_norm": float(np.linalg.norm(d64))}

    def run_adamw_arm(arm: str, mode: str, comp: bool) -> None:
        log("=" * 78)
        k_arm = rooms.room_k[mode]
        log(f"ARM-{arm} — AdamW, {mode} room (k={k_arm}, seeds "
            f"{rooms.rooms[mode].seed_d}/{rooms.rooms[mode].seed_s})"
            + (f" + install lr COMPENSATED x {LR_SCALE_HALF:.10f} "
               f"(0.5 x kept(10k)/kept(1k))" if comp
               else " at the family dose (lr 1e-3)"))
        if comp:
            # ---- the disclosed module rebind (e272's convention) ---------
            E43.LR = LR_BASE * LR_SCALE_HALF
            log(f"[dose05] E43.LR rebound {LR_BASE} -> {E43.LR} "
                f"(x{LR_SCALE_HALF:.10f}) for {arm}'s install ONLY")
        try:
            inst = E261.chunked_install(
                f"{arm}:inst", mode, G1.evl_load(base_sd), rooms,
                inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
                r_eval_xy, zid, inst_resume_path(arm), dev)
        finally:
            if comp:
                E43.LR = LR_BASE
                log(f"[dose05] E43.LR restored {E43.LR}")
        reads = post_reads(inst["sd"], arm, mode)
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        lr_mult = LR_SCALE_HALF if comp else 1.0
        arms_rec[arm] = {
            "desc": (f"AdamW {mode} room k={k_arm} (seeds "
                     f"{rooms.rooms[mode].seed_d}/"
                     f"{rooms.rooms[mode].seed_s}); e261's driver VERBATIM"
                     + (f"; install lr x{LR_SCALE_HALF:.10f} = 0.5 x "
                        "kept(10k)/kept(1k) — the dose-acquittal's third "
                        "lr point" if comp
                        else " — the R66 firing-pair replicate"
                        if arm in ("R1KR", "R2KR") else "")),
            "optimizer": "adamw",
            "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "chunk_table": inst["chunk_table"], "steps": E261.INST_STEPS,
                "lr_scale_applied": lr_mult,
                "resumed_final": bool(inst.get("resumed_final", False)),
                **reads,
            },
        }
        post_ck = save_ckpt(
            f"e280_{arm}_post", inst["sd"],
            {"desc": f"e280 ARM-{arm} install-final (NO CONS — T259/e281 "
                     f"form): e001 + Dmix s{E261.INST_STEPS} (gen "
                     f"{E261.FRESH_GEN}; {mode} room k={k_arm}"
                     + (f", lr x{LR_SCALE_HALF:.6f}" if comp else "")
                     + ") under AdamW (0.9,0.95) wd 0.1",
             "arm": arm, "mode": mode, "k": k_arm,
             "rooms": "runs/checkpoints/e280_rooms.pt"})
        arms_rec[arm]["post_checkpoint"] = post_ck
        log(f"ARM-{arm} DONE: post g0 {reads['post_cells']['g0']:.6f} "
            f"g-12 {reads['post_cells']['gm12']:.6f} CE_R "
            f"{reads['post_cells']['ce_r']:.4f} | d in-own-room "
            f"{reads['displacement_loads']['in_own_room']:.4f} ||P d|| "
            f"{reads['in_room_disp_norm']:.5f} ||d|| "
            f"{reads['disp_norm']:.5f} | kept med {med(led_kept):.4f}")
        metrics["arms"] = arms_rec
        write_partial(f"P2 ARM-{arm} (AdamW{' compensated' if comp else ''})"
                      " install + post reads")

    def run_sgd_arm(arm: str, mode: str) -> None:
        log("=" * 78)
        k_arm = rooms.room_k[mode]
        log(f"ARM-{arm} — SGD-M (momentum {SGD_MOMENTUM}, wd {SGD_WD}) at "
            f"LR_STABLE {LR_STABLE:.10f} x cosine, {mode} room (k={k_arm}, "
            f"seeds {rooms.rooms[mode].seed_d}/{rooms.rooms[mode].seed_s}) "
            f"— DISCLOSED UNDERMATCHED (x{SGD_STABLE_FACTOR} of matched)")
        inst = chunked_install_opt(
            f"{arm}:inst", mode, G1.evl_load(base_sd), rooms,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid, inst_resume_path(arm), dev, "sgd")
        reads = post_reads(inst["sd"], arm, mode)
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        arms_rec[arm] = {
            "desc": f"SGD-M {mode} room k={k_arm} (seeds "
                    f"{rooms.rooms[mode].seed_d}/{rooms.rooms[mode].seed_s})"
                    f"; the committed room VERBATIM — the optimizer is the "
                    f"ONLY delta vs e272's AdamW ladder",
            "optimizer": "sgd",
            "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "chunk_table": inst["chunk_table"], "steps": E261.INST_STEPS,
                "lr_stable": LR_STABLE, "lr_matched": LR_SGD_MATCHED,
                "stable_factor": SGD_STABLE_FACTOR,
                "momentum": SGD_MOMENTUM, "wd": SGD_WD,
                "first_step_applied_l2": inst["first_step"],
                "diverged": inst["diverged"],
                "resumed_final": bool(inst.get("resumed_final", False)),
                **reads,
            },
        }
        fs = inst["first_step"]
        if fs:
            log(f"  [{arm}] FIRST-STEP LEDGER: lr_s1 {fs['lr_s1']:.6f} | "
                f"|d1| {fs['full_l2']:.6f} (momentum-exact pred "
                f"{fs['momentum_exact_pred']:.6f}, rel err "
                f"{fs['momentum_exact_rel_err']:.2e}) | in-room "
                f"{fs['in_room_l2']:.6f} (x{SGD_STABLE_FACTOR} of the "
                "matched target — the undermatch MEASURED)")
        post_ck = save_ckpt(
            f"e280_{arm}_post", inst["sd"],
            {"desc": f"e280 ARM-{arm} install-final (NO CONS — T259/e281 "
                     f"form): e001 + Dmix s{E261.INST_STEPS} (gen "
                     f"{E261.FRESH_GEN}; {mode} room k={k_arm}) under "
                     f"SGD(momentum {SGD_MOMENTUM}, wd {SGD_WD}) at "
                     f"LR_STABLE {LR_STABLE:.12f}",
             "arm": arm, "mode": mode, "k": k_arm,
             "rooms": "runs/checkpoints/e280_rooms.pt"})
        arms_rec[arm]["post_checkpoint"] = post_ck
        log(f"ARM-{arm} DONE: post g0 {reads['post_cells']['g0']:.6f} "
            f"g-12 {reads['post_cells']['gm12']:.6f} CE_R "
            f"{reads['post_cells']['ce_r']:.4f} | d in-own-room "
            f"{reads['displacement_loads']['in_own_room']:.4f} ||P d|| "
            f"{reads['in_room_disp_norm']:.5f} ||d|| "
            f"{reads['disp_norm']:.5f} | kept med {med(led_kept):.4f} | "
            f"diverged={inst['diverged']}")
        metrics["arms"] = arms_rec
        write_partial(f"P3 ARM-{arm} (SGD-M rung) install + post reads + "
                      "the first-step applied-L2 ledger")

    def run_missile_arm() -> None:
        arm = "MISSILE_SGD"
        mode = MISSILE_ROOM
        k_arm = rooms.room_k[mode]
        log("=" * 78)
        log(f"ARM-{arm} — e278's MISSILE construction VERBATIM under ONE "
            f"SGD(momentum {SGD_MOMENTUM}, wd {SGD_WD}) at LR_STABLE "
            f"{LR_STABLE:.10f} x cosine — {mode} room (k={k_arm}); corpus "
            f"gen seed {CORPUS_GEN_SEED}; P-x283b's escape test")
        inst = chunked_missile_sgd(
            f"{arm}:interleave", G1.evl_load(base_sd), rooms, base_flat_np,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid, mode, inst_resume_path(arm), dev)
        reads = post_reads(inst["sd"], arm, mode)
        iv = [row["interval_in_room_frac"] for row in inst["disp_ledger"]
              if row["step"] in (100, 200, 300, 400)
              and row["interval_in_room_frac"] is not None]
        cum = [row["cum_in_room_frac"] for row in inst["disp_ledger"]
               if row["step"] in (100, 200, 300, 400)
               and row["cum_in_room_frac"] is not None]
        primary = med(iv) if iv else None
        arms_rec[arm] = {
            "desc": f"the P-x283b escape test: e278's missile (corpus "
                    f"gradients orthogonalized to the {mode} room, "
                    "verified) under SGD-M at the stable lr — the roach "
                    "motel's architecture question",
            "optimizer": "sgd",
            "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "corpus_ledger": inst["corpus_ledger"],
                "orth_ledger": inst["orth_ledger"],
                "disp_ledger": inst["disp_ledger"],
                "orth_max": inst["orth_max"],
                "orth_norm_ratio_first": inst["orth_norm_ratio_first"],
                "orth_norm_ratio_mean": inst["orth_norm_ratio_mean"],
                "chunk_table": inst["chunk_table"], "steps": E261.INST_STEPS,
                "lr_stable": LR_STABLE,
                "corpus_gen_seed": CORPUS_GEN_SEED,
                "missile_read_interval_median_t100_400": primary,
                "missile_read_cumulative": cum,
                "e278_adamw_band": list(E278_MISSILE_BAND),
                "resumed_final": bool(inst.get("resumed_final", False)),
                **reads,
            },
        }
        post_ck = save_ckpt(
            f"e280_{arm}_post", inst["sd"],
            {"desc": f"e280 ARM-{arm} install-final: the orthogonalized "
                     f"corpus stream under SGD-M at LR_STABLE; {mode} room "
                     f"k={k_arm}",
             "arm": arm, "mode": mode, "k": k_arm,
             "rooms": "runs/checkpoints/e280_rooms.pt"})
        arms_rec[arm]["post_checkpoint"] = post_ck
        log(f"ARM-{arm} DONE: post g0 {reads['post_cells']['g0']:.6f} | "
            f"THE MISSILE READ: interval in-room median (t100-400) "
            f"{primary if primary is not None else float('nan'):.4f} "
            f"(AdamW's committed band "
            f"{E278_MISSILE_BAND[0]:.4f}-{E278_MISSILE_BAND[1]:.4f}; "
            f"escape bar < {MISSILE_ESCAPE_BAR}, motel bar >= "
            f"{MISSILE_MOTEL_BAR}) | orth max {inst['orth_max']:.2e}")
        metrics["arms"] = arms_rec
        write_partial("P4 ARM-MISSILE_SGD (the escape test) + the "
                      "displacement ledger")

    # ---- EXECUTION -------------------------------------------------------
    run_adamw_arm("R1KR", "R1KR", comp=False)
    burst_cooldown("R1KR -> R2KR")
    run_adamw_arm("R2KR", "R2KR", comp=False)
    burst_cooldown("R2KR -> K1KM05")
    run_adamw_arm("K1KM05", "S1K", comp=True)
    burst_cooldown("K1KM05 -> S1K")
    run_sgd_arm("S1K", "S1K")
    burst_cooldown("S1K -> S2K")
    run_sgd_arm("S2K", "S2K")
    burst_cooldown("S2K -> S5K")
    run_sgd_arm("S5K", "S5K")
    burst_cooldown("S5K -> S10K")
    run_sgd_arm("S10K", "S10K")
    burst_cooldown("S10K -> MISSILE_SGD")
    run_missile_arm()

    # ================= P5: the ledgers ===================================
    def _fin(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return x if math.isfinite(x) else float("nan")

    def _form(x):     # 'did-not-form' value for ratio arithmetic only
        x = _fin(x)
        return 0.0 if math.isnan(x) else x

    sgd_post = {1000: _fin(arms_rec["S1K"]["install"]["post_cells"]["g0"]),
                2000: _fin(arms_rec["S2K"]["install"]["post_cells"]["g0"]),
                5000: _fin(arms_rec["S5K"]["install"]["post_cells"]["g0"]),
                10000: _fin(arms_rec["S10K"]["install"]["post_cells"]["g0"])}
    sgd_diverged = {a: bool(arms_rec[a]["install"].get("diverged", False))
                    for a in ARMS_SGD}
    first_step_table = {a: arms_rec[a]["install"].get(
        "first_step_applied_l2") for a in ARMS_SGD}
    metrics["sgd_ladder"] = {
        "posts": sgd_post, "diverged": sgd_diverged,
        "lr": {"lr_stable": LR_STABLE, "stable_factor": SGD_STABLE_FACTOR,
               "lr_matched": LR_SGD_MATCHED,
               "momentum": SGD_MOMENTUM, "wd": SGD_WD},
        "first_step_applied_l2": first_step_table,
        "adamw_ladder_committed": dict(sorted(ADAMW_LADDER.items())),
        "note": "the four SGD-M posts (the WRITE read) + the per-rung "
                "first-step applied-L2 ledger (the undermatch MEASURED: "
                "~x0.01 of the matched s1 in-room target)",
    }
    log("THE SGD-M LADDER: " + " -> ".join(
        f"{k}:{_form(v):.6f}" for k, v in sorted(sgd_post.items())))
    write_partial("P5 the SGD ladder + the first-step applied-L2 ledger")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    ks_sgd = sorted(sgd_post)
    pairs = []
    for i in range(len(ks_sgd) - 1):
        lo, hi = ks_sgd[i], ks_sgd[i + 1]
        ratio = _form(sgd_post[hi]) / max(_form(sgd_post[lo]), RATIO_DEN_FLOOR)
        pairs.append({"from_k": lo, "to_k": hi,
                      "post_lo": sgd_post[lo], "post_hi": sgd_post[hi],
                      "ratio": ratio,
                      "dead_side": bool(_form(sgd_post[lo]) < DEAD_BAR),
                      "fires": bool(ratio >= JUMP_BAR
                                    and _form(sgd_post[lo]) < DEAD_BAR)})
    firing = [p for p in pairs if p["fires"]]
    edge_bracket_sgd = (max(firing, key=lambda p: p["from_k"])["from_k"],
                        max(firing, key=lambda p: p["from_k"])["to_k"]) \
        if firing else None
    floor_cross_sgd = next((k for k in ks_sgd
                            if _form(sgd_post[k]) >= EXPRESS_FLOOR), None)
    p1k = _form(sgd_post[1000])
    p2k = _form(sgd_post[2000])
    pair_12k_ratio = p2k / max(p1k, RATIO_DEN_FLOOR)

    space_fires = bool(p1k < DEAD_BAR and pair_12k_ratio >= JUMP_BAR)
    edge_moved_2x = bool(edge_bracket_sgd is not None
                         and edge_bracket_sgd[0] >= 2 * ADAMW_BRACKET[0])
    adam_fires = bool(p1k >= EXPRESS_FLOOR or edge_moved_2x)
    nothing_forms = bool(max(_form(v) for v in sgd_post.values()) < DEAD_BAR)

    hard = {k: v for k, v in metrics["gates"].items()}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    # ---- THE RIDERS' ADJUDICATIONS --------------------------------------
    r1 = _fin(arms_rec["R1KR"]["install"]["post_cells"]["g0"])
    r2 = _fin(arms_rec["R2KR"]["install"]["post_cells"]["g0"])
    r1_in = bool(REPL_BAND[0] * E261_K1K_POST <= _form(r1)
                 <= REPL_BAND[1] * E261_K1K_POST)
    r2_in = bool(REPL_BAND[0] * E272_K2K_POST <= _form(r2)
                 <= REPL_BAND[1] * E272_K2K_POST)
    replicate_verdict = ("REPLICATE-IN-BAND" if (r1_in and r2_in) else
                         "REPLICATE-LOTTERY (a fresh room left the ~2x "
                         "band — disclosed; the edge pair's n=1 caveat "
                         "sharpens)")
    p05 = _fin(arms_rec["K1KM05"]["install"]["post_cells"]["g0"])
    dose05_verdict = "0.5X-DEAD" if _form(p05) < DEAD_BAR else \
        "0.5X-EXPRESSES (P-280b REFUTED — dose is NOT acquitted at " \
        "1.865x)"
    mil = arms_rec["MISSILE_SGD"]["install"].get(
        "missile_read_interval_median_t100_400")
    mil_v = _fin(mil) if mil is not None else float("nan")
    if math.isnan(mil_v):
        missile_verdict = "MIXED (no readable displacement ledger)"
    elif mil_v < MISSILE_ESCAPE_BAR:
        missile_verdict = "SGD-MISSILE-ESCAPES"
    elif mil_v >= MISSILE_MOTEL_BAR:
        missile_verdict = "MOTEL-IS-SPACE"
    else:
        missile_verdict = (f"MIXED between (in-room {mil_v:.4f} in "
                           f"[{MISSILE_ESCAPE_BAR}, {MISSILE_MOTEL_BAR}))")

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    else:
        div_txt = ""
        if any(sgd_diverged.values()):
            bad = [a for a, v in sgd_diverged.items() if v]
            div_txt = (f" [DIVERGENCE-DISCLOSED: {', '.join(bad)} went "
                       "non-finite — the undermatched-lr caveat's own "
                       "boundary; treated as did-not-form in the ratio "
                       "arithmetic only]")
        if space_fires:
            verdict = "SPACE-INTRINSIC"
            all_fires = ", ".join(f"{p['from_k']}->{p['to_k']} "
                                  f"({p['ratio']:.1f}x, dead side "
                                  f"{_form(p['post_lo']):.6f})"
                                  for p in firing)
            clause = (f"under SGD-M the edge HOLDS at (1k,2k]: 1k post g0 "
                      f"{p1k:.6f} < {DEAD_BAR} AND 2k post g0 {p2k:.6f} = "
                      f"{pair_12k_ratio:.1f}x above 1k (>= {JUMP_BAR:.0f}x)"
                      + (f"; firing pairs: {all_fires}" if all_fires else "")
                      + (f"; the floor {EXPRESS_FLOOR} first crossed at "
                         f"k={floor_cross_sgd}" if floor_cross_sgd
                         else "; the floor not yet crossed (the dead-side "
                              "jump located the edge)")
                      + " — the capacity floor is the space's physics "
                        "(the optimizer's normalizer swapped out and the "
                        "bracket did not move). P-280m CONFIRMED."
                      + div_txt)
        elif adam_fires:
            verdict = "ADAM-CREATED"
            if p1k >= EXPRESS_FLOOR:
                clause = (f"1k EXPRESSES under SGD-M (post g0 {p1k:.6f} >= "
                          f"{EXPRESS_FLOOR}) — the floor is Adam's "
                          "preconditioned geometry (SGD-M's in-room "
                          "accumulate forms the below-edge write)"
                          + div_txt)
            else:
                clause = (f"the edge MOVED by >= 2x: the SGD-M bracket is "
                          f"{edge_bracket_sgd[0]}->{edge_bracket_sgd[1]} "
                          f"(AdamW's committed bracket "
                          f"{ADAMW_BRACKET[0]}->{ADAMW_BRACKET[1]}) — the "
                          "floor is Adam's preconditioned geometry"
                          + div_txt)
        else:
            why = []
            if nothing_forms:
                why.append(f"INCONCLUSIVE-AT-THIS-LR (the named MIXED "
                           f"branch): the SGD-M arms failed to form at ALL "
                           f"rungs (max post g0 "
                           f"{max(_form(v) for v in sgd_post.values()):.6f} "
                           f"< {DEAD_BAR}) — the undermatched-lr caveat "
                           "BINDS (the stable lr is x0.01 of matched; the "
                           "matched lr diverges; the trajectories verbatim "
                           "+ the first-step applied-L2 ledger are the "
                           "cell's honest record)")
            elif DEAD_BAR <= p1k < EXPRESS_FLOOR:
                why.append(f"PARTIAL 1k EXPRESSION: S1K post g0 {p1k:.6f} "
                           f"sits in [{DEAD_BAR}, {EXPRESS_FLOOR}) — above "
                           "the dead bar, below the expression floor (the "
                           "edge slid toward or below the ladder's floor "
                           "without the floor's expression)")
            else:
                why.append("between the bars (no firing pair, 1k not "
                           "expressing — the letter's 'edge moves but "
                           "< 2x' branch or an unresolved ladder)")
            verdict = "MIXED"
            clause = ("; ".join(why)
                      + " — the trajectories verbatim, no inflation."
                      + div_txt)

    log("=" * 78)
    log(f"E280 VERDICT: {verdict}")
    log(f"  THE SGD-M LADDER (post g0): " + " -> ".join(
        f"{k}:{_form(v):.6f}" for k, v in sorted(sgd_post.items())))
    log(f"  the AdamW committed ladder: " + " -> ".join(
        f"{k}:{v:.6f}" for k, v in sorted(ADAMW_LADDER.items())))
    log(f"  firing pairs: "
        f"{[(p['from_k'], p['to_k'], round(p['ratio'], 1)) for p in firing]}"
        f"; SGD bracket {edge_bracket_sgd} (AdamW {ADAMW_BRACKET}); floor "
        f"crossing k={floor_cross_sgd}")
    log(f"  RIDER a: {replicate_verdict} (R1KR {r1:.6f} vs "
        f"{E261_K1K_POST:.6f}; R2KR {r2:.6f} vs {E272_K2K_POST:.6f})")
    log(f"  RIDER b: {dose05_verdict} (K1KM05 post {p05:.6f})")
    log(f"  RIDER c: {missile_verdict} (in-room median "
        f"{mil_v if not math.isnan(mil_v) else float('nan'):.4f})")
    log(f"  {clause}")
    log("=" * 78)

    # ---- THE W046 COORDINATE (the 2-parameter fit per optimizer) --------
    adamw_domain = {k: v for k, v in ADAMW_LADDER.items()
                    if k >= ADAMW_BRACKET[1]}
    adamw_fit = fit_power(adamw_domain)
    adamw_fit_40k = fit_power({**adamw_domain, 40000: E264_K40K_POST})
    if edge_bracket_sgd is not None:
        sgd_domain = {k: v for k, v in sgd_post.items()
                      if k >= edge_bracket_sgd[1]}
    else:
        sgd_domain = {k: v for k, v in sgd_post.items()
                      if _form(v) > DEAD_BAR} or \
            {k: v for k, v in sgd_post.items() if _form(v) > 0}
    sgd_fit = fit_power(sgd_domain)
    w046 = {
        "form": "the W046 coordinate: (floor-bracket, exponent) per "
                "optimizer — the exponent is the least-squares log-log "
                "slope of post(k) ~ A k^beta over the rungs at-or-above "
                "each optimizer's own bracket high side (positive posts); "
                "the primary fits share the domain {2k, 5k, 10k} whenever "
                "the SGD bracket is (1k,2k]",
        "adamw": {"floor_bracket": list(ADAMW_BRACKET),
                  "fit_domain": dict(sorted(adamw_domain.items())),
                  "exponent": adamw_fit["exponent"], "A": adamw_fit["A"],
                  "r2": adamw_fit["r2"],
                  "fit_domain_40k_variant": dict(
                      sorted({**adamw_domain, 40000: E264_K40K_POST}
                             .items())),
                  "exponent_40k_variant": adamw_fit_40k["exponent"],
                  "w046_note": "W046's 118x-width slope ~0.63 (2k-237k) is "
                               "the settled-tail slope; the 2k-10k local "
                               "slope is steeper (the curve is "
                               "convex-to-power early — W046's own "
                               "caveat)"},
        "sgdm": {"floor_bracket": (list(edge_bracket_sgd)
                                   if edge_bracket_sgd else None),
                 "fit_domain": dict(sorted(sgd_domain.items())),
                 "exponent": sgd_fit["exponent"], "A": sgd_fit["A"],
                 "r2": sgd_fit["r2"]},
        "reading": ("a pure bracket shift with the exponent intact says "
                    "Adam's geometry sets the floor but the width-scaling "
                    "is the space's; an exponent change says the optimizer "
                    "shapes the whole climb (W046's frozen fork)"),
    }
    metrics["w046_coordinate"] = w046
    log(f"THE W046 COORDINATE — AdamW: bracket {ADAMW_BRACKET}, exponent "
        f"{adamw_fit['exponent'] if adamw_fit['exponent'] is not None else 'n/a'}"
        f" (40k variant "
        f"{adamw_fit_40k['exponent'] if adamw_fit_40k['exponent'] is not None else 'n/a'}"
        f"); SGD-M: bracket {edge_bracket_sgd}, exponent "
        f"{sgd_fit['exponent'] if sgd_fit['exponent'] is not None else 'n/a'}")

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "riders_bars_verbatim": REGISTERED["riders_bars_verbatim"],
        "composite_order": "hard-gate failure (TEXTURE) -> SPACE-"
                           "INTRINSIC -> ADAM-CREATED -> MIXED (frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "sgd_post_g0": sgd_post,
            "adamw_post_g0_committed": dict(sorted(ADAMW_LADDER.items())),
            "adjacent_pairs": pairs,
            "firing_pairs": firing,
            "edge_bracket_sgd": (list(edge_bracket_sgd)
                                 if edge_bracket_sgd else None),
            "edge_bracket_adamw_committed": list(ADAMW_BRACKET),
            "edge_moved_2x": edge_moved_2x,
            "floor_cross_k_sgd": floor_cross_sgd,
            "nothing_forms": nothing_forms,
            "pair_1k_2k_ratio": pair_12k_ratio,
            "read_status": "the WRITE read (post g0 on the install-final "
                           "net); NO CONS (T259/e281 — the landing read is "
                           "a cons property; states checkpointed)",
        },
        "SPACE_INTRINSIC": space_fires,
        "ADAM_CREATED": adam_fires,
        "MIXED": bool(gates_pass and not space_fires and not adam_fires),
        "riders": {
            "replicate": {"verdict": replicate_verdict,
                          "R1KR_post_g0": r1, "R2KR_post_g0": r2,
                          "band": REPL_BAND,
                          "committed": {"1k": E261_K1K_POST,
                                        "2k": E272_K2K_POST},
                          "R1KR_in_band": r1_in, "R2KR_in_band": r2_in},
            "dose05": {"verdict": dose05_verdict, "post_g0": p05,
                       "lr_scale": LR_SCALE_HALF,
                       "P280b": ("confirmed (dead — dose acquitted at "
                                 "three lr points)" if _form(p05) < DEAD_BAR
                                 else "REFUTED")},
            "missile": {"verdict": missile_verdict,
                        "in_room_median": (mil_v if not math.isnan(mil_v)
                                           else None),
                        "interval_fracs": [row[
                            "interval_in_room_frac"]
                            for row in arms_rec["MISSILE_SGD"]["install"]
                            ["disp_ledger"]],
                        "cumulative_fracs": arms_rec["MISSILE_SGD"]
                        ["install"]["missile_read_cumulative"],
                        "bars": {"escape": MISSILE_ESCAPE_BAR,
                                 "motel": MISSILE_MOTEL_BAR},
                        "adamw_band_e278": list(E278_MISSILE_BAND),
                        "Px283b": ("confirmed (the motel is Adam's "
                                   "architecture)" if missile_verdict
                                   == "SGD-MISSILE-ESCAPES" else
                                   "refuted or mixed — see the verdict")},
        },
        "P280m": ("confirmed (the bracket held under SGD-M)" if space_fires
                  else "refuted or mixed — see the verdict"),
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars + the riders adjudicated")

    # ================= P9: honesty + provenance ==========================
    metrics["honesty"] = {
        "intervention_not_logits": ("the four SGD arms share bit-identical "
            "install streams (one generator, seed 24314, one draw order), "
            "the same fresh fact-free base, the same hook, and rooms "
            "BIT-IDENTICAL to e272's committed ladder (G_ROOMS) — the "
            "optimizer is the ONLY delta; the riders' fresh rooms are the "
            "lottery probe's object; fate differences are "
            "optimizer-caused or nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
            "g-series standing caveat); the firing-pair replicate prices "
            "the room lottery at the edge pair; nothing guaranteed"),
        "undermatch_measured_not_nominal": ("the stable lr is x0.01 of the "
            "matched calibration; each SGD rung's FIRST-STEP applied L2 "
            "(full + in-room + the momentum-exactness self-check) is "
            "MEASURED; if nothing forms the verdict is the named "
            "INCONCLUSIVE-AT-THIS-LR branch, never a silent pass"),
        "the_landing_read_not_run": ("NO CONS (T259/e281): the cons "
            "teaches from anything (0.6508 from a fact-free base); the "
            "bars read the WRITE only; every arm's post state is "
            "checkpointed for any later landing pass"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all branches and the "
            "trajectories are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": {"e261": str(E261.__file__),
                                    "e278_construction": "ported "
                                    "verbatim into this file "
                                    "(orthogonalize_grads + the missile "
                                    "driver)"},
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e272_rooms": f"runs/checkpoints/{ROOMS272_CK}",
            "rooms": rooms_ck,
            "arm_posts": {a: arms_rec[a]["post_checkpoint"] for a in ARMS},
            "arm_inst_resumes": {a: str(inst_resume_path(a).relative_to(
                E43.REPO)).replace("\\", "/") for a in ARMS},
        },
        "machinery": {
            "adamw_arms": "e261's chunked_install VERBATIM BY IMPORT (the "
                          "0.5x rider under the disclosed E43.LR rebind, "
                          "restored in a finally block)",
            "sgd_arms": "chunked_install_opt (this file) — e261's driver "
                        "with the optimizer constructor + lr base "
                        "parameterized + the first-step applied-L2 ledger",
            "missile": "chunked_missile_sgd (this file) — e278's "
                       "chunked_install_threenull MISSILE form with the "
                       "ONE SHARED optimizer = SGD-M at LR_STABLE",
            "hook": "e237's pre-step projection (backward -> clip 1.0 -> "
                    "project CPU fp64 SRCT -> write fp32 -> step; norm "
                    "not rescaled) on every install step",
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
    make_main_plot(RD, sgd_post, pairs, firing, edge_bracket_sgd,
                   floor_cross_sgd, w046, adamw_fit, sgd_fit, r1, r2, p05,
                   mil_v, replicate_verdict, dose05_verdict,
                   missile_verdict, verdict, clause)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e280_sgd_ladder.png"),
                          str(RD / "REPORT.md")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_main_plot(rd, sgd_post, pairs, firing, edge_bracket_sgd,
                   floor_cross_sgd, w046, adamw_fit, sgd_fit, r1, r2, p05,
                   mil_v, replicate_verdict, dose05_verdict,
                   missile_verdict, verdict, clause):
    """THE CELL'S HEADLINE FIGURE: the two ladders overlaid, the W046
    coordinate fit, the riders, the missile."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    N = GB.G1B_PARAMS
    c_adam = "dimgray"
    c_sgd = "#1a6faf"

    def _f(x):
        return x if (x is not None and math.isfinite(x)
                     and x > 0) else None

    # (0,0) THE TWO LADDERS OVERLAID
    ax = axes[0, 0]
    ak = sorted(ADAMW_LADDER) + [40000]
    av = [ADAMW_LADDER[k] for k in sorted(ADAMW_LADDER)] + [E264_K40K_POST]
    ax.plot(ak, av, "o-", ms=6, lw=1.5, color=c_adam,
            label="AdamW (committed ladder: e261/e272/e264)")
    sk = sorted(sgd_post)
    sv = [_f(sgd_post[k]) or 1e-7 for k in sk]
    ax.plot(sk, sv, "s-", ms=9, lw=1.8, color=c_sgd,
            label="SGD-M @ stable lr (this cell, e272's rooms bit-bound)")
    for k, v in zip(sk, sv):
        ax.annotate(f"{sgd_post[k]:.5f}", (k, v),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=c_sgd)
    ax.axhline(EXPRESS_FLOOR, ls=":", lw=1.4, color="crimson",
               label=f"the expression floor ({EXPRESS_FLOOR})")
    ax.axhline(DEAD_BAR, ls=":", lw=1.2, color="darkred", alpha=0.7,
               label=f"the dead bar ({DEAD_BAR})")
    ax.axvspan(ADAMW_BRACKET[0], ADAMW_BRACKET[1], color=c_adam,
               alpha=0.10, zorder=0)
    if edge_bracket_sgd:
        ax.axvspan(edge_bracket_sgd[0], edge_bracket_sgd[1],
                   color=c_sgd, alpha=0.15, zorder=0)
    ax.annotate("AdamW edge (1k,2k]", (math.sqrt(2e6), 2e-6),
                fontsize=8, color=c_adam, ha="center")
    if edge_bracket_sgd:
        ax.annotate(f"SGD-M edge {tuple(edge_bracket_sgd)}",
                    (math.sqrt(edge_bracket_sgd[0]
                               * edge_bracket_sgd[1]), 5e-3),
                    fontsize=8, color=c_sgd, ha="center", weight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("post g0 (the WRITE read, log)")
    ax.legend(fontsize=7.4, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fires_txt = ", ".join(f"{p['from_k']}->{p['to_k']} {p['ratio']:.0f}x"
                          for p in firing) if firing else "NONE"
    ax.set_title(f"THE TWO LADDERS — firing pairs (>= 10x, dead side < "
                 f"{DEAD_BAR}): {fires_txt}", fontsize=9.5)

    # (0,1) THE W046 COORDINATE FIT
    ax = axes[0, 1]
    ax.plot(ak, av, "o", ms=6, color=c_adam, label="AdamW rungs")
    ax.plot(sk, sv, "s", ms=8, color=c_sgd, label="SGD-M rungs")
    af = adamw_fit
    if af["exponent"] is not None:
        fx = np.array(sorted(ADAMW_LADDER), dtype=np.float64)
        ax.plot(fx, af["A"] * fx ** af["exponent"], "-", lw=1.4,
                color=c_adam, alpha=0.8,
                label=f"AdamW fit: beta={af['exponent']:.3f} "
                      f"(bracket {tuple(ADAMW_BRACKET)})")
    sf = sgd_fit
    if sf["exponent"] is not None:
        fx = np.array(sorted(sf["ks"]), dtype=np.float64)
        ax.plot(fx, sf["A"] * fx ** sf["exponent"], "-", lw=1.6,
                color=c_sgd,
                label=f"SGD-M fit: beta={sf['exponent']:.3f} "
                      f"(bracket {tuple(edge_bracket_sgd) if edge_bracket_sgd else 'none'})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("post g0 (log)")
    ax.legend(fontsize=7.4, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    exp_a = af["exponent"] if af["exponent"] is not None else float("nan")
    exp_s = sf["exponent"] if sf["exponent"] is not None else float("nan")
    ax.set_title(f"THE W046 COORDINATE — AdamW ({ADAMW_BRACKET}, "
                 f"{exp_a:.3f}) vs SGD-M "
                 f"({edge_bracket_sgd}, {exp_s:.3f}); intercept shift vs "
                 "exponent change", fontsize=9.0)

    # (1,0) THE RIDERS (a) + (b)
    ax = axes[1, 0]
    labels = ["R1KR\n(fresh 1k)", "R2KR\n(fresh 2k)", "K1KM05\n(lr x1.865)"]
    vals = [max(_f(r1) or 0.0, 1e-7), max(_f(r2) or 0.0, 1e-7),
            max(_f(p05) or 0.0, 1e-7)]
    bars_ = ax.bar(labels, vals, color=["#e67e22", "#e67e22", "crimson"],
                   alpha=0.85, width=0.55)
    # the ~2x bands (riders a) + the committed references
    ax.errorbar([0], [E261_K1K_POST],
                yerr=[[E261_K1K_POST * (1 - REPL_BAND[0])],
                      [E261_K1K_POST * (REPL_BAND[1] - 1)]],
                fmt="D", color="k", capsize=5, ms=7,
                label=f"committed 1k {E261_K1K_POST:.6f} (+-2x band)")
    ax.errorbar([1], [E272_K2K_POST],
                yerr=[[E272_K2K_POST * (1 - REPL_BAND[0])],
                      [E272_K2K_POST * (REPL_BAND[1] - 1)]],
                fmt="D", color="k", capsize=5, ms=7,
                label=f"committed 2k {E272_K2K_POST:.6f} (+-2x band)")
    ax.axhline(DEAD_BAR, ls=":", lw=1.2, color="darkred",
               label=f"the dead bar ({DEAD_BAR})")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.2)
    ax.set_ylabel("post g0 (log)")
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.set_title(f"THE RIDERS — a: {replicate_verdict} | b: "
                 f"{dose05_verdict} (post {p05:.6f})", fontsize=9.0)

    # (1,1) THE MISSILE (P-x283b)
    ax = axes[1, 1]
    dl = metrics["arms"]["MISSILE_SGD"]["install"]["disp_ledger"]
    xs = [row["step"] for row in dl]
    ys = [row["interval_in_room_frac"] if row["interval_in_room_frac"]
          is not None else float("nan") for row in dl]
    yc = [row["cum_in_room_frac"] if row["cum_in_room_frac"] is not None
          else float("nan") for row in dl]
    ax.plot(xs, ys, "o-", ms=5, lw=1.5, color=c_sgd,
            label="SGD-M missile: INTERVAL in-room frac (the read)")
    ax.plot(xs, yc, "s--", ms=4, lw=1.2, color=c_sgd, alpha=0.6,
            label="SGD-M missile: cumulative in-room frac")
    ax.axhspan(E278_MISSILE_BAND[0], E278_MISSILE_BAND[1],
               color="dimgray", alpha=0.25,
               label="AdamW's committed band (e278: 0.48-0.60)")
    ax.axhline(MISSILE_ESCAPE_BAR, ls="--", color="seagreen", lw=1.4,
               label=f"escape bar ({MISSILE_ESCAPE_BAR})")
    ax.axhline(MISSILE_MOTEL_BAR, ls="--", color="crimson", lw=1.4,
               label=f"motel bar ({MISSILE_MOTEL_BAR})")
    if _f(mil_v):
        ax.annotate(f"median {mil_v:.4f}", (xs[len(xs) // 2], mil_v),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=c_sgd, weight="bold")
    ax.set_xlabel("interleave step")
    ax.set_ylabel("in-room share of realized corpus displacement")
    ax.set_ylim(-0.02, 0.85)
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_title(f"THE SGD-MISSILE (P-x283b): {missile_verdict}",
                 fontsize=9.5)

    # the verdict strip
    fig.suptitle(f"E280 — THE CAPACITY LADDER UNDER SGD-M -> {verdict} "
                 f"| riders: {replicate_verdict}; {dose05_verdict}; "
                 f"{missile_verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e280_sgd_ladder.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
