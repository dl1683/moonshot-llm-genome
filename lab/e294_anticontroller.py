"""E294 — THE ANTI-CONTROLLER CELL (targeted forgetting / machine
unlearning — the kill law inverted), carrying THE E305 LANDAUER LEDGER
RIDER (batched onto this cell by the compute directive: "riders batched:
e294 carries e305's Landauer ledger"). This docstring carries the
registered question + bars VERBATIM from the dispatch letter + every
frozen convention, committed at birth BEFORE any compute. Adjudicate
against exactly this; no bar shopping.

THE QUESTION (verbatim): "can ONE fact be surgically erased while the
organism stays healthy and the OTHER facts hold? The lab's kill law
(THE_LAWS_V2.md) is the instrument; the anti-controller inverts the
error gate."

THE CONTEXT (the instrument, hard-bound below): LAW 2(b) TRANSPORT — the
read dies at a few parts in a thousand of orthogonal drift, the threshold
bracket (0.0710%, 0.3233%] of the write's norm (e290 TIGHT-CONSTANT) —
and LAW 4 — the error-gated controller PRESERVES by dosing on the read's
DEFICIT (e288 ERROR-GATED-HOLDS x3.4792). THE ANTI-CONTROLLER inverts
LAW 4's gate: dose on the read's EXCESS — the controller's medicine run
backwards, as a scalpel. T270/e291: the five-fact FAMILY shares one
representation (the antibodies are the same antibody — cross-cosines
0.834-0.952, one controller free-rides its four siblings x1.8-2.4);
e307: the controller's footprint is ONE reusable ~73-dim BRUSH
(consecutive-step cos 0.98, medium-rank, 16x below chance in-room) — the
anti should invert the brush; e306: MASS-IS-NOT-MEMORY — the read's
bearer is the room-overlap TAIL spectrum (a candidate erase target); the
anti is free to find it (UNPROJECTED).

THE FAMILY DISCLOSURE (frozen, verbatim in intent from the dispatch):
THE SELECTIVITY TEST IS WITHIN-FAMILY. The vehicle is the e291 five-fact
FAMILY organism — five 12-window context groups, ALL bound to the SAME
NAME (ZEPHYRA), five exactly-orthogonal rank-10k rooms. e293 proved five
DISTINCT-name facts are NOT serially installable by the protocol (the
contention is at FORMATION: each distinct install erodes the priors below
the live bar — only the family construction yields a five-live-fact
organism). Erasing one sibling while four hold is therefore the HARDEST
selectivity test the lab can currently construct — the siblings SHARE
their representation (T270), so collateral is the expected failure mode,
and a clean erase would be the STRONGEST form of the result. The bars
below are read against the e291 family baselines; the within-family
scope is disclosed everywhere the verdict is written.

THE DESIGN (the dispatch's dials, frozen):
  * THE ORGANISM := e291's committed five-fact family organism, LOADED
    BIT-EXACT (runs/checkpoints/e291_organism.pt; flat-md5 + checkpoint-
    md5 + behavioral panel gated THREE ways — the e288 G_FACTLOAD
    convention applied to the family vehicle; the installs are NOT re-run:
    extend, don't repeat). The five rooms rebuilt from the frozen seeds
    (D 29111 / perm 29112, five disjoint sorted 10k sets) + certified +
    BIT-gated vs e291_rooms.pt.
  * THE ANTI-CONTROLLER on FACT3: dose on the read's EXCESS —
        lr_anti_t = LR_ANTI_MAX * min(1, read_t / 0.05)
    (full dose when alive at read >= 0.05, vanishing when dead — the
    controller's law run backwards); the ANTI-gradient = ASCENT on
    fact3's name-CE (loss_anti := -(name-only CE) on FACT3's own
    12-window pool, ix(16) draws, 112 masked name tokens, mean token CE;
    backward -> clip 1.0 -> UNPROJECTED — the anti may go anywhere,
    in-room or out — through a SEPARATE anti-buffer opt_A, SGD-M momentum
    0.9 wd 0, private persistent, bidirectionally bitwise-isolated from
    the corpus side around EVERY optimizer event).
  * THE ERASE BUDGET B_E := the maintenance form's B_M = 0.40 x BUDGET
    (1.8357686531744601); cadence M=25 -> exactly 16 anti events; the
    per-event cap e288's equal-share form over B_E
    (cap_e = share_e/||b_e||, b_e = 0.9*buf_A + g_ascent the exact
    pending in-step momentum). LR_ANTI_MAX := B_E / 81.57360134901982
    (sum of e287's 16 committed b_m norms — the maintenance calibration's
    own denominator) = 0.02250444534525284 (== e288's LR_M_MAX literal:
    B_E == B_M and one controller — the calibration transfers EXACTLY).
  * THE CORPUS STREAM 1:1 (the family's standard, e268's form, union-
    orthogonalized, B_C = 0.60 x BUDGET) — the organism stays ALIVE in
    traffic while the anti erases; the CE clause of the bars reads it.

THE ARMS (verbatim from the dispatch):
  (a) FIVE-FACT + ANTI-ON-FACT3: the family organism + the corpus stream
      + the anti-controller's 16 events on FACT3.
  (b) THE NO-ANTI TWIN: the SAME organism + the SAME bit-identical corpus
      stream, NO anti events — the passive counterfactual (the arms' ONLY
      delta: the anti). The twin separates the anti's collateral from the
      passive decay (e290's law predicts the passive side erodes too).

FROZEN BARS (verbatim from the dispatch):
  - SURGICAL-ERASE: "FACT3 < 0.01 while >= 3 of 4 siblings hold >= 0.5x
    and the CE improves/holds — the first engineered forgetting"
  - COLLATERAL: "FACT3 dies but siblings fall — the shared representation
    bleeds; the honest within-family reading"
  - IMMUNE: "FACT3 refuses — the read resists ascent; a stability law"
  - MIXED: "anything else."
e305 RIDER BARS (never the primary): "PRICED-AT-THE-CONSTANT /
DIRECTED-CHEAPER (a Maxwell demon of forgetting) / ERASURE-COSTS-MORE
(immunity priced)".

==== THE FROZEN CONVENTIONS (picked + frozen HERE at birth) ============

* THE ERASE BAR := FACT3's t400 g0 battery read < 0.01 ABSOLUTE (the
  dispatch's literal; FACT3's loaded baseline 0.2678 — the bar sits at
  0.037x baseline). "the CE improves/holds" := the family's stream-live
  form VERBATIM (median corpus-batch CE rows t in (300,400] vs [1,100];
  improving := late < early; stable := late <= 1.05 x early).
* THE SIBLING-HOLD CLAUSE (the cell's one picked operationalization,
  disclosed): the dispatch's ">= 3 of 4 siblings hold >= 0.5x" is read
  TWO ways and BOTH must agree for SURGICAL/COLLATERAL: (i) ABSOLUTE —
  read_i(t400) >= 0.5 x BASELINE_i (the family's e291 HOLD form); (ii)
  TWIN-RELATIVE — read_i(t400) >= 0.5 x read_i^TWIN(t400) (the
  anti-corrected reading; the arms' difference is the registered object,
  and e290's law hard-bound below predicts the ABSOLUTE denominator
  confounds passive erosion with anti collateral). If (i) and (ii)
  DISAGREE (one >= 3, the other <= 2) the composite routes to MIXED —
  the honest within-family reading, both tables shown. IMMUNE :=
  FACT3(t400) >= 0.5 x BASELINE_3 (the read held at least half its
  loaded baseline DESPITE 16 ascent events — the organism defended the
  fact); the eroded middle (0.01 <= read < 0.5x baseline) is MIXED.
* THE COMPOSITE (frozen): TEXTURE (any hard bind/isolation gate failure
  — HALT, nothing adjudicated) -> MIXED-by-conditions (G_BUDGET blown OR
  G_ORTH failed OR the stream not live on arm (a) — no bar covers a
  broken standing condition; everything verbatim) -> SURGICAL-ERASE
  (FACT3 < 0.01 AND BOTH sibling readings >= 3 of 4) -> COLLATERAL
  (FACT3 < 0.01 AND BOTH sibling readings <= 2 of 4) -> IMMUNE (FACT3
  >= 0.5 x baseline_3) -> MIXED (else — including the disagreement case
  and the eroded middle).
* THE e305 LANDAUER LEDGER (the erasure thermodynamic table, frozen):
  WORK := the anti stream's REALIZED DRIFT ||anti_cum|| at the erase
  instant t_erase (the first post-anti panel read with FACT3 < 0.01; at
  t400 if never — the immunity case priced) — directly comparable to
  e290's bracket, which is in realized-drift terms; SPENT := S_anti (the
  sum of realized step norms, the maintenance-form accounting)
  co-reported (>= WORK always; the anti's steps are momentum-coherent).
  THE PASSIVE THRESHOLD PRICE := e290's committed bracket
  [0.0007099904808640174, 0.0032331212727527057] x ||F3WRITE|| where
  F3WRITE := flat(e291_install_F3 end state) - flat(e291_install_F2 end
  state), both loaded BIT-EXACT from the committed install checkpoints
  (FACT3's own install write — the e290 law's denominator form). WASTE
  HEAT := the bystanders' absorbed read-loss vs the twin at t400 (per
  sibling: read_i^TWIN - read_i, in read units, plus the twin-relative
  ratios) — the erase's collateral bill. Co-reports: the twin's own
  realized drift from the organism (the in-cell passive reference) and
  the same bracket scaled by the ORGANISM write norm (19.270036448332284
  — the alternative denominator, disclosed). RIDER VERDICT: DIRECTED-
  CHEAPER := WORK < price_lo (the aimed erase cheaper than the cheapest
  passive kill — a Maxwell demon of forgetting); PRICED-AT-THE-CONSTANT
  := price_lo <= WORK <= price_hi (a method-independent Landauer
  number); ERASURE-COSTS-MORE := WORK > price_hi (immunity priced).
* THE CORPUS STREAM: e268's registered form VERBATIM (48 windows = 16
  original-host anchors + 32 random corpus windows; full-window CE;
  lr = LR_STABLE x cosine_lr(t-1, 1000); clip 1.0), THIS cell's ONE
  fresh registered generator seed 29401 (the family's per-cell rule:
  .../28801/29101/HERE 29401); BOTH arms draw the IDENTICAL sequence;
  the gradient ORTHOGONALIZED against the five-room union (the exact
  shared-frame projector; residual verified EVERY step, bar 1e-6); the
  per-step cap over B_C (e288's equal-share form); SGD-M 0.9 wd 0
  through opt_C. The ANTI stream has its OWN registered generator, seed
  29402 (disclosed; the corpus stream untouched).
* THE ANTI TRACE (registered reads): per event — the gate read, the
  dose, the name-CE, |g|, the anti gradient's in-own-room + in-union
  fractions (the brush-inversion read: T270/T272/e307), lr_target, cap,
  binder, realized step, b_e, S_anti, buf_A's room fractions; the FULL
  five-fact panel read after EVERY anti event (the collateral
  microscope); all five facts' reads at t100/200/300/400 (post-anti);
  the budget ledger; the corpus CE; t_erase + WORK at the instant.
* HARD GATES (a failure HALTs): {G_NAMEFREE, G_SPLICE, G_BATTERY,
  G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_FACTSPLIT, G_PARENTS, G_BASE,
  G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMS5 (incl. the bit-bind vs
  e291_rooms.pt), G_FACTLOAD (the five fact-loads: ckpt md5 + flat md5 +
  panel g0 2e-6 / gm12 1e-5 vs the committed baselines), G_F3WRITE (the
  install checkpoints + FACT3's write norm), G_ASCENT (THE
  ASCENT-DIRECTION GATE: one full-dose anti step on a scratch copy of
  the organism LOWERS FACT3's read — the direction verified before any
  phase; the one-stroke panel deltas recorded), G_CORPUSGEN, G_LR_BIND,
  G_ANTIBIND (the anti's law arithmetic + exactly 16 counted events),
  G_BUFSEP-isolation (the anti-buffer separation, bidirectional bitwise
  around EVERY optimizer event), G_BUDGETLEDGER (the ledger-integrity
  gate: the per-stream S accounting machine-consistent with the realized
  displacements)}; NON-HALTING (route the composite to MIXED): {G_ORTH,
  G_BUDGET (the S_total <= BUDGET triangle-blow — an implementation
  outcome to autopsy), bufC-composition}.
* NO CONS (T259/e281; the committed form): the frozen bars read the
  WRITE and the DISPLACEMENT only; the organism + both arms' post states
  CHECKPOINTED for any later landing pass.
* Smoke (E294_SMOKE=1): 8 corpus steps/arm, M=2 (4 anti events),
  milestones 1..8, rooms k=512 at the same seeds (G_ROOMS5's bit-bind vs
  e291_rooms.pt VACUOUS at smoke k — disclosed; the REAL organism still
  loaded + panel-gated: G_FACTLOAD live), budgets scaled per-stream
  (B_C x 8/400; B_E x 4/16; LR_ANTI_MAX NOT scaled — e288's disclosed
  smoke convention), the ascent gate LIVE on the real organism, the
  anti-event full path exercised (dose gate, cap, isolation, panel), all
  paths smoke_-prefixed in its own dir; NOTHING adjudicated (SMOKE stamp
  on every read).

REGISTERED PREDICTIONS (the executor's, frozen at birth):
  - P-e294a (the demon): the aimed ascent erases FACT3 fast (<= 8 of the
    16 events; the gate read IS the battery's own rows — the anti
    maximizes the CE of exactly the distribution the bars read) and
    CHEAP: WORK < 0.0032 x ||F3write|| ~ the passive price — DIRECTED-
    CHEAPER (a Maxwell demon of forgetting).
  - P-e294b (the shared-antibody bleed, T270 inverted): the bystanders
    fall WITH FACT3 — the family's one-representation geometry means the
    anti's stroke (one reusable brush, e307) hits the siblings too;
    COLLATERAL with the twin-relative count <= 2; the anti gradient
    rides OUT of its own room (in-own-room frac small — the bearer is
    the overlap tail, e306).
  - P-e294c (the passive confound): the NO-ANTI twin's five reads erode
    substantially (e290's law at B_C-scale orthogonal spend — 2.75 of
    displacement against a per-fact threshold price of ~0.005-0.021):
    the twin-relative sibling clause is what keeps the adjudication
    honest; the ABSOLUTE clause may fail on passive decay alone — the
    disagreement route to MIXED is the honest outcome if so.
  - P-e294d (the immunity alternative): if the name-CE ascent saturates
    (the softmax floor: p(Z) -> 0 drives CE growth -> vanishing gradient
    through fp32), the read may stall above 0.01 with the dose pinned at
    the taper — IMMUNE or MIXED with the anti trace's dose/CE stalling
    the discriminating datum.

COMPUTE ENVELOPE: 2,739,072 params (<=100M free tier); bursts <= 175s
(dispatch 180), 40s cooldowns (dispatch 30-60), the 78C early-end
margin, the 84C never-past line (dispatch 85), per-step thermal polls
persisted to runs/_envelope_log.jsonl tagged e294:<ARM>:<phase>; CPU
fp64 dense projections (pocketfft workers 2); CPU probing threads 4; NO
concurrent GPU jobs (the arms strictly serial with cooldowns).
TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/e294/{metrics.json (PROGRESSIVE), e294_anticontroller.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e294_*.pt (gitignored; md5s in metrics). NO NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit +
push per phase.

Run:  cd lab && python e294_anticontroller.py    (E294_SMOKE=1 shakedown)
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
import e261_rank_ladder as E261                        # noqa: E402 — the burst
                                                      # machinery + binds (the
                                                      # committed file NOT
                                                      # modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E294_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e294_smoke" if SMOKE else "e294"
assert torch.cuda.is_available(), "e294 owns the GPU lane (dispatch)"

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

# the shared instrument ledgers (defined BEFORE the rebinding below)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e268/e273/e278/e288/e291's disclosed convention): the
# burst machinery resolves its globals through e261's namespace — rebound
# HERE so the launch gates + thermal polls label THIS cell (tags
# e294:<ARM>:<phase> per the dispatch).
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
E261.thermal_log = thermal_log
E261.device_events = device_events

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
CKPT_DIR = GB.CKPT_DIR

# ---- THE FAMILY ORGANISM (loaded, NOT rebuilt — the e288 G_FACTLOAD form)
ORG_CK = "e291_organism.pt"       # e291's committed five-fact organism
ORG_CK_MD5 = "ee2bad6be9f55fd94ebf3a367967da30"
ORG_FLAT_MD5 = "876907be13dc0f08412d5ffa36e4d57a"
ROOMS291_CK = "e291_rooms.pt"     # e291's committed rooms artifact (the bind)
ROOMS291_CK_MD5 = "163f90a1edcfbd1dbc44bfd72296d5cb"
INST_F2_CK = "e291_install_F2_resume.pt"   # FACT2's install end state
INST_F2_CK_MD5 = "8b6b0ba384a2b47261569c2c04d22aaf"
INST_F3_CK = "e291_install_F3_resume.pt"   # FACT3's install end state
INST_F3_CK_MD5 = "51c90f35771c5876dafcd5006e2c06d9"

# ---- THE FIVE ROOMS (e291's frozen seeds — the family's own frame) -------
ROOM_K = 10_000 if not SMOKE else 512
N_ROOMS = 5
ROOM_D_SEED = 29111               # the ONE fresh +-1 diagonal
ROOM_S_SEED = 29112               # the ONE fresh permutation (50k -> 5 x 10k)
FACTS = tuple(f"FACT{i}" for i in range(1, N_ROOMS + 1))
TARGET_FACT = "FACT3"             # THE ERASE TARGET (the dispatch's pick)
TARGET_IDX = 2                    # zero-based

ARMS = ("ANTI-ON-FACT3", "NO-ANTI")
ANT_ARM, TWN_ARM = ARMS

# ---- THE FACTS' SPLICE SPLIT (e291's frozen convention) ------------------
GROUP_SIZE = 12

# THIS cell's ONE fresh registered corpus stream (the family's per-cell
# rule: .../28801/29101/HERE 29401) — BOTH arms draw the IDENTICAL
# sequence. The ANTI stream: seed 29402, its own generator.
CORPUS_GEN_SEED = 29401
ANTI_GEN_SEED = 29402

PHASE_STEPS = 8 if SMOKE else 400               # corpus steps per arm
MILESTONES = tuple(range(1, 9)) if SMOKE else (100, 200, 300, 400)
MAINT_EVERY = 2 if SMOKE else 25                # the anti cadence M
N_ANTI_EXPECTED = PHASE_STEPS // MAINT_EVERY    # 16 full / 4 smoke events

# ---- THE SGD-M / LR CONFIG (e273's committed classes; md5-bound) --------
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
E273_LR_SGD = 21.7385748014537         # the calibration record's literal
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * E273_LR_SGD   # 0.21738574801453703

# ---- THE BUDGET (the maintenance form's, frozen) --------------------------
BUDGET_FRAC = 0.5                 # "0.5 x the write's own norm" (e288's form)
BUDGET_SLACK = 1e-5
E283_WRITE_NORM = 9.1788432658723            # the committed write's norm
BUDGET_NORM_FROZEN = 0.5 * E283_WRITE_NORM   # 4.58942163293615 (== e288/e291)
CORPUS_BUDGET_SHARE = 0.60        # B_C (the corpus stream 1:1 — the family's)
ERASE_BUDGET_SHARE = 0.40         # B_E := the maintenance form's B_M
assert abs(ERASE_BUDGET_SHARE + CORPUS_BUDGET_SHARE - 1.0) < 1e-12

# the anti-controller's calibration (the maintenance form's own denominator)
E287_B_M_SUM = 81.57360134901982    # sum of e287's 16 committed b_m norms
LR_M_MAX_E288 = 0.02250444534525284  # e288's frozen controller ceiling
LR_ANTI_MAX_FROZEN = (ERASE_BUDGET_SHARE * BUDGET_NORM_FROZEN
                      / E287_B_M_SUM)
assert abs(LR_ANTI_MAX_FROZEN - LR_M_MAX_E288) < 1e-15  # B_E == B_M: exact

# ---- THE ANTI'S LAW (frozen) ----------------------------------------------
DOSE_FULL_LEVEL = 0.05            # read >= this -> full dose; -> 0 as read -> 0
ERASE_BAR = 0.01                  # FACT3 < this at t400 := ERASED (absolute)
SURVIVE_FRAC = 0.5                # a sibling HOLDS: >= 0.5x its denominator
SIBLING_MIN_HOLD = 3              # ">= 3 of 4 siblings"
STREAM_STABLE_TOL = 1.05          # "stable": late CE <= 1.05 x early
ORTH_BAR = 1e-6                   # the union orthogonality gate
BUFSEP_ORTH_BAR = 1e-4            # bufC composition (fp-floor bar)
FACT_READ_TOL_G0 = 2e-6           # the family's cross-session read law
FACT_READ_TOL_GM12 = 1e-5
VOCAB_EXPECT = 65

# ---- e290's committed bracket (THE_LAWS_V2 LAW 2(b) — the passive price) --
E290_DRIFT_BRACKET = (0.0007099904808640174, 0.0032331212727527057)
E290_BUDGET_BRACKET = (0.0008, 0.004)
E291_ORG_WRITE_NORM = 19.270036448332284         # the organism write (co-scaling)

# ---- e291's committed baselines (the family's own frozen table) ----------
E291_BASELINES_G0 = {
    "FACT1": 0.26670998334884644,
    "FACT2": 0.36356988549232483,
    "FACT3": 0.2677536904811859,
    "FACT4": 0.28865179419517517,
    "FACT5": 0.3815295398235321,
}
E291_BASELINES_GM12 = {
    "FACT1": 0.10490661859512329,
    "FACT2": 0.13552233524854597,
    "FACT3": 0.08752423524856567,
    "FACT4": 0.11411947747005692,
    "FACT5": 0.14677566289901733,
}
E291_HELD_T0 = 0.2676781713962555

# ---- the parents, HARD-BOUND (md5s read at runtime; Rule 12) ------------
E291_METRICS = E43.REPO / "runs" / "e291" / "metrics.json"
E291_MD5 = "fc9cb859f2ef2603397c971dbcf5c618"
E291_VERDICT = "PEACEFUL-COEXISTENCE"

E290_METRICS = E43.REPO / "runs" / "e290" / "metrics.json"
E290_MD5 = "4ca36a46b04d0045faa833f4f9a49d1f"
E290_VERDICT = "TIGHT-CONSTANT"

E288_METRICS = E43.REPO / "runs" / "e288" / "metrics.json"
E288_MD5 = "31bf8df55b51c8a19c48a155388050be"
E288_VERDICT = "ERROR-GATED-HOLDS"

E287_METRICS = E43.REPO / "runs" / "e287" / "metrics.json"
E287_MD5 = "b7d18b8b286352732087b2ea99af3d2a"
E287_VERDICT = "SAWTOOTH-CONFIRMED"

E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "eb9f624708bcb6576c4115161dfd7042"
E272_VERDICT = "RANK-WRITES-THE-CURVE"

E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_VERDICT = "SHARP-THRESHOLD"

E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"

ARM_DESC = {
    ANT_ARM: "(a) FIVE-FACT + ANTI-ON-FACT3: the e291 family organism "
             "(loaded bit-exact) + the corpus stream 1:1 (seed 29401, the "
             "union-orthogonalized e268 form, B_C = 60%) + THE "
             "ANTI-CONTROLLER on FACT3: 16 events at M=25 cadence; the "
             "dose on the read's EXCESS (lr_anti_t = LR_ANTI_MAX x min(1, "
             "read_t/0.05)); the ANTI-gradient = ASCENT on fact3's "
             "name-CE (loss = -(name-only CE) on FACT3's own 12-window "
             "pool, 112 masked name tokens; clip 1.0; UNPROJECTED) through "
             "the SEPARATE anti-buffer opt_A (SGD-M 0.9 wd 0, private "
             "persistent, bidirectionally isolated); the ERASE budget B_E "
             "= the maintenance form's B_M = 40% of BUDGET with e288's "
             "equal-share cap per event; LR_ANTI_MAX = B_E / 81.5736 = "
             f"{LR_ANTI_MAX_FROZEN!r} (== e288's LR_M_MAX — the "
             "calibration transfers exactly).",
    TWN_ARM: "(b) THE NO-ANTI TWIN: the SAME organism + the SAME "
             "bit-identical corpus stream (seed 29401) — NO anti events. "
             "The passive counterfactual: the arms' ONLY delta is the "
             "anti. The twin separates the anti's collateral from the "
             "passive decay (e290's law predicts the passive side erodes "
             "too); the twin's five-fact panel at the milestones is the "
             "siblings' anti-free reference for the twin-relative clause.",
}

REGISTERED = {
    "question_verbatim": "can ONE fact be surgically erased while the "
        "organism stays healthy and the OTHER facts hold? The lab's kill "
        "law (THE_LAWS_V2.md) is the instrument; the anti-controller "
        "inverts the error gate.",
    "arms_verbatim": {
        ANT_ARM: "FIVE-FACT + ANTI-ON-FACT3.",
        TWN_ARM: "the NO-ANTI twin.",
    },
    "bars_verbatim": {
        "SURGICAL-ERASE": "FACT3 < 0.01 while >= 3 of 4 siblings hold "
            ">= 0.5x and the CE improves/holds — the first engineered "
            "forgetting",
        "COLLATERAL": "FACT3 dies but siblings fall — the shared "
            "representation bleeds; the honest within-family reading",
        "IMMUNE": "FACT3 refuses — the read resists ascent; a stability "
            "law",
        "MIXED": "anything else.",
    },
    "rider_bars_verbatim": "PRICED-AT-THE-CONSTANT / DIRECTED-CHEAPER (a "
        "Maxwell demon of forgetting) / ERASURE-COSTS-MORE (immunity "
        "priced)",
    "family_disclosure": "THE SELECTIVITY TEST IS WITHIN-FAMILY: the "
        "vehicle is the e291 five-fact FAMILY organism (five 12-window "
        "context groups ALL bound to the SAME name, five exactly-"
        "orthogonal 10k rooms) because e293 proved five DISTINCT-name "
        "facts are not serially installable by the protocol — erasing one "
        "sibling while four hold is the HARDEST selectivity test; the "
        "siblings SHARE representation (T270: the antibodies are the same "
        "antibody), so collateral is the expected failure mode and a "
        "clean erase the strongest form of the result.",
    "reads_verbatim": "all five facts' reads at t100/200/300/400; the "
        "anti trace; the budget; the corpus CE; THE e305 LANDAUER RIDER: "
        "the anti's spent displacement (work) vs the bystanders' absorbed "
        "read-loss (waste heat) vs the passive threshold price (e290's "
        "constant) — the erasure thermodynamic table.",
    "registration": "question + bars + arms VERBATIM from the dispatch "
        "letter; every convention picked + frozen HERE at birth BEFORE "
        "compute; this script committed at birth; adjudicate against "
        "exactly this; no bar shopping.",
    "predictions": {
        "P-e294a_demon": "the aimed ascent erases FACT3 fast (<= 8 of 16 "
            "events; the gate read IS the battery's own rows) and CHEAP: "
            "WORK < price_lo — DIRECTED-CHEAPER (a Maxwell demon of "
            "forgetting).",
        "P-e294b_shared_bleed": "the bystanders fall WITH FACT3 (T270's "
            "one-representation geometry inverted; e307's one-brush "
            "stroke) — COLLATERAL; the anti gradient rides out of its "
            "own room (the bearer is the overlap tail, e306).",
        "P-e294c_passive_confound": "the NO-ANTI twin's five reads erode "
            "substantially (e290's law at B_C-scale orthogonal spend) — "
            "the twin-relative clause keeps the adjudication honest; an "
            "ABSOLUTE/TWIN disagreement routes to MIXED.",
        "P-e294d_saturation": "if the name-CE ascent saturates (the "
            "softmax floor), the read stalls above 0.01 with the dose at "
            "the taper — IMMUNE or MIXED with the dose/CE trace the "
            "discriminating datum.",
    },
}

deviations: list[str] = [
    "THE ORGANISM LOADED, NOT REBUILT (extend, don't repeat): e291's "
    "committed five-fact family organism loaded BIT-EXACT from "
    "runs/checkpoints/e291_organism.pt and gated THREE ways (G_FACTLOAD: "
    "checkpoint md5 + flat md5 + the behavioral panel g0 2e-6 / gm12 1e-5 "
    "vs e291's committed baselines — the e288 G_FACTLOAD convention "
    "applied to the family vehicle); the five installs are NOT re-run "
    "(their committed record is the vehicle's provenance). The five rooms "
    "rebuilt from the frozen seeds (D 29111 / perm 29112) + certified + "
    "BIT-gated vs e291_rooms.pt (G_ROOMS5).",
    "THE FAMILY DISCLOSURE (the dispatch's own requirement, carried "
    "verbatim): the selectivity test is WITHIN-FAMILY — the siblings "
    "share representation (T270; e293: five distinct-name facts are not "
    "serially installable; the family construction is the only "
    "five-live-fact organism the protocol yields) — erasing one sibling "
    "while four hold is the HARDEST selectivity test; disclosed at the "
    "verdict, the report, and every table.",
    "THE ANTI-CONTROLLER'S FORM (the inverted error gate, frozen): the "
    "dose on the read's EXCESS (lr_anti_t = LR_ANTI_MAX x min(1, "
    "read_t/0.05) — full dose when alive, vanishing when dead; the gate "
    "read on FACT3's OWN g0 battery at the event instant, the same "
    "battery the bars read); the ANTI-gradient = ASCENT on fact3's "
    "name-CE (loss := -(the name-only CE) — e287's construction scoped "
    "to FACT3's own 12-window pool; backward -> clip 1.0 -> UNPROJECTED "
    "(the anti may go anywhere — the bearer hunt is the anti's own)) "
    "through the SEPARATE anti-buffer opt_A (SGD-M 0.9 wd 0, private "
    "persistent; the bidirectional bitwise isolation probe around EVERY "
    "optimizer event, any mismatch HALTs). B_E := the maintenance form's "
    "B_M = 0.40 x BUDGET (the dispatch); LR_ANTI_MAX := B_E / "
    "81.57360134901982 (e287's 16-event b_m sum — the maintenance "
    "calibration's own denominator) = 0.02250444534525284 == e288's "
    "LR_M_MAX (B_E == B_M and one stream: the calibration transfers "
    "exactly, asserted); the per-event cap e288's equal-share form over "
    "B_E.",
    "THE SIBLING-HOLD CLAUSE'S TWO READINGS (the cell's one picked "
    "operationalization, disclosed): ABSOLUTE (>= 0.5x BASELINE_i — "
    "e291's HOLD form) AND TWIN-RELATIVE (>= 0.5x the NO-ANTI twin's "
    "same-fact t400 read — the anti-corrected reading; the arms' "
    "difference is the registered object). SURGICAL-ERASE and COLLATERAL "
    "require BOTH readings to agree (>= 3 both, or <= 2 both); a "
    "disagreement routes to MIXED with both tables shown — e290's "
    "hard-bound law predicts the absolute denominator can confound "
    "passive erosion with anti collateral at this stream scale.",
    "THE e305 LANDAUER OPERATIONALIZATION (frozen): WORK := the anti's "
    "REALIZED DRIFT ||anti_cum|| at the erase instant (first post-anti "
    "panel read with FACT3 < 0.01; t400 if never); SPENT := S_anti (the "
    "step-norm sum) co-reported; the passive price := e290's committed "
    "realized-drift bracket x ||F3WRITE|| (FACT3's own install write, "
    "both parent states loaded bit-exact — the law's denominator form); "
    "the waste heat := the bystanders' read-loss vs the twin at t400; "
    "co-reports: the twin's own realized drift (the in-cell passive "
    "reference) + the organism-write scaling (19.27) as the alternative "
    "denominator.",
    "THE BUDGET'S LEDGER SPLIT (disclosed): G_BUDGETLEDGER is HARD — the "
    "per-stream S accounting machine-consistent with the realized "
    "displacements (an implementation-integrity gate); G_BUDGET (S_total "
    "<= BUDGET + slack, the triangle bound over the split pools) is "
    "NON-HALTING — a blow routes the composite to MIXED (the family's "
    "convention; an implementation outcome to autopsy, everything "
    "verbatim).",
    "THE NO-HELD-PARAPHRASE SCOPE (disclosed): e291's paraphrase bank is "
    "NOT re-read in this cell — the dispatch's registered reads name the "
    "five panels, the anti trace, the budget, the corpus CE, and the "
    "Landauer table only; the held bank's tumor test belongs to the "
    "preservation line (a successor may read it on the post states).",
    "NO CONS (T259/e281; the committed form): the frozen bars read the "
    "WRITE and the DISPLACEMENT only; the organism + both arms' "
    "post-phase states CHECKPOINTED for any later landing pass.",
    "e261's BURST MACHINERY PORTED BY IMPORT + REBINDING (the family "
    "convention): wait_gpu_free/_open_burst/burst_temp_check/"
    "burst_cooldown resolve their globals through e261's namespace — "
    "rebound HERE (log/NAME/SMOKE/T0/ledgers) so the envelope polls are "
    "tagged e294:<ARM>:<phase> per the dispatch; the committed "
    "lab/e261_rank_ladder.py is NOT modified. The driver (the "
    "anti-phase) is THIS file's.",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the "
    "displacement-load reads; no new history is run.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E294_SMOKE=1): 8 corpus steps/arm, M=2 (4 anti events), "
    "milestones 1..8, rooms k=512 at the same seeds (G_ROOMS5's bit-bind "
    "vs e291_rooms.pt VACUOUS at smoke k — disclosed), the REAL organism "
    "loaded + panel-gated (G_FACTLOAD live), budgets scaled per-stream "
    "(B_C x 8/400; B_E x 4/16; LR_ANTI_MAX NOT scaled — e288's disclosed "
    "smoke convention), the ASCENT GATE live on the real organism, the "
    "anti-event full path exercised (dose gate, cap, isolation, panel); "
    "all paths smoke_-prefixed, own smoke dir; NOTHING adjudicated "
    "(SMOKE stamp on every read).",
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


# ======================================================================
# THE FIVE ROOMS (e291's SharedFrameRooms PORTED WHOLE — the family's
# own instrument; the committed lab/e291_multifact.py is NOT modified)
# ======================================================================
class SharedFrameRooms:
    """Five rank-k SRCT rooms as DISJOINT spectral supports of ONE frame:
    ONE seeded +-1 diagonal D and ONE seeded permutation split into five
    disjoint sorted index sets S_i (k each). Room i's exact projector:
        P_i x = D . idct(mask_{S_i}(dct(D . x)))
    (fp64 pocketfft, CPU — e261's SRCT form verbatim, the frame shared).
    The pairwise projectors compose to ZERO exactly (disjoint supports),
    and the union P_U = sum_i P_i is ONE exact projector with the 50k
    union mask. coeffs-once reads make every per-room fraction nearly
    free: in_room_frac(x, i) = ||c[S_i]|| / ||x|| with c = dct(D x)."""

    def __init__(self, n: int, k: int, n_rooms: int, seed_d: int,
                 seed_s: int, v_flat64: np.ndarray, Vp64: np.ndarray,
                 params_ref, dev: torch.device):
        assert 0 < k * n_rooms <= n
        self.n, self.k, self.n_rooms = int(n), int(k), int(n_rooms)
        self.dev = dev
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        perm = torch.randperm(n, generator=g2)[:self.k * self.n_rooms]
        self.sets = [perm[i * k:(i + 1) * k].sort().values.numpy()
                     for i in range(self.n_rooms)]
        all_idx = np.concatenate(self.sets)
        assert len(set(all_idx.tolist())) == self.k * self.n_rooms  # disjoint
        self.masks = []
        for S in self.sets:
            m = np.zeros(n, dtype=np.float64)
            m[S] = 1.0
            self.masks.append(m)
        self.mask_union = np.zeros(n, dtype=np.float64)
        self.mask_union[all_idx] = 1.0
        self.seed_d, self.seed_s = seed_d, seed_s
        # the span/v-map instruments (read-only probes; e261's form)
        self.v64 = v_flat64.astype(np.float64)
        self.mean_v = float(self.v64.mean())
        self.Vp = Vp64.astype(np.float64)
        self.r_span = int(Vp64.shape[0])
        # the per-parameter offsets (the write-back slicing)
        self.offsets, self.shapes = [], []
        off = 0
        for p in params_ref:
            self.offsets.append((off, off + p.numel()))
            self.shapes.append(tuple(p.shape))
            off += p.numel()
        assert off == self.n, f"flat size {off} != {self.n}"

    # -- the frame transforms -------------------------------------------
    def coeffs(self, x64: np.ndarray) -> np.ndarray:
        return E261.sf.dct(self.D * x64, type=2, norm="ortho",
                           workers=E261.DCT_WORKERS)

    def recon(self, c_masked: np.ndarray) -> np.ndarray:
        return self.D * E261.sf.idct(c_masked, type=2, norm="ortho",
                                     workers=E261.DCT_WORKERS)

    def project_masked(self, x64: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.recon(self.coeffs(x64) * mask)

    def project_room(self, x64: np.ndarray, i: int) -> np.ndarray:
        return self.project_masked(x64, self.masks[i])

    def project_union(self, x64: np.ndarray) -> np.ndarray:
        return self.project_masked(x64, self.mask_union)

    # -- the fractions (coeffs-once; near-free) --------------------------
    def room_fracs_from_coeffs(self, c: np.ndarray) -> list:
        tot = float(np.sqrt((c * c).sum()))
        if tot <= 0:
            return [0.0] * self.n_rooms
        return [float(np.sqrt((c[S] * c[S]).sum())) / tot for S in self.sets]

    def in_room_fracs(self, x64: np.ndarray) -> list:
        """[||P_i x||/||x|| for i in 0..4] — ONE dct."""
        return self.room_fracs_from_coeffs(self.coeffs(x64))

    def union_frac(self, x64: np.ndarray) -> float:
        """||P_union x|| / ||x|| — ONE dct."""
        c = self.coeffs(x64)
        tot = float(np.sqrt((c * c).sum()))
        if tot <= 0:
            return 0.0
        cu = c * self.mask_union
        return float(np.sqrt((cu * cu).sum()) / tot)

    # -- the certification (e261's certify form, per-room + the union +
    #    the pairwise-zero probes) ----------------------------------------
    def certify(self, probes: int = E261.CERT_PROBES,
                seed: int = E261.CERT_SEED) -> dict:
        rng = np.random.default_rng(seed)
        out = {"probes": probes, "seed": seed}
        x = rng.standard_normal(self.n)
        xr = self.recon(self.coeffs(x))
        out["dct_roundtrip_rel"] = float(np.linalg.norm(xr - x)
                                         / np.linalg.norm(x))
        per_room = {}
        for i in range(self.n_rooms):
            idem, kept2 = [], []
            for _ in range(probes):
                y = rng.standard_normal(self.n)
                py = self.project_room(y, i)
                ppy = self.project_room(py, i)
                idem.append(float(np.linalg.norm(ppy - py)
                                  / np.linalg.norm(py)))
                kept2.append(float((py @ py) / (y @ y)))
            ovr = [float(np.linalg.norm(self.project_room(self.Vp[j], i)))
                   for j in range(self.r_span)]
            bar = max(5.0 * math.sqrt(2.0 * self.k) / self.n, 1e-9)
            per_room[f"room{i + 1}"] = {
                "k": self.k, "seeds": [self.seed_d, self.seed_s],
                "index_set": f"perm({self.seed_s})[{i * self.k}:"
                             f"{(i + 1) * self.k}] (disjoint slices)",
                "idempotency_max": max(idem),
                "kept2_mean": float(np.mean(kept2)),
                "kept2_expect": self.k / self.n,
                "kept2_bar_10sig": bar,
                "kept2_pass": bool(abs(float(np.mean(kept2)) - self.k / self.n)
                                   <= bar),
                "idempotency_pass": bool(max(idem) <= 1e-8),
                "span_overlap_mean": float(np.mean(ovr)),
                "span_overlap_expect": math.sqrt(self.k / self.n)}
        # the union's kept^2 (expect 5k/N)
        kept2u = []
        for _ in range(probes):
            y = rng.standard_normal(self.n)
            pu = self.project_union(y)
            kept2u.append(float((pu @ pu) / (y @ y)))
        out["union_kept2_mean"] = float(np.mean(kept2u))
        out["union_kept2_expect"] = self.n_rooms * self.k / self.n
        out["union_kept2_pass"] = bool(
            abs(float(np.mean(kept2u)) - self.n_rooms * self.k / self.n)
            <= max(5.0 * math.sqrt(2.0 * self.n_rooms * self.k) / self.n,
                   1e-9))
        # the pairwise-zero probes: ||P_j(P_i x)|| / ||P_i x|| (exact 0)
        pairwise = np.zeros((self.n_rooms, self.n_rooms))
        for _ in range(probes):
            y = rng.standard_normal(self.n)
            for i in range(self.n_rooms):
                pi = self.project_room(y, i)
                for j in range(self.n_rooms):
                    if i != j:
                        pj = self.project_room(pi, j)
                        pairwise[i, j] = max(
                            pairwise[i, j],
                            float(np.linalg.norm(pj)
                                  / max(np.linalg.norm(pi), 1e-30)))
        out["pairwise_proj_max"] = pairwise.tolist()
        out["pairwise_zero_bar"] = 1e-12
        out["pairwise_pass"] = bool(pairwise.max() <= 1e-12)
        out["per_room"] = per_room
        out["pass"] = bool(out["dct_roundtrip_rel"] <= 1e-8
                           and all(r["kept2_pass"] and r["idempotency_pass"]
                                   for r in per_room.values())
                           and out["union_kept2_pass"]
                           and out["pairwise_pass"])
        return out

    # -- the displacement reads (e261's displacement_loads form) -----------
    def displacement_loads(self, flat_cpu: torch.Tensor) -> dict:
        d = flat_cpu.double().numpy().astype(np.float64)
        d2 = d * d
        tot = float(d2.sum())
        out = {"v_excess": float((d2 * self.v64).sum() / tot / self.mean_v)
               if tot > 0 else 0.0}
        c1 = self.Vp @ d
        out["cos_to_span"] = math.sqrt(float(c1 @ c1) / tot) if tot > 0 \
            else 0.0
        if tot > 0:
            fr = self.in_room_fracs(d)
            out["in_room_frac_per_room"] = fr
            out["in_union_frac"] = float(np.sqrt(sum(f * f for f in fr)))
        else:
            out["in_room_frac_per_room"] = None
            out["in_union_frac"] = None
        return out


# --------------------------------------------- THE MISSILE'S PROJECTION (port)
def orthogonalize_union(rooms: SharedFrameRooms, params) -> dict:
    """e291's orthogonalize_union VERBATIM: replace the (clipped) corpus
    gradient g by g_perp = g - P_union(g) — the component entirely
    orthogonal to ALL FIVE rooms (CPU fp64, write fp32, norm NOT
    rescaled). The verification read ||P_union g_perp|| / ||g_perp|| is
    returned for the gate (exact projector -> ~1e-16)."""
    params = list(params)
    g = torch.cat([p.grad.detach().reshape(-1) for p in params]) \
        .to(CPU).double().numpy().astype(np.float64)
    gn2 = float(g @ g)
    gp_in = rooms.project_union(g)
    gperp = g - gp_in
    gpn2 = float(gperp @ gperp)
    resid = rooms.project_union(gperp)
    rel = (float(np.sqrt(resid @ resid)) / float(np.sqrt(gpn2))
           if gpn2 > 0 else 0.0)
    gp32 = torch.from_numpy(gperp.astype(np.float32))
    with torch.no_grad():
        for p, (a, b), shp in zip(params, rooms.offsets, rooms.shapes):
            p.grad.copy_(gp32[a:b].to(rooms.dev).reshape(shp))
    return {"gn": math.sqrt(gn2), "gperp_norm": math.sqrt(gpn2),
            "norm_ratio": (math.sqrt(gpn2 / gn2) if gn2 > 0 else 0.0),
            "in_union_frac": (math.sqrt(max(0.0, 1.0 - gpn2 / gn2))
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


def buffer_inroom_fracs(rooms: SharedFrameRooms, opt, params) -> tuple:
    """([||P_i buf||/||buf|| per room], ||buf||) — ONE dct per optimizer
    (the shared frame's coeffs-once read; the composition probe)."""
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
    return rooms.in_room_fracs(b), bn


def pending_buffer_sqnorm(opt, params, momentum: float) -> float:
    """||b_t||^2 for PyTorch SGD-M's exact in-step update b_t = mu * buf_{t-1}
    + g (dampening 0): computed on-GPU fp32 from the optimizer's stored
    buffers + the grads now in p.grad — the budget cap's per-step price of
    the applied displacement (theta -= lr_t * b_t)."""
    sq = torch.zeros((), device=next(iter(params)).device)
    st = opt.state
    for p in params:
        d = st.get(p)
        buf_prev = d.get("momentum_buffer") if isinstance(d, dict) else None
        if buf_prev is None:
            v = p.grad.detach()
        else:
            v = momentum * buf_prev + p.grad.detach()
        sq = sq + v.square().sum()
    return float(sq.item())


def med(xs):
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


# ======================================================================
# THE ANTI-PHASE DRIVER — the corpus stream (both arms, bit-identical
# draws) + THE ANTI-CONTROLLER's events on FACT3 (arm (a) only): the
# dose on the read's EXCESS, the ascent gradient through the separate
# anti-buffer, the B_E cap, the full-panel collateral microscope.
# ======================================================================
def anti_phase(tag: str, anti_on: bool, net0_sd: dict,
               rooms: SharedFrameRooms, organism_flat_np: np.ndarray,
               base_flat_np: np.ndarray, budget_c: float, budget_e: float,
               lr_anti_max: float, facts_x: list, facts_mask: list,
               baselines: dict, g0_ids_f: dict, gm12_ids_f: dict,
               anchor_full, train_ids, r_eval_xy, zid,
               resume_ck: Path, dev: torch.device) -> dict:
    """The net starts at the LOADED family organism; per corpus step
    t = 1..400: the e268 corpus step VERBATIM (16 anchors + 32 random
    windows; CE; lr = LR_STABLE x cosine; clip 1.0) -> the UNION
    orthogonalization (g_perp, verified EVERY step) -> the per-step cap
    over B_C -> opt_C.step() (the corpus side's own buffer; BOTH
    optimizers snapshotted bitwise around the step when the anti side
    exists).

    AND THE ANTI EVENT (arm (a), after every M=25 corpus steps — exactly
    16): the gate read on FACT3's OWN battery at the event instant;
    dose_t = min(1, read_t/0.05); lr_target = LR_ANTI_MAX x dose_t; the
    ANTI batch = ix(16) draws from FACT3's OWN 12-window pool (the anti
    stream's own generator); loss := -(the name-only CE) — ASCENT on
    fact3's name-CE (push the read down); backward -> clip 1.0 ->
    UNPROJECTED -> through opt_A (the anti-buffer SGD-M, private
    persistent) at lr = min(lr_target, cap_e), cap_e = share_e/||b_e||
    over B_E's equal-share reservation; the realized displacement enters
    S_anti; THE FULL FIVE-FACT PANEL read after EVERY event (the
    collateral microscope); the anti trace row per event; t_erase
    detected at the first post-event panel with FACT3 < 0.01.

    Ledgers: the family's full set (corpus / orth / disp / buf / budget /
    lr) + the ANTI ledger (every event, the full trace) + the
    EVENT-PANEL ledger (the five-fact panel after every anti event).
    Thermal: a poll after EVERY optimizer event."""
    budget_norm = budget_c + (budget_e if anti_on else 0.0)
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    name_bs = G1.NAME_BS
    n_steps = PHASE_STEPS
    M = MAINT_EVERY
    n_anc = anchor_full.shape[0]
    N = int(organism_flat_np.size)
    ti = TARGET_IDX
    fact_name = TARGET_FACT
    state = {"step": 0, "traj": [], "corpus_ledger": {}, "orth_ledger": {},
             "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
             "lr_ledger": {}, "anti_ledger": [], "event_panel_ledger": [],
             "bufsep": {"corpus_checks": 0, "corpus_violations": 0,
                        "anti_checks": 0, "anti_violations": 0,
                        "optA_steps": 0},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "anti_cum": torch.zeros(N, dtype=torch.float64),
             "tot_prev": torch.zeros(N, dtype=torch.float64),
             "S_corpus": 0.0, "S_anti": 0.0, "n_capped": 0,
             "n_anti": 0, "n_anti_capped": 0, "orth_max": 0.0,
             "t_erase": None, "work_at_erase": None,
             "spent_at_erase": None}
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
                "anti_ledger": state.get("anti_ledger", []),
                "event_panel_ledger": state.get("event_panel_ledger", []),
                "bufsep": state.get("bufsep", {}),
                "S_corpus": state.get("S_corpus"),
                "S_anti": state.get("S_anti"),
                "n_capped": state.get("n_capped"),
                "n_anti": state.get("n_anti"),
                "n_anti_capped": state.get("n_anti_capped"),
                "orth_max": state.get("orth_max"),
                "t_erase": state.get("t_erase"),
                "work_at_erase": state.get("work_at_erase"),
                "spent_at_erase": state.get("spent_at_erase"),
                "anti_cum_norm_final": float(np.linalg.norm(
                    state.get("anti_cum").double().numpy()))
                if state.get("anti_cum") is not None else None,
                "lr_applied_min": min(lrs_c) if lrs_c else None,
                "lr_applied_median": (float(sorted(lrs_c)[len(lrs_c) // 2])
                                      if lrs_c else None),
                "lr_sched_median": None, "cap_min": None,
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt_C = opt_A = cgen = agen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    corp_cum = state["corp_cum"].to(dev)
    anti_cum = state["anti_cum"].to(dev)
    tot_prev = state["tot_prev"].to(dev)
    S_corpus = float(state["S_corpus"])
    S_anti = float(state["S_anti"])
    S_total = S_corpus + S_anti
    n_capped = int(state["n_capped"])
    n_anti = int(state["n_anti"])
    n_anti_capped = int(state["n_anti_capped"])
    orth_max = 0.0

    def _panel_read() -> dict:
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        return {fi: G1.battery_cell(evl, ids, zid)["mean_pz"]
                for fi, ids in g0_ids_f.items()}

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{tag}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = G1.evl_load(net0_sd).to(dev)
            net.train()
            # the topology: ONE corpus SGD-M + (arm (a)) the SEPARATE
            # anti-buffer SGD-M — e285 DIAL 1's separation, the anti side
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_A = torch.optim.SGD(net.parameters(), lr=lr_anti_max,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            agen = torch.Generator().manual_seed(ANTI_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt_C.load_state_dict(state["optC"])
                if anti_on and state.get("optA") is not None:
                    opt_A.load_state_dict(state["optA"])
                cgen.set_state(state["cgen_state"])
                if anti_on and state.get("agen_state") is not None:
                    agen.set_state(state["agen_state"])
                step = state["step"]
                S_corpus = float(state["S_corpus"])
                S_anti = float(state["S_anti"])
                S_total = S_corpus + S_anti
                n_capped = int(state["n_capped"])
                n_anti = int(state["n_anti"])
                n_anti_capped = int(state["n_anti_capped"])
            evl = G1.evl_load(net0_sd)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            lr_sched = LR_STABLE * cosine_lr(step - 1, E261.INST_TOTAL)
            # ---- THE CORPUS STEP (bit-identical draws across arms) ------
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
            # the RAW gradient's in-union fraction (sampled)
            g_in_union = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                g_in_union = rooms.union_frac(g64)
            # ---- the UNION orthogonalization (verified EVERY step) ------
            orth_row = orthogonalize_union(rooms, net.parameters())
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- the corpus cap over B_C (e288's equal-share form) ------
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
            # ---- the isolated corpus step (the anti side untouched) -----
            sn_a_before = (snap_buffers(opt_A, params_live)
                           if anti_on else None)
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()
            with torch.no_grad():
                d_step = torch.cat([p.detach().reshape(-1)
                                    for p in net.parameters()]) - theta_b
                corp_cum += d_step
            realized = float(d_step.norm().item())
            S_corpus += realized
            S_total = S_corpus + S_anti
            state["bufsep"]["corpus_checks"] += 1
            if anti_on:
                if not buffers_bitwise_equal(
                        sn_a_before, snap_buffers(opt_A, params_live)):
                    state["bufsep"]["corpus_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}: "
                        "the corpus step touched the anti side's state — "
                        "HALT")
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c,
                                                "g_in_union_frac": g_in_union}
                state["orth_ledger"][step] = {
                    "gn_clipped": gn_c,
                    "gperp_norm": orth_row["gperp_norm"],
                    "norm_ratio": orth_row["norm_ratio"],
                    "in_union_frac": orth_row["in_union_frac"],
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

            # ---- THE ANTI EVENT (arm (a): the cell's instrument) --------
            if anti_on and step % M == 0:
                n_anti += 1
                # (i) THE ANTI BATCH: ix(16) draws from FACT3's OWN
                # 12-window pool (the anti stream's own generator)
                ix = torch.randint(facts_x[ti].shape[0], (name_bs,),
                                   generator=agen)
                nw = facts_x[ti][ix]
                x_i = nw[:, :-1].to(dev)
                y_i = nw[:, 1:].to(dev)
                m_i = facts_mask[ti][ix].to(dev)
                logits_i, _ = net(x_i)
                nll_i = F.cross_entropy(
                    logits_i.reshape(-1, logits_i.shape[-1]),
                    y_i.reshape(-1), reduction="none"
                ).view(x_i.shape[0], x_i.shape[1])
                nm_i = nll_i[m_i]                    # the 112 name tokens
                name_ce = nm_i.mean()                # the name-only CE
                loss_anti = -name_ce                 # ASCENT (maximize CE)
                net.zero_grad(set_to_none=True)
                loss_anti.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                gn_i = float(torch.cat(
                    [p.grad.detach().reshape(-1) for p in net.parameters()
                     if p.grad is not None]).norm().item())
                # the anti gradient's room fractions (the brush-inversion
                # read: T270/T272/e307 — in-own-room vs in-union vs chance)
                g64i = torch.cat([p.grad.detach().reshape(-1)
                                  for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                inroom_fr = rooms.in_room_fracs(g64i)
                gin_own = inroom_fr[ti]
                gin_union = float(np.sqrt(sum(f * f for f in inroom_fr)))
                # (ii) THE GATE: FACT3's read at the event instant (the
                # same battery the bars read); the dose on the EXCESS
                sd_gate = {k: v.detach().cpu().clone()
                           for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_gate)
                evl.eval()
                read_gate = G1.battery_cell(evl, g0_ids_f[fact_name],
                                            zid)["mean_pz"]
                dose_t = min(1.0, read_gate / DOSE_FULL_LEVEL)
                lr_target = lr_anti_max * dose_t
                # THE ANTI'S LAW (live at EVERY event, smoke and full):
                # never exceeds LR_ANTI_MAX; vanishes at read 0; full at
                # read >= 0.05
                assert lr_target <= lr_anti_max * (1.0 + 1e-12), \
                    f"anti law broken: lr_target {lr_target} > " \
                    f"LR_ANTI_MAX {lr_anti_max}"
                if read_gate <= 0.0:
                    assert lr_target == 0.0
                if read_gate >= DOSE_FULL_LEVEL:
                    assert abs(lr_target - lr_anti_max) < 1e-15
                # (iii) THE B_E CAP (e288's equal-share form over the
                # erase pool)
                be_sq = pending_buffer_sqnorm(opt_A, params_live,
                                              SGD_MOMENTUM)
                be_norm = math.sqrt(max(be_sq, 0.0))
                rem_anti_now = 1 + sum(1 for m in range(step + 1,
                                                        n_steps + 1)
                                       if m % M == 0)
                remaining_e = max(budget_e - S_anti, 0.0)
                share_e = remaining_e / rem_anti_now
                cap_e = share_e / max(be_norm, 1e-12)
                lr_e = min(lr_target, cap_e)
                assert lr_e <= lr_anti_max * (1.0 + 1e-9), \
                    f"applied lr_e {lr_e} exceeds LR_ANTI_MAX {lr_anti_max}"
                capped_e = bool(cap_e < lr_target)
                binder_e = "cap" if capped_e else "anti-gate"
                if capped_e:
                    n_anti_capped += 1
                for g_ in opt_A.param_groups:
                    g_["lr"] = lr_e
                # (iv) THE ISOLATED ANTI STEP (the corpus side untouched)
                sn_c_before = snap_buffers(opt_C, params_live)
                theta_b2 = torch.cat([p.detach().reshape(-1)
                                      for p in net.parameters()])
                opt_A.step()
                with torch.no_grad():
                    d_e = torch.cat([p.detach().reshape(-1)
                                     for p in net.parameters()]) - theta_b2
                    anti_cum += d_e
                realized_e = float(d_e.norm().item())
                S_anti += realized_e
                S_total = S_corpus + S_anti
                state["bufsep"]["anti_checks"] += 1
                state["bufsep"]["optA_steps"] += 1
                if not buffers_bitwise_equal(
                        sn_c_before, snap_buffers(opt_C, params_live)):
                    state["bufsep"]["anti_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}'s "
                        "anti event: it touched the corpus side's state — "
                        "HALT")
                bufA_fr, bufA_n = buffer_inroom_fracs(rooms, opt_A,
                                                      params_live)
                # (v) THE COLLATERAL MICROSCOPE: the full panel after
                # EVERY event + t_erase detection
                panel = _panel_read()
                state["event_panel_ledger"].append({
                    "step": step, "anti_index": n_anti,
                    "panel_g0": panel,
                    "S_anti": S_anti, "elapsed_s": round(time.time() - T0,
                                                         1)})
                if state["t_erase"] is None and panel[fact_name] < ERASE_BAR:
                    state["t_erase"] = step
                    state["work_at_erase"] = float(np.linalg.norm(
                        anti_cum.double().cpu().numpy()))
                    state["spent_at_erase"] = S_anti
                state["anti_ledger"].append({
                    "step": step, "anti_index": n_anti,
                    "name_ce": float(name_ce.item()),
                    "n_name_tokens": int(nm_i.numel()),
                    "gn_clipped": gn_i,
                    "g_in_own_room_frac": gin_own,
                    "g_in_union_frac": gin_union,
                    "g_in_room_fracs": inroom_fr,
                    "read_at_gate": read_gate, "dose_t": dose_t,
                    "lr_anti_max": lr_anti_max, "lr_target": lr_target,
                    "cap_e": cap_e, "lr_anti": lr_e, "capped_e": capped_e,
                    "binder": binder_e,
                    "b_e_norm": be_norm, "share_e": share_e,
                    "realized_step_norm": realized_e,
                    "realized_vs_share": (realized_e / share_e
                                          if share_e > 0 else None),
                    "bufA_in_own_room_frac": (bufA_fr[ti]
                                              if bufA_fr else None),
                    "bufA_in_room_fracs": bufA_fr, "bufA_norm": bufA_n,
                    "panel_g0_post": panel,
                    "S_anti": S_anti, "S_total": S_total,
                    "budget_e": budget_e,
                    "t_erase_hit": bool(state["t_erase"] == step),
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] ANTI #{n_anti:2d} @t{step:3d}: name_ce "
                    f"{float(name_ce.item()):.4f} |g| {gn_i:.3f} in-own "
                    f"{gin_own:.4f} in-union {gin_union:.4f} | GATE read "
                    f"{read_gate:.6f} dose {dose_t:.3f} -> lr {lr_e:.6f} "
                    f"{binder_e.upper()} (target {lr_target:.6f} cap "
                    f"{cap_e:.6f}; b_e {be_norm:.3f} share "
                    f"{share_e:.5f}) -> step {realized_e:.5f} | S_anti "
                    f"{S_anti:.4f}/{budget_e:.4f} "
                    f"({S_anti / budget_e:.1%} of B_E) S_total "
                    f"{S_total:.4f}/{budget_norm:.4f} | panel "
                    + " ".join(f"{k} {v:.4f}" for k, v in panel.items()))
                n_burst += 1
                ok_t, temp = E261.burst_temp_check(
                    f"{tag}:anti:c{n_chunks}.x")
                chunk_temps.append(temp)

            if step in MILESTONES or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                panel = {fi: G1.battery_cell(evl, ids, zid)["mean_pz"]
                         for fi, ids in g0_ids_f.items()}
                panel12 = {fi: G1.battery_cell(evl, ids, zid)["mean_pz"]
                           for fi, ids in gm12_ids_f.items()}
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                theta_t = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                d_org = theta_t - organism_flat_np
                dn = float(np.linalg.norm(d_org))
                fr_org = rooms.in_room_fracs(d_org)
                rem = theta_t - base_flat_np
                rn = float(np.linalg.norm(rem))
                anti_np = anti_cum.double().cpu().numpy()
                anti_n = float(np.linalg.norm(anti_np))
                anti_fr = rooms.in_room_fracs(anti_np) if anti_n > 0 \
                    else None
                corp_np = corp_cum.double().cpu().numpy()
                corp_n = float(np.linalg.norm(corp_np))
                cum_n = float(np.linalg.norm(corp_np + anti_np))
                tot_cum = corp_cum + anti_cum
                v_int_np = (tot_cum - tot_prev).double().cpu().numpy()
                vn = float(np.linalg.norm(v_int_np))
                tot_prev = tot_cum.clone()
                bufC_fr, bufC_n = buffer_inroom_fracs(rooms, opt_C,
                                                      params_live)
                bufA_fr2, bufA_n2 = (buffer_inroom_fracs(rooms, opt_A,
                                                         params_live)
                                     if anti_on else (None, 0.0))
                ratios = {fi: panel[fi] / baselines[fi] for fi in panel}
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_organism_norm": dn,
                    "drift_in_room_frac_per_room": fr_org,
                    "drift_in_union_frac": float(np.sqrt(sum(
                        f * f for f in fr_org))),
                    "anti_cum_norm": anti_n,
                    "anti_cum_in_own_room_frac": (anti_fr[ti]
                                                  if anti_fr else None),
                    "anti_cum_in_room_fracs": anti_fr,
                    "corpus_cum_norm": corp_n,
                    "remaining_from_base_norm": rn,
                    "total_disp_interval_norm": vn,
                    "total_disp_cum_norm": cum_n})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_in_union_frac": float(np.sqrt(sum(
                        f * f for f in bufC_fr))) if bufC_fr else None,
                    "bufC_in_room_fracs": bufC_fr, "bufC_norm": bufC_n,
                    "bufA_in_own_room_frac": (bufA_fr2[ti]
                                              if bufA_fr2 else None),
                    "bufA_norm": bufA_n2,
                    "optA_steps": state["bufsep"]["optA_steps"]})
                state["budget_ledger"].append({
                    "step": step, "S_corpus": S_corpus, "S_anti": S_anti,
                    "S_total": S_total,
                    "S_total_usage_frac": S_total / budget_norm,
                    "cum_corpus_norm": corp_n, "cum_anti_norm": anti_n,
                    "cum_total_norm": cum_n,
                    "cum_total_usage_frac": cum_n / budget_norm,
                    "n_capped_so_far": n_capped, "n_anti_so_far": n_anti,
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped})
                state["traj"].append({
                    "step": step, "panel_g0": panel, "panel_gm12": panel12,
                    "ratios_vs_baseline": ratios,
                    "ce_r": ce_r, "ce_corpus": float(loss_c.item()),
                    "drift_norm": dn, "drift_in_room_frac_per_room": fr_org,
                    "budget_S_total_usage_frac": S_total / budget_norm,
                    "budget_S_corpus": S_corpus, "budget_S_anti": S_anti,
                    "anti_cum_norm": anti_n,
                    "lr_applied": lr_t, "lr_sched": lr_sched,
                    "capped": capped, "n_anti_so_far": n_anti,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} panel "
                    + " ".join(f"{k} {v:.6f}(x{ratios[k]:.3f})"
                               for k, v in panel.items())
                    + f" | CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d_org| {dn:.3f} | "
                    f"BUDGET S_corp {S_corpus:.4f} + S_anti {S_anti:.4f} "
                    f"= {S_total:.4f}/{budget_norm:.4f} "
                    f"({S_total / budget_norm:.1%}) ||anti_cum|| "
                    f"{anti_n:.4f} lr {lr_t:.5f} "
                    f"{'CAP' if capped else 'sched'}")
            if not ok_t:
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at t{step} — resume ckpt "
                    f"saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "optC": opt_C.state_dict(),
                    "optA": opt_A.state_dict() if anti_on else None,
                    "cgen_state": cgen.get_state(),
                    "agen_state": agen.get_state() if anti_on else None,
                    "step": step, "traj": state["traj"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "buf_ledger": state["buf_ledger"],
                    "budget_ledger": state["budget_ledger"],
                    "lr_ledger": state["lr_ledger"],
                    "anti_ledger": state["anti_ledger"],
                    "event_panel_ledger": state["event_panel_ledger"],
                    "bufsep": state["bufsep"],
                    "corp_cum": corp_cum.cpu(),
                    "anti_cum": anti_cum.cpu(),
                    "tot_prev": tot_prev.cpu(),
                    "S_corpus": S_corpus, "S_anti": S_anti,
                    "n_capped": n_capped, "n_anti": n_anti,
                    "n_anti_capped": n_anti_capped,
                    "orth_max": orth_max,
                    "t_erase": state["t_erase"],
                    "work_at_erase": state["work_at_erase"],
                    "spent_at_erase": state["spent_at_erase"],
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
    # G_BUDGETLEDGER (hard): the ledger machine-consistent with the
    # realized displacements (implementation-integrity; HALT on drift) —
    # the anti side is fully enumerable (every event ledgered)
    ledger_anti_sum = sum(v["realized_step_norm"]
                          for v in state["anti_ledger"])
    assert abs(ledger_anti_sum - S_anti) < 1e-4, \
        f"G_BUDGETLEDGER: anti ledger sum {ledger_anti_sum} != S_anti " \
        f"{S_anti}"
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "buf_ledger": state["buf_ledger"],
            "budget_ledger": state["budget_ledger"],
            "lr_ledger": state["lr_ledger"],
            "anti_ledger": state["anti_ledger"],
            "event_panel_ledger": state["event_panel_ledger"],
            "bufsep": state["bufsep"],
            "S_corpus": S_corpus, "S_anti": S_anti,
            "n_capped": n_capped, "n_anti": n_anti,
            "n_anti_capped": n_anti_capped,
            "orth_max": orth_max,
            "t_erase": state["t_erase"],
            "work_at_erase": state["work_at_erase"],
            "spent_at_erase": state["spent_at_erase"],
            "anti_cum_norm_final": float(np.linalg.norm(
                anti_cum.double().cpu().numpy())),
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
           "early_end_margin_c": E261.TEMP_EARLY_END,
           "hard_line_c": E261.TEMP_HARD,
           "dispatch_envelope": "bursts <= 180s, cooldowns 30-60s, never "
                                "past 85C — this cell runs 175/40/84 (all "
                                "inside)",
           "per_step_polls": "after EVERY optimizer event (corpus + anti, "
                             "both arms) — persisted to "
                             "runs/_envelope_log.jsonl tagged "
                             "e294:<ARM>:<phase> per the dispatch"}
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
                   f"polls are tagged e294_smoke: and excluded")
    return out


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e294_anticontroller",
        "phase": "THE ANTI-CONTROLLER CELL (targeted forgetting / machine "
                 "unlearning — the kill law inverted) + THE E305 LANDAUER "
                 "LEDGER RIDER — the e291 five-fact FAMILY organism "
                 "(loaded bit-exact) under the family's corpus stream "
                 "with THE ANTI-CONTROLLER on FACT3 (the dose on the "
                 "read's EXCESS: lr_anti_t = LR_ANTI_MAX x min(1, "
                 "read_t/0.05); the ANTI-gradient = ASCENT on fact3's "
                 "name-CE through the separate anti-buffer; B_E = the "
                 "maintenance form's B_M; 16 events at M=25) vs the "
                 "NO-ANTI twin (the arms' ONLY delta): SURGICAL-ERASE / "
                 "COLLATERAL / IMMUNE / MIXED on FACT3's t400 read + the "
                 "siblings' hold (BOTH the absolute and twin-relative "
                 "readings, frozen) + the CE clause; the rider: the "
                 "erasure thermodynamic table (WORK vs e290's passive "
                 "price vs the bystanders' waste heat) — THE SELECTIVITY "
                 "TEST IS WITHIN-FAMILY (disclosed)",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the frozen bars read the WRITE and "
                           "the DISPLACEMENT only; the organism + both "
                           "arms' post states are checkpointed"},
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "the arms strictly serial with cooldowns — never "
                      "concurrent) + CPU fp64 dense projections (pocketfft "
                      "workers 2), CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-event thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e294:<ARM>:<phase>",
            "trainings": f"2 arms x {PHASE_STEPS} corpus steps + "
                         f"{N_ANTI_EXPECTED} anti events (arm (a) only) "
                         "+ the birth ascent probe; NO cons",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} corpus steps per arm; the anti "
                     f"events after every M={MAINT_EVERY} corpus steps "
                     f"(exactly {N_ANTI_EXPECTED}, arm (a) only)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps (read POST-anti)",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors + 32 random "
                           "corpus windows; full-window CE; THIS cell's "
                           f"ONE fresh registered generator seed "
                           f"{CORPUS_GEN_SEED}; BOTH arms draw the "
                           "IDENTICAL sequence; the gradient "
                           "ORTHOGONALIZED against the five-room union "
                           "(the exact shared-frame projector)",
            "anti_step": "THE ANTI-CONTROLLER (the inverted error gate): "
                         "the gate read on FACT3's OWN g0 battery at the "
                         "event instant (the same battery the bars read); "
                         f"dose_t = min(1, read_t/{DOSE_FULL_LEVEL}); "
                         "lr_target = LR_ANTI_MAX x dose_t; the ANTI batch "
                         ":= ix(16) draws from FACT3's OWN 12-window pool "
                         f"(the anti stream's own generator seed "
                         f"{ANTI_GEN_SEED}); loss := -(the name-only CE — "
                         "mean token CE over the 112 masked name "
                         "positions) — ASCENT on fact3's name-CE; "
                         "backward -> clip 1.0 -> UNPROJECTED -> through "
                         "opt_A (the anti-buffer SGD-M, private "
                         "persistent) at lr = min(lr_target, cap_e), "
                         "cap_e = share_e/||b_e|| over B_E's equal-share "
                         "reservation; the full five-fact panel read "
                         "after EVERY event (the collateral microscope)",
            "adjudication_read": "the t=400 endpoints (post the 16th anti "
                                 "event): FACT3's read (the erase bar "
                                 "< 0.01 ABSOLUTE) + the four siblings' "
                                 "hold under BOTH readings (absolute "
                                 "baseline + twin-relative) + the "
                                 "stream-live CE clause; the twin's "
                                 "panel the anti-free reference; every "
                                 "ledger verbatim",
        },
        "the_constructions": {
            "the_organism": "e291's committed five-fact FAMILY organism "
                            "LOADED BIT-EXACT (checkpoint md5 + flat md5 "
                            "+ behavioral panel; the installs not re-run) "
                            "— THE SELECTIVITY TEST IS WITHIN-FAMILY "
                            "(the siblings share representation; the "
                            "hardest selectivity test; disclosed)",
            "the_rooms": "e291's SHARED-FRAME five rooms rebuilt from the "
                         "frozen seeds (D 29111 / perm 29112; five "
                         "disjoint sorted 10k sets) + certified + "
                         "BIT-gated vs e291_rooms.pt",
            "the_anti": f"LR_ANTI_MAX = B_E / {E287_B_M_SUM!r} = "
                        f"{LR_ANTI_MAX_FROZEN!r} (== e288's LR_M_MAX — "
                        "the maintenance calibration transfers exactly); "
                        f"B_E = {ERASE_BUDGET_SHARE:.0%} x BUDGET = the "
                        "maintenance form's B_M; the dose on the read's "
                        "EXCESS (full at read >= 0.05, vanishing at 0); "
                        "the ascent unprojected through the separate "
                        "anti-buffer",
            "the_budget": f"BUDGET = {BUDGET_FRAC} x {E283_WRITE_NORM} = "
                          f"{BUDGET_NORM_FROZEN!r} (== e288/e291's "
                          "frozen total); B_C = 60% (the corpus stream "
                          "1:1); B_E = 40% (the anti stream's own pool; "
                          "the twin has no anti pool); S_total <= BUDGET "
                          "by the triangle bound BY CONSTRUCTION",
            "the_landauer": "WORK := ||anti_cum|| at the erase instant "
                            "(t400 if never); SPENT := S_anti "
                            "co-reported; the passive price := e290's "
                            "committed realized-drift bracket "
                            f"{E290_DRIFT_BRACKET} x ||F3WRITE|| "
                            "(FACT3's own install write, from the "
                            "committed install checkpoints bit-exact); "
                            "the waste heat := the bystanders' read-loss "
                            "vs the twin at t400; co-reports: the twin's "
                            "own realized drift + the organism-write "
                            "scaling",
        },
        "deviations": deviations,
        "builds_on": [
            "the e294 dispatch letter (the anti-controller spec: the "
            "inverted error gate, the ascent on the name-CE, the separate "
            "anti-buffer, B_E = the maintenance form's B_M, the arms, "
            "the bars, the e305 Landauer rider batched onto this cell)",
            "e291 / THE MULTI-FACT CONTENTION CELL (the vehicle's parent: "
            "the five-fact FAMILY organism + its committed baselines, the "
            "shared-frame rooms + the rooms artifact, the union-"
            "orthogonalized corpus step, the driver form — the organism "
            "loaded bit-exact, its record md5-bound)",
            "T270/e291 (the family is one organism — the antibodies are "
            "the same antibody: the selectivity test's difficulty IS the "
            "design), e293 (five DISTINCT facts not serially installable "
            "— the family construction is the only five-live-fact "
            "organism), T272/e307 (the controller's footprint = one "
            "reusable ~73-dim brush — the anti should invert it), "
            "e306 (MASS-IS-NOT-MEMORY: the bearer is the room-overlap "
            "tail)",
            "e290 / THE COUPLING-CONSTANT LADDER (the kill law's "
            "TRANSPORT channel + the passive threshold price — the "
            "Landauer table's constant; the twin-relative clause's "
            "justification)",
            "e288 / THE ERROR-GATED MAINTENANCE CELL (the controller "
            "being inverted: the gate form, the cap machinery, the 60/40 "
            "reservation, the calibration denominator — LR_ANTI_MAX "
            "transfers exactly)",
            "THE_LAWS_V2.md (the instrument: LAW 1 step-death, LAW 2(b) "
            "transport, LAW 4 the controller — the anti inverts LAW 4)",
            "e287 (the name-only CE construction + the b_m calibration "
            "trace), e272/e264/e261 (the install protocol + the SRCT "
            "rooms + the burst machinery), e273 (the lr classes, "
            "md5-bound)",
        ],
        "whats_new": [
            "THE ANTI-CONTROLLER (the record's first closed-loop "
            "ERASURE instrument): the error gate inverted — the dose on "
            "the read's EXCESS, the gradient ASCENDING the target's own "
            "loss, self-limiting at death (the dose vanishes as the read "
            "dies)",
            "THE SELECTIVITY TEST (the record's first engineered-"
            "forgetting adjudication): one fact erased while the "
            "siblings' hold is read under BOTH the absolute and the "
            "anti-corrected (twin-relative) readings, with the "
            "disagreement route to MIXED — the honest within-family form",
            "THE e305 LANDAUER LEDGER (the record's first erasure "
            "thermodynamic table): the anti's realized drift vs e290's "
            "passive-threshold price vs the bystanders' absorbed "
            "read-loss — the Maxwell-demon test of forgetting",
            "THE COLLATERAL MICROSCOPE: the full five-fact panel after "
            "EVERY anti event (16 instants) — the bleed imaged event by "
            "event",
        ],
        "gates": {},
    })
    log(f"E294 — THE ANTI-CONTROLLER CELL (smoke={SMOKE}) -> {RD}")
    log(f"arms: {' / '.join(ARMS)}; the organism = e291's five-fact FAMILY "
        f"(loaded bit-exact; THE SELECTIVITY TEST IS WITHIN-FAMILY — "
        f"disclosed); the target = {TARGET_FACT} (baseline "
        f"{E291_BASELINES_G0[TARGET_FACT]:.6f}); THE ANTI: lr_anti_t = "
        f"LR_ANTI_MAX x min(1, read_t/{DOSE_FULL_LEVEL}) with "
        f"LR_ANTI_MAX {LR_ANTI_MAX_FROZEN!r}; B_E = "
        f"{ERASE_BUDGET_SHARE:.0%} x BUDGET; M={MAINT_EVERY} -> "
        f"{N_ANTI_EXPECTED} events; the corpus stream = seed "
        f"{CORPUS_GEN_SEED} (bit-identical across arms); the bars: "
        f"SURGICAL-ERASE (FACT3 < {ERASE_BAR} AND >= {SIBLING_MIN_HOLD} "
        f"of 4 siblings hold >= {SURVIVE_FRAC:.0%}x BOTH readings AND "
        f"the CE improves/holds) / COLLATERAL / IMMUNE / MIXED")
    write_partial("startup (bars + arms + the anti's law registered, "
                  "committed at birth)")
    set_seed(CORPUS_GEN_SEED)       # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (the family's gates) ====
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

    bat_ids, held_ids_bank = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids_bank[j] = torch.stack([corpus.encode(c) for c in hs])
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

    # ---- THE FIVE-FACT SPLIT (e291's frozen convention, gated) ----------
    fact_occ = {f"FACT{i + 1}": install_occ[i * GROUP_SIZE:(i + 1)
                                            * GROUP_SIZE]
                for i in range(N_ROOMS)}
    g0_ids_f = {f"FACT{i + 1}": g0_ids[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
                for i in range(N_ROOMS)}
    gm12_ids_f = {f"FACT{i + 1}": gm12_ids[i * GROUP_SIZE:
                                           (i + 1) * GROUP_SIZE]
                  for i in range(N_ROOMS)}
    mix_per_fact = {f"FACT{i + 1}":
                    {"FLORIZEL": sum(1 for _, h in fact_occ[f"FACT{i + 1}"]
                                     if h == "FLORIZEL"),
                     "ELIZABETH": sum(1 for _, h in fact_occ[f"FACT{i + 1}"]
                                      if h == "ELIZABETH")}
                    for i in range(N_ROOMS)}
    G_FACTSPLIT = {
        "form": "e291's frozen five-fact split: the committed 60-window "
                "splice bank split into five disjoint groups of 12 "
                "(install_occ[12i:12(i+1)] in the protocol shuffle's own "
                "order); fact i := the group's contexts + its room; "
                "batteries sliced the same way",
        "mix_per_fact": mix_per_fact,
        "disjoint_cover": bool(sum(len(v) for v in fact_occ.values()) == 60),
        "battery_shapes": {k: list(v.shape) for k, v in g0_ids_f.items()},
        "pass": bool(sum(len(v) for v in fact_occ.values()) == 60
                     and all(len(set(p for p, _ in v))
                             == GROUP_SIZE for v in fact_occ.values())
                     and all(list(v.shape) == [GROUP_SIZE, G1.PRE]
                             for v in g0_ids_f.values())),
    }
    assert G_FACTSPLIT["pass"], f"fact split gate FAILED: {G_FACTSPLIT}"
    metrics["gates"]["G_FACTSPLIT"] = G_FACTSPLIT
    log("P0 G_FACTSPLIT: e291's five-fact split (12 windows each, disjoint "
        "cover; mix "
        + " ".join(f"{k}={v['FLORIZEL']}F+{v['ELIZABETH']}E"
                   for k, v in mix_per_fact.items()) + "): PASS")

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask_bank = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask_bank[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask_bank.sum()),
                  "expected": 60 * len(G1.NAME),
                  "per_fact": int(inst_mask_bank[:GROUP_SIZE].sum()),
                  "pass": bool(int(inst_mask_bank.sum()) == 60 * len(G1.NAME)
                               and int(inst_mask_bank[:GROUP_SIZE].sum())
                               == GROUP_SIZE * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    # ---- G_NAMEWIN (per fact): the TRUE install windows, decode-gated --
    name_ids = corpus.encode(G1.NAME)

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_bank = torch.stack([build_win(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_bank[i][G1.PRE: G1.PRE + len(G1.NAME)]) == G1.NAME
        for i in range(win_bank.shape[0]))
    win_pre_ok = all(torch.equal(win_bank[i][:G1.PRE],
                                 anchor_full[i][:G1.PRE])
                     for i in range(win_bank.shape[0]))
    facts_x = [win_bank[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
               for i in range(N_ROOMS)]
    facts_mask = [inst_mask_bank[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
                  for i in range(N_ROOMS)]
    G_NAMEWIN = {
        "form": "e261's build_win VERBATIM (130-token pre-context | "
                "ZEPHYRA | 119 post); the masked y-positions hold the "
                "NAME in ALL 60 windows (decode-verified) and the "
                "pre-contexts are bit-identical to the anchor bank's; "
                "FACT i's pool := its group's 12 windows — THE ANTI'S "
                "OWN VEHICLE (FACT3's pool the erase instrument's batch)",
        "n_windows": int(win_bank.shape[0]),
        "per_fact_windows": [int(x.shape[0]) for x in facts_x],
        "masked_decode_all_name": bool(win_masked_ok),
        "precontext_bit_equal_anchor": bool(win_pre_ok),
        "pass": bool(win_masked_ok and win_pre_ok
                     and list(win_bank.shape) == [60, G1.BLOCK]),
    }
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"

    # ---- G_HELD-BANK identity (rebuild-verified; NOT re-read in-phase) --
    held_pos = sorted(p for p, _ in held_occ)
    inst_pos = sorted(p for p, _ in install_occ)
    G_HELD = {
        "form": "e291's held paraphrase bank rebuilt for protocol "
                "identity ONLY (30 windows, disjoint, name-free) — NOT "
                "re-read in this cell (the dispatch's reads do not name "
                "it; disclosed); its committed t0 cited",
        "committed_t0": E291_HELD_T0,
        "positions_disjoint": bool(not set(held_pos) & set(inst_pos)),
        "pass": bool(len(held_occ) == 30
                     and not set(held_pos) & set(inst_pos)),
    }
    assert G_HELD["pass"], f"held bank gate FAILED: {G_HELD}"
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY, "G_INSTMASK": G_INSTMASK,
                             "G_NAMEWIN": G_NAMEWIN, "G_HELD": G_HELD})
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes "
        "/ install mask / THE NAME-WINDOW BIND (ZEPHYRA 60/60) / e291's "
        "FIVE-FACT SPLIT identity / the held bank identity / vocab 65)")

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
    metrics["gates"]["G_ANCHOR"] = G_ANCHOR

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e291m = json.loads(E291_METRICS.read_text(encoding="utf-8"))
    e290m = json.loads(E290_METRICS.read_text(encoding="utf-8"))
    e288m = json.loads(E288_METRICS.read_text(encoding="utf-8"))
    e287m = json.loads(E287_METRICS.read_text(encoding="utf-8"))
    e272m = json.loads(E272_METRICS.read_text(encoding="utf-8"))
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e273lr = json.loads(E273_LRCAL.read_text(encoding="utf-8"))
    e290_br = e290m["adjudication"]["threshold_bracket_realized_drift"]
    G_PARENTS = {
        "e291_metrics": {"path": str(E291_METRICS),
                         "md5": md5of(E291_METRICS), "bound_md5": E291_MD5,
                         "verdict": e291m["adjudication"]["verdict"],
                         "organism_ck_md5": md5of(GB.CKPT_DIR / ORG_CK),
                         "organism_ck_bound_md5": ORG_CK_MD5,
                         "rooms_ck_md5": md5of(GB.CKPT_DIR / ROOMS291_CK),
                         "rooms_ck_bound_md5": ROOMS291_CK_MD5,
                         "baseline_panel_g0": e291m["the_organism"]
                                           ["baseline_panel_g0"],
                         "note": "THE VEHICLE'S PARENT (the five-fact "
                                 "family organism + the rooms + the "
                                 "baselines + the driver form — all the "
                                 "cell's provenance; the organism loaded "
                                 "bit-exact)"},
        "e290_metrics": {"path": str(E290_METRICS),
                         "md5": md5of(E290_METRICS), "bound_md5": E290_MD5,
                         "verdict": e290m["adjudication"]["verdict"],
                         "drift_bracket": e290_br,
                         "budget_bracket": e290m["adjudication"]
                                           ["threshold_bracket_budget"],
                         "note": "THE KILL LAW'S CONSTANT (LAW 2(b) "
                                 "transport): the passive threshold price "
                                 "— the Landauer table's denominator and "
                                 "the twin-relative clause's "
                                 "justification"},
        "e288_metrics": {"path": str(E288_METRICS),
                         "md5": md5of(E288_METRICS), "bound_md5": E288_MD5,
                         "verdict": e288m["adjudication"]["verdict"],
                         "note": "THE CONTROLLER BEING INVERTED (the "
                                 "error gate, the cap machinery, the "
                                 "60/40 reservation — the anti's form; "
                                 "LR_ANTI_MAX transfers from its "
                                 "calibration exactly)"},
        "e287_metrics": {"path": str(E287_METRICS),
                         "md5": md5of(E287_METRICS), "bound_md5": E287_MD5,
                         "verdict": e287m["adjudication"]["verdict"],
                         "b_m_sum": E287_B_M_SUM,
                         "note": "the name-only CE construction + the "
                                 "calibration denominator"},
        "e272_metrics": {"path": str(E272_METRICS),
                         "md5": md5of(E272_METRICS), "bound_md5": E272_MD5,
                         "verdict": e272m["adjudication"]["verdict"],
                         "note": "the install protocol the organism's "
                                 "five installs followed"},
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "note": "the committed 10k rung (the "
                                 "single-fact era's reference)"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                "md5": md5of(E273_LRCAL),
                                "bound_md5": E273_LRCAL_MD5,
                                "lr_sgd": e273lr["lr_sgd"],
                                "note": "LR_STABLE's provenance"},
        "pass": bool(
            e291m["adjudication"]["verdict"] == E291_VERDICT
            and e290m["adjudication"]["verdict"] == E290_VERDICT
            and e288m["adjudication"]["verdict"] == E288_VERDICT
            and e287m["adjudication"]["verdict"] == E287_VERDICT
            and e272m["adjudication"]["verdict"] == E272_VERDICT
            and e264m["adjudication"]["verdict"] == E264_VERDICT
            and md5of(E291_METRICS) == E291_MD5
            and md5of(E290_METRICS) == E290_MD5
            and md5of(E288_METRICS) == E288_MD5
            and md5of(E287_METRICS) == E287_MD5
            and md5of(E272_METRICS) == E272_MD5
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E273_LRCAL) == E273_LRCAL_MD5
            and md5of(GB.CKPT_DIR / ORG_CK) == ORG_CK_MD5
            and md5of(GB.CKPT_DIR / ROOMS291_CK) == ROOMS291_CK_MD5
            and abs(e290_br[0] - E290_DRIFT_BRACKET[0]) < 1e-15
            and abs(e290_br[1] - E290_DRIFT_BRACKET[1]) < 1e-15
            and float(e273lr["lr_sgd"]) == E273_LR_SGD),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e291 {E291_VERDICT} (the organism ckpt "
        f"md5-bound), e290 {E290_VERDICT} (the drift bracket "
        f"[{e290_br[0]:.7f}, {e290_br[1]:.7f}] — the passive price), e288 "
        f"{E288_VERDICT}, e287 {E287_VERDICT}, e272 {E272_VERDICT}, e264 "
        f"{E264_VERDICT}, e273 lr-bound — all md5-bound")
    write_partial("P0b parents hard-bound")
    del e291m, e290m, e288m, e287m, e272m, e264m, e273lr

    # ---- G-BASE -----------------------------------------------------------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    base_panel = {f"FACT{i + 1}": G1.battery_cell(base_net, ids, zid)["mean_pz"]
                  for i, ids in enumerate(
                      [g0_ids_f[f"FACT{i + 1}"] for i in range(N_ROOMS)])}
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "base_panel_g0": base_panel,
              "fact_free": bool(base_gm12 <= 0.05
                               and all(v <= 0.05
                                       for v in base_panel.values())),
              "pass": bool(base_gm12 <= 0.05
                           and all(v <= 0.05 for v in base_panel.values()))}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free (g-12 "
        f"{base_gm12:.4f}, panel max {max(base_panel.values()):.4f}): PASS")
    write_partial("P0c G-BASE PASSED (the per-fact base panel read)")

    # ================= P1: THE FIVE ROOMS (rebuilt + certified + bound) ==
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
        "tol": E261.G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12)
                     < E261.G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"P1 G_ROOT: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {E261.G1C_ROOT_GM12:.10f}: PASS")
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
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N),
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
    rooms = SharedFrameRooms(N, ROOM_K, N_ROOMS, ROOM_D_SEED, ROOM_S_SEED,
                             v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": f"each room + the union certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY + kept^2 rank "
                "(||P x||^2/||x||^2 vs k/N at the 10-sigma bar "
                "5*sqrt(2k)/N); span-overlap ~ sqrt(k/N); the union's "
                "kept^2 vs 5k/N",
        "reads": cert,
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    for nm, r in cert["per_room"].items():
        log(f"  room {nm}: k {r['k']} idem {r['idempotency_max']:.1e} "
            f"kept2 {r['kept2_mean']:.6f} vs {r['kept2_expect']:.6f} "
            f"(bar {r['kept2_bar_10sig']:.1e}) span-ovl "
            f"{r['span_overlap_mean']:.4f} "
            f"(expect ~{r['span_overlap_expect']:.4f})")
    log(f"  the union: kept2 {cert['union_kept2_mean']:.6f} vs "
        f"{cert['union_kept2_expect']:.6f}")

    # ---- G_ROOMS5: the pairwise orthogonality + THE BIT-BIND vs the
    # committed e291 rooms artifact (the family's own vehicle bind)
    rooms_art = torch.load(GB.CKPT_DIR / ROOMS291_CK, map_location="cpu",
                           weights_only=False)
    if SMOKE:
        bit_bind = {"form": "SMOKE: k=512 rebuild — the bit-bind vs the "
                            "committed k=10k artifact VACUOUS (disclosed)",
                    "vacuous": True, "pass": True}
    else:
        D_ok = bool(np.array_equal(np.asarray(rooms_art["model"]["D"]),
                                  rooms.D))
        sets_ok = all(
            np.array_equal(np.asarray(rooms_art["model"]["sets"][i]),
                           rooms.sets[i]) for i in range(N_ROOMS))
        bit_bind = {
            "form": "the rebuilt frame BIT-COMPARED vs e291's committed "
                    "rooms artifact (D bit-exact + all five index sets "
                    "bit-exact — the vehicle's rooms are THE organism's "
                    "own rooms, not merely the same construction)",
            "D_bit_equal": D_ok, "sets_bit_equal_all": sets_ok,
            "artifact_k": rooms_art["model"].get("k"),
            "artifact_seeds": rooms_art["model"].get("seeds"),
            "vacuous": False,
            "pass": bool(D_ok and sets_ok
                         and rooms_art["model"].get("k") == ROOM_K
                         and rooms_art["model"].get("n_rooms") == N_ROOMS
                         and list(rooms_art["model"].get("seeds", []))
                         == [ROOM_D_SEED, ROOM_S_SEED]),
        }
    G_ROOMS5 = {
        "form": "THE FIVE ROOMS' pairwise orthogonality (the shared-frame "
                "construction's disjoint spectral supports — exactly "
                "zero) + THE BIT-BIND vs the committed e291 rooms",
        "pairwise_proj_max": cert["pairwise_proj_max"],
        "bar": 1e-12,
        "chance_reference": {"k_over_N": ROOM_K / N,
                             "sqrt_k_over_N": math.sqrt(ROOM_K / N)},
        "construction": {"seed_d": ROOM_D_SEED, "seed_s": ROOM_S_SEED,
                         "k": ROOM_K, "n_rooms": N_ROOMS,
                         "disjoint_slices_of_one_permutation": True},
        "bit_bind": bit_bind,
        "pass": bool(cert["pairwise_pass"] and bit_bind["pass"]),
    }
    assert G_ROOMS5["pass"], "five-room gate FAILED (pairwise or bit-bind)"
    metrics["gates"]["G_ROOMS5"] = G_ROOMS5
    pw = np.array(cert["pairwise_proj_max"])
    log(f"P1 G_ROOMS5: pairwise max {pw.max():.1e} (bar 1e-12)"
        + ("; BIT-BIND vs e291_rooms.pt: D + 5 sets bit-exact: PASS"
           if not SMOKE else "; bit-bind VACUOUS (smoke k)"))
    del rooms_art
    write_partial("P1 the five rooms rebuilt + certified + bit-bound")

    # ---- G_LR_BIND ---------------------------------------------------------
    lr_sgd_runtime = float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    lr_stable_runtime = SGD_STABLE_FACTOR * lr_sgd_runtime
    G_LR_BIND = {
        "form": "the corpus lr := e273's STABLE RIDER POINT (the committed "
                "class): LR_STABLE = x0.01 x LR_SGD_matched, re-derived at "
                "runtime from the md5-bound lr_calibration.json and "
                "asserted; momentum 0.9, wd 0.0 EXACTLY on BOTH streams "
                "(the corpus side + the anti side). THE ANTI'S CEILING: "
                "LR_ANTI_MAX = B_E / 81.57360134901982 (e287's 16-event "
                "b_m sum — the maintenance calibration's own denominator) "
                f"= {LR_ANTI_MAX_FROZEN!r} (== e288's LR_M_MAX: B_E == B_M "
                "and one stream — the calibration transfers EXACTLY)",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "lr_anti_max": LR_ANTI_MAX_FROZEN,
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
        "pass": bool(abs(lr_stable_runtime - LR_STABLE) < 1e-15
                     and float(LR_STABLE) == 0.21738574801453703
                     and abs(LR_ANTI_MAX_FROZEN - LR_M_MAX_E288) < 1e-15
                     and abs(LR_ANTI_MAX_FROZEN * E287_B_M_SUM
                             - ERASE_BUDGET_SHARE * BUDGET_NORM_FROZEN)
                     < 1e-9
                     and SGD_MOMENTUM == 0.9 and SGD_WD == 0.0
                     and lr_sgd_runtime == E273_LR_SGD),
    }
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P1 G_LR_BIND: LR_STABLE {LR_STABLE!r}; LR_ANTI_MAX "
        f"{LR_ANTI_MAX_FROZEN!r} (== e288's LR_M_MAX — the calibration "
        f"transfers exactly): PASS")
    write_partial("P1b G_LR_BIND PASSED")

    # ---- the anti's arithmetic checks (live at birth + asserted at EVERY
    # event in the driver) --------------------------------------------------
    for r_probe in (0.0, 0.025, DOSE_FULL_LEVEL, 1.0):
        d_probe = min(1.0, r_probe / DOSE_FULL_LEVEL)
        lr_probe = LR_ANTI_MAX_FROZEN * d_probe
        assert lr_probe <= LR_ANTI_MAX_FROZEN * (1.0 + 1e-12)
        if r_probe <= 0.0:
            assert lr_probe == 0.0
        if r_probe >= DOSE_FULL_LEVEL:
            assert abs(lr_probe - LR_ANTI_MAX_FROZEN) < 1e-15
    log(f"P1c the anti's law: dose(0)=0, dose({DOSE_FULL_LEVEL})=full, "
        f"dose(0.025)={LR_ANTI_MAX_FROZEN * 0.5:.9f} — asserted")

    # ================= P2: THE ORGANISM (loaded + gated) =================
    log("=" * 78)
    budget_norm = BUDGET_NORM_FROZEN
    budget_c = CORPUS_BUDGET_SHARE * budget_norm
    budget_e = ERASE_BUDGET_SHARE * budget_norm
    if SMOKE:
        budget_c = budget_c * PHASE_STEPS / 400
        budget_e = budget_e * N_ANTI_EXPECTED / 16
    metrics["budget"] = {
        "form": f"BUDGET = {BUDGET_FRAC} x {E283_WRITE_NORM} = "
                f"{budget_norm!r} (== e288/e291's frozen total); B_C = "
                f"{CORPUS_BUDGET_SHARE:.0%} (the corpus stream 1:1, both "
                f"arms); B_E = {ERASE_BUDGET_SHARE:.0%} (the anti stream's "
                "OWN pool — the maintenance form's B_M; the twin has no "
                "anti pool); S_total = S_corpus + S_anti <= BUDGET by the "
                "triangle bound BY CONSTRUCTION",
        "budget_norm": budget_norm,
        "pools": {
            ANT_ARM: {"budget_c": budget_c, "budget_e": budget_e,
                      "lr_anti_max": LR_ANTI_MAX_FROZEN,
                      "reservation": "the corpus cap over B_C per step "
                                     "(e288's form); the anti cap at each "
                                     "event = ((B_E - S_anti)/remaining "
                                     "anti events)/||b_e||"},
            TWN_ARM: {"budget_c": budget_c, "budget_e": 0.0,
                      "lr_anti_max": None,
                      "reservation": "the corpus cap over B_C per step; "
                                     "NO anti pool (the arms' ONLY delta "
                                     "the anti)"},
        },
        "slack": BUDGET_SLACK,
        "smoke_scaling": (f"SMOKE: B_C x {PHASE_STEPS}/400, B_E x "
                          f"{N_ANTI_EXPECTED}/16; LR_ANTI_MAX NOT scaled"
                          if SMOKE else None),
    }
    log(f"P2 THE BUDGET: {budget_norm!r} (== e288/e291's); B_C "
        f"{budget_c:.4f} / B_E {budget_e:.4f}; LR_ANTI_MAX "
        f"{LR_ANTI_MAX_FROZEN!r} (asserted vs the frozen literal)")

    org_art = torch.load(GB.CKPT_DIR / ORG_CK, map_location="cpu",
                         weights_only=False)
    organism_sd = org_art["model"]
    org_net = G1.evl_load(organism_sd)
    org_flat = flat_params_cpu(org_net)
    org_flat_np = org_flat.double().numpy().astype(np.float64)
    org_flat_md5 = hashlib.md5(org_flat.numpy().tobytes()).hexdigest()
    baseline_panel = {f"FACT{i + 1}": G1.battery_cell(
        org_net, g0_ids_f[f"FACT{i + 1}"], zid)["mean_pz"]
        for i in range(N_ROOMS)}
    baseline_panel12 = {f"FACT{i + 1}": G1.battery_cell(
        org_net, gm12_ids_f[f"FACT{i + 1}"], zid)["mean_pz"]
        for i in range(N_ROOMS)}
    baselines = {k: float(v) for k, v in baseline_panel.items()}
    panel_diffs = {k: abs(baselines[k] - E291_BASELINES_G0[k])
                   for k in baselines}
    panel12_diffs = {f"FACT{i + 1}": abs(baseline_panel12[f"FACT{i + 1}"]
                                         - E291_BASELINES_GM12[f"FACT{i + 1}"])
                     for i in range(N_ROOMS)}
    org_ce_r = G1.ce_fixed_cpu(org_net, *r_eval_xy)
    G_FACTLOAD = {
        "form": "THE FIVE FACT-LOADS (the cell's vehicle gate, e288's "
                "G_FACTLOAD convention applied to the family organism): "
                "the committed e291_organism.pt loaded and gated THREE "
                "ways — (i) the checkpoint md5, (ii) the flat-parameter "
                f"md5 == {ORG_FLAT_MD5} (bit-exact), (iii) the "
                "behavioral panel: every fact's g0 read within "
                f"{FACT_READ_TOL_G0} / gm12 within "
                f"{FACT_READ_TOL_GM12} of e291's committed baselines "
                "(the family's cross-session read-determinism law)",
        "checkpoint": f"runs/checkpoints/{ORG_CK}",
        "ckpt_md5": md5of(GB.CKPT_DIR / ORG_CK),
        "flat_md5": org_flat_md5,
        "baseline_panel_g0": baselines,
        "baseline_panel_gm12": {k: float(v)
                                for k, v in baseline_panel12.items()},
        "panel_g0_absdiff_vs_committed": panel_diffs,
        "panel_gm12_absdiff_vs_committed": panel12_diffs,
        "ce_r": org_ce_r,
        "pass": bool(org_flat_md5 == ORG_FLAT_MD5
                     and max(panel_diffs.values()) < FACT_READ_TOL_G0
                     and max(panel12_diffs.values()) < FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    org_write = org_flat_np - base_flat_np
    org_write_norm = float(np.linalg.norm(org_write))
    per_room_write_norms = [float(np.linalg.norm(
        rooms.project_room(org_write, i))) for i in range(N_ROOMS)]
    metrics["the_organism"] = {
        "checkpoint": f"runs/checkpoints/{ORG_CK}",
        "flat_md5": org_flat_md5,
        "baseline_panel_g0": baselines,
        "baseline_panel_gm12": {k: float(v)
                                for k, v in baseline_panel12.items()},
        "ce_r": org_ce_r, "write_norm": org_write_norm,
        "write_norm_committed_e291": E291_ORG_WRITE_NORM,
        "in_room_write_norms": per_room_write_norms,
        "disclosure": REGISTERED["family_disclosure"],
    }
    log("P2 G_FACTLOAD: the five fact-loads PASS — THE BASELINE TABLE "
        + " ".join(f"{k} {v:.6f}" for k, v in baselines.items())
        + f" | max panel diff {max(panel_diffs.values()):.2e} / gm12 "
        f"{max(panel12_diffs.values()):.2e} | flat md5 {org_flat_md5} "
        "(bit-exact) | the write "
        f"{org_write_norm:.4f} (committed {E291_ORG_WRITE_NORM:.4f})")
    del org_net, org_art

    # ---- G_F3WRITE: FACT3's own install write (the Landauer denominator)
    f2_art = torch.load(GB.CKPT_DIR / INST_F2_CK, map_location="cpu",
                        weights_only=False)
    f3_art = torch.load(GB.CKPT_DIR / INST_F3_CK, map_location="cpu",
                        weights_only=False)
    assert int(f2_art.get("step", -1)) == 400 and \
        int(f3_art.get("step", -1)) == 400, "install resume ckpts incomplete"
    f2_flat = flat_params_cpu(G1.evl_load(f2_art["model"])) \
        .double().numpy().astype(np.float64)
    f3_flat = flat_params_cpu(G1.evl_load(f3_art["model"])) \
        .double().numpy().astype(np.float64)
    f3_write = f3_flat - f2_flat
    f3_write_norm = float(np.linalg.norm(f3_write))
    f3_write_in_own_room = float(np.linalg.norm(
        rooms.project_room(f3_write, TARGET_IDX)))
    G_F3WRITE = {
        "form": "FACT3's OWN INSTALL WRITE (the Landauer price's "
                "denominator — e290's law is per the fact's write norm): "
                "||flat(F3 install end) - flat(F2 install end)|| from "
                "e291's committed install checkpoints, both md5-bound + "
                "step-400-complete-gated",
        "f2_ckpt_md5": md5of(GB.CKPT_DIR / INST_F2_CK),
        "f2_ckpt_bound_md5": INST_F2_CK_MD5,
        "f3_ckpt_md5": md5of(GB.CKPT_DIR / INST_F3_CK),
        "f3_ckpt_bound_md5": INST_F3_CK_MD5,
        "f3_write_norm": f3_write_norm,
        "f3_write_in_own_room_norm": f3_write_in_own_room,
        "f3_write_in_own_room_frac": (f3_write_in_own_room / f3_write_norm
                                      if f3_write_norm > 0 else None),
        "pass": bool(md5of(GB.CKPT_DIR / INST_F2_CK) == INST_F2_CK_MD5
                     and md5of(GB.CKPT_DIR / INST_F3_CK) == INST_F3_CK_MD5
                     and f3_write_norm > 0),
    }
    assert G_F3WRITE["pass"], f"G_F3WRITE FAILED: {G_F3WRITE}"
    metrics["gates"]["G_F3WRITE"] = G_F3WRITE
    del f2_art, f3_art
    price_lo = E290_DRIFT_BRACKET[0] * f3_write_norm
    price_hi = E290_DRIFT_BRACKET[1] * f3_write_norm
    log(f"P2 G_F3WRITE: ||F3WRITE|| {f3_write_norm:.6f} (in-own-room "
        f"{f3_write_in_own_room:.4f} = "
        f"{f3_write_in_own_room / f3_write_norm:.1%}) — THE PASSIVE PRICE "
        f"BRACKET [{price_lo:.6f}, {price_hi:.6f}] (e290 x F3's write); "
        f"organism-scaling alternative "
        f"[{E290_DRIFT_BRACKET[0] * E291_ORG_WRITE_NORM:.4f}, "
        f"{E290_DRIFT_BRACKET[1] * E291_ORG_WRITE_NORM:.4f}] (co-report)")
    write_partial("P2 the organism loaded + gated + F3's write priced")

    # ---- G_ASCENT: THE ASCENT-DIRECTION GATE (the smoke/birth check the
    # dispatch ordered: one full-dose anti step must LOWER FACT3's read) --
    log("=" * 78)
    t_b, n_b = E261._open_burst(f"{ANT_ARM}:ascent-probe")
    probe_net = G1.evl_load(organism_sd).to(dev)
    probe_net.train()
    probe_opt = torch.optim.SGD(probe_net.parameters(),
                                lr=LR_ANTI_MAX_FROZEN,
                                momentum=SGD_MOMENTUM, weight_decay=SGD_WD)
    probe_gen = torch.Generator().manual_seed(ANTI_GEN_SEED)
    evl_p = G1.evl_load(organism_sd)
    panel_pre = {f"FACT{i + 1}": G1.battery_cell(evl_p, ids, zid)["mean_pz"]
                 for i, ids in enumerate(
                     [g0_ids_f[f"FACT{i + 1}"] for i in range(N_ROOMS)])}
    ix_p = torch.randint(facts_x[TARGET_IDX].shape[0], (G1.NAME_BS,),
                         generator=probe_gen)
    nw_p = facts_x[TARGET_IDX][ix_p]
    x_p = nw_p[:, :-1].to(dev)
    y_p = nw_p[:, 1:].to(dev)
    m_p = facts_mask[TARGET_IDX][ix_p].to(dev)
    logits_p, _ = probe_net(x_p)
    nll_p = F.cross_entropy(logits_p.reshape(-1, logits_p.shape[-1]),
                            y_p.reshape(-1), reduction="none"
                            ).view(x_p.shape[0], x_p.shape[1])
    ce_pre = float(nll_p[m_p].mean().item())
    loss_p = -nll_p[m_p].mean()          # ASCENT on the name CE
    probe_net.zero_grad(set_to_none=True)
    loss_p.backward()
    torch.nn.utils.clip_grad_norm_(probe_net.parameters(), 1.0)
    theta_pre = torch.cat([p.detach().reshape(-1)
                           for p in probe_net.parameters()])
    probe_opt.step()
    d_probe_vec = torch.cat([p.detach().reshape(-1)
                             for p in probe_net.parameters()]) - theta_pre
    d_probe_norm = float(d_probe_vec.norm().item())
    sd_post = {k: v.detach().cpu().clone()
               for k, v in probe_net.state_dict().items()}
    evl_p.load_state_dict(sd_post)
    evl_p.eval()
    panel_post = {f"FACT{i + 1}": G1.battery_cell(evl_p, ids, zid)["mean_pz"]
                  for i, ids in enumerate(
                      [g0_ids_f[f"FACT{i + 1}"] for i in range(N_ROOMS)])}
    logits_p2, _ = probe_net(x_p)
    nll_p2 = F.cross_entropy(logits_p2.reshape(-1, logits_p2.shape[-1]),
                             y_p.reshape(-1), reduction="none"
                             ).view(x_p.shape[0], x_p.shape[1])
    ce_post = float(nll_p2[m_p].mean().item())
    G_ASCENT = {
        "form": "THE ASCENT-DIRECTION GATE (the dispatch's smoke order, "
                "run at BIRTH on a scratch copy of the organism): ONE "
                "full-dose anti step (lr = LR_ANTI_MAX; the loaded read "
                f"{baselines[TARGET_FACT]:.4f} >= {DOSE_FULL_LEVEL} -> "
                "dose 1) must LOWER FACT3's read — the direction verified "
                "before any phase; the one-stroke panel deltas recorded "
                "(the first collateral read)",
        "name_ce_pre": ce_pre, "name_ce_post": ce_post,
        "ce_rose": bool(ce_post > ce_pre),
        "panel_pre": panel_pre, "panel_post": panel_post,
        "target_pre": panel_pre[TARGET_FACT],
        "target_post": panel_post[TARGET_FACT],
        "target_drop": float(panel_pre[TARGET_FACT]
                             - panel_post[TARGET_FACT]),
        "target_drop_frac": float(1.0 - panel_post[TARGET_FACT]
                                  / panel_pre[TARGET_FACT]),
        "bystander_deltas": {k: float(panel_post[k] - panel_pre[k])
                             for k in FACTS if k != TARGET_FACT},
        "one_stroke_displacement": d_probe_norm,
        "pass": bool(panel_post[TARGET_FACT] < panel_pre[TARGET_FACT]
                     and ce_post > ce_pre),
    }
    n_b += 1
    E261.burst_temp_check(f"{ANT_ARM}:ascent-probe.x")
    E261._end_burst_early(f"{ANT_ARM}:ascent-probe", n_b, t_b)
    del probe_net, evl_p
    assert G_ASCENT["pass"], f"G_ASCENT FAILED (the direction wrong?!): " \
                             f"{G_ASCENT}"
    metrics["gates"]["G_ASCENT"] = G_ASCENT
    log(f"P2 G_ASCENT: one full-dose stroke LOWERS FACT3 "
        f"{panel_pre[TARGET_FACT]:.6f} -> {panel_post[TARGET_FACT]:.6f} "
        f"(drop {G_ASCENT['target_drop']:.6f} = "
        f"{G_ASCENT['target_drop_frac']:.1%}; name CE {ce_pre:.4f} -> "
        f"{ce_post:.4f}; |stroke| {d_probe_norm:.5f}) | the bystanders' "
        "one-stroke deltas "
        + " ".join(f"{k} {v:+.5f}" for k, v
                   in G_ASCENT["bystander_deltas"].items())
        + ": PASS")
    write_partial("P2b G_ASCENT PASSED (the ascent direction verified)")

    # ================= P3: ARM ANTI-ON-FACT3 =============================
    log("=" * 78)
    log(f"ARM-{ANT_ARM} — {ARM_DESC[ANT_ARM]}")
    ant = anti_phase(
        f"{ANT_ARM}", True, organism_sd, rooms, org_flat_np, base_flat_np,
        budget_c, budget_e, LR_ANTI_MAX_FROZEN,
        facts_x, facts_mask, baselines, g0_ids_f, gm12_ids_f,
        anchor_full, train_ids, r_eval_xy, zid,
        CKPT_DIR / (f"smoke_{NAME}_{ANT_ARM}_resume.pt" if SMOKE
                    else f"{NAME}_{ANT_ARM}_resume.pt"), dev)
    sd_a = ant["sd"]
    net_a = G1.evl_load(sd_a)
    final_panel_a = {f"FACT{i + 1}": G1.battery_cell(
        net_a, g0_ids_f[f"FACT{i + 1}"], zid)["mean_pz"]
        for i in range(N_ROOMS)}
    ce_r_a = G1.ce_fixed_cpu(net_a, *r_eval_xy)
    d_final_a = flat_params_cpu(net_a) - org_flat
    loads_final_a = rooms.displacement_loads(d_final_a)
    del net_a
    ant_ck = save_ckpt(f"{NAME}_{ANT_ARM}_post", sd_a,
                       {"desc": "e294 ARM-ANTI-ON-FACT3 post-phase state — "
                                "NO cons", "arm": ANT_ARM,
                        "organism": f"runs/checkpoints/{ORG_CK}"})
    corp_ce_a = [v["ce"] for v in ant["corpus_ledger"].values()]
    metrics["arms"] = {ANT_ARM: {
        "desc": ARM_DESC[ANT_ARM], "phase": {
            "traj": ant["traj"], "corpus_ledger": ant["corpus_ledger"],
            "orth_ledger": ant["orth_ledger"],
            "disp_ledger": ant["disp_ledger"],
            "buf_ledger": ant["buf_ledger"],
            "budget_ledger": ant["budget_ledger"],
            "lr_ledger": ant["lr_ledger"],
            "anti_ledger": ant["anti_ledger"],
            "event_panel_ledger": ant["event_panel_ledger"],
            "corpus_ce_median": med(corp_ce_a),
            "orth_max_rel_err": ant["orth_max"],
            "bufsep": ant["bufsep"],
            "final_panel_g0": final_panel_a,
            "post_ce_r": ce_r_a,
            "drift_from_organism_final": {
                "norm": float(np.linalg.norm(d_final_a.double().numpy())),
                **loads_final_a},
            "anti_cum_norm_final": ant["anti_cum_norm_final"],
            "t_erase": ant["t_erase"],
            "work_at_erase": ant["work_at_erase"],
            "spent_at_erase": ant["spent_at_erase"],
            "chunk_table": ant["chunk_table"], "steps": PHASE_STEPS,
            "checkpoint": ant_ck,
            "resumed_final": bool(ant.get("resumed_final", False)),
        }}}
    log(f"ARM-{ANT_ARM} DONE: final panel "
        + " ".join(f"{k} {v:.6f}" for k, v in final_panel_a.items())
        + f" | S_corp {ant['S_corpus']:.4f} S_anti {ant['S_anti']:.4f} | "
        f"||anti_cum|| {ant['anti_cum_norm_final']:.4f} | anti events "
        f"{ant['n_anti']} | t_erase "
        f"{ant['t_erase']} (work {ant['work_at_erase']}) | ORTH max "
        f"{ant['orth_max']:.2e}")
    write_partial(f"ARM-{ANT_ARM} complete")

    # ================= P4: ARM NO-ANTI (the twin) ========================
    E261.burst_cooldown(f"{ANT_ARM} -> {TWN_ARM}")
    log("=" * 78)
    log(f"ARM-{TWN_ARM} — {ARM_DESC[TWN_ARM]}")
    twn = anti_phase(
        f"{TWN_ARM}", False, organism_sd, rooms, org_flat_np, base_flat_np,
        budget_c, 0.0, LR_ANTI_MAX_FROZEN,
        facts_x, facts_mask, baselines, g0_ids_f, gm12_ids_f,
        anchor_full, train_ids, r_eval_xy, zid,
        CKPT_DIR / (f"smoke_{NAME}_{TWN_ARM}_resume.pt" if SMOKE
                    else f"{NAME}_{TWN_ARM}_resume.pt"), dev)
    sd_t = twn["sd"]
    net_t = G1.evl_load(sd_t)
    final_panel_t = {f"FACT{i + 1}": G1.battery_cell(
        net_t, g0_ids_f[f"FACT{i + 1}"], zid)["mean_pz"]
        for i in range(N_ROOMS)}
    ce_r_t = G1.ce_fixed_cpu(net_t, *r_eval_xy)
    d_final_t = flat_params_cpu(net_t) - org_flat
    loads_final_t = rooms.displacement_loads(d_final_t)
    del net_t
    twn_ck = save_ckpt(f"{NAME}_{TWN_ARM}_post", sd_t,
                       {"desc": "e294 ARM-NO-ANTI post-phase state — NO "
                                "cons", "arm": TWN_ARM,
                        "organism": f"runs/checkpoints/{ORG_CK}"})
    corp_ce_t = [v["ce"] for v in twn["corpus_ledger"].values()]
    metrics["arms"][TWN_ARM] = {
        "desc": ARM_DESC[TWN_ARM], "phase": {
            "traj": twn["traj"], "corpus_ledger": twn["corpus_ledger"],
            "orth_ledger": twn["orth_ledger"],
            "disp_ledger": twn["disp_ledger"],
            "buf_ledger": twn["buf_ledger"],
            "budget_ledger": twn["budget_ledger"],
            "lr_ledger": twn["lr_ledger"],
            "corpus_ce_median": med(corp_ce_t),
            "orth_max_rel_err": twn["orth_max"],
            "bufsep": twn["bufsep"],
            "final_panel_g0": final_panel_t,
            "post_ce_r": ce_r_t,
            "drift_from_organism_final": {
                "norm": float(np.linalg.norm(d_final_t.double().numpy())),
                **loads_final_t},
            "chunk_table": twn["chunk_table"], "steps": PHASE_STEPS,
            "checkpoint": twn_ck,
            "resumed_final": bool(twn.get("resumed_final", False)),
        }}
    log(f"ARM-{TWN_ARM} DONE: final panel "
        + " ".join(f"{k} {v:.6f}" for k, v in final_panel_t.items())
        + f" | S_corp {twn['S_corpus']:.4f} | ORTH max "
        f"{twn['orth_max']:.2e}")
    write_partial(f"ARM-{TWN_ARM} complete")

    # ================= P5: THE ADJUDICATION + THE LANDAUER TABLE =========
    log("=" * 78)

    def stream_live_of(corpus_ledger: dict) -> dict:
        rows = sorted((int(k), v["ce"]) for k, v in corpus_ledger.items())
        early = [c for t, c in rows if 1 <= t <= 100]
        late = [c for t, c in rows if 300 < t <= 400]
        me, ml = (med(early) if early else None,
                  med(late) if late else None)
        improving = bool(me is not None and ml is not None and ml < me)
        stable = bool(me is not None and ml is not None
                      and ml <= STREAM_STABLE_TOL * me)
        return {"early_median": me, "late_median": ml,
                "improving": improving, "stable_1p05": stable,
                "live": bool(improving or stable)}

    stream_a = stream_live_of(ant["corpus_ledger"])
    stream_t = stream_live_of(twn["corpus_ledger"])

    # the budget gates (computed, NON-halting — the family's convention)
    S_total_a = ant["S_corpus"] + ant["S_anti"]
    G_BUDGET = {
        "arm_a": {"S_corpus": ant["S_corpus"], "S_anti": ant["S_anti"],
                  "S_total": S_total_a, "budget_norm": budget_norm,
                  "usage_frac": S_total_a / budget_norm,
                  "within_B_C": bool(ant["S_corpus"]
                                     <= budget_c + BUDGET_SLACK),
                  "within_B_E": bool(ant["S_anti"]
                                     <= budget_e + BUDGET_SLACK)},
        "twin": {"S_corpus": twn["S_corpus"],
                 "S_total": twn["S_corpus"],
                 "within_B_C": bool(twn["S_corpus"]
                                    <= budget_c + BUDGET_SLACK)},
        "form": "S_total = S_corpus + S_anti <= B_C + B_E = BUDGET by the "
                "triangle bound over the split pools; NON-halting — a "
                "blow routes the composite to MIXED",
        "pass": bool(S_total_a <= budget_norm + BUDGET_SLACK
                     and twn["S_corpus"] <= budget_c + BUDGET_SLACK),
    }
    G_ORTH = {
        "arm_a_max_rel_err": ant["orth_max"],
        "twin_max_rel_err": twn["orth_max"],
        "bar": ORTH_BAR,
        "pass": bool(ant["orth_max"] < ORTH_BAR
                     and twn["orth_max"] < ORTH_BAR),
    }
    G_ANTIBIND = {
        "form": "the anti's law + the counted events: exactly "
                f"{N_ANTI_EXPECTED} anti steps (machine-counted), the "
                "dose law asserted at EVERY event (never above the "
                "ceiling; 0 at read 0; full at read >= 0.05), the "
                "isolation probes clean",
        "n_anti": ant["n_anti"], "expected": N_ANTI_EXPECTED,
        "anti_violations": ant["bufsep"]["anti_violations"],
        "corpus_violations": ant["bufsep"]["corpus_violations"],
        "twin_anti_events": twn["n_anti"],
        "pass": bool(ant["n_anti"] == N_ANTI_EXPECTED
                     and twn["n_anti"] == 0
                     and ant["bufsep"]["anti_violations"] == 0
                     and ant["bufsep"]["corpus_violations"] == 0),
    }
    assert G_ANTIBIND["pass"], f"G_ANTIBIND FAILED: {G_ANTIBIND}"
    metrics["gates"].update({"G_BUDGET": G_BUDGET, "G_ORTH": G_ORTH,
                             "G_ANTIBIND": G_ANTIBIND})
    log(f"P5 standing conditions: G_BUDGET "
        f"{'PASS' if G_BUDGET['pass'] else 'BLOWN (-> MIXED)'} (arm a "
        f"{S_total_a:.4f}/{budget_norm:.4f} = "
        f"{S_total_a / budget_norm:.1%}; twin {twn['S_corpus']:.4f}/"
        f"{budget_c:.4f}); G_ORTH "
        f"{'PASS' if G_ORTH['pass'] else 'FAILED (-> MIXED)'} (max "
        f"{max(ant['orth_max'], twn['orth_max']):.2e}); G_ANTIBIND PASS "
        f"({ant['n_anti']} anti events, 0 isolation violations); stream "
        f"live arm a: {stream_a['live']} (early {stream_a['early_median']}"
        f" late {stream_a['late_median']}), twin: {stream_t['live']}")

    R3 = final_panel_a[TARGET_FACT]
    T3_twn = final_panel_t[TARGET_FACT]
    erased = bool(R3 < ERASE_BAR)
    sib_abs, sib_twn = {}, {}
    for f_ in FACTS:
        if f_ == TARGET_FACT:
            continue
        sib_abs[f_] = bool(final_panel_a[f_] >= SURVIVE_FRAC
                           * baselines[f_])
        sib_twn[f_] = bool(final_panel_a[f_] >= SURVIVE_FRAC
                           * final_panel_t[f_])
    n_abs, n_twn = sum(sib_abs.values()), sum(sib_twn.values())
    immune = bool(R3 >= SURVIVE_FRAC * baselines[TARGET_FACT])

    # THE e305 LANDAUER LEDGER
    work = (ant["work_at_erase"] if erased
            else ant["anti_cum_norm_final"])
    spent = (ant["spent_at_erase"] if erased else ant["S_anti"])
    landauer = {
        "work_definition": "||anti_cum|| (the anti's REALIZED DRIFT) at "
                           "the erase instant" + ("" if erased else
                                                  " [NOT ERASED — t400, "
                                                  "the immunity case "
                                                  "priced]"),
        "t_erase": ant["t_erase"],
        "work": work, "spent_S_anti": spent,
        "f3_write_norm": f3_write_norm,
        "passive_price_lo": price_lo, "passive_price_hi": price_hi,
        "price_bracket_form": f"e290's committed realized-drift bracket "
                              f"{E290_DRIFT_BRACKET} x ||F3WRITE|| "
                              f"{f3_write_norm:.6f}",
        "organism_scaling_alt": {
            "write_norm": E291_ORG_WRITE_NORM,
            "price_lo": E290_DRIFT_BRACKET[0] * E291_ORG_WRITE_NORM,
            "price_hi": E290_DRIFT_BRACKET[1] * E291_ORG_WRITE_NORM},
        "twin_passive_reference": {
            "twin_drift_from_organism": float(np.linalg.norm(
                d_final_t.double().numpy())),
            "twin_S_corpus": twn["S_corpus"],
            "twin_fact3_read": T3_twn,
            "twin_fact3_ratio_vs_baseline": T3_twn
            / baselines[TARGET_FACT],
            "note": "the in-cell passive read: what the orthogonal corpus "
                    "stream alone did to FACT3 + the organism"},
        "waste_heat_bystanders": {},
        "directed_effect": {
            "fact3_arm_a": R3, "fact3_twin": T3_twn,
            "fact3_baseline": baselines[TARGET_FACT],
            "drop_vs_twin": float(T3_twn - R3),
            "drop_vs_baseline": float(baselines[TARGET_FACT] - R3),
            "ratio_vs_baseline": R3 / baselines[TARGET_FACT],
            "ratio_vs_twin": (R3 / T3_twn if T3_twn > 0 else None)},
    }
    for f_ in FACTS:
        if f_ == TARGET_FACT:
            continue
        landauer["waste_heat_bystanders"][f_] = {
            "read_arm_a": final_panel_a[f_], "read_twin": final_panel_t[f_],
            "read_baseline": baselines[f_],
            "absorbed_vs_twin": float(final_panel_t[f_]
                                      - final_panel_a[f_]),
            "ratio_vs_twin": final_panel_a[f_] / final_panel_t[f_],
            "ratio_vs_baseline": final_panel_a[f_] / baselines[f_],
            "twin_ratio_vs_baseline": final_panel_t[f_] / baselines[f_],
        }
    if work < price_lo:
        rider = "DIRECTED-CHEAPER"
    elif work <= price_hi:
        rider = "PRICED-AT-THE-CONSTANT"
    else:
        rider = "ERASURE-COSTS-MORE"

    # THE COMPOSITE (frozen ladder)
    reasons = []
    if not G_BUDGET["pass"]:
        verdict = "MIXED"
        reasons.append("G_BUDGET blown (a standing condition; everything "
                       "verbatim)")
    elif not G_ORTH["pass"]:
        verdict = "MIXED"
        reasons.append("G_ORTH failed (a standing condition)")
    elif not stream_a["live"]:
        verdict = "MIXED"
        reasons.append("the corpus stream not live on arm (a) (the CE "
                       "clause's standing form)")
    elif erased and n_abs >= SIBLING_MIN_HOLD and n_twn >= SIBLING_MIN_HOLD:
        verdict = "SURGICAL-ERASE"
        reasons.append(f"FACT3 {R3:.6f} < {ERASE_BAR}; siblings hold "
                       f"{n_abs}/4 ABS + {n_twn}/4 TWIN-REL; CE "
                       f"{'improving' if stream_a['improving'] else 'stable'}"
                       " — the first engineered forgetting")
    elif erased and n_abs <= 2 and n_twn <= 2:
        verdict = "COLLATERAL"
        reasons.append(f"FACT3 {R3:.6f} < {ERASE_BAR} but the siblings "
                       f"fall ({n_abs}/4 ABS + {n_twn}/4 TWIN-REL hold) — "
                       "the shared representation bleeds; the honest "
                       "within-family reading")
    elif not erased and immune:
        verdict = "IMMUNE"
        reasons.append(f"FACT3 {R3:.6f} >= "
                       f"{SURVIVE_FRAC:.0%} x baseline "
                       f"{baselines[TARGET_FACT]:.4f} (x"
                       f"{R3 / baselines[TARGET_FACT]:.3f}) — the read "
                       "resists ascent; a stability law")
    else:
        verdict = "MIXED"
        if erased:
            reasons.append(
                f"FACT3 erased ({R3:.6f} < {ERASE_BAR}) but the sibling "
                f"readings DISAGREE or partially hold (ABS {n_abs}/4, "
                f"TWIN {n_twn}/4) — the anti-vs-passive decomposition "
                "confounded; both tables shown")
        else:
            reasons.append(
                f"FACT3 eroded but neither erased (>= {ERASE_BAR}) nor "
                f"immune (< {SURVIVE_FRAC:.0%} x baseline): "
                f"{R3:.6f} = x{R3 / baselines[TARGET_FACT]:.3f} baseline, "
                f"x{R3 / T3_twn:.3f} the twin — the middle zone")
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        reasons = ["smoke mode: every read carries the SMOKE stamp"]

    clause = ("; ".join(reasons) + f" | THE e305 RIDER: {rider} (work "
              f"{work:.6f} vs the passive price bracket "
              f"[{price_lo:.6f}, {price_hi:.6f}]; spent S_anti "
              f"{spent:.4f}; t_erase {ant['t_erase']}) | THE SELECTIVITY "
              "TEST IS WITHIN-FAMILY (the siblings share representation "
              "— disclosed)")

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "rider_bars_verbatim": REGISTERED["rider_bars_verbatim"],
        "composite_order": "TEXTURE (hard-gate failure -> HALT) -> "
                           "MIXED-by-conditions (G_BUDGET OR G_ORTH OR "
                           "stream not live) -> SURGICAL-ERASE (erased "
                           "AND BOTH sibling readings >= 3/4) -> "
                           "COLLATERAL (erased AND BOTH <= 2/4) -> IMMUNE "
                           "(>= 0.5x baseline) -> MIXED (else — incl. the "
                           "disagreement case + the eroded middle)",
        "gates_pass": bool(G_BUDGET["pass"] and G_ORTH["pass"]
                           and G_ANTIBIND["pass"]),
        "halt_gates_pass": True,
        "reads": {
            ANT_ARM: {
                "final_panel_g0": final_panel_a,
                "ratios_vs_baseline": {k: final_panel_a[k] / baselines[k]
                                       for k in FACTS},
                "fact3_read": R3, "erased": erased,
                "n_siblings_hold_abs": n_abs,
                "n_siblings_hold_twin_relative": n_twn,
                "sibling_hold_abs": sib_abs,
                "sibling_hold_twin_relative": sib_twn,
                "immune": immune,
                "stream": stream_a,
                "S_corpus": ant["S_corpus"], "S_anti": ant["S_anti"],
                "n_anti": ant["n_anti"],
                "t_erase": ant["t_erase"],
            },
            TWN_ARM: {
                "final_panel_g0": final_panel_t,
                "ratios_vs_baseline": {k: final_panel_t[k] / baselines[k]
                                       for k in FACTS},
                "stream": stream_t,
                "S_corpus": twn["S_corpus"],
            },
        },
        "landauer_table": landauer,
        "rider_verdict": rider,
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": SMOKE,
    }
    log(f"P5 ADJUDICATION: {verdict} — {clause}")

    # ---- the draw-integrity check (the arms' corpus streams bit-identical)
    ce_a_rows = {k: v["ce"] for k, v in ant["corpus_ledger"].items()}
    ce_t_rows = {k: v["ce"] for k, v in twn["corpus_ledger"].items()}
    shared = sorted(set(ce_a_rows) & set(ce_t_rows))
    metrics["adjudication"]["draw_integrity"] = {
        "n_shared_corpus_rows": len(shared),
        "max_ce_absdiff": max((abs(ce_a_rows[k] - ce_t_rows[k])
                               for k in shared), default=None),
        "note": "the arms share the corpus generator seed: the corpus CE "
                "rows agree UNTIL the anti events begin moving the state "
                "(the arms' ONLY delta the anti)",
    }
    metrics["outputs"] = [
        str(RD / "metrics.json"), str(RD / "e294_anticontroller.png"),
        str(RD / "REPORT.md")]
    metrics["envelope_summary"] = _envelope_summary()
    metrics["status"] = ("SMOKE COMPLETE (nothing adjudicated)" if SMOKE
                         else "COMPLETE")
    write_partial("P5 adjudicated" + (" (SMOKE)" if SMOKE else ""))

    # ================= P6: THE FIGURE ====================================
    traj_a = ant["traj"]
    traj_t = twn["traj"]
    anti_led = ant["anti_ledger"]
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle(
        f"e294 — THE ANTI-CONTROLLER (targeted forgetting, the kill law "
        f"inverted) | verdict: {verdict} | rider: {rider}\nTHE SELECTIVITY "
        "TEST IS WITHIN-FAMILY (the siblings share representation — "
        "disclosed)", fontsize=11)

    # (1) the five facts' reads, arm (a) vs twin
    ax = axs[0, 0]
    cols = {"FACT1": "#1f77b4", "FACT2": "#2ca02c", "FACT3": "#d62728",
            "FACT4": "#9467bd", "FACT5": "#8c564b"}
    ts = [r["step"] for r in traj_a]
    for f_ in FACTS:
        ax.plot(ts, [r["panel_g0"][f_] for r in traj_a], "-o", ms=3,
                color=cols[f_], label=f"{f_} (anti arm)")
        ax.plot([r["step"] for r in traj_t],
                [r["panel_g0"][f_] for r in traj_t], "--", alpha=0.55,
                color=cols[f_], label=f"{f_} (twin)")
    ax.axhline(ERASE_BAR, color="k", ls=":", lw=1)
    ax.axhline(SURVIVE_FRAC * baselines[TARGET_FACT], color="#d62728",
               ls=":", lw=1, alpha=0.7)
    ax.text(ts[-1], ERASE_BAR, " erase bar 0.01", va="bottom", fontsize=8)
    ax.set_xlabel("corpus step"); ax.set_ylabel("g0 read (mean pZ)")
    ax.set_title("THE FIVE FACTS — anti arm solid / NO-ANTI twin dashed")
    ax.legend(fontsize=6, ncol=2)

    # (2) FACT3: the target picture
    ax = axs[0, 1]
    ax.plot(ts, [r["panel_g0"][TARGET_FACT] for r in traj_a], "-o",
            color="#d62728", ms=4, label="FACT3 (anti arm)")
    ax.plot([r["step"] for r in traj_t],
            [r["panel_g0"][TARGET_FACT] for r in traj_t], "--",
            color="#d62728", alpha=0.6, label="FACT3 (twin)")
    ax.axhline(ERASE_BAR, color="k", ls=":", lw=1)
    ax.axhline(baselines[TARGET_FACT], color="gray", ls=":", lw=1)
    ax.axhline(SURVIVE_FRAC * baselines[TARGET_FACT], color="gray",
               ls=":", lw=1, alpha=0.6)
    if anti_led:
        ev_ts = [r["step"] for r in anti_led]
        ax.plot(ev_ts, [r["read_at_gate"] for r in anti_led], "x",
                color="k", ms=4, alpha=0.6, label="the gate read/event")
        if ant["t_erase"] is not None:
            ax.axvline(ant["t_erase"], color="#d62728", ls=":", lw=1.5)
            ax.text(ant["t_erase"], baselines[TARGET_FACT] * 0.55,
                    f" t_erase={ant['t_erase']}", color="#d62728",
                    fontsize=8)
    ax.set_xlabel("corpus step"); ax.set_ylabel("FACT3 g0 read")
    ax.set_title("THE TARGET: FACT3 vs the erase bar / baseline")
    ax.legend(fontsize=8)

    # (3) the anti trace: gate read + dose + lr
    ax = axs[0, 2]
    if anti_led:
        ev_ts = [r["step"] for r in anti_led]
        ax.plot(ev_ts, [r["read_at_gate"] for r in anti_led], "-o", ms=3,
                color="#d62728", label="gate read (pre-event)")
        ax.plot(ev_ts, [r["panel_g0_post"][TARGET_FACT] for r in anti_led],
                "-s", ms=3, color="#ff9896",
                label="FACT3 post-event (panel)")
        ax2 = ax.twinx()
        ax2.plot(ev_ts, [r["lr_anti"] for r in anti_led], "-^", ms=3,
                 color="#1f77b4", label="applied lr")
        ax2.plot(ev_ts, [r["name_ce"] for r in anti_led], "-.", ms=3,
                 color="#2ca02c", alpha=0.7, label="name CE (ascended)")
        ax2.set_yscale("log")
        ax2.set_ylabel("lr / name-CE (log)", fontsize=8)
        ax2.legend(fontsize=7, loc="center right")
        ax.axhline(DOSE_FULL_LEVEL, color="k", ls=":", lw=1)
    ax.set_xlabel("corpus step"); ax.set_ylabel("read")
    ax.set_title("THE ANTI TRACE (the inverted gate)")

    # (4) the bystanders: arm a vs twin (the collateral bill)
    ax = axs[1, 0]
    sibs = [f_ for f_ in FACTS if f_ != TARGET_FACT]
    x = np.arange(len(sibs))
    w = 0.27
    ax.bar(x - w, [baselines[f_] for f_ in sibs], w, color="gray",
           alpha=0.5, label="baseline")
    ax.bar(x, [final_panel_t[f_] for f_ in sibs], w, color="#1f77b4",
           alpha=0.7, label="twin (passive)")
    ax.bar(x + w, [final_panel_a[f_] for f_ in sibs], w, color="#d62728",
           alpha=0.7, label="anti arm")
    ax.axhline(SURVIVE_FRAC * min(baselines[f_] for f_ in sibs), ls=":",
               lw=1, color="k")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f_}\nABS{'+' if sib_abs[f_] else '-'} "
                        f"TWIN{'+' if sib_twn[f_] else '-'}"
                        for f_ in sibs], fontsize=8)
    ax.set_ylabel("g0 read at t400")
    ax.set_title("THE BYSTANDERS at t400 (hold marks: ABS/TWIN)")
    ax.legend(fontsize=8)

    # (5) the budget
    ax = axs[1, 1]
    bl = ant["budget_ledger"]
    ax.plot([r["step"] for r in bl], [r["S_corpus"] for r in bl], "-",
            color="#1f77b4", label=f"S_corpus (B_C {budget_c:.3f})")
    ax.plot([r["step"] for r in bl], [r["S_anti"] for r in bl], "-",
            color="#d62728", label=f"S_anti (B_E {budget_e:.3f})")
    ax.plot([r["step"] for r in bl], [r["S_total"] for r in bl], "-",
            color="k", alpha=0.6,
            label=f"S_total (BUDGET {budget_norm:.3f})")
    ax.plot([r["step"] for r in bl], [r["cum_anti_norm"] for r in bl],
            ":", color="#d62728", label="||anti_cum|| (the WORK)")
    ax.axhline(budget_c, ls=":", lw=1, color="#1f77b4")
    ax.axhline(budget_e, ls=":", lw=1, color="#d62728")
    ax.set_xlabel("corpus step"); ax.set_ylabel("displacement")
    ax.set_title("THE BUDGET (the split pools)")
    ax.legend(fontsize=7)

    # (6) THE LANDAUER TABLE (the e305 rider)
    ax = axs[1, 2]
    ax.barh(["PASSIVE\nprice_lo", "PASSIVE\nprice_hi",
             "WORK\n(||anti_cum||)", "SPENT\n(S_anti)"],
            [price_lo, price_hi, work, spent],
            color=["#7f7f7f", "#7f7f7f", "#d62728", "#ff9896"])
    ax.axhline(price_lo, ls=":", lw=1, color="k")
    ax.axhline(price_hi, ls=":", lw=1, color="k")
    ax.set_xscale("log")
    ax.set_xlabel("displacement (log)")
    ax.set_title(f"THE e305 LANDAUER LEDGER — {rider}\n"
                 f"price = e290 bracket x ||F3W|| {f3_write_norm:.3f}; "
                 f"twin passive drift "
                 f"{landauer['twin_passive_reference']['twin_drift_from_organism']:.3f}")
    for i, v in enumerate([price_lo, price_hi, work, spent]):
        ax.text(v, i, f" {v:.5f}", va="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RD / "e294_anticontroller.png", dpi=130)
    plt.close(fig)
    log(f"[fig] wrote {RD / 'e294_anticontroller.png'}")

    # ================= P7: THE REPORT (executor-written) ================
    wh = landauer["waste_heat_bystanders"]
    rep = []
    rep.append(f"# E294 — THE ANTI-CONTROLLER CELL (targeted forgetting / "
               f"machine unlearning — the kill law inverted) + the e305 "
               f"Landauer ledger rider\n")
    rep.append(f"**Verdict: {verdict}** — {clause}\n")
    rep.append(f"The question (frozen): *"
               f"{REGISTERED['question_verbatim']}*\n")
    rep.append("\n## THE FAMILY DISCLOSURE\n\n"
               + REGISTERED["family_disclosure"] + "\n")
    rep.append("\n## The vehicle (the five fact-loads)\n\n"
               "e291's committed five-fact family organism loaded "
               "BIT-EXACT (flat md5 "
               f"{org_flat_md5}; the behavioral panel within "
               f"{max(panel_diffs.values()):.1e} of the committed "
               "baselines — G_FACTLOAD PASS three ways; the rooms rebuilt "
               "from the frozen seeds + certified + BIT-bound vs "
               "e291_rooms.pt).\n\n| fact | baseline g0 | base net | "
               "twin t400 | anti arm t400 | ABS hold | TWIN hold |\n"
               "|---|---|---|---|---|---|---|\n")
    for f_ in FACTS:
        if f_ == TARGET_FACT:
            rep.append(f"| **{f_} (TARGET)** | **{baselines[f_]:.6f}** | "
                       f"{base_panel[f_]:.2e} | **{T3_twn:.6f}** | "
                       f"**{R3:.6f}** | erased={erased} | "
                       f"x{R3 / T3_twn:.3f} vs twin |\n")
        else:
            rep.append(f"| {f_} | {baselines[f_]:.6f} | "
                       f"{base_panel[f_]:.2e} | {final_panel_t[f_]:.6f} | "
                       f"{final_panel_a[f_]:.6f} | "
                       f"{'HOLD' if sib_abs[f_] else 'FALL'} "
                       f"(x{final_panel_a[f_] / baselines[f_]:.2f}) | "
                       f"{'HOLD' if sib_twn[f_] else 'FALL'} "
                       f"(x{final_panel_a[f_] / final_panel_t[f_]:.2f}) |\n")
    rep.append("\n## The anti trace (the inverted gate)\n\n"
               "| # | t | gate read | dose | lr | binder | name CE | "
               "in-own-room | step | S_anti |\n|---|---|---|---|---|---|"
               "---|---|---|---|\n")
    for r in anti_led:
        rep.append(f"| {r['anti_index']} | {r['step']} | "
                   f"{r['read_at_gate']:.5f} | {r['dose_t']:.3f} | "
                   f"{r['lr_anti']:.2e} | {r['binder']} | "
                   f"{r['name_ce']:.4f} | {r['g_in_own_room_frac']:.4f} | "
                   f"{r['realized_step_norm']:.4f} | {r['S_anti']:.4f} |\n")
    rep.append("\n## THE e305 LANDAUER LEDGER (the erasure thermodynamic "
               "table)\n\n| quantity | value |\n|---|---|\n"
               f"| WORK (the anti's realized drift at "
               f"{'t_erase=' + str(ant['t_erase']) if erased else 't400 (not erased)'} | "
               f"{work:.6f} |\n| SPENT (S_anti, the step-norm sum) | "
               f"{spent:.4f} |\n| the passive price bracket (e290 x "
               f"||F3WRITE|| = x{f3_write_norm:.4f}) | "
               f"[{price_lo:.6f}, {price_hi:.6f}] |\n"
               f"| the organism-scaling alternative (x "
               f"{E291_ORG_WRITE_NORM:.2f}) | "
               f"[{E290_DRIFT_BRACKET[0] * E291_ORG_WRITE_NORM:.4f}, "
               f"{E290_DRIFT_BRACKET[1] * E291_ORG_WRITE_NORM:.4f}] |\n"
               f"| the twin's passive drift from the organism | "
               f"{landauer['twin_passive_reference']['twin_drift_from_organism']:.4f} "
               f"(S_corpus {twn['S_corpus']:.4f}) |\n"
               f"| FACT3 on the twin (the passive reference) | "
               f"{T3_twn:.6f} (x{T3_twn / baselines[TARGET_FACT]:.3f} "
               f"baseline) |\n| RIDER VERDICT | **{rider}** |\n")
    rep.append("\nThe waste heat (the bystanders' absorbed read-loss vs "
               "the twin at t400):\n\n| sibling | twin | anti arm | "
               "absorbed | ABS hold | TWIN hold |\n|---|---|---|---|---|"
               "---|\n")
    for f_ in sibs:
        w_ = wh[f_]
        rep.append(f"| {f_} | {w_['read_twin']:.6f} | "
                   f"{w_['read_arm_a']:.6f} | "
                   f"{w_['absorbed_vs_twin']:+.6f} | "
                   f"{'HOLD' if sib_abs[f_] else 'FALL'} | "
                   f"{'HOLD' if sib_twn[f_] else 'FALL'} |\n")
    def _fm(v, spec=".4f"):
        return format(v, spec) if v is not None else "n/a"

    rep.append("\n## The standing conditions\n\n"
               f"- G_BUDGET: {'PASS' if G_BUDGET['pass'] else 'BLOWN'} — "
               f"arm (a) S_corpus {ant['S_corpus']:.4f} + S_anti "
               f"{ant['S_anti']:.4f} = {S_total_a:.4f}/{budget_norm:.4f} "
               f"({S_total_a / budget_norm:.1%}); the twin S_corpus "
               f"{twn['S_corpus']:.4f}/{budget_c:.4f}\n"
               f"- G_ORTH: {'PASS' if G_ORTH['pass'] else 'FAILED'} — max "
               f"rel-err {max(ant['orth_max'], twn['orth_max']):.2e} "
               f"(bar {ORTH_BAR})\n"
               f"- G_ANTIBIND: PASS — {ant['n_anti']} anti events "
               "(machine-counted), the dose law asserted at every event, "
               "zero isolation violations\n"
               f"- The stream: arm (a) live={stream_a['live']} (CE "
               f"{_fm(stream_a['early_median'])} -> "
               f"{_fm(stream_a['late_median'])}); twin live="
               f"{stream_t['live']} ({_fm(stream_t['early_median'])} -> "
               f"{_fm(stream_t['late_median'])})\n"
               f"- The ascent gate (at birth): one full-dose stroke "
               f"lowered FACT3 {G_ASCENT['target_pre']:.6f} -> "
               f"{G_ASCENT['target_post']:.6f} "
               f"({G_ASCENT['target_drop_frac']:.1%}); the bystanders' "
               "one-stroke deltas "
               + " ".join(f"{k} {v:+.5f}" for k, v
                          in G_ASCENT["bystander_deltas"].items()) + "\n")
    rep.append("\n## Disclosures\n\n"
               "THE SELECTIVITY TEST IS WITHIN-FAMILY (the siblings share "
               "representation — the hardest selectivity test; e293: "
               "distinct facts not serially installable). The sibling-hold "
               "clause read BOTH ways (absolute + twin-relative); the "
               "WORK = the anti's realized drift; the passive price = "
               "e290's bracket x FACT3's own install write; n=1 per arm "
               "one lineage one session; NO cons (both post states "
               "checkpointed); no NOTES/THINKING/QUEUE/STATE edits.\n")
    (RD / "REPORT.md").write_text("".join(rep), encoding="utf-8")
    log(f"[report] wrote {RD / 'REPORT.md'}")

    # ---------------- the final envelope + provenance -------------------
    metrics["envelope_summary"] = _envelope_summary()
    metrics["provenance"] = {
        "birth_commit": metrics.get("birth_commit"),
        "final_commit": git_head(),
        "machine": common.gpu_status(),
        "written_at": common.now_iso(),
    }
    write_partial("COMPLETE (figure + report written)")
    log(f"E294 {'SMOKE ' if SMOKE else ''}COMPLETE — verdict {verdict}; "
        f"rider {rider}; outputs in {RD}")


if __name__ == "__main__":
    main()
