"""X39 — THE STEP-DOSE RUNG (the height remainder's cheapest probe; R78's
card, sharpened by e341/T315). This docstring carries the registered
background + design + bars + lab lean + P-x39a VERBATIM from the dispatch
letter + every frozen operationalization, committed at birth BEFORE any
compute. Adjudicate against exactly this; no bar shopping.

THE BACKGROUND (dispatch verbatim): "the recovery gradient after e341:
fresh 0.045 < controller-annealed(800 steps) 0.130 < varied-annealed(300)
0.244 < fixed-annealed(300) 0.320 << cons ~1.0. The HEIGHT AXis is the
named remainder — but the rungs confound read height with step count. THE
CHEAPEST CONTROL (on disk): e336's committed t400 checkpoint (read 0.815
at t400, controller-annealed 400 steps — HALF the t800 steps at a HIGH
read). If its retention matches e340's t800 (0.130) at matched read
height, steps beyond 400 buy nothing and the HEIGHT clause stands; if it
lands at e338's class (0.045) despite the 0.815 read, the recovery is
step-bought and the gradient's 'height' clause re-labels to steps."

THE DESIGN (dispatch verbatim):
  "1. THE SUBJECT: e336's committed t400 ERROR-GATED checkpoint
   (md5-bound; read 0.815).
  2. THE EVENT + TEST: commit(0.7) then e322's 100-step unbiased wash —
   e341's harness by import (the rebind set + the commit-placement
   convention it froze); retention + the s1 panel.
  3. Classes as references (runtime-read, md5-bound): e338's 0.045
   (fresh), e340's 0.130 (800 steps), e341's 0.244/0.320 (300 steps),
   the cons's ~1.0."

BARS (dispatch verbatim; frozen in this birth commit BEFORE compute):
  - HEIGHT-CLAUSE-STANDS: "retention ~0.13 (matching e340's 800-step
    value at matched read) — steps beyond 400 buy nothing; the recovery
    axis is read-height."
  - STEPS-BOUGHT: "retention ~0.045 (e338's class despite the 0.815
    read) — the recovery is step-dose; the gradient re-labels."
  - INTERMEDIATE: "between the classes — the dose-response curve has
    room between 400 and 800 steps; name the interpolation."

LAB LEAN (dispatch verbatim): "HEIGHT-CLAUSE-STANDS, weakly — the
controller's dose self-limits (e339's 2-cycle: the marginal steps may be
near-zero dose) so extra steps should buy little; the countervailing: the
2-cycle's alternating events still touch the read. State your own read."

P-x39a (THE EXECUTOR'S OWN READ, registered per the dispatch's "Register
P-x39a BEFORE compute ... State your own read", frozen HERE at birth;
predictions are scored): INTERMEDIATE, WEAKLY — diverging from the lab
lean, with HEIGHT-CLAUSE-STANDS the registered runner-up. GROUNDS: (1)
THE LINEAR-DOSE MIDPOINT — the t400 and t800 rungs are ONE organism on
ONE controller path (e339's REA arm resumed e336's committed t400 state
model+buffers+generators and ran the SAME driver for t401-t800), and the
dose split is exactly even: 16 of the 32 error-gated maintenance events
and 400 of the 800 corpus-annealing steps fall in the t400-t800 segment
(e336's t400 row n_maint_so_far 16; e339's desc: 'events 17-32'), so ANY
response linear in dose lands at (0.0448 + 0.1303)/2 = 0.0876 — inside
the INTERMEDIATE window, a hair above its lower edge; the dose-response
curve would then interpolate smoothly between the rungs. (2) THE
HEIGHT-PREDICTS-INVARIANCE ARGUMENT (why HEIGHT is the runner-up, not
the guess): the read moved only 0.815 -> 0.873 (+7%) across the segment,
so a pure read-height axis predicts t400 retention ~= t800's 0.130 — the
dispatch lean's own logic, and it earns HEIGHT the runner-up slot; but
e341 just falsified height-monotonicity ACROSS protocols (its 300-step
arms at LOWER reads 0.752/0.756 out-recover the controller's t800 0.130
by 2-2.5x), which discounts the height axis enough to drop it below
INTERMEDIATE on my card. (3) THE DOSE-IS-REAL REBUTTAL TO THE LEAN — the
lean's 'near-zero marginal dose' premise is falsified by the record: 16
more maintenance events FIRED in the segment and the read moved
0.815 -> ~0.900 (t600 peak) -> 0.873; the marginal segment is
maintenance-dominant but not dose-free. COUNTERVAILING (equal force):
(a) the recovery capacity may have been purchased ENTIRELY during the
initial climb (t0-t400, the read's 0.286 -> 0.815 rise), with the
maintenance segment only holding station — that lands t400 AT 0.130
(HEIGHT); (b) the s1 annihilation killed the t800 read to 0.0037 DESPITE
all 32 restore cycles — the restore cycles demonstrably do NOT protect
the first step, so their count may not purchase recovery either (HEIGHT,
by elimination); (c) retention@100 is a noisy single-seed statistic and
the wash's s25-s100 decay drift (e340's 0.236@s5 -> 0.114@s100) can move
a rung +-0.03 — enough to cross a thirds boundary. PREDICTED SHAPE: s1
ANNIHILATES (0.002-0.010 — the controller family's signature; dead at
s1, unlike e341's annealed arms which sat at 0.10-0.12); recovery fast by
s5 (0.15-0.25 absolute), then the slow decay to s100; retention@s100 in
[0.07, 0.11] — the INTERMEDIATE window straddling the 0.0876 midpoint;
gm12 tracks the g0 read within ~0.02 (the controller family's
geometry-tight pairing; e341's varied arm was the outlier at +0.14).
FALSIFIER: retention <= 0.0733 (STEPS-BOUGHT — the second 400 steps
carry the purchase; the gradient re-labels to steps); retention >= 0.1018
(HEIGHT-CLAUSE-STANDS — the purchase completed by t400); retention >
0.1588 (the residual SUPER-800 cell — t400 out-recovers t800, the
controller's own dose curve non-monotonic; the table verbatim, the
e311 convention). SCORED: TRUE iff the verdict == INTERMEDIATE.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE SUBJECT := e336's committed t400 ERROR-GATED end state —
  runs/e336/ERROR-GATED_post.pt (raw-md5 6c42e8131994122fcdb9ef56c0314e0a,
  size 10977183 — bound at birth, re-verified on disk), loaded BIT-EXACT,
  never re-formed. Committed reads (e336's t400 milestone row, md5-bound
  via runs/e336/metrics.json f90b439a0a4eff55e7f115901af469d6 and
  re-asserted at run time on CPU, tol 2e-6 read / 5e-3 CE — the family's
  cross-session read-determinism law): p(T) at the host g0 battery
  0.8147322535514832 (THE READ — near-cons-class height, HALF the t800
  steps), gm12 p(T) 0.7971041202545166, CE_R 1.6229653358459473; the
  row's disp_norm 1.1536193280751479 and n_maint_so_far 16 are co-bound
  (the dose ledger). The organism: the e311 TAVIREN install lineage +
  e336's ERROR-GATED controller run to t400 (16 name-only maintenance
  events, corpus steps budgeted + orthogonalized into the K10K room).
  THE RUNG'S CLEANLINESS (the design's own point, verified at run time):
  e339's REA arm RESUMED this exact state (model + buffers + generators)
  and ran the same driver for t401-t800 — t400 and t800 are ONE organism
  on ONE path at two dose points (16 vs 32 events; 400 vs 800 steps);
  the read heights 0.815 vs 0.873 sit 7% apart. net0 class (the standing
  rule): BASE-formed lineage (e311 install) + controller-annealed to t400
  — the STEP-DOSE RUNG (half the t800 dose at high read).

* THE EVENT + TEST := e341's harness BY IMPORT — which IS e338's harness
  (e341 imported lab/e338_commit_consolidator.py; md5
  5ddc8754cb358c09278ddd926059d050, bound at birth and re-verified), with
  THE REBIND SET + THE COMMIT-PLACEMENT CONVENTION e341 froze: the
  subject arrives PRE-ANNEALED (the controller did the annealing), so
  the placement is SUBJECT -> COMMIT(0.7) -> WASH — e341's own order
  (anneal -> commit -> wash) with the annealing leg already on disk.
  phase_P2's firing executes VERBATIM by import (the 8/8 event-quote
  gate, the era-rig md5s, G_BITROOT):
      net0 = G1.CommittedGPT(GB.G1B_CFG); net0.load_state_dict(theta0);
      net0.commit(0.7)                                # R = the W1 dial
  then e322's committed 100-step unbiased wash through wash_arm (seed
  10902, batch 32 = 16 install-anchor + 16 random windows, all-token
  mean CE, AdamW const lr 1e-3 betas (0.9, 0.95) wd 0.1 clip 1.0,
  panels {0, 1, 5, 10, 25, 50, 100}), the committed panels read through
  the ARMED CPU eval twin (the wall settles before every read).
  ONE ARM (the dispatch's design): T400-COMMIT. The harness's returned
  twin0 is run NOWHERE (the twin classes ride free, md5-bound: e338's
  twin 0.0036 SAND-STRICT; e340's twin 0.0018 SAND-STRICT).

* THE BITSTREAM GATE (this cell's G_DRAWS/G_INPUTS, frozen at birth):
  the single arm's wash draws + input-batch md5s must be BIT-IDENTICAL
  to e340's committed-arm wash bitstream (runtime-read from
  runs/e340/wash_committed_resume.pt, md5
  409e82c133c5e1d047e682a9ae478999 — the same seed-10902 bitstream
  e340's t800 subject and e341's five legs consumed): the t400 subject
  faces the EXACT wash the t800 subject faced; the rung comparison is
  wash-matched, not merely seed-matched. (Smoke: prefix comparison over
  the smoke horizon, disclosed.)

* THE CLASSES (runtime-read from md5-bound committed records, never
  retyped; the committed literals re-gated at run time):
  - FRESH (0 steps, read 0.286): e338's COMMIT arm — retention
    0.04483826753900113, s1 0.004078532103449106
    (runs/e338/metrics.json, md5 0fe1cc6c72a51741621e334530703d70).
  - CONTROLLER-800 (800 steps, 32 events, read 0.873): e340's COMMIT arm
    — retention 0.1303297438183166, s1 0.00365515798330307
    (runs/e340/metrics.json, md5 5e6e552ece380306915cccbb6d070c5a).
  - CONS-PROTOCOL-300 (300 steps, reads 0.752/0.756): e341's VARIED arm
    retention 0.2436974665024519 (s1 0.1154203936457634) + FIXED arm
    retention 0.32002314609430166 (s1 0.09718827158212662)
    (runs/e341/metrics.json, md5 279bd235e43da13dc78998fe9da7b63d).
  - CONS (the ball): the installed-fact band via the harness's phase_P3
    == [0.9572, 1.0879] (+ the sand class, co-reported).
  - THE CONTINUATION RECORD: e339's metrics (md5
    00e5f9822b6559ede2ab0ee9987c5121) — the t400 -> t800 leg's own
    ledger (events 17-32; the rung cleanliness evidence), md5-bound as a
    parent.

* THE BAR OPERATIONALIZATION (the load-bearing birth decision, frozen;
  the dispatch's '~0.13' / '~0.045' are class-MATCH regions, and the two
  named classes are neighbors on the retention line, so a plain midpoint
  partition would leave INTERMEDIATE no territory): THE THIRDS RULE —
  each class owns the third of the inter-class gap nearest it, the
  middle third is INTERMEDIATE, and a fourth residual cell is named
  above the t800 match region. With the committed literals:
  G = 0.1303297438183166 - 0.04483826753900113 = 0.08549147627931547;
  TH = G/3 = 0.028497158759771823.
  - STEPS-BOUGHT:          R <= 0.07333542629877295  (= fresh + TH)
  - INTERMEDIATE: 0.07333542629877295 < R < 0.10183258505854478
                                            (= t800 - TH)
  - HEIGHT-CLAUSE-STANDS: 0.10183258505854478 <= R <= 0.15882690257808842
                                            (= t800 + TH)
  - SUPER-800 (the named residual): R > 0.15882690257808842 — t400
    out-recovering t800; the controller's own dose curve non-monotonic;
    the table verbatim (the e311 convention).
  The boundaries are frozen as literals AND re-derived at run time from
  the runtime-read class values (G_BARS gates equality to 1e-9); the
  s1 panel is a CO-READ (never adjudicates — the bars are retention-
  only, per the dispatch), with the family's 0.05 aliveness bar stamped.

* THE COMPOSITE (frozen at birth): TEXTURE (any hard-gate failure —
  HALT, nothing adjudicated) -> the retention partition above (SMOKE
  stamps everything, nothing adjudicated). The aliveness clause does
  NOT gate the bars (the dispatch's bars are retention-only); a
  sub-0.05 panel or a live s1 is stamped into the clause + the report,
  never re-routed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); the
ACTIVE COMPUTE ENVELOPE honored through the harness's e261 form: bursts
<= 175s (inside the <=180s dispatch window), per-step thermal polls at
the 78C margin, 40s cooldowns (the 30-60s window), the 84C never-past
line (inside the 85C dispatch line), wait_gpu_free/gpu_ok before every
burst (the launch gate inside _open_burst), NO concurrent GPU jobs; polls
to runs/_envelope_log.jsonl tagged x39:* (the retag wrapper); CPU park
if no CUDA (2.74M is CPU-viable; cap 1800s; loudly recorded).
TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/x39/{metrics.json (PROGRESSIVE), x39_step_dose.png,
REPORT.md, run.log (gitignored), the wash resume checkpoint + final
state (runs/x39/, never runs/checkpoints/ — this cell writes ONLY
lab/x39_* and runs/x39/*)}. No NOTES/THINKING/QUEUE/STATE edits
(dispatch; the heartbeat folds). Commit + push per phase. Run:
cd lab && python x39_step_dose.py (X39_SMOKE=1 shakedown).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# NOTE: E338_SMOKE is deliberately NOT set — the harness imports in FULL
# mode; this cell's own smoke horizon is applied by the rebind below.

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import torch                                          # noqa: E402

import common                                          # noqa: E402
from common import run_dir, save_json, set_seed        # noqa: E402

# ======================================================================
# THIS CELL'S CONTEXT (before the harness import, so the rebind is
# immediate)
# ======================================================================
SMOKE = os.environ.get("X39_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "x39_smoke" if SMOKE else "x39"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


import e338_commit_consolidator as E38                 # noqa: E402 — THE
                                                      # HARNESS (by import;
                                                      # e341's own harness
                                                      # WAS this module by
                                                      # import — the rebind
                                                      # set + commit
                                                      # placement e341 froze
                                                      # are this cell's
                                                      # convention; disclosed
                                                      # side effects: opens
                                                      # runs/e338/run.log +
                                                      # runs/e261/run.log in
                                                      # append, asserts CUDA;
                                                      # zero bytes written by
                                                      # this cell — the logs
                                                      # are rebound before
                                                      # any use)

# ---- THE REBIND SET (frozen at birth; e341's leg convention) -------------
thermal_log: list[dict] = []
device_events: list[dict] = []
E38.log = log
E38.RD = RD
E38.NAME = NAME
E38.T0 = T0                        # the harness's traj elapsed_s origin
E38.SUBJ_READ_T = 0.8147322535514832          # e336's committed t400 read
E38.NET0_CLASS = (
    "BASE-formed lineage (e311 TAVIREN install) + e336 ERROR-GATED "
    "controller-annealed to t400 (read 0.815, 16 maintenance events — HALF "
    "the t800 dose) — THE STEP-DOSE RUNG")
E38.E261.log = log
E38.E261.NAME = NAME
E38.E261.SMOKE = SMOKE
E38.E261.T0 = T0
E38.E261.thermal_log = thermal_log
E38.E261.device_events = device_events
E38.E261.DCT_WORKERS = 4                      # e307/e311's desk convention

# THE ENVELOPE-TAG WRAPPER (tags only; the burst arithmetic untouched)
_orig_open_burst = E38.E261._open_burst


def _open_burst_retag(tag: str):
    return _orig_open_burst(tag.replace("e338:", ""))


E38.E261._open_burst = _open_burst_retag

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
REPO = common.REPO

# ---- THE SUBJECT (e336's committed t400 ERROR-GATED end state) -----------
SUBJ_CK = REPO / "runs" / "e336" / "ERROR-GATED_post.pt"
SUBJ_MD5 = "6c42e8131994122fcdb9ef56c0314e0a"
SUBJ_SIZE = 10977183
SUBJ_READ_T = 0.8147322535514832             # e336's committed t400 p(T)@g0
SUBJ_GM12_T = 0.7971041202545166             # e336's committed t400 gm12
SUBJ_CE_R = 1.6229653358459473               # e336's committed t400 CE_R
SUBJ_DISP_NORM = 1.1536193280751479          # the t400 row's disp_norm
SUBJ_N_MAINT = 16                            # the t400 dose ledger
READ_TOL = 2e-6                              # CPU read-determinism law
CE_TOL = 5e-3

# ---- the committed records this cell binds (parents + classes) -----------
E336_METRICS = REPO / "runs" / "e336" / "metrics.json"
E336_METRICS_MD5 = "f90b439a0a4eff55e7f115901af469d6"
E339_METRICS = REPO / "runs" / "e339" / "metrics.json"
E339_METRICS_MD5 = "00e5f9822b6559ede2ab0ee9987c5121"
E338_METRICS = REPO / "runs" / "e338" / "metrics.json"
E338_METRICS_MD5 = "0fe1cc6c72a51741621e334530703d70"
E340_METRICS = REPO / "runs" / "e340" / "metrics.json"
E340_METRICS_MD5 = "5e6e552ece380306915cccbb6d070c5a"
E341_METRICS = REPO / "runs" / "e341" / "metrics.json"
E341_METRICS_MD5 = "279bd235e43da13dc78998fe9da7b63d"
E38_HARNESS = REPO / "lab" / "e338_commit_consolidator.py"
E38_HARNESS_MD5 = "5ddc8754cb358c09278ddd926059d050"   # e340/e341's bind —
                                             # the harness is unchanged
E341_LAB = REPO / "lab" / "e341_varied_annealing.py"
E341_LAB_MD5 = "18039ab39d1b04d1adef2a22b9b3a350"      # the convention
                                             # parent (the rebind set +
                                             # commit placement it froze)
E340_WASH_REF = REPO / "runs" / "e340" / "wash_committed_resume.pt"
E340_WASH_REF_MD5 = "409e82c133c5e1d047e682a9ae478999"  # the bitstream
                                             # reference (e340's committed
                                             # arm's draws + input md5s)

# ---- THE CLASS LITERALS (committed; re-gated at run time) ----------------
E338_RET = 0.04483826753900113               # fresh (0 steps, read 0.286)
E338_S1 = 0.004078532103449106
E338_T0 = 0.285851389169693
E340_RET = 0.1303297438183166                # controller t800
E340_S1 = 0.00365515798330307
E340_T0 = 0.8732402324676514
E341V_RET = 0.2436974665024519               # cons-protocol varied t300
E341V_S1 = 0.1154203936457634
E341V_T0 = 0.7519720792770386
E341F_RET = 0.32002314609430166              # cons-protocol fixed t300
E341F_S1 = 0.09718827158212662
E341F_T0 = 0.7557547688484192

# ---- THE BAR BOUNDARIES (the thirds rule; frozen + re-derived) -----------
BAR_GAP = E340_RET - E338_RET                # 0.08549147627931547
BAR_TH = BAR_GAP / 3.0                       # 0.028497158759771823
B_STEPS = E338_RET + BAR_TH                  # 0.07333542629877295
B_HEIGHT = E340_RET - BAR_TH                 # 0.10183258505854478
B_CAP = E340_RET + BAR_TH                    # 0.15882690257808842
BAR_TOL = 1e-9

# ---- the wash horizon (the harness's own; rebound only for smoke) --------
WASH_STEPS_FULL = 100
WASH_PANELS_FULL = (0, 1, 5, 10, 25, 50, 100)
if SMOKE:
    E38.WASH_STEPS = 5
    E38.WASH_PANELS = (0, 1, 5)
    E38.SMOKE = True

READ_BAR = 0.05                             # the family's frozen 0.05 bar

# the birth commit (frozen; pinned so reruns/finalizations from resume
# checkpoints keep the true provenance instead of recording the current
# head): bars + P-x39a committed + pushed BEFORE any compute
BIRTH_COMMIT_PINNED = "PIN-AT-BIRTH-STEP2"   # set by the pin commit
                                             # immediately after the birth
                                             # commit (e341's 1abf259
                                             # convention)

REGISTERED = {
    "background_verbatim": (
        "the recovery gradient after e341: fresh 0.045 < controller-"
        "annealed(800 steps) 0.130 < varied-annealed(300) 0.244 < "
        "fixed-annealed(300) 0.320 << cons ~1.0. The HEIGHT AXis is the "
        "named remainder — but the rungs confound read height with step "
        "count. THE CHEAPEST CONTROL (on disk): e336's committed t400 "
        "checkpoint (read 0.815 at t400, controller-annealed 400 steps — "
        "HALF the t800 steps at a HIGH read). If its retention matches "
        "e340's t800 (0.130) at matched read height, steps beyond 400 "
        "buy nothing and the HEIGHT clause stands; if it lands at e338's "
        "class (0.045) despite the 0.815 read, the recovery is step-"
        "bought and the gradient's 'height' clause re-labels to steps."),
    "design_verbatim": {
        "1_THE_SUBJECT": ("e336's committed t400 ERROR-GATED checkpoint "
                          "(md5-bound; read 0.815)."),
        "2_THE_EVENT_TEST": (
            "commit(0.7) then e322's 100-step unbiased wash — e341's "
            "harness by import (the rebind set + the commit-placement "
            "convention it froze); retention + the s1 panel."),
        "3_THE_CLASSES": (
            "Classes as references (runtime-read, md5-bound): e338's "
            "0.045 (fresh), e340's 0.130 (800 steps), e341's 0.244/0.320 "
            "(300 steps), the cons's ~1.0."),
    },
    "bars_verbatim": {
        "HEIGHT-CLAUSE-STANDS": (
            "retention ~0.13 (matching e340's 800-step value at matched "
            "read) — steps beyond 400 buy nothing; the recovery axis is "
            "read-height."),
        "STEPS-BOUGHT": (
            "retention ~0.045 (e338's class despite the 0.815 read) — "
            "the recovery is step-dose; the gradient re-labels."),
        "INTERMEDIATE": (
            "between the classes — the dose-response curve has room "
            "between 400 and 800 steps; name the interpolation."),
    },
    "lab_lean_verbatim": (
        "HEIGHT-CLAUSE-STANDS, weakly — the controller's dose "
        "self-limits (e339's 2-cycle: the marginal steps may be "
        "near-zero dose) so extra steps should buy little; the "
        "countervailing: the 2-cycle's alternating events still touch "
        "the read. State your own read."),
    "P-x39a": {
        "my_guess": ("INTERMEDIATE, WEAKLY (diverging from the lab lean; "
                     "HEIGHT-CLAUSE-STANDS the registered runner-up)"),
        "registered": (
            "GROUNDS: (1) THE LINEAR-DOSE MIDPOINT — t400 and t800 are "
            "ONE organism on ONE controller path (e339's REA resumed "
            "e336's t400 state model+buffers+generators; the same "
            "driver), and the dose split is exactly even (16 of 32 "
            "events, 400 of 800 steps in the segment), so any linear-in-"
            "dose response lands at (0.0448+0.1303)/2 = 0.0876 — inside "
            "the INTERMEDIATE window. (2) HEIGHT-PREDICTS-INVARIANCE "
            "(the runner-up's own argument — the read moved only +7% "
            "across the segment) but e341 just falsified height-"
            "monotonicity across protocols (300-step arms at LOWER "
            "reads out-recover the controller's t800 2-2.5x), "
            "discounting the height axis below INTERMEDIATE on my card. "
            "(3) THE DOSE-IS-REAL REBUTTAL — the lean's 'near-zero "
            "marginal dose' premise is falsified by the record: 16 more "
            "events fired in the segment and the read moved 0.815 -> "
            "~0.900(t600 peak) -> 0.873. COUNTERVAILING (equal force): "
            "(a) the recovery capacity may be purchased entirely during "
            "the initial climb (t0-t400) with the segment only holding "
            "station -> HEIGHT; (b) the s1 annihilation killed t800 to "
            "0.0037 DESPITE all 32 restore cycles — the cycles do not "
            "protect the first step and may not purchase recovery "
            "either; (c) retention@100 is a noisy single-seed statistic "
            "(the s25-s100 decay drift can move a rung +-0.03, enough to "
            "cross a thirds boundary)."),
        "predicted_shape": (
            "s1 ANNIHILATES (0.002-0.010, the controller family's "
            "signature — dead at s1, unlike e341's annealed arms at "
            "0.10-0.12); recovery fast by s5 (0.15-0.25 absolute) then "
            "the slow decay; retention@s100 in [0.07, 0.11] straddling "
            "the 0.0876 midpoint; gm12 tracks g0 within ~0.02 (the "
            "controller's geometry-tight pairing)."),
        "falsifier": (
            "retention <= 0.0733 (STEPS-BOUGHT); retention >= 0.1018 "
            "(HEIGHT-CLAUSE-STANDS); retention > 0.1588 (the residual "
            "SUPER-800 cell — t400 out-recovering t800)."),
        "scored": "TRUE iff the verdict == INTERMEDIATE",
    },
    "registration": ("background + design + bars + lab lean + P-x39a "
                     "VERBATIM from the dispatch letter; every convention "
                     "picked + frozen HERE at birth BEFORE compute; this "
                     "script committed at birth; adjudicate against "
                     "exactly this; no bar shopping."),
}

deviations: list[str] = [
    "THE BAR OPERATIONALIZATION (the load-bearing birth decision): the "
    "dispatch's '~0.13'/'~0.045' are class-match regions and the two "
    "named classes are NEIGHBORS on the retention line, so a midpoint "
    "partition would leave INTERMEDIATE no territory — THE THIRDS RULE "
    "frozen instead: each class owns the third of the inter-class gap "
    "nearest it, the middle third is INTERMEDIATE, and the residual "
    "above the t800 match region is the named SUPER-800 cell (t400 "
    "out-recovering t800; the controller's own dose curve non-monotonic; "
    "the table verbatim, the e311 convention). Boundaries frozen as "
    "literals + re-derived at run time from the runtime-read classes "
    "(G_BARS, 1e-9).",
    "THE ALIVENESS CLAUSE DOES NOT GATE (frozen at birth): the dispatch's "
    "bars are RETENTION-ONLY; the s1 panel is a co-read stamped into the "
    "clause + report (the family's 0.05 bar), never re-routed — unlike "
    "e340's READ-HOLDS aliveness gate, which belonged to ITS bars.",
    "ONE ARM (the dispatch's design): T400-COMMIT alone is washed; the "
    "harness's returned twin0 is run NOWHERE (the twin classes ride "
    "free, md5-bound: e338's twin 0.0036 SAND-STRICT; e340's twin 0.0018 "
    "SAND-STRICT). The harness's two-arm phase_P5 therefore CANNOT "
    "execute verbatim — this cell's own P5 keeps its gate semantics "
    "(draws/inputs/t0/pin/guard/net0) with the bit-identity reference "
    "swapped from 'the other arm' to e340's committed-arm wash bitstream "
    "(the BITSTREAM GATE below).",
    "THE BITSTREAM GATE (G_DRAWS/G_INPUTS, frozen): the arm's wash draws "
    "+ input-batch md5s must be BIT-IDENTICAL to e340's committed arm's "
    "recorded bitstream (runtime-read from runs/e340/"
    "wash_committed_resume.pt, md5-bound) — the t400 subject faces the "
    "EXACT wash the t800 subject faced; the rung comparison is wash-"
    "matched. Smoke: prefix comparison over the smoke horizon.",
    "THE HARNESS IS e338 BY IMPORT under e341's convention (the dispatch's "
    "mandate 'e341's harness by import' — e341's harness WAS e338 by "
    "import): phase_P0 (the protocol rebuild), phase_P2 (the event + its "
    "8/8 source quotes + era-rig md5s + G_BITROOT), phase_P3 (the bands "
    "+ sand), wash_arm (e322's wash arithmetic) and arm_summary execute "
    "VERBATIM; the touched knobs are exactly THE REBIND SET (the cell "
    "context, the subject literals, the e261 envelope context, the smoke "
    "horizon, the envelope-tag wrapper — tags only).",
    "THE SUBJECT IS NOT THE HARNESS'S OWN (e340's precedent): phase_P1 is "
    "this cell's own phase_P1_x39 — the harness's phase_P1 loads e311's "
    "install, this cell's subject is e336's t400 state; the subject "
    "literals (SUBJ_READ_T et al.) are rebound into the harness's "
    "namespace so its G_T0 semantics read the t400 read.",
    "THE FILENAME WART (disclosed, inherited): the harness hardcodes its "
    "final wash-state names, so the wash final lands as runs/x39/"
    "e338_wash_T400-COMMIT_s100.pt (INSIDE this cell's run dir; the "
    "artifact meta's experiment field = x39, the rebind).",
    "IMPORT SIDE EFFECTS (disclosed, inherited): importing the harness "
    "opens runs/e338/run.log in append and importing e261 opens "
    "runs/e261/run.log + asserts CUDA — ZERO bytes written to either by "
    "this cell (the logs are rebound before any use); g1b_continuity's "
    "module-level G1.G1_CFG rebind IS the intended mechanism (the 2.74M "
    "family).",
    "THE WASH RESUME + FINAL STATES live under runs/x39/ (this cell "
    "writes ONLY lab/x39_* and runs/x39/* — no new files in "
    "runs/checkpoints/; e335/e340's convention).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
    "Smoke mode (X39_SMOKE=1): 5-step wash, panels {0,1,5}, the harness "
    "rebound to its own smoke horizon — NOTHING adjudicated (SMOKE stamp "
    "on every read; the classes runtime-read but the bars VACUOUS at the "
    "smoke horizon, disclosed).",
]

metrics: dict = {
    "experiment": "x39_step_dose",
    "phase": ("THE STEP-DOSE RUNG — e336's committed t400 ERROR-GATED state "
              "(TAVIREN 0.815, controller-annealed at HALF the t800 dose: "
              "16 vs 32 events, 400 vs 800 steps, one organism on one path) "
              "under commit(0.7) then e322's 100-step unbiased wash on the "
              "t800 cell's own bitstream — the retention read vs the "
              "fresh/controller-800/cons-300/cons classes: does the "
              "recovery axis track read-height or step-dose"),
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "envelope": {
        "device": "GPU (e322's envelope form via the harness) for the "
                  "wash; CPU park if no CUDA (loudly recorded)",
        "bursts": f"<= {E38.E261.BURST_MAX_S:.0f}s (inside the <=180s "
                  f"dispatch window), per-step thermal polls at a "
                  f"{E38.E261.TEMP_EARLY_END}C margin, "
                  f"{E38.E261.COOLDOWN_S:.0f}s cooldowns (the 30-60s "
                  f"window), the {E38.E261.TEMP_HARD}C never-past line "
                  f"(inside the 85C dispatch line)",
        "poll_log": "runs/_envelope_log.jsonl tagged x39:* (the retag "
                    "wrapper: the harness's hardcoded e338: tag prefix "
                    "stripped, tags only)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e336 (THE SUBJECT + THE DOSE LEDGER: its committed t400 ERROR-"
        "GATED end state — read 0.815, 16 maintenance events, HALF the "
        "t800 dose; its t400 milestone literals md5-bound + re-asserted)",
        "e339 (THE CONTINUATION RECORD: its REA arm resumed e336's t400 "
        "state model+buffers+generators and ran the same driver t401-t800 "
        "— the rung cleanliness evidence: t400 and t800 are one organism "
        "on one path at two dose points)",
        "e340 (THE DIRECT CLASS + THE DESIGN TEMPLATE: its committed t800 "
        "arm (retention 0.1303) is the height-matched 800-step class AND "
        "its wash bitstream is this cell's G_DRAWS reference; this cell "
        "is e340's design with the subject swapped to the half-dose "
        "rung)",
        "e341 (THE GRADIENT + THE CONVENTION: its 300-step arms (0.244/"
        "0.320) complete the class ladder its verdict named; its rebind "
        "set + commit-placement convention are this cell's harness "
        "convention, md5-bound)",
        "e338 (THE HARNESS: its commit(0.7) port, wash convention, bands "
        "construction and match-gate semantics execute verbatim by "
        "import; its COMMIT arm (retention 0.0448) is the fresh class)",
        "e311 (the TAVIREN organism underneath: the install lineage + "
        "the parasite battery read channel)",
        "g1_anchored_ball / g1b / g1c (the event's own code + the W1 "
        "dial + the ball's birth cell — inherited through the harness)",
        "e322 (THE SQUATTER'S DEED: the wash convention + the sand "
        "class)",
        "R78 / T315 (the review + thinking card that sharpened this "
        "discriminator; the recovery gradient's height-vs-steps fork)",
    ],
    "whats_new": [
        "THE FIRST INTERIOR POINT ON THE CONTROLLER'S OWN DOSE AXIS: "
        "every prior rung of the recovery gradient sat at an endpoint "
        "(fresh 0 steps / t800 800 steps / cons-protocol 300 steps on a "
        "different protocol); t400 is the first HALF-dose rung of the "
        "SAME organism under the SAME event + wash bitstream",
        "THE HEIGHT-VS-STEPS DECOMPOSITION AT MATCHED EVENT: the read "
        "heights 0.815 vs 0.873 sit 7% apart while the dose halves — the "
        "cheapest on-disk control the gradient admits",
        "THE WASH-MATCHED RUNG COMPARISON: the t400 subject consumes the "
        "bit-identical wash bitstream e340's t800 subject consumed "
        "(G_DRAWS against the md5-bound resume record)",
    ],
    "gates": {},
}

# the harness's phases write their partials into THIS cell's metrics
E38.metrics = metrics


# ------------------------------------------------------------------ helpers
def md5of(p: Path) -> str:
    import hashlib
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ======================================================================
# P1 — THE SUBJECT (e336's committed t400 state: load bit-exact + the
#      committed reads re-asserted + the record gates)
# ======================================================================
def phase_P1_x39(p0: dict) -> dict:
    log("P1 — THE SUBJECT (e336's committed t400 ERROR-GATED state, "
        "md5-bound; the STEP-DOSE RUNG)")

    # ---- G_E336RECORD: the subject's own record md5-bound + the t400 row
    rec_md5 = md5of(E336_METRICS)
    e336 = json.loads(E336_METRICS.read_text(encoding="utf-8"))
    traj = e336["arms"]["ERROR-GATED"]["phase"]["traj"]
    t400 = next(r for r in traj if r["step"] == 400)
    g_rec = {"path": "runs/e336/metrics.json", "md5": rec_md5,
             "bound_md5": E336_METRICS_MD5,
             "t400_row": {k: t400[k] for k in ("step", "g0_pz", "gm12_pz",
                                               "ce_r", "disp_norm",
                                               "n_maint_so_far")},
             "subject_provenance": ("e311 install -> e336 ERROR-GATED "
                                    "controller (t400); DIVERGES verdict "
                                    "ratio_400 2.8502x"),
             "claim": "the subject's committed record md5-bound; its t400 "
                      "milestone literals are the read gate's parents",
             "pass": bool(rec_md5 == E336_METRICS_MD5
                          and t400["g0_pz"] == SUBJ_READ_T
                          and t400["gm12_pz"] == SUBJ_GM12_T
                          and t400["ce_r"] == SUBJ_CE_R
                          and t400["disp_norm"] == SUBJ_DISP_NORM
                          and t400["n_maint_so_far"] == SUBJ_N_MAINT)}
    assert g_rec["pass"], f"G_E336RECORD FAILED: {g_rec}"

    # ---- G_HARNESS: the imported harness md5-bound ("by import" provenance)
    har_md5 = md5of(E38_HARNESS)
    g_harness = {"path": "lab/e338_commit_consolidator.py", "md5": har_md5,
                 "bound_md5": E38_HARNESS_MD5,
                 "convention_parent": {"path": "lab/e341_varied_annealing.py",
                                       "md5": md5of(E341_LAB),
                                       "bound_md5": E341_LAB_MD5},
                 "rebind_set": ["log", "RD", "metrics", "NAME", "T0",
                                "SUBJ_READ_T", "NET0_CLASS",
                                "SMOKE (smoke-only horizon)",
                                "WASH_STEPS/WASH_PANELS (smoke only)",
                                "E261.{log,NAME,SMOKE,T0,thermal_log,"
                                "device_events}",
                                "E261._open_burst (the tag wrapper)"],
                 "claim": "the harness this cell imports is the committed "
                          "birth-bind (e340/e341's own unchanged bind) and "
                          "e341's convention parent is md5-bound; the "
                          "rebind set is exactly the frozen set",
                 "pass": bool(har_md5 == E38_HARNESS_MD5
                              and md5of(E341_LAB) == E341_LAB_MD5)}
    assert g_harness["pass"], f"G_HARNESS FAILED: {g_harness}"

    # ---- G_SUBJECT: the artifact md5 + the reads re-asserted on CPU
    art_md5, art_size = md5of(SUBJ_CK), SUBJ_CK.stat().st_size
    g_subject = {"path": "runs/e336/ERROR-GATED_post.pt", "md5": art_md5,
                 "bound_md5": SUBJ_MD5, "size": art_size,
                 "bound_size": SUBJ_SIZE,
                 "pass": bool(art_md5 == SUBJ_MD5
                              and art_size == SUBJ_SIZE)}
    assert g_subject["pass"], f"G_SUBJECT FAILED: {g_subject}"

    art = torch.load(SUBJ_CK, map_location="cpu", weights_only=False)
    theta0 = {k: v.detach().clone() for k, v in art["model"].items()}
    g_subject["meta"] = {k: (v if isinstance(v, (int, float, str))
                             else str(v))
                         for k, v in art.get("meta", {}).items()}
    del art

    net = E38.G1.evl_load(theta0)            # the plain (uncommitted) load
    net.eval()
    rd_g0 = E38.G1.battery_cell(net, p0["g0_ids"], p0["tid"])["mean_pz"]
    rd_gm12 = E38.G1.battery_cell(net, p0["gm12_ids"], p0["tid"])["mean_pz"]
    rd_ce = E38.G1.ce_fixed_cpu(net, *p0["r_eval_xy"])
    del net
    g_subject.update({
        "read_g0_t": {"mine": rd_g0, "committed": SUBJ_READ_T,
                      "abs_diff": abs(rd_g0 - SUBJ_READ_T)},
        "read_gm12_t": {"mine": rd_gm12, "committed": SUBJ_GM12_T,
                        "abs_diff": abs(rd_gm12 - SUBJ_GM12_T)},
        "ce_r": {"mine": rd_ce, "committed": SUBJ_CE_R,
                 "abs_diff": abs(rd_ce - SUBJ_CE_R)},
        "net0_class": E38.NET0_CLASS,
        "height_note": ("near-cons-class height (0.815 vs t800's 0.873 — "
                        "7% apart; the fresh class was 0.286) at HALF the "
                        "t800 dose — the rung's own point"),
    })
    g_subject["pass"] = bool(
        g_subject["pass"]
        and abs(rd_g0 - SUBJ_READ_T) <= READ_TOL
        and abs(rd_gm12 - SUBJ_GM12_T) <= READ_TOL
        and abs(rd_ce - SUBJ_CE_R) <= CE_TOL)
    assert g_subject["pass"], f"G_SUBJECT reads FAILED: {g_subject}"

    metrics["gates"].update({"G_E336RECORD": g_rec, "G_HARNESS": g_harness,
                             "G_SUBJECT": g_subject})
    log(f"G_E336RECORD + G_HARNESS + G_SUBJECT PASS: t400 md5 {art_md5}; "
        f"p(T)@g0 {rd_g0:.12f} == {SUBJ_READ_T:.12f} (|d| "
        f"{abs(rd_g0 - SUBJ_READ_T):.1e}); gm12 {rd_gm12:.6f}; CE_R "
        f"{rd_ce:.4f}; dose ledger 16 events / disp 1.1536; net0 "
        f"{E38.NET0_CLASS}")
    write_partial("P1 subject loaded bit-exact (the step-dose rung)")
    return {"theta0": theta0}


# ======================================================================
# P3b — THE CLASSES (runtime-read from md5-bound committed records) +
#       the bar boundaries re-derived (G_BARS)
# ======================================================================
def phase_classes() -> dict:
    log("P3b — THE CLASSES (runtime-read from md5-bound records; the "
        "bar boundaries re-derived)")

    out: dict = {}

    # ---- FRESH: e338's committed arm --------------------------------------
    md5 = md5of(E338_METRICS)
    e338 = json.loads(E338_METRICS.read_text(encoding="utf-8"))
    comm = e338["arms"]["committed"]
    s1_38 = next(r for r in comm["traj"]
                 if r["step"] == 1)["read_g0_pT"]
    g = {"path": "runs/e338/metrics.json", "md5": md5,
         "bound_md5": E338_METRICS_MD5,
         "pass": bool(md5 == E338_METRICS_MD5
                      and comm["retention"] == E338_RET
                      and s1_38 == E338_S1 and comm["t0"] == E338_T0)}
    assert g["pass"], f"G_CLASSES(e338) FAILED: {g}"
    out["fresh"] = {"source": "e338 COMMIT arm (0 steps, read 0.286)",
                    "retention": comm["retention"], "s1": s1_38,
                    "t0": comm["t0"],
                    "traj": [(r["step"], r["read_g0_pT"])
                             for r in comm["traj"]]}
    log(f"  fresh class (e338 COMMIT): retention {comm['retention']:.6f}, "
        f"s1 {s1_38:.6f}")

    # ---- CONTROLLER-800: e340's committed arm -----------------------------
    md5 = md5of(E340_METRICS)
    e340 = json.loads(E340_METRICS.read_text(encoding="utf-8"))
    comm40 = e340["arms"]["committed"]
    s1_40 = next(r for r in comm40["traj"]
                 if r["step"] == 1)["read_g0_pT"]
    g = {"path": "runs/e340/metrics.json", "md5": md5,
         "bound_md5": E340_METRICS_MD5,
         "pass": bool(md5 == E340_METRICS_MD5
                      and comm40["retention"] == E340_RET
                      and s1_40 == E340_S1 and comm40["t0"] == E340_T0)}
    assert g["pass"], f"G_CLASSES(e340) FAILED: {g}"
    out["controller800"] = {"source": "e340 COMMIT arm (t800, 32 events, "
                            "read 0.873)",
                            "retention": comm40["retention"], "s1": s1_40,
                            "t0": comm40["t0"],
                            "traj": [(r["step"], r["read_g0_pT"])
                                     for r in comm40["traj"]]}
    log(f"  controller-800 class (e340 COMMIT): retention "
        f"{comm40['retention']:.6f}, s1 {s1_40:.6f}")

    # ---- CONS-PROTOCOL-300: e341's two committed arms ----------------------
    md5 = md5of(E341_METRICS)
    e341 = json.loads(E341_METRICS.read_text(encoding="utf-8"))
    arms341 = e341["adjudication"]["arms"]
    var, fix = arms341["varied_committed"], arms341["fixed_committed"]
    g = {"path": "runs/e341/metrics.json", "md5": md5,
         "bound_md5": E341_METRICS_MD5,
         "pass": bool(md5 == E341_METRICS_MD5
                      and var["retention"] == E341V_RET
                      and var["s1"] == E341V_S1
                      and var["t0"] == E341V_T0
                      and fix["retention"] == E341F_RET
                      and fix["s1"] == E341F_S1
                      and fix["t0"] == E341F_T0)}
    assert g["pass"], f"G_CLASSES(e341) FAILED: {g}"
    out["cons300"] = {
        "source": "e341 committed arms (cons-protocol 300 steps: VARIED "
                  "jitter pool / FIXED j=0 pool)",
        "varied": {"retention": var["retention"], "s1": var["s1"],
                   "t0": var["t0"],
                   "traj": [(r["step"], r["read_g0_pT"])
                            for r in var["traj"]]},
        "fixed": {"retention": fix["retention"], "s1": fix["s1"],
                  "t0": fix["t0"],
                  "traj": [(r["step"], r["read_g0_pT"])
                           for r in fix["traj"]]}}
    log(f"  cons-300 classes (e341): varied retention "
        f"{var['retention']:.6f} (s1 {var['s1']:.4f}); fixed "
        f"{fix['retention']:.6f} (s1 {fix['s1']:.4f})")

    # ---- THE CONTINUATION RECORD: e339 (the rung-cleanliness evidence) ----
    md5 = md5of(E339_METRICS)
    e339 = json.loads(E339_METRICS.read_text(encoding="utf-8"))
    rea_desc = e339["arms"]["ERROR-GATED"]["desc"]
    g = {"path": "runs/e339/metrics.json", "md5": md5,
         "bound_md5": E339_METRICS_MD5,
         "rea_desc_head": rea_desc[:120],
         "pass": bool(md5 == E339_METRICS_MD5
                      and "e336" in rea_desc
                      and "t400" in rea_desc
                      and "17-32" in rea_desc)}
    assert g["pass"], f"G_CLASSES(e339 lineage) FAILED: {g}"
    out["continuation"] = {
        "source": "e339 ERROR-GATED (REA) — the t401-t800 leg",
        "desc_head": rea_desc[:200],
        "claim": ("t400 and t800 are ONE organism on ONE path: e339's "
                  "REA resumed e336's committed t400 state (model + "
                  "buffers + generators) and ran the same driver for "
                  "events 17-32")}
    log("  continuation record (e339 REA): resumed e336's t400 state; "
        "events 17-32 in the segment — the rung is clean")

    # ---- G_BARS: the thirds rule re-derived from the runtime-read classes --
    gap = comm40["retention"] - comm["retention"]
    th = gap / 3.0
    g_bars = {
        "rule": ("THE THIRDS RULE (frozen at birth): each named class "
                 "owns the third of the inter-class gap nearest it; the "
                 "middle third is INTERMEDIATE; above the t800 match "
                 "region is the named residual SUPER-800 cell"),
        "gap": gap, "third": th,
        "frozen": {"gap": BAR_GAP, "third": BAR_TH, "steps_cap": B_STEPS,
                   "height_lo": B_HEIGHT, "height_hi": B_CAP},
        "derived": {"gap": gap, "third": th,
                    "steps_cap": comm["retention"] + th,
                    "height_lo": comm40["retention"] - th,
                    "height_hi": comm40["retention"] + th},
        "claim": "the runtime-derived boundaries == the frozen literals "
                 "(1e-9); the partition is total on the retention line",
        "pass": bool(abs(gap - BAR_GAP) <= BAR_TOL
                     and abs(th - BAR_TH) <= BAR_TOL
                     and abs(comm["retention"] + th - B_STEPS) <= BAR_TOL
                     and abs(comm40["retention"] - th - B_HEIGHT)
                     <= BAR_TOL
                     and abs(comm40["retention"] + th - B_CAP) <= BAR_TOL)}
    assert g_bars["pass"], f"G_BARS FAILED: {g_bars}"
    metrics["gates"]["G_BARS"] = g_bars
    log(f"G_BARS PASS: STEPS-BOUGHT <= {B_STEPS:.10f} < INTERMEDIATE < "
        f"{B_HEIGHT:.10f} <= HEIGHT-CLAUSE-STANDS <= {B_CAP:.10f} < "
        f"SUPER-800 (residual)")

    metrics["gates"]["G_CLASSES"] = {
        "claim": "every class reference runtime-read from its md5-bound "
                 "committed record (e338/e340/e341 + e339's continuation "
                 "lineage + the harness-bound g1b/g1bR/g1c/e322/e311)",
        "pass": True}
    write_partial("P3b the classes runtime-read; the bars re-derived")
    return out


# ======================================================================
# P5 — THE MATCH GATES (this cell's own: the bitstream gate + the harness
#      semantics; disclosed deviation — one arm)
# ======================================================================
def phase_P5_x39(wash: dict) -> dict:
    # ---- G_DRAWS + G_INPUTS: bit-identical to e340's committed bitstream --
    ref_md5 = md5of(E340_WASH_REF)
    ref = torch.load(E340_WASH_REF, map_location="cpu", weights_only=False)
    n = min(len(wash["draws"]), E38.WASH_STEPS)
    same_draws = (wash["draws"][:n] == ref["draws"][:n])
    same_x = (wash["xhash"][:n] == ref["xhash"][:n])
    g_draws = {
        "reference": {"path": "runs/e340/wash_committed_resume.pt",
                      "md5": ref_md5, "bound_md5": E340_WASH_REF_MD5,
                      "n_draws_recorded": len(ref["draws"])},
        "n_steps": len(wash["draws"]), "compared": n,
        "bit_identical": bool(same_draws),
        "smoke_prefix": bool(SMOKE),
        "claim": ("the arm's wash consumed the IDENTICAL aj/rj sequence "
                  "as e340's committed t800 arm (e322's seed 10902 "
                  "bitstream) — the rung comparison is wash-matched"
                  + ("; smoke prefix comparison over the smoke horizon"
                     if SMOKE else "")),
        "pass": bool(ref_md5 == E340_WASH_REF_MD5 and same_draws
                     and len(wash["draws"]) == E38.WASH_STEPS)}
    assert g_draws["pass"], f"G_DRAWS FAILED: {g_draws}"
    g_inputs = {
        "n_steps": len(wash["xhash"]), "compared": n,
        "bit_identical": bool(same_x),
        "claim": "the arm's input batches bit-identical to e340's "
                 "committed arm's (per-step x md5; g1's G-INPUTS "
                 "convention)",
        "pass": bool(same_x and len(wash["xhash"]) == E38.WASH_STEPS)}
    assert g_inputs["pass"], f"G_INPUTS FAILED: {g_inputs}"
    del ref

    # ---- G_T0: the arm's t0 == the committed t400 subject read ------------
    t0v = wash["traj"][0]["read_g0_pT"]
    g_t0 = {"arm_t0": t0v, "bound": SUBJ_READ_T,
            "abs_diff": abs(t0v - SUBJ_READ_T),
            "claim": "the arm starts at the committed t400 subject read "
                     "(the rung's own height)",
            "pass": bool(abs(t0v - SUBJ_READ_T) <= READ_TOL)}
    assert g_t0["pass"], f"G_T0 FAILED: {g_t0}"

    # ---- G_PIN: the committed arm's raw displacement <= R + fuzz at panels
    pins = [r["disp_raw"] for r in wash["traj"]]
    g_pin = {"R": E38.EVENT_R, "bound": E38.EVENT_R + E38.PIN_FUZZ,
             "max_raw_disp_at_panels": max(pins),
             "per_panel": {r["step"]: r["disp_raw"]
                           for r in wash["traj"]},
             "claim": "the committed arm never leaves R + the licensed "
                      "fuzz between forwards (g1b's G-PIN verbatim)",
             "pass": bool(max(pins) <= E38.EVENT_R + E38.PIN_FUZZ)}
    assert g_pin["pass"], f"G_PIN FAILED: {g_pin}"

    # ---- G_GUARD: the name-free guard never fired -------------------------
    g_guard = {"zeph_hits": wash["zeph"],
               "claim": "the name-free guard never fired on the wash",
               "pass": bool(wash["zeph"] == 0)}
    assert g_guard["pass"], f"G_GUARD FAILED: {g_guard}"

    # ---- G_NET0: every state classed (the standing rule) -------------------
    g_net0 = {
        "subject": E38.NET0_CLASS,
        "arm": E38.NET0_CLASS + " + COMMIT(0.7) armed + e322's 100-step "
                                  "unbiased wash (e340's own bitstream)",
        "reference_classes": {
            "e338 COMMIT": "install-fresh (0 steps, read 0.286) + "
                           "COMMIT(0.7) — the fresh class",
            "e340 COMMIT": "controller t800 (32 events, read 0.873) + "
                           "COMMIT(0.7) — the 800-step class",
            "e341 VARIED/FIXED COMMIT": "cons-protocol t300 (read "
                                        "0.752/0.756) + COMMIT(0.7) — "
                                        "the 300-step classes",
            "the W1 legs": "cons-shaped + COMMIT(0.7) — the survival "
                           "class (~1.0)"},
        "claim": "every state classed (x32's standing rule)", "pass": True}

    metrics["gates"].update({"G_DRAWS": g_draws, "G_INPUTS": g_inputs,
                             "G_T0": g_t0, "G_PIN": g_pin,
                             "G_GUARD": g_guard, "G_NET0": g_net0})
    for k, g in (("G_DRAWS", g_draws), ("G_INPUTS", g_inputs),
                 ("G_T0", g_t0), ("G_PIN", g_pin), ("G_GUARD", g_guard),
                 ("G_NET0", g_net0)):
        log(f"{k}: {'PASS' if g['pass'] else 'FAIL'}"
            + (f" (max raw |d| {g_pin['max_raw_disp_at_panels']:.4f} <= "
               f"{E38.EVENT_R + E38.PIN_FUZZ:.2f})" if k == "G_PIN"
               else ""))
    return {"g_draws": g_draws, "g_inputs": g_inputs, "g_t0": g_t0,
            "g_pin": g_pin}


# ======================================================================
# ADJUDICATION (the frozen composite: TEXTURE -> the retention partition)
# ======================================================================
def adjudicate_x39(arm_s: dict, classes: dict) -> dict:
    if SMOKE:
        return {"verdict": "SMOKE",
                "clause": ("smoke horizon — nothing adjudicated (the "
                           "bars need the 100-step horizon)"),
                "P-x39a": {"guess": REGISTERED["P-x39a"]["my_guess"],
                           "lab_lean": REGISTERED["lab_lean_verbatim"],
                           "hit": None,
                           "scored": REGISTERED["P-x39a"]["scored"]},
                "arm": arm_s}

    R = arm_s["retention"]
    reads = {r["step"]: r["read_g0_pT"] for r in arm_s["traj"]}
    s1 = reads.get(1)
    s1_alive = bool(s1 is not None and s1 >= READ_BAR)
    deep_note = ("; DEEPER than the fresh class itself — the rung under-"
                 "performs even the 0-step class" if R < E338_RET else "")

    if R <= B_STEPS:
        verdict = "STEPS-BOUGHT"
        clause = (f"retention {R:.4f} <= {B_STEPS:.4f} (the fresh class's "
                  f"third: e338's {E338_RET:.4f} DESPITE the 0.815 "
                  f"read) — THE RECOVERY IS STEP-DOSE: the purchase sits "
                  f"in the t400->t800 segment (the gradient's 'height' "
                  f"clause re-labels to steps; the dose-response curve "
                  f"rises through the segment){deep_note}")
    elif R < B_HEIGHT:
        verdict = "INTERMEDIATE"
        frac = (R - E338_RET) / (E340_RET - E338_RET)
        clause = (f"retention {R:.4f} strictly between the classes "
                  f"(fresh {E338_RET:.4f} < R < t800 {E340_RET:.4f}) — "
                  f"THE DOSE-RESPONSE CURVE HAS ROOM BETWEEN 400 AND 800 "
                  f"STEPS: the interpolation lands at {frac:.2f} of the "
                  f"fresh->t800 span ({16 + 32 * frac:.1f}-of-32-events "
                  f"equivalent); steps beyond 400 bought "
                  f"{(R - E338_RET) / (E340_RET - E338_RET) * 100:.0f}% "
                  f"of the segment's recovery purchase")
    elif R <= B_CAP:
        verdict = "HEIGHT-CLAUSE-STANDS"
        clause = (f"retention {R:.4f} matches e340's t800 value "
                  f"{E340_RET:.4f} at matched read (0.815 vs 0.873 — the "
                  f"t800 match region [{B_HEIGHT:.4f}, {B_CAP:.4f}]) — "
                  f"STEPS BEYOND 400 BUY NOTHING: the recovery axis is "
                  f"read-HEIGHT (the purchase completed by t400; the "
                  f"controller's t400-t800 segment is station-keeping)")
    else:
        verdict = "SUPER-800 (residual)"
        clause = (f"retention {R:.4f} > the t800 match region's cap "
                  f"{B_CAP:.4f} — t400 OUT-RECOVERS t800 on the same "
                  f"organism, same event, same wash bitstream: the "
                  f"controller's own dose curve is NON-MONOTONIC (the "
                  f"extra 400 steps bought NEGATIVE recovery). The "
                  f"table verbatim (the e311 convention): the height "
                  f"clause is neither confirmed nor re-labelled by a "
                  f"non-monotone dose axis; the named discriminator is "
                  f"the maintenance-segment's footprint (what the "
                  f"t400-t800 events changed that HURT recovery)")

    # the s1 co-read (never adjudicates; the bars are retention-only)
    if s1 is not None:
        clause += (f" [s1 co-read: {s1:.6f} — "
                   + ("ALIVE at s1 (>= 0.05: the FIRST controller-lineage "
                      "state to survive the first step — a new signature, "
                      "named in the report)" if s1_alive else
                      "dead at s1 (the controller family's annihilation "
                      "signature, shared with t800's 0.0037 and the fresh "
                      "class's 0.0041)") + "]")
    gm_end = arm_s["traj"][-1]["gm12_pT"] if arm_s.get("traj") else None
    if gm_end is not None:
        clause += f" [gm12@s100 {gm_end:.4f} — the geometry co-read]"

    p_hit = bool(verdict == "INTERMEDIATE")
    return {"verdict": verdict, "clause": clause,
            "reads": {"retention": R, "s1": s1, "s1_alive": s1_alive,
                      "min_panel_read": arm_s["min_panel_read"],
                      "all_panels_ge_005": arm_s["all_panels_ge_005"],
                      "first_below_005": arm_s["first_below_005"],
                      "bars": {"steps_cap": B_STEPS,
                               "height_lo": B_HEIGHT, "height_hi": B_CAP,
                               "rule": "the thirds rule (frozen at "
                                       "birth; G_BARS re-derived)"}},
            "P-x39a": {"guess": REGISTERED["P-x39a"]["my_guess"],
                       "lab_lean": REGISTERED["lab_lean_verbatim"],
                       "hit": p_hit,
                       "scored": REGISTERED["P-x39a"]["scored"]},
            "arm": arm_s}


# ======================================================================
# OUTPUTS
# ======================================================================
def make_png(adj: dict, classes: dict, band: dict) -> None:
    arm = adj.get("arm", {})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- panel 1: the wash read curve + the four class ghosts -------------
    if arm.get("traj"):
        st = [r["step"] for r in arm["traj"]]
        rd = [max(r["read_g0_pT"], 1e-7) for r in arm["traj"]]
        ax1.semilogy(st, rd, "o-", color="tab:blue", lw=2.6, ms=6,
                     label=f"T400-COMMIT (this cell; t0 "
                           f"{arm['t0']:.4f} -> s100 "
                           f"{arm['s_end']:.4f}, retention "
                           f"{arm['retention']:.4f})")
    for tr, col, lab in ((classes["controller800"]["traj"], "tab:red",
                          f"t800 + commit (e340; retention "
                          f"{E340_RET:.4f})"),
                         (classes["fresh"]["traj"], "tab:gray",
                          f"fresh + commit (e338; {E338_RET:.4f})"),
                         (classes["cons300"]["varied"]["traj"],
                          "tab:green", f"varied-300 + commit (e341; "
                                       f"{E341V_RET:.4f})"),
                         (classes["cons300"]["fixed"]["traj"],
                          "tab:orange", f"fixed-300 + commit (e341; "
                                        f"{E341F_RET:.4f})")):
        ax1.semilogy([s for s, _ in tr], [max(v, 1e-7) for _, v in tr],
                     "^:", color=col, lw=1.6, alpha=0.85, label=lab)
    ax1.axhline(READ_BAR, color="black", ls="--", lw=1.0,
                label="the 0.05 bar")
    ax1.axhline(SUBJ_READ_T, color="tab:blue", ls=":", lw=1.2,
                label=f"t0 = the t400 read {SUBJ_READ_T:.4f} (t800 was "
                      f"{E340_T0:.4f})")
    ax1.set_xlabel("wash step (e322's unbiased corpus wash, seed 10902 — "
                   "e340's own bitstream)")
    ax1.set_ylabel("THE READ (battery g0 p(T), log)")
    ax1.set_title(f"X39 THE STEP-DOSE RUNG: {adj.get('verdict')}")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(alpha=0.25)

    # ---- panel 2: the retention ruler + the decision regions ---------------
    ax2.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax2.axhspan(band["lo"], band["hi"], color="tab:green", alpha=0.15,
                label=f"cons band [{band['lo']:.3f}, {band['hi']:.3f}] "
                      f"(the ~1.0 class)")
    ax2.axhspan(E338_RET - 0.01, B_STEPS, color="tab:gray", alpha=0.22,
                label=f"STEPS-BOUGHT (<= {B_STEPS:.4f})")
    ax2.axhspan(B_STEPS, B_HEIGHT, color="tab:purple", alpha=0.18,
                label=f"INTERMEDIATE ({B_STEPS:.4f}, {B_HEIGHT:.4f})")
    ax2.axhspan(B_HEIGHT, B_CAP, color="tab:red", alpha=0.15,
                label=f"HEIGHT-CLAUSE-STANDS [{B_HEIGHT:.4f}, "
                      f"{B_CAP:.4f}]")
    if arm.get("traj"):
        rel = [(r["step"], r["read_g0_pT"] / arm["t0"])
               for r in arm["traj"]]
        ax2.plot([s for s, _ in rel], [v for _, v in rel], "o-",
                 color="tab:blue", lw=2.6, ms=6,
                 label=f"T400-COMMIT retention@s{arm['end_step']} "
                       f"{arm['retention']:.4f}")
    for tr, col, lab in ((classes["controller800"]["traj"], "tab:red",
                          "t800 (e340)"),
                         (classes["fresh"]["traj"], "tab:gray",
                          "fresh (e338)"),
                         (classes["cons300"]["varied"]["traj"],
                          "tab:green", "varied-300 (e341)"),
                         (classes["cons300"]["fixed"]["traj"],
                          "tab:orange", "fixed-300 (e341)")):
        t0e = tr[0][1]
        ax2.plot([s for s, _ in tr], [v / t0e for _, v in tr], "^:",
                 color=col, lw=1.5, alpha=0.85, label=lab)
    ax2.set_xlabel("wash step")
    ax2.set_ylabel("retention (read / own t0)")
    ax2.set_title("the retention ruler: the rung vs the four classes + "
                  "the frozen decision regions")
    ax2.legend(fontsize=6.5, loc="center right")
    ax2.grid(alpha=0.25)

    # ---- panel 3: THE STEP-DOSE PICTURE (the cell's own axis) --------------
    ax3.axhspan(band["lo"], band["hi"], color="tab:green", alpha=0.12,
                label=f"cons band ~1.0")
    # the controller lineage (this cell's axis): fresh 0 -> t400 -> t800
    ax3.plot([0, 800], [E338_RET, E340_RET], ":", color="tab:red", lw=1.2,
             alpha=0.7)
    ax3.plot([0], [E338_RET], "s", color="tab:gray", ms=9,
             label=f"fresh: 0 steps, read 0.286 -> {E338_RET:.4f}")
    ax3.plot([800], [E340_RET], "^", color="tab:red", ms=10,
             label=f"controller t800: 800 steps (32 events), read 0.873 "
                   f"-> {E340_RET:.4f}")
    if not SMOKE:
        ax3.plot([400], [arm["retention"]], "*", color="tab:blue",
                 ms=17, label=f"controller t400: 400 steps (16 events), "
                              f"read 0.815 -> {arm['retention']:.4f} "
                              f"(THIS CELL)")
        ax3.plot([400, 800], [arm["retention"], E340_RET], "-",
                 color="tab:blue", lw=1.6, alpha=0.8)
    # the cons-protocol contrast arms (a different protocol's 300 steps)
    ax3.plot([300, 300], [E341V_RET, E341F_RET], "|", color="tab:green",
             ms=22, mew=2.5)
    ax3.plot([300], [E341V_RET], "v", color="tab:green", ms=8,
             label=f"cons-protocol 300: varied {E341V_RET:.4f} (read "
                   f"0.752)")
    ax3.plot([300], [E341F_RET], "P", color="tab:orange", ms=9,
             label=f"cons-protocol 300: fixed {E341F_RET:.4f} (read "
                   f"0.756)")
    ax3.axhspan(B_STEPS, B_HEIGHT, color="tab:purple", alpha=0.12)
    ax3.set_xlabel("annealing steps on the organism (the controller's own "
                   "axis: 0 -> 400 -> 800; the cons-protocol arms plotted "
                   "at their own dose)")
    ax3.set_ylabel("retention@s100 after commit(0.7) + the wash")
    ax3.set_title("THE STEP-DOSE PICTURE: recovery vs dose at (near-)"
                  "matched read height")
    ax3.legend(fontsize=7, loc="center right")
    ax3.grid(alpha=0.25)

    # ---- panel 4: the s1 annihilation panel (the survival co-read) ---------
    labels, vals, cols = [], [], []
    if not SMOKE and arm.get("s1") is not None:
        labels.append("T400\n+commit\n(this cell)")
        vals.append(max(arm["s1"], 1e-7))
        cols.append("tab:blue")
    for v, col, lab in ((E338_S1, "tab:gray", "fresh\n(e338)"),
                        (E340_S1, "tab:red", "t800\n(e340)"),
                        (E341V_S1, "tab:green", "varied-300\n(e341)"),
                        (E341F_S1, "tab:orange", "fixed-300\n(e341)")):
        labels.append(lab)
        vals.append(max(v, 1e-7))
        cols.append(col)
    bars = ax4.bar(labels, vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        ax4.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.4f}",
                 ha="center", fontsize=8)
    ax4.axhline(READ_BAR, color="black", ls="--", lw=1.2,
                label="the 0.05 aliveness bar")
    ax4.set_yscale("log")
    ax4.set_ylim(1e-4, 1.0)
    ax4.set_ylabel("the s1 read (battery g0 p(T), log)")
    ax4.set_title("THE S1 PANEL (co-read; never adjudicates — the bars "
                  "are retention-only)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25, axis="y")

    fig.suptitle("X39 THE STEP-DOSE RUNG — e336's t400 state (read 0.815, "
                 "HALF the t800 dose, one organism one path) under "
                 "commit(0.7) + the wash", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RD / "x39_step_dose.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def write_report(adj: dict, classes: dict, band: dict, sand: dict,
                 ball_ref: dict, p5: dict) -> None:
    L = []
    A = L.append
    A("# X39 — THE STEP-DOSE RUNG (the height remainder's cheapest probe)")
    A("")
    A(f"* VERDICT: **{adj['verdict']}** — {adj['clause']}")
    A(f"* P-x39a (my registered read): "
      f"{'HIT' if adj['P-x39a']['hit'] else 'MISSED'} "
      f"(guess: {adj['P-x39a']['guess']}; the dispatch's lab lean: "
      f"HEIGHT-CLAUSE-STANDS, weakly — diverged from)")
    A("* gates: " + str(sum(1 for g in metrics["gates"].values()
                             if g.get("pass"))) + "/"
      + str(len(metrics["gates"])) + " PASS"
      + ("" if all(g.get("pass") for g in metrics["gates"].values())
         else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## The subject (the rung + its cleanliness)")
    A("")
    A("e336's committed t400 ERROR-GATED end state — `runs/e336/"
      "ERROR-GATED_post.pt` (md5-bound): TAVIREN at raw **0.814732** "
      "(near-cons-class height; t800's 0.873 is 7% above; the fresh "
      "class was 0.286), controller-annealed for **400 steps with 16 "
      "name-only maintenance events** — HALF the t800 dose. THE RUNG'S "
      "CLEANLINESS (verified against e339's md5-bound record): e339's "
      "REA arm RESUMED this exact state (model + buffers + generators) "
      "and ran the same driver for t401-t800 (events 17-32) — t400 and "
      "t800 are ONE organism on ONE controller path at two dose points; "
      "the read heights sit 7% apart. The dose confound the gradient "
      "carried (height moves with steps) is thereby halved at matched "
      "protocol and near-matched height.")
    A("")
    A("## The event + the test (the harness by import, e341's convention)")
    A("")
    A("`lab/e338_commit_consolidator.py` (md5-bound at birth; e341's own "
      "harness was this module by import) executes VERBATIM: phase_P2's "
      "event port (the 8/8 source quotes, the era-rig md5s, G_BITROOT), "
      "phase_P0's protocol rebuild, phase_P3's bands, wash_arm's e322 "
      "wash arithmetic and arm_summary. The firing is the era's own:")
    A("")
    A("```python")
    A("net0 = G1.CommittedGPT(GB.G1B_CFG)   # the 2.74M family (GB rebind)")
    A("net0.load_state_dict(theta0)          # the body == the t400 subject")
    A("net0.commit(0.7)                      # THE EVENT (R = the W1 dial)")
    A("```")
    A("")
    A("ONE ARM: T400-COMMIT (the dispatch's design); the twin classes "
      "ride free md5-bound (e338's twin 0.0036 SAND-STRICT; e340's twin "
      "0.0018 SAND-STRICT). The wash consumed the BIT-IDENTICAL bitstream "
      "e340's t800 committed arm consumed (G_DRAWS/G_INPUTS against the "
      "md5-bound resume record) — the rung comparison is wash-matched.")
    A("")
    A("## The bar operationalization (the thirds rule, frozen at birth)")
    A("")
    A("The dispatch's '~0.13'/'~0.045' are class-match regions and the "
      "two named classes are NEIGHBORS on the retention line — a "
      "midpoint partition would leave INTERMEDIATE no territory. THE "
      "THIRDS RULE: each class owns the third of the inter-class gap "
      f"(G = {BAR_GAP:.10f}) nearest it; the middle third is "
      "INTERMEDIATE; above the t800 match region is the named residual "
      "SUPER-800 cell. Boundaries re-derived at run time from the "
      "runtime-read classes (G_BARS, equality to 1e-9):")
    A("")
    A(f"- STEPS-BOUGHT: R <= {B_STEPS:.10f}")
    A(f"- INTERMEDIATE: {B_STEPS:.10f} < R < {B_HEIGHT:.10f}")
    A(f"- HEIGHT-CLAUSE-STANDS: {B_HEIGHT:.10f} <= R <= {B_CAP:.10f}")
    A(f"- SUPER-800 (residual): R > {B_CAP:.10f}")
    A("")
    A("## The classes (runtime-read from md5-bound committed records)")
    A("")
    A("| class | steps | events | read t0 | s1 | retention | source |")
    A("|---|---|---|---|---|---|---|")
    A(f"| fresh | 0 | 0 | 0.2859 | {E338_S1:.6f} | {E338_RET:.6f} | "
      f"e338 COMMIT |")
    A(f"| cons-protocol varied | 300 | - | {E341V_T0:.4f} | "
      f"{E341V_S1:.4f} | {E341V_RET:.6f} | e341 VARIED+commit |")
    A(f"| cons-protocol fixed | 300 | - | {E341F_T0:.4f} | "
      f"{E341F_S1:.4f} | {E341F_RET:.6f} | e341 FIXED+commit |")
    A(f"| **controller t400 (THIS CELL)** | **400** | **16** | "
      f"**{SUBJ_READ_T:.4f}** | see table | **see verdict** | "
      f"**x39 T400-COMMIT** |")
    A(f"| controller t800 | 800 | 32 | {E340_T0:.4f} | "
      f"{E340_S1:.6f} | {E340_RET:.6f} | e340 COMMIT |")
    A(f"| cons (the ball) | - | - | ~0.90 | first-panel dips 0.75-0.96 | "
      f"band [{band['lo']:.4f}, {band['hi']:.4f}] | the W1 legs |")
    A("")
    A(f"* sand class (co-reported): retention <= {E38.SAND_OPS_CAP} AND "
      f"read@s100 < {E38.READ_BAR}; strict [{E38.SAND_STRICT[0]}, "
      f"{E38.SAND_STRICT[1]}]; legs: "
      + "; ".join(f"{t['arm']} {t['retention_100']:.4f}"
                  for t in sand["legs"]))
    A(f"* the ball's committed retention (the reference): "
      f"{ball_ref['leg']} = {ball_ref['retention_100']:.4f}")
    A("")
    arm = adj.get("arm", {})
    reads_arm = ({r["step"]: r["read_g0_pT"] for r in arm["traj"]}
                 if arm.get("traj") else {})
    s1_arm = reads_arm.get(1)
    s1_alive_r = adj.get("reads", {}).get("s1_alive")
    if arm.get("traj"):
        A("## THE ARM — T400-COMMIT (e336-t400 + commit(0.7); e322's "
          "100-step unbiased wash, seed 10902, e340's bitstream)")
        A("")
        A("| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | "
          "|d| raw | proj | note |")
        A("|---|---|---|---|---|---|---|---|")
        for r in arm["traj"]:
            note = ""
            if r["step"] == 0:
                note = "t0 (== the committed t400 subject read)"
            elif r["read_g0_pT"] < READ_BAR:
                note = "**< 0.05**"
            A(f"| {r['step']} | {r['read_g0_pT']:.6f} | "
              f"{r['gm12_pT']:.4f} | {r['host_g0_pZ']:.4f} | "
              f"{r['ce_r']:.4f} | {r['disp_raw']:.4f} | "
              f"{r['disp_proj'] if r['disp_proj'] is None else round(r['disp_proj'], 4)}"
              f" | {note} |")
        A("")
        band_note = ("VACUOUS at the smoke horizon"
                     if arm["in_installed_band"] is None else
                     ("IN the cons band" if arm["in_installed_band"]
                      else "outside the cons band"))
        A(f"* retention s{arm['end_step']}/t0: **{arm['retention']:.6f}**"
          f" — {arm['sand_stamp']}; {band_note}")
        if s1_arm is not None:
            A(f"* s1: {s1_arm:.6f} "
              f"({'ALIVE' if s1_alive_r else 'dead'} at the "
              f"0.05 bar); min panel read "
              f"{arm['min_panel_read']:.6f}; all panels >= 0.05: "
              f"{arm['all_panels_ge_005']}")
    A("")
    A("## The match gates")
    A("")
    A(f"* G_DRAWS/G_INPUTS: the arm's wash bit-identical to e340's "
      f"committed t800 arm's bitstream ({p5['g_draws']['n_steps']} "
      f"steps; the md5-bound resume record)")
    A(f"* G_T0: the arm's t0 == the committed t400 subject read "
      f"{SUBJ_READ_T} (|d| {p5['g_t0']['abs_diff']:.1e})")
    A(f"* G_PIN: max raw displacement "
      f"{p5['g_pin']['max_raw_disp_at_panels']:.4f} <= R + "
      f"{E38.PIN_FUZZ} = {E38.EVENT_R + E38.PIN_FUZZ}")
    A("* G_E336RECORD + G_SUBJECT + G_HARNESS + G_CLASSES + G_BARS: the "
      "subject artifact, its record, the imported harness, the "
      "convention parent, every class record and the bar boundaries all "
      "md5-bound/re-derived; the t400 reads re-asserted at the family's "
      "2e-6 law")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* every artifact md5-bound (see metrics.gates); timestamps UTC "
      "only; envelope polls to runs/_envelope_log.jsonl tagged x39:* "
      "(the retag wrapper)")
    A("* the thirds rule + the retention-only bars + the one-arm design "
      "frozen at birth (see metrics.registered + deviations)")
    (RD / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X39 — THE STEP-DOSE RUNG (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = BIRTH_COMMIT_PINNED
    write_partial("startup (bars registered, committed at birth)")
    set_seed(39001)              # global init only; every RNG is its own

    # ---- the harness's own phases (VERBATIM, into this cell's context) ---
    p0 = E38.phase_P0()                        # the protocol rebuild
    p1 = phase_P1_x39(p0)                      # the subject (e336's t400)
    p2 = E38.phase_P2(p1["theta0"])            # THE EVENT (commit 0.7)
    p3 = E38.phase_P3()                        # the bands + sand + ball
    classes = phase_classes()                  # the four classes + G_BARS

    # ---- THE TEST (one arm; the harness hot; GPU single-lane) ------------
    log("=" * 78)
    log("PHASE C — THE WASH (e322's 100-step unbiased wash, seed 10902; "
        "e340's own bitstream)")
    wash = E38.wash_arm("T400-COMMIT", p2["net0"],
                        RD / "wash_T400-COMMIT_resume.pt", p0, armed=True)
    del p2

    # ---- the match gates + the adjudication --------------------------------
    p5 = phase_P5_x39(wash)
    arm_s = E38.arm_summary(wash["traj"], p3["band"], p3["sand"])
    adj = adjudicate_x39(arm_s, classes)

    metrics["arm"] = arm_s
    metrics["adjudication"] = adj
    metrics["thermal"] = {
        "max_temp": (max((r["temp"] for r in thermal_log), default=None)),
        "polls": len(thermal_log),
        "device_events": device_events + E38.device_events,
    }
    metrics["status"] = ("DONE (smoke; nothing adjudicated)" if SMOKE
                         else f"DONE: {adj['verdict']}")
    metrics["git_head_final"] = git_head()
    metrics["completed_utc"] = utcnow()
    save_json(RD / "metrics.json", metrics)

    make_png(adj, classes, p3["band"])
    write_report(adj, classes, p3["band"], p3["sand"], p3["ball_ref"], p5)
    log(f"X39 VERDICT: {adj['verdict']}")


if __name__ == "__main__":
    main()
