"""E285 — THE SANCTUARY CELL — the lab's FIRST BUILD-LANE EXPERIMENT: can a
memory be ENGINEERED to survive its own organism's continued training? This
docstring carries the registered question + bars VERBATIM from the dispatch
letter + the three dials' settings, committed at birth BEFORE any compute.
Adjudicate against exactly this; no bar shopping.

THE LAW BEING TESTED (R67's two-channel kill law + e284's closure): a write
dies if (i) the optimizer's shared state re-aims concurrent traffic into its
room (collision — e284 proved buffer separation kills the re-aiming: the
separate-buffer missile's in-room share 0.0000) OR (ii) the cumulative
free-stream displacement exceeds the write's own norm (transport — e283/x14:
the stream carried the state 14.45 past the 9.18 write). PRESERVATION =
STATE SEPARATION x DISPLACEMENT BUDGET. The sanctuary composes both
protections at their minimal form.

THE ARMS (the established quiet-formed 10k fact, loaded bit-exact; the
corpus stream 1:1):
  (a) SANCTUARY (the composed build): the corpus steps under SGD-M (the
      stable x0.01-class lr, momentum 0.9) with SEPARATE momentum buffers
      (the install/fact side never shares with the corpus side), the corpus
      gradients PROJECTED ORTHOGONAL to the room (the missile construction,
      orthogonality gate instantiated), AND the displacement budget: a
      per-step lr cap such that the projected cumulative corpus displacement
      cannot exceed 0.5 x the write's norm 9.18 over the 400 steps (compute
      the cap from the projected step norms; disclose the realized budget).
  (b) UNPROTECTED-TWIN (the same-session control): the e283 form verbatim
      (this session's reference).
READS: post g0 at t100/200/300/400; the displacement ledger (in-room share +
cumulative norm per milestone); the budget's realized usage; the corpus CE;
the orthogonality ledger (machine-exact).

THE THREE DIALS (frozen HERE at their minimal composed form):
  * DIAL 1 — STATE SEPARATION (the collision channel's cut): SGD-M
    (momentum 0.9, wd 0.0) with TWO SGD instances over the SAME parameters:
    opt_C (the corpus side — the ONLY optimizer stepped in this phase) and
    opt_F (the fact/install side — present but NEVER stepped: the install is
    DONE, the fact stands; the isolation probe machine-verifies per step
    that opt_F's state stays EMPTY — the corpus stream never creates or
    touches fact-side optimizer state). The load-bearing composition check:
    ||P_room buf_C|| / ||buf_C|| < 1e-4 per milestone (the corpus buffer
    carries NO in-room content — SGD momentum is linear in its orthogonal-
    ized grads, so the buffer sits at the fp floor; e284's bar, 100x
    headroom). e284's buf_I composition side is VACUOUS here (no install
    stream in-phase) — disclosed.
  * DIAL 2 — ORTHOGONAL PROJECTION (the collision channel's second cut):
    the corpus gradients orthogonalized g_perp = g - P_room(g) VERBATIM
    (e278/e280/e284's orthogonalize_grads: CPU fp64, write fp32, norm NOT
    rescaled), verified EVERY corpus step; G_ORTH (max ||P_room g_perp||/
    ||g_perp|| over ALL steps < 1e-6) INSTANTIATED (e284's corrected form).
  * DIAL 3 — DISPLACEMENT BUDGET (the transport channel's cut): BUDGET :=
    0.5 x ||theta_fact - theta_base|| (computed fp64 at runtime from the
    loaded fact; ~4.59). Per-step lr cap with EQUAL-SHARE RESERVATION: at
    step t, cap_t = ((BUDGET - S_{t-1}) / (n_steps - t + 1)) / ||b_t|| where
    b_t = 0.9*buf_{t-1} + g_perp is the pending SGD-M buffer (PyTorch's
    exact update: theta -= lr_t * b_t) and S is the running sum of REALIZED
    projected step norms; lr_t = min(LR_STABLE x cosine_lr(t-1,1000),
    cap_t). TRIANGLE-GUARANTEED: S_400 <= BUDGET (every step consumes at
    most its equal share of the remaining budget; surplus rolls forward),
    and the realized cumulative corpus displacement ||theta_400 - theta_fact||
    <= S_400 <= BUDGET. The realized budget is DISCLOSED (S, the realized
    cumulative norm, the usage fractions, the cap's binding frequency, the
    lr price).

FROZEN BARS (survival ratio = post g0 / the loaded baseline 0.26464763283729553):
  - SANCTUARY-HOLDS: ">= 0.5x survival at t400 with both channels verifiably
    cut (the orthogonality gate pass + the budget held) — THE FIRST
    ENGINEERED SURVIVAL; the build lane's founding result; the two-channel
    law CONFIRMED as sufficient."
  - AIM-ONLY-KILLS: "< 0.5x with the budget held but (check the ledger) —
    separation+projection insufficient alone; the transport channel claims
    the kill despite the budget (disclose the leakage arithmetic)."
  - DRIFT-ONLY-KILLS: "< 0.5x with the orthogonality holding but the budget
    blown (the cap mis-set; the transport channel wins) — the law stands,
    the engineering failed; re-cap and one re-run allowed (disclosed)."
  - MIXED/NONE: "everything else — the trajectories verbatim, every ledger,
    no inflation."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * THE PHASE := 400 CORPUS steps, NO install steps (the write already
    stands — e283's convention freeze verbatim); milestones t = 100/200/300/
    400 corpus steps; the phase counter counts corpus steps only.
  * THE VEHICLE := the committed QUIET-FORMED 10k fact (e261's serial cut
    completed by e264; e261_K10K_inst_resume.pt), loaded BIT-EXACT and
    gated THREE ways (artifact md5/size/step/traj/ledger + flat-md5 +
    behavioral read vs the committed literals — G_FACTLOAD); the room := the
    fact's OWN committed K10K room (seeds 26113/26114), rebuilt + bit-gated
    vs e264_rooms.pt (G_ROOMK10K).
  * THE CORPUS STEP := e268's registered form VERBATIM: 48 windows = 16
    original-host anchors (the same 60-window bank) + 32 random corpus
    windows (contiguous train_ids slices); full-window CE; draws
    (aj_c(16), rj_c(32)) from THIS cell's ONE fresh registered generator
    seed 28501 (the family's per-cell rule: 26801/26901/27001/27101/27301/
    27801/28301/28401/HERE 28501); BOTH arms draw the IDENTICAL sequence
    (each arm its own generator instance at the same seed) — bit-identical
    corpus batches across arms, the protection stack + optimizer the arms'
    ONLY delta.
  * THE SANCTUARY OPTIMIZER := SGD-M momentum 0.9, wd 0.0 at LR_STABLE =
    0.01 x 21.7385748014537 = 0.21738574801453703 x cosine_lr(t-1,1000)
    (the nominal schedule; e280/e284's committed lr class, provenance
    re-derived at runtime from the md5-bound runs/e273/lr_calibration.json;
    G_LR_BIND) — then the DIAL-3 cap may reduce lr below the schedule
    (disclosed per step); backward -> clip 1.0 -> g_perp (DIAL 2) -> the
    capped lr -> opt_C.step (buffer C only; DIAL 1).
  * THE TWIN OPTIMIZER := e283's form VERBATIM: ONE fresh AdamW (0.9, 0.95)
    wd 0.1, lr 1e-3 x cosine_lr(t-1,1000); backward -> clip 1.0 ->
    opt.step FREE (NO projection, NO separation, NO cap) — this session's
    unprotected reference (e283's committed twin hard-bound in G_PARENTS as
    the cross-check: post 9.04614535102155e-06, ratio 0.0000342x).
  * THE SURVIVAL RATIO := post g0(t) / 0.26464763283729553 (the committed
    loaded baseline, PRIMARY; the session's loaded read co-reported).
  * THE ADJUDICATION READ := the t=400 endpoint (both channels' gate
    verdicts at the same endpoint); full milestone trajectories + the twin's
    co-reported verbatim.
  * COMPOSITE := TEXTURE (any hard-gate failure — nothing adjudicated) ->
    SANCTUARY-HOLDS (ratio_400 >= 0.5x AND G_ORTH pass AND G_BUDGET pass) ->
    AIM-ONLY-KILLS (ratio_400 < 0.5x AND G_BUDGET pass — the leakage
    arithmetic disclosed from the ledgers) -> DRIFT-ONLY-KILLS (ratio_400 <
    0.5x AND G_BUDGET blown AND G_ORTH pass — one disclosed re-cap re-run
    allowed) -> MIXED/NONE (everything else — trajectories verbatim, every
    ledger, no inflation).
  * HARD GATES (a failure HALTS): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND,
    G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND, G_ORTH,
    G_BUFSEP, G_BUDGET}.
  * G_ORTH := max over ALL sanctuary corpus steps of ||P_room g_perp|| /
    ||g_perp|| < 1e-6 (INSTANTIATED — checked every step, written into the
    gates dict; e280's omission corrected at e284, carried).
  * G_BUFSEP := (i) ISOLATION, every step: opt_F's state snapshotted before
    every corpus opt_C.step() and compared bitwise after — the fact side
    stays EMPTY, never created, never touched (any mismatch HALTS); (ii)
    COMPOSITION, per milestone + final: ||P_room buf_C||/||buf_C|| < 1e-4
    (the fp-floor bar; e284's SEP arm measured 1.5e-9-5.9e-9). The install
    side's composition bar is VACUOUS in this phase (no install stream —
    e284's buf_I > 0.99 check has no object; disclosed).
  * G_BUDGET := S_400 <= BUDGET + 1e-5 (the fp32 accumulation slack,
    1000x the expected accumulation error, disclosed) AND the realized
    cumulative corpus displacement ||theta_400 - theta_fact|| <= BUDGET +
    1e-5; the ledger discloses budget_norm, S_400, the realized cumulative
    norm, both usage fractions, the cap's binding count, and the lr price
    (median/min/max applied lr vs the nominal schedule).
  * NO CONS (T259/e281; e278/e280/e283/e284's committed form): the frozen
    bars read the WRITE and the DISPLACEMENT only; both arms' post-phase
    states are CHECKPOINTED for any later landing pass.
  * NO SERIAL ARM (cited + hard-bound): e283's same-form committed numbers
    (its storage-decay control read the loaded fact bit-identically at
    every milestone — the null; its concurrent arm post 9.04614535102155e-06)
    are the reference record; the same-session twin carries the replication.

REGISTERED PREDICTIONS (from the dispatch letter's law, cited not new):
  - P-e285a (the law's sufficiency reading): with both channels cut the
    write SURVIVES — the orthogonalized, buffer-separated, budget-capped
    stream cannot re-aim into the room (in-room share at the fp floor, the
    e284 SEP arithmetic) and cannot transport the state past half the
    write's own norm; SANCTUARY-HOLDS.
  - P-e285b (the x14 counter-reading, disclosed): the transport kill is
    DIRECTIONAL — x14 measured the kill carried by the OUT-OF-ROOM context
    (the orthogonal-subtraction arm resurrected the read to 0.0392 = 4328x
    the dead state; the in-room subtraction left it dead) — an ORTHOGONAL
    stream is exactly an out-of-room context mover, so even a
    budget-respecting orthogonal displacement may shift the read through
    function-space coupling (LN/softmax): AIM-ONLY-KILLS with the leakage
    arithmetic = the drift ledger's decomposition (how far the state moved,
    at what in-room share, vs the write's mass and alignment). The read
    decides; the bars cover both.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (inside the dispatch's 180s), per-step thermal polls at a 78C
margin, 40s cooldowns (the 30-60s window), the 84C never-past line (inside
the dispatch's 85C), polls persisted to runs/_envelope_log.jsonl tagged
e285:<ARM>:<phase>; CPU fp64 dense projections (pocketfft workers 2); CPU
probing threads 4; NO concurrent GPU jobs (the two arms run sequentially
with cooldowns between).

Outputs: runs/e285/{metrics.json (PROGRESSIVE), e285_sanctuary.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e285_*.pt (gitignored; md5s in metrics). No NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit + push
per phase.

Run:  cd lab && python e285_sanctuary.py    (E285_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E285_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e285_smoke" if SMOKE else "e285"
assert torch.cuda.is_available(), "e285 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268/e273/e278/e283/e284's disclosed convention): e261's
# drivers resolve their module globals (log / NAME / LADDER / RUNG_NAMES /
# SMOKE / T0 / thermal ledgers) AT CALL TIME through e261's module namespace
# — rebound HERE so the room build + envelope polls label THIS cell. The
# committed lab/e261_rank_ladder.py is untouched.
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
# room the established fact was WRITTEN IN (the vehicle's own room — the
# sanctuary protects the write IN ITS OWN ROOM; e283's convention)
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
# unprotected twin — the same-session reference)
ARMS = ("SANCTUARY", "UNPROTECTED-TWIN")
SAN_ARM, TWIN_ARM = ARMS
ARM_DESC = {
    SAN_ARM: "(a) SANCTUARY (the composed build): the corpus steps under "
             "SGD-M (the stable x0.01-class lr, momentum 0.9) with SEPARATE "
             "momentum buffers (the install/fact side never shares with the "
             "corpus side), the corpus gradients PROJECTED ORTHOGONAL to "
             "the room (the missile construction, orthogonality gate "
             "instantiated), AND the displacement budget: a per-step lr cap "
             "such that the projected cumulative corpus displacement cannot "
             "exceed 0.5 x the write's norm 9.18 over the 400 steps "
             "(compute the cap from the projected step norms; disclose the "
             "realized budget)",
    TWIN_ARM: "(b) UNPROTECTED-TWIN (the same-session control): the e283 "
              "form verbatim (this session's reference) — ONE fresh AdamW "
              "(0.9,0.95) wd 0.1 at lr 1e-3 x cosine_lr(t-1,1000), clip "
              "1.0 -> opt.step FREE (no projection, no separation, no cap)",
}

# THIS cell's ONE fresh registered corpus stream (the e268-family
# convention: one fresh registered stream per concurrent cell — e268 26801,
# e269 26901, e270 27001, e271 27101, e273 27301, e278 27801, e280 28001,
# e283 28301, e284 28401, HERE 28501). BOTH arms draw the IDENTICAL sequence
# (each arm its own generator instance at the same seed) — bit-identical
# corpus batches across arms, the protection stack + optimizer the arms'
# ONLY delta.
CORPUS_GEN_SEED = 28501

# THE CONVENTION FREEZES (e283's, carried verbatim — no install exists)
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

# ---- THE SGD-M CONFIG (frozen; e280/e284's committed lr class VERBATIM) ----
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.21738574801453703

# ---- DIAL 3 — THE DISPLACEMENT BUDGET (frozen) --------------------------
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
E283_WRITE_NORM = 9.1788432658723            # the write's own norm (||fact-base||)

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

E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268's G_PARENTS bind

# the frozen bars' numbers ------------------------------------------------
SURVIVE_FRAC = 0.5               # SANCTUARY-HOLDS: >= 0.5x at t400
ORTH_BAR = 1e-6                  # G_ORTH: the missile's orthogonality gate
BUFSEP_ORTH_BAR = 1e-4           # G_BUFSEP: ||P_room buf_C||/||buf_C|| below
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
RATIO_DEN_FLOOR = 1e-6           # the ladder's floor-guard convention
VOCAB_EXPECT = 65

REGISTERED = {
    "law_verbatim": "a write dies if (i) the optimizer's shared state re-aims "
        "concurrent traffic into its room (collision — e284 proved buffer "
        "separation kills the re-aiming: the separate-buffer missile's "
        "in-room share 0.0000) OR (ii) the cumulative free-stream "
        "displacement exceeds the write's own norm (transport — e283/x14: "
        "the stream carried the state 14.45 past the 9.18 write). "
        "PRESERVATION = STATE SEPARATION x DISPLACEMENT BUDGET. The "
        "sanctuary composes both protections at their minimal form.",
    "arms_verbatim": ARM_DESC,
    "bars_verbatim": {
        "SANCTUARY-HOLDS": ">= 0.5x survival at t400 with both channels "
            "verifiably cut (the orthogonality gate pass + the budget held) "
            "— THE FIRST ENGINEERED SURVIVAL; the build lane's founding "
            "result; the two-channel law CONFIRMED as sufficient.",
        "AIM-ONLY-KILLS": "< 0.5x with the budget held but (check the "
            "ledger) — separation+projection insufficient alone; the "
            "transport channel claims the kill despite the budget (disclose "
            "the leakage arithmetic).",
        "DRIFT-ONLY-KILLS": "< 0.5x with the orthogonality holding but the "
            "budget blown (the cap mis-set; the transport channel wins) — "
            "the law stands, the engineering failed; re-cap and one re-run "
            "allowed (disclosed).",
        "MIXED/NONE": "everything else — the trajectories verbatim, every "
            "ledger, no inflation.",
    },
    "reads_verbatim": "post g0 at t100/200/300/400; the displacement ledger "
        "(in-room share + cumulative norm per milestone); the budget's "
        "realized usage; the corpus CE; the orthogonality ledger "
        "(machine-exact).",
    "operationalizations": (
        "frozen BEFORE compute: THE PHASE := 400 CORPUS steps, no install "
        "(e283's convention freeze); milestones t=100/200/300/400; THE "
        "VEHICLE := the committed quiet-formed 10k fact loaded BIT-EXACT "
        "(three-way G_FACTLOAD); the room := the fact's OWN K10K room "
        "(seeds 26113/26114, bit-gated vs e264_rooms.pt); THE CORPUS STEP "
        f":= e268's registered form VERBATIM on THIS cell's ONE fresh "
        f"registered stream seed {CORPUS_GEN_SEED} (BOTH arms draw the "
        "identical sequence — the protection stack is the arms' ONLY "
        "delta); DIAL 1 (STATE SEPARATION) := SGD-M momentum 0.9 wd 0 with "
        "TWO SGD instances over the same params — opt_C (corpus side, the "
        "only one stepped) + opt_F (fact side, NEVER stepped in-phase; the "
        "isolation probe verifies per step it stays EMPTY) + the "
        "composition bar ||P_room buf_C||/||buf_C|| < 1e-4 per milestone "
        "(e284's fp-floor bar; the buf_I side VACUOUS — no install stream, "
        "disclosed); DIAL 2 (ORTHOGONAL PROJECTION) := g_perp = g - "
        "P_room(g) VERBATIM (e278/e280/e284), verified EVERY step, G_ORTH "
        "< 1e-6 INSTANTIATED; DIAL 3 (DISPLACEMENT BUDGET) := BUDGET = "
        f"{BUDGET_FRAC} x ||fact - base|| (fp64 at runtime, ~4.59); the "
        "per-step lr cap with EQUAL-SHARE RESERVATION cap_t = ((BUDGET - "
        "S_{t-1})/(n_steps - t + 1))/||b_t||, b_t = 0.9*buf_{t-1} + g_perp "
        "(PyTorch's exact update: theta -= lr_t * b_t); lr_t = "
        "min(LR_STABLE x cosine_lr(t-1,1000), cap_t); TRIANGLE-GUARANTEED "
        "S_400 <= BUDGET and ||cum displacement|| <= S_400 <= BUDGET; the "
        "realized budget DISCLOSED; THE TWIN := e283's form VERBATIM (one "
        "fresh AdamW 0.9/0.95 wd 0.1 at 1e-3 x cosine, clip 1.0 -> "
        "opt.step FREE); THE SURVIVAL RATIO := post g0(t) / "
        f"{FACT_BASELINE_G0} (committed, PRIMARY); THE ADJUDICATION READ "
        ":= the t=400 endpoint with both gate verdicts at the same "
        "endpoint; COMPOSITE := TEXTURE (any hard-gate failure) -> "
        "SANCTUARY-HOLDS (ratio_400 >= 0.5 AND G_ORTH AND G_BUDGET) -> "
        "AIM-ONLY-KILLS (ratio_400 < 0.5 AND G_BUDGET held; the leakage "
        "arithmetic disclosed) -> DRIFT-ONLY-KILLS (ratio_400 < 0.5 AND "
        "G_BUDGET blown AND G_ORTH held; one disclosed re-cap re-run "
        "allowed) -> MIXED/NONE (everything else); HARD GATES := "
        "{G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, "
        "G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, "
        "G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND, G_ORTH, G_BUFSEP, "
        "G_BUDGET} (a failure HALTS); NO CONS (T259/e281; both post states "
        "checkpointed); NO SERIAL ARM (e283's committed same-form record "
        "hard-bound as the reference)."),
    "registration": "bars + question + arms + dials frozen VERBATIM from "
        "the dispatch letter (the e284 fold's dispatch; the build lane's "
        "founding cell); this script committed at birth BEFORE any "
        "compute; adjudicate against exactly this; no bar shopping.",
    "predictions": {
        "P-e285a_law_sufficiency": "with both channels cut the write "
            "SURVIVES — the orthogonalized, buffer-separated, budget-"
            "capped stream cannot re-aim into the room (in-room share at "
            "the fp floor, e284's SEP arithmetic) and cannot transport the "
            "state past half the write's own norm; SANCTUARY-HOLDS.",
        "P-e285b_x14_counter_reading": "the transport kill is DIRECTIONAL "
            "(x14: the out-of-room context carries the kill — the "
            "orthogonal-subtraction arm resurrected the read to 0.0392 = "
            f"{X14_RESURRECTION_X:.0f}x the dead state while the in-room "
            "subtraction left it dead) — an ORTHOGONAL stream is exactly "
            "an out-of-room context mover, so even a budget-respecting "
            "orthogonal displacement may shift the read through function-"
            "space coupling (LN/softmax): AIM-ONLY-KILLS with the leakage "
            "arithmetic from the drift ledger. The read decides; the bars "
            "cover both.",
    },
}

deviations: list[str] = [
    "THE OPT_F DISCLOSURE (the separation's honest minimal form): in this "
    "phase there is NO install stream — the fact stands, formed long ago "
    "under e261's own optimizer state. DIAL 1's separation therefore "
    "instantiates as: the corpus stream's momentum buffer is PRIVATE (its "
    "own SGD instance, the only optimizer stepped), with a second SGD "
    "instance opt_F (the fact side) present-but-NEVER-stepped — the "
    "isolation probe machine-verifies per corpus step that opt_F's state "
    "stays EMPTY (never created, never touched; any mismatch HALTS). The "
    "load-bearing composition check is buf_C's in-room fraction at the fp "
    "floor (< 1e-4; e284's SEP arm measured 1.5e-9-5.9e-9). e284's buf_I "
    "> 0.99 composition bar is VACUOUS here (no install stream — no "
    "object); disclosed, not gated.",
    "THE BUDGET'S PRICE (disclosed at birth, measured in the CE ledger): "
    "the equal-share reservation holds the per-step displacement at "
    "~BUDGET/400 ~ 0.0115 — under the nominal LR_STABLE schedule the "
    "post-warmup steps want ~0.4-0.7 — so the cap binds from the first "
    "post-warmup steps and the sanctuary's corpus stream runs at a "
    "~x0.02-of-stable effective lr. The corpus CE will improve far less "
    "than the twin's (e283's free stream: 0.93 -> 0.77); the sanctuary "
    "buys preservation at the price of corpus learning — the cell's own "
    "economics read, never a bar.",
    "THE TWIN IS THIS SESSION'S OWN DRAW STREAM (seed 28501, shared "
    "bit-identically with the sanctuary arm — e284's tightened precedent): "
    "e283's committed twin ran seed 28301, so this twin replicates the "
    "FORM not the bits; its expected landing is the e283 class (post "
    "~1e-5, ratio ~3e-5x), cross-checked against the hard-bound committed "
    "record. The class replication is the reference; bit-replication was "
    "never the family's rule.",
    "THE SANCTUARY'S SGD-M vs THE TWIN'S ADAMW (the composed delta, "
    "disclosed): the arms differ in the WHOLE protection stack (optimizer "
    "class + buffers + projection + cap), not one dial at a time — the "
    "build lane's question is whether the COMPOSED form preserves, not "
    "which dial contributes what (the dial decomposition is e280/e283/"
    "e284's already-committed record). The one-dial contrasts stand: "
    "e283 (AdamW free vs nothing), e284 (SEP vs SHA), x14 (in-room vs "
    "out-of-room subtraction).",
    "NO CONS (T259/e281; e278/e280/e283/e284's committed form): the frozen "
    "bars read the WRITE and the DISPLACEMENT only; both arms' post-phase "
    "states are checkpointed (e285_SANCTUARY_post.pt / "
    "e285_UNPROTECTED-TWIN_post.pt) for any later landing pass.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector + "
    "LadderRooms (the room rebuild, certification, displacement loads), "
    "the thermal envelope (per-step polls, 78C margin, 175s bursts inside "
    "the dispatch's 180s, 40s cooldowns, the 84C line inside the "
    "dispatch's 85C), the progressive-metrics + resume-ckpt conventions — "
    "the module-global rebinding (log/NAME/LADDER/RUNG_NAMES/T0/thermal "
    "ledgers, disclosed in-code) retargets the machinery's I/O to this "
    "cell; the committed lab/e261_rank_ladder.py is NOT modified. The TWO "
    "drivers are THIS file's: chunked_sanctuary_phase (the composed build) "
    "and chunked_twin_phase (e283's chunked_corpus_phase ported verbatim "
    "in body — tags and checkpoint names adapted).",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the measured "
    "displacement loads; no new history is run.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E285_SMOKE=1): 8 corpus steps per arm, milestones at "
    "every step 1..8, room k=512 at the same seed pair (G_ROOMK10K "
    "vacuous — no committed record at smoke k; disclosed), the REAL "
    "committed fact loaded and read (G_FACTLOAD live), G_CORPUSGEN live, "
    "G_ORTH live (every step), G_BUFSEP live (isolation + composition), "
    "G_BUDGET live — the smoke budget is SCALED to the full run's "
    "PER-STEP SHARE (BUDGET x 8/400), so the cap's binding path runs at "
    "exactly the full-run share scale (disclosed); the adjudication form + "
    "figure exercised; all paths smoke_-prefixed, own smoke dir; NOTHING "
    "adjudicated or gated for the record (SMOKE stamp on every read).",
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
    """e278/e280/e284's orthogonalize_grads VERBATIM: replace the (clipped)
    corpus gradient g by g_perp = g - P_room(g) — the component ENTIRELY
    ORTHOGONAL to the room (CPU fp64, write fp32, norm NOT rescaled). The
    verification read ||P_room g_perp|| / ||g_perp|| is returned for the
    gate (the projector is exact: P(I-P) = 0 to fp64 roundoff ~1e-15; any
    drift above 1e-6 is an implementation bug, not numerics)."""
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
    buffers + the (orthogonalized) grads now in p.grad — the DIAL-3 cap's
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
# THE SANCTUARY DRIVER — the composed build (DIALS 1+2+3)
# ======================================================================
def chunked_sanctuary_phase(tag: str, net0, proj: "E261.LadderRooms",
                            fact_flat_np: np.ndarray,
                            base_flat_np: np.ndarray, budget_norm: float,
                            anchor_full, train_ids, g0_ids, gm12_ids,
                            r_eval_xy, zid, resume_ck: Path,
                            dev: torch.device) -> dict:
    """THE SANCTUARY ARM'S DRIVER (the cell's composed build). The net
    starts at the LOADED formed fact (net0 carries it); per corpus step
    t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 28501 — bit-identical to
      the twin's draws); corpus batch = 16 original-host anchors + 32
      random corpus windows; full-window CE; lr_sched = LR_STABLE x
      cosine_lr(t-1, 1000); backward -> clip 1.0 -> DIAL 2: g_perp = g -
      P_room(g) (verified EVERY step) -> DIAL 3: the per-step lr cap
      (equal-share reservation against the pending buffer norm) -> DIAL 1:
      opt_C.step() (the corpus side's OWN momentum buffer; opt_F snapshotted
      bitwise around the step — the fact side stays EMPTY).

    Ledgers: the orthogonality ledger (every step checked, rows every 10);
    the budget ledger (per milestone: S, realized cum, usage, the cap's
    binding); the buffer ledger (per milestone: buf_C composition + the
    isolation counts); the displacement ledger (e283's form: the drift from
    the fact with in-room share, the remaining-from-base occupancy, the
    corpus interval/cumulative displacement with in-room fractions); the
    WRITE read (g0/gm12 battery + CE_R) at the milestones. Thermal: a poll
    after EVERY corpus opt step."""
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
                f"{SAN_ARM}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            # DIAL 1: TWO SGD instances over the SAME parameters — the
            # corpus side's buffer is PRIVATE; the fact side never steps.
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
            # ---- DIAL 3: the per-step lr cap (equal-share reservation)
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
                f"{SAN_ARM}:corpus:c{n_chunks}.x")
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


# ======================================================================
# THE TWIN DRIVER — e283's chunked_corpus_phase VERBATIM in body (the
# unprotected reference: fresh AdamW, FREE steps, e283's drift ledger)
# ======================================================================
def chunked_twin_phase(tag: str, net0, proj: "E261.LadderRooms",
                       fact_flat_np: np.ndarray, base_flat_np: np.ndarray,
                       anchor_full, train_ids, g0_ids, gm12_ids,
                       r_eval_xy, zid, resume_ck: Path,
                       dev: torch.device) -> dict:
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
                f"{TWIN_ARM}:corpus:chunk{n_chunks + 1}")
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
            f = cosine_lr(step - 1, E261.INST_TOTAL)
            for g in opt.param_groups:
                g["lr"] = lr * f
            # ---- THE CORPUS STEP (e283's form VERBATIM, bit-identical
            # draws to the sanctuary arm's)
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
            ok_t, temp = E261.burst_temp_check(
                f"{TWIN_ARM}:corpus:c{n_chunks}.x")
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
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "disp_ledger": state["disp_ledger"],
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
           "per_step_polls": "after EVERY corpus opt step (both arms) — "
                             "aggregated from runs/_envelope_log.jsonl (the "
                             "persisted ledger; survives resume passes; "
                             "tags e285:<ARM>:phase per the dispatch)",
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
                   f"polls are tagged e285_smoke: and excluded")
    return out


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e285_sanctuary",
        "phase": "THE SANCTUARY CELL — the lab's FIRST BUILD-LANE "
                 "EXPERIMENT: can a memory be ENGINEERED to survive its own "
                 "organism's continued training? R67's two-channel kill law "
                 "(collision via shared optimizer state OR transport via "
                 "free-stream displacement beyond the write's norm) tested "
                 "at its composed minimal form — STATE SEPARATION (SGD-M, "
                 "separate buffers) x DISPLACEMENT BUDGET (orthogonal "
                 "projection + per-step lr cap at 0.5x the write's norm) — "
                 "vs the same-session unprotected twin (e283's form): "
                 "SANCTUARY-HOLDS vs AIM-ONLY-KILLS vs DRIFT-ONLY-KILLS vs "
                 "MIXED/NONE, adjudicated on the WRITE read at t400 with "
                 "both channels' gate verdicts",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property; "
                           "the frozen bars read the WRITE and the "
                           "DISPLACEMENT only; e278/e280/e283/e284's "
                           "committed NO-CONS form; both arms' states are "
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
                      "e285:<ARM>:<phase>",
            "trainings": "2 corpus-only phases x 400 opt steps (SANCTUARY: "
                         "SGD-M separate buffers + orthogonal projection + "
                         "budget-capped lr; UNPROTECTED-TWIN: e283's fresh "
                         "AdamW FREE); NO install (the fact is loaded "
                         "bit-exact); NO cons",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} CORPUS steps (no install steps — "
                     "e283's convention freeze carried verbatim)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors (the same "
                           "60-window bank) + 32 random corpus windows; "
                           "full-window CE; THIS cell's ONE fresh registered "
                           f"generator seed {CORPUS_GEN_SEED}; BOTH arms "
                           "draw the IDENTICAL sequence (the protection "
                           "stack is the arms' ONLY delta)",
            "survival_ratio": f"post g0(t) / {FACT_BASELINE_G0} (the "
                              "committed loaded baseline, PRIMARY; the "
                              "session's loaded read co-reported)",
            "adjudication_read": "the t=400 endpoint with both channels' "
                                 "gate verdicts at the same endpoint; "
                                 "trajectories + the twin co-reported",
        },
        "the_three_dials": {
            "dial_1_state_separation": "SGD-M (momentum 0.9, wd 0) with TWO "
                                       "SGD instances over the same params: "
                                       "opt_C (corpus side, the ONLY one "
                                       "stepped) + opt_F (fact side, NEVER "
                                       "stepped in-phase — the isolation "
                                       "probe verifies per step it stays "
                                       "EMPTY); composition bar "
                                       "||P_room buf_C||/||buf_C|| < 1e-4 "
                                       "per milestone (e284's fp-floor bar; "
                                       "the buf_I side vacuous — disclosed)",
            "dial_2_orthogonal_projection": "g_perp = g - P_room(g) VERBATIM "
                                            "(e278/e280/e284), CPU fp64 / "
                                            "write fp32, verified EVERY "
                                            "corpus step; G_ORTH < 1e-6 "
                                            "INSTANTIATED",
            "dial_3_displacement_budget": "BUDGET = 0.5 x ||fact - base|| "
                                          "(fp64 at runtime, ~4.59); "
                                          "per-step lr cap with equal-share "
                                          "reservation: cap_t = ((BUDGET - "
                                          "S_{t-1})/(n-t+1))/||b_t||, "
                                          "b_t = 0.9*buf_{t-1} + g_perp; "
                                          "lr_t = min(LR_STABLE x "
                                          "cosine_lr(t-1,1000), cap_t); "
                                          "TRIANGLE-GUARANTEED S_400 <= "
                                          "BUDGET and ||cum|| <= S_400 <= "
                                          "BUDGET; the realized budget "
                                          "disclosed",
        },
        "deviations": deviations,
        "builds_on": [
            "R67 / the ideator's two-channel kill law (PRESERVATION = STATE "
            "SEPARATION x DISPLACEMENT BUDGET — the morning's four cells "
            "its special cases; THIS cell the law's first composed test)",
            "T264 / e284 (MOMENTUM-OWNED: the separate-buffer missile's "
            "corpus walk 0.0000 in-room vs the shared twin's 0.4503 — "
            "buffer separation kills the re-aiming; DIAL 1 + DIAL 2's "
            "machinery ports from here)",
            "T261 / e283 (the established-write rig VERBATIM: the loaded "
            "fact, the 400-corpus-step phase, the conventions; the "
            "unprotected reference post 9.04614535102155e-06, ratio "
            "0.0000342x, drift 14.45 vs the write's 9.18 — the transport "
            "cite)",
            "T263 / x14 (the transport intervention: DIRECTIONAL TRANSPORT "
            "with the coupling caveat — the out-of-room context carries "
            "4,328x of the kill; the write's mass 84.35% intact + "
            "write-aligned; P-e285b's source)",
            "T260 / e278 (THE ROACH MOTEL: the missile construction — the "
            "orthogonal projection, the orthogonality ledger — ORIGINATES "
            "here)",
            "T258 / e273 (TRAJECTORY-TWO-BODY + the lr calibration: "
            "LR_STABLE's provenance, md5-bound)",
            "T242 / e264 + T239 / e261 (the committed threshold rung: the "
            "loaded fact's baseline; the ladder machinery PORTED WHOLE BY "
            "IMPORT)",
            "T259 / e281 (the NO-CONS form)",
        ],
        "whats_new": [
            "THE BUILD LANE OPENS (the record's first): every prior cell "
            "measured a kill; this cell ENGINEERS a preservation and tests "
            "whether it holds — the two-channel law's sufficiency question",
            "THE COMPOSED PROTECTION (the record's first): state separation "
            "+ orthogonal projection + a displacement budget in one "
            "interleaved organism — each dial already proven alone (e284 "
            "SEP; e278 missile; the budget new), never composed",
            "THE DISPLACEMENT BUDGET (the record's first transport-side "
            "intervention IN TRAINING): a per-step lr cap with equal-share "
            "reservation, triangle-guaranteed to hold the cumulative "
            "corpus displacement under 0.5x the write's own norm — the "
            "engineering dial the transport channel predicts is necessary",
            "G_BUDGET + G_BUFSEP + G_ORTH INSTANTIATED (the build lane's "
            "verification stack: the budget ledger machine-checked, the "
            "buffer isolation machine-checked every step, the "
            "orthogonality machine-checked every step)",
        ],
        "gates": {},
    })
    log(f"E285 — THE SANCTUARY CELL (the build lane opens; smoke={SMOKE}) "
        f"-> {RD}")
    log(f"arms: {' / '.join(ARMS)}; the fact = {FACT_CK} (md5-bound, post "
        f"g0 {FACT_BASELINE_G0:.8f}); the room = the fact's own committed "
        f"K10K (seeds {LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs "
        f"{ROOMS264_CK}); the phase = {PHASE_STEPS} corpus steps; "
        f"milestones {'/'.join(str(m) for m in MILESTONES)}; SANCTUARY = "
        f"SGD-M m{SGD_MOMENTUM} wd{SGD_WD} @ LR_STABLE "
        f"{LR_STABLE:.13f} x cosine, SEPARATE buffers + g_perp + BUDGET "
        f"{BUDGET_FRAC}x write norm (equal-share cap); TWIN = e283's "
        f"AdamW (0.9,0.95) wd 0.1 @ 1e-3 x cosine FREE; the corpus stream "
        f"= seed {CORPUS_GEN_SEED} (bit-identical across arms); the bar: "
        f"HOLDS >= {SURVIVE_FRAC:.0%}x at t400 with both channels cut")
    write_partial("startup (bars + dials registered, committed at birth)")
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
        "/ e170 bank / install mask identity / vocab 65)")
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
                         "note": "THE UNPROTECTED REFERENCE (the twin's "
                                 "form; post 9.05e-06, drift 14.45 vs the "
                                 "write's 9.18 — the transport cite; its "
                                 "storage-decay control read the fact "
                                 "bit-identically — the null)"},
        "e284_metrics": {"path": str(E284_METRICS),
                         "md5": md5of(E284_METRICS), "bound_md5": E284_MD5,
                         "verdict": e284m["adjudication"]["verdict"],
                         "sep_primary": e284_sep, "sha_primary": e284_sha,
                         "note": "THE SEPARATION RECORD (the separate-"
                                 "buffer missile's in-room share 0.0000 — "
                                 "DIAL 1+2's machinery + provenance)"},
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
                                "out-of-room context carries the kill; "
                                "P-e285b's source; the coupling caveat "
                                "disclosed in its own record)"},
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
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E283_METRICS) == E283_MD5
            and md5of(E284_METRICS) == E284_MD5
            and md5of(X14_METRICS) == X14_MD5
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
    log(f"P0b: G_PARENTS PASS — e283 {E283_VERDICT} (post {e283_post:.2e}, "
        f"ratio {E283_RATIO:.2e}, drift {e283_drift:.2f} vs the write's "
        f"{E283_WRITE_NORM:.2f}); e284 {E284_VERDICT} (SEP in-room "
        f"{e284_sep:.1e} vs SHA {e284_sha:.4f}); x14 {X14_VERDICT} "
        f"(arm A {X14_ARM_A:.4f} = {X14_RESURRECTION_X:.0f}x the dead "
        f"state); the fact ckpt md5/size/step-bound")
    write_partial("P0b parents hard-bound (the fact ckpt loaded)")
    del e264m, e268m, e283m, e284m, x14m

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
        "e285_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e285's room (the fact's own room): the committed K10K "
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

    # ---- G_LR_BIND: the lr/momentum provenance bind (e280/e284's record)
    lr_stable_runtime = SGD_STABLE_FACTOR * float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the SANCTUARY's nominal lr := e273's STABLE RIDER POINT "
                "(e280/e284's committed class) — LR_STABLE = x0.01 x "
                "LR_SGD_matched, re-derived at runtime from the md5-bound "
                "runs/e273/lr_calibration.json and asserted == the frozen "
                "literal; momentum 0.9, wd 0.0 EXACTLY (e273's stable "
                "rider); the nominal schedule LR_STABLE x "
                "cosine_lr(t-1,1000); the DIAL-3 budget cap may reduce the "
                "APPLIED lr below the schedule (disclosed per step in the "
                "lr_ledger) — the cap is part of the intervention's body",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
        "twin_lr": "e283's form VERBATIM: AdamW at E43.LR (1e-3) x "
                   "cosine_lr(t-1,1000), betas (0.9,0.95), wd 0.1",
        "pass": bool(abs(lr_stable_runtime - LR_STABLE) < 1e-15
                     and float(LR_STABLE) == 0.21738574801453703
                     and SGD_MOMENTUM == 0.9 and SGD_WD == 0.0
                     and E43.LR == 1e-3),
    }
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P1 G_LR_BIND: LR_STABLE = {SGD_STABLE_FACTOR} x {E273_LR_SGD} = "
        f"{LR_STABLE!r} (runtime {lr_stable_runtime!r}); momentum "
        f"{SGD_MOMENTUM}, wd {SGD_WD}; the twin at AdamW 1e-3: PASS")
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
                    "channel's reference quantity; DIAL 3's budget = "
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

    # ---- the BUDGET (DIAL 3's denominator, frozen at load time) --------
    budget_norm = BUDGET_FRAC * write_norm
    if SMOKE:
        # the smoke budget is SCALED to the full run's PER-STEP SHARE so
        # the cap's binding path runs at the full-run share scale
        budget_norm = budget_norm * PHASE_STEPS / 400
    metrics["budget"] = {
        "form": f"BUDGET := {BUDGET_FRAC} x ||fact - base|| (the write's "
                "own norm, fp64 from the loaded fact)",
        "write_norm": write_norm,
        "budget_norm": budget_norm,
        "reservation": "EQUAL-SHARE: cap_t = ((BUDGET - S_{t-1}) / "
                       "(n_steps - t + 1)) / ||b_t||; lr_t = "
                       "min(LR_STABLE x cosine, cap_t); S += realized "
                       "||step||; triangle-guaranteed S_400 <= BUDGET and "
                       "||cum|| <= S_400 <= BUDGET",
        "slack": BUDGET_SLACK,
        "smoke_scaling": ("SMOKE: budget x 8/400 — the full run's per-step "
                          "share preserved so the binding path runs"
                          if SMOKE else None),
    }
    log(f"P2b THE BUDGET: {BUDGET_FRAC} x {write_norm:.4f} = "
        f"{budget_norm:.4f} (the transport channel's cap; per-step share "
        f"~{budget_norm / PHASE_STEPS:.5f})")
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
                "26801/26901/27001/27101/27301/27801/28301/28401/28501), "
                "its own generator; per corpus step the draws (aj_c(16), "
                "rj_c(32)) in that order; the first step's draws logged "
                "here from a scratch generator (the drivers' cgens start "
                "identically by construction); BOTH arms draw the "
                "IDENTICAL sequence",
        "seed": CORPUS_GEN_SEED,
        "first_step_aj": [int(x) for x in aj_probe.tolist()],
        "first_step_rj": [int(x) for x in rj_probe.tolist()],
        "composition": "16 original-host anchors (the 60-window bank) + 32 "
                       "random corpus windows (contiguous train_ids "
                       "slices); full-window CE; clip 1.0 -> the arm's own "
                       "step (SANCTUARY: g_perp + capped lr + opt_C; TWIN: "
                       "opt.step FREE)",
        "pass": True,
    }
    metrics["gates"]["G_CORPUSGEN"] = G_CORPUSGEN
    log(f"P2c G_CORPUSGEN: the corpus stream REGISTERED (seed "
        f"{CORPUS_GEN_SEED}; first draws aj[:4]="
        f"{G_CORPUSGEN['first_step_aj'][:4]} "
        f"rj[:4]={G_CORPUSGEN['first_step_rj'][:4]})")
    write_partial("P2c the corpus generator registered")

    # ================= P3: ARM SANCTUARY (the composed build) ===========
    log("=" * 78)
    log(f"ARM-{SAN_ARM} — {ARM_DESC[SAN_ARM]}")
    san = chunked_sanctuary_phase(
        f"{SAN_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, budget_norm, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e285_SANCTUARY_resume.pt" if SMOKE
                    else "e285_SANCTUARY_resume.pt"), dev)
    sd_s = san["sd"]
    net_s = G1.evl_load(sd_s)
    cells_s = {"gm12": G1.battery_cell(net_s, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_s, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_s, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_s, *r_eval_xy)}
    d_final_s = flat_params_cpu(net_s) - fact_flat
    loads_final_s = rooms.displacement_loads(d_final_s, ROOM_MODE)
    del net_s
    corp_ce_s = [v["ce"] for v in san["corpus_ledger"].values()]
    corp_gn_s = [v["gn_clipped"] for v in san["corpus_ledger"].values()]
    corp_gin_s = [v["g_in_room_frac"] for v in san["corpus_ledger"].values()
                  if v.get("g_in_room_frac") is not None]
    san_ck = save_ckpt(
        "e285_SANCTUARY_post", sd_s,
        {"desc": "e285 ARM-SANCTUARY post-phase state: the committed "
                 "quiet-formed 10k fact + 400 budgeted orthogonal SGD-M "
                 "corpus steps (separate buffers, generator "
                 f"{CORPUS_GEN_SEED}) — NO install, NO cons",
         "arm": SAN_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "budget_norm": budget_norm, "S": san["S"],
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"] = {SAN_ARM: {
        "desc": ARM_DESC[SAN_ARM], "phase": {
            "traj": san["traj"], "corpus_ledger": san["corpus_ledger"],
            "orth_ledger": san["orth_ledger"],
            "disp_ledger": san["disp_ledger"],
            "buf_ledger": san["buf_ledger"],
            "budget_ledger": san["budget_ledger"],
            "lr_ledger": san["lr_ledger"],
            "corpus_ce_median": med(corp_ce_s),
            "corpus_gn_clipped_median": med(corp_gn_s),
            "corpus_g_in_room_frac_median": med(corp_gin_s),
            "orth_max_rel_err": san["orth_max"],
            "bufsep": san["bufsep"],
            "budget": {"budget_norm": budget_norm, "S_final": san["S"],
                       "S_usage_frac": san["S"] / budget_norm,
                       "cum_final_norm":
                           san["disp_ledger"][-1]["corpus_disp_cum_norm"]
                           if san["disp_ledger"] else None,
                       "n_capped": san["n_capped"],
                       "n_steps": PHASE_STEPS,
                       "capped_frac": san["n_capped"] / PHASE_STEPS,
                       "lr_applied_min": san["lr_applied_min"],
                       "lr_applied_median": san["lr_applied_median"],
                       "lr_sched_median": san["lr_sched_median"],
                       "cap_min": san["cap_min"]},
            "chunk_table": san["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_s,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_s.double().numpy())), **loads_final_s},
            "checkpoint": san_ck,
            "resumed_final": bool(san.get("resumed_final", False)),
        }}}
    log(f"ARM-{SAN_ARM} DONE: post g0 {cells_s['g0']:.7f} "
        f"(x{cells_s['g0'] / FACT_BASELINE_G0:.4f}) g-12 {cells_s['gm12']:.7f} "
        f"CE_R {cells_s['ce_r']:.4f} | BUDGET S {san['S']:.4f}/"
        f"{budget_norm:.4f} ({san['S'] / budget_norm:.1%}) | capped "
        f"{san['n_capped']}/{PHASE_STEPS} | lr med "
        f"{san['lr_applied_median'] if san['lr_applied_median'] is not None else float('nan'):.5f} "
        f"(sched med {san['lr_sched_median'] if san['lr_sched_median'] is not None else float('nan'):.5f}) "
        f"| ORTH max {san['orth_max']:.2e} | isolation "
        f"{san['bufsep']['isolation_checks']} checks "
        f"{san['bufsep']['isolation_violations']} violations | corpus CE med "
        f"{med(corp_ce_s):.4f}")
    write_partial(f"ARM-{SAN_ARM} complete (the composed build's ledgers)")

    # ================= P4: ARM UNPROTECTED-TWIN (e283's form) ===========
    E261.burst_cooldown(f"{SAN_ARM} -> {TWIN_ARM}")
    log("=" * 78)
    log(f"ARM-{TWIN_ARM} — {ARM_DESC[TWIN_ARM]}")
    tw = chunked_twin_phase(
        f"{TWIN_ARM}", G1.evl_load(fact_sd), rooms, fact_flat_np,
        base_flat_np, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e285_TWIN_resume.pt" if SMOKE
                    else "e285_TWIN_resume.pt"), dev)
    sd_t = tw["sd"]
    net_t = G1.evl_load(sd_t)
    cells_t = {"gm12": G1.battery_cell(net_t, gm12_ids, zid)["mean_pz"],
               "g0": G1.battery_cell(net_t, g0_ids, zid)["mean_pz"],
               "gp12": G1.battery_cell(net_t, bat_ids[12], zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_t, *r_eval_xy)}
    d_final_t = flat_params_cpu(net_t) - fact_flat
    loads_final_t = rooms.displacement_loads(d_final_t, ROOM_MODE)
    del net_t
    corp_ce_t = [v["ce"] for v in tw["corpus_ledger"].values()]
    corp_gn_t = [v["gn_clipped"] for v in tw["corpus_ledger"].values()]
    corp_gin_t = [v["g_in_room_frac"] for v in tw["corpus_ledger"].values()
                  if v.get("g_in_room_frac") is not None]
    tw_ck = save_ckpt(
        "e285_UNPROTECTED-TWIN_post", sd_t,
        {"desc": "e285 ARM-UNPROTECTED-TWIN post-phase state: the committed "
                 "quiet-formed 10k fact + 400 FREE AdamW corpus steps "
                 f"(e283's form, generator {CORPUS_GEN_SEED}) — the "
                 "same-session unprotected reference",
         "arm": TWIN_ARM, "corpus_gen_seed": CORPUS_GEN_SEED,
         "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
         "rooms": rooms_ck})
    metrics["arms"][TWIN_ARM] = {
        "desc": ARM_DESC[TWIN_ARM], "phase": {
            "traj": tw["traj"], "corpus_ledger": tw["corpus_ledger"],
            "disp_ledger": tw["disp_ledger"],
            "corpus_ce_median": med(corp_ce_t),
            "corpus_gn_clipped_median": med(corp_gn_t),
            "corpus_g_in_room_frac_median": med(corp_gin_t),
            "chunk_table": tw["chunk_table"], "steps": PHASE_STEPS,
            "post_cells": cells_t,
            "drift_from_fact_final": {
                "norm": float(np.linalg.norm(
                    d_final_t.double().numpy())), **loads_final_t},
            "checkpoint": tw_ck,
            "resumed_final": bool(tw.get("resumed_final", False)),
        }}
    log(f"ARM-{TWIN_ARM} DONE: post g0 {cells_t['g0']:.7f} "
        f"(x{cells_t['g0'] / FACT_BASELINE_G0:.4f}) g-12 {cells_t['gm12']:.7f} "
        f"CE_R {cells_t['ce_r']:.4f} | drift "
        f"{float(np.linalg.norm(d_final_t.double().numpy())):.3f} | corpus "
        f"CE med {med(corp_ce_t):.4f} (e283's committed twin: post "
        f"{E283_POST:.2e}, x{E283_RATIO:.2e}, drift {E283_DRIFT:.2f})")
    write_partial(f"ARM-{TWIN_ARM} complete (the same-session reference)")

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
        f"bit-identical draws — the arms' ONLY delta is the protection "
        f"stack)")

    # ================= P5: the instantiated gates ========================
    # G_ORTH — the orthogonality gate (INSTANTIATED, every step)
    orth_max_s = san["orth_max"]
    G_ORTH = {
        "form": "the stepped corpus gradient is ENTIRELY ORTHOGONAL to the "
                "room on the SANCTUARY arm: max over ALL corpus steps of "
                "||P_room g_perp|| / ||g_perp|| < 1e-6 (checked EVERY "
                "corpus step via e278/e280/e284's orthogonalize_grads; the "
                "SRCT projector is exact, so any drift above fp64 roundoff "
                "is an implementation bug). INSTANTIATED (e280's omission "
                "corrected at e284, carried)",
        "orth_max_rel_err": orth_max_s,
        "bar": ORTH_BAR,
        "n_steps_checked": PHASE_STEPS,
        "note": "the TWIN's gradients are deliberately UNPROJECTED (e283's "
                "form) — no orthogonality applies to the reference",
        "pass": bool(orth_max_s is not None and orth_max_s < ORTH_BAR),
    }
    assert G_ORTH["pass"], f"orthogonality gate FAILED: {G_ORTH}"
    metrics["gates"]["G_ORTH"] = G_ORTH

    # G_BUFSEP — the separation's machine verification
    bufC_fr = [b["bufC_inroom_frac"] for b in san["buf_ledger"]
               if b.get("bufC_inroom_frac") is not None]
    optF_entries = [b.get("optF_state_entries") for b in san["buf_ledger"]]
    G_BUFSEP = {
        "form": "the buffer separation's machine verification (the "
                "minimal in-phase form): (i) ISOLATION — the fact side's "
                "optimizer (opt_F) snapshotted bitwise before EVERY "
                "corpus opt_C.step() and compared after; it must stay "
                "EMPTY (never created, never touched — there is no "
                "install stream in this phase; any mismatch HALTS); (ii) "
                "COMPOSITION — per milestone, ||P_room buf_C||/||buf_C|| "
                "< 1e-4 (the fp-floor bar; e284's SEP arm measured "
                "1.5e-9-5.9e-9). e284's buf_I > 0.99 composition side is "
                "VACUOUS here (no install stream — no object; disclosed "
                "at birth)",
        "isolation_checks": san["bufsep"]["isolation_checks"],
        "isolation_violations": san["bufsep"]["isolation_violations"],
        "optF_state_entries_max": max(optF_entries) if optF_entries else 0,
        "optF_ever_stepped": bool(
            san["bufsep"].get("optF_ever_stepped", False)),
        "bufC_inroom_frac_max": max(bufC_fr) if bufC_fr else None,
        "bufC_inroom_frac_last": bufC_fr[-1] if bufC_fr else None,
        "bufC_bar": BUFSEP_ORTH_BAR,
        "e284_sep_cite": E284_SEP_PRIMARY,
        "pass": None,           # computed explicitly below (a hard gate)
    }
    G_BUFSEP["pass"] = bool(
        san["bufsep"]["isolation_violations"] == 0
        and san["bufsep"]["isolation_checks"] == PHASE_STEPS
        and max(optF_entries if optF_entries else [0]) == 0
        and not san["bufsep"].get("optF_ever_stepped", False)
        and bufC_fr and max(bufC_fr) < BUFSEP_ORTH_BAR)
    assert G_BUFSEP["pass"], f"buffer separation gate FAILED: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    # G_BUDGET — the displacement budget's machine verification
    S_final = float(san["S"])
    cum_final = (san["disp_ledger"][-1]["corpus_disp_cum_norm"]
                 if san["disp_ledger"] else None)
    drift_final = (float(np.linalg.norm(d_final_s.double().numpy())))
    G_BUDGET = {
        "form": "the displacement budget's machine verification: the "
                "equal-share per-step lr cap holds the running sum of "
                "realized projected step norms S under BUDGET "
                f"(triangle-guaranteed), and the realized cumulative "
                "corpus displacement ||theta_400 - theta_fact|| <= S <= "
                f"BUDGET; slack {BUDGET_SLACK:.0e} absolute (fp32 "
                "accumulation, 1000x the expected error, disclosed)",
        "budget_norm": budget_norm,
        "budget_frac_of_write_norm": BUDGET_FRAC,
        "write_norm": write_norm,
        "S_final": S_final,
        "S_usage_frac": S_final / budget_norm,
        "cum_disp_final_norm": cum_final,
        "cum_usage_frac": (cum_final / budget_norm
                           if cum_final is not None else None),
        "drift_from_fact_final_norm": drift_final,
        "n_capped": san["n_capped"], "n_steps": PHASE_STEPS,
        "capped_frac": san["n_capped"] / PHASE_STEPS,
        "lr_price": {"lr_applied_median": san["lr_applied_median"],
                     "lr_applied_min": san["lr_applied_min"],
                     "lr_sched_median": san["lr_sched_median"]},
        "slack": BUDGET_SLACK,
        "twin_contrast": {
            "drift": float(np.linalg.norm(d_final_t.double().numpy())),
            "cite_e283_drift": E283_DRIFT,
            "note": "the unprotected streams' realized cumulative "
                    "displacement (this session's twin + e283's committed "
                    "arm) — what the budget exists to prevent"},
        "pass": bool(S_final <= budget_norm + BUDGET_SLACK
                     and cum_final is not None
                     and cum_final <= budget_norm + BUDGET_SLACK),
    }
    assert G_BUDGET["pass"], f"budget gate FAILED: {G_BUDGET}"
    metrics["gates"]["G_BUDGET"] = G_BUDGET
    log(f"P5 GATES: G_ORTH PASS (max {orth_max_s:.2e} < {ORTH_BAR:.0e} "
        f"over {PHASE_STEPS} steps); G_BUFSEP PASS "
        f"({G_BUFSEP['isolation_checks']} isolation checks, 0 violations, "
        f"opt_F EMPTY; bufC in-room max "
        f"{max(bufC_fr) if bufC_fr else float('nan'):.2e} < "
        f"{BUFSEP_ORTH_BAR:.0e}); G_BUDGET PASS (S {S_final:.4f} <= "
        f"{budget_norm:.4f} + {BUDGET_SLACK:.0e}; cum {cum_final:.4f}; "
        f"capped {san['n_capped']}/{PHASE_STEPS})")
    write_partial("P5 gates complete (ORTH + BUFSEP + BUDGET)")

    # ================= P6: ADJUDICATION (the frozen bars) ================
    ratio_400 = cells_s["g0"] / FACT_BASELINE_G0
    ratio_session_denom = cells_s["g0"] / max(fact_g0, RATIO_DEN_FLOOR)
    twin_ratio = cells_t["g0"] / FACT_BASELINE_G0
    ratio_mile = {t["step"]: t["g0_pz"] / FACT_BASELINE_G0
                  for t in san["traj"]}
    ratio_min = min(ratio_mile.values()) if ratio_mile else None

    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))
    budget_held = bool(G_BUDGET["pass"])
    orth_held = bool(G_ORTH["pass"])

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif ratio_400 >= SURVIVE_FRAC and orth_held and budget_held:
        verdict = "SANCTUARY-HOLDS"
        clause = (f">= {SURVIVE_FRAC:.0%}x survival at t400 "
                  f"(ratio {ratio_400:.4f}x) with both channels verifiably "
                  f"cut (G_ORTH max {orth_max_s:.1e} < {ORTH_BAR:.0e} over "
                  f"all {PHASE_STEPS} steps; G_BUDGET S {S_final:.3f} <= "
                  f"{budget_norm:.3f}, realized cum {cum_final:.3f}, capped "
                  f"{san['n_capped']}/{PHASE_STEPS} steps) — THE FIRST "
                  f"ENGINEERED SURVIVAL; the build lane's founding result; "
                  f"the two-channel law CONFIRMED as sufficient (the "
                  f"same-session unprotected twin died at {twin_ratio:.2e}x "
                  f"on bit-identical draws; e283's committed reference "
                  f"{E283_RATIO:.2e}x). The price: corpus CE med "
                  f"{med(corp_ce_s):.4f} vs the twin's {med(corp_ce_t):.4f} "
                  f"(the budget's lr cost, disclosed)")
    elif ratio_400 < SURVIVE_FRAC and budget_held:
        verdict = "AIM-ONLY-KILLS"
        dl_s = {int(d["step"]): d for d in san["disp_ledger"]}
        last = san["disp_ledger"][-1] if san["disp_ledger"] else {}
        in_room_cum = last.get("corpus_disp_cum_in_room_frac")
        rem_in_room = last.get("remaining_from_base_in_room_frac")
        clause = (
            f"< {SURVIVE_FRAC:.0%}x survival at t400 (ratio "
            f"{ratio_400:.4f}x) with the budget HELD (S {S_final:.3f} <= "
            f"{budget_norm:.3f}; realized cum {cum_final:.3f} = "
            f"{cum_final / budget_norm:.1%} of budget) AND the "
            f"orthogonality + separation holding (orth {orth_max_s:.1e}; "
            f"bufC in-room "
            f"{(max(bufC_fr) if bufC_fr else float('nan')):.1e}) — "
            f"separation+projection+cap insufficient alone; the transport "
            f"channel claims the kill DESPITE the budget. THE LEAKAGE "
            f"ARITHMETIC: the state moved {drift_final:.3f} "
            f"({drift_final / write_norm:.1%} of the write's "
            f"{write_norm:.2f}) with the realized displacement's in-room "
            f"share at the fp floor "
            f"({('%.*e' % (2, in_room_cum)) if in_room_cum is not None else 'n/a'} "
            f"— the movement is OUT-OF-ROOM by construction), yet the "
            f"write's read fell to {ratio_400:.4f}x while its own mass "
            f"occupancy reads "
            f"{('%.*f' % (4, rem_in_room)) if rem_in_room is not None else 'n/a'} "
            f"in-room — the kill rides FUNCTION-SPACE COUPLING (x14's "
            f"directional-transport reading: the out-of-room context "
            f"carried {X14_RESURRECTION_X:.0f}x of the kill there; the "
            f"orthogonal stream moves exactly that context). The "
            f"two-channel law's transport clause UNDER-BUDGETED: the kill "
            f"threshold sits BELOW 0.5x the write's norm — the budget dial "
            f"is real but its 0.5 setting was not conservative enough at "
            f"this organism's coupling")
    elif ratio_400 < SURVIVE_FRAC and (not budget_held) and orth_held:
        verdict = "DRIFT-ONLY-KILLS"
        clause = (
            f"< {SURVIVE_FRAC:.0%}x survival at t400 (ratio "
            f"{ratio_400:.4f}x) with the orthogonality holding (max "
            f"{orth_max_s:.1e}) BUT the budget BLOWN (S {S_final:.4f} > "
            f"{budget_norm:.4f} + {BUDGET_SLACK:.0e} or cum "
            f"{cum_final:.4f}) — the cap mis-set; the transport channel "
            f"wins; the law stands, the engineering failed; re-cap and "
            f"ONE disclosed re-run allowed (the dispatch's freeze)")
    else:
        verdict = "MIXED/NONE"
        clause = (f"everything else — ratio_400 {ratio_400:.4f}x, "
                  f"G_ORTH {'PASS' if orth_held else 'FAIL'}, G_BUDGET "
                  f"{'PASS' if budget_held else 'FAIL'} (e.g. survival "
                  f"with a gate blown, or a non-monotone trajectory the "
                  f"bars did not name) — the trajectories verbatim, every "
                  f"ledger, no inflation")

    log("=" * 78)
    log(f"E285 VERDICT: {verdict}")
    log(f"  SANCTUARY: post g0 {cells_s['g0']:.8f} (x{ratio_400:.4f} "
        f"committed-denom; x{ratio_session_denom:.4f} session-denom)")
    log(f"  UNPROTECTED-TWIN: post g0 {cells_t['g0']:.8f} "
        f"(x{twin_ratio:.4f}; e283's committed x{E283_RATIO:.2e})")
    log(f"  milestones (SANCTUARY): "
        + "; ".join(f"t{t} x{r:.4f}" for t, r in ratio_mile.items()))
    log(f"  the two channels: ORTH max {orth_max_s:.2e} (bar "
        f"{ORTH_BAR:.0e}); BUDGET S {S_final:.4f}/{budget_norm:.4f} "
        f"({S_final / budget_norm:.1%}), cum {cum_final:.4f} "
        f"({(cum_final / budget_norm) if cum_final else float('nan'):.1%})")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any hard-gate failure) -> "
                           "SANCTUARY-HOLDS (ratio_400 >= 0.5 AND G_ORTH "
                           "AND G_BUDGET) -> AIM-ONLY-KILLS (ratio_400 < "
                           "0.5 AND G_BUDGET held; leakage arithmetic "
                           "disclosed) -> DRIFT-ONLY-KILLS (ratio_400 < 0.5 "
                           "AND G_BUDGET blown AND G_ORTH held; one "
                           "disclosed re-cap re-run allowed) -> MIXED/NONE "
                           "(everything else) — frozen at birth",
        "gates_pass": gates_pass,
        "reads": {
            SAN_ARM: {"post_g0": cells_s["g0"],
                      "survival_ratio_committed_denom": ratio_400,
                      "survival_ratio_session_denom": ratio_session_denom,
                      "post_gm12": cells_s["gm12"],
                      "post_ce_r": cells_s["ce_r"],
                      "traj_g0": {t["step"]: t["g0_pz"]
                                  for t in san["traj"]},
                      "survival_ratio_milestones": ratio_mile,
                      "min_milestone_ratio": ratio_min,
                      "corpus_ce_median": med(corp_ce_s),
                      "corpus_g_in_room_frac_median": med(corp_gin_s),
                      "drift_ledger": san["disp_ledger"],
                      "budget_ledger": san["budget_ledger"],
                      "buf_ledger": san["buf_ledger"],
                      "orth_ledger_max": orth_max_s,
                      "drift_from_fact_final_in_room":
                          loads_final_s["in_own_room"]},
            TWIN_ARM: {"post_g0": cells_t["g0"],
                       "survival_ratio": twin_ratio,
                       "post_gm12": cells_t["gm12"],
                       "post_ce_r": cells_t["ce_r"],
                       "traj_g0": {t["step"]: t["g0_pz"]
                                   for t in tw["traj"]},
                       "corpus_ce_median": med(corp_ce_t),
                       "drift_ledger": tw["disp_ledger"],
                       "drift_from_fact_final_norm":
                           float(np.linalg.norm(d_final_t.double().numpy())),
                       "drift_from_fact_final_in_room":
                           loads_final_t["in_own_room"]},
            "the_unprotected_reference_e283": {
                "post_g0": E283_POST, "ratio": E283_RATIO,
                "drift": E283_DRIFT, "write_norm": E283_WRITE_NORM,
                "note": "the committed same-form record (hard-bound in "
                        "G_PARENTS); this session's twin replicates the "
                        "FORM on its own draw stream"},
            "the_two_channels": {
                "collision_side": {
                    "orth_max": orth_max_s, "bar": ORTH_BAR,
                    "bufC_inroom_max": max(bufC_fr) if bufC_fr else None,
                    "bar_bufC": BUFSEP_ORTH_BAR,
                    "verdict": "CUT (the stream never aims in-room; the "
                               "corpus buffer at the fp floor)"},
                "transport_side": {
                    "budget_norm": budget_norm, "S_final": S_final,
                    "cum_final": cum_final, "drift_final": drift_final,
                    "write_norm": write_norm,
                    "verdict": ("HELD" if budget_held else "BLOWN")}},
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
            "corpus draws (one generator, seed 28501, one draw order) — "
            "the ONLY delta is the protection stack (SGD-M separate "
            "buffers + orthogonal projection + the budget cap) vs e283's "
            "free AdamW. The rooms' eigenstructures are identical across "
            "arms BY CONSTRUCTION (one bit-gated room)."),
        "the_budget_disclosure": (
            "the budget's equal-share reservation holds the per-step "
            "displacement at ~BUDGET/400 — under the nominal schedule the "
            "post-warmup steps want far more — so the cap binds and the "
            "sanctuary's corpus CE improves far less than the twin's: the "
            "cell's economics read (preservation bought at corpus "
            "learning's price), measured in the CE ledgers, never assumed"),
        "the_separation_disclosure": (
            "in this phase there is no install stream; DIAL 1's separation "
            "is the corpus buffer's PRIVACY (one stepped SGD) + opt_F's "
            "machine-checked emptiness + buf_C's fp-floor composition — "
            "e284's buf_I composition side is vacuous here (disclosed at "
            "birth). The isolation is machine-checked every step (bitwise "
            "snapshots)"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; nothing guaranteed"),
        "loads_measured_not_nominal": (
            "every read is measured: the per-step realized displacement "
            "and its norm (the budget's S), the per-milestone interval + "
            "cumulative projections (fp64), the buffer-composition ledger, "
            "the orthogonality ledger (every step), the corpus CE + "
            "clipped-grad ledgers — never nominal"),
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
            "post_states": {SAN_ARM: san_ck, TWIN_ARM: tw_ck},
        },
        "machinery": {
            "sanctuary_phase": "THIS file's chunked_sanctuary_phase (the "
                               "cell's composed build): e268's corpus step "
                               "+ e284's orthogonalize_grads + the "
                               "two-SGD buffer topology + the equal-share "
                               "budget cap",
            "twin_phase": "THIS file's chunked_twin_phase: e283's "
                          "chunked_corpus_phase VERBATIM in body (the "
                          "unprotected reference)",
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
    make_sanctuary_plot(RD, metrics, verdict, clause, thermal_log,
                        budget_norm, write_norm)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e285_sanctuary.png"),
                          str(RD / "REPORT.md")]
    write_partial("P8 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_sanctuary_plot(rd, metrics, verdict, clause, thermal_log,
                        budget_norm, write_norm):
    """THE CELL'S HEADLINE FIGURE: the survival curves (the sanctuary vs
    the unprotected twin vs the baseline + bars), the budget ledger (S +
    realized cum vs the budget line), the two channels' ledgers (the
    in-room shares + the buffer composition + the orthogonality), the
    corpus CE (the budget's price), the thermal envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    san = metrics["arms"][SAN_ARM]["phase"]
    tw = metrics["arms"][TWIN_ARM]["phase"]
    st, tt = san["traj"], tw["traj"]
    dl_s = {int(d["step"]): d for d in san["disp_ledger"]}
    dl_t = {int(d["step"]): d for d in tw["disp_ledger"]}
    bl = san["budget_ledger"]
    ol = san["orth_ledger"]

    # (0,0) THE SURVIVAL CURVES
    ax = axes[0, 0]
    ax.plot([t["step"] for t in st], [max(t["g0_pz"], 1e-7) for t in st],
            "o-", lw=1.9, ms=5, color="tab:green",
            label=f"{SAN_ARM} (the composed build)")
    ax.plot([t["step"] for t in tt], [max(t["g0_pz"], 1e-7) for t in tt],
            "s--", lw=1.6, ms=4.5, color="tab:red",
            label=f"{TWIN_ARM} (e283's form, this session)")
    ax.axhline(FACT_BASELINE_G0, color="black", ls=":", lw=1.3,
               label=f"the loaded fact {FACT_BASELINE_G0:.4f}")
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.4, label=f"the {SURVIVE_FRAC:.0%}x HOLDS bar "
               f"({SURVIVE_FRAC * FACT_BASELINE_G0:.4f})")
    ax.axhline(E283_POST, color="darkorange", ls="-.", lw=1.0,
               label=f"e283's committed death {E283_POST:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("corpus step t (NO install — the phase's own counter)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE SURVIVAL CURVES — the sanctuary vs the unprotected "
                 "twin (bit-identical draws)", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE BUDGET LEDGER (DIAL 3)
    ax = axes[0, 1]
    if bl:
        ax.plot([b["step"] for b in bl], [b["S"] for b in bl], "o-",
                lw=1.8, ms=5, color="tab:blue",
                label="S = sum of realized step norms (triangle bound)")
        ax.plot([b["step"] for b in bl],
                [b.get("cum_norm") for b in bl], "s--", lw=1.4, ms=4,
                color="tab:cyan",
                label="realized cumulative displacement ||cum||")
    ax.axhline(budget_norm, color="crimson", ls="--", lw=1.5,
               label=f"THE BUDGET {BUDGET_FRAC}x write norm = "
               f"{budget_norm:.3f}")
    ax.axhline(write_norm, color="black", ls=":", lw=1.1,
               label=f"the write's own norm {write_norm:.2f} (e283's death "
               f"line: drift {E283_DRIFT:.1f})")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("cumulative displacement (L2)")
    ax.set_title("THE DISPLACEMENT BUDGET (the transport channel's cap, "
                 "machine-held)", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)

    # (0,2) THE TWO CHANNELS' IN-ROOM LEDGERS
    ax = axes[0, 2]
    steps_s = sorted(dl_s)
    ax.semilogy(steps_s,
                [max(dl_s[s]["corpus_disp_cum_in_room_frac"], 1e-12)
                 for s in steps_s], "o-", lw=1.6, ms=5,
                color="tab:green", label="SANCTUARY cum in-room frac")
    steps_t = sorted(dl_t)
    ax.plot(steps_t, [dl_t[s]["corpus_disp_cum_in_room_frac"]
                      for s in steps_t], "s--", lw=1.4, ms=4.5,
            color="tab:red", label="TWIN cum in-room frac")
    ax.axhline(BUFSEP_ORTH_BAR, color="crimson", ls="--", lw=1.0,
               label=f"the fp-floor bar {BUFSEP_ORTH_BAR:.0e}")
    ax.axhline(math.sqrt(LADDER[0][0] / 2739072), color="gray", ls=":",
               lw=1.1, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(LADDER[0][0] / 2739072):.4f}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("in-room share of the realized displacement")
    ax.set_title("CHANNEL 1 (collision): the sanctuary's stream never "
                 "aims in-room", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (1,0) THE BUFFER COMPOSITION + ORTHOGONALITY
    ax = axes[1, 0]
    sb = [b for b in san["buf_ledger"]
          if b.get("bufC_inroom_frac") is not None]
    ax.semilogy([b["step"] for b in sb],
                [max(b["bufC_inroom_frac"], 1e-15) for b in sb], "o-",
                lw=1.6, ms=5, color="tab:purple",
                label="buf_C in-room frac (the corpus buffer)")
    ax.axhline(BUFSEP_ORTH_BAR, color="crimson", ls="--", lw=1.2,
               label=f"G_BUFSEP bar {BUFSEP_ORTH_BAR:.0e}")
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("||P_room buf_C|| / ||buf_C|| (log)")
    osteps = sorted(int(k) for k in ol.keys())

    def _orth_get(k):
        return ol[str(k)]["orth_rel_err"] if str(k) in ol \
            else ol[k]["orth_rel_err"]
    ax2 = ax.twinx()
    ax2.semilogy(osteps, [max(_orth_get(k), 1e-18) for k in osteps],
                 "s--", lw=0.9, ms=2.5, color="dimgray", alpha=0.7,
                 label="orth rel err ||P g_perp||/||g_perp|| (right)")
    ax2.axhline(ORTH_BAR, color="crimson", ls=":", lw=1.0)
    ax2.set_ylabel("orthogonality rel err (log)", fontsize=8.5)
    ax.set_title(f"CHANNEL 1 (state): buf_C at the fp floor + the "
                 f"orthogonality (max "
                 f"{san['orth_max_rel_err']:.1e}; opt_F EMPTY, "
                 f"{san['bufsep']['isolation_checks']} checks)", fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6)
    ax.grid(alpha=0.25, which="both")

    # (1,1) THE CORPUS CE + THE BUDGET'S LR PRICE
    ax = axes[1, 1]
    for a, col, mk in ((SAN_ARM, "tab:green", "o"), (TWIN_ARM, "tab:red",
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
    lr_rows = [(int(k), v) for k, v in san["lr_ledger"].items()]
    ax3.plot([k for k, _ in lr_rows],
             [v["lr_applied"] for _, v in lr_rows], "^-", lw=1.0, ms=3,
             color="tab:blue", alpha=0.8, label="SANCTUARY lr applied")
    ax3.plot([k for k, _ in lr_rows],
             [v["lr_sched"] for _, v in lr_rows], "^--", lw=0.8, ms=2,
             color="tab:blue", alpha=0.35, label="nominal schedule")
    ax3.set_ylabel("SANCTUARY lr (applied vs nominal)", fontsize=8.5,
                   color="tab:blue")
    ax.set_title(f"THE BUDGET'S PRICE (the sanctuary's corpus CE vs the "
                 f"twin's; capped "
                 f"{san['budget']['n_capped']}/{san['budget']['n_steps']}"
                 f" steps)", fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6)
    ax.grid(alpha=0.25)

    # (1,2) THE THERMAL ENVELOPE + THE DISCRIMINATOR BAR READ
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
                 f"reads: SANCTUARY x"
                 f"{metrics['arms'][SAN_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f} "
                 f"vs TWIN x"
                 f"{metrics['arms'][TWIN_ARM]['phase']['post_cells']['g0'] / FACT_BASELINE_G0:.4f}",
                 fontsize=8.6)

    fig.suptitle(f"E285 — THE SANCTUARY CELL (the build lane opens) -> "
                 f"{verdict}", fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e285_sanctuary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
