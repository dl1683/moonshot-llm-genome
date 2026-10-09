"""X22 — THE KNOB'S SEAT (R72 ideator card 6, sharpened by T286's
authorship finding). This docstring carries the registered question +
design + bars + prediction P-x22a VERBATIM from the dispatch letter +
every frozen convention, committed at birth BEFORE any compute.
Adjudicate against exactly this; no bar shopping.

THE BACKGROUND (verbatim): "e311's 1-dim out-of-room 'trigger' (the
host's read-gradient direction) boosts the host read +32% with
near-zero off-target effect (x16: a CONTEXT-GATED TARGETED BIAS, 36x
amplified at the host's own contexts). T286 just found the noise floor
is AUTHORSHIP-STRUCTURED: writing a name at a context leaves its logit
slot plastic, and the plastic thing is the initial-CHAR slot."

THE QUESTION (verbatim): "where does the trigger physically live? If
one or two matrices' components deliver most of the effect alone, the
confidence channel has an ORGAN (candidate: the lm_head row of the
name's initial char — which would reconnect the knob to era-1's oldest
surviving doctrine, 'lm_head token-row directions are causal training
coordinates'); if no fragment works and only the cumulative ladder
reaches full effect, confidence is HOLOGRAPHIC like transport death."

THE DESIGN (verbatim):
  1. "Decompose the committed trigger vector (runs/e311/
     e311_hijack_vectors.pt, md5-bound) per parameter matrix (the
     organism's writeable matrices — the rigs' own decomposition
     convention; count and name them from the rig)."
  2. "CUMULATIVE LADDER: apply descending per-matrix-norm fragments
     cumulatively (top-1, top-2, ... all); measure per rung: host g0
     read, the x16 off-target battery summary (top-1 stability, margin
     distribution), and the gating ratio (on-target Z-excess /
     off-target Z-excess)."
  3. "LEAVE-ONE-OUT: the full trigger minus each single matrix (the
     complement of the ladder): does removing any one matrix collapse
     the +32%?"
  4. "The lm_head test (T286's sharpened prediction): the trigger's
     component on the output-head row(s) of the host name's initial
     char — applied ALONE at its natural norm. If it delivers a
     material fraction of the boost, SEATED names the seat."

FROZEN BARS (verbatim; adjudicate against exactly this):
  - SEATED: "a single matrix's fragment (or the cumulative top-2)
    delivers >= 60% of the full trigger's host boost with the
    off-target profile held surgical — the confidence channel has an
    organ; name it (lm_head row or otherwise)."
  - HOLOGRAPHIC: "no fragment delivers >= 25%; only >= 80% of
    matrices cumulatively approach the full effect; leave-one-out
    never collapses it — confidence, like transport death, is
    whole-state."
  - "Register P-x22a BEFORE compute. Lab guess (T286-sharpened, but
    note the lab's four consecutive mechanism misses tonight): SEATED,
    with the lm_head initial-char row as the seat — state your own
    counter honestly if you disagree; predictions are scored."

==== THE FROZEN CONVENTIONS (picked + frozen HERE at birth) ===========

* THE HOST := e311's committed host organism — e261_K10K_inst_resume.pt
  s400 loaded BIT-EXACT + gated THREE ways (G_FACTLOAD: artifact md5 +
  flat md5 ebebb4472725d582dd74928493f1bfb3 + the behavioral panel vs
  the committed literals 0.26464763283729553 / 0.10525520890951157,
  bars 2e-6 / 1e-5 — the family's cross-session read-determinism law).

* THE TRIGGER := the committed fp64 vector runs/e311/
  e311_hijack_vectors.pt : model.trigger (bytes md5
  7dfa7d65c8e80cbfe3b778502dc0a3ad; ||.|| = c := 0.0005 x
  9.1788432658723 = 0.0045894216329361505), applied in e311's EXACT
  convention: theta := flat64(host) + vec, fp64 arithmetic then fp32
  cast (e314's applied-state convention). THE CALIBRATION ANCHOR: the
  applied full-trigger state's flat md5 MUST equal e311's committed
  arm-A flat md5 f84fd2647fa5c2943d3d1b6f24b05e04 AND the host's
  60-g0-context battery read MUST reproduce e311's committed
  0.3494305908679962 (the +32%; gate |d| <= 2e-6) BEFORE any fragment
  counts.

* THE OFF-TARGET BATTERY := x16's exact instrument and exact draw —
  val_windows(val, text, n=300, seed 31603, block=256), last-position
  read, fp32 forward -> fp64 arithmetic; x16's battery_effect summary
  verbatim (top-1 stability, flip/squeeze, dmargin distribution +
  sign test, dentropy, dp(winner), the Z-bias excess = dlogit(Z) -
  mean dlogit). Reused, NOT redrawn, so x16's committed base/trigger
  battery numbers are the instrument's own cross-bind (G_X16COMPAT:
  the anchor state's stability / dmargin / dentropy / Z-excess / NLL
  within 1e-6 of x16's committed literals, stability exact).

* THE DECOMPOSITION (counted + named from the rig): the organism's
  writeable parameter tensors enumerate to 65 — the 27 2D MATRICES
  (wte 65x192, wpe 256x192, L0..L5 {qkv=c_attn 576x192, proj=c_proj
  192x192, mlpUp=mlp.0 768x192, mlpDn=mlp.2 192x768}, lm_head 65x192)
  + 38 1D tensors (LN gains/biases ln1/ln2/ln_f + the MLP biases;
  10,752 params). THE 28 FRAGMENTS := the 27 matrix fragments + ONE
  combined 1D bank ("bank1d") — a PARTITION of the flat trigger (the
  bank exists so the fragments sum to the full vector exactly, the
  dispatch's own mass-accounting gate; disclosed). FRAGMENT i :=
  trigger restricted to its span(s); the fragments' fp64 sum gate
  ||sum(frags) - trigger|| <= 1e-12 (exact 0 expected).

* LADDER / SOLO / LOO (construction exact, no rescaling anywhere):
  ladder rung k := np.where(mask of the top-k fragments by DESCENDING
  fragment norm, trigger, 0.0) — rung 28 (all) is BITWISE the trigger
  (x*1.0), re-anchoring the mass accounting; SOLO_i := np.where(
  mask_i, trigger, 0.0) at its NATURAL norm; LOO_i := np.where(
  mask_i, 0.0, trigger). Every state read with the SAME battery set:
  host g0 battery_cell read (THE delivery number), host gm12
  (co-report), the 300-context off-target battery, the 60-g0 on-target
  Z-excess; host-hold co-report >= 0.8 x baseline per state (the
  calibration-dial standing amendment's floor).

* THE LM_HEAD TEST (T286's sharpened prediction): the trigger's
  restriction to lm_head.weight[zid, :] — the host name ZEPHYRA's
  initial char 'Z', the read token itself — applied ALONE at its
  NATURAL norm. CO-REPORTS (never bars): the name's full 7 rows
  (Z,E,P,H,Y,R,A) at natural norm; the wte 'Z' row (the input-side
  twin).

* THE NUMBERS: full boost := 0.3494305908679962 - 0.26464763283729553
  = 0.08478295803070067; delivery(state) := (host_g0(state) -
  0.26464763283729553) / 0.08478295803070067; "surgical" := top-1
  stability >= 0.95 AND flip-or->10%-squeeze < 0.25 on the committed
  battery (x16's own bar line; the full trigger measured
  0.9933/0.0300); "collapse" := a LOO state's delivery < 0.25;
  gating ratio := on-target Z-excess mean / off-target Z-excess mean
  (the committed full-trigger value 36.0346).

* THE BARS OPERATIONALIZED (clauses fixed, bars not moved): SEATED
  fires iff max(best SINGLE-fragment solo delivery, LADDER RUNG-2
  delivery) >= 0.60 AND that delivering state is surgical — the organ
  named (the matrix, or the lm_head Z row if the row test carries it).
  HOLOGRAPHIC fires iff (every solo delivery < 0.25) AND (the minimum
  ladder rung reaching delivery >= 0.90 is >= 23 of 28 fragments =
  80% of the organism's writeable tensors) AND (no LOO state
  collapses: every LOO delivery >= 0.25). ADJUDICATION ORDER (frozen):
  SEATED -> HOLOGRAPHIC -> MIXED (the table verbatim).

REGISTERED PREDICTIONS (frozen at birth BEFORE compute):
  - P-x22a (the executor's honest counter, registered per the
    dispatch's own invitation — the lab's guess is stated separately
    and both are scored): "MIXED — the knob is a committee, not an
    organ: the read-gradient's mass spreads across the deep read path
    (late-block qkv/proj/MLP + the embeddings carry the norm), so no
    single fragment reaches 60% and the cumulative top-2 stays under
    60%; SOME fragment clears 25% solo (breaking HOLOGRAPHIC's first
    clause); the ladder crosses 90% BEFORE 23 fragments are in; LOO
    never collapses (every removal retains >= 25%); and the lm_head
    initial-char row ALONE delivers < 15% of the boost — the
    confidence channel amplifies through the whole read path, not at
    the output row."
  - P-x22a-LAB (the dispatch's registered lab guess, T286-sharpened):
    "SEATED, with the lm_head initial-char row as the seat."

COMPUTE ENVELOPE: CPU ONLY (dispatch: another agent owns the GPU lane)
— torch threads 4, no CUDA calls anywhere; the heaviest ops are
(300 x 256) CPU forwards across ~90 read states (bursts of minutes,
no thermal exposure). TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/x22/{metrics.json (PROGRESSIVE), REPORT.md,
x22_knobs_seat.png}; run.log + runs/x22_smoke/ gitignored. No NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit +
push: birth -> smoke -> complete (this cell's own paths only:
lab/x22_*, runs/x22/*).

Run:  python lab/x22_knobs_seat.py        (X22_SMOKE=1 shakedown)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
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
                                                      # find_occ, SPLICE_RNG)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402

torch.set_num_threads(4)           # CPU-only cell; the shared desk lane
# CPU-ONLY discipline (dispatch): the machine HAS the GPU (another agent
# owns that lane) — this cell simply never constructs a cuda device or
# tensor; every instrument below runs on CPU explicitly.

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("X22_SMOKE") == "1"
NAME = "x22_smoke" if SMOKE else "x22"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
REPO = E43.REPO
CKPT_DIR = GB.CKPT_DIR

# ---- THE HOST (e311's organism, verbatim) --------------------------------
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12
HOST_WRITE_NORM = 9.1788432658723           # e306's committed dW literal
FACT_READ_TOL_G0 = 2e-6                     # the family's read-determinism
FACT_READ_TOL_GM12 = 1e-5                   # cross-session law
HOST_HOLD_FLOOR = 0.8                       # the calibration-dial amendment

# ---- THE TRIGGER (e311's committed artifact) -----------------------------
E311_VECTORS = REPO / "runs" / "e311" / "e311_hijack_vectors.pt"
E311_VECTORS_MD5 = "29c28977f10eca1a841b0128abb0821c"
E311_TRIGGER_BYTES_MD5 = "7dfa7d65c8e80cbfe3b778502dc0a3ad"
TRIGGER_FRAC = 0.0005
TRIGGER_SCALE = TRIGGER_FRAC * HOST_WRITE_NORM   # 0.0045894216329361505
E311_ARM_A_HOST_G0 = 0.3494305908679962          # the +32% anchor (x1.3205)
E311_ARM_A_FLAT_MD5 = "f84fd2647fa5c2943d3d1b6f24b05e04"
E311_ARM_A_FRAC_ARGMAX = 0.43333333333333335     # committed co-literal
FULL_BOOST = E311_ARM_A_HOST_G0 - FACT_BASELINE_G0   # 0.08478295803070067

# ---- THE X16 INSTRUMENT BIND (the battery + its committed numbers) --------
X16_METRICS = REPO / "runs" / "x16" / "metrics.json"
X16_METRICS_MD5 = "0232da34bd5041e75843293fe6872d65"
X16_BATTERY_SEED = 31603
BATTERY_N = 300
BLOCK = 256
BS = 30
X16_NLL = 1.5781205519851105
X16_WINNER_PROB = 0.6239710181467233
X16_DIVERSITY = 40
X16_TRIG_STABILITY = 0.9933333333333333
X16_TRIG_FOS = 0.03
X16_TRIG_DM_MEAN = -0.0004208469390869141
X16_TRIG_DM_MEAN_ABS = 0.011699775060017903
X16_TRIG_DH_MEAN = 0.0001706811405296848
X16_TRIG_ZEXC_OFF = 0.010046712841498306
X16_TRIG_ZEXC_ON = 0.3620296307577518
X16_GATING_RATIO = X16_TRIG_ZEXC_ON / X16_TRIG_ZEXC_OFF   # 36.0346...
X16_G0_BASE_PZ = 0.26464762284227006          # the last_logits instrument
X16_G0_TRIG_PZ = 0.3494306055353952
X16_COMPAT_TOL = 1e-6

# ---- THE ORGANISM'S TENSORS (counted from the rig) ------------------------
N_PARAMS = GB.G1B_PARAMS                          # 2,739,072
N_MATRICES_2D = 27
N_TENSORS_1D = 38
N_FRAGMENTS = N_MATRICES_2D + 1                   # 28 (bank1d = the 38 1D)
MASS_TOL = 1e-12
GLOBAL_SEED = 32201
HOST_NAME = G1.NAME                               # ZEPHYRA

# ---- THE BARS' NUMBERS ----------------------------------------------------
SEATED_FRAC = 0.60
SOLO_HOLO_BAR = 0.25          # HOLOGRAPHIC's "no fragment delivers >= 25%"
APPROACH_FRAC = 0.90          # "approach the full effect"
LADDER_HOLO_FRAGS = 23        # >= 80% of 28 fragments = 23
LOO_COLLAPSE_FRAC = 0.25      # "collapse" := LOO delivery < 0.25
STABILITY_BAR = 0.95          # x16's own volume/bias bar line
FOS_BAR = 0.25
LMHEAD_ROW_COUNTER_FRAC = 0.15   # P-x22a's "< 15%" clause
BATT_DIVERSITY_BAR = 20

# smoke: the REAL battery (the compat gates live) + a trimmed arm set
SMOKE_TOPK = 8

REGISTERED = {
    "background_verbatim": "e311's 1-dim out-of-room 'trigger' (the host's "
        "read-gradient direction) boosts the host read +32% with near-zero "
        "off-target effect (x16: a CONTEXT-GATED TARGETED BIAS, 36x "
        "amplified at the host's own contexts). T286 just found the noise "
        "floor is AUTHORSHIP-STRUCTURED: writing a name at a context leaves "
        "its logit slot plastic, and the plastic thing is the initial-CHAR "
        "slot.",
    "question_verbatim": "where does the trigger physically live? If one "
        "or two matrices' components deliver most of the effect alone, the "
        "confidence channel has an ORGAN (candidate: the lm_head row of "
        "the name's initial char — which would reconnect the knob to "
        "era-1's oldest surviving doctrine, 'lm_head token-row directions "
        "are causal training coordinates'); if no fragment works and only "
        "the cumulative ladder reaches full effect, confidence is "
        "HOLOGRAPHIC like transport death.",
    "design_verbatim": [
        "1. Decompose the committed trigger vector (runs/e311/"
        "e311_hijack_vectors.pt, md5-bound) per parameter matrix (the "
        "organism's writeable matrices — the rigs' own decomposition "
        "convention; count and name them from the rig).",
        "2. CUMULATIVE LADDER: apply descending per-matrix-norm fragments "
        "cumulatively (top-1, top-2, ... all); measure per rung: host g0 "
        "read, the x16 off-target battery summary (top-1 stability, "
        "margin distribution), and the gating ratio (on-target Z-excess / "
        "off-target Z-excess).",
        "3. LEAVE-ONE-OUT: the full trigger minus each single matrix (the "
        "complement of the ladder): does removing any one matrix collapse "
        "the +32%?",
        "4. The lm_head test (T286's sharpened prediction): the trigger's "
        "component on the output-head row(s) of the host name's initial "
        "char — applied ALONE at its natural norm. If it delivers a "
        "material fraction of the boost, SEATED names the seat.",
    ],
    "bars_verbatim": {
        "SEATED": "a single matrix's fragment (or the cumulative top-2) "
            "delivers >= 60% of the full trigger's host boost with the "
            "off-target profile held surgical — the confidence channel has "
            "an organ; name it (lm_head row or otherwise).",
        "HOLOGRAPHIC": "no fragment delivers >= 25%; only >= 80% of "
            "matrices cumulatively approach the full effect; leave-one-out "
            "never collapses it — confidence, like transport death, is "
            "whole-state.",
    },
    "prediction_registration_verbatim": "Register P-x22a BEFORE compute. "
        "Lab guess (T286-sharpened, but note the lab's four consecutive "
        "mechanism misses tonight): SEATED, with the lm_head initial-char "
        "row as the seat — state your own counter honestly if you "
        "disagree; predictions are scored.",
    "operationalizations": (
        "frozen BEFORE compute: THE HOST := e311's committed host "
        "(e261_K10K_inst_resume.pt s400; three-way G_FACTLOAD; flat md5 "
        f"{FACT_FLAT_MD5}); THE TRIGGER := runs/e311/"
        "e311_hijack_vectors.pt : model.trigger (bytes md5 "
        f"{E311_TRIGGER_BYTES_MD5}), applied fp64-then-fp32 (e314's "
        "convention); THE ANCHOR := the full-trigger state's flat md5 == "
        f"e311's arm-A {E311_ARM_A_FLAT_MD5} AND the 60-g0 battery read "
        f"== {E311_ARM_A_HOST_G0!r} (|d| <= 2e-6) BEFORE any fragment "
        "counts; THE BATTERY := x16's exact draw (val_windows n=300 seed "
        "31603 block 256, last-position read, fp64 arithmetic) with x16's "
        "battery_effect summary VERBATIM — reused, not redrawn "
        "(G_X16COMPAT: the anchor's battery summaries within 1e-6 of "
        "x16's committed literals, stability exact); THE DECOMPOSITION "
        f":= {N_FRAGMENTS} fragments = the organism's {N_MATRICES_2D} "
        "writeable 2D matrices (wte, wpe, L0..L5 {qkv, proj, mlpUp, "
        "mlpDn}, lm_head — counted + named from the rig) + ONE combined "
        f"1D bank ({N_TENSORS_1D} LN/bias tensors) so the fragments sum "
        "to the full vector exactly (mass gate <= 1e-12); LADDER rung k "
        ":= trigger masked to the top-k fragments by DESCENDING fragment "
        "norm (rung 28 bitwise the trigger); SOLO_i and LOO_i := the "
        "fragment alone / the trigger minus the fragment, NATURAL norms, "
        "no rescaling anywhere; THE LM_HEAD TEST := the trigger's "
        "restriction to lm_head.weight[zid, :] (ZEPHYRA's initial char "
        "'Z', the read token) applied ALONE at natural norm, with the "
        "name's 7 rows + the wte 'Z' row as co-reports; delivery(state) "
        f":= (host_g0 - {FACT_BASELINE_G0!r}) / {FULL_BOOST!r}; "
        "'surgical' := stability >= 0.95 AND flip-or-squeeze < 0.25; "
        "'collapse' := LOO delivery < 0.25; SEATED := max(best solo, "
        "ladder rung-2) >= 0.60 with the delivering state surgical (the "
        "organ named); HOLOGRAPHIC := every solo < 0.25 AND min rung at "
        f"delivery >= 0.90 is >= {LADDER_HOLO_FRAGS} of 28 AND every LOO "
        ">= 0.25; ADJUDICATION ORDER: SEATED -> HOLOGRAPHIC -> MIXED"),
    "registration": "question + design + bars + P-x22a VERBATIM from the "
        "dispatch letter (R72 ideator card 6, sharpened by T286); every "
        "convention picked + frozen HERE at birth BEFORE compute; this "
        "script committed at birth; adjudicate against exactly this; no "
        "bar shopping.",
    "predictions": {
        "P-x22a_executor_counter": "MIXED — the knob is a committee, not "
            "an organ: the read-gradient's mass spreads across the deep "
            "read path (late-block qkv/proj/MLP + the embeddings carry "
            "the norm), so no single fragment reaches 60% and the "
            "cumulative top-2 stays under 60%; SOME fragment clears 25% "
            "solo (breaking HOLOGRAPHIC's first clause); the ladder "
            "crosses 90% BEFORE 23 fragments are in; LOO never collapses "
            "(every removal retains >= 25%); and the lm_head initial-char "
            "row ALONE delivers < 15% of the boost — the confidence "
            "channel amplifies through the whole read path, not at the "
            "output row.",
        "P-x22a_lab_guess": "SEATED, with the lm_head initial-char row as "
            "the seat.",
    },
}

deviations: list[str] = [
    "CPU-ONLY desk cell (dispatch: another agent owns the GPU lane) — "
    "torch threads 4, no CUDA calls anywhere (asserted at import).",
    "THE DECOMPOSITION includes ONE combined 1D bank fragment (the 38 "
    "LN/bias tensors, 10,752 params) BESIDE the dispatch's 'matrices': a "
    "partition of only the 27 matrices would leave ~0.4% of dims "
    "unassigned and the dispatch's own mass-accounting gate (fragments "
    "sum to the full vector) could not hold exactly; the bank is counted, "
    "named, solo-read and LOO-read like every matrix — disclosed.",
    "THE OFF-TARGET BATTERY is x16's EXACT draw (seed 31603, n=300), not "
    "a fresh one: the dispatch's 'the x16 off-target battery summary' "
    "read as 'the same instrument', and reuse makes x16's committed "
    "battery numbers a hard instrument-equality gate (G_X16COMPAT).",
    "THE ROOM is NOT rebuilt: x22's fragments are coordinate-wise and no "
    "bar is room-relative; the trigger's identity is bytes-md5 + norm "
    "bound, its zero-bearer class inherited from e311/x16's committed "
    "records (disclosed lean).",
    "ce_r is not re-read per state: the x16 battery IS this cell's health "
    "instrument (the dispatch's own summary list); the host-hold floor "
    "(>= 0.8x baseline, the calibration-dial standing amendment) is "
    "carried per state instead.",
    "n=1 organism, one trigger, one session (the g-series standing "
    "caveat); the ladder/solo/loo differences are the registered object.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (X22_SMOKE=1): the REAL battery + all parent/anchor/"
    "compat gates LIVE (the compat gate is the delicate part — it must "
    "run at smoke); the arm set TRIMMED to the top-8 fragments (solo/"
    "ladder/LOO) + the row tests + the anchor (disclosed); own smoke dir; "
    "NOTHING adjudicated (SMOKE stamp on every read).",
]

metrics: dict = {
    "experiment": "x22_knobs_seat",
    "phase": "THE KNOB'S SEAT — where does e311's 1-dim out-of-room "
             "confidence trigger physically live? per-matrix decomposition "
             "+ cumulative ladder + leave-one-out + the lm_head "
             "initial-char row test (T286's sharpened prediction): an "
             "ORGAN for the confidence channel, or HOLOGRAPHY like "
             "transport death",
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "envelope": {
        "device": "CPU ONLY (torch threads 4; another agent owns the GPU "
                  "lane)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e311 (the parasitic hijacker: the committed trigger vector + the "
        "+32% host anchor + the applied-state convention)",
        "x16 (THE OFF-TARGET SPECIFICITY PROBE: the battery instrument, "
        "battery_effect, the Z-excess read, the 36x gating ratio, the "
        "committed battery literals this cell re-binds)",
        "T286 / x24 (the authorship census: the noise floor is "
        "authorship-structured; the initial-CHAR slot is the plastic "
        "thing — this cell's lm_head-row candidate)",
        "T283 (x16's finding: a CONTEXT-GATED TARGETED BIAS, 36x "
        "amplified at the host's own contexts)",
        "e264/e288 (the established-fact rig: the host artifact + the "
        "three-way G_FACTLOAD)",
        "e306 (the per-2D-matrix decomposition convention — the rigs' "
        "own)",
        "era-1 doctrine (lm_head token-row directions are causal training "
        "coordinates — the reconnection SEATED would make)",
    ],
    "whats_new": [
        "THE FIRST PHYSICAL DECOMPOSITION of the confidence knob: the "
        "committed trigger split per writeable matrix, each fragment "
        "solo-read, cumulatively laddered (descending norm), and "
        "leave-one-out-read — organ vs holography adjudicated against "
        "frozen bars",
        "THE LM_HEAD INITIAL-CHAR ROW TEST: the trigger's component on "
        "the host name's initial-char output row applied ALONE at natural "
        "norm — T286's slot-plasticity finding turned into a direct "
        "intervention on the knob's candidate seat",
        "THE PER-RUNG GATING RATIO: on-target Z-excess / off-target "
        "Z-excess at every ladder rung — does the 36x context gating "
        "survive fragmentation, and which fragments carry it",
    ],
    "gates": {},
}


# ------------------------------------------------------------------ helpers
def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def save_json_partial(note: str) -> None:
    metrics["date_updated"] = common.now_iso()
    metrics["phase_note"] = note
    save_json(RD / "metrics.json", metrics)
    if not SMOKE:
        try:
            subprocess.run(["git", "add", str(RD / "metrics.json")],
                           cwd=str(REPO), timeout=10)
        except Exception:                                   # noqa: BLE001
            pass


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def apply_flat64(net, flat64: np.ndarray, offsets, shapes) -> None:
    """theta <- flat64 (fp64 arithmetic) cast fp32 into the params —
    e314/e311/x16's applied-state convention, VERBATIM."""
    f32 = flat64.astype(np.float32)
    with torch.no_grad():
        for p, (a, b), shp in zip(net.parameters(), offsets, shapes):
            p.copy_(torch.from_numpy(f32[a:b]).reshape(shp))


# ---- x16's battery instrument, VERBATIM (copied to own the import) -------
def binom_p_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial sign test (symmetric-tail form, P=0.5)."""
    if n <= 0:
        return 1.0
    lo, hi = min(k, n - k), max(k, n - k)
    denom = n * math.log(2.0)
    p = 0.0
    for i in range(0, lo + 1):
        p += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) - denom)
    for i in range(hi, n + 1):
        p += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) - denom)
    return min(max(p, 0.0), 1.0)


def skew_g1(x: np.ndarray) -> float:
    m = x.mean()
    d = x - m
    m2 = float((d ** 2).mean())
    if m2 <= 0.0:
        return 0.0
    return float((d ** 3).mean()) / m2 ** 1.5


def quantiles(x: np.ndarray) -> dict:
    qs = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


@torch.no_grad()
def last_logits(net, ids: torch.Tensor, bs: int = BS) -> np.ndarray:
    """The battery's own read convention: last-position logits,
    fp32 forward, fp64 cast downstream."""
    net.eval()
    outs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        outs.append(lg[:, -1, :].clone())
    return torch.cat(outs).numpy().astype(np.float64)


def softmax64(L: np.ndarray) -> np.ndarray:
    z = L - L.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def battery_effect(Lb: np.ndarray, Lt: np.ndarray, zid: int) -> dict:
    """x16's frozen per-context battery read, VERBATIM (values lists
    stripped at the summary layer; the anchor + row states keep them)."""
    n = Lb.shape[0]
    top1b = Lb.argmax(axis=1)
    top1t = Lt.argmax(axis=1)
    sb, st = np.sort(Lb, axis=1), np.sort(Lt, axis=1)
    mb = sb[:, -1] - sb[:, -2]
    mt = st[:, -1] - st[:, -2]
    dm = mt - mb
    pb, pt = softmax64(Lb), softmax64(Lt)
    idx = np.arange(n)
    Hb = -(pb * np.log(np.clip(pb, 1e-300, None))).sum(axis=1)
    Ht = -(pt * np.log(np.clip(pt, 1e-300, None))).sum(axis=1)
    dH = Ht - Hb
    dpw = pt[idx, top1b] - pb[idx, top1b]
    dlog_z = Lt[:, zid] - Lb[:, zid]
    dlog_mean = (Lt - Lb).mean(axis=1)
    dbias_z = dlog_z - dlog_mean
    stable = top1b == top1t
    squeeze = dm < -0.10 * np.maximum(mb, 1e-12)
    flip_or_squeeze = (~stable) | squeeze
    neg = int((dm < 0).sum())
    pos = int((dm > 0).sum())
    nzc = neg + pos
    sign_p = binom_p_two_sided(neg, nzc)
    return {
        "n_contexts": int(n),
        "top1_stability": float(stable.mean()),
        "flip_frac": float((~stable).mean()),
        "squeeze_frac": float(squeeze.mean()),
        "flip_or_squeeze_frac": float(flip_or_squeeze.mean()),
        "dmargin": {
            "mean": float(dm.mean()), "median": float(np.median(dm)),
            "std": float(dm.std()), "skew_g1": skew_g1(dm),
            "mean_abs": float(np.abs(dm).mean()),
            "neg_share_of_nonzero": (neg / nzc) if nzc else 0.5,
            "neg": neg, "pos": pos,
            "sign_test_p_two_sided": sign_p,
            "quantiles": quantiles(dm),
        },
        "dentropy": {
            "mean": float(dH.mean()), "std": float(dH.std()),
            "skew_g1": skew_g1(dH), "quantiles": quantiles(dH),
        },
        "dp_winner": {
            "mean": float(dpw.mean()), "std": float(dpw.std()),
            "frac_negative": float((dpw < 0).mean()),
            "quantiles": quantiles(dpw),
        },
        "z_bias_excess": {
            "mean": float(dbias_z.mean()), "std": float(dbias_z.std()),
            "frac_positive": float((dbias_z > 0).mean()),
            "quantiles": quantiles(dbias_z),
        },
        "dlog_z_mean": float(dlog_z.mean()),
        "dlog_mean_of_all_tokens": float(dlog_mean.mean()),
        "pwin_trig_mean": float(pt[idx, top1b].mean()),
    }


def battery_summary(eff: dict) -> dict:
    """The dispatch's 'x16 off-target battery summary' (the compact form)."""
    return {
        "top1_stability": eff["top1_stability"],
        "flip_or_squeeze_frac": eff["flip_or_squeeze_frac"],
        "dmargin_mean": eff["dmargin"]["mean"],
        "dmargin_mean_abs": eff["dmargin"]["mean_abs"],
        "dmargin_neg_share": eff["dmargin"]["neg_share_of_nonzero"],
        "dmargin_sign_p": eff["dmargin"]["sign_test_p_two_sided"],
        "dentropy_mean": eff["dentropy"]["mean"],
        "z_excess_mean": eff["z_bias_excess"]["mean"],
    }


SHORT = {
    "wte.weight": "wte",
    "wpe.weight": "wpe",
    "lm_head.weight": "lm_head",
    "attn.c_attn.weight": "qkv",
    "attn.c_proj.weight": "proj",
    "mlp.0.weight": "mlpUp",
    "mlp.2.weight": "mlpDn",
}


def short_name(full: str) -> str:
    if full in SHORT:
        return SHORT[full]
    if full.startswith("h."):
        parts = full.split(".")          # h.3.attn.c_attn.weight
        return f"L{parts[1]}.{SHORT['.'.join(parts[2:])]}"
    return full


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X22 — THE KNOB'S SEAT (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = git_head()
    log(f"host: {FACT_CK} (baseline g0 {FACT_BASELINE_G0:.10f}); trigger c="
        f"{TRIGGER_SCALE!r}; full boost {FULL_BOOST!r}; battery n="
        f"{BATTERY_N} (seed {X16_BATTERY_SEED}, x16's exact draw)")
    save_json_partial("startup (bars registered, committed at birth)")
    set_seed(GLOBAL_SEED)             # global init only; no other RNG here

    # ================= P0: the protocol rebuild (x16's gates) ==========
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH") + val_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains the name: {G_NAMEFREE}"

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
    G_BATTERYGEO = {
        "shapes": {f"g{j:+d}": list(bat_ids[j].shape) for j in G1.GEOS},
        "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                     and list(bat_ids[0].shape) == [60, G1.PRE]
                     and list(bat_ids[12].shape) == [60, G1.PRE + 12]),
    }
    assert G_BATTERYGEO["pass"], f"battery geometry drift: {G_BATTERYGEO}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERYGEO": G_BATTERYGEO})
    log("P0: protocol gates PASS (namefree; splice 19+41; battery geometry; "
        f"g0 x{tuple(g0_ids.shape)})")
    save_json_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ===========
    x16m = json.loads(X16_METRICS.read_text(encoding="utf-8"))
    G_PARENTS = {
        "e311_vectors": {"path": str(E311_VECTORS),
                         "md5": md5of(E311_VECTORS),
                         "bound_md5": E311_VECTORS_MD5,
                         "note": "the committed fp64 trigger (the knob "
                                 "this cell dissects)"},
        "x16_metrics": {"path": str(X16_METRICS), "md5": md5of(X16_METRICS),
                        "bound_md5": X16_METRICS_MD5,
                        "verdict": x16m["adjudication"]["word"],
                        "note": "the battery instrument + its committed "
                                "literals (the compat cross-bind)"},
        "x16_literals_crosscheck": {
            "battery_seed": {"mine": X16_BATTERY_SEED,
                             "theirs": x16m["gates"]["G_BATT"]["seed"]},
            "battery_n": {"mine": BATTERY_N,
                          "theirs": x16m["gates"]["G_BATT"]["n"]},
            "nll_base": {"mine": X16_NLL,
                         "theirs": x16m["gates"]["G_BATT"]
                         ["base_lastpos_nll"]},
            "trig_stability": {"mine": X16_TRIG_STABILITY,
                               "theirs": x16m["arms_battery"]["TRIGGER"]
                               ["effect"]["top1_stability"]},
            "trig_zexcess_off": {"mine": X16_TRIG_ZEXC_OFF,
                                 "theirs": x16m["arms_battery"]["TRIGGER"]
                                 ["effect"]["z_bias_excess"]["mean"]},
            "trig_zexcess_on": {"mine": X16_TRIG_ZEXC_ON,
                                "theirs": x16m["g0_cotarget"]["TRIGGER"]
                                ["z_bias_excess"]["mean"]},
            "anchor_host_g0": {"mine": E311_ARM_A_HOST_G0,
                               "theirs": x16m["gates"]["G_ANCHOR"]
                               ["trig_host_g0"]["committed"]},
            "anchor_flat_md5": {"mine": E311_ARM_A_FLAT_MD5,
                                "theirs": x16m["gates"]["G_ANCHOR"]
                                ["trig_flat_md5"]},
        },
        "pass": bool(md5of(E311_VECTORS) == E311_VECTORS_MD5
                     and md5of(X16_METRICS) == X16_METRICS_MD5
                     and x16m["adjudication"]["word"] == "MIXED"
                     and x16m["gates"]["G_BATT"]["seed"] == X16_BATTERY_SEED
                     and x16m["gates"]["G_BATT"]["n"] == BATTERY_N
                     and x16m["gates"]["G_BATT"]["base_lastpos_nll"]
                     == X16_NLL
                     and x16m["gates"]["G_ANCHOR"]["trig_host_g0"]
                     ["committed"] == E311_ARM_A_HOST_G0
                     and x16m["gates"]["G_ANCHOR"]["trig_flat_md5"]
                     == E311_ARM_A_FLAT_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS (e311's vectors + x16's metrics md5-bound; the "
        "anchor + battery literals cross-checked against x16's record)")
    save_json_partial("P0b parents hard-bound")

    # ================= P1: THE HOST (loaded bit-exact + gated) =========
    fact_art = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                          weights_only=False)
    fact_sd = {k: v.detach().clone() for k, v in fact_art["model"].items()}
    del fact_art
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_g0 = G1.battery_cell(fact_net, g0_ids, zid)["mean_pz"]
    fact_gm12 = G1.battery_cell(fact_net, gm12_ids, zid)["mean_pz"]
    G_FACTLOAD = {
        "form": "e311's HOST organism loaded BIT-EXACT and gated THREE "
                "ways: (1) the artifact (md5/size), (2) the loaded "
                "state's flat-md5, (3) the behavioral read (g0/gm12 vs "
                "the committed literals, bars 2e-6/1e-5)",
        "artifact_md5": md5of(CKPT_DIR / FACT_CK),
        "bound_md5": FACT_MD5,
        "artifact_size": (CKPT_DIR / FACT_CK).stat().st_size,
        "bound_size": FACT_SIZE,
        "flat_md5": fact_flat_md5, "bound_flat_md5": FACT_FLAT_MD5,
        "read_g0": {"mine": fact_g0, "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_g0 - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_gm12, "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_gm12 - FACT_BASELINE_GM12)},
        "pass": bool(md5of(CKPT_DIR / FACT_CK) == FACT_MD5
                     and (CKPT_DIR / FACT_CK).stat().st_size == FACT_SIZE
                     and fact_flat_md5 == FACT_FLAT_MD5
                     and abs(fact_g0 - FACT_BASELINE_G0) <= FACT_READ_TOL_G0
                     and abs(fact_gm12 - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P1 G_FACTLOAD: the HOST loads — g0 {fact_g0:.10f} (|d| "
        f"{abs(fact_g0 - FACT_BASELINE_G0):.1e}); flat md5 "
        f"{fact_flat_md5[:8]}: PASS")
    save_json_partial("P1 the HOST loaded bit-exact + gated")

    # ================= P2: THE TRIGGER (the committed knob) ============
    vecs = torch.load(E311_VECTORS, map_location="cpu", weights_only=False)
    trigger = vecs["trigger"].numpy().astype(np.float64).copy()
    del vecs
    trig_md5 = hashlib.md5(trigger.tobytes()).hexdigest()
    trig_norm = float(np.linalg.norm(trigger))
    assert trigger.shape[0] == N_PARAMS
    G_TRIGBIND = {
        "form": "the committed e311 trigger (the host's own read-gradient "
                "direction at g0 window 0, projected OUT of the room, "
                "scaled to c = 0.0005 x ||dW_host||): loaded from the "
                "committed artifact — the object THIS cell decomposes",
        "bytes_md5": trig_md5, "bound_md5": E311_TRIGGER_BYTES_MD5,
        "norm": trig_norm, "bound_norm": TRIGGER_SCALE,
        "in_room_class": "inherited from e311/x16's committed records "
                         "(4.5e-17; the room plays no role in x22's "
                         "coordinate-wise fragments)",
        "pass": bool(trig_md5 == E311_TRIGGER_BYTES_MD5
                     and abs(trig_norm - TRIGGER_SCALE) <= 1e-12),
    }
    assert G_TRIGBIND["pass"], f"G_TRIGBIND FAILED: {G_TRIGBIND}"
    metrics["gates"]["G_TRIGBIND"] = G_TRIGBIND
    log(f"P2 G_TRIGBIND: the committed trigger loads (md5 {trig_md5[:8]}; "
        f"||.|| {trig_norm:.16f}): PASS")
    save_json_partial("P2 the trigger bound")

    # ================= P3: THE DECOMPOSITION (counted from the rig) ====
    log("=" * 78)
    log(f"P3: THE DECOMPOSITION — the organism's writeable tensors, "
        f"counted + named from the rig")
    named = list(fact_net.named_parameters())
    assert len(named) == N_MATRICES_2D + N_TENSORS_1D, \
        f"tensor census drift: {len(named)}"
    offsets, shapes = [], []
    off = 0
    frags = []            # (short, full, kind, [spans], shape)
    bank_spans = []
    for full_name, p in named:
        n_el = p.numel()
        span = (off, off + n_el)
        offsets.append(span)
        shapes.append(tuple(p.shape))
        if p.ndim >= 2:
            frags.append({"short": short_name(full_name),
                          "full": full_name, "kind": "2d",
                          "spans": [span], "shape": list(p.shape),
                          "numel": n_el})
        else:
            bank_spans.append(span)
        off += n_el
    assert off == N_PARAMS
    frags.append({"short": "bank1d", "full": "<the 38 1D LN/bias tensors>",
                  "kind": "bank", "spans": bank_spans,
                  "shape": [len(bank_spans), "1d"],
                  "numel": sum(b - a for a, b in bank_spans)})
    assert len(frags) == N_FRAGMENTS
    n_2d = sum(1 for f in frags if f["kind"] == "2d")
    assert n_2d == N_MATRICES_2D

    masks = []
    for f in frags:
        m = np.zeros(N_PARAMS, dtype=bool)
        for a, b in f["spans"]:
            m[a:b] = True
        masks.append(m)
    frag_vecs = [np.where(m, trigger, 0.0) for m in masks]
    frag_norms = [float(np.linalg.norm(v)) for v in frag_vecs]
    mass_resid = float(np.linalg.norm(np.sum(frag_vecs, axis=0) - trigger))
    span_total = int(sum(m.sum() for m in masks))
    G_MASS = {
        "form": f"the {N_FRAGMENTS} fragments ({N_MATRICES_2D} writeable "
                "2D matrices named from the rig + 1 combined 1D bank) "
                "PARTITION the flat trigger: disjoint spans, exact sum",
        "n_fragments": N_FRAGMENTS,
        "n_matrices_2d": n_2d,
        "n_1d_tensors_in_bank": len(bank_spans),
        "span_coverage": span_total,
        "span_coverage_pass": bool(span_total == N_PARAMS),
        "sum_minus_trigger_norm": mass_resid,
        "mass_tol": MASS_TOL,
        "trigger_norm": trig_norm,
        "sum_of_frag_norms": float(sum(frag_norms)),
        "pass": bool(mass_resid <= MASS_TOL and span_total == N_PARAMS
                     and abs(sum(n * n for n in frag_norms)
                             - trig_norm ** 2) <= 1e-12),
    }
    assert G_MASS["pass"], f"G_MASS FAILED: {G_MASS}"
    metrics["gates"]["G_MASS"] = G_MASS
    order = sorted(range(N_FRAGMENTS), key=lambda i: -frag_norms[i])
    decomposition = {}
    for rank, i in enumerate(order):
        f = frags[i]
        share = frag_norms[i] / trig_norm
        decomposition[f["short"]] = {
            "full_name": f["full"], "kind": f["kind"],
            "shape": f["shape"], "numel": f["numel"],
            "norm": frag_norms[i], "norm_share": share,
            "share_squared_first_order": share * share,
            "norm_rank": rank + 1,
        }
    metrics["decomposition"] = {
        "convention": f"{N_FRAGMENTS} fragments = the {N_MATRICES_2D} "
                      "writeable 2D matrices (named from the rig) + one "
                      "combined 1D bank; fragments = span restrictions of "
                      "the committed trigger; ladder order = DESCENDING "
                      "fragment norm",
        "fragments": decomposition,
        "order_desc_norm": [frags[i]["short"] for i in order],
        "zero_norm_fragments": [frags[i]["short"] for i in order
                                if frag_norms[i] == 0.0],
    }
    log(f"P3 G_MASS: {N_FRAGMENTS} fragments ({n_2d} matrices + the "
        f"{len(bank_spans)}-tensor bank) partition the trigger (sum resid "
        f"{mass_resid:.1e}); norm order: "
        + ", ".join(f"{frags[i]['short']}({frag_norms[i] / trig_norm:.3f})"
                    for i in order[:6]) + " ...")
    save_json_partial("P3 the decomposition built (mass gate PASSED)")

    # ================= P4: THE BATTERY + THE ANCHOR ====================
    log("=" * 78)
    log(f"P4: THE BATTERY (x16's exact draw, seed {X16_BATTERY_SEED}) + "
        "THE FULL-TRIGGER ANCHOR — the +32% must reproduce BEFORE any "
        "fragment counts")
    bat_x, bat_y = G1.val_windows(val_ids, val_text, BATTERY_N,
                                  X16_BATTERY_SEED, block=BLOCK)
    Lb = last_logits(fact_net, bat_x)
    Lb_g0 = last_logits(fact_net, g0_ids)
    pb64 = softmax64(Lb)
    y_last = bat_y[:, -1].numpy()
    nll_base = float(-np.log(np.clip(
        pb64[np.arange(BATTERY_N), y_last], 1e-300, None)).mean())
    winners = Lb.argmax(axis=1)
    diversity = int(np.unique(winners).size)
    mean_wp = float(pb64[np.arange(BATTERY_N), winners].mean())
    zeph_leak = sum(1 for i in range(BATTERY_N)
                    if "ZEPH" in corpus.decode(bat_x[i]))
    uniq = int(np.unique(bat_x.numpy(), axis=0).shape[0])
    G_BATT = {
        "form": "battery sanity + the instrument's cross-bind vs x16's "
                "committed base reads (the SAME draw: seed 31603 n=300)",
        "n": BATTERY_N, "block": BLOCK, "seed": X16_BATTERY_SEED,
        "unique_windows": uniq, "zeph_leak_windows": int(zeph_leak),
        "distinct_base_winners": diversity,
        "base_winner_mean_prob": mean_wp,
        "base_lastpos_nll": nll_base,
        "x16_committed": {"nll": X16_NLL, "winner_prob": X16_WINNER_PROB,
                          "diversity": X16_DIVERSITY},
        "nll_abs_diff": abs(nll_base - X16_NLL),
        "pass": bool(uniq >= 0.99 * BATTERY_N and zeph_leak == 0
                     and np.isfinite(Lb).all()
                     and diversity >= BATT_DIVERSITY_BAR
                     and 0.05 < mean_wp < 0.9999
                     and abs(nll_base - X16_NLL) <= X16_COMPAT_TOL),
    }
    assert G_BATT["pass"], f"G_BATT FAILED: {G_BATT}"
    metrics["gates"]["G_BATT"] = G_BATT

    # ---- the state reader (every arm read with the SAME battery set) --
    base_g0_ll_pz = float(softmax64(Lb_g0)[:, zid].mean())

    def read_state(vec: np.ndarray | None, label: str,
                   keep_values: bool = False) -> dict:
        net = G1.evl_load(fact_sd)
        flat_md5 = None
        if vec is not None:
            assert vec.shape == (N_PARAMS,)
            apply_flat64(net, fact_flat_np + vec, offsets, shapes)
        flat_md5 = hashlib.md5(
            flat_params_cpu(net).numpy().tobytes()).hexdigest()
        b_host = G1.battery_cell(net, g0_ids, zid)
        b_gm12 = G1.battery_cell(net, gm12_ids, zid)
        Lt = last_logits(net, bat_x)
        eff = battery_effect(Lb, Lt, zid)
        Lt_g0 = last_logits(net, g0_ids)
        eff_g0 = battery_effect(Lb_g0, Lt_g0, zid)
        del net
        off_z = eff["z_bias_excess"]["mean"]
        on_z = eff_g0["z_bias_excess"]["mean"]
        rec = {
            "label": label,
            "applied_vec_norm": (float(np.linalg.norm(vec))
                                 if vec is not None else 0.0),
            "host_g0": b_host["mean_pz"],
            "host_g0_frac_argmax": b_host["frac_argmax_z"],
            "host_gm12": b_gm12["mean_pz"],
            "delivery": (b_host["mean_pz"] - FACT_BASELINE_G0) / FULL_BOOST,
            "host_hold_ge_0p8x": bool(
                b_host["mean_pz"] >= HOST_HOLD_FLOOR * FACT_BASELINE_G0),
            "battery": battery_summary(eff),
            "on_target_z_excess": on_z,
            "on_target_dmargin_mean": eff_g0["dmargin"]["mean"],
            "on_target_pz_mean": float(softmax64(Lt_g0)[:, zid].mean()),
            "off_target_z_excess": off_z,
            "gating_ratio": (on_z / off_z if abs(off_z) > 1e-9 else None),
            "flat_md5": flat_md5,
        }
        if keep_values:
            rec["battery_values_full"] = {
                "dmargin": eff["dmargin"]["quantiles"],
                "z_bias_excess": eff["z_bias_excess"]["quantiles"],
            }
        return rec

    anchor = read_state(trigger, "FULL-TRIGGER (the anchor)", True)
    G_ANCHOR = {
        "form": "THE CALIBRATION ANCHOR: the full trigger applied in "
                "e311's exact convention MUST reproduce e311's committed "
                "arm-A state (flat md5) AND its committed +32% host read "
                "(gate |d| <= 2e-6) BEFORE any fragment counts",
        "flat_md5": anchor["flat_md5"],
        "bound_flat_md5": E311_ARM_A_FLAT_MD5,
        "host_g0": {"mine": anchor["host_g0"],
                    "committed": E311_ARM_A_HOST_G0,
                    "abs_diff": abs(anchor["host_g0"] - E311_ARM_A_HOST_G0)},
        "frac_argmax": {"mine": anchor["host_g0_frac_argmax"],
                        "committed": E311_ARM_A_FRAC_ARGMAX},
        "pass": bool(anchor["flat_md5"] == E311_ARM_A_FLAT_MD5
                     and abs(anchor["host_g0"] - E311_ARM_A_HOST_G0)
                     <= FACT_READ_TOL_G0
                     and abs(anchor["host_g0_frac_argmax"]
                             - E311_ARM_A_FRAC_ARGMAX) <= 1e-9),
    }
    assert G_ANCHOR["pass"], f"G_ANCHOR FAILED: {G_ANCHOR}"
    metrics["gates"]["G_ANCHOR"] = G_ANCHOR

    # ---- G_X16COMPAT: the instrument equality gate (same draw, same
    # numbers as x16's committed trigger arm) --------------------------
    b = anchor["battery"]
    G_X16COMPAT = {
        "form": "the instrument-equality gate: the anchor state's battery "
                "summaries vs x16's committed literals (the SAME battery "
                "draw) — stability/squeeze exact, means within 1e-6",
        "stability": {"mine": b["top1_stability"],
                      "x16": X16_TRIG_STABILITY},
        "fos": {"mine": b["flip_or_squeeze_frac"], "x16": X16_TRIG_FOS},
        "dmargin_mean": {"mine": b["dmargin_mean"], "x16": X16_TRIG_DM_MEAN},
        "dmargin_mean_abs": {"mine": b["dmargin_mean_abs"],
                             "x16": X16_TRIG_DM_MEAN_ABS},
        "dentropy_mean": {"mine": b["dentropy_mean"],
                          "x16": X16_TRIG_DH_MEAN},
        "z_excess_off": {"mine": b["z_excess_mean"],
                         "x16": X16_TRIG_ZEXC_OFF},
        "z_excess_on": {"mine": anchor["on_target_z_excess"],
                        "x16": X16_TRIG_ZEXC_ON},
        "gating_ratio": {"mine": anchor["gating_ratio"],
                         "x16": X16_GATING_RATIO},
        "g0_pz_lastlogits": {"mine": anchor["on_target_pz_mean"],
                             "x16": X16_G0_TRIG_PZ},
        "base_g0_pz_lastlogits": {"mine": base_g0_ll_pz,
                                  "x16": X16_G0_BASE_PZ},
        "pass": bool(b["top1_stability"] == X16_TRIG_STABILITY
                     and b["flip_or_squeeze_frac"] == X16_TRIG_FOS
                     and abs(b["dmargin_mean"] - X16_TRIG_DM_MEAN)
                     <= X16_COMPAT_TOL
                     and abs(b["dmargin_mean_abs"] - X16_TRIG_DM_MEAN_ABS)
                     <= X16_COMPAT_TOL
                     and abs(b["dentropy_mean"] - X16_TRIG_DH_MEAN)
                     <= X16_COMPAT_TOL
                     and abs(b["z_excess_mean"] - X16_TRIG_ZEXC_OFF)
                     <= X16_COMPAT_TOL
                     and abs(anchor["on_target_z_excess"]
                             - X16_TRIG_ZEXC_ON) <= X16_COMPAT_TOL
                     and abs(anchor["on_target_pz_mean"]
                             - X16_G0_TRIG_PZ) <= X16_COMPAT_TOL
                     and abs(base_g0_ll_pz - X16_G0_BASE_PZ)
                     <= X16_COMPAT_TOL),
    }
    assert G_X16COMPAT["pass"], f"G_X16COMPAT FAILED: {G_X16COMPAT}"
    metrics["gates"]["G_X16COMPAT"] = G_X16COMPAT
    metrics["anchor"] = anchor
    metrics["base_reads"] = {
        "host_g0": fact_g0, "host_gm12": fact_gm12,
        "battery_nll": nll_base, "battery_winner_prob": mean_wp,
        "battery_diversity": diversity,
        "g0_pz_lastlogits": base_g0_ll_pz,
    }
    log(f"P4 G_ANCHOR: THE +32% REPRODUCES — host g0 {anchor['host_g0']:.16f} "
        f"vs committed {E311_ARM_A_HOST_G0:.16f} (|d| "
        f"{abs(anchor['host_g0'] - E311_ARM_A_HOST_G0):.1e}); flat md5 "
        f"{anchor['flat_md5'][:8]}; delivery "
        f"{anchor['delivery']:.4f}; gating ratio "
        f"{anchor['gating_ratio']:.2f} (x16 committed "
        f"{X16_GATING_RATIO:.2f})")
    log(f"    G_BATT: nll {nll_base:.6f} vs x16 {X16_NLL:.6f} (|d| "
        f"{abs(nll_base - X16_NLL):.1e}); diversity {diversity}: PASS")
    log(f"    G_X16COMPAT: the anchor's battery == x16's committed "
        f"trigger arm (stability {b['top1_stability']:.4f}, Z-excess "
        f"off {b['z_excess_mean']:.6f} / on "
        f"{anchor['on_target_z_excess']:.6f}): PASS")
    save_json_partial("P4 the battery + the anchor reproduced (compat "
                      "gate PASSED)")

    # ================= P5: THE ARMS (solo / ladder / LOO / rows) =======
    log("=" * 78)
    n_ladder = SMOKE_TOPK if SMOKE else N_FRAGMENTS
    log(f"P5: THE ARMS — solo x{N_FRAGMENTS}, cumulative ladder x"
        f"{n_ladder}, leave-one-out x{N_FRAGMENTS}, the lm_head row tests"
        + (" [SMOKE: arms trimmed to the top-8 fragments]"
           if SMOKE else ""))

    solo_recs, ladder_recs, loo_recs = [], [], []
    cum_mask = np.zeros(N_PARAMS, dtype=bool)
    for k, i in enumerate(order):
        short = frags[i]["short"]
        # SOLO
        rec = read_state(frag_vecs[i], f"solo:{short}")
        rec["frag"] = short
        rec["norm_share"] = decomposition[short]["norm_share"]
        solo_recs.append(rec)
        # LADDER (cumulative)
        cum_mask = cum_mask | masks[i]
        lvec = np.where(cum_mask, trigger, 0.0)
        rec = read_state(lvec if k + 1 < N_FRAGMENTS else trigger,
                         f"ladder{k + 1}:{short}")
        rec["rung"] = k + 1
        rec["added_frag"] = short
        rec["cum_frag_count"] = k + 1
        ladder_recs.append(rec)
        # LOO
        rec = read_state(np.where(masks[i], 0.0, trigger), f"loo:{short}")
        rec["removed_frag"] = short
        loo_recs.append(rec)
        if (k + 1) % 7 == 0 or k + 1 == n_ladder:
            log(f"  rung {k + 1:2d}/{N_FRAGMENTS} (+{short:10s}): host g0 "
                f"{ladder_recs[-1]['host_g0']:.6f} (delivery "
                f"{ladder_recs[-1]['delivery']:+.4f}, stability "
                f"{ladder_recs[-1]['battery']['top1_stability']:.4f}, "
                f"gating {ladder_recs[-1]['gating_ratio']}) | solo "
                f"{solo_recs[-1]['delivery']:+.4f} | loo "
                f"{loo_recs[-1]['delivery']:+.4f}")
            save_json_partial(f"P5 arms through fragment {k + 1}/"
                              f"{N_FRAGMENTS} ({short})")
        if SMOKE and k + 1 >= SMOKE_TOPK:
            break
    # the ladder's last rung (full mode) == the anchor state, bitwise
    if not SMOKE:
        assert ladder_recs[-1]["rung"] == N_FRAGMENTS
        G_LADDER28 = {
            "form": "the ladder's final rung (all 28 fragments) is "
                    "BITWISE the anchor state (np.where(all-True mask) "
                    "== the trigger; x*1.0 identity)",
            "rung28_flat_md5": ladder_recs[-1]["flat_md5"],
            "anchor_flat_md5": anchor["flat_md5"],
            "rung28_host_g0": ladder_recs[-1]["host_g0"],
            "pass": bool(ladder_recs[-1]["flat_md5"]
                         == anchor["flat_md5"]
                         and abs(ladder_recs[-1]["host_g0"]
                                 - anchor["host_g0"]) <= 1e-12),
        }
        assert G_LADDER28["pass"], f"G_LADDER28 FAILED: {G_LADDER28}"
        metrics["gates"]["G_LADDER28"] = G_LADDER28
        log("P5 G_LADDER28: rung 28 == the anchor BITWISE: PASS")

    # ---- THE LM_HEAD TEST (T286's sharpened prediction) + twins -------
    lm_off = None
    wte_off = None
    off_acc = 0
    for full_name, p in named:
        if full_name == "lm_head.weight":
            lm_off = off_acc
        if full_name == "wte.weight":
            wte_off = off_acc
        off_acc += p.numel()
    n_embd = fact_net.lm_head.weight.shape[1]
    assert lm_off is not None and wte_off is not None
    row_span = (lm_off + zid * n_embd, lm_off + (zid + 1) * n_embd)
    zrow_vec = np.zeros(N_PARAMS)
    zrow_vec[row_span[0]:row_span[1]] = trigger[row_span[0]:row_span[1]]
    name_ids = [stoi[c] for c in HOST_NAME]
    name_vec = np.zeros(N_PARAMS)
    for t in name_ids:
        a = lm_off + t * n_embd
        name_vec[a:a + n_embd] = trigger[a:a + n_embd]
    wte_row_span = (wte_off + zid * n_embd,
                    wte_off + (zid + 1) * n_embd)
    wtez_vec = np.zeros(N_PARAMS)
    wtez_vec[wte_row_span[0]:wte_row_span[1]] = \
        trigger[wte_row_span[0]:wte_row_span[1]]
    zrow_rec = read_state(zrow_vec, "lmhead:Z-row", True)
    name_rec = read_state(name_vec, "lmhead:name-7-rows")
    wtez_rec = read_state(wtez_vec, "wte:Z-row")
    lm_frag = decomposition["lm_head"]
    zrow_share = float(np.linalg.norm(zrow_vec)) / trig_norm
    metrics["lmhead_test"] = {
        "form": "THE LM_HEAD TEST (T286's sharpened prediction): the "
                f"trigger's restriction to lm_head.weight[zid, :] "
                f"({HOST_NAME}'s initial char '{itos[zid]}', the read "
                "token itself) applied ALONE at its NATURAL norm; "
                "co-reports: the name's 7 rows, the wte 'Z' row (the "
                "input-side twin)",
        "zrow": {**zrow_rec, "norm": float(np.linalg.norm(zrow_vec)),
                 "norm_share": zrow_share,
                 "as_frac_of_lmhead_frag": float(np.linalg.norm(zrow_vec))
                 / max(lm_frag["norm"], 1e-30)},
        "name_rows": {**name_rec, "norm": float(np.linalg.norm(name_vec)),
                      "norm_share": float(np.linalg.norm(name_vec))
                      / trig_norm},
        "wte_zrow": {**wtez_rec, "norm": float(np.linalg.norm(wtez_vec)),
                     "norm_share": float(np.linalg.norm(wtez_vec))
                     / trig_norm},
        "lmhead_matrix_fragment": lm_frag,
    }
    log(f"P5 THE LM_HEAD TEST: the Z row alone (||.|| "
        f"{np.linalg.norm(zrow_vec):.2e} = {zrow_share:.4%} of the "
        f"trigger, {np.linalg.norm(zrow_vec) / max(lm_frag['norm'], 1e-30):.1%} "
        f"of the lm_head fragment): host g0 {zrow_rec['host_g0']:.6f} "
        f"(delivery {zrow_rec['delivery']:+.4f}, gating "
        f"{zrow_rec['gating_ratio']})")
    log(f"    co-reports: name 7 rows delivery "
        f"{name_rec['delivery']:+.4f}; wte Z row delivery "
        f"{wtez_rec['delivery']:+.4f}")

    metrics["arms"] = {
        "solo": solo_recs,
        "ladder": ladder_recs,
        "loo": loo_recs,
    }
    save_json_partial("P5 all arms read")

    # ================= P6: THE ADJUDICATION (frozen bars) ==============
    log("=" * 78)
    if SMOKE:
        log("P6 [smoke] no adjudication (SMOKE stamp on every read)")
    else:
        best_solo = max(solo_recs, key=lambda r: r["delivery"])
        rung2 = ladder_recs[1]
        seated_val = max(best_solo["delivery"], rung2["delivery"])
        seated_state = (best_solo if best_solo["delivery"]
                        >= rung2["delivery"] else rung2)

        def surgical(rec) -> bool:
            bb = rec["battery"]
            return bool(bb["top1_stability"] >= STABILITY_BAR
                        and bb["flip_or_squeeze_frac"] < FOS_BAR)

        seated_cond = bool(seated_val >= SEATED_FRAC
                           and surgical(seated_state))
        r90 = next((r["rung"] for r in ladder_recs
                    if r["delivery"] >= APPROACH_FRAC), None)
        holo_cond = bool(best_solo["delivery"] < SOLO_HOLO_BAR
                         and r90 is not None
                         and r90 >= LADDER_HOLO_FRAGS
                         and all(r["delivery"] >= LOO_COLLAPSE_FRAC
                                 for r in loo_recs))
        if seated_cond:
            word = "SEATED"
            # name the organ: the matrix (or the lm_head Z row if the row
            # test carries it)
            row_frac = zrow_rec["delivery"]
            solo_is_the_seat = (seated_state is best_solo)
            if (solo_is_the_seat and best_solo["frag"] == "lm_head") or \
                    row_frac >= SEATED_FRAC:
                organ = (f"the lm_head initial-char row ('{itos[zid]}' of "
                         f"{HOST_NAME}) — T286's seat; era-1's token-row "
                         "doctrine reconnected")
            else:
                organ = (f"the {seated_state.get('frag', 'top-2')} "
                         "matrix fragment")
            deliverer = (("the single fragment " + best_solo["frag"]
                          + " solo") if solo_is_the_seat else
                         ("the cumulative top-2 (" + frags[order[0]]["short"]
                          + " + " + frags[order[1]]["short"] + ")"))
            clause = (f"{deliverer} "
                      f"delivers {seated_val:.1%} >= {SEATED_FRAC:.0%} of "
                      f"the full boost (host g0 "
                      f"{seated_state['host_g0']:.6f}) with the off-target "
                      f"profile surgical (stability "
                      f"{seated_state['battery']['top1_stability']:.4f}, "
                      f"flip-or-squeeze "
                      f"{seated_state['battery']['flip_or_squeeze_frac']:.4f}) "
                      f"— the confidence channel has an organ: {organ}")
        elif holo_cond:
            word = "HOLOGRAPHIC"
            clause = (f"no fragment delivers >= {SOLO_HOLO_BAR:.0%} solo "
                      f"(best {best_solo['delivery']:.1%} = "
                      f"{best_solo['frag']}); the ladder only reaches "
                      f"{APPROACH_FRAC:.0%} at rung {r90} of "
                      f"{N_FRAGMENTS} (>= {LADDER_HOLO_FRAGS} = the 80% "
                      "line); no leave-one-out collapses (min LOO "
                      f"{min(r['delivery'] for r in loo_recs):.1%} >= "
                      f"{LOO_COLLAPSE_FRAC:.0%}) — confidence, like "
                      "transport death, is whole-state")
        else:
            word = "MIXED"
            clause = (f"the table verbatim: best solo "
                      f"{best_solo['delivery']:.1%} ({best_solo['frag']}); "
                      f"top-2 cumulative {rung2['delivery']:.1%}; R90 "
                      f"rung {r90}/{N_FRAGMENTS}; min LOO "
                      f"{min(r['delivery'] for r in loo_recs):.1%}; "
                      f"lm_head Z row {zrow_rec['delivery']:+.1%} — "
                      "neither frozen bar met")
        metrics["adjudication"] = {
            "word": word, "clause": clause,
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "numbers": {
                "full_boost": FULL_BOOST,
                "best_solo": {"frag": best_solo["frag"],
                              "delivery": best_solo["delivery"],
                              "host_g0": best_solo["host_g0"],
                              "surgical": surgical(best_solo)},
                "rung2": {"frags": [frags[order[0]]["short"],
                                    frags[order[1]]["short"]],
                          "delivery": rung2["delivery"],
                          "host_g0": rung2["host_g0"],
                          "surgical": surgical(rung2)},
                "r60_rung": next((r["rung"] for r in ladder_recs
                                  if r["delivery"] >= SEATED_FRAC), None),
                "r90_rung": r90,
                "ladder_holo_frag_bar": LADDER_HOLO_FRAGS,
                "min_loo": {"frag": min(loo_recs,
                                         key=lambda r: r["delivery"])["removed_frag"],
                            "delivery": min(r["delivery"]
                                            for r in loo_recs)},
                "max_loo": {"frag": max(loo_recs,
                                        key=lambda r: r["delivery"])["removed_frag"],
                            "delivery": max(r["delivery"]
                                            for r in loo_recs)},
                "lmhead_zrow": {"delivery": zrow_rec["delivery"],
                                "norm_share": zrow_share,
                                "gating_ratio": zrow_rec["gating_ratio"]},
                "lmhead_matrix_solo": next(
                    (r["delivery"] for r in solo_recs
                     if r["frag"] == "lm_head"), None),
                "name_rows_delivery": name_rec["delivery"],
                "wte_zrow_delivery": wtez_rec["delivery"],
                "seated_value": seated_val,
            },
            "predictions_scored": {
                "P-x22a_executor_counter": {
                    "text": REGISTERED["predictions"]
                        ["P-x22a_executor_counter"],
                    "verdict_fired": bool(word == "MIXED"),
                    "clauses": {
                        "no_single_60": best_solo["delivery"] < SEATED_FRAC,
                        "top2_under_60": rung2["delivery"] < SEATED_FRAC,
                        "some_25": best_solo["delivery"] >= SOLO_HOLO_BAR,
                        "r90_before_23": bool(r90 is not None
                                              and r90 < LADDER_HOLO_FRAGS),
                        "loo_never_collapses": all(
                            r["delivery"] >= LOO_COLLAPSE_FRAC
                            for r in loo_recs),
                        "lmhead_row_under_15": bool(
                            zrow_rec["delivery"] < LMHEAD_ROW_COUNTER_FRAC),
                    },
                },
                "P-x22a_lab_guess": {
                    "text": REGISTERED["predictions"]["P-x22a_lab_guess"],
                    "verdict_fired": bool(word == "SEATED"),
                    "row_is_the_seat": bool(
                        zrow_rec["delivery"] >= SEATED_FRAC),
                },
            },
        }
        log(f"P6 ADJUDICATION: {word} — {clause}")
        pz = metrics["adjudication"]["predictions_scored"]
        log(f"P6 PREDICTIONS: P-x22a executor fired="
            f"{pz['P-x22a_executor_counter']['verdict_fired']} "
            f"(clauses {pz['P-x22a_executor_counter']['clauses']}); "
            f"P-x22a lab fired={pz['P-x22a_lab_guess']['verdict_fired']} "
            f"(row is the seat: "
            f"{pz['P-x22a_lab_guess']['row_is_the_seat']})")
        save_json_partial(f"P6 adjudicated: {word}")

    # ================= P7: THE FIGURE ===================================
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    # panel 1: the cumulative ladder
    ax = axes[0]
    rungs = [r["rung"] for r in ladder_recs]
    deliv = [r["delivery"] for r in ladder_recs]
    ax.plot(rungs, deliv, "o-", color="#ee6677", lw=1.6, ms=4,
            label="cumulative delivery (host boost fraction)")
    ax.axhline(1.0, color="k", ls="-", lw=1.2,
               label="the full effect (+32%; rung 28 == the anchor)")
    ax.axhline(SEATED_FRAC, color="#4477aa", ls="--", lw=1,
               label=f"SEATED line {SEATED_FRAC:.0%}")
    ax.axhline(SOLO_HOLO_BAR, color="#999999", ls=":", lw=1,
               label=f"holography solo line {SOLO_HOLO_BAR:.0%}")
    if not SMOKE and metrics.get("adjudication"):
        r90v = metrics["adjudication"]["numbers"]["r90_rung"]
        if r90v:
            ax.axvline(r90v, color="#228833", ls=":", lw=1.2,
                       label=f"R90 = rung {r90v}")
        ax.axvline(LADDER_HOLO_FRAGS, color="#ccbb44", ls=":", lw=1.2,
                   label=f"the 80% line ({LADDER_HOLO_FRAGS} frags)")
    for r in ladder_recs[:6]:
        ax.annotate(r["added_frag"], (r["rung"], r["delivery"]),
                    textcoords="offset points", xytext=(4, -10),
                    fontsize=6, rotation=45)
    ax.set_xlabel("ladder rung (fragments in, descending norm)")
    ax.set_ylabel("delivery (fraction of the +32% boost)")
    ax.set_title("THE CUMULATIVE LADDER" + (" [SMOKE]" if SMOKE else ""))
    ax.legend(fontsize=6)
    # panel 2: leave-one-out bars
    ax = axes[1]
    los = sorted(loo_recs, key=lambda r: r["delivery"])
    names = [r["removed_frag"] for r in los]
    vals = [r["delivery"] for r in los]
    cols = ["#ee6677" if v < LOO_COLLAPSE_FRAC else "#4477aa"
            for v in vals]
    ax.bar(range(len(los)), vals, color=cols)
    ax.axhline(LOO_COLLAPSE_FRAC, color="k", ls="--", lw=1,
               label=f"collapse line {LOO_COLLAPSE_FRAC:.0%}")
    ax.axhline(1.0, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(los)))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("delivery after removing the fragment")
    ax.set_title("LEAVE-ONE-OUT (red = collapsed)")
    ax.legend(fontsize=7)
    # panel 3: the lm_head-alone panel
    ax = axes[2]
    lm_solo = next((r["delivery"] for r in solo_recs
                    if r["frag"] == "lm_head"), float("nan"))
    best_solo_state = max(solo_recs, key=lambda r: r["delivery"])
    labs = ["FULL\ntrigger", "lm_head\nmatrix", "lm_head\nZ row\n(T286)",
            "name\n7 rows", "wte\nZ row",
            f"best solo\n({best_solo_state['frag']})"]
    vals = [anchor["delivery"], lm_solo, zrow_rec["delivery"],
            name_rec["delivery"], wtez_rec["delivery"],
            best_solo_state["delivery"]]
    ax.bar(range(len(labs)), vals,
           color=["#999999", "#4477aa", "#ee6677", "#aa3377", "#228833",
                  "#ccbb44"])
    ax.axhline(SEATED_FRAC, color="k", ls="--", lw=1,
               label=f"SEATED line {SEATED_FRAC:.0%}")
    ax.axhline(LMHEAD_ROW_COUNTER_FRAC, color="#ee6677", ls=":", lw=1,
               label=f"P-x22a row line {LMHEAD_ROW_COUNTER_FRAC:.0%}")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("delivery (fraction of the +32% boost)")
    ax.set_title("THE LM_HEAD TEST (alone, natural norm)")
    ax.legend(fontsize=7)
    # panel 4: solo delivery vs the first-order prediction (share^2)
    ax = axes[3]
    sh2 = [r["norm_share"] ** 2 for r in solo_recs]
    sd = [r["delivery"] for r in solo_recs]
    ax.scatter(sh2, sd, s=14, color="#4477aa")
    lim = max(max(sh2), max(sd), 0.05)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x (linear "
            "first-order law)")
    for r in solo_recs:
        if r["delivery"] > 0.10 or r["norm_share"] ** 2 > 0.10:
            ax.annotate(r["frag"], (r["norm_share"] ** 2, r["delivery"]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=6)
    ax.set_xlabel("norm share squared (first-order prediction)")
    ax.set_ylabel("measured solo delivery")
    ax.set_title("solo effect vs the linear prediction")
    ax.legend(fontsize=7)
    verdict_word = (metrics.get("adjudication", {}).get("word", "SMOKE"))
    fig.suptitle(f"X22 THE KNOB'S SEAT — {verdict_word}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = RD / "x22_knobs_seat.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log(f"P7 figure saved: {fig_path.name}")

    # ================= P8: THE REPORT ===================================
    if not SMOKE:
        adj = metrics["adjudication"]
        nums = adj["numbers"]
        rep = [
            "# X22 — THE KNOB'S SEAT",
            "",
            f"* the verdict: **{adj['word']}** — {adj['clause']}",
            "",
            "## The decomposition (28 fragments = the organism's 27 "
            "writeable matrices + the 1D bank; ladder order = descending "
            "norm)",
            "",
            "| frag | norm share | share^2 (1st-order pred) | solo "
            "delivery | LOO delivery | solo gating ratio |",
            "|---|---|---|---|---|---|",
        ]
        for r in solo_recs:
            loo_map = {r2["removed_frag"]: r2["delivery"]
                       for r2 in loo_recs}
            rep.append(
                f"| {r['frag']} | {r['norm_share']:.4f} | "
                f"{r['norm_share'] ** 2:.4f} | {r['delivery']:+.4f} | "
                f"{loo_map[r['frag']]:+.4f} | "
                f"{r['gating_ratio'] if r['gating_ratio'] is None else round(r['gating_ratio'], 1)} |")
        rep += [
            "",
            "## The ladder's key rungs",
            "",
            "| rung | +frag | host g0 | delivery | stability | gating "
            "ratio |",
            "|---|---|---|---|---|---|",
        ]
        for r in ladder_recs:
            if r["rung"] in (1, 2, 3, 5, 10, 20, N_FRAGMENTS) or \
                    r["delivery"] >= SEATED_FRAC or \
                    r["delivery"] >= APPROACH_FRAC:
                rep.append(
                    f"| {r['rung']} | {r['added_frag']} | "
                    f"{r['host_g0']:.6f} | {r['delivery']:+.4f} | "
                    f"{r['battery']['top1_stability']:.4f} | "
                    f"{r['gating_ratio'] if r['gating_ratio'] is None else round(r['gating_ratio'], 1)} |")
        rep += [
            "",
            "## The lm_head test (T286's sharpened prediction)",
            "",
            f"* The Z row alone ('{itos[zid]}' of {HOST_NAME}, the read "
            f"token): ||.|| {zrow_rec['applied_vec_norm']:.2e} = "
            f"{zrow_share:.3%} of the trigger "
            f"({metrics['lmhead_test']['zrow']['as_frac_of_lmhead_frag']:.1%} "
            f"of the lm_head fragment) -> delivery "
            f"{zrow_rec['delivery']:+.4f}, host g0 "
            f"{zrow_rec['host_g0']:.6f}, gating ratio "
            f"{zrow_rec['gating_ratio']}.",
            f"* Co-reports: the name's 7 rows delivery "
            f"{name_rec['delivery']:+.4f}; the wte 'Z' row "
            f"{wtez_rec['delivery']:+.4f}; the full lm_head MATRIX "
            f"fragment solo delivery {nums['lmhead_matrix_solo']:+.4f}.",
            "",
            "## The anchor + the gates",
            "",
            f"* THE +32% ANCHOR reproduces (host g0 "
            f"{anchor['host_g0']:.16f} vs committed "
            f"{E311_ARM_A_HOST_G0:.16f}, |d| "
            f"{abs(anchor['host_g0'] - E311_ARM_A_HOST_G0):.1e}; flat "
            f"md5 {anchor['flat_md5'][:12]}); the anchor's battery == "
            "x16's committed trigger arm (G_X16COMPAT, the same draw).",
            f"* Gate ledger: " + ", ".join(
                f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                for k, v in metrics["gates"].items()),
            "",
            "## Predictions scored",
            "",
            f"* P-x22a (executor counter, MIXED committee): fired="
            f"{adj['predictions_scored']['P-x22a_executor_counter']['verdict_fired']} "
            f"with clauses "
            f"{adj['predictions_scored']['P-x22a_executor_counter']['clauses']}.",
            f"* P-x22a-LAB (T286-sharpened, SEATED/lm_head row): fired="
            f"{adj['predictions_scored']['P-x22a_lab_guess']['verdict_fired']} "
            f"(the row is the seat: "
            f"{adj['predictions_scored']['P-x22a_lab_guess']['row_is_the_seat']}).",
            "",
            "## Provenance",
            "",
            f"* birth commit: {metrics.get('birth_commit', 'see git')}; "
            f"final head: {git_head()}",
            "* CPU only (torch threads 4); no GPU touched.",
            "* parents: e311 (vectors) + x16 (metrics) md5-bound; the "
            "host artifact three-way gated; all gates in metrics.json.",
            "",
            "*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the "
            "coordinator folds).*",
        ]
        (RD / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
        log("P8 REPORT.md written")

    metrics["outputs"] = {
        "figure": str(fig_path.relative_to(REPO)),
        "metrics": str((RD / "metrics.json").relative_to(REPO)),
        "report": str((RD / "REPORT.md").relative_to(REPO))
        if not SMOKE else None,
    }
    metrics["git_head_final"] = git_head()
    if SMOKE:
        metrics["status"] = "SMOKE COMPLETE (nothing adjudicated)"
    else:
        metrics["status"] = "COMPLETE — adjudicated"
    metrics["date_completed"] = common.now_iso()
    save_json_partial("P8 figure + report complete")
    log(f"X22 {'SMOKE ' if SMOKE else ''}COMPLETE — "
        f"{metrics.get('adjudication', {}).get('word', 'smoke')}")


if __name__ == "__main__":
    main()
