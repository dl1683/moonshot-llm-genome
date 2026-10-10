# -*- coding: utf-8 -*-
"""W055 — THE RELAY TUNER'S FIRST PROBE (R79's minted card; W049's dial
question graduated by x41; the period-1 weak-dosing regime's stability).

Run:  cd lab && python w055_relay_probe.py     (W055_SMOKE=1 shakedown)

THE BACKGROUND (verbatim from the dispatch letter):
    x41 (runs/x41/) proved the 2-cycle ENTRAINABLE — the period re-locks
    to the event spacing, the amplitude tracks the lr cap — and its LRCAP
    leg's break pattern (ZERO, DOSE, MIXED, MIXED) suggested the halved
    cap pushes toward PERIOD-1 WEAK DOSING, the tuner's candidate
    intermediate regime. W049's dial question graduates: is the period-1
    weak regime STABLE (a usable setpoint), or does it re-bifurcate or
    decay? THE INHERITED AMENDMENT: every read must be PHASE-DECLARED per
    the doc's METHODS 4 (the milestone-costume lesson: a sampler locked
    to the phenomenon's phase shows the phase, not the phenomenon) — this
    cell reads EVERY event, both phases.

THE QUESTION (verbatim): "W049's dial question graduates: is the
    period-1 weak regime STABLE (a usable setpoint), or does it
    re-bifurcate or decay?"

THE DESIGN (verbatim from the dispatch letter):
    1. THE RESUME: x41's committed LRCAP leg's final state (md5-bound;
       the state where MIXED appeared).
    2. THE CONTINUATION: 8 more events at the SAME halved cap (the
       regime held constant — the stability question, not another
       perturbation), reading EVERY event's pre/post pair
       (phase-declared: the full cycle, not milestones).
    3. THE OBSERVABLES: the dose pattern (does period-1 weak dosing
       persist — every event a small dose?); the read's rails (the
       floor/upper trajectory); aliveness (>= 0.05 throughout?); the
       off-target battery at the end (the standing amendment's
       collateral check).

THE BARS (frozen VERBATIM from the dispatch letter, BEFORE any compute):
    STABLE-PERIOD-1: the 8 events all dose weakly (no ZERO events; lr_m
        bounded away from 0 and the cap) and the read stays alive
        (>= 0.05) with a stable floor — THE INTERMEDIATE REGIME IS A
        USABLE SETPOINT; W055's tuner has its third knob-setting
        (2-cycle / period-1 / off); the calibration program's kinetics
        complete.
    RE-BIFURCATION: a new 2-cycle forms at the halved cap (ZERO events
        return) — the relay's bifurcation is cap-dependent; the period-1
        regime was transient.
    RAIL-DECAY: the read's floor decays through aliveness (< 0.05) — the
        halved cap under-doses; the regime is not viable; the tuner's
        floor is named.
    Register P-W055a BEFORE compute. Lab lean: STABLE-PERIOD-1, weakly —
        x41's MIXED pattern suggested convergence, and the relay's
        self-limiting logic (dose on deficit) supports a stable
        weak-dosing equilibrium; the countervailing: the read's recovery
        dynamics at the halved dose may be too slow to hold the floor.
        State your own read.

MACHINERY: x41's harness BY IMPORT — e336's COMMITTED driver
(chunked_errorgated_phase) EXECUTED VERBATIM under the family's
disclosed re-binding convention: the ONLY module globals re-bound are
the leg parameters (PHASE_STEPS, MILESTONES, WIN1, WIN2, MAINT_EVERY)
+ e261.INST_TOTAL (THE SCHEDULE EXTENSION, below) + this leg's lr_m_max
ARGUMENT (== x41's LRCAP leg's halved cap, byte-equal — the regime held
constant); every other protocol constant (LR_STABLE,
LR_M_MAX_FROZEN, seeds, baseline, momentum, room, budget fractions) is
read from the committed module and asserted against x41's md5-bound
metrics BEFORE any compute. The base is x41's committed LRCAP leg's
final state (runs/x41/x41_LRCAP_REA_resume.pt: the model + BOTH SGD-M
buffer states + BOTH generator states at t900 — the state where MIXED
appeared; the post mirror bit-compared) — staged into this cell with the
leg pools zeroed (the family's staging convention, digest-verified).
The stream CONTINUES (the LRCAP branch's own generator states restored,
never re-seeded): corpus steps 901.. and events 37.. draw the
continuation of that branch's own sequences.

THE CONTINUATION LEG (frozen): t901..t1100 (200 steps), M=25 UNCHANGED,
lr_m_max = 0.5 x LR_M_MAX_FROZEN = 0.010971232006621194 (THE SAME
halved cap — byte-equal to x41's LRCAP leg's), 8 events at
t925/950/975/1000/1025/1050/1075/1100 (the LRCAP branch's GLOBAL events
37-44; the staging zeroes n_maint, the family convention).

THE POOL CONVENTION (frozen, x41's verbatim): the leg's budget pools
SCALED TO THE LEG'S OWN LENGTH — B_C = 2.6848887180471723 x 200/400,
B_M = 1.7899258120314483 x 8/16 — the PER-STEP and PER-EVENT
equal-share reservations are then IDENTICAL to e339's committed leg-2
(and to x41's legs; asserted via the applied-lr ledger). LR_M_MAX
itself is NEVER scaled.

THE SCHEDULE EXTENSION (a birth decision, frozen and disclosed): the
house cosine (warmup 100, total E261.INST_TOTAL=1000) clamps to ZERO at
t >= 1000 — a verbatim continuation would silently KILL the corpus wash
for this leg's last four events (a protocol change smuggled in as decay:
RE-BIFURCATION-by-schedule, not by the relay). REJECTED. Instead
E261.INST_TOTAL is re-bound 1000 -> 1250 (the ONLY other module global
re-bound), chosen so the corpus schedule stays >= ~2.5x the family's
maximum observed budget cap through t1100 — the cap (the length-scaled
pool share over b_norm) remains the ONLY binding corpus constraint, as
it was at EVERY logged step of e339's leg-2 and BOTH of x41's cap-bound
legs (the schedule NEVER bound in any committed phase). Asserted at
every logged step (G_SCHED). This re-binding changes NO committed
behavior; it holds the regime's operative variable (the cap-bound wash)
constant — the stability question, not another perturbation.

PHASE-DECLARED READS (METHODS 4, frozen): EVERY event read at BOTH
phases — the PRE read = the controller's own gate read (battery p(T) at
the event instant, pre-dose, from the maint ledger) + the POST read =
the traj milestone at the same step (measured AFTER the event's dose;
MILESTONES = ALL 8 event steps) — plus the window ledger's intra-cycle
flanks (t924/t926, t1099) and pre/post brackets at the first and last
events. The aliveness floor is read over EVERY declared read. A
milestone-only cadence (the costume x41's rider caught on e339) is
impossible by construction here.

PHASE CLASSIFICATION (frozen, x41's verbatim): ZERO-phase event :=
deficit_t == 0.0 (the gate read at/above the baseline — lr_m == 0
exactly, the dose self-limits); DOSE-phase event := deficit_t >= 0.40;
MIXED := anything else (the establishment regime's small doses).

THE BARS OPERATIONALIZED (frozen at birth, in order):
    STABLE-PERIOD-1 := (a) NO ZERO events (all 8 deficits > 0) AND
        (b) every event's dose INTERIOR — 0.10 <= lr_m / lr_m_max_halved
        <= 0.90 (bounded away from 0 and the cap by >= 10% margins) AND
        (c) ALIVE — every declared read >= 0.05 AND (d) STABLE FLOOR —
        the 8 gate reads' range <= 0.08 AND mean(last 4) >= mean(first
        4) - 0.08 (0.08 = the family's PHASE_MARGIN, ~2.4x the
        committed intra-phase spread 0.033).
    RE-BIFURCATION := >= 2 ZERO events AND at least one ZERO event
        immediately followed by a dosing event (the period-2 signature).
    RAIL-DECAY := any declared read < 0.05 (the floor pierces
        aliveness).
    COMPOSITE := STABLE-PERIOD-1 (all four clauses) -> RE-BIFURCATION
        -> RAIL-DECAY -> MIXED (the residue, disclosed). The three bars
        are mutually exclusive by construction (ZERO events exclude (a);
        a sub-0.05 read forces deficits ~0.83 -> doses near the cap,
        excluding (b)); the precedence is frozen anyway.

P-W055a (registered BEFORE compute): see REGISTERED below.
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

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402

import e336_founding_replicate as E336                 # noqa: E402 — THE
                                                      # COMMITTED DRIVER

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SMOKE = os.environ.get("W055_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "w055_smoke" if SMOKE else "w055"
if not SMOKE:
    assert torch.cuda.is_available(), "w055 owns the GPU lane (dispatch)"
assert common.gpu_ok(), "gpu_ok() gate at startup (check BEFORE launch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE RE-BINDING (the family's disclosed convention): the committed
# modules' logs + burst machinery label THIS cell (tags w055:...); nothing
# of e336's is written (its run.log handle is re-pointed here).
E336._logf = _logf
E336.T0 = T0
E336.NAME = NAME
device_events: list[dict] = []
thermal_log: list[dict] = []
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
E261.thermal_log = thermal_log
E261.device_events = device_events

# ======================================================================
# THE CONFIG (frozen) — every protocol constant BOUND to the committed
# modules + x41's md5-bound record (asserted at runtime)
# ======================================================================
X41_DIR = E43.REPO / "runs" / "x41"
X41_LRCAP_RESUME_CK = X41_DIR / "x41_LRCAP_REA_resume.pt"
X41_LRCAP_POST_CK = X41_DIR / "x41_LRCAP_REA_post.pt"
X41_METRICS = X41_DIR / "metrics.json"

# x41's committed artifacts — raw md5s, VERIFIED AT BIRTH (2026-10-10,
# git head 9f68f48); a drift FAILS the resume gate. The resume ck is the
# LRCAP leg's FULL final state at t900 (model + both SGD-M buffers +
# both generator states — the state where MIXED appeared); the post ck
# is the model-only mirror.
X41_LRCAP_RESUME_MD5 = "aa2a57373f90dd2868881b259ffc9b1a"
X41_LRCAP_POST_MD5 = "248fc9f14c03f068305b84bc11c8258b"
X41_METRICS_MD5 = "63c2c4eb23802dc4576e5c78cb14cda1"

# x41's committed LRCAP-leg literals (the repro targets + the regime's
# committed signature; read from the md5-bound record at runtime and
# asserted == these): (global_event, step, gate_read, deficit, lr_m,
# post_read)
X41_LRCAP_CYCLE = [
    (33, 825, 0.33782604336738586, 0.0, 0.0, 0.33782604336738586),
    (34, 850, 0.14292903244495392, 0.4999883230929291,
     0.005485487893254003, 0.6403168439865112),
    (35, 875, 0.19037844240665436, 0.3339950421103674,
     0.0036643370960540564, 0.5482984781265259),
    (36, 900, 0.174992173910141, 0.38782115273801054,
     0.004254875843763988, 0.5949878096580505),
]
X41_LRCAP_CLASSES = ["ZERO", "DOSE", "MIXED", "MIXED"]
X41_LRCAP_S_CORPUS = 0.6712221801280975
X41_LRCAP_S_MAINT = 0.11064264364540577
X41_T900_POST_READ = 0.5949878096580505    # the LRCAP leg's final read
# x41's committed LRCAP rails (display + the comparison key)
X41_FLOOR_RAIL = 0.33782604336738586       # the ZERO-phase gate (event 33)
X41_UPPER_RAIL = 0.6403168439865112        # the biggest post-dose read
X41_WEAK_RAIL = 0.14292903244495392        # the DOSE-phase gate (event 34)
# e339's committed 2-cycle rails (the regime W055's branch departed)
E339_FLOOR_RAIL = 0.3734504304
E339_UPPER_RAIL = 0.8800075054
E339_WEAK_RAIL = 0.1244877942
# the corpus-lr protocol-identity band (x41's frozen LR_BAND: e339's
# committed leg-2 range widened 20%)
LR_BAND = (0.0018, 0.0043)

# ---- the base organism class (recorded per the dispatch's gate) -------
NET0_CLASS = {
    "name": "the lineage's char-transformer base (e001.pt)",
    "checkpoint": "runs/checkpoints/e001.pt",
    "params": 2739072,
    "family": "the 2.74M pre-LN char transformer (e311's TAVIREN organism "
              "built on it; e336/e339's controller + x41's legs ran on "
              "that organism; this cell continues x41's LRCAP branch)",
}

# ---- the continuation leg (frozen) -------------------------------------
B_C_400 = 2.6848887180471723          # e339's committed per-400-step pools
B_M_400 = 1.7899258120314483
LR_M_MAX_FROZEN = E336.LR_M_MAX_FROZEN   # 0.021942464013242388 (asserted)
LR_M_MAX_HALF = 0.5 * LR_M_MAX_FROZEN    # x41's LRCAP cap (asserted equal)
BASE_STEP = 900                       # x41's LRCAP leg's final step
CONT_STEPS = 200                      # 8 events x 25 steps
CONT_M = 25                           # UNCHANGED (the regime constant)
END_STEP = BASE_STEP + CONT_STEPS     # 1100
GLOBAL_EVENT_BASE = 36                # x41's LRCAP events ended at global 36

SMOKE_BASE_STEPS = 12                 # the smoke base (fresh, tiny)
SMOKE_BASE_M = 2

LEG_STEPS = 8 if SMOKE else CONT_STEPS
LEG_M = 2 if SMOKE else CONT_M
LEG_START = (SMOKE_BASE_STEPS if SMOKE else BASE_STEP)
LEG_END = LEG_START + LEG_STEPS

# ---- the bars' numbers (frozen) ----------------------------------------
ALIVE_FLOOR = 0.05        # the aliveness line (the RAIL-DECAY bar)
DOSE_DEFICIT_MIN = 0.40   # the DOSE-phase classification floor (x41's)
INTERIOR_LO = 0.10        # the dose-interior bounds (fractions of the
INTERIOR_HI = 0.90        # halved cap — "bounded away from 0 and the cap")
FLOOR_BAND = 0.08         # the stable-floor band (the family's PHASE_MARGIN)
STREAM_STABLE_TOL = 1.05
ORTH_BAR = 1e-6
BUFSEP_ORTH_BAR = 1e-4
BUDGET_SLACK = 1e-5
SCHED_EXT_TOTAL = 1250    # THE SCHEDULE EXTENSION (disclosed above)
SCHED_MARGIN_MIN = 2.0    # asserted: lr_sched >= 2x cap at every logged step

REGISTERED = {
    "registration": (
        "question + design + bars + lab lean frozen VERBATIM from the "
        "dispatch letter (R79's minted card — W055 THE RELAY TUNER'S FIRST "
        "PROBE; W049's graduation); this script committed at birth BEFORE "
        "any compute; P-W055a registered below BEFORE compute; adjudicate "
        "against exactly this; no bar shopping."),
    "question_verbatim": (
        "W049's dial question graduates: is the period-1 weak regime "
        "STABLE (a usable setpoint), or does it re-bifurcate or decay?"),
    "design_verbatim": (
        "1. THE RESUME: x41's committed LRCAP leg's final state (md5-bound; "
        "the state where MIXED appeared). 2. THE CONTINUATION: 8 more "
        "events at the SAME halved cap (the regime held constant — the "
        "stability question, not another perturbation), reading EVERY "
        "event's pre/post pair (phase-declared: the full cycle, not "
        "milestones). 3. THE OBSERVABLES: the dose pattern (does period-1 "
        "weak dosing persist — every event a small dose?); the read's "
        "rails (the floor/upper trajectory); aliveness (>= 0.05 "
        "throughout?); the off-target battery at the end (the standing "
        "amendment's collateral check)."),
    "bars_verbatim": {
        "STABLE-PERIOD-1": (
            "STABLE-PERIOD-1: the 8 events all dose weakly (no ZERO "
            "events; lr_m bounded away from 0 and the cap) and the read "
            "stays alive (>= 0.05) with a stable floor — THE INTERMEDIATE "
            "REGIME IS A USABLE SETPOINT; W055's tuner has its third "
            "knob-setting (2-cycle / period-1 / off); the calibration "
            "program's kinetics complete."),
        "RE-BIFURCATION": (
            "RE-BIFURCATION: a new 2-cycle forms at the halved cap (ZERO "
            "events return) — the relay's bifurcation is cap-dependent; "
            "the period-1 regime was transient."),
        "RAIL-DECAY": (
            "RAIL-DECAY: the read's floor decays through aliveness "
            "(< 0.05) — the halved cap under-doses; the regime is not "
            "viable; the tuner's floor is named."),
    },
    "lab_lean_verbatim": (
        "Lab lean: STABLE-PERIOD-1, weakly — x41's MIXED pattern "
        "suggested convergence, and the relay's self-limiting logic (dose "
        "on deficit) supports a stable weak-dosing equilibrium; the "
        "countervailing: the read's recovery dynamics at the halved dose "
        "may be too slow to hold the floor. State your own read."),
    "operationalizations": (
        f"frozen BEFORE compute: THE RESUME := x41's committed "
        f"runs/x41/x41_LRCAP_REA_resume.pt (raw-md5 {X41_LRCAP_RESUME_MD5}; "
        f"the LRCAP leg's FULL final state at t{BASE_STEP}: the model + "
        f"both SGD-M buffers + both generator states — the state where "
        f"MIXED appeared), the post mirror "
        f"runs/x41/x41_LRCAP_REA_post.pt (raw-md5 {X41_LRCAP_POST_MD5}) "
        f"model-bit-compared, and runs/x41/metrics.json (raw-md5 "
        f"{X41_METRICS_MD5}) asserted literal-for-literal (the four "
        f"events' gate/deficit/lr/post, classes "
        f"{'/'.join(X41_LRCAP_CLASSES)}, spends S_corpus "
        f"{X41_LRCAP_S_CORPUS!r} / S_maint {X41_LRCAP_S_MAINT!r}, step "
        f"{BASE_STEP}, n_maint 4); THE CONTINUATION := t{LEG_START + 1}.."
        f"t{LEG_END} ({CONT_STEPS} steps), M={CONT_M} UNCHANGED, lr_m_max "
        f"= 0.5 x LR_M_MAX_FROZEN = {LR_M_MAX_HALF!r} (THE SAME halved "
        f"cap — asserted byte-equal to x41's LRCAP leg's own cap), 8 "
        f"events at t925/950/975/1000/1025/1050/1075/1100 (the LRCAP "
        f"branch's GLOBAL events {GLOBAL_EVENT_BASE + 1}-"
        f"{GLOBAL_EVENT_BASE + 8}; the staging zeroes n_maint, the family "
        f"convention); THE POOLS := B_C = {B_C_400!r} x 200/400, B_M = "
        f"{B_M_400!r} x 8/16 (x41's length-scaled convention — the "
        f"per-step/per-event equal shares IDENTICAL to e339's committed "
        f"leg-2 and x41's legs); THE SCHEDULE EXTENSION := E261.INST_TOTAL "
        f"re-bound 1000 -> {SCHED_EXT_TOTAL} (the house cosine clamps to 0 "
        f"at t >= 1000, silently killing the corpus wash for the last "
        f"four events — REJECTED; {SCHED_EXT_TOTAL} keeps the schedule "
        f">= ~2.5x the family's max observed cap through t{LEG_END}, the "
        f"budget cap remaining the ONLY binding corpus constraint — as at "
        f"every logged step of e339's leg-2 and both of x41's cap-bound "
        f"legs; G_SCHED asserts cap < sched at EVERY logged step and "
        f"records the sched/cap margin); PHASE-DECLARED READS "
        f"(METHODS 4) := "
        f"every event at BOTH phases (the PRE = the controller's gate "
        f"read, the POST = the traj milestone at the same step, measured "
        f"after the dose; MILESTONES = ALL event steps) + the window "
        f"ledger's intra-cycle flanks and first/last-event brackets; "
        f"aliveness over EVERY declared read; THE CLASSIFICATION := "
        f"ZERO-phase := deficit == 0.0 exactly, DOSE-phase := deficit >= "
        f"{DOSE_DEFICIT_MIN}, MIXED := the rest (x41's frozen forms); "
        f"THE BARS := STABLE-PERIOD-1 [(a) no ZERO events AND (b) every "
        f"lr_m/{LR_M_MAX_HALF!r} in [{INTERIOR_LO}, {INTERIOR_HI}] AND "
        f"(c) every declared read >= {ALIVE_FLOOR} AND (d) the 8 gate "
        f"reads' range <= {FLOOR_BAND} and mean(last 4) >= mean(first 4) "
        f"- {FLOOR_BAND}] -> RE-BIFURCATION [>= 2 ZERO events AND a ZERO "
        f"immediately followed by a dosing event] -> RAIL-DECAY [any "
        f"declared read < {ALIVE_FLOOR}] -> MIXED (the residue, "
        f"disclosed); the off-target battery (gm12, g0_z, CE_R) read at "
        f"the end (the standing amendment's collateral check, reported "
        f"never adjudicating); HARD GATES := G_NAMEWIN, G_NAMEFREE, "
        f"G_BASE, G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, "
        f"G_SUBSTRATE, G_X41BIND, G_STAGING, G_T900_REPRO, G_REGIME, "
        f"G_CONT, G_PHASEDECL, G_BUFSEP, G_SCHED, G_ORTH, G_BUDGET, "
        f"G_STREAMELIVE; NON-HALTING := {{G_ORTH, G_BUDGET, "
        f"G_STREAMELIVE}}; the driver is e336's COMMITTED body executed "
        f"VERBATIM under the disclosed re-binding; the leg's final state "
        f"checkpointed."),
    "predictions": {
        "P-W055a_stable_period_1": (
            "THE EXECUTOR'S OWN READ (per the dispatch's 'State your own "
            "read'): STABLE-PERIOD-1 — MODERATELY, one notch above the "
            "lab's 'weakly'. THE MECHANISM: the controller is a negative "
            "feedback on the read (dose ∝ deficit); x41's LRCAP tail "
            "shows the loop CLOSING — the gate drifted 0.190 -> 0.175 "
            "while the post CLIMBED 0.548 -> 0.595 (the correction "
            "strengthening exactly as the gate drifts down); the ZERO "
            "boundary needs a gate read >= the baseline 0.2859, ~0.10 "
            "ABOVE the observed gate band 0.143-0.190, never observed "
            "post-establishment under a dosing regime; the decay boundary "
            "needs the halved dose to under-compensate, but the dose has "
            "~2.0-2.7x headroom before its own cap (observed lr_targets "
            "0.0037-0.0055 vs cap 0.0110) and the deficit-proportional "
            "law GROWS the dose as the read decays. COUNTERVAILING "
            "(honestly registered): (1) the wash cap drifts with b_norm "
            "(committed band 0.0025-0.0035) — a sustained stronger wash "
            "could walk the floor down faster than the feedback corrects "
            "(RAIL-DECAY's path); (2) n=1, ONE slot's deterministic "
            "continuation — the fixed point is one branch's realization; "
            "(3) the floor-stability clause is the tightest: the "
            "committed weak-phase gates spanned 0.098-0.190 across cells "
            "— a range at the clause's own 0.08 band edge. DISCRIMINATING "
            "DATA: events 5-6 (t1025/t1050) gate reads — inside [0.10, "
            "0.26] with interior doses => the fixed point holds; climbing "
            "past 0.24 => ZERO risk (RE-BIFURCATION); below 0.10 => "
            "floor-drift risk (RAIL-DECAY). SCORED: TRUE iff the verdict "
            "== STABLE-PERIOD-1."),
    },
}
G_NAMEWIN_KEYSET = None  # retired (inlined into the operationalizations)

deviations: list[str] = [
    "THE SCHEDULE EXTENSION (a birth decision, disclosed in the "
    "operationalizations): E261.INST_TOTAL re-bound 1000 -> "
    f"{SCHED_EXT_TOTAL} — the ONLY module global re-bound beyond the "
    "family's leg-parameter set; without it the house cosine clamps to 0 "
    "at t >= 1000 and silently kills the corpus wash for this leg's last "
    "four events (a protocol change smuggled in as decay — rejected); "
    "G_SCHED asserts the cap stayed the ONLY binding corpus constraint",
    "LEG LENGTH 200 steps (8 events x M=25) vs the dispatch's bare '8 "
    "more events' — the event count fixes the length under M=25 "
    "UNCHANGED (the regime constant); no other reading is available",
]
builds_on = [
    "R79's minted card (the dispatch: 'W055 THE RELAY TUNER (W049 "
    "graduates: the LRCAP post resumed 8 events — STABLE-PERIOD-1 / "
    "RE-BIFURCATION / RAIL-DECAY; the setpoint observable phase-declared "
    "per METHODS 4)')",
    "x41 / THE RELAY'S KINETICS (THE PARENT CELL: its committed LRCAP "
    "leg's final state resumed here; its ENTRAINABLE verdict + the MIXED "
    "break pattern the object; its staging + leg conventions carried "
    "verbatim; its harness imported)",
    "METHODS 4 of THE_LAWS_V3.md (the phase-declaration amendment, "
    "R79-era T318: 'a sampler locked to the phenomenon's phase shows the "
    "phase, not the phenomenon')",
    "W049 (the calibration-dial question the verdict arms with a third "
    "knob-setting)",
    "e336 / THE FOUNDING-CLASS SECOND INSTANCE (the driver, executed "
    "verbatim), e339 (the t800 parent + the pool conventions), e288 (the "
    "ERROR-GATED controller rig), e287 (the b_m calibration + LR_M_MAX "
    "identity), e311 (the TAVIREN organism), e273 (LR_STABLE), e261 (the "
    "room + the burst machinery)",
]
whats_new = [
    "THE FIRST STABILITY QUESTION ASKED OF THE CONTROLLER: every prior "
    "cell (e288 -> x41) PERTURBED and read the response; W055 holds the "
    "regime CONSTANT (the same halved cap, the same M, the same stream) "
    "and asks whether the candidate intermediate regime persists — the "
    "tuner's first probe",
    "THE FIRST PHASE-DECLARED CONTINUATION: every event read at both "
    "phases by construction (METHODS 4 enacted in the rig, not audited "
    "afterward — x41's rider caught the milestone cadence phase-locked "
    "to e339's dose phase; that costume is impossible here)",
    "THE SETPOINT QUESTION: the relay tuner's knob-settings enumerated "
    "(2-cycle / period-1 / off) — the calibration program's kinetics "
    "completed if the middle setting holds",
]

metrics: dict = {}
metrics["gates"] = {}


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    metrics["last_write"] = common.now_iso()
    save_json(RD / "metrics.json", metrics)
    log(f"  [partial write] {note}")


def med(xs):
    ys = sorted(float(x) for x in xs)
    return ys[len(ys) // 2] if ys else None


def md5of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_head() -> str:
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=E43.REPO, capture_output=True,
        text=True).stdout.strip()


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()])


# ======================================================================
# THE LEG RE-BINDING MACHINE — the e336-module globals this cell touches;
# asserted: same set in, same set out, nothing else changed
# ======================================================================
LEG_KEYS = ("PHASE_STEPS", "MILESTONES", "WIN1", "WIN2", "MAINT_EVERY")
_LEG_ORIGINALS: dict | None = None


def patch_leg_globals(n_steps: int, milestones, win1, win2,
                      maint_every: int) -> dict:
    global _LEG_ORIGINALS
    before = {k: getattr(E336, k) for k in LEG_KEYS}
    if _LEG_ORIGINALS is None:
        _LEG_ORIGINALS = dict(before)      # the committed module state
    E336.PHASE_STEPS = n_steps
    E336.MILESTONES = tuple(milestones)
    E336.WIN1 = tuple(win1)
    E336.WIN2 = tuple(win2)
    E336.MAINT_EVERY = int(maint_every)
    after = {k: getattr(E336, k) for k in LEG_KEYS}
    log(f"  [leg-bind] e336 module globals re-bound: {after} "
        f"(was {before}); every protocol constant untouched")
    return {"before": {k: str(v) for k, v in before.items()},
            "after": {k: str(v) for k, v in after.items()}}


def restore_leg_globals() -> None:
    assert _LEG_ORIGINALS is not None
    for k, v in _LEG_ORIGINALS.items():
        setattr(E336, k, v)
    log(f"  [leg-bind] e336 module globals RESTORED to the committed "
        f"state {_LEG_ORIGINALS}")


# ======================================================================
# THE STAGING MACHINE — x41's committed LRCAP final state -> this cell's
# base: the state bits (model/optC/optF/cgen/igen) pass through UNTOUCHED
# (digest-verified); the leg ledgers + pools zeroed (the leg convention)
# ======================================================================
def _tensor_digest(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        return hashlib.md5(x.detach().cpu().contiguous()
                           .numpy().tobytes()).hexdigest()
    if isinstance(x, dict):
        parts = []
        for k in sorted(x.keys(), key=repr):
            parts.append(repr(k))
            parts.append(_tensor_digest(x[k]))
        return hashlib.md5("|".join(parts).encode()).hexdigest()
    if isinstance(x, (list, tuple)):
        return hashlib.md5("|".join(_tensor_digest(v) for v in x)
                           .encode()).hexdigest()
    return hashlib.md5(repr(x).encode()).hexdigest()


def staged_leg_state(st: dict, N: int) -> dict:
    """x41's LRCAP final state -> the leg-staged form (pools zeroed,
    bits kept)."""
    leg = {
        "model": st["model"],                       # bits unchanged
        "optC": st["optC"], "optF": st["optF"],
        "cgen_state": st["cgen_state"],
        "step": int(st["step"]),
        # ---- the leg reset (the ONLY transformation; disclosed) ----
        "traj": [], "corpus_ledger": {}, "orth_ledger": {},
        "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
        "lr_ledger": {}, "maint_ledger": [], "window_ledger": [],
        "bufsep": {"corpus_checks": 0, "corpus_violations": 0,
                   "maint_checks": 0, "maint_violations": 0,
                   "optF_steps": 0},
        "corp_cum": torch.zeros(N, dtype=torch.float64),
        "tot_prev": torch.zeros(N, dtype=torch.float64),
        "S_corpus": 0.0, "n_capped": 0,
        "n_chunks": 0, "chunk_table": [],
        "orth_max": 0.0,
        "igen_state": st["igen_state"],             # bits unchanged
        "maint_cum": torch.zeros(N, dtype=torch.float64),
        "S_maint": 0.0,
        "n_maint": 0, "n_maint_capped": 0,
    }
    return leg


def state_bits(st: dict) -> dict:
    return {"model": {k: _tensor_digest(v) for k, v in st["model"].items()},
            "optC": _tensor_digest(st["optC"]),
            "optF": _tensor_digest(st["optF"]),
            "cgen_state": _tensor_digest(st["cgen_state"]),
            "igen_state": _tensor_digest(st["igen_state"])}


# ======================================================================
# THE PHASE-DECLARED EVENT READER — every event, both phases (METHODS 4)
# ======================================================================
def classify_events(maint_rows: list[dict], traj_rows: list[dict],
                    window_rows: list[dict]) -> dict:
    post_by_step = {int(r["step"]): float(r["g0_pz"])
                    for r in traj_rows}
    win_by_key = {}
    for r in window_rows:
        win_by_key[(r["window_phase"], int(r["step"]))] = float(r["g0_pz"])
    ev = []
    for r in maint_rows:
        d = float(r["deficit_t"])
        cls = "ZERO" if d == 0.0 else ("DOSE" if d >= DOSE_DEFICIT_MIN
                                       else "MIXED")
        ev.append({
            "leg_index": int(r["maint_index"]),
            "global_index": GLOBAL_EVENT_BASE + int(r["maint_index"]),
            "step": int(r["step"]),
            # ---- THE PHASE-DECLARED PAIR (both phases, every event) ----
            "pre_read_gate": float(r["read_at_gate"]),
            "phase_pre": "gate (pre-dose): the read the controller saw",
            "post_read": post_by_step.get(int(r["step"])),
            "phase_post": "traj milestone (post-dose): the read after "
                          "the event's step",
            "deficit": d, "lr_m": float(r["lr_maint"]),
            "lr_fraction_of_halved_cap": float(r["lr_maint"]) / LR_M_MAX_HALF,
            "binder": r.get("binder"),
            "realized": float(r["realized_step_norm"]),
            "class": cls})
    classes = [e["class"] for e in ev]
    gate_reads = [e["pre_read_gate"] for e in ev]
    post_reads = [e["post_read"] for e in ev
                  if e["post_read"] is not None]
    n = len(gate_reads)
    floor_range = (max(gate_reads) - min(gate_reads)) if gate_reads else None
    floor_drift = (sum(gate_reads[n // 2:]) / max(len(gate_reads[n // 2:]), 1)
                   - sum(gate_reads[:n // 2])
                   / max(len(gate_reads[:n // 2]), 1)) if n >= 2 else None
    # the alternation signature: a ZERO event immediately followed by a
    # dosing (non-ZERO) event
    zero_then_dose = any(classes[i] == "ZERO" and classes[i + 1] != "ZERO"
                         for i in range(len(classes) - 1))
    return {
        "events": ev, "classes": classes,
        "n_zero": classes.count("ZERO"), "n_dose": classes.count("DOSE"),
        "n_mixed": classes.count("MIXED"),
        "zero_then_dose_signature": zero_then_dose,
        "floor_rail_mean": (sum(gate_reads) / len(gate_reads)
                            if gate_reads else None),
        "floor_rail_range": floor_range,
        "floor_rail_second_half_minus_first_half": floor_drift,
        "upper_rail_mean": (sum(post_reads) / len(post_reads)
                            if post_reads else None),
        "dosing_gate_reads": [e["pre_read_gate"] for e in ev
                              if e["class"] in ("DOSE", "MIXED")],
    }


def declared_reads(cyc: dict, window_rows: list[dict]) -> list[dict]:
    """THE PHASE-DECLARED TRACE: every read this cell takes, each labeled
    with its phase — the aliveness bar's domain (frozen at birth)."""
    out: list[dict] = []
    for e in cyc["events"]:
        out.append({"step": e["step"], "read": e["pre_read_gate"],
                    "declared_phase": f"event {e['global_index']} PRE "
                                      f"(gate, pre-dose)"})
        if e["post_read"] is not None:
            out.append({"step": e["step"], "read": e["post_read"],
                        "declared_phase": f"event {e['global_index']} "
                                          f"POST (post-dose settle)"})
    for r in window_rows:
        out.append({"step": int(r["step"]), "read": float(r["g0_pz"]),
                    "declared_phase": f"window {r['window_phase']} "
                                      f"(intra-cycle)"})
    out.sort(key=lambda x: (x["step"], str(x["declared_phase"])))
    return out


# ======================================================================
def main():
    global dev
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "w055_relay_probe",
        "phase": (
            "THE RELAY TUNER'S FIRST PROBE (R79's card; W049's "
            "graduation): x41's committed LRCAP leg's final state (the "
            "model + both SGD-M buffers + both generator states at t900, "
            "md5-bound — the state where MIXED appeared) resumed into ONE "
            f"continuation leg — t901..t1100 ({CONT_STEPS} steps), M=25 "
            f"UNCHANGED, lr_m_max = the SAME halved cap "
            f"({LR_M_MAX_HALF!r}) — reading EVERY event's pre/post pair "
            "(phase-declared per METHODS 4); the bars: STABLE-PERIOD-1 / "
            "RE-BIFURCATION / RAIL-DECAY"),
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "net0_class": NET0_CLASS,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "the leg runs in bursts with cooldowns between — "
                      "never concurrent) + CPU fp64 projections, "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line (dispatch 85), recorded to "
                      "runs/_envelope_log.jsonl tagged w055:CONT",
            "trainings": "1 continuation leg (e336's driver verbatim; "
                         "budgeted orthogonalized corpus steps over a "
                         "fresh length-scaled B_C + the NAME-ONLY "
                         "maintenance events through opt_F at the "
                         "error-gated HALVED lr); NO twin (by design); "
                         "NO cons",
        },
        "builds_on": builds_on,
        "whats_new": whats_new,
        "deviations": deviations,
    })
    write_partial("startup (bars + lean + P-W055a registered, committed "
                  "at birth)")

    # ---- the committed x41 record (md5-bound; the repro targets) -------
    for p, want in [(X41_LRCAP_RESUME_CK, X41_LRCAP_RESUME_MD5),
                    (X41_LRCAP_POST_CK, X41_LRCAP_POST_MD5),
                    (X41_METRICS, X41_METRICS_MD5)]:
        got = md5of(p)
        assert got == want, f"x41 artifact drifted: {p.name} {got} != {want}"
    x41m = json.loads(X41_METRICS.read_text(encoding="utf-8"))
    assert x41m["adjudication"]["verdict"] == "ENTRAINABLE", \
        "x41's committed verdict drift"
    xl = x41m["legs"]["LRCAP"]
    xl_ev = xl["cycle"]["events"]
    for (g, st_, gate, dfd, lr_, post), row in zip(X41_LRCAP_CYCLE, xl_ev):
        assert int(row["global_index"]) == g \
            and int(row["step"]) == st_ \
            and abs(float(row["gate_read"]) - gate) < 1e-15 \
            and abs(float(row["deficit"]) - dfd) < 1e-15 \
            and abs(float(row["lr_m"]) - lr_) < 1e-15 \
            and (row["post_read"] is None
                 or abs(float(row["post_read"]) - post) < 1e-15), \
            f"committed x41 LRCAP cycle drift at global event {g}"
    assert xl["cycle"]["classes"] == X41_LRCAP_CLASSES, \
        "committed x41 LRCAP class drift"
    assert abs(xl["S_corpus"] - X41_LRCAP_S_CORPUS) < 1e-15 \
        and abs(xl["S_maint"] - X41_LRCAP_S_MAINT) < 1e-15, \
        "committed x41 LRCAP spend drift"
    # the regime's own cap: byte-equal to this cell's halved constant
    assert float(xl["lr_m_max"]) == LR_M_MAX_HALF, \
        "x41's LRCAP leg's cap != this cell's halved cap"
    # the post-cell mirror (the off-target battery x41 read at t900)
    x41_post_cells = xl["post_cells"]
    log("P-1 x41's committed record BOUND (metrics md5 "
        f"{X41_METRICS_MD5[:8]}...; the LRCAP leg's four events asserted "
        "literal-for-literal; classes "
        f"{'/'.join(X41_LRCAP_CLASSES)}; spends S_corpus "
        f"{X41_LRCAP_S_CORPUS:.4f} / S_maint {X41_LRCAP_S_MAINT:.4f}; the "
        f"halved cap {LR_M_MAX_HALF!r} byte-equal)")

    # ================= P0: the protocol rebuild (the family mirror) =====
    set_seed(E336.CORPUS_GEN_SEED)       # global init only
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    vocab = corpus.vocab_size
    zid = stoi["Z"]
    tid = stoi[E336.PARASITE_NAME[0]]    # the read channel: p(T)
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert vocab == 65, f"vocab drift {vocab}"

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
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    assert list(g0_ids.shape) == [60, G1.PRE]

    anchor_full = torch.stack([train_ids[p - G1.PRE:
                                         p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True

    tav_ids = corpus.encode(E336.PARASITE_NAME)

    def build_win_t(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], tav_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_t = torch.stack([build_win_t(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_t[i][G1.PRE: G1.PRE + len(E336.PARASITE_NAME)])
        == E336.PARASITE_NAME for i in range(win_t.shape[0]))
    win_pre_ok = all(torch.equal(win_t[i][:G1.PRE],
                                 anchor_full[i][:G1.PRE])
                     for i in range(win_t.shape[0]))
    G_NAMEWIN = {
        "form": "the maintenance vehicle re-bound (the family's G_NAMEWIN "
                "mirror): the TRUE TAVIREN install windows, decode-gated, "
                "pre-context bit-equal the anchor bank",
        "n_windows": int(win_t.shape[0]),
        "masked_decode_all_name": bool(win_masked_ok),
        "precontext_bit_equal_anchor": bool(win_pre_ok),
        "pass": bool(win_masked_ok and win_pre_ok
                     and list(win_t.shape) == [60, G1.BLOCK])}
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"
    metrics["gates"]["G_NAMEWIN"] = G_NAMEWIN

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"]["G_NAMEFREE"] = {
        "corpus_zeph_count": train_text.count("ZEPH"),
        "corpus_taviren_count": train_text.count(E336.PARASITE_NAME),
        "pass": bool(train_text.count("ZEPH") == 0
                     and train_text.count(E336.PARASITE_NAME) == 0)}
    assert metrics["gates"]["G_NAMEFREE"]["pass"]
    log("P0: the protocol rebuilt (corpus + battery + anchors + the name "
        "windows + the eval banks) — the family mirror PASS")

    # ================= P1: the room + the parents (the mirror) ==========
    _base_art = torch.load(E336.CKPT_DIR / E336.BASE_CK,
                           map_location="cpu", weights_only=False)
    base_sd = (_base_art["model"] if "model" in _base_art
               else _base_art)      # e001.pt is a RAW state dict
    del _base_art
    base_net = G1.evl_load(base_sd)
    n_par = sum(p.numel() for p in base_net.parameters())
    assert n_par == 2739072, f"param drift {n_par}"
    assert md5of(E336.CKPT_DIR / E336.BASE_CK) == E336.BASE001_MD5
    base_flat = flat_params_cpu(base_net)
    base_flat_np = base_flat.double().numpy().astype(np.float64)
    base_prior = G1.battery_cell(base_net, g0_ids, tid)["mean_pz"]
    G_BASE = {"checkpoint": f"runs/checkpoints/{E336.BASE_CK}",
              "md5": E336.BASE001_MD5, "params": n_par,
              "class": NET0_CLASS["name"],
              "base_prior_parasite_g0": base_prior,
              "committed_base_prior": E336.BASE_PRIOR_T,
              "abs_diff": abs(base_prior - E336.BASE_PRIOR_T),
              "pass": bool(abs(base_prior - E336.BASE_PRIOR_T) <= 2e-6)}
    assert G_BASE["pass"], f"G_BASE failed: {G_BASE}"
    metrics["gates"]["G_BASE"] = G_BASE
    del base_net

    vmap_art = torch.load(E336.CKPT_DIR / E336.VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {"path": f"runs/checkpoints/{E336.VMAP_CK}",
                "md5": md5of(E336.CKPT_DIR / E336.VMAP_CK),
                "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
                "size": int(v_flat32.numel()),
                "pass": bool(vmap_art.get("meta", {}).get("experiment")
                             == "e258" and int(v_flat32.numel()) == n_par)}
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND

    span_art = torch.load(E336.CKPT_DIR / E336.SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(E336.CKPT_DIR / E336.SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta",
                                                  {}).get("experiment"),
                  "pass": bool(md5of(E336.CKPT_DIR / E336.SPAN_CK)
                               == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == n_par)}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND

    # the room: re-derived from the committed seeds and bit-compared vs
    # e336's committed room (the room the whole lineage orthogonalized
    # against, x41's LRCAP stream included)
    rooms_here = E261.LadderRooms(n_par, E336.LADDER, v64_np,
                                  Vp.numpy().astype(np.float64),
                                  list(G1.evl_load(base_sd).parameters()),
                                  dev)
    if not SMOKE:
        cert = rooms_here.certify()
        assert cert["pass"], f"room cert failed: {cert}"
        rooms336 = torch.load(E43.REPO / "runs" / "e336" / "e336_rooms.pt",
                              map_location="cpu", weights_only=False)

        def _np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        D336 = _np(rooms336["model"]["K10K"]["D_int8"]).astype(np.float64)
        S336 = _np(rooms336["model"]["K10K"]["S"])
        D_mine = rooms_here.rooms["K10K"].D
        S_mine = rooms_here.rooms["K10K"].S
        G_ROOMK10K = {
            "form": "the room re-derived from the committed seeds and "
                    "bit-compared vs e336's committed e336_rooms.pt (the "
                    "same room e339's + x41's streams orthogonalized "
                    "against)",
            "D_bit_equal_e336": bool(np.array_equal(D_mine, D336)),
            "S_bit_equal_e336": bool(np.array_equal(S_mine, S336)),
            "e264_rooms_md5": E336.ROOMS264_MD5,
            "cert": {k: v for k, v in cert["per_rung"]["K10K"].items()
                     if not isinstance(v, list)},
            "pass": bool(np.array_equal(D_mine, D336)
                         and np.array_equal(S_mine, S336))}
        assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
        del rooms336
    else:
        G_ROOMK10K = {"form": "SMOKE: the room cert vacuous (disclosed; "
                              "the full run bit-binds)",
                      "pass": True, "vacuous": True}
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    proj = rooms_here
    log("P1 the room: K10K re-derived + bit-bound to e336's committed "
        "room: PASS")

    # ---- G_LR_BIND: the lr constants re-verified (the module bind) -----
    lr_sgd_runtime = float(json.loads(
        (E43.REPO / "runs" / "e273" / "lr_calibration.json")
        .read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the leg runs on the COMMITTED lr constants (the "
                "modules' own globals, asserted): LR_STABLE = x0.01 x "
                "LR_SGD_matched; LR_M_MAX = 0.40 x BUDGET / e287's b_m "
                "sum (the SAME identity, the SAME value); this leg's "
                "lr_m_max = 0.5 x LR_M_MAX_FROZEN — x41's LRCAP leg's "
                "own cap, byte-equal (the regime held constant); "
                "momentum 0.9, wd 0.0 EXACTLY",
        "lr_sgd_record": E336.E273_LR_SGD,
        "lr_sgd_runtime": lr_sgd_runtime,
        "lr_stable_module": E336.LR_STABLE,
        "lr_m_max_module": E336.LR_M_MAX_FROZEN,
        "lr_m_max_this_leg": LR_M_MAX_HALF,
        "x41_lrcap_leg_cap": float(xl["lr_m_max"]),
        "momentum": E336.SGD_MOMENTUM, "wd": E336.SGD_WD,
        "pass": bool(E336.LR_STABLE == 0.21738574801453703
                     and E336.LR_M_MAX_FROZEN
                     == 0.021942464013242388
                     and abs(LR_M_MAX_HALF - 0.010971232006621194) < 1e-15
                     and float(xl["lr_m_max"]) == LR_M_MAX_HALF
                     and lr_sgd_runtime == E336.E273_LR_SGD
                     and E336.SGD_MOMENTUM == 0.9 and E336.SGD_WD == 0.0)}
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND

    # ================= P2: the substrate (the mirror) ====================
    sub_art = torch.load(E336.SUBSTRATE_CK, map_location="cpu",
                         weights_only=False)
    sub_meta = sub_art.get("meta", {}) or {}
    sub_sd = {k: v.detach().clone() for k, v in sub_art["model"].items()}
    sub_net = G1.evl_load(sub_sd)
    sub_flat = flat_params_cpu(sub_net)
    sub_flat_np = sub_flat.double().numpy().astype(np.float64)
    sub_g0 = G1.battery_cell(sub_net, g0_ids, tid)["mean_pz"]
    sub_g0_z = G1.battery_cell(sub_net, g0_ids, zid)["mean_pz"]
    write_norm = float(np.linalg.norm(sub_flat_np - base_flat_np))
    G_SUBSTRATE = {
        "form": "the substrate gate mirror: the organism artifact "
                "raw-md5-bound; its reads gated vs e311's committed "
                "G_FRESH literals (the family's 2e-6 cross-session law); "
                "the write norm recomputed fp64",
        "artifact_md5": md5of(E336.SUBSTRATE_CK),
        "committed_md5": E336.SUBSTRATE_MD5,
        "post_g0": sub_g0, "committed_post_g0": E336.TAV_POST_G0,
        "g0_abs_diff": abs(sub_g0 - E336.TAV_POST_G0),
        "g0_z_channel": sub_g0_z,
        "write_norm": write_norm, "committed_write_norm":
            E336.TAV_WRITE_NORM,
        "write_norm_rel_diff": abs(write_norm - E336.TAV_WRITE_NORM)
        / E336.TAV_WRITE_NORM,
        "pass": bool(md5of(E336.SUBSTRATE_CK) == E336.SUBSTRATE_MD5
                     and sub_meta.get("experiment") == "e311"
                     and abs(sub_g0 - E336.TAV_POST_G0) <= 2e-6
                     and abs(write_norm - E336.TAV_WRITE_NORM)
                     / E336.TAV_WRITE_NORM <= 1e-8)}
    assert G_SUBSTRATE["pass"], f"substrate gate FAILED: {G_SUBSTRATE}"
    metrics["gates"]["G_SUBSTRATE"] = G_SUBSTRATE
    del sub_net, sub_art
    log(f"P2 G_SUBSTRATE: the TAVIREN organism re-bound (g0 |d| "
        f"{G_SUBSTRATE['g0_abs_diff']:.1e}; write norm |rel d| "
        f"{G_SUBSTRATE['write_norm_rel_diff']:.1e}): PASS")
    write_partial("P0-P2 the protocol + room + substrate bound")

    # ================= P3: the x41 LRCAP final-state gates + staging ====
    if SMOKE:
        # the smoke base: manufacture a tiny committed-form base via
        # e336's own driver (12 steps, M=2) — machinery validation only;
        # the smoke numbers are MEANINGLESS (disclosed)
        log("P3 SMOKE: manufacturing a smoke base checkpoint via e336's "
            "own driver (12 steps, M=2) — machinery validation only")
        patch_leg_globals(SMOKE_BASE_STEPS, tuple(range(2, 13, 2)),
                          (1, 2, 3), (10, 11, 12), SMOKE_BASE_M)
        bc0 = B_C_400 * SMOKE_BASE_STEPS / 400
        bm0 = B_M_400 * (SMOKE_BASE_STEPS // SMOKE_BASE_M) / 16
        _ = E336.chunked_errorgated_phase(
            "W055-SMOKE-BASE", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, bc0, bm0, win_t, inst_mask, anchor_full,
            train_ids, g0_ids, gm12_ids, r_eval_xy, tid, LR_M_MAX_HALF,
            RD / "smoke_base_REA_resume.pt", dev)
        base_src = RD / "smoke_base_REA_resume.pt"
        base_step_expect = SMOKE_BASE_STEPS
        base_nmaint_expect = SMOKE_BASE_STEPS // SMOKE_BASE_M
    else:
        base_src = X41_LRCAP_RESUME_CK
        base_step_expect = BASE_STEP
        base_nmaint_expect = 4

    base_st = torch.load(base_src, map_location="cpu", weights_only=False)
    bits_src = state_bits(base_st)
    G_X41BIND = {
        "form": "the w055 base identity: x41's committed LRCAP leg's "
                "final state raw-md5-bound; step/n_maint/spend asserted "
                "== the committed metrics; the resume model bit-compared "
                "vs the committed post checkpoint; the LRCAP cycle "
                "asserted literal-for-literal; the leg's cap asserted "
                "byte-equal to this cell's",
        "resume_md5": md5of(base_src),
        "committed_md5": None if SMOKE else X41_LRCAP_RESUME_MD5,
        "step": int(base_st["step"]),
        "n_maint": int(base_st["n_maint"]),
        "S_corpus": float(base_st["S_corpus"]),
        "S_maint": float(base_st["S_maint"]),
        "expected": {"step": base_step_expect,
                     "n_maint": base_nmaint_expect,
                     "S_corpus": (None if SMOKE else X41_LRCAP_S_CORPUS),
                     "S_maint": (None if SMOKE else X41_LRCAP_S_MAINT)},
        "the_state_where_MIXED_appeared": (
            "x41's LRCAP leg ended with classes "
            + "/".join(X41_LRCAP_CLASSES)
            + f" — events 35/36 the MIXED (weak-dose) pattern; this "
            f"state (t{BASE_STEP}) is the period-1 candidate's birth"),
    }
    model_bit_equal = None
    if not SMOKE:
        x41_post = torch.load(X41_LRCAP_POST_CK, map_location="cpu",
                              weights_only=False)
        model_bit_equal = all(
            torch.equal(base_st["model"][k], x41_post["model"][k])
            for k in base_st["model"])
        del x41_post
        G_X41BIND["resume_model_bit_equal_post"] = bool(model_bit_equal)
    G_X41BIND["pass"] = bool(
        (SMOKE or md5of(base_src) == X41_LRCAP_RESUME_MD5)
        and int(base_st["step"]) == base_step_expect
        and int(base_st["n_maint"]) == base_nmaint_expect
        and (SMOKE or (abs(float(base_st["S_corpus"])
                           - X41_LRCAP_S_CORPUS) < 1e-9
                       and abs(float(base_st["S_maint"])
                               - X41_LRCAP_S_MAINT) < 1e-9
                       and model_bit_equal)))
    assert G_X41BIND["pass"], f"x41 LRCAP final-state bind FAILED: {G_X41BIND}"
    metrics["gates"]["G_X41BIND"] = G_X41BIND

    # ---- the staging: state bits through, leg pools zeroed -------------
    base_ck = RD / ("smoke_w055_base.pt" if SMOKE else "w055_REA_base.pt")
    leg_ck = RD / ("smoke_w055_cont_resume.pt" if SMOKE
                   else "w055_cont_resume.pt")
    torch.save(staged_leg_state(base_st, n_par), base_ck)
    torch.save(staged_leg_state(base_st, n_par), leg_ck)
    chk = torch.load(leg_ck, map_location="cpu", weights_only=False)
    staging_ok = (state_bits(chk) == bits_src
                  and int(chk["step"]) == base_step_expect
                  and chk["S_corpus"] == 0.0 and chk["S_maint"] == 0.0
                  and chk["n_maint"] == 0)
    G_STAGING = {
        "form": "the ONLY protocol-touching transformation: x41's "
                "committed LRCAP final state bits (model/optC/optF/cgen/"
                "igen) pass through UNTOUCHED (digest-verified); the leg "
                "ledgers + spend pools zeroed (the leg convention); the "
                "generator states carried — the LRCAP BRANCH's OWN stream "
                "CONTINUES (never re-seeded)",
        "base": str(base_ck), "leg_resume": str(leg_ck),
        "bits_unchanged": bool(staging_ok),
        "global_event_mapping": "the staging zeroes n_maint (the family "
                                "convention): this leg's event i is the "
                                "LRCAP branch's GLOBAL event "
                                f"{GLOBAL_EVENT_BASE} + i",
        "pass": bool(staging_ok)}
    assert G_STAGING["pass"], f"staging FAILED: {G_STAGING}"
    metrics["gates"]["G_STAGING"] = G_STAGING

    # ---- G_T900_REPRO: the staged base reproduces the committed read ---
    net_b = G1.evl_load({k: v for k, v in base_st["model"].items()})
    repro_g0 = G1.battery_cell(net_b, g0_ids, tid)["mean_pz"]
    repro_g0_z = G1.battery_cell(net_b, g0_ids, zid)["mean_pz"]
    committed_g0 = (base_st["traj"][-1]["g0_pz"] if SMOKE
                    else X41_T900_POST_READ)
    G_T900_REPRO = {
        "form": "the staged bits reproduce x41's committed LRCAP-final "
                "battery read (the family's cross-session read-"
                "determinism law: same bits + same battery + same CPU "
                "fp32 code path)",
        "g0_reproduced": repro_g0,
        "g0_committed": committed_g0,
        "absdiff": abs(repro_g0 - committed_g0),
        "g0_z_channel": repro_g0_z,
        "tol": 2e-6,
        "pass": bool(abs(repro_g0 - committed_g0) <= 2e-6)}
    assert G_T900_REPRO["pass"], f"t900 repro FAILED: {G_T900_REPRO}"
    metrics["gates"]["G_T900_REPRO"] = G_T900_REPRO
    del net_b, base_st, chk
    log(f"P3 G_X41BIND + G_STAGING + G_T900_REPRO: the LRCAP final state "
        f"bound (read repro |d| {G_T900_REPRO['absdiff']:.1e}); the leg "
        "copy staged digest-equal: PASS")
    write_partial("P3 the base bound + staged")

    # ================= P4: THE CONTINUATION LEG =========================
    start = base_step_expect
    end = start + LEG_STEPS
    m = LEG_M
    ev_steps = [s for s in range(start + 1, end + 1) if s % m == 0]
    assert len(ev_steps) == (8 if not SMOKE else 4), \
        f"event-count drift: {ev_steps}"
    milestones = tuple(ev_steps)          # EVERY event read post-dose
    win1 = (ev_steps[0] - 1, ev_steps[0], ev_steps[0] + 1)
    win2 = (ev_steps[-1] - 1, ev_steps[-1], ev_steps[-1] + 1)
    bind_rec = patch_leg_globals(end, milestones, win1, win2, m)

    # THE SCHEDULE EXTENSION (the disclosed re-bind; restored after)
    inst_total_committed = E261.INST_TOTAL
    assert inst_total_committed == 1000, f"INST_TOTAL drift: {inst_total_committed}"
    E261.INST_TOTAL = SCHED_EXT_TOTAL
    rea_arm_before = E336.REA_ARM
    E336.REA_ARM = "w055:CONT"            # label-only (the burst tags);
    # disclosed under the family's re-binding convention
    budget_c = B_C_400 * LEG_STEPS / 400
    budget_m = B_M_400 * len(ev_steps) / 16
    lr_m = float(LR_M_MAX_HALF)
    log("=" * 78)
    log(f"LEG-CONT — t{start + 1}..t{end} ({LEG_STEPS} steps, M={m} "
        f"UNCHANGED, lr_m_max {lr_m:.10f} = THE SAME HALVED CAP, "
        f"{len(ev_steps)} events at {ev_steps}; B_C {budget_c:.4f} B_M "
        f"{budget_m:.4f} — the length-scaled pools; INST_TOTAL "
        f"{inst_total_committed} -> {SCHED_EXT_TOTAL} (THE SCHEDULE "
        "EXTENSION); e336's committed driver VERBATIM)")
    assert common.gpu_ok(), "gpu_ok() gate before LEG-CONT"
    res = E336.chunked_errorgated_phase(
        "W055-CONT", G1.evl_load(sub_sd), proj, sub_flat_np,
        base_flat_np, budget_c, budget_m, win_t, inst_mask,
        anchor_full, train_ids, g0_ids, gm12_ids, r_eval_xy, tid,
        lr_m, leg_ck, dev)
    E261.INST_TOTAL = inst_total_committed   # restore (leave e261 as found)
    E336.REA_ARM = rea_arm_before            # restore (leave e336 as found)
    restore_leg_globals()

    sd_l = res["sd"]
    net_l = G1.evl_load(sd_l)
    # ---- THE OFF-TARGET BATTERY at the end (the standing amendment's
    # collateral check): the gm12 offset battery, the Z channel, CE_R
    cells_l = {"gm12": G1.battery_cell(net_l, gm12_ids, tid)["mean_pz"],
               "g0": G1.battery_cell(net_l, g0_ids, tid)["mean_pz"],
               "g0_z": G1.battery_cell(net_l, g0_ids, zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_l, *r_eval_xy)}
    torch.save({"model": sd_l, "meta": {
        "experiment": "w055", "leg": "CONT",
        "desc": "w055 the relay tuner's first probe: the LRCAP-final "
                "state continued 8 events at the SAME halved cap",
        "m_every": m, "lr_m_max": lr_m,
        "S_corpus": res["S_corpus"], "S_maint": res["S_maint"]}},
        RD / ("smoke_w055_CONT_REA_post.pt" if SMOKE
              else "w055_CONT_REA_post.pt"))
    del net_l

    # ---- the phase-declared cycle reader ----
    cyc = classify_events(res["maint_ledger"], res["traj"],
                          res["window_ledger"])
    dreads = declared_reads(cyc, res["window_ledger"])
    min_read = min(d["read"] for d in dreads)
    min_read_row = min(dreads, key=lambda d: d["read"])

    # ---- G_CONT: the leg's protocol identity ----
    lr_rows = res["lr_ledger"]
    applied = [v["lr_applied"] for v in lr_rows.values()]
    cap_below = {int(k): bool(v["cap"] <= v["lr_sched"])
                 for k, v in lr_rows.items()}
    sched_bound = [s for s, ok in cap_below.items() if not ok]
    margins = {int(k): float(v["lr_sched"] / max(v["cap"], 1e-30))
               for k, v in lr_rows.items()}
    med_applied = med(applied) if applied else None
    G_CONT = {
        "form": f"the leg's protocol identity: M={m} (UNCHANGED), "
                f"lr_m_max={lr_m!r} (THE SAME HALVED CAP — x41's LRCAP "
                f"leg's own value), PHASE_STEPS={end}, events at "
                f"{ev_steps}, pools B_C={budget_c!r} B_M={budget_m!r} "
                f"(length-scaled); the applied corpus lr's MEDIAN within "
                f"the frozen protocol band {LR_BAND} (e339's committed "
                f"leg-2 range widened 20% — the wash kinetics matched)",
        "m_every": m, "lr_m_max": lr_m, "phase_steps": end,
        "event_steps": ev_steps, "milestones": list(milestones),
        "budget_c": budget_c, "budget_m": budget_m,
        "leg_bind_record": bind_rec,
        "applied_lr_median": med_applied,
        "applied_lr_min": min(applied) if applied else None,
        "applied_lr_max": max(applied) if applied else None,
        "lr_band": LR_BAND,
        "pass": bool(med_applied is not None
                     and LR_BAND[0] <= med_applied <= LR_BAND[1]
                     and [int(r["step"]) for r in res["maint_ledger"]]
                     == ev_steps
                     and int(res["steps_ran"]) == end)}
    assert G_CONT["pass"], f"leg identity FAILED: {G_CONT}"
    metrics["gates"]["G_CONT"] = G_CONT

    # ---- G_REGIME: the regime identity (the stability question's twin
    # pillars: the SAME cap, the SAME cadence; the ONLY deltas disclosed)
    G_REGIME = {
        "form": "THE REGIME HELD CONSTANT: lr_m_max == x41's LRCAP leg's "
                "cap (byte-equal, asserted); MAINT_EVERY == 25 (x41's "
                "LRCAP leg's own M, unchanged); the schedule extension "
                "disclosed + the corpus cap asserted the ONLY binding "
                "constraint (the operative wash variable held constant)",
        "lr_m_max_this_leg": lr_m,
        "lr_m_max_x41_lrcap": float(xl["lr_m_max"]),
        "byte_equal": bool(lr_m == float(xl["lr_m_max"])),
        "m_this_leg": m, "m_x41_lrcap": int(xl["m_every"]),
        "m_unchanged": bool(m == int(xl["m_every"])),
        "schedule_extension": {"from": inst_total_committed,
                               "to": SCHED_EXT_TOTAL,
                               "restored_after": True},
        "pass": bool(lr_m == float(xl["lr_m_max"])
                     and m == int(xl["m_every"])
                     and inst_total_committed == 1000
                     and E261.INST_TOTAL == 1000)}
    assert G_REGIME["pass"], f"regime identity FAILED: {G_REGIME}"
    metrics["gates"]["G_REGIME"] = G_REGIME

    # ---- G_PHASEDECL: every event, both phases ----
    ev_posts_ok = all(e["post_read"] is not None for e in cyc["events"])
    n_pairs = sum(1 for e in cyc["events"] if e["post_read"] is not None)
    # the window brackets: the first + last events carry pre-window rows;
    # the gate read and the win-pre read are the SAME instrument on the
    # SAME state — assert the family's 2e-6 determinism
    win_pre_by_step = {}
    for r in res["window_ledger"]:
        ph = str(r["window_phase"])
        if ph.endswith("-pre"):
            win_pre_by_step[int(r["step"])] = float(r["g0_pz"])
    bracket_diffs = {}
    for e in cyc["events"]:
        if e["step"] in win_pre_by_step:
            bracket_diffs[f"event@t{e['step']}"] = abs(
                e["pre_read_gate"] - win_pre_by_step[e["step"]])
    brackets_ok = all(v <= 2e-6 for v in bracket_diffs.values())
    G_PHASEDECL = {
        "form": "METHODS 4 ENACTED IN THE RIG: every event read at BOTH "
                "phases (the PRE = the controller's gate read; the POST "
                "= the traj milestone at the same step, MILESTONES == "
                "ALL event steps) + the window ledger's intra-cycle "
                "flanks; a milestone-only cadence (the costume x41's "
                "rider caught on e339) is impossible by construction",
        "n_events": len(cyc["events"]),
        "n_pre_post_pairs": n_pairs,
        "declared_read_count": len(dreads),
        "window_bracket_gate_vs_pre_absdiffs": bracket_diffs,
        "brackets_within_2e-6": bool(brackets_ok),
        "aliveness_domain": "every declared read (pre gates + post "
                            "settles + intra-cycle windows)",
        "pass": bool(ev_posts_ok and brackets_ok
                     and len(dreads) >= 2 * len(cyc["events"]))}
    assert G_PHASEDECL["pass"], f"phase-declaration FAILED: {G_PHASEDECL}"
    metrics["gates"]["G_PHASEDECL"] = G_PHASEDECL
    log(f"P4 LEG-CONT DONE: classes {cyc['classes']} | floor rail mean "
        f"{cyc['floor_rail_mean']} range {cyc['floor_rail_range']} | "
        f"upper rail mean {cyc['upper_rail_mean']} | min declared read "
        f"{min_read:.6f} ({min_read_row['declared_phase']}) | post g0 "
        f"{cells_l['g0']:.6f} | S {res['S_corpus']:.4f}+"
        f"{res['S_maint']:.4f} | sched-bound steps {sched_bound}")
    write_partial("LEG-CONT complete")

    # ================= P5: the standing gates ===========================
    G_ORTH = {
        "form": "the orthogonalized-stream gate (NON-HALTING): the "
                "stepped corpus gradient entirely orthogonal to the room "
                "at every step of the leg",
        "orth_max_rel_err": res["orth_max"],
        "bar": ORTH_BAR,
        "pass": bool(res["orth_max"] < ORTH_BAR)}
    metrics["gates"]["G_ORTH"] = G_ORTH

    G_BUFSEP = {
        "form": "the isolation gate (HALT carried): bidirectional buffer "
                "separation at every optimizer event of the leg",
        "per_leg": res["bufsep"],
        "pass": bool(res["bufsep"]["corpus_violations"] == 0
                     and res["bufsep"]["maint_violations"] == 0)}
    assert G_BUFSEP["pass"], f"G_BUFSEP HALT: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    G_BUDGET = {
        "form": "the length-scaled leg pools respected (NON-HALTING)",
        "S_corpus": res["S_corpus"], "S_maint": res["S_maint"],
        "B_C": budget_c, "B_M": budget_m,
        "pass": bool(res["S_corpus"] <= budget_c + BUDGET_SLACK
                     and res["S_maint"] <= budget_m + BUDGET_SLACK)}
    metrics["gates"]["G_BUDGET"] = G_BUDGET

    rows = sorted((int(s), v) for s, v in res["lr_ledger"].items())
    cl = res.get("corpus_ledger", {})
    crows = sorted((int(s), v) for s, v in cl.items())
    if len(crows) >= 4:
        early = [v["ce"] for s, v in crows if s <= crows[0][0] + 30]
        late = [v["ce"] for s, v in crows if s > crows[-1][0] - 30]
    else:
        tr = res["traj"]
        early = [r["ce_corpus"] for r in tr[:max(1, len(tr) // 2)]]
        late = [r["ce_corpus"] for r in tr[max(1, len(tr) // 2):]]
    ok = (None if len(early) < 1 or len(late) < 1 else bool(
        med(late) <= STREAM_STABLE_TOL * med(early)))
    G_STREAMELIVE = {
        "form": "the standing stream condition (NON-HALTING): the corpus "
                "stream live in the leg (late-vs-early median CE "
                f"improving OR stable within {STREAM_STABLE_TOL}x)",
        "median_early": med(early) if early else None,
        "median_late": med(late) if late else None,
        "pass": True if ok is None else ok}
    metrics["gates"]["G_STREAMELIVE"] = G_STREAMELIVE

    G_SCHED = {
        "form": f"THE SCHEDULE EXTENSION verified: E261.INST_TOTAL re-"
                f"bound 1000 -> {SCHED_EXT_TOTAL} (the house cosine "
                "clamps to 0 at t >= 1000 — a verbatim continuation "
                "would kill the corpus wash for the last 4 events); "
                "asserted: the corpus budget cap the ONLY binding "
                "constraint at EVERY logged step (cap < sched; the "
                "sched/cap margin recorded per step)",
        "sched_bound_steps": sched_bound,
        "min_sched_over_cap_margin": (min(margins.values())
                                      if margins else None),
        "margins_by_step": margins,
        "design_margin_target": SCHED_MARGIN_MIN,
        "pass": bool(sched_bound == [] and bool(margins))}
    metrics["gates"]["G_SCHED"] = G_SCHED

    # ================= P6: the adjudication (the frozen bars) ===========
    classes = cyc["classes"]
    gate_reads = [e["pre_read_gate"] for e in cyc["events"]]
    fracs = [e["lr_fraction_of_halved_cap"] for e in cyc["events"]]
    n4 = len(gate_reads) // 2
    floor_range = cyc["floor_rail_range"]
    floor_drift = cyc["floor_rail_second_half_minus_first_half"]

    c_a = bool(cyc["n_zero"] == 0)                       # no ZERO events
    c_b = bool(all(INTERIOR_LO <= f <= INTERIOR_HI for f in fracs))
    c_c = bool(min_read >= ALIVE_FLOOR)                  # alive
    c_d = bool(floor_range is not None and floor_range <= FLOOR_BAND
               and floor_drift is not None
               and floor_drift >= -FLOOR_BAND)           # stable floor
    stable_period_1 = bool(c_a and c_b and c_c and c_d)
    re_bifurcation = bool(cyc["n_zero"] >= 2
                          and cyc["zero_then_dose_signature"])
    rail_decay = bool(min_read < ALIVE_FLOOR)

    standing_ok = bool(G_ORTH["pass"] and G_BUFSEP["pass"]
                       and G_BUDGET["pass"] and G_STREAMELIVE["pass"]
                       and G_SCHED["pass"])
    if not standing_ok:
        verdict = "INSTRUMENT-FAILS"
    elif stable_period_1:
        verdict = "STABLE-PERIOD-1"
    elif re_bifurcation:
        verdict = "RE-BIFURCATION"
    elif rail_decay:
        verdict = "RAIL-DECAY"
    else:
        verdict = "MIXED"

    clause = {
        "STABLE-PERIOD-1": (
            f"the {len(classes)} events all dose weakly (classes "
            f"{'/'.join(classes)} — no ZERO events) with every dose "
            f"interior ({min(fracs):.3f}-{max(fracs):.3f} of the halved "
            f"cap), the read alive (min declared read {min_read:.4f} >= "
            f"{ALIVE_FLOOR}) and the floor stable (range "
            f"{floor_range:.4f} <= {FLOOR_BAND}; drift "
            f"{floor_drift:+.4f}) — THE INTERMEDIATE REGIME IS A USABLE "
            "SETPOINT; W055's tuner has its third knob-setting (2-cycle "
            "/ period-1 / off); the calibration program's kinetics "
            "complete"),
        "RE-BIFURCATION": (
            f"ZERO events returned ({cyc['n_zero']} of {len(classes)}; "
            f"classes {'/'.join(classes)}; the period-2 signature "
            f"ZERO->dose present: {cyc['zero_then_dose_signature']}) — a "
            "new 2-cycle formed at the halved cap; the relay's "
            "bifurcation is cap-dependent; the period-1 regime was "
            "transient"),
        "RAIL-DECAY": (
            f"the read's floor decayed through aliveness (min declared "
            f"read {min_read:.4f} < {ALIVE_FLOOR}, at "
            f"'{min_read_row['declared_phase']}') — the halved cap "
            "under-doses; the regime is not viable; the tuner's floor is "
            "named"),
        "MIXED": (
            "neither frozen bar fired cleanly — the residue, fully "
            f"disclosed: no-ZERO {c_a}; doses interior {c_b} "
            f"(fractions {['%.3f' % f for f in fracs]}); alive {c_c} "
            f"(min read {min_read:.4f}); floor stable {c_d} (range "
            f"{floor_range}, drift {floor_drift}); the full trace in "
            "metrics"),
        "INSTRUMENT-FAILS": (
            "a standing gate failed (G_ORTH / G_BUFSEP / G_BUDGET / "
            "G_STREAMELIVE / G_SCHED) — no adjudication of the bars "
            "(the instrument outcome, disclosed)"),
    }[verdict]

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "INSTRUMENT-FAILS (a standing gate) -> "
                           "STABLE-PERIOD-1 (clauses a-d all hold) -> "
                           "RE-BIFURCATION (>= 2 ZERO + ZERO->dose "
                           "adjacency) -> RAIL-DECAY (any declared read "
                           "< 0.05) -> MIXED (the residue) — frozen at "
                           "birth",
        "gates_pass": bool(all(g.get("pass", True) for g in
                               metrics["gates"].values())),
        "clauses": {
            "a_no_zero_events": c_a, "b_doses_interior": c_b,
            "c_alive": c_c, "d_floor_stable": c_d,
            "re_bifurcation_signature": re_bifurcation,
            "rail_decay": rail_decay},
        "reads": {
            "the_8_events_phase_declared": {
                "classes": classes,
                "events": cyc["events"],
                "floor_rail": {"per_event_gates": gate_reads,
                               "mean": cyc["floor_rail_mean"],
                               "range": floor_range,
                               "second_half_minus_first_half": floor_drift},
                "upper_rail": {"per_event_posts": [e["post_read"]
                                                   for e in cyc["events"]],
                               "mean": cyc["upper_rail_mean"]},
                "dose_fractions_of_halved_cap": fracs,
                "vs_x41_lrcap": {
                    "classes": X41_LRCAP_CLASSES,
                    "floor_rail": X41_FLOOR_RAIL,
                    "upper_rail": X41_UPPER_RAIL,
                    "weak_rail": X41_WEAK_RAIL},
                "vs_e339_committed_cycle": {
                    "floor_rail": E339_FLOOR_RAIL,
                    "upper_rail": E339_UPPER_RAIL,
                    "weak_rail": E339_WEAK_RAIL}},
            "aliveness": {
                "floor": ALIVE_FLOOR,
                "min_declared_read": min_read,
                "at": min_read_row,
                "n_declared_reads": len(dreads)},
            "the_declared_trace": dreads,
            "the_off_target_battery_end": {
                **cells_l,
                "x41_t900_cells": x41_post_cells,
                "note": "the standing amendment's collateral check: the "
                        "gm12 offset battery + the Z channel + CE_R — "
                        "reported, never adjudicating"},
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — machinery validation only; the numbers "
                        "are meaningless (a tiny manufactured base); the "
                        "bars NOT scored" if SMOKE else None),
    }
    metrics["adjudication"]["predictions_scored"] = {
        "P-W055a_stable_period_1": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "STABLE-PERIOD-1"),
            "note": "SCORED: TRUE iff the verdict == STABLE-PERIOD-1"
                    + ("" if not SMOKE else " (not scored — smoke)")},
        "lab_lean": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "STABLE-PERIOD-1"),
            "note": "the dispatch's lean: STABLE-PERIOD-1, weakly"},
    }
    log("=" * 78)
    log(f"P6 ADJUDICATED: {verdict} — {clause}")
    write_partial("P6 ADJUDICATED (the frozen bars)")

    # ================= P7: the figure ==================================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    # (0,0) the gate reads (the floor rail trajectory) — every event PRE
    ax = axes[0][0]
    ax.axhline(E336.FACT_BASELINE_G0, color="k", ls="--", lw=1.0,
               label=f"the baseline {E336.FACT_BASELINE_G0:.4f} "
                     "(deficit = 0)")
    ax.axhline(ALIVE_FLOOR, color="red", ls=":", lw=1.4,
               label=f"aliveness {ALIVE_FLOOR}")
    ax.axhspan(E339_WEAK_RAIL - 0.026, E339_WEAK_RAIL + 0.026,
               color="orange", alpha=0.15,
               label=f"e339 weak rail {E339_WEAK_RAIL:.3f}")
    xs = [e["global_index"] for e in cyc["events"]]
    ys = [e["pre_read_gate"] for e in cyc["events"]]
    ax.plot(xs, ys, "o-", color="tab:blue", ms=6,
            label="W055 gate reads (PRE, every event)")
    for e in cyc["events"]:
        if e["class"] == "ZERO":
            ax.annotate("Z", (e["global_index"], e["pre_read_gate"]),
                        textcoords="offset points", xytext=(4, -12),
                        fontsize=8, color="tab:blue")
        elif e["class"] == "DOSE":
            ax.annotate("D", (e["global_index"], e["pre_read_gate"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=8, color="tab:blue")
        else:
            ax.annotate("m", (e["global_index"], e["pre_read_gate"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=8, color="tab:blue")
    cx = [g for g, *_ in X41_LRCAP_CYCLE]
    cy = [gate for _, _, gate, *_ in X41_LRCAP_CYCLE]
    ax.plot(cx, cy, "s", color="gray", ms=4, alpha=0.6,
            label="x41 LRCAP committed (events 33-36)")
    ax.set_xlabel("global maintenance event (the LRCAP branch)")
    ax.set_ylabel("pre-event gate read (raw)")
    ax.set_title("the floor rail trajectory (Z=zero-dose, D=dose, "
                 "m=mixed)" + (" [SMOKE]" if SMOKE else ""))
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3)

    # (0,1) the dose per event — the period-1 pattern
    ax = axes[0][1]
    ax.plot(xs, [e["lr_m"] for e in cyc["events"]], "o-",
            color="tab:green", ms=6, label="W055 dose per event")
    ax.plot(cx, [lr_ for _, _, _, _, lr_, _ in X41_LRCAP_CYCLE], "s",
            color="gray", ms=4, alpha=0.6, label="x41 LRCAP committed")
    ax.axhline(LR_M_MAX_HALF, color="k", ls="--", lw=0.9,
               label=f"the halved cap {LR_M_MAX_HALF:.5f}")
    ax.axhspan(INTERIOR_LO * LR_M_MAX_HALF, INTERIOR_HI * LR_M_MAX_HALF,
               color="tab:green", alpha=0.10,
               label=f"interior [{INTERIOR_LO}, {INTERIOR_HI}] x cap")
    ax.set_xlabel("global maintenance event (the LRCAP branch)")
    ax.set_ylabel("applied lr_m (the relay's dose)")
    ax.set_title("the dose pattern (period-1 weak dosing?)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) the post-event reads (the upper rail trajectory)
    ax = axes[1][0]
    ax.axhspan(E339_UPPER_RAIL - 0.013, E339_UPPER_RAIL + 0.013,
               color="gray", alpha=0.18,
               label=f"e339 upper rail {E339_UPPER_RAIL:.3f}")
    ax.axhspan(X41_UPPER_RAIL - 0.013, X41_UPPER_RAIL + 0.013,
               color="tab:olive", alpha=0.18,
               label=f"x41 LRCAP upper rail {X41_UPPER_RAIL:.3f}")
    pts = [(e["global_index"], e["post_read"]) for e in cyc["events"]
           if e["post_read"] is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-",
                color="tab:purple", ms=7,
                label="W055 post-dose reads (POST, every event)")
    ax.plot(cx, [post for _, _, _, _, _, post in X41_LRCAP_CYCLE], "s",
            color="gray", ms=5, alpha=0.6, label="x41 LRCAP committed")
    ax.set_xlabel("global maintenance event (the LRCAP branch)")
    ax.set_ylabel("post-event read (raw)")
    ax.set_title("the upper rail trajectory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) THE PHASE-DECLARED FULL TRACE (every read, both phases)
    ax = axes[1][1]
    ax.axhline(E336.FACT_BASELINE_G0, color="k", ls="--", lw=1.0,
               label=f"baseline {E336.FACT_BASELINE_G0:.3f}")
    ax.axhline(ALIVE_FLOOR, color="red", ls=":", lw=1.4,
               label=f"aliveness {ALIVE_FLOOR}")
    steps_all = [d["step"] for d in dreads]
    reads_all = [d["read"] for d in dreads]
    ax.plot(steps_all, reads_all, "o", color="tab:blue", ms=3,
            alpha=0.5, label="every declared read")
    for e in cyc["events"]:
        ax.plot([e["step"], e["step"]],
                [e["pre_read_gate"], e["post_read"]], "-",
                color="tab:red", lw=1.2, alpha=0.7)
    if cyc["events"]:
        ax.plot([e["step"] for e in cyc["events"]],
                [e["pre_read_gate"] for e in cyc["events"]], "o",
                color="tab:red", ms=5, label="PRE (gate)")
        ax.plot([e["step"] for e in cyc["events"]],
                [e["post_read"] for e in cyc["events"]], "^",
                color="tab:purple", ms=6, label="POST (settle)")
    ax.set_xlabel("corpus step t")
    ax.set_ylabel("the g0 read (raw)")
    ax.set_title("THE PHASE-DECLARED TRACE (METHODS 4: every event, "
                 "both phases)")
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"W055 THE RELAY TUNER'S FIRST PROBE — {verdict}"
                 + (" [SMOKE — machinery only]" if SMOKE else ""),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = RD / ("smoke_w055_relay_probe.png" if SMOKE
                     else "w055_relay_probe.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ================= P8: REPORT.md ====================================
    ts = common.now_iso()
    gcount = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    lines = [
        f"# W055 — THE RELAY TUNER'S FIRST PROBE (R79's card; W049's "
        f"graduation)\n",
        f"**Date:** {ts} (datetime.now(UTC) stamps in metrics) · "
        f"**Executor report**\n",
        "## THE VERDICT (against the frozen bars)\n",
        f"**{verdict}**"
        + (" _(smoke — machinery validation only; not scored)_"
           if SMOKE else ""),
        "\n```",
        clause,
        "```\n",
        "## The 8-event trace (phase-declared — every event, both "
        "phases)\n",
        "| global event | step | PRE gate read | deficit | dose lr_m "
        "(x cap) | class | POST read |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in cyc["events"]:
        lines.append(
            f"| {e['global_index']} | t{e['step']} | "
            f"{e['pre_read_gate']:.6f} | {e['deficit']:.4f} | "
            f"{e['lr_m']:.6f} ({e['lr_fraction_of_halved_cap']:.3f}) | "
            f"{e['class']} | "
            f"{e['post_read'] if e['post_read'] is None else round(e['post_read'], 6)} |")
    lines += [
        "",
        f"- the rails: floor mean {cyc['floor_rail_mean']:.6f} (range "
        f"{floor_range:.4f}, 2nd-half minus 1st-half "
        f"{floor_drift:+.4f}); upper mean {cyc['upper_rail_mean']:.6f}",
        f"- vs x41's LRCAP leg: classes {'/'.join(X41_LRCAP_CLASSES)}, "
        f"floor {X41_FLOOR_RAIL:.4f}, upper {X41_UPPER_RAIL:.4f}, weak "
        f"{X41_WEAK_RAIL:.4f}",
        f"- vs e339's committed 2-cycle: floor {E339_FLOOR_RAIL:.4f} / "
        f"upper {E339_UPPER_RAIL:.4f} / weak {E339_WEAK_RAIL:.4f}",
        f"- aliveness: min declared read {min_read:.6f} "
        f"(>= {ALIVE_FLOOR}: {c_c}) at '{min_read_row['declared_phase']}' "
        f"— {len(dreads)} declared reads\n",
        "## The off-target battery at the end (the standing amendment's "
        "collateral check)\n",
        f"- gm12 (offset battery, p(T)): {cells_l['gm12']:.6f} "
        f"(x41 t900: {x41_post_cells['gm12']:.6f})",
        f"- g0 Z-channel (wrong token): {cells_l['g0_z']:.2e} "
        f"(x41 t900: {x41_post_cells['g0_z']:.2e})",
        f"- CE_R (the eval bank): {cells_l['ce_r']:.4f} "
        f"(x41 t900: {x41_post_cells['ce_r']:.4f})\n",
        "## The base and the gates\n",
        f"- the base: x41's committed LRCAP leg's final state (resume md5 "
        f"{X41_LRCAP_RESUME_MD5[:8]}... — the model + both SGD-M buffers "
        f"+ both generator states at t900, the state where MIXED "
        f"appeared), bit-compared vs the committed post checkpoint, "
        f"read-reproduced |d| {G_T900_REPRO['absdiff']:.1e}",
        f"- the regime held constant: lr_m_max {lr_m!r} (byte-equal to "
        f"x41's LRCAP cap), M={m} unchanged; THE SCHEDULE EXTENSION "
        f"INST_TOTAL 1000 -> {SCHED_EXT_TOTAL} (disclosed; G_SCHED "
        f"verifies the cap stayed the only binding corpus constraint, "
        f"min margin {min(margins.values()):.2f}x)",
        f"- the net0 class: {NET0_CLASS['name']} "
        f"({NET0_CLASS['params']:,} params)",
        f"- {gcount} gate classes instantiated, "
        f"{'all PASS' if gcount == len(metrics['gates']) else 'SEE metrics'}: "
        + ", ".join(metrics["gates"]),
        "",
        "## Envelope\n",
        f"- {len(thermal_log)} thermal polls; max temp "
        f"{max((r.get('temp', 0) for r in thermal_log), default=0):.1f}C; "
        f"burst cap {E261.BURST_MAX_S:.0f}s; cooldowns "
        f"{E261.COOLDOWN_S:.0f}s between bursts (inside the driver); "
        f"zero concurrent GPU jobs; every burst logged to "
        "runs/_envelope_log.jsonl tagged w055:CONT",
        "",
        "## Registered predictions\n",
        f"- P-W055a (the executor's own read: STABLE-PERIOD-1, "
        f"moderately): "
        f"{'HIT' if verdict == 'STABLE-PERIOD-1' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        f"- the dispatch's lab lean (STABLE-PERIOD-1, weakly): "
        f"{'HIT' if verdict == 'STABLE-PERIOD-1' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        "",
        "## Catches / disclosures\n",
        "- THE SCHEDULE EXTENSION: the house cosine clamps to 0 at "
        "t >= 1000; a verbatim continuation would have silently KILLED "
        "the corpus wash for the last four events (RE-BIFURCATION-by-"
        "schedule, not by the relay). E261.INST_TOTAL re-bound 1000 -> "
        f"{SCHED_EXT_TOTAL}, asserted + restored; the budget cap stayed "
        "the ONLY binding corpus constraint (G_SCHED) — the operative "
        "wash variable held constant.",
        "- THE PHASE-DECLARATION (METHODS 4) is enacted in the rig: "
        "MILESTONES == ALL event steps (every event read post-dose) + "
        "the controller's own gate read (pre-dose) + window brackets; "
        "the milestone-costume failure mode (a cadence locked to one "
        "phase) is impossible by construction.",
        "- NO TWIN, BY DESIGN: the regime under test is an ERROR-GATED-"
        "arm property (the sanctuary twin has no maintenance events).",
        "- n=1, ONE organism, one lineage, ONE branch (the LRCAP "
        "branch's own deterministic continuation — not a draw); the "
        "fixed point is one slot's realization.",
        "- The off-target battery is REPORTED, never adjudicating (the "
        "standing amendment's collateral check).",
        "- Checkpoints live INSIDE runs/w055/. No NOTES/THINKING/QUEUE/"
        "STATE edits (the heartbeat folds this cell).",
        "",
        "*This cell does not edit NOTES/THINKING/QUEUE/STATE — the "
        "heartbeat folds.*",
    ]
    (RD / ("smoke_REPORT.md" if SMOKE else "REPORT.md")).write_text(
        "\n".join(lines), encoding="utf-8")

    # ================= P9: the final metrics ===========================
    metrics["leg"] = {
        "spec": {"steps": LEG_STEPS, "m_every": m, "lr_m_max": lr_m,
                 "desc": "the LRCAP-final state continued 8 events at "
                         "the SAME halved cap (the regime held constant)"},
        "phase": {k: res[k] for k in
                  ("traj", "corpus_ledger", "lr_ledger", "maint_ledger",
                   "window_ledger", "bufsep", "chunk_table", "steps_ran",
                   "budget_ledger", "orth_ledger", "disp_ledger",
                   "buf_ledger")},
        "cycle": cyc,
        "post_cells": cells_l,
        "S_corpus": float(res["S_corpus"]),
        "S_maint": float(res["S_maint"]),
        "budget_c": budget_c, "budget_m": budget_m,
        "orth_max": res["orth_max"],
        "checkpoint": str(RD / ("smoke_w055_CONT_REA_post.pt" if SMOKE
                                else "w055_CONT_REA_post.pt")),
    }
    metrics["honesty"] = {
        "intervention_not_logits": "the leg resumes x41's committed LRCAP "
                                   "final state (md5 + repro gated) and "
                                   "continues that branch's own stream "
                                   "(bit-identical draws); the ONLY "
                                   "deltas are the length-scaled pools' "
                                   "horizon and the disclosed schedule "
                                   "extension — the regime's own "
                                   "constants (the halved cap, M) are "
                                   "byte-equal UNCHANGED",
        "the_phase_declaration": "every event read at BOTH phases "
                                 "(the controller's gate + the post-dose "
                                 "milestone); aliveness over every "
                                 "declared read — the milestone-costume "
                                 "failure mode impossible by construction",
        "n_and_scope": "n=1, ONE organism, one lineage, one branch; the "
                       "continuation is deterministic (the branch's own "
                       "generator states resumed, never re-seeded)",
        "loads_measured_not_nominal": "every read measured: per-event "
                                      "gate reads, doses, realized "
                                      "displacements, post-event reads, "
                                      "thermal polls",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": [
            "lab/e336_founding_replicate.py (the committed driver, "
            "executed verbatim under the disclosed leg re-binding + the "
            "schedule extension)",
            "lab/e261_rank_ladder.py (the burst machinery + rooms)",
        ],
        "machinery": {
            "leg_driver": "e336's chunked_errorgated_phase EXECUTED "
                          "VERBATIM under the disclosed re-binding "
                          "(PHASE_STEPS/MILESTONES/WIN1/WIN2/MAINT_EVERY "
                          "+ E261.INST_TOTAL 1000->1250 restored after) "
                          "+ the lr_m_max argument == x41's LRCAP cap",
            "staging": "THIS file's staged_leg_state (the only "
                       "protocol-touching transformation; "
                       "digest-verified)",
            "cons": "NONE (the bars read the dose pattern + rails only)",
            "twin": "NONE (by design — disclosed)",
        },
        "checkpoints": {
            "resumed_from": [str(base_src)],
            "x41_metrics": str(X41_METRICS),
            "leg_resume": str(leg_ck),
            "base": str(base_ck),
        },
        "thermal_envelope": {
            "polls": len(thermal_log),
            "max_temp_c": max((r.get("temp", 0) for r in thermal_log),
                              default=None),
            "temp_early_end_margin_c": E261.TEMP_EARLY_END,
            "temp_hard_line_c": E261.TEMP_HARD,
            "cooldown_s": E261.COOLDOWN_S,
            "burst_cap_s": E261.BURST_MAX_S,
            "concurrent_gpu_jobs": 0,
            "note": "every burst logged to runs/_envelope_log.jsonl "
                    "tagged w055:CONT",
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["outputs"] = [str(fig_path), str(RD / "REPORT.md"),
                          str(RD / "metrics.json")]
    metrics["status"] = f"COMPLETE — adjudicated {verdict}" \
                        + (" (SMOKE)" if SMOKE else "") \
                        + f" ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)
    log(f"P9 DONE: {fig_path.name} + REPORT.md + metrics.json — "
        f"verdict {verdict}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        _logf.write(traceback.format_exc() + "\n")
        _logf.flush()
        metrics["status"] = f"FAILED ({common.now_iso()})"
        save_json(RD / "metrics.json", metrics)
        raise
