# -*- coding: utf-8 -*-
"""X41 — THE RELAY'S KINETICS (R78's card; W049's sharpened dial question).

Run:  cd lab && python x41_relay_kinetics.py       (X41_SMOKE=1 shakedown)

THE BACKGROUND (verbatim from the dispatch letter):
    e339 discovered the controller's sawtooth BIFURCATED INTO A 2-CYCLE
    (from event 25: the dose fully self-limits every other event —
    lr_m -> 0; pre-event reads alternate weak 0.09-0.15 / strong
    0.21-0.39; the 'equilibrium' is the limit-cycle's average). W049's
    dial question is now a RELAY question: is the 2-cycle ENTRAINABLE
    (it tracks a gate parameter — a tuner can shift it) or STRUCTURAL
    (it is the slot's own dynamics)?

THE QUESTION (verbatim): "is the 2-cycle ENTRAINABLE (it tracks a gate
    parameter — a tuner can shift it) or STRUCTURAL (it is the slot's
    own dynamics)?"

THE DESIGN (verbatim from the dispatch letter):
    1. THE BASE: e339's committed t800 state (md5-bound — the state at
       which the 2-cycle is established).
    2. THE PERTURBATIONS (three short legs, ~100 steps each — enough to
       see the cycle re-form): (a) EVENT PERIOD doubled (the maintenance
       events spaced 2x — does the cycle's period track?); (b) LR_M CAP
       halved (the dose ceiling lowered — does the strong phase weaken
       or the cycle break?); (c) the control leg (unperturbed
       continuation — the cycle must reproduce).
    3. THE VERIFICATION RIDER (CPU, free): from e339's committed trace,
       verify numerically that 'the equilibrium is the limit-cycle's
       average' (the milestone reads vs the two phases' means).

THE BARS (frozen VERBATIM from the dispatch letter, BEFORE any compute):
    ENTRAINABLE: at least one perturbation moves the cycle's observable
        structure (the period tracks the event spacing, or the phase
        amplitude tracks the lr cap) — THE CYCLE IS TUNABLE; W049's
        dial designs a relay tuner; the calibration program gains its
        kinetics.
    STRUCTURAL: the cycle's structure is invariant under both
        perturbations (the same period/amplitudes within noise) — the
        2-cycle is the slot's own dynamics; the dial is bounded by it.
    Register P-x41a BEFORE compute. Lab lean: ENTRAINABLE, weakly — the
    cycle is driven by the gate's own self-limiting (lr_m -> 0 on the
    weak phase), so spacing the events should space the cycle; the
    countervailing: the slot's recovery dynamics may set their own
    rhythm. State your own read.

MACHINERY: e336's COMMITTED driver (chunked_errorgated_phase) EXECUTED
VERBATIM BY IMPORT under the family's disclosed re-binding convention
(e339's form): the ONLY module globals re-bound are the leg parameters
(PHASE_STEPS, MILESTONES, WIN1/WIN2, MAINT_EVERY) + this leg's lr_m_max
ARGUMENT (the controller-law constant — the LRCAP leg's single delta);
every other protocol constant (LR_STABLE, LR_M_MAX_FROZEN, seeds,
baseline, momentum, room, budget fractions) is read from the committed
module and asserted against e339's md5-bound metrics BEFORE any compute.
The base is e339's committed t800 resume state (runs/e339/
e339_REA_resume.pt: the model + BOTH SGD-M buffer states + BOTH
generator states at t800, md5-bound, bit-compared vs the committed post
checkpoint, read-reproduced) — staged into this cell's base with the leg
pools zeroed (e339's staging convention, digest-verified). The stream
CONTINUES (generator states restored, never re-seeded): corpus steps
801.. draw the continuation of seed 33601's sequence; x41's maintenance
events draw the continuation of seed 33602's (global event = 32 + the
leg's own index — the staging zeroes n_maint, the family convention).

THE THREE LEGS (frozen):
    LEG-CONTROL  t801..t900 (100 steps), M=25, lr_m_max = the frozen
                 constant; events at t825/850/875/900 (global 33-36;
                 expected ZERO/DOSE/ZERO/DOSE under the committed
                 parity: the t800 event was a DOSE).
    LEG-PERIOD2X t801..t950 (150 steps), M=50, lr_m_max unchanged;
                 events at t850/900/950 (global 33-35).
    LEG-LRCAP    t801..t900 (100 steps), M=25, lr_m_max = 0.5 x the
                 frozen constant (the dose ceiling halved); events
                 t825/850/875/900.

THE POOL CONVENTION (frozen at birth): the legs' budget pools SCALED TO
THE LEG'S OWN LENGTH (e336's smoke arithmetic applied to a full leg):
B_C = 2.6848887180471723 x leg_steps/400, B_M = 1.7899258120314483 x
leg_events/16 — the PER-STEP and PER-EVENT equal-share reservations are
then IDENTICAL to e339's committed leg-2 (the wash kinetics carried by
the cap are protocol-matched; asserted via the applied-lr ledger).
LR_M_MAX itself is NEVER scaled (it is the controller-law constant; the
LRCAP leg halves it as its single perturbation). REJECTED AT BIRTH: the
unscaled full-leg pool (a 100-step leg with the 400-step B_C quadruples
the per-step share -> a different wash protocol — a confound, not a
perturbation).

THE SCHEDULE DISCLOSURE (leg-A, frozen): the house cosine (warmup 100,
total 1000) decays below the corpus cap (~0.0034) from ~step 930 —
LEG-PERIOD2X's third event (t950) sits in the schedule tail (a weaker
wash); its primary discriminating events (t850, t900) are cap-bound.
Per-step cap-vs-sched is recorded per leg (the lr ledger); CONTROL and
LRCAP (ending t900) assert cap-below-sched at every logged step.

PHASE CLASSIFICATION (frozen): ZERO-phase event := deficit_t == 0.0
(the gate read at/above the baseline — lr_m == 0 exactly, the dose
self-limits); DOSE-phase event := deficit_t >= 0.40; MIXED := anything
else (the establishment regime's small doses). THE COMMITTED CYCLE
(e339 leg-2 events 9-16, global 25-32): strict ZERO/DOSE alternation;
floor rail (ZERO-phase gate mean) 0.37345; upper rail (DOSE-phase
post-event mean, the t700/t800 milestones) 0.88001; weak rail (DOSE-
phase gate mean) 0.12449.

THE COMPOSITE (frozen at birth, in order):
    CONTROL-FAILS   the control leg's classes != [ZERO, DOSE, ZERO,
                    DOSE] -> no adjudication (the instrument outcome,
                    disclosed).
    ENTRAINABLE     any MOVE observable fires:
                    (M1) PERIOD-RELOCK: leg-A has ZERO zero-phase
                    events AND >= 2 DOSE events (every event doses at
                    the doubled spacing — the cycle re-locked to the
                    event cadence);
                    (M2) AMPLITUDE-MOVE: a leg's rail (floor = ZERO-
                    phase gate mean; upper = DOSE-phase post-event
                    mean) differs from the CONTROL leg's same rail by
                    > PHASE_MARGIN = 0.08 (raw read units — 2.4x the
                    committed intra-phase spread 0.033, ~1/3 of the
                    phase gap 0.25);
                    (M3) CLASS-FLIP: a leg's class sequence differs
                    from the CONTROL's same-index classes.
    STRUCTURAL      no MOVE observable fires AND the control reproduces
                    AND leg-A still alternates (>= 1 ZERO and >= 1 DOSE
                    event — the 2-event period survives the doubled
                    spacing).
    MIXED           everything else (e.g. leg-A MIXED-only with leg-B
                    invariant — the structure moved ambiguously; full
                    disclosure in the clause).

THE VERIFICATION RIDER'S MINI-BARS (frozen at birth; committed trace
only, no compute): from e339's established cycle (global events 25-32):
    R1 CONFIRMED-STRICT  |plateau - mean(the two phases' post-event
       means)| <= 0.05, plateau := the committed t800 milestone read
       (0.8732402324676514 = the PLATEAUS-BELOW settle).
    R2 REFINED-UPPER-RAIL   R1 fails AND |plateau - the DOSE-phase
       post-event mean| <= 0.05 (the milestone cadence phase-locked to
       the cycle's dose phase — the 'equilibrium' is the upper rail,
       not the average).
    R3 (descriptive, never adjudicating): the cycle's time-average
       reconstructed piecewise-geometrically between the committed
       anchors over t700->t800 (the t750 dose-jump post bracketed by
       the neighbouring committed posts, disclosed) — reported against
       the plateau.

THE HONESTY NOTE ON THE LEAN'S WORDING (a birth catch, disclosed): the
dispatch's lean says "lr_m -> 0 on the weak phase"; the committed trace
puts the zero doses on the STRONG-read phase (the pre-read exceeds the
baseline 0.285851389169693 -> the deficit clamps to 0 -> lr_m == 0).
The lean is frozen VERBATIM regardless; the operationalizations carry
the read-level fact.
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
import e339_long_landing as E339                       # noqa: E402 — THE
                                                      # t800 PARENT (import
                                                      # runs no main; writes
                                                      # nothing)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SMOKE = os.environ.get("X41_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "x41_smoke" if SMOKE else "x41"
assert torch.cuda.is_available(), "x41 owns the GPU lane (dispatch)"
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
# modules' logs + burst machinery label THIS cell (tags x41:...); nothing
# of e336's/e339's is written (their run.log handles are re-pointed here).
E336._logf = _logf
E336.T0 = T0
E336.NAME = NAME
E339._logf = _logf
E339.T0 = T0
E339.NAME = NAME
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
# modules + e339's md5-bound record (asserted at runtime)
# ======================================================================
E339_DIR = E43.REPO / "runs" / "e339"
E339_REA_RESUME_CK = E339_DIR / "e339_REA_resume.pt"
E339_REA_POST_CK = E339_DIR / "e339_REA_post.pt"
E339_METRICS = E339_DIR / "metrics.json"

# e339's committed artifacts — raw md5s, VERIFIED AT BIRTH (2026-10-10,
# git head c9e5c6f); a drift FAILS the resume gate.
E339_REA_RESUME_MD5 = "134034c5b53a39edfe36ee8710e2381a"
E339_REA_POST_MD5 = "e54df68360c2540d825fc0fb12806bc7"
# [pre-compute repair, disclosed] the birth commit froze a mistranscribed
# metrics md5 (2b6555ede... for 2b6559ede...); repaired at the first smoke
# launch BEFORE any compute; the on-disk artifact verified unmodified vs
# git HEAD (the resume + post literals were correct at birth)
E339_METRICS_MD5 = "00e5f9822b6559ede2ab0ee9987c5121"

# e339's committed t800 literals (the repro targets + the cycle's
# committed signature; read from the md5-bound record at runtime and
# asserted == these)
E339_REA_T800_G0 = 0.8732402324676514
E339_REA_T800_RATIO = 3.054874894972998
E339_LEG2_S_CORPUS = 2.684888717252761
E339_LEG2_S_MAINT = 0.8751672571524978
E339_MILESTONE_G0 = {500: 0.8529938459396362, 600: 0.8999966382980347,
                     700: 0.8867747783660889, 800: 0.8732402324676514}
# the established cycle (e339 leg-2 events 9-16 == global 25-32):
# (global_event, step, gate_read, deficit, lr_m)
E339_CYCLE = [
    (25, 625, 0.3541751801967621, 0.0, 0.0),
    (26, 650, 0.09832171350717545, 0.6560390565434416, 0.014395113389485957),
    (27, 675, 0.3844239115715027, 0.0, 0.0),
    (28, 700, 0.1165490597486496, 0.5922739431591101, 0.01299594968374994),
    (29, 725, 0.3873353600502014, 0.0, 0.0),
    (30, 750, 0.13392327725887299, 0.5314933481769064, 0.011662273665649476),
    (31, 775, 0.36786726117134094, 0.0, 0.0),
    (32, 800, 0.14915722608566284, 0.4782001006924719, 0.010492888500573451),
]
E339_FLOOR_RAIL = 0.3734504304       # ZERO-phase gate mean (events 25-32 odd)
E339_UPPER_RAIL = 0.8800075054       # DOSE-phase post mean (t700/t800)
E339_WEAK_RAIL = 0.1244877942        # DOSE-phase gate mean
# e339 leg-2's applied corpus lr (the protocol-identity band for x41's
# legs: e339's own min/median/max, widened 20%)
LR_BAND = (0.0018, 0.0043)

# ---- the base organism class (recorded per the dispatch's gate) -------
NET0_CLASS = {
    "name": "the lineage's char-transformer base (e001.pt)",
    "checkpoint": "runs/checkpoints/e001.pt",
    "params": 2739072,
    "family": "the 2.74M pre-LN char transformer (e311's TAVIREN organism "
              "built on it; e336/e339's controller ran on that organism)",
}

# ---- the leg table (frozen) -------------------------------------------
B_C_400 = 2.6848887180471723          # e339's committed per-400-step pools
B_M_400 = 1.7899258120314483
LR_M_MAX_FROZEN = E336.LR_M_MAX_FROZEN   # 0.021942464013242388 (asserted)
LR_M_MAX_HALF = 0.5 * LR_M_MAX_FROZEN
LEG_BASE_STEP = 800                   # e339's committed t800

SMOKE_BASE_STEPS = 12                 # the smoke base (fresh, tiny)
SMOKE_BASE_M = 2

LEGS = {
    "CONTROL": dict(
        steps=8 if SMOKE else 100, m_every=2 if SMOKE else 25,
        lr_m_max=LR_M_MAX_FROZEN,
        desc="the unperturbed continuation — the committed cycle must "
             "reproduce (events global 33-36 expected ZERO/DOSE/ZERO/DOSE)",
    ),
    "PERIOD2X": dict(
        steps=8 if SMOKE else 150, m_every=4 if SMOKE else 50,
        lr_m_max=LR_M_MAX_FROZEN,
        desc="EVENT PERIOD doubled (M 25 -> 50): does the cycle's period "
             "track the event spacing? (3 events at t850/900/950)",
    ),
    "LRCAP": dict(
        steps=8 if SMOKE else 100, m_every=2 if SMOKE else 25,
        lr_m_max=LR_M_MAX_HALF,
        desc="LR_M CAP halved (the dose ceiling 0.02194 -> 0.01097): does "
             "the strong phase weaken or the cycle break? (events "
             "t825/850/875/900)",
    ),
}
LEG_ORDER = ("CONTROL", "PERIOD2X", "LRCAP")

# ---- the bars' numbers (frozen) ----------------------------------------
PHASE_MARGIN = 0.08        # the rail-move bar (raw read units)
DOSE_DEFICIT_MIN = 0.40    # the DOSE-phase classification floor
RIDER_TOL = 0.05           # the rider's equation tolerance
STREAM_STABLE_TOL = 1.05
ORTH_BAR = 1e-6
BUFSEP_ORTH_BAR = 1e-4
BUDGET_SLACK = 1e-5
SCHED_TAIL_FROM = 929      # the house cosine crosses below the cap ~here

REGISTERED = {
    "registration": (
        "question + design + bars + lab lean frozen VERBATIM from the "
        "dispatch letter (R78's card — x41 THE RELAY'S KINETICS; W049's "
        "sharpened dial question); this script committed at birth BEFORE "
        "any compute; P-x41a registered below BEFORE compute; adjudicate "
        "against exactly this; no bar shopping."),
    "question_verbatim": (
        "e339 discovered the controller's sawtooth BIFURCATED INTO A "
        "2-CYCLE (from event 25: the dose fully self-limits every other "
        "event — lr_m -> 0; pre-event reads alternate weak 0.09-0.15 / "
        "strong 0.21-0.39; the 'equilibrium' is the limit-cycle's "
        "average). W049's dial question is now a RELAY question: is the "
        "2-cycle ENTRAINABLE (it tracks a gate parameter — a tuner can "
        "shift it) or STRUCTURAL (it is the slot's own dynamics)?"),
    "design_verbatim": (
        "1. THE BASE: e339's committed t800 state (md5-bound — the state "
        "at which the 2-cycle is established). 2. THE PERTURBATIONS "
        "(three short legs, ~100 steps each — enough to see the cycle "
        "re-form): (a) EVENT PERIOD doubled (the maintenance events "
        "spaced 2x — does the cycle's period track?); (b) LR_M CAP "
        "halved (the dose ceiling lowered — does the strong phase weaken "
        "or the cycle break?); (c) the control leg (unperturbed "
        "continuation — the cycle must reproduce). 3. THE VERIFICATION "
        "RIDER (CPU, free): from e339's committed trace, verify "
        "numerically that 'the equilibrium is the limit-cycle's average' "
        "(the milestone reads vs the two phases' means)."),
    "bars_verbatim": {
        "ENTRAINABLE": (
            "ENTRAINABLE: at least one perturbation moves the cycle's "
            "observable structure (the period tracks the event spacing, "
            "or the phase amplitude tracks the lr cap) — THE CYCLE IS "
            "TUNABLE; W049's dial designs a relay tuner; the calibration "
            "program gains its kinetics."),
        "STRUCTURAL": (
            "STRUCTURAL: the cycle's structure is invariant under both "
            "perturbations (the same period/amplitudes within noise) — "
            "the 2-cycle is the slot's own dynamics; the dial is bounded "
            "by it."),
    },
    "lab_lean_verbatim": (
        "Lab lean: ENTRAINABLE, weakly — the cycle is driven by the "
        "gate's own self-limiting (lr_m -> 0 on the weak phase), so "
        "spacing the events should space the cycle; the countervailing: "
        "the slot's recovery dynamics may set their own rhythm. State "
        "your own read."),
    "operationalizations": (
        f"frozen BEFORE compute: THE BASE := e339's committed "
        f"runs/e339/e339_REA_resume.pt (raw-md5 {E339_REA_RESUME_MD5}; "
        f"the model + both SGD-M buffers + both generator states at "
        f"t800), bit-compared vs the committed post checkpoint and "
        f"read-reproduced (the t800 g0 read {E339_REA_T800_G0!r} within "
        f"the family's 2e-6 law) BEFORE any stepping; THE LEGS := three "
        f"independent branches from ONE staged base (state bits through "
        f"UNTOUCHED, digest-verified; leg pools zeroed; the stream "
        f"CONTINUES — generators restored, never re-seeded; the legs "
        f"share their draws by construction); LEG-CONTROL t801-900 M=25 "
        f"lr_m_max {LR_M_MAX_FROZEN!r}; LEG-PERIOD2X t801-950 M=50 "
        f"lr_m_max unchanged; LEG-LRCAP t801-900 M=25 lr_m_max "
        f"{LR_M_MAX_HALF!r} (the ONLY re-bound controller constant in "
        f"the whole cell); THE POOL CONVENTION := pools scaled to the "
        f"leg's own length (B_C = {B_C_400!r} x steps/400, B_M = "
        f"{B_M_400!r} x events/16 — the per-step/per-event equal shares "
        f"then IDENTICAL to e339's committed leg-2; the unscaled pool "
        f"REJECTED at birth as a wash-protocol confound); LR_M_MAX "
        f"never scaled except the LRCAP perturbation itself; THE "
        f"SCHEDULE := the house cosine VERBATIM on the global step — "
        f"DISCLOSED: from ~step {SCHED_TAIL_FROM} the schedule decays "
        f"below the corpus cap, so LEG-PERIOD2X's t950 event sits in a "
        f"weaker-wash tail (its primary events t850/t900 cap-bound; "
        f"CONTROL/LRCAP assert cap-below-sched at every logged step); "
        f"THE CYCLE'S OBSERVABLE STRUCTURE := (the per-event phase "
        f"classes, the rails) with ZERO-phase := deficit_t == 0.0 (lr_m "
        f"== 0 exactly), DOSE-phase := deficit_t >= {DOSE_DEFICIT_MIN}, "
        f"MIXED := the rest; floor rail := the ZERO-phase gate-read "
        f"mean; upper rail := the DOSE-phase post-event read mean; "
        f"THE COMMITTED CYCLE (the comparison key, from e339's "
        f"md5-bound trace, global events 25-32): strict ZERO/DOSE "
        f"alternation, floor rail {E339_FLOOR_RAIL!r}, upper rail "
        f"{E339_UPPER_RAIL!r}, weak rail {E339_WEAK_RAIL!r}; "
        f"CONTROL-REPRODUCES := the control leg's four classes == "
        f"[ZERO, DOSE, ZERO, DOSE]; THE MOVE OBSERVABLES: (M1) "
        f"PERIOD-RELOCK := LEG-PERIOD2X has ZERO zero-phase events AND "
        f">= 2 DOSE events; (M2) AMPLITUDE-MOVE := any leg's rail "
        f"differs from the CONTROL's same rail by > {PHASE_MARGIN}; "
        f"(M3) CLASS-FLIP := a leg's class sequence differs from the "
        f"CONTROL's same-index classes; COMPOSITE := CONTROL-FAILS (the "
        f"control does not reproduce -> no adjudication) -> ENTRAINABLE "
        f"(M1 or M2 or M3) -> STRUCTURAL (no move observable AND "
        f"LEG-PERIOD2X still alternates: >= 1 ZERO and >= 1 DOSE) -> "
        f"MIXED (everything else, fully disclosed); THE RIDER (CPU, "
        f"committed trace only): R1 CONFIRMED-STRICT iff "
        f"|plateau - mean(the two phases' post-event means)| <= "
        f"{RIDER_TOL} (plateau := the committed t800 milestone "
        f"{E339_REA_T800_G0!r}); R2 REFINED-UPPER-RAIL iff R1 fails and "
        f"|plateau - the DOSE-phase post-event mean| <= {RIDER_TOL}; R3 "
        f"descriptive: the piecewise-geometric time-average over the "
        f"committed t700->t800 anchors (the t750 post bracketed by its "
        f"neighbours, disclosed) — reported, never adjudicating; THE "
        f"TWIN := omitted BY DESIGN (the 2-cycle is an ERROR-GATED-arm "
        f"property; the sanctuary twin has no maintenance events and no "
        f"cycle — the family's x-ratio co-read is not applicable; "
        f"disclosed); HARD GATES := G_NAMEWIN, G_NAMEFREE, G_BASE, "
        f"G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, G_SUBSTRATE, "
        f"G_E339BIND, G_T800_REPRO, G_STAGING, G_RIDER, G_LEG_CONTROL, "
        f"G_LEG_PERIOD2X, G_LEG_LRCAP, G_CYCLE_REPRO, G_BUFSEP, "
        f"G_SCHED, G_ORTH, G_BUDGET, G_STREAMELIVE; NON-HALTING := "
        f"{{G_ORTH, G_BUDGET, G_STREAMELIVE}}; the driver is e336's "
        f"COMMITTED body executed VERBATIM under the disclosed leg "
        f"re-binding; every leg's final state checkpointed."),
    "predictions": {
        "P-x41a_entrainable": (
            "THE EXECUTOR'S OWN READ (per the dispatch's 'State your own "
            "read'): ENTRAINABLE, via BOTH legs — (1) LEG-PERIOD2X "
            "RE-LOCKS (M1): the committed event-free decay (post-dose "
            "0.90 -> 0.37 at +25 -> 0.098 at +50; the baseline 0.286 "
            "crossed ~step +32) puts the +50-after-dose gate read FAR "
            "below the baseline, so under the strobe picture every "
            "M=50 event doses — the alternation dissolves into "
            "every-event dosing, the cycle re-locked to the event "
            "cadence; a ZERO at t850 would require the read to hold or "
            "re-elevate above the baseline with NO intervening dose — "
            "unobserved anywhere in the committed trace; (2) LEG-LRCAP "
            "MOVES THE RAILS (M2): the dose-response of the committed "
            "jumps (lr 0.0057 -> 3.0x; lr 0.0105 -> 5.9x; lr 0.0130 -> "
            "7.6x) says halving the ceiling drops the t850 post-dose "
            "peak by > the 0.08 margin and the t875 floor follows (25 "
            "steps after a weaker dose); COUNTERVAILING (the structural "
            "side, honestly registered): if the slot's read pathway has "
            "its own sub-baseline recovery ring (the name-write's "
            "residence re-elevating the read without a dose), the t850 "
            "gate read could sit at/above the baseline -> ZERO -> the "
            "structural side; DISCRIMINATING DATA: leg-A's t850 gate "
            "read (the first-ever +50-after-dose event read) and "
            "leg-B's t850 post-event read. SCORED: TRUE iff the verdict "
            "== ENTRAINABLE."),
        "P-x41a_rider": (
            "THE RIDER'S OWN PREDICTION (registered BEFORE compute): "
            "REFINED-UPPER-RAIL (R2) — the committed milestone cadence "
            "(every 4th event, the same parity as the dose phase) is "
            "phase-locked to the cycle's upper rail, so the 'equilibrium' "
            "= the upper rail (|plateau - upper-rail mean| ~ 0.007), NOT "
            "the two-phase average (~0.627 — 0.25 away); T310's sentence "
            "'the equilibrium is the cycle's average' is expected to "
            "REFINE, not confirm. SCORED: TRUE iff the rider verdict == "
            "REFINED-UPPER-RAIL."),
    },
}
G_NAMEWIN_KEYSET = None  # retired (inlined into the operationalizations)

deviations: list[str] = [
    "PRE-COMPUTE REPAIR (before any compute, disclosed): the birth "
    "commit's E339_METRICS_MD5 literal was mistranscribed "
    "(2b6555ede... vs the true 2b6559ede...); caught by the md5 gate "
    "at the first smoke launch; the artifact itself verified "
    "UNMODIFIED vs git HEAD; the resume + post literals were correct "
    "at birth",
]
builds_on = [
    "R78's ideator card (the dispatch: 'THE RELAY'S KINETICS + THE R* "
    "INTERIOR (resume e339's t800 with one gate parameter perturbed: "
    "ENTRAINABLE vs STRUCTURAL)')",
    "e339 / THE LONG LANDING (THE PARENT CELL: its committed t800 resume "
    "state resumed here; its 2-cycle discovery the object; its "
    "staging + leg conventions carried verbatim)",
    "T310 (the limit-cycle reading of e339 — the 'equilibrium is the "
    "cycle's average' sentence the rider tests numerically)",
    "W049 (the calibration-dial question the verdict arms or bounds)",
    "e336 / THE FOUNDING-CLASS SECOND INSTANCE (the driver, executed "
    "verbatim), e288 (the ERROR-GATED controller rig), e287 (the b_m "
    "calibration + LR_M_MAX identity), e311 (the TAVIREN organism), "
    "e273 (LR_STABLE), e261 (the room + the burst machinery)",
]
whats_new = [
    "THE FIRST PERTURBATION OF THE CONTROLLER'S GATE PARAMETERS: every "
    "controller run in the record (e288 -> e339) held MAINT_EVERY and "
    "LR_M_MAX fixed — x41 moves them for the first time, on the "
    "committed t800 state where the 2-cycle is established",
    "THE ENTRAINMENT QUESTION: relay (the cycle strobed by the event "
    "cadence — tunable, W049's dial gains a kinetics) vs structural "
    "(the slot's own rhythm — the dial is bounded by it)",
    "THE FIRST +50-AFTER-DOSE EVENT READ (leg-A's t850 gate read — the "
    "committed trace's event-free decay says 0.098; the event read "
    "settles the strobe-vs-ring question directly)",
    "THE VERIFICATION RIDER: T310's 'the equilibrium is the limit-"
    "cycle's average' tested numerically against the committed trace "
    "for the first time",
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
# THE LEG RE-BINDING MACHINE — the ONLY e336-module globals this cell
# touches; asserted: same set in, same set out, nothing else changed
# ======================================================================
LEG_KEYS = ("PHASE_STEPS", "MILESTONES", "WIN1", "WIN2", "MAINT_EVERY")


def patch_leg_globals(n_steps: int, milestones, win1, win2,
                      maint_every: int) -> dict:
    before = {k: getattr(E336, k) for k in LEG_KEYS}
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


# ======================================================================
# THE STAGING MACHINE — e339's committed t800 resume -> this cell's base:
# the state bits (model/optC/optF/cgen/igen) pass through UNTOUCHED
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
    """e339's resume state -> the leg-staged form (pools zeroed, bits kept)."""
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
# THE CYCLE READER — phase classes + rails from a leg's maint/traj records
# ======================================================================
def classify_events(maint_rows: list[dict], traj_rows: list[dict]) -> dict:
    post_by_step = {int(r["step"]): float(r["g0_pz"])
                    for r in traj_rows}
    ev = []
    for r in maint_rows:
        d = float(r["deficit_t"])
        cls = "ZERO" if d == 0.0 else ("DOSE" if d >= DOSE_DEFICIT_MIN
                                       else "MIXED")
        ev.append({
            "leg_index": int(r["maint_index"]),
            "global_index": 32 + int(r["maint_index"]),
            "step": int(r["step"]),
            "gate_read": float(r["read_at_gate"]),
            "deficit": d, "lr_m": float(r["lr_maint"]),
            "binder": r.get("binder"),
            "realized": float(r["realized_step_norm"]),
            "post_read": post_by_step.get(int(r["step"])),
            "class": cls})
    zero_gates = [e["gate_read"] for e in ev if e["class"] == "ZERO"]
    dose_posts = [e["post_read"] for e in ev
                  if e["class"] == "DOSE" and e["post_read"] is not None]
    dose_gates = [e["gate_read"] for e in ev if e["class"] == "DOSE"]
    classes = [e["class"] for e in ev]
    # strict alternation: ZERO and DOSE both present, no MIXED, and no
    # two consecutive same-class events
    strict_alt = bool(
        "ZERO" in classes and "DOSE" in classes and "MIXED" not in classes
        and all(classes[i] != classes[i - 1]
                for i in range(1, len(classes))))
    return {
        "events": ev, "classes": classes,
        "n_zero": classes.count("ZERO"), "n_dose": classes.count("DOSE"),
        "n_mixed": classes.count("MIXED"),
        "strict_alternation": strict_alt,
        "floor_rail": (sum(zero_gates) / len(zero_gates)
                       if zero_gates else None),
        "upper_rail": (sum(dose_posts) / len(dose_posts)
                       if dose_posts else None),
        "weak_rail": (sum(dose_gates) / len(dose_gates)
                      if dose_gates else None),
    }


# ======================================================================
# THE VERIFICATION RIDER (CPU; e339's committed trace only)
# ======================================================================
def run_rider(e339m: dict) -> dict:
    ph = e339m["arms"]["ERROR-GATED"]["phase"]
    ml = ph["maint_ledger"]
    traj = {int(r["step"]): float(r["g0_pz"]) for r in ph["traj"]}
    # the established cycle: leg-2 events 9-16 (global 25-32)
    est = ml[8:16]
    zero_gates, dose_gates, dose_posts = [], [], []
    for r in est:
        if float(r["deficit_t"]) == 0.0:
            zero_gates.append(float(r["read_at_gate"]))
            # no step taken: the ZERO-phase post-event read == the gate read
        else:
            dose_gates.append(float(r["read_at_gate"]))
            if int(r["step"]) in traj:
                dose_posts.append(traj[int(r["step"])])
    zero_posts = zero_gates[:]           # post == pre at a zero-dose event
    plateau = traj[800]                  # the PLATEAUS-BELOW settle read
    zero_mean = sum(zero_posts) / len(zero_posts)
    dose_mean = sum(dose_posts) / len(dose_posts)
    two_phase_mean = 0.5 * (zero_mean + dose_mean)
    d_strict = abs(plateau - two_phase_mean)
    d_upper = abs(plateau - dose_mean)
    if d_strict <= RIDER_TOL:
        rverdict = "CONFIRMED-STRICT"
    elif d_upper <= RIDER_TOL:
        rverdict = "REFINED-UPPER-RAIL"
    else:
        rverdict = "REFUTED"
    # R3: the piecewise-geometric time-average over the committed t700->t800
    # cycle: (t700post -> t725gate), (t725gate -> t750gate), [the t750 dose
    # jump: instantaneous], (t750post-bracketed -> t775gate), (t775gate ->
    # t800gate). The t750 post is bracketed by the neighbouring committed
    # dose-jump posts (t700 0.8868, t800 0.8732) widened to [0.85, 0.90] —
    # DISCLOSED reconstruction uncertainty. ml[i] = the leg-2 ledger's
    # row i (0-based): global event = 17 + i.
    def _geo_mean(a: float, b: float) -> float:
        if a <= 0 or b <= 0 or abs(a - b) < 1e-12:
            return 0.5 * (a + b)
        return (a - b) / math.log(a / b)
    g725 = float(ml[12]["read_at_gate"])   # global 29
    g750 = float(ml[13]["read_at_gate"])   # global 30
    g775 = float(ml[14]["read_at_gate"])   # global 31
    g800 = float(ml[15]["read_at_gate"])   # global 32
    lo_post, hi_post = 0.85, 0.90
    seg_a = _geo_mean(traj[700], g725)     # the post-dose decay to the ZERO
    seg_b = _geo_mean(g725, g750)          # the ZERO-phase decay to the dose
    seg_c = lambda post: _geo_mean(post, g775)   # after the t750 dose
    seg_d = _geo_mean(g775, g800)          # the ZERO-phase decay to the dose
    ta_lo = 0.25 * (seg_a + seg_b + seg_c(lo_post) + seg_d)
    ta_hi = 0.25 * (seg_a + seg_b + seg_c(hi_post) + seg_d)
    return {
        "form": "THE VERIFICATION RIDER: T310's 'the equilibrium is the "
                "limit-cycle's average' tested numerically against e339's "
                "md5-bound committed trace (global events 25-32, the "
                "established cycle; CPU only; no new compute)",
        "plateau_t800": plateau,
        "zero_phase": {"gate_reads": zero_gates,
                       "post_mean (= gate mean, no step)": zero_mean},
        "dose_phase": {"gate_reads": dose_gates,
                       "post_reads_committed": dose_posts,
                       "post_mean": dose_mean},
        "equations": {
            "R1_strict": {"value": two_phase_mean,
                          "absdiff_vs_plateau": d_strict,
                          "tol": RIDER_TOL,
                          "holds": bool(d_strict <= RIDER_TOL)},
            "R2_upper_rail": {"value": dose_mean,
                              "absdiff_vs_plateau": d_upper,
                              "tol": RIDER_TOL,
                              "holds": bool(d_upper <= RIDER_TOL)},
            "R3_time_average_reconstructed": {
                "bracket": [ta_lo, ta_hi],
                "t750_post_bracket_disclosed": [lo_post, hi_post],
                "note": "piecewise-geometric over the committed t700->t800 "
                        "anchors; descriptive, never adjudicating"}},
        "verdict": rverdict,
        "note": "CONFIRMED-STRICT: the plateau IS the two-phase average; "
                "REFINED-UPPER-RAIL: the milestone cadence (every 4th "
                "event, the dose phase's parity) is phase-locked to the "
                "cycle's upper rail — the 'equilibrium' is the upper "
                "rail, not the average; REFUTED: neither equation holds",
    }


# ======================================================================
def main():
    global dev
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "x41_relay_kinetics",
        "phase": (
            "THE RELAY'S KINETICS (R78's card; W049's sharpened dial "
            "question): e339's committed t800 state (the model + both "
            "SGD-M buffers + both generator states, md5-bound) resumed "
            "into THREE short branches — LEG-CONTROL (the unperturbed "
            "continuation; the cycle must reproduce), LEG-PERIOD2X (the "
            "event period doubled M 25->50), LEG-LRCAP (the dose ceiling "
            "halved) — each leg a fresh length-scaled pool over the "
            "continued stream; the phase classes + rails read per leg; "
            "the verification rider tests T310's 'the equilibrium is "
            "the limit-cycle's average' against the committed trace"),
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "net0_class": NET0_CLASS,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "the legs run SEQUENTIALLY with cooldowns between "
                      "— never concurrent) + CPU fp64 projections, "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line (dispatch 85), recorded to "
                      "runs/_envelope_log.jsonl tagged x41:<LEG>",
            "trainings": f"{len(LEGS)} continuation legs (e336's driver "
                         "verbatim; budgeted orthogonalized corpus steps "
                         "over a fresh length-scaled B_C + the "
                         "NAME-ONLY maintenance events through opt_F at "
                         "the error-gated lr); NO twin (by design); "
                         "NO cons",
        },
        "builds_on": builds_on,
        "whats_new": whats_new,
        "deviations": deviations,
    })
    write_partial("startup (bars + lean + P-x41a registered, committed "
                  "at birth)")

    # ---- the committed e339 record (md5-bound; the repro targets) ------
    for p, want in [(E339_REA_RESUME_CK, E339_REA_RESUME_MD5),
                    (E339_REA_POST_CK, E339_REA_POST_MD5),
                    (E339_METRICS, E339_METRICS_MD5)]:
        got = md5of(p)
        assert got == want, f"e339 artifact drifted: {p.name} {got} != {want}"
    e339m = json.loads(E339_METRICS.read_text(encoding="utf-8"))
    e339_rea_ph = e339m["arms"]["ERROR-GATED"]["phase"]
    e339_traj = {int(r["step"]): r for r in e339_rea_ph["traj"]}
    assert abs(e339_traj[800]["g0_pz"] - E339_REA_T800_G0) < 1e-15 \
        and abs(e339_traj[800]["survival_ratio_vs_committed"]
                - E339_REA_T800_RATIO) < 1e-15, "committed t800 read drift"
    assert abs(e339_rea_ph["budget"]["S_total_final"]
               - (E339_LEG2_S_CORPUS + E339_LEG2_S_MAINT)) < 1e-9, \
        "committed leg-2 spend drift"
    # the committed cycle's signature asserted literal-for-literal
    for (g, st_, gate, dfd, lr_) , row in zip(
            E339_CYCLE, e339_rea_ph["maint_ledger"][8:16]):
        assert 32 + int(row["maint_index"]) == g \
            and int(row["step"]) == st_ \
            and abs(float(row["read_at_gate"]) - gate) < 1e-15 \
            and abs(float(row["deficit_t"]) - dfd) < 1e-15 \
            and abs(float(row["lr_maint"]) - lr_) < 1e-15, \
            f"committed cycle drift at global event {g}"
    log("P-1 e339's committed record BOUND (metrics md5 "
        f"{E339_METRICS_MD5[:8]}...; t800 read {E339_REA_T800_G0:.10f} = "
        f"x{E339_REA_T800_RATIO:.4f}; the established 2-cycle asserted "
        "literal-for-literal)")

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
    # against, e339's t800 stream included)
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
                    "same room e339's t800 stream orthogonalized against)",
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
        "form": "the legs run on the COMMITTED lr constants (the "
                "modules' own globals, asserted): LR_STABLE = x0.01 x "
                "LR_SGD_matched; LR_M_MAX = 0.40 x BUDGET / e287's b_m "
                "sum (the SAME identity, the SAME value — the LRCAP "
                "leg's halving is the SINGLE re-bound controller "
                "constant in this cell, passed as the driver's own "
                "lr_m_max argument); momentum 0.9, wd 0.0 EXACTLY",
        "lr_sgd_record": E336.E273_LR_SGD,
        "lr_sgd_runtime": lr_sgd_runtime,
        "lr_stable_module": E336.LR_STABLE,
        "lr_m_max_module": E336.LR_M_MAX_FROZEN,
        "lr_m_max_lrcap_leg": LR_M_MAX_HALF,
        "momentum": E336.SGD_MOMENTUM, "wd": E336.SGD_WD,
        "pass": bool(E336.LR_STABLE == 0.21738574801453703
                     and E336.LR_M_MAX_FROZEN
                     == 0.021942464013242388
                     and abs(LR_M_MAX_HALF - 0.010971232006621194) < 1e-15
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

    # ================= P3: the e339 t800 resume gates + staging =========
    if SMOKE:
        # the smoke base: manufacture a tiny committed-form t800-equivalent
        # via e336's own driver (12 steps, M=2) — machinery validation
        # only; the smoke numbers are MEANINGLESS (disclosed)
        log("P3 SMOKE: manufacturing a smoke base checkpoint via e336's "
            "own driver (12 steps, M=2) — machinery validation only")
        patch_leg_globals(SMOKE_BASE_STEPS, tuple(range(2, 13, 2)),
                          (1, 2, 3), (10, 11, 12), SMOKE_BASE_M)
        bc0 = B_C_400 * SMOKE_BASE_STEPS / 400
        bm0 = B_M_400 * (SMOKE_BASE_STEPS // SMOKE_BASE_M) / 16
        _ = E336.chunked_errorgated_phase(
            "SMOKE-BASE", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, bc0, bm0, win_t, inst_mask, anchor_full,
            train_ids, g0_ids, gm12_ids, r_eval_xy, tid, LR_M_MAX_FROZEN,
            RD / "smoke_base_REA_resume.pt", dev)
        base_src = RD / "smoke_base_REA_resume.pt"
        base_step_expect = SMOKE_BASE_STEPS
    else:
        base_src = E339_REA_RESUME_CK
        base_step_expect = LEG_BASE_STEP

    base_st = torch.load(base_src, map_location="cpu", weights_only=False)
    bits_src = state_bits(base_st)
    G_E339BIND = {
        "form": "the x41 base identity: e339's committed t800 resume "
                "state raw-md5-bound; step/n_maint/spend asserted == the "
                "committed metrics; the resume model bit-compared vs the "
                "committed post checkpoint; the established-cycle ledger "
                "asserted literal-for-literal",
        "resume_md5": md5of(base_src),
        "committed_md5": None if SMOKE else E339_REA_RESUME_MD5,
        "step": int(base_st["step"]),
        "n_maint": int(base_st["n_maint"]),
        "S_corpus": float(base_st["S_corpus"]),
        "S_maint": float(base_st["S_maint"]),
        "expected": {"step": base_step_expect,
                     "n_maint": (SMOKE_BASE_STEPS // SMOKE_BASE_M
                                 if SMOKE else 16),
                     "S_corpus": E339_LEG2_S_CORPUS,
                     "S_maint": E339_LEG2_S_MAINT},
    }
    model_bit_equal = None
    if not SMOKE:
        e339_post = torch.load(E339_REA_POST_CK, map_location="cpu",
                               weights_only=False)
        model_bit_equal = all(
            torch.equal(base_st["model"][k], e339_post["model"][k])
            for k in base_st["model"])
        del e339_post
        G_E339BIND["resume_model_bit_equal_post"] = bool(model_bit_equal)
    G_E339BIND["pass"] = bool(
        (SMOKE or md5of(base_src) == E339_REA_RESUME_MD5)
        and int(base_st["step"]) == base_step_expect
        and (SMOKE or int(base_st["n_maint"]) == 16)
        and (SMOKE or (abs(float(base_st["S_corpus"])
                           - E339_LEG2_S_CORPUS) < 1e-9
                       and abs(float(base_st["S_maint"])
                               - E339_LEG2_S_MAINT) < 1e-9
                       and model_bit_equal)))
    assert G_E339BIND["pass"], f"e339 t800 bind FAILED: {G_E339BIND}"
    metrics["gates"]["G_E339BIND"] = G_E339BIND

    # ---- the staging: state bits through, leg pools zeroed -------------
    base_ck = RD / ("smoke_x41_base.pt" if SMOKE else "x41_REA_t800base.pt")
    leg0 = staged_leg_state(base_st, n_par)
    torch.save(leg0, base_ck)
    chk = torch.load(base_ck, map_location="cpu", weights_only=False)
    bits_dst = state_bits(chk)
    staging_ok = (bits_src == bits_dst
                  and int(chk["step"]) == base_step_expect
                  and chk["S_corpus"] == 0.0 and chk["S_maint"] == 0.0
                  and chk["n_maint"] == 0)
    leg_ck_paths = {}
    for leg in LEG_ORDER:
        p = RD / f"{'smoke_' if SMOKE else ''}x41_{leg}_REA_resume.pt"
        torch.save(staged_leg_state(base_st, n_par), p)
        c = torch.load(p, map_location="cpu", weights_only=False)
        if state_bits(c) != bits_src or int(c["step"]) != base_step_expect:
            staging_ok = False
        leg_ck_paths[leg] = p
    G_STAGING = {
        "form": "the ONLY protocol-touching transformation: e339's "
                "committed state bits (model/optC/optF/cgen/igen) pass "
                "through UNTOUCHED (digest-verified across the base AND "
                "every per-leg copy); the leg ledgers + spend pools "
                "zeroed (the leg convention); the generator states "
                "carried — the stream CONTINUES (never re-seeded); the "
                "legs are BRANCHES of one base (shared draws by "
                "construction)",
        "base": str(base_ck),
        "leg_copies": {k: str(v) for k, v in leg_ck_paths.items()},
        "bits_unchanged": bool(staging_ok),
        "global_event_mapping": "the staging zeroes n_maint (the family "
                                "convention): a leg's event i is the "
                                "GLOBAL event 32 + i",
        "pass": bool(staging_ok)}
    assert G_STAGING["pass"], f"staging FAILED: {G_STAGING}"
    metrics["gates"]["G_STAGING"] = G_STAGING

    # ---- G_T800_REPRO: the staged base reproduces the committed read ---
    net_b = G1.evl_load({k: v for k, v in base_st["model"].items()})
    repro_g0 = G1.battery_cell(net_b, g0_ids, tid)["mean_pz"]
    repro_g0_z = G1.battery_cell(net_b, g0_ids, zid)["mean_pz"]
    committed_g0 = (base_st["traj"][-1]["g0_pz"] if SMOKE
                    else e339_traj[800]["g0_pz"])
    G_T800_REPRO = {
        "form": "the staged bits reproduce the committed t800 battery "
                "read (the family's cross-session read-determinism law: "
                "same bits + same battery + same CPU fp32 code path)",
        "g0_reproduced": repro_g0,
        "g0_committed": committed_g0,
        "absdiff": abs(repro_g0 - committed_g0),
        "g0_z_channel": repro_g0_z,
        "tol": 2e-6,
        "pass": bool(abs(repro_g0 - committed_g0) <= 2e-6)}
    assert G_T800_REPRO["pass"], f"t800 repro FAILED: {G_T800_REPRO}"
    metrics["gates"]["G_T800_REPRO"] = G_T800_REPRO
    del net_b, base_st, chk, leg0
    log(f"P3 G_E339BIND + G_STAGING + G_T800_REPRO: the t800 base bound "
        f"(read repro |d| {G_T800_REPRO['absdiff']:.1e}); {len(LEGS)} leg "
        "copies staged digest-equal: PASS")
    write_partial("P3 the base bound + staged")

    # ================= P4: THE VERIFICATION RIDER (CPU, free) ===========
    rider = run_rider(e339m)
    G_RIDER = {
        "form": rider["form"],
        "verdict": rider["verdict"],
        "equations": rider["equations"],
        "pass": True,   # the rider REPORTS; its verdicts are pre-registered
    }
    metrics["gates"]["G_RIDER"] = G_RIDER
    metrics["rider"] = rider
    log(f"P4 THE RIDER: verdict {rider['verdict']} — R1 strict |d| "
        f"{rider['equations']['R1_strict']['absdiff_vs_plateau']:.4f} "
        f"(tol {RIDER_TOL}); R2 upper-rail |d| "
        f"{rider['equations']['R2_upper_rail']['absdiff_vs_plateau']:.4f}; "
        "the reconstructed time-average bracket "
        f"{rider['equations']['R3_time_average_reconstructed']['bracket']}"
        " vs the plateau "
        f"{rider['plateau_t800']:.4f}")

    # ================= P5-P7: THE THREE LEGS ============================
    leg_results: dict[str, dict] = {}

    def leg_event_steps(leg: str, m: int, start: int, end: int) -> list[int]:
        return [s for s in range(start + 1, end + 1) if s % m == 0]

    for li, leg in enumerate(LEG_ORDER):
        if li > 0:
            E261.burst_cooldown(f"{LEG_ORDER[li - 1]} -> {leg}")
        assert common.gpu_ok(), f"gpu_ok() gate before leg {leg}"
        spec = LEGS[leg]
        m = int(spec["m_every"])
        start = base_step_expect
        end = start + int(spec["steps"])
        ev_steps = leg_event_steps(leg, m, start, end)
        assert len(ev_steps) >= 2, f"leg {leg}: too few events {ev_steps}"
        milestones = tuple(ev_steps)
        win1 = (ev_steps[0] - 1, ev_steps[0], ev_steps[0] + 1)
        win2 = (ev_steps[-1] - 1, ev_steps[-1], ev_steps[-1] + 1)
        bind_rec = patch_leg_globals(end, milestones, win1, win2, m)
        rea_arm_before = E336.REA_ARM
        E336.REA_ARM = f"x41:{leg}"    # label-only (the burst tags);
        # disclosed under the family's re-binding convention
        budget_c = B_C_400 * spec["steps"] / 400
        budget_m = B_M_400 * len(ev_steps) / 16
        lr_m = float(spec["lr_m_max"])
        log("=" * 78)
        log(f"LEG-{leg} — t{start + 1}..t{end} ({spec['steps']} steps, "
            f"M={m}, lr_m_max {lr_m:.10f}, {len(ev_steps)} events at "
            f"{ev_steps}; B_C {budget_c:.4f} B_M {budget_m:.4f} — the "
            "length-scaled pools; e336's committed driver VERBATIM)")
        res = E336.chunked_errorgated_phase(
            f"X41-{leg}", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, budget_c, budget_m, win_t, inst_mask,
            anchor_full, train_ids, g0_ids, gm12_ids, r_eval_xy, tid,
            lr_m, leg_ck_paths[leg], dev)
        sd_l = res["sd"]
        net_l = G1.evl_load(sd_l)
        cells_l = {"gm12": G1.battery_cell(net_l, gm12_ids, tid)["mean_pz"],
                   "g0": G1.battery_cell(net_l, g0_ids, tid)["mean_pz"],
                   "g0_z": G1.battery_cell(net_l, g0_ids, zid)["mean_pz"],
                   "ce_r": G1.ce_fixed_cpu(net_l, *r_eval_xy)}
        torch.save({"model": sd_l, "meta": {
            "experiment": "x41", "leg": leg,
            "desc": f"x41 relay kinetics leg {leg}: {spec['desc']}",
            "m_every": m, "lr_m_max": lr_m,
            "S_corpus": res["S_corpus"], "S_maint": res["S_maint"]}},
            RD / (f"smoke_x41_{leg}_REA_post.pt" if SMOKE
                  else f"x41_{leg}_REA_post.pt"))
        del net_l
        cyc = classify_events(res["maint_ledger"], res["traj"])
        E336.REA_ARM = rea_arm_before   # restore (leave e336 as found)
        lr_rows = res["lr_ledger"]
        applied = [v["lr_applied"] for v in lr_rows.values()]
        cap_below = {int(k): bool(v["cap"] <= v["lr_sched"])
                     for k, v in lr_rows.items()}
        sched_bound = [s for s, ok in cap_below.items() if not ok]
        med_applied = med(applied) if applied else None
        # ---- the per-leg identity gate ----
        G_LEG = {
            "form": f"the leg's protocol identity: M={m} (the committed "
                    f"25{' — UNCHANGED' if m == 25 else ' PERTURBED'}), "
                    f"lr_m_max={lr_m!r} "
                    f"({'unchanged' if lr_m == LR_M_MAX_FROZEN else 'HALVED (the perturbation)'}), "
                    f"PHASE_STEPS={end}, events at {ev_steps}, pools "
                    f"B_C={budget_c!r} B_M={budget_m!r} (length-scaled); "
                    f"the applied corpus lr's MEDIAN within the frozen "
                    f"protocol band {LR_BAND} (e339's committed leg-2 "
                    f"range widened 20% — the wash kinetics matched)",
            "m_every": m, "lr_m_max": lr_m, "phase_steps": end,
            "event_steps": ev_steps, "milestones": list(milestones),
            "budget_c": budget_c, "budget_m": budget_m,
            "leg_bind_record": bind_rec,
            "applied_lr_median": med_applied,
            "applied_lr_min": min(applied) if applied else None,
            "applied_lr_max": max(applied) if applied else None,
            "lr_band": LR_BAND,
            "sched_bound_steps": sched_bound,
            "pass": bool(med_applied is not None
                         and LR_BAND[0] <= med_applied <= LR_BAND[1]
                         and [int(r["step"]) for r in res["maint_ledger"]]
                         == ev_steps
                         and int(res["steps_ran"]) == end)}
        assert G_LEG["pass"], f"leg identity FAILED: {G_LEG}"
        metrics["gates"][f"G_LEG_{leg}"] = G_LEG
        leg_results[leg] = {
            "spec": {k: (v if not isinstance(v, float) else v)
                     for k, v in spec.items()},
            "m_every": m, "lr_m_max": lr_m, "event_steps": ev_steps,
            "phase": {k: res[k] for k in
                      ("traj", "corpus_ledger", "lr_ledger",
                       "maint_ledger", "window_ledger", "bufsep",
                       "chunk_table", "steps_ran", "budget_ledger",
                       "orth_ledger", "disp_ledger", "buf_ledger")},
            "cycle": cyc,
            "post_cells": cells_l,
            "S_corpus": float(res["S_corpus"]),
            "S_maint": float(res["S_maint"]),
            "budget_c": budget_c, "budget_m": budget_m,
            "orth_max": res["orth_max"],
            "checkpoint": str(RD / (f"smoke_x41_{leg}_REA_post.pt"
                                    if SMOKE
                                    else f"x41_{leg}_REA_post.pt")),
        }
        log(f"LEG-{leg} DONE: classes {cyc['classes']} | floor rail "
            f"{cyc['floor_rail']} upper rail {cyc['upper_rail']} | "
            f"post g0 {cells_l['g0']:.6f} | S "
            f"{res['S_corpus']:.4f}+{res['S_maint']:.4f} | sched-bound "
            f"steps {sched_bound}")
        write_partial(f"LEG-{leg} complete")

    # restore the committed module globals (leave e336 as found)
    patch_leg_globals(400, (100, 200, 300, 400), (24, 25, 26),
                      (199, 200, 201), 25)

    # ================= P8: the standing gates ===========================
    G_ORTH = {
        "form": "the orthogonalized-stream gate (NON-HALTING): the "
                "stepped corpus gradient entirely orthogonal to the room "
                "at every step of every leg",
        "orth_max_rel_err_per_leg": {k: leg_results[k]["orth_max"]
                                     for k in LEG_ORDER},
        "bar": ORTH_BAR,
        "pass": bool(all(leg_results[k]["orth_max"] < ORTH_BAR
                         for k in LEG_ORDER))}
    metrics["gates"]["G_ORTH"] = G_ORTH

    G_BUFSEP = {
        "form": "the isolation gate (HALT carried): bidirectional buffer "
                "separation at every optimizer event of every leg",
        "per_leg": {k: leg_results[k]["phase"]["bufsep"]
                    for k in LEG_ORDER},
        "pass": bool(all(
            leg_results[k]["phase"]["bufsep"]["corpus_violations"] == 0
            and leg_results[k]["phase"]["bufsep"]["maint_violations"] == 0
            for k in LEG_ORDER))}
    assert G_BUFSEP["pass"], f"G_BUFSEP HALT: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    G_BUDGET = {
        "form": "the length-scaled leg pools respected (NON-HALTING)",
        "per_leg": {k: {"S_corpus": leg_results[k]["S_corpus"],
                        "S_maint": leg_results[k]["S_maint"],
                        "B_C": leg_results[k]["budget_c"],
                        "B_M": leg_results[k]["budget_m"]}
                    for k in LEG_ORDER},
        "pass": bool(all(
            leg_results[k]["S_corpus"]
            <= leg_results[k]["budget_c"] + BUDGET_SLACK
            and leg_results[k]["S_maint"]
            <= leg_results[k]["budget_m"] + BUDGET_SLACK
            for k in LEG_ORDER))}
    metrics["gates"]["G_BUDGET"] = G_BUDGET

    stream_rows = {}
    for k in LEG_ORDER:
        rows = sorted((int(s), v) for s, v in
                      leg_results[k]["phase"]["lr_ledger"].items())
        # corpus CE rows live in corpus_ledger; fall back to traj rows
        cl = leg_results[k]["phase"].get("corpus_ledger", {})
        crows = sorted((int(s), v) for s, v in cl.items())
        if len(crows) >= 4:
            early = [v["ce"] for s, v in crows if s <= crows[0][0] + 30]
            late = [v["ce"] for s, v in crows if s > crows[-1][0] - 30]
        else:
            tr = leg_results[k]["phase"]["traj"]
            early = [r["ce_corpus"] for r in tr[:max(1, len(tr) // 2)]]
            late = [r["ce_corpus"] for r in tr[max(1, len(tr) // 2):]]
        ok = (None if len(early) < 1 or len(late) < 1 else bool(
            med(late) <= STREAM_STABLE_TOL * med(early)))
        stream_rows[k] = {"median_early": med(early) if early else None,
                          "median_late": med(late) if late else None,
                          "pass": True if ok is None else ok}
    G_STREAMELIVE = {
        "form": "the standing stream condition (NON-HALTING): the corpus "
                "stream live in every leg (late-vs-early median CE "
                f"improving OR stable within {STREAM_STABLE_TOL}x)",
        "per_leg": stream_rows,
        "pass": bool(all(v["pass"] for v in stream_rows.values()))}
    metrics["gates"]["G_STREAMELIVE"] = G_STREAMELIVE

    G_SCHED = {
        "form": "the schedule-vs-cap record: CONTROL/LRCAP (ending t900) "
                "assert cap-below-sched at every logged step (the wash "
                "cap-determined, protocol-matched to e339); "
                "PERIOD2X's t950 event DISCLOSED schedule-tail (from "
                f"~step {SCHED_TAIL_FROM} the house cosine decays below "
                "the cap — its primary events t850/t900 remain "
                "cap-bound)",
        "sched_bound_per_leg": {
            k: metrics["gates"][f"G_LEG_{k}"]["sched_bound_steps"]
            for k in LEG_ORDER},
        "pass": bool(
            metrics["gates"]["G_LEG_CONTROL"]["sched_bound_steps"] == []
            and metrics["gates"]["G_LEG_LRCAP"]["sched_bound_steps"] == []
            and all(s >= SCHED_TAIL_FROM - 10
                    for s in
                    metrics["gates"]["G_LEG_PERIOD2X"]
                    ["sched_bound_steps"]))}
    metrics["gates"]["G_SCHED"] = G_SCHED

    # ================= P9: the cycle gates + the adjudication ===========
    ctrl = leg_results["CONTROL"]["cycle"]
    expected_ctrl = ["ZERO", "DOSE", "ZERO", "DOSE"]
    control_ok = (ctrl["classes"] == expected_ctrl)
    G_CYCLE_REPRO = {
        "form": "the control leg reproduces the committed cycle: the "
                "four continuation events (global 33-36) classified "
                "ZERO/DOSE/ZERO/DOSE — the committed parity (the t800 "
                "event was a DOSE) and the committed rails",
        "classes": ctrl["classes"],
        "expected": expected_ctrl,
        "floor_rail": ctrl["floor_rail"],
        "upper_rail": ctrl["upper_rail"],
        "committed_floor_rail": E339_FLOOR_RAIL,
        "committed_upper_rail": E339_UPPER_RAIL,
        "pass": bool(control_ok)}
    metrics["gates"]["G_CYCLE_REPRO"] = G_CYCLE_REPRO
    log(f"P9 G_CYCLE_REPRO: the control leg's classes {ctrl['classes']} "
        f"(expected {expected_ctrl}) — "
        f"{'PASS' if control_ok else 'FAIL'}")

    # ---- the MOVE observables ----
    m1 = bool(leg_results["PERIOD2X"]["cycle"]["n_zero"] == 0
              and leg_results["PERIOD2X"]["cycle"]["n_dose"] >= 2)
    moves = []
    if m1:
        moves.append("M1 PERIOD-RELOCK (leg-A: every event doses at the "
                     "doubled spacing — the cycle re-locked to the event "
                     f"cadence; classes "
                     f"{leg_results['PERIOD2X']['cycle']['classes']})")
    rail_moves = []
    for k in ("PERIOD2X", "LRCAP"):
        for rail in ("floor_rail", "upper_rail"):
            a = leg_results[k]["cycle"][rail]
            b = ctrl[rail]
            if a is not None and b is not None \
                    and abs(a - b) > PHASE_MARGIN:
                rail_moves.append(
                    f"{k} {rail} {a:.4f} vs control {b:.4f} "
                    f"(|d| {abs(a - b):.4f} > {PHASE_MARGIN})")
    m2 = bool(rail_moves)
    if m2:
        moves.append("M2 AMPLITUDE-MOVE (" + "; ".join(rail_moves) + ")")
    flips = []
    for k in ("PERIOD2X", "LRCAP"):
        seq_a = leg_results[k]["cycle"]["classes"]
        seq_b = ctrl["classes"][:len(seq_a)]
        if seq_a != seq_b:
            flips.append(f"{k}: {seq_a} vs control {seq_b}")
    m3 = bool(flips)
    if m3:
        moves.append("M3 CLASS-FLIP (" + "; ".join(flips) + ")")
    legA_alt = bool(leg_results["PERIOD2X"]["cycle"]["n_zero"] >= 1
                    and leg_results["PERIOD2X"]["cycle"]["n_dose"] >= 1)

    standing_ok = bool(G_ORTH["pass"] and G_BUFSEP["pass"]
                       and G_BUDGET["pass"] and G_STREAMELIVE["pass"])
    if not control_ok:
        verdict = "CONTROL-FAILS"
    elif (m1 or m2 or m3):
        verdict = "ENTRAINABLE"
    elif legA_alt:
        verdict = "STRUCTURAL"
    else:
        verdict = "MIXED"

    clause = {
        "ENTRAINABLE": (
            "at least one perturbation moved the cycle's observable "
            "structure: " + ("; ".join(moves) if moves else "") +
            " — THE CYCLE IS TUNABLE; W049's dial designs a relay tuner; "
            "the calibration program gains its kinetics"),
        "STRUCTURAL": (
            "no move observable fired; the cycle alternates still at the "
            "doubled spacing and the rails held within the margin — the "
            "2-cycle is the slot's own dynamics; the dial is bounded by "
            "it"),
        "CONTROL-FAILS": (
            f"the control leg's classes {ctrl['classes']} != "
            f"{expected_ctrl} — the committed cycle did NOT reproduce in "
            "the continuation; no adjudication (the instrument outcome, "
            "disclosed)"),
        "MIXED": (
            "neither branch fired cleanly — the structure moved "
            "ambiguously (full trace in metrics; the composite's "
            "residue class)"),
    }[verdict]

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "CONTROL-FAILS -> ENTRAINABLE (M1 or M2 or "
                           "M3) -> STRUCTURAL (no move + leg-A "
                           "alternates) -> MIXED — frozen at birth",
        "gates_pass": bool(all(g.get("pass", True) for g in
                               metrics["gates"].values())),
        "reads": {
            "the_control_leg": {
                "classes": ctrl["classes"],
                "events": ctrl["events"],
                "floor_rail": ctrl["floor_rail"],
                "upper_rail": ctrl["upper_rail"],
                "weak_rail": ctrl["weak_rail"],
                "vs_committed": {
                    "classes": ["ZERO", "DOSE"] * 4,
                    "floor_rail": E339_FLOOR_RAIL,
                    "upper_rail": E339_UPPER_RAIL,
                    "weak_rail": E339_WEAK_RAIL}},
            "the_period2x_leg": {
                "classes": leg_results["PERIOD2X"]["cycle"]["classes"],
                "events": leg_results["PERIOD2X"]["cycle"]["events"],
                "rails": {r: leg_results["PERIOD2X"]["cycle"][r]
                          for r in ("floor_rail", "upper_rail",
                                    "weak_rail")}},
            "the_lrcap_leg": {
                "classes": leg_results["LRCAP"]["cycle"]["classes"],
                "events": leg_results["LRCAP"]["cycle"]["events"],
                "rails": {r: leg_results["LRCAP"]["cycle"][r]
                          for r in ("floor_rail", "upper_rail",
                                    "weak_rail")}},
            "the_move_observables": {
                "M1_period_relock": m1,
                "M2_amplitude_move": m2, "rail_moves": rail_moves,
                "M3_class_flip": m3, "flips": flips,
                "legA_alternation_survives": legA_alt},
            "the_post_cells": {k: leg_results[k]["post_cells"]
                               for k in LEG_ORDER},
            "the_rider": rider,
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — machinery validation only; the numbers "
                        "are meaningless (tiny manufactured legs); the "
                        "bars NOT scored" if SMOKE else None),
    }
    metrics["adjudication"]["predictions_scored"] = {
        "P-x41a_entrainable": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "ENTRAINABLE"),
            "note": "SCORED: TRUE iff the verdict == ENTRAINABLE"
                    + ("" if not SMOKE else " (not scored — smoke)")},
        "P-x41a_rider": {
            "scored": bool(not SMOKE),
            "true": bool(rider["verdict"] == "REFINED-UPPER-RAIL"),
            "note": "SCORED: TRUE iff the rider verdict == "
                    "REFINED-UPPER-RAIL"
                    + ("" if not SMOKE else " (not scored — smoke)")},
        "lab_lean": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "ENTRAINABLE"),
            "note": "the dispatch's lean: ENTRAINABLE, weakly"},
    }
    log("=" * 78)
    log(f"P9 ADJUDICATED: {verdict} — {clause}")
    write_partial("P9 ADJUDICATED (the frozen bars)")

    # ================= P10: the figure ==================================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = {"CONTROL": "tab:blue", "PERIOD2X": "tab:red",
              "LRCAP": "tab:green"}
    # (0,0) the gate reads per event
    ax = axes[0][0]
    ax.axhline(E336.FACT_BASELINE_G0, color="k", ls="--", lw=1.0,
               label=f"the baseline {E336.FACT_BASELINE_G0:.4f} "
                     "(deficit = 0)")
    ax.axhspan(E339_FLOOR_RAIL - 0.016, E339_FLOOR_RAIL + 0.016,
               color="gray", alpha=0.18,
               label=f"committed floor rail {E339_FLOOR_RAIL:.3f}")
    ax.axhspan(E339_WEAK_RAIL - 0.026, E339_WEAK_RAIL + 0.026,
               color="orange", alpha=0.15,
               label=f"committed weak rail {E339_WEAK_RAIL:.3f}")
    for k in LEG_ORDER:
        cyc = leg_results[k]["cycle"]
        xs = [e["global_index"] for e in cyc["events"]]
        ys = [e["gate_read"] for e in cyc["events"]]
        ax.plot(xs, ys, "o-", color=colors[k], ms=6,
                label=f"{k} (M={leg_results[k]['m_every']}, "
                      f"lr_cap×{leg_results[k]['lr_m_max'] / LR_M_MAX_FROZEN:.1f})")
        for e in cyc["events"]:
            if e["class"] == "ZERO":
                ax.annotate("Z", (e["global_index"], e["gate_read"]),
                            textcoords="offset points", xytext=(4, -12),
                            fontsize=8, color=colors[k])
            elif e["class"] == "DOSE":
                ax.annotate("D", (e["global_index"], e["gate_read"]),
                            textcoords="offset points", xytext=(4, 6),
                            fontsize=8, color=colors[k])
    cx = [g for g, _, _, _, _ in E339_CYCLE]
    cy = [gate for _, _, gate, _, _ in E339_CYCLE]
    ax.plot(cx, cy, "s", color="gray", ms=4, alpha=0.6,
            label="e339 committed cycle (events 25-32)")
    ax.set_xlabel("global maintenance event index")
    ax.set_ylabel("pre-event gate read (raw)")
    ax.set_title("the relay's phases: the gate reads (Z=zero-dose, "
                 "D=dose)"
                 + (" [SMOKE]" if SMOKE else ""))
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3)

    # (0,1) the dose trace
    ax = axes[0][1]
    for k in LEG_ORDER:
        cyc = leg_results[k]["cycle"]
        ax.plot([e["global_index"] for e in cyc["events"]],
                [e["lr_m"] for e in cyc["events"]], "o-",
                color=colors[k], ms=6, label=k)
    ax.plot(cx, [lr_ for _, _, _, _, lr_ in E339_CYCLE], "s",
            color="gray", ms=4, alpha=0.6, label="e339 committed")
    ax.axhline(LR_M_MAX_FROZEN, color="k", ls=":", lw=1.0,
               label=f"LR_M_MAX {LR_M_MAX_FROZEN:.4f}")
    ax.axhline(LR_M_MAX_HALF, color="k", ls="--", lw=0.8,
               label=f"halved cap {LR_M_MAX_HALF:.4f}")
    ax.set_xlabel("global maintenance event index")
    ax.set_ylabel("applied lr_m (the relay's dose)")
    ax.set_title("the relay: the applied dose per event")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) the post-event reads (the upper rail)
    ax = axes[1][0]
    ax.axhspan(E339_UPPER_RAIL - 0.013, E339_UPPER_RAIL + 0.013,
               color="gray", alpha=0.18,
               label=f"committed upper rail {E339_UPPER_RAIL:.3f}")
    for k in LEG_ORDER:
        cyc = leg_results[k]["cycle"]
        pts = [(e["global_index"], e["post_read"]) for e in cyc["events"]
               if e["post_read"] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-",
                    color=colors[k], ms=7, label=k)
    ax.plot([28, 32], [E339_MILESTONE_G0[700], E339_MILESTONE_G0[800]],
            "s", color="gray", ms=5, alpha=0.6,
            label="e339 committed milestones")
    ax.set_xlabel("global maintenance event index")
    ax.set_ylabel("post-event read (raw)")
    ax.set_title("the cycle's upper rail: the post-event reads")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) the verification panel
    ax = axes[1][1]
    seg_x = [700, 725, 750, 775, 800]
    t750_post = 0.5 * (E339_MILESTONE_G0[700] + E339_MILESTONE_G0[800])
    # E339_CYCLE indices 0..7 = global 25..32: t725 = idx 4, t775 = idx 6
    seg_y = [E339_MILESTONE_G0[700], E339_CYCLE[4][2],
             t750_post, E339_CYCLE[6][2], E339_MILESTONE_G0[800]]
    xs_full, ys_full = [700], [E339_MILESTONE_G0[700]]
    for i in range(4):
        for j in range(1, 26):
            f = j / 25
            a, b = seg_y[i], seg_y[i + 1]
            xs_full.append(seg_x[i] + j)
            ys_full.append(a * (b / a) ** f)
    ax.plot(xs_full, ys_full, "-", color="tab:purple", lw=1.2,
            label="the reconstructed cycle (piecewise-geometric; the "
                  "t750 post bracketed)")
    ax.plot(seg_x, seg_y, "o", color="tab:purple", ms=5)
    ax.axhline(rider["plateau_t800"], color="k", ls="--", lw=1.2,
               label=f"the 'equilibrium' (t800 plateau) "
                     f"{rider['plateau_t800']:.4f}")
    ax.axhline(rider["zero_phase"]["post_mean (= gate mean, no step)"],
               color="tab:cyan", ls=":", lw=1.4,
               label=f"ZERO-phase mean "
                     f"{rider['zero_phase']['post_mean (= gate mean, no step)']:.4f}")
    ta = rider["equations"]["R3_time_average_reconstructed"]["bracket"]
    ax.axhspan(ta[0], ta[1], color="gold", alpha=0.25,
               label=f"the cycle's time-average "
                     f"[{ta[0]:.3f}, {ta[1]:.3f}] (R3)")
    ax.axhline(0.5 * (rider["zero_phase"]
                      ["post_mean (= gate mean, no step)"]
                      + rider["dose_phase"]["post_mean"]),
               color="tab:red", ls="-.", lw=1.2,
               label=f"the two-phase average "
                     f"{0.5 * (rider['zero_phase']['post_mean (= gate mean, no step)'] + rider['dose_phase']['post_mean']):.4f} (R1)")
    ax.set_xlabel("corpus step t (the committed cycle t700-t800)")
    ax.set_ylabel("the g0 read (raw)")
    ax.set_title(f"THE VERIFICATION RIDER — {rider['verdict']}: the "
                 "plateau vs the phase means")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"X41 THE RELAY'S KINETICS — {verdict}"
                 + (" [SMOKE — machinery only]" if SMOKE else ""),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = RD / ("smoke_x41_relay_kinetics.png" if SMOKE
                     else "x41_relay_kinetics.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ================= P11: REPORT.md ====================================
    ts = common.now_iso()
    gcount = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    lines = [
        f"# X41 — THE RELAY'S KINETICS (R78's card; W049's dial)\n",
        f"**Date:** {ts} (datetime.now(UTC) stamps in metrics) · "
        f"**Executor report**\n",
        "## THE VERDICT (against the frozen bars)\n",
        f"**{verdict}**"
        + (" _(smoke — machinery validation only; not scored)_"
           if SMOKE else ""),
        "\n```",
        clause,
        "```\n",
        "## The three legs' cycle structures vs the committed cycle\n",
        "| leg | M | lr_m_max | events (global) | classes | floor rail | "
        "upper rail |",
        "|---|---|---|---|---|---|---|",
        f"| committed (e339 25-32) | 25 | 0.021942 | 25-32 | "
        "ZERO/DOSE alternating | "
        f"{E339_FLOOR_RAIL:.4f} | {E339_UPPER_RAIL:.4f} |",
    ]
    for k in LEG_ORDER:
        cyc = leg_results[k]["cycle"]
        lines.append(
            f"| {k} | {leg_results[k]['m_every']} | "
            f"{leg_results[k]['lr_m_max']:.6f} | "
            f"{[e['global_index'] for e in cyc['events']]} | "
            f"{'/'.join(cyc['classes'])} | "
            f"{cyc['floor_rail'] if cyc['floor_rail'] is None else round(cyc['floor_rail'], 4)} | "
            f"{cyc['upper_rail'] if cyc['upper_rail'] is None else round(cyc['upper_rail'], 4)} |")
    lines += [
        "",
        f"- M1 PERIOD-RELOCK: {m1}; M2 AMPLITUDE-MOVE: {m2} "
        f"({rail_moves}); M3 CLASS-FLIP: {m3} ({flips})",
        f"- leg-A's alternation survives: {legA_alt}",
        f"- the move-observable margin: {PHASE_MARGIN} (raw units); the "
        f"committed cycle's rails: floor {E339_FLOOR_RAIL:.4f} / upper "
        f"{E339_UPPER_RAIL:.4f} / weak {E339_WEAK_RAIL:.4f}\n",
        "## The verification rider (CPU; e339's committed trace)\n",
        f"- **{rider['verdict']}** — R1 (the strict average): "
        f"|{rider['plateau_t800']:.4f} - "
        f"{rider['equations']['R1_strict']['value']:.4f}| = "
        f"{rider['equations']['R1_strict']['absdiff_vs_plateau']:.4f} "
        f"(tol {RIDER_TOL}) — "
        f"{'HOLDS' if rider['equations']['R1_strict']['holds'] else 'FAILS'}",
        f"- R2 (the upper rail): |{rider['plateau_t800']:.4f} - "
        f"{rider['equations']['R2_upper_rail']['value']:.4f}| = "
        f"{rider['equations']['R2_upper_rail']['absdiff_vs_plateau']:.4f} "
        f"— {'HOLDS' if rider['equations']['R2_upper_rail']['holds'] else 'FAILS'}",
        f"- R3 (descriptive): the reconstructed cycle time-average "
        f"{ta[0]:.4f}-{ta[1]:.4f} vs the plateau "
        f"{rider['plateau_t800']:.4f} — the plateau is "
        f"{rider['plateau_t800'] / (0.5 * (ta[0] + ta[1])):.2f}x the "
        "time-average",
        f"- T310's sentence, tested: the milestone cadence (every 4th "
        "event — the dose phase's parity) is phase-locked to the "
        "cycle's UPPER RAIL; the 'equilibrium' is the upper rail, not "
        "the cycle's average\n" if rider["verdict"] == "REFINED-UPPER-RAIL"
        else "- (see metrics for the rider's full equations)\n",
        "## The base and the gates\n",
        f"- the base: e339's committed t800 resume state (md5 "
        f"{E339_REA_RESUME_MD5[:8]}...), bit-compared vs the committed "
        "post checkpoint, read-reproduced |d| "
        f"{G_T800_REPRO['absdiff']:.1e}; three digest-verified branch "
        "copies (shared stream draws by construction)",
        f"- the net0 class: {NET0_CLASS['name']} "
        f"({NET0_CLASS['params']:,} params)",
        f"- {gcount} gate classes instantiated, "
        f"{'all PASS' if gcount == len(metrics['gates']) else 'SEE metrics'}: "
        + ", ".join(metrics["gates"]),
        "",
        "## Envelope\n",
        f"- {len(thermal_log)} thermal polls; max temp "
        f"{max((r.get('temp', 0) for r in thermal_log), default=0):.1f}C; "
        f"burst cap {E261.BURST_MAX_S:.0f}s; cooldown "
        f"{E261.COOLDOWN_S:.0f}s between legs; zero concurrent GPU jobs; "
        "every burst logged to runs/_envelope_log.jsonl tagged x41:...",
        "",
        "## Registered predictions\n",
        f"- P-x41a (the executor's own read: ENTRAINABLE via both legs): "
        f"{'HIT' if verdict == 'ENTRAINABLE' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        f"- P-x41a-rider (REFINED-UPPER-RAIL): "
        f"{'HIT' if rider['verdict'] == 'REFINED-UPPER-RAIL' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        f"- the dispatch's lab lean (ENTRAINABLE, weakly): "
        f"{'HIT' if verdict == 'ENTRAINABLE' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        "",
        "## Catches / disclosures\n",
        "- THE LEAN'S WORDING: the dispatch's lean says 'lr_m -> 0 on "
        "the weak phase'; the committed trace puts the zero doses on "
        "the STRONG-read phase (the pre-read exceeds the baseline -> "
        "deficit 0). Frozen verbatim regardless; the operationalizations "
        "carry the read-level fact.",
        "- LEG-A LENGTH: 150 steps (3 events at the doubled spacing) vs "
        "the dispatch's '~100' — a birth decision (2 events cannot show "
        "a re-lock pattern), disclosed at birth.",
        "- LEG-A'S SCHEDULE TAIL: from ~step 929 the house cosine "
        "decays below the corpus cap; the t950 event sits in the weaker-"
        "wash tail (recorded per-step in G_SCHED); the primary events "
        "t850/t900 are cap-bound.",
        "- NO TWIN, BY DESIGN: the 2-cycle is an ERROR-GATED-arm "
        "property; the sanctuary twin has no events and no cycle (the "
        "family's x-ratio co-read not applicable).",
        "- THE LEG POOLS are length-scaled (the per-step/per-event "
        "shares identical to e339's committed leg-2 — the wash "
        "protocol-matched; the unscaled pool rejected at birth as a "
        "confound).",
        "- n=1 per leg, ONE organism, one lineage — the phase classes "
        "are deterministic continuations (not draws), but the cycle "
        "structure itself is one slot's realization.",
        "- Checkpoints live INSIDE runs/x41/. No NOTES/THINKING/QUEUE/"
        "STATE edits (the heartbeat folds this cell).",
        "",
        "*This cell does not edit NOTES/THINKING/QUEUE/STATE — the "
        "heartbeat folds.*",
    ]
    (RD / ("smoke_REPORT.md" if SMOKE else "REPORT.md")).write_text(
        "\n".join(lines), encoding="utf-8")

    # ================= P12: the final metrics ===========================
    metrics["legs"] = {k: dict(leg_results[k]) for k in LEG_ORDER}
    metrics["honesty"] = {
        "intervention_not_logits": "the three legs share ONE resumed "
                                   "base (e339's committed t800 state, "
                                   "md5 + repro gated) and the same "
                                   "continued stream (bit-identical "
                                   "draws); the ONLY deltas are the "
                                   "gate parameters M and lr_m_max",
        "the_branch_disclosure": "the legs are independent branches of "
                                 "one base, not a chain — no leg "
                                 "resumes another's end state "
                                 "(digest-verified staging)",
        "the_pool_disclosure": "the length-scaled pools keep the "
                               "per-step/per-event equal-share "
                               "reservations identical to e339's "
                               "committed leg-2; asserted via the "
                               "applied-lr medians",
        "n_and_scope": "n=1 per leg, ONE organism, one lineage; the "
                       "phase classes are deterministic continuations "
                       "of the committed state",
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
            "executed verbatim under the disclosed leg re-binding)",
            "lab/e339_long_landing.py (the t800 parent; import only)",
            "lab/e261_rank_ladder.py (the burst machinery + rooms)",
        ],
        "machinery": {
            "leg_driver": "e336's chunked_errorgated_phase EXECUTED "
                          "VERBATIM under the disclosed leg re-binding "
                          "(PHASE_STEPS/MILESTONES/WIN1/WIN2/"
                          "MAINT_EVERY) + the per-leg lr_m_max argument",
            "staging": "THIS file's staged_leg_state (the only "
                       "protocol-touching transformation; "
                       "digest-verified)",
            "cons": "NONE (the bars read the cycle's structure only)",
            "twin": "NONE (by design — disclosed)",
        },
        "checkpoints": {
            "resumed_from": [str(base_src)],
            "e339_metrics": str(E339_METRICS),
            "leg_resumes": {k: str(v) for k, v in leg_ck_paths.items()},
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
                    "tagged x41:<LEG>",
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["outputs"] = [str(fig_path), str(RD / "REPORT.md"),
                          str(RD / "metrics.json")]
    metrics["status"] = (f"COMPLETE — adjudicated {verdict}"
                         + (f"; rider {rider['verdict']}" if not SMOKE
                            else "")
                         + (" (SMOKE)" if SMOKE else "")
                         + f" ({common.now_iso()})")
    save_json(RD / "metrics.json", metrics)
    log(f"P12 DONE: {fig_path.name} + REPORT.md + metrics.json — "
        f"verdict {verdict}; rider {rider['verdict']}")


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
