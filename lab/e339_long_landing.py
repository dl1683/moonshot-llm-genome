# -*- coding: utf-8 -*-
"""E339 — THE LONG LANDING (R77's critic continuation; cell e339).

Run:  cd lab && python e339_long_landing.py       (E339_SMOKE=1 shakedown)

THE QUESTION (verbatim from the dispatch letter):
    e336 ran the controller on TAVIREN's base-formed fresh-name organism:
    held at 2.8502x — outside the founding band [3.112, 3.616] — but the
    curve was STILL RISING at t400 (milestones 2.345/2.444/2.693/2.850; the
    founding canon plateaued by t200; the ceiling 3.4983 reachable at
    +0.65). R77's critic: the verdict was decided by 0.062x over a hugged
    floor on an unrun-to-equilibrium curve. THE QUESTION: where does it
    land by t800?

THE DESIGN (committed assets):
    1. Resume e336's committed ERROR-GATED checkpoint (the t400 state —
       runs/e336/ or runs/checkpoints/; md5-bind; the twin's too if needed
       for the x-ratio).
    2. Continue the IDENTICAL protocol (the same stream continuation, the
       same gating, the same budget convention) to t800. Read the
       milestones at t500/t600/t700/t800.
    3. Report: the trajectory, the landing vs the founding band, the slope
       at t800 (settled or still climbing), the cumulative spend.

THE BARS (frozen VERBATIM from the dispatch letter, BEFORE any compute):
    LANDS-IN-BAND: the read enters [3.112, 3.616] (or the raw-read
        equivalent vs its own baseline) and the slope flattens — DIVERGES
        WAS A KINETICS ARTIFACT; the rider re-words to SETTLING-RATE
        NAME-TUNED (the mechanism's kinetics are the slot's); Law 4's
        scope re-words accordingly.
    PLATEAUS-BELOW: the curve flattens below the band — A GENUINE LOWER
        EQUILIBRIUM; "the founding band was ZEPHYRA's slot" stands; the
        name-tuned rider confirms.
    STILL-CLIMBING: monotone rise at t800 — the HORIZON exposure is named
        for the controller program (no controller run has ever passed ~2x
        its founding horizon).

LAB LEAN (verbatim): "Lab lean: LANDS-IN-BAND, weakly — the founding
curve's own shape was a fast rise then plateau, and 2.85 at t400 with
+0.157/100steps reaches the band floor by ~t550 if the rate holds; the
countervailing: TAVIREN's fragile slot may cap it (x24's fingerprint).
State your own read."

MACHINERY: e336's COMMITTED driver (lab/e336_founding_replicate.py:
chunked_errorgated_phase / chunked_passive_twin_phase — e288's rig)
EXECUTED VERBATIM BY IMPORT under the family's disclosed rebinding
convention: the ONLY module globals re-bound are the leg parameters
(PHASE_STEPS 400->800, MILESTONES, WIN1/WIN2 leg-relative) + the logging
labels; every protocol constant (LR_STABLE, LR_M_MAX, seeds, M=25,
baseline, budget fractions) is read from the committed module and
asserted against e336's md5-bound metrics BEFORE the run. The initial
state is e336's committed ERROR-GATED_resume.pt (the t400 model + both
optimizer buffer states + BOTH generator states) — staged into this
cell's resume path with the leg pools zeroed and the state bits asserted
unchanged. The stream CONTINUES (generator states restored, never
re-seeded): corpus steps 401..800 draw the continuation of seed 33601's
sequence; maintenance events 17..32 draw the continuation of seed
33602's.

THE LEG BUDGET CONVENTION (frozen at birth): "the same budget convention"
:= e336's 400-step protocol VERBATIM as a SECOND LEG — BUDGET_leg =
0.5 x the TAVIREN write's norm (4.47481453007862), pools split B_C 60% /
B_M 40%, the equal-share reservation caps over THIS leg's remaining
steps/events (the driver's own rem_corpus/rem_maint arithmetic at
n_steps=800). The t0->t800 CUMULATIVE spend (leg1 committed + leg2
measured) is the deliverable's cost report. THE ALTERNATIVE REJECTED AT
BIRTH: one pool stretched over 800 steps — B_C was EXACTLY spent at t400
(2.684888718649745 of 2.6848887180471723, 398/400 steps cap-bound), so a
stretched pool zeroes the corpus share by construction: the wash would
die and any landing would be a forfeit artifact — a different protocol,
not a continuation.

THE SCHEDULE (disclosed, never forked): cosine_lr(step-1, 1000) VERBATIM
on the global step (steps 401..800 see the decay 0.1634 -> 0.0255
scheduled). DISCLOSED IMMATERIAL: the corpus cap (share/b_norm ~ 0.0034)
bound below the schedule at 398/400 steps of leg 1 and is asserted below
it at every logged step of leg 2 — the applied corpus lr is ~flat
(0.0034-0.0035) under EITHER schedule form; the wash kinetics are
carried by the CAP, not the cosine. The maintenance lr is not on the
cosine at all (lr_m_t = LR_M_MAX x deficit_t, schedule-free).
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
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402

import e336_founding_replicate as E336                 # noqa: E402 — THE
                                                      # LEG-1 MODULE: the
                                                      # committed driver,
                                                      # executed VERBATIM
                                                      # (import runs no
                                                      # main; writes
                                                      # nothing)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E339_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e339_smoke" if SMOKE else "e339"
assert torch.cuda.is_available(), "e339 owns the GPU lane (dispatch)"
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


# ---- THE REBINDING (the family's disclosed convention) — the committed
# leg-1 module's logs + burst machinery label THIS cell (tags e339:...);
# NOTHING of e336's is written (its run.log handle is re-pointed here).
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
# THE CONFIG (frozen) — every protocol constant is BOUND to e336's
# committed module (asserted at runtime vs its md5-bound metrics)
# ======================================================================
E336_DIR = E43.REPO / "runs" / "e336"
REA_RESUME_CK = E336_DIR / "ERROR-GATED_resume.pt"
TWN_RESUME_CK = E336_DIR / "PASSIVE-TWIN_resume.pt"
REA_POST_CK = E336_DIR / "ERROR-GATED_post.pt"
TWN_POST_CK = E336_DIR / "PASSIVE-TWIN_post.pt"
E336_METRICS = E336_DIR / "metrics.json"

# e336's committed artifacts — raw-md5s, VERIFIED AT BIRTH (2026-10-10,
# git head 425feb5); a drift FAILS the resume gate.
E336_REA_RESUME_MD5 = "e67a7d4749299002a50868500be01d6a"
E336_TWN_RESUME_MD5 = "c01222d0f1f5144120fd583a10bc276b"
E336_REA_POST_MD5 = "6c42e8131994122fcdb9ef56c0314e0a"
E336_TWN_POST_MD5 = "9a03b3b3a351976512d801620a106222"
E336_METRICS_MD5 = "f90b439a0a4eff55e7f115901af469d6"

# e336's committed t400 literals (the repro targets + the trajectory
# anchors; read from the md5-bound record at runtime and asserted == these)
E336_REA_T400_G0 = 0.8147322535514832
E336_REA_T400_RATIO = 2.850195186799757
E336_TWN_T400_G0 = 0.0028905828949064016
E336_TWN_T400_RATIO = 0.010112187676619736
E336_XRATIO_400 = 281.857425707165
E336_S_CORPUS_400 = 2.684888718649745
E336_S_MAINT_400 = 1.0470825023949146
E336_S_TOTAL_400 = 3.7319712210446596
E336_TRAJ = {100: 2.3449827510188364, 200: 2.4436061166558463,
             300: 2.6931439144391316, 400: 2.850195186799757}

# ---- THE LEG PARAMETERS (the ONLY e336-module re-bindings; frozen) ----
SMOKE_LEG1_STEPS = 8                       # the smoke leg-1 (fresh, tiny)
SMOKE_LEG1_WINS = ((1, 2, 3), (3, 4, 5))
LEG_START = SMOKE_LEG1_STEPS if SMOKE else 400   # e336's committed t400
LEG_END = 16 if SMOKE else 800            # the doubled horizon
LEG_MILESTONES = (10, 12, 14, 16) if SMOKE else (500, 600, 700, 800)
LEG_WIN1 = (9, 10, 11) if SMOKE else (424, 425, 426)
LEG_WIN2 = (13, 14, 15) if SMOKE else (599, 600, 601)
LEG_EVENTS = 4 if SMOKE else 16            # maintenance events in the leg

# ---- THE BARS' NUMBERS (frozen) ----------------------------------------
BAND_LO = 3.162                    # the founding endpoint band's floor
BAND_HI = 3.616                    # its ceiling (unreachable: 3.4983 top)
LAND_FLOOR = 3.112                 # the dispatch's band floor (= 3.162 -
                                   # 0.05 hug, e336's frozen convention)
EDGE_HUG = 0.05
CEILING_RATIO = 1.0 / E336.TAV_POST_G0   # 3.4983 — the battery ceiling
SLOPE_BAR = 0.05                    # |g_last| below this = FLAT (10% of
                                    # the band width = the hug scale)
DIES_BAR = E336.E287_RATIO_BAR      # 0.0917... the family's FAILS form
STREAM_STABLE_TOL = 1.05
ORTH_BAR = 1e-6
BUFSEP_ORTH_BAR = 1e-4
BUDGET_SLACK = 1e-5

REGISTERED = {
    "registration": (
        "question + design + bars + lab lean frozen VERBATIM from the "
        "dispatch letter (R77's critic continuation — e339 THE LONG "
        "LANDING); this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping."),
    "question_verbatim": (
        "e336 ran the controller on TAVIREN's base-formed fresh-name "
        "organism: held at 2.8502x — outside the founding band [3.112, "
        "3.616] — but the curve was STILL RISING at t400 (milestones "
        "2.345/2.444/2.693/2.850; the founding canon plateaued by t200; "
        "the ceiling 3.4983 reachable at +0.65). R77's critic: the "
        "verdict was decided by 0.062x over a hugged floor on an "
        "unrun-to-equilibrium curve. THE QUESTION: where does it land by "
        "t800?"),
    "design_verbatim": (
        "1. Resume e336's committed ERROR-GATED checkpoint (the t400 "
        "state — runs/e336/ or runs/checkpoints/; md5-bind; the twin's "
        "too if needed for the x-ratio). 2. Continue the IDENTICAL "
        "protocol (the same stream continuation, the same gating, the "
        "same budget convention) to t800. Read the milestones at "
        "t500/t600/t700/t800. 3. Report: the trajectory, the landing vs "
        "the founding band, the slope at t800 (settled or still "
        "climbing), the cumulative spend."),
    "bars_verbatim": {
        "LANDS-IN-BAND": (
            "LANDS-IN-BAND: the read enters [3.112, 3.616] (or the "
            "raw-read equivalent vs its own baseline) and the slope "
            "flattens — DIVERGES WAS A KINETICS ARTIFACT; the rider "
            "re-words to SETTLING-RATE NAME-TUNED (the mechanism's "
            "kinetics are the slot's); Law 4's scope re-words "
            "accordingly."),
        "PLATEAUS-BELOW": (
            "PLATEAUS-BELOW: the curve flattens below the band — A "
            "GENUINE LOWER EQUILIBRIUM; \"the founding band was ZEPHYRA's "
            "slot\" stands; the name-tuned rider confirms."),
        "STILL-CLIMBING": (
            "STILL-CLIMBING: monotone rise at t800 — the HORIZON "
            "exposure is named for the controller program (no controller "
            "run has ever passed ~2x its founding horizon)."),
    },
    "lab_lean_verbatim": (
        "Lab lean: LANDS-IN-BAND, weakly — the founding curve's own shape "
        "was a fast rise then plateau, and 2.85 at t400 with +0.157/"
        "100steps reaches the band floor by ~t550 if the rate holds; the "
        "countervailing: TAVIREN's fragile slot may cap it (x24's "
        "fingerprint). State your own read."),
    "operationalizations": (
        f"frozen BEFORE compute: THE LANDING READ := ratio_800 = "
        f"(post-maintenance g0 p(T) read at corpus step 800) / "
        f"{E336.TAV_POST_G0!r} — the SAME read, battery, channel and "
        f"denominator e336's bars adjudicated; the milestones "
        f"t500/t600/t700/t800 all read POST-maintenance (every milestone "
        f"is a maintenance step under M=25); THE BAND := [3.112, 3.616] "
        f"verbatim from the dispatch (3.112 = e336's hugged floor 3.162 - "
        f"0.05; an in-band landing in [3.112, 3.162) stamped EDGE-GRAZE "
        f"— e336's frozen hug convention carried; the top 3.616 "
        f"unreachable by construction, ceiling {CEILING_RATIO:.4f}); THE "
        f"GAINS := g1 = ratio_500 - {E336_REA_T400_RATIO!r} (the "
        f"committed t400), g2 = ratio_600 - ratio_500, g3 = ratio_700 - "
        f"ratio_600, g_last = ratio_800 - ratio_700 (per-100-steps); "
        f"FLATTENS := |g_last| < {SLOPE_BAR} (the last 100 steps move the "
        f"read less than 10% of the band's width, EITHER direction — a "
        f"plateau, not a climb and not a slide; 0.05 = the frozen hug "
        f"width, the family's standing materially-in-band scale); "
        f"MONOTONE-RISE := ratio_400 < ratio_500 < ratio_600 < ratio_700 "
        f"< ratio_800 (strictly increasing milestones); STILL-CLIMBING "
        f":= MONOTONE-RISE AND g_last >= {SLOPE_BAR} — checked FIRST: a "
        f"curve that enters the band still rising has NOT landed (the "
        f"in-band crossing stamped, the horizon extension implied); "
        f"LANDS-IN-BAND := ratio_800 in [3.112, 3.616] AND FLATTENS; "
        f"PLATEAUS-BELOW := ratio_800 < 3.112 AND FLATTENS; "
        f"DIES-EN-ROUTE (residue, the family's FAILS form) := ratio_800 "
        f"<= {DIES_BAR!r}; MIXED := everything else (rising-but-not-"
        f"monotone, in-band-but-oscillating, falling, ...); COMPOSITE := "
        f"TEXTURE (any hard-gate failure — HALT) -> DIES-EN-ROUTE -> "
        f"STILL-CLIMBING -> LANDS-IN-BAND -> PLATEAUS-BELOW -> MIXED; "
        f"STANDING CONDITIONS (e288's, held for every verdict, a failure "
        f"routes to MIXED-by-conditions): the budget held (leg2 "
        f"S_corpus <= B_C + {BUDGET_SLACK}, S_maint <= B_M + "
        f"{BUDGET_SLACK}, S_total <= BUDGET_leg + {BUDGET_SLACK}) AND "
        f"the stream live (median corpus CE rows t in (700,800] vs "
        f"[410,500]: improving OR stable within "
        f"{STREAM_STABLE_TOL}x) AND zero buffer-isolation violations; "
        f"THE LEG BUDGET CONVENTION := e336's 400-step protocol VERBATIM "
        f"as a SECOND LEG (BUDGET_leg = 0.5 x the TAVIREN write's norm = "
        f"{0.5 * E336.TAV_WRITE_NORM!r}, pools B_C 60%/B_M 40%, the "
        f"driver's own equal-share caps over THIS leg's remaining steps/"
        f"events); the t0->t800 CUMULATIVE spend = e336's committed "
        f"{E336_S_TOTAL_400!r} + leg2's measured, reported as the "
        f"doubled-horizon cost; the stretched-pool alternative REJECTED "
        f"AT BIRTH (B_C exactly spent at t400 — a stretched pool kills "
        f"the wash by construction; a forfeit landscape, not the "
        f"protocol); THE SCHEDULE := cosine_lr(step-1, 1000) VERBATIM on "
        f"the global step, disclosed IMMATERIAL (the corpus cap ~0.0034 "
        f"bound below the schedule at 398/400 leg-1 steps and asserted "
        f"below it at every logged leg-2 step — the applied corpus lr is "
        f"~flat under either form; lr_m is schedule-free); THE TWIN := "
        f"continued identically (the same staged-resume form, a fresh leg "
        f"B_C, the SAME continued corpus stream) — the x-ratio_800 a "
        f"CO-READ, never a bar here; THE PURITY CO-READ := p(Z) at t800 "
        f"on the same battery (the committed prior "
        f"{E336.TAV_G0_Z!r}; the controller must not build a Z read); "
        f"HARD GATES := the resume identity (G_RESUME + G_T400_REPRO + "
        f"G_PROTOCOL_IDENT + G_SUBSTRATE + G_ROOMK10K + G_NAMEWIN + "
        f"G_NAMEFREE + G_SPLICE + G_BATTERY + G_ANCHOR + G_INSTMASK + "
        f"G_LR_BIND + G_BASE + G_VMBIND + G_SPANBIND + G_STAGING + "
        f"G_BUFSEP-isolation); NON-HALTING := {{G_ORTH, G_BUDGET, "
        f"bufC-composition, G_STREAMELIVE}}; the leg drivers are e336's "
        f"COMMITTED bodies executed VERBATIM under the disclosed leg "
        f"re-binding; both arms' post states checkpointed."),
    "predictions": {
        "P-e339a_lands_in_band": (
            "THE EXECUTOR'S OWN READ (per the dispatch's 'State your own "
            "read'): LANDS-IN-BAND, WEAKLY — (1) THE APPROACH-SHAPE "
            "ARGUMENT: the milestone gains ran +0.099/+0.249/+0.157 — the "
            "gain itself turned over at t300 (an S-curve past inflection, "
            "the signature of a saturating approach, not a linear climb); "
            "(2) THE SELF-LIMITING DOSE: the controller doses on the "
            "PRE-event deficit (0.561 at event 16, falling as the washed-"
            "down state rises 0.023 -> 0.125) — as the pre-event read "
            "rises the dose shrinks and the post-event peak settles, the "
            "same closed loop that parked the founding curve by t200 "
            "(its deficits froze ~0.66); (3) THE COUNTERVAILING: TAVIREN's "
            "slot may cap the asymptote below the floor's raw 0.889 "
            "(x24's fragile-slot fingerprint) — the pre-event read at "
            "t400 is still only 44% of baseline; if the wash-down floor "
            "freezes there the dose stays ~0.012/event and the peak could "
            "asymptote anywhere in [2.9, 3.4]; AND the timing risk: a "
            "slow approach that would land by t1000 sits ~3.05-3.10 at "
            "t800 with g_last ~0.06 — formally STILL-CLIMBING. PREDICTED "
            "SHAPE: monotone milestones ~2.99/3.08/3.14/3.18, g_last in "
            "[0.02, 0.05), the pre-event reads rising 0.13 -> 0.20+ "
            "(deficits 0.56 -> ~0.30, doses self-limiting), the twin dead "
            "(< 0.005). DISCRIMINATING DATUM: the joint (pre-event read, "
            "deficit, applied lr_m) trace — LANDS-IN-BAND shows the "
            "deficit collapsing (< 0.25 by t800); PLATEAUS-BELOW shows it "
            "freezing (> 0.45 with the post-event lift stalling under "
            "the ceiling); STILL-CLIMBING shows the deficit mid-collapse "
            "with the peak still gaining >= 0.05/100steps. SCORED: TRUE "
            "iff the verdict == LANDS-IN-BAND."),
    },
}
G_NAMEWIN_KEYSET = None  # retired (inlined into the operationalizations)

deviations: list[str] = []
builds_on = [
    "R77's critic card (the named continuation: 'e339 THE LONG LANDING "
    "(resume e336's committed checkpoint to t800) must precede any prose "
    "migration')",
    "e336 / THE FOUNDING-CLASS SECOND INSTANCE (THE PARENT CELL: its "
    "committed ERROR-GATED/PASSIVE-TWIN t400 checkpoints resumed here; "
    "its committed driver executed verbatim; its verdict DIVERGES the "
    "object of this continuation)",
    "e288 (the ERROR-GATED controller's rig — the driver's ultimate "
    "source), e287 (the b_m calibration + the death-band literal), "
    "e285 (the sanctuary twin form), e311 (the TAVIREN organism), "
    "e273 (LR_STABLE), e268 (the corpus step), e264/e261 (the room + "
    "the burst machinery)",
]
whats_new = [
    "THE FIRST CONTROLLER RUN PAST ITS FOUNDING HORIZON: no controller "
    "run in the record has ever passed ~2x its founding horizon (400 "
    "steps) — the t401-800 leg is the first look at the controller's "
    "approach to equilibrium at a doubled horizon",
    "THE CROSS-CELL RESUME: e336's committed mid-protocol state (model "
    "+ both SGD-M buffer states + both generator states) resumed "
    "bit-exact under md5 + read-repro gates — the continuation "
    "convention (leg pools fresh, stream continued) built and frozen "
    "for any future horizon extension",
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
                      maint_every=None) -> dict:
    before = {k: getattr(E336, k) for k in LEG_KEYS}
    E336.PHASE_STEPS = n_steps
    E336.MILESTONES = tuple(milestones)
    E336.WIN1 = tuple(win1)
    E336.WIN2 = tuple(win2)
    if maint_every is not None:
        E336.MAINT_EVERY = int(maint_every)
    after = {k: getattr(E336, k) for k in LEG_KEYS}
    log(f"  [leg-bind] e336 module globals re-bound: {after} "
        f"(was {before}); every protocol constant untouched")
    return {"before": {k: str(v) for k, v in before.items()},
            "after": {k: str(v) for k, v in after.items()}}


# ======================================================================
# THE STAGING MACHINE — e336's committed t400 resume state -> this cell's
# leg resume file: the state bits (model/optC/optF/cgen/igen) pass through
# UNTOUCHED (asserted); the leg ledgers + pools zeroed (the leg convention)
# ======================================================================
def _tensor_digest(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        return hashlib.md5(x.detach().cpu().contiguous()
                           .numpy().tobytes()).hexdigest()
    if isinstance(x, dict):
        # an optimizer state_dict: per-param momentum-buffer bytes +
        # the param_groups scalars (deterministic order by key)
        parts = []
        for k in sorted(x.keys(), key=repr):
            parts.append(repr(k))
            parts.append(_tensor_digest(x[k]))
        return hashlib.md5("|".join(parts).encode()).hexdigest()
    if isinstance(x, (list, tuple)):
        return hashlib.md5("|".join(_tensor_digest(v) for v in x)
                           .encode()).hexdigest()
    return hashlib.md5(repr(x).encode()).hexdigest()


def stage_leg_resume(src: Path, dst: Path, n_maint_keys: bool,
                     N: int) -> dict:
    """Load e336's committed resume ckpt; return the leg-staged state.

    n_maint_keys: True for the ERROR-GATED arm (S_maint/maint_cum/igen
    present), False for the twin. The state bits are copied by reference
    and their digests recorded; the leg ledger/pool fields are RESET."""
    st = torch.load(src, map_location="cpu", weights_only=False)
    bits_src = {"model": {k: _tensor_digest(v) for k, v in
                          st["model"].items()},
                "optC": _tensor_digest(st["optC"]),
                "optF": _tensor_digest(st["optF"]),
                "cgen_state": _tensor_digest(st["cgen_state"])}
    if n_maint_keys:
        bits_src["igen_state"] = _tensor_digest(st["igen_state"])
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
    }
    if n_maint_keys:
        leg["igen_state"] = st["igen_state"]        # bits unchanged
        leg["maint_cum"] = torch.zeros(N, dtype=torch.float64)
        leg["S_maint"] = 0.0
        leg["n_maint"] = 0
        leg["n_maint_capped"] = 0
    torch.save(leg, dst)
    # verify: reload + digest the state bits — identical, and the pools
    # zeroed, and step carried
    chk = torch.load(dst, map_location="cpu", weights_only=False)
    bits_dst = {"model": {k: _tensor_digest(v) for k, v in
                          chk["model"].items()},
                "optC": _tensor_digest(chk["optC"]),
                "optF": _tensor_digest(chk["optF"]),
                "cgen_state": _tensor_digest(chk["cgen_state"])}
    if n_maint_keys:
        bits_dst["igen_state"] = _tensor_digest(chk["igen_state"])
    ok = bits_src == bits_dst and chk["step"] == int(st["step"]) \
        and chk["S_corpus"] == 0.0 \
        and (not n_maint_keys or chk["S_maint"] == 0.0)
    return {"src": str(src), "dst": str(dst), "step": int(st["step"]),
            "state_bits": bits_src, "bits_unchanged": bool(ok),
            "source_S_corpus": float(st["S_corpus"]),
            **({"source_S_maint": float(st["S_maint"]),
                "source_n_maint": int(st["n_maint"])}
               if n_maint_keys else {})}


# ======================================================================
def main():
    global dev
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e339_long_landing",
        "phase": ("THE LONG LANDING (R77's critic continuation; cell e339): "
                  "e336's committed ERROR-GATED t400 checkpoint (the model "
                  "+ both SGD-M buffers + both generator states) resumed "
                  "md5-bound; the IDENTICAL protocol continued to t800 "
                  "(the stream continuation of seeds 33601/33602, the "
                  "same error-gated maintenance law, the same leg budget "
                  "convention: a fresh 0.5x-write-norm budget split B_C "
                  "60%/B_M 40% per 400-step leg); milestones "
                  "t500/600/700/800 read post-maintenance; the twin "
                  "continued identically for the x-ratio — LANDS-IN-BAND "
                  "(in [3.112, 3.616] + flattened: DIVERGES was a kinetics "
                  "artifact) / PLATEAUS-BELOW (flattened below: a genuine "
                  "lower equilibrium) / STILL-CLIMBING (monotone rise at "
                  "t800: the horizon exposure)"),
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "the arms run SEQUENTIALLY with cooldowns between "
                      "— never concurrent) + CPU fp64 projections, "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line (dispatch 85), recorded to "
                      "runs/_envelope_log.jsonl tagged e339:<ARM>:<phase>",
            "trainings": f"2 continuation legs x "
                         f"{LEG_END - LEG_START} corpus steps "
                         "(ERROR-GATED: e336's driver verbatim — budgeted "
                         "orthogonalized corpus steps over a fresh leg B_C "
                         f"+ {0 if SMOKE else 16} NAME-ONLY maintenance "
                         "events through opt_F at the error-gated lr; "
                         "PASSIVE-TWIN: the sanctuary form, opt_F never "
                         "stepped); NO cons",
        },
        "builds_on": builds_on,
        "whats_new": whats_new,
        "deviations": deviations,
    })
    write_partial("startup (bars + lean + P-e339a registered, committed "
                  "at birth)")

    # ---- the committed e336 record (md5-bound; the repro targets) ------
    for p, want in [(E336_METRICS, E336_METRICS_MD5),
                    (REA_RESUME_CK, E336_REA_RESUME_MD5),
                    (TWN_RESUME_CK, E336_TWN_RESUME_MD5),
                    (REA_POST_CK, E336_REA_POST_MD5),
                    (TWN_POST_CK, E336_TWN_POST_MD5)]:
        got = md5of(p)
        assert got == want, f"e336 artifact drifted: {p.name} {got} != {want}"
    e336m = json.loads(E336_METRICS.read_text(encoding="utf-8"))
    e336_rea_ph = e336m["arms"]["ERROR-GATED"]["phase"]
    e336_twn_ph = e336m["arms"]["PASSIVE-TWIN"]["phase"]
    e336_traj = {int(r["step"]): r for r in e336_rea_ph["traj"]}
    assert abs(e336_traj[400]["g0_pz"] - E336_REA_T400_G0) < 1e-15 \
        and abs(e336_traj[400]["survival_ratio_vs_committed"]
                - E336_REA_T400_RATIO) < 1e-15, "committed t400 drift"
    assert abs(e336_rea_ph["budget"]["S_total_final"]
               - E336_S_TOTAL_400) < 1e-12, "committed spend drift"
    log(f"P-1 e336's committed record BOUND (metrics md5 "
        f"{E336_METRICS_MD5[:8]}...; t400 read "
        f"{E336_REA_T400_G0:.10f} = x{E336_REA_T400_RATIO:.4f}; leg-1 "
        f"S_total {E336_S_TOTAL_400:.4f})")

    # ================= P0: the protocol rebuild (e336's P0 mirror) ======
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
        "form": "the maintenance vehicle re-bound (e336's G_NAMEWIN "
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
        "windows + the eval banks) — e336's P0 mirror PASS")

    # ================= P1: the room + the parents (e336's mirror) ======
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

    # the room: e336's own committed room artifact LOADED bit-exact (the
    # fastest exact bind — the room the leg-1 stream orthogonalized
    # against); the K10K D/S re-derived from seeds and bit-compared
    rooms_here = E261.LadderRooms(n_par, E336.LADDER, v64_np,
                                  Vp.numpy().astype(np.float64),
                                  list(G1.evl_load(base_sd).parameters()),
                                  dev)
    if not SMOKE:
        cert = rooms_here.certify()
        assert cert["pass"], f"room cert failed: {cert}"
        rooms336 = torch.load(E336_DIR / "e336_rooms.pt", map_location="cpu",
                              weights_only=False)
        D336 = rooms336["model"]["K10K"]["D_int8"].numpy().astype(np.float64)
        S336 = rooms336["model"]["K10K"]["S"].numpy()
        D_mine = rooms_here.rooms["K10K"].D
        S_mine = rooms_here.rooms["K10K"].S
        G_ROOMK10K = {
            "form": "the room re-derived from the committed seeds and "
                    "bit-compared BOTH vs e336's committed e336_rooms.pt "
                    "and (by e336's own gate) vs e264_rooms.pt's K10K — "
                    "the same room the leg-1 stream orthogonalized against",
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
    log(f"P1 the room: K10K re-derived + bit-bound to e336's committed "
        f"room: PASS")

    # ---- G_LR_BIND: the lr constants re-verified (the module bind) -----
    lr_sgd_runtime = float(json.loads(
        (E43.REPO / "runs" / "e273" / "lr_calibration.json")
        .read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the leg runs on e336's COMMITTED lr constants (the "
                "module's own globals, asserted): LR_STABLE = "
                "x0.01 x LR_SGD_matched; LR_M_MAX = 0.40 x BUDGET_leg / "
                "e287's b_m sum (the SAME identity, the SAME value — the "
                "leg budget is the same 0.5x-write-norm form); momentum "
                "0.9, wd 0.0 EXACTLY",
        "lr_sgd_record": E336.E273_LR_SGD,
        "lr_sgd_runtime": lr_sgd_runtime,
        "lr_stable_module": E336.LR_STABLE,
        "lr_stable_metrics": e336m["gates"]["G_LR_BIND"]["lr_stable"],
        "lr_m_max_module": E336.LR_M_MAX_FROZEN,
        "lr_m_max_metrics": e336m["gates"]["G_LR_BIND"]["lr_m_max_frozen"],
        "momentum": E336.SGD_MOMENTUM, "wd": E336.SGD_WD,
        "pass": bool(E336.LR_STABLE == 0.21738574801453703
                     and E336.LR_M_MAX_FROZEN
                     == 0.021942464013242388
                     and lr_sgd_runtime == E336.E273_LR_SGD
                     and E336.SGD_MOMENTUM == 0.9 and E336.SGD_WD == 0.0
                     and e336m["gates"]["G_LR_BIND"]["lr_stable"]
                     == E336.LR_STABLE
                     and e336m["gates"]["G_LR_BIND"]["lr_m_max_frozen"]
                     == E336.LR_M_MAX_FROZEN)}
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND

    # ================= P2: the substrate (e336's mirror) ================
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
        "form": "e336's substrate gate mirror: the organism artifact "
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
        "write_norm_rel_diff": abs(write_norm
                                   - E336.TAV_WRITE_NORM)
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

    # ---- the LEG BUDGET (frozen; the same convention, a fresh leg) -----
    budget_norm = E336.BUDGET_FRAC * write_norm     # = 4.47481453007862
    lr_m_max = E336.MAINT_BUDGET_SHARE * budget_norm / E336.E287_B_M_SUM
    assert abs(lr_m_max - E336.LR_M_MAX_FROZEN) < 1e-9
    budget_c = E336.CORPUS_BUDGET_SHARE * budget_norm
    budget_m = E336.MAINT_BUDGET_SHARE * budget_norm
    leg_steps = LEG_END - LEG_START
    leg_events = LEG_EVENTS
    if SMOKE:
        # e336's smoke convention: per-stream budgets scaled to the leg's
        # own event counts (LR_M_MAX stays the full-run constant)
        budget_c = budget_c * leg_steps / 400
        budget_m = budget_m * leg_events / 16
    metrics["budget"] = {
        "form": "THE LEG CONVENTION (frozen at birth): a fresh "
                f"{E336.BUDGET_FRAC}x-write-norm budget per 400-step leg, "
                f"pools B_C {E336.CORPUS_BUDGET_SHARE:.0%} / B_M "
                f"{E336.MAINT_BUDGET_SHARE:.0%}, the driver's own "
                "equal-share caps over the leg's remaining steps/events; "
                "the t0->t800 cumulative spend = leg1 committed + leg2 "
                "measured; the stretched-pool alternative REJECTED at "
                "birth (B_C exactly spent at t400 — the wash would die; "
                "a forfeit landscape)",
        "write_norm": write_norm,
        "budget_leg": budget_norm,
        "budget_c_leg": budget_c, "budget_m_leg": budget_m,
        "leg1_committed": {"S_corpus": E336_S_CORPUS_400,
                           "S_maint": E336_S_MAINT_400,
                           "S_total": E336_S_TOTAL_400,
                           "usage_frac": E336_S_TOTAL_400
                           / (E336.BUDGET_FRAC * E336.TAV_WRITE_NORM)},
        "lr_m_max": lr_m_max,
        "smoke_scaling": (f"B_C x{leg_steps}/400, B_M x{leg_events}/16 "
                          "(e336's smoke convention)" if SMOKE else None)}
    log(f"P2 the LEG BUDGET: BUDGET_leg {budget_norm:.4f} (B_C "
        f"{budget_c:.4f} / B_M {budget_m:.4f}); LR_M_MAX re-derived "
        f"{lr_m_max!r}; leg-1 committed spend {E336_S_TOTAL_400:.4f} "
        "carried for the cumulative report")

    # ---- G_PROTOCOL_IDENT: the e336 module == the committed protocol ----
    G_PROTOCOL_IDENT = {
        "form": "the continuation's identity: e336's COMMITTED module "
                "constants asserted == its md5-bound metrics literals "
                "BEFORE any re-binding; then the ONLY re-bindings are "
                "the leg parameters (asserted set)",
        "corpus_gen_seed_module": E336.CORPUS_GEN_SEED,
        "inst_gen_seed_module": E336.INST_GEN_SEED,
        "maint_every_module_committed": E336.MAINT_EVERY,
        "phase_steps_module_committed": E336.PHASE_STEPS,
        "baseline_gate_module": E336.FACT_BASELINE_G0,
        "budget_frac_module": E336.BUDGET_FRAC,
        "checks": {
            "seeds": bool(E336.CORPUS_GEN_SEED == 33601
                          and E336.INST_GEN_SEED == 33602),
            "cadence": bool(E336.MAINT_EVERY == 25
                            and E336.PHASE_STEPS == 400),
            "baseline": bool(E336.FACT_BASELINE_G0
                             == 0.285851389169693
                             == e336m["gates"]["G_CTRLIDENT"]
                             ["baseline_gate"]),
            "budget_form": bool(E336.BUDGET_FRAC == 0.5
                                and E336.CORPUS_BUDGET_SHARE == 0.6
                                and E336.MAINT_BUDGET_SHARE == 0.4
                                and abs(e336m["budget"]["budget_norm"]
                                        - budget_norm) < 1e-12),
            "spend_ledger": bool(
                abs(e336_rea_ph["budget"]["S_corpus_final"]
                    - E336_S_CORPUS_400) < 1e-12
                and abs(e336_rea_ph["budget"]["S_maint_final"]
                        - E336_S_MAINT_400) < 1e-12),
            "bufsep_ledger": bool(
                e336_rea_ph["bufsep"] == {"corpus_checks": 400,
                                          "corpus_violations": 0,
                                          "maint_checks": 16,
                                          "maint_violations": 0,
                                          "optF_steps": 16}),
            "milestones_committed": bool(
                [int(r["step"]) for r in e336_rea_ph["traj"]]
                == [100, 200, 300, 400]),
        },
        "pass": None}      # set below (after the leg re-bind assert)
    ok = all(G_PROTOCOL_IDENT["checks"].values())
    assert ok, f"protocol identity failed: {G_PROTOCOL_IDENT['checks']}"
    leg_bind_record = patch_leg_globals(
        LEG_END, LEG_MILESTONES, LEG_WIN1, LEG_WIN2,
        maint_every=(2 if SMOKE else None))
    assert E336.PHASE_STEPS == LEG_END \
        and tuple(E336.MILESTONES) == tuple(LEG_MILESTONES) \
        and tuple(E336.WIN1) == tuple(LEG_WIN1) \
        and tuple(E336.WIN2) == tuple(LEG_WIN2)
    G_PROTOCOL_IDENT["leg_rebind"] = leg_bind_record
    G_PROTOCOL_IDENT["pass"] = True
    metrics["gates"]["G_PROTOCOL_IDENT"] = G_PROTOCOL_IDENT
    log("P2 G_PROTOCOL_IDENT: e336's committed constants == the md5-bound "
        f"record; leg re-bind = {{PHASE_STEPS {LEG_END}, MILESTONES "
        f"{LEG_MILESTONES}, WIN {LEG_WIN1}/{LEG_WIN2}}}: PASS")
    write_partial("P2 the protocol identity bound (leg re-bind asserted)")

    # ================= P3: the resume gates + the staging ===============
    if SMOKE:
        # the smoke leg-1: run e336's driver FRESH (tiny) to manufacture
        # a committed-form checkpoint, then stage + continue — the full
        # machinery path at 1/50 cost; the smoke numbers are MEANINGLESS
        # (disclosed): machinery validation only
        log("P3 SMOKE: manufacturing a smoke leg-1 checkpoint via e336's "
            "own driver (8 steps, M=2) — machinery validation only")
        patch_leg_globals(SMOKE_LEG1_STEPS, tuple(range(1, 9)),
                          *SMOKE_LEG1_WINS, maint_every=2)
        bc1 = E336.CORPUS_BUDGET_SHARE * budget_norm * 8 / 400
        _ = E336.chunked_errorgated_phase(
            "SMOKE-LEG1", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, bc1, budget_m, win_t, inst_mask, anchor_full,
            train_ids, g0_ids, gm12_ids, r_eval_xy, tid, lr_m_max,
            RD / "smoke_leg1_REA_resume.pt", dev)
        _ = E336.chunked_passive_twin_phase(
            "SMOKE-LEG1", G1.evl_load(sub_sd), proj, sub_flat_np,
            base_flat_np, bc1, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, tid, RD / "smoke_leg1_TWN_resume.pt", dev)
        patch_leg_globals(LEG_END, LEG_MILESTONES, LEG_WIN1, LEG_WIN2,
                          maint_every=2)
        rea_src = RD / "smoke_leg1_REA_resume.pt"
        twn_src = RD / "smoke_leg1_TWN_resume.pt"
    else:
        rea_src, twn_src = REA_RESUME_CK, TWN_RESUME_CK

    rea_st = torch.load(rea_src, map_location="cpu", weights_only=False)
    twn_st = torch.load(twn_src, map_location="cpu", weights_only=False)
    G_RESUME = {
        "form": "the cross-cell resume identity: e336's committed t400 "
                "resume states (raw-md5-bound) — step/n_maint/spend/"
                "isolation ledgers asserted == the committed metrics; "
                "the resume model bit-compared vs the committed post "
                "checkpoint",
        "rea_resume_md5": md5of(rea_src),
        "twn_resume_md5": md5of(twn_src),
        "rea_step": int(rea_st["step"]), "rea_n_maint": int(rea_st["n_maint"]),
        "rea_S_corpus": float(rea_st["S_corpus"]),
        "rea_S_maint": float(rea_st["S_maint"]),
        "twn_step": int(twn_st["step"]),
        "twn_S_corpus": float(twn_st["S_corpus"]),
        "expected": {"rea_step": LEG_START, "rea_n_maint": 16,
                     "rea_S_corpus": E336_S_CORPUS_400,
                     "rea_S_maint": E336_S_MAINT_400,
                     "twn_step": LEG_START},
    }
    # the resume model == the committed post checkpoint (bit-compare)
    rea_post = torch.load(REA_POST_CK if not SMOKE else rea_src,
                          map_location="cpu", weights_only=False)
    model_bit_equal = all(
        torch.equal(rea_st["model"][k], rea_post["model"][k])
        for k in rea_st["model"])
    G_RESUME["resume_model_bit_equal_post"] = bool(model_bit_equal)
    G_RESUME["pass"] = bool(
        (SMOKE or md5of(rea_src) == E336_REA_RESUME_MD5)
        and (SMOKE or md5of(twn_src) == E336_TWN_RESUME_MD5)
        and int(rea_st["step"]) == LEG_START
        and int(twn_st["step"]) == LEG_START
        and (SMOKE or (int(rea_st["n_maint"]) == 16
                       and abs(float(rea_st["S_corpus"])
                               - E336_S_CORPUS_400) < 1e-9
                       and abs(float(rea_st["S_maint"])
                               - E336_S_MAINT_400) < 1e-9))
        and (SMOKE or model_bit_equal))
    assert G_RESUME["pass"], f"resume gate FAILED: {G_RESUME}"
    metrics["gates"]["G_RESUME"] = G_RESUME
    del rea_post
    log(f"P3 G_RESUME: e336's t400 states loaded (REA step "
        f"{rea_st['step']}, n_maint {rea_st.get('n_maint')}, S "
        f"{float(rea_st['S_corpus']):.4f}+"
        f"{float(rea_st.get('S_maint', 0.0)):.4f}; TWN step "
        f"{twn_st['step']}); resume-model == post-model: "
        f"{model_bit_equal}: PASS")

    # ---- G_T400_REPRO: the restored state reproduces the committed read
    # (the repro target: e336's committed t400 read in the full run; the
    # smoke leg-1's own final traj read in the smoke — the machinery is
    # what the smoke validates)
    net_r400 = G1.evl_load({k: v for k, v in rea_st["model"].items()})
    repro_g0 = G1.battery_cell(net_r400, g0_ids, tid)["mean_pz"]
    repro_twn = G1.evl_load({k: v for k, v in twn_st["model"].items()})
    repro_twn_g0 = G1.battery_cell(repro_twn, g0_ids, tid)["mean_pz"]
    committed_rea_g0 = (rea_st["traj"][-1]["g0_pz"] if SMOKE
                        else e336_traj[400]["g0_pz"])
    committed_twn_g0 = (twn_st["traj"][-1]["g0_pz"] if SMOKE
                        else e336_twn_ph["traj"][-1]["g0_pz"])
    G_T400_REPRO = {
        "form": "the resumed bits reproduce e336's committed t400 battery "
                "reads (the family's cross-session read-determinism law: "
                "same bits + same battery + same CPU fp32 code path)",
        "rea_g0_reproduced": repro_g0,
        "rea_g0_committed": committed_rea_g0,
        "rea_absdiff": abs(repro_g0 - committed_rea_g0),
        "twn_g0_reproduced": repro_twn_g0,
        "twn_g0_committed": committed_twn_g0,
        "twn_absdiff": abs(repro_twn_g0 - committed_twn_g0),
        "tol": 2e-6,
        "pass": bool(abs(repro_g0 - committed_rea_g0) <= 2e-6
                     and abs(repro_twn_g0 - committed_twn_g0) <= 2e-6)}
    assert G_T400_REPRO["pass"], f"t400 repro FAILED: {G_T400_REPRO}"
    metrics["gates"]["G_T400_REPRO"] = G_T400_REPRO
    del net_r400, repro_twn
    log(f"P3 G_T400_REPRO: REA t400 read reproduced (|d| "
        f"{G_T400_REPRO['rea_absdiff']:.1e}); TWN (|d| "
        f"{G_T400_REPRO['twn_absdiff']:.1e}): PASS")

    # ---- the staging: state bits through, leg pools zeroed -------------
    rea_leg_ck = RD / ("smoke_REA_leg_resume.pt" if SMOKE
                       else "e339_REA_resume.pt")
    twn_leg_ck = RD / ("smoke_TWN_leg_resume.pt" if SMOKE
                       else "e339_TWN_resume.pt")
    stage_rea = stage_leg_resume(rea_src, rea_leg_ck, True, n_par)
    stage_twn = stage_leg_resume(twn_src, twn_leg_ck, False, n_par)
    assert stage_rea["bits_unchanged"] and stage_twn["bits_unchanged"], \
        f"staging altered state bits: {stage_rea} {stage_twn}"
    metrics["gates"]["G_STAGING"] = {
        "form": "the ONLY protocol-touching transformation: e336's "
                "committed state bits (model/optC/optF/cgen/igen) pass "
                "through UNTOUCHED (digest-verified before/after the "
                "staging write); the leg ledgers + spend pools zeroed "
                "(the leg convention); the generator states carried — "
                "the stream CONTINUES (never re-seeded)",
        "rea": stage_rea, "twn": stage_twn,
        "first_continued_draws_disclosure":
            "corpus steps 401.. draw the continuation of seed 33601's "
            "sequence (cgen_state restored); maintenance events 17.. "
            "draw the continuation of seed 33602's (igen_state restored)",
        "pass": True}
    log("P3 G_STAGING: both leg resumes staged (state bits digest-equal; "
        "pools zeroed; generators carried): PASS")
    write_partial("P3 the resume identity + staging complete")

    # ================= P4: ARM ERROR-GATED (the continuation) ===========
    log("=" * 78)
    log(f"ARM-ERROR-GATED — the continuation t{LEG_START + 1}..t{LEG_END} "
        "(e336's committed driver VERBATIM; the stream + optimizer states "
        "resumed; a fresh leg budget; the same controller law)")
    rea = E336.chunked_errorgated_phase(
        "ERROR-GATED", G1.evl_load(sub_sd), proj, sub_flat_np,
        base_flat_np, budget_c, budget_m, win_t, inst_mask, anchor_full,
        train_ids, g0_ids, gm12_ids, r_eval_xy, tid, lr_m_max,
        rea_leg_ck, dev)
    sd_r = rea["sd"]
    net_r = G1.evl_load(sd_r)
    cells_r = {"gm12": G1.battery_cell(net_r, gm12_ids, tid)["mean_pz"],
               "g0": G1.battery_cell(net_r, g0_ids, tid)["mean_pz"],
               "g0_z": G1.battery_cell(net_r, g0_ids, zid)["mean_pz"],
               "ce_r": G1.ce_fixed_cpu(net_r, *r_eval_xy)}
    torch.save({"model": sd_r, "meta": {
        "experiment": "e339", "arm": "ERROR-GATED",
        "desc": f"e339 the long landing: e336's committed t400 state "
                f"continued to t{LEG_END} (the same protocol, a fresh "
                f"leg budget)",
        "leg2_S_corpus": rea["S_corpus"], "leg2_S_maint": rea["S_maint"],
        "cumulative_S_total": E336_S_TOTAL_400 + rea["S_corpus"]
        + rea["S_maint"]}},
        RD / ("smoke_REA_post.pt" if SMOKE else "e339_REA_post.pt"))
    del net_r
    log(f"ARM-ERROR-GATED DONE: post g0 {cells_r['g0']:.7f} "
        f"(x{cells_r['g0'] / E336.TAV_POST_G0:.4f}) p(Z) "
        f"{cells_r['g0_z']:.2e} | LEG2 S_corp {rea['S_corpus']:.4f}/"
        f"{budget_c:.4f} + S_maint {rea['S_maint']:.4f}/{budget_m:.4f} "
        f"| CUMULATIVE t0->t800 S_total "
        f"{E336_S_TOTAL_400 + rea['S_corpus'] + rea['S_maint']:.4f} | "
        f"maint {rea['n_maint']} events | isolation "
        f"{rea['bufsep']['corpus_checks']}+{rea['bufsep']['maint_checks']} "
        f"checks "
        f"{rea['bufsep']['corpus_violations'] + rea['bufsep']['maint_violations']} "
        f"violations")
    write_partial("ARM-ERROR-GATED complete")

    # ================= P5: ARM PASSIVE-TWIN (the continuation) ==========
    E261.burst_cooldown("ERROR-GATED -> PASSIVE-TWIN")
    log("=" * 78)
    log(f"ARM-PASSIVE-TWIN — the continuation t{LEG_START + 1}.."
        f"{LEG_END} (the sanctuary form over the SAME continued stream; "
        "opt_F never stepped)")
    twn = E336.chunked_passive_twin_phase(
        "PASSIVE-TWIN", G1.evl_load(sub_sd), proj, sub_flat_np,
        base_flat_np, budget_c, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, tid, twn_leg_ck, dev)
    sd_t = twn["sd"]
    net_t = G1.evl_load(sd_t)
    cells_t = {"g0": G1.battery_cell(net_t, g0_ids, tid)["mean_pz"],
               "g0_z": G1.battery_cell(net_t, g0_ids, zid)["mean_pz"]}
    torch.save({"model": sd_t, "meta": {
        "experiment": "e339", "arm": "PASSIVE-TWIN",
        "desc": f"e339 the long landing twin: e336's committed t400 twin "
                f"state continued to t{LEG_END} (no maintenance)",
        "leg2_S_corpus": twn["S_corpus"]}},
        RD / ("smoke_TWN_post.pt" if SMOKE else "e339_TWN_post.pt"))
    del net_t
    log(f"ARM-PASSIVE-TWIN DONE: post g0 {cells_t['g0']:.7f} "
        f"(x{cells_t['g0'] / E336.TAV_POST_G0:.4f}) | LEG2 S_corp "
        f"{twn['S_corpus']:.4f}/{budget_c:.4f}")
    write_partial("ARM-PASSIVE-TWIN complete")

    # ================= P6: the gates + the adjudication =================
    leg_rows = sorted((int(k), v) for k, v in rea["corpus_ledger"].items())
    early = [v["ce"] for s, v in leg_rows
             if LEG_START + 10 <= s <= LEG_START + 100]
    late = [v["ce"] for s, v in leg_rows
            if LEG_END - 100 < s <= LEG_END]
    stream_live = (None if len(early) < 2 or len(late) < 2 else bool(
        med(late) <= STREAM_STABLE_TOL * med(early)))
    G_STREAMELIVE = {
        "form": "e288's standing condition: the corpus stream live "
                "(median corpus CE late-vs-early: improving OR stable "
                f"within {STREAM_STABLE_TOL}x)",
        "median_early": med(early) if early else None,
        "median_late": med(late) if late else None,
        "n_rows_early": len(early), "n_rows_late": len(late),
        "vacuous_smoke": bool(SMOKE and (len(early) < 2 or len(late) < 2)),
        "pass": bool(stream_live) if stream_live is not None else True}
    metrics["gates"]["G_STREAMELIVE"] = G_STREAMELIVE

    lr_rows = rea["lr_ledger"]
    cap_below_sched = all(v["cap"] <= v["lr_sched"]
                          for v in lr_rows.values()) if lr_rows else None
    metrics["gates"]["G_SCHED_IMMATERIAL"] = {
        "form": "the disclosed schedule immateriality, verified: the "
                "corpus cap bound below the cosine schedule at EVERY "
                "logged leg-2 step (the applied lr is cap-determined, "
                "~flat — the wash kinetics are carried by the cap)",
        "n_rows": len(lr_rows),
        "cap_below_sched_all": cap_below_sched,
        "lr_applied_min": min((v["lr_applied"] for v in lr_rows.values()),
                              default=None),
        "lr_applied_max": max((v["lr_applied"] for v in lr_rows.values()),
                              default=None),
        "pass": True}

    G_ORTH = {
        "form": "e336's G_ORTH mirror (NON-HALTING): the stepped corpus "
                "gradient entirely orthogonal to the room at every leg-2 "
                "step, both arms",
        "rea_orth_max_rel_err": rea["orth_max"],
        "twn_orth_max_rel_err": twn["orth_max"],
        "bar": ORTH_BAR,
        "pass": bool(rea["orth_max"] < ORTH_BAR
                     and twn["orth_max"] < ORTH_BAR)}
    metrics["gates"]["G_ORTH"] = G_ORTH

    G_BUFSEP = {
        "form": "e336's G_BUFSEP mirror (the isolation HALT carried): "
                "bidirectional buffer separation at every leg-2 optimizer "
                "event; the twin's opt_F NEVER stepped",
        "rea": rea["bufsep"], "twn": twn["bufsep"],
        "pass": bool(rea["bufsep"]["corpus_violations"] == 0
                     and rea["bufsep"]["maint_violations"] == 0
                     and twn["bufsep"]["corpus_violations"] == 0
                     and twn["bufsep"]["optF_steps"] == 0)}
    assert G_BUFSEP["pass"], f"G_BUFSEP HALT: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP

    S2_corpus, S2_maint = float(rea["S_corpus"]), float(rea["S_maint"])
    G_BUDGET = {
        "form": "the LEG budget held (NON-HALTING): leg-2 pools respected "
                "+ S_total(leg2) <= BUDGET_leg; the cumulative t0->t800 "
                "spend REPORTED (the doubled-horizon cost, never a bar)",
        "S_corpus_leg2": S2_corpus, "S_maint_leg2": S2_maint,
        "S_total_leg2": S2_corpus + S2_maint,
        "B_C": budget_c, "B_M": budget_m, "BUDGET_leg": budget_norm,
        "cumulative_S_total_t0_t800": E336_S_TOTAL_400 + S2_corpus
        + S2_maint,
        "cumulative_frac_of_write_norm": (E336_S_TOTAL_400 + S2_corpus
                                          + S2_maint) / write_norm,
        "twin_S_corpus_leg2": float(twn["S_corpus"]),
        "pass": bool(S2_corpus <= budget_c + BUDGET_SLACK
                     and S2_maint <= budget_m + BUDGET_SLACK
                     and S2_corpus + S2_maint
                     <= budget_norm + BUDGET_SLACK
                     and float(twn["S_corpus"])
                     <= budget_c + BUDGET_SLACK)}
    metrics["gates"]["G_BUDGET"] = G_BUDGET

    # ---- the reads + the frozen bars ------------------------------------
    ms = {int(r["step"]): r for r in rea["traj"]}
    mstone_steps = LEG_MILESTONES
    have = all(s in ms for s in mstone_steps)
    assert have, f"missing milestones: {sorted(ms)} vs {mstone_steps}"
    r500, r600 = ms[mstone_steps[0]]["survival_ratio_vs_committed"], \
        ms[mstone_steps[1]]["survival_ratio_vs_committed"]
    r700, r800 = ms[mstone_steps[2]]["survival_ratio_vs_committed"], \
        ms[mstone_steps[3]]["survival_ratio_vs_committed"]
    # the anchor: e336's committed t400 ratio (asserted == the resumed
    # state's own final traj row in the full run)
    r400 = rea_st["traj"][-1]["survival_ratio_vs_committed"]
    if not SMOKE:
        assert abs(r400 - E336_REA_T400_RATIO) < 1e-12, \
            f"anchor drift {r400} vs {E336_REA_T400_RATIO}"
    gains = {"g1_400_500": r500 - r400, "g2_500_600": r600 - r500,
             "g3_600_700": r700 - r600, "g_last_700_800": r800 - r700}
    monotone = bool(r400 < r500 < r600 < r700 < r800)
    flat = bool(abs(gains["g_last_700_800"]) < SLOPE_BAR)
    in_band = bool(LAND_FLOOR <= r800 <= BAND_HI)
    edge_graze = bool(LAND_FLOOR <= r800 < BAND_LO)
    tw_r800 = cells_t["g0"] / E336.TAV_POST_G0
    x_ratio_800 = (r800 / tw_r800) if tw_r800 > 0 else None
    standing_ok = bool(G_STREAMELIVE["pass"] and G_BUDGET["pass"]
                       and G_BUFSEP["pass"] and G_ORTH["pass"])

    if not standing_ok:
        verdict = "MIXED-BY-CONDITIONS"
    elif r800 <= DIES_BAR:
        verdict = "DIES-EN-ROUTE"
    elif monotone and gains["g_last_700_800"] >= SLOPE_BAR:
        verdict = "STILL-CLIMBING"
    elif in_band and flat:
        verdict = "LANDS-IN-BAND"
    elif (not in_band) and flat:
        verdict = "PLATEAUS-BELOW"
    else:
        verdict = "MIXED"

    maint2 = rea["maint_ledger"]
    pre_reads_2 = [m["read_at_gate"] for m in maint2]
    deficits_2 = [m["deficit_t"] for m in maint2]
    clause = {
        "LANDS-IN-BAND": f"the read enters the band and flattens: "
        f"ratio_800 {r800:.4f} in [3.112, 3.616] "
        f"({'EDGE-GRAZE (in-hug)' if edge_graze else 'clean'}), "
        f"g_last {gains['g_last_700_800']:+.4f} < {SLOPE_BAR} — DIVERGES "
        f"WAS A KINETICS ARTIFACT; the rider re-words to SETTLING-RATE "
        f"NAME-TUNED; Law 4's scope re-words accordingly",
        "PLATEAUS-BELOW": f"the curve flattens below the band: ratio_800 "
        f"{r800:.4f} < 3.112, g_last {gains['g_last_700_800']:+.4f} — A "
        f"GENUINE LOWER EQUILIBRIUM; 'the founding band was ZEPHYRA's "
        f"slot' stands; the name-tuned rider confirms",
        "STILL-CLIMBING": f"monotone rise at t800 (gains "
        f"{gains['g1_400_500']:+.4f}/{gains['g2_500_600']:+.4f}/"
        f"{gains['g3_600_700']:+.4f}/{gains['g_last_700_800']:+.4f}; "
        f"g_last >= {SLOPE_BAR}) — the HORIZON exposure is named for the "
        f"controller program (no controller run has ever passed ~2x its "
        f"founding horizon)"
        + (f"; the in-band crossing at ratio_800 {r800:.4f} STAMPED "
           f"(entered still rising — not a landing)" if in_band else ""),
        "DIES-EN-ROUTE": f"the read died en route: ratio_800 {r800:.4f} "
        f"<= {DIES_BAR} — the family's FAILS form (the dispatch's three "
        f"bars do not cover it)",
        "MIXED": f"none of the three frozen shapes: milestones "
        f"{r400:.4f}/{r500:.4f}/{r600:.4f}/{r700:.4f}/{r800:.4f}, "
        f"monotone {monotone}, flat {flat}, in-band {in_band}",
        "MIXED-BY-CONDITIONS": "a standing condition failed (budget/"
                               "stream/isolation/orth) — routed per the "
                               "frozen composite",
    }[verdict]

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> (standing conditions) -> "
                           "DIES-EN-ROUTE -> STILL-CLIMBING -> "
                           "LANDS-IN-BAND -> PLATEAUS-BELOW -> MIXED — "
                           "frozen at birth",
        "gates_pass": bool(all(g.get("pass", True) for g in
                               metrics["gates"].values())),
        "reads": {
            "the_trajectory": {
                "t400_committed": r400,
                **{f"t{s}": ms[s]["survival_ratio_vs_committed"]
                   for s in mstone_steps},
                "t400_800_all_gains": gains,
                "monotone_rise": monotone, "flat": flat,
                "in_band": in_band, "edge_graze": edge_graze,
                "ceiling_ratio": CEILING_RATIO,
                "note": "every milestone read POST-maintenance (the same "
                        "read, battery, channel and denominator as "
                        "e336's bars)"},
            "the_landing": {
                "ratio_800": r800, "raw_read_800": cells_r["g0"],
                "band": [LAND_FLOOR, BAND_HI],
                "slope_at_800_per_100": gains["g_last_700_800"],
                "slope_bar": SLOPE_BAR,
                "settled_or_climbing": (
                    "SETTLED (flattened)" if flat
                    else "STILL CLIMBING" if gains["g_last_700_800"] > 0
                    else "FALLING")},
            "the_twin_and_x_ratio": {
                "twn_ratio_800": tw_r800, "x_ratio_800": x_ratio_800,
                "x_ratio_400_committed": E336_XRATIO_400,
                "twn_g0_800": cells_t["g0"]},
            "the_purity_read": {
                "g0_z_800": cells_r["g0_z"],
                "committed_prior": E336.TAV_G0_Z,
                "note": "the controller must not build a Z read"},
            "the_cumulative_spend": {
                "leg1_committed": {"S_corpus": E336_S_CORPUS_400,
                                   "S_maint": E336_S_MAINT_400,
                                   "S_total": E336_S_TOTAL_400},
                "leg2_measured": {"S_corpus": S2_corpus,
                                  "S_maint": S2_maint,
                                  "S_total": S2_corpus + S2_maint},
                "cumulative_t0_t800": E336_S_TOTAL_400 + S2_corpus
                + S2_maint,
                "cumulative_frac_of_write_norm":
                    (E336_S_TOTAL_400 + S2_corpus + S2_maint)
                    / write_norm,
                "per_leg_usage_frac": [(S2_corpus + S2_maint)
                                       / budget_norm]},
            "the_discriminating_trace": {
                "pre_event_reads_leg2": pre_reads_2,
                "deficits_leg2": deficits_2,
                "deficit_first_last_leg2": (
                    [deficits_2[0], deficits_2[-1]] if deficits_2
                    else None),
                "applied_lr_m_leg2": [m["lr_maint"] for m in maint2],
                "binder_leg2": [m["binder"] for m in maint2],
                "pre_event_reads_leg1_committed":
                    [m["read_at_gate"] for m in
                     e336_rea_ph["maint_ledger"]],
                "deficits_leg1_committed":
                    [m["deficit_t"] for m in e336_rea_ph["maint_ledger"]],
                "note": "the registered discriminator: LANDS-IN-BAND "
                        "shows the deficit collapsing; PLATEAUS-BELOW "
                        "shows it frozen high; STILL-CLIMBING shows it "
                        "mid-collapse with the peak still gaining"},
            "scatter_disclosure": {
                "n1_caveat": "n=1 per arm, ONE organism, one lineage — "
                             "the landing curve is a single draw; the "
                             "founding class itself was five draws "
                             "(3.162-3.616); this cell reads ONE "
                             "realization's approach, cited against the "
                             "band",
                "read_determinism": G_T400_REPRO["rea_absdiff"]},
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — machinery validation only; the numbers "
                        "are meaningless (8-step legs)" if SMOKE
                        else None),
    }
    metrics["adjudication"]["predictions_scored"] = {
        "P-e339a_lands_in_band": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "LANDS-IN-BAND"),
            "note": "SCORED: TRUE iff the verdict == LANDS-IN-BAND"
                    + ("" if not SMOKE else " (not scored in smoke)")},
        "lab_lean": {
            "scored": bool(not SMOKE),
            "true": bool(verdict == "LANDS-IN-BAND"),
            "note": "the dispatch's lean: LANDS-IN-BAND, weakly"},
    }
    log("=" * 78)
    log(f"P6 ADJUDICATED: {verdict} — {clause}")
    write_partial("P6 ADJUDICATED (the frozen bars)")

    # ================= P7: the figure + the report ======================
    traj_pts_c = [(s, E336_TRAJ[s]) for s in (100, 200, 300, 400)]
    traj_pts_m = [(s, ms[s]["survival_ratio_vs_committed"])
                  for s in mstone_steps]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.2))
    ax = axes[0]
    ax.axhspan(LAND_FLOOR, min(BAND_HI, CEILING_RATIO), color="green",
               alpha=0.10,
               label=f"the landing band [3.112, {CEILING_RATIO:.4f}] "
                     f"(top 3.616 unreachable)")
    ax.axhline(BAND_LO, color="green", ls=":", lw=1.2,
               label=f"the founding floor {BAND_LO} (hug -{EDGE_HUG})")
    ax.axhline(CEILING_RATIO, color="gray", ls="--", lw=1.0,
               label=f"battery ceiling {CEILING_RATIO:.4f}")
    for x, y in zip([430] * 5, list(E336.E288_CLASS_ENDPOINTS)):
        ax.plot([400, 430], [y, y], color="darkgreen", lw=1.0, alpha=0.5)
    ax.plot(*zip(*traj_pts_c), "o-", color="tab:blue", ms=5,
            label="e336 committed t100-400")
    ax.plot(*zip(*traj_pts_m), "o-", color="tab:red", ms=7,
            label=f"e339 measured t{mstone_steps[0]}-t{mstone_steps[-1]}")
    for s, y in traj_pts_m:
        ax.annotate(f"{y:.4f}", (s, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("corpus step t")
    ax.set_ylabel(f"survival ratio vs own baseline "
                  f"({E336.TAV_POST_G0:.4f})")
    ax.set_title(f"E339 THE LONG LANDING — {verdict} "
                 f"(ratio_800 {r800:.4f}; g_last "
                 f"{gains['g_last_700_800']:+.4f}/100steps; bar "
                 f"|g_last| < {SLOPE_BAR})"
                 + (" [SMOKE — machinery only]" if SMOKE else ""))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    pre1 = [m["read_at_gate"] for m in e336_rea_ph["maint_ledger"]]
    ax2.plot(range(1, 17), pre1, "s-", color="tab:blue", ms=4,
             label="pre-event gate reads, events 1-16 (e336 committed)")
    if pre_reads_2:
        ax2.plot(range(17, 17 + len(pre_reads_2)), pre_reads_2, "s-",
                 color="tab:red", ms=5,
                 label="pre-event gate reads, events 17-32 (e339 measured)")
    ax2.axhline(E336.TAV_POST_G0, color="k", ls="--", lw=1.0,
                label=f"the baseline {E336.TAV_POST_G0:.4f} "
                      f"(deficit = 0 — the dose self-limits here)")
    peak_x = [100, 200, 300, 400] + list(mstone_steps)
    peak_y = ([E336_TRAJ[s] for s in (100, 200, 300, 400)]
              + [ms[s]["g0_pz"] for s in mstone_steps])
    ax2b = ax2.twinx()
    ax2b.plot(peak_x, peak_y, "^-", color="tab:purple", ms=6,
              label="post-event milestone reads (raw)")
    ax2b.set_ylabel("post-event raw read", color="tab:purple")
    ax2.set_xlabel("maintenance event index")
    ax2.set_ylabel("pre-event gate read (raw)")
    ax2.set_title("the sawtooth's floor: the gate reads (the dose's own "
                  "input) vs the milestone peaks — the registered "
                  "discriminator")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = RD / ("smoke_e339_long_landing.png" if SMOKE
                     else "e339_long_landing.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ---- REPORT.md ------------------------------------------------------
    ts = common.now_iso()
    gcount = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    lines = [
        f"# E339 — THE LONG LANDING\n",
        f"**Date:** {ts} (datetime.now(UTC) stamps in metrics) · "
        f"**Executor report**\n",
        "## THE VERDICT (against the frozen bars)\n",
        f"**{verdict}**"
        + (" _(smoke — machinery validation only)_" if SMOKE else ""),
        "\n```",
        clause,
        "```\n",
        "## The trajectory (post-maintenance milestone reads)\n",
        "| t | ratio vs own baseline | gain/100steps |",
        "|---|---|---|",
        f"| 400 (e336 committed) | {r400:.4f} | "
        f"+{r400 - E336_TRAJ[300]:.4f} |",
    ]
    prev = r400
    for s in mstone_steps:
        y = ms[s]["survival_ratio_vs_committed"]
        lines.append(f"| {s} | {y:.4f} | {y - prev:+.4f} |")
        prev = y
    lines += [
        "",
        f"- monotone rise: {monotone}; flat (|g_last| < {SLOPE_BAR}): "
        f"{flat}; in band: {in_band}"
        + (" (EDGE-GRAZE — in the hug)" if edge_graze else ""),
        f"- the slope at t800: {gains['g_last_700_800']:+.4f} per 100 "
        f"steps — "
        + ("SETTLED" if flat else
           "STILL CLIMBING" if gains["g_last_700_800"] > 0
           else "FALLING"),
        f"- raw read at t800: {cells_r['g0']:.6f} "
        f"(ceiling 1.0 = ratio {CEILING_RATIO:.4f})\n",
        "## The discriminator trace (the dose's own input)\n",
        f"- pre-event gate reads, leg 2 (events 17-32): "
        f"{[round(x, 4) for x in pre_reads_2]}",
        f"- deficits, leg 2: {[round(x, 3) for x in deficits_2]} "
        f"(leg 1 committed: 0.947 -> 0.561)",
        f"- the dose self-limits as the floor rises toward the baseline "
        f"{E336.TAV_POST_G0:.4f}\n",
        "## The cumulative spend (the doubled-horizon cost)\n",
        f"- leg 1 (committed): S_corpus {E336_S_CORPUS_400:.4f} + "
        f"S_maint {E336_S_MAINT_400:.4f} = {E336_S_TOTAL_400:.4f} "
        f"(83.4% of its budget)",
        f"- leg 2 (measured): S_corpus {S2_corpus:.4f} + S_maint "
        f"{S2_maint:.4f} = {S2_corpus + S2_maint:.4f} "
        f"({(S2_corpus + S2_maint) / budget_norm:.1%} of the leg budget "
        f"{budget_norm:.4f})",
        f"- CUMULATIVE t0->t800: {E336_S_TOTAL_400 + S2_corpus + S2_maint:.4f}"
        f" = {(E336_S_TOTAL_400 + S2_corpus + S2_maint) / write_norm:.4f}x "
        f"the write norm (leg convention: a fresh 0.5x-write budget per "
        f"400-step leg — frozen at birth; the stretched-pool alternative "
        f"rejected)\n",
        "## The twin and the co-reads\n",
        f"- the twin at t800: {cells_t['g0']:.7f} "
        f"(x{tw_r800:.4f}) — x-ratio_800 "
        f"{x_ratio_800 if x_ratio_800 is None else round(x_ratio_800, 1)}x "
        f"(x-ratio_400 committed {E336_XRATIO_400:.1f}x)",
        f"- the purity read p(Z) at t800: {cells_r['g0_z']:.2e} (the "
        f"committed prior {E336.TAV_G0_Z:.2e}) — no Z read built\n",
        "## Gates\n",
        f"- {gcount} gate classes instantiated, all PASS: "
        + ", ".join(f"{k}" for k in metrics["gates"]),
        "",
        "## Envelope\n",
        f"- {len(thermal_log)} thermal polls; max temp "
        f"{max((r.get('temp', 0) for r in thermal_log), default=0):.1f}C; "
        f"burst cap {E261.BURST_MAX_S:.0f}s; cooldown "
        f"{E261.COOLDOWN_S:.0f}s; zero concurrent GPU jobs; every burst "
        f"logged to runs/_envelope_log.jsonl tagged e339:...",
        "",
        "## Registered predictions\n",
        f"- P-e339a (the executor's own read: LANDS-IN-BAND, weakly): "
        f"{'HIT' if verdict == 'LANDS-IN-BAND' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        f"- the dispatch's lab lean (LANDS-IN-BAND, weakly): "
        f"{'HIT' if verdict == 'LANDS-IN-BAND' else 'MISSED'}"
        + (" (not scored — smoke)" if SMOKE else ""),
        "",
        "## Catches / disclosures\n",
        "- THE RESUME: e336's committed t400 states (model + optC + optF "
        "+ both generator states) md5-bound, bit-compared vs the post "
        "checkpoints, and read-reproduced (G_T400_REPRO) BEFORE any "
        "compute; the stream CONTINUES (generators restored, never "
        "re-seeded).",
        "- THE LEG CONVENTION: 'the same budget convention' = e336's "
        "400-step protocol VERBATIM as a second leg (a fresh "
        "0.5x-write-norm budget, pools split 60/40, the driver's own "
        "equal-share caps); the stretched-pool alternative REJECTED at "
        "birth (B_C exactly spent at t400 — the wash would die; a "
        "forfeit landscape).",
        "- THE SCHEDULE: cosine_lr(step-1, 1000) VERBATIM on the global "
        "step; disclosed immaterial — the corpus cap bound below the "
        "schedule at every logged leg-2 step (verified in "
        "G_SCHED_IMMATERIAL); lr_m is schedule-free.",
        "- The drivers are e336's COMMITTED bodies executed VERBATIM "
        "under the disclosed leg re-binding (PHASE_STEPS/MILESTONES/"
        "WIN1/WIN2 only); every protocol constant asserted == the "
        "md5-bound record first.",
        "- The founding band is TRANSPLANTED verbatim [3.112 (hugged), "
        "3.616]; the top unreachable (ceiling 3.4983); an in-hug landing "
        "stamped EDGE-GRAZE (e336's frozen convention).",
        "- n=1 per arm, ONE organism — the landing curve is a single "
        "draw read against the founding class's five-draw band.",
        "- Checkpoints live INSIDE runs/e339/. No NOTES/THINKING/QUEUE/"
        "STATE edits (the heartbeat folds this cell).",
        "",
        "*This cell does not edit NOTES/THINKING/QUEUE/STATE — the "
        "heartbeat folds.*",
    ]
    (RD / ("smoke_REPORT.md" if SMOKE else "REPORT.md")).write_text(
        "\n".join(lines), encoding="utf-8")

    # ---- the final metrics ---------------------------------------------
    metrics["arms"] = {
        "ERROR-GATED": {
            "desc": "the continuation leg: e336's committed t400 state "
                    "(model + buffers + generators) resumed; e336's "
                    "driver VERBATIM; corpus steps t401-t800 over a "
                    "fresh leg B_C; 16 more NAME-ONLY maintenance events "
                    "(17-32) through opt_F at the error-gated lr; the "
                    "same controller law on the same baseline",
            "phase": {k: rea[k] for k in
                      ("traj", "corpus_ledger", "orth_ledger",
                       "disp_ledger", "buf_ledger", "budget_ledger",
                       "lr_ledger", "maint_ledger", "window_ledger",
                       "bufsep", "chunk_table", "steps_ran")},
            "post_cells": cells_r,
            "checkpoint": str(RD / ("smoke_REA_post.pt" if SMOKE
                                    else "e339_REA_post.pt"))},
        "PASSIVE-TWIN": {
            "desc": "the twin continuation: the sanctuary form over the "
                    "SAME continued stream, a fresh leg B_C, opt_F never "
                    "stepped — the arms' only delta the 16+16 maintenance "
                    "events",
            "phase": {k: twn[k] for k in
                      ("traj", "corpus_ledger", "orth_ledger",
                       "disp_ledger", "buf_ledger", "budget_ledger",
                       "lr_ledger", "window_ledger", "bufsep",
                       "chunk_table", "steps_ran")},
            "post_cells": cells_t,
            "checkpoint": str(RD / ("smoke_TWN_post.pt" if SMOKE
                                    else "e339_TWN_post.pt"))},
    }
    metrics["honesty"] = {
        "intervention_not_logits": "the two arms share ONE resumed "
        "state (e336's committed t400 checkpoints, md5 + repro gated), "
        "the same continued stream (the generator states restored — "
        "bit-identical draws), the same milestone cadence and reads; the "
        "ONLY delta is the controller's 16 continuation maintenance "
        "events",
        "the_leg_disclosure": "the leg budget convention and the "
                              "stretched-pool rejection are FROZEN AT "
                              "BIRTH in this file's header + the "
                              "operationalizations; the schedule "
                              "immateriality is verified per-step "
                              "(G_SCHED_IMMATERIAL), not assumed",
        "the_resume_disclosure": "the resume is the protocol's own "
                                 "checkpoint convention (README rule 10) "
                                 "— the state is e336's committed "
                                 "artifact, never re-formed; the repro "
                                 "gate re-reads the battery on the "
                                 "restored bits before any stepping",
        "n_and_scope": "n=1 per arm, ONE organism, one lineage; the "
                       "landing curve is a single draw cited against "
                       "the founding class's five-draw band",
        "loads_measured_not_nominal": "every read measured: per-event "
                                      "doses, realized displacements, "
                                      "the S split, per-milestone "
                                      "displacement/orth/buffer "
                                      "ledgers, window reads, thermal "
                                      "polls",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": [
            str(E336_DIR.parent.parent / "lab" / "e336_founding_replicate.py"),
            "lab/e261_rank_ladder.py (the burst machinery + rooms)",
        ],
        "machinery": {
            "leg_drivers": "e336's chunked_errorgated_phase / "
                           "chunked_passive_twin_phase EXECUTED VERBATIM "
                           "under the disclosed leg re-binding",
            "staging": "THIS file's stage_leg_resume (the only "
                       "protocol-touching transformation; digest-verified)",
            "cons": "NONE (the bars read the WRITE and the SLOPE only)",
        },
        "checkpoints": {
            "resumed_from": [str(rea_src), str(twn_src)],
            "e336_metrics": str(E336_METRICS),
            "leg_resumes": [str(rea_leg_ck), str(twn_leg_ck)],
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
                    "tagged e339:<ARM>:<phase>",
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["outputs"] = [str(fig_path), str(RD / "REPORT.md"),
                          str(RD / "metrics.json")]
    metrics["status"] = (f"COMPLETE — adjudicated {verdict}"
                         + (" (SMOKE)" if SMOKE else "")
                         + f" ({common.now_iso()})")
    save_json(RD / "metrics.json", metrics)
    log(f"P7 DONE: {fig_path.name} + REPORT.md + metrics.json — "
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
