"""E268 — THE ROOM-OPTIMIZER INTERFACE (T244's registered discriminator;
the day's capstone cell: the cliff's carrier test). Design dispatched
2026-10-05 (commit f6746b5); this docstring carries the registered bars
VERBATIM, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): is the expression cliff's
carrier the DYNAMICAL INTERFERENCE between the writing room's directions
and the CORPUS TRAJECTORY during the install (a trajectory object no
eigenstructure sees) — or is it a static property of the room (which the
eigen-campaign already failed to find, so if this cell also nulls, the
deep-spectrum/Lanczos branch inherits)?

THE CELL (2.74M, the g1c fresh-root conventions; the ladder's own
dose/N=400 install at k=10k — the THRESHOLD rung — as the vehicle):
  1. ARM-CONCURRENT: the k=10k random-room install run with the CORPUS
     stream running CONCURRENTLY (the install's interleaved-with-corpus
     schedule — alternate the install batch with a corpus batch at the
     standard ratio; the corpus batches from the committed g1c
     conventions; state the interleave exactly);
  2. ARM-SERIAL (the control): the identical k=10k install run ALONE
     (the ladder's committed condition — e264's committed rung cited as
     the anchor AND re-run fresh at the same seeds: one instrument, one
     session);
  3. the readouts: post g0 (expression) + root g0 (landing) + kept, per
     arm.

REGISTERED BARS (frozen in the dispatch letter, VERBATIM in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping):
  - DYNAMICAL-CARRIER — "the concurrent arm's expression or landing
    differs from the serial arm's beyond the cons/session scatter (post
    g0 ratio >= 2x or the root outside +-10% of the serial's) — the
    cliff's carrier is the room-trajecatory INTERFERENCE: the barrier is
    written in the dynamics, completing the static/dynamical split's
    mechanism story"
  - STATIC-OR-DEEPER — "the arms agree within the disclosed scatter —
    the interface is not the carrier at this dose; the deep-spectrum/
    Lanczos branch inherits (the honest bound; the day's carrier
    question stays open with exactly one unmeasured branch left)"
  - MIXED — "the trajectories verbatim, no inflation"

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE VEHICLE := the committed THRESHOLD rung VERBATIM: the K10K room
    (k=10,000; seeds 26113/26114 — rebuilt from its registered seeds and
    bit-gated against e264_rooms.pt's stored K10K D/S (G_ROOMK10K)), the
    e001 fresh fact-free base, the Dmix install s400 gen 24314 (E43's
    arithmetic: batch = 16 masked install windows + 48 corpus windows
    (16 original-host anchors + 32 random); AdamW (0.9,0.95) wd 0.1; lr
    1e-3 x the house cosine(total=1000); clip 1.0), the hook VERBATIM
    (backward -> clip 1.0 -> project onto the room (CPU fp64, write
    fp32) -> opt.step; norm NOT rescaled), the cons e113-VERBATIM s300
    seed 10901 HELD NATURAL on both arms.
  * ARM-SERIAL := that vehicle run ALONE — e261's chunked_install driver
    VERBATIM BY IMPORT at the registered seeds, re-run FRESH on this
    session (the primary comparison object), anchored to e264's
    committed K10K rung (post g0 0.26464763 / root g0 0.77078849 / kept
    0.06004516; install L2 vs the committed vehicle e261_K10K_inst_
    resume.pt) by the NON-HALTING texture gate G_SERIAL_ANCHOR (a
    failure is disclosed session texture, never a bar move — e261's
    G_ANCHOR precedent).
  * THE INTERLEAVE (registered exactly; the corpus batches' provenance):
    after EVERY install step s (s = 1..400), ONE corpus step — the 1:1
    alternation (e174's interleaved-replay convention) — where a corpus
    step := a 48-window batch from the COMMITTED g1c Dmix corpus
    convention: 16 windows drawn from the same 60-window original-host
    anchor bank (anchor_full) + 32 random corpus windows (contiguous
    train_ids slices), full-window CE (no mask), drawn from the
    REGISTERED corpus-stream generator (seed 26801 — its own generator,
    so the install stream's draws stay bit-identical to SERIAL's by
    construction), at the PAIRED install step's lr (cosine_lr(s-1,1000)
    — no new schedule object), backward -> clip 1.0 -> opt.step FREE (NO
    projection: the corpus stream is the natural stream; the room
    constrains the install's WRITING directions only) — ONE shared
    AdamW across both streams (the moment coupling IS the interference
    channel). 800 optimizer steps total vs SERIAL's 400; the install
    dose (16 masked windows x 400) is IDENTICAL across arms; the corpus
    exposure triples (in-batch 48 + interleaved 48 per iteration vs
    SERIAL's in-batch 48) — that tripling IS the concurrent stream, the
    object under test, disclosed as the arms' total-dose delta.
  * "post g0 ratio >= 2x" := ratio = max(post_c, post_s) /
    max(min(post_c, post_s), 1e-6) >= 2 (the ladder's floor-guard
    convention; symmetric — "differs" is directional-agnostic);
  * "the root outside +-10% of the serial's" := concurrent root g0
    outside [0.9 x SERIAL's root g0, 1.1 x SERIAL's root g0];
  * "beyond the cons/session scatter" := the bars are pre-sized to
    exceed it and the scatter is DISCLOSED on every read: install
    determinism |d post g0| ~ 5e-7 cross-session (e261's G_ANCHOR) <<
    the 2x bar; the cons lottery's cross-session root-g0 scatter 0.0419
    (e261's G_ANCHOR, same bit-identical arm) and session-scale cons-
    seed scatter 0.0137 (e264's replicate) < the +-10% band (0.077 at
    root 0.77); THIS session's serial-vs-committed anchor delta is
    co-reported as the freshest scatter point; root g-12 is NEVER
    adjudicated (the wild lottery — e264's three-draw range 0.67 on a
    bit-identical arm; the g0 ruler is the stable one);
  * DYNAMICAL-CARRIER fires iff the hard gates pass AND (ratio >= 2 OR
    the root is outside the +-10% window); STATIC-OR-DEEPER fires iff
    the hard gates pass AND (ratio < 2 AND the root is inside); MIXED
    otherwise (named branch: a headline read exactly AT a bar; the
    headline agrees but the install trajectories differ qualitatively
    inside the bars; or G_SERIAL_ANCHOR texture failure contaminating a
    would-be firing cross-session read) — "the trajectories verbatim, no
    inflation"; composite order TEXTURE (hard-gate failure; nothing
    adjudicated) -> DYNAMICAL-CARRIER -> STATIC-OR-DEEPER -> MIXED.
  * HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR(bank),
    G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ,
    G_ROOMK10K} — a failure HALTS (nothing adjudicated). G_SERIAL_ANCHOR
    is NON-halting texture. NO fresh FREE arm runs (disclosed: e264's
    G_FREE PASS on this session-generation — install L2 6.2e-4 — is the
    instrument validation; the serial anchor is the live vehicle
    control).
  * MEASURED, NEVER NOMINAL: per-arm kept-fraction ledger (install
    steps), v-excess pre/post (vs e258's LOADED committed v-map), in-span
    pre AND applied (vs e246's committed LATE span), displacement
    cos-to-span + in-own-room at post-install and root, the corpus
    stream's own ledger (CE + clipped-grad norm), and the install
    trajectories (g0/g-12/CE_R at s = 1,100,200,300,400 — comparable
    across arms by construction).

CHECKS (the dispatch's, in force): the ladder machinery smoke FIRST
(e260's record: 2-3 bugs caught per build); the room certified AND
bit-bound to the committed K10K room; the interleave schedule registered
exactly (above); n=1 per arm (the lottery note); the cons-scatter
disclosure carried (the g0 ruler stable, the g-12 wild — use g0);
nothing guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); the
owner's max-priority window — bursts <= 175 s, per-step thermal polls at
a 78 C margin, 40 s cooldowns (the 30-60 s window), the 84 C never-past
line recorded; CPU fp64 dense projections (pocketfft workers 2); CPU
probing threads 4.

Outputs: runs/e268/{metrics.json (PROGRESSIVE),
e268_room_interface.png, e268_instrument.png, REPORT.md,
DRAFT_NOTES_ENTRY.md, run.log (gitignored)}; checkpoints
runs/checkpoints/e268_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e268_room_interface.py    (E268_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E268_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e268_smoke" if SMOKE else "e268"
assert torch.cuda.is_available(), "e268 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e264's disclosed convention, in force before ANY
# machinery call): e261's ported drivers resolve their module globals (log /
# NAME / LADDER / RUNG_NAMES / INST_STEPS / CONS_STEPS / SMOKE) AT CALL TIME
# through e261's module namespace — rebound HERE so they write THIS cell's
# log, label THIS cell's envelope polls, and run THIS cell's single-rung
# ladder. The committed lab/e261_rank_ladder.py itself is untouched.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
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

# ---- THE VEHICLE: the committed THRESHOLD rung (k=10k), e264's record
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — e261's registered seed pair
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (512, 26113, 26114),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG = {k: ("K10K" if not SMOKE else f"K{k}") for k, _, _ in LADDER}
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG
ARMS = ("SERIAL", "CONCURRENT")

# the corpus stream's OWN generator (registered; its own draws so the
# install stream stays bit-identical to SERIAL's by construction)
CORPUS_GEN_SEED = 26801

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
E264_VERDICT = "SHARP-THRESHOLD"
E264_K10K_POST_G0 = 0.26464763283729553
E264_K10K_ROOT_G0 = 0.7707884907722473
E264_K10K_KEPT_MED = 0.060045162390265784
E264_K10K_POST_GM12 = 0.10525520890951157

# e261's committed record (the stitch law's source; hard-bound)
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "f460475d8e6b76f0719e91c1e9c6041b"     # e264's bind, carried
ANCHOR_SCATTER_G0 = 0.04194730520248413           # e261's G_ANCHOR, same arm
CONS_SEED_SCATTER_G0 = 0.0137                     # e264's cons-seed replicate

# the committed K10K install vehicle (e261's cut, completed by e264 pass 1)
K10K_CK = "e261_K10K_inst_resume.pt"
K10K_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
K10K_SIZE = 32958479
K10K_STEP = 400
K10K_TRAJ_STEPS = [1, 100, 200, 300, 400]
K10K_LEDGER_MAX = 400

# the discriminator's frozen numbers
POST_RATIO_BAR = 2.0              # "post g0 ratio >= 2x"
ROOT_BAND_FRAC = 0.10             # "+-10% of the serial's"
RATIO_DEN_FLOOR = 1e-6            # the ladder's floor-guard convention
G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 — the expression floor (ctx)
MATCH_BAND = E261.MATCH_BAND              # the landing band (context)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
ANCHOR_BEHAV_TOL = 0.02           # the family's session-texture bar

ARM_DESC = {
    "SERIAL": "the committed K10K threshold rung run ALONE (the ladder's "
              "committed condition): e261's chunked_install driver VERBATIM "
              "at the registered seeds, re-run FRESH on this session — "
              "the primary comparison object, anchored to e264's committed "
              "rung (G_SERIAL_ANCHOR, non-halting texture)",
    "CONCURRENT": "the identical install stream (bit-identical draws, "
                  "batch, lr per step; the SAME room projection on every "
                  "install gradient) + the CORPUS stream running "
                  "CONCURRENTLY: after EVERY install step, ONE corpus "
                  "step (48 windows = 16 original-host anchors + 32 "
                  "random corpus windows, the g1c Dmix corpus convention; "
                  "corpus generator seed 26801; the paired step's lr; "
                  "backward -> clip 1.0 -> opt.step FREE) through the ONE "
                  "shared AdamW — the interleaved-with-corpus schedule, "
                  "1:1",
}

REGISTERED = {
    "question_verbatim": "is the expression cliff's carrier the DYNAMICAL "
        "INTERFERENCE between the writing room's directions and the CORPUS "
        "TRAJECTORY during the install (a trajectory object no "
        "eigenstructure sees) — or is it a static property of the room "
        "(which the eigen-campaign already failed to find, so if this cell "
        "also nulls, the deep-spectrum/Lanczos branch inherits)?",
    "bars_verbatim": {
        "DYNAMICAL-CARRIER": "the concurrent arm's expression or landing "
            "differs from the serial arm's beyond the cons/session scatter "
            "(post g0 ratio >= 2x or the root outside +-10% of the "
            "serial's) — the cliff's carrier is the room-trajecatory "
            "INTERFERENCE: the barrier is written in the dynamics, "
            "completing the static/dynamical split's mechanism story",
        "STATIC-OR-DEEPER": "the arms agree within the disclosed scatter — "
            "the interface is not the carrier at this dose; the "
            "deep-spectrum/Lanczos branch inherits (the honest bound; the "
            "day's carrier question stays open with exactly one "
            "unmeasured branch left)",
        "MIXED": "the trajectories verbatim, no inflation",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE VEHICLE := the committed THRESHOLD "
        "rung VERBATIM (K10K room seeds 26113/26114 rebuilt + bit-gated vs "
        "e264_rooms.pt; e001 base; Dmix s400 gen 24314; hook = clip 1.0 -> "
        "project CPU fp64 -> opt.step, norm NOT rescaled; e113 cons s300 "
        "seed 10901 HELD on both arms); ARM-SERIAL := the vehicle run "
        "ALONE (e261's driver VERBATIM, re-run fresh; anchored to e264's "
        f"committed rung post {E264_K10K_POST_G0:.8f} / root "
        f"{E264_K10K_ROOT_G0:.8f} / kept {E264_K10K_KEPT_MED:.8f} by the "
        "NON-HALTING G_SERIAL_ANCHOR); THE INTERLEAVE := after EVERY "
        "install step s, ONE corpus step (1:1, e174's convention): 48 "
        "windows = 16 anchors from the same 60-window bank + 32 random "
        "corpus windows (the g1c Dmix corpus convention; corpus generator "
        f"seed {CORPUS_GEN_SEED}; the paired step's lr; clip 1.0 -> "
        "opt.step FREE — no projection: the room constrains the install's "
        "writing directions only) through the ONE shared AdamW; 800 opt "
        "steps vs 400; install dose IDENTICAL, corpus exposure tripled "
        "(the concurrent stream IS the object under test); 'post g0 "
        "ratio >= 2x' := max/min with the 1e-6 floor guard, symmetric; "
        "'root outside +-10%' := concurrent root g0 outside [0.9, 1.1] x "
        "SERIAL's; scatter disclosed on every read (install |d post g0| "
        "~5e-7; cons cross-session root g0 0.0419; session cons-seed "
        "0.0137; root g-12 NEVER adjudicated — the wild lottery); "
        "DYNAMICAL-CARRIER iff hard gates pass AND (ratio >= 2 OR root "
        "outside); STATIC-OR-DEEPER iff hard gates pass AND (ratio < 2 "
        "AND root inside); MIXED otherwise (named branch); composite "
        "TEXTURE -> DYNAMICAL-CARRIER -> STATIC-OR-DEEPER -> MIXED; HARD "
        "GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, "
        "G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, "
        "G_ROOMK10K} (a failure HALTS); no fresh FREE arm (e264's G_FREE "
        "PASS on this session-generation is the instrument validation; "
        "disclosed)."),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (T244's registered discriminator; the day's capstone cell); "
        "this script committed at birth BEFORE any compute; adjudicate "
        "against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "NO FRESH FREE ARM (disclosed at birth): e264's G_FREE PASS on this "
    "session-generation (install L2 6.2e-4 vs the committed g1c install, "
    "behavioral |d| 1.3e-5, root in band) validates the instrumented path "
    "on this machinery; THIS cell's live vehicle control is G_SERIAL_ANCHOR "
    "(the serial re-run vs e264's committed rung). No ceiling arm runs; no "
    "bar or gate-form change follows from the skip.",
    "THE SERIAL ARM IS A FRESH RE-RUN, NOT A CITE (disclosed): the "
    "dispatch allows citing e264's committed rung; a same-session re-run "
    "at the registered seeds makes the discriminator one instrument on "
    "one session (e261's anchor philosophy), and its delta vs the "
    "committed rung doubles as the freshest cross-session scatter point "
    "(co-reported; the family's inherited law: cross-pass cuBLAS/atomics "
    "noise, install L2 ~ 6.7e-5 on a bit-identical arm).",
    "THE CORPUS STEP IS UNPROJECTED (the registered choice, disclosed): "
    "the room constrains the INSTALL's writing directions (the vehicle "
    "verbatim); the concurrent corpus stream runs FREE — the natural "
    "stream, as in any wash/cons — so its trajectory can drag the state "
    "through and off the room between install writes. Projecting it would "
    "remove exactly the interface under test.",
    "THE TOTAL-DOSE DELTA IS THE OBJECT (disclosed): the concurrent arm's "
    "install dose is bit-identical to SERIAL's, but its corpus exposure "
    "triples (in-batch 48 + interleaved 48 per iteration vs SERIAL's "
    "in-batch 48). The tripling is not a confound to remove — the "
    "concurrent corpus stream IS the intervention; the corpus ledger (CE "
    "+ clipped-grad norm per step) and the trajectory reads make its "
    "footprint visible, never nominal.",
    "e261's machinery PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked chunked install/cons drivers (bit-identical arithmetic + draw "
    "order), the thermal envelope (per-step polls, 78C margin, 175s "
    "bursts, 40s cooldowns, the 84C line), the progressive-metrics + "
    "resume-ckpt conventions — the module-global rebinding (log/NAME/"
    "LADDER/RUNG_NAMES, disclosed in-code) retargets the drivers' I/O to "
    "this cell; the committed lab/e261_rank_ladder.py is NOT modified. "
    "The ONE new driver is chunked_install_concurrent (this file).",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger; no new "
    "history is run. The installs' own fresh Adam state remains the "
    "mechanism's channel (the arms' shared-optimizer moments are the "
    "interference channel under test).",
    "NO WASH IN THIS CELL (the ladder's inherited registration): the "
    "readouts are post g0 + root g0 + kept per arm; the retention/"
    "occupancy question stays T237's separately registered follow-up.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E268_SMOKE=1): 8 install + 8 interleaved corpus steps, "
    "8-step cons, room k=512 at the same seed pair (G_ROOMK10K vacuous — "
    "no committed record at smoke k; disclosed), all paths smoke_-"
    "prefixed, own smoke dir; NOTHING adjudicated or gated (SMOKE stamp "
    "on every read).",
]

device_events: list[dict] = []
thermal_log: list[dict] = []


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


# -------------------------------------------------- THE CONCURRENT DRIVER
def chunked_install_concurrent(tag, net0, proj: "E261.LadderRooms",
                               inst_x, inst_mask, anchor_full, train_ids,
                               g0_ids, gm12_ids, r_eval_xy, zid,
                               resume_ck: Path, dev: torch.device) -> dict:
    """THE CONCURRENT ARM'S DRIVER (this cell's only new machinery; the
    SERIAL arm runs e261.chunked_install VERBATIM).

    Per iteration s = 1..400 (the install step index — BOTH steps carry
    the paired lr; the install step's draws/batch/hook are BIT-IDENTICAL
    to SERIAL's step s by construction):
      1. INSTALL STEP: e261's chunked_install arithmetic VERBATIM —
         draws (ix(16), aj(16), rj(32)) from gen (seed 24314, the same
         draw order); batch = 16 masked install windows + 48 corpus
         windows (16 anchors + 32 random); masked union CE; lr = 1e-3 x
         cosine_lr(s-1, 1000); backward -> clip 1.0 -> HOOK: project onto
         the K10K room (CPU fp64, write fp32) + fp64 ledger dots ->
         opt.step.
      2. CORPUS STEP (the interleave, 1:1): draws (aj_c(16), rj_c(32))
         from cgen (seed 26801); batch = the SAME 48-window corpus
         composition as the install's corpus part (the g1c Dmix corpus
         convention); full-window CE; the PAIRED install step's lr;
         backward -> clip 1.0 -> opt.step FREE (NO projection; corpus
         ledger: CE + clipped-grad norm on device).

    ONE shared AdamW across both streams (the moment coupling IS the
    interference channel). Thermal: a poll after EVERY opt step (both
    streams — the per-step discipline). Trajectory reads at install-step
    milestones (s = 1, 100, 200, 300, 400 — matching SERIAL's traj)."""
    name_bs, corp_bs, mix_random, lr = (G1.NAME_BS, E43.CORP_BS,
                                        E43.MIX_RANDOM, E43.LR)
    n_steps = E261.INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    state = {"step": 0, "traj": [], "ledger": {}, "corpus_ledger": {}}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at install step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net, opt, gen, cgen, evl = None, None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
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
            f = common.cosine_lr(step - 1, E261.INST_TOTAL)
            for g in opt.param_groups:
                g["lr"] = lr * f
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            led = proj.step_hook(list(net.parameters()), "K10K")
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
            opt.zero_grad(set_to_none=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gn_c = float(torch.cat([p.grad.detach().reshape(-1)
                                    for p in net.parameters()
                                    if p.grad is not None]).norm().item())
            opt.step()                          # FREE — no projection
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c}
            n_burst += 1
            ok_t2, temp2 = burst_temp_check(f"{tag}-c{n_chunks}.x")
            chunk_temps.append(temp2)
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
                                      "ce_corpus": float(loss_c.item()),
                                      "kept_frac": led["kept_frac"],
                                      "v_excess_post": led["v_excess_post"],
                                      "in_span_frac": led["in_span_frac"],
                                      "applied_in_span_frac":
                                          led["applied_in_span_frac"],
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.4f} g-12 "
                    f"{bz12['mean_pz']:.4f} CE_R {ce_r:.4f} CE_inst "
                    f"{float(loss.item()):.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |g| {led['gn']:.3f} kept "
                    f"{led['kept_frac']:.4f} |g_corp| {gn_c:.3f}")
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


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e268_room_interface",
        "phase": "THE ROOM-OPTIMIZER INTERFACE — the cliff's carrier test "
                 "at the threshold rung: the identical k=10k random-room "
                 "install with the corpus stream running CONCURRENTLY "
                 "(1:1 interleave, free corpus steps, one shared AdamW) vs "
                 "ALONE (the ladder's committed condition, re-run fresh) — "
                 "DYNAMICAL-CARRIER vs STATIC-OR-DEEPER vs MIXED",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (the owner's max-priority window; "
                      "this cell owns the GPU lane) + CPU fp64 dense "
                      "projections (pocketfft workers 2), CPU probing "
                      "threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s, per-step thermal polls "
                      f"(BOTH streams' opt steps) at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line recorded",
            "trainings": "2 installs s400 (SERIAL = e261's driver VERBATIM; "
                         "CONCURRENT = the interleaved driver, 800 opt "
                         "steps) + 2 cons s300 (seed 10901 HELD); NO "
                         "washes; NO fresh FREE (e264's G_FREE PASS cited)",
        },
        "interleave": {
            "ratio": "1:1 — after EVERY install step s, ONE corpus step "
                     "(e174's interleaved convention)",
            "install_step": "bit-identical to SERIAL's step s: draws "
                            "ix(16)/aj(16)/rj(32) from gen seed 24314 (the "
                            "same draw order), the 64-window Dmix batch, "
                            "the masked union CE, lr 1e-3 x "
                            f"cosine_lr(s-1, {E261.INST_TOTAL}), clip 1.0 "
                            "-> PROJECT onto the K10K room -> opt.step",
            "corpus_step": "48 windows from the committed g1c Dmix corpus "
                           "convention: 16 original-host anchors (the same "
                           "60-window anchor bank) + 32 random corpus "
                           "windows (contiguous train_ids slices); "
                           "full-window CE; draws from the REGISTERED "
                           f"corpus generator seed {CORPUS_GEN_SEED}; the "
                           "PAIRED install step's lr; clip 1.0 -> "
                           "opt.step FREE (unprojected — the natural "
                           "stream)",
            "shared_optimizer": "ONE AdamW (0.9, 0.95) wd 0.1 across both "
                                "streams — the moment coupling IS the "
                                "interference channel",
            "dose_delta_disclosed": "install dose IDENTICAL across arms "
                                    "(16 masked windows x 400); corpus "
                                    "exposure triples in CONCURRENT "
                                    "(in-batch 48 + interleaved 48 per "
                                    "iteration vs SERIAL's in-batch 48) — "
                                    "the concurrent stream is the object "
                                    "under test",
        },
        "deviations": deviations,
        "builds_on": [
            "T244 / e267 (THE registration: the Jacobian campaign's close "
            "found no spectral object kneeling at the cliff on either "
            "stream — the room-optimizer interface is the surviving "
            "non-spectral carrier form; this cell is its registered "
            "discriminator)",
            "T242 / e264 (the LOCATED cliff ~10k: the 6%-dose threshold "
            "rung THIS cell uses as its vehicle — committed post g0 "
            f"{E264_K10K_POST_G0:.6f}, root {E264_K10K_ROOT_G0:.6f}, kept "
            f"{E264_K10K_KEPT_MED:.4f}; the cons-scatter law (g0 stable / "
            "g-12 wild) and the G_FREE PASS cited from its record)",
            "T245 / e263 (the counterfeit inert — the static/dynamical "
            "split complete: the barrier, the killer and the destination "
            "all refuse static geometry; this cell tests the barrier's "
            "dynamical carrier directly)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT: the "
            "SRCT projector, the hooked drivers, the thermal envelope; the "
            "K10K room's registered seeds and the committed install "
            "vehicle e261_K10K_inst_resume.pt)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314 "
            "+ e113 cons 10901; the committed root + install records are "
            "the controls)",
        ],
        "whats_new": [
            "THE INTERFACE TEST ITSELF: the cliff's carrier question asked "
            "mechanistically for the first time — the SAME room, the SAME "
            "install stream (bit-identical draws), the SAME cons, with "
            "ONLY the corpus stream's LIVE TRAJECTORY added between the "
            "install writes (a delta no per-state eigenstructure can see "
            "by construction — the rooms and their spectra are identical "
            "across arms)",
            "THE INTERLEAVED DRIVER: the concurrent arm's 1:1 "
            "install:corpus schedule through one shared optimizer — the "
            "corpus steps' clipped gradients measured per step (the "
            "corpus ledger), the trajectory read at paired milestones",
            "THE SAME-SESSION PAIR + THE FRESHEST SCATTER POINT: the "
            "serial control re-run fresh (one instrument, one session) "
            "AND its delta vs e264's committed rung co-reported as the "
            "discriminator's disclosed noise floor",
        ],
        "gates": {},
    })
    log(f"E268 — THE ROOM-OPTIMIZER INTERFACE (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)}; vehicle = the committed K10K threshold "
        f"rung (room seeds {LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs "
        f"{ROOMS264_CK}); e001 + Dmix s{E261.INST_STEPS} gen "
        f"{E261.FRESH_GEN} + e113 cons s{E261.CONS_STEPS} seed "
        f"{E261.CONS_SEED} HELD; discriminator: post-g0 ratio >= "
        f"{POST_RATIO_BAR:.0f}x OR root outside +-{ROOT_BAND_FRAC:.0%} of "
        f"SERIAL's")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(26801)                 # global init only; every RNG is its own

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

    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} at j={j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]

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
                                         "wash runs in this cell"},
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
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_k10k_post = e264m["arms"]["K10K"]["install"]["post_cells"]["g0"]
    e264_k10k_root = e264m["arms"]["K10K"]["root"]["g0"]
    e264_k10k_kept = e264m["arms"]["K10K"]["install"][
        "ledger_kept_frac_median"]
    e264_free_pass = bool(e264m["gates"]["G_FREE"]["pass"])
    e264_free_l2 = e264m["gates"]["G_FREE"]["install_reproduction"][
        "l2_vs_committed_install_final"]
    vehicle = torch.load(CKPT_DIR / K10K_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    del vehicle
    G_PARENTS = {
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "K10K_post_g0": e264_k10k_post,
                         "K10K_root_g0": e264_k10k_root,
                         "K10K_kept_median": e264_k10k_kept,
                         "G_FREE_pass": e264_free_pass,
                         "G_FREE_install_l2": e264_free_l2},
        "e261_metrics": {"path": str(E261_METRICS),
                         "md5": md5of(E261_METRICS), "bound_md5": E261_MD5},
        "k10k_vehicle": {"path": f"runs/checkpoints/{K10K_CK}",
                         "md5": md5of(CKPT_DIR / K10K_CK),
                         "bound_md5": K10K_MD5,
                         "size": (CKPT_DIR / K10K_CK).stat().st_size,
                         "bound_size": K10K_SIZE, "state": vehicle_state},
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS264_CK)},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "e264_verdict": E264_VERDICT,
            "e264_K10K_post_g0": E264_K10K_POST_G0,
            "e264_K10K_root_g0": E264_K10K_ROOT_G0,
            "e264_K10K_kept_med": E264_K10K_KEPT_MED,
            "e264_K10K_post_gm12": E264_K10K_POST_GM12,
            "anchor_scatter_cross_session_root_g0": ANCHOR_SCATTER_G0,
            "cons_seed_scatter_session_root_g0": CONS_SEED_SCATTER_G0,
            "vehicle_md5": K10K_MD5, "vehicle_step": K10K_STEP,
            "vehicle_traj_steps": K10K_TRAJ_STEPS,
            "vehicle_ledger_max": K10K_LEDGER_MAX},
        "pass": bool(
            e264m["adjudication"]["verdict"] == E264_VERDICT
            and abs(e264_k10k_post - E264_K10K_POST_G0) < 1e-12
            and abs(e264_k10k_root - E264_K10K_ROOT_G0) < 1e-12
            and abs(e264_k10k_kept - E264_K10K_KEPT_MED) < 1e-12
            and e264_free_pass
            and md5of(E264_METRICS) == E264_MD5
            and md5of(E261_METRICS) == E261_MD5
            and md5of(CKPT_DIR / K10K_CK) == K10K_MD5
            and (CKPT_DIR / K10K_CK).stat().st_size == K10K_SIZE
            and vehicle_state["step"] == K10K_STEP
            and vehicle_state["traj_steps"] == K10K_TRAJ_STEPS
            and vehicle_state["ledger_max"] == K10K_LEDGER_MAX
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
            and (SMOKE or LADDER[0][0] == 10_000)),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e264 {E264_VERDICT} (K10K post g0 "
        f"{e264_k10k_post:.6f}, root {e264_k10k_root:.6f}, kept "
        f"{e264_k10k_kept:.4f}; G_FREE PASS at L2 {e264_free_l2:.1e}); the "
        f"vehicle {K10K_CK} md5/step-bound at s{K10K_STEP}")
    write_partial("P0b parents hard-bound")
    del e264m

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
            "form": "the vehicle's room == the committed K10K room (seeds "
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
    log(f"P1 G_ROOMK10K: the vehicle's room "
        f"{('bit-identical to e264_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e268_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e268's vehicle room (the flat basis is "
                 "net.parameters() order): the committed K10K room "
                 "(seeds 26113/26114), rebuilt + bit-gated vs "
                 "e264_rooms.pt",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][k10k_name].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": k10k_name,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS264_CK} "
                                    "(e264's committed K10K room)"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       "LATE span; rank "
                       f"{rooms.r_span}; the ledger's in-span columns)",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE VEHICLE ROOM: {k10k_name} (k={LADDER[0][0]}): BUILT + "
        "CERTIFIED + BIT-BOUND")
    write_partial("P1 the vehicle room built (parents bound + v-map loaded "
                  "+ span loaded + certification + bit-bind)")

    # ================= P2-P4: THE ARMS (SERIAL first — it anchors) ======
    arms_rec: dict = {}

    def read_arm_cells(sd: dict) -> tuple:
        net_ = G1.evl_load(sd)
        cells_ = {"gm12": G1.battery_cell(net_, gm12_ids, zid)["mean_pz"],
                  "g0": G1.battery_cell(net_, g0_ids, zid)["mean_pz"],
                  "gp12": G1.battery_cell(net_, bat_ids[12], zid)["mean_pz"],
                  "ce_r": G1.ce_fixed_cpu(net_, *r_eval_xy)}
        return net_, cells_

    def run_cons(arm: str, sd_install: dict) -> dict:
        cons = E261.chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e268_{arm}_cons_resume.pt" if SMOKE
                        else f"e268_{arm}_cons_resume.pt"), dev)
        return cons

    def read_root(arm: str, theta0: dict) -> dict:
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"]
                 for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - base_flat
        load_root = rooms.displacement_loads(d_root, k10k_name)
        root_ck = save_ckpt(
            f"e268_{arm}_root", theta0,
            {"desc": f"e268 ARM-{arm} root: e001 + Dmix s{E261.INST_STEPS} "
                     f"(gen {E261.FRESH_GEN}; {ARM_DESC[arm][:120]}...) + "
                     f"e113 cons s{E261.CONS_STEPS} (seed {E261.CONS_SEED} "
                     "HELD)",
             "arm": arm, "install_seed": E261.FRESH_GEN,
             "corpus_gen_seed": CORPUS_GEN_SEED,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e268_rooms.pt"})
        del root_net_arm
        landed = (E261.G1C_ROOT_G0 * (1 - MATCH_BAND)
                  <= cells["g+0"] <= E261.G1C_ROOT_G0 * (1 + MATCH_BAND))
        return {"cells": cells, "gm12": cells["g-12"], "g0": cells["g+0"],
                "displacement_loads": load_root, "checkpoint": root_ck,
                "landed": bool(landed)}

    # ---- ARM-SERIAL (e261's driver VERBATIM at the registered seeds) ----
    log("=" * 78)
    log(f"ARM-SERIAL — {ARM_DESC['SERIAL']}")
    inst_s = E261.chunked_install(
        "SERIAL-inst", k10k_name, G1.evl_load(base_sd), rooms,
        inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e268_SERIAL_inst_resume.pt" if SMOKE
                    else "e268_SERIAL_inst_resume.pt"), dev)
    sd_s = inst_s["sd"]
    net_s, cells_s = read_arm_cells(sd_s)
    d_s = flat_params_cpu(net_s) - base_flat
    load_s = rooms.displacement_loads(d_s, k10k_name)
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

    # ---- G_SERIAL_ANCHOR (non-halting texture vs e264's committed rung) -
    if not SMOKE:
        veh = torch.load(CKPT_DIR / K10K_CK, map_location="cpu",
                         weights_only=False)["model"]
        mine_s = torch.load(CKPT_DIR / "e268_SERIAL_inst_resume.pt",
                            map_location="cpu", weights_only=False)["model"]
        serial_l2 = float(np.sqrt(sum(float(((mine_s[k].float()
                                              - veh[k].float()) ** 2).sum())
                                    for k in veh if k in mine_s)))
        del veh, mine_s
    else:
        serial_l2 = None
    G_SERIAL_ANCHOR = {
        "form": "the SERIAL re-run vs e264's committed K10K rung (the "
                "vehicle's integrity): install L2 vs the committed vehicle "
                "<= 5e-3; behavioral |d| <= 0.02 on post g0 / root g0 "
                "(root g-12 reported with the lottery note, never gated). "
                "NON-HALTING: a failure is disclosed session texture "
                "(e261's G_ANCHOR precedent), never a bar move — the "
                "discriminator's primary comparison is the SAME-SESSION "
                "arm pair",
        "install_l2_vs_vehicle": serial_l2,
        "post_g0": {"mine": cells_s["g0"], "e264": E264_K10K_POST_G0,
                    "abs_diff": abs(cells_s["g0"] - E264_K10K_POST_G0)},
        "post_gm12": {"mine": cells_s["gm12"], "e264": E264_K10K_POST_GM12,
                      "abs_diff": abs(cells_s["gm12"] - E264_K10K_POST_GM12)},
        "kept_median": {"mine": med(led_s_kept), "e264": E264_K10K_KEPT_MED,
                        "abs_diff": abs(med(led_s_kept)
                                        - E264_K10K_KEPT_MED)},
        "bars": {"l2": 5e-3, "behavior": ANCHOR_BEHAV_TOL},
        "pass": None,          # filled after the root read (root g0 join)
    }
    metrics["gates"]["G_SERIAL_ANCHOR"] = G_SERIAL_ANCHOR

    if not inst_s.get("resumed_final", False):
        burst_cooldown("SERIAL inst->cons")
    cons_s = run_cons("SERIAL", sd_s)
    arms_rec["SERIAL"]["consolidation"] = {"traj": cons_s["traj"],
                                           "chunk_table":
                                               cons_s["chunk_table"]}
    arms_rec["SERIAL"]["root"] = read_root("SERIAL", cons_s["sd"])
    r_s = arms_rec["SERIAL"]["root"]
    G_SERIAL_ANCHOR["root_g0"] = {"mine": r_s["g0"], "e264": E264_K10K_ROOT_G0,
                                  "abs_diff": abs(r_s["g0"]
                                                  - E264_K10K_ROOT_G0)}
    G_SERIAL_ANCHOR["root_gm12_lottery"] = {
        "mine": r_s["gm12"], "note": "root g-12 is the wild lottery "
        "(e264's three cons draws on a bit-identical arm ranged 0.67); "
        "reported, never gated, never adjudicated"}
    if not SMOKE:
        G_SERIAL_ANCHOR["pass"] = bool(
            serial_l2 <= 5e-3
            and abs(cells_s["g0"] - E264_K10K_POST_G0) <= ANCHOR_BEHAV_TOL
            and abs(r_s["g0"] - E264_K10K_ROOT_G0) <= ANCHOR_BEHAV_TOL)
    else:
        G_SERIAL_ANCHOR["pass"] = True
        G_SERIAL_ANCHOR["vacuous"] = True
    metrics["arms"] = arms_rec
    log(f"ARM-SERIAL ROOT: g0 {r_s['g0']:.4f} landed={r_s['landed']} | g-12 "
        f"{r_s['gm12']:.4f} (lottery) | CE_R {r_s['cells']['ce_r']:.4f} | "
        f"d-root in-own-room {r_s['displacement_loads']['in_own_room']:.4f}")
    serial_l2_txt = ("SMOKE-vacuous" if serial_l2 is None
                     else f"{serial_l2:.3e}")
    anch_txt = ("PASS" if G_SERIAL_ANCHOR["pass"]
                else "FAIL — disclosed session texture")
    log(f"G_SERIAL_ANCHOR (non-halting texture): install L2 "
        f"{serial_l2_txt} (bar 5e-3), post g0 |d| "
        f"{abs(cells_s['g0'] - E264_K10K_POST_G0):.4f}, root g0 |d| "
        f"{abs(r_s['g0'] - E264_K10K_ROOT_G0):.4f} (bars "
        f"{ANCHOR_BEHAV_TOL}): {anch_txt}")
    write_partial("P3 ARM-SERIAL root + G_SERIAL_ANCHOR (non-halting)")

    # ---- ARM-CONCURRENT (the interleaved driver) -------------------------
    burst_cooldown("SERIAL -> CONCURRENT")
    log("=" * 78)
    log(f"ARM-CONCURRENT — {ARM_DESC['CONCURRENT']}")
    inst_c = chunked_install_concurrent(
        "CONCURRENT-inst", G1.evl_load(base_sd), rooms,
        inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
        r_eval_xy, zid,
        CKPT_DIR / ("smoke_e268_CONCURRENT_inst_resume.pt" if SMOKE
                    else "e268_CONCURRENT_inst_resume.pt"), dev)
    sd_c = inst_c["sd"]
    net_c, cells_c = read_arm_cells(sd_c)
    d_c = flat_params_cpu(net_c) - base_flat
    load_c = rooms.displacement_loads(d_c, k10k_name)
    del net_c
    led_c_kept = [v["kept_frac"] for v in inst_c["ledger"].values()]
    corp_ce = [v["ce"] for v in inst_c["corpus_ledger"].values()]
    corp_gn = [v["gn_clipped"] for v in inst_c["corpus_ledger"].values()]
    arms_rec["CONCURRENT"] = {
        "desc": ARM_DESC["CONCURRENT"], "install": {
            "traj": inst_c["traj"], "ledger": inst_c["ledger"],
            "corpus_ledger": inst_c["corpus_ledger"],
            "corpus_ce_median": med(corp_ce),
            "corpus_gn_clipped_median": med(corp_gn),
            "ledger_kept_frac_median": med(led_c_kept),
            "chunk_table": inst_c["chunk_table"], "steps": E261.INST_STEPS,
            "opt_steps": 2 * E261.INST_STEPS,
            "post_cells": cells_c, "displacement_loads": load_c,
            "resumed_final": bool(inst_c.get("resumed_final", False))},
    }
    log(f"ARM-CONCURRENT install done: post g0 {cells_c['g0']:.6f} g-12 "
        f"{cells_c['gm12']:.6f} CE_R {cells_c['ce_r']:.4f} | d v-excess "
        f"{load_c['v_excess']:.2f} cos-to-span {load_c['cos_to_span']:.4f} "
        f"in-own-room {load_c['in_own_room']:.4f} | kept med "
        f"{med(led_c_kept):.4f} | corpus CE med {med(corp_ce):.4f} "
        f"|g_corp| med {med(corp_gn):.3f}")
    write_partial("P4 ARM-CONCURRENT install + post dial + measured loads")

    if not inst_c.get("resumed_final", False):
        burst_cooldown("CONCURRENT inst->cons")
    cons_c = run_cons("CONCURRENT", sd_c)
    arms_rec["CONCURRENT"]["consolidation"] = {"traj": cons_c["traj"],
                                               "chunk_table":
                                                   cons_c["chunk_table"]}
    arms_rec["CONCURRENT"]["root"] = read_root("CONCURRENT", cons_c["sd"])
    r_c = arms_rec["CONCURRENT"]["root"]
    log(f"ARM-CONCURRENT ROOT: g0 {r_c['g0']:.4f} landed={r_c['landed']} | "
        f"g-12 {r_c['gm12']:.4f} (lottery) | CE_R "
        f"{r_c['cells']['ce_r']:.4f} | d-root in-own-room "
        f"{r_c['displacement_loads']['in_own_room']:.4f}")
    write_partial("P5 ARM-CONCURRENT root built + landing read")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    post_s = cells_s["g0"]
    post_c = cells_c["g0"]
    root_s = r_s["g0"]
    root_c = r_c["g0"]
    lo, hi = ROOT_BAND_FRAC, 1.0 + ROOT_BAND_FRAC
    root_lo, root_hi = lo * root_s, hi * root_s
    ratio = max(post_s, post_c) / max(min(post_s, post_c), RATIO_DEN_FLOOR)
    ratio_fires = bool(ratio >= POST_RATIO_BAR)
    root_outside = bool(root_c < root_lo or root_c > root_hi)

    hard = {k: v for k, v in metrics["gates"].items()
            if k != "G_SERIAL_ANCHOR"}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif ratio_fires or root_outside:
        verdict = "DYNAMICAL-CARRIER"
        clause = (f"the concurrent arm's "
                  f"{'expression' if ratio_fires else 'landing'} differs "
                  f"from the serial arm's beyond the cons/session scatter "
                  f"(post g0 {post_c:.6f} vs {post_s:.6f}, ratio "
                  f"{ratio:.2f}x {'>=' if ratio_fires else '<'} the "
                  f"{POST_RATIO_BAR:.0f}x bar; root g0 {root_c:.4f} vs "
                  f"{root_s:.4f}, window [{root_lo:.4f}, {root_hi:.4f}] "
                  f"{'OUTSIDE' if root_outside else 'inside'} the "
                  f"+-{ROOT_BAND_FRAC:.0%} bar) — the cliff's carrier is "
                  f"the room-trajecatory INTERFERENCE: the barrier is "
                  f"written in the dynamics, completing the static/"
                  f"dynamical split's mechanism story")
    elif (abs(ratio - POST_RATIO_BAR) < 0.25
          or min(root_c - root_lo, root_hi - root_c) < 0.01
          or not G_SERIAL_ANCHOR["pass"]):
        why = []
        if abs(ratio - POST_RATIO_BAR) < 0.25:
            why.append(f"the post-g0 ratio {ratio:.2f}x sits AT the "
                       f"{POST_RATIO_BAR:.0f}x bar's edge")
        if min(root_c - root_lo, root_hi - root_c) < 0.01:
            why.append(f"the concurrent root {root_c:.4f} sits AT the "
                       f"+-{ROOT_BAND_FRAC:.0%} window's edge")
        if not G_SERIAL_ANCHOR["pass"]:
            why.append("the serial anchor failed (cross-session texture) "
                       "— the same-session pair stands but any firing read "
                       "would carry the contamination caveat")
        verdict = "MIXED"
        clause = ("; ".join(why) + " — the trajectories verbatim, no "
                  "inflation")
    else:
        verdict = "STATIC-OR-DEEPER"
        clause = (f"the arms agree within the disclosed scatter (post g0 "
                  f"{post_c:.6f} vs {post_s:.6f}, ratio {ratio:.2f}x < the "
                  f"{POST_RATIO_BAR:.0f}x bar; root g0 {root_c:.4f} vs "
                  f"{root_s:.4f}, inside the [{root_lo:.4f}, {root_hi:.4f}] "
                  f"+-{ROOT_BAND_FRAC:.0%} window; disclosed scatter: "
                  f"install |d post g0| ~ 5e-7, cons cross-session root g0 "
                  f"{ANCHOR_SCATTER_G0:.4f}, session cons-seed "
                  f"{CONS_SEED_SCATTER_G0:.4f}) — the interface is not the "
                  f"carrier at this dose; the deep-spectrum/Lanczos branch "
                  f"inherits (the honest bound; the day's carrier question "
                  f"stays open with exactly one unmeasured branch left)")

    log("=" * 78)
    log(f"E268 VERDICT: {verdict}")
    log(f"  SERIAL     (the committed condition): post g0 {post_s:.6f} -> "
        f"root g0 {root_s:.4f} | kept {med(led_s_kept):.4f} | "
        f"{E261.INST_STEPS} opt steps")
    log(f"  CONCURRENT (the 1:1 interleave):     post g0 {post_c:.6f} -> "
        f"root g0 {root_c:.4f} | kept {med(led_c_kept):.4f} | "
        f"{2 * E261.INST_STEPS} opt steps")
    log(f"  post-g0 ratio {ratio:.2f}x (bar >= {POST_RATIO_BAR:.0f}x -> "
        f"{'FIRES' if ratio_fires else 'does not fire'}); root window "
        f"[{root_lo:.4f}, {root_hi:.4f}] -> "
        f"{'OUTSIDE' if root_outside else 'inside'}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> DYNAMICAL-CARRIER -> "
                           "STATIC-OR-DEEPER -> MIXED (frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "SERIAL": {"post_g0": post_s, "root_g0": root_s,
                       "kept_frac_median": med(led_s_kept),
                       "post_gm12": cells_s["gm12"],
                       "root_gm12": r_s["gm12"],
                       "post_ce_r": cells_s["ce_r"],
                       "root_ce_r": r_s["cells"]["ce_r"],
                       "root_in_own_room":
                           r_s["displacement_loads"]["in_own_room"]},
            "CONCURRENT": {"post_g0": post_c, "root_g0": root_c,
                           "kept_frac_median": med(led_c_kept),
                           "post_gm12": cells_c["gm12"],
                           "root_gm12": r_c["gm12"],
                           "post_ce_r": cells_c["ce_r"],
                           "root_ce_r": r_c["cells"]["ce_r"],
                           "root_in_own_room":
                               r_c["displacement_loads"]["in_own_room"],
                           "corpus_ce_median": med(corp_ce),
                           "corpus_gn_clipped_median": med(corp_gn)},
            "e264_committed_rung": {"post_g0": E264_K10K_POST_G0,
                                    "root_g0": E264_K10K_ROOT_G0,
                                    "kept": E264_K10K_KEPT_MED},
            "post_g0_ratio": ratio, "ratio_bar": POST_RATIO_BAR,
            "ratio_fires": ratio_fires,
            "root_window": [root_lo, root_hi],
            "root_outside_window": root_outside,
        },
        "scatter_disclosure": {
            "install_determinism_post_g0": "~5e-7 cross-session (e261's "
                                           "G_ANCHOR, bit-identical arm)",
            "cons_cross_session_root_g0": ANCHOR_SCATTER_G0,
            "cons_seed_session_root_g0": CONS_SEED_SCATTER_G0,
            "root_gm12_never_adjudicated": "the wild lottery (e264's "
                                           "three cons draws ranged 0.67 "
                                           "on a bit-identical arm); the "
                                           "g0 ruler is the stable one",
            "serial_anchor_this_session": {
                "install_l2": serial_l2,
                "post_g0_abs_diff": abs(post_s - E264_K10K_POST_G0),
                "root_g0_abs_diff": abs(root_s - E264_K10K_ROOT_G0)},
        },
        "verdict": verdict,
        "clause": clause,
    }
    write_partial("P7 ADJUDICATED (the frozen bars)")

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("the two arms share bit-identical "
            "install streams (one generator, seed 24314, one draw order), "
            "the same room (bit-gated to e264's committed K10K room), the "
            "same install dose and lr schedule, the same fresh fact-free "
            "base, and the same cons (seed 10901 HELD) — the ONLY delta "
            "is the corpus stream running CONCURRENTLY: one free 48-window "
            "corpus step after every install step, through the same "
            "optimizer. The rooms' eigenstructures are identical across "
            "arms BY CONSTRUCTION — a fate difference is the live "
            "trajectory's doing or nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
            "g-series standing lottery caveat carried verbatim); the "
            "arms' DIFFERENCE is the registered object, not any single "
            "point; the corpus-dose tripling is the intervention's own "
            "body, disclosed and ledgered (the corpus CE + clipped-grad "
            "norm per step)"),
        "cons_scatter_carried": ("the g0 ruler is the stable one (three "
            "cons draws on a bit-identical arm ranged 0.056); root g-12 "
            "is a height lottery (ranged 0.67) — reported, never "
            "adjudicated; the discriminator's bars are pre-sized beyond "
            "both the cross-session (0.0419) and session (0.0137) g0 "
            "scatter, and THIS session's serial-anchor delta is "
            "co-reported as the freshest point"),
        "loads_measured_not_nominal": ("every arm's ACTUAL geometry is "
            "reported: per-step kept fraction (install steps; the dose "
            "actually delivered), v-excess pre/post (vs e258's LOADED "
            "committed v-map), in-span pre AND applied (vs e246's "
            "committed LATE span), displacement cos-to-span + in-own-room "
            "at post-install and root, and the concurrent corpus stream's "
            "own ledger — never nominal"),
        "the_free_corpus_choice": ("the corpus steps are UNPROJECTED (the "
            "registered choice): the room constrains the install's "
            "writing directions — the vehicle verbatim; the natural "
            "corpus stream runs free so its trajectory can drag the state "
            "through and off the room between install writes; projecting "
            "it would remove exactly the interface under test"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all three branches and "
            "the trajectories are reported verbatim regardless"),
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
            "k10k_vehicle": {"file": f"runs/checkpoints/{K10K_CK}",
                             "md5": K10K_MD5,
                             "note": "the committed serial anchor (e261's "
                                     "cut, completed by e264); read-only "
                                     "here"},
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
            "rooms": rooms_ck,
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS},
        },
        "machinery": {
            "install_serial": "e261's chunked_install VERBATIM BY IMPORT "
                              "(E43.exposure Dmix; the dense-room hook; "
                              "the thermal envelope); the committed "
                              "lab/e261_rank_ladder.py unmodified",
            "install_concurrent": "THIS file's chunked_install_concurrent "
                                  "(the cell's one new driver): the "
                                  "install steps bit-identical to SERIAL "
                                  "+ the 1:1 interleaved free corpus "
                                  "steps through the shared AdamW",
            "cons": "e261's chunked_consolidate VERBATIM (e113; seed 10901 "
                    "HELD, NATURAL on both arms)",
            "hook": "e237's pre-Adam projection (backward -> clip 1.0 -> "
                    "project CPU fp64 SRCT -> write fp32 -> step; norm "
                    "not rescaled) — INSTALL steps only",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": {
            "burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
            "per_step_polls": "after EVERY opt step (both streams)",
            "early_end_margin_c": E261.TEMP_EARLY_END,
            "hard_line_c": E261.TEMP_HARD,
            "max_temp_seen_c": max((r["temp"] for r in thermal_log),
                                   default=None),
            "violations_ge_84c": sum(1 for r in thermal_log
                                     if r["temp"] >= E261.TEMP_HARD),
            "n_polls": len(thermal_log),
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }

    # ================= P10: figures ======================================
    make_interface_plot(rd, arms_rec, ratio, ratio_fires, root_lo, root_hi,
                        root_outside, verdict, clause, thermal_log)
    make_instrument_plot(rd, arms_rec, cert, thermal_log)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e268_room_interface.png"),
                          str(RD / "e268_instrument.png"),
                          str(RD / "REPORT.md"),
                          str(RD / "DRAFT_NOTES_ENTRY.md")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_interface_plot(rd, arms_rec, ratio, ratio_fires, root_lo, root_hi,
                        root_outside, verdict, clause, thermal_log):
    """THE CELL'S HEADLINE FIGURE: the install trajectories, the two
    discriminator reads (post g0 ratio + the root window), the corpus
    ledger, and the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    s = arms_rec["SERIAL"]["install"]
    c = arms_rec["CONCURRENT"]["install"]
    cols = {"SERIAL": "tab:blue", "CONCURRENT": "tab:red"}

    # (0,0) THE INSTALL TRAJECTORIES (expression through the install)
    ax = axes[0, 0]
    for arm, rec in arms_rec.items():
        tr = rec["install"]["traj"]
        ax.plot([t["step"] for t in tr], [t["g0_pz"] for t in tr],
                "o-", lw=1.6, ms=4, color=cols[arm], label=f"{arm} (install)")
    ax.axhline(E264_K10K_POST_G0, color="gray", ls=":", lw=1.2,
               label=f"e264 committed K10K post g0 {E264_K10K_POST_G0:.4f}")
    ax.set_xlabel("install step s (the corpus step follows each in "
                  "CONCURRENT)")
    ax.set_ylabel("g0 battery (mean p(Z))")
    ax.set_title("THE INSTALL TRAJECTORIES (expression through the install)",
                 fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

    # (0,1) THE DISCRIMINATOR READS
    ax = axes[0, 1]
    xs = [0, 1]
    posts = [arms_rec["SERIAL"]["install"]["post_cells"]["g0"],
             arms_rec["CONCURRENT"]["install"]["post_cells"]["g0"]]
    roots = [arms_rec["SERIAL"]["root"]["g0"],
             arms_rec["CONCURRENT"]["root"]["g0"]]
    ax2 = ax.twinx()
    b1 = ax.bar([x - 0.11 for x in xs], posts, width=0.2,
                color=[cols["SERIAL"], cols["CONCURRENT"]], alpha=0.85,
                label="post g0 (expression)")
    ax.axhline(E264_K10K_POST_G0, color="gray", ls=":", lw=1.0)
    b2 = ax2.bar([x + 0.11 for x in xs], roots, width=0.2,
                 color=[cols["SERIAL"], cols["CONCURRENT"]], alpha=0.45,
                 hatch="//", label="root g0 (landing)")
    ax2.axhline(root_lo, color="crimson", ls="--", lw=1.2)
    ax2.axhline(root_hi, color="crimson", ls="--", lw=1.2,
                label=f"+-10% window [{root_lo:.3f}, {root_hi:.3f}]")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"SERIAL\npost {posts[0]:.4f}\nroot {roots[0]:.4f}",
                        f"CONCURRENT\npost {posts[1]:.4f}\nroot "
                        f"{roots[1]:.4f}"], fontsize=8.5)
    ax.set_ylabel("post g0 (expression)")
    ax2.set_ylabel("root g0 (landing)")
    ax.set_title(f"THE DISCRIMINATOR READS — post ratio {ratio:.2f}x "
                 f"(bar >= {POST_RATIO_BAR:.0f}x: "
                 f"{'FIRES' if ratio_fires else 'no'}; root "
                 f"{'OUTSIDE' if root_outside else 'inside'})", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    # (1,0) THE CORPUS LEDGER + KEPT
    ax = axes[1, 0]
    cl = c["corpus_ledger"]
    steps = sorted(int(k) for k in cl.keys())
    ax.plot(steps, [cl[str(k)]["ce"] if str(k) in cl else cl[k]["ce"]
                    for k in steps], "s-", lw=1.4, ms=4, color="tab:red",
             label="corpus step CE (the concurrent stream)")
    sl = s["ledger"]
    ssteps = sorted(int(k) for k in sl.keys())
    ax.plot(ssteps, [sl[str(k)]["ce"] if str(k) in sl else sl[k]["ce"]
                     for k in ssteps], "o-", lw=1.2, ms=3,
             color="tab:blue", alpha=0.8,
             label="SERIAL install CE (in-batch corpus only)")
    ax.set_xlabel("install step s")
    ax.set_ylabel("batch CE")
    ax.set_title(f"THE CONCURRENT STREAM'S LEDGER (corpus CE; kept med "
                 f"S {arms_rec['SERIAL']['install'].get('ledger_kept_frac_median', float('nan')):.4f} "
                 f"/ C {arms_rec['CONCURRENT']['install'].get('ledger_kept_frac_median', float('nan')):.4f})",
                 fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

    # (1,1) THE VERDICT PANEL + THE THERMAL ENVELOPE
    ax = axes[1, 1]
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
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in thermal_log if r['temp'] >= E261.TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle(f"E268 — THE ROOM-OPTIMIZER INTERFACE -> {verdict}",
                 fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 150), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e268_room_interface.png", dpi=130)
    plt.close(fig)


def make_instrument_plot(rd, arms_rec, cert, thermal_log):
    """THE INSTRUMENT PAGE: the room's certification, the measured loads
    (v-excess / in-span / in-own-room), the consolidation trajectories,
    and the corpus stream's gradient ledger."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    k10k_name = RUNG[LADDER[0][0]]

    # (0,0) THE ROOM CERTIFICATION
    ax = axes[0, 0]
    r = cert["per_rung"][k10k_name]
    ax.bar(["kept2 measured\n(||Px||^2/||x||^2)", "k/N (expected)"],
           [r["kept2_mean"], r["kept2_expect"]], color=["tab:blue", "gray"])
    ax.errorbar([0], [r["kept2_mean"]],
                yerr=[r["kept2_bar_10sig"]], fmt="none", ecolor="black",
                capsize=4, label=f"10-sigma bar {r['kept2_bar_10sig']:.1e}")
    ax.set_ylabel("fraction")
    ax.set_title(f"THE VEHICLE ROOM (k={r['k']}, seeds {r['seeds']}, "
                 f"bit-bound to e264_rooms.pt; idem "
                 f"{r['idempotency_max']:.1e}; DCT rt "
                 f"{cert['dct_roundtrip_rel']:.1e}; span-ovl "
                 f"{r['span_overlap_mean']:.4f} vs ~"
                 f"{r['span_overlap_expect']:.4f})", fontsize=9.0)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y")

    # (0,1) THE MEASURED LOADS (post + root, both arms)
    ax = axes[0, 1]
    rows = [("post v-excess", "v_excess", "post"), ("post in-own-room", "in_own_room", "post"),
            ("root v-excess", "v_excess", "root"), ("root in-own-room", "in_own_room", "root")]
    xs = np.arange(len(rows))
    for i, arm in enumerate(ARMS):
        vals = [arms_rec[arm][("install" if ph == "post" else "root")]
                ["displacement_loads"][key] for _, key, ph in rows]
        ax.bar(xs + (i - 0.5) * 0.38, vals, width=0.36, alpha=0.85,
               color={"SERIAL": "tab:blue", "CONCURRENT": "tab:red"}[arm],
               label=arm)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in rows], fontsize=8.5)
    ax.set_ylabel("load (measured)")
    ax.set_title("THE MEASURED LOADS (v-excess vs e258's v-map; in-own-room "
                 "vs the vehicle room; cos-to-span annotated in metrics)",
                 fontsize=9.0)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y")

    # (1,0) THE CONSOLIDATION TRAJECTORIES
    ax = axes[1, 0]
    for arm in ARMS:
        tr = arms_rec[arm]["consolidation"]["traj"]
        ax.plot([t["step"] for t in tr], [t["g0_pz"] for t in tr], "o-",
                lw=1.4, ms=4,
                color={"SERIAL": "tab:blue", "CONCURRENT": "tab:red"}[arm],
                label=f"{arm} cons (g0)")
    ax.axhline(E261.G1C_ROOT_G0, color="black", ls=":", lw=1.2,
               label=f"committed g1c root g0 {E261.G1C_ROOT_G0:.4f}")
    ax.set_xlabel("cons step")
    ax.set_ylabel("g0 battery")
    ax.set_title("THE CONSOLIDATION (e113 VERBATIM, seed 10901 HELD — the "
                 "landing read; the g0 ruler, never g-12)", fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

    # (1,1) THE CORPUS STREAM'S GRADIENT LEDGER
    ax = axes[1, 1]
    cl = arms_rec["CONCURRENT"]["install"]["corpus_ledger"]
    steps = sorted(int(k) for k in cl.keys())
    gn = [cl[str(k)]["gn_clipped"] if str(k) in cl else cl[k]["gn_clipped"]
          for k in steps]
    ax.plot(steps, gn, "s-", lw=1.4, ms=4, color="tab:red",
            label="corpus step ||g_clipped||")
    il = arms_rec["CONCURRENT"]["install"]["ledger"]
    gn_i = [il[str(k)]["gn"] if str(k) in il else il[k]["gn"]
            for k in steps if (str(k) in il or k in il)]
    ax.plot([k for k in steps if (str(k) in il or k in il)], gn_i, "o-",
            lw=1.2, ms=3, color="tab:blue", alpha=0.8,
            label="install step ||g|| (pre-projection)")
    ax.set_xlabel("install step s")
    ax.set_ylabel("gradient norm")
    ax.set_title("THE CONCURRENT STREAM'S GRADIENT LEDGER (free steps; the "
                 "moment-coupling channel made visible)", fontsize=9.5)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

    fig.suptitle("E268 — the instrument page: the room, the loads, the "
                 "cons, the concurrent stream", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "e268_instrument.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
