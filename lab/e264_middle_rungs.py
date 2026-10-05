"""E264 — THE MIDDLE RUNGS (e261's deferred ladder, recovered; the cliff
pinned inside [1k, 237k]). Design dispatched 2026-10-05 (e261's triage
recovery protocol + T239's registered ask); this docstring carries e261's
ORIGINAL frozen bars VERBATIM — the un-triaged registration this cell
completes — committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim from the recovery dispatch): pin the expression
cliff inside [1k, 237k]. e261's coarse bracket read the expression curve a
STEP at 2-rung resolution (rank-10 dead 2.86e-5; rank-1k dead 0.000435;
rank-237k alive 0.3844 with an in-band landing; 884x adjacent jump). WHERE
inside [1k, 237k] does the cliff sit? The three deferred rungs {10k, 40k,
100k} complete the registered 5-rung curve at the same dose/conventions —
the anti-substrate's final quantitative form at the resolution the
original registration asked for.

REGISTERED BARS (e261's ORIGINAL bars, VERBATIM, still frozen — this cell
un-triages the ladder, it never re-registers):
  - SHARP-THRESHOLD — "the expression curve is step-like (post_g0 jumps
    > 10x between adjacent rungs somewhere) AND the landing curve enters
    the band at a locatable rung — the anti-substrate's final form: a
    dimensional threshold at k ~ [the rung]; memories need k*
    dimensions, full stop"
  - GRADUAL — "both curves rise smoothly (no rung-to-rung jump > 10x in
    expression; the landing approaches the band asymptotically) — the
    barrier is a soft capacity gradient, not a threshold; the last 0.08
    of root g0 is spread across the ladder"
  - MIXED — "the curves verbatim; expression step-like but landing
    gradual (or vice versa) — mapped honestly"

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE CELL := the three deferred rungs, e261's seeds VERBATIM — K10K
    (k=10,000; seeds 26113/26114) RESUMED from e261's cut vehicle
    runs/checkpoints/e261_K10K_inst_resume.pt (s278/400; md5 + step +
    ledger-extent bit-gated at runtime: G_K10KRESUME), K40K (k=40,000;
    26115/26116) and K100K (k=100,000; 26117/26118) FRESH; every arm the
    SAME fresh root (e001, fact-free-gated) and the SAME dose (Dmix s400
    gen 24314 + e113 cons s300 seed 10901 HELD; installs share
    bit-identical streams by construction); the hook VERBATIM (backward
    -> clip 1.0 -> project (CPU fp64, write fp32) -> opt.step; norm NOT
    rescaled); cons NATURAL on all arms.
  * THE ADJUDICATION LADDER := the FULL registered 5 rungs {1k, 10k, 40k,
    100k, 237k} — rungs 1k/237k CITED from runs/e261/metrics.json
    (md5-bound; values hard-bound below), the middle rungs THIS session's
    fresh reads; ARM-FREE co-plotted at k=N (this session's fresh re-run,
    G_FREE-gated); e246's committed ALIGNED 2.86e-5 at rank 10 co-marked
    as the dead context point (NOT a rung); e260's committed RANDOM
    co-marked at 237k (context, not a rung).
  * THE STITCH (this cell's one interpretive act, disclosed at birth):
    post-install reads stitch cleanly across sessions — the install is
    deterministic given its ckpt (e261's G_ANCHOR: install L2 6.7e-5, |
    d post g0| 5e-7 across sessions on a bit-identical arm) — but ROOT
    reads carry the cons lottery's cross-session scatter (e261's G_ANCHOR:
    |d root g0| 0.0419, |d root g-12| 0.1483 on that same bit-identical
    arm). THE CONS-SCATTER DISCLOSURE, carried in the clause and the
    reads: any rung whose root g0 sits within 0.0419 of a band edge is
    flagged scatter-fragile (the named instance: the 237k rung — e261's
    read 0.7106 IN-band by 0.0403, e260's committed read 0.6686 OUT by
    0.0017, the SAME arm).
  * THE CONS-SEED REPLICATE (the anchor scatter priced, T239's
    registered ask): e261's committed K237K install-final + a fresh cons
    s300 at seed 10902 (the next seed; nothing shopped) — a CO-REPORT,
    never a rung; |d root g0| vs the 237k rung's cited value prices the
    landing read's noise floor alongside e261's cross-session anchor
    scatter.
  * e264's G_FREE := e261's tiers (i)-(iii) VERBATIM on THIS session's
    fresh FREE re-run (the ceiling control; a failure HALTS the cell):
    (i) install L2 <= 5e-3 vs g1c_install_resume.pt AND behavioral g-12
    |d| <= 5e-3; (ii) FREE's root in the matched band; (iii) cons
    tracking median |g0 diff| <= 0.05 vs g1c's committed traj. Tier (iv)
    stays dropped (no washes, inherited).
  * e264's GATE SET := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR(bank),
    G_INSTMASK, G_PARENTS(+ the e261 stitch bind), G_BASE, G_ROOT,
    G_VMBIND, G_SPANBIND, G_PROJ, G_K10KRESUME, G_FREE}. e261's G_ANCHOR
    failure is INHERITED AS DISCLOSURE (the stitch's disclosed noise
    floor), NOT as an e264 gate: the 237k rung is cited, never re-run,
    and no anchor rung exists in the fresh set — the anchor gate's
    original object (binding a RE-RUN rung to e260's committed arm)
    does not arise here; no bar moves.
  * The adjudication math VERBATIM from e261's frozen
    operationalizations: 'post_g0 jumps > 10x' := max adjacent ratio
    post_g0(k_hi)/max(post_g0(k_lo), 1e-6) > 10 over the FULL 5-rung
    ascending ladder; the expression floor 0.05; the matched band
    [0.670278, 0.819229] (+-10% of the committed g1c root g0
    0.7447534203529358); threshold := the smallest in-band k, bracket
    co-reported; GRADUAL reach := top rung root g0 >= floor - 0.05;
    composite TEXTURE -> SHARP-THRESHOLD -> GRADUAL -> MIXED.

CHECKS (the dispatch's, in force): the K10K resume bit-gated to its ckpt;
the rooms certified per rung (the script's own G_PROJ); the
kept~sqrt(k/N) covariation co-plotted; the cons-scatter disclosure
carried; n=1 per rung (the lottery note carried verbatim); nothing
guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); the
owner's max-priority window — bursts <= 175 s, per-step thermal polls at
a 78 C margin, 40 s cooldowns, the 84 C never-past line recorded; CPU
fp64 dense projections (pocketfft workers 2); CPU probing threads 4.

Outputs: runs/e264/{metrics.json (PROGRESSIVE),
e264_middle_rungs.png, e264_instrument.png, REPORT.md,
DRAFT_NOTES_ENTRY.md, run.log (gitignored)}; checkpoints
runs/checkpoints/e264_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e264_middle_rungs.py    (E264_SMOKE=1 shakedown)
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
import torch.nn.functional as F                        # noqa: E402

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

SMOKE = os.environ.get("E264_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e264_smoke" if SMOKE else "e264"
assert torch.cuda.is_available(), "e264 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (disclosed at birth, in force before ANY machinery call):
# e261's ported drivers resolve their module globals (log / NAME / LADDER /
# RUNG_NAMES / INST_STEPS / CONS_STEPS / CONS_SEED / SMOKE) AT CALL TIME
# through e261's module namespace — rebound HERE so they write THIS cell's
# log, label THIS cell's envelope polls, and run THIS cell's ladder. The
# committed lab/e261_rank_ladder.py itself is untouched (extend, don't
# repeat: the drivers' arithmetic is imported, not copied).
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
CKPT_DIR = GB.CKPT_DIR

# ---- THE MIDDLE RUNGS (e261's deferred seeds VERBATIM) + e264's own paths.
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — RESUMED from e261's cut vehicle
    (40_000, 26115, 26116),       # K40K — fresh
    (100_000, 26117, 26118),      # K100K — fresh
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (64, 26113, 26114),
    (512, 26117, 26118),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG_NAMES = ({k: f"K{k // 1000}K" for k, _, _ in LADDER} if not SMOKE
              else {k: f"K{k}" for k, _, _ in LADDER})
ARMS = ("FREE",) + tuple(RUNG_NAMES[k] for k, _, _ in LADDER)
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG_NAMES

# THE ADJUDICATION LADDER: the FULL 5 rungs (1k/237k cited from e261)
FULL_LADDER_KS = (1_000, 10_000, 40_000, 100_000, 237_123)
CITED_KS = (1_000, 237_123)

# THE RESUME VEHICLE (bit-gated): e261's cut, frozen as-is
K10K_CK = "e261_K10K_inst_resume.pt"
K10K_MD5 = "ce70e580c5f2c8223dc2eb6a5b11e6e6"
K10K_STEP = 278
K10K_SIZE = 32956239
K10K_TRAJ_STEPS = [1, 100, 200]
K10K_LEDGER_MAX = 270

# THE STITCH (e261's committed record, HARD-BOUND — Rule 12; md5-gated)
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "f460475d8e6b76f0719e91c1e9c6041b"
E261_K1K = {"post_g0": 0.00043458465370349586,
            "post_gm12": 0.00035702373133972287,
            "root_g0": 0.5768678784370422,
            "root_gm12": 0.16171452403068542,
            "kept": 0.016095496225535792}
E261_K237K = {"post_g0": 0.38436421751976013,
              "post_gm12": 0.07495748135108948,
              "root_g0": 0.7105798125267029,
              "root_gm12": 0.3573005497455597,
              "kept": 0.2945979051104054}
E261_FREE = {"post_g0": 0.5302198529243469,
             "root_g0": 0.769062340259552,
             "root_gm12": 0.9118462204933167}
# the cons lottery's cross-session law (e261's G_ANCHOR, the same
# bit-identical arm): the stitch's disclosed noise floor
ANCHOR_SCATTER_G0 = 0.04194730520248413
ANCHOR_SCATTER_GM12 = 0.14829494059085846
E260_RANDOM = {"post_g0": 0.38436469435691833, "root_g0": 0.6686325073242188}

CONS_SEED_HELD = E261.CONS_SEED          # 10901 (verbatim, all rungs)
CONS2_SEED = 10902                       # the replicate's seed (the next one)

G0_ZERO_FLOOR = E261.G0_ZERO_FLOOR       # 0.05 — the frozen expression floor
JUMP_BAR = E261.JUMP_BAR                 # 10.0 — the registered >10x bar
RATIO_DEN_FLOOR = E261.RATIO_DEN_FLOOR   # 1e-6 — the ratio denominator floor
GRADUAL_REACH = E261.GRADUAL_REACH       # 0.05
MATCH_BAND = E261.MATCH_BAND             # 0.10
G_READ_TOL = E261.G_READ_TOL             # 5e-3

REGISTERED = {
    "question_verbatim": "pin the expression cliff inside [1k, 237k]. "
        "e261's coarse bracket read the expression curve a STEP at 2-rung "
        "resolution (rank-10 dead 2.86e-5; rank-1k dead 0.000435; "
        "rank-237k alive 0.3844 with an in-band landing; 884x adjacent "
        "jump). WHERE inside [1k, 237k] does the cliff sit? The three "
        "deferred rungs {10k, 40k, 100k} complete the registered 5-rung "
        "curve at the same dose/conventions.",
    "bars_verbatim": E261.REGISTERED["bars_verbatim"],   # e261's ORIGINAL
    "operationalizations": (
        "frozen BEFORE compute: THE CELL := the three deferred rungs at "
        "e261's registered seeds (K10K 26113/26114 RESUMED from e261's cut "
        f"vehicle {K10K_CK} at s{K10K_STEP}/400, md5/step/ledger bit-gated "
        "(G_K10KRESUME); K40K 26115/26116 and K100K 26117/26118 FRESH); "
        "every arm the SAME fresh root (e001) and SAME dose (Dmix s400 gen "
        "24314 + e113 cons s300 seed 10901 HELD; bit-identical streams); "
        "the hook VERBATIM (backward -> clip 1.0 -> project CPU fp64 -> "
        "opt.step; norm NOT rescaled); cons NATURAL; THE ADJUDICATION "
        "LADDER := the FULL 5 rungs {1k, 10k, 40k, 100k, 237k} with rungs "
        "1k/237k CITED from e261's committed metrics (md5-bound, values "
        "hard-bound) and FREE re-run fresh here (G_FREE tiers i-iii, HALT "
        "on failure); e246's rank-10 2.86e-5 co-marked as the dead context "
        "point; THE STITCH: post g0 install-deterministic (e261's G_ANCHOR "
        "install L2 6.7e-5, |d post g0| 5e-7 cross-session), root reads "
        f"carry the cons-lottery scatter (|d root g0| {ANCHOR_SCATTER_G0:.4f}"
        f" / |d root g-12| {ANCHOR_SCATTER_GM12:.4f} on a bit-identical "
        "arm) — rungs within 0.0419 of a band edge flagged scatter-fragile "
        "(named instance: the 237k rung — e261 0.7106 in by 0.0403, e260 "
        "committed 0.6686 out by 0.0017, the SAME arm); THE CONS-SEED "
        "REPLICATE := e261's committed K237K install-final + fresh cons "
        "s300 seed 10902 — a CO-REPORT pricing the anchor scatter, never a "
        "rung; e264's gate set := {G_NAMEFREE, G_SPLICE, G_BATTERY, "
        "G_ANCHOR(bank), G_INSTMASK, G_PARENTS(+e261 stitch bind), G_BASE, "
        "G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_K10KRESUME, G_FREE} — "
        "e261's G_ANCHOR failure inherited AS DISCLOSURE (the stitch's "
        "noise floor), not as an e264 gate (no anchor rung is re-run here; "
        "no bar moves); adjudication math VERBATIM from e261 (adjacent "
        f"ratio with the {RATIO_DEN_FLOOR:.0e} denominator floor; the "
        f"{G0_ZERO_FLOOR} expression floor; the band "
        "[0.670278, 0.819229]; threshold = smallest in-band k with "
        "bracket; GRADUAL reach 0.05; composite TEXTURE -> SHARP-THRESHOLD "
        "-> GRADUAL -> MIXED)."),
    "registration": "bars + question frozen VERBATIM (e261's ORIGINAL "
        "un-triaged registration, recovered whole); this script committed "
        "at birth BEFORE any compute; adjudicate against exactly this; no "
        "bar shopping.",
}

deviations: list[str] = [
    "THE RECOVERY (disclosed): e264's first executor died ~2 min in on a "
    "model-request failure; verified NO partial artifacts (clean tree, no "
    "runs/e264, no e264 ckpts) — this cell re-dispatches whole.",
    "THE RESUME VEHICLE: completing K10K overwrites "
    f"{K10K_CK} with the s400 final (the journal-resume path's design; "
    "the machinery saves its vehicle every chunk); the s278 cut state is "
    "preserved by the frozen md5 in this registration + e261's committed "
    "triage record; the K10K bit-gate (md5 + step + ledger extent, "
    "checked BEFORE compute) binds the load to exactly that cut.",
    "THE STITCH (this cell's one interpretive act, disclosed at birth): "
    "rungs 1k/237k/FREE-ceiling-context are CITED from e261's committed "
    "record rather than re-run — the post-install curve stitches cleanly "
    "(install-deterministic), the landing curve carries the cons "
    "lottery's cross-session scatter (0.0419 g0 / 0.1483 g-12, e261's "
    "G_ANCHOR) — flagged per-rung where it threatens a band-edge call, "
    "and priced by the K237K cons-seed replicate.",
    "NO ANCHOR RE-RUN in this cell (disclosed): the 237k rung is e261's "
    "committed read (itself the re-run of e260's committed RANDOM arm); "
    "e264 constructs no 237k room and runs no 237k install — the anchor "
    "gate has no object here; e261's G_ANCHOR failure is inherited as "
    "the stitch's disclosed noise floor, never as an e264 gate.",
    "e261's machinery PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "hooked chunked install/cons drivers (bit-identical arithmetic + "
    "draw order), the thermal envelope (per-step polls, 78C margin, 175s "
    "bursts, 40s cooldowns, the 84C line), the progressive-metrics + "
    "resume-ckpt conventions — the module-global rebinding (log/NAME/"
    "LADDER/RUNG_NAMES, disclosed in-code) retargets the drivers' I/O to "
    "this cell; the committed lab/e261_rank_ladder.py is NOT modified.",
    "NO WASH IN THIS CELL (inherited from e261's registration): the "
    "registered readouts are post-install g0 + root g0 + kept-fraction "
    "per rung — the CURVES are the object; the retention/occupancy "
    "question stays T237's separately registered follow-up.",
    "THE KEPT-K COUPLING (inherited, disclosed): the projection's norm "
    "is NOT rescaled, so the delivered dose covaries with the rung "
    "(kept ~ sqrt(k/N): 0.0604 at 10k -> 0.1911 at 100k); the "
    "kept-fraction ledger is the disclosed covariate on every rung and "
    "the instrument page co-plots root g0 against kept (the confound "
    "view); e258's VLIGHT landed at kept ~0.10 — dose alone does not "
    "write the curve's shape.",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger; no new "
    "history is run. The installs' own fresh Adam state remains the "
    "mechanism's channel (the rungs restrict the GRADIENT; the "
    "in-own-room ledger says how much the state stayed).",
    "n=1 per rung, one lineage, TWO sessions stitched (e261 + e264; the "
    "g-series standing lottery caveat carried verbatim); the curve's "
    "SHAPE is the registered object, not any single point; nothing "
    "guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E264_SMOKE=1): 8-step installs/cons, rooms {64, 512}, "
    "all paths smoke_-prefixed, G_K10KRESUME vacuous (explicit pass, "
    "disclosed), own smoke dir; NOTHING adjudicated or gated (SMOKE "
    "stamp on every read).",
]

device_events: list[dict] = []
thermal_log: list[dict] = []


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
    if SMOKE:
        return CKPT_DIR / f"smoke_e264_{arm}_inst_resume.pt"
    if arm == "K10K":
        return CKPT_DIR / K10K_CK            # THE RESUME VEHICLE (e261's cut)
    return CKPT_DIR / f"e264_{arm}_inst_resume.pt"


def cons_resume_path(arm: str) -> Path:
    if SMOKE:
        return CKPT_DIR / f"smoke_e264_{arm}_cons_resume.pt"
    return CKPT_DIR / f"e264_{arm}_cons_resume.pt"


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
        "experiment": "e264_middle_rungs",
        "phase": "THE MIDDLE RUNGS — e261's deferred ladder recovered: "
                 "K10K (resume) + K40K + K100K at the same dose/conventions; "
                 "the FULL 5-rung expression + landing curves vs k; the "
                 "cliff pinned inside [1k, 237k] — SHARP vs GRADUAL vs MIXED",
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
                      f"at a {E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s, the {E261.TEMP_HARD:.0f}C "
                      "never-past line recorded",
            "trainings": "1 resume install tail (K10K s279-400) + 2 fresh "
                         "installs s400 (K40K/K100K) + 1 fresh FREE install "
                         "s400 + 4 cons s300 (3 rungs + FREE) + 1 cons-seed "
                         "replicate s300; NO washes",
        },
        "deviations": deviations,
        "builds_on": [
            "T239 / e261 (THE registration this cell completes: the three "
            "deferred rungs at the registered seeds; the committed bracket "
            "rungs cited (K1K post g0 0.000435 -> root 0.5769; K237K post "
            "0.3844 -> root 0.7106 in-band); the committed machinery "
            "imported whole; the K10K cut vehicle s278/400 on disk)",
            "T237 / e260 (the anchor arm e261's top rung re-ran; its "
            "committed RANDOM read root g0 0.6686 vs e261's 0.7106 is the "
            "cons-scatter disclosure's named instance)",
            "T235 / e258 (the committed v-map this cell LOADS; the VLIGHT "
            "kept-0.10 landing — the dose-honesty precedent)",
            "T226 / e246 (the anti-substrate's origin: the ALIGNED rank-10 "
            "dead point 2.86e-5; the committed LATE span feeds the in-span "
            "ledger columns)",
            "T181 / g1c (the fresh-root lineage: e001 + Dmix s400 gen 24314 "
            "+ e113 cons 10901; the committed root + install records are "
            "the controls)",
        ],
        "whats_new": [
            "THE MIDDLE RUNGS: the deferred k in {10k, 40k, 100k} run at "
            "last — the rank axis walked INSIDE e261's cliff bracket for "
            "the first time; the FULL 5-rung expression + landing curves "
            "complete at the original resolution (the K10K resume "
            "bit-gated to e261's cut vehicle; every rung the same fresh "
            "root, the same dose, bit-identical streams — the room is the "
            "only delta)",
            "THE STITCH + THE CONS-SEED REPLICATE: the cross-session "
            "cons-lottery scatter (0.0419 g0 / 0.1483 g-12) made a "
            "first-class disclosed covariate of the landing curve (per-"
            "rung scatter-fragile flags at the band edges) and PRICED by "
            "a fresh cons draw (seed 10902) on e261's committed K237K "
            "install-final — the anchor scatter measured, not assumed",
            "THE COMPLETED CURVE: post_g0(k) and root_g0(k) over the full "
            "registered ladder with the threshold located (floor-crossing "
            "rung + band-entry bracket) and the kept-fraction covariate "
            "co-plotted (root g0 vs kept — the confound view)",
        ],
        "gates": {},
    })
    log(f"E264 — THE MIDDLE RUNGS (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)} at identical dose (e001 + Dmix s400 gen "
        f"{E261.FRESH_GEN} + e113 cons s300 seed {CONS_SEED_HELD}); the "
        f"adjudication ladder: {'/'.join(str(k) for k in FULL_LADDER_KS)} "
        f"(1k/237k cited from e261, md5 {E261_MD5[:8]}); landing = root g0 "
        f"within +-{MATCH_BAND:.0%} of {E261.G1C_ROOT_G0:.4f} -> "
        f"[{E261.G1C_ROOT_G0 * (1 - MATCH_BAND):.4f}, "
        f"{E261.G1C_ROOT_G0 * (1 + MATCH_BAND):.4f}]; expression floor g0 "
        f"{G0_ZERO_FLOOR}; jump bar >{JUMP_BAR:.0f}x; scatter flag "
        f"{ANCHOR_SCATTER_G0:.4f}")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(26401)                 # global init only; every RNG is its own

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
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape) for j in G1.GEOS},
                 "expected": {"g-12": [60, G1.PRE - 12], "g0": [60, G1.PRE],
                              "g+12": [60, G1.PRE + 12]},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape) == [60, G1.PRE + 12])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()                                  # (60, 256)
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])    # (60, 256)
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
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])          # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

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
    g1c = json.loads(E261.G1C_METRICS.read_text(encoding="utf-8"))
    g1c_root_cells = g1c["root_build"]["root_cells"]
    g1c_root_gm12 = g1c_root_cells["gm12"]
    g1c_root_g0 = g1c_root_cells["g0"]
    g1c_post_inst = g1c["root_build"]["install"]["post_cells"]
    e246 = json.loads(E261.E246_METRICS.read_text(encoding="utf-8"))
    e246_aligned_post_g0 = e246["arms"]["ALIGNED"]["install"]["post_cells"]["g0"]
    e258 = json.loads(E261.E258_METRICS.read_text(encoding="utf-8"))
    e260 = json.loads(E261.E260_METRICS.read_text(encoding="utf-8"))
    e260_rand_post = e260["arms"]["RANDOM"]["install"]["post_cells"]["g0"]
    e260_rand_root_g0 = e260["arms"]["RANDOM"]["root"]["g0"]
    e260_rand_root_gm12 = e260["arms"]["RANDOM"]["root"]["gm12"]
    e260_rand_kept = e260["arms"]["RANDOM"]["install"]["ledger_kept_frac_median"]
    e260_k = int(e260["rooms"]["k_rule"]["k"])
    # the stitch source: e261's committed record
    e261m = json.loads(E261_METRICS.read_text(encoding="utf-8"))
    e261_k1k = e261m["arms"]["K1K"]
    e261_k237k = e261m["arms"]["K237K"]
    e261_free = e261m["arms"]["FREE"]
    e261_adj = e261m["adjudication"]

    G_PARENTS = {
        "g1c_metrics": {"path": str(E261.G1C_METRICS),
                        "md5": md5of(E261.G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root_gm12, "root_g0": g1c_root_g0},
        "e246_metrics": {"path": str(E261.E246_METRICS),
                         "md5": md5of(E261.E246_METRICS),
                         "verdict": e246["adjudication"]["verdict"],
                         "ALIGNED_post_g0": e246_aligned_post_g0},
        "e258_metrics": {"path": str(E261.E258_METRICS),
                         "md5": md5of(E261.E258_METRICS),
                         "verdict": e258["adjudication"]["verdict"],
                         "k": e258["vmap"]["k_rule"]["k"]},
        "e260_metrics": {"path": str(E261.E260_METRICS),
                         "md5": md5of(E261.E260_METRICS),
                         "verdict": e260["adjudication"]["verdict"],
                         "RANDOM_post_g0": e260_rand_post,
                         "RANDOM_root_g0": e260_rand_root_g0,
                         "RANDOM_root_gm12": e260_rand_root_gm12,
                         "RANDOM_kept_median": e260_rand_kept, "k": e260_k},
        "e261_metrics": {"path": str(E261_METRICS), "md5": md5of(E261_METRICS),
                         "verdict": e261_adj["verdict"],
                         "K1K": E261_K1K, "K237K": E261_K237K,
                         "FREE": E261_FREE,
                         "G_ANCHOR_scatter": {
                             "root_g0_abs_diff": ANCHOR_SCATTER_G0,
                             "root_gm12_abs_diff": ANCHOR_SCATTER_GM12}},
        "hardbound": {
            "g1c_root_g0": E261.G1C_ROOT_G0,
            "e246_ALIGNED_post_g0": E261.E246_ALIGNED_POST_G0,
            "e258_k": E261.E258_K_HARD,
            "e260_RANDOM_post_g0": E261.E260_RANDOM_POST_G0,
            "e260_RANDOM_root_g0": E261.E260_RANDOM_ROOT_G0,
            "e261_K1K": E261_K1K, "e261_K237K": E261_K237K,
            "e261_FREE": E261_FREE, "e261_md5": E261_MD5},
        "pass": bool(
            g1c["adjudication"]["verdict"] == E261.G1C_VERDICT
            and abs(g1c_root_g0 - E261.G1C_ROOT_G0) < 1e-12
            and e246["adjudication"]["verdict"] == E261.E246_VERDICT
            and abs(e246_aligned_post_g0 - E261.E246_ALIGNED_POST_G0) < 1e-12
            and e258["adjudication"]["verdict"] == E261.E258_VERDICT
            and int(e258["vmap"]["k_rule"]["k"]) == E261.E258_K_HARD
            and e260["adjudication"]["verdict"] == E261.E260_VERDICT
            and abs(e260_rand_post - E261.E260_RANDOM_POST_G0) < 1e-12
            and abs(e260_rand_root_g0 - E261.E260_RANDOM_ROOT_G0) < 1e-12
            and e260_k == E261.E258_K_HARD
            and md5of(E261_METRICS) == E261_MD5
            and abs(e261_k1k["install"]["post_cells"]["g0"]
                    - E261_K1K["post_g0"]) < 1e-12
            and abs(e261_k1k["root"]["g0"] - E261_K1K["root_g0"]) < 1e-12
            and abs(e261_k1k["install"]["ledger_kept_frac_median"]
                    - E261_K1K["kept"]) < 1e-12
            and abs(e261_k237k["install"]["post_cells"]["g0"]
                    - E261_K237K["post_g0"]) < 1e-12
            and abs(e261_k237k["root"]["g0"] - E261_K237K["root_g0"]) < 1e-12
            and abs(e261_k237k["install"]["ledger_kept_frac_median"]
                    - E261_K237K["kept"]) < 1e-12
            and abs(e261_free["install"]["post_cells"]["g0"]
                    - E261_FREE["post_g0"]) < 1e-12
            and abs(e261_free["root"]["g0"] - E261_FREE["root_g0"]) < 1e-12),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS — g1c root g0 "
        f"{g1c_root_g0:.6f}; e246/e258/e260 bound; THE STITCH: e261 "
        f"(md5 {md5of(E261_METRICS)[:8]}) K1K post g0 "
        f"{E261_K1K['post_g0']:.6f} -> root {E261_K1K['root_g0']:.4f} | "
        f"K237K post {E261_K237K['post_g0']:.6f} -> root "
        f"{E261_K237K['root_g0']:.4f} | FREE post {E261_FREE['post_g0']:.4f}"
        f" -> root {E261_FREE['root_g0']:.4f}")
    write_partial("P0b parents + the e261 stitch bound")

    # ---- G-K10KRESUME (the bit-gate, BEFORE any compute) ---------------
    k10k_path = CKPT_DIR / K10K_CK
    if not SMOKE:
        ck = torch.load(k10k_path, map_location="cpu", weights_only=False)
        G_K10K = {
            "form": f"the K10K resume vehicle bit-gated BEFORE compute: "
                    f"{K10K_CK} exists with md5 == the frozen "
                    f"{K10K_MD5}, step == {K10K_STEP}, traj steps == "
                    f"{K10K_TRAJ_STEPS}, max ledger key == {K10K_LEDGER_MAX} "
                    f"(e261's cut, committed in its triage record)",
            "exists": bool(k10k_path.exists()),
            "md5": md5of(k10k_path), "md5_expected": K10K_MD5,
            "size": k10k_path.stat().st_size, "size_expected": K10K_SIZE,
            "step": int(ck["step"]), "step_expected": K10K_STEP,
            "traj_steps": [t["step"] for t in ck["traj"]],
            "ledger_max_key": max(int(k) for k in ck["ledger"].keys()),
            "n_chunks_at_cut": int(ck.get("n_chunks", 0)),
            "pass": bool(md5of(k10k_path) == K10K_MD5
                         and int(ck["step"]) == K10K_STEP
                         and [t["step"] for t in ck["traj"]] == K10K_TRAJ_STEPS
                         and max(int(k) for k in ck["ledger"].keys())
                         == K10K_LEDGER_MAX),
        }
        del ck
        assert G_K10K["pass"], f"K10K resume bit-gate FAILED: {G_K10K}"
        metrics["gates"]["G_K10KRESUME"] = G_K10K
        log(f"P0c G_K10KRESUME: {K10K_CK} at s{G_K10K['step']}/400 (md5 "
            f"{G_K10K['md5'][:8]}, traj {G_K10K['traj_steps']}, ledger to "
            f"s{G_K10K['ledger_max_key']}): the bit-gate HOLDS")
        write_partial("P0c G_K10KRESUME bit-gate HELD (the cut vehicle intact)")
    else:
        metrics["gates"]["G_K10KRESUME"] = {
            "form": "SMOKE: the resume gate is VACUOUS (no smoke copy of the "
                    "cut vehicle; the full run's bit-gate is the real one)",
            "pass": True, "vacuous": True}

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
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, CE_R {base_ce_r:.4f}): PASS")
    write_partial("P0d G-BASE PASSED")

    # ================= P1: THE ROOMS (v-map + span + ladder + cert) ======
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

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N
                     and int(vmap_art["meta"]["k"]) == E261.E258_K_HARD),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND

    span_art = torch.load(CKPT_DIR / E261.SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / E261.SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / E261.SPAN_CK) == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND

    # ---- THE MIDDLE RUNGS BUILT + CERTIFIED (the machinery's own gates) --
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the middle rungs' SRCT rooms certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the DCT "
                "roundtrip identity; each rung's IDEMPOTENCY and kept^2 "
                "rank probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N); each rung's span-overlap (expect "
                "~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N) per rung"},
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

    rooms_ck = save_ckpt(
        "e264_rooms",
        {RUNG_NAMES[k]: {"D_int8": rooms.rooms[RUNG_NAMES[k]].D.astype(np.int8),
                         "S": rooms.rooms[RUNG_NAMES[k]].S,
                         "k": k, "seeds": [sd, ss]}
         for k, sd, ss in LADDER},
        {"desc": "e264's middle-rung rooms (the flat basis is "
                 "net.parameters() order): one INDEPENDENT SRCT room per "
                 "deferred rung (e261's registered seeds; reconstructible "
                 "from them)",
         "ladder": [k for k, _, _ in LADDER], "n": N,
         "span_rank": rooms.r_span,
         "cert": {nm: {kk: vv for kk, vv in r.items()
                       if not isinstance(vv, list)}
                  for nm, r in cert["per_rung"].items()}})
    metrics["rooms"] = {
        "ladder": {"rule": "e261's deferred rungs VERBATIM — {10k, 40k, "
                           "100k} (the recovery protocol's cell); the "
                           "adjudication ladder adds the cited 1k/237k",
                   "rungs": [{"k": k, "name": RUNG_NAMES[k],
                              "seeds": [sd, ss],
                              "k_fraction_of_N": k / N,
                              "state": "RESUMED from e261's cut s278/400 "
                                       "vehicle" if k == 10_000 and not SMOKE
                                       else "fresh"}
                             for k, sd, ss in LADDER]},
        "seeds": {RUNG_NAMES[k]: [sd, ss] for k, sd, ss in LADDER},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{E261.SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE MIDDLE RUNGS: {' + '.join(f'{RUNG_NAMES[k]}(k={k})' for k, _, _ in LADDER)}"
        f" + FREE: BUILT + CERTIFIED")
    write_partial("P1 the middle rungs built (parents + stitch bound + v-map "
                  "loaded + span loaded + per-rung certification)")

    # ================= P2-P4: THE ARMS (FREE first — the halt gate) ======
    arms_rec: dict = {}

    def run_arm(arm: str) -> None:
        log("=" * 78)
        log(f"ARM-{arm} — "
            + ("the natural install through the instrumented path (the "
               "ceiling reference; G_FREE)" if arm == "FREE" else
               f"each install gradient PROJECTED onto an INDEPENDENT random "
               f"rank-{rooms.room_k[arm]} SRCT room (seeds "
               f"{rooms.rooms[arm].seed_d}/{rooms.rooms[arm].seed_s}) "
               f"before Adam — one rung of the ladder"
               + (" [RESUMED from e261's cut vehicle]" if arm == "K10K"
                  and not SMOKE else "")))
        inst = E261.chunked_install(
            f"{arm}-inst", arm, G1.evl_load(base_sd), rooms,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid, inst_resume_path(arm), dev)
        sd_install = inst["sd"]
        inst_net = G1.evl_load(sd_install)
        inst_cells = {"gm12": G1.battery_cell(inst_net, gm12_ids, zid)["mean_pz"],
                      "g0": G1.battery_cell(inst_net, g0_ids, zid)["mean_pz"],
                      "gp12": G1.battery_cell(inst_net, bat_ids[12], zid)["mean_pz"],
                      "ce_r": G1.ce_fixed_cpu(inst_net, *r_eval_xy)}
        d_inst = flat_params_cpu(inst_net) - base_flat
        load_inst = rooms.displacement_loads(d_inst, arm)
        del inst_net
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        led_vpre = [v["v_excess_pre"] for v in inst["ledger"].values()]
        led_vpost = [v["v_excess_post"] for v in inst["ledger"].values()]
        led_span = [v["in_span_frac"] for v in inst["ledger"].values()]
        led_aspan = [v["applied_in_span_frac"]
                     for v in inst["ledger"].values()]
        med = lambda xs: float(sorted(xs)[len(xs) // 2]) if xs else None
        arms_rec[arm] = {
            "desc": "e264 arm", "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "ledger_v_excess_pre_median": med(led_vpre),
                "ledger_v_excess_post_median": med(led_vpost),
                "ledger_in_span_frac_median": med(led_span),
                "ledger_applied_in_span_frac_median": med(led_aspan),
                "chunk_table": inst["chunk_table"],
                "steps": E261.INST_STEPS,
                "post_cells": inst_cells,
                "displacement_loads": load_inst,
                "resumed_final": bool(inst.get("resumed_final", False)),
            },
        }
        log(f"ARM-{arm} install done: post g0 {inst_cells['g0']:.6f} g-12 "
            f"{inst_cells['gm12']:.6f} CE_R {inst_cells['ce_r']:.4f} | d "
            f"v-excess {load_inst['v_excess']:.2f} cos-to-span "
            f"{load_inst['cos_to_span']:.4f}"
            + (f" in-own-room {load_inst['in_own_room']:.4f}"
               if load_inst["in_own_room"] is not None else "")
            + f" | ledger kept med {med(led_kept):.4f} v-exc pre "
            f"{med(led_vpre):.2f} post {med(led_vpost):.2f} in-span "
            f"{med(led_span):.3f}->applied {med(led_aspan):.3f}")
        write_partial(f"P2 ARM-{arm} install + post dial + measured loads")

        if not inst.get("resumed_final", False):
            burst_cooldown(f"{arm} inst->cons")
        cons = E261.chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            cons_resume_path(arm), dev)
        theta0 = cons["sd"]
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"] for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - base_flat
        load_root = rooms.displacement_loads(d_root, arm)
        root_ck = save_ckpt(
            f"e264_{arm}_root", theta0,
            {"desc": f"e264 ARM-{arm} root: e001 + Dmix s{E261.INST_STEPS} "
                     f"(gen {E261.FRESH_GEN}) + e113 cons s{E261.CONS_STEPS} "
                     f"(seed {CONS_SEED_HELD} HELD)",
             "arm": arm, "install_seed": E261.FRESH_GEN, "mode": arm,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e264_rooms.pt"})
        arms_rec[arm]["consolidation"] = {"traj": cons["traj"],
                                          "chunk_table": cons["chunk_table"]}
        arms_rec[arm]["root"] = {"cells": cells,
                                 "gm12": cells["g-12"],
                                 "g0": cells["g+0"],
                                 "displacement_loads": load_root,
                                 "checkpoint": root_ck}
        landed = (E261.G1C_ROOT_G0 * (1 - MATCH_BAND)
                  <= cells["g+0"] <= E261.G1C_ROOT_G0 * (1 + MATCH_BAND))
        arms_rec[arm]["root"]["landed"] = bool(landed)
        arms_rec[arm]["root"]["expresses"] = bool(
            inst_cells["g0"] >= G0_ZERO_FLOOR)
        log(f"ARM-{arm} ROOT: g0 {cells['g+0']:.4f} (band "
            f"[{E261.G1C_ROOT_G0 * (1 - MATCH_BAND):.4f}, "
            f"{E261.G1C_ROOT_G0 * (1 + MATCH_BAND):.4f}]) landed={landed} "
            f"(post-install g0 {inst_cells['g0']:.6f} "
            f"expresses={inst_cells['g0'] >= G0_ZERO_FLOOR}) | g-12 "
            f"{cells['g-12']:.4f} | held30 {cells['held30_gm12']:.4f} CE_R "
            f"{cells['ce_r']:.4f} | d-root v-excess "
            f"{load_root['v_excess']:.2f} cos-to-span "
            f"{load_root['cos_to_span']:.4f}"
            + (f" in-own-room {load_root['in_own_room']:.4f}"
               if load_root["in_own_room"] is not None else ""))
        del root_net_arm
        metrics["arms"] = arms_rec
        write_partial(f"P3/P4 ARM-{arm} root built + landing read")

    run_arm("FREE")

    # ---- G_FREE: THE SESSION CEILING CONTROL (tiers i-iii; iv dropped) --
    free_post = arms_rec["FREE"]["install"]["post_cells"]
    comm_inst = torch.load(CKPT_DIR / "g1c_install_resume.pt",
                           map_location="cpu", weights_only=False)["model"]
    my_inst = torch.load(inst_resume_path("FREE"), map_location="cpu",
                         weights_only=False)["model"]
    inst_l2 = float(np.sqrt(sum(float(((my_inst[k].float()
                                        - comm_inst[k].float()) ** 2).sum())
                                for k in comm_inst if k in my_inst)))
    del comm_inst, my_inst
    g1c_cons_traj = g1c["root_build"]["consolidation"]["traj"]
    my_cons_traj = arms_rec["FREE"]["consolidation"]["traj"]
    cons_diffs = [abs(a["g0_pz"] - b["g0_pz"]) for a, b in
                  zip(my_cons_traj, g1c_cons_traj)
                  if a["step"] == b["step"]]
    cons_track_median = float(sorted(cons_diffs)[len(cons_diffs) // 2]) \
        if cons_diffs else None
    G_FREE = {
        "form": ("e261's G_FREE tiers (i)-(iii) VERBATIM on THIS session's "
                 "fresh FREE re-run (the ceiling control; tier (iv), the W1 "
                 "wash, stays DROPPED — no wash in this cell): (i) the "
                 "instrumented-path install reproduces the committed "
                 "install final (L2 <= 5e-3 AND behavioral g-12 |d| <= "
                 "5e-3); (ii) FREE's root lands in the matched band; "
                 "(iii) the cons tracks the committed cons (median step "
                 "|g0 diff| <= 0.05, texture)"),
        "install_reproduction": {
            "l2_vs_committed_install_final": inst_l2,
            "behavioral_gm12": free_post["gm12"],
            "behavioral_gm12_committed": g1c_post_inst["gm12"],
            "behavioral_abs_diff": abs(free_post["gm12"]
                                       - g1c_post_inst["gm12"]),
            "bar": 5e-3,
            "pass": bool(inst_l2 < 5e-3
                         and abs(free_post["gm12"]
                                 - g1c_post_inst["gm12"]) < G_READ_TOL)},
        "root_band": {"g0": arms_rec["FREE"]["root"]["g0"],
                      "band": [E261.G1C_ROOT_G0 * (1 - MATCH_BAND),
                               E261.G1C_ROOT_G0 * (1 + MATCH_BAND)],
                      "pass": bool(arms_rec["FREE"]["root"]["landed"])},
        "cons_tracking": {"median_step_g0_abs_diff": cons_track_median,
                          "bar": 0.05,
                          "pass": bool(cons_track_median is not None
                                       and cons_track_median <= 0.05)},
        "w1_wash": None,
        "pass": bool(inst_l2 < 5e-3
                     and abs(free_post["gm12"]
                             - g1c_post_inst["gm12"]) < G_READ_TOL
                     and arms_rec["FREE"]["root"]["landed"]
                     and cons_track_median is not None
                     and cons_track_median <= 0.05),
    }
    metrics["gates"]["G_FREE"] = G_FREE
    log(f"G_FREE (tiers i-iii; iv dropped — no wash): install L2 "
        f"{inst_l2:.3e} (bar 5e-3), behavioral |d| "
        f"{abs(free_post['gm12'] - g1c_post_inst['gm12']):.1e}; root g0 "
        f"{arms_rec['FREE']['root']['g0']:.4f} in band "
        f"{arms_rec['FREE']['root']['landed']}; cons-tracking median "
        f"{cons_track_median}: "
        f"{'PASS (i-iii)' if G_FREE['pass'] else 'FAIL — THE CELL HALTS'}")
    write_partial("P4a G_FREE read (tiers i-iii)"
                  + ("" if G_FREE["pass"] or SMOKE else " — FAILED"))
    if not G_FREE["pass"] and not SMOKE:
        metrics["status"] = ("HALTED — G_FREE FAILED (the amended tiers; "
                             "nothing adjudicated)")
        write_partial("HALTED (G_FREE)")
        return 1

    # the FREE cross-session co-report (e261's FREE + the committed ceiling)
    metrics["gates"]["G_FREE_XCHECK"] = {
        "form": "THIS session's FREE root g0 vs e261's FREE root g0 vs the "
                "committed g1c root g0 (the ceiling's own cons scatter — "
                "a cross-session co-report, NOT a gate)",
        "mine": arms_rec["FREE"]["root"]["g0"],
        "e261_free": E261_FREE["root_g0"],
        "g1c_committed": E261.G1C_ROOT_G0,
        "abs_diff_vs_e261": abs(arms_rec["FREE"]["root"]["g0"]
                                - E261_FREE["root_g0"]),
        "abs_diff_vs_g1c": abs(arms_rec["FREE"]["root"]["g0"]
                               - E261.G1C_ROOT_G0),
        "pass": True,
    }
    log(f"FREE cross-session x-check: root g0 mine "
        f"{arms_rec['FREE']['root']['g0']:.4f} vs e261's "
        f"{E261_FREE['root_g0']:.4f} vs g1c committed "
        f"{E261.G1C_ROOT_G0:.4f} (co-report)")

    # ---- THE RUNGS (K10K first — the resume; then ascending) -------------
    burst_cooldown("FREE -> the middle rungs")
    for k, _, _ in LADDER:
        run_arm(RUNG_NAMES[k])
        if RUNG_NAMES[k] != ARMS[-1]:
            burst_cooldown(f"{RUNG_NAMES[k]} -> next rung")

    # ---- G_K10KRESUME post-read (the resume verifiably continued) --------
    if not SMOKE:
        k10k_traj_steps = [t["step"] for t in arms_rec["K10K"]["install"]["traj"]]
        k10k_ledger_max = max(int(kk) for kk in
                              arms_rec["K10K"]["install"]["ledger"].keys())
        post = {
            "traj_steps": k10k_traj_steps,
            "expected_traj_steps": K10K_TRAJ_STEPS + [300, 400],
            "ledger_max_key": k10k_ledger_max,
            "expected_ledger_max": 400,
            "vehicle_step_now": int(torch.load(
                k10k_path, map_location="cpu", weights_only=False)["step"]),
            "steps_ran": arms_rec["K10K"]["install"]["steps"],
        }
        post["pass"] = bool(k10k_traj_steps == K10K_TRAJ_STEPS + [300, 400]
                            and k10k_ledger_max == 400
                            and post["vehicle_step_now"] == 400)
        metrics["gates"]["G_K10KRESUME"]["post"] = post
        log(f"G_K10KRESUME post-read: traj {k10k_traj_steps}, ledger to "
            f"s{k10k_ledger_max}, vehicle now s{post['vehicle_step_now']}: "
            f"{'the resume verifiably continued the cut' if post['pass'] else 'MISMATCH'}")
        write_partial("P4b G_K10KRESUME post-read (the resume continued)")

    # ================= P5: THE CONS-SEED REPLICATE (the scatter priced) ==
    cons2_rec = None
    if not SMOKE:
        log("=" * 78)
        log("P5 THE CONS-SEED REPLICATE — e261's committed K237K "
            "install-final + a fresh cons at seed "
            f"{CONS2_SEED} (the anchor scatter priced; a CO-REPORT, never "
            "a rung)")
        k237k_inst_sd = torch.load(CKPT_DIR / "e261_K237K_inst_resume.pt",
                                   map_location="cpu",
                                   weights_only=False)["model"]
        E261.CONS_SEED = CONS2_SEED            # the ONE draw difference
        cons2 = E261.chunked_consolidate(
            "K237KCONS2-cons", G1.evl_load(k237k_inst_sd), pool_a_x,
            pool_a_mask, cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / "e264_K237KCONS2_cons_resume.pt", dev)
        E261.CONS_SEED = CONS_SEED_HELD        # restored (nothing downstream)
        c2_net = G1.evl_load(cons2["sd"])
        c2_cells = {f"g{j:+d}": G1.battery_cell(c2_net, bat_ids[j],
                                                zid)["mean_pz"]
                    for j in G1.GEOS}
        c2_cells["ce_r"] = G1.ce_fixed_cpu(c2_net, *r_eval_xy)
        # the 237k room reconstructed standalone (for the in-own-room read)
        room237 = E261.SRCT(N, 237_123, 26_011, 26_012)
        d_c2 = flat_params_cpu(c2_net) - base_flat
        d64 = d_c2.double().numpy().astype(np.float64)
        pd = room237.project(d64)
        c2_in_own = math.sqrt(float((pd @ d64) / (d64 @ d64)))
        # the room's bit-identity vs e260's committed artifact (e261's gate)
        rooms260 = torch.load(CKPT_DIR / E261.ROOMS260_CK, map_location="cpu",
                              weights_only=False)

        def _to_np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        D260 = _to_np(rooms260["model"]["D_rand_int8"]).astype(np.float64)
        S260 = _to_np(rooms260["model"]["S_rand"])
        del rooms260, c2_net
        cons2_rec = {
            "desc": f"e261's committed K237K install-final (its ckpt's "
                    f"model, bit-as-saved) + e113-cons-VERBATIM arithmetic "
                    f"at seed {CONS2_SEED} (the next seed; the ONE delta vs "
                    f"the rung's cons is the draw) — prices the anchor "
                    f"scatter (co-report, never a rung)",
            "cons_seed": CONS2_SEED, "steps": E261.CONS_STEPS,
            "traj": cons2["traj"], "chunk_table": cons2["chunk_table"],
            "root": {"g0": c2_cells["g+0"], "gm12": c2_cells["g-12"],
                     "ce_r": c2_cells["ce_r"],
                     "in_own_room": c2_in_own},
            "checkpoint": save_ckpt(
                "e264_K237KCONS2_root", cons2["sd"],
                {"desc": "the K237K cons-seed replicate root (seed "
                         f"{CONS2_SEED}): e261's committed install-final + "
                         "a fresh cons draw",
                 "cons_seed": CONS2_SEED, "install_source":
                     "runs/checkpoints/e261_K237K_inst_resume.pt"}),
            "reads_vs": {
                "e261_rung_root_g0": E261_K237K["root_g0"],
                "abs_diff_root_g0": abs(c2_cells["g+0"]
                                        - E261_K237K["root_g0"]),
                "e260_committed_root_g0": E260_RANDOM["root_g0"],
                "abs_diff_vs_e260": abs(c2_cells["g+0"]
                                        - E260_RANDOM["root_g0"]),
                "e261_rung_root_gm12": E261_K237K["root_gm12"],
                "abs_diff_root_gm12": abs(c2_cells["g-12"]
                                          - E261_K237K["root_gm12"]),
                "anchor_scatter_cross_session": {
                    "root_g0": ANCHOR_SCATTER_G0,
                    "root_gm12": ANCHOR_SCATTER_GM12}},
            "room237_bit_identity": {
                "D_bit_equal": bool(np.array_equal(room237.D, D260)),
                "S_bit_equal": bool(np.array_equal(room237.S, S260)),
                "pass": bool(np.array_equal(room237.D, D260)
                             and np.array_equal(room237.S, S260))},
        }
        metrics["cons_seed_replicate"] = cons2_rec
        log(f"P5 THE CONS-SEED REPLICATE: root g0 {c2_cells['g+0']:.4f} "
            f"(e261's rung {E261_K237K['root_g0']:.4f} |d| "
            f"{abs(c2_cells['g+0'] - E261_K237K['root_g0']):.4f}; e260's "
            f"committed {E260_RANDOM['root_g0']:.4f} |d| "
            f"{abs(c2_cells['g+0'] - E260_RANDOM['root_g0']):.4f}) | g-12 "
            f"{c2_cells['g-12']:.4f} (|d| "
            f"{abs(c2_cells['g-12'] - E261_K237K['root_gm12']):.4f}) | "
            f"in-own-room {c2_in_own:.4f} | the room bit-identical to "
            f"e260_rooms.pt: {cons2_rec['room237_bit_identity']['pass']}")
        write_partial("P5 the cons-seed replicate read (the anchor scatter "
                      "priced)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    band_lo_g0 = E261.G1C_ROOT_G0 * (1 - MATCH_BAND)
    band_hi_g0 = E261.G1C_ROOT_G0 * (1 + MATCH_BAND)
    # THE FULL 5-RUNG CURVE (the stitch: cited + fresh)
    src_map = {1000: "e261 (committed)", 10000: "e264 (K10K resume)",
               40000: "e264 (fresh)", 100000: "e264 (fresh)",
               237123: "e261 (committed)"}

    def post_g0_of(k: int) -> float:
        if k == 1_000:
            return E261_K1K["post_g0"]
        if k == 237_123:
            return E261_K237K["post_g0"]
        return arms_rec[RUNG_NAMES[k]]["install"]["post_cells"]["g0"]

    def root_g0_of(k: int) -> float:
        if k == 1_000:
            return E261_K1K["root_g0"]
        if k == 237_123:
            return E261_K237K["root_g0"]
        return arms_rec[RUNG_NAMES[k]]["root"]["g0"]

    def root_gm12_of(k: int) -> float:
        if k == 1_000:
            return E261_K1K["root_gm12"]
        if k == 237_123:
            return E261_K237K["root_gm12"]
        return arms_rec[RUNG_NAMES[k]]["root"]["gm12"]

    def kept_of(k: int) -> float:
        if k == 1_000:
            return E261_K1K["kept"]
        if k == 237_123:
            return E261_K237K["kept"]
        return arms_rec[RUNG_NAMES[k]]["install"]["ledger_kept_frac_median"]

    ladder_ks = list(FULL_LADDER_KS) if not SMOKE else \
        [k for k, _, _ in LADDER]
    pg = [post_g0_of(k) for k in ladder_ks]
    rg = [root_g0_of(k) for k in ladder_ks]
    kept_curve = {str(k): kept_of(k) for k in ladder_ks}
    ratios = [{"from_k": ladder_ks[i], "to_k": ladder_ks[i + 1],
               "ratio": pg[i + 1] / max(pg[i], RATIO_DEN_FLOOR)}
              for i in range(len(ladder_ks) - 1)]
    max_ratio_row = max(ratios, key=lambda r: r["ratio"])
    max_ratio = max_ratio_row["ratio"]
    jump_fires = bool(max_ratio > JUMP_BAR)
    floor_cross = next((ladder_ks[i] for i in range(len(ladder_ks))
                        if pg[i] >= G0_ZERO_FLOOR), None)
    in_band = [k for k, v in zip(ladder_ks, rg) if band_lo_g0 <= v <= band_hi_g0]
    over_band = [k for k, v in zip(ladder_ks, rg) if v > band_hi_g0]
    under_all = [k for k, v in zip(ladder_ks, rg) if v < band_lo_g0]
    threshold_k = min(in_band) if in_band else None
    bracket = None
    if in_band:
        i_in = ladder_ks.index(min(in_band))
        bracket = [ladder_ks[i_in - 1] if i_in > 0 else None, min(in_band)]
    top_root_g0 = rg[-1]
    approaches = bool(top_root_g0 >= band_lo_g0 - GRADUAL_REACH)
    # THE SCATTER-FRAGILE FLAGS (the cons-scatter disclosure, mechanical)
    fragile = {str(k): bool(min(abs(v - band_lo_g0), abs(v - band_hi_g0))
                            < ANCHOR_SCATTER_G0)
               for k, v in zip(ladder_ks, rg)}
    edge_of = {str(k): ("floor" if abs(v - band_lo_g0)
                        <= abs(v - band_hi_g0) else "ceiling")
               for k, v in zip(ladder_ks, rg)}

    hard = {k: v for k, v in metrics["gates"].items()}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))
    sharp_fires = bool(gates_pass and jump_fires and bool(in_band))
    gradual_fires = bool(gates_pass and not jump_fires and not in_band
                         and approaches)

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif sharp_fires:
        verdict = "SHARP-THRESHOLD"
        clause = (f"the expression curve is step-like (post g0 jumps "
                  f"{max_ratio:.1f}x between k={max_ratio_row['from_k']} and "
                  f"k={max_ratio_row['to_k']}; the floor {G0_ZERO_FLOOR} "
                  f"first crossed at "
                  f"{f'k={floor_cross}' if floor_cross else 'no rung'}) AND "
                  f"the landing curve enters the band at a locatable rung "
                  f"(first in-band k={threshold_k}, bracket "
                  f"{bracket[0]}->{bracket[1]}) — the anti-substrate's "
                  f"final form: a dimensional threshold at k ~ "
                  f"{threshold_k}; memories need k* dimensions, full stop")
    elif gradual_fires:
        verdict = "GRADUAL"
        clause = (f"both curves rise smoothly (no rung-to-rung jump > "
                  f"{JUMP_BAR:.0f}x in expression — max ratio "
                  f"{max_ratio:.2f}x at k={max_ratio_row['from_k']}-"
                  f"{max_ratio_row['to_k']}; the floor first crossed at "
                  f"{f'k=' + str(floor_cross) if floor_cross else 'no rung'}) "
                  f"and the landing approaches the band asymptotically (no "
                  f"in-band rung; the top rung's root g0 {top_root_g0:.4f} "
                  f"sits {band_lo_g0 - top_root_g0:.4f} under the floor "
                  f"{band_lo_g0:.4f}, within the {GRADUAL_REACH} reach) — "
                  f"the barrier is a soft capacity gradient, not a "
                  f"threshold; the last 0.08 of root g0 is spread across "
                  f"the ladder")
    else:
        why = []
        if jump_fires and not in_band:
            why.append(f"expression step-like (max jump {max_ratio:.1f}x at "
                       f"k={max_ratio_row['from_k']}-"
                       f"{max_ratio_row['to_k']}) but the landing NEVER "
                       f"enters the band (best rung root g0 "
                       f"{max(rg):.4f} vs floor {band_lo_g0:.4f}; "
                       f"{len(under_all)}/{len(ladder_ks)} rungs under)")
        if not jump_fires and in_band:
            why.append(f"expression smooth (max ratio {max_ratio:.2f}x) but "
                       f"the landing enters the band at a locatable rung "
                       f"(first in-band k={threshold_k})")
        if not jump_fires and not in_band and not approaches:
            why.append(f"the ladder never approaches the band in this "
                       f"construction (top rung root g0 {top_root_g0:.4f} "
                       f"sits {band_lo_g0 - top_root_g0:.4f} under the "
                       f"floor, beyond the {GRADUAL_REACH} reach) — the "
                       f"last 0.08 does not live on the k axis as mapped")
        if not why:
            why.append("between the bars")
        verdict = "MIXED"
        clause = "; ".join(why) + " — the curves verbatim, mapped honestly"

    # THE STITCH SENTENCE (the cons-scatter disclosure, carried verbatim)
    if not SMOKE:
        frag_ks = [k for k in ladder_ks if fragile[str(k)]]
        frag_txt = (", ".join(f"k={k} (near its {edge_of[str(k)]})"
                             for k in frag_ks) if frag_ks else "none")
        repl_txt = ""
        if cons2_rec is not None:
            repl_txt = (f"; the K237K cons-seed replicate (seed "
                        f"{CONS2_SEED}) read root g0 "
                        f"{cons2_rec['root']['g0']:.4f} — |d| "
                        f"{cons2_rec['reads_vs']['abs_diff_root_g0']:.4f} "
                        f"vs the cited rung, |d| "
                        f"{cons2_rec['reads_vs']['abs_diff_vs_e260']:.4f} "
                        f"vs e260's committed")
        clause = ("THE STITCH: rungs 1k/237k cited from e261's committed "
                  "record (md5-bound; post g0 stitches cleanly — the "
                  "install is deterministic; root reads carry the cons "
                  f"lottery's scatter {ANCHOR_SCATTER_G0:.4f} g0 / "
                  f"{ANCHOR_SCATTER_GM12:.4f} g-12 cross-session on a "
                  "bit-identical arm): scatter-fragile calls: "
                  f"{frag_txt}{repl_txt}. " + clause)

    log("=" * 78)
    log(f"E264 VERDICT: {verdict}")
    log("  THE FULL 5-RUNG LADDER (post g0 -> root g0 | kept | in-band):")
    for k, p_, r_ in zip(ladder_ks, pg, rg):
        mark = "IN-BAND" if band_lo_g0 <= r_ <= band_hi_g0 else (
            "OVER" if r_ > band_hi_g0 else "under")
        flag = " [scatter-fragile]" if fragile[str(k)] else ""
        log(f"    k={k:7d}: post g0 {p_:.6f} -> root g0 {r_:.4f} | kept "
            f"{kept_curve[str(k)]:.4f} | {mark}{flag} "
            f"[{src_map.get(k, 'e264') if not SMOKE else 'smoke'}]")
    log(f"    FREE (full space): post g0 "
        f"{arms_rec['FREE']['install']['post_cells']['g0']:.4f} -> root "
        f"g0 {arms_rec['FREE']['root']['g0']:.4f} | kept 1.0 | the ceiling "
        f"(e261's FREE: root {E261_FREE['root_g0']:.4f}; g1c committed "
        f"{E261.G1C_ROOT_G0:.4f})")
    log(f"  max adjacent expression ratio {max_ratio:.2f}x "
        f"(k={max_ratio_row['from_k']}->{max_ratio_row['to_k']}; bar "
        f">{JUMP_BAR:.0f}x -> {'FIRES' if jump_fires else 'does not fire'})")
    log(f"  floor {G0_ZERO_FLOOR} first crossed at "
        f"{floor_cross if floor_cross else 'no rung'}; in-band rungs "
        f"{in_band if in_band else 'NONE'} (threshold k "
        f"{threshold_k if threshold_k else 'none'}; bracket {bracket})")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> SHARP-THRESHOLD -> GRADUAL -> MIXED "
                           "(frozen; e261's original)",
        "gates_pass": gates_pass,
        "reads": {
            "ladder_ks": ladder_ks,
            "rung_sources": {str(k): src_map[k] for k in ladder_ks},
            "post_g0_curve": {str(k): post_g0_of(k) for k in ladder_ks},
            "root_g0_curve": {str(k): root_g0_of(k) for k in ladder_ks},
            "root_gm12_curve": {str(k): root_gm12_of(k) for k in ladder_ks},
            "kept_frac_curve": kept_curve,
            "FREE_this_session": {
                "post_g0": arms_rec["FREE"]["install"]["post_cells"]["g0"],
                "root_g0": arms_rec["FREE"]["root"]["g0"],
                "root_gm12": arms_rec["FREE"]["root"]["gm12"]},
            "FREE_e261_cited": E261_FREE,
            "FREE_g1c_committed_root_g0": E261.G1C_ROOT_G0,
            "e246_context": {"rank": E261.E246_ALIGNED_RANK,
                             "post_g0": E261.E246_ALIGNED_POST_G0},
            "e260_committed_RANDOM_at_237k": E260_RANDOM,
            "adjacent_ratios": ratios,
            "max_ratio": max_ratio, "max_ratio_row": max_ratio_row,
            "jump_fires": jump_fires, "jump_bar": JUMP_BAR,
            "floor_cross_k": floor_cross, "expression_floor": G0_ZERO_FLOOR,
            "in_band_rungs": in_band, "over_band_rungs": over_band,
            "under_band_rungs": under_all,
            "threshold_k": threshold_k, "threshold_bracket": bracket,
            "top_rung_root_g0": top_root_g0,
            "approaches_band": approaches, "reach": GRADUAL_REACH,
            "matched_band_g0": [band_lo_g0, band_hi_g0],
            "scatter_fragile": fragile,
            "scatter_edge": edge_of,
            "scatter_bar_g0": ANCHOR_SCATTER_G0,
            "landed": {a: arms_rec[a]["root"]["landed"] for a in ARMS},
            "expresses": {a: arms_rec[a]["root"]["expresses"] for a in ARMS},
        },
        "SHARP_THRESHOLD": sharp_fires,
        "GRADUAL": gradual_fires,
        "MIXED": bool(gates_pass and not sharp_fires and not gradual_fires),
        "the_stitch": None if SMOKE else {
            "cited_rungs": {"1000": E261_K1K, "237123": E261_K237K},
            "source_md5": E261_MD5,
            "post_g0_stitch_rationale": "install-deterministic (e261's "
                "G_ANCHOR: install L2 6.7e-5, |d post g0| 5e-7 cross-session)",
            "root_g0_scatter": {"cross_session_anchor":
                                 ANCHOR_SCATTER_G0,
                                "cons_seed_replicate":
                                 cons2_rec["reads_vs"]["abs_diff_root_g0"]
                                 if cons2_rec else None},
        },
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars adjudicated (the full 5-rung curve)")

    # ================= P8: the figures ====================================
    make_ladder_plot(RD, arms_rec, ladder_ks, pg, rg, kept_curve, verdict,
                     clause, in_band, threshold_k, max_ratio_row, jump_fires,
                     fragile)
    make_instrument_plot(RD, rooms, arms_rec, cert, verdict, ladder_ks)

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("all arms share bit-identical install "
            "streams (one generator, seed 24314, one draw order), identical "
            "dose/schedule/optimizer, the same fresh fact-free base; the "
            "ONLY delta across rungs is the pre-Adam gradient PROJECTION "
            "onto that rung's independent random rank-k room (each room "
            "certified: idempotency ~1e-15, kept^2 == k/N at the 10-sigma "
            "bar); K10K's install CONTINUED bit-exactly from e261's cut "
            "vehicle (model + optimizer + generator state + ledger, "
            "md5/step-gated) — fate differences across rungs are "
            "rank-caused or nothing is"),
        "n_and_scope": ("n=1 per rung, one lineage, TWO sessions stitched "
            "(e261: rungs 1k/237k + FREE-context; e264: the middle rungs + "
            "FREE; the g-series standing lottery caveat carried verbatim); "
            "the curve's SHAPE is the registered object, not any single "
            "point; the bracket between adjacent rungs is the honest "
            "resolution of the threshold's location"),
        "the_stitch_disclosed": ("rungs 1k/237k are e261's committed reads "
            "(md5-bound): the post-install curve stitches cleanly (the "
            "install is deterministic; e261's G_ANCHOR install L2 6.7e-5, "
            "|d post g0| 5e-7 cross-session), the landing curve carries "
            "the cons lottery's cross-session scatter (root g0 |d| "
            "0.0419, root g-12 |d| 0.1483 on a bit-identical arm) — "
            "per-rung scatter-fragile flags mark every call within that "
            "band of an edge, and the cons-seed replicate at 237k prices "
            "it directly"),
        "loads_measured_not_nominal": ("every rung's ACTUAL geometry is "
            "reported: the per-step kept fraction (the dose actually "
            "delivered; median over the install ledger), v-excess pre/post "
            "(vs e258's LOADED committed v-map), in-span fraction pre AND "
            "applied (vs e246's committed LATE span), and the displacement "
            "cos-to-span + in-own-room fractions at post-install and root "
            "— never nominal"),
        "the_confound_disclosed": ("the kept-k coupling: the projection's "
            "norm is NOT rescaled, so the delivered dose covaries with the "
            "rung (kept ~ sqrt(k/N)) — an inherent property of a rank-k "
            "restriction, disclosed on every rung and co-plotted on the "
            "instrument page (root g0 vs kept); e258's committed VLIGHT "
            "LANDED at kept ~0.10, so kept alone does not bar landing"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all three branches and "
            "the curves are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_ledger": f"runs/checkpoints/{E261.SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "k10k_resume_vehicle": {
                "file": f"runs/checkpoints/{K10K_CK}",
                "md5_at_registration": K10K_MD5,
                "md5_at_load": metrics["gates"]["G_K10KRESUME"].get("md5"),
                "note": "overwritten by the resume to the s400 final (the "
                        "journal-resume path's design; the cut preserved "
                        "by this frozen md5)"},
            "rooms": metrics["rooms"]["checkpoint"],
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"] for a in ARMS},
            "cons_seed_replicate_root":
                cons2_rec["checkpoint"] if cons2_rec else None,
        },
        "machinery": {
            "install_cons": "e261's chunked drivers VERBATIM BY IMPORT "
                            "(E43.exposure Dmix / e113 consolidate; the "
                            "dense-room hook + the thermal envelope); the "
                            "committed lab/e261_rank_ladder.py unmodified",
            "hook": "e237's pre-Adam projection as a DENSE ROOM projector "
                    "(backward -> clip 1.0 -> project (CPU fp64 SRCT; "
                    "write fp32) -> step; norm not rescaled; fp64 ledger "
                    "dots)",
            "rooms": "one INDEPENDENT SRCT room per rung (e261's "
                     "registered deferred seeds 26113-26118)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": {
            "burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
            "per_step_polls": True,
            "early_end_margin_c": E261.TEMP_EARLY_END,
            "hard_line_c": E261.TEMP_HARD,
            "max_temp_seen_c": max((r["temp"] for r in E261.thermal_log),
                                   default=None),
            "violations_ge_84c": sum(1 for r in E261.thermal_log
                                     if r["temp"] >= E261.TEMP_HARD),
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e264_middle_rungs.png"),
                          str(RD / "e264_instrument.png")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_ladder_plot(rd, arms_rec, ladder_ks, pg, rg, kept_curve, verdict,
                     clause, in_band, threshold_k, max_ratio_row, jump_fires,
                     fragile):
    """THE CURVES FIGURE: the full 5-rung expression + landing curves (the
    stitch marked), the dose covariate (kept vs k; root g0 vs kept), and
    the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    N = GB.G1B_PARAMS
    fresh_ks = [k for k in ladder_ks
                if SMOKE or k not in (1_000, 237_123)]
    cited_ks = [k for k in ladder_ks if not SMOKE and k in (1_000, 237_123)]
    cols = plt.cm.viridis(np.linspace(0.25, 0.95, max(len(ladder_ks), 1)))

    # (0,0) THE EXPRESSION CURVE (post-install g0 vs k)
    ax = axes[0, 0]
    ax.plot(ladder_ks, pg, "o-", ms=8, lw=2.0, color="#1a6faf", alpha=0.95,
            label="the 5 rungs (post-install g0)")
    for k, v, c in zip(ladder_ks, pg, cols):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=c)
    for k in cited_ks:
        i = ladder_ks.index(k)
        ax.plot([k], [pg[i]], "o", ms=13, mfc="none", mec="k", mew=1.6,
                label="cited from e261's committed record" if k == cited_ks[0]
                else None)
    ax.plot([N], [post_g0_free := arms_rec["FREE"]["install"]["post_cells"]
                  ["g0"]], "*", ms=17, color="#e67e22",
            label=f"FREE this session (k=N: post g0 {post_g0_free:.3f})")
    ax.plot([E261.E246_ALIGNED_RANK], [E261.E246_ALIGNED_POST_G0], "x",
            ms=9, mew=2.2, color="darkred",
            label=f"e246 ALIGNED (rank 10, post g0 "
                  f"{E261.E246_ALIGNED_POST_G0:.1e}) — dead context")
    if not SMOKE:
        ax.plot([237_123], [E260_RANDOM["post_g0"]], "s", ms=10, mfc="none",
                mec="dimgray", mew=1.5,
                label=f"e260 committed RANDOM (post g0 "
                      f"{E260_RANDOM['post_g0']:.3f})")
    ax.axhline(G0_ZERO_FLOOR, ls=":", lw=1.4, color="crimson",
               label=f"the expression floor ({G0_ZERO_FLOOR})")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log; SRCT random dense rooms)")
    ax.set_ylabel("post-install g0 (the expression ruler)")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25, which="both")
    ax.set_title(f"THE EXPRESSION CURVE (completed) — max adjacent jump "
                 f"{max_ratio_row['ratio']:.1f}x (k="
                 f"{max_ratio_row['from_k']}->{max_ratio_row['to_k']}; the "
                 f">10x bar {'FIRES' if jump_fires else 'does not fire'})",
                 fontsize=9.5)

    # (0,1) THE LANDING CURVE (root g0 vs k, the band + the threshold)
    ax = axes[0, 1]
    ax.plot(ladder_ks, rg, "o-", ms=8, lw=2.0, color="#1a6faf", alpha=0.95,
            label="the 5 rungs (root g0)")
    for k, v, c in zip(ladder_ks, rg, cols):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=c)
    for k in cited_ks:
        i = ladder_ks.index(k)
        ax.plot([k], [rg[i]], "o", ms=13, mfc="none", mec="k", mew=1.6,
                label="cited from e261" if k == cited_ks[0] else None)
    ax.plot([N], [root_g0_free := arms_rec["FREE"]["root"]["g0"]], "*",
            ms=17, color="#e67e22",
            label=f"FREE this session (root g0 {root_g0_free:.3f}) — the "
                  f"ceiling")
    if not SMOKE:
        ax.plot([237_123], [E260_RANDOM["root_g0"]], "s", ms=10, mfc="none",
                mec="dimgray", mew=1.5,
                label=f"e260 committed RANDOM (root g0 "
                      f"{E260_RANDOM['root_g0']:.3f}) — the scatter's "
                      f"other read")
    ax.axhspan(E261.G1C_ROOT_G0 * (1 - MATCH_BAND),
               E261.G1C_ROOT_G0 * (1 + MATCH_BAND),
               color="#b8d8f0", alpha=0.35, zorder=0,
               label=f"the matched band (+-{MATCH_BAND:.0%} of "
                     f"{E261.G1C_ROOT_G0:.3f})")
    ax.axhline(E261.G1C_ROOT_G0, color="k", ls=":", lw=0.9)
    # the cons-scatter band around the floor (the disclosure, drawn)
    ax.axhspan(band_lo := E261.G1C_ROOT_G0 * (1 - MATCH_BAND),
               band_lo + ANCHOR_SCATTER_G0, color="crimson", alpha=0.10,
               zorder=0, label=f"the cons-scatter reach (+"
                               f"{ANCHOR_SCATTER_G0:.4f} above the floor)")
    if threshold_k is not None:
        ax.axvline(threshold_k, color="seagreen", ls="--", lw=1.8, alpha=0.8)
        ax.annotate(f"THRESHOLD\nk={threshold_k}", (threshold_k, 0.06),
                    fontsize=8, color="seagreen", ha="right", weight="bold")
    else:
        ax.annotate("no rung enters the band", (0.97, 0.05),
                    xycoords="axes fraction", fontsize=8, color="crimson",
                    ha="right", style="italic")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("ROOT g0 (post install + cons)")
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25, which="both")
    ax.set_title("THE LANDING CURVE (completed) — in-band rungs: "
                 f"{in_band if in_band else 'NONE'}; crimson veil = the "
                 "scatter-fragile zone", fontsize=9.5)

    # (1,0) THE DOSE COVARIATE (kept vs k + root g0 vs kept — the confound)
    ax = axes[1, 0]
    kpts = [kept_curve[str(k)] for k in ladder_ks]
    exp_pts = [math.sqrt(k / N) for k in ladder_ks]
    ax.plot(ladder_ks, kpts, "o-", ms=7, lw=1.8, color="#8e44ad",
            label="kept ||g'||/||g|| (ledger median; cited where e261)")
    ax.plot(ladder_ks, exp_pts, "s--", ms=5, lw=1.2, color="dimgray",
            label="the sqrt(k/N) expectation")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("kept fraction (the delivered dose)", color="#8e44ad")
    ax.tick_params(axis="y", colors="#8e44ad")
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25, which="both")
    ax2 = ax.twiny()
    ax2.plot(kpts, rg, "^", ms=9, color="#c0392b",
             label="root g0 vs kept (the confound view)")
    for kk, vv, k_ in zip(kpts, rg, ladder_ks):
        ax2.annotate(f"k={k_}", (kk, vv), textcoords="offset points",
                     xytext=(6, 4), fontsize=7, color="#c0392b")
    ax2.set_xlabel("kept fraction (the confound's own axis)", color="#c0392b")
    ax2.tick_params(axis="x", colors="#c0392b")
    ax2.legend(fontsize=7.2, loc="lower right")
    ax.set_title("THE KEPT-K COUPLING (dose honesty: kept ~ sqrt(k/N) over "
                 "the full ladder)", fontsize=9.5)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E264 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=94, break_long_words=False)[:14]:
        ax.text(0.02, y, wd, fontsize=6.8, va="top", family="monospace")
        y -= 0.024
    y -= 0.012
    gates_txt = "  ".join(
        f"{g}={'PASS' if v.get('pass') else 'FAIL'}"
        for g, v in metrics["gates"].items() if isinstance(v, dict))
    for wd in textwrap.wrap("GATES: " + gates_txt, width=96)[:2]:
        ax.text(0.02, y, wd, fontsize=6.4, va="top", family="monospace")
        y -= 0.02
    fig.suptitle("E264 — THE MIDDLE RUNGS: the completed 5-rung ladder "
                 f"{{{', '.join(str(k) for k in ladder_ks)}}} + FREE (full "
                 f"space) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e264_middle_rungs.png", dpi=130)
    plt.close(fig)


def make_instrument_plot(rd, rooms: "E261.LadderRooms", arms_rec, cert,
                         verdict, ladder_ks):
    """THE INSTRUMENT FIGURE: the per-rung certification, the kept ledgers,
    the measured per-arm loads, and the thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    N = rooms.n
    cols = plt.cm.viridis(np.linspace(0.25, 0.95, max(len(ladder_ks), 1)))

    # (0,0) THE ROOMS' CERTIFICATION
    ax = axes[0, 0]
    names = [RUNG_NAMES[k] for k, _, _ in LADDER]
    idem = [cert["per_rung"][nm]["idempotency_max"] for nm in names]
    kept_dev = [abs(cert["per_rung"][nm]["kept2_mean"]
                    - cert["per_rung"][nm]["kept2_expect"]) for nm in names]
    bars_ = [max(v, 1e-18) for v in idem]
    ax.bar([f"{n}\nidem" for n in names], bars_, color=cols, alpha=0.85)
    for b, v in zip(ax.patches, idem):
        ax.annotate(f"{v:.0e}", (b.get_x() + b.get_width() / 2,
                                 max(v, 1e-18)), ha="center", va="bottom",
                    fontsize=6.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-2)
    ax.axhline(1e-8, ls=":", color="k", lw=1.0, label="the idem bar 1e-8")
    ax.set_ylabel("max relative deviation (log)")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y", which="both")
    kt = " | ".join(
        f"{nm}: kept2 {cert['per_rung'][nm]['kept2_mean']:.2e} vs "
        f"{cert['per_rung'][nm]['kept2_expect']:.2e} (d {d:.0e})"
        for nm, d in zip(names, kept_dev))
    ax.set_title("THE MIDDLE RUNGS' CERTIFICATION (idem shown; " + kt + ")",
                 fontsize=7.6)

    # (0,1) THE KEPT-FRACTION LEDGERS
    ax = axes[0, 1]
    for a in ARMS:
        led = arms_rec[a]["install"]["ledger"]
        xs_l = sorted(int(s) for s in led)
        get = lambda s: led[s] if s in led else led[str(s)]
        ax.plot(xs_l, [get(s)["kept_frac"] for s in xs_l], "o-", ms=3.2,
                lw=1.3, alpha=0.9,
                color="#e67e22" if a == "FREE" else cols[
                    [RUNG_NAMES[k] for k, _, _ in LADDER].index(a)],
                label=f"ARM-{a} ||g'||/||g|| (med "
                      f"{arms_rec[a]['install']['ledger_kept_frac_median']:.3f})")
    for k, c in zip([kk for kk, _, _ in LADDER], cols):
        ax.axhline(math.sqrt(k / N), ls=":", lw=0.8, color=c, alpha=0.5)
    ax.set_xlabel("install step")
    ax.set_ylabel("the kept-gradient fraction (dose honesty)")
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25)
    ax.set_title("THE KEPT-FRACTION LEDGERS (dotted: each rung's sqrt(k/N) "
                 "expectation)", fontsize=9.5)

    # (1,0) THE MEASURED PER-ARM LOADS
    ax = axes[1, 0]
    names_a = list(ARMS)
    w = 0.2
    xs = np.arange(len(names_a))
    vpost = [arms_rec[a]["install"]["ledger_v_excess_post_median"] or 0
             for a in names_a]
    vpre = [arms_rec[a]["install"]["ledger_v_excess_pre_median"] or 0
            for a in names_a]
    ior = [arms_rec[a]["root"]["displacement_loads"]["in_own_room"] or 0
           for a in names_a]
    cts = [arms_rec[a]["root"]["displacement_loads"]["cos_to_span"]
           for a in names_a]
    ax.bar(xs - w, vpre, width=w, color="#7fb3d5", alpha=0.9,
           label="install |g| v-excess (pre)")
    ax.bar(xs, vpost, width=w, color="#1a6faf", alpha=0.9,
           label="install |g'| v-excess (applied)")
    ax.bar(xs + w, ior, width=w, color="#8e44ad", alpha=0.8,
           label="root displacement in-own-room")
    ax.bar(xs + 2 * w, cts, width=w, color="#c0392b", alpha=0.55,
           label="root displacement cos-to-span")
    ax.set_xticks(xs)
    ax.set_xticklabels(names_a, fontsize=8, rotation=20)
    ax.set_ylabel("measured load / fraction")
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE MEASURED PER-ARM LOADS (never nominal; v-map loaded "
                 "from e258; span from e246)", fontsize=9.5)

    # (1,1) THE THERMAL ENVELOPE
    ax = axes[1, 1]
    tl = E261.thermal_log
    if tl:
        ax.plot([r["t"] for r in tl], [r["temp"] for r in tl],
                "-", lw=0.8, color="dimgray", alpha=0.7)
    ax.axhline(E261.TEMP_EARLY_END, color="crimson", ls=":", lw=1.0,
               label=f"burst-end margin {E261.TEMP_EARLY_END:.0f}C")
    ax.axhline(E261.TEMP_HARD, color="crimson", ls="--", lw=1.2,
               label=f"never-past line {E261.TEMP_HARD:.0f}C")
    ax.set_xlabel("run seconds")
    ax.set_ylabel("GPU temp (C) per-step polls")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in tl), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in tl if r['temp'] >= E261.TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle("E264 — the instrument page: the middle rungs' rooms, the "
                 f"certification, the kept ledgers, the measured loads "
                 f"-> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e264_instrument.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
