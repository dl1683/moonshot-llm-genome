"""E261 — THE RANK/DOSE LADDER (T237's registered ask: "where does the
last ~0.08 of root g0 live between k=237k and full space?"). Design
dispatched 2026-10-05; this docstring carries the registered bars
VERBATIM, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): the expression barrier lives
somewhere between rank 10 (dead: e246's ALIGNED, post g0 2.86e-5) and
rank 237,123 (within 0.2% of the band's floor: e260's RANDOM arm, root
g0 0.6686 vs floor 0.6703; FREE 0.7322). WHERE? The landing curve
root_g0(k) over the ladder maps the threshold — the anti-substrate's
final quantitative form.

THE CELL (2.74M, the g1c conventions; e260's cell shape, PORTED WHOLE):
  1. THE LADDER: random-room installs (the SRCT projector at e260's
     certified construction) at k in a registered geometric-ish ladder —
     {1k, 10k, 40k, 100k, 237k(matched-committed)} (5 rungs; each an
     independent install from the same fresh root at the SAME dose —
     the dose fixed at e260's; the seeds registered);
  2. the readouts per rung: post-install g0 (expression) + root g0
     (landing) + kept-fraction (the dose actually delivered);
  3. ARM-FREE at the same dose (the ceiling reference; must reproduce
     the committed root);
  4. the curve: root_g0 vs k, post_g0 vs k — the threshold located
     where post_g0 leaves the expression floor and where root_g0 enters
     the band.

REGISTERED BARS (frozen in the dispatch letter, VERBATIM in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping):
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
  * THE LADDER := k in {1000, 10000, 40000, 100000, 237123} (the
    dispatch's registered rungs verbatim); the top rung IS e258/e260's
    committed k (hard-bound 237,123). Each rung an INDEPENDENT SRCT
    room from its own registered seed pair — 1k: 26111/26112, 10k:
    26113/26114, 40k: 26115/26116, 100k: 26117/26118, 237k: 26011/26012
    — the top rung's room is e260's committed RANDOM room VERBATIM (the
    same seeds = the same room; bit-identity gated against
    e260_rooms.pt's stored D/S), so the anchor rung re-runs e260's
    RANDOM arm on this session's silicon.
  * Every arm: the SAME fresh root (e001 base, fact-free-gated), the
    SAME dose (E43.exposure Dmix install s400 gen 24314 + e113 cons
    s300 seed 10901 HELD; AdamW (0.9,0.95) wd 0.1; the house cosine; clip
    1.0) — the installs share bit-identical streams (one generator, one
    draw order); the ONLY delta across rungs is the room the post-clip
    gradient is projected onto (backward -> clip 1.0 VERBATIM ->
    project (CPU fp64; the write fp32) -> opt.step(); the norm is NOT
    rescaled — e237's convention).
  * "expression curve" := post-install g0 (the g0 battery on the
    install-final net) vs k over the rungs; ARM-FREE co-plotted at
    k = N (full space, the ceiling); e246's committed ALIGNED post g0
    2.86e-5 at rank 10 co-marked as the dead context point (NOT a rung).
  * "post_g0 jumps > 10x between adjacent rungs" := max over adjacent
    rung pairs (ascending k) of post_g0(k_hi)/max(post_g0(k_lo), 1e-6)
    > 10 (the 1e-6 denominator floor guards division; a post g0 at or
    under 1e-6 reads as dead, and 10x over dead is any 1e-5+ — the
    e246 ALIGNED scale; disclosed).
  * "the landing curve enters the band at a locatable rung" := some
    rung's root g0 lies in the frozen matched band [0.670278, 0.819229]
    (+-10% of the committed g1c root g0 0.7447534203529358); the
    threshold rung := the smallest such k; the bracket [last-out rung,
    first-in rung] co-reported; over-band rungs (> 0.819229) reported
    as over, not in.
  * "the landing approaches the band asymptotically" (the GRADUAL
    clause) := no rung in band AND the top rung's root g0 >= floor
    - 0.05 (within 5 points of the floor — "approaching"; e260's
    committed 237k read 0.6686 sits 0.0016 under the floor, so the
    clause is live at the top of this ladder by the committed record).
  * SHARP-THRESHOLD fires iff gates pass AND the max adjacent post_g0
    ratio > 10 AND some rung is in band.
  * GRADUAL fires iff gates pass AND the max adjacent post_g0 ratio
    <= 10 AND no rung is in band AND the top rung's root g0 >= floor
    - 0.05.
  * MIXED: everything else, with named branches (step-like expression
    but the landing never enters; smooth expression but a locatable
    landing; or the ladder never approaching the band — the honest
    "the last 0.08 does not live on the k axis as mapped" world).
    Composite order: TEXTURE (gate failure; G_FREE failure HALTS) ->
    SHARP-THRESHOLD -> GRADUAL -> MIXED.
  * G_FREE (the ceiling control) := e246/e258/e260's AMENDED tiers
    (i)-(iii) VERBATIM: (i) the instrumented-path install reproduces
    the committed install final (L2 <= 5e-3 AND behavioral g-12 |d| <=
    5e-3); (ii) FREE's root lands in the matched band; (iii) the cons
    tracks the committed cons (median step |g0 diff| <= 0.05). Tier
    (iv) (the W1 wash) is DROPPED — this cell runs NO wash (disclosed
    deviation; the retention/occupancy question is T237's separately
    registered follow-up).
  * G_ANCHOR (the 237k rung vs e260's committed RANDOM arm — the
    anchor gate, "within the disclosed session texture"): install-final
    L2 vs e260_RANDOM_inst_resume.pt <= 5e-3 AND |d post g0| <= 0.02
    AND |d root g0| <= 0.02 AND |d root g-12| <= 0.02 (vs e260's
    committed literals), AND the room bit-identical to e260_rooms.pt's
    stored D_rand/S_rand (exact). A G_ANCHOR failure is the TEXTURE
    branch (it runs last; nothing halts).
  * MEASURED, NEVER NOMINAL: per-rung kept fraction ||g'||/||g|| (the
    dose actually delivered; median over the install ledger), v-excess
    pre/post projection (vs e258's LOADED committed v-map), in-span
    fraction pre AND applied (vs e246's committed LATE span), and the
    displacement cos-to-span + in-own-room fractions at post-install
    and root.
  * THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's
    committed 2.74M v-map (runs/checkpoints/e258_vmap.pt) feeds the
    measured v-excess ledger and NOTHING else; no new history is run.
  * THE CONSOLIDATION runs NATURAL on all arms (e113 VERBATIM, seed
    10901 HELD): the intervention is the INSTALL geometry only; cons
    re-growth outside the room is part of the honest design and is
    measured by each root's in-own-room fraction.

CHECKS (the dispatch's, in force): e260's machinery smoke FIRST (its
record: 2-3 bugs caught per build); the k=237k rung should reproduce
e260's committed RANDOM arm (the anchor gate — within the disclosed
session texture); the rooms' certification per rung; n=1 per rung (the
lottery note carried; the curve's shape is the object, not any single
point); nothing guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
g-series' own reference scale). GPU trainings (6 installs s400 — 5
hooked rungs + FREE — + 6 cons s300; NO washes in this cell), each in
<= 175 s bursts (within the owner's <= 180 s window), per-step thermal
polls at a 78C margin from the first burst, 40 s cooldowns (the 30-60 s
window), the 84C never-past line recorded; CPU probing threads 4; the
dense projections run CPU fp64 (pocketfft workers 2).

Outputs: runs/e261/{metrics.json (PROGRESSIVE), e261_rank_ladder.png,
e261_ladder_instrument.png, run.log}; checkpoints
runs/checkpoints/e261_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e261_rank_ladder.py    (E261_SMOKE=1 shakedown)
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
import scipy.fft as sf                                # noqa: E402

import common                                          # noqa: E402
from common import (CharCorpus, cosine_lr, gpu_status,  # noqa: E402
                    run_dir, save_json, set_seed)

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # CORP_BS, MIX_RANDOM,
                                                      # LR, jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
                                                      # (the 2.74M patch)
import g1_anchored_ball as G1                          # noqa: E402 — the
                                                      # machinery (patched to
                                                      # 2.74M by g1b's import)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E261_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e261_smoke" if SMOKE else "e261"
assert torch.cuda.is_available(), "e261 owns the GPU lane (dispatch)"

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

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
ROOMS260_CK = "e260_rooms.pt"     # e260's committed rooms (the anchor room)
CKPT_DIR = GB.CKPT_DIR
FRESH_GEN = 24314                 # the g1c fresh draw's install gen (VERBATIM)
CONS_SEED = 10901                 # HELD (g2e: the cons stream is shared)
INST_STEPS = 400 if not SMOKE else 8
INST_TOTAL = 1000                 # e048's house-cosine total (verbatim)
CONS_STEPS = 300 if not SMOKE else 8
MATCH_BAND = 0.10                 # the pre-registered +-10% matched band

# ---- THE LADDER (registered): (k, seed_d, seed_s); the TOP rung is e260's
# committed RANDOM room VERBATIM (seeds 26011/26012 = the anchor)
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (1_000, 26111, 26112),
    (10_000, 26113, 26114),
    (40_000, 26115, 26116),
    (100_000, 26117, 26118),
    (237_123, 26011, 26012),      # THE ANCHOR (e260's RANDOM room, bit-bound)
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (64, 26111, 26112),
    (512, 26011, 26112),          # e260's own smoke room seeds (no committed
                                  # full-run record at smoke k — anchor vacuous)
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
ANCHOR_K = 237_123
RUNG_NAMES = {k: f"K{k // 1000}K" if not SMOKE else f"K{k}" for k, _, _ in LADDER}
ARMS = ("FREE",) + tuple(RUNG_NAMES[k] for k, _, _ in LADDER)

G0_ZERO_FLOOR = 0.05               # the frozen "g0 ~ 0" expression floor
G_READ_TOL = 5e-3                  # the family's cross-device read tolerance
ANCHOR_BEHAV_TOL = 0.02            # the anchor's disclosed session-texture bar
GRADUAL_REACH = 0.05               # "approaches the band": top rung >= floor-0.05
RATIO_DEN_FLOOR = 1e-6             # the jump-ratio denominator floor (disclosed)
JUMP_BAR = 10.0                    # the registered >10x expression-jump bar

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
G1C_METRICS = E43.REPO / "runs" / "g1c_root" / "metrics.json"
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562
G1C_ROOT_G0 = 0.7447534203529358      # THE landing anchor (the g0 battery)
G1C_POST_INSTALL = {"gm12": 0.16888952255249023,   # hard-bound (Rule 12)
                    "g0": 0.5302218198776245,
                    "ce_r": 1.673575520515442}

# e246's committed textures (the anti-substrate's own record; hard-bound)
E246_METRICS = E43.REPO / "runs" / "e246" / "metrics.json"
E246_VERDICT = "GEOMETRY-IRRELEVANT"
E246_ALIGNED_POST_G0 = 2.8589747671503574e-05      # the rank-10 dead anchor
E246_ALIGNED_RANK = 10
E246_SPAN_MD5 = "3464ff6081d8e402103dd9ca65bfb53a"
E246_SPAN_RANK = 10

# e258's committed record (the k convention + the dose-honesty precedent)
E258_METRICS = E43.REPO / "runs" / "e258" / "metrics.json"
E258_VERDICT = "NOT-THE-LOAD"
E258_K_HARD = 237_123              # the ladder's top rung (hard-bound)
E258_FREE_VEXC_PRE_MEDIAN = 29.061850539663425     # co-report texture (non-gate)

# e260's committed record (THE anchor arm; hard-bound)
E260_METRICS = E43.REPO / "runs" / "e260" / "metrics.json"
E260_VERDICT = "RANK-IS-THE-BARRIER"
E260_RANDOM_POST_G0 = 0.38436469435691833
E260_RANDOM_ROOT_G0 = 0.6686325073242188
E260_RANDOM_ROOT_GM12 = 0.20900560915470123
E260_RANDOM_KEPT_MED = 0.2945979050906026
E260_RANDOM_INST_CK = "e260_RANDOM_inst_resume.pt"
E260_ROOMS_CK = ROOMS260_CK

ARM_DESC = {
    "FREE": "the natural install through the instrumented path (fp64 "
            "ledger dots READ ONLY, no write) — must reproduce the "
            "committed g1c root or the cell HALTS (the ceiling reference)",
}
ARM_DESC.update({
    RUNG_NAMES[k]: f"each install gradient PROJECTED onto an INDEPENDENT "
                   f"random rank-{k} SRCT room (dense random orthonormal "
                   f"subspace; seeds {sd}/{ss}) before Adam — one rung of "
                   f"the ladder"
    for k, sd, ss in LADDER})
ARM_DESC[RUNG_NAMES[ANCHOR_K] if not SMOKE else RUNG_NAMES[LADDER[-1][0]]] += (
    " — THE ANCHOR: e260's committed RANDOM room VERBATIM (bit-identity "
    "gated vs e260_rooms.pt)")

# ---- THE THERMAL ENVELOPE (the e246/e258/e260 discipline on this chip; the
# owner's max-priority window): per-step polls from the FIRST burst at the
# 78C margin; bursts <= 175 s; cooldowns 40 s (the 30-60 s window); the 84C
# never-past line recorded (e260 matched: max 80.0C, zero >= 84C).
BURST_MAX_S = 175.0
COOLDOWN_S = 40.0
TEMP_EARLY_END = 78.0              # per-step-poll burst-end margin
TEMP_HARD = 84.0                   # the recorded never-past line
POLL_EVERY = 1                     # PER-STEP mid-burst temp polls
LAUNCH_POLL_GAP_S = 5.0
DCT_WORKERS = 2                    # pocketfft workers (shared machine)
CERT_PROBES = 4
CERT_SEED = 26115

REGISTERED = {
    "question_verbatim": "the expression barrier lives somewhere between "
        "rank 10 (dead) and rank 237,123 (within 0.2% of the band). WHERE? "
        "The landing curve root_g0(k) over the ladder maps the threshold — "
        "the anti-substrate's final quantitative form. (T237's registered "
        "ask: where does the last ~0.08 of root g0 live between k=237k and "
        "full space?)",
    "bars_verbatim": {
        "SHARP-THRESHOLD": "the expression curve is step-like (post_g0 "
            "jumps > 10x between adjacent rungs somewhere) AND the landing "
            "curve enters the band at a locatable rung — the "
            "anti-substrate's final form: a dimensional threshold at "
            "k ~ [the rung]; memories need k* dimensions, full stop",
        "GRADUAL": "both curves rise smoothly (no rung-to-rung jump > 10x "
            "in expression; the landing approaches the band "
            "asymptotically) — the barrier is a soft capacity gradient, "
            "not a threshold; the last 0.08 of root g0 is spread across "
            "the ladder",
        "MIXED": "the curves verbatim; expression step-like but landing "
            "gradual (or vice versa) — mapped honestly",
    },
    "operationalizations": (
        "frozen BEFORE compute: the ladder := k in {1000, 10000, 40000, "
        "100000, 237123} (the dispatch's registered rungs verbatim; the top "
        "rung hard-bound == e258/e260's committed 237,123); each rung an "
        "independent SRCT room (seeds 26111-26118; the top rung = e260's "
        "committed RANDOM room VERBATIM, seeds 26011/26012, bit-identity "
        "gated vs e260_rooms.pt); every arm the SAME fresh root (e001) and "
        "the SAME dose (Dmix s400 gen 24314 + e113 cons s300 seed 10901 "
        "HELD; installs share bit-identical streams); the hook is "
        "backward -> clip 1.0 VERBATIM -> project (CPU fp64, write fp32) -> "
        "opt.step, norm NOT rescaled; 'expression curve' := post-install g0 "
        "vs k (FREE co-plotted at k=N; e246's ALIGNED 2.86e-5 at rank 10 "
        "co-marked as context, not a rung); 'post_g0 jumps > 10x' := max "
        "adjacent ratio post_g0(k_hi)/max(post_g0(k_lo), 1e-6) > 10; "
        "'enters the band at a locatable rung' := some rung's root g0 in "
        "[0.670278, 0.819229] (+-10% of the committed g1c root g0 "
        "0.7447534203529358), threshold = the smallest such k, bracket "
        "co-reported, over-band rungs reported as over; 'approaches the "
        "band asymptotically' := no rung in band AND top-rung root g0 >= "
        "floor - 0.05; SHARP-THRESHOLD fires iff gates pass AND max ratio "
        "> 10 AND some rung in band; GRADUAL fires iff gates pass AND max "
        "ratio <= 10 AND no rung in band AND top rung >= floor - 0.05; "
        "MIXED otherwise with named branches; composite TEXTURE -> "
        "SHARP-THRESHOLD -> GRADUAL -> MIXED; G_FREE := the amended tiers "
        "(i)-(iii) VERBATIM (install L2 <= 5e-3 + behavioral; root band; "
        "cons tracking <= 0.05) — tier (iv) DROPPED (no wash in this "
        "cell); G_ANCHOR := the 237k rung vs e260's committed RANDOM arm "
        "(install L2 <= 5e-3 vs e260_RANDOM_inst_resume.pt; |d| <= 0.02 on "
        "post g0 / root g0 / root g-12; room bit-identity exact) — a "
        "failure is the TEXTURE branch (runs last; no halt); the v-map "
        "LOADED from e258's committed artifact (the measured v-excess "
        "ledger only); cons NATURAL on all arms; kept fraction + v-excess "
        "+ in-span + in-own-room measured everywhere — never nominal."),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (T237's registered next cell); this script committed at "
        "birth BEFORE any compute; adjudicate against exactly this; no bar "
        "shopping.",
}

deviations: list[str] = [
    "NO WASH IN THIS CELL (the largest intentional trim, disclosed at "
    "birth): the registered readouts are post-install g0 + root g0 + "
    "kept-fraction per rung — the CURVES are the object; e260's wash "
    "driver, the margin/thermal-family readouts, G_INPUTS/G_BITROOT and "
    "G_FREE tier (iv) (the W1 wash tier) are all DROPPED with it; the "
    "retention/occupancy question (does the corpus's room erode faster "
    "what was written into it?) is T237's separately registered "
    "follow-up, not this cell.",
    "THE KEPT-K COUPLING (the ladder's one interpretive act, disclosed "
    "at birth): the projection's norm is NOT rescaled (e237/e260's "
    "convention), so the DELIVERED DOSE covaries with the rung by "
    "construction — kept ~ sqrt(k/N) (0.019 at 1k -> 0.294 at 237k). The "
    "registered bars read the curves vs k as dispatched; the kept-fraction "
    "ledger is the disclosed covariate on every rung, and the instrument "
    "page co-plots root g0 against kept (the confound view). e258's "
    "committed texture bounds the confound's reach: its VLIGHT arm LANDED "
    "at kept ~0.10 — a kept-0.10 gradient still lands at matched strength "
    "— and e260's RANDOM arm at kept 0.294 missed the band floor by only "
    "0.0016; dose alone does not write the curve's shape.",
    "THE ANCHOR RUNG IS A RE-RUN, NOT A COPY (disclosed): the top rung "
    "re-runs e260's committed RANDOM arm on this session's silicon (the "
    "room bit-identical by seed; the stream bit-identical by construction) "
    "so the ladder is one instrument on one session; the anchor gate "
    "binds the re-run to the committed record (L2 + behavior within the "
    "disclosed session texture). Cross-pass cuBLAS/atomics noise is the "
    "family's inherited law (e260's G_FREE matched g1c at L2 7.06e-4).",
    "e260's machinery PORTED WHOLE: the SRCT projector, the hooked "
    "chunked install/cons drivers (bit-identical arithmetic + draw "
    "order), the G_FREE amended tiers (i-iii), the thermal envelope "
    "(per-step polls, 78C margin, 175s bursts, 40s cooldowns, the 84C "
    "line), the smoke discipline, the progressive-metrics + resume-ckpt "
    "conventions — only the room set (5 independent rungs + the anchor "
    "vs e260's 2 rooms), the adjudication clauses, and the dropped wash "
    "lane are new.",
    "THE DENSE PROJECTION RUNS ON CPU fp64 (pocketfft, workers 2): the "
    "rooms' directions cross parameter boundaries (that is what 'dense' "
    "means here), so the hook gathers the flat post-clip gradient, "
    "projects it exactly in fp64, and writes back fp32 (e246/e258/e260's "
    "two-pass discipline).",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger; no new "
    "history is run. The installs' own fresh Adam state remains the "
    "mechanism's channel (e258's disclosed scope): the APPLIED update is "
    "not confined to the room (Adam's per-coordinate normalizer + the "
    "natural cons grow off-room — e260's measured root in-own-room was "
    "0.659) — the rungs restrict the GRADIENT, and the in-own-room "
    "ledger says by how much the state itself stayed.",
    "e225 is NOT bound in this cell (no wash -> no retention spread); "
    "the matched band (not the draw spread) is the landing ruler, per "
    "the dispatch. e228's margin instrument is NOT imported (no margin "
    "readouts without the wash states; the e228_run.log append side "
    "effect avoided).",
    "n=1 per rung, one lineage, one session (the g-series standing "
    "caveat — the critic's lottery note carried verbatim); the curve's "
    "SHAPE is the registered object, not any single point; nothing "
    "guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E261_SMOKE=1): 8-step install/cons, ladder {64, 512} "
    "FIXED (the anchor room at e260's own smoke seeds but with NO "
    "committed full-run record — the anchor gate is an explicit vacuous "
    "pass, disclosed), own smoke dir; NOTHING adjudicated or gated "
    "(SMOKE stamp on every read).",
]

device_events: list[dict] = []
thermal_log: list[dict] = []


# ------------------------------------------------------------------ envelope
def gpu_poll(tag: str) -> dict:
    s = gpu_status()
    ok = s["util"] <= common.GPU_UTIL_CEIL and s["temp"] <= common.GPU_TEMP_CEIL
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"], ok)
    log(f"  [gpu:{tag}] util {s['util']:.0f}% temp {s['temp']:.0f}C "
        f"mem {s['mem_used']:.0f}/{s['mem_total']:.0f}MB "
        f"power {s['power']:.1f}W -> {'OK' if ok else 'HOLD'}")
    return {"poll": s, "ok": bool(ok)}


def wait_gpu_free(tag: str, max_wait_s: float = 1800.0) -> list[dict]:
    """Launch gate: common.gpu_ok() semantics, first launch double-polled
    (>= 5 s apart). The owner's max-priority window is active — short
    waits only."""
    polls = [gpu_poll(f"{tag}#1")]
    time.sleep(LAUNCH_POLL_GAP_S)
    polls.append(gpu_poll(f"{tag}#2"))
    t0w = time.time()
    while not (polls[-2]["ok"] and polls[-1]["ok"]):
        if time.time() - t0w > max_wait_s:
            raise RuntimeError(f"GPU window never opened for {tag}")
        log(f"  [gpu:{tag}] waiting 20s for the envelope "
            f"(util<={common.GPU_UTIL_CEIL:.0f}% "
            f"temp<={common.GPU_TEMP_CEIL:.0f}C)")
        time.sleep(20.0)
        polls.append(gpu_poll(f"{tag}#w"))
    return polls


def burst_temp_check(tag: str) -> tuple[bool, float]:
    """Mid-burst thermal guard: (keep_going, temp). PER-STEP polls from the
    FIRST burst (the 5090 ramps ~9C/s at burst start — e233's lesson, the
    default here); the burst ends at the 78C margin so a one-step sensor
    jump stays under the never-past line; a >= 84C read is a recorded
    VIOLATION (e260 matched: max 80.0C, zero >= 84C)."""
    s = gpu_status()
    common._log_envelope_poll(f"{NAME}:{tag}:mid", s["util"], s["temp"],
                              s["temp"] < TEMP_EARLY_END)
    row = {"tag": tag, "temp": s["temp"], "t": round(time.time() - T0, 1)}
    thermal_log.append(row)
    if s["temp"] >= TEMP_HARD:
        device_events.append({"tag": tag, "event": "THERMAL VIOLATION "
                              f"(>= {TEMP_HARD:.0f}C)", "status": s})
        log(f"  [gpu:{tag}:mid] TEMP {s['temp']:.0f}C >= {TEMP_HARD:.0f}C — "
            f"HARD VIOLATION recorded; ending burst")
        return False, s["temp"]
    if s["temp"] >= TEMP_EARLY_END:
        log(f"  [gpu:{tag}:mid] temp {s['temp']:.0f}C >= "
            f"{TEMP_EARLY_END:.0f}C margin — ending burst "
            f"(85C line protected)")
        return False, s["temp"]
    return True, s["temp"]


def burst_cooldown(tag: str) -> None:
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s ({tag})")
    time.sleep(COOLDOWN_S)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


# ------------------------------------------------------ THE ROOMS (the core)
class SRCT:
    """A random rank-k DENSE orthonormal subspace of R^N — the SRCT
    ensemble: the k columns {D . phi_s : s in S} with D a seeded +-1
    diagonal and phi_j the orthonormal DCT-II basis vectors. The exact
    orthogonal projector is applied as
        P x = D . idct(mask_S(dct(D . x)))
    (fp64 pocketfft, CPU). The basis itself is NEVER materialized (an
    N x k fp32 basis would be ~2.6 TB at k=237k) — only the projector
    exists, and it is exact (certified in G_PROJ).

    PORTED VERBATIM from lab/e260_rank_matched.py (its own provenance
    e258/e246; this cell's rooms differ only in the registered seeds)."""

    def __init__(self, n: int, k: int, seed_d: int, seed_s: int):
        assert 0 < k <= n
        self.n, self.k = int(n), int(k)
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        self.S = torch.randperm(n, generator=g2)[:self.k].sort().values \
            .numpy()
        self.mask = np.zeros(n, dtype=np.float64)
        self.mask[self.S] = 1.0
        self.seed_d, self.seed_s = seed_d, seed_s

    def coeffs(self, x64: np.ndarray) -> np.ndarray:
        """c = F^T x (the room's coordinates of x; full N vector)."""
        return sf.dct(self.D * x64, type=2, norm="ortho",
                      workers=DCT_WORKERS)

    def recon(self, c_masked: np.ndarray) -> np.ndarray:
        """F c (reconstruction from (masked) room coordinates)."""
        return self.D * sf.idct(c_masked, type=2, norm="ortho",
                                workers=DCT_WORKERS)

    def project(self, x64: np.ndarray) -> np.ndarray:
        c = self.coeffs(x64)
        c *= self.mask
        return self.recon(c)


class LadderRooms:
    """THE CELL'S ONLY INTERVENTION: the ladder's rank-k rooms + the
    read-only ledger. The flat basis IS net.parameters() order (the
    lineage's convention); the rooms' directions cross parameter
    boundaries (dense).

    step_hook(params, mode):
      FREE    — ledger dots READ ONLY (gn, v-excess pre, in-span pre);
                the gradient is NEVER touched (the verbatim trajectory);
      KxxxK   — g' = P_room g (that rung's SRCT room), written fp32.

    The ledger (fp64): gn = ||g||, gpn = ||g'||, kept_frac = ||g'||/||g||,
    v_excess_pre/post (e257's qov convention vs the LOADED v-map),
    in_span_frac (pre-projection, ||Vp g||/||g||), applied_in_span_frac
    (||Vp g'||/||g'|| — how much span the APPLIED gradient carries)."""

    def __init__(self, n: int, ladder, v_flat64: np.ndarray,
                 Vp64: np.ndarray, params_ref, dev: torch.device):
        self.n = int(n)
        self.dev = dev
        self.r_span = int(Vp64.shape[0])
        self.v64 = v_flat64.astype(np.float64)
        self.mean_v = float(self.v64.mean())
        self.Vp = Vp64.astype(np.float64)                  # (r, N)
        self.rooms: dict[str, SRCT] = {}
        self.room_k: dict[str, int] = {}
        for k, sd, ss in ladder:
            nm = RUNG_NAMES[k]
            assert nm not in self.rooms
            self.rooms[nm] = SRCT(n, k, sd, ss)
            self.room_k[nm] = int(k)
        # the per-parameter offsets (the write-back slicing)
        self.offsets, self.shapes = [], []
        off = 0
        for p in params_ref:
            self.offsets.append((off, off + p.numel()))
            self.shapes.append(tuple(p.shape))
            off += p.numel()
        assert off == self.n, f"flat size {off} != {self.n}"

    # -- the rooms' projectors (flat fp64 numpy) -----------------------
    def proj_of(self, mode: str, g: np.ndarray) -> np.ndarray:
        if mode in self.rooms:
            return self.rooms[mode].project(g)
        if mode != "FREE":
            raise KeyError(f"unknown room mode {mode}")
        return g

    # -- the hook ----------------------------------------------------------
    def step_hook(self, params, mode: str) -> dict:
        params = list(params)
        g = torch.cat([p.grad.detach().reshape(-1) for p in params]) \
            .to(CPU).double().numpy().astype(np.float64)
        gn2 = float(g @ g)
        vexc_pre = float((g * g * self.v64).sum() / gn2 / self.mean_v) \
            if gn2 > 0 else 0.0
        c1 = self.Vp @ g
        in_span = math.sqrt(float(c1 @ c1) / gn2) if gn2 > 0 else 0.0
        if mode == "FREE":
            return {"gn": math.sqrt(gn2), "gpn": math.sqrt(gn2),
                    "kept_frac": 1.0, "v_excess_pre": vexc_pre,
                    "v_excess_post": vexc_pre, "in_span_frac": in_span,
                    "applied_in_span_frac": in_span}
        gp = self.proj_of(mode, g)
        gpn2 = float(gp @ gp)
        vexc_post = float((gp * gp * self.v64).sum() / gpn2 / self.mean_v) \
            if gpn2 > 0 else 0.0
        c1p = self.Vp @ gp
        applied_in_span = math.sqrt(float(c1p @ c1p) / gpn2) \
            if gpn2 > 0 else 0.0
        gp32 = torch.from_numpy(gp.astype(np.float32))
        with torch.no_grad():
            for p, (a, b), shp in zip(params, self.offsets, self.shapes):
                p.grad.copy_(gp32[a:b].to(self.dev).reshape(shp))
        return {"gn": math.sqrt(gn2), "gpn": math.sqrt(gpn2),
                "kept_frac": math.sqrt(gpn2 / gn2) if gn2 > 0 else 0.0,
                "v_excess_pre": vexc_pre, "v_excess_post": vexc_post,
                "in_span_frac": in_span,
                "applied_in_span_frac": applied_in_span}

    # -- the read-only probes (certification + measured overlaps) ----------
    def certify(self) -> dict:
        """Per-rung certification: the DCT roundtrip identity; each room's
        IDEMPOTENCY (||P^2 x - P x||/||P x||) and kept^2 rank probe
        (||P x||^2/||x||^2 vs k/N); each room's measured overlap with the
        span directions (||P v_j||, expect ~sqrt(k/N)). The kept^2 bar is
        the 10-sigma statistical bar for the 4-probe mean
        (5*sqrt(2k)/N) — e260's fixed 0.02 form is meaninglessly loose at
        k=1k (it would sit ~1000x above the mean); the tightening is
        disclosed and catches size bugs either way."""
        rng = np.random.default_rng(CERT_SEED)
        out: dict = {"probes": CERT_PROBES, "seed": CERT_SEED}
        x = rng.standard_normal(self.n)
        rt = self.rooms[RUNG_NAMES[LADDER[0][0]]]
        xr = rt.recon(rt.coeffs(x))
        out["dct_roundtrip_rel"] = float(np.linalg.norm(xr - x)
                                         / np.linalg.norm(x))
        per_rung = {}
        for k, _, _ in LADDER:
            nm = RUNG_NAMES[k]
            room = self.rooms[nm]
            idem, kept2 = [], []
            for _ in range(CERT_PROBES):
                x = rng.standard_normal(self.n)
                px = room.project(x)
                ppx = room.project(px)
                idem.append(float(np.linalg.norm(ppx - px)
                                  / np.linalg.norm(px)))
                kept2.append(float((px @ px) / (x @ x)))
            ovr = [float(np.linalg.norm(room.project(self.Vp[j])))
                   for j in range(self.r_span)]
            bar = max(5.0 * math.sqrt(2.0 * k) / self.n, 1e-9)
            per_rung[nm] = {
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
        out["per_rung"] = per_rung
        out["pass"] = bool(out["dct_roundtrip_rel"] <= 1e-8
                           and all(r["kept2_pass"] and r["idempotency_pass"]
                                   for r in per_rung.values()))
        return out

    # -- displacement reads -------------------------------------------------
    def displacement_loads(self, flat_cpu: torch.Tensor, mode: str) -> dict:
        d = flat_cpu.double().numpy().astype(np.float64)
        d2 = d * d
        tot = float(d2.sum())
        out = {"v_excess": float((d2 * self.v64).sum() / tot / self.mean_v)
               if tot > 0 else 0.0}
        c1 = self.Vp @ d
        out["cos_to_span"] = math.sqrt(float(c1 @ c1) / tot) if tot > 0 else 0.0
        if mode in self.rooms:
            pd = self.proj_of(mode, d)
            out["in_own_room"] = math.sqrt(float((pd * d).sum() / tot)) \
                if tot > 0 else 0.0
        else:
            out["in_own_room"] = None
        return out


# ------------------------------------------------------------------ trainers
def _open_burst(tag: str) -> tuple[float, int]:
    wait_gpu_free(tag)
    t_burst = time.time()
    log(f"[{tag}] burst opens")
    return t_burst, 0


def _end_burst_early(tag: str, n_burst: int, t_burst: float) -> None:
    log(f"  [{tag}] burst ended: {n_burst} steps in "
        f"{time.time() - t_burst:.1f}s (thermal margin)")


def chunked_install(tag, mode, net0, proj: LadderRooms,
                    inst_x, inst_mask, anchor_full, train_ids, g0_ids,
                    gm12_ids, r_eval_xy, zid, resume_ck: Path,
                    dev: torch.device) -> dict:
    """g1c's chunked_install arithmetic VERBATIM (E43.exposure Dmix: ix(16)
    install w/ name mask + aj(16) paired + rj(32) random; masked token-level
    union CE; AdamW (0.9,0.95) wd 0.1; lr 1e-3 x house cosine(total=1000);
    clip 1.0) + THE HOOK: for each rung each post-clip gradient is
    PROJECTED onto that rung's rank-k room (CPU fp64; write fp32) before
    opt.step(); for FREE the ledger dots are READ ONLY (the verbatim
    trajectory). The traj carries BOTH batteries (g0 = the expression
    ruler; g-12 = the wash ruler)."""
    name_bs, corp_bs, mix_random, lr = (G1.NAME_BS, E43.CORP_BS,
                                        E43.MIX_RANDOM, E43.LR)
    n_steps = INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    state = {"step": 0, "traj": [], "ledger": {}}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net, opt, gen, evl = None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = _open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            gen = torch.Generator().manual_seed(FRESH_GEN)
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
            f = cosine_lr(step - 1, INST_TOTAL)          # house schedule
            for g in opt.param_groups:
                g["lr"] = lr * f
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
            # ---- THE HOOK (the cell's only intervention) ---------------
            led = proj.step_hook(list(net.parameters()), mode)
            opt.step()          # BOTH moments + the update see the hook's g'
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
            # per-step thermal polls (the e233/e237/e246/e258/e260 discipline)
            ok_t, temp = burst_temp_check(f"{tag}-c{n_chunks}")
            chunk_temps.append(temp)
            if not ok_t:
                _end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{BURST_MAX_S:.0f}s at s{step} — resume ckpt saved")
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
                    "ledger": state["ledger"], "n_chunks": n_chunks,
                    "chunk_table": chunk_table}, resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "ledger": state["ledger"],
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}


def chunked_consolidate(tag, net0, pool_a_x, pool_a_mask, cons_anchor,
                        train_ids, g0_ids, r_eval_xy, zid,
                        resume_ck: Path, dev: torch.device) -> dict:
    """g1c's chunked_consolidate arithmetic VERBATIM (G1.consolidate =
    e113's finetune_arm: batch 32 = 16 jittered-pool install windows +
    16 anchors; token-level union CE; AdamW (0.9,0.95) wd 0.1 const lr
    1e-3; clip 1.0; s300 seed 10901). NATURAL on all arms (no hook)."""
    n_steps = CONS_STEPS
    n_pool, n_anc = pool_a_x.shape[0], cons_anchor.shape[0]
    state = {"step": 0, "traj": []}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net, opt, gen, evl = None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = _open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            gen = torch.Generator().manual_seed(CONS_SEED)
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
            ix = torch.randint(n_pool, (16,), generator=gen)
            aj = torch.randint(n_anc, (8,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                               generator=gen)
            nw = pool_a_x[ix].to(dev)
            anc = torch.cat([cons_anchor[aj],
                             torch.stack([train_ids[s: s + G1.BLOCK]
                                          for s in rj])], 0).to(dev)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
            m[:16] = pool_a_mask[ix].to(dev)
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1), reduction="none"
                                  ).view(x.shape[0], x.shape[1])
            nm = nll[:16][m[:16]]
            cm = nll[16:]
            loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 25 == 0 or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step, "g0_pz": bz["mean_pz"],
                                      "ce_r": ce_r,
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f}")
            n_burst += 1
            ok_t, temp = burst_temp_check(f"{tag}-c{n_chunks}")
            chunk_temps.append(temp)
            if not ok_t:
                _end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap at s{step}")
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
    return {"sd": sd_cpu, "traj": state["traj"], "steps_ran": step,
            "n_chunks": n_chunks, "chunk_table": chunk_table}


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
        "experiment": "e261_rank_ladder",
        "phase": "THE RANK/DOSE LADDER — where does the last ~0.08 of root "
                 "g0 live between k=237k and full space? Random-room "
                 "installs at k in {1k, 10k, 40k, 100k, 237k(anchor)} + "
                 "FREE (the ceiling); the expression + landing curves vs k "
                 "locate the threshold — SHARP vs GRADUAL vs MIXED",
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
            "bursts": f"<= {BURST_MAX_S:.0f}s, per-step thermal polls at a "
                      f"{TEMP_EARLY_END:.0f}C margin from the first burst, "
                      f"cooldown {COOLDOWN_S:.0f}s, the {TEMP_HARD:.0f}C "
                      "never-past line recorded",
            "trainings": f"{len(ARMS)} installs s{INST_STEPS} ({len(LADDER)}"
                         " hooked rungs + FREE) + "
                         f"{len(ARMS)} cons s{CONS_STEPS}; NO washes (the "
                         "registered readouts are the curves; the v-map is "
                         "LOADED from e258's committed artifact)",
        },
        "deviations": deviations,
        "builds_on": [
            "T237 / e260 (THE registration: the rank/dose ladder is its "
            "registered next cell; the machinery PORTED WHOLE — the SRCT "
            "projector, the hook order, the certified-construction rooms, "
            "the G_FREE tiers, the thermal envelope; the committed RANDOM "
            "arm is this ladder's TOP RUNG and ANCHOR: root g0 0.6686, "
            "post g0 0.3844, kept 0.2946, k 237,123)",
            "T235 / e258 (the k50 rank convention the top rung freezes; "
            "the committed v-map this cell LOADS; the VLIGHT kept-0.10 "
            "landing — the dose-honesty precedent that bounds the kept-k "
            "coupling)",
            "T226 / e246 (the anti-substrate's origin: the ALIGNED "
            "rank-10 install expressed NOTHING, post g0 2.86e-5 — this "
            "ladder's DEAD context point; the committed LATE span feeds "
            "the in-span ledger columns)",
            "T181 / g1c (the fresh-root lineage: e001+Dmix s400 gen 24314 + "
            "e113 cons 10901; the committed root + install records are the "
            "controls)",
        ],
        "whats_new": [
            "THE LADDER: five independent random-room installs at k in "
            "{1k, 10k, 40k, 100k, 237k} — the rank axis walked for the "
            "first time between e246's dead rank-10 and e260's "
            "0.2%-from-the-band rank-237k; every rung the SAME fresh root, "
            "the SAME dose, bit-identical streams (the room is the only "
            "delta)",
            "THE ANCHOR RUNG: the top rung re-runs e260's committed RANDOM "
            "arm VERBATIM (bit-identical room, gated vs e260_rooms.pt) on "
            "this session's silicon — the ladder is one instrument on one "
            "session, bound to the committed record",
            "THE CURVES + THE CONFOUND VIEW: post_g0(k) and root_g0(k) "
            "with the threshold located (floor-crossing rung + band-entry "
            "bracket), and the kept-fraction ledger co-plotted (root g0 vs "
            "kept — the delivered-dose covariate made visible, never "
            "nominal)",
        ],
        "gates": {},
    })
    log(f"E261 — THE RANK/DOSE LADDER (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)} at identical dose (e001 + Dmix s"
        f"{INST_STEPS} gen {FRESH_GEN} + e113 cons s{CONS_STEPS} seed "
        f"{CONS_SEED}); landing = root g0 within +-{MATCH_BAND:.0%} of the "
        f"committed root g0 {G1C_ROOT_G0:.4f} -> "
        f"[{G1C_ROOT_G0 * (1 - MATCH_BAND):.4f}, "
        f"{G1C_ROOT_G0 * (1 + MATCH_BAND):.4f}]; expression floor g0 "
        f"{G0_ZERO_FLOOR}; ladder k: "
        + "/".join(str(k) for k, _, _ in LADDER)
        + f"; anchor {RUNG_NAMES[LADDER[-1][0]]} vs e260's committed RANDOM "
        f"(room seeds {LADDER[-1][1]}/{LADDER[-1][2]})")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(26101)                 # global init only; every RNG is its own

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

    # batteries (e119/e176n verbatim): install-60 + held30 at {-12,0,+12}
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

    # install windows + original-host anchor bank (g1's phase-0 construction)
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

    # jittered install pool (e113's construction VERBATIM) for consolidation
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

    # the neutral bank (e170's construction VERBATIM) — built for the
    # record (no wash in this cell); the gate keeps the protocol identical
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
    g1c = json.loads(G1C_METRICS.read_text(encoding="utf-8"))
    g1c_root_cells = g1c["root_build"]["root_cells"]
    g1c_root_gm12 = g1c_root_cells["gm12"]
    g1c_root_g0 = g1c_root_cells["g0"]
    g1c_post_inst = g1c["root_build"]["install"]["post_cells"]
    e246 = json.loads(E246_METRICS.read_text(encoding="utf-8"))
    e246_aligned_post_g0 = e246["arms"]["ALIGNED"]["install"]["post_cells"]["g0"]
    e258 = json.loads(E258_METRICS.read_text(encoding="utf-8"))
    e260 = json.loads(E260_METRICS.read_text(encoding="utf-8"))
    e260_rand_post = e260["arms"]["RANDOM"]["install"]["post_cells"]["g0"]
    e260_rand_root_g0 = e260["arms"]["RANDOM"]["root"]["g0"]
    e260_rand_root_gm12 = e260["arms"]["RANDOM"]["root"]["gm12"]
    e260_rand_kept = e260["arms"]["RANDOM"]["install"][
        "ledger_kept_frac_median"]
    e260_k = int(e260["rooms"]["k_rule"]["k"])
    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root_gm12, "root_g0": g1c_root_g0,
                        "post_install_cells": g1c_post_inst},
        "e246_metrics": {"path": str(E246_METRICS), "md5": md5of(E246_METRICS),
                         "verdict": e246["adjudication"]["verdict"],
                         "ALIGNED_post_g0": e246_aligned_post_g0},
        "e258_metrics": {"path": str(E258_METRICS), "md5": md5of(E258_METRICS),
                         "verdict": e258["adjudication"]["verdict"],
                         "k": e258["vmap"]["k_rule"]["k"]},
        "e260_metrics": {"path": str(E260_METRICS), "md5": md5of(E260_METRICS),
                         "verdict": e260["adjudication"]["verdict"],
                         "RANDOM_post_g0": e260_rand_post,
                         "RANDOM_root_g0": e260_rand_root_g0,
                         "RANDOM_root_gm12": e260_rand_root_gm12,
                         "RANDOM_kept_median": e260_rand_kept,
                         "k": e260_k},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK)},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "e260_rooms": {"path": f"runs/checkpoints/{ROOMS260_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS260_CK)},
        "e260_random_inst_resume": {
            "path": f"runs/checkpoints/{E260_RANDOM_INST_CK}",
            "exists": bool((CKPT_DIR / E260_RANDOM_INST_CK).exists())},
        "hardbound": {"root_gm12": G1C_ROOT_GM12, "root_g0": G1C_ROOT_G0,
                      "g1c_verdict": G1C_VERDICT,
                      "e246_verdict": E246_VERDICT,
                      "e246_ALIGNED_post_g0": E246_ALIGNED_POST_G0,
                      "e246_ALIGNED_rank": E246_ALIGNED_RANK,
                      "e246_span_md5": E246_SPAN_MD5,
                      "e246_span_rank": E246_SPAN_RANK,
                      "e258_verdict": E258_VERDICT,
                      "e258_k": E258_K_HARD,
                      "e260_verdict": E260_VERDICT,
                      "e260_RANDOM_post_g0": E260_RANDOM_POST_G0,
                      "e260_RANDOM_root_g0": E260_RANDOM_ROOT_G0,
                      "e260_RANDOM_root_gm12": E260_RANDOM_ROOT_GM12,
                      "e260_RANDOM_kept_med": E260_RANDOM_KEPT_MED},
        "pass": bool(
            g1c["adjudication"]["verdict"] == G1C_VERDICT
            and abs(g1c_root_gm12 - G1C_ROOT_GM12) < 1e-12
            and abs(g1c_root_g0 - G1C_ROOT_G0) < 1e-12
            and e246["adjudication"]["verdict"] == E246_VERDICT
            and abs(e246_aligned_post_g0 - E246_ALIGNED_POST_G0) < 1e-12
            and e258["adjudication"]["verdict"] == E258_VERDICT
            and int(e258["vmap"]["k_rule"]["k"]) == E258_K_HARD
            and e260["adjudication"]["verdict"] == E260_VERDICT
            and abs(e260_rand_post - E260_RANDOM_POST_G0) < 1e-12
            and abs(e260_rand_root_g0 - E260_RANDOM_ROOT_G0) < 1e-12
            and abs(e260_rand_root_gm12 - E260_RANDOM_ROOT_GM12) < 1e-12
            and abs(e260_rand_kept - E260_RANDOM_KEPT_MED) < 1e-12
            and e260_k == E258_K_HARD
            and (SMOKE or LADDER[-1][0] == E258_K_HARD)
            and md5of(CKPT_DIR / SPAN_CK) == E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G1C_VERDICT} (root g0 {g1c_root_g0:.6f}); "
        f"e246 {E246_VERDICT} (ALIGNED post g0 {e246_aligned_post_g0:.2e} at "
        f"rank {E246_ALIGNED_RANK}); e258 {E258_VERDICT} (k {E258_K_HARD}); "
        f"e260 {E260_VERDICT} (RANDOM root g0 {e260_rand_root_g0:.6f}, post "
        f"g0 {e260_rand_post:.4f}, kept {e260_rand_kept:.4f})")
    # refresh the hard-bound post-install literals at runtime (Rule 12)
    global G1C_POST_INSTALL
    G1C_POST_INSTALL = {"gm12": g1c_post_inst["gm12"],
                        "g0": g1c_post_inst.get("g0"),
                        "ce_r": g1c_post_inst.get("ce_r")}
    write_partial("P0b parents hard-bound")

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
        "battery_read_committed": G1C_ROOT_GM12,
        "abs_diff": abs(root_read - G1C_ROOT_GM12), "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - G1C_ROOT_GM12) < G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"P1 G_ROOT: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {G1C_ROOT_GM12:.10f} "
        f"(|d| {abs(root_read - G1C_ROOT_GM12):.1e}): PASS")
    write_partial("P1 G_ROOT PASSED")
    del root_net

    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))   # the params basis

    # ---- THE V-MAP LOADED (e258's committed artifact; extend, don't repeat)
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
                     and (SMOKE or int(vmap_art["meta"]["k"]) == E258_K_HARD)),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    log(f"P1 G_VMBIND: {VMAP_CK} (md5 {G_VMBIND['md5'][:8]}..., "
        f"{G_VMBIND['size']} coords, meta k {G_VMBIND['meta_k']}): PASS")

    # ---- THE SPAN LOADED (e246's committed LATE span; the ledger's columns)
    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == E246_SPAN_MD5
                               and int(Vp.shape[0]) == E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND
    log(f"P1 G_SPANBIND: {SPAN_CK} (rank {G_SPANBIND['rank']}, "
        f"md5-bound): PASS")

    # ---- THE LADDER BUILT + CERTIFIED -------------------------------------
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                        params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the ladder's SRCT rooms certified (fp64 CPU, "
                f"{CERT_PROBES} probes, seed {CERT_SEED}): the DCT roundtrip "
                "identity; each rung's IDEMPOTENCY and kept^2 rank probe "
                "(||P x||^2/||x||^2 vs k/N, the 10-sigma 4-probe bar "
                "5*sqrt(2k)/N — e260's fixed 0.02 form is meaninglessly "
                "loose at k=1k; disclosed); each rung's span-overlap "
                "(expect ~sqrt(k/N))",
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

    # ---- THE ANCHOR ROOM'S BIT-IDENTITY (vs e260_rooms.pt, exact) --------
    rooms260 = torch.load(CKPT_DIR / ROOMS260_CK, map_location="cpu",
                          weights_only=False)
    anchor_name = RUNG_NAMES[LADDER[-1][0]]
    if not SMOKE:
        D260 = rooms260["model"]["D_rand_int8"].numpy().astype(np.float64)
        S260 = rooms260["model"]["S_rand"].numpy()
        D_mine = rooms.rooms[anchor_name].D
        S_mine = rooms.rooms[anchor_name].S
        G_ANCHORROOM = {
            "form": "the anchor rung's room == e260's committed RANDOM room "
                    "(seeds 26011/26012 at k=237,123): the +-1 diagonal and "
                    "the index set bit-identical to e260_rooms.pt's stored "
                    "D_rand/S_rand (exact equality)",
            "D_bit_equal": bool(np.array_equal(D_mine, D260)),
            "S_bit_equal": bool(np.array_equal(S_mine, S260)),
            "e260_rooms_md5": md5of(CKPT_DIR / ROOMS260_CK),
            "pass": bool(np.array_equal(D_mine, D260)
                         and np.array_equal(S_mine, S260)
                         and int(rooms260["meta"]["k"]) == LADDER[-1][0]),
        }
        del rooms260
    else:
        G_ANCHORROOM = {
            "form": "SMOKE: e260's smoke room shares the seeds (26011/26012) "
                    "but no committed full-run record exists at smoke k — "
                    "the anchor identity is VACUOUS (explicit pass, "
                    "disclosed)",
            "pass": True, "vacuous": True,
        }
        del rooms260
    assert G_ANCHORROOM["pass"], f"anchor room bind failed: {G_ANCHORROOM}"
    metrics["gates"]["G_ANCHORROOM"] = G_ANCHORROOM
    log(f"P1 G_ANCHORROOM: the {anchor_name} rung's room "
        f"{('bit-identical to e260_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    # save the rooms artifact (reconstructible from the registered seeds)
    rooms_ck = save_ckpt(
        "e261_rooms",
        {RUNG_NAMES[k]: {"D_int8": rooms.rooms[RUNG_NAMES[k]].D.astype(np.int8),
                         "S": rooms.rooms[RUNG_NAMES[k]].S,
                         "k": k, "seeds": [sd, ss]}
         for k, sd, ss in LADDER},
        {"desc": "e261's ladder rooms (the flat basis is "
                 "net.parameters() order): one INDEPENDENT SRCT room per "
                 "rung (seeds registered per rung; the top rung = e260's "
                 "committed RANDOM room verbatim)",
         "ladder": [k for k, _, _ in LADDER], "n": N,
         "span_rank": rooms.r_span,
         "cert": {nm: {kk: vv for kk, vv in r.items()
                       if not isinstance(vv, list)}
                  for nm, r in cert["per_rung"].items()}})
    metrics["rooms"] = {
        "ladder": {"rule": "the dispatch's registered rungs verbatim — "
                           "{1k, 10k, 40k, 100k, 237k(matched-committed)}; "
                           "the top rung hard-bound == e258/e260's "
                           "committed k",
                   "rungs": [{"k": k, "name": RUNG_NAMES[k],
                              "seeds": [sd, ss],
                              "k_fraction_of_N": k / N}
                             for k, sd, ss in LADDER],
                   "top_rung_is_anchor": bool(not SMOKE)},
        "seeds": {RUNG_NAMES[k]: [sd, ss] for k, sd, ss in LADDER},
        "cert_probes_seed": CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       "LATE span; rank {rooms.r_span}; the ledger's "
                       "in-span columns)",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE LADDER: {' + '.join(f'{RUNG_NAMES[k]}(k={k})' for k, _, _ in LADDER)}"
        f" + FREE: BUILT + CERTIFIED")
    write_partial("P1 THE LADDER built (parents bound + v-map loaded + span "
                  "loaded + per-rung certification + anchor-room identity)")

    # ================= P2-P4: THE ARMS (FREE first — the halt gate) ======
    arms_rec: dict = {}
    theta0s: dict = {}

    def run_arm(arm: str) -> None:
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        inst = chunked_install(
            f"{arm}-inst", arm, G1.evl_load(base_sd), rooms,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e261_{arm}_inst_resume.pt" if SMOKE
                        else f"e261_{arm}_inst_resume.pt"), dev)
        sd_install = inst["sd"]
        # post-install dial + measured loads
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
            "desc": ARM_DESC[arm], "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_kept_frac_median": med(led_kept),
                "ledger_v_excess_pre_median": med(led_vpre),
                "ledger_v_excess_post_median": med(led_vpost),
                "ledger_in_span_frac_median": med(led_span),
                "ledger_applied_in_span_frac_median": med(led_aspan),
                "chunk_table": inst["chunk_table"], "steps": INST_STEPS,
                "post_cells": inst_cells,
                "displacement_loads": load_inst},
        }
        log(f"ARM-{arm} install done: post g0 {inst_cells['g0']:.4f} g-12 "
            f"{inst_cells['gm12']:.4f} CE_R {inst_cells['ce_r']:.4f} | d "
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
        cons = chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e261_{arm}_cons_resume.pt" if SMOKE
                        else f"e261_{arm}_cons_resume.pt"), dev)
        theta0 = cons["sd"]
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"] for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - base_flat
        load_root = rooms.displacement_loads(d_root, arm)
        theta0s[arm] = theta0
        root_ck = save_ckpt(
            f"e261_{arm}_root", theta0,
            {"desc": f"e261 ARM-{arm} root: e001 + Dmix s{INST_STEPS} "
                     f"(gen {FRESH_GEN}, {ARM_DESC[arm]}) + e113 cons "
                     f"s{CONS_STEPS} (seed {CONS_SEED} HELD)",
             "arm": arm, "install_seed": FRESH_GEN, "mode": arm,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e261_rooms.pt"})
        arms_rec[arm]["consolidation"] = {"traj": cons["traj"],
                                          "chunk_table": cons["chunk_table"]}
        arms_rec[arm]["root"] = {"cells": cells,
                                 "gm12": cells["g-12"],
                                 "g0": cells["g+0"],
                                 "displacement_loads": load_root,
                                 "checkpoint": root_ck}
        landed = (G1C_ROOT_G0 * (1 - MATCH_BAND)
                  <= cells["g+0"] <= G1C_ROOT_G0 * (1 + MATCH_BAND))
        arms_rec[arm]["root"]["landed"] = bool(landed)
        arms_rec[arm]["root"]["expresses"] = bool(
            inst_cells["g0"] >= G0_ZERO_FLOOR)
        log(f"ARM-{arm} ROOT: g0 {cells['g+0']:.4f} (band "
            f"[{G1C_ROOT_G0 * (1 - MATCH_BAND):.4f}, "
            f"{G1C_ROOT_G0 * (1 + MATCH_BAND):.4f}]) landed={landed} "
            f"(post-install g0 {inst_cells['g0']:.4f} "
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
    if not arms_rec["FREE"]["install"].get("resumed_final", False):
        pass  # G_FREE's CPU reads follow; the explicit ladder cooldown below

    # ---- G_FREE: THE CEILING CONTROL (tiers i-iii; tier iv dropped) ------
    free_post = arms_rec["FREE"]["install"]["post_cells"]
    comm_inst = torch.load(CKPT_DIR / "g1c_install_resume.pt",
                           map_location="cpu", weights_only=False)["model"]
    my_inst = torch.load(
        CKPT_DIR / ("smoke_e261_FREE_inst_resume.pt" if SMOKE
                    else "e261_FREE_inst_resume.pt"),
        map_location="cpu", weights_only=False)["model"]
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
        "form": ("e246's AMENDED G_FREE tiers, ported VERBATIM via "
                 "e258/e260, tiers (i)-(iii) (tier (iv), the W1 wash, is "
                 "DROPPED — no wash in this cell, disclosed): (i) the "
                 "instrumented-path install reproduces the committed "
                 "install final (L2 <= 5e-3 AND behavioral g-12 |d| <= "
                 "5e-3); (ii) FREE's root lands in the matched band; "
                 "(iii) the cons tracks the committed cons (median step "
                 "|g0 diff| <= 0.05, texture)"),
        "install_reproduction": {
            "l2_vs_committed_install_final": inst_l2,
            "behavioral_gm12": free_post["gm12"],
            "behavioral_gm12_committed": G1C_POST_INSTALL["gm12"],
            "behavioral_abs_diff": abs(free_post["gm12"]
                                       - G1C_POST_INSTALL["gm12"]),
            "bar": 5e-3,
            "pass": bool(inst_l2 < 5e-3
                         and abs(free_post["gm12"]
                                 - G1C_POST_INSTALL["gm12"]) < G_READ_TOL)},
        "root_band": {"g0": arms_rec["FREE"]["root"]["g0"],
                      "band": [G1C_ROOT_G0 * (1 - MATCH_BAND),
                               G1C_ROOT_G0 * (1 + MATCH_BAND)],
                      "pass": bool(arms_rec["FREE"]["root"]["landed"])},
        "cons_tracking": {"median_step_g0_abs_diff": cons_track_median,
                          "bar": 0.05,
                          "pass": bool(cons_track_median is not None
                                       and cons_track_median <= 0.05)},
        "w1_wash": None,       # tier (iv) DROPPED (no wash in this cell)
        "pass": bool(inst_l2 < 5e-3
                     and abs(free_post["gm12"]
                             - G1C_POST_INSTALL["gm12"]) < G_READ_TOL
                     and arms_rec["FREE"]["root"]["landed"]
                     and cons_track_median is not None
                     and cons_track_median <= 0.05),
    }
    metrics["gates"]["G_FREE"] = G_FREE
    log(f"G_FREE (tiers i-iii; iv dropped — no wash): install L2 "
        f"{inst_l2:.3e} (bar 5e-3), behavioral |d| "
        f"{abs(free_post['gm12'] - G1C_POST_INSTALL['gm12']):.1e}; root g0 "
        f"{arms_rec['FREE']['root']['g0']:.4f} in band "
        f"{arms_rec['FREE']['root']['landed']}; cons-tracking median "
        f"{cons_track_median}: "
        f"{'PASS (i-iii)' if G_FREE['pass'] else 'FAIL — THE CELL HALTS'}")
    write_partial("P4a G_FREE read (tiers i-iii)"
                  + ("" if G_FREE["pass"] else " — FAILED"))
    if not G_FREE["pass"] and not SMOKE:
        metrics["status"] = ("HALTED — G_FREE FAILED (the amended tiers; "
                             "nothing adjudicated)")
        write_partial("HALTED (G_FREE)")
        return 1

    # the FREE ledger cross-check vs e258's committed record (texture)
    free_vpre_med = arms_rec["FREE"]["install"]["ledger_v_excess_pre_median"]
    metrics["gates"]["G_FREE_XCHECK"] = {
        "form": "ARM-FREE's measured v-excess ledger median vs e258's "
                "committed FREE median (the same stream, fresh recompute) — "
                "a cross-cell instrument co-report, NOT a gate (cross-pass "
                "cuBLAS/atomics noise; e258's own inherited law)",
        "mine": free_vpre_med, "e258_committed": E258_FREE_VEXC_PRE_MEDIAN,
        "abs_diff": abs(free_vpre_med - E258_FREE_VEXC_PRE_MEDIAN),
        "pass": True,
    }
    log(f"FREE-ledger x-check vs e258: v-exc pre med {free_vpre_med:.2f} vs "
        f"committed {E258_FREE_VEXC_PRE_MEDIAN:.2f} (|d| "
        f"{abs(free_vpre_med - E258_FREE_VEXC_PRE_MEDIAN):.2f}; co-report)")

    # ---- THE RUNGS (ascending; the anchor LAST) --------------------------
    burst_cooldown("FREE -> the ladder")
    for k, _, _ in LADDER:
        run_arm(RUNG_NAMES[k])
        if RUNG_NAMES[k] != ARMS[-1]:
            burst_cooldown(f"{RUNG_NAMES[k]} -> next rung")

    # ---- G_ANCHOR: the top rung vs e260's committed RANDOM arm -----------
    anchor_arm = RUNG_NAMES[LADDER[-1][0]]
    if not SMOKE:
        a_post = arms_rec[anchor_arm]["install"]["post_cells"]["g0"]
        a_root_g0 = arms_rec[anchor_arm]["root"]["g0"]
        a_root_gm12 = arms_rec[anchor_arm]["root"]["gm12"]
        a_kept = arms_rec[anchor_arm]["install"]["ledger_kept_frac_median"]
        ck260 = torch.load(CKPT_DIR / E260_RANDOM_INST_CK, map_location="cpu",
                           weights_only=False)["model"]
        my_a = torch.load(CKPT_DIR / f"e261_{anchor_arm}_inst_resume.pt",
                          map_location="cpu", weights_only=False)["model"]
        anchor_l2 = float(np.sqrt(sum(float(((my_a[kk].float()
                                              - ck260[kk].float()) ** 2).sum())
                                      for kk in ck260 if kk in my_a)))
        del ck260, my_a
        G_ANCH = {
            "form": f"the {anchor_arm} rung re-runs e260's committed RANDOM "
                    f"arm (bit-identical room + stream): install-final L2 "
                    f"vs {E260_RANDOM_INST_CK} <= 5e-3; behavioral |d| <= "
                    f"{ANCHOR_BEHAV_TOL} on post g0 / root g0 / root g-12 "
                    f"(the disclosed session texture — cross-pass "
                    f"cuBLAS/atomics noise; e260's G_FREE matched g1c at "
                    f"L2 7.06e-4)",
            "install_l2_vs_e260": anchor_l2,
            "post_g0": {"mine": a_post, "e260": E260_RANDOM_POST_G0,
                        "abs_diff": abs(a_post - E260_RANDOM_POST_G0)},
            "root_g0": {"mine": a_root_g0, "e260": E260_RANDOM_ROOT_G0,
                        "abs_diff": abs(a_root_g0 - E260_RANDOM_ROOT_G0)},
            "root_gm12": {"mine": a_root_gm12, "e260": E260_RANDOM_ROOT_GM12,
                          "abs_diff": abs(a_root_gm12
                                          - E260_RANDOM_ROOT_GM12)},
            "kept_median": {"mine": a_kept, "e260": E260_RANDOM_KEPT_MED,
                            "abs_diff": abs(a_kept - E260_RANDOM_KEPT_MED)},
            "bars": {"l2": 5e-3, "behavior": ANCHOR_BEHAV_TOL},
            "pass": bool(anchor_l2 <= 5e-3
                         and abs(a_post - E260_RANDOM_POST_G0)
                         <= ANCHOR_BEHAV_TOL
                         and abs(a_root_g0 - E260_RANDOM_ROOT_G0)
                         <= ANCHOR_BEHAV_TOL
                         and abs(a_root_gm12 - E260_RANDOM_ROOT_GM12)
                         <= ANCHOR_BEHAV_TOL),
        }
    else:
        G_ANCH = {"form": "SMOKE: no committed full-run record at smoke k — "
                          "the anchor gate is VACUOUS (explicit pass, "
                          "disclosed)",
                  "pass": True, "vacuous": True}
    metrics["gates"]["G_ANCHOR"] = G_ANCH
    log(f"G_ANCHOR: {'SMOKE-vacuous' if SMOKE else f'install L2 {anchor_l2:.3e} (bar 5e-3), post g0 |d| {abs(a_post - E260_RANDOM_POST_G0):.4f}, root g0 |d| {abs(a_root_g0 - E260_RANDOM_ROOT_G0):.4f}, root g-12 |d| {abs(a_root_gm12 - E260_RANDOM_ROOT_GM12):.4f} (bars {ANCHOR_BEHAV_TOL})'}: "
        f"{'PASS' if G_ANCH['pass'] else 'FAIL — the TEXTURE branch'}")
    write_partial("P4b G_ANCHOR read (the top rung vs e260's committed arm)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    ladder_ks = [k for k, _, _ in LADDER]
    rung_arms = [RUNG_NAMES[k] for k in ladder_ks]
    band_lo_g0 = G1C_ROOT_G0 * (1 - MATCH_BAND)
    band_hi_g0 = G1C_ROOT_G0 * (1 + MATCH_BAND)
    post_curve = {RUNG_NAMES[k]: arms_rec[RUNG_NAMES[k]]["install"]
                  ["post_cells"]["g0"] for k in ladder_ks}
    root_curve = {RUNG_NAMES[k]: arms_rec[RUNG_NAMES[k]]["root"]["g0"]
                  for k in ladder_ks}
    kept_curve = {RUNG_NAMES[k]: arms_rec[RUNG_NAMES[k]]["install"]
                  ["ledger_kept_frac_median"] for k in ladder_ks}
    post_curve["FREE"] = arms_rec["FREE"]["install"]["post_cells"]["g0"]
    root_curve["FREE"] = arms_rec["FREE"]["root"]["g0"]

    pg = [post_curve[RUNG_NAMES[k]] for k in ladder_ks]
    rg = [root_curve[RUNG_NAMES[k]] for k in ladder_ks]
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
        if jump_fires and in_band and not gates_pass:
            why.append("gates")  # unreachable (branch order); guarded
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

    log("=" * 78)
    log(f"E261 VERDICT: {verdict}")
    log("  THE LADDER (post g0 -> root g0 | kept | in-band):")
    for k, p_, r_, kf in zip(ladder_ks, pg, rg,
                             [kept_curve[RUNG_NAMES[k]] for k in ladder_ks]):
        nm = RUNG_NAMES[k]
        mark = "IN-BAND" if band_lo_g0 <= r_ <= band_hi_g0 else (
            "OVER" if r_ > band_hi_g0 else "under")
        log(f"    k={k:7d} ({nm:6s}): post g0 {p_:.6f} -> root g0 "
            f"{r_:.4f} | kept {kf:.4f} | {mark}")
    log(f"    FREE (full space): post g0 {post_curve['FREE']:.4f} -> root "
        f"g0 {root_curve['FREE']:.4f} | kept 1.0 | the ceiling")
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
                           "(frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "post_g0_curve": {str(k): post_curve[RUNG_NAMES[k]]
                              for k in ladder_ks},
            "root_g0_curve": {str(k): root_curve[RUNG_NAMES[k]]
                              for k in ladder_ks},
            "kept_frac_curve": {str(k): kept_curve[RUNG_NAMES[k]]
                                for k in ladder_ks},
            "root_gm12_curve": {str(k): arms_rec[RUNG_NAMES[k]]["root"]["gm12"]
                                for k in ladder_ks},
            "FREE": {"post_g0": post_curve["FREE"],
                     "root_g0": root_curve["FREE"],
                     "root_gm12": arms_rec["FREE"]["root"]["gm12"]},
            "e246_context": {"rank": E246_ALIGNED_RANK,
                             "post_g0": E246_ALIGNED_POST_G0},
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
            "landed": {a: arms_rec[a]["root"]["landed"] for a in ARMS},
            "expresses": {a: arms_rec[a]["root"]["expresses"] for a in ARMS},
            "anchor_arm": anchor_arm,
        },
        "SHARP_THRESHOLD": sharp_fires,
        "GRADUAL": gradual_fires,
        "MIXED": bool(gates_pass and not sharp_fires and not gradual_fires),
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars adjudicated")

    # ================= P8: the figures ====================================
    make_ladder_plot(RD, arms_rec, ladder_ks, pg, rg, kept_curve, verdict,
                     clause, in_band, threshold_k, max_ratio_row, jump_fires)
    make_instrument_plot(RD, rooms, arms_rec, cert, thermal_log, verdict,
                         ladder_ks)

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("all arms share bit-identical install "
            "streams (one generator, seed 24314, one draw order), identical "
            "dose/schedule/optimizer, the same fresh fact-free base; the "
            "ONLY delta across rungs is the pre-Adam gradient PROJECTION "
            "onto that rung's independent random rank-k room (each room "
            "certified: idempotency ~1e-15, kept^2 == k/N at the 10-sigma "
            "bar); the top rung's room is BIT-IDENTICAL to e260's committed "
            "RANDOM room (D/S exact vs e260_rooms.pt) and re-runs that arm "
            "under the anchor gate; G_FREE validates the instrumented path "
            "against the committed g1c install + root — fate differences "
            "across rungs are rank-caused or nothing is"),
        "n_and_scope": ("n=1 per rung, one lineage, one session (the "
            "g-series standing caveat — the critic's lottery note carried "
            "verbatim); the curve's SHAPE is the registered object, not any "
            "single point; the bracket between adjacent rungs is the "
            "honest resolution of the threshold's location"),
        "loads_measured_not_nominal": ("every rung's ACTUAL geometry is "
            "reported: the per-step kept fraction (the dose actually "
            "delivered; median over the install ledger), v-excess pre/post "
            "(vs e258's LOADED committed v-map), in-span fraction pre AND "
            "applied (vs e246's committed LATE span), and the displacement "
            "cos-to-span + in-own-room fractions at post-install and root "
            "— never nominal"),
        "the_confound_disclosed": ("the kept-k coupling: the projection's "
            "norm is NOT rescaled (e237/e260's convention), so the "
            "delivered dose covaries with the rung (kept ~ sqrt(k/N): 0.019 "
            "at 1k -> 0.294 at 237k) — an inherent property of a rank-k "
            "restriction, disclosed on every rung and co-plotted on the "
            "instrument page (root g0 vs kept); e258's committed VLIGHT "
            "LANDDED at kept ~0.10, so kept alone does not bar landing, "
            "and the record says so"),
        "the_scope_disclosed": ("the rungs restrict the GRADIENT, not the "
            "state: Adam's per-coordinate normalizer + the natural cons "
            "grow the state off-room (e260's measured root in-own-room "
            "0.659; this cell's per-rung in-own-room ledger carries the "
            "same read) — 'rank-k room' names the writing channel, and "
            "the displacement ledger says how much of the landing lived "
            "inside it"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all three branches and "
            "the curves are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e260_rooms": f"runs/checkpoints/{ROOMS260_CK}",
            "e260_random_inst_resume": f"runs/checkpoints/"
                                       f"{E260_RANDOM_INST_CK}",
            "rooms": metrics["rooms"]["checkpoint"],
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS},
            "arm_inst_resumes": {
                a: f"runs/checkpoints/"
                  f"{'smoke_' if SMOKE else ''}e261_{a}_inst_resume.pt"
                for a in ARMS},
        },
        "machinery": {
            "install_cons": "g1c's chunked drivers VERBATIM arithmetic "
                            "(E43.exposure Dmix / e113 consolidate) via "
                            "e246/e258/e260's port, with the dense-room "
                            "hook + this cell's thermal envelope; NO wash "
                            "driver in this cell (the registered readouts "
                            "are the curves)",
            "hook": "e237's pre-Adam projection as a DENSE ROOM projector "
                    "(backward -> clip 1.0 -> project (CPU fp64 SRCT; "
                    "write fp32) -> step; norm not rescaled; fp64 ledger "
                    "dots)",
            "rooms": "one INDEPENDENT SRCT room per rung (a seeded random "
                     "+-1 isometry x a random k-subset of the orthonormal "
                     "DCT-II basis; exact projector, never materialized); "
                     "seeds registered per rung (26111-26118; the anchor "
                     "rung = e260's committed 26011/26012, bit-identity "
                     "gated)",
            "vmap": "e258's committed 2.74M v-map LOADED from "
                    "e258_vmap.pt (md5 recorded; k-bound) — the measured "
                    "v-excess ledger only; no new history run",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": {
            "burst_cap_s": BURST_MAX_S, "cooldown_s": COOLDOWN_S,
            "per_step_polls": True, "early_end_margin_c": TEMP_EARLY_END,
            "hard_line_c": TEMP_HARD,
            "max_temp_seen_c": max((r["temp"] for r in thermal_log),
                                   default=None),
            "violations_ge_84c": sum(1 for r in thermal_log
                                     if r["temp"] >= TEMP_HARD),
        },
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e261_rank_ladder.png"),
                          str(RD / "e261_ladder_instrument.png")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_ladder_plot(rd, arms_rec, ladder_ks, pg, rg, kept_curve, verdict,
                     clause, in_band, threshold_k, max_ratio_row, jump_fires):
    """THE CURVES FIGURE: the expression curve (post g0 vs k), the landing
    curve (root g0 vs k, the band + the threshold marked), the dose
    covariate (kept vs k; root g0 vs kept — the confound view), and the
    verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    N = GB.G1B_PARAMS
    rung_cols = plt.cm.viridis(np.linspace(0.25, 0.95, len(ladder_ks)))

    # (0,0) THE EXPRESSION CURVE (post-install g0 vs k)
    ax = axes[0, 0]
    ax.plot(ladder_ks, pg, "o-", ms=8, lw=2.0, color="#1a6faf", alpha=0.95,
            label="the rungs (post-install g0)")
    for k, v, c in zip(ladder_ks, pg, rung_cols):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=c)
    ax.plot([N], [post_g0_free := arms_rec["FREE"]["install"]["post_cells"]
                  ["g0"]], "*", ms=17, color="#e67e22",
            label=f"FREE (full space, k=N: post g0 {post_g0_free:.3f})")
    ax.plot([E246_ALIGNED_RANK], [E246_ALIGNED_POST_G0], "x", ms=9, mew=2.2,
            color="darkred",
            label=f"e246 ALIGNED (rank 10, post g0 "
                  f"{E246_ALIGNED_POST_G0:.1e}) — the dead context point")
    if not SMOKE:
        ax.plot([E258_K_HARD], [E260_RANDOM_POST_G0], "o", ms=13,
                mfc="none", mec="k", mew=1.6,
                label=f"e260's committed RANDOM (post g0 "
                      f"{E260_RANDOM_POST_G0:.3f})")
    ax.axhline(G0_ZERO_FLOOR, ls=":", lw=1.4, color="crimson",
               label=f"the expression floor ({G0_ZERO_FLOOR})")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log; SRCT random dense rooms)")
    ax.set_ylabel("post-install g0 (the expression ruler)")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25, which="both")
    ax.set_title(f"THE EXPRESSION CURVE — max adjacent jump "
                 f"{max_ratio_row['ratio']:.1f}x (k="
                 f"{max_ratio_row['from_k']}->{max_ratio_row['to_k']}; the "
                 f">10x bar {'FIRES' if jump_fires else 'does not fire'})",
                 fontsize=9.5)

    # (0,1) THE LANDING CURVE (root g0 vs k, the band + the threshold)
    ax = axes[0, 1]
    ax.plot(ladder_ks, rg, "o-", ms=8, lw=2.0, color="#1a6faf", alpha=0.95,
            label="the rungs (root g0)")
    for k, v, c in zip(ladder_ks, rg, rung_cols):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=c)
    ax.plot([N], [root_g0_free := arms_rec["FREE"]["root"]["g0"]], "*",
            ms=17, color="#e67e22",
            label=f"FREE (full space: root g0 {root_g0_free:.3f}) — the "
                  f"ceiling")
    if not SMOKE:
        ax.plot([E258_K_HARD], [E260_RANDOM_ROOT_G0], "o", ms=13,
                mfc="none", mec="k", mew=1.6,
                label=f"e260's committed RANDOM (root g0 "
                      f"{E260_RANDOM_ROOT_G0:.3f})")
    ax.axhspan(G1C_ROOT_G0 * (1 - MATCH_BAND), G1C_ROOT_G0 * (1 + MATCH_BAND),
               color="#b8d8f0", alpha=0.35, zorder=0,
               label=f"the matched band (+-{MATCH_BAND:.0%} of the committed "
                     f"root g0 {G1C_ROOT_G0:.3f})")
    ax.axhline(G1C_ROOT_G0, color="k", ls=":", lw=0.9)
    if threshold_k is not None:
        ax.axvline(threshold_k, color="seagreen", ls="--", lw=1.8, alpha=0.8)
        ax.annotate(f"THRESHOLD\nk={threshold_k}", (threshold_k, 0.06),
                    fontsize=8, color="seagreen", ha="right",
                    weight="bold")
        if max(in_band) != threshold_k:
            for kk in in_band:
                ax.annotate("in-band", (kk, G1C_ROOT_G0 * (1 + MATCH_BAND)),
                            fontsize=6.5, color="seagreen", ha="center",
                            va="bottom")
    else:
        ax.annotate("no rung enters the band", (0.97, 0.05),
                    xycoords="axes fraction", fontsize=8, color="crimson",
                    ha="right", style="italic")
    ax.set_xscale("log")
    ax.set_xlabel("room rank k (log)")
    ax.set_ylabel("ROOT g0 (post install + cons)")
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25, which="both")
    ax.set_title("THE LANDING CURVE — where root g0 enters the band "
                 f"(in-band rungs: {in_band if in_band else 'NONE'})",
                 fontsize=9.5)

    # (1,0) THE DOSE COVARIATE (kept vs k + root g0 vs kept — the confound)
    ax = axes[1, 0]
    kpts = [kept_curve[RUNG_NAMES[k]] for k in ladder_ks]
    exp_pts = [math.sqrt(k / N) for k in ladder_ks]
    ax.plot(ladder_ks, kpts, "o-", ms=7, lw=1.8, color="#8e44ad",
            label="kept ||g'||/||g|| (ledger median)")
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
    ax.set_title("THE KEPT-K COUPLING (dose honesty: kept ~ sqrt(k/N); "
                 "e258's VLIGHT landed at kept ~0.10)", fontsize=9.5)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E261 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=94, break_long_words=False)[:11]:
        ax.text(0.02, y, wd, fontsize=6.8, va="top", family="monospace")
        y -= 0.024
    y -= 0.012
    gates_txt = "  ".join(
        f"{g}={'PASS' if v.get('pass') else 'FAIL'}"
        for g, v in metrics["gates"].items() if isinstance(v, dict))
    for wd in textwrap.wrap("GATES: " + gates_txt, width=96)[:2]:
        ax.text(0.02, y, wd, fontsize=6.4, va="top", family="monospace")
        y -= 0.02
    fig.suptitle("E261 — THE RANK/DOSE LADDER: post g0 + root g0 vs k over "
                 f"{{{', '.join(str(k) for k in ladder_ks)}}} + FREE (full "
                 f"space) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e261_rank_ladder.png", dpi=130)
    plt.close(fig)


def make_instrument_plot(rd, rooms: LadderRooms, arms_rec, cert, thermal_log,
                         verdict, ladder_ks):
    """THE INSTRUMENT FIGURE: the per-rung certification, the kept-fraction
    ledgers through the installs, the measured per-arm loads, and the
    thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    N = rooms.n
    rung_cols = plt.cm.viridis(np.linspace(0.25, 0.95, len(ladder_ks)))

    # (0,0) THE ROOMS' CERTIFICATION (per rung, log scale)
    ax = axes[0, 0]
    names = [RUNG_NAMES[k] for k in ladder_ks]
    idem = [cert["per_rung"][nm]["idempotency_max"] for nm in names]
    kept_dev = [abs(cert["per_rung"][nm]["kept2_mean"]
                    - cert["per_rung"][nm]["kept2_expect"]) for nm in names]
    bars_ = [max(v, 1e-18) for v in idem]
    ax.bar([f"{n}\nidem" for n in names], bars_, color=rung_cols, alpha=0.85)
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
    ax.set_title("THE LADDER'S CERTIFICATION (idem shown; " + kt + ")",
                 fontsize=7.6)

    # (0,1) THE KEPT-FRACTION LEDGERS (per arm through the install)
    ax = axes[0, 1]
    for a in ARMS:
        led = arms_rec[a]["install"]["ledger"]
        xs_l = sorted(int(s) for s in led)
        get = lambda s: led[s] if s in led else led[str(s)]
        ax.plot(xs_l, [get(s)["kept_frac"] for s in xs_l], "o-", ms=3.2,
                lw=1.3, alpha=0.9,
                color="#e67e22" if a == "FREE" else rung_cols[
                    [RUNG_NAMES[k] for k in ladder_ks].index(a)],
                label=f"ARM-{a} ||g'||/||g|| (med "
                      f"{arms_rec[a]['install']['ledger_kept_frac_median']:.3f})")
    for k, c in zip(ladder_ks, rung_cols):
        ax.axhline(math.sqrt(k / N), ls=":", lw=0.8, color=c, alpha=0.5)
    ax.set_xlabel("install step")
    ax.set_ylabel("the kept-gradient fraction (dose honesty)")
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25)
    ax.set_title("THE KEPT-FRACTION LEDGERS (dotted: each rung's sqrt(k/N) "
                 "expectation)", fontsize=9.5)

    # (1,0) THE MEASURED PER-ARM LOADS
    ax = axes[1, 0]
    names = list(ARMS)
    w = 0.2
    xs = np.arange(len(names))
    vpost = [arms_rec[a]["install"]["ledger_v_excess_post_median"]
             or 0 for a in names]
    vpre = [arms_rec[a]["install"]["ledger_v_excess_pre_median"]
            or 0 for a in names]
    ior = [arms_rec[a]["root"]["displacement_loads"]["in_own_room"] or 0
           for a in names]
    cts = [arms_rec[a]["root"]["displacement_loads"]["cos_to_span"]
           for a in names]
    ax.bar(xs - w, vpre, width=w, color="#7fb3d5", alpha=0.9,
           label="install |g| v-excess (pre)")
    ax.bar(xs, vpost, width=w, color="#1a6faf", alpha=0.9,
           label="install |g'| v-excess (applied)")
    ax.bar(xs + w, ior, width=w, color="#8e44ad", alpha=0.8,
           label="root displacement in-own-room")
    ax.bar(xs + 2 * w, cts, width=w, color="#c0392b", alpha=0.55,
           label="root displacement cos-to-span")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=20)
    ax.set_ylabel("measured load / fraction")
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE MEASURED PER-ARM LOADS (never nominal; v-map loaded "
                 "from e258; span from e246)", fontsize=9.5)

    # (1,1) THE THERMAL ENVELOPE
    ax = axes[1, 1]
    if thermal_log:
        ax.plot([r["t"] for r in thermal_log], [r["temp"] for r in thermal_log],
                "-", lw=0.8, color="dimgray", alpha=0.7)
    ax.axhline(TEMP_EARLY_END, color="crimson", ls=":", lw=1.0,
               label=f"burst-end margin {TEMP_EARLY_END:.0f}C")
    ax.axhline(TEMP_HARD, color="crimson", ls="--", lw=1.2,
               label=f"never-past line {TEMP_HARD:.0f}C")
    ax.set_xlabel("run seconds")
    ax.set_ylabel("GPU temp (C) per-step polls")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C; violations "
                 f"{sum(1 for r in thermal_log if r['temp'] >= TEMP_HARD)})",
                 fontsize=9.5)

    fig.suptitle("E261 — the instrument page: the ladder's rooms, the "
                 f"certification, the kept ledgers, the measured loads "
                 f"-> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e261_ladder_instrument.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
