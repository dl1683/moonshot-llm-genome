"""E327 — THE SEED-STRATIFIED MODE CENSUS (R73's cascade pick; the gate on
"rare", e326's premise, and the laws-v3 drafting). This docstring carries the
registered question + background + design + bars + P-e327a VERBATIM from the
dispatch letter, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (dispatch verbatim): "THIS CENSUS decides: bimodal vs wide, the
s1 gap's emptiness, and the die-mode's population share."

BACKGROUND (dispatch verbatim): "e324 (runs/e324/) found the install wheel
BIMODAL at n=4 consecutive seeds (32401-32404, all survive-and-sprawl: s1 g0
~0.745, post 0.48-0.53, norm ~27, in-room 0.632) + the committed canon gen
(24314, die-then-recover: s1 1.35e-5, post 0.2646, norm 9.18, in-room 0.944).
R73's critic: n=4 consecutive seeds cannot establish bimodality, zero test
mass probed the s1 gap's interior, and 'the canon is RARE' is anti-licensed
(the trust band that selected it admits die-mode and excludes survive-mode).
THIS CENSUS decides: bimodal vs wide, the s1 gap's emptiness, and the
die-mode's population share."

THE DESIGN (dispatch verbatim):
  "1. N=8 FRESH INSTALL GENS at MAXIMALLY NON-CONSECUTIVE seeds (e.g. spread
  across the full 32-bit range with stated derivation — kill the seed-block
  objection) through the committed e261/e264 install rig VERBATIM at the
  HELD room (K10K, seeds 26113/26114, bit-bound), 400 Dmix steps each, no
  shopping, no redraws.
  2. Per draw, the census axes: s1 g0 (the fork), post g0 at its battery,
  write norm, in-own-room fraction (e324's exact conventions; the canon's
  row read from artifacts, never rerun).
  3. THE ANALYSIS: the s1 distribution across 8+1 draws (does anything land
  INSIDE the [0.01, 0.15] gap?); the die-mode count (the share estimate);
  the within-mode spreads; the canon's position in the enlarged census."

THE SEED DERIVATION (stated at birth, as the dispatch demands): THE EIGHT
GENS := (2i+1) * 2^28 for i = 0..7 — the midpoints of eight EQUAL bins
spanning the full 32-bit unsigned range [0, 2^32 - 1]:
    268435456,  805306368, 1342177280, 1879048192,
   2415919104, 2952790016, 3489660928, 4026531840.
Every adjacent pair is 2^29 = 536,870,912 apart (maximally non-consecutive:
no two share a bit-28-and-below tail, and the block sits at 16 bits'
distance from e324's 32401-32404 and every family allocation). Verified at
birth: the eight are distinct; none equals the committed 24314; none appears
in any family allocation (24314, 26011/26012, 26111-26118, 27215/27216,
27301, 28301, 28401, 28501, 28801, 29001, 10901, 1337, 32301-32325,
32401-32404); an EXACT-TOKEN grep (boundary-delimited) of lab/*.py +
runs/*/metrics.json for each of the eight returned ZERO hits; and
torch.Generator().manual_seed accepts the full range (exercised at birth:
each seed draws cleanly; the committed 24314 path unchanged).

FROZEN BARS (dispatch verbatim; adjudicated against exactly this):
  - BIMODAL-CONFIRMED: "zero draws land inside the s1 gap AND the draws
    split into the two committed textures (die: norm <= ~12, in-room >= 0.9;
    survive: norm >= ~24, in-room ~0.63) with within-mode spreads <= ~15% —
    the wheel is bimodal; the die-mode share gets its first estimate;
    e326's substrate comparison licenses; the v3 drafting's mode riders
    draft as written."
  - WIDE-WHEEL: ">= 2 draws land inside the gap or the textures interleave —
    the 'modes' were small-sample artifacts; every mode claim re-opens; the
    v3 drafting's formation riders hold until a bigger census."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * THE S1 GAP := [0.01, 0.15] INCLUSIVE (e324's fork convention: DIE side
    s1 < 0.01, SURVIVE side s1 > 0.15). A draw "lands inside the gap" iff
    0.01 <= s1_g0 <= 0.15.
  * THE CENSUS SET := the 8 fresh draws + the canon (9 draws). The census
    member's side: DIE / SURVIVE / GAP by its s1.
  * THE TEXTURE TEMPLATES (the dispatch's parenthetical made operational,
    frozen): DIE-TEXTURE := write_norm <= 12.0 AND in_own_room >= 0.90
    (the committed canon: 9.1788, 0.9442); SURVIVE-TEXTURE :=
    write_norm >= 24.0 AND in_own_room in [0.53, 0.73] (the dispatch's
    "~0.63" as 0.63 +- 0.10; the committed survive mass spans 0.6319-0.6339
    across five draws in two rooms — the band is deliberately generous);
    any other (norm, in-room) pair := NEITHER.
  * CONCORDANCE: a non-gap census draw is CONCORDANT iff its side's texture
    template matches its measured bulky texture (DIE side <-> DIE-TEXTURE;
    SURVIVE side <-> SURVIVE-TEXTURE). "THE TEXTURES INTERLEAVE" (the
    WIDE-WHEEL trigger's second clause) := at least one non-gap census draw
    is NOT concordant (its texture is NEITHER, or disagrees with its side).
    The canon is held to the same rule (it is a DIE-side draw and must sit
    inside the DIE-TEXTURE box).
  * WITHIN-MODE SPREAD := (max/min - 1), computed PER MODE on the mode's
    census members, SEPARATELY on write norm and on in-own-room (the two
    axes the texture templates name). The <= ~15% bar := <= 0.15 on BOTH
    axes for EVERY mode holding >= 2 census members; a mode with < 2
    members is VACUOUS (disclosed as n<2, unbounded). The g0 / s1 / gm12
    within-mode spreads are CO-REPORTED as context (they inform the v3
    error bars) but are NOT bar axes — the dispatch's textures are
    (norm, in-room) pairs.
  * THE DIE-MODE SHARE := (count of DIE-side draws among the EIGHT FRESH
    draws) / 8. The canon is EXCLUDED from the share's denominator (it was
    committed-SELECTED through the [0.15, 0.45] trust band, not sampled —
    including it would bias the share toward die; R73's own point). The
    binomial band := the two-sided 95% Clopper-Pearson exact interval on
    Binomial(8, p), computed with integer-exact arithmetic; the 9-draw
    table (canon included) co-reported.
  * THE CANON'S POSITION := gen 24314's rank among the 9 census points on
    each axis (g0, write norm, in-own-room, s1), reported per axis.
  * COMPOSITE ORDER: TEXTURE (any hard-gate failure) -> WIDE-WHEEL (>= 2
    draws inside the gap OR any non-concordant draw) -> BIMODAL-CONFIRMED
    (zero in gap AND all concordant AND within-mode spreads within bar) ->
    the named residues below.
  * THE NAMED RESIDUES (no bar fires; named at birth so the honest
    in-between is not shopped later): ONE-IN-GAP := exactly 1 census draw
    inside the s1 gap with every non-gap draw concordant (spread breaches,
    if any, disclosed inside its clause); SPREADY-MODES := zero in gap +
    all concordant but some within-mode spread > 15% (the modes are real
    but wider than e324's consecutive block suggested).
  * e324's four draws are CO-PLOTTED as context (prior census mass at the
    same room) and NEVER bar inputs — the bars read THIS census's 9 draws
    only.
  * THE HELD ROOM := the committed K10K room rebuilt from its registered
    seeds 26113/26114 (e261's registration; e264's committed record) and
    BIT-BOUND against runs/checkpoints/e264_rooms.pt's stored K10K D/S
    (exact array equality; file md5 hard-bound) — the same room the canon
    and e324's four draws formed in. Certified (G_PROJ) as always.
  * THE RIG := e261's chunked_install BY IMPORT, driven VERBATIM: 400 Dmix
    steps (ix(16) install w/ name mask + aj(16) paired + rj(32) random;
    masked token-level union CE; AdamW (0.9,0.95) wd 0.1; lr 1e-3 x
    cosine(step-1,1000); clip 1.0 -> P_room(g) fp64 CPU, write fp32), from
    the SAME committed root (g1c_root.pt), the SAME banks. The ONLY delta
    across the eight draws is the Dmix generator seed (E261.FRESH_GEN
    rebound per draw). NO consolidation phase (e324's disclosed convention:
    every census axis is an install-phase quantity).
  * THE CANON IS NOT RERUN: gen 24314's census row is READ from the
    artifacts at runtime (e264's metrics: post g0/gm12/gp12 + the install
    traj's s1 + in-own-room; e290's metrics: the write norm; e272's
    metrics: the K10KR s1, the die's SECOND-ROOM confirmation) and asserted
    against the birth literals (Rule 12). e324's four-draw record is
    likewise READ at runtime and asserted (G_E324READ). Never retyped into
    the census.

REGISTERED PREDICTION (registered at birth, before compute):
  - P-e327a (the dispatch's lab lean, ADOPTED — with my own read stated):
    BIMODAL-CONFIRMED, with the die-mode share expected SMALL — point
    prediction: 0 or 1 of the 8 fresh draws land DIE (share <= 12.5%).
    Basis: the survive mode's s1 is nearly deterministic (the four e324
    draws span 0.0004) and GEN-tracked (the fork is the draw's first-step
    geometry: gen 24314 died at s1 in BOTH rooms; every fresh gen survived
    in the canon's own room); five independent survive draws across two
    rooms with zero gap-interior mass. My own read of the two-sided risks:
    (i) the selection-band argument cuts both ways — the [0.15, 0.45] band
    that selected the canon ADMITS die-mode post-g0 (0.2646) and EXCLUDES
    survive-mode (0.48-0.53), so committed draws were die-biased BY
    SELECTION; that is evidence about the band, not the wheel, and it
    leaves the share genuinely open (0/5 vs 1/1 both consistent at this n);
    (ii) the consecutive-seed objection is exactly what this census kills —
    if seeds 32401-32404 were secretly correlated, the spread seeds could
    widen the survive cluster's g0 beyond its 12.2% (that alone does not
    fall the bar: the spread axes are norm + in-room) or drop a draw into
    the gap (that does). If the census lands ONE-IN-GAP, SPREADY-MODES or
    WIDE-WHEEL, P-e327a is scored a MISS and the verdict stands on the
    bars. (The eight-miss ledger's lesson is applied: this registers the
    LEAN the record supports, not a guess at the mechanism's seat.)

GATES (a failure HALTS the record as TEXTURE): {G_VOCAB, G_NAMEFREE,
G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_PARENTS,
G_CANONREAD, G_E324READ, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ,
G_ROOMHELD, G_GENFRESH, G_FACTLOAD, G_PROTOIDENT} — G_PROTOIDENT +
G_FACTLOAD instantiated PER DRAW (all eight must pass).

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (dispatch 180), per-step thermal polls at a 78C margin, 40s
cooldowns (dispatch 30-60), the 84C never-past line (dispatch 85), polls
persisted to runs/_envelope_log.jsonl tagged e327:<GEN>:chunkN; gpu_ok()
at startup; the eight installs SEQUENTIAL — never concurrent; CPU fp64
dense projections (pocketfft), CPU threads 4 (the x26 CPU agent is live —
normal CPU sharing only).

Outputs: runs/e327/{metrics.json (PROGRESSIVE), e327_mode_census.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e327_gen*.pt + e327_*_resume.pt (gitignored; md5s in
metrics). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat
folds). Commit + push per phase.

Run:  cd lab && python e327_mode_census.py    (E327_SMOKE=1 shakedown)
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
                                                      # COMMITTED INSTALL RIG,
                                                      # PORTED WHOLE BY IMPORT

torch.set_num_threads(4)           # shared machine (the x26 CPU agent live)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E327_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e327_smoke" if SMOKE else "e327"
assert torch.cuda.is_available(), "e327 owns the GPU lane (dispatch)"
assert common.gpu_ok(), "gpu_ok() gate at startup (dispatch: check BEFORE launch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE REBINDING (the family's disclosed convention): AFTER the e261
# import, so the burst machinery + thermal polls label THIS cell (tags
# e327:...). The e261 import's own append handle on runs/e261/run.log is
# opened but never written through (e323/e324's convention; disclosed).
G1.log = log                                          # unify the timeline
device_events: list[dict] = []
thermal_log: list[dict] = []
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
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (THE held room's record)
CKPT_DIR = GB.CKPT_DIR

# ---- THE HELD ROOM (the canon's own room; the census's fixed frame) ----
HELD_ROOM_SEEDS = (26113, 26114)      # e261's registered K10K pair (e264's rung)
ROOM_K = 10_000 if not SMOKE else 512
ROOM_MODE = "K10K"                    # the committed rung's own mode key
LADDER: tuple[tuple[int, int, int], ...] = (
    (ROOM_K, HELD_ROOM_SEEDS[0], HELD_ROOM_SEEDS[1]),)
E261.LADDER = LADDER                  # the machinery's certify()/rooms read it
E261.RUNG_NAMES = {ROOM_K: ROOM_MODE}

# ---- THE EIGHT FRESH INSTALL DRAWS (derivation in the docstring) -------
FRESH_GENS: tuple[int, ...] = tuple((2 * i + 1) * (1 << 28) for i in range(8))
#     = (268435456, 805306368, 1342177280, 1879048192,
#        2415919104, 2952790016, 3489660928, 4026531840)
COMMITTED_INST_GEN = 24314            # the g1c draw (e261's FRESH_GEN)
E324_GENS = (32401, 32402, 32403, 32404)   # e324's consecutive block (context)
FAMILY_ALLOCATIONS = frozenset(
    [24314, 26011, 26012, 26111, 26112, 26113, 26114, 26115, 26116,
     26117, 26118, 27215, 27216, 27301, 28301, 28401, 28501, 28801,
     29001, 10901, 1337]
    + list(range(32301, 32326))       # e323's whole allocation block
    + list(E324_GENS))                # e324's block

# ---- THE FROZEN BAR NUMBERS -------------------------------------------
S1_DIE_BAR = 0.01                     # s1 g0 < this := DIE side (e324's fork)
S1_GAP_HI = 0.15                      # s1 g0 > this := SURVIVE side
DIE_TEX_NORM_MAX = 12.0               # DIE-TEXTURE: norm <= ~12 (frozen 12)
DIE_TEX_INROOM_MIN = 0.90             # DIE-TEXTURE: in-room >= 0.9
SURV_TEX_NORM_MIN = 24.0              # SURVIVE-TEXTURE: norm >= ~24 (frozen 24)
SURV_TEX_INROOM_BAND = (0.53, 0.73)   # SURVIVE-TEXTURE: in-room ~0.63 (0.63+-0.10)
MODE_SPREAD_BAR = 0.15                # within-mode spread <= ~15% (frozen 0.15)
WIDE_GAP_COUNT_BAR = 2                # >= 2 draws inside the gap := WIDE-WHEEL
RATIO_DEN_FLOOR = 1e-6                # e261's jump-ratio denominator floor

# ---- THE PARENT RECORDS, HARD-BOUND (read at runtime; Rule 12). The JSON
# records are bound on their GIT-CANONICAL md5s (newline-normalized bytes —
# the OneDrive CRLF convention, e290's repair); the .pt artifacts RAW.
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "a1aee7420d08bb257db5af5f45625f6f"          # verified at birth
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "1149f97c633072c3b865b6a93bfa1e44"          # verified at birth
E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "856d3f0da57f4a6b5f7f033d130e9a85"          # verified at birth
E290_METRICS = E43.REPO / "runs" / "e290" / "metrics.json"
E290_MD5 = "fc2ba5767e65090e09278e8b55454ef0"          # verified at birth
E324_METRICS = E43.REPO / "runs" / "e324" / "metrics.json"
E324_MD5 = "67ea86fd3474587d25fd3bba43b0fd8a"          # verified at birth
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"      # verified at birth

# ---- THE CANON'S CENSUS LITERALS (typed at birth from the artifacts ONLY
# to cross-assert the runtime reads — the census row itself uses the
# RUNTIME-READ artifact values, never these literals; G_CANONREAD).
CANON_POST_G0 = 0.26464763283729553            # e264 K10K install post g0
CANON_POST_GM12 = 0.10525520890951157          # e264 K10K install post gm12
CANON_POST_GP12 = 0.11036071181297302          # e264 K10K install post gp12
CANON_S1_G0 = 1.3466433301800862e-05           # e264 K10K install traj s1
CANON_IN_OWN_ROOM = 0.9441588788935659         # e264 K10K install displ.
CANON_WRITE_NORM = 9.1788432658723             # e290 G_FACTLOAD write norm
E272_K10KR_POST_G0 = 0.2097209095954895        # the room-lottery replicate
E272_K10KR_S1_G0 = 1.3466953532770276e-05      # the die's 2nd-room confirm

# ---- e324's CENSUS LITERALS (the parent's four consecutive draws; typed
# at birth from runs/e324/metrics.json ONLY to cross-assert the runtime
# reads; G_E324READ. They are CONTEXT (co-plotted, never bar inputs).
E324_AXES_LITERALS = {
    "post_g0": {32401: 0.5349767208099365, 32402: 0.5200912356376648,
                32403: 0.47680848836898804, 32404: 0.5079231858253479},
    "write_norm": {32401: 27.028411030988025, 32402: 27.082530679359728,
                   32403: 27.086985923693728, 32404: 27.034233378293848},
    "in_own_room": {32401: 0.6319414419445407, 32402: 0.6337978576435206,
                    32403: 0.6339092796377008, 32404: 0.6319980996502677},
    "s1_g0": {32401: 0.7449015974998474, 32402: 0.7451865077018738,
              32403: 0.7452579140663147, 32404: 0.7449929118156433},
}

VOCAB_EXPECT = 65
G_READ_TOL = E261.G_READ_TOL             # 5e-3

REGISTERED = {
    "question_verbatim": (
        "THIS CENSUS decides: bimodal vs wide, the s1 gap's emptiness, and "
        "the die-mode's population share."),
    "background_verbatim": (
        "e324 (runs/e324/) found the install wheel BIMODAL at n=4 "
        "consecutive seeds (32401-32404, all survive-and-sprawl: s1 g0 "
        "~0.745, post 0.48-0.53, norm ~27, in-room 0.632) + the committed "
        "canon gen (24314, die-then-recover: s1 1.35e-5, post 0.2646, norm "
        "9.18, in-room 0.944). R73's critic: n=4 consecutive seeds cannot "
        "establish bimodality, zero test mass probed the s1 gap's interior, "
        "and 'the canon is RARE' is anti-licensed (the trust band that "
        "selected it admits die-mode and excludes survive-mode). THIS "
        "CENSUS decides: bimodal vs wide, the s1 gap's emptiness, and the "
        "die-mode's population share."),
    "design_verbatim": (
        "1. N=8 FRESH INSTALL GENS at MAXIMALLY NON-CONSECUTIVE seeds "
        "(e.g. spread across the full 32-bit range with stated derivation "
        "— kill the seed-block objection) through the committed e261/e264 "
        "install rig VERBATIM at the HELD room (K10K, seeds 26113/26114, "
        "bit-bound), 400 Dmix steps each, no shopping, no redraws. "
        "2. Per draw, the census axes: s1 g0 (the fork), post g0 at its "
        "battery, write norm, in-own-room fraction (e324's exact "
        "conventions; the canon's row read from artifacts, never rerun). "
        "3. THE ANALYSIS: the s1 distribution across 8+1 draws (does "
        "anything land INSIDE the [0.01, 0.15] gap?); the die-mode count "
        "(the share estimate); the within-mode spreads; the canon's "
        "position in the enlarged census."),
    "bars_verbatim": {
        "BIMODAL-CONFIRMED": (
            "zero draws land inside the s1 gap AND the draws split into "
            "the two committed textures (die: norm <= ~12, in-room >= 0.9; "
            "survive: norm >= ~24, in-room ~0.63) with within-mode spreads "
            "<= ~15% — the wheel is bimodal; the die-mode share gets its "
            "first estimate; e326's substrate comparison licenses; the v3 "
            "drafting's mode riders draft as written."),
        "WIDE-WHEEL": (
            ">= 2 draws land inside the gap or the textures interleave — "
            "the 'modes' were small-sample artifacts; every mode claim "
            "re-opens; the v3 drafting's formation riders hold until a "
            "bigger census."),
    },
    "operationalizations": (
        "frozen BEFORE compute: the s1 gap := [0.01, 0.15] INCLUSIVE (DIE "
        "side s1 < 0.01, SURVIVE side s1 > 0.15 — e324's fork convention); "
        "the census set := the 8 fresh draws + the canon (9); DIE-TEXTURE "
        ":=" " write_norm <= 12.0 AND in_own_room >= 0.90; SURVIVE-TEXTURE "
        ":= write_norm >= 24.0 AND in_own_room in [0.53, 0.73] (the "
        "dispatch's ~0.63 as 0.63+-0.10); any other (norm, in-room) := "
        "NEITHER; CONCORDANT := the non-gap draw's side's template matches "
        "its measured texture; INTERLEAVE := >= 1 non-concordant census "
        "draw (NEITHER counts); WITHIN-MODE SPREAD := (max/min - 1) per "
        "mode on the mode's census members, separately on write norm and "
        "in-own-room, the bar <= 0.15 on BOTH axes for every mode with >= "
        "2 members (n<2 vacuous, disclosed); g0/s1/gm12 within-mode "
        "spreads co-reported as context only; the die-mode share := "
        "DIE-count among the 8 FRESH draws / 8 (the canon EXCLUDED — "
        "committed-selected, not sampled; disclosed) with the two-sided 95% "
        "Clopper-Pearson exact band; the canon's position := its rank "
        "among the 9 census points per axis; COMPOSITE: TEXTURE -> "
        "WIDE-WHEEL (>= 2 in gap OR interleave) -> BIMODAL-CONFIRMED (zero "
        "in gap AND all concordant AND spreads within bar) -> the named "
        "residues ONE-IN-GAP (exactly 1 in gap, non-gap all concordant) / "
        "SPREADY-MODES (zero in gap, all concordant, a spread > 15%); "
        "e324's four draws co-plotted as CONTEXT, never bar inputs; the "
        "held room rebuilt from seeds 26113/26114 and BIT-BOUND vs "
        "e264_rooms.pt's K10K D/S; the rig = e261's chunked_install BY "
        "IMPORT with E261.FRESH_GEN rebound per draw — the ONLY delta; NO "
        "consolidation (e324's convention: census axes are install-phase "
        "quantities); NO formation gate, NO redraws, no shopping; the "
        "canon NOT rerun — its row READ from e264/e290/e272's artifacts at "
        "runtime + asserted vs the birth literals; e324's record READ at "
        "runtime + asserted (G_E324READ)"),
    "registration": (
        "bars + question + design + operationalizations frozen VERBATIM "
        "from the dispatch letter (R73's cascade pick; the gate on 'rare', "
        "e326's premise, and the laws-v3 drafting); this script committed "
        "at birth BEFORE any compute; adjudicate against exactly this; no "
        "bar shopping."),
    "predictions": {
        "P-e327a_bimodal_small_die_share": (
            "ADOPTED from the dispatch with my own read stated: "
            "BIMODAL-CONFIRMED, die-mode share SMALL — point prediction 0-1 "
            "of the 8 fresh draws DIE (share <= 12.5%). Basis: survive s1 "
            "nearly deterministic (span 0.0004) + gen-tracked (24314 died "
            "at s1 in BOTH rooms; five independent survive draws across "
            "two rooms, zero gap mass). Risks owned: (i) the selection "
            "band admits die-mode post-g0 and excludes survive-mode — "
            "committed draws were die-biased BY SELECTION (evidence about "
            "the band, not the wheel; the share stays genuinely open); "
            "(ii) if 32401-32404 were secretly seed-correlated, the spread "
            "seeds could widen the survive cluster's g0 (not a bar axis) "
            "or drop a draw into the gap (a bar axis). ONE-IN-GAP / "
            "SPREADY-MODES / WIDE-WHEEL each score P-e327a a MISS."),
    },
}

deviations: list[str] = [
    "THE EIGHT INSTALLS RUN THROUGH e261's chunked_install BY IMPORT (the "
    "committed rig VERBATIM — extend, don't repeat): the module-global "
    "FRESH_GEN is rebound per draw to the spread seeds; the room ladder is "
    "the HELD pair 26113/26114; the step body (draws, clip, the room hook, "
    "the milestone battery) is the committed file's, NOT modified.",
    "THE SEED BLOCK IS SPREAD, NOT CONSECUTIVE (the dispatch's own design): "
    "gens (2i+1)*2^28, i=0..7 — eight equal-bin midpoints across the full "
    "32-bit range, adjacent spacing 2^29. This kills R73's seed-block "
    "objection by construction: no adjacency, no shared prefix structure, "
    "16 bits away from every prior allocation.",
    "NO CONSOLIDATION PHASE (e324's disclosed convention, carried): every "
    "census axis — s1 g0, post g0, write norm, in-own-room (gm12 "
    "co-reported as context) — is an INSTALL-phase quantity.",
    "THE CANON IS NOT RERUN and e324 IS NOT RERUN: both records READ from "
    "their artifacts at runtime + asserted against birth literals "
    "(G_CANONREAD / G_E324READ); e324's four draws co-plotted as context, "
    "never bar inputs.",
    "THE DIE-SHARE DENOMINATOR EXCLUDES THE CANON (disclosed at birth): "
    "gen 24314 was committed-SELECTED through the [0.15, 0.45] trust band "
    "(die-admitting), not sampled — including it would bias the share "
    "toward die. The 9-draw table co-reports it.",
    "n=8 fresh draws, one lineage, one room, one session (the g-series "
    "standing lottery caveat carried verbatim): an 8+1 census prices the "
    "two modes' EXISTENCE and coarse share; the tails and the room x gen "
    "interaction remain single-room-anchored; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).",
    "Smoke mode (E327_SMOKE=1): 8-step installs, room at k=512 (the "
    "bit-bind vs e264_rooms.pt's k=10k record is VACUOUS at smoke — "
    "disclosed, LIVE and binding at the full run), all paths "
    "smoke_-prefixed, own smoke dir; NOTHING adjudicated (SMOKE stamp on "
    "every read).",
]

# ------------------------------------------------------------------ helpers
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


def md5of_norm(p: Path) -> str:
    """md5 of the NEWLINE-NORMALIZED bytes (CRLF -> LF) — the git canonical
    form (e290's OneDrive CRLF repair; JSON parents bound on this hash)."""
    return hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def _bisect_p(fn, target: float, lo: float = 0.0, hi: float = 1.0,
              iters: int = 200) -> float:
    """Bisection on a DECREASING fn over [0,1] (fn(0) > target > fn(1)):
    returns the p where fn crosses target downward. The binomial CDF
    P(X<=x | p) is DECREASING in p for fixed x — the helper's direction is
    matched to both Clopper-Pearson solvers (unit-checked at birth:
    k=0/n=8 -> [0, 0.3694]; k=1 -> [0.0032, 0.5265]; k=4 -> [0.157, 0.843])."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial CI (Clopper-Pearson), integer-exact CDF via
    math.comb + bisection (n is tiny here; no scipy dependency)."""
    def cdf(p: float, x: int) -> float:      # P(X <= x), X ~ Bin(n, p);
        return sum(math.comb(n, i) * p ** i  # decreasing in p for fixed x
                   * (1.0 - p) ** (n - i)
                   for i in range(x + 1))
    if k <= 0:
        lower = 0.0
    else:                                    # solve P(X <= k-1) = 1-a/2
        lower = _bisect_p(lambda p: cdf(p, k - 1), 1.0 - alpha / 2.0)
    if k >= n:
        upper = 1.0
    else:                                    # solve P(X <= k) = a/2
        upper = _bisect_p(lambda p: cdf(p, k), alpha / 2.0)
    return lower, upper


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
           "per_step_polls": "after EVERY opt step (all draws) — aggregated "
                             "from runs/_envelope_log.jsonl (tags e327:*)",
           "early_end_margin_c": E261.TEMP_EARLY_END,
           "hard_line_c": E261.TEMP_HARD,
           "dispatch_envelope": "bursts <= 180s, cooldowns 30-60s, never "
                                "past 85C — this cell runs 175/40/84 (all "
                                "inside); gpu_ok() checked at startup"}
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
    return out


def _side_of(s1: float) -> str:
    if s1 < S1_DIE_BAR:
        return "DIE"
    if s1 > S1_GAP_HI:
        return "SURVIVE"
    return "GAP"


def _texture_of(norm: float, inroom: float) -> str:
    if norm <= DIE_TEX_NORM_MAX and inroom >= DIE_TEX_INROOM_MIN:
        return "DIE"
    if (norm >= SURV_TEX_NORM_MIN
            and SURV_TEX_INROOM_BAND[0] <= inroom <= SURV_TEX_INROOM_BAND[1]):
        return "SURVIVE"
    return "NEITHER"


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e327_mode_census",
        "phase": "THE SEED-STRATIFIED MODE CENSUS — the install wheel's "
                 "bimodality tested at maximally non-consecutive seeds (R73's "
                 "cascade pick; the gate on 'rare', e326's premise, and the "
                 "laws-v3 drafting): N=8 spread gens ((2i+1)*2^28) through "
                 "the committed e261/e264 rig VERBATIM at the HELD K10K room "
                 "(seeds 26113/26114, bit-bound), 400 Dmix steps each, no "
                 "shopping; the census axes s1 g0 / post g0 / write norm / "
                 "in-own-room; the canon from artifacts, never rerun; "
                 "BIMODAL-CONFIRMED vs WIDE-WHEEL on the frozen bars",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; the "
                      "eight installs SEQUENTIAL with cooldowns — never "
                      "concurrent) + CPU fp64 dense projections (pocketfft), "
                      "CPU threads 4 (the x26 CPU agent live — shared)",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e327:<GEN>:chunkN; gpu_ok() at startup",
            "trainings": "8 installs x 400 Dmix steps (the committed rig's "
                         "own budget), one per spread gen",
        },
        "the_held_room": {
            "seeds": list(HELD_ROOM_SEEDS),
            "mode": ROOM_MODE,
            "k": ROOM_K,
            "k_fraction_of_N": ROOM_K / 2739072,
            "derivation": "e261's registered K10K rung pair — the canon's "
                          "own writing room (e264's committed K10K install; "
                          "e324's four consecutive draws); bit-bound vs "
                          "runs/checkpoints/e264_rooms.pt",
        },
        "the_fresh_gens": {
            "gens": list(FRESH_GENS),
            "derivation": "gens := (2i+1)*2^28, i=0..7 — the midpoints of "
                          "eight EQUAL bins spanning the full 32-bit range "
                          "[0, 2^32-1]; adjacent spacing 2^29 = 536,870,912 "
                          "(maximally non-consecutive); verified distinct + "
                          "exact-token absent from lab/*.py + "
                          "runs/*/metrics.json at birth",
            "committed_inst_gen": COMMITTED_INST_GEN,
            "e324_context_gens": list(E324_GENS),
        },
        "deviations": deviations,
        "builds_on": [
            "R73 (the frontier review that picked this cascade: the bimodal "
            "'rare' word pulled back pending THIS census; the seed-block + "
            "zero-gap-mass objections this cell kills)",
            "T291 / e324 (THE PARENT CENSUS: the wheel read BIMODAL at n=4 "
            "consecutive seeds — 4 survive + the canon's die; S1-FORK-CLEARS "
            "gen-tracked; survive s1 span 0.0004)",
            "T288 / e323 (the die-then-recover vs survive-and-sprawl fork's "
            "discovery; the +81% install-draw miss)",
            "T239 / e261 + e264 (THE INSTALL RIG: chunked_install VERBATIM "
            "by import + the committed K10K room's seeds + the canon's own "
            "record)",
            "T242 / e272 (RANK-WRITES-THE-CURVE: the die's second-room "
            "confirmation — the fork is gen-tracked, not room-tracked)",
            "T269 / e290 (the committed write norm's own record — the "
            "canon's norm read from its G_FACTLOAD)",
        ],
        "whats_new": [
            "THE S1 GAP'S INTERIOR PROBED FOR THE FIRST TIME (the record's "
            "first): zero prior test mass landed inside [0.01, 0.15] — "
            "eight spread draws + the canon decide its emptiness",
            "THE SEED-BLOCK OBJECTION KILLED (the record's first): eight "
            "gens at 2^29 spacing across the full 32-bit range — no "
            "consecutive-seed correlation can survive this design",
            "THE DIE-MODE'S FIRST POPULATION SHARE (the record's first): "
            "k/8 over unconditioned spread draws with the exact binomial "
            "band (the canon excluded as committed-selected)",
            "THE MODES' WITHIN-MODE SPREADS AT SPREAD SEEDS (the record's "
            "first): do the two committed textures hold their widths when "
            "the block structure is removed — the v3 drafting's error bars "
            "and e326's substrate license hang on this",
        ],
        "gates": {},
    })
    log(f"E327 — THE SEED-STRATIFIED MODE CENSUS (smoke={SMOKE}) -> {RD}")
    log(f"held room: {ROOM_MODE} k={ROOM_K} seeds {HELD_ROOM_SEEDS}; the "
        f"eight spread gens {FRESH_GENS} (committed {COMMITTED_INST_GEN}); "
        f"bars: zero-in-gap + all-concordant + spreads <= "
        f"{MODE_SPREAD_BAR:.0%} -> BIMODAL-CONFIRMED; >= "
        f"{WIDE_GAP_COUNT_BAR} in gap or interleave -> WIDE-WHEEL; the "
        f"residues ONE-IN-GAP / SPREADY-MODES named at birth")
    write_partial("startup (bars + design + P-e327a registered, committed "
                  "at birth)")
    set_seed(327)                    # global init only; every RNG is its own

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
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    # the install bank (e261's build_win VERBATIM)
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])
    win_bank = torch.stack([build_win(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_bank[i][G1.PRE: G1.PRE + len(G1.NAME)]) == G1.NAME
        for i in range(win_bank.shape[0]))
    G_NAMEWIN = {"n_windows": int(win_bank.shape[0]),
                 "masked_decode_all_name": bool(win_masked_ok),
                 "pass": bool(win_masked_ok
                              and list(win_bank.shape) == [60, G1.BLOCK])}
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"

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
                                 "n_windows": 16, "starts": n_starts},
                "pass": bool(len(n_starts) == 16)}
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_VOCAB": G_VOCAB, "G_NAMEFREE": G_NAMEFREE,
                             "G_SPLICE": G_SPLICE, "G_BATTERY": G_BATTERY,
                             "G_ANCHOR": G_ANCHOR, "G_INSTMASK": G_INSTMASK,
                             "G_NAMEWIN": G_NAMEWIN})
    log("P0: protocol gates PASS (vocab 65 / namefree / splice 19+41 / "
        "battery shapes / namewin / e170 bank / install mask)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e272m = json.loads(E272_METRICS.read_text(encoding="utf-8"))
    e290m = json.loads(E290_METRICS.read_text(encoding="utf-8"))
    e324m = json.loads(E324_METRICS.read_text(encoding="utf-8"))
    e261m = json.loads(E261_METRICS.read_text(encoding="utf-8"))

    def _pbind(path, bound, extra=None):
        return {"path": str(path), "md5": md5of_norm(path),
                "raw_worktree_md5": md5of(path), "bound_md5": bound,
                **(extra or {})}

    # ---- THE CANON'S ROW: READ from the artifacts at runtime -----------
    canon_inst = e264m["arms"]["K10K"]["install"]
    canon_s1_row = next(t for t in canon_inst["traj"] if t["step"] == 1)
    canon_row = {
        "gen": COMMITTED_INST_GEN,
        "role": "THE CANON — the committed draw (room 26113/26114, the "
                "held room itself); NOT rerun; every value READ from the "
                "committed artifacts at runtime (e264 post_cells + traj; "
                "e290 G_FACTLOAD write norm; e272's K10KR s1 as the die's "
                "second-room confirmation)",
        "post_g0": canon_inst["post_cells"]["g0"],
        "post_gm12": canon_inst["post_cells"]["gm12"],
        "post_gp12": canon_inst["post_cells"]["gp12"],
        "ce_r": canon_inst["post_cells"]["ce_r"],
        "write_norm": e290m["gates"]["G_FACTLOAD"]["write_norm"],
        "in_own_room": canon_inst["displacement_loads"]["in_own_room"],
        "s1_g0": canon_s1_row["g0_pz"],
        "s1_argmax": canon_s1_row["g0_argmax"],
        "s1_second_room_e272": next(
            t for t in e272m["arms"]["K10KR"]["install"]["traj"]
            if t["step"] == 1)["g0_pz"],
        "sources": {"post_cells+traj+in_own_room":
                        "runs/e264/metrics.json arms.K10K.install",
                    "write_norm": "runs/e290/metrics.json "
                                  "gates.G_FACTLOAD",
                    "s1_2nd_room": "runs/e272/metrics.json "
                                   "arms.K10KR.install.traj"},
    }
    G_CANONREAD = {
        "form": "the canon's census row is READ from the artifacts (never "
                "rerun, never retyped) and cross-asserted against the birth "
                "literals (Rule 12 — catches artifact drift in either "
                "direction)",
        "runtime_reads": {k: canon_row[k] for k in
                          ("post_g0", "post_gm12", "post_gp12",
                           "write_norm", "in_own_room", "s1_g0",
                           "s1_second_room_e272")},
        "birth_literals": {"post_g0": CANON_POST_G0,
                           "post_gm12": CANON_POST_GM12,
                           "post_gp12": CANON_POST_GP12,
                           "write_norm": CANON_WRITE_NORM,
                           "in_own_room": CANON_IN_OWN_ROOM,
                           "s1_g0": CANON_S1_G0,
                           "s1_second_room_e272": E272_K10KR_S1_G0},
        "pass": bool(
            abs(canon_row["post_g0"] - CANON_POST_G0) < 1e-15
            and abs(canon_row["post_gm12"] - CANON_POST_GM12) < 1e-15
            and abs(canon_row["post_gp12"] - CANON_POST_GP12) < 1e-15
            and abs(canon_row["write_norm"] - CANON_WRITE_NORM) < 1e-12
            and abs(canon_row["in_own_room"] - CANON_IN_OWN_ROOM) < 1e-15
            and abs(canon_row["s1_g0"] - CANON_S1_G0) < 1e-18
            and abs(canon_row["s1_second_room_e272"] - E272_K10KR_S1_G0)
            < 1e-18),
    }
    assert G_CANONREAD["pass"], f"canon re-read gate FAILED: {G_CANONREAD}"
    metrics["gates"]["G_CANONREAD"] = G_CANONREAD
    log(f"P0b G_CANONREAD: the canon (gen {COMMITTED_INST_GEN}) re-verified "
        f"from artifacts: g0 {canon_row['post_g0']:.10f} norm "
        f"{canon_row['write_norm']:.4f} in-room "
        f"{canon_row['in_own_room']:.4f} s1 {canon_row['s1_g0']:.2e} "
        f"(2nd room {canon_row['s1_second_room_e272']:.2e}) — PASS")

    # ---- e324's four-draw record: READ at runtime + asserted ------------
    e324_axes = e324m["the_census"]["axes_table"]
    e324_reads = {
        axis: {g: e324_axes[axis][f"gen{g}"] for g in E324_GENS}
        for axis in ("post_g0", "write_norm", "in_own_room", "s1_g0")}
    e324_reads["canon_s1"] = e324_axes["s1_g0"]["canon"]
    _ok_e324 = all(
        abs(e324_reads[axis][g] - E324_AXES_LITERALS[axis][g]) < 1e-12
        for axis in E324_AXES_LITERALS for g in E324_GENS)
    G_E324READ = {
        "form": "e324's four-draw record READ from runs/e324/metrics.json "
                "at runtime and cross-asserted against the birth literals "
                "(Rule 12); the four draws are CONTEXT (co-plotted), never "
                "bar inputs",
        "runtime_reads": e324_reads,
        "verdict_then": e324m["adjudication"]["verdict"],
        "s1_fork_then": e324m["the_census"]["s1_fork"]["finding"],
        "pass": bool(_ok_e324
                     and e324m["adjudication"]["verdict"] == "ROOM-COMPARABLE"
                     and e324m["the_census"]["s1_fork"]["finding"]
                     == "S1-FORK-CLEARS"
                     and abs(e324_reads["canon_s1"] - CANON_S1_G0) < 1e-18),
    }
    assert G_E324READ["pass"], f"e324 re-read gate FAILED: {G_E324READ}"
    metrics["gates"]["G_E324READ"] = G_E324READ
    log("P0b G_E324READ: e324's four draws re-verified from artifacts "
        "(s1 " + " ".join(f"{e324_reads['s1_g0'][g]:.4f}"
                          for g in E324_GENS)
        + f"; verdict then {G_E324READ['verdict_then']}, fork "
        f"{G_E324READ['s1_fork_then']}): PASS")

    # e272's room-lottery numbers, read at runtime (the comparator)
    e272_k10kr_g0 = e272m["arms"]["K10KR"]["install"]["post_cells"]["g0"]

    G_PARENTS = {
        "e261_metrics": _pbind(E261_METRICS, E261_MD5,
                               {"note": "THE RIG (chunked_install — "
                                        "imported + driven by THIS cell)"}),
        "e264_metrics": _pbind(
            E264_METRICS, E264_MD5,
            {"verdict": e264m["adjudication"]["verdict"],
             "K10K_post_g0": canon_row["post_g0"],
             "note": "the canon's own record (the held room's committed "
                     "install; the census's canon row)"}),
        "e272_metrics": _pbind(
            E272_METRICS, E272_MD5,
            {"K10KR_post_g0": e272_k10kr_g0,
             "K10KR_s1": canon_row["s1_second_room_e272"],
             "note": "the room-lottery precedent + the die's second-room "
                     "confirmation (the fork's gen-tracking evidence)"}),
        "e290_metrics": _pbind(
            E290_METRICS, E290_MD5,
            {"write_norm": canon_row["write_norm"],
             "note": "the canon's write norm's own record"}),
        "e324_metrics": _pbind(
            E324_METRICS, E324_MD5,
            {"verdict": e324m["adjudication"]["verdict"],
             "s1_fork": e324m["the_census"]["s1_fork"]["finding"],
             "note": "THE PARENT CENSUS (n=4 consecutive seeds: 4 survive "
                     "+ canon die) — the bimodality claim THIS cell tests "
                     "at spread seeds; its four draws are this census's "
                     "context mass"}),
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS264_CK),
                       "bound_md5": ROOMS264_MD5,
                       "note": "THE HELD ROOM's committed record (the "
                               "bit-bind target, G_ROOMHELD)"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "pass": bool(
            e264m["adjudication"]["verdict"] == "SHARP-THRESHOLD"
            and e272m["adjudication"]["verdict"] == "RANK-WRITES-THE-CURVE"
            and e324m["adjudication"]["verdict"] == "ROOM-COMPARABLE"
            and abs(e272_k10kr_g0 - E272_K10KR_POST_G0) < 1e-15
            and md5of_norm(E261_METRICS) == E261_MD5
            and md5of_norm(E264_METRICS) == E264_MD5
            and md5of_norm(E272_METRICS) == E272_MD5
            and md5of_norm(E290_METRICS) == E290_MD5
            and md5of_norm(E324_METRICS) == E324_MD5
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b G_PARENTS PASS — e261/e264/e272/e290/e324 metrics + the room "
        "artifacts bound (canon g0 "
        f"{canon_row['post_g0']:.6f}; e272 K10KR {e272_k10kr_g0:.6f})")
    write_partial("P0b parents hard-bound + the canon + e324's record "
                  "re-verified from artifacts")
    del e264m, e272m, e290m, e324m, e261m

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

    # ================= P1: THE ROOT + THE HELD ROOM =====================
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

    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))
    base_flat_np = base_flat.double().numpy().astype(np.float64)

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}",
        "md5": md5of(CKPT_DIR / VMAP_CK),
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
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": f"the HELD rank-{ROOM_K} room certified (fp64 CPU, "
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

    # ---- G_ROOMHELD: the room is THE CANON'S OWN (bit-bound) -----------
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    D_mine = rooms.rooms[ROOM_MODE].D
    S_mine = rooms.rooms[ROOM_MODE].S
    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"])
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    seeds_equal = (LADDER[0][1], LADDER[0][2]) == HELD_ROOM_SEEDS \
        == tuple(rooms264["model"]["K10K"]["seeds"])
    D_bit = bool(np.array_equal(D_mine.astype(np.int8), D264.astype(np.int8)))
    S_bit = bool(np.array_equal(S_mine.astype(np.int64),
                                S264.astype(np.int64)))
    k_match = bool(int(rooms264["model"]["K10K"]["k"]) == ROOM_K
                   and ROOM_K == 10_000)
    G_ROOMHELD = {
        "form": "the writing room is THE CANON'S OWN: rebuilt from the "
                f"registered seeds {HELD_ROOM_SEEDS} and BIT-BOUND against "
                "e264_rooms.pt's committed K10K record (the +-1 diagonal D "
                "AND the kept-index set S exactly equal — the same "
                "subspace the canon and e324's draws formed in, not a "
                "re-hash); certified in G_PROJ",
        "seeds_match_e264_record": bool(seeds_equal),
        "D_bit_equal": D_bit,
        "S_bit_equal": S_bit,
        "k_match": k_match,
        "rooms_file_md5": md5of(CKPT_DIR / ROOMS264_CK),
        "pass": bool(seeds_equal and D_bit and S_bit and k_match),
    }
    if SMOKE:
        G_ROOMHELD["pass"] = True
        G_ROOMHELD["vacuous"] = (
            "SMOKE: the bit-bind is VACUOUS at smoke k=512 (the committed "
            "record is k=10k; the smoke room shares only the seeds) — LIVE "
            "and binding at the full run's k=10k")
    del rooms264
    assert G_ROOMHELD["pass"], f"held-room gate FAILED: {G_ROOMHELD}"
    metrics["gates"]["G_ROOMHELD"] = G_ROOMHELD
    log(f"P1 G_ROOMHELD: the room (seeds {HELD_ROOM_SEEDS}) is the canon's "
        f"own — D bit {D_bit}, S bit {S_bit}, k {ROOM_K}: "
        f"{'PASS' if not SMOKE else 'PASS (SMOKE-vacuous)'}")
    write_partial("P1 the held room built + certified + bit-bound")

    # ---- G_GENFRESH: the eight draws are FRESH + MAXIMALLY SPREAD -------
    seed_bits = [int(s).bit_length() for s in FRESH_GENS]
    G_GENFRESH = {
        "form": "the eight install gens are a FRESH SPREAD block: distinct, "
                "not the committed 24314, absent from every family "
                "allocation, exact-token absent from lab/*.py + "
                "runs/*/metrics.json (grepped at birth; the grep itself is "
                "a static disclosure recorded here, not a runtime gate), "
                "and maximally non-consecutive by derivation — (2i+1)*2^28, "
                "i=0..7, the eight equal-bin midpoints of the full 32-bit "
                "range, adjacent spacing 2^29 = 536,870,912 (R73's "
                "seed-block objection killed by construction)",
        "gens": list(FRESH_GENS),
        "derivation": "(2*i+1) * 2**28 for i in 0..7",
        "min_adjacent_spacing": int(2 << 28),
        "all_distinct": bool(len(set(FRESH_GENS)) == len(FRESH_GENS)),
        "all_inside_32bit": bool(all(0 <= s < (1 << 32) for s in FRESH_GENS)),
        "bit_lengths": seed_bits,
        "none_committed": bool(COMMITTED_INST_GEN not in FRESH_GENS),
        "none_in_family_allocations": bool(
            not (set(FRESH_GENS) & FAMILY_ALLOCATIONS)),
        "birth_grep": "exact-token (boundary-delimited) grep for each of "
                      "the eight gens over lab/*.py + runs/*/metrics.json "
                      "returned ZERO hits at birth (2026-10-09, pre-compute)",
        "pass": bool(len(set(FRESH_GENS)) == len(FRESH_GENS)
                     and COMMITTED_INST_GEN not in FRESH_GENS
                     and not (set(FRESH_GENS) & FAMILY_ALLOCATIONS)
                     and all(0 <= s < (1 << 32) for s in FRESH_GENS)),
    }
    assert G_GENFRESH["pass"], f"fresh-gen gate FAILED: {G_GENFRESH}"
    metrics["gates"]["G_GENFRESH"] = G_GENFRESH
    log(f"P1 G_GENFRESH: {len(FRESH_GENS)} spread gens (spacing 2^29, "
        f"bit-lengths {min(seed_bits)}-{max(seed_bits)}), distinct, not "
        f"24314, absent from the family's allocations: PASS")
    write_partial("P1b G_GENFRESH PASSED")

    # ================= P2: THE EIGHT INSTALLS (the committed rig) =======
    census: dict[int, dict] = {}
    metrics["draws"] = {}
    expected_steps = E261.INST_STEPS
    for i_gen, gen in enumerate(FRESH_GENS):
        if i_gen:
            E261.burst_cooldown(f"GEN{FRESH_GENS[i_gen - 1]} -> GEN{gen}")
        log("=" * 78)
        log(f"GEN{gen} — DRAW {i_gen + 1}/{len(FRESH_GENS)}: e261's "
            f"chunked_install VERBATIM from the committed root, the HELD "
            f"room {ROOM_MODE}, the Dmix generator FRESH (seed {gen} vs "
            f"the committed {COMMITTED_INST_GEN}) — the draw's ONLY delta")
        E261.FRESH_GEN = int(gen)          # THE REBINDING (the fresh draw)
        resume_ck = CKPT_DIR / (f"smoke_e327_gen{gen}_resume.pt" if SMOKE
                                else f"e327_gen{gen}_resume.pt")
        out = E261.chunked_install(
            f"GEN{gen}", ROOM_MODE, root_net, rooms, win_bank, inst_mask,
            anchor_full, train_ids, g0_ids, gm12_ids, r_eval_xy, zid,
            resume_ck, dev)
        sd = out["sd"]
        net = G1.evl_load(sd)
        flat = flat_params_cpu(net)
        flat_np = flat.double().numpy().astype(np.float64)
        post_g0 = G1.battery_cell(net, g0_ids, zid)["mean_pz"]
        post_gm12 = G1.battery_cell(net, gm12_ids, zid)["mean_pz"]  # context
        ce_r = G1.ce_fixed_cpu(net, *r_eval_xy)
        write_norm = float(np.linalg.norm(flat_np - base_flat_np))
        loads = rooms.displacement_loads(flat - base_flat, ROOM_MODE)
        del net

        traj = out.get("traj", [])
        s1_row = next((t for t in traj if t["step"] == 1), None)
        assert s1_row is not None, f"GEN{gen}: no s1 row in the install traj"
        kept_med = med([v["kept_frac"] for v in out["ledger"].values()])

        # ---- G_FACTLOAD: read-determinism (re-load) --------------------
        net2 = G1.evl_load({k: v.clone() for k, v in sd.items()})
        flat2 = flat_params_cpu(net2)
        g0_2 = G1.battery_cell(net2, g0_ids, zid)["mean_pz"]
        del net2
        md5_1 = hashlib.md5(flat.numpy().tobytes()).hexdigest()
        md5_2 = hashlib.md5(flat2.numpy().tobytes()).hexdigest()
        G_FL = {
            "form": "the draw's read-determinism: the final state re-loaded "
                    "reproduces the flat-md5 and the g0 read EXACTLY",
            "flat_md5": md5_1, "flat_md5_reload": md5_2,
            "g0_reload": g0_2, "g0_abs_diff": abs(g0_2 - post_g0),
            "pass": bool(md5_1 == md5_2 and abs(g0_2 - post_g0) <= 2e-6),
        }
        assert G_FL["pass"], f"GEN{gen} G_FACTLOAD FAILED: {G_FL}"

        # ---- G_PROTOIDENT: the committed rig, unchanged -----------------
        res_state = torch.load(resume_ck, map_location="cpu",
                               weights_only=False)
        pg = res_state["opt"]["param_groups"][0]
        lr_final_expected = E43.LR * cosine_lr(int(res_state["step"]) - 1,
                                               E261.INST_TOTAL)
        G_PI = {
            "form": "per draw: the committed rig's protocol identity — the "
                    "machinery IS e261's chunked_install (import, not a "
                    "copy); INST_STEPS the committed 400 (smoke 8); lr 1e-3 "
                    "x house cosine(total=1000); AdamW (0.9,0.95) wd 0.1; "
                    "clip 1.0 then P_room; the Dmix arithmetic "
                    "(ix16+aj16+rj32, union CE) — verified from the run's "
                    "own resume artifact (the optimizer state the rig left) "
                    "+ static config asserts",
            "machinery_module": E261.chunked_install.__module__,
            "inst_steps": E261.INST_STEPS, "expected_steps": expected_steps,
            "steps_ran": int(out["steps_ran"]),
            "lr_literal": E43.LR, "inst_total": E261.INST_TOTAL,
            "corp_bs": E43.CORP_BS, "mix_random": E43.MIX_RANDOM,
            "name_bs": G1.NAME_BS,
            "opt_class_from_artifact": type(res_state["opt"]).__name__
                if not isinstance(res_state["opt"], dict) else "state_dict",
            "opt_betas_from_artifact": list(pg["betas"]),
            "opt_wd_from_artifact": pg["weight_decay"],
            "opt_lr_final_from_artifact": pg["lr"],
            "opt_lr_final_expected": lr_final_expected,
            "smoke_stamp": bool(SMOKE),
            "pass": bool(
                E261.chunked_install.__module__ == "e261_rank_ladder"
                and E43.LR == 1e-3 and E261.INST_TOTAL == 1000
                and E43.CORP_BS == 48 and E43.MIX_RANDOM == 32
                and G1.NAME_BS == 16
                and (SMOKE or E261.INST_STEPS == 400)
                and E261.INST_STEPS == expected_steps
                and int(out["steps_ran"]) == expected_steps
                and list(pg["betas"]) == [0.9, 0.95]
                and float(pg["weight_decay"]) == 0.1
                and abs(float(pg["lr"]) - lr_final_expected) < 1e-12),
        }
        assert G_PI["pass"], f"GEN{gen} G_PROTOIDENT FAILED: {G_PI}"

        ck = save_ckpt(f"e327_gen{gen}", sd,
                       {"desc": f"e327 census draw gen {gen}: the committed "
                                f"root + e261's install rig VERBATIM (400 "
                                f"Dmix steps, HELD room seeds "
                                f"{HELD_ROOM_SEEDS}, gen {gen} of the "
                                f"spread block)",
                        "gen": int(gen), "post_g0": post_g0,
                        "write_norm": write_norm,
                        "in_own_room": loads["in_own_room"],
                        "s1_g0": s1_row["g0_pz"]})
        row = {
            "gen": int(gen),
            "draw_index": i_gen + 1,
            "role": "FRESH SPREAD DRAW at the held room",
            "post_g0": post_g0, "post_gm12": post_gm12,
            "ce_r": ce_r,
            "write_norm": write_norm,
            "in_own_room": loads["in_own_room"],
            "v_excess": loads["v_excess"],
            "cos_to_span": loads["cos_to_span"],
            "s1_g0": s1_row["g0_pz"], "s1_argmax": s1_row["g0_argmax"],
            "flat_md5": md5_1,
            "kept_frac_median": kept_med,
            "checkpoint": ck,
            "traj": traj, "chunk_table": out.get("chunk_table", []),
            "resumed_final": bool(out.get("resumed_final", False)),
            "G_FACTLOAD": G_FL, "G_PROTOIDENT": G_PI,
        }
        census[gen] = row
        metrics["draws"][f"GEN{gen}"] = {k: v for k, v in row.items()
                                         if k != "chunk_table"}
        metrics["draws"][f"GEN{gen}"]["chunk_table"] = row["chunk_table"]
        log(f"GEN{gen} CENSUS ROW: s1 {s1_row['g0_pz']:.6f} "
            f"({_side_of(s1_row['g0_pz'])}) g0 {post_g0:.6f} norm "
            f"{write_norm:.4f} in-own-room {loads['in_own_room']:.4f} "
            f"kept-med {kept_med:.4f} | G_FACTLOAD PASS G_PROTOIDENT PASS")
        write_partial(f"GEN{gen} complete (census row {i_gen + 1}/8)")

    # ================= P3: THE ANALYSIS + THE FROZEN BARS ===============
    log("=" * 78)

    # ---- the 9-row census table (side + texture per draw) --------------
    census_rows = [{
        "gen": COMMITTED_INST_GEN, "label": "CANON",
        "s1_g0": canon_row["s1_g0"], "post_g0": canon_row["post_g0"],
        "write_norm": canon_row["write_norm"],
        "in_own_room": canon_row["in_own_room"],
        "post_gm12": canon_row["post_gm12"],
    }] + [{
        "gen": g, "label": f"FRESH{i + 1}",
        "s1_g0": census[g]["s1_g0"], "post_g0": census[g]["post_g0"],
        "write_norm": census[g]["write_norm"],
        "in_own_room": census[g]["in_own_room"],
        "post_gm12": census[g]["post_gm12"],
    } for i, g in enumerate(FRESH_GENS)]
    for r in census_rows:
        r["side"] = _side_of(r["s1_g0"])
        r["texture"] = _texture_of(r["write_norm"], r["in_own_room"])
        r["concordant"] = (r["side"] in ("DIE", "SURVIVE")
                           and r["texture"] == r["side"])

    in_gap = [r for r in census_rows if r["side"] == "GAP"]
    non_concordant = [r for r in census_rows
                      if r["side"] != "GAP" and not r["concordant"]]
    n_gap = len(in_gap)
    n_interleave = len(non_concordant)

    # ---- the modes + within-mode spreads --------------------------------
    die_members = [r for r in census_rows if r["side"] == "DIE"]
    surv_members = [r for r in census_rows if r["side"] == "SURVIVE"]

    def _spread(vals):
        if len(vals) < 2:
            return None
        return max(vals) / max(min(vals), RATIO_DEN_FLOOR) - 1.0

    mode_spreads = {}
    for mname, members in (("DIE", die_members), ("SURVIVE", surv_members)):
        mode_spreads[mname] = {
            "n_members": len(members),
            "members": [r["label"] for r in members],
            "write_norm_spread": _spread([r["write_norm"] for r in members]),
            "in_own_room_spread": _spread([r["in_own_room"]
                                           for r in members]),
            # context-only axes (frozen: not bar axes)
            "g0_spread_ctx": _spread([r["post_g0"] for r in members]),
            "s1_span_ctx": (None if len(members) < 2 else
                            max(r["s1_g0"] for r in members)
                            - min(r["s1_g0"] for r in members)),
            "gm12_spread_ctx": _spread([r["post_gm12"] for r in members]),
        }
    spreads_ok = all(
        (v["write_norm_spread"] is None
         or v["write_norm_spread"] <= MODE_SPREAD_BAR)
        and (v["in_own_room_spread"] is None
             or v["in_own_room_spread"] <= MODE_SPREAD_BAR)
        for v in mode_spreads.values())

    # ---- the die-mode share (fresh draws only; canon excluded) ----------
    k_die = sum(1 for g in FRESH_GENS if _side_of(census[g]["s1_g0"]) == "DIE")
    n_fresh = len(FRESH_GENS)
    share = k_die / n_fresh
    cp_lo, cp_hi = clopper_pearson(k_die, n_fresh)
    die_share = {
        "k_die_fresh": k_die, "n_fresh": n_fresh, "share": share,
        "clopper_pearson_95": [cp_lo, cp_hi],
        "canon_excluded_because": "committed-SELECTED through the "
                                  "[0.15, 0.45] trust band (die-admitting), "
                                  "not sampled",
        "with_canon_9draw_table": f"{k_die + 1} DIE / {n_fresh + 1} census "
                                  f"draws (the canon is the +1 DIE)",
    }

    # ---- the canon's position in the enlarged census --------------------
    def _rank(axis):
        ordered = sorted(r[axis] for r in census_rows)
        return ordered.index(canon_row_axis[axis]) + 1, len(ordered)
    canon_row_axis = {"post_g0": canon_row["post_g0"],
                      "write_norm": canon_row["write_norm"],
                      "in_own_room": canon_row["in_own_room"],
                      "s1_g0": canon_row["s1_g0"]}
    n9 = len(census_rows)
    canon_position = {}
    for axis in ("post_g0", "write_norm", "in_own_room", "s1_g0"):
        rank, _ = _rank(axis)
        pos = ("EXTREME (min)" if rank == 1
               else "EXTREME (max)" if rank == n9
               else "INTERIOR")
        canon_position[axis] = {"rank": rank, "of": n9, "position": pos}

    # ---- the frozen bars ------------------------------------------------
    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))
    all_concordant = (n_interleave == 0)

    if not gates_pass:
        verdict = "TEXTURE (GATE FAILURE: " + ", ".join(
            k for k, g in hard.items() if not g.get("pass")) + ")"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif n_gap >= WIDE_GAP_COUNT_BAR or n_interleave >= 1:
        verdict = "WIDE-WHEEL"
        fired = []
        if n_gap >= WIDE_GAP_COUNT_BAR:
            fired.append(f"{n_gap} census draws land INSIDE the [0.01, "
                         f"0.15] s1 gap ({[r['label'] for r in in_gap]})")
        if n_interleave >= 1:
            fired.append(f"{n_interleave} draw(s) interleave the textures "
                         f"({[(r['label'], r['side'], r['texture'])
                             for r in non_concordant]})")
        clause = (
            "the 'modes' were small-sample artifacts: " + "; ".join(fired)
            + " — every mode claim re-opens; the v3 drafting's formation "
            "riders hold until a bigger census; the die-share estimate is "
            "reported for the record but licenses nothing")
    elif n_gap == 0 and all_concordant and spreads_ok:
        verdict = "BIMODAL-CONFIRMED"
        clause = (
            f"zero of the {n9} census draws land inside the [0.01, 0.15] "
            f"gap; every draw splits into the two committed textures "
            f"({len(surv_members)} SURVIVE / {len(die_members)} DIE, all "
            f"concordant); within-mode spreads: DIE norm "
            f"{mode_spreads['DIE']['write_norm_spread'] if mode_spreads['DIE']['write_norm_spread'] is not None else 'n<2'}"
            f" / in-room "
            f"{mode_spreads['DIE']['in_own_room_spread'] if mode_spreads['DIE']['in_own_room_spread'] is not None else 'n<2'}"
            f", SURVIVE norm {mode_spreads['SURVIVE']['write_norm_spread']:.1%}"
            f" / in-room "
            f"{mode_spreads['SURVIVE']['in_own_room_spread']:.1%} (bar "
            f"<= {MODE_SPREAD_BAR:.0%}) — the wheel is BIMODAL at spread "
            f"seeds; the die-mode share gets its first estimate: "
            f"{k_die}/{n_fresh} = {share:.1%} (95% CP "
            f"[{cp_lo:.1%}, {cp_hi:.1%}]) over unconditioned draws; "
            "e326's substrate comparison licenses; the v3 drafting's mode "
            "riders draft as written")
    elif n_gap == 1:
        verdict = "ONE-IN-GAP"
        clause = (
            f"exactly one census draw ({in_gap[0]['label']}, gen "
            f"{in_gap[0]['gen']}, s1 {in_gap[0]['s1_g0']:.4f}) lands INSIDE "
            f"the [0.01, 0.15] gap — the named residue: neither bar fires; "
            f"the other {n9 - 1} draws' concordance: "
            f"{'all concordant' if all_concordant else 'interleaved'}; "
            f"spread breaches "
            f"{'none' if spreads_ok else 'PRESENT (see mode_spreads)'}; "
            "the honest in-between, disclosed as registered at birth")
    else:
        verdict = "SPREADY-MODES"
        breach = [f"{m}: norm {v['write_norm_spread']:.1%}, in-room "
                  f"{v['in_own_room_spread']:.1%}"
                  for m, v in mode_spreads.items()
                  if (v["write_norm_spread"] or 0) > MODE_SPREAD_BAR
                  or (v["in_own_room_spread"] or 0) > MODE_SPREAD_BAR]
        clause = (
            f"zero draws inside the gap and all draws concordant, BUT a "
            f"within-mode spread exceeds the {MODE_SPREAD_BAR:.0%} bar "
            f"({'; '.join(breach)}) — the modes are real but wider than "
            "e324's consecutive block suggested; the named residue: no bar "
            "fires; the v3 error bars draft with the wider spreads")

    # ---- prediction scoring ----------------------------------------------
    p_hit = bool(verdict == "BIMODAL-CONFIRMED" and k_die <= 1)

    # ---- the log block ----------------------------------------------------
    log("=" * 78)
    log(f"E327 VERDICT: {verdict}")
    log("  the s1 census: canon "
        f"{canon_row['s1_g0']:.2e} | fresh "
        + " ".join(f"{g}:{census[g]['s1_g0']:.4f}" for g in FRESH_GENS))
    log(f"  in-gap draws: {n_gap} ({[r['label'] for r in in_gap]}); "
        f"non-concordant: {n_interleave} "
        f"({[r['label'] for r in non_concordant]})")
    log(f"  die share (fresh only): {k_die}/{n_fresh} = {share:.1%} "
        f"(95% CP [{cp_lo:.1%}, {cp_hi:.1%}])")
    for m, v in mode_spreads.items():
        log(f"  mode {m}: n={v['n_members']} norm-spread "
            f"{v['write_norm_spread']} in-room-spread "
            f"{v['in_own_room_spread']} (g0 ctx {v['g0_spread_ctx']})")
    log(f"  canon position: " + "; ".join(
        f"{a} rank {c['rank']}/{c['of']} ({c['position']})"
        for a, c in canon_position.items()))
    log(f"  {clause}")
    log(f"  P-e327a (BIMODAL-CONFIRMED + die share small (<=1 of 8)): "
        f"{'HIT' if p_hit else 'MISS'}")
    log("=" * 78)

    metrics["the_census"] = {
        "canon": canon_row,
        "fresh": {f"GEN{g}": census[g] for g in FRESH_GENS},
        "census_rows": census_rows,
        "mode_spreads": mode_spreads,
        "die_share": die_share,
        "canon_position": canon_position,
        "e324_context": e324_reads,
        "bars_frozen": {
            "s1_gap": [S1_DIE_BAR, S1_GAP_HI],
            "die_texture": {"norm_max": DIE_TEX_NORM_MAX,
                            "inroom_min": DIE_TEX_INROOM_MIN},
            "survive_texture": {"norm_min": SURV_TEX_NORM_MIN,
                                "inroom_band": list(SURV_TEX_INROOM_BAND)},
            "mode_spread_bar": MODE_SPREAD_BAR,
            "wide_gap_count_bar": WIDE_GAP_COUNT_BAR,
        },
    }
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any hard-gate failure) -> WIDE-WHEEL "
                           "(>= 2 draws inside the gap OR any non-concordant "
                           "draw) -> BIMODAL-CONFIRMED (zero in gap AND all "
                           "concordant AND within-mode spreads <= 15%) -> "
                           "ONE-IN-GAP / SPREADY-MODES (the named residues) "
                           "— frozen at birth",
        "gates_pass": gates_pass,
        "reads": {
            "n_census": n9,
            "n_in_gap": n_gap,
            "in_gap_labels": [r["label"] for r in in_gap],
            "n_non_concordant": n_interleave,
            "non_concordant": [(r["label"], r["side"], r["texture"])
                               for r in non_concordant],
            "n_survive": len(surv_members), "n_die": len(die_members),
            "die_share_fresh": share,
            "die_share_cp95": [cp_lo, cp_hi],
            "mode_spreads": mode_spreads,
            "canon_position": {a: c["position"] for a, c
                               in canon_position.items()},
        },
        "prediction_scoring": {
            "P-e327a_bimodal_small_die_share": p_hit,
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P3 the census adjudicated (the frozen bars + the die "
                  "share + the within-mode spreads + the canon's position)")

    # ================= P4: honesty + provenance + the artifacts ==========
    metrics["honesty"] = {
        "no_shopping": ("eight spread draws registered at birth, eight run, "
                        "eight reported — wherever they landed; no formation "
                        "gate, no redraws"),
        "canon_never_rerun": ("gen 24314's row read from e264/e290/e272's "
                              "artifacts at runtime + cross-asserted vs the "
                              "birth literals; e324's record likewise "
                              "(G_E324READ)"),
        "n8_caveat": ("an 8+1 census decides the modes' EXISTENCE and a "
                      "coarse die share (the CP band is wide: 0/8 leaves "
                      "the share's upper bound ~34%); the tails and the "
                      "room x gen interaction remain single-room-anchored; "
                      "nothing guaranteed"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "thermal_envelope": _envelope_summary(),
    }
    make_census_plot(RD, census_rows, census, canon_row, e324_reads,
                     die_share, mode_spreads, verdict)
    write_report(RD, metrics, census_rows, census, canon_row,
                 mode_spreads, die_share, canon_position, verdict, clause,
                 e324_reads)
    metrics["status"] = ("RUN RECORDED — the census adjudicated; the "
                         "heartbeat folds" if not SMOKE else
                         "SMOKE — shakedown only, nothing adjudicated")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e327_mode_census.png"),
                          str(RD / "REPORT.md")]
    write_partial("DONE (census + plot + report)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ the plot
def make_census_plot(rd, census_rows, census, canon_row, e324_ctx,
                     die_share, mode_spreads, verdict) -> None:
    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.78],
                          wspace=0.26)

    # ---- panel A: the s1 distribution with the gap shaded --------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xscale("log")
    ax.axvspan(S1_DIE_BAR, S1_GAP_HI, color="tab:orange", alpha=0.22,
               label=f"THE GAP [0.01, 0.15] — zero prior test mass")
    ax.axvspan(1e-6, S1_DIE_BAR, color="tab:blue", alpha=0.08)
    ax.axvspan(S1_GAP_HI, 1.0, color="tab:red", alpha=0.08)
    # e324's four draws as hollow context markers (never bar inputs)
    for j, g in enumerate(E324_GENS):
        ax.scatter(e324_ctx["s1_g0"][g], 0.16 + 0.05 * (j % 2), s=46,
                   facecolors="none", edgecolors="tab:gray", linewidths=1.2,
                   zorder=2)
    for i, r in enumerate(census_rows):
        if r["label"] == "CANON":
            ax.scatter(r["s1_g0"], 0.62, s=340, c="gold", marker="*",
                       zorder=4, edgecolors="k", linewidths=1.0)
            ax.annotate(f"CANON {COMMITTED_INST_GEN}\n{r['s1_g0']:.1e}",
                        (r["s1_g0"], 0.62), textcoords="offset points",
                        xytext=(10, -4), fontsize=8, fontweight="bold")
        else:
            col = {"DIE": "tab:blue", "SURVIVE": "tab:red",
                   "GAP": "tab:orange"}[r["side"]]
            y = 0.30 + 0.075 * ((i - 1) % 8)
            ax.scatter(r["s1_g0"], y, s=80, c=col, zorder=3,
                       edgecolors="k", linewidths=0.6)
            ax.annotate(f"{r['s1_g0']:.4f}", (r["s1_g0"], y),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=6.5, color=col, ha="center")
    ax.set_ylim(0, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("s1 g0 (the fork; log scale)")
    ax.set_title(f"the s1 distribution — {len(census_rows)} census draws "
                 f"(8 spread gens + the canon)\n"
                 f"{die_share['k_die_fresh']}/8 fresh DIE; "
                 f"gap draws: "
                 f"{sum(1 for r in census_rows if r['side'] == 'GAP')}",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    # ---- panel B: the texture scatter with the two template boxes -------
    axb = fig.add_subplot(gs[0, 1])
    axb.axhspan(DIE_TEX_INROOM_MIN, 1.0, xmin=0.0,
                xmax=DIE_TEX_NORM_MAX / 30.0, color="tab:blue", alpha=0.10)
    axb.axhspan(SURV_TEX_INROOM_BAND[0], SURV_TEX_INROOM_BAND[1],
                xmin=SURV_TEX_NORM_MIN / 30.0, xmax=1.0, color="tab:red",
                alpha=0.08)
    axb.axvline(DIE_TEX_NORM_MAX, color="tab:blue", ls=":", lw=1.2)
    axb.axvline(SURV_TEX_NORM_MIN, color="tab:red", ls=":", lw=1.2)
    axb.axhline(DIE_TEX_INROOM_MIN, color="tab:blue", ls=":", lw=1.0)
    axb.axhline(SURV_TEX_INROOM_BAND[0], color="tab:red", ls=":", lw=0.8)
    axb.axhline(SURV_TEX_INROOM_BAND[1], color="tab:red", ls=":", lw=0.8)
    for g in E324_GENS:      # e324 context (hollow)
        axb.scatter(e324_ctx["write_norm"][g], e324_ctx["in_own_room"][g],
                    s=42, facecolors="none", edgecolors="tab:gray",
                    linewidths=1.1, zorder=2)
    for i, r in enumerate(census_rows):
        if r["label"] == "CANON":
            axb.scatter(r["write_norm"], r["in_own_room"], s=340, c="gold",
                        marker="*", zorder=4, edgecolors="k", linewidths=1.0)
            axb.annotate(f"CANON {COMMITTED_INST_GEN}", (r["write_norm"],
                        r["in_own_room"]), textcoords="offset points",
                        xytext=(8, 2), fontsize=8, fontweight="bold")
        else:
            col = {"DIE": "tab:blue", "SURVIVE": "tab:red",
                   "GAP": "tab:orange"}[r["side"]]
            axb.scatter(r["write_norm"], r["in_own_room"], s=80, c=col,
                        zorder=3, edgecolors="k", linewidths=0.6)
            axb.annotate(f"D{r['label'][5:]}", (r["write_norm"],
                        r["in_own_room"]), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, color=col, ha="center")
    axb.set_xlim(0, 30)
    axb.set_ylim(0.35, 1.02)
    axb.set_xlabel("write norm")
    axb.set_ylabel("in-own-room fraction")
    axb.set_title("the texture scatter — DIE box (norm<=12, in-room>=0.9) "
                  "vs SURVIVE box (norm>=24, in-room 0.53-0.73)\n"
                  "hollow gray = e324's four consecutive draws (context)",
                  fontsize=8.5)
    axb.grid(alpha=0.25)

    # ---- panel C: the die-share estimate with its binomial band --------
    axc = fig.add_subplot(gs[0, 2])
    share = die_share["share"]
    lo, hi = die_share["clopper_pearson_95"]
    axc.bar([0], [share], width=0.5, color="tab:blue", alpha=0.75,
            edgecolor="k")
    axc.errorbar([0], [share], yerr=[[share - lo], [hi - share]], fmt="none",
                 ecolor="k", capsize=6, lw=1.4)
    axc.axhline(0.5, color="tab:gray", ls="--", lw=0.8)
    axc.annotate(f"{die_share['k_die_fresh']}/8 = {share:.1%}\n95% CP "
                 f"[{lo:.1%}, {hi:.1%}]\n(canon excluded:\ncommitted-"
                 f"selected)", (0, share), textcoords="offset points",
                 xytext=(14, -6), fontsize=9)
    axc.set_xlim(-0.55, 0.75)
    axc.set_ylim(0, 1.0)
    axc.set_xticks([])
    axc.set_ylabel("die-mode share (fresh draws)")
    axc.set_title("the die-mode's first population estimate", fontsize=9)
    axc.grid(alpha=0.25, axis="y")

    fig.suptitle("e327 THE SEED-STRATIFIED MODE CENSUS — 8 spread gens "
                 "((2i+1)*2^28) + the canon at the HELD K10K room "
                 f"(seeds {HELD_ROOM_SEEDS[0]}/{HELD_ROOM_SEEDS[1]}) -> "
                 f"{verdict}", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "e327_mode_census.png", dpi=130)
    plt.close(fig)
    log(f"[plot] wrote {rd / 'e327_mode_census.png'}")


# ------------------------------------------------------------------ the report
def write_report(rd, m, census_rows, census, canon_row, mode_spreads,
                 die_share, canon_position, verdict, clause,
                 e324_ctx) -> None:
    adj = m["adjudication"]
    env = m["provenance"]["thermal_envelope"]
    gates = m["gates"]
    L: list[str] = []
    L.append("# e327 — THE SEED-STRATIFIED MODE CENSUS "
             f"(run {m['date']})\n")
    L.append("**THE QUESTION** (dispatch verbatim): *THIS CENSUS decides: "
             "bimodal vs wide, the s1 gap's emptiness, and the die-mode's "
             "population share.*\n")
    L.append("## The census table (all 9 draws, all axes)\n")
    L.append("| draw | gen | s1 g0 (side) | texture | concordant | "
             "post g0 | gm12 (ctx) | write norm | in-own-room |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in census_rows:
        s1f = (f"{r['s1_g0']:.2e}" if r["s1_g0"] < 0.01
               else f"{r['s1_g0']:.4f}")
        L.append(f"| {r['label']} | {r['gen']} | {s1f} ({r['side']}) | "
                 f"{r['texture']} | {'yes' if r['concordant'] else 'NO'} | "
                 f"{r['post_g0']:.6f} | {r['post_gm12']:.5f} | "
                 f"{r['write_norm']:.4f} | {r['in_own_room']:.4f} |")
    L.append("")
    L.append("## The s1-gap outcome\n")
    n_gap = sum(1 for r in census_rows if r["side"] == "GAP")
    L.append(f"- draws inside [0.01, 0.15]: **{n_gap}** ("
             + ", ".join(f"{r['label']} gen {r['gen']} s1 {r['s1_g0']:.4f}"
                         for r in census_rows if r["side"] == "GAP")
             + ("none — the gap's interior stays EMPTY at n=8+1 spread "
                "mass" if n_gap == 0 else "") + ")\n")
    L.append("## The die-mode share (the first estimate)\n")
    L.append(f"- **{die_share['k_die_fresh']}/8 = "
             f"{die_share['share']:.1%}** (95% Clopper-Pearson "
             f"[{die_share['clopper_pearson_95'][0]:.1%}, "
             f"{die_share['clopper_pearson_95'][1]:.1%}]) over the EIGHT "
             "UNCONDITIONED spread draws; the canon excluded "
             "(committed-selected through the die-admitting trust band — "
             "R73's own point)\n")
    L.append("## The within-mode spreads (bar axes: norm + in-room)\n")
    for mname, v in mode_spreads.items():
        def _f(x):
            return "n<2 (vacuous)" if x is None else f"{x:.1%}"
        L.append(f"- **{mname}** (n={v['n_members']}: "
                 f"{', '.join(v['members'])}): norm {_f(v['write_norm_spread'])}"
                 f", in-room {_f(v['in_own_room_spread'])} "
                 f"(bar <= 15%; context: g0 "
                 f"{_f(v['g0_spread_ctx'])}, gm12 "
                 f"{_f(v['gm12_spread_ctx'])})")
    L.append("")
    L.append("## The canon's position in the enlarged census\n")
    L.append("; ".join(f"**{a}**: rank {c['rank']}/{c['of']} "
                       f"({c['position']})"
                       for a, c in canon_position.items()) + "\n")
    L.append("## The frozen bars, scored\n")
    L.append(f"- **VERDICT: {verdict}**\n")
    L.append(f"> {clause}\n")
    L.append("## e324 context (never bar inputs)\n")
    L.append("- the four consecutive draws (32401-32404): s1 "
             + "/".join(f"{e324_ctx['s1_g0'][g]:.4f}" for g in E324_GENS)
             + f", norms {min(e324_ctx['write_norm'].values()):.2f}-"
             f"{max(e324_ctx['write_norm'].values()):.2f}, in-room "
             f"{min(e324_ctx['in_own_room'].values()):.4f}-"
             f"{max(e324_ctx['in_own_room'].values()):.4f}\n")
    L.append("## Gates\n")
    n_g = sum(1 for g in gates.values() if g.get("pass"))
    L.append(f"- {n_g}/{len(gates)} named gate classes PASS"
             + (" (all PASS)" if n_g == len(gates)
                else " — FAILURES: " + ", ".join(
                    k for k, g in gates.items() if not g.get("pass"))))
    L.append(f"- per-draw: G_PROTOIDENT 8/8 (protocol identity from each "
             f"run's own optimizer artifact) + G_FACTLOAD 8/8 "
             f"(read-determinism re-loads)\n")
    L.append("## Envelope\n")
    L.append(f"- bursts <= {env['burst_cap_s']:.0f}s, cooldowns "
             f"{env['cooldown_s']:.0f}s, hard line {env['hard_line_c']:.0f}C; "
             f"{env['n_polls']} polls tagged e327:*; max temp "
             f"{env['max_temp_seen_c']}C; violations >= "
             f"{env['hard_line_c']:.0f}C: {env['violations_ge_84c']}\n")
    L.append("## Prediction scoring\n")
    L.append(f"- P-e327a (BIMODAL-CONFIRMED + die share small (<=1 of 8 "
             f"fresh DIE)): "
             f"{'HIT' if adj['prediction_scoring']['P-e327a_bimodal_small_die_share'] else 'MISS'}"
             "\n")
    L.append("## Catches / disclosures\n")
    for d in m["deviations"]:
        L.append(f"- {d}")
    L.append("")
    L.append("*This cell does not edit NOTES/THINKING/QUEUE/STATE — the "
             "heartbeat folds.*")
    (rd / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"[report] wrote {rd / 'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
