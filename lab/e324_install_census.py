"""E324 — THE INSTALL-DRAW CENSUS (T288's follow-up ladder, step 1; the
monoculture question's distribution-maker). This docstring carries the
registered question + design + bars + P-e324a/P-e324b VERBATIM from the
dispatch letter, committed at birth BEFORE any compute. Adjudicate against
exactly this; no bar shopping.

THE QUESTION (dispatch verbatim): "what is the install-draw lottery's
spread on the formation texture axes, at HELD room?"

BACKGROUND (dispatch verbatim): "e323 (runs/e323/) found the committed
formation canon secretly shares ONE install draw (gen 24314) whose first
AdamW step KILLS the read (g0 0.0000 at s1, then recovery —
'die-then-recover', pruned/localized: 94% in-room, norm 9.18), while fresh
draws SURVIVE step one and sprawl (e323's draw: s1 0.7451, 63% in-room,
norm 27.06, reads +81% over the committed 0.2646; the rider's five fresh
family installs propagated the texture). THE INSTALL-DRAW LOTTERY (+81%)
DOMINATES THE ROOM LOTTERY (-21%, e272). Nobody knows the install
lottery's DISTRIBUTION — that's this cell."

THE DESIGN (dispatch verbatim):
  "1. HELD ROOM: the committed K10K room (seeds 26113/26114, bit-bound —
  the same room the canon formed in; e323 used a fresh room, this cell
  isolates the GEN wheel).
  2. N=4 FRESH INSTALL GENS (disclose derivation; e.g. 32401-32404) run
  through the committed e261/e264 install rig VERBATIM (same protocol,
  same corpus, same budget — one install per gen, each within the burst
  cap with cooldowns; no shopping, no redraws).
  3. THE CENSUS per draw (including the committed gen, re-verified from
  its committed values — do not rerun it): g0 at its own battery; write
  norm; in-own-room fraction; gm12 (the mid-length spread measure e323
  used); S1-SURVIVAL (the read at step 1: the die-then-recover vs
  survive-and-sprawl fork).
  4. THE S1-FORK DISCRIMINATOR (rides free): do the draws split cleanly
  by step-one survival, and does survival predict the bulky texture
  (norm/in-room/gm12)? Report the split as a finding.
  5. THE CANON'S POSITION: where does gen 24314 sit in the census
  distribution (an extreme? the median?)."

THE FRESH-GEN DERIVATIONS (disclosed at birth): THE FOUR GENS :=
32401, 32402, 32403, 32404 — this cell's allocation block 324NN, slots
01-04 (the dispatch's own example range, adopted verbatim; the family's
<cell>01 corpus-stream rule does NOT apply — this cell has no separate
corpus stream, the Dmix generator IS the draw — so slots 01-04 go to the
four install draws). Verified at birth: the four gens are distinct, none
equals the committed 24314, none appears in any family allocation
(26011/26012, 26111-26118, 27215/27216, 29001, 32301-32325), and an
EXACT-TOKEN grep of lab/*.py + runs/*/metrics.json for 32401/32402/
32403/32404 returned ZERO hits (substring digits inside long floats are
not tokens; the grep used token boundaries).

FROZEN BARS (dispatch verbatim; adjudicated against exactly this):
  - INSTALL-DOMINATES: "the fresh gens' g0 spread (max/min across draws)
    exceeds e272's room-lottery 21% band materially (>= 40%), OR the
    texture axes (norm, in-room) vary > 2x across draws — the install
    lottery is the formation wheel; every canon number gains the census's
    spread as its error bar; Law 3 drafts with it."
  - ROOM-COMPARABLE: "the fresh gens' g0 spread lands within ~25%
    (comparable to e272's room spread) — the e323 miss was a tail draw,
    not a wider wheel; the canon's monoculture caveat softens to a
    two-wheel lottery of similar sizes."
  - "The S1-fork reports as a finding (S1-FORK-CLEARS / S1-FORK-MUDDY),
    never a bar."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * SPREAD_g0 := (max/min - 1) over the FOUR fresh draws' post-install
    g0 — the multiplicative max/min form the dispatch names. e272's
    SAME-FORMULA room value := committed 0.26464763283729553 / K10KR
    0.2097209095954895 - 1 = 26.2% (read from the artifacts at runtime);
    the "~21%" figure's committed-relative form (1 - K10KR/committed =
    20.7%) co-reported. Denominator floor 1e-6 (e261's convention).
  * "texture axes vary > 2x" := max/min > 2.0 over the four fresh draws,
    applied SEPARATELY to write norm and to in-own-room fraction.
  * ROOM-COMPARABLE's "within ~25%" := SPREAD_g0 < 0.25.
  * THE GAP the two bars leave (0.25 <= SPREAD_g0 < 0.40 with no texture
    axis > 2x) := BETWEEN-WHEELS — no bar fires; the honest in-between,
    disclosed (named at birth so the residue is not shopped later).
  * COMPOSITE ORDER: TEXTURE (any hard-gate failure) -> INSTALL-DOMINATES
    -> ROOM-COMPARABLE -> BETWEEN-WHEELS.
  * THE S1-FORK (a finding, never a bar): S1-DIE := s1 g0 < 0.01;
    S1-SURVIVE := s1 g0 > 0.15 (the gap [0.01, 0.15] is the no-man's
    land; the committed family's s1 reads 1.35e-5 and e323's 0.7451 sit
    3+ orders either side — the bars only formalize the gap). The census
    set for the fork := the four fresh draws + the canon (5 draws).
    S1-FORK-CLEARS iff (i) every census draw lands OUTSIDE the gap
    (clean bimodality) AND (ii) survival predicts the bulk: the SURVIVE
    side's min write norm > the DIE side's max write norm. Anything else
    := S1-FORK-MUDDY (the failed clause disclosed). The in-room + gm12
    orderings co-reported descriptively.
  * THE CANON'S POSITION := the rank of gen 24314 among the 5 census
    points on the g0 axis (norm + in-room co-reported): rank 1 or 5 :=
    an EXTREME; rank 3 := the MEDIAN; rank 2 or 4 := INTERIOR.
  * THE HELD ROOM := the committed K10K room rebuilt from its registered
    seeds 26113/26114 (e261's registration; e264's committed record) and
    BIT-BOUND against runs/checkpoints/e264_rooms.pt's stored K10K D/S
    (exact array equality; file md5 hard-bound) — the same room the canon
    formed in. Certified (G_PROJ) as always.
  * THE RIG := e261's chunked_install BY IMPORT, driven VERBATIM: 400
    Dmix steps (ix(16) install w/ name mask + aj(16) paired + rj(32)
    random; masked token-level union CE; AdamW (0.9,0.95) wd 0.1; lr 1e-3
    x cosine(step-1,1000); clip 1.0 -> P_room(g) fp64 CPU, write fp32),
    from the SAME committed root (g1c_root.pt), the SAME banks (the
    committed splice bank, battery, anchors — rebuilt by the committed
    deterministic path and gated). The ONLY delta across the four draws
    is the Dmix generator seed (E261.FRESH_GEN rebound per draw). NO
    consolidation phase (disclosed: the census axes — g0/norm/in-room/
    gm12/s1 — are all INSTALL-phase quantities; the canon's 0.2646 is the
    install-phase read; e323's +81% was install-phase too).
  * THE CANON IS NOT RERUN: gen 24314's census row is READ from the
    artifacts at runtime (e264's metrics: post g0/gm12/gp12 + the install
    traj's s1 + in-own-room; e290's metrics: the write norm 9.1788 — the
    record that recomputed it from the committed fact; e272's metrics:
    the K10KR s1, the die's SECOND-ROOM confirmation) and asserted
    against the birth literals (Rule 12; catches artifact drift). Never
    retyped into the census.
  * NO FORMATION GATE (disclosed): e323's [0.15, 0.45] band is drawn on
    the census plot as CONTEXT only — a census does not reject its own
    data. All four draws are the datum wherever they land; no redraws,
    no shopping.

REGISTERED PREDICTIONS (registered at birth, before compute):
  - P-e324a (the dispatch's lab guess, ADOPTED — I agree):
    INSTALL-DOMINATES — e323's autopsy saw the texture propagate across
    six independent fresh installs (the primary + five rider installs);
    that breadth smells like a wide wheel. (The five-miss hedge stands:
    the lab's last five mechanism guesses missed.)
  - P-e324b (my registered counter-branch, named before compute):
    ROOM-COMPARABLE at the held room — e323's +81% needed BOTH wheels
    fresh (fresh room AND fresh gen); at the canon's own room the
    capture machinery reasserts (the committed write sat 94.4% in-room —
    this room's projector passes ~6.1% of every clipped gradient along
    the SAME 10k dims for every draw, and the basin those dims carve may
    dominate whatever direction a draw sprays). Note the s1-die is
    gen-tracked not room-tracked (gen 24314 died at s1 in BOTH e264's
    room AND e272's fresh room — 1.35e-5 twice), so my counter predicts
    the fresh draws SURVIVE s1 yet still CLUSTER in g0: the die-then-
    recover vs survive-and-sprawl fork is real, but the FORK'S SIZE at
    held room is room-lottery-class, not +81%-class. Discriminating
    texture if it fires: the four survivors' in-room fractions cluster
    HIGH (>= ~0.85) — the held room recaptures what the draw sprays.

GATES (a failure HALTS the record as TEXTURE): {G_VOCAB, G_NAMEFREE,
G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_PARENTS,
G_CANONREAD, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMHELD,
G_GENFRESH, G_PROTOIDENT, G_FACTLOAD} — G_PROTOIDENT + G_FACTLOAD
instantiated PER DRAW (all four must pass).

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (dispatch 180), per-step thermal polls at a 78C margin, 40s
cooldowns (dispatch 30-60), the 84C never-past line (dispatch 85), polls
persisted to runs/_envelope_log.jsonl tagged e324:<GEN>:chunkN; gpu_ok()
at startup; the four installs SEQUENTIAL — never concurrent; CPU fp64
dense projections (pocketfft), CPU threads 4 (the x19 CPU agent is live
— normal CPU sharing only).

Outputs: runs/e324/{metrics.json (PROGRESSIVE), e324_install_census.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e324_gen*.pt + e324_*_resume.pt (gitignored; md5s in
metrics). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat
folds). Commit + push per phase.

Run:  cd lab && python e324_install_census.py    (E324_SMOKE=1 shakedown)
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
import torch.nn.functional as F                        # noqa: E402

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

torch.set_num_threads(4)           # shared machine (the x19 CPU agent live)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E324_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e324_smoke" if SMOKE else "e324"
assert torch.cuda.is_available(), "e324 owns the GPU lane (dispatch)"
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
# e324:...). The e261 import's own append handle on runs/e261/run.log is
# opened but never written through (e323's convention; disclosed).
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

# ---- THE FOUR FRESH INSTALL DRAWS (derivations in the docstring) -------
FRESH_GENS: tuple[int, ...] = (32401, 32402, 32403, 32404)
COMMITTED_INST_GEN = 24314            # the g1c draw (e261's FRESH_GEN)
FAMILY_ALLOCATIONS = frozenset(
    [24314, 26011, 26012, 26111, 26112, 26113, 26114, 26115, 26116,
     26117, 26118, 27215, 27216, 27301, 28301, 28401, 28501, 28801,
     29001, 10901, 1337]
    + list(range(32301, 32326)))      # e323's whole allocation block

# ---- THE FROZEN BAR NUMBERS -------------------------------------------
SPREAD_DOMINATES_BAR = 0.40           # INSTALL-DOMINATES: SPREAD_g0 >= 40%
SPREAD_ROOM_BAR = 0.25                # ROOM-COMPARABLE: SPREAD_g0 < 25%
TEXTURE_2X_BAR = 2.0                  # texture axes vary > 2x := max/min > 2
RATIO_DEN_FLOOR = 1e-6                # e261's jump-ratio denominator floor
S1_DIE_BAR = 0.01                     # s1 g0 < this := S1-DIE
S1_GAP_HI = 0.15                      # s1 g0 > this := S1-SURVIVE
FORMATION_BAND = (0.15, 0.45)         # e272's band — CONTEXT ONLY, never a gate

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
E323_METRICS = E43.REPO / "runs" / "e323" / "metrics.json"
E323_MD5 = "bf01f994cda19c9b04e151f8e2c457aa"          # verified at birth
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
E323_FRESH_POST_G0 = 0.4788312613964081        # e323's fresh-room fresh-gen
E323_FRESH_S1_G0 = 0.7451097965240479          # the survive exemplar
E323_FRESH_WRITE_NORM = 27.056774262360456
E323_FRESH_IN_OWN_ROOM = 0.6329243240923356

VOCAB_EXPECT = 65
G_READ_TOL = E261.G_READ_TOL             # 5e-3

REGISTERED = {
    "question_verbatim": (
        "what is the install-draw lottery's spread on the formation texture "
        "axes, at HELD room?"),
    "background_verbatim": (
        "e323 (runs/e323/) found the committed formation canon secretly "
        "shares ONE install draw (gen 24314) whose first AdamW step KILLS "
        "the read (g0 0.0000 at s1, then recovery — 'die-then-recover', "
        "pruned/localized: 94% in-room, norm 9.18), while fresh draws "
        "SURVIVE step one and sprawl (e323's draw: s1 0.7451, 63% in-room, "
        "norm 27.06, reads +81% over the committed 0.2646; the rider's "
        "five fresh family installs propagated the texture). THE "
        "INSTALL-DRAW LOTTERY (+81%) DOMINATES THE ROOM LOTTERY (-21%, "
        "e272). Nobody knows the install lottery's DISTRIBUTION — that's "
        "this cell."),
    "design_verbatim": (
        "1. HELD ROOM: the committed K10K room (seeds 26113/26114, "
        "bit-bound — the same room the canon formed in; e323 used a fresh "
        "room, this cell isolates the GEN wheel). 2. N=4 FRESH INSTALL "
        "GENS (disclose derivation; e.g. 32401-32404) run through the "
        "committed e261/e264 install rig VERBATIM (same protocol, same "
        "corpus, same budget — one install per gen, each within the burst "
        "cap with cooldowns; no shopping, no redraws). 3. THE CENSUS per "
        "draw (including the committed gen, re-verified from its committed "
        "values — do not rerun it): g0 at its own battery; write norm; "
        "in-own-room fraction; gm12 (the mid-length spread measure e323 "
        "used); S1-SURVIVAL (the read at step 1: the die-then-recover vs "
        "survive-and-sprawl fork). 4. THE S1-FORK DISCRIMINATOR (rides "
        "free): do the draws split cleanly by step-one survival, and does "
        "survival predict the bulky texture (norm/in-room/gm12)? Report "
        "the split as a finding. 5. THE CANON'S POSITION: where does gen "
        "24314 sit in the census distribution (an extreme? the median?)"),
    "bars_verbatim": {
        "INSTALL-DOMINATES": (
            "the fresh gens' g0 spread (max/min across draws) exceeds "
            "e272's room-lottery 21% band materially (>= 40%), OR the "
            "texture axes (norm, in-room) vary > 2x across draws — the "
            "install lottery is the formation wheel; every canon number "
            "gains the census's spread as its error bar; Law 3 drafts "
            "with it."),
        "ROOM-COMPARABLE": (
            "the fresh gens' g0 spread lands within ~25% (comparable to "
            "e272's room spread) — the e323 miss was a tail draw, not a "
            "wider wheel; the canon's monoculture caveat softens to a "
            "two-wheel lottery of similar sizes."),
        "S1-FORK": (
            "The S1-fork reports as a finding (S1-FORK-CLEARS / "
            "S1-FORK-MUDDY), never a bar."),
    },
    "operationalizations": (
        "frozen BEFORE compute: SPREAD_g0 := (max/min - 1) over the FOUR "
        "fresh draws' post-install g0 (the multiplicative max/min form; "
        "denominator floor 1e-6); e272's same-formula room value = "
        "0.26464763283729553 / 0.2097209095954895 - 1 = 26.2% (runtime-"
        "read; the committed-relative 20.7% co-reported as the '~21%' "
        "figure's origin); 'texture varies > 2x' := max/min > 2.0 over "
        "the four fresh draws, separately on write norm + in-own-room; "
        "ROOM-COMPARABLE := SPREAD_g0 < 0.25; the residue 0.25 <= "
        "SPREAD_g0 < 0.40 with no texture > 2x := BETWEEN-WHEELS (no "
        "bar, the honest gap); COMPOSITE: TEXTURE -> INSTALL-DOMINATES -> "
        "ROOM-COMPARABLE -> BETWEEN-WHEELS; the S1 fork: DIE := s1 < "
        "0.01, SURVIVE := s1 > 0.15, S1-FORK-CLEARS iff all 5 census "
        "draws sit outside the [0.01, 0.15] gap AND the SURVIVE side's "
        "min write norm > the DIE side's max write norm, else MUDDY (the "
        "failed clause disclosed); the canon's position := its rank among "
        "the 5 census points on g0 (1/5 = EXTREME, 3 = MEDIAN, 2/4 = "
        "INTERIOR; norm + in-room co-reported); the HELD room rebuilt "
        "from seeds 26113/26114 and BIT-BOUND vs e264_rooms.pt's K10K "
        "D/S; the rig = e261's chunked_install BY IMPORT with "
        "E261.FRESH_GEN rebound per draw — the ONLY delta; NO "
        "consolidation (the census axes are install-phase quantities); "
        "NO formation gate (the [0.15, 0.45] band is plot context only — "
        "a census does not reject its own data; no redraws, no shopping); "
        "the canon NOT rerun — its row READ from e264/e290/e272's "
        "artifacts at runtime + asserted vs the birth literals"),
    "registration": (
        "bars + question + design + operationalizations frozen VERBATIM "
        "from the dispatch letter (T288's follow-up ladder step 1; the "
        "monoculture question's distribution-maker); this script "
        "committed at birth BEFORE any compute; adjudicate against "
        "exactly this; no bar shopping."),
    "predictions": {
        "P-e324a_install_dominates": (
            "ADOPTED from the dispatch (I agree): INSTALL-DOMINATES — "
            "e323's autopsy saw the texture propagate across six "
            "independent fresh installs (the primary + five rider "
            "installs); that breadth smells like a wide wheel. (The "
            "five-miss hedge stands.)"),
        "P-e324b_room_comparable_held_room": (
            "my registered counter-branch: ROOM-COMPARABLE at the held "
            "room — e323's +81% needed BOTH wheels fresh; at the canon's "
            "own room the capture machinery reasserts (the committed "
            "write sat 94.4% in-room; this room's projector passes "
            "~6.1% of every clipped gradient along the SAME 10k dims for "
            "every draw). The s1-die is gen-tracked not room-tracked "
            "(gen 24314 died at s1 in BOTH rooms, 1.35e-5 twice), so the "
            "counter predicts the fresh draws SURVIVE s1 yet CLUSTER in "
            "g0 — the fork is real but its SIZE at held room is "
            "room-lottery-class. Discriminating texture: the survivors' "
            "in-room fractions cluster HIGH (>= ~0.85)."),
    },
}

deviations: list[str] = [
    "THE FOUR INSTALLS RUN THROUGH e261's chunked_install BY IMPORT (the "
    "committed rig VERBATIM — extend, don't repeat): the module-global "
    "FRESH_GEN is rebound per draw (32401-32404; the committed 24314 is "
    "co-reported); the room ladder is the HELD pair 26113/26114; the "
    "step body (draws, clip, the room hook, the milestone battery) is "
    "the committed file's, NOT modified.",
    "NO CONSOLIDATION PHASE (disclosed): the census axes — g0 at its own "
    "battery, write norm, in-own-room, gm12, s1 — are all INSTALL-phase "
    "quantities; the canon's 0.2646 is the install-phase read (e264's "
    "post_cells); e323's +81% was install-phase too. The cons (e113 "
    "seed 10901) is a different question's machinery.",
    "NO FORMATION GATE (disclosed): e323's halt taught that the "
    "[0.15, 0.45] band does not contain fresh draws; a CENSUS does not "
    "reject its own data — the band is drawn on the plot as context "
    "only; every draw lands wherever it lands (no shopping, no "
    "redraws).",
    "THE CANON IS NOT RERUN: gen 24314's row is READ from e264's metrics "
    "(post g0/gm12/gp12, install-traj s1, in-own-room) + e290's metrics "
    "(the write norm, its G_FACTLOAD record) + e272's metrics (the "
    "K10KR s1 — the die's second-room confirmation) at RUNTIME, asserted "
    "against the birth literals; the census row carries the "
    "runtime-read values.",
    "e290 IS ADDED TO THE DISPATCH'S PARENT SET (e261/e264/e272/e323 + "
    "the room artifacts): the committed write norm 9.1788 lives in "
    "e290's G_FACTLOAD record (e264's metrics holds the fractions, not "
    "the norm) — an extension of the chain, never a replacement.",
    "THE FRESH INSTALLS EACH START FROM THE SAME COMMITTED ROOT "
    "(g1c_root.pt, bit-gated once, held in memory across draws — the "
    "census's held organism; e264's committed K10K install started from "
    "the same root).",
    "n=4 fresh draws, one lineage, one session (the g-series standing "
    "lottery caveat — the critic's note carried verbatim); a 4-draw "
    "census estimates a distribution's coarse spread, not its tails; "
    "nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E324_SMOKE=1): 8-step installs, room at k=512 (the "
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
                             "from runs/_envelope_log.jsonl (tags e324:*)",
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


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e324_install_census",
        "phase": "THE INSTALL-DRAW CENSUS — the formation lottery's gen "
                 "wheel mapped at HELD room (the committed K10K room "
                 "26113/26114, bit-bound): N=4 fresh install gens "
                 "(32401-32404) through the committed e261/e264 rig "
                 "VERBATIM (400 Dmix steps each, one install per gen, no "
                 "shopping) + the committed gen 24314 re-verified from its "
                 "artifacts; the census axes: g0 / write norm / in-own-room "
                 "/ gm12 / s1-survival; the s1-fork rides as a finding; "
                 "INSTALL-DOMINATES vs ROOM-COMPARABLE on the frozen "
                 "spread bars",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; the "
                      "four installs SEQUENTIAL with cooldowns — never "
                      "concurrent) + CPU fp64 dense projections (pocketfft), "
                      "CPU threads 4 (the x19 CPU agent live — shared)",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e324:<GEN>:chunkN; gpu_ok() at startup",
            "trainings": "4 installs x 400 Dmix steps (the committed rig's "
                         "own budget), one per fresh gen",
        },
        "the_held_room": {
            "seeds": list(HELD_ROOM_SEEDS),
            "mode": ROOM_MODE,
            "k": ROOM_K,
            "k_fraction_of_N": ROOM_K / 2739072,
            "derivation": "e261's registered K10K rung pair — the canon's "
                          "own writing room (e264's committed K10K install); "
                          "bit-bound vs runs/checkpoints/e264_rooms.pt",
        },
        "the_fresh_gens": {
            "gens": list(FRESH_GENS),
            "derivation": "cell 324's allocation block 324NN slots 01-04 "
                          "(the dispatch's own example range, adopted); no "
                          "separate corpus stream in this cell — the Dmix "
                          "generator IS the draw; verified distinct + "
                          "exact-token absent from lab/*.py + "
                          "runs/*/metrics.json at birth",
            "committed_inst_gen": COMMITTED_INST_GEN,
        },
        "deviations": deviations,
        "builds_on": [
            "T288 / e323 (THE CASCADE GUARD's autopsy: the committed canon "
            "shares ONE install draw gen 24314 whose first AdamW step kills "
            "the read; the fresh draw survived and sprawled +81%; the "
            "rider's five fresh family installs propagated the texture — "
            "THIS cell maps that lottery's spread)",
            "T239 / e261 + e264 (THE INSTALL RIG: chunked_install VERBATIM "
            "by import + the committed K10K room's seeds + the canon's own "
            "record)",
            "T242 / e272 (RANK-WRITES-THE-CURVE: the room lottery priced — "
            "K10KR 0.2097 vs committed 0.2646, the ~21% spread + the "
            "[0.15, 0.45] trust band, both the census's comparator)",
            "T269 / e290 (the committed write norm's own record — the "
            "canon's norm read from its G_FACTLOAD)",
            "T287 / T288's dispatch (the follow-up ladder step 1: the "
            "monoculture question's distribution-maker)",
        ],
        "whats_new": [
            "THE INSTALL-DRAW LOTTERY'S FIRST DISTRIBUTION (the record's "
            "first): four fresh gens at HELD room — the gen wheel isolated "
            "from the room wheel for the first time (e272 held the gen and "
            "moved the room; e323 moved both)",
            "THE CENSUS AXES AT FORMATION (the record's first): g0 + write "
            "norm + in-own-room + gm12 + s1-survival ON ONE table, the "
            "canon's row re-verified from its artifacts (never rerun)",
            "THE S1-FORK DISCRIMINATOR (the record's first): does step-one "
            "survival predict the bulky texture across draws (a finding, "
            "never a bar)",
            "THE CANON'S POSITION NAMED (the record's first): where gen "
            "24314 sits in its own lottery's distribution — the monoculture "
            "caveat's quantitative form",
        ],
        "gates": {},
    })
    log(f"E324 — THE INSTALL-DRAW CENSUS (smoke={SMOKE}) -> {RD}")
    log(f"held room: {ROOM_MODE} k={ROOM_K} seeds {HELD_ROOM_SEEDS}; the "
        f"four fresh gens {FRESH_GENS} (committed {COMMITTED_INST_GEN}); "
        f"bars: SPREAD_g0 >= {SPREAD_DOMINATES_BAR:.0%} or texture > "
        f"{TEXTURE_2X_BAR:.0f}x -> INSTALL-DOMINATES; SPREAD_g0 < "
        f"{SPREAD_ROOM_BAR:.0%} -> ROOM-COMPARABLE; the gap -> BETWEEN-WHEELS")
    write_partial("startup (bars + design + predictions registered, "
                  "committed at birth)")
    set_seed(324)                    # global init only; every RNG is its own

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
    e323m = json.loads(E323_METRICS.read_text(encoding="utf-8"))
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
        f"from artifacts: g0 {canon_row['post_g0']:.10f} gm12 "
        f"{canon_row['post_gm12']:.5f} norm {canon_row['write_norm']:.4f} "
        f"in-room {canon_row['in_own_room']:.4f} s1 {canon_row['s1_g0']:.2e} "
        f"(2nd room {canon_row['s1_second_room_e272']:.2e}) — PASS")

    # e272's room-lottery numbers, read at runtime (the comparator)
    e272_k10kr_g0 = e272m["arms"]["K10KR"]["install"]["post_cells"]["g0"]
    room_spread_same_form = (canon_row["post_g0"]
                             / max(e272_k10kr_g0, RATIO_DEN_FLOOR)) - 1.0
    room_spread_committed_rel = 1.0 - e272_k10kr_g0 / canon_row["post_g0"]
    e323_fresh_g0 = e323m["the_fresh_fact"]["post_g0"]

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
             "room_spread_same_form": room_spread_same_form,
             "note": "the room-lottery precedent (the census's comparator "
                     "+ the die's second-room confirmation)"}),
        "e290_metrics": _pbind(
            E290_METRICS, E290_MD5,
            {"write_norm": canon_row["write_norm"],
             "note": "the canon's write norm's own record (the dispatch's "
                     "chain EXTENDED with it — disclosed)"}),
        "e323_metrics": _pbind(
            E323_METRICS, E323_MD5,
            {"fresh_post_g0": e323_fresh_g0,
             "fresh_s1": e323m["the_fresh_fact"]["install_traj"][0]
                 ["g0_pz"],
             "note": "THE PARENT QUESTION's own record (the +81% miss + "
                     "the die-then-recover vs survive-and-sprawl fork this "
                     "cell maps)"}),
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
            and e323m["adjudication"]["verdict"] == "TEXTURE (G_FORMATION)"
            and abs(e272_k10kr_g0 - E272_K10KR_POST_G0) < 1e-15
            and abs(e323_fresh_g0 - E323_FRESH_POST_G0) < 1e-15
            and md5of_norm(E261_METRICS) == E261_MD5
            and md5of_norm(E264_METRICS) == E264_MD5
            and md5of_norm(E272_METRICS) == E272_MD5
            and md5of_norm(E290_METRICS) == E290_MD5
            and md5of_norm(E323_METRICS) == E323_MD5
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b G_PARENTS PASS — e264 {canon_row['post_g0']:.6f} (the canon) + "
        f"e272 K10KR {e272_k10kr_g0:.6f} (room spread same-form "
        f"{room_spread_same_form:.1%}; committed-rel "
        f"-{room_spread_committed_rel:.1%}) + e323 fresh "
        f"{e323_fresh_g0:.6f} + e261/e290 bound")
    write_partial("P0b parents hard-bound + the canon re-verified from "
                  "artifacts")
    del e264m, e272m, e290m, e323m, e261m

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
                "subspace the canon formed in, not a re-hash); certified "
                "in G_PROJ",
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

    # ---- G_GENFRESH: the four draws are FRESH --------------------------
    G_GENFRESH = {
        "form": "the four install gens are a FRESH block: distinct, not the "
                "committed 24314, absent from every family allocation, and "
                "exact-token absent from lab/*.py + runs/*/metrics.json "
                "(grepped at birth; the grep itself is a static disclosure "
                "recorded here, not a runtime gate)",
        "gens": list(FRESH_GENS),
        "all_distinct": bool(len(set(FRESH_GENS)) == len(FRESH_GENS)),
        "none_committed": bool(COMMITTED_INST_GEN not in FRESH_GENS),
        "none_in_family_allocations": bool(
            not (set(FRESH_GENS) & FAMILY_ALLOCATIONS)),
        "birth_grep": "exact-token (boundary-delimited) grep for 32401/"
                      "32402/32403/32404 over lab/*.py + runs/*/metrics.json "
                      "returned ZERO hits at birth (2026-10-09, pre-compute)",
        "pass": bool(len(set(FRESH_GENS)) == len(FRESH_GENS)
                     and COMMITTED_INST_GEN not in FRESH_GENS
                     and not (set(FRESH_GENS) & FAMILY_ALLOCATIONS)),
    }
    assert G_GENFRESH["pass"], f"fresh-gen gate FAILED: {G_GENFRESH}"
    metrics["gates"]["G_GENFRESH"] = G_GENFRESH
    log(f"P1 G_GENFRESH: {FRESH_GENS} distinct, not 24314, absent from the "
        f"family's allocations: PASS")
    write_partial("P1b G_GENFRESH PASSED")

    # ================= P2: THE FOUR INSTALLS (the committed rig) ========
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
        resume_ck = CKPT_DIR / (f"smoke_e324_gen{gen}_resume.pt" if SMOKE
                                else f"e324_gen{gen}_resume.pt")
        out = E261.chunked_install(
            f"GEN{gen}", ROOM_MODE, root_net, rooms, win_bank, inst_mask,
            anchor_full, train_ids, g0_ids, gm12_ids, r_eval_xy, zid,
            resume_ck, dev)
        sd = out["sd"]
        net = G1.evl_load(sd)
        flat = flat_params_cpu(net)
        flat_np = flat.double().numpy().astype(np.float64)
        post_g0 = G1.battery_cell(net, g0_ids, zid)["mean_pz"]
        post_gm12 = G1.battery_cell(net, gm12_ids, zid)["mean_pz"]
        post_gp12 = G1.battery_cell(net, bat_ids[12], zid)["mean_pz"]
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

        ck = save_ckpt(f"e324_gen{gen}", sd,
                       {"desc": f"e324 census draw gen {gen}: the committed "
                                f"root + e261's install rig VERBATIM (400 "
                                f"Dmix steps, HELD room seeds "
                                f"{HELD_ROOM_SEEDS}, gen {gen})",
                        "gen": int(gen), "post_g0": post_g0,
                        "write_norm": write_norm,
                        "in_own_room": loads["in_own_room"],
                        "s1_g0": s1_row["g0_pz"]})
        row = {
            "gen": int(gen),
            "draw_index": i_gen + 1,
            "role": "FRESH DRAW at the held room",
            "post_g0": post_g0, "post_gm12": post_gm12,
            "post_gp12": post_gp12, "ce_r": ce_r,
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
        log(f"GEN{gen} CENSUS ROW: g0 {post_g0:.6f} gm12 {post_gm12:.5f} "
            f"norm {write_norm:.4f} in-own-room {loads['in_own_room']:.4f} "
            f"s1 {s1_row['g0_pz']:.4f} kept-med {kept_med:.4f} | "
            f"G_FACTLOAD PASS G_PROTOIDENT PASS")
        write_partial(f"GEN{gen} complete (census row {i_gen + 1}/4)")

    # ================= P3: THE CENSUS + THE FROZEN BARS =================
    log("=" * 78)
    fresh_g0s = [census[g]["post_g0"] for g in FRESH_GENS]
    fresh_norms = [census[g]["write_norm"] for g in FRESH_GENS]
    fresh_inroom = [census[g]["in_own_room"] for g in FRESH_GENS]
    fresh_gm12s = [census[g]["post_gm12"] for g in FRESH_GENS]
    fresh_s1s = [census[g]["s1_g0"] for g in FRESH_GENS]

    def _ratio(vals):
        return max(vals) / max(min(vals), RATIO_DEN_FLOOR)

    spread_g0 = _ratio(fresh_g0s) - 1.0
    norm_ratio = _ratio(fresh_norms)
    inroom_ratio = _ratio(fresh_inroom)
    gm12_ratio = _ratio([max(v, RATIO_DEN_FLOOR) for v in fresh_gm12s])
    g0_dominates = bool(spread_g0 >= SPREAD_DOMINATES_BAR)
    tex_dominates = bool(norm_ratio > TEXTURE_2X_BAR
                         or inroom_ratio > TEXTURE_2X_BAR)
    room_comparable = bool(spread_g0 < SPREAD_ROOM_BAR)

    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        verdict = "TEXTURE (GATE FAILURE: " + ", ".join(
            k for k, g in hard.items() if not g.get("pass")) + ")"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif g0_dominates or tex_dominates:
        verdict = "INSTALL-DOMINATES"
        fired = []
        if g0_dominates:
            fired.append(f"SPREAD_g0 {spread_g0:.1%} >= "
                         f"{SPREAD_DOMINATES_BAR:.0%} (e272's same-form room "
                         f"value {room_spread_same_form:.1%})")
        if norm_ratio > TEXTURE_2X_BAR:
            fired.append(f"write norm varies {norm_ratio:.2f}x > "
                         f"{TEXTURE_2X_BAR:.0f}x")
        if inroom_ratio > TEXTURE_2X_BAR:
            fired.append(f"in-own-room varies {inroom_ratio:.2f}x > "
                         f"{TEXTURE_2X_BAR:.0f}x")
        clause = (
            "the install lottery IS the formation wheel: " + "; ".join(fired)
            + f" — the four fresh gens at the canon's OWN room span g0 "
            f"[{min(fresh_g0s):.4f}, {max(fresh_g0s):.4f}] vs the canon "
            f"{canon_row['post_g0']:.4f} (the room lottery spans only "
            f"{room_spread_same_form:.1%} at held gen). Every canon number "
            "gains the census's spread as its error bar; Law 3 drafts "
            "with it.")
    elif room_comparable:
        verdict = "ROOM-COMPARABLE"
        clause = (
            f"the fresh gens' g0 spread {spread_g0:.1%} lands within "
            f"{SPREAD_ROOM_BAR:.0%} (comparable to e272's room spread "
            f"{room_spread_same_form:.1%}) — e323's miss was a tail draw "
            f"(or the room x gen interaction), not a wider gen wheel: the "
            "canon's monoculture caveat softens to a two-wheel lottery of "
            "similar sizes")
    else:
        verdict = "BETWEEN-WHEELS"
        clause = (
            f"the fresh gens' g0 spread {spread_g0:.1%} lands in the GAP "
            f"the two bars leave [{SPREAD_ROOM_BAR:.0%}, "
            f"{SPREAD_DOMINATES_BAR:.0%}) with no texture axis > "
            f"{TEXTURE_2X_BAR:.0f}x (norm {norm_ratio:.2f}x, in-room "
            f"{inroom_ratio:.2f}x) — wider than the room wheel "
            f"({room_spread_same_form:.1%}), not materially dominating at "
            "this n; the honest in-between, disclosed as registered at "
            "birth")

    # ---- the s1-fork (a finding, never a bar) ---------------------------
    all5 = [(COMMITTED_INST_GEN, canon_row["s1_g0"],
             canon_row["write_norm"], canon_row["in_own_room"],
             canon_row["post_gm12"], canon_row["post_g0"], "CANON")
            ] + [(g, census[g]["s1_g0"], census[g]["write_norm"],
                  census[g]["in_own_room"], census[g]["post_gm12"],
                  census[g]["post_g0"], "FRESH") for g in FRESH_GENS]

    def _side(s1):
        if s1 < S1_DIE_BAR:
            return "DIE"
        if s1 > S1_GAP_HI:
            return "SURVIVE"
        return "GAP"

    s1_table = [{"draw": lbl, "gen": g, "s1_g0": s1, "side": _side(s1),
                 "g0": g0v, "norm": nm, "in_own_room": ir, "gm12": gm}
                for g, s1, nm, ir, gm, g0v, lbl in all5]
    in_gap = [r for r in s1_table if r["side"] == "GAP"]
    sides = [r["side"] for r in s1_table if r["side"] != "GAP"]
    surv_norms = [r["norm"] for r in s1_table if r["side"] == "SURVIVE"]
    die_norms = [r["norm"] for r in s1_table if r["side"] == "DIE"]
    clean_bimodal = bool(not in_gap and "SURVIVE" in sides and "DIE" in sides)
    if surv_norms and die_norms:
        bulk_sep = bool(min(surv_norms) > max(die_norms))
    else:
        bulk_sep = None      # one-sided census: separation vacuously n/a
    if clean_bimodal and bulk_sep:
        s1_fork = "S1-FORK-CLEARS"
    elif in_gap:
        s1_fork = "S1-FORK-MUDDY"
    else:
        s1_fork = "S1-FORK-MUDDY"
    s1_finding = {
        "table": s1_table,
        "bars": {"die": S1_DIE_BAR, "gap_hi": S1_GAP_HI,
                 "clean_bimodal": clean_bimodal,
                 "bulk_separation_min_survive_gt_max_die": bulk_sep},
        "finding": s1_fork,
        "clause": (
            (f"the draws split cleanly by step-one survival (all "
              f"{len(s1_table)} census draws outside the [0.01, 0.15] gap: "
              f"{sides.count('SURVIVE')} SURVIVE / {sides.count('DIE')} DIE) "
              + (f"AND the SURVIVE side's min write norm {min(surv_norms):.2f}"
                 f" > the DIE side's max {max(die_norms):.2f} — survival "
                 "predicts the bulky texture" if bulk_sep
                 else ("AND the bulk clause FAILED: the SURVIVE side's min "
                       f"norm {min(surv_norms):.2f} <= the DIE side's max "
                       f"{max(die_norms):.2f}" if surv_norms and die_norms
                       else "with a ONE-SIDED census (no DIE draws among "
                            "the fresh four + canon is the only DIE — the "
                            "separation clause reads on n_die=1)")))
            if not in_gap else
            (f"the fork is MUDDY: draw(s) {[r['gen'] for r in in_gap]} "
             "land INSIDE the [0.01, 0.15] no-man's land — no clean "
             "bimodality at this n")),
        "one_sided_note": (None if (surv_norms and die_norms) else
                           "the census has DIE draws only from the canon "
                           "(gen 24314) — a one-sided fork read, disclosed"),
    }
    # co-descriptive: survival vs in-room ordering
    surv_ir = [r["in_own_room"] for r in s1_table if r["side"] == "SURVIVE"]
    die_ir = [r["in_own_room"] for r in s1_table if r["side"] == "DIE"]
    s1_finding["descriptive"] = {
        "survive_inroom_range": (min(surv_ir), max(surv_ir))
        if surv_ir else None,
        "die_inroom_range": (min(die_ir), max(die_ir)) if die_ir else None,
        "survive_gm12_range": (
            min(r["gm12"] for r in s1_table if r["side"] == "SURVIVE"),
            max(r["gm12"] for r in s1_table if r["side"] == "SURVIVE"))
        if "SURVIVE" in sides else None,
        "die_gm12_range": (
            min(r["gm12"] for r in s1_table if r["side"] == "DIE"),
            max(r["gm12"] for r in s1_table if r["side"] == "DIE"))
        if "DIE" in sides else None,
    }

    # ---- the canon's position -------------------------------------------
    all_g0 = sorted([(canon_row["post_g0"], COMMITTED_INST_GEN)]
                    + [(census[g]["post_g0"], g) for g in FRESH_GENS])
    canon_rank_g0 = [g for _, g in all_g0].index(COMMITTED_INST_GEN) + 1
    all_norm = sorted([(canon_row["write_norm"], COMMITTED_INST_GEN)]
                      + [(census[g]["write_norm"], g) for g in FRESH_GENS])
    canon_rank_norm = [g for _, g in all_norm].index(COMMITTED_INST_GEN) + 1
    all_ir = sorted([(canon_row["in_own_room"], COMMITTED_INST_GEN)]
                    + [(census[g]["in_own_room"], g) for g in FRESH_GENS])
    canon_rank_ir = [g for _, g in all_ir].index(COMMITTED_INST_GEN) + 1

    def _pos(rank):
        return ("EXTREME (min)" if rank == 1
                else "EXTREME (max)" if rank == len(all_g0)
                else "MEDIAN" if rank == (len(all_g0) + 1) // 2
                else "INTERIOR")

    canon_position = {
        "rank_g0": canon_rank_g0, "of": len(all_g0),
        "position_g0": _pos(canon_rank_g0),
        "rank_write_norm": canon_rank_norm,
        "position_write_norm": _pos(canon_rank_norm),
        "rank_in_own_room": canon_rank_ir,
        "position_in_own_room": _pos(canon_rank_ir),
        "ordered_g0": [(round(v, 6), g) for v, g in all_g0],
        "clause": (f"gen 24314 sits at rank {canon_rank_g0}/"
                   f"{len(all_g0)} on g0 ({_pos(canon_rank_g0)}), rank "
                   f"{canon_rank_norm}/{len(all_norm)} on write norm "
                   f"({_pos(canon_rank_norm)}), rank {canon_rank_ir}/"
                   f"{len(all_ir)} on in-own-room "
                   f"({_pos(canon_rank_ir)})"),
    }

    log("=" * 78)
    log(f"E324 VERDICT: {verdict}")
    log(f"  the census (g0): canon {canon_row['post_g0']:.6f} | fresh "
        + " ".join(f"{g}:{census[g]['post_g0']:.6f}" for g in FRESH_GENS))
    log(f"  SPREAD_g0 {spread_g0:.1%} (bar >= {SPREAD_DOMINATES_BAR:.0%} "
        f"DOMINATES / < {SPREAD_ROOM_BAR:.0%} ROOM); e272 same-form room "
        f"value {room_spread_same_form:.1%}")
    log(f"  texture: norm x{norm_ratio:.2f} "
        f"[{min(fresh_norms):.2f},{max(fresh_norms):.2f}] vs canon "
        f"{canon_row['write_norm']:.2f}; in-room x{inroom_ratio:.2f} "
        f"[{min(fresh_inroom):.4f},{max(fresh_inroom):.4f}] vs canon "
        f"{canon_row['in_own_room']:.4f}; gm12 x{gm12_ratio:.2f}")
    log(f"  s1 fork: {s1_fork} — {s1_finding['clause']}")
    log(f"  canon position: {canon_position['clause']}")
    log(f"  {clause}")
    log("=" * 78)

    metrics["the_census"] = {
        "canon": canon_row,
        "fresh": {f"GEN{g}": census[g] for g in FRESH_GENS},
        "axes_table": {
            "g0": {"canon": canon_row["post_g0"],
                   **{f"gen{g}": census[g]["post_g0"]
                      for g in FRESH_GENS}},
            "write_norm": {"canon": canon_row["write_norm"],
                           **{f"gen{g}": census[g]["write_norm"]
                              for g in FRESH_GENS}},
            "in_own_room": {"canon": canon_row["in_own_room"],
                            **{f"gen{g}": census[g]["in_own_room"]
                               for g in FRESH_GENS}},
            "gm12": {"canon": canon_row["post_gm12"],
                     **{f"gen{g}": census[g]["post_gm12"]
                        for g in FRESH_GENS}},
            "s1_g0": {"canon": canon_row["s1_g0"],
                      **{f"gen{g}": census[g]["s1_g0"]
                         for g in FRESH_GENS}},
        },
        "s1_fork": s1_finding,
        "canon_position": canon_position,
        "comparators": {
            "e272_room_lottery_same_form": room_spread_same_form,
            "e272_room_lottery_committed_rel": -room_spread_committed_rel,
            "e323_fresh_draw_g0_both_wheels": e323_fresh_g0,
            "e323_fresh_draw_ratio_vs_canon": (
                e323_fresh_g0 / canon_row["post_g0"] - 1.0),
            "formation_band_context_only": list(FORMATION_BAND),
        },
    }
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any hard-gate failure) -> "
                           "INSTALL-DOMINATES (SPREAD_g0 >= 40% OR texture "
                           "> 2x) -> ROOM-COMPARABLE (SPREAD_g0 < 25%) -> "
                           "BETWEEN-WHEELS (the disclosed gap) — frozen at "
                           "birth",
        "gates_pass": gates_pass,
        "reads": {
            "spread_g0": spread_g0,
            "spread_bars": {"dominates": SPREAD_DOMINATES_BAR,
                            "room": SPREAD_ROOM_BAR},
            "norm_ratio": norm_ratio,
            "inroom_ratio": inroom_ratio,
            "gm12_ratio": gm12_ratio,
            "fresh_g0s": fresh_g0s,
            "e272_room_same_form": room_spread_same_form,
            "s1_fork_finding": s1_fork,
            "canon_position": canon_position["clause"],
        },
        "prediction_scoring": {
            "P-e324a_install_dominates": bool(verdict == "INSTALL-DOMINATES"),
            "P-e324b_room_comparable": bool(verdict == "ROOM-COMPARABLE"),
        },
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P3 the census adjudicated (the frozen bars + the s1-fork "
                  "+ the canon's position)")

    # ================= P4: honesty + provenance + the artifacts ==========
    metrics["honesty"] = {
        "no_shopping": ("four draws registered at birth, four draws run, "
                        "four draws reported — wherever they landed; the "
                        "[0.15, 0.45] band drawn as context only, never a "
                        "gate"),
        "canon_never_rerun": ("gen 24314's row read from e264/e290/e272's "
                              "artifacts at runtime + cross-asserted vs the "
                              "birth literals"),
        "n4_caveat": ("a 4-draw census prices the wheel's COARSE spread; "
                      "tails (the e323 draw was one sample of BOTH wheels "
                      "fresh) remain n=1-anchored; nothing guaranteed"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "thermal_envelope": _envelope_summary(),
    }
    make_census_plot(RD, census, canon_row, s1_table, spread_g0,
                     norm_ratio, inroom_ratio, verdict,
                     e272_k10kr_g0, e323_fresh_g0)
    write_report(RD, metrics, census, canon_row, s1_finding, canon_position,
                 spread_g0, norm_ratio, inroom_ratio, gm12_ratio, verdict,
                 clause, room_spread_same_form, e272_k10kr_g0,
                 e323_fresh_g0)
    metrics["status"] = ("RUN RECORDED — the census adjudicated; the "
                         "heartbeat folds" if not SMOKE else
                         "SMOKE — shakedown only, nothing adjudicated")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e324_install_census.png"),
                          str(RD / "REPORT.md")]
    write_partial("DONE (census + plot + report)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ the plot
def make_census_plot(rd, census, canon_row, s1_table, spread_g0,
                     norm_ratio, inroom_ratio, verdict, e272_g0,
                     e323_g0) -> None:
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    # the room-lottery band (e272's gen-24314 family across two rooms)
    ax.axvspan(e272_g0, canon_row["post_g0"], color="tab:gray", alpha=0.18,
               label=f"room lottery (e272: gen 24314 across 2 rooms, "
                     f"{(canon_row['post_g0'] / e272_g0 - 1):.0%} wide)")
    # the trust band as context (dashed)
    ax.axvline(FORMATION_BAND[0], color="tab:purple", ls=":", lw=1.2)
    ax.axvline(FORMATION_BAND[1], color="tab:purple", ls=":", lw=1.2,
               label="formation trust band [0.15, 0.45] (context, NOT a gate)")
    ax.axvline(e323_g0, color="tab:green", ls="--", lw=1.0, alpha=0.7,
               label=f"e323's both-wheels-fresh draw {e323_g0:.4f}")
    # stagger the fresh-draw labels by x-rank (the four draws cluster tight:
    # the render catch caught by inspection post-run; disclosed) — alternate
    # above/below/below-left so the annotations never pile
    _xrank = {g: i for i, (v, g) in enumerate(
        sorted((census[g]["post_g0"], g) for g in census))}
    _offsets = [(8, 8), (8, -18), (-76, -20), (8, 8)]
    for g, row in census.items():
        side = ("DIE" if row["s1_g0"] < S1_DIE_BAR
                else "SURVIVE" if row["s1_g0"] > S1_GAP_HI else "GAP")
        col = {"DIE": "tab:blue", "SURVIVE": "tab:red",
               "GAP": "tab:orange"}[side]
        ax.scatter(row["post_g0"], row["in_own_room"], s=90, c=col,
                   marker="o", zorder=3, edgecolors="k", linewidths=0.6)
        ax.annotate(f"gen {g}\ns1 {row['s1_g0']:.3f}",
                    (row["post_g0"], row["in_own_room"]),
                    textcoords="offset points",
                    xytext=_offsets[_xrank[g] % len(_offsets)],
                    fontsize=7.5, color=col,
                    ha="left" if _offsets[_xrank[g] % len(_offsets)][0] > 0
                    else "right")
    ax.scatter(canon_row["post_g0"], canon_row["in_own_room"], s=340,
               c="gold", marker="*", zorder=4, edgecolors="k",
               linewidths=1.0)
    ax.annotate(f"THE CANON gen {COMMITTED_INST_GEN}\n"
                f"s1 {canon_row['s1_g0']:.1e} (DIE)",
                (canon_row["post_g0"], canon_row["in_own_room"]),
                textcoords="offset points", xytext=(-6, -38),
                fontsize=8, fontweight="bold", ha="right")
    ax.margins(y=0.14)
    ax.set_xlabel("post-install g0 (the draw's own battery)")
    ax.set_ylabel("in-own-room fraction (write displacement)")
    ax.set_title(f"e324 THE INSTALL-DRAW CENSUS — held room K10K "
                 f"(seeds {HELD_ROOM_SEEDS[0]}/{HELD_ROOM_SEEDS[1]})\n"
                 f"fresh-g0 spread {spread_g0:.0%} (norm x"
                 f"{norm_ratio:.2f}, in-room x{inroom_ratio:.2f}) -> "
                 f"{verdict}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)

    axt = fig.add_subplot(gs[0, 1])
    axt.axis("off")
    cols = ["draw", "gen", "s1 g0", "side", "g0", "norm", "in-room", "gm12"]
    rows = [[r["draw"], str(r["gen"]),
             (f"{r['s1_g0']:.1e}" if r["s1_g0"] < 0.01
              else f"{r['s1_g0']:.4f}"),
             r["side"], f"{r['g0']:.4f}", f"{r['norm']:.2f}",
             f"{r['in_own_room']:.3f}", f"{r['gm12']:.4f}"]
            for r in s1_table]
    tbl = axt.table(cellText=rows, colLabels=cols, loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.7)
    for (i, _j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#d0d0d0")
        elif s1_table[i - 1]["draw"] == "CANON":
            cell.set_facecolor("#fff2b2")
        elif s1_table[i - 1]["side"] == "SURVIVE":
            cell.set_facecolor("#ffd9d0")
        elif s1_table[i - 1]["side"] == "DIE":
            cell.set_facecolor("#d0e4ff")
    axt.set_title("the s1 fork (DIE < 0.01 < GAP < 0.15 < SURVIVE) — "
                  "a finding, never a bar", fontsize=9)
    fig.suptitle("THE INSTALL-DRAW LOTTERY'S FIRST DISTRIBUTION — the gen "
                 "wheel at HELD room (e324; 4 fresh gens + the canon, "
                 "e261/e264 rig verbatim)", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(rd / "e324_install_census.png", dpi=130)
    plt.close(fig)
    log(f"[plot] wrote {rd / 'e324_install_census.png'}")


# ------------------------------------------------------------------ the report
def write_report(rd, m, census, canon_row, s1_finding, canon_position,
                 spread_g0, norm_ratio, inroom_ratio, gm12_ratio, verdict,
                 clause, room_spread, e272_g0, e323_g0) -> None:
    adj = m["adjudication"]
    env = m["provenance"]["thermal_envelope"]
    gates = m["gates"]
    L: list[str] = []
    L.append("# e324 — THE INSTALL-DRAW CENSUS "
             f"(run {m['date']})\n")
    L.append("**THE QUESTION** (dispatch verbatim): *what is the "
             "install-draw lottery's spread on the formation texture "
             "axes, at HELD room?*\n")
    L.append("## The census table (all draws, all axes)\n")
    L.append("| draw | gen | s1 g0 (side) | post g0 | gm12 | write norm | "
             "in-own-room |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| THE CANON (not rerun; artifacts) | {canon_row['gen']} | "
             f"{canon_row['s1_g0']:.2e} (DIE; 2nd room "
             f"{canon_row['s1_second_room_e272']:.2e}) | "
             f"{canon_row['post_g0']:.6f} | {canon_row['post_gm12']:.5f} | "
             f"{canon_row['write_norm']:.4f} | "
             f"{canon_row['in_own_room']:.4f} |")
    for g, row in census.items():
        side = s1_finding["table"][[r["gen"] for r in
                                    s1_finding["table"]].index(g)]["side"]
        L.append(f"| FRESH {row['draw_index']}/4 | {g} | "
                 f"{row['s1_g0']:.4f} ({side}) | {row['post_g0']:.6f} | "
                 f"{row['post_gm12']:.5f} | {row['write_norm']:.4f} | "
                 f"{row['in_own_room']:.4f} |")
    L.append("")
    L.append("## The frozen bars, scored\n")
    L.append(f"- **SPREAD_g0** (max/min - 1 over the four fresh draws) = "
             f"**{spread_g0:.1%}** (bar: >= 40% INSTALL-DOMINATES / < 25% "
             f"ROOM-COMPARABLE; e272's same-form room value "
             f"{room_spread:.1%})")
    L.append(f"- **write norm varies x{norm_ratio:.2f}** (bar > 2x); "
             f"**in-own-room varies x{inroom_ratio:.2f}** (bar > 2x); "
             f"gm12 varies x{gm12_ratio:.2f} (co-reported)")
    L.append(f"- **VERDICT: {verdict}**\n")
    L.append(f"> {clause}\n")
    L.append("## The s1-fork finding (never a bar)\n")
    L.append(f"**{s1_finding['finding']}** — {s1_finding['clause']}\n")
    L.append("## The canon's position\n")
    L.append(f"{canon_position['clause']} — ordered g0: "
             + ", ".join(f"{g}={v:.4f}" for v, g in
                         canon_position["ordered_g0"]) + "\n")
    L.append("## Comparators\n")
    L.append(f"- e272's room lottery (same form): {room_spread:.1%} "
             f"(K10KR {e272_g0:.4f} vs canon "
             f"{canon_row['post_g0']:.4f}, same gen 24314, two rooms)")
    L.append(f"- e323's both-wheels-fresh draw: {e323_g0:.4f} "
             f"(x{e323_g0 / canon_row['post_g0']:.2f} the canon) — the "
             "census's held-room spread prices how much of that was the "
             "room's\n")
    L.append("## Gates\n")
    L.append(f"- {sum(1 for g in gates.values() if g.get('pass'))}/"
             f"{len(gates)} named gates PASS"
             + (" (all PASS)" if all(g.get("pass") for g in gates.values())
                else " — FAILURES: " + ", ".join(
                    k for k, g in gates.items() if not g.get("pass"))))
    L.append(f"- per-draw: G_PROTOIDENT 4/4 (the committed rig's protocol "
             f"identity, verified from each run's own optimizer artifact) + "
             f"G_FACTLOAD 4/4 (read-determinism re-loads)\n")
    L.append("## Envelope\n")
    L.append(f"- bursts <= {env['burst_cap_s']:.0f}s, cooldowns "
             f"{env['cooldown_s']:.0f}s, hard line {env['hard_line_c']:.0f}C; "
             f"{env['n_polls']} polls tagged e324:*; max temp "
             f"{env['max_temp_seen_c']}C; violations >= "
             f"{env['hard_line_c']:.0f}C: {env['violations_ge_84c']}\n")
    L.append("## Prediction scoring\n")
    L.append(f"- P-e324a (lab guess, adopted): INSTALL-DOMINATES — "
             f"{'HIT' if adj['prediction_scoring']['P-e324a_install_dominates'] else 'MISS'}")
    L.append(f"- P-e324b (executor counter): ROOM-COMPARABLE — "
             f"{'HIT' if adj['prediction_scoring']['P-e324b_room_comparable'] else 'MISS'}"
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
