"""E260 — THE RANK-MATCHED-RANDOM INSTALL (T235's decisive discriminator;
the anti-substrate's two suspects). Design dispatched 2026-10-05; this
docstring carries the registered bars VERBATIM, committed at birth BEFORE
any compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (verbatim from the dispatch): is the install-unwritability
the RANK RESTRICTION itself (any rank-k projected install fails) or the
SPAN'S SPECIFIC DIRECTIONS (only the corpus work's low-rank attractor
bars writes)? e246: ALIGNED (rank-2, the span) expressed NOTHING while
ORTHO (the complement) landed; e258: both v-heavy and v-light landed
(span-balanced). THE DECISIVE ARM: a rank-matched RANDOM subspace install
— if it fails like ALIGNED, the rank restriction is the barrier; if it
lands, the span's specific structure is.

THE CELL (2.74M, the g1c fresh-root conventions; e258's cell shape):
  1. ARM-RANDOM: the install gradient projected onto a RANDOM rank-k
     orthonormal subspace (k matched to the committed root-displacement's
     k50 convention e258 froze — the same k; the draw seed registered);
  2. ARM-SPAN control (the ALIGNED re-run at the SAME k, must reproduce
     e246's failure signature at matched k — the gate);
  3. ARM-FREE (the natural install, must reproduce the committed g1c
     root);
  4. identical dose/steps; the landing readouts (g0 expression;
     matched-strength band);
  5. THE WASH on whatever lands (the retentions for the record).

REGISTERED BARS (frozen in the dispatch letter, VERBATIM in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping):
  - RANK-IS-THE-BARRIER — "ARM-RANDOM fails to express (g0 ~ 0 or
    cannot land) like the span control — ANY rank-k restriction bars
    installs; the anti-substrate is dimensional: new memories need
    full-space (or at least much-higher-rank) room; the writing is
    bottlenecked by expressivity, not by the corpus's occupancy"
  - SPAN-IS-SPECIAL — "ARM-RANDOM lands at matched strength while the
    span control still fails — the barrier is the corpus work's specific
    directions (the low-rank attractor), not rank; the anti-substrate is
    occupancy: the ongoing work's own subspace bars writes (an
    interference object, e246's H-i confirmed causally)"
  - MIXED — "any between — the trajectories verbatim, no inflation"

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE RANK k := THE SAME k e258 froze — clip(k50 of the committed g1c
    ROOT displacement's squared-L2 support, 100_000, 400_000), recomputed
    from committed checkpoints in P1 and HARD-BOUND equal to e258's
    committed k 237,123 (assert; the k-matching disclosed exactly);
    smoke: fixed 512. BOTH rooms are rank k by construction AND by
    measurement (the kept^2 probe: ||P x||^2/||x||^2 ~= k/N, bar +-0.02).
  * ARM-RANDOM's room := the SRCT ensemble (a Subsampled Random Cosine
    Transform — the standard feasible random dense subspace at k=237k,
    where a Haar-uniform basis alone would be N x k = 2.6 TB): the k
    columns {D . phi_s : s in S}, D a +-1 diagonal (torch CPU generator,
    seed 26011), S a k-subset of {0..N-1} (randperm, seed 26012), phi_j
    the orthonormal DCT-II basis vectors (scipy 'ortho', fp64 pocketfft)
    — DENSE (every direction full-support over ALL parameters, like the
    span's own dense directions, unlike e258's axis-aligned coordinate
    rooms), EXACTLY orthonormal and an EXACT orthogonal projector:
    P_R x = D . idct(mask_S(dct(D . x))); certified (roundtrip,
    idempotency, kept^2) to 1e-8 in G_PROJ.
  * ARM-SPAN's room := the committed late span (e246_late_span.pt,
    md5 + rank-10 hard-bound) EXTENDED to the SAME k with an INDEPENDENT
    SRCT filler (seeds 26013/26014, k-10 columns): a rank-k subspace that
    CONTAINS the span's specific directions. The projector is the EXACT
    orthogonal projector onto span{Vp, filler} — the Gram Woodbury solve
    (a 30-dim correction around the identity, fp64); certified by
    containment (P_S v_j = v_j to 1e-5, the Vp's own fp32-storage dev is
    2e-6) + idempotency + the kept^2 rank probe.
  * THE HOOK (e237/e246/e258's order): backward -> clip_grad_norm_ 1.0
    VERBATIM -> PROJECT (CPU fp64; the write fp32) -> opt.step(); the
    norm is NOT rescaled (e237's convention); ARM-FREE runs the ledger
    dots READ ONLY (the verbatim trajectory).
  * "fails to express (g0 ~ 0)" := post-install g0 < 0.05 (the frozen
    g0~0 floor; e246's committed textures: ALIGNED 2.9e-5, ORTHO 0.4505,
    FREE 0.5302); "cannot land" := root g0 under the matched band floor;
    "lands at matched strength" := root g0 within the +-10% band around
    the committed g1c root g0 0.7447534203529358 -> [0.670278,
    0.819229] AND install-carried (post-install g0 >= 0.05 — the
    cons-rescue separation, e258's tier discipline).
  * RANK-IS-THE-BARRIER fires iff gates pass AND ARM-RANDOM fails (g0~0
    or cannot land) AND the span control fails (the bar's own "like the
    span control" conjunction — both rooms fail together at matched k).
  * SPAN-IS-SPECIAL fires iff gates pass AND ARM-RANDOM landed AND
    expresses AND the span control fails.
  * MIXED: everything else. Composite order: TEXTURE (gate failure;
    G_FREE failure HALTS) -> RANK-IS-THE-BARRIER -> SPAN-IS-SPECIAL ->
    MIXED.
  * THE SPAN-CONTROL GATE LANGUAGE ("must reproduce e246's failure
    signature at matched k — the gate") is handled as the REGISTERED
    CONTROL READ: reproduction is measured and reported (and required
    for SPAN-IS-SPECIAL); NON-reproduction is the honest MIXED branch,
    NOT a halt — a hard gate would adjudicate the finding at birth, and
    the dispatch's own MIXED bar covers "any between".
  * THE WASH (frozen): the g-cell W1 form VERBATIM — commit(R=0.7 RAW
    L2) at each LANDED arm's own root, then the seed-10902 300-step
    neutral wash; landed arms only; retention = min g-12 over
    {10,50,100,300} / own root g-12 (the full-8 + +300 flavors
    co-reported).
  * THE V-MAP is LOADED from e258's committed artifact
    (runs/checkpoints/e258_vmap.pt — the lab's certified 2.74M standing-
    load map; extend, don't repeat): it feeds the measured v-excess
    ledger (e257's qov convention at step level) and NOTHING else; no
    new history is run. The installs' own fresh Adam state is the
    mechanism's channel (e258's disclosed scope).
  * MEASURED, NEVER NOMINAL: per-step kept fraction ||g'||/||g||,
    v-excess pre/post projection, in-span fraction pre AND applied
    (||Vp g'||/||g'||); the per-arm overlap with the span at three
    levels — the subspace probes (containment; ||P_R v_j|| per span
    direction), the per-step gradient fractions, the displacement
    cos-to-span + in-own-room fractions at post-install and root.
  * THE CONSOLIDATION runs NATURAL on all arms (e113 VERBATIM, seed
    10901 HELD): the intervention is the INSTALL geometry only; cons
    re-growth outside the room is part of the honest design and is
    measured by the final root's in-own-room fraction.

CHECKS (the dispatch's, in force): e258's machinery smoke FIRST (its
record: 2-3 bugs caught per build); the k-matching disclosed exactly;
the random draw seed registered; the measured per-arm overlap with the
span reported (not nominal); n=1 per arm (the lottery caveat); nothing
guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
g-series' own reference scale). GPU trainings (3 installs s400 + 3 cons
s300 + up to 3 wall washes s300), each in <= 175 s bursts (within the
owner's <= 180 s window), per-step thermal polls at a 78C margin from
the first burst, 40 s cooldowns (the 30-60 s window), the 84C
never-past line recorded; CPU probing threads 4; the dense projections
run CPU fp64 (pocketfft workers 2).

Outputs: runs/e260/{metrics.json (PROGRESSIVE), e260_rank_matched.png,
e260_subspace_instrument.png, run.log}; checkpoints
runs/checkpoints/e260_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e260_rank_matched.py    (E260_SMOKE=1 shakedown)
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
from types import SimpleNamespace

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
import e228_margin_landscape as E228                   # noqa: E402 — THE
                                                      # margin instrument
                                                      # (margin_pass, VERBATIM;
                                                      # opens runs/e228_run.log
                                                      # append as a module
                                                      # side effect — benign,
                                                      # e242's precedent)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E260_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e260_smoke" if SMOKE else "e260"
assert torch.cuda.is_available(), "e260 owns the GPU lane (dispatch)"

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
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the rooms' core)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
CKPT_DIR = GB.CKPT_DIR
FRESH_GEN = 24314                 # the g1c fresh draw's install gen (VERBATIM)
CONS_SEED = 10901                 # HELD (g2e: the cons stream is shared)
WASH_SEED = 10902                 # HELD (the lineage's locked wash seed)
INST_STEPS = 400 if not SMOKE else 8
INST_TOTAL = 1000                 # e048's house-cosine total (verbatim)
CONS_STEPS = 300 if not SMOKE else 8
WASH_STEPS = 300 if not SMOKE else 4
R_CLAIM = 0.7                     # RAW L2 — the 2.74M convention (verbatim)
MATCH_BAND = 0.10                 # the pre-registered +-10% matched band

CK_WASH: tuple[int, ...] = (1, 2, 4, 10, 50, 100, 200, 300) if not SMOKE \
    else (1, 2, 4)
READ_GRID: tuple[int, ...] = (1, 2, 10, 50, 100, 300) if not SMOKE \
    else (1, 2, 4)
FLAT_260: tuple[int, ...] = tuple(s for s in (10, 50, 100, 300)
                                  if s in READ_GRID)      # the primary flat set
FLAT_FULL8: tuple[int, ...] = tuple(s for s in (10, 50, 100, 200, 300)
                                    if s in CK_WASH)      # e225's set (co-report)

# the rank rule (frozen): the SAME k e258 froze — k50 of the committed root
# displacement, clipped, HARD-BOUND to e258's committed record
K_QUANT = 0.5
K_MIN, K_MAX = 100_000, 400_000
K_SMOKE = 512                      # smoke: fixed, no committed-record anchor
E258_K_HARD = 237_123              # e258's committed k (the k-matching bar)

# THE REGISTERED RANDOM-DRAW SEEDS (frozen here; disclosed in metrics)
SEED_D_RAND = 26011                # ARM-RANDOM's +-1 diagonal
SEED_S_RAND = 26012                # ARM-RANDOM's index set S
SEED_D_FILL = 26013                # ARM-SPAN's filler +-1 diagonal
SEED_S_FILL = 26014                # ARM-SPAN's filler index set T

G0_ZERO_FLOOR = 0.05               # the frozen "g0 ~ 0" expression floor

G_READ_TOL = 5e-3                  # the family's cross-device read tolerance

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
G1C_METRICS = E43.REPO / "runs" / "g1c_root" / "metrics.json"
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562
G1C_ROOT_G0 = 0.7447534203529358      # THE landing anchor (the g0 battery)
G1C_POST_INSTALL = {"gm12": 0.16888952255249023,   # hard-bound (Rule 12)
                    "g0": 0.5302218198776245,
                    "ce_r": 1.673575520515442}
G1C_W1_GM12 = {
    1: 0.821441113948822, 2: 0.9093567132949829, 4: 0.8956068754196167,
    10: 0.9277286529541016, 50: 0.9397175908088684, 100: 0.9368361234664917,
    200: 0.9264556169509888, 300: 0.9405527114868164,
}
G1C_C_STEP1 = {"ce": 1.3502024412155151, "disp": 1.654317855834961}
E185_XHASH_1 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's stored step-1 md5

# e246's committed textures (the anti-substrate's own record; hard-bound)
E246_METRICS = E43.REPO / "runs" / "e246" / "metrics.json"
E246_VERDICT = "GEOMETRY-IRRELEVANT"
E246_ALIGNED_POST_G0 = 2.8589747671503574e-05      # the anti-substrate's g0~0
E246_ORTHO_POST_G0 = 0.4504633843898773
E246_FREE_POST_G0 = 0.5302095413208008
E246_SPAN_MD5 = "3464ff6081d8e402103dd9ca65bfb53a"
E246_SPAN_RANK = 10

# e258's committed record (the direct predecessor; hard-bound)
E258_METRICS = E43.REPO / "runs" / "e258" / "metrics.json"
E258_VERDICT = "NOT-THE-LOAD"
E258_FREE_VEXC_PRE_MEDIAN = 29.061850539663425     # co-report texture (non-gate)

# the draw spread (e225's committed 2.74M wall-family retention band)
E225_METRICS = E43.REPO / "runs" / "e225" / "metrics.json"
DRAW_SPREAD = {"lo": 0.6153917590189493, "hi": 1.3266428034732003,
               "members": {"g1b": 0.9979692766675983,
                           "g1c": 1.0263911969648605,
                           "g1d": 1.3266428034732003,
                           "g1e": 0.6153917590189493,
                           "g1f": 0.9317064573129114}}

ARMS = ("FREE", "RANDOM", "SPAN")     # FREE runs FIRST (the halt gate)
ARM_DESC = {
    "RANDOM": "each install gradient PROJECTED onto a RANDOM rank-k "
              "orthonormal subspace (the SRCT ensemble: k dense random "
              "directions, seeds 26011/26012; k = e258's frozen 237,123) "
              "before Adam — the rank-matched control with NO corpus "
              "structure in the room",
    "SPAN": "each install gradient projected onto the committed late span "
            "EXTENDED to the SAME k (10 span directions + k-10 independent "
            "dense random filler, seeds 26013/26014; the exact Woodbury "
            "projector) — the ALIGNED re-run at matched k: the room "
            "CONTAINS the corpus work's own directions",
    "FREE": "the natural install through the instrumented path (fp64 "
            "ledger dots READ ONLY, no write) — must reproduce the "
            "committed g1c root or the cell HALTS",
}

# ---- THE THERMAL ENVELOPE (the e246/e258 discipline on this chip; the
# owner's max-priority window): per-step polls from the FIRST burst at the
# 78C margin; bursts <= 175 s; cooldowns 40 s (the 30-60 s window); the 84C
# never-past line recorded (e258 matched: max 80.0C, zero >= 84C).
BURST_MAX_S = 175.0
COOLDOWN_S = 40.0
TEMP_EARLY_END = 78.0              # per-step-poll burst-end margin
TEMP_HARD = 84.0                   # the recorded never-past line
POLL_EVERY = 1                     # PER-STEP mid-burst temp polls
LAUNCH_POLL_GAP_S = 5.0
CHUNK_DOT = 4_000_000              # per-parameter fp64 chunk (e237's cut)
DCT_WORKERS = 2                    # pocketfft workers (shared machine)

REGISTERED = {
    "question_verbatim": "is the install-unwritability the RANK RESTRICTION "
        "itself (any rank-k projected install fails) or the SPAN'S SPECIFIC "
        "DIRECTIONS (only the corpus work's low-rank attractor bars "
        "writes)? e246: ALIGNED (rank-2, the span) expressed NOTHING while "
        "ORTHO (the complement) landed; e258: both v-heavy and v-light "
        "landed (span-balanced). THE DECISIVE ARM: a rank-matched RANDOM "
        "subspace install — if it fails like ALIGNED, the rank restriction "
        "is the barrier; if it lands, the span's specific structure is.",
    "bars_verbatim": {
            "RANK-IS-THE-BARRIER": "ARM-RANDOM fails to express (g0 ~ 0 or "
            "cannot land) like the span control — ANY rank-k restriction "
            "bars installs; the anti-substrate is dimensional: new memories "
            "need full-space (or at least much-higher-rank) room; the "
            "writing is bottlenecked by expressivity, not by the corpus's "
            "occupancy",
        "SPAN-IS-SPECIAL": "ARM-RANDOM lands at matched strength while the "
            "span control still fails — the barrier is the corpus work's "
            "specific directions (the low-rank attractor), not rank; the "
            "anti-substrate is occupancy: the ongoing work's own subspace "
            "bars writes (an interference object, e246's H-i confirmed "
            "causally)",
        "MIXED": "any between — the trajectories verbatim, no inflation",
    },
    "operationalizations": (
        "frozen BEFORE compute: k := THE SAME k e258 froze (k50 of the "
        "committed g1c root displacement, clipped [100k, 400k]), recomputed "
        "and HARD-BOUND == 237,123; both rooms rank k by construction and "
        "by the kept^2 probe; ARM-RANDOM := the SRCT ensemble (dense random "
        "orthonormal rank-k; seeds 26011/26012; DCT-II 'ortho' fp64); "
        "ARM-SPAN := the committed late span extended to the SAME k with an "
        "independent SRCT filler (seeds 26013/26014; the exact Gram-"
        "Woodbury orthogonal projector; containment certified); the hook is "
        "backward -> clip 1.0 VERBATIM -> project (CPU fp64, write fp32) -> "
        "opt.step, norm NOT rescaled; 'fails to express (g0 ~ 0)' := "
        "post-install g0 < 0.05; 'cannot land' := root g0 under the band "
        "floor; 'lands at matched strength' := root g0 within +-10% of the "
        "committed root g0 0.7447534203529358 -> [0.670278, 0.819229] AND "
        "install-carried (post g0 >= 0.05); RANK-IS-THE-BARRIER fires iff "
        "gates pass AND RANDOM fails AND the span control fails; "
        "SPAN-IS-SPECIAL fires iff gates pass AND RANDOM landed AND "
        "expresses AND the span control fails; MIXED otherwise; composite "
        "TEXTURE -> RANK-IS-THE-BARRIER -> SPAN-IS-SPECIAL -> MIXED; the "
        "span-control gate language is the REGISTERED CONTROL READ "
        "(reproduction reported + required for SPAN-IS-SPECIAL; "
        "non-reproduction is the MIXED branch, NOT a halt); the wash = the "
        "g-cell W1 form on landed arms (R=0.7 RAW at each own root + "
        "seed-10902 s300); retention = min g-12 over {10,50,100,300} / own "
        "root; the v-map LOADED from e258's committed artifact (the "
        "measured v-excess ledger); cons NATURAL on all arms; the measured "
        "kept/span/v loads reported everywhere — never nominal."),
    "registration": "bars + question frozen VERBATIM from the dispatch "
        "letter (T235's registered decisive discriminator); this script "
        "committed at birth BEFORE any compute; adjudicate against exactly "
        "this; no bar shopping.",
}

deviations: list[str] = [
    "THE SPAN CONTROL AT MATCHED k (the design's one interpretive act, "
    "disclosed at birth): the dispatch's 'the ALIGNED re-run at the SAME "
    "k' is operationalized as the committed late span (10 dense "
    "directions, md5-bound) EXTENDED to rank k by an independent dense "
    "random filler — a rank-k room that CONTAINS the span's specific "
    "directions. This is the only reading that keeps the cell decisive: "
    "both arms share rank k and differ ONLY in whether the corpus work's "
    "own directions are in the room (the alternative — a pure rank-10 "
    "span arm next to a rank-237k random arm — would confound rank with "
    "specificity, and e258 already showed rank-237k COORDINATE rooms "
    "land). The dispatch's gate language ('must reproduce e246's failure "
    "signature at matched k') is therefore handled as the registered "
    "control READ: non-reproduction is the honest MIXED branch, not a "
    "halt (a hard gate would adjudicate the finding at birth).",
    "THE RANDOM ENSEMBLE (disclosed): a Haar-uniform rank-k subspace at "
    "k=237,123 cannot even be materialized (N x k fp32 = 2.6 TB); the "
    "SRCT construction (a seeded random +-1 isometry composed with a "
    "random k-subset of the orthonormal DCT-II basis) is the standard "
    "feasible random dense subspace — every direction has FULL support "
    "over all 2.74M coordinates (like the span's own dense directions, "
    "unlike e258's axis-aligned coordinate rooms), the projector is exact "
    "(idempotency ~1e-16), and the kept^2 probe certifies the rank. The "
    "ensemble is isotropic in expectation, not literally Haar — "
    "disclosed, with the seeds registered.",
    "THE DENSE PROJECTION RUNS ON CPU fp64 (pocketfft, workers 2): the "
    "rooms' directions cross parameter boundaries (that is what 'dense' "
    "means here), so the hook gathers the flat post-clip gradient, "
    "projects it exactly in fp64, and writes back fp32 — the same "
    "two-pass discipline e246's smoke forced for its dense span "
    "projector (the block-diag bug cannot recur: there IS no per-slice "
    "approximation anywhere in this cell).",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map (runs/checkpoints/e258_vmap.pt) feeds the "
    "measured v-excess ledger; this cell runs NO new 20-step history "
    "(~65 s of GPU saved; the instrument is already certified to 1e-7 in "
    "the committed record). The installs' own fresh Adam state remains "
    "the mechanism's channel (e258's disclosed scope).",
    "e258's machinery PORTED WHOLE: the cell shape, the chunked "
    "install/cons/wash drivers, the G_FREE amended tiers (i-iv), the "
    "wash/readout/adjudication skeleton, the thermal envelope (matched "
    "max 80.0C, zero violations) — only the projector core and the "
    "adjudication clauses are new.",
    "Identical dose everywhere (s400 install / s300 cons / gen 24314 / "
    "cons seed 10901 / wash seed 10902): NO per-arm dose adjustment — "
    "an under-landing arm is the CANNOT-LAND finding for that arm by "
    "design; the kept-fraction ledger discloses the projection's dose "
    "honestly (the norm is NOT rescaled; e258's own texture: VLIGHT "
    "landed at kept ~0.10 — a ~0.29 kept here cannot be blamed for a "
    "failure by dose alone, and the record says so).",
    "n=1 per arm, one lineage, one wash draw (the g-series standing "
    "caveat — the critic's lottery note); the draw-spread band is the "
    "family's n=5-root committed record (e225), itself one lineage — "
    "disclosed, never inflated.",
    "e225/e229/e242 are NOT module-imported (each forces "
    "CUDA_VISIBLE_DEVICES=-1 at module level — this cell OWNS the GPU "
    "lane): cos64 (e225) and the _E228NetShim adapter (e229) + TempFamily "
    "+ logit_pass (e242's verbatim ports of e238) are COPIED VERBATIM "
    "with inline provenance notes. e228 IS module-imported for "
    "margin_pass (arithmetic untouched; opens runs/e228_run.log append — "
    "benign side effect, e242's disclosed precedent).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E260_SMOKE=1): 8-step install/cons, 4-step wash, "
    "k = 512 FIXED (no committed-record anchor), ckpts {1,2,4}, own smoke "
    "dir; NOTHING adjudicated or gated (SMOKE stamp on every read).",
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
    VIOLATION (e258 matched: max 80.0C, zero >= 84C)."""
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


# ------------------------------------------------------------------ fp64 math
def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    # PORTED VERBATIM from lab/e225_one_currency.py via e246/e258 (its own
    # provenance e191/e205/e209) — fp64 cosine of two flat fp32 CPU tensors.
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64) + 1e-30))


# ------------------------------------------------- the shim + thermal ports
class _E228NetShim:
    """PORTED VERBATIM from lab/e229_wall_currency.py via e246/e258
    (adapter only — NO arithmetic): TinyGPT-lineage net(idx) -> (logits,
    loss) becomes the net(input_ids=...) -> .logits object e228's
    margin_pass expects."""

    def __init__(self, net):
        self.net = net

    def eval(self):
        self.net.eval()
        return self

    def __call__(self, input_ids=None):
        lg, _ = self.net(input_ids)
        return SimpleNamespace(logits=lg)


GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201     # log10(T) in [-1,1] -> T in [0.1,10]
EPS_T = 1e-12


class TempFamily:
    """PORTED VERBATIM (via e242's port of lab/e238_temperature_null.py —
    the one-T family on the t=0 dumped logits; byte-identical arithmetic:
    same max-shifted softmax, grid, golden polish, NLL definition)."""

    def __init__(self, L0: np.ndarray, ans_ids: np.ndarray):
        self.L0 = L0.astype(np.float64)
        self.ans = ans_ids.astype(np.int64)
        self.lmax = self.L0.max(axis=1)
        self.d = self.L0 - self.lmax[:, None]
        self.da = self.d[np.arange(len(self.ans)), self.ans]
        self.n = len(self.ans)

    def q(self, T: float) -> np.ndarray:
        with np.errstate(over="ignore"):
            num = np.exp(self.da / T)
            den = np.exp(self.d / T).sum(axis=1)
        return num / np.maximum(den, EPS_T)

    def grid_Q(self, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
        grid = np.linspace(GRID_LO, GRID_HI, n)
        Q = np.stack([self.q(10.0 ** g) for g in grid])
        return grid, Q

    def _golden(self, f, lo: float, hi: float) -> float:
        gr = (5 ** 0.5 - 1) / 2
        a, b = lo, hi
        for _ in range(80):
            c, d_ = b - gr * (b - a), a + gr * (b - a)
            if f(c) < f(d_):
                b = d_
            else:
                a = c
            if b - a < 1e-8:
                break
        return (a + b) / 2

    def nll_bernoulli(self, T: float, p_obs: np.ndarray) -> float:
        qv = np.clip(self.q(T), EPS_T, 1.0 - EPS_T)
        return float(-(p_obs * np.log(qv)
                       + (1.0 - p_obs) * np.log(1.0 - qv)).sum())

    def fit_T_bernoulli(self, p_obs: np.ndarray,
                        grid=None, Q=None) -> tuple[float, float]:
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        Qc = np.clip(Q, EPS_T, 1.0 - EPS_T)
        nll_g = -(p_obs * np.log(Qc)
                  + (1.0 - p_obs) * np.log(1.0 - Qc)).sum(axis=1)
        k = int(np.argmin(nll_g))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xstar = self._golden(
            lambda g: self.nll_bernoulli(10.0 ** g, p_obs), lo, hi)
        return float(10.0 ** xstar), float(
            self.nll_bernoulli(10.0 ** xstar, p_obs))


@torch.no_grad()
def logit_pass(net, battery: list[dict]) -> dict:
    """PORTED VERBATIM from lab/e242_wall_commitment.py (its own
    provenance: e228's margin_pass forward path): one forward per probe —
    the FULL answer-position logit vector (fp32) for the one-T family; p
    recomputed on the same logits."""
    net.eval()
    rows, logits = [], []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        rows.append({"fact": pr["fact"], "p": float(p[pr["ans_id"]])})
        logits.append(lg.float().numpy().astype(np.float32))
    return {"rows": rows, "logits": np.stack(logits),
            "mean_p": float(np.mean([r["p"] for r in rows]))}


# ------------------------------------------------------ THE ROOMS (the core)
class SRCT:
    """A random rank-k DENSE orthonormal subspace of R^N — the SRCT
    ensemble: the k columns {D . phi_s : s in S} with D a seeded +-1
    diagonal and phi_j the orthonormal DCT-II basis vectors. The exact
    orthogonal projector is applied as
        P x = D . idct(mask_S(dct(D . x)))
    (fp64 pocketfft, CPU). The basis itself is NEVER materialized (an
    N x k fp32 basis would be ~2.6 TB at k=237k) — only the projector
    exists, and it is exact (certified in G_PROJ)."""

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


class DenseRooms:
    """THE CELL'S ONLY INTERVENTION: the two rank-k rooms + the read-only
    ledger. The flat basis IS net.parameters() order (the lineage's
    convention); the rooms' directions cross parameter boundaries (dense).

    step_hook(params, mode):
      FREE    — ledger dots READ ONLY (gn, v-excess pre, in-span pre);
                the gradient is NEVER touched (the verbatim trajectory);
      RANDOM  — g' = P_R g (the pure SRCT room), written fp32;
      SPAN    — g' = P_S g (the span-extended room via the exact Gram-
                Woodbury solve), written fp32.

    The ledger (fp64): gn = ||g||, gpn = ||g'||, kept_frac = ||g'||/||g||,
    v_excess_pre/post (e257's qov convention vs the LOADED v-map),
    in_span_frac (pre-projection, ||Vp g||/||g||), applied_in_span_frac
    (||Vp g'||/||g'|| — how much span the APPLIED gradient actually
    carries)."""

    def __init__(self, n: int, k: int, v_flat64: np.ndarray,
                 Vp64: np.ndarray, params_ref, dev: torch.device):
        self.n, self.k = int(n), int(k)
        self.dev = dev
        self.r_span = int(Vp64.shape[0])
        assert self.r_span < self.k
        self.v64 = v_flat64.astype(np.float64)
        self.mean_v = float(self.v64.mean())
        self.Vp = Vp64.astype(np.float64)                  # (r, N)
        self.rand = SRCT(n, k, SEED_D_RAND, SEED_S_RAND)   # the pure room
        kf = self.k - self.r_span
        self.fill = SRCT(n, kf, SEED_D_FILL, SEED_S_FILL)  # the span filler
        # the span-extended room B = [Vp | F2]: G = B^T B = I + E with E's
        # blocks {T = Vp Vp^T - I, G01 = Vp F2^T; G10 = G01^T} — E = M Nmat^T
        # with M, Nmat (k x 3r) — Woodbury in fp64 (30-dim at r=10).
        A01 = np.stack([self.fill.coeffs(self.Vp[j])[self.fill.S]
                        for j in range(self.r_span)])      # (r, kf)
        G00 = self.Vp @ self.Vp.T
        E_T = G00 - np.eye(self.r_span)
        r_, kf_ = self.r_span, kf
        I_r = np.eye(r_); Z_r = np.zeros((r_, r_)); Z_kf = np.zeros((kf_, r_))
        U1 = np.concatenate([I_r, Z_kf], 0); V1 = np.concatenate([E_T.T, Z_kf], 0)
        U2 = np.concatenate([I_r, Z_kf], 0); V2 = np.concatenate([Z_r, A01.T], 0)
        U3 = np.concatenate([Z_r, A01.T], 0); V3 = np.concatenate([I_r, Z_kf], 0)
        self.M_w = np.concatenate([U1, U2, U3], 1)         # (k, 3r)
        self.N_w = np.concatenate([V1, V2, V3], 1)         # (k, 3r)
        self.W_w = np.linalg.inv(np.eye(3 * r_) + self.N_w.T @ self.M_w)
        self.A01 = A01
        # the per-parameter offsets (the write-back slicing)
        self.offsets, self.shapes = [], []
        off = 0
        for p in params_ref:
            self.offsets.append((off, off + p.numel()))
            self.shapes.append(tuple(p.shape))
            off += p.numel()
        assert off == self.n, f"flat size {off} != {self.n}"

    # -- the two rooms' projectors (flat fp64 numpy) -----------------------
    def proj_random(self, g: np.ndarray) -> np.ndarray:
        return self.rand.project(g)

    def proj_span(self, g: np.ndarray) -> np.ndarray:
        c1 = self.Vp @ g                                   # (r,)
        cf = self.fill.coeffs(g)[self.fill.S]              # (kf,)
        c = np.concatenate([c1, cf])
        w = self.W_w @ (self.N_w.T @ c)
        z = c - self.M_w @ w
        scatter = np.zeros(self.n, dtype=np.float64)
        scatter[self.fill.S] = z[self.r_span:]
        return self.Vp.T @ z[:self.r_span] + self.fill.recon(scatter)

    def proj_of(self, mode: str, g: np.ndarray) -> np.ndarray:
        if mode == "RANDOM":
            return self.proj_random(g)
        if mode == "SPAN":
            return self.proj_span(g)
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
    def certify(self, n_probes: int = 4, seed: int = 26015) -> dict:
        rng = np.random.default_rng(seed)
        out: dict = {"probes": n_probes, "seed": seed}
        # (1) the DCT roundtrip identity
        x = rng.standard_normal(self.n)
        rt = self.rand.recon(self.rand.coeffs(x))
        out["dct_roundtrip_rel"] = float(np.linalg.norm(rt - x)
                                         / np.linalg.norm(x))
        # (2) idempotency + kept^2 per room + span containment + overlap
        for nm, fn in (("random", self.proj_random),
                       ("span", self.proj_span)):
            idem, kept2 = [], []
            for _ in range(n_probes):
                x = rng.standard_normal(self.n)
                px = fn(x)
                ppx = fn(px)
                idem.append(float(np.linalg.norm(ppx - px)
                                  / np.linalg.norm(px)))
                kept2.append(float((px @ px) / (x @ x)))
            out[f"{nm}_idempotency_max"] = max(idem)
            out[f"{nm}_kept2_mean"] = float(np.mean(kept2))
            out[f"{nm}_kept2_expect"] = self.k / self.n
        # (3) the span directions: containment (P_S v = v) + the RANDOM
        # room's measured overlap with each span direction
        cont = [float(np.linalg.norm(self.proj_span(self.Vp[j]) - self.Vp[j]))
                for j in range(self.r_span)]
        ovr = [float(np.linalg.norm(self.proj_random(self.Vp[j])))
               for j in range(self.r_span)]
        out["span_containment_max_dev"] = max(cont)
        out["random_room_span_overlap_norms"] = ovr
        out["random_room_span_overlap_mean"] = float(np.mean(ovr))
        out["random_room_span_overlap_expect"] = math.sqrt(self.k / self.n)
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
        if mode != "FREE":
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


def chunked_install(tag, mode, net0, proj: DenseRooms,
                    inst_x, inst_mask, anchor_full, train_ids, g0_ids,
                    gm12_ids, r_eval_xy, zid, resume_ck: Path,
                    dev: torch.device) -> dict:
    """g1c's chunked_install arithmetic VERBATIM (E43.exposure Dmix: ix(16)
    install w/ name mask + aj(16) paired + rj(32) random; masked token-level
    union CE; AdamW (0.9,0.95) wd 0.1; lr 1e-3 x house cosine(total=1000);
    clip 1.0) + THE HOOK: for RANDOM/SPAN each post-clip gradient is
    PROJECTED onto the arm's rank-k room (CPU fp64; write fp32) before
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
            # per-step thermal polls (the e233/e237/e246/e258 discipline)
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


def chunked_wash(tag, net0, anchor_neutral, train_ids, itos, r_eval_xy,
                 gm12_ids, g0_ids, zid, resume_ck: Path,
                 dev: torch.device, lr: float = G1.FT_LR,
                 seed: int = WASH_SEED) -> dict:
    """g1c's chunked_wash arithmetic VERBATIM (G1.g1_wash = e185/e176N:
    per step aj(16) neutral + rj(16) random; full-token CE; AdamW (0.9,0.95)
    wd 0.1 const lr; clip 1.0; the wall projects at every forward on the
    committed net; displacement bookkeeping; the CPU ARMED eval twin's
    light g-12/g0/CE_R at ckpt steps; per-step md5 x-hashes)."""
    ckpt_steps = CK_WASH
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    n_anc = anchor_neutral.shape[0]
    step = 0
    traj, sds, x_hashes, deltas = [], {}, {}, {}
    zeph_checks = 0
    net, opt, gen, evl = None, None, None, None
    theta0, prev = None, None
    wall_R = getattr(net0, "R", None)
    n_chunks, chunk_table = 0, []
    chunk_temps: list[float] = []
    t_burst, n_burst = None, 0
    if resume_ck.exists():
        _pre = torch.load(resume_ck, map_location="cpu", weights_only=False)
        if int(_pre.get("step", 0)) >= n_steps:
            log(f"  [{tag}] resume ckpt already COMPLETE at s{_pre['step']}")
            return {"sds": _pre["sds"], "traj": _pre.get("traj", []),
                    "steps_ran": n_steps, "seed": seed, "lr": lr,
                    "zeph_violations": _pre.get("zeph", 0),
                    "x_hashes": _pre.get("x_hashes", {}),
                    "deltas": _pre.get("deltas", {}),
                    "wall_R": _pre.get("wall_R", wall_R),
                    "n_chunks": _pre.get("n_chunks", 0),
                    "chunk_table": _pre.get("chunk_table", []),
                    "active_s": _pre.get("active_s", 0.0),
                    "resumed_final": True}
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
            gen = torch.Generator().manual_seed(seed)
            theta0 = flat_params_cpu(net)          # displacement origin (CPU)
            prev = theta0.clone()
            if resume_ck.exists():
                state = torch.load(resume_ck, map_location="cpu",
                                   weights_only=False)
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
                traj, sds = state.get("traj", []), state.get("sds", {})
                deltas = state.get("deltas", {})
                x_hashes = state.get("x_hashes", {})
                zeph_checks = state.get("zeph", 0)
                prev = flat_params_cpu(net)
                log(f"  [{tag}] RESUMED at step {step}/{n_steps} "
                    f"({len(traj)} traj rows)")
            evl = copy.deepcopy(net0).to(CPU)      # CPU eval twin (ARMED)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            aj = torch.randint(n_anc, (G1.ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                               generator=gen)
            anc = anchor_neutral[aj]
            rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
            for w in rnd:                    # name-free VERIFY (hard-fail)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt:
                    zeph_checks += 1
            x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
            y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
            x_hashes[step] = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
            xd, yd = x.to(dev), y.to(dev)
            logits, _ = net(xd)                    # <- the wall projects here
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   yd.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            cur = flat_params_cpu(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = float(torch.norm(cur - prev))
            prev = cur
            if step in ckpt_set:
                deltas[step] = cur - theta0
            row = {"step": step, "ce_batch": float(loss.item()),
                   "cum_disp": cum_disp, "step_disp": inc_disp,
                   "d_proj": min(cum_disp, wall_R) if wall_R else None,
                   "elapsed_s": round(time.time() - T0, 1)}
            if step in ckpt_set:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                sds[step] = sd_cpu
                evl.load_state_dict(sd_cpu)
                evl.eval()
                gz = G1.battery_cell(evl, gm12_ids, zid)
                gz0 = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                row.update({"g_m12_mean_pz": gz["mean_pz"],
                            "g0_mean_pz": gz0["mean_pz"],
                            "frac_argmax_z": gz["frac_argmax_z"],
                            "ce_r": ce_r})
                log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| "
                    f"{cum_disp:.4f} (CE {float(loss.item()):.4f})")
            traj.append(row)
            if step % 50 == 0 and step not in ckpt_set:
                log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                    f"|d| {cum_disp:.4f}")
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
                    "step": step, "traj": traj, "sds": sds,
                    "deltas": deltas, "x_hashes": x_hashes,
                    "zeph": zeph_checks, "wall_R": wall_R,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        burst_cooldown(tag)
        t_burst = None
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "zeph_violations": zeph_checks,
            "x_hashes": x_hashes, "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0)),
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
        "experiment": "e260_rank_matched",
        "phase": "THE RANK-MATCHED-RANDOM INSTALL — is the "
                 "install-unwritability the RANK RESTRICTION itself (any "
                 "rank-k projected install fails) or the SPAN'S SPECIFIC "
                 "DIRECTIONS (only the corpus work's low-rank attractor "
                 "bars writes)? A random rank-k dense room (RANDOM) vs the "
                 "span extended to the same k (SPAN) vs the natural install "
                 "(FREE), then the standard wall wash",
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
            "trainings": "3 installs s400 + 3 cons s300 + up to 3 wall "
                         "washes s300 (the v-map is LOADED from e258's "
                         "committed artifact — no new history)",
        },
        "deviations": deviations,
        "builds_on": [
            "T235 / e258 (THE registration: this cell's decisive arm + the "
            "v-map this cell LOADS for its measured-load ledger + the "
            "machinery PORTED WHOLE: the cell shape, the hook order, the "
            "amended G_FREE, the smoke discipline, the frozen k50 rank "
            "convention)",
            "T226 / e246 (the anti-substrate: the ALIGNED install cannot "
            "express — g0 2.9e-5 — while ORTHO lands; H-i interference vs "
            "H-ii rank-starvation are THIS cell's two suspects; the "
            "committed LATE span is the SPAN arm's core, md5-bound)",
            "T231 / e254 + T233 / e257 (the composition whose causal leg "
            "e258 closed NOT-THE-LOAD — the narrowing that promoted this "
            "cell from filler to decisive)",
            "T181 / g1c (the fresh-root lineage: e001+Dmix s400 gen 24314 + "
            "e113 cons 10901; the committed root + W1 records are the "
            "controls)",
            "e225 (the draw spread: the 2.74M wall family's n=5-root "
            "retention band), e237 (the pre-Adam hook + the not-rescaled "
            "convention), e233/e234/e236 (the span machinery's conventions "
            "behind e246's LATE span)",
        ],
        "whats_new": [
            "THE DECISIVE ARM: the first rank-matched RANDOM room — a "
            "dense random orthonormal rank-k subspace (the SRCT ensemble, "
            "seeds registered) at exactly e258's frozen k (237,123, "
            "hard-bound) — the anti-substrate's two suspects (rank vs the "
            "span's specific directions) separated at MATCHED rank",
            "the span-extended room: the committed late span CONTAINED in "
            "a rank-k random room (the exact Gram-Woodbury orthogonal "
            "projector; containment certified) — the ALIGNED re-run at "
            "matched k, the registered control read",
            "the measured overlap stack: the rooms' span overlap at the "
            "subspace level (containment + ||P_R v_j|| probes), the "
            "per-step gradient span fractions pre AND applied, and the "
            "displacement cos-to-span + in-own-room fractions — never "
            "nominal",
        ],
        "gates": {},
    })
    log(f"E260 — THE RANK-MATCHED-RANDOM INSTALL (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)} at identical dose (e001 + Dmix s"
        f"{INST_STEPS} gen {FRESH_GEN} + e113 cons s{CONS_STEPS} seed "
        f"{CONS_SEED}); wall R={R_CLAIM} RAW; wash seed {WASH_SEED}; "
        f"landing = root g0 within +-{MATCH_BAND:.0%} of the committed root "
        f"g0 {G1C_ROOT_G0:.4f}; expression floor g0 {G0_ZERO_FLOOR}; "
        f"seeds D/S {SEED_D_RAND}/{SEED_S_RAND} + {SEED_D_FILL}/"
        f"{SEED_S_FILL}")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(26001)                 # global init only; every RNG is its own

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

    # the neutral stream (e170's construction VERBATIM)
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
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])
    G_ANCHOR = {"neutral_bank": {"seed": G1.E170_ANCHOR_SEED,
                                 "n_windows": 16, "starts": n_starts},
                "pass": bool(anchor_neutral.shape == (16, G1.BLOCK))}
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
    g1c_w1 = {int(k): v for k, v in
              g1c["adjudication"]["wall"]["W1"]["g_m12"].items()}
    g1c_root_cells = g1c["root_build"]["root_cells"]
    g1c_root_gm12 = g1c_root_cells["gm12"]
    g1c_root_g0 = g1c_root_cells["g0"]
    g1c_post_inst = g1c["root_build"]["install"]["post_cells"]
    g1c_c1 = g1c["arms"]["C"]["traj"][0]
    e246 = json.loads(E246_METRICS.read_text(encoding="utf-8"))
    e246_aligned_post_g0 = e246["arms"]["ALIGNED"]["install"]["post_cells"]["g0"]
    e258 = json.loads(E258_METRICS.read_text(encoding="utf-8"))
    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root_gm12, "root_g0": g1c_root_g0,
                        "post_install_cells": g1c_post_inst,
                        "W1_g_m12": {str(k): v for k, v in
                                     sorted(g1c_w1.items())},
                        "C_step1": {"ce": g1c_c1["ce_batch"],
                                    "disp": g1c_c1["cum_disp"]}},
        "e246_metrics": {"path": str(E246_METRICS), "md5": md5of(E246_METRICS),
                         "verdict": e246["adjudication"]["verdict"],
                         "ALIGNED_post_g0": e246_aligned_post_g0,
                         "ORTHO_post_g0":
                             e246["arms"]["ORTHO"]["install"]["post_cells"]["g0"],
                         "FREE_post_g0":
                             e246["arms"]["FREE"]["install"]["post_cells"]["g0"]},
        "e258_metrics": {"path": str(E258_METRICS), "md5": md5of(E258_METRICS),
                         "verdict": e258["adjudication"]["verdict"],
                         "k": e258["vmap"]["k_rule"]["k"],
                         "reads": {kk: e258["adjudication"]["reads"][kk]
                                   for kk in ("post_install_g0", "root_g0",
                                              "landed", "expresses")}},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK)},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "e225_metrics": {"path": str(E225_METRICS), "md5": md5of(E225_METRICS)},
        "hardbound": {"root_gm12": G1C_ROOT_GM12, "root_g0": G1C_ROOT_G0,
                      "W1_g_m12": G1C_W1_GM12, "C_step1": G1C_C_STEP1,
                      "e246_verdict": E246_VERDICT,
                      "e246_ALIGNED_post_g0": E246_ALIGNED_POST_G0,
                      "e246_ORTHO_post_g0": E246_ORTHO_POST_G0,
                      "e246_FREE_post_g0": E246_FREE_POST_G0,
                      "e246_span_md5": E246_SPAN_MD5,
                      "e246_span_rank": E246_SPAN_RANK,
                      "e258_verdict": E258_VERDICT,
                      "e258_k": E258_K_HARD},
        "pass": bool(
            g1c["adjudication"]["verdict"] == G1C_VERDICT
            and abs(g1c_root_gm12 - G1C_ROOT_GM12) < 1e-12
            and abs(g1c_root_g0 - G1C_ROOT_G0) < 1e-12
            and all(abs(g1c_w1[s] - G1C_W1_GM12[s]) < 1e-12
                    for s in G1C_W1_GM12)
            and abs(g1c_c1["ce_batch"] - G1C_C_STEP1["ce"]) < 1e-12
            and abs(g1c_c1["cum_disp"] - G1C_C_STEP1["disp"]) < 1e-12
            and e246["adjudication"]["verdict"] == E246_VERDICT
            and abs(e246_aligned_post_g0 - E246_ALIGNED_POST_G0) < 1e-12
            and e258["adjudication"]["verdict"] == E258_VERDICT
            and int(e258["vmap"]["k_rule"]["k"]) == E258_K_HARD
            and md5of(CKPT_DIR / SPAN_CK) == E246_SPAN_MD5),
    }
    e225 = json.loads(E225_METRICS.read_text(encoding="utf-8"))
    fam = {r["id"]: r["retention_flat_min"] for r in e225["joined_table"]
           if r["scale"] == "2.74M"}
    ids_map = {"J1": "g1b", "J2": "g1c", "J3": "g1d", "J4": "g1e", "J5": "g1f"}
    G_PARENTS["draw_spread"] = {
        "source": "runs/e225/metrics.json joined_table (2.74M rows)",
        "members": {ids_map[k]: v for k, v in fam.items()},
        "band": [min(fam.values()), max(fam.values())],
        "pass": bool(all(abs(fam[k] - DRAW_SPREAD["members"][ids_map[k]])
                         < 1e-9 for k in fam)
                     and abs(min(fam.values()) - DRAW_SPREAD["lo"]) < 1e-9
                     and abs(max(fam.values()) - DRAW_SPREAD["hi"]) < 1e-9)}
    G_PARENTS["pass"] = bool(G_PARENTS["pass"] and
                             G_PARENTS["draw_spread"]["pass"])
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G1C_VERDICT} (root g0 {g1c_root_g0:.6f}); "
        f"e246 {E246_VERDICT} (ALIGNED post g0 {e246_aligned_post_g0:.2e}); "
        f"e258 {E258_VERDICT} (k {E258_K_HARD}); draw spread "
        f"[{DRAW_SPREAD['lo']:.4f}, {DRAW_SPREAD['hi']:.4f}] (n=5 roots)")
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

    # ================= P1: THE ROOMS (k + v-map + span + certification) ==
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

    # ---- THE RANK k (the SAME rule e258 froze; recomputed + HARD-BOUND) --
    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))   # the params basis
    root_disp = theta_root - base_flat
    e2 = root_disp.double() ** 2
    order_desc = torch.argsort(e2, descending=True)
    cums = torch.cumsum(e2[order_desc], 0) / float(e2.sum())
    k_raw = int((cums < K_QUANT).sum().item()) + 1
    k = K_SMOKE if SMOKE else int(min(max(k_raw, K_MIN), K_MAX))
    k_rule = {"rule": "k := THE SAME k e258 froze — clip(k50 of the "
                      "committed g1c ROOT displacement's squared-L2 support, "
                      "100_000, 400_000), recomputed from committed "
                      "checkpoints and HARD-BOUND == e258's committed "
                      "237,123; smoke: fixed 512",
              "k_raw": k_raw, "k": k, "k_fraction_of_N": k / N,
              "root_disp_l2": float(torch.norm(root_disp)),
              "hardbound_e258_k": E258_K_HARD,
              "matches_e258": bool(k == E258_K_HARD),
              "pass": bool(k == E258_K_HARD or SMOKE)}
    assert k_rule["pass"], f"K-MATCH FAILED: recomputed {k} != e258's {E258_K_HARD}"
    log(f"P1 K-RULE: k {k} ({k / N * 100:.2f}% of N; raw {k_raw}) — "
        f"{'MATCHES' if k == E258_K_HARD else 'SMOKE-fixed vs'} e258's "
        f"committed {E258_K_HARD}")

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
                     and (SMOKE or int(vmap_art["meta"]["k"]) == k)),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    log(f"P1 G_VMBIND: {VMAP_CK} (md5 {G_VMBIND['md5'][:8]}..., "
        f"{G_VMBIND['size']} coords, meta k {G_VMBIND['meta_k']}): PASS")

    # ---- THE SPAN LOADED (e246's committed LATE span; the SPAN room's core)
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

    # ---- THE ROOMS BUILT + CERTIFIED -------------------------------------
    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = DenseRooms(N, k, v64_np, Vp.numpy().astype(np.float64),
                       params_ref, dev)
    cert = rooms.certify(n_probes=4, seed=26015)
    # the kept^2 bar is adaptive: kept2 concentrates as sqrt(2/k) — at the
    # real k (237,123) the fixed 0.02 bar is ~6 sigma; smoke's toy k=512
    # needs the statistical widening (disclosed; catches size bugs either
    # way)
    kept2_bar = max(0.02, 8.0 * math.sqrt(2.0 / k))
    G_PROJ = {
        "form": "the SRCT rooms' certification (fp64 CPU, 4 probes): the "
                "DCT roundtrip identity; each room's IDEMPOTOTENCY "
                "(||P^2 x - P x||/||P x||) and kept^2 rank probe "
                "(||P x||^2/||x||^2 vs k/N); the span CONTAINMENT "
                "(||P_S v_j - v_j||, the SPAN room must contain the "
                "committed span); the RANDOM room's measured overlap with "
                "each span direction (||P_R v_j||, expect ~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2_abs_dev": kept2_bar, "containment": 1e-5},
        "pass": bool(cert["dct_roundtrip_rel"] <= 1e-8
                     and cert["random_idempotency_max"] <= 1e-8
                     and cert["span_idempotency_max"] <= 1e-8
                     and abs(cert["random_kept2_mean"] - k / N) <= kept2_bar
                     and abs(cert["span_kept2_mean"] - k / N) <= kept2_bar
                     and cert["span_containment_max_dev"] <= 1e-5),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    log(f"P1 G_PROJ: roundtrip {cert['dct_roundtrip_rel']:.1e}; idem "
        f"R {cert['random_idempotency_max']:.1e} / S "
        f"{cert['span_idempotency_max']:.1e}; kept2 R "
        f"{cert['random_kept2_mean']:.4f} / S {cert['span_kept2_mean']:.4f} "
        f"(expect {k / N:.4f}); containment "
        f"{cert['span_containment_max_dev']:.1e}; RANDOM room's span "
        f"overlap {cert['random_room_span_overlap_mean']:.3f} "
        f"(expect ~{cert['random_room_span_overlap_expect']:.3f}): PASS")

    # save the rooms artifact (reconstructible from the registered seeds)
    rooms_ck = save_ckpt(
        "e260_rooms",
        {"D_rand_int8": rooms.rand.D.astype(np.int8),
         "S_rand": rooms.rand.S, "D_fill_int8": rooms.fill.D.astype(np.int8),
         "S_fill": rooms.fill.S},
        {"desc": "e260's two rank-k rooms (the flat basis is "
                 "net.parameters() order): ARM-RANDOM's SRCT room (seeds "
                 f"{SEED_D_RAND}/{SEED_S_RAND}) + ARM-SPAN's filler (seeds "
                 f"{SEED_D_FILL}/{SEED_S_FILL}; the span itself is "
                 "e246_late_span.pt, md5-bound)",
         "k": k, "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert.items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "k_rule": k_rule,
        "seeds": {"D_rand": SEED_D_RAND, "S_rand": SEED_S_RAND,
                  "D_fill": SEED_D_FILL, "S_fill": SEED_S_FILL,
                  "cert_probes": 26015},
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       "LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE ROOMS: RANDOM (pure SRCT, rank {k}) + SPAN (rank "
        f"{rooms.r_span} span + {k - rooms.r_span} SRCT filler = rank "
        f"{rooms.r_span + (k - rooms.r_span)}): BUILT + CERTIFIED")
    write_partial("P1 THE ROOMS built (k hard-bound + v-map loaded + span "
                  "loaded + certification)")

    # ================= P2/P3/P4: THE THREE ARMS ==========================
    arms_rec: dict = {}
    theta0s: dict = {}
    for arm in ARMS:
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        inst = chunked_install(
            f"{arm}-inst", arm, G1.evl_load(base_sd), rooms,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e260_{arm}_inst_resume.pt" if SMOKE
                        else f"e260_{arm}_inst_resume.pt"), dev)
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
            CKPT_DIR / (f"smoke_e260_{arm}_cons_resume.pt" if SMOKE
                        else f"e260_{arm}_cons_resume.pt"), dev)
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
            f"e260_{arm}_root", theta0,
            {"desc": f"e260 ARM-{arm} root: e001 + Dmix s{INST_STEPS} "
                     f"(gen {FRESH_GEN}, {ARM_DESC[arm]}) + e113 cons "
                     f"s{CONS_STEPS} (seed {CONS_SEED} HELD)",
             "arm": arm, "install_seed": FRESH_GEN, "mode": arm,
             "base": f"runs/checkpoints/{BASE_CK}",
             "rooms": "runs/checkpoints/e260_rooms.pt"})
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
        if arm != ARMS[-1] and not inst.get("resumed_final", False):
            burst_cooldown(f"{arm} -> next arm")

    # ---- the FREE ledger cross-check vs e258's committed record (texture)
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

    # ---- G_FREE: THE CONTROL (e246's AMENDED tiers, ported verbatim) ----
    free_root_g0 = arms_rec["FREE"]["root"]["g0"]
    locked_root = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                             weights_only=False)
    locked_root_sd = locked_root["model"] if isinstance(locked_root, dict) \
        and "model" in locked_root else locked_root
    free_sd = theta0s["FREE"]
    mdf = max(float((free_sd[k].float() - locked_root_sd[k].float())
                    .abs().max()) for k in locked_root_sd if k in free_sd)
    l2f = float(np.sqrt(sum(float(((free_sd[k].float()
                                    - locked_root_sd[k].float()) ** 2).sum())
                            for k in locked_root_sd if k in free_sd)))
    del locked_root, locked_root_sd
    # tier (i): the install-final reproduction vs g1c's install resume ckpt
    comm_inst = torch.load(CKPT_DIR / "g1c_install_resume.pt",
                           map_location="cpu", weights_only=False)["model"]
    my_inst = torch.load(
        CKPT_DIR / ("smoke_e260_FREE_inst_resume.pt" if SMOKE
                    else "e260_FREE_inst_resume.pt"),
        map_location="cpu", weights_only=False)["model"]
    inst_l2 = float(np.sqrt(sum(float(((my_inst[k].float()
                                        - comm_inst[k].float()) ** 2).sum())
                                for k in comm_inst if k in my_inst)))
    inst_md = max(float((my_inst[k].float() - comm_inst[k].float())
                        .abs().max()) for k in comm_inst if k in my_inst)
    del comm_inst, my_inst
    free_post = arms_rec["FREE"]["install"]["post_cells"]
    # tier (iii): the cons-tracking texture (median step |g0 diff|)
    g1c_cons_traj = g1c["root_build"]["consolidation"]["traj"]
    my_cons_traj = arms_rec["FREE"]["consolidation"]["traj"]
    cons_diffs = [abs(a["g0_pz"] - b["g0_pz"]) for a, b in
                  zip(my_cons_traj, g1c_cons_traj)
                  if a["step"] == b["step"]]
    cons_track_median = float(sorted(cons_diffs)[len(cons_diffs) // 2]) \
        if cons_diffs else None
    G_FREE = {
        "form": ("e246's AMENDED G_FREE tiers, ported VERBATIM via e258 "
                 "(the 2026-10-04 halt's autopsy is inherited law): (i) the "
                 "instrumented-path install reproduces the committed "
                 "install final (L2 <= 5e-3 AND behavioral g-12 |d| <= "
                 "5e-3); (ii) FREE's root lands in the matched band; "
                 "(iii) the cons tracks the committed cons (median step "
                 "|g0 diff| <= 0.05, texture); (iv) [after P5] FREE's W1 "
                 "retention within the draw spread AND maintains >= 0.50 "
                 "at every ckpt."),
        "root_distance_texture": {"root_gm12": arms_rec["FREE"]["root"]["gm12"],
                                  "committed_root_gm12": G1C_ROOT_GM12,
                                  "max_abs_diff_vs_ckpt": mdf,
                                  "l2_vs_ckpt": l2f},
        "install_reproduction": {
            "l2_vs_committed_install_final": inst_l2,
            "max_abs_diff_vs_committed_install_final": inst_md,
            "behavioral_gm12": free_post["gm12"],
            "behavioral_gm12_committed": G1C_POST_INSTALL["gm12"],
            "behavioral_abs_diff": abs(free_post["gm12"]
                                       - G1C_POST_INSTALL["gm12"]),
            "bar": 5e-3,
            "pass": bool(inst_l2 < 5e-3
                         and abs(free_post["gm12"]
                                 - G1C_POST_INSTALL["gm12"]) < G_READ_TOL)},
        "root_band": {"g0": free_root_g0,
                      "band": [G1C_ROOT_G0 * (1 - MATCH_BAND),
                               G1C_ROOT_G0 * (1 + MATCH_BAND)],
                      "pass": bool(arms_rec["FREE"]["root"]["landed"])},
        "cons_tracking": {"median_step_g0_abs_diff": cons_track_median,
                          "bar": 0.05,
                          "pass": bool(cons_track_median is not None
                                       and cons_track_median <= 0.05)},
        "w1_wash": None,       # tier (iv) — filled + folded after P5
        "pass": bool(inst_l2 < 5e-3
                     and abs(free_post["gm12"]
                             - G1C_POST_INSTALL["gm12"]) < G_READ_TOL
                     and arms_rec["FREE"]["root"]["landed"]
                     and cons_track_median is not None
                     and cons_track_median <= 0.05),
    }
    metrics["gates"]["G_FREE"] = G_FREE
    log(f"G_FREE (tiers i-iii): install L2 {inst_l2:.3e} (bar 5e-3), "
        f"behavioral |d| {abs(free_post['gm12'] - G1C_POST_INSTALL['gm12']):.1e}; "
        f"root g0 {free_root_g0:.4f} in band "
        f"{arms_rec['FREE']['root']['landed']}; cons-tracking median "
        f"{cons_track_median}: "
        f"{'PASS (i-iii)' if G_FREE['pass'] else 'FAIL — THE CELL HALTS'}")
    write_partial("P4 G_FREE read (tiers i-iii)"
                  + ("" if G_FREE["pass"] else " — FAILED"))
    if not G_FREE["pass"] and not SMOKE:
        metrics["status"] = ("HALTED — G_FREE FAILED (the amended tiers; "
                             "nothing adjudicated)")
        write_partial("HALTED (G_FREE)")
        return 1

    # ================= P5: THE WALL WASHES (landed arms only) ===========
    landed_arms = [a for a in ARMS if arms_rec[a]["root"]["landed"]]
    skipped = [a for a in ARMS if a not in landed_arms]
    if skipped:
        log(f"P5: arms NOT landed -> wash SKIPPED (the registration's "
            f"letter; CANNOT-LAND is the finding for {skipped})")
    washes: dict = {}
    G_BITROOT, G_INPUTS = {}, {"note": "per-step input batches bit-identical "
                                       "across the washed arms (md5)"}
    for arm in landed_arms:
        log("=" * 78)
        log(f"ARM-{arm} WALL WASH — commit(R={R_CLAIM} RAW) + the "
            f"seed-{WASH_SEED} {WASH_STEPS}-step neutral wash")
        net0 = G1.CommittedGPT(GB.G1B_CFG)
        net0.load_state_dict(theta0s[arm])
        net0.commit(R_CLAIM)
        body, _ = G1.split_anchored_sd(net0.state_dict())
        md = max(float((body[k].float() - theta0s[arm][k].float()).abs().max())
                 for k in theta0s[arm])
        anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                      for n, p in net0.named_parameters())
        G_BITROOT[arm] = {"max_abs_diff": md,
                          "anchors_bit_equal": bool(anch_ok),
                          "pass": bool(md == 0.0 and anch_ok)}
        assert G_BITROOT[arm]["pass"], f"{arm}: wall root != theta0"
        arm_w = chunked_wash(
            f"{arm}-wash", net0, anchor_neutral, train_ids, itos, r_eval_xy,
            gm12_ids, g0_ids, zid,
            CKPT_DIR / (f"smoke_e260_{arm}_wash_resume.pt" if SMOKE
                        else f"e260_{arm}_wash_resume.pt"), dev)
        assert arm_w["zeph_violations"] == 0, \
            f"{arm}: name token leaked into a wash window"
        washes[arm] = arm_w
        # measured-load texture: the +300 wash displacement
        if WASH_STEPS in arm_w.get("deltas", {}):
            d300 = arm_w["deltas"][WASH_STEPS]
            washes[arm]["load_d300"] = rooms.displacement_loads(d300, arm)
        metrics["arms"][arm]["wash"] = {
            "R": R_CLAIM, "seed": WASH_SEED, "steps_ran": arm_w["steps_ran"],
            "ckpt_steps": list(CK_WASH), "traj": arm_w["traj"],
            "chunk_table": arm_w["chunk_table"],
            "load_d300": washes[arm].get("load_d300"),
            "theta0_norm": arm_w.get("theta0_norm"),
            "wall_R": arm_w.get("wall_R")}
        write_partial(f"P5 ARM-{arm} wall wash done")
        if arm != landed_arms[-1] and not arm_w.get("resumed_final", False):
            burst_cooldown(f"{arm} wash -> next")
    # cross-arm input identity (over the washed arms)
    if len(washes) >= 1:
        common_steps = set(washes[landed_arms[0]]["x_hashes"])
        for arm in landed_arms[1:]:
            common_steps &= set(washes[arm]["x_hashes"])
        G_INPUTS["steps_compared"] = len(common_steps)
        G_INPUTS["identical"] = bool(common_steps and all(
            washes[a]["x_hashes"][s] == washes[landed_arms[0]]["x_hashes"][s]
            for a in landed_arms for s in common_steps))
        G_INPUTS["pass"] = G_INPUTS["identical"]
        assert G_INPUTS["pass"], "cross-arm wash input streams diverged"
    metrics["gates"]["G_INPUTS"] = G_INPUTS
    if G_BITROOT:
        G_BITROOT["pass"] = bool(all(v["pass"] for v in G_BITROOT.values()
                                     if isinstance(v, dict) and "pass" in v))
    metrics["gates"]["G_BITROOT"] = G_BITROOT
    log(f"G_INPUTS: per-step wash inputs bit-identical across the washed "
        f"arms (md5, {G_INPUTS.get('steps_compared', 0)} steps): PASS")

    # FREE's wash vs the committed g1c W1 record — tier (iv)
    if "FREE" in washes:
        free_gm12 = {t["step"]: t["g_m12_mean_pz"]
                     for t in washes["FREE"]["traj"]
                     if "g_m12_mean_pz" in t}
        free_w1_rows = {s: {"measured": free_gm12.get(s),
                            "committed": G1C_W1_GM12.get(s),
                            "abs_diff": abs(free_gm12.get(s, 0)
                                            - G1C_W1_GM12.get(s, 0))}
                        for s in sorted(set(free_gm12) & set(G1C_W1_GM12))}
        free_flat = [free_gm12[s] for s in FLAT_FULL8 if s in free_gm12]
        free_root_g12 = arms_rec["FREE"]["root"]["gm12"]
        free_ret = (min(free_flat) / free_root_g12
                    if free_flat and free_root_g12 > 0 else None)
        G_FREE["w1_wash"] = {
            "rows_vs_committed_w1": free_w1_rows,
            "max_abs_diff_vs_committed": max(r["abs_diff"]
                                             for r in free_w1_rows.values()),
            "retention_full8": free_ret,
            "root_gm12": free_root_g12,
            "draw_spread": [DRAW_SPREAD["lo"], DRAW_SPREAD["hi"]],
            "min_gm12_all_ckpts": min(free_gm12.values()),
            "maintain_bar": float(G1.MAINTAIN_BAR),
            "pass": bool(free_ret is not None
                         and DRAW_SPREAD["lo"] <= free_ret <= DRAW_SPREAD["hi"]
                         and min(free_gm12.values()) >= G1.MAINTAIN_BAR),
            "note": "tier (iv) of the amended gate: the per-step rows vs "
                    "the committed W1 are CO-REPORTED texture (cross-pass "
                    "chaos); the fate tiers gate",
        }
        G_FREE["pass"] = bool(G_FREE["pass"] and G_FREE["w1_wash"]["pass"])
        metrics["gates"]["G_FREE"] = G_FREE
        _fr = "n/a" if free_ret is None else f"{free_ret:.4f}"
        log(f"G_FREE tier (iv): FREE W1 retention {_fr} in spread "
            f"[{DRAW_SPREAD['lo']:.3f}, {DRAW_SPREAD['hi']:.3f}], min g-12 "
            f"{min(free_gm12.values()):.4f} >= {G1.MAINTAIN_BAR}: "
            f"{'PASS' if G_FREE['w1_wash']['pass'] else 'FAIL'}")
        if not G_FREE["pass"] and not SMOKE:
            metrics["status"] = ("HALTED — G_FREE tier (iv) FAILED (the "
                                 "amended gate; nothing adjudicated)")
            write_partial("HALTED (G_FREE tier iv)")
            return 1
    write_partial("P5 washes complete (landed arms)")

    # ================= P6: THE READOUTS (ruler + margins + thermal) ======
    pbatt = [{"ids": gm12_ids[i: i + 1], "fact": f"install{i:02d}@g-12",
              "relation": "install60_g-12", "ans_id": zid}
             for i in range(gm12_ids.shape[0])]
    readouts: dict = {}
    for arm in landed_arms:
        cells_state: dict = {}
        net0 = G1.evl_load(theta0s[arm])
        shim = _E228NetShim(net0)
        mrec = E228.margin_pass(shim, pbatt)
        lrec = logit_pass(shim, pbatt)
        dp = max(abs(r["p"] - m["p"])
                 for r, m in zip(lrec["rows"], mrec["probes"]))
        ms0 = [r["margin_sigma"] for r in mrec["probes"]]
        cells_state["t0"] = {
            "step": 0, "gm12": arms_rec[arm]["root"]["gm12"],
            "g0": arms_rec[arm]["root"]["g0"],
            "margin_median_sigma": mrec["median_margin_sigma"],
            "margin_mean_sigma": mrec["mean_margin_sigma"],
            "margin_p25_sigma": float(np.percentile(ms0, 25)),
            "frac_argmax_z": float(sum(1 for r in mrec["probes"]
                                       if r["top1_id"] == zid) / len(ms0)),
            "logits": lrec["logits"],
            "p_obs": np.array([r["p"] for r in lrec["rows"]]),
            "logit_pass_max_dp": dp,
        }
        log(f"  [{arm} t0] ruler {cells_state['t0']['gm12']:.4f} | median "
            f"margin {mrec['median_margin_sigma']:.4f}s argmax-Z "
            f"{cells_state['t0']['frac_argmax_z']:.2f} | dp {dp:.1e}")
        del net0, shim
        for s in sorted(washes[arm]["sds"]):
            net = G1.evl_load(washes[arm]["sds"][s])
            shim = _E228NetShim(net)
            mrec = E228.margin_pass(shim, pbatt)
            lrec = logit_pass(shim, pbatt)
            ms = [r["margin_sigma"] for r in mrec["probes"]]
            ruler = next((t["g_m12_mean_pz"] for t in washes[arm]["traj"]
                          if t["step"] == s and "g_m12_mean_pz" in t), None)
            cells_state[str(s)] = {
                "step": s, "gm12": ruler,
                "margin_median_sigma": mrec["median_margin_sigma"],
                "margin_mean_sigma": mrec["mean_margin_sigma"],
                "margin_p25_sigma": float(np.percentile(ms, 25)),
                "frac_argmax_z": float(sum(1 for r in mrec["probes"]
                                           if r["top1_id"] == zid) / len(ms)),
                "logits": lrec["logits"],
                "p_obs": np.array([r["p"] for r in lrec["rows"]]),
                "role": "ADJUDICATES" if s in READ_GRID else "texture",
            }
            del net, shim
            log(f"  [{arm} +{s}] ({cells_state[str(s)]['role']}) ruler "
                f"{ruler:.4f} | median margin "
                f"{mrec['median_margin_sigma']:.4f}s | argmax-Z "
                f"{cells_state[str(s)]['frac_argmax_z']:.2f}")
            write_partial(f"P6 ARM-{arm} +{s} readout")
        # the thermal family on this arm's OWN t0 logits (e238/e242's port)
        fam_arm = TempFamily(cells_state["t0"]["logits"],
                             np.full(len(pbatt), zid, dtype=np.int64))
        grid, Q = fam_arm.grid_Q()
        p0_arm = cells_state["t0"]["p_obs"]
        th_rows = []
        for key in ["t0"] + [str(s) for s in sorted(washes[arm]["sds"])]:
            c = cells_state[key]
            p_obs = c["p_obs"]
            T_mle, nll = fam_arm.fit_T_bernoulli(p_obs, grid, Q)
            q = fam_arm.q(T_mle)
            ss_res = float(((p_obs - q) ** 2).sum())
            ss_tot = float(((p_obs - p0_arm) ** 2).sum())
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            c["T_mle"] = T_mle
            c["R2_pooled"] = R2
            c["ss_tot"] = ss_tot
            c.pop("logits", None)
            c.pop("p_obs", None)
            th_rows.append({"state": key, "step": c["step"],
                            "T_mle": T_mle, "R2_pooled": R2})
        readouts[arm] = {"states": cells_state, "thermal": th_rows}
        log(f"  [{arm} thermal] " + " ".join(
            f"{r['state']}:T{r['T_mle']:.3f}" for r in th_rows))
    metrics["readouts"] = readouts

    # wash-health co-report (the wash must still be a wash)
    def arm_health(arm):
        ces = [t["ce_batch"] for t in washes[arm]["traj"]]
        n10 = min(10, len(ces) // 2)
        return {"ce_first": sum(ces[:n10]) / n10,
                "ce_last": sum(ces[-n10:]) / n10,
                "ce_improves": bool(sum(ces[-n10:]) / n10
                                    < sum(ces[:n10]) / n10)}
    metrics["wash_health"] = {a: arm_health(a) for a in landed_arms}
    write_partial("P6 readouts COMPLETE (ruler + margins + thermal)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    def gm12_series(arm):
        out = {0: arms_rec[arm]["root"]["gm12"]}
        for t in washes[arm]["traj"]:
            if "g_m12_mean_pz" in t:
                out[t["step"]] = t["g_m12_mean_pz"]
        return out

    retention = {}
    for arm in landed_arms:
        g = gm12_series(arm)
        r_root = g[0]
        flat_primary = [g[s] for s in FLAT_260 if s in g]
        flat_full8 = [g[s] for s in FLAT_FULL8 if s in g]
        retention[arm] = {
            "root_gm12": r_root,
            "primary_flat_min": min(flat_primary) if flat_primary else None,
            "retention_primary": (min(flat_primary) / r_root
                                  if flat_primary else None),
            "retention_full8": (min(flat_full8) / r_root
                                if flat_full8 else None),
            "retention_plus300": (g.get(WASH_STEPS, float("nan")) / r_root
                                  if WASH_STEPS in g else None),
            "g_m12": {str(k): v for k, v in sorted(g.items())},
            "landed": arms_rec[arm]["root"]["landed"],
        }
    metrics["retention"] = retention

    landed = {a: bool(arms_rec[a]["root"]["landed"]) for a in ARMS}
    expresses = {a: bool(arms_rec[a]["root"]["expresses"]) for a in ARMS}
    band_lo_g0 = G1C_ROOT_G0 * (1 - MATCH_BAND)
    band_hi_g0 = G1C_ROOT_G0 * (1 + MATCH_BAND)
    rnd_root_g0 = arms_rec["RANDOM"]["root"]["g0"]
    sp_root_g0 = arms_rec["SPAN"]["root"]["g0"]
    rnd_post_g0 = arms_rec["RANDOM"]["install"]["post_cells"]["g0"]
    sp_post_g0 = arms_rec["SPAN"]["install"]["post_cells"]["g0"]

    def arm_fails(arm) -> bool:
        """the registered failure: g0 ~ 0 at install OR cannot land."""
        return bool(arms_rec[arm]["install"]["post_cells"]["g0"]
                    < G0_ZERO_FLOOR
                    or arms_rec[arm]["root"]["g0"] < band_lo_g0)

    random_fails = arm_fails("RANDOM")
    span_fails = arm_fails("SPAN")
    rank_fires = bool(random_fails and span_fails)
    span_special_fires = bool(landed["RANDOM"] and expresses["RANDOM"]
                              and span_fails)
    # the dispatch's gate language: the SPAN control's reproduction read
    span_control_reproduces = bool(sp_post_g0 < G0_ZERO_FLOOR)

    # the hard gate set
    hard = {k: v for k, v in metrics["gates"].items()}
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    over_land = [a for a in ARMS
                 if arms_rec[a]["root"]["g0"] > band_hi_g0]

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif rank_fires:
        verdict = "RANK-IS-THE-BARRIER"
        clause = (f"ARM-RANDOM fails to express (post-install g0 "
                  f"{rnd_post_g0:.4f} {'<' + format(G0_ZERO_FLOOR, '.2f') if rnd_post_g0 < G0_ZERO_FLOOR else '>=' + format(G0_ZERO_FLOOR, '.2f')}"
                  f"; root g0 {rnd_root_g0:.4f} vs floor {band_lo_g0:.4f})"
                  f" like the span control (post g0 {sp_post_g0:.4f}; root "
                  f"g0 {sp_root_g0:.4f}) — ANY rank-k restriction bars "
                  f"installs; the anti-substrate is dimensional: new "
                  f"memories need full-space (or at least much-higher-rank) "
                  f"room; the writing is bottlenecked by expressivity, not "
                  f"by the corpus's occupancy")
    elif span_special_fires:
        verdict = "SPAN-IS-SPECIAL"
        clause = (f"ARM-RANDOM lands at matched strength (root g0 "
                  f"{rnd_root_g0:.4f} in [{band_lo_g0:.4f}, {band_hi_g0:.4f}], "
                  f"post-install g0 {rnd_post_g0:.4f} install-carried) while "
                  f"the span control still fails (post g0 {sp_post_g0:.4f}"
                  f"{'' if sp_post_g0 < G0_ZERO_FLOOR else ' >= floor'}; "
                  f"root g0 {sp_root_g0:.4f}) — the barrier is the corpus "
                  f"work's specific directions (the low-rank attractor), "
                  f"not rank; the anti-substrate is occupancy: the ongoing "
                  f"work's own subspace bars writes (an interference object, "
                  f"e246's H-i confirmed causally)")
    else:
        why = []
        if not landed["RANDOM"] and not random_fails:
            why.append(f"RANDOM above the band ceiling (root g0 "
                       f"{rnd_root_g0:.4f} > {band_hi_g0:.4f})")
        if landed["RANDOM"] and expresses["RANDOM"] and not span_fails:
            why.append(f"BOTH rooms land (RANDOM root g0 {rnd_root_g0:.4f} / "
                       f"SPAN {sp_root_g0:.4f}; the span control does NOT "
                       f"reproduce e246's failure at matched k — post g0 "
                       f"{sp_post_g0:.4f}): the failure does not survive "
                       f"rank-padding at k={k}")
        if landed["RANDOM"] and not expresses["RANDOM"] and not span_fails:
            why.append(f"RANDOM landed cons-rescued (post g0 {rnd_post_g0:.4f}"
                       f" < {G0_ZERO_FLOOR})")
        if span_fails and landed["RANDOM"] and not expresses["RANDOM"]:
            why.append(f"RANDOM landed cons-rescued (post g0 "
                       f"{rnd_post_g0:.4f}) while the span control failed")
        if not why:
            why.append("between the bars")
        verdict = "MIXED"
        clause = "; ".join(why) + " — the trajectories verbatim, no inflation"

    log("=" * 78)
    log(f"E260 VERDICT: {verdict}")
    for a in ARMS:
        row = (f"  {a}: post g0 "
               f"{arms_rec[a]['install']['post_cells']['g0']:.4f} -> root g0 "
               f"{arms_rec[a]['root']['g0']:.4f} (landed {landed[a]}, "
               f"expresses {expresses[a]})")
        if a in retention:
            g = gm12_series(a)
            row += (" | wash g-12 " + " -> ".join(
                f"+{s}:{v:.4f}" for s, v in sorted(g.items()))
                + f" | retention {retention[a]['retention_primary']:.4f}")
        else:
            row += " | wash SKIPPED (not landed)"
        log(row)
    log(f"  span-control reproduction read: e246 ALIGNED post g0 "
        f"{E246_ALIGNED_POST_G0:.2e} vs SPAN-at-k post g0 {sp_post_g0:.4f} "
        f"-> {'REPRODUCES the failure signature' if span_control_reproduces else 'does NOT reproduce (the MIXED branch, disclosed)'}")
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> RANK-IS-THE-BARRIER -> SPAN-IS-SPECIAL "
                           "-> MIXED (frozen)",
        "gates_pass": gates_pass,
        "reads": {
            "post_install_g0": {a: arms_rec[a]["install"]["post_cells"]["g0"]
                                for a in ARMS},
            "root_g0": {a: arms_rec[a]["root"]["g0"] for a in ARMS},
            "root_gm12": {a: arms_rec[a]["root"]["gm12"] for a in ARMS},
            "retention_primary": {a: (retention[a]["retention_primary"]
                                      if a in retention else None)
                                  for a in ARMS},
            "retention_full8": {a: (retention[a]["retention_full8"]
                                    if a in retention else None)
                                for a in ARMS},
            "retention_plus300": {a: (retention[a]["retention_plus300"]
                                      if a in retention else None)
                                  for a in ARMS},
            "landed": landed, "expresses": expresses,
            "over_landing_arms": over_land,
            "matched_band_g0": [band_lo_g0, band_hi_g0],
            "draw_spread": [DRAW_SPREAD["lo"], DRAW_SPREAD["hi"]],
            "g0_zero_floor": G0_ZERO_FLOOR,
            "k": k, "k_matches_e258": bool(k == E258_K_HARD),
            "random_fails": random_fails, "span_fails": span_fails,
            "span_control_reproduces_e246": span_control_reproduces,
            "washes_run": list(washes.keys()),
            "washes_skipped": skipped,
        },
        "RANK_IS_THE_BARRIER": bool(gates_pass and rank_fires),
        "SPAN_IS_SPECIAL": bool(gates_pass and span_special_fires),
        "MIXED": bool(gates_pass and not rank_fires
                      and not span_special_fires),
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars adjudicated")

    # ================= P8: the figures ====================================
    make_fates_plot(RD, arms_rec, retention, washes, readouts, verdict,
                    clause, landed, expresses, landed_arms)
    make_rooms_plot(RD, rooms, arms_rec, cert, k, thermal_log, verdict)

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("the arms share bit-identical install "
            "streams (one generator, seed 24314, one draw order), identical "
            "dose/schedule/optimizer; the ONLY delta is the pre-Adam "
            "gradient PROJECTION onto the arm's rank-k room — RANDOM's "
            "room carries NO corpus structure (a seeded dense random "
            "subspace), SPAN's room differs from it ONLY by CONTAINING "
            "the committed late span's 10 directions (containment "
            "certified to 1e-5; both rooms' rank certified by the kept^2 "
            "probe at the SAME hard-bound k); G_FREE validates the "
            "instrumented path against the committed g1c root AND its "
            "committed W1 wash; the washes share bit-identical input "
            "streams across the washed arms (G_INPUTS, md5) with each "
            "landed arm's wall committed at its own theta0 (G_BITROOT) — "
            "fate differences are room-caused or nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage (the g1c fresh-root "
            "convention), one wash draw (seed 10902) — the critic's "
            "lottery note carried verbatim; the draw spread is the "
            "family's committed n=5-root band (e225), itself one lineage; "
            "the retention reads are single-draw statistics against that "
            "band, quoted with both flavors (primary flat-{10,50,100,300} "
            "+ the full-8 co-report)"),
        "loads_measured_not_nominal": ("every arm's ACTUAL geometry is "
            "reported: the rooms' span overlap at the subspace level "
            "(containment + the ||P_R v_j|| probes vs the sqrt(k/N) "
            "expectation), the per-step kept fraction, v-excess pre/post "
            "(vs e258's LOADED committed v-map), the in-span fraction pre "
            "AND applied, and the displacement cos-to-span + in-own-room "
            "fractions at post-install, root, and wash+300 — never "
            "nominal"),
        "the_confound_disclosed": ("a rank-k restriction at k=237,123 is "
            "8.66% of N: e258's AXIS-ALIGNED coordinate rooms at the same "
            "k already landed (VHEAVY/VLIGHT), so this cell's RANDOM arm "
            "is a DENSE random room (SRCT) — the faithful test of 'any "
            "rank-k restriction' for span-like dense directions; if "
            "RANDOM lands while a coordinate room also landed, 'ANY "
            "rank-k fails' is false twice over; the SPAN room's filler is "
            "dense random too, so a landing SPAN arm does not by itself "
            "acquit the span's directions of anything (the failure needs "
            "the pure/low-rank span) — the bars' MIXED branch is the "
            "honest outcome for that world and the record says so"),
        "dose_honesty": ("the projection's norm is NOT rescaled (e237's "
            "convention): the kept-fraction ledger discloses the applied "
            "dose (~sqrt(k/N) ~ 0.29 expected for both rooms — and "
            "e258's VLIGHT landed at kept ~0.10, so a kept ~0.29 failure "
            "cannot be blamed on dose alone)"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
            "outcome was promised; the bars cover all three branches and "
            "the trajectories are reported verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_core": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "rooms": metrics["rooms"]["checkpoint"],
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS},
            "arm_wash_resumes": {
                a: f"runs/checkpoints/"
                  f"{'smoke_' if SMOKE else ''}e260_{a}_wash_resume.pt"
                for a in ARMS},
        },
        "machinery": {
            "install_cons_wash": "g1c's chunked drivers VERBATIM "
                                 "arithmetic (E43.exposure Dmix / e113 "
                                 "consolidate / G1.g1_wash) via e246/e258's "
                                 "port, with the dense-room hook + this "
                                 "cell's thermal envelope",
            "hook": "e237's pre-Adam projection as a DENSE ROOM projector "
                    "(backward -> clip 1.0 -> project (CPU fp64 SRCT / "
                    "Gram-Woodbury; write fp32) -> step; norm not "
                    "rescaled; fp64 ledger dots)",
            "rooms": "ARM-RANDOM: the SRCT ensemble (a seeded random +-1 "
                     "isometry x a random k-subset of the orthonormal "
                     "DCT-II basis; exact projector, never materialized); "
                     "ARM-SPAN: e246's committed LATE span extended to k "
                     "by an independent SRCT filler (the exact Gram-"
                     "Woodbury orthogonal projector; containment "
                     "certified); seeds registered (26011-26014)",
            "vmap": "e258's committed 2.74M v-map LOADED from "
                    "e258_vmap.pt (md5 recorded; k-bound) — the measured "
                    "v-excess ledger only; no new history run",
            "margins": "e228's margin_pass MODULE-IMPORTED via e229's "
                       "_E228NetShim adapter (ported copy; adapter only)",
            "thermal": "e238's TempFamily via e242's port (copied "
                       "verbatim; grid 1201 + golden polish + Bernoulli "
                       "NLL/R2)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads(),
                 "margin_batch_shape": "1 x L per probe (e228's shape)"},
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
                          str(RD / "e260_rank_matched.png"),
                          str(RD / "e260_subspace_instrument.png")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_fates_plot(rd, arms_rec, retention, washes, readouts, verdict,
                    clause, landed, expresses, landed_arms):
    """THE FATE FIGURE: the expression trajectories (post-install g0 per
    step), the landing bars vs the band, the wash g-12 trajectories + the
    retentions vs the draw spread, and the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    col = {"RANDOM": "#1a6faf", "SPAN": "#c0392b", "FREE": "#e67e22"}

    # (0,0) THE EXPRESSION TRAJECTORIES (install-phase g0)
    ax = axes[0, 0]
    for arm in ARMS:
        traj = arms_rec[arm]["install"]["traj"]
        ax.plot([t["step"] for t in traj], [t["g0_pz"] for t in traj],
                "o-", ms=5, lw=2.0, color=col[arm], alpha=0.95,
                label=f"ARM-{arm} (post g0 "
                      f"{arms_rec[arm]['install']['post_cells']['g0']:.3f})")
    ax.axhline(G0_ZERO_FLOOR, ls=":", lw=1.4, color="crimson",
               label=f"the g0~0 floor ({G0_ZERO_FLOOR})")
    ax.axhline(E246_ALIGNED_POST_G0, ls="-.", lw=1.2, color="darkred",
               alpha=0.8,
               label=f"e246 ALIGNED post g0 {E246_ALIGNED_POST_G0:.1e} "
                     f"(the failure signature)")
    ax.axhline(G1C_POST_INSTALL["g0"], ls="--", lw=1.1, color="seagreen",
               label=f"committed g1c post-install g0 "
                     f"{G1C_POST_INSTALL['g0']:.3f}")
    ax.set_xlabel("install step (Dmix s400, identical dose)")
    ax.set_ylabel("g0 battery (install-60 mean p(Z))")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("THE EXPRESSION TRAJECTORIES (g0 through the install)",
                 fontsize=10)

    # (0,1) THE LANDING (root g0 vs the matched band)
    ax = axes[0, 1]
    names = list(ARMS)
    vals = [arms_rec[a]["root"]["g0"] for a in names]
    bars_ = ax.bar(names, vals, color=[col[a] for a in names], alpha=0.85)
    for b, v, a in zip(bars_, vals, names):
        ax.annotate(f"{v:.4f}" + ("" if landed[a] else "\n(NOT landed)")
                    + ("" if expresses[a] else "\n(g0~0 at install)"),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhspan(G1C_ROOT_G0 * (1 - MATCH_BAND), G1C_ROOT_G0 * (1 + MATCH_BAND),
               color="#b8d8f0", alpha=0.35, zorder=0,
               label=f"the matched band (+-{MATCH_BAND:.0%} of the "
                     f"committed root g0 {G1C_ROOT_G0:.3f})")
    ax.axhline(G1C_ROOT_G0, color="k", ls=":", lw=0.9)
    ax.set_ylabel("ROOT g0 (post install + cons)")
    ax.legend(fontsize=7.4, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE LANDING READOUT (matched strength, root g0)",
                 fontsize=10)

    # (1,0) THE WASH TRAJECTORIES (g-12 through the wall wash)
    ax = axes[1, 0]
    for arm in landed_arms:
        g = {0: arms_rec[arm]["root"]["gm12"]}
        for t in washes[arm]["traj"]:
            if "g_m12_mean_pz" in t:
                g[t["step"]] = t["g_m12_mean_pz"]
        xs, ys = sorted(g), [g[s] for s in sorted(g)]
        ax.plot(xs, ys, "o-", ms=6, lw=2.2, color=col[arm], alpha=0.95,
                label=f"ARM-{arm} (root {g[0]:.3f}; ret "
                      f"{retention[arm]['retention_primary']:.3f})")
    xs = sorted(int(k) for k in G1C_W1_GM12)
    ax.plot(xs, [G1C_W1_GM12[s] for s in xs], "s--", ms=5, lw=1.4,
            color="seagreen", alpha=0.9,
            label="committed g1c W1 (the FREE control's record)")
    for yv, c, lab in ((G1.MAINTAIN_BAR, "seagreen", "maintain 0.50"),
                       (G1.SHUT_BAR, "tab:purple", "kill 0.27")):
        ax.axhline(yv, ls="--", lw=1.1, color=c, alpha=0.8)
        ax.annotate(lab, (0.99, yv), xycoords=("axes fraction", "data"),
                    ha="right", fontsize=7, color=c, va="bottom")
    ax.set_xlabel("wall-wash step (commit R=0.7 then the 10902 wash)")
    ax.set_ylabel("g-12 (install-60 ruler, mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("THE WASH (g-12 through the wall wash; landed arms only)",
                 fontsize=10)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E260 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=94, break_long_words=False)[:10]:
        ax.text(0.02, y, wd, fontsize=6.8, va="top", family="monospace")
        y -= 0.024
    y -= 0.012
    gates_txt = "  ".join(
        f"{g}={'PASS' if v.get('pass') else 'FAIL'}"
        for g, v in metrics["gates"].items() if isinstance(v, dict))
    for wd in textwrap.wrap("GATES: " + gates_txt, width=96)[:2]:
        ax.text(0.02, y, wd, fontsize=6.4, va="top", family="monospace")
        y -= 0.02
    fig.suptitle("E260 — THE RANK-MATCHED-RANDOM INSTALL: a random dense "
                 f"rank-k room vs the span-extended room at the SAME k "
                 f"({metrics['rooms']['k_rule']['k']}, e258's frozen "
                 f"k50) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e260_rank_matched.png", dpi=130)
    plt.close(fig)


def make_rooms_plot(rd, rooms: DenseRooms, arms_rec, cert, k, thermal_log,
                    verdict):
    """THE INSTRUMENT FIGURE: the rooms' certification + the measured
    per-arm loads + the span overlap stack + the thermal envelope."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    col = {"RANDOM": "#1a6faf", "SPAN": "#c0392b", "FREE": "#e67e22"}

    # (0,0) THE ROOMS (certification bars, log scale)
    ax = axes[0, 0]
    labels = ["DCT roundtrip", "idem RANDOM", "idem SPAN",
              "span containment"]
    vals = [cert["dct_roundtrip_rel"], cert["random_idempotency_max"],
            cert["span_idempotency_max"], cert["span_containment_max_dev"]]
    bars_ = ax.bar(labels, [max(v, 1e-18) for v in vals],
                   color=["#8e44ad", "#1a6faf", "#c0392b", "#27ae60"],
                   alpha=0.85)
    for b, v in zip(bars_, vals):
        ax.annotate(f"{v:.1e}", (b.get_x() + b.get_width() / 2,
                                 max(v, 1e-18)),
                    ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-2)
    ax.axhline(1e-8, ls=":", color="k", lw=1.0, label="the bar 1e-8")
    ax.axhline(1e-5, ls="--", color="dimgray", lw=1.0,
               label="containment bar 1e-5")
    ax.set_ylabel("max relative deviation (log)")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25, axis="y", which="both")
    kn = k / rooms.n
    ax.set_title(f"THE ROOMS' CERTIFICATION (rank probe: kept2 R "
                 f"{cert['random_kept2_mean']:.4f} / S "
                 f"{cert['span_kept2_mean']:.4f} vs k/N {kn:.4f})",
                 fontsize=9.5)

    # (0,1) THE MEASURED LOADS per arm
    ax = axes[0, 1]
    names = list(ARMS)
    w = 0.2
    xs = np.arange(len(names))
    pre = [arms_rec[a]["install"]["ledger_v_excess_pre_median"]
           for a in names]
    post = [arms_rec[a]["install"]["ledger_v_excess_post_median"]
            for a in names]
    droot = [arms_rec[a]["root"]["displacement_loads"]["v_excess"]
             for a in names]
    aspan = [arms_rec[a]["install"]["ledger_applied_in_span_frac_median"]
             or 0.0 for a in names]
    ax.bar(xs - 1.5 * w, [p if p is not None else 0 for p in pre], width=w,
           color="#7fb3d5", alpha=0.9, label="install |g| v-excess (ledger med)")
    ax.bar(xs - 0.5 * w, [p if p is not None else 0 for p in post], width=w,
           color="#1a6faf", alpha=0.9, label="install |g'| v-excess (applied)")
    ax.bar(xs + 0.5 * w, droot, width=w, color="#c0392b", alpha=0.65,
           label="root displacement v-excess")
    ax2 = ax.twinx()
    ax2.bar(xs + 1.5 * w, aspan, width=w, color="#27ae60", alpha=0.8)
    ax2.set_ylabel("applied gradient in-span fraction (green)",
                   color="#27ae60", fontsize=8)
    ax2.tick_params(axis="y", colors="#27ae60")
    ax2.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("v-excess (measured; e257's qov convention)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE MEASURED PER-ARM LOADS (never nominal; the green "
                 "bars: the APPLIED gradient's span fraction)", fontsize=9.5)

    # (1,0) THE SPAN-OVERLAP STACK
    ax = axes[1, 0]
    names = list(ARMS)
    w = 0.22
    xs = np.arange(len(names))
    led_span = [arms_rec[a]["install"]["ledger_in_span_frac_median"]
                for a in names]
    led_aspan = [arms_rec[a]["install"]["ledger_applied_in_span_frac_median"]
                 for a in names]
    dcos = [arms_rec[a]["root"]["displacement_loads"]["cos_to_span"]
            for a in names]
    droom = [arms_rec[a]["root"]["displacement_loads"]["in_own_room"]
             or 0.0 for a in names]
    ax.bar(xs - 1.5 * w, led_span, width=w, color="#7fb3d5", alpha=0.9,
           label="natural |g| in-span (pre)")
    ax.bar(xs - 0.5 * w, led_aspan, width=w, color="#27ae60", alpha=0.9,
           label="applied |g'| in-span")
    ax.bar(xs + 0.5 * w, dcos, width=w, color="#c0392b", alpha=0.65,
           label="root displacement cos-to-span")
    ax.bar(xs + 1.5 * w, droom, width=w, color="#8e44ad", alpha=0.7,
           label="root displacement in-own-room")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("THE SPAN-OVERLAP STACK (the measured per-arm overlap "
                 "with the span — the dispatch's check)", fontsize=9.5)

    # (1,1) the kept-fraction ledger + the thermal envelope
    ax = axes[1, 1]
    for arm in ARMS:
        led = arms_rec[arm]["install"]["ledger"]
        xs_l = sorted(int(s) for s in led)
        get = lambda s: led[s] if s in led else led[str(s)]
        ax.plot(xs_l, [get(s)["kept_frac"] for s in xs_l], "o-",
                ms=3.2, lw=1.3, color=col[arm], alpha=0.9,
                label=f"ARM-{arm} ||g'||/||g||")
    ax.axhline(math.sqrt(k / rooms.n), ls=":", color="k", lw=1.0,
               label=f"the sqrt(k/N) expectation {math.sqrt(k / rooms.n):.3f}")
    ax.set_xlabel("install step")
    ax.set_ylabel("the kept-gradient fraction (dose honesty)")
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25)
    axb = ax.twinx()
    if thermal_log:
        axb.plot([r["t"] for r in thermal_log], [r["temp"] for r in thermal_log],
                 "-", lw=0.8, color="dimgray", alpha=0.7)
        axb.axhline(TEMP_EARLY_END, color="crimson", ls=":", lw=1.0)
        axb.axhline(TEMP_HARD, color="crimson", ls="--", lw=1.2)
        axb.set_ylabel("GPU temp (C) per-step polls", color="dimgray")
        axb.tick_params(axis="y", colors="dimgray")
    ax.set_title("THE KEPT-FRACTION LEDGER + the thermal envelope (bursts "
                 "end at the 78C margin)", fontsize=9.5)

    fig.suptitle("E260 — the instrument page: the rank-matched rooms, the "
                 "certification, the measured loads, the span-overlap stack",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e260_subspace_instrument.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
