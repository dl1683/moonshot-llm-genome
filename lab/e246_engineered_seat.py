"""E246 — THE ENGINEERED SEAT (the build lane's first cell; the ambition
directive 2026-10-04's answer; W020's alternative-architecture clause made
concrete). Design frozen in scratch/e246_design.md — THE registration;
this docstring carries its bars VERBATIM. Committed at birth, BEFORE any
compute.

THE AMBITION (the design's letter): every seat result so far has been
OBSERVED (e226 found the dying probe aligned; e233/e237 cut the alignment
and moved the fate). THE GENERATIVE MOVE: BUILD the seat. Install a fact
in a CHOSEN geometry relative to the wash's active span, then wash — and
ask whether fate follows the ENGINEERED geometry. If it does, the lab's
deepest object (the third dimension's seat) becomes a CONSTRUCTION TOOL:
forgetting by design, retention by design.

THE CELL (per the frozen design; 2.74M; the g1c fresh-root lineage's
conventions throughout):
  1. THE SPAN: the FRESH ROOT's wash-span — the seed-10902 20-step
     unwalled history's Gram-SVD basis at the committed g1c root (e225's
     chunked-fp64 svd_basis VERBATIM, ported), G_SHADOW applied: the
     LATE half (segments 11..20) is used so the sign-step's shadow is
     out (e236's |cos(seg_k, -u0)| < 0.5 validity gate, recorded per
     segment).
  2. THREE INSTALL ARMS at identical dose/steps/protocol (the g1c
     fresh-install arithmetic VERBATIM: e001 base + e043-Dmix s400/
     total=1000, gen 24314, then e113 cons s300 seed 10901 HELD); ONLY
     the gradient geometry at the INSTALL differs (e237's pre-Adam hook,
     ported to the install phase: backward -> clip 1.0 VERBATIM ->
     PROJECT -> opt.step()):
       ARM-ORTHO   each install gradient's span component REMOVED
                   (g' = g - S(S^T g)) before Adam;
       ARM-ALIGNED each install gradient restricted TO the span
                   (g' = S(S^T g)) before Adam;
       ARM-FREE    the natural install — the verbatim path through the
                   instrumented code (the fp64 ledger dots READ ONLY,
                   no write) — THE CONTROL: must reproduce the committed
                   g1c root or the cell HALTS.
  3. THE WASH: the standard WALL wash (the g-cell W1 convention
     VERBATIM) on all three arms: commit(R=0.7 RAW L2 — the 2.74M
     convention) at each arm's OWN root, then the seed-10902 300-step
     neutral wash, per-step inputs bit-identical across arms.
  4. READOUTS: the fact ruler (install-60 g-12) at t=0 — the installs
     must LAND at matched root strength (the pre-registered +-10% band
     around the committed g1c root read; CANNOT-LAND is a FINDING, not
     a failure); the ruler + the margins (e242's instrument: e228's
     margin_pass via the shim) + the thermal reads (e238's one-T MLE,
     per-arm family base L0 = the arm's OWN t0 logits — e242's port)
     through {+1,+2,+10,+50,+100,+300} ({+4,+200} ride as texture); the
     retentions (the flat-phase convention).

REGISTERED BARS (frozen in scratch/e246_design.md, VERBATIM in this
docstring BEFORE any compute; adjudicate against exactly this; no bar
shopping):
  - SEAT-IS-ENGINEERABLE — "ARM-ORTHO retains >= 1.5x ARM-ALIGNED's
    flat-phase retention (both landed at matched t=0 strength) — fate
    follows the engineered geometry; the seat is a construction tool"
    (the design's fuller letter adds: "the third dimension's practical
    name: INSTALL GEOMETRY")
  - GEOMETRY-IRRELEVANT — "the three arms' retentions sit within the
    draw spread — the seat does not transfer to installs at 2.74M; the
    honest bound" (the design: "the worlds stay separate")
  - CANNOT-LAND — "the orthogonal install fails to reach the matched
    t=0 strength — the span is the install's substrate; a finding of a
    different kind"
  - ANY — "the trajectories verbatim"

REGISTERED PREDICTIONS (the design's, verbatim; no retrofit):
  (a) "Under SEAT-IS-ENGINEERABLE: ARM-ORTHO's margins THICKEN faster
      than ARM-ALIGNED's (the forge works harder when the undertow
      cannot reach — e242's thickening + the exposure law composed)."
  (b) "The t=0 thermal read: ARM-ALIGNED's +1 heat exceeds ARM-ORTHO's
      (the transient is the gust, and the gust needs the span)."
  (c) "The u0-cloud caveat carried: installs are stream-relative
      clouds; the arms' geometries verified per-arm by measured overlap
      (report actual cos-to-span per arm, not the nominal design)."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE WASH FORM: "the standard wall wash (the g-cell convention)" :=
    the g-cell's W1 form VERBATIM — commit(R=0.7 RAW L2) at the arm's
    own root, then the seed-10902 300-step neutral wash (g1b/g1c's own
    convention). Frozen on the bars' own vocabulary: "flat-phase
    retention" and predictions (a)/(b) (e242's forge thickening + the
    +1 heat) are WALL-family objects that only exist under the wall.
  * retention := min g-12 over the FLAT subset of the registered
    readout grid {10,50,100,300} divided by the arm's OWN root g-12
    (e225's flat-min convention on e242's registered grid). Co-reported:
    the full-8 flavor (min over {10,50,100,200,300}, e225's own set)
    and the +300 point flavor. The bar's ratio uses the primary.
  * landed (matched t=0 strength) := the arm's root g-12 within the
    +-10% band around the committed g1c root g-12 (0.9026340246200562):
    [0.812371, 0.992897]. CANNOT-LAND fires iff ARM-ORTHO's root g-12
    lands UNDER the band's floor ("fails to reach"); an OVER-landing
    arm (above the ceiling) is DISCLOSED, matched fails for it, and the
    verdict falls through to the honest residual — never dose-trimmed.
  * the draw spread := the 2.74M wall family's committed retention band
    [g1e 0.615392, g1d 1.326643] (e225's joined_table, hard-bound at
    runtime; n=5 roots of one lineage — g1b/g1c/g1d/g1e/g1f).
    GEOMETRY-IRRELEVANT := SEAT did not fire AND all three arms'
    primary retentions sit inside that band.
  * COMPOSITE ORDER (frozen): hard-gate failure => TEXTURE (nothing
    adjudicated; the record completes; G_FREE failure HALTS the cell as
    the design demands) -> CANNOT-LAND (ORTHO under the band floor) ->
    SEAT-IS-ENGINEERABLE (both ORTHO and ALIGNED landed AND
    retention(ORTHO) >= 1.5 x retention(ALIGNED)) -> GEOMETRY-IRRELEVANT
    (SEAT did not fire AND all three retentions within the draw spread)
    -> ANY (the trajectories verbatim).
  * prediction (a)'s read: per-arm margin thickening = the battery
    MEDIAN margin_sigma trajectory relative to the arm's own t0; "ORTHO
    thickens faster" := 100*(med(+300)/med(t0) - 1) is larger for ORTHO
    than ALIGNED. (b)'s read: T_mle(+1) per arm (each arm's own L0
    family); fires iff T_ALIGNED(+1) > T_ORTHO(+1). (c)'s read: the
    measured cos-to-span per arm = ||S^T d||/||d|| (the amplitude
    fraction of the displacement inside the span; fp64) at post-install
    AND at the final root, plus the gradient-level ledger medians
    (||S^T g||/||g|| per step).
  * THE SPAN is fixed for the whole cell (root-fixed at the committed
    g1c root — e233's fixed-direction convention, disclosed); the
    install arms train from the e001 BASE with that fixed basis.
  * THE HOOK ordering (e237's, ported): backward -> clip_grad_norm_ 1.0
    VERBATIM -> PROJECT -> opt.step(); the projection is the LAST
    operation before Adam (both moments + the update supply-cut); norm
    NOT rescaled; all dots/norms fp64 (chunked per-parameter).

HONESTY GUARDS (the design's, verbatim in force):
  * The install dose ladder may need per-arm adjustment to land matched
    strength — the tolerance is pre-registered above (+-10%); any
    per-arm dose change would be a DISCLOSED deviation. In the primary
    registration NO arm's dose is adjusted: identical steps/lr/seeds
    everywhere; an under-landing ORTHO is the CANNOT-LAND finding.
  * One lineage, one wash draw per arm (the g-series standing caveat);
    the retentions are n=1 per arm against a family band of n=5 roots.
  * GPU bursts with per-step thermal polls at a 78C margin from the
    FIRST burst (the e233/e237 discipline; bursts <= 175 s < the 180 s
    lab cap; cooldowns 45 s in the owner's 30-60 s window; the 84C
    never-past line recorded).
  * THE BUILD LANE'S FIRST CELL: any instrument surprise gets the full
    autopsy before the verdict is read.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
g-series' own reference scale — CONTINUITY on the g1c line). 7 GPU
trainings (3 installs s400 + 3 cons s300 + 3 wall washes s300) + the
20-step span history, each in <=175 s bursts, per-step thermal polls;
CPU probing between bursts (threads 4 — e240/e247 own the CPU lane).

Outputs: runs/e246/{metrics.json (PROGRESSIVE), e246_engineered_seat.png,
e246_geometry_thermal.png, run.log}; checkpoints
runs/checkpoints/e246_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e246_engineered_seat.py    (E246_SMOKE=1 shakedown)
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

torch.set_num_threads(4)           # shared machine (e240/e247 own the CPU lane)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E246_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e246_smoke" if SMOKE else "e246"
assert torch.cuda.is_available(), "e246 owns the GPU lane (dispatch)"

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
FLAT_246: tuple[int, ...] = tuple(s for s in (10, 50, 100, 300)
                                  if s in READ_GRID)      # the primary flat set
FLAT_FULL8: tuple[int, ...] = tuple(s for s in (10, 50, 100, 200, 300)
                                    if s in CK_WASH)      # e225's set (co-report)

HIST_STEPS = 20 if not SMOKE else 4
LATE_START = 11 if not SMOKE else 3          # the LATE half's first segment
SHADOW_BAR = 0.5                             # e236's |cos| validity gate

G_READ_TOL = 5e-3                 # the family's cross-device read tolerance

# the committed g1c record, HARD-BOUND (read at runtime from
# runs/g1c_root/metrics.json and asserted against these literals; Rule 12)
G1C_METRICS = E43.REPO / "runs" / "g1c_root" / "metrics.json"
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562
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

# the draw spread (e225's committed 2.74M wall-family retention band)
E225_METRICS = E43.REPO / "runs" / "e225" / "metrics.json"
DRAW_SPREAD = {"lo": 0.6153917590189493, "hi": 1.3266428034732003,
               "members": {"g1b": 0.9979692766675983,
                           "g1c": 1.0263911969648605,
                           "g1d": 1.3266428034732003,
                           "g1e": 0.6153917590189493,
                           "g1f": 0.9317064573129114}}

ARMS = ("ORTHO", "ALIGNED", "FREE")     # FREE runs FIRST (the halt gate)
ARM_DESC = {
    "ORTHO": "each install gradient's span component REMOVED before Adam "
             "(g' = g - S(S^T g); e237's pre-Adam hook ported to the "
             "install phase)",
    "ALIGNED": "each install gradient restricted TO the span before Adam "
               "(g' = S(S^T g))",
    "FREE": "the natural install through the instrumented path (fp64 ledger "
            "dots READ ONLY, no write) — must reproduce the committed g1c "
            "root or the cell HALTS",
}

# ---- THE THERMAL ENVELOPE (the e233/e237 discipline, the owner's
# max-priority window): per-step polls from the FIRST burst at the 78C
# margin; bursts <= 175 s; cooldowns 45 s (the 30-60 s window); the 84C
# never-past line recorded (e237's own observed max: 82C, zero >= 84C).
BURST_MAX_S = 175.0
COOLDOWN_S = 45.0
TEMP_EARLY_END = 78.0              # per-step-poll burst-end margin
TEMP_HARD = 84.0                   # the recorded never-past line
POLL_EVERY = 1                     # PER-STEP mid-burst temp polls
LAUNCH_POLL_GAP_S = 5.0
CHUNK_DOT = 4_000_000              # per-parameter fp64 chunk (e237's cut)

REGISTERED = {
    "bars_verbatim": {
        "SEAT-IS-ENGINEERABLE": "ARM-ORTHO retains >= 1.5x ARM-ALIGNED's "
            "flat-phase retention (both landed at matched t=0 strength) — "
            "fate follows the engineered geometry; the seat is a "
            "construction tool",
        "GEOMETRY-IRRELEVANT": "the three arms' retentions sit within the "
            "draw spread — the seat does not transfer to installs at 2.74M; "
            "the honest bound",
        "CANNOT-LAND": "the orthogonal install fails to reach the matched "
            "t=0 strength — the span is the install's substrate; a finding "
            "of a different kind",
        "ANY": "the trajectories verbatim",
    },
    "predictions_verbatim": {
        "(a)": "Under SEAT-IS-ENGINEERABLE: ARM-ORTHO's margins THICKEN "
            "faster than ARM-ALIGNED's (the forge works harder when the "
            "undertow cannot reach — e242's thickening + the exposure law "
            "composed).",
        "(b)": "The t=0 thermal read: ARM-ALIGNED's +1 heat exceeds "
            "ARM-ORTHO's (the transient is the gust, and the gust needs "
            "the span).",
        "(c)": "The u0-cloud caveat carried: installs are stream-relative "
            "clouds; the arms' geometries verified per-arm by measured "
            "overlap (report actual cos-to-span per arm, not the nominal "
            "design).",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE WASH = the g-cell W1 form VERBATIM "
        "(commit R=0.7 RAW L2 at each arm's own root + the seed-10902 "
        "300-step neutral wash — the bars' flat-phase retention and "
        "predictions (a)/(b) are wall-family objects); retention = min "
        "g-12 over {10,50,100,300} / the arm's OWN root g-12 (e225's "
        "flat-min convention on e242's registered grid; the full-8 "
        "{10,50,100,200,300} and +300 flavors co-reported); landed := "
        "root g-12 within +-10% of the committed g1c root read "
        "0.9026340246200562 -> [0.812371, 0.992897]; CANNOT-LAND fires "
        "iff ARM-ORTHO lands UNDER the floor (an over-landing arm is "
        "disclosed, never dose-trimmed); the draw spread := the 2.74M "
        "wall family's committed retention band [0.615392, 1.326643] "
        "(e225's joined table, n=5 roots, hard-bound at runtime); "
        "composite order TEXTURE(halt on G_FREE) -> CANNOT-LAND -> SEAT "
        "-> GEOMETRY-IRRELEVANT -> ANY; prediction reads: (a) 100*"
        "(med_margin(+300)/med_margin(t0)-1) larger for ORTHO than "
        "ALIGNED; (b) T_mle(+1) with each arm's OWN t0 L0 family, "
        "T_ALIGNED(+1) > T_ORTHO(+1); (c) measured cos-to-span ||S^T d||/"
        "||d|| at post-install AND final root + the gradient-level ledger "
        "medians; the span is root-fixed (e233's fixed-direction "
        "convention); the hook is backward -> clip 1.0 VERBATIM -> "
        "project (fp64 chunked) -> opt.step(), norm NOT rescaled."),
    "registration": "bars + predictions frozen VERBATIM from "
        "scratch/e246_design.md (the frozen design note) and the dispatch "
        "letter; this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE WASH FORM (the operationalization, frozen before compute): 'the "
    "standard wall wash (the g-cell convention)' is the g-cell's W1 form "
    "VERBATIM — commit(R=0.7 RAW L2) at each arm's OWN root, then the "
    "seed-10902 300-step neutral wash — read off the bars' own vocabulary "
    "('flat-phase retention'; prediction (a)'s forge thickening = e242's "
    "WALL phenomenon; prediction (b)'s +1 heat = e242's wall transient "
    "T(+1)=1.31). All three arms get the identical wall treatment; the "
    "only cross-arm deltas are the install gradient geometry and the "
    "roots it produces.",
    "THE HOOK (e237's pre-Adam projection ported to the INSTALL phase): "
    "backward -> clip_grad_norm_ 1.0 VERBATIM -> PROJECT -> opt.step(); "
    "the projection is the last operation before Adam (both moments + "
    "the update supply-cut); the projected gradient's norm is NOT "
    "rescaled; every dot/norm is fp64 (chunked per-parameter, e237/e226's "
    "instrument finding). ORTHO removes the span component; ALIGNED keeps "
    "only the span component.",
    "ARM-FREE runs the INSTRUMENTED path (the fp64 ledger dots computed "
    "every step, READ ONLY — no gradient write): the verbatim trajectory "
    "through the hook's code path, so G_FREE validates the instrument "
    "against the committed g1c root AND its committed W1 wash record — "
    "the strongest form of e237's C2 hook-path control.",
    "THE SPAN's provenance: computed ONCE at the COMMITTED g1c root (the "
    "design's 'the fresh root's wash-span'), fixed for the whole cell "
    "(root-fixed; e233's fixed-direction convention, disclosed); the "
    "install arms train from the e001 BASE with that fixed basis. LATE "
    "half = segments 11..20 of the 20-step seed-10902 unwalled history "
    "under G_SHADOW (e236's |cos(seg_k, -u0)| < 0.5 validity gate, per "
    "segment recorded); if any late segment violates, the EMPIRICAL clean "
    "trailing range inside the late half is used and disclosed; if none "
    "clears, STILL-PINNED -> the cell halts with the autopsy.",
    "e225/e229/e242 are NOT module-imported (each forces "
    "CUDA_VISIBLE_DEVICES=-1 at module level — this cell OWNS the GPU "
    "lane): svd_basis/cos64 (e225), the _E228NetShim adapter (e229), "
    "TempFamily + logit_pass (e242's verbatim ports of e238) are COPIED "
    "VERBATIM with inline provenance notes (e242's own precedent for "
    "e238's thermal class). e228 IS module-imported for margin_pass "
    "(arithmetic untouched; opens runs/e228_run.log append — benign side "
    "effect, e242's disclosed precedent).",
    "The margin/thermal readout states are read through g1's evl_load "
    "(settle onto the ball + disarm — the lineage's registered instrument "
    "adaptation; e242's convention); the thermal family base L0 is EACH "
    "ARM's OWN t0 full-logit re-probe (the family must be per-arm — the "
    "question is each arm's own T(+1) heat).",
    "The consolidation runs NATURAL on all three arms (e113 VERBATIM, "
    "seed 10901 HELD): the design's intervention is the INSTALL geometry "
    "only; cons re-growth of span components is part of the honest "
    "design and is captured by the measured cos-to-span at the final "
    "root (prediction (c)).",
    "Identical dose everywhere (s400 install / s300 cons / gen 24314 / "
    "cons seed 10901 / wash seed 10902): NO per-arm dose adjustment is "
    "attempted in the primary registration — an under-landing ORTHO is "
    "the CANNOT-LAND finding by design; an arm above the band ceiling is "
    "disclosed verbatim.",
    "n=1 per arm, one lineage, one wash draw (the g-series standing "
    "caveat); the draw-spread band is the family's n=5-root committed "
    "record (e225), itself one lineage — disclosed, never inflated.",
    "The span-history step-1 is certified against g1c's committed C-arm "
    "step-1 (CE/disp, tol 5e-3) and e185's stored step-1 batch md5 "
    "(bit-identity of the stream recipe, device-independent).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E246_SMOKE=1): 8-step install/cons, 4-step wash + "
    "4-step history (late start 3), ckpts {1,2,4}, own smoke dir; "
    "NOTHING adjudicated or gated (SMOKE stamp on every read).",
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
    jump stays under the never-past-85C line; a >= 84C read is a recorded
    VIOLATION (e237 matched: max 82C, zero >= 84C)."""
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
    # PORTED VERBATIM from lab/e225_one_currency.py (its own provenance
    # e191/e205/e209) — fp64 cosine of two flat fp32 CPU tensors.
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def participation_ratio(sv: torch.Tensor) -> float:
    # PORTED VERBATIM from e225.
    s2 = (sv.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """PORTED VERBATIM from lab/e225_one_currency.py (e_chart/e193b/e205/
    e209's Gram-based right-singular basis; fp64 arithmetic, CHUNKED)."""
    k, P = H.shape
    G = torch.zeros(k, k, dtype=torch.float64)
    chunk = max(1, int(40e6 // max(k, 1)))          # ~40M doubles per pass
    for s in range(0, P, chunk):
        Hc = H[:, s:s + chunk].to(torch.float64)
        G += Hc @ Hc.T
    evals, evecs = torch.linalg.eigh(G)     # ascending
    evals = torch.flip(evals, dims=(0,)).clamp(min=0)
    evecs = torch.flip(evecs, dims=(1,))
    sv = torch.sqrt(evals)
    pr = participation_ratio(sv) if float(sv[0]) > 0 else 0.0
    rank_eff = int((sv > sv[0] * 1e-7).sum())
    Vp = torch.empty(0, P)
    if rank_eff:
        acc = torch.zeros(rank_eff, P, dtype=torch.float64)
        for s in range(0, P, chunk):
            Hc = H[:, s:s + chunk].to(torch.float64)
            acc[:, s:s + chunk] += evecs[:, :rank_eff].T @ Hc
        rows = [acc[i] / sv[i].clamp(min=1e-30) for i in range(rank_eff)]
        Vp = torch.stack([r.to(torch.float32) for r in rows])
    return {"sv": sv, "Vp": Vp, "pr": pr, "rank_eff": rank_eff,
            "cond": float(sv[0] / sv[-1].clamp(min=1e-30))}


class SpanProjector:
    """The fixed span basis, held as per-parameter fp64 GPU slices (the
    flat order IS net.parameters() order). All dots fp64 (e237's
    instrument convention); the projection writes are fp32 (the grads'
    own dtype)."""

    def __init__(self, Vp: torch.Tensor, params_ref, dev: torch.device):
        self.k = int(Vp.shape[0])
        self.P = int(Vp.shape[1])
        self.slices = []
        off = 0
        for p in params_ref:
            n = p.numel()
            self.slices.append(Vp[:, off:off + n].to(dev, torch.float64)
                               .contiguous())
            off += n
        assert off == self.P, f"span flat size {off} != {self.P}"

    def step_hook(self, params, mode: str) -> dict:
        """THE PRE-ADAM HOOK (e237's, ported): ledger dots fp64 on the
        post-clip gradient; ORTHO removes the span component; ALIGNED
        keeps only the span component; FREE computes the ledger and
        writes NOTHING (the verbatim trajectory).

        TWO-PASS (the smoke-run autopsy fix): the in-span coordinates
        c = Vp g are a SUM over all parameter slices (cross-slice terms
        included) — a per-slice c is a block-diagonal approximation and
        is WRONG; pass 1 accumulates the full c, pass 2 writes each
        slice's true block Vp_s.T c."""
        params = list(params)
        c_full = torch.zeros(self.k, dtype=torch.float64,
                             device=self.slices[0].device)
        gn2 = 0.0
        g64s = []
        for p, S in zip(params, self.slices):
            g64 = p.grad.detach().reshape(-1).to(torch.float64)
            g64s.append(g64)
            c_full += S @ g64
            gn2 += float(g64 @ g64)
        ins2 = float(c_full @ c_full)
        gpn2 = 0.0
        if mode != "FREE":
            with torch.no_grad():
                for p, S, g64 in zip(params, self.slices, g64s):
                    corr = S.T @ c_full                       # fp64 (n,)
                    c32 = corr.to(torch.float32)
                    if mode == "ORTHO":
                        p.grad.add_((-c32).reshape(p.shape))
                        post = g64 - corr
                    else:                                     # ALIGNED
                        p.grad.copy_(c32.reshape(p.shape))
                        post = corr
                    gpn2 += float(post @ post)
        else:
            gpn2 = gn2
        return {"gn": math.sqrt(gn2), "in_span": math.sqrt(ins2),
                "gpn": math.sqrt(gpn2),
                "in_span_frac": math.sqrt(ins2) / math.sqrt(gn2)
                if gn2 > 0 else 0.0}

    def cos_to_span(self, flat_cpu: torch.Tensor) -> float:
        """||S^T d|| / ||d|| — the amplitude fraction of a displacement
        inside the span (fp64, CPU chunks; the FULL cross-slice sum —
        the same two-pass fix)."""
        d64 = flat_cpu.to(torch.float64)
        n2 = float(d64 @ d64)
        c_full = torch.zeros(self.k, dtype=torch.float64)
        off = 0
        for S in self.slices:
            n = S.shape[1]
            Sc = S.detach().to(CPU)
            c_full += Sc @ d64[off:off + n]
            off += n
        s2 = float(c_full @ c_full)
        return math.sqrt(s2) / math.sqrt(n2) if n2 > 0 else 0.0


# ------------------------------------------------- the shim + thermal ports
class _E228NetShim:
    """PORTED VERBATIM from lab/e229_wall_currency.py (adapter only — NO
    arithmetic): TinyGPT-lineage net(idx) -> (logits, loss) becomes the
    net(input_ids=...) -> .logits object e228's margin_pass expects."""

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


# ------------------------------------------------------------------ trainers
def _open_burst(tag: str) -> tuple[float, int]:
    wait_gpu_free(tag)
    t_burst = time.time()
    log(f"[{tag}] burst opens")
    return t_burst, 0


def _end_burst_early(tag: str, n_burst: int, t_burst: float) -> None:
    log(f"  [{tag}] burst ended: {n_burst} steps in "
        f"{time.time() - t_burst:.1f}s (thermal margin)")


def chunked_install(tag, mode, net0, proj: SpanProjector, inst_x, inst_mask,
                    anchor_full, train_ids, g0_ids, r_eval_xy, zid,
                    resume_ck: Path, dev: torch.device) -> dict:
    """g1c's chunked_install arithmetic VERBATIM (E43.exposure Dmix: ix(16)
    install w/ name mask + aj(16) paired + rj(32) random; masked token-level
    union CE; AdamW (0.9,0.95) wd 0.1; lr 1e-3 x house cosine(total=1000);
    clip 1.0) + THE HOOK: for ORTHO/ALIGNED each post-clip gradient is
    projected (fp64) before opt.step(); for FREE the ledger dots are READ
    ONLY (the verbatim trajectory)."""
    name_bs, corp_bs, mix_random, lr = G1.NAME_BS, E43.CORP_BS, E43.MIX_RANDOM, E43.LR
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
                "chunk_table": state.get("chunk_table", [])}
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
                    "in_span": led["in_span"], "gpn": led["gpn"],
                    "in_span_frac": led["in_span_frac"]}
            if step % 100 == 0 or step == n_steps or step == 1 or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step, "g0_pz": bz["mean_pz"],
                                      "ce_r": ce_r,
                                      "ce_batch": float(loss.item()),
                                      "in_span_frac": led["in_span_frac"],
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f} CE {float(loss.item()):.4f} |g| "
                    f"{led['gn']:.3f} in-span {led['in_span_frac']:.4f}")
            n_burst += 1
            # per-step thermal polls (the e233/e237 discipline)
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
                "chunk_table": state.get("chunk_table", [])}
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
                    "active_s": _pre.get("active_s", 0.0)}
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
        "experiment": "e246_engineered_seat",
        "phase": "THE BUILD LANE'S FIRST CELL — install a fact in a CHOSEN "
                 "geometry relative to the wash's active span (ORTHO / "
                 "ALIGNED / FREE), then the standard wall wash: does fate "
                 "follow the ENGINEERED geometry?",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 (the owner's max-priority window; this "
                      "cell owns the GPU lane), CPU probing threads 4 "
                      "(e240/e247 own the CPU lane)",
            "bursts": f"<= {BURST_MAX_S:.0f}s, per-step thermal polls at a "
                      f"{TEMP_EARLY_END:.0f}C margin from the first burst, "
                      f"cooldown {COOLDOWN_S:.0f}s, the {TEMP_HARD:.0f}C "
                      "never-past line recorded",
            "trainings": "3 installs s400 + 3 cons s300 + 3 wall washes s300 "
                         "+ the 20-step span history",
        },
        "deviations": deviations,
        "builds_on": [
            "scratch/e246_design.md (THE frozen registration — this cell's "
            "spec; the ambition directive 2026-10-04 / W020's generative "
            "turn)",
            "T223 / e237 (the triangle's closed book; the pre-Adam "
            "projection hook PORTED to the install phase — this cell tests "
            "whether the seat graduates from finding to tool)",
            "T215/T217 / e233+e234 (the undertow's anatomy: the wind is a "
            "scalpel, span-stable, anchor-scale at the extremes)",
            "T181 / g1c (the fresh-root lineage this cell builds on: "
            "e001+Dmix s400 gen 24314 + e113 cons 10901; the committed "
            "root + W1 records are the controls)",
            "T213/e231+e236 design / e225 (the span machinery: the "
            "seed-10902 20-step history, the Gram-SVD chunked-fp64 basis, "
            "G_SHADOW's late-half convention)",
            "T220 / e242 (the margins + thermal instruments this cell "
            "re-uses: e228's margin_pass; e238's one-T MLE)",
        ],
        "whats_new": [
            "THE FIRST BUILD CELL: the fact installed in an ENGINEERED "
            "geometry (orthogonal-to-span / in-span / natural) at identical "
            "dose — fate vs geometry, the seat as a construction tool",
            "the pre-Adam projection (e237) ported from the wash phase to "
            "the INSTALL phase — the instrument that proved causality "
            "becomes the construction tool",
            "the per-arm measured geometry ledger (cos-to-span at "
            "post-install/root + the gradient-level in-span fractions) — "
            "the u0-cloud caveat made quantitative per arm",
        ],
        "gates": {},
    })
    log(f"E246 — THE ENGINEERED SEAT (smoke={SMOKE}) -> {RD}")
    log(f"arms: ORTHO / ALIGNED / FREE at identical dose (e001 + Dmix "
        f"s{INST_STEPS} gen {FRESH_GEN} + e113 cons s{CONS_STEPS} seed "
        f"{CONS_SEED}); wall R={R_CLAIM} RAW; wash seed {WASH_SEED}; "
        f"matched band +-{MATCH_BAND:.0%} of the committed root "
        f"{G1C_ROOT_GM12:.4f}")
    write_partial("startup (bars registered, committed at birth)")
    set_seed(24601)                 # global init only; every RNG is its own

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
    g1c_root_gm12 = g1c["root_build"]["root_cells"]["gm12"]
    g1c_post_inst = g1c["root_build"]["install"]["post_cells"]
    g1c_c1 = g1c["arms"]["C"]["traj"][0]
    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"],
                        "root_gm12": g1c_root_gm12,
                        "post_install_cells": g1c_post_inst,
                        "W1_g_m12": {str(k): v for k, v in
                                     sorted(g1c_w1.items())},
                        "C_step1": {"ce": g1c_c1["ce_batch"],
                                    "disp": g1c_c1["cum_disp"]}},
        "e225_metrics": {"path": str(E225_METRICS), "md5": md5of(E225_METRICS)},
        "hardbound": {"root_gm12": G1C_ROOT_GM12, "W1_g_m12": G1C_W1_GM12,
                      "C_step1": G1C_C_STEP1},
        "pass": bool(
            g1c["adjudication"]["verdict"] == G1C_VERDICT
            and abs(g1c_root_gm12 - G1C_ROOT_GM12) < 1e-12
            and all(abs(g1c_w1[s] - G1C_W1_GM12[s]) < 1e-12
                    for s in G1C_W1_GM12)
            and abs(g1c_c1["ce_batch"] - G1C_C_STEP1["ce"]) < 1e-12
            and abs(g1c_c1["cum_disp"] - G1C_C_STEP1["disp"]) < 1e-12),
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
    G_PARENTS["pass"] = bool(G_PARENTS["pass"] and G_PARENTS["draw_spread"]["pass"])
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G1C_VERDICT} (root {g1c_root_gm12:.6f}, "
        f"W1 +300 {g1c_w1[300]:.6f}); draw spread "
        f"[{DRAW_SPREAD['lo']:.4f}, {DRAW_SPREAD['hi']:.4f}] (n=5 roots)")
    # refresh the hard-bound post-install gm12 literal at runtime (Rule 12:
    # the number lives at its path; the literal is the bind)
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

    # ================= P1: THE SPAN at the committed fresh root ==========
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    root_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
    G_ROOTSPAN = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "battery_read_measured": root_read,
        "battery_read_committed": G1C_ROOT_GM12,
        "abs_diff": abs(root_read - G1C_ROOT_GM12), "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - G1C_ROOT_GM12) < G_READ_TOL)}
    assert G_ROOTSPAN["pass"], f"root gate FAILED: {G_ROOTSPAN}"
    metrics["gates"]["G_ROOTSPAN"] = G_ROOTSPAN
    log(f"P1 G_ROOTSPAN: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {G1C_ROOT_GM12:.10f} "
        f"(|d| {abs(root_read - G1C_ROOT_GM12):.1e}): PASS")
    write_partial("P1 G_ROOTSPAN PASSED")

    # ---- the 20-step seed-10902 unwalled history + u0 + G_SHADOW --------
    wait_gpu_free("span-history")
    t_burst = time.time()
    hist_net = copy.deepcopy(root_net).to(dev)
    hist_net.train()
    hist_opt = torch.optim.AdamW(hist_net.parameters(), lr=G1.FT_LR,
                                 betas=(0.9, 0.95), weight_decay=0.1)
    hist_gen = torch.Generator().manual_seed(WASH_SEED)
    hist_prev = theta_root.clone()
    hist_segs, hist_rows = [], []
    g_ray = sign_ray = None
    for k in range(1, HIST_STEPS + 1):
        aj = torch.randint(16, (G1.ANCH_BS,), generator=hist_gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                           generator=hist_gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[q: q + G1.BLOCK] for q in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        if k == 1:
            x1_md5 = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
        xd, yd = x.to(dev), y.to(dev)
        logits, _ = hist_net(xd)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yd.reshape(-1))
        hist_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hist_net.parameters(), 1.0)
        if k == 1:                    # the t=0 post-clip rays (e225's conv.)
            g0 = torch.cat([p.grad.detach().reshape(-1).cpu()
                            for p in hist_net.parameters()]).clone()
            g_ray = (g0 / torch.norm(g0)).clone()
            sgn = torch.sign(g0)
            sign_ray = (sgn / torch.norm(sgn)).clone()
        hist_opt.step()
        cur = flat_params_cpu(hist_net)
        seg = cur - hist_prev
        hist_segs.append(seg)
        hist_rows.append({"step": k,
                          "L2": float(torch.norm(seg)),
                          "cum": float(torch.norm(cur - theta_root)),
                          "ce": float(loss.item())})
        hist_prev = cur
        log(f"  [hist s{k}] L2 {hist_rows[-1]['L2']:.4f} cum "
            f"{hist_rows[-1]['cum']:.4f} CE {hist_rows[-1]['ce']:.4f}")
        ok_t, _temp = burst_temp_check("span-history")
        if not ok_t:
            log("  [span-history] thermal margin hit — pausing")
            burst_cooldown("span-history")
            wait_gpu_free("span-history-cont")
            t_burst = time.time()
        elif (time.time() - t_burst) > BURST_MAX_S:
            log("  [span-history] burst cap — continuing in a fresh burst")
            burst_cooldown("span-history")
            wait_gpu_free("span-history-2")
            t_burst = time.time()
    assert len(hist_segs) == HIST_STEPS, \
        f"history incomplete: {len(hist_segs)}/{HIST_STEPS} (thermal?)"
    hist_net.eval()
    del hist_net, hist_opt

    # G_STREAM: the step-1 batch + CE + first displacement vs g1c's C-arm
    step1_disp = float(torch.norm(hist_segs[0]))
    G_STREAM = {
        "seed": WASH_SEED, "x1_md5": x1_md5,
        "md5_match_e185": bool(x1_md5 == E185_XHASH_1),
        "ce1_measured": hist_rows[0]["ce"],
        "ce1_committed": G1C_C_STEP1["ce"],
        "disp1_measured": step1_disp,
        "disp1_committed": G1C_C_STEP1["disp"],
        "pass": bool(x1_md5 == E185_XHASH_1
                     and abs(hist_rows[0]["ce"] - G1C_C_STEP1["ce"])
                     < G_READ_TOL
                     and abs(step1_disp - G1C_C_STEP1["disp"]) < G_READ_TOL),
        "note": "the span-history's step-1 batch/CE/displacement vs g1c's "
                "committed C-arm step-1 (same root, same 10902 stream; "
                "cross-device texture tol) + e185's stored stream md5",
    }
    metrics["gates"]["G_STREAM"] = G_STREAM
    log(f"G_STREAM: md5 {'OK' if G_STREAM['md5_match_e185'] else 'DRIFT'}, "
        f"CE1 {hist_rows[0]['ce']:.6f} (committed {G1C_C_STEP1['ce']:.6f}), "
        f"disp1 {step1_disp:.6f} (committed {G1C_C_STEP1['disp']:.6f}): "
        f"{'PASS' if G_STREAM['pass'] else 'FAIL'}")
    assert G_STREAM["pass"], "stream gate FAILED"

    # G_SHADOW (e236's convention): cos(seg_k, -u0) decay + the late half
    shadow = [cos64(seg, -sign_ray) for seg in hist_segs]
    late_idx = list(range(LATE_START - 1, HIST_STEPS))    # segments LATE_START..20
    late_shadow = [shadow[i] for i in late_idx]
    late_clean = [abs(c) < SHADOW_BAR for c in late_shadow]
    G_SHADOW = {
        "instrument": "cos(seg_k, -u0) for k=1..20 (e236's sign-shadow "
                      "decay; u0 = the t=0 post-clip sign-ray of the "
                      "seed-10902 stream's first step at the g1c root)",
        "decay": dict(zip(range(1, HIST_STEPS + 1), shadow)),
        "late_half_segments": [i + 1 for i in late_idx],
        "late_half_max_abs_cos": float(max(abs(c) for c in late_shadow)),
        "shadow_bar": SHADOW_BAR,
        "late_half_all_clean": bool(all(late_clean)),
        "cos_g_sign_t0": cos64(g_ray, sign_ray),
        "pass": True,        # informational validity gate; see G_SPAN
    }
    log("G_SHADOW: " + " ".join(f"s{k}:{c:+.3f}" for k, c in
                                zip(range(1, HIST_STEPS + 1), shadow)))
    log(f"  late half (s{LATE_START}..{HIST_STEPS}) max|cos| "
        f"{G_SHADOW['late_half_max_abs_cos']:.4f} vs bar {SHADOW_BAR}: "
        f"{'CLEAN' if G_SHADOW['late_half_all_clean'] else 'PINNED'} "
        f"(cos(g,sign) at t0 {G_SHADOW['cos_g_sign_t0']:.4f})")

    # THE LATE SPAN (Gram-SVD of the late half's segments)
    used_idx = late_idx
    if not G_SHADOW["late_half_all_clean"]:
        # the empirical clean trailing range inside the late half (e236's
        # instrument clause; disclosed) — the longest clean suffix
        last_dirty = max((i for i, ok in zip(late_idx, late_clean)
                          if not ok), default=None)
        if last_dirty is not None:
            used_idx = [i for i in late_idx if i > last_dirty]
        G_SHADOW["empirical_clean_range"] = [i + 1 for i in used_idx]
        if len(used_idx) < 2:
            G_SHADOW["pass"] = False
    assert G_SHADOW["pass"] and len(used_idx) >= 2, (
        "STILL-PINNED: no >=2-segment clean late range under the 0.5 "
        f"shadow bar — the instrument is invalid; autopsy required "
        f"(decay {G_SHADOW['decay']})")
    H_late = torch.stack([hist_segs[i] for i in used_idx])
    basis = svd_basis(H_late)
    Vp = basis["Vp"].contiguous()
    gram_chk = Vp @ Vp.T
    G_SPAN = {
        "n_segments": int(Vp.shape[0]), "rank_eff": basis["rank_eff"],
        "sv": [float(s) for s in basis["sv"][:basis["rank_eff"]]],
        "participation_ratio": basis["pr"], "cond": basis["cond"],
        "orthonormality_max_dev": float((gram_chk - torch.eye(
            Vp.shape[0])).abs().max()),
        "segments_used": [i + 1 for i in used_idx],
        "late_half_all_clean": G_SHADOW["late_half_all_clean"],
        "pass": bool(basis["rank_eff"] >= 2
                     and float((gram_chk - torch.eye(Vp.shape[0])).abs().max())
                     < 1e-3),
    }
    assert G_SPAN["pass"], f"span gate FAILED: {G_SPAN}"
    metrics["gates"]["G_SHADOW"] = G_SHADOW
    metrics["gates"]["G_SPAN"] = G_SPAN
    torch.save({"Vp": Vp, "sv": basis["sv"],
                "meta": {"experiment": NAME,
                         "desc": "the LATE-half Gram-SVD wash-span at the "
                                 "committed g1c root (seed-10902 20-step "
                                 "history, G_SHADOW-gated)",
                         "segments": [i + 1 for i in used_idx],
                         "root": f"runs/checkpoints/{ROOT_CK}"}},
               CKPT_DIR / ("smoke_e246_late_span.pt" if SMOKE
                           else "e246_late_span.pt"))
    metrics["span"] = {
        "history": {"steps": HIST_STEPS, "seed": WASH_SEED,
                    "rows": hist_rows, "unwalled": True},
        "G_SHADOW": G_SHADOW, "span": G_SPAN,
        "checkpoint": "runs/checkpoints/"
                      + ("smoke_e246_late_span.pt" if SMOKE
                         else "e246_late_span.pt"),
        "fixed_for_whole_cell": True,
    }
    log(f"P1 SPAN: {G_SPAN['n_segments']} segments (used "
        f"{G_SPAN['segments_used']}), rank {G_SPAN['rank_eff']}, PR "
        f"{G_SPAN['participation_ratio']:.2f}, cond {G_SPAN['cond']:.1f}, "
        f"ortho dev {G_SPAN['orthonormality_max_dev']:.1e}: PASS")
    del hist_segs, H_late
    write_partial("P1 THE SPAN built (history + G_SHADOW + Gram-SVD)")
    burst_cooldown("span->installs")

    # ================= P2/P3/P4: THE THREE ARMS ==========================
    proj = SpanProjector(Vp, list(G1.evl_load(base_sd).parameters()), dev)
    arms_rec: dict = {}
    theta0s: dict = {}
    for arm in ARMS:
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        inst = chunked_install(
            f"{arm}-inst", arm, G1.evl_load(base_sd), proj, inst_x,
            inst_mask, anchor_full, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e246_{arm}_inst_resume.pt" if SMOKE
                        else f"e246_{arm}_inst_resume.pt"), dev)
        sd_install = inst["sd"]
        # post-install lean dial + geometry
        inst_net = G1.evl_load(sd_install)
        inst_cells = {"gm12": G1.battery_cell(inst_net, gm12_ids, zid)["mean_pz"],
                      "g0": G1.battery_cell(inst_net, g0_ids, zid)["mean_pz"],
                      "gp12": G1.battery_cell(inst_net, bat_ids[12], zid)["mean_pz"],
                      "ce_r": G1.ce_fixed_cpu(inst_net, *r_eval_xy)}
        d_inst = flat_params_cpu(inst_net) - flat_params_cpu(G1.evl_load(base_sd))
        cos_span_inst = proj.cos_to_span(d_inst)
        del inst_net
        led_fracs = [v["in_span_frac"] for v in inst["ledger"].values()]
        arms_rec[arm] = {
            "desc": ARM_DESC[arm], "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "ledger_in_span_frac_median":
                    float(sorted(led_fracs)[len(led_fracs) // 2])
                    if led_fracs else None,
                "chunk_table": inst["chunk_table"], "steps": INST_STEPS,
                "post_cells": inst_cells,
                "cos_to_span_displacement": cos_span_inst},
        }
        log(f"ARM-{arm} install done: post g-12 {inst_cells['gm12']:.4f} "
            f"g0 {inst_cells['g0']:.4f} CE_R {inst_cells['ce_r']:.4f} | "
            f"cos-to-span(d) {cos_span_inst:.4f} | ledger in-span median "
            f"{arms_rec[arm]['install']['ledger_in_span_frac_median']:.4f}")
        write_partial(f"P2 ARM-{arm} install + post dial + geometry")

        burst_cooldown(f"{arm} inst->cons")
        cons = chunked_consolidate(
            f"{arm}-cons", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e246_{arm}_cons_resume.pt" if SMOKE
                        else f"e246_{arm}_cons_resume.pt"), dev)
        theta0 = cons["sd"]
        root_net_arm = G1.evl_load(theta0)
        cells = {f"g{j:+d}": G1.battery_cell(root_net_arm, bat_ids[j],
                                             zid)["mean_pz"] for j in G1.GEOS}
        cells["held30_gm12"] = G1.battery_cell(root_net_arm, held_ids[-12],
                                               zid)["mean_pz"]
        cells["ce_r"] = G1.ce_fixed_cpu(root_net_arm, *r_eval_xy)
        d_root = flat_params_cpu(root_net_arm) - flat_params_cpu(
            G1.evl_load(base_sd))
        cos_span_root = proj.cos_to_span(d_root)
        theta0s[arm] = theta0
        root_ck = save_ckpt(
            f"e246_{arm}_root", theta0,
            {"desc": f"e246 ARM-{arm} root: e001 + Dmix s{INST_STEPS} "
                     f"(gen {FRESH_GEN}, {ARM_DESC[arm]}) + e113 cons "
                     f"s{CONS_STEPS} (seed {CONS_SEED} HELD)",
             "arm": arm, "install_seed": FRESH_GEN, "mode": arm,
             "base": f"runs/checkpoints/{BASE_CK}",
             "span": "runs/checkpoints/e246_late_span.pt"})
        arms_rec[arm]["consolidation"] = {"traj": cons["traj"],
                                          "chunk_table": cons["chunk_table"]}
        arms_rec[arm]["root"] = {"cells": cells,
                                 "gm12": cells["g-12"],
                                 "cos_to_span_displacement": cos_span_root,
                                 "checkpoint": root_ck}
        landed = (G1C_ROOT_GM12 * (1 - MATCH_BAND)
                  <= cells["g-12"] <= G1C_ROOT_GM12 * (1 + MATCH_BAND))
        arms_rec[arm]["root"]["landed"] = bool(landed)
        log(f"ARM-{arm} ROOT: g-12 {cells['g-12']:.4f} (band "
            f"[{G1C_ROOT_GM12 * (1 - MATCH_BAND):.4f}, "
            f"{G1C_ROOT_GM12 * (1 + MATCH_BAND):.4f}]) landed={landed} | "
            + " ".join(f"g{j:+d} {cells[f'g{j:+d}']:.4f}" for j in G1.GEOS)
            + f" | held30 {cells['held30_gm12']:.4f} CE_R "
            f"{cells['ce_r']:.4f} | cos-to-span(d_root) "
            f"{cos_span_root:.4f}")
        del root_net_arm
        metrics["arms"] = arms_rec
        write_partial(f"P3/P4 ARM-{arm} root built + matched read")
        if arm != ARMS[-1]:
            burst_cooldown(f"{arm} -> next arm")

    # ---- G_FREE: THE CONTROL (the design's halt clause) -----------------
    free_root_gm12 = arms_rec["FREE"]["root"]["gm12"]
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
    free_post = arms_rec["FREE"]["install"]["post_cells"]
    G_FREE = {
        "form": "ARM-FREE must reproduce the committed g1c root (the "
                "design's halt clause): root g-12 within 5e-3 of the "
                "committed read AND the post-install cells within 5e-3; "
                "max|diff|/L2 vs g1c_root.pt recorded (device texture)",
        "root_gm12": free_root_gm12,
        "committed_root_gm12": G1C_ROOT_GM12,
        "root_gm12_abs_diff": abs(free_root_gm12 - G1C_ROOT_GM12),
        "max_abs_diff_vs_ckpt": mdf, "l2_vs_ckpt": l2f,
        "post_install_gm12": free_post["gm12"],
        "committed_post_install_gm12": G1C_POST_INSTALL["gm12"],
        "post_install_abs_diff": abs(free_post["gm12"]
                                     - G1C_POST_INSTALL["gm12"]),
        "tol": G_READ_TOL,
        "pass": bool(abs(free_root_gm12 - G1C_ROOT_GM12) < G_READ_TOL
                     and abs(free_post["gm12"]
                             - G1C_POST_INSTALL["gm12"]) < G_READ_TOL),
    }
    metrics["gates"]["G_FREE"] = G_FREE
    log(f"G_FREE (THE CONTROL): FREE root g-12 {free_root_gm12:.6f} vs "
        f"committed {G1C_ROOT_GM12:.6f} (|d| "
        f"{abs(free_root_gm12 - G1C_ROOT_GM12):.1e}); post-install g-12 "
        f"{free_post['gm12']:.6f} vs {G1C_POST_INSTALL['gm12']:.6f}; "
        f"vs ckpt max|diff| {mdf:.3e} L2 {l2f:.3f}: "
        f"{'PASS' if G_FREE['pass'] else 'FAIL — THE CELL HALTS'}")
    write_partial("P4 G_FREE read" + ("" if G_FREE["pass"] else " — FAILED"))
    if not G_FREE["pass"] and not SMOKE:
        metrics["status"] = ("HALTED — G_FREE FAILED (the natural install "
                             "did not reproduce the committed g1c root; the "
                             "design's halt clause; nothing adjudicated)")
        write_partial("HALTED (G_FREE)")
        return 1

    # ================= P5: THE WALL WASHES ===============================
    washes: dict = {}
    G_BITROOT, G_INPUTS = {}, {"note": "per-step input batches bit-identical "
                                       "across the three arms (md5)"}
    for arm in ARMS:
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
            CKPT_DIR / (f"smoke_e246_{arm}_wash_resume.pt" if SMOKE
                        else f"e246_{arm}_wash_resume.pt"), dev)
        assert arm_w["zeph_violations"] == 0, \
            f"{arm}: name token leaked into a wash window"
        washes[arm] = arm_w
        # geometry texture: the +300 wash displacement's cos-to-span
        if WASH_STEPS in arm_w.get("deltas", {}):
            washes[arm]["cos_to_span_d300"] = proj.cos_to_span(
                arm_w["deltas"][WASH_STEPS])
        metrics["arms"][arm]["wash"] = {
            "R": R_CLAIM, "seed": WASH_SEED, "steps_ran": arm_w["steps_ran"],
            "ckpt_steps": list(CK_WASH), "traj": arm_w["traj"],
            "chunk_table": arm_w["chunk_table"],
            "cos_to_span_d300": washes[arm].get("cos_to_span_d300"),
            "theta0_norm": arm_w.get("theta0_norm"),
            "wall_R": arm_w.get("wall_R")}
        write_partial(f"P5 ARM-{arm} wall wash done")
        if arm != ARMS[-1]:
            burst_cooldown(f"{arm} wash -> next")
    # cross-arm input identity
    common_steps = set(washes[ARMS[0]]["x_hashes"])
    for arm in ARMS[1:]:
        common_steps &= set(washes[arm]["x_hashes"])
    G_INPUTS["steps_compared"] = len(common_steps)
    G_INPUTS["identical"] = bool(common_steps and all(
        washes[a]["x_hashes"][s] == washes[ARMS[0]]["x_hashes"][s]
        for a in ARMS for s in common_steps))
    G_INPUTS["pass"] = G_INPUTS["identical"]
    assert G_INPUTS["pass"], "cross-arm wash input streams diverged"
    metrics["gates"]["G_INPUTS"] = G_INPUTS
    G_BITROOT["pass"] = bool(all(v["pass"] for v in G_BITROOT.values()
                                 if isinstance(v, dict) and "pass" in v))
    metrics["gates"]["G_BITROOT"] = G_BITROOT
    log(f"G_INPUTS: per-step wash inputs bit-identical across all three "
        f"arms (md5, {len(common_steps)} steps): PASS")

    # FREE's wash vs the committed g1c W1 record (the control's second half)
    free_gm12 = {t["step"]: t["g_m12_mean_pz"] for t in washes["FREE"]["traj"]
                 if "g_m12_mean_pz" in t}
    free_w1_rows = {s: {"measured": free_gm12.get(s),
                        "committed": G1C_W1_GM12.get(s),
                        "abs_diff": abs(free_gm12.get(s, 0)
                                        - G1C_W1_GM12.get(s, 0))}
                    for s in sorted(set(free_gm12) & set(G1C_W1_GM12))}
    G_FREE["w1_match"] = {
        "rows": free_w1_rows,
        "max_abs_diff": max(r["abs_diff"] for r in free_w1_rows.values()),
        "tol": G_READ_TOL,
        "pass": bool(all(r["abs_diff"] < G_READ_TOL
                         for r in free_w1_rows.values()))}
    G_FREE["pass"] = bool(G_FREE["pass"] and G_FREE["w1_match"]["pass"])
    metrics["gates"]["G_FREE"] = G_FREE
    log(f"G_FREE w1_match: max |d| vs committed W1 "
        f"{G_FREE['w1_match']['max_abs_diff']:.1e} (tol {G_READ_TOL}): "
        f"{'PASS' if G_FREE['w1_match']['pass'] else 'FAIL'}")
    write_partial("P5 G_FREE w1_match read")

    # ================= P6: THE READOUTS (ruler + margins + thermal) ======
    pbatt = [{"ids": gm12_ids[i: i + 1], "fact": f"install{i:02d}@g-12",
              "relation": "install60_g-12", "ans_id": zid}
             for i in range(gm12_ids.shape[0])]
    readouts: dict = {}
    for arm in ARMS:
        cells_state: dict = {}
        # t0 = the arm's root (pre-wash)
        net0 = G1.evl_load(theta0s[arm])
        shim = _E228NetShim(net0)
        mrec = E228.margin_pass(shim, pbatt)
        lrec = logit_pass(shim, pbatt)
        dp = max(abs(r["p"] - m["p"])
                 for r, m in zip(lrec["rows"], mrec["probes"]))
        ms0 = [r["margin_sigma"] for r in mrec["probes"]]
        cells_state["t0"] = {
            "step": 0, "gm12": arms_rec[arm]["root"]["gm12"],
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
    metrics["wash_health"] = {a: arm_health(a) for a in ARMS}
    write_partial("P6 readouts COMPLETE (ruler + margins + thermal)")

    # ================= P7: ADJUDICATION (the frozen bars) ================
    def gm12_series(arm):
        out = {0: arms_rec[arm]["root"]["gm12"]}
        for t in washes[arm]["traj"]:
            if "g_m12_mean_pz" in t:
                out[t["step"]] = t["g_m12_mean_pz"]
        return out

    retention = {}
    for arm in ARMS:
        g = gm12_series(arm)
        r_root = g[0]
        flat_primary = [g[s] for s in FLAT_246 if s in g]
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

    landed = {a: bool(retention[a]["landed"]) for a in ARMS}
    ret_o = retention["ORTHO"]["retention_primary"]
    ret_a = retention["ALIGNED"]["retention_primary"]
    ret_f = retention["FREE"]["retention_primary"]
    ratio = (ret_o / ret_a if (ret_o is not None and ret_a
                               and ret_a > 0) else None)
    band_lo, band_hi = DRAW_SPREAD["lo"], DRAW_SPREAD["hi"]
    rets_in_band = all(r is not None and band_lo <= r <= band_hi
                       for r in (ret_o, ret_a, ret_f))

    cannot_land = bool(not landed["ORTHO"]
                       and arms_rec["ORTHO"]["root"]["gm12"]
                       < G1C_ROOT_GM12 * (1 - MATCH_BAND))
    seat_fires = bool(landed["ORTHO"] and landed["ALIGNED"]
                      and ratio is not None and ratio >= 1.5)
    geom_fires = bool(not seat_fires and rets_in_band)

    # wash-health texture + over-landing disclosures
    over_land = [a for a in ARMS
                 if arms_rec[a]["root"]["gm12"]
                 > G1C_ROOT_GM12 * (1 + MATCH_BAND)]

    # predictions (co-reports, never bars)
    def med(arm, key):
        return readouts[arm]["states"][key]["margin_median_sigma"]

    marg_growth = {a: 100.0 * (med(a, str(WASH_STEPS)) / med(a, "t0") - 1.0)
                   for a in ARMS}
    pred_a = {"margin_growth_pct_300": marg_growth,
              "ORTHO_faster": bool(marg_growth["ORTHO"]
                                   > marg_growth["ALIGNED"])}
    T1 = {a: readouts[a]["states"].get("1", {}).get("T_mle")
          for a in ARMS}
    pred_b = {"T_mle_plus1": T1,
              "ALIGNED_exceeds_ORTHO": bool(
                  T1["ALIGNED"] is not None and T1["ORTHO"] is not None
                  and T1["ALIGNED"] > T1["ORTHO"])}
    pred_c = {a: {"cos_to_span_post_install":
                      arms_rec[a]["install"]["cos_to_span_displacement"],
                  "cos_to_span_root":
                      arms_rec[a]["root"]["cos_to_span_displacement"],
                  "ledger_in_span_frac_median":
                      arms_rec[a]["install"]["ledger_in_span_frac_median"],
                  "cos_to_span_wash_d300":
                      washes[a].get("cos_to_span_d300")} for a in ARMS}
    metrics["predictions"] = {"(a)": pred_a, "(b)": pred_b, "(c)": pred_c}

    # the hard gate set
    hard = {k: v for k, v in metrics["gates"].items()
            if k not in ("G_SHADOW",)}
    gates_pass = bool(all(g.get("pass") for g in hard.values())
                      and G_SHADOW["pass"])

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy (the build lane's first-cell "
                  "discipline)")
    elif cannot_land:
        verdict = "CANNOT-LAND"
        clause = (f"the orthogonal install failed to reach the matched t=0 "
                  f"strength: ARM-ORTHO's root g-12 "
                  f"{retention['ORTHO']['root_gm12']:.4f} < the band floor "
                  f"{G1C_ROOT_GM12 * (1 - MATCH_BAND):.4f} (the committed "
                  f"root's {G1C_ROOT_GM12:.4f} - 10%) — the span is the "
                  f"install's substrate; a finding of a different kind "
                  f"(ARM-ALIGNED landed {retention['ALIGNED']['root_gm12']:.4f}, "
                  f"ARM-FREE {retention['FREE']['root_gm12']:.4f}; the "
                  f"geometries verbatim in predictions.(c))")
    elif seat_fires:
        verdict = "SEAT-IS-ENGINEERABLE"
        clause = (f"ARM-ORTHO retains {ret_o:.4f} >= 1.5x ARM-ALIGNED's "
                  f"{ret_a:.4f} (ratio {ratio:.2f}x; both landed at matched "
                  f"t=0 strength {landed['ORTHO']}/{landed['ALIGNED']}) — "
                  f"fate follows the engineered geometry; the seat is a "
                  f"construction tool (the third dimension's practical "
                  f"name: INSTALL GEOMETRY; ARM-FREE "
                  f"{ret_f:.4f}; the design's letter)")
    elif geom_fires:
        verdict = "GEOMETRY-IRRELEVANT"
        clause = (f"the three arms' retentions sit within the draw spread "
                  f"[{band_lo:.3f}, {band_hi:.3f}] (ORTHO {ret_o:.4f} / "
                  f"ALIGNED {ret_a:.4f} / FREE {ret_f:.4f}; ratio "
                  f"{ratio:.2f}x < 1.5) — the seat observed at 124M does "
                  f"not transfer to installs at 2.74M; the honest bound "
                  f"(and the worlds stay separate)")
    else:
        why = []
        if not landed["ORTHO"] or not landed["ALIGNED"]:
            why.append("a bar arm landed OUTSIDE the matched band "
                       f"(ORTHO landed {landed['ORTHO']}, ALIGNED "
                       f"{landed['ALIGNED']}"
                       + (f"; over-landing arms: {over_land}" if over_land
                          else "") + ")")
        if ratio is not None and ratio < 1.5 and not rets_in_band:
            why.append(f"the retentions fall OUTSIDE the draw spread too "
                       f"(ratio {ratio:.2f}; the family band "
                       f"[{band_lo:.3f}, {band_hi:.3f}])")
        verdict = "ANY"
        clause = ("; ".join(why) + " — the trajectories verbatim"
                  if why else "the trajectories verbatim")

    log("=" * 78)
    log(f"E246 VERDICT: {verdict}")
    for a in ARMS:
        g = gm12_series(a)
        log(f"  {a} (wash {WASH_SEED}): g-12 "
            + " -> ".join(f"+{s}:{v:.4f}" for s, v in sorted(g.items())))
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"
    log(f"  retentions: ORTHO {f4(ret_o)} | ALIGNED {f4(ret_a)} | FREE "
        f"{f4(ret_f)} (ratio {ratio}); landed "
        + "/".join(f"{a}:{landed[a]}" for a in ARMS))
    log(f"  predictions: (a) ORTO-thickens-faster {pred_a['ORTHO_faster']} "
        f"(growth " + "/".join(f"{a}:{marg_growth[a]:+.1f}%"
                               for a in ARMS) + ") | (b) "
        f"ALIGNED(+1)>{ 'ORTHO' if pred_b['ALIGNED_exceeds_ORTHO'] else 'read' } "
        f"(T+1 " + "/".join(f"{a}:{T1[a]:.3f}" for a in ARMS)
        + f") | (c) cos-to-span(root) "
        + "/".join(f"{a}:{pred_c[a]['cos_to_span_root']:.4f}" for a in ARMS))
    log(f"  {clause}")
    log("=" * 78)
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE -> CANNOT-LAND -> SEAT-IS-ENGINEERABLE "
                           "-> GEOMETRY-IRRELEVANT -> ANY (frozen)",
        "gates_pass": gates_pass,
        "reads": {"retention_primary": {a: retention[a]["retention_primary"]
                                        for a in ARMS},
                  "retention_full8": {a: retention[a]["retention_full8"]
                                      for a in ARMS},
                  "retention_plus300": {a: retention[a]["retention_plus300"]
                                        for a in ARMS},
                  "ratio_ORTHO_over_ALIGNED": ratio,
                  "draw_spread": [band_lo, band_hi],
                  "rets_in_band": rets_in_band,
                  "landed": landed, "over_landing_arms": over_land,
                  "matched_band": [G1C_ROOT_GM12 * (1 - MATCH_BAND),
                                   G1C_ROOT_GM12 * (1 + MATCH_BAND)]},
        "SEAT_IS_ENGINEERABLE": bool(gates_pass and seat_fires),
        "GEOMETRY_IRRELEVANT": bool(gates_pass and not cannot_land
                                    and not seat_fires and geom_fires),
        "CANNOT_LAND": bool(gates_pass and cannot_land),
        "verdict": verdict, "clause": clause,
        "smoke_stamp": "SMOKE — nothing adjudicated" if SMOKE else None,
    }
    if SMOKE:
        metrics["adjudication"]["verdict"] = "SMOKE (nothing adjudicated)"
    write_partial("P7 the frozen bars adjudicated")

    # ================= P8: the figures ====================================
    make_fates_plot(RD, gm12_series, retention, washes, readouts, verdict,
                    clause, landed, ratio, G1C_W1_GM12)
    make_geometry_thermal_plot(RD, G_SHADOW, G_SPAN, pred_c, pred_a, pred_b,
                               readouts, arms_rec, thermal_log)

    # ================= P9: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": ("the arms share bit-identical install "
            "streams (one generator, seed 24314, one draw order), identical "
            "dose/schedule/optimizer; the ONLY delta is the pre-Adam "
            "gradient geometry (G_FREE validates the instrumented path "
            "against the committed g1c root AND its committed W1 wash); the "
            "washes share bit-identical input streams across arms "
            "(G_INPUTS, md5) with each arm's wall committed at its own "
            "theta0 (G_BITROOT) — fate differences are geometry-caused or "
            "nothing is"),
        "n_and_scope": ("n=1 per arm, one lineage (the g1c fresh-root "
            "convention), one wash draw (seed 10902) — the standing g-series "
            "caveat; the draw spread is the family's committed n=5-root "
            "band (e225), itself one lineage; the retention ratio is a "
            "single-draw statistic against that band, quoted with both "
            "flavors (primary flat-{10,50,100,300} + the full-8 co-report)"),
        "geometry_measured_not_nominal": ("prediction (c) honored: every "
            "arm's ACTUAL cos-to-span is reported (post-install "
            "displacement, final-root displacement, the +300 wash "
            "displacement, and the per-step gradient ledger medians) — the "
            "nominal design (ORTHO ~0 in-span, ALIGNED ~1) is checked, "
            "never assumed; installs are stream-relative clouds and the "
            "consolidation runs natural on all arms (its span re-growth is "
            "visible in the root-vs-post-install cos table)"),
        "dose_disclosure": ("identical dose everywhere (s400/s300, lr "
            "schedules verbatim); NO per-arm dose adjustment was attempted "
            "— the projection is not norm-rescaled (e237's convention); "
            "ARM-ORTHO's per-step effective gradient loses the in-span "
            "component (the ledger's in_span_frac median is the measured "
            "size of that loss); an under-landing ORTHO is the CANNOT-LAND "
            "finding, never re-dosed"),
        "span_provenance": ("the span is ONE history realization (the "
            "seed-10902 20-step unwalled wash at the committed g1c root), "
            "LATE half only, G_SHADOW-gated (e236's convention; the decay "
            "curve is recorded per segment); it is FIXED for the whole "
            "cell (e233's fixed-direction convention); e231's "
            "RECIPE-IDENTITY autopsy is the reason the early half is "
            "excluded — the sign-step's shadow would pin any early span"),
        "first_build_cell": ("any instrument surprise gets the full autopsy "
            "before the verdict is read (the design's honesty guard): the "
            "record carries the ledger, the shadow curve, the span "
            "spectrum, the geometry table, and the thermal/margin "
            "trajectories verbatim"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOTSPAN["flat_md5"]},
            "late_span": metrics["span"]["checkpoint"],
            "arm_roots": {a: arms_rec[a]["root"]["checkpoint"]
                          for a in ARMS},
            "arm_wash_resumes": {
                a: f"runs/checkpoints/"
                  f"{'smoke_' if SMOKE else ''}e246_{a}_wash_resume.pt"
                for a in ARMS},
        },
        "machinery": {
            "install_cons_wash": "g1c's chunked drivers VERBATIM "
                                 "arithmetic (E43.exposure Dmix / e113 "
                                 "consolidate / G1.g1_wash), ported with "
                                 "the pre-Adam hook + this cell's thermal "
                                 "envelope",
            "hook": "e237's pre-Adam projection (fp64 chunked "
                    "per-parameter dots; backward -> clip 1.0 -> project "
                    "-> step; norm not rescaled), projected onto/onto-off "
                    "the LATE-half span",
            "span": "e225's svd_basis + cos64 PORTED VERBATIM (copied; "
                    "e225 forces CUDA off at import — this cell owns the "
                    "GPU lane)",
            "margins": "e228's margin_pass MODULE-IMPORTED via e229's "
                       "_E228NetShim adapter (ported copy; adapter only)",
            "thermal": "e238's TempFamily via e242's port (copied "
                       "verbatim; grid 1201 + golden polish + Bernoulli "
                       "NLL/R2)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training",
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
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e246_engineered_seat.png"),
                          str(RD / "e246_geometry_thermal.png")]
    write_partial("P9 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_fates_plot(rd, gm12_series, retention, washes, readouts, verdict,
                    clause, landed, ratio, w1_committed):
    """THE FATE FIGURE: the three arms' g-12 trajectories through the wall
    wash (+ the committed g1c W1/C references) + the margins + the
    verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.6))
    col = {"ORTHO": "#c0392b", "ALIGNED": "#8e44ad", "FREE": "#e67e22"}

    # (0,0) THE FATE TRAJECTORIES
    ax = axes[0, 0]
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    for arm in ARMS:
        g = gm12_series(arm)
        xs, ys = sorted(g), [g[s] for s in sorted(g)]
        ax.plot(xs, ys, "o-", ms=6, lw=2.2, color=col[arm], alpha=0.95,
                label=f"ARM-{arm} (root {g[0]:.3f}; ret "
                      f"{f3(retention[arm]['retention_primary'])})")
    xs = sorted(int(k) for k in w1_committed)
    ax.plot(xs, [w1_committed[str(s)] if str(s) in w1_committed
                 else w1_committed[s] for s in xs], "s--", ms=5, lw=1.4,
            color="seagreen", alpha=0.9,
            label=f"committed g1c W1 (the FREE control's record)")
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
    ax.set_title("THE FATE TRAJECTORIES — fate vs the engineered geometry",
                 fontsize=10)

    # (0,1) THE RETENTIONS vs the draw spread
    ax = axes[0, 1]
    names = list(ARMS)
    rets = [retention[a]["retention_primary"] for a in names]
    bars_ = ax.bar(names, [r if r is not None else 0.0 for r in rets],
                   color=[col[a] for a in names], alpha=0.85)
    for b, v, a in zip(bars_, rets, names):
        ax.annotate((f"{v:.4f}" if v is not None else "n/a")
                    + ("" if landed[a] else "\n(NOT landed)"),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhspan(DRAW_SPREAD["lo"], DRAW_SPREAD["hi"], color="#b8d8f0",
               alpha=0.35, zorder=0,
               label="the draw spread (2.74M family, n=5 roots, e225)")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    if rets[1] is not None:
        ax.axhline(1.5 * rets[1], color="#c0392b", ls="-.", lw=1.4,
                   label=f"the SEAT bar: 1.5x ALIGNED = {1.5 * rets[1]:.4f}")
    ax.set_ylabel("flat-phase retention (min g-12 {10,50,100,300} / root)")
    ax.legend(fontsize=7.4, loc="best")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"THE CLAIM'S STATISTIC — ORTHO/ALIGNED ratio "
                 f"{ratio if ratio is not None else float('nan'):.2f}x "
                 f"(bar >= 1.5x, both landed)", fontsize=10)

    # (1,0) THE MARGINS through the wash (e242's instrument)
    ax = axes[1, 0]
    for arm in ARMS:
        st = readouts[arm]["states"]
        keys = ["t0"] + [str(s) for s in sorted(washes[arm]["sds"])
                         if str(s) in st]
        xs = [st[k]["step"] for k in keys]
        ys = [st[k]["margin_median_sigma"] for k in keys]
        ax.plot(xs, ys, "s-", ms=6, lw=1.9, color=col[arm],
                label=f"ARM-{arm}")
    ax.set_xlabel("wall-wash step")
    ax.set_ylabel("battery MEDIAN argmax margin (sigma)")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25)
    ax.set_title("THE COMMITMENT LAYER — margins through the wash "
                 "(prediction (a): ORTHO thickens faster)", fontsize=10)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E246 VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=94, break_long_words=False)[:10]:
        ax.text(0.02, y, wd, fontsize=6.8, va="top", family="monospace")
        y -= 0.024
    y -= 0.012
    ax.text(0.02, y, "GATES: " + "  ".join(
        f"{g}={'PASS' if v.get('pass') else 'FAIL'}"
        for g, v in metrics["gates"].items() if isinstance(v, dict)),
        fontsize=6.6, va="top", family="monospace")
    fig.suptitle("E246 — THE ENGINEERED SEAT: install a fact ORTHOGONAL to "
                 "the wash-span / IN the span / FREE, then the standard "
                 f"wall wash (R={R_CLAIM}) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e246_engineered_seat.png", dpi=130)
    plt.close(fig)


def make_geometry_thermal_plot(rd, G_SHADOW, G_SPAN, pred_c, pred_a, pred_b,
                               readouts, arms_rec, thermal_log):
    """THE INSTRUMENT FIGURE: the G_SHADOW decay + the span spectrum + the
    per-arm measured geometry + the thermal reads."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    col = {"ORTHO": "#c0392b", "ALIGNED": "#8e44ad", "FREE": "#e67e22"}

    # (0,0) G_SHADOW: the sign-shadow decay + the late half
    ax = axes[0, 0]
    ks = sorted(int(k) for k in G_SHADOW["decay"])
    ax.plot(ks, [G_SHADOW["decay"][k] for k in ks], "o-", ms=6, lw=1.8,
            color="#1a6faf", label="cos(seg_k, -u0)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-G_SHADOW["shadow_bar"], G_SHADOW["shadow_bar"],
               color="#b8e0b8", alpha=0.35,
               label=f"|cos| < {G_SHADOW['shadow_bar']} (clean)")
    late_ks = [k for k in ks if k >= LATE_START]
    ax.axvspan(late_ks[0] - 0.5, ks[-1] + 0.5, color="dimgray", alpha=0.10,
               label=f"THE LATE HALF (s{LATE_START}..{ks[-1]})")
    ax.set_xlabel("history step k")
    ax.set_ylabel("cos(seg_k, -u0)")
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.25)
    ax.set_title(f"G_SHADOW — the sign-step's shadow vs the late half "
                 f"(max|cos| late {G_SHADOW['late_half_max_abs_cos']:.3f})",
                 fontsize=10)

    # (0,1) the span spectrum + the measured per-arm geometry
    ax = axes[0, 1]
    ax.semilogy(range(1, len(G_SPAN["sv"]) + 1), G_SPAN["sv"], "s-",
                ms=7, lw=1.8, color="#8e44ad",
                label=f"the LATE span (rank {G_SPAN['rank_eff']}, PR "
                      f"{G_SPAN['participation_ratio']:.2f})")
    ax.set_xlabel("basis index")
    ax.set_ylabel("singular value")
    ax.legend(fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    axb = ax.twinx()
    xs = np.arange(len(ARMS))
    w = 0.27
    axb.bar(xs - w, [pred_c[a]["cos_to_span_post_install"] for a in ARMS],
            width=w, color="#7fb3d5", alpha=0.9, label="post-install d")
    axb.bar(xs, [pred_c[a]["cos_to_span_root"] for a in ARMS], width=w,
            color="#1a6faf", alpha=0.9, label="final-root d")
    axb.bar(xs + w, [pred_c[a]["cos_to_span_wash_d300"] or 0.0
                     for a in ARMS], width=w, color="#c0392b", alpha=0.6,
            label="+300 wash d")
    for i, a in enumerate(ARMS):
        axb.annotate(f"ledg {pred_c[a]['ledger_in_span_frac_median']:.3f}",
                     (i, 1.02), ha="center", fontsize=6.4, color="dimgray")
    axb.set_xticks(xs)
    axb.set_xticklabels(ARMS, fontsize=8)
    axb.set_ylabel("||S^T d||/||d|| (measured cos-to-span)", color="#1a6faf")
    axb.set_ylim(0, 1.15)
    axb.tick_params(axis="y", colors="#1a6faf")
    axb.legend(fontsize=6.8, loc="upper right")
    ax.set_title("THE SPAN + THE MEASURED GEOMETRY (prediction (c): actual, "
                 "not nominal)", fontsize=10)

    # (1,0) THE THERMAL READS: T(t) per arm
    ax = axes[1, 0]
    for arm in ARMS:
        rows = readouts[arm]["thermal"]
        xs = [r["step"] for r in rows]
        ax.plot(xs, [r["T_mle"] for r in rows], "D-", ms=6, lw=1.8,
                color=col[arm], label=f"ARM-{arm} (own t0 family)")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("wall-wash step")
    ax.set_ylabel("fitted T (one-T MLE)")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25)
    t1 = pred_b["T_mle_plus1"]
    ax.set_title("THE THERMAL READS — prediction (b): ALIGNED's +1 heat "
                 "exceeds ORTHO's (T+1 "
                 + " ".join(f"{a}:{t1[a]:.3f}" if t1[a] is not None
                            else f"{a}:n/a" for a in ARMS) + ")", fontsize=9.5)

    # (1,1) the geometry ledger + the thermal envelope
    ax = axes[1, 1]
    for arm in ARMS:
        led = arms_rec[arm]["install"]["ledger"]
        xs = sorted(int(k) for k in led)
        get = lambda s: led[s] if s in led else led[str(s)]
        ax.plot(xs, [get(s)["in_span_frac"] for s in xs], "o-",
                ms=3.2, lw=1.3, color=col[arm], alpha=0.9,
                label=f"ARM-{arm} install |<S,g>|/||g||")
    ax.set_xlabel("install step")
    ax.set_ylabel("the install gradient's in-span fraction")
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.25)
    axb = ax.twinx()
    if thermal_log:
        axb.plot([r["t"] for r in thermal_log], [r["temp"] for r in thermal_log],
                 "-", lw=0.8, color="dimgray", alpha=0.7)
        axb.axhline(TEMP_EARLY_END, color="crimson", ls=":", lw=1.0)
        axb.axhline(TEMP_HARD, color="crimson", ls="--", lw=1.2)
        axb.set_ylabel("GPU temp (C) per-step polls", color="dimgray")
        axb.tick_params(axis="y", colors="dimgray")
    ax.set_title("THE GRADIENT GEOMETRY LEDGER + the thermal envelope "
                 "(bursts end at the 78C margin)", fontsize=9.5)

    fig.suptitle("E246 — the instrument page: G_SHADOW, the LATE span, the "
                 "measured per-arm geometry, the thermal reads", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "e246_geometry_thermal.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
