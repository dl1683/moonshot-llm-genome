"""X44 — R*-AT-THE-OFFSETS (R79's minted card; composes x38's flip with
x40's support-breadth; CPU desk lane, dispatched 2026-10-09/10, W055
owns the GPU lane). This docstring carries the registered question +
background + design + bars + lab lean + P-x44a VERBATIM from the
dispatch letter, plus every frozen operationalization, committed at
birth BEFORE any compute. Adjudicate against exactly this; no bar
shopping.

THE BACKGROUND (dispatch verbatim): "x38 found the flip (the complement
lifts empty slots, collapses living reads; the sign dead-or-alive below
the lowest living read — x42's refinement) at the BATTERY geometry. x40
found the varied arm's support tail at the OFFSETS (gm12 28/60 alive vs
the fixed arm's 2/60 — the variety's generalization channel as support
breadth). THE QUESTION: is the flip a SLOT property (the same R* at
every context) or a STATE-AT-CONTEXT property (offset contexts sit
nearer the flip — the variety channel re-read as flip-margin)?"

THE DESIGN (dispatch verbatim):
  "1. THE STATES: e341's committed post-anneal states (VARIED and
   FIXED — md5-bound) + the fresh install (the anchor).
  2. THE DISPLACEMENT: x38's exact K10K complement (bit-equal carrier)
   applied to each state.
  3. THE READ: per-CONTEXT flip behavior at the three geometries
   (g-12 / g0 / g+12 — the batteries' own context sets): for each
   context, the pre-displacement read and the post-displacement read;
   classify per context: LIFT (post > pre) / COLLAPSE (post << pre).
   Then the per-geometry flip statistics: at which read level does
   each geometry's response flip?
  4. THE ANALYSIS: R*-UNIFORM (the flip level is the same at
   g-12/g0/g+12 — a slot property) vs R*-TRACKS-THE-SUPPORT-TAIL (the
   offsets flip at LOWER read levels than g0 — the offset contexts are
   nearer the flip; the varied arm's surviving offsets are the ones
   whose local flip threshold sits below their local read)."

BARS (dispatch VERBATIM; frozen in this birth commit BEFORE compute):
  - R*-UNIFORM: "the three geometries' flip levels agree within ~2x —
    the flip is a slot property; the two-channel law's parameter is
    context-independent."
  - R*-TRACKS-THE-SUPPORT-TAIL: "g0 flips at a materially lower local
    read than the offsets (or vice versa — report the direction) — the
    flip is context-local; the support breadth IS flip-margin; the
    variety channel's gift is a wider margin distribution, not more
    contexts above a uniform threshold."

LAB LEAN (dispatch verbatim): "Lab lean: R*-TRACKS-THE-SUPPORT-TAIL,
weakly — x40's support picture showed the offsets' survival is
varied-arm-specific, suggesting local thresholds differ; the
countervailing: the read's addressing may be global enough to impose
one threshold. State your own read."

P-x44a (THE EXECUTOR'S OWN READ, registered per the dispatch's
"Register P-x44a BEFORE compute ... State your own read", frozen HERE
at birth; predictions are scored): R*-TRACKS-THE-SUPPORT-TAIL, weakly
— concurring with the lab lean, with one registered refinement about
WHAT tracks. (1) THE BRACKET-LEVEL SLOT PROPERTY WILL SURVIVE: the
complement is ONE fixed vector acting on ONE logit field; x38's flip
held across six states and x42's interior kept the sign monotone with
no saturation-up anywhere — I predict the nine cells' constraint sets
will be MUTUALLY CONSISTENT (a common R* interval exists under the
primary classification), i.e. no cell's LIFT-side lower bound crosses
another's COLLAPSE-side upper bound. (2) THE POINT ESTIMATES WILL
STILL DRIFT BY GEOMETRY, THE OFFSETS LOWER: the bracket localizes only
where the context distribution crosses R*, and the offsets' low-tail
contexts (weaker elicitation — the fixed arm's gm12 t0 mean 0.131 vs
g0 0.756) cross lower — so at least one comparably-covered state's
two-sided point estimates will differ by > 2x, g0's above the
offsets'. THE HONEST READING IF BOTH HOLD: the flip level is slot-like
at bracket resolution but the RESOLVABLE threshold tracks the support
tail — the variety channel's gift reads as a margin distribution,
exactly the TRACKS bar's wording. FALSIFIERS OF MY READ: (a) genuinely
DISJOINT brackets (a constraint contradiction, not drift) — context-
locality stronger than I predict: TRACKS honestly, refinement (1)
missed; (b) all point estimates within 2x with the comparison
existing — R*-UNIFORM: the lean and I both missed; (c) too few
two-sided cells to compare at all (the annealed states near-all-alive,
FRESH's low tail absent) — the honest word is the else-branch MIXED,
disclosed as under-determined, not adjudicated. SCORED: TRUE iff the
verdict == R*-TRACKS-THE-SUPPORT-TAIL.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ===

* THE STATES (three, md5-bound at birth; every one a committed
  artifact, never re-formed; read as PLAIN uncommitted organisms via
  G1.evl_load — the states x40 panel-read, the same convention):
  - VARIED := runs/e341/e341_anneal_VARIED_post.pt — e311's TAVIREN
    subject + the cons's e113 300-step VARIED-context annealing
    (post-anneal, UNCOMMITTED).
  - FIXED := runs/e341/e341_anneal_FIXED_post.pt — the critic's
    matched-step FIXED-context control (post-anneal, UNCOMMITTED).
  - FRESH := runs/checkpoints/e311_TAVINST_post.pt (the anchor) —
    the harness's own subject, loaded + read-gated by E38.phase_P1
    VERBATIM.
  Committed t0 reads re-asserted on CPU (tol 2e-6): VARIED g0
  0.7519720792770386 / gm12 0.6881482601165771; FIXED g0
  0.7557547688484192 / gm12 0.13110429048538208 (e341's
  annealed_t0, runtime-read); FRESH g0 0.285851389169693 / gm12
  0.1744154542684555 (the harness's own literals, runtime-read).

* THE DISPLACEMENT := x38's exact K10K complement, taken from x15's
  committed carrier runs/x15/x15_comp_xFULL.pt (md5-bound) — the SAME
  fp32 per-key delta x38 verified bit-equal to its rebuild; applied to
  each state as substrate_fp32 + delta_fp32 per key (x15's fp32
  injection, x38's apply_disp line). PROVENANCE GATE: the carrier's
  own model == e001 base + delta bit-equal on every key (x38's
  G_ONESTATE convention), AND base+delta read at the host-g0 battery
  reproduces x24's committed panel cells (11 names x {base, comp},
  tol 1e-12 — x38's G_X24REPRO convention at the host site) — my
  injection path certified bit-identical to x38/x24's end to end.
  DISCLOSED: the complement was constructed against the BASE
  lineage's K10K room; applied to the TAVINST-family states it is a
  fixed vector in a different landscape — x38's own precedent (six
  rungs, all substrates != the construction base).

* THE BATTERIES := the harness's phase_P0 splice bank (e261's
  construction, mix FLORIZEL 19 / ELIZABETH 41): g0 (60x130) and
  gm12 (60x118) from phase_P0, PLUS gp12 (60x142) rebuilt by
  phase_P0's own construction line at j=+12 (x40's G_GP12
  convention). The same 60 host positions underlie all three
  geometries (the offset axis is the context LENGTH, not the bank).

* THE READ := p(T) at the final context position (TAVIREN[0], the
  family's read channel), per context, battery_cell's exact batching
  (bs=30, softmax at the final position — x40's per-context reader,
  certified vs G1.battery_cell to 1e-9). PRE := the bare state's
  read; POST := the state+complement read. The Q and Z columns
  (QELVARO[0], ZEPHYRA[0] — never-written on this lineage, gated by
  phase_P0's G_NAMEFREE) are co-read per context as the never-written
  band.

* THE PER-CONTEXT CLASSIFICATION — PRIMARY FORM (x38's convention,
  made per-context; adjudicates):
  - band_i := max(|dQ_i|, |dZ_i|) (floored at 1e-9; floor firings
    counted + disclosed).
  - COLLAPSE_i iff |dT_i| > 5.0 x band_i (x38's "~5x band" bar;
    EITHER direction — big UP moves class by the magnitude clause
    with direction co-reported, x38's own note; SAT-UP counted
    separately for disclosure).
  - LIFT_i iff (not COLLAPSE_i) and POST_i > PRE_i (the dispatch's
    literal "LIFT (post > pre)"; x38's ratio clause).
  - else AMBIG_i.
  THE FLIP BRACKET per (state, geometry): L_max := max PRE over LIFT
  contexts; C_min := min PRE over COLLAPSE contexts (None if a class
  is empty); the cell is TWO-SIDED-CONSISTENT iff both exist and
  L_max < C_min (violations counted + disclosed; inconsistent cells
  are EXCLUDED from the adjudicating set); then R*(state,geo) is
  BRACKETED in (L_max, C_min] with point estimate
  R_hat := sqrt(L_max x C_min). One-sided cells contribute
  constraints only (has-LIFT-no-COLLAPSE => R* > L_max;
  has-COLLAPSE-no-LIFT => R* <= C_min).

* THE PER-CONTEXT CLASSIFICATION — SENSITIVITY FORM (parameter-free,
  the dispatch's literal "post << pre"; co-reported, never
  adjudicates): COLLAPSE_i iff POST_i <= 0.5 x PRE_i; LIFT_i iff
  POST_i > PRE_i; else AMBIG. Same bracket machinery; the two forms'
  verdicts compared + any disagreement disclosed in the verdict
  clause.

* THE VERDICT COMPOSITE (frozen; mechanical; adjudicates on the
  PRIMARY form; "within ~2x" := ratio <= 2.0; "materially" := ratio
  > 2.0):
  - THE COMPARISON SET: per state, its two-sided-consistent
    geometries. PRIMARY COMPARISON EXISTS iff some state covers g0
    AND at least one offset among its two-sided geometries; ELSE fall
    to the SECONDARY comparison (disclosed at birth): the pooled
    cross-state per-geometry brackets (per geometry: pooled
    (max L_max, min C_min] over its two-sided cells, if two-sided).
    If neither exists => the else-branch (MIXED, under-determined).
  - R*-TRACKS-THE-SUPPORT-TAIL fires iff (primary exists and some
    state's max/min point-estimate ratio across its two-sided
    geometries > 2.0) OR (secondary exists and some geometry pair's
    ratio > 2.0) — the direction reported (which geometry sits
    lower).
  - R*-UNIFORM fires iff (the comparison exists, primary or
    secondary) AND (every state's ratio <= 2.0 AND every secondary
    pair <= 2.0) AND (the nine cells' PRIMARY constraints are POOLED-
    CONSISTENT: max of all lower bounds < min of all upper bounds —
    a common R* interval exists).
  - TRACKS overrides UNIFORM (a single > 2x geometry separation
    kills the slot property). Else => MIXED (the table verbatim; no
    wording change without a new registered cell).

* THE RIDERS (descriptive, never bars): (a) THE MARGIN MAP — per
  (VARIED, offset) cell: P(post >= 0.05 | pre > R_hat) vs P(post >=
  0.05 | pre <= R_hat) — the dispatch's "the varied arm's surviving
  offsets are the ones whose local flip threshold sits below their
  local read", made countable; (b) noise-zone lifts (LIFT contexts
  with pre < 1e-4) counted; (c) the paired-context elicitation
  correlation (per state, Pearson r of the 60 pre-reads across
  geometry pairs — the same host positions, the context axis's own
  structure); (d) ce_r per state x {bare, comp}; (e) per-geometry
  mean pre/post + alive fractions (the x40 support table's t0
  counterpart); (f) the per-bin response curve (deciles of log-pre,
  mean signed log-ratio) for the figure.

* THE REFERENCE REPRODUCTIONS (a failure HALTS => TEXTURE): the six
  committed t0 means + three ce_r (G_T0); the per-context reader vs
  battery_cell (G_PERCTX, 9 cells, 1e-9); the carrier's model ==
  base + delta bit-equal + x24's 22 host-g0 panel cells bit-exact
  (G_CARRIER); x38's committed R* bracket + x42's committed window
  runtime-read from the md5-bound records for the report's reference
  framing (G_REF).

* GATES (a failure HALTS => TEXTURE, nothing adjudicated): G_MD5 (19
  binds: the three states + the carrier + the x15/x24/x38/x40/e341/
  e338/e335 metrics + the six rig parents + the x40 rig (the net0
  class provenance) + the e001 base), G_QUOTES (every quoted source
  line a verbatim substring), G_P0P1 (the harness phases' own
  internal gates), G_GP12 (the +12 battery rebuild), G_KEYSET (the
  canonical key order across the three states AND the carrier delta
  AND the base parameters; N = 2,739,072), G_T0, G_PERCTX, G_CARRIER
  (+ G_X24REPRO inside it), G_REF, G_NET0 (every state classed; the
  class strings' building blocks verbatim substrings of the
  md5-bound parents), G_FLIPCLASS (the classification machinery's
  internal consistency: counts sum to 60; brackets well-formed).

COMPUTE ENVELOPE: CPU ONLY (dispatch: W055 owns the GPU lane) —
CUDA_VISIBLE_DEVICES="" BEFORE torch import (x38's bulletproof line),
torch threads 4, no envelope-log writes, no GPU code path, no other
runs/ touched (write ONLY runs/x44/*, runs/x44_smoke/ (gitignored));
~24 battery passes + 6 ce_r evals on the 2.74M organism — minutes.
TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/x44/{metrics.json (PROGRESSIVE), REPORT.md,
x44_rstar_offsets.png} — the per-geometry flip curves (the local read
vs the local response, the three geometries overlaid, per state) +
the R* bracket map (the nine cells' brackets vs x38/x42's committed
reference windows). NO .pt artifacts. No NOTES/THINKING/QUEUE/STATE
edits (dispatch; the heartbeat folds). Birth commit BEFORE compute,
pushed; smoke pass disclosed; final commit AND push.

DISCLOSED OMISSIONS: the neutral-site panel (x38's site axis) is NOT
run — the offsets ARE this cell's site axis; x38's committed neutral
record stands. The gaussian rider is NOT run — the displacement axis
is x38's committed record (GAUSS-BREAKS-AT-4x); no new dose question
is asked here.

Smoke (X44_SMOKE=1): the FULL gate path + all reads + the
classification machinery LIVE (nothing to shrink — the whole cell is
minutes); own gitignored dir runs/x44_smoke/; NOTHING adjudicated
(SMOKE stamp on every read; no figure, no report).

Run:  python lab/x44_rstar_offsets.py    (X44_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-ONLY, bulletproof

import hashlib                                     # noqa: E402
import json                                        # noqa: E402
import math                                        # noqa: E402
import subprocess                                  # noqa: E402
import sys                                         # noqa: E402
import time                                        # noqa: E402
from datetime import datetime, timezone            # noqa: E402
from pathlib import Path                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                             # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                # noqa: BLE001
    pass

import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402

import common                                     # noqa: E402
from common import run_dir, save_json, set_seed   # noqa: E402

import e338_commit_consolidator as E38            # noqa: E402 — THE
                                                  # HARNESS (by import;
                                                  # disclosed side
                                                  # effects: opens
                                                  # runs/e338/run.log in
                                                  # append — zero bytes
                                                  # written by this cell;
                                                  # e261's import-time
                                                  # cuda assert passes
                                                  # vacuously with
                                                  # device_count 0 on
                                                  # this box, verified
                                                  # pre-birth; no GPU
                                                  # path is ever taken)

torch.set_num_threads(4)          # CPU-only cell; x37/x38/x40's setting

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

G1 = E38.G1                                       # the era's own module
GB = E38.GB                                       # the 2.74M family rebind

SMOKE = os.environ.get("X44_SMOKE") == "1"
NAME = "x44_smoke" if SMOKE else "x44"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE REBIND SET (e341/x40's exact set, frozen; the leg convention) ----
thermal_log: list[dict] = []
device_events: list[dict] = []
E38.log = log
E38.RD = RD
E38.NAME = NAME
E38.T0 = T0
E38.E261.log = log
E38.E261.NAME = NAME
E38.E261.SMOKE = SMOKE
E38.E261.T0 = T0
E38.E261.thermal_log = thermal_log
E38.E261.device_events = device_events
E38.E261.DCT_WORKERS = 4           # e307/e311's desk convention

REPO = common.REPO

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
READ_BAR = 0.05                    # the family's frozen 0.05 aliveness bar
READ_TOL = 2e-6                    # the family's cross-session read law
CE_TOL = 5e-3                      # e338's CE_R convention
REPRO_TOL = 1e-12                  # x38's "bit-exact" panel convention
PERCTX_TOL = 1e-9                  # x40's G_PERCTX convention
BAND_X = 5.0                       # x38's "~5x band" collapse bar
BAND_FLOOR = 1e-9                  # the band floor (counted if it fires)
COLLAPSE_RATIO = 0.5               # the sensitivity form's halving bar
UNIFORM_X = 2.0                    # "within ~2x" := <= 2.0; "> 2.0" := material
NOISE_ZONE = 1e-4                  # the noise-zone lift rider's line
GLOBAL_SEED = 44001                # the design's ONLY new number (init only)

STATES = ("VARIED", "FIXED", "FRESH")
GEOS = ("g-12", "g0", "g+12")

# ---- the md5 binds (frozen at birth; verified on disk pre-birth) ----------
MD5_BINDS = {                      # (rel path, md5, size)
    "varied_state": ("runs/e341/e341_anneal_VARIED_post.pt",
                     "8a36b82ee61223674e8550499d0cfee6", 10977104),
    "fixed_state": ("runs/e341/e341_anneal_FIXED_post.pt",
                    "365056e844cc4b7e4c1b8f330596b599", 10977033),
    "subject": ("runs/checkpoints/e311_TAVINST_post.pt",
                "9886b25242c90dd669c6363c39a2a3bf", 10976550),
    "carrier": ("runs/x15/x15_comp_xFULL.pt",
                "b0a1785c6f759fdd1d8deeae7fee4a9b", 21951155),
    "base_ckpt": ("runs/checkpoints/e001.pt",
                  "d114536d1c0983ab3be67f67ff0667c8", 10974347),
    "x15_metrics": ("runs/x15/metrics.json",
                    "c90dd9371a5a7e248fdbd92c06bb243a", 18275),
    "x24_metrics": ("runs/x24/metrics.json",
                    "7f1a91cc1d42a202c8fb23c4f8fb3e7e", 61589),
    "x38_metrics": ("runs/x38/metrics.json",
                    "6b073d97f09aad1d09f4fb39fd45aa5d", 201351),
    "x40_metrics": ("runs/x40/metrics.json",
                    "e4b62f6f591e0ce0d6621bf49bf131d1", 85972),
    "e341_metrics": ("runs/e341/metrics.json",
                     "279bd235e43da13dc78998fe9da7b63d", 50070),
    "e338_metrics": ("runs/e338/metrics.json",
                     "0fe1cc6c72a51741621e334530703d70", 32454),
    "e335_metrics": ("runs/e335/metrics.json",
                     "5d92359bc9b2e943c01bf521487cf177", 44042),
    "harness": ("lab/e338_commit_consolidator.py",
                "5ddc8754cb358c09278ddd926059d050", 76039),
    "g1_rig": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01", 105063),
    "g1b_rig": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a", 70679),
    "e043_rig": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa", 67163),
    "e341_rig": ("lab/e341_varied_annealing.py",
                 "18039ab39d1b04d1adef2a22b9b3a350", 93004),
    "common_rig": ("lab/common.py",
                   "bb2a9ad6fc7c21c06d520a3ef9feffa7", 15934),
    "x40_rig": ("lab/x40_first_step_mechanism.py",
                "d1f8874a06c43b828eb8de334b20b896", 88906),
}

# ---- the quoted source lines (substring-gated; G_QUOTES) ------------------
QUOTES = [
    ("lab/e338_commit_consolidator.py",
     "cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]",
     "the battery contexts' construction line (j in GEOS; the gp12 "
     "rebuild rides it)"),
    ("lab/e338_commit_consolidator.py",
     'tid = stoi["T"]',
     "the read channel (TAVIREN[0], e311's)"),
    ("lab/g1_anchored_ball.py",
     "pr = F.softmax(lg[:, -1], -1)",
     "the battery read's softmax at the final position"),
    ("lab/x40_first_step_mechanism.py",
     "x15's fp32 per-key injection: substrate fp32 + delta fp32",
     "the injection convention's own words"),
    ("lab/x38_flip_threshold.py",
     "return {k: substrate_sds[sub_key][k] + displ[disp_key][k]",
     "x38's apply_disp (the complement's application line)"),
    ("lab/x40_first_step_mechanism.py",
     '"VARIED": ("BASE-formed (e311\'s TAVIREN install) + the cons\'s e113 "',
     "the VARIED class's building block (the net0 provenance)"),
    ("lab/x40_first_step_mechanism.py",
     '"FIXED": ("BASE-formed (e311\'s TAVIREN install) + the FIXED-context "',
     "the FIXED class's building block"),
    ("lab/x40_first_step_mechanism.py",
     '"FRESH": ("BASE-formed (e311\'s TAVIREN install; K10K-room-projected "',
     "the FRESH class's building block"),
    ("lab/e338_commit_consolidator.py",
     'NET0_CLASS = "BASE-formed (e311\'s TAVIREN install; K10K-room-projected "',
     "the harness's own subject class (the anchor's provenance)"),
]

# ---- the net0 classes (the standing rule: every state classed) ------------
NET0_CLASSES = {
    "VARIED": ("BASE-formed (e311's TAVIREN install) + the cons's e113 "
               "VARIED-context annealing (300 steps, seed 10901) — e341's "
               "VARIED POST-ANNEAL state, UNCOMMITTED (this cell reads the "
               "annealed organism itself, not the wash arm)"),
    "FIXED": ("BASE-formed (e311's TAVIREN install) + the FIXED-context "
              "control annealing (matched steps/seed; the j=0 pool) — "
              "e341's FIXED POST-ANNEAL state, UNCOMMITTED"),
    "FRESH": ("BASE-formed (e311's TAVIREN install; K10K-room-projected "
              "write, pruned/localized class) — the anchor subject (the "
              "harness's own, phase_P1-loaded); UNCOMMITTED"),
}

# ---- x24's frozen panel (the reproduction gate's 11 names) ----------------
X24_PANEL = ["ZEPHYRA", "TAVIREN", "QELVARO", "BUVONDI", "NYSTORA",
             "VIRETAN", "MAMILLIUS", "LEONTES", "KING", "ELIZABETH",
             "FLORIZEL"]

# ---- the committed reference numbers (runtime-read, never retyped) --------
X38_BRACKET_KEYS = ("verdict", "ladder_split", "R_star_bracket")
X42_WINDOW_KEYS = ("verdict", "ladder_split", "R_star_window")


REGISTERED = {
    "background_verbatim": (
        "x38 found the flip (the complement lifts empty slots, collapses "
        "living reads; the sign dead-or-alive below the lowest living "
        "read — x42's refinement) at the BATTERY geometry. x40 found the "
        "varied arm's support tail at the OFFSETS (gm12 28/60 alive vs the "
        "fixed arm's 2/60 — the variety's generalization channel as "
        "support breadth). THE QUESTION: is the flip a SLOT property (the "
        "same R* at every context) or a STATE-AT-CONTEXT property (offset "
        "contexts sit nearer the flip — the variety channel re-read as "
        "flip-margin)?"),
    "question_verbatim": (
        "is the flip a SLOT property (the same R* at every context) or a "
        "STATE-AT-CONTEXT property (offset contexts sit nearer the flip — "
        "the variety channel re-read as flip-margin)?"),
    "design_verbatim": {
        "1_THE_STATES": ("e341's committed post-anneal states (VARIED and "
                         "FIXED — md5-bound) + the fresh install (the "
                         "anchor)."),
        "2_THE_DISPLACEMENT": ("x38's exact K10K complement (bit-equal "
                               "carrier) applied to each state."),
        "3_THE_READ": ("per-CONTEXT flip behavior at the three geometries "
                       "(g-12 / g0 / g+12 — the batteries' own context "
                       "sets): for each context, the pre-displacement read "
                       "and the post-displacement read; classify per "
                       "context: LIFT (post > pre) / COLLAPSE (post << "
                       "pre). Then the per-geometry flip statistics: at "
                       "which read level does each geometry's response "
                       "flip?"),
        "4_THE_ANALYSIS": ("R*-UNIFORM (the flip level is the same at "
                           "g-12/g0/g+12 — a slot property) vs R*-TRACKS-"
                           "THE-SUPPORT-TAIL (the offsets flip at LOWER "
                           "read levels than g0 — the offset contexts are "
                           "nearer the flip; the varied arm's surviving "
                           "offsets are the ones whose local flip "
                           "threshold sits below their local read)."),
    },
    "bars_verbatim": {
        "R*-UNIFORM": ("the three geometries' flip levels agree within "
                       "~2x — the flip is a slot property; the two-channel "
                       "law's parameter is context-independent."),
        "R*-TRACKS-THE-SUPPORT-TAIL": (
            "g0 flips at a materially lower local read than the offsets "
            "(or vice versa — report the direction) — the flip is "
            "context-local; the support breadth IS flip-margin; the "
            "variety channel's gift is a wider margin distribution, not "
            "more contexts above a uniform threshold."),
    },
    "lab_lean_verbatim": (
        "Lab lean: R*-TRACKS-THE-SUPPORT-TAIL, weakly — x40's support "
        "picture showed the offsets' survival is varied-arm-specific, "
        "suggesting local thresholds differ; the countervailing: the "
        "read's addressing may be global enough to impose one threshold. "
        "State your own read."),
    "P_x44a": {
        "my_guess": ("R*-TRACKS-THE-SUPPORT-TAIL, weakly (concurring with "
                     "the lab lean)"),
        "registered": (
            "ONE REGISTERED REFINEMENT ABOUT WHAT TRACKS. (1) THE "
            "BRACKET-LEVEL SLOT PROPERTY WILL SURVIVE: the complement is "
            "ONE fixed vector acting on ONE logit field; x38's flip held "
            "across six states and x42's interior kept the sign monotone "
            "with no saturation-up anywhere — I predict the nine cells' "
            "constraint sets will be MUTUALLY CONSISTENT (a common R* "
            "interval exists under the primary classification). (2) THE "
            "POINT ESTIMATES WILL STILL DRIFT BY GEOMETRY, THE OFFSETS "
            "LOWER: the bracket localizes only where the context "
            "distribution crosses R*, and the offsets' low-tail contexts "
            "(weaker elicitation — the fixed arm's gm12 t0 mean 0.131 vs "
            "g0 0.756) cross lower — so at least one comparably-covered "
            "state's two-sided point estimates will differ by > 2x, g0's "
            "above the offsets'. THE HONEST READING IF BOTH HOLD: the "
            "flip level is slot-like at bracket resolution but the "
            "RESOLVABLE threshold tracks the support tail — the variety "
            "channel's gift reads as a margin distribution, exactly the "
            "TRACKS bar's wording. FALSIFIERS: (a) genuinely DISJOINT "
            "brackets (a constraint contradiction, not drift) — context-"
            "locality stronger than I predict: TRACKS honestly, "
            "refinement (1) missed; (b) all point estimates within 2x "
            "with the comparison existing — R*-UNIFORM: the lean and I "
            "both missed; (c) too few two-sided cells to compare at all — "
            "the honest word is the else-branch MIXED, disclosed as "
            "under-determined, not adjudicated."),
        "scored": "TRUE iff the verdict == R*-TRACKS-THE-SUPPORT-TAIL",
    },
    "clauses_fixed": {
        "read": "p(T) at the final context position, per context, "
                "battery_cell's exact batching (bs=30; x40's per-context "
                "reader)",
        "pre_post": "PRE := the bare state's read; POST := the "
                    "state+complement read (x15's fp32 per-key injection)",
        "primary_classification": "band_i := max(|dQ_i|, |dZ_i|) floored "
                                  "1e-9 (the per-context never-written "
                                  "envelope; Q/Z never written on this "
                                  "lineage); COLLAPSE_i iff |dT_i| > "
                                  "5.0 x band_i (either direction, x38's "
                                  "magnitude clause; SAT-UP counted "
                                  "separately); LIFT_i iff not COLLAPSE "
                                  "and POST > PRE; else AMBIG",
        "sensitivity_classification": "COLLAPSE_i iff POST <= 0.5 x PRE; "
                                      "LIFT_i iff POST > PRE; else AMBIG "
                                      "(parameter-free; co-reported, "
                                      "never adjudicates)",
        "bracket": "L_max := max PRE over LIFT; C_min := min PRE over "
                   "COLLAPSE; two-sided-consistent iff both exist and "
                   "L_max < C_min (violations counted; inconsistent cells "
                   "excluded from the adjudicating set); R_hat := "
                   "sqrt(L_max x C_min); R* in (L_max, C_min]",
        "one_sided": "has-LIFT-no-COLLAPSE => R* > L_max; "
                     "has-COLLAPSE-no-LIFT => R* <= C_min",
        "uniform_x": "within ~2x := ratio <= 2.0; materially := ratio > "
                     "2.0",
        "comparison_set": "primary: per-state two-sided geometries, a "
                          "state compares iff it covers g0 AND an offset; "
                          "secondary (if no state qualifies, disclosed): "
                          "pooled cross-state per-geometry brackets; "
                          "neither exists => MIXED under-determined",
        "verdict": "TRACKS iff some compared geometry ratio > 2.0 "
                   "(direction reported); UNIFORM iff comparison exists "
                   "AND all ratios <= 2.0 AND the nine primary "
                   "constraints pooled-consistent (max lower < min "
                   "upper); TRACKS overrides UNIFORM; else MIXED (table "
                   "verbatim)",
        "margin_rider": "per (VARIED, offset): P(post >= 0.05 | pre > "
                        "R_hat) vs P(post >= 0.05 | pre <= R_hat) — the "
                        "survivors-above-threshold claim, countable",
    },
    "registration": ("background + question + design + bars + lab lean "
                     "+ P-x44a VERBATIM from the dispatch letter; every "
                     "convention picked + frozen HERE at birth BEFORE "
                     "compute; this script committed at birth; adjudicate "
                     "against exactly this; no bar shopping."),
}

DEVIATIONS = [
    "CPU-ONLY CELL (dispatch: W055 owns the GPU lane): CUDA_VISIBLE_DEVICES"
    "='' before torch import, torch threads 4, no envelope-log writes, no "
    "GPU code path. The e338 harness is imported (phase_P0/phase_P1 "
    "verbatim); its GPU branches are never taken.",
    "THE DISPLACEMENT IS THE CARRIER'S OWN DELTA (x15's committed "
    "x15_comp_xFULL.pt), not a fresh rebuild: x38 already certified the "
    "carrier bit-equal to the rebuild; this cell re-certifies the "
    "injection path end-to-end (carrier model == base + delta bit-equal + "
    "x24's 22 committed host-g0 panel cells bit-exact through my own "
    "reader).",
    "THE COMPLEMENT'S PROVENANCE: constructed against the BASE lineage's "
    "K10K room; applied to the TAVINST-family states it is a fixed vector "
    "in a different landscape — x38's own precedent (its ladder applied "
    "the same delta to six substrates, none the construction base "
    "either).",
    "THE STATES ARE READ AS PLAIN UNCOMMITTED ORGANISMS (G1.evl_load) — "
    "the post-anneal states themselves, not the wash arms; the commit "
    "machinery is NOT fired here (x40 owns the s1 event; this cell owns "
    "the flip response).",
    "THE BAND IS PER-CONTEXT (x38's band was the cohort's mean move at "
    "rung level; here band_i := max(|dQ_i|, |dZ_i|), the per-context "
    "never-written envelope — noisier by construction, hence the "
    "parameter-free sensitivity form co-reported and the two forms' "
    "verdict disagreement disclosed in the clause).",
    "Z IS A BAND MEMBER ON THIS LINEAGE (never written here; on the base "
    "family Z was the ANCHOR slot — the complement's own Z-lift makes the "
    "band conservative there; each column recorded separately in the "
    "metrics).",
    "THE NEUTRAL SITE + THE GAUSSIAN RIDER ARE NOT RUN (disclosed "
    "omissions): the offsets are this cell's site axis; x38's committed "
    "neutral/gauss record stands.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).",
    "Smoke (X44_SMOKE=1): the full gate path + all reads + the "
    "classification machinery live (the whole cell is minutes; nothing "
    "shrunk); own gitignored dir; NOTHING adjudicated.",
    "IMPORT SIDE EFFECTS (inherited, x40's disclosure): importing the "
    "harness opens runs/e338/run.log in append — ZERO bytes written to it "
    "by this cell.",
    "n=1 per cell (one complement, one session, 60 committed contexts per "
    "geometry — the g-series standing lottery note); the bit-exact x24 "
    "reproduction doubles as the in-cell instrument replicate.",
]

metrics: dict = {
    "experiment": "x44_rstar_offsets",
    "phase": ("R*-AT-THE-OFFSETS — is x38's flip a SLOT property (the same "
              "R* at every context) or a STATE-AT-CONTEXT property (the "
              "offset contexts sit nearer the flip)? The per-context flip "
              "response of e341's three committed states (varied/fixed "
              "post-anneal + the fresh anchor) at the three battery "
              "geometries (g-12/g0/g+12) under ONE fixed K10K complement "
              "(x38's exact carrier); the per-geometry flip levels "
              "compared; x40's support tail re-read as flip-margin"),
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "cpu_only": True,
    "threads": {"torch": 4},
    "envelope": {
        "device": ("CPU ONLY (CUDA_VISIBLE_DEVICES=''; W055 owns the GPU "
                   "lane; no envelope-log writes)"),
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": DEVIATIONS,
    "builds_on": [
        "x38 (THE FLIP THRESHOLD: the K10K complement's sign flip — lifts "
        "empty, collapses living; R* bracketed (1.34e-5, 0.5302] at the "
        "battery geometry; its exact carrier delta is this cell's "
        "displacement; its 5x-band + living-read conventions the "
        "per-context classifier's parents)",
        "x42 (THE R* INTERIOR: the sign settles DOWN by read 0.114, the "
        "window (0.0217, 0.5052]; dead-or-alive in sign, graded in "
        "magnitude — the refinement this cell carries)",
        "x40 (THE SUPPORT TAIL: the varied arm's offset survival 28/60 vs "
        "2/60 at gm12 — the generalization channel this cell re-reads as "
        "flip-margin; its per-context reader + gp12 rebuild + state "
        "loading conventions inherited verbatim)",
        "e341 (THE STATES: its committed post-anneal artifacts + "
        "annealed_t0 literals; AMOUNT-NOT-TYPE — the annealing is a dose, "
        "the states this cell dissects)",
        "e338 (THE HARNESS: phase_P0/phase_P1 by import — the splice bank, "
        "the batteries' construction line, the subject gates)",
        "e311 (THE ANCHOR: the fresh TAVIREN install subject; its "
        "committed t0 literals the FRESH gates)",
        "x15 (THE CARRIER: the K10K complement's construction + the fp32 "
        "per-key injection convention)",
        "x24 (THE INSTRUMENT CERTIFICATION: the 11-name panel whose "
        "committed host-g0 cells certify this cell's injection path "
        "bit-exact)",
        "R79 (the review that minted this card — the composition cell)",
    ],
    "whats_new": [
        "THE PER-CONTEXT FLIP MAP — x38's flip was read at battery MEANS "
        "across states; this cell reads it PER CONTEXT (60 x 3 geometries "
        "x 3 states x pre/post), making the flip level estimable WITHIN a "
        "state at each geometry — the slot-vs-context question's first "
        "measurement",
        "THE FLIP BRACKET PER GEOMETRY — the never-written band made "
        "per-context (the Q/Z envelope) so x38's 5x-band classifier "
        "descends from rungs to contexts, with a parameter-free "
        "sensitivity form co-reported",
        "THE COMPOSITION — x38's R* and x40's support tail in one "
        "instrument: whether the varied arm's surviving offsets are the "
        "contexts whose local read sits above the local threshold (the "
        "margin-map rider)",
        "THE LINEAGE TRANSPLANT OF THE FLIP INSTRUMENT — the complement's "
        "carrier applied outside its birth family (the TAVINST lineage), "
        "the second family to see the flip question at all",
    ],
    "gates": {},
}
E38.metrics = metrics                     # the harness's phases write here


# ------------------------------------------------------------------ helpers
def md5of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- THE PER-CONTEXT READER (battery_cell's exact batching) ---------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, cids: list[int],
                    bs: int = 30) -> torch.Tensor:
    """battery_cell's exact instrument (x40's) returning the FULL
    per-context probability matrix for the wanted columns."""
    net.eval()
    rows = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        rows.append(pr[:, cids])
    return torch.cat(rows)                    # [60, len(cids)]


def load_state(rel: str) -> dict:
    art = torch.load(REPO / rel, map_location="cpu", weights_only=False)
    return {k: v.detach().clone() for k, v in art["model"].items()}


def apply_delta(theta: dict, delta: dict) -> dict:
    """x15's fp32 injection (x38's apply_disp line): substrate fp32 +
    delta fp32 per key."""
    return {k: theta[k] + delta[k] for k in theta}


def texture(halt: str) -> SystemExit:
    metrics["verdict"] = {"word": "TEXTURE",
                          "why": f"{halt} — nothing adjudicated"}
    write_partial(f"HALT: TEXTURE ({halt})")
    return SystemExit(f"{halt} — nothing adjudicated")


# ======================================================================
# P1 — THE BIND GATES (md5s + quotes + the runtime-read references)
# ======================================================================
def phase_gates() -> dict:
    log("P1 — THE BIND GATES (19 md5 binds + the quoted source lines)")
    binds = {}
    for key, (rel, md5, size) in MD5_BINDS.items():
        p = REPO / rel
        mine = md5of(p)
        ok = (mine == md5) and p.stat().st_size == size
        binds[key] = {"path": rel, "md5": mine, "bound_md5": md5,
                      "size": p.stat().st_size, "bound_size": size,
                      "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": "every examined state, the displacement carrier, "
                      "every reference record and every rig parent "
                      "md5-bound at run time",
             "pass": bool(all(b["pass"] for b in binds.values()))}
    if not g_md5["pass"]:
        raise texture(f"G_MD5 FAILED: {g_md5}")
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)}/{len(binds)} binds")

    quotes = []
    for src, frag, why in QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": "every ported line is a verbatim substring of its "
                    "committed source rig (the battery construction, the "
                    "read channel, the injection, x38's application line, "
                    "the net0 class building blocks)",
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    if not g_q["pass"]:
        raise texture(f"G_QUOTES FAILED: {g_q}")
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim quotes")

    # G_REF: the committed reference windows, runtime-read (never retyped)
    x38m = json.loads((REPO / "runs" / "x38" / "metrics.json")
                      .read_text(encoding="utf-8"))
    x40m = json.loads((REPO / "runs" / "x40" / "metrics.json")
                      .read_text(encoding="utf-8"))
    x42p = REPO / "runs" / "x42" / "metrics.json"
    x38_bracket = x38m[X38_BRACKET_KEYS[0]][X38_BRACKET_KEYS[1]][
        X38_BRACKET_KEYS[2]]
    x42_window = None
    if x42p.exists():
        x42m = json.loads(x42p.read_text(encoding="utf-8"))
        x42_window = x42m[X42_WINDOW_KEYS[0]][X42_WINDOW_KEYS[1]][
            X42_WINDOW_KEYS[2]]
    refs = {
        "x38_R_star_bracket_host_g0": list(x38_bracket),
        "x38_verdict_word": x38m["verdict"]["word"],
        "x42_R_star_window_host_g0": (list(x42_window)
                                      if x42_window else None),
        "x40_support_alive_s1": {
            a: x40m["perctx_s1"][a]["alive"] for a in x40m["perctx_s1"]},
        "e341_annealed_t0": json.loads(
            (REPO / "runs" / "e341" / "metrics.json")
            .read_text(encoding="utf-8"))["annealed_t0"],
    }
    g_ref = {"refs": refs,
             "claim": "x38's committed R* bracket + verdict, x42's "
                      "committed interior window, x40's committed s1 "
                      "support table and e341's committed t0 literals — "
                      "runtime-read from the md5-bound records, never "
                      "retyped",
             "pass": bool(x38_bracket[0] < x38_bracket[1]
                          and x38m["verdict"]["word"] == "SHARP-FLIP"
                          and refs["x40_support_alive_s1"]["VARIED"]["g-12"]
                          > refs["x40_support_alive_s1"]["FIXED"]["g-12"]
                          and refs["e341_annealed_t0"]["VARIED"]["g0"]
                          > 0.5)}
    if not g_ref["pass"]:
        raise texture(f"G_REF FAILED: {g_ref}")
    metrics["gates"]["G_REF"] = g_ref
    metrics["references"] = refs
    log(f"G_REF PASS: x38 R* {x38_bracket[0]:.3e}..{x38_bracket[1]:.4f}; "
        f"x42 window "
        + (f"{x42_window[0]:.3e}..{x42_window[1]:.4f}"
           if x42_window else "absent")
        + f"; x40 s1 alive gm12 varied "
          f"{refs['x40_support_alive_s1']['VARIED']['g-12']:.3f} vs fixed "
          f"{refs['x40_support_alive_s1']['FIXED']['g-12']:.3f}")
    write_partial("P1 the bind gates passed")
    return {"x38m": x38m, "x40m": x40m, "refs": refs}


# ======================================================================
# P2 — THE HARNESS'S OWN PHASES + THE gp12 REBUILD
# ======================================================================
def phase_harness() -> dict:
    log("P2 — THE HARNESS'S OWN PHASES (phase_P0 + phase_P1, verbatim)")
    p0 = E38.phase_P0()                     # the protocol rebuild (CPU)
    p1 = E38.phase_P1(p0)                   # the subject (e311's, CPU)
    g = {"harness_md5": MD5_BINDS["harness"][1],
         "claim": "E38.phase_P0 + E38.phase_P1 executed VERBATIM on CPU "
                  "(their internal gates G_NAMEFREE/G_SPLICE/G_BATTERY/"
                  "G_BANK/G_SUBJECT all asserted inside the harness)",
         "pass": True}
    metrics["gates"]["G_P0P1"] = g
    # the gp12 battery (phase_P0's own construction line, j=+12; x40's
    # G_GP12 convention verbatim)
    train_text = p0["train_text"]
    install_occ = p0["install_occ"]
    cs = [train_text[p - G1.PRE - 12: p] for p, _ in install_occ]
    gp12_ids = torch.stack([p0["corpus"].encode(c) for c in cs])
    g12 = {"shape": list(gp12_ids.shape),
           "expected": [60, G1.PRE + 12],
           "construction": "cs = [train_text[p - G1.PRE - j: p] for p, _ "
                           "in install_occ] at j=+12 (phase_P0's own "
                           "line, quoted + gated)",
           "claim": "the g+12 offset battery rebuilt by the harness's "
                    "own construction",
           "pass": bool(list(gp12_ids.shape) == [60, G1.PRE + 12])}
    if not g12["pass"]:
        raise texture(f"G_GP12 FAILED: {g12}")
    metrics["gates"]["G_GP12"] = g12
    log(f"G_GP12 PASS: the g+12 battery {gp12_ids.shape}")
    batteries = {"g-12": p0["gm12_ids"], "g0": p0["g0_ids"],
                 "g+12": gp12_ids}
    return {"p0": p0, "p1": p1, "batteries": batteries}


# ======================================================================
# P3 — THE DISPLACEMENT (the carrier, certified end-to-end)
# ======================================================================
def phase_carrier(hp: dict) -> dict:
    log("P3 — THE DISPLACEMENT (x38's exact complement, from x15's "
        "committed carrier; the end-to-end certification)")
    p0 = hp["p0"]
    ck = torch.load(REPO / MD5_BINDS["carrier"][0], map_location="cpu",
                    weights_only=False)
    delta = {k: v.detach().clone() for k, v in ck["delta"].items()}
    carrier_model = {k: v.detach().clone() for k, v in ck["model"].items()}
    del ck

    base_net = G1.load_g1(REPO / MD5_BINDS["base_ckpt"][0])
    base_params = {k: v.detach().clone()
                   for k, v in base_net.named_parameters()}
    base_keys = list(base_params.keys())
    del base_net

    # G_KEYSET (part 1): the carrier's key order == the base parameters'
    key_ok = (list(delta.keys()) == base_keys
              and list(carrier_model.keys()) == base_keys)
    n_params = sum(v.numel() for v in base_params.values())
    g_keys = {"carrier_keys_eq_base_params": bool(key_ok),
              "n_params": n_params,
              "n_params_expected": GB.G1B_PARAMS,
              "claim": "the flat space is the base's parameter key order; "
                       "the carrier rides it exactly",
              "pass": bool(key_ok and n_params == GB.G1B_PARAMS)}

    # the carrier's model == base + delta, bit-equal (x38's G_ONESTATE)
    model_ok = all(torch.equal(carrier_model[k],
                               base_params[k] + delta[k])
                   for k in base_keys)
    g_carrier = {
        "form": "the carrier's own model == e001 + delta BIT-EQUAL on "
                "every key (x38's G_ONESTATE convention) AND x24's "
                "committed host-g0 panel cells reproduce BIT-EXACT "
                "(tol 1e-12) through THIS cell's own reader + injection "
                "path — my path certified identical to x38/x24's",
        "carrier_model_eq_base_plus_delta": bool(model_ok),
        "delta_l2": float(torch.norm(torch.cat(
            [delta[k].reshape(-1) for k in base_keys]).float())),
        "pass": bool(model_ok),
    }
    if not g_carrier["pass"]:
        raise texture(f"G_CARRIER FAILED (model!=base+delta): {g_carrier}")

    # x24's panel cells through my own path: base + base+delta at host-g0
    x24m = json.loads((REPO / MD5_BINDS["x24_metrics"][0])
                      .read_text(encoding="utf-8"))
    stoi = p0["stoi"]
    names_ok = all(nm in x24m["panel_reads"] for nm in X24_PANEL)
    cids = [stoi[nm[0]] for nm in X24_PANEL]
    g0_ids = hp["batteries"]["g0"]
    net_base = G1.evl_load(base_params)
    net_comp = G1.evl_load(apply_delta(base_params, delta))
    pm_base = battery_percell(net_base, g0_ids, cids)
    pm_comp = battery_percell(net_comp, g0_ids, cids)
    del net_base, net_comp
    cells = {}
    for j, nm in enumerate(X24_PANEL):
        for cond, mat in (("base", pm_base), ("comp", pm_comp)):
            mine = float(mat[:, j].mean())
            theirs = x24m["panel_reads"][nm][f"{cond}_host_g0"]["mean_pz"]
            cells[f"{nm}|{cond}|host_g0"] = {
                "mine": mine, "committed": theirs,
                "abs_diff": abs(mine - theirs)}
    worst = max(cells.values(), key=lambda r: r["abs_diff"])
    worst_cell = next(k for k, v in cells.items() if v is worst)
    g_carrier["x24repro"] = {
        "n_cells": len(cells), "worst_cell": worst_cell,
        "worst_abs_diff": worst["abs_diff"], "tol": REPRO_TOL,
        "rows": cells,
        "zephyra_lift_54x": (cells["ZEPHYRA|comp|host_g0"]["mine"]
                             / cells["ZEPHYRA|base|host_g0"]["mine"]),
    }
    g_carrier["pass"] = bool(g_carrier["pass"]
                             and names_ok
                             and worst["abs_diff"] <= REPRO_TOL)
    if not g_carrier["pass"]:
        raise texture(f"G_CARRIER FAILED (x24 repro, worst {worst_cell} "
                      f"{worst['abs_diff']:.2e}): {g_carrier}")
    metrics["gates"]["G_CARRIER"] = g_carrier
    metrics["gates"]["G_KEYSET"] = g_keys      # completed in phase_states
    log(f"G_CARRIER PASS: model==base+delta bit-equal; x24's {len(cells)} "
        f"host-g0 cells bit-exact (worst |d| {worst['abs_diff']:.1e}; "
        f"ZEPHYRA lift "
        f"{g_carrier['x24repro']['zephyra_lift_54x']:.1f}x)")
    write_partial("P3 the displacement certified end-to-end")
    return {"delta": delta, "base_params": base_params,
            "base_keys": base_keys, "g_keyset_partial": g_keys}


# ======================================================================
# P4 — THE STATES + THE PRE/POST PER-CONTEXT READS
# ======================================================================
def phase_states(hp: dict, car: dict) -> dict:
    log("P4 — THE STATES (three committed classes; the pre/post per-context "
        "reads at the three geometries)")
    p0 = hp["p0"]
    batteries = hp["batteries"]
    arms = {
        "VARIED": {"rel": "runs/e341/e341_anneal_VARIED_post.pt"},
        "FIXED": {"rel": "runs/e341/e341_anneal_FIXED_post.pt"},
        "FRESH": {"rel": "runs/checkpoints/e311_TAVINST_post.pt",
                  "theta": hp["p1"]["theta0"]},
    }
    for a in ("VARIED", "FIXED"):
        arms[a]["theta"] = load_state(arms[a]["rel"])
    arms["FRESH"]["theta"] = hp["p1"]["theta0"]

    # G_KEYSET (completed): all three states ride the carrier's key order
    key_ok = all(list(arms[a]["theta"].keys()) == car["base_keys"]
                 for a in arms)
    g_keys = car["g_keyset_partial"]
    g_keys["states_keys_eq_base_params"] = bool(key_ok)
    g_keys["pass"] = bool(g_keys["pass"] and key_ok)
    if not g_keys["pass"]:
        raise texture(f"G_KEYSET FAILED: {g_keys}")
    metrics["gates"]["G_KEYSET"] = g_keys
    log(f"G_KEYSET PASS: the three states + the carrier + the base ride "
        f"one key order; N = {g_keys['n_params']:,}")

    cids = {"T": p0["tid"], "Q": p0["stoi"]["Q"], "Z": p0["zid"]}
    cid_list = [cids["T"], cids["Q"], cids["Z"]]

    out = {}
    for a in STATES:
        theta = arms[a]["theta"]
        net_pre = G1.evl_load(theta)                 # the plain read owner
        net_post = G1.evl_load(apply_delta(theta, car["delta"]))
        reads = {"pre": {}, "post": {}}
        for cond, net in (("pre", net_pre), ("post", net_post)):
            for bn, ids in batteries.items():
                mat = battery_percell(net, ids, cid_list)   # [60, 3]
                reads[cond][bn] = {
                    "T": [float(v) for v in mat[:, 0]],
                    "Q": [float(v) for v in mat[:, 1]],
                    "Z": [float(v) for v in mat[:, 2]],
                    "T_mean": float(mat[:, 0].mean()),
                    "T_alive_frac": float((mat[:, 0] >= READ_BAR)
                                          .float().mean()),
                }
        reads["ce_r_bare"] = G1.ce_fixed_cpu(net_pre, *p0["r_eval_xy"])
        reads["ce_r_comp"] = G1.ce_fixed_cpu(net_post, *p0["r_eval_xy"])
        del net_pre, net_post
        out[a] = reads
        log(f"  [{a}] pre g0 {reads['pre']['g0']['T_mean']:.4f} -> post "
            f"{reads['post']['g0']['T_mean']:.4f} | gm12 "
            f"{reads['pre']['g-12']['T_mean']:.4f} -> "
            f"{reads['post']['g-12']['T_mean']:.4f} | gp12 "
            f"{reads['pre']['g+12']['T_mean']:.4f} -> "
            f"{reads['post']['g+12']['T_mean']:.4f} | CE_R "
            f"{reads['ce_r_bare']:.4f} -> {reads['ce_r_comp']:.4f}")

    # ---- G_T0: the committed t0 means + ce_r reproduce ------------------
    e341t0 = json.loads((REPO / "runs" / "e341" / "metrics.json")
                        .read_text(encoding="utf-8"))["annealed_t0"]
    drifts = {}
    for a, bn, committed, tol in (
            ("VARIED", "g0", e341t0["VARIED"]["g0"], READ_TOL),
            ("VARIED", "g-12", e341t0["VARIED"]["gm12"], READ_TOL),
            ("FIXED", "g0", e341t0["FIXED"]["g0"], READ_TOL),
            ("FIXED", "g-12", e341t0["FIXED"]["gm12"], READ_TOL),
            ("FRESH", "g0", E38.SUBJ_READ_T, READ_TOL),
            ("FRESH", "g-12", E38.SUBJ_GM12_T, READ_TOL),
            ("VARIED", "ce_r", e341t0["VARIED"]["ce_r"], CE_TOL),
            ("FIXED", "ce_r", e341t0["FIXED"]["ce_r"], CE_TOL),
            ("FRESH", "ce_r", E38.SUBJ_CE_R, CE_TOL)):
        mine = (out[a]["ce_r_bare"] if bn == "ce_r"
                else out[a]["pre"][bn]["T_mean"])
        drifts[f"{a}:{bn}"] = {"mine": mine, "committed": committed,
                               "abs_diff": abs(mine - committed),
                               "tol": tol}
    g_t0 = {"drifts": drifts,
            "claim": "the three states' committed t0 reads + ce_r "
                     "reproduce on CPU (e341's annealed_t0 + the "
                     "harness's subject literals, runtime-read)",
            "pass": bool(all(d["abs_diff"] <= d["tol"]
                             for d in drifts.values()))}
    if not g_t0["pass"]:
        raise texture(f"G_T0 FAILED: {g_t0}")
    metrics["gates"]["G_T0"] = g_t0
    log("G_T0 PASS: all nine committed t0 cells reproduce (max |d| "
        + f"{max(d['abs_diff'] for d in drifts.values()):.1e})")

    # ---- G_PERCTX: the per-context reader vs battery_cell ---------------
    certs = {}
    for a in STATES:
        net = G1.evl_load(arms[a]["theta"])
        for bn, ids in batteries.items():
            mine = out[a]["pre"][bn]["T_mean"]
            ref = G1.battery_cell(net, ids, cids["T"])["mean_pz"]
            certs[f"{a}:{bn}"] = abs(mine - ref)
        del net
    worst_pc = max(certs.values())
    g_pc = {"cells": certs, "max_abs_diff": worst_pc, "tol": PERCTX_TOL,
            "claim": "the per-context reader's mean == battery_cell's "
                     "mean_pz (the instrument certification, 9 cells)",
            "pass": bool(worst_pc <= PERCTX_TOL)}
    if not g_pc["pass"]:
        raise texture(f"G_PERCTX FAILED: {g_pc}")
    metrics["gates"]["G_PERCTX"] = g_pc
    log(f"G_PERCTX PASS: max |mean diff| {worst_pc:.2e} <= {PERCTX_TOL:g}")

    # ---- G_NET0: every state classed, building blocks gated -------------
    g_n0 = {"classes": NET0_CLASSES,
            "claim": "every state classed; the class strings' building "
                     "blocks verbatim substrings of the md5-bound parents "
                     "(x40's rig + the harness)",
            "pass": True}
    metrics["gates"]["G_NET0"] = g_n0     # substrings already gated in
                                           # G_QUOTES; here: recorded
    log("G_NET0: all three states classed")
    write_partial("P4 the states read pre/post at the three geometries")
    return {"arms": arms, "reads": out, "cids": cids}


# ======================================================================
# P5 — THE FLIP ANALYSIS (classification + brackets + the composite)
# ======================================================================
def classify_cell(pre: list[float], post: list[float],
                  dQ: list[float], dZ: list[float]) -> dict:
    """The frozen per-context classifier (both forms) + the bracket."""
    n = len(pre)
    cls_band, cls_ratio = [], []
    band_floor_fired = 0
    sat_up = 0
    noise_lifts = 0
    for i in range(n):
        dT = abs(post[i] - pre[i])
        band = max(abs(dQ[i]), abs(dZ[i]))
        if band < BAND_FLOOR:
            band = BAND_FLOOR
            band_floor_fired += 1
        if dT > BAND_X * band:
            cls_band.append("COLLAPSE")
            if post[i] > pre[i]:
                sat_up += 1
        elif post[i] > pre[i]:
            cls_band.append("LIFT")
            if pre[i] < NOISE_ZONE:
                noise_lifts += 1
        else:
            cls_band.append("AMBIG")
        if post[i] <= COLLAPSE_RATIO * pre[i]:
            cls_ratio.append("COLLAPSE")
        elif post[i] > pre[i]:
            cls_ratio.append("LIFT")
        else:
            cls_ratio.append("AMBIG")

    def bracket(cls):
        lift_pre = [pre[i] for i in range(n) if cls[i] == "LIFT"]
        coll_pre = [pre[i] for i in range(n) if cls[i] == "COLLAPSE"]
        l_max = max(lift_pre) if lift_pre else None
        c_min = min(coll_pre) if coll_pre else None
        two_sided = l_max is not None and c_min is not None
        consistent = bool(two_sided and l_max < c_min)
        violations = 0
        if two_sided:
            violations = sum(1 for i in range(n)
                             if (cls[i] == "LIFT" and pre[i] >= c_min)
                             or (cls[i] == "COLLAPSE" and pre[i] <= l_max))
        r_hat = (math.sqrt(l_max * c_min)
                 if consistent else None)
        return {"n_lift": len(lift_pre), "n_collapse": len(coll_pre),
                "n_ambig": n - len(lift_pre) - len(coll_pre),
                "L_max": l_max, "C_min": c_min,
                "two_sided": bool(two_sided),
                "consistent": consistent, "violations": violations,
                "R_hat": r_hat,
                "lower_bound": (l_max if lift_pre and not coll_pre
                                else (l_max if consistent else None)),
                "upper_bound": (c_min if coll_pre and not lift_pre
                                else (c_min if consistent else None))}

    return {"cls_band": cls_band, "cls_ratio": cls_ratio,
            "bracket_band": bracket(cls_band),
            "bracket_ratio": bracket(cls_ratio),
            "band_floor_fired": band_floor_fired, "sat_up": sat_up,
            "noise_lifts": noise_lifts}


def phase_flip(st: dict) -> dict:
    log("P5 — THE FLIP ANALYSIS (classification + brackets)")
    cells = {}
    for a in STATES:
        cells[a] = {}
        for bn in GEOS:
            pre = st["reads"][a]["pre"][bn]["T"]
            post = st["reads"][a]["post"][bn]["T"]
            dQ = [post_q - pre_q for post_q, pre_q in
                  zip(st["reads"][a]["post"][bn]["Q"],
                      st["reads"][a]["pre"][bn]["Q"])]
            dZ = [post_z - pre_z for post_z, pre_z in
                  zip(st["reads"][a]["post"][bn]["Z"],
                      st["reads"][a]["pre"][bn]["Z"])]
            cells[a][bn] = classify_cell(pre, post, dQ, dZ)
            bb = cells[a][bn]["bracket_band"]
            log(f"  [{a}:{bn}] band {bb['n_lift']}L/{bb['n_collapse']}C/"
                f"{bb['n_ambig']}A; bracket "
                + (f"({bb['L_max']:.4g}, {bb['C_min']:.4g}] "
                   f"R_hat {bb['R_hat']:.4g}" if bb["consistent"]
                   else ("one-sided" if bb["two_sided"] else "no bracket"))
                + f" (viol {bb['violations']})")

    # G_FLIPCLASS: the machinery's internal consistency
    ok = True
    for a in cells:
        for bn in cells[a]:
            bb = cells[a][bn]["bracket_band"]
            br = cells[a][bn]["bracket_ratio"]
            for b in (bb, br):
                if b["n_lift"] + b["n_collapse"] + b["n_ambig"] != 60:
                    ok = False
                if b["two_sided"] and not b["consistent"]:
                    if b["violations"] == 0:
                        ok = False      # inconsistent without violations
                if b["consistent"] and b["violations"] != 0:
                    ok = False
    g_fc = {"claim": "counts sum to 60 per cell; brackets well-formed "
                     "(consistent iff zero violations)",
            "pass": bool(ok)}
    if not g_fc["pass"]:
        raise texture(f"G_FLIPCLASS FAILED: {g_fc}")
    metrics["gates"]["G_FLIPCLASS"] = g_fc
    write_partial("P5 the flip cells classified")
    return {"cells": cells}


def adjudicate(flip: dict) -> dict:
    """The frozen verdict composite (adjudicates on the PRIMARY band form;
    the sensitivity ratio form co-reported)."""
    cells = flip["cells"]

    # ---- the state-level ratios (primary comparison set) ----------------
    state_ratios = {}
    state_compare = {}
    for a in STATES:
        r_hats = {bn: cells[a][bn]["bracket_band"]["R_hat"]
                  for bn in GEOS}
        two_sided = {bn: r for bn, r in r_hats.items() if r is not None}
        geo_two = [bn for bn in GEOS if bn in two_sided]
        has_g0 = "g0" in two_sided
        has_off = any(bn in two_sided for bn in ("g-12", "g+12"))
        if len(two_sided) >= 2:
            vals = list(two_sided.values())
            state_ratios[a] = {"two_sided_geos": geo_two,
                               "R_hats": two_sided,
                               "ratio_max_over_min":
                                   max(vals) / min(vals),
                               "lowest_geo": min(two_sided,
                                                 key=two_sided.get),
                               "highest_geo": max(two_sided,
                                                  key=two_sided.get)}
        state_compare[a] = {"covers_g0_and_offset": bool(has_g0 and has_off),
                            "n_two_sided": len(two_sided)}
    primary_exists = any(state_compare[a]["covers_g0_and_offset"]
                         for a in STATES)

    # ---- the secondary (pooled cross-state per-geometry brackets) -------
    pooled = {}
    for bn in GEOS:
        l_maxes = [cells[a][bn]["bracket_band"]["L_max"] for a in STATES
                   if cells[a][bn]["bracket_band"]["consistent"]]
        c_mins = [cells[a][bn]["bracket_band"]["C_min"] for a in STATES
                  if cells[a][bn]["bracket_band"]["consistent"]]
        pooled[bn] = {
            "pooled_L_max": max(l_maxes) if l_maxes else None,
            "pooled_C_min": min(c_mins) if c_mins else None,
            "n_two_sided": len(l_maxes)}
        if l_maxes and c_mins and max(l_maxes) < min(c_mins):
            pooled[bn]["R_hat"] = math.sqrt(max(l_maxes) * min(c_mins))
        else:
            pooled[bn]["R_hat"] = None
    pooled_two = {bn: pooled[bn]["R_hat"] for bn in GEOS
                  if pooled[bn]["R_hat"] is not None}
    pooled_geos = [bn for bn in GEOS if bn in pooled_two]
    secondary_exists = (len(pooled_two) >= 2
                        and any(bn != "g0" for bn in pooled_geos)
                        and "g0" in pooled_two)

    # ---- the nine-cell pooled consistency (primary form) ----------------
    lowers, uppers = [], []
    for a in STATES:
        for bn in GEOS:
            b = cells[a][bn]["bracket_band"]
            if b["consistent"]:
                lowers.append(b["L_max"])
                uppers.append(b["C_min"])
            elif b["n_lift"] > 0 and b["n_collapse"] == 0:
                lowers.append(b["L_max"])          # R* > L_max
            elif b["n_collapse"] > 0 and b["n_lift"] == 0:
                uppers.append(b["C_min"])          # R* <= C_min
    max_lower = max(lowers) if lowers else None
    min_upper = min(uppers) if uppers else None
    pooled_consistent = bool(
        max_lower is not None and min_upper is not None
        and max_lower < min_upper)

    # ---- TRACKS / UNIFORM ------------------------------------------------
    tracks_fire = False
    tracks_source = None
    if primary_exists:
        for a in STATES:
            if a in state_ratios and \
                    state_ratios[a]["ratio_max_over_min"] > UNIFORM_X:
                tracks_fire = True
                hi_geo = state_ratios[a]["highest_geo"]
                lo_geo = state_ratios[a]["lowest_geo"]
                hi_r = state_ratios[a]["R_hats"][hi_geo]
                lo_r = state_ratios[a]["R_hats"][lo_geo]
                ratio = state_ratios[a]["ratio_max_over_min"]
                tracks_source = (f"state {a}: {hi_geo} R_hat {hi_r:.4g} "
                                 f"vs {lo_geo} {lo_r:.4g} = "
                                 f"{ratio:.2f}x")
                break
    if not tracks_fire and secondary_exists:
        vals = pooled_two
        hi = max(vals, key=vals.get)
        lo = min(vals, key=vals.get)
        if vals[hi] / vals[lo] > UNIFORM_X:
            tracks_fire = True
            tracks_source = (f"pooled: {hi} R_hat {vals[hi]:.4g} vs "
                             f"{lo} {vals[lo]:.4g} = "
                             f"{vals[hi] / vals[lo]:.2f}x")

    all_ratios_le_2x = True
    for a in state_ratios:
        if state_ratios[a]["ratio_max_over_min"] > UNIFORM_X:
            all_ratios_le_2x = False
    if secondary_exists:
        vals = list(pooled_two.values())
        if max(vals) / min(vals) > UNIFORM_X:
            all_ratios_le_2x = False
    comparison_exists = bool(primary_exists or secondary_exists)
    uniform_fire = bool(comparison_exists and all_ratios_le_2x
                        and pooled_consistent)

    # direction (for TRACKS): which geometry sits lower
    direction = None
    if tracks_fire:
        if primary_exists and tracks_source.startswith("state"):
            a = tracks_source.split()[1].rstrip(":")
            direction = {f"{state_ratios[a]['lowest_geo']}": "lowest",
                         f"{state_ratios[a]['highest_geo']}": "highest"}
        elif secondary_exists:
            vals = pooled_two
            direction = {min(vals, key=vals.get): "lowest",
                         max(vals, key=vals.get): "highest"}

    # ---- the verdict word ------------------------------------------------
    if SMOKE:
        word, clause = "SMOKE", "smoke — nothing adjudicated"
    elif tracks_fire:
        word = "R*-TRACKS-THE-SUPPORT-TAIL"
        clause = (f"the flip level is context-local: {tracks_source} "
                  f"(> {UNIFORM_X:g}x across geometries; direction: "
                  f"{direction}) — the offsets and g0 do NOT share one "
                  "read-level threshold; the support breadth reads as "
                  "flip-margin")
    elif uniform_fire:
        word = "R*-UNIFORM"
        clause = (f"the geometries' flip levels agree within ~{UNIFORM_X:g}x "
                  f"(max state ratio "
                  f"{max((state_ratios[a]['ratio_max_over_min']
                          for a in state_ratios), default=float('nan')):.2f}"
                  f") AND the nine cells' constraints are pooled-"
                  f"consistent (max lower {max_lower:.4g} < min upper "
                  f"{min_upper:.4g}) — the flip is a slot property; the "
                  "two-channel law's parameter is context-independent")
    else:
        word = "MIXED"
        why = ("under-determined: fewer than two comparable geometries "
               "have two-sided brackets" if not comparison_exists
               else "constraints inconsistent without >2x point-estimate "
                    "separation (the table verbatim)")
        clause = (f"neither frozen bar fires — {why}; no wording change "
                  "without a new registered cell")

    # ---- the sensitivity form's verdict (never adjudicates) --------------
    sens = {}
    for a in STATES:
        rh = {bn: cells[a][bn]["bracket_ratio"]["R_hat"] for bn in GEOS}
        vals = [v for v in rh.values() if v is not None]
        sens[a] = {"R_hats": rh,
                   "ratio_max_over_min": (max(vals) / min(vals)
                                          if len(vals) >= 2 else None)}

    # ---- the riders -------------------------------------------------------
    # (a) the margin map: P(alive post | pre > R_hat) vs P(alive | pre <=)
    margin = {}
    for a in STATES:
        margin[a] = {}
        for bn in GEOS:
            r_hat = cells[a][bn]["bracket_band"]["R_hat"]
            if r_hat is None:
                # fall to the geometry's pooled estimate if present
                r_hat = pooled[bn]["R_hat"]
            if r_hat is None:
                margin[a][bn] = {"R_hat_used": None}
                continue
            pre = st_pre_cache[a][bn]
            post = st_post_cache[a][bn]
            above = [i for i in range(60) if pre[i] > r_hat]
            below = [i for i in range(60) if pre[i] <= r_hat]
            p_above = (sum(1 for i in above if post[i] >= READ_BAR)
                       / len(above)) if above else None
            p_below = (sum(1 for i in below if post[i] >= READ_BAR)
                       / len(below)) if below else None
            margin[a][bn] = {"R_hat_used": r_hat, "n_above": len(above),
                             "n_below": len(below),
                             "p_alive_post_given_pre_above": p_above,
                             "p_alive_post_given_pre_atbelow": p_below}
    # (c) the paired elicitation correlations
    corr = {}
    for a in STATES:
        corr[a] = {}
        for b1, b2 in (("g-12", "g0"), ("g0", "g+12"), ("g-12", "g+12")):
            x = st_pre_cache[a][b1]
            y = st_pre_cache[a][b2]
            mx, my = sum(x) / 60, sum(y) / 60
            cov = sum((x[i] - mx) * (y[i] - my) for i in range(60))
            vx = math.sqrt(sum((v - mx) ** 2 for v in x))
            vy = math.sqrt(sum((v - my) ** 2 for v in y))
            corr[a][f"{b1}|{b2}"] = cov / (vx * vy) if vx > 0 and vy > 0 \
                else None

    return {"verdict": word, "clause": clause, "direction": direction,
            "state_ratios": state_ratios, "state_compare": state_compare,
            "pooled_brackets": pooled, "pooled_consistent":
                pooled_consistent,
            "pooled_bounds": {"max_lower": max_lower,
                              "min_upper": min_upper},
            "primary_exists": primary_exists,
            "secondary_exists": secondary_exists,
            "comparison_exists": comparison_exists,
            "tracks_fire": tracks_fire, "tracks_source": tracks_source,
            "uniform_fire": uniform_fire,
            "sensitivity_form": {"state_ratios": sens,
                                 "verdict_agrees": None},  # filled below
            "margin_rider": margin, "elicitation_corr": corr,
            "P_x44a": {"guess": REGISTERED["P_x44a"]["my_guess"],
                       "lab_lean": REGISTERED["lab_lean_verbatim"],
                       "hit": bool(word == "R*-TRACKS-THE-"
                                   "SUPPORT-TAIL"),
                       "scored": REGISTERED["P_x44a"]["scored"]}}


# module-level caches for the riders (set in main before adjudicate)
st_pre_cache: dict = {}
st_post_cache: dict = {}


# ======================================================================
# P6 — THE OUTPUTS (the PNG + the REPORT)
# ======================================================================
def make_png(flip: dict, adj: dict, st: dict) -> None:
    refs = metrics["references"]
    x38b = refs["x38_R_star_bracket_host_g0"]
    x42w = refs["x42_R_star_window_host_g0"]
    cols = {"g-12": "tab:blue", "g0": "tab:red", "g+12": "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))
    for j, a in enumerate(STATES):
        ax = axes[0, j]
        for bn in GEOS:
            pre = st["reads"][a]["pre"][bn]["T"]
            post = st["reads"][a]["post"][bn]["T"]
            ratios = [post[i] / max(pre[i], 1e-12) for i in range(60)]
            ax.scatter(pre, ratios, s=16, color=cols[bn], alpha=0.75,
                       label=bn)
        ax.axhline(1.0, color="black", ls="-", lw=1.0)
        ax.axhline(COLLAPSE_RATIO, color="gray", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("the local read (pre, p(T) per context)")
        ax.set_ylabel("response (post/pre)")
        bb_g0 = flip["cells"][a]["g0"]["bracket_band"]
        ax.set_title(f"{a} — g0 cell {bb_g0['n_lift']}L/"
                     f"{bb_g0['n_collapse']}C/{bb_g0['n_ambig']}A",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")

    # panel 4: THE BRACKET MAP (the verdict picture)
    ax = axes[1, 0]
    ax.axvspan(x38b[0], x38b[1], color="lightgray", alpha=0.6,
               label=f"x38 host-g0 bracket ({x38b[0]:.1e}, {x38b[1]:.3f}]")
    if x42w:
        ax.axvline(x42w[0], color="gray", ls=":", lw=1.2)
        ax.axvline(x42w[1], color="gray", ls=":", lw=1.2)
    yy = 0
    ytick, ylab = [], []
    for a in STATES:
        for bn in GEOS:
            b = flip["cells"][a][bn]["bracket_band"]
            yy += 1
            if b["consistent"]:
                ax.plot([b["L_max"], b["C_min"]], [yy, yy], "-",
                        color=cols[bn], lw=3.0, alpha=0.85)
                ax.plot([b["R_hat"]], [yy], "o", color=cols[bn], ms=7)
            elif b["two_sided"]:
                ax.plot([b["L_max"], b["C_min"]], [yy, yy], "-",
                        color=cols[bn], lw=1.2, ls="--", alpha=0.6)
            else:
                if b["n_collapse"] > 0 and b["C_min"] is not None:
                    ax.plot([1e-7, b["C_min"]], [yy, yy], ":", lw=1.2,
                            color=cols[bn], alpha=0.6)
                if b["n_lift"] > 0 and b["L_max"] is not None:
                    ax.plot([b["L_max"], 1.0], [yy, yy], ":", lw=1.2,
                            color=cols[bn], alpha=0.6)
            ytick.append(yy)
            ylab.append(f"{a}:{bn}")
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1.0)
    ax.set_xlabel("the read level (log); bracket (L_max, C_min]")
    ax.set_title("THE R* BRACKETS per (state, geometry) — solid=two-sided, "
                 "dotted=one-sided; gray=x38/x42 host-g0 reference",
                 fontsize=9)
    ax.grid(alpha=0.25, which="both", axis="x")
    ax.legend(fontsize=7, loc="lower right")

    # panel 5: the response-vs-read binned curves (the three geometries
    # overlaid, all states pooled per geometry)
    ax = axes[1, 1]
    for bn in GEOS:
        xs, ys = [], []
        prevx = prevy = None
        for a in STATES:
            pre = st["reads"][a]["pre"][bn]["T"]
            post = st["reads"][a]["post"][bn]["T"]
            xs += pre
            ys += [math.log10(max(post[i], 1e-12) / max(pre[i], 1e-12))
                   for i in range(60)]
        pts = sorted(zip(xs, ys))
        k = 15                                                # bins
        for b0 in range(0, 180, k):
            chunk = pts[b0:b0 + k]
            if not chunk:
                continue
            mx = sum(p[0] for p in chunk) / len(chunk)
            my = sum(p[1] for p in chunk) / len(chunk)
            ax.plot(mx, my, "o", color=cols[bn], ms=5, alpha=0.8)
            if prevx is not None:
                ax.plot([prevx, mx], [prevy, my], "-", color=cols[bn],
                        lw=1.2, alpha=0.7)
            prevx, prevy = mx, my
        ax.scatter([], [], s=16, color=cols[bn], label=bn)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("the local read (pre, pooled over states)")
    ax.set_ylabel("mean signed log10(post/pre)")
    ax.set_title("THE FLIP CURVES — the local read vs the local response "
                 "(3 geometries overlaid)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # panel 6: the margin map (the varied arm's offsets)
    ax = axes[1, 2]
    w = 0.35
    for i, bn in enumerate(("g-12", "g+12")):
        mr = adj["margin_rider"]["VARIED"][bn]
        if mr.get("R_hat_used") is None:
            continue
        vals = [mr.get("p_alive_post_given_pre_atbelow") or 0.0,
                mr.get("p_alive_post_given_pre_above") or 0.0]
        ax.bar([i - w / 2, i + w / 2], vals, width=w,
               color=["lightgray", "tab:blue"], alpha=0.85)
        for xx, v in ((i - w / 2, vals[0]), (i + w / 2, vals[1])):
            ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["g-12", "g+12"])
    ax.set_ylabel("P(post >= 0.05)")
    ax.set_title("THE MARGIN MAP (VARIED, offsets): alive-after-complement "
                 "below vs above R_hat (gray=below, blue=above)",
                 fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"X44 R*-AT-THE-OFFSETS — {adj['verdict']} (P-x44a "
                 f"{'HIT' if adj['P_x44a']['hit'] else 'MISSED'}; "
                 f"pooled-consistent {adj['pooled_consistent']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RD / "x44_rstar_offsets.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def write_report(flip: dict, adj: dict, st: dict) -> None:
    L = []
    A = L.append
    A("# X44 — R*-AT-THE-OFFSETS (is x38's flip a slot property or a "
      "state-at-context property?)")
    A("")
    A(f"* VERDICT: **{adj['verdict']}** — {adj['clause']}")
    A(f"* P-x44a (my registered read): "
      f"{'HIT' if adj['P_x44a']['hit'] else 'MISSED'} — guess "
      f"{adj['P_x44a']['guess']}; the dispatch's lean: "
      f"R*-TRACKS-THE-SUPPORT-TAIL, weakly")
    A(f"* direction: {adj['direction']}")
    A("* gates: " + str(sum(1 for g in metrics["gates"].values()
                             if g.get("pass"))) + "/"
      + str(len(metrics["gates"])) + " PASS"
      + ("" if all(g.get("pass") for g in metrics["gates"].values())
         else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## 1. The flip cells (the primary band form; the dispatch's "
      "per-context question)")
    A("")
    A("| state | geo | pre mean (alive) | post mean (alive) | L/C/A | "
      "L_max | C_min | R_hat | viol | floor | satup |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in STATES:
        for bn in GEOS:
            c = flip["cells"][a][bn]
            bb = c["bracket_band"]
            pre_m = st["reads"][a]["pre"][bn]["T_mean"]
            post_m = st["reads"][a]["post"][bn]["T_mean"]
            pre_a = st["reads"][a]["pre"][bn]["T_alive_frac"]
            post_a = st["reads"][a]["post"][bn]["T_alive_frac"]
            A(f"| {a} | {bn} | {pre_m:.4f} ({pre_a:.2f}) | "
              f"{post_m:.4f} ({post_a:.2f}) | {bb['n_lift']}/"
              f"{bb['n_collapse']}/{bb['n_ambig']} | "
              f"{bb['L_max'] if bb['L_max'] is None else round(bb['L_max'], 5)} | "
              f"{bb['C_min'] if bb['C_min'] is None else round(bb['C_min'], 5)} | "
              f"{bb['R_hat'] if bb['R_hat'] is None else round(bb['R_hat'], 5)} | "
              f"{bb['violations']} | {c['band_floor_fired']} | "
              f"{c['sat_up']} |")
    A("")
    A("## 2. The geometry comparison (the frozen composite)")
    A("")
    A("```json")
    A(json.dumps({"state_ratios": adj["state_ratios"],
                  "pooled_brackets": adj["pooled_brackets"],
                  "pooled_bounds": adj["pooled_bounds"],
                  "pooled_consistent": adj["pooled_consistent"],
                  "primary_exists": adj["primary_exists"],
                  "secondary_exists": adj["secondary_exists"],
                  "tracks_fire": adj["tracks_fire"],
                  "tracks_source": adj["tracks_source"],
                  "uniform_fire": adj["uniform_fire"]},
                 indent=1, default=float))
    A("```")
    A("")
    A("## 3. The riders (descriptive, never bars)")
    A("")
    A("### The margin map (the dispatch's closing claim, countable)")
    A("```json")
    A(json.dumps(adj["margin_rider"], indent=1, default=float))
    A("```")
    A("")
    A("### The sensitivity form (parameter-free; never adjudicates)")
    A("```json")
    A(json.dumps(adj["sensitivity_form"], indent=1, default=float))
    A("```")
    A("")
    A("### The paired elicitation correlations (same 60 host positions "
      "across geometries)")
    A("```json")
    A(json.dumps(adj["elicitation_corr"], indent=1, default=float))
    A("```")
    A("")
    A("### ce_r per state (bare -> comp) + noise-zone lifts")
    A("")
    for a in STATES:
        A(f"* {a}: CE_R {st['reads'][a]['ce_r_bare']:.4f} -> "
          f"{st['reads'][a]['ce_r_comp']:.4f}; noise-zone lifts "
          + "/".join(f"{bn} {flip['cells'][a][bn]['noise_lifts']}"
                     for bn in GEOS))
    A("")
    A("## 4. The references (runtime-read from the md5-bound records)")
    A("")
    x38b = metrics["references"]["x38_R_star_bracket_host_g0"]
    A(f"* x38's committed host-g0 bracket: ({x38b[0]:.4g}, {x38b[1]:.4g}] "
      f"(verdict {metrics['references']['x38_verdict_word']})")
    x42w = metrics["references"]["x42_R_star_window_host_g0"]
    A(f"* x42's committed interior window: "
      + (f"({x42w[0]:.4g}, {x42w[1]:.4g}]" if x42w else "absent"))
    sup = metrics["references"]["x40_support_alive_s1"]
    A(f"* x40's committed s1 support table (the reference this cell "
      f"re-reads): {json.dumps(sup)}")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* 19 md5 binds; the carrier certified end-to-end (model == base + "
      "delta bit-equal + x24's 22 host-g0 panel cells bit-exact through "
      "this cell's own reader); the six committed t0 reads + three ce_r "
      "reproduce; CPU-only (threads 4); timestamps UTC only; no "
      "NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds)")
    (RD / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X44 — R*-AT-THE-OFFSETS (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = git_head()
    write_partial("startup (bars + P-x44a registered, committed at birth)")
    set_seed(GLOBAL_SEED)          # global init only; no fresh draws exist

    gate_res = phase_gates()               # md5 + quotes + references
    hp = phase_harness()                   # phase_P0 + phase_P1 + gp12
    car = phase_carrier(hp)                # the certified displacement
    st = phase_states(hp, car)             # the states + pre/post reads
    flip = phase_flip(st)                  # the classification + brackets

    # the riders' caches
    for a in STATES:
        st_pre_cache[a] = {bn: st["reads"][a]["pre"][bn]["T"]
                           for bn in GEOS}
        st_post_cache[a] = {bn: st["reads"][a]["post"][bn]["T"]
                            for bn in GEOS}

    adj = adjudicate(flip)                 # the frozen composite

    # the sensitivity form's agreement (descriptive)
    sens_tracks = any((r["ratio_max_over_min"] or 0) > UNIFORM_X
                      for r in adj["sensitivity_form"]["state_ratios"]
                      .values())
    adj["sensitivity_form"]["verdict_agrees"] = bool(
        sens_tracks == adj["tracks_fire"])

    metrics["reads_summary"] = {
        a: {bn: {"pre_mean": st["reads"][a]["pre"][bn]["T_mean"],
                 "pre_alive": st["reads"][a]["pre"][bn]["T_alive_frac"],
                 "post_mean": st["reads"][a]["post"][bn]["T_mean"],
                 "post_alive": st["reads"][a]["post"][bn]["T_alive_frac"]}
            for bn in GEOS} for a in STATES}
    metrics["ce_r"] = {a: {"bare": st["reads"][a]["ce_r_bare"],
                           "comp": st["reads"][a]["ce_r_comp"]}
                       for a in STATES}
    metrics["flip_cells"] = flip["cells"]
    metrics["adjudication"] = adj

    if SMOKE:
        metrics["status"] = ("SMOKED — full gate path + all reads + the "
                             "classification exercised; NOTHING adjudicated")
        write_partial("SMOKE COMPLETE — nothing adjudicated")
        log("SMOKE COMPLETE (nothing adjudicated)")
        raise SystemExit(0)

    make_png(flip, adj, st)
    write_report(flip, adj, st)
    metrics["verdict"] = {"word": adj["verdict"], "clause": adj["clause"],
                          "P_x44a_hit": adj["P_x44a"]["hit"]}
    metrics["git_head_final"] = git_head()
    metrics["date_completed"] = common.now_iso()
    metrics["outputs"] = [str((RD / "x44_rstar_offsets.png")
                              .relative_to(REPO)),
                          str((RD / "REPORT.md").relative_to(REPO)),
                          str((RD / "metrics.json").relative_to(REPO))]
    metrics["status"] = "COMPLETE"
    save_json(RD / "metrics.json", metrics)
    log(f"X44 COMPLETE — verdict {adj['verdict']} "
        f"(P-x44a {'HIT' if adj['P_x44a']['hit'] else 'MISSED'}); "
        f"gates {sum(1 for g in metrics['gates'].values() if g.get('pass'))}"
        f"/{len(metrics['gates'])} PASS")


if __name__ == "__main__":
    main()
