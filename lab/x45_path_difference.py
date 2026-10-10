"""X45 — THE PATH DIFFERENCE (R80's first minted card; THE
FORMATION-PROTOCOL ERA'S FIRST CELL). This docstring carries the
registered question + background + design + bars + lab lean + P-x45a
VERBATIM from the dispatch letter + every frozen operationalization,
committed at birth BEFORE any compute. Adjudicate against exactly this;
no bar shopping.

THE BACKGROUND (dispatch verbatim): "e335's walk holds the cons's
formation PATH (the committed trajectory: s1/s100/s200/s300 reads + the
states); x43 holds the anneal's panels (the every-25-step anneal
trajectory on the fresh install, committed in runs/x43/). The capstone
proved the ENDPOINTS differ (the cons reaches 0.90 recovery-in-band; the
anneal saturates at 0.26-0.33). THE QUESTION: do the PATHS differ at
matched reads — is there a mid-formation signature the cons passes
through that the anneal skips?"

THE DESIGN (dispatch verbatim; all committed; the diff at matched reads):
  "1. MATCH: pair the two trajectories by read level (not by step): for
   each cons walk rung (s100 0.55, s200, s300, s400 0.90 — e335's table)
   find the anneal panel state at the NEAREST read (x43's every-25-step
   series covers 0.28-0.75; the cons's high rungs 0.7-0.9 may exceed it —
   disclose the match window).
  2. THE OBSERVABLES at each matched pair (the era's three instrument
     families, all committed):
     (a) SUPPORT BREADTH (x40's instrument): the alive-context fraction
         at g0 and the offsets per state;
     (b) ELICITATION STRUCTURE (x44's instrument): the per-geometry
         correlation matrix (g0 x g-12 x g+12) per state;
     (c) THE ROOM/CONTENT SPLIT (the standard census conventions): write
         norm, in-room fraction at matched reads.
  3. THE DIFF: which observable separates the two paths at matched
     reads?"

BARS (dispatch VERBATIM, frozen in this birth commit BEFORE compute):
  - PROTOCOL-SECRET-IN-THE-PATH: "at matched reads, at least one
    observable separates (e.g., the cons's mid-formation states show
    broader offset support or decorrelated elicitation families that the
    anneal's matched states lack) — THE FIRST POSITIVE FINGERPRINT of
    the missing formation protocol; the era's target list opens."
  - ENDPOINTS-ONLY: "the paths are equivalent at matched reads on every
    observable (within disclosed noise) — the gap is NOT trajectory
    shape; it is something no continuous anneal does (curriculum order,
    a discrete event, or the organism-history coupling) — the protocol
    hunt changes class."

LAB LEAN (dispatch verbatim): "ENDPOINTS-ONLY, weakly — the anneal was
the cons's own protocol ported and bought the s1 survival (the biggest
path signature) already; the countervailing: the cons's walk passed
through the base-formed install's slow formation first (a two-phase
path) while the anneal started from the formed install — the paths
differ in ORIGIN by construction. State your own read."

P-x45a (THE EXECUTOR'S OWN READ, registered per the dispatch's "Register
P-x45a BEFORE compute ... State your own read", frozen HERE at birth;
predictions are scored): ENDPOINTS-ONLY, MODERATELY (concurring with the
dispatch's weak lean, strengthened). SCORED: TRUE iff the verdict ==
ENDPOINTS-ONLY (plain or with the named origin clause). GROUNDS: (1)
THE INSTRUMENTS' OWN RECORD says the anneal ALREADY buys the
mid-formation signatures on the primary observables — x40: the annealed
states' alive fractions survive (0.38-0.47 at g0 vs the fresh's 0.00)
and x44: the annealed states' elicitation families DECORRELATE (VARIED
r(g-12|g0) 0.33 vs the fresh's 0.97) — the anneal's own trajectory
passes through the decorrelated-support regime; what x43 proved missing
is the ENDPOINT response (composed retention 0.24-0.33 vs the band
0.957-1.088), not a static mid-formation signature. (2) THE ARITHMETIC
IDENTITY: the anneal IS the cons's own protocol (seed 10901, the same
16+8+8 masked-union-CE batch, the same optimizer — G_STREAM-provable
bit-identical draws); the remaining structural differences are the
ORIGIN (a two-phase slow install with a FREE write vs a room-projected
formed install) and the taught name — origin effects live in the census
observable (c), pre-registered as the ORIGIN clause, never a protocol
secret. (3) THE READ-HEIGHT ARGUMENT: in every committed comparison the
static observables tracked the READ HEIGHT, not the path history (x40's
three classes ordered by read 0.29 -> 0.75/0.76; the alive fractions and
correlations ordered with them). PREDICTED SHAPE: at matched reads the
primary observables agree within the frozen floors (alive |d| mostly <
0.10, correlation |d| mostly < 0.15, no consistent sign); the census
observable separates WITH THE ORIGIN-EXPECTED SIGNS (the walk's in-room
fraction BELOW the anneal's at matched reads — free vs projected
install; the walk's write norm ABOVE — the bulky batch-64 install leg,
the e324 bulky-survivor texture). FALSIFIER: any PRIMARY observable
separating under the frozen rule (>= 75% sign consistency + floor) —
especially the dispatch's own e.g. (the cons's mid-formation states
showing broader offset support or more decorrelated elicitation
families) -> PROTOCOL-SECRET-IN-THE-PATH; or a hard-gate failure ->
TEXTURE (nothing adjudicated).

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ===

* THE RUNG SET := the cons walk's COMMITTED in-run trajectory rungs,
  runtime-read from runs/g1c_root/metrics.json (md5-bound; never
  retyped): the install leg {1, 100, 200, 300, 400} + the cons leg
  {25, 50, ..., 300} (17 rungs; the dispatch's "(s100 0.55, s200, s300,
  s400 0.90 — e335's table)" maps onto this committed table — the
  '0.90' is the cons final's gm12 read, co-reported). The base (e001)
  rides as the floor row (co-report, never matched).

* THE PANEL SET := x43's committed every-25-step anneal panel series
  (37 rows, steps {0, 25, ..., 900}), runtime-read from
  runs/x43/anneal_resume.pt's traj record (md5-bound; the series x43's
  REPORT printed) + the committed leg-boundary snapshot states
  (runs/x43/x43_anneal_leg{300,600,900}.pt, md5-bound).

* THE LEG-ANCHORED REPLAY (the interior-state regeneration; disclosed):
  the interior rung/panel STATES do not exist on disk — each leg is
  regenerated by DETERMINISTIC CPU replay of its committed arithmetic
  from its committed BIT-EXACT start state: install leg from e001.pt
  (e043-Dmix VERBATIM: batch 16 install (name-masked) + 16 paired
  originals + 32 random; masked token-level union CE; AdamW (0.9, 0.95)
  wd 0.1; lr 1e-3 x the house cosine total=1000 warmup=100; clip 1.0;
  gen 24314; s400); cons leg from the COMMITTED g1c_install_resume.pt
  model (e113 VERBATIM: batch 32 = 16 jittered-pool windows
  (name-masked) + 8 paired originals + 8 random; const lr 1e-3; gen
  10901; s300); anneal leg from the COMMITTED e311_TAVINST_post.pt
  subject (x43's anneal_extend arithmetic VERBATIM: the same cons
  arithmetic on the TAVIREN pool, gen 10901, s900, panels every 25).
  LEG BOUNDARIES USE THE COMMITTED ANCHOR STATES (install s400 -> the
  committed own-install state; cons s300 -> the committed root; anneal
  0/300/600/900 -> the committed subject + x43's leg snapshots); the
  replay exists only to regenerate the INTERIOR states. CPU-only (the
  dispatch's desk lane; the GPU lane (g1bS8) untouched); threads 4;
  1800s slice caps with full-state resume (the parked-CPU convention).

* THE REPLAY FIDELITY GATES (cross-device reality, frozen honestly):
  G_STREAM (HARD) — my anneal replay's draws bit-identical to x43's
  committed 900-draw record (the same generator arithmetic; device-
  independent); G_ENDPOINTS (HARD, 2e-6) — the committed anchor states'
  reads reproduce on CPU (e001 prior, the own-install g0, the root
  g0+gm12 via e335's committed rows, the TAVIREN subject's g0+gm12 via
  x40's committed FRESH row, the three x43 leg snapshots' reload reads
  vs their committed t0s); G_REPLCLASS (DISCLOSED tolerance) — every
  replayed read inside its committed series' range +-0.10 AND the
  replayed leg-endpoint reads within 0.15 of the committed endpoints
  (the lottery-bounce scale; per-rung drifts reported verbatim — the
  reads are lottery samples and cross-device replay resamples them; the
  CLASS, not the sample, is certified).

* THE MATCH := nearest-read pairing on MY OWN realized reads (one ruler,
  one realization; provenance-labeled): for each walk rung the anneal
  panel with the minimal |read difference| (many-to-one allowed;
  disclosed). MATCH WINDOW := 0.10 (frozen); rungs whose nearest panel
  lies farther are OUT-OF-WINDOW (named, disclosed, never adjudicating;
  the install-s1 rung at read ~1e-5 is expected out-of-window — the
  anneal series' floor is the formed subject's 0.286). The window
  disclosure includes the anneal series' read span vs the walk rungs'
  span (the dispatch's "the cons's high rungs 0.7-0.9 may exceed it").

* THE OBSERVABLES (per state; the two paths on their OWN name channel —
  p(Z) for the walk, p(T) for the anneal — over the SAME committed
  60-context host battery (SPLICE_RNG 19/41; e335's P0 == e338's P0 ==
  e311's bank, gated)):
  (a) SUPPORT BREADTH (x40's instrument, ported VERBATIM): per-context
      p(NAME) at g-12/g0/g+12 (battery_percell, bs=30, softmax at the
      final position); alive_frac := frac of the 60 contexts with
      p >= 0.05 (x40's READ_BAR);
  (b) ELICITATION STRUCTURE (x44's rider form, ported VERBATIM): the
      paired-context Pearson r across the 60 contexts for the three
      geometry pairs (g-12|g0, g0|g+12, g-12|g+12);
  (c) ROOM/CONTENT SPLIT (the e311/e324 census conventions): the
      cumulative write theta_state - theta_e001 (fp64; the canonical
      parameters() order) — its NORM and its IN-ROOM FRACTION under the
      committed K10K room (the SRCT projector: D * idct(mask_S(dct(D *
      x))), fp64; the room rebuilt from its registered seeds
      26113/26114 at k=10000 and BIT-BOUND to e264_rooms.pt's D_int8/S
      — the same fixed 10,000-dim subspace ruler for BOTH paths; the
      span/v-map ledger pieces are not used).

* THE INSTRUMENT CERTIFICATION (G_INSTRUMENT, HARD): my ported
  instrument reproduces the committed x40/x44 rows on the ONE state both
  families measured — the TAVIREN subject (x40's FRESH t0 row: the
  three means within 2e-6, the three alive fracs exact (discrete, 1/60
  quanta); x44's FRESH elicitation correlations within 1e-6), all
  runtime-read from the md5-bound records.

* THE SEPARATION RULE (frozen; the dispatch's "which observable
  separates" made countable): for each observable o, over the IN-WINDOW
  matched pairs: d_i := value_cons(i) - value_anneal(j(i)); o SEPARATES
  iff [sign consistency: >= 75% of the pairs share d's sign] AND [mean
  |d| >= floor_o]; floors frozen: alive 0.10 (x40's own clause margin);
  correlation 0.15 (covers 1/sqrt(59) ~ 0.13); in-room 0.05; write norm
  RELATIVE 0.15 (mean |d| / mean pair norm). Correlation pairs with a
  None side (zero variance; the pre-formation floor) are excluded and
  counted (disclosed).

* THE CARRIER CLASSES (the dispatch's own e.g.'s are the primary):
  PRIMARY := {(a) alive_gm12, alive_g0, alive_gp12; (b) r(g-12|g0),
  r(g0|g+12), r(g-12|g+12)} — a PRIMARY separation alone fires
  PROTOCOL-SECRET-IN-THE-PATH. CENSUS := {(c) write_norm, in_room} —
  the census observable is ORIGIN-CONFOUNDED BY CONSTRUCTION (the
  anneal path's install was K10K-PROJECTED (in-room 0.9424 committed)
  while the walk's install was a FREE batch-64 write; the dispatch's
  own countervailing names the origin difference); a CENSUS-only
  separation adjudicates the NAMED RESIDUE ORIGIN-ONLY-SEPARATION (an
  ENDPOINTS-ONLY sub-class with the origin clause explicit), never
  PROTOCOL-SECRET alone. FROZEN COMPOSITE: TEXTURE (any hard-gate
  failure; nothing adjudicated) -> PROTOCOL-SECRET-IN-THE-PATH (>= 1
  primary separates) -> ENDPOINTS-ONLY (no primary separates; the
  census clause names the origin residue when it fires).

* THE NET0 CLASS of every state recorded (x32's standing rule): BASE /
  BASE-forming / BASE-formed / ROOT-forming / ROOT (the walk, e335's
  classes) and BASE-formed(K10K-projected) / ANNEAL-forming (the anneal
  panels, x43's subject class).

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
family's own reason: CONTINUITY on the committed e335/x43 artifacts —
the two paths' own organisms). CPU desk ONLY (torch threads 4; zero
CUDA calls — the GPU lane is g1bS8's); 1800s slice caps with resume;
timestamps datetime.now(UTC) only.

Outputs: runs/x45/{metrics.json (PROGRESSIVE), REPORT.md,
x45_path_difference.png} (the matched-pair table: each observable side
by side along the read axis; the separation panel). The replay resumes
live under runs/x45/ (*.pt gitignored by the house convention,
regenerable deterministically). No NOTES/THINKING/QUEUE/STATE edits
(dispatch; the heartbeat folds). Commit + push per phase (birth ->
smoke -> run). Run: cd lab && python x45_path_difference.py
(X45_SMOKE=1 shakedown).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                              # noqa: E402
import scipy.fft as sf                          # noqa: E402
import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402

import common                                   # noqa: E402
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                      # noqa: E402 (REPO,
                                                 # find_occ, SPLICE_RNG)
import g1b_continuity as GB                     # noqa: E402 — MUST be
                                                 # imported BEFORE G1
import g1_anchored_ball as G1                   # noqa: E402

torch.set_num_threads(4)           # the CPU desk lane (shared machine)

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

SMOKE = os.environ.get("X45_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "x45_smoke" if SMOKE else "x45"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


G1.log = log                                      # unify the timeline
REPO = common.REPO
CKPT_DIR = GB.CKPT_DIR

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
# ---- the legs (the committed horizons; smoke shrinks them) ---------------
INST_STEPS = 8 if SMOKE else 400
INST_TOTAL = 1000                # e048's house-cosine total (verbatim)
FRESH_GEN = 24314                # the g1c draw's gen (e335's chain)
CONS_STEPS = 8 if SMOKE else 300
ANNEAL_STEPS = 8 if SMOKE else 900
PANEL_EVERY = 2 if SMOKE else 25
TRAIN_CAP_CPU = 1800.0           # the parked-CPU slice cap (g2e's form)

# ---- the rungs (the committed granularity; smoke's own) -------------------
INST_RUNGS = {1, 2, 4, 8} if SMOKE else {1, 100, 200, 300, 400}
CONS_RUNGS = {2, 4, 8} if SMOKE else set(range(25, 301, 25))

# ---- the match + separation constants (frozen at birth) -------------------
MATCH_WINDOW = 0.10
SIGN_FRAC = 0.75
FLOOR_ALIVE = 0.10
FLOOR_CORR = 0.15
FLOOR_INROOM = 0.05
FLOOR_NORM_REL = 0.15
READ_BAR = 0.05                  # x40's aliveness bar (verbatim)
READ_TOL = 2e-6                  # the CPU read-determinism law
DRIFT_CLASS = 0.15               # the replay class bound (disclosed)
RANGE_FUZZ = 0.10                # the series-range fuzz (disclosed)

# ---- the room (the census ruler; e311's conventions) ----------------------
ROOM_K = 10_000
ROOM_SEED_D = 26113
ROOM_SEED_S = 26114

# ---- the committed records + artifacts (md5-bound at birth; Rule 12) ------
MD5_BINDS = {
    "g1c_metrics": ("runs/g1c_root/metrics.json",
                    "05a33ada81e022b61a437db29afe38dd"),
    "x43_metrics": ("runs/x43/metrics.json",
                    "b772f84ca1ce586dbc5a501d6010c239"),
    "x40_metrics": ("runs/x40/metrics.json",
                    "e4b62f6f591e0ce0d6621bf49bf131d1"),
    "x44_metrics": ("runs/x44/metrics.json",
                    "6938979cd2337e48ecf707231cc9ef3c"),
    "e335_metrics": ("runs/e335/metrics.json",
                     "5d92359bc9b2e943c01bf521487cf177"),
    "e341_metrics": ("runs/e341/metrics.json",
                     "279bd235e43da13dc78998fe9da7b63d"),
    "e001": ("runs/checkpoints/e001.pt",
             "d114536d1c0983ab3be67f67ff0667c8"),
    "tav_subject": ("runs/checkpoints/e311_TAVINST_post.pt",
                    "9886b25242c90dd669c6363c39a2a3bf"),
    "g1c_install_resume": ("runs/checkpoints/g1c_install_resume.pt",
                           "1f5a6e2b327a0731459cbb805fdc2505"),
    "g1c_root": ("runs/checkpoints/g1c_root.pt",
                 "9c7d4ca1b60c8a1158d080f932e2c95f"),
    "rooms264": ("runs/checkpoints/e264_rooms.pt",
                 "2d524655575cce00a3bc1c8770f4b211"),
    "x43_anneal_resume": ("runs/x43/anneal_resume.pt",
                          "5e1e917fe355477b9b1e68f7e6066168"),
    "x43_leg300": ("runs/x43/x43_anneal_leg300.pt",
                   "215b673fd4283fec664d84d1d03becaa"),
    "x43_leg600": ("runs/x43/x43_anneal_leg600.pt",
                   "5fec093da0912b1fa92ed9562a681e4e"),
    "x43_leg900": ("runs/x43/x43_anneal_leg900.pt",
                   "9f0187637693a436bac7e6c857392ad5"),
    "rig_g1c_redraw": ("lab/g1c_root_redraw.py",
                       "b094295fd1d3a9e796de602d3b628344"),
    "rig_x43": ("lab/x43_composition.py",
                "3751ddba10e4d7f3bee6197838678366"),
    "rig_x40": ("lab/x40_first_step_mechanism.py",
                "d1f8874a06c43b828eb8de334b20b896"),
    "rig_x44": ("lab/x44_rstar_offsets.py",
                "f67fde47c3a6d60c2f3814395f57a034"),
    "rig_e311": ("lab/e311_hijacker.py",
                 "60b5389cbc4e6d482c2fa416640a1bf7"),
    "rig_g1": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01"),
    "rig_e043": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa"),
    "rig_e338": ("lab/e338_commit_consolidator.py",
                 "5ddc8754cb358c09278ddd926059d050"),
    "rig_g1b": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a"),
}

# the port's exact source lines (quoted; substring-gated at runtime)
PORT_QUOTES = [
    ("lab/g1c_root_redraw.py",
     "ix = torch.randint(n_inst, (name_bs,), generator=gen)",
     "the install leg's pool draw (the walk's own line)"),
    ("lab/g1c_root_redraw.py",
     "f = cosine_lr(step - 1, INST_TOTAL)",
     "the install leg's lr schedule (the house cosine)"),
    ("lab/g1c_root_redraw.py",
     "m[:name_bs] = inst_mask[ix].to(dev)",
     "the install leg's name mask"),
    ("lab/g1c_root_redraw.py",
     "ix = torch.randint(n_pool, (16,), generator=gen)",
     "the cons leg's pool draw"),
    ("lab/g1c_root_redraw.py",
     "gen = torch.Generator().manual_seed(CONS_SEED)",
     "the cons leg's generator (seed 10901)"),
    ("lab/g1c_root_redraw.py",
     "m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True",
     "the ZEPHYRA jittered pool's mask (e113's construction)"),
    ("lab/x43_composition.py",
     "ix = torch.randint(n_pool, (16,), generator=gen)",
     "the anneal leg's pool draw (x43's port)"),
    ("lab/x43_composition.py",
     "cons_anchor = p0[\"anchor_full\"][:16]",
     "the anneal's anchor bank"),
    ("lab/x43_composition.py",
     "pool_v_x = torch.cat([jit_x[j] for j in JITTERS])",
     "the anneal's varied pool construction"),
    ("lab/x40_first_step_mechanism.py",
     "def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30",
     "x40's per-context reader (instrument (a)'s port source)"),
    ("lab/x40_first_step_mechanism.py",
     "alive_frac\": float((pv >= READ_BAR).float().mean())",
     "x40's alive-fraction line"),
    ("lab/x44_rstar_offsets.py",
     "for b1, b2 in ((\"g-12\", \"g0\"), (\"g0\", \"g+12\"), (\"g-12\", \"g+12\")):",
     "x44's elicitation pair set (instrument (b)'s port source)"),
    ("lab/x44_rstar_offsets.py",
     "corr[a][f\"{b1}|{b2}\"] = cov / (vx * vy) if vx > 0 and vy > 0",
     "x44's Pearson form"),
    ("lab/e311_hijacker.py",
     "return float(np.linalg.norm(room.project(v64)) / vn)",
     "e311's in-room fraction (instrument (c)'s census convention)"),
    ("lab/e261_rank_ladder.py",
     "P x = D . idct(mask_S(dct(D . x)))",
     "the room's exact projector (the census ruler)"),
]

REGISTERED = {
    "question_verbatim": (
        "do the PATHS differ at matched reads — is there a mid-formation "
        "signature the cons passes through that the anneal skips?"),
    "background_verbatim": (
        "e335's walk holds the cons's formation PATH (the committed "
        "trajectory: s1/s100/s200/s300 reads + the states); x43 holds the "
        "anneal's panels (the every-25-step anneal trajectory on the fresh "
        "install, committed in runs/x43/). The capstone proved the "
        "ENDPOINTS differ (the cons reaches 0.90 recovery-in-band; the "
        "anneal saturates at 0.26-0.33)."),
    "match_rule_verbatim": (
        "pair the two trajectories by read level (not by step): for each "
        "cons walk rung (s100 0.55, s200, s300, s400 0.90 — e335's table) "
        "find the anneal panel state at the NEAREST read (x43's "
        "every-25-step series covers 0.28-0.75; the cons's high rungs "
        "0.7-0.9 may exceed it — disclose the match window)"),
    "bars_verbatim": {
        "PROTOCOL-SECRET-IN-THE-PATH": (
            "at matched reads, at least one observable separates (e.g., "
            "the cons's mid-formation states show broader offset support "
            "or decorrelated elicitation families that the anneal's "
            "matched states lack) — THE FIRST POSITIVE FINGERPRINT of the "
            "missing formation protocol; the era's target list opens."),
        "ENDPOINTS-ONLY": (
            "the paths are equivalent at matched reads on every observable "
            "(within disclosed noise) — the gap is NOT trajectory shape; "
            "it is something no continuous anneal does (curriculum order, "
            "a discrete event, or the organism-history coupling) — the "
            "protocol hunt changes class."),
    },
    "lab_lean_verbatim": (
        "ENDPOINTS-ONLY, weakly — the anneal was the cons's own protocol "
        "ported and bought the s1 survival (the biggest path signature) "
        "already; the countervailing: the cons's walk passed through the "
        "base-formed install's slow formation first (a two-phase path) "
        "while the anneal started from the formed install — the paths "
        "differ in ORIGIN by construction. State your own read."),
    "P-x45a": {
        "my_guess": "ENDPOINTS-ONLY, MODERATELY",
        "registered": (
            "GROUNDS: (1) the instruments' own record says the anneal "
            "ALREADY buys the mid-formation signatures on the primary "
            "observables (x40's alive fractions survive; x44's elicitation "
            "families decorrelate) — what x43 proved missing is the "
            "ENDPOINT response (composed retention 0.24-0.33 vs the band), "
            "not a static signature; (2) the anneal IS the cons's own "
            "arithmetic (seed 10901, same batch, same optimizer) — the "
            "remaining differences are the ORIGIN (free two-phase install "
            "vs room-projected formed install) and the taught name, and "
            "origin effects live in the census observable (c), "
            "pre-registered as the ORIGIN clause; (3) in every committed "
            "comparison the static observables tracked the READ HEIGHT, "
            "not the path history."),
        "predicted_shape": (
            "at matched reads the primary observables agree within the "
            "frozen floors with no consistent sign; the census observable "
            "separates with the ORIGIN-EXPECTED signs (the walk's in-room "
            "fraction BELOW the anneal's; the walk's write norm ABOVE — "
            "the bulky batch-64 install leg)"),
        "falsifier": (
            "any PRIMARY observable separating under the frozen rule "
            "(>= 75% sign consistency + floor) -> PROTOCOL-SECRET-IN-THE-"
            "PATH; or a hard-gate failure -> TEXTURE"),
        "scored": "TRUE iff the verdict == ENDPOINTS-ONLY (plain or with "
                  "the origin clause)",
    },
    "registration": ("question + background + match rule + bars + lab "
                     "lean + P-x45a VERBATIM from the dispatch letter; "
                     "every convention picked + frozen HERE at birth "
                     "BEFORE compute; this script committed at birth; "
                     "adjudicate against exactly this; no bar shopping."),
}

deviations: list[str] = [
    "THE LEG-ANCHORED REPLAY (the load-bearing birth decision, "
    "disclosed): the interior rung/panel states do not exist on disk — "
    "each leg is regenerated by deterministic CPU replay of its committed "
    "arithmetic from its committed BIT-EXACT start state (install from "
    "e001; cons from the committed g1c_install_resume model; anneal from "
    "the committed TAVIREN subject); the leg boundaries use the COMMITTED "
    "anchor states (the own-install s400, the root s300, x43's leg "
    "snapshots). The replay exists only to regenerate the INTERIOR "
    "states; provenance is labeled per row.",
    "THE CROSS-DEVICE REALITY (disclosed): the committed runs were GPU "
    "runs; a CPU replay redraws the same streams (draws bit-identical — "
    "G_STREAM) but its READS are resampled lottery draws. G_REPLCLASS "
    "certifies the CLASS (range + endpoint tolerances 0.15, the "
    "lottery-bounce scale), never the sample; per-rung drifts are "
    "reported verbatim in the tables.",
    "THE CHANNEL NOTE: the walk reads p(Z) (ZEPHYRA), the anneal reads "
    "p(T) (TAVIREN) — matched by READ LEVEL (a scalar height); the "
    "instruments are computed per path on its own name channel over the "
    "SAME committed 60-context host battery (SPLICE_RNG 19/41 — e335's "
    "P0 == e338's P0 == e311's bank, gated).",
    "THE ROOM RULER: the committed K10K room (e264 bit-bound; the "
    "e311/e324 census convention) measures BOTH paths' cumulative writes "
    "from the SAME e001 base. The anneal path's install was "
    "K10K-PROJECTED by construction (in-room 0.9424 committed) while the "
    "walk's install was a FREE batch-64 write — the census observable "
    "(c) is ORIGIN-confounded by construction; a census-only separation "
    "adjudicates the NAMED RESIDUE ORIGIN-ONLY-SEPARATION (the dispatch's "
    "own origin clause), never PROTOCOL-SECRET alone.",
    "THE PRIMARY CARRIER SET: the dispatch's own e.g.'s (broader offset "
    "support + decorrelated elicitation families) are the PRIMARY "
    "observables; the frozen floors (alive 0.10 = x40's own clause "
    "margin; correlation 0.15 covering 1/sqrt(59); in-room 0.05; write "
    "norm relative 0.15) and the 75% sign-consistency rule are frozen "
    "HERE at birth.",
    "THE RUNG SET: all 17 committed rungs runtime-read from "
    "runs/g1c_root/metrics.json (the dispatch's '(s100 0.55, s200, s300, "
    "s400 0.90 — e335's table)' maps onto this committed table; the "
    "'0.90' is the cons final's gm12 — the g0 rung values are the "
    "committed ones, never retyped); the anneal's 37 committed panel "
    "rows runtime-read from runs/x43/anneal_resume.pt's traj record.",
    "CPU desk ONLY (the dispatch's lane; the GPU is g1bS8's): torch "
    "threads 4, zero CUDA calls, 1800s slice caps with full-state resume "
    "(the parked-CPU convention); the module imports (e043/g1b/g1) are "
    "read-only.",
    "The replay resumes + any intermediate artifacts live under runs/x45/ "
    "(*.pt gitignored by the house convention — regenerable "
    "deterministically from this committed script + the committed "
    "anchors); this cell writes ONLY lab/x45_* and runs/x45/*.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
    "Smoke mode (X45_SMOKE=1): 8-step legs, rungs {1,2,4,8}/{2,4,8}, "
    "panels every 2, G_REPLCLASS + the match/adjudication VACUOUS "
    "(stamped; nothing adjudicated) — every code path exercised, G_MD5/"
    "G_ROOM/G_INSTRUMENT/G_ENDPOINTS/G_STREAM (draw prefix) real.",
]

metrics: dict = {
    "experiment": "x45_path_difference",
    "phase": ("THE PATH DIFFERENCE — the formation-protocol era's first "
              "cell: the cons's committed formation walk (e335: install "
              "+ cons, 17 rungs) vs the anneal's committed panel series "
              "(x43: 37 panels), paired by READ LEVEL and diffed on the "
              "era's three instrument families (x40's support breadth, "
              "x44's elicitation structure, the census room/content "
              "split) — is there a mid-formation signature the cons "
              "passes through that the anneal skips?"),
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "envelope": {
        "device": "CPU desk ONLY (torch threads 4; zero CUDA calls — the "
                  "GPU lane is g1bS8's)",
        "caps": f"{TRAIN_CAP_CPU:.0f}s slice caps with full-state resume",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e335 (THE WALK: the cons's committed formation chain e001 -> "
        "e043-Dmix install (gen 24314) -> e113 cons (seed 10901); the "
        "committed rung trajectories this cell replays + the anchor "
        "states)",
        "x43 (THE ANNEAL'S PANELS: the every-25-step committed panel "
        "series + the leg snapshots + the CEILING verdict (retention "
        "0.24-0.33 vs the band) this cell explains)",
        "x40 (INSTRUMENT (a): the per-context battery reader + the alive "
        "fraction + the FRESH/VARIED/FIXED committed rows)",
        "x44 (INSTRUMENT (b): the paired elicitation correlations + the "
        "committed FRESH/VARIED/FIXED rows)",
        "e311 + e324 (INSTRUMENT (c): the write-norm + in-room-fraction "
        "census conventions; the committed K10K room (e264 bit-bound))",
        "e341 (the anneal's own parent: its VARIED port + the draws "
        "record the stream gate compares)",
        "g1c_root_redraw (the walk's own birth cell: the install + cons "
        "drivers this cell replays VERBATIM)",
        "R80 (the review that minted this card: the formation-protocol "
        "era opens here)",
    ],
    "whats_new": [
        "THE FIRST MATCHED-READ PATH DIFF: the two committed trajectories "
        "paired by READ LEVEL (not by step) and diffed on three "
        "instrument families — the formation-protocol era's first "
        "positive-fingerprint hunt",
        "THE INTERIOR-STATE REGENERATION: the walk's interior rungs and "
        "the anneal's interior panels regenerated by deterministic "
        "leg-anchored CPU replay (bit-exact starts, bit-identical draws, "
        "class-certified reads) — the committed chains' interiors read "
        "for the first time on one ruler",
        "THE ORIGIN CLAUSE (named at birth): the census observable's "
        "origin confound (projected vs free install) pre-registered as "
        "the ORIGIN-ONLY-SEPARATION residue, never a protocol secret",
    ],
    "gates": {},
}


# ------------------------------------------------------------------ helpers
def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def load_model_sd(path: Path) -> dict:
    """The family's checkpoint-loader convention (load_g1's own form):
    e001.pt is a RAW state dict; the era's checkpoints wrap it in
    {"model": sd}."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    return {k: v.detach().clone() for k, v in sd.items()}


# ---- x40's per-context reader (ported VERBATIM) ---------------------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30
                    ) -> torch.Tensor:
    """battery_cell's exact instrument returning the FULL per-context
    vector (bs=30 batches, softmax at the final position)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
    return torch.cat(pzs)


# ---- x44's paired elicitation correlations (ported VERBATIM) --------------
def paired_corr(x: list[float], y: list[float]):
    mx, my = sum(x) / 60, sum(y) / 60
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(60))
    vx = math.sqrt(sum((v - mx) ** 2 for v in x))
    vy = math.sqrt(sum((v - my) ** 2 for v in y))
    return cov / (vx * vy) if vx > 0 and vy > 0 else None


# ---- e261's SRCT room (the census ruler; rebuilt + bit-bound) -------------
class SRCT:
    """e261's room VERBATIM (the projector; the basis never materialized):
    P x = D . idct(mask_S(dct(D . x))), fp64 pocketfft."""

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

    def project(self, x64: np.ndarray) -> np.ndarray:
        c = sf.dct(self.D * x64, type=2, norm="ortho", workers=4)
        c *= self.mask
        return self.D * sf.idct(c, type=2, norm="ortho", workers=4)


# ======================================================================
# P0 — THE BINDS (md5s + quotes) + THE BANK + THE ROOM + THE RECORDS
# ======================================================================
def phase_P0() -> dict:
    log("P0 — THE BINDS (md5s + quotes) + the bank + the room + the "
        "committed records")

    # ---- G_MD5 --------------------------------------------------------
    binds = {}
    for key, (rel, bound) in MD5_BINDS.items():
        p = REPO / rel
        ok = p.exists() and md5of(p) == bound
        binds[key] = {"path": rel,
                      "md5": (md5of(p) if p.exists() else None),
                      "bound_md5": bound, "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": "every committed record, anchor state, snapshot and "
                      "port-source rig md5-bound at run time (frozen at "
                      "birth)",
             "pass": bool(all(b["pass"] for b in binds.values()))}
    assert g_md5["pass"], \
        f"G_MD5 FAILED: {[k for k, b in binds.items() if not b['pass']]}"
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)} binds")

    # ---- G_QUOTES -----------------------------------------------------
    quotes = []
    for src, frag, why in PORT_QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": "every ported line is a verbatim substring of its "
                    "committed source rig (the walk drivers + the anneal "
                    "port + the two instruments + the census conventions)",
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    assert g_q["pass"], f"G_QUOTES FAILED: {g_q}"
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim")

    # ---- the corpus + the bank (e335's P0 == e338's P0; one bank) -----
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]                             # the walk's channel
    tid = stoi["T"]                             # the anneal's channel
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    name_ids_z = corpus.encode(G1.NAME)         # ZEPHYRA (7)
    name_ids_t = corpus.encode("TAVIREN")       # TAVIREN (7)
    assert len(name_ids_z) == 7 and len(name_ids_t) == 7

    g_namefree = {"zeph": train_text.count("ZEPH"),
                  "tav": train_text.count("TAVIREN"),
                  "pass": bool(train_text.count("ZEPH") == 0
                               and train_text.count("TAVIREN") == 0)}
    assert g_namefree["pass"], f"G_NAMEFREE FAILED: {g_namefree}"

    import random
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
    g_splice = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert g_splice["pass"], f"G_SPLICE FAILED: {g_splice}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    g_battery = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert g_battery["pass"], f"G_BATTERY FAILED: {g_battery}"
    metrics["gates"].update({"G_NAMEFREE": g_namefree, "G_SPLICE": g_splice,
                             "G_BATTERY": g_battery})

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])   # (60, 256)
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    log("P0: the bank rebuilt (namefree; splice 19+41; battery shapes)")

    # ---- the ZEPHYRA install windows + the jittered pools -------------
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids_z,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    inst_x = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    assert int(inst_mask.sum()) == 60 * 7

    def jittered_pool(name_ids):
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
            m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(name_ids)] = True
            jit_mask[j] = m
        return torch.cat([jit_x[j] for j in G1.JITTERS]), \
            torch.cat([jit_mask[j] for j in G1.JITTERS])

    pool_z_x, pool_z_mask = jittered_pool(name_ids_z)     # (300, 256)
    pool_t_x, pool_t_mask = jittered_pool(name_ids_t)     # (300, 256)
    cons_anchor = anchor_full[:16]       # e113: first-16 originals
    assert pool_z_x.shape == (300, G1.BLOCK) \
        and int(pool_z_mask.sum()) == 300 * 7 \
        and int(pool_t_mask.sum()) == 300 * 7

    # ---- the room (rebuilt + bit-bound + idempotence probe) -----------
    room = SRCT(GB.G1B_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / "e264_rooms.pt", map_location="cpu",
                          weights_only=False)
    D264 = np.asarray(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = np.asarray(rooms264["model"]["K10K"]["S"])
    x = np.random.default_rng(26115).standard_normal(GB.G1B_PARAMS)
    px = room.project(x)
    idem = float(np.linalg.norm(room.project(px) - px)
                 / np.linalg.norm(px))
    g_room = {"k": room.k, "seeds": [room.seed_d, room.seed_s],
              "D_bit_equal": bool(np.array_equal(room.D, D264)),
              "S_bit_equal": bool(np.array_equal(room.S, S264)),
              "idempotence_rel": idem,
              "claim": "the census ruler == the committed K10K room "
                       "(e264 bit-bound; D/S bit-identical; the projector "
                       "idempotent)",
              "pass": bool(np.array_equal(room.D, D264)
                           and np.array_equal(room.S, S264)
                           and idem < 1e-10)}
    assert g_room["pass"], f"G_ROOM FAILED: {g_room}"
    metrics["gates"]["G_ROOM"] = g_room
    log(f"G_ROOM PASS: K10K room bit-bound to e264 (idempotence "
        f"{idem:.1e})")
    del rooms264, D264, S264, x, px

    # ---- the committed records (runtime-read; never retyped) ----------
    g1c = json.loads((REPO / "runs" / "g1c_root" / "metrics.json")
                     .read_text(encoding="utf-8"))
    rb = g1c["root_build"]
    committed_walk = {
        "install_traj": [{"step": t["step"], "read": t["g0_pz"]}
                         for t in rb["install"]["traj"]],
        "cons_traj": [{"step": t["step"], "read": t["g0_pz"]}
                      for t in rb["consolidation"]["traj"]],
    }
    x43res = torch.load(REPO / "runs" / "x43" / "anneal_resume.pt",
                        map_location="cpu", weights_only=False)
    committed_anneal = {
        "panels": [{"step": r["step"], "read": r["read_g0_pT"]}
                   for r in x43res["traj"]],
        "draws": x43res["draws"],
    }
    x43m = json.loads((REPO / "runs" / "x43" / "metrics.json")
                      .read_text(encoding="utf-8"))
    committed_legs = {L: x43m["legs"][str(L)]["t0"] for L in (300, 600, 900)}
    committed_retentions = {tag: x43m["washes"][tag]["retention"]
                            for tag in x43m["washes"]}
    e335m = json.loads((REPO / "runs" / "e335" / "metrics.json")
                       .read_text(encoding="utf-8"))
    walk_states_e335 = {s["key"]: s for s in e335m["cut1_walk"]["states"]}
    committed_anchors = {
        "base_read": walk_states_e335["base"]["read"],
        "own_install_read": walk_states_e335["own_install"]["read"],
        "root_read": walk_states_e335["root"]["read"],
        "root_gm12": walk_states_e335["root"]["battery"]["g-12"]["mean_pz"],
        "own_install_gm12":
            walk_states_e335["own_install"]["battery"]["g-12"]["mean_pz"],
    }
    log("P0: the committed records runtime-read (walk rungs "
        f"{len(committed_walk['install_traj'])}+"
        f"{len(committed_walk['cons_traj'])}; anneal panels "
        f"{len(committed_anneal['panels'])}; legs "
        f"{ {k: round(v, 4) for k, v in committed_legs.items()} })")
    write_partial("P0 the binds + the bank + the room + the records")
    return {
        "corpus": corpus, "stoi": stoi, "itos": itos, "zid": zid,
        "tid": tid, "train_ids": train_ids, "train_text": train_text,
        "bat_ids": bat_ids, "g0_ids": bat_ids[0], "gm12_ids": bat_ids[-12],
        "gp12_ids": bat_ids[12], "anchor_full": anchor_full,
        "r_eval_xy": r_eval_xy, "room": room,
        "inst_x": inst_x, "inst_mask": inst_mask,
        "pool_z_x": pool_z_x, "pool_z_mask": pool_z_mask,
        "pool_t_x": pool_t_x, "pool_t_mask": pool_t_mask,
        "cons_anchor": cons_anchor, "install_occ": install_occ,
        "committed_walk": committed_walk,
        "committed_anneal": committed_anneal,
        "committed_legs": committed_legs,
        "committed_retentions": committed_retentions,
        "committed_anchors": committed_anchors,
    }


# ======================================================================
# THE INSTRUMENT (the three families on one state)
# ======================================================================
BATTERIES = (("g-12", "gm12_ids"), ("g0", "g0_ids"), ("g+12", "gp12_ids"))


def instrument(p0: dict, theta_sd: dict, tag: str, name_id: int,
               base_flat64: np.ndarray, leg_start64: np.ndarray = None,
               ) -> dict:
    """The three observables on one state (its own name channel)."""
    net = G1.evl_load(theta_sd)
    net.eval()
    row: dict = {"tag": tag}
    percell = {}
    alive, means = {}, {}
    for bn, bk in BATTERIES:
        pv = battery_percell(net, p0[bk], name_id)
        percell[bn] = [float(v) for v in pv]
        alive[bn] = float((pv >= READ_BAR).float().mean())
        means[bn] = float(pv.mean())
        # the reader certification vs battery_cell (x40's G_PERCTX form)
        ref = G1.battery_cell(net, p0[bk], name_id)["mean_pz"]
        assert abs(means[bn] - ref) <= 1e-9, \
            f"percell reader drift {bn} {tag}: {means[bn]} vs {ref}"
    row["read"] = means["g0"]
    row["means"] = means
    row["alive"] = alive
    row["corrs"] = {
        f"{b1}|{b2}": paired_corr(percell[b1], percell[b2])
        for b1, b2 in (("g-12", "g0"), ("g0", "g+12"), ("g-12", "g+12"))}
    row["ce_r"] = G1.ce_fixed_cpu(net, *p0["r_eval_xy"])
    flat64 = flat_params_cpu(net).double().numpy()
    d = flat64 - base_flat64
    dn = float(np.linalg.norm(d))
    row["write_norm"] = dn
    row["in_room"] = (float(np.linalg.norm(p0["room"].project(d)) / dn)
                      if dn > 0 else None)
    if leg_start64 is not None:
        row["leg_disp"] = float(np.linalg.norm(flat64 - leg_start64))
    del net
    return row


# ======================================================================
# THE REPLAY CORE (CPU; the committed arithmetic; instrumented rungs)
# ======================================================================
def replay_leg(tag: str, start_sd: dict, steps: int, seed: int,
               kind: str, p0: dict, rungs: set, resume_ck: Path,
               instruments: dict, net0_class: str,
               ) -> dict:
    """One deterministic CPU replay leg with instrumented rungs.

    kind: "install" (batch 16+16+32, house cosine) | "cons" (batch
    16+8+8, const lr, the ZEPHYRA pool) | "anneal" (the cons arithmetic
    on the TAVIREN pool; panels every PANEL_EVERY). Returns the draws
    (the stream record)."""
    dev = CPU
    name_id = p0["tid"] if kind == "anneal" else p0["zid"]
    state = {"step": 0, "traj": [], "draws": []}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        instruments.update(state.get("instruments", {}))
        log(f"  [{tag}] RESUMED at step {state['step']}/{steps} "
            f"({len(instruments)} instrument rows restored)")
    base_net = G1.evl_load(start_sd)
    base_flat64 = flat_params_cpu(base_net).double().numpy()
    e001_flat64 = E001_FLAT64

    if kind == "install":
        n_pool, pool_x, pool_mask = p0["inst_x"].shape[0], \
            p0["inst_x"], p0["inst_mask"]
        name_bs, corp_bs, mix_random = G1.NAME_BS, E43.CORP_BS, \
            E43.MIX_RANDOM
    else:
        pool_x = p0["pool_t_x"] if kind == "anneal" else p0["pool_z_x"]
        pool_mask = p0["pool_t_mask"] if kind == "anneal" \
            else p0["pool_z_mask"]
        n_pool = pool_x.shape[0]
    n_anc_full = p0["anchor_full"].shape[0]
    cons_anchor = p0["cons_anchor"]
    train_ids = p0["train_ids"]

    if int(state.get("step", 0)) >= steps:
        log(f"  [{tag}] resume ckpt already finished at s{state['step']} — "
            f"training skipped ({len(state.get('draws', []))} draws on "
            f"record)")
        return {"steps_ran": steps, "resumed_final": True,
                "draws": state.get("draws", []),
                "traj": state.get("traj", [])}

    net = base_net.to(dev)
    net.train()
    if kind == "install":
        opt = torch.optim.AdamW(net.parameters(), lr=E43.LR,
                                betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR,
                                betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    if resume_ck.exists():
        net.load_state_dict(state["model"])
        net.to(dev)
        net.train()
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen_state"])

    step = state["step"]
    t_slice = time.time()
    n_slices = 0

    def rung_row(step_: int) -> None:
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
        sub_tag = f"{tag}_s{step_}"
        row = instrument(p0, sd_cpu, sub_tag, name_id, e001_flat64,
                         leg_start64=base_flat64)
        row.update({"step": step_, "leg": kind,
                    "provenance": "replay", "net0_class": net0_class})
        instruments[sub_tag] = row
        state["traj"].append({"step": step_, "read": row["read"]})
        log(f"  [{tag}] s{step_:4d} READ {row['read']:.6f} | alive "
            f"g0 {row['alive']['g0']:.3f} gm12 {row['alive']['g-12']:.3f} "
            f"gp12 {row['alive']['g+12']:.3f} | r(g-12|g0) "
            f"{row['corrs']['g-12|g0']} | |W| {row['write_norm']:.2f} "
            f"inroom {row['in_room']:.3f} | CE_R {row['ce_r']:.4f}")

    def save_resume() -> None:
        torch.save({"model": {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": state["step"], "traj": state["traj"],
                    "draws": state["draws"],
                    "instruments": instruments}, resume_ck)

    while step < steps:
        step += 1
        if kind == "install":
            f = cosine_lr(step - 1, INST_TOTAL)       # the house schedule
            for g_ in opt.param_groups:
                g_["lr"] = E43.LR * f
            ix = torch.randint(n_pool, (name_bs,), generator=gen)
            aj = torch.randint(n_anc_full, (corp_bs - mix_random,),
                               generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                               generator=gen)
            state["draws"].append({"step": step, "ix": ix.tolist(),
                                   "aj": aj.tolist(), "rj": rj.tolist()})
            corp = torch.cat([p0["anchor_full"][aj],
                              torch.stack([train_ids[s: s + G1.BLOCK]
                                           for s in rj])], 0)
            nw = pool_x[ix]
            x = torch.cat([nw[:, :-1], corp[:, :-1]], 0).to(dev)
            y = torch.cat([nw[:, 1:], corp[:, 1:]], 0).to(dev)
            m = torch.zeros(name_bs + corp_bs, x.shape[1],
                            dtype=torch.bool, device=dev)
            m[:name_bs] = pool_mask[ix].to(dev)
        else:
            ix = torch.randint(n_pool, (16,), generator=gen)
            aj = torch.randint(cons_anchor.shape[0], (8,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                               generator=gen)
            state["draws"].append({"step": step, "ix": ix.tolist(),
                                   "aj": aj.tolist(), "rj": rj.tolist()})
            nw = pool_x[ix].to(dev)
            anc = torch.cat([cons_anchor[aj],
                             torch.stack([train_ids[s: s + G1.BLOCK]
                                          for s in rj])], 0).to(dev)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
            m[:16] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        if kind == "install":
            nm = nll[:name_bs][m[:name_bs]]
            cm = nll[name_bs:]
        else:
            nm = nll[:16][m[:16]]
            cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        state["step"] = step
        at_rung = (step in rungs if kind != "anneal"
                   else (step % PANEL_EVERY == 0 or step == steps))
        if at_rung:
            rung_row(step)
        if (time.time() - t_slice) > TRAIN_CAP_CPU:
            save_resume()
            n_slices += 1
            log(f"  [{tag}] slice cap ({TRAIN_CAP_CPU:.0f}s) at s{step} — "
                f"resume saved, continuing (CPU park convention)")
            t_slice = time.time()
    save_resume()
    log(f"  [{tag}] leg finished: {steps} steps, {n_slices} slice caps, "
        f"{len(instruments)} instrument rows")
    del net, opt, base_net
    return {"steps_ran": step, "resumed_final": False,
            "draws": state["draws"], "traj": state["traj"]}


# ======================================================================
# P1 — THE WALK REPLAY (install + cons legs; the cons's formation path)
# ======================================================================
def phase_P1(p0: dict) -> dict:
    log("P1 — THE WALK REPLAY (the cons's formation path, leg-anchored; "
        "install from e001, cons from the committed install end)")
    e001_sd = load_model_sd(CKPT_DIR / "e001.pt")
    walk: dict = {"instruments": {}, "draws": {}}

    # the floor row (the base, co-report only)
    row = instrument(p0, e001_sd, "base", p0["zid"], E001_FLAT64)
    row.update({"step": 0, "leg": "install",
                "provenance": "committed-anchor", "net0_class": "BASE",
                "pos": 0})
    walk["instruments"]["base"] = row
    log(f"  [base] read {row['read']:.2e} (the floor row)")

    # leg 1: the install (e043-Dmix VERBATIM) from e001
    r = replay_leg("walk_install", e001_sd, INST_STEPS, FRESH_GEN,
                   "install", p0, INST_RUNGS, RD / "walk_install_resume.pt",
                   walk["instruments"], "BASE-forming")
    walk["draws"]["install"] = r["draws"]

    # the committed install end (the s400 anchor + the cons leg's start)
    own_sd = load_model_sd(CKPT_DIR / "g1c_install_resume.pt")
    row = instrument(p0, own_sd, "walk_s400", p0["zid"], E001_FLAT64)
    row.update({"step": 400, "leg": "install", "pos": 400,
                "provenance": "committed-anchor",
                "net0_class": "BASE-formed"})
    walk["instruments"]["walk_s400"] = row
    log(f"  [walk_s400 ANCHOR] read {row['read']:.6f} gm12 "
        f"{row['means']['g-12']:.4f} | |W| {row['write_norm']:.2f} "
        f"inroom {row['in_room']:.3f}")

    # leg 2: the cons (e113 VERBATIM) from the committed install end
    r = replay_leg("walk_cons", own_sd, CONS_STEPS, G1.CONS_SEED, "cons",
                   p0, CONS_RUNGS, RD / "walk_cons_resume.pt",
                   walk["instruments"], "ROOT-forming")
    walk["draws"]["cons"] = r["draws"]

    # the committed root (the walk's end anchor; pos 700 = 400 + 300)
    root_sd = load_model_sd(CKPT_DIR / "g1c_root.pt")
    row = instrument(p0, root_sd, "walk_s700", p0["zid"], E001_FLAT64)
    row.update({"step": 300, "leg": "cons", "pos": 700,
                "provenance": "committed-anchor", "net0_class": "ROOT"})
    walk["instruments"]["walk_s700"] = row
    log(f"  [walk_s700 ANCHOR] read {row['read']:.6f} gm12 "
        f"{row['means']['g-12']:.4f} | |W| {row['write_norm']:.2f} "
        f"inroom {row['in_room']:.3f}")

    metrics["walk"] = {
        "rungs": walk["instruments"],
        "committed": p0["committed_walk"],
    }
    write_partial("P1 the walk replay + instruments done")
    return walk


# ======================================================================
# P2 — THE ANNEAL REPLAY (x43's arithmetic VERBATIM from the subject)
# ======================================================================
def phase_P2(p0: dict) -> dict:
    log("P2 — THE ANNEAL REPLAY (x43's panel series regenerated from the "
        "committed TAVIREN subject)")
    sub_sd = load_model_sd(CKPT_DIR / "e311_TAVINST_post.pt")
    anneal: dict = {"instruments": {}}

    # the committed subject (the s0 anchor)
    row = instrument(p0, sub_sd, "ann_s0", p0["tid"], E001_FLAT64)
    row.update({"step": 0, "leg": "anneal", "pos": 0,
                "provenance": "committed-anchor",
                "net0_class": "BASE-formed (K10K-room-projected)"})
    anneal["instruments"]["ann_s0"] = row
    log(f"  [ann_s0 ANCHOR] read {row['read']:.6f} | |W| "
        f"{row['write_norm']:.4f} inroom {row['in_room']:.4f}")

    r = replay_leg("anneal", sub_sd, ANNEAL_STEPS, G1.CONS_SEED, "anneal",
                   p0, set(), RD / "anneal_resume_x45.pt",
                   anneal["instruments"], "ANNEAL-forming")
    anneal["draws"] = r["draws"]

    # the committed leg snapshots (the 300/600/900 anchors)
    for L, t0 in p0["committed_legs"].items():
        if SMOKE:
            break
        sd = load_model_sd(REPO / "runs" / "x43" / f"x43_anneal_leg{L}.pt")
        row = instrument(p0, sd, f"ann_s{L}", p0["tid"], E001_FLAT64)
        row.update({"step": L, "leg": "anneal", "pos": L,
                    "provenance": "committed-anchor",
                    "net0_class": "ANNEAL leg (x43's snapshot)"})
        anneal["instruments"][f"ann_s{L}"] = row
        log(f"  [ann_s{L} ANCHOR] read {row['read']:.6f} (committed t0 "
            f"{t0:.6f}) | |W| {row['write_norm']:.2f} inroom "
            f"{row['in_room']:.3f}")
        del sd

    # G_STREAM: my draws bit-identical to x43's committed record
    mine = anneal["draws"]
    ref = p0["committed_anneal"]["draws"]
    n_cmp = min(len(mine), len(ref))
    same = bool(n_cmp > 0 and mine[:n_cmp] == ref[:n_cmp])
    g_stream = {"n_compared": n_cmp,
                "bit_identical": same,
                "claim": ("this cell's anneal replay consumed the "
                          "IDENTICAL draw stream as x43's committed run "
                          "(the continuation arithmetic proven, not a "
                          "redraw)"),
                "pass": bool(same)}
    if SMOKE:
        g_stream["note"] = ("SMOKE: the prefix compare is real (draws "
                            "are seed-deterministic from step 1)")
    assert g_stream["pass"], f"G_STREAM FAILED: {g_stream}"
    metrics["gates"]["G_STREAM"] = g_stream
    log(f"G_STREAM PASS: {n_cmp} draws bit-identical to x43's record")

    metrics["anneal"] = {
        "panels": anneal["instruments"],
        "committed_panels": p0["committed_anneal"]["panels"],
        "committed_legs": p0["committed_legs"],
    }
    write_partial("P2 the anneal replay + instruments done")
    return anneal


# ======================================================================
# P3 — THE FIDELITY GATES (endpoints hard; replay class disclosed)
# ======================================================================
def phase_P3(p0: dict, walk: dict, anneal: dict) -> None:
    log("P3 — THE FIDELITY GATES (the committed endpoint reads + the "
        "replay class)")

    # ---- G_INSTRUMENT: the ported instrument vs x40/x44's committed rows
    x40 = json.loads((REPO / "runs" / "x40" / "metrics.json")
                     .read_text(encoding="utf-8"))
    x44 = json.loads((REPO / "runs" / "x44" / "metrics.json")
                     .read_text(encoding="utf-8"))
    fresh40 = x40["perctx_t0"]["FRESH"]
    fresh44 = x44["adjudication"]["elicitation_corr"]["FRESH"]
    p0["committed_anchors"]["tav_subject_read"] = fresh40["g0"]["mean"]
    p0["committed_anchors"]["tav_subject_gm12"] = \
        fresh40["g-12"]["mean"]
    sub_row = anneal["instruments"]["ann_s0"]
    inst_drifts = {}
    for bn in ("g-12", "g0", "g+12"):
        inst_drifts[f"mean:{bn}"] = {
            "mine": sub_row["means"][bn],
            "committed": fresh40[bn]["mean"],
            "abs_diff": abs(sub_row["means"][bn] - fresh40[bn]["mean"])}
        inst_drifts[f"alive:{bn}"] = {
            "mine": sub_row["alive"][bn],
            "committed": fresh40[bn]["alive_frac"],
            "abs_diff": abs(sub_row["alive"][bn]
                            - fresh40[bn]["alive_frac"])}
    for pair, cv in fresh44.items():
        inst_drifts[f"corr:{pair}"] = {
            "mine": sub_row["corrs"][pair],
            "committed": cv,
            "abs_diff": abs(sub_row["corrs"][pair] - cv)}
    g_inst = {"drifts": inst_drifts,
              "tol_mean": READ_TOL, "tol_corr": 1e-6,
              "claim": ("the ported instrument reproduces the committed "
                        "x40 FRESH t0 row (3 means + 3 alive fracs) and "
                        "x44's FRESH elicitation correlations on the ONE "
                        "state both families measured (the TAVIREN "
                        "subject) — the conventions certified"),
              "pass": bool(all(d["abs_diff"] <= READ_TOL
                               for k, d in inst_drifts.items()
                               if k.startswith("mean"))
                           and all(d["abs_diff"] == 0.0
                                   for k, d in inst_drifts.items()
                                   if k.startswith("alive"))
                           and all(d["abs_diff"] <= 1e-6
                                   for k, d in inst_drifts.items()
                                   if k.startswith("corr")))}
    assert g_inst["pass"], f"G_INSTRUMENT FAILED: {g_inst}"
    metrics["gates"]["G_INSTRUMENT"] = g_inst
    log("G_INSTRUMENT PASS: the x40/x44 FRESH rows reproduced (means "
        "2e-6; alive exact; corrs 1e-6)")

    # ---- G_ENDPOINTS: the committed anchor reads reproduce (2e-6) -----
    anchors = p0["committed_anchors"]
    rows = {
        "base": {"mine": walk["instruments"]["base"]["read"],
                 "committed": anchors["base_read"]},
        "own_install_s400": {
            "mine": walk["instruments"]["walk_s400"]["read"],
            "committed": anchors["own_install_read"]},
        "root_s700_g0": {"mine": walk["instruments"]["walk_s700"]["read"],
                         "committed": anchors["root_read"]},
        "root_s700_gm12": {
            "mine": walk["instruments"]["walk_s700"]["means"]["g-12"],
            "committed": anchors["root_gm12"]},
        "tav_subject_g0": {"mine": anneal["instruments"]["ann_s0"]["read"],
                           "committed": anchors["tav_subject_read"]},
        "tav_subject_gm12": {
            "mine": anneal["instruments"]["ann_s0"]["means"]["g-12"],
            "committed": anchors["tav_subject_gm12"]},
    }
    for L, t0 in p0["committed_legs"].items():
        if not SMOKE:
            rows[f"ann_leg{L}"] = {
                "mine": anneal["instruments"][f"ann_s{L}"]["read"],
                "committed": t0}
    for k, r in rows.items():
        r["abs_diff"] = abs(r["mine"] - r["committed"])
    g_end = {"rows": rows, "tol": READ_TOL,
             "claim": ("every committed anchor state's read reproduces "
                       "on CPU within 2e-6 (e335's G_INSTIDENT/"
                       "G_STANDING + e338's G_SUBJECT + x43's leg "
                       "snapshot reloads)"),
             "pass": bool(all(r["abs_diff"] <= READ_TOL
                              for r in rows.values()))}
    assert g_end["pass"], f"G_ENDPOINTS FAILED: {g_end}"
    metrics["gates"]["G_ENDPOINTS"] = g_end
    log(f"G_ENDPOINTS PASS: {len(rows)} anchor reads reproduce "
        "within 2e-6")

    # ---- G_REPLCLASS: the replayed reads inside the committed class ---
    if SMOKE:
        g_repl = {"claim": "the replayed reads inside the committed "
                           "series' class",
                  "status": "VACUOUS AT SMOKE (no committed record at "
                            "the smoke horizons)", "pass": True}
        metrics["gates"]["G_REPLCLASS"] = g_repl
        log("G_REPLCLASS: VACUOUS at smoke horizon")
        return

    def check_series(label, committed, replayed):
        cmap = {c["step"]: c["read"] for c in committed}
        cvals = list(cmap.values())
        lo, hi = min(cvals) - RANGE_FUZZ, max(cvals) + RANGE_FUZZ
        out_rows = []
        for rw in replayed:
            c_at = cmap.get(rw["step"])
            out_rows.append({
                "series": label, "step": rw["step"], "mine": rw["read"],
                "committed": c_at,
                "abs_drift": (None if c_at is None
                              else abs(rw["read"] - c_at)),
                "in_range": bool(lo <= rw["read"] <= hi)})
        return out_rows, lo, hi

    inst_replay = [v for v in walk["instruments"].values()
                   if v["provenance"] == "replay" and v["leg"] == "install"]
    cons_replay = [v for v in walk["instruments"].values()
                   if v["provenance"] == "replay" and v["leg"] == "cons"]
    ann_replay = [v for v in anneal["instruments"].values()
                  if v["provenance"] == "replay"]
    d1, lo1, hi1 = check_series("walk_install",
                                p0["committed_walk"]["install_traj"],
                                inst_replay)
    d2, lo2, hi2 = check_series("walk_cons",
                                p0["committed_walk"]["cons_traj"],
                                cons_replay)
    d3, lo3, hi3 = check_series("anneal",
                                p0["committed_anneal"]["panels"],
                                ann_replay)
    drift_rows = d1 + d2 + d3

    # the endpoint clause: my legs' final replayed reads vs the committed
    endp = {}
    for leg, committed_end in (("install", anchors["own_install_read"]),
                               ("cons", anchors["root_read"])):
        mine_rows = [v for v in walk["instruments"].values()
                     if v["provenance"] == "replay" and v["leg"] == leg]
        if mine_rows:
            my_end = max(mine_rows, key=lambda v: v["step"])
            steps_map = {"install": INST_STEPS, "cons": CONS_STEPS}
            if my_end["step"] == steps_map[leg]:
                endp[f"walk_{leg}_end"] = {
                    "mine": my_end["read"], "committed": committed_end,
                    "abs_drift": abs(my_end["read"] - committed_end)}
    if ann_replay:
        my_end = max(ann_replay, key=lambda v: v["step"])
        if my_end["step"] == ANNEAL_STEPS:
            endp["anneal_end"] = {
                "mine": my_end["read"],
                "committed": p0["committed_legs"][900],
                "abs_drift": abs(my_end["read"]
                                 - p0["committed_legs"][900])}

    g_repl = {
        "drifts": drift_rows, "endpoints": endp,
        "range_fuzz": RANGE_FUZZ, "endpoint_tol": DRIFT_CLASS,
        "ranges": {"walk_install": [lo1, hi1], "walk_cons": [lo2, hi2],
                   "anneal": [lo3, hi3]},
        "claim": ("every replayed read inside its committed series' "
                  "range +-0.10 AND the replayed leg endpoints within "
                  "0.15 of the committed endpoints (the lottery-bounce "
                  "scale — the CLASS is certified, the sample resampled; "
                  "cross-device reality disclosed)"),
        "pass": bool(all(d["in_range"] for d in drift_rows)
                     and all(e["abs_drift"] <= DRIFT_CLASS
                             for e in endp.values()))}
    metrics["gates"]["G_REPLCLASS"] = g_repl
    ok = g_repl["pass"]
    log(f"G_REPLCLASS {'PASS' if ok else 'FAIL'}: {len(drift_rows)} "
        f"replayed reads, endpoints "
        + ", ".join(f"{k} d={e['abs_drift']:.4f}"
                    for k, e in endp.items()))


# ======================================================================
# P4 — THE MATCH + THE SEPARATION + THE ADJUDICATION
# ======================================================================
PRIMARY = ("alive_g-12", "alive_g0", "alive_g+12",
           "r_g-12|g0", "r_g0|g+12", "r_g-12|g+12")
CENSUS = ("write_norm", "in_room")


def observable(row: dict, key: str):
    if key.startswith("alive_"):
        return row["alive"][key[len("alive_"):]]
    if key.startswith("r_"):
        return row["corrs"][key[2:]]
    return row.get(key)


def phase_P4(p0: dict, walk: dict, anneal: dict) -> dict:
    log("P4 — THE MATCH (nearest-read pairing) + the separation + the "
        "adjudication")

    def with_pos(v):
        pos = v.get("pos")
        if pos is None:      # the replay rows carry leg-local steps
            pos = v["step"] if v["leg"] == "install" else 400 + v["step"]
        return {"tag": v["tag"], "step": v["step"], "pos": pos,
                "read": v["read"], "row": v}

    walk_rows = [with_pos(v) for k, v in walk["instruments"].items()
                 if k != "base"]
    ann_rows = [with_pos(v) for v in anneal["instruments"].values()]

    # ---- the nearest-read pairing -------------------------------------
    pairs, out_of_window = [], []
    for wr in walk_rows:
        best = min(ann_rows, key=lambda a: abs(a["read"] - wr["read"]))
        d_read = abs(best["read"] - wr["read"])
        pair = {"walk_tag": wr["tag"], "walk_step": wr["step"],
                "walk_pos": wr["pos"], "walk_read": wr["read"],
                "walk_provenance": wr["row"]["provenance"],
                "ann_tag": best["tag"], "ann_step": best["step"],
                "ann_read": best["read"],
                "ann_provenance": best["row"]["provenance"],
                "abs_d_read": d_read,
                "in_window": bool(d_read <= MATCH_WINDOW)}
        for key in PRIMARY + CENSUS:
            cv = observable(wr["row"], key)
            av = observable(best["row"], key)
            pair[key] = {"cons": cv, "anneal": av,
                         "d": (None if cv is None or av is None
                               else cv - av)}
        (pairs if pair["in_window"] else out_of_window).append(pair)

    read_span = {
        "walk_reads": [round(wr["read"], 6) for wr in walk_rows],
        "ann_read_min": round(min(a["read"] for a in ann_rows), 6),
        "ann_read_max": round(max(a["read"] for a in ann_rows), 6),
        "window": MATCH_WINDOW,
        "note": ("the dispatch's window disclosure: x43's committed panel "
                 "series spans 0.286-0.803 on p(T); the walk's rungs span "
                 "~1e-5 (install s1) to ~0.77 on p(Z) — the high gm12 "
                 "rungs (~0.90) have no g0 counterpart on either path "
                 "(co-reported); install s1 sits far BELOW the anneal's "
                 "floor"),
    }

    # ---- the separation scoring (the frozen rule) ---------------------
    in_window = [p for p in pairs if p["in_window"]]
    separations = {}
    for key in PRIMARY + CENSUS:
        ds = [p[key]["d"] for p in in_window if p[key]["d"] is not None]
        n_none = sum(1 for p in in_window if p[key]["d"] is None)
        if not ds:
            separations[key] = {"n_pairs": 0, "n_none": n_none,
                                "separates": False,
                                "note": "no valid pairs"}
            continue
        n_pos = sum(1 for d in ds if d > 0)
        n_neg = sum(1 for d in ds if d < 0)
        dominant_sign = "+" if n_pos >= n_neg else "-"
        sign = max(n_pos, n_neg) / len(ds)
        mean_abs_d = sum(abs(d) for d in ds) / len(ds)
        if key == "write_norm":
            pair_means = [(abs(p[key]["cons"]) + abs(p[key]["anneal"])) / 2
                          for p in in_window
                          if p[key]["d"] is not None]
            floor = FLOOR_NORM_REL * (sum(pair_means) / len(pair_means))
            floor_kind = f"relative 0.15 x mean pair norm = {floor:.3f}"
        elif key == "in_room":
            floor, floor_kind = FLOOR_INROOM, "absolute 0.05"
        elif key.startswith("alive"):
            floor, floor_kind = FLOOR_ALIVE, "absolute 0.10 (x40's clause)"
        else:
            floor, floor_kind = FLOOR_CORR, "absolute 0.15 (1/sqrt(59))"
        separations[key] = {
            "n_pairs": len(ds), "n_none": n_none,
            "sign_consistency": round(sign, 4),
            "dominant_sign": dominant_sign,
            "mean_abs_d": round(mean_abs_d, 6),
            "floor": round(floor, 6), "floor_kind": floor_kind,
            "separates": bool(sign >= SIGN_FRAC and mean_abs_d >= floor),
        }
        log(f"  [sep] {key:12s} pairs {len(ds):2d} sign {dominant_sign} "
            f"{sign:.2f} mean|d| {mean_abs_d:.4f} floor {floor:.3f} -> "
            f"{'SEPARATES' if separations[key]['separates'] else 'no'}")

    primary_sep = [k for k in PRIMARY if separations[k]["separates"]]
    census_sep = [k for k in CENSUS if separations[k]["separates"]]

    # ---- the verdict (the frozen composite) ---------------------------
    hard_ok = all(bool(g.get("pass")) for g in metrics["gates"].values())
    if not hard_ok:
        verdict = "TEXTURE"
        clause = ("hard-gate failure: "
                  + ", ".join(k for k, g in metrics["gates"].items()
                              if not g.get("pass"))
                  + " — nothing adjudicated; tables verbatim")
    elif SMOKE:
        verdict, clause = "SMOKE", "nothing adjudicated (smoke stamp)"
    elif not in_window:
        verdict, clause = "TEXTURE", (
            "no in-window matched pairs — the match window admits "
            "nothing; the match table is the report")
    elif primary_sep:
        verdict = "PROTOCOL-SECRET-IN-THE-PATH"
        clause = ("at matched reads the PRIMARY observables separate: "
                  + ", ".join(
                      f"{k} (sign {separations[k]['dominant_sign']}, "
                      f"mean|d| {separations[k]['mean_abs_d']:.4f} >= "
                      f"floor {separations[k]['floor']:.3f}, consistency "
                      f"{separations[k]['sign_consistency']:.2f})"
                      for k in primary_sep)
                  + " — THE FIRST POSITIVE FINGERPRINT of the missing "
                    "formation protocol; the era's target list opens")
    else:
        verdict = "ENDPOINTS-ONLY"
        if census_sep:
            clause = ("no primary observable separates at matched reads "
                      f"({len(in_window)} in-window pairs; every primary "
                      "mean|d| under its floor or sign-inconsistent); "
                      "the CENSUS observable separates with the origin-"
                      "expected sign: "
                      + ", ".join(
                          f"{k} (sign {separations[k]['dominant_sign']}, "
                          f"mean|d| {separations[k]['mean_abs_d']:.4f})"
                          for k in census_sep)
                      + " — ORIGIN-ONLY-SEPARATION (the projected-vs-"
                        "free-install confound the dispatch itself names)"
                        "; the gap is NOT trajectory shape; the protocol "
                        "hunt changes class")
        else:
            clause = (f"no observable separates at matched reads "
                      f"({len(in_window)} in-window pairs; every "
                      "mean|d| under its floor or sign-inconsistent) — "
                      "the paths are equivalent at matched reads within "
                      "the disclosed noise; the gap is NOT trajectory "
                      "shape; the protocol hunt changes class")

    adj = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "match_rule": REGISTERED["match_rule_verbatim"],
        "read_span": read_span,
        "n_walk_rows": len(walk_rows), "n_ann_rows": len(ann_rows),
        "n_pairs": len(pairs), "n_in_window": len(in_window),
        "n_out_of_window": len(out_of_window),
        "out_of_window": [{"walk_tag": p["walk_tag"],
                           "walk_read": p["walk_read"],
                           "nearest_ann_read": p["ann_read"],
                           "abs_d_read": p["abs_d_read"]}
                          for p in out_of_window],
        "separations": separations,
        "primary_separations": primary_sep,
        "census_separations": census_sep,
        "P-x45a": {"guess": REGISTERED["P-x45a"]["my_guess"],
                   "hit": bool(verdict == "ENDPOINTS-ONLY"),
                   "scored": REGISTERED["P-x45a"]["scored"]},
        "verdict": verdict, "clause": clause,
        "composite_order": ("TEXTURE (hard gates) -> PROTOCOL-SECRET-"
                            "IN-THE-PATH (>= 1 primary separates) -> "
                            "ENDPOINTS-ONLY (none; the census clause "
                            "names the origin residue when it fires)"),
        "endpoint_backdrop": {
            "x43_retentions": p0["committed_retentions"],
            "note": ("the ENDPOINTS this cell explains: the composed "
                     "retentions 0.24-0.33 vs the cons's recovery-in-band "
                     "~0.9-1.09 (x43's CEILING verdict)")},
    }
    metrics["adjudication"] = adj
    metrics["match"] = {"pairs": pairs}
    write_partial("P4 the match + adjudication done")
    log(f"P4: verdict {verdict} ({len(in_window)} in-window / "
        f"{len(out_of_window)} out)")
    return {"pairs": pairs, "adj": adj, "walk_rows": walk_rows,
            "ann_rows": ann_rows}


# ======================================================================
# P5 — THE OUTPUTS (the PNG + the REPORT)
# ======================================================================
def make_png(p4: dict) -> None:
    adj = p4["adj"]
    pairs = p4["pairs"]
    in_window = [p for p in pairs if p["in_window"]]
    fig = plt.figure(figsize=(16.5, 12.5))
    gs = fig.add_gridspec(3, 4, height_ratios=(1.05, 0.97, 0.97))

    # ---- panel 1: the two trajectories on the read axis ---------------
    ax1 = fig.add_subplot(gs[0, :2])
    wr = [(p["walk_pos"], p["walk_read"]) for p in pairs]
    ar = [(p["ann_step"], p["ann_read"])
          for p in sorted(p4["ann_rows"], key=lambda a: a["step"])]
    ax1.plot([s for s, _ in wr], [r for _, r in wr], "o-",
             color="tab:blue", label="the cons walk (p(Z), my realization)")
    ax1.plot([s for s, _ in ar], [r for _, r in ar], "s--",
             color="tab:red", alpha=0.85,
             label="the anneal panels (p(T), my realization)")
    for p in pairs:
        sty = "-" if p["in_window"] else ":"
        alph = 0.5 if p["in_window"] else 0.15
        ax1.plot([p["walk_pos"], p["ann_step"]],
                 [p["walk_read"], p["ann_read"]], sty,
                 color="gray", lw=0.8, alpha=alph)
    ax1.axhline(READ_BAR, color="black", ls="--", lw=1,
                label="the 0.05 bar")
    ax1.set_xlabel("lineage position (walk: install 0-400 then cons "
                   "400-700; anneal: 0-900)")
    ax1.set_ylabel("the read (battery mean p(NAME))")
    ax1.set_title("the two paths (matched pairs joined; dotted = "
                  "out-of-window)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25)

    # ---- panel 2: the separation summary ------------------------------
    ax2 = fig.add_subplot(gs[0, 2:])
    keys = list(PRIMARY + CENSUS)
    xs = range(len(keys))
    vals = [adj["separations"][k].get("mean_abs_d") or 0.0 for k in keys]
    floors = [adj["separations"][k].get("floor", 0.0) for k in keys]
    colors = ["tab:green" if adj["separations"][k]["separates"]
              else "tab:gray" for k in keys]
    ax2.barh([i + 0.2 for i in xs], vals, height=0.38, color=colors,
             label="mean |d| at matched pairs")
    ax2.barh([i - 0.2 for i in xs], floors, height=0.38, color="black",
             alpha=0.35, label="the frozen floor")
    for i, k in enumerate(keys):
        s = adj["separations"][k]
        ax2.text(max(vals[i], floors[i]) + 0.012, i,
                 f"sign {s.get('dominant_sign', '?')} "
                 f"{s.get('sign_consistency', 0):.2f}",
                 va="center", fontsize=7)
    ax2.set_yticks(list(xs), keys)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean |cons - anneal| over in-window pairs")
    ax2.set_title(f"the separation panel — verdict {adj['verdict']}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="x")

    # ---- panels 3-10: each observable side by side along the read axis
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        xs_c = [p["walk_read"] for p in in_window
                if p[key]["d"] is not None]
        ys_c = [p[key]["cons"] for p in in_window
                if p[key]["d"] is not None]
        xs_a = [p["ann_read"] for p in in_window
                if p[key]["d"] is not None]
        ys_a = [p[key]["anneal"] for p in in_window
                if p[key]["d"] is not None]
        ax.plot(xs_c, ys_c, "o", color="tab:blue", ms=5, label="cons walk")
        ax.plot(xs_a, ys_a, "s", color="tab:red", ms=5, label="anneal")
        allv = [v for v in ys_c + ys_a if v is not None]
        if allv:
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.15 + 1e-3
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--",
                    lw=0.8, alpha=0.5)
        s = adj["separations"][key]
        ax.set_title(f"{key}\n{'SEPARATES' if s['separates'] else 'no sep'}"
                     f" (mean|d| {s.get('mean_abs_d', 0):.3f})",
                     fontsize=8)
        ax.set_xlabel("the read", fontsize=7)
        ax.set_ylabel(key, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=6)
    fig.suptitle("X45 THE PATH DIFFERENCE — the cons's formation walk vs "
                 "the anneal's panels at matched reads", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RD / "x45_path_difference.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def write_report(p4: dict, p0: dict) -> None:
    adj = p4["adj"]
    L: list[str] = []
    A = L.append
    A("# X45 — THE PATH DIFFERENCE (the formation-protocol era's first "
      "cell)")
    A("")
    A(f"* VERDICT: **{adj['verdict']}** — {adj['clause']}")
    A(f"* P-x45a (my registered read): "
      f"{'HIT' if adj['P-x45a']['hit'] else 'FELL'} (guess: "
      f"{adj['P-x45a']['guess']}; the dispatch's lab lean: "
      "ENDPOINTS-ONLY, weakly)")
    n_pass = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    n_all = len(metrics["gates"])
    A(f"* gates: {n_pass}/{n_all} PASS"
      + ("" if n_pass == n_all else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## The match (nearest-read pairing; the frozen window)")
    A("")
    A(f"* walk rows: {adj['n_walk_rows']} | anneal rows: "
      f"{adj['n_ann_rows']} | pairs: {adj['n_pairs']} "
      f"(in-window {adj['n_in_window']}, out {adj['n_out_of_window']})")
    A(f"* the read spans — walk: {adj['read_span']['walk_reads']}")
    A(f"* anneal: [{adj['read_span']['ann_read_min']}, "
      f"{adj['read_span']['ann_read_max']}] (window "
      f"{adj['read_span']['window']}); {adj['read_span']['note']}")
    if adj["out_of_window"]:
        A("* OUT-OF-WINDOW rungs (named, disclosed, never adjudicating): "
          + "; ".join(f"{o['walk_tag']} read {o['walk_read']:.6f} "
                      f"(nearest panel {o['nearest_ann_read']:.6f}, "
                      f"|d| {o['abs_d_read']:.3f})"
                      for o in adj["out_of_window"]))
    A("")
    A("## The matched-pair table (every observable side by side along "
      "the read axis)")
    A("")
    A("| walk rung | read (cons) | anneal panel | read (anneal) | "
      "|d read| | alive g-12 | alive g0 | alive g+12 | "
      "r(g-12|g0) | r(g0|g+12) | r(g-12|g+12) | write norm | "
      "in-room |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in p4["pairs"]:
        def f3(key, side):
            v = p[key][side]
            return "-" if v is None else f"{v:.3f}"

        A(f"| {p['walk_tag']}{'' if p['in_window'] else ' (OUT)'} | "
          f"{p['walk_read']:.4f} | {p['ann_tag']} | "
          f"{p['ann_read']:.4f} | {p['abs_d_read']:.3f} | "
          f"{f3('alive_g-12', 'cons')}/{f3('alive_g-12', 'anneal')} | "
          f"{f3('alive_g0', 'cons')}/{f3('alive_g0', 'anneal')} | "
          f"{f3('alive_g+12', 'cons')}/{f3('alive_g+12', 'anneal')} | "
          f"{f3('r_g-12|g0', 'cons')}/{f3('r_g-12|g0', 'anneal')} | "
          f"{f3('r_g0|g+12', 'cons')}/{f3('r_g0|g+12', 'anneal')} | "
          f"{f3('r_g-12|g+12', 'cons')}/{f3('r_g-12|g+12', 'anneal')} | "
          f"{p['write_norm']['cons']:.2f}/"
          f"{p['write_norm']['anneal']:.2f} | "
          f"{f3('in_room', 'cons')}/{f3('in_room', 'anneal')} |")
    A("")
    A("(each cell cons/anneal; provenance: the leg-boundary rows are the "
      "committed anchor states, the interior rows the class-certified "
      "replay — the drift tables sit in metrics.gates.G_REPLCLASS)")
    A("")
    A("## The separation (the frozen rule: >= 75% sign consistency AND "
      "mean |d| >= floor)")
    A("")
    A("| observable | n pairs | sign | consistency | mean |d| | floor | "
      "separates |")
    A("|---|---|---|---|---|---|---|")
    for k in PRIMARY + CENSUS:
        s = adj["separations"][k]
        A(f"| {k} | {s['n_pairs']} ({s['n_none']} None) | "
          f"{s.get('dominant_sign', '-')} | "
          f"{s.get('sign_consistency', 0):.2f} | "
          f"{s.get('mean_abs_d', 0):.4f} | {s.get('floor', 0):.3f} | "
          f"{'**YES**' if s['separates'] else 'no'} |")
    A("")
    A("## The endpoints backdrop (what this cell explains)")
    A("")
    A("* x43's composed retentions (the CEILING): "
      + "; ".join(f"{k} {v:.4f}" for k, v in
                  adj["endpoint_backdrop"]["x43_retentions"].items())
      + " vs the cons's recovery-in-band ~0.9-1.09")
    A("* the committed anchors: install s400 g0 "
      f"{p0['committed_anchors']['own_install_read']:.6f}; root s300 g0 "
      f"{p0['committed_anchors']['root_read']:.6f} gm12 "
      f"{p0['committed_anchors']['root_gm12']:.4f}; the TAVIREN subject "
      f"g0 {p0['committed_anchors'].get('tav_subject_read', float('nan')):.6f}")
    A("")
    A("## The floor row + the base (co-report)")
    A("")
    b = metrics["walk"]["rungs"]["base"]
    A(f"* base (e001): read {b['read']:.2e}; write 0; net0 BASE — the "
      "walk's floor, out-of-window by construction")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* CPU desk only (threads 4; zero CUDA calls); timestamps UTC "
      "only; every artifact md5-bound (see metrics.gates.G_MD5); the "
      "replay drift tables in metrics.gates.G_REPLCLASS")
    A("* catches + disclosures: see metrics.deviations")
    A("")
    A("*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
      "folds).*")
    (RD / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
E001_FLAT64: np.ndarray = np.zeros(0)     # set in main (the census origin)


def main() -> None:
    global E001_FLAT64
    log(f"X45 — THE PATH DIFFERENCE (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = git_head()
    write_partial("startup (bars + P-x45a registered, committed at "
                  "birth)")
    set_seed(45500)             # global init only; every RNG is its own
    assert torch.get_num_threads() == 4

    p0 = phase_P0()

    # the e001 flat (fp64) — the census ruler's origin for BOTH paths
    e001_net = G1.load_g1(CKPT_DIR / "e001.pt")
    E001_FLAT64 = flat_params_cpu(e001_net).double().numpy()
    assert E001_FLAT64.shape[0] == GB.G1B_PARAMS
    del e001_net

    walk = phase_P1(p0)
    anneal = phase_P2(p0)
    phase_P3(p0, walk, anneal)
    p4 = phase_P4(p0, walk, anneal)

    gates_pass = all(bool(g.get("pass")) for g in metrics["gates"].values())
    if not gates_pass:
        p4["adj"]["verdict"] = "TEXTURE"
    metrics["status"] = (
        "SMOKE: nothing adjudicated" if SMOKE else
        f"ADJUDICATED: {p4['adj']['verdict']}")
    make_png(p4)
    metrics["git_head_final"] = git_head()
    metrics["date_finished"] = common.now_iso()
    save_json(RD / "metrics.json", metrics)
    write_report(p4, p0)
    log(f"DONE — verdict {p4['adj']['verdict']} (gates "
        f"{sum(1 for g in metrics['gates'].values() if g.get('pass'))}/"
        f"{len(metrics['gates'])})")


if __name__ == "__main__":
    main()
