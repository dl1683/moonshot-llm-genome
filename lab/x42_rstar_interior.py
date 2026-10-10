"""X42 — THE R* INTERIOR (R78's card 4; the flip threshold's bracket
refinement; the two-channel law's last coarse edge; dispatched
2026-10-10, CPU desk lane). This docstring carries the question, the
design, the bars and P-x42a VERBATIM from the dispatch letter, plus
every frozen operationalization, committed at birth BEFORE any
compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (dispatch, verbatim): "x38 bracketed the flip threshold
R* in (1.3384e-05, 0.5302] — between the empty slot and the LOWEST
living read — with the six-rung ladder's zero ambiguous rungs. But
the bracket's interior is 4+ orders of magnitude wide and EMPTY of
test mass. THE QUESTION: inside the bracket, is the flip still
dead-or-alive (a sharp R* in the interior) or does it grade?"

THE DESIGN (dispatch, verbatim in substance):
  1. INTERIOR RUNGS: construct mid-level reads on the same slot
     (ZEPHYRA at the host contexts): (a) partially-formed installs —
     the committed install trajectories' intermediate states if
     checkpointed, else SHORT installs (the e261 rig's convention run
     for 25/50/100/200 steps — each a fresh read level; disclose the
     construction; CPU-only if the rig allows, else name it); (b) the
     existing mid-formation states from e335's walk (the install's
     own s25/s50/s100/s200 if recoverable from the committed
     records); aim for 4-6 interior rungs spanning 0.01-0.5.
  2. THE FIXED DISPLACEMENT: x38's exact K10K complement (bit-equal
     carrier); read Z at the host contexts per rung; both currencies
     (ratio + move).
  3. THE CLASSIFICATION: per rung, LIFT (ratio > 1) / COLLAPSE
     (move > ~5x the band) / MIXED (the in-between — report it).

BARS (frozen VERBATIM from the dispatch, BEFORE any compute):
  - SHARP-INTERIOR: "a narrow R* window (<= ~1 order of magnitude)
    separates all-lift-below from all-collapse-above with the mixed
    zone inside it — dead-or-alive confirmed at fine grain; R* joins
    the doc's Law 7 as a located constant."
  - GRADED-INTERIOR: "a broad mixed zone (the response grades over
    1+ orders) — the flip is soft inside the bracket; the two
    channels overlap; Law 7's parameter re-words to a band."

REGISTERED PREDICTION P-x42a (frozen VERBATIM from the dispatch,
BEFORE compute): "Register P-x42a BEFORE compute. Lab lean:
SHARP-INTERIOR, weakly — the softmax denominator is a thresholding
machine and x38's outer rungs were total; the countervailing: partial
formation may be broadly distributed (the mid-formation states are
not mini-adults). State your own read."

EXECUTOR POSITION P-x42a-exec (registered at birth, BEFORE compute;
the dispatch's own invitation): "SHARP-INTERIOR, weakly — concurring
with the lab lean, WITH A LOCATED GUESS: the mechanism is slot
OWNERSHIP by the softmax committee, and the committed install
trajectory reads 0.55 by step 100 (e335/g1c_root's own record) —
ownership forms EARLY, so I expect the flip LOW in the interior:
LIFT rungs only near the bottom (priors <= ~0.05, the ratio-currency
zone), full COLLAPSE above ~0.1, and the R* window inside roughly
(0.01, 0.25] — <= 1.4 orders, most likely <= 1. REGISTERED SECONDARY
bits (scored, never bars): (1) sharp_interior fires; (2) flip_low —
if sharp, the window's top C <= 0.25; (3) no_broad_saturation — at
most ONE interior rung classed COLLAPSE with direction UP (the
mid-formation saturation risk below); (4) all interior rungs with
prior > 0.10 are COLLAPSE. REGISTERED RISKS (the countervailing, made
concrete): (i) MID-FORMATION SATURATION — a partially-formed read
may saturate UP under the complement (the slot not yet
committee-owned; the frozen magnitude clause classes it COLLAPSE with
direction UP co-reported — the SIGN MAP is the discriminating
observation); (ii) NON-MONOTONE FORMATION — the committed trajectory
swings (0.55 s100 -> 0.75 s200 -> 0.51 s300), so first-crossing
states may sit on transients; the rung's read level is its read
level (the currency), not its age; (iii) the mid-formation states
are NOT mini-adults — a broad band of MIXED rungs spanning >= 1
order fires GRADED-INTERIOR honestly."

OPERATIONALIZATIONS (frozen HERE at birth BEFORE compute; they fix
the clauses, they do not move the bars):
  * THE OUTER RUNGS (references, x38's ladder VERBATIM): BASE :=
    e001.pt; INST_SIB := e048_repro.pt; OWN_INST :=
    g1c_install_resume.pt; ROOT := g1c_root.pt (== g1c_cons_resume,
    bit-gated in-cell); LOCKED := e131_consolidated_e113.pt; BALL :=
    g1c_W1_s300.pt (anchored + SETTLED — e337's instrument,
    inherited verbatim: load_g1 -> _enforce_wall; the settled state
    is the substrate). Every rung md5-bound; every bare host read
    gated vs e335's committed walk reads (tol 2e-6, the
    cross-session law); every {bare, +comp} panel cell gated
    bit-exact (tol 1e-12) vs x38's committed metrics — "the outer
    rungs reproduce" per the dispatch's gate.
  * THE INTERIOR RUNGS (the design's cut (b), FUSED with (a)):
    e335's committed records hold the install trajectory's READS at
    steps {1,100,200,300,400} only — the mid-formation STATES
    (s25/s50/...) were never checkpointed and are NOT recoverable as
    states. DISCLOSED THEREFORE: the interior rungs are constructed
    by RE-RUNNING THE ROOT'S OWN INSTALL on CPU — e001 + the
    e043-Dmix teach VERBATIM (name_bs 16 install windows w/ name
    mask + corp_bs 48 [16 paired originals + 32 random corpus],
    masked token-level union CE, AdamW(0.9,0.95) wd 0.1, lr 1e-3 x
    house cosine(total=1000, warmup 100), clip 1.0; the rig's torch
    CPU generator seeded 24314 — THE root-draw knob, g1c's own
    FRESH_GEN) — with PER-STEP battery reads (the rig's own
    instrument) and in-memory state capture. THE RIG ALLOWS CPU
    (2.74M is CPU-viable, e113's precedent; the committed run was
    one CUDA burst — cross-device fp arithmetic is the reconstruction's
    named noise, certified below). The batch-composition stream (the
    generator draws) is device-independent BY CONSTRUCTION.
  * THE INSTALL RE-RUN'S CERTIFICATION (the label rule, frozen):
    HARD structural gates — the base bit-loaded (md5; the training
    net is a deepcopy of the bit-loaded base), the rig constants
    cross-checked vs the e043/G1 sources AND g1c_root.pt's own meta
    (install_seed 24314, steps 400, recipe "e043-Dmix"), the first
    three steps' generator draws recorded (the stream's identity
    record), the run reaching >= 100 steps with the step-1 read
    measured. The NUMERIC agreement vs e335's committed trajectory
    reads is the LABEL: step-1 (committed 1.5183924915618263e-05,
    tol 2e-6 abs) and s100/s200/s300/s400 (committed
    0.5544/0.7542/0.5087/0.5302, tol 0.30 abs each — the
    trajectory's own swing envelope; DESCRIPTIVE) — all obtained
    cells within tol => label COMMITTED-TRAJECTORY (the committed
    install's own mid-formation states, CPU reconstruction);
    step-1 within tol but some later cell out => PARTIAL-DRIFT
    (disclosed); step-1 out => FRESH-REDRAW (a fresh CPU re-draw of
    the same construction — the dispatch's (a)-branch: short
    installs, each a fresh read level). ALL THREE LABELS ADJUDICATE;
    the label is disclosure, never a bar.
  * THE RUNG SELECTION (frozen): the interior grid T in {0.01, 0.02,
    0.05, 0.10, 0.20, 0.35, 0.50}; per T, the rung is the FIRST
    formation step whose per-step battery read >= T; PLUS the s1
    state (the committed trajectory's own step-1 read 1.5e-5 — the
    interior's bottom edge, 13% above the empty slot). A grid state
    qualifies as an INTERIOR RUNG iff its measured host-g0 site read
    lies in (base_site_read, own_inst_site_read] (the bracket, as
    measured in THIS cell — own_inst is x38's lowest living rung);
    overshoot states (read above the bracket top) are DISCLOSED,
    never rungs. Certification states (s100/s200/s300/s400) are
    never rungs; their panel cells are co-reported as the honesty
    rider (the reconstruction's final-state comp cell vs own_inst's
    committed cell). G_INTERIOR hard-gates: >= 4 distinct interior
    rungs (SMOKE: >= 1), spanning >= 1 order (max/min >= 10;
    SMOKE: waived), every rung's formation step + construction
    disclosed, net0 classes recorded (every interior rung: class
    BASE-forming — e001 + a partial e043-Dmix install).
  * THE FIXED DISPLACEMENT (x38 VERBATIM): COMP := the K10K write's
    complement at full dose — x15's construction rebuilt bit-exact
    AND gated bit-equal to the committed carrier
    runs/x15/x15_comp_xFULL.pt (G_REGEN), injected as
    substrate_fp32 + carrier_delta_fp32 per key (x15's fp32 method;
    on the base this re-derives the carrier's model bit-exact,
    G_ONESTATE).
  * THE PANEL + SITES (x24 VERBATIM, x38's port): the 11 names;
    HOST-G0 = e261's splice bank (corpus seed 1337, find_occ
    p >= 280, SPLICE_RNG 24301 shuffle, first 60, mix FLORIZEL 19 /
    ELIZABETH 41, PRE 130, [60,130]) — ADJUDICATES; NEUTRAL = 60
    train-split windows (x24's rule, seed 24001) — the
    site-generality rider, never adjudicated.
  * THE READ CURRENCY (the series' convention): a name's read :=
    p(name[0]) at the final context position, mean over the site's
    60 contexts. LIFT := p_comp/p_base (floor 1e-12; x24). MOVE :=
    |p_comp - p_base| (x37; sign always co-reported). BOTH
    currencies on EVERY rung.
  * THE NEVER-WRITTEN BAND on rung S := x24's frozen cohort-A
    {TAVIREN, QELVARO, BUVONDI, NYSTORA, VIRETAN, MAMILLIUS} read on
    S at host-g0: band_ratio := max lift, band_move := max |dp|.
    (No cohort-A name was ever written on ANY rung — the lineages
    wrote only ZEPHYRA via the Dmix teach; on the BASE, TAVIREN is
    lab-written and sets x24's own committed band, inherited
    verbatim.)
  * THE LIVING-READ RULE (x37's, inherited VERBATIM): IF
    p_base(ZEPHYRA,S,host) > 0.05 the rung adjudicates in the MOVE
    currency; ELSE the committed RATIO currency.
  * RUNG CLASSIFICATION (x38's frozen mechanical form, the
    in-between named MIXED per THIS dispatch's wording): a
    RATIO-currency rung (prior <= 0.05) is LIFT iff lift > 1.0, else
    MIXED; a MOVE-currency rung (prior > 0.05) is COLLAPSE iff move
    > 5.0 x band_move(S), else MIXED — the magnitude clause governs
    (a rung whose read SATURATES UP is classed COLLAPSE with
    direction UP co-reported; the sign map is the rider that sees
    it).
  * THE BARS, MECHANICALLY (frozen): over ALL rungs (the six outer
    references + the interior rungs) — L := max prior over LIFT
    rungs; C := min prior over COLLAPSE rungs; the R* window :=
    (L, C]; width_orders := log10(C/L); monotone := every LIFT
    prior < every COLLAPSE prior. SHARP-INTERIOR := (>= 1 LIFT) AND
    (>= 1 COLLAPSE) AND monotone AND (every MIXED rung's prior
    inside (L, C]) AND width_orders <= 1.0 ("<= ~1 order of
    magnitude"; an empty mixed zone is vacuously inside).
    GRADED-INTERIOR := (NOT sharp) AND (>= 1 MIXED rung) AND
    (max MIXED prior / min MIXED prior >= 10 — "the response grades
    over 1+ orders"). Residues (honest, frozen): clean monotone
    split with NO mixed rung and width_orders > 1.0 => UNRESOLVED-
    WIDE (the interior ladder failed to refine the bracket to ~1
    order; the table verbatim); anything else => PARTIAL-GRADING
    (the table verbatim; no wording change without a new registered
    cell).
  * REGISTERED SEEDS (all INHERITED except the design's own global):
    global 42000 (init only; the design's ONLY new number); install
    gen 24314 (g1c's FRESH_GEN — the committed root draw, inherited);
    neutral site 24001 (x24's); room cert 24005. The grid {0.01 ...
    0.50} is the design's only new ladder.
  * HONESTY RIDERS (never bars): ce_r per state; per (state, site):
    mean top-1 p, mean entropy; the neutral-site ladder; the full
    400-step formation trajectory (per-step battery reads) recorded;
    the sign map (signed moves on every rung); the reconstruction's
    cert-state panel cells; the unselected currency co-reported on
    every rung.
  * CPU ENVELOPE (dispatch: a review agent owns the GPU): CUDA_
    VISIBLE_DEVICES="" BEFORE torch import; torch threads 4 (x37's
    exact setting — reproduction-critical); pocketfft workers 4; NO
    GPU, no envelope-log writes, no other runs/ touched (write ONLY
    runs/x42/*; runs/x42_smoke/ is gitignored, never committed);
    timestamps datetime.now(UTC) only. NO new .pt artifacts (x38's
    convention): the interior states live in-memory; the
    construction is deterministic (seeded rig) and every read is
    recorded.
  * Outputs: runs/x42/{metrics.json (PROGRESSIVE), REPORT.md,
    x42_rstar_interior.png} (the response vs read level across the
    interior; the outer rungs drawn as references; the formation
    trajectory panel). NO NOTES/THINKING/QUEUE/STATE edits
    (dispatch — the heartbeat folds). Birth commit BEFORE compute;
    smoke pass disclosed; final commit AND push.
  * Smoke (X42_SMOKE=1): the FULL gate path LIVE (parents + panel +
    sites + complement + an 8-step install re-run + provisional
    rungs at steps {1,2,3,5,8} + every gate and read exercised);
    G_INTERIOR's >= 4 rungs and the >= 10 span waived (>= 1 rung);
    the cert label rule runs on step-1 only (s100 does not exist at
    8 steps); own smoke dir; NO adjudication, NO figure, NO report
    (SMOKE stamp on every read; nothing adjudicated).

COMPUTE: the install re-run (400 steps, ~0.5-1 s/step CPU) + ~36
panel states x 2 sites (60x130 CPU forwards) + ce_r evals on the
2.74M organism — minutes, CPU-only.

Run:  python lab/x42_rstar_interior.py          (X42_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-ONLY, bulletproof

import copy                                     # noqa: E402
import hashlib                                  # noqa: E402
import json                                     # noqa: E402
import math                                     # noqa: E402
import random                                   # noqa: E402
import re as _re                                # noqa: E402
import subprocess                               # noqa: E402
import sys                                      # noqa: E402
import time                                     # noqa: E402
from datetime import datetime, timezone         # noqa: E402
from pathlib import Path                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                             # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                # noqa: BLE001
    pass

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402
import torch.nn.functional as F                  # noqa: E402

import common                                    # noqa: E402
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                       # noqa: E402 (REPO,
                                                  # find_occ, SPLICE_RNG)
import g1b_continuity as GB                      # noqa: E402 — MUST be
                                                  # imported BEFORE G1
import g1_anchored_ball as G1                    # noqa: E402
import e261_rank_ladder as E261                  # noqa: E402
from e261_rank_ladder import SRCT                # noqa: E402

torch.set_num_threads(4)          # CPU-only cell; x37's exact setting
E261.DCT_WORKERS = 4              # x15/x17/x23/x24/x37/x38's convention

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

SMOKE = os.environ.get("X42_SMOKE") == "1"
NAME = "x42_smoke" if SMOKE else "x42"

T0 = time.time()
RD = run_dir(NAME)
REPO = E43.REPO
CKPT_DIR = GB.CKPT_DIR
CPU = torch.device("cpu")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(m: str) -> None:
    print(f"[x42 {time.time() - T0:7.1f}s] {m}", flush=True)


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
    except Exception:                                    # noqa: BLE001
        return "unavailable"


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
# ---- the outer ladder (x38's six rungs VERBATIM; e335's walk, md5-bound) ----
LADDER = [
    # (key, file, kind, net0_class, formation, committed walk read)
    ("base", "e001.pt", "plain", "BASE",
     "the 2.74M corpus base (the canon-lineage net0)", 1.3383959412749391e-05),
    ("inst_sib", "e048_repro.pt", "plain", "BASE-formed",
     "the LOCKED draw's install final (e043-Dmix s400, gen 24313) — the "
     "install SIBLING", 0.5563086867332458),
    ("own_inst", "g1c_install_resume.pt", "resume-model", "BASE-formed",
     "the root's OWN install final (e043-Dmix s400, gen 24314)", 0.5302218198776245),
    ("root", "g1c_root.pt", "plain", "ROOT",
     "THE g1c root == the cons final (bit-gated in-cell)", 0.7447534203529358),
    ("locked", "e131_consolidated_e113.pt", "plain", "ROOT-locked",
     "the LOCKED root (install gen 24313 + cons 10901)", 0.7850371599197388),
    ("ball", "g1c_W1_s300.pt", "plain-anchored+settle",
     "ROOT+washed+committed",
     "the committed WALL class (the family's only wash-proof object)",
     0.70429927110672),
]
LADDER_MD5 = {
    "e001.pt": "d114536d1c0983ab3be67f67ff0667c8",
    "e048_repro.pt": "0254042cdcd13724db6cde2ec978ce4f",
    "g1c_install_resume.pt": "1f5a6e2b327a0731459cbb805fdc2505",
    "g1c_root.pt": "9c7d4ca1b60c8a1158d080f932e2c95f",
    "e131_consolidated_e113.pt": "757ab0defa6cb1b1539c2d7dbf3432f1",
    "g1c_W1_s300.pt": "f80ee59e5f887a582e0e1b6546f501bc",
    "g1c_cons_resume.pt": "7abcfc5d5265b4b1117e78c6e39e9938",
}
CONS_CK = "g1c_cons_resume.pt"                     # the root==cons check
ROOT_CK = REPO / "runs" / "checkpoints" / "g1c_root.pt"
BALL_CK = REPO / "runs" / "checkpoints" / "g1c_W1_s300.pt"
FACT_CK = "e261_K10K_inst_resume.pt"               # the displacement's parent
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
ROOMS264_CK = "e264_rooms.pt"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"
ROOM_K = 10_000
ROOM_SEED_D, ROOM_SEED_S = 26113, 26114
N_PARAMS = GB.G1B_PARAMS                           # 2,739,072
READ_TOL = 2e-6                                    # cross-session law

# ---- the interior construction (the rig, frozen) ---------------------------
FRESH_GEN = 24314                # g1c's own root-draw knob (the committed gen)
INST_TOTAL = 1000                # e048's house-cosine total (verbatim)
INST_STEPS = 400 if not SMOKE else 8
CERT_STEPS = (1, 100, 200, 300, 400) if not SMOKE else (1, 8)
E335_TRAJ = {                    # e335's committed install-traj reads (g0_pz)
    1: 1.5183924915618263e-05,
    100: 0.5543805360794067,
    200: 0.7542406320571899,
    300: 0.5086620450019836,
    400: 0.5302218198776245,
}
CERT_TOL_S1 = 2e-6               # step-1: the near-deterministic regime
CERT_TOL_LATE = 0.30             # later cells: the trajectory's own swing
INTERIOR_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50) if not SMOKE \
    else (0.01, 0.05)
SMOKE_RUNG_STEPS = (1, 2, 3, 5, 8)                # SMOKE-only provisional rungs
INST_TIME_BUDGET_S = 1200.0       # the install leg's disclosed cap

# ---- x38's committed literals (the outer endpoints this cell pins) ---------
X38_METRICS = REPO / "runs" / "x38" / "metrics.json"
X38_METRICS_MD5 = "6b073d97f09aad1d09f4fb39fd45aa5d"
X38_VERDICT_WORD = "SHARP-FLIP"
X38_RSTAR_BRACKET = [1.3383960322244093e-05, 0.5302218198776245]
X38_ENVELOPE_X = 1.7139624959705781
X38_LADDER_CLS = {"base": "LIFT", "inst_sib": "COLLAPSE",
                  "own_inst": "COLLAPSE", "root": "COLLAPSE",
                  "locked": "COLLAPSE", "ball": "COLLAPSE"}
X38_MOVES = {                     # x38's committed (move_signed, move_x) cells
    "base": (0.000717416074621724, 0.00983851057473381),
    "inst_sib": (-0.43252459168434143, 6.925709866149691),
    "own_inst": (-0.17051097750663757, 11.870406968553983),
    "root": (-0.5268798768520355, 7.754921898696189),
    "locked": (-0.6687552779912949, 10.99874099853345),
    "ball": (-0.4061734080314636, 7.679464693745435),
}

# ---- the parents' committed artifacts (md5-bound) --------------------------
X15_METRICS = REPO / "runs" / "x15" / "metrics.json"
X15_METRICS_MD5 = "c90dd9371a5a7e248fdbd92c06bb243a"
X15_COMP_CARRIER = REPO / "runs" / "x15" / "x15_comp_xFULL.pt"
X15_COMP_CARRIER_MD5 = "b0a1785c6f759fdd1d8deeae7fee4a9b"
X24_METRICS = REPO / "runs" / "x24" / "metrics.json"
X24_METRICS_MD5 = "7f1a91cc1d42a202c8fb23c4f8fb3e7e"
X25_METRICS = REPO / "runs" / "x25" / "metrics.json"
X25_METRICS_MD5 = "aa711be2785ce33281796fc3fc5653a0"
X37_METRICS = REPO / "runs" / "x37" / "metrics.json"
X37_METRICS_MD5 = "b2090c28d33960c5687dc1f9ac63d2e4"
E337_METRICS = REPO / "runs" / "e337" / "metrics.json"
E337_METRICS_MD5 = "3d448b5307c0ff04df7d1bedfc41617b"
E335_METRICS = REPO / "runs" / "e335" / "metrics.json"
E335_METRICS_MD5 = "5d92359bc9b2e943c01bf521487cf177"
G1CROOT_METRICS = REPO / "runs" / "g1c_root" / "metrics.json"
G1CROOT_METRICS_MD5 = "05a33ada81e022b61a437db29afe38dd"
E293_SOURCE = REPO / "lab" / "e293_distinct_contention.py"   # the bank
E293_SOURCE_MD5 = "1f3e16968359171d3db1a52824c48cf4"
# the rig's own sources (the install construction's provenance, md5-bound)
E043_SOURCE = REPO / "lab" / "e043_install.py"
E043_SOURCE_MD5 = "f82806b369452b05b8d89ca6cebe70fa"
G1C_SOURCE = REPO / "lab" / "g1c_root_redraw.py"
G1C_SOURCE_MD5 = "b094295fd1d3a9e796de602d3b628344"
G1_SOURCE = REPO / "lab" / "g1_anchored_ball.py"
G1_SOURCE_MD5 = "f4b6997b6a66013ee25da67f2b4faf01"
E261_SOURCE = REPO / "lab" / "e261_rank_ladder.py"
E261_SOURCE_MD5 = "e498031fe4fcd0147c3094c86034b1c5"

# ---- frozen literals (cross-checked against the md5-bound artifacts) ------
K10K_WRITE_NORM = 9.1788432658723                 # ||dW_k|| (x15/e318)
X15_S_FULL = 3.0349885643664365                   # x15's committed scalar
X15_COMP_L2_64 = 3.024341960836486                # natural complement L2
X15_DIAG_KK = 0.0007308000349439681               # x15's committed KK cell
X23_KT = 0.027287840843200684                     # x23's committed KT cell
E293_BANK = ("ZEPHYRA", "TAVIREN", "QELVARO", "BUVONDI", "NYSTORA")

# ---- the panel (x24 VERBATIM) ----------------------------------------------
PANEL = [
    ("ZEPHYRA",   "anchor_formed_host",  "anchor", 0),
    ("TAVIREN",   "synthetic_count0",    "A",      0),
    ("QELVARO",   "synthetic_count0_fresh_bank", "A", 0),
    ("BUVONDI",   "synthetic_count0_fresh_bank", "A", 0),
    ("NYSTORA",   "synthetic_count0_fresh_bank", "A", 0),
    ("VIRETAN",   "scrambled_control",   "A",      0),
    ("MAMILLIUS", "corpus_rare_seen",    "A",      13),
    ("LEONTES",   "corpus_mid_proper",   "C",      125),
    ("KING",      "corpus_common_token", "C",      556),
    ("ELIZABETH", "corpus_host_high",    "C",      105),
    ("FLORIZEL",  "corpus_host_high",    "C",      45),
]
COHORT_A = [n for n, _, c, _ in PANEL if c == "A"]
ANCHOR = "ZEPHYRA"

# ---- the bars' numbers (frozen) --------------------------------------------
LIFT_BAR = 1.0              # "LIFT (ratio > 1)" — the bar's literal clause
COLLAPSE_X = 5.0            # "move > ~5x band" (x37/x38's fragility bar)
LIVING_BAR = 0.05           # the living-read rule's saturation threshold
WIDTH_BAR_ORDERS = 1.0      # "<= ~1 order of magnitude" (SHARP-INTERIOR)
GRADED_SPAN_X = 10.0        # "grades over 1+ orders" (GRADED-INTERIOR)
MIN_INTERIOR_RUNGS = 4      # "aim for 4-6 interior rungs" (SMOKE: 1)
INTERIOR_SPAN_X = 10.0      # the interior ladder must span >= 1 order
REPRO_TOL = 1e-12           # x38's bit-exact convention
CE_REPRO_TOL = 1e-9         # ce_r cross-session reproduction (fp32 mean)
PRIOR_FLOOR = 1e-12
GLOBAL_SEED = 42000
CERT_SEED = 24005
NEUTRAL_SEED = 24001        # x24's own (site identity)

REGISTERED = {
    "question_verbatim":
        "x38 bracketed the flip threshold R* in (1.3384e-05, 0.5302] — "
        "between the empty slot and the LOWEST living read — with the "
        "six-rung ladder's zero ambiguous rungs. But the bracket's "
        "interior is 4+ orders of magnitude wide and EMPTY of test mass. "
        "THE QUESTION: inside the bracket, is the flip still dead-or-alive "
        "(a sharp R* in the interior) or does it grade?",
    "design_verbatim": {
        "interior_rungs":
            "construct mid-level reads on the same slot (ZEPHYRA at the "
            "host contexts): (a) partially-formed installs — the committed "
            "install trajectories' intermediate states if checkpointed, "
            "else SHORT installs (the e261 rig's convention run for "
            "25/50/100/200 steps — each a fresh read level; disclose the "
            "construction; CPU-only if the rig allows, else name it); (b) "
            "the existing mid-formation states from e335's walk (the "
            "install's own s25/s50/s100/s200 if recoverable from the "
            "committed records); aim for 4-6 interior rungs spanning "
            "0.01-0.5.",
        "fixed_displacement":
            "x38's exact K10K complement (bit-equal carrier); read Z at "
            "the host contexts per rung; both currencies (ratio + move).",
        "classification":
            "per rung, LIFT (ratio > 1) / COLLAPSE (move > ~5x the band) / "
            "MIXED (the in-between — report it).",
    },
    "bars_verbatim": {
        "SHARP-INTERIOR":
            "a narrow R* window (<= ~1 order of magnitude) separates "
            "all-lift-below from all-collapse-above with the mixed zone "
            "inside it — dead-or-alive confirmed at fine grain; R* joins "
            "the doc's Law 7 as a located constant.",
        "GRADED-INTERIOR":
            "a broad mixed zone (the response grades over 1+ orders) — "
            "the flip is soft inside the bracket; the two channels "
            "overlap; Law 7's parameter re-words to a band.",
    },
    "P_x42a_verbatim":
        "Register P-x42a BEFORE compute. Lab lean: SHARP-INTERIOR, weakly "
        "— the softmax denominator is a thresholding machine and x38's "
        "outer rungs were total; the countervailing: partial formation "
        "may be broadly distributed (the mid-formation states are not "
        "mini-adults). State your own read.",
    "executor_position": (
        "SHARP-INTERIOR, weakly — concurring with the lab lean, WITH A "
        "LOCATED GUESS: the mechanism is slot OWNERSHIP by the softmax "
        "committee, and the committed install trajectory reads 0.55 by "
        "step 100 (e335/g1c_root's own record) — ownership forms EARLY, "
        "so I expect the flip LOW in the interior: LIFT rungs only near "
        "the bottom (priors <= ~0.05, the ratio-currency zone), full "
        "COLLAPSE above ~0.1, and the R* window inside roughly (0.01, "
        "0.25] — <= 1.4 orders, most likely <= 1. REGISTERED SECONDARY "
        "bits (scored, never bars): (1) sharp_interior fires; (2) "
        "flip_low — if sharp, the window's top C <= 0.25; (3) "
        "no_broad_saturation — at most ONE interior rung classed COLLAPSE "
        "with direction UP (the mid-formation saturation risk); (4) all "
        "interior rungs with prior > 0.10 are COLLAPSE. REGISTERED RISKS "
        "(the countervailing, made concrete): (i) MID-FORMATION "
        "SATURATION — a partially-formed read may saturate UP under the "
        "complement (the slot not yet committee-owned; the frozen "
        "magnitude clause classes it COLLAPSE with direction UP "
        "co-reported — the SIGN MAP is the discriminating observation); "
        "(ii) NON-MONOTONE FORMATION — the committed trajectory swings "
        "(0.55 s100 -> 0.75 s200 -> 0.51 s300), so first-crossing states "
        "may sit on transients; the rung's read level is its read level "
        "(the currency), not its age; (iii) the mid-formation states are "
        "NOT mini-adults — a broad band of MIXED rungs spanning >= 1 "
        "order fires GRADED-INTERIOR honestly."),
    "clauses_fixed": {
        "read_currency": "a name's read := p(name[0]) at the final context "
                         "position, mean over the site's 60 contexts "
                         "(x15/x23/x24/x37/x38's committed currency)",
        "lift": "lift(i,S) := p_comp(i,S) / max(p_base(i,S), 1e-12)",
        "move": "move(i,S) := |p_comp(i,S) - p_base(i,S)| (x37's "
                "registered living-read currency; sign always co-reported)",
        "never_written_band": "x24's frozen cohort-A {TAVIREN, QELVARO, "
                              "BUVONDI, NYSTORA, VIRETAN, MAMILLIUS} read "
                              "on the SAME rung at host-g0; band_ratio := "
                              "max lift, band_move := max |dp_comp - "
                              "dp_base|",
        "living_read_rule": "IF p_base(ZEPHYRA,S,host) > 0.05 the rung "
                            "adjudicates in the MOVE currency; ELSE the "
                            "committed RATIO currency. x37's registered "
                            "rule, inherited VERBATIM; both currencies "
                            "co-reported either way.",
        "rung_classification": "RATIO rung (prior <= 0.05): LIFT iff lift "
                               "> 1.0, else MIXED; MOVE rung (prior > "
                               "0.05): COLLAPSE iff move > 5.0 x "
                               "band_move, else MIXED — the magnitude "
                               "clause governs; direction co-reported "
                               "(x38's frozen form; the in-between named "
                               "MIXED per this dispatch's wording)",
        "verdict_sharp": ">= 1 LIFT AND >= 1 COLLAPSE AND every LIFT "
                         "prior < every COLLAPSE prior AND every MIXED "
                         "rung inside (L, C] AND log10(C/L) <= 1.0 — "
                         "R* window (L, C]; the mixed zone (possibly "
                         "empty) inside it",
        "verdict_graded": "NOT sharp AND >= 1 MIXED rung AND max-MIXED-"
                          "prior / min-MIXED-prior >= 10 (the response "
                          "grades over 1+ orders; the MIXED rungs are the "
                          "overlap band)",
        "verdict_residues": "UNRESOLVED-WIDE (clean monotone split, no "
                            "MIXED rung, width > 1 order — the interior "
                            "ladder failed to refine the bracket) / "
                            "PARTIAL-GRADING (anything else; the table "
                            "verbatim; no wording change without a new "
                            "registered cell)",
        "interior_construction": "the root's OWN install (e001 + e043-Dmix "
                                 "VERBATIM, gen 24314) re-run on CPU with "
                                 "per-step battery reads; rungs = s1 + the "
                                 "FIRST formation step whose per-step read "
                                 ">= T, T in {0.01, 0.02, 0.05, 0.10, 0.20, "
                                 "0.35, 0.50}, kept iff its measured host "
                                 "read lies in (base_read, own_inst_read] "
                                 "(the bracket as measured in this cell); "
                                 "overshoots disclosed, never rungs",
        "cert_label_rule": "step-1 vs committed (tol 2e-6) and s100/s200/"
                           "s300/s400 vs committed (tol 0.30 each; "
                           "DESCRIPTIVE): all within => COMMITTED-"
                           "TRAJECTORY; step-1 within, some late cell out "
                           "=> PARTIAL-DRIFT; step-1 out => FRESH-REDRAW "
                           "(the dispatch's (a)-branch: fresh short "
                           "installs). ALL THREE LABELS ADJUDICATE; the "
                           "label is disclosure, never a bar.",
    },
    "registration": "bars + P-x42a VERBATIM from the dispatch letter; "
                    "operationalizations frozen HERE at birth BEFORE "
                    "compute; this script committed at birth; adjudicate "
                    "against exactly this; no bar shopping.",
}

DEVIATIONS = [
    "CPU-ONLY cell (dispatch: a review agent owns the GPU lane) — "
    "CUDA_VISIBLE_DEVICES='' before torch import, torch threads 4 "
    "(x37's exact setting, reproduction-critical), pocketfft workers 4, "
    "no GPU code path, no envelope-log writes, no other runs/ touched.",
    "CONSTRUCTION (b) IS NOT RECOVERABLE AS STATES (disclosed): e335's "
    "committed records hold the install trajectory's READS at steps "
    "{1,100,200,300,400} only; the mid-formation states were never "
    "checkpointed. The interior rungs are therefore constructed by "
    "(a)+(b) FUSED: the root's OWN install rig (e043-Dmix, gen 24314) "
    "re-run on CPU (the rig allows CPU: 2.74M is CPU-viable, e113's "
    "precedent) with per-step reads and state capture — the committed "
    "trajectory's own intermediate states up to cross-device fp "
    "arithmetic, certified by the frozen label rule.",
    "THE COMMITTED INSTALL RAN ON CUDA (one 19.4 s burst; g1c_root's "
    "chunk_table); this cell's re-run is CPU (the dispatch's ban) — "
    "bit-identity with the committed trajectory is IMPOSSIBLE and is "
    "NOT claimed; the batch-composition stream (the CPU generator "
    "draws) is device-independent by construction; the numeric "
    "agreement is the LABEL (COMMITTED-TRAJECTORY / PARTIAL-DRIFT / "
    "FRESH-REDRAW), never a bar.",
    "THE x38 GAUSSIAN RIDER IS NOT RE-RUN (disclosed omission): the "
    "dispatch's design names only the interior rungs + the fixed "
    "displacement + the classification; x38's rider (the gaussian at "
    "1x/2x/4x) is x38's committed record.",
    "The interior rungs are NEW SUBJECTS for the panel (never "
    "panel-probed by any prior cell); their per-step formation reads "
    "use the rig's own instrument (battery_cell — the committed "
    "trajectory cells' convention); their adjudicated reads use x38's "
    "site_read (the committed ladder currency); both recorded per "
    "rung.",
    "The displacement vector is the FIXED complement from x15's carrier "
    "applied IDENTICALLY to every rung (outer + interior) — "
    "substrate_fp32 + delta_fp32 per key (x15's fp32 method). On the "
    "base this re-derives the carrier's model bit-exact (gated).",
    "THE BALL'S SETTLE (e337's registered instrument, inherited): the "
    "artifact's raw body is one optimizer step OUTSIDE the wall; "
    "load_g1 -> _enforce_wall settles; THE SUBSTRATE IS THE SETTLED "
    "STATE; pre/post wall_report recorded.",
    "ROOT == CONS re-gated in-cell (e335's G_ROOTIDENT, bit-exact).",
    "NO new .pt artifacts (x38's convention): the interior states live "
    "in-memory; the construction is deterministic (seeded rig) and "
    "every read is recorded in metrics.json.",
    "n=1 per rung (one trajectory, one session — the g-series standing "
    "lottery note); the bit-exact outer-rung reproduction (264 panel "
    "cells) doubles as the in-cell instrument replicate.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
    "Smoke (X42_SMOKE=1): 8-step install; provisional rungs at steps "
    "{1,2,3,5,8}; G_INTERIOR's >= 4 rungs and >= 10 span waived; the "
    "cert label rule runs on step-1 only; own (gitignored) smoke dir; "
    "NOTHING adjudicated.",
]

METRICS: dict = {
    "experiment": "x42_rstar_interior",
    "phase": "THE R* INTERIOR (R78 card 4; x38's bracket refinement): "
             "mid-level reads constructed on the same slot (the root's "
             "own install re-formed on CPU, per-step rungs) inside x38's "
             "(1.34e-5, 0.53] bracket, under x38's exact K10K complement, "
             "both currencies per rung, LIFT/COLLAPSE/MIXED per rung — is "
             "the flip still dead-or-alive inside the bracket or does it "
             "grade?",
    "date": now_utc(),
    "status": "PARTIAL: startup (bars + P-x42a registered at birth)",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": 4},
    "envelope": {
        "device": "CPU ONLY (CUDA_VISIBLE_DEVICES=''; a review agent owns "
                  "the GPU lane; no envelope-log writes)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": DEVIATIONS,
    "builds_on": [
        "x38 (THE FLIP THRESHOLD: the ladder, the complement port, the "
        "rung-classification form, the six outer rungs' committed panel "
        "cells — this cell's bit-exact references and the bracket being "
        "refined)",
        "x37 (THE ROOT AS SUBJECT: the sign flip's discovery; the MOVE "
        "currency; the living-read rule)",
        "e337 (THE BALL AS SUBJECT: the settle instrument)",
        "e335 (THE ROOT'S PROVENANCE: the committed install trajectory "
        "reads at s1/s100/s200/s300/s400 — the reconstruction's "
        "certification cells; the walk's bit-identity gates)",
        "g1c_root_redraw (THE RIG: the e043-Dmix install construction "
        "VERBATIM — chunked_install's arithmetic, the gen-24314 root "
        "draw, the in-loop battery instrument)",
        "x24 (the 11-name panel, the two sites, cohort-A, the lift "
        "currency)",
        "x15 (the complement construction + the committed carrier)",
        "e261/e264 (the K10K room; the installed-fact parent)",
        "R78 (the review that minted this card — the R* interior)",
    ],
    "whats_new": [
        "THE INTERIOR LADDER: mid-level reads on the SAME slot inside "
        "x38's 4-order-empty bracket — the flip's control parameter "
        "measured at fine grain, not asserted",
        "THE MID-FORMATION CONSTRUCTION: the committed install's own "
        "trajectory re-formed on CPU with per-step reads and "
        "first-crossing rung selection on a registered geometric grid",
        "THE THREE-CLASS ADJUDICATION (LIFT/COLLAPSE/MIXED) with the R* "
        "window's WIDTH in orders as the bar's currency — sharp vs "
        "graded made mechanical",
        "THE SIGN MAP ACROSS FORMATION: signed moves on every rung — "
        "mid-formation saturation (COLLAPSE with direction UP) is the "
        "registered risk the map alone can see",
        "THE RECONSTRUCTION LABEL RULE: CPU re-run vs committed CUDA "
        "trajectory certified by disclosed tolerances — provenance "
        "graded, never assumed",
    ],
    "gates": {},
}
METRICS_PATH = RD / "metrics.json"


def write_partial(note: str) -> None:
    METRICS["phase_note"] = note
    METRICS["date_updated"] = now_utc()
    save_json(METRICS_PATH, METRICS)
    log(f"[metrics] partial saved ({note})")


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def site_read(net, ids: torch.Tensor, bs: int = 30) -> dict:
    """x24/x37/x38's site_read verbatim: one pass over a site's 60
    contexts, the full final softmax row-matrix kept so EVERY name
    column reads from the SAME pass."""
    net.eval()
    rows, tops, ents = [], [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        rows.append(pr)
        tops.append(pr.max(-1).values)
        ents.append(-(pr * torch.log(pr.clamp_min(1e-30))).sum(-1))
    probs = torch.cat(rows)                       # [60, vocab]
    return {"probs": probs,
            "rider_mean_top1_p": float(torch.cat(tops).mean()),
            "rider_mean_entropy": float(torch.cat(ents).mean())}


def name_read(site: dict, cid: int) -> dict:
    """x24/x37/x38's name_read statistics verbatim."""
    p = site["probs"][:, cid]
    lg = torch.log(p.clamp_min(1e-30))
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_pz_ge_0.05": float((p >= 0.05).float().mean()),
            "frac_argmax_z": float((site["probs"].argmax(-1) == cid)
                                   .float().mean()),
            "mean_log_pz": float(lg.mean())}


def inroom_frac(room: SRCT, v64: np.ndarray) -> float:
    vn = float(np.linalg.norm(v64))
    if vn == 0.0:
        return 0.0
    return float(np.linalg.norm(room.project(v64)) / vn)


def inroom_energy_share(room: SRCT, v64: np.ndarray) -> float:
    e = float(v64 @ v64)
    if e == 0.0:
        return 0.0
    p = room.project(v64)
    return float((p @ p) / e)


def texture(halt: str) -> SystemExit:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": f"{halt} — nothing adjudicated"}
    write_partial(f"HALT: TEXTURE ({halt})")
    return SystemExit(f"{halt} — nothing adjudicated")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X42 — THE R* INTERIOR (smoke={SMOKE}) -> {RD}")
    METRICS["birth_commit"] = git_head()
    write_partial("startup (bars + P-x42a registered, committed at birth)")
    set_seed(GLOBAL_SEED)          # global init only; no fresh draws exist

    # ============ P0: corpus, panel gates, the two sites ================
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    zid = stoi["Z"]

    # G_NAMEFREE + G_PANEL (x24's convention, ported verbatim via x38)
    x38m = json.loads(X38_METRICS.read_text(encoding="utf-8"))
    x24m = json.loads(X24_METRICS.read_text(encoding="utf-8"))
    x25m = json.loads(X25_METRICS.read_text(encoding="utf-8"))
    e335m = json.loads(E335_METRICS.read_text(encoding="utf-8"))
    g1crm = json.loads(G1CROOT_METRICS.read_text(encoding="utf-8"))

    e293_src = E293_SOURCE.read_text(encoding="utf-8")
    m = _re.search(r"NAME_BANK\s*=\s*\(([^)]*)\)", e293_src)
    bank_on_disk = tuple(s.strip().strip('"\'') for s in
                         m.group(1).split(",")) if m else ()
    counts = {nm: train_text.count(nm) for nm, *_ in PANEL}
    initials = [nm[0] for nm, *_ in PANEL]
    fresh_bank = [nm for nm, cls, _, _ in PANEL
                  if cls == "synthetic_count0_fresh_bank"]
    scrambled = next(nm for nm, cls, _, _ in PANEL
                     if cls == "scrambled_control")
    synth_all = [nm for nm, cls, _, _ in PANEL
                 if cls.startswith("synthetic") or cls == "scrambled_control"]
    gpanel = {
        "form": "x24's gate convention ported verbatim (the instrument's "
                "own panel gate)",
        "e293_bank_on_disk": list(bank_on_disk),
        "e293_bank_matches_frozen": bool(bank_on_disk == E293_BANK),
        "fresh_bank_members_in_e293_bank":
            {nm: bool(nm in bank_on_disk) for nm in fresh_bank},
        "counts": counts,
        "counts_match_frozen": bool(all(
            counts[nm] == exp for nm, _, _, exp in PANEL)),
        "chars_in_vocab": {nm: all(c in stoi for c in nm)
                           for nm, *_ in PANEL},
        "seven_letters": {nm: len(nm) == 7 for nm in synth_all},
        "not_g1_name": {nm: nm != G1.NAME for nm in synth_all},
        "initials": initials,
        "initials_distinct": bool(len(set(initials)) == len(initials)),
        "scrambled_is_permutation":
            bool(sorted(scrambled) == sorted("TAVIREN")),
        "scrambled_not_a_bank_name": bool(scrambled not in E293_BANK),
    }
    gpanel["pass"] = bool(
        bank_on_disk == E293_BANK
        and all(nm in bank_on_disk for nm in fresh_bank)
        and gpanel["counts_match_frozen"]
        and all(gpanel["chars_in_vocab"].values())
        and all(gpanel["seven_letters"].values())
        and all(gpanel["not_g1_name"].values())
        and gpanel["initials_distinct"]
        and gpanel["scrambled_is_permutation"]
        and gpanel["scrambled_not_a_bank_name"])
    METRICS["gates"]["G_PANEL"] = gpanel
    if not gpanel["pass"]:
        raise texture(f"G_PANEL FAILURE: {gpanel}")
    gnamefree = {"synthetic_counts": {nm: counts[nm] for nm in synth_all},
                 "pass": bool(all(counts[nm] == 0 for nm in synth_all))}
    METRICS["gates"]["G_NAMEFREE"] = gnamefree
    if not gnamefree["pass"]:
        raise texture(f"G_NAMEFREE FAILURE: {gnamefree}")

    # the host-g0 site (e261's splice bank verbatim — x24's construction)
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
    host_ids = torch.stack(
        [corpus.encode(train_text[p - G1.PRE: p]) for p, _ in install_occ])
    gbatt = {
        "form": "THE HOST-G0 SITE: e261's splice bank verbatim (the exact "
                "bank on which x24's/x37's/x38's committed cells were "
                "read; adjudicating site) == g1c's g0 battery ids",
        "install_mix": mix,
        "shape": list(host_ids.shape),
        "expected": {"mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                     "shape": [60, 130]},
        "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                     and list(host_ids.shape) == [60, 130]),
    }
    METRICS["gates"]["G_BATTERY"] = gbatt
    if not gbatt["pass"]:
        raise texture(f"G_BATTERY FAILURE: {gbatt}")

    # the neutral site (x24's frozen rule, seed 24001)
    x24_neut_cands = int(x24m["gates"]["G_NEUTRAL"]["n_candidates"])
    look = 30
    cands = [p for p in range(G1.PRE, len(train_text))
             if train_text[p].isupper() and train_text[p].isascii()
             and train_text[p].isalpha()
             and not any(h in train_text[p - G1.PRE - 8: p + look]
                         for h in G1.HOSTS)]
    nrng = random.Random(NEUTRAL_SEED)
    nrng.shuffle(cands)
    neutral_pos = cands[:60]
    neutral_ids = torch.stack(
        [corpus.encode(train_text[p - G1.PRE: p]) for p in neutral_pos])
    upper_ok = sum(1 for p in neutral_pos if train_text[p].isupper())
    hostfree_ok = sum(
        1 for p in neutral_pos
        if not any(h in train_text[p - G1.PRE - 8: p + look]
                   for h in G1.HOSTS))
    gneut = {
        "form": "THE NEUTRAL SITE (x24 verbatim): 60 train windows, 130 "
                "chars, ending uppercase-ASCII, host-free + 30 lookahead, "
                "random.Random(24001), first 60 — the site-generality "
                "rider",
        "n_candidates": len(cands),
        "x24_committed_n_candidates": x24_neut_cands,
        "shape": list(neutral_ids.shape),
        "next_char_upper": f"{upper_ok}/60",
        "host_free": f"{hostfree_ok}/60",
        "pass": bool(list(neutral_ids.shape) == [60, 130]
                     and upper_ok == 60 and hostfree_ok == 60
                     and len(cands) == x24_neut_cands),
    }
    METRICS["gates"]["G_NEUTRAL"] = gneut
    if not gneut["pass"]:
        raise texture(f"G_NEUTRAL FAILURE: {gneut}")
    sites = {"host_g0": host_ids, "neutral": neutral_ids}
    log(f"P0: G_PANEL + G_NAMEFREE + G_BATTERY + G_NEUTRAL PASS — "
        f"initials {''.join(initials)}; host mix 19/41; neutral "
        f"{len(cands)} candidates (x24 committed {x24_neut_cands})")
    write_partial("P0 panel + both sites gated")

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)

    # ============ P0b: the parents hard-bound (Rule 12) =================
    def near(a, b, tol):
        return abs(float(a) - float(b)) <= tol

    walk_reads = {s["key"]: s["read"] for s in e335m["cut1_walk"]["states"]}
    walk_key_map = {"base": "base", "inst_sib": "locked_install",
                    "own_inst": "own_install", "root": "root",
                    "locked": "locked_root", "ball": "W1_s300"}
    lit_checks = {
        "x24_verdict_word": (x24m["verdict"]["word"],
                             "NAME-SPECIFIC-STRUCTURE", None),
        "x25_fork_word": (x25m["verdict"]["fork"]["word"], "PLASTICITY",
                          None),
        "x38_verdict_word": (x38m["verdict"]["word"], X38_VERDICT_WORD,
                             None),
        "x38_rstar_bracket_bot": (
            x38m["verdict"]["ladder_split"]["R_star_bracket"][0],
            X38_RSTAR_BRACKET[0], 0.0),
        "x38_rstar_bracket_top": (
            x38m["verdict"]["ladder_split"]["R_star_bracket"][1],
            X38_RSTAR_BRACKET[1], 0.0),
        "x38_grading_envelope_x": (x38m["grading_rider"]["envelope_x"],
                                   X38_ENVELOPE_X, 0.0),
        "e335_verdict_word": (e335m["adjudication"]["verdict"],
                              "WASHES-OUT", None),
        "g1c_root_verdict_word": (g1crm["adjudication"]["verdict"],
                                  "ROOT-WALL-HOLDS", None),
        "x24_kk_cell_vs_x15_literal": (
            x24m["panel_reads"]["ZEPHYRA"]["comp_host_g0"]["mean_pz"],
            X15_DIAG_KK, 1e-12),
        "x24_kt_cell_vs_x23_literal": (
            x24m["panel_reads"]["TAVIREN"]["comp_host_g0"]["mean_pz"],
            X23_KT, 1e-12),
    }
    for key, cls in X38_LADDER_CLS.items():
        lit_checks[f"x38_class[{key}]"] = (
            x38m["ladder_clauses"][key]["response_class"], cls, None)
        lit_checks[f"x38_move[{key}]"] = (
            x38m["ladder_clauses"][key]["move_signed"],
            X38_MOVES[key][0], 0.0)
        lit_checks[f"x38_move_x[{key}]"] = (
            x38m["ladder_clauses"][key]["move_currency_x"],
            X38_MOVES[key][1], 0.0)
    for key, walk_key in walk_key_map.items():
        committed = LADDER_BY_KEY[key][5]
        lit_checks[f"e335_walk_read[{key}]"] = (
            walk_reads[walk_key], committed, 0.0)
    traj_artifact = {int(c["step"]): c["g0_pz"] for c in
                     e335m["cut1_walk"]["committed"]["install_traj"]}
    for stp, val in E335_TRAJ.items():
        lit_checks[f"e335_install_traj[s{stp}]"] = (
            traj_artifact.get(stp), val, 0.0)

    def lit_ok(v) -> bool:
        return v[0] == v[1] if v[2] is None else near(v[0], v[1], v[2])

    binds = [("ladder:" + key, CKPT_DIR / fname, LADDER_MD5[fname])
             for key, fname, *_ in LADDER]
    binds += [
        ("g1c_cons_resume", CKPT_DIR / CONS_CK, LADDER_MD5[CONS_CK]),
        ("e261_K10K_installed", CKPT_DIR / FACT_CK, FACT_MD5),
        ("e264_rooms", CKPT_DIR / ROOMS264_CK, ROOMS264_MD5),
        ("x15_metrics", X15_METRICS, X15_METRICS_MD5),
        ("x15_comp_carrier", X15_COMP_CARRIER, X15_COMP_CARRIER_MD5),
        ("x24_metrics", X24_METRICS, X24_METRICS_MD5),
        ("x25_metrics", X25_METRICS, X25_METRICS_MD5),
        ("x37_metrics", X37_METRICS, X37_METRICS_MD5),
        ("e337_metrics", E337_METRICS, E337_METRICS_MD5),
        ("e335_metrics", E335_METRICS, E335_METRICS_MD5),
        ("g1c_root_metrics", G1CROOT_METRICS, G1CROOT_METRICS_MD5),
        ("x38_metrics", X38_METRICS, X38_METRICS_MD5),
        ("e293_bank_source", E293_SOURCE, E293_SOURCE_MD5),
        ("e043_rig_source", E043_SOURCE, E043_SOURCE_MD5),
        ("g1c_rig_source", G1C_SOURCE, G1C_SOURCE_MD5),
        ("g1_rig_source", G1_SOURCE, G1_SOURCE_MD5),
        ("e261_rig_source", E261_SOURCE, E261_SOURCE_MD5),
    ]
    bind_records = {}
    for nm, p, b in binds:
        got = md5of(p)
        bind_records[nm] = {"path": str(p.relative_to(REPO)), "md5": got,
                            "bound": b, "match": got == b}
        if got != b:
            raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} {got} != {b}")
    gendp = {
        "form": "23 parents md5-bound (the six outer rungs + the cons twin "
                "+ the displacement's parents + the instrument metrics "
                "x24/x25/x37/e337/x38 + e335/g1c_root provenance + the "
                "rig's four sources) + committed references READ FROM THE "
                "ARTIFACTS and cross-checked against the frozen literals",
        "md5_binds": bind_records,
        "literal_crosscheck": {
            k: {"artifact": v[0], "frozen": v[1], "match": lit_ok(v)}
            for k, v in lit_checks.items()},
        "pass": bool(all(r["match"] for r in bind_records.values())
                     and all(lit_ok(v) for v in lit_checks.values())),
    }
    if not gendp["pass"]:
        raise texture(f"G_ENDPOINTS failure: {gendp}")
    METRICS["gates"]["G_ENDPOINTS"] = gendp
    log(f"P0b: G_ENDPOINTS PASS — {len(binds)} parents md5-bound; "
        f"{len(lit_checks)} committed references cross-checked from "
        f"artifacts, never retyped")
    write_partial("P0b parents hard-bound")

    # ============ P1: the outer rungs + flat basis + load gates ==========
    base_net = G1.load_g1(CKPT_DIR / "e001.pt")
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    named_p = list(base_net.named_parameters())
    sd_keys = [k for k, _ in named_p]
    N = base_net.num_params()
    del base_net

    # ---- THE BALL: settle first (e337's registered instrument) ----------
    ball_net = G1.load_g1(BALL_CK)                 # anchored=True on load
    wall_pre = ball_net.wall_report()
    ball_net._enforce_wall()                       # SETTLE (defined state)
    wall_post = ball_net.wall_report()
    ball_settled_sd = {k: v.detach().clone()
                       for k, v in ball_net.named_parameters()}
    del ball_net

    # ---- every outer rung's parameter dict (e335's loading conventions) -
    substrate_sds, substrate_meta = {}, {}
    for key, fname, kind, net0_cls, formation, committed in LADDER:
        if kind == "resume-model":
            st = torch.load(CKPT_DIR / fname, map_location="cpu",
                            weights_only=False)
            net = G1.evl_load(st["model"])
            del st
        elif kind.startswith("plain-anchored"):    # the ball: SETTLED
            net = G1.evl_load(ball_settled_sd)
        else:
            net = G1.load_g1(CKPT_DIR / fname)
        kp = [k for k, _ in net.named_parameters()]
        substrate_sds[key] = {k: v.detach().clone()
                              for k, v in net.named_parameters()}
        substrate_meta[key] = {
            "file": f"runs/checkpoints/{fname}",
            "md5": LADDER_MD5[fname], "kind": kind,
            "net0_class": net0_cls, "formation": formation,
            "params": net.num_params(),
            "param_key_order_matches_base": bool(kp == sd_keys),
            "committed_walk_read": committed,
            "role": "OUTER REFERENCE (x38's committed rung)",
        }
        del net
    substrate_meta["ball"]["wall_report_pre_settle"] = wall_pre
    substrate_meta["ball"]["wall_report_post_settle"] = wall_post
    substrate_meta["ball"]["settle_fired"] = bool(
        wall_pre["d_raw"] > wall_pre["R"])
    METRICS["outer_ladder"] = substrate_meta

    gflat = {
        "form": "flat basis: the base's parameter key order is the flat "
                "space; EVERY substrate's parameter key order must match "
                "it exactly (anchor buffers excluded from the flat space)",
        "params_count": len(named_p),
        "n_params": N,
        "n_params_expected": N_PARAMS,
        "rungs": {k: v["param_key_order_matches_base"]
                  for k, v in substrate_meta.items()},
        "pass": bool(N == N_PARAMS
                     and all(v["param_key_order_matches_base"]
                             and v["params"] == N_PARAMS
                             for v in substrate_meta.values())),
    }
    if not gflat["pass"]:
        raise texture(f"FLAT-BASIS GATE FAILURE: {gflat}")
    METRICS["gates"]["G_FLATBASIS"] = gflat

    # ---- G_LADDERLOAD: every outer rung's bare host read + root==cons ---
    subload = {}
    for key, fname, kind, net0_cls, formation, committed in LADDER:
        net_s = G1.evl_load(substrate_sds[key])
        read = float(site_read(net_s, host_ids)["probs"][:, zid].mean())
        subload[key] = {"mine": read, "committed_walk": committed,
                        "abs_diff": abs(read - committed), "tol": READ_TOL,
                        "net0_class": net0_cls,
                        "pass": bool(abs(read - committed) <= READ_TOL)}
        del net_s
    # the ball's settle consistency: settled-anchored read == disarmed read
    ball_disarm_net = G1.evl_load(substrate_sds["ball"])
    ball_disarm_read = float(
        site_read(ball_disarm_net, host_ids)["probs"][:, zid].mean())
    del ball_disarm_net
    subload["ball"]["disarmed_bare_read"] = ball_disarm_read
    subload["ball"]["disarm_vs_settled_abs_diff"] = abs(
        ball_disarm_read - subload["ball"]["mine"])
    subload["ball"]["disarm_tol"] = REPRO_TOL
    subload["ball"]["pass"] = bool(
        subload["ball"]["pass"]
        and abs(ball_disarm_read - subload["ball"]["mine"]) <= REPRO_TOL)
    # root == cons (bit-exact; e335's G_ROOTIDENT, re-run)
    st_c = torch.load(CKPT_DIR / CONS_CK, map_location="cpu",
                      weights_only=False)
    st_r = torch.load(ROOT_CK, map_location="cpu", weights_only=False)
    root_cons_bit = (
        set(st_c["model"]) == set(st_r["model"])
        and all(torch.equal(st_c["model"][k], st_r["model"][k])
                for k in st_r["model"]))
    del st_c, st_r
    subload["root_cons_bit_identical"] = bool(root_cons_bit)
    rung_keys_early = [l[0] for l in LADDER]
    subload["net0_classes_recorded"] = bool(
        all(v.get("net0_class") for k, v in subload.items()
            if k in rung_keys_early))
    subload["pass"] = bool(all(v.get("pass", False)
                               for k, v in subload.items()
                               if k in rung_keys_early)
                           and root_cons_bit
                           and subload["net0_classes_recorded"])
    METRICS["gates"]["G_LADDERLOAD"] = subload
    if not subload["pass"]:
        raise texture(f"G_LADDERLOAD FAILURE: {subload}")
    log("P1: G_FLATBASIS + G_LADDERLOAD PASS — "
        + "; ".join(f"{k} {subload[k]['mine']:.6f} (|d| "
                    f"{subload[k]['abs_diff']:.1e})"
                    for k, *_ in LADDER)
        + f"; ball settle fired {substrate_meta['ball']['settle_fired']}"
          f" (d_raw {wall_pre['d_raw']:.4f} -> {wall_post['d_proj']:.4f}"
          f" = R); root==cons bit-identical {root_cons_bit}")
    write_partial("P1 the six outer rungs loaded + behaviorally gated")

    # THE BRACKET as measured in this cell (frozen selection bounds)
    bracket_bot = subload["base"]["mine"]
    bracket_top = subload["own_inst"]["mine"]
    log(f"THE BRACKET (measured): ({bracket_bot:.6e}, {bracket_top:.6e}]")

    def unflat_like_base(flat64: np.ndarray) -> dict:
        out, off = {}, 0
        for k in sd_keys:
            n = base_sd[k].numel()
            out[k] = torch.from_numpy(
                np.ascontiguousarray(flat64[off:off + n])) \
                .reshape(base_sd[k].shape)
            off += n
        return out

    # ============ P2: THE ROOM + THE COMPLEMENT (x38 verbatim) ==========
    room = SRCT(N_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    del rooms264
    cert_rng = np.random.default_rng(CERT_SEED)
    idem, kept2 = [], []
    for _ in range(2):
        x = cert_rng.standard_normal(N_PARAMS)
        px = room.project(x)
        ppx = room.project(px)
        idem.append(float(np.linalg.norm(ppx - px) / np.linalg.norm(px)))
        kept2.append(float((px @ px) / (x @ x)))
    groom = {
        "form": "the committed K10K room (SRCT k=10,000, seeds 26113/26114): "
                "D/S bit-identical to e264_rooms.pt; light in-cell "
                "certification (x17/x23/x24/x37/x38's convention)",
        "D_bit_equal": bool(np.array_equal(room.D, D264)),
        "S_bit_equal": bool(np.array_equal(room.S, S264)),
        "idempotency_max": max(idem),
        "kept2_mean": float(np.mean(kept2)),
        "kept2_expect": ROOM_K / N_PARAMS,
        "pass": bool(np.array_equal(room.D, D264)
                     and np.array_equal(room.S, S264)
                     and max(idem) <= 1e-8
                     and abs(float(np.mean(kept2)) - ROOM_K / N_PARAMS)
                     <= 5.0 * math.sqrt(2.0 * ROOM_K) / N_PARAMS),
    }
    if not groom["pass"]:
        raise texture(f"ROOM GATE FAILURE: {groom}")
    METRICS["gates"]["G_ROOM"] = groom
    log(f"P2: G_ROOM PASS — K10K room D/S BIT-BOUND (idem {max(idem):.1e})")

    fact_art = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                          weights_only=False)
    fact_sd = {k: v.detach().clone() for k, v in fact_art["model"].items()
               if k in sd_keys}
    del fact_art
    dW_k64 = {k: fact_sd[k].double() - base_sd[k].double() for k in sd_keys}
    dW_k_flat = torch.cat([dW_k64[k].reshape(-1)
                           for k in sd_keys]).numpy()
    DWK_L2 = float(np.linalg.norm(dW_k_flat))
    if abs(DWK_L2 - K10K_WRITE_NORM) > 1e-9:
        raise texture(f"K10K write norm {DWK_L2} != {K10K_WRITE_NORM}")
    E_FULL_K = float(dW_k_flat @ dW_k_flat)
    pin_k = room.project(dW_k_flat)
    comp_k_flat = dW_k_flat - pin_k
    E_IN_K = float(pin_k @ pin_k)
    E_OUT_K = float(comp_k_flat @ comp_k_flat)
    gorth = {
        "energy_sum_rel_resid": abs(E_IN_K + E_OUT_K - E_FULL_K) / E_FULL_K,
        "cross_dot_rel": abs(float(pin_k @ comp_k_flat)) / E_FULL_K,
        "bar": 1e-9,
    }
    gorth["pass"] = bool(gorth["energy_sum_rel_resid"] < 1e-9
                         and gorth["cross_dot_rel"] < 1e-9)
    if not gorth["pass"]:
        raise texture(f"G_ORTH (K10K write) FAILURE: {gorth}")
    METRICS["gates"]["G_ORTH"] = gorth

    COMP_K_L2 = float(np.linalg.norm(comp_k_flat))
    s_full_k = DWK_L2 / COMP_K_L2                    # x15's exact arithmetic
    scaled_k64 = s_full_k * comp_k_flat

    scaled_k32 = {k: v.float()
                  for k, v in unflat_like_base(scaled_k64).items()}
    fl_k32 = np.concatenate([scaled_k32[k].double().numpy().reshape(-1)
                             for k in sd_keys])
    gdose = {
        "form": "||s_full_k * comp_k|| == ||dW_k|| to 1e-12 (fp64, derived "
                "never trusted)",
        "s_full": s_full_k, "committed": X15_S_FULL,
        "s_full_abs_diff": abs(s_full_k - X15_S_FULL),
        "scaled_fp64_l2": float(np.linalg.norm(scaled_k64)),
        "write_l2": DWK_L2,
        "abs_diff": abs(float(np.linalg.norm(scaled_k64)) - DWK_L2),
        "bar": 1e-12,
        "pass": bool(abs(float(np.linalg.norm(scaled_k64)) - DWK_L2) <= 1e-12
                     and abs(s_full_k - X15_S_FULL) <= 1e-15),
    }
    ginroom = {
        "form": "x15's exact G_INROOM_SCALED form (in-room ENERGY share "
                "<= 1e-12; fp64 intended AND fp32 injected)",
        "scaled_fp64": inroom_energy_share(room, scaled_k64),
        "fp32_injected": inroom_energy_share(room, fl_k32),
        "bar": 1e-12,
    }
    ginroom["pass"] = bool(ginroom["scaled_fp64"] <= 1e-12
                           and ginroom["fp32_injected"] <= 1e-12)
    if not (gdose["pass"] and ginroom["pass"]):
        raise texture(f"DOSE/INROOM FAILURE: {gdose} {ginroom}")
    METRICS["gates"]["G_DOSEMATCH"] = gdose
    METRICS["gates"]["G_INROOM"] = ginroom

    # G_REGEN: the complement vs its CARRIER (x24/x37/x38's convention)
    ck_k = torch.load(X15_COMP_CARRIER, map_location="cpu",
                      weights_only=False)
    bit_equal = {k: bool(torch.equal(ck_k["delta"][k].float(), scaled_k32[k]))
                 for k in sd_keys}
    model_ok = all(torch.equal(ck_k["model"][k],
                               base_sd[k] + ck_k["delta"][k])
                   for k in sd_keys)
    gregen = {
        "form": "the regenerated full-dose complement (fp32) vs the "
                "committed carrier's on-disk delta — bit-equal on EVERY "
                "key; + natural fp64 comp L2 vs committed (1e-9); + "
                "carrier model == base + delta re-derived",
        "carrier": str(X15_COMP_CARRIER.relative_to(REPO)),
        "keys_bit_equal": int(sum(bit_equal.values())),
        "keys_total": len(sd_keys),
        "all_keys_bit_equal": all(bit_equal.values()),
        "my_comp64_l2": COMP_K_L2,
        "committed_l2": X15_COMP_L2_64,
        "l2_abs_diff": abs(COMP_K_L2 - X15_COMP_L2_64),
        "l2_bar": 1e-9,
        "carrier_model_eq_base_plus_delta": bool(model_ok),
        "pass": bool(all(bit_equal.values())
                     and abs(COMP_K_L2 - X15_COMP_L2_64) < 1e-9 and model_ok),
    }
    if not gregen["pass"]:
        raise texture(f"G_REGEN FAILURE: {gregen}")
    METRICS["gates"]["G_REGEN"] = gregen
    log(f"P2b: G_ORTH + G_DOSEMATCH + G_INROOM + G_REGEN PASS — the "
        f"displacement BIT-EQUAL to x15's committed carrier (every key)")
    write_partial("P2 room + complement bit-equal to the carrier")

    # ==================================================================
    # P3: THE INSTALL RE-RUN (the interior construction) ===============
    # ==================================================================
    log(f"P3: THE INSTALL RE-RUN — e001 + e043-Dmix s{INST_STEPS}/"
        f"total={INST_TOTAL}, gen {FRESH_GEN}, CPU threads 4, per-step "
        f"battery reads")
    # the rig's own batch constituents (g1c's construction VERBATIM)
    name_ids = corpus.encode(G1.NAME)
    g1c_meta = torch.load(ROOT_CK, map_location="cpu",
                          weights_only=False)["meta"]

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
    names_in_place = bool(all(
        torch.equal(w[G1.PRE: G1.PRE + len(G1.NAME)], name_ids)
        for w in win_i))
    ginmask = {
        "form": "g1c's install windows + mask VERBATIM (16 spliced install "
                "windows w/ name mask + the original-host anchor bank)",
        "inst_x_shape": list(inst_x.shape),
        "anchor_full_shape": list(anchor_full.shape),
        "name_positions": int(inst_mask.sum()),
        "expected_name_positions": 60 * len(G1.NAME),
        "all_windows_name_in_place": names_in_place,
        "pass": bool(list(inst_x.shape) == [60, G1.BLOCK]
                     and int(inst_mask.sum()) == 60 * len(G1.NAME)
                     and names_in_place),
    }
    METRICS["gates"]["G_INSTMASK"] = ginmask
    if not ginmask["pass"]:
        raise texture(f"G_INSTMASK FAILURE: {ginmask}")

    name_bs, corp_bs, mix_random, lr = (G1.NAME_BS, E43.CORP_BS,
                                        E43.MIX_RANDOM, E43.LR)
    grig = {
        "form": "the rig's constants cross-checked vs the e043/G1 sources "
                "AND g1c_root.pt's own meta (the committed root draw)",
        "name_bs": name_bs, "corp_bs": corp_bs, "mix_random": mix_random,
        "lr": lr, "betas": "(0.9, 0.95)", "weight_decay": 0.1,
        "clip": 1.0, "total": INST_TOTAL, "warmup": 100,
        "cosine": "house cosine_lr(step-1, total=1000, warmup=100)",
        "gen_seed": FRESH_GEN,
        "g1c_meta_install_seed": g1c_meta.get("install_seed"),
        "g1c_meta_install_steps": g1c_meta.get("install_steps"),
        "g1c_recipe_names_dmix": bool(
            "e043-Dmix" in str(g1crm["root_build"]["install"]["recipe"])),
        "e43_corpbs_eq_g1_corpbs": bool(E43.CORP_BS == G1.CORP_BS == 48
                                        and E43.MIX_RANDOM == G1.MIX_RANDOM
                                        and E43.LR == 1e-3),
        "pass": bool(name_bs == 16 and corp_bs == 48 and mix_random == 32
                     and lr == 1e-3 and INST_TOTAL == 1000
                     and g1c_meta.get("install_seed") == FRESH_GEN
                     and g1c_meta.get("install_steps") == 400
                     and "e043-Dmix" in str(
                         g1crm["root_build"]["install"]["recipe"])),
    }
    METRICS["gates"]["G_RIGCONST"] = grig
    if not grig["pass"]:
        raise texture(f"G_RIGCONST FAILURE: {grig}")

    net0 = G1.evl_load(base_sd)                  # g1c's own net0 form
    net0_params = {k: v.detach().clone()
                   for k, v in net0.named_parameters()}
    base_bit = all(torch.equal(net0_params[k], substrate_sds["base"][k])
                   for k in sd_keys)
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(FRESH_GEN)
    evl = copy.deepcopy(net0)                    # the measurement twin
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    traj = []                                    # per-step battery reads
    captured: dict[int, dict] = {}               # step -> params dict
    crossing_step = {T: None for T in INTERIOR_GRID}
    first_draws = []
    t_inst0 = time.time()
    stopped_early = None
    for step in range(1, INST_STEPS + 1):
        f = cosine_lr(step - 1, INST_TOTAL)          # house schedule
        for g in opt.param_groups:
            g["lr"] = lr * f
        ix = torch.randint(n_inst, (name_bs,), generator=gen)
        aj = torch.randint(n_anc, (corp_bs - mix_random,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                           generator=gen)
        if step <= 3:
            first_draws.append({
                "step": step,
                "ix": [int(v) for v in ix],
                "aj": [int(v) for v in aj],
                "rj_head": [int(v) for v in rj[:8]],
            })
        corp = torch.cat([anchor_full[aj],
                          torch.stack([train_ids[s: s + G1.BLOCK]
                                       for s in rj])], 0)
        nw = inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        msk = torch.zeros(name_bs + corp_bs, x.shape[1], dtype=torch.bool)
        msk[:name_bs] = inst_mask[ix]
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:name_bs][msk[:name_bs]]
        cm = nll[name_bs:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        # ---- the per-step read (the rig's own instrument, on the twin)
        sd_cpu = {k: v.detach().clone() for k, v in net.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        bz = G1.battery_cell(evl, host_ids, zid)
        traj.append({"step": step, "g0_pz": bz["mean_pz"],
                     "ce_batch": float(loss.item())})
        want_capture = (step in CERT_STEPS
                        or (SMOKE and step in SMOKE_RUNG_STEPS))
        for T in INTERIOR_GRID:
            if crossing_step[T] is None and bz["mean_pz"] >= T:
                crossing_step[T] = step
                want_capture = True
        if want_capture and step not in captured:
            captured[step] = {k: v.detach().clone()
                              for k, v in net.named_parameters()}
            captured[step]["__ce_r"] = float(
                G1.ce_fixed_cpu(evl, r_eval_x, r_eval_y))
        if step % 25 == 0 or step == 1:
            log(f"  [install] s{step:4d} g0 {bz['mean_pz']:.6f} "
                f"CE {float(loss.item()):.4f} "
                f"({time.time() - t_inst0:.0f}s)")
        if (not SMOKE and time.time() - t_inst0 > INST_TIME_BUDGET_S
                and step >= 100):
            stopped_early = step
            break
    del net, opt
    METRICS["install_rerun"] = {
        "form": "THE INTERIOR CONSTRUCTION: the root's OWN install "
                "(e001 + e043-Dmix, gen 24314) re-run on CPU with "
                "per-step battery reads; states captured in-memory at "
                "cert steps + first grid crossings",
        "steps_run": traj[-1]["step"] if traj else 0,
        "stopped_early_at": stopped_early,
        "elapsed_s": round(time.time() - t_inst0, 1),
        "first_draws": first_draws,
        "crossing_steps": {f"{T:g}": crossing_step[T]
                           for T in INTERIOR_GRID},
        "traj_every_step": traj,
        "note": "the committed run was one CUDA burst (19.4 s for 400 "
                "steps); this CPU re-run follows the same batch stream "
                "(device-independent generator draws) but not its fp "
                "arithmetic — certified by the label rule below",
    }
    log(f"P3: install re-run done — {len(traj)} steps in "
        f"{time.time() - t_inst0:.0f}s; crossings "
        + str({f'{T:g}': crossing_step[T] for T in INTERIOR_GRID}))

    # ---- G_INSTREBUILD: the structural certification (hard) ------------
    steps_run = traj[-1]["step"] if traj else 0
    ginstr = {
        "form": "the reconstruction's STRUCTURAL certification (hard) + "
                "the numeric label (descriptive, frozen rule)",
        "base_bit_loaded": bool(base_bit),
        "net0_is_g1c_form": "G1.evl_load(base_sd) — g1c's own net0",
        "gen_seed": FRESH_GEN,
        "first_draws_recorded": bool(len(first_draws) == min(3, steps_run)),
        "reached_s100": bool(steps_run >= 100 or SMOKE),
        "steps_run": steps_run,
    }
    # the numeric label (frozen rule)
    cert_cells = {}
    for stp in CERT_STEPS:
        if stp <= steps_run and stp in E335_TRAJ:
            mine = traj[stp - 1]["g0_pz"]
            committed = E335_TRAJ[stp]
            tol = CERT_TOL_S1 if stp == 1 else CERT_TOL_LATE
            cert_cells[f"s{stp}"] = {
                "mine": mine, "committed": committed,
                "abs_diff": abs(mine - committed), "tol": tol,
                "within": bool(abs(mine - committed) <= tol)}
    ginstr["cert_cells"] = cert_cells
    s1_ok = cert_cells.get("s1", {}).get("within", False)
    late_cells = [c for k, c in cert_cells.items() if k != "s1"]
    if all(c["within"] for c in cert_cells.values()):
        label = ("COMMITTED-TRAJECTORY (the committed install's own "
                 "mid-formation states, CPU reconstruction)")
    elif s1_ok:
        label = ("PARTIAL-DRIFT (step-1 within tol; some later cert cells "
                 "outside 0.30 — disclosed)")
    else:
        label = ("FRESH-REDRAW (step-1 outside 2e-6 — the dispatch's "
                 "(a)-branch: a fresh CPU re-draw of the e043-Dmix "
                 "construction; each rung a fresh read level)")
    ginstr["label"] = label
    ginstr["label_rule"] = REGISTERED["clauses_fixed"]["cert_label_rule"]
    ginstr["pass"] = bool(base_bit and ginstr["first_draws_recorded"]
                          and ginstr["reached_s100"]
                          and (cert_cells.get("s1") is not None))
    METRICS["gates"]["G_INSTREBUILD"] = ginstr
    if not ginstr["pass"]:
        raise texture(f"G_INSTREBUILD FAILURE: {ginstr}")
    log(f"P3b: G_INSTREBUILD PASS — label {label}; cert cells: "
        + "; ".join(f"{k} {c['mine']:.4g} vs {c['committed']:.4g} "
                    f"({'OK' if c['within'] else 'OUT'})"
                    for k, c in cert_cells.items()))
    write_partial("P3 the install re-run + reconstruction certified")

    # ---- the rung selection (frozen rule) ------------------------------
    interior_defs = []          # (key, step, threshold_or_None)
    if 1 in captured:
        interior_defs.append(("ins_s1", 1, None))
    for T in INTERIOR_GRID:
        stp = crossing_step[T]
        if stp is not None and stp in captured and not SMOKE:
            interior_defs.append((f"ins_t{str(T).replace('.', 'p')}",
                                  stp, T))
    if SMOKE:
        interior_defs = [(f"ins_s{s}", s, None)
                         for s in SMOKE_RUNG_STEPS if s in captured]

    # qualify by measured host read inside the bracket
    qualified, overshoot = [], []
    for key, stp, T in interior_defs:
        net_s = G1.evl_load({k: v for k, v in captured[stp].items()
                             if not k.startswith("__")})
        read = float(site_read(net_s, host_ids)["probs"][:, zid].mean())
        rec = {"key": key, "formation_step": stp, "grid_threshold": T,
               "battery_read_inloop": traj[stp - 1]["g0_pz"],
               "host_site_read": read,
               "construction": ("e001 + e043-Dmix gen-24314 install, "
                                f"re-formed on CPU, step {stp}/400 "
                                "(cosine total=1000)"),
               "net0_class": "BASE-forming"}
        if SMOKE or (bracket_bot < read <= bracket_top):
            qualified.append((rec, net_s))
        else:
            overshoot.append(rec)
        del net_s
    METRICS["interior_overshoots"] = overshoot
    min_rungs = 1 if SMOKE else MIN_INTERIOR_RUNGS
    priors_list = [r["host_site_read"] for r, _ in qualified] or [1.0]
    ginterior = {
        "form": "the rung selection: s1 + first-crossing states of the "
                "grid {0.01..0.50} whose measured host read lies inside "
                "the bracket (base_read, own_inst_read]; >= 4 rungs "
                "spanning >= 1 order required (SMOKE: waived)",
        "n_rungs": len(qualified),
        "n_required": min_rungs,
        "span_x": (max(priors_list) / min(priors_list))
                  if len(priors_list) > 1 else None,
        "span_required_x": INTERIOR_SPAN_X,
        "bracket": [bracket_bot, bracket_top],
        "overshoots": len(overshoot),
        "disclosures": [r for r, _ in qualified],
        "pass": bool(len(qualified) >= min_rungs
                     and (SMOKE or len(priors_list) < 2
                          or (max(priors_list) / min(priors_list))
                          >= INTERIOR_SPAN_X)),
    }
    METRICS["gates"]["G_INTERIOR"] = ginterior
    if not ginterior["pass"]:
        raise texture(f"G_INTERIOR FAILURE: {ginterior}")
    log(f"P3c: G_INTERIOR PASS — {len(qualified)} interior rungs "
        + "; ".join(f"{r['key']} s{r['formation_step']} read "
                    f"{r['host_site_read']:.4g}"
                    for r, _ in qualified)
        + (f"; {len(overshoot)} overshoots disclosed"
           if overshoot else ""))

    # register the interior substrates
    for rec, net_s in qualified:
        key = rec["key"]
        sd_q = {k: v.detach().clone()
                for k, v in net_s.named_parameters()}
        substrate_sds[key] = sd_q
        substrate_meta[key] = {
            "file": None,
            "construction": rec["construction"],
            "formation_step": rec["formation_step"],
            "grid_threshold": rec["grid_threshold"],
            "battery_read_inloop": rec["battery_read_inloop"],
            "kind": "re-formed-install",
            "net0_class": rec["net0_class"],
            "params": N_PARAMS,
            "param_key_order_matches_base": True,
            "committed_walk_read": None,
            "role": "INTERIOR RUNG (this cell's construction)",
        }
        del net_s
    interior_keys = [r["key"] for r, _ in qualified]
    cert_keys = []
    for stp in CERT_STEPS:                      # the cert states (rider only)
        if stp in captured and stp not in [r["formation_step"]
                                           for r, _ in qualified]:
            ck = f"cert_s{stp}"
            sd_c = {k: v for k, v in captured[stp].items()
                    if not k.startswith("__")}
            substrate_sds[ck] = sd_c
            substrate_meta[ck] = {
                "file": None,
                "construction": f"the reconstruction's cert state s{stp}",
                "formation_step": stp, "grid_threshold": None,
                "battery_read_inloop": traj[stp - 1]["g0_pz"],
                "kind": "re-formed-install", "net0_class": "BASE-forming",
                "params": N_PARAMS, "param_key_order_matches_base": True,
                "committed_walk_read": E335_TRAJ.get(stp),
                "role": "CERT RIDER (never a rung)",
            }
            cert_keys.append(ck)
    METRICS["ladder"] = substrate_meta

    # ============ P4: THE STATES =========================================
    displ = {"comp": scaled_k32}

    def apply_disp(sub_key: str, disp_key: str) -> dict:
        """x15's fp32 injection: substrate_fp32 + delta_fp32 per key."""
        return {k: substrate_sds[sub_key][k] + displ[disp_key][k]
                for k in sd_keys}

    state_defs = []
    for key, *_ in LADDER:
        state_defs.append((key, key, None))
        state_defs.append((f"{key}_comp", key, "comp"))
    for key in interior_keys:
        state_defs.append((key, key, None))
        state_defs.append((f"{key}_comp", key, "comp"))
    for key in cert_keys:
        state_defs.append((key, key, None))
        state_defs.append((f"{key}_comp", key, "comp"))
    states = {}
    for st, sub, dk in state_defs:
        sd = substrate_sds[sub] if dk is None else apply_disp(sub, dk)
        net = G1.evl_load(sd)                    # DISARMED (params-only)
        states[st] = {
            "substrate": sub, "displacement": dk or "bare",
            "site_passes": {s: site_read(net, ids)
                            for s, ids in sites.items()},
            "ce_r": float(G1.ce_fixed_cpu(net, r_eval_x, r_eval_y)),
            "source": (f"{sub} + {dk} (x15's fp32 injection, disarmed)"
                       if dk else f"{sub} (bare)"),
        }
        del net
    # the base_comp state MUST re-derive the carrier's own model bit-exact
    carrier_model_ok = all(
        torch.equal(ck_k["model"][k],
                    substrate_sds["base"][k] + displ["comp"][k])
        for k in sd_keys)   # re-derived HERE on the base substrate
    del ck_k
    gone_state = {
        "form": "the displacement vectors are FIXED vectors applied "
                "identically to every rung (substrate_fp32 + delta_fp32); "
                "on the base this re-derives the carrier's model bit-exact "
                "(verified in G_REGEN and re-derived on the base substrate "
                "here)",
        "carrier_model_eq_base_plus_complement": bool(carrier_model_ok),
        "pass": bool(carrier_model_ok),
    }
    METRICS["gates"]["G_ONESTATE"] = gone_state
    if not gone_state["pass"]:
        raise texture(f"G_ONESTATE FAILURE: {gone_state}")
    log(f"P4: {len(states)} states read "
        f"(6 outer + {len(interior_keys)} interior + {len(cert_keys)} cert "
        f"runners, x {{bare,+comp}}, 2 sites); ce_r — "
        + " ".join(f"{k} {v['ce_r']:.3f}" for k, v in list(states.items())
                   [::4]))
    METRICS["ce_r"] = {k: v["ce_r"] for k, v in states.items()}
    write_partial("P4 the states read at both sites")

    # ============ P5: the panel table ====================================
    reads = {}
    for nm, cls, cohort, _ in PANEL:
        cid = stoi[nm[0]]
        reads[nm] = {"class": cls, "cohort": cohort, "cid": cid}
        for st in states:
            for s in sites:
                reads[nm][f"{st}_{s}"] = name_read(
                    states[st]["site_passes"][s], cid)
    # lifts + moves per (rung, site) — cert states included (the rider)
    all_rung_keys = [l[0] for l in LADDER] + interior_keys
    panel_subs = all_rung_keys + cert_keys
    for nm in reads:
        for sub in panel_subs:
            st = f"{sub}_comp"
            if f"{st}_host_g0" not in reads[nm]:
                continue
            for s in sites:
                b = reads[nm][f"{sub}_{s}"]["mean_pz"]
                d = reads[nm][f"{st}_{s}"]["mean_pz"]
                reads[nm][f"{st}_lift_{s}"] = d / max(b, PRIOR_FLOOR)
                reads[nm][f"{st}_move_{s}"] = d - b
    METRICS["panel_reads"] = reads
    write_partial("P5 the panel read "
                  f"({len(states)} states x 2 sites x 11 names)")

    # ---- G_X38REPRO: the outer rungs reproduce x38 bit-exact -----------
    cells = {}
    for nm, *_ in PANEL:
        for key, *_ in LADDER:
            for cond in ("bare", "comp"):
                for s in sites:
                    mine_st = key if cond == "bare" else f"{key}_comp"
                    mine = reads[nm][f"{mine_st}_{s}"]["mean_pz"]
                    theirs = x38m["panel_reads"][nm][f"{mine_st}_{s}"]\
                        ["mean_pz"]
                    cells[f"{nm}|{mine_st}|{s}"] = {
                        "mine": mine, "committed": theirs,
                        "abs_diff": abs(mine - theirs)}
    worst = max(cells.values(), key=lambda r: r["abs_diff"])
    gx38 = {
        "form": "x38's committed outer-rung panel cells reproduce "
                "bit-exact (12 states x 2 sites x 11 names; tol 1e-12) — "
                "'the outer rungs reproduce' (the dispatch's gate)",
        "n_cells": len(cells),
        "worst_abs_diff": worst["abs_diff"],
        "worst_cell": next(k for k, v in cells.items() if v is worst),
        "tol": REPRO_TOL,
        "rows": cells,
        "pass": bool(worst["abs_diff"] <= REPRO_TOL),
    }
    METRICS["gates"]["G_X38REPRO"] = gx38
    if not gx38["pass"]:
        raise texture(f"G_X38REPRO FAILURE: worst {gx38['worst_cell']} |d| "
                      f"{worst['abs_diff']:.3e} > {REPRO_TOL}")
    log(f"P5b: G_X38REPRO PASS ({len(cells)} cells, worst |d| "
        f"{worst['abs_diff']:.1e})")

    # G_CER: the overlapping ce_r cells (tol 1e-9, e337/x38's convention)
    cer_cells = {}
    for key, *_ in LADDER:
        for cond in ("", "_comp"):
            st = f"{key}{cond}"
            cer_cells[st] = {
                "mine": METRICS["ce_r"][st],
                "committed": x38m["ce_r"][st],
                "abs_diff": abs(METRICS["ce_r"][st] - x38m["ce_r"][st])}
    worst_ce = max(cer_cells.values(), key=lambda r: r["abs_diff"])
    gcer = {
        "form": "the overlapping ce_r cells reproduce (tol 1e-9 — the "
                "fp32-mean cross-session convention, e337's disclosure)",
        "n_cells": len(cer_cells),
        "worst_abs_diff": worst_ce["abs_diff"],
        "worst_cell": next(k for k, v in cer_cells.items() if v is worst_ce),
        "tol": CE_REPRO_TOL,
        "rows": cer_cells,
        "pass": bool(worst_ce["abs_diff"] <= CE_REPRO_TOL),
    }
    METRICS["gates"]["G_CER"] = gcer
    if not gcer["pass"]:
        raise texture(f"G_CER FAILURE: {gcer}")
    log(f"P5c: G_CER PASS ({len(cer_cells)} cells, worst |d| "
        f"{worst_ce['abs_diff']:.2e} <= 1e-9)")
    write_partial("P5b-c reproduction gates passed bit-exact")

    if SMOKE:
        METRICS["status"] = ("SMOKED — full gate path + all reads "
                             "exercised; NOTHING adjudicated")
        write_partial("SMOKE COMPLETE — nothing adjudicated")
        log("SMOKE COMPLETE (nothing adjudicated)")
        raise SystemExit(0)

    # ============ P6: THE INTERIOR ADJUDICATION (frozen) ================
    def rung_clauses(key: str) -> dict:
        pb = reads["ZEPHYRA"][f"{key}_host_g0"]["mean_pz"]
        pc = reads["ZEPHYRA"][f"{key}_comp_host_g0"]["mean_pz"]
        lift = pc / max(pb, PRIOR_FLOOR)
        move = abs(pc - pb)
        band_ratio = max(reads[nm][f"{key}_comp_lift_host_g0"]
                         for nm in COHORT_A)
        band_move = max(abs(reads[nm][f"{key}_comp_move_host_g0"])
                        for nm in COHORT_A)
        living = bool(pb > LIVING_BAR)
        selected = "move" if living else "ratio"
        ratio_x = lift / band_ratio if band_ratio > 0 else float("inf")
        move_x = move / band_move if band_move > 0 else float("inf")
        if living:
            cls = "COLLAPSE" if move_x > COLLAPSE_X else "MIXED"
        else:
            cls = "LIFT" if lift > LIFT_BAR else "MIXED"
        return {
            "prior_host": pb, "comp_host": pc,
            "lift": lift, "move": move, "move_signed": pc - pb,
            "living_read_rule": {"p_base_gt_0.05": living,
                                 "selected_currency": selected},
            "band_ratio": band_ratio, "band_move": band_move,
            "ratio_currency_x": ratio_x, "move_currency_x": move_x,
            "response_class": cls,
            "neutral_rider": {
                "prior": reads["ZEPHYRA"][f"{key}_neutral"]["mean_pz"],
                "comp": reads["ZEPHYRA"][f"{key}_comp_neutral"]["mean_pz"],
                "move_signed":
                    reads["ZEPHYRA"][f"{key}_comp_move_neutral"],
            },
        }

    ladder_cls = {key: rung_clauses(key) for key in all_rung_keys}
    METRICS["ladder_clauses"] = ladder_cls
    for key in cert_keys:                      # the cert rider (never a bar)
        METRICS.setdefault("cert_rider", {})[key] = rung_clauses(key)

    interior_cls = {k: ladder_cls[k] for k in interior_keys}
    lift_rungs = [k for k, v in ladder_cls.items()
                  if v["response_class"] == "LIFT"]
    coll_rungs = [k for k, v in ladder_cls.items()
                  if v["response_class"] == "COLLAPSE"]
    mixed_rungs = [k for k, v in ladder_cls.items()
                   if v["response_class"] == "MIXED"]
    L_pri = max((ladder_cls[k]["prior_host"] for k in lift_rungs),
                default=None)
    C_pri = min((ladder_cls[k]["prior_host"] for k in coll_rungs),
                default=None)
    monotone = bool(lift_rungs and coll_rungs
                    and L_pri < C_pri
                    and max(ladder_cls[k]["prior_host"]
                            for k in lift_rungs)
                    < min(ladder_cls[k]["prior_host"] for k in coll_rungs))
    width_orders = (math.log10(C_pri / L_pri)
                    if (L_pri and C_pri and L_pri > 0) else None)
    mixed_inside = bool(
        all(L_pri < ladder_cls[k]["prior_host"] <= C_pri
            for k in mixed_rungs)) if (L_pri and C_pri) else False
    mixed_span = (max(ladder_cls[k]["prior_host"] for k in mixed_rungs)
                  / min(ladder_cls[k]["prior_host"] for k in mixed_rungs)
                  if mixed_rungs else None)

    sharp = bool(lift_rungs and coll_rungs and monotone and mixed_inside
                 and width_orders is not None
                 and width_orders <= WIDTH_BAR_ORDERS)
    graded = bool((not sharp) and mixed_rungs and mixed_span is not None
                  and mixed_span >= GRADED_SPAN_X)

    if sharp:
        word = "SHARP-INTERIOR"
        clause = (
            f"a narrow R* window separates all-lift-below from "
            f"all-collapse-above with the mixed zone inside it — "
            f"{len(lift_rungs)} LIFT / {len(coll_rungs)} COLLAPSE / "
            f"{len(mixed_rungs)} MIXED; R* = ({L_pri:.6g}, {C_pri:.6g}] "
            f"(width {width_orders:.3f} orders <= 1.0); dead-or-alive "
            f"confirmed at fine grain; R* joins the doc's Law 7 as a "
            f"located constant")
    elif graded:
        word = "GRADED-INTERIOR"
        amb_bits = "; ".join(
            f"{k} (prior {ladder_cls[k]['prior_host']:.4g}, "
            + (f"move_x {ladder_cls[k]['move_currency_x']:.2f} <= 5x "
               f"band {ladder_cls[k]['band_move']:.4g})"
               if ladder_cls[k]["living_read_rule"]["p_base_gt_0.05"]
               else f"lift {ladder_cls[k]['lift']:.2f} <= 1)")
            for k in mixed_rungs)
        clause = (
            f"a broad mixed zone — the response grades over "
            f"{math.log10(mixed_span):.2f} orders: {amb_bits}; "
            f"{len(lift_rungs)} LIFT / {len(coll_rungs)} COLLAPSE rungs "
            f"on either side; the flip is soft inside the bracket; the "
            f"two channels overlap; Law 7's parameter re-words to a band")
    elif (not mixed_rungs) and monotone and width_orders is not None \
            and width_orders > WIDTH_BAR_ORDERS:
        word = "UNRESOLVED-WIDE"
        clause = (
            f"clean monotone split ({len(lift_rungs)} LIFT / "
            f"{len(coll_rungs)} COLLAPSE / 0 MIXED) but the window "
            f"({L_pri:.6g}, {C_pri:.6g}] spans {width_orders:.2f} orders "
            f"> 1.0 — the interior ladder did not refine the bracket to "
            f"~1 order; the table verbatim")
    else:
        word = "PARTIAL-GRADING"
        clause = (
            "neither frozen bar fires — the table verbatim: "
            f"{len(lift_rungs)} LIFT / {len(coll_rungs)} COLLAPSE / "
            f"{len(mixed_rungs)} MIXED; monotone {monotone}; window "
            f"({L_pri}, {C_pri}] width {width_orders}; no wording change "
            "without a new registered cell")

    # the interior sign map (the mid-formation saturation rider)
    up_coll = [k for k in interior_keys
               if interior_cls[k]["response_class"] == "COLLAPSE"
               and interior_cls[k]["move_signed"] > 0]
    sign_map = {
        "form": "the SIGN MAP across formation (the registered risk's "
                "discriminating observation): signed moves on every rung; "
                "COLLAPSE-with-direction-UP = mid-formation saturation "
                "(the magnitude clause classed it; the map sees it)",
        "interior_signed_moves": {k: interior_cls[k]["move_signed"]
                                  for k in interior_keys},
        "up_classed_collapses": up_coll,
        "interior_neutral_moves": {k: interior_cls[k]["neutral_rider"]
                                   ["move_signed"]
                                   for k in interior_keys},
    }
    METRICS["sign_map"] = sign_map

    # ---- the prediction scoring -----------------------------------------
    p_out = {
        "P_x42a_statement": REGISTERED["P_x42a_verbatim"],
        "P_x42a_outcome": ("HIT — SHARP-INTERIOR"
                           if word == "SHARP-INTERIOR"
                           else f"MISS — the verdict is {word}"),
        "executor_position": REGISTERED["executor_position"],
    }
    exec_bits = {
        "sharp_interior": sharp,
        "flip_low_window_top_le_0p25": bool(sharp and C_pri <= 0.25),
        "no_broad_saturation_max_one_up_collapse":
            bool(len(up_coll) <= 1),
        "all_interior_above_0p10_collapse": bool(all(
            interior_cls[k]["response_class"] == "COLLAPSE"
            for k in interior_keys
            if interior_cls[k]["prior_host"] > 0.10)),
    }
    n_bits = sum(exec_bits.values())
    p_out["exec_bits"] = exec_bits
    p_out["P_x42a_exec_outcome"] = (
        f"{n_bits}/{len(exec_bits)} registered bits — "
        + ("HIT" if n_bits == len(exec_bits)
           else ("HIT-ON-THE-CONJUNCTION (sharp)" if sharp
                 else "MISS (see bits)")))
    METRICS["verdict"] = {
        "word": word, "clause": clause,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "ladder_split": {"LIFT": lift_rungs, "COLLAPSE": coll_rungs,
                         "MIXED": mixed_rungs,
                         "R_star_window": (L_pri, C_pri)
                         if (L_pri and C_pri) else None,
                         "width_orders": width_orders,
                         "mixed_span_x": mixed_span,
                         "monotone": monotone},
        "interior_only_split": {
            "LIFT": [k for k in interior_keys
                     if interior_cls[k]["response_class"] == "LIFT"],
            "COLLAPSE": [k for k in interior_keys
                         if interior_cls[k]["response_class"] == "COLLAPSE"],
            "MIXED": [k for k in interior_keys
                      if interior_cls[k]["response_class"] == "MIXED"]},
        "reconstruction_label": label,
        "prediction": p_out,
    }
    write_partial(f"P6 ADJUDICATED: {word}")

    rw = METRICS["verdict"]["ladder_split"]["R_star_window"]
    r_win = (f"({rw[0]:.6g}, {rw[1]:.6g}] ({width_orders:.2f} orders)"
             if rw else "none (see table)")

    # ============ P7: the figure =========================================
    order_all = sorted(all_rung_keys,
                       key=lambda k: ladder_cls[k]["prior_host"])
    order_int = sorted(interior_keys,
                       key=lambda k: interior_cls[k]["prior_host"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # (a) BOTH CURRENCIES vs read level — the interior + the references
    ax = axes[0]
    if rw:
        ax.axvspan(rw[0], rw[1], color="#f39c12", alpha=0.18,
                   label=f"R* window {r_win}")
    for k in order_all:
        v = ladder_cls[k]
        x = max(v["prior_host"], 1e-6)
        interior = k in interior_keys
        living = v["living_read_rule"]["p_base_gt_0.05"]
        col = "#c0392b" if living else "#2c3e50"
        ax.plot([x], [max(v["move_currency_x"], 1e-3)], "o",
                ms=10 if interior else 7, color=col,
                mfc=(col if interior else "none"),
                mew=1.8, zorder=3 if interior else 2)
        ax.annotate(k, (x, max(v["move_currency_x"], 1e-3)),
                    textcoords="offset points", xytext=(6, 4), fontsize=6)
        ax.plot([x], [max(v["lift"], 1e-3)], "s", ms=7, mfc="none",
                color="#2980b9", mew=1.6)
    ax.axhline(1.0, color="#2980b9", ls=":", lw=1.2,
               label="ratio bar: lift = 1x (the LIFT clause)")
    ax.axhline(COLLAPSE_X, color="#c0392b", ls="--", lw=1.2,
               label="move bar: 5x band (the COLLAPSE clause)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bare read level p(Z) at host (log)")
    ax.set_ylabel("response (log): lift (squares) / move_x (dots)")
    ax.set_title("(a) THE INTERIOR LADDER — both currencies vs read "
                 "level\n(filled = interior rungs; open = x38's outer "
                 "references; navy squares = lift)")
    ax.legend(fontsize=7)

    # (b) the SIGN MAP: signed move vs read level
    ax = axes[1]
    for k in order_all:
        v = ladder_cls[k]
        x = max(v["prior_host"], 1e-6)
        interior = k in interior_keys
        col = "#27ae60" if v["move_signed"] > 0 else "#8e44ad"
        ax.plot([x], [v["move_signed"]], "o", ms=10 if interior else 7,
                color=col, mfc=(col if interior else "none"), mew=1.8)
        ax.annotate(k, (x, v["move_signed"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6)
        ax.plot([x], [v["neutral_rider"]["move_signed"]], "_", ms=12,
                color="#7f8c8d", mew=2.0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("bare read level p(Z) at host (log)")
    ax.set_ylabel("signed move p_comp - p_base")
    ax.set_title("(b) THE SIGN MAP ACROSS FORMATION\n"
                 "(green UP / purple DOWN; grey dashes = neutral site; "
                 "UP+COLLAPSE = mid-formation saturation)")

    # (c) the formation trajectory (the construction's own record)
    ax = axes[2]
    steps_ax = [t["step"] for t in traj]
    pz_ax = [t["g0_pz"] for t in traj]
    ax.plot(steps_ax, pz_ax, "-", lw=1.2, color="#16a085",
            label="this cell's CPU re-formation (per-step g0 read)")
    cx = [int(k[1:]) for k, c in cert_cells.items()]
    ax.plot(cx, [c["committed"] for c in cert_cells.values()], "x", ms=9,
            mew=2.0, color="#c0392b",
            label="e335's committed trajectory cells")
    for T in INTERIOR_GRID:
        ax.axhline(T, color="#95a5a6", ls=":", lw=0.6)
    for k in interior_keys:
        stp = substrate_meta[k]["formation_step"]
        ax.plot([stp], [substrate_meta[k]["battery_read_inloop"]],
                "o", ms=9, mfc="#f39c12", mec="k", mew=0.8, zorder=5)
    ax.axhline(LIVING_BAR, color="#2c3e50", ls="--", lw=1.0,
               label="the living-read rule (0.05)")
    ax.set_yscale("log")
    ax.set_xlabel("install formation step (the rig, gen 24314)")
    ax.set_ylabel("g0 battery read p(Z)")
    ax.set_title(f"(c) THE FORMATION TRAJECTORY — rung capture\n"
                 f"label: {label.split(' (')[0]}")
    ax.legend(fontsize=7)

    fig.suptitle(f"X42 — THE R* INTERIOR: {word} (R* window {r_win})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RD / "x42_rstar_interior.png", dpi=140)
    plt.close(fig)
    log(f"P7: figure written -> {RD / 'x42_rstar_interior.png'}")

    # ============ P8: the REPORT =========================================
    rep = []
    rep.append(f"# X42 — THE R* INTERIOR ({now_utc()})\n")
    rep.append(f"**VERDICT: {word}** — {clause}\n")
    rep.append("## The question (verbatim)\n")
    rep.append(f"> {REGISTERED['question_verbatim']}\n")
    rep.append("## P-x42a (verbatim) + scoring\n")
    rep.append(f"> {REGISTERED['P_x42a_verbatim']}\n")
    rep.append(f"- P-x42a outcome: **{p_out['P_x42a_outcome']}**")
    rep.append(f"- Executor position: {p_out['P_x42a_exec_outcome']} "
               f"({json.dumps(exec_bits)})\n")
    rep.append("## THE INTERIOR LADDER (host contexts; both currencies "
               "per rung; construction disclosed)\n")
    rep.append("| rung | construction (formation step) | net0 class | "
               "prior | comp read | lift | signed move | move/band | "
               "band_move | class |")
    rep.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in order_int:
        v = interior_cls[k]
        stp = substrate_meta[k]["formation_step"]
        rep.append(
            f"| {k} | e043-Dmix gen-24314 re-formed, s{stp} "
            f"({label.split(' (')[0]}) "
            f"| {substrate_meta[k]['net0_class']} "
            f"| {v['prior_host']:.4e} | {v['comp_host']:.4e} "
            f"| {v['lift']:.2f}x | {v['move_signed']:+.4f} "
            f"| {v['move_currency_x']:.2f}x | {v['band_move']:.4f} "
            f"| {v['response_class']} |")
    rep.append("")
    rep.append("## THE OUTER REFERENCES (x38's rungs; re-read bit-exact)\n")
    rep.append("| rung | net0 class | prior | comp read | lift | signed "
               "move | move/band | class |")
    rep.append("|---|---|---|---|---|---|---|---|")
    for k, *_ in LADDER:
        v = ladder_cls[k]
        rep.append(
            f"| {k} | {substrate_meta[k]['net0_class']} "
            f"| {v['prior_host']:.4e} | {v['comp_host']:.4e} "
            f"| {v['lift']:.2f}x | {v['move_signed']:+.4f} "
            f"| {v['move_currency_x']:.2f}x | {v['response_class']} |")
    rep.append("")
    if rw:
        rep.append(f"**R\\* window = ({L_pri:.6g}, {C_pri:.6g}] — "
                   f"{width_orders:.3f} orders wide** (bar: <= 1.0; "
                   "x38's outer bracket was "
                   f"{math.log10(X38_RSTAR_BRACKET[1] / X38_RSTAR_BRACKET[0]):.2f}"
                   " orders).\n")
    rep.append("Neutral-site rider (site-generality, never adjudicated): "
               + "; ".join(
                   f"{k} {interior_cls[k]['neutral_rider']['move_signed']:+.4f}"
                   for k in order_int) + "\n")
    rep.append("## The construction + its certification\n")
    rep.append(f"- Interior rungs: the root's OWN install (e001 + "
               "e043-Dmix, gen 24314) re-run on CPU with per-step battery "
               "reads; rung = first crossing of each grid threshold "
               "{0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50} + s1; kept iff "
               f"read in ({bracket_bot:.4g}, {bracket_top:.4g}].")
    rep.append(f"- Reconstruction label: **{label}**")
    rep.append("- Certification cells (mine vs e335's committed "
               "trajectory): " + "; ".join(
                   f"s{k[1:]} {c['mine']:.4g} vs {c['committed']:.4g} "
                   f"({'OK' if c['within'] else 'OUT'}, tol {c['tol']})"
                   for k, c in cert_cells.items()))
    if overshoot:
        rep.append(f"- Overshoots (first-crossing states reading ABOVE "
                   f"the bracket top — disclosed, never rungs): "
                   + "; ".join(f"{r['key']} s{r['formation_step']} "
                               f"{r['host_site_read']:.4g}"
                               for r in overshoot))
    rep.append(f"- Cert-rider (never a rung): the reconstruction's "
               "s400 state's own comp cell "
               + (f"{METRICS['cert_rider'][cert_keys[-1]]['comp_host']:.4g} "
                  f"vs own_inst's committed 0.3597"
                  if cert_keys else "(no cert state)"))
    rep.append("")
    rep.append("## Gates\n")
    npass = sum(1 for g in METRICS["gates"].values() if g.get("pass"))
    rep.append(f"{npass}/{len(METRICS['gates'])} gates PASSED: "
               + ", ".join(k for k, g in METRICS["gates"].items()
                           if g.get("pass")) + "\n")
    rep.append("## Deviations (registered at birth)\n")
    for d in DEVIATIONS:
        rep.append(f"- {d}")
    rep.append("")
    (RD / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    METRICS["outputs"] = {
        "metrics": str(METRICS_PATH.relative_to(REPO)),
        "report": str((RD / "REPORT.md").relative_to(REPO)),
        "figure": str((RD / "x42_rstar_interior.png").relative_to(REPO)),
    }
    METRICS["status"] = f"COMPLETE: {word}"
    METRICS["git_head_final"] = git_head()
    METRICS["date_completed"] = now_utc()
    write_partial(f"COMPLETE: {word}")
    log(f"X42 COMPLETE — {word}")


LADDER_BY_KEY = {l[0]: l for l in LADDER}


if __name__ == "__main__":
    main()
