"""E193B — THE CRITIC'S REPLICATE, CELL 2: FRESH ROOT AT ORGANISM-1'S EXACT
ARCHITECTURE, TWO FACTS INSTALLED (the fact axis varied for the first time).

RECOVERY NOTE (2026-10-01, second dispatch): two predecessor executors were
killed by machine disruptions BEFORE any artifact; the frozen brief is the
only surviving spec. This file is the first artifact of the cell. Progressive
writes after every phase (the outage lesson — seven disruptions today); the
script is chunk-resumable across ALL training stages (base / two installs /
consolidation via checkpoints; the terrain pass is eval-only and re-reads the
committed root checkpoint, so it is restartable at zero training cost).

WHY (R60-critic's FORCED cell, scratch/r60_critic.md "THE FORCED EXPERIMENT";
T146/T153): every load-bearing terrain number lives on ONE organism-fact
pair (e131/ZEPHYRA). e193 replicated the terrain ORDER on a second lineage
(organism 2: 873k 4L; g 0.2 < sign 0.58 < random unresolved, ratios
1.108/1.117/1.119/2.629) and found THE PUMP ABSENT there (max rise -0.0788)
— but ARCHITECTURE CO-VARIED with lineage (T153's disclosed slip). THIS CELL
pins the axes: a FRESH root at organism 1's EXACT architecture (6L/6H/192d/
256-ctx, 2,739,072 params — absolute Ds are same-currency with organism 1)
with TWO FACTS installed (the lineage axis and the fact axis both
first-time-varied). With e193 this cell completes the 2x2: organism 2 varied
lineage+architecture; e193b varies root-draw+fact at fixed architecture.

THE ORGANISM (built here, never trained before):
  1. FRESH base: standard recipe, corpus seed 1337 family (e053c-class per
     e098's operationalization: batch 32, lr 1e-3, cosine warmup 100, 2000
     steps, 180 s cap, chunk-resumable), 6L/6H/192d/256-ctx TinyGPT, fresh
     init seed 5301 (registry-clean).
  2. TWO FACTS installed, e043-Dmix convention (exposure VERBATIM: 16 name
     draws + 48 anchors (16 paired originals + 32 random corpus), 100 steps,
     token-weighted masked union CE, AdamW (0.9,0.95) wd 0.1, lr 1e-3, clip
     1.0; e082 GATE-0 bars install-60 p(Z) >= 0.20 AND R1i NLL <= 4.5, ONE
     200-step patience branch on miss):
       FACT 1 (the standard fact) = ZEPHYRA — install sites host_occ[:60]
         under SPLICE_RNG 24301 (the SAME install-60 battery construction as
         organisms 1 and 2; mix gate {FLORIZEL:19, ELIZABETH:41}), install
         gen seed 5311.
       FACT 2 (the new fact) = MIRABEL — e154's nonce convention (7-char,
         0 corpus occurrences verified, onset char M distinct from Z); fresh
         DISJOINT sites host_occ[90:150] in the same SPLICE_RNG order (mix
         {FLORIZEL:16, ELIZABETH:44} — reported, not gated; the 19/41 gate
         is ZEPHYRA's), install gen seed 5312. DISCLOSED: the corpus yields
         exactly 150 filtered host sites, so MIRABEL has NO held-30 pool
         (ZEPHYRA keeps host_occ[60:90]); held30 co-report is ZEPHYRA-only.
     Install order: ZEPHYRA -> MIRABEL (sequential; ZEPHYRA's dial is read
     before/after the MIRABEL install — the interference co-report e154
     priced on a consolidated root; here it rides free).
  3. CONSOLIDATE (e113 recipe via e157 stage A): jittered pool offsets
     {-8,-4,0,+4,+8} on BOTH facts' install sites (300 windows each, union
     600), anchor bank = each fact's first 16 install positions' ORIGINAL
     host windows (32), batch 32 = 16 pool draws + 16 anchors (8 paired + 8
     random), e043 token-level union CE (name-masked on pool windows; each
     window masked on ITS OWN fact's 7 name-char targets), AdamW (0.9,0.95)
     wd 0.1 constant lr 1e-3 clip 1.0, 300 steps, gen seed 10901 (e113 arm
     (a) VERBATIM). TWO-FACT ADAPTATION (documented): the pool and anchor
     bank are the per-fact unions; nothing else changes.
  THE ROOT = the consolidated net (runs/checkpoints/e193b_root.pt).

RULERS (frozen BEFORE compute; e193's ruler lesson applied): each fact's
PRIMARY ruler = its OWN install-60 battery at ctx offset g-12 (e192's
organism-1 convention — the architecture-pinned comparison demands the same
geometry). ROOT-STRENGTH GATE per fact BEFORE the terrain pass (Rule 12): a
fact whose g-12 root read is under the 0.27 kill bar at D=0 is e193's
G_CONS lesson — REPORTED, not adjudicated away; its primary MECHANICALLY
falls back to the max root read among the four per-grid geometries
{g-12, g-4, g0, g+12} (e193's own precedent, generalized and frozen here;
the g-12 tables ALWAYS co-report; every geometry's kills reported in every
table — no ruler shopping is possible after the fact). The seven-geometry
root dial (e157's) is measured per fact for provenance.

THE TERRAIN PASS (e193's machinery VERBATIM, per fact): the FIVE RAYS —
g-ray (THIS root's t=0 post-clip wash-stream batch gradient, seed 10902
stream, e185 convention; measured, NEVER ported — the neutral stream is
name-free so the direction is fact-independent BY CONSTRUCTION and both
facts' terrains are read along the SAME ray set; this is the frozen reading
of "each fact's own g-ray convention measured, never ported"), static
sign(g_0) ray, 3 Gaussian unit rays (seeds 11901-3, g3K per-tensor draw).
D GRID to 4.0 (e192's 15 VERBATIM + e193's 12 densifiers = 27 points).
Dual currency (D + rms = D/sqrt(2,739,072); same denominator as organism
1 — architecture-pinned, absolute Ds comparable). Both facts' rulers at
EVERY grid point. THE PUMP READ per fact (small-D band {0.05..0.50}, ridge
margin 0.01, family-1 committed rises +0.0449 g / +0.0373 sign cited).
THE DENSITY LADDER on the PRIMARY fact (ZEPHYRA, frozen): topk {1%,10%,50%}
+ raw + sign at THIS root's MEASURED step L2 (fresh AdamW step-1 L2 at t=0;
never ported), matched per-step L2 asserted every step (fp64, tol 1e-5),
every-step kill reads + densified brackets (opt1c's convention), shared
bit-identical input stream across arms (each arm restarts the seed-10902
generator) — REPORT-ONLY (no ladder bar is registered in this dispatch).
THE RIDER per fact: 300 pinned steps of L2 STEP_L2/300 down the FROZEN
g-ray, reads every 30 steps on both facts' primaries (report-only).
THE IN-SPAN 3-DRAW RANGE (the missing number, R60-critic Attack 1): THIS
root's own wash-step history (20 CPU AdamW steps on the neutral stream from
the root, displacements saved) -> Gram SVD span -> THREE in-span random
unit draws (seeds 11911-3) each walked down the full 27-point grid; the
per-fact kill-D RANGE (min..max) is the deliverable (e131's suppressed 3x
seed spread is the number the record was missing).
THE MAGNITUDE-SHUFFLE BREAKER (R60-critic Attack 2 rank-1; report-only):
a_topk10 with the top-10% SUPPORT and coordinate SIGNS kept and the |g|
VALUES permuted within the support (per-step permutation, stream from seed
11921); kills at ~the topk-10 kill -> SUPPORT not PAIRING; spares ->
magnitude-pairing convicted interventionally.

REGISTERED BARS (the dispatch's registration VERBATIM, frozen before
compute; no bar shopping — adjudicate against exactly this):
  - TERRAIN-REPLICATES: "for BOTH facts the ray order g < sign < random
    holds (randoms >= 2x the g-kill) and kill-Ds within [0.5x, 2x] of
    organism-1's — the terrain at n=3 organisms, fact-varied,
    architecture-pinned."
  - ORDER-SCRAMBLES: "any fact's order changes or kill-Ds move > 2x —
    'terrain' rescopes to biography."
  - PUMP-PER-FACT (co-registered, no bar): "each fact's ridge present or
    absent — with e193, this decides the ridge-vs-cliff thesis."
The QUEUE row's earlier letter (randoms >= 4x, shuffled-sign inert,
pump-then-cliff) is CO-REPORTED (readings, never adjudicated); this
dispatch's bars supersede it. The shuffled-SIGN ray was e_chart's object
(organism 1, inert, committed); this cell's breaker is the shuffled-
MAGNITUDE arm — a different, never-intervened object.

REGISTERED PREDICTION (frozen): the ORDER + kill-D windows hold for both
facts (architecture pinned; the terrain is draw-physics); the pump is the
open axis — T153's ridge-vs-cliff thesis predicts the ridge may be present
for one fact and absent for the other (fact-biography) or present for both
(draw-physics); absence for both joins e193's lineage-biography reading.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * kill = first grid D (ascending) with the fact's PRIMARY ruler mean p
    <= 0.27 (e185's DISSOLVE bar); no kill on the grid = unresolved-high
    (> 4.0). Ladder kills: every-step read + linear-in-D interpolation
    DENSIFIED along the killing step at f in {0.2,0.4,0.6,0.8} (opt1c).
  * order (per fact) = g_kill < sign_kill < rand_eff, rand_eff = min over
    the 3 Gaussian rays' first-dead Ds (all-unresolved = +inf). A sign kill
    unresolved-high leaves the order UNESTABLISHED for that fact (e192's
    own convention, carried): TERRAIN-REPLICATES cannot fire for it; a
    GRADED note with that verbatim is recorded (unmeasured is not
    "changed" — ORDER-SCRAMBLES does not fire on it).
  * random_wide (per fact) = every Gaussian ray's first-dead D >= 2 x
    g_kill or unresolved-high.
  * kill windows (organism-1 committed anchors, hard-bound at load from
    runs/e192/metrics.json): g in [0.46, 1.84]; sign in [1.25, 5.0] —
    measurable only to the 4.0 grid cap, so a RESOLVED sign kill <= 4.0 is
    inside iff >= 1.25; randoms carry no window (organism 1's are
    unresolved-high > 4.0; e_chart's band > 12) — the width clause is
    their test.
  * per-fact: terrain_fact = order_holds AND random_wide AND g_kill in
    window AND sign_kill resolved in window. scrambles_fact = (order
    MEASURED and != g<sign<random) OR g_kill resolved outside [0.46,1.84]
    OR sign_kill resolved outside [1.25,5.0] (on this grid: sign < 1.25).
  * composite headline (the bars' own letters): ORDER-SCRAMBLES if ANY
    fact scrambles ("any fact's ... rescopes"); else TERRAIN-REPLICATES if
    BOTH facts replicate; else GRADED (partial pattern, tables verbatim).
    Per-fact clauses are ALWAYS reported alongside the headline.
  * pump ridge (per fact per ray) = max rise over the small-D band >
    0.01 (family-1's frozen margin verbatim); strict single-point rises
    co-reported so nothing hides behind the margin.
  * in-span kill-D = first-dead grid D on each fact's primary (grid
    resolution disclosed); RANGE = min..max over the three draws.
  * the breaker "kills at ~the topk-10 kill" = its densified D_kill within
    +-25% of this root's own a_topk10 D_kill (the same materiality band
    the front arc uses); "spares" = no kill by D_TARGET 5.0.

PRE-DISPATCH CHECKS (Rule 12, asserted BEFORE the terrain pass): the fresh
root's full provenance (seed lineage, checkpoint chain, flat md5 of the
root, val CE plausibility); G_NAMEFREE (corpus ZEPH count 0 AND MIRABEL
count 0); G_SPLICE (ZEPHYRA mix 19/41; MIRABEL mix 16/44 disclosed);
G_BATTERY (both facts' batteries at all seven geometries, shapes 60 x
(130 +- j)); G_ANCHOR (the seed-170 neutral bank BIT-MATCHES e185's stored
16 starts); G_T0 (the step-1 batch md5 == e185's stored hash — the stream
construction is net-independent, so it MUST bit-match here too; a fresh
AdamW recipe step (lr 1e-3, betas 0.9/0.95, wd 0.1, clip 1.0) from the
root whose L2 IS this cell's matched step L2, checkpointed for later
cells); G_STREAM (seed-10902 stream md5-verified steps 1..4 without
training); G_GATE0 per fact (the install gates above, BEFORE compute);
G_ROOTSTRENGTH per fact (both facts' g-12 root reads vs the 0.27 bar,
reported before the terrain pass). WHAT EACH ARM GUARANTEES: NOTHING — the
order could reorder per fact; kill-Ds could leave the windows; the pump
could appear on random rays; the in-span spread could swallow the g-ray;
the breaker could spare or kill; that openness is the point.

COMPUTE ENVELOPE (dispatch): trainings GPU-gated (gpu_ok() double-poll;
PAUSE-AND-WAIT under the neighbor's jobs — g1bS2-r2/g2g2-r2 own their
cells; NEVER migrate mid-run); every training chunk <= 180 s;
cooldown(120) between trainings; the base/installs/consolidation are
checkpoint-resumable so a kill mid-chunk restarts, never re-pays. The
entire terrain pass is CPU-ONLY eval (torch threads 8, no GPU claim;
CUDA_VISIBLE_DEVICES left up to the training gate). PROGRESSIVE PARTIAL
metrics.json writes after the gates, after every ray, after the rider,
after every ladder arm, after every in-span draw. n=1 fresh-root draw;
each fact n=1; no NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Outputs: runs/e193b/{metrics.json, journal.jsonl, e193b_fig5_two_fact.png,
e193b_ladder.png, e193b_inspan_range.png, e193b_pump_per_fact.png};
checkpoints runs/checkpoints/e193b_*.pt (base, two installs, root, wash s1,
g-ray direction, wash-history span).

Run:   cd lab && python e193b_two_fact.py      (E193B_SMOKE=1 shakedown)
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

import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e143/e151/e158 CPU convention

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,         # noqa: E402
                    cooldown, estimate_loss, gpu_ok, gpu_status,
                    run_dir, save_json, set_seed, train_model)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, exposure, eval_seq, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E193B_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ facts
FACT1 = "ZEPHYRA"                 # the standard fact (organisms 1/2's name)
FACT2 = "MIRABEL"                 # e154's nonce convention (verified 0 occ)
PRE, POST_CAP = 130, 119          # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
L_NAME = 7                        # both names are 7 chars (symmetric masks)

# ------------------------------------------------------------------ seeds
# fresh, registry-clean (checked vs the e193 registry note + repo grep):
# lineage 10901/10902/170/1337/24301/26502/4305-8/42/43/202/7/110/960250/
# 15301/11401-3/11411-13/11501-3/11601-3/11611-13/11621-3/11701/11801-3
SEED_BASE = 5301                  # fresh base init
SEED_INSTALL = {FACT1: 5311, FACT2: 5312}
SEED_CONS = 10901                 # e113 arm (a) VERBATIM (the recipe's own)
FREEZE_SEED = 10902               # the wash stream (e176n/e185/e192/e193)
RAND_SEEDS = (11901, 11902, 11903)   # the 3 Gaussian rays (fresh)
INSPAN_SEEDS = (11911, 11912, 11913)  # the 3 in-span draws (fresh)
MAGSHUF_SEED = 11921              # the breaker's permutation stream (fresh)

# ------------------------------------------------------------- recipe consts
BASE_STEPS = 40 if SMOKE else 2000      # e098's e053c-class operationalization
BASE_BATCH = 4 if SMOKE else 32
BASE_CAP_S = 30.0 if SMOKE else 180.0
BASE_EVAL_EVERY = 10 if SMOKE else 250
VCE_PLAUSIBLE = 1.70                    # e053c/e098 plausibility gate
INSTALL_STEPS = 8 if SMOKE else 100
INSTALL_EVAL_AT = {4, 8} if SMOKE else {25, 50, 100}
PATIENCE_STEPS = 16 if SMOKE else 200
GATE_PZ_FLOOR, GATE_R1I_MAX = 0.20, 4.5  # e082 GATE-0 bars
CONS_STEPS = 16 if SMOKE else 300        # e113 jitter-replay
JITTERS = (-8, -4, 0, 4, 8)
WASH_HIST_STEPS = 4 if SMOKE else 20     # the in-span history segments
LR_ADAMW = 1e-3                          # the t=0/wash recipe

# --------------------------------------------------------------- terrain consts
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE), verbatim
PUMP_MARGIN = 0.01                # family-1's frozen ridge margin
SMALL_D_PUMP_BAND = (0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5)
BASE_GRID = (0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.92, 1.0, 1.17,
             1.5, 2.0, 2.5, 3.0, 4.0)          # e192's 15, verbatim
EXTRA_GRID = (0.15, 0.25, 0.3, 0.4, 0.58, 0.74, 0.87, 1.08, 1.33,
              1.75, 2.25, 3.5)                  # e193's 12 densifiers
D_GRID = tuple(sorted(set(BASE_GRID) | set(EXTRA_GRID)))
RIDER_N, RIDER_EVERY = 300, 30
TOPK_FRACS = {"a_topk1": 0.01, "a_topk10": 0.10, "a_topk50": 0.50}
L2_ASSERT_TOL = 1e-5
D_TARGET = 5.0                    # opt2's horizon, verbatim absolute
STEP_CAP = 8
CKPT_STEPS = (1, 2, 4, 8)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)
INTER_ARM_S = 5.0
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor + 16 random
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)     # the e157 dial's seven geometries
GRID_GEOS = (-12, -4, 0, 12)               # the per-grid-point geometries
RULER_J = -12                                # PRIMARY (e192-verbatim)
if SMOKE:
    D_GRID = (0.1, 0.5, 1.5)
    RAND_SEEDS = (11901,)
    INSPAN_SEEDS = (11911,)
    RIDER_EVERY = 150
    STEP_CAP, CKPT_STEPS, DENSIFY_F = 2, (1,), ()

# ------------------------------------------------------------ organism-1 anchors
F1_GKILL = 0.92                    # e192 committed static g-ray kill
F1_SIGN_KILL = 2.5                 # e192 committed static sign-ray kill
F1_PUMP = {"g": 0.0449, "sign": 0.0373, "rand_max": 0.0002}
F1_STEP_L2 = 1.6543                # e185's wash-1x (context only, never used)
F1_LADDER = {"topk10": 0.90664498444579, "topk50": 0.9202602091418797,
             "raw": 0.9203406595225093, "sign_path": 1.749640490742179,
             "a0_adam": 2.4892616271972656}
F1_RATIOS = {k: v / F1_GKILL for k, v in F1_LADDER.items()}
G_WINDOW = (0.5 * F1_GKILL, 2.0 * F1_GKILL)      # [0.46, 1.84]
SIGN_WINDOW = (0.5 * F1_SIGN_KILL, 2.0 * F1_SIGN_KILL)  # [1.25, 5.0]

# e193 organism-2 committed anchors (co-cited; loaded + asserted below)
E193_GKILL, E193_SIGNKILL = 0.2, 0.58
E193_PUMP_RISE = -0.07882732152938843

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                     # per-step input-batch md5 (seed-10902 stream)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
}

CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E192_METRICS = E43.REPO / "runs" / "e192" / "metrics.json"
OPT2_METRICS = E43.REPO / "runs" / "opt2" / "metrics.json"
E193_METRICS = E43.REPO / "runs" / "e193" / "metrics.json"

REGISTERED_BARS = {
    "TERRAIN_REPLICATES": "TERRAIN-REPLICATES: \"for BOTH facts the ray "
        "order g < sign < random holds (randoms >= 2x the g-kill) and "
        "kill-Ds within [0.5x, 2x] of organism-1's — the terrain at n=3 "
        "organisms, fact-varied, architecture-pinned.\"",
    "ORDER_SCRAMBLES": "ORDER-SCRAMBLES: \"any fact's order changes or "
        "kill-Ds move > 2x — 'terrain' rescopes to biography.\"",
    "PUMP_PER_FACT": "PUMP-PER-FACT (co-registered, no bar): \"each fact's "
        "ridge present or absent — with e193, this decides the "
        "ridge-vs-cliff thesis.\"",
    "registered_prediction": "the ORDER + kill-D windows hold for both "
        "facts (architecture pinned; the terrain is draw-physics); the "
        "pump is the open axis — T153's ridge-vs-cliff thesis predicts "
        "the ridge may split by fact; absence for both joins e193's "
        "lineage-biography reading.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar "
        "shopping. The QUEUE row's earlier letter (randoms >= 4x, "
        "shuffled-sign inert) is co-reported, never adjudicated.",
}

deviations: list[str] = [
    "RECOVERY (2nd dispatch): predecessors killed pre-artifact; this file "
    "is the cell's first artifact. All stages chunk-resumable; progressive "
    "metrics.json writes after every phase.",
    "MIRABEL's install sites are host_occ[90:150] in the SAME SPLICE_RNG "
    "24301 shuffle order (fresh, disjoint from ZEPHYRA's install60+held30); "
    "mix {FLORIZEL:16, ELIZABETH:44} reported ungated — the 19/41 gate is "
    "ZEPHYRA's battery-identity gate, not a law. The corpus yields exactly "
    "150 filtered sites: MIRABEL has NO held-30 pool (disclosed; held30 "
    "co-report is ZEPHYRA-only).",
    "THE RAY-SET READING (frozen): the five rays are root-level directions "
    "— the t=0 wash-stream gradient is fact-independent by construction "
    "(the neutral stream carries no name), so both facts' terrains are "
    "read along the SAME measured-at-this-root ray set; PER FACT applies "
    "to the rulers, kills, pump, rider and in-span reads. 'Never ported' "
    "is honored: nothing is imported from organism 1/2's directions, step "
    "L2s, or kill-Ds.",
    "PRIMARY-RULER FALLBACK (mechanical, frozen before compute): a fact "
    "whose g-12 root read < 0.27 (under its kill bar at D=0 — e193's "
    "G_CONS lesson) is REPORTED, not adjudicated away; its primary falls "
    "back to the max root read among the four per-grid geometries "
    "{g-12, g-4, g0, g+12}. g-12 tables always co-report.",
    "The e113 consolidation pool is the per-fact UNION (600 windows; each "
    "window masked on ITS OWN fact's name targets); the anchor bank is "
    "each fact's first 16 install-position originals (32). Everything "
    "else is e157-stage-A VERBATIM (300 steps, seed 10901, batch 32 = "
    "16 pool + 8 paired + 8 random, constant lr 1e-3, clip 1.0).",
    "The ladder and the breaker are REPORT-ONLY (no ladder bar in this "
    "dispatch); the QUEUE row's FRONT bars belong to e193's cell.",
    "The in-span span is THIS root's own realized-wash-step history (20 "
    "CPU AdamW steps' displacements on the seed-10902 neutral stream) — "
    "e_chart's convention (Adam's sign-flattened steps, its own caveat "
    "carried), never a ported span.",
    "Trainings GPU-gated with pause-and-wait (never migrate mid-run; the "
    "neighbor cells g1bS2-r2/g2g2-r2 own their slots); the terrain pass "
    "is CPU-only eval. Device path per training recorded.",
    "Smoke mode trims: D grid {0.1,0.5,1.5}, one Gaussian + one in-span "
    "draw, base 40 steps / installs 8 / consolidation 16 / wash-history 4 "
    "steps, rider reads every 150, ladder cap 2 steps / no densification; "
    "nothing adjudicated (verdict stamped SMOKE).",
]

# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e193_organism_replicate.py (whose own provenance is
# lab/e192_all_ray_terrain.py / lab/opt2_density.py via lab/e191_pump_cliff.py
# / lab/e185_noise_wash.py — the e176n lineage; and lab/e098_seed_ladder.py /
# lab/e157_lineage_replication.py for the install/consolidation stages).
# Copied rather than imported to own the device policy.


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(name onset) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def flat_params(net: TinyGPT) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def participation_ratio(sv: torch.Tensor) -> float:
    s2 = (sv.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """e_chart's Gram-based right-singular basis (fp64)."""
    H64 = H.to(torch.float64)
    G = (H64 @ H64.T)
    evals, evecs = torch.linalg.eigh(G)     # ascending
    evals = torch.flip(evals, dims=(0,)).clamp(min=0)
    evecs = torch.flip(evecs, dims=(1,))
    sv = torch.sqrt(evals)
    pr = participation_ratio(sv) if float(sv[0]) > 0 else 0.0
    rank_eff = int((sv > sv[0] * 1e-7).sum())
    rows = []
    for i in range(rank_eff):
        v = H64.T @ evecs[:, i]
        rows.append((v / sv[i].clamp(min=1e-30)).to(torch.float32))
    Vp = torch.stack(rows) if rows else torch.empty(0, H.shape[1])
    return {"sv": sv, "Vp": Vp, "pr": pr, "rank_eff": rank_eff,
            "cond": float(sv[0] / sv[-1].clamp(min=1e-30))}


# ---- the ladder's update rules (opt2 VERBATIM + the breaker) --------------

def topk_update(g: torch.Tensor, frac: float, step_l2: float):
    k = int(round(frac * g.numel()))
    idx = torch.topk(g.abs(), k).indices
    sub = g[idx]
    nrm = float(torch.norm(sub.double()))
    assert nrm > 0, "top-k subgradient is zero — the arm is undefined"
    delta = torch.zeros_like(g)
    delta[idx] = -step_l2 * (sub / nrm)
    mask = torch.zeros(g.numel(), dtype=torch.bool)
    mask[idx] = True
    return delta, k, mask


def magshuffle_update(g: torch.Tensor, frac: float, step_l2: float,
                      gen: torch.Generator):
    """THE BREAKER (R60-critic Attack 2, rank-1): top-frac SUPPORT and
    coordinate SIGNS kept; the |g| VALUES permuted within the support."""
    k = int(round(frac * g.numel()))
    idx = torch.topk(g.abs(), k).indices
    mags = g[idx].abs().clone()
    signs = torch.sign(g[idx])
    perm = torch.randperm(k, generator=gen)
    mags_perm = mags[perm]
    vals = signs * mags_perm
    nrm = float(torch.norm(vals.double()))
    assert nrm > 0, "shuffled subgradient is zero — the arm is undefined"
    delta = torch.zeros_like(g)
    delta[idx] = -step_l2 * (vals / nrm)
    mask = torch.zeros(g.numel(), dtype=torch.bool)
    mask[idx] = True
    frac_kept = float((signs == torch.sign(g[idx])).float().mean())
    mag_pairing_broken = 1.0 - float((perm == torch.arange(k)).float().mean())
    return delta, k, mask, frac_kept, mag_pairing_broken


def sign_update(g: torch.Tensor, step_l2: float):
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


def raw_update(g: torch.Tensor, step_l2: float):
    nrm = float(torch.norm(g.double()))
    assert nrm > 0, "raw direction is zero — the arm is undefined"
    return -step_l2 * (g / nrm)


ARM_SPECS = [
    ("a_topk1", "TOPK-1% — opt2's licensed new rung (report-only here)",
     "topk", 0.01),
    ("a_topk10", "TOPK-10% — opt2's rung (report-only here)", "topk", 0.10),
    ("a_topk50", "TOPK-50% — the density midpoint (report-only)", "topk", 0.50),
    ("a_raw", "RAW-G — the magnitude-informative anchor (opt1c)", "raw", None),
    ("a_sign", "SIGN — the full sign(g) stretch arm (opt2)", "sign", None),
    ("a_magshuf", "MAGNITUDE-SHUFFLE — THE BREAKER: top-10% support + signs "
     "kept, |g| permuted within the support (report-only; R60-critic "
     "Attack 2 rank-1)", "magshuf", 0.10),
]
ARM_ORDER = [t for t, _, _, _ in ARM_SPECS]
ARM_DESC = {t: d for t, d, _, _ in ARM_SPECS}


def interp_d_kill(v0, v1, d0, d1):
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    pts = [(d0, v0)] + [(r["D"], r["gm"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


# ------------------------------------------------------------------ gpu gate
GPU_WAIT_MAX_S = 2400 if not SMOKE else 120     # pause-and-wait bound (40 min)
GPU_POLL_S = 30
COOLDOWN_S = 5 if SMOKE else 120                # e098's smoke convention


def gpu_ok_double(poll_gap_s: float = 2.0) -> bool:
    if not gpu_ok():
        return False
    time.sleep(poll_gap_s)
    return gpu_ok()


def acquire_gpu(tag: str) -> str:
    """PAUSE-AND-WAIT under the neighbor's jobs (never migrate mid-run);
    bounded wait with loud logging; falls back to CPU only BETWEEN chunks
    (documented deviation) if the GPU never clears within the bound."""
    t0 = time.time()
    while time.time() - t0 < GPU_WAIT_MAX_S:
        if gpu_ok_double():
            return "cuda"
        s = gpu_status()
        log(f"[gpu_guard] {tag}: HOLD (neighbor/thermal) {s} — waiting "
            f"{GPU_POLL_S}s ({time.time() - t0:.0f}/{GPU_WAIT_MAX_S}s)")
        time.sleep(GPU_POLL_S)
    deviations.append(
        f"{tag}: GPU never cleared within {GPU_WAIT_MAX_S}s — chunk ran "
        f"CPU-side from its checkpoint (documented; float arithmetic "
        f"differs by device; gates are absolute-value)")
    return "cpu"


def set_device(dev: str) -> None:
    common.DEVICE = dev
    E43.DEVICE = dev


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("e193b_smoke" if SMOKE else "e193b")
    jpath = rd / "journal.jsonl"
    log(f"E193B THE CRITIC'S REPLICATE, CELL 2 (smoke={SMOKE}) -> {rd}")
    log("compute: trainings GPU-gated pause-and-wait (chunks <=180 s, "
        f"cooldown 120); terrain CPU-only; {len(D_GRID)}-D grid x "
        f"{2 + len(RAND_SEEDS)} rays + {len(INSPAN_SEEDS)} in-span draws, "
        f"{len(ARM_ORDER)} ladder arms (report-only), stream seed "
        f"{FREEZE_SEED}")

    def journal(rec: dict) -> None:
        with open(jpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(E43.jsonable(rec), default=float) + "\n")

    # ---------------- committed parents (loaded, never rerun) ---------------
    for p in (E192_METRICS, OPT2_METRICS, E193_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    e192m = json.loads(E192_METRICS.read_text(encoding="utf-8"))
    opt2m = json.loads(OPT2_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    k192 = e192m["adjudication"]["kills"]
    assert abs(k192["g"] - F1_GKILL) < 1e-12
    assert abs(k192["sign"] - F1_SIGN_KILL) < 1e-12
    opt2_ladder = {r["rung"].split(" (")[0]: r["D"]
                   for r in opt2m["density_ladder"]["raw_rows"]}
    for key, ck in (("topk-10", "topk10"), ("topk-50", "topk50"),
                    ("raw", "raw"), ("sign", "sign_path"),
                    ("A0-Adam", "a0_adam")):
        assert abs(opt2_ladder[key] - F1_LADDER[ck]) < 1e-12
    e193b_ = e193m["adjudication"]["bars"]["TERRAIN_REPLICATES"]
    assert abs(e193b_["g_kill"] - E193_GKILL) < 1e-12
    assert abs(e193b_["sign_kill"] - E193_SIGNKILL) < 1e-12
    assert abs(e193m["pump_read"]["rays"]["R1_G"]["max_rise"]
               - E193_PUMP_RISE) < 1e-9
    log("parents hard-bound: e192 (organism-1 terrain g 0.92 < sign 2.5 < "
        "random >4.0), opt2 (organism-1 ladder), e193 (organism-2: g 0.20, "
        "sign 0.58, pump max rise -0.0788 ridge ABSENT) — COMMITTED, "
        "never rerun")

    # ---------------- protocol rebuild ---------------------------------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    zids = {FACT1: stoi["Z"], FACT2: stoi["M"]}
    name_ids = {f: torch.tensor([stoi[c] for c in f], dtype=torch.long)
                for f in (FACT1, FACT2)}

    G_NAMEFREE = {
        "corpus_counts": {FACT1: train_text.count(FACT1[:4]),
                          FACT2: len(E43.find_occ(train_text, FACT2))},
        "pass": bool(train_text.count("ZEPH") == 0
                     and len(E43.find_occ(train_text, FACT2)) == 0)}
    assert G_NAMEFREE["pass"], f"nonce leaked: {G_NAMEFREE}"
    log(f"G_NAMEFREE: ZEPH x0, {FACT2} x0 in train: PASS")

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = _rng = random.Random(E43.SPLICE_RNG)
    _rng.shuffle(host_occ)
    install_occ = {FACT1: host_occ[:60], FACT2: host_occ[90:150]}
    held_occ = {FACT1: host_occ[60:90], FACT2: []}
    mix1 = {"FLORIZEL": sum(1 for _, h in install_occ[FACT1] if h == "FLORIZEL"),
            "ELIZABETH": sum(1 for _, h in install_occ[FACT1] if h == "ELIZABETH")}
    mix2 = {"FLORIZEL": sum(1 for _, h in install_occ[FACT2] if h == "FLORIZEL"),
            "ELIZABETH": sum(1 for _, h in install_occ[FACT2] if h == "ELIZABETH")}
    G_SPLICE = {
        "install_mix": {FACT1: mix1, FACT2: mix2},
        "held_n": {f: len(v) for f, v in held_occ.items()},
        "sites_disjoint": True, "n_filtered_total": len(host_occ),
        "pass": bool(mix1 == {"FLORIZEL": 19, "ELIZABETH": 41}),
        "note": "ZEPHYRA's mix gate = battery identity with organisms 1/2; "
                "MIRABEL's mix reported ungated (fresh disjoint sites in "
                "the same SPLICE_RNG order); MIRABEL held pool EMPTY "
                "(corpus yields exactly 150 filtered sites).",
    }
    assert G_SPLICE["pass"], f"splice drift {G_SPLICE}"
    log(f"G_SPLICE: {FACT1} install60 {mix1} (gate) | {FACT2} install60 "
        f"{mix2} (disjoint, ungated) | held {G_SPLICE['held_n']}")

    # the batteries: install-60 contexts at the seven read geometries
    bat_ids = {f: {} for f in (FACT1, FACT2)}
    for f in (FACT1, FACT2):
        for j in READ_GEOS:
            cs = [train_text[p - PRE - j: p] for p, _ in install_occ[f]]
            bat_ids[f][j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {f: {str(j): list(bat_ids[f][j].shape) for j in READ_GEOS}
                   for f in (FACT1, FACT2)},
        "pass": bool(all(list(bat_ids[f][j].shape) == [60, PRE + j]
                         for f in (FACT1, FACT2) for j in READ_GEOS))}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    log(f"G_BATTERY: both facts x 7 geometries, shapes 60 x (130 +- j): PASS")
    held_ids = {f: (torch.stack([corpus.encode(train_text[p - PRE:p])
                                 for p, _ in held_occ[f]])
                    if held_occ[f] else None) for f in (FACT1, FACT2)}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, 26502)
    r_eval_xy = (r_eval_x, r_eval_y)

    # the neutral bank + stream (e170/e185 VERBATIM via e193)
    arng = random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    reject_names = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in reject_names):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": "16 plain corpus windows, RNG seed 170, rejection "
                        "on FLORIZEL/ELIZABETH/ZEPH/MIRABEL — e170 VERBATIM",
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], f"neutral bank drifted: {n_starts}"
    log("G_ANCHOR: neutral bank bit-matches e185's stored 16 starts: PASS")

    # =====================================================================
    # STAGE 1 — the FRESH base (e053c-class recipe, organism-1 architecture)
    # =====================================================================
    cfg = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
    ck_base = CKPT_DIR / ("e193b_smoke_base.pt" if SMOKE else "e193b_base.pt")
    dev = acquire_gpu("base-train")
    set_device(dev)
    set_seed(SEED_BASE)                       # init on CPU (e040 rule)
    base_net = TinyGPT(cfg)
    n_params = base_net.num_params()
    assert n_params == 2_739_072, f"params {n_params} != 2739072"
    RMS_DEN = float(math.sqrt(float(n_params)))
    base_net = base_net.to(dev)
    t1 = time.time()
    hist = train_model(base_net, corpus, steps=BASE_STEPS, lr=1e-3,
                       batch_size=BASE_BATCH, max_seconds=BASE_CAP_S,
                       eval_every=BASE_EVAL_EVERY, ckpt=ck_base)
    base_wall = time.time() - t1
    base_dev = dev
    set_device("cpu")                       # ALL readouts CPU-side
    base_net_cpu = base_net.cpu().eval()
    val_ce = estimate_loss(base_net_cpu, corpus, "val", n_batches=12)
    base_sd = {k: v.detach().clone() for k, v in base_net_cpu.state_dict().items()}
    del base_net
    set_device("cpu")
    if base_dev == "cuda":
        cooldown(COOLDOWN_S)
    log(f"STAGE 1 base: {hist[-1]['step'] if hist else 0}/{BASE_STEPS} steps "
        f"on {base_dev} in {base_wall:.0f}s | val CE {val_ce:.4f} "
        f"({'plausible' if val_ce < VCE_PLAUSIBLE else 'OFF'})")
    stage1 = {"seed": SEED_BASE, "steps_target": BASE_STEPS,
              "steps_done": int(hist[-1]["step"]) if hist else 0,
              "val_ce": val_ce, "plausible": bool(val_ce < VCE_PLAUSIBLE),
              "wall_s": round(base_wall, 1), "device": base_dev,
              "ckpt": ck_base.name,
              "history": [{"step": h["step"], "val_loss": h["val_loss"]}
                          for h in hist]}
    journal({"stage": "base", **stage1})

    # =====================================================================
    # STAGE 2/3 — the TWO INSTALLS (e043 exposure VERBATIM, e082 GATE-0)
    # =====================================================================
    def build_fact_windows(fact: str):
        occ = install_occ[fact]
        nid = name_ids[fact]
        wins = torch.stack([
            torch.cat([train_ids[p - PRE: p], nid,
                       train_ids[p + len(h): p + len(h) + POST_CAP]])
            for p, h in occ])
        mask = torch.zeros(len(occ), BLOCK - 1, dtype=torch.bool)
        mask[:, PRE - 1: PRE - 1 + L_NAME] = True
        anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                                   for p, _ in occ])
        bat_seq = wins[:, :PRE + L_NAME]
        return wins, mask, anchor_full, bat_seq

    def gate_readout_cpu(sd: dict, fact: str) -> dict:
        m = TinyGPT(cfg)
        m.load_state_dict(sd)
        m.eval()
        pz = battery_cell(m, bat_ids[fact][0], zids[fact])
        _saved_dev = E43.DEVICE           # eval_seq must run CPU-side here
        E43.DEVICE = "cpu"
        try:
            r1 = E43.eval_seq(m, fact_bats[fact]["bat_seq"], L_NAME,
                              PRE - 1)
        finally:
            E43.DEVICE = _saved_dev
        out = {"install60_pz": pz["mean_pz"],
               "install60_argmax": pz["frac_argmax_z"],
               "r1i_nll": r1["nll"], "r1i_acc": r1["acc"],
               "onset_acc": r1["per_pos_acc"][0]}
        if held_ids[fact] is not None:
            out["held30_pz"] = battery_cell(m, held_ids[fact],
                                            zids[fact])["mean_pz"]
        return out

    fact_bats = {}
    for fact in (FACT1, FACT2):
        w, msk, anc, bseq = build_fact_windows(fact)
        fact_bats[fact] = {"win": w, "mask": msk, "anchor": anc,
                           "bat_seq": bseq}

    install_recs: dict[str, dict] = {}
    prev_sd = base_sd
    for fact in (FACT1, FACT2):
        ck = CKPT_DIR / (f"e193b_smoke_install_{fact}.pt" if SMOKE
                         else f"e193b_install_{fact}.pt")
        if not ck.exists():
            dev_i = acquire_gpu(f"install-{fact}")
        else:
            dev_i = acquire_gpu(f"install-{fact}-resume")
        set_device(dev_i)
        net_i = TinyGPT(cfg)
        net_i.load_state_dict(prev_sd)
        net_i = net_i.to(dev_i)
        fb = fact_bats[fact]

        def _on_eval(m, step, _f=fact):
            sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            return {"step": step, **gate_readout_cpu(sd, _f)}

        t2 = time.time()
        traj = E43.exposure(
            net_i, fb["win"].to(dev_i), fb["mask"].to(dev_i),
            fb["anchor"].to(dev_i),
            steps=INSTALL_STEPS, total=INSTALL_STEPS,
            gen=torch.Generator().manual_seed(SEED_INSTALL[fact]),
            tag=f"e193b_install_{fact}", ckpt=ck,
            eval_at=set(INSTALL_EVAL_AT), on_eval=_on_eval,
            mix_random=E43.MIX_RANDOM, train_ids=train_ids, log=log)
        wall = time.time() - t2
        dev_used = dev_i
        sd_i = {k: v.detach().cpu().clone() for k, v in net_i.state_dict().items()}
        final_ro = gate_readout_cpu(sd_i, fact)
        gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                         and final_ro["r1i_nll"] <= GATE_R1I_MAX)
        patience_used = False
        if not gate_pass and not SMOKE:
            log(f"  GATE-0 miss for {fact} (pZ {final_ro['install60_pz']:.3f}"
                f" / R1i {final_ro['r1i_nll']:.2f}) — e082 patience branch: "
                f"ONE fresh {PATIENCE_STEPS}-step run from the PRE-{fact} net")
            net_i = TinyGPT(cfg)
            net_i.load_state_dict(prev_sd)
            net_i = net_i.to(dev_i)
            t3 = time.time()
            ck_p = CKPT_DIR / (f"e193b_smoke_pat_{fact}.pt" if SMOKE
                               else f"e193b_pat_{fact}.pt")
            traj = E43.exposure(
                net_i, fb["win"].to(dev_i), fb["mask"].to(dev_i),
                fb["anchor"].to(dev_i),
                steps=PATIENCE_STEPS, total=PATIENCE_STEPS,
                gen=torch.Generator().manual_seed(SEED_INSTALL[fact] + 1000),
                tag=f"e193b_pat_{fact}", ckpt=ck_p,
                eval_at={PATIENCE_STEPS // 2, PATIENCE_STEPS},
                on_eval=_on_eval, mix_random=E43.MIX_RANDOM,
                train_ids=train_ids, log=log)
            wall += time.time() - t3
            sd_i = {k: v.detach().cpu().clone()
                    for k, v in net_i.state_dict().items()}
            final_ro = gate_readout_cpu(sd_i, fact)
            gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                             and final_ro["r1i_nll"] <= GATE_R1I_MAX)
            patience_used = True
        del net_i
        set_device("cpu")
        if dev_used == "cuda":
            cooldown(COOLDOWN_S)
        # the interference co-report: the OTHER fact's dial after this install
        other = FACT2 if fact == FACT1 else FACT1
        other_ro = gate_readout_cpu(sd_i, other)
        install_recs[fact] = {
            "protocol": "e043 exposure VERBATIM (Dmix 16+48, 100 steps, "
                        "token-weighted masked union CE)",
            "gen_seed": SEED_INSTALL[fact] + (1000 if patience_used else 0),
            "steps": PATIENCE_STEPS if patience_used else INSTALL_STEPS,
            "traj": traj, "final": final_ro, "gate_pass": gate_pass,
            "patience_used": patience_used,
            "bars": {"pz_floor": GATE_PZ_FLOOR, "r1i_max": GATE_R1I_MAX},
            "after_install_other_fact_read": other_ro,
            "wall_s": round(wall, 1), "device": dev_used, "ckpt": ck.name,
        }
        log(f"STAGE install {fact}: pZ {final_ro['install60_pz']:.4f} "
            f"R1i {final_ro['r1i_nll']:.3f}/{final_ro['r1i_acc']:.3f} "
            f"-> GATE {'PASS' if gate_pass else 'FAIL'}"
            + (" [patience]" if patience_used else "")
            + f" | other fact ({other}) g0 read after: "
            f"{other_ro['install60_pz']:.4f}")
        journal({"stage": "install", "fact": fact,
                 **E43.jsonable(install_recs[fact])})
        prev_sd = sd_i
    G_GATE0 = {f: {"pass": install_recs[f]["gate_pass"],
                   "final": install_recs[f]["final"]}
               for f in (FACT1, FACT2)}
    both_gates = all(g["pass"] for g in G_GATE0.values())
    gate_note = ("both pass" if both_gates else
                 "AT LEAST ONE FAIL — reported, the cell continues "
                 "(Rule 12: report, do not adjudicate)")
    log(f"G_GATE0: {FACT1} {'PASS' if G_GATE0[FACT1]['pass'] else 'FAIL'}, "
        f"{FACT2} {'PASS' if G_GATE0[FACT2]['pass'] else 'FAIL'} "
        f"({gate_note})")

    # =====================================================================
    # STAGE 4 — the CONSOLIDATION (e113 recipe, two-fact union)
    # =====================================================================
    def offset_pool_windows(fact: str, j: int):
        nid = name_ids[fact]
        wins, masks = [], []
        for p, h in install_occ[fact]:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            if len(pre) != PRE + j or len(post) != POST_CAP - j:
                raise RuntimeError(f"window short at p={p} j={j} {fact}")
            wins.append(torch.cat([pre, nid, post]))
            m = torch.zeros(BLOCK - 1, dtype=torch.bool)
            m[PRE - 1 + j: PRE - 1 + j + L_NAME] = True
            masks.append(m)
        return torch.stack(wins), torch.stack(masks)

    pool_x, pool_m = [], []
    for fact in (FACT1, FACT2):
        for j in JITTERS:
            px, pm = offset_pool_windows(fact, j)
            pool_x.append(px)
            pool_m.append(pm)
    pool_x = torch.cat(pool_x)          # (600, 256)
    pool_m = torch.cat(pool_m)          # (600, 255)
    cons_anchor = torch.cat([
        torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                     for p, _ in install_occ[f][:16]])
        for f_ in (FACT1, FACT2)])      # (32, 256) — each fact's first 16
    G_GEO = {
        "jitter_offsets": list(JITTERS), "pool_shape": list(pool_x.shape),
        "mask_targets_per_window": int(pool_m[0].sum()),
        "anchor_shape": list(cons_anchor.shape),
        "pass": bool(pool_x.shape == (2 * 60 * len(JITTERS), BLOCK)
                     and int(pool_m[0].sum()) == L_NAME
                     and cons_anchor.shape == (32, BLOCK)),
    }
    assert G_GEO["pass"], f"pool geometry gate FAILED: {G_GEO}"
    log(f"G_GEO: two-fact jitter pool {tuple(pool_x.shape)} (offsets "
        f"{list(JITTERS)}), anchors {tuple(cons_anchor.shape)}: PASS")

    ck_cons = CKPT_DIR / ("e193b_smoke_root.pt" if SMOKE else "e193b_root.pt")
    dev_c = acquire_gpu("consolidation")
    set_device(dev_c)
    net_c = TinyGPT(cfg)
    net_c.load_state_dict(prev_sd)
    net_c = net_c.to(dev_c)
    opt_c = torch.optim.AdamW(net_c.parameters(), lr=LR_ADAMW,
                              betas=(0.9, 0.95), weight_decay=0.1)
    gen_c = torch.Generator().manual_seed(SEED_CONS)
    start_c, traj_c = 0, []
    if ck_cons.exists():
        st = torch.load(ck_cons, map_location="cpu", weights_only=False)
        net_c.load_state_dict(st["model"])
        opt_c.load_state_dict(st["opt"])
        gen_c.set_state(st["gen_state"].cpu().to(torch.uint8))
        start_c, traj_c = st["step"], st.get("traj", [])
        log(f"  consolidation resumed at step {start_c}")
    net_c.train()
    evl_c = TinyGPT(cfg)
    evl_c.load_state_dict(prev_sd)
    n_pool, n_anc = pool_x.shape[0], cons_anchor.shape[0]
    t4 = time.time()
    cap_c = 180.0 if dev_c == "cuda" else 1500.0
    for step in range(start_c + 1, CONS_STEPS + 1):
        ix = torch.randint(n_pool, (16,), generator=gen_c)
        aj = torch.randint(n_anc, (8,), generator=gen_c)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (8,), generator=gen_c)
        nw = pool_x[ix].to(dev_c)
        anc = torch.cat([cons_anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])],
                        0).to(dev_c)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev_c)
        m[:16] = pool_m[ix].to(dev_c)
        logits, _ = net_c(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt_c.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_c.parameters(), 1.0)
        opt_c.step()
        if step % 25 == 0 or step == CONS_STEPS or (time.time() - t4) > cap_c:
            sd_c = {k: v.detach().cpu().clone()
                    for k, v in net_c.state_dict().items()}
            evl_c.load_state_dict(sd_c)
            row = {"step": step, "loss": float(loss.item())}
            for f in (FACT1, FACT2):
                row[f"pz_{f}"] = battery_cell(evl_c, bat_ids[f][0],
                                              zids[f])["mean_pz"]
            row["ce_r"] = ce_fixed_cpu(evl_c, *r_eval_xy)
            traj_c.append(row)
            log(f"  [cons] s{step:3d} loss {row['loss']:.4f} "
                f"pz_{FACT1} {row[f'pz_{FACT1}']:.4f} "
                f"pz_{FACT2} {row[f'pz_{FACT2}']:.4f} CE_R {row['ce_r']:.4f}")
            torch.save({"model": net_c.state_dict(),
                        "opt": opt_c.state_dict(),
                        "gen_state": gen_c.get_state(), "step": step,
                        "traj": traj_c}, ck_cons)
        if (time.time() - t4) > cap_c:
            deviations.append(f"consolidation: time cap {cap_c:.0f}s at "
                              f"step {step}")
            break
    cons_wall = time.time() - t4
    net_c.eval()
    root_sd = {k: v.detach().cpu().clone()
               for k, v in net_c.state_dict().items()}
    del net_c, opt_c
    set_device("cpu")
    if dev_c == "cuda":
        cooldown(COOLDOWN_S)
    # THE ROOT — freeze it with full meta
    root_net = TinyGPT(cfg)
    root_net.load_state_dict(root_sd)
    root_net.eval()
    root_meta = {
        "experiment": "e193b", "desc": "FRESH two-fact consolidated root",
        "architecture": "6L/6H/192d/256-ctx TinyGPT (2,739,072 params — "
                        "organism 1's EXACT architecture)",
        "base": {"seed": SEED_BASE, "steps": BASE_STEPS,
                 "val_ce": val_ce, "ckpt": ck_base.name},
        "installs": {f: {"seed": install_recs[f]["gen_seed"],
                         "steps": install_recs[f]["steps"],
                         "gate_pass": install_recs[f]["gate_pass"],
                         "sites": "host_occ[:60]" if f == FACT1
                         else "host_occ[90:150]"} for f in (FACT1, FACT2)},
        "consolidation": {"recipe": "e113 via e157 stage A (two-fact union)",
                          "seed": SEED_CONS, "steps": CONS_STEPS},
        "built": common.now_iso(),
    }
    torch.save({"model": root_sd, "meta": root_meta}, ck_cons)
    theta0 = flat_params(root_net)
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    stage4 = {"recipe": "e113 jitter replay {-8,-4,0,+4,+8} x both facts' "
                        "60 sites (600-window union pool), batch 32 = 16 "
                        "pool + 8 paired + 8 random, seed 10901",
              "steps_ran": (traj_c[-1]["step"] if traj_c else 0),
              "traj": traj_c, "wall_s": round(cons_wall, 1),
              "device": dev_c, "ckpt": ck_cons.name}
    journal({"stage": "consolidation", **E43.jsonable(stage4)})
    log(f"STAGE 4 consolidation: {stage4['steps_ran']} steps on {dev_c} in "
        f"{cons_wall:.0f}s | root md5 {root_md5[:12]}")

    # =====================================================================
    # ROOT DIAL + G_ROOTSTRENGTH (both facts, BEFORE the terrain pass)
    # =====================================================================
    root_cells = {f: {f"g{j:+d}":
                      battery_cell(root_net, bat_ids[f][j], zids[f])["mean_pz"]
                      for j in READ_GEOS} for f in (FACT1, FACT2)}
    root_ce_r = ce_fixed_cpu(root_net, *r_eval_xy)

    primary_of: dict[str, dict] = {}
    for f in (FACT1, FACT2):
        g12 = root_cells[f]["g-12"]
        gcons = bool(g12 < SHUT_BAR)
        if gcons:
            best_j = max(GRID_GEOS, key=lambda j: root_cells[f][f"g{j:+d}"])
            primary_of[f] = {"ruler_j": best_j, "g_consolidated": True,
                             "note": f"g-12 root read {g12:.4f} < {SHUT_BAR} "
                             f"(e193's G_CONS lesson — REPORTED, not "
                             f"adjudicated); primary falls back to g{best_j:+d}"}
        else:
            primary_of[f] = {"ruler_j": RULER_J, "g_consolidated": False,
                             "note": "g-12 (e192-verbatim primary)"}
    G_ROOTSTRENGTH = {
        "dial": root_cells, "ce_r": root_ce_r,
        "primary": primary_of,
        "g_consolidated_facts": [f for f in (FACT1, FACT2)
                                 if primary_of[f]["g_consolidated"]],
        "note": "frozen fallback rule applied mechanically; g-12 tables "
                "always co-report; no post-hoc ruler choice is possible",
    }
    for f in (FACT1, FACT2):
        pr = primary_of[f]
        log(f"G_ROOTSTRENGTH {f}: g-12 {root_cells[f]['g-12']:.4f}"
            + (" (G_CONS — under the kill bar at D=0; REPORTED)"
               if pr["g_consolidated"] else " (above the bar)")
            + f" | primary g{pr['ruler_j']:+d} "
            f"{root_cells[f][f'g{pr['ruler_j']:+d}']:.4f} | dial "
            + " ".join(f"g{j:+d}:{root_cells[f][f'g{j:+d}']:.3f}"
                       for j in READ_GEOS))
    journal({"stage": "root_dial", **E43.jsonable(G_ROOTSTRENGTH)})

    # ruler battery handles
    primary_ids = {f: bat_ids[f][primary_of[f]["ruler_j"]] for f in (FACT1, FACT2)}
    coruler_ids = {f: {j: bat_ids[f][j] for j in GRID_GEOS}
                   for f in (FACT1, FACT2)}

    # =====================================================================
    # G_T0 — the t=0 gate + THIS root's measured matched step L2
    # =====================================================================
    def draw_step_batch(step_no: int):
        """The seed-10902 stream batch (net-independent construction)."""
        g = torch.Generator().manual_seed(FREEZE_SEED)
        for _ in range(step_no - 1):
            torch.randint(16, (ANCH_BS,), generator=g)
            torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
        aj = torch.randint(16, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    x1, y1 = draw_step_batch(1)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(root_net)
    tw.train()
    optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                             weight_decay=0.1)
    logits, _ = tw(x1)
    ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y1.reshape(-1)).item())
    optw.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y1.reshape(-1)).backward()
    gn1 = float(torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0))
    optw.step()
    theta1_adam = flat_params(tw)
    disp1 = float(torch.norm(theta1_adam - theta0))
    del tw, optw, logits
    G_T0 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1,
        "preclip_gnorm_measured": gn1, "clip_binds": bool(gn1 > 1.0),
        "adamw_step1_L2_measured": disp1,
        "note": "the step-1 batch md5 vs e185's stored hash — the stream "
                "construction is net-independent, so it MUST bit-match on "
                "this organism too; the fresh CPU AdamW step's L2 IS this "
                "cell's matched step L2 (never ported; organism 1's was "
                f"{F1_STEP_L2}, organism 2's 0.9164)",
        "pass": bool(x1_md5 == E185_XHASH[1] and disp1 > 0),
    }
    log(f"G_T0: x_md5 {'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}"
        f"; CE {ce1:.10f}; pre-clip gn {gn1:.4f} (clip "
        f"{'BINDS' if G_T0['clip_binds'] else 'slack'}); measured STEP_L2 "
        f"{disp1:.10f}: " + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 gate FAILED — abort (control failure)")
    STEP_L2 = disp1
    ck_s1 = CKPT_DIR / ("e193b_smoke_wash_s1.pt" if SMOKE
                        else "e193b_wash_s1.pt")
    torch.save({"model": {k: v.clone() for k, v in root_sd.items()},
                "theta1": theta1_adam, "meta": {"experiment": "e193b",
                "desc": "root + the fresh CPU AdamW step-1 state (the "
                        "cross-anchor for later cells)", "root_md5": root_md5,
                "step_l2": STEP_L2}}, ck_s1)

    # G_STREAM: the stream machinery verified steps 1..4 WITHOUT training
    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 5):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_4_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values()))}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..4): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # ---- the t=0 post-clip gradient: THIS root's g-ray
    gnet = copy.deepcopy(root_net)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1)
    loss_g = F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                             y1.reshape(-1))
    assert abs(float(loss_g.item()) - ce1) < 1e-9, "CE drifted between gates"
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(gnet.parameters(), 1.0)
    g0 = torch.cat([p.grad.detach().reshape(-1)
                    for p in gnet.parameters()]).clone()   # post-clip
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g, loss_g
    u_g = (g0 / torch.norm(g0)).clone()
    s_raw = torch.sign(g0)
    u_sign = (s_raw / torch.norm(s_raw)).clone()
    del g0, s_raw
    dir_ck = CKPT_DIR / ("smoke_e193b_static_dir_u.pt" if SMOKE
                         else "e193b_static_dir_u.pt")
    torch.save({"u": u_g, "theta0_md5": root_md5,
                "meta": {"experiment": "e193b", "desc":
                         "the e193b g-ray: u = g0/||g0||, g0 the post-clip "
                         "t=0 wash-stream batch gradient (seed 10902, e185 "
                         "convention, fresh two-fact root)",
                         "root": ck_cons.name, "gn_preclip": gn1}}, dir_ck)

    log("WHAT THESE ARMS GUARANTEE: NOTHING — the order could reorder per "
        "fact; kill-Ds could leave the windows; the pump could appear on "
        "random rays; the in-span spread could swallow the g-ray; the "
        "breaker could spare or kill; that openness is the point.")

    # ---------------- progressive partial writes -----------------------------
    rays: list[dict] = []
    rider_state = {"done": False, "rows": []}
    ladder_state: dict = {}
    inspan_state: dict = {}

    def write_partial(status: str, extra: dict | None = None) -> None:
        payload = {
            "experiment": "e193b_two_fact",
            "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"rays_done": len(rays), "grid_total": len(D_GRID),
                      "rider_done": rider_state["done"],
                      "ladder_done": list(ladder_state),
                      "inspan_done": list(inspan_state),
                      "elapsed_s": round(time.time() - T0, 1)},
            "registration": REGISTERED_BARS["registration"],
            "stages_partial": {
                "base": stage1,
                "installs": {f: install_recs[f] for f in (FACT1, FACT2)},
                "consolidation": stage4,
                "root": {"ckpt": ck_cons.name, "root_md5": root_md5,
                         "meta": root_meta},
            },
            "gates_partial": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                              "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                              "G_GEO": G_GEO, "G_GATE0": G_GATE0,
                              "G_ROOTSTRENGTH": G_ROOTSTRENGTH,
                              "G_T0": G_T0, "G_STREAM": G_STREAM},
            "step_l2_measured": STEP_L2,
            "rays_partial": {r["key"]: r["rows"] for r in rays},
            "rider_partial": rider_state["rows"],
            "ladder_partial": {k: v["traj"] for k, v in ladder_state.items()},
            "inspan_partial": {k: v["rows"] for k, v in inspan_state.items()},
        }
        if extra:
            payload.update(extra)
        save_json(rd / "metrics.json", E43.jsonable(payload))

    write_partial("PARTIAL — organism built + all pre-dispatch gates read "
                  "(namefree/splice/battery/anchor/geo/gate0/rootstrength/"
                  "t0/stream); rays starting")
    log("[partial] metrics.json written (organism + gates)")

    # =====================================================================
    # (1) THE FIVE-RAY TERRAIN MAP — static graded jumps, dual currency,
    #     BOTH facts' rulers at every grid point
    # =====================================================================
    evl = copy.deepcopy(root_net)
    evl.eval()

    def static_jump_read(u_vec: torch.Tensor, D: float) -> dict:
        thD = theta0 - D * u_vec
        disp_check = float(torch.norm(thD - theta0))
        load_flat(evl, thD)
        evl.eval()
        row = {"D": float(D), "rms": float(D) / RMS_DEN,
               "ce_r": ce_fixed_cpu(evl, *r_eval_xy),
               "disp_check": disp_check,
               "disp_dev": abs(disp_check - float(D))}
        for f in (FACT1, FACT2):
            gz = battery_cell(evl, primary_ids[f], zids[f])
            row[f"gm_{f}"] = gz["mean_pz"]
            row[f"argz_{f}"] = gz["frac_argmax_z"]
            for j in GRID_GEOS:
                row[f"g{j:+d}_{f}"] = battery_cell(
                    evl, coruler_ids[f][j], zids[f])["mean_pz"]
        return row

    def draw_gaussian_ray(seed: int) -> tuple[torch.Tensor, dict]:
        g = torch.Generator().manual_seed(seed)
        blocks = [torch.randn(p.shape, generator=g).reshape(-1)
                  for p in root_net.parameters()]
        v = torch.cat(blocks)
        sq = v * v
        tot = float(torch.sum(sq))
        spreads = {}
        i = 0
        for (nm, p) in root_net.named_parameters():
            n = p.numel()
            spreads[nm] = {"numel_share": n / n_params,
                           "l2_share": float(torch.sum(sq[i:i + n])) / tot}
            i += n
        max_spread = max(abs(s["l2_share"] - s["numel_share"])
                         for s in spreads.values())
        return (v / torch.norm(v)).clone(), {
            "seed": seed, "draw": "per-tensor torch.randn(shape, generator) "
            "in net.parameters() order (g3K convention, flat coordinates)",
            "even_spread": {"max_abs_share_dev": max_spread,
                            "note": "co-read only — never gated"}}

    rays = [
        {"key": "R1_G", "label": "g-ray (this root's own t=0 wash gradient)",
         "u": u_g, "prov": "u = g0/||g0||, g0 the post-clip t=0 wash-stream "
         "batch gradient (seed 10902, e185 convention, fresh two-fact root; "
         "direction checkpointed; fact-independent by construction)",
         "rows": []},
        {"key": "R2_SIGN", "label": "static sign(g_0) ray, normalized",
         "u": u_sign, "prov": "sign(g_0)/||sign(g_0)|| (clip-invariant)",
         "rows": []},
    ]
    for sd in RAND_SEEDS:
        uv, meta = draw_gaussian_ray(sd)
        rays.append({"key": f"R{len(rays) + 1}_GAUSS_s{sd}",
                     "label": f"Gaussian unit ray seed {sd}",
                     "u": uv, "prov": meta, "rows": []})

    for r in rays:
        for D in D_GRID:
            row = static_jump_read(r["u"], D)
            r["rows"].append(row)
            journal({"ray": r["key"], "kind": "grid", **row})
            log(f"  {r['key']:>14s} D {D:5.3f} (rms {row['rms']:.2e}): "
                f"{FACT1} {row[f'gm_{FACT1}']:.4f} "
                f"{FACT2} {row[f'gm_{FACT2}']:.4f} CE_R {row['ce_r']:.4f} "
                f"[disp dev {row['disp_dev']:.1e}]")
        write_partial(f"PARTIAL: {r['key']} complete ({len(D_GRID)} points)")

    def ray_summary(r: dict) -> dict:
        rows = r["rows"]
        out = {"key": r["key"], "label": r["label"], "kills": {},
               "pump": {}}
        for f in (FACT1, FACT2):
            def kill_for(field):
                dead = [row for row in rows if row[field] <= SHUT_BAR]
                fd = dead[0] if dead else None
                return {"first_dead_D": (fd["D"] if fd is not None else None),
                        "first_dead_rms": (fd["rms"] if fd is not None
                                           else None),
                        "unresolved_high": fd is None,
                        "first_dead_read": (fd[field] if fd is not None
                                            else None)}
            out["kills"][f] = {"primary": kill_for(f"gm_{f}")}
            for j in GRID_GEOS:
                out["kills"][f][f"coruler_g{j:+d}"] = kill_for(f"g{j:+d}_{f}")
            root_p = root_cells[f][f"g{primary_of[f]['ruler_j']:+d}"]
            small = [row for row in rows if row["D"] in SMALL_D_PUMP_BAND]
            max_rise = max((row[f"gm_{f}"] - root_p) for row in small)
            out["pump"][f] = {
                "small_D_max_rise": max_rise,
                "small_D_strict_rise": any(row[f"gm_{f}"] > root_p
                                           for row in small),
                "pump_ridge": bool(max_rise > PUMP_MARGIN),
                "margin": PUMP_MARGIN,
                "profile": [{"D": row["D"], "gm": row[f"gm_{f}"]}
                            for row in small],
            }
        return out

    summaries = [ray_summary(r) for r in rays]
    for s in summaries:
        ktxt = []
        for f in (FACT1, FACT2):
            kd = s["kills"][f]["primary"]["first_dead_D"]
            ktxt.append(f"{f} kill " + (f"{kd}" if kd is not None
                                        else ">4.0"))
        log(f"  SUMMARY {s['key']:>14s}: " + "; ".join(ktxt)
            + "; pumps " + "; ".join(
                f"{f} {s['pump'][f]['small_D_max_rise']:+.4f}"
                f"({'RIDGE' if s['pump'][f]['pump_ridge'] else 'no ridge'})"
                for f in (FACT1, FACT2)))
    write_partial("PARTIAL: all ray families complete + summarized",
                  {"ray_summaries": summaries})

    pump_read = {
        "band": list(SMALL_D_PUMP_BAND),
        "family1_committed": F1_PUMP,
        "e193_organism2_committed": {"g_max_rise": E193_PUMP_RISE,
                                     "ridge": False},
        "rays": {s["key"]: s["pump"] for s in summaries},
        "thesis_note": "PUMP-PER-FACT (co-registered, no bar): present/"
                       "absent per fact — with e193 (absent on organism 2), "
                       "this decides the ridge-vs-cliff thesis (T153).",
    }

    # =====================================================================
    # THE RIDER — 300 pinned steps down the FROZEN g-ray (report-only,
    #             both facts' primaries at every read)
    # =====================================================================
    step_vec = (STEP_L2 / RIDER_N) * u_g
    th_r = theta0.clone()
    evl_r = copy.deepcopy(root_net)
    evl_r.eval()
    for k in range(1, RIDER_N + 1):
        th_r.sub_(step_vec)
        if k % RIDER_EVERY == 0 or k == RIDER_N:
            disp_k = float(torch.norm(th_r - theta0))
            load_flat(evl_r, th_r)
            row = {"step": k, "D_cum": disp_k, "rms": disp_k / RMS_DEN,
                   "ce_r": ce_fixed_cpu(evl_r, *r_eval_xy),
                   "disp_dev": abs(disp_k - k * (STEP_L2 / RIDER_N))}
            for f in (FACT1, FACT2):
                gz = battery_cell(evl_r, primary_ids[f], zids[f])
                row[f"gm_{f}"] = gz["mean_pz"]
                row[f"argz_{f}"] = gz["frac_argmax_z"]
            rider_state["rows"].append(row)
            journal({"ray": "RIDER", "kind": "rider", **row})
            log(f"  RIDER step {k:3d} (D {row['D_cum']:.4f}): "
                f"{FACT1} {row[f'gm_{FACT1}']:.4f} "
                f"{FACT2} {row[f'gm_{FACT2}']:.4f} CE_R {row['ce_r']:.4f}")
    rider_state["done"] = True
    rider_pattern = {}
    for f in (FACT1, FACT2):
        r_dead = [row for row in rider_state["rows"]
                  if row[f"gm_{f}"] <= SHUT_BAR]
        static_kill = next(s for s in summaries
                           if s["key"] == "R1_G")["kills"][f]["primary"]
        rider_pattern[f] = {
            "first_dead_step": (r_dead[0]["step"] if r_dead else None),
            "first_dead_D": (r_dead[0]["D_cum"] if r_dead else None),
            "static_g_ray_kill_D": static_kill["first_dead_D"],
            "walk_D_cap": STEP_L2,
            "dies_at_static_cliff": bool(
                r_dead and static_kill["first_dead_D"] is not None
                and r_dead[0]["D_cum"] <= static_kill["first_dead_D"] * 1.1),
            "survives_walk": not r_dead,
            "note": "report-only (no registered rider bar); the pinned "
                    "walk's points coincide with the static g-ray BY "
                    "CONSTRUCTION (e192's disclosed coincidence)",
        }
    log("RIDER pattern: " + "; ".join(
        f"{f}: {'dies at/near static cliff' if v['dies_at_static_cliff'] else ('survives' if v['survives_walk'] else 'dies inside walk')}"
        for f, v in rider_pattern.items()))
    write_partial("PARTIAL: rider complete", {"rider": rider_pattern})

    # =====================================================================
    # (2) THE DENSITY LADDER + THE BREAKER (report-only; primary fact =
    #     ZEPHYRA frozen; every-step read on ITS primary ruler)
    # =====================================================================
    PF = FACT1
    pf_j = primary_of[PF]["ruler_j"]

    def run_arm(tag: str) -> dict:
        kind = next(k for t, _, k, _ in ARM_SPECS if t == tag)
        frac = next(fr for t, _, _, fr in ARM_SPECS if t == tag)
        net = copy.deepcopy(root_net)
        net.train()
        evl_a = copy.deepcopy(root_net)
        gen = torch.Generator().manual_seed(FREEZE_SEED)
        mgen = torch.Generator().manual_seed(MAGSHUF_SEED)
        step, stop = 0, None
        traj: list[dict] = []
        x_hashes: dict[int, str] = {}
        max_l2_dev, jaccards, prev_mask = 0.0, [], None
        magshuf_cos = []
        prev_flat, prev_gm, prev_d = (theta0,
                                      root_cells[PF][f"g{pf_j:+d}"], 0.0)
        zeph = 0
        ckpt_set = set(s for s in CKPT_STEPS if s <= STEP_CAP)
        t_arm = time.time()
        while stop is None:
            step += 1
            if step > STEP_CAP:
                stop = {"kind": "cap", "step": step - 1,
                        "reason": f"step cap {STEP_CAP}",
                        "final_D": prev_d, "final_gm": prev_gm}
                break
            aj = torch.randint(16, (ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                               generator=gen)
            anc = anchor_neutral[aj]
            rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
            for w in rnd:                    # name-free VERIFY (hard-fail)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt or FACT2 in txt:   # the FULL nonce only
                    zeph += 1                 # ('MIRA' alone occurs 34x in
                                             # the corpus as a substring)
            x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
            y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
            x_hashes[step] = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
            logits_a, _ = net(x)
            loss = F.cross_entropy(logits_a.reshape(-1, logits_a.shape[-1]),
                                   y.reshape(-1))
            net.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                         1.0))
            g_t = torch.cat([p.grad.detach().reshape(-1)
                             for p in net.parameters()]).clone()  # post-clip
            if kind == "topk":
                delta, k, mask = topk_update(g_t, frac, STEP_L2)
                if prev_mask is not None:
                    inter = int((mask & prev_mask).sum())
                    jaccards.append(inter / (2 * k - inter))
                prev_mask = mask
            elif kind == "magshuf":
                delta, k, mask, kept, broken = magshuffle_update(
                    g_t, frac, STEP_L2, mgen)
                tk_delta, _, _ = topk_update(g_t, frac, STEP_L2)
                magshuf_cos.append(cos64(delta, tk_delta))
                if prev_mask is not None:
                    inter = int((mask & prev_mask).sum())
                    jaccards.append(inter / (2 * k - inter))
                prev_mask = mask
            elif kind == "sign":
                delta, k = sign_update(g_t, STEP_L2), None
            else:
                delta, k = raw_update(g_t, STEP_L2), None
            l2dev = abs(float(torch.norm(delta.double())) - STEP_L2)
            assert l2dev < L2_ASSERT_TOL, \
                f"{tag}: per-step L2 dev {l2dev:.2e} — matched-L2 assertion"
            max_l2_dev = max(max_l2_dev, l2dev)
            cur = prev_flat + delta
            load_flat(net, cur)
            cum_disp = float(torch.norm(cur - theta0))
            evl_a.load_state_dict({k_: v.detach().cpu().clone()
                                   for k_, v in net.state_dict().items()})
            evl_a.eval()
            gz = battery_cell(evl_a, primary_ids[PF], zids[PF])
            row = {"step": step, "ce_batch": float(loss.item()),
                   "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
                   "l2_dev": l2dev, "preclip_gnorm": gnorm, "k": k,
                   "jaccard": (jaccards[-1] if jaccards else None),
                   "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
                   "ratio_vs_g_kill": None}
            if step in ckpt_set:
                row["ce_r"] = ce_fixed_cpu(evl_a, *r_eval_xy)
                for f in (FACT1, FACT2):
                    row[f"gm_{f}"] = battery_cell(
                        evl_a, primary_ids[f], zids[f])["mean_pz"]
                    for j in GRID_GEOS:
                        row[f"g{j:+d}_{f}"] = battery_cell(
                            evl_a, coruler_ids[f][j], zids[f])["mean_pz"]
                log(f"  [{tag}] +{step}: {PF} {row['gm']:.4f} "
                    f"{FACT2} {row[f'gm_{FACT2}']:.4f} CE_R "
                    f"{row['ce_r']:.4f} D {cum_disp:.4f}")
            traj.append(row)
            journal({"arm": tag, **row})
            prev_flat, prev_gm, prev_d = cur, row["gm"], cum_disp
            if gz["mean_pz"] <= SHUT_BAR:       # KILL: densify + stop (opt1c)
                dens = []
                for f_ in DENSIFY_F:
                    pt_flat = prev_flat + (f_ - 1.0) * delta
                    load_flat(evl_a, pt_flat)
                    evl_a.eval()
                    gzd = battery_cell(evl_a, primary_ids[PF], zids[PF])
                    dens.append({"f": f_, "gm": gzd["mean_pz"],
                                 "D": float(torch.norm(pt_flat - theta0))})
                prev_row = traj[-2] if len(traj) >= 2 else \
                    {"gm": root_cells[PF][f"g{pf_j:+d}"], "cum_disp": 0.0}
                d_raw = interp_d_kill(prev_row["gm"], row["gm"],
                                      prev_row["cum_disp"], cum_disp)
                d_dens = dens_d_kill(prev_row["gm"], row["gm"],
                                     prev_row["cum_disp"], cum_disp, dens)
                stop = {"kind": "kill", "step": step,
                        "gm_at_kill": row["gm"], "D_kill_raw": d_raw,
                        "D_kill": d_dens, "dens": dens,
                        "reason": f"{PF} primary ruler <= {SHUT_BAR} at an "
                                  "every-step read"}
                log(f"  [{tag}] KILL at +{step}: {PF} {row['gm']:.4f} "
                    f"D_kill(dens) {d_dens:.4f} (raw {d_raw:.4f})")
            elif cum_disp >= D_TARGET:
                stop = {"kind": "target", "step": step,
                        "reason": f"D {cum_disp:.4f} >= D_TARGET {D_TARGET}",
                        "final_D": cum_disp, "final_gm": row["gm"]}
                log(f"  [{tag}] D_TARGET reached alive at +{step} "
                    f"({PF} {row['gm']:.4f}, D {cum_disp:.4f})")
        net.eval()
        assert zeph == 0, f"{tag}: name token leaked into a window"
        return {"traj": traj, "stop": stop, "x_hashes": x_hashes,
                "max_l2_dev": max_l2_dev,
                "jaccard_mean": (float(sum(jaccards) / len(jaccards))
                                 if jaccards else None),
                "magshuf_cos_mean": (float(sum(magshuf_cos) / len(magshuf_cos))
                                     if magshuf_cos else None),
                "seconds": round(time.time() - t_arm, 1)}

    for k_i, (tag, desc, _kind, _frac) in enumerate(ARM_SPECS):
        if k_i and not SMOKE:
            time.sleep(INTER_ARM_S)
        log("=" * 78)
        log(f"LADDER ARM {tag} — {desc}; cap {STEP_CAP}, D_target "
            f"{D_TARGET}, per-step L2 {STEP_L2:.6f} (measured), readout = "
            f"{PF} primary g{pf_j:+d}")
        ladder_state[tag] = run_arm(tag)
        write_partial(f"PARTIAL: ladder arm {tag} complete")

    # G_INPUTS: the shared stream across arms AND vs e185's stored hashes
    n_shared = min(len(v["x_hashes"]) for v in ladder_state.values())
    per_step, vs_stored = {}, {}
    for step in range(1, n_shared + 1):
        hs = {t: ladder_state[t]["x_hashes"].get(step) for t in ARM_ORDER}
        same = bool(all(h is not None for h in hs.values())
                    and len(set(hs.values())) == 1)
        per_step[step] = {**hs, "identical": same}
        if step in E185_XHASH:
            vs_stored[step] = {"match": bool(hs[ARM_ORDER[0]]
                                             == E185_XHASH[step])}
    G_INPUTS = {"per_step": per_step, "vs_e185_stored": vs_stored,
                "pass": bool(all(v["identical"] for v in per_step.values())
                             and all(v["match"]
                                     for v in vs_stored.values()))}
    log(f"G_INPUTS: per-step batches bit-identical across all "
        f"{len(ARM_ORDER)} arms AND vs e185's stored hashes: "
        + ("PASS" if G_INPUTS["pass"] else "FAIL"))
    assert G_INPUTS["pass"], "ladder input streams diverged"

    # =====================================================================
    # THE IN-SPAN 3-DRAW RANGE (this root's own wash-step-history span)
    # =====================================================================
    log("=" * 78)
    log(f"IN-SPAN — {WASH_HIST_STEPS} CPU AdamW wash steps (seed "
        f"{FREEZE_SEED} stream) -> Gram SVD span -> 3 random in-span unit "
        f"draws, each walked down the {len(D_GRID)}-point grid")
    wh_net = copy.deepcopy(root_net)
    wh_net.train()
    wh_opt = torch.optim.AdamW(wh_net.parameters(), lr=LR_ADAMW,
                               betas=(0.9, 0.95), weight_decay=0.1)
    wh_gen = torch.Generator().manual_seed(FREEZE_SEED)
    wh_theta = theta0.clone()
    hist_disp = []
    for s_wh in range(1, WASH_HIST_STEPS + 1):
        aj = torch.randint(16, (ANCH_BS,), generator=wh_gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=wh_gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        logits_w, _ = wh_net(x)
        lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                             y.reshape(-1))
        wh_opt.zero_grad(set_to_none=True)
        lw.backward()
        torch.nn.utils.clip_grad_norm_(wh_net.parameters(), 1.0)
        wh_opt.step()
        th_new = flat_params(wh_net)
        hist_disp.append(th_new - wh_theta)
        wh_theta = th_new
    wh_net.eval()
    del wh_net, wh_opt
    H = torch.stack(hist_disp)
    basis = svd_basis(H)
    Vr = basis["Vp"].contiguous()             # (r, N)
    Vr64 = Vr.to(torch.float64)
    r_avail = basis["rank_eff"]
    removed = float(torch.norm(Vr64 @ u_g.to(torch.float64)))
    torch.save({"H": H, "meta": {"experiment": "e193b",
                "desc": f"{WASH_HIST_STEPS} CPU AdamW wash-step "
                        "displacements (seed 10902 stream) — the in-span "
                        "span source", "root_md5": root_md5}},
               CKPT_DIR / ("e193b_smoke_wash_hist.pt" if SMOKE
                           else "e193b_wash_hist.pt"))
    span_meta = {
        "n_segments": int(Vr.shape[0]), "rank_eff": r_avail,
        "participation_ratio": basis["pr"],
        "wash_dir_removed_fraction": removed,
        "note": "e_chart's convention: the span of REALIZED wash steps "
                "(Adam's sign-flattened displacements — its own caveat "
                "carried); this root's OWN history, never a ported span",
    }
    log(f"  span: {span_meta['n_segments']} segments, rank {r_avail}, PR "
        f"{basis['pr']:.2f}; ||Vr^T u_g|| = {removed:.4f}")
    del H, hist_disp

    for sd in INSPAN_SEEDS:
        g_ = torch.Generator().manual_seed(sd)
        c = torch.randn(r_avail, generator=g_, dtype=torch.float64)
        v = Vr64.T @ c
        u_in = (v / torch.norm(v)).to(torch.float32)
        rows = []
        for D in D_GRID:
            row = static_jump_read(u_in, D)
            rows.append(row)
            journal({"ray": f"INSPAN_s{sd}", "kind": "grid", **row})
            log(f"  INSPAN s{sd} D {D:5.3f}: {FACT1} "
                f"{row[f'gm_{FACT1}']:.4f} {FACT2} {row[f'gm_{FACT2}']:.4f} "
                f"CE_R {row['ce_r']:.4f}")
        inspan_state[f"s{sd}"] = {
            "rows": rows, "u_md5": hashlib.md5(
                u_in.numpy().tobytes()).hexdigest(),
            "cos_to_wash_dir": cos64(u_in, u_g),
        }
        write_partial(f"PARTIAL: in-span draw s{sd} complete")

    # =====================================================================
    # ADJUDICATION (registered clauses; frozen operationalizations)
    # =====================================================================
    summ = {s["key"]: s for s in summaries}
    g_s, sign_s = summ["R1_G"], summ["R2_SIGN"]
    rand_s = [s for s in summaries if "_GAUSS_" in s["key"]]
    per_fact: dict = {}
    for f in (FACT1, FACT2):
        g_kill = g_s["kills"][f]["primary"]["first_dead_D"]
        sign_kill = sign_s["kills"][f]["primary"]["first_dead_D"]
        rand_kills = [s["kills"][f]["primary"]["first_dead_D"]
                      for s in rand_s]
        rand_eff = (min(k for k in rand_kills if k is not None)
                    if any(k is not None for k in rand_kills)
                    else float("inf"))
        order_holds = bool(g_kill is not None and sign_kill is not None
                           and g_kill < sign_kill and sign_kill < rand_eff)
        order_unestablished = sign_kill is None or g_kill is None
        random_wide = all(
            (s["kills"][f]["primary"]["first_dead_D"] is None)
            or (s["kills"][f]["primary"]["first_dead_D"] >= 2 * g_kill)
            for s in rand_s) and g_kill is not None
        g_in_window = bool(g_kill is not None
                           and G_WINDOW[0] <= g_kill <= G_WINDOW[1])
        sign_in_window = bool(sign_kill is not None
                              and SIGN_WINDOW[0] <= sign_kill
                              and sign_kill <= 4.0)  # grid cap < 5.0 upper
        terrain_fact = bool(order_holds and random_wide and g_in_window
                            and sign_in_window)
        order_change = bool(not order_holds and not order_unestablished)
        g_move = bool(g_kill is not None and not g_in_window)
        sign_move = bool(sign_kill is not None
                         and not (SIGN_WINDOW[0] <= sign_kill <= 5.0))
        scrambles_fact = bool(order_change or g_move or sign_move)
        rand4x = all(
            (s["kills"][f]["primary"]["first_dead_D"] is None)
            or (s["kills"][f]["primary"]["first_dead_D"] >= 4 * g_kill)
            for s in rand_s) if g_kill is not None else None
        per_fact[f] = {
            "primary_ruler": f"install-60 g{primary_of[f]['ruler_j']:+d}",
            "g_consolidated": primary_of[f]["g_consolidated"],
            "g_kill": g_kill, "sign_kill": sign_kill,
            "rand_kills": {s["key"]: s["kills"][f]["primary"]["first_dead_D"]
                           for s in rand_s},
            "rand_eff": (rand_eff if rand_eff != float("inf") else None),
            "order_holds": order_holds,
            "order_unestablished": order_unestablished,
            "random_wide": random_wide, "randoms_ge_4x": rand4x,
            "g_window": list(G_WINDOW), "g_in_window": g_in_window,
            "sign_window": list(SIGN_WINDOW), "sign_in_window":
                sign_in_window,
            "sign_grid_cap_note": "sign window upper 5.0 exceeds the 4.0 "
                                  "grid: a resolved sign kill <= 4.0 is "
                                  "inside iff >= 1.25; unresolved sign "
                                  "leaves the order unestablished",
            "terrain_fact": terrain_fact, "order_change": order_change,
            "g_move": g_move, "sign_move": sign_move,
            "scrambles_fact": scrambles_fact,
            "kill_ratios": {
                "sign_over_g": (sign_kill / g_kill
                                if (sign_kill and g_kill) else None),
                "rand_min_over_g": ((rand_eff / g_kill)
                                    if (rand_eff != float("inf")
                                        and g_kill)
                                    else ">grid-cap (unresolved-high)")},
            "pump_g_ridge": g_s["pump"][f]["pump_ridge"],
            "pump_g_max_rise": g_s["pump"][f]["small_D_max_rise"],
            "pump_sign_ridge": sign_s["pump"][f]["pump_ridge"],
            "pump_sign_max_rise": sign_s["pump"][f]["small_D_max_rise"],
            "pump_any_rand_ridge": any(s["pump"][f]["pump_ridge"]
                                       for s in rand_s),
            "rider": rider_pattern[f],
            "inspan": {
                sd_k: next((r["D"] for r in v["rows"]
                            if r[f"gm_{f}"] <= SHUT_BAR), None)
                for sd_k, v in inspan_state.items()},
        }
        kf = per_fact[f]
        log(f"  FACT {f}: g {g_kill} (window {G_WINDOW}) sign {sign_kill} "
            f"(window {SIGN_WINDOW}) randoms "
            + "/".join((str(k) if k is not None else ">4.0")
                       for k in rand_kills)
            + f" | order {'HOLDS' if order_holds else ('UNESTABLISHED' if order_unestablished else 'CHANGED')}"
            + f" | terrain_fact {terrain_fact} scrambles_fact "
            f"{scrambles_fact} | pump ridge "
            f"{'YES' if kf['pump_g_ridge'] else 'no'}")

    # in-span RANGE per fact (the missing number)
    inspan_range = {}
    for f in (FACT1, FACT2):
        kills = [per_fact[f]["inspan"][k] for k in inspan_state]
        resolved = [k for k in kills if k is not None]
        inspan_range[f] = {
            "kill_Ds": kills,
            "n_resolved": len(resolved),
            "range": ([min(resolved), max(resolved)]
                      if resolved else None),
            "spread_x": (max(resolved) / min(resolved)
                         if len(resolved) >= 2 and min(resolved) > 0
                         else None),
            "g_kill_for_scale": per_fact[f]["g_kill"],
            "note": "THE MISSING NUMBER (R60-critic Attack 1): e131's "
                    "suppressed in-span seed spread was 3x; this root's "
                    "own three draws' kill-D RANGE, grid resolution "
                    f"{max(b - a for a, b in zip(D_GRID[:-1], D_GRID[1:])):.2f}",
        }
        log(f"  INSPAN RANGE {f}: kills {kills} -> range "
            + (f"[{min(resolved):.2f}, {max(resolved):.2f}]"
               if resolved else "none resolved")
            + f" (g-ray kill {per_fact[f]['g_kill']} for scale)")

    # the ladder + breaker outcomes (report-only, ratio form)
    g_kill_pf = per_fact[PF]["g_kill"]
    arm_out = {}
    for t in ARM_ORDER:
        stop = ladder_state[t]["stop"]
        final = ladder_state[t]["traj"][-1]
        killed = stop["kind"] == "kill"
        arm_out[t] = {
            "outcome": ("kill" if killed else
                        ("target" if stop["kind"] == "target" else "cap")),
            "D_kill": (stop["D_kill"] if killed else None),
            "D_kill_raw": (stop.get("D_kill_raw") if killed else None),
            "final_D": final["cum_disp"], "final_gm": final["gm"],
            "ratio_vs_g_kill": ((stop["D_kill"] / g_kill_pf)
                                if (killed and g_kill_pf) else None),
            "max_l2_dev": ladder_state[t]["max_l2_dev"],
            "jaccard_mean": ladder_state[t]["jaccard_mean"],
            "magshuf_cos_mean": ladder_state[t]["magshuf_cos_mean"],
            "k": final.get("k"),
        }
    breaker = {
        "registration": "REPORT-ONLY (R60-critic Attack 2 rank-1): top-10% "
                        "SUPPORT and coordinate SIGNS kept, |g| VALUES "
                        "permuted within the support (per-step perm, "
                        f"stream seed {MAGSHUF_SEED})",
        "topk10_D_kill": arm_out["a_topk10"]["D_kill"],
        "magshuf_D_kill": arm_out["a_magshuf"]["D_kill"],
        "verdict_letter": None, "_note": "filled below",
    }
    if arm_out["a_magshuf"]["D_kill"] is not None \
            and arm_out["a_topk10"]["D_kill"]:
        within = abs(arm_out["a_magshuf"]["D_kill"]
                     - arm_out["a_topk10"]["D_kill"]) \
            <= 0.25 * arm_out["a_topk10"]["D_kill"]
        breaker["verdict_letter"] = (
            f"kills at {arm_out['a_magshuf']['D_kill']:.4f} "
            + ("~= the topk-10 kill — SUPPORT (which coordinates), not "
               "PAIRING, carries the kill" if within
               else f"vs topk-10 {arm_out['a_topk10']['D_kill']:.4f} — "
                    "outside the +-25% band; magnitude-PAIRING contributes"))
    elif arm_out["a_magshuf"]["outcome"] != "kill":
        breaker["verdict_letter"] = (
            "SPARED — magnitude-pairing is convicted interventionally "
            "(support + signs alone cannot kill)")
    breaker.pop("_note")

    # ---- composite headline
    any_scrambles = any(per_fact[f]["scrambles_fact"] for f in (FACT1, FACT2))
    all_terrain = all(per_fact[f]["terrain_fact"] for f in (FACT1, FACT2))
    if SMOKE:
        headline = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif any_scrambles:
        headline = "ORDER-SCRAMBLES"
        parts = []
        for f in (FACT1, FACT2):
            if per_fact[f]["scrambles_fact"]:
                why = []
                if per_fact[f]["order_change"]:
                    why.append("order changed")
                if per_fact[f]["g_move"]:
                    why.append(f"g-kill {per_fact[f]['g_kill']} outside "
                               f"{list(G_WINDOW)}")
                if per_fact[f]["sign_move"]:
                    why.append(f"sign-kill {per_fact[f]['sign_kill']} "
                               f"outside {list(SIGN_WINDOW)}")
                parts.append(f"{f}: " + "; ".join(why))
        clause = ("any fact's order changes or kill-Ds move > 2x — "
                  "'terrain' rescopes to biography. " + ". ".join(parts))
    elif all_terrain:
        headline = "TERRAIN-REPLICATES"
        clause = ("for BOTH facts the ray order g < sign < random holds "
                  "(randoms >= 2x the g-kill) and kill-Ds within [0.5x, "
                  "2x] of organism-1's — the terrain at n=3 organisms, "
                  "fact-varied, architecture-pinned. "
                  + "; ".join(
                      f"{f}: g {per_fact[f]['g_kill']} "
                      f"({per_fact[f]['kill_ratios']['sign_over_g']:.2f}x "
                      f"sign, randoms "
                      + ("all >= 2x" if per_fact[f]["random_wide"]
                         else "width short")
                      + ")" for f in (FACT1, FACT2)))
    else:
        headline = "GRADED"
        graded_parts = []
        for f in (FACT1, FACT2):
            o = ("holds" if per_fact[f]["order_holds"]
                 else ("unestablished" if per_fact[f]["order_unestablished"]
                       else "changed"))
            graded_parts.append(
                f"{f}: terrain_fact {per_fact[f]['terrain_fact']} (order "
                f"{o}, g "
                f"{'in' if per_fact[f]['g_in_window'] else 'OUT of'} window"
                f", sign "
                + ("in window" if per_fact[f]["sign_in_window"]
                   else "out/unresolved"))
        clause = ("a partial pattern — " + "; ".join(graded_parts)
                  + ". The tables verbatim.")
    pump_verdict = {
        f: ("RIDGE PRESENT (max rise "
            f"{per_fact[f]['pump_g_max_rise']:+.4f} > {PUMP_MARGIN})"
            if per_fact[f]["pump_g_ridge"]
            else f"ridge ABSENT (max rise {per_fact[f]['pump_g_max_rise']:+.4f})")
        for f in (FACT1, FACT2)}
    log("=" * 78)
    log(f"E193B HEADLINE: {headline}")
    log(f"  {clause}")
    log("  PUMP-PER-FACT (no bar): " + "; ".join(
        f"{f}: {v}" for f, v in pump_verdict.items())
        + " | organism 1: RIDGE (+0.0449); organism 2 (e193): ABSENT "
          "(-0.0788)")
    log("=" * 78)

    # ---- the CE_R canary (ordering report, never adjudicated)
    ce_canary = {
        "root": root_ce_r,
        "g_ray_small_D": [r["ce_r"] for r in rays[0]["rows"]
                          if r["D"] in SMALL_D_PUMP_BAND],
        "g_ray_at_cap": rays[0]["rows"][-1]["ce_r"],
        "note": "the ordering (CE_R rises as the fact dies) reported "
                "verbatim, never adjudicated",
    }

    # =====================================================================
    # figures
    # =====================================================================
    def fig_terrain():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, f in zip(axes, (FACT1, FACT2)):
            fld = f"gm_{f}"
            for r, s in zip(rays, summaries):
                style = ("-", "#1a6faa") if r["key"] == "R1_G" else (
                    ("-", "#c27d16") if r["key"] == "R2_SIGN"
                    else ("--", "#777777"))
                ax.plot([x["D"] for x in r["rows"]],
                        [x[fld] for x in r["rows"]], style[0],
                        color=style[1], lw=1.6,
                        label=r["label"].split(" (")[0], alpha=0.9)
            for sd_k, v in inspan_state.items():
                ax.plot([x["D"] for x in v["rows"]],
                        [x[fld] for x in v["rows"]], ":",
                        color="#9467bd", lw=1.1, alpha=0.8)
            ax.axhline(SHUT_BAR, color="k", ls=":", lw=1)
            ax.axvline(F1_GKILL, color="#1a6faa", ls=":", lw=1, alpha=0.7)
            ax.axvline(F1_SIGN_KILL, color="#c27d16", ls=":", lw=1,
                       alpha=0.7)
            ax.set_xlabel("D (L2 displacement; rms = D/1655.01)")
            ax.set_title(f"{f} — primary ruler "
                         f"g{primary_of[f]['ruler_j']:+d}"
                         + (" (G_CONS fallback)" if primary_of[f]["g_consolidated"] else ""))
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.02)
        axes[0].set_ylabel("mean p(name onset) — install-60 battery")
        axes[0].legend(fontsize=7, loc="lower left")
        fig.suptitle("E193B — the two-fact terrain at organism-1's exact "
                     "architecture (fresh root s5301; dotted vlines = "
                     "organism-1's committed kills 0.92/2.5; purple dotted "
                     "= the 3 in-span draws)", fontsize=9)
        fig.tight_layout()
        fig.savefig(rd / "e193b_fig5_two_fact.png", dpi=130)
        plt.close(fig)

    def fig_ladder():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
        names = [t for t in ARM_ORDER if arm_out[t]["D_kill"] is not None]
        xs = range(len(names))
        ax1.bar(xs, [arm_out[t]["D_kill"] for t in names],
                color=["#1a6faa" if not t.startswith("a_mag") else "#d62728"
                       for t in names])
        ax1.axhline(F1_GKILL, color="k", ls=":", lw=1)
        for t, x in zip(names, xs):
            ax1.text(x, arm_out[t]["D_kill"], f"{arm_out[t]['D_kill']:.3f}",
                     ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(list(xs))
        ax1.set_xticklabels(names, rotation=30, fontsize=7)
        ax1.set_ylabel("densified D_kill")
        ax1.set_title(f"ladder kills ({PF} primary; report-only; "
                      f"g-ray kill {g_kill_pf} ref line)")
        f1_map = {"a_topk10": "topk10", "a_topk50": "topk50",
                  "a_raw": "raw", "a_sign": "sign_path"}
        both = [(t, arm_out[t]["ratio_vs_g_kill"], F1_RATIOS[f1_map[t]])
                for t in f1_map if arm_out[t]["ratio_vs_g_kill"] is not None]
        ax2.scatter([f1_map[t] for t, _, _ in both],
                    [r for _, r, _ in both], color="#1a6faa",
                    label="e193b (fresh root)")
        ax2.scatter([f1_map[t] for t, _, _ in both],
                    [fr for _, _, fr in both], color="#c27d16",
                    label="organism 1 (f1 anchors)")
        ax2.set_ylabel("r = D_kill / own g-kill")
        ax2.legend(fontsize=8)
        ax2.set_title("ratio form (report-only)")
        fig.tight_layout()
        fig.savefig(rd / "e193b_ladder.png", dpi=130)
        plt.close(fig)

    def fig_inspan():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
        for ax, f in zip(axes, (FACT1, FACT2)):
            fld = f"gm_{f}"
            for sd_k, v in inspan_state.items():
                ax.plot([x["D"] for x in v["rows"]],
                        [x[fld] for x in v["rows"]], "-o", ms=2.5,
                        lw=1.2, label=f"in-span {sd_k}")
            ax.plot([x["D"] for x in rays[0]["rows"]],
                    [x[fld] for x in rays[0]["rows"]], "--", color="k",
                    lw=1.2, label="g-ray (ref)")
            ax.axhline(SHUT_BAR, color="k", ls=":", lw=1)
            rg = inspan_range[f]["range"]
            if rg:
                ax.axvspan(rg[0], rg[1], color="#9467bd", alpha=0.15)
                sp = inspan_range[f]["spread_x"]
                sp_txt = f"{sp:.2f}x" if sp is not None else "n/a (1 draw)"
                ax.set_title(f"{f}: in-span kill range [{rg[0]:.2f}, "
                             f"{rg[1]:.2f}] (spread "
                             f"{sp_txt}) vs g-kill "
                             f"{per_fact[f]['g_kill']}")
            else:
                ax.set_title(f"{f}: in-span kills unresolved")
            ax.set_xscale("log")
            ax.set_xlabel("D (L2)")
        axes[0].set_ylabel("mean p(name onset)")
        axes[0].legend(fontsize=7)
        fig.suptitle("E193B — THE IN-SPAN 3-DRAW RANGE (the missing "
                     "number; e131's suppressed spread was 3x)", fontsize=9)
        fig.tight_layout()
        fig.savefig(rd / "e193b_inspan_range.png", dpi=130)
        plt.close(fig)

    def fig_pump():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, f in zip(axes, (FACT1, FACT2)):
            root_p = root_cells[f][f"g{primary_of[f]['ruler_j']:+d}"]
            for r, s in zip(rays, summaries):
                prof = s["pump"][f]["profile"]
                color = ("#1a6faa" if r["key"] == "R1_G" else
                         ("#c27d16" if r["key"] == "R2_SIGN" else "#777777"))
                ax.plot([p["D"] for p in prof],
                        [p["gm"] - root_p for p in prof], "-o", ms=3,
                        color=color, lw=1.3,
                        label=r["label"].split(" (")[0])
            ax.axhline(PUMP_MARGIN, color="k", ls=":", lw=1)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{f}: max rise "
                         f"{per_fact[f]['pump_g_max_rise']:+.4f} -> ridge "
                         + ("PRESENT" if per_fact[f]["pump_g_ridge"]
                            else "ABSENT"))
            ax.set_xlabel("D (small-D band)")
        axes[0].set_ylabel("rise over root read")
        axes[0].legend(fontsize=7)
        fig.suptitle("E193B — PUMP-PER-FACT (no bar): organism 1 +0.0449 "
                     "RIDGE; organism 2 (e193) -0.0788 ABSENT — the "
                     "ridge-vs-cliff thesis", fontsize=9)
        fig.tight_layout()
        fig.savefig(rd / "e193b_pump_per_fact.png", dpi=130)
        plt.close(fig)

    for fn in (fig_terrain, fig_ladder, fig_inspan, fig_pump):
        try:
            fn()
            log(f"figure {fn.__name__} written")
        except Exception as e:                                  # noqa: BLE001
            log(f"figure {fn.__name__} FAILED: {e}")

    # =====================================================================
    # final metrics
    # =====================================================================
    metrics = {
        "experiment": "e193b_two_fact",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown complete (nothing adjudicated)"
                   if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all "
                   "PARTIAL progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "question": ("the R60-critic's forced replicate, cell 2: does the "
                     "terrain's order + kill-D windows hold for BOTH facts "
                     "on a FRESH root at organism-1's EXACT architecture "
                     "(6L/6H/192d/256-ctx, 2.74M) — the fact axis and the "
                     "root-draw axis varied for the first time, "
                     "architecture pinned?"),
        "organism": {
            "fresh": True,
            "root_ckpt": ck_cons.name, "root_md5": root_md5,
            "root_meta": root_meta,
            "architecture": "6L/6H/192d/256-ctx TinyGPT (2,739,072) — "
                            "organism 1's EXACT architecture "
                            "(architecture-pinned; absolute Ds same-currency)",
            "stages": {"base": stage1,
                       "installs": {f: {k: v for k, v in
                                        install_recs[f].items()
                                        if k != "traj"}
                                    for f in (FACT1, FACT2)},
                       "consolidation": stage4},
            "facts": {FACT1: "the standard fact (organisms 1/2's name); "
                             "sites host_occ[:60] (SPLICE_RNG 24301)",
                      FACT2: "e154's nonce; DISJOINT sites host_occ[90:150] "
                             "(same shuffle order); no held pool"},
            "step_l2_measured": STEP_L2,
            "step_l2_note": "THIS root's measured AdamW step-1 L2 at t=0 "
                            "(never ported; organism 1: 1.6543, organism "
                            "2: 0.9164)",
        },
        "rulers": {
            "primary": {f: {"battery": f"install-60 g{primary_of[f]['ruler_j']:+d}",
                            "root_read": root_cells[f][
                                f"g{primary_of[f]['ruler_j']:+d}"],
                            "rule": "g-12 (e192-verbatim); mechanical "
                                    "fallback to the max root read among "
                                    "{g-12,g-4,g0,g+12} iff g-12 < 0.27 "
                                    "(frozen before compute)"}
                        for f in (FACT1, FACT2)},
            "root_dial_seven_geos": root_cells,
            "ce_r": root_ce_r,
        },
        "cell": {
            "rays": [{"key": r["key"], "label": r["label"],
                      "provenance": r["prov"], "u_md5": hashlib.md5(
                          r["u"].numpy().tobytes()).hexdigest()}
                     for r in rays],
            "D_grid": list(D_GRID),
            "static_jump": "theta_D = theta_0 - D * u_ray; perturb-and-eval; "
                           "NO optimizer (e191/e192/e193 machinery verbatim)",
            "ladder_arms_report_only": [
                {"tag": t, "desc": ARM_DESC[t], "per_step_L2": STEP_L2}
                for t in ARM_ORDER],
            "rider": f"{RIDER_N} pinned steps of L2 "
                     f"{STEP_L2 / RIDER_N:.6f} along the FROZEN g-ray "
                     "(report-only)",
            "inspan": {"n_draws": len(INSPAN_SEEDS), "span": span_meta},
            "breaker": breaker,
            "input_seed": FREEZE_SEED,
            "measure_light": "both facts' primaries + 4 co-ruler batteries "
                             "+ CE_R at every grid point; every-step "
                             "primary-fact read on the ladder; densified "
                             "kill brackets (opt1c)",
            "random_seed_registry": {
                "seeds_used": {"base": SEED_BASE,
                               "installs": SEED_INSTALL,
                               "consolidation": SEED_CONS,
                               "stream": FREEZE_SEED,
                               "gaussian_rays": list(RAND_SEEDS),
                               "inspan_draws": list(INSPAN_SEEDS),
                               "magshuffle": MAGSHUF_SEED},
                "no_collisions": "checked vs the e193 registry note + repo "
                                 "grep (5301/5311/5312/11901-3/11911-3/"
                                 "11921 fresh); recipe seeds 10901/10902/"
                                 "170/1337/24301/26502 are VERBATIM uses"},
        },
        "dual_currency": {
            "N_param": n_params, "rms_denominator": RMS_DEN,
            "convention": "per-coordinate RMS = D / sqrt(N) on every row "
                          "and figure axis",
            "organism1_context": {"N_param": 2_739_072,
                                  "rms_denominator": 1655.014,
                                  "note": "SAME architecture — absolute Ds "
                                          "ARE same-currency with organism "
                                          "1 (the point of this cell); "
                                          "organism 2 (873k, denom 934.59) "
                                          "is NOT"},
        },
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_GEO": G_GEO, "G_GATE0": G_GATE0,
                  "G_ROOTSTRENGTH": G_ROOTSTRENGTH, "G_T0": G_T0,
                  "G_STREAM": G_STREAM, "G_INPUTS": G_INPUTS},
        "profiles": {r["key"]: r["rows"] for r in rays},
        "ray_summaries": summaries,
        "pump_read": pump_read,
        "pump_per_fact_verdict": pump_verdict,
        "rider": rider_pattern,
        "ladder": {"arms": arm_out, "report_only": True,
                   "f1_anchors_absolute": F1_LADDER,
                   "f1_anchors_ratios": F1_RATIOS,
                   "e193_organism2_ratios": e193m["adjudication"]["bars"][
                       "FRONT_REPLICATES"]["ratios"]},
        "inspan_range": inspan_range,
        "ce_r_canary": ce_canary,
        "references": {
            "organism1": {"terrain": "e192 (g 0.92 < sign 2.5 < random "
                                     ">4.0), committed",
                          "ladder": "opt2 (topk10 0.9066 < topk50 0.9203 "
                                    "~= raw 0.9203 < sign 1.7496 < A0 "
                                    "2.4893), committed",
                          "pump": "e191 (+0.0449 g-ray ridge), committed"},
            "organism2": {"terrain": "e193 (g 0.20 < sign 0.58 < random "
                                     "unresolved; ratios 1.108/1.117/1.119/"
                                     "2.629), committed",
                          "pump": "e193 (max rise -0.0788, ridge ABSENT), "
                                  "committed",
                          "step_l2": 0.9164195656776428},
            "critic": "scratch/r60_critic.md (Attack 1: the missing in-span "
                      "spread; Attack 2 rank-1: the magnitude-shuffle "
                      "breaker; THE FORCED EXPERIMENT: this cell)",
        },
        "adjudication": {
            "registered_bars_verbatim": REGISTERED_BARS,
            "headline": headline, "clause": clause,
            "per_fact": per_fact,
            "composite": {
                "terrain_replicates_fires": bool(all_terrain and not SMOKE),
                "order_scrambles_fires": bool(any_scrambles and not SMOKE),
                "pump_per_fact": pump_verdict,
                "rule": "ORDER-SCRAMBLES if ANY fact scrambles (the bar's "
                        "'any' letter); else TERRAIN-REPLICATES iff BOTH "
                        "facts replicate; else GRADED — per-fact clauses "
                        "always reported",
            },
        },
        "honesty_reflex": {
            "draws": "the fresh root is ONE draw (n=1 root at n=1 "
                     "architecture); each fact is n=1 (two facts in ONE "
                     "organism is fact-varied, not fact-replicated); the "
                     "composite 'n=3 organisms' of the bar counts e193's "
                     "organism 2 (different architecture — its absolute "
                     "Ds are not same-currency)",
            "g_cons": {f: primary_of[f]["g_consolidated"]
                       for f in (FACT1, FACT2)},
            "ruler": "each fact's kills reported on its primary AND all "
                     "four per-grid geometries; the fallback was mechanical "
                     "and frozen; no ruler shopping is possible after the "
                     "fact",
            "what_arms_guarantee": "NOTHING — the openness is the point "
                                   "(stated before compute, Rule 12)",
            "logits_caution": "kill readings are behavior (battery mean p), "
                              "not logits; the honesty-reflex question "
                              "(do logits alone predict it) is answered by "
                              "construction: the reads ARE the behavior",
        },
        "trims": [],
        "deviations": deviations,
        "timing": {"elapsed_s": round(time.time() - T0, 1)},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log(f"DONE -> {rd / 'metrics.json'} ({metrics['timing']['elapsed_s']}s)")


if __name__ == "__main__":
    main()
