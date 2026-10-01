"""E198 — THE ARCHITECTURE-VS-BIOGRAPHY CUT (T160's named discriminating
cell: the flight map on e193b's MIRABEL root).

WHY (T160, verbatim): "ARCHITECTURE/SIZE — organism 1 is 2.74M 6L; this
organism 873k 4L; e193b's fresh root was 2.74M and its MIRABEL fact
replicated the whole TERRAIN — but the FLIGHT map was never read there: THE
DISCRIMINATING CUT IS NAMED (the flight map on e193b's MIRABEL root: if the
concentration appears, architecture carries it; if not, organism 1's
specific biography)." The flight concentration (org1: u1/u0 ratio 0.17) is
absent on the 873k organism — DEAD lineage (e196: ratio 2.58) AND ALIVE
half-step lineage (e197: ratio 3.17) — aliveness necessary not sufficient.
THIS CELL reads the flight map on the one organism that holds everything
else organism 1 had: e193b's fresh 2.74M 6L root (organism-1's EXACT
architecture, fresh draw) whose MIRABEL fact passed its gate and replicated
the whole terrain (T155) — AND whose theta_1 is ALIVE on MIRABEL's ruler
(committed 0.3673 at the sign walk's step-1 endpoint), so the dead-anchor
asymmetry that disqualified e196's flight read is ABSENT HERE. If the
concentration appears on this root, the 2.74M architecture carries the
dynamics; if not, organism 1's specific lineage biography does.

THE ORGANISM: runs/checkpoints/e193b_root.pt — e193b's FRESH two-fact
consolidated root (6L/6H/192d/256-ctx TinyGPT, 2,739,072 params —
organism-1's EXACT architecture; flat md5 9113c7593dbac14fb57d47f0b96c1587),
its committed static-direction checkpoint runs/checkpoints/
e193b_static_dir_u.pt (g-ray u md5 159263074c9e8a48fc898bca48017f47) and
its committed wash step-1 checkpoint runs/checkpoints/e193b_wash_s1.pt (the
G_T0/G_S1CK anchor). THE FACT: MIRABEL — e154's nonce, install sites
host_occ[90:150] under SPLICE_RNG 24301 (mix {FLORIZEL:16, ELIZABETH:44}
reported ungated), PRIMARY RULER = its install-60 battery at g-12 (root
read 0.6236 committed; the e192-verbatim geometry, alive at D=0 — no
fallback rule triggers).

THE CELL (eval-only CPU, minutes): on e193b's MIRABEL root (its gates
verbatim), e193b's own stream/step convention (seed-10902 neutral stream,
anchor bank seed 170 rejecting FLORIZEL/ELIZABETH/ZEPH/MIRABEL, measured
STEP_L2 1.6544127464294434 — never ported, re-measured and gated):
  (1) its wash walk t=0..t=3 — the k=1 sign walk (e193b's a_sign arm
      VERBATIM arithmetic: draw order, post-clip gradients, sign_update
      with fp64 support norm, every-step MIRABEL-primary read), 3 steps on
      the licensed stream, MIRABEL's theta_1 read ALIVE vs the bar (the
      committed step-1 row says 0.3673 — re-read and gated G_REPRO);
  (2) THE FLIGHT MAP: static graded jumps along ITS u0 = sign(g_0) (=
      e193b's committed R2_SIGN ray, md5-gated), u1 = sign(g_1) (THE
      FLIGHT RAY — the t=1 fresh front, read at an ALIVE theta_1), u2 =
      sign(g_2) (the t=2 front) from its root — D grid 0.05..3.00 step
      0.05 (e195's grid VERBATIM), dual currency (D + per-coordinate RMS
      / 1655.014 — same denominator as organism 1), MIRABEL's rulers
      (primary g-12 + co-rulers g-4/g0/g+12 on every row);
  (3) the theta_1 panel — the same three families from theta_1 (the
      walk's step-1 endpoint, ALIVE 0.3673 — the first LIVE theta_1 panel
      since organism 1's; it rides verbatim and never adjudicates);
  (4) the recomputation check — its k=1 sign path (the walk's own
      densified kill-D on MIRABEL's ruler) vs its static u0 edge.
RATIO-FORM WITHIN-ORGANISM THROUGHOUT (T153's biography lesson): every
clause is a ratio to THIS organism's OWN u0 edge; organism 1 (e195),
organism 2 (e196) and the alive-window lineage (e197) ride as committed
side-by-sides, never as bars.

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - ARCHITECTURE-CARRIES: "fires if MIRABEL's flight ray concentrates (u1
    ratio < 0.7 vs its OWN u0 edge) and/or its recomputation bonus appears
    (walk ratio < 0.85) — the 2.74M architecture carries the dynamics; the
    fork resolves to architecture."
  - BIOGRAPHY-CARRIES: "fires if MIRABEL (2.74M, terrain-replicating,
    alive at theta_1) shows NEITHER effect — organism 1's specific lineage
    biography carries the flight; the architecture suspect cleared; the
    remaining candidates named."
  - GRADED: "any mix — the profiles verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * "kills at D" = the family profile's FIRST MIRABEL-primary (g-12
    install-60 battery mean p(M)) <= 0.27 downcrossing on the static grid,
    linear-in-D interpolated (e192/e194/e195/e196's convention); None = no
    downcrossing within [0, 3.0] (the family never kills in the window —
    any bar clause needing that number resolves as stated below, reported
    verbatim).
  * "u1 ratio" = D_kill(root, u1) / D_kill(root, u0), both present;
    the FLIGHT CLAUSE FIRES iff the ratio < 0.70; the flight clause is
    ABSENT iff the u0 edge is present and (the ratio is >= 0.70 or the u1
    kill is None — a flight ray that does not even kill within [0, 3.0]
    cannot be concentrating below an edge <= 3.0/0.70); UNRESOLVED iff the
    u0 edge itself is None (nothing resolves; GRADED).
  * "walk ratio" = D_kill(walk) / D_kill(root, u0): the walk's own first
    MIRABEL-primary downcrossing along its t=0..t=3 path, densified at the
    killing step (opt1c); the WALK CLAUSE FIRES iff the ratio < 0.85; the
    walk clause is ABSENT iff the u0 edge is present and (the ratio is
    >= 0.85 or the walk is ALIVE at t=3 past the static edge — cum_D(t=3)
    > D_kill(root, u0): the path survived its own static edge through the
    full window, e197's "path safer than its ray" texture — the bonus
    cannot be hiding past the edge inside this cell's window); UNRESOLVED
    iff the u0 edge is None.
  * ARCHITECTURE-CARRIES fires iff (flight FIRES) OR (walk FIRES).
  * BIOGRAPHY-CARRIES fires iff (flight ABSENT) AND (walk ABSENT).
  * GRADED covers every other combination (including any UNRESOLVED
    clause) — the profiles verbatim.
  * composite order frozen: ARCHITECTURE-CARRIES -> BIOGRAPHY-CARRIES ->
    GRADED (first that fires is the verdict; every bar's fires flag
    reported verbatim).
  * rays: u_t = sign(g_t)/||sign(g_t)|| (e192's fp32-norm construction)
    with g_t the licensed stream's t=0/1/2 post-clip batch gradients along
    the rebuilt k=1 sign walk (seed 10902, e193b's draw order); anchors =
    theta_0 (the e193b root) and theta_1 (the walk's step-1 endpoint,
    gated G_REPRO; ALIVE 0.3673 committed).
  * the theta_1 panel (3 families from theta_1) rides verbatim and NEVER
    adjudicates (e195/e196's convention — cross-panel D comparisons are
    not bars).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): e193b's
provenance gates REUSED VERBATIM (G_NAMEFREE — corpus ZEPH 0 AND MIRABEL 0;
G_SPLICE — ZEPHYRA install mix {FLORIZEL:19, ELIZABETH:41} gated, MIRABEL's
{16,44} reported; G_BATTERY — both facts' install-60 batteries at the seven
read geometries, shapes 60 x (130 +- j); G_ANCHOR — the seed-170 neutral
bank BIT-MATCHES e185's stored 16 starts; G_ROOT — the root's seven-geometry
dial reproduces e193b's committed cells BIT-TIGHT + flat md5; G_T0 — the
step-1 batch md5 vs e185's stored hash + the forward CE vs e193b's committed
a_sign step-1 CE + the fresh CPU AdamW step's L2 vs the committed MEASURED
step L2 (never ported) + the wash_s1 checkpoint cross-anchor; G_STREAM —
seed-10902 stream md5 steps 1..4) PLUS the organism's committed-ray gates:
G_DIRCK (the e193b_static_dir_u.pt checkpoint's u md5 + theta0_md5 vs the
fresh g-ray/root) and G_SIGNRAY (u0 rebuilt md5 == e193b's committed R2_SIGN
md5) PLUS G_REPRO (the rebuilt walk must reproduce e193b's committed a_sign
step-1 row — CE, cum_disp, preclip_gnorm, the ZEPHYRA read, the MIRABEL
primary + all four MIRABEL geometry reads, ce_r — AND its ZEPHYRA kill
bracket D_kill/D_kill_raw bit-class; failure = control failure, abort)
PLUS the profile machinery gates: G_ROOTPROF (the root-u0 family must
reproduce e193b's committed R2_SIGN gm_MIRABEL rows at ALL 27 shared Ds +
the D=STEP_L2 landing vs the committed walked step-1 MIRABEL read, texture
class) and G_TH1PROF (the theta_1 families' D=0 rows must reproduce the
committed walked step-1 MIRABEL read bit-class + the th1-u1 D=STEP_L2
landing vs the walk's theta_2 read). THE DUAL-ESTIMATOR LESSON (T150)
carried: every alignment read states its evaluation point (matched-point AT
the anchor) with the walked-delta and unit-ray forms cross-reported; no
cross-organism cosine comparisons (different instruments). WHAT EACH ARM
GUARANTEES: NOTHING — every family is a static graded jump at lethal scale
from a state that could kill anywhere; the walk could kill at its edge,
before it, or past it; that openness is the point.

REGISTERED PREDICTION (frozen before compute): T160's own fork —
architecture is the RANKED suspect (this root holds organism 1's exact
architecture, a terrain-replicating fact, AND an alive theta_1), so the
prior leans ARCHITECTURE-CARRIES, but the e197 lesson (alive window
necessary NOT sufficient) keeps it genuinely open; either verdict is
informative and no bar moves.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
the GPU is another agent's, never claimed), torch threads 8 (e193b's gate
convention — its committed dial + rows reproduce bit-tight under it),
sequential phases, every phase < 180 s, PROGRESSIVE partial metrics.json
writes after every phase (the outage lesson), n=1 root / n=1 fact, single
stream seed 10902 lineage.

Outputs: runs/e198/{metrics.json, e198_flight_map.png,
e198_three_organism.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e198_flight_arch.py    (E198_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e196's convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e193b's gate convention (bit-tightness)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E198_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e198 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

FACT1 = "ZEPHYRA"                  # the standard fact (gates + the G_REPRO co-read)
FACT2 = "MIRABEL"                  # THE FACT (e154's nonce; the ruler of this cell)
PRE, POST_CAP = 130, 119           # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                        # training-window block (the e098 line's own convention)
ROOT_CK = "e193b_root.pt"          # THE FRESH TWO-FACT CONSOLIDATED ROOT (e193b's, committed)
S1_CK = "e193b_wash_s1.pt"         # e193b's committed wash step-1 checkpoint (G_T0/G_S1CK)
DIR_CK = "e193b_static_dir_u.pt"   # e193b's committed g-ray direction checkpoint
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E193B_METRICS = E43.REPO / "runs" / "e193b" / "metrics.json"
E195_METRICS = E43.REPO / "runs" / "e195" / "metrics.json"
E196_METRICS = E43.REPO / "runs" / "e196" / "metrics.json"
E197_METRICS = E43.REPO / "runs" / "e197" / "metrics.json"

# ---- the organism-1-architecture net (6L/6H/192d/256-ctx, 2,739,072 params) ----
CFG1 = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
N_PARAM = 2_739_072
RMS_DENOM = 1655.0141993348577      # sqrt(2,739,072) — the per-coordinate RMS currency

# ---- rulers (e193b's, frozen in ITS cell; carried verbatim) -----------------------
RULER_J = -12                       # PRIMARY: MIRABEL's install-60 g-12 (e192-verbatim; alive 0.6236)
CO_RULERS_J = (-4, 0, 12)           # e193b's per-grid-point geometries (the other three)
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries (gates)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193/e193b)
LR_ADAMW = 1e-3                   # the t=0 recipe (e193b's wash; the ray arms have NO lr)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (e195's grid VERBATIM)
CE_EVERY = 0.2                    # ce_r at D in {0.2, 0.4, ..., 3.0} (e195's decimation)
WALK_STEPS = 3                    # the walk t=0..t=3 (states theta_0..theta_3)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # opt1c's along-path kill-bracket densification
FLIGHT_RATIO_BAR = 0.70           # frozen: "u1 ratio < 0.7 vs its OWN u0 edge"
WALK_RATIO_BAR = 0.85             # frozen: "walk ratio < 0.85"
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_STATIC_TOL = 1e-3               # e193b's committed R2 rows (same code path; expect ~0)
G_LANDING_TOL = 2e-3              # fp32-norm ray landing vs walked fp64-norm step (e195's class)
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    CE_EVERY = None

# ---- e193b's committed lineage reads (hard-bound at load; the gates' targets) -----
E193B_ROOT_MD5 = "9113c7593dbac14fb57d47f0b96c1587"
E193B_STEP_L2 = 1.6544127464294434          # THIS root's measured AdamW step-1 L2 (never ported)
E193B_DIAL = {                                # the committed seven-geometry root dial + ce_r
    "ZEPHYRA": {"g-12": 0.14864006638526917, "g-8": 0.33425578474998474,
                "g-4": 0.4511719346046448, "g+0": 0.4618377089500427,
                "g+4": 0.38778820633888245, "g+8": 0.4226633310317993,
                "g+12": 0.17708639800548553},
    "MIRABEL": {"g-12": 0.6235591173171997, "g-8": 0.7508792281150818,
                "g-4": 0.7817148566246033, "g+0": 0.42963913083076477,
                "g+4": 0.8612411618232727, "g+8": 0.7882601022720337,
                "g+12": 0.6515362858772278},
    "ce_r": 1.7411779165267944,
}
E193B_SPLICE_MIX = {                            # e193b's committed splice gate values
    "ZEPHYRA": {"FLORIZEL": 19, "ELIZABETH": 41},
    "MIRABEL": {"FLORIZEL": 16, "ELIZABETH": 44},
}
E193B_ASIGN_S1 = {                              # e193b's committed a_sign step-1 row (G_REPRO)
    "ce_batch": 1.531902551651001,
    "cum_disp": 1.6568037271499634,
    "step_disp": 1.6568037271499634,
    "preclip_gnorm": 0.9484267234802246,
    "gm_ZEPHYRA": 0.052623070776462555,          # its ladder ruler (ZEPHYRA g+0)
    "frac_argmax_z": 0.0,
    "ce_r": 2.4445571899414062,
    "gm_MIRABEL": 0.3672811686992645,            # THE ALIVE THETA_1 READ (g-12)
    "g-12_MIRABEL": 0.3672811686992645,
    "g-4_MIRABEL": 0.5463535785675049,
    "g+0_MIRABEL": 0.16293005645275116,
    "g+12_MIRABEL": 0.1902925670146942,
}
E193B_ASIGN_STOP = {                            # e193b's committed a_sign stop (ZEPHYRA bracket)
    "kind": "kill", "step": 1,
    "D_kill_raw": 0.7767010305763323,
    "D_kill": 0.8969455194321004,
}
E193B_KILLS = {"g": 0.8, "sign": 2.0}           # e193b's committed grid kills (MIRABEL primary)
E193B_UG_MD5 = "159263074c9e8a48fc898bca48017f47"   # its committed g-ray direction
E193B_USIGN_MD5 = "fbb878c1926c165ebdb54eef1259ab65"  # its committed R2_SIGN direction
E193B_R2_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.4, 0.5, 0.58, 0.66,
                 0.74, 0.8, 0.87, 0.92, 1.0, 1.08, 1.17, 1.33, 1.5, 1.75, 2.0,
                 2.25, 2.5, 3.0, 3.5, 4.0]      # its 27-point D grid (G_ROOTPROF shared Ds <= 3.0)
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
}

# ---- the committed side-by-sides (ride, never adjudicate) ------------------------
E195_ROOT_DKILLS = {                        # organism 1 (e131 root 2.74M; alive theta_1 0.679)
    "root_u0": 2.269916581032063,
    "root_u1": 0.3875040789853107,
    "root_u2": 0.8979436419840907,
}
E195_WALK = {"D_kill": 1.749640490742179,   # opt2 a_sign's committed path kill
             "static_edge": 2.269916581032063}
E196_ROOT_DKILLS = {                        # organism 2 (e157_f2 root 873k; dead theta_1 0.0068)
    "root_u0": 0.5252331635450007,
    "root_u1": 1.3575583149916892,
    "root_u2": 0.46025487385682545,
}
E196_WALK = {"D_kill": 0.5257438700572483,  # e193's committed a_sign kill (dead at step 1)
             "static_edge": 0.5252331635450007}
E197_ALIVE = {                              # org2's half-step ALIVE lineage (theta_1 0.4192)
    "u1_D_kill": 1.6644735674638536, "u0_edge": 0.5252331635450007,
    "walk_D_kill": 0.838723200837794, "theta1_gm": 0.41916516423225403,
}

REGISTERED_BARS = {
    "ARCHITECTURE_CARRIES": "ARCHITECTURE-CARRIES: \"fires if MIRABEL's "
        "flight ray concentrates (u1 ratio < 0.7 vs its OWN u0 edge) and/or "
        "its recomputation bonus appears (walk ratio < 0.85) — the 2.74M "
        "architecture carries the dynamics; the fork resolves to "
        "architecture.\"",
    "BIOGRAPHY_CARRIES": "BIOGRAPHY-CARRIES: \"fires if MIRABEL (2.74M, "
        "terrain-replicating, alive at theta_1) shows NEITHER effect — "
        "organism 1's specific lineage biography carries the flight; the "
        "architecture suspect cleared; the remaining candidates named.\"",
    "GRADED": "GRADED: \"any mix — the profiles verbatim.\"",
    "operationalizations": "static graded jumps theta_D = anchor - D*u for "
        "u in {u0, u1, u2} = sign(g_t)/||sign(g_t)|| (e192's fp32-norm "
        "construction) with g_t the licensed stream's t=0/1/2 post-clip "
        "batch gradients along the rebuilt k=1 sign walk (e193b's a_sign, "
        "seed 10902, STEP_L2 1.6544 measured, gated G_REPRO vs e193b's "
        "committed step-1 row + ZEPHYRA kill bracket); anchors = theta_0 "
        "(the e193b root, gated G_ROOT/G_DIRCK) and theta_1 (the walk's "
        "step-1 endpoint, committed MIRABEL read 0.3673 — ALIVE at the "
        "anchor, so the flight ray's gradient is a LIVE mid-flight read); "
        "grid D in {0.05..3.00 step 0.05} + D=0 anchor rows + the "
        "D=STEP_L2 landing gate points; ruler = MIRABEL's install-60 g-12 "
        "battery, SHUT 0.27 (co-rulers g-4/g0/g+12 on every row, never "
        "adjudicated); KILL = the profile's first primary-ruler <= 0.27 "
        "downcrossing, linear-in-D interpolated; 'u1 ratio' = "
        "D_kill(root,u1)/D_kill(root,u0) — the flight clause FIRES iff "
        "< 0.70, is ABSENT iff the u0 edge is present and (ratio >= 0.70 "
        "or the u1 kill is None within [0,3.0]), UNRESOLVED iff the u0 "
        "edge is None; 'walk ratio' = D_kill(walk)/D_kill(root,u0) with "
        "the walk's own densified kill on the MIRABEL ruler — the walk "
        "clause FIRES iff < 0.85, is ABSENT iff the u0 edge is present "
        "and (ratio >= 0.85 or the walk is alive at t=3 past the static "
        "edge), UNRESOLVED iff the u0 edge is None; the theta_1 panel "
        "rides verbatim and NEVER adjudicates; composite order frozen "
        "ARCHITECTURE-CARRIES -> BIOGRAPHY-CARRIES -> GRADED.",
    "registered_prediction": "T160's own fork (architecture the RANKED "
        "suspect — this root holds organism 1's exact architecture, a "
        "terrain-replicating fact, AND an alive theta_1) leans "
        "ARCHITECTURE-CARRIES, but the e197 lesson (the alive window "
        "necessary NOT sufficient) keeps the prior genuinely open; either "
        "verdict is informative; no bar shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE FACT CALL (e193b's, carried verbatim): MIRABEL is this cell's "
    "ruler — the fact that passed its gate and replicated the whole "
    "terrain (T155); its primary g-12 root read 0.6236 is ALIVE (no "
    "fallback rule triggers). ZEPHYRA co-reads only where e193b's "
    "committed row requires it (G_REPRO).",
    "THE ALIVE-THETA_1 DISCLOSURE (this cell's load-bearing asymmetry, "
    "the OPPOSITE of e196's): MIRABEL's theta_1 reads 0.3673 on its ruler "
    "at the sign walk's step-1 endpoint (committed, gated G_REPRO) — THIS "
    "organism's step is 1.6544, BELOW its static sign edge (grid 2.0), so "
    "the one-stepped state is ALIVE: the flight ray u1 is a LIVE "
    "mid-flight front read (organism 1's condition, T158's mechanism "
    "requirement, satisfied at n=1 organism beyond organism 1).",
    "walk_rebuild is lab/e193b_two_fact.py's run_arm (a_sign arm) VERBATIM "
    "arithmetic (its own stream/step convention: draw order, name-free "
    "verify on ZEPH and the FULL nonce MIRABEL, post-clip gradients, "
    "sign_update with fp64 support norm, densified kill bracket) with "
    "ADDITIVE changes only: the every-step readout is MIRABEL's primary "
    "(the ruler), all four MIRABEL geometries per step, a 3-step horizon "
    "with read-only post-kill continuation (e196's precedent, disclosed), "
    "and the g_fronts/endpoints stashes; no arithmetic path changes.",
    "e193b's ladder read ZEPHYRA (its frozen PF) and STOPPED at the step-1 "
    "ZEPHYRA kill; this cell's walk continues steps 2-3 on the same "
    "licensed stream (batches md5-gated vs e185's stored hashes) reading "
    "MIRABEL — the continuation is bit-rebuildable and disclosed; the "
    "ZEPHYRA kill bracket at step 1 is re-derived with e193b's own "
    "arithmetic and gates G_REPRO.",
    "No new checkpoints (e195/e196's convention): the rays and anchors "
    "are bit-rebuildable from the licensed stream + e193b's committed "
    "artifacts; the new direction md5s (u1/u2) are registered in "
    "metrics.json for any follow-on.",
    "ce_r (second currency) decimated to 0.2-multiples + D=0 (e195's "
    "decimation) + e193b's shared R2_SIGN Ds on the root-u0 family; the "
    "g-12 battery is the ruler and is read at EVERY grid point, with the "
    "three co-rulers on every row; dual displacement currency (D and "
    "per-coordinate RMS / 1655.014) on every row.",
    "Light reads only (batteries + ce_r + fact-battery matched-point "
    "cosines in dual form); CPU-ONLY, threads 8 (e193b's gate "
    "convention), n=1 root / n=1 fact, seed lineage 10902.",
    "Smoke mode trims: grid {0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0}, ce at "
    "anchors only (verdict stamped SMOKE; nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e196_flight_replicate.py's instruments (whose own
# provenance is lab/e193_organism_replicate.py / lab/e193b_two_fact.py /
# lab/e192_all_ray_terrain.py via lab/e195_rotated_ray.py — the e176n
# lineage). Copied rather than imported to own the device policy and the
# bit-exact arithmetic.

def load_root(path) -> TinyGPT:
    m = TinyGPT(CFG1)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(Z) at the last position."""
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
    """e065 val_windows verbatim: name-free val-split windows (e193b's)."""
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
    """The fp32 flat parameter vector (net.parameters() order; 2,739,072)."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    """Copy a flat vector back into parameters (the static-jump loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 cosine (the chart's estimator precision)."""
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def sign_update(g: torch.Tensor, step_l2: float):
    """e193b/opt2's sign_update VERBATIM: delta = -step_l2 * sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — the matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ (e195/e196's fact_grad, ported): gradient of the
    PRIMARY battery's mean log p(Z) readout at the eval twin's current
    weights. NEGATIVE cos(-u, grad) convention: a NEGATIVE cos(-u, grad)
    means the jump ray u is ANTI-ALIGNED with the fact's own gradient.
    Consumes no RNG; run on the eval twin."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    F_obj = torch.stack(sums).sum() / ids.shape[0]
    F_obj.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
    net.zero_grad(set_to_none=True)
    return g


# ------------------------------------------------------------------ kill edges

def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    """The densified bracket (opt1c/e193b's convention; this value gates)."""
    pts = [(d0, v0)] + [(r["D"], r["gm"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) + any upcrosses (context).
    A dead anchor (row D=0 already <= bar) has no downcrossing unless the
    profile first resurrects above the bar — both reported verbatim."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm"] <= SHUT_BAR and a["gm"] > SHUT_BAR:
            edge = interp_d_kill(a["gm"], b["gm"], a["D"], b["D"])
            break
    ups = [(rows[i - 1]["D"], rows[i]["D"]) for i in range(1, len(rows))
           if rows[i]["gm"] > SHUT_BAR and rows[i - 1]["gm"] <= SHUT_BAR]
    return edge, ups


# ------------------------------------------------------------------ the walk

def walk_rebuild(net0, anchor_neutral, train_ids, itos, mir_ids, mir_geos,
                 zid_m, zeph_ids, zid_z, theta0, step_l2, root_gm_m,
                 root_gm_z, n_steps=WALK_STEPS):
    """e193b's run_arm (a_sign) VERBATIM arithmetic + ADDITIVE changes:
    MIRABEL-primary every-step readout (the ruler), all four MIRABEL
    geometries per step, a fixed 3-step horizon with a read-only post-kill
    continuation (e196's precedent), and the g_fronts/endpoints stashes.
    The ZEPHYRA bracket at step 1 is re-derived with e193b's own
    densification arithmetic (G_REPRO); the MIRABEL bracket at ITS killing
    step is this cell's walk-kill number (clause 4)."""
    net = copy.deepcopy(net0)
    net.train()
    evl_a = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    step, kill_m, kill_z = 0, None, None
    traj, x_hashes = [], {}
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_flat = theta0
    t0a = time.time()
    while step < n_steps:
        step += 1
        post_kill = kill_m is not None
        aj = torch.randint(16, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt or FACT2 in txt:   # e193b's rule (full nonce)
                zeph += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        if step in E185_XHASH:
            assert x_hashes[step] == E185_XHASH[step], \
                f"step-{step} batch md5 diverged from the licensed stream"
        logits_a, _ = net(x)
        loss = F.cross_entropy(logits_a.reshape(-1, logits_a.shape[-1]),
                               y.reshape(-1))
        net.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        g_t = torch.cat([p.grad.detach().reshape(-1)
                         for p in net.parameters()]).clone()  # post-clip
        delta = sign_update(g_t, step_l2)
        l2dev = abs(float(torch.norm(delta.double())) - step_l2)
        assert l2dev < 1e-5, f"per-step L2 dev {l2dev:.2e}"
        max_l2_dev = max(max_l2_dev, l2dev)
        cur = prev_flat + delta
        load_flat(net, cur)
        cum_disp = float(torch.norm(cur - theta0))
        evl_a.load_state_dict({k_: v.detach().cpu().clone()
                               for k_, v in net.state_dict().items()})
        evl_a.eval()
        gm = battery_cell(evl_a, mir_ids, zid_m)
        row = {"step": step, "post_kill": post_kill,
               "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gm["mean_pz"], "frac_argmax_z": gm["frac_argmax_z"]}
        for j, ids_j in mir_geos.items():          # all four MIRABEL geos
            row[f"g{j:+d}"] = battery_cell(evl_a, ids_j, zid_m)["mean_pz"]
        if step == 1:                              # the G_REPRO co-reads
            row["ce_r"] = ce_fixed_cpu(evl_a, *R_EVAL_XY_GLOBAL)
            row["gm_ZEPHYRA"] = battery_cell(evl_a, zeph_ids, zid_z)["mean_pz"]
        traj.append(row)
        g_fronts.append(g_t.clone())               # ADDITIVE stash
        endpoints[step] = cur.clone()              # ADDITIVE stash
        prev_flat = cur
        log(f"  [a_sign s{step}]{' POSTKILL' if post_kill else ''} "
            f"MIRABEL g-12 {row['gm']:.10f} D {cum_disp:.10f} ce "
            f"{row['ce_batch']:.6f}")
        if step == 1:
            # the ZEPHYRA bracket, e193b's arithmetic VERBATIM (G_REPRO):
            # e193b densified at prev_flat + (f-1)*delta with prev_flat
            # ALREADY advanced to cur — i.e. theta0 + f*delta along the step
            prev_gm_z = root_gm_z
            dens_z = []
            for f_ in DENSIFY_F:
                pt_flat = cur + (f_ - 1.0) * delta
                load_flat(evl_a, pt_flat)
                evl_a.eval()
                dens_z.append({"f": f_,
                               "gm": battery_cell(evl_a, zeph_ids,
                                                  zid_z)["mean_pz"],
                               "D": float(torch.norm(pt_flat - theta0))})
            kill_z = {"kind": "kill", "step": 1,
                      "gm_at_kill": row["gm_ZEPHYRA"],
                      "D_kill_raw": interp_d_kill(prev_gm_z,
                                                  row["gm_ZEPHYRA"],
                                                  0.0, cum_disp),
                      "D_kill": dens_d_kill(prev_gm_z, row["gm_ZEPHYRA"],
                                            0.0, cum_disp, dens_z),
                      "dens": dens_z,
                      "note": "e193b's committed a_sign kill (ZEPHYRA g+0 "
                              "ruler) re-derived — the G_REPRO comparator"}
            log(f"  [a_sign] ZEPHYRA bracket re-derived: D_kill "
                f"{kill_z['D_kill']:.10f} (committed "
                f"{E193B_ASIGN_STOP['D_kill']:.10f})")
            load_flat(evl_a, cur)
            evl_a.eval()
        if kill_m is None and row["gm"] <= SHUT_BAR:
            # the MIRABEL bracket (THIS cell's walk-kill, clause 4):
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm_m, "cum_disp": 0.0}   # e193b's fallback
            dens_m = []
            for f_ in DENSIFY_F:
                pt_flat = prev_flat + (f_ - 1.0) * delta
                load_flat(evl_a, pt_flat)
                evl_a.eval()
                dens_m.append({"f": f_,
                               "gm": battery_cell(evl_a, mir_ids,
                                                  zid_m)["mean_pz"],
                               "D": float(torch.norm(pt_flat - theta0))})
            kill_m = {"kind": "kill", "step": step,
                      "gm_at_kill": row["gm"],
                      "D_kill_raw": interp_d_kill(prev_row["gm"], row["gm"],
                                                  prev_row["cum_disp"],
                                                  cum_disp),
                      "D_kill": dens_d_kill(prev_row["gm"], row["gm"],
                                            prev_row["cum_disp"], cum_disp,
                                            dens_m),
                      "dens": dens_m,
                      "reason": "MIRABEL primary ruler <= SHUT at an "
                                "every-step read"}
            log(f"  [a_sign] MIRABEL KILL at s{step}: D_kill(dens) "
                f"{kill_m['D_kill']:.10f} — continuing read-only to t=3")
            load_flat(evl_a, cur)
            evl_a.eval()
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "kill_mirabel": kill_m, "kill_zephyra": kill_z,
            "x_hashes": x_hashes, "max_l2_dev": max_l2_dev,
            "g_fronts": g_fronts, "endpoints": endpoints,
            "seconds": round(time.time() - t0a, 1)}


# ------------------------------------------------------------------ profiles

def static_profile(evl, anchor_flat, u, dgrid, mir_primary, mir_corulers,
                   zid_m, r_eval_xy, ce_ds):
    """Static graded jumps theta_D = anchor - D*u (e192's placement
    VERBATIM), every point read on MIRABEL's primary ruler + the three
    co-rulers; ce_r (second currency) at the frozen decimated Ds; dual
    displacement currency on every row."""
    rows = []
    for D in dgrid:
        thD = anchor_flat - D * u
        disp_check = float(torch.norm(thD - anchor_flat))
        load_flat(evl, thD)
        evl.eval()
        gm = battery_cell(evl, mir_primary, zid_m)
        row = {"D": float(D), "rms": float(D) / RMS_DENOM,
               "gm": gm["mean_pz"], "frac_argmax_z": gm["frac_argmax_z"],
               "disp_check": disp_check, "disp_dev": abs(disp_check - float(D))}
        for j in mir_corulers:
            row[f"g{j:+d}"] = battery_cell(evl, mir_corulers[j],
                                           zid_m)["mean_pz"]
        if ce_ds is not None and any(abs(D - c) < 1e-9 for c in ce_ds):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ main

R_EVAL_XY_GLOBAL = None   # set in main (the walk's step-1 ce_r co-read)


def main():
    global R_EVAL_XY_GLOBAL
    rd = run_dir("e198_smoke" if SMOKE else "e198")
    log(f"E198 THE ARCHITECTURE-VS-BIOGRAPHY CUT (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), phases "
        f"< 180s, progressive writes, n=1 root / n=1 fact, stream seed "
        f"{FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (E193B_METRICS, E195_METRICS, E196_METRICS, E197_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    e193bm = json.loads(E193B_METRICS.read_text(encoding="utf-8"))
    e195m = json.loads(E195_METRICS.read_text(encoding="utf-8"))
    e196m = json.loads(E196_METRICS.read_text(encoding="utf-8"))
    e197m = json.loads(E197_METRICS.read_text(encoding="utf-8"))
    # hard-bind the committed references (asserts catch committed-file drift)
    assert e193bm["organism"]["root_md5"] == E193B_ROOT_MD5
    assert abs(e193bm["organism"]["step_l2_measured"]
               - E193B_STEP_L2) < 1e-12
    dial = e193bm["rulers"]["root_dial_seven_geos"]
    for f in (FACT1, FACT2):
        for j in READ_GEOS:
            assert abs(dial[f][f"g{j:+d}"] - E193B_DIAL[f][f"g{j:+d}"]) < 1e-12
    assert abs(e193bm["rulers"]["ce_r"] - E193B_DIAL["ce_r"]) < 1e-12
    asign = e193bm["ladder"]["arms"]["a_sign"]
    assert abs(asign["D_kill"] - E193B_ASIGN_STOP["D_kill"]) < 1e-12
    assert abs(asign["D_kill_raw"] - E193B_ASIGN_STOP["D_kill_raw"]) < 1e-12
    # e193b's committed a_sign step-1 row (from its journal — hard-bound here)
    e193b_r2 = e193bm["profiles"]["R2_SIGN"]      # the committed static sign profile
    assert len(e193b_r2) == 27, "e193b R2_SIGN row count drift"
    e195_root = e195m["profiles"]
    for k, v in E195_ROOT_DKILLS.items():
        assert abs(e195_root[k]["D_kill"] - v) < 1e-12, f"e195 {k} drift"
    assert e195m["adjudication"]["verdict"] == "FLEEING-IS-LETHAL"
    e196_adj = e196m["adjudication"]["root_panel"]
    for k, v in E196_ROOT_DKILLS.items():
        assert abs(e196_adj[k] - v) < 1e-12, f"e196 {k} drift"
    assert e196m["adjudication"]["verdict"] == "GRADED"
    e197b = e197m["adjudication"]["bars"]["ALIVE_WINDOW_CAUSES"]["detail"]
    assert abs(e197b["flight_concentration"]["u1_D_kill"]
               - E197_ALIVE["u1_D_kill"]) < 1e-12
    log("parents: e193b (the organism: root md5 + STEP_L2 "
        f"{E193B_STEP_L2:.10f} measured, a_sign step-1 MIRABEL read "
        f"{E193B_ASIGN_S1['gm_MIRABEL']:.4f} ALIVE, ZEPHYRA bracket kill "
        f"{E193B_ASIGN_STOP['D_kill']:.4f} committed, static grid kills g "
        f"{E193B_KILLS['g']} / sign {E193B_KILLS['sign']} on MIRABEL's "
        "primary), e195 (organism 1: u1/u0 0.1707, walk ratio 0.7708), "
        "e196 (organism 2 dead: 2.5847 / 1.0010), e197 (organism 2 alive "
        "half-step: 3.1690 / 1.5969) — loaded COMMITTED, never rerun")

    # ---------------- protocol rebuild (e193b verbatim) -------------------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid_m, zid_z = stoi["M"], stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    corpus_mira = len(E43.find_occ(train_text, FACT2))
    G_NAMEFREE = {"corpus_counts": {"ZEPHYRA": corpus_zeph,
                                    "MIRABEL": corpus_mira},
                  "pass": bool(corpus_zeph == 0 and corpus_mira == 0)}
    assert G_NAMEFREE["pass"], f"nonce leaked: {G_NAMEFREE}"

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = {FACT1: host_occ[:60], FACT2: host_occ[90:150]}
    mix = {f: {"FLORIZEL": sum(1 for _, h in install_occ[f]
                               if h == "FLORIZEL"),
               "ELIZABETH": sum(1 for _, h in install_occ[f]
                                if h == "ELIZABETH")}
           for f in (FACT1, FACT2)}
    G_SPLICE = {"install_mix": mix, "n_filtered_total": len(host_occ),
                "pass": bool(mix[FACT1] == E193B_SPLICE_MIX[FACT1]),
                "note": "ZEPHYRA's mix gate = battery identity (e193b's); "
                        "MIRABEL's mix {16,44} reported ungated (e193b's "
                        "own disclosure, hard-bound here)",
                "mirabel_mix_match_committed":
                    bool(mix[FACT2] == E193B_SPLICE_MIX[FACT2])}
    assert G_SPLICE["pass"] and G_SPLICE["mirabel_mix_match_committed"], \
        f"splice drift {G_SPLICE}"

    bat_ids = {f: {} for f in (FACT1, FACT2)}
    for f in (FACT1, FACT2):
        for j in READ_GEOS:
            cs = [train_text[p - PRE - j: p] for p, _ in install_occ[f]]
            bat_ids[f][j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {f: {str(j): list(bat_ids[f][j].shape)
                       for j in READ_GEOS} for f in (FACT1, FACT2)},
        "pass": bool(all(list(bat_ids[f][j].shape) == [60, PRE + j]
                         for f in (FACT1, FACT2) for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): both facts' install-60 "
                "batteries at the e157 dial's seven read geometries, shapes "
                "60 x (130 +- j) — e193b's gate verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    mir_primary = bat_ids[FACT2][RULER_J]
    mir_corulers = {j: bat_ids[FACT2][j] for j in CO_RULERS_J}
    mir_geos = {j: bat_ids[FACT2][j] for j in (RULER_J,) + CO_RULERS_J}
    zeph_primary_g0 = bat_ids[FACT1][0]       # ZEPHYRA g+0 = e193b's ladder ruler
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    R_EVAL_XY_GLOBAL = r_eval_xy

    # the neutral stream (e170 VERBATIM via e185/e193/e193b)
    arng = random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", FACT2)):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                         "in [s, s+257) — e170 VERBATIM (e193b's rule)"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    # ---------------- the e193b root + G_ROOT (bit vs its committed dial) ------
    net0 = load_root(CKPT_DIR / ROOT_CK)
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    root_meta = E43.jsonable(st_raw.get("meta", {}))
    theta0 = flat_params(net0)
    assert int(theta0.numel()) == N_PARAM, \
        f"params {theta0.numel()} != {N_PARAM}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()

    evl0 = copy.deepcopy(net0)
    root_cells = {f: {f"g{j:+d}": battery_cell(evl0, bat_ids[f][j],
                                               zid_m if f == FACT2 else zid_z
                                               )["mean_pz"]
                      for j in READ_GEOS} for f in (FACT1, FACT2)}
    root_cells["ce_r"] = ce_fixed_cpu(evl0, *r_eval_xy)
    refs = {f: {f"g{j:+d}": E193B_DIAL[f][f"g{j:+d}"] for j in READ_GEOS}
            for f in (FACT1, FACT2)}
    refs["ce_r"] = E193B_DIAL["ce_r"]
    rdiffs = {f: {k: root_cells[f][k] - refs[f][k] for k in refs[f]}
              for f in (FACT1, FACT2)}
    rdiffs["ce_r"] = root_cells["ce_r"] - refs["ce_r"]
    rmax = max(abs(v) for f in (FACT1, FACT2) for v in rdiffs[f].values())
    rmax = max(rmax, abs(rdiffs["ce_r"]))
    G_ROOT = {"cells": E43.jsonable(root_cells), "refs": E43.jsonable(refs),
              "diffs": E43.jsonable(rdiffs), "max_abs_diff": rmax,
              "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
              "bit": bool(rmax < G_BIT_TOL), "flat_md5": root_md5,
              "flat_md5_match_e193b_committed":
                  bool(root_md5 == E193B_ROOT_MD5),
              "pass": bool(rmax < G_FALLBACK_TOL
                           and root_md5 == E193B_ROOT_MD5),
              "note": "e193b's G_ROOT VERBATIM (two-fact form): the root's "
                      "dial reproduces e193b's committed seven-geometry "
                      "cells BIT-TIGHT (threads 8) + the flat md5 matches "
                      "e193b's committed root identity"}
    log(f"G_ROOT (vs e193b committed dial, 15 cells): max|diff| {rmax:.2e}, "
        f"flat md5 {'match' if G_ROOT['flat_md5_match_e193b_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("e193b root gate FAILED vs its committed dial")
    root_gm_m = root_cells[FACT2][f"g{RULER_J:+d}"]
    root_gm_z = root_cells[FACT1]["g+0"]

    # ---------------- G_T0 / G_S1CK (e193b's t=0 gates, verbatim) --------------
    def draw_step1_batch():
        g = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(16, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    x1, y1 = draw_step1_batch()
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(net0)
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

    s1ck = torch.load(CKPT_DIR / S1_CK, map_location="cpu",
                      weights_only=False)
    theta_s1 = s1ck["theta1"]
    d_fresh = theta1_adam - theta0
    d_ck = theta_s1 - theta0
    cos_s1 = cos64(d_fresh, d_ck)
    rel_l2 = abs(float(torch.norm(d_ck)) - disp1) / disp1
    G_T0 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1,
        "ce_batch_committed_e193b_asign": E193B_ASIGN_S1["ce_batch"],
        "d_ce": abs(ce1 - E193B_ASIGN_S1["ce_batch"]),
        "preclip_gnorm_measured": gn1, "clip_binds": bool(gn1 > 1.0),
        "adamw_step1_L2_measured": disp1,
        "committed_e193b_step_l2": E193B_STEP_L2,
        "d_step_l2": abs(disp1 - E193B_STEP_L2),
        "note": "e193b's G_T0 VERBATIM (two-fact form): the step-1 batch "
                "md5 vs e185's stored hash (net-independent); the forward "
                "CE vs e193b's committed a_sign step-1 CE (same batch, "
                "same root); the fresh CPU AdamW step's L2 must equal "
                "e193b's committed MEASURED step L2 (never ported)",
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and abs(ce1 - E193B_ASIGN_S1["ce_batch"]) < G_FALLBACK_TOL
                     and abs(disp1 - E193B_STEP_L2) < G_FALLBACK_TOL),
    }
    log(f"G_T0 (t=0 gate): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; CE |d| "
        f"{G_T0['d_ce']:.2e}; measured L2 {disp1:.10f} vs committed "
        f"{E193B_STEP_L2:.10f} (|d| {G_T0['d_step_l2']:.2e}): "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 gate FAILED — abort (control failure)")
    STEP_L2 = disp1                        # measured, never ported

    G_S1CK = {
        "path": str(CKPT_DIR / S1_CK),
        "cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
        "tol_cos": 0.999, "ckpt_meta": E43.jsonable(s1ck.get("meta", {})),
        "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
        "note": "e193b's committed wash step-1 checkpoint cross-anchor "
                "(e196's G_S1CK form): fp64 cosine > 0.999 + relative L2 "
                "< 5%",
    }
    log(f"G_S1CK: cos64 {cos_s1:.8f}, rel L2 dev {rel_l2:.2e}: "
        + ("PASS" if G_S1CK["pass"] else "FAIL"))
    if not G_S1CK["pass"]:
        raise RuntimeError("wash s1 checkpoint anchor FAILED — abort")

    # the t=0 post-clip gradient: the root's g-ray + static sign ray
    gnet = copy.deepcopy(net0)
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
    u_g_md5 = hashlib.md5(u_g.numpy().tobytes()).hexdigest()
    s_raw = torch.sign(g0)
    n_zero_g = int((g0 == 0).sum())
    u0 = (s_raw / torch.norm(s_raw)).clone()    # e193b's construction verbatim
    u0_md5 = hashlib.md5(u0.numpy().tobytes()).hexdigest()
    del s_raw

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
                "pass": bool(all(stream_ok.values())),
                "note": "e193b's G_STREAM VERBATIM: the seed-10902 stream "
                        "construction (net-independent) md5-matches e185's "
                        "stored hashes"}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..4): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # G_DIRCK: e193b's committed g-ray direction checkpoint (the root identity)
    uck = torch.load(CKPT_DIR / DIR_CK, map_location="cpu", weights_only=False)
    G_DIRCK = {
        "path": str(CKPT_DIR / DIR_CK),
        "meta_gate": {"experiment": uck["meta"].get("experiment"),
                      "match": bool(uck["meta"].get("experiment") == "e193b")},
        "loaded_u_md5": hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest(),
        "fresh_u_md5": u_g_md5,
        "md5_match": bool(hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest() == u_g_md5),
        "theta0_md5_match_root": bool(uck.get("theta0_md5") == root_md5),
        "u_norm_fp32": float(torch.norm(u_g)),
        "note": "e191's direction-checkpoint convention, e193b's file: the "
                "committed u must BE the fresh t=0 g-ray bit-exactly and "
                "the committed theta0_md5 must be THIS root's flat md5",
    }
    G_DIRCK["pass"] = bool(G_DIRCK["meta_gate"]["match"]
                           and G_DIRCK["md5_match"]
                           and G_DIRCK["theta0_md5_match_root"])
    log(f"G_DIRCK (e193b's committed g-ray): md5 "
        + ("match" if G_DIRCK["md5_match"] else "DRIFT")
        + f", theta0_md5 {'match' if G_DIRCK['theta0_md5_match_root'] else 'DRIFT'}: "
        + ("PASS" if G_DIRCK["pass"] else "FAIL"))

    # G_SIGNRAY: u0 rebuilt == e193b's committed R2_SIGN direction
    G_SIGNRAY = {
        "u0_md5": u0_md5, "committed_e193b_md5": E193B_USIGN_MD5,
        "md5_match": bool(u0_md5 == E193B_USIGN_MD5),
        "u_norm_fp32": float(torch.norm(u0)),
        "u_norm_fp64": float(torch.norm(u0.double())),
        "n_zero_g_coords": n_zero_g,
        "note": "u0 = the static sign(g_0) ray rebuilt from the gated t=0 "
                "gradient (e193b's fp32-norm construction VERBATIM); "
                "md5-gated vs e193b's committed R2_SIGN ray",
    }
    G_SIGNRAY["pass"] = bool(G_SIGNRAY["md5_match"])
    log(f"G_SIGNRAY (u0 = static sign ray): md5 "
        + ("match" if G_SIGNRAY["md5_match"] else "DRIFT")
        + f" (fp32 norm {G_SIGNRAY['u_norm_fp32']:.7f}): "
        + ("PASS" if G_SIGNRAY["pass"] else "FAIL"))
    if not (G_DIRCK["pass"] and G_SIGNRAY["pass"]):
        raise RuntimeError("committed-ray gates FAILED — abort")

    log("WHAT THESE ARMS GUARANTEE: NOTHING — the flight concentration "
        "could appear, stay absent, or grade; the walk could kill at its "
        "edge, before it, or past it; that openness is the point.")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e198_flight_arch",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE,
                               "torch": torch.__version__,
                               "threads": torch.get_num_threads()},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    stub["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_ROOT": G_ROOT, "G_T0": G_T0, "G_S1CK": G_S1CK,
                     "G_STREAM": G_STREAM, "G_DIRCK": G_DIRCK,
                     "G_SIGNRAY": G_SIGNRAY}
    write_partial("standard cell gates passed (e193b's, verbatim)")
    log("standard cell gates passed; partial metrics written")

    # =====================================================================
    # PHASE 0 — the wash walk t=0..t=3 (e193b's a_sign + MIRABEL ruler)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the k=1 sign walk t=0..t=3 (e193b's a_sign arithmetic; "
        "MIRABEL primary every step; read-only past any MIRABEL kill)")
    walk = walk_rebuild(net0, anchor_neutral, train_ids, itos,
                        mir_primary, mir_geos, zid_m,
                        zeph_primary_g0, zid_z, theta0, STEP_L2,
                        root_gm_m, root_gm_z)
    assert len(walk["traj"]) == WALK_STEPS, \
        f"expected the {WALK_STEPS}-step walk, got {len(walk['traj'])}"
    s1 = walk["traj"][0]

    # ---- G_REPRO: the rebuilt walk vs e193b's committed a_sign row + bracket
    repro_rows = {
        "s1_ce_batch": (s1["ce_batch"], E193B_ASIGN_S1["ce_batch"]),
        "s1_cum_disp": (s1["cum_disp"], E193B_ASIGN_S1["cum_disp"]),
        "s1_step_disp": (s1["step_disp"], E193B_ASIGN_S1["step_disp"]),
        "s1_preclip_gnorm": (s1["preclip_gnorm"],
                             E193B_ASIGN_S1["preclip_gnorm"]),
        "s1_gm_ZEPHYRA": (s1["gm_ZEPHYRA"], E193B_ASIGN_S1["gm_ZEPHYRA"]),
        "s1_ce_r": (s1["ce_r"], E193B_ASIGN_S1["ce_r"]),
        "s1_gm_MIRABEL": (s1["gm"], E193B_ASIGN_S1["gm_MIRABEL"]),
        "s1_g-4_MIRABEL": (s1["g-4"], E193B_ASIGN_S1["g-4_MIRABEL"]),
        "s1_g+0_MIRABEL": (s1["g+0"], E193B_ASIGN_S1["g+0_MIRABEL"]),
        "s1_g+12_MIRABEL": (s1["g+12"], E193B_ASIGN_S1["g+12_MIRABEL"]),
        "zephyra_D_kill": (walk["kill_zephyra"]["D_kill"],
                           E193B_ASIGN_STOP["D_kill"]),
        "zephyra_D_kill_raw": (walk["kill_zephyra"]["D_kill_raw"],
                               E193B_ASIGN_STOP["D_kill_raw"]),
    }
    G_REPRO = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows.items()},
        "x_hashes_vs_e185": {s_: (walk["x_hashes"][s_] == E185_XHASH[s_])
                             for s_ in (1, 2, 3)},
        "tol": G_REPRO_TOL,
        "max_l2_dev": walk["max_l2_dev"],
        "pass": bool(max(abs(vv[0] - vv[1]) for vv in repro_rows.values())
                     < G_REPRO_TOL
                     and all(walk["x_hashes"][s_] == E185_XHASH[s_]
                             for s_ in (1, 2, 3))),
        "note": "THE REBUILT-TRAJECTORY PROVENANCE GATE (Rule 12): the "
                "k=1 walk must reproduce e193b's committed a_sign step-1 "
                "row (CE, displacements, gnorm, the ZEPHYRA read, ALL "
                "MIRABEL reads, ce_r) AND its ZEPHYRA kill bracket "
                "bit-class, on the licensed stream (batches 1-3 "
                "md5-gated) — failure = control failure, abort",
    }
    log("G_REPRO (vs e193b committed a_sign s1 + bracket): max|diff| "
        f"{max(abs(vv[0] - vv[1]) for vv in repro_rows.values()):.2e}: "
        + ("PASS" if G_REPRO["pass"] else "FAIL"))
    if not G_REPRO["pass"]:
        stub["gates"]["G_REPRO"] = G_REPRO
        write_partial("CONTROL FAILURE — walk provenance gate failed")
        raise RuntimeError("G_REPRO FAILED — abort before any read is believed")
    stub["gates"]["G_REPRO"] = G_REPRO

    # ---- the rays: u1/u2 from the walk's refresh gradients (new; md5s
    #      registered); theta_1 = the walked step-1 endpoint
    g0_w, g1_w, g2_w = walk["g_fronts"][0], walk["g_fronts"][1], \
        walk["g_fronts"][2]
    assert cos64(torch.sign(g0_w), u0) > 1 - 1e-9, "t=0 front != u0"
    u1 = (torch.sign(g1_w) / torch.norm(torch.sign(g1_w))).clone()
    u2 = (torch.sign(g2_w) / torch.norm(torch.sign(g2_w))).clone()
    u1_md5 = hashlib.md5(u1.numpy().tobytes()).hexdigest()
    u2_md5 = hashlib.md5(u2.numpy().tobytes()).hexdigest()
    theta1 = walk["endpoints"][1].clone()
    evl_t1 = copy.deepcopy(net0)
    load_flat(evl_t1, theta1)
    evl_t1.eval()
    t1_gm = battery_cell(evl_t1, mir_primary, zid_m)["mean_pz"]
    s2, s3 = walk["traj"][1], walk["traj"][2]
    G_TH1ANCHOR = {
        "cum_disp": float(torch.norm(theta1 - theta0)),
        "committed_s1_cum_disp": E193B_ASIGN_S1["cum_disp"],
        "gm_measured": t1_gm,
        "gm_committed_s1": E193B_ASIGN_S1["gm_MIRABEL"],
        "gm_diff": abs(t1_gm - E193B_ASIGN_S1["gm_MIRABEL"]),
        "alive_at_anchor": bool(t1_gm > SHUT_BAR),
        "adamw_t1_note": "the AdamW sibling state is cross-anchored in "
                         "G_S1CK (e193b's wash_s1 checkpoint)",
        "tol": G_REPRO_TOL,
        "pass": bool(abs(float(torch.norm(theta1 - theta0))
                         - E193B_ASIGN_S1["cum_disp"]) < G_REPRO_TOL
                     and abs(t1_gm
                             - E193B_ASIGN_S1["gm_MIRABEL"]) < G_REPRO_TOL),
        "note": "theta_1 = the rebuilt walk's step-1 endpoint (G_REPRO); "
                "ALIVE on MIRABEL's primary ruler (0.3673 > 0.27) — the "
                "flight ray's gradient is a LIVE mid-flight read (T158's "
                "mechanism condition satisfied; the OPPOSITE of e196's "
                "dead anchor)",
    }
    log(f"G_TH1ANCHOR (theta_1): cum {G_TH1ANCHOR['cum_disp']:.10f}, "
        f"MIRABEL g-12 {t1_gm:.10f} vs committed (|d| "
        f"{G_TH1ANCHOR['gm_diff']:.2e}), ALIVE at anchor: "
        f"{G_TH1ANCHOR['alive_at_anchor']}: "
        + ("PASS" if G_TH1ANCHOR["pass"] else "FAIL"))
    if not G_TH1ANCHOR["pass"]:
        stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR
        write_partial("CONTROL FAILURE — theta_1 anchor gate failed")
        raise RuntimeError("theta_1 anchor gate FAILED — abort")
    stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR

    theta2_dead = bool(s2["gm"] <= SHUT_BAR)
    ray_geometry = {
        "cos_u0_u1": cos64(u0, u1),
        "cos_u0_u2": cos64(u0, u2),
        "cos_u1_u2": cos64(u1, u2),
        "cos_u0_ug": cos64(u0, u_g),
        "cos_u1_ug": cos64(u1, u_g),
        "cos_u2_ug": cos64(u2, u_g),
        "iso_floor": 1.0 / (N_PARAM ** 0.5),
        "organism1_committed": {"cos_u0_u1": -0.15452721980584394,
                                "cos_u0_u2": 0.03151076362117073,
                                "cos_u1_u2": -0.20298347429184507},
        "note": "the rays' mutual geometry (fp64 cosines); organism 1's "
                "committed e195 geometry rides for the side-by-side "
                "(never a bar)",
    }
    log(f"ray geometry: cos(u0,u1) {ray_geometry['cos_u0_u1']:+.4f}, "
        f"cos(u0,u2) {ray_geometry['cos_u0_u2']:+.4f}, "
        f"cos(u1,u2) {ray_geometry['cos_u1_u2']:+.4f} "
        f"(iso floor {ray_geometry['iso_floor']:.2e})")

    # ---- the dual-estimator matched-point alignment reads (T150's lesson:
    #      every read states its point; the walked-delta form and the
    #      unit-ray form cross-consistency reported = the dual estimator)
    root_prim_grad = fact_grad(evl0, mir_primary, zid_m)   # grad at theta_0
    t1_prim_grad = fact_grad(evl_t1, mir_primary, zid_m)   # grad at theta_1
    dual_d1_root = cos64(sign_update(g0_w, STEP_L2), root_prim_grad)
    dual_d2_t1 = cos64(sign_update(g1_w, STEP_L2), t1_prim_grad)
    unit_d1_root = cos64(-u0, root_prim_grad)
    unit_d2_t1 = cos64(-u1, t1_prim_grad)
    alignment_reads = {
        "evaluation_points": "cos(jump direction, grad of the MIRABEL "
                             "PRIMARY battery readout AT the anchor where "
                             "that step began) — matched-point (T150); the "
                             "jump direction is DESCENT (-u)",
        "root_anchor": {
            "cos_neg_u0": unit_d1_root,
            "cos_neg_u1": cos64(-u1, root_prim_grad),
            "cos_neg_u2": cos64(-u2, root_prim_grad),
            "delta_form_s1": dual_d1_root,
            "dual_form_texture_diff_s1": abs(dual_d1_root - unit_d1_root),
        },
        "theta1_anchor": {
            "cos_neg_u0": cos64(-u0, t1_prim_grad),
            "cos_neg_u1": unit_d2_t1,
            "cos_neg_u2": cos64(-u2, t1_prim_grad),
            "delta_form_s2": dual_d2_t1,
            "dual_form_texture_diff_s2": abs(dual_d2_t1 - unit_d2_t1),
        },
        "organism1_committed": {
            "note": "organism 1's committed e195 alignment reads were on "
                    "ITS g-12 ZEPHYRA ruler (a different fact instrument); "
                    "cross-organism cosine comparisons are NOT made",
        },
        "note": "negative cos(-u, grad) = the ray is ANTI-aligned with the "
                "fact's own gradient at that point (T157's flight twist: "
                "organism 1's flight ray was anti-aligned yet deadliest); "
                "delta-form vs unit-ray form agree to fp32-multiplication "
                "texture (the dual-estimator consistency read)",
    }
    stub["phases_partial"]["0_walk_rebuild"] = E43.jsonable({
        "walk_journal": walk["traj"], "kill_mirabel": walk["kill_mirabel"],
        "kill_zephyra_rederived": walk["kill_zephyra"],
        "x_hashes": walk["x_hashes"], "max_l2_dev": walk["max_l2_dev"],
        "post_kill_disclosure": (
            "the walk runs t=0..t=3 on the licensed stream regardless of "
            "any MIRABEL kill (steps past a kill are flagged post_kill, "
            "read-only, and never adjudicate; only their gradients' SIGNS "
            "are probed as ray directions — e196's precedent). theta_1 is "
            f"ALIVE ({t1_gm:.4f}); theta_2 reads {s2['gm']:.4f} "
            + ("(DEAD — u2 is a post-kill front read, disclosed)"
               if theta2_dead else "(alive — u2 is a live t=2 front)")
            + f"; theta_3 reads {s3['gm']:.4f}."),
        "ray_geometry": ray_geometry, "alignment_reads": alignment_reads,
    })
    write_partial("phase 0 complete (walk rebuilt + provenance gates)")
    rays_meta = [
        {"key": "u0", "label": "sign(g_0) — the original static sign ray",
         "u_md5": u0_md5,
         "provenance": "e193b's committed R2_SIGN construction verbatim "
                       "(md5-gated, G_SIGNRAY)"},
        {"key": "u1", "label": "sign(g_1) — THE FLIGHT RAY (t=1 fresh "
                               "front, read at an ALIVE theta_1)",
         "u_md5": u1_md5,
         "provenance": "the rebuilt walk's step-2 refresh gradient (at "
                       "theta_1, batch 2, post-clip; theta_1 is ALIVE "
                       f"({t1_gm:.4f}) — a LIVE mid-flight front, "
                       "organism 1's condition); gated machinery: G_REPRO "
                       "+ G_TH1ANCHOR"},
        {"key": "u2", "label": "sign(g_2) — the t=2 front (rotation's "
                               "continuation"
                               + ("; POST-KILL read, disclosed)"
                                  if theta2_dead else ")"),
         "u_md5": u2_md5,
         "provenance": "the rebuilt walk's step-3 refresh gradient (at "
                       "theta_2, batch 3); gated machinery: G_REPRO"},
    ]
    anchors_meta = {
        "theta_0": {"label": "e193b's fresh two-fact consolidated root "
                             "(2.74M 6L — organism-1's EXACT architecture)",
                    "gm": root_gm_m, "ce_r": root_cells["ce_r"],
                    "provenance": f"runs/checkpoints/{ROOT_CK}, gated vs "
                                  "e193b's committed dial BIT-TIGHT + flat "
                                  "md5 (G_ROOT/G_DIRCK)"},
        "theta_1": {"label": "the one-stepped state (the sign walk's)",
                    "gm": t1_gm, "alive_at_anchor": True,
                    "cum_disp": G_TH1ANCHOR["cum_disp"],
                    "provenance": "the rebuilt walk's step-1 endpoint "
                                  "(G_REPRO + G_TH1ANCHOR); committed read "
                                  f"{E193B_ASIGN_S1['gm_MIRABEL']:.4f} — "
                                  "ALIVE (organism 1's 0.679 was alive; "
                                  "organism 2's 0.0068 was dead)"},
    }

    # =====================================================================
    # PHASE A — the ray families (root panel + theta_1 panel)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE A — six families x {len(D_GRID)}+1 grid points (+ the "
        f"D=STEP_L2 landing gate points)")
    evp = copy.deepcopy(net0)
    ce_ds_common = ([0.0] + [round(CE_EVERY * i, 2)
                             for i in range(1, int(3.0 / CE_EVERY) + 1)]
                    if CE_EVERY else [0.0])
    shared_e193b_Ds = [d for d in E193B_R2_GRID if d <= 3.0]

    def build_family(key, anchor_key, u, extra_ds):
        base = [0.0] + list(D_GRID)
        ds = base + [d for d in extra_ds
                     if not any(abs(d - g) < 1e-9 for g in base)]
        return {"key": key, "anchor_key": anchor_key, "u": u,
                "dgrid": sorted(set(round(d, 10) for d in ds))}

    fam_root_u0 = build_family("root_u0", "theta_0", u0,
                               [STEP_L2] + shared_e193b_Ds)
    fam_root_u0["ce_ds"] = sorted(set(ce_ds_common + shared_e193b_Ds))
    fam_root_u1 = build_family("root_u1", "theta_0", u1, [])
    fam_root_u1["ce_ds"] = list(ce_ds_common)
    fam_root_u2 = build_family("root_u2", "theta_0", u2, [])
    fam_root_u2["ce_ds"] = list(ce_ds_common)
    fam_th1_u0 = build_family("th1_u0", "theta_1", u0, [])
    fam_th1_u0["ce_ds"] = list(ce_ds_common)
    fam_th1_u1 = build_family("th1_u1", "theta_1", u1, [STEP_L2])
    fam_th1_u1["ce_ds"] = list(ce_ds_common)
    fam_th1_u2 = build_family("th1_u2", "theta_1", u2, [])
    fam_th1_u2["ce_ds"] = list(ce_ds_common)
    fam_order = [fam_root_u0, fam_root_u1, fam_root_u2,
                 fam_th1_u0, fam_th1_u1, fam_th1_u2]
    anchors = {"theta_0": theta0, "theta_1": theta1}

    families = {}
    for fam in fam_order:
        tA = time.time()
        rows = static_profile(evp, anchors[fam["anchor_key"]], fam["u"],
                              fam["dgrid"], mir_primary, mir_corulers,
                              zid_m, r_eval_xy, fam["ce_ds"])
        edge, ups = profile_d_kill(rows)
        families[fam["key"]] = {
            "key": fam["key"], "anchor": fam["anchor_key"],
            "ray": fam["key"].split("_")[1],
            "label": next(r["label"] for r in rays_meta
                          if r["key"] == fam["key"].split("_")[1]),
            "u_md5": next(r["u_md5"] for r in rays_meta
                          if r["key"] == fam["key"].split("_")[1]),
            "placement": "anchor - D*u (e192's static placement verbatim)",
            "grid": fam["dgrid"], "n_points": len(rows),
            "gate_points": {}, "rows": rows, "D_kill": edge,
            "any_upcross": ups,
            "anchor_dead": bool(rows[0]["gm"] <= SHUT_BAR),
            "seconds": round(time.time() - tA, 1),
        }
        stub["phases_partial"][f"prof_{fam['key']}"] = E43.jsonable(
            {"D_kill": edge, "any_upcross": ups, "rows": rows,
             "anchor": fam["anchor_key"]})
        write_partial(f"profile {fam['key']} complete")
        log(f"  {fam['key']}: {len(rows)} pts in "
            f"{families[fam['key']]['seconds']}s — D_kill "
            + (f"{edge:.4f}" if edge is not None
               else "None (no downcrossing <= 3.0)")
            + (f", upcrosses {ups}" if ups else ""))

    # ---- G_ROOTPROF: the root-u0 family vs e193b's committed R2_SIGN rows
    r2_by_D = {round(r["D"], 4): r for r in e193b_r2
               if round(r["D"], 4) != 0.0}   # D=0 gated already (G_ROOT)
    xc = []
    for row in families["root_u0"]["rows"]:
        key = round(row["D"], 4)
        if key in r2_by_D and key <= 3.0:
            entry = {"D": key, "gm_measured": row["gm"],
                     "gm_e193b": r2_by_D[key]["gm_MIRABEL"],
                     "abs_diff": abs(row["gm"] - r2_by_D[key]["gm_MIRABEL"])}
            if "ce_r" in row and "ce_r" in r2_by_D[key]:
                entry["ce_r_measured"] = row["ce_r"]
                entry["ce_r_e193b"] = r2_by_D[key]["ce_r"]
                entry["ce_r_abs_diff"] = abs(row["ce_r"]
                                             - r2_by_D[key]["ce_r"])
            xc.append(entry)
    landings = []
    for row in families["root_u0"]["rows"]:
        if abs(row["D"] - STEP_L2) < 1e-9:
            landings.append({"D": row["D"], "gm": row["gm"],
                             "expect": E193B_ASIGN_S1["gm_MIRABEL"],
                             "abs_diff": abs(row["gm"]
                                             - E193B_ASIGN_S1["gm_MIRABEL"])})
    G_ROOTPROF = {
        "e193b_R2_crosscheck": xc, "n_shared": len(xc),
        "max_abs_diff": (max(r["abs_diff"] for r in xc) if xc else None),
        "max_ce_abs_diff": (max((r.get("ce_r_abs_diff", 0.0) for r in xc),
                                default=None)
                            if any("ce_r_abs_diff" in r for r in xc) else None),
        "step_l2_landing": (landings[0] if landings else None),
        "tol": G_STATIC_TOL, "tol_landing": G_LANDING_TOL,
        "required_shared": (len(shared_e193b_Ds) if not SMOKE else 3),
        "pass": bool(len(xc) >= (len(shared_e193b_Ds) if not SMOKE else 3)
                     and all(r["abs_diff"] < G_STATIC_TOL for r in xc)
                     and landings
                     and landings[0]["abs_diff"] < G_LANDING_TOL),
        "note": "THE STATIC-PROFILE MACHINERY GATE: the root-u0 family "
                "must reproduce e193b's committed R2_SIGN gm_MIRABEL rows "
                "at ALL shared Ds <= 3.0 (+ ce_r where shared) and the "
                "D=STEP_L2 landing must reproduce the committed walked "
                "step-1 MIRABEL read (fp32-norm ray vs fp64-norm walked "
                "step — texture class) before any new point is believed",
    }
    log(f"G_ROOTPROF (root-u0 vs e193b R2_SIGN): {len(xc)} shared pts, "
        f"max|diff| {(G_ROOTPROF['max_abs_diff'] or 0):.2e}, STEP_L2 "
        f"landing |d| "
        f"{(landings[0]['abs_diff'] if landings else float('nan')):.2e}: "
        + ("PASS" if G_ROOTPROF["pass"] else "FAIL"))

    # ---- G_TH1PROF: the theta_1 families' anchor gate
    th1_anchor_rows = [families[f"th1_{r}"]["rows"][0]
                       for r in ("u0", "u1", "u2")]
    anchor_ok = all(abs(r["gm"] - E193B_ASIGN_S1["gm_MIRABEL"]) < G_REPRO_TOL
                    for r in th1_anchor_rows)
    th1_land = None
    for row in families["th1_u1"]["rows"]:
        if abs(row["D"] - STEP_L2) < 1e-9:
            th1_land = {"D": row["D"], "gm": row["gm"],
                        "expect_walk_theta2": s2["gm"],
                        "abs_diff": abs(row["gm"] - s2["gm"])}
    G_TH1PROF = {
        "anchor_d0_gm": [r["gm"] for r in th1_anchor_rows],
        "anchor_expect": E193B_ASIGN_S1["gm_MIRABEL"], "anchor_ok": anchor_ok,
        "anchor_alive_disclosure": "theta_1 is ALIVE on MIRABEL's primary "
                                   f"({t1_gm:.4f} > {SHUT_BAR}): the "
                                   "theta_1 panel's kill-Ds CAN resolve "
                                   "(the first live theta_1 panel since "
                                   "organism 1's) — it still rides "
                                   "verbatim and NEVER adjudicates "
                                   "(e195/e196's convention)",
        "u1_step_l2_landing": th1_land,
        "tol_anchor": G_REPRO_TOL, "tol_landing": G_LANDING_TOL,
        "pass": bool(anchor_ok and th1_land
                     and th1_land["abs_diff"] < G_LANDING_TOL),
        "note": "THE THETA_1-PANEL MACHINERY GATE: all three theta_1 "
                "families' D=0 rows must reproduce the committed walked "
                "step-1 MIRABEL read bit-class; the th1-u1 family's "
                "D=STEP_L2 landing must reproduce the walk's theta_2 read "
                "(fp32-norm ray vs fp64-norm step — texture class)",
    }
    log(f"G_TH1PROF (theta_1 panel): anchor |d| "
        + ", ".join(f"{abs(r['gm'] - E193B_ASIGN_S1['gm_MIRABEL']):.2e}"
                    for r in th1_anchor_rows)
        + (f", u1 STEP_L2 landing vs theta_2 |d| "
           f"{th1_land['abs_diff']:.2e}" if th1_land
           else ", NO LANDING POINT")
        + ": " + ("PASS" if G_TH1PROF["pass"] else "FAIL"))
    if not (G_ROOTPROF["pass"] and G_TH1PROF["pass"]):
        stub["gates"]["G_ROOTPROF"] = G_ROOTPROF
        stub["gates"]["G_TH1PROF"] = G_TH1PROF
        write_partial("CONTROL FAILURE — profile machinery gates failed")
        raise RuntimeError("profile machinery gates FAILED — abort")
    stub["gates"]["G_ROOTPROF"] = G_ROOTPROF
    stub["gates"]["G_TH1PROF"] = G_TH1PROF
    write_partial("all profile machinery gates passed")

    # =====================================================================
    # ADJUDICATION (frozen clauses; composite ARCHITECTURE-CARRIES ->
    # BIOGRAPHY-CARRIES -> GRADED; the ROOT panel + the walk only; no
    # shopping)
    # =====================================================================
    dk = {k: families[k]["D_kill"] for k in families}
    root_dk = {k: dk[k] for k in ("root_u0", "root_u1", "root_u2")}
    u0_edge = root_dk["root_u0"]
    u1_kill = root_dk["root_u1"]
    walk_kill = walk["kill_mirabel"]["D_kill"] if walk["kill_mirabel"] \
        else None
    walk_final_D = walk["traj"][-1]["cum_disp"]
    walk_final_gm = walk["traj"][-1]["gm"]

    def ratio(a, b):
        return None if (a is None or b is None) else a / b

    u1_ratio = ratio(u1_kill, u0_edge)
    walk_ratio = ratio(walk_kill, u0_edge)

    # the FLIGHT clause (frozen operationalization)
    if u0_edge is None:
        flight_state, flight_fires = "UNRESOLVED", False
    elif u1_ratio is not None and u1_ratio < FLIGHT_RATIO_BAR:
        flight_state, flight_fires = "FIRES", True
    else:
        flight_state, flight_fires = "ABSENT", False
    # the WALK clause (frozen operationalization)
    walk_past_edge_alive = bool(u0_edge is not None
                                and walk_kill is None
                                and walk_final_D > u0_edge
                                and walk_final_gm > SHUT_BAR)
    if u0_edge is None:
        walk_state, walk_fires = "UNRESOLVED", False
    elif walk_ratio is not None and walk_ratio < WALK_RATIO_BAR:
        walk_state, walk_fires = "FIRES", True
    elif (walk_ratio is not None and walk_ratio >= WALK_RATIO_BAR) \
            or walk_past_edge_alive:
        walk_state, walk_fires = "ABSENT", False
    else:
        walk_state, walk_fires = "UNRESOLVED", False

    fires_arch = bool(flight_fires or walk_fires)
    fires_bio = bool(flight_state == "ABSENT" and walk_state == "ABSENT")

    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "ARCHITECTURE_CARRIES": {
                "fires": fires_arch,
                "detail": {
                    "flight_clause": {"state": flight_state,
                                      "D_kill_root_u0": u0_edge,
                                      "D_kill_root_u1": u1_kill,
                                      "u1_ratio": u1_ratio,
                                      "flight_ratio_bar": FLIGHT_RATIO_BAR,
                                      "organism1_committed_ratio":
                                          E195_ROOT_DKILLS["root_u1"]
                                          / E195_ROOT_DKILLS["root_u0"]},
                    "walk_clause": {"state": walk_state,
                                    "walk_D_kill": walk_kill,
                                    "walk_ratio": walk_ratio,
                                    "walk_ratio_bar": WALK_RATIO_BAR,
                                    "walk_final_D": walk_final_D,
                                    "walk_final_gm": walk_final_gm,
                                    "walk_past_edge_alive":
                                        walk_past_edge_alive,
                                    "organism1_committed_ratio":
                                        E195_WALK["D_kill"]
                                        / E195_WALK["static_edge"]},
                },
            },
            "BIOGRAPHY_CARRIES": {
                "fires": fires_bio,
                "detail": {"flight_state": flight_state,
                           "walk_state": walk_state,
                           "alive_at_theta1": True,
                           "terrain_replicating": True,
                           "remaining_candidates_if_fires": [
                               "FACT STRENGTH (org1's theta_1 0.679 vs "
                               "this 0.367 — a strength threshold?)",
                               "lineage biography (the specific base draw "
                               "+ install + consolidation history — "
                               "untestable except by draws)",
                               "fact identity (ZEPHYRA-vs-MIRABEL support "
                               "geometry — org1's fact is ZEPHYRA; this "
                               "cell's is MIRABEL, the one variable the "
                               "fork could not hold fixed)"],
                           },
            },
            "GRADED": {"fires": not (fires_arch or fires_bio)},
        }
        fmt = lambda v: ("None" if v is None else f"{v:.4f}")
        if fires_arch:
            verdict = "ARCHITECTURE-CARRIES"
            which = []
            if flight_fires:
                which.append(f"the flight ray concentrates: u1 kills at "
                             f"{fmt(u1_kill)} vs its OWN u0 edge "
                             f"{fmt(u0_edge)} (ratio {u1_ratio:.3f} < "
                             f"{FLIGHT_RATIO_BAR})")
            if walk_fires:
                which.append(f"the recomputation bonus appears: the walk "
                             f"kills at {fmt(walk_kill)} (ratio "
                             f"{walk_ratio:.3f} < {WALK_RATIO_BAR})")
            clause = (" and ".join(which)
                      + " — the 2.74M architecture carries the dynamics; "
                        "the fork resolves to architecture (org1 committed: "
                        "u1 ratio 0.171, walk ratio 0.771).")
        elif fires_bio:
            verdict = "BIOGRAPHY-CARRIES"
            clause = (f"MIRABEL (2.74M, terrain-replicating, alive at "
                      f"theta_1 {t1_gm:.4f}) shows NEITHER effect — flight "
                      f"clause {flight_state} "
                      f"(u1 ratio {fmt(u1_ratio)} vs bar "
                      f"{FLIGHT_RATIO_BAR}; u1 kill {fmt(u1_kill)}, u0 "
                      f"edge {fmt(u0_edge)}), walk clause {walk_state} "
                      f"(walk kill {fmt(walk_kill)}, ratio "
                      f"{fmt(walk_ratio)} vs bar {WALK_RATIO_BAR}, final "
                      f"D {walk_final_D:.3f} at gm {walk_final_gm:.4f}) — "
                      "organism 1's specific lineage biography carries the "
                      "flight; the architecture suspect cleared; the "
                      "remaining candidates: fact strength (0.679 vs "
                      "0.367), lineage draw, fact identity.")
        else:
            verdict = "GRADED"
            clause = ("any mix — the profiles verbatim: root panel "
                      + ", ".join(f"{k}:{fmt(v)}" for k, v in root_dk.items())
                      + f"; u1 ratio {fmt(u1_ratio)} (bar "
                      f"{FLIGHT_RATIO_BAR}, state {flight_state}); walk "
                      f"kill {fmt(walk_kill)}, ratio {fmt(walk_ratio)} "
                      f"(bar {WALK_RATIO_BAR}, state {walk_state}); "
                      "theta_1 panel (alive anchor, never adjudicates): "
                      + ", ".join(f"{k}:{fmt(dk[k])}"
                                  for k in ("th1_u0", "th1_u1", "th1_u2"))
                      + ".")
    log("=" * 78)
    log(f"E198 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e198_flight_arch",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "question": ("T160's named cut: does the flight concentration (and "
                     "the recomputation bonus) appear on e193b's fresh "
                     "2.74M MIRABEL root — organism-1's EXACT architecture, "
                     "a terrain-replicating fact, an ALIVE theta_1 — or is "
                     "it organism 1's specific biography? RATIO-form "
                     "within-organism throughout; org1 (e195) / org2 (e196, "
                     "e197) ride committed, never as bars"),
        "organism": {
            "root": f"runs/checkpoints/{ROOT_CK}",
            "root_meta": root_meta, "root_flat_md5": root_md5,
            "N_param": N_PARAM,
            "architecture": "6L/6H/192d/256-ctx TinyGPT (2,739,072) — "
                            "organism 1's EXACT architecture, FRESH draw "
                            "(e193b; architecture-pinned, lineage-varied)",
            "fact": "MIRABEL (e154's nonce; e193b's fact 2 — the fact that "
                    "passed its gate and replicated the whole terrain)",
            "step_l2_measured": STEP_L2,
            "step_l2_note": "THIS root's measured AdamW step-1 L2 at t=0 "
                            "(re-measured, gated vs e193b's committed "
                            "1.6544 — never ported); it is BELOW MIRABEL's "
                            "static sign edge (grid 2.0), so theta_1 is "
                            "ALIVE on the ruler",
        },
        "rulers": {
            "primary": {"battery": "MIRABEL install-60 g-12 (e192-verbatim)",
                        "root_read": root_gm_m,
                        "why": "e193b's frozen ruler call (g-12 alive "
                               "0.6236 — no fallback rule triggers)"},
            "co_rulers": {f"g{j:+d}": root_cells[FACT2][f"g{j:+d}"]
                          for j in CO_RULERS_J},
            "zephyra_co_read": {"battery": "ZEPHYRA install-60 g+0 "
                                           "(e193b's ladder ruler)",
                                "root_read": root_gm_z,
                                "why": "the G_REPRO comparator only"},
            "ce_r": root_cells["ce_r"],
        },
        "cell": {
            "licensed_cell": "e193b's stream/step convention VERBATIM "
                             "(seed 10902, draw order, name-free rule "
                             "(ZEPH + full MIRABEL), post-clip gradients, "
                             "measured STEP_L2, threads 8); the k=1 sign "
                             "walk rebuilt bit-exactly as the ray/state "
                             "factory (its a_sign arm), t=0..t=3",
            "grid": {"D": D_GRID, "extras": {"root_u0": [STEP_L2],
                                             "th1_u1": [STEP_L2]},
                     "note": "e195's D grid VERBATIM (0.05..3.00 step 0.05); "
                             "D=0 anchor rows in every family; e193b's "
                             "committed R2_SIGN gm_MIRABEL rows re-gated at "
                             "all shared Ds <= 3.0 inside root-u0"},
            "rays": rays_meta,
            "anchors": anchors_meta,
            "input_seed": FREEZE_SEED,
            "matched_step_L2": {"value": STEP_L2,
                                "provenance": "e193b's committed MEASURED "
                                "step-1 L2 (recomputed at t=0 and gated "
                                "G_T0/G_S1CK/G_REPRO)"},
        },
        "gates": stub["gates"],
        "phase0_walk_rebuild": stub["phases_partial"]["0_walk_rebuild"],
        "profiles": {k: {kk: vv for kk, vv in v.items()}
                     for k, v in families.items()},
        "ray_geometry": ray_geometry,
        "alignment_reads": alignment_reads,
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "composite_order": "ARCHITECTURE-CARRIES -> BIOGRAPHY-CARRIES "
                               "-> GRADED (frozen before compute; the ROOT "
                               "panel + the walk only — the theta_1 panel "
                               "never adjudicates)",
            "root_panel": root_dk,
            "theta1_panel": {k: dk[k] for k in
                             ("th1_u0", "th1_u1", "th1_u2")},
            "theta1_panel_note": f"ALIVE anchor ({t1_gm:.4f} > {SHUT_BAR}) "
                                 "— the first live theta_1 panel since "
                                 "organism 1's; still verbatim, never "
                                 "adjudicating",
            "walk": {"kill_mirabel": walk["kill_mirabel"],
                     "journal": walk["traj"],
                     "final_D": walk_final_D, "final_gm": walk_final_gm},
            "ratio_form": {
                "e193b_mirabel": {
                    "u1_over_u0": u1_ratio,
                    "u2_over_u0": ratio(root_dk["root_u2"], u0_edge),
                    "walk_over_u0": walk_ratio,
                    "flight_bar": FLIGHT_RATIO_BAR,
                    "walk_bar": WALK_RATIO_BAR,
                    "flight_state": flight_state, "walk_state": walk_state,
                },
                "organism1_committed_e195": {
                    "D_kills": E195_ROOT_DKILLS,
                    "u1_over_u0": E195_ROOT_DKILLS["root_u1"]
                                  / E195_ROOT_DKILLS["root_u0"],
                    "u2_over_u0": E195_ROOT_DKILLS["root_u2"]
                                  / E195_ROOT_DKILLS["root_u0"],
                    "walk_over_static": E195_WALK["D_kill"]
                                       / E195_WALK["static_edge"],
                    "theta1_gm": 0.679,
                },
                "organism2_committed_e196_dead": {
                    "D_kills": E196_ROOT_DKILLS,
                    "u1_over_u0": E196_ROOT_DKILLS["root_u1"]
                                  / E196_ROOT_DKILLS["root_u0"],
                    "u2_over_u0": E196_ROOT_DKILLS["root_u2"]
                                  / E196_ROOT_DKILLS["root_u0"],
                    "walk_over_static": E196_WALK["D_kill"]
                                        / E196_WALK["static_edge"],
                    "theta1_gm": 0.0068,
                },
                "organism2_committed_e197_alive_halfstep": {
                    "u1_over_u0": E197_ALIVE["u1_D_kill"]
                                  / E197_ALIVE["u0_edge"],
                    "walk_over_static": E197_ALIVE["walk_D_kill"]
                                        / E197_ALIVE["u0_edge"],
                    "theta1_gm": E197_ALIVE["theta1_gm"],
                },
                "context_never_adjudicated": {
                    "e193b_static_grid_kills_mirabel_primary":
                        E193B_KILLS,
                    "note": "e193b's committed grid kills on MIRABEL's "
                            "primary ruler (its 27-point grid; this cell's "
                            "kills are interpolated on the 0.05 grid — a "
                            "finer instrument, same convention as e195/"
                            "e196)"},
            },
            "constants": {"SHUT_BAR": SHUT_BAR, "D_grid": D_GRID,
                          "FLIGHT_RATIO_BAR": FLIGHT_RATIO_BAR,
                          "WALK_RATIO_BAR": WALK_RATIO_BAR,
                          "STEP_L2": STEP_L2, "WALK_STEPS": WALK_STEPS},
        },
        "references": {
            "e193b_two_fact": {
                "metrics": "runs/e193b/metrics.json",
                "checkpoints": [f"runs/checkpoints/{ROOT_CK}",
                                f"runs/checkpoints/{S1_CK}",
                                f"runs/checkpoints/{DIR_CK}"],
                "role": "THE PARENT: the fresh 2.74M two-fact root + its "
                        "committed dial, measured STEP_L2, a_sign step-1 "
                        "row (MIRABEL 0.3673 ALIVE), ZEPHYRA kill bracket, "
                        "R2_SIGN profile and ray md5s — loaded COMMITTED, "
                        "gated, never rerun"},
            "e195_rotated_ray": {
                "metrics": "runs/e195/metrics.json",
                "role": "the flight-map machinery + organism 1's committed "
                        "flight result (u1/u0 0.1707 FLEEING-IS-LETHAL) — "
                        "the side-by-side, never a bar"},
            "e196_flight_replicate": {
                "metrics": "runs/e196/metrics.json",
                "role": "organism 2's dead-lineage flight read (u1/u0 "
                        "2.5847, GRADED) + the walk-rebuild machinery this "
                        "cell ports"},
            "e197_alive_window": {
                "metrics": "runs/e197/metrics.json",
                "role": "organism 2's ALIVE half-step lineage (u1/u0 "
                        "3.1690; walk ratio 1.5969 path-safer) — the "
                        "alive-not-sufficient lesson that named this cell"},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 root / n=1 fact: ONE fresh draw at "
                            "organism-1's architecture with ONE fact "
                            "(MIRABEL — not organism 1's ZEPHYRA; fact "
                            "identity co-varies and is named as a "
                            "remaining candidate under BIOGRAPHY); each "
                            "read is THIS wash trajectory's own (seed "
                            "10902, md5-gated at every consumed step) — a "
                            "single-path biography, not a population claim"),
            "fresh_draw_caveat": ("'architecture carries' would rest on "
                                  "ONE draw at the 2.74M architecture; "
                                  "'biography carries' clears the "
                                  "architecture suspect for THIS draw+fact "
                                  "only — either verdict wants replicate "
                                  "draws before it generalizes (the "
                                  "critic's standing n=1 ledger)"),
            "openness": ("WHAT EACH FAMILY GUARANTEES: NOTHING — every "
                         "family is a static graded jump at lethal scale "
                         "from a state that could kill anywhere; cross-"
                         "panel D comparisons are NOT adjudicated; that "
                         "openness is the point"),
            "estimator_lesson": ("every alignment read states its "
                                 "evaluation point (matched-point AT the "
                                 "anchor, dual form: walked-delta vs "
                                 "unit-ray cross-consistency) — T150's "
                                 "lesson; no cross-organism cosine "
                                 "comparisons (different instruments)"),
            "projections_never_adjudicate": ("D_kill is a linear-in-D "
                                             "interpolation on measured "
                                             "grid points; the 0.70/0.85 "
                                             "margins were frozen before "
                                             "compute"),
            "float_texture": ("CPU fp32 texture, this process, 8 threads "
                              "(e193b's gate convention); the walk "
                              "reproduced e193b's committed a_sign row to "
                              f"{max(abs(vv[0] - vv[1]) for vv in repro_rows.values()):.1e}"),
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_flight_map(rd / "e198_flight_map.png", families, walk, root_dk,
                    walk_kill, u1_ratio, walk_ratio)
    plot_three_organism(rd / "e198_three_organism.png", families, root_dk,
                        u1_ratio, walk_ratio, e195_root,
                        e196m["profiles"])
    log(f"outputs: {rd / 'metrics.json'} + 2 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

RAY_STYLE = {
    "u0": {"color": "navy", "label": "sign(g_0) — the static sign ray"},
    "u1": {"color": "crimson",
           "label": "sign(g_1) — THE FLIGHT RAY (t=1 live front)"},
    "u2": {"color": "darkorange", "label": "sign(g_2) — the t=2 front"},
}


def _dual_axis(ax):
    ax.set_xlabel(r"static displacement $D = \|\theta_D - \theta_{anchor}\|_2$")
    axr = ax.secondary_xaxis(
        "top", functions=(lambda d: d / RMS_DENOM,
                          lambda r: r * RMS_DENOM))
    axr.set_xlabel("per-coordinate RMS (/ 1655.014 — same currency as organism 1)")


def plot_flight_map(path, families, walk, root_dk, walk_kill, u1_ratio,
                    walk_ratio):
    """MIRABEL's flight map: the root panel + the LIVE theta_1 panel + the
    walk's own path (the recomputation check)."""
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6))
    for ax, panel, anchor_lbl in (
            (axes[0], ("root_u0", "root_u1", "root_u2"),
             "THE ROOT PANEL — e193b MIRABEL (g-12 ruler; ADJUDICATES)"),
            (axes[1], ("th1_u0", "th1_u1", "th1_u2"),
             "FROM $\\theta_1$ (ALIVE anchor 0.367; never adjudicates)")):
        for key in panel:
            fam = families[key]
            st = RAY_STYLE[fam["ray"]]
            ax.plot([r["D"] for r in fam["rows"]],
                    [r["gm"] for r in fam["rows"]], "o-", ms=2.6,
                    lw=1.7, color=st["color"], label=st["label"])
            if fam["D_kill"] is not None:
                ax.axvline(fam["D_kill"], color=st["color"], ls=":", lw=1.4)
                ax.text(fam["D_kill"], 0.985,
                        f"{fam['ray']} kills {fam['D_kill']:.3f}",
                        rotation=90, fontsize=6.6, va="top", ha="right",
                        color=st["color"])
        ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
                   label=f"{SHUT_BAR} SHUT bar (the ruler)")
        _dual_axis(ax)
        ax.set_ylabel("MIRABEL g-12 (install-60 battery mean p(M))")
        ax.set_ylim(-0.03, 1.02)
        ax.set_title(anchor_lbl, fontsize=10)
        ax.legend(fontsize=7.0, loc="upper right")
    axes[0].axvline(E193B_KILLS["sign"], color="dimgray", ls="-.", lw=1.3)
    axes[0].text(E193B_KILLS["sign"] + 0.03, 0.55,
                 f"e193b static sign grid-kill {E193B_KILLS['sign']}",
                 fontsize=6.8, rotation=90, color="dimgray")

    # panel 3: the walk's own path (the recomputation check)
    ax = axes[2]
    xs = [0.0] + [r["cum_disp"] for r in walk["traj"]]
    ys = [families["root_u0"]["rows"][0]["gm"]] + [r["gm"]
                                                   for r in walk["traj"]]
    ax.plot(xs, ys, "o-", ms=5, lw=2.0, color="seagreen",
            label="the k=1 sign walk t=0..t=3 (its own path)")
    for x, y, s_ in zip(xs[1:], ys[1:], (1, 2, 3)):
        ax.annotate(f"t={s_}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5)
    if root_dk["root_u0"] is not None:
        ax.axvline(root_dk["root_u0"], color="navy", ls=":", lw=1.5)
        ax.text(root_dk["root_u0"] - 0.07, 0.85,
                f"static u0 edge {root_dk['root_u0']:.3f}",
                fontsize=7.2, rotation=90, color="navy", ha="right")
    if walk_kill is not None:
        ax.axvline(walk_kill, color="seagreen", ls=":", lw=1.5)
        ax.text(walk_kill - 0.07, 0.5,
                f"walk kill {walk_kill:.3f}\nratio {walk_ratio:.3f} "
                f"(bar {WALK_RATIO_BAR})", fontsize=7.2, rotation=90,
                color="seagreen", ha="right")
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple")
    _dual_axis(ax)
    ax.set_ylabel("MIRABEL g-12 (the walk's every-step read)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE RECOMPUTATION CHECK — the walk vs its static edge",
                 fontsize=10)
    ax.legend(fontsize=7.4, loc="upper right")
    fig.suptitle("E198 — MIRABEL's FLIGHT MAP on e193b's fresh 2.74M root "
                 "(organism-1's exact architecture; e193b's gates verbatim; "
                 "ratio-form within-organism)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_three_organism(path, families, root_dk, u1_ratio, walk_ratio,
                        e195_rows_all, e196_rows_all):
    """THE THREE-ORGANISM RATIO SIDE-BY-SIDE: every root-panel curve on its
    own u0-edge-normalized axis + the flight/walk ratio bars (org1 e195,
    org2 e196 dead + e197 alive half-step, e193b-MIRABEL e198)."""
    fig = plt.figure(figsize=(19.5, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.2, 1.0, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    # e198's own curves
    if root_dk["root_u0"]:
        for key in ("root_u0", "root_u1", "root_u2"):
            fam = families[key]
            st = RAY_STYLE[fam["ray"]]
            xs = [r["D"] / root_dk["root_u0"] for r in fam["rows"]]
            ys = [r["gm"] for r in fam["rows"]]
            ax.plot(xs, ys, "o-", ms=2.4, lw=1.9, color=st["color"],
                    label=f"e193b MIRABEL {fam['ray']}")
    # organism 1 (e195 committed; its rows carry the g-12 field)
    for key, col in (("root_u0", "navy"), ("root_u1", "crimson"),
                     ("root_u2", "darkorange")):
        rows = e195_rows_all[key]["rows"]
        xs = [r["D"] / E195_ROOT_DKILLS["root_u0"] for r in rows]
        ys = [r["gm12"] for r in rows]
        ax.plot(xs, ys, "--", lw=1.4, color=col, alpha=0.65,
                label=f"org1 {key.split('_')[1]} (e195)")
    # organism 2 (e196 committed; its rows carry the gm field)
    for key, col in (("root_u0", "navy"), ("root_u1", "crimson"),
                     ("root_u2", "darkorange")):
        rows = e196_rows_all[key]["rows"]
        xs = [r["D"] / E196_ROOT_DKILLS["root_u0"] for r in rows]
        ys = [r["gm"] for r in rows]
        ax.plot(xs, ys, ":", lw=1.6, color=col, alpha=0.8,
                label=f"org2 {key.split('_')[1]} (e196)")
    ax.axvline(1.0, color="navy", ls=":", lw=1.2)
    ax.axvline(FLIGHT_RATIO_BAR, color="crimson", ls=":", lw=1.5)
    ax.text(FLIGHT_RATIO_BAR - 0.03, 0.92,
            f"flight bar {FLIGHT_RATIO_BAR}\norg1 0.171 / org2 2.585",
            fontsize=7, color="crimson", ha="right")
    if u1_ratio is not None:
        ax.plot([u1_ratio], [SHUT_BAR], "*", ms=15, color="crimson",
                markeredgecolor="k", zorder=5,
                label=f"e193b MIRABEL u1 kills at ratio {u1_ratio:.3f}")
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple")
    ax.set_xlabel(r"$D / D_{kill}(\mathrm{own\ root}, u_0)$ — RATIO form "
                  "(each organism on its own edge; own rulers)")
    ax.set_ylabel("own ruler (org1 ZEPHYRA g-12; org2 g-4; e193b MIRABEL "
                  "g-12 — different instruments, disclosed)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE THREE-ORGANISM RATIO SIDE-BY-SIDE (root panels)",
                 fontsize=10)
    ax.legend(fontsize=6.3, loc="upper right")

    # the ratio bars: flight + walk per lineage
    ax = fig.add_subplot(gs[0, 1])
    labels = ["org1\n(e195)\n2.74M\nalive 0.679",
              "org2\n(e196)\n873k\ndead 0.007",
              "org2 half\n(e197)\n873k\nalive 0.419",
              "e193b\nMIRABEL\n(e198)\n2.74M\nalive 0.367"]
    flight_vals = [E195_ROOT_DKILLS["root_u1"] / E195_ROOT_DKILLS["root_u0"],
                   E196_ROOT_DKILLS["root_u1"] / E196_ROOT_DKILLS["root_u0"],
                   E197_ALIVE["u1_D_kill"] / E197_ALIVE["u0_edge"],
                   u1_ratio if u1_ratio is not None else 0.0]
    cols = ["tab:blue", "tab:blue", "tab:blue", "crimson"]
    bars_ = ax.bar(np.arange(4), flight_vals, color=cols, alpha=0.85)
    for i, v in enumerate(flight_vals):
        txt = f"{v:.3f}" if (i < 3 or u1_ratio is not None) else "None"
        ax.text(i, v + 0.06, txt, ha="center", fontsize=8)
    ax.axhline(FLIGHT_RATIO_BAR, color="crimson", ls=":", lw=1.6)
    ax.text(3.4, FLIGHT_RATIO_BAR + 0.03, f"flight bar {FLIGHT_RATIO_BAR}",
            fontsize=7.5, color="crimson", ha="right")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("flight ratio  D_kill(root,u1)/D_kill(root,u0)")
    ax.set_title("THE FLIGHT CONCENTRATION, RATIO FORM", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    walk_vals = [E195_WALK["D_kill"] / E195_WALK["static_edge"],
                 E196_WALK["D_kill"] / E196_WALK["static_edge"],
                 E197_ALIVE["walk_D_kill"] / E197_ALIVE["u0_edge"],
                 walk_ratio if walk_ratio is not None else 0.0]
    ax.bar(np.arange(4), walk_vals, color=cols, alpha=0.85)
    for i, v in enumerate(walk_vals):
        txt = f"{v:.3f}" if (i < 3 or walk_ratio is not None) else "None"
        ax.text(i, v + 0.04, txt, ha="center", fontsize=8)
    ax.axhline(WALK_RATIO_BAR, color="seagreen", ls=":", lw=1.6)
    ax.text(3.4, WALK_RATIO_BAR + 0.02, f"walk bar {WALK_RATIO_BAR}",
            fontsize=7.5, color="seagreen", ha="right")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("walk ratio  D_kill(walk)/static u0 edge")
    ax.set_title("THE RECOMPUTATION BONUS, RATIO FORM", fontsize=10)

    fig.suptitle("E198 — THE ARCHITECTURE-VS-BIOGRAPHY CUT: the flight "
                 "concentration across organisms (all committed side-by-"
                 "sides; e198's numbers in crimson)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
