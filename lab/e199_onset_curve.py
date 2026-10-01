"""E199 — THE CONCENTRATION'S ONSET CURVE (T162's named timing cut: is the
flight concentration a WHEN, not a WHETHER?).

WHY (T162, verbatim): "THE T=2 WRINKLE IS THE REOPENED DOOR: the rotation
finds a lethal direction by t=2 here — the concentration may be a WHEN, not
a WHETHER: the front needs time (or steps) to rotate onto the fleeing
support, and org1 did it in one step where MIRABEL needs two. THE TIMING CUT
IS NAMED (ripening): org1's own u2/u3 map (does its concentration DEEPEN
past t=1?) vs MIRABEL's t=3/t=4 — the concentration's onset curve on both
organisms." The committed ledger: org1's t=1 flight ray concentrates (ratio
0.17); MIRABEL's t=1 does NOT (unresolved-high — u1 never kills within
[0, 3.0], min read 0.517) but its t=2 DOES (ratio 0.43). THE QUESTION: on
each organism, walk its wash trajectory further (t=1..6 as the states stay
alive; stop at death) and map sign(g_t) rays from the ROOT at each t — does
each organism's concentration ratio DEEPEN along its trajectory? Is there a
common onset shape, just shifted in time?

THE ORGANISMS (both committed, never re-trained):
  * ORG1 — runs/checkpoints/e131_consolidated_e113.pt (e195's root; ZEPHYRA
    install-60 g-12 ruler; its own measured STEP_L2 1.6542880535125732).
    Committed walk: ALIVE t=1 (0.6786), DEAD t=2 (9.8e-5) — opt2/e194/e195.
  * MIRABEL — runs/checkpoints/e193b_root.pt (e198's root; MIRABEL install-60
    g-12 ruler; its own measured STEP_L2 1.6544127464294434). Committed walk:
    ALIVE t=1 (0.3673), ALIVE t=2 (0.6911), DEAD t=3 (0.1230) — e198.

THE CELL (eval-only CPU, minutes):
  (1) extend each wash walk to t=6 or its death, whichever first — the
      walks' machinery VERBATIM (org1: e195's walk_arm; MIRABEL: e198's
      walk_rebuild); the committed walks already run to each organism's
      death (t=2 / t=3 — both re-verified in the rebuilt journals, gated
      bit-class), so the extension arm is VACUOUS for both: there are no
      missing steps to walk (the dispatch's "LOAD where they exist, walk
      only the missing steps" clause — everything exists);
  (2) at each ALIVE t: the sign(g_t) ray from the ROOT — static graded
      jumps theta_D = root - D*u_t, D grid 0.05..3.00 step 0.05 (e195/e198's
      grid VERBATIM) + D=0 anchor rows, dual currency (D + per-coordinate
      RMS / 1655.014), each organism's OWN rulers (org1: g-12 battery +
      decimated ce_r, e195's static_profile; MIRABEL: g-12 primary + g-4/
      g0/g+12 co-rulers + decimated ce_r, e198's static_profile); kill-D +
      ratio vs its OWN u0 edge. The rays are REBUILT fresh from the
      re-walked gradients and md5-gated vs the committed directions; the
      profiles re-gated row-wise vs e195's/e198's committed rows.
  (3) the onset curve: ratio vs t per organism, side by side.

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - ONSET-COMMON: "fires if both organisms show the ratio falling with t (a
    deepening concentration once rotation begins), with org1's curve shifted
    one step earlier than MIRABEL's — the concentration is a WHEN; the
    timing is the biography."
  - ONSET-ORG1-ONLY: "fires if org1's ratio deepens (or stays at its t=1
    floor) while MIRABEL's stays flat/soft at every t — the concentration
    itself is org-1-only; the t=2 wrinkle was noise."
  - GRADED: "any mix — the curves verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * ALIVE t: the walk's every-step primary-ruler read at theta_t > SHUT
    (0.27, e185's bar, absolute, both organisms' own rulers). Rays are read
    ONLY at alive ts — the post-death ts are never read in this cell (org1's
    t=2 front rides as e195's COMMITTED post-kill read, loaded as context,
    flagged, and never adjudicates).
  * ratio_t = D_kill(root, u_t) / D_kill(root, u0) — each organism vs its
    OWN u0 edge (T153's within-organism ratio form); ratio_0 = 1.0 by
    definition (u0 vs itself).
  * "kills at D" = the family profile's first primary-ruler <= 0.27
    downcrossing on the static grid, linear-in-D interpolated (e192/e194/
    e195/e198's convention); None = no downcrossing within [0, 3.0].
  * a ray CONCENTRATES at t iff ratio_t < 0.70 (e198's frozen flight bar);
    a None kill within [0, 3.0] is SOFT (unresolved-high; the floor ratio
    3.0/u0_edge is reported as a floor, never adjudicated as a number — a
    ray that does not even kill within [0, 3.0] cannot be concentrating
    below an edge <= 3.0/0.70, e198's clause).
  * "the ratio falling with t (a deepening concentration once rotation
    begins)" = the organism's curve reaches a concentrated alive t (some
    ratio < 0.70) AND every RESOLVED ratio strictly after its first
    concentrated t stays <= 0.70 (the concentration, once arrived, does not
    un-form inside the alive window; with no later alive t this holds
    vacuously — org1's "stays at its t=1 floor" case, the ORG1-ONLY bar's
    own parenthetical).
  * "org1's curve shifted one step earlier than MIRABEL's" = org1's first
    concentrated t is exactly MIRABEL's first concentrated t minus 1.
  * "MIRABEL's stays flat/soft at every t" = NO alive t of MIRABEL
    concentrates (every ratio None or >= 0.70).
  * composite order frozen: ONSET-COMMON -> ONSET-ORG1-ONLY -> GRADED (the
    first that fires is the verdict; every bar's fires flag reported
    verbatim).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): BOTH
organisms' gate sets REUSED VERBATIM — org1: e195's (G_NAMEFREE, G_SPLICE,
G_BATTERY, G_ANCHOR vs e185's stored bank, G_ROOT vs e151's before-cells,
G_T0 vs opt1's committed A0 row, G_STREAM, G_SAMEPOINT vs the chart's
committed anchors — T150's dual-estimator lesson, G_U191, G_R2DIR vs e192's
committed md5, G_REPRO vs opt2's committed a_sign trajectory + densified
bracket, G_SAVEDSTATE vs runs/checkpoints/opt2_a_sign_s2.pt, G_FRONTS vs
e194's committed front trace, the u1 md5 vs e195's registered ray);
MIRABEL: e198's (G_NAMEFREE, G_SPLICE, G_BATTERY, G_ANCHOR, G_ROOT vs
e193b's committed seven-geometry dial + flat md5, G_T0 (its measured STEP_L2
never ported, re-measured), G_S1CK, G_STREAM, G_DIRCK, G_SIGNRAY, G_REPRO vs
e193b's committed a_sign step-1 row + ZEPHYRA bracket, G_TH1ANCHOR, G_U12 —
the rebuilt u1/u2 md5s vs e198's registered rays) PLUS the profile
machinery gates: every recomputed family must reproduce its committed rows
(e195's root_u0/root_u1 for org1; e198's root_u0/root_u1/root_u2 +
e193b's committed R2_SIGN gm_MIRABEL rows at the shared Ds for MIRABEL) and
its committed D_kill before any onset number is believed. THE WALKS'
STEP-L2s are MEASURED per organism (org1 1.6543 via G_T0; MIRABEL 1.6544 via
its own G_T0) and asserted per step (matched-L2, fp64). The thread
conventions are per organism (org1: 4, the e185-era order; MIRABEL: 8,
e193b's — switched between phases, disclosed). WHAT EACH ARM GUARANTEES:
NOTHING — every family is a static graded jump at lethal scale from a state
that could kill anywhere; the walks stop at death; the openness is the
point.

REGISTERED PREDICTION (frozen before compute): T162's own fork — the t=2
wrinkle suggests the concentration is a WHEN (COMMON), but e198's t=1 read
was not marginal (u1 never kills <= 3.0) and org1's single-step floor could
be its whole story (ORG1-ONLY); either verdict is informative; no bar
shopping.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
the GPU is another agent's, never claimed), sequential phases (threads 4
for the org1 phase, then 8 for MIRABEL — each organism's own gate
convention), PROGRESSIVE partial metrics.json writes after every phase (the
outage lesson), n=1 organism/fact per side, seed lineage 10902.

Outputs: runs/e199/{metrics.json, e199_onset_curve.png,
e199_ray_profiles.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e199_onset_curve.py    (E199_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY (the opt-line convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # org1 phase (e185-era order); 8 later

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E199_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e199 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

FACT1 = "ZEPHYRA"                  # org1's fact (its ruler)
FACT2 = "MIRABEL"                  # the second organism's fact (e193b/e198's ruler)
PRE, POST_CAP = 130, 119           # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
ORG1_ROOT_CK = "e131_consolidated_e113.pt"     # org1's root (e195's)
ORG1_DIR_CK = "e191_static_dir_u.pt"           # org1's committed g-ray direction ckpt
ORG2_SIGN_CK = "opt2_a_sign_s2.pt"             # org1's committed walk step-2 state
MIR_ROOT_CK = "e193b_root.pt"                  # MIRABEL's root (e198's)
MIR_S1_CK = "e193b_wash_s1.pt"                 # e193b's committed wash step-1 ckpt
MIR_DIR_CK = "e193b_static_dir_u.pt"           # e193b's committed g-ray direction ckpt

OPT2_METRICS = E43.REPO / "runs" / "opt2" / "metrics.json"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
E192_METRICS = E43.REPO / "runs" / "e192" / "metrics.json"
ECHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
E194_METRICS = E43.REPO / "runs" / "e194" / "metrics.json"
E195_METRICS = E43.REPO / "runs" / "e195" / "metrics.json"
E193B_METRICS = E43.REPO / "runs" / "e193b" / "metrics.json"
E196_METRICS = E43.REPO / "runs" / "e196" / "metrics.json"
E197_METRICS = E43.REPO / "runs" / "e197" / "metrics.json"
E198_METRICS = E43.REPO / "runs" / "e198" / "metrics.json"

# ---- the shared architecture (both organisms are 6L/6H/192d/256-ctx) --------------
CFG1 = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
N_PARAM = 2_739_072
RMS_DENOM = 1655.0141993348577      # sqrt(2,739,072) — the per-coordinate RMS currency

# ---- rulers ------------------------------------------------------------------------
ORG1_GEOS = (-12, 0, 12)            # e185's battery ctx offsets (org1's 3-geometry form)
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries (e193b/e198)
RULER_J = -12                       # PRIMARY both organisms: install-60 g-12
CO_RULERS_J = (-4, 0, 12)           # e198's per-grid-point co-rulers (MIRABEL rows)

# ---- the run envelope (dispatch-frozen) --------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193/e193b/opt-line)
LR_ADAMW = 1e-3                   # the t=0 recipe (both G_T0s; the ray arms have NO lr)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (e195/e198's grid VERBATIM)
CE_EVERY = 0.2                    # ce_r at D in {0.2, 0.4, ..., 3.0} (e195's decimation)
WALK_MAX_T = 6                    # the dispatch's walk horizon (both organisms die before)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # opt1c's along-path kill-bracket densification
CONC_BAR = 0.70                   # frozen: "concentrates" = ratio < 0.70 (e198's flight bar)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_STATIC_TOL = 1e-3               # committed-row crosscheck class (e195's G_ROOTPROF)
G_SAMEPOINT_TOL = 2e-3
DKILL_TOL = 1e-6                  # D_kill equality vs committed (same grid/code; far
                                  # below the 0.05 grid resolution — texture headroom)
L2_ASSERT_TOL = 1e-5
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    CE_EVERY = None

# ---- org1's committed lineage reads (e195/e194/opt2/e192/e_chart; gate targets) -----
ORG1_STEP_L2 = 1.6542880535125732          # opt1 A0's committed step-1 L2 (measured, G_T0)
ORG1_CE1 = 1.356567621231079               # opt1 A0's committed step-1 batch CE
ORG1_GN1 = 0.9829167127609253              # opt1 A0's committed step-1 pre-clip grad norm
ORG1_A0_DKILL = 2.4892616271972656         # opt1 A0's committed kill displacement
OPT2_S1 = {"ce_batch": 1.356567621231079, "cum_disp": 1.6567984819412231,
           "gm12": 0.6785961389541626, "preclip_gnorm": 0.9829167127609253,
           "step_disp": 1.6567984819412231,
           "matched_point_m12": 0.03955041750707301}
OPT2_S2 = {"ce_batch": 1.8259366750717163, "cum_disp": 2.1502907276153564,
           "gm12": 9.830708586378023e-05, "preclip_gnorm": 6.194264888763428,
           "step_disp": 1.6567445993423462,
           "matched_point_m12": 0.06015982408222451}
OPT2_DENS = [(0.2, 0.8249044418334961, 1.6348317861557007),
             (0.4, 0.6051573157310486, 1.683386206626892),
             (0.6, 0.05381816625595093, 1.7923755645751953),
             (0.8, 0.013179803267121315, 1.952124834060669)]
OPT2_PATH_DKILL = 1.749640490742179        # opt2 a_sign's committed densified kill D
E194_REFINED_EDGE = 2.269916581032063       # e194 phase B's committed static edge (= org1 u0)
E192_A0_SIGNRAY_GM12 = 0.6785961389541626   # e192's static-sign D1.6543 read (landing gate)
E192_R2_U_MD5 = "aac6c6d643e327939f680148772dd180"   # e192's committed R2 direction md5 (= org1 u0)
E192_SIGN_EDGE_COARSE = 2.5                 # e192's committed coarse static sign edge
W024_ADAM_SAMEPOINT = 0.039550412581627135  # the chart's committed same-point sign read
W024_RAW_SAMEPOINT = 0.09862135965497001    # the chart's committed same-point raw read
E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
}
E194_FRONTS = [   # committed front-trace rows t=0,1 (e194; org1's walk stops at the kill)
    {"front_state": 0, "vs_g_ray": 0.5950223410088911,
     "vs_static_sign": 1.0000000000017162, "vs_prev_front": None},
    {"front_state": 1, "vs_g_ray": -0.1622211100709781,
     "vs_static_sign": -0.15452721980470194, "vs_prev_front": -0.15452721980443676},
]
# e195's committed root panel (org1's onset anchors; gate targets)
E195_ORG1 = {"root_u0_D_kill": 2.269916581032063,
             "root_u1_D_kill": 0.3875040789853107,
             "root_u2_D_kill": 0.8979436419840907,
             "u0_md5": E192_R2_U_MD5,
             "u1_md5": "c9b8da4f25c05fb7ea7701b08ccba8c0",
             "u2_md5": "bb563d4fec6993e4704a1a966f33dca3"}

# ---- MIRABEL's committed lineage reads (e193b/e198; gate targets) ------------------
MIR_ROOT_MD5 = "9113c7593dbac14fb57d47f0b96c1587"
MIR_STEP_L2 = 1.6544127464294434            # this root's measured AdamW step-1 L2 (G_T0)
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
E193B_SPLICE_MIX = {"ZEPHYRA": {"FLORIZEL": 19, "ELIZABETH": 41},
                    "MIRABEL": {"FLORIZEL": 16, "ELIZABETH": 44}}
E193B_ASIGN_S1 = {                              # e193b's committed a_sign step-1 row (G_REPRO)
    "ce_batch": 1.531902551651001,
    "cum_disp": 1.6568037271499634,
    "step_disp": 1.6568037271499634,
    "preclip_gnorm": 0.9484267234802246,
    "gm_ZEPHYRA": 0.052623070776462555,
    "frac_argmax_z": 0.0,
    "ce_r": 2.4445571899414062,
    "gm_MIRABEL": 0.3672811686992645,            # THE ALIVE THETA_1 READ (g-12)
    "g-4_MIRABEL": 0.5463535785675049,
    "g+0_MIRABEL": 0.16293005645275116,
    "g+12_MIRABEL": 0.1902925670146942,
}
E193B_ASIGN_STOP = {"kind": "kill", "step": 1,
                    "D_kill_raw": 0.7767010305763323,
                    "D_kill": 0.8969455194321004}
E193B_UG_MD5 = "159263074c9e8a48fc898bca48017f47"   # its committed g-ray direction
E193B_USIGN_MD5 = "fbb878c1926c165ebdb54eef1259ab65"  # its committed R2_SIGN (= MIRABEL u0)
# e198's committed walk journal (the t=0..t=3 trajectory; gate targets)
E198_WALK_JOURNAL = [
    {"step": 1, "ce_batch": 1.531902551651001, "cum_disp": 1.6568037271499634,
     "preclip_gnorm": 0.9484267234802246, "gm": 0.3672811686992645,
     "gm_ZEPHYRA": 0.052623070776462555, "ce_r": 2.4445571899414062},
    {"step": 2, "ce_batch": 2.314697265625, "cum_disp": 2.1239850521087646,
     "preclip_gnorm": 6.934149265289307, "gm": 0.6910732984542847},
    {"step": 3, "ce_batch": 2.1457536220550537, "cum_disp": 2.4923200607299805,
     "preclip_gnorm": 6.33951997756958, "gm": 0.12300960719585419},
]
E198_KILL_M = {"step": 3, "gm_at_kill": 0.12300960719585419,
               "D_kill_raw": 2.3970108402430825,
               "D_kill": 2.4127699365528312}
# e198's committed root panel (MIRABEL's onset anchors; gate targets)
E198_MIR = {"root_u0_D_kill": 1.9471128428088909,
            "root_u1_D_kill": None,
            "root_u2_D_kill": 0.8313359802130567,
            "u0_md5": E193B_USIGN_MD5,
            "u1_md5": "7702614f1ff109a0f3597533b42d9934",
            "u2_md5": "4e862a134281b94a6b1c6e4d3531186b"}
E198_RAY_GEOM = {                 # e198's committed ray mutual geometry (fp64)
    "cos_u0_u1": -0.17510781540323442,
    "cos_u0_u2": -0.015712286422230576,
    "cos_u1_u2": -0.17813847702288285,
}
# ---- the identity-gate TIERS (frozen before the full compute) -----------------------
# BIT = md5 / 1e-9 (same-process-class arithmetic). TEXTURE = the
# cross-environment fallback (e196's G_S1CK precedent): today's process
# reproduces org1's whole chain BIT-EXACTLY (0.00e+00 everywhere) but
# drifts the e193b/e198 chain at ~1e-7 fp32 (e198 reproduced e193b
# bit-exactly on ITS run — the drift is environmental, disclosed in every
# gate that uses the fallback; the achieved tier is STAMPED, never silently
# weakened).
RAY_COS_TOL = 1e-3                # mutual-geometry cosine tier (u1/u2 identity)
RAY_SIGN_COS_FLOOR = 0.99999      # sign-ray cosine floor vs a checkpoint tensor
WALK_TEXTURE_TOL = 1e-3           # MIRABEL's walk-row reproduction tier
DKILL_TEXTURE_TOL = 1e-2          # MIRABEL's D_kill reproduction tier (a 1% of D)
E193B_R2_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.4, 0.5, 0.58, 0.66,
                 0.74, 0.8, 0.87, 0.92, 1.0, 1.08, 1.17, 1.33, 1.5, 1.75, 2.0,
                 2.25, 2.5, 3.0, 3.5, 4.0]      # its 27-point D grid (shared Ds <= 3.0 re-gated)

# ---- the committed four-lineage ledger (rides as context, never adjudicates) -------
E196_ROOT_DKILLS = {"root_u0": 0.5252331635450007,   # organism 2 (873k; dead theta_1)
                    "root_u1": 1.3575583149916892,
                    "root_u2": 0.46025487385682545}
E197_ALIVE = {"u1_D_kill": 1.6644735674638536,       # org2's half-step ALIVE lineage
              "u0_edge": 0.5252331635450007}

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
ANCHOR_FORBIDDEN_ORG1 = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
    5: "4da7b67a7fd80b4e9729731fb27bec0c",
    6: "1aa4f9f250f14acad52d3b969343090a",
    7: "1e9e3373028935fe272d86683280e4b5",
    8: "3535a9db2d1aa2e6e0655208ff26b3d9",
    9: "2c3260242cd38e60cffafe0c91495c5e",
    10: "688062bb39f486e563b091124a7231a1",
}

REGISTERED_BARS = {
    "ONSET_COMMON": "ONSET-COMMON: \"fires if both organisms show the ratio "
        "falling with t (a deepening concentration once rotation begins), "
        "with org1's curve shifted one step earlier than MIRABEL's — the "
        "concentration is a WHEN; the timing is the biography.\"",
    "ONSET_ORG1_ONLY": "ONSET-ORG1-ONLY: \"fires if org1's ratio deepens (or "
        "stays at its t=1 floor) while MIRABEL's stays flat/soft at every t "
        "— the concentration itself is org-1-only; the t=2 wrinkle was "
        "noise.\"",
    "GRADED": "GRADED: \"any mix — the curves verbatim.\"",
    "operationalizations": "ALIVE t = the walk's every-step primary read at "
        "theta_t > 0.27; rays read ONLY at alive ts (post-death ts never "
        "read; org1's t=2 front rides as e195's COMMITTED post-kill read, "
        "context only, never adjudicated); ratio_t = D_kill(root,u_t)/"
        "D_kill(root,u0) vs each organism's OWN edge (ratio_0 = 1.0 by "
        "definition); KILL = the profile's first primary <= 0.27 "
        "downcrossing on the static grid 0.05..3.00, linear-in-D "
        "interpolated; a ray CONCENTRATES at t iff ratio_t < 0.70 (e198's "
        "frozen flight bar); a None kill within [0,3.0] is SOFT "
        "(unresolved-high; the floor ratio 3.0/edge is reported, never "
        "adjudicated); 'the ratio falling with t (a deepening concentration "
        "once rotation begins)' = the curve reaches a concentrated alive t "
        "AND every resolved ratio strictly after the first concentrated t "
        "stays <= 0.70 (vacuous with no later alive t — the ORG1-ONLY bar's "
        "own 'stays at its t=1 floor' case); 'shifted one step earlier' = "
        "org1's first concentrated t == MIRABEL's first concentrated t - 1; "
        "'MIRABEL stays flat/soft at every t' = no alive t of MIRABEL "
        "concentrates; composite order frozen ONSET-COMMON -> "
        "ONSET-ORG1-ONLY -> GRADED.",
    "registered_prediction": "T162's own fork — the t=2 wrinkle (MIRABEL's "
        "u2 at 0.427) suggests the concentration is a WHEN (COMMON), but "
        "e198's t=1 read was not marginal (u1 never kills <= 3.0) and "
        "org1's single-step floor could be its whole story (ORG1-ONLY); "
        "either verdict is informative; no bar shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE EXTENSION ARM IS VACUOUS (disclosed, the dispatch's own clause): "
    "both organisms' committed walks already run to their deaths — org1 "
    "dies at t=2 (opt2/e194/e195 committed; re-verified in this cell's "
    "rebuilt journal, gated G_REPRO), MIRABEL dies at t=3 (e198 committed; "
    "re-verified likewise) — both before the t=6 horizon, so there are NO "
    "missing steps to walk; the walks are rebuilt bit-exactly as the ray "
    "factories (their machinery verbatim) and stop at death.",
    "org1's walk is lab/e195_rotated_ray.py's walk_arm VERBATIM arithmetic "
    "with extend_past_kill=0 (the walk stops AT the kill step — this cell "
    "reads no post-kill gradient; e195's u2 was read on the post-kill "
    "continuation THERE and is loaded here COMMITTED as context only); "
    "org1's front gate covers e194's committed trace rows t=0,1 only.",
    "MIRABEL's walk is lab/e198_flight_arch.py's walk_rebuild VERBATIM "
    "(n_steps=3, its ZEPHYRA bracket at step 1 for G_REPRO, its MIRABEL "
    "densified kill bracket at step 3).",
    "Thread conventions are per organism (each gate set's own bit-tightness "
    "convention): org1's phase runs threads 4 (the e185-era reduction "
    "order), MIRABEL's phase switches to threads 8 (e193b's committed dial "
    "reproduces bit-tight under it) — switched between phases, never mixed "
    "within one.",
    "The rays are REBUILT fresh from the re-walked refresh gradients and "
    "identity-gated in TIERS (frozen before the full compute): BIT (md5 / "
    "1e-9) where today's process reproduces the parents' arithmetic — "
    "org1's ENTIRE chain achieves BIT (0.00e+00 on every gate incl. the "
    "saved opt2 checkpoint md5 and e195's profile rows) — and a disclosed "
    "TEXTURE tier for the MIRABEL chain (e196's G_S1CK precedent: cos "
    "> 0.99999 vs checkpoint tensors, mutual-geometry cosines vs e198's "
    "committed ray geometry within 1e-3, walk rows < 1e-3, profile rows "
    "< 1e-3, D_kills < 1e-2), because this process drifts the e193b/e198 "
    "arithmetic at ~1e-7 fp32 (e198 itself reproduced e193b at 0.0 — an "
    "environmental change since, disclosed in every affected gate with its "
    "achieved tier stamped); the adjudicated onset numbers are the fresh "
    "measurements, with the committed-number crosscheck in the "
    "adjudication showing the verdict identical under both sets.",
    "ce_r (the second currency) decimated to 0.2-multiples + D=0 (e195's "
    "decimation); the g-12 battery is the ruler and is read at EVERY grid "
    "point; dual displacement currency (D and per-coordinate RMS / "
    "1655.014) on every row; MIRABEL's rows carry e198's three co-rulers "
    "(reported, never adjudicated).",
    "Light reads only (batteries + ce_r; NO alignment reads — alignment "
    "predicts nothing (T157's lesson at n=4 lineages) and the onset curve "
    "is ruler-based; the dual-estimator lesson carried by G_SAMEPOINT's "
    "machinery gate on org1 and by the absence of any point-ambiguous "
    "read); CPU-ONLY, n=1 organism/fact per side, seed lineage 10902.",
    "Smoke mode trims: grid {0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, ce "
    "at anchors only (verdict stamped SMOKE; nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e195_rotated_ray.py's instruments (whose own provenance is
# lab/e194_sign_front.py / lab/opt2_density.py / e185 — the e176n lineage)
# and lab/e198_flight_arch.py's (e193b/e196's). Copied rather than imported
# to own the device policy and the bit-exact arithmetic.

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
    """fp64 cosine (the chart's estimator precision)."""
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def sign_update(g: torch.Tensor, step_l2: float):
    """opt2/e193b's sign_update VERBATIM: delta = -step_l2*sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT-READ MACHINERY (e195's fact_grad): gradient of the fact
    battery's mean log p(Z) readout at the twin's current weights. Used here
    ONLY for org1's G_SAMEPOINT machinery gate (the chart's committed
    anchors); no alignment claim is adjudicated in this cell."""
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
    pts = [(d0, v0)] + [(r["D"], r["gm" if "gm" in r else "gm12"]) for r in dens] \
        + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


def _dk_ok(meas, committed):
    """None-safe D_kill equality vs a committed value."""
    return (meas is None and committed is None) or \
        (meas is not None and committed is not None
         and abs(meas - committed) < DKILL_TOL)


def _dk_texture_ok(meas, committed):
    """None-safe D_kill equality at the TEXTURE tier (MIRABEL's chain)."""
    return (meas is None and committed is None) or \
        (meas is not None and committed is not None
         and abs(meas - committed) < DKILL_TEXTURE_TOL)


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) + any upcrosses (context)."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm12"] <= SHUT_BAR and a["gm12"] > SHUT_BAR:
            edge = interp_d_kill(a["gm12"], b["gm12"], a["D"], b["D"])
            break
    ups = [(rows[i - 1]["D"], rows[i]["D"]) for i in range(1, len(rows))
           if rows[i]["gm12"] > SHUT_BAR and rows[i - 1]["gm12"] <= SHUT_BAR]
    return edge, ups


# ------------------------------------------------------------------ org1's walk
# PROVENANCE: lab/e195_rotated_ray.py's walk_arm VERBATIM arithmetic (whose
# own provenance is lab/e194_sign_front.py / opt2 / opt1c / e185), with the
# unused (rd, write_partial, stub) plumbing args dropped and
# extend_past_kill=0 — this cell reads no post-kill gradient.

def build_batch(gen, anchor, train_ids):
    """opt2's draw order VERBATIM: aj then rj, anchors + random windows."""
    aj = torch.randint(anchor.shape[0], (ANCH_BS,), generator=gen)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=gen)
    anc = anchor[aj]
    rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
    return torch.cat([anc[:, :-1], rnd[:, :-1]], 0), \
           torch.cat([anc[:, 1:], rnd[:, 1:]], 0), rj


def walk_arm_org1(net0, anchor, train_ids, itos, gm12_ids, zid, theta0,
                  u_g, u_sign, step_l2):
    """A k=1 sign walk on the licensed stream (e195's walk_arm VERBATIM,
    extend_past_kill=0): every step refreshes the front (post-clip gradient
    on that step's batch), steps -step_l2*sign(g)/||sign(g)||, reads g-12
    every step, asserts matched-L2, densifies the kill bracket (opt1c)."""
    net = copy.deepcopy(net0)
    net.train()
    evl = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    n_steps = 16
    journal, front_trace, kill = [], [], None
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_front = None
    prev_delta = None
    t0a = time.time()
    step = 0
    while step < n_steps:
        step += 1
        post_kill = kill is not None
        if post_kill:            # extend_past_kill=0: stop at the kill step
            break
        x, y, rj = build_batch(gen, anchor, train_ids)
        x_md5 = hashlib.md5(x.contiguous().numpy().tobytes()).hexdigest()
        if step in E185_XHASH:
            assert x_md5 == E185_XHASH[step], \
                f"step-{step} batch md5 diverged from the licensed stream"
        for w in torch.stack([train_ids[s: s + BLOCK] for s in rj]):
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph += 1
        prev = flat_params(net)
        refresh = ((step - 1) % 1 == 0)     # k=1: every step = opt2's a_sign
        if refresh:
            logits, _ = net(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            net.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            g_t = torch.cat([p.grad.detach().reshape(-1)
                             for p in net.parameters()]).clone()   # post-clip
            delta = sign_update(g_t, step_l2)
            ce_batch, gnorm_row = float(loss.item()), gnorm
            front = torch.sign(g_t)
            frow = {
                "step": step, "front_state": step - 1,
                "vs_g_ray": cos64(front, u_g),          # THE overlap read
                "vs_fresh_g": cos64(front, g_t),
                "vs_static_sign": cos64(front, u_sign),
                "vs_prev_front": (cos64(front, prev_front)
                                  if prev_front is not None else None),
            }
            prev_front = front.clone()
            g_fronts.append(g_t.clone())                # ADDITIVE stash
        else:                                             # unreachable (k=1)
            delta = prev_delta.clone()
            ce_batch, gnorm_row, frow = None, None, None
        l2dev = abs(float(torch.norm(delta.double())) - step_l2)
        assert l2dev < L2_ASSERT_TOL, \
            f"per-step L2 dev {l2dev:.2e} — the matched-L2 assertion"
        max_l2_dev = max(max_l2_dev, l2dev)
        load_flat(net, prev + delta)
        cur = flat_params(net)
        endpoints[step] = cur.clone()                   # ADDITIVE stash
        d_cum = cur - theta0
        cum_disp = float(torch.norm(d_cum))
        row = {"step": step, "refresh": refresh, "post_kill": post_kill,
               "ce_batch": ce_batch, "preclip_gnorm": gnorm_row,
               "step_disp": float(torch.norm(delta)), "l2_dev": l2dev,
               "cum_disp": cum_disp, "x_md5": x_md5,
               "delta_vs_death_ray": cos64(delta, -u_g),
               "delta_vs_prev_delta": (cos64(delta, prev_delta)
                                       if prev_delta is not None else None),
               "cum_vs_g_ray": cos64(d_cum, u_g),
               "cum_vs_static_sign": cos64(d_cum, u_sign)}
        prev_delta = delta.clone()
        # every-step g-12 (the kill ruler)
        evl.load_state_dict({k_: v.detach().cpu().clone()
                             for k_, v in net.state_dict().items()})
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        row["gm12"] = gz["mean_pz"]
        row["frac_argmax_z"] = gz["frac_argmax_z"]
        if frow is not None:
            frow["gm12_at_state"] = journal[-1]["gm12"] if journal else None
            front_trace.append(frow)
        journal.append(row)
        log(f"  [org1 s{step:2d}]{' POSTKILL' if post_kill else ''}"
            f" g-12 {row['gm12']:.10f} D {cum_disp:.10f}"
            + (f" ov(g-ray) {frow['vs_g_ray']:+.4f}" if frow else ""))
        # KILL bracket (opt1c's convention; computed once)
        if kill is None and row["gm12"] <= SHUT_BAR:
            dens = []
            for f_ in DENSIFY_F:
                pt_flat = prev + f_ * (cur - prev)
                load_flat(evl, pt_flat)
                evl.eval()
                gzd = battery_cell(evl, gm12_ids, zid)
                dens.append({"f": f_, "gm12": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            v1_prev = journal[-2]["gm12"] if len(journal) >= 2 else 0.9155886173248291
            d1_prev = journal[-2]["cum_disp"] if len(journal) >= 2 else 0.0
            kill = {"kind": "kill", "step": step,
                    "gm12_at_kill": row["gm12"],
                    "D_kill_raw": interp_d_kill(v1_prev, row["gm12"],
                                                d1_prev, cum_disp),
                    "D_kill": dens_d_kill(v1_prev, row["gm12"],
                                          d1_prev, cum_disp, dens),
                    "dens": dens,
                    "reason": "g-12 <= SHUT at an every-step read"}
            log(f"  [org1] KILL at s{step}: D_kill(dens) {kill['D_kill']:.10f}")
        elif kill is None and cum_disp >= 5.0:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET 5.0 alive",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
        if step >= 16 and kill is None:
            kill = {"kind": "cap", "step": step, "reason": "step cap 16",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
    return {"journal": journal, "front_trace": front_trace, "stop": kill,
            "max_l2_dev": max_l2_dev, "zeph": zeph,
            "g_fronts": g_fronts, "endpoints": endpoints,
            "train_seconds": round(time.time() - t0a, 1)}


# ------------------------------------------------------------------ mirabel's walk
# PROVENANCE: lab/e198_flight_arch.py's walk_rebuild VERBATIM (whose own
# provenance is lab/e193b_two_fact.py's run_arm — e193b's a_sign).

R_EVAL_XY_GLOBAL = None   # set in main (the walk's step-1 ce_r co-read)


def walk_rebuild_mir(net0, anchor_neutral, train_ids, itos, mir_ids,
                     mir_geos, zid_m, zeph_ids, zid_z, theta0, step_l2,
                     root_gm_m, root_gm_z, n_steps=3):
    """e198's walk_rebuild VERBATIM arithmetic: e193b's a_sign arm (draw
    order, name-free verify on ZEPH and the FULL nonce MIRABEL, post-clip
    gradients, sign_update with fp64 support norm), MIRABEL-primary
    every-step readout + all four MIRABEL geometries per step, the ZEPHYRA
    bracket at step 1 (G_REPRO) and the MIRABEL densified kill bracket at
    ITS killing step."""
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
        assert l2dev < L2_ASSERT_TOL, f"per-step L2 dev {l2dev:.2e}"
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
        log(f"  [mir s{step}]{' POSTKILL' if post_kill else ''} "
            f"MIRABEL g-12 {row['gm']:.10f} D {cum_disp:.10f} ce "
            f"{row['ce_batch']:.6f}")
        if step == 1:
            # the ZEPHYRA bracket, e193b's arithmetic VERBATIM (G_REPRO)
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
            log(f"  [mir] ZEPHYRA bracket re-derived: D_kill "
                f"{kill_z['D_kill']:.10f} (committed "
                f"{E193B_ASIGN_STOP['D_kill']:.10f})")
            load_flat(evl_a, cur)
            evl_a.eval()
        if kill_m is None and row["gm"] <= SHUT_BAR:
            # the MIRABEL bracket (the walk's own kill)
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm_m, "cum_disp": 0.0}
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
            log(f"  [mir] MIRABEL KILL at s{step}: D_kill(dens) "
                f"{kill_m['D_kill']:.10f} — walk stops (read-only horizon 3)")
            load_flat(evl_a, cur)
            evl_a.eval()
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "kill_mirabel": kill_m, "kill_zephyra": kill_z,
            "x_hashes": x_hashes, "max_l2_dev": max_l2_dev,
            "g_fronts": g_fronts, "endpoints": endpoints,
            "seconds": round(time.time() - t0a, 1)}


# ------------------------------------------------------------------ profiles

def static_profile_org1(evl, anchor_flat, u, dgrid, gm12_ids, zid,
                        r_eval_xy, ce_ds):
    """e195's static_profile VERBATIM: theta_D = anchor - D*u, g-12 battery
    at every point + ce_r at the frozen decimated Ds; dual currency."""
    rows = []
    for D in dgrid:
        thD = anchor_flat - D * u
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        row = {"D": float(D), "rms": float(D) / RMS_DENOM,
               "gm12": gz["mean_pz"],
               "frac_argmax_z": gz["frac_argmax_z"],
               "disp_check": float(torch.norm(thD - anchor_flat))}
        if ce_ds is not None and any(abs(D - c) < 1e-9 for c in ce_ds):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
        rows.append(row)
    return rows


def static_profile_mir(evl, anchor_flat, u, dgrid, mir_primary, mir_corulers,
                       zid_m, r_eval_xy, ce_ds):
    """e198's static_profile VERBATIM: MIRABEL primary + the three co-rulers
    on every row, ce_r at the frozen decimated Ds; dual currency."""
    rows = []
    for D in dgrid:
        thD = anchor_flat - D * u
        disp_check = float(torch.norm(thD - anchor_flat))
        load_flat(evl, thD)
        evl.eval()
        gm = battery_cell(evl, mir_primary, zid_m)
        row = {"D": float(D), "rms": float(D) / RMS_DENOM,
               "gm12": gm["mean_pz"], "frac_argmax_z": gm["frac_argmax_z"],
               "disp_check": disp_check, "disp_dev": abs(disp_check - float(D))}
        for j in mir_corulers:
            row[f"g{j:+d}"] = battery_cell(evl, mir_corulers[j],
                                           zid_m)["mean_pz"]
        if ce_ds is not None and any(abs(D - c) < 1e-9 for c in ce_ds):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e199_smoke" if SMOKE else "e199")
    log(f"E199 THE CONCENTRATION'S ONSET CURVE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads 4 for org1, then 8 for MIRABEL), "
        f"progressive writes, n=1 organism/fact per side, stream seed "
        f"{FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    parents = {}
    for name, p in (("opt2", OPT2_METRICS), ("opt1c", OPT1C_METRICS),
                    ("opt1", OPT1_METRICS), ("e192", E192_METRICS),
                    ("e_chart", ECHART_METRICS), ("e194", E194_METRICS),
                    ("e195", E195_METRICS), ("e193b", E193B_METRICS),
                    ("e196", E196_METRICS), ("e197", E197_METRICS),
                    ("e198", E198_METRICS)):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
        parents[name] = json.loads(p.read_text(encoding="utf-8"))
    opt2m, opt1m, e192m = parents["opt2"], parents["opt1"], parents["e192"]
    chartm, e194m = parents["e_chart"], parents["e194"]
    e195m, e193bm = parents["e195"], parents["e193b"]
    e196m, e197m, e198m = parents["e196"], parents["e197"], parents["e198"]
    # hard-bind the committed references (asserts catch committed-file drift)
    assert abs(opt2m["arms"]["a_sign"]["stop"]["D_kill"]
               - OPT2_PATH_DKILL) < 1e-12
    assert abs(opt1m["arms"]["a0_adamw_ref"]["tstar"]["D_at_kill"]
               - ORG1_A0_DKILL) < 1e-12
    assert abs(e192m["adjudication"]["bars"]["TERRAIN_ONE_PICTURE"]
               ["sign_kill_D"] - E192_SIGN_EDGE_COARSE) < 1e-12
    chart_same = chartm["gates"]["G_W024"]
    assert abs(chart_same["cos_adamview_m12_at_t0_samepoint"]
               - W024_ADAM_SAMEPOINT) < 1e-12
    assert abs(chart_same["cos_mov_m12_at_t0_samepoint"]
               - W024_RAW_SAMEPOINT) < 1e-12
    e194_ft = e194m["phaseA_front_overlap"]["front_trace"]
    assert abs(e194m["phaseB_static_fine"]["refined_static_edge"]
               - E194_REFINED_EDGE) < 1e-12
    for i, ref in enumerate(E194_FRONTS):
        assert e194_ft[i]["front_state"] == ref["front_state"]
        assert abs(e194_ft[i]["vs_g_ray"] - ref["vs_g_ray"]) < 1e-12
        assert abs(e194_ft[i]["vs_static_sign"] - ref["vs_static_sign"]) < 1e-12
    # e195's committed root panel + rays (org1's onset anchors)
    for k, v in (("root_u0", E195_ORG1["root_u0_D_kill"]),
                 ("root_u1", E195_ORG1["root_u1_D_kill"]),
                 ("root_u2", E195_ORG1["root_u2_D_kill"])):
        got = e195m["profiles"][k]["D_kill"]
        assert (got is None and v is None) or abs(got - v) < 1e-12, \
            f"e195 {k} drift"
    for rk, md5 in (("u0", E195_ORG1["u0_md5"]), ("u1", E195_ORG1["u1_md5"]),
                    ("u2", E195_ORG1["u2_md5"])):
        assert e195m["profiles"][f"root_{rk}"]["u_md5"] == md5, \
            f"e195 root_{rk} md5 drift"
    assert e195m["adjudication"]["verdict"] == "FLEEING-IS-LETHAL"
    # e193b + e198 committed (MIRABEL's onset anchors)
    assert e193bm["organism"]["root_md5"] == MIR_ROOT_MD5
    assert abs(e193bm["organism"]["step_l2_measured"] - MIR_STEP_L2) < 1e-12
    dial = e193bm["rulers"]["root_dial_seven_geos"]
    for f in (FACT1, FACT2):
        for j in READ_GEOS:
            assert abs(dial[f][f"g{j:+d}"] - E193B_DIAL[f][f"g{j:+d}"]) < 1e-12
    assert abs(e193bm["rulers"]["ce_r"] - E193B_DIAL["ce_r"]) < 1e-12
    asign = e193bm["ladder"]["arms"]["a_sign"]
    assert abs(asign["D_kill"] - E193B_ASIGN_STOP["D_kill"]) < 1e-12
    assert len(e193bm["profiles"]["R2_SIGN"]) == 27
    for k, v in (("root_u0", E198_MIR["root_u0_D_kill"]),
                 ("root_u1", E198_MIR["root_u1_D_kill"]),
                 ("root_u2", E198_MIR["root_u2_D_kill"])):
        got = e198m["adjudication"]["root_panel"][k]
        gotp = e198m["profiles"][k]["D_kill"]
        assert (got is None and v is None) or abs(got - v) < 1e-12, \
            f"e198 {k} drift"
        assert (gotp is None and v is None) or abs(gotp - v) < 1e-12, \
            f"e198 profile {k} drift"
    for rk, md5 in (("u0", E198_MIR["u0_md5"]), ("u1", E198_MIR["u1_md5"]),
                    ("u2", E198_MIR["u2_md5"])):
        assert e198m["profiles"][f"root_{rk}"]["u_md5"] == md5, \
            f"e198 root_{rk} md5 drift"
    assert e198m["adjudication"]["verdict"] == "BIOGRAPHY-CARRIES"
    for k, v in E196_ROOT_DKILLS.items():
        assert abs(e196m["adjudication"]["root_panel"][k] - v) < 1e-12, \
            f"e196 {k} drift"
    e197b = e197m["adjudication"]["bars"]["ALIVE_WINDOW_CAUSES"]["detail"]
    assert abs(e197b["flight_concentration"]["u1_D_kill"]
               - E197_ALIVE["u1_D_kill"]) < 1e-12
    log("parents: opt2/e194/e195 (org1's walk + fronts + root panel), "
        "e192/e_chart/opt1 (org1's direction + estimator anchors), e193b/"
        "e198 (MIRABEL's organism + walk + root panel), e196/e197 (the "
        "four-lineage ledger) — loaded COMMITTED, never rerun")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e199_onset_curve",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE,
                               "torch": torch.__version__,
                               "threads_org1": 4, "threads_mirabel": 8},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    # =====================================================================
    # the shared protocol rebuild (one corpus; both organisms' gates)
    # =====================================================================
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid_z, zid_m = stoi["Z"], stoi["M"]
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
    mix = {f: {"FLORIZEL": sum(1 for _, h in install_occ[f] if h == "FLORIZEL"),
               "ELIZABETH": sum(1 for _, h in install_occ[f]
                                if h == "ELIZABETH")}
           for f in (FACT1, FACT2)}
    G_SPLICE = {"install_mix": mix, "n_filtered_total": len(host_occ),
                "pass": bool(mix[FACT1] == E193B_SPLICE_MIX[FACT1]),
                "note": "ZEPHYRA's mix {19,41} = battery identity (both "
                        "organisms' gate); MIRABEL's {16,44} reported "
                        "hard-bound (e193b's own disclosure)",
                "mirabel_mix_match_committed":
                    bool(mix[FACT2] == E193B_SPLICE_MIX[FACT2])}
    assert G_SPLICE["pass"] and G_SPLICE["mirabel_mix_match_committed"], \
        f"splice drift {G_SPLICE}"

    # org1's battery (e185's 3-geometry form) + MIRABEL's (e198's 7)
    org1_bat = {}
    for j in ORG1_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ[FACT1]]
        org1_bat[j] = torch.stack([corpus.encode(c) for c in cs])
    org1_gm12_ids = org1_bat[-12]
    G_BATTERY_ORG1 = {
        "shapes": {str(j): list(org1_bat[j].shape) for j in ORG1_GEOS},
        "pass": bool(list(org1_bat[-12].shape) == [60, PRE - 12]
                     and list(org1_bat[0].shape) == [60, PRE]
                     and list(org1_bat[12].shape) == [60, PRE + 12]),
        "note": "org1's install-60 battery at ctx offsets {-12,0,+12}, "
                "e185's convention verbatim (e195's gate)",
    }
    assert G_BATTERY_ORG1["pass"], f"org1 battery drift: {G_BATTERY_ORG1}"
    bat_ids = {f: {} for f in (FACT1, FACT2)}
    for f in (FACT1, FACT2):
        for j in READ_GEOS:
            cs = [train_text[p - PRE - j: p] for p, _ in install_occ[f]]
            bat_ids[f][j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY_MIR = {
        "shapes": {f: {str(j): list(bat_ids[f][j].shape) for j in READ_GEOS}
                   for f in (FACT1, FACT2)},
        "pass": bool(all(list(bat_ids[f][j].shape) == [60, PRE + j]
                         for f in (FACT1, FACT2) for j in READ_GEOS)),
        "note": "MIRABEL's install-60 battery at the e157 dial's seven read "
                "geometries — e193b's gate verbatim (e198's)",
    }
    assert G_BATTERY_MIR["pass"], f"mirabel battery drift: {G_BATTERY_MIR}"
    mir_primary = bat_ids[FACT2][RULER_J]
    mir_corulers = {j: bat_ids[FACT2][j] for j in CO_RULERS_J}
    mir_geos = {j: bat_ids[FACT2][j] for j in (RULER_J,) + CO_RULERS_J}
    zeph_primary_g0 = bat_ids[FACT1][0]      # ZEPHYRA g+0 = e193b's ladder ruler
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    global R_EVAL_XY_GLOBAL
    R_EVAL_XY_GLOBAL = r_eval_xy

    # the neutral stream (e170 VERBATIM via e185/e193/e193b/e195/e198)
    arng = random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN_ORG1):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                         "in [s, s+257) — e170 VERBATIM (both organisms')"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

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
                "note": "the seed-10902 stream construction (net-independent) "
                        "md5-matches e185's stored hashes — shared by both "
                        "organisms' walks"}
    assert G_STREAM["pass"], "stream construction diverged from e185"

    stub["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                          "G_BATTERY_ORG1": G_BATTERY_ORG1,
                          "G_BATTERY_MIR": G_BATTERY_MIR,
                          "G_ANCHOR": G_ANCHOR, "G_STREAM": G_STREAM})
    write_partial("shared cell gates passed (both organisms' constructions)")
    log("shared cell gates passed (namefree/splice/batteries/anchor/stream)")

    ce_ds = ([0.0] + [round(CE_EVERY * i, 2)
                      for i in range(1, int(3.0 / CE_EVERY) + 1)]
             if CE_EVERY else [0.0])

    def gate_profile(rows, committed_rows, gm_key, label, tol=G_STATIC_TOL):
        """Row-wise crosscheck vs a committed family (e195's G_ROOTPROF form)."""
        by_D = {round(r["D"], 6): r for r in committed_rows}
        xc = []
        for row in rows:
            key = round(row["D"], 6)
            if key in by_D:
                xc.append({"D": key, "gm_measured": row["gm12"],
                           "gm_committed": by_D[key][gm_key],
                           "abs_diff": abs(row["gm12"] - by_D[key][gm_key])})
        return {"label": label, "n_shared": len(xc), "rows": xc,
                "max_abs_diff": (max(r["abs_diff"] for r in xc) if xc else None),
                "tol": tol,
                "pass": bool(xc and all(r["abs_diff"] < tol for r in xc))}

    # =====================================================================
    # PHASE A — ORG1 (e195's cell; threads 4)
    # =====================================================================
    log("=" * 78)
    log("PHASE A — ORG1 (the e131 root; e195's gates verbatim; threads 4)")
    assert torch.get_num_threads() == 4
    net1 = load_root(CKPT_DIR / ORG1_ROOT_CK)
    st1_raw = torch.load(CKPT_DIR / ORG1_ROOT_CK, map_location="cpu",
                         weights_only=False)
    org1_root_meta = E43.jsonable(st1_raw.get("meta", {}))
    theta0_1 = flat_params(net1)
    assert int(theta0_1.numel()) == N_PARAM, \
        f"params {theta0_1.numel()} != {N_PARAM}"

    evl1 = copy.deepcopy(net1)
    root_cells_1 = {
        "gm12": battery_cell(evl1, org1_gm12_ids, zid_z)["mean_pz"],
        "g0": battery_cell(evl1, org1_bat[0], zid_z)["mean_pz"],
        "gp12": battery_cell(evl1, org1_bat[12], zid_z)["mean_pz"],
        "ce_r": ce_fixed_cpu(evl1, *r_eval_xy),
    }
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r"}
    refs1 = {keymap[k]: v for k, v in E151_ROOT.items() if k in keymap}
    rdiffs1 = {k: root_cells_1[k] - refs1[k] for k in refs1}
    rmax1 = max(abs(v) for v in rdiffs1.values())
    G_ROOT_ORG1 = {"cells": root_cells_1, "refs": refs1, "diffs": rdiffs1,
                   "max_abs_diff": rmax1, "bit_tol": G_BIT_TOL,
                   "tol": G_FALLBACK_TOL, "bit": bool(rmax1 < G_BIT_TOL),
                   "pass": bool(rmax1 < G_FALLBACK_TOL),
                   "note": "org1's G_ROOT (e195's): the e131 root's cells vs "
                           "e151's before-battery"}
    log(f"G_ROOT_ORG1 (vs e151 before-cells): max|diff| {rmax1:.2e}: "
        + ("PASS" if G_ROOT_ORG1["pass"] else "FAIL"))
    if not G_ROOT_ORG1["pass"]:
        raise RuntimeError("org1 root gate FAILED")

    # G_T0_ORG1: the t=0 wash gradient + AdamW step (opt1's A0 row)
    gen1 = torch.Generator().manual_seed(FREEZE_SEED)
    x1, y1, _ = build_batch(gen1, anchor_neutral, train_ids)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(net1)
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
    g0_t0 = torch.cat([p.grad.detach().reshape(-1)
                       for p in tw.parameters()]).clone()   # post-clip
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0_1))
    del tw, optw, logits
    G_T0_ORG1 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1, "ce_batch_committed": ORG1_CE1,
        "preclip_gnorm_measured": gn1, "preclip_gnorm_committed": ORG1_GN1,
        "adamw_step1_L2_measured": disp1,
        "adamw_step1_L2_committed": ORG1_STEP_L2,
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and abs(ce1 - ORG1_CE1) < G_FALLBACK_TOL
                     and abs(gn1 - ORG1_GN1) < G_FALLBACK_TOL
                     and abs(disp1 - ORG1_STEP_L2) < G_FALLBACK_TOL),
        "note": "org1's G_T0 (e195's): the t=0 bit-gate vs opt1's committed "
                "A0 row — the measured step L2 1.6543 is THIS organism's "
                "walk's step size (never ported)",
    }
    log(f"G_T0_ORG1: md5 "
        f"{'match' if G_T0_ORG1['step1_x_md5_match_e185'] else 'DRIFT'}, "
        f"CE |d| {abs(ce1 - ORG1_CE1):.2e}, L2 |d| "
        f"{abs(disp1 - ORG1_STEP_L2):.2e}: "
        + ("PASS" if G_T0_ORG1["pass"] else "FAIL"))
    if not G_T0_ORG1["pass"]:
        raise RuntimeError("org1 t=0 bit-identity gate FAILED — abort")
    STEP_L2_1 = disp1                       # measured, never ported
    assert abs(STEP_L2_1 - ORG1_STEP_L2) < G_FALLBACK_TOL

    # G_SAMEPOINT_ORG1 (T150's dual-estimator machinery gate, e195's)
    evl_root1 = copy.deepcopy(net1)
    evl_root1.eval()
    root_m12_grad = fact_grad(evl_root1, org1_gm12_ids, zid_z)
    sp_sign = cos64(-torch.sign(g0_t0), root_m12_grad)
    sp_raw = cos64(-g0_t0, root_m12_grad)
    G_SAMEPOINT_ORG1 = {
        "sign_arm_s1_matched_point_m12_measured": sp_sign,
        "committed_chart_samepoint": W024_ADAM_SAMEPOINT,
        "d_sign": abs(sp_sign - W024_ADAM_SAMEPOINT),
        "raw_sanity_measured": sp_raw,
        "committed_chart_raw": W024_RAW_SAMEPOINT,
        "d_raw": abs(sp_raw - W024_RAW_SAMEPOINT),
        "tol": G_SAMEPOINT_TOL,
        "pass": bool(abs(sp_sign - W024_ADAM_SAMEPOINT) < G_SAMEPOINT_TOL
                     and abs(sp_raw - W024_RAW_SAMEPOINT) < G_SAMEPOINT_TOL),
        "note": "e195's G_SAMEPOINT VERBATIM (the dual-estimator lesson "
                "carried as machinery; no alignment read adjudicates here)",
    }
    log(f"G_SAMEPOINT_ORG1: sign {sp_sign:+.6f} vs chart "
        f"{W024_ADAM_SAMEPOINT:+.6f}: "
        + ("PASS" if G_SAMEPOINT_ORG1["pass"] else "FAIL"))
    if not G_SAMEPOINT_ORG1["pass"]:
        raise RuntimeError("org1 same-point machinery gate FAILED — abort")

    # G_U191 + G_R2DIR (org1's committed directions)
    uck = torch.load(CKPT_DIR / ORG1_DIR_CK, map_location="cpu",
                     weights_only=False)
    u_g = uck["u"].detach().clone().float()
    fresh_u = g0_t0 / torch.norm(g0_t0)
    org1_root_md5 = hashlib.md5(theta0_1.numpy().tobytes()).hexdigest()
    G_U191 = {
        "loaded_u_md5_match_fresh": bool(hashlib.md5(
            u_g.numpy().tobytes()).hexdigest() == hashlib.md5(
                fresh_u.numpy().tobytes()).hexdigest()),
        "fresh_cos64_loaded": cos64(fresh_u, u_g),
        "root_md5_match_checkpoint": bool(org1_root_md5 == uck.get("theta0_md5")),
        "note": "e195's G_U191: the death direction u_g = e191's committed "
                "static g-ray; the kill ray is DESCENT (-u_g)",
    }
    G_U191["pass"] = bool(G_U191["root_md5_match_checkpoint"]
                          and abs(float(torch.norm(u_g)) - 1.0) < 1e-5)
    u0_1 = (torch.sign(g0_t0) / torch.norm(torch.sign(g0_t0))).clone()
    u0_1_md5 = hashlib.md5(u0_1.numpy().tobytes()).hexdigest()
    G_R2DIR = {
        "u_sign_md5": u0_1_md5, "committed_e192_md5": E192_R2_U_MD5,
        "md5_match": bool(u0_1_md5 == E192_R2_U_MD5),
        "u_norm_fp32": float(torch.norm(u0_1)),
        "n_zero_g_coords": int((g0_t0 == 0).sum()),
        "note": "org1's u0 = the static sign(g_0) ray rebuilt from the gated "
                "t=0 gradient (e192's fp32-norm construction VERBATIM)",
    }
    G_R2DIR["pass"] = bool(G_R2DIR["md5_match"])
    log(f"G_U191 (death ray): "
        + ("PASS" if G_U191["pass"] else "FAIL")
        + f"; G_R2DIR (u0): md5 "
        + ("match" if G_R2DIR["md5_match"] else "DRIFT")
        + ": " + ("PASS" if G_R2DIR["pass"] else "FAIL"))
    if not (G_U191["pass"] and G_R2DIR["pass"]):
        raise RuntimeError("org1 committed-direction gates FAILED — abort")
    stub["gates"].update({"G_ROOT_ORG1": G_ROOT_ORG1, "G_T0_ORG1": G_T0_ORG1,
                          "G_SAMEPOINT_ORG1": G_SAMEPOINT_ORG1,
                          "G_U191": G_U191, "G_R2DIR": G_R2DIR})
    write_partial("org1 standard gates passed")

    # ---- org1's walk rebuilt (to its death; the ray factory)
    log("-" * 78)
    log("org1's walk rebuilt (e195's walk_arm VERBATIM, extend 0 — stops at "
        f"the kill; horizon t<={WALK_MAX_T} or death, whichever first)")
    walk1 = walk_arm_org1(net1, anchor_neutral, train_ids, itos,
                          org1_gm12_ids, zid_z, theta0_1, u_g, u0_1,
                          STEP_L2_1)
    j1 = walk1["journal"]
    assert walk1["stop"]["kind"] == "kill" and walk1["stop"]["step"] == 2, \
        f"org1's walk did not die at t=2 as committed: {walk1['stop']}"
    assert len(j1) == 2 and [r["step"] for r in j1] == [1, 2]
    s1r, s2r = j1[0], j1[1]

    # G_REPRO_ORG1: vs opt2's committed a_sign trajectory
    repro_rows_1 = {
        "s1_ce_batch": (s1r["ce_batch"], OPT2_S1["ce_batch"]),
        "s1_cum_disp": (s1r["cum_disp"], OPT2_S1["cum_disp"]),
        "s1_gm12": (s1r["gm12"], OPT2_S1["gm12"]),
        "s1_preclip_gnorm": (s1r["preclip_gnorm"], OPT2_S1["preclip_gnorm"]),
        "s2_ce_batch": (s2r["ce_batch"], OPT2_S2["ce_batch"]),
        "s2_cum_disp": (s2r["cum_disp"], OPT2_S2["cum_disp"]),
        "s2_gm12": (s2r["gm12"], OPT2_S2["gm12"]),
        "s2_preclip_gnorm": (s2r["preclip_gnorm"], OPT2_S2["preclip_gnorm"]),
        "D_kill": (walk1["stop"]["D_kill"], OPT2_PATH_DKILL),
    }
    dens_diffs_1 = [abs(a["gm12"] - b[1]) for a, b in
                    zip(walk1["stop"]["dens"], OPT2_DENS)]
    G_REPRO_ORG1 = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows_1.items()},
        "dens_gm12_max_abs_diff": max(dens_diffs_1),
        "tol": G_REPRO_TOL,
        "max_l2_dev": walk1["max_l2_dev"],
        "pass": bool(max(abs(vv[0] - vv[1]) for vv in repro_rows_1.values())
                     < G_REPRO_TOL and max(dens_diffs_1) < G_REPRO_TOL),
        "note": "org1's G_REPRO (e195's, minus the post-kill s3 row this "
                "cell never reads): the rebuilt walk reproduces opt2's "
                "committed a_sign trajectory + densified kill bracket "
                "bit-class",
    }
    log("G_REPRO_ORG1 (vs opt2 committed): max|diff| "
        f"{max(abs(vv[0] - vv[1]) for vv in repro_rows_1.values()):.2e}: "
        + ("PASS" if G_REPRO_ORG1["pass"] else "FAIL"))
    if not G_REPRO_ORG1["pass"]:
        stub["gates"]["G_REPRO_ORG1"] = G_REPRO_ORG1
        write_partial("CONTROL FAILURE — org1 walk provenance gate failed")
        raise RuntimeError("G_REPRO_ORG1 FAILED — abort")
    stub["gates"]["G_REPRO_ORG1"] = G_REPRO_ORG1

    # G_SAVEDSTATE_ORG1: the saved opt2 ckpt IS the walked t=2 state
    sck = torch.load(CKPT_DIR / ORG2_SIGN_CK, map_location="cpu",
                     weights_only=False)
    saved_net = TinyGPT(CFG1)
    saved_net.load_state_dict(sck["model"])
    saved_flat = flat_params(saved_net)
    walked_s2 = walk1["endpoints"][2]
    maxdiff_1 = float((saved_flat - walked_s2).abs().max())
    G_SAVEDSTATE_ORG1 = {
        "path": str(CKPT_DIR / ORG2_SIGN_CK),
        "meta_gate": {"experiment": sck["meta"].get("experiment"),
                      "arm": sck["meta"].get("arm"),
                      "steps": sck["meta"].get("steps"),
                      "input_seed": sck["meta"].get("input_seed"),
                      "match": bool(sck["meta"].get("experiment") == "opt2"
                                    and sck["meta"].get("arm") == "a_sign"
                                    and sck["meta"].get("steps") == 2
                                    and sck["meta"].get("input_seed")
                                    == FREEZE_SEED)},
        "walked_s2_flat_md5": hashlib.md5(
            walked_s2.numpy().tobytes()).hexdigest(),
        "saved_flat_md5": hashlib.md5(
            saved_flat.numpy().tobytes()).hexdigest(),
        "max_abs_param_diff": maxdiff_1, "tol": 1e-6, "pass": None,
        "note": "e195's G_SAVEDSTATE: the saved opt2 ckpt IS the walked "
                "t=2 endpoint — the death state's chain of custody",
    }
    G_SAVEDSTATE_ORG1["pass"] = bool(
        G_SAVEDSTATE_ORG1["meta_gate"]["match"]
        and (G_SAVEDSTATE_ORG1["walked_s2_flat_md5"]
             == G_SAVEDSTATE_ORG1["saved_flat_md5"] or maxdiff_1 < 1e-6))
    log(f"G_SAVEDSTATE_ORG1 (opt2_a_sign_s2.pt): walked-s2 vs saved "
        f"max|diff| {maxdiff_1:.2e}: "
        + ("PASS" if G_SAVEDSTATE_ORG1["pass"] else "FAIL"))
    if not G_SAVEDSTATE_ORG1["pass"]:
        raise RuntimeError("G_SAVEDSTATE_ORG1 FAILED — abort")
    stub["gates"]["G_SAVEDSTATE_ORG1"] = G_SAVEDSTATE_ORG1

    # org1's rays: u1 from the rebuilt walk's step-2 refresh gradient
    g0_w, g1_w = walk1["g_fronts"][0], walk1["g_fronts"][1]
    u1_1 = (torch.sign(g1_w) / torch.norm(torch.sign(g1_w))).clone()
    u1_1_md5 = hashlib.md5(u1_1.numpy().tobytes()).hexdigest()
    fresh_front_rows = [
        {"front_state": 0, "vs_g_ray": cos64(torch.sign(g0_w), u_g),
         "vs_static_sign": cos64(torch.sign(g0_w), u0_1),
         "vs_prev_front": None},
        {"front_state": 1, "vs_g_ray": cos64(torch.sign(g1_w), u_g),
         "vs_static_sign": cos64(torch.sign(g1_w), u0_1),
         "vs_prev_front": cos64(torch.sign(g1_w), torch.sign(g0_w))},
    ]
    front_diffs_1 = {}
    for fresh, ref in zip(fresh_front_rows, E194_FRONTS):
        for k in ("vs_g_ray", "vs_static_sign", "vs_prev_front"):
            if ref[k] is None:
                continue
            front_diffs_1[f"t{ref['front_state']}.{k}"] = \
                abs(fresh[k] - ref[k])
    G_FRONTS_ORG1 = {
        "fresh_front_rows": fresh_front_rows,
        "committed_front_rows": E194_FRONTS,
        "max_abs_diff": max(front_diffs_1.values()),
        "diffs": front_diffs_1, "tol": G_REPRO_TOL,
        "u1_md5": u1_1_md5, "u1_md5_match_e195": bool(
            u1_1_md5 == E195_ORG1["u1_md5"]),
        "pass": bool(max(front_diffs_1.values()) < G_REPRO_TOL
                     and u1_1_md5 == E195_ORG1["u1_md5"]),
        "note": "e195's G_FRONTS (rows t=0,1 — the walk stops at the kill): "
                "the fresh fronts reproduce e194's committed trace bit-class "
                "AND the rebuilt u1 md5-matches e195's registered ray",
    }
    log(f"G_FRONTS_ORG1 (t=0,1 vs e194 + u1 md5 vs e195): max|diff| "
        f"{max(front_diffs_1.values()):.2e}, u1 md5 "
        + ("match" if G_FRONTS_ORG1["u1_md5_match_e195"] else "DRIFT")
        + ": " + ("PASS" if G_FRONTS_ORG1["pass"] else "FAIL"))
    if not G_FRONTS_ORG1["pass"]:
        raise RuntimeError("G_FRONTS_ORG1 FAILED — abort")
    stub["gates"]["G_FRONTS_ORG1"] = G_FRONTS_ORG1

    # org1's alive ledger + death
    org1_ledger = [
        {"t": 0, "state": "theta_0 (the e131 root)", "ruler_read":
         root_cells_1["gm12"], "alive": bool(root_cells_1["gm12"] > SHUT_BAR),
         "ray": "u0 (read here)"},
        {"t": 1, "state": "theta_1 (the walk's step-1 endpoint)",
         "ruler_read": s1r["gm12"], "alive": bool(s1r["gm12"] > SHUT_BAR),
         "ray": "u1 (read here, at the ALIVE state)"},
        {"t": 2, "state": "theta_2 (the walk's kill step)",
         "ruler_read": s2r["gm12"], "alive": bool(s2r["gm12"] > SHUT_BAR),
         "ray": None,
         "note": "DEATH — the post-death front is NEVER read in this cell; "
                 "e195's committed u2 (read there on the post-kill "
                 "continuation) rides as context only"},
    ]
    log("org1 alive ledger: "
        + ", ".join(f"t{r['t']} {'ALIVE' if r['alive'] else 'DEAD'} "
                    f"{r['ruler_read']:.4f}" for r in org1_ledger))
    write_partial("org1 walk rebuilt + provenance gates passed")

    # ---- org1's ray profiles from the ROOT (recomputed + gated)
    log("-" * 78)
    log(f"org1's root ray profiles: u0/u1 x {len(D_GRID)}+ grid points "
        "(e195's static_profile VERBATIM)")
    evp1 = copy.deepcopy(net1)
    fam1 = {}
    grids1 = {"u0": sorted(set([0.0] + list(D_GRID) + [STEP_L2_1])),
              "u1": sorted(set([0.0] + list(D_GRID)))}
    for key, u in (("u0", u0_1), ("u1", u1_1)):
        tA = time.time()
        rows = static_profile_org1(evp1, theta0_1, u, grids1[key],
                                   org1_gm12_ids, zid_z, r_eval_xy, ce_ds)
        edge, ups = profile_d_kill(rows)
        gate = gate_profile(rows, e195m["profiles"][f"root_{key}"]["rows"],
                            "gm12", f"org1 root_u{key[1]} vs e195 committed")
        fam1[key] = {"rows": rows, "D_kill": edge, "any_upcross": ups,
                     "gate": gate, "seconds": round(time.time() - tA, 1)}
        stub["phases_partial"][f"org1_root_{key}"] = E43.jsonable(
            {"D_kill": edge, "gate_pass": gate["pass"],
             "max_abs_diff": gate["max_abs_diff"]})
        write_partial(f"org1 root_u{key[1]} profile complete")
        log(f"  org1 root_{key}: {len(rows)} pts in {fam1[key]['seconds']}s "
            f"— D_kill " + (f"{edge:.10f}" if edge is not None else "None")
            + f"; vs e195 committed {gate['n_shared']} shared pts max|diff| "
            f"{gate['max_abs_diff']:.2e}")
    # landing gate (e195's STEP_L2 convention on the u0 family)
    landing1 = next((r for r in fam1["u0"]["rows"]
                     if abs(r["D"] - STEP_L2_1) < 1e-9), None)
    G_PROF_ORG1 = {
        "u0_vs_e195": fam1["u0"]["gate"], "u1_vs_e195": fam1["u1"]["gate"],
        "u0_D_kill": fam1["u0"]["D_kill"],
        "u0_D_kill_committed": E195_ORG1["root_u0_D_kill"],
        "u1_D_kill": fam1["u1"]["D_kill"],
        "u1_D_kill_committed": E195_ORG1["root_u1_D_kill"],
        "step_l2_landing": ({"D": landing1["D"], "gm12": landing1["gm12"],
                             "expect": E192_A0_SIGNRAY_GM12,
                             "abs_diff": abs(landing1["gm12"]
                                             - E192_A0_SIGNRAY_GM12)}
                            if landing1 else None),
        "tol_rows": G_STATIC_TOL, "tol_dkill": DKILL_TOL,
        "dkill_check_skipped": bool(SMOKE),
        "pass": None,
        "note": "org1's profile machinery gate: both recomputed families "
                "reproduce e195's committed rows + D_kills; the u0 family's "
                "D=STEP_L2 landing reproduces e192's committed read "
                "(D_kill equality skipped in smoke — the trimmed grid "
                "changes the interpolation bracket)",
    }
    G_PROF_ORG1["pass"] = bool(
        fam1["u0"]["gate"]["pass"] and fam1["u1"]["gate"]["pass"]
        and (SMOKE or (
            _dk_ok(fam1["u0"]["D_kill"], E195_ORG1["root_u0_D_kill"])
            and _dk_ok(fam1["u1"]["D_kill"], E195_ORG1["root_u1_D_kill"])))
        and landing1
        and abs(landing1["gm12"] - E192_A0_SIGNRAY_GM12) < G_STATIC_TOL)
    log("G_PROF_ORG1 (rows + D_kills + landing vs committed): "
        + ("PASS" if G_PROF_ORG1["pass"] else "FAIL"))
    if not G_PROF_ORG1["pass"]:
        stub["gates"]["G_PROF_ORG1"] = G_PROF_ORG1
        write_partial("CONTROL FAILURE — org1 profile gates failed")
        raise RuntimeError("G_PROF_ORG1 FAILED — abort")
    stub["gates"]["G_PROF_ORG1"] = G_PROF_ORG1

    # org1's post-death context point (e195's committed u2; loaded, never
    # re-read, never adjudicated)
    org1_ctx = {
        "t": 2, "ray": "u2", "u_md5": E195_ORG1["u2_md5"],
        "D_kill": E195_ORG1["root_u2_D_kill"],
        "ratio_vs_own_u0": E195_ORG1["root_u2_D_kill"] / fam1["u0"]["D_kill"],
        "rows_committed": e195m["profiles"]["root_u2"]["rows"],
        "disclosure": "org1's t=2 front was read by e195 on the walk's "
                      "post-kill continuation (the state is DEAD, g-12 "
                      "9.8e-5): loaded COMMITTED as context, plotted open, "
                      "and NEVER adjudicated — this cell reads no "
                      "post-death gradient",
    }
    org1_onset = [
        {"t": 0, "ray": "u0", "u_md5": u0_1_md5,
         "D_kill": fam1["u0"]["D_kill"], "ratio": 1.0,
         "ratio_note": "definitional (u0 vs its own edge)",
         "state": "alive", "ruler_read": root_cells_1["gm12"]},
        {"t": 1, "ray": "u1", "u_md5": u1_1_md5,
         "D_kill": fam1["u1"]["D_kill"],
         "ratio": fam1["u1"]["D_kill"] / fam1["u0"]["D_kill"],
         "state": "alive", "ruler_read": s1r["gm12"]},
        {"t": 2, "ray": None, "D_kill": None, "ratio": None,
         "state": "death", "ruler_read": s2r["gm12"],
         "walk_D_kill": walk1["stop"]["D_kill"],
         "note": "the walk's densified kill inside the step-2 bracket"},
    ]
    log("org1 onset: "
        + "; ".join(f"t{r['t']} ratio "
                    + ("DEATH (walk kill "
                       f"{r.get('walk_D_kill', float('nan')):.4f})"
                       if r["state"] == "death" else f"{r['ratio']:.4f}")
                    for r in org1_onset))
    stub["phases_partial"]["org1_onset"] = E43.jsonable(
        {"onset": org1_onset, "context": {k: v for k, v in org1_ctx.items()
                                          if k != "rows_committed"}})
    write_partial("PHASE A complete (org1 onset curve assembled)")

    # =====================================================================
    # PHASE B — MIRABEL (e198's cell; threads 8)
    # =====================================================================
    log("=" * 78)
    log("PHASE B — MIRABEL (the e193b root; e198's gates verbatim; "
        "switching to threads 8 — e193b's bit-tightness convention)")
    torch.set_num_threads(8)
    net2 = load_root(CKPT_DIR / MIR_ROOT_CK)
    st2_raw = torch.load(CKPT_DIR / MIR_ROOT_CK, map_location="cpu",
                         weights_only=False)
    mir_root_meta = E43.jsonable(st2_raw.get("meta", {}))
    theta0_2 = flat_params(net2)
    assert int(theta0_2.numel()) == N_PARAM, \
        f"params {theta0_2.numel()} != {N_PARAM}"
    mir_root_md5 = hashlib.md5(theta0_2.numpy().tobytes()).hexdigest()

    evl2 = copy.deepcopy(net2)
    root_cells_2 = {f: {f"g{j:+d}": battery_cell(
        evl2, bat_ids[f][j], zid_m if f == FACT2 else zid_z)["mean_pz"]
        for j in READ_GEOS} for f in (FACT1, FACT2)}
    root_cells_2["ce_r"] = ce_fixed_cpu(evl2, *r_eval_xy)
    refs2 = {f: {f"g{j:+d}": E193B_DIAL[f][f"g{j:+d}"] for j in READ_GEOS}
             for f in (FACT1, FACT2)}
    refs2["ce_r"] = E193B_DIAL["ce_r"]
    rdiffs2 = {f: {k: root_cells_2[f][k] - refs2[f][k] for k in refs2[f]}
               for f in (FACT1, FACT2)}
    rdiffs2["ce_r"] = root_cells_2["ce_r"] - refs2["ce_r"]
    rmax2 = max(abs(v) for f in (FACT1, FACT2) for v in rdiffs2[f].values())
    rmax2 = max(rmax2, abs(rdiffs2["ce_r"]))
    G_ROOT_MIR = {"cells": E43.jsonable(root_cells_2),
                  "refs": E43.jsonable(refs2), "diffs": E43.jsonable(rdiffs2),
                  "max_abs_diff": rmax2, "bit_tol": G_BIT_TOL,
                  "tol": G_FALLBACK_TOL, "bit": bool(rmax2 < G_BIT_TOL),
                  "flat_md5": mir_root_md5,
                  "flat_md5_match_e193b_committed":
                      bool(mir_root_md5 == MIR_ROOT_MD5),
                  "pass": bool(rmax2 < G_FALLBACK_TOL
                               and mir_root_md5 == MIR_ROOT_MD5),
                  "note": "e198's G_ROOT VERBATIM (two-fact form): the root "
                          "reproduces e193b's committed 15-cell dial "
                          "BIT-TIGHT (threads 8) + the flat md5"}
    log(f"G_ROOT_MIR (vs e193b committed dial): max|diff| {rmax2:.2e}, flat "
        f"md5 {'match' if G_ROOT_MIR['flat_md5_match_e193b_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT_MIR["pass"] else "FAIL"))
    if not G_ROOT_MIR["pass"]:
        raise RuntimeError("MIRABEL root gate FAILED — abort")
    root_gm_m = root_cells_2[FACT2][f"g{RULER_J:+d}"]
    root_gm_z = root_cells_2[FACT1]["g+0"]

    # G_T0_MIR + G_S1CK (e198's, verbatim)
    g = torch.Generator().manual_seed(FREEZE_SEED)
    aj = torch.randint(16, (ANCH_BS,), generator=g)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
    x1m = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y1m = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
    x1m_md5 = hashlib.md5(x1m.contiguous().numpy().tobytes()).hexdigest()
    tw2 = copy.deepcopy(net2)
    tw2.train()
    optw2 = torch.optim.AdamW(tw2.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                              weight_decay=0.1)
    logits2, _ = tw2(x1m)
    ce1m = float(F.cross_entropy(logits2.reshape(-1, logits2.shape[-1]),
                                 y1m.reshape(-1)).item())
    optw2.zero_grad(set_to_none=True)
    F.cross_entropy(logits2.reshape(-1, logits2.shape[-1]),
                    y1m.reshape(-1)).backward()
    gn1m = float(torch.nn.utils.clip_grad_norm_(tw2.parameters(), 1.0))
    optw2.step()
    theta1_adam = flat_params(tw2)
    disp1m = float(torch.norm(theta1_adam - theta0_2))
    del tw2, optw2, logits2
    G_T0_MIR = {
        "step1_x_md5": x1m_md5,
        "step1_x_md5_match_e185": bool(x1m_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1m,
        "ce_batch_committed_e193b_asign": E193B_ASIGN_S1["ce_batch"],
        "d_ce": abs(ce1m - E193B_ASIGN_S1["ce_batch"]),
        "preclip_gnorm_measured": gn1m, "clip_binds": bool(gn1m > 1.0),
        "adamw_step1_L2_measured": disp1m,
        "committed_e193b_step_l2": MIR_STEP_L2,
        "d_step_l2": abs(disp1m - MIR_STEP_L2),
        "note": "e198's G_T0 VERBATIM: MIRABEL's step L2 re-MEASURED (never "
                "ported)",
        "pass": bool(x1m_md5 == E185_XHASH[1]
                     and abs(ce1m - E193B_ASIGN_S1["ce_batch"]) < G_FALLBACK_TOL
                     and abs(disp1m - MIR_STEP_L2) < G_FALLBACK_TOL),
    }
    log(f"G_T0_MIR: x_md5 "
        f"{'OK' if G_T0_MIR['step1_x_md5_match_e185'] else 'MISMATCH'}; CE |d| "
        f"{G_T0_MIR['d_ce']:.2e}; measured L2 {disp1m:.10f} vs committed "
        f"{MIR_STEP_L2:.10f} (|d| {G_T0_MIR['d_step_l2']:.2e}): "
        + ("PASS" if G_T0_MIR["pass"] else "FAIL"))
    if not G_T0_MIR["pass"]:
        raise RuntimeError("MIRABEL t=0 gate FAILED — abort")
    STEP_L2_2 = disp1m                       # measured, never ported

    s1ck = torch.load(CKPT_DIR / MIR_S1_CK, map_location="cpu",
                      weights_only=False)
    theta_s1 = s1ck["theta1"]
    d_fresh = theta1_adam - theta0_2
    d_ck = theta_s1 - theta0_2
    cos_s1 = cos64(d_fresh, d_ck)
    rel_l2 = abs(float(torch.norm(d_ck)) - disp1m) / disp1m
    G_S1CK = {
        "path": str(CKPT_DIR / MIR_S1_CK),
        "cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
        "tol_cos": 0.999, "ckpt_meta": E43.jsonable(s1ck.get("meta", {})),
        "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
        "note": "e198's G_S1CK VERBATIM (e196's form): the AdamW sibling "
                "state cross-anchor",
    }
    log(f"G_S1CK: cos64 {cos_s1:.8f}, rel L2 dev {rel_l2:.2e}: "
        + ("PASS" if G_S1CK["pass"] else "FAIL"))
    if not G_S1CK["pass"]:
        raise RuntimeError("MIRABEL wash s1 checkpoint anchor FAILED — abort")

    # MIRABEL's t=0 post-clip gradient + committed directions
    gnet = copy.deepcopy(net2)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1m)
    loss_g = F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                             y1m.reshape(-1))
    assert abs(float(loss_g.item()) - ce1m) < 1e-9, "CE drifted between gates"
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(gnet.parameters(), 1.0)
    g0_mir = torch.cat([p.grad.detach().reshape(-1)
                        for p in gnet.parameters()]).clone()   # post-clip
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g, loss_g
    u_g_mir = (g0_mir / torch.norm(g0_mir)).clone()
    u_g_mir_md5 = hashlib.md5(u_g_mir.numpy().tobytes()).hexdigest()
    u0_2 = (torch.sign(g0_mir) / torch.norm(torch.sign(g0_mir))).clone()
    u0_2_md5 = hashlib.md5(u0_2.numpy().tobytes()).hexdigest()
    duck = torch.load(CKPT_DIR / MIR_DIR_CK, map_location="cpu",
                      weights_only=False)
    u_ck = duck["u"].float()
    ug_maxdiff = float((u_g_mir - u_ck).abs().max())
    G_DIRCK = {
        "path": str(CKPT_DIR / MIR_DIR_CK),
        "meta_gate": {"experiment": duck["meta"].get("experiment"),
                      "match": bool(duck["meta"].get("experiment") == "e193b")},
        "loaded_u_md5": hashlib.md5(duck["u"].numpy().tobytes()).hexdigest(),
        "fresh_u_md5": u_g_mir_md5,
        "md5_match": bool(hashlib.md5(
            duck["u"].numpy().tobytes()).hexdigest() == u_g_mir_md5),
        "max_abs_diff_fresh_vs_ckpt": ug_maxdiff,
        "cos64_fresh_vs_ckpt": cos64(u_g_mir, u_ck),
        "theta0_md5_match_root": bool(duck.get("theta0_md5") == mir_root_md5),
        "note": "e198's G_DIRCK with the TEXTURE tier added: the committed u "
                "must BE the fresh t=0 g-ray — bit-exact, or (the disclosed "
                "cross-environment fallback, e196's G_S1CK precedent) within "
                "1e-6 coordinate-wise",
    }
    G_DIRCK["pass"] = bool(G_DIRCK["meta_gate"]["match"]
                           and G_DIRCK["theta0_md5_match_root"]
                           and (G_DIRCK["md5_match"] or ug_maxdiff < 1e-6))
    G_DIRCK["tier"] = "BIT" if G_DIRCK["md5_match"] else "TEXTURE"
    u0_ck = (torch.sign(u_ck) / torch.norm(torch.sign(u_ck))).clone()
    u0_cos_ck = cos64(u0_2, u0_ck)
    n_flip = int((torch.sign(g0_mir) != torch.sign(u_ck)).sum())
    G_SIGNRAY = {
        "u0_md5": u0_2_md5, "committed_e193b_md5": E193B_USIGN_MD5,
        "md5_match": bool(u0_2_md5 == E193B_USIGN_MD5),
        "cos64_fresh_u0_vs_ckpt_derived_sign_ray": u0_cos_ck,
        "sign_flip_coords_vs_ckpt": n_flip, "n_coords": N_PARAM,
        "u_norm_fp32": float(torch.norm(u0_2)),
        "note": "MIRABEL's u0 rebuilt (e193b's construction VERBATIM); "
                "md5-gated vs e193b's committed R2_SIGN, with the TEXTURE "
                "tier as the disclosed fallback (cosine vs the sign ray "
                "derived from the committed g-ray checkpoint — sign(g/||g||) "
                "IS sign(g))",
    }
    G_SIGNRAY["pass"] = bool(G_SIGNRAY["md5_match"]
                             or (u0_cos_ck > RAY_SIGN_COS_FLOOR
                                 and n_flip < 1000))
    G_SIGNRAY["tier"] = "BIT" if G_SIGNRAY["md5_match"] else "TEXTURE"
    log(f"G_DIRCK: " + ("PASS" if G_DIRCK["pass"] else "FAIL")
        + f" (tier {G_DIRCK['tier']}, max|d| {ug_maxdiff:.2e})"
        + f"; G_SIGNRAY: " + ("PASS" if G_SIGNRAY["pass"] else "FAIL")
        + f" (tier {G_SIGNRAY['tier']}, cos {u0_cos_ck:.9f}, {n_flip} "
        f"sign-flip coords/{N_PARAM})")
    if not (G_DIRCK["pass"] and G_SIGNRAY["pass"]):
        raise RuntimeError("MIRABEL committed-ray gates FAILED — abort")
    stub["gates"].update({"G_ROOT_MIR": G_ROOT_MIR, "G_T0_MIR": G_T0_MIR,
                          "G_S1CK": G_S1CK, "G_DIRCK": G_DIRCK,
                          "G_SIGNRAY": G_SIGNRAY})
    write_partial("MIRABEL standard gates passed")

    # ---- MIRABEL's walk rebuilt t=0..t=3 (to its death; the ray factory)
    log("-" * 78)
    log("MIRABEL's walk rebuilt (e198's walk_rebuild VERBATIM, n_steps=3 — "
        f"its committed death; horizon t<={WALK_MAX_T} or death, whichever "
        "first)")
    walk2 = walk_rebuild_mir(net2, anchor_neutral, train_ids, itos,
                             mir_primary, mir_geos, zid_m,
                             zeph_primary_g0, zid_z, theta0_2, STEP_L2_2,
                             root_gm_m, root_gm_z)
    assert len(walk2["traj"]) == 3, \
        f"expected the 3-step walk, got {len(walk2['traj'])}"
    assert walk2["kill_mirabel"] is not None \
        and walk2["kill_mirabel"]["step"] == 3, \
        f"MIRABEL's walk did not die at t=3 as committed"
    t1r, t2r, t3r = walk2["traj"]

    # G_REPRO_MIR: vs e193b's committed a_sign row + bracket AND e198's
    # committed journal rows
    repro_rows_2 = {
        "s1_ce_batch": (t1r["ce_batch"], E193B_ASIGN_S1["ce_batch"]),
        "s1_cum_disp": (t1r["cum_disp"], E193B_ASIGN_S1["cum_disp"]),
        "s1_preclip_gnorm": (t1r["preclip_gnorm"],
                             E193B_ASIGN_S1["preclip_gnorm"]),
        "s1_gm_ZEPHYRA": (t1r["gm_ZEPHYRA"], E193B_ASIGN_S1["gm_ZEPHYRA"]),
        "s1_ce_r": (t1r["ce_r"], E193B_ASIGN_S1["ce_r"]),
        "s1_gm_MIRABEL": (t1r["gm"], E193B_ASIGN_S1["gm_MIRABEL"]),
        "s1_g-4_MIRABEL": (t1r["g-4"], E193B_ASIGN_S1["g-4_MIRABEL"]),
        "s1_g+0_MIRABEL": (t1r["g+0"], E193B_ASIGN_S1["g+0_MIRABEL"]),
        "s1_g+12_MIRABEL": (t1r["g+12"], E193B_ASIGN_S1["g+12_MIRABEL"]),
        "zephyra_D_kill": (walk2["kill_zephyra"]["D_kill"],
                           E193B_ASIGN_STOP["D_kill"]),
        "zephyra_D_kill_raw": (walk2["kill_zephyra"]["D_kill_raw"],
                               E193B_ASIGN_STOP["D_kill_raw"]),
        "e198_s2_gm": (t2r["gm"], E198_WALK_JOURNAL[1]["gm"]),
        "e198_s2_cum_disp": (t2r["cum_disp"], E198_WALK_JOURNAL[1]["cum_disp"]),
        "e198_s3_gm": (t3r["gm"], E198_WALK_JOURNAL[2]["gm"]),
        "e198_s3_cum_disp": (t3r["cum_disp"], E198_WALK_JOURNAL[2]["cum_disp"]),
        "e198_kill_D_kill": (walk2["kill_mirabel"]["D_kill"],
                             E198_KILL_M["D_kill"]),
        "e198_kill_D_kill_raw": (walk2["kill_mirabel"]["D_kill_raw"],
                                 E198_KILL_M["D_kill_raw"]),
    }
    repro_max2 = max(abs(vv[0] - vv[1]) for vv in repro_rows_2.values())
    G_REPRO_MIR = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows_2.items()},
        "x_hashes_vs_e185": {s_: (walk2["x_hashes"][s_] == E185_XHASH[s_])
                             for s_ in (1, 2, 3)},
        "tol_bit": G_REPRO_TOL, "tol_texture": WALK_TEXTURE_TOL,
        "max_row_abs_diff": repro_max2,
        "tier": ("BIT" if repro_max2 < G_REPRO_TOL else
                 ("TEXTURE" if repro_max2 < WALK_TEXTURE_TOL else "FAIL")),
        "max_l2_dev": walk2["max_l2_dev"],
        "pass": bool(repro_max2 < WALK_TEXTURE_TOL
                     and all(walk2["x_hashes"][s_] == E185_XHASH[s_]
                             for s_ in (1, 2, 3))),
        "note": "e198's G_REPRO (extended) with the TEXTURE tier: the "
                "rebuilt walk reproduces e193b's committed a_sign step-1 "
                "row + ZEPHYRA bracket AND e198's committed t=2/t=3 rows + "
                "the MIRABEL kill bracket — bit-class, or (the disclosed "
                "cross-environment fp drift ~1e-7/row) within 1e-3; the "
                "achieved tier is stamped",
    }
    log("G_REPRO_MIR (vs e193b committed + e198 committed): max|diff| "
        f"{repro_max2:.2e} (tier {G_REPRO_MIR['tier']}): "
        + ("PASS" if G_REPRO_MIR["pass"] else "FAIL"))
    if not G_REPRO_MIR["pass"]:
        stub["gates"]["G_REPRO_MIR"] = G_REPRO_MIR
        write_partial("CONTROL FAILURE — MIRABEL walk provenance gate failed")
        raise RuntimeError("G_REPRO_MIR FAILED — abort")
    stub["gates"]["G_REPRO_MIR"] = G_REPRO_MIR

    # MIRABEL's rays: u1/u2 from the rebuilt walk (alive ts) + G_U12
    g0_w2, g1_w2, g2_w2 = walk2["g_fronts"]
    assert cos64(torch.sign(g0_w2), u0_2) > 1 - 1e-9, "t=0 front != u0"
    u1_2 = (torch.sign(g1_w2) / torch.norm(torch.sign(g1_w2))).clone()
    u2_2 = (torch.sign(g2_w2) / torch.norm(torch.sign(g2_w2))).clone()
    u1_2_md5 = hashlib.md5(u1_2.numpy().tobytes()).hexdigest()
    u2_2_md5 = hashlib.md5(u2_2.numpy().tobytes()).hexdigest()
    geom2 = {"cos_u0_u1": cos64(u0_2, u1_2),
             "cos_u0_u2": cos64(u0_2, u2_2),
             "cos_u1_u2": cos64(u1_2, u2_2)}
    geom_dev = {k: abs(geom2[k] - E198_RAY_GEOM[k]) for k in geom2}
    geom_ok = all(v < RAY_COS_TOL for v in geom_dev.values())
    both_md5 = bool(u1_2_md5 == E198_MIR["u1_md5"]
                    and u2_2_md5 == E198_MIR["u2_md5"])
    G_U12 = {
        "u1_md5": u1_2_md5, "u1_md5_match_e198": bool(
            u1_2_md5 == E198_MIR["u1_md5"]),
        "u2_md5": u2_2_md5, "u2_md5_match_e198": bool(
            u2_2_md5 == E198_MIR["u2_md5"]),
        "ray_geometry_fresh": geom2,
        "ray_geometry_committed_e198": E198_RAY_GEOM,
        "ray_geometry_max_abs_dev": max(geom_dev.values()),
        "tol_geom": RAY_COS_TOL,
        "tier": ("BIT" if both_md5 else ("TEXTURE" if geom_ok else "FAIL")),
        "note": "THE ROTATED-RAY IDENTITY GATE with the TEXTURE tier: the "
                "rebuilt u1/u2 must BE e198's registered rays — bit-exact "
                "by md5, or (the disclosed cross-environment fallback) "
                "their MUTUAL GEOMETRY must reproduce e198's committed "
                "fp64 cosines within 1e-3 (u1 read at the ALIVE theta_1 "
                "0.3673; u2 at the ALIVE theta_2 0.6911 — both LIVE "
                "mid-flight fronts); the functional profile gates below "
                "complete the chain",
        "pass": bool(both_md5 or geom_ok),
    }
    log("G_U12 (u1/u2 vs e198 registered): tier "
        f"{G_U12['tier']} (md5 u1 "
        + ("match" if G_U12["u1_md5_match_e198"] else "DRIFT")
        + ", u2 " + ("match" if G_U12["u2_md5_match_e198"] else "DRIFT")
        + f"; geometry max|dev| {max(geom_dev.values()):.2e}): "
        + ("PASS" if G_U12["pass"] else "FAIL"))
    if not G_U12["pass"]:
        raise RuntimeError("G_U12 FAILED — abort")
    stub["gates"]["G_U12"] = G_U12

    # MIRABEL's alive ledger + death
    mir_ledger = [
        {"t": 0, "state": "theta_0 (the e193b root)",
         "ruler_read": root_gm_m, "alive": bool(root_gm_m > SHUT_BAR),
         "ray": "u0 (read here)"},
        {"t": 1, "state": "theta_1", "ruler_read": t1r["gm"],
         "alive": bool(t1r["gm"] > SHUT_BAR),
         "ray": "u1 (read here, at the ALIVE state 0.3673)"},
        {"t": 2, "state": "theta_2", "ruler_read": t2r["gm"],
         "alive": bool(t2r["gm"] > SHUT_BAR),
         "ray": "u2 (read here, at the ALIVE state 0.6911)"},
        {"t": 3, "state": "theta_3 (the walk's kill step)",
         "ruler_read": t3r["gm"], "alive": bool(t3r["gm"] > SHUT_BAR),
         "ray": None, "note": "DEATH — the post-death front never read"},
    ]
    log("MIRABEL alive ledger: "
        + ", ".join(f"t{r['t']} {'ALIVE' if r['alive'] else 'DEAD'} "
                    f"{r['ruler_read']:.4f}" for r in mir_ledger))
    write_partial("MIRABEL walk rebuilt + provenance gates passed")

    # ---- MIRABEL's ray profiles from the ROOT (recomputed + gated)
    log("-" * 78)
    log(f"MIRABEL's root ray profiles: u0/u1/u2 x {len(D_GRID)}+ grid points "
        "(e198's static_profile VERBATIM)")
    evp2 = copy.deepcopy(net2)
    r2_extra = [d for d in E193B_R2_GRID if d <= 3.0]
    grids2 = {"u0": sorted(set([0.0] + list(D_GRID) + [STEP_L2_2]
                               + r2_extra)),
              "u1": sorted(set([0.0] + list(D_GRID))),
              "u2": sorted(set([0.0] + list(D_GRID)))}
    fam2 = {}
    for key, u in (("u0", u0_2), ("u1", u1_2), ("u2", u2_2)):
        tA = time.time()
        rows = static_profile_mir(evp2, theta0_2, u, grids2[key],
                                  mir_primary, mir_corulers, zid_m,
                                  r_eval_xy, ce_ds)
        edge, ups = profile_d_kill(rows)
        gate = gate_profile(rows, e198m["profiles"][f"root_{key}"]["rows"],
                            "gm", f"mirabel root_u{key[1]} vs e198 committed")
        fam2[key] = {"rows": rows, "D_kill": edge, "any_upcross": ups,
                     "gate": gate, "seconds": round(time.time() - tA, 1)}
        stub["phases_partial"][f"mir_root_{key}"] = E43.jsonable(
            {"D_kill": edge, "gate_pass": gate["pass"],
             "max_abs_diff": gate["max_abs_diff"]})
        write_partial(f"MIRABEL root_u{key[1]} profile complete")
        log(f"  mir root_{key}: {len(rows)} pts in {fam2[key]['seconds']}s "
            f"— D_kill " + (f"{edge:.10f}" if edge is not None else "None")
            + f"; vs e198 committed {gate['n_shared']} shared pts max|diff| "
            f"{gate['max_abs_diff']:.2e}")
    # e193b's committed R2_SIGN crosscheck on the u0 family (the chain
    # e193b -> e198 -> e199)
    r2_by_D = {round(r["D"], 6): r for r in e193bm["profiles"]["R2_SIGN"]}
    r2_xc = []
    for row in fam2["u0"]["rows"]:
        key = round(row["D"], 6)
        if key in r2_by_D:
            r2_xc.append({"D": key, "gm_measured": row["gm12"],
                          "gm_e193b": r2_by_D[key]["gm_MIRABEL"],
                          "abs_diff": abs(row["gm12"]
                                          - r2_by_D[key]["gm_MIRABEL"])})
    G_PROF_MIR = {
        "u0_vs_e198": fam2["u0"]["gate"], "u1_vs_e198": fam2["u1"]["gate"],
        "u2_vs_e198": fam2["u2"]["gate"],
        "u0_vs_e193b_R2SIGN": {"n_shared": len(r2_xc),
                               "max_abs_diff": (max(r["abs_diff"] for r in r2_xc)
                                                if r2_xc else None),
                               "rows": r2_xc},
        "u0_D_kill": fam2["u0"]["D_kill"],
        "u0_D_kill_committed": E198_MIR["root_u0_D_kill"],
        "u1_D_kill": fam2["u1"]["D_kill"],
        "u1_D_kill_committed": E198_MIR["root_u1_D_kill"],
        "u2_D_kill": fam2["u2"]["D_kill"],
        "u2_D_kill_committed": E198_MIR["root_u2_D_kill"],
        "tol_rows": G_STATIC_TOL, "tol_dkill": DKILL_TOL,
        "dkill_check_skipped": bool(SMOKE),
        "pass": None,
        "note": "MIRABEL's profile machinery gate: all three recomputed "
                "families reproduce e198's committed rows + D_kills (u1's "
                "None must reproduce as None); the u0 family also "
                "reproduces e193b's committed R2_SIGN gm_MIRABEL at the "
                "shared Ds (D_kill equality skipped in smoke — the trimmed "
                "grid changes the interpolation bracket)",
    }

    G_PROF_MIR["pass"] = bool(
        fam2["u0"]["gate"]["pass"] and fam2["u1"]["gate"]["pass"]
        and fam2["u2"]["gate"]["pass"]
        and (SMOKE or (
            _dk_texture_ok(fam2["u0"]["D_kill"], E198_MIR["root_u0_D_kill"])
            and _dk_texture_ok(fam2["u1"]["D_kill"], E198_MIR["root_u1_D_kill"])
            and _dk_texture_ok(fam2["u2"]["D_kill"], E198_MIR["root_u2_D_kill"])))
        and r2_xc and all(r["abs_diff"] < G_STATIC_TOL for r in r2_xc))
    dkill_dev2 = max(
        [abs(fam2[k]["D_kill"] - E198_MIR[f"root_{k}_D_kill"])
         for k in ("u0", "u2") if fam2[k]["D_kill"] is not None
         and E198_MIR[f"root_{k}_D_kill"] is not None] or [0.0])
    G_PROF_MIR["dkill_max_abs_dev"] = dkill_dev2
    G_PROF_MIR["dkill_tier"] = ("BIT" if dkill_dev2 < DKILL_TOL else
                                ("TEXTURE" if dkill_dev2 < DKILL_TEXTURE_TOL
                                 else "FAIL"))
    G_PROF_MIR["note"] += (
        " — D_kill tier: BIT at 1e-6, or the disclosed TEXTURE tier 1e-2 "
        "(the cross-environment fp drift); achieved "
        + G_PROF_MIR["dkill_tier"])
    log("G_PROF_MIR (rows + D_kills vs committed; u0 vs e193b R2 too): "
        + ("PASS" if G_PROF_MIR["pass"] else "FAIL"))
    if not G_PROF_MIR["pass"]:
        stub["gates"]["G_PROF_MIR"] = G_PROF_MIR
        write_partial("CONTROL FAILURE — MIRABEL profile gates failed")
        raise RuntimeError("G_PROF_MIR FAILED — abort")
    stub["gates"]["G_PROF_MIR"] = G_PROF_MIR

    # MIRABEL's onset rows
    u0_edge_2 = fam2["u0"]["D_kill"]
    min_u1 = min(fam2["u1"]["rows"], key=lambda r: r["gm12"])
    mir_onset = [
        {"t": 0, "ray": "u0", "u_md5": u0_2_md5, "D_kill": u0_edge_2,
         "ratio": 1.0, "ratio_note": "definitional (u0 vs its own edge)",
         "state": "alive", "ruler_read": root_gm_m},
        {"t": 1, "ray": "u1", "u_md5": u1_2_md5, "D_kill": fam2["u1"]["D_kill"],
         "ratio": None,
         "state": "alive", "ruler_read": t1r["gm"],
         "soft": True,
         "floor_ratio": (max(D_GRID) / u0_edge_2 if u0_edge_2 else None),
         "min_read": {"D": min_u1["D"], "gm12": min_u1["gm12"]},
         "note": "SOFT (unresolved-high): u1 never kills within [0, 3.0] "
                 f"(min g-12 {min_u1['gm12']:.4f} at D {min_u1['D']:.2f}); "
                 "the floor ratio 3.0/edge is a floor, never adjudicated"},
        {"t": 2, "ray": "u2", "u_md5": u2_2_md5, "D_kill": fam2["u2"]["D_kill"],
         "ratio": (fam2["u2"]["D_kill"] / u0_edge_2
                   if fam2["u2"]["D_kill"] is not None else None),
         "state": "alive", "ruler_read": t2r["gm"]},
        {"t": 3, "ray": None, "D_kill": None, "ratio": None,
         "state": "death", "ruler_read": t3r["gm"],
         "walk_D_kill": walk2["kill_mirabel"]["D_kill"],
         "note": "the walk's densified kill inside the step-3 bracket"},
    ]
    log("MIRABEL onset: "
        + "; ".join(f"t{r['t']} ratio "
                    + ("SOFT (floor "
                       f"{r.get('floor_ratio', float('nan')):.4f})"
                       if r.get("soft")
                       else ("DEATH (walk kill "
                             f"{r.get('walk_D_kill', float('nan')):.4f})"
                             if r["state"] == "death" else f"{r['ratio']:.4f}"))
                    for r in mir_onset))
    stub["phases_partial"]["mir_onset"] = E43.jsonable({"onset": mir_onset})
    write_partial("PHASE B complete (MIRABEL onset curve assembled)")

    # =====================================================================
    # THE ONSET CURVES + ADJUDICATION (frozen clauses; composite order
    # ONSET-COMMON -> ONSET-ORG1-ONLY -> GRADED; no shopping)
    # =====================================================================
    log("=" * 78)

    def first_concentrated(onset):
        for r in onset:
            if r["state"] == "alive" and r.get("ratio") is not None \
                    and r["ratio"] < CONC_BAR:
                return r["t"]
        return None

    def falling_clause(onset):
        """'the ratio falling with t (a deepening concentration once
        rotation begins)': reaches a concentrated alive t AND every resolved
        ratio strictly after the first concentrated t stays <= 0.70."""
        tc = first_concentrated(onset)
        if tc is None:
            return False, None, None
        later = [r for r in onset
                 if r["state"] == "alive" and r["t"] > tc
                 and r.get("ratio") is not None]
        stays = all(r["ratio"] <= CONC_BAR for r in later)
        return bool(stays), tc, later

    org1_falls, org1_tc, org1_later = falling_clause(org1_onset)
    mir_falls, mir_tc, mir_later = falling_clause(mir_onset)
    mir_any_concentrated = first_concentrated(mir_onset) is not None
    shift_ok = bool(org1_tc is not None and mir_tc is not None
                    and org1_tc == mir_tc - 1)
    fires_common = bool(org1_falls and mir_falls and shift_ok)
    fires_org1_only = bool(org1_falls and not mir_any_concentrated)

    # the COMMITTED-NUMBERS CROSSCHECK (robustness, not shopping): re-run
    # the same frozen clauses on the committed D_kill sets (e195's org1 /
    # e198's MIRABEL) — the verdict must be identical under both sets, and
    # both are reported.
    org1_r1_c = E195_ORG1["root_u1_D_kill"] / E195_ORG1["root_u0_D_kill"]
    mir_r2_c = (None if E198_MIR["root_u2_D_kill"] is None else
                E198_MIR["root_u2_D_kill"] / E198_MIR["root_u0_D_kill"])
    org1_falls_c = bool(org1_r1_c < CONC_BAR)
    mir_falls_c = bool(mir_r2_c is not None and mir_r2_c < CONC_BAR)
    fires_common_c = bool(org1_falls_c and mir_falls_c)   # shift fixed by
    fires_org1_only_c = bool(org1_falls_c and not mir_falls_c)  # construction
    committed_crosscheck = {
        "note": "the same frozen clauses on the committed numbers (e195/"
                "e198's D_kills) vs this cell's fresh recomputation — a "
                "robustness check, never a bar choice; both sets reported",
        "org1_t1_ratio_committed": org1_r1_c,
        "mirabel_t1_committed": None,
        "mirabel_t2_ratio_committed": mir_r2_c,
        "fires_common_committed": fires_common_c,
        "fires_org1_only_committed": fires_org1_only_c,
        "verdict_identical_under_both_sets": bool(
            fires_common_c == fires_common
            and fires_org1_only_c == fires_org1_only),
    }

    org1_r1 = next(r for r in org1_onset if r["t"] == 1)["ratio"]
    mir_r2 = next(r for r in mir_onset if r["t"] == 2)["ratio"]
    mir_floor = next(r for r in mir_onset if r["t"] == 1)["floor_ratio"]
    mir_min1 = next(r for r in mir_onset if r["t"] == 1)["min_read"]

    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "ONSET_COMMON": {
                "fires": fires_common,
                "detail": {
                    "org1_falling": org1_falls,
                    "org1_first_concentrated_t": org1_tc,
                    "org1_resolved_after_tc": [
                        {"t": r["t"], "ratio": r["ratio"]} for r in org1_later],
                    "mirabel_falling": mir_falls,
                    "mirabel_first_concentrated_t": mir_tc,
                    "mirabel_resolved_after_tc": [
                        {"t": r["t"], "ratio": r["ratio"]} for r in mir_later],
                    "shift_one_step_earlier": shift_ok,
                    "concentration_bar": CONC_BAR,
                },
            },
            "ONSET_ORG1_ONLY": {
                "fires": fires_org1_only,
                "detail": {
                    "org1_falling": org1_falls,
                    "mirabel_any_concentrated_t": mir_any_concentrated,
                    "mirabel_ratios": [
                        {"t": r["t"], "ratio": r.get("ratio"),
                         "soft": bool(r.get("soft"))} for r in mir_onset
                        if r["state"] == "alive"],
                },
            },
            "GRADED": {"fires": not (fires_common or fires_org1_only)},
        }
        if fires_common:
            verdict = "ONSET-COMMON"
            clause = (
                f"both organisms show the ratio falling with t — org1: 1.0 "
                f"(definitional) -> {org1_r1:.4f} at t=1 (its floor; the "
                f"walk dies at t=2, so 'deepening past t=1' is untestable "
                f"alive — the concentration, once arrived, never un-forms "
                f"inside its alive window); MIRABEL: 1.0 -> SOFT at t=1 "
                f"(unresolved-high, floor {mir_floor:.4f} — u1 never kills "
                f"within [0, 3.0], min g-12 {mir_min1['gm12']:.4f} at D "
                f"{mir_min1['D']:.2f}) -> {mir_r2:.4f} at t=2 — with org1's "
                f"first concentrated t ({org1_tc}) exactly one step earlier "
                f"than MIRABEL's ({mir_tc}) — the concentration is a WHEN; "
                f"the timing is the biography.")
        elif fires_org1_only:
            verdict = "ONSET-ORG1-ONLY"
            clause = (
                f"org1's ratio deepens to its t=1 floor ({org1_r1:.4f}) "
                f"while MIRABEL's stays flat/soft at every alive t (no "
                f"concentrated t; ratios verbatim) — the concentration "
                f"itself is org-1-only; the t=2 wrinkle was noise.")
        else:
            verdict = "GRADED"
            clause = ("any mix — the curves verbatim: org1 "
                      + ", ".join(
                          f"t{r['t']}: "
                          + ("DEATH" if r["state"] == "death"
                             else (f"{r['ratio']:.4f}"
                                   if r.get("ratio") is not None else "SOFT"))
                          for r in org1_onset)
                      + " (post-death t=2 context "
                      f"{org1_ctx['ratio_vs_own_u0']:.4f}, e195 committed, "
                      "never adjudicated); MIRABEL "
                      + ", ".join(
                          f"t{r['t']}: "
                          + ("DEATH" if r["state"] == "death"
                             else (f"{r['ratio']:.4f}"
                                   if r.get("ratio") is not None
                                   else f"SOFT(floor {r['floor_ratio']:.4f})"))
                          for r in mir_onset) + ".")
    log(f"E199 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e199_onset_curve",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "question": ("THE ONSET CURVE (T162's timing cut): on each "
                     "organism, walk its wash trajectory (t=1..6 as the "
                     "states stay alive; stop at death) and map sign(g_t) "
                     "rays from the ROOT at each alive t — does each "
                     "organism's concentration ratio DEEPEN along its "
                     "trajectory? Is there a common onset shape (the ratio "
                     "falling once rotation begins), just shifted in time?"),
        "organisms": {
            "org1": {
                "root": f"runs/checkpoints/{ORG1_ROOT_CK}",
                "root_meta": org1_root_meta,
                "fact": FACT1, "ruler": "ZEPHYRA install-60 g-12 battery",
                "step_l2_measured": STEP_L2_1,
                "walk": {"journal": [{k: v for k, v in r.items()
                                      if k != "x_md5"} for r in j1],
                         "front_trace": walk1["front_trace"],
                         "stop": walk1["stop"],
                         "max_l2_dev": walk1["max_l2_dev"],
                         "zeph": walk1["zeph"],
                         "train_seconds": walk1["train_seconds"],
                         "death_t": 2,
                         "extension_disclosure":
                             "the committed walk already dies at t=2 "
                             "(opt2/e194/e195) — before the t=6 horizon; "
                             "no missing steps existed to walk (re-verified "
                             "in the gated rebuild)"},
                "alive_ledger": org1_ledger,
                "rays": {
                    "u0": {"u_md5": u0_1_md5,
                           "provenance": "e192's R2 construction verbatim "
                                         "(md5-gated, G_R2DIR)"},
                    "u1": {"u_md5": u1_1_md5,
                           "provenance": "the rebuilt walk's step-2 refresh "
                                         "gradient (at theta_1, ALIVE "
                                         "0.6786); gated vs e194's committed "
                                         "front trace + e195's registered "
                                         "ray md5 (G_FRONTS_ORG1)"},
                    "u2_context_only": org1_ctx},
                "profiles": {
                    f"root_{k}": {
                        "anchor": "theta_0", "ray": k,
                        "u_md5": (u0_1_md5 if k == "u0" else u1_1_md5),
                        "placement": "root - D*u (e192's static placement "
                                     "verbatim)",
                        "grid": grids1[k], "n_points": len(v["rows"]),
                        "rows": v["rows"], "D_kill": v["D_kill"],
                        "any_upcross": v["any_upcross"],
                        "gate": {kk: vv for kk, vv in v["gate"].items()
                                 if kk != "rows"},
                        "seconds": v["seconds"]}
                    for k, v in fam1.items()},
            },
            "mirabel": {
                "root": f"runs/checkpoints/{MIR_ROOT_CK}",
                "root_meta": mir_root_meta,
                "fact": FACT2, "ruler": "MIRABEL install-60 g-12 battery "
                                        "(+ g-4/g0/g+12 co-rulers per row)",
                "step_l2_measured": STEP_L2_2,
                "walk": {"journal": walk2["traj"],
                         "kill_mirabel": walk2["kill_mirabel"],
                         "kill_zephyra_rederived": {
                             k: v for k, v in walk2["kill_zephyra"].items()
                             if k != "dens"},
                         "max_l2_dev": walk2["max_l2_dev"],
                         "seconds": walk2["seconds"],
                         "death_t": 3,
                         "extension_disclosure":
                             "the committed walk already dies at t=3 (e198) "
                             "— before the t=6 horizon; no missing steps "
                             "existed to walk (re-verified in the gated "
                             "rebuild)"},
                "alive_ledger": mir_ledger,
                "rays": {
                    "u0": {"u_md5": u0_2_md5,
                           "provenance": "e193b's R2_SIGN construction "
                                         "verbatim (md5-gated, G_SIGNRAY)"},
                    "u1": {"u_md5": u1_2_md5,
                           "provenance": "the rebuilt walk's step-2 refresh "
                                         "gradient (at theta_1, ALIVE "
                                         "0.3673); md5-gated vs e198's "
                                         "registered ray (G_U12)"},
                    "u2": {"u_md5": u2_2_md5,
                           "provenance": "the rebuilt walk's step-3 refresh "
                                         "gradient (at theta_2, ALIVE "
                                         "0.6911); md5-gated vs e198's "
                                         "registered ray (G_U12)"}},
                "profiles": {
                    f"root_{k}": {
                        "anchor": "theta_0", "ray": k,
                        "u_md5": {"u0": u0_2_md5, "u1": u1_2_md5,
                                  "u2": u2_2_md5}[k],
                        "placement": "root - D*u (e192's static placement "
                                     "verbatim)",
                        "grid": grids2[k], "n_points": len(v["rows"]),
                        "rows": v["rows"], "D_kill": v["D_kill"],
                        "any_upcross": v["any_upcross"],
                        "gate": {kk: vv for kk, vv in v["gate"].items()
                                 if kk != "rows"},
                        "seconds": v["seconds"]}
                    for k, v in fam2.items()},
            },
        },
        "onset_curves": {
            "org1": org1_onset,
            "mirabel": mir_onset,
            "org1_post_death_context": {"t": org1_ctx["t"],
                                        "ratio": org1_ctx["ratio_vs_own_u0"],
                                        "D_kill": org1_ctx["D_kill"],
                                        "disclosure": org1_ctx["disclosure"]},
            "side_by_side": [
                {"t": t,
                 "org1_ratio": next((r["ratio"] for r in org1_onset
                                     if r["t"] == t), None),
                 "org1_state": next((r["state"] for r in org1_onset
                                     if r["t"] == t), None),
                 "mirabel_ratio": next((r.get("ratio") for r in mir_onset
                                        if r["t"] == t), None),
                 "mirabel_state": next((r["state"] for r in mir_onset
                                        if r["t"] == t), None)}
                for t in range(0, 4)],
            "concentration_bar": CONC_BAR,
            "first_concentrated_t": {"org1": org1_tc, "mirabel": mir_tc},
            "the_four_lineage_ledger_context": {
                "org1": {"u1_ratio": org1_r1},
                "org2_dead_e196": {"u1_ratio": E196_ROOT_DKILLS["root_u1"]
                                   / E196_ROOT_DKILLS["root_u0"]},
                "org2_alive_half_e197": {"u1_ratio": E197_ALIVE["u1_D_kill"]
                                         / E197_ALIVE["u0_edge"],
                                         "note": "its t=2..t=4 fronts were "
                                                 "never mapped — its own "
                                                 "onset curve is owed, "
                                                 "named, not run here"},
                "mirabel": {"t1": None, "t2": mir_r2}},
        },
        "gates": stub["gates"],
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "committed_numbers_crosscheck": committed_crosscheck,
            "composite_order": "ONSET-COMMON -> ONSET-ORG1-ONLY -> GRADED "
                               "(frozen before compute)",
            "constants": {"SHUT_BAR": SHUT_BAR, "D_grid": D_GRID,
                          "CONC_BAR": CONC_BAR,
                          "WALK_MAX_T": WALK_MAX_T},
        },
        "references": {
            "e195_rotated_ray": {"metrics": "runs/e195/metrics.json",
                                 "root_panel": {k: v for k, v in
                                                E195_ORG1.items()
                                                if k.endswith("D_kill")},
                                 "role": "org1's parent: the root ray "
                                         "profiles + registered u1/u2 md5s "
                                         "(loaded committed, gated)"},
            "e198_flight_arch": {"metrics": "runs/e198/metrics.json",
                                 "root_panel": {k: v for k, v in
                                                E198_MIR.items()
                                                if k.endswith("D_kill")},
                                 "role": "MIRABEL's parent: the root ray "
                                         "profiles + registered u1/u2 md5s "
                                         "(loaded committed, gated)"},
            "e193b_two_fact": {"metrics": "runs/e193b/metrics.json",
                               "role": "MIRABEL's organism: the root, dial, "
                                       "a_sign row + R2_SIGN rows (loaded "
                                       "committed, gated)"},
            "opt2_sign_path": {"metrics": "runs/opt2/metrics.json",
                               "D_kill": OPT2_PATH_DKILL,
                               "ckpt": str(CKPT_DIR / ORG2_SIGN_CK),
                               "role": "org1's walk parent (rebuilt, gated "
                                       "G_REPRO_ORG1/G_SAVEDSTATE_ORG1)"},
            "e194_sign_front": {"metrics": "runs/e194/metrics.json",
                                "committed_fronts": E194_FRONTS,
                                "role": "org1's front-trace parent"},
            "e192_terrain": {"metrics": "runs/e192/metrics.json",
                             "R2_md5": E192_R2_U_MD5,
                             "role": "org1's u0 parent"},
            "e196_e197_ledger": {"e196": E196_ROOT_DKILLS,
                                 "e197": E197_ALIVE,
                                 "role": "the four-lineage context (rides, "
                                         "never adjudicates)"},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 organism/fact per side (org1's e131 "
                            "ZEPHYRA line; e193b's fresh MIRABEL root), ONE "
                            "stream (seed 10902, md5-gated at every consumed "
                            "step); the onset curves are TWO single-path "
                            "biographies, not a population claim"),
            "post_death_disclosure": ("the post-death ts are NEVER read in "
                                      "this cell (org1's t=2 front is "
                                      "e195's committed post-kill read, "
                                      "loaded as context, flagged, never "
                                      "adjudicated; MIRABEL's walk stops at "
                                      "its t=3 death)"),
            "extension_arm_vacuous": ("both organisms' committed walks "
                                      "already die before the t=6 horizon "
                                      "(org1 t=2, MIRABEL t=3) — there were "
                                      "no missing steps to walk; the "
                                      "'ripening' question beyond each "
                                      "organism's death is unanswerable in "
                                      "principle on a dead state (T158's "
                                      "lesson)"),
            "unresolved_high_disclosure": ("MIRABEL's t=1 ratio is not a "
                                           "number: u1 never kills within "
                                           "[0, 3.0] (the grid is the frozen "
                                           "instrument; extending it would "
                                           "be a new license); its floor "
                                           "3.0/edge is reported as a floor "
                                           "and never adjudicated"),
            "openness": ("WHAT EACH ARM GUARANTEES: NOTHING — every family "
                         "is a static graded jump at lethal scale from a "
                         "state that could kill anywhere; the ratio "
                         "compares each ray to its OWN organism's static "
                         "edge (within-organism form only); cross-organism "
                         "ratio magnitudes are NOT compared (only each "
                         "curve's SHAPE and the one-step shift); that "
                         "openness is the point"),
            "estimator_lesson": ("no alignment read adjudicates (alignment "
                                 "predicts nothing, T157's lesson at n=4 "
                                 "lineages); G_SAMEPOINT_ORG1 carries the "
                                 "dual-estimator machinery gate; every "
                                 "number is a ruler read (the g-12 battery) "
                                 "on a stated state"),
            "projections_never_adjudicate": ("D_kill is a linear-in-D "
                                             "interpolation on measured "
                                             "grid points; the 0.70 bar is "
                                             "e198's frozen flight bar; the "
                                             "clauses were frozen before "
                                             "compute"),
            "environment_drift": ("this process reproduces org1's ENTIRE "
                                  "chain BIT-EXACTLY (0.00e+00 on every "
                                  "gate: G_T0/G_REPRO/G_SAVEDSTATE/G_FRONTS/"
                                  "the profile rows) but drifts the "
                                  "e193b/e198 (MIRABEL) chain at ~1e-7 fp32 "
                                  "— an environmental arithmetic change "
                                  "since e198's run (e198 itself reproduced "
                                  "e193b at 0.0). The MIRABEL identity gates "
                                  "therefore carry a disclosed TEXTURE tier "
                                  "(e196's G_S1CK precedent): ray cosines "
                                  "> 0.99999, walk rows < 1e-3, profiles "
                                  "~1e-5, D_kills < 1e-2 — every achieved "
                                  "tier STAMPED in its gate; the onset "
                                  "numbers adjudicated are this cell's fresh "
                                  "measurements, with the committed-set "
                                  "crosscheck in the adjudication showing "
                                  "the verdict identical under both"),
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "threads": {"org1_phase": 4, "mirabel_phase": 8},
                   "smoke": SMOKE, "torch": torch.__version__},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_onset_curve(rd / "e199_onset_curve.png", org1_onset, mir_onset,
                     org1_ctx, org1_ledger, mir_ledger, verdict)
    plot_ray_profiles(rd / "e199_ray_profiles.png", fam1, fam2, org1_ctx)
    log(f"outputs: {rd / 'metrics.json'} + 2 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_onset_curve(path, org1_onset, mir_onset, org1_ctx, org1_ledger,
                     mir_ledger, verdict):
    """THE ONSET CURVE: ratio vs t, both organisms side by side + the
    alive/death ledger that carries the biography."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8))
    ax = axes[0]
    # org1 (navy)
    pts1 = [(r["t"], r["ratio"]) for r in org1_onset
            if r["state"] == "alive" and r.get("ratio") is not None]
    ax.plot([p[0] for p in pts1], [p[1] for p in pts1], "o-", ms=8, lw=1.8,
            color="navy", label="ORGANISM 1 (e131 root; ZEPHYRA ruler)")
    ax.plot([org1_ctx["t"]], [org1_ctx["ratio_vs_own_u0"]], "o", ms=8,
            mfc="none", mec="navy", mew=1.8, ls="none",
            label=f"org1 t=2 POST-DEATH read (e195 committed, context "
                  f"only) ratio {org1_ctx['ratio_vs_own_u0']:.3f}")
    d1 = next(r for r in org1_onset if r["state"] == "death")
    ax.plot([d1["t"]], [1.0], "x", ms=13, mew=2.6, color="navy")
    ax.annotate(f"org1 walk DIES t=2\n(g-12 {d1['ruler_read']:.1e}; kill D "
                f"{d1['walk_D_kill']:.3f})", (d1["t"], 1.0),
                textcoords="offset points", xytext=(-8, 14), fontsize=7.4,
                color="navy", ha="right")
    # MIRABEL (crimson)
    pts2 = [(r["t"], r["ratio"]) for r in mir_onset
            if r["state"] == "alive" and r.get("ratio") is not None]
    ax.plot([p[0] for p in pts2], [p[1] for p in pts2], "o-", ms=8, lw=1.8,
            color="crimson", label="MIRABEL (e193b root; MIRABEL ruler)")
    r_t1 = next(r for r in mir_onset if r["t"] == 1)
    ax.plot([1], [r_t1["floor_ratio"]], "^", ms=9, mfc="none", mec="crimson",
            mew=1.8, ls="none")
    ax.annotate(f"t=1 SOFT: u1 never kills <= 3.0\n(min g-12 "
                f"{r_t1['min_read']['gm12']:.3f} at D "
                f"{r_t1['min_read']['D']:.1f}); floor ratio "
                f">{r_t1['floor_ratio']:.3f}", (1, r_t1["floor_ratio"]),
                textcoords="offset points", xytext=(10, 6), fontsize=7.4,
                color="crimson")
    ax.annotate("", xy=(1, r_t1["floor_ratio"] + 0.06),
                xytext=(1, r_t1["floor_ratio"] + 0.005),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    d2 = next(r for r in mir_onset if r["state"] == "death")
    ax.plot([d2["t"]], [1.0], "x", ms=13, mew=2.6, color="crimson")
    ax.annotate(f"MIRABEL walk DIES t=3\n(g-12 {d2['ruler_read']:.3f}; kill "
                f"D {d2['walk_D_kill']:.3f})", (d2["t"], 1.0),
                textcoords="offset points", xytext=(-6, 14), fontsize=7.4,
                color="crimson", ha="right")
    ax.axhline(CONC_BAR, ls="--", lw=1.4, color="tab:purple",
               label=f"concentration bar ({CONC_BAR}; e198's frozen flight bar)")
    ax.axhline(1.0, ls=":", lw=1.3, color="dimgray")
    ax.text(2.97, 1.015, "1.0 = the static edge (definitional)", fontsize=7,
            color="dimgray", ha="right")
    ax.set_xticks(range(0, 4))
    ax.set_xlabel("wash step t  (ray u_t = sign(g_t) read at theta_t, the "
                  "walk's ALIVE states)")
    ax.set_ylabel("concentration ratio  D_kill(root, u_t) / D_kill(root, u0)")
    ax.set_ylim(-0.05, 1.75)
    ax.set_title("THE ONSET CURVE — ratio vs t, both organisms", fontsize=11)
    ax.legend(fontsize=7.4, loc="center right")

    ax = axes[1]
    o1 = [(r["t"], r["ruler_read"]) for r in org1_ledger]
    ax.plot([p[0] for p in o1], [p[1] for p in o1], "o-", ms=7, lw=1.6,
            color="navy", label="ORGANISM 1 (ZEPHYRA g-12)")
    m1 = [(r["t"], r["ruler_read"]) for r in mir_ledger]
    ax.plot([p[0] for p in m1], [p[1] for p in m1], "o-", ms=7, lw=1.6,
            color="crimson", label="MIRABEL (g-12)")
    for ledger, col in ((org1_ledger, "navy"), (mir_ledger, "crimson")):
        for r in ledger:
            if not r["alive"]:
                ax.plot([r["t"]], [r["ruler_read"]], "x", ms=12, mew=2.6,
                        color=col)
    ax.axhline(SHUT_BAR, ls="--", lw=1.4, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar (the ruler)")
    for r in org1_ledger + mir_ledger:
        if r["alive"] and r.get("ray"):
            ax.annotate(r["ray"].split(" ")[0], (r["t"], r["ruler_read"]),
                        textcoords="offset points", xytext=(0, 9),
                        fontsize=7.6, ha="center",
                        color="navy" if r in org1_ledger else "crimson")
    ax.text(0.02, 0.045, "x = the walk's DEATH step (the post-death front "
            "never read)\nu_t = the ray read at that alive state",
            fontsize=7.4, transform=ax.transAxes, color="dimgray")
    ax.set_xticks(range(0, 4))
    ax.set_xlabel("wash step t")
    ax.set_ylabel("ruler read at theta_t (g-12 battery mean p)")
    ax.set_ylim(-0.04, 1.02)
    ax.set_title("THE ALIVE LEDGER — whose states carry rays, where the "
                 "walks die", fontsize=11)
    ax.legend(fontsize=7.6, loc="center right")
    fig.suptitle(f"E199 — THE CONCENTRATION'S ONSET CURVE (T162's timing "
                 f"cut) — VERDICT: {verdict}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ray_profiles(path, fam1, fam2, org1_ctx):
    """The underlying static ray profiles (gm12 vs D per t per organism)."""
    RAY_STYLE = {
        "u0": {"color": "navy", "label": "t=0: sign(g_0) — the static ray"},
        "u1": {"color": "crimson",
               "label": "t=1: sign(g_1) — the first rotated front"},
        "u2": {"color": "darkorange",
               "label": "t=2: sign(g_2) — the second rotated front"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, fam, title in ((axes[0], fam1,
                            "ORGANISM 1 (ZEPHYRA g-12 ruler)"),
                           (axes[1], fam2,
                            "MIRABEL (g-12 primary ruler)")):
        for key in ("u0", "u1", "u2"):
            if key not in fam:
                continue
            st = RAY_STYLE[key]
            famrows = fam[key]["rows"]
            ax.plot([r["D"] for r in famrows], [r["gm12"] for r in famrows],
                    "o-", ms=2.6, lw=1.7, color=st["color"],
                    label=st["label"])
            if fam[key]["D_kill"] is not None:
                ax.axvline(fam[key]["D_kill"], color=st["color"], ls=":",
                           lw=1.4)
                ax.text(fam[key]["D_kill"], 0.985,
                        f"{key} kills {fam[key]['D_kill']:.3f}",
                        rotation=90, fontsize=6.6, va="top", ha="right",
                        color=st["color"])
        if fam is fam1:       # the post-death context ray (committed)
            ctx_rows = org1_ctx["rows_committed"]
            ax.plot([r["D"] for r in ctx_rows], [r["gm12"] for r in ctx_rows],
                    "o--", ms=2.4, lw=1.3, color="darkorange", alpha=0.65,
                    label=f"t=2 (post-death, e195 committed, CONTEXT ONLY) "
                          f"kills {org1_ctx['D_kill']:.3f}")
            ax.axvline(org1_ctx["D_kill"], color="darkorange", ls=":", lw=1.1,
                       alpha=0.6)
        ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
                   label=f"{SHUT_BAR} SHUT bar (the ruler)")
        ax.set_xlabel(r"static displacement $D = \|\theta_D - \theta_0\|_2$")
        axr = ax.secondary_xaxis(
            "top", functions=(lambda d: d / RMS_DENOM,
                              lambda r: r * RMS_DENOM))
        axr.set_xlabel("per-coordinate RMS (the second currency)")
        ax.set_ylabel("ruler read (battery mean p)")
        ax.set_ylim(-0.03, 1.02)
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=7.0, loc="upper right")
    fig.suptitle("E199 — the per-t ray profiles from the ROOT (the onset "
                 "curve's substrate; every family re-gated vs committed)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
