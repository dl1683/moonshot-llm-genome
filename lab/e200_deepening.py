"""E200 — THE DEEPENING TEST (T163's named follow-on: e197's half-step
lineage's own onset curve — the one curve that could DEEPEN).

RECOVERY NOTE: e200 was dispatched 2026-10-01 23:02Z (2bce76d) and DIED
PRE-ARTIFACT in the ninth disruption (overnight heat shutdown ~9h; STATE's
current_experiment line). This is the RESTART under the tightened owner
envelope (2026-10-02, permanent): CPU-ONLY, torch threads CAPPED AT 4 (the
CPU is shared; e197's committed chain was threads 8 — the deviation and its
tiered identity gates are disclosed below), sequential modest phases,
PROGRESSIVE metrics.json writes after every phase (the outage lesson), n=1.

WHY (T163, verbatim): "THE ARC CLOSES AT A SHAPE: every organism so far
concentrates before it dies (org1 t=1; MIRABEL t=2; org2 full-step never —
it died at t=1 BEFORE its onset; its half-step lineage alive t=1..4 owes
the deepening test — the one curve that could fall across three alive
steps)." And T160's committed substrate: e197's half-step lineage (organism
2's root, s* = STEP_L2/2 = 0.4582, the direction machinery VERBATIM) is
ALIVE t=1..4 (g-4 0.4192 / 0.8506 / 0.5281 / 0.6482), dies at t=5
(0.1648); its t=1 flight ray did NOT concentrate (u1/u0 = 3.169 — the
ALIVE-BUT-NO-FLIGHT verdict); its t=2 front was MAPPED (e197's root_u2,
D_kill 0.3483) but NEVER ADJUDICATED as onset (e199's ledger: "its t=2..t=4
fronts were never mapped — its own onset curve is owed, named, not run
here"). THE QUESTION THIS CELL OWNS: does THIS lineage's ratio DEEPEN
across its alive window — the onset shape holding wherever the race
(onset vs death) allows time — or does it stay soft across all four alive
steps (a FOURTH timing class: alive-but-unconcentrating), or grade?

THE CELL (eval-only CPU, minutes): rebuild e197's half-step walk VERBATIM
in THIS process (the ray factory — its committed journal gated row-by-row),
read sign(g_t) rays at EVERY ALIVE t (t=1..4; the post-kill t=5/t=6 fronts
are NEVER read as rays, e199's convention), and map each ray from the ROOT
by static graded jumps (theta_D = root - D*u_t, e192/e195/e197's placement
VERBATIM, grid 0.05..3.00 step 0.05), kill-D per e192's interpolated
downcrossing convention, ratio vs its OWN u0 edge (T153's within-organism
form). The RECOMPUTATION CHECK AT ITS DEATH (t=5): the walk's densified
kill D vs the static u0 edge — e197's committed instrument (ratio 1.5969,
absent, bar 0.85), re-anchored here as the death-point context (never an
e200 bar — it is already adjudicated in e197).

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - ONSET-DEEPENS: "the ratio falls monotonically or falls-then-holds
    across the alive window — the onset shape holds wherever the race
    allows time."
  - ONSET-STALLS: "the ratio stays soft across all four alive steps — a
    fourth timing class: alive-but-unconcentrating."
  - GRADED: "any mix."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * ALIVE t = the rebuilt walk's every-step primary-ruler (g-4 install-60)
    read at theta_t > SHUT (0.27, e185's bar, absolute). Rays are read
    ONLY at alive ts; u_t = sign(g_t)/||sign(g_t)|| (e192's fp32-norm
    construction) with g_t the licensed stream's step-(t+1) post-clip
    refresh gradient computed AT theta_t. The post-kill step-6 gradient is
    computed only to reproduce e197's committed journal row and is NEVER
    read as a ray.
  * ratio_t = D_kill(root, u_t) / D_kill(root, u0) — THIS lineage vs its
    OWN u0 edge; ratio_0 = 1.0 by definition (u0 vs itself; the definitional
    point is EXCLUDED from monotonicity clauses — e199's MIRABEL precedent:
    its curve 1.0 -> SOFT(higher) -> 0.427 still fired ONSET-COMMON).
  * KILL = the family profile's first primary-ruler <= 0.27 downcrossing on
    the static grid, linear-in-D interpolated (e192/e194/e195/e197/e199's
    convention); None = no downcrossing within [0, 3.0].
  * a ray CONCENTRATES at t iff ratio_t < 0.70 (e198's frozen flight bar,
    e199's concentration bar); a None kill is SOFT (unresolved-high; the
    floor ratio 3.0/edge is reported as a floor, never adjudicated).
  * "the ratio stays soft across all four alive steps" = NO alive t in
    {1,2,3,4} concentrates (every ratio None or >= 0.70).
  * "the ratio falls monotonically or falls-then-holds across the alive
    window" = (i) some alive t in {1,2,3,4} CONCENTRATES (the onset SHAPE
    is T163's shape — soft first, then the rotation lands on the lethal
    direction; a bare decline that never reaches 0.70 is NOT the onset
    shape holding — that case grades), AND (ii) every RESOLVED ratio
    strictly after the first concentrated t stays <= 0.70 (the
    concentration, once arrived, does not un-form inside the alive
    window — falls-then-HOLDS; vacuous with no later alive t), AND
    (iii) the RESOLVED ratios from t=1 through the first concentrated t
    are non-increasing (the FALL; unresolved SOFT ts are skipped, e199's
    MIRABEL precedent).
  * the RECOMPUTATION CHECK AT DEATH (t=5) = the walk's densified kill D vs
    the static u0 edge, e197's instrument VERBATIM (bar 0.85, the day's 15%
    materiality margin); reported at the death point as context, never an
    e200 bar.
  * composite order frozen: ONSET-DEEPENS -> ONSET-STALLS -> GRADED (the
    first that fires is the verdict; every bar's fires flag reported
    verbatim; the three are mutually exclusive by construction).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): e197's
gate set REUSED VERBATIM (its own port of e193's): G_NAMEFREE, G_SPLICE,
G_BATTERY at the e157 dial's seven geometries, G_ANCHOR vs e185's stored
bank, G_ROOT — the f2 root's dial reproduces e157's committed cells
BIT-TIGHT + the flat md5 matches e193's committed root identity, G_T0 —
the step-1 batch md5 + forward CE + the MEASURED step-1 L2, G_S1CK,
G_STREAM, G_DIRCK — e193's committed g-ray checkpoint, G_SIGNRAY — u0
rebuilt md5 == e193's committed R2_SIGN md5, G_REPRO — the FULL-STEP k=1
walk rebuilt and gated vs e193's committed a_sign step-1 row + densified
kill bracket (the dead-at-t=1 baseline) PLUS THE NEW ARM GATES: G_WALKE197
(the HALF-STEP walk rebuilt vs e197's committed journal — every row's
ce_batch/cum_disp/step_disp/preclip_gnorm/gm/frac_argmax_z/co-rulers, the
x-hashes, the stop row and its densified bracket), G_RAYS (u1/u2 md5 vs
e197's registered rays; u3/u4 registered HERE as new directions with their
mutual geometry), G_ALIVE (the alive window re-anchored: t=1..t=4 alive,
t=5 dead, matching e197's committed flags) and G_PROF (the recomputed
root-u0/u1/u2 families reproduce e197's committed profile rows at ALL
shared Ds + their D_kills, before any new point is believed; root-u0's
chain to e193's committed R2_SIGN rows rides through e197's own committed
gate transitively, disclosed).
IDENTITY TIERS (frozen before the full compute; e199's disclosed
convention): BIT = md5 / 1e-9 (same-code-path arithmetic) where this
process reproduces the parents' bits — e197's chain was committed at
threads 8 and this process runs threads 4 (the owner envelope's cap), so
cross-thread/environment drift at ~1e-7 fp32 is POSSIBLE; the disclosed
TEXTURE tier then gates (walk rows < 1e-3, profile rows < 1e-3, ray mutual
geometry < 1e-3, D_kills < 1e-2, sign-flip coords < 1000); the achieved
tier is STAMPED in every gate, never silently weakened; the adjudicated
onset numbers are THIS cell's fresh measurements, with the
committed-numbers crosscheck reported alongside.

REGISTERED PREDICTION (frozen before compute): T163's ONSET-COMMON shape —
plus e197's committed-but-unadjudicated root_u2 (D_kill 0.3483, ratio
0.6631 < 0.70: the t=2 wrinkle ON THIS LINEAGE TOO) — predicts
ONSET-DEEPENS iff the concentration ARRIVES at t=2 (fresh) and HOLDS
through t=3/t=4. The competing prior is T160's honest fork: this lineage
is a COUNTERFACTUAL wash (the natural step 0.9164 is what killed the
organism; the experimenter halves it) whose t=1 dynamics never formed —
its rotation may not track the support, so the t=2 concentration could
un-form or the approach could rise (GRADED). ONSET-STALLS requires the
fresh t=2 read to contradict e197's committed 0.6631 — registered as the
unlikely branch, disclosed. No bar shopping either way; the fresh reads
adjudicate and the committed crosscheck reports robustness.

WHAT EACH ARM GUARANTEES: NOTHING — every family is a static graded jump
at lethal scale from a state that could kill anywhere; the walk is one
realized path of a stochastic stream read through a counterfactual step
size; the openness is the point.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never claimed),
torch threads 4 (the cap; e197's convention was 8 — disclosed above),
sequential phases, every phase modest, PROGRESSIVE partial metrics.json
writes after every phase (the outage lesson), n=1, single stream seed
10902 lineage, organism 2 of 2.

Outputs: runs/e200/{metrics.json, e200_deepening.png (the four-organism
onset picture), e200_ray_profiles.png}. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds).

Run:  cd lab && python e200_deepening.py    (E200_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e193's convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # THE OWNER ENVELOPE'S CAP (e197's was 8; disclosed)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E200_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e200 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                       # training-window block (the e098 line's own convention)
ROOT_CK = "e157_f2_consolidated.pt"   # THE FAMILY-2 CONSOLIDATED ROOT (e157's, committed)
S1_CK = "e157_f2_neutral_s1.pt"       # the committed wash step-1 checkpoint (the G_T0/G_S1CK anchor)
DIR_CK = "e193_f2_static_dir_u.pt"    # e193's committed g-ray direction checkpoint
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E157_METRICS = E43.REPO / "runs" / "e157" / "metrics.json"
E193_METRICS = E43.REPO / "runs" / "e193" / "metrics.json"
E196_METRICS = E43.REPO / "runs" / "e196" / "metrics.json"
E197_METRICS = E43.REPO / "runs" / "e197" / "metrics.json"
E199_METRICS = E43.REPO / "runs" / "e199" / "metrics.json"

# ---- the family-2 net (e098 s4305 line: 4L/4H/128d/512-ctx, 873,472 params) ----
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- rulers (e193's, frozen before its compute; carried verbatim) -----------------
RULER_J = -4                     # PRIMARY: g-4 install-60 battery (max committed root read)
CO_RULERS_J = (12, -12, 0)       # novel-side analog / e192-verbatim / install dial
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries (gates)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193/e197 family-2)
LR_ADAMW = 1e-3                   # the t=0 recipe (e157/e193's wash; the ray arms have NO lr)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (e195/e197/e199's grid VERBATIM)
CE_EVERY = 0.2                    # ce_r at D in {0.2, 0.4, ..., 3.0} (e195's decimation)
CONC_BAR = 0.70                   # frozen: "concentrates" = ratio < 0.70 (e198/e199's bar)
BONUS_RATIO_BAR = 0.85            # e197's recomputation-bonus bar (the death check, never an e200 bar)
EXT_STEPS = 1                     # read-only steps past the sub-step walk's kill (e197's convention)
WALK_CAP = 8                      # step cap (e197's; the committed walk ran 6 steps)
WALK_D_TARGET = 5.0               # e193's target convention
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_STATIC_TOL = 1e-3               # committed-row crosscheck class (e195/e199's G_ROOTPROF)
DKILL_TOL = 1e-6                  # D_kill equality vs committed (BIT)
DKILL_TEXTURE_TOL = 1e-2          # D_kill equality vs committed (TEXTURE; e199's tier)
WALK_TEXTURE_TOL = 1e-3           # walk-row reproduction tier (e199's)
RAY_COS_TOL = 1e-3                # mutual-geometry cosine tier (e199's)
RAY_SIGN_COS_FLOOR = 0.99999      # sign-ray cosine floor (e199's)
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
RMS_DENOM = 934.597239456655      # sqrt(873472) — organism 2's per-coordinate RMS currency
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    CE_EVERY = None

# ---- e157's committed lineage reads (e193's copies, verbatim) ----------------------
E157_DIAL = {
    "gm12": 0.19826222956180573, "g-8": 0.8519253134727478,
    "g-4": 0.8872273564338684, "g0": 0.5784125924110413,
    "g+4": 0.8801683187484741, "g+8": 0.8754011988639832,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
E157_WASH_S1 = {"corpus_ce": 1.7748003005981445,
                "gm12": 0.001129803480580449}
ROOT_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"
E193_STEP_L2 = 0.9164195656776428          # THIS organism's measured AdamW step-1 L2
E193_ASIGN_S1 = {                          # e193's committed a_sign step-1 row (G_REPRO)
    "ce_batch": 1.7748003005981445,
    "cum_disp": 0.9160122275352478,
    "step_disp": 0.916012167930603,
    "preclip_gnorm": 2.299248695373535,
    "gm": 0.006826397497206926,
    "frac_argmax_z": 0.0,
}
E193_ASIGN_STOP = {                        # e193's committed a_sign stop (G_REPRO)
    "kind": "kill", "step": 1,
    "gm_at_kill": 0.006826397497206926,
    "D_kill_raw": 0.6421935368466083,
    "D_kill": 0.5257438700572483,
    "dens": [(0.2, 0.785190224647522, 0.1833570897579193),
             (0.4, 0.6062278151512146, 0.3667244017124176),
             (0.6, 0.21901635825634003, 0.5498566627502441),
             (0.8, 0.03626079857349396, 0.7333946824072292)],
}
E193_UG_MD5 = "46e71718b2ee67f2cd87f32372652baf"
E193_USIGN_MD5 = "9395918d36425e65248dbd93b6fd50bc"

# ---- e197's committed lineage reads (THE PARENT; hard-bound, drift-asserted) -------
E197_SUB_S = 0.4582097828388214             # the committed pick: STEP_L2/2 (registered before ITS walk)
E197_SUB_FRAC = 0.5
E197_PROBE_GM = 0.41916516423225403         # the committed static first-landing probe (= walk s1 gm)
E197_U1_MD5 = "3ce920e0dde4342133f466edc27c5138"   # e197's registered ALIVE flight ray (t=1)
E197_U2_MD5 = "f333e7f8c48f0d59364bfa659c2a82f2"   # e197's registered t=2 front
E197_ROOT_DKILLS = {                        # e197's committed root panel (the onset anchors)
    "root_u0": 0.5252331635450007,
    "root_u1": 1.6644735674638536,
    "root_u2": 0.3482935934103707,
}
E197_WALK_STOP = {                          # e197's committed walk stop (kill at t=5)
    "kind": "kill", "step": 5,
    "gm_at_kill": 0.1648186892271042,
    "D_kill_raw": 0.8303853930774336,
    "D_kill": 0.838723200837794,
}
E197_BONUS_RATIO = 1.596858803006513         # e197's committed recomputation check (absent)
E197_FLIGHT_RATIO = 3.1690184150400578       # e197's committed t=1 flight ratio (absent)
E197_GEOM = {                               # e197's committed ray mutual geometry (fp64)
    "cos_u0_u1": -0.26317857219197976,
    "cos_u0_u2": 0.1405511960995524,
    "cos_u1_u2": -0.3128523818598815,
}
E197_ALIVE_GMS = {1: 0.41916516423225403, 2: 0.8505910634994507,
                  3: 0.5280560255050659, 4: 0.6481676697731018,
                  5: 0.1648186892271042}     # t=5 = the DEATH step (its front never read)

# ---- the four-organism committed onset ledger (e195/e196/e198/e199; context, never
#      re-run here; hard-bound and asserted vs the parents' metrics at load) ---------
E195_ORG1 = {"root_u0_D_kill": 2.269916581032063,
             "root_u1_D_kill": 0.3875040789853107,
             "root_u2_D_kill": 0.8979436419840907}      # u2 = org1's POST-DEATH t=2 read
E198_MIR = {"root_u0_D_kill": 1.9471128428088909,
            "root_u2_D_kill": 0.8313359802130567}       # t=2 concentrated; t=1 None (SOFT)
E196_DEAD = {"root_u0_D_kill": 0.5252331635450007,
             "root_u1_D_kill": 1.3575583149916892}      # the DEAD-lineage u1 (post-kill read)
E199_ORG1_T1 = 0.1707129161588503                       # e199's committed fresh org1 t=1 ratio
E199_MIR_T2 = 0.42691585091139433                       # e199's committed fresh MIRABEL t=2 ratio
E199_MIR_T1_FLOOR = 1.5407588035110036                  # MIRABEL t=1 SOFT floor (3.0/edge)

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
    5: "4da7b67a7fd80b4e9729731fb27bec0c",
    6: "1aa4f9f250f14acad52d3b969343090a",
    7: "1e9e3373028935fe272d86683280e4b5",
    8: "3535a9db2d1aa2e6e0655208ff26b3d9",
}

REGISTERED_BARS = {
    "ONSET_DEEPENS": "ONSET-DEEPENS: \"the ratio falls monotonically or "
        "falls-then-holds across the alive window — the onset shape holds "
        "wherever the race allows time.\"",
    "ONSET_STALLS": "ONSET-STALLS: \"the ratio stays soft across all four "
        "alive steps — a fourth timing class: alive-but-unconcentrating.\"",
    "GRADED": "GRADED: \"any mix.\"",
    "operationalizations": "ALIVE t = the rebuilt walk's every-step primary "
        "(g-4) read at theta_t > 0.27; rays read ONLY at alive ts (the "
        "post-kill step-6 gradient is computed only to reproduce e197's "
        "committed journal row and is NEVER read as a ray); u_t = "
        "sign(g_t)/||sign(g_t)|| (e192's fp32-norm construction), g_t = the "
        "licensed stream's step-(t+1) post-clip refresh gradient AT theta_t; "
        "ratio_t = D_kill(root,u_t)/D_kill(root,u0) vs THIS lineage's OWN "
        "edge (ratio_0 = 1.0 definitional, EXCLUDED from monotonicity — "
        "e199's MIRABEL precedent); KILL = the profile's first primary "
        "<= 0.27 downcrossing on the static grid 0.05..3.00, linear-in-D "
        "interpolated; a ray CONCENTRATES at t iff ratio_t < 0.70 (e198/"
        "e199's frozen bar); a None kill is SOFT (unresolved-high; floor "
        "3.0/edge reported, never adjudicated); 'stays soft across all "
        "four alive steps' = no alive t in {1,2,3,4} concentrates; 'falls "
        "monotonically or falls-then-holds' = (i) some alive t CONCENTRATES "
        "(the onset SHAPE — T163's: soft first, then the rotation lands; a "
        "bare decline never reaching 0.70 is NOT the shape holding — it "
        "grades), AND (ii) every resolved ratio strictly after the first "
        "concentrated t stays <= 0.70 (falls-then-HOLDS; the concentration "
        "does not un-form inside the alive window), AND (iii) the RESOLVED "
        "ratios from t=1 through the first concentrated t are non-increasing "
        "(the FALL; unresolved SOFT ts skipped); the RECOMPUTATION CHECK AT "
        "DEATH (t=5) = the walk's densified kill vs the static u0 edge "
        "(e197's instrument VERBATIM, bar 0.85) — reported at the death "
        "point as context, NEVER an e200 bar; composite order frozen "
        "ONSET-DEEPENS -> ONSET-STALLS -> GRADED (mutually exclusive by "
        "construction).",
    "registered_prediction": "T163's ONSET-COMMON shape — plus e197's "
        "committed-but-unadjudicated root_u2 (D_kill 0.3483, ratio 0.6631 < "
        "0.70: the t=2 wrinkle ON THIS LINEAGE TOO) — predicts ONSET-DEEPENS "
        "iff the concentration ARRIVES at t=2 (fresh) and HOLDS through "
        "t=3/t=4. The competing prior is T160's honest fork: this lineage is "
        "a COUNTERFACTUAL wash (the natural step 0.9164 is what killed the "
        "organism; the experimenter halves it) whose t=1 dynamics never "
        "formed — its rotation may not track the support, so the t=2 "
        "concentration could un-form or the approach could rise (GRADED). "
        "ONSET-STALLS requires the fresh t=2 read to contradict e197's "
        "committed 0.6631 — registered as the unlikely branch, disclosed. "
        "No bar shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "RECOVERY RESTART: e200 died pre-artifact in the ninth disruption "
    "(overnight heat shutdown ~9h); this is the restart under the tightened "
    "owner envelope — CPU-ONLY, torch threads CAPPED AT 4, sequential "
    "modest phases, progressive metrics writes after every phase.",
    "THREADS 4 (the owner envelope's cap), where e197's committed chain ran "
    "threads 8: cross-thread/environment drift at ~1e-7 fp32 is possible, so "
    "every identity gate carries e199's disclosed TIER system (BIT md5/1e-9 "
    "or TEXTURE at e199's tolerances), the achieved tier STAMPED per gate; "
    "the adjudicated onset numbers are this cell's fresh measurements, with "
    "the committed-numbers crosscheck reported alongside (verdict must be "
    "readable under both sets).",
    "THE RULER CALL (e193/e197's, carried verbatim): the e192-verbatim g-12 "
    "ruler reads 0.1983 at this root — UNDER the 0.27 bar at D=0. PRIMARY "
    "RULER = g-4 install-60 battery (root read 0.8872); co-rulers g+12, "
    "g-12, g+0 on every row and every walk step, never adjudicated.",
    "THE COUNTERFACTUAL-WASH DEVIATION (carried from e197, load-bearing): "
    "this lineage's walk keeps the direction machinery VERBATIM (k=1 "
    "post-clip sign refresh on the licensed seed-10902 stream) at HALF the "
    "natural step (s* 0.4582 = STEP_L2/2, e197's committed pick, registered "
    "before ITS walk); any onset shape found belongs to the alive window OF "
    "THIS CONSTRUCTION — the step back to the natural trajectory (dead at "
    "t=1, e193/e196) is an inference, disclosed.",
    "the walk is REBUILT (not loaded): e197 saved no walk checkpoints (its "
    "rays and endpoints are bit-rebuildable from the licensed stream — "
    "e197's own convention); the rebuilt journal is gated row-by-row vs "
    "e197's committed phase1_alive_walk journal (G_WALKE197) before any ray "
    "is believed; u1/u2 are md5-gated vs e197's registered rays; u3/u4 are "
    "registered HERE as new directions (their only parents are the gated "
    "walk).",
    "root family grids are [0.0] + the base D grid (61 points): e197's "
    "root_u0 carried 6 extra landing Ds (its ladder probes + landings — all "
    "OUTSIDE the kill brackets); the gates therefore cover the 61 shared "
    "base Ds per family; every committed D_kill fell inside base-grid "
    "brackets (verified against e197's rows before this freeze), so the "
    "D_kill comparisons are unaffected — disclosed.",
    "NO ALIGNMENT READS (e199's lesson carried): alignment predicts nothing "
    "(T157, n=4 lineages); every adjudicated number is a ruler read (the "
    "g-4 battery) on a stated state; the dual-estimator lesson rides as the "
    "absence of any point-ambiguous read.",
    "ce_r (second currency) decimated to 0.2-multiples + D=0 (e195's "
    "decimation); the g-4 battery is the ruler, read at EVERY grid point "
    "with the three co-rulers on every row; dual displacement currency (D "
    "and per-coordinate RMS / 934.597) on every row.",
    "the four-organism onset picture loads org1/MIRABEL/org2-dead curves "
    "COMMITTED (e195/e198/e196/e199 — hard-bound, asserted at load, never "
    "re-run here); org2's DEAD full-step lineage never had an alive ray "
    "beyond t=0 (its theta_1 died at 0.0068) — it appears as the death-at-"
    "t=1 pole, with e196's post-kill u1 ratio 2.585 flagged CONTEXT ONLY.",
    "Smoke mode trims: grid {0.05, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, "
    "2.0, 3.0}, ce at anchors only (verdict stamped SMOKE; nothing "
    "adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e197_alive_window.py's instruments VERBATIM (whose own
# provenance is lab/e196_flight_replicate.py / lab/e193_organism_replicate.py
# + lab/e195_rotated_ray.py — the e176n lineage). Copied rather than imported
# to own the device policy and the arithmetic.

def load_f2(path) -> TinyGPT:
    m = TinyGPT(F2_CFG)
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
    """The fp32 flat parameter vector (net.parameters() order; 873,472)."""
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
    """e193/opt2's sign_update VERBATIM: delta = -step_l2 * sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — the matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


# ------------------------------------------------------------------ kill edges

def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    """The densified bracket (opt1c/e193's convention; this value gates)."""
    pts = [(d0, v0)] + [(r["D"], r["gm"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) + any upcrosses (context)."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm"] <= SHUT_BAR and a["gm"] > SHUT_BAR:
            edge = interp_d_kill(a["gm"], b["gm"], a["D"], b["D"])
            break
    ups = [(rows[i - 1]["D"], rows[i]["D"]) for i in range(1, len(rows))
           if rows[i]["gm"] > SHUT_BAR and rows[i - 1]["gm"] <= SHUT_BAR]
    return edge, ups


def _dk_ok(meas, committed, tol):
    """None-safe D_kill equality vs a committed value at a tier."""
    return (meas is None and committed is None) or \
        (meas is not None and committed is not None
         and abs(meas - committed) < tol)


# ------------------------------------------------------------------ the walk

def walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids, coruler_ids,
             zid, theta0, step_l2, root_gm, step_cap=WALK_CAP,
             d_target=WALK_D_TARGET, extend_past_kill=EXT_STEPS, tag="sub"):
    """e197's walk_sub VERBATIM in its arithmetic (its own port of e193's
    run_arm): every step refreshes the front (k=1, post-clip); every-step
    primary battery + co-rulers; densified kill bracket at the killing step;
    stashes g_fronts + endpoints (the ray factory)."""
    net = copy.deepcopy(net0)
    net.train()
    evl_a = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    step, kill = 0, None
    traj, x_hashes = [], {}
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_flat, prev_d = theta0, 0.0
    t0a = time.time()
    while True:
        step += 1
        post_kill = kill is not None
        if post_kill and step > kill["step"] + extend_past_kill:
            break
        if step > step_cap and kill is None:
            kill = {"kind": "cap", "step": step - 1,
                    "reason": f"step cap {step_cap} alive",
                    "final_D": traj[-1]["cum_disp"],
                    "final_gm": traj[-1]["gm"]}
            break
        aj = torch.randint(16, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
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
        gz = battery_cell(evl_a, primary_ids, zid)
        row = {"step": step, "post_kill": post_kill,
               "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"]}
        for j in coruler_ids:                     # per-step co-rulers (e197's)
            row[f"g{j:+d}"] = battery_cell(evl_a, coruler_ids[j],
                                           zid)["mean_pz"]
        traj.append(row)
        g_fronts.append(g_t.clone())               # stash (the ray factory)
        endpoints[step] = cur.clone()              # stash
        prev_flat, prev_d = cur, cum_disp
        log(f"  [{tag} s{step}]{' POSTKILL' if post_kill else ''} "
            f"g-4 {row['gm']:.10f} g-12 {row.get('g-12', float('nan')):.6f} "
            f"D {cum_disp:.10f} ce {row['ce_batch']:.6f}")
        if kill is None and gz["mean_pz"] <= SHUT_BAR:
            dens = []
            for f_ in (0.2, 0.4, 0.6, 0.8):
                pt_flat = prev_flat + (f_ - 1.0) * delta  # f along the step
                load_flat(evl_a, pt_flat)
                evl_a.eval()
                gzd = battery_cell(evl_a, primary_ids, zid)
                dens.append({"f": f_, "gm": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm, "cum_disp": 0.0}   # e193's fallback VERBATIM
            d_raw = interp_d_kill(prev_row["gm"], row["gm"],
                                  prev_row["cum_disp"], cum_disp)
            d_dens = dens_d_kill(prev_row["gm"], row["gm"],
                                 prev_row["cum_disp"], cum_disp, dens)
            kill = {"kind": "kill", "step": step,
                    "gm_at_kill": row["gm"], "D_kill_raw": d_raw,
                    "D_kill": d_dens, "dens": dens,
                    "reason": "primary ruler <= SHUT at an every-step read"}
            log(f"  [{tag}] KILL at s{step}: D_kill(dens) {d_dens:.10f}")
        elif kill is None and cum_disp >= d_target:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {d_target} alive",
                    "final_D": cum_disp, "final_gm": row["gm"]}
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "stop": kill, "x_hashes": x_hashes,
            "max_l2_dev": max_l2_dev, "g_fronts": g_fronts,
            "endpoints": endpoints, "seconds": round(time.time() - t0a, 1)}


# ------------------------------------------------------------------ profiles

def static_profile(evl, anchor_flat, u, dgrid, primary_ids, coruler_ids,
                   zid, r_eval_xy, ce_ds):
    """Static graded jumps theta_D = anchor - D*u (e192's placement
    VERBATIM, e197's static_profile), every point read on the primary ruler
    + the three co-rulers; ce_r (second currency) at the frozen decimated
    Ds; dual displacement currency on every row."""
    rows = []
    for D in dgrid:
        thD = anchor_flat - D * u
        disp_check = float(torch.norm(thD - anchor_flat))
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, primary_ids, zid)
        row = {"D": float(D), "rms": float(D) / RMS_DENOM,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
               "disp_check": disp_check, "disp_dev": abs(disp_check - float(D))}
        for j in coruler_ids:
            row[f"g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        if ce_ds is not None and any(abs(D - c) < 1e-9 for c in ce_ds):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
        rows.append(row)
    return rows


def gate_profile(rows, committed_rows, gm_key, label, tol=G_STATIC_TOL):
    """Row-wise crosscheck vs a committed family (e195/e199's form)."""
    by_D = {round(r["D"], 6): r for r in committed_rows}
    xc = []
    for row in rows:
        key = round(row["D"], 6)
        if key in by_D:
            xc.append({"D": key, "gm_measured": row["gm"],
                       "gm_committed": by_D[key][gm_key],
                       "abs_diff": abs(row["gm"] - by_D[key][gm_key])})
    return {"label": label, "n_shared": len(xc), "rows": xc,
            "max_abs_diff": (max(r["abs_diff"] for r in xc) if xc else None),
            "tol": tol,
            "pass": bool(xc and all(r["abs_diff"] < tol for r in xc))}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e200_smoke" if SMOKE else "e200")
    log(f"E200 THE DEEPENING TEST (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()} — the owner "
        f"envelope's cap; e197's chain was threads 8, tiered gates), "
        f"progressive writes, n=1, organism 2 of 2, stream seed {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for name, p in (("e157", E157_METRICS), ("e193", E193_METRICS),
                    ("e196", E196_METRICS), ("e197", E197_METRICS),
                    ("e199", E199_METRICS)):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    e157m = json.loads(E157_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    e196m = json.loads(E196_METRICS.read_text(encoding="utf-8"))
    e197m = json.loads(E197_METRICS.read_text(encoding="utf-8"))
    e199m = json.loads(E199_METRICS.read_text(encoding="utf-8"))
    # hard-bind the committed references (asserts catch committed-file drift)
    dial = e157m["stages"]["A_consolidate"]["dial"]
    jit = e157m["stages"]["A_consolidate"]["consolidated_jitter_geos"]
    assert abs(dial["base"]["-12"]["mean_pz"] - E157_DIAL["gm12"]) < 1e-12
    assert abs(dial["base"]["0"]["mean_pz"] - E157_DIAL["g0"]) < 1e-12
    assert abs(dial["base"]["12"]["mean_pz"] - E157_DIAL["gp12"]) < 1e-12
    assert abs(dial["ce_r"] - E157_DIAL["ce_r"]) < 1e-12
    for j, k in ((-8, "g-8"), (-4, "g-4"), (4, "g+4"), (8, "g+8")):
        assert abs(jit[f"g{j:+d}"] - E157_DIAL[k]) < 1e-12
    assert abs(e193m["organism"]["step_l2_measured"] - E193_STEP_L2) < 1e-12
    asign = e193m["ladder"]["arms"]["a_sign"]
    s1c, stopc = asign["traj"][0], asign["stop"]
    assert s1c["step"] == 1 and stopc["kind"] == "kill" and stopc["step"] == 1
    for k in ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm", "gm"):
        assert abs(s1c[k] - E193_ASIGN_S1[k]) < 1e-12, f"e193 s1 {k} drift"
    assert abs(stopc["D_kill"] - E193_ASIGN_STOP["D_kill"]) < 1e-12
    # e197 (THE PARENT): the walk journal, stop, rays, profiles, panel
    e197_walk = e197m["phase1_alive_walk"]
    e197_journal = e197_walk["journal"]
    assert [r["step"] for r in e197_journal] == [1, 2, 3, 4, 5, 6]
    for t, gm in E197_ALIVE_GMS.items():
        assert abs(e197_journal[t - 1]["gm"] - gm) < 1e-12, f"e197 t{t} gm drift"
    assert e197_walk["stop"]["kind"] == "kill" \
        and e197_walk["stop"]["step"] == E197_WALK_STOP["step"] \
        and abs(e197_walk["stop"]["D_kill"] - E197_WALK_STOP["D_kill"]) < 1e-12
    for k, v in E197_ROOT_DKILLS.items():
        assert abs(e197m["profiles"][k]["D_kill"] - v) < 1e-12, f"e197 {k} drift"
    assert e197m["cell"]["rays"][1]["u_md5"] == E197_U1_MD5
    assert e197m["cell"]["rays"][2]["u_md5"] == E197_U2_MD5
    assert e197m["adjudication"]["verdict"] == "ALIVE-BUT-NO-FLIGHT"
    assert abs(e197m["adjudication"]["flight_check"]["ratio"]
               - E197_FLIGHT_RATIO) < 1e-12
    assert abs(e197m["adjudication"]["recomputation_check"]["ratio"]
               - E197_BONUS_RATIO) < 1e-12
    for k, v in E197_GEOM.items():
        assert abs(e197m["phase1b_rays"]["ray_geometry"][k] - v) < 1e-12
    e197_xh = e197_walk["x_hashes"]
    # e196/e199: the four-organism ledger's committed anchors
    for k, v in E196_DEAD.items():
        assert abs(e196m["adjudication"]["root_panel"][f"root_{k.split('_')[1]}"]
                   - v) < 1e-12, f"e196 {k} drift"
    oc = e199m["onset_curves"]
    assert abs(oc["org1"][1]["ratio"] - E199_ORG1_T1) < 1e-12
    assert abs(oc["mirabel"][2]["ratio"] - E199_MIR_T2) < 1e-12
    assert abs(oc["mirabel"][1]["floor_ratio"] - E199_MIR_T1_FLOOR) < 1e-12
    assert e199m["adjudication"]["verdict"] == "ONSET-COMMON"
    log("parents: e157 (the dial + wash s1 anchors), e193 (the organism: "
        f"STEP_L2 {E193_STEP_L2:.10f}, a_sign step-1 kill — the DEAD-at-t=1 "
        "baseline), e197 (THE PARENT: the half-step walk journal t=1..6, the "
        "kill at t=5, rays u1/u2 md5s, the root panel u0 0.5252 / u1 1.6645 "
        "/ u2 0.3483 — the last UNADJUDICATED as onset), e196 (org2's dead "
        "full-step lineage), e199 (the onset method + the other organisms' "
        "committed curves) — loaded COMMITTED, never rerun")

    # ---------------- protocol rebuild (e193/e197 verbatim) --------------------
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
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in READ_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in READ_GEOS},
        "pass": bool(all(list(bat_ids[j].shape) == [60, PRE + j]
                         for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at the "
                "e157 dial's seven read geometries, shapes 60 x (130 +- j) "
                "— e193/e197's gate verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]
    coruler_ids = {j: bat_ids[j] for j in CO_RULERS_J}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # the neutral stream (e170 VERBATIM via e185/e193/e197)
    arng = _random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                         "in [s, s+257) — e170 VERBATIM"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    # ---------------- the family-2 root + G_ROOT (bit vs e157's dial) ---------
    net0 = load_f2(CKPT_DIR / ROOT_CK)
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    root_meta = E43.jsonable(st_raw.get("meta", {}))
    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    assert N_PARAM == F2_PARAMS, f"params {N_PARAM} != {F2_PARAMS}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()

    evl0 = copy.deepcopy(net0)
    root_cells = {f"g{j:+d}": battery_cell(evl0, bat_ids[j], zid)["mean_pz"]
                  for j in READ_GEOS}
    root_cells["ce_r"] = ce_fixed_cpu(evl0, *r_eval_xy)
    keymap = {"gm12": "g-12", "g0": "g+0", "gp12": "g+12",
              "g-4": "g-4", "g+4": "g+4", "g-8": "g-8", "g+8": "g+8",
              "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E157_DIAL.items()}
    rdiffs = {k: root_cells[k] - root_refs[k] for k in root_refs}
    rmax = max(abs(v) for v in rdiffs.values())
    G_ROOT = {"cells": root_cells, "refs": root_refs, "diffs": rdiffs,
              "max_abs_diff": rmax, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax < G_BIT_TOL),
              "flat_md5": root_md5,
              "flat_md5_match_e193_committed": bool(root_md5 == ROOT_FLAT_MD5),
              "pass": bool(rmax < G_FALLBACK_TOL
                           and root_md5 == ROOT_FLAT_MD5),
              "note": "e193/e197's G_ROOT VERBATIM: the f2 consolidated "
                      "root's dial reproduces e157's committed cells "
                      "BIT-TIGHT + the flat md5 matches e193's committed "
                      "root identity"}
    log(f"G_ROOT (vs e157 committed dial, 8 cells): max|diff| {rmax:.2e}, "
        f"flat md5 "
        f"{'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("family-2 root gate FAILED vs e157 committed dial")

    # ---------------- G_T0 / G_S1CK (e193/e197's t=0 gates, verbatim) ---------
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
    evl_t1a = copy.deepcopy(net0)
    load_flat(evl_t1a, theta1_adam)
    t1_primary = battery_cell(evl_t1a, primary_ids, zid)["mean_pz"]

    s1_net = load_f2(CKPT_DIR / S1_CK)
    s1_meta = E43.jsonable(torch.load(CKPT_DIR / S1_CK, map_location="cpu",
                                      weights_only=False).get("meta", {}))
    theta_s1 = flat_params(s1_net)
    d_fresh = theta1_adam - theta0
    d_ck = theta_s1 - theta0
    cos_s1 = cos64(d_fresh, d_ck)
    rel_l2 = abs(float(torch.norm(d_ck)) - disp1) / disp1
    G_T0 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1, "ce_batch_committed_e157":
            E157_WASH_S1["corpus_ce"],
        "d_ce": abs(ce1 - E157_WASH_S1["corpus_ce"]),
        "preclip_gnorm_measured": gn1, "clip_binds": bool(gn1 > 1.0),
        "adamw_step1_L2_measured": disp1,
        "committed_e193_step_l2": E193_STEP_L2,
        "d_step_l2": abs(disp1 - E193_STEP_L2),
        "s1_ckpt_disp_L2": float(torch.norm(d_ck)),
        "poststep_primary_read": t1_primary,
        "committed_e193_a_sign_s1_gm": E193_ASIGN_S1["gm"],
        "note": "e193/e197's G_T0 VERBATIM: the step-1 batch md5 vs e185's "
                "stored hash (net-independent); the forward CE vs e157's "
                "committed wash step-1 corpus CE; the fresh AdamW step's "
                "L2 vs e193's committed MEASURED step L2 (never ported)",
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and abs(ce1 - E157_WASH_S1["corpus_ce"]) < G_FALLBACK_TOL
                     and abs(disp1 - E193_STEP_L2) < G_FALLBACK_TOL
                     and rel_l2 < 0.05),
    }
    log(f"G_T0 (t=0 gate): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; CE |d| "
        f"{G_T0['d_ce']:.2e}; measured L2 {disp1:.10f} vs committed "
        f"{E193_STEP_L2:.10f} (|d| {G_T0['d_step_l2']:.2e}): "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 gate FAILED — abort (control failure)")
    STEP_L2_MEASURED = disp1             # measured, never ported

    G_S1CK = {
        "cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
        "tol_cos": 0.999, "s1_meta": s1_meta,
        "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
        "note": "e193/e197's G_S1CK VERBATIM: the fresh CPU AdamW step vs "
                "the committed cuda-trained e157_f2_neutral_s1 checkpoint's "
                "displacement — fp64 cosine > 0.999 + relative L2 < 5%",
    }
    log(f"G_S1CK: cos64 {cos_s1:.8f}, rel L2 dev {rel_l2:.2e}: "
        + ("PASS" if G_S1CK["pass"] else "FAIL"))
    if not G_S1CK["pass"]:
        raise RuntimeError("s1 checkpoint anchor FAILED — abort")

    # the t=0 post-clip gradient: organism 2's g-ray + static sign ray
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
    u0 = (s_raw / torch.norm(s_raw)).clone()    # e193's construction verbatim
    u0_md5 = hashlib.md5(u0.numpy().tobytes()).hexdigest()
    del s_raw

    # G_STREAM: the stream machinery verified WITHOUT training
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
                "note": "e193/e197's G_STREAM VERBATIM: the seed-10902 "
                        "stream construction (net-independent) md5-matches "
                        "e185's stored hashes"}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..4): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # G_DIRCK: e193's committed g-ray direction checkpoint (the root identity)
    uck = torch.load(CKPT_DIR / DIR_CK, map_location="cpu", weights_only=False)
    ug_maxdiff = float((u_g - uck["u"].float()).abs().max())
    G_DIRCK = {
        "path": str(CKPT_DIR / DIR_CK),
        "meta_gate": {"experiment": uck["meta"].get("experiment"),
                      "root": uck["meta"].get("root"),
                      "match": bool(uck["meta"].get("experiment") == "e193"
                                    and uck["meta"].get("root") == ROOT_CK)},
        "loaded_u_md5": hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest(),
        "fresh_u_md5": u_g_md5,
        "md5_match": bool(hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest() == u_g_md5),
        "max_abs_diff_fresh_vs_ckpt": ug_maxdiff,
        "cos64_fresh_vs_ckpt": cos64(u_g, uck["u"].float()),
        "theta0_md5_match_root": bool(uck.get("theta0_md5") == root_md5),
        "u_norm_fp32": float(torch.norm(u_g)),
        "note": "e191's direction-checkpoint convention, e193's file, with "
                "the disclosed TEXTURE tier (e199's G_DIRCK precedent): the "
                "committed u must BE the fresh t=0 g-ray — bit-exact by "
                "md5, or (the cross-thread/environment fallback; the raw "
                "normalized gradient is md5-brittle under ~1e-7 fp drift, "
                "unlike the discrete sign rays) coordinate-wise within "
                "1e-6; the committed theta0_md5 must be THIS root's flat "
                "md5 (the root state's identity gate)",
    }
    G_DIRCK["tier"] = ("BIT" if G_DIRCK["md5_match"] else "TEXTURE")
    G_DIRCK["pass"] = bool(G_DIRCK["meta_gate"]["match"]
                           and (G_DIRCK["md5_match"] or ug_maxdiff < 1e-6)
                           and G_DIRCK["theta0_md5_match_root"])
    log(f"G_DIRCK (e193's committed g-ray): md5 "
        + ("match" if G_DIRCK["md5_match"] else "DRIFT")
        + f" (tier {G_DIRCK['tier']}, max|d| {ug_maxdiff:.2e}), theta0_md5 "
        f"{'match' if G_DIRCK['theta0_md5_match_root'] else 'DRIFT'}: "
        + ("PASS" if G_DIRCK["pass"] else "FAIL"))

    # G_SIGNRAY: u0 rebuilt == e193's committed R2_SIGN direction
    G_SIGNRAY = {
        "u0_md5": u0_md5, "committed_e193_md5": E193_USIGN_MD5,
        "md5_match": bool(u0_md5 == E193_USIGN_MD5),
        "u_norm_fp32": float(torch.norm(u0)),
        "u_norm_fp64": float(torch.norm(u0.double())),
        "n_zero_g_coords": n_zero_g,
        "note": "family u0 = the static sign(g_0) ray rebuilt from the gated "
                "t=0 gradient (e193's fp32-norm construction VERBATIM); "
                "md5-gated vs e193's committed R2_SIGN ray",
    }
    G_SIGNRAY["pass"] = bool(G_SIGNRAY["md5_match"])
    log(f"G_SIGNRAY (u0 = static sign ray): md5 "
        + ("match" if G_SIGNRAY["md5_match"] else "DRIFT")
        + f" (fp32 norm {G_SIGNRAY['u_norm_fp32']:.7f}): "
        + ("PASS" if G_SIGNRAY["pass"] else "FAIL"))
    if not (G_DIRCK["pass"] and G_SIGNRAY["pass"]):
        raise RuntimeError("committed-ray gates FAILED — abort")

    log("WHAT THESE ARMS GUARANTEE: NOTHING — the half-step lineage is a "
        "counterfactual wash (the size is the experimenter's); the onset "
        "shape could deepen, stall, or grade; that openness is the point.")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e200_deepening",
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

    stub["gates"] = {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_ROOT": G_ROOT, "G_T0": G_T0, "G_S1CK": G_S1CK,
                     "G_STREAM": G_STREAM, "G_DIRCK": G_DIRCK,
                     "G_SIGNRAY": G_SIGNRAY}
    write_partial("standard cell gates passed (e193/e197's, verbatim)")
    log("standard cell gates passed; partial metrics written")

    # =====================================================================
    # PHASE 0 — the FULL-STEP walk rebuilt (the dead-at-t=1 baseline, G_REPRO)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the FULL-STEP k=1 sign walk rebuilt (e193's a_sign; the "
        "dead-at-t=1 baseline re-anchored in THIS process)")
    wfull = walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids,
                     coruler_ids, zid, theta0, E193_STEP_L2,
                     root_gm=root_cells[f"g{RULER_J:+d}"],
                     step_cap=1, d_target=5.0, extend_past_kill=0,
                     tag="full")
    assert len(wfull["traj"]) == 1, \
        f"expected the 1-step full walk (kill s1), got {len(wfull['traj'])}"
    fs1 = wfull["traj"][0]

    repro_rows = {
        "s1_ce_batch": (fs1["ce_batch"], E193_ASIGN_S1["ce_batch"]),
        "s1_cum_disp": (fs1["cum_disp"], E193_ASIGN_S1["cum_disp"]),
        "s1_step_disp": (fs1["step_disp"], E193_ASIGN_S1["step_disp"]),
        "s1_preclip_gnorm": (fs1["preclip_gnorm"],
                             E193_ASIGN_S1["preclip_gnorm"]),
        "s1_gm": (fs1["gm"], E193_ASIGN_S1["gm"]),
        "s1_frac_argmax_z": (fs1["frac_argmax_z"],
                             E193_ASIGN_S1["frac_argmax_z"]),
        "D_kill": (wfull["stop"]["D_kill"], E193_ASIGN_STOP["D_kill"]),
        "D_kill_raw": (wfull["stop"]["D_kill_raw"],
                       E193_ASIGN_STOP["D_kill_raw"]),
    }
    dens_diffs = [abs(a["gm"] - b[1]) for a, b in
                  zip(wfull["stop"]["dens"], E193_ASIGN_STOP["dens"])]
    dens_D_diffs = [abs(a["D"] - b[2]) for a, b in
                    zip(wfull["stop"]["dens"], E193_ASIGN_STOP["dens"])]
    repro_max0 = max(abs(vv[0] - vv[1]) for vv in repro_rows.values())
    G_REPRO = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows.items()},
        "dens_gm_max_abs_diff": max(dens_diffs),
        "dens_D_max_abs_diff": max(dens_D_diffs),
        "x_hash_vs_e185": bool(wfull["x_hashes"][1] == E185_XHASH[1]),
        "tol_bit": G_REPRO_TOL, "tol_texture": WALK_TEXTURE_TOL,
        "max_row_abs_diff": repro_max0,
        "max_l2_dev": wfull["max_l2_dev"],
        "tier": ("BIT" if repro_max0 < G_REPRO_TOL and max(dens_diffs)
                 < G_REPRO_TOL else
                 ("TEXTURE" if repro_max0 < WALK_TEXTURE_TOL
                  and max(dens_diffs) < WALK_TEXTURE_TOL else "FAIL")),
        "pass": bool(wfull["stop"]["kind"] == "kill"
                     and wfull["stop"]["step"] == 1
                     and repro_max0 < WALK_TEXTURE_TOL
                     and max(dens_diffs) < WALK_TEXTURE_TOL
                     and max(dens_D_diffs) < WALK_TEXTURE_TOL
                     and wfull["x_hashes"][1] == E185_XHASH[1]),
        "note": "THE BASELINE PROVENANCE GATE (Rule 12): the full-step k=1 "
                "walk must reproduce e193's committed a_sign step-1 row AND "
                "its densified kill bracket — bit-class, or (the disclosed "
                "cross-thread/environment tier, e199's precedent) within "
                "1e-3; the DEAD-at-t=1 baseline re-anchored in this process",
    }
    log(f"G_REPRO (vs e193 committed a_sign s1 + bracket): max|diff| "
        f"{repro_max0:.2e} (tier {G_REPRO['tier']}): "
        + ("PASS" if G_REPRO["pass"] else "FAIL"))
    if not G_REPRO["pass"]:
        stub["gates"]["G_REPRO"] = G_REPRO
        write_partial("CONTROL FAILURE — full-step baseline gate failed")
        raise RuntimeError("G_REPRO FAILED — abort before any read is believed")
    stub["gates"]["G_REPRO"] = G_REPRO
    stub["phases_partial"]["0_fullstep_baseline"] = E43.jsonable({
        "walk_journal": wfull["traj"], "stop": wfull["stop"],
        "x_hashes": wfull["x_hashes"], "max_l2_dev": wfull["max_l2_dev"],
        "disclosure": "the natural-size walk (STEP_L2 0.9164): DEAD at t=1 "
                      "(g-4 0.0068) — e193's committed result, re-anchored "
                      "here; no onset curve exists on this lineage (it died "
                      "BEFORE its onset — T163)",
    })
    write_partial("phase 0 complete (full-step dead baseline rebuilt)")

    # =====================================================================
    # PHASE 1 — THE HALF-STEP WALK REBUILT (the ray factory; G_WALKE197)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE 1 — THE HALF-STEP WALK REBUILT: e197's intervention VERBATIM "
        f"(k=1 sign machinery, size s* {E197_SUB_S:.10f} = STEP_L2/"
        f"{round(1 / E197_SUB_FRAC)}, e197's committed pick) — the ray "
        f"factory; journal gated row-by-row vs e197's committed walk")
    wsub = walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids,
                    coruler_ids, zid, theta0, E197_SUB_S,
                    root_gm=root_cells[f"g{RULER_J:+d}"],
                    step_cap=WALK_CAP, d_target=WALK_D_TARGET,
                    extend_past_kill=EXT_STEPS, tag="sub")

    # G_WALKE197: every journal row + stop + x_hashes vs e197's committed
    row_keys = ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm", "gm",
                "frac_argmax_z", "g+12", "g-12", "g+0")
    wl_rows = {}
    for mr, cr in zip(wsub["traj"], e197_journal):
        per = {}
        for k in row_keys:
            if k in cr:
                per[k] = {"measured": mr[k], "committed": cr[k],
                          "abs_diff": abs(mr[k] - cr[k])}
        wl_rows[f"s{mr['step']}"] = per
    wl_max = max(v["abs_diff"] for per in wl_rows.values()
                 for v in per.values())
    wl_xh_ok = {s: bool(wsub["x_hashes"][int(s)] == h)
                for s, h in e197_xh.items()}
    stop_ok = bool(wsub["stop"]["kind"] == E197_WALK_STOP["kind"]
                   and wsub["stop"]["step"] == E197_WALK_STOP["step"])
    stop_dkill_diff = abs(wsub["stop"]["D_kill"] - E197_WALK_STOP["D_kill"])
    dens_e197 = e197_walk["stop"]["dens"]
    dens_diffs_197 = [abs(a["gm"] - b["gm"]) for a, b in
                      zip(wsub["stop"]["dens"], dens_e197)]
    dens_D_diffs_197 = [abs(a["D"] - b["D"]) for a, b in
                        zip(wsub["stop"]["dens"], dens_e197)]
    G_WALKE197 = {
        "rows": wl_rows, "max_row_abs_diff": wl_max,
        "x_hashes_vs_e197": wl_xh_ok,
        "x_hashes_vs_e185": {s: bool(wsub["x_hashes"][int(s)]
                                     == E185_XHASH[int(s)])
                             for s in wl_xh_ok},
        "stop_kind_step_match": stop_ok,
        "stop_D_kill_measured": wsub["stop"]["D_kill"],
        "stop_D_kill_committed": E197_WALK_STOP["D_kill"],
        "stop_D_kill_abs_diff": stop_dkill_diff,
        "dens_gm_max_abs_diff": max(dens_diffs_197),
        "dens_D_max_abs_diff": max(dens_D_diffs_197),
        "tol_bit": G_REPRO_TOL, "tol_texture": WALK_TEXTURE_TOL,
        "tol_dkill_bit": DKILL_TOL, "tol_dkill_texture": DKILL_TEXTURE_TOL,
        "tier": ("BIT" if wl_max < G_REPRO_TOL and stop_dkill_diff
                 < DKILL_TOL else
                 ("TEXTURE" if wl_max < WALK_TEXTURE_TOL and stop_dkill_diff
                  < DKILL_TEXTURE_TOL else "FAIL")),
        "max_l2_dev": wsub["max_l2_dev"],
        "pass": bool(len(wsub["traj"]) == 6 and stop_ok
                     and wl_max < WALK_TEXTURE_TOL
                     and stop_dkill_diff < DKILL_TEXTURE_TOL
                     and max(dens_diffs_197) < WALK_TEXTURE_TOL
                     and max(dens_D_diffs_197) < WALK_TEXTURE_TOL
                     and all(wl_xh_ok.values())),
        "note": "THE PARENT-WALK PROVENANCE GATE (Rule 12): the rebuilt "
                "half-step walk must reproduce e197's committed journal "
                "(all 6 rows incl. co-rulers and the post-kill step-6 row), "
                "its x-hashes, its kill at t=5 and its densified bracket — "
                "bit-class, or the disclosed TEXTURE tier; the ray factory "
                "is not believed until this passes",
    }
    log(f"G_WALKE197 (vs e197 committed journal + stop): rows max|diff| "
        f"{wl_max:.2e}, D_kill |d| {stop_dkill_diff:.2e}, dens "
        f"{max(dens_diffs_197):.2e}, x_hashes "
        f"{sum(wl_xh_ok.values())}/{len(wl_xh_ok)} (tier "
        f"{G_WALKE197['tier']}): "
        + ("PASS" if G_WALKE197["pass"] else "FAIL"))
    if not G_WALKE197["pass"]:
        stub["gates"]["G_WALKE197"] = G_WALKE197
        write_partial("CONTROL FAILURE — the parent-walk gate failed")
        raise RuntimeError("G_WALKE197 FAILED — abort before any ray is "
                           "believed")
    stub["gates"]["G_WALKE197"] = G_WALKE197
    stub["phases_partial"]["1_walk_rebuild"] = E43.jsonable({
        "journal": wsub["traj"], "stop": wsub["stop"],
        "x_hashes": wsub["x_hashes"], "max_l2_dev": wsub["max_l2_dev"],
        "sub_s": E197_SUB_S, "sub_fraction": E197_SUB_FRAC,
        "disclosure": "e197's half-step walk rebuilt VERBATIM (the ray "
                      "factory); the post-kill step-6 gradient is computed "
                      "for the journal gate only and NEVER read as a ray",
    })
    write_partial("phase 1 complete (the parent walk rebuilt + gated)")

    # ---- G_ALIVE: the alive window re-anchored
    alive_flags = {r["step"]: bool(r["gm"] > SHUT_BAR)
                   for r in wsub["traj"] if not r["post_kill"]}
    G_ALIVE = {
        "alive_steps": sorted(t for t, a in alive_flags.items() if a),
        "dead_step": wsub["stop"]["step"],
        "per_step_gm": {r["step"]: r["gm"] for r in wsub["traj"]},
        "bar": SHUT_BAR,
        "expected_alive_e197": [1, 2, 3, 4],
        "expected_death_e197": 5,
        "gm_vs_e197_committed": {t: abs(
            next(r for r in wsub["traj"] if r["step"] == t)["gm"] - g)
            for t, g in E197_ALIVE_GMS.items()},
        "probe_vs_walk_s1": abs(wsub["traj"][0]["gm"] - E197_PROBE_GM),
        "pass": bool(alive_flags.get(1) and alive_flags.get(2)
                     and alive_flags.get(3) and alive_flags.get(4)
                     and wsub["stop"]["step"] == 5),
        "note": "THE ALIVE WINDOW RE-ANCHORED: t=1..t=4 ALIVE, t=5 DEAD — "
                "matching e197's committed flags (the intervention holds in "
                "this process); rays are read at t=1..4 only",
    }
    log(f"G_ALIVE: alive steps {G_ALIVE['alive_steps']}, death at t"
        f"{G_ALIVE['dead_step']} (e197: t1..t4 alive, t5 dead): "
        + ("PASS" if G_ALIVE["pass"] else "FAIL"))
    if not G_ALIVE["pass"]:
        stub["gates"]["G_ALIVE"] = G_ALIVE
        write_partial("CONTROL FAILURE — the alive window did not reproduce")
        raise RuntimeError("G_ALIVE FAILED — abort (the alive window is this "
                           "cell's substrate)")
    stub["gates"]["G_ALIVE"] = G_ALIVE

    # ---- the rays: u1..u4 from the rebuilt walk's refresh gradients (alive ts)
    g_fronts = wsub["g_fronts"]
    assert cos64(torch.sign(g_fronts[0]), u0) > 1 - 1e-9, "t=0 front != u0"

    def make_ray(g):
        s = torch.sign(g)
        return (s / torch.norm(s)).clone()

    u1, u2 = make_ray(g_fronts[1]), make_ray(g_fronts[2])
    u3, u4 = make_ray(g_fronts[3]), make_ray(g_fronts[4])
    u1_md5 = hashlib.md5(u1.numpy().tobytes()).hexdigest()
    u2_md5 = hashlib.md5(u2.numpy().tobytes()).hexdigest()
    u3_md5 = hashlib.md5(u3.numpy().tobytes()).hexdigest()
    u4_md5 = hashlib.md5(u4.numpy().tobytes()).hexdigest()
    ray_geometry = {
        "cos_u0_u1": cos64(u0, u1), "cos_u0_u2": cos64(u0, u2),
        "cos_u0_u3": cos64(u0, u3), "cos_u0_u4": cos64(u0, u4),
        "cos_u1_u2": cos64(u1, u2), "cos_u1_u3": cos64(u1, u3),
        "cos_u1_u4": cos64(u1, u4), "cos_u2_u3": cos64(u2, u3),
        "cos_u2_u4": cos64(u2, u4), "cos_u3_u4": cos64(u3, u4),
        "cos_u0_ug": cos64(u0, u_g), "cos_u1_ug": cos64(u1, u_g),
        "cos_u2_ug": cos64(u2, u_g), "cos_u3_ug": cos64(u3, u_g),
        "cos_u4_ug": cos64(u4, u_g),
        "iso_floor": 1.0 / (F2_PARAMS ** 0.5),
        "committed_e197": E197_GEOM,
        "note": "the rays' mutual geometry (fp64 cosines); e197's committed "
                "trio rides for the identity gate; iso floor for scale",
    }
    ray_geometry["geometry_dev_vs_e197"] = {
        k: abs(ray_geometry[k] - E197_GEOM[k]) for k in E197_GEOM}
    u12_md5_match = bool(u1_md5 == E197_U1_MD5 and u2_md5 == E197_U2_MD5)
    geom_dev_ok = all(v < RAY_COS_TOL
                      for v in ray_geometry["geometry_dev_vs_e197"].values())
    G_RAYS = {
        "u1_md5": u1_md5, "u1_md5_match_e197": bool(u1_md5 == E197_U1_MD5),
        "u2_md5": u2_md5, "u2_md5_match_e197": bool(u2_md5 == E197_U2_MD5),
        "u3_md5": u3_md5, "u3_md5_new_registered_here": True,
        "u4_md5": u4_md5, "u4_md5_new_registered_here": True,
        "geometry_max_abs_dev_vs_e197":
            max(ray_geometry["geometry_dev_vs_e197"].values()),
        "tol_geom": RAY_COS_TOL,
        "tier": ("BIT" if u12_md5_match else
                 ("TEXTURE" if geom_dev_ok else "FAIL")),
        "pass": bool(u12_md5_match or geom_dev_ok),
        "note": "THE RAY IDENTITY GATE with the disclosed TEXTURE tier: u1/"
                "u2 must BE e197's registered rays (md5 BIT), or (the "
                "cross-thread/environment fallback, e199's precedent) the "
                "fresh mutual geometry must reproduce e197's committed fp64 "
                "cosines within 1e-3; u3 (t=3) and u4 (t=4) are NEW — their "
                "only parents are the gated walk (G_WALKE197), their md5s "
                "registered here; u_t is read at theta_t, the ALIVE states "
                "(u4 = the killing step's own direction, read at the last "
                "alive state theta_4); the post-kill front is NEVER read",
    }
    log(f"G_RAYS: u1 md5 "
        + ("match" if G_RAYS["u1_md5_match_e197"] else "DRIFT")
        + f", u2 md5 "
        + ("match" if G_RAYS["u2_md5_match_e197"] else "DRIFT")
        + f", geometry max|dev| "
        f"{G_RAYS['geometry_max_abs_dev_vs_e197']:.2e} (tier "
        f"{G_RAYS['tier']}): "
        + ("PASS" if G_RAYS["pass"] else "FAIL"))
    if not G_RAYS["pass"]:
        stub["gates"]["G_RAYS"] = G_RAYS
        write_partial("CONTROL FAILURE — the ray identity gate failed")
        raise RuntimeError("G_RAYS FAILED — abort")
    stub["gates"]["G_RAYS"] = G_RAYS
    rays_meta = [
        {"key": "u0", "t": 0,
         "label": "sign(g_0) — the static sign ray (the OWN edge)",
         "u_md5": u0_md5,
         "provenance": "e193's committed R2_SIGN construction verbatim "
                       "(md5-gated, G_SIGNRAY)"},
        {"key": "u1", "t": 1,
         "label": "sign(g_1) — the t=1 front (e197's ALIVE flight ray)",
         "u_md5": u1_md5,
         "provenance": "the half-step walk's step-2 refresh gradient (at "
                       "theta_1 ALIVE "
                       f"{wsub['traj'][0]['gm']:.6f}); md5-gated vs e197's "
                       "registered ray (G_RAYS)"},
        {"key": "u2", "t": 2,
         "label": "sign(g_2) — the t=2 front (e197 mapped, unadjudicated)",
         "u_md5": u2_md5,
         "provenance": "the walk's step-3 refresh gradient (at theta_2 ALIVE "
                       f"{wsub['traj'][1]['gm']:.6f}); md5-gated vs e197's "
                       "registered ray (G_RAYS)"},
        {"key": "u3", "t": 3,
         "label": "sign(g_3) — the t=3 front (NEW)",
         "u_md5": u3_md5,
         "provenance": "the walk's step-4 refresh gradient (at theta_3 ALIVE "
                       f"{wsub['traj'][2]['gm']:.6f}) — registered here; "
                       "parent: the gated walk (G_WALKE197)"},
        {"key": "u4", "t": 4,
         "label": "sign(g_4) — the t=4 front (NEW; the killing step's own "
                  "direction)",
         "u_md5": u4_md5,
         "provenance": "the walk's step-5 refresh gradient (at theta_4 ALIVE "
                       f"{wsub['traj'][3]['gm']:.6f}; this step KILLED the "
                       "organism at its endpoint) — registered here; parent: "
                       "the gated walk (G_WALKE197)"},
    ]
    stub["phases_partial"]["1b_rays"] = E43.jsonable({
        "rays": rays_meta, "ray_geometry": ray_geometry})
    write_partial("phase 1b complete (the rays u1..u4 + geometry)")

    # =====================================================================
    # PHASE A — the root ray families (u0..u4; u3/u4 new)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE A — five root families x {len(D_GRID)}+1 grid points "
        "(e197's static_profile VERBATIM)")
    ce_ds_common = ([0.0] + [round(CE_EVERY * i, 2)
                             for i in range(1, int(3.0 / CE_EVERY) + 1)]
                    if CE_EVERY else [0.0])
    evp = copy.deepcopy(net0)
    fams = {}
    for key, u in (("u0", u0), ("u1", u1), ("u2", u2), ("u3", u3),
                   ("u4", u4)):
        tA = time.time()
        dgrid = [0.0] + list(D_GRID)
        rows = static_profile(evp, theta0, u, dgrid, primary_ids,
                              coruler_ids, zid, r_eval_xy, ce_ds_common)
        edge, ups = profile_d_kill(rows)
        fams[key] = {"rows": rows, "D_kill": edge, "any_upcross": ups,
                     "seconds": round(time.time() - tA, 1)}
        # row-wise gate vs e197's committed families where they exist
        gate = None
        if key in ("u0", "u1", "u2"):
            gate = gate_profile(rows, e197m["profiles"][f"root_{key}"]["rows"],
                                "gm", f"root_{key} vs e197 committed")
        fams[key]["gate"] = gate
        stub["phases_partial"][f"prof_root_{key}"] = E43.jsonable(
            {"D_kill": edge, "any_upcross": ups,
             "gate_pass": (gate["pass"] if gate else None),
             "max_abs_diff": (gate["max_abs_diff"] if gate else None)})
        write_partial(f"profile root_{key} complete")
        log(f"  root_{key}: {len(rows)} pts in {fams[key]['seconds']}s — "
            f"D_kill "
            + (f"{edge:.10f}" if edge is not None else "None (soft <= 3.0)")
            + (f"; vs e197 committed {gate['n_shared']} shared pts max|diff| "
               f"{gate['max_abs_diff']:.2e}" if gate else "; NEW (no "
               "committed comparator)"))

    # ---- G_PROF: the profile machinery gate
    G_PROF = {
        "u0_vs_e197": fams["u0"]["gate"], "u1_vs_e197": fams["u1"]["gate"],
        "u2_vs_e197": fams["u2"]["gate"],
        "u0_D_kill": fams["u0"]["D_kill"],
        "u0_D_kill_committed": E197_ROOT_DKILLS["root_u0"],
        "u1_D_kill": fams["u1"]["D_kill"],
        "u1_D_kill_committed": E197_ROOT_DKILLS["root_u1"],
        "u2_D_kill": fams["u2"]["D_kill"],
        "u2_D_kill_committed": E197_ROOT_DKILLS["root_u2"],
        "u3_D_kill": fams["u3"]["D_kill"], "u3_new": True,
        "u4_D_kill": fams["u4"]["D_kill"], "u4_new": True,
        "tol_rows": G_STATIC_TOL,
        "tol_dkill_bit": DKILL_TOL,
        "tol_dkill_texture": DKILL_TEXTURE_TOL,
        "dkill_check_skipped": bool(SMOKE),
        "note": "THE PROFILE MACHINERY GATE: the recomputed root-u0/u1/u2 "
                "families must reproduce e197's committed rows at ALL shared "
                "(base-grid) Ds and their D_kills (bit or the disclosed "
                "TEXTURE tier; root-u0's chain to e193's R2_SIGN rows rides "
                "through e197's own committed gate transitively) before any "
                "new point (u3/u4) is believed; D_kill equality skipped in "
                "smoke (the trimmed grid changes the brackets)",
    }
    dkill_dev = max(
        [abs(fams[k]["D_kill"] - E197_ROOT_DKILLS[f"root_{k}"])
         for k in ("u0", "u1", "u2")
         if fams[k]["D_kill"] is not None
         and E197_ROOT_DKILLS[f"root_{k}"] is not None] or [0.0])
    G_PROF["dkill_max_abs_dev"] = dkill_dev
    G_PROF["dkill_tier"] = ("BIT" if dkill_dev < DKILL_TOL else
                            ("TEXTURE" if dkill_dev < DKILL_TEXTURE_TOL
                             else "FAIL"))
    G_PROF["pass"] = bool(
        fams["u0"]["gate"]["pass"] and fams["u1"]["gate"]["pass"]
        and fams["u2"]["gate"]["pass"]
        and (SMOKE or (
            _dk_ok(fams["u0"]["D_kill"], E197_ROOT_DKILLS["root_u0"],
                   DKILL_TEXTURE_TOL)
            and _dk_ok(fams["u1"]["D_kill"], E197_ROOT_DKILLS["root_u1"],
                       DKILL_TEXTURE_TOL)
            and _dk_ok(fams["u2"]["D_kill"], E197_ROOT_DKILLS["root_u2"],
                       DKILL_TEXTURE_TOL))))
    log(f"G_PROF (rows + D_kills vs e197 committed): "
        + ("PASS" if G_PROF["pass"] else "FAIL")
        + f" (D_kill tier {G_PROF['dkill_tier']}, max dev {dkill_dev:.2e})")
    if not G_PROF["pass"]:
        stub["gates"]["G_PROF"] = G_PROF
        write_partial("CONTROL FAILURE — profile machinery gates failed")
        raise RuntimeError("G_PROF FAILED — abort before any onset number "
                           "is believed")
    stub["gates"]["G_PROF"] = G_PROF
    write_partial("all profile machinery gates passed")

    # =====================================================================
    # THE ONSET CURVE + ADJUDICATION (frozen clauses; composite
    # ONSET-DEEPENS -> ONSET-STALLS -> GRADED; no shopping)
    # =====================================================================
    log("=" * 78)
    u0_edge = fams["u0"]["D_kill"]
    onset = [{"t": 0, "ray": "u0", "u_md5": u0_md5, "D_kill": u0_edge,
              "ratio": 1.0, "ratio_note": "definitional (u0 vs its own edge)",
              "state": "alive", "ruler_read": root_cells[f"g{RULER_J:+d}"]}]
    for t in (1, 2, 3, 4):
        key = f"u{t}"
        dk = fams[key]["D_kill"]
        row = next(r for r in wsub["traj"] if r["step"] == t)
        entry = {"t": t, "ray": key,
                 "u_md5": {"u1": u1_md5, "u2": u2_md5, "u3": u3_md5,
                           "u4": u4_md5}[key],
                 "D_kill": dk,
                 "ratio": (dk / u0_edge if dk is not None else None),
                 "state": "alive", "ruler_read": row["gm"]}
        if dk is None:
            entry["soft"] = True
            entry["floor_ratio"] = (max(D_GRID) / u0_edge
                                    if u0_edge else None)
            minrow = min(fams[key]["rows"], key=lambda r: r["gm"])
            entry["min_read"] = {"D": minrow["D"], "gm": minrow["gm"]}
            entry["note"] = ("SOFT (unresolved-high): u_t never kills within "
                             f"[0, 3.0] (min g-4 {minrow['gm']:.4f} at D "
                             f"{minrow['D']:.2f}); floor reported, never "
                             "adjudicated")
        onset.append(entry)
    # the death row + the recomputation check at death (context, never a bar)
    walk_kill = wsub["stop"]["D_kill"]
    recompute_ratio = (walk_kill / u0_edge if u0_edge else None)
    onset.append({"t": 5, "ray": None, "D_kill": None, "ratio": None,
                  "state": "death", "ruler_read": wsub["stop"]["gm_at_kill"],
                  "walk_D_kill": walk_kill,
                  "recompute_check_at_death": {
                      "walk_D_kill": walk_kill, "static_u0_edge": u0_edge,
                      "ratio": recompute_ratio, "bar": BONUS_RATIO_BAR,
                      "e197_committed_ratio": E197_BONUS_RATIO,
                      "reading": ("context only — e197's committed "
                                  "adjudication stands (absent); never an "
                                  "e200 bar")},
                  "note": "DEATH — the post-death front (step 6) never read; "
                          "the walk's densified kill inside the step-5 "
                          "bracket carries e197's recomputation check"})
    log("the onset curve: "
        + "; ".join(f"t{r['t']} "
                    + ("DEATH (walk kill "
                       f"{r.get('walk_D_kill', float('nan')):.4f}, recompute "
                       f"ratio {r['recompute_check_at_death']['ratio']:.4f})"
                       if r["state"] == "death"
                       else (f"SOFT (floor >{r['floor_ratio']:.4f})"
                             if r.get("soft")
                             else f"ratio {r['ratio']:.4f}"))
                    for r in onset))

    # ---- the frozen clauses
    def first_concentrated(onset_rows):
        for r in onset_rows:
            if r["state"] == "alive" and r["t"] >= 1 \
                    and r.get("ratio") is not None and r["ratio"] < CONC_BAR:
                return r["t"]
        return None

    tc = first_concentrated(onset)
    resolved = [r for r in onset
                if r["state"] == "alive" and r["t"] >= 1
                and r.get("ratio") is not None]
    any_concentrated = tc is not None
    # (ii) holds: every resolved ratio strictly after tc stays <= 0.70
    holds = True if tc is not None else False
    unformed = []
    if tc is not None:
        for r in resolved:
            if r["t"] > tc and r["ratio"] > CONC_BAR:
                holds = False
                unformed.append({"t": r["t"], "ratio": r["ratio"]})
    # (iii) the fall: resolved ratios t=1..tc non-increasing (SOFT skipped)
    falls = True if tc is not None else False
    rises = []
    if tc is not None:
        pre = [r for r in resolved if 1 <= r["t"] <= tc]
        for a, b in zip(pre, pre[1:]):
            if b["ratio"] > a["ratio"]:
                falls = False
                rises.append({"from_t": a["t"], "to_t": b["t"],
                              "from_ratio": a["ratio"],
                              "to_ratio": b["ratio"]})
    fires_deepens = bool(any_concentrated and holds and falls)
    fires_stalls = bool(not any_concentrated)

    # ---- the COMMITTED-NUMBERS CROSSCHECK (robustness, never a bar choice)
    t1_c = E197_ROOT_DKILLS["root_u1"] / E197_ROOT_DKILLS["root_u0"]
    t2_c = E197_ROOT_DKILLS["root_u2"] / E197_ROOT_DKILLS["root_u0"]
    conc_arrives_committed = bool(t2_c < CONC_BAR and t1_c >= CONC_BAR)
    t1_fresh = next(r for r in onset if r["t"] == 1)
    t2_fresh = next(r for r in onset if r["t"] == 2)
    t1_fresh_ratio = (t1_fresh["ratio"] if t1_fresh["ratio"] is not None
                      else float("inf"))     # SOFT = not concentrated
    conc_arrives_fresh = bool(
        tc == 2 and t2_fresh["ratio"] is not None
        and t2_fresh["ratio"] < CONC_BAR and t1_fresh_ratio >= CONC_BAR)
    committed_crosscheck = {
        "note": "the same frozen clauses on e197's committed D_kills (the "
                "t=1/t=2 prefix) vs this cell's fresh recomputation — a "
                "robustness check, never a bar choice; t=3/t=4 exist only "
                "fresh (e197 never mapped them); both sets reported",
        "t1_ratio_committed": t1_c, "t2_ratio_committed": t2_c,
        "concentration_arrives_at_t2_committed": conc_arrives_committed,
        "concentration_arrives_at_t2_fresh": conc_arrives_fresh,
        "t1_ratio_fresh": next(r for r in onset if r["t"] == 1)["ratio"],
        "t2_ratio_fresh": next(r for r in onset if r["t"] == 2)["ratio"],
    }

    fmt = lambda v: ("None" if v is None else f"{v:.4f}")
    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "ONSET_DEEPENS": {
                "fires": fires_deepens,
                "detail": {"concentration_arrives_t": tc,
                           "holds_after_tc": holds,
                           "unformed_after_tc": unformed,
                           "falls_approach": falls, "approach_rises": rises,
                           "resolved_ratios": [{"t": r["t"],
                                                "ratio": r["ratio"]}
                                               for r in resolved],
                           "concentration_bar": CONC_BAR},
            },
            "ONSET_STALLS": {
                "fires": fires_stalls,
                "detail": {"any_concentrated_t": any_concentrated,
                           "ratios": [{"t": r["t"],
                                       "ratio": r.get("ratio"),
                                       "soft": bool(r.get("soft"))}
                                      for r in onset if r["state"] == "alive"
                                      and r["t"] >= 1]},
            },
            "GRADED": {"fires": not (fires_deepens or fires_stalls)},
        }
        curve_str = ", ".join(
            f"t{r['t']}: " + ("DEATH" if r["state"] == "death"
                              else (f"{r['ratio']:.4f}"
                                    if r.get("ratio") is not None
                                    else f"SOFT(floor >{r['floor_ratio']:.4f})"))
            for r in onset)
        if fires_deepens:
            verdict = "ONSET-DEEPENS"
            clause = (f"the ratio falls then holds across the alive window — "
                      f"the curve is {curve_str}; the concentration ARRIVES "
                      f"at t={tc} (ratio "
                      f"{next(r for r in onset if r['t']==tc)['ratio']:.4f} "
                      f"< {CONC_BAR}) and every resolved ratio after it "
                      f"stays <= {CONC_BAR} "
                      + (f"(un-formed at {unformed})" if unformed else
                         "(none un-formed)")
                      + f"; the approach from t=1 is non-increasing "
                        f"(resolved "
                      + ", ".join(f"t{r['t']} {r['ratio']:.4f}"
                                  for r in resolved if r["t"] <= tc)
                      + ") — the onset shape holds wherever the race allows "
                        "time: THIS lineage, given four alive steps (the "
                        "longest window on record), shows the same "
                        "soft-then-concentrated shape as org1 (t=1) and "
                        "MIRABEL (t=2) — a third arrival time, not a fourth "
                        "timing class; the counterfactual-wash caveat rides.")
        elif fires_stalls:
            verdict = "ONSET-STALLS"
            clause = (f"the ratio stays soft across all four alive steps — "
                      f"the curve is {curve_str}; no alive t concentrates "
                      f"(every ratio None or >= {CONC_BAR}) — a FOURTH "
                      "timing class: alive-but-unconcentrating; the alive "
                      "window (even four steps of it) does not by itself "
                      "bring the concentration; the counterfactual-wash "
                      "caveat rides.")
        else:
            verdict = "GRADED"
            # GRADED is reachable only with concentration arrived but not
            # holding or not falling (no-concentration fires STALLS first)
            why = "; ".join(
                [f"it UN-FORMS at t{u['t']} (ratio {u['ratio']:.4f} > "
                 f"{CONC_BAR})" for u in unformed]
                + [f"the approach RISES t{r['from_t']}->t{r['to_t']} "
                   f"({r['from_ratio']:.4f}->{r['to_ratio']:.4f})"
                   for r in rises])
            clause = (f"any mix — the curve verbatim: {curve_str}; the "
                      f"concentration ARRIVES at t={tc} but {why} — a mix "
                      "of onset and un-formation (or a non-falling "
                      "approach); the profiles verbatim; the "
                      "counterfactual-wash caveat rides.")
    log(f"E200 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e200_deepening",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "recovery_note": "e200 died pre-artifact in the ninth disruption "
                         "(overnight heat shutdown ~9h); this is the restart "
                         "under the tightened owner envelope (CPU-only, "
                         "threads 4, progressive writes)",
        "question": ("THE DEEPENING TEST (T163's named follow-on): does "
                     "e197's half-step lineage — the only lineage ALIVE "
                     "long enough (t=1..4) — show the concentration ratio "
                     "DEEPENING across its alive window (the onset shape "
                     "holding wherever the race allows time), staying soft "
                     "across all four alive steps (a fourth timing class: "
                     "alive-but-unconcentrating), or grading?"),
        "organism": {
            "root": f"runs/checkpoints/{ROOT_CK}",
            "root_meta": root_meta, "root_flat_md5": root_md5,
            "N_param": N_PARAM,
            "architecture": "4L/4H/128d/512-ctx TinyGPT (the e098 s4305 "
                            "line) vs organism-1's 6L/6H/192d/256-ctx — "
                            "architecture co-varies with lineage (e193's "
                            "disclosure, carried)",
            "step_l2_measured_this_process": STEP_L2_MEASURED,
            "step_l2_committed_e193": E193_STEP_L2,
            "sub_step": {"s": E197_SUB_S, "fraction": E197_SUB_FRAC,
                         "natural_step": E193_STEP_L2,
                         "note": "e197's committed pick (its frozen ladder, "
                                 "registered before ITS walk); the "
                                 "counterfactual-wash deviation carried "
                                 "verbatim: the direction machinery is the "
                                 "wash's own, the size the experimenter's"},
        },
        "rulers": {
            "primary": {"battery": f"install-60 g{RULER_J:+d}",
                        "root_read": root_cells[f"g{RULER_J:+d}"],
                        "why": "e193's frozen ruler call, carried verbatim "
                               "through e197"},
            "co_rulers": {f"g{j:+d}": root_cells[f"g{j:+d}"]
                          for j in CO_RULERS_J},
            "disclosure": "the e192-verbatim g-12 ruler reads 0.1983 at "
                          "this root — UNDER the 0.27 kill bar at D=0 "
                          "(e157 stage-A's G_CONS failure); co-rulers on "
                          "every row and every walk step, never adjudicated",
            "ce_r": root_cells["ce_r"],
        },
        "cell": {
            "licensed_cell": "e193/e197's stream/step convention VERBATIM "
                             "(seed 10902, draw order, post-clip gradients, "
                             "threads — capped at 4 by the owner envelope, "
                             "tiered gates); the full-step k=1 walk rebuilt "
                             "(the dead baseline) and e197's HALF-STEP walk "
                             "rebuilt (the ray factory), both gated",
            "grid": {"D": D_GRID,
                     "note": "e195/e197/e199's D grid VERBATIM (0.05..3.00 "
                             "step 0.05) + D=0 anchor rows in every family; "
                             "e197's 6 extra root-u0 landing Ds not repeated "
                             "(all outside the kill brackets — disclosed)"},
            "rays": rays_meta,
            "anchors": {
                "theta_0": {"label": "the e157_f2 consolidated root "
                                     "(organism 2)",
                            "gm": root_cells[f"g{RULER_J:+d}"],
                            "provenance": f"runs/checkpoints/{ROOT_CK}, "
                                          "gated vs e157's dial BIT-TIGHT + "
                                          "flat-md5 (G_ROOT/G_DIRCK)"},
                "note": "the rays are mapped from the ROOT only (e199's "
                        "onset form); no theta_1 panel in this cell "
                        "(e197's rides committed, never adjudicated)",
            },
            "input_seed": FREEZE_SEED,
        },
        "gates": stub["gates"],
        "phase0_fullstep_baseline": stub["phases_partial"]["0_fullstep_baseline"],
        "phase1_walk_rebuild": stub["phases_partial"]["1_walk_rebuild"],
        "phase1b_rays": stub["phases_partial"]["1b_rays"],
        "alive_ledger": [{"t": r["step"],
                          "ruler_read": r["gm"],
                          "alive": bool(r["gm"] > SHUT_BAR),
                          "post_kill": r["post_kill"]}
                         for r in wsub["traj"]],
        "ray_geometry": ray_geometry,
        "profiles": {
            f"root_{k}": {
                "anchor": "theta_0", "ray": k, "t": int(k[1]),
                "u_md5": v_u_md5,
                "label": next(r["label"] for r in rays_meta
                              if r["key"] == k),
                "placement": "root - D*u (e192's static placement verbatim)",
                "grid": [0.0] + list(D_GRID), "n_points": len(v["rows"]),
                "rows": v["rows"], "D_kill": v["D_kill"],
                "any_upcross": v["any_upcross"],
                "gate": ({kk: vv for kk, vv in v["gate"].items()
                          if kk != "rows"} if v["gate"] else None),
                "seconds": v["seconds"]}
            for k, v, v_u_md5 in (("u0", fams["u0"], u0_md5),
                                  ("u1", fams["u1"], u1_md5),
                                  ("u2", fams["u2"], u2_md5),
                                  ("u3", fams["u3"], u3_md5),
                                  ("u4", fams["u4"], u4_md5))},
        "onset_curve": {
            "rows": onset,
            "side_by_side": [
                {"t": t,
                 "ratio": next((r.get("ratio") for r in onset if r["t"] == t),
                               None),
                 "state": next((r["state"] for r in onset if r["t"] == t),
                               None)}
                for t in range(0, 6)],
            "concentration_bar": CONC_BAR,
            "first_concentrated_t": tc,
            "recompute_check_at_death": onset[-1]["recompute_check_at_death"],
        },
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "committed_numbers_crosscheck": committed_crosscheck,
            "composite_order": "ONSET-DEEPENS -> ONSET-STALLS -> GRADED "
                               "(frozen before compute)",
            "constants": {"SHUT_BAR": SHUT_BAR, "D_grid": D_GRID,
                          "CONC_BAR": CONC_BAR,
                          "BONUS_RATIO_BAR_ctx_only": BONUS_RATIO_BAR,
                          "SUB_S": E197_SUB_S},
        },
        "four_organism_context": {
            "note": "the committed four-lineage ledger (rides as context; "
                    "org1/MIRABEL/org2-dead loaded committed, never re-run "
                    "here; org2-half is THIS cell's fresh curve)",
            "org1_e195_e199": {
                "t1_ratio": E199_ORG1_T1,
                "t2_postdeath_ctx_ratio": (E195_ORG1["root_u2_D_kill"]
                                           / E195_ORG1["root_u0_D_kill"]),
                "death_t": 2},
            "mirabel_e198_e199": {
                "t1": None, "t1_soft_floor": E199_MIR_T1_FLOOR,
                "t2_ratio": E199_MIR_T2, "death_t": 3},
            "org2_deadlineage_e193_e196": {
                "u0_edge": E196_DEAD["root_u0_D_kill"],
                "death_t": 1,
                "postkill_u1_ratio_ctx_only":
                    E196_DEAD["root_u1_D_kill"] / E196_DEAD["root_u0_D_kill"],
                "note": "died BEFORE its onset (T163); e196's u1 is a "
                        "dead-state read — context only, flagged"},
            "org2_halflinelineage_e197_e200": {
                "t1_ratio_committed_e197": E197_FLIGHT_RATIO,
                "t2_ratio_committed_e197":
                    E197_ROOT_DKILLS["root_u2"] / E197_ROOT_DKILLS["root_u0"],
                "t3_t4": "THIS cell (fresh)", "death_t": 5,
                "recompute_ratio_at_death_committed": E197_BONUS_RATIO},
        },
        "references": {
            "e197_alive_window": {
                "metrics": "runs/e197/metrics.json",
                "role": "THE PARENT: the half-step walk (the journal, the "
                        "kill at t=5), the registered u1/u2 rays, the root "
                        "panel (u0/u1/u2) and the whole gate set — loaded "
                        "COMMITTED, gated, never rerun"},
            "e193_organism_replicate": {
                "metrics": "runs/e193/metrics.json",
                "role": "the organism's committed state: the a_sign step-1 "
                        "row + kill bracket (the dead-at-t=1 baseline), the "
                        "R2_SIGN profile (via e197's gate, transitively), "
                        "the ray checkpoints, the measured STEP_L2"},
            "e199_onset_curve": {
                "metrics": "runs/e199/metrics.json",
                "role": "the onset method + the other organisms' committed "
                        "curves + the tiered-identity-gate convention"},
            "e196_flight_replicate": {
                "metrics": "runs/e196/metrics.json",
                "role": "org2's DEAD full-step lineage (the death-at-t=1 "
                        "pole of the four-organism picture)"},
            "e157": {"metrics": "runs/e157/metrics.json",
                     "role": "this root's committed dial (G_ROOT) + the "
                             "wash step-1 anchors (G_T0/G_S1CK)"},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 root, n=1 stream (seed 10902, md5-gated "
                            "at every consumed step), ONE lineage read "
                            "through a counterfactual step size: the "
                            "deepening claim is causal FOR THIS BIOGRAPHY "
                            "(organism 2's f2 half-step lineage), not a "
                            "population claim; org1 and MIRABEL remain the "
                            "only natural trajectories on record"),
            "counterfactual_wash_caveat": ("the lineage is alive only "
                                           "because the experimenter halved "
                                           "the step (the natural 0.9164 "
                                           "step killed it at t=1): any "
                                           "onset shape found belongs to "
                                           "the alive window OF THIS "
                                           "CONSTRUCTION; whether the "
                                           "natural trajectory would have "
                                           "shown it is unanswerable in "
                                           "principle on a dead state "
                                           "(T158's lesson) — disclosed in "
                                           "the verdict clause itself"),
            "openness": ("WHAT EACH ARM GUARANTEES: NOTHING — every family "
                         "is a static graded jump at lethal scale from a "
                         "state that could kill anywhere; the ratio "
                         "compares each ray to its OWN lineage's static "
                         "edge (within-organism form only); cross-organism "
                         "ratio magnitudes are NOT compared (only each "
                         "curve's SHAPE and arrival time); that openness is "
                         "the point"),
            "estimator_lesson": ("no alignment read adjudicates (alignment "
                                 "predicts nothing — T157's lesson at n=4 "
                                 "lineages; e199's deviation carried); every "
                                 "number is a ruler read (the g-4 battery) "
                                 "on a stated state"),
            "projections_never_adjudicate": ("D_kill is a linear-in-D "
                                             "interpolation on measured "
                                             "grid points; the 0.70 bar is "
                                             "e198/e199's frozen "
                                             "concentration bar; the "
                                             "clauses were frozen before "
                                             "compute; the death-point "
                                             "recomputation check is "
                                             "context only (e197's "
                                             "adjudication stands)"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "owner envelope's cap; e197's chain was "
                              "committed at 8): every identity gate is "
                              "TIERED (BIT or the disclosed e199-class "
                              "TEXTURE tier), the achieved tier stamped "
                              "per gate; the fresh measurements adjudicate "
                              "and the committed-numbers crosscheck rides "
                              "in the adjudication"),
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                   "block_size": 512, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_deepening(rd / "e200_deepening.png", onset, verdict)
    plot_ray_profiles(rd / "e200_ray_profiles.png", fams)
    log(f"outputs: {rd / 'metrics.json'} + 2 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_deepening(path, onset, verdict):
    """THE FOUR-ORGANISM ONSET PICTURE: ratio vs t for all four lineages
    (org1 / MIRABEL / org2-dead loaded committed; org2's half-step lineage
    fresh, with e197's committed t=1/t=2 as open crosscheck circles) + this
    lineage's alive ledger."""
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8))
    ax = axes[0]
    # (committed) ORG1: t0 1.0, t1 concentrated, death t2 (+post-death ctx)
    ax.plot([0, 1], [1.0, E199_ORG1_T1], "o-", ms=8, lw=1.8, color="navy",
            label="ORGANISM 1 (e131; natural walk; e195/e199 committed)")
    ax.plot([2], [E195_ORG1["root_u2_D_kill"] / E195_ORG1["root_u0_D_kill"]],
            "o", ms=8, mfc="none", mec="navy", mew=1.8, ls="none",
            label=f"org1 t=2 POST-DEATH read (context) "
                  f"{E195_ORG1['root_u2_D_kill'] / E195_ORG1['root_u0_D_kill']:.3f}")
    ax.plot([2], [1.0], "x", ms=13, mew=2.6, color="navy")
    ax.annotate("org1 DIES t=2", (2, 1.0), textcoords="offset points",
                xytext=(-6, 14), fontsize=7.6, color="navy", ha="right")
    # (committed) MIRABEL: t0 1.0, t1 SOFT floor, t2 concentrated, death t3
    ax.plot([0, 2], [1.0, E199_MIR_T2], "o-", ms=8, lw=1.8, color="crimson",
            label="MIRABEL (e193b; natural walk; e198/e199 committed)")
    ax.plot([1], [E199_MIR_T1_FLOOR], "^", ms=9, mfc="none", mec="crimson",
            mew=1.8, ls="none")
    ax.annotate(f"t=1 SOFT: never kills <= 3.0\n(floor >"
                f"{E199_MIR_T1_FLOOR:.3f})", (1, E199_MIR_T1_FLOOR),
                textcoords="offset points", xytext=(10, 6), fontsize=7.4,
                color="crimson")
    ax.plot([3], [1.0], "x", ms=13, mew=2.6, color="crimson")
    ax.annotate("MIRABEL DIES t=3", (3, 1.0), textcoords="offset points",
                xytext=(-6, 14), fontsize=7.6, color="crimson", ha="right")
    # (committed) org2 DEAD full-step: the death-at-t=1 pole
    ax.plot([0], [1.0], "o", ms=8, lw=0, color="dimgray",
            label="ORGANISM 2 full-step (DEAD at t=1 — died BEFORE its "
                  "onset; e193/e196 committed)")
    ax.plot([1], [1.0], "x", ms=13, mew=2.6, color="dimgray")
    ax.annotate("org2 full-step DIES t=1\n(no curve; e196's post-kill u1 "
                "ratio 2.585 context only)", (1, 1.0),
                textcoords="offset points", xytext=(10, -22), fontsize=7.2,
                color="dimgray")
    # (fresh) org2 HALF-STEP: this cell's curve t0..t4 + death t5
    pts = [(r["t"], r["ratio"]) for r in onset
           if r["state"] == "alive" and r.get("ratio") is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=9, lw=2.2,
            color="seagreen",
            label="ORGANISM 2 HALF-STEP (e197's alive lineage; THIS cell, "
                  "fresh)")
    softs = [r for r in onset if r["state"] == "alive" and r.get("soft")]
    for r in softs:
        ax.plot([r["t"]], [r["floor_ratio"]], "^", ms=9, mfc="none",
                mec="seagreen", mew=1.8, ls="none")
        ax.annotate(f"t{r['t']} SOFT (floor >{r['floor_ratio']:.2f})",
                    (r["t"], r["floor_ratio"]), textcoords="offset points",
                    xytext=(8, 2), fontsize=7.2, color="seagreen")
    ax.plot([5], [1.0], "x", ms=13, mew=2.6, color="seagreen")
    ax.annotate("org2 half-step DIES t=5\n(walk kill D "
                f"{onset[-1]['walk_D_kill']:.3f}; recompute ratio "
                f"{onset[-1]['recompute_check_at_death']['ratio']:.3f} — "
                "context)", (5, 1.0), textcoords="offset points",
                xytext=(-8, -30), fontsize=7.2, color="seagreen",
                ha="right")
    # e197's committed t=1/t=2 as open crosscheck circles
    ax.plot([1], [E197_FLIGHT_RATIO], "o", ms=10, mfc="none", mec="seagreen",
            mew=1.4, ls="none", alpha=0.85)
    ax.plot([2], [E197_ROOT_DKILLS["root_u2"] / E197_ROOT_DKILLS["root_u0"]],
            "o", ms=10, mfc="none", mec="seagreen", mew=1.4, ls="none",
            alpha=0.85,
            label="e197 committed t=1/t=2 (open circles; t=2 was "
                  "UNADJUDICATED there)")
    ax.axhline(CONC_BAR, ls="--", lw=1.4, color="tab:purple",
               label=f"concentration bar {CONC_BAR} (e198/e199's frozen bar)")
    ax.axhline(1.0, ls=":", lw=1.3, color="dimgray")
    ax.text(5.15, 1.015, "1.0 = the own static edge (definitional)",
            fontsize=7, color="dimgray", ha="right")
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("wash step t  (ray u_t = sign(g_t) read at theta_t, the "
                  "walk's ALIVE states)")
    ax.set_ylabel("concentration ratio  D_kill(root, u_t) / "
                  "D_kill(root, u0)")
    ax.set_ylim(-0.08, max(3.6, E197_FLIGHT_RATIO * 1.15))
    ax.set_title("THE FOUR-ORGANISM ONSET PICTURE — the concentration's "
                 "arrival times", fontsize=11)
    ax.legend(fontsize=7.0, loc="upper right")

    ax = axes[1]
    led = [{"t": r["t"], "gm": r["ruler_read"]} for r in onset]
    ax.plot([p["t"] for p in led], [p["gm"] for p in led], "o-", ms=7,
            lw=1.6, color="seagreen",
            label="org2 half-step walk (g-4 ruler at theta_t)")
    for r in onset:
        if r["state"] == "alive" and r["t"] >= 1:
            ax.annotate(f"u{r['t']}", (r["t"], r["ruler_read"]),
                        textcoords="offset points", xytext=(0, 9),
                        fontsize=7.6, ha="center", color="seagreen")
    death = next(r for r in onset if r["state"] == "death")
    ax.plot([death["t"]], [death["ruler_read"]], "x", ms=12, mew=2.6,
            color="seagreen")
    ax.annotate(f"DEATH t={death['t']} (g-4 {death['ruler_read']:.4f})",
                (death["t"], death["ruler_read"]),
                textcoords="offset points", xytext=(-4, -16), fontsize=7.6,
                color="seagreen")
    ax.axhline(SHUT_BAR, ls="--", lw=1.4, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar (the ruler)")
    ax.text(0.02, 0.045, "u_t = the ray read at that alive state (u4 = the "
            "killing step's own direction);\nthe post-death front (step 6) "
            "is NEVER read", fontsize=7.4, transform=ax.transAxes,
            color="dimgray")
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("wash step t")
    ax.set_ylabel("ruler read at theta_t (g-4 battery mean p)")
    ax.set_ylim(-0.04, 1.02)
    ax.set_title("THE ALIVE LEDGER — the longest window on record "
                 "(t=1..t=4 alive)", fontsize=11)
    ax.legend(fontsize=7.6, loc="center right")
    fig.suptitle(f"E200 — THE DEEPENING TEST (T163's follow-on) — VERDICT: "
                 f"{verdict}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ray_profiles(path, fams):
    """The underlying static ray profiles (gm vs D per t) — the onset
    curve's substrate."""
    RAY_STYLE = {
        "u0": {"color": "navy", "label": "t=0: sign(g_0) — the static ray "
                                         "(the OWN edge)"},
        "u1": {"color": "crimson", "label": "t=1: sign(g_1) — e197's flight "
                                            "ray (soft, 3.17x)"},
        "u2": {"color": "darkorange", "label": "t=2: sign(g_2) — e197 "
                                               "mapped, unadjudicated"},
        "u3": {"color": "seagreen", "label": "t=3: sign(g_3) — NEW"},
        "u4": {"color": "mediumpurple", "label": "t=4: sign(g_4) — NEW (the "
                                                 "killing step's direction)"},
    }
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for key in ("u0", "u1", "u2", "u3", "u4"):
        st = RAY_STYLE[key]
        rows = fams[key]["rows"]
        ax.plot([r["D"] for r in rows], [r["gm"] for r in rows], "o-",
                ms=2.6, lw=1.7, color=st["color"], label=st["label"])
        if fams[key]["D_kill"] is not None:
            ax.axvline(fams[key]["D_kill"], color=st["color"], ls=":", lw=1.4)
            ax.text(fams[key]["D_kill"], 0.985,
                    f"{key} kills {fams[key]['D_kill']:.3f}", rotation=90,
                    fontsize=6.6, va="top", ha="right", color=st["color"])
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar (the ruler)")
    ax.set_xlabel(r"static displacement $D = \|\theta_D - \theta_0\|_2$")
    axr = ax.secondary_xaxis(
        "top", functions=(lambda d: d / RMS_DENOM,
                          lambda r: r * RMS_DENOM))
    axr.set_xlabel("per-coordinate RMS (organism 2's currency, / 934.597)")
    ax.set_ylabel("g-4 ruler read (battery mean p)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("E200 — the per-t root ray profiles (the deepening curve's "
                 "substrate; u0/u1/u2 re-gated vs e197 committed)", fontsize=10.5)
    ax.legend(fontsize=7.0, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
