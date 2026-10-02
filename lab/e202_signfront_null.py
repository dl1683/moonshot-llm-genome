"""E202 — THE SIGNFRONT-NULL CELL (the registered falsifier).

WHY (scratch/sign_front_null.md, the pure desk derivation, 2026-10-02): the
committed lag structure of the sign fronts (negative at lag 1, positive at
lag 2, ~0 at lags 3-4, on every lineage measured) is the PERIOD-2
FINGERPRINT of sign-descent overshoot — a DETERMINISTIC MINORITY (~8-28% of
coordinates, the flip class |mu| <= h*s) riding a NOISE-REDRAWN MAJORITY.
T166 resolved provisionally ("THE ALTERNATION IS THE ALGORITHM'S") and named
the null's two debts: (i) the only sharp in-domain prediction (the step-size
law) had ZERO committed in-domain evidence — the one two-size pair reverses
direction from OUTSIDE the quadratic domain (org2's full step = 1.74x its
own kill edge); (ii) nothing measured whether ANY of the alternation's
parameters move when the FACT is removed. This cell collects both debts in
CPU minutes (the e201 u3-burst class).

THE CELL (two arms, CPU-only, threads 4, tiny sequential bursts):
  (1) THE FACT-FREE TWIN — a sign walk on the PRE-INSTALL net (the committed
      e131-family pre-install state: e131_consolidated_e113.pt's own meta
      names runs/checkpoints/e048_repro.pt as its base — the inventory
      checkpoint, gated below), the SAME seed-10902 licensed wash stream
      (md5-gated vs e185's stored hashes), the SAME per-step L2 (org1's
      committed family step 1.6542880535125732 — the matched-step design),
      the same machinery (k=1 post-clip refresh, u_t = sign(g_t)/||sign(g_t)||,
      e192's fp32-norm construction), t=0..4 (the fact-free lineage has NO
      kill — there is no fact to dissolve; walk to t=4, all pairs alive by
      construction, flagged). Read cos(u_t, u_{t+1}) at every pair + the
      full lag-2/lag-3 matrix + per-step preclip gnorm + per-step battery.
      THE COMPARISON: the fact-carrying twins' committed lag-1 cosines
      (org1's, MIRABEL's, the half-step's — hard-bound from the parents).
  (2) THE STEP LADDER — sign walks on the FACT-CARRYING org2 root
      (e157_f2_consolidated, gated) at steps {s/2, s/4, s/8} x
      s = 0.9164195656776428 (the natural s kills at t=1 — no in-domain lag
      pairs; the sub-steps are strictly inside the 0.5252 static sign edge).
      The s/2 rung IS e197's committed walk REBUILT and gated row-by-row
      (G_WALKE197) — its rays md5-gated vs e197's (u1/u2) and e200's
      registered (u3/u4) — anchoring the ladder at the committed -0.26318.
      The s/4 and s/8 rungs are NEW (parents: the gated root + the gated
      stream + the registered step arithmetic).

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - FACT-IN-THE-FRONT: "fires iff the fact-free twin's lag-1 cosines sit
    >= 0.05 away from the fact-carrying twins' in the direction that makes
    the fact's fronts deeper — the front carries fact information; the
    rotation reading survives in restricted form."
  - SIGN-DESCENT-BOUNCES: "fires otherwise (the twins match within 0.05
    AND the ladder follows the null's step-size prediction) — SIGN DESCENT
    BOUNCES, AS IT MUST; the alternation noun retires finally; the
    survivors (death-at-deepest-landing, the onset curves) stand alone."
  - GRADED: "any split (e.g. the twins match but the ladder violates the
    prediction) — the null is partial; reported verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * front u_t = sign(g_t)/||sign(g_t)|| (e192's fp32-norm construction);
    g_t = the licensed seed-10902 stream's step-(t+1) post-clip batch
    gradient read AT theta_t; every cosine fp64 (cos64).
  * the twin's MATCHED pair index t = the fact-carrying twins' pair t of
    the SAME architecture class (2.74M 6L/6H/192d), the SAME natural step
    (org1's 1.6543; MIRABEL's 1.6544, same to 0.01%) and the SAME stream:
    pair 0 -> org1 (u0,u1) -0.15452721980584394 (e195, ALIVE) AND MIRABEL
    (u0,u1) -0.17510781540323442 (e198, ALIVE); pair 1 -> MIRABEL (u1,u2)
    -0.17813847702288285 (e198, ALIVE). Pairs 2/3 have NO committed alive
    matched comparator (org1 died at t=2; MIRABEL at t=3) — they ride as
    context (org1's post-death continuation, the half-step series),
    flagged, never matched. The half-step series is matched at the
    LADDER's s/2 rung, not at the twin (a different organism at half the
    step).
  * "in the direction that makes the fact's fronts deeper" = the
    fact-carrying walk reads DEEPER (cos1 more negative) than the twin:
    the deviation Delta_t = twin_cos1(t) - comparator(t) >= +0.05 (the
    fact-free front is SHALLOWER by >= 0.05, so the fact's is deeper).
    With two comparators at pair 0 (org1 + MIRABEL), the deviation must
    hold vs EVERY comparator (the strict reading).
  * FACT-IN-THE-FRONT fires iff the fact-deepening deviation (>= +0.05 vs
    every comparator) holds at BOTH matched pair indices (0 and 1) — the
    derivation's fuller draft (scratch sec.5) demands ">= 0.05 ... at
    >= 2 consecutive pair indices"; exactly 2 matched indices exist.
  * "the twins match within 0.05" = |twin_cos1(t) - comparator(t)| < 0.05
    vs EVERY comparator at EVERY matched pair index.
  * "the ladder follows the null's step-size prediction" = (a) the
    anti-phase core at t=0, core = (cos2(u0,u2) - cos1(u0,u1))/2, is
    NON-DECREASING in s across {s/8, s/4, s/2}, AND (b) cos1(u0,u1) at
    t=0 is NON-INCREASING (more negative or equal) across the same rungs
    — the derivation's letter: "the flip zone |mu| <= h*s widens with s,
    so the deterministic core (and |cos1| at matched states and steps)
    must be NON-DECREASING in s". Both statistics are reported verbatim;
    a split between (a) and (b) is itself a GRADED split, disclosed.
  * ladder rays are read at ALIVE ts only (the rung's own g-4 ruler >
    0.27, e197/e200's convention); a rung that dies before t=1 contributes
    no in-domain pair and the prediction grades on what survives
    (disclosed; nothing guaranteed). The natural-s point (org2-dead-full,
    cos1 -0.16531, e196 via e197) rides as OUT-OF-DOMAIN context only
    (1.74x its own kill edge — the derivation's own exclusion).
  * the PERIOD-2 FINGERPRINT (the scratch's PERIOD-2-UNIVERSAL: cos1 < 0
    AND cos2 > 0 AND |cos3| <= 0.05 per alive t) is computed for every
    arm and reported as CONTEXT. The scratch's fuller draft used it as a
    precondition on FACT-IN-THE-FRONT ("WHILE the fact-free arm itself
    passes PERIOD-2-UNIVERSAL"); the DISPATCH's letter does not — the
    verdict stamps per the dispatch's letter, and if the twin deviates
    while failing the fingerprint, that failure is flagged verbatim in
    the adjudication (no silent weakening either way).
  * composite order frozen: FACT-IN-THE-FRONT -> SIGN-DESCENT-BOUNCES ->
    GRADED (first that fires is the verdict; every bar's fires flag
    reported verbatim).

REGISTERED PREDICTION (frozen before compute): the null (scratch sec.6)
predicts the twin SITS ON THE CURVE (all matched |Delta| < 0.05) and the
ladder is monotone — SIGN-DESCENT-BOUNCES. The rescuer branch
(FACT-IN-THE-FRONT) stays live until the gates pass; no bar shopping
either way; the openness is the point.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication):
  G_FILES (the seven parent metrics exist, correct experiment names,
  COMPLETE status, pulled keys present) + G_CROSSFILE (the committed
  comparators are cross-consistent: e200's committed_e197 trio == e197's
  ray_geometry; e197's organism1_committed == e194's front_trace anchors;
  e198's organism1_committed == e195's ray_geometry; e200's lag matrix ==
  this module's frozen constants) + the stream gates VERBATIM
  (G_NAMEFREE, G_SPLICE, G_BATTERY at both dial conventions, G_ANCHOR vs
  e185's stored bank, G_STREAM steps 1..6 md5 vs e185's stored hashes) +
  G_ROOT (the org2 root's dial reproduces e157's committed cells
  BIT-TIGHT + the flat md5 matches e193's committed root identity) +
  G_SIGNRAY (the ladder's u0 rebuilt md5 == e193's committed R2_SIGN) +
  G_PREINSTALL (THE PRE-INSTALL STATE'S PROVENANCE GATE: the loaded
  e048_repro.pt must BE the committed install's parent — e131_consolidated's
  OWN meta names it as base, e131's committed ckpt_inventory agrees, the
  architecture class matches org1's 2,739,072 params; its flat md5 is
  registered here) + G_WALKE197 (the s/2 rung must reproduce e197's
  committed journal rows 1-6 incl. co-rulers, its x-hashes, its kill at
  t=5 and its densified bracket — bit-class or the disclosed cross-thread
  TEXTURE tier at e199's tolerances) + G_RAYSLADDER (u0 md5 vs e193's
  R2_SIGN; u1/u2 md5 vs e197's registered rays; u3/u4 md5 vs e200's
  registered rays; or the fresh mutual geometry within 1e-3 of e200's
  committed lag matrix) + THE LADDER'S MOVEMENT ARITHMETIC REGISTERED
  (the three sizes {1/2, 1/4, 1/8} x 0.9164195656776428 frozen here,
  before any walk; per-step L2 asserted EXACTLY (fp64 norm, dev < 1e-5
  asserted every step of every arm — e194's convention); the s/2 size
  asserted == e197's committed pick 0.4582097828388214 EXACTLY).

WHAT EACH ARM GUARANTEES: NOTHING — n=1 per arm; the twin is one realized
path of the licensed stream on one pre-install state; the ladder rungs
are one path each at counterfactual sizes; the fact-free walk has NO kill
by construction (its ruler baseline sits under the 0.27 bar because there
is no fact to dissolve — the kill machinery is DISABLED on that arm,
disclosed, all steps flagged alive-by-construction); the openness is the
point.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02, permanent): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never claimed),
torch threads CAPPED AT 4, load-check before launch, tiny sequential
bursts (four <= 6-step sign walks + battery reads — minutes, every phase
modest), PROGRESSIVE metrics.json writes after every phase (the outage
lesson), n=1.

Outputs: runs/e202/{metrics.json, e202_twin_vs_twins.png,
e202_ladder.png}. No NOTES/THINKING/QUEUE/STATE edits (the coordinator
folds).

Run:  cd lab && python e202_signfront_null.py    (E202_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (the envelope)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # the owner envelope's cap

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,        # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E202_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e202 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

RUNS = E43.REPO / "runs"
PARENTS = {
    "e131": RUNS / "e131" / "metrics.json",
    "e157": RUNS / "e157" / "metrics.json",
    "e193": RUNS / "e193" / "metrics.json",
    "e194": RUNS / "e194" / "metrics.json",
    "e195": RUNS / "e195" / "metrics.json",
    "e197": RUNS / "e197" / "metrics.json",
    "e198": RUNS / "e198" / "metrics.json",
    "e200": RUNS / "e200" / "metrics.json",
}

# ---- the two net classes -----------------------------------------------------
CFG1 = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)   # org1 class
N_PARAM1 = 2_739_072
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)  # org2 class
N_PARAM2 = 873_472

PREINSTALL_CK = "e048_repro.pt"                # the committed e131-family pre-install state
PREINSTALL_BASE_REF = "runs/checkpoints/e048_repro.pt"
ROOT2_CK = "e157_f2_consolidated.pt"           # the FACT-CARRYING org2 root (the ladder's root)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- the stream / battery conventions (e193/e194/e197/e200/e201 verbatim) ----
PRE, POST_CAP = 130, 119
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)
RULER_J1, CO_RULERS_J1 = -12, (-4, 0, 12)      # the twin/org1-class ruler (ZEPHYRA g-12)
RULER_J2, CO_RULERS_J2 = -4, (12, -12, 0)      # the org2-class ruler (g-4 install-60)
FREEZE_SEED = 10902
ANCH_BS, RAND_BS = 16, 16
SHUT_BAR = 0.27
R_EVAL_SEED = 26502
LADDER_STEPS = 5                               # the registered walk length per rung
TWIN_STEPS = 5                                 # t = 0..4

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

# ---- org2 committed anchors (the ladder's parents; hard-bound at load) --------
E157_DIAL = {
    "gm12": 0.19826222956180573, "g-8": 0.8519253134727478,
    "g-4": 0.8872273564338684, "g0": 0.5784125924110413,
    "g+4": 0.8801683187484741, "g+8": 0.8754011988639832,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
ROOT2_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"
E193_STEP_L2 = 0.9164195656776428              # org2's measured AdamW step-1 L2 (the natural s)
E193_USIGN_MD5 = "9395918d36425e65248dbd93b6fd50bc"    # e193's committed R2_SIGN u0
E197_SUB_S = 0.4582097828388214                # e197's committed pick (== E193_STEP_L2/2 EXACTLY)
E197_U1_MD5 = "3ce920e0dde4342133f466edc27c5138"
E197_U2_MD5 = "f333e7f8c48f0d59364bfa659c2a82f2"
E200_U3_MD5 = "d556419b910ef3a96863055fe7f18051"
E200_U4_MD5 = "9818a56ef541ca0625e93d2523d82785"
E197_STOP = {"kind": "kill", "step": 5,
             "gm_at_kill": 0.1648186892271042,
             "D_kill_raw": 0.8303853930774336,
             "D_kill": 0.838723200837794}
ROOT2_GM = 0.8872273564338684                  # the org2 root's primary ruler read

# ---- org1 committed anchors (the twin's parents; hard-bound at load) ----------
ORG1_STEP_L2 = 1.6542880535125732              # e194's matched-step constant (opt1 A0's step-1 L2)

# ---- THE COMPARATORS (the fact-carrying twins' committed lag-1 cosines) ------
ORG1_C1_ALIVE = -0.15452721980584394           # e195 ray_geometry (u0,u1) — org1's ONLY alive pair
MIR_C1 = {-0.17510781540323442, -0.17813847702288285}   # e198 (u0,u1), (u1,u2) — both ALIVE
MIR_C1_01 = -0.17510781540323442
MIR_C1_12 = -0.17813847702288285
HALF_C1 = [-0.26317857219199464, -0.3128523818599012,   # e200's committed lag matrix (context at
           -0.3412904771804916, -0.352800001023403]     # the twin; MATCHED at the ladder's s/2 rung)
ORG1_POSTDEATH_C1 = [-0.20298347429000238, -0.1856504857212099,   # e194 front_trace rows 2-5 (u1,u2)..(u4,u5) —
                     -0.15233073884753343, -0.14850772082119404]  # POST-DEATH context, never matched
DEAD_FULL_C1 = -0.1653095242890399             # e196 via e197 (org2 full step) — OUT-OF-DOMAIN context
HALF_GEOM = {                                  # e200's committed lag matrix (the G_RAYSLADDER refs)
    "cos_u0_u1": -0.26317857219199464, "cos_u1_u2": -0.3128523818599012,
    "cos_u2_u3": -0.3412904771804916, "cos_u3_u4": -0.352800001023403,
    "cos_u0_u2": 0.14055119609955105, "cos_u0_u3": -0.037019047726428826,
    "cos_u0_u4": -0.02586469602319846, "cos_u1_u3": 0.23745856827065193,
    "cos_u1_u4": 0.01718571433556464, "cos_u2_u4": 0.14245581289065523,
}

# ---- the registered ladder (FROZEN HERE, before any walk) --------------------
LADDER = [                                     # (key, fraction of the natural s, the size)
    ("eighth", 0.125, E193_STEP_L2 * 0.125),
    ("quarter", 0.25, E193_STEP_L2 * 0.25),
    ("half", 0.5, E193_STEP_L2 * 0.5),
]
assert abs(LADDER[2][2] - E197_SUB_S) < 1e-15, "the s/2 rung must BE e197's committed pick"

MATCH_BAR = 0.05                               # the registered twin-vs-twins band
PERIOD2_C3_BAR = 0.05                          # the scratch's fingerprint |cos3| bar (context)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                             # bit-class ceiling
G_TEXTURE_TOL = 1e-3                           # e199's disclosed cross-thread/environment tier
G_DKILL_TOL = 1e-2                             # D_kill texture tier (e199's)

REGISTERED_BARS = {
    "FACT_IN_THE_FRONT": "FACT-IN-THE-FRONT: \"fires iff the fact-free twin's "
        "lag-1 cosines sit >= 0.05 away from the fact-carrying twins' in the "
        "direction that makes the fact's fronts deeper — the front carries "
        "fact information; the rotation reading survives in restricted "
        "form.\"",
    "SIGN_DESCENT_BOUNCES": "SIGN-DESCENT-BOUNCES: \"fires otherwise (the "
        "twins match within 0.05 AND the ladder follows the null's step-size "
        "prediction) — SIGN DESCENT BOUNCES, AS IT MUST; the alternation noun "
        "retires finally; the survivors (death-at-deepest-landing, the onset "
        "curves) stand alone.\"",
    "GRADED": "GRADED: \"any split (e.g. the twins match but the ladder "
        "violates the prediction) — the null is partial; reported verbatim.\"",
    "operationalizations": "front u_t = sign(g_t)/||sign(g_t)|| (e192's "
        "fp32-norm construction), g_t = the licensed seed-10902 stream's "
        "step-(t+1) post-clip batch gradient read AT theta_t, cosines fp64; "
        "the twin = the pre-install net (e048_repro, e131's committed base), "
        "org1's committed family step 1.6542880535125732 (matched-step "
        "design), NO kill gating (no fact to dissolve — all steps flagged "
        "alive-by-construction); matched pair indices = 0 (org1 -0.15453 "
        "ALIVE + MIRABEL -0.17511 ALIVE) and 1 (MIRABEL -0.17814 ALIVE); "
        "pairs 2/3 unmatched (org1/MIRABEL post-death context, flagged); "
        "'the direction that makes the fact's fronts deeper' = twin "
        "SHALLOWER: Delta_t = twin - comparator >= +0.05 vs EVERY "
        "comparator; FACT-IN-THE-FRONT fires iff the fact-deepening "
        "deviation holds at BOTH matched indices (the derivation's '>= 2 "
        "consecutive pair indices'; exactly 2 exist); 'match within 0.05' = "
        "|Delta| < 0.05 vs every comparator at every matched index; the "
        "ladder follows the null's step-size prediction iff (a) core(t=0) = "
        "(cos2(u0,u2) - cos1(u0,u1))/2 non-decreasing across {s/8, s/4, "
        "s/2} AND (b) cos1(u0,u1) non-increasing across the same rungs (a "
        "split between (a) and (b) is itself GRADED, disclosed); ladder "
        "rays at ALIVE ts only (own g-4 ruler > 0.27; a rung dying before "
        "t=1 contributes no in-domain pair and the prediction grades on "
        "what survives); the natural-s point (org2-dead-full -0.16531) is "
        "OUT-OF-DOMAIN context (1.74x its own kill edge — the derivation's "
        "own exclusion); the PERIOD-2 fingerprint (cos1<0, cos2>0, |cos3| "
        "<= 0.05) is computed for every arm as CONTEXT (the scratch's "
        "fuller draft used it as a precondition on FACT-IN-THE-FRONT; the "
        "dispatch's letter adjudicates — a deviating twin that fails the "
        "fingerprint is flagged verbatim, never silently weakened); "
        "composite order frozen FACT-IN-THE-FRONT -> SIGN-DESCENT-BOUNCES "
        "-> GRADED.",
    "registered_prediction": "the null (scratch/sign_front_null.md sec.6) "
        "predicts the twin SITS ON THE CURVE (all matched |Delta| < 0.05) "
        "and the ladder is monotone — SIGN-DESCENT-BOUNCES; the rescuer "
        "branch (FACT-IN-THE-FRONT) stays live until the gates pass; no "
        "bar shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE SPLIT: the fact-carrying twins' lag-1 cosines are LOADED COMMITTED "
    "and hard-bound (org1's from e195, MIRABEL's from e198, the half-step "
    "lag matrix from e200 with e197's trio cross-gated); the ONLY fresh "
    "compute is the fact-free twin walk (5 sign steps on the gated "
    "pre-install state) + the ladder rungs {s/2, s/4, s/8} (6/5/5 sign "
    "steps on the gated org2 root) + battery reads — the e201 u3-burst "
    "class, CPU minutes.",
    "THREADS 4 (the owner envelope's cap) where e197's committed chain ran "
    "threads 8: cross-thread/environment drift at ~1e-7 fp32 is possible, "
    "so every recompute gate is TIERED (BIT md5/1e-9 or e199's disclosed "
    "TEXTURE tier at 1e-3), the achieved tier STAMPED per gate; e200 "
    "rebuilt the same walk at threads 4 to TEXTURE/BIT tier — this process "
    "runs the same cap on the same machine.",
    "THE TWIN'S STEP (matched-step design, load-bearing): the walk uses "
    "org1's COMMITTED family step 1.6542880535125732 (e194's "
    "matched_step_L2, hard-bound at load), NOT a re-measured AdamW L2 at "
    "the pre-install state — the comparison is at matched s by "
    "construction (the scratch's letter: 'full-step 1.6543 walk').",
    "THE TWIN HAS NO KILL BY CONSTRUCTION: the pre-install net's ruler "
    "baseline sits under the 0.27 bar because there is no fact to dissolve "
    "(p(Z) ~ unigram baseline); the kill machinery is DISABLED on that arm "
    "(disclosed), the battery is read every step and REPORTED, never "
    "adjudicated; all four consecutive pairs are alive-by-construction.",
    "the ladder's s/4 and s/8 rungs are NEW walks (no committed anchor): "
    "their only parents are the gated root (G_ROOT + G_SIGNRAY), the "
    "md5-gated stream and the registered step arithmetic (per-step L2 "
    "asserted exact, sizes frozen in LADDER before any walk); the s/2 rung "
    "is e197's walk REBUILT and gated row-by-row (G_WALKE197) with its "
    "rays md5-gated vs e197's (u1/u2) and e200's registered (u3/u4).",
    "no static ray mapping in this cell (no D_kill profiles): the ladder "
    "reads only the walk fronts' mutual cosines — the lethality "
    "instruments (death-at-deepest-landing, onset curves) are the null's "
    "SURVIVORS and are not touched here.",
    "Smoke mode trims: twin 2 steps, rungs 2 steps, G_WALKE197 gated on the "
    "first 2 rows only (verdict stamped SMOKE; nothing adjudicated).",
]


# ---------------------------------------------------------------- instruments
# PROVENANCE: lab/e201_rotation_census.py's instruments (whose own provenance
# is e198/e193/e193b/e192 via e195 — the e176n lineage) + lab/e200_deepening.py's
# walk_sub arithmetic. Copied rather than imported to own the device policy.

def load_net(path, cfg) -> TinyGPT:
    m = TinyGPT(cfg)
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
    return {"mean_pz": float(p.mean()), "frac_argmax_z": amax / ids.shape[0]}


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
    """e193/e194/opt2's sign_update VERBATIM: delta = -step_l2*sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — the matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


def cpu_load_probe() -> int | None:
    """Best-effort CPU load snapshot (the envelope's load-check record)."""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


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


# ------------------------------------------------------------------ the walk

def sign_walk(net0, anchor_neutral, train_ids, itos, primary_ids, coruler_ids,
              zid, theta0, step_l2, step_cap, kill_gate, root_gm, tag,
              extend_past_kill=1, d_target=5.0):
    """The e194/e197/e200 sign walk VERBATIM in its arithmetic (k=1 post-clip
    refresh, per-step battery, densified kill bracket at the killing step;
    stashes g_fronts + endpoints — the ray factory).

    kill_gate=False (THE TWIN): no kill machinery at all — exactly step_cap
    steps, every row flagged alive-by-construction (the fact-free lineage
    has no fact to dissolve; disclosed in the registration).
    """
    net = copy.deepcopy(net0)
    net.train()
    evl = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    traj, x_hashes = [], {}
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_flat = theta0
    kill = None
    step = 0
    t0a = time.time()
    while True:
        step += 1
        post_kill = kill is not None
        if kill_gate:
            if post_kill and step > kill["step"] + extend_past_kill:
                break
            if kill is None and step > step_cap:
                kill = {"kind": "cap", "step": step - 1,
                        "reason": f"step cap {step_cap - 1} alive "
                                  f"(the registered {step_cap - 1}-step rung)",
                        "final_D": traj[-1]["cum_disp"],
                        "final_gm": traj[-1]["gm"]}
                break
        else:
            if step > step_cap:
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
                f"[{tag}] step-{step} batch md5 diverged from the licensed stream"
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
        assert l2dev < 1e-5, f"[{tag}] per-step L2 dev {l2dev:.2e}"
        max_l2_dev = max(max_l2_dev, l2dev)
        cur = prev_flat + delta
        load_flat(net, cur)
        cum_disp = float(torch.norm(cur - theta0))
        evl.load_state_dict({k_: v.detach().cpu().clone()
                             for k_, v in net.state_dict().items()})
        evl.eval()
        gz = battery_cell(evl, primary_ids, zid)
        row = {"step": step, "post_kill": post_kill,
               "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"]}
        for j in coruler_ids:
            row[f"g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        traj.append(row)
        g_fronts.append(g_t.clone())
        endpoints[step] = cur.clone()
        prev_flat = cur
        log(f"  [{tag} s{step}]{' POSTKILL' if post_kill else ''} "
            f"gm {row['gm']:.10f} D {cum_disp:.10f} ce {row['ce_batch']:.6f} "
            f"gnorm {gnorm:.4f}")
        if kill_gate and kill is None and gz["mean_pz"] <= SHUT_BAR:
            dens = []
            for f_ in (0.2, 0.4, 0.6, 0.8):
                pt_flat = prev_flat + (f_ - 1.0) * delta
                load_flat(evl, pt_flat)
                evl.eval()
                gzd = battery_cell(evl, primary_ids, zid)
                dens.append({"f": f_, "gm": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm, "cum_disp": 0.0}
            d_raw = interp_d_kill(prev_row["gm"], row["gm"],
                                  prev_row["cum_disp"], cum_disp)
            d_dens = dens_d_kill(prev_row["gm"], row["gm"],
                                 prev_row["cum_disp"], cum_disp, dens)
            kill = {"kind": "kill", "step": step,
                    "gm_at_kill": row["gm"], "D_kill_raw": d_raw,
                    "D_kill": d_dens, "dens": dens,
                    "reason": "primary ruler <= SHUT at an every-step read"}
            log(f"  [{tag}] KILL at s{step}: D_kill(dens) {d_dens:.10f}")
        elif kill_gate and kill is None and cum_disp >= d_target:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {d_target} alive",
                    "final_D": cum_disp, "final_gm": row["gm"]}
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "stop": kill, "x_hashes": x_hashes,
            "max_l2_dev": max_l2_dev, "g_fronts": g_fronts,
            "endpoints": endpoints, "seconds": round(time.time() - t0a, 1)}


def rays_and_geometry(g_fronts, n_alive):
    """u_t = sign(g_t)/||sign(g_t)|| for t < n_alive (ALIVE reads only);
    the full mutual-cosine matrix (fp64); lag-k series; the period-2
    fingerprint per available t."""
    us = []
    for t in range(n_alive):
        s = torch.sign(g_fronts[t])
        us.append((s / torch.norm(s)).clone())
    md5s = {f"u{t}": hashlib.md5(u.numpy().tobytes()).hexdigest()
            for t, u in enumerate(us)}
    n = len(us)
    cos = {(i, j): cos64(us[i], us[j]) for i in range(n) for j in range(i + 1, n)}
    lag = {k: [cos[(t, t + k)] for t in range(n - k)]
           for k in range(1, min(4, n))}
    p2 = {}
    for t in range(n - 1):
        c1 = cos[(t, t + 1)]
        c2 = cos[(t, t + 2)] if t + 2 < n else None
        c3 = cos[(t, t + 3)] if t + 3 < n else None
        p2[f"t{t}"] = {"cos1": c1, "cos2": c2, "cos3": c3,
                       "cos1_neg": bool(c1 < 0),
                       "cos2_pos": bool(c2 is not None and c2 > 0),
                       "cos3_small": bool(c3 is None or abs(c3) <= PERIOD2_C3_BAR),
                       "fingerprint_holds": bool(
                           c1 < 0 and (c2 is not None and c2 > 0)
                           and (c3 is None or abs(c3) <= PERIOD2_C3_BAR))}
    geom = {"n_fronts": n, "ray_md5s": md5s,
            "mutual_cosines": {f"u{i}_u{j}": v for (i, j), v in cos.items()},
            "lag_series": {f"lag{k}": v for k, v in lag.items()},
            "period2_fingerprint": p2}
    if n >= 3:
        geom["core_t0"] = (cos[(0, 2)] - cos[(0, 1)]) / 2.0
        geom["cos1_t0"] = cos[(0, 1)]
    return us, geom


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e202_smoke" if SMOKE else "e202")
    log(f"E202 THE SIGNFRONT-NULL CELL (smoke={SMOKE}) -> {rd}")
    load_pct = cpu_load_probe()
    if load_pct is not None and load_pct > 85:
        log(f"load probe {load_pct}% > 85 — waiting 60 s (the envelope's "
            f"stagger rule)")
        time.sleep(60)
        load_pct = cpu_load_probe()
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), load probe "
        f"{load_pct if load_pct is not None else 'n/a'}%, four <= 6-step "
        f"sign walks + battery reads (the e201 u3-burst class)")

    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e202_signfront_null",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE, "torch": torch.__version__,
                               "threads": torch.get_num_threads(),
                               "load_probe_pct": load_pct},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    # =====================================================================
    # PHASE 0 — parents loaded + the stream/root gates (Rule 12)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the parent ledger + the standard cell gates")
    missing = [str(p) for p in PARENTS.values() if not p.exists()]
    G_FILES = {"parents": {k: str(v) for k, v in PARENTS.items()},
               "missing": missing}
    assert not missing, f"missing parent metrics: {missing}"
    M = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in PARENTS.items()}
    name_checks = {k: (M[k].get("experiment") or "").startswith(k)
                   for k in PARENTS}
    # e131/e157 are older-schema metrics (no status field — the e13x/e15x
    # era predates the COMPLETE convention); their identity rides on the
    # experiment name + the pulled keys + G_CROSSFILE's hard-binds. The six
    # e19x-era parents must read COMPLETE.
    MODERN_PARENTS = ("e193", "e194", "e195", "e197", "e198", "e200")
    status_checks = {k: ("COMPLETE" in str(M[k].get("status", "")))
                     for k in MODERN_PARENTS}
    keys_needed = {
        "e131": ["ckpt_inventory"],
        "e157": ["stages"],
        "e193": ["organism"],
        "e194": ["cell", "phaseA_front_overlap"],
        "e195": ["ray_geometry"],
        "e197": ["phase1_alive_walk", "phase1b_rays"],
        "e198": ["ray_geometry"],
        "e200": ["ray_geometry", "cell"],
    }
    key_checks = {k: all(n in M[k] for n in ns)
                  for k, ns in keys_needed.items()}
    G_FILES.update({"experiment_names": name_checks,
                    "statuses_complete": status_checks,
                    "status_check_note": "COMPLETE required of the six "
                                         "e19x-era parents only (e131/e157 "
                                         "predate the status convention — "
                                         "disclosed)",
                    "pulled_keys_present": key_checks,
                    "pass": bool(all(name_checks.values())
                                 and all(status_checks.values())
                                 and all(key_checks.values()))})
    log(f"G_FILES: 8 parents, names {sum(name_checks.values())}/8, COMPLETE "
        f"{sum(status_checks.values())}/8, keys {sum(key_checks.values())}/8: "
        + ("PASS" if G_FILES["pass"] else "FAIL"))
    assert G_FILES["pass"], "parent identity gate failed"

    # ---- G_CROSSFILE: the committed comparators agree across files ---------
    e194_cell = M["e194"]["cell"]
    e194ft = M["e194"]["phaseA_front_overlap"]["front_trace"]
    g195 = M["e195"]["ray_geometry"]
    anchors195 = g195["committed_anchors"]
    g197 = M["e197"]["phase1b_rays"]["ray_geometry"]
    g198 = M["e198"]["ray_geometry"]
    g200 = M["e200"]["ray_geometry"]
    e197_walk = M["e197"]["phase1_alive_walk"]

    def exact(a, b):
        return abs(a - b) == 0.0

    xf = {
        "e194_step_l2_constants": exact(
            e194_cell["matched_step_L2"]["value"], ORG1_STEP_L2),
        "e193_step_l2_constants": exact(
            M["e193"]["organism"]["step_l2_measured"], E193_STEP_L2),
        "e197_sub_s_constants": exact(e197_walk["sub_s"], E197_SUB_S),
        "e157_dial_constants": all(
            abs(M["e157"]["stages"]["A_consolidate"]["dial"]["base"][k]
                ["mean_pz"] - v) < 1e-12
            for k, v in (("-12", E157_DIAL["gm12"]), ("0", E157_DIAL["g0"]),
                         ("12", E157_DIAL["gp12"]))),
        "e200_lag_matrix_constants": all(
            exact(g200[k], v) for k, v in HALF_GEOM.items()),
        "e200_committed_e197_trio": all(
            exact(g200["committed_e197"][k], g197[k])
            for k in ("cos_u0_u1", "cos_u0_u2", "cos_u1_u2")),
        # the org1 chain has TWO precision classes: e195's committed_anchors
        # (and e197's/e198's copies of them) are fp32-rounded sign-form
        # reads (agree to ~1e-12 among themselves); e195's normalized
        # mutuals and e194's front_trace rows are the fp64 values (exact)
        "e197_org1_committed_vs_e195_anchors": all(
            abs(g197["organism1_committed"][k] - anchors195[a]) < 1e-12
            for k, a in (("cos_u0_u1", "cos_sign0_sign1_e194"),
                         ("cos_u0_u2", "cos_sign0_sign2_e194"),
                         ("cos_u1_u2", "cos_sign1_sign2_e194"))),
        "e198_org1_committed_vs_e195_normalized": all(
            exact(g198["organism1_committed"][k], g195[k])
            for k in ("cos_u0_u1", "cos_u0_u2", "cos_u1_u2")),
        "e195_anchors_vs_e194_fp32class": all(
            abs(anchors195[a] - v) < 1e-9
            for a, v in (("cos_sign0_sign1_e194",
                          e194ft[1]["vs_prev_front"]),
                         ("cos_sign0_sign2_e194",
                          e194ft[2]["vs_static_sign"]),
                         ("cos_sign1_sign2_e194",
                          e194ft[2]["vs_prev_front"]))),
        "e195_normalized_vs_e194_exact": all(
            abs(g195[k] - v) < 1e-8
            for k, v in (("cos_u0_u1", e194ft[1]["vs_prev_front"]),
                         ("cos_u0_u2", e194ft[2]["vs_static_sign"]),
                         ("cos_u1_u2", e194ft[2]["vs_prev_front"]))),
        "e195_org1_constants": exact(g195["cos_u0_u1"], ORG1_C1_ALIVE),
        "e198_mirabel_constants": exact(g198["cos_u0_u1"], MIR_C1_01)
                                  and exact(g198["cos_u1_u2"], MIR_C1_12),
        "e194_org1_postdeath_constants": all(
            abs(e194ft[i]["vs_prev_front"] - v) < 1e-15
            for i, v in ((2, ORG1_POSTDEATH_C1[0]), (3, ORG1_POSTDEATH_C1[1]),
                         (4, ORG1_POSTDEATH_C1[2]), (5, ORG1_POSTDEATH_C1[3]))),
        "e197_deadfull_constants": abs(
            g197["organism2_deadlineage_committed"]["cos_u0_u1"]
            - DEAD_FULL_C1) < 1e-15,
        "e197_ray_md5s_constants": all(
            M["e197"]["cell"]["rays"][i]["u_md5"] == h
            for i, h in ((1, E197_U1_MD5), (2, E197_U2_MD5))),
        "e200_ray_md5s_constants": all(
            M["e200"]["cell"]["rays"][i]["u_md5"] == h
            for i, h in ((3, E200_U3_MD5), (4, E200_U4_MD5))),
        "e200_u0_md5_vs_e193_r2sign": (
            M["e200"]["cell"]["rays"][0]["u_md5"] == E193_USIGN_MD5),
        "e131_inventory_names_preinstall_base": (
            M["e131"]["ckpt_inventory"]["saved"]["e131_consolidated_e113.pt"]
            ["meta"]["base"] == PREINSTALL_BASE_REF),
        "e197_journal_rows": [r["step"] for r in e197_walk["journal"]]
                             == [1, 2, 3, 4, 5, 6],
        "e197_stop_constants": (
            e197_walk["stop"]["kind"] == E197_STOP["kind"]
            and e197_walk["stop"]["step"] == E197_STOP["step"]
            and abs(e197_walk["stop"]["D_kill"] - E197_STOP["D_kill"]) < 1e-12),
    }
    G_CROSSFILE = {
        "checks": {k: bool(v) for k, v in xf.items()},
        "pass": bool(all(xf.values())),
        "note": "the committed comparators (the fact-carrying twins' lag-1 "
                "cosines + the half-step lag matrix + the ray md5s + the "
                "pre-install provenance record) are cross-consistent across "
                "the eight parents BEFORE anything is compared (Rule 12)",
    }
    bad = [k for k, v in xf.items() if not v]
    log(f"G_CROSSFILE: {sum(bool(v) for v in xf.values())}/{len(xf)} checks "
        f"{'PASS' if G_CROSSFILE['pass'] else 'FAIL: ' + str(bad)}")
    assert G_CROSSFILE["pass"], f"cross-file consistency failed: {bad}"
    e197_journal = e197_walk["journal"]
    e197_xh = e197_walk["x_hashes"]

    # ---- protocol rebuild (the shared corpus/battery/stream machinery) ------
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
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in READ_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in READ_GEOS},
        "pass": bool(all(list(bat_ids[j].shape) == [60, PRE + j]
                         for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): the install-60 battery (the "
                "ZEPHYRA install — org1's and org2's own host set) at the "
                "seven read geometries, 60 x (130 +- j) — e193/e194/e197/"
                "e200's gate verbatim; it serves BOTH classes' rulers "
                "(twin g-12 / org2 g-4)",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    twin_primary, twin_corulers = bat_ids[RULER_J1], {j: bat_ids[j] for j in CO_RULERS_J1}
    org2_primary, org2_corulers = bat_ids[RULER_J2], {j: bat_ids[j] for j in CO_RULERS_J2}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)

    arng = random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")):
            rejections += 1
            continue
        n_starts.append(s)
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": "16 plain corpus windows from train_ids, RNG seed "
                        "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                        "in [s, s+257) — e170 VERBATIM",
        "starts": n_starts,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 7):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_6_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values())),
                "note": "e193/e194/e197's G_STREAM VERBATIM: the seed-10902 "
                        "stream construction (net-independent — the SAME "
                        "batches serve the twin and every ladder rung) "
                        "md5-matches e185's stored hashes, steps 1..6"}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..6): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # ---- G_ROOT: the org2 root (the ladder's root) -------------------------
    net2 = load_net(CKPT_DIR / ROOT2_CK, F2_CFG)
    theta2 = flat_params(net2)
    assert int(theta2.numel()) == N_PARAM2, f"org2 params {theta2.numel()}"
    root2_md5 = hashlib.md5(theta2.numpy().tobytes()).hexdigest()
    evl2 = copy.deepcopy(net2)
    root2_cells = {f"g{j:+d}": battery_cell(evl2, bat_ids[j], zid)["mean_pz"]
                   for j in READ_GEOS}
    root2_cells["ce_r"] = ce_fixed_cpu(evl2, r_eval_x, r_eval_y)
    keymap = {"gm12": "g-12", "g0": "g+0", "gp12": "g+12",
              "g-4": "g-4", "g+4": "g+4", "g-8": "g-8", "g+8": "g+8",
              "ce_r": "ce_r"}
    refs2 = {keymap[k]: v for k, v in E157_DIAL.items()}
    diffs2 = {k: root2_cells[k] - refs2[k] for k in refs2}
    rmax2 = max(abs(v) for v in diffs2.values())
    G_ROOT = {"cells": root2_cells, "refs": refs2, "diffs": diffs2,
              "max_abs_diff": rmax2, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax2 < G_BIT_TOL),
              "flat_md5": root2_md5,
              "flat_md5_match_e193_committed": bool(root2_md5 == ROOT2_FLAT_MD5),
              "pass": bool(rmax2 < G_FALLBACK_TOL
                           and root2_md5 == ROOT2_FLAT_MD5),
              "note": "e193/e197/e200's G_ROOT VERBATIM: the f2 "
                      "consolidated root's dial reproduces e157's committed "
                      "cells BIT-TIGHT + the flat md5 matches e193's "
                      "committed root identity"}
    log(f"G_ROOT (org2, vs e157 committed dial): max|diff| {rmax2:.2e}, md5 "
        f"{'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    assert G_ROOT["pass"], "org2 root gate FAILED"

    # ---- G_PREINSTALL: the pre-install state's provenance gate -------------
    st_pre = torch.load(CKPT_DIR / PREINSTALL_CK, map_location="cpu",
                        weights_only=False)
    net_pre = load_net(CKPT_DIR / PREINSTALL_CK, CFG1)
    theta_pre = flat_params(net_pre)
    n_pre = int(theta_pre.numel())
    pre_md5 = hashlib.md5(theta_pre.numpy().tobytes()).hexdigest()
    evl_pre = copy.deepcopy(net_pre)
    pre_cells = {f"g{j:+d}": battery_cell(evl_pre, bat_ids[j], zid)["mean_pz"]
                 for j in READ_GEOS}
    pre_cells["ce_r"] = ce_fixed_cpu(evl_pre, r_eval_x, r_eval_y)
    e131_meta_base = M["e131"]["ckpt_inventory"]["saved"][
        "e131_consolidated_e113.pt"]["meta"]["base"]
    ck_meta = st_pre.get("meta", {}) if isinstance(st_pre, dict) else {}
    G_PREINSTALL = {
        "path": str(CKPT_DIR / PREINSTALL_CK),
        "identity": "THE PRE-INSTALL STATE'S PROVENANCE GATE: the loaded net "
                    "must BE the committed install's parent — the org1 root "
                    "(e131_consolidated_e113.pt)'s OWN meta names "
                    "runs/checkpoints/e048_repro.pt as its base AND e131's "
                    "committed ckpt_inventory agrees; the architecture "
                    "class matches org1's (6L/6H/192d/256-ctx, 2,739,072 "
                    "params); the flat md5 is registered here for any "
                    "follow-on",
        "e131_root_meta_base": e131_meta_base,
        "e131_root_meta_base_names_this_file": bool(
            e131_meta_base == PREINSTALL_BASE_REF),
        "ckpt_top_keys": list(st_pre.keys()) if isinstance(st_pre, dict) else [],
        "ckpt_meta": E43.jsonable(ck_meta),
        "ckpt_train_step": st_pre.get("step") if isinstance(st_pre, dict) else None,
        "n_param": n_pre,
        "n_param_matches_org1_class": bool(n_pre == N_PARAM1),
        "flat_md5_registered": pre_md5,
        "fact_free_baseline_dial": pre_cells,
        "baseline_note": "the pre-install battery reads p(Z) at the SAME "
                         "install windows org1's ruler uses: baseline-level "
                         "(no fact installed) — the ruler is undefined-by-"
                         "construction for kill gating (this is WHY the "
                         "twin's kill machinery is disabled); reported, "
                         "never adjudicated",
        "pass": bool(e131_meta_base == PREINSTALL_BASE_REF
                     and n_pre == N_PARAM1),
    }
    log(f"G_PREINSTALL: e131's meta base == {e131_meta_base} | params "
        f"{n_pre} | flat md5 {pre_md5[:8]}...: "
        + ("PASS" if G_PREINSTALL["pass"] else "FAIL"))
    assert G_PREINSTALL["pass"], "pre-install provenance gate FAILED"

    stub["gates"] = {"G_FILES": G_FILES, "G_CROSSFILE": G_CROSSFILE,
                     "G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_STREAM": G_STREAM, "G_ROOT": G_ROOT,
                     "G_PREINSTALL": G_PREINSTALL}
    stub["registered_ladder"] = {
        "natural_s": E193_STEP_L2,
        "rungs": [{"key": k, "fraction": f, "s": s} for k, f, s in LADDER],
        "movement_arithmetic": "per-step L2 asserted EXACT (fp64 norm, dev "
                               "< 1e-5, every step of every arm — e194's "
                               "convention); the s/2 rung == e197's committed "
                               "pick 0.4582097828388214 EXACTLY; all three "
                               "rungs strictly inside the 0.5252 static sign "
                               "edge (in-domain)",
        "twin_step": ORG1_STEP_L2,
        "twin_step_provenance": "e194's committed matched_step_L2 (org1's "
                                "natural wash step — the matched-step design; "
                                "hard-bound at load, never re-measured)",
    }
    write_partial("phase 0 complete (parents + stream/root/pre-install gates)")
    log("phase 0 complete; partial metrics written")

    log("WHAT THESE ARMS GUARANTEE: NOTHING — n=1 per arm; the twin is one "
        "path of the licensed stream on the pre-install state; the rungs are "
        "one path each at counterfactual sizes; the openness is the point.")

    # =====================================================================
    # PHASE 1 — THE FACT-FREE TWIN (the walk the falsifier is built on)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE 1 — THE FACT-FREE TWIN: the pre-install net, the seed-10902 "
        f"stream, org1's family step {ORG1_STEP_L2:.10f}, {TWIN_STEPS} steps, "
        f"NO kill (by construction)")
    n_twin = 2 if SMOKE else TWIN_STEPS
    wt = sign_walk(net_pre, anchor_neutral, train_ids, itos, twin_primary,
                   twin_corulers, zid, theta_pre, ORG1_STEP_L2, n_twin,
                   kill_gate=False, root_gm=pre_cells[f"g{RULER_J1:+d}"],
                   tag="twin")
    assert len(wt["traj"]) == n_twin and wt["stop"] is None
    g0_pre = wt["g_fronts"][0]
    twin_zero_coords = int((g0_pre == 0).sum())
    us_twin, twin_geom = rays_and_geometry(wt["g_fronts"], n_twin)
    twin_block = {
        "state": {"ckpt": PREINSTALL_CK, "flat_md5": pre_md5,
                  "n_param": n_pre, "ruler": f"g{RULER_J1:+d} ZEPHYRA install-60",
                  "step_l2": ORG1_STEP_L2,
                  "kill_gating": "DISABLED BY CONSTRUCTION — the fact-free "
                                 "lineage has no fact to dissolve; all "
                                 "steps flagged alive-by-construction"},
        "walk_journal": wt["traj"], "x_hashes": wt["x_hashes"],
        "max_l2_dev": wt["max_l2_dev"], "seconds": wt["seconds"],
        "n_zero_g_coords_t0": twin_zero_coords,
        "geometry": twin_geom,
        "lag1_series": twin_geom["lag_series"].get("lag1"),
        "note": "THE FACT-FREE TWIN's cosine series — the falsifier's "
                "left-hand side; the fronts are read at every theta_t "
                "t=0..4 (all alive by construction); the battery column is "
                "the fact-free baseline (reported, never adjudicated)",
    }
    log(f"twin lag-1 series: "
        + ", ".join(f"{v:+.5f}" for v in twin_geom["lag_series"]["lag1"]))
    stub["phase1_factfree_twin"] = twin_block
    write_partial("phase 1 complete (the fact-free twin walked)")
    log("phase 1 complete; partial metrics written")

    # =====================================================================
    # PHASES 2-4 — THE STEP LADDER {s/2, s/4, s/8} on the FACT-CARRYING root
    # =====================================================================
    log("=" * 78)
    log("THE LADDER — sign walks on the FACT-CARRYING org2 root at the "
        "registered sizes " + ", ".join(f"{k} s={s:.10f}" for k, _, s in LADDER))
    rungs = []
    for key, frac, s in LADDER:
        cap = (6 if key == "half" else (2 if SMOKE else LADDER_STEPS))
        if key == "half" and not SMOKE:
            cap = 8        # e197's convention: kill at 5 + 1 post-kill row
        wk = sign_walk(net2, anchor_neutral, train_ids, itos, org2_primary,
                       org2_corulers, zid, theta2, s, cap,
                       kill_gate=True, root_gm=ROOT2_GM, tag=f"ladder-{key}")
        kill_step = wk["stop"]["step"] if wk["stop"]["kind"] == "kill" else None
        n_alive = kill_step if kill_step is not None else len(wk["traj"])
        us_k, geom_k = rays_and_geometry(wk["g_fronts"], n_alive)
        rung = {"key": key, "fraction": frac, "s": s, "step_cap": cap,
                "walk_journal": wk["traj"], "stop": wk["stop"],
                "x_hashes": wk["x_hashes"], "max_l2_dev": wk["max_l2_dev"],
                "seconds": wk["seconds"],
                "alive_ts": list(range(n_alive)),
                "alive_convention": "fronts read at theta_t for t < kill step "
                                    "(own g-4 ruler > 0.27; the post-kill "
                                    "row rides in the journal only)",
                "geometry": geom_k,
                "lag1_series": geom_k["lag_series"].get("lag1"),
                "period2": geom_k["period2_fingerprint"]}
        rungs.append((key, frac, s, wk, us_k, geom_k, rung))
        if "phase_ladder" not in stub:
            stub["phase_ladder"] = {}
        stub["phase_ladder"][key] = rung
        write_partial(f"ladder rung {key} complete (s={s:.6f})")
        log(f"rung {key}: lag-1 series "
            + ", ".join(f"{v:+.5f}" for v in geom_k["lag_series"]["lag1"]))
        del us_k

    # ---- G_WALKE197: the s/2 rung vs e197's committed journal ---------------
    half = next(r for r in rungs if r[0] == "half")
    wk_half = half[3]
    n_gate_rows = 2 if SMOKE else 6
    row_keys = ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm",
                "gm", "frac_argmax_z", "g+12", "g-12", "g+0")
    rows_gate = {}
    for i, (a, b) in enumerate(zip(wk_half["traj"][:n_gate_rows],
                                   e197_journal[:n_gate_rows])):
        for k in row_keys:
            mv, cv = a.get(k), b.get(k)
            if mv is None or cv is None:
                continue
            rows_gate[f"s{i+1}_{k}"] = {"measured": mv, "committed": cv,
                                        "abs_diff": abs(mv - cv)}
    md = max(v["abs_diff"] for v in rows_gate.values()) if rows_gate else 1.0
    stop_gate_ok = True
    stop_detail = {}
    if not SMOKE:
        sc, wc = wk_half["stop"], e197_walk["stop"]
        stop_detail = {
            "kind": (sc["kind"] == wc["kind"] == "kill"),
            "step": (sc["step"] == wc["step"] == 5),
            "gm_at_kill": abs(sc["gm_at_kill"] - wc["gm_at_kill"]),
            "D_kill_raw": abs(sc["D_kill_raw"] - wc["D_kill_raw"]),
            "D_kill": abs(sc["D_kill"] - wc["D_kill"]),
            "dens_gm_max": max(abs(a["gm"] - b["gm"]) for a, b in
                               zip(sc["dens"], wc["dens"])),
            "dens_D_max": max(abs(a["D"] - b["D"]) for a, b in
                              zip(sc["dens"], wc["dens"])),
        }
        stop_gate_ok = (stop_detail["kind"] and stop_detail["step"]
                        and stop_detail["D_kill"] < G_DKILL_TOL
                        and stop_detail["dens_gm_max"] < G_TEXTURE_TOL)
    xh_ok = {s_: bool(wk_half["x_hashes"][s_] == e197_xh[str(s_)])
             for s_ in range(1, n_gate_rows + 1)}
    tier_w = "BIT" if md < G_REPRO_TOL else ("TEXTURE" if md < G_TEXTURE_TOL
                                             else "FAIL")
    G_WALKE197 = {
        "rows": rows_gate, "max_abs_diff": md,
        "tol_bit": G_REPRO_TOL, "tol_texture": G_TEXTURE_TOL, "tier": tier_w,
        "stop_gate": stop_detail, "stop_gate_ok": bool(stop_gate_ok),
        "x_hashes_vs_e197": xh_ok, "max_l2_dev": wk_half["max_l2_dev"],
        "pass": bool(tier_w != "FAIL" and stop_gate_ok
                     and all(xh_ok.values())),
        "note": "THE LADDER'S ANCHOR GATE (Rule 12): the s/2 rung must "
                "reproduce e197's committed journal (rows incl. co-rulers, "
                "the x-hashes, the kill at t=5 and its densified bracket) "
                "— bit-class, or the disclosed cross-thread TEXTURE tier "
                "(threads 4 here vs e197's 8; e200's precedent); the s/4 "
                "and s/8 rungs are not believed until the s/2 rung passes",
    }
    log(f"G_WALKE197 (s/2 rung vs e197 committed): max|diff| {md:.2e} "
        f"({tier_w}), stop {'OK' if stop_gate_ok else 'DRIFT'}: "
        + ("PASS" if G_WALKE197["pass"] else "FAIL"))
    if not G_WALKE197["pass"]:
        stub["gates"]["G_WALKE197"] = G_WALKE197
        write_partial("CONTROL FAILURE — the s/2 anchor rung failed")
        raise RuntimeError("G_WALKE197 FAILED — abort before any rung is believed")
    stub["gates"]["G_WALKE197"] = G_WALKE197

    # ---- G_RAYSLADDER: the s/2 rung's rays vs the committed md5s ------------
    us_half = []
    n_alive_half = 5 if not SMOKE else min(5, len(wk_half["g_fronts"]))
    for t in range(n_alive_half):
        sg = torch.sign(wk_half["g_fronts"][t])
        us_half.append((sg / torch.norm(sg)).clone())
    half_md5s = {f"u{t}": hashlib.md5(u.numpy().tobytes()).hexdigest()
                 for t, u in enumerate(us_half)}
    committed_md5s = {"u0": E193_USIGN_MD5, "u1": E197_U1_MD5,
                      "u2": E197_U2_MD5, "u3": E200_U3_MD5, "u4": E200_U4_MD5}
    md5_ok = {k: bool(half_md5s.get(k) == committed_md5s[k])
              for k in committed_md5s if k in half_md5s}
    fresh_half_geom = {}
    for i in range(len(us_half)):
        for j in range(i + 1, len(us_half)):
            fresh_half_geom[f"cos_u{i}_u{j}"] = cos64(us_half[i], us_half[j])
    geom_dev = {k: abs(fresh_half_geom[k] - HALF_GEOM[k])
                for k in fresh_half_geom if k in HALF_GEOM}
    max_geom_dev = max(geom_dev.values()) if geom_dev else 0.0
    tier_r = ("BIT" if all(md5_ok.values()) and len(md5_ok) == 5
              else ("TEXTURE" if max_geom_dev < G_TEXTURE_TOL else "FAIL"))
    G_RAYSLADDER = {
        "u_md5s": half_md5s, "committed_md5s": committed_md5s,
        "md5_match": md5_ok,
        "fresh_mutual_geometry": fresh_half_geom,
        "geometry_dev_vs_e200_committed": geom_dev,
        "max_geom_dev": max_geom_dev, "tol_geom": G_TEXTURE_TOL, "tier": tier_r,
        "pass": bool(tier_r != "FAIL"),
        "note": "THE RAY IDENTITY GATE (the s/2 rung): u0 must BE e193's "
                "committed R2_SIGN ray, u1/u2 e197's registered rays, u3/u4 "
                "e200's registered rays (md5 BIT), or the fresh mutual "
                "geometry must reproduce e200's committed lag matrix within "
                "1e-3 (e199/e200's disclosed TEXTURE tier)",
    }
    log(f"G_RAYSLADDER: tier {tier_r} (md5 {sum(md5_ok.values())}/"
        f"{len(md5_ok)}, geom dev max {max_geom_dev:.2e}): "
        + ("PASS" if G_RAYSLADDER["pass"] else "FAIL"))
    assert G_RAYSLADDER["pass"], "ladder ray identity gate FAILED"
    stub["gates"]["G_RAYSLADDER"] = G_RAYSLADDER
    write_partial("ladder complete + anchor gates passed (G_WALKE197, G_RAYSLADDER)")
    del us_half

    # =====================================================================
    # PHASE 5 — the comparison + the ladder adjudication + the verdict
    # =====================================================================
    log("=" * 78)
    log("PHASE 5 — THE ADJUDICATION (the frozen bars)")

    twin_c1 = twin_geom["lag_series"]["lag1"]
    matched = {
        "0": {"twin_cos1": twin_c1[0],
              "comparators": {"org1_e195_alive": ORG1_C1_ALIVE,
                              "MIRABEL_e198_alive": MIR_C1_01}},
        "1": {"twin_cos1": twin_c1[1] if len(twin_c1) > 1 else None,
              "comparators": {"MIRABEL_e198_alive": MIR_C1_12}},
    }
    for pi, entry in matched.items():
        if entry["twin_cos1"] is None:
            entry["dev"] = None
            entry["fact_deepening"] = False
            entry["match_within_bar"] = False
            continue
        entry["dev"] = {k: entry["twin_cos1"] - v
                        for k, v in entry["comparators"].items()}
        entry["fact_deepening"] = bool(all(d >= MATCH_BAR
                                           for d in entry["dev"].values()))
        entry["match_within_bar"] = bool(all(abs(d) < MATCH_BAR
                                             for d in entry["dev"].values()))
    twin_matches = all(e["match_within_bar"] for e in matched.values())
    twin_fact_deepening = all(e["fact_deepening"] for e in matched.values())
    # the anti-direction disclosure (twin DEEPER than the fact-carrying):
    twin_anti = {pi: {k: d for k, d in e["dev"].items() if d <= -MATCH_BAR}
                 for pi, e in matched.items() if e["dev"]}

    ladder_tbl = []
    for key, frac, s, wk, _us, geom_k, rung in rungs:
        ladder_tbl.append({
            "key": key, "fraction": frac, "s": s,
            "lag1_series": rung["lag1_series"],
            "cos1_t0": rung["lag1_series"][0] if rung["lag1_series"] else None,
            "core_t0": geom_k.get("core_t0"),
            "kill": (wk["stop"]["kind"], wk["stop"]["step"]),
            "n_alive_fronts": geom_k["n_fronts"],
            "period2_all_holds": bool(all(
                v["fingerprint_holds"] for v in geom_k["period2_fingerprint"]
                .values())),
        })
    order = [r for r in ladder_tbl]        # already ascending in s
    cores = [r["core_t0"] for r in order]
    c1s = [r["cos1_t0"] for r in order]
    cores_ok = all(a is not None for a in cores) and \
        all(cores[i] <= cores[i + 1] + 1e-12 for i in range(len(cores) - 1))
    c1s_ok = all(a is not None for a in c1s) and \
        all(c1s[i] >= c1s[i + 1] - 1e-12 for i in range(len(c1s) - 1))
    ladder_follows = bool(cores_ok and c1s_ok)
    anchor_diff = abs(order[-1]["cos1_t0"] - HALF_C1[0])

    ladder_prediction = {
        "null_prediction": "the flip zone |mu| <= h*s widens with s, so the "
                           "anti-phase core (and |cos1| at matched states "
                           "and steps) must be NON-DECREASING in s — the "
                           "cosines MORE negative at larger steps",
        "rungs_ascending_s": [r["key"] for r in order],
        "sizes_ascending_s": [r["s"] for r in order],
        "cores_t0": cores, "cores_nondecreasing": bool(cores_ok),
        "cos1_t0": c1s, "cos1_nonincreasing": bool(c1s_ok),
        "statistic_split": bool(cores_ok != c1s_ok),
        "ladder_follows_prediction": ladder_follows,
        "anchor": {"rung": "half", "fresh_cos1_t0": order[-1]["cos1_t0"],
                   "committed_e197": HALF_C1[0], "abs_diff": anchor_diff,
                   "tol": G_TEXTURE_TOL,
                   "anchor_holds": bool(anchor_diff < G_TEXTURE_TOL)},
        "out_of_domain_context": {
            "org2_full_step_natural_s": {"s": E193_STEP_L2,
                                         "cos1": DEAD_FULL_C1,
                                         "note": "OUT OF DOMAIN: 1.74x its "
                                                 "own 0.5252 static kill "
                                                 "edge — the derivation's "
                                                 "own exclusion; context "
                                                 "only"},
            "half_series_committed_e200": HALF_C1},
    }

    period2_ctx = {
        "twin": twin_geom["period2_fingerprint"],
        **{r["key"]: r["period2"] for _, _, _, _, _, _, r in rungs},
    }

    # ---- the frozen composite ------------------------------------------------
    bars = {
        "FACT_IN_THE_FRONT": {"fires": bool(twin_fact_deepening)},
        "SIGN_DESCENT_BOUNCES": {"fires": bool(
            (not twin_fact_deepening) and twin_matches and ladder_follows)},
        "GRADED": {"fires": bool(
            not twin_fact_deepening
            and not ((not twin_fact_deepening) and twin_matches
                     and ladder_follows))},
    }
    if SMOKE:
        verdict, clause = "SMOKE", "shakedown — nothing adjudicated"
    elif twin_fact_deepening:
        verdict = "FACT-IN-THE-FRONT"
        clause = (
            "the fact-free twin's lag-1 cosines sit >= 0.05 away from the "
            "fact-carrying twins' in the direction that makes the fact's "
            "fronts deeper — the front carries fact information; the "
            "rotation reading survives in restricted form. Numbers: pair 0 "
            f"twin {twin_c1[0]:+.5f} vs org1 {ORG1_C1_ALIVE:+.5f} "
            f"(Delta {matched['0']['dev']['org1_e195_alive']:+.5f}) / "
            f"MIRABEL {MIR_C1_01:+.5f} (Delta "
            f"{matched['0']['dev']['MIRABEL_e198_alive']:+.5f}); pair 1 "
            f"twin {twin_c1[1]:+.5f} vs MIRABEL {MIR_C1_12:+.5f} (Delta "
            f"{matched['1']['dev']['MIRABEL_e198_alive']:+.5f}) — the "
            "fact-carrying fronts are DEEPER by >= 0.05 at both matched "
            "indices. Period-2 fingerprint status: "
            + ("twin HOLDS" if all(v["fingerprint_holds"]
                                   for v in twin_geom["period2_fingerprint"]
                                   .values()) else
               "twin FAILS (flagged verbatim — the scratch's fuller draft "
               "made this a precondition; the dispatch's letter "
               "adjudicates)") + ".")
    elif twin_matches and ladder_follows:
        verdict = "SIGN-DESCENT-BOUNCES"
        clause = (
            "the twins match within 0.05 AND the ladder follows the null's "
            "step-size prediction — SIGN DESCENT BOUNCES, AS IT MUST; the "
            "alternation noun retires finally; the survivors "
            "(death-at-deepest-landing, the onset curves) stand alone. "
            f"Numbers: pair 0 twin {twin_c1[0]:+.5f} vs org1 "
            f"{ORG1_C1_ALIVE:+.5f} (Delta {matched['0']['dev']['org1_e195_alive']:+.5f}) "
            f"/ MIRABEL {MIR_C1_01:+.5f} (Delta "
            f"{matched['0']['dev']['MIRABEL_e198_alive']:+.5f}); pair 1 "
            f"twin {twin_c1[1]:+.5f} vs MIRABEL {MIR_C1_12:+.5f} (Delta "
            f"{matched['1']['dev']['MIRABEL_e198_alive']:+.5f}) — all "
            "|Delta| < 0.05 (FACT-FREE-ON-CURVE). Ladder: cores "
            + " -> ".join(f"{c:.5f}" for c in cores) + " (non-decreasing: "
            + ("yes" if cores_ok else "NO") + "); cos1(t=0) "
            + " -> ".join(f"{c:.5f}" for c in c1s) + " (non-increasing: "
            + ("yes" if c1s_ok else "NO") + "); the s/2 anchor reproduces "
            f"e197's committed -0.26318 to {anchor_diff:.1e}.")
    else:
        verdict = "GRADED"
        splits = []
        if not twin_matches and not twin_fact_deepening:
            for pi, e in matched.items():
                for k, d in (e["dev"] or {}).items():
                    if abs(d) >= MATCH_BAR:
                        splits.append(f"pair {pi} vs {k}: twin "
                                      f"{e['twin_cos1']:+.5f}, comparator "
                                      f"{e['comparators'][k]:+.5f}, Delta "
                                      f"{d:+.5f} ("
                                      + ("fact-deepening direction but not "
                                         "at both indices"
                                         if d >= MATCH_BAR else
                                         "ANTI direction — the twin is "
                                         "DEEPER") + ")")
        if not ladder_follows:
            splits.append("the ladder violates the null's step-size "
                          "prediction: cores "
                          + " -> ".join(f"{c:.5f}" for c in cores)
                          + " (non-decreasing: " + ("yes" if cores_ok else "NO")
                          + "); cos1(t=0) "
                          + " -> ".join(f"{c:.5f}" for c in c1s)
                          + " (non-increasing: " + ("yes" if c1s_ok else "NO")
                          + ")")
        clause = ("any split — the null is partial; reported verbatim: "
                  + "; ".join(splits) + ".")

    adjudication = {
        "bars": bars,
        "verdict": verdict,
        "clause": clause,
        "composite_order": "FACT-IN-THE-FRONT -> SIGN-DESCENT-BOUNCES -> "
                           "GRADED (frozen before compute)",
        "matched_comparison": matched,
        "twin_matches_within_bar": bool(twin_matches),
        "twin_fact_deepening_both_indices": bool(twin_fact_deepening),
        "anti_direction_deviations": {k: v for k, v in twin_anti.items() if v},
        "ladder_prediction": ladder_prediction,
        "ladder_table": ladder_tbl,
        "period2_fingerprint_context": period2_ctx,
        "constants": {"MATCH_BAR": MATCH_BAR, "SHUT_BAR": SHUT_BAR,
                      "PERIOD2_C3_BAR": PERIOD2_C3_BAR},
    }
    log(f"ADJUDICATION: {verdict}")
    log(f"  clause: {clause}")

    # ---- honesty -------------------------------------------------------------
    honesty = {
        "n": "n=1 per arm — one twin walk, one walk per rung; single stream "
             "seed 10902; nothing is a distribution",
        "twin_no_kill": "the fact-free walk has NO kill by construction "
                        "(its ruler baseline sits under the 0.27 bar because "
                        "there is no fact to dissolve) — all four "
                        "consecutive pairs are alive-by-construction, "
                        "flagged, and the battery column never adjudicates",
        "matched_set": "only pair indices 0 and 1 have committed ALIVE "
                       "fact-carrying comparators at matched class/step/"
                       "stream; pairs 2/3 ride as flagged context (org1 "
                       "post-death continuation; the half-step series at "
                       "HALF the step on a different organism)",
        "cross_organism": "org2's ladder and org1's twin differ in "
                          "architecture (873k 4L vs 2.74M 6L) and per-coord "
                          "RMS — the ladder is a WITHIN-organism step-size "
                          "test (its own in-domain null), never a "
                          "cross-organism magnitude comparison (the "
                          "derivation's own exclusion)",
        "h_unmeasured": "curvature h and the bath agreements a_k remain "
                        "unmeasured — the null's consistency regions, not "
                        "measurements; this cell tests its two sharpest "
                        "parameter-free predictions only",
        "openness": "nothing guaranteed — a different pre-install state, "
                    "stream, or rung set could read differently; that "
                    "openness is the point",
    }

    # ---- the plots ------------------------------------------------------------
    try:
        # Fig 1 — the twin vs the fact-carrying twins
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ts = list(range(len(twin_c1)))
        ax.plot(ts, twin_c1, "o-", color="black", lw=2.2, ms=8, zorder=5,
                label="FACT-FREE TWIN (pre-install e048_repro, s=1.6543, "
                      "seed 10902 — THIS CELL)")
        ax.plot([0], [ORG1_C1_ALIVE], "s", color="tab:blue", ms=11, zorder=5,
                label="org1 alive pair (committed e195)")
        ax.plot([1, 2, 3, 4], ORG1_POSTDEATH_C1, "s", mfc="none",
                color="tab:blue", ms=8, zorder=4,
                label="org1 post-death continuation (context, flagged)")
        ax.plot([0, 1], [MIR_C1_01, MIR_C1_12], "D", color="tab:orange",
                ms=9, zorder=5, label="MIRABEL alive pairs (committed e198)")
        ax.plot([0, 1, 2, 3], HALF_C1, "^--", color="tab:green", ms=7,
                lw=1.2, zorder=3,
                label="org2 HALF-step series (committed e200 — different "
                      "organism, half step; matched at the ladder instead)")
        lo = min(ORG1_C1_ALIVE, MIR_C1_01, MIR_C1_12)
        ax.fill_between([-0.25, 1.25], lo - MATCH_BAR, lo + MATCH_BAR,
                        color="tab:red", alpha=0.12, zorder=1,
                        label=f"the ±{MATCH_BAR} match band around the "
                              f"matched comparators")
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["pair 0\n(MATCHED: org1, MIRABEL)",
                            "pair 1\n(MATCHED: MIRABEL)",
                            "pair 2\n(unmatched — context)",
                            "pair 3\n(unmatched — context)", "pair 4"])
        ax.set_xlabel("consecutive-front pair index t  —  cos(u_t, u_{t+1})")
        ax.set_ylabel("lag-1 cosine (fp64)")
        ax.set_title("E202 — the fact-free twin vs the fact-carrying twins\n"
                     "(the SIGNFRONT-NULL falsifier: does the front carry "
                     "fact information?)"
                     + ("  [SMOKE]" if SMOKE else ""))
        ax.legend(loc="lower left", fontsize=8.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(rd / "e202_twin_vs_twins.png", dpi=130)
        plt.close(fig)

        # Fig 2 — the ladder vs the null's step-size prediction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.5))
        xs = [r["s"] for r in order]
        ax1.plot(xs, c1s, "o-", color="tab:purple", lw=2, ms=9,
                 label="fresh rungs (THIS CELL: s/8, s/4, s/2)")
        ax1.plot([E197_SUB_S], [HALF_C1[0]], "*", color="tab:red", ms=18,
                 zorder=6, label="e197 committed anchor (s/2): -0.26318")
        ax1.plot([E193_STEP_L2], [DEAD_FULL_C1], "x", color="gray", ms=11,
                 mew=2, label="org2 natural s (OUT OF DOMAIN: 1.74x kill "
                              "edge)")
        ax1.set_xscale("log")
        ax1.set_xlabel("per-step L2 s (log scale)")
        ax1.set_ylabel("cos(u0, u1) at t=0")
        ax1.set_title("the ladder: lag-1 depth vs step size")
        ax1.annotate("the null predicts\nMORE negative at larger s",
                     xy=(xs[-1], c1s[-1]), xytext=(0.30, min(c1s) - 0.05),
                     arrowprops=dict(arrowstyle="->", color="tab:purple"),
                     fontsize=9, color="tab:purple")
        ax1.axhline(0, color="gray", lw=0.8)
        ax1.legend(fontsize=8.5)
        ax1.grid(alpha=0.3)
        ax2.plot(xs, cores, "o-", color="tab:brown", lw=2, ms=9,
                 label="anti-phase core (cos2-cos1)/2 at t=0")
        ax2.set_xscale("log")
        ax2.set_xlabel("per-step L2 s (log scale)")
        ax2.set_ylabel("core(t=0) = (cos2 - cos1)/2")
        ax2.set_title("the null's registered statistic\n(non-decreasing in s "
                      "required)"
                      + ("  [SMOKE]" if SMOKE else ""))
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        fig.suptitle("E202 — the step ladder on the FACT-CARRYING org2 root "
                     "(in-domain for the first time)", fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "e202_ladder.png", dpi=130)
        plt.close(fig)
        plots_ok = True
    except Exception as e:      # plots never block the adjudication
        log(f"plot failure (disclosed): {e}")
        plots_ok = False

    # ---- the final COMPLETE write ---------------------------------------------
    stub.pop("phases_partial", None)
    stub.update({
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "question": ("E202 — THE SIGNFRONT-NULL CELL: does the fact-free "
                     "twin's front rotation differ from the fact-carrying "
                     "twins' (the front carries fact information), and does "
                     "the in-domain step ladder follow the overshoot null's "
                     "monotonicity prediction — or do both read as SIGN "
                     "DESCENT BOUNCES, AS IT MUST?"),
        "provenance": {
            "parents_loaded_committed": sorted(PARENTS.keys()),
            "fresh_compute": "the fact-free twin walk (5 sign steps, "
                             "pre-install e048_repro) + the ladder rungs "
                             "{s/8, s/4, s/2} (5-6 sign steps each, org2 "
                             "root) + battery reads — CPU-only, threads 4, "
                             "the e201 u3-burst class",
            "instruments": "lab/e201_rotation_census.py + lab/e200_deepening"
                           ".py (walk_sub) machinery VERBATIM; bars frozen "
                           "in the module docstring before compute",
        },
        "phase1_factfree_twin": twin_block,
        "phase_ladder": {r["key"]: r for _, _, _, _, _, _, r in rungs},
        "comparison": {
            "twin_lag1_series": twin_c1,
            "fact_carrying_committed": {
                "org1_alive": {"pair": [0, 1], "cos": ORG1_C1_ALIVE,
                               "source": "e195 ray_geometry (gated)"},
                "MIRABEL_alive": [{"pair": [0, 1], "cos": MIR_C1_01},
                                  {"pair": [1, 2], "cos": MIR_C1_12}],
                "org2_half_context": {"series": HALF_C1,
                                      "source": "e200 ray_geometry (gated)",
                                      "note": "different organism at HALF "
                                              "the step — matched at the "
                                              "ladder's s/2 rung, not here"},
                "org1_postdeath_context": {
                    "series": ORG1_POSTDEATH_C1, "source": "e194/e195",
                    "note": "POST-DEATH reads — flagged, never matched"},
            },
        },
        "adjudication": adjudication,
        "honesty": honesty,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1),
                   "twin_s": wt["seconds"],
                   "rung_s": {r["key"]: r["seconds"]
                              for _, _, _, _, _, _, r in rungs}},
        "config": {"smoke": SMOKE, "torch": torch.__version__,
                   "threads": torch.get_num_threads(),
                   "device": "cpu", "load_probe_pct": load_pct,
                   "plots_written": plots_ok},
    })
    save_json(rd / "metrics.json", E43.jsonable(stub))
    log(f"FINAL WRITE: {rd / 'metrics.json'} (verdict {verdict})")
    log(f"plots: {'written' if plots_ok else 'FAILED (disclosed)'}")


if __name__ == "__main__":
    main()
