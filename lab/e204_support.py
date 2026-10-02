"""E204 — THE SUPPORT MEASUREMENT (the survivor claim's missing leg).

WHY. T164's survivor clause — "death = the rotation's deepest landing on
the support" (the one perfect 4/4 rank-order, censored at the event) —
has never had its SUPPORT leg measured: the only support object on
record is a PROXY, the STATIC t=0 g-ray u_g, whose overlap with the
killing front is -0.030 (e200's committed cos_u4_ug — near-orthogonal;
T166's R61-critic stamp says so in as many words: "the support-proxy
overlap -0.030 — the support itself never measured"). THE QUESTION THIS
CELL OWNS: what does the deepest landing land ON? Measure, at every
state of e197/e200's half-step lineage (the only lineage with a full
front rank-order), the fact's LOCAL sensitivity direction — the gradient
of the fact readout (the g-12 battery, e194/e195's fact_grad convention)
AT THAT STATE — and ask: does the killing front's advantage (e200's
committed kill-ratio curve; the deepest landing = t4 = the killing step)
track alignment with the LOCAL fact-sensitivity?

BUILDS ON (directive 1): e200 (the half-step lineage's four fronts
u1..u4 and their committed kill-Ds/ratios: t1 3.1690, t2 0.6631, t3
1.2921, t4 0.3973 — t4 the deepest, the killing step; the walk journal
and its gates), e197 (the half-step walk itself + the registered u1/u2
rays + the kill at t=5), e193 (the root identity, STEP_L2, the R2_SIGN
ray), e157 (the dial), e194/e195 (THE FACT-GRADIENT CONVENTION, VERBATIM
in its arithmetic: F_obj = mean log p(Z) over the g-12 install battery,
one backward, the critic's sign convention "NEGATIVE cos(delta, grad) =
displacement aligned with the DEATH gradient"; and the dual-estimator
MATCHED-POINT lesson, T150: every alignment read states its evaluation
point), e185/e170 (the licensed seed-10902 stream + the anchor bank).
WHAT IS NEW: the fact's sensitivity directions s_t on the f2 half-step
lineage's states (theta_0..theta_5) — never measured on ANY f2 state —
the landing metric per t, and the tracking correlation against e200's
committed kill-ratio curve. No static profiles are re-run (e200's
committed D_kills ARE the kill-ratio curve, hard-bound + asserted).

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - LANDS-ON-SENSITIVITY: "fires if the landing metric correlates with
    the kill-ratio curve across t (Spearman >= 0.8 on the 4 points; the
    deepest landing t4 at the max) — the survivor claim gains its
    mechanism leg: the deepest landing is the most fact-erasing-aligned
    front."
  - LANDING-IS-GEOMETRIC: "fires if the landing metric is flat/
    uncorrelated — the deepest landing is NOT special in erasing
    alignment; the rank-order is geometry (front norm/subspace
    effects), not fact-directed; the survivor stays a coincidence-
    shaped object."
  - GRADED: "any partial — the table verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * states theta_t = the REBUILT half-step walk's endpoints (t=0..5),
    the walk gated row-by-row vs e197's committed journal (G_WALKE197,
    e200's gate VERBATIM) and the alive window re-anchored (G_ALIVE:
    t=1..4 alive, t=5 dead).
  * the fronts u_t (t=0..4) = the loaded/committed rays, md5-gated in
    THIS process from the gated walk: u0 vs e193's committed R2_SIGN,
    u1/u2 vs e197's registered rays, u3/u4 vs e200's registered rays
    (G_RAYS). The post-kill step-6 front is NEVER read (e200's
    convention, carried).
  * s_t = the LOCAL fact-sensitivity direction at theta_t: the fp32
    unit-normalized gradient of the fact readout — mean log p(Z) over
    the g-12 install-60 battery — recomputed AT theta_t (e194/e195's
    fact_grad VERBATIM in its arithmetic; small eval bursts: one 60-
    window forward+backward per state). MATCHED-POINT: u_t and s_t are
    read at the SAME state theta_t (T150's dual-estimator lesson; the
    increment at the point it began). Provenance disclosure: e194/
    e195's convention was minted on ORGANISM 1; no committed f2
    fact-gradient comparator exists — G_SENSDIR substitutes a finite-
    difference sign check (below).
  * THE SIGN IDENTITY (Rule 12; frozen): e200's front u_t = sign(g_t)
    points OPPOSITE the wash's landing displacement (the step is
    -s*.u_t; the static family is theta_D = root - D*u_t), and the
    fact-erasing ray is -s_t. By e194's own double-negation identity
    ("the step is -sign(g_t), the death ray is -u_g, so
    cos(sign(g_t), u_g) IS the step-vs-death overlap"), the landing
    displacement's alignment with the fact-ERASING ray is
    L_t := cos(-u_t, -s_t) = cos(u_t, s_t).
    THE DISPATCH'S LITERAL FORMULA "cos(u_t, -s_t)" equals -L_t under
    the readout-gradient reading of s_t, and equals +L_t under the
    dispatch's own "g-12 battery LOSS" reading (s = grad of the loss =
    -grad of log p) — the two readings differ by exactly the sign the
    identity resolves. THE ADJUDICATED ORIENTATION IS L_t = cos(u_t,
    s_t) (the parenthetical fixes it: "the deepest landing t4 AT THE
    MAX" of the ERASING alignment); BOTH columns ride verbatim in the
    table (the literal column is L's exact negation); the verdict is
    orientation-invariant because the bars are stated in ranks (t4 at
    the max; concordance with depth).
  * the kill-ratio curve = e200's COMMITTED onset_curve ratios (t1
    3.1690, t2 0.6631, t3 1.2921, t4 0.3973), hard-bound + asserted at
    load (G_E200CURVE); kill-DEPTH_t = -ratio_t (deeper landing =
    smaller ratio = larger depth). The 4-point correlation set is
    t in {1,2,3,4} (the bar freezes "on the 4 points"; t0's ratio is
    definitional 1.0, excluded — e199's MIRABEL precedent; t0's landing
    metric is reported as CONTEXT).
  * THE CORRELATION = Spearman(L_t, depth_t) over t in {1,2,3,4}
    (hand-ranked fp64; with n=4, rho in {0, +-0.2, ..., +-1}); the
    bar's "Spearman >= 0.8" is read in the CONCORDANT-with-depth
    orientation the parenthetical fixes (equivalently Spearman(L,
    ratio) <= -0.8; both raw numbers reported).
  * LANDS-ON-SENSITIVITY fires iff Spearman(L, depth) >= 0.8 AND
    argmax_t L_t = t4 (the deepest landing at the max).
  * LANDING-IS-GEOMETRIC fires iff Spearman(L, depth) <= 0 (flat or
    anti-concordant: the landing metric does not track the kill-ratio
    curve in the claim's direction).
  * GRADED fires otherwise (any partial; the table verbatim).
  * composite order frozen: LANDS-ON-SENSITIVITY -> LANDING-IS-
    GEOMETRIC -> GRADED (the first that fires is the verdict; the three
    are mutually exclusive by construction).
  * THE DEATH ROW (theta_5) is a landing-POINT read, CONTEXT ONLY,
    never in the correlation: L_death = cos(u_4, s_5) — the killing
    front read against the sensitivity AT ITS OWN LANDING POINT
    (theta_5 is where u_4's step killed the organism); the post-kill
    front (step 6) is never read (e200's convention).
  * CO-READ (never adjudicated): the same landing metric against the
    g-4 battery's sensitivity s4_t (the f2 lineage's PRIMARY ruler's
    geometry) — reported per row + its own Spearman as a robustness
    column, disclosed in deviations; the g-12 battery is the dispatch's
    frozen choice (e194/e195's convention) and is DISCLOSED WEAK at
    this root (g-12 reads 0.198 at D=0, under the 0.27 bar — e200's
    ruler disclosure, carried).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication):
e197/e200's gate set REUSED VERBATIM — G_NAMEFREE, G_SPLICE, G_BATTERY,
G_ANCHOR, G_ROOT, G_T0, G_S1CK, G_STREAM, G_DIRCK, G_SIGNRAY, the walk
rebuild gate G_WALKE197 (vs e197's committed journal, all 6 rows + stop
+ x-hashes), G_ALIVE (t1..4 alive, t5 dead), G_RAYS (u0..u4 identity;
u3/u4 vs E200's registered md5s), G_E200CURVE (the committed kill-ratio
curve + D_kills + ray mutual geometry hard-bound, asserted vs
runs/e200/metrics.json), AND THE NEW INSTRUMENT GATE G_SENSDIR: (i) the
convention stated (e194/e195 fact_grad verbatim; g-12 battery;
matched-point); (ii) THE FINITE-DIFFERENCE SIGN CHECK at the root
(HARD): the g-12 mean-log-p readout at theta_0 +- eps*s_0 (eps in
{0.02, 0.05}, tiny eval bursts) must satisfy read(+eps) > read(theta_0)
> read(-eps) strictly at BOTH eps — the direction must BE the fact's
sensitivity; the same probe at theta_4 reported (soft: a failure there
is a local-landscape finding at a weak readout, not an instrument
failure — the instrument is the same code path gated hard at the root);
(iii) every state's readout values stated in the table (mean p and mean
log p, both currencies).
IDENTITY TIERS (e199's disclosed convention, carried): BIT = md5/1e-9
where this process reproduces the parents' bits; else the TEXTURE tier
(walk rows < 1e-3, ray mutual geometry < 1e-3, D_kill < 1e-2); the
achieved tier STAMPED per gate, never silently weakened.

REGISTERED PREDICTION (frozen before compute): the survivor claim
predicts LANDS-ON-SENSITIVITY — L_t maximal at t4 (the killing front
the most erasing-aligned) and rank-concordant with depth. The competing
prior is T166's null-derivation residue: the alternation was the
algorithm's (sign-descent bounce), and the kill-rank-order may be pure
geometry (front norm/subspace effects) — predicting LANDING-IS-GEOMETRIC
(flat/uncorrelated, as the -0.030 static-proxy overlap already hints).
The honest third branch (GRADED) is the 4-point reality: at n=4,
Spearman moves in 0.2 quanta and one flipped pair drops it to 0.6 —
partial trackings are likely; the table verbatim either way.

WHAT EACH ARM GUARANTEES: NOTHING — this is an alignment read (the
class T157 taught the lab to distrust: alignment predicts nothing at
n=4 lineages); it cannot establish causality (no intervention here),
only whether the committed rank-order's geometry is fact-directed;
n=1 lineage, 4 points, one battery geometry; the openness is the point.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never claimed
— g1bS6 owns the GPU lane), torch threads 4, load-check at start,
sequential tiny eval bursts, PROGRESSIVE partial metrics.json writes
after every phase, n=1.

Outputs: runs/e204/{metrics.json, e204_support.png}. No NOTES/THINKING/
QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e204_support.py    (E204_SMOKE=1 shakedown)
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

torch.set_num_threads(4)                              # THE OWNER ENVELOPE'S CAP

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E204_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e204 is CPU-only by dispatch"

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
E197_METRICS = E43.REPO / "runs" / "e197" / "metrics.json"
E200_METRICS = E43.REPO / "runs" / "e200" / "metrics.json"

# ---- the family-2 net (e098 s4305 line: 4L/4H/128d/512-ctx, 873,472 params) ----
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- rulers (e193's, frozen before its compute; carried verbatim) -----------------
RULER_J = -4                     # PRIMARY: g-4 install-60 battery (max committed root read)
CO_RULERS_J = (12, -12, 0)       # novel-side analog / e192-verbatim / install dial
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries (gates)
SENS_J = -12                     # THE SENSITIVITY BATTERY (the dispatch's frozen choice:
                                 # e194/e195's fact_grad convention — the g-12 install battery)
CO_SENS_J = -4                   # the co-read sensitivity battery (the primary ruler's; never adjudicated)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193/e197 family-2)
LR_ADAMW = 1e-3                   # the t=0 recipe (e157/e193's wash)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
EXT_STEPS = 1                     # read-only steps past the sub-step walk's kill (e197's convention)
WALK_CAP = 8                      # step cap (e197's; the committed walk ran 6 steps)
WALK_D_TARGET = 5.0               # e193's target convention
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
DKILL_TOL = 1e-6                  # D_kill equality vs committed (BIT)
DKILL_TEXTURE_TOL = 1e-2          # D_kill equality vs committed (TEXTURE; e199's tier)
WALK_TEXTURE_TOL = 1e-3           # walk-row reproduction tier (e199's)
RAY_COS_TOL = 1e-3                # mutual-geometry cosine tier (e199's)
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
RMS_DENOM = 934.597239456655      # sqrt(873472) — organism 2's per-coordinate RMS currency
SPEARMAN_BAR = 0.8                # the frozen bar (>= 0.8 concordant with depth, n=4 points)
FD_EPS = (0.05, 0.02)             # the finite-difference probe sizes (L2 units; tiny eval bursts)
if SMOKE:
    FD_EPS = (0.05,)

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
E193_ASIGN_S1 = {                          # e193's committed a_sign step-1 row (context anchor)
    "ce_batch": 1.7748003005981445,
    "cum_disp": 0.9160122275352478,
    "step_disp": 0.916012167930603,
    "preclip_gnorm": 2.299248695373535,
    "gm": 0.006826397497206926,
    "frac_argmax_z": 0.0,
}
E193_UG_MD5 = "46e71718b2ee67f2cd87f32372652baf"
E193_USIGN_MD5 = "9395918d36425e65248dbd93b6fd50bc"

# ---- e197's committed lineage reads (THE WALK PARENT; hard-bound, drift-asserted) --
E197_SUB_S = 0.4582097828388214             # the committed pick: STEP_L2/2
E197_SUB_FRAC = 0.5
E197_PROBE_GM = 0.41916516423225403
E197_U1_MD5 = "3ce920e0dde4342133f466edc27c5138"
E197_U2_MD5 = "f333e7f8c48f0d59364bfa659c2a82f2"
E197_WALK_STOP = {"kind": "kill", "step": 5,
                  "gm_at_kill": 0.1648186892271042,
                  "D_kill_raw": 0.8303853930774336,
                  "D_kill": 0.838723200837794}
E197_GEOM = {                               # e197's committed ray mutual geometry (fp64)
    "cos_u0_u1": -0.26317857219197976,
    "cos_u0_u2": 0.1405511960995524,
    "cos_u1_u2": -0.3128523818598815,
}
E197_ALIVE_GMS = {1: 0.41916516423225403, 2: 0.8505910634994507,
                  3: 0.5280560255050659, 4: 0.6481676697731018,
                  5: 0.1648186892271042}     # t=5 = the DEATH step (its front never read)

# ---- e200's committed reads (THE CURVE PARENT; hard-bound, drift-asserted) ---------
E200_U3_MD5 = "d556419b910ef3a96863055fe7f18051"
E200_U4_MD5 = "9818a56ef541ca0625e93d2523d82785"
E200_DKILLS = {0: 0.5252331597771716, 1: 1.664473633460733,
               2: 0.34829356925017246, 3: 0.6786533732491854,
               4: 0.20865737305118012}       # the committed kill-Ds (t4 = the deepest)
E200_ONSET_RATIOS = {1: 3.169018563425966, 2: 0.6631218207889519,
                     3: 1.292099252714931, 4: 0.39726618391668633}
E200_WALK_DKILL = 0.8387231990565389         # e200's fresh walk kill (recompute anchor)
E200_T1T2_E197CROSS = {1: 3.1690184150400578, 2: 0.6631218620309564}
E200_GEOM = {                               # e200's committed fresh ray geometry (fp64)
    "cos_u0_u1": -0.26317857219199464, "cos_u0_u2": 0.14055119609955105,
    "cos_u0_u3": -0.037019047726428826, "cos_u0_u4": -0.02586469602319846,
    "cos_u1_u2": -0.3128523818599012, "cos_u1_u3": 0.23745856827065193,
    "cos_u1_u4": 0.01718571433556464, "cos_u2_u3": -0.3412904771804916,
    "cos_u2_u4": 0.14245581289065523, "cos_u3_u4": -0.352800001023403,
    "cos_u0_ug": 0.5471640532576972, "cos_u1_ug": -0.26860287517351,
    "cos_u2_ug": 0.14817080573542485, "cos_u3_ug": -0.07695150390808077,
    "cos_u4_ug": -0.03047445756990283,      # THE STATIC-PROXY OVERLAP AT THE KILLING FRONT
}
E200_U4_UG = -0.03047445756990283            # the survivor claim's proxy leg (T166's stamp)

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
    "LANDS_ON_SENSITIVITY": "LANDS-ON-SENSITIVITY: \"fires if the landing "
        "metric correlates with the kill-ratio curve across t (Spearman >= "
        "0.8 on the 4 points; the deepest landing t4 at the max) — the "
        "survivor claim gains its mechanism leg: the deepest landing is the "
        "most fact-erasing-aligned front.\"",
    "LANDING_IS_GEOMETRIC": "LANDING-IS-GEOMETRIC: \"fires if the landing "
        "metric is flat/uncorrelated — the deepest landing is NOT special in "
        "erasing alignment; the rank-order is geometry (front norm/subspace "
        "effects), not fact-directed; the survivor stays a coincidence-"
        "shaped object.\"",
    "GRADED": "GRADED: \"any partial — the table verbatim.\"",
    "operationalizations": "states theta_t = the rebuilt half-step walk's "
        "endpoints (t=0..5), gated row-by-row vs e197's committed journal "
        "(G_WALKE197, e200's gate VERBATIM) + G_ALIVE; fronts u_t = the "
        "loaded/committed rays (u0 vs e193's R2_SIGN, u1/u2 vs e197, u3/u4 "
        "vs e200 — md5-gated in THIS process, G_RAYS); the post-kill "
        "step-6 front NEVER read; s_t = the LOCAL fact-sensitivity: the "
        "fp32 unit-normalized gradient of mean log p(Z) over the g-12 "
        "install-60 battery AT theta_t (e194/e195's fact_grad VERBATIM; "
        "small eval bursts; MATCHED-POINT: u_t and s_t read at the SAME "
        "state — T150's dual-estimator lesson); THE SIGN IDENTITY (Rule 12, "
        "frozen): the step is -s*.u_t and the erasing ray is -s_t, so by "
        "e194's double-negation identity the landing displacement's "
        "alignment with the fact-ERASING ray is L_t = cos(u_t, s_t); the "
        "dispatch's literal formula cos(u_t, -s_t) is its exact negation "
        "under the readout-gradient reading (and equals L_t under the "
        "dispatch's own 'battery loss' reading) — BOTH columns ride "
        "verbatim, the adjudicated orientation is L_t (the parenthetical "
        "fixes it: t4 AT THE MAX of the erasing alignment), and the "
        "rank-based verdict is orientation-invariant; the kill-ratio curve "
        "= e200's COMMITTED onset_curve ratios (t1 3.1690, t2 0.6631, t3 "
        "1.2921, t4 0.3973; hard-bound + asserted, G_E200CURVE), "
        "kill-DEPTH_t = -ratio_t; the 4-point correlation set is t in "
        "{1,2,3,4} (t0 definitional-excluded per e199's MIRABEL precedent, "
        "reported as context); THE CORRELATION = Spearman(L_t, depth_t) "
        "(hand-ranked fp64), the bar's '>= 0.8' read CONCORDANT with depth "
        "(equivalently Spearman(L, ratio) <= -0.8; both reported); "
        "LANDS-ON-SENSITIVITY fires iff Spearman(L, depth) >= 0.8 AND "
        "argmax_t L_t = t4; LANDING-IS-GEOMETRIC fires iff Spearman(L, "
        "depth) <= 0 (flat or anti-concordant); GRADED fires otherwise; "
        "composite order frozen LANDS-ON-SENSITIVITY -> LANDING-IS-"
        "GEOMETRIC -> GRADED (mutually exclusive by construction); the "
        "death row (theta_5) is a landing-POINT read, CONTEXT ONLY: "
        "L_death = cos(u_4, s_5), never in the correlation; CO-READ "
        "(never adjudicated): the same metric against the g-4 battery's "
        "sensitivity (the primary ruler's geometry; the g-12 battery is "
        "the dispatch's frozen choice and is disclosed WEAK at this root: "
        "0.198 at D=0, under the 0.27 bar).",
    "registered_prediction": "the survivor claim predicts LANDS-ON-"
        "SENSITIVITY — L_t maximal at t4 (the killing front the most "
        "erasing-aligned) and rank-concordant with depth. The competing "
        "prior is T166's null-derivation residue: the alternation was the "
        "algorithm's (sign-descent bounce) and the kill-rank-order may be "
        "pure geometry — predicting LANDING-IS-GEOMETRIC (flat/"
        "uncorrelated, as the -0.030 static-proxy overlap already hints). "
        "The honest third branch (GRADED) is the 4-point reality: at n=4 "
        "Spearman moves in 0.2 quanta and one flipped pair drops it to "
        "0.6 — partial trackings are likely; the table verbatim either "
        "way. No bar shopping.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE SIGN-IDENTITY RESOLUTION (Rule 12, frozen before compute): the "
    "dispatch's literal landing formula cos(u_t, -s_t) and the intent "
    "(\"the front's alignment with the fact-ERASING direction\") differ by "
    "a sign under e200's front convention (the step is -s*.u_t); by "
    "e194's double-negation identity the erasing alignment is L_t = "
    "cos(u_t, s_t) (= cos(-u_t, -s_t)); the dispatch's own 'battery LOSS' "
    "wording yields the same L_t. BOTH columns reported verbatim; the "
    "adjudicated orientation is L_t; the verdict is orientation-invariant "
    "(the bars are stated in ranks).",
    "NO COMMITTED f2 FACT-GRADIENT COMPARATOR EXISTS: e194/e195's "
    "fact_grad convention was minted on ORGANISM 1 (the e131 line); no "
    "prior cell ever computed a fact gradient on an f2 state. G_SENSDIR "
    "substitutes the finite-difference sign check (hard at the root, "
    "reported at theta_4) + the convention verbatim + matched-point "
    "statements.",
    "THE g-12 WEAKNESS (carried from e200's ruler disclosure): the g-12 "
    "battery reads 0.198 at this root — UNDER the 0.27 bar at D=0. The "
    "sensitivity battery is the dispatch's frozen choice (e194/e195's "
    "convention); the g-4-battery co-read rides per row (never "
    "adjudicated) so the choice is auditable.",
    "THREADS 4 (the owner envelope's cap), where e197's committed chain "
    "ran threads 8: cross-thread/environment drift at ~1e-7 fp32 is "
    "possible, so every identity gate carries e199's disclosed TIER "
    "system (BIT md5/1e-9 or TEXTURE at e199's tolerances), the achieved "
    "tier STAMPED per gate.",
    "the walk is REBUILT (not loaded): e197 saved no walk checkpoints "
    "(e200's own convention — the rays and endpoints are bit-rebuildable "
    "from the licensed stream); the rebuilt journal is gated row-by-row "
    "vs e197's committed journal (G_WALKE197) before any state or front "
    "is believed.",
    "NO STATIC PROFILES ARE RE-RUN: e200's committed D_kills/ratios ARE "
    "the kill-ratio curve (hard-bound + asserted at load, G_E200CURVE); "
    "this cell adds only the sensitivity directions and the landing "
    "metrics (eval-only, minutes).",
    "the CPU load-check is recorded, not gating (the envelope's 'check "
    "load first'): this cell is a 4-thread CPU eval burst of a few "
    "minutes; the GPU is never claimed (g1bS6 owns the GPU lane).",
    "Smoke mode trims: FD eps set to {0.05} (verdict stamped SMOKE; "
    "nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e200's instruments VERBATIM (whose own provenance is
# e197/e196/e193/e195 — the e176n lineage), PLUS e194/e195's fact_grad
# VERBATIM (the fact-sensitivity instrument this cell exists to carry
# onto the f2 line). Copied rather than imported to own the device
# policy and the arithmetic.

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
def fact_readout(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e194/e195's fact objective, VALUE ONLY: mean log p(Z) over the
    battery (the fact_grad objective; the readout currency of G_SENSDIR)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        tot += float(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
        n += ids[i:i + bs].shape[0]
    return tot / max(n, 1)


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE SENSITIVITY INSTRUMENT (e194/e195's fact_grad VERBATIM in its
    arithmetic): gradient of the fact battery's mean log p(Z) readout at
    the net's CURRENT weights. Sign convention (the critic's, carried):
    -s = the fact-ERASING ray; the step displacement is -s*.u_t (e200's
    front convention) — the landing metric is resolved in the module
    docstring's SIGN IDENTITY. Consumes no RNG; tiny eval burst."""
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


def _ranks(xs):
    """Average ranks (ties-safe) for the hand-rolled Spearman."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys) -> float:
    """Spearman rho on equal-length lists (fp64; Pearson on average ranks)."""
    assert len(xs) == len(ys) and len(xs) >= 2
    rx, ry = np.array(_ranks(xs), dtype=np.float64), \
        np.array(_ranks(ys), dtype=np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


# ------------------------------------------------------------------ the walk

def walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids, coruler_ids,
             zid, theta0, step_l2, root_gm, step_cap=WALK_CAP,
             d_target=WALK_D_TARGET, extend_past_kill=EXT_STEPS, tag="sub"):
    """e197's walk_sub VERBATIM (e200's copy): every step refreshes the
    front (k=1, post-clip); every-step primary battery + co-rulers;
    densified kill bracket at the killing step; stashes g_fronts +
    endpoints (the ray factory + the state ladder)."""
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
        endpoints[step] = cur.clone()              # stash (the state ladder)
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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e204_smoke" if SMOKE else "e204")
    log(f"E204 THE SUPPORT MEASUREMENT (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()} — the owner "
        f"envelope's cap), load-check recorded, tiny sequential eval bursts, "
        f"progressive writes, n=1, organism 2 of 2, stream seed {FREEZE_SEED}")
    load_note = {"cpu_count": os.cpu_count(),
                 "torch_threads": torch.get_num_threads(),
                 "note": "the envelope's load-check: a 4-thread CPU eval "
                         "burst of a few minutes; the GPU lane (g1bS6) is "
                         "never claimed"}
    log(f"load-check: {load_note}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for name, p in (("e157", E157_METRICS), ("e193", E193_METRICS),
                    ("e197", E197_METRICS), ("e200", E200_METRICS)):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    e157m = json.loads(E157_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    e197m = json.loads(E197_METRICS.read_text(encoding="utf-8"))
    e200m = json.loads(E200_METRICS.read_text(encoding="utf-8"))
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
    # e197 (THE WALK PARENT)
    e197_walk = e197m["phase1_alive_walk"]
    e197_journal = e197_walk["journal"]
    assert [r["step"] for r in e197_journal] == [1, 2, 3, 4, 5, 6]
    for t, gm in E197_ALIVE_GMS.items():
        assert abs(e197_journal[t - 1]["gm"] - gm) < 1e-12, f"e197 t{t} gm drift"
    assert e197_walk["stop"]["kind"] == "kill" \
        and e197_walk["stop"]["step"] == E197_WALK_STOP["step"] \
        and abs(e197_walk["stop"]["D_kill"] - E197_WALK_STOP["D_kill"]) < 1e-12
    assert e197m["cell"]["rays"][1]["u_md5"] == E197_U1_MD5
    assert e197m["cell"]["rays"][2]["u_md5"] == E197_U2_MD5
    for k, v in E197_GEOM.items():
        assert abs(e197m["phase1b_rays"]["ray_geometry"][k] - v) < 1e-12
    e197_xh = e197_walk["x_hashes"]
    # e200 (THE CURVE PARENT)
    oc = e200m["onset_curve"]["rows"]
    assert [r["t"] for r in oc] == [0, 1, 2, 3, 4, 5]
    for t, v in E200_DKILLS.items():
        assert abs(next(r for r in oc if r["t"] == t)["D_kill"] - v) < 1e-12, \
            f"e200 t{t} D_kill drift"
    for t, v in E200_ONSET_RATIOS.items():
        assert abs(next(r for r in oc if r["t"] == t)["ratio"] - v) < 1e-12, \
            f"e200 t{t} ratio drift"
    assert abs(next(r for r in oc if r["t"] == 5)["walk_D_kill"]
               - E200_WALK_DKILL) < 1e-12, "e200 walk kill drift"
    e200_rays = {r["key"]: r["u_md5"] for r in e200m["cell"]["rays"]}
    assert e200_rays["u3"] == E200_U3_MD5 and e200_rays["u4"] == E200_U4_MD5
    e200_geom_c = e200m["phase1b_rays"]["ray_geometry"]
    for k, v in E200_GEOM.items():
        assert abs(e200_geom_c[k] - v) < 1e-12, f"e200 geometry {k} drift"
    led_gm = {r["t"]: r["ruler_read"] for r in e200m["alive_ledger"]}
    for t in (1, 2, 3, 4, 5):
        # e200's fresh walk reads differ from e197's committed values at the
        # disclosed ~1e-7 fp-texture tier — bind at that tier, not bit
        assert abs(led_gm[t] - E197_ALIVE_GMS[t]) < 1e-6, \
            f"e200 alive ledger t{t} drift vs e197"
    assert e200m["adjudication"]["verdict"] == "GRADED"
    log("parents: e157 (the dial + wash s1 anchors), e193 (the organism: "
        "STEP_L2, root identity), e197 (THE WALK PARENT: the half-step "
        "journal t=1..6, the kill at t=5, rays u1/u2), e200 (THE CURVE "
        "PARENT: the committed onset ratios t1..t4 = the kill-ratio curve, "
        "rays u3/u4, the fresh ray geometry incl. cos_u4_ug "
        f"{E200_U4_UG:+.4f} — the static-proxy overlap) — loaded COMMITTED, "
        "never rerun")

    # ---------------- protocol rebuild (e193/e197/e200 verbatim) ---------------
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
                "— e193/e197/e200's gate verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]
    coruler_ids = {j: bat_ids[j] for j in CO_RULERS_J}
    sens_ids = bat_ids[SENS_J]           # THE SENSITIVITY BATTERY (g-12)
    co_sens_ids = bat_ids[CO_SENS_J]     # the co-read battery (g-4)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # the neutral stream (e170 VERBATIM via e185/e193/e197/e200)
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
              "note": "e193/e197/e200's G_ROOT VERBATIM: the f2 "
                      "consolidated root's dial reproduces e157's committed "
                      "cells BIT-TIGHT + the flat md5 matches e193's "
                      "committed root identity"}
    log(f"G_ROOT (vs e157 committed dial, 8 cells): max|diff| {rmax:.2e}, "
        f"flat md5 "
        f"{'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("family-2 root gate FAILED vs e157 committed dial")

    # ---------------- G_T0 / G_S1CK (e193's t=0 gates, verbatim) --------------
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
        "note": "e193/e197/e200's G_T0 VERBATIM: the step-1 batch md5 vs "
                "e185's stored hash; the forward CE vs e157's committed "
                "wash step-1 corpus CE; the fresh AdamW step's L2 vs e193's "
                "committed MEASURED step L2",
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

    G_S1CK = {
        "cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
        "tol_cos": 0.999, "s1_meta": s1_meta,
        "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
        "note": "e193/e197/e200's G_S1CK VERBATIM: the fresh CPU AdamW step "
                "vs the committed cuda-trained e157_f2_neutral_s1 "
                "checkpoint's displacement — fp64 cosine > 0.999 + relative "
                "L2 < 5%",
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
                "note": "e193/e197/e200's G_STREAM VERBATIM: the seed-10902 "
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
                "the disclosed TEXTURE tier (e199's G_DIRCK precedent)",
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

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e204_support",
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
    write_partial("standard cell gates passed (e193/e197/e200's, verbatim)")
    log("standard cell gates passed; partial metrics written")

    # =====================================================================
    # PHASE 1 — THE HALF-STEP WALK REBUILT (the state ladder + ray factory)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE 1 — THE HALF-STEP WALK REBUILT: e197's intervention VERBATIM "
        f"(k=1 sign machinery, size s* {E197_SUB_S:.10f} = STEP_L2/"
        f"{round(1 / E197_SUB_FRAC)}) — the state ladder theta_0..theta_5 + "
        f"the fronts; journal gated row-by-row vs e197's committed walk")
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
        "note": "THE PARENT-WALK PROVENANCE GATE (Rule 12; e200's VERBATIM): "
                "the rebuilt half-step walk must reproduce e197's committed "
                "journal (all 6 rows incl. co-rulers and the post-kill "
                "step-6 row), its x-hashes, its kill at t=5 and its "
                "densified bracket — bit-class, or the disclosed TEXTURE "
                "tier; the state ladder and fronts are not believed until "
                "this passes",
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
        raise RuntimeError("G_WALKE197 FAILED — abort before any state is "
                           "believed")
    stub["gates"]["G_WALKE197"] = G_WALKE197
    stub["phases_partial"]["1_walk_rebuild"] = E43.jsonable({
        "journal": wsub["traj"], "stop": wsub["stop"],
        "x_hashes": wsub["x_hashes"], "max_l2_dev": wsub["max_l2_dev"],
        "sub_s": E197_SUB_S, "sub_fraction": E197_SUB_FRAC,
        "disclosure": "e197's half-step walk rebuilt VERBATIM (the state "
                      "ladder theta_0..theta_5 + the ray factory); the "
                      "post-kill step-6 gradient is computed for the journal "
                      "gate only and NEVER read as a ray or a front",
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
        "pass": bool(alive_flags.get(1) and alive_flags.get(2)
                     and alive_flags.get(3) and alive_flags.get(4)
                     and wsub["stop"]["step"] == 5),
        "note": "THE ALIVE WINDOW RE-ANCHORED: t=1..t=4 ALIVE, t=5 DEAD — "
                "matching e197/e200's committed flags; the fronts are read "
                "at t=0..4; the sensitivities at t=0..5 (the death state's "
                "sensitivity is a landing-POINT read, context only)",
    }
    log(f"G_ALIVE: alive steps {G_ALIVE['alive_steps']}, death at t"
        f"{G_ALIVE['dead_step']} (e197/e200: t1..t4 alive, t5 dead): "
        + ("PASS" if G_ALIVE["pass"] else "FAIL"))
    if not G_ALIVE["pass"]:
        stub["gates"]["G_ALIVE"] = G_ALIVE
        write_partial("CONTROL FAILURE — the alive window did not reproduce")
        raise RuntimeError("G_ALIVE FAILED — abort (the alive window is this "
                           "cell's substrate)")
    stub["gates"]["G_ALIVE"] = G_ALIVE

    # ---- the rays: u1..u4 from the rebuilt walk's refresh gradients
    g_fronts = wsub["g_fronts"]
    assert cos64(torch.sign(g_fronts[0]), u0) > 1 - 1e-9, "t=0 front != u0"

    def make_ray(g):
        s = torch.sign(g)
        return (s / torch.norm(s)).clone()

    u1, u2 = make_ray(g_fronts[1]), make_ray(g_fronts[2])
    u3, u4 = make_ray(g_fronts[3]), make_ray(g_fronts[4])
    u_md5s = {0: u0_md5}
    for t, u in ((1, u1), (2, u2), (3, u3), (4, u4)):
        u_md5s[t] = hashlib.md5(u.numpy().tobytes()).hexdigest()
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
        "committed_e200": E200_GEOM,
        "note": "the rays' mutual geometry (fp64 cosines); e200's committed "
                "fresh geometry rides for the identity gate (cos_u4_ug "
                f"{E200_U4_UG:+.6f} = THE STATIC-PROXY OVERLAP AT THE "
                "KILLING FRONT — the survivor claim's proxy leg); iso floor "
                "for scale",
    }
    ray_geometry["geometry_dev_vs_e200"] = {
        k: abs(ray_geometry[k] - E200_GEOM[k]) for k in E200_GEOM}
    md5_e200 = {0: E193_USIGN_MD5, 1: E197_U1_MD5, 2: E197_U2_MD5,
                3: E200_U3_MD5, 4: E200_U4_MD5}
    md5_all_match = all(u_md5s[t] == md5_e200[t] for t in range(5))
    geom_dev_ok = all(v < RAY_COS_TOL
                      for v in ray_geometry["geometry_dev_vs_e200"].values())
    G_RAYS = {
        "u_md5s": u_md5s,
        "committed_md5s": md5_e200,
        "md5_matches": {t: bool(u_md5s[t] == md5_e200[t])
                        for t in range(5)},
        "geometry_max_abs_dev_vs_e200":
            max(ray_geometry["geometry_dev_vs_e200"].values()),
        "tol_geom": RAY_COS_TOL,
        "tier": ("BIT" if md5_all_match else
                 ("TEXTURE" if geom_dev_ok else "FAIL")),
        "pass": bool(md5_all_match or geom_dev_ok),
        "note": "THE FRONT IDENTITY GATE with the disclosed TEXTURE tier: "
                "ALL FIVE loaded fronts must BE the committed rays (md5 "
                "BIT: u0 vs e193's R2_SIGN, u1/u2 vs e197's registered "
                "rays, u3/u4 vs E200's registered rays), or (the cross-"
                "thread/environment fallback, e199's precedent) the fresh "
                "mutual geometry must reproduce e200's committed fp64 "
                "cosines within 1e-3; u_t is read at theta_t, the ALIVE "
                "states (u4 = the killing step's own direction, read at "
                "the last alive state theta_4); the post-kill front "
                "(step 6) is NEVER read",
    }
    log(f"G_RAYS: u0..u4 md5 "
        + ("all match" if md5_all_match else
           f"{sum(u_md5s[t] == md5_e200[t] for t in range(5))}/5")
        + f", geometry max|dev| vs e200 "
        f"{G_RAYS['geometry_max_abs_dev_vs_e200']:.2e} (tier "
        f"{G_RAYS['tier']}): "
        + ("PASS" if G_RAYS["pass"] else "FAIL"))
    if not G_RAYS["pass"]:
        stub["gates"]["G_RAYS"] = G_RAYS
        write_partial("CONTROL FAILURE — the front identity gate failed")
        raise RuntimeError("G_RAYS FAILED — abort")
    stub["gates"]["G_RAYS"] = G_RAYS

    # ---- G_E200CURVE: the kill-ratio curve hard-bound + asserted
    G_E200CURVE = {
        "source": "runs/e200/metrics.json onset_curve.rows (COMMITTED; "
                  "asserted at load, never rerun here)",
        "D_kills": E200_DKILLS, "ratios": E200_ONSET_RATIOS,
        "u0_edge": E200_DKILLS[0], "walk_D_kill": E200_WALK_DKILL,
        "deepest_t": 4,
        "e197_crosscheck_t1_t2": E200_T1T2_E197CROSS,
        "verdict_parent": e200m["adjudication"]["verdict"],
        "pass": bool(all(abs(E200_ONSET_RATIOS[t]
                             - E200_DKILLS[t] / E200_DKILLS[0]) < 1e-9
                         for t in (1, 2, 3, 4))
                     and min(E200_ONSET_RATIOS,
                             key=lambda k: E200_ONSET_RATIOS[k]) == 4),
        "note": "THE CURVE GATE: e200's committed ratios are internally "
                "consistent with their own D_kills and their argmin is "
                "t=4 (the deepest landing = the killing step — T164's "
                "4/4 rank-order, hard-bound before this cell computes "
                "anything)",
    }
    log(f"G_E200CURVE: committed kill-ratio curve t1..t4 = "
        + ", ".join(f"{E200_ONSET_RATIOS[t]:.4f}" for t in (1, 2, 3, 4))
        + f" (argmin t4 = the deepest landing; u0 edge {E200_DKILLS[0]:.4f}):"
        + ("PASS" if G_E200CURVE["pass"] else "FAIL"))
    if not G_E200CURVE["pass"]:
        raise RuntimeError("G_E200CURVE FAILED — the committed curve drifted")
    stub["gates"]["G_E200CURVE"] = G_E200CURVE
    stub["phases_partial"]["1b_fronts"] = E43.jsonable({
        "rays": [{"key": f"u{t}", "t": t, "u_md5": u_md5s[t],
                  "md5_match_committed": bool(u_md5s[t] == md5_e200[t])}
                 for t in range(5)],
        "ray_geometry": ray_geometry})
    write_partial("phase 1b complete (the fronts u0..u4 identity-gated)")
    log("WHAT THESE ARMS GUARANTEE: NOTHING — the sensitivities are alignment "
        "objects (the class T157 taught the lab to distrust); they cannot "
        "establish causality, only whether the committed rank-order's "
        "geometry is fact-directed; that openness is the point.")

    # =====================================================================
    # PHASE S — THE SENSITIVITY LADDER (s_t at theta_0..theta_5) + G_SENSDIR
    # =====================================================================
    log("=" * 78)
    log(f"PHASE S — THE FACT'S SENSITIVITY LADDER: s_t = grad mean log p(Z) "
        f"on the {f'g{SENS_J:+d}'} install-60 battery AT theta_t "
        f"(e194/e195's fact_grad VERBATIM; matched-point), t=0..5 — six "
        f"tiny eval bursts")
    evs = copy.deepcopy(net0)
    states = {0: theta0}
    for t in (1, 2, 3, 4, 5):
        states[t] = wsub["endpoints"][t]
    sens = {}
    for t in range(0, 6):
        load_flat(evs, states[t])
        evs.eval()
        read_mlp = fact_readout(evs, sens_ids, zid)
        read_mp = battery_cell(evs, sens_ids, zid)["mean_pz"]
        g_s = fact_grad(evs, sens_ids, zid)
        s_t = (g_s / torch.norm(g_s)).clone()
        g_s4 = fact_grad(evs, co_sens_ids, zid)
        s4_t = (g_s4 / torch.norm(g_s4)).clone()
        sens[t] = {"s": s_t, "s4": s4_t, "read_meanlogp": read_mlp,
                   "read_meanp": read_mp,
                   "g_norm": float(torch.norm(g_s)),
                   "cos_s_s4": cos64(s_t, s4_t)}
        log(f"  theta_{t}: g-12 meanp {read_mp:.6f} meanlogp {read_mlp:+.4f} "
            f"|grad| {sens[t]['g_norm']:.4f} cos(s_g12, s_g4) "
            f"{sens[t]['cos_s_s4']:+.4f}")
    # the sensitivity ladder's own geometry (context: the support's rotation
    # + the static proxy's adequacy)
    sens_geom = {}
    for a in range(0, 6):
        for b in range(a + 1, 6):
            sens_geom[f"cos_s{a}_s{b}"] = cos64(sens[a]["s"], sens[b]["s"])
    sens_geom.update({f"cos_s{t}_ug": cos64(sens[t]["s"], u_g)
                      for t in range(0, 6)})
    sens_geom["iso_floor"] = 1.0 / (F2_PARAMS ** 0.5)
    log("  sensitivity geometry: "
        + " ".join(f"s{a}-s{b} {sens_geom[f'cos_s{a}_s{b}']:+.3f}"
                   for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
        + " | s_t-vs-static-proxy(u_g): "
        + " ".join(f"{sens_geom[f'cos_s{t}_ug']:+.3f}" for t in range(6)))

    # G_SENSDIR: the finite-difference sign check (the instrument gate)
    def fd_probe(t):
        """read(theta_t + eps*s_t) vs read(theta_t) vs read(theta_t -
        eps*s_t) at each FD eps: +eps*s must RAISE the readout (s points
        along the fact's strengthening direction)."""
        out = {}
        for eps in FD_EPS:
            vals = {}
            for tag, vec in (("plus", states[t] + eps * sens[t]["s"]),
                             ("zero", states[t]),
                             ("minus", states[t] - eps * sens[t]["s"])):
                load_flat(evs, vec)
                evs.eval()
                vals[tag] = fact_readout(evs, sens_ids, zid)
            out[f"eps_{eps}"] = {"plus": vals["plus"], "zero": vals["zero"],
                                 "minus": vals["minus"],
                                 "raises": bool(vals["plus"] > vals["zero"]
                                                > vals["minus"]),
                                 "d_plus": vals["plus"] - vals["zero"],
                                 "d_minus": vals["minus"] - vals["zero"]}
        load_flat(evs, states[t])
        return out

    fd_root = fd_probe(0)
    fd_t4 = fd_probe(4)
    G_SENSDIR = {
        "convention": "e194/e195's fact_grad VERBATIM (mean log p(Z) over "
                      "the g-12 install-60 battery, one backward); s_t "
                      "fp32-unit-normalized; MATCHED-POINT: s_t computed AT "
                      "theta_t, the SAME state the front u_t is read at "
                      "(T150's dual-estimator lesson)",
        "fd_root_HARD": fd_root,
        "fd_theta4_reported": fd_t4,
        "no_committed_f2_comparator": "e194/e195 minted the convention on "
                                      "ORGANISM 1 (the e131 line); this is "
                                      "the FIRST fact gradient on an f2 "
                                      "state — the FD probe substitutes for "
                                      "a committed comparator",
        "pass": bool(all(v["raises"] for v in fd_root.values())),
        "note": "THE INSTRUMENT GATE: at the root, moving +eps*s_0 must "
                "RAISE the g-12 mean-log-p readout and -eps*s_0 LOWER it, "
                "strictly, at every probe eps (the direction must BE the "
                "fact's sensitivity); the same probe at theta_4 (the last "
                "alive state) is REPORTED, not gating — a failure there is "
                "a local-landscape finding at a weak readout, not an "
                "instrument failure",
    }
    log(f"G_SENSDIR: FD sign check at theta_0 "
        + ("PASS (raises at every eps)" if G_SENSDIR["pass"]
           else "FAIL")
        + f"; at theta_4 "
        + ("raises at every eps" if all(v["raises"] for v in fd_t4.values())
           else "REPORTED: not raising at some eps (context)"))
    stub["gates"]["G_SENSDIR"] = G_SENSDIR
    if not G_SENSDIR["pass"]:
        write_partial("CONTROL FAILURE — the sensitivity instrument failed "
                      "its FD sign check at the root")
        raise RuntimeError("G_SENSDIR FAILED — the sensitivity direction is "
                           "not the fact's; abort before any landing read")
    stub["phases_partial"]["S_sensitivity_ladder"] = E43.jsonable({
        "per_t": {t: {k: v for k, v in sens[t].items() if k not in ("s", "s4")}
                  for t in range(0, 6)},
        "sensitivity_geometry": sens_geom,
        "fd_theta4": fd_t4,
        "disclosure": "the fact's LOCAL sensitivity directions on the f2 "
                      "half-step lineage — the FIRST such read on any f2 "
                      "state; the support object T164/T166's survivor claim "
                      "always lacked",
    })
    write_partial("phase S complete (the sensitivity ladder + G_SENSDIR)")

    # =====================================================================
    # PHASE L — THE LANDING TABLE + THE CORRELATION + ADJUDICATION
    # =====================================================================
    log("=" * 78)
    log("PHASE L — THE LANDING METRIC vs THE KILL-RATIO CURVE (frozen "
        "clauses; composite LANDS-ON-SENSITIVITY -> LANDING-IS-GEOMETRIC -> "
        "GRADED; no shopping)")
    rays = {0: u0, 1: u1, 2: u2, 3: u3, 4: u4}
    table = []
    for t in range(0, 5):
        L = cos64(rays[t], sens[t]["s"])            # the adjudicated metric
        L_lit = cos64(rays[t], -sens[t]["s"])       # the literal column (= -L)
        L_co = cos64(rays[t], sens[t]["s4"])        # the g-4 co-read
        row = {"t": t, "state": "alive" if t <= 4 else "death",
               "front": f"u{t}", "front_md5": u_md5s[t],
               "theta": f"theta_{t}",
               "ruler_read_g4": next(r for r in wsub["traj"]
                                     if r["step"] == t)["gm"]
               if t >= 1 else root_cells[f"g{RULER_J:+d}"],
               "sens_read_meanp_g12": sens[t]["read_meanp"],
               "sens_read_meanlogp_g12": sens[t]["read_meanlogp"],
               "grad_norm_g12": sens[t]["g_norm"],
               "L_eras": L, "L_literal_dispatch": L_lit,
               "co_L_eras_g4battery": L_co,
               "kill_ratio_e200_committed": (E200_ONSET_RATIOS[t]
                                             if t in E200_ONSET_RATIOS
                                             else (1.0 if t == 0 else None)),
               "role": ("CONTEXT (t=0: the static ray; ratio definitional "
                        "1.0, excluded from the 4-point set — e199's "
                        "MIRABEL precedent)" if t == 0 else
                        "CORRELATION SET (alive; one of the 4 points)")}
        table.append(row)
    # the death row: the killing front read against the sensitivity AT ITS
    # OWN LANDING POINT (theta_5) — context only, never in the correlation
    L_death = cos64(u4, sens[5]["s"])
    death_row = {"t": 5, "state": "death", "front": "u4",
                 "front_md5": u_md5s[4],
                 "theta": "theta_5 (the killing step's landing point)",
                 "ruler_read_g4": next(r for r in wsub["traj"]
                                       if r["step"] == 5)["gm"],
                 "sens_read_meanp_g12": sens[5]["read_meanp"],
                 "sens_read_meanlogp_g12": sens[5]["read_meanlogp"],
                 "grad_norm_g12": sens[5]["g_norm"],
                 "L_eras": L_death,
                 "L_literal_dispatch": cos64(u4, -sens[5]["s"]),
                 "co_L_eras_g4battery": cos64(u4, sens[5]["s4"]),
                 "kill_ratio_e200_committed": None,
                 "role": ("DEATH LANDING-POINT READ (context only, never in "
                          "the correlation): the killing front vs the "
                          "sensitivity AT ITS OWN LANDING POINT theta_5; "
                          "the post-kill front (step 6) NEVER read")}
    table.append(death_row)
    for r in table:
        log(f"  t{r['t']} {r['state']:5s} {r['front']}: L_eras "
            f"{r['L_eras']:+.6f} (literal {r['L_literal_dispatch']:+.6f}) "
            f"[g-4 co-read {r['co_L_eras_g4battery']:+.6f}]  g-12 read "
            f"{r['sens_read_meanp_g12']:.4f}  ratio "
            + (f"{r['kill_ratio_e200_committed']:.4f}"
               if r["kill_ratio_e200_committed"] is not None else "--"))

    # ---- the frozen correlation + adjudication
    ts = [1, 2, 3, 4]
    Ls = [next(r for r in table if r["t"] == t)["L_eras"] for t in ts]
    Ls_co = [next(r for r in table if r["t"] == t)["co_L_eras_g4battery"]
             for t in ts]
    ratios = [E200_ONSET_RATIOS[t] for t in ts]
    depths = [-r for r in ratios]
    rho_depth = spearman(Ls, depths)          # the bar's orientation
    rho_ratio = spearman(Ls, ratios)          # the raw curve (=-rho_depth)
    rho_co_depth = spearman(Ls_co, depths)    # the co-read (never adjudicated)
    argmax_t = ts[int(np.argmax(Ls))]
    t4_at_max = bool(argmax_t == 4)
    fires_lands = bool(rho_depth >= SPEARMAN_BAR and t4_at_max)
    fires_geometric = bool(rho_depth <= 0.0)
    fires_graded = not (fires_lands or fires_geometric)

    committed_crosscheck = {
        "note": "the same correlation on the e197-committed t1/t2 ratios "
                "(e200's crosscheck pair) — a robustness read, never a bar "
                "choice; t3/t4 exist only in e200's fresh set",
        "t1_t2_pairs_landing": [Ls[0], Ls[1]],
        "t1_t2_ratios_e197": [E200_T1T2_E197CROSS[1], E200_T1T2_E197CROSS[2]],
        "spearman_2pt_t1t2_e197": spearman(
            Ls[:2], [-E200_T1T2_E197CROSS[1], -E200_T1T2_E197CROSS[2]]),
    }

    fmt = lambda v: f"{v:+.6f}"
    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "LANDS_ON_SENSITIVITY": {
                "fires": fires_lands,
                "detail": {"spearman_L_depth": rho_depth,
                           "spearman_L_ratio": rho_ratio,
                           "bar": SPEARMAN_BAR, "argmax_t": argmax_t,
                           "t4_at_max": t4_at_max,
                           "L_t1_t4": Ls},
            },
            "LANDING_IS_GEOMETRIC": {
                "fires": fires_geometric,
                "detail": {"spearman_L_depth": rho_depth,
                           "spearman_L_ratio": rho_ratio,
                           "reading": "flat or anti-concordant — the landing "
                                      "metric does not track the kill-ratio "
                                      "curve in the claim's direction"},
            },
            "GRADED": {"fires": fires_graded},
        }
        curve_str = ", ".join(
            f"t{r['t']}: L {fmt(r['L_eras'])}"
            + (f" ratio {r['kill_ratio_e200_committed']:.4f}"
               if r["kill_ratio_e200_committed"] is not None
               and r["t"] >= 1 else "")
            for r in table)
        if fires_lands:
            verdict = "LANDS-ON-SENSITIVITY"
            clause = (f"the landing metric tracks the kill-ratio curve: "
                      f"Spearman(L, depth) {rho_depth:+.2f} >= "
                      f"{SPEARMAN_BAR} on the 4 alive points AND the "
                      f"deepest landing (t4) is at the max (argmax t"
                      f"{argmax_t}); the table verbatim: {curve_str}; the "
                      f"survivor claim gains its mechanism leg — the "
                      f"deepest landing is the most fact-erasing-aligned "
                      f"front; n=1 lineage, 4 points, the counterfactual-"
                      f"wash caveat rides.")
        elif fires_geometric:
            verdict = "LANDING-IS-GEOMETRIC"
            clause = (f"the landing metric is flat/uncorrelated: Spearman(L, "
                      f"depth) {rho_depth:+.2f} <= 0 — the deepest landing "
                      f"(t4) is NOT special in erasing alignment (argmax t"
                      f"{argmax_t}); the table verbatim: {curve_str}; the "
                      f"rank-order is geometry (front norm/subspace "
                      f"effects), not fact-directed; the survivor stays a "
                      f"coincidence-shaped object; n=1 lineage, 4 points.")
        else:
            verdict = "GRADED"
            clause = (f"any partial — the table verbatim: {curve_str}; "
                      f"Spearman(L, depth) {rho_depth:+.2f} (bar >= "
                      f"{SPEARMAN_BAR} with t4 at the max for LANDS-ON-"
                      f"SENSITIVITY; <= 0 for LANDING-IS-GEOMETRIC); the "
                      f"argmax is t{argmax_t} "
                      + ("(t4, the deepest landing — but the concordance "
                         "is partial)" if t4_at_max else
                         "(NOT t4 — the deepest landing is not the most "
                         "erasing-aligned)")
                      + f"; the g-4-battery co-read's own Spearman "
                        f"{rho_co_depth:+.2f} (never adjudicated); n=1 "
                        f"lineage, 4 points, the counterfactual-wash "
                        f"caveat rides.")
    log(f"E204 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e204_support",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "question": ("THE SUPPORT MEASUREMENT: what does the deepest "
                     "landing land on? At each state of e197/e200's "
                     "half-step lineage (the one with the full front "
                     "rank-order), measure the fact's LOCAL sensitivity "
                     "direction (the gradient of the g-12 battery readout "
                     "at that state) and ask: does the killing front's "
                     "advantage (e200's committed kill-ratio curve; t4 the "
                     "deepest, the killing step) track alignment with the "
                     "LOCAL fact-sensitivity?"),
        "organism": {
            "root": f"runs/checkpoints/{ROOT_CK}",
            "root_meta": root_meta, "root_flat_md5": root_md5,
            "N_param": N_PARAM,
            "architecture": "4L/4H/128d/512-ctx TinyGPT (the e098 s4305 "
                            "line) — architecture co-varies with lineage "
                            "(e193's disclosure, carried)",
            "step_l2_committed_e193": E193_STEP_L2,
            "sub_step": {"s": E197_SUB_S, "fraction": E197_SUB_FRAC,
                         "natural_step": E193_STEP_L2,
                         "note": "e197's committed pick; the "
                                 "counterfactual-wash deviation carried "
                                 "verbatim: the direction machinery is the "
                                 "wash's own, the size the experimenter's"},
        },
        "rulers": {
            "primary": {"battery": f"install-60 g{RULER_J:+d}",
                        "root_read": root_cells[f"g{RULER_J:+d}"],
                        "why": "e193's frozen ruler call, carried verbatim "
                               "through e197/e200 (kill bars live here)"},
            "sensitivity_battery": {
                "battery": f"install-60 g{SENS_J:+d}",
                "objective": "mean log p(Z) (e194/e195's fact_grad "
                             "convention, VERBATIM)",
                "root_read_meanp": root_cells[f"g{SENS_J:+d}"],
                "disclosure": "the g-12 battery is the dispatch's frozen "
                              "choice; it reads 0.198 at this root — UNDER "
                              "the 0.27 bar at D=0 (e200's ruler "
                              "disclosure, carried); the g-4-battery "
                              "co-read rides per row, never adjudicated"},
            "co_rulers": {f"g{j:+d}": root_cells[f"g{j:+d}"]
                          for j in CO_RULERS_J},
            "ce_r": root_cells["ce_r"],
        },
        "cell": {
            "licensed_cell": "e193/e197/e200's stream/step convention "
                             "VERBATIM (seed 10902, draw order, post-clip "
                             "gradients, threads — capped at 4 by the owner "
                             "envelope, tiered gates); e197's HALF-STEP walk "
                             "rebuilt + gated (the state ladder theta_0.."
                             "theta_5 + the fronts u0..u4); NO static "
                             "profiles re-run (e200's committed D_kills ARE "
                             "the curve)",
            "states": {f"theta_{t}": {"provenance": "theta_0 = the gated "
                                       "root; theta_t = the rebuilt walk's "
                                       "step-t endpoint (G_WALKE197)",
                                       "ruler_read_g4":
                                           (next(r for r in wsub["traj"]
                                                 if r["step"] == t)["gm"]
                                            if t >= 1
                                            else root_cells[f"g{RULER_J:+d}"]),
                                       "alive": bool(t <= 4)}
                       for t in range(0, 6)},
            "fronts": [{"key": f"u{t}", "t": t, "u_md5": u_md5s[t],
                        "md5_match_committed":
                            bool(u_md5s[t] == md5_e200[t]),
                        "committed_parent": {0: "e193 (R2_SIGN)",
                                             1: "e197", 2: "e197",
                                             3: "e200", 4: "e200"}[t]}
                       for t in range(5)],
            "input_seed": FREEZE_SEED,
        },
        "gates": stub["gates"],
        "phase1_walk_rebuild": stub["phases_partial"]["1_walk_rebuild"],
        "phase1b_fronts": stub["phases_partial"]["1b_fronts"],
        "phaseS_sensitivity": stub["phases_partial"]["S_sensitivity_ladder"],
        "sensitivity_ladder": {
            "convention": "s_t = fp32 unit grad of mean log p(Z) on the "
                          "g-12 install-60 battery at theta_t (e194/e195's "
                          "fact_grad VERBATIM; matched-point with u_t)",
            "per_t": {t: {k: v for k, v in sens[t].items()
                          if k not in ("s", "s4")} for t in range(0, 6)},
            "geometry": sens_geom,
            "static_proxy_context": {
                "cos_u4_ug_committed_e200": E200_U4_UG,
                "reading": "the only support object before this cell: the "
                           "STATIC t=0 g-ray, whose overlap with the "
                           "killing front is near-orthogonal — T166's "
                           "stamp ('the support-proxy overlap -0.030 — the "
                           "support itself never measured'); compare "
                           "cos_s4_ug (the LOCAL sensitivity at the last "
                           "alive state vs the same static proxy)",
            },
        },
        "landing_table": table,
        "sign_convention_resolution": {
            "identity": "the step is -s*.u_t (e200's front convention) and "
                        "the erasing ray is -s_t; by e194's double-"
                        "negation identity the landing displacement's "
                        "alignment with the fact-ERASING ray is L_t = "
                        "cos(-u_t, -s_t) = cos(u_t, s_t)",
            "dispatch_literal": "cos(u_t, -s_t) = -L_t under the "
                                "readout-gradient reading of s_t (and = L_t "
                                "under the dispatch's own 'battery LOSS' "
                                "reading); both columns in the table "
                                "verbatim",
            "adjudicated_orientation": "L_t = cos(u_t, s_t); the "
                                       "rank-based bars are orientation-"
                                       "invariant",
        },
        "correlation": {
            "set": "t in {1,2,3,4} (the bar's 'the 4 points'; t0 "
                   "definitional-excluded per e199's MIRABEL precedent; the "
                   "death row context-only)",
            "L_t1_t4": Ls,
            "kill_ratios_e200_committed": ratios,
            "depths": depths,
            "spearman_L_depth": rho_depth,
            "spearman_L_ratio": rho_ratio,
            "bar": SPEARMAN_BAR,
            "argmax_t": argmax_t, "t4_at_max": t4_at_max,
            "co_read_g4battery": {"L_t1_t4": Ls_co,
                                  "spearman_L_depth": rho_co_depth,
                                  "note": "never adjudicated"},
            "committed_numbers_crosscheck": committed_crosscheck,
        },
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "composite_order": "LANDS-ON-SENSITIVITY -> LANDING-IS-"
                               "GEOMETRIC -> GRADED (frozen before compute)",
            "constants": {"SPEARMAN_BAR": SPEARMAN_BAR,
                          "SENS_BATTERY": f"g{SENS_J:+d}",
                          "CO_SENS_BATTERY": f"g{CO_SENS_J:+d}",
                          "SUB_S": E197_SUB_S, "FD_EPS": list(FD_EPS)},
        },
        "references": {
            "e200_deepening": {
                "metrics": "runs/e200/metrics.json",
                "role": "THE CURVE PARENT: the committed kill-ratio curve "
                        "(t1..t4; t4 the deepest = the killing step), the "
                        "u3/u4 ray md5s, the fresh ray geometry (incl. "
                        "cos_u4_ug — the static-proxy overlap) — loaded "
                        "COMMITTED, asserted, never rerun"},
            "e197_alive_window": {
                "metrics": "runs/e197/metrics.json",
                "role": "THE WALK PARENT: the half-step walk journal (the "
                        "kill at t=5), the registered u1/u2 rays — the "
                        "state ladder's provenance"},
            "e194_e195_fact_grad": {
                "metrics": "runs/e194/metrics.json, runs/e195/metrics.json",
                "role": "THE SENSITIVITY CONVENTION'S PARENT (organism 1): "
                        "fact_grad = grad of mean log p(Z) on the g-12 "
                        "battery, the critic's sign convention, the "
                        "dual-estimator matched-point lesson (T150)"},
            "e193_organism_replicate": {
                "metrics": "runs/e193/metrics.json",
                "role": "the organism's committed state: root identity, "
                        "STEP_L2, the R2_SIGN ray"},
            "e157": {"metrics": "runs/e157/metrics.json",
                     "role": "this root's committed dial (G_ROOT) + the "
                             "wash step-1 anchors (G_T0/G_S1CK)"},
            "thinking": "T164 (the survivor clause: death = the rotation's "
                        "deepest landing on the support), T166 + R61-critic "
                        "+ the null-derivation resolution (the survivors: "
                        "death-at-deepest-landing, 4/4 rank-order; the "
                        "support leg never measured — THIS cell)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 lineage (organism 2's f2 half-step line, a "
                            "COUNTERFACTUAL wash at half the natural step), "
                            "4 correlation points, one battery geometry, "
                            "one stream seed: whatever the verdict, it is a "
                            "biography read, not a population claim"),
            "alignment_reads_never_predict": ("this is an alignment read — "
                                              "the class T157 retired "
                                              "(alignment predicts nothing "
                                              "at n=4 lineages); it cannot "
                                              "establish causality (no "
                                              "intervention here) and does "
                                              "not try: the question is "
                                              "whether the committed "
                                              "rank-order's geometry is "
                                              "fact-directed"),
            "dual_estimator_lesson": ("every sensitivity is MATCHED-POINT: "
                                      "s_t is computed AT theta_t, the same "
                                      "state the front u_t is read at (the "
                                      "increment at the point it began — "
                                      "T150's lesson, stated per row)"),
            "four_point_statistics": ("Spearman at n=4 moves in 0.2 quanta; "
                                      "one flipped pair drops a perfect "
                                      "concordance to 0.6; the registered "
                                      "bars were frozen knowing this — "
                                      "GRADED is the expected honest branch"),
            "counterfactual_wash_caveat": ("the lineage is alive only "
                                           "because the experimenter halved "
                                           "the step (the natural 0.9164 "
                                           "step killed it at t=1); the "
                                           "sensitivity ladder belongs to "
                                           "the alive window OF THIS "
                                           "CONSTRUCTION (T158/T160, "
                                           "carried)"),
            "battery_weakness": ("the g-12 sensitivity battery reads 0.198 "
                                 "at this root (under the 0.27 bar at D=0); "
                                 "the g-4 co-read rides per row so the "
                                 "choice is auditable; neither adjudicates "
                                 "the other"),
            "openness": ("WHAT EACH ARM GUARANTEES: NOTHING — the "
                         "sensitivities are local linearizations of a weak "
                         "readout on states that could kill anywhere; the "
                         "correlation is a geometry statement about ONE "
                         "committed rank-order; the openness is the point"),
        },
        "deviations": deviations,
        "load_check": load_note,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                   "block_size": 512, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_support(rd / "e204_support.png", table, rho_depth, verdict,
                 rho_co_depth)
    log(f"outputs: {rd / 'metrics.json'} + PNG; total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_support(path, table, rho_depth, verdict, rho_co_depth):
    """THE LANDING METRIC vs THE KILL-RATIO CURVE (the cell's one picture):
    left = the per-t landing metric (adjudicated L + the literal column's
    negation identity implicit; the g-4 co-read; the death landing-point
    read as context); right = the same t-axis against e200's committed
    kill-ratio curve (inverted: deeper down) with the landing metric
    overlaid — the tracking the bars adjudicate."""
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8))

    ax = axes[0]
    ts_all = [r["t"] for r in table]
    Ls = [r["L_eras"] for r in table]
    ax.plot([t for t in ts_all if t <= 4], [Ls[t] for t in ts_all if t <= 4],
            "o-", ms=9, lw=2.0, color="seagreen",
            label="L_t = cos(u_t, s_t) — the landing metric (t0 context, "
                  "t1..t4 the 4-point set)")
    ax.plot([0], [Ls[0]], "o", ms=10, mfc="none", mec="seagreen", mew=1.6,
            ls="none", label="t0: the static ray (context; excluded)")
    d5 = next(r for r in table if r["t"] == 5)
    ax.plot([5], [d5["L_eras"]], "X", ms=12, mew=2.4, color="seagreen",
            label=f"death landing-point read cos(u4, s5) {d5['L_eras']:+.3f} "
                  f"(context only)")
    ax.plot([t for t in ts_all if 1 <= t <= 4],
            [r["co_L_eras_g4battery"] for r in table if 1 <= r["t"] <= 4],
            "s--", ms=6, lw=1.3, color="mediumpurple", alpha=0.85,
            label=f"g-4-battery co-read (never adjudicated; Spearman "
                  f"{rho_co_depth:+.2f})")
    ax.axhline(0.0, ls=":", lw=1.2, color="dimgray")
    ax.annotate("post-kill front (step 6)\nNEVER read", (5, 0.0),
                textcoords="offset points", xytext=(8, 6), fontsize=7.2,
                color="dimgray")
    for r in table:
        if r["t"] <= 4:
            ax.annotate(f"L{r['t']} {r['L_eras']:+.3f}", (r["t"], r["L_eras"]),
                        textcoords="offset points", xytext=(0, 11),
                        fontsize=8, ha="center", color="seagreen")
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("wash step t (front u_t and sensitivity s_t both read at "
                  "theta_t — matched-point)")
    ax.set_ylabel("landing metric  L_t = cos(u_t, s_t)")
    ax.set_title("THE LANDING METRIC — what each front's landing alignment "
                 "with the LOCAL fact-sensitivity is", fontsize=10.5)
    ax.legend(fontsize=7.2, loc="best")

    ax = axes[1]
    ts = [1, 2, 3, 4]
    ratios = [E200_ONSET_RATIOS[t] for t in ts]
    Ls4 = [next(r for r in table if r["t"] == t)["L_eras"] for t in ts]
    ax.plot(ts, ratios, "o-", ms=9, lw=2.2, color="navy",
            label="kill-ratio curve (e200 COMMITTED; DOWN = deeper = more "
                  "killing)")
    ax.set_ylabel("kill ratio  D_kill(root,u_t)/D_kill(root,u0)  (navy)",
                  color="navy")
    ax.tick_params(axis="y", labelcolor="navy")
    ax2 = ax.twinx()
    ax2.plot(ts, Ls4, "^-", ms=9, lw=2.0, color="seagreen",
             label="landing metric L_t (seagreen)")
    ax2.set_ylabel("landing metric L_t (seagreen)", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax.annotate("t4: the deepest landing\n= the killing step (ratio 0.397)",
                (4, ratios[3]), textcoords="offset points", xytext=(-10, 26),
                fontsize=8, color="navy", ha="right")
    ax.axvline(4, ls=":", lw=1.4, color="dimgray")
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("wash step t")
    ax.set_title(f"THE TRACKING TEST — Spearman(L, depth) {rho_depth:+.2f} "
                 f"(bar >= {SPEARMAN_BAR}, t4 at the max)", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.4, loc="upper center")

    fig.suptitle(f"E204 — THE SUPPORT MEASUREMENT (the survivor claim's "
                 f"missing leg) — VERDICT: {verdict}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
