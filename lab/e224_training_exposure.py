"""E224 — THE TRAINING-STEP EXPOSURE (e222/e223's last named limit).

WHY: e222 (runs/e222/metrics.json, T199) and e223 (runs/e223/metrics.json,
T201) ran the exposure-immunity causal test at one dose on BOTH organisms
(the flat-ordering f2 root and the strong-ordering e131 root) and at BOTH
signs — every same-axis row sat on the pass-back arithmetic to four
decimals, and T182's exposure-immunity ordering retired as a correlation.
Both cells named the SAME remaining limit: the pre-exposure was a DIRECT
weight displacement (eps 0.08-0.28 L2) — the wash's mechanism at ONE
REMOVE. The wash itself moves by AdamW steps (per-step L2 0.92-1.65)
whose directions are gradient-determined and normalizer-shaped. Does
immunity appear when the exposure is REAL TRAINING along the span?

WHAT IT BUILDS ON: e223's cell VERBATIM as the chassis (the pristine e131
root loaded as committed with e211's root-P gates; the contiguous 20-step
unwalled AdamW wash on the seed-10902 stream with g1b's committed C-arm
anchors; the fp64 Gram-SVD span, PR-gated to reproduce e211's root-P; the
install-60 g-12 ruler, bar 0.27; the onset-grid kill walk, first
downcrossing interpolated; the CE_R canary; the control's random in-span
draw); e222's dose/matched-L2 discipline; g12/T139's normalizer findings
(AdamW turns a direction-aligned gradient into a concentrated
sign-pattern step — the reason the dose must be matched on the
DISPLACEMENT, not the stream); e131's committed root provenance chain
hard-bound (Rule 12: e223's chain, extended by e223's own committed
records). NOTHING from the parents is re-adjudicated; every committed
number loads and hard-binds at its metrics path.

WHAT IS NEW: the exposure is TRAINING, not displacement. (1) THE SYNTHETIC
STREAM: 2 AdamW steps (the wash's own optimizer settings — lr 1e-3, betas
(0.9,0.95), wd 0.1, clip 1.0) on the wash's OWN first two batches, each
batch's gradient REPLACED BY ITS PROJECTION onto v0 (g~_i = (g_i.v0) v0) —
a stream flowing along the span's top direction; gradients are evaluated
LIVE at the exposure's current state (real training, the batches are the
wash's own). (2) THE MATCHED DOSE: the total displacement is matched to
~1 step's L2 (the wash's committed step-1 L2, 1.6543) by scaling the
2-step trajectory's NET displacement (gamma = T/||d||) — AdamW's
gradient-scale invariance (the normalizer sets the pace, T139) makes the
dose unreachable through the stream itself; DISCLOSED. (3) THE CONTROL:
the same 2-step procedure with the projections onto a RANDOM in-span unit
(seed 12503; accepted as the first of <= 5 draws for which the ladder
finds a both-alive rung — the shakedown's self-cancellation finding,
registered before the real compute). (4) THE SHARED RUNG LADDER: random
in-span directions kill fast (e223's ctrl baseline ~0.58 — far below the
top ray's 2.76), so the wash-step dose can kill the control arm
mechanically; the dose is accepted at the FIRST rung of {1x, 1/2, 1/4,
1/8} x 1.6543 at which BOTH arms' exposed states read alive (> 0.27)
under the extrapolation cap gamma <= 2.0; every rung of every draw
disclosed; the SAME T for both arms (matched L2, the control's
requirement); the arithmetic uses each arm's ACHIEVED displacement, so
no bias enters any residue.

DESK PRE-READ DISCLOSURE (arithmetic stated BEFORE compute, e205/e211/
e222/e223's convention; honesty, not a prediction — no bar is moved):
(1) THE REALIZED GEOMETRY: an AdamW step on g~ = a*v0 (a>0) is
approximately -lr*sign(v0) per coordinate — only ~0.8-aligned with -v0,
with L2 ~ lr*sqrt(N) ~ 1.65 PER STEP (the normalizer's geometry, g12);
2 verbatim steps therefore displace ~3.3 L2 — more than the kill along
some rays — hence the displacement matching above. (2) THE SIGN: the
projections are +v0-sided (the wash drifts along -v0, e223's measured
-4.73; the AdamW step moves AGAINST the gradient), so the training
exposure flows along the DRIFT side — the same-axis arithmetic is the
FLOOR form: predicted D = t_base + c, with c = <theta_exp - theta0, v>
(the ACHIEVED on-axis component, NEGATIVE). (3) THE ARITHMETIC'S LIMIT:
unlike e222/e223 (displacements EXACTLY along span axes), the training
displacement has an irreducible OFF-AXIS component (the normalizer's
contribution — the very content of "real training"); the pass-back
arithmetic t_base + c uses ONLY the on-axis part, so the residue now
carries the off-axis geometry too. That is the point of the cell: residue
= 0 means the optimizer's traversal adds NOTHING beyond position-along-
axis. (4) THE CROSS ROWS keep e223's status: no arithmetic, naive
prediction "unchanged", the differential (v0-arm minus control at the
same ray) is the registered comparison — EXCEPT at the u ray, where the
control's member is its OWN floor-loaded same-axis row (delta ~ c_u <
0): the raw formula would fire IMMEDIATELY-CLAUSE-B by construction
(+|c_u| mechanical excess) — the e166 instrument-tautology lesson
applied preemptively: clause (b) is adjudicated at the BOTH-CROSS ray
(mid) with the raw formula; the u-ray pair (and the control arm's top
row, loaded the same way) are reported verbatim, never adjudicated. The
bars below adjudicate the raw numbers exactly as registered; no bar
shopping.

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - TRAINING-IMMUNITY: "fires if the training exposure along v0 raises
    v0's kill-D BEYOND the pass-back arithmetic (residue >= +0.05) or
    the cross rays more than the control (>= +0.05) — the vaccination
    is a TRAINING phenomenon (the displacement was never the mechanism;
    the optimizer's traversal is)."
  - TRAINING-NULL: "fires if all arms sit on the arithmetic (residues
    within +-0.01) — the exposure-immunity reading retires at the
    training mechanism too; the ordering is descriptive geometry, full
    stop; the arc closes."
  - SENSITIZATION: "fires if the v0-training LOWERS the kill-D (residue
    <= -0.05) — training along a direction SENSITIZES it (the wash's
    drift as the mechanism of its own fragility) — an inversion worth
    naming."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * organism = the pristine e131 root, runs/checkpoints/
    e131_consolidated_e113.pt (2,739,072 params), used AS COMMITTED
    (flat md5-gated vs e211's committed 36ae2447...); ruler = install-60
    g-12 (the e131 lineage's battery family; e_chart's committed rung-0
    read 0.9155886173248291); co-rulers g0/g+12 ride in every table,
    never adjudicated; kill bar 0.27; kill = first downcrossing,
    linear-in-D interpolated, early stop; unresolved past the grid =
    censored (pairwise-dropped, disclosed).
  * D grid 0.05..3.50 step 0.05 (e223's extension carried: this root's
    t_top is 2.757 and the ctrl-arm cross rows can walk long).
  * span = the root's OWN contiguous 20-step unwalled AdamW wash (lr
    1e-3, betas (0.9,0.95), wd 0.1, clip 1.0, batch 32 = 16 e170-bank
    anchors + 16 random windows, the seed-10902 stream = g1b's C-arm,
    e211/e223 verbatim); basis = the fp64 Gram-SVD right-singular
    vectors Vp, SVD-emitted signs; the fresh span PR must reproduce
    e211's committed root-P PR within 1e-6 rel.
  * directions: TOP = dir 0, MID = dir 9 (of 20); CTRL-DIR = u = a
    randn(20)-coeffs in-span unit from seed 12503 (e222 drew 12501,
    e223 12502; the registry moves forward) — with THE DRAW-ACCEPTANCE
    RULE below (the smoke shakedown revealed the random-direction
    stream's self-cancellation lottery — the projections flip sign
    between steps, the net displacement collapses, and the gamma
    scaling extrapolates it; registered BEFORE the real compute, all
    draws disclosed).
  * the training arms: V0-TRAIN = 2 AdamW steps on the wash's own
    batches 1-2 with gradients replaced by their v0-projections
    (project-then-clip, the wash's clip convention); CTRL-TRAIN = the
    same procedure with the projections onto u; both from a fresh
    optimizer at theta0, gradients evaluated live along the exposure
    trajectory; both trajectories' NET displacements then gamma-scaled
    to the SAME accepted rung T.
  * the rung ladder: T in {1.6543, 0.8271, 0.4136, 0.2068} (= 1x, 1/2,
    1/4, 1/8 x the wash's committed step-1 L2 1.6542880535125732);
    accepted = the FIRST rung at which BOTH arms' scaled exposed states
    read > 0.27 on the primary ruler AND the endpoint scaling stays
    within the extrapolation cap gamma = T/||d_raw|| <= 2.0 for BOTH
    arms (the scaled endpoint must not more than double the actually-
    trained net displacement); every rung of every draw disclosed; the
    control direction is accepted as the first of <= 5 draws (seed
    12503, sequential) for which SOME rung accepts — e223's control
    acceptance rule composed with the ladder (the selection biases the
    control toward forgiveness, i.e. AGAINST the IMMUNITY clause —
    conservative, disclosed); if no draw accepts, the cell aborts with
    metrics written first (the e187 lesson). The verbatim (unscaled)
    2-step endpoints are also read, as context.
  * rows: kill-D of the top ray (v0), the mid ray (dir 9) and the
    control ray (u) from theta0 (the BEFORE baselines) and from BOTH
    exposed states (the AFTER): 3 baselines + 6 arm rows.
  * same-axis rows: V0-TRAIN/top and CTRL-TRAIN/ctrl carry the
    arithmetic t_base + c (c = the achieved on-axis displacement along
    that ray, fp64), the mechanical ratio 1 + c/t_base, and the residue
    kill-D_post - (t_base + c).
  * TRAINING-IMMUNITY clauses: (a) residue(V0-TRAIN/top) >= +0.05;
    (b) the cross-ray differential at the both-cross ray mid:
    [D(V0-TRAIN,mid) - t_base(mid)] - [D(CTRL-TRAIN,mid) -
    t_base(mid)] >= +0.05. Fires if (a) OR (b). A censored V0-TRAIN/top
    row makes (a) unresolvable and a censored mid pair makes (b)
    unresolvable (reported, never guessed). The u-ray pair and the
    ctrl-arm's top row are reported verbatim (raw + residue-reduced
    forms) with their mechanical loading stated — never adjudicated.
  * SENSITIZATION clause: residue(V0-TRAIN/top) <= -0.05.
  * TRAINING-NULL clause: neither TRAINING-IMMUNITY nor SENSITIZATION
    fired AND both same-axis residues resolved and within +-0.01.
  * composite order TRAINING-IMMUNITY / SENSITIZATION / TRAINING-NULL /
    GRADED (the first two mutually exclusive — the same residue cannot
    be both >= +0.05 and <= -0.05).
  * CE_R canary (the organism-health honesty reflex): val-window CE
    (60 windows, seed 26502) at the root, both exposed states, and the
    root's dir-0 kill position; reported, never adjudicated.

PRE-DISPATCH CHECKS (Rule 12): the root's provenance (e223's chain:
e131's committed gates, e211's root-P block, e222's NULL, e223's
NULL-CONFIRMS + its four residues, e_chart's rung-0 read, g1b's C-arm
rows — all hard-bound); the span conventions (the PR reproduction gate
+ the per-step C-arm anchors); the synthetic-stream construction
documented (the projection arithmetic g~ = (g.v0)v0 with live
gradients on the wash's own batches 1-2, bit-checked against the wash's
own draws; the displacement matched and ASSERTED: ||theta_exp-theta0||
= T to 1e-4 (fp32 endpoint arithmetic; ~1e-5 realized)). WHAT THE GATES GUARANTEE: NOTHING — the training residues
could sit at zero (the arc closes), the traversal could sensitize (the
inversion), or the off-axis normalizer geometry could finally separate
the v0-training from the control. The openness is the point.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is never claimed), torch threads 4, load-check
recorded not gating (e204/e205/e209/e211/e222/e223's convention), small
training bursts (2 AdamW steps per arm — seconds), PROGRESSIVE
metrics.json writes, decisive.

Outputs: runs/e224/{metrics.json (PROGRESSIVE), e224_training_exposure.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e224_training_exposure.py    (E224_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e185/e209/e223)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)          # the owner envelope (e211/e223's CPU reduction order)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E224_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e224 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- the organism (the pristine e131 root, e211/e223's constants verbatim) ----
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"          # the pristine e131 root itself
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E131_METRICS = E43.REPO / "runs" / "e131" / "metrics.json"
E211_METRICS = E43.REPO / "runs" / "e211" / "metrics.json"
E222_METRICS = E43.REPO / "runs" / "e222" / "metrics.json"
E223_METRICS = E43.REPO / "runs" / "e223" / "metrics.json"
E_CHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
G1B_METRICS = E43.REPO / "runs" / "g1b" / "metrics.json"

N_PARAMS = 2_739_072
ROOT_FLAT_MD5 = "36ae244756475dda71266346e7a0f196"     # e211's committed root-P flat md5

GEOS = (-12, 0, 12)               # battery ctx offsets (e185/e209/e211/e223 convention)
RULER_J = -12                     # THE RULER: install-60 g-12 (the e131 lineage's)
CO_RULERS_J = (0, 12)

# ---- the committed parents, hard-bound at load -------------------------------
ECHART_PRISTINE_READ = 0.9155886173248291        # e_chart partB.e131.wash curve rung 0
E211_P_COMMITTED = {             # runs/e211/metrics.json roots.P
    "span_pr": 4.117363094870968,
    "span_sv0": 1.9597355390295585,
    "dir0_kill": 2.7565262446927674,
    "spearman_p": 0.8095238095238095,
    "spearman_pooled": 0.6391129032258065,
    "verdict": "GRADED",
}
E222_COMMITTED = {               # runs/e222/metrics.json adjudication
    "verdict": "NULL",
    "residue_dir0": 0.0006171217475904323,
    "ratio_dir0_after_top_exp": 1.1007696847105353,
}
E223_COMMITTED = {               # runs/e223/metrics.json adjudication (this cell's parent)
    "verdict": "NULL-CONFIRMS",
    "residue_top_exp_top": 0.00014882415150596628,
    "residue_drift_exp_top": 0.00015737293170614564,
    "residue_ctrl_exp_ctrl": -0.00012952470655325232,
    "eps": 0.27565262446927674,
    "drift_dir0": -4.725435313142226,
}
E131_COMMITTED = {               # runs/e131/metrics.json (the root's provenance)
    "g_e048_battery_pz": 0.5563086867332458,
    "fired": "RE-KEYED",
    "install_mix": {"FLORIZEL": 19, "ELIZABETH": 41},
}
G1B_C1 = {"ce_batch": 1.3565675020217896, "step_disp": 1.6542880535125732}

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH_1 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's seed-10902 step-1 batch md5

# ---- the run envelope (dispatch-frozen) ----------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (g1b's C-arm at the pristine root)
LR_ADAMW = 1e-3
BETAS = (0.9, 0.95)
WD = 0.1
CLIP = 1.0
ANCH_BS, RAND_BS = 16, 16
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 71)]   # 0.05..3.50 (e223's grid)
WASH_HIST_STEPS = 4 if SMOKE else 20
TRAIN_STEPS = 2                   # the registered training exposure (2 AdamW steps)
DIR_TOP, DIR_MID = 0, 9
CTRL_SEED = 12503                 # fresh (e222 drew 12501, e223 12502; registry forward)
CTRL_MAX_DRAWS = 5                # the draw-acceptance rule's budget (e222/e223 convention)
GAMMA_CAP = 2.0                   # endpoint-scaling extrapolation cap (registered)
RESIDUE_BAR = 0.05                # the IMMUNITY / SENSITIZATION residue bar (registered)
NULL_RESIDUE_TOL = 0.01           # the TRAINING-NULL arithmetic tolerance (registered)
G_BIT_TOL = 5e-6
G_READ_TOL = 5e-3                 # cross-device texture tier (e211/e223's G_HIST_TOL)
G_PR_TOL = 1e-6                   # span PR reproduction, CPU->CPU same threads (e211)
VAL_SEED = 26502                  # e222/e223's CE_R val-window seed
MATCH_TOL = 1e-4                  # the displacement-matching assert tolerance
                                   # (fp32 endpoint arithmetic loses ~1e-5 to
                                   # cancellation over 2.74M params; 1e-4 is
                                   # still a real assert of the match)

# the rung ladder: fractions of the wash's own committed step-1 L2
RUNG_FRACS = [1.0, 0.5, 0.25, 0.125]
T_WASH_STEP1 = G1B_C1["step_disp"]

if SMOKE:                         # shakedown trims (disclosed; nothing adjudicated)
    D_GRID = [0.05, 0.5, 1.5, 3.5]
    DIR_MID = 1                   # the 4-dir span re-indexed (top stays 0)
    RUNG_FRACS = [1.0, 0.5, 0.25]

REGISTERED_BARS = {
    "TRAINING_IMMUNITY": "TRAINING-IMMUNITY: \"fires if the training exposure "
        "along v0 raises v0's kill-D BEYOND the pass-back arithmetic (residue "
        ">= +0.05) or the cross rays more than the control (>= +0.05) — the "
        "vaccination is a TRAINING phenomenon (the displacement was never the "
        "mechanism; the optimizer's traversal is).\"",
    "TRAINING_NULL": "TRAINING-NULL: \"fires if all arms sit on the arithmetic "
        "(residues within +-0.01) — the exposure-immunity reading retires at "
        "the training mechanism too; the ordering is descriptive geometry, "
        "full stop; the arc closes.\"",
    "SENSITIZATION": "SENSITIZATION: \"fires if the v0-training LOWERS the "
        "kill-D (residue <= -0.05) — training along a direction SENSITIZES it "
        "(the wash's drift as the mechanism of its own fragility) — an "
        "inversion worth naming.\"",
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": (
        "organism = the pristine e131 root as committed (flat md5-gated vs "
        "e211's root-P block); ruler = install-60 g-12 (bar 0.27); grid "
        "0.05..3.50; kill = first downcrossing interpolated; span = the "
        "root's own contiguous 20-step unwalled AdamW wash (seed-10902 g1b "
        "C-arm stream, per-step L2 gated vs g1b's committed CUDA rows at "
        "5e-3; the fresh span PR must reproduce e211's committed root-P PR "
        "within 1e-6 rel), fp64 Gram-SVD basis, SVD-emitted signs; TOP=dir0 "
        "MID=dir9; CTRL-DIR u = first randn(20)-coeffs in-span unit from "
        "seed 12503; ARMS: V0-TRAIN = 2 AdamW steps (the wash's own settings "
        "lr 1e-3 / betas (0.9,0.95) / wd 0.1 / clip 1.0) on the wash's own "
        "batches 1-2, each gradient REPLACED by its projection onto v0 "
        "(g~ = (g.v0) v0, live gradients, project-then-clip); CTRL-TRAIN = "
        "the same with the projections onto u; each trajectory's NET "
        "displacement gamma-scaled to the SAME accepted rung T of {1x, 1/2, "
        "1/4, 1/8} x 1.6543 (the wash's committed step-1 L2; the FIRST rung "
        "at which BOTH arms' exposed states read > 0.27 under the "
        "extrapolation cap gamma = T/||d_raw|| <= 2.0 — the control "
        "direction accepted as the first of <= 5 sequential seed-12503 "
        "draws for which some rung accepts, every rung of every draw "
        "disclosed, the selection biased AGAINST the IMMUNITY clause; the "
        "displacement match ASSERTED to 1e-4); rays "
        "top/mid/ctrl walked from theta0 and from BOTH exposed states; "
        "same-axis arithmetic t_base + c with c = the achieved on-axis "
        "displacement (fp64); residue = kill-D_post - (t_base + c); "
        "IMMUNITY clause (a) residue(V0-TRAIN/top) >= +0.05 OR (b) the "
        "both-cross differential [D(V0T,mid)-t(mid)] - [D(CTRLT,mid)-"
        "t(mid)] >= +0.05; SENSITIZATION residue(V0-TRAIN/top) <= -0.05; "
        "TRAINING-NULL = neither fired AND both same-axis residues within "
        "+-0.01; the u-ray pair and the ctrl-arm's top row are reported "
        "verbatim (mechanically loaded by their own-axis arithmetic — "
        "clause (b) is NOT adjudicated there, the e166 tautology lesson "
        "preemptively); CE_R canary report-only; composite TRAINING-IMMUNITY "
        "/ SENSITIZATION / TRAINING-NULL / GRADED."),
    "desk_pre_read_disclosure": (
        "ARITHMETIC, stated before compute (no bar moved): (1) an AdamW step "
        "on g~ = a*v0 is ~ -lr*sign(v0) per coordinate (the normalizer's "
        "geometry, g12/T139): ~0.8-aligned with -v0, L2 ~ lr*sqrt(N) ~ 1.65 "
        "PER STEP — the matched dose is therefore set on the DISPLACEMENT "
        "(gamma-scaled endpoint), not the stream (Adam's gradient-scale "
        "invariance), disclosed. (2) The projections are +v0-sided (the wash "
        "drifts -v0; e223 measured -4.73), so the AdamW steps flow along the "
        "DRIFT side: the same-axis arithmetic is the FLOOR form t_base + c "
        "with c < 0 (c = the ACHIEVED on-axis displacement). (3) Unlike "
        "e222/e223, the training displacement has an irreducible OFF-AXIS "
        "component (the normalizer's contribution); the arithmetic uses only "
        "the on-axis part, so the residue carries the off-axis geometry — "
        "that is the cell's content (residue = 0 => the traversal adds "
        "nothing beyond position-along-axis). (4) The cross rows keep 'no "
        "arithmetic, naive unchanged' EXCEPT the u-ray pair (the control's "
        "member is its own floor-loaded same-axis row: the raw formula fires "
        "+|c_u| mechanically — clause (b) is adjudicated at the both-cross "
        "mid ray only; the loaded rows reported verbatim). The bars "
        "adjudicate the raw numbers exactly as registered."),
    "registration": "the dispatch's registration IS the registration (the "
        "four bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE EXPOSURE IS TRAINING, NOT DISPLACEMENT (the dispatch's cell): 2 "
    "AdamW steps per arm on the wash's own batches 1-2, each gradient "
    "replaced by its projection onto the target direction — the first "
    "exposure in the arc whose path is the optimizer's.",
    "THE DOSE IS SET ON THE DISPLACEMENT, NOT THE STREAM (disclosed): "
    "AdamW's gradient-scale invariance (the normalizer sets the pace — "
    "T139/g12) means no stream scaling can set the total displacement; the "
    "2-step trajectories run VERBATIM at lr 1e-3 and their NET displacements "
    "are gamma-scaled to the accepted rung T; the rung ladder {1x, 1/2, 1/4, "
    "1/8} x the wash's step-1 L2 is accepted at the first rung where BOTH "
    "arms read alive (every rung disclosed; e223's control-acceptance rule "
    "extended to both arms symmetrically).",
    "THE GRADIENTS ARE LIVE (the state-dependence choice, disclosed): the "
    "wash's own batches in the wash's own order, but each gradient is "
    "evaluated at the EXPOSURE's current state (real training), not at the "
    "wash's historical states; step-1's gradient is identical to the wash's "
    "own (same batch, same theta0) by construction.",
    "THE CTRL-DIRECTION DRAW-ACCEPTANCE RULE (registered BEFORE the real "
    "compute, after the SMOKE-stamped shakedown revealed the issue): a "
    "random in-span direction's projected stream can SELF-CANCEL (the "
    "projections flip sign between steps, the net displacement collapses, "
    "the gamma scaling extrapolates a tiny net into a lethal arbitrary "
    "displacement); the control is therefore accepted as the first of <= 5 "
    "sequential draws (seed 12503) for which the shared rung ladder finds "
    "a both-alive rung under the extrapolation cap gamma <= 2.0 — e223's "
    "own control-acceptance convention composed with the ladder; every "
    "draw and every rung disclosed; the selection biases the control "
    "toward forgiveness (AGAINST the IMMUNITY clause — conservative).",
    "THE SAME-AXIS ARITHMETIC IS NOW AN APPROXIMATION BY DESIGN (unlike "
    "e222/e223's exact pass-back): the training displacement's off-axis "
    "component is not (and cannot be) in the arithmetic t_base + c; the "
    "residue carries it — the desk disclosure states this before compute.",
    "CLAUSE (b) IS ADJUDICATED AT THE BOTH-CROSS MID RAY ONLY: at the u ray "
    "the control's member is its own floor-loaded same-axis row (the raw "
    "formula would fire +|c_u| mechanically — the e166 instrument-tautology "
    "lesson applied preemptively); the u-ray pair and the ctrl-arm's top row "
    "are reported verbatim (raw + residue-reduced), never adjudicated.",
    "THE DESK DISCLOSURE'S MID-RAY CLAIM, TESTED AND FALSIFIED BY THE "
    "MEASURED NUMBERS (disclosed post-compute; nothing un-fired): the "
    "disclosure asserted 'no mechanical loading at the both-cross mid ray' "
    "— but the training displacements are not span axes, and the control's "
    "displacement carries its own on-mid component, which shifts the mid "
    "kill-D mechanically under axis-separability; the raw registered "
    "formula adjudicates exactly as frozen (no bar shopping), and the "
    "on-ray decomposition (mechanical vs terrain parts, the control's own "
    "shift confirmation) is computed alongside in "
    "adjudication.bars.TRAINING_IMMUNITY.clause_b_mid.on_ray_decomposition "
    "and carried into the verdict clause and honesty — the e166 "
    "instrument-tautology discipline applied AFTER the fact rather than "
    "before it, reported plainly.",
    "The load-check is recorded, not gating (e204/e205/e209/e211/e223's "
    "convention).",
    "Smoke mode trims: D grid {0.05,0.5,1.5,3.5}, 4-step history (basis dirs "
    "re-indexed, mid=1), 3 rungs; nothing adjudicated (verdict stamped "
    "SMOKE).",
]

# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e223_exposure_e131.py VERBATIM as the chassis (itself
# lab/e211_walled_band.py's e131-family gates / battery / stream / wash
# history / Gram-SVD span / onset-grid kill walk + lab/e222_exposure.py's
# exposure-cell grammar). Copied rather than imported to own the device
# policy and the arithmetic. NEW this cell: the projected-stream training
# arms, the displacement-matching rung ladder, the training-adjudication.


def load_body(path) -> tuple[TinyGPT, dict, dict]:
    """Load a checkpoint: (net with BODY weights, body sd, anch__ buffers)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    meta = st.get("meta") if isinstance(st, dict) else None
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    anch = {k: v for k, v in sd.items() if k.startswith("anch__")}
    body = {k: v for k, v in sd.items() if not k.startswith("anch__")}
    m = TinyGPT(Cfg())
    m.load_state_dict(body)
    m.eval()
    return m, body, {"meta": meta, "anch": anch}


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e209/e211/e223 battery on CPU: p(Z) at the last position."""
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
        lg, _ = net(x[i:i + bs], y[i:i + bs])
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[i:i + bs].reshape(-1))
        tot += float(ce.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065/e193/e222/e223 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y, tries = [], [], 0
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


def interp_d_kill(v0, v1, d0, d1):
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def participation_ratio(sv) -> float:
    s2 = sv if isinstance(sv, torch.Tensor) else torch.tensor(sv, dtype=torch.float64)
    s2 = (s2.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """e_chart/e193b/e205/e209/e211/e222/e223's Gram-based right-singular basis (fp64)."""
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


def cpu_load_probe() -> int | None:
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def train_projected(net0: TinyGPT, theta0: torch.Tensor, batches, v_target):
    """THE TRAINING EXPOSURE (new this cell): TRAIN_STEPS AdamW steps (the
    wash's own optimizer settings) on the wash's own batches, each gradient
    REPLACED BY ITS PROJECTION onto v_target (live, project-then-clip).
    Returns (d_raw, step_rows)."""
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR_ADAMW, betas=BETAS,
                            weight_decay=WD)
    step_rows = []
    th_prev = theta0.clone()
    for i, (x_, y_) in enumerate(batches, start=1):
        logits, _ = net(x_)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             y_.reshape(-1))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        g_flat = torch.cat([p.grad.detach().reshape(-1)
                            if p.grad is not None
                            else torch.zeros(p.numel())
                            for p in net.parameters()])
        a = float(torch.dot(g_flat.double(), v_target.double()))
        g_proj = a * v_target
        proj_norm = float(torch.norm(g_proj))
        with torch.no_grad():
            k = 0
            for p in net.parameters():
                n = p.numel()
                p.grad = g_proj[k:k + n].view_as(p).clone()
                k += n
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        th_new = flat_params(net)
        step_rows.append({
            "step": i,
            "ce_batch": float(ce.item()),
            "grad_L2_raw": float(torch.norm(g_flat)),
            "proj_coeff_a": a,
            "cos_grad_vs_target": float(
                torch.dot(g_flat.double(), v_target.double())
                / (torch.norm(g_flat.double()) * torch.norm(v_target.double()) + 1e-30)),
            "proj_grad_L2": proj_norm,
            "clipped": bool(proj_norm > CLIP),
            "step_L2": float(torch.norm(th_new - th_prev)),
            "cum_L2_from_root": float(torch.norm(th_new - theta0)),
        })
        th_prev = th_new
    d_raw = flat_params(net) - theta0
    del net, opt
    return d_raw, step_rows


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e224_smoke" if SMOKE else "e224")
    metrics: dict = {
        "experiment": "e224_training_exposure",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never claimed)",
            "torch_threads": torch.get_num_threads(),
            "load_check_recorded_not_gating": True,
            "phases": "P0 gates -> P1 span -> P2 baselines -> P3 the training "
                      "arms (projected stream + rung ladder + kill rows) -> "
                      "P4 adjudication + figure; progressive writes",
            "organism": f"runs/checkpoints/{ROOT_CK} (2,739,072 params, the "
                        "pristine e131 root = e211's root P = e223's organism)",
            "ruler": f"install-60 g{RULER_J:+d} (the e131 lineage's battery "
                     f"family; bar {SHUT_BAR})",
            "dirs": {"top": DIR_TOP, "mid": DIR_MID},
            "train_steps": TRAIN_STEPS,
            "optimizer": f"AdamW lr {LR_ADAMW} betas {BETAS} wd {WD} clip {CLIP} "
                         "(the wash's own settings)",
            "ctrl_seed": CTRL_SEED,
        },
        "deviations": deviations,
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    load0 = cpu_load_probe()
    metrics["envelope"]["cpu_load_pct_at_launch"] = load0
    log(f"E224 THE TRAINING-STEP EXPOSURE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), load-check "
        f"recorded (launch: {load0}%), progressive writes, decisive")

    # ================= P0a: protocol rebuild (e211/e223's gate set) ============
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
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == E131_COMMITTED["install_mix"])}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "expected_shapes": {"-12": [60, PRE - 12], "0": [60, PRE],
                            "12": [60, PRE + 12]},
        "pass": bool(list(bat_ids[RULER_J].shape) == [60, PRE + RULER_J]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at ctx "
                "offsets {-12,0,+12} — e185/e209/e211/e223's convention = "
                f"the e131 lineage's own battery; PRIMARY ruler = g{RULER_J:+d}",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]
    coruler_ids = {j: bat_ids[j] for j in CO_RULERS_J}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, VAL_SEED)

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
    G_ANCHOR = {"starts": n_starts, "tries": tries, "rejections": rejections,
                "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
                "n_windows": 16, "block": BLOCK, "seed": 170}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"
    log("P0a: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank bit-match)")
    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}
    write_partial("P0a protocol gates PASSED")

    # ================= P0b: the parents, hard-bound (Rule 12) ==================
    for p in (E131_METRICS, E211_METRICS, E222_METRICS, E223_METRICS,
              E_CHART_METRICS, G1B_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")

    def md5of(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    e131m = json.loads(E131_METRICS.read_text(encoding="utf-8"))
    e211m = json.loads(E211_METRICS.read_text(encoding="utf-8"))
    e222m = json.loads(E222_METRICS.read_text(encoding="utf-8"))
    e223m = json.loads(E223_METRICS.read_text(encoding="utf-8"))
    echm = json.loads(E_CHART_METRICS.read_text(encoding="utf-8"))
    g1bm = json.loads(G1B_METRICS.read_text(encoding="utf-8"))
    parents_md5 = {"e131": md5of(E131_METRICS), "e211": md5of(E211_METRICS),
                   "e222": md5of(E222_METRICS), "e223": md5of(E223_METRICS),
                   "e_chart": md5of(E_CHART_METRICS), "g1b": md5of(G1B_METRICS)}

    # hard-bind the committed references (asserts catch committed-file drift)
    assert e131m["gates"]["G_E048"]["battery_pz"] == E131_COMMITTED["g_e048_battery_pz"]
    assert e131m["gates"]["G_SPLICE"]["install_mix"] == E131_COMMITTED["install_mix"]
    assert e131m["adjudication"]["fired"] == E131_COMMITTED["fired"]
    rp = e211m["roots"]["P"]
    assert abs(rp["span_primary"]["participation_ratio"]
               - E211_P_COMMITTED["span_pr"]) < 1e-12
    assert abs(rp["span_primary"]["sv"][0]
               - E211_P_COMMITTED["span_sv0"]) < 1e-12
    assert abs(rp["dir_kills"][0]["D_kill"]
               - E211_P_COMMITTED["dir0_kill"]) < 1e-12
    e211_corr = e211m["band_re_read"]["correlations_context"]
    assert abs(e211_corr["P"]["spearman_kill_vs_sv"]
               - E211_P_COMMITTED["spearman_p"]) < 1e-12
    assert abs(e211_corr["pooled"]["spearman_kill_vs_sv"]
               - E211_P_COMMITTED["spearman_pooled"]) < 1e-12
    assert e211m["adjudication"]["verdict"] == E211_P_COMMITTED["verdict"]
    assert rp["G_ROOT"]["settled_flat_md5"] == ROOT_FLAT_MD5
    a222 = e222m["adjudication"]
    assert a222["verdict"] == E222_COMMITTED["verdict"]
    assert abs(a222["inputs"]["residue_dir0"]
               - E222_COMMITTED["residue_dir0"]) < 1e-12
    assert abs(a222["inputs"]["ratio_dir0_after_top_exp"]
               - E222_COMMITTED["ratio_dir0_after_top_exp"]) < 1e-12
    a223 = e223m["adjudication"]
    assert a223["verdict"] == E223_COMMITTED["verdict"]
    assert abs(a223["inputs"]["residue_top_exp_top"]
               - E223_COMMITTED["residue_top_exp_top"]) < 1e-12
    assert abs(a223["inputs"]["residue_drift_exp_top"]
               - E223_COMMITTED["residue_drift_exp_top"]) < 1e-12
    assert abs(a223["inputs"]["residue_ctrl_exp_ctrl"]
               - E223_COMMITTED["residue_ctrl_exp_ctrl"]) < 1e-12
    assert abs(e223m["dose"]["eps_L2"] - E223_COMMITTED["eps"]) < 1e-12
    assert abs(e223m["span"]["wash_drift_proj_dirs"]["dir0"]
               - E223_COMMITTED["drift_dir0"]) < 1e-12
    e131_spec = echm["partB_subspace"]["e131"]
    assert abs(e131_spec["wash"]["curve"][0]["gm12"] - ECHART_PRISTINE_READ) < 1e-12
    carm_rows = {int(r["step"]): r for r in g1bm["arms"]["C"]["traj"]}
    assert 1 in carm_rows and 20 in carm_rows
    assert abs(carm_rows[1]["ce_batch"] - G1B_C1["ce_batch"]) < 1e-12
    assert abs(carm_rows[1]["step_disp"] - G1B_C1["step_disp"]) < 1e-12
    G_PARENTS = {
        "files_md5": parents_md5,
        "hardbound": {
            "e131.g_e048.battery_pz": E131_COMMITTED["g_e048_battery_pz"],
            "e131.fired": E131_COMMITTED["fired"],
            "e211.rootP.span_pr": E211_P_COMMITTED["span_pr"],
            "e211.rootP.span_sv0": E211_P_COMMITTED["span_sv0"],
            "e211.rootP.dir0_kill": E211_P_COMMITTED["dir0_kill"],
            "e211.spearman_P": E211_P_COMMITTED["spearman_p"],
            "e211.spearman_pooled": E211_P_COMMITTED["spearman_pooled"],
            "e211.verdict": E211_P_COMMITTED["verdict"],
            "e222.verdict": E222_COMMITTED["verdict"],
            "e222.residue_dir0": E222_COMMITTED["residue_dir0"],
            "e223.verdict": E223_COMMITTED["verdict"],
            "e223.residue_top_exp_top": E223_COMMITTED["residue_top_exp_top"],
            "e223.residue_drift_exp_top": E223_COMMITTED["residue_drift_exp_top"],
            "e223.residue_ctrl_exp_ctrl": E223_COMMITTED["residue_ctrl_exp_ctrl"],
            "e223.eps": E223_COMMITTED["eps"],
            "e223.drift_dir0": E223_COMMITTED["drift_dir0"],
            "e_chart.pristine_root_read": ECHART_PRISTINE_READ,
            "g1b.C_traj_rows_1_to_20": "loaded (the pristine history anchors)",
        },
        "note": "e223's NULL-CONFIRMS (residues +-0.0002, both signs — THIS "
                "cell's direct parent and replication target), e222's NULL, "
                "e211's root-P ordering block, e131's committed root "
                "provenance — all loaded, never re-run",
        "pass": True,
    }
    log("P0b: parents hard-bound — e131 (RE-KEYED), e211 (root P: PR 4.1174, "
        "dir0 kill 2.7565, Spearman +0.809), e222 (NULL), e223 (NULL-CONFIRMS, "
        "residues +-0.0002), e_chart (0.9156), g1b (C-arm rows 1..20)")
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    write_partial("P0b parent gates PASSED (e131/e211/e222/e223/e_chart/g1b hard-bound)")

    # ================= P0c: the root ===========================================
    net0, _, root_extra = load_body(CKPT_DIR / ROOT_CK)
    theta0 = flat_params(net0)
    N_PAR = int(theta0.numel())
    assert N_PAR == N_PARAMS, f"params {N_PAR} != {N_PARAMS}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    evl = copy.deepcopy(net0)
    root_cells = {f"g{j:+d}": battery_cell(evl, bat_ids[j], zid)["mean_pz"]
                  for j in GEOS}
    root_cells["ce_r"] = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
    root_read = root_cells[f"g{RULER_J:+d}"]
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "meta": root_extra.get("meta"),
        "n_params": N_PAR,
        "cells": root_cells,
        "settled_read_measured": root_read,
        "settled_read_committed": ECHART_PRISTINE_READ,
        "abs_diff": abs(root_read - ECHART_PRISTINE_READ),
        "bit_tol": G_BIT_TOL, "tol": G_READ_TOL,
        "bit": bool(abs(root_read - ECHART_PRISTINE_READ) < G_BIT_TOL),
        "flat_md5": root_md5,
        "flat_md5_match_e211_committed": bool(root_md5 == ROOT_FLAT_MD5),
        "pass": bool(abs(root_read - ECHART_PRISTINE_READ) < G_READ_TOL
                     and root_md5 == ROOT_FLAT_MD5),
        "note": "the pristine e131 root used AS COMMITTED (e211's root-P "
                "convention: flat md5 BIT-gated vs e211's committed block; "
                "read gated vs e_chart's rung-0 install-60 g-12)",
    }
    log(f"P0c G_ROOT: read {root_read:.7f} (|d| {G_ROOT['abs_diff']:.1e}), "
        f"md5 {'OK' if G_ROOT['flat_md5_match_e211_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("e131 root gate FAILED")
    metrics["gates"]["G_ROOT"] = G_ROOT
    write_partial("P0c G_ROOT PASSED (the pristine e131 root as committed)")

    # ================= P0d: G_STREAM (e211/e223's pristine stream gate) ========
    g = torch.Generator().manual_seed(FREEZE_SEED)
    aj = torch.randint(16, (ANCH_BS,), generator=g)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
    x1 = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y1 = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(net0)
    tw.train()
    optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW, betas=BETAS,
                             weight_decay=WD)
    logits, _ = tw(x1)
    ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y1.reshape(-1)).item())
    optw.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y1.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(tw.parameters(), CLIP)
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0))
    del tw, optw, logits
    g1b_c1 = carm_rows[1]
    G_STREAM = {
        "seed": FREEZE_SEED, "src": "runs/g1b/metrics.json arms.C.traj[0] "
                                    "(the seed-10902 wash AT the pristine root)",
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH_1),
        "ce1_measured": ce1, "ce1_committed": g1b_c1["ce_batch"],
        "d_ce": abs(ce1 - g1b_c1["ce_batch"]),
        "disp1_measured": disp1, "disp1_committed": g1b_c1["step_disp"],
        "d_disp": abs(disp1 - g1b_c1["step_disp"]), "tol": G_READ_TOL,
        "pass": bool(abs(ce1 - g1b_c1["ce_batch"]) < G_READ_TOL
                     and abs(disp1 - g1b_c1["step_disp"]) < G_READ_TOL
                     and x1_md5 == E185_XHASH_1),
        "note": "the pristine stream IS g1b's own C-arm stream; step-1 CE + "
                "AdamW L2 vs the committed CUDA row (cross-device texture "
                "tier, e211/e223's pre-verified convention)",
    }
    log(f"P0d G_STREAM: CE |d| {G_STREAM['d_ce']:.1e}, disp |d| "
        f"{G_STREAM['d_disp']:.1e}, md5 "
        f"{'OK' if G_STREAM['step1_x_md5_match_e185'] else 'DRIFT'}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    if not G_STREAM["pass"]:
        raise RuntimeError("pristine stream gate FAILED")
    metrics["gates"]["G_STREAM"] = G_STREAM
    del evl
    write_partial("P0d G_STREAM PASSED (the seed-10902 stream = g1b's C-arm)")

    # ================= P1: the wash history + the span =========================
    log("=" * 78)
    log(f"P1: the e131 root's own contiguous {WASH_HIST_STEPS}-step unwalled "
        f"AdamW wash (seed {FREEZE_SEED}) -> the span")
    wgen = torch.Generator().manual_seed(FREEZE_SEED)
    wnet = copy.deepcopy(net0)
    wnet.train()
    wopt = torch.optim.AdamW(wnet.parameters(), lr=LR_ADAMW, betas=BETAS,
                             weight_decay=WD)
    wtheta = theta0.clone()
    segs, hist_rows = [], []
    wash_first_batches = []                 # the synthetic stream's own batches
    evl_w = copy.deepcopy(net0)
    for s_wh in range(1, WASH_HIST_STEPS + 1):
        aj_ = torch.randint(16, (ANCH_BS,), generator=wgen)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=wgen)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        y_ = torch.cat([anc_[:, 1:], rnd_[:, 1:]], 0)
        if s_wh <= TRAIN_STEPS:
            wash_first_batches.append((x_.clone(), y_.clone()))
        logits_w, _ = wnet(x_)
        lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                             y_.reshape(-1))
        wopt.zero_grad(set_to_none=True)
        lw.backward()
        torch.nn.utils.clip_grad_norm_(wnet.parameters(), CLIP)
        wopt.step()
        th_new = flat_params(wnet)
        segs.append(th_new - wtheta)
        load_flat(evl_w, th_new)
        gz = battery_cell(evl_w, primary_ids, zid)
        gp = battery_cell(evl_w, coruler_ids[12], zid)
        g0 = battery_cell(evl_w, coruler_ids[0], zid)
        hist_rows.append({"step": s_wh,
                          "L2": float(torch.norm(th_new - wtheta)),
                          "cum": float(torch.norm(th_new - theta0)),
                          "ce": float(lw.item()),
                          "g_m12_mean_pz": gz["mean_pz"],
                          "g0_mean_pz": g0["mean_pz"],
                          "g_p12_mean_pz": gp["mean_pz"]})
        wtheta = th_new
    wnet.eval()
    del wnet, wopt, evl_w
    H = torch.stack(segs)
    hist_dead = next((h["step"] for h in hist_rows
                      if h["g_m12_mean_pz"] <= SHUT_BAR), None)
    if not SMOKE:
        devs = [abs(h["L2"] - carm_rows[h["step"]]["step_disp"])
                for h in hist_rows]
        max_dev = max(devs)
        G_HIST = {
            "gate": "per-step displacement L2 vs g1b's committed C-arm rows "
                    "1.." + str(WASH_HIST_STEPS),
            "max_abs_dev_L2": max_dev,
            "per_step_dev_first3": [round(d, 8) for d in devs[:3]],
            "tol": G_READ_TOL,
            "committed_death_step": 2,
            "measured_death_step": hist_dead,
            "pass": bool(max_dev < G_READ_TOL),
            "note": "the pristine contiguous history IS g1b's C-arm wash "
                    "(same stream, same recipe); per-step anchored vs the "
                    "committed CUDA rows (cross-device texture tier)",
        }
    else:
        G_HIST = {"gate": "smoke (4-step trim)", "pass": True,
                  "measured_death_step": hist_dead}
    log(f"  G_HIST: max|dev| {G_HIST.get('max_abs_dev_L2', float('nan')):.2e}; "
        f"fact died in-history at step {hist_dead}: "
        + ("PASS" if G_HIST["pass"] else "FAIL"))
    if not G_HIST["pass"]:
        raise RuntimeError("history gate FAILED")
    metrics["gates"]["G_HIST"] = G_HIST

    basis = svd_basis(H)
    sv = [float(s) for s in basis["sv"]]
    Vp = basis["Vp"]
    pr_rel = abs(basis["pr"] - E211_P_COMMITTED["span_pr"]) / E211_P_COMMITTED["span_pr"]
    sv0_rel = abs(sv[0] - E211_P_COMMITTED["span_sv0"]) / E211_P_COMMITTED["span_sv0"]
    G_SPAN = {
        "gate": "the fresh span PR must reproduce e211's committed root-P PR "
                "(CPU->CPU, same threads)",
        "pr_measured": basis["pr"], "pr_committed_e211":
            E211_P_COMMITTED["span_pr"], "pr_rel_dev": pr_rel,
        "sv0_measured": sv[0], "sv0_committed_e211":
            E211_P_COMMITTED["span_sv0"], "sv0_rel_dev": sv0_rel,
        "tol": G_PR_TOL,
        "pass": bool(SMOKE or pr_rel < G_PR_TOL),
        "note": "this run's span IS e211's root-P span (the ordering's own "
                "axes) — the reproduction is the gate; sv0 rides as a "
                "report (fp64 eigh degeneracy can permute near-equal SVs; "
                "PR is permutation-invariant)",
    }
    log(f"  G_SPAN: PR {basis['pr']:.6f} vs e211 {E211_P_COMMITTED['span_pr']:.6f} "
        f"(rel {pr_rel:.1e}); sv0 rel {sv0_rel:.1e}: "
        + ("PASS" if G_SPAN["pass"] else "FAIL"))
    if not G_SPAN["pass"]:
        raise RuntimeError("span reproduction gate FAILED")
    metrics["gates"]["G_SPAN"] = G_SPAN

    cum20 = wtheta - theta0
    drift = {f"dir{i}": float(torch.dot(cum20.double(),
                                        (Vp[i] / torch.norm(Vp[i])).double()))
             for i in (DIR_TOP, DIR_MID)}
    span_block = {
        "n_segments": int(Vp.shape[0]), "rank_eff": basis["rank_eff"],
        "sv": sv, "participation_ratio": basis["pr"], "cond": basis["cond"],
        "sv_top": sv[0], "energy_frac_top1": sv[0] ** 2 / sum(s * s for s in sv),
        "profile": [s / sv[0] for s in sv],
        "energy_frac_dirs": {f"dir{i}": sv[i] ** 2 / sum(s * s for s in sv)
                             for i in (DIR_TOP, DIR_MID)},
        "wash_drift_proj_dirs": drift,
        "history": {
            "n_steps": WASH_HIST_STEPS, "rows": hist_rows,
            "step1_L2": hist_rows[0]["L2"],
            "cum_final": hist_rows[-1]["cum"],
            "step1_L2_vs_G_STREAM": abs(hist_rows[0]["L2"] - disp1),
            "fact_died_in_history_at_step": hist_dead,
            "note": "the wash-history trace (the span's source AND the "
                    "synthetic stream's own batches); the fact dies at step "
                    "2 in the REAL wash — the projected stream is the wash's "
                    "gradient field constrained to one axis",
        },
        "note": "the contiguous wash-history span (e211/e223's machinery at "
                "the e131 root, reproduced under G_SPAN); drift = "
                "<theta20-theta0, v_i> (the wash's own drift along each "
                "axis, sign vs the SVD-emitted +v)",
    }
    log(f"  span: PR {basis['pr']:.4f}, cond {basis['cond']:.2f}, top SV "
        f"{sv[0]:.4f} (energy {span_block['energy_frac_top1']:.3f}); "
        f"drift along v0 {drift[f'dir{DIR_TOP}']:+.3f}")
    metrics["span"] = span_block
    write_partial("P1 span done (PR %.6f, reproduced)" % basis["pr"])
    del H, segs, wtheta, cum20

    # ================= P2: the un-exposed baselines ============================
    log("=" * 78)
    log("P2: the un-exposed baseline kill rays (the BEFORE rows)")

    def unit(i: int) -> torch.Tensor:
        return (Vp[i] / torch.norm(Vp[i])).clone()

    v_top, v_mid = unit(DIR_TOP), unit(DIR_MID)
    evl = copy.deepcopy(net0)

    @torch.no_grad()
    def read_state(theta: torch.Tensor) -> dict:
        load_flat(evl, theta)
        out = {"primary": battery_cell(evl, primary_ids, zid)["mean_pz"]}
        for j in CO_RULERS_J:
            out[f"coruler_g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        out["ce_r"] = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
        return out

    def primary_read(theta: torch.Tensor) -> float:
        load_flat(evl, theta)
        return battery_cell(evl, primary_ids, zid)["mean_pz"]

    def walk_kill(theta_s: torch.Tensor, v: torch.Tensor, tag: str):
        """e211/e222/e223's onset-grid kill walk from theta_s along -v (early stop)."""
        rows, kill = [], None
        load_flat(evl, theta_s)
        gz0 = battery_cell(evl, primary_ids, zid)
        rows.append({"D": 0.0, "gm": gz0["mean_pz"],
                     "frac": gz0["frac_argmax_z"]})
        prev = gz0["mean_pz"]
        for D in D_GRID:
            load_flat(evl, theta_s - D * v)
            gz = battery_cell(evl, primary_ids, zid)
            rows.append({"D": float(D), "gm": gz["mean_pz"],
                         "frac": gz["frac_argmax_z"]})
            if gz["mean_pz"] <= SHUT_BAR and prev > SHUT_BAR:
                kill = interp_d_kill(prev, gz["mean_pz"], float(D) - 0.05,
                                     float(D))
                break
            prev = gz["mean_pz"]
        log(f"  [{tag}] kill "
            + (f"{kill:.4f}" if kill is not None else f"CENSORED (>{max(D_GRID)})")
            + f" ({len(rows)} grid evals)")
        return rows, kill

    # (the ctrl ray's baseline follows the accepted draw — P3b)
    base_dir_kills, baselines = {}, {}
    for tag, v in (("top", v_top), ("mid", v_mid)):
        rows, kill = walk_kill(theta0, v, f"BASE ray-{tag}")
        base_dir_kills[tag] = {"dir": {"top": DIR_TOP, "mid": DIR_MID}[tag],
                               "energy_frac": sv[{"top": DIR_TOP,
                                                  "mid": DIR_MID}[tag]] ** 2
                               / sum(s * s for s in sv),
                               "D_kill": kill, "rows": rows}
        baselines[tag] = kill
    metrics["baselines"] = base_dir_kills
    write_partial("P2 baseline rays done (top/mid; the ctrl ray follows the "
                  "accepted draw)")

    t_top = baselines["top"]
    if t_top is None:
        metrics["dose_gate_failure"] = (
            "the top-SV baseline ray is CENSORED past the grid — the "
            "same-axis arithmetic cannot be set; cell aborts (e187)")
        write_partial("P2 BASELINE GATE FAILED (top ray censored)")
        raise RuntimeError("top-SV baseline kill-D unresolved")

    # ================= P3: the training arms ===================================
    log("=" * 78)
    log(f"P3: THE TRAINING EXPOSURE — {TRAIN_STEPS} AdamW steps per arm on the "
        f"wash's own batches, gradients projected onto the target direction")

    # G_SYNTH: the synthetic stream's batches ARE the wash's own first batches
    sgen = torch.Generator().manual_seed(FREEZE_SEED)
    synth_batches = []
    for _ in range(TRAIN_STEPS):
        aj_ = torch.randint(16, (ANCH_BS,), generator=sgen)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=sgen)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        synth_batches.append((torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0),
                              torch.cat([anc_[:, 1:], rnd_[:, 1:]], 0)))
    synth_md5 = [hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
                 for x_, _ in synth_batches]
    wash_md5 = [hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
                for x_, _ in wash_first_batches]
    G_SYNTH = {
        "gate": "the synthetic stream's batches are bitwise the wash's own "
                "first two batches (fresh seed-10902 draw vs the P1 capture)",
        "synth_md5": synth_md5, "wash_md5": wash_md5,
        "bitwise_match": bool(synth_md5 == wash_md5),
        "step1_x_md5_match_e185": bool(synth_md5[0] == E185_XHASH_1),
        "pass": bool(synth_md5 == wash_md5 and synth_md5[0] == E185_XHASH_1),
        "note": "'the wash's own batch gradients' — the same stream, the "
                "same order; the projection arithmetic g~ = (g.v)v with "
                "live gradients is documented in training_arms.*.steps",
    }
    log(f"  G_SYNTH: batches bitwise-identical to the wash's own: "
        + ("PASS" if G_SYNTH["pass"] else "FAIL"))
    if not G_SYNTH["pass"]:
        raise RuntimeError("synthetic stream gate FAILED")
    metrics["gates"]["G_SYNTH"] = G_SYNTH

    # ---- the training trajectories + the draw-acceptance rung ladder --------
    rungs = [{"idx": k + 1, "frac": f, "T": f * T_WASH_STEP1}
             for k, f in enumerate(RUNG_FRACS)]

    def raw_stats(d_raw, v_t, step_rows):
        cos_raw = float(torch.dot(d_raw.double(), v_t.double())
                        / (torch.norm(d_raw.double())
                           * torch.norm(v_t.double()) + 1e-30))
        return {
            "d_raw_L2": float(torch.norm(d_raw)),
            "cos_d_vs_target": cos_raw,
            "on_axis_c_raw": float(torch.dot(d_raw.double(), v_t.double())),
            "steps": step_rows,
        }

    raw_trajs: dict = {}
    d_raws: dict = {}
    d_v0, steps_v0 = train_projected(net0, theta0, synth_batches, v_top)
    d_raws["V0-TRAIN"] = d_v0
    raw_trajs["V0-TRAIN"] = raw_stats(d_v0, v_top, steps_v0)
    log(f"  V0-TRAIN: raw 2-step displacement "
        f"{raw_trajs['V0-TRAIN']['d_raw_L2']:.4f} L2 "
        f"(cos vs v0 {raw_trajs['V0-TRAIN']['cos_d_vs_target']:+.3f}; "
        "a-coeffs "
        + ", ".join(f"{s['proj_coeff_a']:+.3f}"
                    for s in raw_trajs["V0-TRAIN"]["steps"]) + ")")

    # the ctrl-draw acceptance loop: sequential randn draws from CTRL_SEED;
    # accept the first draw for which some rung leaves BOTH arms alive
    # under the gamma cap (all disclosed)
    gen_c = torch.Generator().manual_seed(CTRL_SEED)
    draw_records, accepted, u_ctrl = [], None, None
    for k_draw in range(1, CTRL_MAX_DRAWS + 1):
        coeffs = torch.randn(Vp.shape[0], generator=gen_c)
        u_k = torch.mv(Vp.T, coeffs)
        u_k = (u_k / torch.norm(u_k)).clone()
        d_ck, steps_ck = train_projected(net0, theta0, synth_batches, u_k)
        st_k = raw_stats(d_ck, u_k, steps_ck)
        ladder_rows, acc_row = [], None
        for rung in rungs:
            T = rung["T"]
            gv = T / raw_trajs["V0-TRAIN"]["d_raw_L2"]
            gc = T / st_k["d_raw_L2"]
            row = {"rung": rung["idx"], "frac_of_wash_step1": rung["frac"],
                   "T_L2": T,
                   "gamma_v0": gv, "gamma_ctrl": gc,
                   "within_gamma_cap": bool(gv <= GAMMA_CAP and gc <= GAMMA_CAP)}
            if row["within_gamma_cap"]:
                reads = {"V0-TRAIN": primary_read(theta0 + gv * d_v0),
                         "CTRL-TRAIN": primary_read(theta0 + gc * d_ck)}
                row["reads"] = reads
                row["both_alive"] = bool(all(v > SHUT_BAR
                                             for v in reads.values()))
            else:
                row["both_alive"] = False
                row["note"] = "gamma cap — the endpoint would extrapolate " \
                              f"beyond {GAMMA_CAP}x the trained net"
            ladder_rows.append(row)
            log(f"  draw {k_draw} rung {rung['idx']} (T = "
                f"{rung['frac']}x wash-step1 = {T:.4f} L2): V0-TRAIN "
                + (f"reads {row['reads']['V0-TRAIN']:.4f}, CTRL-TRAIN reads "
                   f"{row['reads']['CTRL-TRAIN']:.4f}"
                   if "reads" in row else "(gamma-capped)")
                + " -> " + ("BOTH ALIVE (accepted)"
                            if row["both_alive"] else "not both"))
            if row["both_alive"]:
                acc_row = row
                break
        draw_records.append({
            "draw": k_draw,
            "target": f"u (random in-span unit, seed {CTRL_SEED} draw "
                      f"{k_draw})",
            "raw": {kk: st_k[kk] for kk in ("d_raw_L2", "cos_d_vs_target",
                                            "on_axis_c_raw")},
            "steps": steps_ck,
            "ladder": ladder_rows,
            "accepted": bool(acc_row is not None),
        })
        if acc_row is not None:
            u_ctrl, d_ctrl, ctrl_k = u_k, d_ck, k_draw
            accepted = acc_row
            raw_trajs["CTRL-TRAIN"] = st_k
            d_raws["CTRL-TRAIN"] = d_ck
            break
        del u_k, d_ck
    metrics["training_arms_raw"] = raw_trajs
    metrics["rung_ladder"] = {
        "anchor": "the wash's committed step-1 L2 (g1b C-arm row 1)",
        "T_wash_step1": T_WASH_STEP1,
        "rule": "the ctrl direction = the first of <= "
                f"{CTRL_MAX_DRAWS} sequential seed-{CTRL_SEED} draws for "
                "which the FIRST rung of the ladder leaves BOTH arms' "
                "scaled exposed states reading > 0.27 under the "
                "extrapolation cap gamma = T/||d_raw|| <= "
                f"{GAMMA_CAP} (e223's control-acceptance convention "
                "composed with the ladder; biases the control toward "
                "forgiveness — AGAINST the IMMUNITY clause, conservative); "
                "the SAME T for both arms (matched L2); every rung of every "
                "draw disclosed",
        "draws": draw_records,
        "accepted_draw": ctrl_k if accepted is not None else None,
        "accepted_rung": accepted,
        "unscaled_2step_endpoint_reads": {
            "V0-TRAIN": primary_read(theta0 + d_v0),
            "CTRL-TRAIN": (primary_read(theta0 + d_ctrl)
                           if accepted is not None else None)},
        "unscaled_note": "the verbatim 2-step training endpoints (the "
                         "wash-scale dose) read as context only",
    }
    raw_reads = metrics["rung_ladder"]["unscaled_2step_endpoint_reads"]
    if accepted is None:
        metrics["dose_gate_failure"] = (
            f"no rung of {[r['frac'] for r in rungs]}x the wash's step-1 L2 "
            f"leaves BOTH arms' exposed states alive within {CTRL_MAX_DRAWS} "
            "ctrl draws under the gamma cap — the matched training exposure "
            "cannot be constructed; cell aborts (metrics written first — "
            "the e187 lesson)")
        write_partial("P3 RUNG LADDER FAILED (no both-alive rung in the "
                      "draw budget)")
        raise RuntimeError("rung ladder exhausted — no accepted dose")
    T_acc = accepted["T_L2"]
    log(f"  ACCEPTED draw {ctrl_k} rung {accepted['rung']} "
        f"(T = {T_acc:.4f} L2 = {accepted['frac_of_wash_step1']}x the "
        f"wash's step-1; {T_acc / E223_COMMITTED['eps']:.2f}x e223's eps "
        f"{E223_COMMITTED['eps']:.4f})")
    write_partial("P3 rung ladder accepted (ctrl draw "
                  f"{ctrl_k}, rung {accepted['rung']})")

    # ---- P3b: the accepted ctrl ray's baseline (the BEFORE row) -------------
    rows, kill = walk_kill(theta0, u_ctrl, "BASE ray-ctrl")
    base_dir_kills["ctrl"] = {"dir": f"ctrl (random in-span, seed "
                                     f"{CTRL_SEED} draw {ctrl_k})",
                              "energy_frac": None,
                              "D_kill": kill, "rows": rows}
    baselines["ctrl"] = kill
    metrics["baselines"] = base_dir_kills
    log(f"  the accepted ctrl ray's own baseline: "
        + (f"kill {kill:.4f}" if kill is not None else "CENSORED"))
    write_partial("P3b ctrl baseline done")

    ARMS = [("V0-TRAIN", v_top, {"own": "top",
                                 "target": "v0 (the span's top-SV direction)"}),
            ("CTRL-TRAIN", u_ctrl, {"own": "ctrl",
                                    "target": f"u (random in-span unit, seed "
                                              f"{CTRL_SEED} draw {ctrl_k})"})]

    # ---- the exposed states: full reads + the kill rows ----------------------
    RAYS = {"top": v_top, "mid": v_mid, "ctrl": u_ctrl}
    arm_results: dict = {}
    for arm_tag, v_t, spec in ARMS:
        d_raw = d_raws[arm_tag]
        gamma = T_acc / float(torch.norm(d_raw))
        d_ach = gamma * d_raw
        theta_exp = theta0 + d_ach
        ach_dev = abs(float(torch.norm(theta_exp - theta0)) - T_acc)
        assert ach_dev < MATCH_TOL, f"displacement match dev {ach_dev:.2e}"
        c_own = float(torch.dot(d_ach.double(), RAYS[spec["own"]].double()))
        off_axis = float(max(T_acc ** 2 - c_own ** 2, 0.0)) ** 0.5
        # the in-span fraction of the achieved displacement (fp64, all dirs)
        d64 = d_ach.double()
        in_span_sq = 0.0
        for i in range(Vp.shape[0]):
            vi = (Vp[i] / torch.norm(Vp[i])).double()
            in_span_sq += float(torch.dot(d64, vi)) ** 2
        on_ray = {r: float(torch.dot(d64, RAYS[r].double())) for r in RAYS}
        state_read = read_state(theta_exp)
        sublethal = bool(state_read["primary"] > SHUT_BAR)
        gate_cell = {
            "target": spec["target"],
            "T_L2": T_acc,
            "gamma_endpoint_scale": gamma,
            "raw_2step_L2": raw_trajs[arm_tag]["d_raw_L2"],
            "achieved_L2": float(torch.norm(theta_exp - theta0)),
            "achieved_L2_abs_dev_from_T": ach_dev,
            "match_assert_tol": MATCH_TOL,
            "on_axis_c_own": c_own,
            "off_axis_L2": off_axis,
            "cos_d_vs_own_ray": c_own / T_acc,
            "cos_d_vs_target_raw_traj": raw_trajs[arm_tag]["cos_d_vs_target"],
            "in_span_energy_frac_of_d": in_span_sq / (T_acc ** 2),
            "on_ray_components": on_ray,
            "exposed_state_primary_read": state_read["primary"],
            "corulers": {k: v for k, v in state_read.items()
                         if k.startswith("coruler")},
            "ce_r": state_read["ce_r"],
            "ce_r_delta_vs_root": state_read["ce_r"] - root_cells["ce_r"],
            "sublethal": sublethal,
            "pass": sublethal,
        }
        log(f"P3[{arm_tag}] exposed state @ T={T_acc:.4f}: primary "
            f"{state_read['primary']:.4f}, CE_R {state_read['ce_r']:.4f} "
            f"(root {root_cells['ce_r']:.4f}); on-axis c "
            f"{c_own:+.4f}, off-axis {off_axis:.4f}, cos {c_own / T_acc:+.3f}: "
            + ("PASS" if sublethal else "FAIL"))
        cell = {"exposure": arm_tag, "T": T_acc, "gate": gate_cell}
        if sublethal:
            krows = {}
            for ray_tag in ("top", "mid", "ctrl"):
                same = (ray_tag == spec["own"])
                rows_k, kill_k = walk_kill(theta_exp, RAYS[ray_tag],
                                           f"{arm_tag} ray-{ray_tag}")
                t_b = baselines[ray_tag]
                ratio = (kill_k / t_b) if (kill_k is not None
                                           and t_b is not None) else None
                # the same-axis arithmetic AT THE ACHIEVED DISPLACEMENT:
                # predicted D = t_base + c (c = the on-axis component of the
                # achieved displacement; the off-axis part is the residue's
                # content — the desk disclosure)
                arith, ceilr, resid = None, None, None
                if same and kill_k is not None and t_b is not None:
                    arith = t_b + on_ray[ray_tag]
                    ceilr = 1.0 + on_ray[ray_tag] / t_b
                    resid = kill_k - arith
                krows[ray_tag] = {
                    "D_kill": kill_k, "D_kill_baseline": t_b,
                    "arithmetic_prediction_D": arith,
                    "ratio_vs_baseline": ratio,
                    "same_axis": same,
                    "mechanical_floor_ratio": ceilr if same else None,
                    "causal_residue_D": resid if same else None,
                    "arm_on_ray_component": on_ray[ray_tag],
                    "cross_delta_D": ((kill_k - t_b) if (not same
                                     and kill_k is not None
                                     and t_b is not None) else None),
                    "rows": rows_k,
                }
                if same and kill_k is not None:
                    log(f"    same-axis row: kill {kill_k:.4f} vs base "
                        f"{t_b:.4f} -> arithmetic {arith:.4f} "
                        f"(ratio {ratio:.3f} vs mechanical {ceilr:.3f}; "
                        f"residue {resid:+.4f})")
                elif same:
                    log(f"    same-axis row: CENSORED (arithmetic {arith})")
            cell["kill_rows"] = krows
        else:
            cell["note"] = ("LETHAL at the accepted rung — impossible under "
                            "the ladder rule; disclosed")
            raise RuntimeError(f"{arm_tag} lethal at the accepted rung")
        arm_results[arm_tag] = cell
        metrics["arms"] = arm_results
        write_partial(f"P3[{arm_tag}] done")
    metrics["envelope"]["cpu_load_pct_after_P3"] = cpu_load_probe()

    # CE_R canary at the root's dir-0 kill position (context)
    load_flat(evl, theta0 - t_top * v_top)
    ce_r_at_kill = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
    metrics["ce_r_canary"] = {
        "root": root_cells["ce_r"],
        "root_dir0_kill_position": ce_r_at_kill,
        "exposed_states": {a: arm_results[a]["gate"]["ce_r"]
                           for a in arm_results},
        "note": "the organism-health honesty reflex (report-only): a training "
                "exposure that wrecks general CE while the ruler stays alive "
                "would confound any immunity reading",
    }

    # ================= P4: the adjudication (frozen bars) ======================
    log("=" * 78)
    log("P4: adjudication (the frozen bars; the desk disclosure's arithmetic "
        "carried alongside, never replacing them)")

    def row(arm, ray):
        try:
            return metrics["arms"][arm]["kill_rows"][ray]
        except KeyError:
            return None

    r_v = row("V0-TRAIN", "top")
    resid_v = r_v["causal_residue_D"] if r_v else None
    r_c = row("CTRL-TRAIN", "ctrl")
    resid_c = r_c["causal_residue_D"] if r_c else None

    # TRAINING-IMMUNITY clause (a): the own-ray residue beyond the arithmetic
    clause_a = bool(resid_v is not None and resid_v >= RESIDUE_BAR)
    clause_a_state = ("resolved: residue %+.4f %s +%0.2f"
                      % (resid_v, ">=" if clause_a else "<", RESIDUE_BAR)
                      if resid_v is not None else
                      "UNRESOLVABLE (the V0-TRAIN/top row is censored)")

    # clause (b): the both-cross differential at mid (the registered clean ray)
    rr_v = row("V0-TRAIN", "mid")
    rr_c = row("CTRL-TRAIN", "mid")
    d_v = (rr_v["D_kill"] - rr_v["D_kill_baseline"]
           if rr_v and rr_v["D_kill"] is not None else None)
    d_c = (rr_c["D_kill"] - rr_c["D_kill_baseline"]
           if rr_c and rr_c["D_kill"] is not None else None)
    if d_v is None or d_c is None:
        clause_b_mid = {"v0t_delta": d_v, "ctrlt_delta": d_c,
                        "excess": None,
                        "state": "unresolvable (a censored member of the pair)"}
        clause_b = False
    else:
        excess = d_v - d_c
        clause_b = bool(excess >= RESIDUE_BAR)
        # THE ON-RAY DECOMPOSITION (the honesty reflex, computed alongside,
        # never replacing the registered formula): unlike e223's exact-axis
        # exposures, the training displacements are not span axes — each
        # arm's displacement carries an ON-RAY component along the mid ray,
        # and under axis-separability that component shifts the mid kill-D
        # mechanically. The desk disclosure's "no mechanical loading at the
        # both-cross ray" is TESTED here, not assumed.
        on_v_mid = rr_v["arm_on_ray_component"]
        on_c_mid = rr_c["arm_on_ray_component"]
        v0t_terrain = d_v - on_v_mid
        ctrlt_terrain = d_c - on_c_mid
        corrected_excess = v0t_terrain - ctrlt_terrain
        clause_b_mid = {"v0t_delta": d_v, "ctrlt_delta": d_c,
                        "excess": excess, "fires": clause_b,
                        "on_ray_decomposition": {
                            "v0t_on_mid_component": on_v_mid,
                            "ctrlt_on_mid_component": on_c_mid,
                            "mechanical_prediction_of_excess":
                                on_v_mid - on_c_mid,
                            "v0t_terrain_part": v0t_terrain,
                            "ctrlt_terrain_part": ctrlt_terrain,
                            "ctrl_mid_mechanical_confirmation": {
                                "ctrl_delta_minus_own_on_mid_shift":
                                    d_c - on_c_mid,
                                "note": "~0 means the ctrl member IS its own "
                                        "mechanical shift (axis-separable)"},
                            "on_ray_corrected_excess": corrected_excess,
                            "corrected_fires": bool(corrected_excess
                                                    >= RESIDUE_BAR),
                            "note": ("the RAW formula adjudicates as "
                                     "registered (no bar shopping); the "
                                     "corrected form is REPORTED, never "
                                     "adjudicated")},
                        "note": "the both-cross ray, the RAW registered "
                                "formula; the on-ray decomposition rides "
                                "alongside (see deviations)"}
    imm_fires = bool(clause_a or clause_b)

    # the mechanically-loaded context rows (reported verbatim, never adjudicated)
    loaded_rows = {}
    rr_vu = row("V0-TRAIN", "ctrl")
    if rr_vu is not None and rr_vu["D_kill"] is not None:
        delta_vu = rr_vu["D_kill"] - rr_vu["D_kill_baseline"]
        excess_raw = (delta_vu - d_c) if d_c is not None else None
        excess_res = ((delta_vu - resid_c)
                      if (resid_c is not None and d_c is not None) else None)
        loaded_rows["u_ray_pair"] = {
            "v0t_delta": delta_vu, "ctrlt_delta": d_c,
            "ctrlt_same_axis_arithmetic_delta": (
                (r_c["D_kill"] - r_c["D_kill_baseline"]
                 - r_c["arm_on_ray_component"]) if r_c is not None else None),
            "raw_formula_excess": excess_raw,
            "residue_reduced_excess": excess_res,
            "mechanical_loading": (
                "the CTRL member is its own floor-loaded same-axis row "
                "(delta ~ c_u < 0): the raw formula carries +|c_u| "
                "mechanically and is NOT adjudicated (the e166 lesson); the "
                "residue-reduced form subtracts the control's own residue "
                "from the ctrl member's arithmetic part"),
        }
    rr_ct = row("CTRL-TRAIN", "top")
    if rr_ct is not None and rr_ct["D_kill"] is not None:
        loaded_rows["ctrlt_top_row"] = {
            "ctrlt_delta": rr_ct["D_kill"] - rr_ct["D_kill_baseline"],
            "note": "the reverse cross (the control's training along u, "
                    "tested on the TOP ray) — context: does random-direction "
                    "training move the top ray at all?",
        }

    # SENSITIZATION: the v0-training lowers the top-ray kill-D below arithmetic
    sens_fires = bool(resid_v is not None and resid_v <= -RESIDUE_BAR)
    sens_state = ("resolved: residue %+.4f %s -%0.2f"
                  % (resid_v, "<=" if sens_fires else ">", RESIDUE_BAR)
                  if resid_v is not None else
                  "UNRESOLVABLE (the V0-TRAIN/top row is censored)")

    # TRAINING-NULL: all arms sit on the arithmetic (+-0.01)
    residues = {"V0-TRAIN/top": resid_v, "CTRL-TRAIN/ctrl": resid_c}
    all_resolved = all(v is not None for v in residues.values())
    null_ok = bool(not imm_fires and not sens_fires and all_resolved
                   and all(abs(v) <= NULL_RESIDUE_TOL
                           for v in residues.values()))

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif imm_fires:
        verdict = "TRAINING-IMMUNITY"
        which = []
        if clause_a:
            which.append(f"clause (a): the v0-training's own-ray residue "
                         f"{resid_v:+.4f} >= +{RESIDUE_BAR} (kill-D beyond "
                         f"the pass-back arithmetic "
                         f"{r_v['arithmetic_prediction_D']:.4f})")
        if clause_b_mid.get("fires"):
            which.append(f"clause (b) at mid: the v0-training raises the cross "
                         f"ray {clause_b_mid['excess']:+.4f} more than the "
                         f"control does (>= +{RESIDUE_BAR})")
        clause = ("; ".join(which) + " — the vaccination is a TRAINING "
                  "phenomenon (the displacement was never the mechanism; "
                  "the optimizer's traversal is). "
                  "THE ON-RAY DECOMPOSITION RIDES WITH THE VERDICT: "
                  + (f"the mid excess {clause_b_mid['excess']:+.4f} "
                     f"decomposes into {clause_b_mid['on_ray_decomposition']['mechanical_prediction_of_excess']:+.4f} "
                     "mechanical (the control's own on-mid component, "
                     "axis-separable; its mid delta sits "
                     f"{clause_b_mid['on_ray_decomposition']['ctrl_mid_mechanical_confirmation']['ctrl_delta_minus_own_on_mid_shift']:+.4f} "
                     "from its own shift) + "
                     f"{clause_b_mid['on_ray_decomposition']['on_ray_corrected_excess']:+.4f} "
                     "on-ray-corrected (sub-bar: "
                     f"{clause_b_mid['on_ray_decomposition']['corrected_fires']}); "
                     f"clause (a) at {resid_v:+.4f} is sub-bar by "
                     f"{RESIDUE_BAR - resid_v:.4f}; BOTH arms' same-axis "
                     "residues are positive (the traversal gentler than the "
                     "displacement arithmetic in both) — the raw formula "
                     "governs, the decomposition governs the interpretation"
                     if clause_b_mid.get("on_ray_decomposition") else ""))
    elif sens_fires:
        verdict = "SENSITIZATION"
        clause = (f"the v0-training LOWERED the top-ray kill-D to "
                  f"{r_v['D_kill']:.4f}, residue {resid_v:+.4f} below the "
                  f"floor arithmetic {r_v['arithmetic_prediction_D']:.4f} "
                  f"(<= -{RESIDUE_BAR}) — training along a direction "
                  "SENSITIZES it (the wash's drift as the mechanism of its "
                  "own fragility) — an inversion worth naming")
    elif null_ok:
        verdict = "TRAINING-NULL"
        clause = ("all arms sit on the arithmetic: both same-axis residues "
                  "within +-0.01 ("
                  + "; ".join(f"{k} {v:+.4f}" for k, v in residues.items())
                  + ") — the exposure-immunity reading retires at the "
                  "training mechanism too; the ordering is descriptive "
                  "geometry, full stop; the arc closes")
    else:
        verdict = "GRADED"
        off = [f"{k} {v:+.4f}" for k, v in residues.items()
               if v is None or abs(v) > NULL_RESIDUE_TOL]
        clause = ("a partial — the tables verbatim; residues off the "
                  "+-0.01 arithmetic: " + ("; ".join(off) if off else "none")
                  + f"; IMMUNITY (a) {clause_a_state}; clause (b) mid "
                  + (f"excess {clause_b_mid['excess']:+.4f}"
                     if clause_b_mid.get("excess") is not None else
                     "unresolvable")
                  + f"; SENSITIZATION {sens_state}")

    metrics["adjudication"] = {
        "bars": {"TRAINING_IMMUNITY": {"fires": imm_fires,
                                        "clause_a_own_ray": clause_a,
                                        "clause_b_cross_vs_ctrl_mid": clause_b,
                                        "clause_a_state": clause_a_state,
                                        "clause_b_mid": clause_b_mid,
                                        "loaded_context_rows": loaded_rows},
                 "SENSITIZATION": {"fires": sens_fires,
                                   "state": sens_state},
                 "TRAINING_NULL": {"fires": bool(null_ok and not SMOKE)},
                 "GRADED": {"fires": bool(not imm_fires and not sens_fires
                                          and not null_ok and not SMOKE)}},
        "inputs": {
            "residue_v0_train_top": resid_v,
            "residue_ctrl_train_ctrl": resid_c,
            "same_axis_residues": residues,
            "T_accepted": T_acc,
            "T_frac_of_wash_step1": accepted["frac_of_wash_step1"],
            "c_v0_on_axis": arm_results["V0-TRAIN"]["gate"]["on_axis_c_own"],
            "c_u_on_axis": arm_results["CTRL-TRAIN"]["gate"]["on_axis_c_own"],
            "floor_arithmetic_v0t_top": (r_v["arithmetic_prediction_D"]
                                         if r_v else None),
            "floor_arithmetic_ctrlt_ctrl": (r_c["arithmetic_prediction_D"]
                                            if r_c else None),
            "clause_b_mid": {"raw_excess": clause_b_mid.get("excess"),
                             "on_ray_corrected_excess":
                                 (clause_b_mid.get("on_ray_decomposition")
                                  or {}).get("on_ray_corrected_excess")},
            "all_ratios": {f"{a}/{r}": rr["ratio_vs_baseline"]
                           for a in arm_results
                           for r, rr in metrics["arms"][a]
                           .get("kill_rows", {}).items()},
        },
        "verdict": verdict,
        "clause": clause,
        "composite_order": "TRAINING-IMMUNITY / SENSITIZATION / TRAINING-NULL "
                           "/ GRADED (frozen before compute)",
        "desk_arithmetic_recap": (
            f"the training exposures flow along the DRIFT side (c_v0 "
            f"{arm_results['V0-TRAIN']['gate']['on_axis_c_own']:+.4f}, c_u "
            f"{arm_results['CTRL-TRAIN']['gate']['on_axis_c_own']:+.4f}): "
            "the same-axis arithmetic is the FLOOR form t_base + c at the "
            "ACHIEVED displacement; the off-axis component (the normalizer's "
            "geometry) is the residue's content; the cross differential is "
            "adjudicated at the both-cross mid ray only"),
    }
    log(f"VERDICT: {verdict} — {clause}")
    write_partial("P4 adjudicated")

    # ================= the figure ==============================================
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.5))
    fig.suptitle("E224 — THE TRAINING-STEP EXPOSURE (2 AdamW steps on the "
                 f"wash's own projected stream, the pristine e131 root, CPU) "
                 f"| T = {accepted['frac_of_wash_step1']}x wash-step1 = "
                 f"{T_acc:.3f} L2 | verdict: {verdict}", fontsize=11)

    ax = axes[0][0]
    ax.semilogy(range(1, len(sv) + 1), sv, "o-", ms=4, color="tab:gray")
    for i, c, nm in ((DIR_TOP, "tab:red", "top (d0)"),
                     (DIR_MID, "tab:purple", "mid (d9)")):
        ax.semilogy(i + 1, sv[i], "o", ms=9, color=c, label=nm)
    ax.set_xlabel("span direction (by SV rank)")
    ax.set_ylabel("singular value (fp64)")
    ax.set_title(f"the wash-span spectrum (PR {basis['pr']:.2f}, e211's root-P "
                 f"reproduced); drift along v0 {drift[f'dir{DIR_TOP}']:+.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for arm_tag, c in (("V0-TRAIN", "tab:red"), ("CTRL-TRAIN", "tab:green")):
        st = raw_trajs[arm_tag]["steps"]
        ax.plot([s["step"] for s in st], [s["proj_coeff_a"] for s in st],
                "o-", color=c, label=f"{arm_tag}: proj coeff a_i")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("training step (the wash's own batches)")
    ax.set_ylabel("(g_i . v_target)  [fp64]")
    ax.set_title("the synthetic stream: the wash's own gradients' "
                 "projections (live, sign-disclosing)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][2]
    acc_ladder = [r for r in draw_records[ctrl_k - 1]["ladder"]
                  if "reads" in r]
    for arm_tag, c in (("V0-TRAIN", "tab:red"), ("CTRL-TRAIN", "tab:green")):
        xs = [r["T_L2"] for r in acc_ladder]
        ys = [r["reads"][arm_tag] for r in acc_ladder]
        ax.plot(xs, ys, "o-", color=c,
                label=f"{arm_tag} (ctrl draw {ctrl_k}, scaled states)")
        ax.plot([raw_trajs[arm_tag]["d_raw_L2"]],
                [raw_reads[arm_tag]], "x", ms=10, color=c,
                label=f"{arm_tag} verbatim 2-step endpoint")
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8, label="kill bar 0.27")
    ax.axvline(T_acc, color="tab:blue", ls=":", lw=1.5,
               label=f"accepted T {T_acc:.3f}")
    ax.axvline(E223_COMMITTED["eps"], color="tab:gray", ls="--", lw=1,
               label=f"e223's eps {E223_COMMITTED['eps']:.3f}")
    ax.set_xlabel("exposure displacement L2 (gamma-scaled endpoint)")
    ax.set_ylabel("primary ruler (g-12 install-60 mean pZ)")
    ax.set_title("the shared rung ladder: the matched dose binds at BOTH "
                 "arms' survival")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    b_rows = metrics["baselines"]["top"]["rows"]
    ax.plot([r["D"] for r in b_rows], [r["gm"] for r in b_rows], "o-",
            ms=3, color="tab:red", label="baseline walk from theta0")
    if r_v:
        _lab = (f"V0-TRAIN state (ratio {r_v['ratio_vs_baseline']:.3f})"
                if r_v["ratio_vs_baseline"] is not None
                else "V0-TRAIN state (CENSORED)")
        ax.plot([r["D"] for r in r_v["rows"]], [r["gm"] for r in r_v["rows"]],
                "s-", ms=3, color="tab:orange", label=_lab)
        cv = r_v["arm_on_ray_component"]
        ax.plot([r["D"] + cv for r in b_rows], [r["gm"] for r in b_rows],
                "--", lw=1, color="tab:orange", alpha=0.7,
                label=f"the mechanical shift ({cv:+.3f}, the floor)")
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8)
    ax.set_xlabel("D (L2 along the walk)")
    ax.set_ylabel("primary ruler")
    if r_v is not None and resid_v is not None:
        ax.set_title(f"THE SAME-AXIS TEST (top ray): residue {resid_v:+.4f} "
                     f"vs floor {r_v['arithmetic_prediction_D']:.3f}")
    else:
        ax.set_title("THE SAME-AXIS TEST (top ray): CENSORED")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    bc_rows = metrics["baselines"]["ctrl"]["rows"]
    ax.plot([r["D"] for r in bc_rows], [r["gm"] for r in bc_rows], "o-",
            ms=3, color="tab:green", label="baseline walk (u ray)")
    if r_c:
        _lab = (f"CTRL-TRAIN state (ratio {r_c['ratio_vs_baseline']:.3f})"
                if r_c["ratio_vs_baseline"] is not None
                else "CTRL-TRAIN state (CENSORED)")
        ax.plot([r["D"] for r in r_c["rows"]], [r["gm"] for r in r_c["rows"]],
                "s-", ms=3, color="tab:olive", label=_lab)
        cu = r_c["arm_on_ray_component"]
        ax.plot([r["D"] + cu for r in bc_rows], [r["gm"] for r in bc_rows],
                "--", lw=1, color="tab:olive", alpha=0.7,
                label=f"the mechanical shift ({cu:+.3f}, the floor)")
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8)
    ax.set_xlabel("D (L2 along the walk)")
    ax.set_ylabel("primary ruler")
    ax.set_title(f"the control's own axis: residue "
                 + (f"{resid_c:+.4f}" if resid_c is not None else "CENSORED"))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][2]
    bm_rows = metrics["baselines"]["mid"]["rows"]
    colors = {"V0-TRAIN": "tab:red", "CTRL-TRAIN": "tab:green"}
    ax.plot([r["D"] for r in bm_rows], [r["gm"] for r in bm_rows], "o-",
            ms=3, color="tab:gray", label="baseline (theta0)")
    for arm in ("V0-TRAIN", "CTRL-TRAIN"):
        rr = row(arm, "mid")
        if rr:
            _lab = (f"{arm} (delta {rr['cross_delta_D']:+.3f})"
                    if rr["cross_delta_D"] is not None
                    else f"{arm} (CENSORED)")
            ax.plot([r["D"] for r in rr["rows"]], [r["gm"] for r in rr["rows"]],
                    "s-", ms=3, color=colors[arm], label=_lab)
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8)
    ax.set_xlabel("D (L2 along the walk)")
    ax.set_ylabel("primary ruler")
    ax.set_title("THE CROSS TEST (dir-9, both-cross): the differential "
                 + (f"{clause_b_mid['excess']:+.4f}"
                    if clause_b_mid.get("excess") is not None else
                    "unresolvable"))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = rd / "e224_training_exposure.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log(f"figure -> {fig_path}")

    # ================= provenance + honesty + final write ======================
    metrics["provenance"] = {
        "parents_files_md5": parents_md5,
        "machinery": {
            "e131_gates": "the root's committed provenance hard-bound from "
                          "runs/e131/metrics.json (G_SPLICE 19+41, G_E048 "
                          "0.5563 bit-read, fired RE-KEYED)",
            "wash_span_kill": "lab/e223_exposure_e131.py VERBATIM as the "
                              "chassis (itself lab/e211_walled_band.py: the "
                              "pristine-root load + md5 gate, the contiguous "
                              "20-step unwalled AdamW wash on the seed-10902 "
                              "g1b C-arm stream with per-step anchors, the "
                              "fp64 Gram-SVD basis, the onset-grid kill walk "
                              "with first-downcrossing interpolation, "
                              "SVD-emitted signs, install-60 g-12 ruler, "
                              "grid 0.05..3.50)",
            "the_training_cell": "NEW this cell (registered): the "
                                 "projected-stream training arms (2 AdamW "
                                 "steps per arm on the wash's own batches "
                                 "1-2, gradients replaced by their "
                                 "projections, live, project-then-clip, the "
                                 "wash's own optimizer settings); the "
                                 "displacement matching (gamma-scaled "
                                 "endpoints, the shared rung ladder "
                                 "{1x,1/2,1/4,1/8} x the wash's step-1 L2, "
                                 "first both-alive rung, asserted to 1e-5); "
                                 "the floor-form arithmetic at the ACHIEVED "
                                 "displacement; the training adjudication",
            "committed_records": "e131 (the root provenance), e211 (the "
                                 "root-P span + dir-0 kill + the ordering "
                                 "Spearmans), e222 (the NULL), e223 (the "
                                 "NULL-CONFIRMS + its four residues + eps + "
                                 "the drift — this cell's direct parent), "
                                 "e_chart (the rung-0 root read), g1b (the "
                                 "C-arm stream anchors) — loaded, "
                                 "hard-bound, never re-run",
        },
        "registry_note": f"the ctrl-direction seed {CTRL_SEED} is fresh "
                         "(e222 drew 12501, e223 12502; the registry moves "
                         "forward)",
    }
    metrics["honesty"] = {
        "n_and_scope": ("n=1 organism (the pristine e131 root = e211's root "
                        "P = e223's organism, one lineage); one span "
                        "realization (THE draw e211's ordering was measured "
                        "on, PR-gated); ONE accepted dose (the accepted "
                        "draw's first both-alive rung); the control "
                        f"direction = draw {ctrl_k} of a {CTRL_MAX_DRAWS}-"
                        "draw acceptance budget (all disclosed); every kill "
                        "ray walked once"),
        "synthetic_stream_caveat": ("the projections are the wash's "
                                    "gradients CONSTRAINED, not the wash "
                                    "itself: the full-gradient path (its "
                                    "20-axis interference), the mixed-batch "
                                    "dynamics beyond 2 steps, and the "
                                    "state-dependence beyond the exposure's "
                                    "own trajectory are absent — this cell "
                                    "tests immunity-to-TRAINING-ALONG-ONE-"
                                    "AXIS, not immunity-to-the-wash"),
        "endpoint_scaling": ("AdamW's gradient-scale invariance (the "
                             "normalizer sets the pace — T139/g12) means the "
                             "matched dose is set on the DISPLACEMENT: the "
                             "2-step trajectories run verbatim at lr 1e-3 "
                             "and their NET displacements are gamma-scaled "
                             "to the accepted rung T under the extrapolation "
                             f"cap gamma <= {GAMMA_CAP} (the scaled endpoint "
                             "never more than doubles the trained net); the "
                             "path shape (the normalizer's sign-pattern "
                             "geometry) is preserved, the endpoint length is "
                             "not the verbatim 2-step length"),
        "control_selection": (f"the control direction was accepted as draw "
                              f"{ctrl_k} of the {CTRL_MAX_DRAWS}-draw budget "
                              f"(seed {CTRL_SEED}) — the first draw for which "
                              "the shared ladder found a both-alive rung; "
                              "the acceptance biases the control toward "
                              "forgiveness, i.e. AGAINST the IMMUNITY clause "
                              "(conservative); the discarded draws' full "
                              "ladders ride in rung_ladder.draws"),
        "shared_T_binding": (f"the matched-dose constraint binds at BOTH "
                             f"arms' survival: the accepted T "
                             f"{T_acc:.4f} = "
                             f"{accepted['frac_of_wash_step1']}x the wash's "
                             f"step-1 L2 {T_WASH_STEP1:.4f} "
                             f"({T_acc / T_WASH_STEP1:.2f}x; "
                             f"{T_acc / E223_COMMITTED['eps']:.2f}x e223's "
                             f"eps {E223_COMMITTED['eps']:.4f}) — the "
                             "control's own fragility (random in-span rays "
                             "kill at ~0.58 vs the top ray's 2.76) is what "
                             "caps the shared dose; every rung disclosed"),
        "off_axis_content": ("the same-axis arithmetic uses ONLY the on-axis "
                             "component c; the training displacement's "
                             "off-axis component (the normalizer's "
                             "geometry, cos(d, own ray) "
                             f"{arm_results['V0-TRAIN']['gate']['cos_d_vs_own_ray']:+.3f} "
                             "at the v0 arm) is NOT in the arithmetic — the "
                             "residue carries it: residue = 0 means the "
                             "traversal adds nothing beyond "
                             "position-along-axis"),
        "clause_b_loading": ("clause (b) is adjudicated at the both-cross "
                             "mid ray with the RAW registered formula; the "
                             "u-ray pair and the ctrl-arm's top row are "
                             "mechanically loaded by their own-axis "
                             "arithmetic and are reported verbatim never "
                             "adjudicated (the e166 lesson applied "
                             "preemptively); the mid row ITSELF turned out "
                             "mechanically loaded through the control's "
                             "own on-mid displacement component (see "
                             "deviations + the on_ray_decomposition): the "
                             "raw fire stands as registered, the "
                             "decomposition governs the interpretation"),
        "off_axis_tolerization": (
            "BOTH arms' same-axis residues are POSITIVE (v0 "
            f"{resid_v:+.4f} at off-axis {arm_results['V0-TRAIN']['gate']['off_axis_L2']:.3f} L2; "
            f"ctrl {resid_c:+.4f} at off-axis "
            f"{arm_results['CTRL-TRAIN']['gate']['off_axis_L2']:.3f} L2) — "
            "e222/e223's exact-axis exposures (off-axis = 0) sat at "
            "+-0.0002; the training traversals' off-axis (normalizer) "
            "geometry TOLERIZES the rays, in BOTH arms, roughly scaling "
            "with the off-axis L2 — a traversal-is-gentler-than-"
            "displacement texture that is NOT v0-specific and bounds any "
            "vaccination-specific reading"),
        "ce_r_canary": metrics["ce_r_canary"],
        "wash_context": (f"the REAL wash on this stream kills the fact at "
                         f"step 2 (g1b's committed death step; e223's "
                         f"history) — the projected stream at the same "
                         f"scale reads "
                         f"{raw_reads['V0-TRAIN']:.4f} (v0 arm) / "
                         f"{raw_reads['CTRL-TRAIN']:.4f} (ctrl arm) at its "
                         f"verbatim 2-step endpoints: constraining the "
                         "gradient to one axis is already a different "
                         "organism-trajectory, disclosed"),
        "nothing_guaranteed": ("the openness was the point: the training "
                               "residues could have been anything; the "
                               f"observed outcome is '{verdict}'"),
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["elapsed_s"] = round(time.time() - T0, 1)
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log(f"E224 DONE in {time.time() - T0:.1f}s — verdict {verdict}")


if __name__ == "__main__":
    main()
