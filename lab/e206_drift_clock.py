"""E206 — THE DRIFT-RATE CLOCK (W027's named cut, question 2: does the
fact's own support-drift rate predict death time?).

WHY. W027/T171's two-rotator picture: on the half-step lineage the fact's
sensitivity ladder decorrelates SMOOTHLY and MONOTONICALLY
(0.776 -> 0.680 -> 0.564 -> 0.333 -> 0.192) hitting near-orthogonality
(0.192) EXACTLY at death (t=5) — while the wash's sign front bounced at
period 2 (the algorithm's). THE QUESTION THIS CELL OWNS: is the support's
decorrelation rate a DEATH TIMER — the fact's own watch, measurable from
gradients alone, extrapolable from EARLY t to the death step? W027 named
the cut and predicted its cheapness: eval-only, gradients only. TEST
ACROSS THE THREE LINEAGES WITH COMMITTED STATE LADDERS: org1 (the e131
root's k=1 sign walk; died t=2; e194/e195's committed states), MIRABEL
(the e193b root's; died t=3; e198/e201's committed states), and the half
lineage (the e157_f2 root's half-step wash; died t=5; e204's COMMITTED
sensitivity ladder — the only ladder on record).

BUILDS ON (directive 1): e204 (THE SENSITIVITY MACHINERY: fact_grad
VERBATIM — the fp32 gradient of mean log p(Z) over the g-12 install-60
battery, eval-mode twin, matched-point at theta_t; the FD sign-probe
instrument gate; the half lineage's committed ladder + its gates), e194/
e195 (org1's committed walk rows + the ray md5s + the saved
opt2_a_sign_s2.pt state; the fact_grad convention was MINTED on org1),
e198/e201 (MIRABEL's committed walk journal + ray md5s; e201's
multi-organism census convention for gating two organisms in one cell),
e193b/e131/e157 (the three roots), e185/e170 (the licensed seed-10902
stream + the anchor bank), e151 (org1's root gate), e199 (the disclosed
identity-TIER system). WHAT IS NEW: the sensitivity ladders s_t on org1's
and MIRABEL's walk states (never measured on either organism's WALK
states before — e194 read matched-point gradients only at the root and
theta_1), the consecutive-cosine ladders on all three lineages under ONE
convention, the drift rates, and the early-t extrapolation-vs-death
adjudication. No static profiles are re-run; the half lineage's ladder is
LOADED COMMITTED (e204's), hard-bound + asserted.

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - CLOCK-PREDICTS: "fires if early-t drift extrapolation predicts each
    lineage's death step within +-1 on all three — the support's
    decorrelation rate IS a death timer, measurable from gradients alone."
  - CLOCK-ONE-LINEAGE: "fires if the prediction works on some lineages
    but not others — the drift-as-clock is biography; reported with the
    table."
  - NO-CLOCK: "fires if the drift rates don't extrapolate to death on any
    — the near-orthogonality-at-death was one lineage's coincidence; the
    second rotation is not a timer."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * the lineages and their death steps are the COMMITTED ledgers (never
    re-derived): org1 alive t={0,1}, death t=2 (opt2/e194's gm12
    0.6786 -> 9.83e-5); MIRABEL alive t={0,1,2}, death t=3 (e198's
    0.3673 -> 0.6911 -> 0.1230); half alive t={0..4}, death t=5
    (e197/e204's). Aliveness on each organism's OWN ruler (its g-12
    install-60 battery vs the 0.27 bar — org1/MIRABEL; the half lineage's
    g-4, carried from e204's G_ALIVE).
  * s_t = e194/e195's fact_grad VERBATIM in its arithmetic: the gradient
    of mean log p(Z) over the organism's OWN g-12 install-60 battery at
    theta_t (eval-mode twin, one backward, no RNG); MATCHED-POINT: each
    s_t computed AT the state it names (T150's dual-estimator lesson).
  * the ladder c_k = cos_fp64(s_{k-1}, s_k) for k = 1..death_t; c_k is an
    ALIVE cosine iff both its states are alive; c_death (the into-death
    cosine) is the death-approach read, reported, never fed to the fit.
  * THE THRESHOLD tau = 0.20 (registered; the dispatch's "~0.2").
    DISCLOSED PROVENANCE OF tau: the only sensitivity ladder on record at
    registration time is the half lineage's (e204's, committed), whose
    death cosine is 0.192 — tau is part of the REGISTRATION, not an
    outcome; the tau-sensitivity co-read at {0.15, 0.25, 0.30} rides and
    never adjudicates.
  * THE EARLY WINDOW = the ladder's ALIVE cosines with k <= 2 (the
    dispatch's "t <= 2"). org1's early window holds ONE cosine (c_1 — its
    death step IS 2, so k=2 is the into-death read): a rate needs two
    ticks; org1's PRIMARY prediction is therefore UNDEFINED (the clock
    cannot be set on a lineage that dies inside its own early window).
    The DISPATCH-LITERAL window (k <= 2 regardless of aliveness, i.e.
    fitting org1's into-death cosine) rides as a CIRCULAR context column
    and never adjudicates (its prediction consumes the outcome row).
  * THE DRIFT RATE (PRIMARY) = the mean first difference of the early
    window's ladder IN COSINE SPACE (the dispatch's literal "first
    differences" of the ladder sequence); with the two-point windows here
    it is exactly c_2 - c_1. UNDEFINED for org1 (above).
  * THE PREDICTION (PRIMARY) = t_hat = the smallest integer t >= 1 with
    c_1 + (t - 1) * d_hat <= tau (the linear extrapolation of the early
    fit); the ERROR = t_hat - death_t; a lineage LANDS iff its prediction
    is defined AND |error| <= 1.
  * CLOCK-PREDICTS fires iff all three lineages LAND. CLOCK-ONE-LINEAGE
    fires iff not all three AND at least one LANDS. NO-CLOCK fires iff no
    lineage LANDS and at least one prediction is DEFINED. GRADED is the
    catch-all for any other partial (e.g. a gate-forced partial table).
    Composite order frozen: CLOCK-PREDICTS -> CLOCK-ONE-LINEAGE ->
    NO-CLOCK -> GRADED.
  * CO-READS (reported, never adjudicated): (i) ANGLE space — the
    rotation-rate reading (a_k = arccos(c_k); the linear extrapolation in
    angle, crossing arccos(tau); a monotone reparameterization that
    changes the linear answer, disclosed as such); (ii) tau-sensitivity
    {0.15, 0.25, 0.30}; (iii) PER-D drift — d_hat divided by the
    lineage's own step L2 (the half lineage ticks 0.458/step vs 1.654 on
    the 2.74M organisms: cross-lineage rate comparisons live in D units);
    (iv) the OBSERVED ladder-vs-tau first crossing per lineage (the
    W027-at-death replication read: does near-orthogonality-at-death
    itself replicate, unconditioned on any extrapolation?).

REGISTERED PREDICTION (frozen before compute — including what the
COMMITTED record already fixes): the half lineage's committed ladder
gives d_hat = 0.6802 - 0.7764 = -0.0962 and t_hat = 7 vs the observed
death 5 (error +2, OUTSIDE +-1) — CLOCK-PREDICTS cannot fire under the
primary protocol; this is computable from e204's committed ladder before
this cell runs anything, and is stated NOW, not discovered after (no bar
shopping). org1's primary prediction is UNDEFINED by construction (its
early window holds one tick; death t=2 is inside the window). THE LIVE
QUESTIONS: (a) does MIRABEL's fresh ladder's early-rate extrapolation
land within +-1 of its death t=3? (b) does the W027-at-death
near-orthogonality (c_death <= tau) replicate on org1 (c_2) and MIRABEL
(c_3)? The honest expected branches: CLOCK-ONE-LINEAGE (MIRABEL lands
and/or the at-death read replicates partially) or NO-CLOCK (MIRABEL
misses too). WHAT EACH ARM GUARANTEES: NOTHING — three single-stream
biographies, one battery geometry per organism, a threshold registered
from the one ladder on record; the openness is the point.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): per
organism, the parents' gate set REUSED VERBATIM (G_NAMEFREE, G_SPLICE,
G_BATTERY, G_ANCHOR, G_ROOT, G_T0, G_STREAM via the x-hash asserts, the
walk-rebuild journal gates G_WALK_O1/G_WALK_MIR at e199's disclosed
tiers, the state/ray identity gates G_SAVEDSTATE_O1/G_RAYS_O1/G_RAYS_MIR,
the alive-window gates G_ALIVE_O1/G_ALIVE_MIR vs the committed ledgers),
the committed-ladder gate G_E204LADDER (e204's consecutive cosines
hard-bound + asserted at load, never rerun), AND THE INSTRUMENT GATE
G_SENS per organism: the FD sign probe at the ROOT (HARD: moving
+eps*s_0 must RAISE the g-12 mean-log-p readout and -eps*s_0 LOWER it,
strictly, at every probe eps in {0.05, 0.02}; tiny eval bursts) + the
same probe at the LAST ALIVE state (REPORTED, not gating — e204's
convention: a failure there is a local-landscape finding, not an
instrument failure). IDENTITY TIERS (e199's disclosed convention):
BIT = md5/1e-9 where this process reproduces the parents' bits; else
TEXTURE (walk rows < 1e-3, ray mutual geometry < 1e-3); the achieved
tier STAMPED per gate, never silently weakened.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU lane is never
claimed), torch threads 4, load-check recorded at start, sequential tiny
eval bursts, PROGRESSIVE partial metrics.json writes after every phase,
n=1 per lineage.

Outputs: runs/e206/{metrics.json, e206_drift_clock.png}. No NOTES/
THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e206_drift_clock.py    (E206_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random as _random
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

SMOKE = os.environ.get("E206_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e206 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- shared cell constants (e195/e198/e201's, verbatim) ---------------------------
PRE, POST_CAP = 130, 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
RUNS = E43.REPO / "runs"
PARENTS = {
    "e194": RUNS / "e194" / "metrics.json",
    "e195": RUNS / "e195" / "metrics.json",
    "e198": RUNS / "e198" / "metrics.json",
    "e201": RUNS / "e201" / "metrics.json",
    "e204": RUNS / "e204" / "metrics.json",
    "opt2": RUNS / "opt2" / "metrics.json",
}
FREEZE_SEED = 10902               # the licensed wash-stream seed (the e176n line)
LR_ADAMW = 1e-3                   # the t=0 recipe (the wash reproduction gates)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar — absolute, verbatim
RULER_J = -12                     # org1/MIRABEL's ruler + sensitivity battery
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
CFG1 = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
N_PARAM = 2_739_072               # org1 AND MIRABEL (the e131/e193b lines)
FD_EPS = (0.05, 0.02)             # the finite-difference probe sizes (L2 units)
if SMOKE:
    FD_EPS = (0.05,)

# ---- THE CLOCK (registered, frozen before compute) --------------------------------
TAU = 0.20                        # the near-orthogonality threshold (registered)
TAU_SENS = (0.15, 0.25, 0.30)     # the tau-sensitivity co-read (never adjudicated)
EARLY_KMAX = 2                    # the early window: alive c_k with k <= 2
ERR_BAR = 1                       # the bar's "+-1"

# ---- org1 (the e131 root's k=1 sign walk; ZEPHYRA g-12; death t=2) -----------------
ORG1_ROOT_CK = "e131_consolidated_e113.pt"
ORG1_OPT2_S2_CK = "opt2_a_sign_s2.pt"        # the SAVED theta_2 (state custody)
ORG1_STEP_L2 = 1.6542880535125732            # opt1 A0's committed step-1 L2 (the matched L2)
ORG1_CE1 = 1.356567621231079                 # opt1 A0's committed step-1 batch CE
ORG1_GN1 = 0.9829167127609253                # opt1 A0's committed step-1 pre-clip grad norm
ORG1_S1 = {"ce_batch": 1.356567621231079, "cum_disp": 1.6567984819412231,
           "gm12": 0.6785961389541626, "preclip_gnorm": 0.9829167127609253,
           "step_disp": 1.6567984819412231}
ORG1_S2 = {"ce_batch": 1.8259366750717163, "cum_disp": 2.1502907276153564,
           "gm12": 9.830708586378023e-05, "preclip_gnorm": 6.194264888763428,
           "step_disp": 1.6567445993423462}
ORG1_U0_MD5 = "aac6c6d643e327939f680148772dd180"   # e192's committed R2 direction
ORG1_U1_MD5 = "c9b8da4f25c05fb7ea7701b08ccba8c0"   # e195's committed rotated ray
ORG1_MUTUAL = {"cos_u0_u1": -0.15452721980584394,
               "cos_u0_u2": 0.03151076362117073,
               "cos_u1_u2": -0.20298347429184507}
ORG1_LEDGER = {"alive_ts": [0, 1], "death_t": 2,
               "ruler_reads": {0: 0.9155886173248291, 1: 0.6785961389541626,
                               2: 9.830708586378023e-05}}
E151_ROOT = {                                  # runs/e151 'before' battery (e176n's gate)
    "base_gm12": 0.9155886769294739, "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171, "ce_r": 1.663516640663147,
}
ORG1_SPLICE_MIX = {"FLORIZEL": 19, "ELIZABETH": 41}

# ---- MIRABEL (the e193b root's k=1 sign walk; MIRABEL g-12; death t=3) -------------
MIR_ROOT_CK = "e193b_root.pt"
MIR_ROOT_MD5 = "9113c7593dbac14fb57d47f0b96c1587"
MIR_STEP_L2 = 1.6544127464294434              # e193b/e198's committed measured step L2
MIR_S1_CE = 1.531902551651001                  # e198's committed step-1 batch CE
MIR_DIAL = {"ZEPHYRA": {"g-12": 0.14864006638526917, "g-8": 0.33425578474998474,
                        "g-4": 0.4511719346046448, "g+0": 0.4618377089500427,
                        "g+4": 0.38778820633398245, "g+8": 0.4226633313310318,
                        "g+12": 0.17708639800548553},
            "MIRABEL": {"g-12": 0.6235591173171997, "g-8": 0.7508792281150818,
                        "g-4": 0.7817148566246033, "g+0": 0.42963913083076477,
                        "g+4": 0.8612411617232727, "g+8": 0.7882601020720337,
                        "g+12": 0.6515362858772278},
            "ce_r": 1.7411779165267944}
MIR_SPLICE_MIX = {"ZEPHYRA": {"FLORIZEL": 19, "ELIZABETH": 41},
                  "MIRABEL": {"FLORIZEL": 16, "ELIZABETH": 44}}
MIR_JOURNAL = [   # e198's committed walk journal (the reproduction targets)
    {"step": 1, "ce_batch": 1.531902551651001, "cum_disp": 1.6568037271499634,
     "step_disp": 1.6568037271499634, "preclip_gnorm": 0.9484267234802246,
     "gm": 0.3672811686992645},
    {"step": 2, "ce_batch": 2.314697265625, "cum_disp": 2.1239850521087646,
     "step_disp": 1.6567463874816895, "preclip_gnorm": 6.934149265289307,
     "gm": 0.6910732984542847},
    {"step": 3, "ce_batch": 2.1457536220550537, "cum_disp": 2.4923200607299805,
     "step_disp": 1.6568037271499634, "preclip_gnorm": 6.33951997756958,
     "gm": 0.12300960719585419},
]
MIR_RAY_MD5S = {"u0": "fbb878c1926c165ebdb54eef1259ab65",
                "u1": "7702614f1ff109a0f3597533b42d9934",
                "u2": "4e862a134281b94a6b1c6e4d3531186b"}
MIR_MUTUAL = {"cos_u0_u1": -0.17510781540323442,
              "cos_u0_u2": -0.015712286422230576,
              "cos_u1_u2": -0.17813847702288285}
MIR_LEDGER = {"alive_ts": [0, 1, 2], "death_t": 3,
              "ruler_reads": {0: 0.6235591173171997, 1: 0.3672811686992645,
                              2: 0.6910732984542847, 3: 0.12300960719585419}}

# ---- the half lineage (e157_f2's half-step wash; g-4 ruler; death t=5) -------------
HALF_LADDER = {   # e204's COMMITTED consecutive sensitivity cosines (hard-bound, G_E204LADDER)
    1: 0.7763956990420287, 2: 0.6802009111119693, 3: 0.5635647719069462,
    4: 0.33338362717467435, 5: 0.1916621629393665,
}
HALF_LEDGER = {"alive_ts": [0, 1, 2, 3, 4], "death_t": 5,
               "ruler": "g-4 (the f2 line's primary — e193's frozen call)",
               "step_L2": 0.4582097828388214}

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
}

G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_TEXTURE_TOL = 1e-3              # e199's disclosed cross-thread/environment tier
DKILL_L2_ASSERT = 1e-5

REGISTERED_BARS = {
    "CLOCK_PREDICTS": "CLOCK-PREDICTS: \"fires if early-t drift extrapolation "
        "predicts each lineage's death step within +-1 on all three — the "
        "support's decorrelation rate IS a death timer, measurable from "
        "gradients alone.\"",
    "CLOCK_ONE_LINEAGE": "CLOCK-ONE-LINEAGE: \"fires if the prediction works "
        "on some lineages but not others — the drift-as-clock is biography; "
        "reported with the table.\"",
    "NO_CLOCK": "NO-CLOCK: \"fires if the drift rates don't extrapolate to "
        "death on any — the near-orthogonality-at-death was one lineage's "
        "coincidence; the second rotation is not a timer.\"",
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": "lineages/death steps = the COMMITTED ledgers "
        "(org1 t=2; MIRABEL t=3; half t=5); s_t = e194/e195's fact_grad "
        "VERBATIM (mean log p(Z) over the organism's OWN g-12 install-60 "
        "battery AT theta_t, eval-mode twin, matched-point); the ladder c_k "
        "= cos_fp64(s_{k-1}, s_k), k = 1..death_t; alive c_k iff both "
        "states alive on the lineage's own ruler; THE THRESHOLD tau = 0.20 "
        "registered (the dispatch's ~0.2; DISCLOSED: the only ladder on "
        "record at registration is the half lineage's, death cosine 0.192 — "
        "tau is registration, not outcome; tau-sensitivity {0.15, 0.25, "
        "0.30} rides, never adjudicates); THE EARLY WINDOW = alive c_k with "
        "k <= 2 (the dispatch's t <= 2); org1's window holds ONE cosine "
        "(death t=2 is inside it) — its PRIMARY prediction is UNDEFINED "
        "(a rate needs two ticks); org1's DISPATCH-LITERAL window (k <= 2 "
        "including the into-death cosine) rides as a CIRCULAR context "
        "column, never adjudicates; THE DRIFT RATE (PRIMARY) = the mean "
        "first difference of the early window in COSINE space (the "
        "dispatch's literal first differences of the ladder); THE PREDICTION "
        "(PRIMARY) = the smallest integer t with c_1 + (t-1)*d_hat <= tau; "
        "a lineage LANDS iff defined AND |t_hat - death_t| <= 1; "
        "CLOCK-PREDICTS iff all three LAND; CLOCK-ONE-LINEAGE iff not all "
        "and >= 1 LANDS; NO-CLOCK iff none LANDS and >= 1 defined; GRADED "
        "the catch-all; composite order frozen CLOCK-PREDICTS -> "
        "CLOCK-ONE-LINEAGE -> NO-CLOCK -> GRADED; CO-READS never "
        "adjudicated: (i) ANGLE-space extrapolation (arccos ladder, the "
        "rotation-rate reading — a monotone reparameterization that changes "
        "the linear answer, disclosed), (ii) tau-sensitivity, (iii) per-D "
        "drift (d_hat / step_L2 — the half lineage ticks 0.458/step vs "
        "1.654: cross-lineage rates compare in D units), (iv) the OBSERVED "
        "ladder-vs-tau first crossing per lineage (the W027-at-death "
        "replication read, descriptive).",
    "registered_prediction": "frozen BEFORE compute, including what the "
        "committed record already fixes: the half lineage's committed "
        "ladder gives d_hat = -0.0962 and t_hat = 7 vs observed death 5 "
        "(error +2, OUTSIDE +-1) — CLOCK-PREDICTS cannot fire under the "
        "primary protocol (computable from e204's committed ladder before "
        "this cell runs; stated now, not discovered after — no bar "
        "shopping); org1's primary prediction is UNDEFINED by construction "
        "(one tick in its early window; the clock cannot be set on a "
        "lineage that dies inside its own early window — itself a finding, "
        "reported in the table); THE LIVE QUESTIONS: (a) does MIRABEL's "
        "fresh early-rate extrapolation land within +-1 of its death t=3? "
        "(b) does the W027-at-death near-orthogonality (c_death <= 0.2) "
        "replicate on org1 (c_2) and MIRABEL (c_3)? Honest expected "
        "branches: CLOCK-ONE-LINEAGE (MIRABEL lands and/or the at-death "
        "read replicates partially) or NO-CLOCK (MIRABEL misses too). No "
        "bar shopping.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE SPLIT (this cell's design): the half lineage's ladder is LOADED "
    "COMMITTED (e204's sensitivity_ladder geometry, hard-bound + asserted, "
    "G_E204LADDER — never rerun; e204's own convention: committed parents "
    "are loaded, not recomputed); the NEW compute is org1's and MIRABEL's "
    "sensitivity ladders (e194/e195's fact_grad convention, minted on "
    "org1, carried to MIRABEL per e201's census precedent) — eval-only "
    "walk rebuilds (2 and 3 sign steps) + one fact_grad per state + FD "
    "probes; minutes, CPU, threads 4.",
    "e194's matched-point gradient reads existed on org1 only at the root "
    "and theta_1 (the chart/e194 rows); the LADDER view (consecutive "
    "cosines of s_t along a walk) is new on org1 and MIRABEL — no "
    "committed comparator exists for these exact reads; the state identity "
    "is carried by the parents' gates (org1: OPT2_S1/S2 rows + the saved "
    "opt2_a_sign_s2.pt checkpoint + the u0/u1 ray md5s; MIRABEL: e198's "
    "journal rows + the u0/u1/u2 ray md5s) and the instrument by the FD "
    "sign probe (G_SENS, hard at each root).",
    "THREADS 4 (the owner envelope's cap), where e195/e198's committed "
    "chains ran threads 8: cross-thread/environment drift at ~1e-7 fp32 "
    "is possible, so every recompute gate is TIERED (BIT md5/1e-9 or "
    "e199's disclosed TEXTURE tier at 1e-3), the achieved tier STAMPED "
    "per gate; the half lineage's adjudicated numbers are the COMMITTED "
    "cosines (loaded, gated), never this process's texture.",
    "org1's u2 (the post-death front) is NOT re-read: the ladder needs "
    "STATES, and theta_2's custody is carried by the saved checkpoint "
    "(G_SAVEDSTATE_O1) + the committed s2 row; e194's front-trace "
    "continuation (states 3-5) is out of scope (post-death, never "
    "adjudicated).",
    "MIRABEL's walk rebuild skips e198's co-ruler reads (g-4/g0/g+12 per "
    "step) and the kill-bracket densification: nothing in this cell "
    "consumes them (the death step is the committed ledger), and the "
    "walk's identity is carried by the md5-gated batches 1-3, the "
    "arithmetic rows (ce/disp/gnorm), the primary-ruler reads, and the "
    "ray md5s — e201's own G_WALKE198 skipped the same class of reads it "
    "did not consume; disclosed.",
    "the sensitivity batteries differ in STRENGTH at their roots "
    "(disclosed, carried): org1's g-12 reads 0.9156 (the convention's "
    "home, strong), MIRABEL's 0.6236 (alive), the half lineage's 0.198 "
    "(UNDER the 0.27 bar at D=0 — e204's ruler disclosure): the ladder "
    "object is the same instrument at three different readout strengths.",
    "the CPU load-check is recorded, not gating (the envelope's 'check "
    "load first'): this cell is a 4-thread CPU eval burst of a few "
    "minutes; the GPU is never claimed.",
    "Smoke mode trims: FD eps set to {0.05} (verdict stamped SMOKE; "
    "nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e195's org1 instruments + e198/e201's MIRABEL instruments +
# e204's fact_grad/FD machinery (whose own provenance is e194/e195 — the
# e176n lineage). Copied rather than imported to own the device policy
# and the arithmetic.

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
    return {"mean_pz": float(p.mean()), "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def fact_readout(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e194/e195's fact objective, VALUE ONLY: mean log p(Z) over the
    battery (the fact_grad objective; the FD probe's readout currency)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        tot += float(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
        n += ids[i:i + bs].shape[0]
    return tot / max(n, 1)


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE SENSITIVITY INSTRUMENT (e194/e195's fact_grad VERBATIM in its
    arithmetic; e204's port): gradient of the fact battery's mean log p(Z)
    readout at the net's CURRENT weights. Consumes no RNG; tiny eval
    burst; run on the eval-mode twin."""
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
    """opt2/e193's sign_update VERBATIM: delta = -step_l2*sign(g)/||sign(g)||
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


# ---- the clock math (registered; pure functions of the ladder) --------------------

def first_cross_int(c1: float, d: float, tau: float, t_max: int = 200):
    """The smallest integer t >= 1 with c1 + (t-1)*d <= tau; None if never."""
    if c1 <= tau:
        return 1
    if d >= 0.0:
        return None
    t = 1 + math.ceil((c1 - tau) / (-d) - 1e-12)
    return t if t <= t_max else None


def first_cross_int_angle(a1: float, d_ang: float, tau: float,
                          t_max: int = 200):
    """ANGLE-space co-read: smallest integer t with a1 + (t-1)*d_ang >=
    arccos(tau); None if never."""
    tgt = math.acos(tau)
    if a1 >= tgt:
        return 1
    if d_ang <= 0.0:
        return None
    t = 1 + math.ceil((tgt - a1) / d_ang - 1e-12)
    return t if t <= t_max else None


def clock_row(name: str, ladder: dict, alive: dict, death_t: int,
              step_l2: float, committed: bool):
    """One lineage's full clock read: ladder, early fit, prediction,
    errors, and every registered co-read. Pure function of the ladder."""
    ks = sorted(ladder)
    cs = [ladder[k] for k in ks]
    early = [ladder[k] for k in ks if k <= EARLY_KMAX and alive.get(k, False)]
    d_hat = (early[1] - early[0]) if len(early) >= 2 else None
    defined = d_hat is not None
    t_hat = first_cross_int(early[0], d_hat, TAU) if defined else None
    err = (t_hat - death_t) if t_hat is not None else None
    lands = bool(defined and err is not None and abs(err) <= ERR_BAR)
    # co-reads -----------------------------------------------------------------
    literal_early = [ladder[k] for k in ks if k <= EARLY_KMAX]  # includes into-death
    d_lit = (literal_early[1] - literal_early[0]) if len(literal_early) >= 2 else None
    t_lit = first_cross_int(literal_early[0], d_lit, TAU) if d_lit is not None else None
    ang = [math.acos(min(1.0, max(-1.0, c))) * 180.0 / math.pi for c in cs]
    early_ang = [math.acos(min(1.0, max(-1.0, c))) * 180.0 / math.pi
                 for c in early]
    d_ang = (early_ang[1] - early_ang[0]) if len(early_ang) >= 2 else None
    t_ang = (first_cross_int_angle(early_ang[0], d_ang, TAU)
             if d_ang is not None else None)
    tau_hat = {}
    if defined:
        for tt in TAU_SENS:
            tau_hat[f"tau_{tt:.2f}"] = first_cross_int(early[0], d_hat, tt)
    obs_cross = next((k for k in ks if ladder[k] <= TAU), None)
    return {
        "lineage": name, "death_t": death_t, "alive_window": alive,
        "step_L2": step_l2, "ladder_committed": committed,
        "ladder": {str(k): ladder[k] for k in ks},
        "ladder_angles_deg": {str(k): a for k, a in zip(ks, ang)},
        "first_differences": {str(ks[i]): cs[i + 1] - cs[i]
                              for i in range(len(cs) - 1)},
        "monotone_decreasing": bool(all(cs[i + 1] < cs[i]
                                        for i in range(len(cs) - 1))),
        "early_window": {"kmax": EARLY_KMAX, "values": early,
                         "n_points": len(early)},
        "drift_rate_primary": d_hat,
        "prediction_primary": {"defined": defined, "t_hat": t_hat,
                               "err": err, "lands": lands},
        "literal_window_circular": {"values": literal_early, "d": d_lit,
                                    "t_hat": t_lit,
                                    "circular": bool(
                                        len(literal_early) > len(early)),
                                    "note": "the dispatch-literal k <= 2 "
                                            "window INCLUDING the into-death "
                                            "cosine when the early window is "
                                            "censored by death — CIRCULAR "
                                            "(consumes the outcome row), "
                                            "context only"},
        "co_reads_never_adjudicated": {
            "angle_space": {"early_angles_deg": early_ang, "d_deg_per_step":
                            d_ang, "t_hat_angle": t_ang,
                            "err_angle": (t_ang - death_t)
                            if t_ang is not None else None,
                            "note": "the rotation-rate reading: linear in "
                                    "arccos (a monotone reparameterization "
                                    "that changes the linear answer)"},
            "tau_sensitivity": tau_hat,
            "per_D_drift": (d_hat / step_l2) if defined else None,
            "observed_vs_tau": {"tau": TAU, "first_cross_t": obs_cross,
                                "cross_at_death": bool(obs_cross == death_t),
                                "c_death": ladder.get(death_t),
                                "note": "the W027-at-death replication read "
                                        "(descriptive): does the OBSERVED "
                                        "ladder hit near-orthogonality at "
                                        "the death step?"},
        },
    }


# ------------------------------------------------------------------ the walks

def sign_walk(net0, anchor_neutral, train_ids, itos, ruler_ids, zid, theta0,
              step_l2, n_steps, tag):
    """The k=1 sign walk, n_steps steps, on the licensed seed-10902 stream
    (e194/e195's org1 + e198's MIRABEL arithmetic VERBATIM: draw order,
    post-clip gradients, sign_update with fp64 support norm, every-step
    own-ruler read). Stashes the refresh gradients (the ray factory) and
    the endpoints (the state ladder)."""
    net = copy.deepcopy(net0)
    net.train()
    evl_a = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    prev_flat = theta0.clone()
    traj, x_hashes, fronts, endpoints = [], {}, {}, {}
    max_l2_dev = 0.0
    for step in range(1, n_steps + 1):
        aj = torch.randint(16, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            assert "ZEPH" not in txt and "MIRABEL" not in txt, \
                "name token leaked into a window"
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
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
        assert l2dev < DKILL_L2_ASSERT, f"[{tag}] per-step L2 dev {l2dev:.2e}"
        max_l2_dev = max(max_l2_dev, l2dev)
        cur = prev_flat + delta
        load_flat(net, cur)
        cum_disp = float(torch.norm(cur - theta0))
        evl_a.load_state_dict({k_: v.detach().cpu().clone()
                               for k_, v in net.state_dict().items()})
        evl_a.eval()
        gz = battery_cell(evl_a, ruler_ids, zid)
        row = {"step": step, "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"]}
        traj.append(row)
        fronts[step] = g_t.clone()     # the gradient READ AT theta_{step-1}
        endpoints[step] = cur.clone()  # the state ladder
        prev_flat = cur
        log(f"  [{tag} s{step}] ruler {row['gm']:.10f} D {cum_disp:.10f} "
            f"ce {row['ce_batch']:.6f}")
    net.eval()
    return {"traj": traj, "x_hashes": x_hashes, "fronts": fronts,
            "endpoints": endpoints, "max_l2_dev": max_l2_dev}


def fd_probe(net0, theta, s_hat, ids, zid, eps_list, hard, point_name):
    """e204's G_SENSDIR FD sign probe: the readout at theta +- eps*s_hat
    must move UP/DOWN strictly (the direction must BE the sensitivity)."""
    evl = copy.deepcopy(net0)
    out = {"point": point_name, "hard": hard, "eps": {}}
    load_flat(evl, theta)
    base = fact_readout(evl, ids, zid)
    out["zero"] = base
    ok_all = True
    for eps in eps_list:
        p = theta + eps * s_hat
        load_flat(evl, p)
        rp = fact_readout(evl, ids, zid)
        m = theta - eps * s_hat
        load_flat(evl, m)
        rm = fact_readout(evl, ids, zid)
        raises = bool(rp > base > rm)
        ok_all = ok_all and raises
        out["eps"][f"{eps:g}"] = {"plus": rp, "zero": base, "minus": rm,
                                  "raises": raises,
                                  "d_plus": rp - base, "d_minus": rm - base}
    out["raises_at_every_eps"] = bool(ok_all)
    return out


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e206_smoke" if SMOKE else "e206")
    log(f"E206 THE DRIFT-RATE CLOCK (smoke={SMOKE}) -> {rd}")
    load_pct = cpu_load_probe()
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()} — the owner "
        f"envelope's cap), load probe {load_pct if load_pct is not None else 'n/a'}%, "
        f"sequential tiny eval bursts, progressive writes, n=1 per lineage")
    load_check = {"cpu_count": os.cpu_count(),
                  "torch_threads": torch.get_num_threads(),
                  "load_pct_at_start": load_pct,
                  "note": "the envelope's load-check, recorded not gating: a "
                          "4-thread CPU eval burst of a few minutes; the GPU "
                          "lane is never claimed"}

    # =====================================================================
    # PHASE 0 — the parents loaded + the committed ladders hard-bound
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the parents loaded COMMITTED + hard-bound (e204's ladder "
        "asserted; org1/MIRABEL rows + ray md5s asserted)")
    missing = [str(p) for p in PARENTS.values() if not p.exists()]
    G_FILES = {"parents": {k: str(v) for k, v in PARENTS.items()},
               "missing": missing, "pass": bool(not missing)}
    assert G_FILES["pass"], f"missing parent metrics: {missing}"
    M = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in PARENTS.items()}
    G_FILES["experiment_names"] = {k: M[k].get("experiment") for k in PARENTS}
    G_FILES["statuses_complete"] = {k: ("COMPLETE" in str(M[k].get("status", "")))
                                    for k in PARENTS}
    G_FILES["pass"] = bool(all(G_FILES["statuses_complete"].values())
                           and not missing)
    log(f"G_FILES: {len(PARENTS)} parents, statuses COMPLETE "
        f"{sum(G_FILES['statuses_complete'].values())}/{len(PARENTS)}: "
        + ("PASS" if G_FILES["pass"] else "FAIL"))
    assert G_FILES["pass"], "parent identity gate failed"

    # hard-bind this module's constants vs the parent files (drift-asserts)
    e194wt = M["e194"]["phaseA_front_overlap"]["walk_traj"]
    for i, ref in enumerate((ORG1_S1, ORG1_S2)):
        r = e194wt[i]
        assert abs(r["ce_batch"] - ref["ce_batch"]) < 1e-12
        assert abs(r["cum_disp"] - ref["cum_disp"]) < 1e-12
        assert abs(r["gm12"] - ref["gm12"]) < 1e-12
        assert abs(r["preclip_gnorm"] - ref["preclip_gnorm"]) < 1e-12
    e195rays = {r["key"]: r["u_md5"] for r in M["e195"]["cell"]["rays"]}
    assert e195rays["u0"] == ORG1_U0_MD5 and e195rays["u1"] == ORG1_U1_MD5
    for k, v in ORG1_MUTUAL.items():
        assert abs(M["e195"]["ray_geometry"][k] - v) < 1e-15
    for i, ref in enumerate(MIR_JOURNAL):
        r = M["e198"]["phase0_walk_rebuild"]["walk_journal"][i]
        for k in ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm", "gm"):
            assert abs(r[k] - ref[k]) < 1e-12, f"e198 journal s{ref['step']} {k} drift"
    assert M["e198"]["phase0_walk_rebuild"]["kill_mirabel"]["step"] == 3
    e198rays = {r["key"]: r["u_md5"] for r in M["e198"]["cell"]["rays"]}
    for k in ("u0", "u1", "u2"):
        assert e198rays[k] == MIR_RAY_MD5S[k], f"e198 {k} md5 drift"
    for k, v in MIR_MUTUAL.items():
        assert abs(M["e198"]["ray_geometry"][k] - v) < 1e-15
    # the half lineage: e204's COMMITTED consecutive ladder + its ledgers
    e204g = M["e204"]["sensitivity_ladder"]["geometry"]
    for k, v in HALF_LADDER.items():
        assert abs(e204g[f"cos_s{k-1}_s{k}"] - v) < 1e-15, \
            f"e204 ladder c{str(k)} drift"
    assert M["e204"]["gates"]["G_ALIVE"]["dead_step"] == 5
    assert M["e204"]["gates"]["G_ALIVE"]["alive_steps"] == [1, 2, 3, 4]
    G_E204LADDER = {
        "source": "runs/e204/metrics.json sensitivity_ladder.geometry "
                  "(COMMITTED; asserted at load, never rerun here)",
        "consecutive_cosines": {str(k): v for k, v in HALF_LADDER.items()},
        "death_step_committed": 5, "alive_steps_committed": [1, 2, 3, 4],
        "verdict_parent": M["e204"]["adjudication"]["verdict"],
        "pass": True,
        "note": "THE HALF LINEAGE'S LADDER GATE: e204's committed "
                "sensitivity ladder (0.7764 -> 0.6802 -> 0.5636 -> 0.3334 "
                "-> 0.1917) + its G_ALIVE window — the adjudicated half-"
                "lineage numbers in this cell are these committed cosines, "
                "loaded, never recomputed",
    }
    log("G_E204LADDER: e204's committed ladder asserted (5 consecutive "
        "cosines + the t=5 death window)")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e206_drift_clock",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "load_check": load_check,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE, "torch": torch.__version__,
                               "threads": torch.get_num_threads()},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    stub["gates"]["G_FILES"] = G_FILES
    stub["gates"]["G_E204LADDER"] = G_E204LADDER
    write_partial("phase 0 complete (parents + the committed ladder bound)")
    log("phase 0 complete; partial metrics written")

    # =====================================================================
    # the shared protocol rebuild (corpus, batteries, anchor bank)
    # =====================================================================
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid_z, zid_m = stoi["Z"], stoi["M"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    corpus_mira = len(E43.find_occ(train_text, "MIRABEL"))
    G_NAMEFREE = {"corpus_counts": {"ZEPHYRA": corpus_zeph,
                                    "MIRABEL": corpus_mira},
                  "pass": bool(corpus_zeph == 0 and corpus_mira == 0)}
    assert G_NAMEFREE["pass"], f"nonce leaked: {G_NAMEFREE}"
    stub["gates"]["G_NAMEFREE"] = G_NAMEFREE

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = {"ZEPHYRA": host_occ[:60],   # org1's install (e043/e131 line)
                   "MIRABEL": host_occ[90:150]}  # e193b's fact 2 (e154's nonce)
    mix = {f: {"FLORIZEL": sum(1 for _, h in install_occ[f] if h == "FLORIZEL"),
               "ELIZABETH": sum(1 for _, h in install_occ[f] if h == "ELIZABETH")}
           for f in ("ZEPHYRA", "MIRABEL")}
    G_SPLICE = {"install_mix": mix,
                "expected": {"org1_ZEPHYRA": ORG1_SPLICE_MIX,
                             "e193b": MIR_SPLICE_MIX},
                "pass": bool(mix["ZEPHYRA"] == ORG1_SPLICE_MIX
                             and mix["MIRABEL"] == MIR_SPLICE_MIX["MIRABEL"]
                             and mix["ZEPHYRA"] == MIR_SPLICE_MIX["ZEPHYRA"]),
                "note": "one shuffle, both batteries: org1's ZEPHYRA "
                        "install host_occ[:60] (e195's gate) + e193b's "
                        "MIRABEL install host_occ[90:150] (e198/e201's) — "
                        "SPLICE_RNG 24301"}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    stub["gates"]["G_SPLICE"] = G_SPLICE

    bat_ids = {f: {} for f in ("ZEPHYRA", "MIRABEL")}
    for f in ("ZEPHYRA", "MIRABEL"):
        for j in READ_GEOS:
            cs = [train_text[p - PRE - j: p] for p, _ in install_occ[f]]
            bat_ids[f][j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {f: {str(j): list(bat_ids[f][j].shape) for j in READ_GEOS}
                   for f in ("ZEPHYRA", "MIRABEL")},
        "pass": bool(all(list(bat_ids[f][j].shape) == [60, PRE + j]
                         for f in ("ZEPHYRA", "MIRABEL") for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): both facts' install-60 "
                "batteries at the seven read geometries, 60 x (130 +- j) "
                "— e195/e198/e201's gate verbatim; THE SENSITIVITY "
                "BATTERY for both organisms is g-12 (e194/e195's "
                "convention, each on its OWN install)",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    stub["gates"]["G_BATTERY"] = G_BATTERY
    org1_sens_ids = bat_ids["ZEPHYRA"][RULER_J]
    mir_sens_ids = bat_ids["MIRABEL"][RULER_J]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

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
        "construction": "16 plain corpus windows from train_ids, RNG seed "
                        "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                        "in [s, s+257) — e170 VERBATIM",
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"
    stub["gates"]["G_ANCHOR"] = G_ANCHOR
    log("shared gates: G_NAMEFREE + G_SPLICE + G_BATTERY + G_ANCHOR PASS")

    def t0_gate(net0, committed_ce, committed_l2, tag):
        """e195/e198/e201's G_T0: the fresh batch-1 CE + AdamW step-1 L2
        (the MEASURED step L2, never ported for MIRABEL)."""
        g1 = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(16, (ANCH_BS,), generator=g1)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g1)
        anc1 = anchor_neutral[aj]
        rnd1 = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x1 = torch.cat([anc1[:, :-1], rnd1[:, :-1]], 0)
        y1 = torch.cat([anc1[:, 1:], rnd1[:, 1:]], 0)
        x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
        tw = copy.deepcopy(net0)
        tw.train()
        optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW,
                                 betas=(0.9, 0.95), weight_decay=0.1)
        logits, _ = tw(x1)
        ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    y1.reshape(-1)).item())
        optw.zero_grad(set_to_none=True)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                        y1.reshape(-1)).backward()
        gn1 = float(torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0))
        optw.step()
        disp1 = float(torch.norm(flat_params(tw) - flat_params(net0)))
        del tw, optw, logits
        gt = {"tag": tag, "step1_x_md5": x1_md5,
              "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
              "ce_batch_measured": ce1, "ce_batch_committed": committed_ce,
              "d_ce": abs(ce1 - committed_ce),
              "preclip_gnorm_measured": gn1,
              "adamw_step1_L2_measured": disp1,
              "committed_step_l2": committed_l2,
              "d_step_l2": abs(disp1 - committed_l2),
              "pass": bool(x1_md5 == E185_XHASH[1]
                           and abs(ce1 - committed_ce) < G_FALLBACK_TOL
                           and abs(disp1 - committed_l2) < G_FALLBACK_TOL),
              "note": "e195/e198/e201's G_T0 VERBATIM: the step-1 batch md5 "
                      "vs e185's stored hash + the forward CE + the fresh "
                      "CPU AdamW step's L2 vs the committed MEASURED step L2"}
        return gt

    # =====================================================================
    # PHASE 1 — ORG1 (the e131 root; death t=2)
    # =====================================================================
    log("=" * 78)
    log("PHASE 1 — ORG1: the e131 root, its 2-step sign walk rebuilt + "
        "gated, then the sensitivity ladder s_0/s_1/s_2")
    net_o = load_root(CKPT_DIR / ORG1_ROOT_CK)
    theta0_o = flat_params(net_o)
    assert int(theta0_o.numel()) == N_PARAM, f"params {theta0_o.numel()}"
    evl0 = copy.deepcopy(net_o)
    root_cells_o = {
        "gm12": battery_cell(evl0, org1_sens_ids, zid_z)["mean_pz"],
        "g0": battery_cell(evl0, bat_ids["ZEPHYRA"][0], zid_z)["mean_pz"],
        "gp12": battery_cell(evl0, bat_ids["ZEPHYRA"][12], zid_z)["mean_pz"],
        "ce_r": ce_fixed_cpu(evl0, *r_eval_xy),
    }
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r"}
    refs_o = {keymap[k]: v for k, v in E151_ROOT.items()}
    diffs_o = {k: root_cells_o[k] - refs_o[k] for k in refs_o}
    rmax_o = max(abs(v) for v in diffs_o.values())
    G_ROOT_O1 = {"cells": root_cells_o, "refs": refs_o, "diffs": diffs_o,
                 "max_abs_diff": rmax_o, "bit_tol": G_BIT_TOL,
                 "tol": G_FALLBACK_TOL, "bit": bool(rmax_o < G_BIT_TOL),
                 "pass": bool(rmax_o < G_FALLBACK_TOL),
                 "note": "e195's G_ROOT VERBATIM: the e131 consolidated "
                         "root's reads vs e151's committed before-cells"}
    log(f"G_ROOT_O1 (vs e151 before-cells): max|diff| {rmax_o:.2e}: "
        + ("PASS" if G_ROOT_O1["pass"] else "FAIL"))
    assert G_ROOT_O1["pass"], "org1 root gate FAILED"
    stub["gates"]["G_ROOT_O1"] = G_ROOT_O1

    # G_T0 + the walk
    gt_o = t0_gate(net_o, ORG1_CE1, ORG1_STEP_L2, "org1")
    log(f"G_T0_O1: x_md5 {'OK' if gt_o['step1_x_md5_match_e185'] else 'MISMATCH'}"
        f", CE |d| {gt_o['d_ce']:.2e}, L2 |d| {gt_o['d_step_l2']:.2e}: "
        + ("PASS" if gt_o["pass"] else "FAIL"))
    assert gt_o["pass"], "org1 t=0 gate FAILED"
    stub["gates"]["G_T0_O1"] = gt_o

    walk_o = sign_walk(net_o, anchor_neutral, train_ids, itos,
                       org1_sens_ids, zid_z, theta0_o, ORG1_STEP_L2,
                       n_steps=2, tag="org1")
    gw_rows = {}
    for a, b in zip(walk_o["traj"], (ORG1_S1, ORG1_S2)):
        per = {}
        for k in ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm"):
            per[k] = {"measured": a[k], "committed": b[k],
                      "abs_diff": abs(a[k] - b[k])}
        per["gm12"] = {"measured": a["gm"], "committed": b["gm12"],
                       "abs_diff": abs(a["gm"] - b["gm12"])}
        gw_rows[f"s{a['step']}"] = per
    gw = {"rows": gw_rows,
          "max_abs_diff": max(v["abs_diff"] for per in gw_rows.values()
                              for v in per.values()),
          "tol_bit": G_REPRO_TOL, "tol_texture": G_TEXTURE_TOL,
          "x_hashes_vs_e185": {s: bool(walk_o["x_hashes"][s] == E185_XHASH[s])
                               for s in (1, 2)},
          "max_l2_dev": walk_o["max_l2_dev"]}
    gw["tier"] = ("BIT" if gw["max_abs_diff"] < G_REPRO_TOL
                  else ("TEXTURE" if gw["max_abs_diff"] < G_TEXTURE_TOL
                        else "FAIL"))
    gw["tag"], gw["pass"] = "org1", bool(gw["tier"] != "FAIL"
                                         and all(gw["x_hashes_vs_e185"].values()))
    gw["note"] = ("THE ORG1 WALK GATE (Rule 12; e195's G_REPRO class): the "
                  "rebuilt 2-step k=1 sign walk must reproduce opt2/e194's "
                  "committed s1/s2 rows (ce, cum/step disp, pre-clip gnorm, "
                  "the g-12 ruler read) on md5-gated batches 1-2 — bit "
                  "class, or the disclosed cross-thread TEXTURE tier")
    log(f"G_WALK_O1: max|diff| {gw['max_abs_diff']:.2e} ({gw['tier']}): "
        + ("PASS" if gw["pass"] else "FAIL"))
    assert gw["pass"], "org1 walk gate FAILED — abort before any state is believed"
    stub["gates"]["G_WALK_O1"] = gw

    # G_SAVEDSTATE_O1: the walked theta_2 IS opt2's saved checkpoint
    sck = torch.load(CKPT_DIR / ORG1_OPT2_S2_CK, map_location="cpu",
                     weights_only=False)
    saved_net = TinyGPT(CFG1)
    saved_net.load_state_dict(sck["model"] if "model" in sck else sck)
    saved_flat = flat_params(saved_net)
    walked_s2 = walk_o["endpoints"][2]
    maxdiff_s2 = float((walked_s2 - saved_flat).abs().max())
    G_SAVEDSTATE_O1 = {
        "path": str(CKPT_DIR / ORG1_OPT2_S2_CK),
        "meta_gate": {"experiment": sck.get("meta", {}).get("experiment"),
                      "arm": sck.get("meta", {}).get("arm"),
                      "steps": sck.get("meta", {}).get("steps"),
                      "input_seed": sck.get("meta", {}).get("input_seed"),
                      "match": bool(sck.get("meta", {}).get("experiment") == "opt2"
                                    and sck.get("meta", {}).get("arm") == "a_sign"
                                    and sck.get("meta", {}).get("steps") == 2
                                    and sck.get("meta", {}).get("input_seed")
                                    == FREEZE_SEED)},
        "walked_s2_flat_md5": hashlib.md5(
            walked_s2.numpy().tobytes()).hexdigest(),
        "saved_flat_md5": hashlib.md5(saved_flat.numpy().tobytes()).hexdigest(),
        "max_abs_param_diff": maxdiff_s2, "tol": 1e-6,
        "pass": None,
        "note": "e195's G_SAVEDSTATE VERBATIM: opt2's committed checkpoint "
                "must BE the walked k=1 s2 endpoint — theta_2's chain of "
                "custody; theta_1 = theta_0 + delta_1 (the gated s1 row)",
    }
    G_SAVEDSTATE_O1["pass"] = bool(
        G_SAVEDSTATE_O1["meta_gate"]["match"]
        and (G_SAVEDSTATE_O1["walked_s2_flat_md5"]
             == G_SAVEDSTATE_O1["saved_flat_md5"] or maxdiff_s2 < 1e-6))
    log(f"G_SAVEDSTATE_O1: meta "
        + ("match" if G_SAVEDSTATE_O1["meta_gate"]["match"] else "DRIFT")
        + f", walked-s2 vs saved max|diff| {maxdiff_s2:.2e}: "
        + ("PASS" if G_SAVEDSTATE_O1["pass"] else "FAIL"))
    assert G_SAVEDSTATE_O1["pass"], "org1 saved-state gate FAILED"
    stub["gates"]["G_SAVEDSTATE_O1"] = G_SAVEDSTATE_O1

    # G_RAYS_O1: u0/u1 md5-gated (u0 vs e192's R2; u1 vs e195's ray)
    u0_o = (torch.sign(walk_o["fronts"][1])
            / torch.norm(torch.sign(walk_o["fronts"][1]))).clone()
    u1_o = (torch.sign(walk_o["fronts"][2])
            / torch.norm(torch.sign(walk_o["fronts"][2]))).clone()
    md5s_o = {"u0": hashlib.md5(u0_o.numpy().tobytes()).hexdigest(),
              "u1": hashlib.md5(u1_o.numpy().tobytes()).hexdigest()}
    geom_o = {"cos_u0_u1": cos64(u0_o, u1_o)}
    md5_ok_o = {"u0": md5s_o["u0"] == ORG1_U0_MD5,
                "u1": md5s_o["u1"] == ORG1_U1_MD5}
    geom_dev_o = abs(geom_o["cos_u0_u1"] - ORG1_MUTUAL["cos_u0_u1"])
    G_RAYS_O1 = {
        "u_md5s": md5s_o,
        "committed_md5s": {"u0": ORG1_U0_MD5, "u1": ORG1_U1_MD5},
        "md5_match": md5_ok_o, "fresh_mutual_geometry": geom_o,
        "geometry_dev_vs_e195_committed": geom_dev_o,
        "tol_geom": G_TEXTURE_TOL,
        "tier": ("BIT" if all(md5_ok_o.values())
                 else ("TEXTURE" if geom_dev_o < G_TEXTURE_TOL else "FAIL")),
        "pass": None,
        "note": "THE ORG1 RAY GATE: u0 vs e192's committed R2_SIGN md5, u1 "
                "vs e195's committed rotated ray — md5 BIT, or the fresh "
                "mutual geometry vs e195's committed cosine within 1e-3 "
                "(e199's tier); u2 NOT re-read (theta_2's custody is the "
                "saved checkpoint; e194's post-death continuation is out "
                "of scope)",
    }
    G_RAYS_O1["pass"] = bool(G_RAYS_O1["tier"] != "FAIL")
    log(f"G_RAYS_O1: tier {G_RAYS_O1['tier']} (md5 "
        f"{sum(md5_ok_o.values())}/2, geom dev {geom_dev_o:.2e}): "
        + ("PASS" if G_RAYS_O1["pass"] else "FAIL"))
    assert G_RAYS_O1["pass"], "org1 ray gate FAILED"
    stub["gates"]["G_RAYS_O1"] = G_RAYS_O1

    # G_ALIVE_O1: the alive window vs the committed ledger
    gms_o = {0: root_cells_o["gm12"], 1: walk_o["traj"][0]["gm"],
             2: walk_o["traj"][1]["gm"]}
    alive_o = {t: bool(v > SHUT_BAR) for t, v in gms_o.items()}
    G_ALIVE_O1 = {"per_state_ruler_read": gms_o, "bar": SHUT_BAR,
                  "alive": alive_o, "death_t": 2,
                  "expected_alive": [0, 1], "expected_death_t": 2,
                  "ledger_reads": ORG1_LEDGER["ruler_reads"],
                  "reads_match_ledger": bool(all(
                      abs(gms_o[t] - ORG1_LEDGER["ruler_reads"][t]) < 1e-3
                      for t in (0, 1, 2))),
                  "pass": bool(alive_o[0] and alive_o[1] and not alive_o[2]),
                  "note": "org1's alive window re-anchored on its OWN g-12 "
                          "ruler: t=0/1 ALIVE, t=2 DEAD — the committed "
                          "ledger (opt2/e194/e199)"}
    log(f"G_ALIVE_O1: reads {['%.4f' % gms_o[t] for t in (0, 1, 2)]} vs bar "
        f"{SHUT_BAR}: " + ("PASS" if G_ALIVE_O1["pass"] else "FAIL"))
    assert G_ALIVE_O1["pass"], "org1 alive-window gate FAILED"
    stub["gates"]["G_ALIVE_O1"] = G_ALIVE_O1

    # the org1 sensitivity ladder (the NEW compute)
    states_o = {0: theta0_o, 1: walk_o["endpoints"][1],
                2: walk_o["endpoints"][2]}
    sens_o = {}
    evl_s = copy.deepcopy(net_o)
    for t in (0, 1, 2):
        load_flat(evl_s, states_o[t])
        evl_s.eval()
        val = fact_readout(evl_s, org1_sens_ids, zid_z)
        g = fact_grad(evl_s, org1_sens_ids, zid_z)
        gn = float(torch.norm(g.double()))
        s_t = (g / torch.norm(g)).clone()
        sens_o[t] = {"read_meanlogp": val,
                     "read_meanp": math.exp(val) if val < 0 else None,
                     "g_norm": gn, "s": s_t}
        log(f"  [org1 s_{t}] meanlogp {val:+.6f} |g| {gn:.4f}")
    # the FD instrument gate (hard at the root; reported at theta_1)
    fd_root_o = fd_probe(net_o, states_o[0],
                         sens_o[0]["s"] / torch.norm(sens_o[0]["s"]),
                         org1_sens_ids, zid_z, FD_EPS, True, "org1 theta_0 (root)")
    fd_t1_o = fd_probe(net_o, states_o[1],
                       sens_o[1]["s"] / torch.norm(sens_o[1]["s"]),
                       org1_sens_ids, zid_z, FD_EPS, False, "org1 theta_1 (last alive)")
    G_SENS_O1 = {
        "convention": "e194/e195's fact_grad VERBATIM (mean log p(Z) over "
                      "the g-12 install-60 battery, one backward, eval "
                      "twin); matched-point: s_t computed AT theta_t",
        "battery_root_read": root_cells_o["gm12"],
        "battery_strength": "STRONG (0.9156 at D=0 — the convention's home)",
        "fd_root_HARD": fd_root_o, "fd_theta1_reported": fd_t1_o,
        "pass": bool(fd_root_o["raises_at_every_eps"]),
        "note": "THE INSTRUMENT GATE (e204's G_SENSDIR): at the root, "
                "+eps*s_0 must RAISE the readout and -eps*s_0 LOWER it, "
                "strictly, at every eps — the direction must BE the fact's "
                "sensitivity; the theta_1 probe reported, not gating",
    }
    log(f"G_SENS_O1 (FD root, hard): raises at every eps "
        f"{fd_root_o['raises_at_every_eps']}; theta_1 probe raises "
        f"{fd_t1_o['raises_at_every_eps']}: "
        + ("PASS" if G_SENS_O1["pass"] else "FAIL"))
    assert G_SENS_O1["pass"], "org1 sensitivity instrument gate FAILED"
    stub["gates"]["G_SENS_O1"] = G_SENS_O1

    lad_o = {1: cos64(sens_o[0]["s"], sens_o[1]["s"]),
             2: cos64(sens_o[1]["s"], sens_o[2]["s"])}
    org1_row = clock_row("org1", lad_o,
                         {1: True, 2: False}, 2, ORG1_STEP_L2, committed=False)
    stub["phase1_org1"] = E43.jsonable({
        "walk_journal": walk_o["traj"],
        "states_provenance": "theta_0 = the gated e131 root; theta_1/2 = "
                             "the gated 2-step walk's endpoints (G_WALK_O1 "
                             "+ G_SAVEDSTATE_O1)",
        "sensitivity_per_t": {str(t): {k: v for k, v in sens_o[t].items()
                                       if k != "s"} for t in (0, 1, 2)},
        "sensitivity_geometry": {
            "cos_s0_s1": lad_o[1], "cos_s1_s2": lad_o[2],
            "cos_s0_s2": cos64(sens_o[0]["s"], sens_o[2]["s"]),
            "iso_floor": 1.0 / (N_PARAM ** 0.5)},
        "clock": org1_row,
    })
    write_partial("phase 1 complete (org1 ladder + clock row)")
    log(f"org1 ladder: c1 {lad_o[1]:+.6f} (alive), c2 {lad_o[2]:+.6f} "
        f"(into-death); partial metrics written")
    del net_o, evl0, evl_s, saved_net

    # =====================================================================
    # PHASE 2 — MIRABEL (the e193b root; death t=3)
    # =====================================================================
    log("=" * 78)
    log("PHASE 2 — MIRABEL: the e193b root, its 3-step sign walk rebuilt + "
        "gated, then the sensitivity ladder s_0..s_3")
    net_m = load_root(CKPT_DIR / MIR_ROOT_CK)
    theta0_m = flat_params(net_m)
    assert int(theta0_m.numel()) == N_PARAM, f"params {theta0_m.numel()}"
    root_md5_m = hashlib.md5(theta0_m.numpy().tobytes()).hexdigest()
    evl0m = copy.deepcopy(net_m)
    root_cells_m = {f: {f"g{j:+d}": battery_cell(
        evl0m, bat_ids[f][j], zid_m if f == "MIRABEL" else zid_z)["mean_pz"]
        for j in READ_GEOS} for f in ("ZEPHYRA", "MIRABEL")}
    root_cells_m["ce_r"] = ce_fixed_cpu(evl0m, *r_eval_xy)
    rdiff_m = max(abs(root_cells_m[f][f"g{j:+d}"] - MIR_DIAL[f][f"g{j:+d}"])
                  for f in ("ZEPHYRA", "MIRABEL") for j in READ_GEOS)
    rdiff_m = max(rdiff_m, abs(root_cells_m["ce_r"] - MIR_DIAL["ce_r"]))
    G_ROOT_MIR = {"max_abs_diff": rdiff_m, "bit_tol": G_BIT_TOL,
                  "tol": G_FALLBACK_TOL, "bit": bool(rdiff_m < G_BIT_TOL),
                  "flat_md5": root_md5_m,
                  "flat_md5_match_committed": bool(root_md5_m == MIR_ROOT_MD5),
                  "pass": bool(rdiff_m < G_FALLBACK_TOL
                               and root_md5_m == MIR_ROOT_MD5),
                  "note": "e198/e201's G_ROOT verbatim: the root's dial "
                          "reproduces e193b's committed cells (both facts) "
                          "+ the flat md5"}
    log(f"G_ROOT_MIR: max|diff| {rdiff_m:.2e}, md5 "
        f"{'match' if G_ROOT_MIR['flat_md5_match_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT_MIR["pass"] else "FAIL"))
    assert G_ROOT_MIR["pass"], "MIRABEL root gate FAILED"
    stub["gates"]["G_ROOT_MIR"] = G_ROOT_MIR

    gt_m = t0_gate(net_m, MIR_S1_CE, MIR_STEP_L2, "MIRABEL")
    log(f"G_T0_MIR: x_md5 {'OK' if gt_m['step1_x_md5_match_e185'] else 'MISMATCH'}"
        f", CE |d| {gt_m['d_ce']:.2e}, L2 |d| {gt_m['d_step_l2']:.2e}: "
        + ("PASS" if gt_m["pass"] else "FAIL"))
    assert gt_m["pass"], "MIRABEL t=0 gate FAILED"
    stub["gates"]["G_T0_MIR"] = gt_m

    walk_m = sign_walk(net_m, anchor_neutral, train_ids, itos,
                       mir_sens_ids, zid_m, theta0_m, MIR_STEP_L2,
                       n_steps=3, tag="MIRABEL")
    gw_rows_m = {}
    for a, b in zip(walk_m["traj"], MIR_JOURNAL):
        per = {}
        for k in ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm"):
            per[k] = {"measured": a[k], "committed": b[k],
                      "abs_diff": abs(a[k] - b[k])}
        per["gm"] = {"measured": a["gm"], "committed": b["gm"],
                     "abs_diff": abs(a["gm"] - b["gm"])}
        gw_rows_m[f"s{a['step']}"] = per
    G_WALK_MIR = {
        "rows": gw_rows_m,
        "max_abs_diff": max(v["abs_diff"] for per in gw_rows_m.values()
                            for v in per.values()),
        "tol_bit": G_REPRO_TOL, "tol_texture": G_TEXTURE_TOL,
        "x_hashes_vs_e185": {s: bool(walk_m["x_hashes"][s] == E185_XHASH[s])
                             for s in (1, 2, 3)},
        "max_l2_dev": walk_m["max_l2_dev"],
        "note": "THE MIRABEL WALK GATE (Rule 12; e201's G_WALKE198 class): "
                "the rebuilt 3-step walk must reproduce e198's committed "
                "journal (ce, cum/step disp, pre-clip gnorm, the MIRABEL "
                "g-12 ruler read) on md5-gated batches 1-3 — bit class, or "
                "the disclosed cross-thread TEXTURE tier; co-rulers and "
                "the kill bracket are not consumed here (deviation)",
    }
    G_WALK_MIR["tier"] = ("BIT" if G_WALK_MIR["max_abs_diff"] < G_REPRO_TOL
                          else ("TEXTURE" if G_WALK_MIR["max_abs_diff"]
                                < G_TEXTURE_TOL else "FAIL"))
    G_WALK_MIR["pass"] = bool(G_WALK_MIR["tier"] != "FAIL"
                              and all(G_WALK_MIR["x_hashes_vs_e185"].values()))
    log(f"G_WALK_MIR: max|diff| {G_WALK_MIR['max_abs_diff']:.2e} "
        f"({G_WALK_MIR['tier']}): "
        + ("PASS" if G_WALK_MIR["pass"] else "FAIL"))
    assert G_WALK_MIR["pass"], "MIRABEL walk gate FAILED"
    stub["gates"]["G_WALK_MIR"] = G_WALK_MIR

    rays_m = {}
    for k in (1, 2, 3):
        sg = torch.sign(walk_m["fronts"][k])
        rays_m[f"u{k-1}"] = (sg / torch.norm(sg)).clone()
    md5s_m = {k: hashlib.md5(v.numpy().tobytes()).hexdigest()
              for k, v in rays_m.items()}
    md5_ok_m = {k: md5s_m[k] == MIR_RAY_MD5S[k] for k in rays_m}
    geom_m = {"cos_u0_u1": cos64(rays_m["u0"], rays_m["u1"]),
              "cos_u0_u2": cos64(rays_m["u0"], rays_m["u2"]),
              "cos_u1_u2": cos64(rays_m["u1"], rays_m["u2"])}
    geom_dev_m = max(abs(geom_m[k] - MIR_MUTUAL[k]) for k in geom_m)
    G_RAYS_MIR = {
        "u_md5s": md5s_m, "committed_md5s": MIR_RAY_MD5S,
        "md5_match": md5_ok_m, "fresh_mutual_geometry": geom_m,
        "geometry_dev_vs_e198_committed": {k: abs(geom_m[k] - MIR_MUTUAL[k])
                                           for k in geom_m},
        "tol_geom": G_TEXTURE_TOL,
        "tier": ("BIT" if all(md5_ok_m.values())
                 else ("TEXTURE" if geom_dev_m < G_TEXTURE_TOL else "FAIL")),
        "pass": None,
        "note": "THE MIRABEL RAY GATE (e201's G_RAYSE198 class): u0/u1/u2 "
                "md5-gated vs e198's committed rays, or the fresh mutual "
                "geometry within 1e-3",
    }
    G_RAYS_MIR["pass"] = bool(G_RAYS_MIR["tier"] != "FAIL")
    log(f"G_RAYS_MIR: tier {G_RAYS_MIR['tier']} (md5 "
        f"{sum(md5_ok_m.values())}/3, geom dev {geom_dev_m:.2e}): "
        + ("PASS" if G_RAYS_MIR["pass"] else "FAIL"))
    assert G_RAYS_MIR["pass"], "MIRABEL ray gate FAILED"
    stub["gates"]["G_RAYS_MIR"] = G_RAYS_MIR

    gms_m = {0: root_cells_m["MIRABEL"][f"g{RULER_J:+d}"],
             1: walk_m["traj"][0]["gm"], 2: walk_m["traj"][1]["gm"],
             3: walk_m["traj"][2]["gm"]}
    alive_m = {t: bool(v > SHUT_BAR) for t, v in gms_m.items()}
    G_ALIVE_MIR = {"per_state_ruler_read": gms_m, "bar": SHUT_BAR,
                   "alive": alive_m, "death_t": 3,
                   "expected_alive": [0, 1, 2], "expected_death_t": 3,
                   "ledger_reads": MIR_LEDGER["ruler_reads"],
                   "reads_match_ledger": bool(all(
                       abs(gms_m[t] - MIR_LEDGER["ruler_reads"][t]) < 1e-3
                       for t in (0, 1, 2, 3))),
                   "pass": bool(all(alive_m[t] for t in (0, 1, 2))
                                and not alive_m[3]),
                   "note": "MIRABEL's alive window re-anchored on its OWN "
                           "g-12 ruler: t=0/1/2 ALIVE, t=3 DEAD — e198's "
                           "committed ledger (the wash bounce 0.367 -> "
                           "0.691 visible in the window)"}
    log(f"G_ALIVE_MIR: reads "
        f"{['%.4f' % gms_m[t] for t in (0, 1, 2, 3)]} vs bar {SHUT_BAR}: "
        + ("PASS" if G_ALIVE_MIR["pass"] else "FAIL"))
    assert G_ALIVE_MIR["pass"], "MIRABEL alive-window gate FAILED"
    stub["gates"]["G_ALIVE_MIR"] = G_ALIVE_MIR

    # the MIRABEL sensitivity ladder (the NEW compute)
    states_m = {0: theta0_m, 1: walk_m["endpoints"][1],
                2: walk_m["endpoints"][2], 3: walk_m["endpoints"][3]}
    sens_m = {}
    evl_sm = copy.deepcopy(net_m)
    for t in (0, 1, 2, 3):
        load_flat(evl_sm, states_m[t])
        evl_sm.eval()
        val = fact_readout(evl_sm, mir_sens_ids, zid_m)
        g = fact_grad(evl_sm, mir_sens_ids, zid_m)
        gn = float(torch.norm(g.double()))
        sens_m[t] = {"read_meanlogp": val,
                     "read_meanp": math.exp(val) if val < 0 else None,
                     "g_norm": gn, "s": (g / torch.norm(g)).clone()}
        log(f"  [MIRABEL s_{t}] meanlogp {val:+.6f} |g| {gn:.4f}")
    fd_root_m = fd_probe(net_m, states_m[0],
                         sens_m[0]["s"] / torch.norm(sens_m[0]["s"]),
                         mir_sens_ids, zid_m, FD_EPS, True,
                         "MIRABEL theta_0 (root)")
    fd_t2_m = fd_probe(net_m, states_m[2],
                       sens_m[2]["s"] / torch.norm(sens_m[2]["s"]),
                       mir_sens_ids, zid_m, FD_EPS, False,
                       "MIRABEL theta_2 (last alive)")
    G_SENS_MIR = {
        "convention": "e194/e195's fact_grad VERBATIM, carried to MIRABEL "
                      "per e201's census precedent (mean log p(M) over "
                      "MIRABEL's own g-12 install-60 battery); matched-point",
        "battery_root_read": gms_m[0],
        "battery_strength": "ALIVE (0.6236 at D=0; above the 0.27 bar)",
        "fd_root_HARD": fd_root_m, "fd_theta2_reported": fd_t2_m,
        "pass": bool(fd_root_m["raises_at_every_eps"]),
        "note": "THE INSTRUMENT GATE (e204's G_SENSDIR): hard at the root; "
                "the theta_2 (last alive) probe reported, not gating",
    }
    log(f"G_SENS_MIR (FD root, hard): raises at every eps "
        f"{fd_root_m['raises_at_every_eps']}; theta_2 probe raises "
        f"{fd_t2_m['raises_at_every_eps']}: "
        + ("PASS" if G_SENS_MIR["pass"] else "FAIL"))
    assert G_SENS_MIR["pass"], "MIRABEL sensitivity instrument gate FAILED"
    stub["gates"]["G_SENS_MIR"] = G_SENS_MIR

    lad_m = {1: cos64(sens_m[0]["s"], sens_m[1]["s"]),
             2: cos64(sens_m[1]["s"], sens_m[2]["s"]),
             3: cos64(sens_m[2]["s"], sens_m[3]["s"])}
    mir_row = clock_row("MIRABEL", lad_m, {1: True, 2: True, 3: False},
                        3, MIR_STEP_L2, committed=False)
    stub["phase2_mirabel"] = E43.jsonable({
        "walk_journal": walk_m["traj"],
        "states_provenance": "theta_0 = the gated e193b root; theta_1/2/3 = "
                             "the gated 3-step walk's endpoints (G_WALK_MIR "
                             "+ G_RAYS_MIR)",
        "sensitivity_per_t": {str(t): {k: v for k, v in sens_m[t].items()
                                       if k != "s"} for t in (0, 1, 2, 3)},
        "sensitivity_geometry": {
            "cos_s0_s1": lad_m[1], "cos_s1_s2": lad_m[2],
            "cos_s2_s3": lad_m[3],
            "cos_s0_s2": cos64(sens_m[0]["s"], sens_m[2]["s"]),
            "cos_s0_s3": cos64(sens_m[0]["s"], sens_m[3]["s"]),
            "iso_floor": 1.0 / (N_PARAM ** 0.5)},
        "clock": mir_row,
    })
    write_partial("phase 2 complete (MIRABEL ladder + clock row)")
    log(f"MIRABEL ladder: c1 {lad_m[1]:+.6f}, c2 {lad_m[2]:+.6f} (alive), "
        f"c3 {lad_m[3]:+.6f} (into-death); partial metrics written")
    del net_m, evl0m, evl_sm

    # =====================================================================
    # PHASE 3 — THE CLOCK: the three ladders, the drift rates, the
    # predictions, the adjudication (bars verbatim, no bar shopping)
    # =====================================================================
    log("=" * 78)
    log("PHASE 3 — THE DRIFT-RATE CLOCK: three ladders, early-window fits, "
        "extrapolations vs the committed death steps")
    alive_half = {1: True, 2: True, 3: True, 4: True, 5: False}
    half_row = clock_row("half (org2 f2, committed)", HALF_LADDER, alive_half,
                         5, HALF_LEDGER["step_L2"], committed=True)
    rows = [org1_row, mir_row, half_row]
    for r in rows:
        p = r["prediction_primary"]
        log(f"  [{r['lineage']}] ladder "
            + " -> ".join(f"{r['ladder'][str(k)]:.3f}" for k in sorted(
                int(k) for k in r["ladder"]))
            + f"; d_hat {r['drift_rate_primary'] if r['drift_rate_primary'] is not None else 'UNDEF'}"
            + (f"; t_hat {p['t_hat']} vs death {r['death_t']} "
               f"(err {p['err']:+d}, {'LANDS' if p['lands'] else 'MISS'})"
               if p["defined"] else "; prediction UNDEFINED (one tick)"))

    n_defined = sum(1 for r in rows if r["prediction_primary"]["defined"])
    lands = [r["lineage"] for r in rows if r["prediction_primary"]["lands"]]
    clock_predicts = bool(len(lands) == 3)
    clock_one = bool(not clock_predicts and len(lands) >= 1)
    no_clock = bool(len(lands) == 0 and n_defined >= 1)
    graded = not (clock_predicts or clock_one or no_clock)
    bars = {
        "CLOCK_PREDICTS": {
            "fires": clock_predicts,
            "detail": {"lands": lands,
                       "note": "requires ALL THREE defined AND within +-1"}},
        "CLOCK_ONE_LINEAGE": {
            "fires": clock_one,
            "detail": {"lands": lands, "n_defined": n_defined,
                       "undefined": [r["lineage"] for r in rows
                                     if not r["prediction_primary"]["defined"]],
                       "misses": [r["lineage"] for r in rows
                                  if r["prediction_primary"]["defined"]
                                  and not r["prediction_primary"]["lands"]]}},
        "NO_CLOCK": {
            "fires": no_clock,
            "detail": {"lands": lands, "n_defined": n_defined}},
        "GRADED": {"fires": graded, "detail": {"lands": lands,
                                               "n_defined": n_defined}},
    }
    verdict = ("CLOCK-PREDICTS" if clock_predicts else
               "CLOCK-ONE-LINEAGE" if clock_one else
               "NO-CLOCK" if no_clock else "GRADED")
    clause = ("; ".join(
        f"{r['lineage']}: ladder "
        + " -> ".join(f"{r['ladder'][str(k)]:.3f}" for k in sorted(
            int(k) for k in r["ladder"]))
        + f", d_hat "
        + (f"{r['drift_rate_primary']:+.4f}" if r["drift_rate_primary"] is not None
           else "UNDEFINED")
        + (f", t_hat {r['prediction_primary']['t_hat']} vs death "
           f"{r['death_t']} (err {r['prediction_primary']['err']:+d}, "
           f"{'LANDS' if r['prediction_primary']['lands'] else 'MISS'})"
           if r["prediction_primary"]["defined"]
           else ", prediction UNDEFINED (one tick in the early window)")
        + f", c_death {r['ladder'].get(str(r['death_t'])):.3f} "
        + ("<= " if r["ladder"].get(str(r["death_t"])) <= TAU else "> ")
        + f"tau {TAU:g}" for r in rows)
        + f"; verdict {verdict}")
    log("ADJUDICATION: " + verdict)
    log("clause: " + clause)

    # ---- the PNG: three ladders + the extrapolations -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, r in zip(axes, rows):
        name = r["lineage"]
        ks = sorted(int(k) for k in r["ladder"])
        cs = [r["ladder"][str(k)] for k in ks]
        alive_flags = [r["alive_window"].get(k, False) for k in ks]
        ax.axhline(TAU, color="tab:red", ls=":", lw=1.4,
                   label=f"tau = {TAU:g} (registered)")
        ka = [k for k, a in zip(ks, alive_flags) if a]
        ca = [r["ladder"][str(k)] for k in ka]
        ax.plot(ka, ca, "o-", color="tab:blue", lw=1.8, ms=7,
                label="ladder (alive cosines)")
        kd = [k for k, a in zip(ks, alive_flags) if not a]
        cd = [r["ladder"][str(k)] for k in kd]
        if kd:
            ax.plot(kd, cd, "X", color="darkred", ms=11,
                    label="into-death cosine")
        if r["drift_rate_primary"] is not None:
            d = r["drift_rate_primary"]
            c1 = r["early_window"]["values"][0]
            t_hat = r["prediction_primary"]["t_hat"]
            xs = np.arange(1, max(t_hat + 1 if t_hat else 3, r["death_t"] + 2))
            ax.plot(xs, c1 + (xs - 1) * d, "--", color="tab:green", lw=1.6,
                    label=f"early-fit extrapolation (d={d:+.3f})")
            if t_hat:
                ax.axvline(t_hat, color="tab:green", ls="--", lw=1.2)
        ax.axvline(r["death_t"], color="black", lw=1.6,
                   label=f"death t={r['death_t']} (committed)")
        p = r["prediction_primary"]
        ttl = name + ("\n" + (f"t_hat={p['t_hat']}, err {p['err']:+d} "
                              f"({'LANDS' if p['lands'] else 'MISS'})"
                              if p["defined"]
                              else "prediction UNDEFINED (one tick)"))
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("ladder index k   [c_k = cos(s_(k-1), s_k)]")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("consecutive sensitivity cosine")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"E206 THE DRIFT-RATE CLOCK — verdict: {verdict}   "
                 f"(tau={TAU:g} registered; early window k<=2; +-1 bar; "
                 f"half lineage = e204's committed ladder)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = rd / ("e206_drift_clock_smoke.png" if SMOKE
                else "e206_drift_clock.png")
    fig.savefig(png, dpi=130)
    log(f"PNG written: {png}")

    # =====================================================================
    # the final COMPLETE write
    # =====================================================================
    final = dict(stub)
    final.update({
        "status": (f"COMPLETE — adjudicated{' (SMOKE — nothing adjudicated)' if SMOKE else ''} "
                   "(this write replaces all PARTIAL progressive writes)"),
        "question": ("THE DRIFT-RATE CLOCK (W027's named cut): is the "
                     "support's decorrelation rate a DEATH TIMER? The half "
                     "lineage's sensitivity ladder hit near-orthogonality "
                     "(0.192) exactly at its death (t=5) — does the "
                     "early-t (k <= 2) drift rate, extrapolated to the "
                     "tau=0.2 near-orthogonality threshold, predict the "
                     "observed death step on each of the three lineages "
                     "(org1 t=2, MIRABEL t=3, half t=5)?"),
        "organisms": {
            "org1": {"root": f"runs/checkpoints/{ORG1_ROOT_CK}",
                     "architecture": "6L/6H/192d/256-ctx TinyGPT "
                                     "(2,739,072) — the e131 line",
                     "fact": "ZEPHYRA (its own g-12 install-60 battery)",
                     "step_L2": ORG1_STEP_L2, "death_t": 2,
                     "sens_battery_root_read": root_cells_o["gm12"]},
            "MIRABEL": {"root": f"runs/checkpoints/{MIR_ROOT_CK}",
                        "root_flat_md5": MIR_ROOT_MD5,
                        "architecture": "6L/6H/192d/256-ctx TinyGPT "
                                        "(2,739,072) — e193b, organism 1's "
                                        "EXACT architecture, fresh draw",
                        "fact": "MIRABEL (its own g-12 install-60 battery)",
                        "step_L2": MIR_STEP_L2, "death_t": 3,
                        "sens_battery_root_read": gms_m[0]},
            "half": {"root": "runs/checkpoints/e157_f2_consolidated.pt",
                     "architecture": "4L/4H/128d/512-ctx TinyGPT "
                                     "(873,472) — architecture co-varies "
                                     "with lineage (e193's disclosure)",
                     "fact": "ZEPHYRA (the f2 install; ladder on e204's "
                             "g-12 battery, committed)",
                     "step_L2": HALF_LEDGER["step_L2"], "death_t": 5,
                     "note": "the counterfactual half-step wash (the "
                             "natural 0.9164 step killed it at t=1); its "
                             "ladder is e204's COMMITTED geometry, loaded "
                             "and gated, never recomputed"},
        },
        "the_clock": {
            "threshold_tau_registered": TAU,
            "early_window": "alive c_k with k <= 2 (the dispatch's t <= 2)",
            "primary_protocol": "d_hat = mean first difference of the early "
                                "window in cosine space; t_hat = smallest "
                                "integer t with c_1 + (t-1)*d_hat <= tau; "
                                "lands iff defined and |err| <= 1",
            "rows": rows,
        },
        "adjudication": {
            "bars": {k: v["fires"] for k, v in bars.items()},
            "bar_details": bars,
            "verdict": verdict,
            "clause": clause,
            "composite_order": "CLOCK-PREDICTS -> CLOCK-ONE-LINEAGE -> "
                               "NO-CLOCK -> GRADED (frozen before compute)",
            "constants": {"TAU": TAU, "EARLY_KMAX": EARLY_KMAX,
                          "ERR_BAR": ERR_BAR, "FD_EPS": list(FD_EPS)},
        },
        "references": {
            "e204_support": {"metrics": "runs/e204/metrics.json",
                             "role": "THE HALF LINEAGE'S COMMITTED LADDER "
                                     "(loaded, gated) + the fact_grad/FD "
                                     "machinery this cell ports to org1/"
                                     "MIRABEL"},
            "e194_e195": {"metrics": "runs/e194/metrics.json, "
                                     "runs/e195/metrics.json",
                          "role": "org1's committed walk rows + rays + the "
                                  "saved theta_2 checkpoint; the fact_grad "
                                  "convention (minted on org1)"},
            "e198_e201": {"metrics": "runs/e198/metrics.json, "
                                     "runs/e201/metrics.json",
                          "role": "MIRABEL's committed walk journal + rays; "
                                  "e201's multi-organism census convention"},
            "thinking": "W027 (the wonder card that named the cut: does the "
                        "support's drift rate predict death time?), T171 "
                        "(the two-rotator picture: the front bounces, the "
                        "sensitivity drifts smoothly, 0.78 -> 0.19 nearly "
                        "orthogonal at death)",
        },
        "honesty_reflex": {
            "n_and_scope": "n=1 stream per lineage, three lineages, one "
                           "battery geometry per organism, a threshold "
                           "registered from the one ladder on record: "
                           "whatever the verdict, it is a three-biography "
                           "read, not a population claim",
            "threshold_provenance": "tau = 0.2 was registered FROM the half "
                                    "lineage (the only ladder on record, "
                                    "death cosine 0.192): the at-death "
                                    "crossing on that lineage is partially "
                                    "tautological; the REPLICATION reads "
                                    "(org1, MIRABEL) are the live content",
            "circularity_disclosures": "org1's death (t=2) is INSIDE the "
                                       "early window (k <= 2): its primary "
                                       "prediction is UNDEFINED (one tick; "
                                       "a rate needs two), and its "
                                       "dispatch-literal prediction is "
                                       "CIRCULAR (consumes the outcome row) "
                                       "— both ride stamped, never "
                                       "adjudicated as successes",
            "alignment_reads_never_predict": "the extrapolation is an "
                                             "arithmetic claim about "
                                             "gradient directions, not an "
                                             "intervention; nothing here "
                                             "establishes causality "
                                             "(T157's class caveat, carried)",
            "counterfactual_wash_caveat": "the half lineage is alive only "
                                          "because the experimenter halved "
                                          "the step (the natural 0.9164 "
                                          "step killed it at t=1); its "
                                          "ladder belongs to the alive "
                                          "window OF THIS CONSTRUCTION "
                                          "(T158/T160, carried from e204)",
            "step_size_confound": "the half lineage ticks 0.458 L2/step vs "
                                  "1.654 on the 2.74M organisms: per-STEP "
                                  "drift rates are not commensurable across "
                                  "lineages; the per-D co-read rides and "
                                  "never adjudicates",
            "battery_strengths": "the sensitivity batteries differ in "
                                 "strength at their roots: org1 0.9156 "
                                 "(the convention's home), MIRABEL 0.6236, "
                                 "half 0.198 (UNDER the 0.27 bar — e204's "
                                 "disclosure, carried)",
            "openness": "WHAT EACH ARM GUARANTEES: NOTHING — the ladders "
                        "are local linearizations of one readout each on "
                        "single-stream walks; the clock is one arithmetic "
                        "protocol on three numbers per lineage; the "
                        "openness is the point",
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"torch": torch.__version__,
                   "threads": torch.get_num_threads(),
                   "device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "smoke": SMOKE,
                   "eval_only": True,
                   "n_lineages": 3},
    })
    save_json(rd / "metrics.json", E43.jsonable(final))
    log(f"DONE -> {rd / 'metrics.json'} ({verdict})")


if __name__ == "__main__":
    main()
