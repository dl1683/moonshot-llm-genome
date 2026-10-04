"""E255 — FQ13, THE THERMAL-LEDGER IDENTITY (desk + eval-only, CPU, minutes).

THE QUESTION (the R64 ideator's Rank 2, agy consult #001(a)'s sharpened
survivor): is the walled world's margin thickening (now known GENERIC LN
redistribution — T227/e251: fact +19.2%, ctrl +17.2%, held-30 +72.5%)
SUPPLIED by the wall's own cooling — the thermal contraction of the residual
stream narrowing LN's denominator — so that "the wall cools" (e242's
committed T(t): 1.31 at +1, cooling to 0.84-0.90) and "the margins thicken"
are ONE mechanism seen through two dials? THE CROSS-WORLD FALSIFIER the
ideator named: in the UNWALLED world margins COLLAPSE (0.823->0.153-class,
T212/e232) while T RISES (e238: 1.00 -> 1.45 at +80); in the walled world
margins THICKEN while T FALLS — pure erosion-redistribution cannot survive
that contrast (more erosion should thicken MORE); thermal contraction can.

THE CELL (the dispatch letter, on e242/e251's bit-certified states):
  (1) THE REGRESSION: per-probe margin growth (t0 -> each state) against the
      state's T-decline (T_t0 - T_t), pooled and per battery;
  (2) THE DECOMPOSITION: split each probe's sigma-normalized margin change
      into the NUMERATOR (the probe's own aligned logit component: e228's
      margin_raw = top1-top2) vs the DENOMINATOR (the state's logit spread:
      e228's sigma = std(vocab logits) at the answer position) — which side
      carries the thickening?;
  (3) THE CROSS-WORLD CONTROL: the same decomposition on the committed
      unwalled +80 numbers (e232's margins n=8 fact-w1, e238's committed
      T_mle, e238's committed logit dumps for the numerator/denominator) —
      the sign flip test.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
the script is committed at birth; adjudicate against exactly this; no bar
shopping):
  - IDENTITY-CONFIRMED — "margin growth tracks the T-decline quantitatively
    (the pooled regression's R^2 >= 0.6 with the expected sign) AND the
    decomposition shows the DENOMINATOR carrying the thickening (numerator
    ~inert) AND the unwalled control flips sign — the budget's origin named:
    THE COOLING; W038's law 4 collapses to 'the wall cools; LN does the
    rest'"
  - NUMERATOR-RESIDUE — "the fact battery's numerator grows beyond ctrl (a
    battery-specific constructive residue) — the re-formation layer keeps a
    small real object; report its size honestly"
  - MIXED — "the tables verbatim, both batteries, both worlds"

OPERATIONALIZATIONS (frozen BEFORE compute; they fix the clauses, they do
not move the bars):
  * THE T THE LEDGER READS (T228's answer-locality amendment CARRIED): the
    regression's x is e242's COMMITTED T_mle — its family is q_i(T) =
    softmax(L0_i/T)[Z] with a Bernoulli MLE over the fact battery's 60 p's,
    i.e. an ANSWER-POSITION P-SHAPE dial, NOT the literal logit spread. The
    decomposition's D is the LITERAL per-probe logit spread (e228's sigma).
    Both dials are reported side by side at every state (T_fit vs the direct
    spread ratio T_sigma = median_i D_{i,t}/D_{i,0}); the read states which
    is which wherever a number is used.
  * THE REGRESSION: y_{i,s} = 100 x (m_{i,s}/m_{i,0} - 1) per probe i at
    state s (m = margin_sigma); x_s = T_mle(t0) - T_mle(s) from e242's
    committed thermal leg (hard-bound literals, runtime-asserted). POOLED =
    all (probe, state) pairs over the ADJUDICATING batteries (fact 60 +
    ctrl 60) at the ADJUDICATING states {1,2,10,50,100,300} (720 points);
    OLS with intercept. Expected sign = POSITIVE slope (cooling ->
    thickening). REGRESSION-CLAUSE := (pooled R^2 >= 0.6) AND (slope > 0).
    Companions (never adjudicate alone): per-battery pooled OLS; the
    texture states {4,200} included; held-30 pooled; the STATE-MEDIAN
    regression (battery-median growth per state, 6 points — where the
    tracking lives if probe noise dominates); the DIRECT-SPREAD regression
    (x' = 1 - T_sigma(s), the literal dial).
  * THE DECOMPOSITION: per probe, EXACTLY log(m_{i,s}/m_{i,0}) =
    log(N_{i,s}/N_{i,0}) - log(D_{i,s}/D_{i,0}) (N = e228.margin_raw,
    D = e228.sigma — the imported instrument's own fields; asserted to
    machine precision). c_num(B,s) := median_i log(N ratio); c_den(B,s) :=
    median_i log(D ratio). DENOMINATOR-CARRIES := at s = +300, for BOTH
    B in {fact, ctrl}: c_den < 0 AND |c_den| > |c_num|. NUMERATOR-INERT :=
    at s = +300, for BOTH B: |c_num| <= 0.05 (log units, ~ +/-5.1% — the
    "~inert" band). DECOMPOSITION-CLAUSE := DENOMINATOR-CARRIES AND
    NUMERATOR-INERT. Probes with a zero numerator or denominator at either
    endpoint are excluded from log-medians (counted + disclosed).
  * THE CROSS-WORLD CONTROL (all committed): y-side = e232's committed
    fact-w1 margins (n=8, runs/e232/metrics.json read1_zombie_lag.
    probe_level.fact.w1 — margin_t0/margin_80, hard-bound); T-side = e238's
    committed T_mle (w1 {2,10,50,80}, hard-bound; T(t0) := 1.0 by the
    family's construction); the decomposition = computed from e238's
    committed logit dumps (sha-gated) on the SAME n=8 (join certified vs
    e232's committed margins, tol 1e-6). CONTROL-CLAUSE (the sign flip) :=
    walled_fact_growth(+300) > 0 [committed +19.15%] AND walled
    T-decline(+300) > 0 AND unwalled_fact_growth(+80) < -25% (a real
    collapse, not a wiggle) AND unwalled T-decline(+80) < 0 AND the
    unwalled collapse is NUMERATOR-carried (c_num(n8,+80) < 0 AND
    |c_num| > |c_den| — the mirror of the walled denominator-carriage).
  * VERDICTS, precedence frozen: IDENTITY-CONFIRMED := (hard gates PASS)
    AND REGRESSION-CLAUSE AND DECOMPOSITION-CLAUSE AND CONTROL-CLAUSE.
    NUMERATOR-RESIDUE := (hard gates PASS) AND NOT IDENTITY-CONFIRMED AND
    (fact c_num(+300) > +0.05) AND (fact c_num(+300) > ctrl c_num(+300))
    — "the fact battery's numerator grows beyond ctrl"; its size is
    reported in both log and % currency. MIXED := otherwise, INCLUDING any
    hard-gate failure (the bars stand down, e242/e251's precedent) — the
    tables verbatim, both batteries, both worlds.
  * The batteries: fact = the install-60 g-12 ruler (certification
    re-measure vs e242's committed per-probe margin_sigma, tol 1e-7);
    ctrl = e251's frozen construction REBUILT (same seed 26502; the
    position list asserted EXACTLY equal to e251's committed battery;
    medians re-certified vs e251's committed trajectory, tol 1e-7);
    held-30 = e043's convention (co-report only, e251's precedent: its
    answers span only {F, E}); ADJUDICATING batteries = fact + ctrl.

CHECKS (the dispatch's letter): the states' provenance (the bit-exact md5s
from e242/e251 — root flat md5, resume-ckpt sha, per-state body md5s, ruler
reads); the T(t) values committed (hard-bound + runtime-asserted); the
answer-locality amendment carried (the T-vs-spread statement above); n=1
lineage, ONE wall commit (R=0.7), ONE wash draw (seed 10902); nothing
guaranteed.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
e248 owns the GPU; e254 is CPU — load-polite: threads 4, load checks per
burst, one state = one eval burst); ~1350 single-probe forwards + 9 light
dials; minutes total; progressive metrics.json writes after every phase.

Outputs: runs/e255/{metrics.json (PROGRESSIVE), e255_regression.png,
e255_decomposition.png, e255_crossworld.png}. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds). Commit + push per phase.

Run:  cd lab && python e255_thermal_ledger.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e251's convention)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # e228's offline convention (imported)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 — REPO, find_occ, SPLICE_RNG, jsonable
import e225_one_currency as E225                       # noqa: E402 — the roster + battery constants
import e228_margin_landscape as E228                   # noqa: E402 — THE margin instrument (VERBATIM)
import e229_wall_currency as E229                      # noqa: E402 — the shim + the P0a pattern
import g1b_continuity as GB                            # noqa: E402 — the 2.74M patch (BEFORE G1)
import g1_anchored_ball as G1                          # noqa: E402 — evl_load (settle+disarm), MAINTAIN_BAR

torch.set_num_threads(4)                                # the dispatch envelope (g1's import resets to 8)

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import textwrap                                         # noqa: E402

CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e255 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ frozen constants
RUNS = E43.REPO / "runs"
CKPT_DIR = GB.CKPT_DIR
G1C_ROOT_CK = "g1c_root.pt"
G1C_W1_RESUME_CK = "g1c_W1_resume.pt"
G1C_METRICS = RUNS / "g1c_root" / "metrics.json"
E242_METRICS = RUNS / "e242" / "metrics.json"
E251_METRICS = RUNS / "e251" / "metrics.json"
E232_METRICS = RUNS / "e232" / "metrics.json"
E238_METRICS = RUNS / "e238" / "metrics.json"
E238_NPZ = {0: RUNS / "e238" / "logits_t0.npz",
            2: RUNS / "e238" / "logits_w1s2.npz",
            10: RUNS / "e238" / "logits_w1s10.npz",
            50: RUNS / "e238" / "logits_w1s50.npz",
            80: RUNS / "e238" / "logits_w1s80.npz"}

STATE_GRID: tuple[int, ...] = (1, 2, 10, 50, 100, 300)   # the registered grid (adjudicates)
TEXTURE_STATES: tuple[int, ...] = (4, 200)               # committed; texture co-reports only
G_READ_TOL = 5e-3           # e225/e229/e242/e251's G_ROOT tolerance
REPRO_TOL = 1e-7            # same loads + same imported instrument + same device -> bit-exact (T204)
CTRL_CTX = 118              # the ruler's own g-12 context length (E225.PRE - 12)
CTRL_N_PER = 30             # 30 word + 30 sentence = 60 probes (e251's battery)
UNWALLED_JOIN_TOL = 1e-6    # npz-derived margin_sigma vs e232's committed margins (fp32 npz)
INERT_BAND_LOG = 0.05       # the "~inert" band on |c_num| (log units, ~ +/-5.1%)
COLLAPSE_PCT = -25.0        # the unwalled collapse floor for the flip test (a real collapse)
R2_BAR = 0.6                # the frozen R^2 bar

# g1c's committed record, HARD-BOUND (e242/e251's own literals; read at
# runtime from runs/g1c_root/metrics.json and asserted; Rule 12).
G1C_VERDICT = "ROOT-WALL-HOLDS"
G1C_ROOT_GM12 = 0.9026340246200562
G1C_W1_GM12 = {
    1: 0.821441113948822, 2: 0.9093567132949829, 4: 0.8956068754196167,
    10: 0.9277286529541016, 50: 0.9397175908088684, 100: 0.9368361234664917,
    200: 0.9264556169509888, 300: 0.9405527114868164,
}
G1C_ROOT_FLAT_MD5 = "a7f02b367c5342535aecfc814d780631"
G1C_W1_SHA16 = "650e631c2dbcf44f"
G1C_BODY_MDS = {
    1: "100c15c07c3fc34fb1d40ceee9494613", 2: "756360f07eae3d1f32581847e49f46b3",
    4: "15ee413d45785475ac3a18a178ba5d9d", 10: "464069503316930cef3e98234fbff5d9",
    50: "8cb39e128291db825da066bb07da8b7e", 100: "3ee37433c15b8e2c439b7eeb2423e10f",
    200: "29f8b5b49ab6af621a721533ecf88b74", 300: "d5d1bb6894944a9c877c9aa3964c61ba",
}

# e242's committed fact-battery margin trajectory + THERMAL LEG (the ledger's
# committed dials), HARD-BOUND; read at runtime and asserted.
E242_MD5 = "9bf5ae17b3091673f47e7c2a0845b4ab"
E242_VERDICT = "FLAT-COMMITMENT"
E242_MEDIANS = {   # state -> committed battery MEDIAN margin_sigma
    0: 0.9794073282786698, 1: 0.9611918699600377, 2: 0.9933262426349674,
    4: 0.9632744808830553, 10: 1.0801870072706619, 50: 1.1753321683243787,
    100: 1.1397841935007293, 200: 1.0717460701295862, 300: 1.1670049513746559,
}
E242_T_MLE = {     # the wall's committed T(t) — e242's thermal leg, T_mle
    0: 1.0000001089780448, 1: 1.3089933564469043, 2: 0.9642851663129937,
    4: 1.0289645331892097, 10: 0.8938458726638275, 50: 0.8423125157928674,
    100: 0.8645557494479506, 200: 0.9021596334061261, 300: 0.8495807216775402,
}

# e251's committed ctrl + held trajectories and battery positions, HARD-BOUND
# (the ctrl battery is REBUILT from e251's frozen construction; the lists and
# the medians must reproduce EXACTLY).
E251_MD5 = "aab7df28eab3728d11c6561cd59879e8"
E251_VERDICT = "ZERO-SUM-LN"
E251_CTRL_MED = {
    0: 0.3013793138250127, 1: 0.22758515599384765, 2: 0.2936599478737389,
    4: 0.33202773246101414, 10: 0.3538132281257641, 50: 0.3297853415831098,
    100: 0.3117759778389322, 200: 0.35952256434768004, 300: 0.35333095075986726,
}
E251_HELD_MED = {
    0: 0.6317739544142845, 1: 0.4803387958575124, 2: 0.7767639306506209,
    4: 0.5312984180438729, 10: 0.7363467855947401, 50: 0.9112485787308677,
    100: 0.8994714828924557, 200: 0.8108906673367494, 300: 1.0895252487313163,
}
E251_CTRL_POS = [
    ("ctrl00@word", 66528), ("ctrl01@word", 75785), ("ctrl02@word", 90431),
    ("ctrl03@word", 76679), ("ctrl04@word", 51480), ("ctrl05@word", 82306),
    ("ctrl06@word", 40992), ("ctrl07@word", 3020), ("ctrl08@word", 110330),
    ("ctrl09@word", 83112), ("ctrl10@word", 98858), ("ctrl11@word", 49395),
    ("ctrl12@word", 86418), ("ctrl13@word", 20390), ("ctrl14@word", 1361),
    ("ctrl15@word", 15198), ("ctrl16@word", 9265), ("ctrl17@word", 103984),
    ("ctrl18@word", 30845), ("ctrl19@word", 81265), ("ctrl20@word", 48645),
    ("ctrl21@word", 52089), ("ctrl22@word", 13747), ("ctrl23@word", 16876),
    ("ctrl24@word", 67898), ("ctrl25@word", 73767), ("ctrl26@word", 49242),
    ("ctrl27@word", 95053), ("ctrl28@word", 40787), ("ctrl29@word", 37142),
    ("ctrl30@sentence", 94344), ("ctrl31@sentence", 26098), ("ctrl32@sentence", 98455),
    ("ctrl33@sentence", 83883), ("ctrl34@sentence", 12264), ("ctrl35@sentence", 29899),
    ("ctrl36@sentence", 7042), ("ctrl37@sentence", 54128), ("ctrl38@sentence", 36691),
    ("ctrl39@sentence", 100617), ("ctrl40@sentence", 88428), ("ctrl41@sentence", 63924),
    ("ctrl42@sentence", 31683), ("ctrl43@sentence", 61805), ("ctrl44@sentence", 63706),
    ("ctrl45@sentence", 39589), ("ctrl46@sentence", 95581), ("ctrl47@sentence", 47413),
    ("ctrl48@sentence", 9960), ("ctrl49@sentence", 38120), ("ctrl50@sentence", 45551),
    ("ctrl51@sentence", 95518), ("ctrl52@sentence", 28788), ("ctrl53@sentence", 64409),
    ("ctrl54@sentence", 66097), ("ctrl55@sentence", 43993), ("ctrl56@sentence", 103505),
    ("ctrl57@sentence", 38007), ("ctrl58@sentence", 25082), ("ctrl59@sentence", 39334),
]
E251_HELD_POS = [
    ("held00@ELIZ", 206667), ("held01@FLOR", 803836), ("held02@FLOR", 800243),
    ("held03@ELIZ", 290744), ("held04@ELIZ", 283488), ("held05@ELIZ", 285476),
    ("held06@ELIZ", 294089), ("held07@FLOR", 803210), ("held08@FLOR", 783897),
    ("held09@ELIZ", 285891), ("held10@ELIZ", 291478), ("held11@ELIZ", 221432),
    ("held12@ELIZ", 276792), ("held13@ELIZ", 189033), ("held14@FLOR", 827039),
    ("held15@FLOR", 805049), ("held16@FLOR", 788699), ("held17@ELIZ", 672419),
    ("held18@ELIZ", 293541), ("held19@ELIZ", 293656), ("held20@ELIZ", 286114),
    ("held21@FLOR", 829744), ("held22@ELIZ", 287023), ("held23@ELIZ", 264382),
    ("held24@ELIZ", 221112), ("held25@ELIZ", 274165), ("held26@ELIZ", 285297),
    ("held27@FLOR", 808131), ("held28@FLOR", 829441), ("held29@ELIZ", 220839),
]

# the unwalled world's committed numbers, HARD-BOUND:
E232_SHA16 = "307d046b7e7e9c5f"     # sha256_16 of runs/e232/metrics.json
E238_SHA16 = "95290a45b9aaf7e4"     # sha256_16 of runs/e238/metrics.json (committed in e252's G_TFIT)
E232_FACT_W1 = {                    # e232's committed fact-w1 margins (n=8; the journal's shared subset)
    "Greece->Athens": (1.1781698104110228, 0.36234748537294087),
    "Poland->Warsaw": (0.8231108730177659, 0.09858356653577356),
    "Portugal->Lisbon": (0.7570613400170839, 0.19482384049806584),
    "Egypt->Cairo": (0.6357743715268048, 0.08184035244937499),
    "Ireland->Dublin": (0.4925394483890496, 0.04650722290517785),
    "the United Kingdom->pound": (0.9509313345344212, 0.2678971444006647),
    "the United States->dollar": (0.2922677329736964, 0.07057359403389044),
    "China->yuan": (0.8816438614677309, 0.002336873704543759),
}
E238_T_MLE_W1 = {                   # e238's committed T_mle, wash 1 (e252's G_TFIT literals)
    2: 1.0014473256519658, 10: 1.0597956498145238,
    50: 1.3489839391244727, 80: 1.4525923182890421,
}
E238_T_MLE_W2 = {                   # co-report (the second wash)
    10: 1.0744525359278094, 50: 1.3363590444428126, 80: 1.413836040516422,
}
E238_NPZ_SHA = {0: "fd1d302f5713ec82", 2: "59e6c0aa1b220518",
                10: "cd4fdedb5a7d605a", 50: "465bac8aca6b4076",
                80: "f13897401fe60341"}
UNWALLED_GRID = (2, 10, 50, 80)     # the shared unwalled states (w1 primary)

REGISTERED_BARS = {
    "IDENTITY-CONFIRMED": 'IDENTITY-CONFIRMED — "margin growth tracks the '
        'T-decline quantitatively (the pooled regression\'s R^2 >= 0.6 with '
        'the expected sign) AND the decomposition shows the DENOMINATOR '
        'carrying the thickening (numerator ~inert) AND the unwalled control '
        'flips sign — the budget\'s origin named: THE COOLING; W038\'s law 4 '
        'collapses to \'the wall cools; LN does the rest\'"',
    "NUMERATOR-RESIDUE": 'NUMERATOR-RESIDUE — "the fact battery\'s numerator '
        'grows beyond ctrl (a battery-specific constructive residue) — the '
        're-formation layer keeps a small real object; report its size '
        'honestly"',
    "MIXED": 'MIXED — "the tables verbatim, both batteries, both worlds"',
    "registration": "bars frozen VERBATIM from the dispatch brief in the "
                    "module docstring BEFORE any compute (script committed "
                    "at birth); adjudicate against exactly this; no bar "
                    "shopping.",
    "clause_fixes":
        "THE REGRESSION: y=100*(m_{i,s}/m_{i,0}-1) per probe; x=T_mle(t0)-"
        "T_mle(s) from e242's committed thermal leg; POOLED = fact+ctrl "
        "probes x adjudicating states {1,2,10,50,100,300}, OLS with "
        "intercept; REGRESSION-CLAUSE := pooled R^2 >= 0.6 AND slope > 0. "
        "THE DECOMPOSITION: per probe exactly log(m ratio)=log(N ratio)-"
        "log(D ratio) (N=e228.margin_raw, D=e228.sigma); c_num/c_den = "
        "battery medians of the log ratios; DENOMINATOR-CARRIES := at +300, "
        "both fact and ctrl: c_den<0 AND |c_den|>|c_num|; NUMERATOR-INERT := "
        "at +300, both: |c_num|<=0.05 (log); DECOMPOSITION-CLAUSE := both. "
        "THE CROSS-WORLD CONTROL: y=e232's committed fact-w1 margins (n=8, "
        "hard-bound); T=e238's committed T_mle (w1, hard-bound; T(t0):=1.0 "
        "by the family's construction); decomposition from e238's committed "
        "logit dumps on the same n=8 (join tol 1e-6); CONTROL-CLAUSE := "
        "walled fact growth(+300)>0 AND walled T-decline(+300)>0 AND "
        "unwalled fact growth(+80)<-25% AND unwalled T-decline(+80)<0 AND "
        "unwalled NUMERATOR-carried (c_num<0 AND |c_num|>|c_den| at +80, "
        "n=8). VERDICTS (precedence): IDENTITY-CONFIRMED := gates PASS AND "
        "all three clauses; NUMERATOR-RESIDUE := gates PASS AND NOT "
        "IDENTITY AND fact c_num(+300)>+0.05 AND fact c_num>ctrl c_num "
        "(size reported honestly); MIXED := otherwise, INCLUDING any "
        "hard-gate failure (stand-down, e242/e251's precedent). The T the "
        "ledger reads (T228 carried): e242's committed T_mle is an "
        "answer-position P-SHAPE fit, not the literal logit spread; the "
        "decomposition's D is the literal spread; both dials reported at "
        "every state.",
}

deviations: list[str] = [
    "DESK + EVAL-ONLY: the walled side re-measures e242/e251's bit-certified "
    "states (the pristine root + the W1 resume ckpt's embedded wash states) "
    "through the SAME module-imported instrument (e228.margin_pass via "
    "e229's shim) — the NUMERATOR (margin_raw) and DENOMINATOR (sigma) "
    "fields e242/e251 did not store per-probe are the cell's fresh reads; "
    "the unwalled side is PURE DESK on committed artifacts (e232's margins, "
    "e238's T_mle + logit dumps). No training, no wash, no GPU (e248 owns "
    "the GPU; e254 is CPU — load-polite: threads 4, load checks, one state "
    "= one eval burst).",
    "The ctrl battery is e251's frozen construction REBUILT (same corpus, "
    "same seed 26502, same filters); its position list is asserted EXACTLY "
    "equal to e251's committed battery and its per-state medians re-"
    "certified against e251's committed trajectory (tol 1e-7). The fact "
    "battery is re-measured per-probe against e242's committed per-probe "
    "margin_sigma (tol 1e-7) — certifications, never adjudications.",
    "held-30 rides as a CO-REPORT only (e251's precedent: its answers span "
    "only {F, E}); the adjudicating batteries are fact + ctrl.",
    "The unwalled primary is e232's committed fact-w1 battery (n=8, the "
    "journal's shared subset — the battery the ideator's falsifier quotes); "
    "the full npz n=20 fact battery rides as a co-report. The unwalled "
    "T(t0):=1.0 by e238's family construction (q=softmax(L0/T) on t0's own "
    "logits), stated wherever used.",
    "Importing e228 (via e229) opens runs/e228_run.log in append mode as a "
    "module side effect — nothing is written to it by this cell (own stdout "
    "log). g1's import resets torch threads to 8 — reset to 4 after import.",
    "The state grid: the registered {t0,+1,+2,+10,+50,+100,+300} adjudicates; "
    "{+4,+200} ride as TEXTURE co-reports (e242/e251's own convention; they "
    "enter sensitivity re-runs only).",
    "Probes with a zero (or non-finite) numerator/denominator at either "
    "endpoint of a log-ratio are EXCLUDED from that log-median (counted + "
    "disclosed) — no epsilon fudging.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
]


# ------------------------------------------------------------------ helpers
def cpu_load_probe() -> float | None:
    try:
        import psutil
        return round(float(psutil.cpu_percent(interval=1.0)), 1)
    except Exception:                                      # noqa: BLE001
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).LoadPercentage"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            return float(out) if out else None
        except Exception:                                  # noqa: BLE001
            return None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                      # noqa: BLE001
        return "unavailable"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def body_flat_md5(sd: dict) -> str:
    """md5 of a state dict's BODY tensors (anch__ buffers excluded) — e242's
    function VERBATIM (provenance for the states)."""
    flat = torch.cat([sd[k].reshape(-1).float()
                      for k in sorted(sd) if not k.startswith("anch__")])
    return hashlib.md5(flat.numpy().tobytes()).hexdigest()


def ols(xs, ys) -> dict:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    slope, intercept = np.polyfit(x, y, 1)
    yh = slope * x + intercept
    ssr = float(((y - yh) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    return {"n": int(len(x)), "slope": float(slope),
            "intercept": float(intercept), "r2": float(r2)}


def log_median(ratios: list[float]) -> tuple[float, int]:
    """median of log(r) over finite, positive entries; returns (median,
    n_excluded)."""
    vals = [float(np.log(r)) for r in ratios
            if np.isfinite(r) and r > 0.0]
    excl = len(ratios) - len(vals)
    return (float(np.median(vals)) if vals else float("nan")), excl


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e255")
    metrics: dict = {
        "experiment": "e255_thermal_ledger",
        "phase": "FQ13 — THE THERMAL-LEDGER IDENTITY: is the walled world's "
                 "margin thickening SUPPLIED by the wall's cooling (thermal "
                 "contraction narrowing LN's denominator)? the regression + "
                 "the numerator/denominator decomposition + the cross-world "
                 "sign-flip control",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": False,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; e248 owns "
                      "the GPU; e254 is CPU — load-polite)",
            "torch_threads": torch.get_num_threads(),
            "bursts": "one state = one eval burst (3 batteries x margin_pass "
                      "+ the light dial)",
            "n_evals": "n=1 per state per battery (deterministic, T204)",
            "cpu_load_pct_at_launch": cpu_load_probe(),
        },
        "deviations": deviations,
        "builds_on": [
            "FQ13 / scratch/review_r64_ideator.md (the R64 ideator's Rank 2; "
            "this cell's spec) + agy consult #001(a) (the sharpened survivor)",
            "T227 / e251 (ZERO-SUM-LN: the thickening is generic — the "
            "budget's ORIGIN question this cell answers; the ctrl battery's "
            "frozen construction + committed trajectory, re-certified here)",
            "T220 / e242 (the wall's committed T(t): 1.31 at +1, cooling to "
            "0.84-0.90; the committed fact trajectory + the bit-certified "
            "states; the thermal-leg instrument whose committed T_mle is "
            "this cell's x-axis)",
            "T228 / e252 (the answer-locality amendment: the fitted T is an "
            "answer-position p-shape, not the literal spread — CARRIED: the "
            "ledger read states which spread it uses at every dial)",
            "T212 / e232 + T218 / e238 (the unwalled world's committed "
            "collapse + committed T rise; the logit dumps that make the "
            "unwalled decomposition a pure desk read)",
            "T207 / e228 + T208 / e229 (the argmax-margin instrument with "
            "its margin_raw/sigma fields, module-imported verbatim; the "
            "shim; the P0a pattern)",
            "T181 / g1c (the fresh root lineage: the committed W1 wash "
            "states this cell reads)",
            "W038 (the laws draft's law 4 — the clause IDENTITY-CONFIRMED "
            "would collapse)",
        ],
        "whats_new": [
            "the THERMAL-LEDGER JOIN: per-probe margin growth regressed on "
            "the committed T-decline across the wall's whole state grid — "
            "the first quantitative coupling of the two dials e242 left "
            "side-by-side",
            "the EXACT numerator/denominator decomposition of the "
            "sigma-normalized margin change (per probe: log(m ratio) = "
            "log(N ratio) - log(D ratio), machine-asserted) on THREE "
            "batteries — which side carries the thickening",
            "the same decomposition run on the UNWALLED world's committed "
            "+80 archive (e238's logit dumps) — the cross-world sign-flip "
            "test of the ideator's falsifier",
        ],
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase_note"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    log(f"E255 — FQ13, THE THERMAL-LEDGER IDENTITY -> {rd}")

    # ================= P0a: the batteries (e251's constructions) ============
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    G_NAMEFREE = {"corpus_zeph_count": train_text.count("ZEPH"),
                  "val_zeph_count": val_text.count("ZEPH"),
                  "pass": bool(train_text.count("ZEPH") == 0
                               and val_text.count("ZEPH") == 0)}
    assert G_NAMEFREE["pass"], f"corpus/val contains ZEPH: {G_NAMEFREE}"

    host_occ = []
    for host in E225.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E225.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "held_mix":
                {"FLORIZEL": sum(1 for _, h in held_occ if h == "FLORIZEL"),
                 "ELIZABETH": sum(1 for _, h in held_occ if h == "ELIZABETH")},
                "n_held": len(held_occ),
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                             and len(held_occ) == 30)}
    assert G_SPLICE["pass"], f"splice drift {G_SPLICE}"

    j = E225.RULER_J
    ruler_ids = torch.stack(
        [corpus.encode(train_text[p - E225.PRE - j: p]) for p, _ in install_occ])
    G_BATTERY = {"shapes": {"g-12": list(ruler_ids.shape)},
                 "pass": bool(list(ruler_ids.shape) == [60, E225.PRE - 12]),
                 "note": "install-60 g-12 ruler battery (SPLICE_RNG 24301; "
                         "e242/e251's construction re-run from its module "
                         "constants)"}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    # THE CTRL battery: e251's frozen construction REBUILT.
    g_ctrl = torch.Generator().manual_seed(int(G1.R_EVAL_SEED))
    n_val = len(val_ids)
    word_pos, sent_pos, tries = [], [], 0
    while ((len(word_pos) < CTRL_N_PER or len(sent_pos) < CTRL_N_PER)
           and tries < 500_000):
        tries += 1
        i = int(torch.randint(280, n_val - 1, (1,), generator=g_ctrl))
        ch, prev = val_text[i], val_text[i - 1]
        if not ch.isalpha() or ch == "Z":
            continue
        window = val_text[i - CTRL_CTX: i + 1]
        if any(f in window for f in E225.ANCHOR_FORBIDDEN):
            continue
        if prev.isalpha():
            if len(word_pos) < CTRL_N_PER:
                word_pos.append(i)
        elif prev.isspace() and i >= 2 and val_text[i - 2] in ".!?":
            if len(sent_pos) < CTRL_N_PER:
                sent_pos.append(i)
    ctrl_pos = word_pos + sent_pos
    ctrl_stratum = ["word"] * len(word_pos) + ["sentence"] * len(sent_pos)
    ctrl_probes = [{"ids": corpus.encode(val_text[i - CTRL_CTX: i]).unsqueeze(0),
                    "fact": f"ctrl{k:02d}@{s}", "relation": f"ctrl_{s}",
                    "ans_id": stoi[val_text[i]], "pos": i}
                   for k, (i, s) in enumerate(zip(ctrl_pos, ctrl_stratum))]
    ctrl_ans_chars = [val_text[i] for i in ctrl_pos]
    built_list = [(p["fact"], p["pos"]) for p in ctrl_probes]
    G_CTRL = {
        "n_word": len(word_pos), "n_sentence": len(sent_pos),
        "expected_per_stratum": CTRL_N_PER, "tries": tries,
        "ctx_len": CTRL_CTX,
        "all_ctx_len_118": bool(all(p["ids"].shape[1] == CTRL_CTX
                                    for p in ctrl_probes)),
        "all_answers_alpha": bool(all(c.isalpha() and c != "Z"
                                      for c in ctrl_ans_chars)),
        "forbidden_free": bool(not any(
            f in val_text[i - CTRL_CTX: i + 1]
            for i in ctrl_pos for f in E225.ANCHOR_FORBIDDEN)),
        "positions_unique": bool(len(set(ctrl_pos)) == len(ctrl_pos)),
        "position_list_matches_e251": bool(built_list == E251_CTRL_POS),
        "pass": bool(len(word_pos) == CTRL_N_PER
                     and len(sent_pos) == CTRL_N_PER
                     and all(p["ids"].shape[1] == CTRL_CTX for p in ctrl_probes)
                     and all(c.isalpha() and c != "Z" for c in ctrl_ans_chars)
                     and len(set(ctrl_pos)) == len(ctrl_pos)
                     and built_list == E251_CTRL_POS),
        "note": "e251's ctrl battery REBUILT from its frozen construction "
                "(seed 26502) — the position list asserted EXACTLY equal to "
                "e251's committed battery; medians re-certified per state "
                "(G_REPRO)",
    }
    assert G_CTRL["pass"], f"ctrl battery rebuild FAILED: {G_CTRL}"

    held_probes = [{"ids": corpus.encode(train_text[p - E225.PRE - j: p]).unsqueeze(0),
                    "fact": f"held{k:02d}@{h[:4]}", "relation": "held30_incumbent",
                    "ans_id": stoi[h[0]], "pos": p, "host": h}
                   for k, (p, h) in enumerate(held_occ)]
    held_built = [(p["fact"], p["pos"]) for p in held_probes]
    G_HELD = {
        "n": len(held_probes), "expected": 30,
        "position_list_matches_e251": bool(held_built == E251_HELD_POS),
        "pass": bool(len(held_probes) == 30 and held_built == E251_HELD_POS),
        "note": "e043's held convention (e251's co-report battery, rebuilt "
                "and position-asserted); never adjudicates",
    }
    assert G_HELD["pass"], f"held battery rebuild FAILED: {G_HELD}"

    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_CTRL": G_CTRL,
                        "G_HELD": G_HELD}
    log(f"P0a: batteries rebuilt — fact 60 (19+41); ctrl {len(word_pos)}W+"
        f"{len(sent_pos)}S == e251's list; held 30 == e251's list")
    write_partial("P0a battery gates PASSED")

    # ================= P0b: G_PARENTS — committed records hard-bound ========
    for pth in (G1C_METRICS, E242_METRICS, E251_METRICS, E232_METRICS,
                E238_METRICS):
        if not pth.exists():
            raise RuntimeError(f"missing parent record: {pth}")
    g1c = json.loads(G1C_METRICS.read_text(encoding="utf-8"))
    e242m = json.loads(E242_METRICS.read_text(encoding="utf-8"))
    e251m = json.loads(E251_METRICS.read_text(encoding="utf-8"))
    e232m = json.loads(E232_METRICS.read_text(encoding="utf-8"))
    e238m = json.loads(E238_METRICS.read_text(encoding="utf-8"))

    g1c_w1 = {int(k): v for k, v in
              g1c["adjudication"]["wall"]["W1"]["g_m12"].items()}
    ok_g1c = (g1c["adjudication"]["verdict"] == G1C_VERDICT
              and abs(g1c["root_build"]["root_cells"]["gm12"] - G1C_ROOT_GM12)
              < 1e-12
              and all(abs(g1c_w1[s] - G1C_W1_GM12[s]) < 1e-12
                      for s in G1C_W1_GM12))

    # e242: medians + THERMAL LEG T_mle hard-bound
    tleg = {r["step"]: r["T_mle"] for r in e242m["thermal_leg"]["rows"]}
    traj242 = {r["step"]: r for r in e242m["trajectory"]}
    ok_e242 = (e242m["adjudication"]["verdict"] == E242_VERDICT
               and set(traj242) == set(E242_MEDIANS)
               and all(abs(traj242[s]["margin_median_sigma"] - E242_MEDIANS[s])
                       < 1e-12 for s in E242_MEDIANS)
               and all(abs(tleg[s] - E242_T_MLE[s]) < 1e-12
                       for s in E242_T_MLE))

    # e251: md5 + verdict + ctrl/held medians + battery lists hard-bound
    ctrl_list_251 = [(p["fact"], p["pos"])
                     for p in e251m["batteries"]["ctrl"]["probes"]]
    held_list_251 = [(p["fact"], p["pos"])
                     for p in e251m["batteries"]["held30"]["probes"]]
    ctrl_traj = {r["step"]: r["median_sigma"]
                 for r in e251m["trajectory"]["ctrl"]}
    held_traj = {r["step"]: r["median_sigma"]
                 for r in e251m["trajectory"]["held30_coreport"]}
    ok_e251 = (md5of(E251_METRICS) == E251_MD5
               and e251m["adjudication"]["verdict"] == E251_VERDICT
               and ctrl_list_251 == E251_CTRL_POS
               and held_list_251 == E251_HELD_POS
               and all(abs(ctrl_traj[s] - E251_CTRL_MED[s]) < 1e-12
                       for s in E251_CTRL_MED)
               and all(abs(held_traj[s] - E251_HELD_MED[s]) < 1e-12
                       for s in E251_HELD_MED))

    # e232: the committed fact-w1 margins hard-bound (sha-gated)
    fw1 = {r["probe"]: r for r in
           e232m["read1_zombie_lag"]["probe_level"]["fact"]["w1"]}
    ok_e232 = (sha256_of(E232_METRICS) == E232_SHA16
               and set(fw1) == set(E232_FACT_W1)
               and all(abs(fw1[k]["margin_t0"] - v[0]) < 1e-12
                       and abs(fw1[k]["margin_80"] - v[1]) < 1e-12
                       for k, v in E232_FACT_W1.items()))

    # e238: the committed T_mle hard-bound (sha-gated)
    t238 = {}
    for r in e238m["two_moment_bound"]["rows"]:
        if r["wash"] == "w1":
            t238[r["state"]] = r["T_full"]
    ok_e238 = (sha256_of(E238_METRICS) == E238_SHA16
               and all(abs(t238[s] - E238_T_MLE_W1[s]) < 1e-12
                       for s in E238_T_MLE_W1) and len(t238) == 4)

    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS),
                        "md5": md5of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"]},
        "e242_metrics": {"path": str(E242_METRICS), "md5": md5of(E242_METRICS),
                         "sha256_16": sha256_of(E242_METRICS),
                         "verdict": e242m["adjudication"]["verdict"]},
        "e251_metrics": {"path": str(E251_METRICS), "md5": md5of(E251_METRICS),
                         "sha256_16": sha256_of(E251_METRICS),
                         "verdict": e251m["adjudication"]["verdict"]},
        "e232_metrics": {"path": str(E232_METRICS),
                         "sha256_16": sha256_of(E232_METRICS)},
        "e238_metrics": {"path": str(E238_METRICS),
                         "sha256_16": sha256_of(E238_METRICS)},
        "pass": bool(ok_g1c and ok_e242 and ok_e251 and ok_e232 and ok_e238),
        "note": "every committed record this cell reads (g1c's states, "
                "e242's fact trajectory + thermal leg, e251's ctrl/held "
                "trajectories + battery lists, e232's unwalled fact-w1 "
                "margins, e238's unwalled T_mle) read at runtime and "
                "asserted against the literals frozen in this script "
                "pre-compute (Rule 12)",
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS — g1c/e242/e251/e232/e238 all hard-bound")
    write_partial("P0b parent gates PASSED")

    # ================= P1: the states — load, certify, measure ==============
    all_steps = tuple(sorted(STATE_GRID + TEXTURE_STATES))
    wck = torch.load(CKPT_DIR / G1C_W1_RESUME_CK, map_location=CPU,
                     weights_only=False)
    sds_keys = sorted(int(k) for k in wck["sds"].keys())
    body_keys_final = {k: v for k, v in wck["model"].items()
                       if not str(k).startswith("anch__")}
    sds300 = {k: v for k, v in wck["sds"][300].items()
              if not str(k).startswith("anch__")}
    max_diff_300 = max(float(torch.max(torch.abs(sds300[k] - body_keys_final[k])))
                       for k in body_keys_final)
    state_body_mds = {s: body_flat_md5(wck["sds"][s]) for s in all_steps}
    G_STATES_A = {
        "checkpoint": f"runs/checkpoints/{G1C_W1_RESUME_CK}",
        "sha256_16": sha256_of(CKPT_DIR / G1C_W1_RESUME_CK),
        "step_field": int(wck["step"]),
        "wall_R": float(wck["wall_R"]),
        "sds_steps": sds_keys,
        "sds300_bit_identical_to_final_model": bool(max_diff_300 == 0.0),
        "body_flat_md5s_match_e242_committed": bool(all(
            state_body_mds[s] == G1C_BODY_MDS[s] for s in all_steps)),
        "pass": bool(int(wck["step"]) == 300
                     and abs(float(wck["wall_R"]) - 0.7) < 1e-6
                     and tuple(sds_keys) == all_steps and max_diff_300 == 0.0
                     and all(state_body_mds[s] == G1C_BODY_MDS[s]
                             for s in all_steps)
                     and sha256_of(CKPT_DIR / G1C_W1_RESUME_CK) == G1C_W1_SHA16),
    }
    assert G_STATES_A["pass"], f"W1 resume inventory gate FAILED: {G_STATES_A}"

    root_net, root_sd, root_meta = E225.load_body(CKPT_DIR / G1C_ROOT_CK, Cfg())
    n_par = sum(p.numel() for p in root_net.parameters())
    theta0 = E225.flat_params(root_net)
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    root_read = E225.battery_cell(root_net, ruler_ids, zid)["mean_pz"]
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{G1C_ROOT_CK}",
        "n_params": n_par, "expected_params": 2_739_072,
        "battery_read_measured": root_read,
        "battery_read_committed": G1C_ROOT_GM12,
        "abs_diff": abs(root_read - G1C_ROOT_GM12), "tol": G_READ_TOL,
        "flat_md5": root_md5,
        "flat_md5_matches_e242": bool(root_md5 == G1C_ROOT_FLAT_MD5),
        "pass": bool(n_par == 2_739_072
                     and abs(root_read - G1C_ROOT_GM12) < G_READ_TOL
                     and root_md5 == G1C_ROOT_FLAT_MD5),
    }
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    metrics["gates"]["G_STATES_A"] = G_STATES_A
    log(f"P1: G_ROOT PASS — {n_par} params, battery |d| "
        f"{G_ROOT['abs_diff']:.1e}, flat md5 == e242's")

    fbatt = [{"ids": ruler_ids[i: i + 1],
              "fact": f"install{i:02d}@g-12",
              "relation": "install60_g-12", "ans_id": zid}
             for i in range(ruler_ids.shape[0])]

    def measure_state(tag: str, net) -> dict:
        """One state = one eval burst: the three batteries through e228's
        margin_pass (module-imported, via e229's shim) + the light dial.
        EVERY probe's margin_sigma AND margin_raw AND sigma are kept — the
        decomposition's two sides."""
        loadpct = cpu_load_probe()
        shim = E229._E228NetShim(net)
        out = {"tag": tag, "cpu_load_pct": loadpct,
               "gm12_ruler": G1.battery_cell(net, ruler_ids, zid)["mean_pz"]}
        for name, batt in (("fact", fbatt), ("ctrl", ctrl_probes),
                           ("held30", held_probes)):
            rec = E228.margin_pass(shim, batt)
            out[name] = {
                "aggregate": {"median": rec["median_margin_sigma"],
                              "mean": rec["mean_margin_sigma"],
                              "mean_p": rec["mean_p"],
                              "frac_argmax_answer": rec["frac_argmax_answer"]},
                "probes": [{"fact": r["fact"], "p": r["p"],
                            "margin_sigma": r["margin_sigma"],
                            "margin_raw": r["margin_raw"],
                            "sigma": r["sigma"],
                            "argmax_is_answer": r["argmax_is_answer"]}
                           for r in rec["probes"]]}
        return out

    cells: dict = {}
    cells["t0"] = measure_state("t0", root_net)
    del root_net
    log("  t0 measured (3 batteries, margin_raw + sigma kept)")
    write_partial("P1 t0 measured")

    g_state_rows = {}
    for s in all_steps:
        net = G1.evl_load(wck["sds"][s])
        key = f"w1+{s}"
        cells[key] = measure_state(key, net)
        committed = G1C_W1_GM12[s]
        g_state_rows[s] = {"measured": cells[key]["gm12_ruler"],
                           "committed": committed,
                           "abs_diff": abs(cells[key]["gm12_ruler"] - committed)}
        log(f"  +{s:>3}: ruler {cells[key]['gm12_ruler']:.6f} vs committed "
            f"{committed:.6f} (|d| {g_state_rows[s]['abs_diff']:.1e})")
        del net
        write_partial(f"P1 w1+{s} measured")

    G_STATES = {
        **{k: v for k, v in G_STATES_A.items()},
        "per_state_ruler_reads": {str(k): v for k, v in sorted(g_state_rows.items())},
        "tol": G_READ_TOL,
        "max_abs_diff": max(v["abs_diff"] for v in g_state_rows.values()),
        "pass": bool(G_STATES_A["pass"] and all(
            v["abs_diff"] < G_READ_TOL for v in g_state_rows.values())),
        "note": "every W1 state's light g-12 re-probe vs g1c's committed "
                "adjudication.wall.W1.g_m12, PLUS per-state body-flat md5s "
                "and the resume ckpt's sha asserted EXACTLY EQUAL to "
                "e242's committed provenance — the bit-exactness "
                "certification of the loaded states",
    }
    assert G_STATES["pass"], f"state certification FAILED: {G_STATES}"
    metrics["gates"]["G_STATES"] = G_STATES

    # G_FACT_REPRO: per-probe vs e242's committed per-probe margin_sigma
    fact_devs = []
    for s in [0] + list(all_steps):
        key = "t0" if s == 0 else f"w1+{s}"
        committed_probes = {p["fact"]: p["margin_sigma"]
                            for p in e242m["cells"][key]["probes"]}
        for r in cells[key]["fact"]["probes"]:
            fact_devs.append(abs(r["margin_sigma"]
                                 - committed_probes[r["fact"]]))
    G_FACT_REPRO = {
        "n_pairs": len(fact_devs),
        "tol": REPRO_TOL, "max_abs_diff": max(fact_devs),
        "pass": bool(max(fact_devs) < REPRO_TOL),
        "note": "this cell's per-probe fact margins vs e242's committed "
                "per-probe margin_sigma on the same loads, same imported "
                "instrument, same device — certification, never adjudicates",
    }
    assert G_FACT_REPRO["pass"], f"fact repro FAILED: {G_FACT_REPRO}"

    # G_CTRL_HELD_REPRO: medians vs e251's committed trajectories
    ctrl_devs, held_devs = [], []
    for s in [0] + list(all_steps):
        key = "t0" if s == 0 else f"w1+{s}"
        ctrl_devs.append(abs(cells[key]["ctrl"]["aggregate"]["median"]
                             - E251_CTRL_MED[s]))
        held_devs.append(abs(cells[key]["held30"]["aggregate"]["median"]
                             - E251_HELD_MED[s]))
    G_CTRL_HELD_REPRO = {
        "ctrl_max_abs_diff": max(ctrl_devs), "held30_max_abs_diff": max(held_devs),
        "tol": REPRO_TOL,
        "pass": bool(max(ctrl_devs) < REPRO_TOL and max(held_devs) < REPRO_TOL),
        "note": "this cell's ctrl + held-30 median trajectories vs e251's "
                "committed medians on the same loads — the rebuilt batteries "
                "certified identical",
    }
    assert G_CTRL_HELD_REPRO["pass"], f"ctrl/held repro FAILED: {G_CTRL_HELD_REPRO}"
    metrics["gates"]["G_FACT_REPRO"] = G_FACT_REPRO
    metrics["gates"]["G_CTRL_HELD_REPRO"] = G_CTRL_HELD_REPRO
    log(f"P1 COMPLETE: states certified; fact per-probe repro |d| "
        f"{max(fact_devs):.1e}; ctrl/held repro |d| "
        f"{max(max(ctrl_devs), max(held_devs)):.1e}")
    write_partial("P1 COMPLETE (states certified + 3 batteries measured)")

    # ================= P2: THE REGRESSION ====================================
    # the committed thermal dial (hard-bound): x_s = T_mle(t0) - T_mle(s)
    x_T = {s: E242_T_MLE[0] - E242_T_MLE[s] for s in all_steps}

    def growth_points(batt: str, states) -> tuple[list[float], list[float], list[str]]:
        xs, ys, tags = [], [], []
        m0 = {r["fact"]: r["margin_sigma"] for r in cells["t0"][batt]["probes"]}
        for s in states:
            for r in cells[f"w1+{s}"][batt]["probes"]:
                xs.append(x_T[s])
                ys.append(100.0 * (r["margin_sigma"] / m0[r["fact"]] - 1.0))
                tags.append(f"{r['fact']}@+{s}")
        return xs, ys, tags

    xs_p, ys_p, tags_p = [], [], []
    per_batt_reg = {}
    for batt in ("fact", "ctrl", "held30"):
        xs, ys, _ = growth_points(batt, STATE_GRID)
        per_batt_reg[batt] = ols(xs, ys)
        if batt != "held30":                     # pooled = fact + ctrl
            xs_p += xs
            ys_p += ys
    pooled_reg = ols(xs_p, ys_p)
    # companions (never adjudicate): texture included; state-median; spread dial
    xs_pt, ys_pt = [], []
    for batt in ("fact", "ctrl"):
        xs, ys, _ = growth_points(batt, all_steps)
        xs_pt += xs
        ys_pt += ys
    pooled_reg_with_texture = ols(xs_pt, ys_pt)

    # the direct spread dial T_sigma (the LITERAL spread; T228 carried)
    sigma_ratio = {}          # state -> pooled median_i D_{i,s}/D_{i,0} (fact+ctrl)
    for s in [0] + list(all_steps):
        key = "t0" if s == 0 else f"w1+{s}"
        rs = []
        for batt in ("fact", "ctrl"):
            d0 = {r["fact"]: r["sigma"] for r in cells["t0"][batt]["probes"]}
            rs += [r["sigma"] / d0[r["fact"]] for r in cells[key][batt]["probes"]]
        sigma_ratio[s] = float(np.median(rs))
    x_spread = {s: 1.0 - sigma_ratio[s] for s in all_steps}

    def growth_vs(xdial: dict, states) -> dict:
        """pooled OLS of per-probe growth on an arbitrary per-state dial."""
        xs, ys = [], []
        for batt in ("fact", "ctrl"):
            m0 = {r["fact"]: r["margin_sigma"] for r in cells["t0"][batt]["probes"]}
            for s in states:
                for r in cells[f"w1+{s}"][batt]["probes"]:
                    xs.append(xdial[s])
                    ys.append(100.0 * (r["margin_sigma"] / m0[r["fact"]] - 1.0))
        return ols(xs, ys)

    spread_reg = growth_vs(x_spread, STATE_GRID)

    # the state-median regression (the trajectory's own statistic; co-report)
    st_med_xs, st_med_ys = [], []
    for batt in ("fact", "ctrl"):
        for s in STATE_GRID:
            key = f"w1+{s}"
            st_med_xs.append(x_T[s])
            st_med_ys.append(100.0 * (cells[key][batt]["aggregate"]["median"]
                                      / cells["t0"][batt]["aggregate"]["median"]
                                      - 1.0))
    state_median_reg = ols(st_med_xs, st_med_ys)

    regression_clause = bool(pooled_reg["r2"] >= R2_BAR and pooled_reg["slope"] > 0)
    metrics["regression"] = {
        "x_definition": "x_s = T_mle(t0) - T_mle(s), e242's committed "
                        "thermal leg (an ANSWER-POSITION P-SHAPE fit, not "
                        "the literal spread — T228's answer-locality "
                        "amendment carried)",
        "y_definition": "y = 100*(margin_sigma_{i,s}/margin_sigma_{i,0} - 1)",
        "x_values": {f"+{s}": x_T[s] for s in all_steps},
        "T_fit_committed": {f"+{s}": E242_T_MLE[s] for s in [0] + list(all_steps)},
        "T_sigma_direct_coreport": {f"+{s}": sigma_ratio[s]
                                    for s in [0] + list(all_steps)},
        "pooled_fact_ctrl_ADJUDICATES": pooled_reg,
        "per_battery": per_batt_reg,
        "pooled_with_texture_coreport": pooled_reg_with_texture,
        "state_median_reg_coreport": state_median_reg,
        "direct_spread_reg_coreport": spread_reg,
        "regression_clause": regression_clause,
        "clause_definition": "(pooled R^2 >= 0.6) AND (slope > 0)",
    }
    log(f"P2: pooled regression slope {pooled_reg['slope']:+.2f} pct/deg, "
        f"R2 {pooled_reg['r2']:.4f} (n={pooled_reg['n']}) -> clause "
        f"{regression_clause}; per-battery fact "
        f"R2 {per_batt_reg['fact']['r2']:.4f}, ctrl "
        f"R2 {per_batt_reg['ctrl']['r2']:.4f}; state-median co-report R2 "
        f"{state_median_reg['r2']:.4f} slope "
        f"{state_median_reg['slope']:+.2f}")
    write_partial("P2 the regression computed")

    # ================= P3: THE DECOMPOSITION =================================
    # per probe EXACTLY: log(m ratio) = log(N ratio) - log(D ratio)
    decomp = {}
    ident_max_dev = 0.0
    for batt in ("fact", "ctrl", "held30"):
        rows = {}
        n0N = {r["fact"]: r["margin_raw"] for r in cells["t0"][batt]["probes"]}
        n0D = {r["fact"]: r["sigma"] for r in cells["t0"][batt]["probes"]}
        n0M = {r["fact"]: r["margin_sigma"] for r in cells["t0"][batt]["probes"]}
        for s in all_steps:
            cn, cd, cm = [], [], []
            n_excl = 0
            for r in cells[f"w1+{s}"][batt]["probes"]:
                f_ = r["fact"]
                nrat = r["margin_raw"] / n0N[f_]
                drat = r["sigma"] / n0D[f_]
                mrat = r["margin_sigma"] / n0M[f_]
                # the exact identity check (only where all three finite)
                if all(np.isfinite(v) and v > 0 for v in (nrat, drat, mrat)):
                    ident_max_dev = max(
                        ident_max_dev,
                        abs(np.log(mrat) - (np.log(nrat) - np.log(drat))))
                    cn.append(np.log(nrat))
                    cd.append(-np.log(drat))     # SIGNED contribution to growth
                    cm.append(np.log(mrat))
                else:
                    n_excl += 1
            rows[s] = {
                "c_num_median": float(np.median(cn)) if cn else float("nan"),
                "c_den_median": float(np.median(cd)) if cd else float("nan"),
                "c_total_median": float(np.median(cm)) if cm else float("nan"),
                "c_num_mean": float(np.mean(cn)) if cn else float("nan"),
                "c_den_mean": float(np.mean(cd)) if cd else float("nan"),
                "n_excluded_zero_or_nonfinite": n_excl,
                # levels (the table's currency)
                "median_N": float(np.median(
                    [r["margin_raw"] for r in cells[f"w1+{s}"][batt]["probes"]])),
                "median_D": float(np.median(
                    [r["sigma"] for r in cells[f"w1+{s}"][batt]["probes"]])),
                "median_M": float(np.median(
                    [r["margin_sigma"] for r in cells[f"w1+{s}"][batt]["probes"]])),
            }
            rows[s]["median_D_t0"] = float(np.median(list(n0D.values())))
            rows[s]["median_N_t0"] = float(np.median(list(n0N.values())))
        decomp[batt] = rows
    assert ident_max_dev < 1e-9, \
        f"the exact identity log(m)=log(N)-log(D) BROKE: {ident_max_dev}"

    def carries_and_inert(batt: str) -> dict:
        r300 = decomp[batt][300]
        return {
            "c_num_300": r300["c_num_median"],
            "c_den_300": r300["c_den_median"],
            "denominator_carries": bool(r300["c_den_median"] < 0
                                        and abs(r300["c_den_median"])
                                        > abs(r300["c_num_median"])),
            "numerator_inert": bool(abs(r300["c_num_median"]) <= INERT_BAND_LOG),
        }

    fact_ci, ctrl_ci = carries_and_inert("fact"), carries_and_inert("ctrl")
    denominator_carries = bool(fact_ci["denominator_carries"]
                               and ctrl_ci["denominator_carries"])
    numerator_inert = bool(fact_ci["numerator_inert"]
                           and ctrl_ci["numerator_inert"])
    decomposition_clause = bool(denominator_carries and numerator_inert)
    metrics["decomposition"] = {
        "identity": "per probe EXACTLY log(m_{i,s}/m_{i,0}) = "
                    "log(N_{i,s}/N_{i,0}) - log(D_{i,s}/D_{i,0}); N = "
                    "e228.margin_raw (top1-top2, the probe's own aligned "
                    "logit component); D = e228.sigma (std of the vocab "
                    "logits at the answer position — the state's logit "
                    "spread; LN's denominator in margin units)",
        "identity_max_abs_dev_asserted": ident_max_dev,
        "convention": "c_den is SIGNED toward growth (c_den = -log(D ratio): "
                      "a SHRINKING denominator contributes POSITIVE growth); "
                      "medians over probes; |c_num| <= 0.05 log = the "
                      "'~inert' band",
        "walled": {batt: {f"+{s}": decomp[batt][s] for s in all_steps}
                   for batt in ("fact", "ctrl", "held30")},
        "fact_300": fact_ci, "ctrl_300": ctrl_ci,
        "denominator_carries": denominator_carries,
        "numerator_inert": numerator_inert,
        "decomposition_clause": decomposition_clause,
        "clause_definition": "DENOMINATOR-CARRIES (at +300, both fact and "
                             "ctrl: c_den<0 AND |c_den|>|c_num|) AND "
                             "NUMERATOR-INERT (at +300, both: |c_num|<=0.05)",
    }
    log(f"P3: decomposition at +300 — fact c_num {fact_ci['c_num_300']:+.4f} "
        f"/ c_den {fact_ci['c_den_300']:+.4f}; ctrl c_num "
        f"{ctrl_ci['c_num_300']:+.4f} / c_den {ctrl_ci['c_den_300']:+.4f} -> "
        f"denominator_carries={denominator_carries}, "
        f"numerator_inert={numerator_inert}, clause={decomposition_clause}")
    write_partial("P3 the decomposition computed")

    # ================= P4: THE CROSS-WORLD CONTROL (pure desk) ==============
    # e238's committed logit dumps -> the unwalled decomposition
    unw = {}
    npz_data = {}
    for s in [0] + list(UNWALLED_GRID):
        d = np.load(E238_NPZ[s])
        lg = d["logits"].astype(np.float64)
        top2 = np.sort(lg, axis=1)[:, -2:]
        num = top2[:, 1] - top2[:, 0]
        den = lg.std(axis=1, ddof=1)          # torch-unbiased equivalent
        unw[s] = {"names": [str(n) for n in d["names"]],
                  "battery": [str(b) for b in d["battery"]],
                  "num": num, "den": den, "m": num / den}
        npz_data[s] = {"sha256_16": sha256_of(E238_NPZ[s])}
    G_UNWALLED_NPZ = {
        "files": {f"w1+{s}" if s else "t0": v["sha256_16"]
                  for s, v in npz_data.items()},
        "expected_sha256_16": {f"w1+{s}" if s else "t0": E238_NPZ_SHA[s]
                               for s in [0] + list(UNWALLED_GRID)},
        "shape": list(unw[0]["m"].shape),
        "pass": bool(all(npz_data[s]["sha256_16"] == E238_NPZ_SHA[s]
                         for s in [0] + list(UNWALLED_GRID))),
        "note": "e238's committed logit dumps, sha-gated against its own "
                "provenance records",
    }
    assert G_UNWALLED_NPZ["pass"], f"npz gate FAILED: {G_UNWALLED_NPZ}"
    metrics["gates"]["G_UNWALLED_NPZ"] = G_UNWALLED_NPZ

    # the n=8 join vs e232's committed margins
    idx_of = {nm: i for i, nm in enumerate(unw[0]["names"])}
    join_devs = []
    for nm, (m0c, m80c) in E232_FACT_W1.items():
        join_devs.append(abs(unw[0]["m"][idx_of[nm]] - m0c))
        join_devs.append(abs(unw[80]["m"][idx_of[nm]] - m80c))
    G_UNWALLED_JOIN = {
        "n_joined": len(E232_FACT_W1), "tol": UNWALLED_JOIN_TOL,
        "max_abs_diff": max(join_devs),
        "pass": bool(max(join_devs) < UNWALLED_JOIN_TOL),
        "note": "margin_sigma recomputed from e238's committed logits vs "
                "e232's committed fact-w1 margins on the shared n=8 — the "
                "unwalled decomposition's certification",
    }
    assert G_UNWALLED_JOIN["pass"], f"unwalled join FAILED: {G_UNWALLED_JOIN}"
    metrics["gates"]["G_UNWALLED_JOIN"] = G_UNWALLED_JOIN

    n8 = list(E232_FACT_W1.keys())
    x_T_unw = {s: 1.0 - E238_T_MLE_W1[s] for s in UNWALLED_GRID}
    unw_growth = [100.0 * (unw[80]["m"][idx_of[nm]] / unw[0]["m"][idx_of[nm]] - 1.0)
                  for nm in n8]
    unw_growth_mean = float(np.mean(unw_growth))
    unw_growth_median = float(np.median(unw_growth))
    # the unwalled decomposition (n=8 primary; n=20 co-report)
    def unw_decomp(names: list[str], s: int) -> dict:
        cn, cd = [], []
        n_excl = 0
        for nm in names:
            i = idx_of[nm]
            nrat = unw[s]["num"][i] / unw[0]["num"][i]
            drat = unw[s]["den"][i] / unw[0]["den"][i]
            if nrat > 0 and drat > 0 and np.isfinite(nrat) and np.isfinite(drat):
                cn.append(np.log(nrat))
                cd.append(-np.log(drat))
            else:
                n_excl += 1
        return {"c_num_median": float(np.median(cn)),
                "c_den_median": float(np.median(cd)),
                "n_excluded": n_excl,
                "median_D_ratio": float(np.median(
                    [unw[s]["den"][idx_of[nm]] / unw[0]["den"][idx_of[nm]]
                     for nm in names]))}

    unw_dec_n8 = {s: unw_decomp(n8, s) for s in UNWALLED_GRID}
    fact20 = [nm for nm, b in zip(unw[0]["names"], unw[0]["battery"])
              if b == "fact"]
    unw_dec_n20 = {s: unw_decomp(fact20, s) for s in UNWALLED_GRID}

    walled_fact_growth = 100.0 * (E242_MEDIANS[300] / E242_MEDIANS[0] - 1.0)
    walled_T_decline_300 = E242_T_MLE[0] - E242_T_MLE[300]
    unw_T_decline_80 = 1.0 - E238_T_MLE_W1[80]
    unw_num_carried = bool(unw_dec_n8[80]["c_num_median"] < 0
                           and abs(unw_dec_n8[80]["c_num_median"])
                           > abs(unw_dec_n8[80]["c_den_median"]))
    control_clause = bool(walled_fact_growth > 0
                          and walled_T_decline_300 > 0
                          and unw_growth_median < COLLAPSE_PCT
                          and unw_T_decline_80 < 0
                          and unw_num_carried)
    # the two-world pooled regression (co-report)
    xs_2w, ys_2w = list(xs_p), list(ys_p)
    for nm in n8:
        i = idx_of[nm]
        for s in UNWALLED_GRID:
            xs_2w.append(x_T_unw[s])
            ys_2w.append(100.0 * (unw[s]["m"][i] / unw[0]["m"][i] - 1.0))
    twoworld_reg = ols(xs_2w, ys_2w)

    metrics["crossworld"] = {
        "worlds": "WALLED = g1c W1 (2.74M, R=0.7): margins THICKEN while "
                  "committed T FALLS; UNWALLED = the 124M two-wash archive's "
                  "w1 (e182's organism): margins COLLAPSE while committed T "
                  "RISES",
        "unwalled_T_mle_committed": {f"+{s}": E238_T_MLE_W1[s]
                                     for s in UNWALLED_GRID},
        "unwalled_T_decline": {f"+{s}": x_T_unw[s] for s in UNWALLED_GRID},
        "unwalled_fact_w1_growth_pct_n8": {
            "mean": unw_growth_mean, "median": unw_growth_median,
            "per_probe": {nm: 100.0 * (unw[80]["m"][idx_of[nm]]
                                        / unw[0]["m"][idx_of[nm]] - 1.0)
                          for nm in n8}},
        "unwalled_decomposition_n8_PRIMARY": {f"+{s}": unw_dec_n8[s]
                                              for s in UNWALLED_GRID},
        "unwalled_decomposition_fact20_coreport": {f"+{s}": unw_dec_n20[s]
                                                   for s in UNWALLED_GRID},
        "walled_side_committed": {
            "fact_growth_pct_300": walled_fact_growth,
            "T_decline_300": walled_T_decline_300},
        "unwalled_side": {
            "T_decline_80": unw_T_decline_80,
            "numerator_carried_n8": unw_num_carried},
        "twoworld_pooled_reg_coreport": twoworld_reg,
        "control_clause": control_clause,
        "clause_definition": "walled fact growth(+300)>0 AND walled "
                             "T-decline(+300)>0 AND unwalled fact "
                             "growth(+80)<-25% AND unwalled "
                             "T-decline(+80)<0 AND unwalled "
                             "NUMERATOR-carried (c_num<0 AND |c_num|>|c_den|, "
                             "n=8)",
    }
    log(f"P4: cross-world — walled {walled_fact_growth:+.2f}% @ T-decline "
        f"{walled_T_decline_300:+.3f} vs unwalled "
        f"{unw_growth_median:+.2f}% (n=8 median) @ T-decline "
        f"{unw_T_decline_80:+.3f}; unwalled c_num "
        f"{unw_dec_n8[80]['c_num_median']:+.4f} vs c_den "
        f"{unw_dec_n8[80]['c_den_median']:+.4f} -> control_clause="
        f"{control_clause}")
    write_partial("P4 the cross-world control computed")

    # ================= P5: the frozen adjudication ===========================
    hard_gates = {g: bool(v.get("pass")) for g, v in metrics["gates"].items()}
    gates_ok = all(hard_gates.values())

    identity_fires = bool(gates_ok and regression_clause
                          and decomposition_clause and control_clause)
    fact_cn300 = fact_ci["c_num_300"]
    ctrl_cn300 = ctrl_ci["c_num_300"]
    residue_fires = bool(gates_ok and not identity_fires
                         and fact_cn300 > INERT_BAND_LOG
                         and fact_cn300 > ctrl_cn300)
    mixed_fires = not (identity_fires or residue_fires)

    if identity_fires:
        verdict = "IDENTITY-CONFIRMED"
        clause = (f"margin growth tracks the T-decline quantitatively (pooled "
                  f"R2 {pooled_reg['r2']:.3f} >= 0.6, slope "
                  f"{pooled_reg['slope']:+.2f} pct/deg) AND the decomposition "
                  f"shows the DENOMINATOR carrying the thickening (fact c_den "
                  f"{fact_ci['c_den_300']:+.4f} vs c_num "
                  f"{fact_cn300:+.4f}; ctrl c_den "
                  f"{ctrl_ci['c_den_300']:+.4f} vs c_num "
                  f"{ctrl_cn300:+.4f}; numerator ~inert in both) AND the "
                  f"unwalled control flips sign (unwalled "
                  f"{unw_growth_median:+.1f}% @ T-decline "
                  f"{unw_T_decline_80:+.3f}, numerator-carried — vs walled "
                  f"{walled_fact_growth:+.1f}% @ "
                  f"{walled_T_decline_300:+.3f}, denominator-carried) — the "
                  f"budget's origin named: THE COOLING; W038's law 4 "
                  f"collapses to 'the wall cools; LN does the rest'")
    elif residue_fires:
        verdict = "NUMERATOR-RESIDUE"
        residue_size_log = fact_cn300
        residue_size_pct = 100.0 * (float(np.exp(fact_cn300)) - 1.0)
        ctrl_size_pct = 100.0 * (float(np.exp(ctrl_cn300)) - 1.0)
        clause = (f"the identity did NOT fully confirm (regression clause "
                  f"{regression_clause} (pooled R2 {pooled_reg['r2']:.3f}, "
                  f"slope {pooled_reg['slope']:+.2f}); decomposition clause "
                  f"{decomposition_clause}; control clause {control_clause}) "
                  f"AND the fact battery's numerator grows beyond ctrl "
                  f"(fact c_num {fact_cn300:+.4f} log = "
                  f"{residue_size_pct:+.2f}% vs ctrl {ctrl_cn300:+.4f} log = "
                  f"{ctrl_size_pct:+.2f}%) — a battery-specific constructive "
                  f"residue; the re-formation layer keeps a small real "
                  f"object of size {residue_size_pct:+.2f}% (log "
                  f"{residue_size_log:+.4f}) at +300, honestly reported")
    else:
        why = []
        if not gates_ok:
            failed = [g for g, v in hard_gates.items() if not v]
            why.append(f"the bars STAND DOWN: hard gate(s) FAILED {failed}")
        else:
            why.append(
                f"regression clause {regression_clause} (pooled R2 "
                f"{pooled_reg['r2']:.3f} vs the 0.6 bar, slope "
                f"{pooled_reg['slope']:+.2f} pct/deg); decomposition clause "
                f"{decomposition_clause} (denominator_carries="
                f"{denominator_carries}, numerator_inert={numerator_inert}; "
                f"fact c_num {fact_cn300:+.4f}/c_den "
                f"{fact_ci['c_den_300']:+.4f}, ctrl c_num "
                f"{ctrl_cn300:+.4f}/c_den {ctrl_ci['c_den_300']:+.4f}); "
                f"control clause {control_clause}")
        clause = ("; ".join(why)
                  + " — the tables verbatim, both batteries, both worlds")

    metrics["adjudication"] = {
        "bars": {"IDENTITY-CONFIRMED": {"fires": identity_fires},
                 "NUMERATOR-RESIDUE": {"fires": residue_fires},
                 "MIXED": {"fires": mixed_fires}},
        "clause_fixes_applied": REGISTERED_BARS["clause_fixes"],
        "verdict": verdict, "clause": clause,
        "reads": {
            "pooled_regression": pooled_reg,
            "per_battery_regression": per_batt_reg,
            "state_median_regression_coreport": state_median_reg,
            "direct_spread_regression_coreport": spread_reg,
            "regression_clause": regression_clause,
            "decomposition_clause": decomposition_clause,
            "denominator_carries": denominator_carries,
            "numerator_inert": numerator_inert,
            "fact_c_num_300": fact_cn300,
            "fact_c_den_300": fact_ci["c_den_300"],
            "ctrl_c_num_300": ctrl_cn300,
            "ctrl_c_den_300": ctrl_ci["c_den_300"],
            "control_clause": control_clause,
            "walled_fact_growth_pct_300_committed": walled_fact_growth,
            "walled_T_decline_300_committed": walled_T_decline_300,
            "unwalled_fact_growth_pct_80_median_n8": unw_growth_median,
            "unwalled_T_decline_80_committed": unw_T_decline_80,
            "unwalled_numerator_carried": unw_num_carried,
            "hard_gates_all_pass": gates_ok,
        },
        "gates_summary": hard_gates,
    }
    log("=" * 78)
    log(f"E255 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)
    write_partial("P5 the frozen bars adjudicated")

    # ================= P6: the figures =======================================
    steps_lab = [f"+{s}" for s in all_steps]

    # (1) THE REGRESSION
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 7.0))
    ax = axes[0]
    xf, yf, _ = growth_points("fact", STATE_GRID)
    xc, yc, _ = growth_points("ctrl", STATE_GRID)
    ax.scatter(xf, yf, s=14, alpha=0.45, color="tab:red", label=f"fact probes (n={len(xf)})")
    ax.scatter(xc, yc, s=14, alpha=0.45, color="tab:blue", label=f"ctrl probes (n={len(xc)})")
    xs_line = np.array(sorted(set(x_T[s] for s in STATE_GRID)))
    ax.plot(xs_line, pooled_reg["slope"] * xs_line + pooled_reg["intercept"],
            "k-", lw=2,
            label=f"pooled OLS: slope {pooled_reg['slope']:+.1f}, R2 "
                  f"{pooled_reg['r2']:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("T-decline: T_mle(t0) - T_mle(state)  [committed e242; "
                  "answer-position p-shape fit]")
    ax.set_ylabel("per-probe margin growth (%)")
    ax.set_title(f"(a) THE REGRESSION (adjudicating states) — clause "
                 f"{regression_clause} (R2 bar 0.6)", fontsize=9.5)
    ax.legend(fontsize=7.6, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for batt, col in (("fact", "tab:red"), ("ctrl", "tab:blue"),
                      ("held30", "tab:green")):
        xs_b, ys_b = [], []
        for s in STATE_GRID:
            key = f"w1+{s}"
            xs_b.append(x_T[s])
            ys_b.append(100.0 * (cells[key][batt]["aggregate"]["median"]
                                 / cells["t0"][batt]["aggregate"]["median"] - 1.0))
        ax.plot(xs_b, ys_b, "o-", color=col, ms=8, lw=1.8,
                label=f"{batt} MEDIAN growth")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    for s in STATE_GRID:
        ax.annotate(f"+{s}", (x_T[s], ax.get_ylim()[0]), fontsize=7,
                    ha="center", va="bottom", color="dimgray")
    ax.set_xlabel("T-decline (same dial as (a))")
    ax.set_ylabel("battery-median margin growth (%)")
    ax.set_title(f"(b) STATE-MEDIAN co-report — R2 "
                 f"{state_median_reg['r2']:.3f}, slope "
                 f"{state_median_reg['slope']:+.1f} (never adjudicates)",
                 fontsize=9.5)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axis("off")
    y = 0.96
    ax.text(0.03, y, f"E255 (1) THE REGRESSION — clause: {regression_clause}",
            fontsize=11, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.05
    ax.text(0.03, y, "state   T_mle    x=T-decline   D-spread(T_sigma)",
            fontsize=7.6, va="top", family="monospace", weight="bold")
    y -= 0.024
    for s in [0] + list(all_steps):
        lab = "t0" if s == 0 else f"+{s}"
        role = "" if s == 0 else (" (adjud)" if s in STATE_GRID else " (texture)")
        ax.text(0.03, y,
                f"{lab:>5}  {E242_T_MLE[s]:7.4f}  {x_T[s]:+.4f}{role:<10} "
                f"{sigma_ratio[s]:.4f}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.022
    y -= 0.02
    ax.text(0.03, y, f"POOLED (fact+ctrl, adjudicating): slope "
            f"{pooled_reg['slope']:+.2f} pct/deg, R2 {pooled_reg['r2']:.4f} "
            f"(n={pooled_reg['n']})  [BAR: R2 >= 0.6 AND slope > 0]",
            fontsize=7.4, va="top", family="monospace")
    y -= 0.026
    for batt in ("fact", "ctrl", "held30"):
        r = per_batt_reg[batt]
        ax.text(0.03, y, f"  per-battery {batt:>6}: slope {r['slope']:+8.2f}  "
                f"R2 {r['r2']:.4f}  (n={r['n']})",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.022
    y -= 0.012
    ax.text(0.03, y, f"co-reports: with texture R2 "
            f"{pooled_reg_with_texture['r2']:.4f}; direct-spread dial R2 "
            f"{spread_reg['r2']:.4f} (slope {spread_reg['slope']:+.2f})",
            fontsize=7.4, va="top", family="monospace", color="dimgray")
    y -= 0.03
    for wd in textwrap.wrap(
            "T228 carried: the x dial is e242's committed T_mle — an "
            "answer-position p-shape fit, NOT the literal logit spread; the "
            "T_sigma column is the literal spread dial (median D ratio).",
            width=68):
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace",
                color="dimgray")
        y -= 0.02
    fig.suptitle("E255 (1) THE THERMAL-LEDGER REGRESSION — per-probe margin "
                 "growth vs the committed T-decline (g1c W1, wall R=0.7)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "e255_regression.png", dpi=130)
    plt.close(fig)

    # (2) THE DECOMPOSITION
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 7.0))
    ax = axes[0]
    xs_arr = np.arange(len(all_steps))
    w = 0.36
    for k, (batt, col) in enumerate((("fact", "tab:red"), ("ctrl", "tab:blue"))):
        ax.bar(xs_arr - w / 2 + k * w,
               [decomp[batt][s]["c_num_median"] for s in all_steps],
               width=w, color=col, alpha=0.9,
               label=f"{batt}: NUMERATOR log-contrib" if k == 0 else None)
        ax.bar(xs_arr - w / 2 + k * w,
               [decomp[batt][s]["c_den_median"] for s in all_steps],
               width=w, color=col, alpha=0.35, hatch="//",
               label=f"{batt}: DENOMINATOR log-contrib" if k == 0 else None)
    ax.axhline(0, color="k", lw=0.9)
    ax.axhline(INERT_BAND_LOG, color="gray", ls=":", lw=1.0)
    ax.axhline(-INERT_BAND_LOG, color="gray", ls=":", lw=1.0)
    ax.set_xticks(xs_arr)
    ax.set_xticklabels(steps_lab, fontsize=8)
    ax.set_xlabel("wash step (solid = numerator c_num; hatched = "
                  "denominator c_den, signed toward growth)")
    ax.set_ylabel("battery-median log contribution")
    ax.set_title("(a) THE DECOMPOSITION — which side carries the thickening "
                 "(dotted = the ~inert band)", fontsize=9.5)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for batt, col in (("fact", "tab:red"), ("ctrl", "tab:blue")):
        ax.plot([0] + list(xs_arr),
                [decomp[batt][1]["median_D_t0"]]
                + [decomp[batt][s]["median_D"] for s in all_steps],
                "s-", color=col, ms=7, lw=1.8, label=f"{batt}: median D "
                f"(the literal spread)")
        ax.plot([0] + list(xs_arr),
                [decomp[batt][1]["median_N_t0"]]
                + [decomp[batt][s]["median_N"] for s in all_steps],
                "o--", color=col, ms=6, lw=1.4, alpha=0.8,
                label=f"{batt}: median N (the aligned gap)")
    ax.set_xticks([0] + list(xs_arr))
    ax.set_xticklabels(["t0"] + steps_lab, fontsize=8)
    ax.set_xlabel("wash step")
    ax.set_ylabel("levels (logits)")
    ax.set_title("(b) THE LEVELS — N and D trajectories (fact vs ctrl)",
                 fontsize=9.5)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axis("off")
    y = 0.96
    ax.text(0.03, y, f"E255 (2) THE DECOMPOSITION — clause: "
            f"{decomposition_clause}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    ax.text(0.03, y, "state | fact c_num  c_den | ctrl c_num  c_den | "
            "fact N     D    | ctrl N     D", fontsize=7.4, va="top",
            family="monospace", weight="bold")
    y -= 0.024
    ax.text(0.03, y, f"{'t0':>5} | {'-':>10s} {'-':>7s} | {'-':>10s} "
            f"{'-':>7s} | {decomp['fact'][1]['median_N_t0']:7.3f} "
            f"{decomp['fact'][1]['median_D_t0']:6.3f} | "
            f"{decomp['ctrl'][1]['median_N_t0']:7.3f} "
            f"{decomp['ctrl'][1]['median_D_t0']:6.3f}",
            fontsize=7.4, va="top", family="monospace")
    y -= 0.022
    for s in all_steps:
        ax.text(0.03, y,
                f"+{s:>4} | {decomp['fact'][s]['c_num_median']:+10.4f} "
                f"{decomp['fact'][s]['c_den_median']:+7.4f} | "
                f"{decomp['ctrl'][s]['c_num_median']:+10.4f} "
                f"{decomp['ctrl'][s]['c_den_median']:+7.4f} | "
                f"{decomp['fact'][s]['median_N']:7.3f} "
                f"{decomp['fact'][s]['median_D']:6.3f} | "
                f"{decomp['ctrl'][s]['median_N']:7.3f} "
                f"{decomp['ctrl'][s]['median_D']:6.3f}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.022
    y -= 0.02
    ax.text(0.03, y, f"+300: denominator_carries={denominator_carries} "
            f"numerator_inert={numerator_inert} (|c_num|<=0.05 in BOTH "
            f"fact and ctrl)", fontsize=7.6, va="top", family="monospace",
            weight="bold")
    y -= 0.03
    for wd in textwrap.wrap(
            "Exact per probe: log(m ratio) = log(N ratio) - log(D ratio) "
            "(asserted to machine precision); c_den is signed toward "
            "growth: a SHRINKING denominator contributes positively.",
            width=68):
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace",
                color="dimgray")
        y -= 0.02
    fig.suptitle("E255 (2) THE NUMERATOR/DENOMINATOR DECOMPOSITION — the "
                 "sigma-normalized margin's two sides through the wall's "
                 "flat phase", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "e255_decomposition.png", dpi=130)
    plt.close(fig)

    # (3) THE CROSS-WORLD CONTRAST
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 7.0))
    ax = axes[0]
    ax.scatter(xf, yf, s=13, alpha=0.4, color="tab:red",
               label="WALLED: fact probes")
    ax.scatter(xc, yc, s=13, alpha=0.4, color="tab:blue",
               label="WALLED: ctrl probes")
    xu = [x_T_unw[s] for s in UNWALLED_GRID
          for nm in n8]
    yu = [100.0 * (unw[s]["m"][idx_of[nm]] / unw[0]["m"][idx_of[nm]] - 1.0)
          for s in UNWALLED_GRID for nm in n8]
    ax.scatter(xu, yu, s=44, marker="X", color="black",
               label=f"UNWALLED: e232's fact n=8 (x{len(UNWALLED_GRID)} states)")
    xs_line = np.linspace(min(xs_2w) - 0.05, max(xs_2w) + 0.05, 50)
    ax.plot(xs_line, twoworld_reg["slope"] * xs_line + twoworld_reg["intercept"],
            "k-", lw=1.8,
            label=f"two-world pooled OLS: slope {twoworld_reg['slope']:+.1f}, "
                  f"R2 {twoworld_reg['r2']:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.annotate("walled: THICKENS\nwhile T FALLS", (0.02, 0.97),
                xycoords="axes fraction", fontsize=8, va="top", color="darkblue")
    ax.annotate("unwalled: COLLAPSES\nwhile T RISES", (0.02, 0.03),
                xycoords="axes fraction", fontsize=8, color="black")
    ax.set_xlabel("T-decline (walled: e242's T_mle; unwalled: e238's T_mle, "
                  "T(t0):=1)")
    ax.set_ylabel("per-probe margin growth (%)")
    ax.set_title("(a) THE CROSS-WORLD CONTRAST — the sign flip", fontsize=9.5)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels = ["WALLED\nfact +300", "WALLED\nctrl +300", "UNWALLED\nfact n=8 +80"]
    cnums = [fact_cn300, ctrl_cn300, unw_dec_n8[80]["c_num_median"]]
    cdens = [fact_ci["c_den_300"], ctrl_ci["c_den_300"],
             unw_dec_n8[80]["c_den_median"]]
    xpos = np.arange(3)
    ax.bar(xpos - 0.18, cnums, width=0.36, color="tab:orange",
           label="c_num (the aligned gap)")
    ax.bar(xpos + 0.18, cdens, width=0.36, color="teal",
           label="c_den (the logit spread; signed toward growth)")
    for i, (n_, d_) in enumerate(zip(cnums, cdens)):
        ax.annotate(f"{n_:+.3f}", (xpos[i] - 0.18, n_), fontsize=7.5,
                    ha="center", va="bottom" if n_ > 0 else "top")
        ax.annotate(f"{d_:+.3f}", (xpos[i] + 0.18, d_), fontsize=7.5,
                    ha="center", va="bottom" if d_ > 0 else "top")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("battery-median log contribution")
    ax.set_title("(b) THE DECOMPOSITION MIRROR — which side moves in each "
                 "world", fontsize=9.5)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ax.axis("off")
    y = 0.96
    ax.text(0.03, y, f"E255 VERDICT: {verdict}", fontsize=11.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    for wd in textwrap.wrap(clause, width=64, break_long_words=False)[:14]:
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace")
        y -= 0.023
    y -= 0.02
    ax.text(0.03, y, "clauses: regression "
            f"{'PASS' if regression_clause else 'FAIL'} | decomposition "
            f"{'PASS' if decomposition_clause else 'FAIL'} | control "
            f"{'PASS' if control_clause else 'FAIL'} | gates "
            f"{'PASS' if gates_ok else 'FAIL'}", fontsize=7.6, va="top",
            family="monospace", weight="bold")
    y -= 0.03
    ax.text(0.03, y, "unwalled decomposition (n=8, committed archive):",
            fontsize=7.4, va="top", family="monospace", weight="bold")
    y -= 0.024
    for s in UNWALLED_GRID:
        ax.text(0.03, y, f"  +{s:>3}: c_num {unw_dec_n8[s]['c_num_median']:+.4f}"
                f"  c_den {unw_dec_n8[s]['c_den_median']:+.4f}"
                f"  (median D ratio {unw_dec_n8[s]['median_D_ratio']:.4f})",
                fontsize=7.2, va="top", family="monospace")
        y -= 0.022
    y -= 0.02
    ax.text(0.03, y, f"walled +300: fact {walled_fact_growth:+.2f}% @ "
            f"T-decline {walled_T_decline_300:+.3f}; unwalled +80 (n=8 "
            f"median) {unw_growth_median:+.2f}% @ T-decline "
            f"{unw_T_decline_80:+.3f}", fontsize=7.4, va="top",
            family="monospace", color="dimgray")
    fig.suptitle("E255 (3) THE CROSS-WORLD CONTROL — the erosion-redistribution "
                 "falsifier vs the thermal contraction", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "e255_crossworld.png", dpi=130)
    plt.close(fig)

    # ================= P7: honesty + provenance + close ======================
    metrics["honesty"] = {
        "which_T": "the regression's x is e242's committed T_mle — a "
                   "Bernoulli p-shape fit at the answer positions (T228's "
                   "answer-locality amendment CARRIED: it is NOT the literal "
                   "logit spread); the decomposition's D and the T_sigma "
                   "co-report are the literal spread — the read never mixes "
                   "the two without saying so",
        "regression_power": f"the pooled per-probe R2 "
                            f"({pooled_reg['r2']:.4f}) is dominated by "
                            "within-state probe noise; the state-median "
                            "co-report shows where the tracking lives — "
                            "both are reported; the frozen 0.6 bar was "
                            "applied to the POOLED form as registered",
        "decomposition_exactness": "the per-probe identity log(m ratio) = "
                                   "log(N ratio) - log(D ratio) is asserted "
                                   "to machine precision (no residual term, "
                                   "no approximation); medians summarize, "
                                   "probes decompose",
        "crossworld_scope": "the unwalled world is a DIFFERENT ORGANISM "
                            "(the 124M e182 lineage) at a different wash "
                            "(lr 5e-5-class, 80 steps) — the control is the "
                            "ideator's own falsifier design (contrast, not "
                            "matched pair); its margins are the committed "
                            "n=8 journal subset, its decomposition certified "
                            "against them (max |d| "
                            f"{G_UNWALLED_JOIN['max_abs_diff']:.1e})",
        "n_and_scope": "ONE lineage (g1c's fresh root, ONE wall commit R=0.7, "
                       "ONE wash draw seed 10902), ONE ctrl battery draw "
                       "(seed 26502, e251's), n=1 deterministic reads per "
                       "state (T204); the heights lottery (W028/R64) stands; "
                       "nothing guaranteed — the outcome is recorded "
                       "verbatim against the frozen bars",
        "nothing_guaranteed": "the tables could have landed anywhere; no bar "
                              "shopping",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "parents": {"g1c_metrics": G_PARENTS["g1c_metrics"],
                    "e242_metrics": G_PARENTS["e242_metrics"],
                    "e251_metrics": G_PARENTS["e251_metrics"],
                    "e232_metrics": G_PARENTS["e232_metrics"],
                    "e238_metrics": G_PARENTS["e238_metrics"]},
        "checkpoints": {
            "t0_root": {"file": f"runs/checkpoints/{G1C_ROOT_CK}",
                        "flat_md5": G_ROOT["flat_md5"],
                        "n_params": G_ROOT["n_params"],
                        "battery_read": G_ROOT["battery_read_measured"]},
            "w1_states": {"file": f"runs/checkpoints/{G1C_W1_RESUME_CK}",
                          "sha256_16": G_STATES_A["sha256_16"],
                          "step": G_STATES_A["step_field"],
                          "wall_R": G_STATES_A["wall_R"],
                          "sds_steps": G_STATES_A["sds_steps"],
                          "sds300_bit_identical_to_final_model":
                              G_STATES_A["sds300_bit_identical_to_final_model"],
                          "per_state_body_flat_md5":
                              {f"w1+{s}": state_body_mds[s] for s in all_steps}},
        },
        "unwalled_archive": {
            "logit_dumps": {f"w1+{s}" if s else "t0":
                            {"file": str(E238_NPZ[s]),
                             "sha256_16": npz_data[s]["sha256_16"]}
                            for s in [0] + list(UNWALLED_GRID)},
            "margins": "runs/e232/metrics.json (committed fact-w1 n=8)",
            "T_mle": "runs/e238/metrics.json (committed two_moment_bound "
                     "T_full = the full-fit T_mle; e252's G_TFIT source)",
        },
        "machinery": {
            "margin_instrument": "lab/e228_margin_landscape.py margin_pass "
                                 "MODULE-IMPORTED via e229's _E228NetShim "
                                 "(adapter only; arithmetic untouched); "
                                 "margin_sigma = margin_raw/sigma with "
                                 "margin_raw = top1-top2 logit gap (THE "
                                 "NUMERATOR) and sigma = torch-unbiased "
                                 "std(vocab logits) (THE DENOMINATOR) — "
                                 "e228/e229/e242/e251's exact instrument",
            "state_loads": "t0 via e225.load_body; W1 states via g1's "
                           "evl_load (settle + disarm), certified per-state "
                           "against g1c's committed W1 light-dial record "
                           "(tol 5e-3), e242's committed body-flat md5s "
                           "(EXACT), e242's committed per-probe fact margins "
                           "(tol 1e-7), and e251's committed ctrl/held "
                           "medians (tol 1e-7)",
            "unwalled_loads": "numpy from e238's committed logit dumps "
                              "(float64, std ddof=1 = torch-unbiased); join "
                              "certified vs e232's committed margins "
                              f"(max |d| {G_UNWALLED_JOIN['max_abs_diff']:.1e})",
        },
        "eval": {"device": "cpu fp32", "threads": torch.get_num_threads(),
                 "batch_shape": "1 x 118 per probe (e228's margin_pass shape)",
                 "n_forwards": f"9 states x (60 fact + 60 ctrl + 30 held) "
                               f"margin probes + 9 light-dial batch reads = "
                               f"{9 * 150 + 9}"},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(rd / "metrics.json"),
                          str(rd / "e255_regression.png"),
                          str(rd / "e255_decomposition.png"),
                          str(rd / "e255_crossworld.png")]
    write_partial("P7 DONE (honesty + provenance + figures)")
    log(f"outputs: {metrics['outputs']}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
