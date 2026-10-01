"""E_CHART — THE CHART CELL (e189+e190 merged; the census and the subspace in
one run, one figure). Design: scratch/chart_cell_design.md (the frozen merged
design, INCLUDING the shuffled-sign rider); supersedes scratch/e189_design.md
and scratch/e190_design.md (both read in full before this build).

WHAT IT BUILDS ON (extend, don't repeat — every artifact LOADED, never
recomputed where it exists): W024 (the sign-flip mechanism candidate),
W025 (the effective-subspace picture), opt1's committed per-step gradient
data (pre-clip norms, batch provenance, cos anchors), e188 (the committed
trajectory-row battery reads used here as state gates), e191 (the committed
g-ray direction checkpoint + endpoint read), e192 (the committed five-ray
terrain: R1 g-ray profile = the C_WASH reference; R3-R5 Gaussian rays = the
D_FULL-RANDOM reference, extended by RIDER 2), g3K (the committed wash/iso
ladders + kappas on the store and host organisms — both rulers), e192's
machinery VERBATIM (protocol rebuild, gate set, battery, dual currency).

WHAT IS NEW: nobody has (a) DECOMPOSED the wash gradient by magnitude class
and asked which part carries the pump and which the erosion (the census), or
(b) tested WHERE the killing displacement lives — in an empirical subspace
or anywhere in parameter space (the subspace test) — and the paper needs
them on ONE chart (census x subspace x profile = the terrain's full chart;
day7 skeleton).

THE CELL (eval-only, CPU; one dispatch; progressive PARTIAL writes):

PART A — THE CENSUS (e189's spec): at the e180 snapshots + opt1's step
states (the neutral-stream arms: lr1e-3 = e176n arm A s1..s300, lr3e-5 and
lr1e-5 = e180's arms s2..s300, plus opt1 A0's s10; 21 states + the root):
the wash gradient g (bit-identical batch provenance, opt1's convention —
the seed-10902 neutral stream, the batch the protocol itself draws at step
t+1, gated against opt1's committed pre-clip norms) decomposed by magnitude
class (top-k |g| for k in {0.1%, 1%, 10%} + the continuous percentile
curve); per part: cos(part, grad fact) and the ||part|| share; the ADAM-VIEW
flip census (which magnitude classes change sign under normalization).

PART B — THE SUBSPACE (e190's spec): the SVD basis of the wash-step history
(r in {64, 256, 1024}, capped at the available history rank — the finite-
span proxy, documented); the in-span random arm vs the out-span-projected
wash arm at the rung ladder {1,2,4,8,16,32,64} (matched per-coordinate RMS,
g3K's convention: perturbation L2 = rung x ||wash 1x||), both rulers where
available (the e131 g-12 battery + CE_R as the primary ruler; the g3K store
g0 and host battery rulers on their own organisms — the "both rulers"
clause); the participation ratio vs kappa^-2 * d.

RIDER 1 — THE SHUFFLED-SIGN RAY (report-only; e192's open question): sign
(g_0) with its coordinate assignment shuffled within magnitude deciles —
same sign census per decile, same decile membership, wrong coordinate-sign
pairing. D grid {0.05..2.0} (+2.5, one documented extra point at the sign
ray's kill), both rulers, dual currency — does the pump survive the shuffle?

RIDER 2 — THE WIDER RANDOM GRID (report-only; e192's unresolved-high):
extend the three Gaussian rays' grid to D {6, 8, 10, 12} (LOAD their
existing rungs verbatim; read only the new rungs; +D 6 documented as a
bracket-tightening addition) to turn ">4.0" into a number.

REGISTERED BARS (frozen from the merged design + the dispatch brief,
VERBATIM; no bar shopping — adjudicate against exactly this):
  A1 STITCHES-AND-CUTS: "at some percentile cut the top part's alignment is
     positive while the complement's is negative, both |cos| >= 0.02"
  A2 FLAT-POSITIVE: "all magnitude classes align positively — the flip is
     denominator structure"
  A3 NO-STRUCTURE: "alignments below the declared floor"
  B1 SUBSPACE-CARRIES: "in-span random kills at rung <= 2 while out-span
     wash fails to kill through 16"
  B2 PARTIAL-PROJECTION: "graded — each arm's kill rung as the empirical
     projection profile; both dimension estimates compared"
  B3 SUBSPACE-REFUTED: "in-span random spares like full random"
No cross-part bar (the parts adjudicate independently; the chart is the
synthesis, not a gate). Riders report-only.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * alignment = cos(part of the wash MOVEMENT direction, grad of mean
    log p(Z) on the g-12 battery at the same state), fp64 cosine; positive
    = pump (fact-ascent), negative = erosion. The movement direction is
    the post-clip gradient negated. t=0 anchors gate the convention:
    cos(-g_0, grad m12) must reproduce opt1's committed SGD-arm step-1
    alignment +0.0981 (SGD wd=0, so the delta IS -lr*g), and the Adam-view
    cos(-sign(g_0), grad m12) must land near A0's committed step-1
    -0.0385 (the residual is Adam's weight-decay admixture, ~0.8% of L2).
  * ADAM-VIEW = the flattening limit: the same decomposition applied to
    -sign(g). EXACT at t=0 (Adam's bias-corrected first step is +-lr per
    coordinate); at t>0 it is the fresh-Adam flattening of that state's
    gradient (the accumulated second moment is not stored on disk — the
    approximation is declared, and the flip census reports it as such).
  * census states = the NEUTRAL-STREAM arms only. The lr1e-4 arm (the
    original extinction-anchor stream) is EXCLUDED: its batch provenance
    is a different stream, and the census's own clause is bit-identical
    provenance (e188 flagged that arm's stream mixing everywhere it
    entered a number; carried).
  * A-bar adjudication state = t=0 (the root — the state where W024's
    flip anchors live and where the pump is measured); the full 21-state
    census is reported (the trajectory of the decomposition), with a
    secondary existence scan (how many states show the A1 pattern)
    co-reported, non-adjudicating.
  * A1 evaluated over the three registered cuts {0.1%, 1%, 10%}: fires if
    ANY cut has cos(top) >= +0.02 AND cos(complement) <= -0.02. A2 fires
    if all four disjoint magnitude classes (top 0.1% / 0.1-1% / 1-10% /
    bottom 90%) have positive raw alignment while the whole-gradient flip
    is present. A3 fires if every class and cut alignment has |cos| <
    0.02 (the declared floor). Any other pattern: no A-bar fires; reported
    verbatim as the unnamed fourth outcome.
  * kill convention, e131 leg: first rung with g-12 <= 0.27 (e185's DISSOLVE
    bar; the e191/e192 convention on this organism/ruler); threshold
    interpolated in log2 between adjacent rungs (g3K's machinery); rung-1
    kill = threshold <= 1x bounded above; no kill through 64x =
    off-grid-high.
  * g3 legs (store + host): kill bar 0.5 verbatim (g3K's committed
    convention), own wash-1x rung scaling, references loaded from
    runs/g3K/metrics.json (never rerun).
  * B1 fires iff max over the 3 in-span seeds' kill thresholds <= 2 AND
    the out-span threshold > 16 (or off-grid-high). B3 fires iff min over
    the in-span thresholds >= full-random min threshold / 2 (within half
    an octave = "spares like the full random arm"). Anything else = B2.
    Adjudicated per organism (e131 primary; store and host co-adjudicated
    with their own committed references); disagreement across organisms is
    reported verbatim (organ n=1 per ruler).
  * the SVD basis per organism = the realized wash-step history (e131: the
    e180 snapshot diffs + opt1's stored per-step deltas, the merged
    design's letter; store/host: their own committed wash-trajectory
    snapshot diffs). r in {64, 256, 1024} all cap at the available rank
    (the r-sweep collapses; the full spectrum + participation ratio carry
    the concentration instead). The orthogonalization's numerical rank is
    documented; the removed fraction of the wash direction is reported.
  * dimension estimates: kappa-derived d_eff = d/kappa^2 (e131's kappa
    from the measured full-random threshold over the bounded wash
    threshold, updated by RIDER 2; store/host kappas from g3K's committed
    ladders) vs SVD-derived (available rank + participation ratio of both
    the step history and the census's gradient history).

HONESTY BLOCK (pre-registered): organ n=1 per ruler (the e131 consolidated
root for PART A + the e131 leg of PART B + both riders; one constructed
store organism and one host-fact twin for the g3 legs); snapshot quadrature
(the census is at sampled steps, not a trajectory — the wash gradient is
the protocol's own next-batch gradient at a snapshotted state); the
census's alignment floor 0.02 is a DECLARED floor (below it, "no
structure" is honest); the finite-span proxy caveat (r << true d_eff
possible — sparing on the in-span arm is AMBIGUOUS, reported as such: the
in-span arm tests a conservative LOWER bound of the subspace); B_OUTSPAN at
matched raw D has effective in-span displacement 0 by construction — if it
kills anyway, something outside the span carries death (refutes cleanly);
cross-organism numbers are pointers, never claims (each leg's kill bar is
its own ruler's); the 1e-4 arm's exclusion and the A0-state identity
(opt1 A0's s1/s2/s4 = the e176n arm A protocol's own states; only A0's s10
adds a state) are carried on every table they touch.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; no GPU claim), EVAL-ONLY (no training, no optimizer in any
arm; every grid point an independent perturb-and-eval off a committed
root), sequential, one dispatch, torch threads 4 for the e131 lineage (the
e185/opt1*/e191/e192 reduction order) and 8 for the g3 legs (g3K's own
reduction order), bit-deterministic.

Outputs: runs/e_chart/{metrics.json, the_chart.png}. PROGRESSIVE PARTIAL
metrics.json writes (after the gates, after every census state, after every
arm/rung family — the outage lesson; four machine disruptions this era).
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e_chart.py    (E_CHART_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n..e192)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import matplotlib.gridspec as gridspec                 # noqa: E402

SMOKE = os.environ.get("E_CHART_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e_chart is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1..e192's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E191_DIR_CK = CKPT_DIR / "e191_static_dir_u.pt"
E191_METRICS = E43.REPO / "runs" / "e191" / "metrics.json"
E192_METRICS = E43.REPO / "runs" / "e192" / "metrics.json"
E188_METRICS = E43.REPO / "runs" / "e188" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
G3K_METRICS = E43.REPO / "runs" / "g3K" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the frozen cell constants ------------------------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161..e192)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only)
STEP_SCALE_COMMITTED = 1.6542880535125732   # opt1 A0's committed step-1 L2 = the e131 wash-1x
CE1_COMMITTED = 1.356567621231079
GN1_COMMITTED = 0.9829167127609253
A0_S1_GM12_COMMITTED = 0.6780440807342529
E191_DIR_MD5 = "abb6df7ceefdd048b300c938123d964d"
E191_ENDPOINT_GM12 = 0.0007279703859239817  # e191's committed step-1-endpoint read
SGD_S1_COS_M12_COMMITTED = 0.09807665646076202  # opt1 a1_sgd_1e-3 step-1 alignment (wd=0: delta = -lr*g)
A0_S1_COS_M12_COMMITTED = -0.03851291537284851  # opt1 A0 step-1 alignment (Adam's normalized step)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — e131 leg
KILL_BAR_G3 = 0.50                # g3K's kill bar — store/host legs
RUNGS = (1, 2, 4, 8, 16, 32, 64)
ALIGN_FLOOR = 0.02                # the census's DECLARED floor (e189's letter)
CUT_FRACS = (0.001, 0.01, 0.10)   # the three registered cuts
PCTL_GRID = (50.0, 70.0, 80.0, 90.0, 95.0, 97.0, 99.0, 99.5, 99.9, 99.99)
RIDER1_GRID = (0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.92, 1.0, 1.17,
               1.5, 2.0, 2.5)     # e192's grid to 2.0 + the sign-kill point 2.5 (documented)
RIDER2_GRID = (6.0, 8.0, 10.0, 12.0)   # the letter's 8-12 + D 6 (documented bracket-tightener)
RAND_SEEDS = (11501, 11502, 11503)     # e192's committed Gaussian-ray seeds (regenerated, gated)
INSPAN_SEEDS_E131 = (11601, 11602, 11603)
INSPAN_SEEDS_STORE = (11611, 11612, 11613)
INSPAN_SEEDS_HOST = (11621, 11622, 11623)
RIDER1_SHUFFLE_SEED = 11701
OFFGRID_THR = 64.0 * 1.42         # the just-above-64 proxy for off-grid-high thresholds
SMALL_D_PUMP_BAND = (0.05, 0.1, 0.2, 0.33, 0.5)
PUMP_RIDGE_MARGIN = 0.01
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
DIR_COS_TOL = 1e-6
G_TOL_G3 = 2e-6                   # g3K's gates were 8-thread; 4/8-thread texture allowed

# ---- the census state table (neutral-stream arms only; see operationalizations)
CENSUS_STATES = [
    # (arm, wash_step, ckpt, e188_repro_arm)
    ("root", 0, ROOT_CK, None),
    ("lr1e-3", 1, "e176n_neutral_s1.pt", "lr1e-3_neutral"),
    ("lr1e-3", 2, "e176n_neutral_s2.pt", "lr1e-3_neutral"),
    ("lr1e-3", 4, "e176n_neutral_s4.pt", "lr1e-3_neutral"),
    ("opt1_a0", 10, "opt1_a0_adamw_ref_s10.pt", None),   # opt1's own committed traj row
    ("lr1e-3", 50, "e176n_neutral_s50.pt", "lr1e-3_neutral"),
    ("lr1e-3", 100, "e176n_neutral_s100.pt", "lr1e-3_neutral"),
    ("lr1e-3", 200, "e176n_neutral_s200.pt", "lr1e-3_neutral"),
    ("lr1e-3", 300, "e176n_neutral.pt", "lr1e-3_neutral"),
    ("lr3e-5", 2, "e180_neutral_lr3e5_s2.pt", "lr3e-5_neutral"),
    ("lr3e-5", 10, "e180_neutral_lr3e5_s10.pt", "lr3e-5_neutral"),
    ("lr3e-5", 50, "e180_neutral_lr3e5_s50.pt", "lr3e-5_neutral"),
    ("lr3e-5", 100, "e180_neutral_lr3e5_s100.pt", "lr3e-5_neutral"),
    ("lr3e-5", 200, "e180_neutral_lr3e5_s200.pt", "lr3e-5_neutral"),
    ("lr3e-5", 300, "e180_neutral_lr3e5.pt", "lr3e-5_neutral"),
    ("lr1e-5", 2, "e180_neutral_lr1e5_s2.pt", "lr1e-5_neutral"),
    ("lr1e-5", 10, "e180_neutral_lr1e5_s10.pt", "lr1e-5_neutral"),
    ("lr1e-5", 50, "e180_neutral_lr1e5_s50.pt", "lr1e-5_neutral"),
    ("lr1e-5", 100, "e180_neutral_lr1e5_s100.pt", "lr1e-5_neutral"),
    ("lr1e-5", 200, "e180_neutral_lr1e5_s200.pt", "lr1e-5_neutral"),
    ("lr1e-5", 300, "e180_neutral_lr1e5.pt", "lr1e-5_neutral"),
]
if SMOKE:
    CENSUS_STATES = CENSUS_STATES[:6]
    RUNGS = (1, 4, 16)
    RIDER2_GRID = (8.0, 12.0)
    INSPAN_SEEDS_E131 = (11601,)
    INSPAN_SEEDS_STORE = (11611,)
    INSPAN_SEEDS_HOST = (11621,)

# ---- gates / references (opt1c/e191/e192's set, verbatim where reused)
R_EVAL_SEED = 26502
E151_ROOT = {
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
}
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {1: "1ea27bffde6c4a53be8badf5ab453d64"}

REGISTERED_BARS = {
    "A1_STITCHES_AND_CUTS": "A1 STITCHES-AND-CUTS: \"at some percentile cut "
        "the top part's alignment is positive while the complement's is "
        "negative, both |cos| >= 0.02\"",
    "A2_FLAT_POSITIVE": "A2 FLAT-POSITIVE: \"all magnitude classes align "
        "positively — the flip is denominator structure\"",
    "A3_NO_STRUCTURE": "A3 NO-STRUCTURE: \"alignments below the declared "
        "floor\"",
    "B1_SUBSPACE_CARRIES": "B1 SUBSPACE-CARRIES: \"in-span random kills at "
        "rung <= 2 while out-span wash fails to kill through 16\"",
    "B2_PARTIAL_PROJECTION": "B2 PARTIAL-PROJECTION: \"graded — each arm's "
        "kill rung as the empirical projection profile; both dimension "
        "estimates compared\"",
    "B3_SUBSPACE_REFUTED": "B3 SUBSPACE-REFUTED: \"in-span random spares "
        "like full random\"",
    "no_cross_part_bar": "No cross-part bar (the parts adjudicate "
        "independently; the chart is the synthesis, not a gate).",
    "riders_report_only": "RIDER 1 (the shuffled-sign ray) and RIDER 2 (the "
        "wider random grid) are report-only; they cannot fire or veto a bar.",
    "registration": "the dispatch brief's six bars quoted VERBATIM here and "
        "in the module docstring, frozen before compute. Adjudicate against "
        "exactly this; no bar shopping.",
}

deviations: list[str] = [
    "PART A states: the lr1e-4 arm (the original extinction-anchor stream) "
    "is EXCLUDED from the census — its batch provenance is a different "
    "stream and the census's own clause is bit-identical provenance (e188 "
    "flagged that arm's stream mixing; carried). The A0 step states s1/s2/s4 "
    "are the e176n arm A protocol's own states (same seed-10902 stream); "
    "opt1's A0 s10 adds the one state e176n never snapshotted.",
    "The SVD history follows the merged design's letter per organism: e131 "
    "= the e180 snapshot diffs + opt1's stored per-step deltas; store/host "
    "= their own committed wash-trajectory snapshot diffs. r in {64, 256, "
    "1024} all cap at the available history rank (the r-sweep collapses by "
    "construction; the full spectrum + participation ratio carry the "
    "concentration instead — the finite-span proxy caveat is carried).",
    "ADAM-VIEW is the flattening limit (-sign(g)): EXACT at t=0 (Adam's "
    "bias-corrected first step is +-lr per coordinate); at t>0 the "
    "accumulated second moment is not on disk, so the state-wise Adam-view "
    "is the fresh-Adam flattening — declared on every flip-census row.",
    "RIDER 1 adds D 2.5 to the letter's {0.05..2.0} (one read at the sign "
    "ray's committed kill, so the report can place the shuffled kill "
    "relative to it); RIDER 2 adds D 6 to the letter's 8-12 (bracket "
    "tightening). Both documented additions to report-only riders.",
    "The B bars are adjudicated per organism with frozen clauses (B1: max "
    "in-span threshold <= 2 AND out-span > 16; B3: min in-span >= "
    "full-random/2; else B2); e131 is the primary adjudication and the "
    "store/host legs co-adjudicate with their own committed references — "
    "any disagreement is reported verbatim (organ n=1 per ruler).",
    "The e131 wash leg re-reads the committed g-ray fresh at the 7 rung Ds "
    "(its rung-1 point gated bit-close vs e191's committed endpoint read); "
    "e192's committed R1 profile remains the loaded reference of record.",
    "Smoke mode trims: 6 census states, rungs {1,4,16}, one in-span seed "
    "per organism, rider-2 grid {8,12}; nothing adjudicated (SMOKE stamp).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e192_all_ray_terrain.py VERBATIM (whose own provenance is
# lab/e191_pump_cliff.py via lab/opt1c_direction_size.py — the e176n lineage).

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
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


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def flat_md5(net) -> str:
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64)))


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """opt1/e188's alignment read VERBATIM: gradient of the fact battery's
    mean log p(Z) at the last position, at the eval twin's current weights."""
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


def wash_grad(net_tr: TinyGPT, x, y) -> tuple[torch.Tensor, float, float]:
    """The wash gradient on a train()-mode twin (the protocol's own mode):
    post-clip flat gradient + pre/post norms. Deterministic (no dropout on
    this lineage; e192's G_T0 bit-reproduces in train mode)."""
    net_tr.train()
    net_tr.zero_grad(set_to_none=True)
    logits, _ = net_tr(x)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1)).backward()
    gn_pre = float(torch.nn.utils.clip_grad_norm_(net_tr.parameters(), 1.0))
    g = torch.cat([p.grad.detach().reshape(-1) for p in net_tr.parameters()])
    net_tr.zero_grad(set_to_none=True)
    return g.clone(), gn_pre, float(torch.norm(g))


def kill_analysis(rows: list[dict], read_key: str, bar: float) -> dict:
    """g3K's kill analysis with the bar parameterized (0.27 e131 / 0.5 g3)."""
    ladder = [r for r in rows if r["rung"] > 0]
    ki = next((i for i, r in enumerate(ladder)
               if ladder[i][read_key] < bar), None)
    out = {"curve": [{"rung": r["rung"], read_key: r[read_key],
                      "ce_r": r.get("ce_r")} for r in rows], "bar": bar}
    if ki is None:
        out.update(kill_rung=None, threshold=None, off_grid_high=True,
                   bound=f"no rung through {RUNGS[-1]}x below bar")
        return out
    kr, rung_k = ladder[ki][read_key], ladder[ki]["rung"]
    out.update(kill_rung=rung_k, off_grid_high=False)
    if ki == 0:
        out.update(threshold=1.0, bounded_above=True,
                   note="first rung (1x) already below bar — true threshold "
                        "<= 1x (anchored at root; not interpolated)")
        return out
    prev = ladder[ki - 1]
    lr_prev, lr_kill = np.log2(prev["rung"]), np.log2(rung_k)
    vp, vk = prev[read_key], kr
    t = (vp - bar) / max(vp - vk, 1e-12)
    out.update(threshold=float(2.0 ** (lr_prev + t * (lr_kill - lr_prev))),
               interp_from=prev["rung"], interp_to=rung_k)
    return out


def first_dead_thr_D(rows: list[dict], bar: float) -> float | None:
    """Log2-interpolated kill threshold in D units from a D-sorted profile;
    None = no kill on the grid (unresolved-high)."""
    for i in range(1, len(rows)):
        if rows[i]["gm12"] <= bar:
            prev, dead = rows[i - 1], rows[i]
            t = (prev["gm12"] - bar) / max(prev["gm12"] - dead["gm12"], 1e-12)
            return float(2.0 ** (np.log2(prev["D"])
                                 + t * (np.log2(dead["D"]) - np.log2(prev["D"]))))
    return None


def participation_ratio(vals: torch.Tensor) -> float:
    v = vals.to(torch.float64)
    return float(torch.sum(v) ** 2 / torch.sum(v * v))


def svd_basis(H: torch.Tensor) -> dict:
    """Right-singular basis of H (m x N) via the Gram matrix (fp64), with the
    spectrum + participation ratio. V columns are orthonormal directions in
    parameter space."""
    H64 = H.to(torch.float64)
    G = (H64 @ H64.T)
    evals, evecs = torch.linalg.eigh(G)     # ascending
    evals = torch.flip(evals, dims=(0,)).clamp(min=0)
    evecs = torch.flip(evecs, dims=(1,))
    sv = torch.sqrt(evals)
    pr = participation_ratio(sv) if float(sv[0]) > 0 else 0.0
    rank_eff = int((sv > sv[0] * 1e-7).sum())
    return {"sv": sv, "V": evecs, "pr": pr, "rank_eff": rank_eff,
            "cond": float(sv[0] / sv[rank_eff - 1].clamp(min=1e-30))
            if rank_eff else None}


# ------------------------------------------------------------------ main

def main():
    torch.set_num_threads(4)                  # e185/opt1*/e191/e192 reduction order
    rd = run_dir("e_chart_smoke" if SMOKE else "e_chart")
    log(f"E_CHART THE CHART CELL (smoke={SMOKE}) -> {rd}")
    log("PART A the census + PART B the subspace + riders; CPU-ONLY, "
        "EVAL-ONLY, one dispatch; bars frozen in the docstring before compute")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (E191_METRICS, E192_METRICS, E188_METRICS, OPT1_METRICS, G3K_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    if not E191_DIR_CK.exists():
        raise RuntimeError(f"missing e191 direction checkpoint: {E191_DIR_CK}")
    e191m = json.loads(E191_METRICS.read_text(encoding="utf-8"))
    e192m = json.loads(E192_METRICS.read_text(encoding="utf-8"))
    e188m = json.loads(E188_METRICS.read_text(encoding="utf-8"))
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    g3km = json.loads(G3K_METRICS.read_text(encoding="utf-8"))
    prov_files: dict[str, dict] = {}

    def hash_prov(p: Path, note: str = "") -> dict:
        key = str(p)
        if key not in prov_files:
            prov_files[key] = {"path": str(p.relative_to(E43.REPO)),
                               "sha1": sha1_file(p), "bytes": p.stat().st_size,
                               "note": note or "loaded, hashed at load"}
        return prov_files[key]

    a0 = opt1m["arms"]["a0_adamw_ref"]
    a0_s1 = next(r for r in a0["traj"] if r["step"] == 1)
    assert abs(a0_s1["cum_disp"] - STEP_SCALE_COMMITTED) < 1e-12
    assert abs(a0_s1["gm12"] - A0_S1_GM12_COMMITTED) < 1e-12
    a1_s1 = next(r for r in opt1m["arms"]["a1_sgd_1e-3"]["traj"] if r["step"] == 1)
    assert abs(a1_s1["cos_delta_fact_m12"] - SGD_S1_COS_M12_COMMITTED) < 1e-12
    assert abs(a0_s1["cos_delta_fact_m12"] - A0_S1_COS_M12_COMMITTED) < 1e-12
    # e192's committed ray profiles (the C_WASH + D_FULL-RANDOM references)
    e192_prof = e192m["profiles"]
    e192_r1 = e192_prof["R1_G"]["rows"]
    e192_rand = {k: e192_prof[k]["rows"] for k in
                 ("R3_GAUSS_s11501", "R4_GAUSS_s11502", "R5_GAUSS_s11503")}
    assert e192m["cell"]["rays"][0]["u_md5"] == E191_DIR_MD5, \
        "e192's committed R1 direction drifted vs e191's md5"
    e192_rand_seeds = [r["provenance"]["seed"] if isinstance(r.get("provenance"), dict)
                       else None for r in e192m["cell"]["rays"][2:]]
    assert e192_rand_seeds == list(RAND_SEEDS), f"e192 rand seeds drifted: {e192_rand_seeds}"
    # e188's committed state reads (the census state gates)
    e188_repro = {arm: {int(r["step"]): r for r in
                        e188m["trajectory_row"]["arms"][arm]["repro"]}
                  for arm in ("lr1e-3_neutral", "lr3e-5_neutral", "lr1e-5_neutral")}
    a0_traj = {int(r["step"]): r for r in a0["traj"]}
    # g3K's committed ladders (the store/host references)
    g3k_store = g3km["organisms"]["store"]
    g3k_host = g3km["organisms"]["host"]
    log("parents: e191, e192 (5-ray terrain), e188 (state reads), opt1 "
        "(A0/a1 anchors), g3K (store/host ladders) — all LOADED, none rerun")

    # ---------------- protocol rebuild (e143..e192 VERBATIM) --------------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert train_text.count("ZEPH") == 0, "corpus contains ZEPH"

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
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids = bat_ids[-12]
    assert list(bat_ids[-12].shape) == [60, 118] and \
        list(bat_ids[0].shape) == [60, 130] and \
        list(bat_ids[12].shape) == [60, 142], "battery geometry drift"
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    arng = _random.Random(170)
    n_starts = []
    hi_start = len(train_ids) - BLOCK - 2
    tries = 0
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")):
            continue
        n_starts.append(s)
        tries += 1
    assert n_starts == E170_BANK_STARTS, "neutral bank drifted"
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    n_anc = anchor_neutral.shape[0]
    log("protocol rebuilt (e192 verbatim): install60 FL19/EL41; bank 16; "
        f"battery shapes ok; val60 seed {R_EVAL_SEED}")

    # ---------------- root + gate vs e151 (e192's gate set verbatim) ------------
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    hash_prov(CKPT_DIR / ROOT_CK, "the e131 consolidated root (the cell's primary organism)")
    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    RMS_DEN = float(torch.sqrt(torch.tensor(float(N_PARAM), dtype=torch.float64)))
    evl0 = copy.deepcopy(net0)
    root_cells = {
        "gm12": battery_cell(evl0, gm12_ids, zid)["mean_pz"],
        "g0": battery_cell(evl0, bat_ids[0], zid)["mean_pz"],
        "gp12": battery_cell(evl0, bat_ids[12], zid)["mean_pz"],
        "ce_r": ce_fixed_cpu(evl0, *r_eval_xy),
    }
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items() if k in keymap}
    rmax = max(abs(root_cells[k] - root_refs[k]) for k in root_refs)
    G_ROOT = {"cells": root_cells, "refs": root_refs,
              "max_abs_diff": rmax, "tol": G_FALLBACK_TOL,
              "pass": bool(rmax < G_FALLBACK_TOL)}
    log(f"G_ROOT (vs e151 before-cells): max|diff| {rmax:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED")

    # ---------------- the neutral stream (opt1's draw order, to step 301) -------
    need_batches = sorted({1} | {s + 1 for _, s, _, _ in CENSUS_STATES})
    max_step = max(need_batches)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    batches: dict[int, tuple] = {}
    for k in range(1, max_step + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        if k in need_batches:
            batches[k] = (x, y)
    x1, y1 = batches[1]
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    G_T0 = {"step1_x_md5": x1_md5,
            "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1])}

    # ---- the fresh t=0 wash gradient + AdamW step (G_T0 continued, e192 verbatim)
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
    g0_t0 = torch.cat([p.grad.detach().reshape(-1)
                       for p in tw.parameters()]).clone()   # post-clip
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0))
    del tw, optw, logits
    G_T0.update(ce_batch_measured=ce1, ce_batch_committed=CE1_COMMITTED,
                preclip_gnorm_measured=gn1, preclip_gnorm_committed=GN1_COMMITTED,
                adamw_step1_L2_measured=disp1,
                adamw_step1_L2_committed=STEP_SCALE_COMMITTED,
                pass=bool(G_T0["step1_x_md5_match_e185"]
                          and abs(ce1 - CE1_COMMITTED) < G_FALLBACK_TOL
                          and abs(gn1 - GN1_COMMITTED) < G_FALLBACK_TOL
                          and abs(disp1 - STEP_SCALE_COMMITTED) < G_FALLBACK_TOL))
    log(f"G_T0 (t=0 bit-identity): CE |d| {abs(ce1 - CE1_COMMITTED):.2e}, "
        f"gn |d| {abs(gn1 - GN1_COMMITTED):.2e}, disp |d| "
        f"{abs(disp1 - STEP_SCALE_COMMITTED):.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — abort (control failure)")

    # ---- G_W024: the flip anchors (the census's sign-convention gate)
    evl_root = copy.deepcopy(net0)
    evl_root.eval()
    grad_m12_root = fact_grad(evl_root, gm12_ids, zid)
    grad_g0_root = fact_grad(evl_root, bat_ids[0], zid)
    w024_raw = cos64(-g0_t0, grad_m12_root)
    w024_adam = cos64(-torch.sign(g0_t0), grad_m12_root)
    G_W024 = {
        "cos_mov_m12_at_t0": w024_raw,
        "committed_sgd_s1": SGD_S1_COS_M12_COMMITTED,
        "d_raw": abs(w024_raw - SGD_S1_COS_M12_COMMITTED),
        "cos_adamview_m12_at_t0": w024_adam,
        "committed_a0_s1": A0_S1_COS_M12_COMMITTED,
        "d_adam": abs(w024_adam - A0_S1_COS_M12_COMMITTED),
        "tol_raw": 2e-3, "tol_adam": 1e-2,
        "pass": bool(abs(w024_raw - SGD_S1_COS_M12_COMMITTED) < 2e-3
                     and abs(w024_adam - A0_S1_COS_M12_COMMITTED) < 1e-2),
        "note": "the census's sign convention gated at its own anchors: the "
                "raw movement alignment must reproduce opt1's SGD-arm step-1 "
                "+0.0981 (SGD wd=0 -> delta = -lr*g), the flattened "
                "(Adam-view) alignment must land near A0's step-1 -0.0385 "
                "(residual = Adam's weight-decay admixture ~0.8% of L2)",
    }
    log(f"G_W024 (the flip anchors): raw {w024_raw:+.5f} vs "
        f"{SGD_S1_COS_M12_COMMITTED:+.5f} (|d| {G_W024['d_raw']:.1e}); "
        f"adam-view {w024_adam:+.5f} vs {A0_S1_COS_M12_COMMITTED:+.5f} "
        f"(|d| {G_W024['d_adam']:.1e}): "
        + ("PASS" if G_W024["pass"] else "FAIL"))
    if not G_W024["pass"]:
        raise RuntimeError("flip-anchor gate FAILED — the census convention "
                           "would not be W024's; abort (control failure)")

    # ---- G_U_CK: the e191 direction bit-gate (e192 verbatim)
    uck = torch.load(E191_DIR_CK, map_location="cpu", weights_only=False)
    u_g = uck["u"].detach().clone().float()
    u_g_md5 = hashlib.md5(u_g.numpy().tobytes()).hexdigest()
    theta0_md5 = flat_md5(net0)
    u_fresh = (g0_t0 / torch.norm(g0_t0)).clone()
    fresh_md5 = hashlib.md5(u_fresh.numpy().tobytes()).hexdigest()
    c64 = cos64(u_fresh, u_g)
    G_U_CK = {
        "loaded_u_md5": u_g_md5, "committed_e191_md5": E191_DIR_MD5,
        "md5_match_committed": bool(u_g_md5 == E191_DIR_MD5),
        "root_flat_md5": theta0_md5,
        "checkpoint_theta0_md5": uck.get("theta0_md5"),
        "root_md5_match_checkpoint": bool(theta0_md5 == uck.get("theta0_md5")),
        "fresh_recompute_md5": fresh_md5,
        "fresh_md5_match_loaded": bool(fresh_md5 == u_g_md5),
        "fresh_cos64_loaded": c64,
        "pass": bool(u_g_md5 == E191_DIR_MD5 and u_g.numel() == N_PARAM
                     and abs(float(torch.norm(u_g)) - 1.0) < 1e-5
                     and theta0_md5 == uck.get("theta0_md5")
                     and (fresh_md5 == u_g_md5 or c64 > 1 - DIR_COS_TOL)),
    }
    log(f"G_U_CK (e191 direction bit-gate): "
        + ("PASS" if G_U_CK["pass"] else "FAIL"))
    if not G_U_CK["pass"]:
        raise RuntimeError("e191 direction bit-gate FAILED — abort")
    hash_prov(E191_DIR_CK, "the committed g-ray direction (PART B's wash ray)")

    # ---------------- progressive PARTIAL write helper --------------------------
    state = {"census": [], "svd": None, "partB_e131": {}, "riders": {},
             "partB_g3": {}}
    def write_partial(status: str, extra: dict | None = None) -> None:
        payload = {
            "experiment": "e_chart", "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"census_done": len(state["census"]),
                      "census_total": len(CENSUS_STATES),
                      "elapsed_s": round(time.time() - T0, 1)},
            "gates_partial": {"G_ROOT": G_ROOT, "G_T0": G_T0,
                              "G_W024": G_W024, "G_U_CK": G_U_CK},
            "census_partial": state["census"],
            "svd_partial": state["svd"],
            "partB_e131_partial": state["partB_e131"],
            "riders_partial": state["riders"],
            "partB_g3_partial": state["partB_g3"],
            "registered_bars": REGISTERED_BARS,
        }
        if extra:
            payload.update(extra)
        save_json(rd / "metrics.json", E43.jsonable(payload))
    write_partial("PARTIAL — all pre-dispatch gates PASSED; census starting")
    log("[partial] metrics.json written (gates passed)")

    # =====================================================================
    # PART A — THE CENSUS
    # =====================================================================
    log("=" * 78)
    log("PART A — THE GRADIENT CENSUS OF THE PUMP (e189's spec)")
    evl = copy.deepcopy(net0)
    evl.eval()
    gtr = copy.deepcopy(net0)     # the train-mode wash-gradient twin
    gnorm_gate_rows = []
    N = N_PARAM
    cut_ns = [max(1, int(round(f * N))) for f in CUT_FRACS]
    class_edges = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.10), (0.10, 1.0)]
    class_names = ("top0.1%", "0.1-1%", "1-10%", "bottom90%")
    pctl_ns = [max(1, int(round((p / 100.0) * N))) for p in PCTL_GRID]
    grad_history: list[tuple[str, torch.Tensor]] = []

    for (arm, step, ck, repro_arm) in CENSUS_STATES:
        if step == 0:
            theta_s = theta0
        else:
            theta_s = flat_params(load_cpu(CKPT_DIR / ck))
            hash_prov(CKPT_DIR / ck, f"census state {arm}@s{step}")
        # state gate vs committed reads
        load_flat(evl, theta_s)
        evl.eval()
        gm12_s = battery_cell(evl, gm12_ids, zid)["mean_pz"]
        g0_s = battery_cell(evl, bat_ids[0], zid)["mean_pz"]
        ref = None
        if repro_arm is not None:
            rr = e188_repro[repro_arm].get(step)
            ref = {"gm12": rr["gm12"], "g0": rr["g0"],
                   "src": "runs/e188/metrics.json repro row"} if rr else None
        elif step == 10:
            rr = a0_traj.get(10)
            ref = {"gm12": rr["gm12"], "g0": rr["g0"],
                   "src": "runs/opt1/metrics.json a0 traj step10"} if rr else None
        gate_ok = True
        if ref is not None:
            gate_ok = (abs(gm12_s - ref["gm12"]) < G_BIT_TOL
                       and abs(g0_s - ref["g0"]) < G_BIT_TOL)
        # the wash gradient (the protocol's own next batch) + fact grads
        xb, yb = batches[step + 1]
        load_flat(gtr, theta_s)
        g_w, gn_pre, gn_post = wash_grad(gtr, xb, yb)
        grad_history.append((f"{arm}@s{step}", g_w))
        g_mov = -g_w                                    # the movement direction
        adam_mov = -torch.sign(g_w)                     # the flattening limit
        load_flat(evl, theta_s)
        evl.eval()
        gm12_grad = fact_grad(evl, gm12_ids, zid)
        g0_grad = fact_grad(evl, bat_ids[0], zid)
        # stream gate where opt1 committed the pre-clip norm
        sgate = None
        if step in (0, 1, 2, 4) and (step + 1) in a0_traj:
            committed = a0_traj[step + 1]["preclip_gnorm"]
            sgate = {"step": step, "batch": step + 1,
                     "measured_preclip": gn_pre, "committed": committed,
                     "d": abs(gn_pre - committed),
                     "pass": bool(abs(gn_pre - committed) < G_BIT_TOL)}
            gnorm_gate_rows.append(sgate)
        # decomposition
        absort = torch.argsort(g_w.abs(), descending=True)
        ranks = torch.empty(N, dtype=torch.long)
        ranks[absort] = torch.arange(N, dtype=torch.long)
        gn2 = float(torch.norm(g_w))
        row = {
            "arm": arm, "step": step, "ckpt": ck,
            "gnorm_preclip": gn_pre, "gnorm_postclip": gn_post,
            "state_gate": {"gm12": gm12_s, "g0": g0_s, "ref": ref,
                           "pass": gate_ok},
            "stream_gate": sgate,
            "cos_mov_m12": cos64(g_mov, gm12_grad),
            "cos_mov_g0": cos64(g_mov, g0_grad),
            "cos_adamview_m12": cos64(adam_mov, gm12_grad),
            "sign_census": {"n_pos": int((g_w > 0).sum()),
                            "n_neg": int((g_w < 0).sum()),
                            "n_zero": int((g_w == 0).sum())},
            "cuts": [], "classes": [], "curve": [],
        }
        for f, k in zip(CUT_FRACS, cut_ns):
            top_mask = ranks < k
            part = g_mov * top_mask
            comp = g_mov * (~top_mask)
            row["cuts"].append({
                "k_frac": f, "n_top": int(k),
                "cos_top_m12": cos64(part, gm12_grad),
                "cos_comp_m12": cos64(comp, gm12_grad),
                "cos_top_g0": cos64(part, g0_grad),
                "cos_comp_g0": cos64(comp, g0_grad),
                "share_top": float(torch.norm(part) / gn2),
            })
        for (lo, hi), nm in zip(class_edges, class_names):
            m = (ranks >= int(round(lo * N))) & (ranks < max(1, int(round(hi * N))))
            raw_c = cos64(g_mov * m, gm12_grad)
            adam_c = cos64(adam_mov * m, gm12_grad)
            row["classes"].append({
                "name": nm, "n": int(m.sum()),
                "share": float(torch.norm(g_mov * m) / gn2),
                "cos_raw_m12": raw_c, "cos_adamview_m12": adam_c,
                "cos_raw_g0": cos64(g_mov * m, g0_grad),
                "flip": bool((raw_c > 0) != (adam_c > 0)
                             and min(abs(raw_c), abs(adam_c)) >= ALIGN_FLOOR),
            })
        for p, k in zip(PCTL_GRID, pctl_ns):
            m = ranks < k
            row["curve"].append({
                "p_pct": p, "n_top": int(k),
                "cos_top_m12": cos64(g_mov * m, gm12_grad),
                "cos_comp_m12": cos64(g_mov * (~m), gm12_grad),
                "share_top": float(torch.norm(g_mov * m) / gn2),
            })
        state["census"].append(row)
        log(f"  census {arm:>7s}@s{step:<3d} gnorm {gn_pre:8.4f}: "
            f"mov {row['cos_mov_m12']:+.4f} adam {row['cos_adamview_m12']:+.4f} "
            f"top0.1% {row['classes'][0]['cos_raw_m12']:+.4f} "
            f"bottom90% {row['classes'][3]['cos_raw_m12']:+.4f} "
            + ("" if gate_ok else "[STATE GATE FAIL]"))
        write_partial(f"PARTIAL: census {len(state['census'])}/"
                      f"{len(CENSUS_STATES)} states done")

    G_STREAM = {"rows": gnorm_gate_rows,
                "pass": bool(gnorm_gate_rows
                             and all(r["pass"] for r in gnorm_gate_rows)),
                "note": "the census's wash gradients are the protocol's own "
                        "next-batch gradients: pre-clip norms gated vs "
                        "opt1's committed per-step values at the states "
                        "where opt1 committed them (batches 1,2,3,5)"}
    G_STATES = {"n_states": len(state["census"]),
                "all_pass": bool(all(r["state_gate"]["pass"] for r in state["census"])),
                "note": "every census state's g-12/g0 battery read "
                        "reproduced its committed row (e188 repro / opt1 "
                        "traj) within 5e-6 before use"}
    log(f"G_STREAM: {G_STREAM['pass']}; G_STATES: {G_STATES['all_pass']}")
    if not (G_STREAM["pass"] and G_STATES["all_pass"]):
        write_partial("FAILED GATE — G_STREAM/G_STATES", {"failed": True})
        raise RuntimeError("census provenance gates FAILED — abort")
    write_partial("PARTIAL: PART A census complete (gates PASS)",
                  {"gates_stream": G_STREAM, "gates_states": G_STATES})

    # =====================================================================
    # THE SVD BASIS (PART B's instrument) — e131 step history + gradient history
    # =====================================================================
    log("=" * 78)
    log("PART B — THE SVD BASIS OF THE WASH-STEP HISTORY (e190's spec)")
    hist_rows = []
    seg_specs = [
        ("lr1e-3", ROOT_CK, "e176n_neutral_s1.pt"),
        ("lr1e-3", "e176n_neutral_s1.pt", "e176n_neutral_s2.pt"),
        ("lr1e-3", "e176n_neutral_s2.pt", "e176n_neutral_s4.pt"),
        ("lr1e-3", "e176n_neutral_s4.pt", "e176n_neutral_s50.pt"),
        ("lr1e-3", "e176n_neutral_s50.pt", "e176n_neutral_s100.pt"),
        ("lr1e-3", "e176n_neutral_s100.pt", "e176n_neutral_s200.pt"),
        ("lr1e-3", "e176n_neutral_s200.pt", "e176n_neutral.pt"),
        ("lr3e-5", ROOT_CK, "e180_neutral_lr3e5_s2.pt"),
        ("lr3e-5", "e180_neutral_lr3e5_s2.pt", "e180_neutral_lr3e5_s10.pt"),
        ("lr3e-5", "e180_neutral_lr3e5_s10.pt", "e180_neutral_lr3e5_s50.pt"),
        ("lr3e-5", "e180_neutral_lr3e5_s50.pt", "e180_neutral_lr3e5_s100.pt"),
        ("lr3e-5", "e180_neutral_lr3e5_s100.pt", "e180_neutral_lr3e5_s200.pt"),
        ("lr3e-5", "e180_neutral_lr3e5_s200.pt", "e180_neutral_lr3e5.pt"),
        ("lr1e-5", ROOT_CK, "e180_neutral_lr1e5_s2.pt"),
        ("lr1e-5", "e180_neutral_lr1e5_s2.pt", "e180_neutral_lr1e5_s10.pt"),
        ("lr1e-5", "e180_neutral_lr1e5_s10.pt", "e180_neutral_lr1e5_s50.pt"),
        ("lr1e-5", "e180_neutral_lr1e5_s50.pt", "e180_neutral_lr1e5_s100.pt"),
        ("lr1e-5", "e180_neutral_lr1e5_s100.pt", "e180_neutral_lr1e5_s200.pt"),
        ("lr1e-5", "e180_neutral_lr1e5_s200.pt", "e180_neutral_lr1e5.pt"),
        ("opt1_a0_s4_s10", "e176n_neutral_s4.pt", "opt1_a0_adamw_ref_s10.pt"),
    ]
    state_cache: dict[str, torch.Tensor] = {}

    def state_flat(ck: str) -> torch.Tensor:
        if ck not in state_cache:
            state_cache[ck] = flat_params(load_cpu(CKPT_DIR / ck))
            hash_prov(CKPT_DIR / ck, f"history endpoint {ck}")
        return state_cache[ck]

    for (tag, a, b) in seg_specs:
        pa, pb = CKPT_DIR / a, CKPT_DIR / b
        if not pa.exists() or not pb.exists():
            log(f"  history segment {tag}: MISSING checkpoint ({a}/{b}) — skipped")
            continue
        ta = theta0 if a == ROOT_CK else state_flat(a)
        tb = state_flat(b)
        hist_rows.append((tag, tb - ta))
    H = torch.stack([v for _, v in hist_rows])
    Hu = torch.stack([v / torch.norm(v) for _, v in hist_rows])
    basis = svd_basis(H)
    basis_u = svd_basis(Hu)
    r_avail = basis["rank_eff"]
    Vr = basis["V"].T[:r_avail].to(torch.float32).contiguous()   # (r, N)
    Vr64 = Vr.to(torch.float64)
    proj64 = Vr64 @ u_g.to(torch.float64)
    removed = float(torch.norm(proj64))
    u_out64 = u_g.to(torch.float64) - Vr64.T @ proj64
    u_out = (u_out64 / torch.norm(u_out64)).to(torch.float32).contiguous()
    outspan_residual = float(torch.norm(u_g.to(torch.float64) - Vr64.T @ proj64))
    HG = torch.stack([v for _, v in grad_history])
    basis_g = svd_basis(HG)
    del hist_rows, H, Hu, grad_history, state_cache
    state["svd"] = {
        "history_segments": None,   # filled below (tag list + L2s)
        "n_segments": int(Vr.shape[0]),
        "spectrum_step_history": [float(s) for s in basis["sv"]],
        "spectrum_unitrow": [float(s) for s in basis_u["sv"]],
        "participation_ratio_step_history": basis["pr"],
        "participation_ratio_unitrow": basis_u["pr"],
        "participation_ratio_gradient_history": basis_g["pr"],
        "rank_available": r_avail,
        "numerical_rank": {"n_vectors": len(seg_specs),
                           "rank_eff_1e7": basis["rank_eff"],
                           "cond": basis["cond"],
                           "note": "Gram-based fp64 eigendecomposition; rank "
                                   "threshold sigma > sigma_max*1e-7"},
        "r_grid_requested": [64, 256, 1024],
        "r_cap_note": f"r in {{64,256,1024}} all cap at the available "
                      f"history rank {r_avail} (the finite-span proxy; the "
                      f"span used = the FULL available span)",
        "wash_dir_removed_fraction": removed,
        "wash_dir_removed_note": f"||V_r^T u_g|| = {removed:.6f} — the wash "
                                 f"direction is ~fully in-span BY "
                                 f"CONSTRUCTION (the step-1 segment is in "
                                 f"the history); the out-span arm measures "
                                 f"the residual, disclosed per e190's letter",
        "outspan_residual_L2": outspan_residual,
    }
    log(f"  history: {state['svd']['n_segments']} segments; rank {r_avail}; "
        f"PR(step) {basis['pr']:.2f}; PR(unit-row) {basis_u['pr']:.2f}; "
        f"PR(grad) {basis_g['pr']:.2f}; wash removed fraction {removed:.6f}")
    write_partial("PARTIAL: SVD basis built", {"svd": state["svd"]})

    # =====================================================================
    # PART B — the e131 arms (rung ladder; g3K's matched-RMS convention)
    # =====================================================================
    log("=" * 78)
    log("PART B e131 — in-span random vs out-span wash at the rung ladder "
        f"(wash-1x = {STEP_SCALE_COMMITTED:.4f}; kill g-12 <= {SHUT_BAR})")

    def read_at(theta_vec: torch.Tensor) -> dict:
        load_flat(evl, theta_vec)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        return {"gm12": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
                "ce_r": ce_fixed_cpu(evl, *r_eval_xy)}

    root_read = read_at(theta0)

    def ladder(u_vec: torch.Tensor, tag: str, store_key: str) -> list[dict]:
        rows = [{"rung": 0, "gm12": root_read["gm12"],
                 "frac_argmax_z": root_read["frac_argmax_z"],
                 "ce_r": root_read["ce_r"]}]
        for r in RUNGS:
            D = r * STEP_SCALE_COMMITTED
            th = theta0 - D * u_vec
            rd = read_at(th)
            chk = float(torch.norm(th - theta0))
            rows.append({"rung": r, "D": D, "rms": D / RMS_DEN, **rd,
                         "disp_dev": abs(chk - D)})
            log(f"  {tag:>16s} rung {r:2d}x (D {D:7.3f}): g-12 "
                f"{rd['gm12']:.4f} CE_R {rd['ce_r']:.3f}")
            state["partB_e131"].setdefault(store_key, []).append(rows[-1])
            write_partial(f"PARTIAL: e131 {tag} rung {r} done")
        return rows

    wash_rows = ladder(u_g, "WASH(g-ray)", "wash_rows")
    G_WASHRUNG1 = {
        "read_gm12": wash_rows[1]["gm12"],
        "committed_e191_endpoint": E191_ENDPOINT_GM12,
        "d": abs(wash_rows[1]["gm12"] - E191_ENDPOINT_GM12),
        "tol": 1e-3,
        "pass": bool(abs(wash_rows[1]["gm12"] - E191_ENDPOINT_GM12) < 1e-3),
        "note": "the wash leg's rung-1 point (D 1.6543 on the committed "
                "g-ray) must reproduce e191's committed step-1-endpoint read",
    }
    log(f"G_WASHRUNG1: {wash_rows[1]['gm12']:.7f} vs {E191_ENDPOINT_GM12:.7f}: "
        + ("PASS" if G_WASHRUNG1["pass"] else "FAIL"))
    if not G_WASHRUNG1["pass"]:
        raise RuntimeError("wash rung-1 anchor FAILED — abort")
    wash_an = kill_analysis(wash_rows, "gm12", SHUT_BAR)

    inspan_ans = []
    for sd in INSPAN_SEEDS_E131:
        g = torch.Generator().manual_seed(sd)
        c = torch.randn(r_avail, generator=g, dtype=torch.float64)
        v = Vr64.T @ c
        u_in = (v / torch.norm(v)).to(torch.float32)
        in_cos = cos64(u_in, u_g)
        rows = ladder(u_in, f"INSPAN s{sd}", f"inspan_{sd}")
        an = kill_analysis(rows, "gm12", SHUT_BAR)
        an["draw_seed"] = sd
        an["cos_to_wash_dir"] = in_cos
        inspan_ans.append(an)
        log(f"  INSPAN s{sd}: kill rung {an['kill_rung']} thr "
            f"{an['threshold']}; cos to wash dir {in_cos:+.4f}")

    out_cos = cos64(u_out, u_g)
    out_rows = ladder(u_out, "OUTSPAN wash", "outspan_rows")
    out_an = kill_analysis(out_rows, "gm12", SHUT_BAR)
    log(f"  OUTSPAN: kill rung {out_an['kill_rung']} thr "
        f"{out_an['threshold']}; cos to wash dir {out_cos:+.4f}")

    # =====================================================================
    # RIDER 1 — THE SHUFFLED-SIGN RAY (report-only)
    # =====================================================================
    log("=" * 78)
    log("RIDER 1 — the shuffled-sign ray (sign census + decile magnitude "
        f"profile kept, coordinate-sign pairing destroyed; seed {RIDER1_SHUFFLE_SEED})")
    order = torch.argsort(g0_t0.abs())            # ascending |g|
    signs_sorted = torch.sign(g0_t0)[order]
    gsh = torch.Generator().manual_seed(RIDER1_SHUFFLE_SEED)
    s_shuf_sorted = torch.empty_like(signs_sorted)
    blk = N // 10
    decile_stats = []
    for d in range(10):
        lo, hi = d * blk, ((d + 1) * blk if d < 9 else N)
        chunk = signs_sorted[lo:hi]
        perm = torch.randperm(hi - lo, generator=gsh)
        s_shuf_sorted[lo:hi] = chunk[perm]
        decile_stats.append({
            "decile": d, "n": int(hi - lo),
            "n_pos": int((chunk > 0).sum()), "n_neg": int((chunk < 0).sum()),
            "n_zero": int((chunk == 0).sum()),
            "abs_g_range": [float(g0_t0.abs()[order][lo]),
                            float(g0_t0.abs()[order][hi - 1])],
        })
    s_shuf = torch.empty_like(s_shuf_sorted)
    s_shuf[order] = s_shuf_sorted
    u_shuf = (s_shuf / torch.norm(s_shuf)).to(torch.float32)
    u_sign_ray = torch.sign(g0_t0) / torch.norm(torch.sign(g0_t0))
    r1_rows = []
    for D in RIDER1_GRID:
        load_flat(evl, theta0 - D * u_shuf)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        ce = ce_fixed_cpu(evl, *r_eval_xy)
        r1_rows.append({"D": D, "rms": D / RMS_DEN, "gm12": gz["mean_pz"],
                        "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce})
        log(f"  SHUF-SIGN D {D:5.2f}: g-12 {gz['mean_pz']:.4f} CE_R {ce:.3f}")
        state["riders"]["rider1_rows"] = r1_rows
        write_partial(f"PARTIAL: rider1 D {D} done")
    small = [r for r in r1_rows if r["D"] in SMALL_D_PUMP_BAND]
    r1_rise = max(r["gm12"] - root_cells["gm12"] for r in small)
    rider1 = {
        "registration": "sign(g_0) with its coordinate assignment shuffled "
                        "within |g_0| magnitude deciles (equal-count by "
                        "rank); same sign census per decile, same decile "
                        "membership, wrong coordinate-sign pairing; "
                        "REPORT-ONLY (no bar)",
        "shuffle_seed": RIDER1_SHUFFLE_SEED,
        "decile_stats": decile_stats,
        "cos_to_sign_ray": cos64(u_shuf, u_sign_ray),
        "cos_to_g_ray": cos64(u_shuf, u_g),
        "rows": r1_rows,
        "small_D_max_rise": r1_rise,
        "pump_ridge_margin": PUMP_RIDGE_MARGIN,
        "pump_ridge": bool(r1_rise > PUMP_RIDGE_MARGIN),
        "first_dead_D": next((r["D"] for r in r1_rows
                              if r["gm12"] <= SHUT_BAR), None),
        "context_sign_ray_first_dead": 2.5,
        "context_g_ray_first_dead": 0.92,
        "question": "does the pump survive the shuffle? (e192's open "
                    "question: exact coordinate-sign pairing vs "
                    "magnitude-profile + sign-census)",
    }
    state["riders"]["rider1"] = rider1

    # =====================================================================
    # RIDER 2 — THE WIDER RANDOM GRID (report-only)
    # =====================================================================
    log("=" * 78)
    log("RIDER 2 — extend the three Gaussian rays to D "
        f"{list(RIDER2_GRID)} (existing rungs LOADED verbatim)")

    def draw_gaussian_ray(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        blocks = [torch.randn(p.shape, generator=g).reshape(-1)
                  for p in net0.parameters()]
        v = torch.cat(blocks)
        return (v / torch.norm(v)).clone()

    rider2 = {"registration": "the three committed Gaussian rays re-drawn "
              "from their committed seeds (g3K per-tensor convention), the "
              "committed D=4.0 row reproduced as a HARD machinery gate, then "
              "only the NEW rungs read; REPORT-ONLY",
              "rays": [], "grid_new": list(RIDER2_GRID)}
    rand_keys = list(e192_rand)
    for key, seed in zip(rand_keys, RAND_SEEDS):
        u_ray = draw_gaussian_ray(seed)
        u_md5 = hashlib.md5(u_ray.numpy().tobytes()).hexdigest()
        md5_ok = u_md5 == e192m["cell"]["rays"][2 + RAND_SEEDS.index(seed)]["u_md5"]
        th4 = theta0 - 4.0 * u_ray
        load_flat(evl, th4)
        evl.eval()
        fresh4 = battery_cell(evl, gm12_ids, zid)["mean_pz"]
        committed4 = next(r["gm12"] for r in e192_rand[key] if r["D"] == 4.0)
        g4 = {"fresh_gm12": fresh4, "committed_gm12": committed4,
              "d": abs(fresh4 - committed4), "tol": 1e-3,
              "pass": bool(abs(fresh4 - committed4) < 1e-3),
              "u_md5_match": bool(md5_ok)}
        if not g4["pass"]:
            write_partial("FAILED GATE — rider2 D=4.0 anchor", {"failed": True})
            raise RuntimeError(f"rider2 anchor FAILED for {key}")
        new_rows = []
        for D in RIDER2_GRID:
            load_flat(evl, theta0 - D * u_ray)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            ce = ce_fixed_cpu(evl, *r_eval_xy)
            new_rows.append({"D": D, "rms": D / RMS_DEN,
                             "gm12": gz["mean_pz"],
                             "frac_argmax_z": gz["frac_argmax_z"],
                             "ce_r": ce, "fresh": True})
            log(f"  {key} D {D:5.1f}: g-12 {gz['mean_pz']:.4f} CE_R {ce:.3f}")
            state["riders"]["rider2_partial_last"] = {"key": key, "D": D}
            write_partial(f"PARTIAL: rider2 {key} D {D} done")
        full_rows = sorted(
            [{"D": r["D"], "gm12": r["gm12"], "ce_r": r["ce_r"],
              "fresh": False} for r in e192_rand[key]] + new_rows,
            key=lambda r: r["D"])
        dead = [r for r in full_rows if r["gm12"] <= SHUT_BAR]
        rider2["rays"].append({
            "key": key, "seed": seed, "u_md5": u_md5,
            "u_md5_match_committed": bool(md5_ok),
            "d4_anchor": g4, "new_rows": new_rows, "all_rows": full_rows,
            "first_dead_D": dead[0]["D"] if dead else None,
            "unresolved_high": not dead,
        })
    rand_first_deads = [r["first_dead_D"] for r in rider2["rays"]]
    rider2["random_kill_first_dead"] = rand_first_deads
    rider2["random_kill_effective"] = (
        min(d for d in rand_first_deads if d) if any(rand_first_deads)
        else ">12 (all three unresolved-high at D 12)")
    log(f"  RIDER 2: first-dead per ray {rand_first_deads} -> "
        f"{rider2['random_kill_effective']}")
    state["riders"]["rider2"] = rider2
    write_partial("PARTIAL: riders complete")

    # =====================================================================
    # PART B — the g3 legs (both rulers where available: store g0 + host p(Z))
    # =====================================================================
    log("=" * 78)
    log("PART B g3 legs — the store (g0 ruler) and host (battery p(Z) ruler) "
        "organisms; their own wash histories; g3K's committed ladders LOADED")
    torch.set_num_threads(8)                      # g3K's own reduction order
    import g3R_seed_replicates as R
    import g3_generative_store as G3S
    import g3K_kappa as GK
    common.DEVICE = "cpu"
    pr = R.rebuild_protocol()
    sd_store, skeys, _, store_gates = R.load_root(pr)
    sd_host = GK.load_ck("e157_f2_consolidated.pt")
    hash_prov(CKPT_DIR / "g3_gen.pt", "the store organism (g3/g3R lineage)")
    hash_prov(CKPT_DIR / "e157_f2_consolidated.pt",
              "the host-fact organism (S-DISC twin)")
    host_net = TinyGPT(GK.F2_CFG)
    host_net.load_state_dict(sd_host)

    def g3_gates() -> dict:
        sd_s1 = GK.load_ck("g3_gen_s1.pt")
        sdh_s1 = GK.load_ck("e157_f2_neutral_s1.pt")
        sn = G3S.evl_load("gen", sd_store)
        g0r = G3S.battery_cell(sn, pr["ids130"], pr["zid"])["mean_pz"]
        sn1 = G3S.evl_load("gen", sd_s1)
        g0s1 = G3S.battery_cell(sn1, pr["ids130"], pr["zid"])["mean_pz"]
        host_net.load_state_dict(sd_host)
        pzr = G3S.battery_cell(host_net, pr["ids130"], pr["zid"])["mean_pz"]
        host_net.load_state_dict(sdh_s1)
        pzs1 = G3S.battery_cell(host_net, pr["ids130"], pr["zid"])["mean_pz"]
        D_st = G3S.sd_disp(sd_s1, sd_store)
        D_hs = G3S.sd_disp(sdh_s1, sd_host)
        ref = GK.REF
        out = {
            "store_root_g0": {"measured": g0r, "ref": ref["store"]["root_g0"],
                              "pass": bool(abs(g0r - ref["store"]["root_g0"]) < G_TOL_G3)},
            "store_s1_g0": {"measured": g0s1, "ref": ref["store"]["wash_s1_g0"],
                            "pass": bool(abs(g0s1 - ref["store"]["wash_s1_g0"]) < 1e-4)},
            "store_D_all_s1": {"measured": D_st, "ref": ref["store"]["D_all_s1"],
                               "pass": bool(abs(D_st - ref["store"]["D_all_s1"]) < G_TOL_G3)},
            "host_root_pz": {"measured": pzr, "ref": ref["host"]["root_g0"],
                             "pass": bool(abs(pzr - ref["host"]["root_g0"]) < G_TOL_G3)},
            "host_s1_pz": {"measured": pzs1, "ref": ref["host"]["wash_s1_g0"],
                           "pass": bool(abs(pzs1 - ref["host"]["wash_s1_g0"]) < 1e-4)},
            "host_D_all_s1": {"measured": D_hs, "ref": ref["host"]["D_all_s1"],
                              "pass": bool(abs(D_hs - ref["host"]["D_all_s1"]) < G_TOL_G3)},
        }
        out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
        return out

    GG3 = g3_gates()
    log(f"G_G3 (store/host ruler gates vs g3K's committed refs): "
        + ("PASS" if GG3["pass"] else "FAIL"))
    if not GG3["pass"]:
        write_partial("FAILED GATE — G_G3", {"failed": True})
        raise RuntimeError("g3 organism gates FAILED — abort")

    def run_g3_leg(tag: str, sd_root: dict, hist_cks: list[str],
                   inspan_seeds: tuple[int, ...], reader, net_holder,
                   read_key: str, loaded_ref: dict) -> dict:
        """One organism's leg: history SVD -> in-span/out-span rung ladders."""
        def root_flat() -> torch.Tensor:
            return GK.flat_delta(sd_root, {k: sd_root[k].float()
                                           for k in sd_root})
        # absolute state flats (sorted-key convention, g3K's flat_delta)
        flats = [root_flat()]
        for ck in hist_cks:
            sd_t = GK.load_ck(ck)
            hash_prov(CKPT_DIR / ck, f"{tag} history snapshot")
            flats.append(GK.flat_delta(
                sd_root, {k: sd_t[k].float() for k in sd_root}))
        segments = [flats[i + 1] - flats[i] for i in range(len(flats) - 1)]
        Hh = torch.stack(segments)
        bb = svd_basis(Hh)
        rr = bb["rank_eff"]
        Vv = bb["V"].T[:rr].to(torch.float32).contiguous()
        Vv64 = Vv.to(torch.float64)
        D_w1 = float(torch.norm(segments[0]))       # ||wash 1x|| (root->s1)
        u_w = segments[0] / torch.norm(segments[0])
        proj = Vv64 @ u_w.to(torch.float64)
        removed = float(torch.norm(proj))
        u_o64 = u_w.to(torch.float64) - Vv64.T @ proj
        u_o = (u_o64 / torch.norm(u_o64)).to(torch.float32)
        rf = root_flat()

        def read_flat(flat: torch.Tensor) -> dict:
            i = 0
            sd = {}
            with torch.no_grad():
                for k in sorted(sd_root.keys()):
                    n = sd_root[k].numel()
                    sd[k] = flat[i:i + n].view_as(sd_root[k].clone()).clone()
                    i += n
            return reader(sd, net_holder(), pr)

        base = read_flat(rf)
        rows_wash, rows_out = [{"rung": 0, **base}], [{"rung": 0, **base}]
        for rg in RUNGS:
            rows_wash.append({"rung": rg, **read_flat(rf - rg * D_w1 * u_w)})
            rows_out.append({"rung": rg, **read_flat(rf - rg * D_w1 * u_o)})
            log(f"  [{tag}] rung {rg:2d}x (D {rg * D_w1:7.3f}): wash "
                f"{rows_wash[-1][read_key]:.4f} outspan {rows_out[-1][read_key]:.4f}")
            state["partB_g3"].setdefault(tag, {})[f"rung_{rg}"] = {
                "wash": rows_wash[-1], "outspan": rows_out[-1]}
            write_partial(f"PARTIAL: g3 {tag} rung {rg} done")
        inspan_ans_l, inspan_rows = [], {}
        for sd_ in inspan_seeds:
            g = torch.Generator().manual_seed(sd_)
            c = torch.randn(rr, generator=g, dtype=torch.float64)
            v = Vv64.T @ c
            u_i = (v / torch.norm(v)).to(torch.float32)
            rows = [{"rung": 0, **base}]
            for rg in RUNGS:
                rows.append({"rung": rg, **read_flat(rf - rg * D_w1 * u_i)})
            inspan_rows[sd_] = rows
            an = kill_analysis(rows, read_key, KILL_BAR_G3)
            an["draw_seed"] = sd_
            an["cos_to_wash_dir"] = cos64(u_i, u_w)
            inspan_ans_l.append(an)
            log(f"  [{tag}] inspan s{sd_}: kill rung {an['kill_rung']} "
                f"thr {an['threshold']}")
        return {
            "n_params": int(sum(v.numel() for v in sd_root.values())),
            "D_wash_1x": D_w1, "n_history_segments": len(segments),
            "rank_available": rr,
            "spectrum": [float(s) for s in bb["sv"]],
            "participation_ratio": bb["pr"],
            "wash_dir_removed_fraction": removed,
            "outspan_cos_to_wash": cos64(u_o, u_w),
            "wash": kill_analysis(rows_wash, read_key, KILL_BAR_G3),
            "outspan": kill_analysis(rows_out, read_key, KILL_BAR_G3),
            "inspan": inspan_ans_l,
            "rows_wash": rows_wash, "rows_outspan": rows_out,
            "rows_inspan": {str(k): v for k, v in inspan_rows.items()},
            "references_loaded": {
                "g3k_wash_kill_rung": loaded_ref["kill"]["kill_rung"],
                "g3k_iso_thresholds": [a["threshold"] for a in loaded_ref["iso"]],
                "g3k_iso_kill_rungs": [a["kill_rung"] for a in loaded_ref["iso"]],
                "g3k_kappa": loaded_ref["kappa"],
            },
        }

    store_net = G3S.evl_load("gen", sd_store)
    store_leg = run_g3_leg(
        "store", sd_store,
        ["g3_gen_s1.pt", "g3_gen_s2.pt", "g3_gen_s4.pt", "g3_gen_s10.pt",
         "g3_gen_s50.pt", "g3_gen_s100.pt", "g3_gen_s200.pt", "g3_gen_s300.pt"],
        INSPAN_SEEDS_STORE, GK.read_store, lambda: store_net,
        "g0", g3k_store)
    host_net2 = TinyGPT(GK.F2_CFG)

    def read_host_sd(sd: dict, net, pr_):
        net.load_state_dict(sd)
        bz = G3S.battery_cell(net, pr_["ids130"], pr_["zid"])
        ce = G3S.ce_fixed_cpu(net, *pr_["r_eval_xy"])
        return {"pz": bz["mean_pz"], "frac_argmax_z": bz["frac_argmax_z"],
                "ce_r": ce}

    host_leg = run_g3_leg(
        "host", sd_host,
        ["e157_f2_neutral_s1.pt", "e157_f2_neutral_s2.pt", "e157_f2_neutral_s4.pt",
         "e157_f2_neutral_s50.pt", "e157_f2_neutral_s100.pt",
         "e157_f2_neutral_s200.pt", "e157_f2_neutral.pt"],
        INSPAN_SEEDS_HOST, read_host_sd, lambda: host_net2, "pz", g3k_host)
    state["partB_g3"]["store"] = {k: v for k, v in store_leg.items()
                                  if not k.startswith("rows_")}
    state["partB_g3"]["host"] = {k: v for k, v in host_leg.items()
                                 if not k.startswith("rows_")}
    torch.set_num_threads(4)

    # =====================================================================
    # ADJUDICATION (registered clauses; frozen operationalizations; no shopping)
    # =====================================================================
    log("=" * 78)
    log("ADJUDICATION")

    # ---- PART A (at t=0, the flip's home state; the census reported in full)
    t0_row = next(r for r in state["census"] if r["step"] == 0)
    a1_cut = next((c for c in t0_row["cuts"]
                   if c["cos_top_m12"] >= ALIGN_FLOOR
                   and c["cos_comp_m12"] <= -ALIGN_FLOOR), None)
    flip_present = bool(t0_row["cos_mov_m12"] > 0 > t0_row["cos_adamview_m12"])
    a2_allpos = all(c["cos_raw_m12"] > 0 for c in t0_row["classes"]) and flip_present
    a3_allfloor = all(abs(c["cos_raw_m12"]) < ALIGN_FLOOR
                      and abs(c["cos_adamview_m12"]) < ALIGN_FLOOR
                      for c in t0_row["classes"]) and \
        all(abs(c["cos_top_m12"]) < ALIGN_FLOOR
            and abs(c["cos_comp_m12"]) < ALIGN_FLOOR for c in t0_row["cuts"])
    if a1_cut is not None:
        verdict_a, clause_a = "A1 STITCHES-AND-CUTS", (
            f"at t=0 the cut k={a1_cut['k_frac']} has top alignment "
            f"{a1_cut['cos_top_m12']:+.4f} (>= +{ALIGN_FLOOR}) vs complement "
            f"{a1_cut['cos_comp_m12']:+.4f} (<= -{ALIGN_FLOOR}) — W024's "
            f"mechanism confirmed as structure: the pump lives in the big "
            f"coordinates, the erosion in the mass of small ones; the "
            f"normalizer's flattening lets the cuts outvote the stitches.")
    elif a2_allpos:
        verdict_a, clause_a = "A2 FLAT-POSITIVE", (
            f"all four magnitude classes align positively at t=0 "
            f"({[round(c['cos_raw_m12'], 4) for c in t0_row['classes']]}) "
            f"while the flip is present (raw {t0_row['cos_mov_m12']:+.4f} / "
            f"adam {t0_row['cos_adamview_m12']:+.4f}) — the flip arises from "
            f"the second-moment denominator's correlation structure, not "
            f"from a magnitude-class split; different mechanism, reported.")
    elif a3_allfloor:
        verdict_a, clause_a = "A3 NO-STRUCTURE", (
            f"every class and cut alignment at t=0 sits below the declared "
            f"floor {ALIGN_FLOOR} — the flip is noise-scale; the partition "
            f"carries no structure; reported honestly.")
    else:
        verdict_a, clause_a = "NO A-BAR FIRES (the unnamed fourth pattern)", (
            f"the t=0 census shows a pattern outside the three registered "
            f"shapes (classes "
            f"{[round(c['cos_raw_m12'], 4) for c in t0_row['classes']]}; cuts "
            f"top {[round(c['cos_top_m12'], 4) for c in t0_row['cuts']]} / "
            f"comp {[round(c['cos_comp_m12'], 4) for c in t0_row['cuts']]}) "
            f"— reported verbatim; the full 21-state census is the map.")
    n_a1_states = sum(
        1 for r in state["census"]
        if any(c["cos_top_m12"] >= ALIGN_FLOOR
               and c["cos_comp_m12"] <= -ALIGN_FLOOR for c in r["cuts"]))
    log(f"  PART A verdict: {verdict_a}")
    log(f"    {clause_a}")
    log(f"    secondary scan: {n_a1_states}/{len(state['census'])} states "
        f"show the A1 pattern (co-reported, non-adjudicating)")

    # ---- PART B (per organism; e131 primary)
    def b_verdict(inspan_thrs, outspan_thr, fullrand_min_thr, org):
        if max(inspan_thrs) <= 2 and (outspan_thr is None or outspan_thr > 16):
            return "B1 SUBSPACE-CARRIES", (
                f"[{org}] in-span random kills at rung <= 2 (thresholds "
                f"{[round(t, 2) for t in inspan_thrs]}) while the out-span "
                f"wash fails to kill through 16 (thr "
                f"{outspan_thr if outspan_thr else 'off-grid-high'}) — the "
                f"kappas are the projection ratio sqrt(d/d_eff); 'static "
                f"basin' retires.")
        if min(inspan_thrs) >= fullrand_min_thr / 2:
            return "B3 SUBSPACE-REFUTED", (
                f"[{org}] in-span random spares like the full random arm "
                f"(min in-span thr {min(inspan_thrs):.2f} vs full-random min "
                f"{fullrand_min_thr:.2f}, within half an octave) — the "
                f"picture dies; the static/learned contrast needs a "
                f"different mechanism (the census becomes the lead).")
        return "B2 PARTIAL-PROJECTION", (
            f"[{org}] graded — kill rungs as the empirical projection "
            f"profile (in-span {[round(t, 2) for t in inspan_thrs]}; "
            f"out-span {outspan_thr}); both dimension estimates compared.")

    rand_thrs_d = [first_dead_thr_D(r["all_rows"], SHUT_BAR)
                   for r in rider2["rays"]]
    finite_rand = [t for t in rand_thrs_d if t]
    if finite_rand:
        fullrand_thr_rung = min(finite_rand) / STEP_SCALE_COMMITTED
        fullrand_note = (f"from the rider-2-extended random profile "
                         f"(D {min(finite_rand):.2f})")
    else:
        fullrand_thr_rung = 12.0 / STEP_SCALE_COMMITTED   # bounded below
        fullrand_note = "unresolved-high even at D 12; bounded below by " \
                        "12/wash-1x rungs"
    e131_inspan_thrs = [a["threshold"] if a["threshold"] else OFFGRID_THR
                        for a in inspan_ans]
    vb_e131, clause_b_e131 = b_verdict(e131_inspan_thrs, out_an["threshold"],
                                       fullrand_thr_rung, "e131")
    st_thrs = [a["threshold"] if a["threshold"] else OFFGRID_THR
               for a in store_leg["inspan"]]
    hs_thrs = [a["threshold"] if a["threshold"] else OFFGRID_THR
               for a in host_leg["inspan"]]
    st_ref_thr = min([t for t in (a["threshold"] for a in g3k_store["iso"])
                      if t] or [64.0])
    hs_ref_thr = min([t for t in (a["threshold"] for a in g3k_host["iso"])
                      if t] or [64.0])
    vb_store, clause_b_store = b_verdict(st_thrs,
                                         store_leg["outspan"]["threshold"],
                                         st_ref_thr, "store")
    vb_host, clause_b_host = b_verdict(hs_thrs,
                                       host_leg["outspan"]["threshold"],
                                       hs_ref_thr, "host")
    dims = {
        "e131": {"d": N_PARAM,
                 "kappa_full_random": fullrand_thr_rung,
                 "kappa_source": fullrand_note,
                 "d_eff_kappa": N_PARAM / (fullrand_thr_rung ** 2),
                 "svd_rank": r_avail,
                 "svd_pr_step": basis["pr"],
                 "svd_pr_unitrow": basis_u["pr"],
                 "svd_pr_gradient_history": basis_g["pr"]},
        "store": {"d": store_leg["n_params"],
                  "kappa_g3k_committed": g3k_store["kappa"].get("kappa_value"),
                  "d_eff_kappa": (store_leg["n_params"]
                                  / g3k_store["kappa"]["kappa_value"] ** 2
                                  if g3k_store["kappa"].get("kappa_value")
                                  else None),
                  "svd_rank": store_leg["rank_available"],
                  "svd_pr": store_leg["participation_ratio"]},
        "host": {"d": host_leg["n_params"],
                 "kappa_g3k_committed": g3k_host["kappa"].get("kappa_value"),
                 "d_eff_kappa": (host_leg["n_params"]
                                 / g3k_host["kappa"]["kappa_value"] ** 2
                                 if g3k_host["kappa"].get("kappa_value")
                                 else None),
                 "svd_rank": host_leg["rank_available"],
                 "svd_pr": host_leg["participation_ratio"]},
        "comparison_note": "kappa^-2*d (the projection picture's d_eff) vs "
                           "the SVD-derived quantities (available rank + "
                           "participation ratio): the SVD history spans are "
                           "FINITE-SAMPLE proxies — rank << any candidate "
                           "d_eff — so the comparison reads as 'the "
                           "available span is a tiny lower bound', never as "
                           "an estimate of d_eff itself (the finite-span "
                           "proxy caveat carried verbatim).",
    }
    log(f"  PART B verdict e131: {vb_e131}")
    log(f"    {clause_b_e131}")
    log(f"  PART B co-adjudication store: {vb_store}; host: {vb_host}")
    b_agree = len({vb_e131[:2], vb_store[:2], vb_host[:2]}) == 1
    verdict_b = vb_e131 if b_agree else (
        f"{vb_e131[:2]}-SPLIT (e131 {vb_e131[:2]}, store {vb_store[:2]}, "
        f"host {vb_host[:2]})")

    bars = {
        "A1_STITCHES_AND_CUTS": {"fires": verdict_a.startswith("A1")},
        "A2_FLAT_POSITIVE": {"fires": verdict_a.startswith("A2")},
        "A3_NO_STRUCTURE": {"fires": verdict_a.startswith("A3")},
        "B1_SUBSPACE_CARRIES": {"fires": verdict_b.startswith("B1")},
        "B2_PARTIAL_PROJECTION": {"fires": verdict_b.startswith("B2")},
        "B3_SUBSPACE_REFUTED": {"fires": verdict_b.startswith("B3")},
    }

    # =====================================================================
    # THE CHART (one canvas: census + subspace + the e191/e192 terrain context)
    # =====================================================================
    fig = plt.figure(figsize=(16.5, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.15])
    axA = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[0, 1])
    axAt = fig.add_subplot(gs[0, 2])
    axB = fig.add_subplot(gs[1, 0])
    axBs = fig.add_subplot(gs[1, 1])
    axT = fig.add_subplot(gs[1, 2])

    names = [c["name"] for c in t0_row["classes"]]
    xs = range(len(names))
    axA.bar([x - 0.18 for x in xs], [c["cos_raw_m12"] for c in t0_row["classes"]],
            width=0.36, color="seagreen", alpha=0.9, label="raw movement g")
    axA.bar([x + 0.18 for x in xs],
            [c["cos_adamview_m12"] for c in t0_row["classes"]],
            width=0.36, color="darkorange", alpha=0.9, label="ADAM-VIEW -sign(g)")
    axA.axhline(ALIGN_FLOOR, color="gray", ls=":", lw=1.2)
    axA.axhline(-ALIGN_FLOOR, color="gray", ls=":", lw=1.2)
    axA.axhline(0, color="k", lw=0.8)
    for i, c in enumerate(t0_row["classes"]):
        axA.text(i, max(c["cos_raw_m12"], c["cos_adamview_m12"]) + 0.004,
                 f"{c['share']:.3f}", ha="center", fontsize=7, color="dimgray")
    axA.set_xticks(list(xs))
    axA.set_xticklabels(names, fontsize=8)
    axA.set_ylabel("cos(part, grad m12)  [+ = pump]")
    axA.set_title("PART A THE CENSUS at t=0 — the flip census\n"
                  f"whole-g raw {t0_row['cos_mov_m12']:+.4f} -> adam "
                  f"{t0_row['cos_adamview_m12']:+.4f} (gray = shares of "
                  f"||g||; floor {ALIGN_FLOOR})", fontsize=9)
    axA.legend(fontsize=8)

    axA2.plot([c["p_pct"] for c in t0_row["curve"]],
              [c["cos_top_m12"] for c in t0_row["curve"]], "o-",
              color="seagreen", lw=2, label="top-p part, t=0")
    axA2.plot([c["p_pct"] for c in t0_row["curve"]],
              [c["cos_comp_m12"] for c in t0_row["curve"]], "o--",
              color="crimson", lw=1.6, label="complement, t=0")
    mid = next((r for r in state["census"]
                if r["arm"] == "lr1e-3" and r["step"] == 50),
               state["census"][-1])
    axA2.plot([c["p_pct"] for c in mid["curve"]],
              [c["cos_top_m12"] for c in mid["curve"]], "^-",
              color="seagreen", lw=1.2, alpha=0.55,
              label=f"top-p, lr1e-3@s{mid['step']}")
    axA2.plot([c["p_pct"] for c in mid["curve"]],
              [c["cos_comp_m12"] for c in mid["curve"]], "^--",
              color="crimson", lw=1.1, alpha=0.55,
              label=f"complement, lr1e-3@s{mid['step']}")
    axA2.axhline(ALIGN_FLOOR, color="gray", ls=":", lw=1.2)
    axA2.axhline(-ALIGN_FLOOR, color="gray", ls=":", lw=1.2)
    axA2.set_xscale("log")
    axA2.set_xlabel("percentile cut p (top-p of |g|)")
    axA2.set_ylabel("cos(part, grad m12)")
    axA2.set_title("the continuous percentile curve (the registered cuts "
                   "0.1/1/10% marked)", fontsize=9)
    for f in CUT_FRACS:
        axA2.axvline(f * 100, color="lightgray", lw=0.7, ls=":")
    axA2.legend(fontsize=7)

    for arm, mk, col in (("lr1e-3", "o", "crimson"),
                         ("lr3e-5", "s", "royalblue"),
                         ("lr1e-5", "^", "seagreen")):
        rows = [r for r in state["census"] if r["arm"] == arm and r["step"] > 0]
        axAt.plot([r["step"] for r in rows], [r["cos_mov_m12"] for r in rows],
                  mk + "-", color=col, lw=1.5, ms=5, label=f"{arm} raw")
        axAt.plot([r["step"] for r in rows],
                  [r["cos_adamview_m12"] for r in rows],
                  mk + ":", color=col, lw=1.1, ms=4, alpha=0.7,
                  label=f"{arm} adam")
    axAt.axhline(0, color="k", lw=0.8)
    axAt.set_xscale("log")
    axAt.set_xlabel("wash step at the snapshotted state")
    axAt.set_ylabel("cos(-g, grad m12)")
    axAt.set_title("the census trajectory (snapshot quadrature; the pump "
                   "decays with lr and step)", fontsize=9)
    axAt.legend(fontsize=6.5, ncol=3)

    cols3 = ["royalblue", "seagreen", "darkorange"]
    axB.plot([c["rung"] for c in wash_an["curve"] if c["rung"] > 0],
             [c["gm12"] for c in wash_an["curve"] if c["rung"] > 0], "o-",
             color="crimson", lw=2.2, label="WASH (g-ray; rung-1 gated)")
    for i, a in enumerate(inspan_ans):
        axB.plot([c["rung"] for c in a["curve"] if c["rung"] > 0],
                 [c["gm12"] for c in a["curve"] if c["rung"] > 0], "^--",
                 color=cols3[i % 3], lw=1.4, ms=5,
                 label=f"IN-SPAN random s{a['draw_seed']}")
    axB.plot([c["rung"] for c in out_an["curve"] if c["rung"] > 0],
             [c["gm12"] for c in out_an["curve"] if c["rung"] > 0], "v-",
             color="purple", lw=2.0, ms=5,
             label="OUT-SPAN wash (orthogonalized)")
    for ri, key in enumerate(rand_keys):
        rr = rider2["rays"][ri]
        axB.plot([r["D"] / STEP_SCALE_COMMITTED for r in rr["all_rows"]],
                 [r["gm12"] for r in rr["all_rows"]], ":", lw=1.1,
                 color="dimgray", alpha=0.85,
                 label=("FULL-RANDOM ref (e192 loaded + rider2)"
                        if ri == 0 else None))
    axB.axhline(SHUT_BAR, color="gray", ls=":", lw=1.4,
                label=f"kill bar {SHUT_BAR}")
    axB.set_xscale("log", base=2)
    axB.set_xlabel(f"rung (x wash-1x = {STEP_SCALE_COMMITTED:.4f} L2; "
                   "matched per-coordinate RMS)")
    axB.set_ylabel("g-12 battery")
    axB.set_title(f"PART B e131 — span rank {r_avail} (finite-span proxy; "
                  f"wash removed frac {removed:.4f})", fontsize=9)
    axB.legend(fontsize=6.5)

    for leg, tag, rkey, col in ((store_leg, "store", "g0", "royalblue"),
                                (host_leg, "host", "pz", "seagreen")):
        cx = [c["rung"] for c in leg["wash"]["curve"] if c["rung"] > 0]
        cy = [c[rkey] for c in leg["wash"]["curve"] if c["rung"] > 0]
        axBs.plot(cx, cy, "o-", color=col, lw=1.8, ms=4,
                  label=f"{tag} wash (g3K kill "
                        f"{leg['references_loaded']['g3k_wash_kill_rung']}x)")
        cx = [c["rung"] for c in leg["outspan"]["curve"] if c["rung"] > 0]
        cy = [c[rkey] for c in leg["outspan"]["curve"] if c["rung"] > 0]
        axBs.plot(cx, cy, "v-", color=col, lw=1.6, ms=4, alpha=0.65,
                  label=f"{tag} out-span")
        for a, mk in zip(leg["inspan"], ("^", "x", "*")):
            cx = [c["rung"] for c in a["curve"] if c["rung"] > 0]
            cy = [c[rkey] for c in a["curve"] if c["rung"] > 0]
            axBs.plot(cx, cy, mk + "--", color=col, lw=0.9, ms=4, alpha=0.5,
                      label=f"{tag} in-span s{a['draw_seed']}")
    axBs.axhline(KILL_BAR_G3, color="gray", ls=":", lw=1.4)
    axBs.set_xscale("log", base=2)
    axBs.set_xlabel("rung (x own wash-1x; g3K convention, kill 0.5)")
    axBs.set_ylabel("store g0 / host p(Z)")
    axBs.set_title("PART B both rulers — the g3 organisms (n=1 per ruler; "
                   "references from g3K committed)", fontsize=9)
    axBs.legend(fontsize=5.6, ncol=2)

    for key, col, lb in (("R1_G", "crimson", "g-ray (kill 0.92)"),
                         ("R2_SIGN", "darkorange", "sign ray (kill 2.5)")):
        rows = e192_prof[key]["rows"]
        axT.plot([r["D"] for r in rows], [r["gm12"] for r in rows],
                 "o-", color=col, lw=1.8, ms=3.5, label=lb)
    for ri, key in enumerate(rand_keys):
        rr = rider2["rays"][ri]
        axT.plot([r["D"] for r in rr["all_rows"]],
                 [r["gm12"] for r in rr["all_rows"]], ".:", lw=1.0,
                 color="dimgray",
                 label=("Gaussian rays + rider2" if ri == 0 else None))
    axT.plot([r["D"] for r in r1_rows], [r["gm12"] for r in r1_rows],
             "s--", color="purple", lw=1.8, ms=4,
             label="RIDER 1 shuffled-sign ray")
    axT.axhline(SHUT_BAR, color="gray", ls=":", lw=1.2)
    axT.set_xscale("log")
    axT.set_xlabel("D (absolute L2; dual currency: RMS = D/1655.0)")
    axT.set_ylabel("g-12 battery")
    axT.set_title("THE TERRAIN context (e191/e192 loaded) + both riders",
                  fontsize=9)
    axT.legend(fontsize=6.5)

    fig.suptitle("THE CHART — census x subspace x profile (the terrain's "
                 "full chart; e189+e190 merged; eval-only CPU; organ n=1 per "
                 "ruler; snapshot quadrature; finite-span proxy carried)",
                 fontsize=11)
    fig.text(0.5, 0.005,
             f"VERDICTS: PART A {verdict_a} | PART B {verdict_b} — riders "
             f"report-only (shuffled-sign pump "
             f"{'YES' if rider1['pump_ridge'] else 'no'}; random kill "
             f"{rider2['random_kill_effective']})",
             ha="center", fontsize=9.5,
             bbox=dict(fc="whitesmoke", ec="gray"))
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(rd / "the_chart.png", dpi=130)
    plt.close(fig)
    log(f"THE CHART written -> {rd / 'the_chart.png'}")

    # =====================================================================
    # FINAL METRICS
    # =====================================================================
    metrics = {
        "experiment": "e_chart",
        "date": common.now_iso(),
        "status": ("SMOKE — nothing adjudicated" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": ("scratch/chart_cell_design.md (the frozen merged "
                         "design) + the dispatch brief; the six bars VERBATIM "
                         "in the module docstring and in registered_bars, "
                         "frozen before compute"),
        "registered_bars": REGISTERED_BARS,
        "question": ("THE CHART: (A) which magnitude classes of the wash "
                     "gradient carry the pump and which the erosion (W024's "
                     "flip mechanism); (B) does the killing displacement "
                     "live in an empirical subspace or anywhere in parameter "
                     "space (W025's projection picture); plus the "
                     "shuffled-sign and wider-random-grid riders"),
        "root": f"runs/checkpoints/{ROOT_CK}",
        "gates": {
            "G_ROOT": G_ROOT, "G_T0": G_T0, "G_W024": G_W024, "G_U_CK": G_U_CK,
            "G_STREAM": G_STREAM, "G_STATES": G_STATES,
            "G_WASHRUNG1": G_WASHRUNG1, "G_G3": GG3,
            "all_pass": bool(G_ROOT["pass"] and G_T0["pass"] and G_W024["pass"]
                             and G_U_CK["pass"] and G_STREAM["pass"]
                             and G_STATES["all_pass"] and G_WASHRUNG1["pass"]
                             and GG3["pass"]),
        },
        "partA_census": {
            "convention": {
                "alignment": "cos(part of the wash MOVEMENT direction "
                             "(-post-clip gradient), grad of mean log p(Z) "
                             "on the g-12 battery at the same state); fp64; "
                             "+ = pump, - = erosion",
                "adam_view": "the flattening limit (-sign(g)); exact at t=0, "
                             "fresh-Adam flattening at t>0 (declared)",
                "states": "neutral-stream arms only (lr1e-3/3e-5/1e-5 + "
                          "opt1 A0's s10; 21 states + root); the 1e-4 "
                          "extinction-stream arm EXCLUDED (provenance)",
                "declared_floor": ALIGN_FLOOR,
                "cuts": list(CUT_FRACS),
                "percentile_grid": list(PCTL_GRID),
            },
            "states": state["census"],
            "secondary_scan": {
                "n_states_with_A1_pattern": n_a1_states,
                "n_states_total": len(state["census"]),
                "note": "co-reported, non-adjudicating (the A bars "
                        "adjudicate at t=0, the flip's home state)",
            },
        },
        "partB_subspace": {
            "e131": {
                "convention": {
                    "rungs": list(RUNGS),
                    "wash_1x": STEP_SCALE_COMMITTED,
                    "match": "perturbation L2 = rung x wash-1x; matched mean "
                             "per-coordinate RMS (g3K's convention)",
                    "kill_bar": SHUT_BAR,
                },
                "svd": state["svd"],
                "wash": wash_an, "inspan": inspan_ans, "outspan": out_an,
                "outspan_cos_to_wash": out_cos,
                "rows": {k: v for k, v in state["partB_e131"].items()
                         if isinstance(v, list)},
            },
            "store": store_leg, "host": host_leg,
            "dimension_estimates": dims,
        },
        "riders": {"rider1_shuffled_sign": rider1,
                   "rider2_wider_random_grid": rider2},
        "adjudication": {
            "partA": {"verdict": verdict_a, "clause": clause_a,
                      "secondary_scan_states": n_a1_states,
                      "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                                        ("A1_STITCHES_AND_CUTS",
                                         "A2_FLAT_POSITIVE",
                                         "A3_NO_STRUCTURE")}},
            "partB": {
                "verdict_e131_primary": vb_e131, "clause_e131": clause_b_e131,
                "co_adjudication_store": {"verdict": vb_store,
                                          "clause": clause_b_store},
                "co_adjudication_host": {"verdict": vb_host,
                                         "clause": clause_b_host},
                "composite": verdict_b,
                "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                                  ("B1_SUBSPACE_CARRIES",
                                   "B2_PARTIAL_PROJECTION",
                                   "B3_SUBSPACE_REFUTED")},
            },
            "bars_fired": bars,
            "no_bar_shopping": True,
        },
        "honesty": {
            "organ_n": "n=1 organism per ruler (the e131 consolidated root "
                       "for PART A + the e131 PART B leg + both riders; one "
                       "constructed store organism + one host-fact twin for "
                       "the g3 legs — g3K's standing scope carried)",
            "snapshot_quadrature": "the census is at sampled steps, not a "
                                   "trajectory: the wash gradient is the "
                                   "protocol's own next-batch gradient at a "
                                   "snapshotted state (gated where opt1 "
                                   "committed the norms)",
            "census_floor_declared": ALIGN_FLOOR,
            "finite_span_proxy": "r in {64,256,1024} caps at the available "
                                 "history rank — the in-span arm tests a "
                                 "conservative LOWER bound of the subspace; "
                                 "SPARING ON THE IN-SPAN ARM IS AMBIGUOUS "
                                 "and is reported as such",
            "outspan_construction": "the wash direction is ~fully in-span by "
                                    "construction (the step-1 segment is in "
                                    "the history); the out-span arm measures "
                                    "the orthogonal residual at matched D "
                                    "(effective in-span displacement 0); if "
                                    "it kills, something outside the span "
                                    "carries death — refutes cleanly",
            "numerical_rank_documented": state["svd"]["numerical_rank"],
            "cross_part": "no cross-part bar; the chart is the synthesis",
            "riders_report_only": True,
            "cross_organism": "each leg's kill bar is its own ruler's; "
                              "cross-organism numbers are pointers, never "
                              "claims",
        },
        "provenance": {
            "checkpoints": prov_files,
            "committed_metrics_consumed": [
                "runs/e191/metrics.json (direction md5 + endpoint read)",
                "runs/e192/metrics.json (the five-ray terrain: R1 = the "
                "C_WASH reference; R3-R5 = the D_FULL-RANDOM reference, "
                "their rungs LOADED verbatim)",
                "runs/e188/metrics.json (the census state battery-read gates)",
                "runs/opt1/metrics.json (the t=0 bit-identity anchors + the "
                "flip anchors + A0's s10 state)",
                "runs/g3K/metrics.json (the store/host wash+iso ladders and "
                "kappas — the both-rulers references)",
            ],
            "note": "every checkpoint hashed at load; every battery read "
                    "reproduced its committed row before use; every "
                    "recomputed direction gated against its committed md5 "
                    "(no silent loads)",
        },
        "trims": (["SMOKE: census 6 states; rungs {1,4,16}; one in-span seed "
                   "per organism; rider-2 grid {8,12}"] if SMOKE else []),
        "deviations": deviations,
        "compute": {
            "device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
            "threads": "4 (e131 lineage, e185..e192 reduction order); 8 for "
                       "the g3 legs (g3K's own reduction order)",
            "eval_only": True, "training_runs": 0, "gpu_claimed": False,
            "elapsed_s": round(time.time() - T0, 1),
        },
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("FINAL metrics.json written")
    log("=" * 78)
    log(f"E_CHART COMPLETE: PART A {verdict_a} | PART B {verdict_b}")
    log(f"  riders: shuffled-sign pump "
        f"{'YES' if rider1['pump_ridge'] else 'no'} (max rise "
        f"{rider1['small_D_max_rise']:+.4f}, first dead "
        f"{rider1['first_dead_D']}); random kill "
        f"{rider2['random_kill_effective']}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
