"""E192 — THE ONE-ORGANISM ALL-RAY TERRAIN MAP (R59-critic's forced cell;
the paper Fig-5 license) + THE FIXED-RAY RIDER.

RECOVERY NOTE: this is the SECOND dispatch. The first e192 executor was
killed by a machine disruption BEFORE writing any artifact (commit 3152a83,
"fourth disruption ... killed pre-artifact") — nothing was recovered but
the frozen brief. This file is the fresh build: design re-derived from the
dispatch brief + scratch/r59_critic.md's FORCED EXPERIMENT section only;
machinery taken VERBATIM from lab/e191_pump_cliff.py (which is itself
opt1c's, which is e185's — the e176n lineage). No result was known when
the bars below were frozen.

WHY (R59 attack 1, verbatim): Fig-5's three terrains are "one mapped ray,
one unmapped adaptive path whose 'width 2.5' is a cumulative path length
on a recomputed direction (static sign(g) curve never measured; one
static-equivalent point exists — 0.678 alive at D 1.654), and a
cross-organism static band on a 0.87M net with a different ruler battery,
a different fact, and wash-anchored units whose per-coordinate translation
the figure never states." Until this cell, T144's "lethality ordering
stands on mapped ground" is true for one and one-half of its three bands.
Either the middle and third bands stop being a stitch, or the central
figure dies here at the cheapest price.

WHAT IT BUILDS ON: e191's machinery VERBATIM (same e131_consolidated_e113
root, same g-12 install-60 battery + CE_R instruments, same t=0 bit-gate,
same protocol rebuild and gates, same eval-only CPU envelope); e191's
COMMITTED g-ray profile + its checkpointed direction
(runs/checkpoints/e191_static_dir_u.pt) — R1 re-reads fresh (dual currency
+ the 3 new Ds) and crosschecks against the committed 12 rows; opt1's
committed A0 step-1 read (gm12 0.6780440807342529 at L2 1.6542880535125732)
as R2's built-in crosscheck; g3K's isotropic-ray convention (full-vector
Gaussian, fresh seeds, even-spread expectation, no registered-seed
collisions). WHAT IS NEW: the STATIC sign(g_0) ray — never mapped on this
lineage; 3 fresh Gaussian unit rays (seeds 11501-3); the 15-point D grid
extending to 4.0; DUAL CURRENCY everywhere (absolute L2 + per-coordinate
RMS = D/sqrt(N), the Rule-12 currency R59 demanded); the pinned-walk rider
(the intervention that converts W026's "protection = re-orientation" from
correlation to test); the one-picture Fig-5 candidate.

THE CELL (eval-only, CPU, ~5-10 min): STATIC graded jumps theta_D =
theta_0 - D*u for FIVE ray families on the SAME root, ONE organism, ONE
ruler (the g-12 battery), SAME D grid {0.05, 0.1, 0.2, 0.33, 0.5, 0.66,
0.8, 0.92, 1.0, 1.17, 1.5, 2.0, 2.5, 3.0, 4.0}:
  R1 G-RAY   — u = g_0/||g_0||_2 LOADED from runs/checkpoints/
               e191_static_dir_u.pt (bit-gated: stored md5, root md5, and
               a fresh t=0 recompute must agree); reads at the 12 shared
               Ds must reproduce e191's committed profile.
  R2 SIGN-RAY — u = sign(g_0)/||sign(g_0)||_2 (STATIC sign pattern,
               normalized; never mapped on this lineage). CROSSCHECK
               (asserted BEFORE compute, tolerance frozen): a single
               static jump at D = 1.6542880535125732 must reproduce A0's
               committed step-1 read 0.6780440807342529 within 0.05 —
               the residual is Adam's weight-decay component (~0.008 L2,
               measured and reported); the deviation is reported verbatim
               either way. A second, harder anchor: a static placement at
               A0's EXACT fresh-AdamW step-1 endpoint must reproduce
               0.6780440807342529 within 1e-3 (machinery gate).
  R3-R5      — 3 fresh full-vector Gaussian unit rays, seeds
               11501/11502/11503 (no registered-seed collisions: g3K's
               isotropic seeds were 11401-3/11411-13; the e131 lineage
               seeds are 10902/10901/170/1337/24301/26502 — 11501-3 are
               fresh), g3K's per-tensor draw convention in flat_params
               (net.parameters()) order so each ray lives in the same
               coordinates as u; even-spread co-read reported (per-block
               L2 share vs numel share), guaranteed only in expectation.

RIDER: 300 PINNED small steps along the FIXED g-ray — delta =
(1.6542880535125732/300) * u with u FROZEN (no re-orientation; ~0.005514
L2/step), g-12 + CE_R every 30 steps (10 reads, D = 0.165 ... 1.6543).
DISCLOSED BEFORE COMPUTE: the pinned walk's parameter points coincide
with the static g-ray's points BY CONSTRUCTION (a straight walk down the
same ray, up to fp accumulation) — the rider's content is INTERVENTIONAL
FRAMING, not new geometry: step size held small, orientation denied. If
it dies near the static cliff, "small steps protect BY re-orienting"
becomes interventional; if it survives past 1.6543, step size owns the
sparing and "re-orientation" was a passenger (the e188 service).

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - TERRAIN-ONE-PICTURE: "sign-ray cliffs in the 2.0-3.2 stretch band;
    random rays kill wide >= 4x the g-ray with NO pump ridge at small D —
    Fig-5 licensed as one picture; the pump is gradient-specific"
  - BANDS-REORDER: "the order g < sign < random fails or random rays pump
    — the central figure dies here at the cheapest price; report
    verbatim"
  - RIDER-REORIENTATION-CAUSAL: "pinned walk dies near ~0.92"
  - RIDER-STEPSIZE-OWNS: "pinned walk survives past 1.6543"
  - GRADED: any partial pattern — everything reported verbatim as
    measured (the lab's standing GRADED floor).

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * kill = first grid D (ascending) with g-12 <= 0.27 (e185's DISSOLVE
    bar, the e191 convention); no kill on the grid = kill unresolved-high
    (> 4.0), reported as such (the inequality "kill > 4.0" still
    adjudicates the >= 4x clause).
  * "sign-ray cliffs in the 2.0-3.2 stretch band" = R2's first-dead grid
    D lies in {2.0, 2.5, 3.0} (subset of [2.0, 3.2]).
  * "random rays kill wide >= 4x the g-ray" = EVERY random ray's
    first-dead D >= 3.68 (= 4 x 0.92, the committed e191 cliff — frozen
    constant, not re-measured here) or unresolved-high.
  * "NO pump ridge at small D" on random rays = across {0.05, 0.1, 0.2,
    0.33, 0.5} and all 3 rays, no read exceeds root g-12 + 0.01 (a ridge
    = a rise of at least a quarter of the g-ray's own committed rise
    +0.040); strict single-point rises (any read > root) are co-reported
    verbatim so the clause cannot hide behind the margin.
  * "the order g < sign < random fails" = NOT (g_kill < sign_kill <
    rand_kill_eff), where rand_kill_eff = min over the 3 random rays'
    first-dead Ds with all-unresolved = +inf (the inequality survives an
    unresolved top band); an unresolved SIGN kill cannot be placed — the
    order is then not established and BANDS-REORDER fires with that
    verbatim note.
  * rider: battery reads at D_k = k*1.6542880535125732/300 for k in
    {30, 60, ..., 300} (D = 0.165, 0.331, 0.496, 0.662, 0.827, 0.993,
    1.158, 1.323, 1.489, 1.654); RIDER-REORIENTATION-CAUSAL fires iff the
    first dead read is at step k <= 180 (D <= 0.9926 — the read whose
    alive->dead bracket [0.827, 0.9926] contains 0.92, or one read below
    it); RIDER-STEPSIZE-OWNS fires iff no read through k = 300 is dead;
    anything else is GRADED (death in (0.9926, 1.6543] — above the 0.92
    neighborhood, below the end).
  * composite orders frozen: map TERRAIN-ONE-PICTURE -> BANDS-REORDER ->
    GRADED (the first two are mutually exclusive by construction);
    rider RIDER-REORIENTATION-CAUSAL -> RIDER-STEPSIZE-OWNS -> GRADED.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE the profiles run): e191's
full gate set VERBATIM (corpus ZEPH count 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16 starts
bit-equal to e185's stored list; root gated vs e151's before-cells) PLUS
the t=0 BIT-IDENTITY GATE (opt1c's verbatim: step-1 batch md5 vs e185's
stored hash; step-1 forward CE and pre-clip grad norm vs opt1's committed
values; one fresh AdamW-recipe step from the root reproducing A0's
committed step-1 L2) PLUS G_U_CK (the e191 direction bit-gate: the
loaded u's md5 vs e191's committed direction_md5; the root's flat md5 vs
the checkpoint's stored theta0_md5; a fresh t=0 direction recompute vs
the loaded u — md5-identical, with the documented fp64-cosine fallback
honoring the fp32 reduction-order texture) PLUS G_R1_XCHK (the fresh R1
reads at the 12 shared Ds vs e191's committed profile, max |d| < 1e-3)
PLUS G_A0_XC (the A0 anchors: exact-endpoint read within 1e-3 HARD; the
sign-ray-at-1.6543 read within 0.05 FROZEN — recorded verbatim, failure
flags R2's step-1-equivalence but does not abort the map) PLUS the
rider's own step-300 anchor vs e191's committed step-1 endpoint read.
WHAT EACH RAY GUARANTEES: NOTHING — each ray guarantees only its
geometry (unit L2 from theta_0 along the stated direction family);
whether it pumps, spares, or kills at any D is the measurement; that
openness is the point.

DUAL CURRENCY (R59's demand): every profile row reports D (absolute L2)
AND rms = D/sqrt(N), N = 2,739,072 (denominator 1655.0142...); the Fig-5
candidate carries both axes (bottom: absolute L2; top: per-coordinate
RMS). Cross-organism context (pointer only, no claims): e131's g-ray
kill 0.92 = 5.6e-4 RMS; A0's sign path 2.489 = 1.5e-3 RMS; g3K's random
kill rung ~7.8e-3 RMS on ITS OWN 0.87M organism/ruler (scratch/
r59_critic.md sec 1b).

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; no GPU claim), torch threads 4 (the e185/opt1/opt1b/opt1c/
e191 reduction order), EVAL-ONLY (no training, no optimizer in any
profile arm; every grid point an independent perturb-and-eval off the
root; bit-deterministic), n=1 organism (the e131 consolidated root, seed
lineage 10902), single organism + single fact + single root for every
ray — the honest replicate axis (a second organism) stays a separate
dispatch decision.

Outputs: runs/e192/{metrics.json, e192_fig5_terrain.png,
e192_rider.png, journal.jsonl}. PROGRESSIVE PARTIAL metrics.json writes
(after the gates, after every grid point — a superset of the required
per-family writes) carry the outage honesty (four machine disruptions
this era; the recovery lesson). No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e192_all_ray_terrain.py    (E192_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1/opt1b/opt1c/e191)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1/opt1b/opt1c/e191-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E192_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e192 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1..e191's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E191_DIR_CK = CKPT_DIR / "e191_static_dir_u.pt"     # the committed g-ray direction
E191_METRICS = E43.REPO / "runs" / "e191" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
OPT1B_METRICS = E43.REPO / "runs" / "opt1b" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the map + the run envelope (dispatch-frozen) -------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1*/e191)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; no profile arm has an lr)
STEP_SCALE_COMMITTED = 1.6542880535125732   # opt1 A0's committed step-1 L2 (t=0 gate ref + rider scale)
A0_S1_GM12_COMMITTED = 0.6780440807342529   # opt1 A0's committed step-1 g-12 (the R2 crosscheck target)
CE1_COMMITTED = 1.356567621231079           # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253          # opt1 A0's committed step-1 pre-clip grad norm
E191_DIR_MD5 = "abb6df7ceefdd048b300c938123d964d"     # e191's committed direction_md5
E191_THETA0_MD5 = "36ae244756475dda71266346e7a0f196"  # e191's checkpointed root flat md5
E191_ENDPOINT_GM12 = 0.0007279703859239817  # e191's committed read at the step-1 endpoint (rider anchor)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — the kill convention everywhere
D_GRID = (0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.92, 1.0, 1.17,
          1.5, 2.0, 2.5, 3.0, 4.0)          # the dispatch-frozen 15-point grid
E191_GRID = (0.05, 0.1, 0.2, 0.33, 0.5, 0.66,
             0.8, 0.92, 1.0, 1.17, 1.5, 2.0)  # e191's committed 12 (the R1 crosscheck set)
SMALL_D_PUMP_BAND = (0.05, 0.1, 0.2, 0.33, 0.5)   # "small D" for the pump-ridge clause
PUMP_RIDGE_MARGIN = 0.01          # a ridge = a rise >= a quarter of the g-ray's own +0.040
SIGN_BAND = (2.0, 3.2)            # the sign-ray stretch band (dispatch letter)
RANDOM_KILL_FLOOR = 3.68          # 4 x 0.92 (the committed e191 cliff — frozen, not re-measured)
RAND_SEEDS = (11501, 11502, 11503)  # fresh (g3K used 11401-3/11411-13; lineage 10902/10901/170/1337/24301/26502)
RIDER_N, RIDER_EVERY = 300, 30    # the pinned walk (delta = STEP_SCALE/300 * u, u frozen)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
DIR_COS_TOL = 1e-6                # fp64 direction tolerance (bit-grade)
A0_SIGNRAY_TOL = 0.05             # FROZEN: the sign-ray-at-1.6543 read vs A0's committed 0.678
E191_S1_D = 1.6542682647705078    # opt1c's committed step-1 cumulative displacement (context)
if SMOKE:                         # true shakedown trims (documented in deviations)
    D_GRID = (0.1, 0.33, 1.0)
    RAND_SEEDS = (11501,)
    RIDER_EVERY = 150

# ---- gates / references (full precision, = stored metrics; opt1c/e191's set) -----
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222770098,
    "dall_g0": 0.9047248959541321,
}
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; opt1's embedded set)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
}

REGISTERED_BARS = {
    "TERRAIN_ONE_PICTURE": "TERRAIN-ONE-PICTURE: \"sign-ray cliffs in the "
        "2.0-3.2 stretch band; random rays kill wide >= 4x the g-ray with "
        "NO pump ridge at small D — Fig-5 licensed as one picture; the pump "
        "is gradient-specific\"",
    "BANDS_REORDER": "BANDS-REORDER: \"the order g < sign < random fails or "
        "random rays pump — the central figure dies here at the cheapest "
        "price; report verbatim\"",
    "RIDER_REORIENTATION_CAUSAL": "RIDER-REORIENTATION-CAUSAL: \"pinned walk "
        "dies near ~0.92\"",
    "RIDER_STEPSIZE_OWNS": "RIDER-STEPSIZE-OWNS: \"pinned walk survives past "
        "1.6543\"",
    "GRADED": "GRADED: any partial pattern — everything reported verbatim as "
        "measured (the lab's standing GRADED floor).",
    "interpretations_frozen": {
        "terrain_one_picture": "Fig-5 is licensed as ONE picture, the pump is "
            "gradient-specific, and W025's subspace picture gains its "
            "ordering on one organism (R59's forced-experiment letter).",
        "bands_reorder": "the terrain story as drawn dies inside the lab "
            "instead of in review — the cheapest possible death for the "
            "paper's central figure (R59's forced-experiment letter).",
        "rider_reorientation_causal": "'small steps protect BY re-orienting' "
            "becomes interventional (size held small, orientation denied — "
            "death); the e188 service done to W026's mechanism noun.",
        "rider_stepsize_owns": "step size or adaptive rescaling owns the "
            "protection and 're-orientation' was a passenger.",
    },
    "registration": "the dispatch's registration IS the registration (the "
        "five bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "RECOVERY (second dispatch): the first e192 executor was killed by a "
    "machine disruption before writing any artifact; this build is fresh "
    "from the frozen dispatch brief + scratch/r59_critic.md (forced "
    "experiment). Bars were frozen in this file before any compute.",
    "R1 re-reads the g-ray fresh (15 Ds) rather than reusing e191's 12 rows "
    "verbatim, because (a) dual currency requires rows e191 never wrote "
    "(per-coordinate RMS), (b) the grid extends to 2.5/3.0/4.0, and (c) the "
    "12 shared Ds then serve as a fresh-vs-committed machinery crosscheck "
    "(G_R1_XCHK) — the reuse is the crosscheck, per the dispatch's "
    "'load e191_static_dir_u.pt, bit-gate'.",
    "The A0 sign-ray crosscheck is TWO anchors: (i) G_A0_XC.endpoint — a "
    "static placement at A0's EXACT fresh-AdamW step-1 endpoint must "
    "reproduce 0.6780440807342529 within 1e-3 (HARD machinery gate); (ii) "
    "G_A0_XC.signray — the pure static sign-ray read at D 1.6542880535125732 "
    f"vs the same committed value within {A0_SIGNRAY_TOL} (FROZEN, recorded "
    "verbatim; failure does NOT abort the map — it flags R2's "
    "step-1-equivalence claim, itself a reportable outcome). The residual "
    "between the two anchors is Adam's weight-decay component, measured and "
    "reported (expected ~0.008 L2, ~0.5% of D).",
    "Random rays: g3K's per-tensor Gaussian draw convention executed in "
    "flat_params (net.parameters()) order so the rays live in the same "
    "coordinates as u; seeds 11501-3 are fresh vs every registered seed "
    "(g3K's isotropic 11401-3/11411-13; lineage 10902/10901/170/1337/24301/"
    "26502). Even-spread co-read (per-block L2 share vs numel share) "
    "reported per ray — guaranteed only in expectation, never gated.",
    "The rider's parameter points coincide with the static g-ray's points "
    "BY CONSTRUCTION (straight walk down the frozen ray; disclosed in the "
    "docstring before compute): the rider's content is the interventional "
    "denial of re-orientation at small step size, plus a finer battery "
    "trace of the cliff (reads at D 0.827 and 0.9926 resolve inside the "
    "committed [0.80, 0.92] bracket's neighborhood), plus a free placement "
    "check (step-300 read vs e191's committed endpoint read).",
    "Pump-ridge margin 0.01 frozen (= a quarter of the g-ray's committed "
    "rise +0.040): the bar's own word is 'ridge'; strict single-point rises "
    "(any small-D read > root) are co-reported per ray so the clause cannot "
    "hide behind the margin.",
    "Smoke mode trims: D grid {0.1, 0.33, 1.0}, one random ray (11501), "
    "rider reads every 150 steps; nothing adjudicated (verdict stamped "
    "SMOKE).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e191_pump_cliff.py VERBATIM (whose own provenance is
# lab/opt1c_direction_size.py via lab/e185_noise_wash.py — the e176n lineage).
# Copied rather than imported to own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
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


def flat_md5(net: TinyGPT) -> str:
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 cosine — the truthful estimator on 2.7M elements (e191 texture)."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64)))


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e192_smoke" if SMOKE else "e192")
    log(f"E192 THE ONE-ORGANISM ALL-RAY TERRAIN MAP (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), EVAL-ONLY, "
        f"{len(D_GRID)} graded static jumps x {2 + len(RAND_SEEDS)} rays + "
        f"the pinned rider ({RIDER_N} steps), n=1, seed lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (E191_METRICS, OPT1_METRICS, OPT1B_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    if not E191_DIR_CK.exists():
        raise RuntimeError(f"missing e191 direction checkpoint: {E191_DIR_CK}")
    e191m = json.loads(E191_METRICS.read_text(encoding="utf-8"))
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1bm = json.loads(OPT1B_METRICS.read_text(encoding="utf-8"))
    a0 = opt1m["arms"]["a0_adamw_ref"]
    a0_s1 = next(r for r in a0["traj"] if r["step"] == 1)
    assert abs(a0_s1["cum_disp"] - STEP_SCALE_COMMITTED) < 1e-12, \
        "opt1's committed A0 step-1 displacement drifted vs this file's copy"
    assert abs(a0_s1["ce_batch"] - CE1_COMMITTED) < 1e-12
    assert abs(a0_s1["preclip_gnorm"] - GN1_COMMITTED) < 1e-12
    assert abs(a0_s1["gm12"] - A0_S1_GM12_COMMITTED) < 1e-12
    e191_prof = {round(r["D"], 9): r for r in e191m["profile"] if r["D"] > 0}
    e191_cliff = e191m["adjudication"]["cliff_edge"]
    bleed_curve = [{"step": r["step"], "D": r["D"], "gm12": r["gm12"]}
                   for r in opt1bm["fact_vs_D_curve"]]
    assert e191m["provenance"]["direction_md5"] == E191_DIR_MD5, \
        "e191's committed direction md5 drifted vs this file's copy"
    log(f"parents: e191 {e191m['adjudication']['verdict']} (cliff bracket "
        f"{e191_cliff['bracket_D']}, {len(e191_prof)} committed profile "
        f"rows); opt1 {opt1m['adjudication']['verdict']} (A0 step-1 gm12 "
        f"{A0_S1_GM12_COMMITTED:.4f} @ L2 {STEP_SCALE_COMMITTED:.4f}); "
        f"opt1b bleed {len(bleed_curve)} rows — committed, none rerun")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185/opt1*/e191)
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
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix} (SPLICE_RNG {E43.SPLICE_RNG})")

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids = bat_ids[-12]
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at ctx "
                "offsets {-12,0,+12}, e185's convention verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- the neutral stream (e170 VERBATIM via e185/opt1*/e191)
    arng = _random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
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
                         "170, rejection if [s, s+257) contains "
                         "FLORIZEL/ELIZABETH/ZEPH/MIRABEL — e170 VERBATIM"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], f"neutral bank drifted: {n_starts}"
    log("G_ANCHOR: neutral bank bit-matches e185's stored 16 starts: PASS")

    # ---------------- root net + gate vs e151 (opt1*/e191's gate set verbatim)
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

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
    rdiffs = {k: root_cells[k] - root_refs[k] for k in root_refs}
    rmax = max(abs(v) for v in rdiffs.values())
    G_ROOT = {"cells": root_cells, "refs": root_refs, "diffs": rdiffs,
              "max_abs_diff": rmax, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax < G_BIT_TOL),
              "pass": bool(rmax < G_FALLBACK_TOL)}
    log(f"G_ROOT (vs e151 before-cells, light set): max|diff| {rmax:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")

    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    RMS_DEN = float(torch.sqrt(torch.tensor(float(N_PARAM), dtype=torch.float64)))
    n_anc = anchor_neutral.shape[0]

    def draw_step1_batch():
        """The step-1 batch, bit-identical to the licensed stream."""
        g = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(n_anc, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    # =====================================================================
    # THE T=0 BIT-IDENTITY GATE (opt1c/e191 VERBATIM; theta_1 kept for G_A0_XC)
    # =====================================================================
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
    d_ce, d_gn, d_disp = (abs(ce1 - CE1_COMMITTED), abs(gn1 - GN1_COMMITTED),
                          abs(disp1 - STEP_SCALE_COMMITTED))
    G_T0 = {
        "step1_x_md5": x1_md5, "step1_x_md5_match_e185": bool(
            x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1, "ce_batch_committed": CE1_COMMITTED,
        "preclip_gnorm_measured": gn1, "preclip_gnorm_committed": GN1_COMMITTED,
        "adamw_step1_L2_measured": disp1,
        "adamw_step1_L2_committed": STEP_SCALE_COMMITTED,
        "diffs": {"ce": d_ce, "gnorm": d_gn, "disp": d_disp},
        "max_abs_diff": max(d_ce, d_gn, d_disp),
        "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
        "bit": bool(max(d_ce, d_gn, d_disp) < G_BIT_TOL),
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and max(d_ce, d_gn, d_disp) < G_FALLBACK_TOL),
        "note": "PRE-DISPATCH CHECK (Rule 12; opt1c/e191's gate VERBATIM): "
                "the step-1 batch md5; the forward CE and pre-clip grad norm "
                "vs opt1's committed values; ONE fresh AdamW-recipe step "
                "from the root reproducing A0's committed step-1 L2 — the "
                "gradient provenance gate. theta_1 from this step is KEPT "
                "as the exact A0 step-1 endpoint for G_A0_XC.",
    }
    log(f"G_T0 (t=0 bit-identity vs opt1/e191 committed): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; "
        f"|dCE| {d_ce:.2e} |dgn| {d_gn:.2e} |dDisp| {d_disp:.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL")
        + (" (bit)" if G_T0["bit"] else ""))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — abort (control "
                           "failure)")

    # ---- the fresh t=0 gradient (for G_U_CK's recompute + the sign ray)
    gnet = copy.deepcopy(net0)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1)
    loss_g = F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                             y1.reshape(-1))
    assert abs(float(loss_g.item()) - ce1) < 1e-9, "CE drifted between gates"
    loss_g.backward()
    g0 = torch.cat([p.grad.detach().reshape(-1) for p in gnet.parameters()])
    u_fresh = (g0 / torch.norm(g0)).clone()
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g, loss_g

    # ---- G_U_CK: the e191 direction bit-gate (the dispatch's R1 gate)
    uck = torch.load(E191_DIR_CK, map_location="cpu", weights_only=False)
    u_g = uck["u"].detach().clone().float()
    u_g_md5 = hashlib.md5(u_g.numpy().tobytes()).hexdigest()
    theta0_md5 = flat_md5(net0)
    uck_meta = E43.jsonable(uck.get("meta", {}))
    fresh_md5 = hashlib.md5(u_fresh.numpy().tobytes()).hexdigest()
    c64 = cos64(u_fresh, u_g)
    G_U_CK = {
        "loaded_u_md5": u_g_md5, "committed_e191_md5": E191_DIR_MD5,
        "md5_match_committed": bool(u_g_md5 == E191_DIR_MD5),
        "loaded_u_norm": float(torch.norm(u_g)),
        "loaded_u_numel": int(u_g.numel()), "expected_numel": N_PARAM,
        "root_flat_md5": theta0_md5,
        "checkpoint_theta0_md5": uck.get("theta0_md5"),
        "root_md5_match_checkpoint": bool(theta0_md5 == uck.get("theta0_md5")),
        "fresh_recompute_md5": fresh_md5,
        "fresh_md5_match_loaded": bool(fresh_md5 == u_g_md5),
        "fresh_cos64_loaded": c64,
        "cos_tol": DIR_COS_TOL,
        "recompute_bit": bool(fresh_md5 == u_g_md5 and c64 > 1 - DIR_COS_TOL),
        "pass": bool(u_g_md5 == E191_DIR_MD5 and u_g.numel() == N_PARAM
                     and abs(float(torch.norm(u_g)) - 1.0) < 1e-5
                     and theta0_md5 == uck.get("theta0_md5")
                     and (fresh_md5 == u_g_md5 or c64 > 1 - DIR_COS_TOL)),
        "checkpoint_meta": uck_meta,
        "note": "PRE-DISPATCH CHECK (Rule 12; the dispatch's R1 bit-gate): "
                "the loaded u is e191's committed direction (md5 vs e191 "
                "metrics), on this root (flat md5 vs the checkpoint's "
                "theta0_md5), and a FRESH t=0 recompute agrees — md5-"
                "identical with the fp64-cosine fallback honoring the fp32 "
                "reduction-order texture (e191's own disclosure)",
    }
    log(f"G_U_CK (e191 direction bit-gate): loaded md5 "
        f"{'==' if G_U_CK['md5_match_committed'] else '!='} committed; root "
        f"md5 {'==' if G_U_CK['root_md5_match_checkpoint'] else '!='} ckpt; "
        f"fresh recompute {'md5-identical' if G_U_CK['fresh_md5_match_loaded'] else 'cos ' + format(c64, '.12f')}: "
        + ("PASS" if G_U_CK["pass"] else "FAIL"))
    if not G_U_CK["pass"]:
        raise RuntimeError("e191 direction bit-gate FAILED — R1 would not be "
                           "the committed g-ray; abort")

    # ---- R2's direction: the STATIC sign ray (sign(g_0), normalized)
    s_raw = torch.sign(g0)
    n_zero_g = int((g0 == 0).sum())
    u_sign = (s_raw / torch.norm(s_raw)).clone()
    u_sign_md5 = hashlib.md5(u_sign.numpy().tobytes()).hexdigest()
    del g0, s_raw

    # ---- G_A0_XC part 1 (HARD): battery at A0's EXACT step-1 endpoint
    evl = copy.deepcopy(net0)
    evl.eval()
    load_flat(evl, theta1_adam)
    ep_read = battery_cell(evl, gm12_ids, zid)
    d_ep = abs(ep_read["mean_pz"] - A0_S1_GM12_COMMITTED)

    # the geometric quantification of the sign-ray vs endpoint residual
    d64 = (theta1_adam - theta0).to(torch.float64)
    us64 = u_sign.to(torch.float64)
    r_sign = float(-torch.dot(d64, us64))          # projection onto -u_sign
    w64 = d64 + r_sign * us64                      # the off-sign residual (wd)
    a0_geom = {
        "cos_delta_a0_neg_usign": cos64(theta1_adam - theta0, -u_sign),
        "proj_len_onto_neg_usign": r_sign,
        "off_sign_residual_L2": float(torch.norm(w64)),
        "lr_times_sqrtN": LR_ADAMW * float(RMS_DEN),  # the pure-sign step L2
        "delta_a0_L2_measured": disp1,
        "n_zero_g_coords": n_zero_g,
    }
    G_A0_XC_ENDPOINT = {
        "read_gm12": ep_read["mean_pz"], "frac_argmax_z": ep_read["frac_argmax_z"],
        "committed_gm12": A0_S1_GM12_COMMITTED, "d_gm12": d_ep,
        "tol": 1e-3, "pass": bool(d_ep < 1e-3),
        "note": "PRE-DISPATCH CHECK, HARD: a static placement at A0's EXACT "
                "fresh-AdamW step-1 endpoint must reproduce A0's committed "
                "step-1 g-12 — the machinery anchor for the sign-ray "
                "crosscheck; failure = control failure, abort",
        "a0_geometry": a0_geom,
    }
    log(f"G_A0_XC.endpoint (A0 exact step-1 endpoint): g-12 "
        f"{ep_read['mean_pz']:.6f} vs committed {A0_S1_GM12_COMMITTED:.6f} "
        f"(|d| {d_ep:.2e}): " + ("PASS" if G_A0_XC_ENDPOINT["pass"] else "FAIL")
        + f" [sign-proj {r_sign:.6f}, off-sign residual L2 "
        f"{a0_geom['off_sign_residual_L2']:.6f}, lr*sqrt(N) "
        f"{a0_geom['lr_times_sqrtN']:.6f}]")
    if not G_A0_XC_ENDPOINT["pass"]:
        raise RuntimeError("A0 endpoint crosscheck FAILED — the battery "
                           "machinery cannot reproduce the committed read; "
                           "abort (control failure)")

    log("WHAT THESE RAYS GUARANTEE: NOTHING — each ray guarantees only its "
        "geometry (unit L2 from theta_0 along its direction family); "
        "whether it pumps, spares, or kills at any D is the measurement; "
        "that openness is the point.")

    # ---- PROGRESSIVE PARTIAL WRITE #1 (the outage lesson; 4 disruptions this era)
    def write_partial(status: str, extra: dict | None = None) -> None:
        payload = {
            "experiment": "e192_all_ray_terrain", "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"rays_done": len(ray_state), "grid_total": len(D_GRID),
                      "rider_done": rider_state["done"],
                      "elapsed_s": round(time.time() - T0, 1)},
            "gates_partial": {"G_T0": G_T0, "G_U_CK": G_U_CK,
                              "G_A0_XC_ENDPOINT": G_A0_XC_ENDPOINT},
            "provenance_partial": {
                "N_param": N_PARAM, "rms_denominator": RMS_DEN,
                "u_g_md5": u_g_md5, "u_sign_md5": u_sign_md5,
                "random_seeds": list(RAND_SEEDS)},
            "rays_partial": {k: v["rows"] for k, v in ray_state.items()},
            "rider_partial": rider_state["rows"],
        }
        if extra:
            payload.update(extra)
        save_json(rd / "metrics.json", E43.jsonable(payload))

    # =====================================================================
    # THE RAY FAMILIES (perturb-and-eval; progressive writes per grid point)
    # =====================================================================
    def static_jump_read(u_vec: torch.Tensor, D: float) -> dict:
        """theta_D = theta_0 - D*u; perturb-and-eval; the dual-currency row."""
        thD = theta0 - D * u_vec
        disp_check = float(torch.norm(thD - theta0))
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        ce_r = ce_fixed_cpu(evl, *r_eval_xy)
        return {"D": float(D), "rms": float(D) / RMS_DEN,
                "gm12": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
                "ce_r": ce_r, "disp_check": disp_check,
                "disp_dev": abs(disp_check - float(D))}

    def draw_gaussian_ray(seed: int) -> tuple[torch.Tensor, dict]:
        """g3K's per-tensor draw convention, in flat_params order."""
        g = torch.Generator().manual_seed(seed)
        blocks = [torch.randn(p.shape, generator=g).reshape(-1)
                  for p in net0.parameters()]
        v = torch.cat(blocks)
        sq = v * v
        tot = float(torch.sum(sq))
        spreads = {}
        i = 0
        for (nm, p) in net0.named_parameters():
            n = p.numel()
            spreads[nm] = {"numel_share": n / N_PARAM,
                           "l2_share": float(torch.sum(sq[i:i + n])) / tot}
            i += n
        max_spread = max(abs(s["l2_share"] - s["numel_share"])
                         for s in spreads.values())
        return (v / torch.norm(v)).clone(), {
            "seed": seed, "draw": "per-tensor torch.randn(shape, generator) "
            "in net.parameters() order (g3K convention, flat coordinates)",
            "even_spread": {"per_block": spreads,
                            "max_abs_share_dev": max_spread,
                            "note": "co-read only — even spread guaranteed "
                                    "in expectation, never gated"},
        }

    rays: list[dict] = [
        {"key": "R1_G", "label": "g-ray (raw wash gradient)",
         "u": u_g, "prov": "runs/checkpoints/e191_static_dir_u.pt "
         "(bit-gated, G_U_CK)", "rows": []},
        {"key": "R2_SIGN", "label": "static sign(g_0) ray, normalized",
         "u": u_sign, "prov": "sign(g_0)/||sign(g_0)||, g_0 the fresh t=0 "
         "post-clip gradient (clip-invariant; n_zero_g_coords "
         f"{n_zero_g})", "rows": []},
    ]
    for sd in RAND_SEEDS:
        uv, meta = draw_gaussian_ray(sd)
        rays.append({"key": f"R{len(rays) + 1}_GAUSS_s{sd}",
                     "label": f"Gaussian unit ray seed {sd}",
                     "u": uv, "prov": meta, "rows": []})
    ray_state = {r["key"]: r for r in rays}
    rider_state = {"done": False, "rows": []}
    write_partial("PARTIAL — all pre-dispatch gates PASSED (t=0 bit + e191 "
                  "direction bit + A0 endpoint); rays starting")
    log("[partial] metrics.json written (gates passed)")

    journal: list[dict] = []

    # ---- R1 first (its shared Ds are the G_R1_XCHK machinery gate)
    for r in rays:
        for D in D_GRID:
            row = static_jump_read(r["u"], D)
            r["rows"].append(row)
            journal.append({"ray": r["key"], "kind": "grid", **row})
            log(f"  {r['key']:>14s} D {D:5.3f} (rms {row['rms']:.2e}): "
                f"g-12 {row['gm12']:.4f} CE_R {row['ce_r']:.4f} "
                f"[disp dev {row['disp_dev']:.1e}]")
            write_partial(f"PARTIAL: {r['key']} "
                          f"{len(r['rows'])}/{len(D_GRID)} grid points done")
        if r["key"] == "R1_G":
            # G_R1_XCHK: fresh reads vs e191's committed 12
            xrows = []
            for row in r["rows"]:
                cd = e191_prof.get(round(row["D"], 9))
                if cd is not None:
                    xrows.append({"D": row["D"], "fresh_gm12": row["gm12"],
                                  "committed_gm12": cd["gm12"],
                                  "d_gm12": abs(row["gm12"] - cd["gm12"]),
                                  "fresh_ce_r": row["ce_r"],
                                  "committed_ce_r": cd.get("ce_r")})
            xc_max = max(x["d_gm12"] for x in xrows)
            G_R1_XCHK = {
                "n_points": len(xrows), "max_abs_d_gm12": xc_max,
                "tol": 1e-3, "pass": bool(xc_max < 1e-3),
                "rows": xrows,
                "note": "PRE-DISPATCH CHECK (Rule 12): the fresh R1 reads "
                        "at the 12 Ds shared with e191's committed grid must "
                        "reproduce the committed profile — the dispatch's "
                        "'reuse the committed g-ray' as a bit-close "
                        "machinery crosscheck; HARD",
            }
            log(f"G_R1_XCHK (fresh g-ray vs e191 committed, {len(xrows)} "
                f"shared Ds): max|dgm12| {xc_max:.2e}: "
                + ("PASS" if G_R1_XCHK["pass"] else "FAIL"))
            if not G_R1_XCHK["pass"]:
                write_partial("FAILED GATE — G_R1_XCHK: fresh g-ray reads "
                              "drifted vs e191 committed profile",
                              {"failed_gate": G_R1_XCHK})
                raise RuntimeError("G_R1_XCHK FAILED — abort (control "
                                   "failure)")
            write_partial(f"PARTIAL: G_R1_XCHK PASS (max|d| {xc_max:.1e}); "
                          "R1 complete", {"gates_r1_xchk": G_R1_XCHK})
        if r["key"] == "R2_SIGN":
            # the dispatch's A0 crosscheck point (frozen tol, recorded)
            row = static_jump_read(u_sign, STEP_SCALE_COMMITTED)
            d_sr = abs(row["gm12"] - A0_S1_GM12_COMMITTED)
            G_A0_XC_SIGNRAY = {
                "D": STEP_SCALE_COMMITTED, "rms": STEP_SCALE_COMMITTED / RMS_DEN,
                "read_gm12": row["gm12"], "frac_argmax_z": row["frac_argmax_z"],
                "ce_r": row["ce_r"],
                "committed_gm12": A0_S1_GM12_COMMITTED, "d_gm12": d_sr,
                "tol": A0_SIGNRAY_TOL, "pass": bool(d_sr < A0_SIGNRAY_TOL),
                "note": "PRE-DISPATCH CHECK, FROZEN BEFORE COMPUTE (the "
                        "dispatch's letter): a single STATIC sign-ray jump "
                        "at A0's step-1 D 1.6543 must reproduce A0's "
                        "committed step-1 read ~0.678; the residual vs the "
                        "exact endpoint is Adam's weight-decay component "
                        f"(measured off-sign residual L2 "
                        f"{a0_geom['off_sign_residual_L2']:.6f}); failure "
                        "does NOT abort — it flags R2's step-1-equivalence "
                        "and is reported verbatim",
            }
            journal.append({"ray": "R2_SIGN", "kind": "a0_crosscheck",
                            **row})
            log(f"G_A0_XC.signray (sign-ray @ D {STEP_SCALE_COMMITTED:.4f}): "
                f"g-12 {row['gm12']:.6f} vs committed "
                f"{A0_S1_GM12_COMMITTED:.6f} (|d| {d_sr:.4f}, tol "
                f"{A0_SIGNRAY_TOL}): "
                + ("PASS" if G_A0_XC_SIGNRAY["pass"] else "FLAGGED (verbatim)"))
            r["a0_crosscheck"] = G_A0_XC_SIGNRAY
            write_partial(f"PARTIAL: R2 complete + A0 sign-ray crosscheck "
                          f"(|d| {d_sr:.4f})",
                          {"gates_a0_signray": G_A0_XC_SIGNRAY})

    # ---- the ray-family summaries (kill Ds, pumps — dual currency)
    def ray_summary(r: dict) -> dict:
        rows = r["rows"]
        dead = [row for row in rows if row["gm12"] <= SHUT_BAR]
        first_dead = dead[0] if dead else None
        small = [row for row in rows if row["D"] in SMALL_D_PUMP_BAND]
        max_rise = max((row["gm12"] - root_cells["gm12"]) for row in small)
        strict_rise = any(row["gm12"] > root_cells["gm12"] for row in small)
        return {
            "key": r["key"], "label": r["label"],
            "first_dead_D": first_dead["D"] if first_dead else None,
            "first_dead_rms": (first_dead["rms"] if first_dead else None),
            "unresolved_high": first_dead is None,
            "kill_note": (f"first grid D with g-12 <= {SHUT_BAR}" if first_dead
                          else f"no grid D reads <= {SHUT_BAR} (kill > "
                               f"{D_GRID[-1]}, unresolved-high)"),
            "small_D_max_rise": max_rise,
            "small_D_strict_rise": strict_rise,
            "pump_ridge": bool(max_rise > PUMP_RIDGE_MARGIN),
            "pump_ridge_margin": PUMP_RIDGE_MARGIN,
        }

    summaries = [ray_summary(r) for r in rays]
    for s in summaries:
        log(f"  SUMMARY {s['key']:>14s}: kill D "
            f"{s['first_dead_D'] if s['first_dead_D'] is not None else '>4.0'}"
            f" (rms {s['first_dead_rms'] if s['first_dead_rms'] is not None else float('nan'):.2e}); "
            f"small-D max rise {s['small_D_max_rise']:+.4f} "
            f"(ridge {'YES' if s['pump_ridge'] else 'no'}; strict rise "
            f"{'YES' if s['small_D_strict_rise'] else 'no'})")
    write_partial("PARTIAL: all ray families complete + summarized",
                  {"ray_summaries": summaries})

    # =====================================================================
    # THE RIDER — 300 pinned steps down the FIXED g-ray (no re-orientation)
    # =====================================================================
    step_vec = (STEP_SCALE_COMMITTED / RIDER_N) * u_g
    th_r = theta0.clone()
    evl_r = copy.deepcopy(net0)
    evl_r.eval()
    for k in range(1, RIDER_N + 1):
        th_r.sub_(step_vec)
        if k % RIDER_EVERY == 0 or k == RIDER_N:
            disp_k = float(torch.norm(th_r - theta0))
            load_flat(evl_r, th_r)
            gz = battery_cell(evl_r, gm12_ids, zid)
            ce_k = ce_fixed_cpu(evl_r, *r_eval_xy)
            row = {"step": k, "D_cum": disp_k, "rms": disp_k / RMS_DEN,
                   "gm12": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
                   "ce_r": ce_k,
                   "disp_dev": abs(disp_k - k * (STEP_SCALE_COMMITTED / RIDER_N))}
            rider_state["rows"].append(row)
            journal.append({"ray": "RIDER", "kind": "rider", **row})
            log(f"  RIDER step {k:3d} (D {row['D_cum']:.4f}, rms "
                f"{row['rms']:.2e}): g-12 {row['gm12']:.4f} "
                f"CE_R {row['ce_r']:.4f}")
    rider_state["done"] = True
    rider_last = rider_state["rows"][-1]
    G_RIDER_XC = {
        "step": rider_last["step"], "D_cum": rider_last["D_cum"],
        "read_gm12": rider_last["gm12"],
        "committed_e191_endpoint_gm12": E191_ENDPOINT_GM12,
        "d_gm12": abs(rider_last["gm12"] - E191_ENDPOINT_GM12),
        "tol": 1e-3, "pass": bool(abs(rider_last["gm12"] - E191_ENDPOINT_GM12) < 1e-3),
        "note": "the pinned walk's step-300 point (D 1.6543 on the frozen "
                "ray, accumulated by 300 incremental adds) must read as "
                "e191's committed step-1-endpoint read — placement/"
                "accumulation verification, never an adjudication input",
    }
    log(f"G_RIDER_XC (step-300 vs e191 committed endpoint read): "
        f"{rider_last['gm12']:.6f} vs {E191_ENDPOINT_GM12:.6f} "
        f"(|d| {G_RIDER_XC['d_gm12']:.2e}): "
        + ("PASS" if G_RIDER_XC["pass"] else "FLAGGED (verbatim)"))
    write_partial("PARTIAL: rider complete", {"gates_rider_xc": G_RIDER_XC})

    # =====================================================================
    # ADJUDICATION (registered clauses; frozen operationalizations; no shopping)
    # =====================================================================
    summ = {s["key"]: s for s in summaries}
    g_s, sign_s = summ["R1_G"], summ["R2_SIGN"]
    rand_s = [s for s in summaries if "_GAUSS_" in s["key"]]
    g_kill = g_s["first_dead_D"]
    sign_kill = sign_s["first_dead_D"]
    rand_kills = [s["first_dead_D"] for s in rand_s]
    rand_eff = (min(k for k in rand_kills if k is not None)
                if any(k is not None for k in rand_kills) else float("inf"))
    all_rand_wide = all((s["first_dead_D"] is None)
                        or (s["first_dead_D"] >= RANDOM_KILL_FLOOR)
                        for s in rand_s)
    any_rand_ridge = any(s["pump_ridge"] for s in rand_s)
    any_rand_strict = any(s["small_D_strict_rise"] for s in rand_s)
    sign_in_band = bool(sign_kill is not None
                        and SIGN_BAND[0] <= sign_kill <= SIGN_BAND[1])
    order_holds = bool(g_kill is not None and sign_kill is not None
                       and g_kill < sign_kill and sign_kill < rand_eff)

    terrain_fires = bool(sign_in_band and all_rand_wide and not any_rand_ridge)
    reorder_fires = bool((not order_holds) or any_rand_ridge)
    assert not (terrain_fires and reorder_fires), "bar construction overlap"

    # rider bars
    r_dead = [row for row in rider_state["rows"] if row["gm12"] <= SHUT_BAR]
    r_first = r_dead[0] if r_dead else None
    rider_causal_fires = bool(r_first is not None
                              and r_first["step"] <= 6 * RIDER_EVERY)
    rider_stepsize_fires = bool(r_first is None)

    kill_ratios = {
        "sign_over_g": (sign_kill / g_kill if (sign_kill and g_kill) else None),
        "rand_min_over_g": ((rand_eff / g_kill) if (rand_eff != float("inf")
                                                    and g_kill) else ">4.35 (unresolved-high)"),
    }

    if SMOKE:
        verdict_map = "SMOKE (nothing adjudicated)"
        clause_map = "shakedown only"
        verdict_rider = "SMOKE (nothing adjudicated)"
        clause_rider = "shakedown only"
        bars = {k: {"fires": False} for k in
                ("TERRAIN_ONE_PICTURE", "BANDS_REORDER",
                 "RIDER_REORIENTATION_CAUSAL", "RIDER_STEPSIZE_OWNS",
                 "GRADED")}
    else:
        # ---- the map verdict (composite: TERRAIN-ONE-PICTURE -> BANDS-REORDER -> GRADED)
        if terrain_fires:
            verdict_map = "TERRAIN-ONE-PICTURE"
            clause_map = (
                f"the sign-ray cliffs inside the stretch band (first dead "
                f"grid D {sign_kill} -> rms {sign_kill / RMS_DEN:.2e}; "
                f"{kill_ratios['sign_over_g']:.2f}x the g-ray's "
                f"{g_kill}); every random ray kills wide ("
                + "; ".join(f"{s['key']}: "
                            + (f"D {s['first_dead_D']} ({s['first_dead_D'] / g_kill:.2f}x)"
                               if s["first_dead_D"] is not None
                               else f"> 4.0, unresolved-high (> 4.35x)")
                            for s in rand_s)
                + f") with NO pump ridge at small D (max rises "
                + "; ".join(f"{s['key']}: {s['small_D_max_rise']:+.4f}"
                            for s in rand_s)
                + f" vs margin {PUMP_RIDGE_MARGIN}; strict rises "
                + ("present" if any_rand_strict else "absent")
                + f" — co-reported). Fig-5 IS LICENSED AS ONE PICTURE — one "
                f"organism, one ruler, one currency pair — and THE PUMP IS "
                f"GRADIENT-SPECIFIC (the g-ray's committed ridge +0.040 has "
                f"no random-ray counterpart). The order g < sign < random "
                f"stands on mapped ground: {g_kill} < {sign_kill} < "
                + (f"{rand_eff}" if rand_eff != float("inf") else "> 4.0")
                + ". The middle and third bands stop being a stitch.")
        elif reorder_fires:
            verdict_map = "BANDS-REORDER"
            why = []
            if not order_holds:
                why.append(f"the order g < sign < random FAILS as measured "
                           f"(g kill {g_kill}; sign kill "
                           + (f"{sign_kill}" if sign_kill is not None
                              else "UNRESOLVED-HIGH at the 4.0 grid cap — "
                                   "the order is not established")
                           + "; random min kill "
                           + (f"{rand_eff}" if rand_eff != float("inf")
                              else "> 4.0 unresolved")
                           + ")")
            if any_rand_ridge:
                why.append("RANDOM RAYS PUMP at small D ("
                           + "; ".join(f"{s['key']}: max rise "
                                       f"{s['small_D_max_rise']:+.4f}"
                                       for s in rand_s if s["pump_ridge"])
                           + f" vs margin {PUMP_RIDGE_MARGIN} — the pump is "
                           "a generic small-displacement effect, not the "
                           "gradient's fact-positivity)")
            clause_map = (" and ".join(why)
                          + ". THE CENTRAL FIGURE DIES HERE, INSIDE THE LAB, "
                          "AT THE CHEAPEST PRICE — the terrain story as "
                          "drawn (three bands of increasing width, the pump "
                          "gradient-specific) does not survive one organism, "
                          "one ruler, one currency. Reported verbatim; the "
                          "full curves are the map.")
        else:
            verdict_map = "GRADED"
            clause_map = (
                f"a partial pattern — neither named map bar's full clause-set "
                f"holds (sign kill "
                + (f"{sign_kill} (band [{SIGN_BAND[0]}, {SIGN_BAND[1]}]: "
                   f"{'IN' if sign_in_band else 'OUT'})" if sign_kill is not None
                   else "unresolved-high")
                + f"; random kills wide: {'all' if all_rand_wide else 'NOT all'}; "
                f"random ridges: {'none' if not any_rand_ridge else 'YES'}; "
                f"order holds: {order_holds}); everything reported verbatim "
                f"as measured — the map is the finding.")
        # ---- the rider verdict (RIDER-REORIENTATION-CAUSAL -> RIDER-STEPSIZE-OWNS -> GRADED)
        if rider_causal_fires:
            prev = next((row for row in reversed(rider_state["rows"])
                         if row["step"] < r_first["step"]), None)
            verdict_rider = "RIDER-REORIENTATION-CAUSAL"
            clause_rider = (
                f"the pinned walk DIES near ~0.92: alive at "
                + (f"step {prev['step']} (D {prev['D_cum']:.4f}, g-12 "
                   f"{prev['gm12']:.4f})" if prev else "the root")
                + f", dead at step {r_first['step']} (D {r_first['D_cum']:.4f}"
                f" -> rms {r_first['rms']:.2e}, g-12 {r_first['gm12']:.4f} "
                f"<= {SHUT_BAR}) — bracket contains the static cliff "
                f"(e191 committed edge {e191_cliff['bracket_D']}). Step "
                f"size was held small ({STEP_SCALE_COMMITTED / RIDER_N:.6f} "
                f"L2/step, the bleed's scale) and orientation DENIED — "
                f"death anyway. 'SMALL STEPS PROTECT BY RE-ORIENTING' IS "
                f"NOW INTERVENTIONAL; the re-orienting bleed's survival at "
                f"this D (opt1b committed, alive 0.79-0.86 at D 0.82-0.99) "
                f"is carried by the re-orientation, not the step size.")
        elif rider_stepsize_fires:
            verdict_rider = "RIDER-STEPSIZE-OWNS"
            clause_rider = (
                f"the pinned walk SURVIVES past 1.6543: no battery read "
                f"through step {RIDER_N} (D "
                f"{rider_state['rows'][-1]['D_cum']:.4f}) falls <= "
                f"{SHUT_BAR} (min read "
                f"{min(row['gm12'] for row in rider_state['rows']):.4f}); "
                f"step size or adaptive rescaling OWNS the sparing and "
                "'re-orientation' was a passenger — the static map's own "
                "cliff is not visited by accumulated small steps on this "
                "build. Reported verbatim (this outcome would contradict "
                "the disclosed point-coincidence — see the float-texture "
                "deviation).")
        else:
            verdict_rider = "GRADED"
            clause_rider = (
                f"the pinned walk dies OUTSIDE the 0.92 neighborhood: first "
                f"dead read at step {r_first['step']} (D "
                f"{r_first['D_cum']:.4f}) — inside (0.9926, 1.6543]; above "
                f"the static cliff edge, below the end; reported verbatim.")
        bars = {
            "TERRAIN_ONE_PICTURE": {"fires": terrain_fires,
                "sign_kill_D": sign_kill, "sign_band": list(SIGN_BAND),
                "sign_in_band": sign_in_band,
                "all_randoms_kill_wide": all_rand_wide,
                "random_kill_floor": RANDOM_KILL_FLOOR,
                "any_random_pump_ridge": any_rand_ridge,
                "any_random_strict_rise": any_rand_strict},
            "BANDS_REORDER": {"fires": reorder_fires,
                "order_g_lt_sign_lt_random_holds": order_holds,
                "g_kill": g_kill, "sign_kill": sign_kill,
                "rand_min_kill": (None if rand_eff == float("inf") else rand_eff),
                "any_random_pump_ridge": any_rand_ridge},
            "RIDER_REORIENTATION_CAUSAL": {"fires": rider_causal_fires,
                "first_dead_step": (r_first["step"] if r_first else None),
                "first_dead_D_cum": (r_first["D_cum"] if r_first else None),
                "clause_D_cap": 6 * RIDER_EVERY * (STEP_SCALE_COMMITTED / RIDER_N)},
            "RIDER_STEPSIZE_OWNS": {"fires": rider_stepsize_fires,
                "min_gm12_over_reads": min(row["gm12"] for row in rider_state["rows"]),
                "final_D_cum": rider_state["rows"][-1]["D_cum"]},
            "GRADED": {"fires": verdict_map == "GRADED"
                                 or verdict_rider == "GRADED"},
        }
    log("=" * 78)
    log(f"E192 MAP VERDICT:    {verdict_map}")
    log(f"  {clause_map}")
    log(f"E192 RIDER VERDICT:  {verdict_rider}")
    log(f"  {clause_rider}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e192_all_ray_terrain",
        "date": common.now_iso(),
        "status": "COMPLETE — adjudicated (this write replaces all PARTIAL "
                  "progressive writes)",
        "recovery_note": ("SECOND DISPATCH: the first e192 executor was "
                          "killed by machine disruption before writing any "
                          "artifact (commit 3152a83); nothing was recovered; "
                          "this build is fresh from the frozen dispatch "
                          "brief + scratch/r59_critic.md's forced experiment; "
                          "bars frozen in lab/e192_all_ray_terrain.py before "
                          "any compute"),
        "registration": ("the dispatch's registration IS the registration "
                         "(the five bars quoted verbatim in the module "
                         "docstring and in registered_bars, frozen before "
                         "compute)"),
        "registered_bars": REGISTERED_BARS,
        "question": ("R59-critic's forced cell: on ONE organism (the e131 "
                     "consolidated root), ONE ruler (the install-60 g-12 "
                     "battery + CE_R), ONE D grid, do STATIC graded jumps "
                     "along the g-ray / the static sign(g_0) ray / 3 fresh "
                     "Gaussian rays tile into the three-band terrain "
                     "(g < sign < random, pump gradient-specific) — and "
                     "does a 300-step pinned walk down the frozen g-ray "
                     "(small steps, orientation denied) die at the static "
                     "cliff (re-orientation causal) or survive (step size "
                     "owns the sparing)?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "rays": [
                {"key": r["key"], "label": r["label"], "provenance": r["prov"],
                 "u_md5": hashlib.md5(r["u"].numpy().tobytes()).hexdigest(),
                 "u_norm": float(torch.norm(r["u"]))}
                for r in rays],
            "D_grid": list(D_GRID),
            "static_jump": "theta_D = theta_0 - D * u_ray; perturb-and-eval; "
                           "NO optimizer, NO steps (e191's machinery "
                           "verbatim)",
            "rider": f"{RIDER_N} pinned steps of L2 "
                     f"{STEP_SCALE_COMMITTED / RIDER_N:.6f} along the FROZEN "
                     f"g-ray (no re-orientation), battery every "
                     f"{RIDER_EVERY} steps",
            "input_seed": FREEZE_SEED,
            "measure_light": "fact g-12 battery + frac argmax Z + CE_R at "
                             "every grid point (e191's read list verbatim)",
            "random_seed_registry": {"seeds_used": list(RAND_SEEDS),
                                     "no_collisions": "g3K isotropic used "
                                     "11401-3/11411-13; the e131 lineage "
                                     "10902/10901/170/1337/24301/26502 — "
                                     "11501-3 are fresh"},
        },
        "dual_currency": {
            "N_param": N_PARAM, "rms_denominator": RMS_DEN,
            "convention": "per-coordinate RMS = D / sqrt(N) on every row "
                          "and both figure axes (R59 sec 1b's demand)",
            "context_r59_no_claims": {
                "g_ray_kill_0.92_abs": 0.92, "g_ray_kill_0.92_rms":
                    0.92 / 1655.014,
                "a0_sign_path_2.489_rms": 2.489 / 1655.014,
                "g3K_random_rung8_rms": 7.8e-3,
                "note": "g3K's number is ITS organism/ruler (0.87M, own "
                        "wash-1x) — cross-organism pointer only, no claims"},
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_T0": G_T0, "G_U_CK": G_U_CK,
                  "G_A0_XC_ENDPOINT": G_A0_XC_ENDPOINT,
                  "G_R1_XCHK": G_R1_XCHK, "G_RIDER_XC": G_RIDER_XC},
        "gates_a0_signray": G_A0_XC_SIGNRAY,
        "profiles": {r["key"]: {"label": r["label"],
                                "provenance": r["prov"],
                                "rows": ([{"D": 0.0, "rms": 0.0,
                                           "gm12": root_cells["gm12"],
                                           "frac_argmax_z": None,
                                           "ce_r": root_cells["ce_r"],
                                           "disp_check": 0.0,
                                           "disp_dev": 0.0}]
                                          + r["rows"])}
                     for r in rays},
        "ray_summaries": summaries,
        "rider": {"registration": "300 pinned steps, delta = (1.6543/300)*u "
                  "frozen, battery every 30; the interventional denial of "
                  "re-orientation",
                  "disclosure": "the walk's parameter points coincide with "
                                "the static g-ray BY CONSTRUCTION (straight "
                                "walk down the frozen ray, fp accumulation "
                                "aside) — disclosed before compute; the "
                                "rider's content is the framing + the finer "
                                "cliff trace",
                  "step_L2": STEP_SCALE_COMMITTED / RIDER_N,
                  "rows": rider_state["rows"]},
        "references": {
            "e191": {"metrics": "runs/e191/metrics.json",
                     "role": "the committed g-ray profile (the 12-point "
                             "crosscheck), the cliff-edge bracket, the "
                             "direction checkpoint, and ALL machinery "
                             "(verbatim)"},
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "A0's committed step-1 read (the sign-ray "
                             "crosscheck target) + the t=0 committed values"},
            "opt1b": {"metrics": "runs/opt1b/metrics.json",
                      "role": "the re-orienting bleed curve (committed "
                              "overlay in the rider figure — the surviving "
                              "control)"},
            "r59_critic": {"file": "scratch/r59_critic.md",
                           "role": "the forced experiment (sec THE FORCED "
                                   "EXPERIMENT) this cell executes"},
        },
        "overlays": {"bleed_curve": bleed_curve,
                     "a0_sign_path_len": 2.4892616271972656,
                     "e191_cliff_bracket": e191_cliff["bracket_D"],
                     "e191_committed_profile": [e191_prof[k] for k in
                                                sorted(e191_prof)]},
        "adjudication": {
            "bars": bars, "registered_bars_verbatim": REGISTERED_BARS,
            "verdict_map": verdict_map, "clause_map": clause_map,
            "verdict_rider": verdict_rider, "clause_rider": clause_rider,
            "kills": {"g": g_kill, "sign": sign_kill,
                      "rand": {s["key"]: s["first_dead_D"] for s in rand_s},
                      "kill_ratios": kill_ratios},
            "constants": {"shut_bar": SHUT_BAR, "sign_band": list(SIGN_BAND),
                          "random_kill_floor": RANDOM_KILL_FLOOR,
                          "pump_ridge_margin": PUMP_RIDGE_MARGIN,
                          "small_D_pump_band": list(SMALL_D_PUMP_BAND),
                          "root_gm12": root_cells["gm12"],
                          "rider_clause_D_cap": 6 * RIDER_EVERY * (STEP_SCALE_COMMITTED / RIDER_N)},
            "composite_orders": "map: TERRAIN-ONE-PICTURE -> BANDS-REORDER "
                                "-> GRADED (first two mutually exclusive by "
                                "construction); rider: RIDER-REORIENTATION-"
                                "CAUSAL -> RIDER-STEPSIZE-OWNS -> GRADED",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 organism (the e131 consolidated root, seed "
                            "lineage 10902), ONE fact, ONE ruler, static "
                            "rays on ONE root — every ray family is a "
                            "direction family on the same object; the honest "
                            "replicate axis is a second ORGANISM (e191's own "
                            "honesty block names it), a separate dispatch "
                            "decision; the 3 random rays are 3 draws from "
                            "infinitely many, not a sweep"),
            "sign_ray_scope": ("R2 is ONE static direction — sign(g_0) at "
                               "t=0 — NOT Adam's adaptive two-step path "
                               "(A0's 'width 2.5' was a cumulative path "
                               "length with sign(g_t) recomputed at step 2, "
                               "u_cos between steps unreported); the static "
                               "curve and the path are different objects and "
                               "this cell maps the static one"),
            "rider_disclosure": ("the pinned walk's points coincide with the "
                                 "static g-ray BY CONSTRUCTION — the "
                                 "intervention denies re-orientation at "
                                 "held-small step size; it is not a new "
                                 "geometry, and that is exactly its point"),
            "crosschecks": ("every anchor asserted before compute at frozen "
                            "tolerance: the e191 direction bit-gate, the "
                            "t=0 bit gate, the fresh-vs-committed 12-point "
                            "g-ray crosscheck, the A0 exact-endpoint anchor "
                            "(1e-3), the A0 sign-ray anchor (0.05, "
                            "residual = Adam's weight-decay component, "
                            "measured), the rider step-300 anchor"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185..e191 reduction order); fp64 cosines "
                              "for direction gates (e191's lesson); the "
                              "rider accumulates 300 incremental fp32 subs "
                              "vs the profile's single subtraction — the "
                              "step-300 anchor quantifies the accumulated "
                              "difference"),
            "no_guarantees": ("nothing was guaranteed ex ante: each ray "
                              "guarantees only its geometry (unit L2 from "
                              "theta_0 along its direction family); "
                              "whether it pumps, spares, or kills at any D "
                              "is the measurement; the openness is the "
                              "point. Verdicts actually observed: map "
                              f"'{verdict_map}', rider '{verdict_rider}'"),
        },
        "trims": trims, "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu", "torch_threads": 4,
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    with open(rd / "journal.jsonl", "w", encoding="utf-8") as f:
        for r in journal:
            f.write(json.dumps(r) + "\n")

    plot_fig5(rd / "e192_fig5_terrain.png", rays, summaries, rider_state,
              root_cells, verdict_map, clause_map, verdict_rider,
              clause_rider, RMS_DEN)
    plot_rider(rd / "e192_rider.png", rays, rider_state, root_cells,
               bleed_curve, verdict_rider, clause_rider, e191_cliff)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'e192_fig5_terrain.png'}, "
        f"{rd / 'e192_rider.png'}, journal.jsonl")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_fig5(path, rays, summaries, rider_state, root_cells, verdict_map,
              clause_map, verdict_rider, clause_rider, rms_den):
    """THE Fig-5 candidate: all five ray families, one organism, one ruler,
    dual currency (bottom axis absolute L2, top axis per-coordinate RMS)."""
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    style = {"R1_G": ("black", "D", "-", 2.6, "G-RAY — raw wash gradient "
                                             "(e191 committed direction)"),
             "R2_SIGN": ("darkorange", "p", "-", 2.2,
                         "SIGN-RAY — static sign(g_0), normalized (NEW)")}
    for r in rays:
        if r["key"] in style:
            col, mk, ls, lw, lbl = style[r["key"]]
        else:
            gi = r["key"][-5:]
            col = {"11501": "tab:blue", "11502": "tab:green",
                   "11503": "tab:red"}.get(gi, "tab:gray")
            mk, ls, lw = "o", "--", 1.8
            lbl = f"RANDOM — Gaussian unit ray seed {gi}"
        ax.plot([row["D"] for row in r["rows"]],
                [row["gm12"] for row in r["rows"]],
                marker=mk, ms=7, ls=ls, lw=lw, color=col, alpha=0.92,
                label=lbl)
    # the rider's battery points ride the g-ray (the intervention)
    ax.plot([row["D_cum"] for row in rider_state["rows"]],
            [row["gm12"] for row in rider_state["rows"]], "x", ms=10,
            mew=2.4, color="dimgray", alpha=0.95, zorder=6,
            label="RIDER — 300 pinned steps down the FROZEN g-ray "
                  "(small steps, NO re-orientation)")
    # furniture
    ax.axvspan(0.80, 0.92, color="gold", alpha=0.15,
               label="g-ray cliff edge [0.80, 0.92] (e191 committed)")
    ax.axvline(2.4892616271972656, ls=":", lw=1.6, color="magenta",
               alpha=0.8, label="A0 sign PATH length 2.489 (opt1 committed, "
                                "context)")
    ax.axvline(3.68, ls=":", lw=1.6, color="brown", alpha=0.8,
               label="4x the g-ray kill = 3.68 (the wide-kill floor)")
    for yv, col, lbl in ((SHUT_BAR, "tab:purple", "0.27 DISSOLVE"),
                         (root_cells["gm12"], "gray",
                          f"root g-12 {root_cells['gm12']:.3f}")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    # first-dead stars per ray
    for s in summaries:
        if s["first_dead_D"] is not None:
            ax.plot([s["first_dead_D"]],
                    [next(row["gm12"] for row in
                          next(r for r in rays if r["key"] == s["key"])["rows"]
                          if row["D"] == s["first_dead_D"])],
                    "*", ms=15, color="yellow", mec="k", zorder=8)
    secax = ax.secondary_xaxis(
        "top", functions=((lambda d: d / rms_den),
                          (lambda r: r * rms_den)))
    secax.set_xlabel(f"per-coordinate RMS  (D / sqrt(N), N = 2,739,072 — "
                     f"the Rule-12 currency)", fontsize=8.5)
    ax.set_xlabel(r"STATIC displacement $D=\|\theta_D-\theta_0\|_2$ "
                  r"(absolute L2)", fontsize=9.5)
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))", fontsize=9.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.07, 4.12)
    ax.legend(fontsize=6.6, loc="lower left")
    ax.set_title("THE ONE-ORGANISM ALL-RAY TERRAIN MAP — 5 ray families, "
                 "one root, one ruler, dual currency", fontsize=10.5)

    ax = axes[1]
    ax.axis("off")
    y = 0.985
    ax.text(0.02, y, f"E192 MAP VERDICT:    {verdict_map}", fontsize=10,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.045
    for wd in textwrap.wrap(clause_map, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.021
    y -= 0.02
    ax.text(0.02, y, f"E192 RIDER VERDICT:  {verdict_rider}", fontsize=10,
            va="top", family="monospace", weight="bold", color="darkblue")
    y -= 0.045
    for wd in textwrap.wrap(clause_rider, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.021
    y -= 0.025
    ax.text(0.02, y, "  ray              kill D    kill rms   kill/g    "
                     "pump-rise@smallD", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.024
    for s, r in zip(summaries, rays):
        kd = (f"{s['first_dead_D']:7.3f}" if s["first_dead_D"] is not None
              else "   >4.0 ")
        kr = (f"{s['first_dead_rms']:.2e}" if s["first_dead_rms"] is not None
              else "unres.")
        ratio = (f"{s['first_dead_D'] / 0.92:6.2f}x"
                 if s["first_dead_D"] is not None else " >4.35x")
        ax.text(0.02, y,
                f"  {s['key']:>16s}  {kd}  {kr}  {ratio}  "
                f"{s['small_D_max_rise']:+.4f}"
                + (" RIDGE" if s["pump_ridge"] else ""),
                fontsize=6.9, va="top", family="monospace",
                color=("crimson" if s["pump_ridge"] else "black"))
        y -= 0.022
    fig.suptitle("E192 — THE ONE-ORGANISM ALL-RAY TERRAIN MAP + THE "
                 "PINNED-RAY RIDER (R59's forced cell; the Fig-5 license)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rider(path, rays, rider_state, root_cells, bleed_curve,
               verdict_rider, clause_rider, e191_cliff):
    """The rider figure: pinned small steps (orientation denied) vs the
    re-orienting bleed (committed) vs the static g-ray — W026's mechanism
    noun made interventional."""
    import textwrap
    g_rows = next(r for r in rays if r["key"] == "R1_G")["rows"]
    sign_rows = next(r for r in rays if r["key"] == "R2_SIGN")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 7.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot([row["D"] for row in bleed_curve],
            [row["gm12"] for row in bleed_curve], "o-", ms=4.0, lw=1.6,
            color="royalblue", alpha=0.85,
            label="THE BLEED — re-orienting small steps (opt1b committed)")
    ax.plot([row["D"] for row in g_rows], [row["gm12"] for row in g_rows],
            "D-", ms=7, lw=2.2, color="black", alpha=0.9,
            label="STATIC g-ray profile (e192 fresh = e191 committed)")
    ax.plot([row["D_cum"] for row in rider_state["rows"]],
            [row["gm12"] for row in rider_state["rows"]], "x-", ms=11,
            mew=2.6, lw=1.6, color="crimson", alpha=0.95, zorder=6,
            label=f"RIDER — {RIDER_N} pinned steps, orientation DENIED "
                  f"(this run)")
    ax.plot([row["D"] for row in sign_rows],
            [row["gm12"] for row in sign_rows], "p--", ms=6, lw=1.6,
            color="darkorange", alpha=0.85, label="STATIC sign-ray (e192)")
    ax.plot([STEP_SCALE_COMMITTED], [A0_S1_GM12_COMMITTED], "H", ms=11,
            color="magenta", mec="k", zorder=7,
            label="A0 committed step-1 read 0.678 @ 1.654 (the sign-ray "
                  "crosscheck)")
    ax.axvspan(e191_cliff["bracket_D"][0], e191_cliff["bracket_D"][1],
               color="gold", alpha=0.15,
               label=f"static cliff bracket {e191_cliff['bracket_D']}")
    for yv, col, lbl in ((SHUT_BAR, "tab:purple", "0.27 DISSOLVE"),
                         (root_cells["gm12"], "gray",
                          f"root g-12 {root_cells['gm12']:.3f}")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.set_xlabel(r"displacement $D$ along the g-ray (absolute L2)",
                  fontsize=9.5)
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))", fontsize=9.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.06, 1.85)
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("THE RIDER — small steps down the FROZEN ray vs the "
                 "re-orienting bleed", fontsize=10.5)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E192 RIDER VERDICT: {verdict_rider}", fontsize=10.5,
            va="top", family="monospace", weight="bold", color="darkblue")
    y -= 0.05
    for wd in textwrap.wrap(clause_rider, width=90, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.0, va="top", family="monospace")
        y -= 0.024
    y -= 0.015
    ax.text(0.02, y, "  step    D_cum      rms      g-12    argmax   CE_R",
            fontsize=7.2, va="top", family="monospace")
    y -= 0.026
    for row in rider_state["rows"]:
        ax.text(0.02, y,
                f"  {row['step']:4d}  {row['D_cum']:7.4f}  {row['rms']:.2e}"
                f"  {row['gm12']:.4f}  {row['frac_argmax_z']:.2f}   "
                f"{row['ce_r']:.4f}",
                fontsize=6.9, va="top", family="monospace",
                color=("crimson" if row["gm12"] <= SHUT_BAR else "black"))
        y -= 0.024
    fig.suptitle("E192 RIDER — is the bleed's protection re-orientation "
                 "(walk dies at the cliff) or step size (walk survives)?",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
