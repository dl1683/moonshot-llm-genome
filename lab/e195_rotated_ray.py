"""E195 — THE ROTATED-RAY TERRAIN (e194/T156's named follow-on: WHERE did
the lethal support flee TO?).

WHY: e194 (T156) convicted ROTATION-TO-FLEEING-SUPPORT: the recomputed
sign front pursues the fact's fleeing lethal support — one recomputation
is the whole bonus (the k-ladder a step function: k=1 kills at 1.7496,
k>=2 at the true static edge 2.2699; the 2.5 was coarse-grid). THE
QUESTION THIS CELL OWNS: is the FLEEING DIRECTION ITSELF the more lethal
one — an absolute property of sign(g_1) as a ray — or is the lethality
RELATIVE to the state (the terrain rotates under the walker's feet)?
Map the terrain along the ROTATED rays.

THE CELL (eval-only CPU, minutes): static graded jumps along THREE ray
families from the ROOT state, D grid 0.05..3.00, dual currency, the
standard ruler (g-12 install-60 battery, SHUT bar 0.27):
  (1) sign(g_0) — the original static sign ray (e192's R2; e194's fine
      grid already covers 1.5-2.5 — this cell re-gates those 21 points
      and EXTENDS THE SHOULDERS 0.05-1.5 and 2.5-3.0);
  (2) sign(g_1) — THE ROTATED RAY: the sign of the gradient at the
      one-stepped state theta_1 (the t=1 fresh front — the direction
      the support fled TOWARD; rebuilt from the licensed stream,
      bit-gated vs e194's committed front trace + the saved
      runs/checkpoints/opt2_a_sign_s2.pt);
  (3) sign(g_2) (the t=2 front) — the rotation's continuation.
ALSO the same three families evaluated FROM THE ONE-STEPPED STATE
(jumps from theta_1 along sign(g_0)/sign(g_1)/sign(g_2)) — does the
terrain rotate under the walker's feet, or is the direction itself
special from anywhere?

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - FLEEING-IS-LETHAL: "fires if the rotated ray sign(g_1) from the
    ROOT kills at a materially lower D than sign(g_0) (>= 15% lower) —
    the fleeing direction is itself the more lethal one; the terrain's
    lethality concentrates where the support fled."
  - ROTATION-IS-RELATIVE: "fires if the rays are near-equivalent from
    the root but sign(g_1)-from-theta_1 kills far below
    sign(g_0)-from-theta_1 — the lethality is RELATIVE to the state
    (the terrain rotates under the walker), not a fixed direction."
  - FLAT-TERRAIN: "the three rays are within ~10% everywhere — the
    fleeing-support reading needs a different instrument; reported
    honestly."
  - GRADED: "any mix — the profiles verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * "kills at D" = the family profile's FIRST g-12 <= 0.27
    downcrossing on the static grid, linear-in-D interpolated (e192/
    e194's convention); None = no downcrossing within [0, 3.0] (the
    family never kills in the window — any bar clause needing that
    number dies and is reported verbatim).
  * "materially lower (>= 15% lower)" = D_kill(root, u1) <= 0.85 *
    D_kill(root, u0), both present.
  * "near-equivalent from the root" = pairwise relative spread among
    the root panel's present D_kills <= 0.10 (relative to the root
    panel's max D_kill; any None in the panel -> not near-equivalent,
    reported verbatim).
  * "kills far below" = D_kill(theta_1, u1) <= 0.85 * D_kill(theta_1,
    u0), both present (the mirrored 15% margin — frozen).
  * "within ~10% everywhere" = pairwise relative spread <= 0.10 in
    BOTH panels (root and theta_1) among present D_kills.
  * composite order frozen: FLEEING-IS-LETHAL -> ROTATION-IS-RELATIVE
    -> FLAT-TERRAIN -> GRADED (first that fires is the verdict; every
    bar's fires flag reported verbatim).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): the
standard cell gates (corpus ZEPH 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16 starts
bit-equal e185's stored list; root gated vs e151's before-cells; the
t=0 bit-gate vs opt1's committed A0 row; the stream md5 gates; T150's
same-point machinery gate; e191's u_g checkpoint gate; e192's R2
direction md5 gate) PLUS THE SAVED-STATE PROVENANCE CHAIN (e194's): the
rebuilt k=1 walk must reproduce opt2's committed a_sign trajectory
(G_REPRO, incl. the densified kill bracket and e194's committed
post-kill step-3 row) AND the saved runs/checkpoints/opt2_a_sign_s2.pt
parameters (G_SAVEDSTATE) AND e194's committed front trace at t=0,1,2
(G_FRONTS, bit-class) — failure = control failure, abort. The new
profiles are gated before they are believed: the root-u0 family must
reproduce e194's committed 21 fine-grid rows (G_ROOTPROF; e192's R2 at
1.5/2.0/2.5 cross-reported) and the theta_1 families must reproduce
the walked anchors at their exact-landing Ds (G_TH1PROF: D=0 ->
opt2's committed s1 g-12 0.6785961389541626; D=STEP_L2 along -u1 ->
the committed s2 kill read 9.830708586378023e-05, texture class).
WHAT EACH ARM GUARANTEES: NOTHING — every family is a static graded
jump at lethal scale from a state that could kill anywhere; that
openness is the point.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before
torch; the GPU is another agent's, never claimed), torch threads 4
(the e185-era reduction order), sequential phases, every phase < 180 s,
PROGRESSIVE partial metrics.json writes after every phase (the outage
lesson), n=1, single stream seed 10902 lineage.

Outputs: runs/e195/{metrics.json, e195_rotated_ray.png,
e195_kill_summary.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e195_rotated_ray.py    (E195_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (the opt-line convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E195_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e195 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (the opt-line, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
OPT2_METRICS = E43.REPO / "runs" / "opt2" / "metrics.json"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
E192_METRICS = E43.REPO / "runs" / "e192" / "metrics.json"
ECHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
E194_METRICS = E43.REPO / "runs" / "e194" / "metrics.json"
E191_DIR_CK = CKPT_DIR / "e191_static_dir_u.pt"
OPT2_SIGN_CK = CKPT_DIR / "opt2_a_sign_s2.pt"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e185/opt1/opt2/e194)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; no arm has an lr)
STEP_L2 = 1.6542880535125732      # opt1 A0's committed step-1 L2 = the matched per-step L2
CE1_COMMITTED = 1.356567621231079        # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253       # opt1 A0's committed step-1 pre-clip grad norm
A0_DKILL_COMMITTED = 2.4892616271972656  # opt1 A0's committed kill displacement
RAW_DKILL_COMMITTED = 0.9203406595225093 # opt1c's committed densified raw kill D
SIGN_EDGE_COMMITTED = 2.5                # e192's committed static sign-ray kill D (coarse grid)
PATH_DKILL_COMMITTED = 1.749640490742179 # opt2 a_sign's committed densified kill D
REFINED_STATIC_EDGE = 2.269916581032063  # e194 phase B's committed refined static edge
W024_ADAM_SAMEPOINT = 0.039550412581627135  # the chart's committed same-point sign read
W024_RAW_SAMEPOINT = 0.09862135965497001    # the chart's committed same-point raw read
# opt2 a_sign's committed trajectory (the reproduction targets, G_REPRO)
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
E192_R2_U_MD5 = "aac6c6d643e327939f680148772dd180"   # e192's committed R2 direction md5
E192_R2_UNORM = 0.9998204112052917
E192_R2_GATE = {                    # e192's committed R2 rows at the shared Ds (G_STATIC_XC)
    1.5: {"gm12": 0.7637382745742798, "ce_r": 2.0823192596435547},
    2.0: {"gm12": 0.4483865201473236, "ce_r": 2.5618293285369873},
    2.5: {"gm12": 0.14364269375801086, "ce_r": 3.1449053287506104}}
E192_A0_SIGNRAY_GM12 = 0.6785961389541626  # e192's static-sign D1.6543 read (corroboration)
# e194's committed reads this cell re-gates (the provenance chain for theta_1/u1/u2)
E194_WALK_S3 = {"ce_batch": 1.8582803010940552, "cum_disp": 2.539003849029541,
                "gm12": 0.0039044867735356092, "preclip_gnorm": 3.926907539367676,
                "matched_point_m12": 0.133829189309824}
E194_FRONTS = [   # committed front-trace rows t=0,1,2 (the bit-class comparators)
    {"front_state": 0, "vs_g_ray": 0.5950223410088911,
     "vs_static_sign": 1.0000000000017162, "vs_prev_front": None},
    {"front_state": 1, "vs_g_ray": -0.1622211100709781,
     "vs_static_sign": -0.15452721980470194, "vs_prev_front": -0.15452721980443676},
    {"front_state": 2, "vs_g_ray": 0.034184469615974565,
     "vs_static_sign": 0.03151076362111852, "vs_prev_front": -0.20298347429000238},
]

ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
EXT_STEPS = 1                     # post-kill continuation (read-only; g_2's read step)
D_TARGET = 5.0                    # walk horizon (opt2's; the walk kills long before)
SHUT_BAR = 0.27                   # e185's kill bar (the standard ruler)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # opt1c's along-path kill-bracket densification
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
N_PARAM = 2_739_072
RMS_DENOM = 1655.0141993348577    # sqrt(N_PARAM) — the per-coordinate RMS currency
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (61 pts? no: 60)
CE_EVERY = 0.2                    # ce_r at D in {0.2, 0.4, ..., 3.0} (dual currency)
MATERIAL_RATIO = 0.85             # frozen: ">= 15% lower" / "far below"
FLAT_RATIO = 0.10                 # frozen: "within ~10%"
L2_ASSERT_TOL = 1e-5
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_SAMEPOINT_TOL = 2e-3
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_FRONT_TOL = 1e-9                # bit-class: e194's committed front-trace comparators
G_STATIC_TOL = 1e-3               # e192's own G_R1_XCHK tolerance class
G_LANDING_TOL = 2e-3              # texture class: fp32-norm ray landing vs walked fp64-norm step
G_ALIGN_TOL = 1e-6                # unit-ray form vs delta-form cosine texture
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.25, 0.5, 1.0, 1.5, 1.65, 1.75, 2.0, 2.27, 2.5, 3.0]
    CE_EVERY = None               # ce only at anchors in smoke

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
}
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream)
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

REGISTERED_PREDICTION = {
    "fleeing_is_lethal": "FLEEING-IS-LETHAL: \"fires if the rotated ray "
        "sign(g_1) from the ROOT kills at a materially lower D than "
        "sign(g_0) (>= 15% lower) — the fleeing direction is itself the "
        "more lethal one; the terrain's lethality concentrates where the "
        "support fled.\"",
    "rotation_is_relative": "ROTATION-IS-RELATIVE: \"fires if the rays are "
        "near-equivalent from the root but sign(g_1)-from-theta_1 kills "
        "far below sign(g_0)-from-theta_1 — the lethality is RELATIVE to "
        "the state (the terrain rotates under the walker), not a fixed "
        "direction.\"",
    "flat_terrain": "FLAT-TERRAIN: \"the three rays are within ~10% "
        "everywhere — the fleeing-support reading needs a different "
        "instrument; reported honestly.\"",
    "graded": "GRADED: \"any mix — the profiles verbatim.\"",
    "operationalizations": "static graded jumps theta_D = anchor - D*u for "
        "u in {u0, u1, u2} = sign(g_t)/||sign(g_t)|| (e192's fp32-norm "
        "construction) with g_t the licensed stream's t=0/1/2 post-clip "
        "wash gradients (the k=1 walk rebuilt bit-exactly, gated vs "
        "opt2's committed trajectory + the saved opt2_a_sign_s2.pt + "
        "e194's committed front trace); anchors = theta_0 (the e131 "
        "root) and theta_1 (the walk's one-stepped state, proven by the "
        "s2-md5 chain); grid D in {0.05..3.00 step 0.05} + D=0 anchor "
        "rows + exact-landing gate points; KILL = the profile's first "
        "g-12 <= 0.27 downcrossing, linear-in-D interpolated; "
        "'materially lower (>= 15% lower)' = D_kill(root,u1) <= 0.85 * "
        "D_kill(root,u0); 'near-equivalent from the root' = pairwise "
        "relative spread of the root panel's present D_kills <= 0.10 "
        "(vs the panel max); 'kills far below' = D_kill(th1,u1) <= 0.85 "
        "* D_kill(th1,u0) (the mirrored 15% margin); 'within ~10% "
        "everywhere' = pairwise spread <= 0.10 in BOTH panels; any None "
        "in a comparison kills that clause and is reported verbatim; "
        "composite order frozen FLEEING-IS-LETHAL -> ROTATION-IS-RELATIVE "
        "-> FLAT-TERRAIN -> GRADED.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "walk_arm is lab/e194_sign_front.py's VERBATIM (whose own provenance "
    "is lab/opt2_density.py / opt1c / e185 — the e176n lineage) with "
    "ADDITIVE stashes only: the refresh-gradient tensors (g_fronts) and "
    "the per-step endpoint flats (endpoints) are collected for the ray "
    "construction; no arithmetic path changes.",
    "The walk extends 1 step past the kill (read-only): step 3 is flagged "
    "post_kill and never adjudicates; only its refresh gradient's SIGN "
    "(g_2, the t=2 front) is probed as a ray direction — g_2 was read on "
    "the continuation in e194 too (bit-rebuildable, disclosed).",
    "No new checkpoints (e194's convention): the rays and anchors are "
    "bit-rebuildable from the licensed stream; the direction md5s are "
    "registered in metrics.json for any follow-on.",
    "ce_r (the second currency) at decimated Ds — {0} plus every 0.2 in "
    "(0, 3.0], plus e192's crosscheck Ds {1.5, 2.0, 2.5} on the root-u0 "
    "family; the g-12 battery is the ruler and is read at EVERY grid "
    "point. Dual displacement currency (D and per-coordinate RMS) on "
    "every row.",
    "Light reads only (battery + ce_r + anchor fact-gradient matched-point "
    "cosines); CPU-ONLY, threads 4, n=1, seed lineage 10902.",
    "Smoke mode trims: grid {0.05, 0.25, 0.5, 1.0, 1.5, 1.65, 1.75, 2.0, "
    "2.27, 2.5, 3.0}, ce at anchors only (verdict stamped SMOKE; nothing "
    "adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e194_sign_front.py VERBATIM (whose own provenance is
# lab/opt2_density.py / lab/opt1c_direction_size.py / e185 — the e176n
# lineage). Copied rather than imported to own the device policy and the
# bit-exact arithmetic.

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
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ: gradient of the fact battery's mean log p(Z)
    readout at the twin's current weights. Critic's sign convention:
    NEGATIVE cos(delta, grad) = displacement aligned with the DEATH
    gradient. Consumes no RNG; run on the eval twin."""
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


def flat_md5(net: TinyGPT) -> str:
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 cosine (the chart's estimator precision)."""
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def sign_update(g: torch.Tensor, step_l2: float):
    """opt2's sign_update VERBATIM: delta = -step_l2 * sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — the matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


# ------------------------------------------------------------------ kill bracket

def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    """The densified bracket (opt1c's convention; this value adjudicates)."""
    pts = [(d0, v0)] + [(r["D"], r["gm12"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


# ------------------------------------------------------------------ the walk

def build_batch(gen, anchor, train_ids):
    """opt2's draw order VERBATIM: aj then rj, anchors + random windows."""
    aj = torch.randint(anchor.shape[0], (ANCH_BS,), generator=gen)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=gen)
    anc = anchor[aj]
    rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
    return torch.cat([anc[:, :-1], rnd[:, :-1]], 0), \
           torch.cat([anc[:, 1:], rnd[:, 1:]], 0), rj


def walk_arm(k, net0, anchor, train_ids, itos, gm12_ids, zid, theta0,
             u_g, u_sign, root_m12_grad, rd, write_partial, stub,
             extend_past_kill=0, fact_reads=False):
    """A k-ladder sign walk on the licensed stream (e194's walk_arm
    VERBATIM + additive stashes: g_fronts, endpoints). Front refreshed when
    (step-1) % k == 0 (k=1: every step = opt2's a_sign). Every-step g-12;
    per-step matched-L2 assertion; kill bracket densified (opt1c)."""
    net = copy.deepcopy(net0)
    net.train()
    evl = copy.deepcopy(net0)
    evl.eval()
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    n_steps = 16
    journal, front_trace, kill = [], [], None
    g_fronts = []                       # ADDITIVE stash: refresh gradients
    endpoints = {}                      # ADDITIVE stash: per-step flats
    max_l2_dev, zeph = 0.0, 0
    prev_front = None
    prev_delta = None
    prev_state_m12_grad = root_m12_grad.clone()   # grad m12 at theta_{s-1}
    t0a = time.time()
    step = 0
    while step < n_steps:
        step += 1
        post_kill = kill is not None
        if post_kill and step > (kill["step"] + extend_past_kill):
            break
        x, y, rj = build_batch(gen, anchor, train_ids)
        x_md5 = hashlib.md5(x.contiguous().numpy().tobytes()).hexdigest()
        if step in E185_XHASH:
            assert x_md5 == E185_XHASH[step], \
                f"k={k}: step-{step} batch md5 diverged from the licensed stream"
        for w in torch.stack([train_ids[s: s + BLOCK] for s in rj]):
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph += 1
        prev = flat_params(net)
        refresh = ((step - 1) % k == 0)
        if refresh:
            logits, _ = net(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            net.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            g_t = torch.cat([p.grad.detach().reshape(-1)
                             for p in net.parameters()]).clone()   # post-clip
            delta = sign_update(g_t, STEP_L2)
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
        else:
            delta = prev_delta.clone()
            ce_batch, gnorm_row = None, None
            frow = None
        # matched-L2 assertion (fp64 measured, as opt2)
        l2dev = abs(float(torch.norm(delta.double())) - STEP_L2)
        assert l2dev < L2_ASSERT_TOL, \
            f"k={k}: per-step L2 dev {l2dev:.2e} — the matched-L2 assertion"
        max_l2_dev = max(max_l2_dev, l2dev)
        # the dual-estimator matched-point read (before the step moves the
        # point): cos(delta_s, grad m12(theta_{s-1})) — the increment at the
        # point it began (T150's lesson; s=1 == the chart's same-point read)
        mp = cos64(delta, prev_state_m12_grad) if fact_reads else None
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
               "cum_vs_static_sign": cos64(d_cum, u_sign),
               "matched_point_m12": mp}
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
        if fact_reads:
            g_m12_now = fact_grad(evl, gm12_ids, zid)     # grad m12 at theta_s
            row["post_step_m12"] = cos64(d_cum, g_m12_now)
            row["root_point_m12"] = cos64(d_cum, root_m12_grad)
            prev_state_m12_grad = g_m12_now.clone()
        journal.append(row)
        log(f"  [k={k}] s{step:2d}{' R' if refresh else '  '}{' POSTKILL' if post_kill else ''}"
            f" g-12 {row['gm12']:.4f} D {cum_disp:.4f}"
            + (f" ov(g-ray) {frow['vs_g_ray']:+.4f}" if frow else ""))
        # KILL bracket (opt1c's convention; computed once, walk may continue)
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
            log(f"  [k={k}] KILL at s{step}: D_kill(dens) {kill['D_kill']:.4f} "
                f"(raw {kill['D_kill_raw']:.4f})")
        elif kill is None and cum_disp >= D_TARGET:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {D_TARGET} alive",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
        if step >= 16 and kill is None:
            kill = {"kind": "cap", "step": step,
                    "reason": "step cap 16",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
    out = {"k": k, "journal": journal, "front_trace": front_trace,
           "stop": kill, "max_l2_dev": max_l2_dev, "zeph": zeph,
           "g_fronts": g_fronts, "endpoints": endpoints,
           "net": net, "train_seconds": time.time() - t0a}
    return out


# ------------------------------------------------------------------ profiles

def static_profile(evl, anchor_flat, u, dgrid, gm12_ids, zid, r_eval_xy,
                   ce_ds):
    """Static graded jumps anchor - D*u (e192's placement VERBATIM), every
    point read on the standard ruler (g-12 battery) + dual displacement
    currency; ce_r (second currency) at the frozen decimated Ds."""
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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e195_smoke" if SMOKE else "e195")
    log(f"E195 THE ROTATED-RAY TERRAIN (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), phases < 180s, "
        f"progressive writes, n=1, seed lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT2_METRICS, OPT1C_METRICS, OPT1_METRICS, E192_METRICS,
              ECHART_METRICS, E194_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    opt2m = json.loads(OPT2_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    e192m = json.loads(E192_METRICS.read_text(encoding="utf-8"))
    chartm = json.loads(ECHART_METRICS.read_text(encoding="utf-8"))
    e194m = json.loads(E194_METRICS.read_text(encoding="utf-8"))
    # hard-bind the committed references (asserts catch committed-file drift)
    assert abs(opt2m["arms"]["a_sign"]["stop"]["D_kill"]
               - PATH_DKILL_COMMITTED) < 1e-12
    assert abs(opt1cm["adjudication"]["stop"]["D_kill_interp"]
               - RAW_DKILL_COMMITTED) < 1e-12
    assert abs(e192m["adjudication"]["bars"]["TERRAIN_ONE_PICTURE"]["sign_kill_D"]
               - SIGN_EDGE_COMMITTED) < 1e-12
    assert abs(opt1m["arms"]["a0_adamw_ref"]["tstar"]["D_at_kill"]
               - A0_DKILL_COMMITTED) < 1e-12
    chart_same = chartm["gates"]["G_W024"]
    assert abs(chart_same["cos_adamview_m12_at_t0_samepoint"]
               - W024_ADAM_SAMEPOINT) < 1e-12
    assert abs(chart_same["cos_mov_m12_at_t0_samepoint"]
               - W024_RAW_SAMEPOINT) < 1e-12
    # e194's committed artifacts this cell chains through
    e194_ft = e194m["phaseA_front_overlap"]["front_trace"]
    e194_wt = e194m["phaseA_front_overlap"]["walk_traj"]
    e194_fine = e194m["phaseB_static_fine"]["rows"]
    assert abs(e194m["phaseB_static_fine"]["refined_static_edge"]
               - REFINED_STATIC_EDGE) < 1e-12
    for i, ref in enumerate(E194_FRONTS):
        assert e194_ft[i]["front_state"] == ref["front_state"]
        assert abs(e194_ft[i]["vs_g_ray"] - ref["vs_g_ray"]) < 1e-12
        assert abs(e194_ft[i]["vs_static_sign"] - ref["vs_static_sign"]) < 1e-12
    assert abs(e194_wt[2]["ce_batch"] - E194_WALK_S3["ce_batch"]) < 1e-12
    assert abs(e194_wt[2]["gm12"] - E194_WALK_S3["gm12"]) < 1e-12
    assert abs(e194_wt[1]["matched_point_m12"]
               - OPT2_S2["matched_point_m12"]) < 1e-12
    log("parents: opt2 (a_sign path kill 1.7496), e194 (front trace + fine "
        f"static edge {REFINED_STATIC_EDGE} + the walked s1/s2/s3 rows), "
        "e192 (R2 static edge 2.5 coarse), opt1c/opt1/e_chart — loaded "
        "COMMITTED, never rerun")

    # ---------------- protocol rebuild (the licensed cell, verbatim) -----------
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

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
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

    # the neutral stream (e170 VERBATIM via e185 arm C)
    arng = _random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                         "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — e170's "
                         "construction VERBATIM"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    # root net + gate vs e151
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    theta0 = flat_params(net0)
    assert theta0.numel() == N_PARAM, \
        f"parameter count {theta0.numel()} != {N_PARAM} (the e131 line)"

    evl0 = copy.deepcopy(net0)
    root_cells = {
        "gm12": battery_cell(evl0, gm12_ids, zid)["mean_pz"],
        "g0": battery_cell(evl0, g0_ids, zid)["mean_pz"],
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
    log(f"G_ROOT (vs e151 before-cells): max|diff| {rmax:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED")

    # G_T0: the fresh t=0 wash gradient + AdamW step (the opt-line gate)
    gen1 = torch.Generator().manual_seed(FREEZE_SEED)
    x1, y1, _ = build_batch(gen1, anchor_neutral, train_ids)
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
    g0_t0 = torch.cat([p.grad.detach().reshape(-1)
                       for p in tw.parameters()]).clone()   # post-clip
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0))
    del tw, optw, logits
    G_T0 = {"step1_x_md5": x1_md5,
            "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
            "ce_batch_measured": ce1, "ce_batch_committed": CE1_COMMITTED,
            "preclip_gnorm_measured": gn1,
            "preclip_gnorm_committed": GN1_COMMITTED,
            "adamw_step1_L2_measured": disp1,
            "adamw_step1_L2_committed": STEP_L2,
            "pass": bool(x1_md5 == E185_XHASH[1]
                         and abs(ce1 - CE1_COMMITTED) < G_FALLBACK_TOL
                         and abs(gn1 - GN1_COMMITTED) < G_FALLBACK_TOL
                         and abs(disp1 - STEP_L2) < G_FALLBACK_TOL),
            "note": "the t=0 bit-gate vs opt1's committed A0 row (the "
                    "matched-L2 reference re-derived on THIS device)"}
    log(f"G_T0: md5 {'match' if G_T0['step1_x_md5_match_e185'] else 'DRIFT'}"
        f", CE |d| {abs(ce1 - CE1_COMMITTED):.2e}, gn |d| "
        f"{abs(gn1 - GN1_COMMITTED):.2e}, L2 |d| "
        f"{abs(disp1 - STEP_L2):.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — abort")

    # G_STREAM: the stream construction verified without training
    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 11):
        x_, _, _ = build_batch(gen_s, anchor_neutral, train_ids)
        stream_ok[s_] = bool(hashlib.md5(
            x_.contiguous().numpy().tobytes()).hexdigest() == E185_XHASH[s_])
    G_STREAM = {"steps_1_10_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values()))}
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # the root fact gradient + G_SAMEPOINT (T150's estimator-lesson gate)
    evl_root = copy.deepcopy(net0)
    evl_root.eval()
    root_m12_grad = fact_grad(evl_root, gm12_ids, zid)
    sp_sign = cos64(-torch.sign(g0_t0), root_m12_grad)
    sp_raw = cos64(-g0_t0, root_m12_grad)
    G_SAMEPOINT = {
        "sign_arm_s1_matched_point_m12_measured": sp_sign,
        "committed_chart_samepoint": W024_ADAM_SAMEPOINT,
        "d_sign": abs(sp_sign - W024_ADAM_SAMEPOINT),
        "raw_sanity_measured": sp_raw,
        "committed_chart_raw": W024_RAW_SAMEPOINT,
        "d_raw": abs(sp_raw - W024_RAW_SAMEPOINT),
        "tol": G_SAMEPOINT_TOL,
        "pass": bool(abs(sp_sign - W024_ADAM_SAMEPOINT) < G_SAMEPOINT_TOL
                     and abs(sp_raw - W024_RAW_SAMEPOINT) < G_SAMEPOINT_TOL),
        "note": "the dual-estimator machinery gate: the SIGN arm's step-1 "
                "matched-point read must reproduce the chart's committed "
                "anchor — every alignment read below states its point",
    }
    log(f"G_SAMEPOINT: sign s1 matched-point {sp_sign:+.6f} vs chart "
        f"{W024_ADAM_SAMEPOINT:+.6f}: "
        + ("PASS" if G_SAMEPOINT["pass"] else "FAIL"))
    if not G_SAMEPOINT["pass"]:
        raise RuntimeError("same-point machinery gate FAILED — abort")

    # G_U191: e191's committed u_g (the death ray), bit-gated fresh
    uck = torch.load(E191_DIR_CK, map_location="cpu", weights_only=False)
    u_g = uck["u"].detach().clone().float()
    fresh_u = g0_t0 / torch.norm(g0_t0)
    G_U191 = {
        "loaded_u_md5_match_fresh": bool(hashlib.md5(
            u_g.numpy().tobytes()).hexdigest() == hashlib.md5(
            fresh_u.numpy().tobytes()).hexdigest()),
        "fresh_cos64_loaded": cos64(fresh_u, u_g),
        "root_md5_match_checkpoint": bool(flat_md5(net0) == uck.get("theta0_md5")),
        "note": "the death direction u_g = e191's committed static g-ray; "
                "the kill ray is DESCENT (-u_g)",
    }
    G_U191["pass"] = bool(G_U191["root_md5_match_checkpoint"]
                          and abs(float(torch.norm(u_g)) - 1.0) < 1e-5)
    log(f"G_U191 (death ray u_g): "
        + ("PASS" if G_U191["pass"] else "FAIL"))

    # G_R2DIR: e192's committed static sign-ray direction, rebuilt + gated
    # (this IS u0, the family-(1) ray)
    s_raw = torch.sign(g0_t0)
    u0 = (s_raw / torch.norm(s_raw)).clone()       # e192's construction verbatim
    u0_md5 = hashlib.md5(u0.numpy().tobytes()).hexdigest()
    G_R2DIR = {
        "u_sign_md5": u0_md5, "committed_e192_md5": E192_R2_U_MD5,
        "md5_match": bool(u0_md5 == E192_R2_U_MD5),
        "u_norm_fp32": float(torch.norm(u0)),
        "u_norm_fp64": float(torch.norm(u0.double())),
        "committed_e192_u_norm": E192_R2_UNORM,
        "norm_note": "e192's committed u_norm is its fp32-norm read "
                     "(sequential fp32 accumulation over 2.7M elements "
                     "carries ~1e-4 relative error); the md5 bit-match is "
                     "the definitive gate",
        "n_zero_g_coords": int((g0_t0 == 0).sum()),
        "note": "family (1) u0 = the static sign(g_0) ray rebuilt from the "
                "gated t=0 gradient (e192's fp32-norm construction VERBATIM)",
    }
    G_R2DIR["pass"] = bool(G_R2DIR["md5_match"])
    log(f"G_R2DIR (u0 = static sign ray): md5 "
        + ("match" if G_R2DIR["md5_match"] else "DRIFT")
        + f" (fp32 norm {G_R2DIR['u_norm_fp32']:.7f} vs e192's "
        f"{E192_R2_UNORM:.7f}; fp64 {G_R2DIR['u_norm_fp64']:.9f}): "
        + ("PASS" if G_R2DIR["pass"] else "FAIL"))
    if not G_U191["pass"] or not G_R2DIR["pass"]:
        raise RuntimeError("committed-direction gates FAILED — abort")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e195_rotated_ray",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_PREDICTION["registration"],
            "registered_prediction": REGISTERED_PREDICTION,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE,
                               "torch": torch.__version__,
                               "threads": torch.get_num_threads()},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    stub["gates"] = {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_ROOT": G_ROOT, "G_T0": G_T0, "G_STREAM": G_STREAM,
                     "G_SAMEPOINT": G_SAMEPOINT, "G_U191": G_U191,
                     "G_R2DIR": G_R2DIR}
    write_partial("standard cell gates passed")
    log("standard cell gates passed; partial metrics written")

    # =====================================================================
    # PHASE 0 — the k=1 walk rebuilt (the provenance chain for u1/u2/theta_1)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the k=1 sign walk rebuilt (opt2's a_sign, 1 post-kill "
        "step for the t=2 front; fact reads on)")
    arm1 = walk_arm(1, net0, anchor_neutral, train_ids, itos, gm12_ids, zid,
                    theta0, u_g, u0, root_m12_grad, rd, write_partial, stub,
                    extend_past_kill=EXT_STEPS, fact_reads=True)
    j1 = arm1["journal"]
    assert len(j1) == 3 and [r["step"] for r in j1] == [1, 2, 3], \
        f"expected the 3-step walk (kill s2 + 1 continuation), got {len(j1)}"

    # ---- G_REPRO: the rebuilt walk vs opt2's committed a_sign trajectory
    #      + e194's committed post-kill step-3 row
    s1, s2, s3 = j1[0], j1[1], j1[2]
    repro_rows = {
        "s1_ce_batch": (s1["ce_batch"], OPT2_S1["ce_batch"]),
        "s1_cum_disp": (s1["cum_disp"], OPT2_S1["cum_disp"]),
        "s1_gm12": (s1["gm12"], OPT2_S1["gm12"]),
        "s1_preclip_gnorm": (s1["preclip_gnorm"], OPT2_S1["preclip_gnorm"]),
        "s1_matched_point_m12": (s1["matched_point_m12"],
                                 OPT2_S1["matched_point_m12"]),
        "s2_ce_batch": (s2["ce_batch"], OPT2_S2["ce_batch"]),
        "s2_cum_disp": (s2["cum_disp"], OPT2_S2["cum_disp"]),
        "s2_gm12": (s2["gm12"], OPT2_S2["gm12"]),
        "s2_preclip_gnorm": (s2["preclip_gnorm"], OPT2_S2["preclip_gnorm"]),
        "s2_matched_point_m12": (s2["matched_point_m12"],
                                 OPT2_S2["matched_point_m12"]),
        "s3_ce_batch_e194": (s3["ce_batch"], E194_WALK_S3["ce_batch"]),
        "s3_cum_disp_e194": (s3["cum_disp"], E194_WALK_S3["cum_disp"]),
        "s3_gm12_e194": (s3["gm12"], E194_WALK_S3["gm12"]),
        "s3_preclip_gnorm_e194": (s3["preclip_gnorm"],
                                  E194_WALK_S3["preclip_gnorm"]),
        "s3_matched_point_m12_e194": (s3["matched_point_m12"],
                                      E194_WALK_S3["matched_point_m12"]),
        "D_kill": (arm1["stop"]["D_kill"], PATH_DKILL_COMMITTED),
    }
    # the matched-point read's comparator is the CHART's committed anchor
    # (the family convention; opt2's traj rows carry the fp32-texture
    # caveat — measured and reported, never gated at 1e-9)
    repro_diffs = {kk: abs(vv[0] - vv[1]) for kk, vv in repro_rows.items()}
    strict_diffs = {kk: v for kk, v in repro_diffs.items()
                    if kk not in ("s1_matched_point_m12",
                                  "s2_matched_point_m12",
                                  "s3_matched_point_m12_e194")}
    sp_chart_diff = abs(s1["matched_point_m12"] - W024_ADAM_SAMEPOINT)
    dens_diffs = ([abs(a["gm12"] - b[1]) for a, b in
                  zip(arm1["stop"].get("dens", []), OPT2_DENS)]
                  if arm1["stop"].get("dens") else [])
    G_REPRO = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows.items()},
        "dens_gm12_max_abs_diff": (max(dens_diffs) if dens_diffs else None),
        "tol": G_REPRO_TOL,
        "matched_point_comparator": {
            "s1_vs_chart": sp_chart_diff,
            "s2_measured": s2["matched_point_m12"],
            "s2_committed_e194": OPT2_S2["matched_point_m12"],
            "note": "s1 gated vs the chart's committed anchor (opt2's own "
                    "G_SAMEPOINT convention; opt2's traj rows carry the "
                    "T150 fp32-texture caveat); s2/s3 matched-point rows "
                    "gated vs e194's committed walk (same code path)"},
        "pass": bool(arm1["stop"]["kind"] == "kill"
                     and max(strict_diffs.values()) < G_REPRO_TOL
                     and sp_chart_diff < G_SAMEPOINT_TOL
                     and abs(s2["matched_point_m12"]
                             - OPT2_S2["matched_point_m12"]) < G_REPRO_TOL
                     and abs(s3["matched_point_m12"]
                             - E194_WALK_S3["matched_point_m12"]) < G_REPRO_TOL
                     and (not dens_diffs
                          or max(dens_diffs) < G_REPRO_TOL)),
        "note": "THE REBUILT-TRAJECTORY PROVENANCE GATE (Rule 12): the k=1 "
                "walk must reproduce opt2's committed a_sign trajectory "
                "AND e194's committed post-kill s3 row — failure = control "
                "failure, abort",
    }
    log("G_REPRO (vs opt2 committed + e194 s3): strict max|diff| "
        f"{max(strict_diffs.values()):.2e}"
        + (f", dens max|diff| {max(dens_diffs):.2e}" if dens_diffs else "")
        + f", s1 matched-point vs chart {sp_chart_diff:.2e}: "
        + ("PASS" if G_REPRO["pass"] else "FAIL"))

    # ---- G_SAVEDSTATE: the saved opt2 sign-arm checkpoint vs the walked s2
    sck = torch.load(OPT2_SIGN_CK, map_location="cpu", weights_only=False)
    saved_net = TinyGPT(Cfg())
    saved_net.load_state_dict(sck["model"])
    saved_flat = flat_params(saved_net)
    walked_s2 = arm1["endpoints"][2]
    maxdiff = float((saved_flat - walked_s2).abs().max())
    G_SAVEDSTATE = {
        "path": str(OPT2_SIGN_CK),
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
        "max_abs_param_diff": maxdiff,
        "tol": 1e-6,
        "pass": None,
        "note": "THE SAVED-STATE PROVENANCE GATE: opt2's committed "
                "checkpoint must BE the walked k=1 s2 endpoint — theta_1's "
                "chain of custody runs through this gate (theta_2 = "
                "theta_1 - STEP_L2*u1)",
    }
    G_SAVEDSTATE["pass"] = bool(
        G_SAVEDSTATE["meta_gate"]["match"]
        and (G_SAVEDSTATE["walked_s2_flat_md5"] == G_SAVEDSTATE["saved_flat_md5"]
             or maxdiff < 1e-6))
    log(f"G_SAVEDSTATE (opt2_a_sign_s2.pt): meta "
        + ("match" if G_SAVEDSTATE["meta_gate"]["match"] else "DRIFT")
        + f", walked-s2 vs saved max|diff| {maxdiff:.2e}: "
        + ("PASS" if G_SAVEDSTATE["pass"] else "FAIL"))

    # ---- the rays: u1/u2 from the walk's refresh gradients (e192's
    #      construction); G_FRONTS gates them vs e194's committed trace
    g0_w, g1_w, g2_w = arm1["g_fronts"][0], arm1["g_fronts"][1], \
        arm1["g_fronts"][2]
    u1 = (torch.sign(g1_w) / torch.norm(torch.sign(g1_w))).clone()
    u2 = (torch.sign(g2_w) / torch.norm(torch.sign(g2_w))).clone()
    u1_md5 = hashlib.md5(u1.numpy().tobytes()).hexdigest()
    u2_md5 = hashlib.md5(u2.numpy().tobytes()).hexdigest()
    fresh_front_rows = [
        {"front_state": 0, "vs_g_ray": cos64(torch.sign(g0_w), u_g),
         "vs_static_sign": cos64(torch.sign(g0_w), u0),
         "vs_prev_front": None},
        {"front_state": 1, "vs_g_ray": cos64(torch.sign(g1_w), u_g),
         "vs_static_sign": cos64(torch.sign(g1_w), u0),
         "vs_prev_front": cos64(torch.sign(g1_w), torch.sign(g0_w))},
        {"front_state": 2, "vs_g_ray": cos64(torch.sign(g2_w), u_g),
         "vs_static_sign": cos64(torch.sign(g2_w), u0),
         "vs_prev_front": cos64(torch.sign(g2_w), torch.sign(g1_w))},
    ]
    front_diffs = {}
    for fresh, ref in zip(fresh_front_rows, E194_FRONTS):
        for k in ("vs_g_ray", "vs_static_sign", "vs_prev_front"):
            if ref[k] is None:
                continue
            front_diffs[f"t{ref['front_state']}.{k}"] = \
                abs(fresh[k] - ref[k])
    ray_geometry = {
        "cos_u0_u1": cos64(u0, u1),
        "cos_u0_u2": cos64(u0, u2),
        "cos_u1_u2": cos64(u1, u2),
        "committed_anchors": {"cos_sign0_sign1_e194":
                                  E194_FRONTS[1]["vs_static_sign"],
                              "cos_sign1_sign2_e194":
                                  E194_FRONTS[2]["vs_prev_front"],
                              "cos_sign0_sign2_e194":
                                  E194_FRONTS[2]["vs_static_sign"]},
        "iso_floor": 1.0 / (N_PARAM ** 0.5),
        "note": "the rotated rays' mutual geometry (fp64 cosines); the "
                "committed e194 anchors are the un-normalized sign-form "
                "reads — cosines agree to fp32-normalization texture",
    }
    G_FRONTS = {
        "fresh_front_rows": fresh_front_rows,
        "committed_front_rows": E194_FRONTS,
        "max_abs_diff": max(front_diffs.values()),
        "diffs": front_diffs,
        "tol": G_FRONT_TOL,
        "u1_md5": u1_md5, "u2_md5": u2_md5,
        "pass": bool(max(front_diffs.values()) < G_FRONT_TOL),
        "note": "THE ROTATED-RAY PROVENANCE GATE: the walk's fresh fronts "
                "(t=0,1,2) must reproduce e194's committed front trace "
                "bit-class — u1/u2's chain of custody",
    }
    log(f"G_FRONTS (t=0,1,2 vs e194 committed): max|diff| "
        f"{max(front_diffs.values()):.2e}: "
        + ("PASS" if G_FRONTS["pass"] else "FAIL"))
    log(f"ray geometry: cos(u0,u1) {ray_geometry['cos_u0_u1']:+.4f}, "
        f"cos(u0,u2) {ray_geometry['cos_u0_u2']:+.4f}, "
        f"cos(u1,u2) {ray_geometry['cos_u1_u2']:+.4f}")

    if not (G_REPRO["pass"] and G_SAVEDSTATE["pass"] and G_FRONTS["pass"]):
        stub["gates"]["G_REPRO"] = G_REPRO
        stub["gates"]["G_SAVEDSTATE"] = G_SAVEDSTATE
        stub["gates"]["G_FRONTS"] = G_FRONTS
        write_partial("CONTROL FAILURE — provenance gates failed")
        raise RuntimeError("provenance gates FAILED (G_REPRO/G_SAVEDSTATE/"
                           "G_FRONTS) — abort before any read is believed")
    stub["gates"]["G_REPRO"] = G_REPRO
    stub["gates"]["G_SAVEDSTATE"] = G_SAVEDSTATE
    stub["gates"]["G_FRONTS"] = G_FRONTS

    # ---- the anchors
    theta1 = arm1["endpoints"][1]
    theta1 = theta1.clone()
    evl_t1 = copy.deepcopy(net0)
    load_flat(evl_t1, theta1)
    evl_t1.eval()
    t1_gm12 = battery_cell(evl_t1, gm12_ids, zid)["mean_pz"]
    t1_ce_r = ce_fixed_cpu(evl_t1, *r_eval_xy)
    G_TH1ANCHOR = {
        "cum_disp": float(torch.norm(theta1 - theta0)),
        "committed_s1_cum_disp": OPT2_S1["cum_disp"],
        "gm12_measured": t1_gm12,
        "gm12_committed_s1": OPT2_S1["gm12"],
        "gm12_diff": abs(t1_gm12 - OPT2_S1["gm12"]),
        "ce_r_measured": t1_ce_r,
        "tol": G_REPRO_TOL,
        "pass": bool(abs(float(torch.norm(theta1 - theta0))
                         - OPT2_S1["cum_disp"]) < G_REPRO_TOL
                     and abs(t1_gm12 - OPT2_S1["gm12"]) < G_REPRO_TOL),
        "note": "theta_1 = the walk's step-1 endpoint; proven by cum_disp + "
                "the s1 battery read + the theta_2-md5 chain (G_SAVEDSTATE)",
    }
    log(f"G_TH1ANCHOR (theta_1): cum {G_TH1ANCHOR['cum_disp']:.7f}, "
        f"g-12 {t1_gm12:.10f} vs committed {OPT2_S1['gm12']:.10f}: "
        + ("PASS" if G_TH1ANCHOR["pass"] else "FAIL"))
    if not G_TH1ANCHOR["pass"]:
        stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR
        write_partial("CONTROL FAILURE — theta_1 anchor gate failed")
        raise RuntimeError("theta_1 anchor gate FAILED — abort")
    stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR

    # ---- the dual-estimator matched-point alignment reads (T150: every
    #      read states its evaluation point — here, AT the anchor)
    m12_grad_t1 = fact_grad(evl_t1, gm12_ids, zid)   # grad m12 AT theta_1
    delta2_form = sign_update(g1_w, STEP_L2)          # the walked step-2 form
    align_gate_diff = abs(cos64(delta2_form, m12_grad_t1)
                          - OPT2_S2["matched_point_m12"])
    alignment_reads = {
        "evaluation_points": "cos(jump direction, grad m12(anchor)) — the "
                             "matched-point convention, evaluated AT the "
                             "anchor (T150); delta-form = cos(delta_s, grad "
                             "m12(theta_{s-1})) = e194's committed rows",
        "root_anchor": {
            "cos_neg_u0": cos64(-u0, root_m12_grad),
            "cos_neg_u1": cos64(-u1, root_m12_grad),
            "cos_neg_u2": cos64(-u2, root_m12_grad),
            "committed_s1_delta_form": OPT2_S1["matched_point_m12"],
            "delta_form_diff_s1": abs(cos64(sign_update(g0_w, STEP_L2),
                                            root_m12_grad)
                                       - s1["matched_point_m12"]),
        },
        "theta1_anchor": {
            "cos_neg_u0": cos64(-u0, m12_grad_t1),
            "cos_neg_u1": cos64(-u1, m12_grad_t1),
            "cos_neg_u2": cos64(-u2, m12_grad_t1),
            "committed_s2_delta_form": OPT2_S2["matched_point_m12"],
            "delta_form_diff_s2": align_gate_diff,
        },
        "texture_note": "unit-ray form vs delta-form cosines differ at "
                        "fp32-multiplication texture (~1e-9); the delta-form "
                        "gates the machinery bit-class, the unit-ray form is "
                        "the reported read",
    }
    G_ALIGN = {
        "delta_form_diff_s2": align_gate_diff,
        "delta_form_diff_s1": alignment_reads["root_anchor"]["delta_form_diff_s1"],
        "tol_unit_form": G_ALIGN_TOL,
        "pass": bool(align_gate_diff < G_REPRO_TOL
                     and alignment_reads["root_anchor"]["delta_form_diff_s1"]
                     < G_REPRO_TOL),
        "note": "the theta_1 fact gradient gates bit-class vs e194's "
                "committed s2 matched-point row (same form) — the unit-ray "
                "alignment reads ride on this machinery",
    }
    log(f"G_ALIGN (matched-point machinery): s1 delta-form |d| "
        f"{alignment_reads['root_anchor']['delta_form_diff_s1']:.2e}, s2 "
        f"delta-form |d| {align_gate_diff:.2e}: "
        + ("PASS" if G_ALIGN["pass"] else "FAIL"))
    if not G_ALIGN["pass"]:
        stub["gates"]["G_ALIGN"] = G_ALIGN
        write_partial("CONTROL FAILURE — alignment machinery gate failed")
        raise RuntimeError("alignment machinery gate FAILED — abort")
    stub["gates"]["G_ALIGN"] = G_ALIGN

    phase0 = {
        "walk_journal": [{kk: vv for kk, vv in r.items() if kk != "x_md5"}
                         for r in j1],
        "front_trace": arm1["front_trace"],
        "stop": arm1["stop"], "max_l2_dev": arm1["max_l2_dev"],
        "zeph": arm1["zeph"], "train_seconds": arm1["train_seconds"],
        "post_kill_disclosure": "step 3 is read-only continuation (its "
                                "gradient's SIGN is the u2 ray; nothing "
                                "post-kill adjudicates)",
    }
    stub["phases_partial"]["0_walk_rebuild"] = E43.jsonable(phase0)
    write_partial("phase 0 complete (walk rebuilt + all provenance gates)")
    rays_meta = [
        {"key": "u0", "label": "sign(g_0) — the original static sign ray",
         "u_md5": u0_md5,
         "provenance": "e192's R2 construction verbatim (md5-gated, G_R2DIR)"},
        {"key": "u1", "label": "sign(g_1) — THE ROTATED RAY (t=1 fresh front)",
         "u_md5": u1_md5,
         "provenance": "the rebuilt k=1 walk's step-2 refresh gradient "
                       "(theta_1, batch 2, post-clip); gated vs e194's "
                       "committed front trace (G_FRONTS)"},
        {"key": "u2", "label": "sign(g_2) — the t=2 front (rotation's "
                               "continuation)",
         "u_md5": u2_md5,
         "provenance": "the rebuilt walk's step-3 refresh gradient (theta_2 "
                       "= the saved opt2 ckpt, batch 3, post-kill "
                       "continuation read); gated vs e194 (G_FRONTS)"},
    ]
    anchors_meta = {
        "theta_0": {"label": "the e131 consolidated root",
                    "gm12": root_cells["gm12"], "ce_r": root_cells["ce_r"],
                    "provenance": f"runs/checkpoints/{ROOT_CK}, gated vs "
                                  "e151 (G_ROOT)"},
        "theta_1": {"label": "the one-stepped state (the wash trajectory's)",
                    "gm12": t1_gm12, "ce_r": t1_ce_r,
                    "cum_disp": G_TH1ANCHOR["cum_disp"],
                    "provenance": "the rebuilt walk's step-1 endpoint "
                                  "(G_REPRO + G_TH1ANCHOR + the theta_2 "
                                  "md5 chain)"},
    }

    # =====================================================================
    # PHASE A — the three ray families FROM THE ROOT
    # =====================================================================
    log("=" * 78)
    log(f"PHASE A — FROM THE ROOT: three families x {len(D_GRID)}+1 grid "
        f"points (+exact-landing gate points)")
    evp = copy.deepcopy(net0)
    ce_ds_common = ([0.0] + [round(CE_EVERY * i, 2)
                             for i in range(1, int(3.0 / CE_EVERY) + 1)]
                    if CE_EVERY else [0.0])
    families = {}   # key -> dict(anchor, u, dgrid, ce_ds, gate_points)

    def build_family(key, anchor_key, u, extra_ds, gate_points):
        base = [0.0] + list(D_GRID)
        ds = base + [d for d in extra_ds
                     if not any(abs(d - g) < 1e-9 for g in base)]
        ce_ds = (list(ce_ds_common) if CE_EVERY else [0.0])
        return {"key": key, "anchor_key": anchor_key, "u": u,
                "dgrid": sorted(set(round(d, 10) for d in ds)),
                "ce_ds": ce_ds,
                "gate_points": gate_points}

    fam_root_u0 = build_family("root_u0", "theta_0", u0,
                               [STEP_L2],
                               {str(STEP_L2): {"expect": E192_A0_SIGNRAY_GM12,
                                               "tol": G_STATIC_TOL}})
    fam_root_u0["ce_ds"] = sorted(set(fam_root_u0["ce_ds"] + [1.5, 2.0, 2.5]))
    fam_root_u1 = build_family("root_u1", "theta_0", u1, [], {})
    fam_root_u2 = build_family("root_u2", "theta_0", u2, [], {})
    fam_th1_u0 = build_family("th1_u0", "theta_1", u0, [], {})
    fam_th1_u1 = build_family("th1_u1", "theta_1", u1, [STEP_L2],
                              {str(STEP_L2): {"expect": OPT2_S2["gm12"],
                                              "tol": G_LANDING_TOL}})
    fam_th1_u2 = build_family("th1_u2", "theta_1", u2, [], {})
    fam_order = [fam_root_u0, fam_root_u1, fam_root_u2,
                 fam_th1_u0, fam_th1_u1, fam_th1_u2]
    anchors = {"theta_0": theta0, "theta_1": theta1}

    profiles = {}
    for fam in fam_order:
        tA = time.time()
        anchor_flat = anchors[fam["anchor_key"]]
        rows = static_profile(evp, anchor_flat, fam["u"], fam["dgrid"],
                              gm12_ids, zid, r_eval_xy, fam["ce_ds"])
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
            "gate_points": fam["gate_points"],
            "rows": rows, "D_kill": edge, "any_upcross": ups,
            "seconds": round(time.time() - tA, 1),
        }
        profiles[fam["key"]] = families[fam["key"]]
        stub["phases_partial"][f"prof_{fam['key']}"] = E43.jsonable(
            {"D_kill": edge, "rows": rows, "anchor": fam["anchor_key"]})
        write_partial(f"profile {fam['key']} complete")
        log(f"  {fam['key']}: {len(rows)} pts in {families[fam['key']]['seconds']}s"
            f" — D_kill "
            + (f"{edge:.4f}" if edge is not None else "None (no kill <= 3.0)")
            + (f", upcrosses {ups}" if ups else ""))

    # ---- G_ROOTPROF: the root-u0 family vs e194's committed fine rows
    fine_by_D = {round(r["D"], 4): r["gm12"] for r in e194_fine}
    xc = []
    for row in families["root_u0"]["rows"]:
        key = round(row["D"], 4)
        if key in fine_by_D:
            xc.append({"D": key, "gm12_measured": row["gm12"],
                       "gm12_e194": fine_by_D[key],
                       "abs_diff": abs(row["gm12"] - fine_by_D[key])})
    e192_xc = []
    for row in families["root_u0"]["rows"]:
        key = round(row["D"], 4)
        if key in E192_R2_GATE and "ce_r" in row:
            e192_xc.append({"D": key,
                            "gm12_measured": row["gm12"],
                            "gm12_e192": E192_R2_GATE[key]["gm12"],
                            "ce_r_measured": row["ce_r"],
                            "ce_r_e192": E192_R2_GATE[key]["ce_r"]})
    gate_pts = []
    for row in families["root_u0"]["rows"]:
        if abs(row["D"] - STEP_L2) < 1e-9:
            gate_pts.append({"D": row["D"], "gm12": row["gm12"],
                             "expect": E192_A0_SIGNRAY_GM12,
                             "abs_diff": abs(row["gm12"]
                                             - E192_A0_SIGNRAY_GM12)})
    G_ROOTPROF = {
        "e194_fine_crosscheck": xc, "n_shared": len(xc),
        "max_abs_diff": (max(r["abs_diff"] for r in xc) if xc else None),
        "e192_R2_crosscheck": e192_xc,
        "step_l2_landing": (gate_pts[0] if gate_pts else None),
        "tol": G_STATIC_TOL,
        "pass": bool(len(xc) == (21 if not SMOKE else len(xc))
                     and all(r["abs_diff"] < G_STATIC_TOL for r in xc)
                     and gate_pts
                     and gate_pts[0]["abs_diff"] < G_STATIC_TOL),
        "note": "THE STATIC-PROFILE MACHINERY GATE: the root-u0 family must "
                "reproduce e194's committed 21 fine rows (+ e192's R2 at "
                "1.5/2.0/2.5, cross-reported) and the D=STEP_L2 landing "
                "(= opt2's committed s1 read) before any new point is "
                "believed",
    }
    log(f"G_ROOTPROF (root-u0 vs e194 fine): {len(xc)} shared pts, max|diff| "
        f"{(G_ROOTPROF['max_abs_diff'] or 0):.2e}, STEP_L2 landing |d| "
        f"{(gate_pts[0]['abs_diff'] if gate_pts else float('nan')):.2e}: "
        + ("PASS" if G_ROOTPROF["pass"] else "FAIL"))

    # ---- G_TH1PROF: the theta_1 families' anchor + landing gates
    th1_anchor_row = families["th1_u0"]["rows"][0]     # D=0 is shared
    th1_anchor_rows = [families[f"th1_{r}"]["rows"][0] for r in ("u0", "u1", "u2")]
    anchor_ok = all(abs(r["gm12"] - OPT2_S1["gm12"]) < G_REPRO_TOL
                    for r in th1_anchor_rows)
    landing = None
    for row in families["th1_u1"]["rows"]:
        if abs(row["D"] - STEP_L2) < 1e-9:
            landing = {"D": row["D"], "gm12": row["gm12"],
                       "expect": OPT2_S2["gm12"],
                       "abs_diff": abs(row["gm12"] - OPT2_S2["gm12"])}
    G_TH1PROF = {
        "anchor_d0_gm12": [r["gm12"] for r in th1_anchor_rows],
        "anchor_expect": OPT2_S1["gm12"], "anchor_ok": anchor_ok,
        "u1_step_l2_landing": landing,
        "tol_anchor": G_REPRO_TOL, "tol_landing": G_LANDING_TOL,
        "pass": None,
        "note": "THE THETA_1-PANEL MACHINERY GATE: all three theta_1 "
                "families' D=0 rows must reproduce opt2's committed s1 "
                "read bit-class; the th1-u1 family's D=STEP_L2 landing must "
                "reproduce the committed s2 kill read (fp32-norm ray vs "
                "fp64-norm walked step — texture class)",
    }
    G_TH1PROF["pass"] = bool(anchor_ok and landing
                             and landing["abs_diff"] < G_LANDING_TOL)
    log(f"G_TH1PROF (theta_1 panel): anchor |d| "
        + ", ".join(f"{abs(r['gm12'] - OPT2_S1['gm12']):.2e}"
                    for r in th1_anchor_rows)
        + (f", u1 STEP_L2 landing |d| {landing['abs_diff']:.2e}"
           if landing else ", NO LANDING POINT")
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
    # ADJUDICATION (frozen clauses; composite order FLEEING-IS-LETHAL ->
    # ROTATION-IS-RELATIVE -> FLAT-TERRAIN -> GRADED; no shopping)
    # =====================================================================
    dk = {k: families[k]["D_kill"] for k in families}
    spread = lambda ds: (None if (any(v is None for v in ds) or not ds)
                         else (max(ds) - min(ds)) / max(ds))

    def rel(a, b):     # a relative to b (None-safe)
        return None if (a is None or b is None) else a / b

    root_spread = spread([dk["root_u0"], dk["root_u1"], dk["root_u2"]])
    th1_spread = spread([dk["th1_u0"], dk["th1_u1"], dk["th1_u2"]])
    fl_pair = {"root_u1_vs_root_u0": rel(dk["root_u1"], dk["root_u0"]),
               "th1_u1_vs_th1_u0": rel(dk["th1_u1"], dk["th1_u0"])}
    fires_fleeing = bool(dk["root_u0"] is not None
                         and dk["root_u1"] is not None
                         and dk["root_u1"] <= MATERIAL_RATIO * dk["root_u0"])
    fires_relative = bool(root_spread is not None and root_spread <= FLAT_RATIO
                          and dk["th1_u0"] is not None
                          and dk["th1_u1"] is not None
                          and dk["th1_u1"] <= MATERIAL_RATIO * dk["th1_u0"])
    fires_flat = bool(root_spread is not None and root_spread <= FLAT_RATIO
                      and th1_spread is not None and th1_spread <= FLAT_RATIO)
    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "FLEEING_IS_LETHAL": {
                "fires": fires_fleeing,
                "detail": {"D_kill_root_u0": dk["root_u0"],
                           "D_kill_root_u1": dk["root_u1"],
                           "ratio_u1_over_u0": fl_pair["root_u1_vs_root_u0"],
                           "material_ratio_bar": MATERIAL_RATIO},
            },
            "ROTATION_IS_RELATIVE": {
                "fires": fires_relative,
                "detail": {"root_panel_spread": root_spread,
                           "near_equivalent_bar": FLAT_RATIO,
                           "D_kill_th1_u0": dk["th1_u0"],
                           "D_kill_th1_u1": dk["th1_u1"],
                           "ratio_u1_over_u0_th1": fl_pair["th1_u1_vs_th1_u0"],
                           "material_ratio_bar": MATERIAL_RATIO},
            },
            "FLAT_TERRAIN": {
                "fires": fires_flat,
                "detail": {"root_panel_spread": root_spread,
                           "th1_panel_spread": th1_spread,
                           "flat_bar": FLAT_RATIO,
                           "D_kills": dk},
            },
            "GRADED": {"fires": not (fires_fleeing or fires_relative
                                     or fires_flat)},
        }
        fmt = lambda v: ("None" if v is None else f"{v:.4f}")
        if fires_fleeing:
            verdict = "FLEEING-IS-LETHAL"
            clause = (f"the rotated ray sign(g_1) from the ROOT kills at D "
                      f"{fmt(dk['root_u1'])} vs sign(g_0)'s "
                      f"{fmt(dk['root_u0'])} "
                      f"({(1 - fl_pair['root_u1_vs_root_u0']) * 100:.1f}% "
                      f"lower, bar 15%) — the fleeing direction is itself "
                      f"the more lethal one; the terrain's lethality "
                      f"concentrates where the support fled.")
        elif fires_relative:
            verdict = "ROTATION-IS-RELATIVE"
            clause = (f"the rays are near-equivalent from the root "
                      f"(spread {root_spread * 100:.1f}% <= 10%) but "
                      f"sign(g_1)-from-theta_1 kills at D "
                      f"{fmt(dk['th1_u1'])} vs sign(g_0)-from-theta_1's "
                      f"{fmt(dk['th1_u0'])} "
                      f"({(1 - fl_pair['th1_u1_vs_th1_u0']) * 100:.1f}% "
                      f"lower, bar 15%) — the lethality is RELATIVE to the "
                      f"state (the terrain rotates under the walker), not a "
                      f"fixed direction.")
        elif fires_flat:
            verdict = "FLAT-TERRAIN"
            clause = (f"the three rays are within ~10% everywhere (root "
                      f"spread {root_spread * 100:.1f}%, theta_1 spread "
                      f"{th1_spread * 100:.1f}%) — the fleeing-support "
                      f"reading needs a different instrument; reported "
                      f"honestly.")
        else:
            verdict = "GRADED"
            clause = ("any mix — the profiles verbatim: D_kill by family = "
                      + ", ".join(f"{k}:{fmt(v)}" for k, v in dk.items())
                      + f"; root spread "
                      + (f"{root_spread * 100:.1f}%" if root_spread is not None
                         else "undefined (a None kill)")
                      + ", theta_1 spread "
                      + (f"{th1_spread * 100:.1f}%" if th1_spread is not None
                         else "undefined (a None kill)")
                      + "; ratios " + ", ".join(
                          f"{k}={fmt(v) if v is not None else 'None'}"
                          for k, v in fl_pair.items())
                      + ".")
    log("=" * 78)
    log(f"E195 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e195_rotated_ray",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("WHERE did the lethal support flee TO? — static graded "
                     "jumps along sign(g_0)/sign(g_1)/sign(g_2) from the "
                     "root AND from the one-stepped state theta_1: is the "
                     "fleeing direction itself the more lethal one "
                     "(FLEEING-IS-LETHAL), is the lethality relative to the "
                     "state (ROTATION-IS-RELATIVE), or is the terrain flat "
                     "across rays (FLAT-TERRAIN)?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "the e185 wash stream VERBATIM (opt2/e194's "
                             "conventions); the k=1 sign walk rebuilt "
                             "bit-exactly as the ray/state factory",
            "rulers": "g-12 install-60 battery (mean p(Z)), SHUT bar "
                      f"{SHUT_BAR} — the standard ruler; dual currency: "
                      "displacement in L2 D and per-coordinate RMS "
                      f"(D/{RMS_DENOM:.4f}), plus ce_r at decimated Ds",
            "grid": {"D": D_GRID, "extras": {"root_u0": [STEP_L2],
                                             "th1_u1": [STEP_L2]},
                     "note": "D=0 anchor rows included in every family; "
                             "e194's fine grid (1.5-2.5) re-gated inside "
                             "root-u0 and the shoulders 0.05-1.5 / 2.5-3.0 "
                             "extended"},
            "rays": rays_meta,
            "anchors": anchors_meta,
            "input_seed": FREEZE_SEED,
            "matched_step_L2": {"value": STEP_L2,
                                "provenance": "opt1 A0's committed step-1 "
                                "displacement (recomputed at t=0 and "
                                "bit-gated in G_T0)"},
            "death_direction": "u_g = e191's committed static g-ray; the "
                               "kill ray is descent (-u_g)",
        },
        "gates": stub["gates"],
        "phase0_walk_rebuild": phase0,
        "profiles": {k: {"anchor": v["anchor"], "ray": v["ray"],
                         "label": v["label"], "u_md5": v["u_md5"],
                         "placement": v["placement"],
                         "n_points": v["n_points"],
                         "gate_points": v["gate_points"],
                         "rows": v["rows"], "D_kill": v["D_kill"],
                         "any_upcross": v["any_upcross"],
                         "seconds": v["seconds"]}
                     for k, v in families.items()},
        "ray_geometry": ray_geometry,
        "alignment_reads": alignment_reads,
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "composite_order": "FLEEING-IS-LETHAL -> ROTATION-IS-RELATIVE "
                               "-> FLAT-TERRAIN -> GRADED (frozen before "
                               "compute)",
            "constants": {"SHUT_BAR": SHUT_BAR, "D_grid": D_GRID,
                          "MATERIAL_RATIO": MATERIAL_RATIO,
                          "FLAT_RATIO": FLAT_RATIO, "STEP_L2": STEP_L2},
        },
        "references": {
            "e194_sign_front": {"metrics": "runs/e194/metrics.json",
                                "refined_static_edge": REFINED_STATIC_EDGE,
                                "path_kill_D": PATH_DKILL_COMMITTED,
                                "committed_fronts": E194_FRONTS,
                                "role": "the parent: the front trace + the "
                                        "fine static grid + the walked "
                                        "rows (loaded committed, gated)"},
            "opt2_sign_path": {"metrics": "runs/opt2/metrics.json",
                               "D_kill": PATH_DKILL_COMMITTED,
                               "ckpt": str(OPT2_SIGN_CK),
                               "role": "the parent path (rebuilt, gated "
                                       "G_REPRO/G_SAVEDSTATE — never "
                                       "re-adjudicated)"},
            "e192_terrain": {"metrics": "runs/e192/metrics.json",
                             "static_sign_edge_coarse": SIGN_EDGE_COMMITTED,
                             "R2_md5": E192_R2_U_MD5,
                             "role": "the static rays (loaded committed, "
                                     "crosschecked in G_ROOTPROF)"},
            "opt1c_raw": {"D_kill": RAW_DKILL_COMMITTED},
            "opt1_A0": {"D_kill": A0_DKILL_COMMITTED},
            "e_chart_samepoint": {"adam": W024_ADAM_SAMEPOINT,
                                  "raw": W024_RAW_SAMEPOINT},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 organism/fact (the e131 consolidated line), "
                            "ONE stream (seed 10902, md5-gated at every "
                            "consumed step); the t=1/t=2 states and fronts "
                            "are the WASH TRAJECTORY'S OWN — a single-path "
                            "biography, not a population claim"),
            "post_kill_disclosure": ("u2's gradient was read on the walk's "
                                     "step-3 continuation (post-kill, "
                                     "read-only); as a RAY probe it is "
                                     "legitimate but its 't=2' clock is the "
                                     "single wash path's"),
            "openness": ("WHAT EACH FAMILY GUARANTEES: NOTHING — every "
                         "family is a static graded jump at lethal scale "
                         "from a state that could kill anywhere; cross-"
                         "panel D comparisons are NOT adjudicated (theta_1 "
                         "starts 0.41 closer to the bar by construction); "
                         "that openness is the point"),
            "estimator_lesson": ("every alignment read states its evaluation "
                                 "point (matched-point AT the anchor; "
                                 "delta-form gates vs e194's committed rows "
                                 "bit-class, unit-ray form reported) — "
                                 "T150's lesson"),
            "projections_never_adjudicate": ("D_kill is a linear-in-D "
                                             "interpolation on measured "
                                             "grid points; the 15%/10% "
                                             "margins were frozen before "
                                             "compute"),
            "float_texture": ("CPU fp32 texture, this process, 4 threads "
                              "(the e185-era reduction order); the k=1 "
                              "walk reproduced opt2's committed trajectory "
                              f"to {max(strict_diffs.values()):.1e} and the "
                              "saved s2 checkpoint md5-exactly"),
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_rotated_ray(rd / "e195_rotated_ray.png", families, dk)
    plot_kill_summary(rd / "e195_kill_summary.png", families, dk,
                      ray_geometry, alignment_reads)
    log(f"outputs: {rd / 'metrics.json'} + 2 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

RAY_STYLE = {
    "u0": {"color": "navy", "label": "sign(g_0) — the original static ray"},
    "u1": {"color": "crimson",
           "label": "sign(g_1) — THE ROTATED RAY (the fled-toward front)"},
    "u2": {"color": "darkorange", "label": "sign(g_2) — the t=2 front"},
}


def _dual_axis(ax):
    ax.set_xlabel(r"static displacement $D = \|\theta_D - \theta_{anchor}\|_2$")
    axr = ax.secondary_xaxis(
        "top", functions=(lambda d: d / RMS_DENOM,
                          lambda r: r * RMS_DENOM))
    axr.set_xlabel("per-coordinate RMS (the second currency)")


def plot_rotated_ray(path, families, dk):
    """THE ROTATED-RAY MAP: three rays from the root + the from-theta_1
    panel, dual displacement currency."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, panel, anchor_lbl in (
            (axes[0], ("root_u0", "root_u1", "root_u2"), "FROM THE ROOT"),
            (axes[1], ("th1_u0", "th1_u1", "th1_u2"),
             "FROM THE ONE-STEPPED STATE $\\theta_1$")):
        for key in panel:
            fam = families[key]
            st = RAY_STYLE[fam["ray"]]
            ax.plot([r["D"] for r in fam["rows"]],
                    [r["gm12"] for r in fam["rows"]], "o-", ms=2.6,
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
        ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
        ax.set_ylim(-0.03, 1.02)
        ax.set_title(anchor_lbl, fontsize=10.5)
    axes[0].axvline(REFINED_STATIC_EDGE, color="dimgray", ls="-.", lw=1.3)
    axes[0].text(REFINED_STATIC_EDGE + 0.02, 0.5,
                 f"e194 refined static edge {REFINED_STATIC_EDGE:.4f}",
                 fontsize=6.8, rotation=90, color="dimgray")
    axes[0].axvline(PATH_DKILL_COMMITTED, color="crimson", ls=":", lw=1.2,
                    alpha=0.6)
    axes[0].text(PATH_DKILL_COMMITTED - 0.02, 0.72,
                 f"opt2 path kill {PATH_DKILL_COMMITTED:.4f}",
                 fontsize=6.8, rotation=90, color="crimson", ha="right")
    # the walked step-2 landing on the th1-u1 family (the committed kill read)
    axes[1].plot([STEP_L2], [OPT2_S2["gm12"]], "*", ms=15, color="gold",
                 markeredgecolor="k", zorder=5,
                 label="the walked step-2 landing (= theta_2, committed "
                       f"{OPT2_S2['gm12']:.1e})")
    axes[0].legend(fontsize=7.0, loc="upper right")
    axes[1].legend(fontsize=7.0, loc="upper right")
    fig.suptitle("E195 — THE ROTATED-RAY TERRAIN: where did the lethal "
                 "support flee to? (static graded jumps, three ray "
                 "families, two anchors; e194's front chain, bit-gated)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_kill_summary(path, families, dk, ray_geometry, alignment_reads):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    ax = axes[0]
    keys = ["root_u0", "root_u1", "root_u2", "th1_u0", "th1_u1", "th1_u2"]
    vals = [dk[k] for k in keys]
    cols = [RAY_STYLE[k.split("_")[1]]["color"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, [v if v is not None else 0.0 for v in vals], color=cols,
           alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, (v if v is not None else 0) + 0.05,
                f"{v:.4f}" if v is not None else "no kill\n<= 3.0",
                ha="center", fontsize=8)
    # the frozen 15% margins on the two bar comparators
    if dk["root_u0"] is not None:
        ax.plot([0.6, 2.4], [MATERIAL_RATIO * dk["root_u0"]] * 2,
                ls=":", lw=1.6, color="navy")
        ax.text(2.42, MATERIAL_RATIO * dk["root_u0"],
                f"0.85 x root-u0 = {MATERIAL_RATIO * dk['root_u0']:.3f}",
                fontsize=7, color="navy", va="center")
    if dk["th1_u0"] is not None:
        ax.plot([3.6, 5.4], [MATERIAL_RATIO * dk["th1_u0"]] * 2,
                ls=":", lw=1.6, color="navy")
        ax.text(5.42, MATERIAL_RATIO * dk["th1_u0"],
                f"0.85 x th1-u0 = {MATERIAL_RATIO * dk['th1_u0']:.3f}",
                fontsize=7, color="navy", va="center")
    ax.axvline(2.5, color="k", lw=0.8, ls="--")
    ax.text(2.5, 2.95, "panel switch", fontsize=7.5, ha="center")
    ax.axhline(REFINED_STATIC_EDGE, color="dimgray", ls="-.", lw=1.2)
    ax.text(0.02, REFINED_STATIC_EDGE + 0.04,
            f"e194 refined static edge {REFINED_STATIC_EDGE:.4f}",
            fontsize=7, color="dimgray")
    ax.axhline(PATH_DKILL_COMMITTED, color="crimson", ls=":", lw=1.2)
    ax.text(0.02, PATH_DKILL_COMMITTED - 0.16,
            f"opt2 path kill {PATH_DKILL_COMMITTED:.4f}", fontsize=7,
            color="crimson")
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("D_kill (first 0.27 downcrossing, interpolated)")
    ax.set_ylim(0, 3.15)
    ax.set_title("THE KILL LADDER — the frozen 15% margins (0.85 lines)",
                 fontsize=10)

    ax = axes[1]
    geo_rows = [
        ("cos(u0,u1)", ray_geometry["cos_u0_u1"],
         ray_geometry["committed_anchors"]["cos_sign0_sign1_e194"]),
        ("cos(u0,u2)", ray_geometry["cos_u0_u2"],
         ray_geometry["committed_anchors"]["cos_sign0_sign2_e194"]),
        ("cos(u1,u2)", ray_geometry["cos_u1_u2"],
         ray_geometry["committed_anchors"]["cos_sign1_sign2_e194"]),
        ("align root: cos(-u0, gm12)", 
         alignment_reads["root_anchor"]["cos_neg_u0"], None),
        ("align root: cos(-u1, gm12)",
         alignment_reads["root_anchor"]["cos_neg_u1"], None),
        ("align root: cos(-u2, gm12)",
         alignment_reads["root_anchor"]["cos_neg_u2"], None),
        ("align th1: cos(-u0, gm12)",
         alignment_reads["theta1_anchor"]["cos_neg_u0"], None),
        ("align th1: cos(-u1, gm12)",
         alignment_reads["theta1_anchor"]["cos_neg_u1"], None),
        ("align th1: cos(-u2, gm12)",
         alignment_reads["theta1_anchor"]["cos_neg_u2"], None),
    ]
    labels = [r[0] for r in geo_rows]
    vals2 = [r[1] for r in geo_rows]
    cols2 = (["navy", "navy", "navy", "seagreen", "crimson", "darkorange",
              "seagreen", "crimson", "darkorange"])
    ax.barh(np.arange(len(geo_rows)), vals2, color=cols2, alpha=0.85)
    for i, r in enumerate(geo_rows):
        txt = f"{r[1]:+.4f}"
        if r[2] is not None:
            txt += f"  (e194 {r[2]:+.4f})"
        ax.text(r[1] + (0.012 if r[1] >= 0 else -0.012), i, txt,
                va="center", fontsize=7.4,
                ha="left" if r[1] >= 0 else "right")
    ax.set_yticks(np.arange(len(geo_rows)))
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(1.0 / (N_PARAM ** 0.5), color="tab:purple", ls="--", lw=1.0)
    ax.text(0.002, -0.9, "isotropic floor", fontsize=6.5,
            color="tab:purple", rotation=90)
    ax.set_xlim(-0.35, 0.75)
    ax.invert_yaxis()
    ax.set_title("RAY GEOMETRY + the matched-point alignment reads "
                 "(evaluated AT the anchor)", fontsize=9.5)
    fig.suptitle("E195 — the kill ladder and the rays' geometry/alignment",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
