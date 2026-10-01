"""E194 — THE SIGN-FRONT MECHANISM (R60-ideator's cell; the day's one
unexplained mismatch: opt2's recomputed sign path kills at 1.75, BELOW its
own static ray's 2.5 — the mirror of e192's causal re-orientation finding;
dynamics cut both ways).

WHY: opt2 (T151) named the lethal object — the |g|-weighted front — and
left one inversion unowned: the SIGN PATH (front recomputed every step at
matched per-step L2 1.6543) kills at D 1.7496 while the FROZEN sign ray
(e192's R2 static profile) kills at 2.5. The trajectory is MORE lethal per
displacement than its own shadow. WHERE does the extra lethality live?
Three candidates, R60-ideator's discriminators: (a) FRONT-CHASES — each
fresh sign(g_t) overlaps the remaining lethal subspace better than the
frozen ray; (b) FRONT-ACCUMULATES — consecutive fresh fronts are mutually
non-orthogonal, displacement concentrating in a shrinking subspace (the
Adam-path comparison); (c) STATIC-ARTIFACT — the 2.5 static edge is a
coarse-grid artifact (e192's grid had NO point between 1.5 and 2.0) and
the true sign-class edge is ~1.75 — the path never beat its ray.

THE CELL (eval-only CPU + short fresh arms): on the saved opt2 sign-arm
state (runs/checkpoints/opt2_a_sign_s2.pt) + the e131 root:
  (1) THE FRONT-OVERLAP READ — at each step of the sign path (REBUILT
      bit-exactly and gated vs opt2's committed trajectory AND the saved
      s2 checkpoint): cos(sign(g_t), the g-ray direction) — the front's
      overlap with the death ray (identity: the step is -sign(g_t), the
      death ray is -u_g, so cos(sign(g_t), u_g) IS the step-vs-death
      overlap); cos(sign(g_t), sign(g_{t-1})) — front persistence; co-
      reads cos(front, fresh g_t direction) and the dual-estimator fact
      alignments (matched-point + post-step + root-anchored; T150's
      lesson: every alignment read states its evaluation point). The
      walk CONTINUES 4 steps past the kill (read-only, disclosed, never
      adjudicates) so the trace has more than the 2 pre-kill points.
  (2) THE STATIC-ARTIFACT CHECK — the static sign-ray profile RE-READ at
      fine D (0.05 steps, 1.50..2.50 — 21 points; e192's grid was coarse
      exactly there: {1.5, 2.0} then 2.5), gated at {1.5, 2.0, 2.5} vs
      e192's committed R2 rows; the refined 0.27-crossing placed by
      linear-in-D interpolation on the fine grid.
  (3) SHORT FRESH ARMS — sign-walks with the front recomputed every k
      steps (k in {1, 2, 4, 8}) at matched per-step L2: between
      refreshes the SAME direction is re-walked at full size (the stream
      advances one batch per step; only refresh steps consume theirs for
      a gradient). Does the kill-D move with k (the re-computation
      rate)? k=1 IS the opt2 sign path (reproduction rung, gated). The
      k=inf rung (frozen front) is e192's R2 static profile LOADED
      COMMITTED, never rerun (a frozen-front walk is colinear with the
      static ray by construction — e192's own rider disclosure); e192's
      300-step pinned g-ray rider is co-loaded as the slow-step
      re-orientation-denial context (a different ray, disclosed).

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - FRONT-CHASES: "fires if the front's overlap with the death direction
    RISES along the path and the kill-D falls with the re-computation
    rate — the front tracks the fact's fleeing support; the mirror of
    re-orientation."
  - FRONT-ACCUMULATES: "fires if consecutive fronts are mutually
    correlated (mean |cos| >> 1/sqrt(d)) and the effective displacement
    accumulates super-linearly in step count — a shrinking-subspace
    effect."
  - STATIC-ARTIFACT: "fires if the fine-grid static profile shows the
    2.5 edge was a coarse-grid artifact (the true edge near 1.75) — the
    path never beat its ray."
  - GRADED: "any mix — the reads verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * "overlap with the death direction" = cos64(sign(g_t), u_g) with u_g
    e191's committed static g-ray (the kill ray is descent, -u_g; the
    step is -sign(g_t); double negation makes cos(sign(g_t), u_g) the
    step-direction-vs-death-ray overlap). "RISES along the path" =
    strictly increasing across the sign path's three read fronts t=0,1,2
    (f_0 applied in step 1, f_1 applied in step 2, f_2 the kill-state
    front the path would ride next).
  * "kill-D falls with the re-computation rate" = D_read(k) strictly
    increasing in k over {1,2,4,8} (D_read = the densified kill-D; an
    arm that stops alive contributes its final D) AND D_read(k=1) < the
    refined static edge from read (2). Both conjuncts or the clause
    dies.
  * "mutually correlated" = mean |cos| over all consecutive-refresh
    front pairs pooled across the arms' PRE-KILL walks (k=1's single
    pre-kill pair + each k>=2 arm's consecutive refresh pairs; the
    post-kill continuation's pairs reported as context only) >= 0.05
    (~83x the isotropic floor 1/sqrt(d) = 6.04e-4 — ">>").
  * "accumulates super-linearly in step count" = super-DIFFUSIVE at
    every arm's pre-kill end: cum_disp(n) > 1.1 * STEP_L2 * sqrt(n) for
    every arm with n >= 2 walked steps (efficiency e(n) =
    cum/(STEP_L2*sqrt(n)) > 1.1 — above the orthogonal-walk baseline;
    "like the Adam path" = approaching ballistic). Both conjuncts or
    the clause dies.
  * "the true edge near 1.75" = the fine grid's FIRST 0.27-downcrossing
    (linear-in-D interpolated) at D <= 1.9 (within 0.15 of the path's
    1.7496). A crossing in (1.9, 2.5) REFINES the edge but does NOT
    fire (the inversion stands).
  * KILL = g-12 <= 0.27 at an every-step read; D_kill = the opt1c
    convention densified along the killing step at f in {0.2,0.4,0.6,0.8}
    (points ON the trajectory); per-step L2 == STEP_L2
    1.6542880535125732 EXACTLY (asserted every step, fp64 norm, opt2's
    sign_update VERBATIM); stop per arm = first of kill / D >= 5.0 alive
    / 16-step cap; the k=1 arm additionally walks 4 post-kill steps
    (disclosed context).
  * composite order frozen: FRONT-CHASES -> FRONT-ACCUMULATES ->
    STATIC-ARTIFACT -> GRADED (first that fires is the verdict; every
    bar's fires flag reported verbatim).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): the
standard cell gates (corpus ZEPH 0; splice mix {FLORIZEL: 19, ELIZABETH:
41}; battery shapes 60 x (130 +- j); neutral bank 16 starts bit-equal
e185's stored list; root gated vs e151's before-cells) PLUS the t=0
BIT-GATE vs opt1's committed A0 row PLUS the same-point machinery gate
(T150) PLUS e191's u_g checkpoint gate PLUS e192's R2 direction md5 gate
PLUS THE SAVED-STATE PROVENANCE GATES: the rebuilt k=1 walk must
reproduce opt2's committed a_sign trajectory (per-step CE / cum_disp /
g-12, the densified kill bracket) AND the saved
runs/checkpoints/opt2_a_sign_s2.pt's parameters — failure = control
failure, abort. WHAT EACH ARM GUARANTEES: NOTHING — every arm walks the
licensed stream at lethal-scale per-step L2 and could kill anywhere;
that openness is the point.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
the GPU is another agent's, never claimed), torch threads 4 (the
e185-era reduction order), sequential phases, every phase < 180 s,
PROGRESSIVE partial metrics.json writes after every phase (the outage
lesson), n=1, single stream seed 10902 lineage.

Outputs: runs/e194/{metrics.json, e194_front_overlap.png,
e194_static_fine.png, e194_k_ladder.png}. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds).

Run:  cd lab && python e194_sign_front.py    (E194_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E194_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e194 is CPU-only by dispatch"

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
E191_DIR_CK = CKPT_DIR / "e191_static_dir_u.pt"
OPT2_SIGN_CK = CKPT_DIR / "opt2_a_sign_s2.pt"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e185/opt1/opt2)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; arms have NO lr)
STEP_L2 = 1.6542880535125732      # opt1 A0's committed step-1 L2 = the matched per-step L2
CE1_COMMITTED = 1.356567621231079        # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253       # opt1 A0's committed step-1 pre-clip grad norm
A0_DKILL_COMMITTED = 2.4892616271972656  # opt1 A0's committed kill displacement
RAW_DKILL_COMMITTED = 0.9203406595225093 # opt1c's committed densified raw kill D
SIGN_EDGE_COMMITTED = 2.5                # e192's committed static sign-ray kill D (coarse grid)
PATH_DKILL_COMMITTED = 1.749640490742179 # opt2 a_sign's committed densified kill D
PATH_DKILL_RAW_COMMITTED = 1.9539829631812597  # opt2 a_sign's committed raw kill D
W024_ADAM_SAMEPOINT = 0.039550412581627135  # the chart's committed same-point sign read
W024_RAW_SAMEPOINT = 0.09862135965497001    # the chart's committed same-point raw read
# opt2 a_sign's committed trajectory (the reproduction targets, G_REPRO)
OPT2_S1 = {"ce_batch": 1.356567621231079, "cum_disp": 1.6567984819412231,
           "gm12": 0.6785961389541626, "preclip_gnorm": 0.9829167127609253,
           "step_disp": 1.6567984819412231,
           "matched_point_m12": 0.03955041750707301}
OPT2_S2 = {"ce_batch": 1.8259366750717163, "cum_disp": 2.1502907276153564,
           "gm12": 9.830708586378023e-05, "preclip_gnorm": 6.194264888763428,
           "step_disp": 1.6567445993423462}
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

ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
STEP_CAP = 16                     # drift-stall safety cap (opt2's)
D_TARGET = 5.0                    # 2x the sign edge (opt2's horizon)
SHUT_BAR = 0.27                   # e185's kill bar
K_LADDER = (1, 2, 4, 8)           # the re-computation-rate ladder (k=1 = reproduction)
EXT_STEPS = 4                     # post-kill continuation steps (read-only context)
STATIC_FINE = [round(1.50 + 0.05 * i, 2) for i in range(21)]   # 1.50..2.50
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # opt1c's along-path kill-bracket densification
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
N_PARAM = 2_739_072
ISO_FLOOR = 1.0 / (N_PARAM ** 0.5)          # 1/sqrt(d) = 6.04e-4
ACCUM_COS_BAR = 0.05                        # frozen: ">> 1/sqrt(d)" (~83x the floor)
ACCUM_EFF_BAR = 1.1                         # frozen: super-diffusive efficiency bar
CHASE_RISE_TOL = 0.0                        # strict rise (no tolerance shopping)
STATIC_NEAR_PATH = 1.9                      # frozen: "the true edge near 1.75" = crossing <= 1.9
L2_ASSERT_TOL = 1e-5
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_SAMEPOINT_TOL = 2e-3
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_STATIC_TOL = 1e-3               # e192's own G_R1_XCHK tolerance class
if SMOKE:                         # shakedown trims (documented in deviations)
    STATIC_FINE = [1.5, 1.6, 1.75]
    EXT_STEPS = 1
    K_LADDER = (1, 2)
    DENSIFY_F = ()

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
    "front_chases": "FRONT-CHASES: \"fires if the front's overlap with the "
        "death direction RISES along the path and the kill-D falls with the "
        "re-computation rate — the front tracks the fact's fleeing support; "
        "the mirror of re-orientation.\"",
    "front_accumulates": "FRONT-ACCUMULATES: \"fires if consecutive fronts "
        "are mutually correlated (mean |cos| >> 1/sqrt(d)) and the effective "
        "displacement accumulates super-linearly in step count — a "
        "shrinking-subspace effect.\"",
    "static_artifact": "STATIC-ARTIFACT: \"fires if the fine-grid static "
        "profile shows the 2.5 edge was a coarse-grid artifact (the true "
        "edge near 1.75) — the path never beat its ray.\"",
    "graded": "GRADED: \"any mix — the reads verbatim.\"",
    "operationalizations": "the cell = the licensed e185 wash stream (root "
        "e131, seed-10902, bit-gated batches, TRUE targets, clip 1.0) with "
        "delta_t = -STEP_L2 * sign(g_t)/||sign(g_t)|| recomputed every k "
        "steps (k in {1,2,4,8}; between refreshes the SAME direction is "
        "re-walked at matched L2; the stream advances one batch per step); "
        "the k=1 arm IS opt2's a_sign, rebuilt bit-exactly and gated vs the "
        "committed trajectory + the saved runs/checkpoints/opt2_a_sign_s2.pt "
        "parameters; KILL = g-12 <= 0.27 at an every-step read, D_kill = "
        "opt1c's densified bracket; the k=inf rung is e192's R2 static "
        "profile LOADED COMMITTED (a frozen-front walk is colinear with the "
        "static ray by construction); 'overlap with the death direction' = "
        "cos64(sign(g_t), u_g) (double-negation identity: step -sign, ray "
        "-u_g), RISES = strictly across the path's three read fronts t=0,1,2; "
        "'falls with the re-computation rate' = D_read strictly increasing "
        "in k AND D_read(k=1) < the refined static edge; 'mutually "
        "correlated' = pooled pre-kill consecutive-refresh mean |cos| >= "
        "0.05 (~83x the 1/sqrt(d) floor 6.04e-4); 'super-linearly' = "
        "super-diffusive, cum(n) > 1.1*STEP_L2*sqrt(n) at every arm's "
        "pre-kill end (n >= 2); 'the true edge near 1.75' = the fine grid's "
        "first 0.27-downcrossing (interpolated) <= 1.9, a crossing in (1.9, "
        "2.5) refines but does not fire; composite order frozen "
        "FRONT-CHASES -> FRONT-ACCUMULATES -> STATIC-ARTIFACT -> GRADED.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "The k=1 arm doubles as the front-overlap read and the opt2 "
    "reproduction rung (G_REPRO): identical construction — opt2's sign_update "
    "copied VERBATIM (fp64 support norm), identical stream draw order, "
    "identical every-step reads — so the walk is walked ONCE and gated, not "
    "walked twice.",
    "The k=1 arm does NOT stop at the kill (opt2's stop rule): after the "
    "kill bracket + densification (computed exactly as opt2 committed it) "
    f"the walk CONTINUES {EXT_STEPS} steps as read-only context so the front "
    "trace has more than the 2 pre-kill points. Continuation rows are "
    "flagged post_kill=true and NEVER adjudicate.",
    "k>=2 arms skip the forward/backward on non-refresh steps (the delta is "
    "the previous delta by construction; per-step L2 asserted identical); "
    "the stream still advances one batch draw per step so refresh-step "
    "batch indexing is the licensed stream's own.",
    "No chunk-resume machinery (opt2's): every phase is < 180 s and the "
    "progressive metrics.json writes after every phase carry the outage "
    "insurance instead.",
    "No new checkpoints saved: the deliverables are metrics + figures; the "
    "saved-state role here is INPUT (provenance-gated), not output.",
    "The k=inf rung is e192's R2 static sign profile loaded committed "
    "(frozen-front walk = colinear steps = the static ray by construction, "
    "e192's own rider disclosure); e192's 300-step pinned g-ray rider is "
    "co-loaded as the slow-step re-orientation-denial context (a different "
    "ray at a different step size — context, not a rung).",
    "Light reads only (battery + CE_R + fact-gradient alignment co-reads + "
    "displacement + front cosines); CPU-ONLY, threads 4, n=1, seed lineage "
    "10902.",
    "Smoke mode trims: static grid {1.5, 1.6, 1.75}, 1 continuation step, "
    "k ladder {1, 2}, no densification (verdict stamped SMOKE; nothing "
    "adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt2_density.py VERBATIM (whose own provenance is
# lab/opt1c_direction_size.py / e185 — the e176n lineage). Copied rather
# than imported to own the device policy and the bit-exact arithmetic.

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
    """A k-ladder sign walk on the licensed stream. Front refreshed when
    (step-1) % k == 0 (k=1: every step = opt2's a_sign). Every-step g-12;
    per-step matched-L2 assertion; kill bracket densified (opt1c); optional
    per-step fact-gradient alignment reads (the k=1 mechanism walk); optional
    post-kill continuation (read-only context)."""
    net = copy.deepcopy(net0)
    net.train()
    evl = copy.deepcopy(net0)
    evl.eval()
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    n_steps = STEP_CAP
    journal, front_trace, kill = [], [], None
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
            + (f" ov(g-ray) {frow['vs_g_ray']:+.4f}" if frow else "")
            + (f" f-vs-prev {frow['vs_prev_front']:+.4f}" if frow and
               frow["vs_prev_front"] is not None else ""))
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
            log(f"  [k={k}] D_TARGET reached alive at s{step}")
        if step >= STEP_CAP and kill is None:
            kill = {"kind": "cap", "step": step,
                    "reason": f"step cap {STEP_CAP}",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
    out = {"k": k, "journal": journal, "front_trace": front_trace,
           "stop": kill, "max_l2_dev": max_l2_dev, "zeph": zeph,
           "net": net, "train_seconds": time.time() - t0a}
    return out


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e194_smoke" if SMOKE else "e194")
    log(f"E194 THE SIGN-FRONT MECHANISM (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), phases < 180s, "
        f"progressive writes, n=1, seed lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT2_METRICS, OPT1C_METRICS, OPT1_METRICS, E192_METRICS,
              ECHART_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    opt2m = json.loads(OPT2_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    e192m = json.loads(E192_METRICS.read_text(encoding="utf-8"))
    chartm = json.loads(ECHART_METRICS.read_text(encoding="utf-8"))
    # hard-bind the committed references (asserts catch committed-file drift)
    sign_stop = opt2m["arms"]["a_sign"]["stop"]
    assert abs(sign_stop["D_kill"] - PATH_DKILL_COMMITTED) < 1e-12
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
    e192_r2 = e192m["profiles"]["R2_SIGN"]
    assert e192_r2["u_md5"] == E192_R2_U_MD5
    for D, ref in E192_R2_GATE.items():
        row = next(r for r in e192_r2["rows"] if abs(r["D"] - D) < 1e-9)
        assert abs(row["gm12"] - ref["gm12"]) < 1e-12
    log("parents: opt2 (a_sign path kill 1.7496), opt1c (raw 0.9203), opt1 "
        f"(A0 {A0_DKILL_COMMITTED:.4f}), e192 (R2 static edge {SIGN_EDGE}, "
        "grid coarse in 1.5-2.0), e_chart (same-point anchors) — loaded "
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
        tries += 1
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
    s_raw = torch.sign(g0_t0)
    u_sign = (s_raw / torch.norm(s_raw)).clone()      # e192's construction verbatim
    u_sign_md5 = hashlib.md5(u_sign.numpy().tobytes()).hexdigest()
    G_R2DIR = {
        "u_sign_md5": u_sign_md5, "committed_e192_md5": E192_R2_U_MD5,
        "md5_match": bool(u_sign_md5 == E192_R2_U_MD5),
        "u_norm": float(torch.norm(u_sign.double())),
        "committed_e192_norm": E192_R2_UNORM,
        "n_zero_g_coords": int((g0_t0 == 0).sum()),
        "note": "the static sign(g_0) ray rebuilt from the gated t=0 "
                "gradient (e192's fp32-norm construction VERBATIM)",
    }
    G_R2DIR["pass"] = bool(G_R2DIR["md5_match"]
                           and abs(G_R2DIR["u_norm"] - E192_R2_UNORM) < 1e-4)
    log(f"G_R2DIR (static sign ray): md5 "
        + ("match" if G_R2DIR["md5_match"] else "DRIFT")
        + f", norm {G_R2DIR['u_norm']:.7f} vs {E192_R2_UNORM:.7f}: "
        + ("PASS" if G_R2DIR["pass"] else "FAIL"))
    if not G_U191["pass"] or not G_R2DIR["pass"]:
        raise RuntimeError("committed-direction gates FAILED — abort")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e194_sign_front",
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
    write_partial("gates passed")
    log("gates passed; partial metrics written")

    # =====================================================================
    # PHASE A — the k=1 sign walk: front-overlap read + G_REPRO/G_SAVEDSTATE
    # =====================================================================
    log("=" * 78)
    log(f"PHASE A — the k=1 sign walk (opt2's a_sign rebuilt + front reads + "
        f"{EXT_STEPS} post-kill continuation steps)")
    arm1 = walk_arm(1, net0, anchor_neutral, train_ids, itos, gm12_ids, zid,
                    theta0, u_g, u_sign, root_m12_grad, rd, write_partial,
                    stub, extend_past_kill=EXT_STEPS, fact_reads=True)
    j1 = arm1["journal"]

    # ---- G_REPRO: the rebuilt walk vs opt2's committed a_sign trajectory
    s1, s2 = j1[0], j1[1]
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
        "D_kill": (arm1["stop"].get(
                       "D_kill" if DENSIFY_F else "D_kill_raw",
                       arm1["stop"]["D_kill"]),
                   PATH_DKILL_COMMITTED if DENSIFY_F
                   else PATH_DKILL_RAW_COMMITTED),
    }
    repro_tol = G_REPRO_TOL if not SMOKE else 5e-3   # smoke: no densification
    repro_diffs = {kk: abs(vv[0] - vv[1]) for kk, vv in repro_rows.items()}
    dens_diffs = ([abs(a["gm12"] - b[1]) for a, b in
                  zip(arm1["stop"].get("dens", []), OPT2_DENS)]
                  if arm1["stop"].get("dens") else [])
    G_REPRO = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows.items()},
        "dens_gm12_max_abs_diff": (max(dens_diffs) if dens_diffs else None),
        "tol": repro_tol,
        "e192_static_sign_D16543_gm12": {"measured": s1["gm12"],
                                         "committed": E192_A0_SIGNRAY_GM12,
                                         "abs_diff": abs(
                                             s1["gm12"] - E192_A0_SIGNRAY_GM12),
                                         "note": "corroboration: step 1 of "
                                                 "the sign walk IS a static "
                                                 "sign-ray jump at D=STEP_L2"},
        "pass": bool(arm1["stop"]["kind"] == "kill"
                     and max(repro_diffs.values()) < repro_tol
                     and (not dens_diffs
                          or max(dens_diffs) < repro_tol)),
        "note": "THE SAVED-STATE PROVENANCE GATE (Rule 12): the rebuilt k=1 "
                "walk must reproduce opt2's committed a_sign trajectory and "
                "densified kill bracket — failure = control failure, abort",
    }
    log("G_REPRO (vs opt2 committed a_sign): max|diff| "
        f"{max(repro_diffs.values()):.2e}"
        + (f", dens max|diff| {max(dens_diffs):.2e}" if dens_diffs else "")
        + ": " + ("PASS" if G_REPRO["pass"] else "FAIL"))

    # ---- G_SAVEDSTATE: the saved opt2 sign-arm checkpoint vs the walked s2
    # (the phase-A net continued past the kill, so its CURRENT state is the
    # continuation endpoint, not theta_2 — walk the clean 2-step arm once
    # more WITHOUT continuation/fact-reads for the exact s2 endpoint;
    # deterministic, identical construction, ~5 s)
    arm1b = walk_arm(1, net0, anchor_neutral, train_ids, itos, gm12_ids, zid,
                     theta0, u_g, u_sign, root_m12_grad, rd, write_partial,
                     stub, extend_past_kill=0, fact_reads=False)
    sck = torch.load(OPT2_SIGN_CK, map_location="cpu", weights_only=False)
    saved_net = TinyGPT(Cfg())
    saved_net.load_state_dict(sck["model"])
    saved_flat = flat_params(saved_net)
    walked_s2 = flat_params(arm1b["net"])
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
                "checkpoint must BE the walked k=1 s2 endpoint (md5 or "
                "max|diff| < 1e-6)",
    }
    G_SAVEDSTATE["pass"] = bool(
        G_SAVEDSTATE["meta_gate"]["match"]
        and (G_SAVEDSTATE["walked_s2_flat_md5"] == G_SAVEDSTATE["saved_flat_md5"]
             or maxdiff < 1e-6))
    log(f"G_SAVEDSTATE (opt2_a_sign_s2.pt): meta "
        + ("match" if G_SAVEDSTATE["meta_gate"]["match"] else "DRIFT")
        + f", walked-s2 vs saved max|diff| {maxdiff:.2e}: "
        + ("PASS" if G_SAVEDSTATE["pass"] else "FAIL"))
    if not G_REPRO["pass"] or not G_SAVEDSTATE["pass"]:
        stub["gates"]["G_REPRO"] = G_REPRO
        stub["gates"]["G_SAVEDSTATE"] = G_SAVEDSTATE
        write_partial("CONTROL FAILURE — provenance gates failed")
        raise RuntimeError("provenance gates FAILED (G_REPRO/G_SAVEDSTATE) — "
                           "abort before any read is believed")
    stub["gates"]["G_REPRO"] = G_REPRO
    stub["gates"]["G_SAVEDSTATE"] = G_SAVEDSTATE

    # ---- the front-overlap read summary (pre-kill vs continuation)
    ft = arm1["front_trace"]
    pre_kill = [r for r in ft if r["front_state"] < arm1["stop"]["step"]]
    cont = [r for r in ft if r["front_state"] >= arm1["stop"]["step"]]
    ov = [r["vs_g_ray"] for r in ft]
    phaseA = {
        "walk_traj": [{kk: vv for kk, vv in r.items() if kk != "x_md5"}
                      for r in j1],
        "front_trace": ft,
        "front_trace_pre_kill": pre_kill,
        "front_trace_continuation": cont,
        "stop": arm1["stop"], "max_l2_dev": arm1["max_l2_dev"],
        "continuation_disclosure": (
            f"steps {arm1['stop']['step'] + 1}..{j1[-1]['step']} are "
            "post-kill continuation (read-only context; never adjudicates)"),
        "reads": {
            "ov_vs_g_ray_t0_t1_t2": [r["vs_g_ray"] for r in ft[:3]],
            "front_persistence": [r["vs_prev_front"] for r in ft[1:]],
            "front_vs_fresh_g": [r["vs_fresh_g"] for r in ft],
            "cum_vs_g_ray": [r["cum_vs_g_ray"] for r in j1],
        },
    }
    stub["phases_partial"]["A_front_overlap"] = E43.jsonable(phaseA)
    write_partial("phase A complete (front-overlap read + provenance gates)")
    log(f"PHASE A reads: ov(t=0,1,2) = "
        + ", ".join(f"{v:+.4f}" for v in ov[:3])
        + "; persistence = "
        + ", ".join(f"{v:+.4f}" if v is not None else "n/a"
                    for v in phaseA["reads"]["front_persistence"]))

    # =====================================================================
    # PHASE B — THE STATIC-ARTIFACT CHECK (fine-grid R2 re-read)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE B — the static sign-ray RE-READ at fine D ({STATIC_FINE[0]}.."
        f"{STATIC_FINE[-1]}, {len(STATIC_FINE)} points)")
    evl = copy.deepcopy(net0)
    evl.eval()
    rows, static_xc = [], {}
    for D in STATIC_FINE:
        thD = theta0 - D * u_sign                      # e192's placement verbatim
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        row = {"D": float(D), "rms": float(D) / 1655.0141993348577,
               "gm12": gz["mean_pz"],
               "frac_argmax_z": gz["frac_argmax_z"],
               "disp_check": float(torch.norm(thD - theta0))}
        if any(abs(D - g) < 1e-9 for g in E192_R2_GATE):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
            ref = E192_R2_GATE[next(g for g in E192_R2_GATE
                                    if abs(D - g) < 1e-9)]
            static_xc[f"D{D}"] = {
                "gm12_measured": gz["mean_pz"], "gm12_committed": ref["gm12"],
                "gm12_diff": abs(gz["mean_pz"] - ref["gm12"]),
                "ce_r_measured": row["ce_r"], "ce_r_committed": ref["ce_r"],
                "ce_r_diff": abs(row["ce_r"] - ref["ce_r"])}
        rows.append(row)
    G_STATIC_XC = {
        "crosschecks": static_xc, "tol": G_STATIC_TOL,
        "pass": bool(all(v["gm12_diff"] < G_STATIC_TOL
                         and v["ce_r_diff"] < G_STATIC_TOL
                         for v in static_xc.values())),
        "note": "the fine-grid machinery gate: this build must reproduce "
                "e192's committed R2 rows at the shared Ds before the fine "
                "points are believed",
    }
    log("G_STATIC_XC (vs e192 committed R2 @ 1.5/2.0/2.5): max gm12|d| "
        + f"{max((v['gm12_diff'] for v in static_xc.values()), default=0):.2e}: "
        + ("PASS" if G_STATIC_XC["pass"] else "FAIL"))
    if not G_STATIC_XC["pass"]:
        stub["gates"]["G_STATIC_XC"] = G_STATIC_XC
        write_partial("CONTROL FAILURE — static re-read gate failed")
        raise RuntimeError("static re-read gate FAILED — abort")
    stub["gates"]["G_STATIC_XC"] = G_STATIC_XC
    # the refined edge: first 0.27-downcrossing on the fine grid (interp)
    refined_edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm12"] <= SHUT_BAR and a["gm12"] > SHUT_BAR:
            refined_edge = interp_d_kill(a["gm12"], b["gm12"],
                                         a["D"], b["D"])
            break
    dips = [(a["D"], b["D"]) for i in range(1, len(rows))
            for a, b in [(rows[i - 1], rows[i])]
            if b["gm12"] > SHUT_BAR and a["gm12"] <= SHUT_BAR]   # any upcross (non-monotone)
    phaseB = {
        "rows": rows, "refined_static_edge": refined_edge,
        "refined_edge_note": ("first 0.27-downcrossing, linear-in-D "
                              "interpolated on the fine grid; None = the "
                              "profile never crosses within the window"),
        "any_upcross_inside_window": dips,
        "coarse_grid_gap": "e192's committed grid had NO point in (1.5, 2.0) "
                           "— this window is the check",
        "path_kill_D": PATH_DKILL_COMMITTED,
    }
    stub["phases_partial"]["B_static_fine"] = E43.jsonable(phaseB)
    write_partial("phase B complete (static fine re-read)")
    log(f"PHASE B: refined static edge "
        + (f"{refined_edge:.4f}" if refined_edge is not None else "None "
           "(no crossing in the window)")
        + f"; path kill D {PATH_DKILL_COMMITTED:.4f}")

    # =====================================================================
    # PHASE C — the k-ladder (k=1 from phase A; k=2,4,8 fresh)
    # =====================================================================
    arms = {1: arm1}
    for k in K_LADDER:
        if k == 1:
            continue
        log("=" * 78)
        log(f"PHASE C — arm k={k} (front refreshed every {k} steps)")
        arms[k] = walk_arm(k, net0, anchor_neutral, train_ids, itos,
                           gm12_ids, zid, theta0, u_g, u_sign, root_m12_grad,
                           rd, write_partial, stub, extend_past_kill=0,
                           fact_reads=False)
        stub["phases_partial"][f"C_k{k}"] = E43.jsonable(
            {"k": k, "journal": arms[k]["journal"],
             "front_trace": arms[k]["front_trace"], "stop": arms[k]["stop"]})
        write_partial(f"phase C arm k={k} complete")

    # ---- the ladder read: D_read(k) (densified kill-D or final D)
    ladder_rows, eff_rows = [], []
    front_pairs_prekill = []      # (k, cos) consecutive-refresh pairs, pre-kill
    cont_pairs = []               # continuation pairs (context only)
    kill_step_k1 = arm1["stop"]["step"]
    for k in K_LADDER:
        a = arms[k]
        j = [r for r in a["journal"] if not r["post_kill"]]
        stop = a["stop"]
        d_read = (stop["D_kill"] if stop["kind"] == "kill"
                  else j[-1]["cum_disp"])
        n_pre = stop["step"] if stop["kind"] == "kill" else len(j)
        eff = (float(j[-1]["cum_disp"] / (STEP_L2 * np.sqrt(n_pre)))
               if n_pre >= 1 else None)
        ladder_rows.append({"k": k, "recompute_rate": 1.0 / k,
                            "stop_kind": stop["kind"], "D_read": d_read,
                            "steps_pre_kill": n_pre,
                            "final_cum_disp_pre_kill": j[-1]["cum_disp"],
                            "efficiency_e": eff})
        if n_pre >= 2:
            eff_rows.append({"k": k, "n": n_pre, "e": eff})
        ft_pairs = [r["vs_prev_front"] for r in a["front_trace"]
                    if r["vs_prev_front"] is not None
                    and r["front_state"] < stop["step"]]
        front_pairs_prekill += [(k, v) for v in ft_pairs]
        if k == 1:
            cont_pairs = [r["vs_prev_front"] for r in a["front_trace"]
                          if r["vs_prev_front"] is not None
                          and r["front_state"] >= stop["step"]]
    # the k=inf rung: e192's committed R2 static profile (loaded, never rerun)
    ladder_rows.append({"k": "inf", "recompute_rate": 0.0,
                        "stop_kind": "static_ray_loaded",
                        "D_read": (refined_edge if refined_edge is not None
                                   else SIGN_EDGE_COMMITTED),
                        "source": "runs/e192/metrics.json profiles.R2_SIGN "
                                  "(fine-refined this cell)",
                        "note": "a frozen-front walk is colinear with the "
                                "static ray by construction (e192's rider "
                                "disclosure) — loaded, never rerun"})
    pooled_abs = [abs(v) for _, v in front_pairs_prekill]
    accum = {
        "iso_floor": ISO_FLOOR,
        "cos_bar": ACCUM_COS_BAR,
        "prekill_pairs": [{"k": k, "cos": v} for k, v in front_pairs_prekill],
        "prekill_mean_abs_cos": (float(np.mean(pooled_abs))
                                 if pooled_abs else None),
        "continuation_pairs_context": cont_pairs,
        "continuation_mean_abs_cos": (float(np.mean([abs(v) for v in cont_pairs]))
                                      if cont_pairs else None),
        "efficiencies": eff_rows, "eff_bar": ACCUM_EFF_BAR,
        "note": "pairs = consecutive REFRESHED fronts within each arm's "
                "pre-kill walk; continuation pairs are context only",
    }
    stub["phases_partial"]["C_ladder"] = E43.jsonable(
        {"ladder_rows": ladder_rows, "accumulation": accum})
    write_partial("phase C complete (the k-ladder)")
    log("THE K-LADDER: "
        + " < ".join(f"k={r['k']}:{r['D_read']:.4f}" for r in ladder_rows))
    log(f"accumulation: pre-kill mean|cos(fronts)| "
        f"{accum['prekill_mean_abs_cos'] if accum['prekill_mean_abs_cos'] is not None else float('nan'):.4f}"
        f" (floor {ISO_FLOOR:.1e}, bar {ACCUM_COS_BAR}); efficiencies "
        + ", ".join(f"k={r['k']}:{r['e']:.3f}" for r in eff_rows))

    # =====================================================================
    # ADJUDICATION (frozen clauses; composite order FRONT-CHASES ->
    # FRONT-ACCUMULATES -> STATIC-ARTIFACT -> GRADED; no shopping)
    # =====================================================================
    if SMOKE:
        verdict, clause = "SMOKE", "shakedown — nothing adjudicated"
        bars = {}
    else:
        ov3 = [r["vs_g_ray"] for r in arm1["front_trace"][:3]]
        chase_rise = (len(ov3) == 3 and ov3[0] < ov3[1] < ov3[2])
        d_reads = [next(r["D_read"] for r in ladder_rows if r["k"] == k)
                   for k in K_LADDER]
        chase_monotone = all(d_reads[i] < d_reads[i + 1]
                             for i in range(len(d_reads) - 1))
        chase_below_static = (refined_edge is not None
                              and d_reads[0] < refined_edge)
        fc = bool(chase_rise and chase_monotone and chase_below_static)
        mac = accum["prekill_mean_abs_cos"]
        fa = bool(mac is not None and mac >= ACCUM_COS_BAR
                  and len(eff_rows) > 0
                  and all(r["e"] > ACCUM_EFF_BAR for r in eff_rows))
        sa = bool(refined_edge is not None and refined_edge <= STATIC_NEAR_PATH)
        bars = {
            "FRONT_CHASES": {
                "fires": fc,
                "detail": {"ov_t0_t1_t2": ov3, "rise_strict": chase_rise,
                           "D_reads_by_k": dict(zip(K_LADDER, d_reads)),
                           "strictly_increasing_in_k": chase_monotone,
                           "refined_static_edge": refined_edge,
                           "k1_below_static": chase_below_static},
            },
            "FRONT_ACCUMULATES": {
                "fires": fa,
                "detail": {"prekill_mean_abs_cos": mac,
                           "bar": ACCUM_COS_BAR, "iso_floor": ISO_FLOOR,
                           "efficiencies": eff_rows, "eff_bar": ACCUM_EFF_BAR},
            },
            "STATIC_ARTIFACT": {
                "fires": sa,
                "detail": {"refined_static_edge": refined_edge,
                           "near_path_bar": STATIC_NEAR_PATH,
                           "path_kill_D": PATH_DKILL_COMMITTED,
                           "committed_coarse_edge": SIGN_EDGE_COMMITTED},
            },
            "GRADED": {"fires": not (fc or fa or sa)},
        }
        if fc:
            verdict = "FRONT-CHASES"
            clause = (f"the front's overlap with the death ray RISES along "
                      f"the path ({', '.join(f'{v:+.4f}' for v in ov3)}) and "
                      f"the kill-D falls with the re-computation rate "
                      + " < ".join(f"k={r['k']} {r['D_read']:.4f}"
                                   for r in ladder_rows[:-1])
                      + f" (static {ladder_rows[-1]['D_read']:.4f}) — the "
                      "front tracks the fact's fleeing support; the mirror "
                      "of re-orientation.")
        elif fa:
            verdict = "FRONT-ACCUMULATES"
            clause = (f"consecutive fronts are mutually correlated (pre-kill "
                      f"mean |cos| {mac:.4f} >= {ACCUM_COS_BAR}, "
                      f"{mac / ISO_FLOOR:.0f}x the isotropic floor) and "
                      "displacement accumulates super-diffusively ("
                      + ", ".join(f"k={r['k']} e={r['e']:.3f}"
                                  for r in eff_rows)
                      + f" > {ACCUM_EFF_BAR}) — a shrinking-subspace effect.")
        elif sa:
            verdict = "STATIC-ARTIFACT"
            clause = (f"the fine grid's first 0.27-downcrossing sits at "
                      f"D {refined_edge:.4f} <= {STATIC_NEAR_PATH} — the "
                      f"committed 2.5 edge was coarse-grid artifact (e192's "
                      "grid had no point in (1.5, 2.0)); the path never beat "
                      "its ray.")
        else:
            verdict = "GRADED"
            clause = ("any mix — the reads verbatim: ov(t=0,1,2) = "
                      + ", ".join(f"{v:+.4f}" for v in ov3)
                      + "; D_read by k = "
                      + ", ".join(f"{r['k']}:{r['D_read']:.4f}"
                                  for r in ladder_rows)
                      + (f"; refined static edge "
                         + (f"{refined_edge:.4f}"
                            if refined_edge is not None else "None")
                         + "; pre-kill mean |cos(fronts)| "
                         + (f"{mac:.4f}" if mac is not None else "n/a")
                         + "; efficiencies "
                         + ", ".join(f"{r['k']}:{r['e']:.3f}"
                                     for r in eff_rows)))
    log("=" * 78)
    log(f"E194 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e194_sign_front",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("WHERE does the sign path's extra lethality live — the "
                     "front CHASING the fact's fleeing support (overlap with "
                     "the death ray rising; kill-D falling with the "
                     "re-computation rate), the front ACCUMULATING in a "
                     "shrinking subspace (mutually correlated fronts, "
                     "super-diffusive displacement), or the 2.5 static edge "
                     "being a coarse-grid ARTIFACT (the true edge near "
                     "1.75)? On the saved opt2 sign-arm state + the e131 "
                     "root: the front-overlap read, the fine static re-read, "
                     "and the k-ladder at matched per-step L2."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "the e185 wash stream VERBATIM (opt2's "
                             "conventions), update rule = the sign front at "
                             "matched per-step L2, recomputed every k steps",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random corpus "
                     "windows, full-token CE, clip 1.0",
            "input_seed": FREEZE_SEED,
            "matched_step_L2": {"value": STEP_L2,
                                "provenance": "opt1 A0's committed step-1 "
                                "displacement (recomputed at t=0 and "
                                "bit-gated in G_T0)"},
            "k_ladder": list(K_LADDER) + ["inf(static R2, loaded)"],
            "death_direction": "u_g = e191's committed static g-ray; the "
                               "kill ray is descent (-u_g); overlap read = "
                               "cos64(sign(g_t), u_g) (double-negation "
                               "identity, stated in the operationalizations)",
        },
        "gates": stub["gates"],
        "phaseA_front_overlap": phaseA,
        "phaseB_static_fine": phaseB,
        "phaseC_k_ladder": {
            "arms": {str(k): {"journal": [{kk: vv for kk, vv in r.items()
                                           if kk != "x_md5"}
                                          for r in arms[k]["journal"]],
                             "front_trace": arms[k]["front_trace"],
                             "stop": arms[k]["stop"],
                             "max_l2_dev": arms[k]["max_l2_dev"],
                             "zeph": arms[k]["zeph"],
                             "train_seconds": arms[k]["train_seconds"]}
                     for k in K_LADDER},
            "ladder_rows": ladder_rows,
            "accumulation": accum,
        },
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "composite_order": "FRONT-CHASES -> FRONT-ACCUMULATES -> "
                               "STATIC-ARTIFACT -> GRADED (frozen before "
                               "compute)",
            "constants": {"STEP_L2": STEP_L2, "SHUT_BAR": SHUT_BAR,
                          "D_TARGET": D_TARGET, "STEP_CAP": STEP_CAP,
                          "ACCUM_COS_BAR": ACCUM_COS_BAR,
                          "ACCUM_EFF_BAR": ACCUM_EFF_BAR,
                          "STATIC_NEAR_PATH": STATIC_NEAR_PATH,
                          "ISO_FLOOR": ISO_FLOOR},
        },
        "references": {
            "opt2_sign_path": {"metrics": "runs/opt2/metrics.json",
                               "D_kill": PATH_DKILL_COMMITTED,
                               "ckpt": str(OPT2_SIGN_CK),
                               "role": "the parent path (rebuilt, gated "
                                       "G_REPRO/G_SAVEDSTATE — never "
                                       "re-adjudicated)"},
            "e192_terrain": {"metrics": "runs/e192/metrics.json",
                             "static_sign_edge_coarse": SIGN_EDGE_COMMITTED,
                             "R2_fine_refined": refined_edge,
                             "pinned_rider": "300 pinned steps L2 0.005514 "
                                             "along the frozen g-ray — the "
                                             "slow-step re-orientation-denial "
                                             "context (a different ray; "
                                             "loaded committed)",
                             "role": "the static rays (loaded committed)"},
            "opt1c_raw": {"D_kill": RAW_DKILL_COMMITTED},
            "opt1_A0": {"D_kill": A0_DKILL_COMMITTED},
            "e_chart_samepoint": {"adam": W024_ADAM_SAMEPOINT,
                                  "raw": W024_RAW_SAMEPOINT},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 organism/fact (the e131 consolidated line), "
                            "ONE stream (seed 10902, md5-gated vs e185's "
                            "stored hashes at every consumed step); n=1 per "
                            "k-rung; the k-ladder is ONE fact's trajectory "
                            "family, not a population claim"),
            "openness": ("WHAT EACH ARM GUARANTEES: NOTHING — every arm "
                         "walks at lethal-scale per-step L2 and could kill "
                         "anywhere on the ladder; the k-ladder is at MATCHED "
                         "L2 (size held fixed, only the re-computation rate "
                         "varies); that openness is the point"),
            "estimator_lesson": ("every alignment read states its evaluation "
                                 "point: matched_point cos(delta_s, grad "
                                 "m12(theta_{s-1})) (gated at s=1 vs the "
                                 "chart's committed +0.03955), post_step "
                                 "cos(d_cum, grad m12(theta_s)), root_point "
                                 "— T150's lesson"),
            "post_kill_continuation": (f"the k=1 walk's steps past the kill "
                                       f"({EXT_STEPS} steps) are read-only "
                                       "context — flagged post_kill, never "
                                       "adjudicated"),
            "projections_never_adjudicate": ("the refined static edge is an "
                                             "interpolation ON measured "
                                             "points of the fine grid; no "
                                             "arm outcome is extrapolated"),
            "float_texture": ("CPU fp32 texture, this process, 4 threads "
                              "(the e185-era reduction order); the parent "
                              "references are hard-bound asserts and the "
                              "k=1 rebuild reproduced opt2's committed "
                              "trajectory to "
                              f"{max(repro_diffs.values()):.1e}"),
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

    plot_front_overlap(rd / "e194_front_overlap.png", arm1, u_g)
    plot_static_fine(rd / "e194_static_fine.png", phaseB, e192_r2)
    plot_k_ladder(rd / "e194_k_ladder.png", ladder_rows, arms, accum, eff_rows)
    log(f"outputs: {rd / 'metrics.json'} + 3 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_front_overlap(path, arm1, u_g):
    """THE FRONT-OVERLAP READ: the front's overlap with the death ray and
    front persistence along the sign path (pre-kill solid, continuation
    ghosted)."""
    ft = arm1["front_trace"]
    kill_step = arm1["stop"]["step"]
    ts = [r["front_state"] for r in ft]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    ax = axes[0]
    pre = [r for r in ft if r["front_state"] < kill_step]
    post = [r for r in ft if r["front_state"] >= kill_step]
    ax.plot([r["front_state"] for r in pre], [r["vs_g_ray"] for r in pre],
            "o-", ms=7, lw=2.2, color="crimson",
            label="cos(front_t, g-ray)  = step-vs-DEATH-ray overlap")
    if post:
        ax.plot([r["front_state"] for r in post], [r["vs_g_ray"] for r in post],
                "o--", ms=6, lw=1.6, color="crimson", alpha=0.45,
                label="_post-kill continuation (context)")
    ax.plot(ts, [r["vs_fresh_g"] for r in ft], "s-", ms=5, lw=1.5,
            color="darkorange", alpha=0.9,
            label="cos(front_t, fresh g_t)  (gradient-hugging)")
    tr = [r for r in arm1["journal"] if r["refresh"]]
    ax.plot([r["step"] - 1 for r in tr], [r["cum_vs_g_ray"] for r in tr],
            "^-", ms=6, lw=1.5, color="gray", alpha=0.8,
            label="cos(D_cum, g-ray)  (opt2's committed convention)")
    ax.axvline(kill_step - 0.5, color="k", ls=":", lw=1.2)
    ax.text(kill_step - 0.4, 0.02, "KILL (opt2 s2, D 1.7496)", fontsize=7.5,
            rotation=90, va="bottom")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("front index t (state where sign(g_t) was computed)")
    ax.set_ylabel("cosine (fp64)")
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title("THE FRONT-OVERLAP READ — is the front chasing the death "
                 "ray? (rise = chase; opt2's s1 terrain read -0.595 = "
                 "0.595 death-overlap)", fontsize=9.5)

    ax = axes[1]
    pers = [r for r in ft if r["vs_prev_front"] is not None]
    pre_p = [r for r in pers if r["front_state"] < kill_step]
    post_p = [r for r in pers if r["front_state"] >= kill_step]
    ax.plot([r["front_state"] for r in pre_p],
            [r["vs_prev_front"] for r in pre_p], "o-", ms=7, lw=2.2,
            color="seagreen", label="cos(front_t, front_{t-1}) pre-kill")
    if post_p:
        ax.plot([r["front_state"] for r in post_p],
                [r["vs_prev_front"] for r in post_p], "o--", ms=6, lw=1.6,
                color="seagreen", alpha=0.45,
                label="_post-kill (context)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(ISO_FLOOR, color="tab:purple", ls="--", lw=1.1,
               label=f"isotropic floor 1/sqrt(d) = {ISO_FLOOR:.1e}")
    ax.axhline(-ISO_FLOOR, color="tab:purple", ls="--", lw=1.1)
    ax.axhline(ACCUM_COS_BAR, color="tab:red", ls=":", lw=1.1,
               label=f"ACCUM |cos| bar {ACCUM_COS_BAR}")
    ax.axhline(-ACCUM_COS_BAR, color="tab:red", ls=":", lw=1.1)
    ax.set_xlabel("front index t")
    ax.set_ylabel("cos(front_t, front_{t-1})")
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title("FRONT PERSISTENCE — mutually correlated (accumulate) or "
                 "rotating (anti-/de-correlated)?", fontsize=9.5)
    fig.suptitle("E194 (1/3) — the front-overlap read on the sign path "
                 "(opt2's a_sign rebuilt bit-exactly; k=1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_static_fine(path, phaseB, e192_r2):
    rows = phaseB["rows"]
    coarse = e192_r2["rows"]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.plot([r["D"] for r in coarse], [r["gm12"] for r in coarse], "x",
            ms=9, mew=2.2, color="dimgray",
            label="e192's committed coarse grid (NO point in (1.5, 2.0))")
    ax.plot([r["D"] for r in rows], [r["gm12"] for r in rows], "o-", ms=5,
            lw=2.0, color="navy", label="E194 fine re-read (0.05 steps)")
    ax.axhline(SHUT_BAR, ls="--", lw=1.2, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar")
    ax.axvline(PATH_DKILL_COMMITTED, color="crimson", ls=":", lw=1.6,
               label=f"the PATH's kill D {PATH_DKILL_COMMITTED:.4f}")
    if phaseB["refined_static_edge"] is not None:
        ax.axvline(phaseB["refined_static_edge"], color="navy", ls="-.",
                   lw=1.6,
                   label=f"refined static edge "
                         f"{phaseB['refined_static_edge']:.4f}")
    ax.axvline(SIGN_EDGE_COMMITTED, color="dimgray", ls=":", lw=1.2,
               label=f"committed coarse edge {SIGN_EDGE_COMMITTED}")
    ax.set_xlabel(r"static displacement $D$ along sign(g_0)/||sign(g_0)||$")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.legend(fontsize=7.8, loc="upper right")
    ax.set_title("E194 (2/3) — THE STATIC-ARTIFACT CHECK: the static "
                 "sign-ray profile re-read at fine D (was 1.75-alive a lucky "
                 "point on a steep slope?)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_k_ladder(path, ladder_rows, arms, accum, eff_rows):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    ax = axes[0]
    ks = [r["k"] for r in ladder_rows if isinstance(r["k"], int)]
    ds = [r["D_read"] for r in ladder_rows if isinstance(r["k"], int)]
    labels = [f"k={k}" for k in ks]
    colors = ["crimson", "darkorange", "seagreen", "navy", "gray"][:len(ks)]
    ax.bar(labels, ds, color=colors, alpha=0.85)
    for i, r in enumerate(ladder_rows):
        if not isinstance(r["k"], int):
            ax.axhline(r["D_read"], color="dimgray", ls="-.", lw=1.8)
            ax.text(0.02, r["D_read"] + 0.06,
                    f"k=inf (static R2, loaded): edge "
                    f"{r['D_read']:.4f}", fontsize=8, color="dimgray")
    ax.axhline(PATH_DKILL_COMMITTED, color="crimson", ls=":", lw=1.2,
               label=f"opt2's committed path kill {PATH_DKILL_COMMITTED:.4f}")
    ax.axhline(RAW_DKILL_COMMITTED, color="k", ls=":", lw=1.0,
               label=f"raw-g rung {RAW_DKILL_COMMITTED:.4f} (opt1c)")
    for i, (k, d) in enumerate(zip(ks, ds)):
        ax.text(i, d + 0.05, f"{d:.4f}", ha="center", fontsize=8.5)
    ax.set_ylabel("D_read (densified kill-D or final D)")
    ax.set_xlabel("front re-computation rate (k steps between refreshes; "
                  "k=1 = every step = opt2's path)")
    ax.legend(fontsize=7.8, loc="upper left")
    ax.set_title("THE K-LADDER — does the kill-D move with the "
                 "re-computation rate? (matched per-step L2 1.6543)",
                 fontsize=10)

    ax = axes[1]
    for k, a in arms.items():
        j = [r for r in a["journal"] if not r["post_kill"]]
        col = colors[(K_LADDER.index(k) if k in K_LADDER else 0)
                     % len(colors)]
        ax.plot([r["step"] for r in j], [r["cum_disp"] for r in j], "o-",
                ms=5, lw=1.8, color=col, label=f"k={k} (walked)")
        n = len(j)
        if n >= 2:
            ns = np.arange(1, n + 1)
            ax.plot(ns, STEP_L2 * ns, ":", lw=0.9, color=col, alpha=0.5)
            ax.plot(ns, STEP_L2 * np.sqrt(ns), "--", lw=0.9, color=col,
                    alpha=0.5)
    ax.plot([], [], ":", color="gray", label="ballistic n*L2 (colinear max)")
    ax.plot([], [], "--", color="gray", label="diffusive L2*sqrt(n)")
    ax.set_xlabel("wash step (pre-kill)")
    ax.set_ylabel(r"cumulative $D=\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=7.4, loc="upper left")
    mac = accum["prekill_mean_abs_cos"]
    ax.set_title("THE ACCUMULATION READ — displacement vs the ballistic/"
                 f"diffusive baselines; pre-kill mean|cos(fronts)| "
                 + (f"{mac:.4f}" if mac is not None else "n/a")
                 + f" (floor {ISO_FLOOR:.1e}, bar {ACCUM_COS_BAR})",
                 fontsize=9.5)
    fig.suptitle("E194 (3/3) — the re-computation-rate ladder at matched "
                 "per-step L2", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
