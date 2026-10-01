"""OPT2 — THE DENSITY QUESTION (the trimmed sign-carrier cell; R59's CUT 1
trimmed r58's opt2 to the density arms; the CPU-lane terminal cell of the
optimizer arc).

WHY: opt1 (T139) decomposed the wash kill into CLOCK (Adam's arithmetic:
1.6543/step at lr*sqrt(N), sign-normalized) and GATE (D ~ 2.49-2.84 in
every Adam arm). opt1c (T143) found the raw-gradient DIRECTION at Adam's
size kills at D 0.9203 — BELOW the sign path's 2.5 — magnitude-informative
steps are MORE lethal per displacement, so flattening LOSES lethality
(r59's CUT 1: the WARMV protective arm is dead on arrival). e192 (T146)
mapped the STATIC sign-ray at 2.5 on this organism; the chart (T150) found
the SHUFFLED sign inert — gradient structure is in the coordinate PAIRING.
THE REMAINING QUESTION IS DENSITY: does the lethal structure live in the
NUMBER OF COORDINATES TOUCHED — a sparse |g|-selected structured front, or
dense every-coordinate flattening?

THE CELL: the licensed e185 wash cell VERBATIM (opt1's conventions: root
runs/checkpoints/e131_consolidated_e113.pt, seed-10902 stream, bit-
identical batches — 16 e170 neutral-bank anchors + 16 random corpus
windows, TRUE targets, full-token CE, clip 1.0) with ONLY the update rule
swapped. At MATCHED per-step L2 (STEP_L2 = 1.6542880535125732, opt1 A0's
committed step-1 L2, recomputed at t=0 and bit-gated):

  (a) a_topk10  TOPK-10% — update ONLY the top-10% |g| coordinates,
                 direction g's (the gradient's own magnitudes within the
                 front, L2-normalized), scaled to matched L2.
  (b) a_topk50  TOPK-50% — same at 50% (the density ladder's midpoint).
  (c) a_sign    SIGN-STRETCH GATE — the FULL sign(g) direction at matched
                 L2 for a 2x longer horizon (to D >= 2x the sign edge =
                 5.0): the stretch-invariance check — does the sign kill
                 move with horizon? (r59: the SIGN arm demoted from
                 discrimination to stretch-invariance gate; its kill in
                 the 2.5 band is near-predicted.)
  reference    = opt1's A0 (loaded COMMITTED, never rerun) + opt1c's raw
                 kill 0.9203 (committed rung) + e192's static terrain rays
                 (loaded committed, the overlay) + opt1b's bleed curve.

READS: g-12 EVERY step (opt1c's bracket-tightest cadence; each step moves
1.65 L2 so one-step brackets are the minimum); g-0 battery + CE_R at the
checkpoint cadence {1,2,4,8,16}; D(t) cumulative + per-step + pre-clip
grad norms + batch CE per step; per-checkpoint alignment at BOTH
EVALUATION POINTS — THE CHART'S ESTIMATOR LESSON (T150: opt1's committed
alignment evaluates the fact gradient at the POST-STEP state; the same
displacement at the MATCHED point reads differently — the +0.0986 ->
+0.0396 attenuation vs the -0.0385 post-step flip). BOTH reported:
post_step = cos(delta_s, grad_fact(theta_s)) (opt1 verbatim) and
matched_point = cos(delta_s - delta_s_prev, grad_fact(theta_s_prev)) (the
increment evaluated at the point it began — at s=1 exactly the chart's
same-point convention, gated vs its committed 0.03955) plus the
root-anchored cumulative read cos(delta_s, grad_fact(theta_0)). Top-k set
churn per step (r58's stated read: the mask follows g — the drifting
front is a READ, not a bug) + per-checkpoint cos vs the committed terrain
directions (e191's u_g and sign(g_0)) — which ray each arm rides.

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - DENSITY-CARRIES: "fires if the sparse arms (TOPK-10/50) KILL while
    the full sign arm at matched L2 spares beyond its 2.5 edge — the
    lethal carrier is the sparse structured front, not dense flattening."
  - DENSITY-SPARES: "fires if the sparse arms spare (survive past the
    sign edge alive) while dense killing proceeds — density itself
    carries lethality; structure rides on which coordinates."
  - GRADED: "any partial pattern — the arm table verbatim; the kill-D
    ordering (topk-10, topk-50, sign, raw) reported as the density
    ladder."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * the arms = the licensed cell with delta_t = -STEP_L2 * u_t where
    u_t is (a) g_t[TopK(|g_t|)] restricted, L2-normalized on its support
    (k = round(frac*N), frac in {0.10, 0.50}); (b) sign(g_t)/||sign(g_t)||
    (zeros of g stay zero). g_t = the POST-CLIP batch gradient (the clip
    is a positive rescale: direction clip-invariant; pre-clip norm
    recorded). Per-step L2 == STEP_L2 EXACTLY by construction (asserted
    in code every step; max deviation reported per arm — the matched-L2
    assertion the dispatch demands).
  * KILL = g-12 <= 0.27 (the arc's SHUT bar) at a read, wherever it
    lands (no window veto — unlike opt1c the ladder IS the result; a
    kill below 0.92, between rungs, or past the edge is a rung
    placement, not a null). SPARE-past-edge = alive (> 0.27) at the
    first read with D >= SIGN_EDGE 2.5 AND at the arm's final read.
    D_kill = the linear-in-D interpolation of the 0.27 crossing between
    the last alive and first dead read, DENSIFIED along the killing step
    at f in {0.2,0.4,0.6,0.8} (parameter points ON the trajectory, not
    new steps; opt1c's convention verbatim); both raw and densified
    brackets reported, the densified one adjudicates.
  * stop per arm = whichever FIRST of kill (first dead read; stop after
    densification) / D >= D_TARGET 5.0 (= 2x the sign edge: the 2x longer
    horizon) alive at that read / STEP_CAP 16 (drift-stall safety;
    report as cap).
  * DENSITY-CARRIES fires iff BOTH sparse arms (a_topk10, a_topk50) kill
    AND a_sign spares past the edge (alive at the 2.5-crossing read AND
    its final read). DENSITY-SPARES fires iff BOTH sparse arms spare
    past the edge AND a_sign kills. Everything else (one-of-each, caps,
    stalls below the edge) is GRADED. Arms that never reach D 2.5 CANNOT
    fire a spare clause (cap-limited ambiguity; projections never
    adjudicate). Composite order frozen DENSITY-CARRIES ->
    DENSITY-SPARES -> GRADED.
  * the density ladder (always reported): the kill-D ordering of
    raw 0.9203 (opt1c committed) / topk-10 / topk-50 / sign (this cell)
    / A0-Adam 2.4893 (opt1 committed) — kills ordered by D_kill, spares
    by final D.
  * stretch-invariance co-read (the sign arm's own question): sign
    kill-D vs the 2.5 static edge / A0's 2.4893 — does the sign kill
    move with a 2x horizon? Reported, not adjudicated (r59 demoted the
    arm to gate).

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any training): the
standard cell gates (corpus ZEPH count 0; splice mix {FLORIZEL: 19,
ELIZABETH: 41}; battery shapes 60 x (130 +- j); neutral bank 16 starts
bit-equal e185's stored list; root gated vs e151's before-cells) PLUS
the t=0 BIT-GATE vs opt1's committed A0 values (step-1 batch md5 ==
e185's stored hash; fresh AdamW-recipe step-1 CE / pre-clip gnorm / L2
== the committed row; tol 5e-6 bit / 0.05 fallback; abort on control
failure) PLUS the SAME-POINT MACHINERY GATE (T150's estimator lesson):
cos(-sign(g_0^t0), grad m12 at theta_0) must reproduce the chart's
committed 0.039550412581627135 (scale-invariance makes this the SIGN
arm's step-1 matched-point read by construction). WHAT EACH ARM
GUARANTEES: NOTHING — the arms could kill anywhere on the ladder (the
topk arms carry magnitude information, the most lethal ingredient
measured; the sign arm is near-predicted to kill at 2.5); that openness
is the point.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is another agent's — never claimed), torch threads
4 (the e185/opt1 reduction order), SEQUENTIAL arms, chunk-RESUMABLE
(chunks <= 178 s; full state — model + generator + step + journal +
flags — round-trips through runs/opt2/chunk_state.pt after each chunk;
chunk-boundary parameter md5s form a hash chain; opt1c's convention),
per-arm stop rule above, n=1, single stream seed 10902 lineage.
PROGRESSIVE partial metrics.json writes (the outage lesson, six
disruptions today): after the gates pass and after EVERY chunk save; the
final COMPLETE write replaces them.

Outputs: runs/opt2/{metrics.json, opt2_density_ladder.png,
opt2_trajectory.png, chunk_state.pt}; checkpoints
runs/checkpoints/opt2_<arm>_s<final>.pt. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds).

Run:  cd lab && python opt2_density.py    (OPT2_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187/opt1/opt1b/opt1c)

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

SMOKE = os.environ.get("OPT2_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt2 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1/opt1b/opt1c's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
E192_METRICS = E43.REPO / "runs" / "e192" / "metrics.json"
ECHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
E191_DIR_CK = CKPT_DIR / "e191_static_dir_u.pt"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the arms + the run envelope (dispatch-frozen) -------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1/opt1b/opt1c)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; the arms have NO lr)
STEP_L2_COMMITTED = 1.6542880535125732   # opt1 A0's committed step-1 L2 = the matched per-step L2
CE1_COMMITTED = 1.356567621231079        # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253       # opt1 A0's committed step-1 pre-clip grad norm
A0_DKILL_COMMITTED = 2.4892616271972656  # opt1 A0's committed kill displacement
RAW_DKILL_COMMITTED = 0.9203406595225093 # opt1c's committed densified raw-direction kill D
SIGN_EDGE_COMMITTED = 2.5                # e192's committed static sign-ray kill D (the 2.5 edge)
W024_ADAM_SAMEPOINT = 0.039550412581627135  # the chart's committed same-point sign read
W024_RAW_SAMEPOINT = 0.09862135965497001    # the chart's committed same-point raw read (context)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
CHUNK_CAP = 178.0                 # dispatch: <=180 s per chunk (margin for step overshoot)
STEP_CAP = 16                     # drift-stall safety cap
D_TARGET = 5.0                    # the 2x-longer horizon = 2x the sign edge 2.5
SIGN_EDGE = SIGN_EDGE_COMMITTED   # the spare clause's edge (frozen)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
SURVIVE_NOTE = 0.50               # SPARE reference line (opt1's convention; report-only)
CKPT_STEPS = (1, 2, 4, 8, 16)     # full-read cadence (g0 + CE_R + alignment)
DENSIFY_F = (0.2, 0.4, 0.6, 0.8)  # along-path kill-bracket densification fractions
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
INTER_ARM_S = 10.0                # politeness stagger between sequential arms
N_PARAM = 2_739_072               # the e131 line's parameter count (asserted at load)
TOPK_FRACS = {"a_topk10": 0.10, "a_topk50": 0.50}
L2_ASSERT_TOL = 1e-5              # the in-code matched-L2 assertion (measured dev ~1e-7)
if SMOKE:                         # true shakedown trims (documented in deviations)
    STEP_CAP, CHUNK_CAP, CKPT_STEPS, DENSIFY_F = 3, 12.0, (1, 2), ()

# ---- gates / references (full precision, = stored metrics; opt1's set) ----------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_SAMEPOINT_TOL = 2e-3            # the chart's own flip-anchor tolerance

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
}
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; opt1's embedded set)
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
    "density_carries": "DENSITY-CARRIES: \"fires if the sparse arms (TOPK-10/50) "
        "KILL while the full sign arm at matched L2 spares beyond its 2.5 edge "
        "— the lethal carrier is the sparse structured front, not dense "
        "flattening.\"",
    "density_spares": "DENSITY-SPARES: \"fires if the sparse arms spare (survive "
        "past the sign edge alive) while dense killing proceeds — density "
        "itself carries lethality; structure rides on which coordinates.\"",
    "graded": "GRADED: \"any partial pattern — the arm table verbatim; the "
        "kill-D ordering (topk-10, topk-50, sign, raw) reported as the "
        "density ladder.\"",
    "operationalizations": "the arms = the licensed e185 wash cell VERBATIM "
        "(root e131, seed-10902 stream, bit-identical batches, TRUE targets, "
        "full-token CE, clip 1.0) with delta_t = -STEP_L2 * u_t: (a) u = "
        "g[TopK(|g|, round(0.10*N))] on its support, L2-normalized; (b) same "
        "at 0.50; (c) u = sign(g)/||sign(g)|| (zeros stay zero); g = the "
        "POST-CLIP batch gradient (direction clip-invariant; pre-clip norm "
        "recorded); per-step L2 == STEP_L2 1.6542880535125732 EXACTLY "
        "(asserted every step; recomputed at t=0 and bit-gated vs opt1's "
        "committed A0 step-1 values); KILL = g-12 <= 0.27 at an EVERY-STEP "
        "read wherever it lands (no window veto — the ladder IS the "
        "result), D_kill = linear-in-D interpolation densified along the "
        "killing step at f in {0.2,0.4,0.6,0.8} (opt1c's convention); "
        "SPARE-past-edge = alive at the first read with D >= 2.5 AND at "
        "the final read; stop = first of kill / D >= 5.0 (= 2x the sign "
        "edge, the 2x-longer horizon) alive / 16-step cap; DENSITY-CARRIES "
        "fires iff BOTH sparse arms kill AND a_sign spares past the edge; "
        "DENSITY-SPARES fires iff BOTH sparse arms spare past the edge AND "
        "a_sign kills; arms never reaching D 2.5 cannot fire a spare "
        "clause (projections never adjudicate); composite order frozen "
        "DENSITY-CARRIES -> DENSITY-SPARES -> GRADED; the density ladder "
        "(kill-D ordering of raw 0.9203 / topk-10 / topk-50 / sign / "
        "A0-Adam 2.4893) always reported.",
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "THE TRIM (r59 CUT 1, the dispatch's letter): r58's opt2 spec'd 6 arms "
    "(REF/REF-RAW/SIGN/TOPK-10/TOPK-50/WARMV); after opt1c (magnitude-"
    "informative steps are MORE lethal: raw g kills at 0.9203 vs sign's 2.5) "
    "the WARMV protective arm is dead on arrival and SIGN demotes from "
    "discrimination to stretch-invariance gate; the surviving cell is THE "
    "DENSITY QUESTION: TOPK-10/50 vs the full sign at matched per-step L2. "
    "The lr/8 stretch of r58's spec is NOT used: the arms are size-pinned at "
    "A0's measured step L2 directly (each step is 1.6543 L2, the gate is "
    "crossed at step ~2, and the horizon extends to 2x the edge instead).",
    "The arms have NO learning rate and NO optimizer state: the update is "
    "size-pinned by construction (per-step L2 = STEP_L2 exactly; the "
    "in-code assertion reports max deviation per arm). The batch CE, "
    "pre-clip grad norms, and input md5s are still recorded per step (the "
    "licensed cell's own reads).",
    "g-12 is read at EVERY step (opt1c's bracket-tightest cadence); g-0 + "
    "CE_R + the dual-estimator alignment at the {1,2,4,8,16} cadence. The "
    "alignment carries BOTH evaluation points per T150's estimator lesson: "
    "post_step cos(delta_s, grad fact(theta_s)) (opt1's committed "
    "convention) AND matched_point cos(delta_s - delta_s_prev, grad "
    "fact(theta_s_prev)) (the increment at the point it began; at s=1 this "
    "is EXACTLY the chart's same-point convention, gated vs its committed "
    "+0.03955) plus the root-anchored cumulative read cos(delta_s, grad "
    "fact(theta_0)).",
    "Top-k set churn recorded per step (Jaccard of consecutive masks; r58's "
    "stated read — the drifting front is a read, not a bug); per-checkpoint "
    "cos vs the committed terrain directions (e191's u_g, sign(g_0)) — "
    "which ray each arm rides. torch.topk's tie handling on CPU is "
    "deterministic (sorted by value); fp32-exact ties are rare — stated, "
    "not sampled.",
    "Chunk cap 178 s (dispatch <= 180 s; the margin absorbs one step "
    "overshoot); in-chunk reads count toward the cap (opt1's FT_TIME_CAP "
    "convention). Chunk state round-trips through runs/opt2/chunk_state.pt "
    "per the opt1b/opt1c convention (model + generator + step + journal + "
    "flags + last top-k mask + previous-checkpoint fact grads); "
    "chunk-boundary parameter md5s form a hash chain.",
    "The reference A0 (opt1), the raw rung (opt1c), the bleed curve "
    "(opt1b via e192's overlays), the terrain rays (e192), and the "
    "same-point anchors (e_chart) are LOADED COMMITTED, never rerun; "
    "hard-bound at runtime (asserts catch any drift of the committed "
    "files vs this file's copies).",
    "Light reads only (g-12/g0 batteries + CE_R + dual-estimator alignment "
    "+ displacement + pre-clip grad norms + topk churn) — no census/"
    "deletions. CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch; the GPU is "
    "another agent's, never claimed); threads 4; n=1; single seed lineage "
    "(10902).",
    "Smoke mode trims: 3-step cap, 12 s chunks, cadence {1,2}, no "
    "densification, no adjudication (verdict stamped SMOKE; nothing "
    "adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1_optimizer_controls.py + lab/opt1c_direction_size.py
# VERBATIM (whose own provenance is lab/e185_noise_wash.py via e187's verified
# copies — the e176n lineage). Copied rather than imported to own the device
# policy.

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
    """The fp32 flat parameter vector (all trainable tensors, net.parameters()
    order — the optimizer's own currency; 2,739,072 elements on this line)."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    """Copy a flat vector back into parameters (update + densification loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ: gradient (wrt all params, at the twin's current
    weights theta) of the fact battery's mean log p(Z) readout at the last
    position. The critic's sign convention: NEGATIVE cos(delta, grad) =
    displacement aligned with the DEATH gradient (W022). Consumes no RNG;
    run on the eval twin, never the training net."""
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


# ------------------------------------------------------------------ the arms

def topk_update(g: torch.Tensor, frac: float, step_l2: float):
    """(a)/(b): delta = -step_l2 * g[TopK(|g|)]/||g[TopK(|g|)]|| on its
    support. Returns (delta, k, mask). The support norm is computed in
    fp64 — an fp32 norm over ~1.4M elements carries ~1e-5 relative
    rounding, which would blur the matched-L2 assertion; with the fp64
    norm the realized per-step L2 dev is ~1e-7 (reported per arm)."""
    k = int(round(frac * g.numel()))
    idx = torch.topk(g.abs(), k).indices
    sub = g[idx]
    nrm = float(torch.norm(sub.double()))
    assert nrm > 0, "top-k subgradient is zero — the arm is undefined"
    delta = torch.zeros_like(g)
    delta[idx] = -step_l2 * (sub / nrm)
    mask = torch.zeros(g.numel(), dtype=torch.bool)
    mask[idx] = True
    return delta, k, mask


def sign_update(g: torch.Tensor, step_l2: float):
    """(c): delta = -step_l2 * sign(g)/||sign(g)|| (zeros stay zero; the
    support norm in fp64, same matched-L2 exactness as topk_update)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


ARM_SPECS = [
    ("a_topk10", "TOPK-10% — update only the top-10% |g| coordinates, "
     "direction g's (magnitude-informative within the front), scaled to "
     "matched per-step L2 1.6543", "topk", 0.10),
    ("a_topk50", "TOPK-50% — same at 50% (the density ladder's midpoint)",
     "topk", 0.50),
    ("a_sign", "SIGN-STRETCH GATE — the full sign(g) direction at matched "
     "L2 for a 2x longer horizon (to D >= 2x the 2.5 sign edge): does the "
     "sign kill move with horizon? (r59: near-predicted to kill ~2.5)",
     "sign", None),
]
ARM_ORDER = [t for t, _, _, _ in ARM_SPECS]
ARM_DESC = {t: d for t, d, _, _ in ARM_SPECS}


# ------------------------------------------------------------------ kill bracket

def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    """The densified bracket: insert the along-step reads, interpolate in
    the finest sub-bracket (opt1c's convention; this value adjudicates)."""
    pts = [(d0, v0)] + [(r["D"], r["gm12"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


# ------------------------------------------------------------------ the wash

def run_arm(tag, net0, anchor, train_ids, itos, r_eval_xy,
            gm12_ids, g0_ids, zid, theta0, root_grads, terrain_dirs,
            rd, write_partial, metrics_stub):
    """The licensed wash cell (opt1's arithmetic at target_mode='true': the
    seed-10902 aj/rj draw order, identical batch construction, full-token CE,
    clip 1.0) with the arm's masked/scaled update rule. Every-step g-12;
    cadenced full reads with the dual-estimator alignment; chunk-resumable
    (runs/opt2/chunk_state.pt)."""
    kind = "sign" if tag == "a_sign" else "topk"
    frac = None if tag == "a_sign" else TOPK_FRACS[tag]
    chunk_path = rd / "chunk_state.pt"
    ckpt_set = set(s for s in CKPT_STEPS if s <= STEP_CAP)

    state = None
    if chunk_path.exists():                       # resume / recovery hook
        st = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if st.get("arm") == tag:
            state = st
            log(f"[{tag}] resumed from chunk_state (step {st.get('step')}, "
                f"chunks {len(st.get('chunks_prov', []))}, stop "
                f"{(st.get('stop') or {}).get('kind') if st.get('stop') else None})")

    if state is None:
        net = copy.deepcopy(net0)
        net.train()
        state = {"arm": tag, "step": 0, "net_sd": None,
                 "gen": torch.Generator().manual_seed(FREEZE_SEED).get_state(),
                 "journal": [], "prev_ck": None, "prev_grads": None,
                 "prev_mask": None, "stop": None, "zeph": 0,
                 "max_l2_dev": 0.0, "jaccards": [], "x_hashes": {},
                 "chunks_prov": [], "md5_chain": [], "train_seconds": 0.0}
    else:
        net = copy.deepcopy(net0)
        net.load_state_dict(state["net_sd"])
        net.train()

    evl = copy.deepcopy(net0)          # CPU eval twin (reads + alignment)
    evl.eval()
    gen = torch.Generator()
    gen.set_state(state["gen"])
    n_anc = anchor.shape[0]
    step = state["step"]
    stop = state["stop"]
    prev_gm12, prev_d = state.get("last_gm12"), state.get("last_d")
    t_start = time.time()
    arm_t0 = time.time()
    steps_since_save = 0

    while stop is None:
        # ---- chunk boundary (state round-trips through the file; the
        # same-process load path is exercised by resume, opt1c convention)
        if (time.time() - t_start) > CHUNK_CAP and steps_since_save > 0:
            state.update({"step": step,
                          "net_sd": {k: v.detach().cpu().clone()
                                     for k, v in net.state_dict().items()},
                          "gen": gen.get_state(), "stop": None,
                          "last_gm12": prev_gm12, "last_d": prev_d,
                          "train_seconds": state["train_seconds"]
                          + (time.time() - arm_t0)})
            save_chunk(chunk_path, state, net)
            write_partial(metrics_stub, phase=f"chunk save ({tag} s{step})")
            # reload from the file (proves the round-trip; opt1c convention)
            st = torch.load(chunk_path, map_location="cpu",
                            weights_only=False)
            net.load_state_dict(st["net_sd"])
            gen.set_state(st["gen"])
            t_start = time.time()
            steps_since_save = 0
        step += 1
        if step > STEP_CAP:
            stop = {"kind": "cap", "step": step - 1,
                    "reason": f"step cap {STEP_CAP}",
                    "final_cum_disp": prev_d, "final_gm12": prev_gm12}
            break
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                state["zeph"] += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        state["x_hashes"][step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        prev = flat_params(net)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        net.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        g_t = torch.cat([p.grad.detach().reshape(-1)
                         for p in net.parameters()]).clone()   # post-clip
        if kind == "topk":
            delta, k, mask = topk_update(g_t, frac, STEP_L2_COMMITTED)
            if state["prev_mask"] is not None:
                inter = int((mask & state["prev_mask"]).sum())
                state["jaccards"].append(inter / (2 * k - inter))
            state["prev_mask"] = mask
        else:
            delta = sign_update(g_t, STEP_L2_COMMITTED)
            k = None
        # THE MATCHED-L2 ASSERTION (the dispatch's check): per-step L2 is
        # STEP_L2 exactly by construction — measured in fp64 (an fp32 norm
        # reduction over 2.7M elements carries ~1e-5 rounding that would
        # blur the assertion), asserted, reported.
        l2dev = abs(float(torch.norm(delta.double())) - STEP_L2_COMMITTED)
        assert l2dev < L2_ASSERT_TOL, \
            f"{tag}: per-step L2 dev {l2dev:.2e} — the matched-L2 assertion"
        state["max_l2_dev"] = max(state["max_l2_dev"], l2dev)
        load_flat(net, prev + delta)
        cur = flat_params(net)
        cum_disp = float(torch.norm(cur - theta0))
        row = {"step": step, "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm, "k": k,
               "jaccard": (state["jaccards"][-1]
                           if state["jaccards"] else None),
               "elapsed_s": round(time.time() - t_start, 1)}
        # ---- EVERY-STEP g-12 (the kill ruler; opt1c's cadence)
        evl.load_state_dict({k_: v.detach().cpu().clone()
                             for k_, v in net.state_dict().items()})
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        row["gm12"] = gz["mean_pz"]
        row["frac_argmax_z"] = gz["frac_argmax_z"]
        # ---- cadenced full reads (g0 + CE_R + dual-estimator alignment)
        if step in ckpt_set:
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            g_g0 = fact_grad(evl, g0_ids, zid)
            g_m12 = fact_grad(evl, gm12_ids, zid)
            d_cum = cur - theta0
            align = {
                "post_step": {"g0": cos64(d_cum, g_g0),
                              "m12": cos64(d_cum, g_m12)},
                "root_point": {"g0": cos64(d_cum, root_grads["g0"]),
                               "m12": cos64(d_cum, root_grads["m12"])},
                "terrain": {"u_g": cos64(d_cum, terrain_dirs["u_g"]),
                            "sign_g0": cos64(d_cum, terrain_dirs["sign"])},
            }
            if state["prev_ck"] is not None and state["prev_grads"] is not None:
                d_inc = d_cum - state["prev_ck"]
                align["matched_point"] = {
                    "g0": cos64(d_inc, state["prev_grads"]["g0"]),
                    "m12": cos64(d_inc, state["prev_grads"]["m12"])}
            else:                          # s=1: the chart's same-point read
                align["matched_point"] = {
                    "g0": cos64(d_cum, root_grads["g0"]),
                    "m12": cos64(d_cum, root_grads["m12"])}
            state["prev_ck"] = d_cum.clone()
            state["prev_grads"] = {"g0": g_g0.clone(), "m12": g_m12.clone()}
            row.update({"g0": gz0["mean_pz"], "ce_r": ce_r, "align": align})
            log(f"  [{tag}] CKPT +{step:3d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| {cum_disp:.4f} "
                f"post {align['post_step']['m12']:+.3f} "
                f"match {align['matched_point']['m12']:+.3f} "
                f"vsG {align['terrain']['u_g']:+.3f}")
        state["journal"].append(row)
        steps_since_save += 1
        prev_gm12, prev_d = row["gm12"], cum_disp
        if gz["mean_pz"] <= SHUT_BAR:       # KILL: densify + stop (opt1c)
            dens = []
            for f_ in DENSIFY_F:
                pt_flat = prev + f_ * (cur - prev)
                load_flat(evl, pt_flat)
                evl.eval()
                gzd = battery_cell(evl, gm12_ids, zid)
                dens.append({"f": f_, "gm12": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            d_raw = interp_d_kill(state["journal"][-2]["gm12"]
                                  if len(state["journal"]) >= 2
                                  else root_gm12_DEFAULT, row["gm12"],
                                  state["journal"][-2]["cum_disp"]
                                  if len(state["journal"]) >= 2 else 0.0,
                                  cum_disp)
            d_dens = dens_d_kill(state["journal"][-2]["gm12"]
                                 if len(state["journal"]) >= 2
                                 else root_gm12_DEFAULT, row["gm12"],
                                 state["journal"][-2]["cum_disp"]
                                 if len(state["journal"]) >= 2 else 0.0,
                                 cum_disp, dens)
            stop = {"kind": "kill", "step": step,
                    "gm12_at_kill": row["gm12"],
                    "D_kill_raw": d_raw, "D_kill": d_dens, "dens": dens,
                    "reason": "g-12 <= SHUT at an every-step read"}
            log(f"  [{tag}] KILL at +{step}: g-12 {row['gm12']:.4f} "
                f"D_kill(dens) {d_dens:.4f} (raw {d_raw:.4f})")
        elif cum_disp >= D_TARGET:          # 2x horizon reached alive
            stop = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {D_TARGET}",
                    "final_cum_disp": cum_disp, "final_gm12": row["gm12"]}
            log(f"  [{tag}] D_TARGET {D_TARGET} reached alive at +{step} "
                f"(g-12 {row['gm12']:.4f}, D {cum_disp:.4f})")

    state.update({"step": step, "stop": stop,
                  "net_sd": {k_: v.detach().cpu().clone()
                             for k_, v in net.state_dict().items()},
                  "gen": gen.get_state(), "last_gm12": prev_gm12,
                  "last_d": prev_d,
                  "train_seconds": state["train_seconds"]
                  + (time.time() - arm_t0)})
    save_chunk(chunk_path, state, net)
    net.eval()
    return state, net


root_gm12_DEFAULT = 0.9155886173248291       # the root's committed g-12


def save_chunk(chunk_path, state, net):
    state["md5_chain"].append(flat_md5(net))
    state["chunks_prov"].append({"chunk": len(state["md5_chain"]),
                                 "arm": state["arm"],
                                 "step": state["step"],
                                 "t": round(time.time() - T0, 1)})
    torch.save(state, chunk_path)
    log(f"[{state['arm']}] chunk saved (step {state['step']}, "
        f"chain[{len(state['md5_chain'])}])")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("opt2_smoke" if SMOKE else "opt2")
    log(f"OPT2 THE DENSITY QUESTION (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), sequential "
        f"arms, chunks <= {CHUNK_CAP + 2:.0f}s (ckpt-resumable), per-arm "
        f"stop kill/{D_TARGET}/cap{STEP_CAP}, n=1, seed lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT1_METRICS, OPT1C_METRICS, E192_METRICS, ECHART_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    e192m = json.loads(E192_METRICS.read_text(encoding="utf-8"))
    chartm = json.loads(ECHART_METRICS.read_text(encoding="utf-8"))
    a0 = opt1m["arms"]["a0_adamw_ref"]
    a0_s1 = next(r for r in a0["traj"] if r["step"] == 1)
    # hard-bind the committed references (asserts catch committed-file drift)
    assert abs(a0_s1["cum_disp"] - STEP_L2_COMMITTED) < 1e-12
    assert abs(a0_s1["ce_batch"] - CE1_COMMITTED) < 1e-12
    assert abs(a0_s1["preclip_gnorm"] - GN1_COMMITTED) < 1e-12
    assert abs(a0["tstar"]["D_at_kill"] - A0_DKILL_COMMITTED) < 1e-12
    raw_stop = opt1cm["adjudication"]["stop"]
    assert abs(raw_stop["D_kill_interp"] - RAW_DKILL_COMMITTED) < 1e-12
    e192_sign_kill = e192m["adjudication"]["bars"]["TERRAIN_ONE_PICTURE"]["sign_kill_D"]
    assert abs(e192_sign_kill - SIGN_EDGE_COMMITTED) < 1e-12
    chart_same = chartm["gates"]["G_W024"]
    assert abs(chart_same["cos_adamview_m12_at_t0_samepoint"]
               - W024_ADAM_SAMEPOINT) < 1e-12
    assert abs(chart_same["cos_mov_m12_at_t0_samepoint"]
               - W024_RAW_SAMEPOINT) < 1e-12
    a0_path = a0["ckpt_table"]                    # the committed Adam path
    e192_profiles = e192m["profiles"]             # the committed terrain rays
    bleed = e192m["overlays"]["bleed_curve"]      # opt1b's committed bleed
    log("parents: opt1 (A0 path, t_x "
        f"{a0['tstar']['t_x']:.2f}, D_kill {A0_DKILL_COMMITTED:.4f}); opt1c "
        f"(raw rung D_kill {RAW_DKILL_COMMITTED:.4f}); e192 (terrain rays, "
        f"sign edge {SIGN_EDGE}); e_chart (same-point anchors) — loaded "
        "COMMITTED, never rerun")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185/opt1)
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
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "expected_shapes": {"-12": [60, PRE - 12], "0": [60, PRE],
                            "12": [60, PRE + 12]},
        "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at ctx "
                "offsets {-12,0,+12}, e185's convention verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- the neutral stream (e170 VERBATIM via e185 arm C)
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
                         "construction VERBATIM (= e185's control stream; "
                         "FIXED content, not reseeded)"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], ("neutral bank drifted vs e185's stored "
                              f"starts: {n_starts}")
    log("G_ANCHOR: neutral bank bit-matches e185's stored 16 starts: PASS")

    # ---------------- root net + gate vs e151
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    theta0 = flat_params(net0)
    assert theta0.numel() == N_PARAM, \
        f"parameter count {theta0.numel()} != {N_PARAM} (the e131 line)"
    log(f"root: {ROOT_CK} (meta: {root_meta}); {N_PARAM} params")

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
    log(f"G_ROOT (vs e151 before-cells, light set): max|diff| {rmax:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")

    # ---------------- G_T0: the fresh t=0 wash gradient + AdamW step
    # (opt1c/e_chart convention verbatim) — the matched-L2 reference gate
    gen1 = torch.Generator().manual_seed(FREEZE_SEED)
    aj = torch.randint(anchor_neutral.shape[0], (ANCH_BS,), generator=gen1)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=gen1)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
    x1 = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y1 = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
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
            "adamw_step1_L2_committed": STEP_L2_COMMITTED,
            "step_l2_used": STEP_L2_COMMITTED,
            "pass": bool(x1_md5 == E185_XHASH[1]
                         and abs(ce1 - CE1_COMMITTED) < G_FALLBACK_TOL
                         and abs(gn1 - GN1_COMMITTED) < G_FALLBACK_TOL
                         and abs(disp1 - STEP_L2_COMMITTED) < G_FALLBACK_TOL),
            "note": "the t=0 bit-gate vs opt1's committed A0 row (the "
                    "matched-L2 reference is A0's own step-1 displacement; "
                    "measured on this device, this process)"}
    log(f"G_T0 (t=0 bit-identity vs opt1 A0): md5 "
        + ("match" if G_T0["step1_x_md5_match_e185"] else "DRIFT")
        + f", CE |d| {abs(ce1 - CE1_COMMITTED):.2e}, gn |d| "
        f"{abs(gn1 - GN1_COMMITTED):.2e}, L2 |d| "
        f"{abs(disp1 - STEP_L2_COMMITTED):.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — abort (control "
                           "failure)")

    # root fact grads (the root-anchored + s=1 matched-point estimators)
    evl_root = copy.deepcopy(net0)
    evl_root.eval()
    root_grads = {"g0": fact_grad(evl_root, g0_ids, zid),
                  "m12": fact_grad(evl_root, gm12_ids, zid)}
    # G_SAMEPOINT: the same-point machinery gate (T150's estimator lesson):
    # the SIGN arm's step-1 matched-point read is cos(-sign(g_0^t0), grad
    # m12 at theta_0) BY CONSTRUCTION (scale-invariance) — must reproduce
    # the chart's committed value.
    sp_sign = cos64(-torch.sign(g0_t0), root_grads["m12"])
    sp_raw = cos64(-g0_t0, root_grads["m12"])
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
        "note": "T150's estimator lesson made a HARD machinery gate: the "
                "SIGN arm's step-1 matched-point alignment (cos(-sign(g_0), "
                "grad m12 at theta_0), scale-invariant) must reproduce the "
                "chart's committed same-point anchors — the dual-estimator "
                "reads are anchored at the matched-point end (the post-step "
                "end is anchored by G_T0 reproducing opt1's committed row)",
    }
    log(f"G_SAMEPOINT: sign-arm s1 matched-point {sp_sign:+.5f} vs chart "
        f"{W024_ADAM_SAMEPOINT:+.5f} (|d| "
        f"{abs(sp_sign - W024_ADAM_SAMEPOINT):.1e}); raw sanity {sp_raw:+.5f} "
        f"vs {W024_RAW_SAMEPOINT:+.5f}: "
        + ("PASS" if G_SAMEPOINT["pass"] else "FAIL"))
    if not G_SAMEPOINT["pass"]:
        raise RuntimeError("same-point machinery gate FAILED — the dual "
                           "estimators would not reproduce the chart's "
                           "anchors; abort (control failure)")

    # terrain directions (committed: e191's u_g bit-gated; sign(g_0) fresh)
    terrain_dirs = {}
    G_TERRAIN = {"u_g_ckpt": str(E191_DIR_CK), "pass": None}
    if E191_DIR_CK.exists():
        uck = torch.load(E191_DIR_CK, map_location="cpu", weights_only=False)
        u_g = uck["u"].detach().clone().float()
        fresh = g0_t0 / torch.norm(g0_t0)
        G_TERRAIN.update({
            "loaded_u_md5_match_fresh": bool(
                hashlib.md5(u_g.numpy().tobytes()).hexdigest()
                == hashlib.md5(fresh.numpy().tobytes()).hexdigest()),
            "fresh_cos64_loaded": cos64(fresh, u_g),
            "root_md5_match_checkpoint": bool(
                flat_md5(net0) == uck.get("theta0_md5"))})
        G_TERRAIN["pass"] = bool(G_TERRAIN["root_md5_match_checkpoint"]
                                 and abs(float(torch.norm(u_g)) - 1.0) < 1e-5)
        terrain_dirs["u_g"] = u_g
    else:
        terrain_dirs["u_g"] = g0_t0 / torch.norm(g0_t0)
        G_TERRAIN["pass"] = False
        G_TERRAIN["note"] = "e191 u checkpoint absent — u_g recomputed fresh"
    log(f"G_TERRAIN (e191 u_g): "
        + ("PASS" if G_TERRAIN["pass"] else "RECOMPUTED (disclosed)"))
    terrain_dirs["sign"] = torch.sign(g0_t0).float()
    terrain_dirs["sign"] = terrain_dirs["sign"] / torch.norm(
        terrain_dirs["sign"])

    # the t=0 density read (a READ, not a gate): what the fronts carry
    gabs = g0_t0.abs()
    density_t0 = {}
    for nm, frac in TOPK_FRACS.items():
        k = int(round(frac * N_PARAM))
        vals = torch.topk(gabs, k).values
        density_t0[nm] = {"k": k,
                          "energy_frac": float(
                              (vals ** 2).sum() / (gabs ** 2).sum())}
    n_zero = int((g0_t0 == 0).sum())
    density_t0["sign_zeros"] = n_zero
    log("t=0 density read: top-10% energy "
        f"{density_t0['a_topk10']['energy_frac']:.4f}, top-50% "
        f"{density_t0['a_topk50']['energy_frac']:.4f}, sign zeros {n_zero}")

    # =====================================================================
    # metrics stub + progressive writes (the outage lesson)
    # =====================================================================
    metrics_stub: dict = {"gates": {}, "arms_partial": {}}

    def write_partial(stub: dict, phase: str):
        stub.update({
            "experiment": "opt2_density",
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

    # ---------------- G_STREAM: the seed-10902 stream machinery verified
    # WITHOUT training — regenerate the first 10 steps' input batches from
    # a fresh generator and md5 them vs e185's stored hashes. (An arm that
    # kills at step 1 stops consuming the stream, so the per-arm md5 gate
    # can only cover what that arm actually drew; this pure-machinery gate
    # covers the construction itself, steps 1..10, before any training.)
    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 11):
        aj_ = torch.randint(anchor_neutral.shape[0], (ANCH_BS,),
                            generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_10_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values())),
                "note": "the arms' own draw order is this exact generator "
                        "sequence (verified per consumed step in G_INPUTS); "
                        "this gate verifies the construction for steps the "
                        "arms may never reach"}
    log("G_STREAM: seed-10902 stream construction md5-matches e185's "
        "stored hashes (steps 1..10): "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    write_partial(metrics_stub, "gates passed")
    metrics_stub["question"] = (
        "does the lethal structure of the sign-normalized wash step live in "
        "the NUMBER OF COORDINATES TOUCHED — a sparse |g|-selected "
        "structured front, or dense every-coordinate flattening? The "
        "licensed e185 wash cell VERBATIM at matched per-step L2 1.6543: "
        "TOPK-10%, TOPK-50%, SIGN (2x horizon) + committed references "
        "(opt1 A0, opt1c raw 0.9203, e192 terrain).")
    metrics_stub["gates"] = {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                             "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                             "G_ROOT": G_ROOT, "G_T0": G_T0,
                             "G_SAMEPOINT": G_SAMEPOINT,
                             "G_TERRAIN": G_TERRAIN, "G_STREAM": G_STREAM,
                             "density_t0": density_t0}
    write_partial(metrics_stub, "gates passed")
    log("gates passed; partial metrics written")

    # =====================================================================
    # THE ARMS (sequential; chunk-resumable; per-arm stop rule)
    # =====================================================================
    arms_out: dict = {}
    for k_i, (tag, _desc, _kind, _frac) in enumerate(ARM_SPECS):
        if k_i and not SMOKE:
            log(f"[stagger] {INTER_ARM_S:.0f}s before {tag}")
            time.sleep(INTER_ARM_S)
        log("=" * 78)
        log(f"ARM {tag} — {ARM_DESC[tag]}; cap {STEP_CAP} steps, D_target "
            f"{D_TARGET}, chunks <= {CHUNK_CAP + 2:.0f}s")
        state, _net = run_arm(tag, net0, anchor_neutral, train_ids, itos,
                              r_eval_xy, gm12_ids, g0_ids, zid, theta0,
                              root_grads, terrain_dirs, rd, write_partial,
                              metrics_stub)
        arms_out[tag] = state
        assert state["zeph"] == 0, f"{tag}: name token leaked into a window"
        metrics_stub["arms_partial"][tag] = _arm_summary(state)
        write_partial(metrics_stub, f"arm {tag} complete")

    # =====================================================================
    # G_INPUTS: shared input stream across arms AND vs e185's stored hashes
    # =====================================================================
    n_shared = min(len(arms_out[t]["x_hashes"]) for t in ARM_ORDER)
    per_step, vs_stored = {}, {}
    for step in range(1, n_shared + 1):
        hs = {t: arms_out[t]["x_hashes"].get(step) for t in ARM_ORDER}
        same = bool(all(h is not None for h in hs.values())
                    and len(set(hs.values())) == 1)
        per_step[step] = {**hs, "identical": same}
        if step in E185_XHASH:
            vs_stored[step] = {"match": bool(hs[ARM_ORDER[0]]
                                             == E185_XHASH[step])}
    G_INPUTS = {"per_step": per_step, "vs_e185_stored": vs_stored,
                "pass": bool(all(v["identical"] for v in per_step.values())
                             and all(v["match"]
                                     for v in vs_stored.values()))}
    assert G_INPUTS["pass"], ("input streams diverged across arms (or vs "
                              "e185's stored hashes)")
    log(f"G_INPUTS: per-step batches bit-identical across all three arms "
        f"AND vs e185's stored hashes (steps 1..{n_shared}): PASS")

    # =====================================================================
    # per-arm outcome + THE DENSITY LADDER
    # =====================================================================
    outcomes = {}
    for t in ARM_ORDER:
        j = arms_out[t]["journal"]
        stop = arms_out[t]["stop"]
        first_ge_edge = next((r for r in j if r["cum_disp"] >= SIGN_EDGE),
                             None)
        alive_edge = (bool(first_ge_edge["gm12"] > SHUT_BAR)
                      if first_ge_edge is not None else None)
        final = j[-1]
        if stop["kind"] == "kill":
            outcome, d_kill = "kill", stop["D_kill"]
        elif alive_edge and final["gm12"] > SHUT_BAR:
            outcome, d_kill = "spare_past_edge", None
        else:
            outcome, d_kill = "cap_below_edge", None   # stalled; no spare
        outcomes[t] = {
            "outcome": outcome, "stop_kind": stop["kind"],
            "stop_step": stop["step"], "D_kill": d_kill,
            "D_kill_raw": stop.get("D_kill_raw"),
            "gm12_at_kill": stop.get("gm12_at_kill"),
            "alive_at_2p5_crossing": alive_edge,
            "reached_edge": first_ge_edge is not None,
            "edge_cross_D": (first_ge_edge["cum_disp"]
                             if first_ge_edge else None),
            "edge_cross_gm12": (first_ge_edge["gm12"]
                                if first_ge_edge else None),
            "final_cum_disp": final["cum_disp"], "final_gm12": final["gm12"],
            "steps_ran": len(j), "max_l2_dev": arms_out[t]["max_l2_dev"],
            "train_seconds": arms_out[t]["train_seconds"],
            "topk_overlap_mean": (float(np.mean(arms_out[t]["jaccards"]))
                                  if arms_out[t]["jaccards"] else None),
        }
        log(f"outcome[{t}]: {outcome} D_kill="
            + (f"{d_kill:.4f}" if d_kill is not None else "n/a")
            + f" alive@2.5x={alive_edge} final D {final['cum_disp']:.4f} "
            f"g-12 {final['gm12']:.4f} |dL2|max "
            f"{arms_out[t]['max_l2_dev']:.1e}"
            + (f" jacc {outcomes[t]['topk_overlap_mean']:.3f}"
               if outcomes[t]["topk_overlap_mean"] is not None else ""))

    # THE DENSITY LADDER (kills by D_kill, spares by final D; raw + A0 rungs
    # committed parents)
    rung_names = {"a_topk10": "topk-10 (top-10% |g|)",
                  "a_topk50": "topk-50 (top-50% |g|)",
                  "a_sign": "sign (full sign(g))"}
    ladder = [
        {"rung": "raw (raw-g direction, opt1c committed)", "kind": "kill",
         "D": RAW_DKILL_COMMITTED, "source": "runs/opt1c/metrics.json"},
        {"rung": "A0-Adam (opt1 committed)", "kind": "kill",
         "D": A0_DKILL_COMMITTED, "source": "runs/opt1/metrics.json"},
    ]
    for t in ARM_ORDER:
        o = outcomes[t]
        ladder.append({"rung": rung_names[t], "kind": o["outcome"],
                       "D": (o["D_kill"] if o["D_kill"] is not None
                             else o["final_cum_disp"]),
                       "source": "this cell"})
    ladder_ordered = (sorted([r for r in ladder if r["kind"] == "kill"],
                             key=lambda r: r["D"])
                      + sorted([r for r in ladder if r["kind"] != "kill"],
                               key=lambda r: r["D"]))
    log("THE DENSITY LADDER: "
        + " < ".join(f"{r['rung'].split(' (')[0]}:{r['D']:.3f}"
                     for r in ladder_ordered))

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # DENSITY-CARRIES -> DENSITY-SPARES -> GRADED; no shopping)
    # =====================================================================
    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        sparse = ["a_topk10", "a_topk50"]
        sparse_kill = all(outcomes[t]["outcome"] == "kill" for t in sparse)
        sparse_spare = all(outcomes[t]["outcome"] == "spare_past_edge"
                           for t in sparse)
        sign_spare = outcomes["a_sign"]["outcome"] == "spare_past_edge"
        sign_kill = outcomes["a_sign"]["outcome"] == "kill"
        dc = bool(sparse_kill and sign_spare)
        ds = bool(sparse_spare and sign_kill)
        bars = {
            "DENSITY_CARRIES": {"fires": dc, "sparse_kill": sparse_kill,
                                "sign_spares_past_edge": sign_spare,
                                "detail": {t: outcomes[t]["outcome"]
                                           for t in ARM_ORDER}},
            "DENSITY_SPARES": {"fires": ds, "sparse_spare": sparse_spare,
                               "sign_kills": sign_kill,
                               "detail": {t: outcomes[t]["outcome"]
                                          for t in ARM_ORDER}},
            "GRADED": {"fires": not dc and not ds},
        }
        if dc:
            verdict = "DENSITY-CARRIES"
            clause = ("the sparse |g|-selected fronts KILLED ("
                      + "; ".join(f"{t} D_kill {outcomes[t]['D_kill']:.4f}"
                                  for t in sparse)
                      + ") while the full sign arm at matched per-step L2 "
                      "spared past its 2.5 edge (alive at the "
                      f"{SIGN_EDGE}-crossing and final reads, final D "
                      f"{outcomes['a_sign']['final_cum_disp']:.4f}) — the "
                      "lethal carrier is the SPARSE STRUCTURED FRONT, not "
                      "dense flattening.")
        elif ds:
            verdict = "DENSITY-SPARES"
            clause = ("the sparse arms spared past the sign edge alive ("
                      + "; ".join(f"{t} final g-12 "
                                  f"{outcomes[t]['final_gm12']:.4f}"
                                  for t in sparse)
                      + ") while the dense sign arm killed (D_kill "
                      f"{outcomes['a_sign']['D_kill']:.4f}) — density "
                      "itself carries lethality; structure rides on which "
                      "coordinates.")
        else:
            verdict = "GRADED"
            clause = ("any partial pattern — the arm table verbatim; the "
                      "kill-D ordering reported as the density ladder: "
                      + " < ".join(f"{r['rung'].split(' (')[0]} "
                                   f"{r['D']:.3f}({r['kind']})"
                                   for r in ladder_ordered))
    log("=" * 78)
    log(f"OPT2 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # the stretch-invariance co-read (the sign arm's own question)
    so = outcomes["a_sign"]
    if so["outcome"] == "kill":
        stretch = {"sign_kill_D": so["D_kill"],
                   "static_edge": SIGN_EDGE, "a0_kill": A0_DKILL_COMMITTED,
                   "reading": ("kill-D within ~15% of the 2.5 edge — the "
                               "sign kill does NOT move with a 2x horizon"
                               if abs(so["D_kill"] - SIGN_EDGE) / SIGN_EDGE
                               <= 0.15 else
                               "kill-D OUTSIDE 15% of the 2.5 edge — the "
                               "sign kill MOVES with horizon (reported)")}
    elif so["outcome"] == "spare_past_edge":
        stretch = {"sign_final_D": so["final_cum_disp"],
                   "static_edge": SIGN_EDGE,
                   "reading": ("the full sign arm SPARED past 2x the static "
                               "edge at matched L2 — the sign kill is "
                               "horizon- or path-dependent, not pure static "
                               "terrain (reported)")}
    else:
        stretch = {"sign_final_D": so["final_cum_disp"],
                   "static_edge": SIGN_EDGE,
                   "reading": "cap-stalled below the edge; ambiguity stated"}

    # =====================================================================
    # outputs
    # =====================================================================
    ckpt_inventory = {}
    pref = "smoke_" if SMOKE else ""
    for t in ARM_ORDER:
        s = arms_out[t]["stop"]["step"]
        sd = arms_out[t]["net_sd"]
        name = f"{pref}opt2_{t}_s{s}"
        path = CKPT_DIR / f"{name}.pt"
        torch.save({"model": sd,
                    "meta": {"experiment": "opt2", "arm": t,
                             "steps": int(s), "desc": ARM_DESC[t],
                             "input_seed": FREEZE_SEED,
                             "step_L2": STEP_L2_COMMITTED,
                             "base": f"runs/checkpoints/{ROOT_CK}"}},
                   path)
        ckpt_inventory[name] = {"path": f"runs/checkpoints/{name}.pt",
                                "arm": t, "steps": int(s)}
        log(f"[ckpt] saved {name}.pt")

    def full_arm(t):
        st = arms_out[t]
        return {
            "desc": ARM_DESC[t],
            "update_rule": {
                "kind": ("topk" if t != "a_sign" else "sign"),
                "frac": (TOPK_FRACS[t] if t != "a_sign" else None),
                "k": (int(round(TOPK_FRACS[t] * N_PARAM))
                      if t != "a_sign" else N_PARAM),
                "step_L2": STEP_L2_COMMITTED,
                "assertion": "per-step |delta|_2 == step_L2 (in-code, every "
                             "step); max dev reported",
            },
            "max_l2_dev": st["max_l2_dev"],
            "steps_ran": len(st["journal"]),
            "train_seconds": st["train_seconds"],
            "seed": FREEZE_SEED,
            "stop": st["stop"],
            "traj": st["journal"],
            "outcome": outcomes[t],
            "topk_overlap": {"per_step_jaccard": st["jaccards"],
                             "mean": outcomes[t]["topk_overlap_mean"]},
            "zeph_violations": st["zeph"],
            "chunks": st["chunks_prov"],
            "md5_chain": st["md5_chain"],
        }

    metrics = {
        "experiment": "opt2_density",
        "date": common.now_iso(),
        "status": "COMPLETE — adjudicated (this write replaces all PARTIAL "
                  "progressive writes)",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the lethal structure of the sign-normalized wash "
                     "step live in the NUMBER OF COORDINATES TOUCHED — a "
                     "sparse |g|-selected structured front, or dense "
                     "every-coordinate flattening? The licensed e185 wash "
                     "cell VERBATIM (true targets, seed-10902 stream, the "
                     "consolidated host fact) at matched per-step L2 "
                     "1.6543: TOPK-10%, TOPK-50%, SIGN (2x horizon) + the "
                     "committed references (opt1 A0, opt1c raw 0.9203, e192 "
                     "terrain rays, opt1b bleed)."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) VERBATIM — the "
                             "consolidated host fact under its neutral/"
                             "corpus wash (opt1's conventions)",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random "
                     "corpus windows, full-token CE, clip 1.0",
            "targets": "TRUE (the real neutral wash — the update rule is "
                       "the only delta)",
            "input_seed": FREEZE_SEED,
            "matched_step_L2": {"value": STEP_L2_COMMITTED,
                                "provenance": "opt1 A0's committed step-1 "
                                "displacement (lr*sqrt(N) arithmetic; "
                                "recomputed at t=0 and bit-gated in G_T0)"},
            "ckpt_cadence": "g-12 EVERY step (opt1c's bracket-tightest); "
                            "g-0 + CE_R + dual-estimator alignment at "
                            f"{list(CKPT_STEPS)}",
            "measure_light": "g-12 / g0 batteries + CE_R + dual-estimator "
                             "alignment + displacement + pre-clip grad "
                             "norms + top-k churn (the dispatch's read "
                             "list; no census/deletions)",
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_T0": G_T0, "G_SAMEPOINT": G_SAMEPOINT,
                  "G_TERRAIN": G_TERRAIN, "G_STREAM": G_STREAM,
                  "G_INPUTS": G_INPUTS, "density_t0": density_t0},
        "arms": {t: full_arm(t) for t in ARM_ORDER},
        "density_ladder": {"ordered": ladder_ordered, "raw_rows": ladder,
                           "ordering_note": "kills by D_kill then spares by "
                                            "final D; raw/A0 rungs are "
                                            "committed parents, never rerun"},
        "references": {
            "opt1_A0": {"metrics": "runs/opt1/metrics.json",
                        "t_x": a0["tstar"]["t_x"],
                        "t_ck": a0["tstar"]["t_ck"],
                        "D_at_kill": A0_DKILL_COMMITTED,
                        "ckpt_table_loaded": len(a0_path),
                        "role": "the AdamW reference clock + the matched-L2 "
                                "anchor (loaded committed, never rerun)"},
            "opt1c_raw": {"metrics": "runs/opt1c/metrics.json",
                          "D_kill_interp": RAW_DKILL_COMMITTED,
                          "t_x": raw_stop["t_x"],
                          "role": "the raw-gradient rung of the ladder "
                                  "(loaded committed, never rerun)"},
            "e192_terrain": {"metrics": "runs/e192/metrics.json",
                             "sign_kill_D": SIGN_EDGE,
                             "profiles": {k: len(v["rows"])
                                          for k, v in e192_profiles.items()},
                             "role": "the static terrain rays (overlay)"},
            "opt1b_bleed": {"via": "e192 overlays (loaded committed)",
                            "rows": len(bleed),
                            "role": "the diffusive reference curve"},
            "e_chart_samepoint": {"metrics": "runs/e_chart/metrics.json",
                                  "adam_samepoint": W024_ADAM_SAMEPOINT,
                                  "raw_samepoint": W024_RAW_SAMEPOINT,
                                  "role": "T150's estimator-lesson anchors "
                                          "(the dual-estimator gate)"},
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "density_ladder_ordered": ladder_ordered,
            "stretch_invariance_coread": stretch,
            "composite_order": "DENSITY-CARRIES -> DENSITY-SPARES -> "
                               "GRADED (frozen before compute)",
            "constants": {"STEP_L2": STEP_L2_COMMITTED,
                          "SIGN_EDGE": SIGN_EDGE, "D_TARGET": D_TARGET,
                          "SHUT_BAR": SHUT_BAR, "STEP_CAP": STEP_CAP},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 per arm (single seed per arm; the "
                            "replicate ladder runs only if a bar fires); "
                            "ONE root, ONE input stream (seed 10902, "
                            "bit-gated across all three arms AND vs e185's "
                            "stored md5s); single family (the e131 "
                            "consolidated line); every arm differs ONLY in "
                            "the update rule's mask/arithmetic"),
            "float_texture": ("CPU fp32 texture: all trainings this "
                              "process, this CPU, 4 threads (the e185-era "
                              "reduction order); committed references are "
                              "hard-bound and the t=0 gate re-derives A0's "
                              "step on THIS device"),
            "no_guaranteed_outcome": ("every arm TRAINS on the licensed "
                                      "stream at lethal-scale per-step L2 "
                                      "— the arms could kill anywhere on "
                                      "the ladder; that openness is the "
                                      "point (nothing in this battery is "
                                      "guaranteed-to-spare)"),
            "estimator_lesson_honored": ("alignment co-reads state their "
                                         "evaluation point: post_step ("
                                         "opt1's convention) AND "
                                         "matched_point (the chart's "
                                         "same-point convention; at s=1 "
                                         "gated vs the committed +0.03955) "
                                         "+ the root-anchored cumulative "
                                         "read — T150's chart lesson"),
            "topk_churn": ("the top-k set churns step to step (the mask "
                           "follows g) — that IS the drifting front; "
                           "per-step Jaccard reported, mean co-read"),
            "densification": ("kill brackets densified along the killing "
                              "step at f in {0.2,0.4,0.6,0.8} (points ON "
                              "the trajectory, not new steps); the "
                              "densified value adjudicates, both reported"),
            "projections_never_adjudicate": ("arms stalled below D 2.5 "
                                             "cannot fire a spare clause; "
                                             "no projection is used "
                                             "anywhere"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": ckpt_inventory,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__,
                   "inter_arm_s": INTER_ARM_S, "chunk_cap": CHUNK_CAP},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_ladder(rd / "opt2_density_ladder.png", metrics, e192_profiles,
                a0_path, bleed, verdict, clause, ladder_ordered, outcomes)
    plot_trajectory(rd / "opt2_trajectory.png", metrics, ARM_ORDER)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt2_density_ladder.png'}, "
        f"{rd / 'opt2_trajectory.png'}, {len(ckpt_inventory)} checkpoints "
        f"in runs/checkpoints/opt2_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

ARM_COLORS = {"a_topk10": "crimson", "a_topk50": "darkorange",
              "a_sign": "seagreen"}
ARM_LABELS = {"a_topk10": "TOPK-10% (sparse front, g magnitudes)",
              "a_topk50": "TOPK-50% (mid density)",
              "a_sign": "SIGN full (dense, 2x horizon)"}


def plot_ladder(path, metrics, e192_profiles, a0_path, bleed, verdict,
                clause, ladder_ordered, outcomes):
    """THE CENTERPIECE: the density ladder overlaid on the committed
    terrain rays (e192's static profiles) + A0's committed Adam path."""
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.8),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.plot([r["D"] for r in e192_profiles["R1_G"]["rows"]],
            [r["gm12"] for r in e192_profiles["R1_G"]["rows"]],
            "--", lw=1.8, color="black", alpha=0.75,
            label="static g-ray (e192; kills 0.92)")
    ax.plot([r["D"] for r in e192_profiles["R2_SIGN"]["rows"]],
            [r["gm12"] for r in e192_profiles["R2_SIGN"]["rows"]],
            "--", lw=1.8, color="dimgray", alpha=0.85,
            label="static sign-ray (e192; kills 2.5)")
    for k in ("R3_GAUSS_s11501", "R4_GAUSS_s11502", "R5_GAUSS_s11503"):
        ax.plot([r["D"] for r in e192_profiles[k]["rows"]],
                [r["gm12"] for r in e192_profiles[k]["rows"]],
                ":", lw=1.1, color="lightgray", alpha=0.9,
                label=("_Gaussian rays (e192; kill > 4)"
                       if k == "R3_GAUSS_s11501" else "_nolegend_"))
    if bleed:
        ax.plot([r["D"] for r in bleed], [r["gm12"] for r in bleed],
                "-.", lw=1.3, color="steelblue", alpha=0.8,
                label="the bleed (opt1b committed; diffusive)")
    ax.plot([r["cum_disp"] for r in a0_path],
            [r["gm12"] for r in a0_path], "x-", ms=6, lw=1.5, mew=2,
            color="gray", alpha=0.9,
            label="A0 AdamW (opt1 committed; kills 2.489)")
    root_gm12 = metrics["gates"]["G_ROOT"]["cells"]["gm12"]
    for t in ARM_ORDER:
        rows = metrics["arms"][t]["traj"]
        ax.plot([0.0] + [r["cum_disp"] for r in rows],
                [root_gm12] + [r["gm12"] for r in rows],
                "o-", ms=4.5, lw=2.0, color=ARM_COLORS[t], alpha=0.95,
                label=ARM_LABELS[t])
        o = outcomes[t]
        if o["outcome"] == "kill" and o["D_kill"] is not None:
            ax.axvline(o["D_kill"], color=ARM_COLORS[t], lw=1.2, ls=":",
                       alpha=0.8)
            ax.plot([o["D_kill"]], [SHUT_BAR], "*", ms=15,
                    color=ARM_COLORS[t], mec="k", mew=0.6)
    ax.axhline(SHUT_BAR, ls="--", lw=1.1, color="tab:purple", alpha=0.8,
               label=f"{SHUT_BAR} SHUT")
    ax.axvline(SIGN_EDGE, ls="-", lw=1.0, color="dimgray", alpha=0.5)
    ax.text(SIGN_EDGE + 0.05, 0.97, "the 2.5 sign edge", fontsize=7.5,
            color="dimgray", rotation=90, va="top")
    ax.axvline(D_TARGET, ls="-", lw=1.0, color="dimgray", alpha=0.5)
    ax.text(D_TARGET + 0.05, 0.97, "2x horizon", fontsize=7.5,
            color="dimgray", rotation=90, va="top")
    ax.set_xlabel(r"cumulative displacement $D=\|\theta_t-\theta_0\|_2$")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.1, 6.6)
    ax.legend(fontsize=7.3, loc="center right")
    ax.set_title("THE DENSITY LADDER on the committed terrain — fact vs D "
                 "at matched per-step L2 1.6543 (static rays from e192, "
                 "never rerun)", fontsize=9.5)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "THE DENSITY LADDER (kills by D_kill, spares by "
            "final D)", fontsize=9, va="top", family="monospace",
            weight="bold")
    y -= 0.045
    ax.text(0.02, y, "  rung                                   D      "
            "outcome", fontsize=7.0, va="top", family="monospace")
    y -= 0.032
    for r in ladder_ordered:
        ax.text(0.02, y,
                f"  {r['rung']:<38} {r['D']:6.3f}  {r['kind']}",
                fontsize=7.0, va="top", family="monospace")
        y -= 0.031
    y -= 0.015
    for t in ARM_ORDER:
        o = outcomes[t]
        al = (f"{o['alive_at_2p5_crossing']}"
              if o["alive_at_2p5_crossing"] is not None else "n/a")
        ax.text(0.02, y,
                f"  {t}: steps {o['steps_ran']:2d}  alive@2.5 {al:>5}  "
                f"final D {o['final_cum_disp']:.3f}  final g-12 "
                f"{o['final_gm12']:.4f}  |dL2|max "
                f"{metrics['arms'][t]['max_l2_dev']:.1e}",
                fontsize=6.6, va="top", family="monospace",
                color=ARM_COLORS[t])
        y -= 0.029
    y -= 0.015
    ax.text(0.02, y, f"OPT2 VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.038
    for wd in textwrap.wrap(clause, width=96, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.5, va="top",
                family="monospace")
        y -= 0.023
    fig.suptitle("OPT2 — THE DENSITY QUESTION: sparse structured fronts vs "
                 "dense flattening at matched per-step L2 (the licensed "
                 "e185 cell, update rule swapped)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_trajectory(path, metrics, arm_order):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    # (0,0) D(t) vs step
    ax = axes[0, 0]
    for t in arm_order:
        tr = metrics["arms"][t]["traj"]
        ax.plot([r["step"] for r in tr], [r["cum_disp"] for r in tr],
                "o-", ms=4, lw=1.8, color=ARM_COLORS[t],
                label=ARM_LABELS[t])
    ax.axhline(SIGN_EDGE, ls="--", lw=1.2, color="dimgray",
               label="the 2.5 sign edge")
    ax.axhline(D_TARGET, ls=":", lw=1.4, color="k",
               label=f"D_TARGET {D_TARGET} (2x horizon)")
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"cumulative $D=\|\theta_t-\theta_0\|_2$")
    ax.legend(fontsize=7.2, loc="lower right")
    ax.set_title("D(t) per arm (per-step L2 pinned at 1.6543 — slope "
                 "differences are direction persistence)", fontsize=10)

    # (0,1) THE DUAL-ESTIMATOR ALIGNMENT (the chart's lesson made visible)
    ax = axes[0, 1]
    for t in arm_order:
        rows = [r for r in metrics["arms"][t]["traj"] if "align" in r]
        ax.plot([r["step"] for r in rows],
                [r["align"]["post_step"]["m12"] for r in rows], "o-",
                ms=5, lw=1.7, color=ARM_COLORS[t],
                label=f"{t} POST-STEP (opt1 conv.)")
        ax.plot([r["step"] for r in rows],
                [r["align"]["matched_point"]["m12"] for r in rows], "s--",
                ms=5, lw=1.4, color=ARM_COLORS[t], alpha=0.6,
                label=f"{t} MATCHED-POINT (chart conv.)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("wash step (checkpoint)")
    ax.set_ylabel(r"cos($\Delta\theta$, $\nabla$ fact m12)")
    ax.legend(fontsize=5.8, loc="lower right", ncol=1)
    ax.set_title("THE ESTIMATOR LESSON (T150), both points reported — "
                 "post-step vs matched-point alignment (critic's sign: "
                 "negative = death-aligned)", fontsize=9)

    # (1,0) per-arm battery trace g-12 vs step
    ax = axes[1, 0]
    for t in arm_order:
        tr = metrics["arms"][t]["traj"]
        ax.plot([0] + [r["step"] for r in tr],
                [metrics["gates"]["G_ROOT"]["cells"]["gm12"]]
                + [r["gm12"] for r in tr], "o-", ms=4, lw=1.8,
                color=ARM_COLORS[t], label=ARM_LABELS[t])
    for yv, col, lbl in ((SURVIVE_NOTE, "seagreen", "0.50 SPARE ref"),
                         (SHUT_BAR, "tab:purple", "0.27 SHUT")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.set_xlabel("wash step")
    ax.set_ylabel("g-12 (every-step cadence)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title("the kill clocks — g-12 vs step", fontsize=10)

    # (1,1) terrain riding + front churn
    ax = axes[1, 1]
    for t in arm_order:
        rows = [r for r in metrics["arms"][t]["traj"] if "align" in r]
        if rows:
            ax.plot([r["step"] for r in rows],
                    [r["align"]["terrain"]["u_g"] for r in rows], "o-",
                    ms=5, lw=1.7, color=ARM_COLORS[t],
                    label=f"{t} cos vs g-ray")
            ax.plot([r["step"] for r in rows],
                    [r["align"]["terrain"]["sign_g0"] for r in rows], "s--",
                    ms=5, lw=1.4, color=ARM_COLORS[t], alpha=0.6,
                    label=f"{t} cos vs sign(g0)")
    jac = metrics["arms"]["a_topk10"]["topk_overlap"]["per_step_jaccard"]
    if jac:
        ax2 = ax.twinx()
        ax2.plot(range(2, 2 + len(jac)), jac, "-", lw=1.2, color="gray",
                 alpha=0.7, label="topk set Jaccard (right axis)")
        ax2.set_ylabel("inter-step top-k set Jaccard", fontsize=8,
                       color="gray")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=6.5, loc="upper right")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("wash step (checkpoint)")
    ax.set_ylabel("cos of cumulative delta vs committed directions")
    ax.legend(fontsize=6.0, loc="lower left")
    ax.set_title("which ray does each arm ride (e191's committed u_g; "
                 "sign(g_0)) + front churn", fontsize=9.5)

    fig.suptitle("OPT2 — trajectory reads: displacement, the dual-estimator "
                 "alignment, kill clocks, front churn", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _arm_summary(state):
    """A compact jsonable summary for progressive partial writes."""
    return {"arm": state["arm"], "step": state["step"],
            "stop": state["stop"],
            "journal_tail": state["journal"][-3:],
            "max_l2_dev": state["max_l2_dev"],
            "jaccards": state["jaccards"], "zeph": state["zeph"],
            "chunks": len(state["chunks_prov"])}


if __name__ == "__main__":
    sys.exit(main())
