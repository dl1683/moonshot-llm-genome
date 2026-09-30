"""OPT1 — THE OPTIMIZER CONTROLS (supervisor check-ins C10-4 / C11-x /
C12-5b, asked nine times; the CPU-lane cell queued after g3K).

WHY: every wash/kill result in the arc (e158/e161/e176/e176N/e184/e185/
e187) ran under AdamW (0.9, 0.95, wd 0.1, constant lr 1e-3, fresh state,
no warmup). Check-in 6's arithmetic: a FRESH AdamW's first bias-corrected
step is +-lr on EVERY coordinate regardless of gradient magnitude (e185
measured step-1 |d| 1.6543 ~ 1e-3 * sqrt(2,739,072) = 1.655), so the
"displacement" the no-basin law trades in is, at matched lr, an Adam
artifact; check-in 8's registered prediction: under matched SGD, t*
should scale with the stream's gradient norm (orders slower). THE OPEN
QUESTION: is the fast fact-kill OPTIMIZER-AGNOSTIC memory physics, or
does Adam's moment/normalization structure carry it?

THE CELL: the licensed wash cell VERBATIM — e185's CONTROL arm (the
e185/e157-family consolidated host fact = runs/checkpoints/
e131_consolidated_e113.pt, under its neutral/corpus wash: batch 32 =
16 e170 neutral-bank anchors (seed 170, FIXED) + 16 random corpus
windows, TRUE targets, seed-10902 draw sequence, full-token CE, clip
1.0) — with the OPTIMIZER swapped per arm. Same ruler, same seed
conventions, same light reads as e185 (g-12 / g0 batteries + CE_R at
checkpoints; per-step in-batch CE; per-step cumulative displacement
||theta_t - theta_0||_2 over all 2,739,072 params + increments).

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - OPT-AGNOSTIC: "fires if SGD kills the fact within 2x of the AdamW
    clock at comparable final displacement AND no single Adam knob
    (A3/A4/A5) shifts t* by >=2x — the no-basin trajectory law is
    optimizer-agnostic memory physics."
  - ADAM-AMPLIFIES: "fires if any single Adam knob shifts t* by >=2x
    (either direction) OR matched-displacement SGD survives >=5x longer
    — the kill is substantially Adam-carried; every no-basin statement
    gains an optimizer clause."
  - MIXED-TEXTURE: "any graded outcome — report the full arm table
    verbatim; no binary claim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * the fact = g-12 = ABSOLUTE install-60 battery mean p(Z) at ctx
    offset -12 (e176/e176N/e184/e185's convention verbatim); g0 = the
    same battery at the trained geometry offset 0 (the dispatch's "fact
    readout strength"); KILL = g-12 <= 0.27 (the arc's SHUT bar,
    e158/e161/e176/e176N/e184/e185); SPARE = g-12 >= 0.50.
  * t* (the kill clock) per arm reported BOTH as t_ck = the FIRST
    checkpoint with g-12 <= 0.27 (e185's stored convention: t*=+2) AND
    t_x = the linearly INTERPOLATED step of the 0.27 crossing between
    checkpoints (the dispatch's "t* with interpolation"; the root read
    g-12=0.9156 is the step-0 anchor). Bar ratios use t_x.
  * DISPLACEMENT = cumulative ||theta_t - theta_0||_2 over all
    2,739,072 trainable parameters (fp32, CPU, measured every step) +
    per-step increments, exactly e185's currency; the PRE-CLIP gradient
    norm is additionally recorded at every step (check-in 8's scaling
    variable: under SGD the per-step displacement is lr * min(||g||,1)).
  * D_kill = 2.4892616271972656 (e185's stored control displacement at
    its kill step +2) — and A0 MUST reproduce it: A0 is gated against
    e185's stored control (per-step input md5s, per-step displacements,
    g-12/g0/CE_R lights; tol 5e-6 bit flag / 0.05 fallback). If A0 does
    not kill by +10 the cell ABORTS to MIXED-TEXTURE (control failure;
    e185's convention).
  * "comparable final displacement" (OPT-AGNOSTIC's SGD clause) =
    D at the SGD arm's t* within [0.5, 2.0] x D_kill.
  * "matched-displacement SGD survives >=5x longer" (ADAM-AMPLIFIES's
    SGD clause) = an SGD arm whose MEASURED cumulative displacement
    reaches >= D_kill within its run window AND whose g-12 stays
    >= 0.50 at every checkpoint with D >= D_kill, or whose t_x >= 5x
    A0's t_x. SGD arms that do NOT reach D_kill within the CPU cap
    CANNOT fire this clause (cap-limited; reported as ambiguity with a
    clearly-labeled EXTRAPOLATED clock from the measured steady
    displacement rate — projections never adjudicate).
  * "shifts t* by >=2x (either direction)" = t_x(knob)/t_x(A0) >= 2 or
    <= 0.5. A knob arm that does not kill within its window while its
    displacement stays below D_kill is reported as a delay lower-bound
    (window/t_x(A0)), labeled EXTRAPOLATED, and does not by itself
    fire the clause.
  * ALIGNMENT CO-READ: cos(delta_theta_t, grad g0_t) at every
    checkpoint — ONE CPU backward on the fact battery (g0, offset 0;
    co-reported for g-12) of the mean log p(Z) readout, taken AT
    theta_t on an eval twin; the critic's sign convention: NEGATIVE =
    aligned with the death gradient (W022's wash alignment read ~-0.44;
    ties this cell to the trajectory-vs-static law of T137 and the
    e188 alignment-integral question). delta_theta_t = the CUMULATIVE
    checkpoint displacement vector. Bonus co-read: cos(delta_arm,
    delta_A0) at shared checkpoints (e185's corpus-direction cosine).
  * CE at each step (the arm's own in-batch training CE) + CE_R at
    every checkpoint (organism health, the e065 val-window bank seed
    26502).

THE SEVEN ARMS (one file, one run; the ONLY inter-arm deltas are the
optimizer construction and/or its lr schedule):
  A0 a0_adamw_ref       AdamW (0.9, 0.95) wd 0.1 constant lr 1e-3 — the
                        recipe VERBATIM (the known kill clock t*=+2);
                        doubles as the e185 reproduction gate.
  A1 a1_sgd_1e-3        plain SGD (momentum 0, weight_decay 0, clip 1.0
                        retained — part of the licensed cell) at the
                        same lr 1e-3.
  A2 a2a_sgd_3e-3       SGD lr ladder {3e-3,
     a2b_sgd_1e-2       1e-2} — co-reported, locates the SGD clock.
  A3 a3_adamw_warmup30  AdamW recipe + 30-step LINEAR warmup into the
                        wash (lr_eff = 1e-3 * min(1, step/30)).
  A4 a4_adamw_b2_0.999  AdamW recipe with beta2=0.999 (vs 0.95).
  A5 a5_adamw_moment_reset  AdamW with FRESH optimizer state at wash
                        start, same weights. NULL ARM BY CONSTRUCTION:
                        the e185 cell never loads optimizer state (the
                        wash always starts cold), so A5 is expected
                        BIT-IDENTICAL to A0; it is RUN as the gate that
                        PROVES no consolidation-time optimizer state is
                        inherited (the ninth-time question closed by
                        measurement, not assumption) and doubles as the
                        determinism replicate of the reference clock.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any training): the
battery geometry must match the e185 convention — corpus ZEPH count 0;
splice install mix {FLORIZEL: 19, ELIZABETH: 41}; battery shapes
60 x (130 +- j) at offsets {-12, 0, +12}; the neutral bank's 16 starts
bit-equal to e185's stored list; the root gated vs e151's before-cells.
WHAT EACH ARM GUARANTEES: NOTHING. Every arm TRAINS on the licensed
stream, so every arm could kill — there is no guaranteed-to-spare arm
in this battery; that is why it is honest. (A5's expected null is a
state-inheritance statement, not a sparing guarantee.)

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch, e176n/e185/e187 precedent; the GPU is another agent's —
never claimed), torch threads 4 (e185 bit-reproduction requires the
e185-era reduction order), SEQUENTIAL arms with a 20 s stagger between
trainings (no parallel arms), per-arm cap 180 s measured on the
training loop (e185's FT_TIME_CAP convention: in-loop checkpoint reads
count toward the cap), a single 25 s launch stagger at start. AdamW
arms run to +10 (A3 to +24 — its kill is expected INSIDE the warmup);
SGD arms to +160 steps or the cap (whichever binds first). n=1 per arm
(single training seed per arm, first cell); the replicate ladder runs
only if a bar fires.

CHECKPOINT CADENCE: e185's {1, 2, 4, 10} VERBATIM (A0 gates against
e185's stored values at exactly these steps) + the geometric extension
{8, 20, 40, 80, 160} to locate slow kill clocks (recorded deviation).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/ce_fixed_cpu/
val_windows/flat_params are lab/e185_noise_wash.py VERBATIM (via
e187's verified copies — the e176n lineage); the trainer is e185's
noise_wash VERBATIM ARITHMETIC at target_mode="true" (identical
seed-10902 aj/rj draw order, identical batch construction, identical
clip) with the optimizer construction + lr schedule parameterized and
three additions: pre-clip grad-norm capture, the alignment backward,
the extended checkpoint ladder. Copied, not imported, to own the
device policy.

NETS: runs/checkpoints/e131_consolidated_e113.pt (the fully
consolidated root, 2,739,072 params — e143/e151/e152/e158/e176n/e184/
e185/e187 precedent). References: runs/e185/metrics.json (embedded
copies verified against the stored file at run time, e185's verify
convention) + runs/e187/metrics.json.

Outputs: runs/opt1/{metrics.json, opt1_kill_clocks.png,
opt1_trajectory.png}; checkpoints runs/checkpoints/opt1_*.pt (A0 at
the e185 cadence + each arm's final state). No NOTES/THINKING/QUEUE/
STATE edits (the coordinator folds).

Run:  cd lab && python opt1_optimizer_controls.py    (OPT1_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185/e187)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("OPT1_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "opt1 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E185_METRICS = E43.REPO / "runs" / "e185" / "metrics.json"
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the wash envelope (e185 VERBATIM; only the optimizer differs) -------------
LR = 1e-3                          # the wash's own lr (A0/A3/A4/A5)
SGD_LADDER = (3e-3, 1e-2)          # A2 co-reported ladder
WARMUP_STEPS = 30                  # A3 linear warmup length
FT_TIME_CAP = 180.0                # dispatch: <=180 s per training (measured)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185)
STAGGER_S = 25.0                  # launch stagger (e187 convention)
INTER_ARM_S = 20.0                # politeness stagger between sequential arms
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- checkpoint cadence: e185's {1,2,4,10} VERBATIM + geometric extension ------
CK_LADDER: tuple[int, ...] = (1, 2, 4, 8, 10, 20, 40, 80, 160) \
    if not SMOKE else (1, 2)
MAX_STEPS = {  # per arm (smoke trims below)
    "a0_adamw_ref": 10, "a1_sgd_1e-3": 160, "a2a_sgd_3e-3": 160,
    "a2b_sgd_1e-2": 160, "a3_adamw_warmup30": 24, "a4_adamw_b2_0.999": 10,
    "a5_adamw_moment_reset": 10,
}
if SMOKE:
    MAX_STEPS = {k: min(v, 4) for k, v in MAX_STEPS.items()}

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

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

# ---- E185's stored CONTROL (the cell A0 must reproduce; embedded verbatim from
# runs/e185/metrics.json; verified vs the file at run time — verify_e185).
E185_TRACE_STEPS = [0, 1, 2, 4, 10]
E185_CTRL_GM12 = [0.9155886173248291, 0.6780440807342529,
                  0.027077054604887962, 0.010940761305391788,
                  0.019353048875927925]
E185_CTRL_G0 = [0.7850371599197388, 0.4619811177253723,
                0.1147073358297348, 0.06042749062148519,
                0.09843862056732178]
E185_CTRL_CE_R = [1.663516640663147, 2.2113420963287354,
                  2.032074451446533, 1.8213403224945068,
                  1.7722809314727783]
E185_CTRL_CE_BATCH = [1.356567621231079, 1.8263245820999146,
                      1.669585943222046, 1.3631858825683594,
                      1.3955520391464233, 1.2410888671875,
                      1.1650500297546387, 1.0894068479537964,
                      1.1546937227249146, 0.9942378997802734]
E185_CTRL_DISP = {1: 1.6542880535125732, 2: 2.4892616271972656,
                  4: 3.526789665222168, 10: 5.195915222167969}
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
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
T_KILL_E185 = 2                    # e185's control kill step (g-12 0.0271)
CTRL_GM12_AT_TKILL = 0.027077054604887962
D_KILL_STORED = 2.4892616271972656  # e185's control displacement at +2

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ---------
SHUT_BAR = 0.27                   # DISSOLVE (e158/e161/e176/e176N/e184/e185)
SURVIVE_BAR = 0.50                # SPARE
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "opt_agnostic": "OPT-AGNOSTIC: \"fires if SGD kills the fact within "
        "2x of the AdamW clock at comparable final displacement AND no "
        "single Adam knob (A3/A4/A5) shifts t* by >=2x — the no-basin "
        "trajectory law is optimizer-agnostic memory physics.\"",
    "adam_amplifies": "ADAM-AMPLIFIES: \"fires if any single Adam knob "
        "shifts t* by >=2x (either direction) OR matched-displacement SGD "
        "survives >=5x longer — the kill is substantially Adam-carried; "
        "every no-basin statement gains an optimizer clause.\"",
    "mixed_texture": "MIXED-TEXTURE: \"any graded outcome — report the "
        "full arm table verbatim; no binary claim.\"",
    "operationalizations": "the fact = g-12 (install-60 battery mean p(Z) "
        "at ctx offset -12, e185's convention); KILL = g-12 <= 0.27 "
        "(SHUT), SPARE = g-12 >= 0.50; t* reported as t_ck (first "
        "checkpoint <= 0.27; e185's stored convention) AND t_x (linearly "
        "interpolated 0.27 crossing, root read as step-0 anchor); bar "
        "ratios use t_x; displacement = cumulative ||theta_t - theta_0||_2 "
        "over all 2,739,072 params (fp32, CPU, per step) + increments + "
        "PRE-CLIP grad norms; D_kill = 2.4892616271972656 (e185 stored, "
        "A0-gated); 'comparable final displacement' = D at the SGD arm's "
        "t* within [0.5, 2.0] x D_kill; 'matched-displacement SGD survives "
        ">=5x longer' = an SGD arm reaching D >= D_kill within its window "
        "with g-12 >= 0.50 at every matched checkpoint or t_x >= 5x A0's "
        "t_x (SGD arms that never reach D_kill within the CPU cap CANNOT "
        "fire the clause — cap-limited ambiguity + EXTRAPOLATED clock "
        "co-reported, projections never adjudicate); 'shifts t* by >=2x' "
        "= t_x(knob)/t_x(A0) >= 2 or <= 0.5; alignment = "
        "cos(delta_theta_t, grad of mean log p(Z) on the fact battery at "
        "theta_t), critic's sign convention (negative = death-aligned); "
        "CE at each step + CE_R at every checkpoint; composite order "
        "frozen OPT-AGNOSTIC -> ADAM-AMPLIFIES -> MIXED-TEXTURE; if A0 "
        "does not kill by +10 the cell ABORTS to MIXED-TEXTURE (control "
        "failure).",
    "registration": "the dispatch's registration IS the registration "
        "(the three bars quoted verbatim in the module docstring and "
        "here, frozen before compute). Adjudicate against exactly this; "
        "no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "Checkpoint cadence = e185's {1,2,4,10} VERBATIM (A0 gates against "
    "e185's stored values at exactly these steps) + the geometric "
    "extension {8,20,40,80,160} — required to locate kill clocks slower "
    "than the AdamW clock; the e185-era steps are untouched.",
    "Per-arm max steps: AdamW arms run to +10 (A3 to +24 — its kill is "
    "expected inside the 30-step warmup); SGD arms to +160 or the 180 s "
    "cap, whichever binds first. Matched-lr SGD's projected clock "
    "(~2.5k steps at the clip-capped 1e-3/step rate) is UNREACHABLE on "
    "CPU: the SGD arms report measured displacement rates + clearly-"
    "labeled EXTRAPOLATED clocks; no bar adjudicates on a projection.",
    "plain SGD = momentum 0, weight_decay 0, clip 1.0 retained (the clip "
    "is part of the licensed cell verbatim; its effect on SGD is to cap "
    "per-step displacement at lr — the pre-clip gradient norm is recorded "
    "at every step so the uncapped rate is also known).",
    "A5 is a null arm BY CONSTRUCTION (the e185 cell never loads "
    "optimizer state — every wash starts cold), so A5 is expected "
    "bit-identical to A0; it is RUN as the gate proving no consolidation-"
    "time optimizer state is inherited, and doubles as the determinism "
    "replicate of the reference clock (n=2 for A0's trajectory).",
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "the GPU is another agent's, never claimed): torch threads 4 (the "
    "e185-era reduction order, required for the A0 bit-reproduction "
    "gate), sequential arms, 25 s launch stagger + 20 s between arms "
    "(e187 used dispatch-mandated 60 s cooldowns; this dispatch mandates "
    "staggering only), per-arm 180 s cap measured on the training loop "
    "with in-loop checkpoint reads counted (e185's FT_TIME_CAP "
    "convention).",
    "Light reads only (g-12/g0 batteries + CE_R + alignment + "
    "displacement + pre-clip grad norms) — the dispatch's read list; the "
    "full e185 dial (held30/site-read/census/deletions) is not measured.",
    "n=1 per arm (single training seed per arm, first cell); the "
    "replicate ladder runs only if a bar fires. Single family (the "
    "e131 consolidated line); all trainings this process, this CPU, 4 "
    "threads; every e185 reference is re-derived/gated on this device "
    "before use — no cross-device comparison anywhere.",
    "Smoke mode trims: 4-step arms, checkpoints {1,2}, no staggers; "
    "nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e185_noise_wash.py VERBATIM (via e187's verified copies;
# the e176n lineage). Copied rather than imported to own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
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


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ: gradient (wrt all params, at the twin's current
    weights theta_t) of the fact battery's mean log p(Z) readout at the last
    position. The critic's sign convention: NEGATIVE cos(delta, grad) =
    displacement aligned with the DEATH gradient (W022). Consumes no RNG;
    run on the eval twin, never on the training net."""
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


# ------------------------------------------------------------------ the wash

def opt_wash(tag: str, net0: TinyGPT, anchor: torch.Tensor,
             train_ids: torch.Tensor, itos, r_eval_xy,
             gm12_ids: torch.Tensor, g0_ids: torch.Tensor, zid: int,
             ckpt_steps: tuple[int, ...], max_steps: int,
             make_opt, lr_at, time_cap: float = FT_TIME_CAP):
    """E185's noise_wash VERBATIM ARITHMETIC at target_mode="true" (the
    licensed neutral/corpus wash: identical seed-10902 aj/rj draw order,
    identical batch construction, full-token CE, clip 1.0) with the
    optimizer construction + lr schedule parameterized and three additions:
    pre-clip grad-norm capture (check-in 8's scaling variable), the
    alignment backward on the fact battery at checkpoints, and the extended
    checkpoint ladder. Displacement bookkeeping unchanged (cumulative +
    per-step, fp32, CPU, measured every step)."""
    ckpt_set = set(s for s in ckpt_steps if s <= max_steps)
    net = copy.deepcopy(net0)
    net.train()
    opt = make_opt(net)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    evl = copy.deepcopy(net0)          # CPU eval twin (reads + alignment)
    evl.eval()
    theta0 = flat_params(net)           # the displacement origin
    prev = theta0.clone()
    deltas: dict[int, torch.Tensor] = {}   # checkpoint delta vectors
    x_hashes: dict[int, str] = {}
    step, zeph_checks = 0, 0
    for step in range(1, max_steps + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        # name-free VERIFY (no-op by corpus construction; hard-fail if not)
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        lr_now = float(lr_at(step))
        for grp in opt.param_groups:
            grp["lr"] = lr_now
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()
        cur = flat_params(net)
        cum_disp = float(torch.norm(cur - theta0))
        inc_disp = float(torch.norm(cur - prev))
        prev = cur
        traj.append({"step": step, "ce_batch": float(loss.item()),
                     "cum_disp": cum_disp, "step_disp": inc_disp,
                     "preclip_gnorm": gnorm, "lr_eff": lr_now,
                     "elapsed_s": round(time.time() - t_start, 1)})
        if step in ckpt_set:
            delta = cur - theta0
            deltas[step] = delta
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            g_g0 = fact_grad(evl, g0_ids, zid)        # the fact battery
            g_m12 = fact_grad(evl, gm12_ids, zid)     # the ruler battery
            cos_g0 = float(torch.dot(delta, g_g0)
                           / (torch.norm(delta) * torch.norm(g_g0) + 1e-30))
            cos_m12 = float(torch.dot(delta, g_m12)
                            / (torch.norm(delta) * torch.norm(g_m12) + 1e-30))
            traj[-1].update({"gm12": gz["mean_pz"], "g0": gz0["mean_pz"],
                             "frac_argmax_z": gz["frac_argmax_z"],
                             "ce_r": ce_r,
                             "cos_delta_fact_g0": cos_g0,
                             "cos_delta_fact_m12": cos_m12})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| {cum_disp:.4f} "
                f"cos(g0) {cos_g0:+.3f}")
        if (time.time() - t_start) > time_cap:
            log(f"  [{tag}] time cap {time_cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step} "
                         f"(max_steps {max_steps})")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": FREEZE_SEED, "zeph_violations": zeph_checks,
            "x_hashes": x_hashes, "deltas": deltas,
            "theta0_norm": float(torch.norm(theta0)),
            "train_seconds": round(time.time() - t_start, 1),
            "opt_class": type(opt).__name__}


# ------------------------------------------------------------------ arms

def _adamw_recipe(net):
    """A0/A5: the recipe VERBATIM."""
    return torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)


def _adamw_b2_999(net):
    return torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.999),
                             weight_decay=0.1)


def _sgd_plain(lr):
    def f(net):
        return torch.optim.SGD(net.parameters(), lr=lr)  # plain: m=0, wd=0
    return f


_const_lr = lambda step: LR
_warmup30_lr = lambda step: LR * min(1.0, step / WARMUP_STEPS)
_const = lambda v: (lambda step: v)

ARM_SPECS = [
    ("a0_adamw_ref", "A0 AdamW reference — the recipe VERBATIM (AdamW "
     "0.9/0.95 wd 0.1, constant lr 1e-3, fresh state): the known kill "
     "clock t*=+2; doubles as the e185 reproduction gate",
     _adamw_recipe, _const_lr),
    ("a1_sgd_1e-3", "A1 SGD matched-lr — plain SGD (momentum 0, wd 0, "
     "clip 1.0 retained) at the same lr 1e-3",
     _sgd_plain(1e-3), _const(1e-3)),
    ("a2a_sgd_3e-3", "A2a SGD ladder — plain SGD at lr 3e-3 (co-reported; "
     "locates the SGD kill clock)", _sgd_plain(3e-3), _const(3e-3)),
    ("a2b_sgd_1e-2", "A2b SGD ladder — plain SGD at lr 1e-2 (co-reported; "
     "locates the SGD kill clock)", _sgd_plain(1e-2), _const(1e-2)),
    ("a3_adamw_warmup30", "A3 AdamW recipe + 30-step linear warmup into "
     "the wash (lr_eff = 1e-3 * min(1, step/30))",
     _adamw_recipe, _warmup30_lr),
    ("a4_adamw_b2_0.999", "A4 AdamW recipe with beta2=0.999 (vs the "
     "recipe's 0.95), all else verbatim", _adamw_b2_999, _const_lr),
    ("a5_adamw_moment_reset", "A5 AdamW moment-reset at wash start — FRESH "
     "optimizer state, same weights (NULL ARM BY CONSTRUCTION: the e185 "
     "cell never loads optimizer state; expected bit-identical to A0 — "
     "run as the inherited-state gate + determinism replicate)",
     _adamw_recipe, _const_lr),
]


def verify_e185(path: Path) -> dict:
    """Verify the embedded e185 control copies against the stored metrics
    file (no silent divergence; e185's verify convention)."""
    src = {"source": "embedded verbatim copy (runs/e185 metrics)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None, "checks": {}}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        rows = mm["traces"]["ctrl"]
        diffs = [abs(a - b) for a, b in zip(
            [r["gm12"] for r in rows], E185_CTRL_GM12)]
        diffs += [abs(a - b) for a, b in zip(
            [r["g0"] for r in rows], E185_CTRL_G0)]
        diffs += [abs(a - b) for a, b in zip(
            [r["ce_r"] for r in rows], E185_CTRL_CE_R)]
        src["checks"]["trace_steps"] = bool(
            [r["freeze_steps"] for r in rows] == E185_TRACE_STEPS)
        dt = mm["displacement"]["table"]["ctrl"]
        for r in dt:
            s = r["step"]
            if s in E185_CTRL_DISP:
                diffs.append(abs(r["cum_disp"] - E185_CTRL_DISP[s]))
        for r in dt:
            if r["step"] <= 10:
                diffs.append(abs(r["ce_batch"]
                                 - E185_CTRL_CE_BATCH[r["step"] - 1]))
        for s_str, rec in mm["gates"]["G_INPUTS"]["per_step"].items():
            diffs.append(0.0 if rec["ctrl"] == E185_XHASH[int(s_str)]
                         else 1.0)
        adj = mm["adjudication"]["conditions"]["CONTROL_KILL"]
        diffs.append(0.0 if adj["t_kill"] == T_KILL_E185 else 1.0)
        diffs.append(abs(adj["D_kill"] - D_KILL_STORED))
        src["max_abs_diff"] = max(diffs)
        src["verified_vs_embedded"] = bool(max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e185/metrics.json (embedded copy "
                             f"verified, max|diff| {max(diffs):.1e})")
    return src


# ------------------------------------------------------------------ t* machinery

def interp_cross(steps: list[int], vals: list[float], bar: float):
    """First crossing BELOW bar (linear in step). Returns (t_x, bracket) or
    (None, None). vals[0] is the step-0 root anchor."""
    for i in range(1, len(vals)):
        v0, v1 = vals[i - 1], vals[i]
        if v1 <= bar:
            if v0 <= bar:
                continue
            s0, s1 = steps[i - 1], steps[i]
            t = s0 + (v0 - bar) / (v0 - v1) * (s1 - s0)
            return t, (s0, s1)
    return None, None


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("opt1_smoke" if SMOKE else "opt1")
    log(f"OPT1 THE OPTIMIZER CONTROLS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), sequential "
        f"arms, stagger {STAGGER_S:.0f}s + {INTER_ARM_S:.0f}s inter-arm, "
        f"per-arm cap {FT_TIME_CAP:.0f}s")
    time.sleep(STAGGER_S)

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185)
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
    rng = E43.SPLICE_RNG
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

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
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
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    evl0 = copy.deepcopy(net0)
    root_cells = {
        "gm12": battery_cell(evl0, gm12_ids, zid)["mean_pz"],
        "g0": battery_cell(evl0, g0_ids, zid)["mean_pz"],
        "gp12": battery_cell(evl0, bat_ids[12], zid)["mean_pz"],
        "ce_r": ce_fixed_cpu(evl0, *r_eval_xy),
    }
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items()
                 if k in keymap}
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

    src_e185 = verify_e185(E185_METRICS)
    log(f"e185 reference: {src_e185['source']}")
    if not src_e185["verified_vs_embedded"]:
        raise RuntimeError("embedded e185 references diverged from the "
                           "stored metrics file — the A0 gate is "
                           "untrustworthy")

    # =====================================================================
    # THE SEVEN ARMS (sequential; stagger between; per-arm 180 s cap)
    # =====================================================================
    arms: dict = {}
    for k, (tag, desc, make_opt, lr_at) in enumerate(ARM_SPECS):
        if k and not SMOKE:
            log(f"[stagger] {INTER_ARM_S:.0f}s before {tag}")
            time.sleep(INTER_ARM_S)
        mx = MAX_STEPS[tag]
        cks = tuple(s for s in CK_LADDER if s <= mx)
        log("=" * 78)
        log(f"ARM {tag} — {desc}; max {mx} steps, checkpoints +{list(cks)}")
        arms[tag] = opt_wash(tag, net0, anchor_neutral, train_ids, itos,
                             r_eval_xy, gm12_ids, g0_ids, zid, cks, mx,
                             make_opt, lr_at)
        arms[tag]["desc"] = desc
        G_DRAWFREE = {f"zeph_violations_{tag}":
                      arms[tag]["zeph_violations"],
                      "pass": bool(arms[tag]["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"

    # =====================================================================
    # GATES: shared input stream (all arms AND e185), A0 == e185 ctrl,
    # A5 == A0 (the null-arm gate)
    # =====================================================================
    tags = [t for t, _, _, _ in ARM_SPECS]
    G_INPUTS = {"per_step": {}, "vs_e185_stored": {}, "pass": None}
    n_shared = min(arms[t]["steps_ran"] for t in tags)
    for step in range(1, n_shared + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in tags}
        same = bool(all(h is not None for h in hs.values())
                    and len(set(hs.values())) == 1)
        G_INPUTS["per_step"][step] = {**hs, "identical": same}
        if step in E185_XHASH:      # e185's stored hashes cover steps 1..10
            G_INPUTS["vs_e185_stored"][step] = {
                "match": bool(hs[tags[0]] == E185_XHASH[step])}
    G_INPUTS["pass"] = bool(
        all(v["identical"] for v in G_INPUTS["per_step"].values())
        and all(v["match"] for v in G_INPUTS["vs_e185_stored"].values()))
    assert G_INPUTS["pass"], ("input streams diverged across arms (or vs "
                              "e185's stored hashes)")
    log(f"G_INPUTS: per-step input batches bit-identical across all seven "
        f"arms AND vs e185's stored hashes (steps 1..{n_shared}): PASS")

    # A0 vs e185's stored control (the reproduction gate)
    a0 = arms["a0_adamw_ref"]
    rep = {"disp": {}, "gm12": {}, "g0": {}, "ce_r": {}, "ce_batch": {}}
    gated_steps = []
    for s in (1, 2, 4, 10):
        row = next((r for r in a0["traj"]
                    if r["step"] == s and "gm12" in r), None)
        if row is None:      # smoke trims the cadence; nothing to gate
            continue
        gated_steps.append(s)
        rep["disp"][s] = row["cum_disp"] - E185_CTRL_DISP[s]
        rep["gm12"][s] = row["gm12"] - E185_CTRL_GM12[
            E185_TRACE_STEPS.index(s)]
        rep["g0"][s] = row["g0"] - E185_CTRL_G0[E185_TRACE_STEPS.index(s)]
        rep["ce_r"][s] = row["ce_r"] - E185_CTRL_CE_R[
            E185_TRACE_STEPS.index(s)]
    for r in a0["traj"]:
        s = r["step"]
        if s <= len(E185_CTRL_CE_BATCH):
            rep["ce_batch"][s] = r["ce_batch"] - E185_CTRL_CE_BATCH[s - 1]
    maxrep = max(abs(v) for d in rep.values() for v in d.values())
    G_A0_E185 = {"per_key_max_abs_diff": {k: max(abs(v) for v in d.values())
                                          for k, d in rep.items()},
                 "gated_steps": gated_steps,
                 "max_abs_diff": maxrep, "bit_tol": G_BIT_TOL,
                 "tol": G_FALLBACK_TOL, "bit": bool(maxrep < G_BIT_TOL),
                 "pass": bool(maxrep < G_FALLBACK_TOL),
                 "note": "A0 (this device, this process) vs e185's stored "
                         "CONTROL: displacements, g-12/g0/CE_R lights, "
                         "per-step in-batch CE — the licensed cell "
                         "reproduced"}
    log(f"G_A0_E185 (A0 vs e185 stored ctrl): max|diff| {maxrep:.2e}: "
        + ("PASS" if G_A0_E185["pass"] else "FAIL")
        + (" (bit)" if G_A0_E185["bit"] else ""))
    if not G_A0_E185["pass"]:
        raise RuntimeError("A0 failed to reproduce e185's stored control")

    # A5 vs A0 (the null-arm / inherited-state gate)
    a5 = arms["a5_adamw_moment_reset"]
    per_bit = {}
    for s in sorted(set(a0["deltas"]) & set(a5["deltas"])):
        per_bit[s] = bool(torch.equal(a0["deltas"][s], a5["deltas"][s]))
    G_A5_NULL = {"bit_identical_deltas": per_bit,
                 "all_bit_identical": bool(per_bit and all(per_bit.values())),
                 "note": "A5 (fresh optimizer state at wash start) vs A0: "
                         "expected bit-identical BY CONSTRUCTION (the "
                         "e185 cell never loads optimizer state) — the "
                         "gate PROVES no consolidation-time optimizer "
                         "state is inherited; doubles as the determinism "
                         "replicate"}
    log(f"G_A5_NULL: A5 deltas vs A0 at {sorted(per_bit)}: "
        + ("BIT-IDENTICAL (null arm confirmed)" if G_A5_NULL["all_bit_identical"]
           else "DIVERGED (hidden state?! report!)"))

    # bonus co-read: cos(delta_arm, delta_A0) at shared checkpoints
    cos_vs_a0 = {t: {} for t in tags}
    for t in tags:
        for s, d in arms[t]["deltas"].items():
            if s in a0["deltas"]:
                cos_vs_a0[t][s] = float(
                    torch.dot(d, a0["deltas"][s])
                    / (torch.norm(d) * torch.norm(a0["deltas"][s]) + 1e-30))

    # =====================================================================
    # t* PER ARM + displacement co-reads + SGD projections
    # =====================================================================
    def ck_table(tag: str) -> list[dict]:
        rows = [{"step": 0, "gm12": root_cells["gm12"],
                 "g0": root_cells["g0"], "ce_r": root_cells["ce_r"],
                 "cum_disp": 0.0, "cos_delta_fact_g0": None,
                 "cos_delta_fact_m12": None, "cos_vs_a0": None}]
        for r in arms[tag]["traj"]:
            if "gm12" in r:
                rows.append({"step": r["step"], "gm12": r["gm12"],
                             "g0": r["g0"], "ce_r": r["ce_r"],
                             "cum_disp": r["cum_disp"],
                             "cos_delta_fact_g0": r["cos_delta_fact_g0"],
                             "cos_delta_fact_m12": r["cos_delta_fact_m12"],
                             "cos_vs_a0": cos_vs_a0[tag].get(r["step"])})
        return rows

    tables = {t: ck_table(t) for t in tags}
    tstar = {}
    for t in tags:
        steps = [r["step"] for r in tables[t]]
        vals = [r["gm12"] for r in tables[t]]
        t_ck = next((r["step"] for r in tables[t]
                     if r["step"] > 0 and r["gm12"] <= SHUT_BAR), None)
        t_x, bracket = interp_cross(steps, vals, SHUT_BAR)
        d_at_kill = next((r["cum_disp"] for r in tables[t]
                          if r["step"] == t_ck), None) if t_ck else None
        # steady displacement rate (last 20 steps) + EXTRAPOLATED clock
        tail = arms[t]["traj"][-20:]
        rate = sum(r["step_disp"] for r in tail) / max(len(tail), 1)
        proj = (D_KILL_STORED / rate if rate > 0 else None)
        tstar[t] = {
            "t_ck": t_ck, "t_x": t_x, "bracket": bracket,
            "D_at_kill": d_at_kill, "D_kill_ratio": (
                d_at_kill / D_KILL_STORED if d_at_kill is not None else None),
            "steady_disp_rate": rate,
            "extrapolated_steps_to_D_kill": proj,
            "extrapolation_flag": ("MEASURED t* (kill inside window)"
                                   if t_ck is not None else
                                   "EXTRAPOLATED ONLY (no kill inside the "
                                   "arm's window; D_kill / steady rate — "
                                   "never adjudicates)"),
            "reached_D_kill": bool(any(r["cum_disp"] >= D_KILL_STORED
                                       for r in tables[t])),
            "final_cum_disp": arms[t]["traj"][-1]["cum_disp"],
            "final_gm12": tables[t][-1]["gm12"],
            "steps_ran": arms[t]["steps_ran"],
            "train_seconds": arms[t]["train_seconds"],
        }
        log(f"t*[{t}]: t_ck={t_ck} t_x="
            + (f"{t_x:.2f}" if t_x is not None else "None")
            + f" D@kill={d_at_kill if d_at_kill is None else round(d_at_kill, 4)}"
            + f" rate={rate:.4f}/step final|d|="
            + f"{tstar[t]['final_cum_disp']:.4f}"
            + (f" proj~{proj:.0f} steps (EXTRAPOLATED)"
               if t_ck is None and proj is not None else ""))

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # OPT-AGNOSTIC -> ADAM-AMPLIFIES -> MIXED-TEXTURE; no shopping)
    # =====================================================================
    T0X = tstar["a0_adamw_ref"]["t_x"]
    control_ok = tstar["a0_adamw_ref"]["t_ck"] is not None
    knob_tags = ["a3_adamw_warmup30", "a4_adamw_b2_0.999",
                 "a5_adamw_moment_reset"]
    sgd_tags = ["a1_sgd_1e-3", "a2a_sgd_3e-3", "a2b_sgd_1e-2"]

    if not control_ok:
        verdict = "MIXED-TEXTURE"
        clause = ("CONTROL FAILURE: A0 did not kill by its window — the "
                  "e185 clock did not reproduce on this device; nothing "
                  "adjudicated (e185's abort convention).")
        bars = {}
    else:
        # OPT-AGNOSTIC's SGD clause
        sgd_kill_detail = {}
        for t in sgd_tags:
            ts = tstar[t]
            fast = bool(ts["t_x"] is not None and ts["t_x"] <= 2 * T0X)
            comp = bool(ts["D_at_kill"] is not None
                        and 0.5 <= ts["D_kill_ratio"] <= 2.0)
            sgd_kill_detail[t] = {"kills_within_2x": fast,
                                  "comparable_displacement": comp,
                                  "fires": bool(fast and comp)}
        sgd_kill_fast = any(d["fires"] for d in sgd_kill_detail.values())
        # knob clause
        knob_detail = {}
        for t in knob_tags:
            ts = tstar[t]
            if ts["t_x"] is not None:
                ratio = ts["t_x"] / T0X
                shifts = bool(ratio >= 2.0 or ratio <= 0.5)
                knob_detail[t] = {"t_x_ratio_vs_a0": ratio, "shifts_2x": shifts,
                                  "basis": "measured"}
            else:
                # no kill inside window: a delay LOWER BOUND only if
                # displacement never reached D_kill (rate-carried delay)
                lb = arms[t]["steps_ran"] / T0X
                knob_detail[t] = {
                    "t_x_ratio_vs_a0": None, "shifts_2x": False,
                    "basis": ("no kill inside window; delay lower-bound "
                              f"window/t_x(A0) = {lb:.1f}x — EXTRAPOLATED, "
                              "does not fire the clause"
                              + (" (displacement stayed below D_kill — "
                                 "rate-carried delay)" if not ts["reached_D_kill"]
                                 else " (D_kill reached without a kill — "
                                      "reported as texture)"))}
        knob_quiet = all(not d["shifts_2x"] for d in knob_detail.values())
        opt_agnostic = bool(sgd_kill_fast and knob_quiet)
        # ADAM-AMPLIFIES
        knob_shift = any(d["shifts_2x"] for d in knob_detail.values())
        sgd_md_detail = {}
        for t in sgd_tags:
            ts = tstar[t]
            reached = ts["reached_D_kill"]
            survived = False
            if reached:
                matched = [r for r in tables[t]
                           if r["cum_disp"] >= D_KILL_STORED]
                survived = bool(matched and all(r["gm12"] >= SURVIVE_BAR
                                                for r in matched))
            md_long = bool(ts["t_x"] is not None and ts["t_x"] >= 5 * T0X)
            sgd_md_detail[t] = {
                "reached_D_kill": reached,
                "gm12_ge_0.5_at_all_matched": survived,
                "t_x_ge_5x_a0": md_long,
                "fires": bool(reached and (survived or md_long)),
                "cap_limited": not reached,
            }
        sgd_md_survive = any(d["fires"] for d in sgd_md_detail.values())
        adam_amplifies = bool(knob_shift or sgd_md_survive)

        bars = {
            "OPT_AGNOSTIC": {"fires": opt_agnostic,
                             "sgd_clause": sgd_kill_detail,
                             "knob_quiet": knob_quiet},
            "ADAM_AMPLIFIES": {"fires": adam_amplifies,
                               "knob_clause": knob_detail,
                               "sgd_matched_displacement_clause": sgd_md_detail},
            "MIXED_TEXTURE": {"fires": not opt_agnostic
                              and not adam_amplifies},
        }
        if opt_agnostic:
            verdict = "OPT-AGNOSTIC"
            firing = [t for t, d in sgd_kill_detail.items() if d["fires"]]
            clause = ("SGD killed within 2x of the AdamW clock at "
                      "comparable displacement (" + ", ".join(firing)
                      + f"; A0 t_x {T0X:.2f}) AND no Adam knob shifted t* "
                      "by >=2x — the no-basin trajectory law is "
                      "optimizer-agnostic memory physics.")
        elif adam_amplifies:
            fired_knobs = [t for t, d in knob_detail.items() if d["shifts_2x"]]
            fired_sgd = [t for t, d in sgd_md_detail.items() if d["fires"]]
            bits = []
            if fired_knobs:
                bits.append("Adam knob(s) shifted t* by >=2x: "
                            + "; ".join(
                                f"{t} t_x "
                                f"{knob_detail[t]['t_x_ratio_vs_a0']:.2f}x A0"
                                for t in fired_knobs))
            if fired_sgd:
                bits.append("matched-displacement SGD survived >=5x longer: "
                            + ", ".join(fired_sgd))
            cap_lim = [t for t, d in sgd_md_detail.items() if d["cap_limited"]]
            clause = ("ADAM-AMPLIFIES per the registered letter — "
                      + "; ".join(bits)
                      + ". CO-READS (report, not adjudication): kill "
                      "displacements D(t*) per arm vs D_kill "
                      f"{D_KILL_STORED:.4f} ("
                      + "; ".join(f"{t} {tstar[t]['D_at_kill']:.3f}"
                                  for t in tags if tstar[t]["D_at_kill"])
                      + "); SGD displacement rates and EXTRAPOLATED clocks "
                      + ("; ".join(
                          f"{t} {tstar[t]['steady_disp_rate']:.4f}/step "
                          f"-> ~{tstar[t]['extrapolated_steps_to_D_kill']:.0f} "
                          "steps" for t in sgd_tags))
                      + (f"; SGD arms cap-limited (never reached D_kill "
                         f"within the CPU window): {', '.join(cap_lim)}"
                         if cap_lim else "")
                      + ".")
            verdict = "ADAM-AMPLIFIES"
        else:
            verdict = "MIXED-TEXTURE"
            clause = ("graded outcome — full arm table reported verbatim; "
                      "no binary claim. SGD arms: "
                      + "; ".join(
                          f"{t} t_x={tstar[t]['t_x']}, "
                          f"reached_D_kill={tstar[t]['reached_D_kill']}"
                          for t in sgd_tags)
                      + "; knobs: "
                      + "; ".join(f"{t} ratio="
                                  + (f"{knob_detail[t]['t_x_ratio_vs_a0']:.2f}"
                                     if knob_detail[t]["t_x_ratio_vs_a0"]
                                     else "n/a")
                                  for t in knob_tags)
                      + ".")

    log("=" * 78)
    log(f"OPT1 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    ckpt_inventory = {}
    pref = "smoke_" if SMOKE else ""
    for t in tags:
        # A0: the e185 cadence verbatim; others: final state (provenance)
        save_at = ([1, 2, 4, 10] if t == "a0_adamw_ref"
                   else [arms[t]["steps_ran"]])
        for s in save_at:
            if s not in arms[t]["sds"]:
                continue
            name = f"{pref}opt1_{t}_s{s}"
            path = CKPT_DIR / f"{name}.pt"
            torch.save({"model": arms[t]["sds"][s],
                        "meta": {"experiment": "opt1", "arm": t,
                                 "steps": int(s),
                                 "desc": arms[t]["desc"],
                                 "input_seed": FREEZE_SEED,
                                 "opt": arms[t]["opt_class"],
                                 "base": f"runs/checkpoints/{ROOT_CK}"}},
                       path)
            ckpt_inventory[name] = {"path": f"runs/checkpoints/{name}.pt",
                                    "arm": t, "steps": int(s)}
            log(f"[ckpt] saved {name}.pt")

    metrics = {
        "experiment": "opt1_optimizer_controls",
        "date": common.now_iso(),
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the fast fact-kill OPTIMIZER-AGNOSTIC memory "
                     "physics, or is it carried by Adam's moment/"
                     "normalization structure? The licensed e185 wash cell "
                     "VERBATIM (true targets, seed-10902 stream, the "
                     "consolidated host fact) with ONLY the optimizer "
                     "swapped: A0 AdamW recipe, A1 SGD matched-lr, A2 SGD "
                     "ladder {3e-3, 1e-2}, A3 warmup-30, A4 beta2=0.999, "
                     "A5 moment-reset-at-wash-start. Reads: g-12/g0 "
                     "batteries, CE_R, per-step displacement + pre-clip "
                     "grad norms, and the alignment co-read "
                     "cos(delta_theta_t, grad fact) at every checkpoint."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) VERBATIM — the "
                             "consolidated host fact under its neutral/"
                             "corpus wash",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random "
                     "corpus windows, full-token CE, clip 1.0",
            "targets": "TRUE (the real neutral wash — the optimizer is the "
                       "only delta)",
            "input_seed": FREEZE_SEED,
            "ckpt_cadence": "e185's {1,2,4,10} verbatim + geometric "
                            "extension {8,20,40,80,160}",
            "measure_light": "g-12 / g0 batteries + CE_R + alignment + "
                             "displacement + pre-clip grad norms (the "
                             "dispatch's read list; no census/deletions)",
        },
        "arms": {
            t: {
                "desc": arms[t]["desc"],
                "opt_class": arms[t]["opt_class"],
                "opt_spec": {"kind": ("AdamW" if "adamw" in t else "SGD"),
                             "lr": LR if "adamw" in t else
                                   float(t.split("_")[-1]),
                             "betas": ((0.9, 0.999)
                                       if t == "a4_adamw_b2_0.999"
                                       else (0.9, 0.95)
                                       if "adamw" in t else None),
                             "weight_decay": (0.1 if "adamw" in t else 0.0),
                             "momentum": (0 if "sgd" in t else None),
                             "clip": 1.0,
                             "lr_schedule": ("linear warmup 30 steps"
                                             if t == "a3_adamw_warmup30"
                                             else "constant")},
                "max_steps": MAX_STEPS[t],
                "steps_ran": arms[t]["steps_ran"],
                "train_seconds": arms[t]["train_seconds"],
                "time_cap": FT_TIME_CAP,
                "seed": FREEZE_SEED,
                "traj": arms[t]["traj"],
                "ckpt_table": tables[t],
                "tstar": tstar[t],
                "zeph_violations": arms[t]["zeph_violations"],
            } for t in tags
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_INPUTS": G_INPUTS,
                  "G_A0_E185": G_A0_E185, "G_A5_NULL": G_A5_NULL},
        "references": {
            "e185": {"metrics": "runs/e185/metrics.json",
                     "checkpoints": "runs/checkpoints/e185_ctrl_s{1,2,4,10}.pt",
                     "provenance": src_e185,
                     "role": "the licensed cell + the stored control A0 must "
                             "reproduce (displacements, lights, per-step CE, "
                             "input md5s)"},
            "e187": {"metrics": "runs/e187/metrics.json",
                     "role": "the n=3 no-basin license this cell's question "
                             "sits on (T121); its verified embedded e185 "
                             "copies are this file's provenance"},
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "t_a0": {"t_ck": tstar["a0_adamw_ref"]["t_ck"],
                     "t_x": T0X,
                     "note": "the AdamW reference clock (e185's t*=+2 "
                             "checkpoint convention; interpolated t_x for "
                             "ratios)"},
            "D_kill": D_KILL_STORED,
            "composite_order": "OPT-AGNOSTIC -> ADAM-AMPLIFIES -> "
                               "MIXED-TEXTURE (frozen before compute)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 per arm (single training seed per arm, "
                            "first cell; replicate ladder only if a bar "
                            "fires); ONE root, ONE input stream (seed "
                            "10902, bit-gated across all seven arms AND vs "
                            "e185's stored md5s); single family (the "
                            "e131 consolidated line) — the replicate "
                            "samples nothing here; every arm differs ONLY "
                            "in optimizer construction/lr schedule"),
            "float_texture": ("CPU fp32 texture: all trainings this "
                              "process, this CPU, 4 threads (the e185-era "
                              "reduction order); the GPU-era e185 "
                              "reference is gated ON THIS DEVICE via A0 "
                              "(G_A0_E185) rather than compared across "
                              "devices"),
            "no_guaranteed_spare": ("every arm TRAINS on the licensed "
                                    "stream — every arm could kill; the "
                                    "battery is honest because nothing in "
                                    "it is guaranteed-to-spare (A5's "
                                    "expected null is a state-inheritance "
                                    "statement, not a sparing guarantee)"),
            "projections_never_adjudicate": ("SGD arms that never reach "
                                             "D_kill within the CPU cap "
                                             "carry EXTRAPOLATED clocks "
                                             "(D_kill / steady measured "
                                             "rate); the registered bars "
                                             "fire only on measured "
                                             "crossings; cap-limits are "
                                             "reported as ambiguity"),
            "step_clock_vs_displacement_gate": ("t* in STEPS is confounded "
                                                "by each arm's "
                                                "displacement RATE; the "
                                                "co-read D(t*) vs D_kill "
                                                "and the per-step rates "
                                                "separate rate-carried "
                                                "clock shifts (warmup) "
                                                "from gating-carried ones "
                                                "— reported per arm, "
                                                "never adjudicated away"),
            "logit_prediction_check": ("check-in 6/8's arithmetic "
                                       "predicts A0's step-1 displacement "
                                       "= lr*sqrt(N) = 1.655 and SGD's "
                                       "= lr*min(||g||,1); the pre-clip "
                                       "grad-norm column tests this "
                                       "directly (Adam amplifies ||g||<<1 "
                                       "gradients into lr-sized steps)"),
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
                   "stagger_s": STAGGER_S, "inter_arm_s": INTER_ARM_S},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_kill_clocks(rd / "opt1_kill_clocks.png", tables, tstar, tags,
                     verdict, clause)
    plot_trajectory(rd / "opt1_trajectory.png", tables, arms, tstar, tags,
                    D_KILL_STORED)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'opt1_kill_clocks.png'}, "
        f"{rd / 'opt1_trajectory.png'}, {len(ckpt_inventory)} checkpoints "
        f"in runs/checkpoints/opt1_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

ARM_COLORS = {
    "a0_adamw_ref": "crimson", "a1_sgd_1e-3": "royalblue",
    "a2a_sgd_3e-3": "dodgerblue", "a2b_sgd_1e-2": "darkblue",
    "a3_adamw_warmup30": "darkorange", "a4_adamw_b2_0.999": "seagreen",
    "a5_adamw_moment_reset": "orchid",
}
ARM_LABELS = {
    "a0_adamw_ref": "A0 AdamW recipe (t*=+2)", "a1_sgd_1e-3": "A1 SGD 1e-3",
    "a2a_sgd_3e-3": "A2a SGD 3e-3", "a2b_sgd_1e-2": "A2b SGD 1e-2",
    "a3_adamw_warmup30": "A3 AdamW warmup30",
    "a4_adamw_b2_0.999": "A4 AdamW b2=.999",
    "a5_adamw_moment_reset": "A5 moment-reset (=A0)",
}


def plot_kill_clocks(path, tables, tstar, tags, verdict, clause):
    """(a) g-12 vs step, all arms + bars; (b) the arm table + verdict."""
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for t in tags:
        rows = tables[t]
        ax.plot([r["step"] for r in rows], [r["gm12"] for r in rows],
                "o-", ms=4.5, lw=1.6, color=ARM_COLORS[t], alpha=0.9,
                label=ARM_LABELS[t])
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen", f"{SURVIVE_BAR} SPARE"),
                         (SHUT_BAR, "tab:purple", f"{SHUT_BAR} DISSOLVE")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.set_xlabel("wash step (checkpoint cadence: e185 {1,2,4,10} + extension)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE KILL CLOCKS — g-12 vs step, all arms (the licensed "
                 "e185 cell, optimizer swapped)", fontsize=10)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "THE ARM TABLE (D_kill "
            f"{D_KILL_STORED:.4f}; A0 t_x {tstar['a0_adamw_ref']['t_x']:.2f})",
            fontsize=9, va="top", family="monospace", weight="bold")
    y -= 0.045
    ax.text(0.02, y, "  arm                    t_ck  t_x   ratio  D@kill  "
            "D/Dk  rate/step  steps  read",
            fontsize=7.0, va="top", family="monospace")
    y -= 0.032
    for t in tags:
        ts = tstar[t]
        ratio = (ts["t_x"] / tstar["a0_adamw_ref"]["t_x"]
                 if ts["t_x"] is not None else None)
        if ts["t_ck"] is not None:
            read = "KILL"
        elif ts["reached_D_kill"]:
            read = "D_kill reached, no kill"
        elif ts["extrapolated_steps_to_D_kill"] is not None:
            read = (f"cap-limited "
                    f"(proj ~{ts['extrapolated_steps_to_D_kill']:.0f})")
        else:
            read = "cap-limited"
        ax.text(0.02, y,
                f"  {t:<22} {str(ts['t_ck']):>4} "
                + (f"{ts['t_x']:4.2f}" if ts["t_x"] is not None else " n/a")
                + " "
                + (f"{ratio:5.2f}" if ratio is not None else "  n/a")
                + " "
                + (f"{ts['D_at_kill']:6.3f}" if ts["D_at_kill"] is not None
                   else "   n/a")
                + "  "
                + (f"{ts['D_kill_ratio']:4.2f}" if ts["D_kill_ratio"]
                   is not None else " n/a")
                + f" {ts['steady_disp_rate']:9.4f} {ts['steps_ran']:5d}"
                + f"  {read}",
                fontsize=7.0, va="top", family="monospace",
                color=ARM_COLORS[t])
        y -= 0.031
    y -= 0.012
    ax.text(0.02, y, f"OPT1 VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.04
    for wd in textwrap.wrap(clause, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.024
    fig.suptitle("OPT1 — THE OPTIMIZER CONTROLS: the e185 wash cell with "
                 "the optimizer swapped", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_trajectory(path, tables, arms, tstar, tags, d_kill):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    # (0,0) D(t) vs step + D_kill + SGD projections
    ax = axes[0, 0]
    for t in tags:
        tr = arms[t]["traj"]
        ax.plot([r["step"] for r in tr], [r["cum_disp"] for r in tr],
                "-", lw=1.7, color=ARM_COLORS[t], alpha=0.9,
                label=ARM_LABELS[t])
        ts = tstar[t]
        if ts["t_ck"] is None and ts["extrapolated_steps_to_D_kill"] \
                and ts["extrapolated_steps_to_D_kill"] > ts["steps_ran"]:
            rate = ts["steady_disp_rate"]
            ax.plot([ts["steps_ran"], ts["extrapolated_steps_to_D_kill"]],
                    [ts["final_cum_disp"],
                     ts["final_cum_disp"] + rate * (
                         ts["extrapolated_steps_to_D_kill"]
                         - ts["steps_ran"])],
                    ":", lw=1.4, color=ARM_COLORS[t], alpha=0.7)
    ax.axhline(d_kill, ls="--", lw=1.6, color="k", alpha=0.7,
               label=f"D_kill {d_kill:.3f} (A0 dead at +2)")
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"cumulative $\|\theta_t-\theta_0\|_2$ (2.74M params)")
    ax.legend(fontsize=7.2, loc="lower right")
    ax.set_title("D(t) per arm (dotted = EXTRAPOLATED steady-rate "
                 "projection, never adjudicated)", fontsize=10)

    # (0,1) alignment co-read
    ax = axes[0, 1]
    for t in tags:
        rows = [r for r in tables[t] if r["cos_delta_fact_g0"] is not None]
        ax.plot([r["step"] for r in rows],
                [r["cos_delta_fact_g0"] for r in rows], "o-", ms=4.5,
                lw=1.6, color=ARM_COLORS[t], alpha=0.9,
                label=ARM_LABELS[t])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("wash step (checkpoint)")
    ax.set_ylabel(r"cos($\Delta\theta_t$, $\nabla$ fact g0)")
    ax.legend(fontsize=7.2, loc="lower right")
    ax.set_title("THE ALIGNMENT CO-READ — cumulative displacement vs the "
                 "fact's death gradient (critic's sign: negative = "
                 "death-aligned)", fontsize=9.5)

    # (1,0) displacement-vs-damage plane
    ax = axes[1, 0]
    for t in tags:
        rows = [r for r in tables[t] if r["step"] > 0]
        ax.plot([r["cum_disp"] for r in rows], [r["gm12"] for r in rows],
                "o-", ms=4.5, lw=1.6, color=ARM_COLORS[t], alpha=0.9,
                label=ARM_LABELS[t])
    ax.plot([0.0], [tables["a0_adamw_ref"][0]["gm12"]], "k*", ms=13,
            label="root (g-12 0.916)")
    for yv, col in ((SURVIVE_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    ax.axvline(d_kill, ls=":", lw=1.8, color="k", alpha=0.7,
               label=f"D_kill {d_kill:.3f}")
    ax.set_xlabel(r"cumulative $\|\theta_t-\theta_0\|_2$")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title("the displacement-vs-damage plane (does every arm die at "
                 "the same D? — the trajectory-law co-read)", fontsize=10)

    # (1,1) the Adam-amplification panel: per-step |d| vs pre-clip ||g||
    ax = axes[1, 1]
    for t in tags:
        tr = arms[t]["traj"]
        ax.scatter([r["preclip_gnorm"] for r in tr],
                   [r["step_disp"] for r in tr], s=9,
                   color=ARM_COLORS[t], alpha=0.65,
                   label=ARM_LABELS[t])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"pre-clip $\|g\|_2$ (per step)")
    ax.set_ylabel(r"per-step $\|\Delta\theta\|_2$")
    ax.legend(fontsize=7.2, loc="lower right")
    ax.set_title(r"THE AMPLIFICATION PANEL — Adam turns any $\|g\|$ into "
                 r"an lr-sized step (1e-3·√N≈1.65); SGD moves "
                 r"lr·min($\|g\|$,1)",
                 fontsize=9.5)

    fig.suptitle("OPT1 — trajectory reads: displacement, alignment, and the "
                 "amplification that sets the clock", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
