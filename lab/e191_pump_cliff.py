"""E191 — THE PUMP-CLIFF MAP (T143's discriminator; opt1c's follow-up).

WHY: opt1c (runs/opt1c/metrics.json, T143) ran the raw-gradient
DIRECTION at Adam's measured per-step L2 (1.6543/step) on the
licensed e185 wash cell: the fact PUMPED to 0.955 at D ~ 0.33 then
cliffed to death by D ~ 0.92-1.17 — inside ONE big step. The bleed
(opt1b: SGD 1e-2, ~0.0066/step) pumped at the SAME D ~ 0.32 on tiny
steps and stayed alive through it. THE OPEN QUESTION (T143's first
dispatched split): is the pump-cliff TERRAIN (a static property of
the displacement region — a ridge then a cliff in the g-direction,
present under a single static jump) or DYNAMICS (the big step
overshoots THROUGH the cliff into worse territory a static jump at
the same endpoint would not reach)?

WHAT IT BUILDS ON: opt1c's gradient-direction machinery VERBATIM
(the t=0 bit-identity gate, the licensed e185 cell rebuild, the
flat-vector instruments); opt1c's committed along-path densify rows
(runs/opt1c/metrics.json arm.densify — the dynamic path's own
static-looking reads); opt1b's committed bleed curve; opt1's
committed Adam-A0 curve (overlay context). WHAT IS NEW: the graded
STATIC profile — single perturb-and-eval jumps along the unit
g-direction at 12 graded D values (fresh independent code path, the
direction recomputed and bit-gated), extended BEYOND the dynamic
step's own endpoint (D 2.0 > 1.6543), the cliff edge resolved at
0.8 / 0.92 / 1.0, a 5-point bit-level crosscheck arm at the
committed dynamic points, and CE_R read at every D (does the
organism wreck in step with the fact, or after?).

THE CELL (eval-only, CPU-light; no GPU claim): STATIC single jumps
theta_D = theta_0 - D * u, where u = g_0/||g_0||_2 is the unit
RAW WASH-BATCH GRADIENT at the ROOT (opt1c's t=0 convention — the
step-1 batch, bit-gated) — perturb-and-eval on the eval twin; NO
optimizer, NO steps, NO second gradient. Graded
D in {0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.92, 1.0, 1.17, 1.5,
2.0}. Reads: fact g-12 (install-60 battery, e185's convention) +
CE_R at each. 3 seeds NOT needed — the direction is DETERMINISTIC
(fixed root, fixed bit-gated batch, clip is a positive rescale so
the direction is clip-invariant); documented here per the dispatch.

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - STATIC-CLIFF: "fires if the static profile matches the dynamic
    cliff — pumps near D ~ 0.33 and falls below 0.27 by D ~
    0.92-1.17 — the pump-cliff is TERRAIN (geometry); the
    direction's lethality is a static property; e188's cross-class
    break is a geometry fact."
  - STATIC-SPARES: "fires if the static jump at D 0.92 leaves the
    fact alive (> 0.27) — the kill is DYNAMIC overshoot (the single
    big step passes THROUGH the cliff into worse territory a static
    endpoint-jump never visits); trajectory effects carry lethality
    that geometry alone does not."
  - GRADED-MAP: "any partial pattern — the full profile reported
    verbatim as the map; the terrain/dynamics split stated as
    measured."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses,
they do not move the bars):
  * PUMP clause: gm12 at D = 0.33 strictly above the root read
    (0.9156) — the ridge RISES above the root (the dynamic pump rose
    to 0.9554 at D 0.3309).
  * KILL clause: some graded D in {0.92, 1.0, 1.17} reads
    gm12 <= 0.27 — "falls below 0.27 by D ~ 0.92-1.17".
  * STATIC-CLIFF fires iff PUMP and KILL. STATIC-SPARES fires iff
    gm12 at D = 0.92 > 0.27. GRADED-MAP otherwise (any partial
    pattern). Composite order frozen STATIC-CLIFF -> STATIC-SPARES
    -> GRADED-MAP; the two first bars' letters CAN co-hold (kill
    clause met at 1.0/1.17 while 0.92 stays alive — a cliff edge
    inside (0.92, 1.17]): then STATIC-CLIFF takes the verdict AND
    the overlap is reported verbatim in the clause (no shopping).
  * the cliff-edge bracket = the tightest adjacent alive->dead pair
    on the graded grid (root included as D = 0); reported either way.
  * the CROSSCHECK arm (5 extra jumps at the EXACT committed
    dynamic points: the four densify Ds and the step-1 endpoint
    D 1.6542682647705078) is VERIFICATION (a different code path
    must reproduce the committed reads bit-close), never an
    adjudication input.
  * THE RIDER (±90-degree-rotated g within its 2D span): SKIPPED
    per the dispatch ("unless trivial" — a canonical second basis
    vector for the span is not trivially defined; the core map is
    the graded profile). Recorded in trims.

THE GEOMETRIC FACT THIS CELL MUST DISCLOSE (honesty, stated before
compute): opt1c's dynamic path died INSIDE its first step, so on
[0, 1.6543] the dynamic trajectory IS the straight ray
theta_0 - D*u — the static jump at D and the dynamic path at D are
the SAME parameter point (up to float noise). Terrain-vs-dynamics
is therefore NOT separable by construction WITHIN the dynamic span;
the discriminating content of this map is (a) the fresh independent
recompute of the direction and reads (machinery verification: a
different code path, crosschecked bit-close at 5 committed points),
(b) the extension to D = 2.0 BEYOND the dynamic endpoint, (c) the
tighter cliff-edge bracket (0.8 / 0.92 / 1.0 between the committed
0.66 / 0.99), (d) the CE_R co-read. The verdict states the split AS
MEASURED under this disclosure.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE the profile runs):
the standard cell gates (corpus ZEPH count 0; splice mix
{FLORIZEL: 19, ELIZABETH: 41}; battery shapes 60 x (130 +- j);
neutral bank 16 starts bit-equal to e185's stored list; root gated
vs e151's before-cells) PLUS the t=0 BIT-IDENTITY GATE (opt1c's
verbatim: the step-1 batch md5 vs e185's stored hash; the step-1
forward CE and pre-clip grad norm vs opt1c/opt1's committed values;
one fresh AdamW-recipe step from the root reproducing A0's committed
step-1 L2) PLUS G_DIR, the DIRECTION gate: the recomputed unit
direction u vs opt1c's committed step-1 checkpoint
(runs/checkpoints/opt1c_sgrad_dir_s1.pt) — cos(theta_1 - theta_0,
-u) and ||theta_1 - theta_0|| vs the committed cumulative
displacement. WHAT THE ARMS GUARANTEE: NOTHING — the static profile
could be flat, ridge-then-cliff, or monotone; that openness is the
point.

CO-READS: (1) CE_R at every D (does the organism wreck in step
with the fact, or after?); (2) frac argmax Z (free from the same
battery call); (3) the committed three-way overlay — static (this
run) vs dynamic (opt1c: ckpt_table + densify) vs the bleed (opt1b)
— THE terrain/dynamics figure; (4) the profile's shape vs the
store's earlier flat-vs-cone forms (g3R/g3K conventions; noted in
the honesty block, no claims).

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1
forced before torch; the GPU is g1bS's — never claimed), torch
threads 4 (the e185/opt1/opt1b/opt1c reduction order), EVAL-ONLY
(no training: every grid point is an independent perturb-and-eval
off the root; bit-deterministic; no chunk machinery needed —
PROGRESSIVE PARTIAL metrics.json writes carry the outage honesty
instead), n=1, single organism (the e131 consolidated root), seed
lineage 10902.

Outputs: runs/e191/{metrics.json, e191_pump_cliff.png,
e191_coreads.png, journal.jsonl}; checkpoint
runs/checkpoints/e191_static_dir_u.pt (the direction vector, for
any rider/follow-up). No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e191_pump_cliff.py    (E191_SMOKE=1 shakedown)
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

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185/opt1/opt1b/opt1c-era reduction order

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E191_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e191 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT (opt1/opt1b/opt1c's, verbatim)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E185_METRICS = E43.REPO / "runs" / "e185" / "metrics.json"
OPT1_METRICS = E43.REPO / "runs" / "opt1" / "metrics.json"
OPT1B_METRICS = E43.REPO / "runs" / "opt1b" / "metrics.json"
OPT1C_METRICS = E43.REPO / "runs" / "opt1c" / "metrics.json"
OPT1C_S1_CK = CKPT_DIR / "opt1c_sgrad_dir_s1.pt"   # the committed dynamic step-1 endpoint
GEOS = (-12, 0, 12)               # battery ctx offsets (e185 convention)

# ---- the map + the run envelope (dispatch-frozen) -------------------------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185/opt1/opt1b/opt1c)
LR_ADAMW = 1e-3                   # A0's lr (the t=0 reproduction only; this cell has NO lr)
STEP_SCALE_COMMITTED = 1.6542880535125732   # opt1 A0's committed step-1 L2 (the t=0 gate reference)
CE1_COMMITTED = 1.356567621231079           # opt1 A0's committed step-1 batch CE
GN1_COMMITTED = 0.9829167127609253          # opt1 A0's committed step-1 pre-clip grad norm
DYN_S1_DISP_COMMITTED = 1.6542682647705078  # opt1c's committed step-1 cumulative displacement
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — the bars' 0.27
PUMP_D = 0.33                     # the pump clause's D (dispatch letter)
SPARES_D = 0.92                   # STATIC-SPARES' D (dispatch letter)
KILL_BAND_GRID = (0.92, 1.0, 1.17)  # the kill clause's graded Ds (dispatch letter "0.92-1.17")
D_GRID = (0.05, 0.1, 0.2, 0.33, 0.5, 0.66,
          0.8, 0.92, 1.0, 1.17, 1.5, 2.0)   # the graded static map (dispatch-frozen)
E170_ANCHOR_SEED = 170            # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
DIR_COS_TOL = 1e-6                # G_DIR: 1 - cos must sit under this (bit-grade)
DIR_COS_HARD = 1e-4               # G_DIR: hard-fail above this
if SMOKE:                         # true shakedown trims (documented in deviations)
    D_GRID = (0.1, 0.33, 1.0)

# ---- gates / references (full precision, = stored metrics; opt1c's set) ----------
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
    "static_cliff": "STATIC-CLIFF: \"fires if the static profile matches the "
        "dynamic cliff — pumps near D ~ 0.33 and falls below 0.27 by D ~ "
        "0.92-1.17 — the pump-cliff is TERRAIN (geometry); the direction's "
        "lethality is a static property; e188's cross-class break is a "
        "geometry fact.\"",
    "static_spares": "STATIC-SPARES: \"fires if the static jump at D 0.92 "
        "leaves the fact alive (> 0.27) — the kill is DYNAMIC overshoot (the "
        "single big step passes THROUGH the cliff into worse territory a "
        "static endpoint-jump never visits); trajectory effects carry "
        "lethality that geometry alone does not.\"",
    "graded_map": "GRADED-MAP: \"any partial pattern — the full profile "
        "reported verbatim as the map; the terrain/dynamics split stated as "
        "measured.\"",
    "operationalizations": "STATIC single jumps theta_D = theta_0 - D*u with "
        "u = g_0/||g_0||_2 the unit RAW WASH-BATCH GRADIENT at the ROOT "
        "(opt1c's t=0 convention; the step-1 batch, bit-gated; g_0 post-clip "
        "— the clip is a positive rescale so the direction is "
        "clip-invariant), perturb-and-eval, NO optimizer/steps; graded D in "
        "{0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.92, 1.0, 1.17, 1.5, 2.0}; "
        "reads = fact g-12 + CE_R at each; 3 seeds NOT needed (the direction "
        "is deterministic; documented); PUMP clause = gm12(0.33) > root "
        "gm12; KILL clause = some D in {0.92, 1.0, 1.17} reads gm12 <= "
        "0.27; STATIC-CLIFF fires iff PUMP and KILL; STATIC-SPARES fires "
        "iff gm12(0.92) > 0.27; GRADED-MAP otherwise; composite order frozen "
        "STATIC-CLIFF -> STATIC-SPARES -> GRADED-MAP (a co-hold of the two "
        "first letters — kill at 1.0/1.17 with 0.92 alive — gives the "
        "verdict to STATIC-CLIFF and the overlap is reported verbatim; no "
        "shopping); the 5-point crosscheck arm (the four committed densify "
        "Ds + the committed step-1 endpoint) is verification only, never an "
        "adjudication input; the ±90-degree rider is SKIPPED per dispatch "
        "(recorded in trims).",
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = [
    "THE RIDER (±90-degree-rotated g-direction within its 2D span): SKIPPED "
    "per the dispatch's own letter ('SKIP the rider unless trivial — the "
    "core map is the graded static profile'). It is not trivial: a canonical "
    "second basis vector for u's 2D span is undefined without an extra "
    "convention (which vector pairs with u defines the rotation's meaning). "
    "The direction vector is checkpointed (runs/checkpoints/"
    "e191_static_dir_u.pt) so a rider arm can be registered properly if "
    "wanted.",
]
deviations: list[str] = [
    "EVAL-ONLY cell: no training, no optimizer state, no chunk machinery — "
    "every grid point is an independent perturb-and-eval off the root "
    "(bit-deterministic; a re-run is bit-identical and cheap). PROGRESSIVE "
    "PARTIAL metrics.json writes (after the gates, after every profile "
    "point) carry the outage honesty instead, per the opt1c recovery "
    "lesson.",
    "No seed sweep: the direction u is a deterministic function of (the "
    "root, the bit-gated step-1 batch, the clip rescale) and every read is "
    "a deterministic function of (theta_D, the fixed batteries) — the "
    "dispatch documents '3 seeds not needed'; the honest replicate axis "
    "would be a second ORGANISM (a different root), which is a separate "
    "dispatch decision.",
    "Read list = the dispatch's: fact g-12 + CE_R at each D. frac_argmax_z "
    "is recorded free from the same battery call (one forward pass set); "
    "the g0 battery and the alignment reads are NOT taken (the static delta "
    "is along u by construction — cos(delta, u) = -1 identically; an "
    "alignment read would be a tautology).",
    "The CROSSCHECK arm (5 extra static jumps at the EXACT committed "
    "dynamic points: densify Ds 0.3308580815792084 / 0.6617161631584167 / "
    "0.9925732612609863 / 1.323432207107544 + the step-1 endpoint "
    "1.6542682647705078) was added by this design as VERIFICATION: a "
    "different code path (fresh gradient, direct theta_0 - D*u placement "
    "vs opt1c's per-tensor p.add_ and densify's th_s0 + f*seg) must "
    "reproduce the committed reads bit-close. Verification only — never an "
    "adjudication input (frozen in the operationalizations).",
    "G_DIR added (Rule 12, this cell's own geometry check): the recomputed "
    "unit direction u vs opt1c's committed step-1 CHECKPOINT (cos(theta_1 - "
    "theta_0, -u) and ||theta_1 - theta_0|| vs the committed cumulative "
    "displacement) — the static map's direction is the dynamic step's own "
    "displacement vector, gated before the profile runs.",
    "The on-span coincidence is DISCLOSED in the honesty block and the "
    "docstring: opt1c's dynamic path died inside its first step, so on "
    "[0, 1.6543] static-at-D and dynamic-at-D are the same parameter point "
    "(up to float noise); the map's discriminating content is the fresh "
    "recompute, the D = 2.0 extension beyond the dynamic endpoint, the "
    "tighter cliff-edge bracket, and the CE_R co-read.",
    "The direction vector is checkpointed (runs/checkpoints/"
    "e191_static_dir_u.pt: u + theta0_md5 + step_scale + meta) for any "
    "rider/follow-up cell — cheap provenance, no recompute needed.",
    "Smoke mode trims: D grid {0.1, 0.33, 1.0}, crosscheck = the step-1 "
    "endpoint only, no adjudication (verdict stamped SMOKE; nothing "
    "adjudicated).",
    "FLOAT TEXTURE (observed in the smoke, documented): (a) the post-clip "
    "gradient re-normed via torch.norm over the concatenated flat vector "
    "reads 0.9828128814697266 while clip_grad_norm_'s own returned norm "
    "reads 0.9829167127609253 — a ~1e-4 relative fp32 reduction-order "
    "difference on 2.7M elements; the clip is inactive either way (coef = "
    "1.0) and the direction u is a positive rescale of the same gradient, "
    "so the direction is unaffected. (b) G_DIR's cosine is computed in "
    "fp64: the fp32 torch.dot over 2.7M elements reads cos 1.0002 (above "
    "1) from accumulation order alone — the fp64 cosine reads "
    "0.9999999999992 and ||delta/||delta|| + u/||u||||_2 = 1.2e-5, i.e. "
    "the recomputed direction IS the committed step's displacement "
    "direction to bit grade; the gate's meaning is unchanged, its "
    "estimator is now truthful.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/opt1c_direction_size.py VERBATIM (whose own provenance is
# lab/opt1_optimizer_controls.py / lab/opt1b_sgd_kill.py via
# lab/e185_noise_wash.py — the e176n lineage). Copied rather than imported
# to own the device policy.

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
    """Copy a flat vector back into parameters (the static-jump loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def flat_md5(net: TinyGPT) -> str:
    import hashlib
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e191_smoke" if SMOKE else "e191")
    log(f"E191 THE PUMP-CLIFF MAP (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), EVAL-ONLY, "
        f"{len(D_GRID)} graded static jumps + 5-point crosscheck, n=1, seed "
        f"lineage {FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (OPT1C_METRICS, OPT1B_METRICS, OPT1_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    if not OPT1C_S1_CK.exists():
        raise RuntimeError(f"missing opt1c step-1 checkpoint: {OPT1C_S1_CK}")
    opt1m = json.loads(OPT1_METRICS.read_text(encoding="utf-8"))
    opt1bm = json.loads(OPT1B_METRICS.read_text(encoding="utf-8"))
    opt1cm = json.loads(OPT1C_METRICS.read_text(encoding="utf-8"))
    a0 = opt1m["arms"]["a0_adamw_ref"]
    a0_s1 = next(r for r in a0["traj"] if r["step"] == 1)
    assert abs(a0_s1["cum_disp"] - STEP_SCALE_COMMITTED) < 1e-12, \
        "opt1's committed A0 step-1 displacement drifted vs this file's copy"
    assert abs(a0_s1["ce_batch"] - CE1_COMMITTED) < 1e-12
    assert abs(a0_s1["preclip_gnorm"] - GN1_COMMITTED) < 1e-12
    opt1c_arm = opt1cm["arm"]
    dyn_trajs1 = next(r for r in opt1c_arm["traj"] if r["step"] == 1)
    assert abs(dyn_trajs1["cum_disp"] - DYN_S1_DISP_COMMITTED) < 1e-12, \
        "opt1c's committed step-1 displacement drifted vs this file's copy"
    dyn_ck = {r["step"]: r for r in opt1c_arm["ckpt_table"]}
    dyn_densify = list(opt1c_arm["densify"])          # 4 committed along-path rows
    assert len(dyn_densify) == 4, f"expected 4 densify rows, got {len(dyn_densify)}"
    dyn_curve = [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
                  "ce_r": r["ce_r"], "provenance": "opt1c ckpt (committed)"}
                 for r in opt1c_arm["ckpt_table"]]
    adam_a0_curve = [{"step": r["step"], "D": r["cum_disp"], "gm12": r["gm12"],
                      "ce_r": r.get("ce_r")}
                     for r in a0["ckpt_table"] if r["step"] > 0]
    bleed_curve = [{"step": r["step"], "D": r["D"], "gm12": r["gm12"]}
                   for r in opt1bm["fact_vs_D_curve"]]
    dyn_kill = opt1cm["adjudication"]["stop"]
    log(f"parents: opt1c {opt1cm['adjudication']['verdict']} (D_kill "
        f"{dyn_kill['D_kill_interp']:.4f}); opt1b "
        f"{opt1bm['adjudication']['verdict']}; opt1 "
        f"{opt1m['adjudication']['verdict']} — committed curves loaded "
        f"(dyn {len(dyn_curve)} ckpt + {len(dyn_densify)} densify rows, "
        f"bleed {len(bleed_curve)} rows, A0 {len(adam_a0_curve)} rows); "
        f"none rerun")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185/opt1/opt1c)
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

    # ---------------- the neutral stream (e170 VERBATIM via e185 arm C / opt1/opt1c)
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

    # ---------------- root net + gate vs e151 (opt1/opt1c's gate set verbatim)
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
    n_anc = anchor_neutral.shape[0]

    def draw_step1_batch():
        """The step-1 batch, bit-identical to the licensed stream (a fresh
        Generator(10902) with opt1's aj-then-rj draw order)."""
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
    # THE T=0 BIT-IDENTITY GATE (opt1c's VERBATIM) + THE DIRECTION GATE G_DIR
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
    disp1 = float(torch.norm(flat_params(tw) - theta0))
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
        "note": "PRE-DISPATCH CHECK (Rule 12; opt1c's gate VERBATIM): the "
                "step-1 batch md5 vs e185's stored hash; the forward CE and "
                "pre-clip grad norm vs opt1/opt1c's committed values; ONE "
                "fresh AdamW-recipe step from the root reproducing A0's "
                "committed step-1 L2 — the gradient provenance gate",
    }
    log(f"G_T0 (t=0 bit-identity vs opt1/opt1c committed): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; "
        f"|dCE| {d_ce:.2e} |dgn| {d_gn:.2e} |dDisp| {d_disp:.2e}: "
        + ("PASS" if G_T0["pass"] else "FAIL")
        + (" (bit)" if G_T0["bit"] else ""))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 bit-identity gate FAILED — the gradient "
                           "provenance is untrustworthy; abort (control "
                           "failure)")

    # ---- THE DIRECTION u: fresh gradient at the root on the step-1 batch
    gnet = copy.deepcopy(net0)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1)
    loss_g = F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                             y1.reshape(-1))
    assert abs(float(loss_g.item()) - ce1) < 1e-9, "CE drifted between gates"
    loss_g.backward()
    g0 = torch.cat([p.grad.detach().reshape(-1) for p in gnet.parameters()])
    postclip_gnorm = float(torch.norm(g0))
    u = (g0 / torch.norm(g0)).clone()
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g, loss_g, g0
    u_md5 = hashlib.md5(u.numpy().tobytes()).hexdigest()
    log(f"direction u = g_0/||g_0||_2 recomputed: ||g_postclip|| "
        f"{postclip_gnorm:.16f} (pre-clip {gn1:.16f}; clip "
        f"{'ACTIVE' if gn1 > 1.0 else 'inactive — positive rescale is the identity here'})")

    # ---- G_DIR: the static direction vs opt1c's committed step-1 endpoint
    dyn1 = load_cpu(OPT1C_S1_CK)
    theta1_dyn = flat_params(dyn1)
    delta_dyn = theta1_dyn - theta0
    ndyn = float(torch.norm(delta_dyn))
    # fp64 cosine: the fp32 dot over 2.7M elements reads cos = 1.0002 on this
    # build (reduction-order error, verified in the smoke debug: fp64 cos =
    # 0.9999999999992, ||dn-un|| 1.2e-5) — upcast for a truthful estimator.
    d64 = delta_dyn.to(torch.float64)
    u64 = (-u).to(torch.float64)
    cos_dir = float(torch.dot(d64, u64)
                    / (torch.norm(d64) * torch.norm(u64)))
    dir_dev = abs(1.0 - cos_dir)
    d_disp_dyn = abs(ndyn - DYN_S1_DISP_COMMITTED)
    G_DIR = {
        "cos_delta_dyn_neg_u": cos_dir, "one_minus_cos": dir_dev,
        "disp_dyn_measured": ndyn,
        "disp_dyn_committed": DYN_S1_DISP_COMMITTED,
        "disp_diff_vs_committed": d_disp_dyn,
        "cos_tol_bit": DIR_COS_TOL, "cos_tol_hard": DIR_COS_HARD,
        "disp_tol": G_BIT_TOL,
        "bit": bool(dir_dev < DIR_COS_TOL and d_disp_dyn < G_BIT_TOL),
        "pass": bool(dir_dev < DIR_COS_HARD and d_disp_dyn < G_FALLBACK_TOL),
        "note": "PRE-DISPATCH CHECK (Rule 12, this cell's own geometry): the "
                "recomputed unit direction u must BE the committed dynamic "
                "step's displacement direction — cos(theta_1-theta_0, -u) = 1 "
                "and ||theta_1-theta_0|| = opt1c's committed cumulative "
                "displacement; without it the static map is not mapping the "
                "dynamic step's own ray",
    }
    log(f"G_DIR (direction vs opt1c step-1 checkpoint): cos "
        f"{cos_dir:.10f} (1-cos {dir_dev:.2e}); |dDisp| {d_disp_dyn:.2e}: "
        + ("PASS" if G_DIR["pass"] else "FAIL")
        + (" (bit)" if G_DIR["bit"] else ""))
    if not G_DIR["pass"]:
        raise RuntimeError("direction gate FAILED — the recomputed u is not "
                           "the committed dynamic step's direction")
    STEP_SCALE = float(disp1)          # the MEASURED value (documented; opt1c's convention)
    log("WHAT THIS MAP GUARANTEES: NOTHING — the static profile could be "
        "flat, ridge-then-cliff, or monotone; that openness is the point.")

    # ---- PROGRESSIVE PARTIAL WRITE #1 (the outage lesson)
    save_json(rd / "metrics.json", E43.jsonable({
        "experiment": "e191_pump_cliff", "date": common.now_iso(),
        "status": "PARTIAL — all pre-dispatch gates PASSED (t=0 bit + "
                  "direction), profile starting; no jumps yet",
        "partial": True,
        "phase": {"jumps_done": 0, "elapsed_s": round(time.time() - T0, 1)},
        "gates_partial": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                          "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                          "G_ROOT": G_ROOT, "G_T0": G_T0, "G_DIR": G_DIR},
        "provenance_partial": {
            "step_scale_measured": STEP_SCALE,
            "dyn_endpoint_committed": DYN_S1_DISP_COMMITTED,
            "direction_md5": u_md5,
            "postclip_gnorm": postclip_gnorm},
    }))
    log("[partial] metrics.json written (gates passed, profile starting)")

    # =====================================================================
    # THE GRADED STATIC PROFILE (perturb-and-eval; progressive writes)
    # =====================================================================
    evl = copy.deepcopy(net0)
    evl.eval()
    profile: list[dict] = []          # the graded grid rows
    crosscheck: list[dict] = []       # verification rows at committed Ds
    journal: list[dict] = []

    def static_jump_read(D: float) -> dict:
        """theta_D = theta_0 - D*u; perturb-and-eval; returns the read row."""
        thD = theta0 - D * u
        disp_check = float(torch.norm(thD - theta0))
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, gm12_ids, zid)
        ce_r = ce_fixed_cpu(evl, *r_eval_xy)
        return {"D": float(D), "gm12": gz["mean_pz"],
                "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce_r,
                "disp_check": disp_check,
                "disp_dev": abs(disp_check - float(D))}

    def write_partial(status: str) -> None:
        save_json(rd / "metrics.json", E43.jsonable({
            "experiment": "e191_pump_cliff", "date": common.now_iso(),
            "status": status, "partial": True,
            "phase": {"grid_done": len(profile),
                      "grid_total": len(D_GRID),
                      "crosscheck_done": len(crosscheck),
                      "elapsed_s": round(time.time() - T0, 1)},
            "gates_partial": {"G_T0": G_T0, "G_DIR": G_DIR},
            "provenance_partial": {
                "step_scale_measured": STEP_SCALE,
                "direction_md5": u_md5},
            "profile_partial": profile, "crosscheck_partial": crosscheck,
        }))

    # the root row (D = 0)
    root_row = {"D": 0.0, "gm12": root_cells["gm12"],
                "frac_argmax_z": None, "ce_r": root_cells["ce_r"],
                "disp_check": 0.0, "disp_dev": 0.0}

    for D in D_GRID:
        row = static_jump_read(D)
        profile.append(row)
        journal.append({"kind": "grid", **row})
        log(f"  STATIC D {D:5.3f}: g-12 {row['gm12']:.4f} "
            f"(argmax {row['frac_argmax_z']:.2f}) CE_R {row['ce_r']:.4f} "
            f"[disp dev {row['disp_dev']:.1e}]")
        write_partial(f"PARTIAL: grid {len(profile)}/{len(D_GRID)} jumps "
                      f"done; crosscheck pending")

    # ---- the crosscheck arm (verification only; committed exact points)
    xc_specs = ([{"D": r["D"], "committed_gm12": r["gm12"], "src": "opt1c densify"}
                for r in dyn_densify]
                + [{"D": DYN_S1_DISP_COMMITTED,
                    "committed_gm12": dyn_ck[1]["gm12"],
                    "committed_ce_r": dyn_ck[1]["ce_r"],
                    "src": "opt1c step-1 endpoint"}])
    if SMOKE:
        xc_specs = xc_specs[-1:]
    for spec in xc_specs:
        row = static_jump_read(spec["D"])
        d_gm = abs(row["gm12"] - spec["committed_gm12"])
        d_ce = (abs(row["ce_r"] - spec["committed_ce_r"])
                if "committed_ce_r" in spec else None)
        crosscheck.append({**row, "committed_gm12": spec["committed_gm12"],
                           "committed_ce_r": spec.get("committed_ce_r"),
                           "d_gm12": d_gm, "d_ce_r": d_ce,
                           "src": spec["src"], "bit": bool(d_gm < 1e-4)})
        journal.append({"kind": "crosscheck", **crosscheck[-1]})
        log(f"  XCHK  D {spec['D']:.4f} ({spec['src']}): g-12 "
            f"{row['gm12']:.6f} vs committed {spec['committed_gm12']:.6f} "
            f"(|d| {d_gm:.2e})")
    xc_max = max(c["d_gm12"] for c in crosscheck)
    G_XCHK = {"n_points": len(crosscheck), "max_abs_d_gm12": xc_max,
              "bit_tol": 1e-4,
              "pass": bool(xc_max < 1e-3),
              "note": "verification arm: the fresh static code path "
                      "reproduces opt1c's committed dynamic-path reads "
                      "(densify points + step-1 endpoint) — machinery "
                      "cross-validation, never an adjudication input"}
    log(f"G_XCHK: {len(crosscheck)} committed points reproduced, max|dgm12| "
        f"{xc_max:.2e}: " + ("PASS" if G_XCHK["pass"] else "FAIL"))

    # ---- checkpoint the direction (rider/follow-up provenance)
    u_ckpt = ("smoke_" if SMOKE else "") + "e191_static_dir_u"
    torch.save({"u": u.detach().clone(), "theta0_md5": flat_md5(net0),
                "meta": {"experiment": "e191", "direction": "g_0/||g_0||_2",
                         "step_scale": STEP_SCALE,
                         "input_seed": FREEZE_SEED,
                         "base": f"runs/checkpoints/{ROOT_CK}",
                         "parent": "runs/opt1c/metrics.json"}},
               CKPT_DIR / f"{u_ckpt}.pt")
    log(f"[ckpt] saved {u_ckpt}.pt (the direction vector)")

    # =====================================================================
    # ADJUDICATION (registered clauses; composite order frozen:
    # STATIC-CLIFF -> STATIC-SPARES -> GRADED-MAP; no shopping)
    # =====================================================================
    gm = {r["D"]: r["gm12"] for r in profile}
    all_rows = [root_row] + profile
    pump_val = gm.get(PUMP_D)
    pump = bool(pump_val is not None and pump_val > root_cells["gm12"])
    dead_in_band = [d for d in KILL_BAND_GRID
                    if d in gm and gm[d] <= SHUT_BAR]
    spares_val = gm.get(SPARES_D)
    static_spares_fires = bool(spares_val is not None
                               and spares_val > SHUT_BAR)
    static_cliff_fires = bool(pump and dead_in_band)

    # the cliff-edge bracket: tightest adjacent alive -> dead grid pair
    alive_dead_pairs = []
    for a, b in zip(all_rows, all_rows[1:]):
        if a["gm12"] > SHUT_BAR >= b["gm12"]:
            alive_dead_pairs.append((a["D"], b["D"]))
    cliff_edge = {"bracket_D": list(alive_dead_pairs[0])
                  if alive_dead_pairs else None,
                  "first_dead_grid_D": (alive_dead_pairs[0][1]
                                        if alive_dead_pairs else None),
                  "n_alive_dead_pairs": len(alive_dead_pairs)}

    overlap = static_cliff_fires and static_spares_fires
    if SMOKE:
        verdict, clause = "SMOKE (nothing adjudicated)", "shakedown only"
        bars = {k: {"fires": False} for k in
                ("STATIC_CLIFF", "STATIC_SPARES", "GRADED_MAP")}
    elif static_cliff_fires:
        verdict = "STATIC-CLIFF (TERRAIN)"
        clause = (f"the static profile MATCHES the dynamic cliff: the fact "
                  f"PUMPS to gm12 {pump_val:.4f} at D {PUMP_D} (root "
                  f"{root_cells['gm12']:.4f}) and falls below {SHUT_BAR} by "
                  f"D {dead_in_band[0]:.2f} (band reads: "
                  + ", ".join(f"D {d}: {gm[d]:.4f}" for d in KILL_BAND_GRID)
                  + f"); cliff-edge bracket D {cliff_edge['bracket_D']}. "
                  f"THE PUMP-CLIFF IS TERRAIN (geometry); the direction's "
                  f"lethality is a static property; e188's cross-class "
                  f"break is a geometry fact. Scope as disclosed: within "
                  f"[0, {DYN_S1_DISP_COMMITTED:.4f}] the dynamic path WAS "
                  f"the straight ray (single-step kill), so static-at-D and "
                  f"dynamic-at-D coincide BY CONSTRUCTION there — this map's "
                  f"independent content is the fresh recompute (XCHK "
                  f"max|d| {xc_max:.1e}), the D 2.0 extension "
                  f"(gm12 {gm.get(2.0, float('nan')):.4f} there), the "
                  f"tighter cliff edge, and the CE_R co-read; no "
                  f"trajectory effect is NEEDED to explain the kill."
                  + (f" OVERLAP REPORTED VERBATIM: STATIC-SPARES' letter "
                     f"also holds (gm12 at D {SPARES_D} = "
                     f"{spares_val:.4f} > {SHUT_BAR}) — the cliff edge sits "
                     f"in ({SPARES_D}, {dead_in_band[0]}]; composite order "
                     f"gives the verdict to STATIC-CLIFF." if overlap else ""))
        bars = {"STATIC_CLIFF": {"fires": True, "pump": pump,
                                 "pump_gm12": pump_val,
                                 "dead_in_band_D": dead_in_band,
                                 "band_reads": {str(d): gm[d]
                                                for d in KILL_BAND_GRID},
                                 "cliff_edge": cliff_edge,
                                 "overlap_with_spares": overlap},
                "STATIC_SPARES": {"fires": static_spares_fires,
                                  "gm12_at_092": spares_val},
                "GRADED_MAP": {"fires": False}}
    elif static_spares_fires:
        verdict = "STATIC-SPARES (DYNAMICS)"
        clause = (f"the static jump at D {SPARES_D} leaves the fact ALIVE "
                  f"(gm12 {spares_val:.4f} > {SHUT_BAR}); "
                  f"{'the PUMP clause failed (no rise at D 0.33: ' + format(pump_val, '.4f') + ' vs root ' + format(root_cells['gm12'], '.4f') + ') and ' if not pump else ''}"
                  f"the kill is DYNAMIC overshoot — the single big step "
                  f"passes THROUGH the cliff into worse territory a static "
                  f"endpoint-jump never visits; trajectory effects carry "
                  f"lethality that geometry alone does not. The full graded "
                  f"profile is the map (reported verbatim).")
        bars = {"STATIC_CLIFF": {"fires": False, "pump": pump,
                                 "pump_gm12": pump_val,
                                 "dead_in_band_D": dead_in_band},
                "STATIC_SPARES": {"fires": True, "gm12_at_092": spares_val,
                                  "cliff_edge": cliff_edge},
                "GRADED_MAP": {"fires": False}}
    else:
        verdict = "GRADED-MAP"
        clause = (f"a partial pattern — neither first bar's full clause-set "
                  f"holds (pump {'YES' if pump else 'NO'} "
                  f"(gm12 {pump_val:.4f} at D {PUMP_D} vs root "
                  f"{root_cells['gm12']:.4f}); dead-in-band "
                  f"{dead_in_band or 'none'}; gm12 at D {SPARES_D} = "
                  f"{spares_val}); the full profile is reported verbatim as "
                  f"the map; the terrain/dynamics split stated as measured.")
        bars = {"STATIC_CLIFF": {"fires": False, "pump": pump,
                                 "dead_in_band_D": dead_in_band},
                "STATIC_SPARES": {"fires": False,
                                  "gm12_at_092": spares_val},
                "GRADED_MAP": {"fires": True, "cliff_edge": cliff_edge}}
    log("=" * 78)
    log(f"E191 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e191_pump_cliff",
        "date": common.now_iso(),
        "status": "COMPLETE — adjudicated (this write replaces all PARTIAL "
                  "progressive writes)",
        "registration": ("the dispatch's registration IS the registration "
                         "(the three bars quoted verbatim in the module "
                         "docstring and in registered_prediction, frozen "
                         "before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the pump-cliff TERRAIN (a static property of the "
                     "displacement region — a ridge then a cliff in the "
                     "g-direction, present under a single static jump) or "
                     "DYNAMICS (the big step overshoots THROUGH the cliff "
                     "into worse territory a static jump at the same "
                     "endpoint would not reach)? STATIC single jumps "
                     "theta_0 - D*u at graded D along the unit raw "
                     "wash-batch gradient at the root (opt1c's t=0 "
                     "convention), overlay vs opt1c's dynamic path and "
                     "opt1b's bleed (committed)"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cell": {
            "licensed_cell": "e185 arm C (CONTROL) VERBATIM as the gradient "
                             "source — the consolidated host fact under its "
                             "neutral/corpus wash; opt1c's machinery verbatim",
            "batch": f"32 = {ANCH_BS} neutral-bank anchors (seed "
                     f"{E170_ANCHOR_SEED}, fixed) + {RAND_BS} random corpus "
                     "windows, full-token CE, clip 1.0 (the step-1 batch "
                     "only — the t=0 gradient)",
            "static_jump": "theta_D = theta_0 - D * u, u = g_0/||g_0||_2 "
                           "(post-clip g_0; clip-invariant direction; "
                           "DESCENT sign, opt1c's convention); "
                           "perturb-and-eval; NO optimizer, NO steps",
            "D_grid": list(D_GRID),
            "input_seed": FREEZE_SEED,
            "measure_light": "fact g-12 battery + frac argmax Z + CE_R at "
                             "every D (the dispatch's read list + the free "
                             "argmax; no census/deletions/alignment — the "
                             "static delta is along u identically)",
        },
        "provenance": {
            "step_scale_measured": STEP_SCALE,
            "step_scale_committed_opt1_a0": STEP_SCALE_COMMITTED,
            "dyn_endpoint_committed": DYN_S1_DISP_COMMITTED,
            "direction_md5": u_md5,
            "postclip_gnorm": postclip_gnorm,
            "t0_gate": G_T0,
            "direction_gate": G_DIR,
            "crosscheck_gate": G_XCHK,
            "progressive_writes": ("metrics.json was written progressively "
                                   "(after the gates, after every profile "
                                   "point) per the opt1c outage lesson; "
                                   "intermediate PARTIAL writes are "
                                   "replaced by this COMPLETE write"),
            "committed_data_reuse": ("opt1c's dynamic curve + densify rows, "
                                     "opt1b's bleed and opt1's Adam-A0 are "
                                     "loaded from runs/opt1c/metrics.json + "
                                     "runs/opt1b/metrics.json + runs/opt1/"
                                     "metrics.json verbatim and plotted, "
                                     "never rerun"),
        },
        "profile": [root_row] + profile,
        "crosscheck": crosscheck,
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT, "G_T0": G_T0, "G_DIR": G_DIR,
                  "G_XCHK": G_XCHK},
        "references": {
            "opt1c": {"metrics": "runs/opt1c/metrics.json",
                      "role": "the parent dynamic cell: the t=0 gradient "
                              "convention, the committed along-path densify "
                              "rows + step-1 endpoint (the crosscheck "
                              "targets), the dynamic curve (the overlay), "
                              "and T143's open split this cell adjudicates"},
            "opt1b": {"metrics": "runs/opt1b/metrics.json",
                      "role": "the bleed (the raw-gradient path at its own "
                              "small size): the sampled-point overlay from "
                              "committed data"},
            "opt1": {"metrics": "runs/opt1/metrics.json",
                     "role": "the Adam arms' fact-vs-D curves (A0 overlaid "
                             "for context) + the t=0 committed row"},
            "e185": {"metrics": "runs/e185/metrics.json",
                     "role": "the licensed cell + the stored input md5s "
                             "re-derived at step 1"},
        },
        "overlays": {
            "dynamic_curve": dyn_curve,
            "dynamic_densify": dyn_densify,
            "dynamic_kill": {k: dyn_kill[k] for k in
                             ("D_kill_interp", "D_kill_interp_raw", "t_x",
                              "bracket_f", "D_bracket")},
            "bleed_curve": bleed_curve,
            "adam_a0_curve": adam_a0_curve,
        },
        "adjudication": {
            "bars": bars,
            "verdict": verdict, "clause": clause,
            "cliff_edge": cliff_edge,
            "constants": {"shut_bar": SHUT_BAR, "pump_D": PUMP_D,
                          "spares_D": SPARES_D,
                          "kill_band_grid": list(KILL_BAND_GRID),
                          "root_gm12": root_cells["gm12"],
                          "dyn_D_kill_interp":
                              dyn_kill["D_kill_interp"],
                          "dyn_densify_Ds": [r["D"] for r in dyn_densify],
                          "dyn_endpoint": DYN_S1_DISP_COMMITTED},
            "composite_order": "STATIC-CLIFF -> STATIC-SPARES -> GRADED-MAP "
                               "(frozen before compute; a co-hold gives the "
                               "verdict to STATIC-CLIFF with the overlap "
                               "reported verbatim)",
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 direction-deterministic map on ONE organism "
                            "(the e131 consolidated root, seed lineage "
                            "10902); no seed sweep — the direction u is a "
                            "deterministic function of the bit-gated root + "
                            "batch, and every read of theta_D is "
                            "deterministic (documented per dispatch); the "
                            "honest replicate axis is a second ORGANISM, a "
                            "separate dispatch decision"),
            "on_span_coincidence": ("DISCLOSED (stated before compute): "
                                    "opt1c's dynamic path died INSIDE its "
                                    "first step, so on [0, "
                                    f"{DYN_S1_DISP_COMMITTED:.4f}] the "
                                    "dynamic trajectory IS the straight ray "
                                    "theta_0 - D*u — static-at-D and "
                                    "dynamic-at-D are the same parameter "
                                    "point up to float noise; terrain-vs-"
                                    "dynamics is NOT separable by "
                                    "construction within the span. This "
                                    "map's independent content: the fresh "
                                    "recompute (crosscheck max|dgm12| "
                                    f"{xc_max:.1e} over {len(crosscheck)} "
                                    "committed points), the D = 2.0 "
                                    "extension BEYOND the dynamic endpoint, "
                                    "the 0.8/0.92/1.0 cliff-edge resolution "
                                    "between the committed 0.66/0.99 "
                                    "bracket, and the CE_R co-read"),
            "extrapolation_free": ("every adjudication input is a MEASURED "
                                   "read at a graded D (the pump, the band "
                                   "reads, the 0.92 read); no interpolation "
                                   "is used anywhere in the clauses; the "
                                   "cliff-edge bracket is reported as a "
                                   "bracket, not a point"),
            "float_texture": ("CPU fp32, this process, 4 threads (the "
                              "e185/opt1/opt1b/opt1c reduction order); the "
                              "t=0 gate and the direction gate reproduce "
                              "opt1c's committed values on THIS device "
                              "before any jump (Rule 12); the parents' "
                              "curves are loaded from committed metrics, "
                              "never rerun"),
            "shape_resemblance": ("the static g-direction profile's shape "
                                  "(reported verbatim in .profile) vs the "
                                  "store's earlier flat-vs-cone forms "
                                  "(g3R/g3K conventions): noted as a "
                                  "pointer only — any resemblance is "
                                  "stated, none claimed (the dispatch's "
                                  "no-claims clause); this profile's own "
                                  "shape is a ridge-then-cliff along ONE "
                                  "direction, a different object than a "
                                  "store-type decay curve"),
            "no_guarantees": ("nothing was guaranteed ex ante: the static "
                              "profile could be flat, ridge-then-cliff, or "
                              "monotone — the openness is the point; the "
                              f"verdict actually observed is '{verdict}'"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": {
            u_ckpt: {"path": f"runs/checkpoints/{u_ckpt}.pt",
                     "what": "the unit direction vector u (fp32 flat, "
                             "2,739,072) + theta0_md5 + meta — for any "
                             "rider/follow-up cell"},
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu", "torch_threads": 4,
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    with open(rd / "journal.jsonl", "w", encoding="utf-8") as f:
        for r in journal:
            f.write(json.dumps(r) + "\n")

    plot_pump_cliff(rd / "e191_pump_cliff.png", profile, crosscheck,
                    dyn_curve, dyn_densify, dyn_kill, bleed_curve,
                    adam_a0_curve, root_row, verdict, clause)
    plot_coreads(rd / "e191_coreads.png", profile, crosscheck, dyn_curve,
                 root_row, xc_max)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'e191_pump_cliff.png'}, "
        f"{rd / 'e191_coreads.png'}, journal.jsonl, runs/checkpoints/"
        f"{u_ckpt}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_pump_cliff(path, profile, crosscheck, dyn_curve, dyn_densify,
                    dyn_kill, bleed_curve, adam_a0_curve, root_row,
                    verdict, clause):
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.8),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    # the bleed (opt1b committed)
    ax.plot([r["D"] for r in bleed_curve], [r["gm12"] for r in bleed_curve],
            "o-", ms=4.0, lw=1.6, color="royalblue", alpha=0.85,
            label="THE BLEED — SGD 1e-2, ~0.0066/step (opt1b committed)")
    # Adam A0 (opt1 committed, context)
    ax.plot([r["D"] for r in adam_a0_curve],
            [r["gm12"] for r in adam_a0_curve], "s--", ms=5, lw=1.4,
            color="crimson", alpha=0.8,
            label="Adam A0 1e-3 sign-normalized (opt1 committed)")
    # the dynamic path (opt1c committed: ckpt + densify)
    ax.plot([r["D"] for r in dyn_curve], [r["gm12"] for r in dyn_curve],
            "^-", ms=8, lw=2.0, color="magenta", alpha=0.95, zorder=4,
            label="DYNAMIC — g-direction at 1.6543/step (opt1c committed)")
    ax.plot([r["D"] for r in dyn_densify],
            [r["gm12"] for r in dyn_densify], "x", ms=8, mew=2.2,
            color="darkmagenta", alpha=0.9, zorder=5,
            label="dynamic along-path densify (opt1c committed)")
    # THE STATIC PROFILE (this run)
    ax.plot([r["D"] for r in profile], [r["gm12"] for r in profile], "D-",
            ms=9, lw=2.4, color="black", alpha=0.95, zorder=6,
            label="STATIC single jumps theta_0 - D*u (e191, this run)")
    # the crosscheck points
    ax.plot([r["D"] for r in crosscheck], [r["gm12"] for r in crosscheck],
            "o", ms=11, mfc="none", mec="seagreen", mew=2.0, alpha=0.95,
            zorder=7,
            label="XCHK — exact committed Ds reproduced (verification)")
    # reference furniture
    ax.axvspan(0.92, 1.17, color="gold", alpha=0.13,
               label="STATIC-CLIFF kill band D 0.92-1.17 (registered)")
    ax.axvline(0.33, ls=":", lw=1.6, color="darkgreen", alpha=0.8,
               label="pump clause D 0.33")
    for yv, col, lbl in ((SHUT_BAR, "tab:purple", "0.27 DISSOLVE"),
                         (root_row["gm12"], "gray",
                          f"root g-12 {root_row['gm12']:.3f}")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.plot([dyn_kill["D_kill_interp"]], [SHUT_BAR], "*", ms=17,
            color="yellow", mec="k", zorder=8,
            label=f"dynamic kill (interp D {dyn_kill['D_kill_interp']:.3f})")
    ax.set_xlabel(r"displacement $D$ along the unit g-direction "
                  r"($\|\theta_D-\theta_0\|_2$, 2.74M params)")
    ax.set_ylabel("g-12 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.06, max(2.1, max(r["D"] for r in profile) + 0.1))
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("THE PUMP-CLIFF MAP — static g-direction profile vs the "
                 "dynamic path vs the bleed", fontsize=10)

    ax = axes[1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, f"E191 VERDICT: {verdict}", fontsize=10.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=90, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=7.0, va="top",
                family="monospace")
        y -= 0.024
    y -= 0.015
    ax.text(0.02, y, "  kind      D      g-12   argmax   CE_R", fontsize=7.2,
            va="top", family="monospace")
    y -= 0.026
    for r in [ {"kind": "root", **root_row} ] + \
            [{"kind": "grid", **r} for r in profile] + \
            [{"kind": "xchk", **r} for r in crosscheck]:
        ax.text(0.02, y,
                f"  {r['kind']:>5s}  {r['D']:7.4f}  {r['gm12']:.4f}  "
                + (f"{r['frac_argmax_z']:.2f}   " if r.get("frac_argmax_z")
                   is not None else " n/a   ")
                + f"{r['ce_r']:.4f}",
                fontsize=6.8, va="top", family="monospace",
                color=("seagreen" if r["kind"] == "xchk" else "black"))
        y -= 0.024
    fig.suptitle("E191 — THE PUMP-CLIFF MAP: terrain (a static ridge then "
                 "cliff in the g-direction) or dynamics (the big step "
                 "overshoots through it)?", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coreads(path, profile, crosscheck, dyn_curve, root_row, xc_max):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    # (0,0) CE_R: static vs dynamic — does the organism wreck in step with
    # the fact, or after?
    ax = axes[0, 0]
    ax.plot([r["D"] for r in profile], [r["ce_r"] for r in profile], "D-",
            ms=8, lw=2.2, color="black",
            label="STATIC profile CE_R (e191)")
    drows = [r for r in dyn_curve if r["step"] > 0]
    ax.plot([r["D"] for r in drows], [r["ce_r"] for r in drows], "^--",
            ms=9, lw=1.8, color="magenta",
            label="DYNAMIC step-1 CE_R (opt1c committed)")
    ax.axhline(root_row["ce_r"], ls="--", lw=1.2, color="gray",
               label=f"root CE_R {root_row['ce_r']:.3f}")
    for r in profile:  # where does the FACT die on the same axis?
        if r["gm12"] <= SHUT_BAR:
            ax.axvline(r["D"], ls=":", lw=1.0, color="tab:purple", alpha=0.6)
    ax.set_xlabel("D along the unit g-direction")
    ax.set_ylabel("CE_R (organism)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("CO-READ 1 — organism health along the static profile "
                 "(dotted vlines = grid Ds where g-12 <= 0.27)", fontsize=9.5)

    # (0,1) the crosscheck deltas (machinery verification)
    ax = axes[0, 1]
    ax.plot([r["D"] for r in crosscheck], [r["d_gm12"] for r in crosscheck],
            "o-", ms=8, lw=1.8, color="seagreen")
    ax.axhline(1e-4, ls="--", lw=1.2, color="crimson",
               label="bit tol 1e-4")
    ax.set_yscale("log")
    ax.set_xlabel("D (the exact committed dynamic points)")
    ax.set_ylabel("|gm12_static - gm12_committed|")
    ax.legend(fontsize=7.5)
    ax.set_title(f"CO-READ 2 — the crosscheck arm: fresh code path vs "
                 f"committed reads (max {xc_max:.1e})", fontsize=9.5)

    # (1,0) frac argmax Z vs D
    ax = axes[1, 0]
    rows = [r for r in profile if r["frac_argmax_z"] is not None]
    ax.plot([r["D"] for r in rows], [r["frac_argmax_z"] for r in rows],
            "D-", ms=8, lw=2.2, color="darkcyan",
            label="frac argmax Z (same battery call; free)")
    ax.plot([r["D"] for r in rows], [r["gm12"] for r in rows], "o--",
            ms=5, lw=1.4, color="gray", alpha=0.8, label="g-12 (mean p(Z))")
    ax.axhline(SHUT_BAR, ls="--", lw=1.0, color="tab:purple", alpha=0.7)
    ax.set_xlabel("D along the unit g-direction")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("CO-READ 3 — the argmax read rides with the probability "
                 "read", fontsize=9.5)

    # (1,1) displacement verification (machinery)
    ax = axes[1, 1]
    ax.plot([r["D"] for r in profile], [r["disp_dev"] for r in profile],
            "D-", ms=7, lw=1.8, color="slateblue")
    ax.axhline(1e-5, ls="--", lw=1.2, color="crimson",
               label="tol 1e-5")
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$|\,\|\theta_D-\theta_0\|_2 - D\,|$")
    ax.legend(fontsize=7.5)
    ax.set_title("MACHINERY — every jump lands at its graded D "
                 "(perturb-and-eval placement)", fontsize=9.5)

    fig.suptitle("E191 — co-reads: the organism along the cliff, the "
                 "crosscheck, the argmax rider, placement", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
