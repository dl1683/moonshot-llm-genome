"""E207 — THE LAG-2 RUNG SET (the null's last debt).

WHY (scratch/sign_front_null.md sec.4 + runs/e202/metrics.json + THINKING T168):
e202's in-domain step ladder {s/8, s/4, s/2} on the FACT-CARRYING org2 root
CONFIRMED the null's cos1 law (orthogonal at s/8 -> -0.26318 at s/2,
bit-anchored) but the CORE statistic — core_t = (cos2(u0,u2) -
cos1(u0,u1))/2 at t=0, the overshoot picture's registered anti-phase core —
BROKE at the half rung: 0.13387 (s/8) -> 0.23319 (s/4) -> 0.20186 (s/2),
non-monotone exactly where the null demanded monotone growth (T168: "the
overshoot picture's lag-2 structure is incomplete"). ONE RUNG of grain sits
between the quarter and the half: the break is either REAL (the null's
lag-2 term genuinely incomplete — the derivation's debt stands as a
finding) or COARSE (the statistic wobbles rung-to-rung and e202's break
was grain). This cell fills the missing interior at the same grain the
ladder already has.

THE CELL (CPU-only, threads 4, tiny sequential bursts; e202's machinery
VERBATIM — the e201 u3-burst class):
  - THE RUNG SET {s/4, 3s/8} x s = 0.9164195656776428 on the SAME org2 root
    (e157_f2_consolidated, gated), the SAME licensed seed-10902 stream
    (md5-gated vs e185's stored hashes), the same walk arithmetic (k=1
    post-clip refresh, per-step L2 asserted exact, u_t =
    sign(g_t)/||sign(g_t)||, e192's fp32-norm construction), 5 steps per
    rung with e202's kill conventions (own g-4 ruler > 0.27; +1 post-kill
    row if a rung dies).
  - s/4 is the ANCHOR REPLICATION: e202's committed quarter rung REBUILT
    and gated row-by-row + ray-md5 (G_WALKE202) — the e202 precedent (its
    s/2 rung anchored vs e197) transposed one rung down; the 3s/8 rung is
    not believed until the s/4 anchor passes. 3s/8 is the NEW interior
    rung (parents: the gated root + the gated stream + the registered step
    arithmetic); its u0 must md5 == e193's committed R2_SIGN (every rung
    shares the root's step-1 front).
  - THE EXISTING RUNGS {s/8, s/2} are LOADED COMMITTED from
    runs/e202/metrics.json (hard-bound constants, cross-gated) — never
    recomputed. The FIVE-RUNG TABLE s/8..s/2 = the four in-domain rungs
    {s/8, s/4, 3s/8, s/2} + the OUT-OF-DOMAIN natural-s context row
    (org2-dead-full, 1.74x its own 0.5252 static kill edge — the
    derivation's own exclusion; e202's convention carried).

REGISTERED BARS (frozen here, before compute; the dispatch's registration
VERBATIM; no bar shopping — adjudicate against exactly this):
  - CORE-BREAK-REAL: "fires if the core statistic is smooth/montone
    through {s/4, 3s/8} and breaks only at s/2 — the null's lag-2
    structure is genuinely incomplete; the derivation's debt stands as a
    finding."
  - CORE-GRAINY: "fires if the core wobbles non-monotonically across the
    interior rungs — the statistic is grain; the null's cos1 law (already
    confirmed) is the whole in-domain story; the lag-2 term retires."
  - GRADED: "any partial — the table verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * core_t = (cos2(u0,u2) - cos1(u0,u1))/2 at t=0 per rung — e202's
    registered statistic VERBATIM; cosines fp64; fronts at ALIVE ts only
    (own g-4 ruler > 0.27 — e202's convention; a rung dying before t=2
    contributes no in-domain lag-2 pair and its core is unreadable).
  * the adjudication series (ascending s) = {c18 = e202's committed
    eighth core (LOADED), c14 = THIS CELL's fresh s/4 core (anchored),
    c38 = THIS CELL's fresh 3s/8 core (new), c12 = e202's committed half
    core (LOADED)}; the anchor gate requires |fresh s/4 core - e202's
    committed quarter core| <= 1e-3 (tier-stamped) before anything
    adjudicates; both fresh/committed values reported in the table.
  * CORE-BREAK-REAL fires iff the 3s/8 core is READABLE and the series is
    non-decreasing through the interior — c18 <= c14 <= c38 (tol 1e-12) —
    AND the break lands ONLY at the half rung — c38 > c12 (tol 1e-12).
  * CORE-GRAINY fires iff the 3s/8 core is READABLE and the interior
    turns down — c38 < c14 (beyond the 1e-12 tol): the non-monotone
    wobble across the interior rungs; sub-flagged STRONG if additionally
    c38 <= c12 (the interior already at/below the half's collapsed value).
  * GRADED fires otherwise — any partial, the table verbatim: the 3s/8
    core unreadable (the rung died before t=2), or an anchor-tier
    ambiguity that moves the pivot (fresh-vs-committed quarter dev in
    (1e-3, ...] would have aborted at the gate; a pass at exactly the
    tier boundary is disclosed here).
  * composite order frozen: CORE-BREAK-REAL -> CORE-GRAINY -> GRADED
    (first that fires is the verdict; every bar's fires flag reported).
  * the natural-s point (org2-dead-full, cos1 -0.16531, e196 via e197)
    rides as OUT-OF-DOMAIN context only; its core is undefined-by-
    exclusion (no committed in-domain lag-2 read at 1.74x the kill edge).
  * the full lag-1/lag-2 series + the PERIOD-2 FINGERPRINT (cos1 < 0,
    cos2 > 0, |cos3| <= 0.05 per alive t) are reported per rung as
    CONTEXT (e202's convention).

REGISTERED PREDICTION (frozen before compute): the null's own step-size
law (scratch sec.4: the flip zone |mu| <= h*s widens with s, so the core
must be NON-DECREASING in s, in-domain) predicts the 3s/8 core CONTINUES
the committed rise (>= the quarter's 0.23319); the dispatch's
CORE-BREAK-REAL bar fires exactly when that continuation holds while the
committed half-rung collapse stands (c38 > c12) — the null's cos1 law
confirmed in-domain with its lag-2 term breaking at the half. Nothing is
guaranteed; the openness is the point.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication):
  G_FILES (the five parent metrics exist, correct experiment names,
  COMPLETE status where the convention applies, pulled keys present) +
  G_CROSSFILE (the committed comparators cross-consistent: e193's step L2,
  e197's sub-s, e157's dial, e200's lag matrix + ray md5s, AND e202's
  ladder constants hard-bound — the loaded rungs' cores/cos1/lag series
  exact) + the stream gates VERBATIM (G_NAMEFREE, G_SPLICE, G_BATTERY,
  G_ANCHOR vs e185's stored bank, G_STREAM steps 1..6 md5 vs e185's
  stored hashes) + G_ROOT (the org2 root's dial reproduces e157's
  committed cells BIT-TIGHT + the flat md5 matches e193's committed root
  identity) + THE MOVEMENT ARITHMETIC REGISTERED (the rung set {1/4, 3/8}
  x 0.9164195656776428 frozen here, before any walk; per-step L2 asserted
  EXACTLY — fp64 norm, dev < 1e-5 every step of every rung, e194's
  convention; the s/4 size asserted == e202's committed quarter size
  EXACTLY; both rungs strictly inside the 0.5252 static sign edge) +
  G_WALKE202 (the fresh s/4 rung must reproduce e202's committed quarter
  journal row-by-row, its x-hashes, its ray md5s u0..u4, and its mutual
  geometry within 1e-3 — bit-class or the disclosed cross-thread TEXTURE
  tier) + G_RAYSNEW (the 3s/8 rung's u0 md5 == e193's committed R2_SIGN).

WHAT THE RUNGS GUARANTEE: NOTHING — n=1 per rung; the walks are
DETERMINISTIC (fixed seed-10902 stream, fixed root) so the fresh s/4 is a
replication check of the machinery, not an independent draw; the 3s/8
rung is one realized path at one counterfactual size on ONE org2 root;
the openness is the point.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02, permanent): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never claimed),
torch threads CAPPED AT 4, load-check before launch, tiny sequential
bursts (two <= 6-step sign walks + battery reads — minutes), PROGRESSIVE
metrics.json writes after every phase (the outage lesson), n=1.

Outputs: runs/e207/{metrics.json, e207_lag2_rungs.png}. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e207_lag2_rungs.py    (E207_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (the envelope)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # the owner envelope's cap

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,        # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E207_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e207 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

RUNS = E43.REPO / "runs"
PARENTS = {
    "e157": RUNS / "e157" / "metrics.json",
    "e193": RUNS / "e193" / "metrics.json",
    "e197": RUNS / "e197" / "metrics.json",
    "e200": RUNS / "e200" / "metrics.json",
    "e202": RUNS / "e202" / "metrics.json",
}

# ---- the net class (org2 only — no twin arm in this cell) ---------------------
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)  # org2 class
N_PARAM2 = 873_472
ROOT2_CK = "e157_f2_consolidated.pt"           # the FACT-CARRYING org2 root (the ladder's root)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- the stream / battery conventions (e193/e194/e197/e200/e202 verbatim) -----
PRE, POST_CAP = 130, 119
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)
RULER_J2, CO_RULERS_J2 = -4, (12, -12, 0)      # the org2-class ruler (g-4 install-60)
FREEZE_SEED = 10902
ANCH_BS, RAND_BS = 16, 16
SHUT_BAR = 0.27
R_EVAL_SEED = 26502
RUNG_STEPS = 5                                 # e202's registered walk length per rung

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
    5: "4da7b67a7fd80b4e9729731fb27bec0c",
    6: "1aa4f9f250f14acad52d3b969343090a",
    7: "1e9e3373028935fe272d86683280e4b5",
    8: "3535a9db2d1aa2e6e0655208ff26b3d9",
}

# ---- org2 committed anchors (the ladder's parents; hard-bound at load) --------
E157_DIAL = {
    "gm12": 0.19826222956180573, "g-8": 0.8519253134727478,
    "g-4": 0.8872273564338684, "g0": 0.5784125924110413,
    "g+4": 0.8801683187484741, "g+8": 0.8754011988639832,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
ROOT2_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"
E193_STEP_L2 = 0.9164195656776428              # org2's measured AdamW step-1 L2 (the natural s)
E193_USIGN_MD5 = "9395918d36425e65248dbd93b6fd50bc"    # e193's committed R2_SIGN u0
E197_SUB_S = 0.4582097828388214                # e197's committed pick (== E193_STEP_L2/2 EXACTLY)
E197_U1_MD5 = "3ce920e0dde4342133f466edc27c5138"
E197_U2_MD5 = "f333e7f8c48f0d59364bfa659c2a82f2"
E200_U3_MD5 = "d556419b910ef3a96863055fe7f18051"
E200_U4_MD5 = "9818a56ef541ca0625e93d2523d82785"
ROOT2_GM = 0.8872273564338684                  # the org2 root's primary ruler read
STATIC_SIGN_EDGE = 0.5252                      # e193/e197's committed static sign-ray kill edge
DEAD_FULL_C1 = -0.1653095242890399             # e196 via e197 (org2 full step) — OUT-OF-DOMAIN context

HALF_GEOM = {                                  # e200's committed lag matrix (the s/2 rung)
    "cos_u0_u1": -0.26317857219199464, "cos_u1_u2": -0.3128523818599012,
    "cos_u2_u3": -0.3412904771804916, "cos_u3_u4": -0.352800001023403,
    "cos_u0_u2": 0.14055119609955105, "cos_u0_u3": -0.037019047726428826,
    "cos_u0_u4": -0.02586469602319846, "cos_u1_u3": 0.23745856827065193,
    "cos_u1_u4": 0.01718571433556464, "cos_u2_u4": 0.14245581289065523,
}

# ---- e202's committed ladder (the LOADED rungs; hard-bound at load) -----------
E202_EIGHTH = {                                # e202 phase_ladder.eighth (committed)
    "s": 0.11455244570970535,
    "lag1": [0.0019619047675956303, -0.0831166669077634,
             -0.18563095291942452, -0.17215000049936585],
    "lag2": [0.269698499162363, 0.33203005943139885, 0.35200175224796215],
    "cos1_t0": 0.0019619047675956303, "cos2_t0": 0.269698499162363,
    "core_t0": 0.1338682971973837,
    "stop": ["cap", 5],
}
E202_QUARTER = {                               # e202 phase_ladder.quarter (committed; the anchor)
    "s": 0.2291048914194107,
    "lag1": [-0.20573333393011753, -0.35675476293963715,
             -0.406645239274839, -0.41155000119382884],
    "lag2": [0.2606420463206616, 0.42689919231722745, 0.42437242876795184],
    "cos1_t0": -0.20573333393011753, "cos2_t0": 0.2606420463206616,
    "core_t0": 0.2331876901253896,
    "stop": ["cap", 5],
    "ray_md5s": {
        "u0": "9395918d36425e65248dbd93b6fd50bc",
        "u1": "639ec742fb01e5ad20d27dcac5335589",
        "u2": "9ec966a9601a045d482437cea3714126",
        "u3": "60c1671aa05ab13419fb7108222fd633",
        "u4": "d88c17590b8116ea2798489b86019430",
    },
}
E202_HALF = {                                  # e202 phase_ladder.half (committed)
    "s": 0.4582097828388214,
    "lag1": [-0.26317857219199464, -0.3128523818599012,
             -0.3412904771804916, -0.352800001023403],
    "lag2": [0.14055119609955105, 0.23745856827065193, 0.14245581289065523],
    "cos1_t0": -0.26317857219199464, "cos2_t0": 0.14055119609955105,
    "core_t0": 0.20186488414577286,
    "stop": ["kill", 5],
}

# ---- THE REGISTERED RUNG SET (FROZEN HERE, before any walk) -------------------
LADDER_E207 = [                                # (key, fraction of the natural s, the size)
    ("quarter_anchor", 0.25, E193_STEP_L2 * 0.25),
    ("three_eighths", 0.375, E193_STEP_L2 * 0.375),
]
assert abs(LADDER_E207[0][2] - E202_QUARTER["s"]) < 1e-15, \
    "the s/4 rung must BE e202's committed quarter size"
assert all(s < STATIC_SIGN_EDGE for _, _, s in LADDER_E207), \
    "every fresh rung must sit strictly inside the static sign edge (in-domain)"

MATCH_TOL = 1e-12                              # the monotonicity comparison epsilon (e202's convention)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                             # bit-class ceiling
G_TEXTURE_TOL = 1e-3                           # e199's disclosed cross-thread/environment tier
PERIOD2_C3_BAR = 0.05                          # e202's fingerprint |cos3| bar (context)

REGISTERED_BARS = {
    "CORE_BREAK_REAL": "CORE-BREAK-REAL: \"fires if the core statistic is "
        "smooth/montone through {s/4, 3s/8} and breaks only at s/2 — the "
        "null's lag-2 structure is genuinely incomplete; the derivation's "
        "debt stands as a finding.\"",
    "CORE_GRAINY": "CORE-GRAINY: \"fires if the core wobbles non-"
        "monotonically across the interior rungs — the statistic is grain; "
        "the null's cos1 law (already confirmed) is the whole in-domain "
        "story; the lag-2 term retires.\"",
    "GRADED": "GRADED: \"any partial — the table verbatim.\"",
    "operationalizations": "core_t = (cos2(u0,u2) - cos1(u0,u1))/2 at t=0 "
        "per rung (e202's registered statistic verbatim; cosines fp64; "
        "fronts at ALIVE ts only, own g-4 ruler > 0.27); the adjudication "
        "series (ascending s) = {c18 = e202 committed eighth (LOADED), "
        "c14 = THIS CELL's fresh s/4 (anchored, |dev vs committed| <= 1e-3 "
        "required at the gate), c38 = THIS CELL's fresh 3s/8 (new), c12 = "
        "e202 committed half (LOADED)}; CORE-BREAK-REAL fires iff c38 is "
        "readable AND c18 <= c14 <= c38 (tol 1e-12) AND c38 > c12 (tol "
        "1e-12) — monotone growth through the interior with the break "
        "landing only at the half rung; CORE-GRAINY fires iff c38 is "
        "readable AND c38 < c14 (beyond the tol) — the interior turns "
        "down (sub-flagged STRONG if additionally c38 <= c12); GRADED "
        "fires otherwise — any partial, the table verbatim (an "
        "unreadable c38 — the rung died before t=2 — or an anchor-tier "
        "ambiguity that moves the pivot); composite order frozen "
        "CORE-BREAK-REAL -> CORE-GRAINY -> GRADED; the natural-s point "
        "(org2-dead-full cos1 -0.16531) rides as OUT-OF-DOMAIN context "
        "only (1.74x its own kill edge — the derivation's own exclusion; "
        "core undefined-by-exclusion); the full lag-1/lag-2 series + the "
        "period-2 fingerprint reported per rung as CONTEXT (e202's "
        "convention).",
    "registered_prediction": "the null's own step-size law (scratch "
        "sign_front_null.md sec.4) predicts the 3s/8 core CONTINUES the "
        "committed rise (>= the quarter's 0.23319); CORE-BREAK-REAL fires "
        "exactly when that continuation holds while the committed "
        "half-rung collapse stands (c38 > c12) — the null's cos1 law "
        "confirmed in-domain with its lag-2 term breaking at the half; "
        "nothing guaranteed; no bar shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar "
        "shopping.",
}

deviations: list[str] = [
    "THE SPLIT: the existing rungs {s/8, s/2} are LOADED COMMITTED from "
    "runs/e202/metrics.json (hard-bound constants, cross-gated) — never "
    "recomputed; the ONLY fresh compute is the rung set {s/4, 3s/8} (two "
    "<= 6-step sign walks on the gated org2 root) + battery reads — the "
    "e201 u3-burst class, CPU minutes.",
    "THE ANCHOR REPLICATION: s/4 is e202's committed quarter rung REBUILT "
    "(G_WALKE202: journal rows, x-hashes, ray md5s u0..u4, mutual "
    "geometry — tier-stamped) — the e202 precedent (its s/2 rung anchored "
    "vs e197) transposed one rung down; the 3s/8 rung is not believed "
    "until the s/4 anchor passes; the 3s/8 rung's u0 must md5 == e193's "
    "committed R2_SIGN (every rung shares the root's step-1 front).",
    "THREADS 4 (the owner envelope's cap) where e197's committed chain ran "
    "threads 8: cross-thread/environment drift at ~1e-7 fp32 is possible, "
    "so every recompute gate is TIERED (BIT md5/1e-9 or e199's disclosed "
    "TEXTURE tier at 1e-3), the achieved tier STAMPED per gate; e202 "
    "rebuilt the same walks at threads 4 on this machine to TEXTURE/BIT "
    "tier — this process runs the same cap on the same machine.",
    "THE FIVE-RUNG TABLE = the four in-domain rungs {s/8, s/4, 3s/8, s/2} "
    "+ the OUT-OF-DOMAIN natural-s context row (the e202 convention "
    "carried; the dispatch's 'five-rung table s/8..s/2' spans the "
    "in-domain rungs with the registered out-of-domain point attached).",
    "the adjudication series mixes LOADED (eighth/half) and FRESH "
    "(quarter/3s/8) values — the anchor gate makes the fresh quarter the "
    "same object as e202's committed quarter to tier; both values are "
    "reported in the table; the walks are DETERMINISTIC (fixed "
    "seed-10902 stream, fixed root), so the fresh s/4 is a replication "
    "check of the machinery, not an independent draw.",
    "no twin arm, no static ray mapping in this cell (no D_kill "
    "profiles): the ladder reads only the walk fronts' mutual cosines — "
    "the lethality instruments (death-at-deepest-landing, onset curves) "
    "are the null's SURVIVORS and are not touched here.",
    "Smoke mode trims: rungs 2 steps, G_WALKE202 gated on the first 2 "
    "rows only (verdict stamped SMOKE; nothing adjudicated).",
]


# ---------------------------------------------------------------- instruments
# PROVENANCE: lab/e202_signfront_null.py's instruments (whose own provenance is
# lab/e201_rotation_census.py + lab/e200_deepening.py's walk_sub arithmetic —
# the e176n lineage). Copied rather than imported to own the device policy.

def load_net(path, cfg) -> TinyGPT:
    m = TinyGPT(cfg)
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
    """e193/e194/opt2's sign_update VERBATIM: delta = -step_l2*sign(g)/||sign(g)||
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


def interp_d_kill(v0, v1, d0, d1):
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    pts = [(d0, v0)] + [(r["D"], r["gm"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


# ------------------------------------------------------------------ the walk

def sign_walk(net0, anchor_neutral, train_ids, itos, primary_ids, coruler_ids,
              zid, theta0, step_l2, step_cap, kill_gate, root_gm, tag,
              extend_past_kill=1, d_target=5.0):
    """The e194/e197/e200/e202 sign walk VERBATIM in its arithmetic (k=1
    post-clip refresh, per-step battery, densified kill bracket at the
    killing step; stashes g_fronts + endpoints — the ray factory)."""
    net = copy.deepcopy(net0)
    net.train()
    evl = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    traj, x_hashes = [], {}
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_flat = theta0
    kill = None
    step = 0
    t0a = time.time()
    while True:
        step += 1
        post_kill = kill is not None
        if kill_gate:
            if post_kill and step > kill["step"] + extend_past_kill:
                break
            if kill is None and step > step_cap:
                kill = {"kind": "cap", "step": step - 1,
                        "reason": f"step cap {step_cap - 1} alive "
                                  f"(the registered {step_cap - 1}-step rung)",
                        "final_D": traj[-1]["cum_disp"],
                        "final_gm": traj[-1]["gm"]}
                break
        else:
            if step > step_cap:
                break
        aj = torch.randint(16, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        if step in E185_XHASH:
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
        assert l2dev < 1e-5, f"[{tag}] per-step L2 dev {l2dev:.2e}"
        max_l2_dev = max(max_l2_dev, l2dev)
        cur = prev_flat + delta
        load_flat(net, cur)
        cum_disp = float(torch.norm(cur - theta0))
        evl.load_state_dict({k_: v.detach().cpu().clone()
                             for k_, v in net.state_dict().items()})
        evl.eval()
        gz = battery_cell(evl, primary_ids, zid)
        row = {"step": step, "post_kill": post_kill,
               "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"]}
        for j in coruler_ids:
            row[f"g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        traj.append(row)
        g_fronts.append(g_t.clone())
        endpoints[step] = cur.clone()
        prev_flat = cur
        log(f"  [{tag} s{step}]{' POSTKILL' if post_kill else ''} "
            f"gm {row['gm']:.10f} D {cum_disp:.10f} ce {row['ce_batch']:.6f} "
            f"gnorm {gnorm:.4f}")
        if kill_gate and kill is None and gz["mean_pz"] <= SHUT_BAR:
            dens = []
            for f_ in (0.2, 0.4, 0.6, 0.8):
                pt_flat = prev_flat + (f_ - 1.0) * delta
                load_flat(evl, pt_flat)
                evl.eval()
                gzd = battery_cell(evl, primary_ids, zid)
                dens.append({"f": f_, "gm": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm, "cum_disp": 0.0}
            d_raw = interp_d_kill(prev_row["gm"], row["gm"],
                                  prev_row["cum_disp"], cum_disp)
            d_dens = dens_d_kill(prev_row["gm"], row["gm"],
                                 prev_row["cum_disp"], cum_disp, dens)
            kill = {"kind": "kill", "step": step,
                    "gm_at_kill": row["gm"], "D_kill_raw": d_raw,
                    "D_kill": d_dens, "dens": dens,
                    "reason": "primary ruler <= SHUT at an every-step read"}
            log(f"  [{tag}] KILL at s{step}: D_kill(dens) {d_dens:.10f}")
        elif kill_gate and kill is None and cum_disp >= d_target:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {d_target} alive",
                    "final_D": cum_disp, "final_gm": row["gm"]}
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "stop": kill, "x_hashes": x_hashes,
            "max_l2_dev": max_l2_dev, "g_fronts": g_fronts,
            "endpoints": endpoints, "seconds": round(time.time() - t0a, 1)}


def rays_and_geometry(g_fronts, n_alive):
    """u_t = sign(g_t)/||sign(g_t)|| for t < n_alive (ALIVE reads only);
    the full mutual-cosine matrix (fp64); lag-k series; the period-2
    fingerprint per available t; the core statistic at t=0."""
    us = []
    for t in range(n_alive):
        s = torch.sign(g_fronts[t])
        us.append((s / torch.norm(s)).clone())
    md5s = {f"u{t}": hashlib.md5(u.numpy().tobytes()).hexdigest()
            for t, u in enumerate(us)}
    n = len(us)
    cos = {(i, j): cos64(us[i], us[j]) for i in range(n) for j in range(i + 1, n)}
    lag = {k: [cos[(t, t + k)] for t in range(n - k)]
           for k in range(1, min(4, n))}
    p2 = {}
    for t in range(n - 1):
        c1 = cos[(t, t + 1)]
        c2 = cos[(t, t + 2)] if t + 2 < n else None
        c3 = cos[(t, t + 3)] if t + 3 < n else None
        p2[f"t{t}"] = {"cos1": c1, "cos2": c2, "cos3": c3,
                       "cos1_neg": bool(c1 < 0),
                       "cos2_pos": bool(c2 is not None and c2 > 0),
                       "cos3_small": bool(c3 is None or abs(c3) <= PERIOD2_C3_BAR),
                       "fingerprint_holds": bool(
                           c1 < 0 and (c2 is not None and c2 > 0)
                           and (c3 is None or abs(c3) <= PERIOD2_C3_BAR))}
    geom = {"n_fronts": n, "ray_md5s": md5s,
            "mutual_cosines": {f"u{i}_u{j}": v for (i, j), v in cos.items()},
            "lag_series": {f"lag{k}": v for k, v in lag.items()},
            "period2_fingerprint": p2}
    if n >= 3:
        geom["core_t0"] = (cos[(0, 2)] - cos[(0, 1)]) / 2.0
        geom["cos1_t0"] = cos[(0, 1)]
        geom["cos2_t0"] = cos[(0, 2)]
    return us, geom


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e207_smoke" if SMOKE else "e207")
    log(f"E207 THE LAG-2 RUNG SET (smoke={SMOKE}) -> {rd}")
    load_pct = cpu_load_probe()
    if load_pct is not None and load_pct > 85:
        log(f"load probe {load_pct}% > 85 — waiting 60 s (the envelope's "
            f"stagger rule)")
        time.sleep(60)
        load_pct = cpu_load_probe()
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), load probe "
        f"{load_pct if load_pct is not None else 'n/a'}%, two <= 6-step sign "
        f"walks + battery reads (the e201 u3-burst class)")

    stub: dict = {"gates": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e207_lag2_rungs",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE, "torch": torch.__version__,
                               "threads": torch.get_num_threads(),
                               "load_probe_pct": load_pct},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    # =====================================================================
    # PHASE 0 — parents loaded + the stream/root gates (Rule 12)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the parent ledger + the standard cell gates")
    missing = [str(p) for p in PARENTS.values() if not p.exists()]
    G_FILES = {"parents": {k: str(v) for k, v in PARENTS.items()},
               "missing": missing}
    assert not missing, f"missing parent metrics: {missing}"
    M = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in PARENTS.items()}
    name_checks = {k: (M[k].get("experiment") or "").startswith(k)
                   for k in PARENTS}
    # e157 predates the status convention (the e15x era); the modern parents
    # must read COMPLETE.
    MODERN_PARENTS = ("e193", "e197", "e200", "e202")
    status_checks = {k: ("COMPLETE" in str(M[k].get("status", "")))
                     for k in MODERN_PARENTS}
    keys_needed = {
        "e157": ["stages"],
        "e193": ["organism"],
        "e197": ["phase1_alive_walk", "phase1b_rays"],
        "e200": ["ray_geometry", "cell"],
        "e202": ["phase_ladder", "adjudication"],
    }
    key_checks = {k: all(n in M[k] for n in ns)
                  for k, ns in keys_needed.items()}
    G_FILES.update({
        "experiment_names": name_checks,
        "statuses_complete": status_checks,
        "status_check_note": "COMPLETE required of the four modern parents "
                             "only (e157 predates the status convention — "
                             "disclosed)",
        "pulled_keys_present": key_checks,
        "pass": bool(all(name_checks.values())
                     and all(status_checks.values())
                     and all(key_checks.values()))})
    log(f"G_FILES: 5 parents, names {sum(name_checks.values())}/5, COMPLETE "
        f"{sum(status_checks.values())}/4, keys {sum(key_checks.values())}/5: "
        + ("PASS" if G_FILES["pass"] else "FAIL"))
    assert G_FILES["pass"], "parent identity gate failed"

    # ---- G_CROSSFILE: the committed comparators agree across files ---------
    e202_lad = M["e202"]["phase_ladder"]
    g197 = M["e197"]["phase1b_rays"]["ray_geometry"]
    g200 = M["e200"]["ray_geometry"]
    e197_walk = M["e197"]["phase1_alive_walk"]

    def exact(a, b):
        return abs(a - b) == 0.0

    def series_exact(a, b):
        return len(a) == len(b) and all(x == y for x, y in zip(a, b))

    q = e202_lad["quarter"]
    e_ = e202_lad["eighth"]
    h_ = e202_lad["half"]
    xf = {
        "e193_step_l2_constants": exact(
            M["e193"]["organism"]["step_l2_measured"], E193_STEP_L2),
        "e197_sub_s_constants": exact(e197_walk["sub_s"], E197_SUB_S),
        "e157_dial_constants": all(
            abs(M["e157"]["stages"]["A_consolidate"]["dial"]["base"][k]
                ["mean_pz"] - v) < 1e-12
            for k, v in (("-12", E157_DIAL["gm12"]), ("0", E157_DIAL["g0"]),
                         ("12", E157_DIAL["gp12"]))),
        "e200_lag_matrix_constants": all(
            exact(g200[k], v) for k, v in HALF_GEOM.items()),
        "e200_committed_e197_trio": all(
            exact(g200["committed_e197"][k], g197[k])
            for k in ("cos_u0_u1", "cos_u0_u2", "cos_u1_u2")),
        "e197_ray_md5s_constants": all(
            M["e197"]["cell"]["rays"][i]["u_md5"] == h
            for i, h in ((1, E197_U1_MD5), (2, E197_U2_MD5))),
        "e200_ray_md5s_constants": all(
            M["e200"]["cell"]["rays"][i]["u_md5"] == h
            for i, h in ((3, E200_U3_MD5), (4, E200_U4_MD5))),
        "e200_u0_md5_vs_e193_r2sign": (
            M["e200"]["cell"]["rays"][0]["u_md5"] == E193_USIGN_MD5),
        # e202's loaded ladder hard-bound to this module's frozen constants
        "e202_eighth_constants": (
            exact(e_["s"], E202_EIGHTH["s"])
            and series_exact(e_["lag1_series"], E202_EIGHTH["lag1"])
            and series_exact(e_["geometry"]["lag_series"]["lag2"],
                             E202_EIGHTH["lag2"])
            and exact(e_["geometry"]["core_t0"], E202_EIGHTH["core_t0"])
            and exact(e_["geometry"]["cos1_t0"], E202_EIGHTH["cos1_t0"])
            and e_["stop"]["kind"] == E202_EIGHTH["stop"][0]
            and e_["stop"]["step"] == E202_EIGHTH["stop"][1]),
        "e202_quarter_constants": (
            exact(q["s"], E202_QUARTER["s"])
            and series_exact(q["lag1_series"], E202_QUARTER["lag1"])
            and series_exact(q["geometry"]["lag_series"]["lag2"],
                             E202_QUARTER["lag2"])
            and exact(q["geometry"]["core_t0"], E202_QUARTER["core_t0"])
            and exact(q["geometry"]["cos1_t0"], E202_QUARTER["cos1_t0"])
            and q["stop"]["kind"] == E202_QUARTER["stop"][0]
            and q["stop"]["step"] == E202_QUARTER["stop"][1]),
        "e202_quarter_ray_md5s_constants": all(
            q["geometry"]["ray_md5s"][k] == v
            for k, v in E202_QUARTER["ray_md5s"].items()),
        "e202_half_constants": (
            exact(h_["s"], E202_HALF["s"])
            and series_exact(h_["lag1_series"], E202_HALF["lag1"])
            and series_exact(h_["geometry"]["lag_series"]["lag2"],
                             E202_HALF["lag2"])
            and exact(h_["geometry"]["core_t0"], E202_HALF["core_t0"])
            and exact(h_["geometry"]["cos1_t0"], E202_HALF["cos1_t0"])
            and h_["stop"]["kind"] == E202_HALF["stop"][0]
            and h_["stop"]["step"] == E202_HALF["stop"][1]),
        "e202_half_geometry_vs_e200": all(
            exact(h_["geometry"]["mutual_cosines"][k], v)
            for k, v in HALF_GEOM.items()),
        "e202_ladder_registered_sizes": (
            exact(e_["s"], E193_STEP_L2 * 0.125)
            and exact(q["s"], E193_STEP_L2 * 0.25)
            and exact(h_["s"], E193_STEP_L2 * 0.5)),
    }
    G_CROSSFILE = {
        "checks": {k: bool(v) for k, v in xf.items()},
        "pass": bool(all(xf.values())),
        "note": "the committed comparators (org2's step L2 + e197's sub-s + "
                "the e200 lag matrix and ray md5s + e202's ENTIRE loaded "
                "ladder — sizes, lag series, cores, stops, quarter rays) are "
                "cross-consistent across the five parents BEFORE anything is "
                "compared (Rule 12)",
    }
    bad = [k for k, v in xf.items() if not v]
    log(f"G_CROSSFILE: {sum(bool(v) for v in xf.items())}/{len(xf)} checks "
        f"{'PASS' if G_CROSSFILE['pass'] else 'FAIL: ' + str(bad)}")
    assert G_CROSSFILE["pass"], f"cross-file consistency failed: {bad}"
    e202_quarter_journal = q["walk_journal"]
    e202_quarter_xh = q["x_hashes"]

    # ---- protocol rebuild (the shared corpus/battery/stream machinery) ------
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
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in READ_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in READ_GEOS},
        "pass": bool(all(list(bat_ids[j].shape) == [60, PRE + j]
                         for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): the install-60 battery (the "
                "ZEPHYRA install — org2's own host set) at the seven read "
                "geometries, 60 x (130 +- j) — e193/e194/e197/e200/e202's "
                "gate verbatim; it serves the org2 g-4 ruler",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    org2_primary, org2_corulers = bat_ids[RULER_J2], {j: bat_ids[j] for j in CO_RULERS_J2}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)

    arng = random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")):
            rejections += 1
            continue
        n_starts.append(s)
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": "16 plain corpus windows from train_ids, RNG seed "
                        "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                        "in [s, s+257) — e170 VERBATIM",
        "starts": n_starts,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 7):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_6_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values())),
                "note": "e193/e194/e197/e202's G_STREAM VERBATIM: the "
                        "seed-10902 stream construction (net-independent — "
                        "the SAME batches serve every rung) md5-matches "
                        "e185's stored hashes, steps 1..6"}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..6): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # ---- G_ROOT: the org2 root (the ladder's root) -------------------------
    net2 = load_net(CKPT_DIR / ROOT2_CK, F2_CFG)
    theta2 = flat_params(net2)
    assert int(theta2.numel()) == N_PARAM2, f"org2 params {theta2.numel()}"
    root2_md5 = hashlib.md5(theta2.numpy().tobytes()).hexdigest()
    evl2 = copy.deepcopy(net2)
    root2_cells = {f"g{j:+d}": battery_cell(evl2, bat_ids[j], zid)["mean_pz"]
                   for j in READ_GEOS}
    root2_cells["ce_r"] = ce_fixed_cpu(evl2, r_eval_x, r_eval_y)
    keymap = {"gm12": "g-12", "g0": "g+0", "gp12": "g+12",
              "g-4": "g-4", "g+4": "g+4", "g-8": "g-8", "g+8": "g+8",
              "ce_r": "ce_r"}
    refs2 = {keymap[k]: v for k, v in E157_DIAL.items()}
    diffs2 = {k: root2_cells[k] - refs2[k] for k in refs2}
    rmax2 = max(abs(v) for v in diffs2.values())
    G_ROOT = {"cells": root2_cells, "refs": refs2, "diffs": diffs2,
              "max_abs_diff": rmax2, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax2 < G_BIT_TOL),
              "flat_md5": root2_md5,
              "flat_md5_match_e193_committed": bool(root2_md5 == ROOT2_FLAT_MD5),
              "pass": bool(rmax2 < G_FALLBACK_TOL
                           and root2_md5 == ROOT2_FLAT_MD5),
              "note": "e193/e197/e200/e202's G_ROOT VERBATIM: the f2 "
                      "consolidated root's dial reproduces e157's committed "
                      "cells BIT-TIGHT + the flat md5 matches e193's "
                      "committed root identity"}
    log(f"G_ROOT (org2, vs e157 committed dial): max|diff| {rmax2:.2e}, md5 "
        f"{'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    assert G_ROOT["pass"], "org2 root gate FAILED"

    stub["gates"] = {"G_FILES": G_FILES, "G_CROSSFILE": G_CROSSFILE,
                     "G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_STREAM": G_STREAM, "G_ROOT": G_ROOT}
    stub["registered_rung_set"] = {
        "natural_s": E193_STEP_L2,
        "static_sign_edge": STATIC_SIGN_EDGE,
        "rungs": [{"key": k, "fraction": f, "s": s} for k, f, s in LADDER_E207],
        "movement_arithmetic": "per-step L2 asserted EXACT (fp64 norm, dev "
                               "< 1e-5, every step of every rung — e194's "
                               "convention); the s/4 size asserted == e202's "
                               "committed quarter size EXACTLY; both rungs "
                               "strictly inside the 0.5252 static sign edge "
                               "(in-domain); sizes frozen in LADDER_E207 "
                               "before any walk",
        "loaded_rungs": "eighth + half LOADED COMMITTED from e202's "
                        "metrics.json (hard-bound constants, cross-gated); "
                        "the natural-s row is OUT-OF-DOMAIN context "
                        "(1.74x its own kill edge — the derivation's own "
                        "exclusion)",
    }
    write_partial("phase 0 complete (parents + stream/root gates)")
    log("phase 0 complete; partial metrics written")

    log("WHAT THESE RUNGS GUARANTEE: NOTHING — n=1 per rung; deterministic "
        "walks on one root, one stream; the openness is the point.")

    # =====================================================================
    # PHASES 1-2 — THE RUNG SET {s/4 (anchor), 3s/8 (new)} on the root
    # =====================================================================
    log("=" * 78)
    log("THE RUNG SET — sign walks on the FACT-CARRYING org2 root at the "
        "registered sizes " + ", ".join(f"{k} s={s:.10f}" for k, _, s in LADDER_E207))
    rungs = []
    for key, frac, s in LADDER_E207:
        cap = 2 if SMOKE else RUNG_STEPS
        wk = sign_walk(net2, anchor_neutral, train_ids, itos, org2_primary,
                       org2_corulers, zid, theta2, s, cap,
                       kill_gate=True, root_gm=ROOT2_GM, tag=f"rung-{key}")
        kill_step = wk["stop"]["step"] if wk["stop"]["kind"] == "kill" else None
        n_alive = kill_step if kill_step is not None else len(wk["traj"])
        us_k, geom_k = rays_and_geometry(wk["g_fronts"], n_alive)
        rung = {"key": key, "fraction": frac, "s": s, "step_cap": cap,
                "walk_journal": wk["traj"], "stop": wk["stop"],
                "x_hashes": wk["x_hashes"], "max_l2_dev": wk["max_l2_dev"],
                "seconds": wk["seconds"],
                "alive_ts": list(range(n_alive)),
                "alive_convention": "fronts read at theta_t for t < kill step "
                                    "(own g-4 ruler > 0.27; the post-kill "
                                    "row rides in the journal only)",
                "geometry": geom_k,
                "lag1_series": geom_k["lag_series"].get("lag1"),
                "lag2_series": geom_k["lag_series"].get("lag2"),
                "period2": geom_k["period2_fingerprint"]}
        rungs.append((key, frac, s, wk, us_k, geom_k, rung))
        if "phase_rungs" not in stub:
            stub["phase_rungs"] = {}
        stub["phase_rungs"][key] = rung
        write_partial(f"rung {key} complete (s={s:.6f})")
        log(f"rung {key}: lag-1 series "
            + ", ".join(f"{v:+.5f}" for v in (geom_k["lag_series"].get("lag1") or []))
            + " | lag-2 series "
            + ", ".join(f"{v:+.5f}" for v in (geom_k["lag_series"].get("lag2") or [])))
        del us_k

    # ---- G_WALKE202: the fresh s/4 rung vs e202's committed quarter ---------
    qa = next(r for r in rungs if r[0] == "quarter_anchor")
    wk_q = qa[3]
    geom_q = qa[5]
    n_gate_rows = 2 if SMOKE else 5
    row_keys = ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm",
                "gm", "frac_argmax_z", "g+12", "g-12", "g+0")
    rows_gate = {}
    for i, (a, b) in enumerate(zip(wk_q["traj"][:n_gate_rows],
                                   e202_quarter_journal[:n_gate_rows])):
        for k in row_keys:
            mv, cv = a.get(k), b.get(k)
            if mv is None or cv is None:
                continue
            rows_gate[f"s{i+1}_{k}"] = {"measured": mv, "committed": cv,
                                        "abs_diff": abs(mv - cv)}
    md = max(v["abs_diff"] for v in rows_gate.values()) if rows_gate else 1.0
    xh_ok = {s_: bool(wk_q["x_hashes"][s_] == e202_quarter_xh[str(s_)])
             for s_ in range(1, n_gate_rows + 1)}
    n_md5 = len(wk_q["traj"]) if SMOKE else 5
    q_md5s = geom_q["ray_md5s"]
    md5_ok = {k: bool(q_md5s.get(k) == E202_QUARTER["ray_md5s"][k])
              for k in ("u0", "u1", "u2", "u3", "u4") if k in q_md5s}
    q_mut = geom_q["mutual_cosines"]
    e202_q_mut = q["geometry"]["mutual_cosines"]
    geom_dev = {k: abs(q_mut[k] - e202_q_mut[k])
                for k in q_mut if k in e202_q_mut}
    max_geom_dev = max(geom_dev.values()) if geom_dev else 0.0
    core_dev = abs(geom_q.get("core_t0", float("nan"))
                   - E202_QUARTER["core_t0"]) if "core_t0" in geom_q else None
    tier_w = ("BIT" if (md < G_REPRO_TOL and all(md5_ok.values())
                        and len(md5_ok) == n_md5)
              else ("TEXTURE" if (md < G_TEXTURE_TOL
                                  and max_geom_dev < G_TEXTURE_TOL
                                  and (core_dev or 0.0) < G_TEXTURE_TOL)
                   else "FAIL"))
    G_WALKE202 = {
        "rows": rows_gate, "max_abs_diff": md,
        "tol_bit": G_REPRO_TOL, "tol_texture": G_TEXTURE_TOL, "tier": tier_w,
        "x_hashes_vs_e202": xh_ok,
        "ray_md5s_vs_e202_quarter": md5_ok,
        "geometry_dev_vs_e202_quarter": geom_dev,
        "max_geom_dev": max_geom_dev,
        "core_dev_vs_e202_committed": core_dev,
        "max_l2_dev": wk_q["max_l2_dev"],
        "pass": bool(tier_w != "FAIL" and all(xh_ok.values())),
        "note": "THE ANCHOR REPLICATION GATE (Rule 12): the fresh s/4 rung "
                "must reproduce e202's committed quarter rung — journal rows "
                "incl. co-rulers, the x-hashes, the ray md5s u0..u4 — "
                "bit-class, or the disclosed cross-thread TEXTURE tier "
                "(threads 4 here and in e202; e199's precedent); the 3s/8 "
                "rung is not believed until this gate passes",
    }
    log(f"G_WALKE202 (fresh s/4 vs e202 committed quarter): max|diff| "
        f"{md:.2e}, core dev {core_dev if core_dev is None else format(core_dev, '.2e')}, "
        f"({tier_w}): "
        + ("PASS" if G_WALKE202["pass"] else "FAIL"))
    if not G_WALKE202["pass"]:
        stub["gates"]["G_WALKE202"] = G_WALKE202
        write_partial("CONTROL FAILURE — the s/4 anchor rung failed")
        raise RuntimeError("G_WALKE202 FAILED — abort before any rung is believed")
    stub["gates"]["G_WALKE202"] = G_WALKE202

    # ---- G_RAYSNEW: the 3s/8 rung's u0 == e193's committed R2_SIGN ----------
    te = next(r for r in rungs if r[0] == "three_eighths")
    geom_te = te[5]
    te_md5_u0 = geom_te["ray_md5s"].get("u0")
    G_RAYSNEW = {
        "three_eighths_u0_md5": te_md5_u0,
        "e193_r2sign_md5": E193_USIGN_MD5,
        "u0_match": bool(te_md5_u0 == E193_USIGN_MD5),
        "pass": bool(te_md5_u0 == E193_USIGN_MD5),
        "note": "THE NEW RUNG'S RAY IDENTITY GATE: the 3s/8 rung's u0 must BE "
                "e193's committed R2_SIGN ray (every rung shares the root's "
                "step-1 front — e202 confirmed it for its three rungs; the "
                "root + stream gates make it mandatory here)",
    }
    log(f"G_RAYSNEW (3s/8 u0 vs e193 R2_SIGN): "
        + ("PASS" if G_RAYSNEW["pass"] else "FAIL"))
    assert G_RAYSNEW["pass"], "3s/8 ray identity gate FAILED"
    stub["gates"]["G_RAYSNEW"] = G_RAYSNEW
    write_partial("rung set complete + anchor gates passed (G_WALKE202, G_RAYSNEW)")

    # =====================================================================
    # PHASE 3 — the five-rung table + the adjudication (the frozen bars)
    # =====================================================================
    log("=" * 78)
    log("PHASE 3 — THE ADJUDICATION (the frozen bars)")

    c18 = E202_EIGHTH["core_t0"]
    c14 = geom_q.get("core_t0")
    c38 = geom_te.get("core_t0")
    c12 = E202_HALF["core_t0"]
    core38_readable = c38 is not None

    five_rung_table = [
        {"key": "eighth", "fraction": 0.125, "s": E202_EIGHTH["s"],
         "source": "LOADED — e202 committed (cross-gated)",
         "lag1_series": E202_EIGHTH["lag1"], "lag2_series": E202_EIGHTH["lag2"],
         "cos1_t0": E202_EIGHTH["cos1_t0"], "cos2_t0": E202_EIGHTH["cos2_t0"],
         "core_t0": c18, "kill": list(E202_EIGHTH["stop"]),
         "n_alive_fronts": 5, "in_domain": True},
        {"key": "quarter", "fraction": 0.25, "s": E202_QUARTER["s"],
         "source": "FRESH (this cell) — anchored vs e202 committed "
                   f"(core dev {c14 - E202_QUARTER['core_t0']:+.2e}, tier "
                   f"{tier_w})" if c14 is not None else "FRESH (no core)",
         "lag1_series": geom_q["lag1_series"], "lag2_series": geom_q["lag2_series"],
         "cos1_t0": geom_q.get("cos1_t0"), "cos2_t0": geom_q.get("cos2_t0"),
         "core_t0": c14, "core_t0_committed_e202": E202_QUARTER["core_t0"],
         "kill": [qa[3]["stop"]["kind"], qa[3]["stop"]["step"]],
         "n_alive_fronts": geom_q["n_fronts"], "in_domain": True},
        {"key": "three_eighths", "fraction": 0.375, "s": te[2],
         "source": "FRESH (this cell) — the new interior rung",
         "lag1_series": geom_te["lag1_series"], "lag2_series": geom_te["lag2_series"],
         "cos1_t0": geom_te.get("cos1_t0"), "cos2_t0": geom_te.get("cos2_t0"),
         "core_t0": c38,
         "kill": [te[3]["stop"]["kind"], te[3]["stop"]["step"]],
         "n_alive_fronts": geom_te["n_fronts"], "in_domain": True},
        {"key": "half", "fraction": 0.5, "s": E202_HALF["s"],
         "source": "LOADED — e202 committed (cross-gated; == e197's walk)",
         "lag1_series": E202_HALF["lag1"], "lag2_series": E202_HALF["lag2"],
         "cos1_t0": E202_HALF["cos1_t0"], "cos2_t0": E202_HALF["cos2_t0"],
         "core_t0": c12, "kill": list(E202_HALF["stop"]),
         "n_alive_fronts": 5, "in_domain": True},
        {"key": "natural", "fraction": 1.0, "s": E193_STEP_L2,
         "source": "CONTEXT — org2-dead-full (e196 via e197), OUT OF DOMAIN",
         "lag1_series": [DEAD_FULL_C1], "lag2_series": None,
         "cos1_t0": DEAD_FULL_C1, "cos2_t0": None, "core_t0": None,
         "kill": ["kill", 1],
         "n_alive_fronts": None, "in_domain": False,
         "note": "1.74x its own 0.5252 static kill edge — the derivation's "
                 "own exclusion; core undefined-by-exclusion"},
    ]

    series = {"c18_eighth_loaded": c18, "c14_quarter_fresh": c14,
              "c38_three_eighths_fresh": c38, "c12_half_loaded": c12}
    if core38_readable and c14 is not None:
        interior_monotone = bool(c18 <= c14 + MATCH_TOL
                                 and c14 <= c38 + MATCH_TOL)
        interior_turns_down = bool(c38 < c14 - MATCH_TOL)
        break_only_at_half = bool(c38 > c12 + MATCH_TOL)
        strong_grain = bool(c38 <= c12 + MATCH_TOL)
    else:
        interior_monotone = interior_turns_down = None
        break_only_at_half = strong_grain = None

    real_fires = bool(core38_readable and c14 is not None
                      and interior_monotone and break_only_at_half)
    grainy_fires = bool(core38_readable and c14 is not None
                        and interior_turns_down)
    graded_fires = bool(not real_fires and not grainy_fires)

    bars = {"CORE_BREAK_REAL": {"fires": real_fires},
            "CORE_GRAINY": {"fires": grainy_fires},
            "GRADED": {"fires": graded_fires}}
    if SMOKE:
        verdict, clause = "SMOKE", "shakedown — nothing adjudicated"
    elif real_fires:
        verdict = "CORE-BREAK-REAL"
        clause = ("the core statistic is smooth/monotone through {s/4, 3s/8} "
                  "and breaks only at s/2 — the null's lag-2 structure is "
                  "genuinely incomplete; the derivation's debt stands as a "
                  f"finding. Numbers: cores (ascending s) {c18:.5f} -> "
                  f"{c14:.5f} -> {c38:.5f} -> {c12:.5f}; the interior "
                  "non-decreasing (yes), the 3s/8 core ABOVE the half's "
                  f"collapsed value by {c38 - c12:+.5f} — the collapse lands "
                  "between 3s/8 and s/2; cos1(t=0) "
                  f"{geom_te.get('cos1_t0'):+.5f} at 3s/8 sits on the "
                  "confirmed cos1 law's branch.")
    elif grainy_fires:
        verdict = "CORE-GRAINY"
        grain_word = ("STRONG — already at/below the half's collapsed value "
                      "by 3s/8" if strong_grain else
                      "the interior dips below the quarter but stays above "
                      "the half")
        clause = ("the core wobbles non-monotonically across the interior "
                  "rungs — the statistic is grain; the null's cos1 law "
                  "(already confirmed) is the whole in-domain story; the "
                  "lag-2 term retires. Numbers: cores (ascending s) "
                  f"{c18:.5f} -> {c14:.5f} -> {c38:.5f} -> {c12:.5f}; the "
                  f"interior turns down ({grain_word}); e202's half-rung "
                  "'break' was grain.")
    else:
        verdict = "GRADED"
        bits = []
        if not core38_readable:
            bits.append(f"the 3s/8 core is UNREADABLE (the rung died at "
                        f"step {te[3]['stop'].get('step')} — no in-domain "
                        "lag-2 pair; the rung contributes no core and the "
                        "bars cannot adjudicate the interior)")
        if c14 is None:
            bits.append("the fresh s/4 core is unreadable (anchor rung died "
                        "before t=2 — an anchor-tier ambiguity)")
        clause = ("any partial — the table verbatim: " + "; ".join(bits)
                  + f" Cores (ascending s): {c18:.5f} -> "
                  f"{c14 if c14 is None else format(c14, '.5f')} -> "
                  f"{c38 if c38 is None else format(c38, '.5f')} -> "
                  f"{c12:.5f}.")

    adjudication = {
        "bars": bars,
        "verdict": verdict,
        "clause": clause,
        "composite_order": "CORE-BREAK-REAL -> CORE-GRAINY -> GRADED "
                           "(frozen before compute)",
        "series": series,
        "interior_monotone": interior_monotone,
        "interior_turns_down": interior_turns_down,
        "break_only_at_half": break_only_at_half,
        "strong_grain_subflag": strong_grain,
        "core38_readable": core38_readable,
        "five_rung_table": five_rung_table,
        "period2_fingerprint_context": {
            "quarter_anchor": geom_q["period2_fingerprint"],
            "three_eighths": geom_te["period2_fingerprint"],
        },
        "out_of_domain_context": {
            "org2_full_step_natural_s": {"s": E193_STEP_L2,
                                         "cos1": DEAD_FULL_C1,
                                         "note": "OUT OF DOMAIN: 1.74x its "
                                                 "own 0.5252 static kill "
                                                 "edge — the derivation's "
                                                 "own exclusion; context "
                                                 "only; core "
                                                 "undefined-by-exclusion"},
        },
        "constants": {"MATCH_TOL": MATCH_TOL, "SHUT_BAR": SHUT_BAR,
                      "PERIOD2_C3_BAR": PERIOD2_C3_BAR,
                      "STATIC_SIGN_EDGE": STATIC_SIGN_EDGE},
    }
    log(f"ADJUDICATION: {verdict}")
    log(f"  clause: {clause}")

    # ---- honesty -------------------------------------------------------------
    honesty = {
        "n": "n=1 per rung — one walk per rung, single stream seed 10902; "
             "nothing is a distribution",
        "determinism": "the walks are DETERMINISTIC (fixed seed-10902 "
                       "stream, fixed root): the fresh s/4 is a replication "
                       "check of the machinery (gated to tier vs e202's "
                       "committed quarter), not an independent draw; the "
                       "3s/8 rung is one realized path at one counterfactual "
                       "size",
        "org2_root_caveat": "the whole ladder lives on ONE org2 root "
                            "(873k 4L, one fact, one stream) — a "
                            "within-organism step-size test; never a "
                            "cross-organism magnitude comparison (the "
                            "derivation's own exclusion)",
        "loaded_rungs": "the s/8 and s/2 rows are e202's committed values, "
                        "trusted via e202's own gates + this cell's "
                        "hard-bound cross-checks; only {s/4, 3s/8} were "
                        "walked here",
        "h_unmeasured": "curvature h and the bath agreements a_k remain "
                        "unmeasured — the core statistic is a derived "
                        "consistency object, not a measurement; this cell "
                        "tests its rung-to-rung smoothness only",
        "openness": "nothing guaranteed — a different root, stream, or rung "
                    "set could read differently; that openness is the point",
    }

    # ---- the plot ------------------------------------------------------------
    plots_ok = False
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.6))
        in_dom = [r for r in five_rung_table if r["in_domain"]]
        xs = [r["s"] for r in in_dom]
        cores_plot = [r["core_t0"] for r in in_dom]
        c1_plot = [r["cos1_t0"] for r in in_dom]
        c2_plot = [r["cos2_t0"] for r in in_dom]
        # panel 1 — THE REGISTERED STATISTIC: the core vs step size
        ax1.plot([xs[0], xs[3]], [cores_plot[0], cores_plot[3]], "o",
                 color="tab:gray", ms=10, zorder=4,
                 label="LOADED e202 committed (s/8, s/2)")
        ax1.plot(xs[1:3], cores_plot[1:3], "*", color="tab:red", ms=15,
                 zorder=6, label="FRESH e207 (s/4 anchored, 3s/8 NEW)")
        ax1.plot(xs, cores_plot, "-", color="tab:brown", lw=1.6, zorder=3,
                 alpha=0.8)
        ax1.plot([E202_QUARTER["s"]], [E202_QUARTER["core_t0"]], "o",
                 mfc="none", color="tab:gray", ms=10, zorder=5,
                 label="e202 committed quarter (the anchor's target)")
        ax1.plot([E193_STEP_L2], [DEAD_FULL_C1], "x", color="gray", ms=11,
                 mew=2, zorder=2, label="natural s (OUT OF DOMAIN — cos1 "
                                        "only, core undefined)")
        ax1.annotate("the half rung's\ncommitted collapse",
                     xy=(xs[3], cores_plot[3]),
                     xytext=(0.16, cores_plot[3] - 0.045),
                     arrowprops=dict(arrowstyle="->", color="tab:gray"),
                     fontsize=9, color="tab:gray")
        ax1.annotate("3s/8 — the missing\ninterior rung",
                     xy=(xs[2], cores_plot[2]),
                     xytext=(0.20, cores_plot[2] + 0.035),
                     arrowprops=dict(arrowstyle="->", color="tab:red"),
                     fontsize=9, color="tab:red")
        ax1.set_xscale("log")
        ax1.set_xlabel("per-step L2 s (log scale)")
        ax1.set_ylabel("core(t=0) = (cos2 - cos1)/2")
        ax1.set_title("THE CORE STATISTIC vs STEP SIZE — the five-rung table\n"
                      "(verdict: " + verdict + ")")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        # panel 2 — the raw series behind the core
        ax2.plot(xs, c1_plot, "s-", color="tab:purple", lw=2, ms=8,
                 label="cos1(u0,u1) — the CONFIRMED law (e202)")
        ax2.plot(xs, c2_plot, "^-", color="tab:blue", lw=2, ms=8,
                 label="cos2(u0,u2) — the lag-2 raw read")
        ax2.plot([E193_STEP_L2], [DEAD_FULL_C1], "x", color="gray", ms=11,
                 mew=2, zorder=2, label="natural s cos1 (OUT OF DOMAIN)")
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xscale("log")
        ax2.set_xlabel("per-step L2 s (log scale)")
        ax2.set_ylabel("cosine at t=0 (fp64)")
        ax2.set_title("the raw lag-1/lag-2 reads behind the core\n"
                      "(s/4 + 3s/8 fresh; s/8 + s/2 loaded from e202)")
        ax2.legend(fontsize=8.5)
        ax2.grid(alpha=0.3)
        fig.suptitle("E207 — THE LAG-2 RUNG SET (the null's last debt): is "
                     "the half rung's core collapse real or coarse?",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "e207_lag2_rungs.png", dpi=130)
        plt.close(fig)
        plots_ok = True
    except Exception as e:      # plots never block the adjudication
        log(f"plot failure (disclosed): {e}")
        plots_ok = False

    # ---- the final COMPLETE write ---------------------------------------------
    stub.pop("phases_partial", None)
    stub.update({
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "question": ("E207 — THE LAG-2 RUNG SET: is e202's half-rung core "
                     "collapse REAL (the core grows smoothly through the "
                     "missing interior {s/4, 3s/8} and collapses only at "
                     "s/2 — the null's lag-2 structure genuinely incomplete) "
                     "or COARSE (the core wobbles non-monotonically across "
                     "the interior — the statistic is grain and the null's "
                     "cos1 law is the whole in-domain story)?"),
        "provenance": {
            "parents_loaded_committed": sorted(PARENTS.keys()),
            "fresh_compute": "the rung set {s/4 (anchor replication), 3s/8 "
                             "(new)} — two 5-step sign walks on the gated "
                             "org2 root + battery reads — CPU-only, threads "
                             "4, the e201 u3-burst class",
            "instruments": "lab/e202_signfront_null.py's machinery VERBATIM "
                           "(its own provenance: lab/e201_rotation_census.py "
                           "+ lab/e200_deepening.py's walk_sub arithmetic); "
                           "bars frozen in the module docstring before "
                           "compute",
            "loaded_not_recomputed": "e202's eighth + half rungs (hard-bound "
                                     "constants, cross-gated); the natural-s "
                                     "row is out-of-domain context (e196 via "
                                     "e197)",
        },
        "phase_rungs": {r["key"]: r for _, _, _, _, _, _, r in rungs},
        "adjudication": adjudication,
        "honesty": honesty,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1),
                   "rung_s": {r["key"]: r["seconds"]
                              for _, _, _, _, _, _, r in rungs}},
        "config": {"smoke": SMOKE, "torch": torch.__version__,
                   "threads": torch.get_num_threads(),
                   "device": "cpu", "load_probe_pct": load_pct,
                   "plots_written": plots_ok},
    })
    save_json(rd / "metrics.json", E43.jsonable(stub))
    log(f"FINAL WRITE: {rd / 'metrics.json'} (verdict {verdict})")
    log(f"plots: {'written' if plots_ok else 'FAILED (disclosed)'}")


if __name__ == "__main__":
    main()
