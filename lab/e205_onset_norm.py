"""E205 — THE ONSET NORMALIZATION (R61-critic's Attack 2: the arrival
times were bar-contingent and self-normalized against edges differing
4.3x).

WHY (scratch/r61_critic.md Attack 2, verbatim theses): "(a) The bar moved
inside the same arc, and the half lineage's arrival sits exactly on the
seam" (FLIGHT_RATIO_BAR 0.60 in e196 vs 0.70 in e197/e198/e199/e200; the
half lineage's t2 ratio 0.663 is concentrated under 0.70, NOT under 0.60
— under e196's own bar its first arrival is t4, the killing step);
"(b) The denominators are not comparable" (the u0 edges are 2.2699 org1 /
1.9471 MIRABEL / 0.5252 half — a 4.3x spread; T155: in-span draws alone
spread D_kill 2.0-3.0x; "The lab already owns the correct instrument and
did not use it here: each organism's own in-span random-ray band ...
Arrival should be defined against THAT per-organism reference"); "(d)
Report curves, not first-crossings." The WHEN account on record is
T163/T164: "the concentration is a WHEN; the timing is the biography"
(org1 t1, MIRABEL t2, half t2 — 'the same curve shifted one step').

THE CELL (a desk+eval cell on committed data; ONE tiny fresh burst):
  (1) each organism's OWN in-span random band (the in-span directions'
      kill-Ds): org1 LOADED committed from e_chart's fine-D co-read
      (seeds 11601-3, e131 root, ZEPHYRA install-60 g-12 — the SAME
      ruler as e199's org1 onset reads); MIRABEL LOADED committed from
      e193b's inspan_range (seeds 11911-3, e193b root, MIRABEL
      install-60 g-12 — the SAME ruler as e199's MIRABEL onset reads);
      the half-step lineage's band was NEVER measured (e193's random
      rays are out-of-span Gaussians, all unresolved-high) — RECOMPUTED
      HERE in a tiny burst: e193b's in-span machinery VERBATIM ported to
      the e157_f2 root (20 CPU AdamW wash steps on the licensed seed-
      10902 stream -> Gram SVD span -> 3 fresh in-span draws -> walked
      down the onset grid 0.05..3.00, kill-D per the onset instrument:
      first g-4 read <= 0.27 downcrossing, linear-in-D interpolated).
  (2) each front's kill-D expressed as a MULTIPLE of its organism's
      in-span band MEDIAN (the middle order statistic; right-censored
      draws order above all resolved — with n=3 and <=1 censored the
      median is censoring-robust; >=2 censored => the common axis is
      UNDEFINED for that organism and its arrivals read DISSOLVED,
      disclosed).
  (3) the arrival curves re-plotted on this common axis — do the
      arrival times (org1 t1, MIRABEL t2, half t2) survive the
      normalization, or reorder/dissolve?
  (4) the bar-threshold sensitivity: the concentration bar swept
      {0.60 (e196's committed flight bar), 0.70 (e198/e199's frozen
      concentration bar), 0.80 (the sweep's upper arm)} on BOTH axes
      (SELF = the committed ratio vs own u0 edge; COMMON = the band
      multiple) — which arrival assignments are bar-robust?

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - ARRIVALS-ROBUST: "fires if the arrival ordering (org1 earliest;
    the others at-or-later) is invariant under the in-span
    normalization AND across the bar sweep — the WHEN account stands on
    a common ruler."
  - ARRIVALS-CONTINGENT: "fires if the ordering flips or dissolves
    under normalization or the bar sweep — the WHEN was an artifact of
    self-normalized ratios; the onset story rescopes to within-organism
    shape only."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * the onset fronts are LOADED COMMITTED (never re-run): org1 e199
    (u0 edge 2.2699, u1 0.3875; u2 = the POST-DEATH e195 read, context
    only, never adjudicated), MIRABEL e199 (u0 1.9471, u1 SOFT, u2
    0.8312), half e200 (u0 0.5252, u1 1.6645, u2 0.3483, u3 0.6787,
    u4 0.2087) — hard-bound at load vs the parents' metrics.
  * multiple_t = D_kill(root, u_t) / BAND_MEDIAN(organism); the t=0
    anchor is the u0 EDGE's own multiple (a real number on this axis —
    the self axis hides it inside ratio_0 = 1.0); a SOFT front (None
    D_kill within [0, 3.0]) gets the floor multiple 3.0/BAND_MEDIAN,
    reported, never adjudicated.
  * arrival(axis, bar) = the FIRST ALIVE t >= 1 whose axis value is <
    bar; post-death fronts are context only; SOFT fronts never arrive;
    an organism with no arriving alive t is DISSOLVED in that cell.
  * ARRIVALS-ROBUST fires iff the arrival assignment (org1, MIRABEL,
    half) equals the committed T163/T164 assignment (t1, t2, t2) in
    EVERY one of the six cells (axes {SELF, COMMON} x bars {0.60, 0.70,
    0.80}) — ordering AND instantiation invariant.
  * ARRIVALS-CONTINGENT fires iff ROBUST fails AND a DISSOLUTION or FLIP
    exists: DISSOLUTION = an organism that arrives in one cell fails to
    arrive in another — under the normalization (same bar, SELF vs
    COMMON) or across the bar sweep (within-axis, arrives at 0.80 but
    not at 0.60); FLIP = a pairwise arrival inversion under the
    normalization, or the SELF cell's strictly-earliest organism losing
    earliest-equality on COMMON (a full identity change).
  * GRADED: any other partial (e.g. assignments shift only across bars
    on the SELF axis, or a t moves without flip/dissolve) — the tables
    verbatim.
  * composite order frozen: ARRIVALS-ROBUST -> ARRIVALS-CONTINGENT ->
    GRADED (first that fires is the verdict).

DESK-KNOWN AT REGISTRATION (committed arithmetic, zero fresh compute —
stated on the table so the registration does not fake openness):
org1's t1 multiple = 0.3875/0.6101 = 0.635 (ABOVE the 0.60 bar) and
MIRABEL's t2 multiple = 0.8312/0.92 = 0.903 (above ALL THREE bars), so
org1 dissolves at (COMMON, 0.60) and MIRABEL dissolves on the ENTIRE
common axis => ARRIVALS-ROBUST is already IMPOSSIBLE and CONTINGENT's
dissolution clause is already satisfied — the verdict is desk-forced to
CONTINGENT-or-worse and will be reported as forced. THE OPEN CONTENT
the fresh compute owns: the half lineage's in-span band (never
measured) and therefore its common-axis arrival — does IT survive,
shift, or arrive EARLIER than org1 (a flip)? — plus the verbatim
normalized tables, the anchors, and the figures. MIRABEL's band is
first-dead GRID points (upper bounds on the true crossings): its band
median can only DROP under interpolation, which would only RAISE
MIRABEL's multiple — its dissolution is robust to the convention
difference (disclosed in honesty).

PRE-DISPATCH CHECKS (Rule 12): the loaded bands' provenance gates
(e_chart's e131 fine-D co-read read on the same root+ruler as e199's
org1 fronts; e193b's inspan_range at the e193b root on the same ruler
as e199's MIRABEL fronts; both hard-bound at load); the fresh f2 band's
chain: G_NAMEFREE/G_SPLICE/G_BATTERY/G_ANCHOR (e193/e197/e200's gates
verbatim-light), G_ROOT (flat md5 == e193's committed root identity +
the e157 dial's g-4/g-12 cells bit/texture), G_STREAM (the seed-10902
stream md5 vs e185's stored hashes, steps 1-8; steps 9-20 have no
committed hashes — deterministic-by-construction, disclosed), G_HIST
(the history's step-1 displacement L2 vs e193's committed MEASURED step
L2 0.9164), G_DIRCK (e193's committed g-ray direction checkpoint:
theta0 md5 + coordinate/cosine identity), G_GRAY (the fresh t=0 g-ray
walked down the band grid: its interpolated kill must reproduce e193's
committed R1_G g-4 bracket (0.15 alive / 0.2 dead -> interp 0.1967)
within the TEXTURE tier) — the band is not believed until these pass.
IDENTITY TIERS (e199/e200's disclosed convention): BIT = md5 / 1e-9
where this process reproduces the parents' bits (e193's chain was
threads 8, this process threads 4 — cross-thread drift at ~1e-7 fp32 is
possible); the disclosed TEXTURE tier then gates; the achieved tier is
STAMPED per gate, never silently weakened.

REGISTERED PREDICTION (frozen before the fresh compute): the critic's
account predicts the arrival times are CONTINGENT (the denominators
carry the lottery; the seam at 0.60 already moves half's SELF-axis
arrival to t4). T163/T164's account survives only if the bands scale
the edges so the multiples preserve (t1, t2, t2) — already excluded on
the desk side (above). The genuinely open fork is the HALF band: if
in-span draws at the f2 root are nearly as safe as its out-of-span
Gaussians (e193: all unresolved-high), the half band median runs high
(>=1.5), its multiples compress (u2 0.3483/1.5 = 0.23), and half's
COMMON arrival lands at t2 or even t1 (a flip past org1's 0.635); if
they are as lethal as the root's own g-ray (kill ~0.20), the median
runs ~0.2-0.6 and the u0 edge (0.5252) is barely at parity, pushing
half's arrival LATE or dissolving it. Either way the onset story
rescopes; which organ keeps an arrival is what the burst decides. No
bar shopping; the forced verdict is reported as forced.

WHAT THE FRESH ARM GUARANTEES: NOTHING — the band is n=3 draws of ONE
20-step wash-history realization (seed 10902) of ONE root per organism;
the fronts are single realized walks; the openness is the point.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never
claimed), torch threads 4 (the cap), load-check recorded (not gating —
e204's convention), sequential tiny phases, PROGRESSIVE metrics.json
writes after every phase (the outage lesson), n=1.

Outputs: runs/e205/{metrics.json, e205_common_axis.png (the common-axis
onset curves), e205_bar_sweep.png (the arrival matrix)}. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e205_onset_norm.py    (E205_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # THE OWNER ENVELOPE'S CAP

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E205_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e205 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
HOSTS = ["FLORIZEL", "ELIZABETH"]
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
BLOCK = 256                       # training-window block (the e098 line's own convention)
ROOT_CK = "e157_f2_consolidated.pt"   # THE FAMILY-2 CONSOLIDATED ROOT (e157's, committed)
DIR_CK = "e193_f2_static_dir_u.pt"    # e193's committed g-ray direction checkpoint
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E_CHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
E193B_METRICS = E43.REPO / "runs" / "e193b" / "metrics.json"
E193_METRICS = E43.REPO / "runs" / "e193" / "metrics.json"
E196_METRICS = E43.REPO / "runs" / "e196" / "metrics.json"
E199_METRICS = E43.REPO / "runs" / "e199" / "metrics.json"
E200_METRICS = E43.REPO / "runs" / "e200" / "metrics.json"

# ---- the family-2 net (e098 s4305 line: 4L/4H/128d/512-ctx, 873,472 params) ----
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- rulers (e193's, frozen; carried verbatim through e197/e200) ----------------
RULER_J = -4                     # PRIMARY: g-4 install-60 battery
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries

# ---- envelope / conventions ------------------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193/e197)
LR_ADAMW = 1e-3                   # the t=0/wash recipe (e157/e193b)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (the onset grid VERBATIM)
BARS = (0.60, 0.70, 0.80)         # e196's 0.60 / e198-e200's 0.70 / the 0.80 arm
WASH_HIST_STEPS = 4 if SMOKE else 20     # e193b's in-span history segments
INSPAN_SEEDS = (12001, 12002, 12003)     # FRESH (registry-clean: repo-grep'd; e193b used 11911-3)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_STATIC_TOL = 1e-3
E193_STEP_L2 = 0.9164195656776428          # e193's committed MEASURED AdamW step-1 L2
ROOT_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"   # e193's committed root identity
E157_DIAL_G4 = 0.8872273564338684          # the e157 dial's committed g-4 cell
E157_DIAL_G12 = 0.19826222956180573        # the e157 dial's committed g-12 cell
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
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.5, 1.5, 3.0]
    INSPAN_SEEDS = (12001,)

# ---- the committed onset fronts (LOADED, never re-run; hard-bound at load) --------
# org1 (e199; root e131_consolidated_e113; ruler ZEPHYRA install-60 g-12)
ORG1_ONSET = {
    "root": "runs/checkpoints/e131_consolidated_e113.pt",
    "ruler": "ZEPHYRA install-60 g-12 battery",
    "edge_u0_D_kill": 2.269916581032063,
    "fronts": {1: 0.3875040789853107},          # alive fronts only
    "death_t": 2,
    "post_death_context": {2: 0.8979436419840907},   # e195's read, never adjudicated
}
# MIRABEL (e199; root e193b_root; ruler MIRABEL install-60 g-12)
MIR_ONSET = {
    "root": "runs/checkpoints/e193b_root.pt",
    "ruler": "MIRABEL install-60 g-12 battery",
    "edge_u0_D_kill": 1.9470925580069711,
    "fronts": {1: None, 2: 0.8312446762047895},  # t1 SOFT (floor 3.0/edge)
    "death_t": 3,
    "t1_floor_ratio": 1.5407588035110036,
}
# the half-step lineage (e200; root e157_f2_consolidated; ruler install-60 g-4)
HALF_ONSET = {
    "root": "runs/checkpoints/e157_f2_consolidated.pt",
    "ruler": "install-60 g-4 battery (e193's frozen call)",
    "edge_u0_D_kill": 0.5252331597771716,
    "fronts": {1: 1.664473633460733, 2: 0.34829356925017246,
               3: 0.6786533732491854, 4: 0.20865737305118012},
    "death_t": 5,
}
# the committed T163/T164 arrival assignment (the claim under test)
COMMITTED_ASSIGNMENT = {"org1": 1, "mirabel": 2, "half": 2}

# ---- the committed in-span bands (org1 + MIRABEL; hard-bound at load) -------------
ORG1_BAND = {
    "source": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D",
    "seeds": (11601, 11602, 11603),
    "kill_Ds": {"11601": 0.6100797132671589, "11602": None, "11603": 0.5636664436670049},
    "instrument": "e_chart's fine-D co-read: FINE_GRID (0.05..1.5, 11 pts), kill = "
                  "first g-12 <= 0.27 downcrossing INTERPOLATED linear-in-D; None = "
                  "unresolved > 1.5 (right-censored)",
    "ruler": "ZEPHYRA install-60 g-12 (the SAME ruler as org1's onset fronts)",
    "root": "runs/checkpoints/e131_consolidated_e113.pt",
    "wash_g_ray_kill_context": 0.9085451308227387,   # e192's committed R1, loaded by e_chart
    "promotion_disclosure": "e_chart registered the fine-D co-read 'non-adjudicating' "
                            "(the rung ladder was its axis); e205 PROMOTES it to the band "
                            "instrument (R61-critic's named repair) — disclosed",
}
MIR_BAND = {
    "source": "runs/e193b/metrics.json inspan_range.MIRABEL",
    "seeds": (11911, 11912, 11913),
    "kill_Ds": [1.5, 0.92, 0.74],               # order as committed (s11911/2/3)
    "instrument": "e193b's D_GRID (27 pts to 4.0), kill = FIRST-DEAD GRID POINT (an "
                  "upper bound on the true crossing — NOT interpolated)",
    "ruler": "MIRABEL install-60 g-12 (e193b's per-fact primary = e199's MIRABEL ruler)",
    "root": "runs/checkpoints/e193b_root.pt",
    "span_meta_committed": {"n_segments": 20, "rank_eff": 20,
                            "participation_ratio": 4.284682583904991,
                            "wash_dir_removed_fraction": 0.6032087574002191},
    "bias_disclosure": "first-dead grid points OVERSTATE the band's kill-Ds => the band "
                       "median is an upper bound => MIRABEL's multiples are LOWER bounds "
                       "(biased TOWARD arrival); its dissolution is conservative",
}

REGISTERED_BARS = {
    "ARRIVALS_ROBUST": "ARRIVALS-ROBUST: \"fires if the arrival ordering (org1 "
        "earliest; the others at-or-later) is invariant under the in-span "
        "normalization AND across the bar sweep — the WHEN account stands on a "
        "common ruler.\"",
    "ARRIVALS_CONTINGENT": "ARRIVALS-CONTINGENT: \"fires if the ordering flips or "
        "dissolves under normalization or the bar sweep — the WHEN was an "
        "artifact of self-normalized ratios; the onset story rescopes to "
        "within-organism shape only.\"",
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": "multiple_t = D_kill(root,u_t)/BAND_MEDIAN(organism); "
        "BAND_MEDIAN = the middle order statistic of the organism's own in-span "
        "band kill-Ds (right-censored draws order above all resolved; n=3 with "
        "<=1 censored => censoring-robust; >=2 censored => the common axis is "
        "UNDEFINED for that organism and its arrivals read DISSOLVED); SOFT "
        "fronts get the floor multiple 3.0/BAND_MEDIAN (reported, never "
        "adjudicated); arrival(axis,bar) = first ALIVE t>=1 with axis value < "
        "bar; post-death fronts context only; ROBUST = the assignment (t1,t2,"
        "t2) in ALL SIX cells (axes {SELF,COMMON} x bars {0.60,0.70,0.80}); "
        "CONTINGENT = NOT ROBUST AND a DISSOLUTION exists (an organism "
        "that arrives in one cell fails to arrive in another — under the "
        "normalization at the same bar, or across the bar sweep "
        "within-axis: arrives at 0.80 but not at 0.60) or a FLIP (a "
        "pairwise arrival inversion under the normalization, or the "
        "SELF cell's strictly-earliest organism losing earliest-equality "
        "on COMMON); GRADED = any other "
        "partial; composite order ARRIVALS-ROBUST -> ARRIVALS-CONTINGENT -> "
        "GRADED.",
    "registered_prediction": "desk-forced to CONTINGENT-or-worse by committed "
        "arithmetic at registration (org1 t1 multiple 0.635 > 0.60; MIRABEL t2 "
        "multiple 0.903 > 0.80 — dissolutions under normalization); the fresh "
        "compute owns the HALF band and therefore half's common-axis arrival "
        "(survive / shift / flip-past-org1), the anchors, the tables, the "
        "figures. No bar shopping; the forced verdict is reported as forced.",
    "registration": "the dispatch's registration IS the registration (the bars "
        "quoted verbatim in the module docstring and here, frozen before "
        "compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "DESK+EVAL CELL: org1's and MIRABEL's bands load COMMITTED (e_chart's "
    "fine-D co-read; e193b's inspan_range); only the half lineage's band is "
    "fresh (never measured — e193's random rays are out-of-span Gaussians, "
    "all unresolved-high on this root's g-4 primary).",
    "THE CO-READ PROMOTION: e_chart's fine-D in-span thresholds were "
    "registered non-adjudicating; e205 promotes them to org1's band "
    "instrument (the critic's named repair) — the promotion is disclosed, the "
    "numbers are loaded verbatim.",
    "INSTRUMENT HETEROGENEITY (disclosed, bias directions stated): org1's "
    "band = interpolated crossings on e_chart's FINE_GRID (cap 1.5; one draw "
    "censored there); MIRABEL's band = first-dead GRID points on e193b's "
    "coarser grid (upper bounds -> MIRABEL's multiples are lower bounds, "
    "biased toward arrival); the half band = fresh interpolated crossings on "
    "EXACTLY the onset grid/instrument (the only same-instrument band).",
    "GRID-RESOLUTION CONTEXT: e193's own committed R2_SIGN bracket interpolates "
    "to 0.5271 where e197/e200's onset-grid reads gave 0.5252 (a 0.4% grid "
    "effect) — the bands' coarser grids move kill-Ds by at most this class.",
    "THE HALF-STEP CAVEAT (carried from e197/e200, load-bearing): the half "
    "lineage's FRONTS come from a COUNTERFACTUAL wash (the direction "
    "machinery verbatim at HALF the natural step; the natural step died at "
    "t=1). Its BAND, though, is a root-level property: the wash history is "
    "the root's own AdamW recipe at lr 1e-3 (e193b's convention), independent "
    "of the walk's step size — the band is the organism's, the fronts are the "
    "construction's.",
    "the f2 wash history's steps 9-20 have NO committed x-hashes (e185 stored "
    "1-8): the stream is deterministic-by-construction from seed 10902 and "
    "steps 1-8 are md5-gated; the ungated steps are disclosed, not silently "
    "trusted.",
    "threads 4 (the owner envelope's cap) where e193's chain ran threads 8: "
    "cross-thread fp32 drift at ~1e-7 is possible, so every identity gate "
    "carries e199/e200's tier system (BIT md5/1e-9 or TEXTURE at the "
    "disclosed tolerances), the achieved tier STAMPED per gate.",
    "no CE co-reads and no co-ruler walks on the fresh band rows (tiny "
    "bursts): the g-4 battery is the ruler, read at every grid point; e193b's "
    "ce_r/co-ruler co-reads are not repeated.",
    "the load-check is recorded, not gating (e204's convention).",
    "Smoke mode trims: 4-step history, 1 draw, grid {0.05, 0.5, 1.5, 3.0} "
    "(verdict stamped SMOKE; nothing adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e200_deepening.py's instruments VERBATIM (whose own provenance
# is e193_organism_replicate.py / e197_alive_window.py) + e193b_two_fact.py's
# in-span machinery (svd_basis / participation_ratio). Copied rather than
# imported to own the device policy and the arithmetic.

def load_f2(path) -> TinyGPT:
    m = TinyGPT(F2_CFG)
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
    return {"mean_pz": float(p.mean()),
            "frac_argmax_z": amax / ids.shape[0]}


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
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) — e199/e200's onset instrument."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm"] <= SHUT_BAR and a["gm"] > SHUT_BAR:
            edge = interp_d_kill(a["gm"], b["gm"], a["D"], b["D"])
            break
    return edge


def participation_ratio(sv: torch.Tensor) -> float:
    s2 = (sv.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """e_chart/e193b's Gram-based right-singular basis (fp64)."""
    H64 = H.to(torch.float64)
    G = (H64 @ H64.T)
    evals, evecs = torch.linalg.eigh(G)     # ascending
    evals = torch.flip(evals, dims=(0,)).clamp(min=0)
    evecs = torch.flip(evecs, dims=(1,))
    sv = torch.sqrt(evals)
    pr = participation_ratio(sv) if float(sv[0]) > 0 else 0.0
    rank_eff = int((sv > sv[0] * 1e-7).sum())
    rows = []
    for i in range(rank_eff):
        v = H64.T @ evecs[:, i]
        rows.append((v / sv[i].clamp(min=1e-30)).to(torch.float32))
    Vp = torch.stack(rows) if rows else torch.empty(0, H.shape[1])
    return {"sv": sv, "Vp": Vp, "pr": pr, "rank_eff": rank_eff,
            "cond": float(sv[0] / sv[-1].clamp(min=1e-30))}


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


def band_median(kill_Ds: list) -> dict:
    """The middle order statistic with right-censoring (None = unresolved-high).
    n=3 with <=1 censored => the middle RESOLVED value (censoring-robust);
    >=2 censored => UNDEFINED (the common axis dissolves for that organism)."""
    resolved = sorted([d for d in kill_Ds if d is not None])
    n_cens = sum(1 for d in kill_Ds if d is None)
    n = len(kill_Ds)
    if n_cens >= (n + 1) // 2 and n_cens >= 2:
        return {"median": None, "n": n, "n_resolved": len(resolved),
                "n_censored": n_cens, "defined": False,
                "note": "majority-censored band — the common axis is UNDEFINED "
                        "for this organism; its arrivals read DISSOLVED"}
    order = sorted([d if d is not None else float("inf") for d in kill_Ds])
    med = order[n // 2] if n % 2 == 1 else 0.5 * (order[n // 2 - 1] + order[n // 2])
    med = None if med == float("inf") else float(med)
    return {"median": med, "n": n, "n_resolved": len(resolved),
            "n_censored": n_cens, "defined": med is not None,
            "resolved_sorted": resolved}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e205_smoke" if SMOKE else "e205")
    metrics: dict = {
        "experiment": "e205_onset_norm",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never claimed)",
            "torch_threads": torch.get_num_threads(),
            "load_check_recorded_not_gating": True,
            "phases": "sequential tiny bursts; progressive writes; n=1",
        },
        "deviations": deviations,
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase"] = note
        save_json(rd / "metrics.json", metrics)
        log(f"WROTE partial metrics ({note})")

    load0 = cpu_load_probe()
    metrics["envelope"]["cpu_load_pct_at_launch"] = load0
    log(f"E205 THE ONSET NORMALIZATION (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()} — the owner "
        f"envelope's cap), load-check recorded (launch: {load0}%), "
        f"progressive writes, n=1")

    # ================= P0: the committed parents, hard-bound (Rule 12) =========
    for name, p in (("e_chart", E_CHART_METRICS), ("e193b", E193B_METRICS),
                    ("e193", E193_METRICS), ("e196", E196_METRICS),
                    ("e199", E199_METRICS), ("e200", E200_METRICS)):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    ecm = json.loads(E_CHART_METRICS.read_text(encoding="utf-8"))
    e193bm = json.loads(E193B_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    e196m = json.loads(E196_METRICS.read_text(encoding="utf-8"))
    e199m = json.loads(E199_METRICS.read_text(encoding="utf-8"))
    e200m = json.loads(E200_METRICS.read_text(encoding="utf-8"))

    # --- the onset fronts, hard-bound vs the parents
    oc99 = e199m["onset_curves"]
    assert abs(oc99["org1"][0]["D_kill"] - ORG1_ONSET["edge_u0_D_kill"]) < 1e-12
    assert abs(oc99["org1"][1]["D_kill"] - ORG1_ONSET["fronts"][1]) < 1e-12
    assert abs(oc99["org1_post_death_context"]["D_kill"]
               - ORG1_ONSET["post_death_context"][2]) < 1e-12
    assert abs(oc99["mirabel"][0]["D_kill"] - MIR_ONSET["edge_u0_D_kill"]) < 1e-12
    assert oc99["mirabel"][1]["D_kill"] is None
    assert abs(oc99["mirabel"][1]["floor_ratio"] - MIR_ONSET["t1_floor_ratio"]) < 1e-12
    assert abs(oc99["mirabel"][2]["D_kill"] - MIR_ONSET["fronts"][2]) < 1e-12
    assert abs(oc99["concentration_bar"] - 0.70) < 1e-12
    assert e199m["adjudication"]["verdict"] == "ONSET-COMMON"
    assert e199m["organisms"]["org1"]["root"] == ORG1_ONSET["root"]
    assert e199m["organisms"]["mirabel"]["root"] == MIR_ONSET["root"]
    oc20 = e200m["onset_curve"]["rows"]
    by_t = {r["t"]: r for r in oc20}
    assert abs(by_t[0]["D_kill"] - HALF_ONSET["edge_u0_D_kill"]) < 1e-9
    for t, v in HALF_ONSET["fronts"].items():
        assert abs(by_t[t]["D_kill"] - v) < 1e-9, f"e200 half t{t} drift"
    assert by_t[HALF_ONSET["death_t"]]["state"] == "death"
    assert e200m["adjudication"]["verdict"] == "GRADED"
    assert e200m["organism"]["root"] == HALF_ONSET["root"]
    assert abs(e196m["adjudication"]["constants"]["FLIGHT_RATIO_BAR"] - 0.60) < 1e-12
    e200_verdict = e200m["adjudication"]["verdict"]
    log("P0: onset fronts hard-bound vs e199 (org1 t1 0.3875 / MIRABEL t2 "
        "0.8312 / SOFT t1 floor 1.5408, verdict ONSET-COMMON) + e200 (half "
        "u0..u4, verdict GRADED) + e196 (FLIGHT_RATIO_BAR 0.60) — loaded "
        "COMMITTED, never re-run")

    # --- org1's band, hard-bound vs e_chart's fine-D co-read + root identity
    fd = ecm["partB_subspace"]["e131"]["fine_D_summary"]["inspan_thr_D"]
    assert fd == ORG1_BAND["kill_Ds"], f"e_chart inspan thr drift: {fd}"
    assert abs(ecm["partB_subspace"]["e131"]["fine_D_summary"]["wash_thr_D_loaded"]
               - ORG1_BAND["wash_g_ray_kill_context"]) < 1e-12
    assert ecm["root"] == ORG1_BAND["root"] == ORG1_ONSET["root"]
    # --- MIRABEL's band, hard-bound vs e193b + root identity
    ir = e193bm["inspan_range"]["MIRABEL"]
    assert ir["kill_Ds"] == MIR_BAND["kill_Ds"], f"e193b inspan drift: {ir}"
    assert ir["n_resolved"] == 3
    assert e193bm["cell"]["inspan"]["span"]["n_segments"] == 20
    assert e193bm["cell"]["random_seed_registry"]["seeds_used"]["inspan_draws"] \
        == [11911, 11912, 11913]
    assert e193bm["organism"]["root_md5"] == e199m["organisms"]["mirabel"].get(
        "root_flat_md5", e193bm["organism"]["root_md5"])   # identity present
    # --- e193: the f2 root identity + the g-ray anchor bracket
    assert e193m["organism"]["root_flat_md5"] == ROOT_FLAT_MD5
    assert e193m["organism"]["root"] == HALF_ONSET["root"]
    assert abs(e193m["organism"]["step_l2_measured"] - E193_STEP_L2) < 1e-12
    r1g_rows = e193m["profiles"]["R1_G"]["rows"]
    r1g_by_D = {round(r["D"], 6): r for r in r1g_rows}
    g4_015 = r1g_by_D[0.15]["gm"]
    g4_020 = r1g_by_D[0.20]["gm"]
    assert g4_015 > SHUT_BAR >= g4_020, "e193 R1_G bracket not as committed"
    gray_committed_interp = interp_d_kill(g4_015, g4_020, 0.15, 0.20)
    r2s = {round(r["D"], 6): r["gm"] for r in e193m["profiles"]["R2_SIGN"]["rows"]}
    r2s_interp = interp_d_kill(r2s[0.50], r2s[0.58], 0.50, 0.58)
    log(f"P0: bands hard-bound — org1 (e_chart fine-D, seeds 11601-3, "
        f"{ORG1_BAND['kill_Ds']}) + MIRABEL (e193b inspan_range, seeds "
        f"11911-3, {MIR_BAND['kill_Ds']}); e193 R1_G g-4 bracket interp "
        f"{gray_committed_interp:.4f} (the G_GRAY anchor), R2_SIGN grid-interp "
        f"{r2s_interp:.4f} vs e197/e200's onset-grid 0.5252 (the grid-effect "
        f"context)")

    metrics["parents_hardbound"] = {
        "e199": {"org1_edge": ORG1_ONSET["edge_u0_D_kill"],
                 "org1_t1": ORG1_ONSET["fronts"][1],
                 "org1_t2_postdeath_ctx": ORG1_ONSET["post_death_context"][2],
                 "mirabel_edge": MIR_ONSET["edge_u0_D_kill"],
                 "mirabel_t1": None, "mirabel_t2": MIR_ONSET["fronts"][2],
                 "verdict": "ONSET-COMMON"},
        "e200": {"half_edge": HALF_ONSET["edge_u0_D_kill"],
                 "half_fronts": HALF_ONSET["fronts"], "verdict": e200_verdict},
        "e196": {"flight_ratio_bar": 0.60},
        "e_chart": {"inspan_thr_D": ORG1_BAND["kill_Ds"],
                    "wash_thr_D_loaded": ORG1_BAND["wash_g_ray_kill_context"],
                    "root": ecm["root"]},
        "e193b": {"inspan_kill_Ds": MIR_BAND["kill_Ds"],
                  "span": e193bm["cell"]["inspan"]["span"]},
        "e193": {"root_flat_md5": ROOT_FLAT_MD5,
                 "step_l2": E193_STEP_L2,
                 "R1_G_g4_bracket": {"gm_015": g4_015, "gm_020": g4_020,
                                     "interp": gray_committed_interp},
                 "R2_SIGN_grid_interp_context": r2s_interp},
    }
    write_partial("P0 parents hard-bound")

    # ================= P1: the bands table (committed two + desk medians) ======
    org1_med = band_median(list(ORG1_BAND["kill_Ds"].values()))
    mir_med = band_median(MIR_BAND["kill_Ds"])
    metrics["bands"] = {
        "org1": {**{k: v for k, v in ORG1_BAND.items()},
                 "median_stat": org1_med,
                 "n_draws": 3,
                 "honesty": "n=3 draws; ONE right-censored at >1.5; the median "
                            "(the middle order statistic) is censoring-robust"},
        "mirabel": {**{k: v for k, v in MIR_BAND.items()},
                    "median_stat": mir_med,
                    "n_draws": 3,
                    "honesty": "n=3 draws, all resolved, on e193b's coarser "
                               "grid as first-dead grid points (upper bounds)"},
        "half": {"status": "FRESH — never measured before (e193's random rays "
                           "are out-of-span Gaussians, all unresolved-high on "
                           "g-4); computed in P2"},
    }
    log(f"P1: committed band medians — org1 {org1_med['median']:.4f} "
        f"(draws {org1_med['resolved_sorted']}, 1 censored), MIRABEL "
        f"{mir_med['median']:.4f} (draws {mir_med['resolved_sorted']})")
    write_partial("P1 committed bands loaded")

    # ================= P2: the fresh f2 in-span band (the tiny burst) ==========
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
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
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in READ_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in READ_GEOS},
        "pass": bool(all(list(bat_ids[j].shape) == [60, PRE + j]
                         for j in READ_GEOS)),
        "note": "install-60 battery at the e157 dial's seven read geometries "
                "— e193/e197/e200's gate verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]

    # the neutral stream (e170 VERBATIM via e185/e193/e197/e200)
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
    G_ANCHOR = {"starts": n_starts, "tries": tries, "rejections": rejections,
                "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
                "n_windows": 16, "block": BLOCK, "seed": 170}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    # the root + G_ROOT (light: md5 + the dial's g-4/g-12 cells)
    net0 = load_f2(CKPT_DIR / ROOT_CK)
    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    assert N_PARAM == F2_PARAMS, f"params {N_PARAM} != {F2_PARAMS}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    evl0 = copy.deepcopy(net0)
    root_g4 = battery_cell(evl0, bat_ids[-4], zid)["mean_pz"]
    root_g12 = battery_cell(evl0, bat_ids[-12], zid)["mean_pz"]
    rdev = max(abs(root_g4 - E157_DIAL_G4), abs(root_g12 - E157_DIAL_G12))
    G_ROOT = {
        "cells_measured": {"g-4": root_g4, "g-12": root_g12},
        "cells_committed": {"g-4": E157_DIAL_G4, "g-12": E157_DIAL_G12},
        "max_abs_diff": rdev, "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
        "bit": bool(rdev < G_BIT_TOL),
        "flat_md5": root_md5,
        "flat_md5_match_e193_committed": bool(root_md5 == ROOT_FLAT_MD5),
        "pass": bool(rdev < G_FALLBACK_TOL and root_md5 == ROOT_FLAT_MD5),
        "note": "e200's G_ROOT in light form: the flat md5 must match e193's "
                "committed root identity + the e157 dial's g-4/g-12 cells "
                "bit-tight (or the disclosed texture tier)",
    }
    log(f"G_ROOT: g-4 {root_g4:.10f} / g-12 {root_g12:.10f} (max|d| {rdev:.2e}, "
        f"md5 {'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}): "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("family-2 root gate FAILED vs e193/e157 identity")
    del evl0

    # G_STREAM: the seed-10902 stream, steps 1-8 md5 vs e185's stored hashes
    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 9):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_8_md5_match": stream_ok,
                "steps_9_20": "no committed hashes (e185 stored 1-8) — "
                              "deterministic-by-construction, disclosed",
                "pass": bool(all(stream_ok.values()))}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..8): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                        "G_ROOT": G_ROOT, "G_STREAM": G_STREAM}
    write_partial("P2a protocol gates (namefree/splice/battery/anchor/root/stream)")

    # ---- G_DIRCK + the fresh t=0 g-ray (the G_GRAY anchor object) -------------
    wh_gen = torch.Generator().manual_seed(FREEZE_SEED)
    aj = torch.randint(16, (ANCH_BS,), generator=wh_gen)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=wh_gen)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
    x1 = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y1 = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    assert x1_md5 == E185_XHASH[1], "step-1 batch diverged from the licensed stream"

    gnet = copy.deepcopy(net0)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1)
    F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                    y1.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(gnet.parameters(), 1.0)
    g0 = torch.cat([p.grad.detach().reshape(-1)
                    for p in gnet.parameters()]).clone()   # post-clip
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g
    u_g = (g0 / torch.norm(g0)).clone()
    u_g_md5 = hashlib.md5(u_g.numpy().tobytes()).hexdigest()

    uck = torch.load(CKPT_DIR / DIR_CK, map_location="cpu", weights_only=False)
    ug_maxdiff = float((u_g - uck["u"].float()).abs().max())
    G_DIRCK = {
        "path": str(CKPT_DIR / DIR_CK),
        "meta_gate": {"experiment": uck["meta"].get("experiment"),
                      "root": uck["meta"].get("root"),
                      "match": bool(uck["meta"].get("experiment") == "e193"
                                    and uck["meta"].get("root") == ROOT_CK)},
        "fresh_u_md5": u_g_md5,
        "md5_match": bool(hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest() == u_g_md5),
        "max_abs_diff_fresh_vs_ckpt": ug_maxdiff,
        "cos64_fresh_vs_ckpt": cos64(u_g, uck["u"].float()),
        "theta0_md5_match_root": bool(uck.get("theta0_md5") == root_md5),
        "note": "e200's G_DIRCK verbatim: the committed u must BE the fresh "
                "t=0 g-ray (md5, or the cross-thread TEXTURE fallback "
                "coordinate-wise within 1e-6); theta0_md5 must be THIS root's "
                "flat md5",
    }
    G_DIRCK["tier"] = ("BIT" if G_DIRCK["md5_match"] else "TEXTURE")
    G_DIRCK["pass"] = bool(G_DIRCK["meta_gate"]["match"]
                           and (G_DIRCK["md5_match"] or ug_maxdiff < 1e-6)
                           and G_DIRCK["theta0_md5_match_root"])
    log(f"G_DIRCK: md5 {'match' if G_DIRCK['md5_match'] else 'DRIFT'} "
        f"(tier {G_DIRCK['tier']}, max|d| {ug_maxdiff:.2e}), theta0_md5 "
        f"{'match' if G_DIRCK['theta0_md5_match_root'] else 'DRIFT'}: "
        + ("PASS" if G_DIRCK["pass"] else "FAIL"))
    if not G_DIRCK["pass"]:
        raise RuntimeError("g-ray direction checkpoint gate FAILED")
    metrics["gates"]["G_DIRCK"] = G_DIRCK
    write_partial("P2b G_DIRCK (the fresh t=0 g-ray = e193's committed direction)")

    # ---- the wash history: 20 CPU AdamW steps on the licensed stream ----------
    # e193b's in-span machinery VERBATIM (its wash-history loop), this root:
    # the loop RE-SEEDS its generator (the stream restarts at the root — the
    # g-ray block above consumed a draw pair from a separate generator).
    wh_gen = torch.Generator().manual_seed(FREEZE_SEED)
    wh_net = copy.deepcopy(net0)
    wh_net.train()
    wh_opt = torch.optim.AdamW(wh_net.parameters(), lr=LR_ADAMW,
                               betas=(0.9, 0.95), weight_decay=0.1)
    wh_theta = theta0.clone()
    hist_disp, hist_l2, hist_xh = [], [], {}
    t0h = time.time()
    for s_wh in range(1, WASH_HIST_STEPS + 1):
        aj = torch.randint(16, (ANCH_BS,), generator=wh_gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=wh_gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        hist_xh[s_wh] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        if s_wh <= 8:
            assert hist_xh[s_wh] == E185_XHASH[s_wh]
        logits_w, _ = wh_net(x)
        lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                             y.reshape(-1))
        wh_opt.zero_grad(set_to_none=True)
        lw.backward()
        torch.nn.utils.clip_grad_norm_(wh_net.parameters(), 1.0)
        wh_opt.step()
        th_new = flat_params(wh_net)
        hist_disp.append(th_new - wh_theta)
        hist_l2.append(float(torch.norm(th_new - wh_theta)))
        wh_theta = th_new
        log(f"  [hist s{s_wh}] disp L2 {hist_l2[-1]:.6f} "
            f"ce {float(lw.item()):.4f}")
    wh_net.eval()
    del wh_net, wh_opt
    G_HIST = {
        "n_steps": WASH_HIST_STEPS,
        "step1_L2_measured": hist_l2[0],
        "step1_L2_committed_e193": E193_STEP_L2,
        "step1_L2_abs_diff": abs(hist_l2[0] - E193_STEP_L2),
        "tol": 0.05, "tier": "TEXTURE (threads 4 vs e193's 8)",
        "pass": bool(abs(hist_l2[0] - E193_STEP_L2) < 0.05),
        "xhash_steps_1_8_gated": True,
        "disp_L2_all": hist_l2,
        "cum_disp_final": float(hist_l2 and np.sum(hist_l2)),
        "note": "e193b's wash-history convention at THIS root: fresh AdamW "
                "(lr 1e-3, betas 0.9/0.95, wd 0.1, clip 1.0, batch 16 anchor "
                "+ 16 random, seed-10902 stream); the step-1 displacement L2 "
                "must reproduce e193's committed MEASURED step L2 (never "
                "ported); steps 9-20 ungated (no committed hashes)",
    }
    log(f"G_HIST: step-1 L2 {hist_l2[0]:.10f} vs committed {E193_STEP_L2:.10f} "
        f"(|d| {G_HIST['step1_L2_abs_diff']:.2e}): "
        + ("PASS" if G_HIST["pass"] else "FAIL"))
    if not G_HIST["pass"]:
        raise RuntimeError("wash-history step-1 anchor FAILED")
    metrics["gates"]["G_HIST"] = G_HIST
    write_partial("P2c wash history (20 CPU AdamW steps, step-1 anchor gated)")

    # ---- the span (Gram SVD, e193b/e_chart verbatim) ---------------------------
    H = torch.stack(hist_disp)
    basis = svd_basis(H)
    Vr = basis["Vp"].contiguous()               # (r, N)
    Vr64 = Vr.to(torch.float64)
    r_avail = basis["rank_eff"]
    removed = float(torch.norm(Vr64 @ u_g.to(torch.float64)))
    span_meta = {
        "n_segments": int(Vr.shape[0]), "rank_eff": r_avail,
        "participation_ratio": basis["pr"], "cond": basis["cond"],
        "wash_dir_removed_fraction": removed,
        "e193b_committed_context": MIR_BAND["span_meta_committed"],
        "note": "e193b's convention verbatim at this root: the span of "
                "REALIZED wash steps (Adam's sign-flattened displacements — "
                "e_chart's caveat carried); this root's OWN history, never a "
                "ported span",
    }
    log(f"  span: {span_meta['n_segments']} segments, rank {r_avail}, PR "
        f"{basis['pr']:.2f}, cond {basis['cond']:.1f}; ||Vr^T u_g|| = "
        f"{removed:.4f}")
    del H, hist_disp
    metrics["bands"]["half"]["span"] = span_meta
    write_partial("P2d span built (Gram SVD)")

    # ---- the band walks: 3 fresh draws + the g-ray anchor ----------------------
    evl = copy.deepcopy(net0)
    evl.eval()

    def band_walk(u_vec: torch.Tensor, tag: str) -> dict:
        rows = []
        for D in [0.0] + list(D_GRID):
            thD = theta0 - D * u_vec
            load_flat(evl, thD)
            evl.eval()
            gz = battery_cell(evl, primary_ids, zid)
            rows.append({"D": float(D),
                         "rms": float(D) / 934.597239456655,
                         "gm": gz["mean_pz"],
                         "frac_argmax_z": gz["frac_argmax_z"],
                         "disp_check": float(torch.norm(thD - theta0))})
        kill = profile_d_kill(rows)
        log(f"  {tag}: D_kill {kill if kill is not None else 'SOFT (>3.0)'}")
        return {"rows": rows, "D_kill": kill}

    # G_GRAY first: the fresh g-ray anchors the instrument vs e193's committed
    # R1_G g-4 bracket (interp 0.1967, texture tol 0.02)
    g_anchor = band_walk(u_g, "G_GRAY (t=0 g-ray anchor)")
    G_GRAY = {
        "fresh_u_md5": u_g_md5,
        "fresh_D_kill_interp": g_anchor["D_kill"],
        "e193_R1_G_bracket": {"gm_015": g4_015, "gm_020": g4_020},
        "e193_R1_G_interp_committed": gray_committed_interp,
        "abs_diff": (abs(g_anchor["D_kill"] - gray_committed_interp)
                     if g_anchor["D_kill"] is not None else None),
        "tol": 0.02, "tier": "TEXTURE (e193 ran threads 8; grid interp vs "
                             "this cell's 0.05-grid interp)",
        "pass": bool(g_anchor["D_kill"] is not None
                     and abs(g_anchor["D_kill"] - gray_committed_interp) < 0.02
                     and 0.15 < g_anchor["D_kill"] <= 0.21),
        "note": "THE BAND-INSTRUMENT ANCHOR (Rule 12): the fresh t=0 g-ray "
                "walked down THIS cell's band grid must reproduce e193's "
                "committed R1_G kill bracket (first-dead 0.20 on e193's grid; "
                "interpolated 0.1967) — the band's kill-Ds are not believed "
                "until this passes",
    }
    log(f"G_GRAY: fresh {g_anchor['D_kill']:.4f} vs committed interp "
        f"{gray_committed_interp:.4f} (|d| {G_GRAY['abs_diff']:.4f}): "
        + ("PASS" if G_GRAY["pass"] else "FAIL"))
    if not G_GRAY["pass"] and not SMOKE:
        raise RuntimeError("g-ray band anchor FAILED vs e193's committed R1_G")
    if SMOKE:
        G_GRAY["note"] += " — SMOKE: coarse grid, gate reported not enforced"
    metrics["gates"]["G_GRAY"] = {k: v for k, v in G_GRAY.items()}
    metrics["bands"]["half"]["g_ray_anchor"] = {
        "D_kill": g_anchor["D_kill"], "committed_interp": gray_committed_interp}
    write_partial("P2e G_GRAY anchor passed (the band instrument is believed)")

    # the 3 fresh in-span draws
    half_kills, half_draws = [], {}
    for sd in INSPAN_SEEDS:
        g_ = torch.Generator().manual_seed(sd)
        c = torch.randn(r_avail, generator=g_, dtype=torch.float64)
        v = Vr64.T @ c
        u_in = (v / torch.norm(v)).to(torch.float32)
        walk = band_walk(u_in, f"INSPAN s{sd}")
        half_draws[f"s{sd}"] = {
            "u_md5": hashlib.md5(u_in.numpy().tobytes()).hexdigest(),
            "cos_to_wash_dir": cos64(u_in, u_g),
            "D_kill": walk["D_kill"],
            "rows": walk["rows"],
        }
        half_kills.append(walk["D_kill"])
        metrics["bands"]["half"]["draws"] = half_draws
        metrics["bands"]["half"]["kill_Ds_fresh"] = list(half_kills)
        write_partial(f"P2f in-span draw s{sd} complete "
                      f"(kills so far {half_kills})")
        load_now = cpu_load_probe()
        metrics["envelope"][f"cpu_load_pct_after_draw_s{sd}"] = load_now

    half_med = band_median(half_kills)
    metrics["bands"]["half"].update({
        "seeds": list(INSPAN_SEEDS),
        "kill_Ds": list(half_kills),
        "median_stat": half_med,
        "n_draws": len(half_kills),
        "instrument": "FRESH: onset grid 0.05..3.00 + D=0 anchor, kill = first "
                      "g-4 <= 0.27 downcrossing INTERPOLATED linear-in-D "
                      "(e199/e200's onset instrument VERBATIM) — the only "
                      "band on exactly the onset instrument",
        "ruler": "install-60 g-4 (e193's frozen primary; the SAME ruler as "
                 "the half lineage's onset fronts)",
        "root": "runs/checkpoints/e157_f2_consolidated.pt",
        "honesty": f"n={len(half_kills)} draws of ONE 20-step wash-history "
                   f"realization (seed 10902); None = unresolved-high "
                   f"(right-censored at 3.0)",
    })
    log(f"P2: half band kills {half_kills} -> median "
        + (f"{half_med['median']:.4f}" if half_med["defined"]
           else "UNDEFINED (majority censored)"))
    write_partial("P2 complete (the fresh half band)")

    # ================= P3: the normalization arithmetic + the sweeps ===========
    def build_curves(onset: dict, median: float | None) -> dict:
        """The per-t table on both axes + the arrival function inputs."""
        edge = onset["edge_u0_D_kill"]
        rows = []
        max_t = max(list(onset["fronts"]) + [onset["death_t"],
                                             *onset.get("post_death_context",
                                                        {}).keys()])
        for t in range(0, max_t + 1):
            row = {"t": t}
            if t == 0:
                row.update({"D_kill": edge, "self_ratio": 1.0,
                            "self_note": "definitional (u0 vs itself)"})
                if median is not None:
                    row["band_multiple"] = edge / median
                    row["band_note"] = ("the EDGE's own multiple (a real "
                                        "number on the common axis — the "
                                        "self axis hides it inside ratio_0 "
                                        "= 1.0)")
                else:
                    row["band_multiple"] = None
                rows.append(row)
                continue
            dk = onset["fronts"].get(t, "DEAD")
            if dk == "DEAD":
                pd = onset.get("post_death_context", {}).get(t)
                if pd is not None:
                    row.update({"state": "post_death_context",
                                "D_kill": pd, "self_ratio": pd / edge,
                                "band_multiple": (pd / median
                                                  if median is not None
                                                  else None),
                                "note": "e195's committed POST-DEATH read — "
                                        "plotted open, NEVER adjudicated"})
                else:
                    row.update({"state": "death", "D_kill": None,
                                "self_ratio": None, "band_multiple": None})
                rows.append(row)
                continue
            if dk is None:   # SOFT
                row.update({"state": "alive_SOFT",
                            "D_kill": None,
                            "self_ratio_floor": 3.0 / edge,
                            "band_multiple_floor": (3.0 / median
                                                    if median is not None
                                                    else None),
                            "note": "SOFT (unresolved-high within [0,3.0]) — "
                                    "floors reported, never adjudicated"})
            else:
                row.update({"state": "alive",
                            "D_kill": dk, "self_ratio": dk / edge,
                            "band_multiple": (dk / median
                                              if median is not None
                                              else None)})
            rows.append(row)
        return {"rows": rows, "median": median, "edge": edge}

    curves = {
        "org1": build_curves(ORG1_ONSET, org1_med["median"]),
        "mirabel": build_curves(MIR_ONSET, mir_med["median"]),
        "half": build_curves(HALF_ONSET, half_med["median"]),
    }

    def arrival(curve: dict, axis: str, bar: float):
        """First ALIVE t>=1 with axis value < bar; SOFT/death never arrive."""
        for row in curve["rows"]:
            if row["t"] < 1 or row.get("state") not in ("alive",):
                continue
            v = row.get(axis)
            if v is not None and v < bar:
                return row["t"]
        return None    # DISSOLVED

    sweep = {}
    for axis, key in (("SELF", "self_ratio"), ("COMMON", "band_multiple")):
        sweep[axis] = {}
        for bar in BARS:
            cell = {}
            for org, curve in curves.items():
                a = arrival(curve, key, bar)
                cell[org] = a
            sweep[axis][f"{bar:.2f}"] = cell
            log(f"  arrival[{axis}, {bar:.2f}]: org1 "
                f"t{cell['org1'] if cell['org1'] else 'DISSOLVED'} | MIRABEL "
                f"t{cell['mirabel'] if cell['mirabel'] else 'DISSOLVED'} | "
                f"half t{cell['half'] if cell['half'] else 'DISSOLVED'}")

    # ---- the adjudication (frozen clauses) --------------------------------------
    def assign(cell: dict) -> tuple:
        return (cell["org1"], cell["mirabel"], cell["half"])

    committed_tuple = (COMMITTED_ASSIGNMENT["org1"],
                       COMMITTED_ASSIGNMENT["mirabel"],
                       COMMITTED_ASSIGNMENT["half"])
    robust_all = all(assign(sweep[a][f"{b:.2f}"]) == committed_tuple
                     for a in ("SELF", "COMMON") for b in BARS)

    flips, dissolves = [], []
    # (a) DISSOLUTIONS under the normalization (same bar, SELF vs COMMON)
    for bar in BARS:
        s_cell = sweep["SELF"][f"{bar:.2f}"]
        c_cell = sweep["COMMON"][f"{bar:.2f}"]
        for org in ("org1", "mirabel", "half"):
            if s_cell[org] is not None and c_cell[org] is None:
                dissolves.append({"kind": "under normalization",
                                  "bar": bar, "organism": org,
                                  "self_t": s_cell[org]})
    # (b) DISSOLUTIONS across the bar sweep (within-axis, extremes 0.80 -> 0.60)
    for axis in ("SELF", "COMMON"):
        hi = sweep[axis]["0.80"]
        lo = sweep[axis]["0.60"]
        for org in ("org1", "mirabel", "half"):
            if hi[org] is not None and lo[org] is None:
                dissolves.append({"kind": "across the bar sweep",
                                  "axis": axis, "organism": org,
                                  "t_at_0.80": hi[org]})
    # (c) FLIPS under the normalization (same bar, SELF vs COMMON)
    for bar in BARS:
        s_cell = sweep["SELF"][f"{bar:.2f}"]
        c_cell = sweep["COMMON"][f"{bar:.2f}"]
        s_arr = [(t, o) for o, t in s_cell.items() if t is not None]
        c_arr = [(t, o) for o, t in c_cell.items() if t is not None]
        s_earliest_set = ({o for t, o in s_arr
                           if t == min(t_ for t_, _ in s_arr)}
                          if s_arr else set())
        c_earliest_set = ({o for t, o in c_arr
                           if t == min(t_ for t_, _ in c_arr)}
                          if c_arr else set())
        if s_earliest_set and c_earliest_set and \
                not (s_earliest_set & c_earliest_set):
            flips.append({"bar": bar, "kind": "earliest-identity flip",
                          "self_earliest": sorted(s_earliest_set),
                          "common_earliest": sorted(c_earliest_set)})
        s_order = {o: t for t, o in s_arr}
        c_order = {o: t for t, o in c_arr}
        for o1 in s_order:
            for o2 in s_order:
                if o1 < o2 and s_order[o1] < s_order[o2]:
                    if o1 in c_order and o2 in c_order \
                            and c_order[o1] > c_order[o2]:
                        flips.append({"bar": bar, "kind": "pairwise inversion",
                                      "pair": [o1, o2],
                                      "self": [s_order[o1], s_order[o2]],
                                      "common": [c_order[o1], c_order[o2]]})

    contingent = (not robust_all) and (len(flips) > 0 or len(dissolves) > 0)
    if robust_all:
        verdict = "ARRIVALS-ROBUST"
    elif contingent:
        verdict = "ARRIVALS-CONTINGENT"
    else:
        verdict = "GRADED"

    # the desk-forced disclosure (committed arithmetic, stated at registration)
    desk = {
        "org1_t1_multiple_desk": ORG1_ONSET["fronts"][1] / org1_med["median"],
        "mirabel_t2_multiple_desk": MIR_ONSET["fronts"][2] / mir_med["median"],
        "statement": "org1's t1 multiple (0.635) sits ABOVE the 0.60 bar and "
                     "MIRABEL's t2 multiple (0.903) above ALL THREE bars — "
                     "both desk-computable from committed data at "
                     "registration; ARRIVALS-ROBUST was already impossible "
                     "and the verdict desk-forced to CONTINGENT-or-worse; "
                     "reported as forced, the fresh compute owns the half "
                     "band and its arrival",
    }

    metrics["normalization"] = {
        "org1": curves["org1"], "mirabel": curves["mirabel"],
        "half": curves["half"],
        "band_medians": {"org1": org1_med["median"],
                         "mirabel": mir_med["median"],
                         "half": half_med["median"]},
        "edge_multiples": {
            "org1": ORG1_ONSET["edge_u0_D_kill"] / org1_med["median"],
            "mirabel": MIR_ONSET["edge_u0_D_kill"] / mir_med["median"],
            "half": (HALF_ONSET["edge_u0_D_kill"] / half_med["median"]
                     if half_med["median"] else None)},
    }
    metrics["arrival_sweep"] = {
        "axes": {"SELF": "ratio vs own u0 edge (the committed instrument)",
                 "COMMON": "D_kill / own in-span band median (the critic's "
                           "proposed ruler)"},
        "bars": list(BARS),
        "bars_provenance": {"0.60": "e196's committed FLIGHT_RATIO_BAR",
                            "0.70": "e198/e199/e200's frozen concentration bar",
                            "0.80": "the sweep's upper arm"},
        "matrix": sweep,
        "committed_assignment_T163_T164": COMMITTED_ASSIGNMENT,
    }
    metrics["adjudication"] = {
        "bars": {
            "ARRIVALS_ROBUST": {
                "fires": robust_all,
                "detail": {"all_six_cells": {a: sweep[a] for a in sweep},
                           "required": str(committed_tuple)},
            },
            "ARRIVALS_CONTINGENT": {
                "fires": contingent,
                "detail": {"dissolutions": dissolves, "flips": flips},
            },
            "GRADED": {"fires": verdict == "GRADED"},
        },
        "verdict": verdict,
        "desk_forced_disclosure": desk,
        "composite_order": "ARRIVALS-ROBUST -> ARRIVALS-CONTINGENT -> GRADED",
    }
    write_partial(f"P3 adjudicated — verdict {verdict}")

    # ================= P4: figures ==============================================
    # (1) the common-axis onset curves
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = {"org1": "#1f77b4", "mirabel": "#d62728", "half": "#2ca02c"}
    labels = {"org1": "org1 (e131, g-12)", "mirabel": "MIRABEL (e193b, g-12)",
              "half": "half-step lineage (e157_f2, g-4)"}
    for org, curve in curves.items():
        rows = curve["rows"]
        ts, ms = [], []
        for r in rows:
            st = r.get("state")
            if r["t"] == 0:
                ax.plot(r["t"], r["band_multiple"], marker="s", ms=9,
                        color=colors[org], alpha=0.55)
                ax.annotate(f"edge {r['band_multiple']:.2f}x",
                            (r["t"], r["band_multiple"]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=colors[org], alpha=0.8)
                continue
            if st == "alive":
                ts.append(r["t"])
                ms.append(r["band_multiple"])
            elif st == "alive_SOFT":
                ax.plot(r["t"], r["band_multiple_floor"], marker="v", ms=9,
                        mfc="none", color=colors[org])
                ax.annotate(f"SOFT >{r['band_multiple_floor']:.2f}x",
                            (r["t"], r["band_multiple_floor"]),
                            textcoords="offset points", xytext=(4, 6),
                            fontsize=8, color=colors[org])
            elif st == "post_death_context":
                ax.plot(r["t"], r["band_multiple"], marker="D", ms=8,
                        mfc="none", color=colors[org])
                ax.annotate(f"u2 ctx {r['band_multiple']:.2f}x",
                            (r["t"], r["band_multiple"]),
                            textcoords="offset points", xytext=(4, -12),
                            fontsize=8, color=colors[org], alpha=0.8)
            elif st == "death":
                dt = r["t"]
                ax.axvline(dt - 0.02, color=colors[org], ls=":", lw=1.4,
                           alpha=0.7)
                ax.annotate("death", (dt - 0.02, ax.get_ylim()[1] * 0.55),
                            rotation=90, fontsize=8, color=colors[org],
                            alpha=0.8, ha="right")
        if ts:
            ax.plot(ts, ms, marker="o", ms=7, lw=1.8, color=colors[org],
                    label=labels[org])
            for t_, m_ in zip(ts, ms):
                ax.annotate(f"{m_:.3f}x", (t_, m_),
                            textcoords="offset points", xytext=(5, -3),
                            fontsize=8, color=colors[org])
    for bar, sty in zip(BARS, ("--", "-.", ":")):
        ax.axhline(bar, color="gray", ls=sty, lw=1.1, alpha=0.8)
        ax.annotate(f"bar {bar:.2f}", (0.01, bar), fontsize=8,
                    color="gray", va="bottom")
    ax.axhline(1.0, color="black", lw=1.6, alpha=0.85)
    ax.annotate("band median (parity: 1.0x = a median random in-span "
                "direction's kill-D)", (0.01, 1.0), fontsize=8.5,
                va="bottom", color="black")
    ax.set_yscale("log")
    ax.set_xticks(range(0, 5))
    ax.set_xlabel("t (walk steps; fronts = sign(g_t) rays mapped from the root)")
    ax.set_ylabel("D_kill / own in-span band median  (log scale)")
    ax.set_title(f"E205 — the onset curves on the COMMON axis (band multiples)"
                 f"\nverdict: {verdict} — bands n=3 draws each; "
                 f"open markers = SOFT floors / post-death context")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(rd / "e205_common_axis.png", dpi=130)
    plt.close(fig)
    log("figure: e205_common_axis.png")

    # (2) the bar-sweep arrival matrix
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for axi, axis in zip(axes, ("SELF", "COMMON")):
        axi.set_axis_off()
        tbl = [["", "org1", "MIRABEL", "half"]]
        for bar in BARS:
            cell = sweep[axis][f"{bar:.2f}"]
            tbl.append([f"bar {bar:.2f}"] + [
                (f"t{cell[o]}" if cell[o] is not None else "DISSOLVED")
                for o in ("org1", "mirabel", "half")])
        t = axi.table(cellText=tbl[1:], colLabels=tbl[0], loc="center",
                      cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.7)
        axi.set_title(f"{axis} axis — "
                      + ("ratio vs own u0 edge" if axis == "SELF"
                         else "band multiple"), fontsize=10)
        for (r_, c_), cell in t.get_celld().items():
            if r_ > 0 and cell.get_text().get_text() == "DISSOLVED":
                cell.set_facecolor("#f2b8b5")
            if r_ > 0 and cell.get_text().get_text().startswith("t"):
                cell.set_facecolor("#cfe8cf")
    fig.suptitle(f"E205 — the arrival bar-sweep (committed assignment "
                 f"T163/T164: org1 t1, MIRABEL t2, half t2) — verdict: "
                 f"{verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(rd / "e205_bar_sweep.png", dpi=130)
    plt.close(fig)
    log("figure: e205_bar_sweep.png")

    # ================= P5: honesty + final write ================================
    metrics["honesty"] = {
        "bands_own_ns": "every band is n=3 draws of ONE 20-step wash-history "
                        "realization; org1's carries 1-of-3 right-censored at "
                        ">1.5 (the median is censoring-robust); MIRABEL's are "
                        "first-dead GRID points (upper bounds — its multiples "
                        "are lower bounds, biased TOWARD arrival, so its "
                        "dissolution is conservative); the half band is the "
                        "only one on exactly the onset instrument",
        "grid_heterogeneity": "org1's band grid caps at 1.5 (e_chart's "
                              "FINE_GRID); MIRABEL's rides e193b's 27-point "
                              "grid; e193's own R2_SIGN bracket interpolates "
                              "to 0.5271 where the onset grid reads 0.5252 — "
                              "a 0.4% grid class",
        "the_half_step_caveat": "the half lineage's FRONTS belong to the "
                                "counterfactual half-step construction "
                                "(e197/e200's carried deviation); its BAND is "
                                "root-level (the root's own AdamW wash recipe "
                                "— step-size-independent)",
        "single_realizations": "every onset curve is one realized walk; every "
                               "band is one span realization; nothing here is "
                               "a distribution — the tables are the objects",
        "logits_prediction_check": "the bands and fronts are behavior reads "
                                   "(battery p(Z) downcrossings), not logits; "
                                   "no intervening was done in this cell (the "
                                   "onset fronts are committed; the band is a "
                                   "reference instrument, not an intervention)",
        "nothing_guaranteed": "the openness is the point: the half band "
                              "measured here is the first of its kind on this "
                              "root; a redraw (fresh history seed, fresh "
                              "draws) would move the band median by the 2-3x "
                              "draw lottery T155 committed",
    }
    metrics["provenance"] = {
        "onset_fronts": {"org1": "runs/e199/metrics.json (hard-bound; e195/"
                                 "e199's gated chain)",
                         "mirabel": "runs/e199/metrics.json (hard-bound; "
                                    "e198's committed crosscheck rides)",
                         "half": "runs/e200/metrics.json (hard-bound; e197's "
                                 "committed chain rides transitively)"},
        "bands": {"org1": "runs/e_chart/metrics.json partB.e131 "
                          "fine_D_summary (loaded verbatim; the co-read "
                          "promotion disclosed)",
                  "mirabel": "runs/e193b/metrics.json inspan_range.MIRABEL "
                             "(loaded verbatim)",
                  "half": "THIS CELL (fresh; parents = the gated f2 root + "
                          "the gated stream + G_HIST + G_DIRCK + G_GRAY)"},
        "bars_provenance": {"0.60": "runs/e196/metrics.json "
                                    "FLIGHT_RATIO_BAR (hard-bound)",
                            "0.70": "e198/e199/e200's frozen concentration "
                                    "bar (hard-bound in e199)"},
        "seeds": {"fresh_inspan_draws": list(INSPAN_SEEDS),
                  "wash_stream": FREEZE_SEED,
                  "registry_note": "12001-3 fresh for e205 (repo-grep'd); "
                                   "e193b's 11911-3 and e_chart's 11601-3 "
                                   "load committed"},
    }
    metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)" if not SMOKE
                         else "SMOKE — nothing adjudicated")
    save_json(rd / "metrics.json", metrics)
    log(f"DONE -> {rd / 'metrics.json'}  verdict: {verdict}")


if __name__ == "__main__":
    main()
