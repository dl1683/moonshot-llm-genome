"""E211 — THE WALLED-BAND QUESTION (e209's free find, interrogated).

WHY: e209 (runs/e209/metrics.json, T179) found the walled roots'
in-span noise bands 2-3x WIDER than the pristine family's (the walled
band medians 1.580 / 1.706 / 0.914 vs the pristine-family bands
0.610-0.920) — the noise ball GROWS under the wall. T179's open
question, verbatim: "the walled history leaves the net more
displacement-tolerant in random directions? or the settled mid-stride
states carry a wider functional noise floor?" This cell splits that
question against three candidates: (a) THE SETTLED-STATE EXPLANATION —
the walled s300 states sit at mid-stride positions (d_raw ~1.46) whose
loss landscape is FLATTER in the wash-span directions; (b) THE HISTORY
EXPLANATION — the walled wash histories span a WIDER subspace, so the
band (draws inside that span) is bigger by construction; (c) NEITHER —
an artifact of the band-instrument at mid-stride states.

WHAT IT BUILDS ON: e209's cell machinery VERBATIM (the settled-state
load, the 20-step contiguous unwalled AdamW wash history, the Gram-SVD
span, the onset-grid 0.05..3.00 kill instrument, the install-60 g-12
ruler, the 0.27 DISSOLVE bar); e209's committed per-root records (the
settled-state md5s, the committed span PRs, the committed bands); the
e193b/e205 band lineage (participation ratio, spectrum conventions);
e_chart's committed pristine-root records (the root read 0.9156, the
committed e131 spectrum, the committed pristine in-span band); g1b's
committed C-arm wash (the seed-10902 stream at the pristine root —
300 per-step rows: the pristine history's own anchor bank). NOTHING
from the parents is re-adjudicated; every committed number loads and
hard-binds at its metrics path.

WHAT IS NEW: the two instruments the question needs. (1) THE
SPAN-WIDTH COMPARISON — each root's OWN contiguous wash-history span
spectrum (top-20 SVs, PR, normalized profile), same machinery at all
four states (the pristine span is FRESH — e_chart's committed e131
spectrum is a MIXED-lr checkpoint ladder, not a contiguous wash:
cross-instrument, context only); robustness = 2 fresh history
realizations per root (seeds 12401-12408, registry-clean). (2) THE
CURVATURE CO-READ — finite-difference second derivatives along the
top-4 (bar) / top-8 (correlation) span directions at each state, on
the FIXED wash-batch CE (primary) and the ruler battery's -log pZ
(co-read), eps grid {2e-3, 5e-3, 1e-2}. (3) THE BAND RE-READ —
per-direction kill-Ds along the top-8 span directions (the same onset
instrument as e209's bands), correlated with each direction's SV
energy and curvature; the committed e209/e_chart bands joined per root
(context).

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - FLATTENED-BASIN: "fires if the walled states' curvatures along the
    span directions are materially lower than the pristine's (>= 2x)
    while the span spectra match — the wall flattened the basin (a);
    a new wall property: the ball buys noise-tolerance by flattening."
  - WIDER-EXPLORATION: "fires if the walled spans are intrinsically
    wider (the SV spectra/PR materially larger) — the projection kept
    the trajectory exploring (b); the band is the history's shadow."
  - GRADED: "any partial/mix — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * states = the committed bodies: the pristine e131 root
    (runs/checkpoints/e131_consolidated_e113.pt, used AS committed —
    no anch__ ball of its own) + the three walled s300 roots
    (g1b_W1_s300 / g1bR_W1_10907_s300 / g1bR_W1_10908_s300), each
    SETTLED onto its ball (e209's settled convention; the settled
    flats md5-gate against e209's committed settled_flat_md5s).
  * span (primary) = each root's OWN contiguous 20-step unwalled
    AdamW wash (lr 1e-3, betas (0.9,0.95), wd 0.1, clip 1.0, batch
    32 = 16 neutral-anchor + 16 random, e209's draw order verbatim)
    from its seed stream (walled: the root's own seed = e209's stream
    -> the span PR MUST reproduce e209's committed PR within 1e-6
    relative, else G_HIST fails; pristine: the seed-10902 stream AT
    THE E131 ROOT = g1b's own C-arm stream, every step's displacement
    L2 gated against g1b's committed C-arm rows 1..20, tol 5e-3
    cross-device). Spectrum = the 20 Gram-SVD singular values (fp64);
    PR = (sum sv^2)^2 / sum sv^4; profile = sv_i / sv_1.
  * span (robustness) = 2 fresh streams per root (12401-12408);
    reported as the realization spread, NEVER adjudicated.
  * SPECTRA-MATCH clause (for FLATTENED-BASIN): all three walled
    primary PRs within [0.8, 1.25]x the pristine primary PR AND the
    mean over the three walled roots of mean_i |log2[(sv_i/sv_1)_walled
    / (sv_i/sv_1)_pristine]| <= 0.5 bits (the shape clause).
  * WIDER clause (for WIDER-EXPLORATION): >= 2 of 3 walled primary
    PRs >= 1.25x the pristine primary PR.
  * curvature: v_i = the i-th unit-L2 right-singular vector of the
    primary span; the bar's directions = top-4 (the correlation's =
    top-8); L_wash = cross-entropy on the FIXED primary-history
    step-1 batch (the PRIMARY loss — the basin the wash actually
    explores); L_rul = mean(-log pZ) on the install-60 g-12 battery
    (the ruler's own quantity; co-read, never adjudicated);
    curv(v, eps) = [L(theta + eps v) + L(theta - eps v) - 2 L(theta)]
    / eps^2, eps in {2e-3, 5e-3, 1e-2}, PRIMARY eps = 1e-2 (the FD
    second difference's SNR against float32 roundoff grows with eps;
    the estimate reads curvature at the 1e-2 L2 scale — quartic
    contamination possible, disclosed); per-root curvature = MEDIAN
    over the top-4 directions of curv(v, 1e-2) on L_wash.
  * FLATTENED-BASIN curvature clause: the geometric mean over the
    three walled roots of (curv_root / curv_pristine) <= 0.5 AND >= 2
    of 3 roots individually <= 0.5 (">= 2x lower" with a replication
    flavor).
  * per-direction kill: theta_0 - D v_i on the onset grid
    0.05..3.00 + D=0 anchor (e209's walk verbatim, early stop), kill =
    first g-12 <= 0.27 downcrossing interpolated linear-in-D;
    unresolved-high -> None (censored, pairwise-dropped).
  * correlations (CONTEXT, never adjudicated): per-root Spearman over
    the top-8 directions of (kill vs SV energy w_i = sv_i^2 / sum
    sv^2) and (kill vs curv_i at primary eps on L_wash); pooled
    versions reported with the root-effect caveat; the committed
    bands (e209's walled medians; e_chart's pristine fine-D co-read,
    promoted by e205) joined per root against PR and curvature.
  * composite order FLATTENED-BASIN / WIDER-EXPLORATION / GRADED; the
    first two are mutually exclusive BY CONSTRUCTION (MATCH needs all
    three walled PRs <= 1.25x pristine; WIDER needs >= 2 of them
    >= 1.25x).

DESK PRE-READ DISCLOSURE (stated before compute, e205's convention;
honesty, not a prediction): the committed walled PRs (4.216 / 4.263 /
4.260) sit within 0.1-1.2% of the PR recomputable on the desk from
e_chart's committed pristine spectrum (4.214) — a suggestive MATCH —
but e_chart's e131 "step history" is a mixed-lr checkpoint ladder
(7x lr1e-3 + 7x lr3e-5 + 5x lr1e-5 + 1 opt1 segment), NOT a contiguous
wash: cross-instrument. The same-instrument pristine PR is fresh
compute here and OWNS the verdict. No bar is desk-forced; no
prediction registered; the openness is the point.

PRE-DISPATCH CHECKS (Rule 12): the protocol gate set (G_NAMEFREE /
G_SPLICE / G_BATTERY / G_ANCHOR, e209's conventions); G_PARENTS (the
five parents' md5s + the committed numbers hard-bound: e209's PRs /
bands / settled md5s / medians arithmetic, g1b's C-arm rows 1..20 and
W1 s300 read, g1bR's s300 reads, e_chart's root read + spectrum +
pristine band); G_ROOT per state (the walled settled-flat md5s
BIT-gate vs e209; the pristine read gate vs e_chart's rung-0 read);
G_HIST per root (the walled PR reproduction gate; the pristine
per-step C-arm gate); G_STREAM at the pristine root (seed-10902
step-1 CE + L2 vs g1b's committed arm row, + the e185 batch md5).
WHAT THE GATES GUARANTEE: NOTHING — the pristine PR could land at 2,
at 4, at 6; the curvatures could invert; the correlations could be
zero. The openness is the point.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is never claimed), torch threads 4, load-check
recorded not gating (e204/e205/e209's convention), small eval bursts,
progressive metrics.json writes after every phase, decisive.

Outputs: runs/e211/{metrics.json (PROGRESSIVE), e211_walled_band.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e211_walled_band.py    (E211_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e185/e209)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # the e185/e209 reduction order

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E211_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e211 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E131_ROOT_CK = "e131_consolidated_e113.pt"     # the pristine root itself
E209_METRICS = E43.REPO / "runs" / "e209" / "metrics.json"
E205_METRICS = E43.REPO / "runs" / "e205" / "metrics.json"
E_CHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
G1B_METRICS = E43.REPO / "runs" / "g1b" / "metrics.json"
G1BR_METRICS = E43.REPO / "runs" / "g1bR" / "metrics.json"

GEOS = (-12, 0, 12)               # battery ctx offsets (e185/e209 convention)
RULER_J = -12                     # THE RULER: install-60 g-12 (e191/e209's)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE)
LR_ADAMW = 1e-3                   # the wash recipe (g1/g1b/g1bR + e205/e209)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
WALL_R = 0.7                      # the W1 wall radius (g1b/g1bR's commit)
N_PARAMS = 2_739_072
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 — e199/e205/e209's onset grid
WASH_HIST_STEPS = 4 if SMOKE else 20        # e193b/e205/e209's history segments

# ---- the FD-curvature instrument (registered arithmetic) --------------------
EPS_GRID = (2e-3, 5e-3, 1e-2)     # the eps ladder (SNR vs float32 roundoff)
EPS_PRIMARY = 1e-2                # THE ADJUDICATING eps (registered)
CURV_TOP_BAR = 4                  # the bar's directions (the dispatch's letter)
CURV_TOP_CORR = 8                 # the correlation's directions
KILL_TOP = 8                      # per-direction kill rays

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH_1 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's seed-10902 step-1 batch md5

# e209's committed per-root records (hard-bound at P0b; gates below)
E209_COMMITTED = {
    "R5": {"ckpt": "g1b_W1_s300.pt", "seed": 10902,
           "settled_flat_md5": "2cc89e6eca86442ae3ec688037ad821f",
           "settled_read": 0.918087363243103,
           "pr": 4.216394763676648,
           "hist_step1_L2": 1.6541320085525513,
           "band_median": 1.579816428873246,
           "band_kill_Ds": [1.579816428873246, 1.6956607003316866, 1.1389369879609799],
           "u0_edge": 1.1756714746286159,
           "robust_seeds": (12403, 12404)},
    "R6": {"ckpt": "g1bR_W1_10907_s300.pt", "seed": 10907,
           "settled_flat_md5": "29d78cfb7fee3bf5ab8f6da78fd3f1d2",
           "settled_read": 0.9055948853492737,
           "pr": 4.262736184804125,
           "hist_step1_L2": 1.654052734375,
           "band_median": 1.7055491980018755,
           "band_kill_Ds": [1.089780632096794, 1.7055491980018755, 1.8256070805237203],
           "u0_edge": 1.3412719978825394,
           "robust_seeds": (12405, 12406)},
    "R7": {"ckpt": "g1bR_W1_10908_s300.pt", "seed": 10908,
           "settled_flat_md5": "aab525fb8ed09860f2c2a90c3f15b5ed",
           "settled_read": 0.8999181389808655,
           "pr": 4.259615257577512,
           "hist_step1_L2": 1.6540809869766235,
           "band_median": 0.9141374091130942,
           "band_kill_Ds": [0.4698055204802328, 1.5379877124339503, 0.9141374091130942],
           "u0_edge": 1.1314342273398847,
           "robust_seeds": (12407, 12408)},
}

# the pristine root: e_chart's committed read (rung-0 g-12) + the committed
# pristine band (the fine-D co-read e205 promoted); stream = g1b's C-arm
ECHART_PRISTINE_READ = 0.9155886173248291        # e_chart partB.e131.wash curve rung 0
ECHART_PRISTINE_BAND = {"kill_Ds": {"11601": 0.6100797132671589,
                                    "11602": None, "11603": 0.5636664436670049},
                        "median": 0.6100797132671589,
                        "instrument": "e_chart FINE_GRID 11pts 0.05..1.5 "
                                      "(1-of-3 right-censored > 1.5) — e205's "
                                      "promoted pristine band; CROSS-INSTRUMENT "
                                      "vs e209's onset-grid bands (disclosed)"}

PRISTINE = {"id": "P", "ckpt": E131_ROOT_CK, "seed": 10902,
            "stream_src": "g1b arms.C (the seed-10902 wash AT the pristine "
                          "root — 300 committed per-step rows)",
            "robust_seeds": (12401, 12402)}

ROOTS = [PRISTINE] + [{"id": rid, **E209_COMMITTED[rid]} for rid in ("R5", "R6", "R7")]
ALL_ROOTS = ROOTS
STATES: dict = {}                 # per-rid in-memory tensors (never serialized)
if SMOKE:                         # shakedown trims (disclosed; nothing adjudicated)
    ROOTS = [PRISTINE, {"id": "R5", **E209_COMMITTED["R5"]}]
    D_GRID = [0.05, 0.5, 1.5, 3.0]
    EPS_GRID = (1e-2,)
    CURV_TOP_CORR = 4
    KILL_TOP = 4

G_BIT_TOL = 5e-6                  # settled read vs committed (bit tier)
G_READ_TOL = 5e-3                 # reads vs committed (cross-device texture tier)
G_PR_TOL = 1e-6                   # walled PR reproduction (CPU->CPU, same threads)
G_HIST_TOL = 5e-3                 # pristine per-step L2 vs g1b CUDA rows

MATCH_LO, MATCH_HI = 0.80, 1.25   # the SPECTRA-MATCH PR window (registered)
MATCH_SHAPE_BITS = 0.5            # the shape clause bar (registered)
WIDER_RATIO = 1.25                # the WIDER PR ratio (registered)
CURV_RATIO_BAR = 0.5              # ">= 2x lower" (registered)

REGISTERED_BARS = {
    "FLATTENED_BASIN": "FLATTENED-BASIN: \"fires if the walled states' "
        "curvatures along the span directions are materially lower than the "
        "pristine's (>= 2x) while the span spectra match — the wall flattened "
        "the basin (a); a new wall property: the ball buys noise-tolerance "
        "by flattening.\"",
    "WIDER_EXPLORATION": "WIDER-EXPLORATION: \"fires if the walled spans are "
        "intrinsically wider (the SV spectra/PR materially larger) — the "
        "projection kept the trajectory exploring (b); the band is the "
        "history's shadow.\"",
    "GRADED": "GRADED: \"any partial/mix — the tables verbatim.\"",
    "operationalizations": (
        "states = the pristine e131 root (as committed) + the walled s300 "
        "roots settled per e209 (settled flats md5-gated); span(primary) = "
        "the root's OWN contiguous 20-step unwalled AdamW wash from its seed "
        "stream (walled streams = e209's -> PR must reproduce e209's "
        "committed PR within 1e-6 rel; pristine = the seed-10902 stream at "
        "the e131 root, per-step L2 gated vs g1b's committed C-arm rows "
        "1..20 at tol 5e-3); spectrum = 20 Gram-SVD SVs (fp64); PR = "
        "(sum sv^2)^2/sum sv^4; SPECTRA-MATCH = all 3 walled primary PRs in "
        "[0.8,1.25]x pristine AND mean-over-walled-roots of mean_i "
        "|log2[(sv_i/sv_1)_w/(sv_i/sv_1)_p]| <= 0.5 bits; WIDER = >= 2 of 3 "
        "walled primary PRs >= 1.25x pristine; curvature = FD second "
        "difference [L(th+eps v)+L(th-eps v)-2L(th)]/eps^2 along the top-4 "
        "(bar) unit right-singular vectors, L = the FIXED primary-history "
        "step-1 batch CE (primary loss), eps grid {2e-3,5e-3,1e-2}, primary "
        "eps = 1e-2; per-root curvature = median over top-4 of curv(v,1e-2); "
        "FLAT curvature clause = geomean over the 3 walled roots of "
        "(curv_root/curv_pristine) <= 0.5 AND >= 2 of 3 roots individually "
        "<= 0.5; per-direction kills = theta_0 - D v_i on the onset grid "
        "0.05..3.00 (e209's instrument verbatim, early stop), first g-12 "
        "<= 0.27 downcrossing interpolated; correlations (kill vs SV energy, "
        "kill vs curvature; per-root Spearman over top-8 + pooled) and the "
        "committed-band join are CONTEXT, never adjudicated; robustness "
        "spans (2 fresh streams/root, seeds 12401-12408) are CONTEXT; "
        "composite order FLATTENED-BASIN / WIDER-EXPLORATION / GRADED, the "
        "first two mutually exclusive by construction."),
    "desk_pre_read_disclosure": (
        "the committed walled PRs (4.216/4.263/4.260) sit within 0.1-1.2% "
        "of the PR recomputed on the desk from e_chart's committed pristine "
        "spectrum (4.214) — a suggestive MATCH, but e_chart's e131 'step "
        "history' is a mixed-lr checkpoint ladder (7x lr1e-3 + 7x lr3e-5 + "
        "5x lr1e-5 + 1 opt1 segment), NOT a contiguous wash: "
        "CROSS-INSTRUMENT. The same-instrument pristine PR is fresh compute "
        "here and owns the verdict. No bar desk-forced; no prediction "
        "registered; no bar shopping."),
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE PRISTINE SPAN IS FRESH (disclosed before compute): e_chart's "
    "committed e131 spectrum is a mixed-lr checkpoint ladder, not a "
    "contiguous 20-step wash — it cannot serve as the same-instrument "
    "pristine span. This cell runs the pristine root's OWN contiguous wash "
    "(e209's machinery verbatim, the seed-10902 stream = g1b's own C-arm "
    "stream), anchored per-step against g1b's committed C-arm rows.",
    "THE WALLED PRIMARIES ARE e209'S SPANS RE-DERIVED (not ported): same "
    "checkpoints, same settling, same streams, same arithmetic — the "
    "reproduction itself is the gate (PR within 1e-6 rel of e209's "
    "committed numbers; the settled flats md5-gate vs e209).",
    "THE CURVATURE INSTRUMENT IS NEW (registered arithmetic): a 3-point FD "
    "second difference along unit span directions on a FIXED batch (the "
    "root's own history step-1 batch — the wash's own objective) and on "
    "the ruler battery's -log pZ (co-read). The primary eps is 1e-2: the "
    "estimate reads curvature at the 1e-2 L2 scale (quartic contamination "
    "possible at the largest eps; float32 roundoff at the smallest — the "
    "eps ladder reports both, the honesty block carries the caveat).",
    "THE PER-DIRECTION KILLS ARE A NEW CO-READ (same instrument, new "
    "directions): e209 committed bands from RANDOM in-span draws only — "
    "no per-direction instrument existed. Top-8 span directions walked on "
    "e209's own onset grid + ruler; the kill ray uses the SVD-emitted sign "
    "while curvature is sign-invariant (disclosed on every correlation).",
    "THE COMMITTED-BAND JOIN IS CROSS-INSTRUMENT AT THE PRISTINE ROW: "
    "e_chart's pristine band is a FINE_GRID cap-1.5 co-read (1-of-3 "
    "censored); the walled bands are e209's onset-grid medians. The join "
    "is context, never adjudicated.",
    "The load-check is recorded, not gating (e204/e205/e209's convention).",
    "Smoke mode trims: 2 roots (P + R5), grid {0.05,0.5,1.5,3.0}, 4-step "
    "histories, eps {1e-2} only, top-4 correlations; nothing adjudicated "
    "(verdict stamped SMOKE).",
]

# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e209_census_debt.py VERBATIM (whose own provenance is
# e191/e205 via e185 — the e176n lineage). Copied rather than imported to
# own the device policy and the arithmetic.


def load_body(path) -> tuple[TinyGPT, dict, dict]:
    """Load a checkpoint: (net with BODY weights, body sd, anch__ buffers)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    meta = st.get("meta") if isinstance(st, dict) else None
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    anch = {k: v for k, v in sd.items() if k.startswith("anch__")}
    body = {k: v for k, v in sd.items() if not k.startswith("anch__")}
    m = TinyGPT(Cfg())
    m.load_state_dict(body)
    m.eval()
    return m, body, {"meta": meta, "anch": anch}


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e209 battery on CPU: p(Z) at the last position."""
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
def ruler_nll(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """The ruler battery's own loss: mean(-log pZ) (the curvature co-read)."""
    net.eval()
    nll = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        logp = F.log_softmax(lg[:, -1], -1)[:, zid]
        nll.append(-logp)
    return float(torch.cat(nll).mean())


@torch.no_grad()
def batch_ce(net: TinyGPT, x, y, bs=64) -> float:
    """CE on a FIXED batch (the primary curvature loss)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        lg, _ = net(x[i:i + bs], y[i:i + bs])
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[i:i + bs].reshape(-1))
        tot += float(ce.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def interp_d_kill(v0, v1, d0, d1):
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) — e199/e205/e209's instrument."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm"] <= SHUT_BAR and a["gm"] > SHUT_BAR:
            edge = interp_d_kill(a["gm"], b["gm"], a["D"], b["D"])
            break
    return edge


def participation_ratio(sv) -> float:
    s2 = sv if isinstance(sv, torch.Tensor) else torch.tensor(sv, dtype=torch.float64)
    s2 = (s2.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """e_chart/e193b/e205/e209's Gram-based right-singular basis (fp64)."""
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


def band_median(kill_Ds: list) -> dict:
    """e205's middle order statistic with right-censoring."""
    resolved = sorted([d for d in kill_Ds if d is not None])
    n_cens = sum(1 for d in kill_Ds if d is None)
    n = len(kill_Ds)
    if n_cens >= (n + 1) // 2 and n_cens >= 2:
        return {"median": None, "n": n, "n_resolved": len(resolved),
                "n_censored": n_cens, "defined": False}
    order = sorted([d if d is not None else float("inf") for d in kill_Ds])
    med = order[n // 2] if n % 2 == 1 else 0.5 * (order[n // 2 - 1] + order[n // 2])
    med = None if med == float("inf") else float(med)
    return {"median": med, "n": n, "n_resolved": len(resolved),
            "n_censored": n_cens, "defined": med is not None}


def cpu_load_probe() -> int | None:
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def spearman(xs, ys):
    def ranks(vals):
        order_ = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order_):
            j = i
            while j + 1 < len(order_) and vals[order_[j + 1]] == vals[order_[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order_[k]] = avg
            i = j + 1
        return r
    rx_, ry_ = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx_) / n, sum(ry_) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx_, ry_))
    den = (sum((a - mx) ** 2 for a in rx_)
           * sum((b - my) ** 2 for b in ry_)) ** 0.5
    return num / den if den > 0 else float("nan")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs)
           * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den > 0 else float("nan")


def geomean(xs) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e211_smoke" if SMOKE else "e211")
    metrics: dict = {
        "experiment": "e211_walled_band",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never claimed)",
            "torch_threads": torch.get_num_threads(),
            "load_check_recorded_not_gating": True,
            "phases": "P0 gates -> P1 states+histories+spans -> P2 curvature "
                      "-> P3 per-direction kills + correlations -> P4 "
                      "adjudication + figure; progressive writes",
            "roots": [f"{r['id']}:{r['ckpt']}" for r in ROOTS],
            "eps_grid": list(EPS_GRID), "eps_primary": EPS_PRIMARY,
            "curv_top_bar": CURV_TOP_BAR, "curv_top_corr": CURV_TOP_CORR,
            "kill_top": KILL_TOP,
        },
        "deviations": deviations,
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    load0 = cpu_load_probe()
    metrics["envelope"]["cpu_load_pct_at_launch"] = load0
    log(f"E211 THE WALLED-BAND QUESTION (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), load-check "
        f"recorded (launch: {load0}%), progressive writes, decisive")

    # ================= P0a: protocol rebuild (e209's gate set) ================
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
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "expected_shapes": {"-12": [60, PRE - 12], "0": [60, PRE],
                            "12": [60, PRE + 12]},
        "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at ctx "
                "offsets {-12,0,+12} — e185/e209's convention = g1b/g1bR's "
                "own battery (p(Z) on the install-60 pool)",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    ruler_ids = bat_ids[RULER_J]

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
    G_ANCHOR = {"starts": n_starts, "tries": tries, "rejections": rejections,
                "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
                "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"
    log("P0a: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank bit-match)")
    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}
    write_partial("P0a protocol gates PASSED")

    # ================= P0b: the parents, hard-bound (Rule 12) ================
    for p in (E209_METRICS, E205_METRICS, E_CHART_METRICS, G1B_METRICS,
              G1BR_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")

    def md5of(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    e209m = json.loads(E209_METRICS.read_text(encoding="utf-8"))
    g1bm = json.loads(G1B_METRICS.read_text(encoding="utf-8"))
    g1brm = json.loads(G1BR_METRICS.read_text(encoding="utf-8"))
    echm = json.loads(E_CHART_METRICS.read_text(encoding="utf-8"))
    parents_md5 = {"e209": md5of(E209_METRICS), "e205": md5of(E205_METRICS),
                   "e_chart": md5of(E_CHART_METRICS),
                   "g1b": md5of(G1B_METRICS), "g1bR": md5of(G1BR_METRICS)}

    # e209's committed rows, hard-bound + arithmetic re-checked
    assert e209m["adjudication"]["verdict"] == "MARGIN-BREAKS"
    carm_rows = {int(r["step"]): r for r in g1bm["arms"]["C"]["traj"]}
    assert 1 in carm_rows and 20 in carm_rows
    e131_spec = echm["partB_subspace"]["e131"]
    ech_root_read = e131_spec["wash"]["curve"][0]["gm12"]
    ech_spectrum = list(e131_spec["svd"]["spectrum_step_history"])
    assert abs(ech_root_read - ECHART_PRISTINE_READ) < 1e-12
    assert len(ech_spectrum) == 20
    ech_pr_desk = participation_ratio(ech_spectrum)
    for rid, c in E209_COMMITTED.items():
        cell = e209m["roots"][rid]
        assert abs(cell["span"]["participation_ratio"] - c["pr"]) < 1e-12
        assert abs(cell["G_ROOT"]["settled_read_committed"]
                   - c["settled_read"]) < 1e-12
        assert cell["G_ROOT"]["settled_flat_md5"] == c["settled_flat_md5"]
        med = band_median(c["band_kill_Ds"])
        assert med["defined"] and abs(med["median"] - c["band_median"]) < 1e-12
        assert abs(cell["band"]["median_stat"]["median"]
                   - c["band_median"]) < 1e-12
    g1b_w1_300 = next(r for r in g1bm["arms"]["W1"]["traj"] if r["step"] == 300)
    wall = g1brm["adjudication"]["wall"]
    G_PARENTS = {
        "files_md5": parents_md5,
        "hardbound": {
            "e209.verdict": e209m["adjudication"]["verdict"],
            "e209.PR_R5/R6/R7": [c["pr"] for c in E209_COMMITTED.values()],
            "e209.band_medians": [c["band_median"]
                                  for c in E209_COMMITTED.values()],
            "e209.settled_flat_md5s": [c["settled_flat_md5"]
                                       for c in E209_COMMITTED.values()],
            "g1b.C_traj_rows_1_to_20": "loaded (the pristine history anchors)",
            "g1b.W1.traj[s300].g_m12": g1b_w1_300["g_m12_mean_pz"],
            "g1bR.W1_10907[300]": wall["W1_10907"]["g_m12"]["300"],
            "g1bR.W1_10908[300]": wall["W1_10908"]["g_m12"]["300"],
            "e_chart.pristine_root_read": ech_root_read,
            "e_chart.pristine_spectrum_n": len(ech_spectrum),
            "e_chart.pristine_spectrum_PR_desk": ech_pr_desk,
        },
        "desk_band_median_recheck": "middle order statistic recomputed for "
                                    "R5/R6/R7 from e209's kill_Ds: exact",
        "pass": True,
    }
    log(f"P0b: parents hard-bound — e209 (verdict {G_PARENTS['hardbound']['e209.verdict']}; "
        f"PRs {['%.4f' % c['pr'] for c in E209_COMMITTED.values()]}); g1b C-arm "
        f"rows 1..20 loaded; e_chart pristine read {ech_root_read:.4f} + "
        f"spectrum (desk PR {ech_pr_desk:.4f}, cross-instrument — the "
        f"disclosure's number)")
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    write_partial("P0b parent gates PASSED (e209/e_chart/g1b/g1bR hard-bound)")

    # ================= P0c: G_STREAM — the pristine root's own stream ========
    p_net, _, p_extra = load_body(CKPT_DIR / E131_ROOT_CK)
    p_theta = flat_params(p_net)
    g = torch.Generator().manual_seed(10902)
    aj = torch.randint(16, (ANCH_BS,), generator=g)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
    x1 = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y1 = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(p_net)
    tw.train()
    optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                             weight_decay=0.1)
    logits, _ = tw(x1)
    ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y1.reshape(-1)).item())
    optw.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y1.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0)
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - p_theta))
    del tw, optw, logits
    g1b_c1 = carm_rows[1]
    G_STREAM = {
        "seed": 10902, "src": "runs/g1b/metrics.json arms.C.traj[0]",
        "step1_x_md5": x1_md5, "step1_x_md5_match_e185": x1_md5 == E185_XHASH_1,
        "ce1_measured": ce1, "ce1_committed": g1b_c1["ce_batch"],
        "d_ce": abs(ce1 - g1b_c1["ce_batch"]),
        "disp1_measured": disp1, "disp1_committed": g1b_c1["step_disp"],
        "d_disp": abs(disp1 - g1b_c1["step_disp"]), "tol": G_HIST_TOL,
        "pass": bool(abs(ce1 - g1b_c1["ce_batch"]) < G_HIST_TOL
                     and abs(disp1 - g1b_c1["step_disp"]) < G_HIST_TOL
                     and x1_md5 == E185_XHASH_1),
        "note": "the pristine stream IS g1b's own C-arm stream; step-1 CE + "
                "AdamW L2 vs the committed CUDA row (1e-7-class diffs are "
                "cross-device texture, e209's pre-verified convention)",
    }
    log(f"G_STREAM[pristine]: CE |d| {G_STREAM['d_ce']:.1e}, disp |d| "
        f"{G_STREAM['d_disp']:.1e}, md5 "
        f"{'OK' if G_STREAM['step1_x_md5_match_e185'] else 'DRIFT'}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    if not G_STREAM["pass"]:
        raise RuntimeError("pristine stream gate FAILED")
    metrics["gates"]["G_STREAM"] = G_STREAM
    del p_net, p_theta
    write_partial("P0c G_STREAM PASSED (the pristine stream = g1b's C-arm)")

    # ================= P1: states + histories + spans ========================
    metrics["roots"] = {}
    for r in ROOTS:
        rid = r["id"]
        log("=" * 78)
        log(f"P1[{rid}] {r['ckpt']} (primary stream seed {r['seed']}; "
            f"robustness {r['robust_seeds']})")
        cell = {"id": rid, "ckpt": r["ckpt"], "seed": r["seed"]}

        # ---- load (+ settle for the walled) + G_ROOT
        net0, _, extra = load_body(CKPT_DIR / r["ckpt"])
        anch = extra["anch"]
        n_par = sum(p.numel() for p in net0.parameters())
        assert n_par == N_PARAMS, f"{rid}: param count {n_par}"
        if rid == "P":
            theta_raw = flat_params(net0)
            raw_read = battery_cell(net0, ruler_ids, zid)["mean_pz"]
            theta0, d_raw, d_settled = theta_raw, 0.0, 0.0
            root_read = raw_read
            G_ROOT = {
                "checkpoint": f"runs/checkpoints/{r['ckpt']}",
                "meta": extra["meta"], "n_params": n_par,
                "settled_projection_fired": False,
                "d_raw_vs_anchor": None, "d_settled": None,
                "settled_read_measured": root_read,
                "settled_read_committed": ECHART_PRISTINE_READ,
                "settled_flat_md5": hashlib.md5(
                    theta0.numpy().tobytes()).hexdigest(),
                "abs_diff": abs(root_read - ECHART_PRISTINE_READ),
                "bit_tol": G_BIT_TOL, "tol": G_READ_TOL,
                "bit": bool(abs(root_read - ECHART_PRISTINE_READ) < G_BIT_TOL),
                "pass": bool(abs(root_read - ECHART_PRISTINE_READ) < G_READ_TOL),
                "note": "the PRISTINE root used AS COMMITTED (no anch__ ball "
                        "of its own; no settling); read gated vs e_chart's "
                        "committed rung-0 install-60 g-12 read",
            }
        else:
            theta_raw = flat_params(net0)
            a_flat = torch.cat([anch["anch__" + n_.replace(".", "_")].reshape(-1)
                                for n_, _ in net0.named_parameters()])
            d_raw = float(torch.norm(theta_raw - a_flat))
            raw_read = battery_cell(net0, ruler_ids, zid)["mean_pz"]
            R_ck = float(anch["anch__R"])
            if d_raw > R_ck:
                s = R_ck / d_raw
                load_flat(net0, a_flat + s * (theta_raw - a_flat))
            theta0 = flat_params(net0)
            d_settled = float(torch.norm(theta0 - a_flat))
            root_read = battery_cell(net0, ruler_ids, zid)["mean_pz"]
            flat_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
            G_ROOT = {
                "checkpoint": f"runs/checkpoints/{r['ckpt']}",
                "meta": extra["meta"], "n_params": n_par,
                "settled_projection_fired": d_raw > R_ck,
                "anch__R": R_ck,
                "d_raw_vs_anchor": d_raw, "d_settled": d_settled,
                "raw_read_unsettled": raw_read,
                "settled_read_measured": root_read,
                "settled_read_committed": r["settled_read"],
                "settled_flat_md5": flat_md5,
                "settled_flat_md5_match_e209": flat_md5 == r["settled_flat_md5"],
                "abs_diff": abs(root_read - r["settled_read"]),
                "bit_tol": G_BIT_TOL, "tol": G_READ_TOL,
                "bit": bool(abs(root_read - r["settled_read"]) < G_BIT_TOL),
                "pass": bool(flat_md5 == r["settled_flat_md5"]
                             and abs(root_read - r["settled_read"]) < G_READ_TOL),
                "note": "e209's settled convention verbatim; the settled flat "
                        "md5 BIT-gates vs e209's committed block",
            }
        log(f"G_ROOT[{rid}]: {n_par} params, read {root_read:.6f}: "
            + ("PASS" if G_ROOT["pass"] else "FAIL")
            + (" (bit)" if G_ROOT.get("bit") else ""))
        if not G_ROOT["pass"]:
            raise RuntimeError(f"root gate FAILED at {rid}")
        cell["G_ROOT"] = G_ROOT
        cell["ruler_caveat"] = (
            "install-60 g-12 (NOVEL geometry — the e131 lineage's battery "
            f"family); root read {root_read:.4f} >> 0.27: NO fallback "
            "(T155's convention moot here)")
        metrics["roots"][rid] = cell
        write_partial(f"P1[{rid}] G_ROOT PASSED")

        # ---- the wash runner (e209's loop verbatim; returns segments)
        def run_wash(seed_stream: int, trace_ruler: bool):
            """20-step contiguous unwalled AdamW wash from theta0 (e209 verbatim)."""
            wgen = torch.Generator().manual_seed(seed_stream)
            wnet = copy.deepcopy(net0)
            wnet.train()
            wopt = torch.optim.AdamW(wnet.parameters(), lr=LR_ADAMW,
                                     betas=(0.9, 0.95), weight_decay=0.1)
            wtheta = theta0.clone()
            segs, rows = [], []
            evl = copy.deepcopy(net0)
            evl.eval()
            for s_wh in range(1, WASH_HIST_STEPS + 1):
                aj_ = torch.randint(16, (ANCH_BS,), generator=wgen)
                rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                                    generator=wgen)
                anc_ = anchor_neutral[aj_]
                rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
                x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
                y_ = torch.cat([anc_[:, 1:], rnd_[:, 1:]], 0)
                if s_wh == 1:
                    x1b, y1b = x_.clone(), y_.clone()    # the FIXED batch (curvature)
                logits_w, _ = wnet(x_)
                lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                                     y_.reshape(-1))
                wopt.zero_grad(set_to_none=True)
                lw.backward()
                torch.nn.utils.clip_grad_norm_(wnet.parameters(), 1.0)
                wopt.step()
                th_new = flat_params(wnet)
                segs.append(th_new - wtheta)
                row = {"step": s_wh,
                       "L2": float(torch.norm(th_new - wtheta)),
                       "cum": float(torch.norm(th_new - theta0)),
                       "ce": float(lw.item())}
                if trace_ruler:
                    load_flat(evl, th_new)
                    gz = battery_cell(evl, ruler_ids, zid)
                    row["gm12"] = gz["mean_pz"]
                rows.append(row)
            wnet.eval()
            del wnet, wopt, evl
            return torch.stack(segs), rows, (x1b, y1b)

        # ---- primary history + gates
        H, hist_rows, fixed_batch = run_wash(r["seed"], trace_ruler=True)
        hist_dead = next((h["step"] for h in hist_rows
                          if "gm12" in h and h["gm12"] <= SHUT_BAR), None)
        if rid == "P":
            devs = []
            for h in hist_rows:
                crow = carm_rows[h["step"]]
                devs.append(abs(h["L2"] - crow["step_disp"]))
            max_dev = max(devs)
            G_HIST = {
                "gate": "per-step displacement L2 vs g1b's committed C-arm "
                        "rows 1.." + str(WASH_HIST_STEPS),
                "max_abs_dev_L2": max_dev,
                "per_step_dev_first3": [round(d, 8) for d in devs[:3]],
                "tol": G_HIST_TOL,
                "committed_death_step": 2,
                "measured_death_step": hist_dead,
                "pass": bool(max_dev < G_HIST_TOL),
                "note": "the pristine contiguous history IS g1b's C-arm wash "
                        "(same stream, same recipe); per-step anchored vs the "
                        "committed CUDA rows (cross-device texture tier)",
            }
        else:
            pr_rel = abs(float(participation_ratio(svd_basis(H)["sv"]))
                         - r["pr"]) / r["pr"]
            G_HIST = {
                "gate": "primary span PR must reproduce e209's committed PR",
                "step1_L2_measured": hist_rows[0]["L2"],
                "step1_L2_committed": r["hist_step1_L2"],
                "pr_reproduction_rel_dev": pr_rel,
                "tol": G_PR_TOL,
                "measured_death_step": hist_dead,
                "pass": bool(pr_rel < G_PR_TOL
                             and abs(hist_rows[0]["L2"]
                                     - r["hist_step1_L2"]) < 1e-3),
                "note": "same checkpoint, same settling (md5-gated), same "
                        "stream, same arithmetic as e209 — the reproduction "
                        "IS the gate (the full PR comparison happens at the "
                        "span block)",
            }
        log(f"  G_HIST[{rid}]: " + ("PASS" if G_HIST["pass"] else "FAIL")
            + f" (death step in-history: {hist_dead})")
        if not G_HIST["pass"]:
            raise RuntimeError(f"history gate FAILED at {rid}")
        cell["G_HIST"] = G_HIST
        cell["wash_history"] = {
            "n_steps": WASH_HIST_STEPS, "rows": hist_rows,
            "step1_L2": hist_rows[0]["L2"],
            "cum_final": hist_rows[-1]["cum"],
            "fact_died_in_history_at_step": hist_dead,
            "co_read_note": "UNWALLED continuation from the state (e209's "
                            "band instrument); reported, never adjudicated",
        }
        write_partial(f"P1[{rid}] primary history gated")

        # ---- primary span + robustness spans
        basis = svd_basis(H)
        sv = [float(s) for s in basis["sv"]]
        pr_primary = basis["pr"]
        cell["span_primary"] = {
            "n_segments": int(basis["Vp"].shape[0]),
            "rank_eff": basis["rank_eff"],
            "sv": sv, "participation_ratio": pr_primary,
            "cond": basis["cond"],
            "sv_top": sv[0],
            "energy_frac_top1": sv[0] ** 2 / sum(s * s for s in sv),
            "profile": [s / sv[0] for s in sv],
            "note": "the contiguous 20-step wash span (this run, same "
                    "machinery at every state — the same-instrument cell)",
        }
        if rid != "P":
            pr_rel = abs(pr_primary - r["pr"]) / r["pr"]
            assert pr_rel < G_PR_TOL, f"{rid}: PR reproduction {pr_rel}"
            cell["span_primary"]["pr_vs_e209_rel_dev"] = pr_rel
        log(f"  span: PR {pr_primary:.4f}, cond {basis['cond']:.2f}, "
            f"top SV {sv[0]:.3f}"
            + (f" (vs e209 {r['pr']:.4f}, rel {pr_rel:.1e})" if rid != "P" else ""))

        robust = []
        for rs in (r["robust_seeds"] if not SMOKE else ()):
            Hr, rows_r, _ = run_wash(rs, trace_ruler=False)
            br = svd_basis(Hr)
            robust.append({"seed": rs,
                           "sv": [float(s) for s in br["sv"]],
                           "participation_ratio": br["pr"],
                           "step1_L2": rows_r[0]["L2"]})
            del Hr
            log(f"  robustness wash s{rs}: PR {br['pr']:.4f}")
        cell["span_robustness"] = robust
        if robust:
            prs = [pr_primary] + [q["participation_ratio"] for q in robust]
            cell["span_pr_spread"] = {
                "n_realizations": len(prs),
                "min": min(prs), "max": max(prs),
                "span_rel": (max(prs) - min(prs)) / pr_primary,
                "note": "CONTEXT (never adjudicated): the realization spread "
                        "of the span-width scalar",
            }
            log(f"  PR realization spread: {min(prs):.4f}..{max(prs):.4f} "
                f"(rel {(max(prs)-min(prs))/pr_primary:.1%})")
        del H
        metrics["roots"][rid] = cell
        write_partial(f"P1[{rid}] spans done (PR {pr_primary:.4f})")

        # keep in memory for P2/P3 (NEVER serialized): theta0, basis, batch
        STATES[rid] = {"theta0": theta0, "Vp": basis["Vp"],
                       "sv": basis["sv"], "fixed_batch": fixed_batch,
                       "net0": net0}
        metrics["envelope"][f"cpu_load_pct_after_{rid}"] = cpu_load_probe()

    # ================= P2: the curvature co-read =============================
    log("=" * 78)
    log("P2: the FD-curvature instrument (registered arithmetic)")
    evl = copy.deepcopy(STATES[ROOTS[0]["id"]]["net0"])
    evl.eval()
    for r in ROOTS:
        rid = r["id"]
        st = STATES[rid]
        theta0, Vp, (x1b, y1b) = st["theta0"], st["Vp"], st["fixed_batch"]
        nd = min(CURV_TOP_CORR, Vp.shape[0])
        load_flat(evl, theta0)
        L0_w = batch_ce(evl, x1b, y1b)
        L0_r = ruler_nll(evl, ruler_ids, zid)
        crows = []
        for di in range(nd):
            v = Vp[di] / torch.norm(Vp[di])
            crow = {"dir": di, "sv": float(st["sv"][di]),
                    "energy_frac": float(st["sv"][di]) ** 2
                    / float(sum(float(s) ** 2 for s in st["sv"]))}
            for eps in EPS_GRID:
                th_p, th_m = theta0 + eps * v, theta0 - eps * v
                load_flat(evl, th_p)
                Lp_w = batch_ce(evl, x1b, y1b)
                Lp_r = ruler_nll(evl, ruler_ids, zid)
                load_flat(evl, th_m)
                Lm_w = batch_ce(evl, x1b, y1b)
                Lm_r = ruler_nll(evl, ruler_ids, zid)
                crow[f"curv_wash_eps{eps:g}"] = (Lp_w + Lm_w - 2 * L0_w) / eps ** 2
                crow[f"curv_rul_eps{eps:g}"] = (Lp_r + Lm_r - 2 * L0_r) / eps ** 2
            crows.append(crow)
            log(f"  [{rid} d{di}] sv {crow['sv']:.3f} "
                f"curv_wash(1e-2) {crow.get('curv_wash_eps0.01'):.4g} "
                f"curv_rul(1e-2) {crow.get('curv_rul_eps0.01'):.4g}")
        top4 = [c[f"curv_wash_eps{EPS_PRIMARY:g}"] for c in crows[:CURV_TOP_BAR]]
        curv4 = sorted(top4)[len(top4) // 2]
        curv8 = sorted([c[f"curv_wash_eps{EPS_PRIMARY:g}"] for c in crows])[len(crows) // 2]
        metrics["roots"][rid]["curvature"] = {
            "L0_wash": L0_w, "L0_rul": L0_r,
            "rows": crows,
            "eps_grid": list(EPS_GRID), "eps_primary": EPS_PRIMARY,
            "curv4_median_wash": curv4,
            "curv8_median_wash": curv8,
            "curv4_median_rul": sorted(
                [c[f"curv_rul_eps{EPS_PRIMARY:g}"] for c in crows[:CURV_TOP_BAR]]
            )[CURV_TOP_BAR // 2],
            "curv_rel_wash": curv4 / max(L0_w, 1e-12),
            "note": "FD second difference along the top span directions; "
                    "primary loss = the FIXED step-1 wash batch CE; co-read "
                    "= the ruler battery -log pZ; primary eps = "
                    f"{EPS_PRIMARY:g}",
        }
        log(f"P2[{rid}]: L0_wash {L0_w:.4f} L0_rul {L0_r:.4f} | curv4 "
            f"{curv4:.4g} (rel {curv4 / max(L0_w, 1e-12):.4g})")
        write_partial(f"P2[{rid}] curvature done")

    # ================= P3: the per-direction kills + correlations ============
    log("=" * 78)
    log("P3: per-direction kill rays (e209's onset instrument)")

    def walk_kill(rid, v, tag):
        st = STATES[rid]
        theta0 = st["theta0"]
        rows = []
        kill = None
        load_flat(evl, theta0)
        gz0 = battery_cell(evl, ruler_ids, zid)
        rows.append({"D": 0.0, "gm": gz0["mean_pz"],
                     "frac": gz0["frac_argmax_z"]})
        prev = gz0["mean_pz"]
        for D in D_GRID:
            load_flat(evl, theta0 - D * v)
            gz = battery_cell(evl, ruler_ids, zid)
            rows.append({"D": float(D), "gm": gz["mean_pz"],
                         "frac": gz["frac_argmax_z"]})
            if gz["mean_pz"] <= SHUT_BAR and prev > SHUT_BAR:
                kill = interp_d_kill(prev, gz["mean_pz"],
                                     float(D) - 0.05, float(D))
                break
            prev = gz["mean_pz"]
        log(f"  [{tag}] kill {kill if kill is not None else 'SOFT (>3.0)'} "
            f"({len(rows)} grid evals)")
        return rows, kill

    for r in ROOTS:
        rid = r["id"]
        st = STATES[rid]
        Vp = st["Vp"]
        krows = []
        for di in range(min(KILL_TOP, Vp.shape[0])):
            v = Vp[di] / torch.norm(Vp[di])
            rows, kill = walk_kill(rid, v, f"{rid} d{di}")
            krows.append({"dir": di, "D_kill": kill,
                          "energy_frac": metrics["roots"][rid]["curvature"]
                          ["rows"][di]["energy_frac"],
                          "curv_wash": metrics["roots"][rid]["curvature"]
                          ["rows"][di][f"curv_wash_eps{EPS_PRIMARY:g}"],
                          "rows": rows})
        metrics["roots"][rid]["dir_kills"] = krows
        write_partial(f"P3[{rid}] per-direction kills done")

    # correlations (CONTEXT, never adjudicated)
    corr = {}
    for r in ROOTS:
        rid = r["id"]
        kr = metrics["roots"][rid]["dir_kills"]
        pairs = [(k["D_kill"], k["energy_frac"], k["curv_wash"])
                 for k in kr if k["D_kill"] is not None]
        if len(pairs) >= 4:
            corr[rid] = {
                "n_dirs_resolved": len(pairs),
                "spearman_kill_vs_sv": spearman([p[0] for p in pairs],
                                                [p[1] for p in pairs]),
                "spearman_kill_vs_curv": spearman([p[0] for p in pairs],
                                                  [p[2] for p in pairs]),
            }
        else:
            corr[rid] = {"n_dirs_resolved": len(pairs), "note": "too few"}
    all_pairs = []
    for r in ROOTS:
        rid = r["id"]
        for k in metrics["roots"][rid]["dir_kills"]:
            if k["D_kill"] is not None:
                all_pairs.append((rid, k["D_kill"], k["energy_frac"],
                                  k["curv_wash"]))
    corr["pooled"] = {
        "n": len(all_pairs),
        "spearman_kill_vs_sv": spearman([p[1] for p in all_pairs],
                                        [p[2] for p in all_pairs]),
        "spearman_kill_vs_curv": spearman([p[1] for p in all_pairs],
                                          [p[3] for p in all_pairs]),
        "pearson_kill_vs_sv": pearson([p[1] for p in all_pairs],
                                      [p[2] for p in all_pairs]),
        "pearson_kill_vs_curv": pearson([p[1] for p in all_pairs],
                                        [p[3] for p in all_pairs]),
        "caveat": "pooled across roots — confounds the root effect (walled "
                  "kills are larger overall) with the direction effect; the "
                  "per-root Spearman rows are the clean read",
    }
    # the committed-band join (context)
    band_join = [{"id": "P",
                  "band_median_committed": ECHART_PRISTINE_BAND["median"],
                  "band_instrument": ECHART_PRISTINE_BAND["instrument"],
                  "pr_primary": metrics["roots"]["P"]["span_primary"]
                  ["participation_ratio"],
                  "curv4_median_wash": metrics["roots"]["P"]["curvature"]
                  ["curv4_median_wash"]}]
    for rid in ("R5", "R6", "R7"):
        if rid in metrics["roots"]:
            band_join.append({
                "id": rid,
                "band_median_committed": E209_COMMITTED[rid]["band_median"],
                "band_instrument": "e209 onset grid 0.05..3.00 (same family "
                                   "as this cell's rays)",
                "pr_primary": metrics["roots"][rid]["span_primary"]
                ["participation_ratio"],
                "curv4_median_wash": metrics["roots"][rid]["curvature"]
                ["curv4_median_wash"]})
    metrics["band_re_read"] = {
        "correlations_context": corr,
        "committed_band_join_context": band_join,
        "note": "the per-direction kills use the SVD-emitted ray sign; "
                "curvature is sign-invariant — the correlation carries this "
                "flag; all quantities in this block are CONTEXT, never "
                "adjudicated",
    }
    log("P3 correlations: "
        + "; ".join(f"{k}: rho_sv {v.get('spearman_kill_vs_sv', float('nan')):.2f} "
                    f"rho_curv {v.get('spearman_kill_vs_curv', float('nan')):.2f}"
                    for k, v in corr.items() if k != "pooled"))
    write_partial("P3 kills + correlations done")

    # ================= P4: the adjudication (frozen bars) ====================
    pr_p = metrics["roots"]["P"]["span_primary"]["participation_ratio"]
    pr_w = {rid: metrics["roots"][rid]["span_primary"]["participation_ratio"]
            for rid in ("R5", "R6", "R7") if rid in metrics["roots"]}
    prof_p = metrics["roots"]["P"]["span_primary"]["profile"]
    shape_bits = {}
    for rid, pr in pr_w.items():
        prof_w = metrics["roots"][rid]["span_primary"]["profile"]
        shape_bits[rid] = sum(abs(math.log2(a / b))
                              for a, b in zip(prof_w, prof_p)) / len(prof_p)
    curv_p = metrics["roots"]["P"]["curvature"]["curv4_median_wash"]
    curv_w = {rid: metrics["roots"][rid]["curvature"]["curv4_median_wash"]
              for rid in pr_w}
    ratios = {rid: curv_w[rid] / curv_p for rid in pr_w}
    geo_ratio = geomean(list(ratios.values()))
    n_ratio_le = sum(1 for x in ratios.values() if x <= CURV_RATIO_BAR)

    spectra_match = bool(
        all(MATCH_LO <= pr / pr_p <= MATCH_HI for pr in pr_w.values())
        and len(shape_bits) > 0
        and sum(shape_bits.values()) / len(shape_bits) <= MATCH_SHAPE_BITS)
    spectra_wider = bool(sum(1 for pr in pr_w.values()
                             if pr >= WIDER_RATIO * pr_p) >= 2)
    curv_clause = bool(geo_ratio <= CURV_RATIO_BAR and n_ratio_le >= 2
                       and len(ratios) >= 2)

    flat_fires = bool(curv_clause and spectra_match)
    wide_fires = bool(spectra_wider and not flat_fires)
    graded_fires = not flat_fires and not wide_fires
    assert not (flat_fires and wide_fires), "bars must be exclusive"

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif flat_fires:
        verdict = "FLATTENED-BASIN"
        clause = (f"the walled curvatures along the span directions are "
                  f">= 2x lower than the pristine's (geomean ratio "
                  f"{geo_ratio:.3f}; per-root "
                  + "/".join(f"{rid} {ratios[rid]:.3f}" for rid in ratios)
                  + f") while the span spectra match (PRs "
                  + "/".join(f"{pr:.3f}" for pr in pr_w.values())
                  + f" vs pristine {pr_p:.3f}, all in [{MATCH_LO},{MATCH_HI}]x; "
                  f"shape {sum(shape_bits.values())/len(shape_bits):.3f} bits "
                  f"<= {MATCH_SHAPE_BITS}) — the wall flattened the basin; the "
                  f"ball buys noise-tolerance by flattening")
    elif wide_fires:
        verdict = "WIDER-EXPLORATION"
        clause = (f"the walled spans are intrinsically wider (PRs "
                  + "/".join(f"{rid} {pr_w[rid]:.3f} = {pr_w[rid]/pr_p:.2f}x"
                             for rid in pr_w)
                  + f" vs pristine {pr_p:.3f}; >= 2 of 3 over {WIDER_RATIO}x) "
                  f"— the projection kept the trajectory exploring; the band "
                  f"is the history's shadow")
    else:
        clause_bits = []
        if not spectra_match and not spectra_wider:
            clause_bits.append(f"the spectra neither match nor widen by the "
                               f"registered clauses (PR ratios "
                               + "/".join(f"{rid} {pr_w[rid]/pr_p:.3f}x"
                                          for rid in pr_w)
                               + f"; shape {sum(shape_bits.values())/len(shape_bits):.3f} bits)")
        elif spectra_match:
            clause_bits.append(f"the spectra MATCH (PRs in window, shape "
                               f"{sum(shape_bits.values())/len(shape_bits):.3f} bits)")
        if not curv_clause:
            clause_bits.append(f"the curvature clause fails (geomean ratio "
                               f"{geo_ratio:.3f} vs bar {CURV_RATIO_BAR}; "
                               f"{n_ratio_le}/3 roots individually under)")
        verdict = "GRADED"
        clause = ("a partial/mix — " + "; ".join(clause_bits)
                  + " — the tables verbatim")

    metrics["adjudication"] = {
        "bars": {"FLATTENED_BASIN": {"fires": flat_fires},
                 "WIDER_EXPLORATION": {"fires": wide_fires},
                 "GRADED": {"fires": graded_fires}},
        "inputs": {
            "pristine_pr_primary": pr_p,
            "walled_prs_primary": pr_w,
            "pr_ratios_vs_pristine": {rid: pr / pr_p for rid, pr in pr_w.items()},
            "shape_bits_per_root": shape_bits,
            "shape_bits_mean": (sum(shape_bits.values()) / len(shape_bits)
                                if shape_bits else None),
            "spectra_match_clause": spectra_match,
            "spectra_wider_clause": spectra_wider,
            "curv4_pristine_wash": curv_p,
            "curv4_walled_wash": curv_w,
            "curv_ratios_vs_pristine": ratios,
            "curv_geomean_ratio": geo_ratio,
            "curv_n_roots_le_bar": n_ratio_le,
            "curv_clause": curv_clause,
        },
        "verdict": verdict, "clause": clause,
        "composite_order": "FLATTENED-BASIN / WIDER-EXPLORATION / GRADED "
                           "(frozen before compute; the first two mutually "
                           "exclusive by construction)",
        "desk_pre_read_followup": {
            "e_chart_desk_pr": ech_pr_desk,
            "fresh_pristine_pr": pr_p,
            "note": "the desk pre-read (e_chart PR 4.214 vs walled "
                    "4.216-4.263, cross-instrument) is resolved by the fresh "
                    f"same-instrument pristine PR {pr_p:.4f}; reported, the "
                    "verdict owned by the fresh numbers",
        },
    }
    log("=" * 78)
    log(f"E211 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- provenance + honesty ----------------------------------------------
    metrics["provenance"] = {
        "parents_files_md5": parents_md5,
        "machinery": {
            "histories_and_spans": "lab/e209_census_debt.py VERBATIM (the "
                                   "settled loads, the contiguous 20-step "
                                   "unwalled AdamW wash, the Gram-SVD span, "
                                   "fp64), re-derived at all four states in "
                                   "ONE run — the same-instrument cell",
            "kills": "e209's onset instrument verbatim (grid 0.05..3.00, "
                     "install-60 g-12 ruler, 0.27 bar, linear-in-D "
                     "interpolated first downcrossing, early stop)",
            "curvature": "NEW this cell (registered): 3-point FD second "
                         "difference on a FIXED batch (the primary-history "
                         "step-1 batch CE = the wash's own objective) and "
                         "the ruler battery -log pZ; eps ladder "
                         f"{list(EPS_GRID)}, primary {EPS_PRIMARY:g}",
            "committed_records": "e209 (PRs, bands, settled md5s), g1b "
                                 "(C-arm rows 1..20 = the pristine anchors), "
                                 "g1bR (s300 reads), e_chart (the pristine "
                                 "root read + spectrum + band) — loaded, "
                                 "hard-bound, never re-run",
        },
        "per_root": {r["id"]: {
            "state": "pristine e131 root as committed" if r["id"] == "P"
                     else "e209's settled body (md5-gated: "
                          + str(metrics["roots"][r["id"]]["G_ROOT"]
                                .get("settled_flat_md5_match_e209", "n/a")) + ")",
            "span_realizations": 1 + len(metrics["roots"][r["id"]]
                                         .get("span_robustness", [])),
        } for r in ROOTS},
    }
    hist_deaths = {rid: metrics["roots"][rid]["wash_history"]
                   ["fact_died_in_history_at_step"] for rid in metrics["roots"]}
    metrics["honesty"] = {
        "n_and_scope": ("n=3 walled roots + 1 pristine root; every span is a "
                        "single 20-step history realization (robustness "
                        "spans reported as spread); every curvature is a "
                        "3-point FD estimate on one fixed batch; every "
                        "per-direction kill is one ray; the draw lottery "
                        "(T155) moves every one of these on redraw"),
        "fd_scale_caveat": (f"the FD second difference reads curvature at the "
                            f"eps scale: primary eps {EPS_PRIMARY:g} L2 "
                            "(quartic contamination possible; the eps ladder "
                            "{2e-3,5e-3} rows report the drift; float32 "
                            "roundoff grows as eps shrinks). The instrument "
                            "is a RANKER between states, not an absolute "
                            "Hessian read"),
        "loss_scale_caveat": ("the states sit at different base losses "
                              "(pristine L0_wash "
                              f"{metrics['roots']['P']['curvature']['L0_wash']:.3f} "
                              "vs walled "
                              + "/".join(f"{metrics['roots'][rid]['curvature']['L0_wash']:.3f}"
                                         for rid in pr_w)
                              + "): the RAW curvature ratio adjudicates per "
                              "the registration; the normalized co-read "
                              "(curv/L0) is reported per root, never "
                              "adjudicated"),
        "sign_convention": ("the kill rays use the SVD-emitted direction "
                            "sign; curvature is sign-invariant — the "
                            "kill-vs-curvature correlation carries this flag"),
        "settled_state_load": ("the walled checkpoints store MID-STRIDE "
                               "states (d_raw ~1.46 > R 0.7); everything "
                               "starts at the SETTLED body (e209's "
                               "convention, md5-gated); the pristine root is "
                               "used as committed (no ball of its own)"),
        "cross_instrument_joins": ("the committed-band join's pristine row is "
                                   "e_chart's FINE_GRID cap-1.5 co-read (1-of-3 "
                                   "censored) vs the walled onset-grid "
                                   "medians; e_chart's committed pristine "
                                   "spectrum is a mixed-lr ladder, not a "
                                   "contiguous wash — both joins are "
                                   "context, never adjudicated"),
        "nothing_guaranteed": ("the openness was the point: the fresh "
                               "pristine PR, the curvatures, and the "
                               "correlations were all unmeasured before this "
                               f"cell; the observed outcome is '{verdict}'"),
    }
    metrics["status"] = "COMPLETE — adjudicated (this write replaces all " \
                        "PARTIAL progressive writes)"
    write_partial("P4 adjudicated (+ provenance + honesty)")

    # ================= P4b: the figure ========================================
    plot_cell(rd / "e211_walled_band.png", metrics, verdict)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'e211_walled_band.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_cell(path, metrics, verdict):
    import numpy as np
    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(2, 3)
    colors = {"P": "black", "R5": "darkorange", "R6": "seagreen",
              "R7": "royalblue"}
    labels = {"P": "P (pristine e131)", "R5": "R5 (g1b W1 s10902)",
              "R6": "R6 (g1bR W1 s10907)", "R7": "R7 (g1bR W1 s10908)"}

    # (0,0) the SV spectra
    ax = fig.add_subplot(gs[0, 0])
    for rid, c in metrics["roots"].items():
        sp = metrics["roots"][rid]["span_primary"]
        ax.plot(range(1, len(sp["sv"]) + 1), sp["sv"], "o-", ms=3.5, lw=1.4,
                color=colors.get(rid, "gray"), label=labels.get(rid, rid))
    ax.set_yscale("log")
    ax.set_xlabel("SV index (the contiguous 20-step wash span)")
    ax.set_ylabel("singular value (L2 units)")
    ax.set_title("THE SPAN-WIDTH COMPARISON — the SV spectra (primary "
                 "realizations; Adam's step size homogenizes sv1 by "
                 "construction)", fontsize=9)
    ax.legend(fontsize=8)

    # (0,1) PR bars + spread
    ax = fig.add_subplot(gs[0, 1])
    rids = [rid for rid in ("P", "R5", "R6", "R7") if rid in metrics["roots"]]
    prs = [metrics["roots"][rid]["span_primary"]["participation_ratio"]
           for rid in rids]
    lo = [prs[0] * MATCH_LO] * len(rids)
    hi = [prs[0] * MATCH_HI] * len(rids)
    ax.fill_between(range(len(rids)), lo, hi, color="gray", alpha=0.15,
                    label=f"the MATCH window [{MATCH_LO},{MATCH_HI}]x pristine")
    for i, rid in enumerate(rids):
        spread = metrics["roots"][rid].get("span_pr_spread")
        if spread:
            ax.errorbar(i, prs[i],
                        yerr=[[prs[i] - spread["min"]], [spread["max"] - prs[i]]],
                        fmt="none", ecolor=colors.get(rid, "gray"), alpha=0.6,
                        capsize=4)
        ax.bar(i, prs[i], color=colors.get(rid, "gray"), alpha=0.85)
        ax.annotate(f"{prs[i]:.3f}", (i, prs[i]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.axhline(prs[0] * WIDER_RATIO, ls="--", lw=1.3, color="crimson",
               label=f"the WIDER bar {WIDER_RATIO}x pristine")
    ax.set_xticks(range(len(rids)))
    ax.set_xticklabels(rids)
    ax.set_ylabel("participation ratio of the span spectrum")
    ax.set_title("PR (error bars = the robustness realization spread, "
                 "context)", fontsize=9)
    ax.legend(fontsize=7.5)

    # (0,2) normalized profiles
    ax = fig.add_subplot(gs[0, 2])
    for rid in rids:
        prof = metrics["roots"][rid]["span_primary"]["profile"]
        ax.plot(range(1, len(prof) + 1), prof, "o-", ms=3, lw=1.2,
                color=colors.get(rid, "gray"),
                label=labels.get(rid, rid))
    ax.set_yscale("log")
    ax.set_xlabel("SV index")
    ax.set_ylabel("sv_i / sv_1 (the normalized profile)")
    ax.set_title("the SHAPE clause's input (mean |log2 profile ratio| "
                 f"<= {MATCH_SHAPE_BITS} bits)", fontsize=9)
    ax.legend(fontsize=8)

    # (1,0) curvature
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(rids))
    cw = [metrics["roots"][rid]["curvature"]["curv4_median_wash"]
          for rid in rids]
    cr = [metrics["roots"][rid]["curvature"]["curv4_median_rul"]
          for rid in rids]
    ax.bar(xs - 0.18, cw, 0.34, color=[colors.get(r) for r in rids],
           alpha=0.9, label="L_wash (fixed batch CE) — PRIMARY")
    ax.bar(xs + 0.18, cr, 0.34, color=[colors.get(r) for r in rids],
           alpha=0.45, hatch="//", label="L_rul (-log pZ) — co-read")
    for i, rid in enumerate(rids):
        if rid != "P":
            ratio = cw[i] / cw[0]
            ax.annotate(f"{ratio:.2f}x", (i - 0.18, cw[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=8, weight="bold",
                        color="crimson" if ratio <= CURV_RATIO_BAR else "gray")
    ax.axhline(cw[0] * CURV_RATIO_BAR, ls="--", lw=1.4, color="crimson",
               label=f"the 2x-lower bar ({cw[0]*CURV_RATIO_BAR:.3g})")
    ax.set_xticks(xs)
    ax.set_xticklabels(rids)
    ax.set_ylabel("median FD curvature, top-4 span directions "
                  f"(eps={EPS_PRIMARY:g})")
    ax.set_title("THE CURVATURE CO-READ — the FLATTENED-BASIN axis", fontsize=9)
    ax.legend(fontsize=7.5)

    # (1,1) kill vs SV energy; (1,2) kill vs curvature
    corr = metrics["band_re_read"]["correlations_context"]
    for k, (ykey, ylab) in enumerate((("energy_frac", "SV energy fraction "
                                        "(sv_i^2 / sum sv^2)"),
                                      ("curv_wash", f"FD curvature "
                                        f"(L_wash, eps={EPS_PRIMARY:g})"))):
        ax = fig.add_subplot(gs[1, 1 + k])
        for rid in rids:
            pts = [(d["D_kill"], d[ykey]) for d in metrics["roots"][rid]
                   ["dir_kills"] if d["D_kill"] is not None]
            if pts:
                ax.plot([p[1] for p in pts], [p[0] for p in pts], "o", ms=6,
                        color=colors.get(rid), alpha=0.8,
                        label=labels.get(rid, rid))
        key = ("spearman_kill_vs_sv" if k == 0 else "spearman_kill_vs_curv")
        txt = " | ".join(
            f"{rid}: rho {corr[rid][key]:.2f}" for rid in rids
            if rid in corr and key in corr[rid] and corr[rid][key] == corr[rid][key])
        pool = corr.get("pooled", {}).get(key)
        pool_s = f"{pool:.2f}" if isinstance(pool, (int, float)) and pool == pool else "n/a"
        ax.set_xlabel(ylab)
        ax.set_ylabel("per-direction kill-D (the onset instrument)")
        ax.set_title(f"THE BAND RE-READ: kill vs "
                     f"{'SV weight' if k == 0 else 'curvature'} "
                     f"(pooled rho {pool_s})\n{txt}", fontsize=9)
        ax.legend(fontsize=7.5)

    fig.suptitle(f"E211 — THE WALLED-BAND QUESTION: why does the wall widen "
                 f"the noise ball?    VERDICT: {verdict}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
