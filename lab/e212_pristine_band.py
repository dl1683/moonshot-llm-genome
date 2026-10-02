"""E212 — THE SAME-INSTRUMENT PRISTINE BAND (e211's named debt, paid).

WHY: e211 (runs/e211/metrics.json, T182) retired the instrument shadow —
same-instrument, the walled spans match the pristine's (PR 4.216/4.263/
4.260 vs pristine 4.117, spectra within 0.058 bits; basins within 1.08x)
and the walled per-direction medians sit at 0.85/0.90/0.92 — but the
committed table's PRISTINE ROW was still e_chart's 0.610, a mixed-lr
checkpoint-ladder FINE_GRID cap-1.5 cross-instrument read (1-of-3
censored). T182's letter: "the same-instrument pristine band the named
debt if the scalar is ever wanted." This cell pays it: the random-draw
band at the PRISTINE e131 root on the SAME instrument as the walled rows
(e209's onset-grid band — the same span construction, the same fine
grid, the same draw count 3+2) — the honest apples-to-apples pristine
band. THE HONEST BAND TABLE: is the wall's band wider at the same
instrument AT ALL, or fully matched (the shadow's last word)?

WHAT IT BUILDS ON: e211's same-instrument pristine span machinery
VERBATIM (the pristine e131 root loaded as committed; the seed-10902
stream = g1b's own C-arm stream, G_STREAM md5-gated vs e185; the
20-step contiguous unwalled AdamW wash; the fp64 Gram-SVD span; the
robustness spans 12401/12402); e209's band instrument VERBATIM (the
Gaussian in-span draws, the onset grid 0.05..3.00, the install-60 g-12
ruler, the 0.27 DISSOLVE bar, the interpolated first downcrossing,
e205's middle-order-statistic median with right-censoring); e211's and
e209's committed records (the walled per-direction kill lists, the
walled random-draw bands, e_chart's retired pristine row), loaded and
hard-bound, never re-run.

WHAT IS NEW: the pristine RANDOM-DRAW band itself. No instrument has
ever drawn random in-span rays at the pristine root on the walled rows'
onset-grid instrument (e_chart's 0.610 was the mixed-ladder
cross-instrument read; e211's pristine rays were PURE top-8 SV
directions — a different direction family). n=5 draws = 3 fresh
Gaussian draws in the primary pristine span (seeds 12601-12603,
registry-clean) + 1 fresh Gaussian draw in EACH robustness span
(12604/12605) — the dispatch's "3+2".

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - BANDS-MATCH: "fires if the same-instrument pristine median lands
    within the walled family's own draw-spread (0.85-0.92 ± their
    lottery swing) — the band fully matched; the shadow's last word;
    the band scalar closed as a non-property."
  - BANDS-GAP-REMAINS: "fires if the same-instrument pristine median
    lands materially below the walled rows (< 0.7x the walled median)
    — a real gap survives the instrument fix; the walled-band property
    partially resurrects; the follow-on named."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * the pristine band = n=5 random in-span draws (the dispatch's
    "3+2"): draws 12601-12603 in the PRIMARY span, 12604/12605 in the
    robustness spans (12401/12402); the draw construction is e209's
    verbatim (c ~ N(0,I) over the Vp basis coefficients, v = Vp^T c,
    u = v/||v|| cast fp32).
  * the spans = the pristine root's OWN contiguous 20-step unwalled
    AdamW wash (lr 1e-3, betas (0.9,0.95), wd 0.1, clip 1.0, batch 32
    = 16 neutral-anchor + 16 random, e209's draw order verbatim);
    primary from the seed-10902 stream (= g1b's own C-arm stream; every
    step's L2 gated vs g1b's committed C-arm rows 1..20, tol 5e-3
    cross-device; the span PR AND all 20 SVs gated vs e211's committed
    pristine span, 1e-6 rel); robustness gated the same way (PR +
    step-1 L2 vs e211's committed robustness rows, 1e-6).
  * kill = first g-12 <= 0.27 downcrossing on the onset grid
    0.05..3.00 + D=0 anchor, linear-in-D interpolated, early stop
    (e209/e211's instrument verbatim); unresolved-high -> None
    (right-censored at 3.0).
  * band median = e205's middle order statistic with right-censoring
    (majority-censored -> UNDEFINED -> GRADED with the table).
  * the walled rows = e211's committed PER-DIRECTION medians over the
    RESOLVED top-8 span-direction kills (the dispatch's 0.85/0.90/
    0.92; recomputed at load from e211's committed kill lists, exact):
    R5 0.846233 / R6 0.895479 / R7 0.920202; the None->inf variant is
    reported, never adjudicated.
  * walled family = {min 0.846233, max 0.920202, median 0.895479};
    lottery swing = (max-min)/2 = 0.036985 (the family's own realized
    swing — three independent lottery realizations of the walled
    median); BANDS-MATCH window = [min-swing, max+swing] =
    [0.809248, 0.957187]; BANDS-GAP threshold = 0.7 x family median =
    0.626835. The window's low edge sits ABOVE the gap threshold: the
    two bars are mutually exclusive BY CONSTRUCTION. Composite order
    BANDS-MATCH / BANDS-GAP-REMAINS / GRADED.
  * BANDS-MATCH fires iff the pristine median is defined and lands in
    [win_lo, win_hi]; BANDS-GAP-REMAINS fires iff defined and < 0.7x
    family median; GRADED = any residual (undefined band; between
    threshold and window; above the window).
  * CONTEXT, never adjudicated: (i) the wider within-root-SE window
    (max over walled roots of 1.2533 s_i/sqrt(n_i) from e211's
    committed kills — the most generous lottery-swing reading);
    (ii) the like-for-like RANDOM-DRAW join (this band vs e209's
    committed walled random-draw medians 1.580/1.706/0.914);
    (iii) e211's pristine PER-DIRECTION median (the same root read by
    the walled rows' own direction family); (iv) e_chart's retired
    0.610 row.

DESK PRE-READ DISCLOSURE (stated before compute, e205/e211's
convention): the MATCH window [0.809, 0.957] and the GAP threshold
0.627 are desk-fixed by committed numbers — unavoidable, disclosed.
THE OPEN AXIS IS THE PRISTINE RANDOM-DRAW MEDIAN ITSELF: no instrument
has ever read it (e211's pristine per-direction median 1.031 is a
DIFFERENT direction family — pure SV rays, no low-energy mixing; the
registered join compares the pristine random family to the walled
per-direction family by the dispatch's frozen letter, and both
cross-family joins ride as context). No prediction registered; no bar
desk-forced; no bar shopping.

PRE-DISPATCH CHECKS (Rule 12): the protocol gate set (G_NAMEFREE /
G_SPLICE / G_BATTERY / G_ANCHOR, e209/e211's conventions); G_PARENTS
(e211 + e209 + e205 + e_chart + g1b metrics md5s; the walled resolved
medians recomputed exactly from e211's committed kill lists and pinned
to the dispatch's 0.85/0.90/0.92; e209's band medians recomputed from
its committed kill_Ds; e_chart's root read + retired band; g1b's
C-arm rows 1..20); G_ROOT (the pristine root's provenance: the e113
recipe meta, seed 10901, base e048_repro, 2,739,072 params; the
install-60 g-12 read bit-gated vs e_chart's committed rung-0 read);
G_STREAM (seed-10902 step-1 CE + AdamW L2 vs g1b's committed C-arm
row + the e185 batch md5); G_HIST (per-step L2 vs g1b's committed
C-arm rows); G_SPAN/G_ROBUST (the PR + SV-spectrum reproduction vs
e211's committed pristine spans — the instrument-identity gate: the
same span/grid arithmetic, verified). WHAT THE GATES GUARANTEE:
NOTHING — the pristine band could land at 0.4, at 1.0, at 2.0; the
openness is the point.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is never claimed), torch threads 4 (the
e185/e209/e211 reduction order), load-check recorded not gating
(e204/e205/e209's convention), small eval bursts, progressive
metrics.json writes after every phase and every draw, decisive.

Outputs: runs/e212/{metrics.json (PROGRESSIVE), e212_pristine_band.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e212_pristine_band.py    (E212_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e185/e209/e211)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # the e185/e209/e211 reduction order

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E212_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e212 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E131_ROOT_CK = "e131_consolidated_e113.pt"     # THE pristine root itself
E211_METRICS = E43.REPO / "runs" / "e211" / "metrics.json"
E209_METRICS = E43.REPO / "runs" / "e209" / "metrics.json"
E205_METRICS = E43.REPO / "runs" / "e205" / "metrics.json"
E_CHART_METRICS = E43.REPO / "runs" / "e_chart" / "metrics.json"
G1B_METRICS = E43.REPO / "runs" / "g1b" / "metrics.json"

GEOS = (-12, 0, 12)               # battery ctx offsets (e185/e209/e211 convention)
RULER_J = -12                     # THE RULER: install-60 g-12 (e191/e209/e211's)
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — the kill convention everywhere
LR_ADAMW = 1e-3                   # the wash recipe (g1/g1b/g1bR + e205/e209/e211)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
N_PARAMS = 2_739_072
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 — e199/e205/e209/e211's onset grid VERBATIM
WASH_HIST_STEPS = 4 if SMOKE else 20        # e193b/e205/e209/e211's history segments

PRISTINE_SEED = 10902             # g1b's own C-arm stream (the root's own wash stream)
ROBUST_SEEDS = (12401, 12402)     # e211's committed pristine robustness spans

# the 5 draws (the dispatch's "3+2"; registry-clean, repo-grep'd):
# 3 fresh Gaussian draws in the primary span + 1 in each robustness span
DRAWS = [{"seed": 12601, "span": "primary"},
         {"seed": 12602, "span": "primary"},
         {"seed": 12603, "span": "primary"},
         {"seed": 12604, "span": "robust:12401"},
         {"seed": 12605, "span": "robust:12402"}]

# the walled family (e211's committed per-direction medians — recomputed at
# load from the committed kill lists; the dispatch's 0.85/0.90/0.92)
WALLED_IDS = ("R5", "R6", "R7")
DISPATCH_PINNED_MEDIANS = {"R5": 0.85, "R6": 0.90, "R7": 0.92}   # rounded pins (tol 0.05)
GAP_RATIO = 0.7                   # "materially below" = < 0.7x the walled median (registered)

# e211's committed pristine-span records (hard-bound; the reproduction IS the gate)
E211_PRISTINE = {
    "primary_pr": 4.117363094870968,
    "primary_sv": [1.9597355390295585, 1.4676190745815085, 1.0840231000069138,
                   0.8516531991871176, 0.6754297556601792, 0.5446478048652162,
                   0.4751903432789083, 0.37694177303028514, 0.3406220016987602,
                   0.2792117231232169, 0.23479511170828862, 0.2120952460349488,
                   0.1816930659234477, 0.16720944597868573, 0.15851014009334544,
                   0.12365014301894246, 0.11241917367756023, 0.10215793762375433,
                   0.09377184368376072, 0.08634923178214998],
    "robust": {12401: {"pr": 4.316515143812096, "step1_L2": 1.6542576551437378},
               12402: {"pr": 4.204382894732309, "step1_L2": 1.6540577411651611}},
}

# e209's committed walled random-draw bands (the like-for-like context join)
E209_BANDS = {
    "R5": {"kill_Ds": [1.579816428873246, 1.6956607003316866, 1.1389369879609799],
           "median": 1.579816428873246},
    "R6": {"kill_Ds": [1.089780632096794, 1.7055491980018755, 1.8256070805237203],
           "median": 1.7055491980018755},
    "R7": {"kill_Ds": [0.4698055204802328, 1.5379877124339503, 0.9141374091130942],
           "median": 0.9141374091130942},
}

ECHART_PRISTINE_READ = 0.9155886173248291        # e_chart partB.e131.wash curve rung 0
ECHART_RETIRED_BAND = {"kill_Ds": [0.6100797132671589, None, 0.5636664436670049],
                       "median": 0.6100797132671589,
                       "instrument": "e_chart FINE_GRID 11pts 0.05..1.5 (1-of-3 "
                                     "right-censored > 1.5) — the mixed-ladder "
                                     "CROSS-INSTRUMENT read the committed table "
                                     "carried as its pristine row (retired)"}

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH_1 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's seed-10902 step-1 batch md5

G_BIT_TOL = 5e-6                  # root read vs committed (bit tier)
G_HIST_TOL = 5e-3                 # pristine per-step L2 vs g1b CUDA rows
G_PR_TOL = 1e-6                   # span PR reproduction vs e211 (CPU->CPU, same threads)

REGISTERED_BARS = {
    "BANDS_MATCH": "BANDS-MATCH: \"fires if the same-instrument pristine median "
        "lands within the walled family's own draw-spread (0.85-0.92 ± their "
        "lottery swing) — the band fully matched; the shadow's last word; the "
        "band scalar closed as a non-property.\"",
    "BANDS_GAP_REMAINS": "BANDS-GAP-REMAINS: \"fires if the same-instrument "
        "pristine median lands materially below the walled rows (< 0.7x the "
        "walled median) — a real gap survives the instrument fix; the "
        "walled-band property partially resurrects; the follow-on named.\"",
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": (
        "the pristine band = n=5 random in-span draws (the dispatch's 3+2): "
        "12601-12603 in the primary span, 12604/12605 in the robustness spans "
        "(12401/12402), e209's draw construction verbatim (c ~ N(0,I) over the "
        "Vp basis, u = Vp^T c / ||.||); spans = the pristine root's OWN "
        "contiguous 20-step unwalled AdamW wash (e209's recipe verbatim; the "
        "seed-10902 stream = g1b's own C-arm stream, per-step L2 gated vs "
        "g1b's committed rows 1..20 tol 5e-3; span PR + all 20 SVs gated vs "
        "e211's committed pristine span 1e-6 rel; robustness spans gated the "
        "same); kill = first g-12 <= 0.27 downcrossing on the onset grid "
        "0.05..3.00 + D=0 anchor, linear-in-D interpolated, early stop; "
        "unresolved-high -> None (right-censored); band median = e205's middle "
        "order statistic with right-censoring (majority-censored -> undefined "
        "-> GRADED); the walled rows = e211's committed per-direction medians "
        "over RESOLVED top-8 kills (the dispatch's 0.85/0.90/0.92; recomputed "
        "at load, exact; the None->inf variant reported never adjudicated); "
        "walled family {min,max,median}; lottery swing = (max-min)/2 (the "
        "family's own realized swing); MATCH window = [min-swing, max+swing]; "
        "GAP threshold = 0.7 x family median; the window's low edge sits above "
        "the gap threshold — the two bars mutually exclusive BY CONSTRUCTION; "
        "composite order BANDS-MATCH / BANDS-GAP-REMAINS / GRADED; CONTEXT "
        "never adjudicated: the within-root-SE wider window, the like-for-like "
        "random-draw join vs e209's walled bands, e211's pristine "
        "per-direction median, e_chart's retired 0.610 row."),
    "desk_pre_read_disclosure": (
        "the MATCH window [0.809, 0.957] and GAP threshold 0.627 are "
        "desk-fixed by committed numbers (disclosed, unavoidable). THE OPEN "
        "AXIS is the pristine random-draw median itself — never measured "
        "(e211's pristine per-direction median 1.031 is a different direction "
        "family; e_chart's 0.610 is cross-instrument). No prediction "
        "registered; no bar desk-forced; no bar shopping."),
    "registration": "the dispatch's registration IS the registration (the "
        "three bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE PRISTINE BAND IS FRESH (disclosed before compute): the committed "
    "table's pristine row was e_chart's mixed-ladder FINE_GRID cap-1.5 "
    "cross-instrument co-read (0.610, 1-of-3 censored) — it cannot serve as "
    "the same-instrument pristine band. This cell draws random in-span rays "
    "at the pristine root on e209's onset-grid instrument verbatim.",
    "THE WALLED ROWS ARE e211's COMMITTED PER-DIRECTION MEDIANS (not "
    "re-derived): the dispatch's frozen join — the walled side is the "
    "same-grid same-ruler same-run family from e211's same-instrument cell "
    "(resolved-only medians; the None->inf variant and e209's random-draw "
    "walled bands ride as context, never adjudicated).",
    "THE DRAW COUNT IS 3+2 (the dispatch's letter): 3 draws in the primary "
    "span + 1 in each of e211's two robustness spans — 5 draws total, "
    "matching e209's 3-draw band + e211's 2-robustness-span structure at "
    "the walled rows.",
    "THE REGISTERED JOIN CROSSES DIRECTION FAMILIES (disclosed, frozen): the "
    "pristine side is RANDOM draws (energy-weighted toward the safe top-SV "
    "directions, c ~ N(0,I) => expected energy fraction_i ~ sv_i^2); the "
    "walled side is e211's PURE top-8 SV rays (which include the fast-killing "
    "low-energy pure directions). Both like-for-like joins (random-vs-random "
    "vs e209; dirs-vs-dirs vs e211) are reported as CONTEXT, never "
    "adjudicated.",
    "The load-check is recorded, not gating (e204/e205/e209's convention).",
    "Smoke mode trims: 4-step histories, 1 robustness span, 2 draws (1 "
    "primary + 1 robust), grid {0.05, 0.5, 1.5, 3.0}; nothing adjudicated "
    "(verdict stamped SMOKE).",
]

# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e211_walled_band.py + lab/e209_census_debt.py VERBATIM
# (whose own provenance is e191/e205 via e185 — the e176n lineage). Copied
# rather than imported to own the device policy and the arithmetic.


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
    """e068/e113/e209/e211 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "frac_argmax_z": amax / ids.shape[0]}


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) — e199/e205/e209/e211's instrument."""
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
    """e_chart/e193b/e205/e209/e211's Gram-based right-singular basis (fp64)."""
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


def resolved_median(kill_Ds: list) -> float | None:
    """Median over RESOLVED kills (the dispatch's 0.85/0.90/0.92 convention)."""
    r = sorted([d for d in kill_Ds if d is not None])
    if not r:
        return None
    return r[len(r) // 2] if len(r) % 2 == 1 else 0.5 * (r[len(r) // 2 - 1] + r[len(r) // 2])


def median_se(kill_Ds: list) -> float | None:
    """Asymptotic SE of the median from the within-root kill spread (CONTEXT)."""
    r = [d for d in kill_Ds if d is not None]
    n = len(r)
    if n < 3:
        return None
    mu = sum(r) / n
    s = (sum((x - mu) ** 2 for x in r) / (n - 1)) ** 0.5
    return 1.2533 * s / n ** 0.5


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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e212_smoke" if SMOKE else "e212")
    metrics: dict = {
        "experiment": "e212_pristine_band",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never claimed)",
            "torch_threads": torch.get_num_threads(),
            "load_check_recorded_not_gating": True,
            "phases": "P0 gates -> P1 root + stream + histories + spans (the "
                      "instrument-identity gates) -> P2 the 3+2 draws -> P3 "
                      "the honest band table + adjudication -> P4 figure; "
                      "progressive writes",
            "root": f"P:{E131_ROOT_CK}",
            "draws": [f"{d['seed']}:{d['span']}" for d in DRAWS],
            "grid": list(D_GRID),
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
    log(f"E212 THE SAME-INSTRUMENT PRISTINE BAND (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()} — the owner "
        f"envelope's cap), load-check recorded (launch: {load0}%), "
        f"progressive writes, decisive")

    # ================= P0a: protocol rebuild (e209/e211's gate set) ===========
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
                "offsets {-12,0,+12} — e185/e209/e211's convention = "
                "g1b/g1bR's own battery (p(Z) on the install-60 pool)",
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
    for p in (E211_METRICS, E209_METRICS, E205_METRICS, E_CHART_METRICS,
              G1B_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")

    def md5of(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    e211m = json.loads(E211_METRICS.read_text(encoding="utf-8"))
    e209m = json.loads(E209_METRICS.read_text(encoding="utf-8"))
    g1bm = json.loads(G1B_METRICS.read_text(encoding="utf-8"))
    echm = json.loads(E_CHART_METRICS.read_text(encoding="utf-8"))
    parents_md5 = {"e211": md5of(E211_METRICS), "e209": md5of(E209_METRICS),
                   "e205": md5of(E205_METRICS), "e_chart": md5of(E_CHART_METRICS),
                   "g1b": md5of(G1B_METRICS)}

    # e211's verdict + the committed pristine spans (the reproduction targets)
    assert e211m["adjudication"]["verdict"] == "GRADED"
    e211_p = e211m["roots"]["P"]
    assert abs(e211_p["span_primary"]["participation_ratio"]
               - E211_PRISTINE["primary_pr"]) < 1e-12
    assert max(abs(a - b) for a, b in zip(e211_p["span_primary"]["sv"],
                                          E211_PRISTINE["primary_sv"])) < 1e-12
    e211_rob = {int(q["seed"]): q for q in e211_p["span_robustness"]}
    for sd_, rec in E211_PRISTINE["robust"].items():
        assert abs(e211_rob[sd_]["participation_ratio"] - rec["pr"]) < 1e-12
        assert abs(e211_rob[sd_]["step1_L2"] - rec["step1_L2"]) < 1e-12

    # the walled per-direction medians, recomputed EXACTLY from e211's kills
    walled_kills, walled_med, walled_med_inf = {}, {}, {}
    for rid in WALLED_IDS:
        ks = [d["D_kill"] for d in e211m["roots"][rid]["dir_kills"]]
        walled_kills[rid] = ks
        walled_med[rid] = resolved_median(ks)
        walled_med_inf[rid] = band_median(ks)["median"]
        assert walled_med[rid] is not None
        assert abs(walled_med[rid] - DISPATCH_PINNED_MEDIANS[rid]) < 0.05, \
            f"{rid}: resolved median {walled_med[rid]} vs dispatch pin"
    # e211's pristine per-direction median (the same-root context row)
    p_dir_median = resolved_median(
        [d["D_kill"] for d in e211_p["dir_kills"]])

    # e209's committed walled bands, rechecked from its own kill_Ds
    for rid in WALLED_IDS:
        cell209 = e209m["roots"][rid]
        kd = list(cell209["band"]["kill_Ds"])
        assert max(abs(a - b) for a, b in zip(kd, E209_BANDS[rid]["kill_Ds"])) < 1e-12
        m = band_median(kd)
        assert m["defined"] and abs(m["median"] - E209_BANDS[rid]["median"]) < 1e-12
        assert abs(cell209["band"]["median_stat"]["median"]
                   - E209_BANDS[rid]["median"]) < 1e-12

    # e_chart's retired pristine row + root read
    e131_spec = echm["partB_subspace"]["e131"]
    ech_root_read = e131_spec["wash"]["curve"][0]["gm12"]
    assert abs(ech_root_read - ECHART_PRISTINE_READ) < 1e-12
    assert abs(e211m["band_re_read"]["committed_band_join_context"][0]
               ["band_median_committed"] - ECHART_RETIRED_BAND["median"]) < 1e-12

    # g1b's C-arm rows 1..20 (the pristine history anchors)
    carm_rows = {int(r["step"]): r for r in g1bm["arms"]["C"]["traj"]}
    assert 1 in carm_rows and 20 in carm_rows

    fam = {"min": min(walled_med.values()),
           "max": max(walled_med.values()),
           "median": sorted(walled_med.values())[1]}
    swing = (fam["max"] - fam["min"]) / 2.0
    win_lo, win_hi = fam["min"] - swing, fam["max"] + swing
    gap_thresh = GAP_RATIO * fam["median"]
    assert win_lo > gap_thresh, "bars must be exclusive by construction"

    G_PARENTS = {
        "files_md5": parents_md5,
        "hardbound": {
            "e211.verdict": e211m["adjudication"]["verdict"],
            "e211.P.primary_PR": E211_PRISTINE["primary_pr"],
            "e211.P.primary_sv_n": len(E211_PRISTINE["primary_sv"]),
            "e211.P.robust_PRs": {str(k): v["pr"]
                                  for k, v in E211_PRISTINE["robust"].items()},
            "e211.walled_dir_kill_medians_resolved": walled_med,
            "e211.walled_dir_kill_medians_inf_conv": walled_med_inf,
            "e211.P.dir_kill_median_resolved": p_dir_median,
            "e209.walled_random_band_medians": {k: v["median"]
                                                for k, v in E209_BANDS.items()},
            "e_chart.retired_pristine_band_median": ECHART_RETIRED_BAND["median"],
            "e_chart.pristine_root_read": ech_root_read,
            "g1b.C_traj_rows_1_to_20": "loaded (the pristine history anchors)",
        },
        "recheck": "the walled resolved medians + e209 band medians + e_chart "
                   "row recomputed from the parents' committed lists: exact; "
                   "the resolved medians pin to the dispatch's 0.85/0.90/0.92",
        "pass": True,
    }
    log("P0b: parents hard-bound — e211 (GRADED; walled resolved medians "
        + "/".join(f"{walled_med[r]:.4f}" for r in WALLED_IDS)
        + f"); e209 bands /".join("") + "rechecked; e_chart 0.610 row bound; "
        f"g1b C-arm rows loaded; family min {fam['min']:.6f} max {fam['max']:.6f} "
        f"median {fam['median']:.6f}; swing {swing:.6f} -> window "
        f"[{win_lo:.6f}, {win_hi:.6f}]; gap threshold {gap_thresh:.6f}")
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    write_partial("P0b parent gates PASSED (e211/e209/e_chart/g1b hard-bound)")

    # ================= P0c: G_ROOT + G_STREAM at the pristine root ============
    net0, _, extra = load_body(CKPT_DIR / E131_ROOT_CK)
    n_par = sum(p.numel() for p in net0.parameters())
    assert n_par == N_PARAMS, f"param count {n_par}"
    theta0 = flat_params(net0)
    root_read = battery_cell(net0, ruler_ids, zid)["mean_pz"]
    meta = extra["meta"] or {}
    meta_ok = bool(meta.get("recipe", "").startswith("e113 verbatim")
                   and meta.get("seed") == 10901
                   and meta.get("base") == "runs/checkpoints/e048_repro.pt")
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{E131_ROOT_CK}",
        "meta": meta, "meta_provenance_gate": meta_ok, "n_params": n_par,
        "settled_projection_fired": False,
        "read_measured": root_read, "read_committed_e_chart": ECHART_PRISTINE_READ,
        "abs_diff": abs(root_read - ECHART_PRISTINE_READ),
        "bit_tol": G_BIT_TOL, "tol": 1e-3,
        "bit": bool(abs(root_read - ECHART_PRISTINE_READ) < G_BIT_TOL),
        "flat_md5": hashlib.md5(theta0.numpy().tobytes()).hexdigest(),
        "pass": bool(meta_ok
                     and abs(root_read - ECHART_PRISTINE_READ) < G_BIT_TOL),
        "note": "the PRISTINE root used AS COMMITTED (no anch__ ball of its "
                "own; no settling) — the root of every walled row's lineage "
                "(g1b/g1bR base) and of e211's same-instrument cell; the "
                "provenance gate: the e113 recipe (jittered replay 300 steps, "
                "seed 10901, base e048_repro); the read bit-gates vs "
                "e_chart's committed rung-0 install-60 g-12 read",
    }
    log(f"G_ROOT[P]: {n_par} params, meta "
        f"{'OK' if meta_ok else 'DRIFT'}, read {root_read:.6f} vs committed "
        f"{ECHART_PRISTINE_READ:.6f} (|d| {G_ROOT['abs_diff']:.1e}): "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("pristine root gate FAILED")
    metrics["gates"]["G_ROOT"] = G_ROOT
    metrics["roots"] = {"P": {"id": "P", "ckpt": E131_ROOT_CK,
                              "G_ROOT": G_ROOT,
                              "ruler_caveat": "install-60 g-12 (NOVEL geometry "
                              "— the e131 lineage's battery family); root read "
                              f"{root_read:.4f} >> 0.27: NO fallback (T155's "
                              "convention moot here)"}}
    write_partial("P0c G_ROOT PASSED (the pristine root, provenance-gated)")

    # G_STREAM: the seed-10902 stream at the pristine root = g1b's own C-arm
    g = torch.Generator().manual_seed(PRISTINE_SEED)
    aj = torch.randint(16, (ANCH_BS,), generator=g)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
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
    torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0)
    optw.step()
    disp1 = float(torch.norm(flat_params(tw) - theta0))
    del tw, optw, logits
    g1b_c1 = carm_rows[1]
    G_STREAM = {
        "seed": PRISTINE_SEED, "src": "runs/g1b/metrics.json arms.C.traj[0]",
        "step1_x_md5": x1_md5, "step1_x_md5_match_e185": x1_md5 == E185_XHASH_1,
        "ce1_measured": ce1, "ce1_committed": g1b_c1["ce_batch"],
        "d_ce": abs(ce1 - g1b_c1["ce_batch"]),
        "disp1_measured": disp1, "disp1_committed": g1b_c1["step_disp"],
        "d_disp": abs(disp1 - g1b_c1["step_disp"]), "tol": G_HIST_TOL,
        "pass": bool(abs(ce1 - g1b_c1["ce_batch"]) < G_HIST_TOL
                     and abs(disp1 - g1b_c1["step_disp"]) < G_HIST_TOL
                     and x1_md5 == E185_XHASH_1),
        "note": "the pristine stream IS g1b's own C-arm stream (the committed "
                "unwalled wash of this very root); step-1 CE + AdamW L2 vs the "
                "committed CUDA row (1e-7-class diffs are cross-device "
                "texture, e209/e211's pre-verified convention)",
    }
    log(f"G_STREAM[P]: CE |d| {G_STREAM['d_ce']:.1e}, disp |d| "
        f"{G_STREAM['d_disp']:.1e}, md5 "
        f"{'OK' if G_STREAM['step1_x_md5_match_e185'] else 'DRIFT'}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    if not G_STREAM["pass"]:
        raise RuntimeError("pristine stream gate FAILED")
    metrics["gates"]["G_STREAM"] = G_STREAM
    write_partial("P0c G_STREAM PASSED (the pristine stream = g1b's C-arm)")

    # ================= P1: the spans (the instrument-identity gates) =========
    log("=" * 78)
    log(f"P1: the pristine spans (primary seed {PRISTINE_SEED}; robustness "
        f"{list(ROBUST_SEEDS)})")

    def run_wash(seed_stream: int, trace_ruler: bool):
        """20-step contiguous unwalled AdamW wash from theta0 (e209/e211 verbatim)."""
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
            if s_wh == 1 and seed_stream == PRISTINE_SEED:
                assert hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest() \
                    == E185_XHASH_1
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
            wtheta = th_new
        wnet.eval()
        del wnet, wopt, evl
        return torch.stack(segs), rows

    # ---- primary history + G_HIST (per-step vs g1b's C-arm rows) -----------
    H, hist_rows = run_wash(PRISTINE_SEED, trace_ruler=True)
    hist_dead = next((h["step"] for h in hist_rows
                      if "gm12" in h and h["gm12"] <= SHUT_BAR), None)
    devs = [abs(h["L2"] - carm_rows[h["step"]]["step_disp"])
            for h in hist_rows]
    max_dev = max(devs)
    G_HIST = {
        "gate": "per-step displacement L2 vs g1b's committed C-arm rows 1.."
                + str(WASH_HIST_STEPS),
        "max_abs_dev_L2": max_dev,
        "per_step_dev_first3": [round(d, 8) for d in devs[:3]],
        "tol": G_HIST_TOL,
        "committed_death_step": 2, "measured_death_step": hist_dead,
        "pass": bool(max_dev < G_HIST_TOL),
        "note": "the pristine contiguous history IS g1b's C-arm wash (same "
                "stream, same recipe); per-step anchored vs the committed "
                "CUDA rows (cross-device texture tier)",
    }
    log(f"  G_HIST[P]: max per-step |dL2| {max_dev:.2e} "
        f"(death step in-history: {hist_dead}): "
        + ("PASS" if G_HIST["pass"] else "FAIL"))
    if not G_HIST["pass"]:
        raise RuntimeError("primary history gate FAILED")
    metrics["gates"]["G_HIST"] = G_HIST
    metrics["roots"]["P"]["wash_history"] = {
        "n_steps": WASH_HIST_STEPS, "rows": hist_rows,
        "step1_L2": hist_rows[0]["L2"], "cum_final": hist_rows[-1]["cum"],
        "fact_died_in_history_at_step": hist_dead,
        "co_read_note": "the C-arm's own committed death (+2) reproduced "
                        "in-history — the co-read, never adjudicated",
    }
    write_partial("P1 primary history gated (per-step vs g1b's C-arm)")

    # ---- primary span + G_SPAN (PR + full SV spectrum vs e211's commit) -----
    basis = svd_basis(H)
    sv = [float(s) for s in basis["sv"]]
    pr_rel = abs(basis["pr"] - E211_PRISTINE["primary_pr"]) / E211_PRISTINE["primary_pr"]
    sv_rel = max(abs(a - b) / b for a, b in zip(sv, E211_PRISTINE["primary_sv"]))
    G_SPAN = {
        "gate": "the span PR and all 20 SVs must reproduce e211's committed "
                "pristine span (same machine, same threads, same arithmetic)",
        "pr_measured": basis["pr"], "pr_committed": E211_PRISTINE["primary_pr"],
        "pr_rel_dev": pr_rel, "sv_max_rel_dev": sv_rel, "tol": G_PR_TOL,
        "pass": bool(pr_rel < G_PR_TOL and sv_rel < G_PR_TOL),
        "note": "THE INSTRUMENT-IDENTITY GATE (primary): the same span "
                "construction as the walled rows, verified by bit-grade "
                "reproduction of e211's committed pristine span",
    }
    log(f"  G_SPAN[P]: PR {basis['pr']:.6f} vs e211 "
        f"{E211_PRISTINE['primary_pr']:.6f} (rel {pr_rel:.1e}); max SV rel "
        f"{sv_rel:.1e}: " + ("PASS" if G_SPAN["pass"] else "FAIL"))
    if not G_SPAN["pass"]:
        raise RuntimeError("primary span gate FAILED")
    metrics["gates"]["G_SPAN"] = G_SPAN
    metrics["roots"]["P"]["span_primary"] = {
        "n_segments": int(basis["Vp"].shape[0]), "rank_eff": basis["rank_eff"],
        "sv": sv, "participation_ratio": basis["pr"], "cond": basis["cond"],
        "sv_top": sv[0],
        "energy_frac_top1": sv[0] ** 2 / sum(s * s for s in sv),
        "profile": [s / sv[0] for s in sv],
        "pr_vs_e211_rel_dev": pr_rel,
        "note": "the contiguous 20-step wash span — the same machinery at "
                "every state (e211's same-instrument cell), reproduced",
    }
    write_partial("P1 primary span gated (PR+SVs reproduce e211)")

    # ---- robustness spans + G_ROBUST ---------------------------------------
    robust_bases = {}
    robust_rows = []
    rob_seeds = (ROBUST_SEEDS[:1],) if SMOKE else ROBUST_SEEDS
    for rs in rob_seeds:
        Hr, rows_r = run_wash(rs, trace_ruler=False)
        br = svd_basis(Hr)
        rec = E211_PRISTINE["robust"][rs]
        pr_rel_r = abs(br["pr"] - rec["pr"]) / rec["pr"]
        l2_d = abs(rows_r[0]["L2"] - rec["step1_L2"])
        robust_bases[rs] = br
        robust_rows.append({
            "seed": rs, "sv": [float(s) for s in br["sv"]],
            "participation_ratio": br["pr"],
            "step1_L2": rows_r[0]["L2"],
            "pr_rel_dev_vs_e211": pr_rel_r,
            "step1_L2_abs_dev_vs_e211": l2_d,
            "gate_pass": bool(pr_rel_r < G_PR_TOL and l2_d < 1e-3),
        })
        log(f"  robustness wash s{rs}: PR {br['pr']:.6f} vs e211 "
            f"{rec['pr']:.6f} (rel {pr_rel_r:.1e}); step1 L2 |d| {l2_d:.1e}: "
            + ("PASS" if robust_rows[-1]["gate_pass"] else "FAIL"))
        del Hr
        if not robust_rows[-1]["gate_pass"]:
            raise RuntimeError(f"robustness span gate FAILED at seed {rs}")
        write_partial(f"P1 robustness span s{rs} gated (reproduces e211)")
    metrics["roots"]["P"]["span_robustness"] = robust_rows
    prs = [basis["pr"]] + [q["participation_ratio"] for q in robust_rows]
    metrics["roots"]["P"]["span_pr_spread"] = {
        "n_realizations": len(prs), "min": min(prs), "max": max(prs),
        "span_rel": (max(prs) - min(prs)) / basis["pr"],
        "note": "CONTEXT: the realization spread of the span-width scalar "
                "(matches e211's committed 4.117-4.317 range)",
    }
    del H
    metrics["envelope"]["cpu_load_pct_after_P1"] = cpu_load_probe()

    # ================= P2: the 3+2 draws (e209's instrument verbatim) ========
    log("=" * 78)
    log("P2: the 3+2 random in-span draws (e209's band instrument verbatim)")
    evl = copy.deepcopy(net0)
    evl.eval()

    def walk_kill(v: torch.Tensor, tag: str):
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

    draws = []
    for dspec in (DRAWS[:2] if SMOKE else DRAWS):
        sd_, span_tag = dspec["seed"], dspec["span"]
        if span_tag == "primary":
            B = basis
            sv_list = sv
            sid = f"s{sd_}:primary"
        else:
            rs = int(span_tag.split(":")[1])
            B = robust_bases[rs]
            sv_list = [float(s) for s in B["sv"]]
            sid = f"s{sd_}:{span_tag}"
        Vr64 = B["Vp"].contiguous().to(torch.float64)
        r_avail = B["rank_eff"]
        g_ = torch.Generator().manual_seed(sd_)
        c = torch.randn(r_avail, generator=g_, dtype=torch.float64)
        v = Vr64.T @ c
        u_in = (v / torch.norm(v)).to(torch.float32)
        # the draw's realized energy fractions along the SV ladder (context)
        e_fr = [float((ci * si) ** 2) for ci, si in zip(c.tolist(), sv_list)]
        tot_e = sum(e_fr)
        cum = sorted(e_fr, reverse=True)
        rows, kill = walk_kill(u_in, sid)
        draws.append({
            "seed": sd_, "span": span_tag, "u_md5": hashlib.md5(
                u_in.numpy().tobytes()).hexdigest(),
            "D_kill": kill, "rows": rows,
            "energy_frac_top1": cum[0], "energy_frac_top4": sum(cum[:4]),
            "energy_frac_top8": sum(cum[:8]),
            "energy_frac_total_check": tot_e,
            "note": "e209's draw construction verbatim: c ~ N(0,I) over the "
                    "Vp basis coefficients, u = Vp^T c / ||.|| (fp32)",
        })
        metrics["roots"]["P"]["band_draws_partial"] = {
            "kills_so_far": [d["D_kill"] for d in draws]}
        write_partial(f"P2 draw {sid} (kills so far "
                      f"{[d['D_kill'] for d in draws]})")
    metrics["roots"]["P"].pop("band_draws_partial", None)
    metrics["roots"]["P"]["band"] = {
        "n_draws": len(draws), "draw_structure": "3+2 (the dispatch's letter)",
        "kill_Ds": [d["D_kill"] for d in draws], "draws": draws,
        "median_stat": band_median([d["D_kill"] for d in draws]),
        "instrument": "e209's onset-grid band VERBATIM: grid 0.05..3.00 + "
                      "D=0 anchor, install-60 g-12 ruler, kill = first "
                      "g-12 <= 0.27 downcrossing INTERPOLATED linear-in-D "
                      "(early stop), None = right-censored at 3.0; the span "
                      "construction and the stream are the same machinery "
                      "that produced the walled rows' spans (G_SPAN/G_ROBUST "
                      "reproduce e211's committed pristine spans)",
        "ruler": "install-60 g-12",
    }
    med = metrics["roots"]["P"]["band"]["median_stat"]
    log(f"P2 COMPLETE: kills {[d['D_kill'] for d in draws]} -> median "
        + (f"{med['median']:.4f}" if med["defined"] else "UNDEFINED "
           f"({med['n_censored']}/{med['n']} censored)"))
    write_partial("P2 the pristine band drawn (median "
                  + (f"{med['median']}" if med["defined"] else "UNDEFINED") + ")")
    metrics["envelope"]["cpu_load_pct_after_P2"] = cpu_load_probe()

    # ================= P3: the honest band table + adjudication ==============
    log("=" * 78)
    log("P3: THE HONEST BAND TABLE + the frozen adjudication")
    p_med = med["median"]

    # the registered table: P-RANDOM (this run) joined to e211's walled dirs
    table_rows = [{
        "row": "P-RANDOM (THIS RUN)", "family": "pristine random-draw band",
        "median": p_med, "kills": [d["D_kill"] for d in draws],
        "n": len(draws),
        "instrument": "onset grid 0.05..3.00 interpolated; install-60 g-12 "
                      "ruler; 3+2 draws in the root's OWN contiguous-wash "
                      "spans (e209's instrument verbatim)",
        "source": "runs/e212 (fresh)",
    }]
    for rid in WALLED_IDS:
        table_rows.append({
            "row": f"{rid}-DIRS (e211)", "family": "walled per-direction",
            "median": walled_med[rid], "kills": walled_kills[rid],
            "n": len(walled_kills[rid]),
            "instrument": "the SAME grid + ruler (e211's same-instrument "
                          "cell): top-8 SV pure-direction rays, resolved-only "
                          "median",
            "source": "runs/e211 committed dir_kills (hard-bound)",
        })
    context_rows = [
        {"row": "P-DIRS (e211)", "family": "pristine per-direction",
         "median": p_dir_median,
         "kills": [d["D_kill"] for d in e211_p["dir_kills"]], "n": 8,
         "instrument": "same grid/ruler, top-8 SV pure directions — the "
                       "same-root read by the walled rows' direction family",
         "source": "runs/e211 committed (context)"},
        {"row": "R5/R6/R7-RANDOM (e209)", "family": "walled random-draw",
         "median": sorted([E209_BANDS[r]["median"] for r in WALLED_IDS])[1],
         "kills": {r: E209_BANDS[r]["kill_Ds"] for r in WALLED_IDS},
         "medians": {r: E209_BANDS[r]["median"] for r in WALLED_IDS},
         "n": 3, "instrument": "e209's 3-draw onset-grid bands at the walled "
                               "roots — the like-for-like RANDOM-DRAW walled "
                               "family",
         "source": "runs/e209 committed (context)"},
        {"row": "P-FINE-GRID (e_chart, RETIRED)",
         "family": "pristine cross-instrument (retired)",
         "median": ECHART_RETIRED_BAND["median"],
         "kills": ECHART_RETIRED_BAND["kill_Ds"], "n": 3,
         "instrument": ECHART_RETIRED_BAND["instrument"],
         "source": "the committed table's old pristine row (retired by this "
                   "cell's existence)"},
    ]
    metrics["honest_band_table"] = {
        "rows": table_rows, "context_rows_never_adjudicated": context_rows,
        "join": "THE REGISTERED JOIN: the P-RANDOM median vs the walled "
                "family {min 0.846233, max 0.920202, median 0.895479} "
                "(e211's resolved per-direction medians — the dispatch's "
                "0.85/0.90/0.92)",
    }

    # the frozen clauses
    match_fires = bool(p_med is not None and win_lo <= p_med <= win_hi)
    gap_fires = bool(p_med is not None and p_med < gap_thresh)
    graded_fires = not match_fires and not gap_fires
    assert not (match_fires and gap_fires), "bars must be exclusive"

    # context: the wider within-root-SE window (never adjudicated)
    ses = {rid: median_se(walled_kills[rid]) for rid in WALLED_IDS}
    wide_swing = max(s for s in ses.values() if s is not None)
    wide_window = {"swing": wide_swing,
                   "lo": fam["min"] - wide_swing, "hi": fam["max"] + wide_swing,
                   "note": "CONTEXT: the most generous lottery-swing reading "
                           "(max over walled roots of the within-root median "
                           "SE 1.2533 s/sqrt(n) from e211's committed kills); "
                           "never adjudicated",
                   "per_root_se": ses}
    like_for_like = {
        "pristine_random_median": p_med,
        "walled_random_medians_e209": {r: E209_BANDS[r]["median"]
                                       for r in WALLED_IDS},
        "walled_random_family": {"min": min(E209_BANDS[r]["median"] for r in WALLED_IDS),
                                 "max": max(E209_BANDS[r]["median"] for r in WALLED_IDS),
                                 "median": sorted(E209_BANDS[r]["median"]
                                                  for r in WALLED_IDS)[1]},
        "note": "CONTEXT: the like-for-like RANDOM-DRAW join; never "
                "adjudicated",
    }

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif match_fires:
        verdict = "BANDS-MATCH"
        clause = (f"the same-instrument pristine median {p_med:.4f} lands "
                  f"WITHIN the walled family's own draw-spread (window "
                  f"[{win_lo:.4f}, {win_hi:.4f}] = 0.85-0.92 ± the family's "
                  f"realized swing {swing:.4f}) — the band fully matched; the "
                  f"shadow's last word; the band scalar closed as a "
                  f"non-property")
    elif gap_fires:
        verdict = "BANDS-GAP-REMAINS"
        clause = (f"the same-instrument pristine median {p_med:.4f} lands "
                  f"MATERIALLY BELOW the walled rows ({p_med:.4f} < "
                  f"{gap_thresh:.4f} = 0.7 x the walled family median "
                  f"{fam['median']:.4f}; ratio {p_med / fam['median']:.3f}x) "
                  f"— a real gap survives the instrument fix; the walled-band "
                  f"property partially resurrects; the follow-on named")
    else:
        where = ("ABOVE the match window" if p_med is not None and p_med > win_hi
                 else "between the gap threshold and the window" if p_med is not None
                 else "UNDEFINED (majority-censored band)")
        clause = (f"a partial — the pristine median "
                  + (f"{p_med:.4f} lands {where} (window "
                     f"[{win_lo:.4f}, {win_hi:.4f}], gap threshold "
                     f"{gap_thresh:.4f})" if p_med is not None
                     else f"({med['n_censored']}/{med['n']} draws censored)")
                  + " — the tables verbatim")
        verdict = "GRADED"

    metrics["adjudication"] = {
        "bars": {"BANDS_MATCH": {"fires": match_fires, "window": [win_lo, win_hi]},
                 "BANDS_GAP_REMAINS": {"fires": gap_fires,
                                       "threshold": gap_thresh},
                 "GRADED": {"fires": graded_fires}},
        "inputs": {
            "pristine_band_median": p_med,
            "pristine_band_n": med["n"],
            "pristine_band_n_censored": med["n_censored"],
            "pristine_band_defined": med["defined"],
            "walled_medians_resolved": walled_med,
            "walled_medians_inf_convention": walled_med_inf,
            "walled_family": fam, "lottery_swing": swing,
            "match_window": [win_lo, win_hi], "gap_threshold": gap_thresh,
            "gap_ratio_bar": GAP_RATIO,
            "ratio_pristine_over_walled_median": (p_med / fam["median"]
                                                  if p_med is not None else None),
        },
        "verdict": verdict, "clause": clause,
        "context_never_adjudicated": {
            "wider_se_window": wide_window,
            "like_for_like_random_draw_join": like_for_like,
            "pristine_per_direction_median_e211": p_dir_median,
            "e_chart_retired_row": ECHART_RETIRED_BAND["median"],
        },
        "composite_order": "BANDS-MATCH / BANDS-GAP-REMAINS / GRADED (frozen "
                           "before compute; the first two mutually exclusive "
                           "by construction: the window's low edge "
                           f"{win_lo:.4f} sits above the gap threshold "
                           f"{gap_thresh:.4f})",
    }
    log("=" * 78)
    log(f"E212 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- provenance + honesty (the dispatch's required blocks) --------------
    metrics["provenance"] = {
        "parents_files_md5": parents_md5,
        "machinery": {
            "spans": "lab/e211_walled_band.py's pristine-span machinery "
                     "VERBATIM (the contiguous 20-step unwalled AdamW wash, "
                     "the fp64 Gram-SVD span, the seed-10902 stream = g1b's "
                     "own C-arm), reproduced against e211's committed "
                     "pristine spans at 1e-6 (the reproduction IS the gate)",
            "band": "lab/e209_census_debt.py's band instrument VERBATIM (the "
                    "Gaussian in-span draws, the onset grid 0.05..3.00, the "
                    "install-60 g-12 ruler, the 0.27 bar, the interpolated "
                    "first downcrossing, e205's middle-order-statistic median "
                    "with right-censoring), seeds 12601-12605 "
                    "(registry-clean, repo-grep'd)",
            "walled_rows": "runs/e211/metrics.json roots.R5/R6/R7.dir_kills "
                           "(the committed per-direction kill lists) — "
                           "loaded, medians recomputed exactly, never re-run",
            "committed_records": "e209 (the walled random-draw bands), "
                                 "e_chart (the retired pristine row + the "
                                 "root read), g1b (the C-arm rows 1..20) — "
                                 "loaded, hard-bound, never re-run",
        },
        "root_provenance": {
            "checkpoint": f"runs/checkpoints/{E131_ROOT_CK}",
            "recipe": meta.get("recipe"), "seed": meta.get("seed"),
            "base": meta.get("base"), "n_params": n_par,
            "identity": "the root of every walled row's lineage (g1b/g1bR's "
                        "committed base) and of e211's same-instrument cell; "
                        "used AS COMMITTED (no ball of its own, no settling)",
        },
        "instrument_identity": {
            "statement": "the pristine band is drawn on the SAME instrument "
                         "as the walled rows: the same span construction "
                         "(e209's contiguous 20-step unwalled AdamW wash + "
                         "fp64 Gram-SVD), the same fine grid (0.05..3.00), "
                         "the same ruler (install-60 g-12), the same kill "
                         "(0.27 interpolated first downcrossing), the same "
                         "median convention (e205's)",
            "verified_by": [
                "G_STREAM: the pristine stream is g1b's own C-arm stream "
                "(step-1 batch md5 == e185's stored hash)",
                "G_HIST: the history's per-step L2 reproduces g1b's "
                "committed C-arm rows 1..20 (max dev 9.2e-7 at e211; "
                "cross-device texture tier)",
                "G_SPAN: the span's PR and all 20 SVs reproduce e211's "
                "committed pristine span at 1e-6",
                "G_ROBUST: the robustness spans reproduce e211's committed "
                "robustness PRs at 1e-6",
                "the walled side's rows are e211's SAME-GRID SAME-RULER "
                "same-run kills (hard-bound, medians recomputed exactly)",
            ],
        },
        "draw_provenance": {f"s{d['seed']}": {"span": d["span"],
                                              "u_md5": d["u_md5"]}
                            for d in draws},
    }
    metrics["honesty"] = {
        "n_and_scope": ("n=5 draws (3+2) of ONE pristine root; each span is "
                        "a single 20-step history realization (3 "
                        "realizations: primary + 2 robustness); every kill "
                        "is one ray; the draw lottery (T155) moves the "
                        "median on redraw — the family's own realized swing "
                        f"({swing:.4f}) is the registered allowance, the "
                        "wider within-root-SE reading is context"),
        "single_root": ("the walled family is n=3 roots; the pristine side "
                        "is ONE root (the lineage's only pristine state) — "
                        "the join is 1 root vs 3, disclosed"),
        "direction_family_asymmetry": (
            "the registered join crosses direction families BY THE "
            "DISPATCH'S FROZEN LETTER: the pristine side is RANDOM draws "
            "(expected energy fraction_i ~ sv_i^2 — weighted toward the safe "
            "top-SV directions); the walled side is e211's PURE top-8 SV "
            "rays (including the fast-killing low-energy pure directions). "
            "Both like-for-like joins (random-vs-random vs e209's "
            "1.580/1.706/0.914; dirs-vs-dirs vs e211's pristine 1.031) are "
            "context, never adjudicated"),
        "desk_fixed_side": (
            "the match window [0.809, 0.957] and gap threshold 0.627 are "
            "desk-fixed by committed numbers (disclosed before compute); "
            "THE OPEN AXIS was the pristine random-draw median — never "
            "measured before this cell"),
        "settled_state_load": "the pristine root is used AS COMMITTED (no "
                              "anch__ ball of its own; nothing to settle)",
        "cross_instrument_retired_row": (
            "e_chart's 0.610 (the committed table's old pristine row) is a "
            "mixed-ladder FINE_GRID cap-1.5 co-read, 1-of-3 censored — "
            "retired by this cell's same-instrument band; carried in the "
            "table as context with its instrument disclosed"),
        "nothing_guaranteed": ("the openness was the point: the pristine "
                               "band could have landed below 0.63, inside "
                               "0.81-0.96, or above; the observed outcome "
                               f"is '{verdict}'"),
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    write_partial("P3 adjudicated (+ provenance + honesty)")

    # ================= P4: the figure =========================================
    plot_band(rd / "e212_pristine_band.png", metrics, table_rows,
              context_rows, walled_kills, draws, hist_rows, carm_rows,
              verdict, win_lo, win_hi, gap_thresh, fam)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'e212_pristine_band.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_band(path, metrics, table_rows, context_rows, walled_kills, draws,
              hist_rows, carm_rows, verdict, win_lo, win_hi, gap_thresh, fam):
    fig = plt.figure(figsize=(18.5, 11.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

    # (0,0) THE HONEST BAND TABLE — the dot plot
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(win_lo, win_hi, color="seagreen", alpha=0.10)
    ax.axhline(win_lo, ls="--", lw=1.4, color="seagreen")
    ax.axhline(win_hi, ls="--", lw=1.4, color="seagreen")
    ax.axhline(gap_thresh, ls="--", lw=1.8, color="crimson")
    ax.text(3.28, (win_lo + win_hi) / 2, "MATCH window\n0.85-0.92 ± swing",
            fontsize=8, color="seagreen", va="center", ha="left")
    ax.text(3.28, gap_thresh, "GAP threshold\n0.7x walled median",
            fontsize=8, color="crimson", va="center", ha="left")
    ys, labels = [], []
    y = 0
    for row in table_rows:
        kills = [k for k in row["kills"] if k is not None]
        nc = len(row["kills"]) - len(kills)
        ax.plot([0.30 + 0.02 * i - 0.05 for i in range(len(kills))],
                kills, "o", ms=4, alpha=0.45, color="gray")
        if row["median"] is not None:
            ax.plot([0.05, 0.95], [row["median"]] * 2, "-", lw=2.6,
                    color=("darkorange" if "THIS RUN" in row["row"]
                           else "steelblue"))
            ax.plot([0.5], [row["median"]], "*",
                    ms=17 if "THIS RUN" in row["row"] else 12,
                    color=("darkorange" if "THIS RUN" in row["row"]
                           else "steelblue"), mec="k", zorder=6)
        ax.text(1.02, row["median"] if row["median"] is not None else 0.4,
                f"{row['median']:.3f}" + (f" ({nc} cens)" if nc else ""),
                fontsize=8.5, va="center",
                color=("darkorange" if "THIS RUN" in row["row"] else "black"),
                weight="bold" if "THIS RUN" in row["row"] else "normal")
        ys.append(0)
        labels.append(row["row"])
        y += 1
    # context rows (right half)
    cx = 2.3
    for crow in context_rows:
        if crow["row"].startswith("R5/R6/R7"):
            for r_, m_ in crow["medians"].items():
                ax.plot([cx], [m_], "s", ms=7, mfc="none", mec="purple",
                        alpha=0.8)
                ax.annotate(f"{r_}-rand {m_:.2f}", (cx, m_),
                            textcoords="offset points", xytext=(8, -3),
                            fontsize=7, color="purple")
        else:
            mk = "x" if "RETIRED" in crow["row"] else "d"
            col = "gray" if "RETIRED" in crow["row"] else "purple"
            if crow["median"] is not None:
                ax.plot([cx], [crow["median"]], mk, ms=9, color=col, alpha=0.9)
                ax.annotate(f"{crow['row'].split(' (')[0]} {crow['median']:.2f}"
                            + (" (retired, cross-instr.)"
                               if "RETIRED" in crow["row"] else " (context)"),
                            (cx, crow["median"]), textcoords="offset points",
                            xytext=(8, -3), fontsize=7, color=col)
    ax.set_ylim(0, 3.05)
    ax.set_xlim(0, 4.4)
    ax.set_xticks([0.5, cx])
    ax.set_xticklabels(["ADJUDICATED rows\n(median = thick bar, * = median, "
                        "o = individual kills)",
                        "CONTEXT rows\n(never adjudicated)"], fontsize=8)
    ax.set_ylabel("band median D_kill (onset grid, install-60 g-12 ruler)")
    ax.set_title("THE HONEST BAND TABLE — the same-instrument pristine band "
                 "(n=5) vs e211's walled per-direction medians", fontsize=10)

    # (0,1) the five pristine draw curves
    ax2 = fig.add_subplot(gs[0, 1])
    for i, d in enumerate(draws):
        rr = d["rows"]
        ax2.plot([r["D"] for r in rr], [r["gm"] for r in rr], "-",
                 lw=1.5, alpha=0.85,
                 label=f"s{d['seed']} ({d['span'].split(':')[0]}): "
                       f"{d['D_kill'] if d['D_kill'] is not None else 'SOFT'}")
        if d["D_kill"] is not None:
            ax2.axvline(d["D_kill"], ls=":", lw=1.0, color="gray", alpha=0.5)
    med = metrics["roots"]["P"]["band"]["median_stat"]["median"]
    if med is not None:
        ax2.axvline(med, ls="-", lw=2.2, color="darkorange",
                    label=f"BAND MEDIAN {med:.3f}")
    ax2.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
                label="0.27 DISSOLVE")
    ax2.axhline(metrics["roots"]["P"]["G_ROOT"]["read_measured"], ls="--",
                lw=1.0, color="gray", alpha=0.8, label="root read")
    ax2.set_ylim(-0.03, 1.02)
    ax2.set_xlabel("D along the unit draw (L2; grid 0.05..3.00, early stop)")
    ax2.set_ylabel("g-12 ruler (install-60 mean p(Z))")
    ax2.legend(fontsize=7.2, loc="lower left")
    ax2.set_title("THE 3+2 PRISTINE DRAWS (e209's instrument verbatim)",
                  fontsize=10)

    # (1,0) instrument identity: per-step history L2 vs g1b's committed rows
    ax3 = fig.add_subplot(gs[1, 0])
    steps = [h["step"] for h in hist_rows]
    ax3.plot(steps, [h["L2"] for h in hist_rows], "o-", ms=4, lw=1.4,
             color="darkorange", label="this run (CPU, threads 4)")
    ax3.plot(steps, [carm_rows[s]["step_disp"] for s in steps], "x--", ms=5,
             lw=1.0, color="steelblue",
             label="g1b committed C-arm rows (CUDA)")
    ax3.set_xlabel("wash step")
    ax3.set_ylabel("per-step displacement L2")
    ax3.legend(fontsize=8)
    sp = metrics["roots"]["P"]["span_primary"]
    ax3.set_title(f"INSTRUMENT IDENTITY — the pristine history IS g1b's "
                  f"C-arm (max |d| {max(abs(h['L2'] - carm_rows[s]['step_disp']) for h, s in zip(hist_rows, steps)):.1e}); "
                  f"span PR {sp['participation_ratio']:.4f} reproduces "
                  f"e211's 4.117364 (rel "
                  f"{sp['pr_vs_e211_rel_dev']:.1e})", fontsize=9)

    # (1,1) the span spectrum + the two-family context view
    ax4 = fig.add_subplot(gs[1, 1])
    prof = sp["profile"]
    ax4.semilogy(range(1, len(prof) + 1), prof, "o-", ms=4, lw=1.4,
                 color="darkorange", label="pristine primary span (this run)")
    for q in metrics["roots"]["P"]["span_robustness"]:
        pr_r = [s / q["sv"][0] for s in q["sv"]]
        ax4.semilogy(range(1, len(pr_r) + 1), pr_r, "x--", ms=4, lw=1.0,
                     alpha=0.7, label=f"robust s{q['seed']} (PR "
                                      f"{q['participation_ratio']:.3f})")
    e211_walled_prof = None
    ax4.set_xlabel("SV index (of 20)")
    ax4.set_ylabel("sv_i / sv_1 (normalized profile)")
    ax4.legend(fontsize=7.5)
    ax4.set_title(f"the pristine span spectrum (PR {sp['participation_ratio']:.4f}; "
                  f"e211's walled PRs 4.216/4.263/4.260 — matched, T182)",
                  fontsize=9)

    fig.suptitle(f"E212 — THE SAME-INSTRUMENT PRISTINE BAND (e211's named "
                 f"debt paid): the honest band table    VERDICT: {verdict}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
