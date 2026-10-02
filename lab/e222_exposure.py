"""E222 — THE EXPOSURE-IMMUNITY CAUSAL TEST (T182's named follow-up).

WHY: e211 (runs/e211/metrics.json, T182) found per-direction safety ORDERS
with SV energy at every state alike — Spearman(kill-D vs SV energy) +0.381
/ +0.452 / +0.571 / +0.810 per root, pooled +0.639, while curvature
ANTI-orders (-0.595..-0.786) — "SAFETY LIVES IN THE WASH'S OWN TOP
DIRECTIONS: the directions the wash itself puts energy into are the
directions further displacement forgives — an EXPOSURE-IMMUNITY reading:
the wash vaccinates its own span." T182 left that reading untested: it is
a CORRELATION across directions within one span. e222 makes it CAUSAL: a
controlled sub-lethal displacement (the pre-exposure) along the span's
top-SV direction, then the tolerance re-measured — does prior exposure
along a direction RAISE tolerance (immunity), LOWER it (sensitization —
the ordering was a correlation, not protection), or move nothing at
intervenable doses (null)?

WHAT IT BUILDS ON: e211's span/band machinery VERBATIM (the contiguous
20-step unwalled AdamW wash history; the fp64 Gram-SVD span; the onset
grid 0.05..3.00 per-direction kill instrument, first g<=0.27 downcrossing
interpolated, early stop; the SVD-emitted direction signs); e193's
family-2 organism gates VERBATIM (the f2 root load + dial hard-bind, the
e185-hashed seed-10902 wash stream, the e157 committed anchors, the
e170 neutral bank, the RULER DISCLOSURE: PRIMARY ruler = the install-60
battery at ctx offset -4 — the max committed root read 0.8872; co-rulers
g+12/g-12/g0 ride in every table, never adjudicated); e157's committed
dial + wash step-1 rows (loaded, never rerun); e193's committed G_T0 rows
(the measured step-1 L2 0.91642, the post-step primary read 0.00682);
e193's committed g-ray direction checkpoint (the cos context join).
NOTHING from the parents is re-adjudicated; every committed number loads
and hard-binds at its metrics path.

WHAT IS NEW: the exposure cell itself. (1) THE DOSE: eps = 0.1 x
D_kill^baseline(top-SV dir) — sub-lethal by construction, asserted AFTER
the un-exposed baselines are measured (the dispatch's own check). (2) THE
ARMS: expose theta_pre = theta0 + eps*v along the TOP-SV direction
(dir 0), a LOW-SV in-span direction (dir 19, the same L2 — the
ordering's gradient probe), and a RANDOM in-span direction (fresh seed
12501, same L2 — the matched control); no training — a direct weight
displacement, the wash's mechanism at one remove. (3) THE TESTS: the
kill-D of dir 0 (the exposed one), dir 9 (a mid-SV cross direction —
does immunity generalize within the span?) and dir 19 re-walked FROM
EACH exposed state, plus the control's own direction. (4) THE
DECOMPOSITION: every same-axis row decomposed into the mechanical
pass-back component (see the desk disclosure) and the causal residue.

DESK PRE-READ DISCLOSURE (arithmetic stated BEFORE compute, e205/e211's
convention; honesty, not a prediction — no bar is moved): the kill walk
is theta_state - D*v (e211's convention, the SVD-emitted sign) and the
exposure is +eps*v — OPPOSED to the walk, the pass-back-through
geometry. The post-exposure walk from theta0+eps*v along -v revisits,
for D>=eps, EXACTLY the baseline positions (theta0-(D-eps)*v); reading
is a deterministic function of position, so the same-direction kill-D is
ARITHMETICALLY CAPPED at eps + D_kill^baseline — i.e. ratio <=
1 + eps/t = 1.10 at the registered 0.1x dose, up to grid-interpolation
quantization (+/- the 0.05 step). The IMMUNITY bar's ">= 1.15x" clause
on THAT row is therefore arithmetically unreachable at this dose unless
the grid interpolates pathologically; a reading BELOW 1.10 (in
particular < 1.0, the SENSITIZATION clause) is genuine terrain
structure — the +v exposure segment dips under the kill bar. The
non-mechanical causal content lives in (i) the same-axis RESIDUES
(kill-D_post - eps - t_base; 0 = pure position arithmetic) and (ii) the
CROSS rows (dir 9 / dir 0 tolerance after orthogonal exposures — fresh
lines through space that revisit nothing). The registered bars below
adjudicate the raw numbers exactly as written; this disclosure frames
the reading. No bar shopping.

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - IMMUNITY-CAUSAL: "fires if pre-exposure along the top-SV direction
    raises the kill-D of THAT direction (>= 1.15x the un-exposed
    baseline) with the random-direction control showing less — the
    exposure-immunity ordering is CAUSAL: the wash vaccinates its own
    span."
  - SENSITIZATION: "fires if pre-exposure LOWERS the kill-D
    (sensitization — the ordering is a correlation, not protection; the
    SV-energy ordering's mechanism is elsewhere)."
  - NULL: "no material change either way — the ordering at these doses
    is not intervenable; reported honestly."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * organism = the e193 family-2 root, runs/checkpoints/
    e157_f2_consolidated.pt (873,472 params), used AS COMMITTED (flat
    md5-gated); ruler = e193's PRIMARY (install-60 g-4, committed root
    read 0.8872273564338684); kill bar 0.27; D grid 0.05..3.00 step
    0.05; kill = first downcrossing, linear-in-D interpolated, early
    stop; unresolved past 3.00 = censored (pairwise-dropped, disclosed).
  * span = the root's OWN contiguous 20-step unwalled AdamW wash (lr
    1e-3, betas (0.9,0.95), wd 0.1, clip 1.0, batch 32 = 16 e170-bank
    anchors + 16 random windows, the seed-10902 stream, e209/e211
    verbatim); basis = the fp64 Gram-SVD right-singular vectors Vp;
    v_i = Vp[i] unit-normalized, the SVD-emitted sign everywhere.
  * directions: TOP = dir 0, MID = dir 9, LOW = dir 19 (of 20).
  * dose eps = 0.1 x D_kill^baseline(dir 0) — registered; all three
    arms use the SAME eps (matched L2, the control's requirement).
  * arms: TOP-EXP theta0+eps*v0; LOW-EXP theta0+eps*v19; CTRL-EXP
    theta0+eps*u_ctrl with u_ctrl = a fresh random unit in-span vector
    (seed 12501; randn(20) coefficients in the Vp basis, fp32,
    normalized; drawn AFTER eps is fixed; accepted only if the exposed
    state reads > 0.27 on the primary ruler — up to 5 draws, every draw
    disclosed; the acceptance selection biases the control TOWARD
    forgiveness, i.e. AGAINST the IMMUNITY clause's "control shows
    less" — conservative, disclosed).
  * sub-lethality gates: every exposed state reads > 0.27 on the
    primary ruler at D=0 (asserted; a failure at the TOP arm aborts the
    cell with metrics written first — the e187 lesson).
  * rows per exposed state: kill-D of dir 0, dir 9, dir 19 (+ the
    control's own u ray at the CTRL state). Same-axis rows (TOP arm's
    dir-0 row; LOW arm's dir-19 row; CTRL's own row) carry the
    mechanical ceiling 1 + eps/t_base and the residue kill-D_post -
    eps - t_base.
  * CE_R canary (the organism-health honesty reflex): val-window CE
    (60 windows, seed 26502) at every state D=0 and at the dir-0 kill
    positions; reported, never adjudicated.
  * composite order IMMUNITY-CAUSAL / SENSITIZATION / NULL / GRADED;
    the first two mutually exclusive by construction (>= 1.15 vs <
    1.0). NULL = neither fired AND every resolved ratio in [0.85,
    1.15] AND |residue(dir 0 row)| <= 0.05 (one grid step) — "no
    material change either way". GRADED = any partial. If the control's
    own row is censored, the IMMUNITY control clause is unresolvable
    and IMMUNITY-CAUSAL cannot fire (reported, never guessed).

PRE-DISPATCH CHECKS (Rule 12): the root's provenance gates (e193's set:
G_NAMEFREE / G_SPLICE / G_BATTERY / G_ANCHOR / G_ROOT dial hard-bind /
G_T0 / G_S1CK / G_STREAM); G_PARENTS (e193 + e157 + e211 metrics md5s,
the committed numbers hard-bound: the dial cells, the wash step-1 CE and
reads, e193's measured step-1 L2 + post-step primary read, e211's
per-root + pooled ordering Spearmans — the numbers this cell
interrogates). Threads 4 (the owner envelope; e157's committed reads
were threads-8 — gates at the 5e-3 cross-thread texture tier, bit status
recorded). WHAT THE GATES GUARANTEE: NOTHING — the residues could be
deeply negative, the cross rows could invert, everything could sit at
the mechanical arithmetic. The openness is the point.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch; the GPU is never claimed), torch threads 4, load-check
recorded not gating (e204/e205/e209/e211's convention), small eval
bursts, PROGRESSIVE metrics.json writes, decisive.

Outputs: runs/e222/{metrics.json (PROGRESSIVE), e222_exposure_immunity.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e222_exposure.py    (E222_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e185/e193/e211)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)          # the owner envelope (e157's committed reads were threads-8)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E222_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e222 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- the organism (e193's family-2 root, verbatim constants) --------------------
NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e157_f2_consolidated.pt"
S1_CK = "e157_f2_neutral_s1.pt"
GDIR_CK = "e193_f2_static_dir_u.pt"      # e193's committed g-ray (the cos join)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E157_METRICS = E43.REPO / "runs" / "e157" / "metrics.json"
E193_METRICS = E43.REPO / "runs" / "e193" / "metrics.json"
E211_METRICS = E43.REPO / "runs" / "e211" / "metrics.json"

F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472
ROOT_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"

RULER_J = -4                      # PRIMARY (e193's frozen RULER DISCLOSURE)
CO_RULERS_J = (12, -12, 0)
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)

# ---- e157's committed lineage reads (e193's copies, hard-bound at load) --------
E157_DIAL = {
    "gm12": 0.19826222956180573, "g-8": 0.8519253134727478,
    "g-4": 0.8872273564338684, "g0": 0.5784125924110413,
    "g+4": 0.8801683187484741, "g+8": 0.8754011988639832,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
E157_WASH_S1 = {"corpus_ce": 1.7748003005981445,
                "gm12": 0.001129803480580449}
E193_T1_PRIMARY = 0.0068241604603827   # e193's committed post-step primary read
E193_STEP_L2 = 0.9164195656776428      # e193's committed measured step-1 L2
E193_GKILL = 0.2                       # e193's committed static g-ray kill (context)

# ---- e211's committed ordering (THE CORRELATION this cell interrogates) -------
E211_ORDERING = {
    "verdict": "GRADED",
    "spearman_kill_vs_sv": {"P": 0.8095238095238095, "R5": 0.4523809523809524,
                            "R6": 0.5714285714285714, "R7": 0.38095238095238093},
    "spearman_kill_vs_sv_pooled": 0.6391129032258065,
    "spearman_kill_vs_curv_range": [-0.7857142857142857, -0.5952380952380952],
}

E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {1: "1ea27bffde6c4a53be8badf5ab453d64",
              2: "1d6f0e55cc6a25ece947d2040528225e",
              3: "b5c0b670270406a94aca63071b051468",
              4: "cdccea0c413e603dc52d1873e37b9844"}

# ---- the run envelope (dispatch-frozen) ----------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e157/e193)
LR_ADAMW = 1e-3
ANCH_BS, RAND_BS = 16, 16
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # e209/e211's onset grid
WASH_HIST_STEPS = 4 if SMOKE else 20
DIR_TOP, DIR_MID, DIR_LOW = 0, 9, 19
DOSE_FRAC = 0.10                  # eps = 0.1 x baseline kill-D of dir 0 (registered)
CTRL_SEED = 12501                 # fresh (registry: ...12401-8 e211 robustness; 12501 is fresh)
CTRL_MAX_DRAWS = 5
IMMUNITY_RATIO = 1.15             # the registered bar
SENS_BELOW = 1.0                  # "LOWERS the kill-D"
NULL_LO, NULL_HI = 0.85, 1.15     # "no material change either way"
RESIDUE_QTOL = 0.05               # one grid step (the interpolation quantization)
G_BIT_TOL = 5e-6
G_READ_TOL = 5e-3                 # cross-thread texture tier (e211's G_READ_TOL)
VAL_SEED = 26502                  # e193's CE_R val-window seed

if SMOKE:                         # shakedown trims (disclosed; nothing adjudicated)
    D_GRID = [0.05, 0.5, 1.5, 3.0]
    DIR_MID, DIR_LOW = 1, 3       # the 4-dir span re-indexed (top stays 0)

REGISTERED_BARS = {
    "IMMUNITY_CAUSAL": "IMMUNITY-CAUSAL: \"fires if pre-exposure along the "
        "top-SV direction raises the kill-D of THAT direction (>= 1.15x the "
        "un-exposed baseline) with the random-direction control showing less "
        "— the exposure-immunity ordering is CAUSAL: the wash vaccinates its "
        "own span.\"",
    "SENSITIZATION": "SENSITIZATION: \"fires if pre-exposure LOWERS the "
        "kill-D (sensitization — the ordering is a correlation, not "
        "protection; the SV-energy ordering's mechanism is elsewhere).\"",
    "NULL": "NULL: \"no material change either way — the ordering at these "
        "doses is not intervenable; reported honestly.\"",
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": (
        "organism = the e193 f2 root as committed (md5-gated); ruler = "
        "install-60 g-4 (e193's frozen RULER DISCLOSURE; committed read "
        "0.8872273564338684); kill bar 0.27; grid 0.05..3.00; kill = first "
        "downcrossing interpolated (e209/e211 verbatim); span = the root's "
        "own contiguous 20-step unwalled AdamW wash (seed-10902 stream), "
        "fp64 Gram-SVD basis, SVD-emitted signs; TOP=dir0 MID=dir9 LOW=dir19; "
        "eps = 0.1 x baseline kill-D(dir0), the SAME eps for all arms; arms "
        "TOP-EXP/LOW-EXP/CTRL-EXP = theta0 + eps*v (pass-back-through "
        "geometry, the desk disclosure's arithmetic); the control = fresh "
        "random in-span unit (seed 12501, randn(20) coeffs in Vp), accepted "
        "only if the exposed state reads alive (<= 5 draws, disclosed; the "
        "selection biases AGAINST the IMMUNITY clause — conservative); "
        "sub-lethality asserted at every exposed state; same-axis rows carry "
        "the mechanical ceiling 1 + eps/t_base and the residue kill-D_post - "
        "eps - t_base; CE_R canary at every state + the dir-0 kill positions "
        "(report-only); composite IMMUNITY-CAUSAL / SENSITIZATION / NULL / "
        "GRADED, the first two mutually exclusive; NULL = neither fired AND "
        "all resolved ratios in [0.85,1.15] AND |residue(dir0 row)| <= 0.05; "
        "a censored control-own row makes the IMMUNITY control clause "
        "unresolvable (cannot fire, reported)."),
    "desk_pre_read_disclosure": (
        "ARITHMETIC, stated before compute (no bar moved): the exposure is "
        "+eps*v, the kill walk is theta-D*v — the same-direction post-"
        "exposure walk REVISITS the baseline positions for D>=eps, so its "
        "kill-D is arithmetically capped at eps + t_base (ratio <= 1 + "
        "eps/t_base = 1.10 at the 0.1x dose, up to +/- one 0.05 grid step "
        "of interpolation). The IMMUNITY row's >= 1.15 clause is therefore "
        "arithmetically unreachable at this dose; a reading below 1.10 "
        "(esp. < 1.0 = SENSITIZATION) is genuine terrain structure (the "
        "exposure segment dips under the bar). The causal content lives in "
        "the same-axis residues (=0 means pure position arithmetic) and the "
        "CROSS rows (orthogonal exposures, fresh lines). The bars below "
        "adjudicate the raw numbers exactly as registered."),
    "registration": "the dispatch's registration IS the registration (the "
        "four bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE ORGANISM PORT: e211's span/band machinery (2.74M e131 family) is "
    "run VERBATIM on the e193 family-2 root (873k, 4L/4H/128d/512-ctx) per "
    "the dispatch's cell; the ruler is e193's PRIMARY (install-60 g-4), NOT "
    "e211's install-60 g-12 — the f2 g-12 geometry reads 0.198 at the root "
    "(under the 0.27 bar; e157 stage-A's G_CONS failure, e193's frozen "
    "RULER DISCLOSURE); the g-12 co-ruler rides in every table.",
    "THREADS 4 (the owner envelope), not e157's threads-8 readout "
    "convention: the dial gates run at the 5e-3 cross-thread texture tier "
    "with the bit status (5e-6) recorded; the root flat md5 gates exactly "
    "(thread-independent).",
    "THE EXPOSURE SIGN IS THE SVD-EMITTED +v, OPPOSED to the kill walk's "
    "-v (the pass-back-through geometry) — the only sign choice under which "
    "the registered IMMUNITY clause is not mechanically dead in the "
    "sensitization direction (0.90 tautology); the desk disclosure states "
    "the resulting 1.10 arithmetic ceiling on every same-axis row BEFORE "
    "compute.",
    "THE CONTROL ACCEPTANCE RULE: the random in-span control is drawn AFTER "
    "eps is fixed and accepted only if the exposed state reads alive (> "
    "0.27) — up to 5 draws from seed 12501, every draw's read disclosed; "
    "the selection biases the control toward forgiveness, i.e. AGAINST the "
    "IMMUNITY clause's 'control shows less' (conservative for the bar, "
    "disclosed).",
    "THE LOW ARM IS INCLUDED (the dispatch's optional second arm): the "
    "same-L2 exposure along dir 19 — the ordering's gradient probe (does "
    "exposure along the LEAST-forgiving in-span direction immunize less "
    "than the top direction?).",
    "e193's committed g-ray direction checkpoint is joined by fp64 cosine "
    "to every span direction (context only: how gradient-aligned the span "
    "axes are; e193's g-kill 0.2 vs the span rays' kill-Ds).",
    "No committed 20-step span PR exists for the f2 root (e209's committed "
    "PRs are the walled e131-family roots) — the span here is fresh compute "
    "gated only by the e193 stream/t0 anchors; nothing is reproduced "
    "against e209.",
    "The load-check is recorded, not gating (e204/e205/e209/e211's "
    "convention).",
    "Smoke mode trims: D grid {0.05,0.5,1.5,3.0}, 4-step history (basis "
    "dirs re-indexed to 0/1/2 of 3); nothing adjudicated (verdict stamped "
    "SMOKE).",
]

# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e193_organism_replicate.py (the family-2 gates, battery,
# stream, CE_R) + lab/e211_walled_band.py (the wash history, Gram-SVD span,
# onset-grid kill walk) — both via lab/e185_noise_wash.py (the e176n
# lineage). Copied rather than imported to own the device policy and the
# arithmetic.


def load_f2(path) -> tuple[TinyGPT, dict]:
    m = TinyGPT(F2_CFG)
    st = torch.load(path, map_location="cpu", weights_only=False)
    meta = st.get("meta") if isinstance(st, dict) else None
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m, ({"meta": meta} if isinstance(st, dict) else {})


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e193 battery on CPU: p(Z) at the last position."""
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
        lg, _ = net(x[i:i + bs], y[i:i + bs])
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[i:i + bs].reshape(-1))
        tot += float(ce.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065/e193 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y, tries = [], [], 0
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
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64) / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def interp_d_kill(v0, v1, d0, d1):
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


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
    rd = run_dir("e222_smoke" if SMOKE else "e222")
    metrics: dict = {
        "experiment": "e222_exposure_immunity",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never claimed)",
            "torch_threads": torch.get_num_threads(),
            "load_check_recorded_not_gating": True,
            "phases": "P0 gates -> P1 span -> P2 baselines + dose -> P3 the "
                      "exposure arms -> P4 adjudication + figure; progressive "
                      "writes",
            "organism": f"runs/checkpoints/{ROOT_CK} (873,472 params, the e193 f2 root)",
            "ruler": f"install-60 g{RULER_J:+d} (e193's PRIMARY; bar {SHUT_BAR})",
            "dirs": {"top": DIR_TOP, "mid": DIR_MID, "low": DIR_LOW},
            "dose_frac": DOSE_FRAC,
            "ctrl_seed": CTRL_SEED,
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
    log(f"E222 THE EXPOSURE-IMMUNITY CAUSAL TEST (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), load-check "
        f"recorded (launch: {load0}%), progressive writes, decisive")

    # ================= P0a: protocol rebuild (e193's gate set) =================
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
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at the "
                "e157 dial's seven read geometries (e193 verbatim); PRIMARY "
                f"ruler = g{RULER_J:+d}",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]
    coruler_ids = {j: bat_ids[j] for j in CO_RULERS_J}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, VAL_SEED)

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
    log("P0a: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank bit-match)")
    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}
    write_partial("P0a protocol gates PASSED")

    # ================= P0b: the parents, hard-bound (Rule 12) ==================
    for p in (E157_METRICS, E193_METRICS, E211_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")

    def md5of(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    e157m = json.loads(E157_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    e211m = json.loads(E211_METRICS.read_text(encoding="utf-8"))
    parents_md5 = {"e157": md5of(E157_METRICS), "e193": md5of(E193_METRICS),
                   "e211": md5of(E211_METRICS)}

    # hard-bind the committed references (asserts catch committed-file drift)
    dial = e157m["stages"]["A_consolidate"]["dial"]
    jit = e157m["stages"]["A_consolidate"]["consolidated_jitter_geos"]
    assert abs(dial["base"]["-12"]["mean_pz"] - E157_DIAL["gm12"]) < 1e-12
    assert abs(dial["base"]["0"]["mean_pz"] - E157_DIAL["g0"]) < 1e-12
    assert abs(dial["base"]["12"]["mean_pz"] - E157_DIAL["gp12"]) < 1e-12
    assert abs(dial["ce_r"] - E157_DIAL["ce_r"]) < 1e-12
    for j, k in ((-8, "g-8"), (-4, "g-4"), (4, "g+4"), (8, "g+8")):
        assert abs(jit[f"g{j:+d}"] - E157_DIAL[k]) < 1e-12
    w1 = e157m["stages"]["B_wash"]["traj"][0]
    assert w1["step"] == 1
    assert abs(w1["corpus_ce"] - E157_WASH_S1["corpus_ce"]) < 1e-12
    assert abs(w1["g_m12_mean_pz"] - E157_WASH_S1["gm12"]) < 1e-12
    e193_t0 = e193m["gates"]["G_T0"]
    assert abs(e193_t0["poststep_primary_read"] - E193_T1_PRIMARY) < 1e-12
    assert abs(e193_t0["adamw_step1_L2_measured"] - E193_STEP_L2) < 1e-12
    assert abs(e193m["adjudication"]["bars"]["TERRAIN_REPLICATES"]["g_kill"]
               - E193_GKILL) < 1e-12
    e211_corr = e211m["band_re_read"]["correlations_context"]
    for rid, v in E211_ORDERING["spearman_kill_vs_sv"].items():
        assert abs(e211_corr[rid]["spearman_kill_vs_sv"] - v) < 1e-12
    assert abs(e211_corr["pooled"]["spearman_kill_vs_sv"]
               - E211_ORDERING["spearman_kill_vs_sv_pooled"]) < 1e-12
    assert e211m["adjudication"]["verdict"] == E211_ORDERING["verdict"]
    G_PARENTS = {
        "files_md5": parents_md5,
        "hardbound": {
            "e157.dial_g-4": E157_DIAL["g-4"],
            "e157.dial_ce_r": E157_DIAL["ce_r"],
            "e157.wash_s1.corpus_ce": E157_WASH_S1["corpus_ce"],
            "e157.wash_s1.gm12": E157_WASH_S1["gm12"],
            "e193.G_T0.poststep_primary_read": E193_T1_PRIMARY,
            "e193.G_T0.adamw_step1_L2": E193_STEP_L2,
            "e193.adjudication.g_kill": E193_GKILL,
            "e211.verdict": E211_ORDERING["verdict"],
            "e211.spearman_kill_vs_sv_per_root":
                E211_ORDERING["spearman_kill_vs_sv"],
            "e211.spearman_kill_vs_sv_pooled":
                E211_ORDERING["spearman_kill_vs_sv_pooled"],
        },
        "note": "e211's per-direction ordering (kill-D vs SV energy, "
                "+0.381..+0.810 per root, pooled +0.639, vs curvature "
                "-0.595..-0.786) is THE correlation this cell interrogates",
        "pass": True,
    }
    log("P0b: parents hard-bound — e157 (dial + wash s1), e193 (t0 rows + "
        "g-kill 0.2), e211 (the ordering: per-root +0.381..+0.810, pooled "
        "+0.639, verdict GRADED)")
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    write_partial("P0b parent gates PASSED (e157/e193/e211 hard-bound)")

    # ================= P0c: the root + the dial ================================
    net0, root_extra = load_f2(CKPT_DIR / ROOT_CK)
    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    assert N_PARAM == F2_PARAMS, f"params {N_PARAM} != {F2_PARAMS}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()
    evl = copy.deepcopy(net0)
    root_cells = {f"g{j:+d}": battery_cell(evl, bat_ids[j], zid)["mean_pz"]
                  for j in READ_GEOS}
    root_cells["ce_r"] = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
    keymap = {"gm12": "g-12", "g0": "g+0", "gp12": "g+12", "g-4": "g-4",
              "g+4": "g+4", "g-8": "g-8", "g+8": "g+8", "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E157_DIAL.items()}
    rdiffs = {k: root_cells[k] - root_refs[k] for k in root_refs}
    rmax = max(abs(v) for v in rdiffs.values())
    G_ROOT = {"cells": root_cells, "refs": root_refs, "diffs": rdiffs,
              "max_abs_diff": rmax, "bit_tol": G_BIT_TOL, "tol": G_READ_TOL,
              "bit": bool(rmax < G_BIT_TOL),
              "flat_md5": root_md5,
              "flat_md5_match_committed": bool(root_md5 == ROOT_FLAT_MD5),
              "root_meta": root_extra.get("meta"),
              "threads_note": "threads 4 (the owner envelope) vs e157's "
                              "threads-8 committed reads — the 5e-3 texture "
                              "tier gates; the flat md5 gates exactly",
              "pass": bool(rmax < G_READ_TOL
                           and root_md5 == ROOT_FLAT_MD5)}
    log(f"P0c G_ROOT (vs e157 committed dial, 8 cells, threads 4): max|diff| "
        f"{rmax:.2e}, md5 "
        f"{'OK' if G_ROOT['flat_md5_match_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("family-2 root gate FAILED")
    metrics["gates"]["G_ROOT"] = G_ROOT
    del evl
    write_partial("P0c G_ROOT PASSED (the f2 root as committed)")

    # ================= P0d: G_T0 / G_S1CK / G_STREAM (e193 verbatim) ===========
    def draw_step1_batch():
        g = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(16, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

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
    torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0)
    optw.step()
    theta1_adam = flat_params(tw)
    disp1 = float(torch.norm(theta1_adam - theta0))
    del tw, optw, logits
    evl_t1 = copy.deepcopy(net0)
    load_flat(evl_t1, theta1_adam)
    t1_primary = battery_cell(evl_t1, primary_ids, zid)["mean_pz"]
    del evl_t1

    s1_net, s1_extra = load_f2(CKPT_DIR / S1_CK)
    theta_s1 = flat_params(s1_net)
    d_fresh, d_ck = theta1_adam - theta0, theta_s1 - theta0
    cos_s1 = cos64(d_fresh, d_ck)
    rel_l2 = abs(float(torch.norm(d_ck)) - disp1) / disp1
    del s1_net

    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 5):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        stream_ok[s_] = bool(
            hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
            == E185_XHASH[s_])
    G_T0 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1,
        "ce_batch_committed_e157": E157_WASH_S1["corpus_ce"],
        "d_ce": abs(ce1 - E157_WASH_S1["corpus_ce"]),
        "adamw_step1_L2_measured": disp1,
        "adamw_step1_L2_committed_e193": E193_STEP_L2,
        "poststep_primary_measured": t1_primary,
        "poststep_primary_committed_e193": E193_T1_PRIMARY,
        "d_poststep_primary": abs(t1_primary - E193_T1_PRIMARY),
        "G_S1CK": {"cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
                   "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
                   "s1_meta": s1_extra.get("meta")},
        "G_STREAM": {"steps_1_4_md5_match": stream_ok,
                     "pass": bool(all(stream_ok.values()))},
        "note": "e193's t=0 gate set verbatim at threads 4: the stream "
                "construction is net-independent (md5s must bit-match); the "
                "CE + post-step reads gate at the 5e-3 cross-thread texture "
                "tier; the s1 checkpoint anchors the fresh step (cos > 0.999)",
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and abs(ce1 - E157_WASH_S1["corpus_ce"]) < G_READ_TOL
                     and abs(disp1 - E193_STEP_L2) < G_READ_TOL
                     and abs(t1_primary - E193_T1_PRIMARY) < G_READ_TOL
                     and cos_s1 > 0.999 and rel_l2 < 0.05
                     and all(stream_ok.values())),
    }
    log(f"P0d G_T0: x_md5 {'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; "
        f"CE {ce1:.7f} (|d| {G_T0['d_ce']:.1e}); step L2 {disp1:.7f} "
        f"(|d| {abs(disp1 - E193_STEP_L2):.1e}); post-step primary "
        f"{t1_primary:.7f} (|d| {G_T0['d_poststep_primary']:.1e}); cos_s1 "
        f"{cos_s1:.6f}; stream 1..4 "
        f"{'OK' if G_T0['G_STREAM']['pass'] else 'DRIFT'}: "
        + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 gate FAILED — abort (control failure)")
    metrics["gates"]["G_T0"] = G_T0
    del theta1_adam, d_fresh, d_ck, theta_s1
    write_partial("P0d G_T0/G_S1CK/G_STREAM PASSED (e193's anchors)")
    metrics["envelope"]["cpu_load_pct_after_P0"] = cpu_load_probe()

    # ================= P1: the wash history + the span =========================
    log("=" * 78)
    log(f"P1: the f2 root's own contiguous {WASH_HIST_STEPS}-step unwalled "
        f"AdamW wash (seed {FREEZE_SEED}) -> the span")
    wgen = torch.Generator().manual_seed(FREEZE_SEED)
    wnet = copy.deepcopy(net0)
    wnet.train()
    wopt = torch.optim.AdamW(wnet.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                             weight_decay=0.1)
    wtheta = theta0.clone()
    segs, hist_rows = [], []
    evl_w = copy.deepcopy(net0)
    for s_wh in range(1, WASH_HIST_STEPS + 1):
        aj_ = torch.randint(16, (ANCH_BS,), generator=wgen)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=wgen)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        y_ = torch.cat([anc_[:, 1:], rnd_[:, 1:]], 0)
        logits_w, _ = wnet(x_)
        lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                             y_.reshape(-1))
        wopt.zero_grad(set_to_none=True)
        lw.backward()
        torch.nn.utils.clip_grad_norm_(wnet.parameters(), 1.0)
        wopt.step()
        th_new = flat_params(wnet)
        segs.append(th_new - wtheta)
        load_flat(evl_w, th_new)
        gz = battery_cell(evl_w, primary_ids, zid)
        gp = battery_cell(evl_w, coruler_ids[12], zid)
        hist_rows.append({"step": s_wh,
                          "L2": float(torch.norm(th_new - wtheta)),
                          "cum": float(torch.norm(th_new - theta0)),
                          "ce": float(lw.item()),
                          "g_m4_mean_pz": gz["mean_pz"],
                          "g_p12_mean_pz": gp["mean_pz"]})
        wtheta = th_new
    wnet.eval()
    del wnet, wopt, evl_w
    H = torch.stack(segs)
    hist_dead = next((h["step"] for h in hist_rows
                      if h["g_m4_mean_pz"] <= SHUT_BAR), None)
    hist_check = {
        "n_steps": WASH_HIST_STEPS, "rows": hist_rows,
        "step1_L2": hist_rows[0]["L2"], "cum_final": hist_rows[-1]["cum"],
        "step1_L2_vs_G_T0": abs(hist_rows[0]["L2"] - disp1),
        "step1_g_m4_vs_e193_committed":
            abs(hist_rows[0]["g_m4_mean_pz"] - E193_T1_PRIMARY),
        "fact_died_in_history_at_step": hist_dead,
        "note": "the wash-history trace (the span's source); step-1 rows "
                "double-check vs G_T0 / e193's committed post-step primary "
                "read — no committed 20-step f2 span exists (disclosed)",
    }
    log(f"  history: step1 L2 {hist_rows[0]['L2']:.6f} (vs G_T0 "
        f"{disp1:.6f}); step1 g-4 read {hist_rows[0]['g_m4_mean_pz']:.6f} "
        f"(vs e193 {E193_T1_PRIMARY:.6f}); fact died at step {hist_dead}")
    assert hist_check["step1_L2_vs_G_T0"] < 1e-3

    basis = svd_basis(H)
    sv = [float(s) for s in basis["sv"]]
    Vp = basis["Vp"]
    cum20 = wtheta - theta0
    drift = {f"dir{i}": float(torch.dot(cum20.double(),
                                        (Vp[i] / torch.norm(Vp[i])).double()))
             for i in (DIR_TOP, DIR_MID, DIR_LOW)}
    # e193's committed g-ray join (context)
    if (CKPT_DIR / GDIR_CK).exists():
        u_g = torch.load(CKPT_DIR / GDIR_CK, map_location="cpu",
                         weights_only=False)["u"]
        g_cos = {f"dir{i}": cos64(Vp[i] / torch.norm(Vp[i]), u_g)
                 for i in (DIR_TOP, DIR_MID, DIR_LOW)}
        del u_g
    else:
        g_cos = {"note": "direction checkpoint missing — join skipped"}
    span_block = {
        "n_segments": int(Vp.shape[0]), "rank_eff": basis["rank_eff"],
        "sv": sv, "participation_ratio": basis["pr"], "cond": basis["cond"],
        "sv_top": sv[0], "energy_frac_top1": sv[0] ** 2 / sum(s * s for s in sv),
        "profile": [s / sv[0] for s in sv],
        "energy_frac_dirs": {f"dir{i}": sv[i] ** 2 / sum(s * s for s in sv)
                             for i in (DIR_TOP, DIR_MID, DIR_LOW)},
        "wash_drift_proj_dirs": drift,
        "g_ray_cos_dirs_context": g_cos,
        "history": hist_check,
        "note": "the contiguous wash-history span (e209/e211's machinery at "
                "the f2 root); drift = <theta20-theta0, v_i> (the wash's own "
                "drift along each axis, sign vs the SVD-emitted +v); the "
                "g-ray cosines are context (e193's committed direction)",
    }
    log(f"  span: PR {basis['pr']:.4f}, cond {basis['cond']:.2f}, top SV "
        f"{sv[0]:.4f} (energy {span_block['energy_frac_top1']:.3f}); "
        f"dir energy fracs " + "/".join(
            f"d{i} {span_block['energy_frac_dirs'][f'dir{i}']:.3f}"
            for i in (DIR_TOP, DIR_MID, DIR_LOW))
        + f"; g-ray cos " + "/".join(
            f"{v:.3f}" for k, v in g_cos.items() if k.startswith("dir")))
    metrics["span"] = span_block
    write_partial("P1 span done (PR %.4f)" % basis["pr"])
    del H, segs, wtheta

    # ================= P2: the un-exposed baselines + the dose =================
    log("=" * 78)
    log("P2: the un-exposed baseline kill rays (asserted BEFORE the dose)")

    def unit(i: int) -> torch.Tensor:
        return (Vp[i] / torch.norm(Vp[i])).clone()

    v_top, v_mid, v_low = unit(DIR_TOP), unit(DIR_MID), unit(DIR_LOW)
    evl = copy.deepcopy(net0)

    @torch.no_grad()
    def read_state(theta: torch.Tensor) -> dict:
        load_flat(evl, theta)
        out = {"primary": battery_cell(evl, primary_ids, zid)["mean_pz"]}
        for j in CO_RULERS_J:
            out[f"coruler_g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        out["ce_r"] = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
        return out

    def walk_kill(theta_s: torch.Tensor, v: torch.Tensor, tag: str):
        """e211's onset-grid kill walk from theta_s along -v (early stop)."""
        rows, kill = [], None
        load_flat(evl, theta_s)
        gz0 = battery_cell(evl, primary_ids, zid)
        rows.append({"D": 0.0, "gm": gz0["mean_pz"],
                     "frac": gz0["frac_argmax_z"]})
        prev = gz0["mean_pz"]
        for D in D_GRID:
            load_flat(evl, theta_s - D * v)
            gz = battery_cell(evl, primary_ids, zid)
            rows.append({"D": float(D), "gm": gz["mean_pz"],
                         "frac": gz["frac_argmax_z"]})
            if gz["mean_pz"] <= SHUT_BAR and prev > SHUT_BAR:
                kill = interp_d_kill(prev, gz["mean_pz"], float(D) - 0.05,
                                     float(D))
                break
            prev = gz["mean_pz"]
        log(f"  [{tag}] kill "
            + (f"{kill:.4f}" if kill is not None else "CENSORED (>3.0)")
            + f" ({len(rows)} grid evals)")
        return rows, kill

    base_dir_kills, baselines = {}, {}
    for tag, v in (("top", v_top), ("mid", v_mid), ("low", v_low)):
        rows, kill = walk_kill(theta0, v, f"BASE dir-{tag}")
        base_dir_kills[tag] = {"dir": {"top": DIR_TOP, "mid": DIR_MID,
                                       "low": DIR_LOW}[tag],
                               "energy_frac": sv[{"top": DIR_TOP, "mid": DIR_MID,
                                                  "low": DIR_LOW}[tag]] ** 2
                               / sum(s * s for s in sv),
                               "D_kill": kill, "rows": rows}
        baselines[tag] = kill
    metrics["baselines"] = base_dir_kills
    write_partial("P2 baseline rays done")

    t_top = baselines["top"]
    if t_top is None:
        metrics["dose_gate_failure"] = (
            "the top-SV baseline ray is CENSORED past 3.0 — the dose "
            "arithmetic (eps = 0.1 x t_top) cannot be set; cell aborts per "
            "the registration (metrics written before the assert — e187)")
        write_partial("P2 DOSE GATE FAILED (top ray censored)")
        raise RuntimeError("top-SV baseline kill-D unresolved — no dose")

    eps = DOSE_FRAC * t_top
    metrics["dose"] = {
        "baseline_kill_top": t_top,
        "dose_frac": DOSE_FRAC, "eps_L2": eps,
        "eps_frac_of": {"mid": eps / baselines["mid"] if baselines["mid"] else None,
                        "low": eps / baselines["low"] if baselines["low"] else None},
        "note": "eps = 0.1 x the top-SV direction's own un-exposed kill-D "
                "(sub-lethal by construction along the exposed axis); the "
                "same eps for all three arms (matched L2)",
    }
    log(f"  DOSE: eps = {DOSE_FRAC} x t_top = {eps:.4f} L2 "
        + (f"(= {eps / baselines['mid']:.2f}x mid's kill, "
           f"{eps / baselines['low']:.2f}x low's kill)"
           if baselines["mid"] and baselines["low"] else ""))

    # ---- the control draw (after eps is fixed; acceptance = sub-lethal) ------
    ctrl_draws = []
    u_ctrl = None
    gen_c = torch.Generator().manual_seed(CTRL_SEED)
    for k in range(CTRL_MAX_DRAWS):
        coeffs = torch.randn(Vp.shape[0], generator=gen_c)
        u = torch.mv(Vp.T, coeffs)
        u = (u / torch.norm(u)).clone()
        theta_c = theta0 + eps * u
        load_flat(evl, theta_c)
        r0 = battery_cell(evl, primary_ids, zid)["mean_pz"]
        ctrl_draws.append({"draw": k + 1, "read_at_exposed_state": r0,
                           "accepted": bool(r0 > SHUT_BAR)})
        if r0 > SHUT_BAR:
            u_ctrl = u
            break
        del u
    metrics["control_draws"] = {
        "seed": CTRL_SEED, "max_draws": CTRL_MAX_DRAWS, "draws": ctrl_draws,
        "n_draws": len(ctrl_draws),
        "selection_disclosure": "accepted = first draw whose exposed state "
                                "reads alive; the selection biases the "
                                "control toward forgiveness — i.e. AGAINST "
                                "the IMMUNITY clause (conservative)",
    }
    if u_ctrl is None:
        metrics["dose_gate_failure"] = (
            f"all {CTRL_MAX_DRAWS} control draws lethal at eps — the "
            "matched-dose control cannot be constructed; cell aborts")
        write_partial("P2 CONTROL GATE FAILED (all draws lethal)")
        raise RuntimeError("control draws all lethal at eps")

    rows, kill = walk_kill(theta0, u_ctrl, "BASE ctrl-own")
    base_ctrl = {"dir": "ctrl (random in-span)", "D_kill": kill, "rows": rows,
                 "n_draws_to_accept": len(ctrl_draws)}
    metrics["baselines"]["ctrl"] = base_ctrl
    baselines["ctrl"] = kill
    log(f"  control accepted on draw {len(ctrl_draws)} "
        f"(read {ctrl_draws[-1]['read_at_exposed_state']:.4f}); own baseline "
        + (f"kill {kill:.4f}" if kill is not None else "CENSORED"))

    # ================= P3: the exposure arms ===================================
    log("=" * 78)
    log("P3: the exposure arms (theta_pre = theta0 + eps*v; the "
        "pass-back-through geometry)")

    ARMS = [
        ("TOP-EXP", v_top, {"own": "top", "same_axis_base": "top"}),
        ("LOW-EXP", v_low, {"own": "low", "same_axis_base": "low"}),
        ("CTRL-EXP", u_ctrl, {"own": "ctrl", "same_axis_base": "ctrl"}),
    ]
    RAYS = {"top": v_top, "mid": v_mid, "low": v_low, "ctrl": u_ctrl}
    arm_results: dict = {}
    for arm_tag, v_exp, spec in ARMS:
        theta_pre = theta0 + eps * v_exp
        state_read = read_state(theta_pre)
        sublethal = bool(state_read["primary"] > SHUT_BAR)
        gate_cell = {
            "exposed_state_primary_read": state_read["primary"],
            "corulers": {k: v for k, v in state_read.items()
                         if k.startswith("coruler")},
            "ce_r": state_read["ce_r"],
            "ce_r_delta_vs_root": state_read["ce_r"] - root_cells["ce_r"],
            "sublethal": sublethal,
            "pass": sublethal,
        }
        log(f"P3[{arm_tag}] exposed state: primary {state_read['primary']:.4f}, "
            f"CE_R {state_read['ce_r']:.4f} (root "
            f"{root_cells['ce_r']:.4f}): "
            + ("PASS" if sublethal else "FAIL"))
        cell = {"exposure": arm_tag, "eps": eps, "gate": gate_cell}
        if sublethal:
            krows = {}
            for ray_tag in ("top", "mid", "low", "ctrl"):
                if ray_tag == "ctrl" and arm_tag != "CTRL-EXP":
                    continue          # the ctrl ray is walked only at its own arm
                rows_k, kill_k = walk_kill(theta_pre, RAYS[ray_tag],
                                           f"{arm_tag} ray-{ray_tag}")
                same = (ray_tag == spec["same_axis_base"])
                t_b = baselines[ray_tag]
                ratio = (kill_k / t_b) if (kill_k is not None
                                           and t_b is not None) else None
                mech = (1.0 + eps / t_b) if t_b is not None else None
                resid = (kill_k - eps - t_b) if (kill_k is not None
                                                 and t_b is not None) else None
                krows[ray_tag] = {
                    "D_kill": kill_k, "D_kill_baseline": t_b,
                    "ratio_vs_baseline": ratio,
                    "same_axis": same,
                    "mechanical_ceiling_ratio": mech if same else None,
                    "causal_residue_D": resid if same else None,
                    "rows": rows_k,
                }
                if same:
                    log(f"    same-axis row: kill {kill_k:.4f} vs base "
                        f"{t_b:.4f} -> ratio {ratio:.3f} "
                        f"(mechanical ceiling {mech:.3f}; residue "
                        f"{resid:+.4f})")
            cell["kill_rows"] = krows
        else:
            cell["note"] = ("LETHAL at eps — sub-lethality gate FAILED; "
                            "the arm's rows are skipped (disclosed)")
        arm_results[arm_tag] = cell
        metrics["arms"] = arm_results
        write_partial(f"P3[{arm_tag}] done")
    metrics["envelope"]["cpu_load_pct_after_P3"] = cpu_load_probe()

    # CE_R canary at the root's dir-0 kill position (context)
    load_flat(evl, theta0 - t_top * v_top)
    ce_r_at_kill = ce_fixed_cpu(evl, r_eval_x, r_eval_y)
    metrics["ce_r_canary"] = {
        "root": root_cells["ce_r"],
        "root_dir0_kill_position": ce_r_at_kill,
        "exposed_states": {a: arm_results[a]["gate"]["ce_r"]
                           for a in arm_results},
        "note": "the organism-health honesty reflex (report-only): a "
                "pre-exposure that wrecks general CE while the ruler stays "
                "alive would confound any immunity reading",
    }

    # ================= P4: the adjudication (frozen bars) ======================
    log("=" * 78)
    log("P4: adjudication (the frozen bars; the desk disclosure's arithmetic "
        "carried alongside, never replacing them)")

    def row(arm, ray):
        try:
            return metrics["arms"][arm]["kill_rows"][ray]
        except KeyError:
            return None

    r_a = row("TOP-EXP", "top")
    r_ctrl_own = row("CTRL-EXP", "ctrl")
    ratio_a = r_a["ratio_vs_baseline"] if r_a else None
    ctrl_own_ratio = (r_ctrl_own["ratio_vs_baseline"] if r_ctrl_own else None)
    ctrl_censored = (r_ctrl_own is None
                     or r_ctrl_own["D_kill"] is None)

    imm_fires = bool(ratio_a is not None and ratio_a >= IMMUNITY_RATIO
                     and (ctrl_own_ratio is not None
                          and ctrl_own_ratio < ratio_a))
    imm_control_clause = ("resolved: control-own ratio "
                          f"{ctrl_own_ratio:.3f} < {ratio_a:.3f}"
                          if (imm_fires or ctrl_own_ratio is not None)
                          else ("UNRESOLVABLE (the control's own row is "
                                "censored) — IMMUNITY cannot fire per the "
                                "registration" if ctrl_censored else ""))
    sens_fires = bool(ratio_a is not None and ratio_a < SENS_BELOW)

    all_ratios = []
    for arm in arm_results:
        for ray_tag, rr in metrics["arms"][arm].get("kill_rows",
                                                    {}).items():
            if rr["ratio_vs_baseline"] is not None:
                all_ratios.append((f"{arm}/{ray_tag}",
                                   rr["ratio_vs_baseline"]))
    resid_a = r_a["causal_residue_D"] if r_a else None
    null_ok = bool(
        not imm_fires and not sens_fires
        and all(NULL_LO <= rv <= NULL_HI for _, rv in all_ratios)
        and resid_a is not None and abs(resid_a) <= RESIDUE_QTOL)

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "shakedown only"
    elif imm_fires:
        verdict = "IMMUNITY-CAUSAL"
        clause = (f"the top-SV pre-exposure raised that direction's kill-D "
                  f"to {ratio_a:.3f}x (>= {IMMUNITY_RATIO}) with the "
                  f"random-direction control showing less "
                  f"({imm_control_clause}) — the wash vaccinates its own span")
    elif sens_fires:
        verdict = "SENSITIZATION"
        clause = (f"the top-SV pre-exposure LOWERED that direction's kill-D "
                  f"to {ratio_a:.3f}x (< {SENS_BELOW}) — the ordering is a "
                  f"correlation, not protection; the residue "
                  f"{resid_a:+.4f} vs the mechanical "
                  f"{(r_a['D_kill_baseline'] + eps):.4f} says the +v exposure "
                  f"segment dips under the bar")
    elif null_ok:
        verdict = "NULL"
        clause = ("no material change either way: every resolved ratio in "
                  f"[{NULL_LO}, {NULL_HI}] and the same-axis residue within "
                  f"one grid step of the pass-back arithmetic — the ordering "
                  f"at these doses is not intervenable")
    else:
        verdict = "GRADED"
        off = [f"{k} {v:.3f}" for k, v in all_ratios
               if not (NULL_LO <= v <= NULL_HI)]
        clause = ("a partial — the tables verbatim; off-window rows: "
                  + ("; ".join(off) if off else "none (residue off-window)")
                  + (f"; dir-0 residue {resid_a:+.4f}" if resid_a is not None
                     else ""))

    metrics["adjudication"] = {
        "bars": {"IMMUNITY_CAUSAL": {"fires": imm_fires},
                 "SENSITIZATION": {"fires": sens_fires},
                 "NULL": {"fires": bool(null_ok and not SMOKE)},
                 "GRADED": {"fires": bool(not imm_fires and not sens_fires
                                          and not null_ok and not SMOKE)}},
        "inputs": {
            "ratio_dir0_after_top_exp": ratio_a,
            "mechanical_ceiling_dir0": (1.0 + eps / t_top),
            "residue_dir0": resid_a,
            "ctrl_own_ratio": ctrl_own_ratio,
            "ctrl_own_censored": ctrl_censored,
            "immunity_control_clause": imm_control_clause,
            "all_ratios": dict(all_ratios),
        },
        "verdict": verdict,
        "clause": clause,
        "composite_order": "IMMUNITY-CAUSAL / SENSITIZATION / NULL / GRADED "
                           "(frozen before compute; the first two mutually "
                           "exclusive by construction)",
        "desk_arithmetic_recap": (
            f"the same-direction row is arithmetically capped at 1+eps/t = "
            f"{1.0 + eps / t_top:.3f} (the desk disclosure, stated before "
            f"compute); measured {ratio_a if ratio_a is None else round(ratio_a, 4)}"
            f" — the causal content lives in the residues and the cross rows"),
    }
    log(f"VERDICT: {verdict} — {clause}")
    write_partial("P4 adjudicated")

    # ================= the figure ==============================================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle("E222 — the exposure-immunity causal test (the e193 f2 "
                 f"root, 873k, CPU) | eps = {DOSE_FRAC}x t_top = "
                 f"{eps:.3f} L2 | verdict: {verdict}", fontsize=11)

    ax = axes[0][0]
    ax.semilogy(range(1, len(sv) + 1), sv, "o-", ms=4, color="tab:gray")
    for i, c, nm in ((DIR_TOP, "tab:red", "top (d0)"),
                     (DIR_MID, "tab:purple", "mid (d9)"),
                     (DIR_LOW, "tab:blue", "low (d19)")):
        ax.semilogy(i + 1, sv[i], "o", ms=9, color=c, label=nm)
    ax.set_xlabel("span direction (by SV rank)")
    ax.set_ylabel("singular value (fp64)")
    ax.set_title(f"the wash-span spectrum (PR {basis['pr']:.2f}); "
                 f"g-ray cos d0 {g_cos.get('dir0', float('nan')):+.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    if r_a:
        b_rows = metrics["baselines"]["top"]["rows"]
        ax.plot([r["D"] for r in b_rows], [r["gm"] for r in b_rows], "o-",
                ms=3, color="tab:red", label="baseline walk from theta0")
        a_rows = r_a["rows"]
        ax.plot([r["D"] for r in a_rows], [r["gm"] for r in a_rows], "s-",
                ms=3, color="tab:orange",
                label=f"walk from TOP-EXP state (eps={eps:.3f})")
        ax.axvline(t_top, color="tab:red", ls=":", lw=1)
        ax.axvline(t_top + eps, color="tab:orange", ls=":", lw=1)
        ax.plot([r["D"] + eps for r in b_rows], [r["gm"] for r in b_rows],
                "--", lw=1, color="k", alpha=0.6,
                label="the mechanical shift (baseline + eps)")
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8)
    ax.set_xlabel("D (L2 along the walk)")
    ax.set_ylabel("primary ruler (g-4 install-60 mean pZ)")
    ax.set_title(f"the same-direction test: ratio "
                 f"{ratio_a if ratio_a is None else round(ratio_a, 3)} vs "
                 f"mechanical {1 + eps / t_top:.3f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    colors = {"BASE": "tab:gray", "TOP-EXP": "tab:red",
              "LOW-EXP": "tab:blue", "CTRL-EXP": "tab:green"}
    ax.plot([r["D"] for r in metrics["baselines"]["mid"]["rows"]],
            [r["gm"] for r in metrics["baselines"]["mid"]["rows"]], "o-",
            ms=3, color=colors["BASE"], label="baseline (theta0)")
    for arm in ("TOP-EXP", "LOW-EXP", "CTRL-EXP"):
        rr = row(arm, "mid")
        if rr:
            ax.plot([r["D"] for r in rr["rows"]], [r["gm"] for r in rr["rows"]],
                    "s-", ms=3, color=colors[arm],
                    label=f"{arm} (ratio {rr['ratio_vs_baseline']:.2f})")
    ax.axhline(SHUT_BAR, color="k", ls="-", lw=0.8)
    ax.set_xlabel("D (L2 along the walk)")
    ax.set_ylabel("primary ruler")
    ax.set_title("the cross-direction test: dir-9 (mid-SV) kill walks from "
                 "every state")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    labels, vals, cols, ceilings = [], [], [], []
    for arm in ("TOP-EXP", "LOW-EXP", "CTRL-EXP"):
        for ray_tag in ("top", "mid", "low", "ctrl"):
            rr = row(arm, ray_tag)
            if rr and rr["ratio_vs_baseline"] is not None:
                labels.append(f"{arm[0]}:{ray_tag[0]}")
                vals.append(rr["ratio_vs_baseline"])
                cols.append(colors[arm])
                ceilings.append(rr["mechanical_ceiling_ratio"])
    y = range(len(labels))
    ax.barh(list(y), vals, color=cols, alpha=0.75)
    for yi, (v, c_) in enumerate(zip(vals, ceilings)):
        if c_ is not None:
            ax.plot([c_], [yi], "k|", ms=14, mew=2)
    ax.axvline(1.0, color="k", ls="-", lw=0.8)
    ax.axvline(IMMUNITY_RATIO, color="tab:green", ls="--", lw=1,
               label=f"IMMUNITY bar {IMMUNITY_RATIO}")
    ax.axvline(SENS_BELOW, color="tab:purple", ls="--", lw=1,
               label="SENSITIZATION (< 1.0)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("kill-D ratio vs the un-exposed baseline")
    ax.set_title("the ratio table (| = the same-axis mechanical ceiling "
                 "1+eps/t; T:top L:low C:ctrl arms / t:top m:mid l:low "
                 "c:ctrl rays)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = rd / "e222_exposure_immunity.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log(f"figure -> {fig_path}")

    # ================= provenance + honesty + final write ======================
    metrics["provenance"] = {
        "parents_files_md5": parents_md5,
        "machinery": {
            "family2_gates_battery_stream": "lab/e193_organism_replicate.py "
                                            "VERBATIM (dial hard-bind, e185-"
                                            "hashed stream, e170 bank, CE_R "
                                            "val windows)",
            "wash_span_kill": "lab/e211_walled_band.py VERBATIM (the "
                              "contiguous 20-step unwalled AdamW wash, the "
                              "fp64 Gram-SVD basis, the onset-grid 0.05..3.00 "
                              "kill walk, first-downcrossing interpolation, "
                              "SVD-emitted signs) — the ruler swapped to "
                              "e193's PRIMARY per its frozen RULER DISCLOSURE",
            "the_exposure_cell": "NEW this cell (registered): the sub-lethal "
                                 "displacement arms, the pass-back-through "
                                 "geometry, the mechanical-ceiling/residue "
                                 "decomposition, the control acceptance rule",
            "committed_records": "e157 (the dial + wash step-1 rows), e193 "
                                 "(the t0 rows + the g-ray checkpoint), e211 "
                                 "(the ordering this cell interrogates) — "
                                 "loaded, hard-bound, never re-run",
        },
        "registry_note": f"the control seed {CTRL_SEED} is fresh (the "
                         "registry held through 12401-12408 at e211)",
    }
    metrics["honesty"] = {
        "n_and_scope": ("n=1 organism (the e193 f2 root, one lineage); one "
                        "span realization (a single 20-step history draw — "
                        "the draw lottery moves every direction); ONE dose "
                        f"(eps = {DOSE_FRAC}x t_top); one ruler family; "
                        "every kill ray walked once"),
        "perturbation_vs_training": ("the pre-exposure is a DIRECT weight "
                                     "displacement along a span axis — the "
                                     "wash's mechanism at ONE REMOVE, not the "
                                     "wash itself: the wash moves by AdamW "
                                     "steps whose directions are gradient-"
                                     "determined and whose per-step L2 is "
                                     f"{E193_STEP_L2:.3f} (vs eps {eps:.4f} "
                                     "here); immunity-to-TRAINING is not "
                                     "tested by this cell"),
        "pass_back_arithmetic": ("every same-axis row is arithmetically "
                                 "capped at 1+eps/t_base (the desk "
                                 "disclosure, stated before compute); the "
                                 "residues and cross rows carry the causal "
                                 "content; the bars adjudicated the raw "
                                 "numbers as registered"),
        "sign_convention": ("the exposure is +v (the SVD-emitted sign, "
                            "OPPOSED to the kill walk -v) at every arm; the "
                            "wash's own drift along each axis is reported "
                            "(span.wash_drift_proj_dirs) so the reader can "
                            "see whether +v is with or against the wash"),
        "control_selection": ("the control was accepted as the first "
                              "sub-lethal draw ("
                              f"{len(ctrl_draws)} draw(s) from seed "
                              f"{CTRL_SEED}) — biased toward forgiveness, "
                              "i.e. against the IMMUNITY clause"),
        "ce_r_canary": metrics["ce_r_canary"],
        "nothing_guaranteed": ("the openness was the point: the residues "
                               "could have been anything; the observed "
                               f"outcome is '{verdict}'"),
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["elapsed_s"] = round(time.time() - T0, 1)
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log(f"E222 DONE in {time.time() - T0:.1f}s — verdict {verdict}")


if __name__ == "__main__":
    main()
