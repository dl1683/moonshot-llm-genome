"""E238 — FQ7, THE ONE-TEMPERATURE-PER-STATE NULL (eval-only, CPU).

WHY (the ideator's Rank 2, urgency-flagged, from scratch/
review_ideator_2026-10-04.md): e232/T212 killed the temperature
lens for the MARGIN side by derivation (temperature preserves the
sigma-normalized margin exactly — the family is a vertical line);
but the P side was never fitted. How much of the 54-probe belief
tensor does ONE scalar temperature per state explain? If a single
T(t) per state accounts for most of every probe's p-decline, the
p-side story is "uniform thermal erosion + a small differential
cut" — and e234's join (a) (belief decline vs wind-alignment)
would ride a uniform flattening if one temperature per state
explains the p-side. If the residuals are STRUCTURED (carrying
the erosion order and the Gmail/iPhone split), the third
dimension has a p-side seat and the critic's confound fear
(scratch/review_critic_2026-10-04.md: "is the differentiation
just differential thermal exposure?") is discharged. THIS CELL IS
TIME-SENSITIVE: it must land before the in-flight e234's
adjudication is read.

THE CELL:
  (1) THE PRIMARY FIT on committed p records: for each state t of
      wash-1 (+2/+10/+50/+80) and wash-2 (+10/+50/+80), fit ONE
      T(t) by MLE across ALL 54 probes simultaneously under the
      temperature family p_i(T) = softmax(logits_i/T)[answer_i].
      The journals (runs/e214, runs/e228) commit per-probe
      (p, margin_raw, sigma, top1_id, top2_id) — NOT full logits
      — so the committed states are re-probed ONCE with full
      logit dumps at the answer position (54 probes x 8 states,
      minutes of CPU, the e182-family battery machinery
      module-imported VERBATIM; every state re-probe-certified
      against e214's committed per-probe p, the e228 convention,
      tol 0.005).
  (2) THE VARIANCE-EXPLAINED READ: fraction of the p-decline
      (from t=0) explained by the one-T family, per state,
      pooled (all 54) and per battery.
  (3) THE RESIDUAL STRUCTURE: per-probe residuals vs the
      committed erosion order (e214's records; Spearman) and the
      Gmail/iPhone anchors' residuals against the product-family
      (ctrl/make) band.

REGISTERED BARS (frozen BEFORE compute, VERBATIM from the
dispatch brief — QUEUE row 'DISPATCHED ~15:06Z Oct-4'; the
variance bar is PRE-REGISTERED per the gerrymander guard; the
T-family with per-state freedom is an ~80-parameter null in the
brief's framing — the actual fitted DOF is 7 scalars against
54 x 7 = 378 observed p's, disclosed here):
  - FLATTENING-DOMINANT — ">= 80% of pooled p-decline variance
    explained by the one-T family at every state — the p-side is
    uniform thermal erosion; the differential cut (the residual)
    is the named next instrument"
  - STRUCTURED — "< 80% explained AND the residuals correlate
    with the committed erosion order (|rho| >= 0.4) or the
    anchors separate from the family band at >= 2 sigma — the
    third dimension has a p-side seat; the critic's confound
    discharged"
  - MIXED — "between — the tables verbatim, both reads"
  ALSO REGISTERED (verbatim): "the per-probe two-moment
  approximation check (if re-probing was declined: how much does
  the (p, margin_raw, sigma) triple constrain T per probe — the
  honest bound on the desk-only variant)". Re-probing was NOT
  declined (no committed logits exist), so this runs as the
  desk-only bound co-report: the Gaussian-denominator family
  built from e228's committed (p0, sigma0) triples, fit against
  e214's committed p's, zero forwards — priced against the full
  fit.

OPERATIONALIZATIONS (frozen before compute; they fix the
clauses, they do not move the bars):
  * THE FAMILY: q_i(T) = softmax(L0_i / T)[ans_i], L0_i = the
    re-probed t=0 full answer-position logits (certified by
    p-match against e214's committed t0), ans_i = the battery's
    frozen answer id. ONE T per (wash, state); no other fitting
    freedom (the null's austerity is the point).
  * THE MLE: Bernoulli cross-entropy with the observed beliefs
    as soft targets — NLL(T) = -sum_i [p_obs,i log q_i(T) +
    (1-p_obs,i) log(1-q_i(T))] (fractional-count binomial; the
    argmax is count-independent); minimized over log10(T) on a
    1201-point grid in [-1.0, +1.0] (T in [0.1, 10]) then
    golden-section polished. Deterministic; no seeds.
  * OBSERVED p's: the re-probed state p's (== e214's committed
    records within tol; expected ~0 — same fp32 weights, same
    CPU fp32 probe path).
  * R2 DECLINE (the variance-explained read): R2(w,s) = 1 -
    sum_i (p_obs,i(w,s) - q_i(T*))^2 / sum_i (p_obs,i(w,s) -
    p0_i)^2 — the denominator is the squared decline from t=0
    (the no-decline null; T=1 gives R2 = 0 by construction).
    Pooled over all 54 (the bar's object); per battery
    co-reported.
  * ADJUDICATION STATES: w1/w2 x {10, 50, 80} (6 cells).
    FLATTENING-DOMINANT := gates ok AND every adjudication
    state's POOLED R2 >= 0.80. +2 (wash-1-only) co-reports,
    never adjudicates (e228's frozen convention, inherited).
  * EROSION CLAUSE: erosion_i(w,s) = 1 - p_i(w,s)/p_i(0) from
    e214's COMMITTED records; residual_i(w,s) = p_obs,i -
    q_i(T*). The primary statistic is ONE number: the blocked
    Spearman between residual and erosion, pooled over the deep
    cells (battery x wash x {50,80}), z-scoring both variables
    WITHIN each (battery, wash, state) cell first (the committed
    erosion order is per-battery by construction — battery/wash/
    state mean shifts removed, the within-cell order retained;
    near's n=3 ranks included, flagged coarse, with a
    near-excluded sensitivity echo). Fires at |rho| >= 0.4.
  * ANCHOR CLAUSE: anchors = Gmail ("The email service made by
    Google->Gmail") and iPhone ("The phone made by Apple->
    iPhone"); band = the ctrl/make probes that are not anchors
    (Xbox, Chrome, iPad, iTunes, PlayStation). Per anchor:
    z = (mean residual over the 4 deep cells) - (mean band
    residual over the same cells) / (std of band residuals over
    the same cells, ddof=1). Fires if max |z| >= 2. Co-reported:
    the Gmail-vs-iPhone split z (the opposite-fates object) and
    per-wash versions.
  * STRUCTURED := NOT flattening-dominant AND (erosion clause OR
    anchor clause). MIXED := otherwise.
  * TWO-MOMENT DESK-ONLY BOUND (registered, co-report): family
    q_i^2m(T) = 1/[1 + (N-1) exp((mu_i - a_i)/T + sigma0_i^2/
    (2T^2))] with (mu_i - a_i) = ln((1-p0_i)/(p0_i(N-1))) -
    sigma0_i^2/2 from e228's COMMITTED t0 triples, N = the
    dumped vocab size; same MLE against e214's COMMITTED state
    p's; report T_2m vs T_full and R2_2m vs R2_full. Per-probe
    constraint: T_i = the root of q_i(T) = p_obs,i (full) and
    q_i^2m(T) = p_obs,i (desk-only), root-found on the grid +
    bisection; the per-state spread (IQR/median, and the
    n-probes-outside-family-range count) = how much one probe's
    record constrains T.
  * MULTIPLICITY DISCIPLINE: exactly ONE explained-fraction bar
    + TWO structure clauses adjudicate, all pre-registered
    above; everything else (LSQ-on-p T echo, full-vocab KL T
    echo, per-battery R2, per-cell rho table, T trajectories,
    near-excluded sensitivity, anchor split echoes) is a
    co-report that never adjudicates.

VERIFICATION GATES (registered tolerances, frozen before
compute; the e228 pattern):
  * G_STATES — both archives inventoried (sizes/mtimes); every
    state's per-probe p (from the SAME forward as the logit
    dump) compared to e214's committed journal records within
    0.005 (expected ~0: same fp32 weights, same CPU fp32 probe
    path). This certifies the dumped logits.
  * G_BATT — the four batteries rebuilt VERBATIM (module import)
    reproduce e214's committed journal t0: same kept sets,
    per-probe |dp| <= 0.010.
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to
    e182's recorded filter stats (INHERITED FROZEN).
  * G_ENV — CPU-only (zero GPU calls), torch threads <= 4, load
    checks, one state = one eval burst.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — n=2
washes is texture, not law; wash 1 is a CPU fp32 replay of
e182's GPU original, wash 2 is GPU fp32 (the archive's device
asymmetry, inherited, disclosed); ONE organism (the shared
pristine 124M); the temperature family is a LENS, not a
mechanism claim — a good fit constrains the p-side's
description, it does not assert the wash implements a
temperature; the Bernoulli soft-target MLE is one natural
likelihood choice (the LSQ echo prices it); the blocked pooled
Spearman is one pre-registered aggregation of 12 per-cell joins
(the per-cell table rides along, never adjudicates); near is
n=3 (ranks quantized); the two-moment bound replaces the true
denominator with a Gaussian tail estimate (its gap vs the full
fit prices exactly that crudeness); nothing here is guaranteed.

COMPUTE ENVELOPE: CPU-only desk+eval (dispatch; two other
agents live — load-polite); threads 4; ~54 probes x 8 state
reads + screening forwards + 7 checkpoint loads; minutes.
Progressive PARTIAL metrics + resumable journal after every
state (the standing disruption rule).

PROVENANCE: the organism, corpus filter+verify, fact/ctrl/near
batteries and probe machinery are lab/e182c_forgetting_control.py
VERBATIM via module import (inheriting lab/e182_gpt2_wash.py);
the template battery is lab/e182c2_template.py VERBATIM via
module import; the two-wash archive, the census loop, the gate
pattern and the rank-stat helpers are lab/e214's and
lab/e228's (helpers copied verbatim, noted inline). The
committed records (runs/e214/metrics.json + journal.json,
runs/e228/journal.json) are read at RUNTIME (never
transcribed). Builds on: FQ7 (the ideator's Rank 2 — this
cell's question), T212/e232 (the temperature derivation that
killed the margin side — the p side is the open half), T209/
e230 + T211/e227 (the p-side narratives this constrains:
zombie decisions' beliefs, cross-scale zombie structure),
T187/e214 (the erosion order + the committed per-probe
records), T185/e213 (the two-wash archive), T183/e182c2 + T149/
e182c + T123/e182 (the machinery). NEW: the one-T-per-state MLE
fit of the 54-probe belief tensor; the decline-variance-
explained read; the residual-vs-erosion-order blocked join; the
anchor-vs-band residual separation; the two-moment desk-only
bound; the full answer-position logit dumps (the reusable
artifact).

Run:  cd lab && python e238_temperature_null.py   (E238_SMOKE=1:
      t=0 + the w1 +10 state only, own smoke dir, nothing
      adjudicated)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

import common                                                # noqa: E402
from common import now_iso, run_dir, save_json               # noqa: E402

import e182c_forgetting_control as e1                        # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                                 # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                     # the dispatch envelope (load-polite, <= 4)

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import textwrap                                              # noqa: E402

SMOKE = os.environ.get("E238_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e238_smoke" if SMOKE else "e238"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants (registered BEFORE compute) ---------------------
SHARED_STATES: tuple[int, ...] = (10,) if SMOKE else (10, 50, 80)
W1_ONLY_STATES: tuple[int, ...] = () if SMOKE else (2,)   # wash-1-only curve point
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
W1_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c_s{s}.pt"
           for s in SHARED_STATES + W1_ONLY_STATES}
W2_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c2_fresh_s{s}.pt"
           for s in SHARED_STATES}
E214_METRICS = common.REPO / "runs" / "e214" / "metrics.json"
E214_JOURNAL = common.REPO / "runs" / "e214" / "journal.json"
E228_JOURNAL = common.REPO / "runs" / "e228" / "journal.json"
ANCHORS = ("Gmail", "iPhone")            # substrings of the ctrl/make probe facts
ANCHOR_RELATION = "make"                   # the product-family band's relation

# ---- registered bar constants (frozen) ----------------------------------------
EXPLAINED_BAR = 0.80     # ">= 80% of pooled p-decline variance explained"
RHO_BAR = 0.40           # "|rho| >= 0.4" (the erosion-order residual clause)
Z_BAR = 2.0              # ">= 2 sigma" (the anchor-vs-band clause)
ADJ_STATES = (10, 50, 80)   # the adjudication states (both washes); +2 co-reports
DEEP_STATES = (50, 80)      # the residual-structure cells

# ---- registered verification tolerances (frozen) -------------------------------
TOL_PROBE_DP = 0.010   # per-probe t=0 dp vs e214's committed journal record
TOL_STATE_DP = 0.005   # per-probe dp on LOADED states vs e214's committed records

# the fit grid (deterministic; frozen)
GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201     # log10(T) in [-1, 1] -> T in [0.1, 10]
EPS = 1e-12

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "FLATTENING-DOMINANT": '">= 80% of pooled p-decline variance explained '
            'by the one-T family at every state — the p-side is uniform thermal '
            'erosion; the differential cut (the residual) is the named next '
            'instrument"',
        "STRUCTURED": '"< 80% explained AND the residuals correlate with the '
            'committed erosion order (|rho| >= 0.4) or the anchors separate '
            'from the family band at >= 2 sigma — the third dimension has a '
            'p-side seat; the critic\'s confound discharged"',
        "MIXED": '"between — the tables verbatim, both reads"',
        "ALSO-REGISTERED": '"the per-probe two-moment approximation check (if '
            're-probing was declined: how much does the (p, margin_raw, sigma) '
            'triple constrain T per probe — the honest bound on the desk-only '
            'variant)" [re-probing was NOT declined — no committed logits '
            'exist; this runs as the desk-only bound co-report]',
    },
    "operationalizations": (
        "family q_i(T) = softmax(L0_i/T)[ans_i], L0_i = the re-probed t=0 full "
        "answer-position logits (p-certified vs e214's committed t0); ONE T "
        "per (wash, state), no other fitting freedom; MLE = Bernoulli "
        "cross-entropy with the observed beliefs as soft targets (fractional-"
        "count binomial, count-independent argmax), minimized over log10 T on "
        "a 1201-point grid in [-1,1] then golden-polished; R2(w,s) = 1 - "
        "sum_i (p_obs,i - q_i(T*))^2 / sum_i (p_obs,i - p0_i)^2 pooled over "
        "all 54 (T=1 gives R2=0 by construction; per-battery co-reported); "
        "adjudication states = w1/w2 x {10,50,80}, FLATTENING-DOMINANT := "
        "gates ok AND all pooled R2 >= 0.80 there, +2 co-reports only "
        "(e228's convention); erosion clause = ONE blocked Spearman between "
        "residual_i and committed erosion_i, pooled over battery x wash x "
        "{50,80} cells with both variables z-scored WITHIN cell, fires at "
        "|rho| >= 0.4 (near included, flagged coarse; near-excluded "
        "sensitivity echo; per-cell table co-reported); anchor clause = "
        "Gmail/iPhone mean residual over the 4 deep cells vs the ctrl/make "
        "non-anchor band (Xbox/Chrome/iPad/iTunes/PlayStation), "
        "z = mean-gap / band-std(ddof=1), fires if max |z| >= 2; STRUCTURED "
        ":= NOT flattening-dominant AND (erosion OR anchor); MIXED := "
        "otherwise; two-moment desk-only bound = Gaussian-denominator family "
        "from e228's committed (p0, sigma0) triples + e214's committed "
        "state p's, zero forwards, same MLE; per-probe T_i = grid+bisection "
        "root of q_i(T) = p_obs,i (full) and q_i^2m(T) = p_obs,i (desk-only); "
        "multiplicity: exactly one bar + two clauses adjudicate; every other "
        "read (LSQ T echo, full-vocab KL T echo, per-battery R2, per-cell "
        "rhos, anchor split echoes) co-reports, never adjudicates"),
    "registration": ("bars frozen VERBATIM from the dispatch brief BEFORE any "
                     "compute (commit at script birth; QUEUE row 'DISPATCHED "
                     "~15:06Z Oct-4'); adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY desk+eval on the committed checkpoints (no wash run here); "
    "CPU-only per the dispatch (two other agents live — threads 4, "
    "load-polite).",
    "The full logits were NOT committed anywhere (e214/e228 journals carry "
    "p/margin/sigma/top-id pairs only), so the primary fit uses the ONE-TIME "
    "re-probe with full answer-position logit dumps (54 x 8), every state "
    "re-probe-certified against e214's committed per-probe p (tol 0.005, "
    "expected ~0).",
    "The npz logit dumps (~11 MB/state fp32) are lab artifacts on disk, NOT "
    "git-tracked (the checkpoints/gradient-cache precedent; sha256 + shapes "
    "recorded in metrics provenance; regenerable deterministically from the "
    "committed checkpoints + this script).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: t=0 + the w1 +10 state only, own smoke dir, nothing "
    "adjudicated or verified.",
]


# ------------------------------------------------------------------ helpers
# (rank-stat helpers copied VERBATIM from lab/e228_margin_landscape.py, which
#  copied them from e214 — the house implementations, exact under ties)

def cpu_load_check(tag: str) -> dict:
    """The dispatch envelope's load check (CPU-only cell; nothing GPU)."""
    try:
        import psutil
        pct = float(psutil.cpu_percent(interval=1.0))
        rec = {"tag": tag, "cpu_percent": pct, "tool": "psutil"}
    except Exception as e:                                # noqa: BLE001
        pct, rec = None, {"tag": tag, "cpu_percent": None,
                          "tool": f"unavailable ({e})"}
    log(f"  [load:{tag}] cpu {pct if pct is None else round(pct, 1)}%")
    return rec


def avg_ranks(v: list[float]) -> list[float]:
    """Average ranks (1-based), exact under ties."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5


def spearman(xs, ys) -> tuple[float | None, int]:
    """Spearman rho = Pearson on average ranks (exact under ties)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None, 0
    ties = (len(xs) - len({round(x, 12) for x in xs})
            + len(ys) - len({round(y, 12) for y in ys}))
    return _pearson(avg_ranks(list(xs)), avg_ranks(list(ys))), ties


def kendall_tau_b(xs, ys) -> float | None:
    """Kendall tau-b (tie-adjusted) — the robustness echo, never adjudicated."""
    n = len(xs)
    if n < 2:
        return None
    con = dis = 0
    xt = yt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            p = dx * dy
            if p > 0:
                con += 1
            elif p < 0:
                dis += 1
            if dx == 0:
                xt += 1
            if dy == 0:
                yt += 1
    denom = ((con + dis + xt) * (con + dis + yt)) ** 0.5
    return (con - dis) / denom if denom > 0 else None


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(common.REPO),
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def _f(v, fmt="+.3f") -> str:
    """None-safe numeric formatting (smoke guards)."""
    return "n/a" if v is None else format(v, fmt)


# --------------------------------------------------- the temperature family

class TempFamily:
    """The one-T family on the t=0 dumped logits: q_i(T) = softmax(L0_i/T)[ans_i].

    Everything is float64 numpy; the softmax is max-shifted for stability.
    Precomputes lmax_i and d_i = L0_i - lmax_i once.
    """

    def __init__(self, L0: np.ndarray, ans_ids: np.ndarray):
        self.L0 = L0.astype(np.float64)
        self.ans = ans_ids.astype(np.int64)
        self.lmax = self.L0.max(axis=1)
        self.d = self.L0 - self.lmax[:, None]          # (n, V), <= 0
        self.da = self.d[np.arange(len(self.ans)), self.ans]   # (n,) <= 0
        self.n = len(self.ans)

    def q(self, T: float) -> np.ndarray:
        """The answer probabilities at temperature T (vector over probes)."""
        with np.errstate(over="ignore"):
            num = np.exp(self.da / T)
            den = np.exp(self.d / T).sum(axis=1)
        return num / np.maximum(den, EPS)

    def q_one(self, T: float, i: int) -> float:
        """q at temperature T for probe i only (cheap: one vocab row)."""
        with np.errstate(over="ignore"):
            num = np.exp(self.da[i] / T)
            den = np.exp(self.d[i] / T).sum()
        return float(num / max(den, EPS))

    def grid_Q(self, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
        """One shared grid pass: (log10 T grid, Q (G, n) answer probs)."""
        grid = np.linspace(GRID_LO, GRID_HI, n)
        Q = np.stack([self.q(10.0 ** g) for g in grid])
        return grid, Q

    def _golden(self, f, lo: float, hi: float) -> float:
        gr = (5 ** 0.5 - 1) / 2
        a, b = lo, hi
        for _ in range(80):
            c, d_ = b - gr * (b - a), a + gr * (b - a)
            if f(c) < f(d_):
                b = d_
            else:
                a = c
            if b - a < 1e-8:
                break
        return (a + b) / 2

    def nll_bernoulli(self, T: float, p_obs: np.ndarray) -> float:
        qv = np.clip(self.q(T), EPS, 1.0 - EPS)
        return float(-(p_obs * np.log(qv)
                       + (1.0 - p_obs) * np.log(1.0 - qv)).sum())

    def fit_T_bernoulli(self, p_obs: np.ndarray,
                        grid=None, Q=None) -> tuple[float, float]:
        """MLE over log10 T: shared grid + golden polish. Returns (T*, nll*)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        Qc = np.clip(Q, EPS, 1.0 - EPS)
        nll_g = -(p_obs * np.log(Qc)
                  + (1.0 - p_obs) * np.log(1.0 - Qc)).sum(axis=1)
        k = int(np.argmin(nll_g))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xstar = self._golden(
            lambda g: self.nll_bernoulli(10.0 ** g, p_obs), lo, hi)
        return float(10.0 ** xstar), float(
            self.nll_bernoulli(10.0 ** xstar, p_obs))

    def fit_T_lsq(self, p_obs: np.ndarray, grid=None, Q=None) -> float:
        """Least-squares-on-p echo (never adjudicates)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        k = int(np.argmin(((Q - p_obs) ** 2).sum(axis=1)))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xstar = self._golden(
            lambda g: float(((self.q(10.0 ** g) - p_obs) ** 2).sum()),
            lo, hi)
        return float(10.0 ** xstar)

    def fit_T_kl_full(self, Lstate: np.ndarray) -> float:
        """Full-vocab KL echo: minimize KL(softmax(Lstate) || softmax(L0/T))
        (one T for the WHOLE distribution — the stronger null; never
        adjudicates)."""
        with np.errstate(over="ignore"):
            qs = np.exp(Lstate - Lstate.max(axis=1, keepdims=True))
            qs = qs / qs.sum(axis=1, keepdims=True)
        qs = np.maximum(qs, 1e-300)
        logqs = np.log(qs)

        def kl(g: float) -> float:
            T = 10.0 ** g
            with np.errstate(over="ignore"):
                z = self.d / T
                z = z - z.max(axis=1, keepdims=True)
                lq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return float((qs * (logqs - lq)).sum())

        grid = np.linspace(GRID_LO, GRID_HI, 201)
        k = int(np.argmin([kl(g) for g in grid]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        return float(10.0 ** self._golden(kl, lo, hi))

    def per_probe_T(self, p_obs: np.ndarray,
                    grid=None, Q=None) -> tuple[np.ndarray, np.ndarray]:
        """Per-probe T_i: root of q_i(T) = p_obs,i (grid + bisection).
        Returns (T_i with nan where p_obs is outside the family's reach,
        n_crossings per probe on the grid)."""
        if grid is None or Q is None:
            grid, Q = self.grid_Q()
        n = self.n
        Ti = np.full(n, np.nan)
        ncross = np.zeros(n, dtype=int)
        for i in range(n):
            col = Q[:, i]
            sgn = np.sign(col - p_obs[i])
            cross = np.nonzero(np.diff(sgn) != 0)[0]
            ncross[i] = len(cross)
            if len(cross) == 0:
                continue
            # the crossing nearest T=1 (log10 = 0): grid index k means root
            # in (grid[k], grid[k+1]]
            best = min(cross, key=lambda k: abs(grid[k]))
            a, b = 10.0 ** grid[best], 10.0 ** grid[best + 1]
            fa = col[best] - p_obs[i]
            for _ in range(80):
                m = (a * b) ** 0.5            # geometric midpoint
                fm = self.q_one(m, i) - p_obs[i]
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b = m
                if b / a - 1 < 1e-9:
                    break
            Ti[i] = (a * b) ** 0.5
        return Ti, ncross


class TwoMomentFamily:
    """The desk-only Gaussian-denominator family from committed triples.

    q_i^2m(T) = 1/[1 + (N-1) exp((mu_i - a_i)/T + sigma0_i^2/(2T^2))] with
    (mu_i - a_i) = ln((1-p0_i)/(p0_i (N-1))) - sigma0_i^2/2, from e228's
    committed t0 (p0, sigma0). Zero forwards.
    """

    def __init__(self, p0: np.ndarray, sigma0: np.ndarray, N: int):
        self.p0 = np.clip(p0.astype(np.float64), EPS, 1.0 - EPS)
        self.s0 = sigma0.astype(np.float64)
        self.N = int(N)
        self.gap = np.log((1.0 - self.p0) / (self.p0 * (self.N - 1))) \
            - self.s0 ** 2 / 2.0        # mu_i - a_i at T=1

    def q(self, T: float) -> np.ndarray:
        z = self.gap / T + (self.s0 ** 2) / (2.0 * T * T)
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + (self.N - 1) * np.exp(z))

    def nll(self, T: float, p_obs: np.ndarray) -> float:
        qv = np.clip(self.q(T), EPS, 1.0 - EPS)
        return float(-(p_obs * np.log(qv)
                       + (1.0 - p_obs) * np.log(1.0 - qv)).sum())

    def fit_T(self, p_obs: np.ndarray) -> float:
        grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
        k = int(np.argmin([self.nll(10.0 ** g, p_obs) for g in grid]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, GRID_N - 1)]
        gr = (5 ** 0.5 - 1) / 2
        a, b = lo, hi
        for _ in range(60):
            c, d_ = b - gr * (b - a), a + gr * (b - a)
            if self.nll(10.0 ** c, p_obs) < self.nll(10.0 ** d_, p_obs):
                b = d_
            else:
                a = c
            if b - a < 1e-7:
                break
        return float(10.0 ** ((a + b) / 2))

    def per_probe_T(self, p_obs: np.ndarray) -> np.ndarray:
        grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
        Q = np.array([self.q(10.0 ** g) for g in grid])       # (G, n)
        Ti = np.full(len(p_obs), np.nan)
        for i in range(len(p_obs)):
            col = Q[:, i]
            sgn = np.sign(col - p_obs[i])
            cross = np.nonzero(np.diff(sgn) != 0)[0]
            if len(cross) == 0:
                continue
            best = min(cross, key=lambda k: abs(grid[k]))
            a, b = 10.0 ** grid[best], 10.0 ** grid[best + 1]
            fa = col[best] - p_obs[i]
            for _ in range(80):
                m = (a * b) ** 0.5
                fm = self.q(m)[i] - p_obs[i]
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b = m
                if b / a - 1 < 1e-9:
                    break
            Ti[i] = (a * b) ** 0.5
        return Ti


# --------------------------------------------------- THE LOGIT RE-PROBE PASS

@torch.no_grad()
def logit_pass(net, battery: list[dict]) -> dict:
    """One forward per probe — the e182/e182c/e214 p/rank path VERBATIM plus
    THIS cell's object on the same logits: the FULL answer-position logit
    vector (fp32), for the one-time logit dump."""
    net.eval()
    rows, logits = [], []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        pv = float(p[pr["ans_id"]])
        rank = int((lg > lg[pr["ans_id"]]).sum().item())
        rows.append({
            "fact": pr["fact"], "relation": pr["relation"],
            "p": pv, "rank": rank, "top1": bool(rank == 0),
            "top5": bool(rank < 5),
        })
        logits.append(lg.float().numpy().astype(np.float32))
    ps = [r["p"] for r in rows]
    return {"probes": rows, "logits": np.stack(logits),
            "mean_p": float(sum(ps) / len(ps))}


def state_tag(wash: str, step: int) -> str:
    return "t0" if wash == "t0" else f"{wash}s{step}"


# ------------------------------------------------------------------ plots

def make_fit_plot(rd, fam_rows, verdict):
    """(a) observed-vs-model p at the adjudication states; (b) the T
    trajectory; (c) R2 vs the 0.80 bar; (d) the mean-p trajectories
    observed vs model (the one-T fit curves); (e) per-probe T_i spreads;
    (f) verdict."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))

    # (a) observed vs model p (pooled panels per state)
    ax = axes[0, 0]
    adj = [r for r in fam_rows if r["state"] in ADJ_STATES and not r["smoke"]]
    mk = {"w1": "o", "w2": "s"}
    for r in adj:
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(r["battery"]) if bb == b]
            ax.scatter(np.array(r["p_obs"])[idx], np.array(r["q"])[idx],
                       s=22, marker=mk[r["wash"]],
                       color=cols[b], alpha=0.55, edgecolor="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="y=x (perfect one-T fit)")
    ax.set_xlabel("observed p at the state")
    ax.set_ylabel("p under the one-T fit")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color="gray",
                          label="w1 (circles) / w2 (squares)")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=cols[b],
                           label=b) for b in BATTERIES]
    ax.legend(handles=handles, fontsize=6.4, loc="lower right")
    ax.set_title("(a) observed vs one-T-model p, adjudication states "
                 f"({len(adj)} states overlaid)", fontsize=9.5)

    # (b) T trajectory
    ax = axes[0, 1]
    for wash in ("w1", "w2"):
        rows = sorted((r for r in fam_rows if r["wash"] == wash),
                      key=lambda r: r["state"])
        ax.plot([r["state"] for r in rows], [r["T_mle"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=7, lw=1.8,
                color="tab:purple" if wash == "w1" else "tab:cyan",
                label=f"{wash} one-T MLE")
        ax.plot([r["state"] for r in rows], [r["T_lsq"] for r in rows],
                "^:" if wash == "w1" else "v:", ms=5, lw=1.2, color="gray",
                alpha=0.8, label="LSQ echo" if wash == "w1" else None)
        ax.plot([r["state"] for r in rows], [r["T_kl"] for r in rows],
                "d-." if wash == "w1" else "p-.", ms=5, lw=1.2,
                color="tab:olive", alpha=0.8,
                label="full-vocab KL echo" if wash == "w1" else None)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="T=1 (no decline)")
    ax.set_xlabel("wash step"); ax.set_ylabel("fitted T")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax.set_title("(b) ONE temperature per state — the trajectory", fontsize=9.5)

    # (c) R2 vs the bar
    ax = axes[0, 2]
    ax.axhspan(EXPLAINED_BAR, 1.02, color="gold", alpha=0.18)
    ax.axhline(EXPLAINED_BAR, color="k", ls="--", lw=1.2,
               label=f"the 80% bar")
    for wash in ("w1", "w2"):
        rows = sorted((r for r in fam_rows if r["wash"] == wash),
                      key=lambda r: r["state"])
        ax.plot([r["state"] for r in rows], [r["R2_pooled"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=7, lw=2.0,
                color="tab:purple" if wash == "w1" else "tab:cyan",
                label=f"{wash} pooled R2")
        for b in BATTERIES:
            ax.plot([r["state"] for r in rows],
                    [r["R2_battery"][b] for r in rows],
                    "^:" if wash == "w1" else "v:", ms=4, lw=1.0,
                    color=cols[b], alpha=0.7)
    ax.set_xlabel("wash step")
    ax.set_ylabel("fraction of p-decline variance explained\n"
                  "(bold = pooled [the bar]; faint = per battery)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(c) THE VARIANCE-EXPLAINED READ", fontsize=9.5)

    # (d) mean-p trajectories observed vs model
    ax = axes[1, 0]
    for wash in ("w1", "w2"):
        rows = sorted((r for r in fam_rows if r["wash"] == wash),
                      key=lambda r: r["state"])
        steps = [0] + [r["state"] for r in rows]
        ax.plot(steps, [rows[0]["mean_p0"]] + [r["mean_p_obs"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=7, lw=1.8, color="k",
                alpha=0.75 if wash == "w1" else 0.45,
                label=f"{wash} observed (mean p)")
        ax.plot(steps, [rows[0]["mean_p0"]] + [r["mean_q"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=5, lw=1.4,
                color="tab:purple" if wash == "w1" else "tab:cyan",
                alpha=0.9, label=f"{wash} one-T model")
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(rows[0]["battery"]) if bb == b]
            ax.plot(steps, [float(np.mean(np.array(rows[0]["p0"])[idx]))]
                    + [float(np.mean(np.array(r["p_obs"])[idx]))
                       for r in rows],
                    "^:" if wash == "w1" else "v:", ms=4, lw=1.0,
                    color=cols[b], alpha=0.65)
    ax.set_xlabel("wash step"); ax.set_ylabel("battery mean p")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.6, loc="best")
    ax.set_title("(d) THE ONE-T FIT CURVES — observed vs model means "
                 "(faint: per battery observed)", fontsize=9.5)

    # (e) per-probe T_i spread
    ax = axes[1, 1]
    data, labels, colors = [], [], []
    for r in sorted((x for x in fam_rows), key=lambda x: (x["wash"], x["state"])):
        ti = np.array(r["T_i"], dtype=float)
        ti = ti[~np.isnan(ti)]
        if len(ti):
            data.append(ti.tolist())
            labels.append(f"{r['wash']}+{r['state']}")
            colors.append("tab:purple" if r["wash"] == "w1" else "tab:cyan")
    if data:
        bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                        flierprops={"markersize": 2.5, "alpha": 0.5},
                        medianprops={"color": "k"})
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=6, rotation=30)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.45)
        for r in fam_rows:
            k = labels.index(f"{r['wash']}+{r['state']}") if \
                f"{r['wash']}+{r['state']}" in labels else None
            if k is not None:
                ax.plot([k + 1], [r["T_mle"]], "r*", ms=11, zorder=5)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_ylabel("per-probe T_i (root of q_i(T)=p_obs,i)")
    ax.set_xlabel("state (red star = the one-T MLE)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(e) how much does ONE probe constrain T? "
                 "(spread = identifiability; star = pooled MLE)", fontsize=9.5)

    # (f) verdict + the state table
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "STATE TABLE (pooled):", fontsize=8.6, va="top",
            family="monospace", weight="bold")
    y -= 0.026
    ax.text(0.02, y, "  wash step    T_MLE   R2_pool  R2_fact  R2_ctrl  "
                     "R2_near  R2_tmpl  adj", fontsize=6.6, va="top",
            family="monospace")
    y -= 0.0185
    for r in sorted(fam_rows, key=lambda x: (x["wash"], x["state"])):
        ax.text(0.02, y, f"  {r['wash']:4s} +{r['state']:<4d} {r['T_mle']:7.4f}"
                         f"  {r['R2_pooled']:+7.4f}"
                         + "".join(f" {r['R2_battery'][b]:+7.4f}"
                                   for b in BATTERIES)
                         + f"   {'YES' if r['state'] in ADJ_STATES else 'no'}",
                fontsize=6.6, va="top", family="monospace")
        y -= 0.0175
    y -= 0.012
    ax.text(0.02, y, f"E238 VERDICT: {verdict['bars']['verdict']}",
            fontsize=10.2, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.034
    for wd in textwrap.wrap(verdict["bars"]["clause"], width=88,
                            break_long_words=False)[:11]:
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.019
    y -= 0.006
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in verdict["gates_summary"].items()), fontsize=6.9,
        va="top", family="monospace")

    fig.suptitle("E238 — FQ7: THE ONE-TEMPERATURE-PER-STATE NULL — how much "
                 f"of the 54-probe belief tensor does one T(t) explain? "
                 f"-> {verdict['bars']['verdict']}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    png = rd / "temperature_fit.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_residual_plot(rd, res_read, verdict):
    """(a) residual-vs-erosion blocked scatter; (b) per-cell rho table;
    (c) anchors vs the product-family band; (d) the two-moment desk-only
    bound; (e-f) clause summaries."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))

    # (a) the blocked scatter (z within cell)
    ax = axes[0, 0]
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    for cell in res_read["cells"]:
        b = cell["battery"]
        ax.scatter(cell["erosion_z"], cell["resid_z"], s=26,
                   marker={"w1": "o", "w2": "s"}[cell["wash"]],
                   color=cols[b], alpha=0.55, edgecolor="none")
    rho = res_read["rho_pooled"]
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("committed erosion (z within battery x wash x state)")
    ax.set_ylabel("one-T residual p_obs - q(T*) (z within cell)")
    ax.set_title(f"(a) THE RESIDUAL-EROSION JOIN — blocked Spearman "
                 f"rho {rho:+.3f} (bar |rho| >= {RHO_BAR})"
                 f"{' — FIRES' if abs(rho) >= RHO_BAR else ''}", fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cols[b],
                          label=b) for b in BATTERIES]
    handles += [plt.Line2D([], [], marker="s", ls="", color="gray",
                           label="w2 (squares)")]
    ax.legend(handles=handles, fontsize=6.6, loc="best")
    ax.grid(alpha=0.25)

    # (b) per-cell rho table
    ax = axes[0, 1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "PER-CELL residual-vs-erosion rhos (co-report):",
            fontsize=8.6, va="top", family="monospace", weight="bold")
    y -= 0.024
    ax.text(0.02, y, "  battery wash  state      rho    tau_b     n",
            fontsize=6.9, va="top", family="monospace")
    y -= 0.0185
    for c in res_read["cells"]:
        ax.text(0.02, y, f"  {c['battery']:7s} {c['wash']:4s} +{c['state']:<4d}"
                         f" {c['rho']:+7.3f} {c['tau']:+7.3f} {c['n']:5d}",
                fontsize=6.9, va="top", family="monospace")
        y -= 0.0175
    y -= 0.01
    ax.text(0.02, y, f"  pooled (z-blocked): rho {rho:+.3f} | "
                     f"tau {res_read['tau_pooled']:+.3f} | "
                     f"near-excl echo {res_read['rho_no_near']:+.3f}",
            fontsize=6.9, va="top", family="monospace", weight="bold")

    # (c) anchors vs band
    ax = axes[0, 2]
    ax.axhline(0, color="k", lw=0.8)
    band_cells = res_read["anchor_cells"]
    xs = np.arange(len(band_cells))
    for k, cellname in enumerate(band_cells):
        cell = band_cells[cellname]
        ax.scatter([k] * len(cell["band"]), cell["band"], s=26,
                   color="tab:blue", alpha=0.5, edgecolor="none")
        ax.plot([k], [cell["gmail"]], "o", ms=11, color="tab:green",
                markeredgecolor="k", zorder=5)
        ax.plot([k], [cell["iphone"]], "X", ms=11, color="tab:red",
                markeredgecolor="k", zorder=5)
    ax.plot([], [], "o", ms=9, color="tab:green", label="Gmail (the holder)")
    ax.plot([], [], "X", ms=9, color="tab:red", label="iPhone (the eaten)")
    ax.plot([], [], "o", ms=6, color="tab:blue", alpha=0.6,
            label="ctrl/make band (the other five)")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(band_cells), fontsize=7, rotation=20)
    ax.set_ylabel("one-T residual (p_obs - q(T*))")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.8, loc="best")
    zmax = max(abs(res_read["z_anchor"]["Gmail"]), abs(
        res_read["z_anchor"]["iPhone"]))
    ax.set_title(f"(c) THE ANCHORS vs THE BAND — z(Gmail) "
                 f"{res_read['z_anchor']['Gmail']:+.2f}, z(iPhone) "
                 f"{res_read['z_anchor']['iPhone']:+.2f} "
                 f"(bar |z| >= {Z_BAR}){' — FIRES' if zmax >= Z_BAR else ''}",
                 fontsize=9.2)

    # (d) two-moment desk-only bound
    ax = axes[1, 0]
    if res_read["two_moment"] is not None:
        tm = res_read["two_moment"]
        for wash in ("w1", "w2"):
            rows = sorted((r for r in tm if r["wash"] == wash),
                          key=lambda r: r["state"])
            ax.plot([r["state"] for r in rows], [r["T_full"] for r in rows],
                    "o-", ms=7, lw=1.8, color="tab:purple"
                    if wash == "w1" else "tab:cyan",
                    label=f"{wash} T (full logits)")
            ax.plot([r["state"] for r in rows], [r["T_2m"] for r in rows],
                    "^--" if wash == "w1" else "v--", ms=7, lw=1.4,
                    color="tab:purple" if wash == "w1" else "tab:cyan",
                    alpha=0.55, label=f"{wash} T (desk-only two-moment)")
        ax.axhline(1.0, color="k", ls="--", lw=1.0)
        ax.set_xlabel("wash step"); ax.set_ylabel("fitted T")
        ax.grid(alpha=0.25); ax.legend(fontsize=6.8, loc="best")
        r2f = [r["R2_full"] for r in tm]
        r2m = [r["R2_2m"] for r in tm]
        ax.set_title(f"(d) THE DESK-ONLY BOUND — T_2m vs T_full; R2_2m "
                     f"{np.nanmin(r2m):+.3f}..{np.nanmax(r2m):+.3f} vs "
                     f"R2_full {np.nanmin(r2f):+.3f}..{np.nanmax(r2f):+.3f}",
                     fontsize=9.0)

    # (e) per-probe T_i: full vs two-moment spread (the constraint read)
    ax = axes[1, 1]
    if res_read["two_moment"] is not None:
        for r in res_read["two_moment"]:
            if r["spread_iqr_med_full"] is None \
                    or r["spread_iqr_med_2m"] is None:
                continue
            ax.plot([r["state"] + (0 if r["wash"] == "w1" else 6)],
                    [r["spread_iqr_med_full"]], "o", ms=8,
                    color="tab:purple" if r["wash"] == "w1" else "tab:cyan")
            ax.plot([r["state"] + (1 if r["wash"] == "w1" else 7)],
                    [r["spread_iqr_med_2m"]], "^", ms=8,
                    color="tab:purple" if r["wash"] == "w1" else "tab:cyan",
                    alpha=0.55)
        ax.plot([], [], "o", color="gray", label="full logits")
        ax.plot([], [], "^", color="gray", alpha=0.55,
                label="desk-only two-moment")
        ax.set_xlabel("wash step (circles full / triangles desk-only, "
                      "w2 offset +6)")
        ax.set_ylabel("per-probe T_i spread (IQR / median)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
        ax.set_title("(e) how much does ONE probe's record constrain T? "
                     "(the registered two-moment check)", fontsize=9.5)

    # (f) clause summary
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE STRUCTURED CLAUSES:", fontsize=8.6, va="top",
            family="monospace", weight="bold")
    y -= 0.03
    for line in res_read["clause_lines"]:
        for wd in textwrap.wrap(line, width=86, break_long_words=False)[:4]:
            ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top",
                    family="monospace")
            y -= 0.017
        y -= 0.006
    y -= 0.01
    ax.text(0.02, y, "E238 VERDICT: " + verdict["bars"]["verdict"],
            fontsize=10.0, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(verdict["bars"]["clause"], width=86,
                            break_long_words=False)[:8]:
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.019

    fig.suptitle("E238 — the residual structure read: is what one temperature "
                 "LEAVES BEHIND structured? (erosion order + the Gmail/iPhone "
                 "split; the desk-only bound)", fontsize=11.0)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    png = rd / "residual_structure.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E238 — FQ7, THE ONE-TEMPERATURE-PER-STATE NULL (smoke={SMOKE}) "
        f"-> {rd}")

    metrics = {
        "experiment": "e238_temperature_null",
        "phase": ("eval-only desk+eval on the committed two-wash 124M archive "
                  "(FQ7, the ideator's Rank 2, urgency-flagged: must land "
                  "before e234's adjudication is read)"),
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("how much of the 54-probe belief tensor does ONE scalar "
                     "temperature per state explain? uniform thermal erosion, "
                     "or structured residuals (the erosion order + the "
                     "Gmail/iPhone split) = the third dimension's p-side seat?"),
        "builds_on": ["FQ7 / review_ideator_2026-10-04 (the Rank-2 question)",
                      "T212 / e232 (the temperature derivation that killed "
                      "the MARGIN side — the p side is the open half)",
                      "T209 / e230 (zombie decisions: beliefs halve while "
                      "the argmax stands — the p-side narratives this "
                      "constrains)",
                      "T211 / e227 (the cross-scale zombie structure)",
                      "T187 / e214 (the erosion order + the committed "
                      "per-probe records = the y-side)",
                      "T185 / e213 (the two-wash archive)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "machinery)"],
        "whats_new": ["the one-T-per-state MLE fit of the whole 54-probe "
                      "belief tensor (the p-side temperature null)",
                      "the decline-variance-explained read (pooled + per "
                      "battery, vs the pre-registered 80% bar)",
                      "the blocked residual-vs-erosion-order join",
                      "the Gmail/iPhone anchor-vs-band residual separation",
                      "the two-moment desk-only bound (the registered check)",
                      "the full answer-position logit dumps (reusable "
                      "artifact, sha-recorded)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    for p in (E214_METRICS, E214_JOURNAL, E228_JOURNAL, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e214j = json.loads(E214_JOURNAL.read_text(encoding="utf-8"))
    e228j = json.loads(E228_JOURNAL.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    y_t0 = [r for r in e214j["states"] if r["wash"] == "t0"][0]
    y_st = {(r["wash"], r["step"]): r for r in e214j["states"]
            if r["wash"] in ("w1", "w2")}
    t0_228 = [r for r in e228j["states"] if r["wash"] == "t0"][0]
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    log(f"records: e214 journal states {sorted(k[0] + str(k[1]) for k in y_st)}; "
        f"e228 journal t0 triples loaded")

    # ------------------------------------------ P1 G_STATES part A: inventory
    def inv(p: Path):
        return {"path": str(p), "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                    p.stat().st_mtime)) if p.exists() else None}
    w1_inv = {str(s): inv(p) for s, p in W1_ARCH.items()}
    w2_inv = {str(s): inv(p) for s, p in W2_ARCH.items()}
    c2_inv = json.loads((common.REPO / "runs" / "e182c2" / "metrics.json")
                        .read_text(encoding="utf-8")) \
        .get("inventory", {}).get("files", {})
    w1_match = {str(s): (bool(c2_inv.get(str(s)))
                         and c2_inv[str(s)]["size_bytes"]
                         == w1_inv[str(s)]["size_bytes"]
                         and c2_inv[str(s)]["mtime"]
                         == w1_inv[str(s)]["mtime"])
                for s in W1_ARCH}
    all_exist = all(v["exists"] for v in
                    list(w1_inv.values()) + list(w2_inv.values()))
    G_STATES_A = {
        "wash1_files": w1_inv, "wash2_files": w2_inv,
        "wash1_crosscheck_vs_e182c2_inventory": w1_match,
        "note": "wash 1 = e182c_s* (seed 18202, CPU fp32 replay, incl. the "
                "+2 wash-1-only state); wash 2 = e182c2_fresh_s* (seed "
                "20261002, GPU fp32); sizes/mtimes must match e182c2's "
                "committed inventory for the wash-1 set",
    }
    log(f"G_STATES inventory: wash1 {sorted(w1_inv)} wash2 {sorted(w2_inv)} "
        f"all on disk = {all_exist}")
    metrics["inventory"] = G_STATES_A
    write_metrics("PARTIAL: records read; inventory taken")

    # ---------------------------------------------------- P2 the organism
    load_checks = [cpu_load_check("launch")]
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = org_meta
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"
    metrics["size_gate"] = G_SIZE

    # ------------------------------------------- P3 the frozen corpus (G_CORPUS)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in e1.POOLS
                     for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    base_cand = e1.probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    kept_facts, _sel = e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "lines_total": [G_STR["lines_total"], e_str["lines_total"]],
        "lines_dropped": [G_STR["lines_dropped"], e_str["lines_dropped"]],
        "chars_after": [corpus_stats["chars_after"], e_corp["chars_after"]],
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "bank_windows": [corpus_stats["bank_windows"],
                         e_corp["bank_windows"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN — it feeds the batteries' "
                "contamination scans only; no wash is run here",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"]
        and corpus_stats["bank_windows"] == e_corp["bank_windows"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens; e182 "
        f"{e_corp['tokens_after']})")
    assert G_CORPUS["pass"] or SMOKE, f"corpus rebuild diverged: {G_CORPUS}"
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS}

    # --------------------------------- P4 the four batteries, VERBATIM (G_BATT)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]

    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]

    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}
    log("batteries rebuilt VERBATIM: " + " | ".join(
        f"{b} n={len(bats[b])}" for b in BATTERIES))

    # the flat 54-probe register (order: fact, ctrl, near, tmpl)
    names, rels, bats_of, ans_ids_arr = [], [], [], []
    for b in BATTERIES:
        for pr in bats[b]:
            names.append(pr["fact"]); rels.append(pr["relation"])
            bats_of.append(b); ans_ids_arr.append(int(pr["ans_id"]))
    n_probes = len(names)
    V = int(net0.config.vocab_size)
    log(f"the belief tensor: {n_probes} probes, vocab {V}")

    # ------------------------- P5 THE ONE-TIME LOGIT RE-PROBE (t0 + states)
    # journal: per-state light records; npz: the full logit dumps (on disk)
    ljour = {}
    if jp.exists():
        try:
            ljour = json.loads(
                jp.read_text(encoding="utf-8")).get("logit_states", {})
            log(f"journal: {len(ljour)} logit states restored")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); recomputing")
            ljour = {}

    def logit_all(evl):
        return {b: logit_pass(evl, bats[b]) for b in BATTERIES}

    def dump_and_journal(wash, step, evl):
        """One state = one eval burst: forward all 54, dump fp32 npz, journal
        the light record. Returns the per-state record."""
        tag = state_tag(wash, step)
        npz = rd / f"logits_{tag}.npz"
        cur = logit_all(evl)
        L = np.concatenate([cur[b]["logits"] for b in BATTERIES], axis=0)
        np.savez_compressed(npz, names=np.array(names),
                            battery=np.array(bats_of),
                            relation=np.array(rels),
                            ans_ids=np.array(ans_ids_arr, dtype=np.int64),
                            logits=L)
        rec = {
            "wash": wash, "step": step,
            "npz": str(npz), "npz_sha256_16": sha256_of(npz),
            "logits_shape": list(L.shape), "dtype": "float32",
            "p": [r["p"] for b in BATTERIES for r in cur[b]["probes"]],
            "mean_p": float(np.mean([r["p"] for b in BATTERIES
                                     for r in cur[b]["probes"]])),
            "dp_vs_e214": None,      # filled by the certification pass
        }
        return rec

    # t=0 — ONE read of the shared pristine organism (e214's t0_shared)
    if "t0" not in ljour or not (rd / "logits_t0.npz").exists():
        load_checks.append(cpu_load_check("t0"))
        ljour["t0"] = dump_and_journal("t0", 0, net0)
        jp.write_text(json.dumps({"logit_states": ljour}, indent=1),
                      encoding="utf-8")
        write_metrics("PARTIAL: t=0 logit dump done")
    else:
        log("t=0 dump restored from journal/npz")

    # the state loop
    ck_map = {"w1": W1_ARCH, "w2": W2_ARCH}
    state_list = ([("w1", s_) for s_ in SHARED_STATES + W1_ONLY_STATES]
                  + [("w2", s_) for s_ in SHARED_STATES])
    for wash, s_ in state_list:
        tag = state_tag(wash, s_)
        if tag in ljour and (rd / f"logits_{tag}.npz").exists():
            continue
        load_checks.append(cpu_load_check(tag))
        f = ck_map[wash][s_]
        sd = torch.load(f, map_location=CPU, weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        ljour[tag] = dump_and_journal(wash, s_, evl)
        del evl, sd
        jp.write_text(json.dumps({"logit_states": ljour}, indent=1),
                      encoding="utf-8")
        log(f"  {wash.upper()} +{s_} dumped: mean p "
            f"{ljour[tag]['mean_p']:.4f}")
        write_metrics(f"PARTIAL: logit dumps through {wash} +{s_}")

    # G_BATT + G_STATES: certification of every dumped state vs e214
    gb = {}
    t0p = np.array(ljour["t0"]["p"])
    for b in BATTERIES:
        idx = [i for i, bb in enumerate(bats_of) if bb == b]
        ref = y_t0[b]["probes"]
        gb[b] = {
            "kept_set_equal": bool({pr["fact"] for pr in bats[b]}
                                   == set(ref)),
            "n": len(ref),
            "max_dp_vs_e214_t0": max(abs(t0p[i] - ref[names[i]]["p"])
                                     for i in idx if names[i] in ref),
        }
    G_BATT = {**gb, "tol_per_probe_dp": TOL_PROBE_DP,
              "note": "the four batteries are the phase-1/phase-2 pools "
                      "VERBATIM (module import); my t=0 logit-pass p must "
                      "reproduce e214's committed journal t0 — certifying "
                      "the forward the logits ride on"}
    G_BATT["pass"] = bool(all(G_BATT[b]["kept_set_equal"]
                              and G_BATT[b]["max_dp_vs_e214_t0"]
                              <= TOL_PROBE_DP for b in BATTERIES)) \
        if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} " + " | ".join(
        f"{b}: dp {G_BATT[b]['max_dp_vs_e214_t0']:.2e}" for b in BATTERIES))

    all_dps = []
    for tag, rec in ljour.items():
        if rec["wash"] == "t0":
            rec["dp_vs_e214"] = 0.0
            continue
        ref = y_st[(rec["wash"], rec["step"])]
        dpb = {}
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(bats_of) if bb == b]
            dp = max(abs(rec["p"][i] - ref[b]["probes"][names[i]]["p"])
                     for i in idx)
            dpb[b] = dp
            all_dps.append(dp)
        rec["dp_vs_e214"] = max(dpb.values())
        rec["dp_vs_e214_by_battery"] = dpb
    G_STATES = {**G_STATES_A, "all_exist": all_exist,
                "reprobe_max_dp": max(all_dps) if all_dps else None,
                "n_reprobe_checks": len(all_dps),
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_note": "every battery's p (from the SAME forward "
                                "as the logit dump) compared per-probe to "
                                "e214's committed journal records on every "
                                "LOADED state of BOTH archives — this "
                                "certifies the dumped logits"}
    G_STATES["pass"] = bool(all_exist and (SMOKE or (all_dps
                                                     and max(all_dps)
                                                     <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "load_checks": len(load_checks), "gpu_calls": 0,
             "burst": "one state = one eval burst",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_STATES re-probe: {len(all_dps)} battery-state checks, max dp "
        f"{max(all_dps) if all_dps else 0:.8f} (tol {TOL_STATE_DP})")
    write_metrics("PARTIAL: logit dumps certified — the fit next")

    # ------------------------------------------------------ P6 THE PRIMARY FIT
    L0 = np.load(rd / "logits_t0.npz")["logits"]
    p0 = np.array(ljour["t0"]["p"])
    fam = TempFamily(L0, np.array(ans_ids_arr))
    fit_rows = []
    for wash, s_ in state_list:
        tag = state_tag(wash, s_)
        rec = ljour[tag]
        p_obs = np.array(rec["p"])
        grid, Q = fam.grid_Q()                       # the shared grid pass
        T_mle, nll = fam.fit_T_bernoulli(p_obs, grid, Q)
        q = fam.q(T_mle)
        T_lsq = fam.fit_T_lsq(p_obs, grid, Q)
        Ls = np.load(rd / f"logits_{tag}.npz")["logits"]
        T_kl = fam.fit_T_kl_full(Ls)
        Ti, ncross = fam.per_probe_T(p_obs, grid, Q)
        resid = p_obs - q
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((p_obs - p0) ** 2).sum())
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        R2b = {}
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(bats_of) if bb == b]
            sr = float((resid[idx] ** 2).sum())
            st = float(((p_obs[idx] - p0[idx]) ** 2).sum())
            R2b[b] = 1.0 - sr / st if st > 0 else float("nan")
        ti_ok = Ti[~np.isnan(Ti)]
        fit_rows.append({
            "wash": wash, "state": s_, "smoke": SMOKE,
            "T_mle": T_mle, "nll_at_T": nll, "nll_at_T1":
                fam.nll_bernoulli(1.0, p_obs),
            "T_lsq": T_lsq, "T_kl": T_kl,
            "mean_p_obs": float(p_obs.mean()), "mean_q": float(q.mean()),
            "mean_p0": float(p0.mean()),
            "ss_res": ss_res, "ss_tot_decline": ss_tot,
            "R2_pooled": R2, "R2_battery": R2b,
            "mean_abs_resid": float(np.abs(resid).mean()),
            "max_abs_resid": float(np.abs(resid).max()),
            "T_i": Ti.tolist(), "T_i_multi_crossings":
                int((ncross > 1).sum()),
            "T_i_outside_range": int(np.isnan(Ti).sum()),
            "T_i_iqr_over_median": float(
                (np.percentile(ti_ok, 75) - np.percentile(ti_ok, 25))
                / np.median(ti_ok)) if len(ti_ok) else None,
            "q": q.tolist(), "resid": resid.tolist(),
            "p0": p0.tolist(), "p_obs": p_obs.tolist(),
            "battery": bats_of,
        })
        log(f"  FIT {wash}+{s_}: T* {T_mle:.4f} (LSQ {T_lsq:.4f}, KL "
            f"{T_kl:.4f}) — pooled R2 {R2:+.4f}; mean |resid| "
            f"{np.abs(resid).mean():.4f}")
        write_metrics(f"PARTIAL: one-T fit done for {wash} +{s_}")
    metrics["fit_rows"] = [{k: v for k, v in r.items()
                            if k not in ("q", "resid", "p0", "p_obs",
                                         "T_i", "battery")}
                           for r in fit_rows]

    # ---------------------------------- P7 the residual-structure reads
    # the committed erosion from e214's records
    def committed_p(wash, step):
        rec = y_st[(wash, step)]
        return np.array([rec[b]["probes"][names[i]]["p"]
                         for i, b in enumerate(bats_of)])

    res_cells = []
    zr_pool, ze_pool = [], []
    anchor_cells = {}
    anchors_idx = [i for i, nm in enumerate(names)
                   if any(a in nm for a in ANCHORS)
                   and bats_of[i] == "ctrl" and rels[i] == ANCHOR_RELATION]
    band_idx = [i for i in range(n_probes)
                if bats_of[i] == "ctrl" and rels[i] == ANCHOR_RELATION
                and i not in anchors_idx]
    for wash in ("w1", "w2"):
        for s_ in DEEP_STATES:
            if s_ not in SHARED_STATES:
                continue
            row = next(r for r in fit_rows
                       if r["wash"] == wash and r["state"] == s_)
            resid = np.array(row["resid"])
            e_com = 1.0 - committed_p(wash, s_) / p0       # committed
            for b in BATTERIES:
                idx = [i for i, bb in enumerate(bats_of) if bb == b]
                rho, _t = spearman(resid[idx].tolist(), e_com[idx].tolist())
                tau = kendall_tau_b(resid[idx].tolist(), e_com[idx].tolist())
                # POST-HOC shared-variable calibration (added before the
                # final re-run, after seeing rho -0.898; NEVER adjudicated,
                # labeled post-hoc): both r and e contain p_obs, so a
                # regression-to-the-mean artifact could make rho negative
                # even under unstructured misses. The calibration: does the
                # MODEL'S OWN q-ordering (t=0 logits only, NO state p on
                # that side) already predict the erosion order? If
                # rho_model ~ rho_join, the join adds nothing beyond the
                # model's own ordering; if rho_model is weak, the join's
                # information lives in the MISS.
                qv = row["q"]
                rho_model, _tm = spearman(
                    [-qv[i] for i in idx], e_com[idx].tolist())
                res_cells.append({"battery": b, "wash": wash, "state": s_,
                                  "n": len(idx), "rho": rho, "tau": tau,
                                  "rho_model_posthoc": rho_model,
                                  "erosion_z": zscore(e_com[idx]).tolist(),
                                  "resid_z": zscore(resid[idx]).tolist()})
                zr_pool.extend(zscore(resid[idx]).tolist())
                ze_pool.extend(zscore(e_com[idx]).tolist())
            tag = f"{wash}+{s_}"
            anchor_cells[tag] = {
                "gmail": float(resid[names.index(
                    next(nm for nm in names if "Gmail" in nm))]),
                "iphone": float(resid[names.index(
                    next(nm for nm in names if "iPhone" in nm))]),
                "band": resid[band_idx].tolist(),
            }
    rho_pooled, _ = spearman(zr_pool, ze_pool)
    tau_pooled = kendall_tau_b(zr_pool, ze_pool)
    # near-excluded sensitivity echo
    zr_nn, ze_nn = [], []
    for wash in ("w1", "w2"):
        for s_ in DEEP_STATES:
            if s_ not in SHARED_STATES:
                continue
            row = next(r for r in fit_rows
                       if r["wash"] == wash and r["state"] == s_)
            resid = np.array(row["resid"])
            e_com = 1.0 - committed_p(wash, s_) / p0
            for b in ("fact", "ctrl", "tmpl"):
                idx = [i for i, bb in enumerate(bats_of) if bb == b]
                zr_nn.extend(zscore(resid[idx]).tolist())
                ze_nn.extend(zscore(e_com[idx]).tolist())
    rho_no_near, _ = spearman(zr_nn, ze_nn)
    # the post-hoc calibration's pooled blocked version (never adjudicated)
    zq_pool = []
    for wash in ("w1", "w2"):
        for s_ in DEEP_STATES:
            if s_ not in SHARED_STATES:
                continue
            row = next(r for r in fit_rows
                       if r["wash"] == wash and r["state"] == s_)
            qv = np.array(row["q"])
            e_com = 1.0 - committed_p(wash, s_) / p0
            for b in BATTERIES:
                idx = [i for i, bb in enumerate(bats_of) if bb == b]
                zq_pool.extend(zscore(-qv[idx]).tolist())
    rho_model_pooled, _ = spearman(zq_pool, ze_pool)

    # the anchor z's (mean over the 4 deep cells, vs the band's spread)
    if anchor_cells:
        g_res = np.array([anchor_cells[t]["gmail"] for t in anchor_cells])
        i_res = np.array([anchor_cells[t]["iphone"] for t in anchor_cells])
        band_res = np.array([v for t in anchor_cells
                             for v in anchor_cells[t]["band"]])
        band_sd = float(band_res.std(ddof=1))
        z_anchor = {
            "Gmail": float((g_res.mean() - band_res.mean()) / band_sd),
            "iPhone": float((i_res.mean() - band_res.mean()) / band_sd),
        }
        split_z = float((g_res.mean() - i_res.mean())
                        / (g_res.std(ddof=1) / 2 + i_res.std(ddof=1) / 2
                           + band_sd / np.sqrt(len(band_idx))))
    else:                     # smoke (no deep cells read)
        band_sd, split_z = None, None
        z_anchor = {"Gmail": None, "iPhone": None}
    res_read = {
        "cells": res_cells, "rho_pooled": rho_pooled,
        "tau_pooled": tau_pooled, "rho_no_near": rho_no_near,
        "anchor_cells": anchor_cells, "z_anchor": z_anchor,
        "split_z": split_z, "band_sd": band_sd,
        "anchors": [names[i] for i in anchors_idx],
        "band": [names[i] for i in band_idx],
    }

    # -------------------------------------- P8 the two-moment desk-only bound
    # e228's committed t0 triples -> (p0, sigma0); e214's committed p's
    p0_228 = np.array([r["p"] for b in BATTERIES
                       for r in t0_228[b]["probes"]])
    s0_228 = np.array([r["sigma"] for b in BATTERIES
                       for r in t0_228[b]["probes"]])
    nm_228 = [r["fact"] for b in BATTERIES for r in t0_228[b]["probes"]]
    assert nm_228 == names, "e228 journal probe order diverged from the " \
                            "rebuilt batteries"
    tm_fam = TwoMomentFamily(p0_228, s0_228, V)
    tm_rows = []
    for wash, s_ in state_list:
        row = next(r for r in fit_rows
                   if r["wash"] == wash and r["state"] == s_)
        p_com = committed_p(wash, s_)                     # COMMITTED p's
        T_2m = tm_fam.fit_T(p_com)
        q_2m = tm_fam.q(T_2m)
        R2_2m = 1.0 - float(((p_com - q_2m) ** 2).sum()) \
            / float(((p_com - p0) ** 2).sum())
        Ti_2m = tm_fam.per_probe_T(p_com)
        ti_f = np.array(row["T_i"]); ti_m = Ti_2m
        tm_rows.append({
            "wash": wash, "state": s_,
            "T_2m": T_2m, "T_full": row["T_mle"],
            "rel_T_diff": abs(T_2m - row["T_mle"]) / row["T_mle"],
            "R2_2m": R2_2m, "R2_full": row["R2_pooled"],
            "T_i_2m": ti_m.tolist(), "T_i_full": ti_f.tolist(),
            "spread_iqr_med_full": row["T_i_iqr_over_median"],
            "spread_iqr_med_2m": float(
                (np.percentile(ti_m[~np.isnan(ti_m)], 75)
                 - np.percentile(ti_m[~np.isnan(ti_m)], 25))
                / np.median(ti_m[~np.isnan(ti_m)]))
            if (~np.isnan(ti_m)).sum() else None,
            "n_outside_family_2m": int(np.isnan(ti_m).sum()),
        })
        log(f"  TWO-MOMENT {wash}+{s_}: T_2m {T_2m:.4f} vs full "
            f"{row['T_mle']:.4f} (rel {abs(T_2m - row['T_mle']) / row['T_mle']:.3f}); "
            f"R2_2m {R2_2m:+.4f} vs {row['R2_pooled']:+.4f}")
    # the margin sanity (Gaussian-tail vs observed top-gap, t=0)
    marg0 = np.array([r.get("margin_raw") for b in BATTERIES
                      for r in t0_228[b]["probes"]], dtype=float)
    pred_gap = s0_228 * np.sqrt(2.0 * np.log(V))
    margin_sanity = {
        "observed_margin_raw_mean": float(np.nanmean(marg0)),
        "gaussian_tail_pred_mean": float(pred_gap.mean()),
        "ratio": float(np.nanmean(marg0) / pred_gap.mean()),
        "note": "the two-moment family's denominator replaces the true "
                "top-end structure with a Gaussian tail; this ratio prices "
                "that crudeness (the registered check's honesty clause)",
    }
    res_read["two_moment"] = tm_rows
    res_read["margin_sanity"] = margin_sanity
    write_metrics("PARTIAL: residual structure + two-moment bound computed")

    # ------------------------------------------- P9 the adjudication (frozen)
    adj_rows = [r for r in fit_rows
                if r["state"] in ADJ_STATES and r["wash"] in ("w1", "w2")]
    gates_ok = bool(G_STATES["pass"] and G_BATT["pass"]
                    and G_CORPUS["pass"] and G_ENV["pass"])
    all_explained = bool(adj_rows) and all(
        r["R2_pooled"] >= EXPLAINED_BAR for r in adj_rows)
    erosion_fires = bool(rho_pooled is not None
                         and abs(rho_pooled) >= RHO_BAR)
    zv = [abs(v) for v in z_anchor.values() if v is not None]
    zmax = max(zv) if zv else float("nan")
    anchor_fires = bool(zv and zmax >= Z_BAR)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_STATES", G_STATES["pass"]),
                           ("G_BATT", G_BATT["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif all_explained:
        verdict = "FLATTENING-DOMINANT"
        clause = (">= 80% of pooled p-decline variance explained by the "
                  "one-T family at every adjudication state ("
                  + "; ".join(f"{r['wash']}+{r['state']} R2 "
                              f"{r['R2_pooled']:+.4f} at T {r['T_mle']:.3f}"
                              for r in adj_rows)
                  + ") — the p-side is uniform thermal erosion; the "
                  "differential cut (the residual) is the named next "
                  "instrument")
    elif erosion_fires or anchor_fires:
        verdict = "STRUCTURED"
        clause = ("< 80% explained (pooled R2: "
                  + "; ".join(f"{r['wash']}+{r['state']} "
                              f"{r['R2_pooled']:+.4f}"
                              for r in adj_rows)
                  + f") AND the residuals are structured — "
                  + (f"erosion-order clause: blocked Spearman rho "
                     f"{rho_pooled:+.3f} (bar |rho| >= {RHO_BAR}); "
                     if erosion_fires else "")
                  + (f"anchor clause: z(Gmail) {z_anchor['Gmail']:+.2f}, "
                     f"z(iPhone) {z_anchor['iPhone']:+.2f} "
                     f"(bar |z| >= {Z_BAR})"
                     if anchor_fires else "")
                  + " — the third dimension has a p-side seat; the critic's "
                  "confound discharged")
    else:
        verdict = "MIXED"
        clause = ("between — the tables verbatim, both reads: < 80% "
                  f"explained (pooled R2: "
                  + "; ".join(f"{r['wash']}+{r['state']} "
                              f"{r['R2_pooled']:+.4f}" for r in adj_rows)
                  + f") but the residual clauses did not fire (blocked "
                  f"rho {rho_pooled:+.3f} vs bar {RHO_BAR}; anchor z's "
                  f"{z_anchor['Gmail']:+.2f}/{z_anchor['iPhone']:+.2f} vs "
                  f"bar {Z_BAR})")

    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_ENV": G_ENV["pass"]}
    res_read["clause_lines"] = [
        f"EROSION CLAUSE: blocked Spearman(residual, committed erosion) = "
        f"{_f(rho_pooled)} (tau-b {_f(tau_pooled)}; near-excluded echo "
        f"{_f(rho_no_near)}) — bar |rho| >= {RHO_BAR}: "
        f"{'FIRES' if erosion_fires else 'does not fire'}",
        f"ANCHOR CLAUSE: z(Gmail) {_f(z_anchor['Gmail'], '+.2f')}, "
        f"z(iPhone) {_f(z_anchor['iPhone'], '+.2f')}, split z "
        f"{_f(split_z, '+.2f')}, band sd {_f(band_sd, '.4f')} — bar "
        f"|z| >= {Z_BAR}: {'FIRES' if anchor_fires else 'does not fire'}",
        f"EXPLAINED FRACTION: pooled R2 at the adjudication states: "
        + ", ".join(f"{r['wash']}+{r['state']} {r['R2_pooled']:+.4f}"
                    for r in adj_rows)
        + f" — bar >= {EXPLAINED_BAR}: "
        f"{'ALL PASS' if all_explained else 'NOT all pass'}",
        f"TWO-MOMENT BOUND: worst rel T gap "
        f"{max(r['rel_T_diff'] for r in tm_rows):.3f}; worst R2 gap "
        f"{max(abs(r['R2_2m'] - r['R2_full']) for r in tm_rows):.4f} "
        f"(the desk-only variant's honest bound)",
    ]
    adj = {
        "bars": {
            "FLATTENING_DOMINANT": bool(all_explained and gates_ok),
            "STRUCTURED": bool((not all_explained)
                               and (erosion_fires or anchor_fires)
                               and gates_ok),
            "erosion_clause_fired": erosion_fires,
            "anchor_clause_fired": anchor_fires,
            "verdict": verdict, "clause": clause,
            "order": "FLATTENING-DOMINANT -> STRUCTURED -> MIXED (gated on "
                     "G_STATES/G_BATT/G_CORPUS/G_ENV)",
        },
        "bar_constants": {"EXPLAINED_BAR": EXPLAINED_BAR,
                          "RHO_BAR": RHO_BAR, "Z_BAR": Z_BAR,
                          "ADJ_STATES": list(ADJ_STATES),
                          "DEEP_STATES": list(DEEP_STATES)},
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E238 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")
    log(f"  structure clauses: erosion {'FIRES' if erosion_fires else 'no'} "
        f"(rho {_f(rho_pooled)}); anchors {'FIRES' if anchor_fires else 'no'} "
        f"(z {z_anchor})")

    # --------------------------------------------- P10 honesty + close
    metrics["residual_structure"] = {
        "cells": res_cells, "rho_pooled_blocked": rho_pooled,
        "tau_pooled_blocked": tau_pooled, "rho_no_near_echo": rho_no_near,
        "rho_model_pooled_posthoc": rho_model_pooled,
        "z_anchor": z_anchor, "anchor_split_z": split_z,
        "anchors": res_read["anchors"], "band": res_read["band"],
        "band_sd": band_sd,
    }
    metrics["two_moment_bound"] = {
        "rows": [{k: v for k, v in r.items()
                  if k not in ("T_i_2m", "T_i_full")} for r in tm_rows],
        "margin_sanity": margin_sanity,
        "note": "the registered desk-only bound: the Gaussian-denominator "
                "family from e228's committed (p0, sigma0) triples fit "
                "against e214's committed p's — what the (p, margin_raw, "
                "sigma) records alone would have said; never adjudicates",
    }
    metrics["honesty_reflex"] = {
        "lens_not_mechanism": "a good one-T fit DESCRIBES the p-side as "
                              "uniform flattening; it does not assert the "
                              "wash implements a temperature (T212's "
                              "derivation already showed the margin side "
                              "cannot be thermal — both can be true only "
                              "if the fit is partial)",
        "likelihood_choice": "the Bernoulli soft-target MLE is one natural "
                             "choice; the LSQ echo prices it (the T's agree "
                             "to the printed precision when the read is "
                             "robust)",
        "pooled_aggregation": "the blocked pooled Spearman is one "
                              "pre-registered aggregation of 12 per-cell "
                              "joins; the per-cell table rides along and "
                              "never adjudicates; near (n=3) contributes "
                              "quantized ranks, flagged coarse with a "
                              "near-excluded echo",
        "shared_variable_caveat": "the residual-erosion join shares p_obs "
                                  "between both sides (residual = p_obs - "
                                  "q; erosion = 1 - p_obs/p0), so a "
                                  "regression-to-the-mean artifact could "
                                  "bend rho negative even under "
                                  "unstructured misses of sufficient "
                                  "magnitude; the POST-HOC calibration "
                                  "(rho_model, added after seeing rho "
                                  "-0.898, before the final re-run, never "
                                  "adjudicated) prices this: it asks "
                                  "whether the model's OWN q-ordering "
                                  "(t=0 logits only) predicts the erosion "
                                  "order — the join's information is the "
                                  "MISS only insofar as |rho_join| >> "
                                  "|rho_model|; the verdict's load also "
                                  "rests on the miss SCALE (mean |resid| "
                                  "~0.19-0.22 at +80, roughly half the "
                                  "pooled decline) and the anchor "
                                  "separation, neither of which is "
                                  "reachable by the artifact",
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "the archive's device asymmetry, inherited, disclosed",
        "one_organism": "ONE organism (the shared pristine 124M); the belief "
                        "tensor is its tensor",
        "observed_vs_committed": "the fitted p_obs's are my re-probe "
                                 "certified equal to e214's committed "
                                 "records (max dp "
                                 f"{max(all_dps) if all_dps else 0:.2e}) — "
                                 "the fit is effectively ON the committed "
                                 "records",
        "two_moment": "the desk-only family replaces the true denominator "
                      "with a Gaussian tail estimate; its gap vs the full "
                      "fit prices exactly that crudeness",
        "guarantees_nothing": "nothing here is guaranteed — the openness is "
                              "the point (FQ7's branches were written before "
                              "this cell ran; no retrofit)",
    }
    metrics["compute"] = {
        "envelope": "CPU-only desk+eval (dispatch): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, one state = one eval burst; "
                    "~54 probes x 8 state reads + screening forwards + 7 "
                    "checkpoint loads; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(p) for p in
                          list(W1_ARCH.values()) + list(W2_ARCH.values())],
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "committed_y_side": {
            "e214_metrics": {"path": str(E214_METRICS),
                             "sha256_16": sha256_of(E214_METRICS)},
            "e214_journal": {"path": str(E214_JOURNAL),
                             "sha256_16": sha256_of(E214_JOURNAL)},
            "e228_journal": {"path": str(E228_JOURNAL),
                             "sha256_16": sha256_of(E228_JOURNAL)},
        },
        "logit_dumps": {tag: {"path": rec["npz"],
                              "sha256_16": rec["npz_sha256_16"],
                              "shape": rec["logits_shape"]}
                        for tag, rec in ljour.items()},
        "logit_dumps_git": "NOT tracked (the checkpoints/gradient-cache "
                           "precedent; regenerable deterministically from "
                           "the committed checkpoints + this script; sha256 "
                           "recorded)",
        "batteries": "module import of lab/e182c_forgetting_control.py "
                     "(fact/ctrl/near) and lab/e182c2_template.py (tmpl), "
                     "VERBATIM — import, never retyped",
        "stat_helpers": "avg_ranks/_pearson/spearman/kendall_tau_b copied "
                        "VERBATIM from lab/e228_margin_landscape.py",
        "family": "q_i(T) = softmax(L0_i/T)[ans_i]; Bernoulli soft-target "
                  "MLE; grid log10 T in [-1,1] x 1201 + golden polish; "
                  "float64",
        "versions": {"torch": torch.__version__,
                     "transformers": org_meta["transformers_version"],
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "threads": torch.get_num_threads(),
        "device": "cpu fp32",
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")

    if not SMOKE:
        png1 = make_fit_plot(rd, fit_rows, adj)
        png2 = make_residual_plot(rd, res_read, adj)
        log(f"outputs: {rd / 'metrics.json'}, {png1}, {png2}")
    else:
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
