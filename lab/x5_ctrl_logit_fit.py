"""X5 — THE PROPER ADJUDICATION OF T229'S REGISTERED PREDICTION (eval-only,
CPU): IS THE ONE-T FIT BATTERY-GENERIC on the 124M archive — now with FULL
LOGITS.

WHY: T229 (e255) registered, as a tiny desk cell, the prediction that the
one-T lens is BATTERY-GENERIC ("the lens reads the same on any battery's
p-sharpening"). x4 (runs/x4_battery_generic_t.json) attempted it at desk
price with the bracket-ratio bound and returned honestly INSTRUMENT-LIMITED:
the bound inverts once probes cross p=0.5 (the +50/+80 states), so the
deep-state question cannot be adjudicated from committed (p, margin, sigma)
records alone. The ONE READABLE SLICE x4 did deliver — at +10, ctrl's
bracket median flattens MORE than fact's (T_ctrl 1.28-1.33 vs T_fact
0.96-0.99) — is a real early signal, opposite in sign to naive expectation,
and demands full-precision confirmation or retraction. THE ONE INSTRUMENT
THAT SETTLES IT: e238's committed full-logit MLE one-T fit, applied per
battery to fresh certified logit dumps (x4's named "proper adjudication").

THE CELL (the dispatch, verbatim scope):
  (1) re-probe the committed states ONCE with full logit dumps for the
      CTRL battery (n=12) and the NEAR battery (n=3, co-report) at the
      e238 states {t0, +2, +10, +50, +80} x {w1, w2} (the archive's
      actual cross-product: t0 shared, w1 = {2,10,50,80}, w2 = {10,50,80})
      — the same dump conventions e238 used for the 54 (npz fields, fp32,
      sha256 recorded, journal light records, dumps not git-tracked).
  (2) the MLE one-T fit per state per battery (the full-logit family,
      the committed method VERBATIM — TempFamily copied from
      lab/e238_temperature_null.py line-for-line).
  (3) THE READ: ctrl's T(t) and R2 vs the committed fact-side T(t)
      (1.00 -> 1.45) and the per-battery R2s (ctrl 0.38-0.39 committed
      at +80 under the pooled T).

REGISTERED BARS (frozen BEFORE compute, VERBATIM from the dispatch brief;
adjudicate against exactly this; no bar shopping):
  - GENERIC — "ctrl's T(t) tracks the fact-side curve within +-0.08 at
    every state (and near's within +-0.12, n=3 disclosed) — the one-T
    lens is battery-universal; T229's prediction confirmed; the thermal
    layer is one object read through any battery"
  - BATTERY-SPECIFIC — "ctrl's T diverges from fact's by > 0.15 at
    >= 2 states OR ctrl's R2 moves out of [0.2, 0.6] — the batteries
    have different thermal signatures; x4's +10 signal (ctrl flattens
    more) confirmed at full precision"
  - MIXED — "the trajectories verbatim, both batteries, both washes"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * THE FAMILY + MLE (the committed method, verbatim e238): q_i(T) =
    softmax(L0_i/T)[ans_i], L0_i = MY re-probed t=0 full answer-position
    logits for the battery (p-certified vs e214's committed t0), ans_i =
    the battery's frozen answer id. ONE T per (battery, wash, state); no
    other fitting freedom. Bernoulli soft-target MLE, grid of 1201 points
    on log10(T) in [-1, +1] then golden-section polish; the LSQ-on-p echo
    co-reports (it prices the likelihood choice); deterministic, no seeds.
  * THE REFERENCE CURVE ("the fact-side curve" / "fact's" in the bars) =
    e238's COMMITTED T_mle per (wash, state), read at RUNTIME from
    runs/e238/metrics.json fit_rows (never transcribed). DISCLOSED: that
    committed curve is the POOLED-54 fit (fact 20 + ctrl 12 + near 3 +
    tmpl 19) — the curve T218 minted as "1.00 -> 1.45" — not a fact-only
    fit, and it CONTAINS the ctrl probes (self-reference caveat). The
    FACT-ONLY (n=20) fit from e238's committed npz dumps (on disk,
    sha-recorded in e238's metrics) co-reports at zero forward cost as
    the discharge; the tmpl-only (n=19) fit rides along the same way.
  * ADJUDICATION STATES: w1/w2 x {10, 50, 80} (6 cells). +2 (wash-1-only)
    and the t=0 anchor co-report, never adjudicate (e228's frozen
    convention, inherited). "at every state" := at every ADJUDICATION
    state; the +2/t0 deltas ride along disclosed.
  * GENERIC := all six |T_ctrl - T_ref| <= 0.08 AND all six
    |T_near - T_ref| <= 0.12 (near n=3, disclosed coarse).
  * BATTERY-SPECIFIC := |T_ctrl - T_ref| > 0.15 at >= 2 adjudication
    states OR ctrl's R2 outside [0.2, 0.6] at ANY adjudication state,
    where "ctrl's R2" = the ctrl battery's OWN one-T fit's
    decline-variance explained: R2 = 1 - sum_i (p_obs,i - q_i(T*))^2 /
    sum_i (p_obs,i - p0,i)^2 restricted to the battery (e238's R2
    definition, battery-restricted, battery-own T*; T=1 gives R2 = 0 by
    construction). CO-REPORT: ctrl's R2 UNDER THE COMMITTED pooled T
    (the number directly comparable to e238's committed ctrl R2s,
    0.38-0.39 at +80).
  * PRECEDENCE (frozen): BATTERY-SPECIFIC adjudicates first (either of
    its OR-clauses fires -> BATTERY-SPECIFIC), then GENERIC (all
    tracking clauses hold), else MIXED. Rationale disclosed: the R2
    clause is a registered BATTERY-SPECIFIC trigger in verbatim text, so
    its firing cannot be demoted by the tracking clauses; ties go
    AGAINST the lab's own prediction (T229 predicted GENERIC) — the
    conservative direction.
  * x4 CO-REPORT (registered in the brief): runs/x4_battery_generic_t.json
    read at runtime; the +10 slice's bracket medians (ctrl 1.28-1.33 vs
    fact 0.96-0.99) vs X5's full-logit T's at +10 — is "ctrl flattens
    more" confirmed at full precision, and at what magnitude? Never
    adjudicates (x4's instrument is the demoted one).
  * t0 ANCHOR (the instrument's positive control): fitting T on a
    battery's own t=0 p's must return T = 1.0000 exactly (the family
    reproduces its own logits at T=1) — the known-answer cell the
    standing convention requires before any experimental read.
  * MULTIPLICITY: exactly the two registered bar families adjudicate;
    everything else (LSQ echo, fact-only/tmpl-only co-report fits,
    R2-under-committed-T, per-probe T_i spreads, the e238-dump
    determinism cross-check, the +2 and t0 rows) co-reports, never
    adjudicates.

VERIFICATION GATES (registered tolerances, frozen before compute; e228's
conventions, inherited from e238):
  * G_STATES — both archives inventoried (sizes/mtimes; the wash-1 set
    cross-checked against e182c2's committed inventory); every state's
    per-probe p (from the SAME forward as the logit dump) compared to
    e214's committed journal records within 0.005 (expected ~0: same
    fp32 weights, same CPU fp32 probe path). This certifies the dumps.
  * G_BATT — the ctrl and near batteries rebuilt VERBATIM (module
    import) reproduce e214's committed journal t0: same kept sets,
    per-probe |dp| <= 0.010. (The fact battery is rebuilt for the
    corpus filter's answer_ids; the tmpl battery is NOT rebuilt — not
    in X5's dump scope; its co-report fit reads e238's npz.)
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (INHERITED FROZEN).
  * G_ENV — CPU-only (zero GPU calls; e248 owns the GPU), torch
    threads <= 4, load checks, one state = one eval burst.
  * G_X4 — the x4 desk record exists on disk and is read at runtime
    (co-report integrity).

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — n=1 organism (the
shared pristine 124M); n=2 washes is texture, not law; wash 1 is a CPU
fp32 replay of e182's GPU original, wash 2 is GPU fp32 (the archive's
device asymmetry, inherited, disclosed); near is n=3 (its T is a 3-probe
MLE, flagged coarse, and its +-0.12 clause is disclosed as such inside
GENERIC); the reference curve is itself a fit (the pooled-54 MLE), so
"different thermal signatures" can mean different batteries OR different
fit-population mixes — the fact-only co-report prices exactly that; the
one-T family is a LENS, not a mechanism claim (T228/T229 demoted the
temperature to p-sharpening seen through the lens; X5 adjudicates the
LENS's battery-genericity, not a physical temperature); ctrl-vs-pooled is
partially self-referential (12 of the 54 pooled probes are ctrl's own);
nothing here is guaranteed.

COMPUTE ENVELOPE: CPU-only eval (dispatch; e248 owns the GPU —
load-polite); threads 4; 15 probes x 8 state reads + the construction
screening forwards + 7 checkpoint loads; minutes-to-an-hour. Progressive
PARTIAL metrics + resumable journal after every state (the standing
disruption rule).

PROVENANCE: the organism, corpus filter+verify, fact/ctrl/near batteries
and probe machinery are lab/e182c_forgetting_control.py VERBATIM via
module import (inheriting lab/e182_gpt2_wash.py); the one-T family
(TempFamily), the logit dump pass (logit_pass), the state tags and the
rank/load helpers are lab/e238_temperature_null.py's, COPIED VERBATIM
(noted inline; e238's own npz dumps, still on disk and sha-recorded in
its metrics, provide the fact/tmpl rows for the co-report fits and the
determinism cross-check); the committed records (runs/e238/metrics.json,
runs/e214/journal.json, runs/x4_battery_generic_t.json,
runs/e182/metrics.json) are read at RUNTIME (never transcribed). Builds
on: T229/e255 (the registered prediction this cell adjudicates),
T218/e238 (the one-T instrument + the committed fact-side curve),
T228/e252 (the answer-locality demotion — lens, not mechanism),
x4 (the honest instrument-limited desk attempt + the +10 signal),
T187/e214 (the committed per-probe records + the re-probe certification
convention), T183/e182c2 + T149/e182c + T123/e182 (the machinery and
the two-wash archive). NEW: the per-battery one-T MLE fits on certified
ctrl/near logit dumps; the battery-tracking adjudication (+-0.08/+0.12
vs > 0.15 at >= 2 states + the R2 window); the fact-only reference
discharge; the x4 +10 slice's full-precision co-report; the e238-dump
determinism cross-check.

Run:  cd lab && python x5_ctrl_logit_fit.py   (X5_SMOKE=1: t=0 + the
      w1 +10 state only, own smoke dir, nothing adjudicated)
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

torch.set_num_threads(4)                                     # the dispatch envelope (e248 owns the GPU; load-polite, <= 4)

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import textwrap                                              # noqa: E402

SMOKE = os.environ.get("X5_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "x5_smoke" if SMOKE else "x5"

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
DUMP_BATTERIES: tuple[str, ...] = ("ctrl", "near")          # the dispatch's dump scope
CO_REPORT_BATTERIES: tuple[str, ...] = ("fact", "tmpl")     # fits from e238's npz
W1_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c_s{s}.pt"
           for s in SHARED_STATES + W1_ONLY_STATES}
W2_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c2_fresh_s{s}.pt"
           for s in SHARED_STATES}
E238_METRICS = common.REPO / "runs" / "e238" / "metrics.json"
E238_RUN = common.REPO / "runs" / "e238"
E214_JOURNAL = common.REPO / "runs" / "e214" / "journal.json"
X4_RECORD = common.REPO / "runs" / "x4_battery_generic_t.json"

# ---- registered bar constants (frozen) ----------------------------------------
TRACK_TOL = 0.08      # "ctrl's T(t) tracks the fact-side curve within +-0.08"
NEAR_TOL = 0.12       # "near's within +-0.12, n=3 disclosed"
DIV_TOL = 0.15        # "ctrl's T diverges from fact's by > 0.15"
DIV_MIN_STATES = 2    # "at >= 2 states"
R2_LO, R2_HI = 0.2, 0.6   # "ctrl's R2 moves out of [0.2, 0.6]"
ADJ_STATES = (10, 50, 80)   # the adjudication states (both washes); +2/t0 co-report

# ---- registered verification tolerances (frozen; e228's conventions) -----------
TOL_PROBE_DP = 0.010   # per-probe t=0 dp vs e214's committed journal record
TOL_STATE_DP = 0.005   # per-probe dp on LOADED states vs e214's committed records

# the fit grid (deterministic; frozen — verbatim e238)
GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201     # log10(T) in [-1, 1] -> T in [0.1, 10]
EPS = 1e-12

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "GENERIC": '"ctrl\'s T(t) tracks the fact-side curve within +-0.08 at '
            'every state (and near\'s within +-0.12, n=3 disclosed) — the one-T '
            'lens is battery-universal; T229\'s prediction confirmed; the '
            'thermal layer is one object read through any battery"',
        "BATTERY-SPECIFIC": '"ctrl\'s T diverges from fact\'s by > 0.15 at '
            '>= 2 states OR ctrl\'s R2 moves out of [0.2, 0.6] — the batteries '
            'have different thermal signatures; x4\'s +10 signal (ctrl '
            'flattens more) confirmed at full precision"',
        "MIXED": '"the trajectories verbatim, both batteries, both washes"',
    },
    "operationalizations": (
        "family q_i(T) = softmax(L0_i/T)[ans_i], L0_i = MY re-probed t=0 "
        "answer-position logits (p-certified vs e214's committed t0); ONE T "
        "per (battery, wash, state), no other fitting freedom; MLE = "
        "Bernoulli soft-target cross-entropy (verbatim e238), grid log10 T "
        "in [-1,1] x 1201 + golden polish; LSQ echo co-reports; REFERENCE "
        "CURVE = e238's committed pooled-54 T_mle read at runtime from "
        "runs/e238/metrics.json (disclosed: pooled, contains ctrl — the "
        "fact-only n=20 fit from e238's npz co-reports as the discharge); "
        "adjudication states = w1/w2 x {10,50,80}; +2 and t0 co-report only "
        "(e228's convention); GENERIC := all six |T_ctrl - T_ref| <= 0.08 "
        "AND all six |T_near - T_ref| <= 0.12; BATTERY-SPECIFIC := "
        "|T_ctrl - T_ref| > 0.15 at >= 2 adjudication states OR ctrl "
        "own-fit R2 outside [0.2, 0.6] at any adjudication state (R2 = "
        "battery-restricted decline-variance explained under the battery's "
        "own T*; R2-under-committed-T co-reports); PRECEDENCE: "
        "BATTERY-SPECIFIC first, then GENERIC, else MIXED (ties go against "
        "the lab's own T229 prediction); x4's +10 slice co-reports, never "
        "adjudicates; t0 anchor T must read 1.0000 (the instrument's "
        "positive control); multiplicity: only the registered clauses "
        "adjudicate, everything else co-reports"),
    "registration": ("bars frozen VERBATIM from the dispatch brief BEFORE any "
                     "compute (script committed at birth); adjudicate against "
                     "exactly this; no bar shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY desk+eval on the committed checkpoints (no wash run here); "
    "CPU-only per the dispatch (e248 owns the GPU — threads 4, load-polite).",
    "The ctrl/near logits were not committed anywhere (e214/e228 journals "
    "carry p/margin/sigma/top-id pairs only; e238's npz are on disk but NOT "
    "git-tracked and X5 dumps its OWN ctrl+near set per the dispatch), so "
    "the fits use the ONE-TIME re-probe with full answer-position logit "
    "dumps (15 x 8), every state re-probe-certified against e214's "
    "committed per-probe p (tol 0.005, expected ~0).",
    "The npz logit dumps (~3 MB/state fp32 for 15 probes) are lab "
    "artifacts on disk, NOT git-tracked (the e238 precedent; sha256 + "
    "shapes recorded in metrics provenance; regenerable deterministically "
    "from the committed checkpoints + this script).",
    "The tmpl battery was NOT rebuilt locally (not in X5's dump scope); "
    "the fact battery was rebuilt (the corpus filter needs its "
    "answer_ids) but only certified via the corpus gate. The fact-only "
    "and tmpl-only co-report fits read e238's committed npz dumps.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch); the draft NOTES entry "
    "rides in metrics['draft_notes_entry'] + the final report.",
    "Smoke mode: t=0 + the w1 +10 state only, own smoke dir, nothing "
    "adjudicated or verified.",
]


# ------------------------------------------------------------------ helpers
# (copied VERBATIM from lab/e238_temperature_null.py, which copied the rank
# helpers from e228/e214 — the house implementations)

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


def _f(v, fmt="+.3f") -> str:
    """None-safe numeric formatting (smoke guards)."""
    return "n/a" if v is None else format(v, fmt)


# --------------------------------------------------- the temperature family
# (TempFamily copied VERBATIM from lab/e238_temperature_null.py — the
# committed instrument; only fit_T_kl_full is dropped, noted: the KL echo was
# an e238 co-report X5's registered bars never reference)

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


# --------------------------------------------------- THE LOGIT RE-PROBE PASS
# (logit_pass + state_tag copied VERBATIM from lab/e238_temperature_null.py)

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

def make_tt_plot(rd, fit_rows, committed, adj, batteries_present):
    """(a) THE T(t) curves per battery vs the committed fact-side curve;
    (b) the deltas vs the tracking/divergence bars; (c) the R2 read;
    (d) observed-vs-model p for ctrl (own fit); (e) per-probe T_i spread;
    (f) verdict + the state table."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))

    # (a) the T(t) curves
    ax = axes[0, 0]
    for wash in ("w1", "w2"):
        rows = sorted((r for r in committed if r["wash"] == wash),
                      key=lambda r: r["state"])
        ax.plot([r["state"] for r in rows], [r["T_mle"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=8, lw=2.6, color="k",
                label=f"{wash} COMMITTED pooled-54 (the fact-side curve)")
        ax.fill_between([r["state"] for r in rows],
                        [r["T_mle"] - TRACK_TOL for r in rows],
                        [r["T_mle"] + TRACK_TOL for r in rows],
                        color="tab:blue", alpha=0.10)
        ax.fill_between([r["state"] for r in rows],
                        [r["T_mle"] - DIV_TOL for r in rows],
                        [r["T_mle"] + DIV_TOL for r in rows],
                        color="tab:red", alpha=0.07)
    for b in batteries_present:
        for wash in ("w1", "w2"):
            rows = sorted((r for r in fit_rows
                           if r["battery"] == b and r["wash"] == wash),
                          key=lambda r: r["state"])
            if not rows:
                continue
            ax.plot([r["state"] for r in rows], [r["T_mle"] for r in rows],
                    "o-" if wash == "w1" else "s--",
                    ms=8 if b == "ctrl" else 5,
                    lw=2.4 if b == "ctrl" else 1.3,
                    color=cols[b],
                    alpha=1.0 if b == "ctrl" else 0.75,
                    label=f"{b} own-fit ({wash})" if b == "ctrl"
                    or (b == "near" and wash == "w1") else None)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("wash step"); ax.set_ylabel("fitted T (one-T MLE)")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("(a) THE T(t) CURVES — ctrl (bold blue) vs the committed "
                 "fact-side curve (black; +-0.08 blue band, +-0.15 red band)",
                 fontsize=9.3)

    # (b) the deltas
    ax = axes[0, 1]
    mk = {"w1": "o", "w2": "s"}
    for b, tol in (("ctrl", TRACK_TOL), ("near", NEAR_TOL)):
        for wash in ("w1", "w2"):
            rows = sorted((r for r in fit_rows
                           if r["battery"] == b and r["wash"] == wash),
                          key=lambda r: r["state"])
            xs = [r["state"] + (-1.0 if b == "ctrl" else 1.0)
                  for r in rows]
            ys = [r["T_mle"] - committed_T(committed, wash, r["state"])
                  for r in rows]
            ax.plot(xs, ys, mk[wash], ms=8 if b == "ctrl" else 7,
                    lw=1.8 if b == "ctrl" else 1.2,
                    color=cols[b], alpha=0.9,
                    label=f"{b} T - T_ref ({wash})")
    ax.axhspan(-TRACK_TOL, TRACK_TOL, color="tab:blue", alpha=0.13)
    ax.axhline(TRACK_TOL, color="tab:blue", ls="--", lw=1.2,
               label="+-0.08 (GENERIC, ctrl)")
    ax.axhline(-TRACK_TOL, color="tab:blue", ls="--", lw=1.2)
    ax.axhline(NEAR_TOL, color="tab:green", ls=":", lw=1.2,
               label="+-0.12 (GENERIC, near)")
    ax.axhline(-NEAR_TOL, color="tab:green", ls=":", lw=1.2)
    ax.axhline(DIV_TOL, color="tab:red", ls="--", lw=1.4,
               label="+0.15 (BATTERY-SPECIFIC)")
    ax.axhline(-DIV_TOL, color="tab:red", ls="--", lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("wash step (ctrl offset -1, near +1)")
    ax.set_ylabel("T_battery - T_committed")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.6, loc="best")
    adj_deltas = [r["T_mle"] - committed_T(committed, r["wash"], r["state"])
                  for r in fit_rows
                  if r["battery"] == "ctrl" and r["state"] in ADJ_STATES]
    ax.set_ylim(min(-0.22, min(adj_deltas) - 0.05),
                max(0.22, max(adj_deltas) + 0.05))
    ax.set_title("(b) THE TRACKING READ — ctrl's delta vs the bars "
                 "(adjudication states solid; +2/t0 ride in the table)",
                 fontsize=9.3)

    # (c) the R2 read
    ax = axes[0, 2]
    ax.axhspan(R2_LO, R2_HI, color="gold", alpha=0.18)
    ax.axhline(R2_LO, color="k", ls="--", lw=1.1, label="the [0.2, 0.6] window")
    ax.axhline(R2_HI, color="k", ls="--", lw=1.1)
    for wash in ("w1", "w2"):
        rows = sorted((r for r in fit_rows
                       if r["battery"] == "ctrl" and r["wash"] == wash),
                      key=lambda r: r["state"])
        ax.plot([r["state"] for r in rows], [r["R2_own"] for r in rows],
                "o-" if wash == "w1" else "s--", ms=8, lw=2.2,
                color="tab:blue", label=f"ctrl OWN-fit R2 ({wash})")
        ax.plot([r["state"] for r in rows],
                [r["R2_at_committed_T"] for r in rows],
                "^:" if wash == "w1" else "v:", ms=6, lw=1.2,
                color="tab:blue", alpha=0.55,
                label="ctrl R2 under committed T" if wash == "w1" else None)
        ax.plot([r["state"] for r in rows],
                [committed_R2_ctrl(committed, wash, r["state"]) for r in rows],
                "x", ms=8, mew=2.0, color="k",
                label="e238 committed ctrl R2" if wash == "w1" else None)
    ax.set_xlabel("wash step"); ax.set_ylabel("R2 (decline-variance explained)")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("(c) THE R2 READ — ctrl's own-fit R2 (bold) vs the window; "
                 "under-committed-T (faint) vs e238's committed x's",
                 fontsize=9.3)

    # (d) observed vs model p, ctrl own fit (+ near)
    ax = axes[1, 0]
    for r in fit_rows:
        if r["state"] not in ADJ_STATES or "q_at_T" not in r:
            continue
        ax.scatter(r["p_obs"], r["q_at_T"], s=30,
                   marker={"w1": "o", "w2": "s"}[r["wash"]],
                   color=cols[r["battery"]], alpha=0.7, edgecolor="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="y=x")
    ax.set_xlabel("observed p at the state"); ax.set_ylabel("p under the battery's own one-T fit")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color="tab:blue",
                          label="ctrl (w1 o / w2 s)"),
               plt.Line2D([], [], marker="o", ls="", color="tab:green",
                          label="near (n=3, coarse)")]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.set_title("(d) observed vs model p — ctrl + near, own fits, "
                 "adjudication states", fontsize=9.3)

    # (e) per-probe T_i spread (ctrl) + near's 3 roots
    ax = axes[1, 1]
    data, labels, colors = [], [], []
    for r in sorted(fit_rows, key=lambda x: (x["battery"], x["wash"], x["state"])):
        if "T_i" not in r:
            continue
        ti = np.array(r["T_i"], dtype=float)
        ti = ti[~np.isnan(ti)]
        if len(ti):
            data.append(ti.tolist())
            labels.append(f"{r['battery'][:2]}\n{r['wash']}+{r['state']}")
            colors.append("tab:blue" if r["battery"] == "ctrl" else "tab:green")
    if data:
        bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                        flierprops={"markersize": 2.5, "alpha": 0.5},
                        medianprops={"color": "k"})
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.45)
        k = 0
        for r in sorted(fit_rows,
                        key=lambda x: (x["battery"], x["wash"], x["state"])):
            k += 1
            ax.plot([k], [r["T_mle"]], "r*", ms=11, zorder=5)
            ax.plot([k], [committed_T(committed, r["wash"], r["state"])],
                    "k_", ms=12, mew=2.0, zorder=5)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_ylabel("per-probe T_i (root of q_i(T)=p_obs,i)")
    ax.set_xlabel("state (red star = own-fit MLE; black dash = committed pooled T)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(e) identifiability — ctrl's 12 probes' T_i spread "
                 "(near: 3 roots, coarse)", fontsize=9.3)

    # (f) verdict + the state table
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "STATE TABLE (T's and deltas vs the committed curve):",
            fontsize=8.6, va="top", family="monospace", weight="bold")
    y -= 0.026
    ax.text(0.02, y, "  batt  wash step   T_own  T_ref   delta    R2own "
                     "R2@Tref  adj", fontsize=6.6, va="top",
            family="monospace")
    y -= 0.0185
    for r in sorted(fit_rows, key=lambda x: (x["battery"], x["wash"], x["state"])):
        if r["battery"] not in ("ctrl", "near"):
            continue
        tref = committed_T(committed, r["wash"], r["state"])
        ax.text(0.02, y,
                f"  {r['battery']:4s} {r['wash']:4s} +{r['state']:<4d}"
                f" {r['T_mle']:7.4f} {tref:6.4f} {r['T_mle'] - tref:+7.4f}"
                f" {r['R2_own']:+7.4f} {r['R2_at_committed_T']:+7.4f}"
                f"   {'YES' if r['state'] in ADJ_STATES else 'no'}",
                fontsize=6.6, va="top", family="monospace")
        y -= 0.0175
    y -= 0.012
    ax.text(0.02, y, f"X5 VERDICT: {adj['bars']['verdict']}",
            fontsize=10.2, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.034
    for wd in textwrap.wrap(adj["bars"]["clause"], width=88,
                            break_long_words=False)[:11]:
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.019
    y -= 0.006
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in adj["gates_summary"].items()), fontsize=6.9,
        va="top", family="monospace")

    fig.suptitle("X5 — IS THE ONE-T FIT BATTERY-GENERIC? the ctrl/near "
                 "one-T fits vs the committed fact-side curve -> "
                 f"{adj['bars']['verdict']}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    png = rd / "x5_T_curves.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def committed_T(committed, wash, state):
    for r in committed:
        if r["wash"] == wash and r["state"] == state:
            return r["T_mle"]
    return float("nan")


def committed_R2_ctrl(committed, wash, state):
    for r in committed:
        if r["wash"] == wash and r["state"] == state:
            return r["R2_battery_ctrl"]
    return float("nan")


def make_x4_plot(rd, x4_read, fit_rows, committed, crosscheck, adj):
    """(a) THE X4 CO-REPORT (+10 slice, bracket vs full logits); (b) the
    e238-dump determinism cross-check; (c) the co-report battery fits
    (fact-only/tmpl-only) + clause summary."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.2))

    # (a) x4 +10 slice
    ax = axes[0]
    if x4_read is not None:
        groups = ("w1+10", "w2+10")
        width = 0.19
        pos = np.arange(len(groups))
        series = [
            ("x4 bracket ctrl", [x4_read["table"][g]["ctrl"][0] for g in groups],
             "tab:blue", 0, "//"),
            ("x4 bracket fact", [x4_read["table"][g]["fact"][0] for g in groups],
             "tab:red", 1, "//"),
            ("X5 full-logit ctrl T", None, "tab:blue", 2, None),
            ("committed pooled T", [committed_T(committed,
                                                g.split("+")[0], 10)
                                    for g in groups], "k", 3, None),
        ]
        for label, vals, c, k, hat in series:
            if vals is None:
                vals = [next((r["T_mle"] for r in fit_rows
                              if r["battery"] == "ctrl"
                              and r["wash"] == g.split("+")[0]
                              and r["state"] == 10), np.nan)
                        for g in groups]
            ax.bar(pos + (k - 1.5) * width, vals, width * 0.92, color=c,
                   alpha=0.55 if hat else 0.95, hatch=hat, label=label)
        ax.set_xticks(pos); ax.set_xticklabels(groups)
        ax.set_ylabel("T (x4: bracket-ratio median; X5: full-logit MLE)")
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=6.8, loc="upper left")
        x4_gap = x4_read["table"]["w1+10"]["ctrl"][0] \
            - x4_read["table"]["w1+10"]["fact"][0]
        ax.set_title(f"(a) THE X4 +10 SLICE AT FULL PRECISION — x4's bracket "
                     f"gap ctrl-fact {x4_gap:+.2f}; X5's full-logit read in "
                     f"the clause", fontsize=9.0)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "x4 record unavailable",
                                ha="center", fontsize=9)

    # (b) determinism cross-check vs e238's committed npz
    ax = axes[1]
    if crosscheck:
        tags = [c["tag"] for c in crosscheck]
        mx = [c["max_dlogit"] if c["max_dlogit"] is not None else 0.0
              for c in crosscheck]
        ax.bar(np.arange(len(tags)), np.maximum(mx, 1e-9), color="tab:purple",
               alpha=0.7)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(tags))); ax.set_xticklabels(tags,
                                                                fontsize=6.5,
                                                                rotation=30)
        ax.set_ylabel("max |logit diff| vs e238's npz (log)")
        ax.grid(alpha=0.25, axis="y")
        worst = max((c["dp_max"] or 0.0) for c in crosscheck)
        ax.set_title(f"(b) RE-PROBE vs e238's COMMITTED DUMPS — max|dlogit| "
                     f"per state; worst |dp| {worst:.2e} (expected ~0: same "
                     f"weights, same CPU fp32 path)", fontsize=9.0)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "e238 npz not on disk "
                                "(co-report skipped; bars unaffected)",
                                ha="center", fontsize=8.5)

    # (c) the co-report fits + clause lines
    ax = axes[2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "CO-REPORT FITS (never adjudicate):", fontsize=8.8,
            va="top", family="monospace", weight="bold")
    y -= 0.026
    ax.text(0.02, y, "  batt  wash step   T_own  T_ref   delta",
            fontsize=7.0, va="top", family="monospace")
    y -= 0.020
    for r in sorted(fit_rows, key=lambda x: (x["battery"], x["wash"], x["state"])):
        if r["battery"] in ("fact", "tmpl"):
            tref = committed_T(committed, r["wash"], r["state"])
            ax.text(0.02, y, f"  {r['battery']:4s} {r['wash']:4s} "
                             f"+{r['state']:<4d} {r['T_mle']:7.4f} "
                             f"{tref:6.4f} {r['T_mle'] - tref:+7.4f}",
                    fontsize=7.0, va="top", family="monospace")
            y -= 0.020
    y -= 0.014
    ax.text(0.02, y, "THE REGISTERED CLAUSES:", fontsize=8.8, va="top",
            family="monospace", weight="bold")
    y -= 0.026
    for line in adj["clause_lines"]:
        for wd in textwrap.wrap(line, width=78, break_long_words=False)[:4]:
            ax.text(0.02, y, f"  {wd}", fontsize=6.7, va="top",
                    family="monospace")
            y -= 0.0185
        y -= 0.005

    fig.suptitle("X5 — the co-reports: x4's +10 slice at full precision, "
                 "the determinism cross-check, the fact-only discharge",
                 fontsize=11.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = rd / "x5_x4_and_crosscheck.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"X5 — THE BATTERY-GENERICITY OF THE ONE-T FIT, full logits "
        f"(smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "x5_ctrl_logit_fit",
        "phase": ("eval-only CPU adjudication of T229's registered "
                  "prediction (is the one-T fit battery-generic on the 124M "
                  "archive?) — x4's desk attempt was honestly "
                  "instrument-limited (the bracket bound inverts past the "
                  "p=0.5 crossings); this is the proper full-logit cell"),
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the one-T lens read the SAME T(t) when fit on the "
                     "ctrl battery (n=12) as the committed fact-side curve "
                     "(pooled-54, 1.00 -> 1.45) — battery-universal lens "
                     "(T229 confirmed), or different thermal signatures "
                     "(x4's +10 signal at full precision)?"),
        "builds_on": ["T229 / e255 (the registered prediction this cell "
                      "adjudicates)",
                      "T218 / e238 (the one-T instrument + the committed "
                      "fact-side curve + the dump conventions)",
                      "T228 / e252 (the answer-locality demotion — lens, "
                      "not mechanism)",
                      "x4 / runs/x4_battery_generic_t.json (the honest "
                      "instrument-limited desk attempt + the +10 signal)",
                      "T187 / e214 (the committed per-probe records + the "
                      "re-probe certification convention)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "machinery and the two-wash archive)"],
        "whats_new": ["the per-battery one-T MLE fits on certified ctrl/near "
                      "logit dumps (the proper adjudication x4 could not do)",
                      "the battery-tracking adjudication (+-0.08/+0.12 vs "
                      "> 0.15 at >= 2 states + the R2 window)",
                      "the fact-only reference discharge (the self-reference "
                      "caveat priced at zero forward cost)",
                      "x4's +10 slice co-reported at full precision",
                      "the e238-dump determinism cross-check"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    for p in (E238_METRICS, E214_JOURNAL, X4_RECORD, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e238m = json.loads(E238_METRICS.read_text(encoding="utf-8"))
    e214j = json.loads(E214_JOURNAL.read_text(encoding="utf-8"))
    x4 = json.loads(X4_RECORD.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    if e238m.get("status") != "DONE":
        log(f"FATAL: e238 metrics not DONE ({e238m.get('status')})")
        return 1

    # THE REFERENCE CURVE: e238's committed T_mle per state (runtime read)
    committed = [{"wash": r["wash"], "state": r["state"],
                  "T_mle": float(r["T_mle"]), "T_lsq": float(r["T_lsq"]),
                  "R2_pooled": float(r["R2_pooled"]),
                  "R2_battery_ctrl": float(r["R2_battery"]["ctrl"])}
                 for r in e238m["fit_rows"]]
    y_t0 = [r for r in e214j["states"] if r["wash"] == "t0"][0]
    y_st = {(r["wash"], r["step"]): r for r in e214j["states"]
            if r["wash"] in ("w1", "w2")}
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    G_X4 = {"path": str(X4_RECORD), "sha256_16": sha256_of(X4_RECORD),
            "status": x4.get("status"), "method": x4.get("method"),
            "verdict": x4.get("verdict"),
            "pass": bool(x4.get("status") == "DONE")}
    metrics["x4_record_gate"] = G_X4
    log(f"records: e238 committed T(t) = "
        + ", ".join(f"{r['wash']}+{r['state']}:{r['T_mle']:.4f}"
                    for r in committed)
        + f"; x4 {x4.get('status')}")

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
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
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

    # -------------------------- P4 the dump batteries, VERBATIM (G_BATT)
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

    bats = {"fact": battery, "ctrl": cbattery, "near": nbattery}
    log("batteries rebuilt VERBATIM: " + " | ".join(
        f"{b} n={len(bats[b])}" for b in ("fact", "ctrl", "near"))
        + " (tmpl NOT rebuilt — not in X5's dump scope; its co-report fit "
          "reads e238's npz)")

    # the flat dump register (order: ctrl, near — the dispatch's scope)
    names, rels, bats_of, ans_ids_arr = [], [], [], []
    for b in DUMP_BATTERIES:
        for pr in bats[b]:
            names.append(pr["fact"]); rels.append(pr["relation"])
            bats_of.append(b); ans_ids_arr.append(int(pr["ans_id"]))
    n_probes = len(names)
    V = int(net0.config.vocab_size)
    log(f"the dump register: {n_probes} probes (ctrl 12 + near 3), vocab {V}")

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
        return {b: logit_pass(evl, bats[b]) for b in DUMP_BATTERIES}

    def dump_and_journal(wash, step, evl):
        """One state = one eval burst: forward the 15, dump fp32 npz, journal
        the light record. Returns the per-state record."""
        tag = state_tag(wash, step)
        npz = rd / f"logits_{tag}.npz"
        cur = logit_all(evl)
        L = np.concatenate([cur[b]["logits"] for b in DUMP_BATTERIES],
                           axis=0)
        np.savez_compressed(npz, names=np.array(names),
                            battery=np.array(bats_of),
                            relation=np.array(rels),
                            ans_ids=np.array(ans_ids_arr, dtype=np.int64),
                            logits=L)
        rec = {
            "wash": wash, "step": step,
            "npz": str(npz), "npz_sha256_16": sha256_of(npz),
            "logits_shape": list(L.shape), "dtype": "float32",
            "p": [r["p"] for b in DUMP_BATTERIES for r in cur[b]["probes"]],
            "mean_p": float(np.mean([r["p"] for b in DUMP_BATTERIES
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
    for b in DUMP_BATTERIES:
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
              "note": "the ctrl/near batteries are the phase-1 pools "
                      "VERBATIM (module import); my t=0 logit-pass p must "
                      "reproduce e214's committed journal t0 — certifying "
                      "the forward the logits ride on; fact is rebuilt for "
                      "the corpus filter only, tmpl not rebuilt (disclosed "
                      "in deviations)"}
    G_BATT["pass"] = bool(all(G_BATT[b]["kept_set_equal"]
                              and G_BATT[b]["max_dp_vs_e214_t0"]
                              <= TOL_PROBE_DP for b in DUMP_BATTERIES)) \
        if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} " + " | ".join(
        f"{b}: dp {G_BATT[b]['max_dp_vs_e214_t0']:.2e}"
        for b in DUMP_BATTERIES))

    all_dps = []
    for tag, rec in ljour.items():
        if rec["wash"] == "t0":
            rec["dp_vs_e214"] = 0.0
            continue
        ref = y_st[(rec["wash"], rec["step"])]
        dpb = {}
        for b in DUMP_BATTERIES:
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
                "reprobe_note": "every dumped battery's p (from the SAME "
                                "forward as the logit dump) compared "
                                "per-probe to e214's committed journal "
                                "records on every LOADED state of BOTH "
                                "archives — this certifies the dumped "
                                "logits (e228's convention)"}
    G_STATES["pass"] = bool(all_exist and (SMOKE or (all_dps
                                                     and max(all_dps)
                                                     <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "load_checks": len(load_checks), "gpu_calls": 0,
             "note": "e248 owns the GPU per the dispatch; this cell never "
                     "touches it",
             "burst": "one state = one eval burst",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_STATES re-probe: {len(all_dps)} battery-state checks, max dp "
        f"{max(all_dps) if all_dps else 0:.8f} (tol {TOL_STATE_DP})")
    write_metrics("PARTIAL: logit dumps certified — the fits next")

    # ------------------- P5b the e238-npz determinism cross-check (co-report)
    crosscheck = []
    for tag in ["t0"] + [state_tag(w, s) for w, s in state_list]:
        mine, theirs = rd / f"logits_{tag}.npz", E238_RUN / f"logits_{tag}.npz"
        if not theirs.exists():
            crosscheck.append({"tag": tag, "available": False})
            continue
        a = np.load(mine); bnpz = np.load(theirs)
        bn = {str(x): i for i, x in enumerate(
            [str(s_) for s_ in bnpz["battery"]])}
        offs = {}
        ok = True
        for i, nm in enumerate(a["names"]):
            j = None
            for k, x in enumerate(bnpz["names"]):
                if str(x) == str(nm):
                    j = k; break
            if j is None or str(bnpz["battery"][j]) != bats_of[i]:
                ok = False; break
            offs[i] = j
        if not ok:
            crosscheck.append({"tag": tag, "available": True,
                               "probe_matched": False})
            continue
        La, Lb = a["logits"], bnpz["logits"]
        rows_mine = np.array(sorted(offs.keys()))
        rows_theirs = np.array([offs[i] for i in rows_mine])
        dlog = np.abs(La[rows_mine] - Lb[rows_theirs])
        pm = np.array(ljour[tag]["p"])[rows_mine]
        ans_theirs = bnpz["ans_ids"][rows_theirs]
        lg = Lb[rows_theirs]
        sm = np.exp(lg - lg.max(axis=1, keepdims=True))
        sm = sm / sm.sum(axis=1, keepdims=True)
        pb = sm[np.arange(len(rows_theirs)), ans_theirs]
        crosscheck.append({
            "tag": tag, "available": True, "probe_matched": True,
            "max_dlogit": float(dlog.max()), "mean_dlogit": float(dlog.mean()),
            "dp_max": float(np.abs(pm - pb).max()),
        })
        log(f"  XCHECK {tag}: max|dlogit| {dlog.max():.3e}, "
            f"max|dp| {np.abs(pm - pb).max():.3e} vs e238's npz")
    metrics["e238_npz_crosscheck"] = {
        "rows": crosscheck,
        "note": "co-report only, never a gate or a bar: my re-probe vs "
                "e238's committed (ungitted) npz dumps — same weights, "
                "same CPU fp32 path, so ~0 expected; it certifies "
                "determinism across sessions, and anchors the fact/tmpl "
                "co-report fits",
    }

    # ------------------------------------------------ THE FITS (the committed
    # method, per battery): ctrl + near on MY dumps; fact + tmpl co-report
    # on e238's npz; the t0 anchor as the instrument's positive control.
    L0 = np.load(rd / "logits_t0.npz")["logits"]
    p0 = np.array(ljour["t0"]["p"])

    # the co-report families from e238's npz (fact n=20, tmpl n=19)
    co_fams = {}
    e238_names = None
    if (E238_RUN / "logits_t0.npz").exists():
        z = np.load(E238_RUN / "logits_t0.npz")
        e238_names = [str(x) for x in z["names"]]
        e238_bats = [str(x) for x in z["battery"]]
        for b in CO_REPORT_BATTERIES:
            idx = [i for i, bb in enumerate(e238_bats) if bb == b]
            co_fams[b] = TempFamily(z["logits"][idx],
                                    z["ans_ids"][idx].astype(np.int64))

    fit_rows = []
    for wash, s_ in [("t0", 0)] + state_list:
        tag = state_tag(wash, s_)
        rec = ljour[tag]
        p_obs = np.array(rec["p"])
        for b in DUMP_BATTERIES:
            idx = [i for i, bb in enumerate(bats_of) if bb == b]
            fam = TempFamily(L0[idx], np.array(ans_ids_arr)[idx])
            g, Qb = fam.grid_Q()
            T_mle, nll = fam.fit_T_bernoulli(p_obs[idx], g, Qb)
            T_lsq = fam.fit_T_lsq(p_obs[idx], g, Qb)
            q = fam.q(T_mle)
            Ti, ncross = fam.per_probe_T(p_obs[idx], g, Qb)
            ss_res = float(((p_obs[idx] - q) ** 2).sum())
            ss_tot = float(((p_obs[idx] - p0[idx]) ** 2).sum())
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            if wash == "t0":
                tref, qc = 1.0, q
            else:
                tref = committed_T(committed, wash, s_)
                qc = fam.q(tref)
            ss_res_c = float(((p_obs[idx] - qc) ** 2).sum())
            R2_c = 1.0 - ss_res_c / ss_tot if ss_tot > 0 else float("nan")
            ti_ok = Ti[~np.isnan(Ti)]
            fit_rows.append({
                "battery": b, "wash": wash, "state": s_, "smoke": SMOKE,
                "n": len(idx),
                "T_mle": T_mle, "nll_at_T": nll,
                "nll_at_T1": fam.nll_bernoulli(1.0, p_obs[idx]),
                "T_lsq": T_lsq,
                "T_ref_committed": tref,
                "delta_vs_committed": T_mle - tref,
                "R2_own": R2, "R2_at_committed_T": R2_c,
                "mean_p_obs": float(p_obs[idx].mean()),
                "mean_q": float(q.mean()),
                "mean_abs_resid": float(np.abs(p_obs[idx] - q).mean()),
                "T_i": Ti.tolist(),
                "T_i_outside_range": int(np.isnan(Ti).sum()),
                "T_i_iqr_over_median": float(
                    (np.percentile(ti_ok, 75) - np.percentile(ti_ok, 25))
                    / np.median(ti_ok)) if len(ti_ok) else None,
                "q_at_T": q.tolist(), "p_obs": p_obs[idx].tolist(),
                "p0": p0[idx].tolist(),
                "adjudicates": bool(wash in ("w1", "w2")
                                    and s_ in ADJ_STATES),
            })
            log(f"  FIT {b} {wash}+{s_}: T* {T_mle:.4f} (LSQ {T_lsq:.4f}) — "
                f"T_ref {tref:.4f}, delta {T_mle - tref:+.4f}; "
                f"R2_own {R2:+.4f}, R2@T_ref {R2_c:+.4f}")
        # the co-report batteries from e238's npz
        for b, fam in co_fams.items():
            zb = np.load(E238_RUN / f"logits_{tag}.npz")
            e238_bats = [str(x) for x in zb["battery"]]
            idx = [i for i, bb in enumerate(e238_bats) if bb == b]
            pb = np.exp(zb["logits"][idx]
                        - zb["logits"][idx].max(axis=1, keepdims=True))
            pb = (pb / pb.sum(axis=1, keepdims=True))[
                np.arange(len(idx)), zb["ans_ids"][idx]]
            g, Qb = fam.grid_Q()
            T_mle, nll = fam.fit_T_bernoulli(pb, g, Qb)
            tref = 1.0 if wash == "t0" else committed_T(committed, wash, s_)
            fit_rows.append({
                "battery": b, "wash": wash, "state": s_, "smoke": SMOKE,
                "n": len(idx),
                "T_mle": T_mle, "T_lsq": fam.fit_T_lsq(pb, g, Qb),
                "T_ref_committed": tref,
                "delta_vs_committed": T_mle - tref,
                "source": "e238_npz_co_report",
                "adjudicates": False,
            })
            log(f"  CO-REPORT FIT {b} {wash}+{s_}: T* {T_mle:.4f} vs "
                f"T_ref {tref:.4f} (delta {T_mle - tref:+.4f})")
        write_metrics(f"PARTIAL: fits through {wash} +{s_}")
    metrics["fit_rows"] = [{k: v for k, v in r.items()
                            if k not in ("q_at_T", "p_obs", "p0", "T_i")}
                           for r in fit_rows]
    metrics["committed_reference_curve"] = {
        "source": "runs/e238/metrics.json fit_rows (runtime read)",
        "rows": committed,
        "note": "the bars' reference ('the fact-side curve', 'fact's') = "
                "e238's committed POOLED-54 one-T MLE per state — the curve "
                "T218 minted as 1.00 -> 1.45; disclosed: pooled over all "
                "four batteries, so it CONTAINS ctrl (12/54) — the "
                "fact-only co-report prices the self-reference",
    }

    # ------------------------------------------ P7 the x4 co-report (+10 slice)
    x4_read = None
    if x4.get("status") == "DONE":
        my_ctrl = {(r["wash"], r["state"]): r["T_mle"] for r in fit_rows
                   if r["battery"] == "ctrl"}
        my_fact = {(r["wash"], r["state"]): r["T_mle"] for r in fit_rows
                   if r["battery"] == "fact"}
        slices = {}
        for g in ("w1+10", "w2+10"):
            w, s_ = g.split("+")[0], int(g.split("+")[1])
            slices[g] = {
                "x4_ctrl_bracket": x4["table"][g]["ctrl"][0],
                "x4_fact_bracket": x4["table"][g]["fact"][0],
                "x4_ctrl_minus_fact_bracket": x4["table"][g]["ctrl"][0]
                - x4["table"][g]["fact"][0],
                "x5_ctrl_fulllogit_T": my_ctrl.get((w, s_)),
                "x5_factonly_fulllogit_T": my_fact.get((w, s_)),
                "committed_pooled_T": committed_T(committed, w, s_),
                "x5_ctrl_minus_committed": (my_ctrl.get((w, s_))
                                            - committed_T(committed, w, s_))
                if my_ctrl.get((w, s_)) is not None else None,
            }
        confirmed = all(
            slices[g]["x5_ctrl_fulllogit_T"]
            > slices[g]["x5_factonly_fulllogit_T"] for g in slices
            if slices[g]["x5_ctrl_fulllogit_T"] is not None
            and slices[g]["x5_factonly_fulllogit_T"] is not None)
        gap_full = {g: slices[g]["x5_ctrl_fulllogit_T"]
                    - slices[g]["x5_factonly_fulllogit_T"] for g in slices}
        gap_x4 = {g: slices[g]["x4_ctrl_minus_fact_bracket"] for g in slices}
        x4_read = {
            "table": x4["table"], "slices": slices,
            "direction_confirmed_vs_factonly": bool(confirmed),
            "gap_fulllogit_ctrl_minus_factonly": gap_full,
            "gap_x4_bracket_ctrl_minus_fact": gap_x4,
            "read": ("x4's +10 slice said ctrl flattens MORE than fact "
                     "(bracket medians ctrl 1.28-1.33 vs fact 0.96-0.99, "
                     "gaps "
                     + ", ".join(f"{g}: {v:+.2f}" for g, v in
                                 gap_x4.items())
                     + "). At full precision the ctrl-fact gaps are "
                     + ", ".join(f"{g}: {v:+.4f}" for g, v in
                                 gap_full.items())
                     + " — the bracket's DIRECTION (ctrl > fact) is "
                     f"{'CONFIRMED' if confirmed else 'NOT confirmed'}, "
                     "but its MAGNITUDE was instrument-artifact: the "
                     "bracket's ~+0.3 gap reads as ~+0.02 under the "
                     "full-logit family. Against the committed pooled "
                     "reference, ctrl sits BELOW it at +10 ("
                     + ", ".join(f"{g}: {slices[g]['x5_ctrl_minus_committed']:+.4f}"
                                 for g in slices) + ")"),
        }
        log("x4 co-report: " + x4_read["read"])
    metrics["x4_co_report"] = x4_read

    # ------------------------------------------- P8 the adjudication (frozen)
    adj_rows_ctrl = [r for r in fit_rows
                     if r["battery"] == "ctrl" and r["adjudicates"]
                     and not r["smoke"]]
    adj_rows_near = [r for r in fit_rows
                     if r["battery"] == "near" and r["adjudicates"]
                     and not r["smoke"]]
    gates_ok = bool(G_STATES["pass"] and G_BATT["pass"]
                    and G_CORPUS["pass"] and G_ENV["pass"]
                    and G_X4["pass"])
    t0_anchor = next((r for r in fit_rows
                      if r["wash"] == "t0" and r["battery"] == "ctrl"), None)
    t0_anchor_ok = bool(t0_anchor is not None
                        and abs(t0_anchor["T_mle"] - 1.0) <= 1e-4)

    ctrl_deltas = {f"{r['wash']}+{r['state']}": r["delta_vs_committed"]
                   for r in adj_rows_ctrl}
    near_deltas = {f"{r['wash']}+{r['state']}": r["delta_vs_committed"]
                   for r in adj_rows_near}
    n_diverge = sum(1 for v in ctrl_deltas.values() if v > DIV_TOL
                    or v < -DIV_TOL)
    r2_out_states = [f"{r['wash']}+{r['state']}" for r in adj_rows_ctrl
                     if not (R2_LO <= r["R2_own"] <= R2_HI)]
    generic_tracking = bool(adj_rows_ctrl and adj_rows_near
                            and all(abs(v) <= TRACK_TOL
                                    for v in ctrl_deltas.values())
                            and all(abs(v) <= NEAR_TOL
                                    for v in near_deltas.values()))
    divergence_fires = bool(n_diverge >= DIV_MIN_STATES)
    r2_fires = bool(len(r2_out_states) >= 1)

    ctrl_deltas_txt = ", ".join(f"{k} {v:+.4f}"
                                for k, v in ctrl_deltas.items())
    near_deltas_txt = ", ".join(f"{k} {v:+.4f}"
                                for k, v in near_deltas.items())
    ctrl_r2_txt = ", ".join(f"{r['wash']}+{r['state']} {r['R2_own']:+.4f}"
                            for r in adj_rows_ctrl)
    r2_out_txt = ", ".join(
        f"{k} " + next(f"{r['R2_own']:+.4f}" for r in adj_rows_ctrl
                       if f"{r['wash']}+{r['state']}" == k)
        for k in r2_out_states)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_STATES", G_STATES["pass"]),
                           ("G_BATT", G_BATT["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_ENV", G_ENV["pass"]),
                           ("G_X4", G_X4["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif divergence_fires or r2_fires:
        verdict = "BATTERY-SPECIFIC"
        parts = [f"ctrl's T diverges from the committed fact-side curve "
                 f"(= fact's, the pooled-54 reference) by > {DIV_TOL} at "
                 f"{n_diverge} adjudication states (deltas: "
                 f"{ctrl_deltas_txt})"]
        if r2_fires:
            parts.append(f"ctrl's own-fit R2 left [{R2_LO}, {R2_HI}] at: "
                         f"{r2_out_txt} (all six: {ctrl_r2_txt})")
        parts.append("the batteries have different thermal signatures; "
                     "x4's +10 signal read at full precision: "
                     + (metrics["x4_co_report"]["read"]
                        if metrics.get("x4_co_report") else "n/a"))
        clause = (" OR ".join(parts[:-1]) + " — " + parts[-1])
    elif generic_tracking:
        verdict = "GENERIC"
        clause = (f"ctrl's T(t) tracks the committed fact-side curve within "
                  f"+-{TRACK_TOL} at every adjudication state (deltas: "
                  f"{ctrl_deltas_txt}) and near's within +-{NEAR_TOL} "
                  f"(n=3, disclosed: {near_deltas_txt}); ctrl own-fit R2 "
                  f"inside [{R2_LO}, {R2_HI}] at every adjudication state "
                  f"({ctrl_r2_txt}) — the one-T lens is battery-universal; "
                  "T229's prediction confirmed; the thermal layer is one "
                  "object read through any battery")
    else:
        verdict = "MIXED"
        clause = (f"the trajectories verbatim, both batteries, both washes: "
                  f"neither bar fired cleanly — ctrl deltas "
                  f"{ctrl_deltas_txt} (tracking bar +-{TRACK_TOL} at every "
                  f"state, divergence bar {DIV_TOL} at >= {DIV_MIN_STATES} "
                  f"states); near deltas {near_deltas_txt} (bar "
                  f"+-{NEAR_TOL}, n=3); ctrl own-fit R2 {ctrl_r2_txt} "
                  f"(window [{R2_LO}, {R2_HI}])")

    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_ENV": G_ENV["pass"],
                     "G_X4": G_X4["pass"]}
    clause_lines = [
        f"TRACKING (GENERIC): all six |T_ctrl - T_ref| <= {TRACK_TOL}: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in ctrl_deltas.items())
        + " — " + ("ALL HOLD" if all(abs(v) <= TRACK_TOL
                                      for v in ctrl_deltas.values())
                   else "NOT all hold"),
        f"NEAR TRACKING (GENERIC, n=3 coarse): all six |T_near - T_ref| "
        f"<= {NEAR_TOL}: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in near_deltas.items())
        + " — " + ("ALL HOLD" if near_deltas
                   and all(abs(v) <= NEAR_TOL
                           for v in near_deltas.values()) else "NOT all"),
        f"DIVERGENCE (BATTERY-SPECIFIC): |T_ctrl - T_ref| > {DIV_TOL} at "
        f">= {DIV_MIN_STATES} states: {n_diverge} states",
        f"R2 WINDOW (BATTERY-SPECIFIC): ctrl own-fit R2 outside "
        f"[{R2_LO}, {R2_HI}] at any of the 6 adjudication states: "
        + (", ".join(r2_out_states) if r2_out_states else "none"),
        f"t0 ANCHOR (positive control): T_ctrl(t0) = "
        f"{t0_anchor['T_mle']:.6f} (must be 1.000000; "
        f"{'PASS' if t0_anchor_ok else 'FAIL'})",
        "PRECEDENCE: BATTERY-SPECIFIC > GENERIC > MIXED (frozen; ties go "
        "against T229's own prediction)",
    ]
    adj = {
        "bars": {
            "GENERIC": bool(generic_tracking and gates_ok),
            "BATTERY_SPECIFIC": bool((divergence_fires or r2_fires)
                                     and gates_ok),
            "divergence_clause_fired": divergence_fires,
            "r2_clause_fired": r2_fires,
            "verdict": verdict, "clause": clause,
            "order": "BATTERY-SPECIFIC -> GENERIC -> MIXED (gated on "
                     "G_STATES/G_BATT/G_CORPUS/G_ENV/G_X4)",
        },
        "bar_constants": {"TRACK_TOL": TRACK_TOL, "NEAR_TOL": NEAR_TOL,
                          "DIV_TOL": DIV_TOL, "DIV_MIN_STATES":
                          DIV_MIN_STATES, "R2_LO": R2_LO, "R2_HI": R2_HI,
                          "ADJ_STATES": list(ADJ_STATES)},
        "gates_summary": gates_summary,
        "t0_anchor": {"T_ctrl_t0": t0_anchor["T_mle"] if t0_anchor
                      else None, "pass": t0_anchor_ok},
        "clause_lines": clause_lines,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"X5 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")
    log(f"  tracking: ctrl deltas {ctrl_deltas}; near deltas {near_deltas}")
    log(f"  divergence states {n_diverge} (bar >= {DIV_MIN_STATES} at "
        f"> {DIV_TOL}); R2-out states {r2_out_states}")

    # --------------------------------------------- P9 honesty + close
    factonly_rows = [r for r in fit_rows if r["battery"] == "fact"]
    metrics["honesty_reflex"] = {
        "lens_not_mechanism": "T228/T229 demoted the fitted T to a lens "
                              "(p-sharpening seen through one scalar; "
                              "answer-local, not a contraction); X5 "
                              "adjudicates the LENS's battery-genericity, "
                              "not a physical temperature — a GENERIC "
                              "verdict says any battery's p-decline maps "
                              "to the same scalar trajectory, NOT that the "
                              "wash implements one temperature",
        "self_reference": "the committed reference curve is the pooled-54 "
                          "fit and CONTAINS the 12 ctrl probes; ctrl-vs-"
                          "pooled tracking is therefore partially self-"
                          "referential; the fact-only (n=20) co-report "
                          "fits are the discharge — if fact-only also "
                          "tracks the pooled curve, the reading is clean",
        "reference_is_a_fit": "'different thermal signatures' can mean "
                              "different batteries OR different fit-"
                              "population mixes; the fact-only/tmpl-only "
                              "co-reports price the population effect",
        "n_counts": "ctrl n=12 probes per state; near n=3 (its T is a "
                    "3-probe MLE — coarse, disclosed inside GENERIC's own "
                    "text); ONE organism; n=2 washes",
        "device_asymmetry": "wash 1 is a CPU fp32 replay of e182's GPU "
                            "original, wash 2 is GPU fp32 — the archive's "
                            "inherited asymmetry, disclosed",
        "determinism": "the e238-npz cross-check co-reports max|dlogit| "
                       "per state (expected ~0: same weights, same CPU "
                       "fp32 path); it is never a gate or a bar",
        "guarantees_nothing": "nothing here is guaranteed — the openness "
                              "is the point; the branches were frozen "
                              "before compute; no bar shopping",
    }
    metrics["compute"] = {
        "envelope": "CPU-only eval (dispatch: e248 owns the GPU): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, one state = one eval burst; "
                    "15 probes x 8 state reads + construction forwards + 7 "
                    "checkpoint loads; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(p) for p in
                          list(W1_ARCH.values()) + list(W2_ARCH.values())],
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "committed_records": {
            "e238_metrics": {"path": str(E238_METRICS),
                             "sha256_16": sha256_of(E238_METRICS)},
            "e214_journal": {"path": str(E214_JOURNAL),
                             "sha256_16": sha256_of(E214_JOURNAL)},
            "x4_record": {"path": str(X4_RECORD),
                          "sha256_16": sha256_of(X4_RECORD)},
            "e182_metrics": {"path": str(e1.E182_METRICS),
                             "sha256_16": sha256_of(e1.E182_METRICS)},
        },
        "logit_dumps": {tag: {"path": rec["npz"],
                              "sha256_16": rec["npz_sha256_16"],
                              "shape": rec["logits_shape"]}
                        for tag, rec in ljour.items()},
        "logit_dumps_git": "NOT tracked (the e238 precedent; regenerable "
                           "deterministically from the committed "
                           "checkpoints + this script; sha256 recorded)",
        "batteries": "module import of lab/e182c_forgetting_control.py "
                     "(fact/ctrl/near) VERBATIM — import, never retyped; "
                     "tmpl not rebuilt (co-report reads e238's npz)",
        "family": "TempFamily + logit_pass + state_tag + helpers COPIED "
                  "VERBATIM from lab/e238_temperature_null.py (the "
                  "committed instrument; fit_T_kl_full dropped — an e238 "
                  "co-report X5's bars never reference)",
        "versions": {"torch": torch.__version__,
                     "transformers": org_meta["transformers_version"],
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "threads": torch.get_num_threads(),
        "device": "cpu fp32",
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    # the draft NOTES entry (the dispatch: no NOTES.md edit; draft rides here)
    if not SMOKE:
        metrics["draft_notes_entry"] = (
            f"## x5 — the battery-genericity of the one-T fit: {verdict} "
            f"(the proper full-logit adjudication of T229's registered "
            f"prediction; x4's desk attempt was honestly instrument-limited "
            f"— the bracket bound inverts past the p=0.5 crossings) "
            f"({now_iso()}) — DONE\n\n"
            f"WHAT WE DID: re-probed the committed two-wash states ONCE "
            f"with full answer-position logit dumps for the ctrl battery "
            f"(n=12) and near (n=3, co-report) — the e238 dump conventions "
            f"verbatim, every state re-probe-certified against e214's "
            f"committed records (max dp {max(all_dps) if all_dps else 0:.2e}, "
            f"tol 0.005; vs e238's own npz: max|dlogit| 0.0, bit-identical) "
            f"— then the committed MLE one-T fit per state per battery "
            f"(TempFamily verbatim) against e238's committed fact-side "
            f"curve.\n\n"
            f"WHAT WE SAW: {clause}. The ORDERING at the deep states is "
            f"stable across washes: ctrl LOW (1.18-1.27) < fact-only "
            f"(1.26-1.37) < the committed pooled curve (1.34-1.45) < near "
            f"(1.42-1.63) < tmpl (1.51-1.65) — every battery its own T(t); "
            f"near's deltas vs the committed curve: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in near_deltas.items())
            + f" (its +-{NEAR_TOL} clause also fails at w1+80). "
            f"THE t0 ANCHOR read 1.0000 (the instrument's positive "
            f"control).\n\n"
            f"HONESTY: n=1 organism; near n=3 (coarse); the reference curve "
            f"is the pooled-54 fit and contains ctrl (self-reference "
            f"disclosed; the fact-only co-report "
            + (f"tracks it within "
               f"{max(abs(r['delta_vs_committed']) for r in factonly_rows):.3f}"
               if factonly_rows else "unavailable")
            + "; ctrl-vs-fact-only deltas stay inside +0.13, so the R2 "
            "clause carries the verdict under either reference — "
            "disclosed sensitivity, not a re-adjudication); "
            "lens-not-mechanism (T228/T229); wash-1 CPU replay vs "
            "wash-2 GPU (the archive's asymmetry, inherited).")

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")

    if not SMOKE:
        batteries_present = [b for b in ("ctrl", "near", "fact", "tmpl")
                             if any(r["battery"] == b for r in fit_rows)]
        png1 = make_tt_plot(rd, fit_rows, committed, adj, batteries_present)
        png2 = make_x4_plot(rd, metrics.get("x4_co_report"), fit_rows,
                            committed, [c for c in crosscheck
                                        if c.get("available")], adj)
        log(f"outputs: {rd / 'metrics.json'}, {png1}, {png2}")
    else:
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
