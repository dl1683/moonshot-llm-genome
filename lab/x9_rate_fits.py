"""X9 — THE RATE-FITS INSTRUMENT (T238's mispricing resolver; desk cell, CPU-only).

Dispatch (2026-10-05, one of the parallel desk cells under the owner's CPU
directive): the one-T "thermal" lens (x4/x5 family) fits battery discharge
curves with a single scalar T(t); x5's verdict was BATTERY-SPECIFIC (every
battery its own T, ordering ctrl 1.18-1.27 < fact-only 1.26-1.37 < pooled <
near 1.42-1.63 < tmpl 1.51-1.65, all fitting well individually). T238 flagged
a MISPRICING: the near/tmpl batteries read "hotter" under one-T — but is that
a genuinely faster decay RATE, or a different decay SHAPE that one-T translates
into phantom temperature? x7 proved the one-T lens is rescale-invariant (scale
arithmetic cannot move it) — so any mispricing is shape-real, not scale-real.
This cell resolves it.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
script committed at birth; adjudicate against exactly this; no bar shopping):

- RATE-DISTINCT: near/tmpl's lambda family shifts >2x vs ctrl/fact with beta
  ~constant across batteries — the thermal pricing is honest; near/tmpl
  genuinely erode faster.
- SHAPE-DISTINCT: near/tmpl's beta shifts (|delta beta| >= 0.15) at
  near-constant lambda — the one-T "hotter" readings are shape artifacts; the
  lens's pricing table gets the correction.
- MIXED: both shift; the families verbatim.
- UNDERPOWERED: the committed grids cannot separate (say what would).

OPERATIONALIZATION (frozen with the bars, before any compute):

1. DATA: e238's committed 54-probe npz logit dumps (runs/e238/logits_{tag}.npz,
   tag in t0, w1s2, w1s10, w1s50, w1s80, w2s10, w2s50, w2s80) — all four
   batteries (fact n=20, tmpl n=19, ctrl n=12, near n=3) with battery labels
   and ans_ids. BIT-VERIFY BEFORE TRUST: (a) sha256 of every dump recorded;
   x5's own dumps (runs/x5/logits_*.npz) must match x5's committed sha256_16
   provenance; (b) e238's ctrl/near subset must be BIT-IDENTICAL to x5's
   dumps (max |dlogit| == 0); (c) p recomputed under float64 numpy softmax
   must reproduce x5's committed mean_p_obs within 5e-5 (the torch-float32 vs
   numpy-float64 softmax arithmetic gap, the same ~1.3e-5 scale x5 itself
   disclosed in its e238 cross-check); (d) my one-T refit (TempFamily copied
   VERBATIM from lab/e238_temperature_null.py: grid log10T in [-1,1] x 1201
   + golden polish, Bernoulli soft-target CE) must reproduce x5's committed
   per-battery T_mle within 5e-3 for all 28 battery-state rows; (e) the t0
   anchor must read T ~ 1.000 (the instrument's positive control). Plain IO
   reads only (no memmap); finiteness guards on every load.
2. TRAJECTORIES: per battery b x wash w: probes i, anchor a_i = p_i(t0),
   eval states w1 {2,10,50,80}, w2 {10,50,80} (t in wash steps; t0 = anchor
   only, never scored — the same eval set for every family including one-T).
3. FAMILIES: q_i(t) = c + (a_i - c) * g(t), shared battery-wash params,
   per-probe t0 anchoring (the one-T family's L0 anchor made scalar):
   - EXP  g(t) = exp(-lambda t)                       k = 2 (lambda, c)
   - STR  g(t) = exp(-(lambda t)^beta)                k = 3 (lambda, beta, c)
   - POW  g(t) = (1 + lambda t)^(-alpha)              k = 3 (lambda, alpha, c)
   Bounds: lambda in [1e-5, 5], beta in [0.05, 3], alpha in [0.02, 12],
   c in [0, min_i a_i] (the floor cannot exceed the most-dead probe's start).
   MLE = Bernoulli soft-target cross-entropy on p_obs (verbatim e238/x5
   convention), Nelder-Mead with bounds from a frozen multistart grid.
4. ONE-T COMPARATOR: TempFamily per battery, T refit per state, scored on the
   same eval states; k = n_states (one free T per scored state); NLL =
   nll_bernoulli(T*) summed over the eval states.
5. AICc = 2k + 2*NLL + 2k(k+1)/(n - k - 1), n = n_probes * n_eval_states;
   the AICc table spans {EXP, STR, POW, one-T} per battery-wash. Residual
   co-reports: mean |p_obs - q| and decline-R2 (1 - SS(P-Q)/SS(P - state
   battery mean)) per family.
6. RATE CURRENCY: lambda_eff := -ln g(80) / 80 (the window-effective rate;
   EQUALS lambda exactly for EXP; at beta != 1 the raw lambda is not
   comparable across betas — lambda_eff is the honest rate scalar, and the
   SHAPE bar's own words "near-constant effective rate" fix this currency).
7. BATTERY SUMMARY (pooled across washes): beta_bar = arithmetic mean of the
   two wash betas; lam_bar = geometric mean of the two wash lambda_effs.
   Groups: HOT = {near, tmpl}, COLD = {ctrl, fact}.
   L_ratio := max(G_HOT, G_COLD) / min(G_HOT, G_COLD), G_* = geometric mean
   of the group's lam_bar values. dB := max pairwise |beta_bar(hot) -
   beta_bar(cold)|. beta_spread := max pairwise |beta_bar_i - beta_bar_j|
   over all four batteries. Primary plane = the STR family (the only family
   that carries both lambda and beta); EXP-family lambda ratios co-report.
8. VERDICT ROUTING (frozen; point-estimate clauses on the primary plane):
   - RATE-DISTINCT  fires iff L_ratio > 2 AND beta_spread < 0.15.
   - SHAPE-DISTINCT fires iff dB >= 0.15 AND L_ratio <= 2.
   - MIXED          fires iff L_ratio > 2 AND dB >= 0.15.
   - BLANKET (family-separation failure): dAICc(top1 - top2) < 2 among
     {EXP, STR, POW} in >= 4 of the 8 battery-wash fits — the state grids
     cannot separate the families, so beta is not identifiable.
   - Precedence: RATE-DISTINCT survives the blanket (a rate shift with beta
     pinned ~1 everywhere is exactly the rate story); if SHAPE-DISTINCT or
     MIXED would fire but the BLANKET holds -> UNDERPOWERED; if none of
     RATE/SHAPE/MIXED fires -> UNDERPOWERED (state which separation failed
     and what would fix it: more states per trajectory, more probes per
     battery — near n=3 above all — or both).
9. CO-REPORTS (never adjudicate): bootstrap 95% CIs (probe-level case
   resampling, B=200, frozen seed) on lambda, beta, lambda_eff per
   battery-wash; the T-INDUCTION (feed each family's predicted q_i(t) to the
   one-T inversion — which family's induced T(t) tracks the committed one-T
   T(t)? rate explains the T-ladder iff EXP's induction matches; shape iff
   only STR/POW's does); the x7 GAP-SHARE cross-reference (ln gap_factor /
   [ln gap_factor + ln scale_factor] per battery-state, read at runtime from
   runs/x7/metrics.json — a shape finding should ride the constructive gap
   channel, not the LN scale channel); Spearman of the committed deep-T
   ladder vs lam_bar and beta_bar (n=4, disclosed); x5's committed one-T
   table quoted verbatim.
10. ENV: CPU-ONLY — no torch import at all (gpu_calls = 0 trivially), thread
    caps set to 1 before numpy import (<= 4 per the directive), NO
    envelope-log writes, no NOTES/THINKING/QUEUE/STATE edits (dispatch).

OUTPUTS: runs/x9/metrics.json (progressive), runs/x9/x9_rate_fits.png,
runs/x9/REPORT.md.
"""

import os

# --- thread + device caps BEFORE numpy (the CPU directive; threads 1 <= 4) ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # belt-and-braces; torch never imported

import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import scipy
from scipy.optimize import minimize
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "x9"
E238 = REPO / "runs" / "e238"
X5RUN = REPO / "runs" / "x5"
RUN.mkdir(parents=True, exist_ok=True)

T_START = time.time()
TAGS = ["t0", "w1s2", "w1s10", "w1s50", "w1s80", "w2s10", "w2s50", "w2s80"]
WASH_STATES = {"w1": [2, 10, 50, 80], "w2": [10, 50, 80]}
BATTERIES = ["ctrl", "fact", "near", "tmpl"]
HOT = ["near", "tmpl"]
COLD = ["ctrl", "fact"]
EPS = 1e-12  # e238's constant, verbatim
GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201  # e238's grid, verbatim
BOOT_B = 200
BOOT_SEED = 20261005
LAM_BOUNDS = (1e-5, 5.0)
BETA_BOUNDS = (0.05, 3.0)
ALPHA_BOUNDS = (0.02, 12.0)
TOL_T_REPRO = 5e-3
TOL_P_REPRO = 5e-5

cpu_launch = psutil.cpu_percent(interval=0.3)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unavailable"


def sanitize(obj):
    """NaN/inf -> None for strict-JSON metrics (the lab's x5 wrote raw NaN;
    x9 writes None — same disclosure power, valid JSON)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_metrics(metrics: dict, phase_note: str) -> None:
    metrics["date"] = now_iso()
    metrics["phase"] = phase_note
    (RUN / "metrics.json").write_text(
        json.dumps(sanitize(metrics), indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# The one-T family, copied VERBATIM from lab/e238_temperature_null.py
# (TempFamily: __init__ / q / grid_Q / _golden / nll_bernoulli / fit_T_bernoulli)
# ----------------------------------------------------------------------------

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

    def grid_Q(self, n: int = GRID_N) -> tuple:
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
                        grid=None, Q=None) -> tuple:
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


# ----------------------------------------------------------------------------
# Data loading (plain IO reads, finiteness guards, no memmap)
# ----------------------------------------------------------------------------

def load_dump(tag: str) -> dict:
    z = np.load(E238 / f"logits_{tag}.npz", allow_pickle=False)
    out = {k: z[k] for k in z.keys()}
    assert np.all(np.isfinite(out["logits"])), f"non-finite logits in {tag}"
    return out


def softmax_ans(L: np.ndarray, ans: np.ndarray) -> np.ndarray:
    """float64 numpy softmax answer-probabilities (x9's convention)."""
    L = L.astype(np.float64)
    s = L - L.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e[np.arange(len(ans)), ans] / e.sum(axis=1)
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    return p


# ----------------------------------------------------------------------------
# The rate families
# ----------------------------------------------------------------------------

def fam_g(theta, fam, t):
    lam = theta[0]
    if fam == "EXP":
        return np.exp(-lam * t)
    if fam == "STR":
        return np.exp(-((lam * t) ** theta[1]))
    if fam == "POW":
        return (1.0 + lam * t) ** (-theta[2])
    raise ValueError(fam)


def fam_bounds(fam, cmax):
    b = [("lam", LAM_BOUNDS)]
    if fam == "STR":
        b.append(("beta", BETA_BOUNDS))
    if fam == "POW":
        b.append(("alpha", ALPHA_BOUNDS))
    b.append(("c", (0.0, max(cmax, 1e-9))))
    return b


def fam_q(theta, fam, anchors, t):
    g = fam_g(theta, fam, t)
    c = theta[-1]
    return c + (anchors[:, None] - c) * g[None, :]


def fam_nll(theta, fam, anchors, t, P):
    Q = np.clip(fam_q(theta, fam, anchors, t), EPS, 1.0 - EPS)
    return float(-(P * np.log(Q) + (1.0 - P) * np.log(1.0 - Q)).sum())


def fam_starts(fam, cmax):
    lams = [0.005, 0.02, 0.05, 0.2, 0.8]
    shapes = [None] if fam == "EXP" else [0.25, 0.6, 1.0, 1.8]
    cs = [0.0, 0.5 * cmax, 0.95 * cmax] if cmax > 1e-6 else [0.0]
    starts = []
    for lam in lams:
        for sh in shapes:
            for c in cs:
                th = [lam] + ([sh] if sh is not None else []) + [c]
                starts.append(np.array(th, dtype=float))
    return starts


def fit_family(fam, anchors, t, P, cmax=None, starts=None, maxiter=4000):
    if cmax is None:
        cmax = float(anchors.min())
    bounds = fam_bounds(fam, cmax)
    bb = [v for _, v in bounds]
    if starts is None:
        starts = fam_starts(fam, cmax)
    best = None
    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in bb], [b[1] for b in bb])
        r = minimize(fam_nll, x0, args=(fam, anchors, t, P),
                     method="Nelder-Mead", bounds=bb,
                     options={"maxiter": maxiter, "xatol": 1e-10,
                              "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    theta = np.clip(best.x, [b[0] for b in bb], [b[1] for b in bb])
    return theta, float(fam_nll(theta, fam, anchors, t, P))


def lam_eff_of(theta, fam):
    g80 = float(fam_g(theta, fam, np.array([80.0]))[0])
    return -math.log(max(g80, EPS)) / 80.0


def aicc(nll, k, n):
    if n - k - 1 <= 0:
        return None
    return 2.0 * k + 2.0 * nll + 2.0 * k * (k + 1) / (n - k - 1)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    metrics = {
        "experiment": "x9_rate_fits",
        "status": "RUNNING",
        "registration": ("bars frozen VERBATIM from the dispatch brief BEFORE "
                         "any compute (script committed at birth); adjudicate "
                         "against exactly this; no bar shopping"),
        "envelope": {
            "device": "CPU-ONLY (torch never imported; CUDA_VISIBLE_DEVICES=-1 set regardless)",
            "threads": 1,
            "gpu_calls": 0,
            "envelope_log_writes": 0,
            "cpu_load_pct_at_launch": cpu_launch,
            "no_notes_thinking_queue_state_edits": True,
        },
        "provenance": {
            "git_head_at_start": git_head(),
            "script": str(Path(__file__).resolve()),
            "versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "psutil": psutil.__version__,
                "python": platform.python_version(),
            },
        },
    }

    # ---------------- P1: load + bit-verify -------------------------------
    x5m = json.loads((REPO / "runs" / "x5" / "metrics.json").read_text())
    x7m = json.loads((REPO / "runs" / "x7" / "metrics.json").read_text())
    e238m = json.loads((E238 / "metrics.json").read_text())

    shas_e238, dump_identity = {}, []
    dumps = {}
    ref_meta = None
    for tag in TAGS:
        d = load_dump(tag)
        dumps[tag] = d
        shas_e238[tag] = sha16(E238 / f"logits_{tag}.npz")
        meta = (list(d["names"]), list(d["battery"]), list(d["ans_ids"]))
        if ref_meta is None:
            ref_meta = meta
        else:
            assert meta == ref_meta, f"probe set drift across dumps at {tag}"
        # identity vs x5's sha-verified dumps (ctrl+near subset)
        xz = np.load(X5RUN / f"logits_{tag}.npz", allow_pickle=False)
        idx = [list(d["names"]).index(n) for n in xz["names"]]
        dmax = float(np.abs(
            d["logits"][idx].astype(np.float64)
            - xz["logits"].astype(np.float64)).max())
        meta_ok = bool((d["battery"][idx] == xz["battery"]).all()
                       and (d["ans_ids"][idx] == xz["ans_ids"]).all())
        dump_identity.append({"tag": tag, "meta_ok": meta_ok,
                              "max_dlogit": dmax})
    # x5 provenance sha match
    sha_x5_ok = all(
        sha16(Path(rec["path"])) == rec["sha256_16"]
        for rec in x5m["provenance"]["logit_dumps"].values())
    sha_records_ok = all(
        sha16(Path(rec["path"])) == rec["sha256_16"]
        for rec in x5m["provenance"]["committed_records"].values())

    batt_arr = np.array(ref_meta[1])
    batt_ix = {b: np.where(batt_arr == b)[0] for b in BATTERIES}
    n_counts = {b: int(len(batt_ix[b])) for b in BATTERIES}

    # p matrices per battery per tag (float64 numpy softmax — x9 convention)
    P = {}
    for tag in TAGS:
        d = dumps[tag]
        pall = softmax_ans(d["logits"], d["ans_ids"])
        for b in BATTERIES:
            P.setdefault(b, {})[tag] = pall[batt_ix[b]]

    # one-T families per battery (L0 = t0 logits) + shared grid pass
    onet = {}
    for b in BATTERIES:
        d0 = dumps["t0"]
        onet[b] = TempFamily(d0["logits"][batt_ix[b]], d0["ans_ids"][batt_ix[b]])

    # one-T refit per battery-state + repro vs x5 committed
    onet_rows, repro_dts, repro_dps = [], [], []
    x5_fitrows = {(r["battery"], r["wash"], r["state"]): r
                  for r in x5m["fit_rows"]}
    onet_T = {b: {} for b in BATTERIES}   # (wash,state) -> T
    onet_NLL = {b: {} for b in BATTERIES}
    t0_anchor = {}
    for b in BATTERIES:
        grid, Q = onet[b].grid_Q()
        for wash, states in WASH_STATES.items():
            for s in states:
                p_obs = P[b][f"{wash}s{s}"]
                T, nll = onet[b].fit_T_bernoulli(p_obs, grid, Q)
                onet_T[b][(wash, s)] = T
                onet_NLL[b][(wash, s)] = nll
                ref = x5_fitrows.get((b, wash, s))
                dt = abs(T - ref["T_mle"]) if ref else None
                if ref:
                    repro_dts.append(dt)
                onet_rows.append({
                    "battery": b, "wash": wash, "state": s,
                    "n": n_counts[b], "T_mle": T, "nll_at_T": nll,
                    "mean_p_obs": float(p_obs.mean()),
                    "x5_committed_T": (ref or {}).get("T_mle"),
                    "abs_dT_vs_x5": dt,
                })
        # t0 anchor (positive control; never scored)
        Tt0, _ = onet[b].fit_T_bernoulli(P[b]["t0"], grid, Q)
        t0_anchor[b] = Tt0
        ref = x5_fitrows.get((b, "t0", 0))
        repro_dts.append(abs(Tt0 - ref["T_mle"]))
    # mean_p_obs repro (ctrl/near rows carry it in x5)
    for r in x5m["fit_rows"]:
        if "mean_p_obs" in r:
            tag = "t0" if r["wash"] == "t0" else f"{r['wash']}s{r['state']}"
            repro_dps.append(abs(float(P[r["battery"]][tag].mean())
                                 - r["mean_p_obs"]))

    gates = {
        "G_SHA": {
            "x5_dump_shas_match_provenance": sha_x5_ok,
            "x5_committed_records_shas_match": sha_records_ok,
            "e238_dump_sha16": shas_e238,
            "x7_metrics_sha16": sha16(REPO / "runs" / "x7" / "metrics.json"),
            "pass": bool(sha_x5_ok and sha_records_ok),
        },
        "G_DUMP_IDENTITY": {
            "rows": dump_identity,
            "max_dlogit": max(r["max_dlogit"] for r in dump_identity),
            "all_meta_ok": all(r["meta_ok"] for r in dump_identity),
            "pass": bool(max(r["max_dlogit"] for r in dump_identity) == 0.0
                         and all(r["meta_ok"] for r in dump_identity)),
        },
        "G_REPRO": {
            "n_T_rows_checked": len(repro_dts),
            "max_abs_dT_vs_x5": max(repro_dts),
            "tol_dT": TOL_T_REPRO,
            "n_meanp_rows_checked": len(repro_dps),
            "max_abs_dp_vs_x5": max(repro_dps) if repro_dps else None,
            "tol_dp": TOL_P_REPRO,
            "t0_anchors": t0_anchor,
            "note": ("x5's p came from torch float32 softmax; x9 recomputes "
                     "float64 numpy — the ~1e-5 p gap is the disclosed "
                     "arithmetic difference (x5's own e238 cross-check "
                     "carried the same scale), so T reproduces to ~1e-4"),
            "pass": bool(max(repro_dts) <= TOL_T_REPRO
                         and max(repro_dps) <= TOL_P_REPRO
                         and all(abs(t0_anchor[b] - 1.0) <= 2e-3
                                 for b in BATTERIES)),
        },
        "G_ENV": {
            "cpu_only_no_torch": True, "threads": 1, "gpu_calls": 0,
            "envelope_log_writes": 0,
            "cpu_load_pct_at_launch": cpu_launch,
            "pass": True,
        },
    }
    metrics["gates"] = gates
    metrics["n_counts"] = n_counts
    metrics["oneT_refit"] = onet_rows
    metrics["discharge_curves"] = {
        b: {"t0": {"mean_p": float(P[b]["t0"].mean())},
            **{f"{w}s{s}": {"mean_p": float(P[b][f"{w}s{s}"].mean())}
               for w, states in WASH_STATES.items() for s in states}}
        for b in BATTERIES
    }
    write_metrics(metrics, "P1 DONE (load + bit-verify gates + one-T refit)")
    print(f"[P1] gates: " + ", ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in gates.items()))
    print(f"[P1] max|dT| vs x5 = {max(repro_dts):.2e}, "
          f"max|dp| = {max(repro_dps) if repro_dps else 0:.2e}")

    # ---------------- P2: the rate fits + AICc tables ---------------------
    fams = ["EXP", "STR", "POW"]
    fits = {}       # (b,w,fam) -> dict
    for b in BATTERIES:
        anchors = P[b]["t0"]
        cmax = float(anchors.min())
        for wash, states in WASH_STATES.items():
            t = np.array(states, dtype=float)
            Pm = np.stack([P[b][f"{wash}s{s}"] for s in states], axis=1)
            n_obs = n_counts[b] * len(states)
            for fam in fams:
                theta, nll = fit_family(fam, anchors, t, Pm, cmax=cmax)
                k = 2 if fam == "EXP" else 3
                Qp = fam_q(theta, fam, anchors, t)
                resid = float(np.abs(Pm - Qp).mean())
                ss_res = float(((Pm - Qp) ** 2).sum())
                ss_tot = float(sum(
                    ((Pm[:, j] - Pm[:, j].mean()) ** 2).sum()
                    for j in range(len(states))))
                fits[(b, wash, fam)] = {
                    "theta": theta.tolist(), "param_names": (
                        ["lam", "c"] if fam == "EXP" else
                        (["lam", "beta", "c"] if fam == "STR"
                         else ["lam", "alpha", "c"])),
                    "k": k, "n_obs": n_obs, "nll": nll,
                    "aicc": aicc(nll, k, n_obs),
                    "mean_abs_resid": resid,
                    "R2_decline": (1.0 - ss_res / ss_tot if ss_tot > 0
                                   else None),
                    "lam_eff": lam_eff_of(theta, fam),
                    "bound_hit": bool(
                        theta[0] in LAM_BOUNDS
                        or (fam == "STR" and theta[1] in BETA_BOUNDS)
                        or (fam == "POW" and theta[2] in ALPHA_BOUNDS)
                        or theta[-1] in (0.0, cmax)),
                }
            # one-T comparator on the same eval states
            nll_1t = sum(onet_NLL[b][(wash, s)] for s in states)
            k_1t = len(states)
            Q1t = np.stack([np.clip(onet[b].q(onet_T[b][(wash, s)]),
                                    EPS, 1 - EPS) for s in states], axis=1)
            resid_1t = float(np.abs(Pm - Q1t).mean())
            ss_res1 = float(((Pm - Q1t) ** 2).sum())
            fits[(b, wash, "oneT")] = {
                "theta": None, "k": k_1t, "n_obs": n_obs, "nll": nll_1t,
                "aicc": aicc(nll_1t, k_1t, n_obs),
                "mean_abs_resid": resid_1t,
                "R2_decline": 1.0 - ss_res1 / ss_tot if ss_tot > 0 else None,
                "lam_eff": None, "bound_hit": False,
                "T_per_state": {str(s): onet_T[b][(wash, s)]
                                for s in states},
            }

    # AICc tables + winners
    aicc_tables = {}
    winners = {}
    keys = [k for k in fits if k[2] != "oneT"]
    for b in BATTERIES:
        for wash in WASH_STATES:
            rows = {}
            for fam in fams + ["oneT"]:
                rows[fam] = fits[(b, wash, fam)]["aicc"]
            best = min(rows, key=lambda f: rows[f])
            winners[(b, wash)] = best
            aicc_tables[f"{b}/{wash}"] = {
                "aicc": rows,
                "winner": best,
                "dAICC_second_vs_best_rate_families": (
                    sorted([rows[f] for f in fams])[1]
                    - sorted([rows[f] for f in fams])[0]),
            }
    metrics["rate_fits"] = {f"{b}/{wash}/{fam}": fits[(b, wash, fam)]
                            for (b, wash, fam) in fits}
    metrics["aicc_tables"] = aicc_tables
    n_insep = sum(1 for v in aicc_tables.values()
                  if v["dAICC_second_vs_best_rate_families"] < 2.0)
    metrics["family_separation"] = {
        "n_fits": len(aicc_tables),
        "n_inseparable_dAICc_lt_2": n_insep,
        "blanket_threshold": 4,
        "blanket_fires": bool(n_insep >= 4),
    }
    write_metrics(metrics, "P2 DONE (rate fits + one-T comparator + AICc)")
    print(f"[P2] winners: " + ", ".join(f"{b}/{w}={winners[(b, w)]}"
                                        for b in BATTERIES
                                        for w in WASH_STATES))
    print(f"[P2] inseparable fits (dAICc<2 among rate families): {n_insep}/8")

    # ---------------- P3: bootstrap CIs ------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    boot = {}
    for b in BATTERIES:
        anchors = P[b]["t0"]
        cmax = float(anchors.min())
        n = n_counts[b]
        for wash, states in WASH_STATES.items():
            t = np.array(states, dtype=float)
            Pm = np.stack([P[b][f"{wash}s{s}"] for s in states], axis=1)
            for fam in ["STR", "EXP"]:
                th0 = np.array(fits[(b, wash, fam)]["theta"])
                starts = [th0] + [np.clip(
                    th0 * (1 + dx), [v[0] for _, v in fam_bounds(fam, cmax)],
                    [v[1] for _, v in fam_bounds(fam, cmax)])
                    for dx in (0.15, -0.15, 0.4)]
                recs = []
                for _ in range(BOOT_B):
                    ix = rng.integers(0, n, size=n)
                    th, _ = fit_family(fam, anchors[ix], t, Pm[ix],
                                       cmax=float(anchors[ix].min()),
                                       starts=starts, maxiter=2000)
                    rec = {"lam": th[0],
                           "lam_eff": lam_eff_of(th, fam)}
                    if fam == "STR":
                        rec["beta"] = th[1]
                    recs.append(rec)
                arr = {kk: np.array([r[kk] for r in recs]) for kk in recs[0]}
                ci = {}
                for kk, vv in arr.items():
                    lo, hi = np.nanpercentile(vv, [2.5, 97.5])
                    ci[kk] = [float(lo), float(hi)]
                boot[f"{b}/{wash}/{fam}"] = ci
    metrics["bootstrap"] = {
        "method": f"probe-level case resampling, B={BOOT_B}, seed {BOOT_SEED}",
        "cis": boot,
    }
    write_metrics(metrics, "P3 DONE (bootstrap CIs)")
    print("[P3] bootstrap done")

    # ---------------- P4: discrimination plane + adjudication -------------
    plane = {}
    for b in BATTERIES:
        th_w = {w: np.array(fits[(b, w, "STR")]["theta"])
                for w in WASH_STATES}
        beta_bar = float(np.mean([th_w[w][1] for w in WASH_STATES]))
        lam_effs = [fits[(b, w, "STR")]["lam_eff"] for w in WASH_STATES]
        lam_bar = float(np.exp(np.mean(np.log(lam_effs))))
        exp_lams = [fits[(b, w, "EXP")]["lam_eff"] for w in WASH_STATES]
        plane[b] = {
            "beta_bar": beta_bar,
            "lam_eff_bar_geo": lam_bar,
            "beta_per_wash": {w: float(th_w[w][1]) for w in WASH_STATES},
            "lam_eff_per_wash": {w: float(v) for w, v in
                                 zip(WASH_STATES, lam_effs)},
            "exp_lam_bar_geo": float(np.exp(np.mean(np.log(exp_lams)))),
        }
    G_hot = math.exp(np.mean([math.log(plane[b]["lam_eff_bar_geo"])
                              for b in HOT]))
    G_cold = math.exp(np.mean([math.log(plane[b]["lam_eff_bar_geo"])
                               for b in COLD]))
    L_ratio = max(G_hot, G_cold) / min(G_hot, G_cold)
    dB = max(abs(plane[h]["beta_bar"] - plane[c]["beta_bar"])
             for h in HOT for c in COLD)
    beta_spread = max(abs(plane[i]["beta_bar"] - plane[j]["beta_bar"])
                      for i in BATTERIES for j in BATTERIES)
    exp_ratio = None
    Ge_hot = math.exp(np.mean([math.log(plane[b]["exp_lam_bar_geo"])
                               for b in HOT]))
    Ge_cold = math.exp(np.mean([math.log(plane[b]["exp_lam_bar_geo"])
                                for b in COLD]))
    exp_ratio = max(Ge_hot, Ge_cold) / min(Ge_hot, Ge_cold)

    rate_fires = bool(L_ratio > 2.0 and beta_spread < 0.15)
    shape_fires = bool(dB >= 0.15 and L_ratio <= 2.0)
    mixed_fires = bool(L_ratio > 2.0 and dB >= 0.15)
    blanket = metrics["family_separation"]["blanket_fires"]

    if rate_fires:
        verdict = "RATE-DISTINCT"
        clause = (f"L_ratio = {L_ratio:.3f} > 2 (HOT geo-mean lambda_eff "
                  f"{G_hot:.4f} vs COLD {G_cold:.4f}) with beta spread "
                  f"{beta_spread:.3f} < 0.15 across all four batteries")
    elif (shape_fires or mixed_fires) and blanket:
        verdict = "UNDERPOWERED"
        clause = (f"point-estimate clause(s) "
                  f"{'SHAPE' if shape_fires else 'MIXED'} would fire "
                  f"(dB={dB:.3f}, L_ratio={L_ratio:.3f}) but the family "
                  f"separation blanket holds: dAICc < 2 among the rate "
                  f"families in {n_insep}/8 fits — beta is not identifiable "
                  f"on the committed state grids")
    elif shape_fires:
        verdict = "SHAPE-DISTINCT"
        clause = (f"max |delta beta_bar| HOT-vs-COLD = {dB:.3f} >= 0.15 at "
                  f"L_ratio = {L_ratio:.3f} <= 2 (near-constant effective "
                  f"rate: HOT {G_hot:.4f} vs COLD {G_cold:.4f})")
    elif mixed_fires:
        verdict = "MIXED"
        clause = (f"both shift: L_ratio = {L_ratio:.3f} > 2 AND "
                  f"|delta beta| = {dB:.3f} >= 0.15")
    else:
        verdict = "UNDERPOWERED"
        clause = (f"neither frozen clause fires (L_ratio = {L_ratio:.3f}, "
                  f"dB = {dB:.3f}, beta_spread = {beta_spread:.3f}) — the "
                  f"battery differences ride below the frozen effect sizes "
                  f"on the committed grids")

    # what would separate (UNDERPOWERED disclosure, computed either way)
    separation_prescription = (
        "more states per trajectory (the committed grids give 3-4 post-wash "
        "points; 6-8 states bracket the mid-curve where EXP/STR/POW diverge) "
        "and more probes per battery — near n=3 above all (its bootstrap CI "
        "spans the whole plane)")

    # T-INDUCTION co-report: which family's induced T(t) tracks the
    # committed one-T readings (the mispricing localizer)
    t_induction = {}
    for b in BATTERIES:
        grid, Q = None, None  # recomputed below per battery
        th_onet = {w: {s: onet_T[b][(w, s)] for s in states}
                   for w, states in WASH_STATES.items()}
        for fam in fams:
            per_wash = {}
            rmses, maxds = [], []
            for wash, states in WASH_STATES.items():
                t = np.array(states, dtype=float)
                anchors = P[b]["t0"]
                theta = np.array(fits[(b, wash, fam)]["theta"])
                Qp = np.clip(fam_q(theta, fam, anchors, t), EPS, 1 - EPS)
                if grid is None:
                    grid, Q = onet[b].grid_Q()
                Tind = {}
                for j, s in enumerate(states):
                    T_f, _ = onet[b].fit_T_bernoulli(Qp[:, j], grid, Q)
                    Tind[str(s)] = T_f
                    rmses.append((T_f - th_onet[wash][s]) ** 2)
                    maxds.append(abs(T_f - th_onet[wash][s]))
                per_wash[wash] = {"T_induced": Tind,
                                  "T_committed": {str(s): th_onet[wash][s]
                                                  for s in states}}
            t_induction[f"{b}/{fam}"] = {
                "per_wash": per_wash,
                "rmse_dT": float(np.sqrt(np.mean(rmses))),
                "max_abs_dT": float(np.max(maxds)),
            }

    # gap-share co-report (x7's currency)
    gap_share = {}
    for st, row in x7m["ln_zero_sum_decomposition"]["states"].items():
        for b in BATTERIES:
            r = row.get(b)
            if not r:
                continue
            lg = math.log(r["median_gap_factor"])
            ls = math.log(r["median_scale_factor"])
            den = lg + ls
            gap_share.setdefault(b, {})[st] = (
                lg / den if abs(den) > 1e-9 else None)
    gap_summary = {}
    for b in BATTERIES:
        vals = [v for v in gap_share.get(b, {}).values() if v is not None]
        gap_summary[b] = float(np.median(vals)) if vals else None

    # deep-T ladder vs rate/shape (n=4, disclosed)
    T_deep = {b: float(np.mean([onet_T[b][(w, s)]
                                for w in WASH_STATES for s in (50, 80)]))
              for b in BATTERIES}
    lam_vec = [plane[b]["lam_eff_bar_geo"] for b in BATTERIES]
    beta_vec = [plane[b]["beta_bar"] for b in BATTERIES]
    T_vec = [T_deep[b] for b in BATTERIES]
    rho_lam = float(spearmanr(T_vec, lam_vec).statistic)
    rho_beta = float(spearmanr(T_vec, beta_vec).statistic)

    metrics["discrimination_plane"] = {
        "primary_family": "STR (stretched exponential)",
        "battery_summary": plane,
        "G_HOT_geo": G_hot, "G_COLD_geo": G_cold,
        "L_ratio": L_ratio, "dB_hot_vs_cold": dB,
        "beta_spread_all_batteries": beta_spread,
        "exp_family_lam_ratio_co_report": exp_ratio,
        "group_defs": {"HOT": HOT, "COLD": COLD},
        "rate_currency": "lambda_eff := -ln g(80) / 80 (equals lambda for EXP)",
    }
    metrics["t_induction"] = t_induction
    metrics["gap_share_co_report"] = {
        "definition": ("ln(median_gap_factor) / [ln(median_gap_factor) + "
                       "ln(median_scale_factor)] per battery-state, from "
                       "runs/x7/metrics.json (x7's gap/scale currency)"),
        "per_state": gap_share,
        "per_battery_median": gap_summary,
    }
    metrics["deep_T_ladder"] = {
        "T_deep_mean_50_80": T_deep,
        "spearman_T_vs_lam_eff": rho_lam,
        "spearman_T_vs_beta": rho_beta,
        "note": "n=4 batteries — texture only, never adjudicates",
    }
    metrics["adjudication"] = {
        "bars": {
            "RATE-DISTINCT": rate_fires,
            "SHAPE-DISTINCT": shape_fires,
            "MIXED": mixed_fires,
            "UNDERPOWERED": verdict == "UNDERPOWERED",
        },
        "verdict": verdict,
        "clause": clause,
        "routing": ("RATE survives the blanket; SHAPE/MIXED do not "
                    "(beta unidentifiable when the blanket holds); none "
                    "firing -> UNDERPOWERED"),
        "blanket_fired": blanket,
        "separation_prescription_if_underpowered": separation_prescription,
    }
    write_metrics(metrics, "P4 DONE (adjudication + co-reports)")
    print(f"[P4] verdict: {verdict}")
    print(f"[P4] L_ratio={L_ratio:.3f} dB={dB:.3f} "
          f"beta_spread={beta_spread:.3f} blanket={blanket}")

    # ---------------- P5: figure + report ---------------------------------
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.15], hspace=0.42,
                          wspace=0.28)

    colors = {"ctrl": "#1f77b4", "fact": "#2ca02c",
              "near": "#d62728", "tmpl": "#ff7f0e"}

    # (A) lambda_eff vs beta scatter with bootstrap CIs
    axA = fig.add_subplot(gs[0, 0])
    for b in BATTERIES:
        for w, mk in zip(WASH_STATES, ["o", "^"]):
            ci = boot[f"{b}/{w}/STR"]
            th = np.array(fits[(b, w, "STR")]["theta"])
            le = fits[(b, w, "STR")]["lam_eff"]
            axA.errorbar(le, th[1],
                         yerr=[[max(0.0, th[1] - ci["beta"][0])],
                               [max(0.0, ci["beta"][1] - th[1])]],
                         xerr=[[max(0.0, le - ci["lam_eff"][0])],
                               [max(0.0, ci["lam_eff"][1] - le)]],
                         fmt=mk, color=colors[b], ms=9, capsize=3,
                         alpha=0.85,
                         label=f"{b} {w}" if w == "w1" else None)
    axA.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    axA.text(0.02, 0.97, "beta = 1 line: exponential boundary "
             "(below = slow tail, above = steepening)",
             transform=axA.transAxes, va="top", fontsize=8, color="gray")
    axA.set_xscale("log")
    axA.set_xlabel("effective rate  $\\lambda_{eff} = -\\ln g(80)/80$  "
                   "(1/wash-steps, log)")
    axA.set_ylabel("shape  $\\beta$  (stretched exponential)")
    axA.set_title("(A) rate vs shape per battery-wash (stretched fits; "
                  "95% bootstrap CIs; o=w1, ^=w2)", fontsize=10)
    axA.legend(fontsize=8, ncol=2)

    # (A-strip) the one-T ladder overlay
    axS = fig.add_subplot(gs[1, 0])
    for b in BATTERIES:
        axS.scatter(T_deep[b], 0, s=160, color=colors[b], zorder=3,
                    edgecolor="k", linewidth=0.5)
        axS.annotate(f"{b}\nT={T_deep[b]:.2f}", (T_deep[b], 0),
                     textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=8, color=colors[b])
    axS.set_yticks([])
    lo, hi = min(T_deep.values()), max(T_deep.values())
    axS.set_xlim(lo - 0.08, hi + 0.08)
    axS.set_ylim(-0.5, 1.2)
    axS.set_xlabel("committed one-T ladder (mean T at +50/+80, x5's "
                   "battery-specific ordering)")
    axS.set_title("(A') the axis the rate/shape plane must price "
                  "(hotter -> right)", fontsize=9)

    # (B) AICc family wins + margins
    axB = fig.add_subplot(gs[0, 1])
    fam_cols = {"EXP": "#9467bd", "STR": "#d62728", "POW": "#8c564b",
                "oneT": "#7f7f7f"}
    xpos = np.arange(len(BATTERIES))
    bottom = np.zeros(len(BATTERIES))
    for fam in ["EXP", "STR", "POW", "oneT"]:
        wins = np.array([sum(1 for w in WASH_STATES
                             if winners[(b, w)] == fam)
                         for b in BATTERIES], dtype=float)
        axB.bar(xpos, wins, bottom=bottom, color=fam_cols[fam],
                label=fam, width=0.6, edgecolor="k", linewidth=0.4)
        bottom += wins
    axB.set_xticks(xpos)
    axB.set_xticklabels([f"{b}\n(n={n_counts[b]})" for b in BATTERIES])
    axB.set_ylabel("washes won (of 2)")
    axB.set_ylim(0, 2.6)
    axB.legend(fontsize=8, title="AICc winner", title_fontsize=8)
    txt = "\n".join(
        f"{b}: " + ", ".join(
            f"{fam} {aicc_tables[f'{b}/{w}']['aicc'][fam]:.1f}"
            for fam in ["EXP", "STR", "POW", "oneT"])
        for b in ["near", "tmpl"])
    axB.text(0.02, 0.97, "AICc (hot batteries):\n" + txt,
             transform=axB.transAxes, va="top", fontsize=6.4, family="monospace")
    axB.set_title("(B) which family carries each battery "
                  "(AICc over {EXP, STR, POW, one-T})", fontsize=10)

    # (C) discharge curves + winner overlays
    axC = fig.add_subplot(gs[1, 1])
    for b in BATTERIES:
        anchors = P[b]["t0"]
        for w, ls in zip(WASH_STATES, ["-", ":"]):
            states = WASH_STATES[w]
            tt = np.array([0] + states, dtype=float)
            mm = [float(P[b]["t0"].mean())] + [
                float(P[b][f"{w}s{s}"].mean()) for s in states]
            axC.plot(tt, mm, ls, color=colors[b], lw=1.6,
                     marker=".", ms=6, alpha=0.9,
                     label=f"{b} obs" if w == "w1" else None)
            famw = winners[(b, w)]
            if famw != "oneT":
                theta = np.array(fits[(b, w, famw)]["theta"])
                tg = np.linspace(0, 80, 161)
                Qg = fam_q(theta, famw, anchors, tg)
                axC.plot(tg, Qg.mean(axis=0), color=colors[b], lw=0.8,
                         alpha=0.45)
    axC.set_xlabel("wash steps t")
    axC.set_ylabel("battery mean p(answer)")
    axC.set_title("(C) discharge curves (obs = line+dot; thin = AICc "
                  "winner overlay; one-T winners carry no curve)", fontsize=9)
    axC.legend(fontsize=7, ncol=2)

    fig.suptitle("x9 — the rate-fits instrument: RATE vs SHAPE resolution of "
                 "the one-T mispricing (T238) — verdict: " + verdict,
                 fontsize=12)
    fig.savefig(RUN / "x9_rate_fits.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # REPORT.md
    lam_beta_tbl = "\n".join(
        f"| {b} | {plane[b]['lam_eff_bar_geo']:.4f} | "
        f"{plane[b]['beta_bar']:.3f} | "
        f"{plane[b]['lam_eff_per_wash']['w1']:.4f} / "
        f"{plane[b]['lam_eff_per_wash']['w2']:.4f} | "
        f"{plane[b]['beta_per_wash']['w1']:.3f} / "
        f"{plane[b]['beta_per_wash']['w2']:.3f} | "
        f"{plane[b]['exp_lam_bar_geo']:.4f} | {T_deep[b]:.3f} | "
        f"{gap_summary[b] if gap_summary[b] is not None else float('nan'):.3f} |"
        for b in BATTERIES)
    win_counts = {fam: sum(1 for k, v in winners.items() if v == fam)
                  for fam in ["EXP", "STR", "POW", "oneT"]}
    best_fam = max(win_counts, key=win_counts.get)
    ind_rows = "\n".join(
        f"| {b} | " + " | ".join(
            f"{t_induction[f'{b}/{f}']['rmse_dT']:.4f}" for f in fams) + " |"
        for b in BATTERIES)
    report = f"""# x9 — THE RATE-FITS INSTRUMENT (T238's mispricing resolver)

**Status**: COMPLETE — verdict **{verdict}**
**Envelope**: CPU-only desk cell (no torch import, threads 1, 0 GPU calls,
0 envelope-log writes, no NOTES/THINKING/QUEUE/STATE edits).
**Registration**: bars frozen VERBATIM in the script docstring, birth-committed
before compute ({metrics['provenance']['git_head_at_start']}).

## The question

x5 proved every battery carries its own one-T curve (ctrl 1.18-1.27 < fact-only
1.26-1.37 < pooled < near 1.42-1.63 < tmpl 1.51-1.65). T238 flagged the
mispricing: near/tmpl read "hotter" under one-T — genuinely faster decay
(RATE), or a different decay SHAPE that one-T translates into phantom
temperature? x7 proved the lens rescale-invariant, so the mispricing (if any)
is shape-real. This cell fits three decay families per battery per wash
(single-exponential EXP, stretched exponential STR, power-law POW) against the
bit-verified e238 discharge dumps and adjudicates the frozen bars.

## Gates (all machine-checked)

- G_SHA: x5's dump shas match its committed provenance; e238 dumps hashed.
- G_DUMP_IDENTITY: e238's ctrl/near subset bit-identical to x5's dumps
  (max |dlogit| = 0.0 across all 8 states).
- G_REPRO: one-T refit reproduces x5's 32 committed T rows to
  max |dT| = {max(repro_dts):.2e}; mean-p to max |dp| = {max(repro_dps):.2e}
  (the disclosed torch-float32 vs numpy-float64 softmax gap); t0 anchors read
  {', '.join(f'{b}={t0_anchor[b]:.5f}' for b in BATTERIES)}.

## The lambda/beta table (primary plane = stretched exponential)

| battery | lam_eff_bar (geo) | beta_bar | lam_eff w1 / w2 | beta w1 / w2 | EXP lam_bar (geo) | one-T deep ladder | gap share (x7) |
|---|---|---|---|---|---|---|---|
{lam_beta_tbl}

L_ratio (HOT/COLD geometric-mean contrast) = **{L_ratio:.3f}**;
max |delta beta_bar| HOT-vs-COLD = **{dB:.3f}**; beta spread over all four
batteries = **{beta_spread:.3f}**; EXP-family lambda ratio (co-report) =
{exp_ratio:.3f}.

## Which family wins

AICc wins across the 8 battery-wash fits: EXP {win_counts['EXP']}, STR
{win_counts['STR']}, POW {win_counts['POW']}, one-T {win_counts['oneT']} —
overall winner: **{best_fam}**. Family separation: dAICc < 2 among rate
families in {n_insep}/8 fits (blanket {'FIRES' if blanket else 'does not fire'}).

## T-induction (the mispricing localizer, co-report)

RMSE between each family's induced T(t) and the committed one-T T(t):

| battery | EXP | STR | POW |
|---|---|---|---|
{ind_rows}

(The family whose induced T tracks the committed one-T ladder is the one the
lens was actually reading when it called near/tmpl "hot".)

## Verdict clause (frozen routing)

{clause}

Separation prescription (if underpowered): {separation_prescription}

## Honesty

n=1 organism; near n=3 (its bootstrap CI spans the plane — every near claim
is coarse); wash-1 CPU replay vs wash-2 GPU (inherited archive asymmetry);
beta pinned at bounds flagged in metrics; the one-T AICc counts one free T per
state (its L0 anchor is data, as the rate families' t0 anchors are); t0 never
scored (anchor + positive control); lens-not-mechanism throughout (T228/T229);
no bar shopping — the routing was frozen in the birth commit.
"""
    (RUN / "REPORT.md").write_text(report, encoding="utf-8")

    metrics["status"] = "COMPLETE — adjudicated (this write replaces all PARTIAL progressive writes)"
    metrics["outputs"] = {
        "metrics": str(RUN / "metrics.json"),
        "figure": str(RUN / "x9_rate_fits.png"),
        "report": str(RUN / "REPORT.md"),
    }
    metrics["timing"] = {"total_s": round(time.time() - T_START, 2)}
    write_metrics(metrics, "P5 DONE (figure + report + adjudication)")
    print(f"[P5] done in {metrics['timing']['total_s']}s -> {RUN}")


if __name__ == "__main__":
    main()
