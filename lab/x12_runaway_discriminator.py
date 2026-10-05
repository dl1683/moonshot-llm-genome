"""X12 — THE RUNAWAY DISCRIMINATOR (W043's discriminator; prediction P-x12a
registered in THINKING.md T256) — a CPU-ONLY DESK CELL.

x9 stretched-exponential fits found the FACT battery's committed discharge
curves carry beta = 1.31 (w1 1.2672 / w2 1.3458; bootstrap CIs
[1.137, 2.894] / [1.148, 2.357] — the ONLY battery whose CI excludes 1).
beta > 1 means ACCELERATING decline. TWO LIVE HYPOTHESES:
  H-DRAIN-RUNAWAY — predatory kill: the wash's per-probe damage genuinely
  accelerates as the memory weakens (a runaway drain).
  H-SHAPE-MISMATCH — superposition artifact: a BI-EXPONENTIAL MIXTURE of
  fast and slow probes imitates acceleration in the aggregate curve.
This cell discriminates them with two independent instruments: (i) the
family test (bi-exponential re-fit vs x9's stretched/one-T, AICc) and
(ii) the derivative test (per-step damage vs remaining strength on e228's
committed per-probe margin journal).

BARS — FROZEN VERBATIM FROM THE DISPATCH, BEFORE ANY COMPUTE (no bar
shopping; this docstring is birth-committed before any fitting runs):

  RUNAWAY: bi-exp loses to beta>1 by AICc AND the damage-strength
      anti-correlation is real — Spearman <= -0.5 across states.
  MIXTURE: bi-exp wins by AICc >= 2 — W043's acceleration was
      superposition.
  MIXED / UNDERPOWERED: the remaining contradiction / no-signal cases.

OPERATIONALIZATIONS — FROZEN BEFORE COMPUTE:
  * (i) FAMILY TEST: re-fit the fact battery's committed curves on x9's
    exact data and grid (e238's committed npz logit dumps; anchors a_i =
    p_i(t0); eval states w1 {2,10,50,80}, w2 {10,50,80}; q_i(t) =
    c + (a_i - c) g(t); Bernoulli soft-target CE; Nelder-Mead with bounds
    + x9's multistart, machinery ported BY VALUE from lab/x9_rate_fits.py)
    with the families:
      EXP   g = exp(-lam t)                     k = 2
      STR   g = exp(-(lam t)^beta)              k = 3  (x9's, gated)
      BIEXP g = w exp(-lam1 t) + (1-w) exp(-lam2 t)   k = 4  (NEW)
      one-T comparator (TempFamily, ported VERBATIM) k = n_states
    AICc = 2k + 2NLL + 2k(k+1)/(n-k-1), n = n_probes x n_states (x9's
    formula). BIEXP bounds: lam1, lam2 in [1e-5, 5], w in [0, 1], c in
    [0, min a_i]; starts: lam pairs over {0.002, 0.01, 0.04, 0.16, 0.8}
    (all 25 ordered pairs) x w in {0.2, 0.5, 0.8} x c in {0, 0.5 cmax};
    w at a 0/1 bound flags the EXP-degenerate boundary (bound_hit,
    disclosed).
    dA_w := AICc_BIEXP - AICc_STR per wash (positive = stretched wins).
    A-channel states: STR-wins iff dA_w1 >= 2 AND dA_w2 >= 2;
    BIEXP-wins iff dA_w1 <= -2 AND dA_w2 <= -2; blank otherwise.
  * (ii) DERIVATIVE TEST: e228's committed journal (runs/e228/journal.json;
    margin_raw per probe per state; both wash lineages independent from the
    shared t0 — e214's t0_shared, e228's own operationalization).
    damage_i(s->s') := -(m_i(s') - m_i(s)) / (s' - s) — minus the gap
    derivative per state; remaining strength := m_i(s) (the from-state
    margin_raw). Transitions: w1 0->2->10->50->80, w2 0->10->50->80;
    pairs pooled over BOTH washes (the shared-t0 from-state double-counts
    across washes, disclosed); probes aligned by name across states.
    PRIMARY rho := Spearman(damage, strength) on the FACT battery pooled
    (7 transitions x 20 probes = 140 pairs); the -0.5 bar reads it.
    Co-reports (never bars): per-wash rho; ctrl/near/tmpl batteries; the
    margin_sigma variant. ARITHMETIC DISCLOSURE: under per-probe PURE
    exponential decay damage = lam x strength -> rho -> +1 (and a
    cross-probe mixture of exponentials keeps rho > 0); the -0.5 bar sits
    deep in genuine per-probe acceleration territory — that is exactly
    why it is the runaway-specific channel.
  * VERDICT ROUTING (frozen; precedence as listed):
      MIXTURE      iff A-channel = BIEXP-wins.
      RUNAWAY      iff A-channel = STR-wins AND rho <= -0.5.
      UNDERPOWERED iff neither wash reaches |dA| >= 2 in either direction
                    AND rho > -0.5 (no signal on either channel).
      MIXED        otherwise (one channel fires and the other does not;
                   washes split; partial signal everywhere — verbatim).
  * GATES (all hard SystemExits): G_SHA — the 8 e238 npz dumps sha16 ==
    x9's committed G_SHA table; runs/x9/metrics.json sha16 ==
    10a8ff05c3f62310 (fresh bind); runs/e228/journal.json sha16 ==
    e1f476405746ad70 (fresh bind); runs/e228/metrics.json sha16 ==
    da0fb7af1a3d105e (fresh bind). G_JOURNAL — 8 states, every battery's
    probe set aligned by name at every state, margins finite. G_REPRO —
    this cell's EXP/STR refits reproduce x9's committed fact fits (nll
    within 1e-6, AICc within 1e-4, theta within 1e-3) and the one-T refit
    reproduces x9's committed per-state T within 5e-3 (x9's own gate
    tolerance) — the machinery is x9's BEFORE any BIEXP number is believed.
  * CO-REPORT (never a bar): probe-level case bootstrap (B=200, frozen
    seed 20261006) on dA = AICc_STR - AICc_BIEXP per wash, single-start
    polish from the full-data thetas (disclosed); median + central 95% +
    the fraction of resamples on each side of the +/-2 bars.
  * ENV: CPU-ONLY — no torch import at all (gpu_calls = 0 trivially),
    thread caps 1 before numpy import (<= 4 per the directive), NO
    envelope-log writes, PROGRESSIVE writes to runs/x12/metrics.json, NO
    NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Every
    timestamp from datetime.now(UTC) — never a hand guess (8e1ac9f).

OUTPUTS: runs/x12/metrics.json (progressive), runs/x12/x12_runaway.png,
runs/x12/REPORT.md.
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
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "x12"
E238 = REPO / "runs" / "e238"
E228 = REPO / "runs" / "e228"
X9RUN = REPO / "runs" / "x9"
RUN.mkdir(parents=True, exist_ok=True)

UTC = timezone.utc
T_START = time.time()
TAGS = ["t0", "w1s2", "w1s10", "w1s50", "w1s80", "w2s10", "w2s50", "w2s80"]
WASH_STATES = {"w1": [2, 10, 50, 80], "w2": [10, 50, 80]}
BATTERIES = ["ctrl", "fact", "near", "tmpl"]
EPS = 1e-12            # e238's constant, verbatim
GRID_LO, GRID_HI, GRID_N = -1.0, 1.0, 1201   # e238's grid, verbatim
LAM_BOUNDS = (1e-5, 5.0)
BETA_BOUNDS = (0.05, 3.0)
W_BOUNDS = (0.0, 1.0)
BOOT_B = 200
BOOT_SEED = 20261006
RHO_BAR = -0.5         # the frozen damage-strength anti-correlation bar
DA_BAR = 2.0           # the frozen AICc separation bar

# frozen committed records (all sha-gated below)
X9_SHA16 = "10a8ff05c3f62310"      # runs/x9/metrics.json (fresh bind)
E228_JOURNAL_SHA16 = "e1f476405746ad70"   # runs/e228/journal.json (fresh)
E228_METRICS_SHA16 = "da0fb7af1a3d105e"   # runs/e228/metrics.json (fresh)
E238_SHA16 = {  # x9's committed G_SHA table, verbatim
    "t0": "fd1d302f5713ec82", "w1s2": "59e6c0aa1b220518",
    "w1s10": "cd4fdedb5a7d605a", "w1s50": "465bac8aca6b4076",
    "w1s80": "f13897401fe60341", "w2s10": "b120737f96a65398",
    "w2s50": "06b46035fb76be1a", "w2s80": "9c807dbf906e2cbc",
}


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def log(msg: str) -> None:
    print(f"[x12 {time.time() - T_START:7.1f}s] {msg}", flush=True)


def sanitize(obj):
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


METRICS: dict = {
    "experiment": "x12_runaway_discriminator",
    "phase": "THE RUNAWAY DISCRIMINATOR (P-x12a, T256) — predatory kill "
             "(H-DRAIN-RUNAWAY) or superposition artifact (H-SHAPE-MISMATCH)? "
             "bi-exponential re-fit vs x9's beta>1 (AICc) + the e228 "
             "per-probe damage-vs-strength derivative test; CPU-only desk "
             "cell, committed records only",
    "date": now_iso(),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": 1,
    "provenance": {
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__,
                     "python": platform.python_version()},
    },
}
BARS = {
    "RUNAWAY": "bi-exp loses to beta>1 by AICc AND the damage-strength "
        "anti-correlation is real — Spearman <= -0.5 across states",
    "MIXTURE": "bi-exp wins by AICc >= 2 — W043's acceleration was "
        "superposition",
    "MIXED": "the remaining contradiction / partial-signal cases, verbatim",
    "UNDERPOWERED": "no signal on either channel — say what would fix it",
}
OP = {
    "family_test": "fact battery, x9's data/grid/machinery ported BY VALUE; "
                   "families EXP(k=2)/STR(k=3)/BIEXP(k=4: lam1,lam2,w,c)/"
                   "oneT(k=n_states); AICc x9 formula; dA_w = AICc_BIEXP - "
                   "AICc_STR; STR-wins iff dA>=2 both washes; BIEXP-wins iff "
                   "dA<=-2 both washes",
    "derivative_test": "e228 journal margin_raw; damage_i = -(m(s')-m(s))/"
                       "(s'-s); strength = m(s) (from-state); transitions "
                       "w1 0->2->10->50->80, w2 0->10->50->80; PRIMARY rho = "
                       "Spearman pooled over both washes, fact battery "
                       "(140 pairs); bar rho <= -0.5",
    "arithmetic_disclosure": "per-probe pure exponential gives damage = "
        "lam*strength -> rho -> +1; a cross-probe mixture of exponentials "
        "keeps rho > 0; the -0.5 bar is deep per-probe acceleration "
        "territory — the runaway-specific channel",
    "verdict_routing": "MIXTURE iff BIEXP-wins; else RUNAWAY iff STR-wins "
                       "and rho <= -0.5; else UNDERPOWERED iff |dA| < 2 both "
                       "washes and rho > -0.5; else MIXED",
}


def write_metrics(note: str) -> None:
    METRICS["phase"] = note
    METRICS["date_updated"] = now_iso()
    (RUN / "metrics.json").write_text(
        json.dumps(sanitize(METRICS), indent=1), encoding="utf-8")
    log(f"WROTE metrics.json ({note})")


# ============================================================================
# G_SHA — every source bit-bound before anything is computed
# ============================================================================
sha_checks = {
    "x9_metrics": {"path": "runs/x9/metrics.json", "sha16": sha16(X9RUN / "metrics.json"),
                   "committed": X9_SHA16, "source": "FRESH BIND (this cell)"},
    "e228_journal": {"path": "runs/e228/journal.json",
                     "sha16": sha16(E228 / "journal.json"),
                     "committed": E228_JOURNAL_SHA16, "source": "FRESH BIND (this cell)"},
    "e228_metrics": {"path": "runs/e228/metrics.json",
                     "sha16": sha16(E228 / "metrics.json"),
                     "committed": E228_METRICS_SHA16, "source": "FRESH BIND (this cell)"},
}
for tag in TAGS:
    p = E238 / f"logits_{tag}.npz"
    sha_checks[f"e238_{tag}"] = {
        "path": f"runs/e238/logits_{tag}.npz", "sha16": sha16(p),
        "committed": E238_SHA16[tag], "source": "x9's committed G_SHA table"}
for nm, rec in sha_checks.items():
    rec["match"] = rec["sha16"] == rec["committed"]
    if not rec["match"]:
        raise SystemExit(f"SHA BIND FAILURE: {nm} {rec['sha16']} != {rec['committed']}")
METRICS["gates"] = {"G_SHA": {"checks": sha_checks, "pass": True}}
write_metrics("P0 bars frozen in-script + all 11 sources sha-bound")


# ============================================================================
# the one-T family, copied VERBATIM from lab/x9_rate_fits.py (via e238)
# ============================================================================
class TempFamily:
    def __init__(self, L0: np.ndarray, ans_ids: np.ndarray):
        self.L0 = L0.astype(np.float64)
        self.ans = ans_ids.astype(np.int64)
        self.lmax = self.L0.max(axis=1)
        self.d = self.L0 - self.lmax[:, None]
        self.da = self.d[np.arange(len(self.ans)), self.ans]
        self.n = len(self.ans)

    def q(self, T: float) -> np.ndarray:
        with np.errstate(over="ignore"):
            num = np.exp(self.da / T)
            den = np.exp(self.d / T).sum(axis=1)
        return num / np.maximum(den, EPS)

    def grid_Q(self, n: int = GRID_N) -> tuple:
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

    def fit_T_bernoulli(self, p_obs: np.ndarray, grid=None, Q=None) -> tuple:
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


# ============================================================================
# data loading (plain IO, finiteness guards, no memmap) — x9's verbatim
# ============================================================================
def load_dump(tag: str) -> dict:
    z = np.load(E238 / f"logits_{tag}.npz", allow_pickle=False)
    out = {k: z[k] for k in z.keys()}
    assert np.all(np.isfinite(out["logits"])), f"non-finite logits in {tag}"
    return out


def softmax_ans(L: np.ndarray, ans: np.ndarray) -> np.ndarray:
    L = L.astype(np.float64)
    s = L - L.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e[np.arange(len(ans)), ans] / e.sum(axis=1)
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    return p


dumps = {tag: load_dump(tag) for tag in TAGS}
ref_meta = (list(dumps["t0"]["names"]), list(dumps["t0"]["battery"]),
            list(dumps["t0"]["ans_ids"]))
for tag in TAGS:
    meta = (list(dumps[tag]["names"]), list(dumps[tag]["battery"]),
            list(dumps[tag]["ans_ids"]))
    assert meta == ref_meta, f"probe set drift across dumps at {tag}"
batt_arr = np.array(ref_meta[1])
batt_ix = {b: np.where(batt_arr == b)[0] for b in BATTERIES}
P_obs = {}       # battery -> tag -> (n_probe,) float64
for tag in TAGS:
    pall = softmax_ans(dumps[tag]["logits"], dumps[tag]["ans_ids"])
    for b in BATTERIES:
        P_obs.setdefault(b, {})[tag] = pall[batt_ix[b]]
anchors = {b: P_obs[b]["t0"].copy() for b in BATTERIES}
n_counts = {b: int(len(batt_ix[b])) for b in BATTERIES}
log(f"data loaded: probes {n_counts}; anchors finite: "
    f"{all(np.all(np.isfinite(anchors[b])) for b in BATTERIES)}")

# ============================================================================
# the rate families — x9's machinery ported BY VALUE + BIEXP (new, k=4)
# ============================================================================
def fam_g(theta, fam, t):
    if fam == "EXP":
        return np.exp(-theta[0] * t)
    if fam == "STR":
        return np.exp(-((theta[0] * t) ** theta[1]))
    if fam == "BIEXP":
        return theta[2] * np.exp(-theta[0] * t) \
            + (1.0 - theta[2]) * np.exp(-theta[1] * t)
    raise ValueError(fam)


def fam_bounds(fam, cmax):
    b = [("lam1", LAM_BOUNDS)] if fam == "BIEXP" else [("lam", LAM_BOUNDS)]
    if fam == "BIEXP":
        b.append(("lam2", LAM_BOUNDS))
    if fam == "STR":
        b.append(("beta", BETA_BOUNDS))
    if fam == "BIEXP":
        b.append(("w", W_BOUNDS))
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
    if fam == "EXP":
        lams, shapes, ws = [0.005, 0.02, 0.05, 0.2, 0.8], [None], [None]
    elif fam == "STR":
        lams, shapes, ws = [0.005, 0.02, 0.05, 0.2, 0.8], \
            [0.25, 0.6, 1.0, 1.8], [None]
    elif fam == "BIEXP":
        lams, shapes, ws = [0.002, 0.01, 0.04, 0.16, 0.8], [None], \
            [0.2, 0.5, 0.8]
    else:
        raise ValueError(fam)
    cs = [0.0, 0.5 * cmax, 0.95 * cmax] if cmax > 1e-6 else [0.0]
    starts = []
    if fam == "BIEXP":
        for lam1 in lams:
            for lam2 in lams:
                for w in ws:
                    for c in cs[:2]:
                        starts.append(np.array([lam1, lam2, w, c], dtype=float))
    else:
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
    nll = float(fam_nll(theta, fam, anchors, t, P))
    names = [nm for nm, _ in bounds]
    bound_hit = any(abs(theta[i] - bb[i][0]) < 1e-9
                    or abs(theta[i] - bb[i][1]) < 1e-9
                    for i in range(len(theta)))
    return {"theta": [float(x) for x in theta],
            "param_names": names, "k": len(theta), "nll": nll,
            "bound_hit": bound_hit}


def lam_eff_of(theta, fam):
    g80 = float(fam_g(np.array(theta), fam, np.array([80.0]))[0])
    return -math.log(max(g80, EPS)) / 80.0


def aicc(nll, k, n):
    if n - k - 1 <= 0:
        return None
    return 2.0 * k + 2.0 * nll + 2.0 * k * (k + 1) / (n - k - 1)


# ============================================================================
# (i) the family test on the fact battery
# ============================================================================
x9m = json.loads((X9RUN / "metrics.json").read_text(encoding="utf-8"))
x9_fit = x9m["rate_fits"]          # keys 'battery/wash/FAMILY'

FITS: dict[str, dict] = {}          # wash -> family -> record
REPRO: dict[str, dict] = {}
for wash, states in WASH_STATES.items():
    t = np.array([float(s) for s in states])
    tags = [f"{wash}s{s}" for s in states]
    Pv = np.stack([P_obs["fact"][tg] for tg in tags], axis=1)   # (n, S)
    anc = anchors["fact"]
    n_obs = int(Pv.size)
    fits_w = {}
    for fam in ("EXP", "STR", "BIEXP"):
        rec = fit_family(fam, anc, t, Pv)
        rec["n_obs"] = n_obs
        rec["aicc"] = aicc(rec["nll"], rec["k"], n_obs)
        rec["lam_eff"] = lam_eff_of(rec["theta"], fam)
        fits_w[fam] = rec
        log(f"fact {wash} {fam}: theta={['%.6g' % x for x in rec['theta']]} "
            f"nll={rec['nll']:.6f} aicc={rec['aicc']:.4f}")
    # one-T comparator (k = n_states; NLL summed over per-state best T)
    onet_fam = TempFamily(dumps["t0"]["logits"][batt_ix["fact"]],
                          dumps["t0"]["ans_ids"][batt_ix["fact"]])
    grid, Q = onet_fam.grid_Q()
    nll_sum, Tstar = 0.0, {}
    for tg, s in zip(tags, states):
        T, nll_s = onet_fam.fit_T_bernoulli(P_obs["fact"][tg], grid, Q)
        Tstar[str(s)] = T
        nll_sum += nll_s
    k_onet = len(states)
    fits_w["oneT"] = {"theta": None, "param_names": None, "k": k_onet,
                      "n_obs": n_obs, "nll": nll_sum,
                      "aicc": aicc(nll_sum, k_onet, n_obs),
                      "T_per_state": Tstar, "bound_hit": False}
    FITS[wash] = fits_w
    # G_REPRO vs x9's committed fact fits
    for fam in ("EXP", "STR"):
        ref = x9_fit[f"fact/{wash}/{fam}"]
        got = fits_w[fam]
        d_theta = max(abs(a - b) for a, b in
                      zip(got["theta"], ref["theta"]))
        REPRO[f"fact/{wash}/{fam}"] = {
            "x9_nll": ref["nll"], "this_nll": got["nll"],
            "nll_abs_diff": abs(got["nll"] - ref["nll"]),
            "x9_aicc": ref["aicc"], "this_aicc": got["aicc"],
            "aicc_abs_diff": abs(got["aicc"] - ref["aicc"]),
            "theta_max_abs_diff": d_theta,
            "pass": (abs(got["nll"] - ref["nll"]) <= 1e-6
                     and abs(got["aicc"] - ref["aicc"]) <= 1e-4
                     and d_theta <= 1e-3)}
    ref_onet = x9_fit[f"fact/{wash}/oneT"]
    dT = max(abs(Tstar[str(s)] - ref_onet["T_per_state"][str(s)])
             for s in states)
    REPRO[f"fact/{wash}/oneT"] = {
        "x9_T": ref_onet["T_per_state"], "this_T": Tstar,
        "max_abs_dT": dT, "x9_nll": ref_onet["nll"],
        "this_nll": fits_w["oneT"]["nll"],
        "nll_abs_diff": abs(fits_w["oneT"]["nll"] - ref_onet["nll"]),
        "pass": dT <= 5e-3 and abs(fits_w["oneT"]["nll"] - ref_onet["nll"]) <= 1e-6}
    for nm, rec in REPRO.items():
        if nm.startswith(f"fact/{wash}") and not rec["pass"]:
            raise SystemExit(f"REPRO GATE FAILURE: {nm}: {rec}")
METRICS["gates"]["G_REPRO"] = {**REPRO, "pass": True}
METRICS["family_fits"] = FITS
dA = {w: FITS[w]["BIEXP"]["aicc"] - FITS[w]["STR"]["aicc"]
      for w in WASH_STATES}
a_channel = ("STR-wins" if dA["w1"] >= DA_BAR and dA["w2"] >= DA_BAR
             else "BIEXP-wins" if dA["w1"] <= -DA_BAR and dA["w2"] <= -DA_BAR
             else "blank")
write_metrics(f"P1 family test done: EXP/STR/BIEXP/oneT on fact w1+w2; "
              f"A-channel {a_channel} (dA w1 {dA['w1']:.3f}, w2 {dA['w2']:.3f})")


# ============================================================================
# (ii) the derivative test on e228's committed journal
# ============================================================================
journal = json.loads((E228 / "journal.json").read_text(encoding="utf-8"))
states_j = {s["wash"] + str(s["step"]): s for s in journal["states"]}
assert set(states_j.keys()) == {"t00", "w12", "w110", "w150", "w180",
                                "w210", "w250", "w280"}, sorted(states_j)


def margins_of(battery: str, key: str, field: str = "margin_raw") -> dict:
    probes = states_j[key][battery]["probes"]
    return {p["fact"]: float(p[field]) for p in probes}


DERIV: dict[str, dict] = {}
for battery in BATTERIES:
    names = set(margins_of(battery, "t00").keys())
    per_wash, pooled_d, pooled_s = {}, [], []
    n_neg = 0
    for wash, chain in (("w1", ["t00", "w12", "w110", "w150", "w180"]),
                        ("w2", ["t00", "w210", "w250", "w280"])):
        ms = [margins_of(battery, k) for k in chain]
        for m in ms:
            assert set(m.keys()) == names, f"probe drift in {battery}"
            n_neg += sum(1 for v in m.values() if v < 0)
        d_list, s_list = [], []
        for i in range(len(chain) - 1):
            s_prev = 0.0 if i == 0 else float(chain[i][2:])
            s_next = float(chain[i + 1][2:])
            for nm in sorted(names):
                m0, m1 = ms[i][nm], ms[i + 1][nm]
                damage = -(m1 - m0) / (s_next - s_prev)
                d_list.append(damage)
                s_list.append(m0)
        rho_w = spearmanr(d_list, s_list)
        per_wash[wash] = {"n_pairs": len(d_list),
                          "rho": float(rho_w.statistic),
                          "p": float(rho_w.pvalue)}
        pooled_d += d_list
        pooled_s += s_list
    assert all(math.isfinite(v) for v in pooled_d + pooled_s)
    rho_p = spearmanr(pooled_d, pooled_s)
    DERIV[battery] = {
        "n_probes": len(names), "n_neg_margins": n_neg,
        "rho_pooled": float(rho_p.statistic), "p_pooled": float(rho_p.pvalue),
        "n_pairs_pooled": len(pooled_d), "per_wash": per_wash,
        "_d": pooled_d, "_s": pooled_s,
    }
    log(f"deriv {battery}: rho_pooled={DERIV[battery]['rho_pooled']:.4f} "
        f"(p={DERIV[battery]['p_pooled']:.2e}), "
        f"w1 {per_wash['w1']['rho']:.3f} / w2 {per_wash['w2']['rho']:.3f}")
# the margin_sigma sensitivity variant (fact only, co-report)
for battery in ("fact",):
    d2, s2 = [], []
    for wash, chain in (("w1", ["t00", "w12", "w110", "w150", "w180"]),
                        ("w2", ["t00", "w210", "w250", "w280"])):
        ms = [margins_of(battery, k, "margin_sigma") for k in chain]
        for i in range(len(chain) - 1):
            s_prev = 0.0 if i == 0 else float(chain[i][2:])
            s_next = float(chain[i + 1][2:])
            for nm in sorted(ms[0].keys()):
                d2.append(-(ms[i + 1][nm] - ms[i][nm]) / (s_next - s_prev))
                s2.append(ms[i][nm])
    rho2 = spearmanr(d2, s2)
    DERIV[battery]["rho_pooled_margin_sigma"] = float(rho2.statistic)
METRICS["derivative_test"] = {
    b: {k: v for k, v in rec.items() if not str(k).startswith("_")}
    for b, rec in DERIV.items()}
rho_primary = DERIV["fact"]["rho_pooled"]
write_metrics(f"P2 derivative test done: fact rho_pooled={rho_primary:.4f} "
              f"(bar {RHO_BAR}); per-battery + margin_sigma co-reports in")


# ============================================================================
# bootstrap stability co-report on dA (never a bar)
# ============================================================================
rng = np.random.default_rng(BOOT_SEED)
BOOT = {}
for wash, states in WASH_STATES.items():
    t = np.array([float(s) for s in states])
    tags = [f"{wash}s{s}" for s in states]
    Pv = np.stack([P_obs["fact"][tg] for tg in tags], axis=1)
    anc = anchors["fact"]
    n_obs = int(Pv.size)
    th_str = np.array(FITS[wash]["STR"]["theta"])
    th_bi = np.array(FITS[wash]["BIEXP"]["theta"])
    bstr = fam_bounds("STR", float(anc.min()))
    bbi = fam_bounds("BIEXP", float(anc.min()))
    das = []
    for b in range(BOOT_B):
        ix = rng.integers(0, len(anc), len(anc))
        r_str = minimize(fam_nll, th_str, args=("STR", anc[ix], t, Pv[ix]),
                         method="Nelder-Mead", bounds=[v for _, v in bstr],
                         options={"maxiter": 4000, "xatol": 1e-10,
                                  "fatol": 1e-12})
        r_bi = minimize(fam_nll, th_bi, args=("BIEXP", anc[ix], t, Pv[ix]),
                        method="Nelder-Mead", bounds=[v for _, v in bbi],
                        options={"maxiter": 4000, "xatol": 1e-10,
                                 "fatol": 1e-12})
        a_str = aicc(float(r_str.fun), 3, n_obs)
        a_bi = aicc(float(r_bi.fun), 4, n_obs)
        das.append(a_str - a_bi)     # = -dA: positive = stretched wins
    das = np.array(das)
    BOOT[wash] = {
        "B": BOOT_B, "seed": BOOT_SEED, "note": "single-start polish from "
        "full-data thetas (disclosed); dA_str_minus_biexp = AICc_STR - "
        "AICc_BIEXP, positive = stretched wins",
        "median": float(np.median(das)),
        "ci95": [float(np.percentile(das, 2.5)),
                 float(np.percentile(das, 97.5))],
        "frac_str_wins_ge2": float((das >= DA_BAR).mean()),
        "frac_biexp_wins_le_minus2": float((das <= -DA_BAR).mean()),
    }
    log(f"boot {wash}: median dA(s-b)={BOOT[wash]['median']:.2f} "
        f"ci [{BOOT[wash]['ci95'][0]:.2f}, {BOOT[wash]['ci95'][1]:.2f}]")
METRICS["bootstrap_coreport"] = BOOT
write_metrics("P3 bootstrap stability co-report done (B=200)")


# ============================================================================
# adjudication (frozen routing)
# ============================================================================
b_fires = rho_primary <= RHO_BAR
if a_channel == "BIEXP-wins":
    verdict = "MIXTURE"
elif a_channel == "STR-wins" and b_fires:
    verdict = "RUNAWAY"
elif (abs(dA["w1"]) < DA_BAR and abs(dA["w2"]) < DA_BAR) and not b_fires:
    verdict = "UNDERPOWERED"
else:
    verdict = "MIXED"
underpowered_note = None
if verdict == "UNDERPOWERED":
    underpowered_note = (
        "what would fix it: more eval states per trajectory (the 3-4 "
        "committed states barely identify k=4), more probes per battery "
        "(fact n=20 is the largest), and a per-probe wash grid dense "
        "enough to fit per-probe rates directly")
METRICS["verdict"] = {
    "word": verdict,
    "a_channel": a_channel, "dA_w1": dA["w1"], "dA_w2": dA["w2"],
    "rho_primary_fact_pooled": rho_primary, "rho_bar": RHO_BAR,
    "b_channel_fires": b_fires,
    "clause_trace": {
        "biexp_wins_both_washes": a_channel == "BIEXP-wins",
        "str_wins_both_washes": a_channel == "STR-wins",
        "rho_le_minus_0_5": b_fires,
        "no_dA_signal_either_wash": (abs(dA["w1"]) < DA_BAR
                                     and abs(dA["w2"]) < DA_BAR),
    },
    "underpowered_note": underpowered_note,
    "bars_verbatim": BARS, "operationalizations": OP,
}
write_metrics(f"P4 ADJUDICATED: {verdict}")


# ============================================================================
# figure
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
COL = {"EXP": "tab:blue", "STR": "crimson", "BIEXP": "darkgreen",
       "oneT": "gray"}
for ax, wash in zip((axes[0, 0], axes[0, 1]), WASH_STATES):
    states = WASH_STATES[wash]
    t = np.array([float(s) for s in states])
    tags = [f"{wash}s{s}" for s in states]
    Pv = np.stack([P_obs["fact"][tg] for tg in tags], axis=1)
    ax.plot(t, Pv.mean(axis=0), "ko", ms=8, label="fact mean p_obs")
    tt = np.linspace(0.0, 84.0, 240)
    for fam in ("EXP", "STR", "BIEXP"):
        th = np.array(FITS[wash][fam]["theta"])
        Qm = fam_q(th, fam, anchors["fact"], tt).mean(axis=0)
        lbl = {"EXP": "EXP", "STR": f"STR (beta={th[1]:.3f})",
               "BIEXP": (f"BIEXP (w={th[2]:.3f}, "
                         f"lam1={th[0]:.4f}, lam2={th[1]:.4f})")}[fam]
        ax.plot(tt, Qm, "-", color=COL[fam], lw=1.8,
                label=f"{lbl}; AICc {FITS[wash][fam]['aicc']:.2f}")
    ax.set_xlabel("wash steps t")
    ax.set_ylabel("mean answer probability")
    ax.set_title(f"fact battery, {wash} "
                 f"(dA = AICc_BIEXP - AICc_STR = {dA[wash]:+.2f})")
    ax.legend(fontsize=8)

ax = axes[1, 0]
fams = ["EXP", "STR", "BIEXP", "oneT"]
xp = np.arange(len(fams))
w1s = [FITS["w1"][f]["aicc"] for f in fams]
w2s = [FITS["w2"][f]["aicc"] for f in fams]
ax.bar(xp - 0.18, w1s, 0.36, color="steelblue", alpha=0.85, label="w1")
ax.bar(xp + 0.18, w2s, 0.36, color="navy", alpha=0.85, label="w2")
best = min(min(w1s), min(w2s))
ax.set_ylim(best - 1.5, max(max(w1s), max(w2s)) + 1.5)
for i, (a, b) in enumerate(zip(w1s, w2s)):
    ax.annotate(f"{a:.1f}", (i - 0.18, a), ha="center",
                textcoords="offset points", xytext=(0, 4), fontsize=8)
    ax.annotate(f"{b:.1f}", (i + 0.18, b), ha="center",
                textcoords="offset points", xytext=(0, 4), fontsize=8)
ax.set_xticks(xp, fams)
ax.set_ylabel("AICc (lower = better)")
ax.set_title(f"(c) THE FAMILY TABLE — A-channel: {a_channel}")
ax.legend(fontsize=8)

ax = axes[1, 1]
d_arr = np.array(DERIV["fact"]["_d"])
s_arr = np.array(DERIV["fact"]["_s"])
ax.scatter(s_arr, d_arr, s=14, alpha=0.55, color="purple",
           edgecolors="none")
# binned medians for the trend
qs = np.quantile(s_arr, np.linspace(0, 1, 7))
mids, meds = [], []
for i in range(6):
    m = (s_arr >= qs[i]) & (s_arr <= qs[i + 1] if i == 5 else s_arr < qs[i + 1])
    if m.sum() > 0:
        mids.append(float(s_arr[m].mean()))
        meds.append(float(np.median(d_arr[m])))
ax.plot(mids, meds, "o-", color="black", lw=1.8, ms=7,
        label="binned medians")
ax.axhline(0.0, color="gray", lw=0.7)
ax.set_xlabel("remaining strength = margin_raw at from-state")
ax.set_ylabel("damage = -d(gap)/d(state) per step")
ax.set_title(f"(d) THE DERIVATIVE TEST (fact, pooled)\n"
             f"Spearman rho = {rho_primary:.3f} "
             f"(p = {DERIV['fact']['p_pooled']:.1e}; bar {RHO_BAR})")
ax.legend(fontsize=8)

fig.suptitle(f"x12 THE RUNAWAY DISCRIMINATOR — verdict: {verdict} "
             f"(A-channel {a_channel}: dA w1 {dA['w1']:+.2f} / w2 {dA['w2']:+.2f}; "
             f"rho {rho_primary:+.3f})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(RUN / "x12_runaway.png", dpi=140)
log("FIGURE written")
METRICS["outputs"] = {"figure": "runs/x12/x12_runaway.png",
                      "metrics": "runs/x12/metrics.json",
                      "report": "runs/x12/REPORT.md"}
write_metrics("P5 figure written")


# ============================================================================
# REPORT.md
# ============================================================================
def fmt_fit(rec):
    th = rec["theta"]
    if th is None:
        return f"k={rec['k']}, nll={rec['nll']:.4f}"
    return (f"k={rec['k']}, theta=[{', '.join(f'{x:.6g}' for x in th)}], "
            f"nll={rec['nll']:.4f}")


fit_rows = []
for wash in ("w1", "w2"):
    for fam in fams:
        rec = FITS[wash][fam]
        fit_rows.append(
            f"| {wash} | {fam} | {rec['k']} | "
            f"{('-' if rec['theta'] is None else ', '.join(f'{x:.5g}' for x in rec['theta']))} | "
            f"{rec['nll']:.4f} | **{rec['aicc']:.4f}** | "
            f"{rec.get('lam_eff') if rec.get('lam_eff') is None else round(rec['lam_eff'], 6)} | "
            f"{rec['bound_hit']} |")

deriv_rows = []
for b in BATTERIES:
    r_ = DERIV[b]
    deriv_rows.append(
        f"| {b} | {r_['n_probes']} | {r_['rho_pooled']:+.4f} "
        f"(p={r_['p_pooled']:.1e}) | {r_['per_wash']['w1']['rho']:+.4f} | "
        f"{r_['per_wash']['w2']['rho']:+.4f} | {r_['n_pairs_pooled']} | "
        f"{r_['n_neg_margins']} |")

vt = METRICS["verdict"]["clause_trace"]
report = f"""# x12 — THE RUNAWAY DISCRIMINATOR ({verdict})

**The registered question (P-x12a, T256):** x9's stretched-exponential fit
gave the FACT battery beta = 1.31 (w1 1.2672 / w2 1.3458; bootstrap CIs
[1.137, 2.894] / [1.148, 2.357] — the only battery excluding 1). Is that
acceleration PREDATORY (per-probe damage runs away as strength falls,
H-DRAIN-RUNAWAY) or SUPERPOSITION (a bi-exponential mixture of fast and
slow probes imitating acceleration, H-SHAPE-MISMATCH)?

## (i) The family table (fact battery; x9's data + machinery, BIEXP added)

| wash | family | k | theta | NLL | AICc | lam_eff | bound_hit |
|---|---|---|---|---|---|---|---|
{chr(10).join(fit_rows)}

**dA := AICc_BIEXP - AICc_STR** (positive = stretched wins): w1
**{dA['w1']:+.3f}**, w2 **{dA['w2']:+.3f}** -> A-channel =
**{a_channel}** (bar |dA| >= 2 on both washes).

Bootstrap stability (B=200 probe-level resamples, single-start polish,
co-report never a bar): w1 median AICc_STR - AICc_BIEXP
{BOOT['w1']['median']:+.2f} (95% [{BOOT['w1']['ci95'][0]:+.2f},
{BOOT['w1']['ci95'][1]:+.2f}], stretched wins >= 2 in
{100*BOOT['w1']['frac_str_wins_ge2']:.0f}% of resamples); w2 median
{BOOT['w2']['median']:+.2f} (95% [{BOOT['w2']['ci95'][0]:+.2f},
{BOOT['w2']['ci95'][1]:+.2f}], {100*BOOT['w2']['frac_str_wins_ge2']:.0f}%).
The EXP-vs-STR context: x9's committed fact EXP AICc w1
{x9_fit['fact/w1/EXP']['aicc']:.2f} / w2 {x9_fit['fact/w2/EXP']['aicc']:.2f}
(this cell reproduced them, G_REPRO).

## (ii) The derivative test (e228's committed per-probe margins)

damage = -d(gap)/d(state); strength = from-state margin_raw; pooled over
both wash lineages (independent from the shared t0).

| battery | n probes | rho pooled | rho w1 | rho w2 | pairs | negative margins |
|---|---|---|---|---|---|---|
{chr(10).join(deriv_rows)}

**PRIMARY (fact): rho = {rho_primary:+.4f}** (p =
{DERIV['fact']['p_pooled']:.2e}); margin_sigma variant
{DERIV['fact']['rho_pooled_margin_sigma']:+.4f}; bar rho <= {RHO_BAR} ->
B-channel {'FIRES' if b_fires else 'does not fire'}.

Arithmetic disclosure: under per-probe pure exponential decay damage =
lam x strength -> rho -> +1, and a cross-probe MIXTURE of exponentials
also keeps rho > 0 — the -0.5 bar is deep in genuine per-probe
acceleration territory. That is why the derivative test is the
runaway-specific channel and the family test the mixture-specific one.

## Verdict: {verdict}

Clause trace: BIEXP wins both washes = {vt['biexp_wins_both_washes']};
STR wins both washes = {vt['str_wins_both_washes']}; rho <= -0.5 =
{vt['rho_le_minus_0_5']}; no |dA| >= 2 signal on either wash =
{vt['no_dA_signal_either_wash']}.
{underpowered_note or ''}

## Gates

- G_SHA: the 8 e238 npz dumps sha16-match x9's committed G_SHA table;
  runs/x9/metrics.json (10a8ff05c3f62310), runs/e228/journal.json
  (e1f476405746ad70), runs/e228/metrics.json (da0fb7af1a3d105e)
  fresh-bound (recorded for future cells).
- G_JOURNAL: 8 states, probe sets aligned by name at every state, all
  margins finite.
- G_REPRO: this cell's EXP/STR refits reproduce x9's committed fact fits
  (nll <= 1e-6, AICc <= 1e-4, theta <= 1e-3) and the one-T refit
  reproduces x9's per-state T within 5e-3 — the machinery is x9's before
  any BIEXP number is believed.

## Disclosures

1. The BIEXP family is NEW (k = 4); its w bound at 0 or 1 degenerates to
   EXP — flagged as bound_hit, never hidden.
2. The derivative test pools both wash lineages; the shared t0 from-state
   double-counts across them (disclosed at birth); per-wash co-reports
   shown; the margin_raw field is primary per the dispatch, margin_sigma
   the sensitivity.
3. The bootstrap uses single-start polish from full-data thetas (not the
   full multistart) — a stability co-report, never a bar.
4. CPU-only: threads 1, torch never imported, no envelope-log writes, no
   NOTES/THINKING/QUEUE/STATE edits; all timestamps datetime.now(UTC).
"""
(RUN / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["status"] = "COMPLETE — adjudicated"
write_metrics("P6 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}; dA w1 {dA['w1']:+.3f} / w2 {dA['w2']:+.3f}; "
    f"rho {rho_primary:+.4f}")
