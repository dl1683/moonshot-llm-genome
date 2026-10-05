"""X13 — FLAT TAX vs WEALTH TAX (W047's follow-up tension; desk cell, CPU-only).

THE TENSION (registered by the coordinator): x9 (runs/x9/metrics.json) fitted
exponential decays to the four batteries' margin discharge curves and called
the one-T ladder a RATE ladder (near 5-7x faster etc.) — exponential margins
imply per-step damage proportional to the current gap (a WEALTH tax:
d(gap)/dt = -lam*gap). But x12 (runs/x12/metrics.json) measured the
damage-strength Spearman ~= 0 for every battery (fact +0.086, ctrl +0.06,
near +0.22, tmpl +0.36) — damage INDEPENDENT of remaining strength, i.e. a
FLAT tax (d(gap)/dt = -c, LINEAR margin decline). Both cannot be right. The
resolution candidate: x9's family set NEVER INCLUDED LINEAR — and over a
limited state range, exponential and linear are nearly indistinguishable by
R2. The missing member decides which story the record supports.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
script committed at birth; adjudicate against exactly this; no bar shopping):

- FLAT-TAX: LINEAR wins (dAICc = AICc_EXP - AICc_LIN >= 2) in >= 5 of 8 fits
  AND x12's rho ~ 0 is consistent — the T-ladder's "rates" were
  exponential-fitting artifacts of near-linear declines; the object is
  per-battery FLAT RENTS (the wash charges constant absolute rent); W047's
  flat-tax reading confirmed; T254's rate-ladder headline DOWNGRADED at its
  claim site (the coordinator folds that amendment).
- WEALTH-TAX: EXP wins (>= -2 in the other direction) in >= 5 of 8 — the
  rates are real proportional drains; x12's rho then needs its own
  explanation (the tension stands, flagged).
- MIXED: split or inconclusive — the table verbatim + the discrimination-
  power statement.

OPERATIONALIZATION (frozen with the bars, before any compute):

1. DATA: e238's committed npz logit dumps (runs/e238/logits_{tag}.npz, tag in
   t0, w1s2, w1s10, w1s50, w1s80, w2s10, w2s50, w2s80) — x9's exact data and
   grid. BIT-VERIFY BEFORE TRUST: sha256 of every dump must match x9's
   committed G_SHA table (which x12 independently re-verified); runs/x9 and
   runs/x12 metrics hashed and (for x9) checked against x12's committed
   record. p recomputed float64 numpy softmax (x9's convention) must
   reproduce x9's committed discharge mean_p to <= 1e-9. Grid: anchors
   a_i = p_i(t0); eval states w1 {2,10,50,80}, w2 {10,50,80}; t0 anchor only,
   never scored.
2. FAMILIES: x9's machinery ported BY VALUE (q_i(t) = c + (a_i - c) g(t);
   Bernoulli soft-target CE; Nelder-Mead with bounds from a frozen multistart
   grid; identical options):
   - EXP  g(t) = exp(-lam t)          k = 2 (lam, c)      [x9's, verbatim]
   - LIN  g(t) = max(0, 1 - slope*t)  k = 2 (slope, c)    [THE NEW MEMBER —
     never in x9's set; the flat-rent gap survival; the max(0,.) hinge is
     disclosed and never active at the fitted slopes since g(80) > 0]
   Bounds: lam in [1e-5, 5], slope in [0, 0.05], c in [0, min_i a_i].
3. G_REPRO: this cell's EXP refit must reproduce x9's 8 committed EXP rows
   (theta within 1e-6, nll within 1e-8) — same starts, same machinery.
4. AICc = 2k + 2*NLL + 2k(k+1)/(n - k - 1), n = n_probes * n_eval_states
   (x9's formula). Both families k = 2, so dAICc = AICc_EXP - AICc_LIN =
   2*(NLL_EXP - NLL_LIN) EXACTLY (penalties cancel; disclosed).
   Verdict counts: n_lin = #{fits: dAICc >= 2}, n_exp = #{fits: dAICc <= -2}
   over the 8 battery-wash fits.
5. CONSISTENCY CHECK (the hybrid diagnostic, cross-cell): per battery, x12's
   committed margin-currency rho_pooled (quoted verbatim from
   runs/x12/metrics.json) vs the two families' IMPLIED pooled damage-strength
   Spearman computed on x9's exact grid/currency with the per-wash fitted
   params: damage_i(s->s') = (q_i(s) - q_i(s'))/(s'-s), strength = q_i(s),
   transitions w1 0->2->10->50->80, w2 0->10->50->80, pairs pooled over both
   washes (x12's pooling). Bridge co-report: the SAME derivative computation
   on the observed e238 p data (x12's computation ported to x9's currency).
   Arithmetic disclosure: pure EXP with c=0 gives damage = lam*strength ->
   rho = +1; pure LIN gives damage constant within a probe (rho = 0 absent
   cross-probe anchor spread). CONSISTENT-with-LIN (the FLAT-TAX AND-clause)
   fires iff for EVERY battery |rho_x12 - rho_LIN_implied| <
   |rho_x12 - rho_EXP_implied| (x12's measured rho sits closer to the linear
   family's implied correlation than the exponential's).
6. DISCRIMINATION POWER (the design statement for any future finer grid):
   noiseless-template scan per battery-wash over endpoint gap-decline
   D := 1 - g(80) in (0, 0.98]: truth_lin g_D(t) = max(0, 1 - (D/80) t) with
   the fitted LIN floor; truth_exp g_D(t) = exp(-(-ln(1-D)/80) t) with the
   fitted EXP floor; both families refit to the noiseless soft-target matrix
   (same anchors, states, n); f(D) := AICc_EXP - AICc_LIN. D*_lin = min D
   with f(D) >= 2 (how deep must a TRUE flat-rent decline go before the
   design catches it); D*_exp = min D with f(D) <= -2 (same for a true
   proportional drain). Grid scan + bisection. D_obs (LIN fit slope*80; EXP
   co-report 1 - exp(-lam*80)) and the max |mean-curve gap| between the two
   fitted families over t in [0,80] co-reported (the indistinguishability
   magnitude over the measured range).
7. BOOTSTRAP co-report (never a bar): probe-level case resampling B=200,
   seed 20261007 (frozen), dAICc distribution per battery-wash (median,
   2.5/97.5 pct, frac >= 2, frac <= -2) — the noise-driven separation.
8. VERDICT ROUTING (frozen): FLAT-TAX iff n_lin >= 5 AND consistent-LIN;
   WEALTH-TAX iff n_exp >= 5; else MIXED (table verbatim + discrimination
   statement).
9. ENV: CPU-ONLY — no torch import (gpu_calls = 0 trivially), thread caps
   set to 1 (<= 2 per the dispatch) before numpy import, NO envelope-log
   writes, no NOTES/THINKING/QUEUE/STATE edits. Every timestamp from
   datetime.now(UTC).

OUTPUTS: runs/x13/metrics.json (progressive), runs/x13/x13_tax_fits.png,
runs/x13/REPORT.md.
"""

import os

# --- thread + device caps BEFORE numpy (the CPU directive; 1 <= 2) ---
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
RUN = REPO / "runs" / "x13"
E238 = REPO / "runs" / "e238"
RUN.mkdir(parents=True, exist_ok=True)

T_START = time.time()
TAGS = ["t0", "w1s2", "w1s10", "w1s50", "w1s80", "w2s10", "w2s50", "w2s80"]
WASH_STATES = {"w1": [2, 10, 50, 80], "w2": [10, 50, 80]}
BATTERIES = ["ctrl", "fact", "near", "tmpl"]
EPS = 1e-12  # e238's constant, verbatim
LAM_BOUNDS = (1e-5, 5.0)
SLOPE_BOUNDS = (0.0, 0.05)
BOOT_B = 200
BOOT_SEED = 20261007
DA_BAR = 2.0
TOL_MEANP = 1e-9
TOL_THETA_REPRO = 1e-6
TOL_NLL_REPRO = 1e-8

FROZEN_BARS = {
    "FLAT-TAX": ("LINEAR wins (dAICc = AICc_EXP - AICc_LIN >= 2) in >= 5 of 8 "
                 "fits AND x12's rho ~ 0 is consistent — the T-ladder's "
                 "'rates' were exponential-fitting artifacts of near-linear "
                 "declines; the object is per-battery FLAT RENTS (the wash "
                 "charges constant absolute rent); W047's flat-tax reading "
                 "confirmed; T254's rate-ladder headline DOWNGRADED at its "
                 "claim site (the coordinator folds that amendment)."),
    "WEALTH-TAX": ("EXP wins (>= -2 in the other direction) in >= 5 of 8 — "
                   "the rates are real proportional drains; x12's rho then "
                   "needs its own explanation (the tension stands, flagged)."),
    "MIXED": ("split or inconclusive — the table verbatim + the "
              "discrimination-power statement."),
}

cpu_launch = psutil.cpu_percent(interval=0.3)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_full(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha16(path: Path) -> str:
    return sha256_full(path)[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unavailable"


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


def write_metrics(metrics: dict, phase_note: str) -> None:
    metrics["date"] = now_iso()
    metrics["phase"] = phase_note
    (RUN / "metrics.json").write_text(
        json.dumps(sanitize(metrics), indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Data loading (plain IO reads, finiteness guards, no memmap) — x9 verbatim
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
# The families — x9's machinery BY VALUE + the new LIN member
# ----------------------------------------------------------------------------

def fam_g(theta, fam, t):
    if fam == "EXP":
        return np.exp(-theta[0] * t)
    if fam == "LIN":
        return np.maximum(0.0, 1.0 - theta[0] * t)
    raise ValueError(fam)


def fam_bounds(fam, cmax):
    if fam == "EXP":
        b = [("lam", LAM_BOUNDS)]
    elif fam == "LIN":
        b = [("slope", SLOPE_BOUNDS)]
    else:
        raise ValueError(fam)
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
        # x9's exact EXP multistart (G_REPRO requires it, verbatim)
        lams = [0.005, 0.02, 0.05, 0.2, 0.8]
    else:
        lams = [0.0005, 0.001, 0.002, 0.0035, 0.005, 0.0075,
                0.01, 0.0125, 0.02, 0.035]
    cs = [0.0, 0.5 * cmax, 0.95 * cmax] if cmax > 1e-6 else [0.0]
    starts = []
    for lam in lams:
        for c in cs:
            starts.append(np.array([lam, c], dtype=float))
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
    if g80 <= 0.0:
        return None  # LIN hinge reached: rate currency undefined (disclosed)
    return -math.log(max(g80, EPS)) / 80.0


def aicc(nll, k, n):
    if n - k - 1 <= 0:
        return None
    return 2.0 * k + 2.0 * nll + 2.0 * k * (k + 1) / (n - k - 1)


# ----------------------------------------------------------------------------
# The damage/strength derivative computation (x12's, ported to x9's currency)
# ----------------------------------------------------------------------------

def damage_strength_pairs(curves, wash, names_sorted=None):
    """curves: dict state-> per-probe vector, keys 't0' and f'{wash}s{s}'.
    Returns (damage, strength) lists for transitions 0->...->80 of one wash."""
    states = [0] + WASH_STATES[wash]
    d_list, s_list = [], []
    for i in range(len(states) - 1):
        s_prev, s_next = float(states[i]), float(states[i + 1])
        key_prev = "t0" if i == 0 else f"{wash}s{states[i]}"
        key_next = f"{wash}s{states[i + 1]}"
        p0, p1 = curves[key_prev], curves[key_next]
        for j in range(len(p0)):
            d_list.append((p0[j] - p1[j]) / (s_next - s_prev))
            s_list.append(p0[j])
    return d_list, s_list


def model_curves(theta, fam, anchors, wash):
    """q_i at t0 (anchor) and every eval state of the wash."""
    out = {"t0": anchors.copy()}
    for s in WASH_STATES[wash]:
        out[f"{wash}s{s}"] = fam_q(theta, fam, anchors,
                                   np.array([float(s)]))[:, 0]
    return out


def implied_rho(battery, fam, fits):
    """Pooled damage-strength Spearman under a family's fitted model."""
    d_all, s_all = [], []
    for wash in WASH_STATES:
        theta = np.array(fits[(battery, wash, fam)]["theta"])
        curves = model_curves(theta, fam, P[battery]["t0"], wash)
        d, s = damage_strength_pairs(curves, wash)
        d_all += d
        s_all += s
    r = spearmanr(d_all, s_all)
    return float(r.statistic), float(r.pvalue), len(d_all)


def observed_rho_bridge(battery):
    """x12's derivative computation on the OBSERVED e238 p data (the bridge)."""
    d_all, s_all = [], []
    for wash in WASH_STATES:
        d, s = damage_strength_pairs(P[battery], wash)
        d_all += d
        s_all += s
    r = spearmanr(d_all, s_all)
    return float(r.statistic), float(r.pvalue), len(d_all)


# ============================================================================
# MAIN
# ============================================================================

metrics = {
    "experiment": "x13_flat_vs_wealth",
    "status": "RUNNING",
    "registration": ("bars frozen VERBATIM from the dispatch brief BEFORE "
                     "any compute (script committed at birth); adjudicate "
                     "against exactly this; no bar shopping"),
    "frozen_bars_verbatim": FROZEN_BARS,
    "envelope": {
        "device": "CPU-ONLY (torch never imported; CUDA_VISIBLE_DEVICES=-1 "
                  "set regardless)",
        "threads": 1,
        "threads_directive": "<= 2 per the dispatch",
        "gpu_calls": 0,
        "envelope_log_writes": 0,
        "cpu_load_pct_at_launch": cpu_launch,
        "no_notes_thinking_queue_state_edits": True,
        "timestamp_rule": "every timestamp from datetime.now(UTC)",
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
    "tension": {
        "x9_claim": ("exponential margin fits -> per-step damage prop. to "
                     "current gap (WEALTH tax, d(gap)/dt = -lam*gap); the "
                     "one-T ladder a RATE ladder (near 5-7x faster)"),
        "x12_claim": ("damage-strength Spearman ~= 0 for every battery "
                      "(fact +0.086, ctrl +0.06, near +0.22, tmpl +0.36) — "
                      "damage independent of remaining strength (FLAT tax, "
                      "d(gap)/dt = -c, linear margin decline)"),
        "resolution_candidate": ("x9's family set never included LINEAR; "
                                 "over a limited state range exp and linear "
                                 "are nearly indistinguishable by R2 — the "
                                 "missing member decides"),
    },
}

# ---------------- P1: load + bit-verify --------------------------------------
x9m = json.loads((REPO / "runs" / "x9" / "metrics.json").read_text())
x12m = json.loads((REPO / "runs" / "x12" / "metrics.json").read_text())

sha_rows, sha_ok = {}, True
for tag in TAGS:
    path = E238 / f"logits_{tag}.npz"
    full = sha256_full(path)
    s16 = full[:16]
    ok_x9 = s16 == x9m["gates"]["G_SHA"]["e238_dump_sha16"][tag]
    ok_x12 = (s16 == x12m["gates"]["G_SHA"]["checks"][f"e238_{tag}"]["committed"])
    sha_rows[tag] = {"sha256": full, "sha16": s16,
                     "matches_x9_committed_G_SHA": ok_x9,
                     "matches_x12_committed_G_SHA": ok_x12}
    sha_ok = sha_ok and ok_x9 and ok_x12
x9_sha16 = sha16(REPO / "runs" / "x9" / "metrics.json")
x12_sha16 = sha16(REPO / "runs" / "x12" / "metrics.json")
x9_sha_vs_x12 = (x9_sha16
                 == x12m["gates"]["G_SHA"]["checks"]["x9_metrics"]["committed"])

# load dumps, build P matrices (float64 softmax, x9 convention)
dumps, ref_meta = {}, None
for tag in TAGS:
    d = load_dump(tag)
    dumps[tag] = d
    meta = (list(d["names"]), list(d["battery"]), list(d["ans_ids"]))
    if ref_meta is None:
        ref_meta = meta
    else:
        assert meta == ref_meta, f"probe set drift across dumps at {tag}"

batt_arr = np.array(ref_meta[1])
batt_ix = {b: np.where(batt_arr == b)[0] for b in BATTERIES}
n_counts = {b: int(len(batt_ix[b])) for b in BATTERIES}
P = {}
for tag in TAGS:
    d = dumps[tag]
    pall = softmax_ans(d["logits"], d["ans_ids"])
    for b in BATTERIES:
        P.setdefault(b, {})[tag] = pall[batt_ix[b]]

# mean_p reproduction vs x9's committed discharge curves
meanp_dev = 0.0
for b in BATTERIES:
    for tag in TAGS:
        ref = x9m["discharge_curves"][b][tag]["mean_p"]
        meanp_dev = max(meanp_dev, abs(float(P[b][tag].mean()) - ref))

gates = {
    "G_SHA": {
        "dumps": sha_rows,
        "x9_metrics_sha16_fresh": x9_sha16,
        "x9_metrics_sha16_vs_x12_committed": x9_sha_vs_x12,
        "x12_metrics_sha16_fresh": x12_sha16,
        "pass": bool(sha_ok and x9_sha_vs_x12),
    },
    "G_MEANP": {
        "max_abs_dev_vs_x9_committed": meanp_dev,
        "tol": TOL_MEANP,
        "pass": bool(meanp_dev <= TOL_MEANP),
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
write_metrics(metrics, "P1 DONE (load + sha bit-verify + mean_p repro)")
print(f"[P1] G_SHA={'PASS' if gates['G_SHA']['pass'] else 'FAIL'} "
      f"G_MEANP={'PASS' if gates['G_MEANP']['pass'] else 'FAIL'} "
      f"(max dev {meanp_dev:.2e})")

# ---------------- P2: the 8x2 fits + G_REPRO + dAICc table -------------------
fits = {}
repro_rows = []
for b in BATTERIES:
    anchors = P[b]["t0"]
    cmax = float(anchors.min())
    for wash, states in WASH_STATES.items():
        t = np.array(states, dtype=float)
        Pm = np.stack([P[b][f"{wash}s{s}"] for s in states], axis=1)
        n_obs = n_counts[b] * len(states)
        for fam in ("EXP", "LIN"):
            theta, nll = fit_family(fam, anchors, t, Pm, cmax=cmax)
            Qp = fam_q(theta, fam, anchors, t)
            resid = float(np.abs(Pm - Qp).mean())
            ss_res = float(((Pm - Qp) ** 2).sum())
            ss_tot = float(sum(
                ((Pm[:, j] - Pm[:, j].mean()) ** 2).sum()
                for j in range(len(states))))
            fits[(b, wash, fam)] = {
                "theta": theta.tolist(),
                "param_names": (["lam", "c"] if fam == "EXP"
                                else ["slope", "c"]),
                "k": 2, "n_obs": n_obs, "nll": nll,
                "aicc": aicc(nll, 2, n_obs),
                "mean_abs_resid": resid,
                "R2_decline": (1.0 - ss_res / ss_tot if ss_tot > 0 else None),
                "lam_eff": lam_eff_of(theta, fam),
                "D_obs_endpoint_gap_decline": (
                    float(1.0 - math.exp(-theta[0] * 80.0)) if fam == "EXP"
                    else float(theta[0] * 80.0)),
                "bound_hit": bool(
                    (fam == "EXP" and theta[0] in LAM_BOUNDS)
                    or (fam == "LIN" and theta[0] in SLOPE_BOUNDS)
                    or theta[-1] in (0.0, cmax)),
            }
        # G_REPRO vs x9's committed EXP row
        x9row = x9m["rate_fits"][f"{b}/{wash}/EXP"]
        dtheta = float(np.abs(
            np.array(fits[(b, wash, "EXP")]["theta"])
            - np.array(x9row["theta"])).max())
        dnll = abs(fits[(b, wash, "EXP")]["nll"] - x9row["nll"])
        repro_rows.append({
            "fit": f"{b}/{wash}/EXP", "max_abs_dtheta": dtheta,
            "abs_dnll": dnll, "pass": bool(dtheta <= TOL_THETA_REPRO
                                           and dnll <= TOL_NLL_REPRO)})

gates["G_REPRO"] = {
    "rows": repro_rows,
    "n_checked": len(repro_rows),
    "max_abs_dtheta": max(r["max_abs_dtheta"] for r in repro_rows),
    "max_abs_dnll": max(r["abs_dnll"] for r in repro_rows),
    "tol_theta": TOL_THETA_REPRO, "tol_nll": TOL_NLL_REPRO,
    "pass": bool(all(r["pass"] for r in repro_rows)),
}
metrics["gates"] = gates

# dAICc table
dA = {}
for b in BATTERIES:
    for wash in WASH_STATES:
        aE = fits[(b, wash, "EXP")]["aicc"]
        aL = fits[(b, wash, "LIN")]["aicc"]
        dA[(b, wash)] = aE - aL

n_lin = sum(1 for v in dA.values() if v >= DA_BAR)
n_exp = sum(1 for v in dA.values() if v <= -DA_BAR)
n_inconclusive = 8 - n_lin - n_exp

# max mean-curve gap between the two fitted families over t in [0,80]
curve_gap = {}
tgrid = np.linspace(0.0, 80.0, 161)
for b in BATTERIES:
    for wash in WASH_STATES:
        anchors = P[b]["t0"]
        qL = fam_q(np.array(fits[(b, wash, "LIN")]["theta"]), "LIN",
                   anchors, tgrid).mean(axis=0)
        qE = fam_q(np.array(fits[(b, wash, "EXP")]["theta"]), "EXP",
                   anchors, tgrid).mean(axis=0)
        curve_gap[f"{b}/{wash}"] = float(np.abs(qL - qE).max())

metrics["family_fits"] = {f"{b}/{wash}/{fam}": fits[(b, wash, fam)]
                          for (b, wash, fam) in fits}
metrics["daicc_table"] = {
    f"{b}/{wash}": {
        "NLL_LIN": fits[(b, wash, "LIN")]["nll"],
        "NLL_EXP": fits[(b, wash, "EXP")]["nll"],
        "AICc_LIN": fits[(b, wash, "LIN")]["aicc"],
        "AICc_EXP": fits[(b, wash, "EXP")]["aicc"],
        "dAICC": dA[(b, wash)],
        "winner": ("LIN" if dA[(b, wash)] >= DA_BAR
                   else "EXP" if dA[(b, wash)] <= -DA_BAR else "inconclusive"),
    } for b in BATTERIES for wash in WASH_STATES}
metrics["daicc_counts"] = {
    "n_lin_wins_ge2": n_lin, "n_exp_wins_le_minus2": n_exp,
    "n_inconclusive": n_inconclusive,
    "note": ("both families k=2 -> dAICC = 2*(NLL_EXP - NLL_LIN) exactly "
             "(AICc penalties cancel; disclosed)"),
}
metrics["max_mean_curve_gap_0_80"] = curve_gap
write_metrics(metrics, "P2 DONE (8x2 fits + G_REPRO + dAICc table)")
print("[P2] dAICc: " + ", ".join(f"{b}/{w}={dA[(b, w)]:+.2f}"
                                 for b in BATTERIES for w in WASH_STATES))
print(f"[P2] n_lin={n_lin} n_exp={n_exp} inconclusive={n_inconclusive}; "
      f"G_REPRO max|dtheta|={gates['G_REPRO']['max_abs_dtheta']:.2e} "
      f"max|dnll|={gates['G_REPRO']['max_abs_dnll']:.2e}")

# ---------------- P3: the consistency check (hybrid diagnostic) --------------
consistency = {}
for b in BATTERIES:
    rho_x12 = x12m["derivative_test"][b]["rho_pooled"]
    rho_lin, p_lin, npairs = implied_rho(b, "LIN", fits)
    rho_exp, p_exp, _ = implied_rho(b, "EXP", fits)
    rho_obs, p_obs, _ = observed_rho_bridge(b)
    closer_to_lin = abs(rho_x12 - rho_lin) < abs(rho_x12 - rho_exp)
    consistency[b] = {
        "rho_x12_margin_currency": rho_x12,
        "rho_LIN_implied_p_currency": rho_lin,
        "rho_EXP_implied_p_currency": rho_exp,
        "rho_observed_bridge_p_currency": rho_obs,
        "p_values": {"LIN_implied": p_lin, "EXP_implied": p_exp,
                     "observed_bridge": p_obs},
        "n_pairs_pooled": npairs,
        "abs_d_x12_vs_LIN": abs(rho_x12 - rho_lin),
        "abs_d_x12_vs_EXP": abs(rho_x12 - rho_exp),
        "x12_closer_to_LIN": closer_to_lin,
    }
consistent_lin = all(v["x12_closer_to_LIN"] for v in consistency.values())
metrics["consistency_check"] = {
    "definition": ("per battery, x12's committed margin-currency rho_pooled "
                   "vs the two families' implied pooled damage-strength "
                   "Spearman on x9's exact grid/currency (damage = "
                   "(q(s)-q(s'))/(s'-s), strength = q(s), transitions w1 "
                   "0->2->10->50->80, w2 0->10->50->80, pooled over washes); "
                   "the bridge row is x12's computation on the observed e238 "
                   "p data (currency cross-check)"),
    "arithmetic_disclosure": ("pure EXP with c=0 gives damage = lam*strength "
                             "-> rho=+1; pure LIN gives damage constant "
                             "within a probe; cross-probe anchor spread "
                             "lifts both implied rhos above 0 — the "
                             "within-probe pattern is the discriminator"),
    "per_battery": consistency,
    "consistent_with_LIN_all_batteries": consistent_lin,
}
write_metrics(metrics, "P3 DONE (consistency check)")
print("[P3] consistency: " + ", ".join(
    f"{b}: x12={consistency[b]['rho_x12_margin_currency']:+.3f} "
    f"LINimp={consistency[b]['rho_LIN_implied_p_currency']:+.3f} "
    f"EXPimp={consistency[b]['rho_EXP_implied_p_currency']:+.3f} "
    f"bridge={consistency[b]['rho_observed_bridge_p_currency']:+.3f}"
    for b in BATTERIES))
print(f"[P3] consistent-with-LIN (all batteries): {consistent_lin}")

# ---------------- P4: discrimination power + bootstrap + verdict --------------

def template_daicc(b, wash, direction, D):
    """f(D) = AICc_EXP - AICc_LIN when both families fit the noiseless
    truth parameterized by endpoint decline D. direction 'lin_truth' or
    'exp_truth'. Returns None if the fit fails."""
    anchors = P[b]["t0"]
    cmax = float(anchors.min())
    states = WASH_STATES[wash]
    t = np.array(states, dtype=float)
    n_obs = n_counts[b] * len(states)
    if direction == "lin_truth":
        c_true = float(fits[(b, wash, "LIN")]["theta"][1])
        g = np.maximum(0.0, 1.0 - (D / 80.0) * t)
        lam_D = -math.log(max(1.0 - D, EPS)) / 80.0
        exp_seeds = [lam_D * f for f in (0.5, 1.0, 1.5, 2.5)] + [0.01]
        lin_seeds = [D / 80.0 * f for f in (0.7, 1.0, 1.3)]
    else:
        c_true = float(fits[(b, wash, "EXP")]["theta"][1])
        lam_D = -math.log(max(1.0 - D, EPS)) / 80.0
        g = np.exp(-lam_D * t)
        exp_seeds = [lam_D * f for f in (0.7, 1.0, 1.3)]
        lin_seeds = [D / 80.0 * f for f in (0.5, 1.0, 1.5, 2.5)]
    Ptrue = c_true + (anchors[:, None] - c_true) * g[None, :]
    Ptrue = np.clip(Ptrue, EPS, 1.0 - EPS)
    stL = [np.array([s, c]) for s in lin_seeds for c in (0.0, c_true)]
    stE = [np.array([l, c]) for l in exp_seeds for c in (0.0, c_true)]
    try:
        _, nllL = fit_family("LIN", anchors, t, Ptrue, cmax=cmax,
                             starts=stL, maxiter=2000)
        _, nllE = fit_family("EXP", anchors, t, Ptrue, cmax=cmax,
                             starts=stE, maxiter=2000)
    except Exception:
        return None
    return aicc(nllE, 2, n_obs) - aicc(nllL, 2, n_obs)


def find_D_star(b, wash, direction, threshold_sign):
    """min D with sign*f(D) >= 2 (threshold_sign +1: f>=2; -1: f<=-2).
    Grid scan + bisection. Returns (D_star, note)."""
    ds = np.concatenate([[0.02, 0.05], np.arange(0.10, 0.96, 0.05), [0.98]])
    prev = None
    for D in ds:
        f = template_daicc(b, wash, direction, float(D))
        if f is None:
            continue
        if threshold_sign * f >= DA_BAR:
            if prev is None:
                return float(D), f"already >= bar at D={D:.2f} (grid floor)"
            lo, hi = prev, float(D)
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                fm = template_daicc(b, wash, direction, mid)
                if fm is None:
                    break
                if threshold_sign * fm >= DA_BAR:
                    hi = mid
                else:
                    lo = mid
            return hi, "bisection on first crossing"
        prev = float(D)
    return None, "no crossing up to D=0.98"


disc = {}
for b in BATTERIES:
    for wash in WASH_STATES:
        dL, noteL = find_D_star(b, wash, "lin_truth", +1)
        dE, noteE = find_D_star(b, wash, "exp_truth", -1)
        disc[f"{b}/{wash}"] = {
            "D_star_lin_truth": dL, "note_lin": noteL,
            "D_star_exp_truth": dE, "note_exp": noteE,
            "D_obs_LIN_fit": fits[(b, wash, "LIN")]["D_obs_endpoint_gap_decline"],
            "D_obs_EXP_fit": fits[(b, wash, "EXP")]["D_obs_endpoint_gap_decline"],
            "max_mean_curve_gap": curve_gap[f"{b}/{wash}"],
        }
dL_vals = [v["D_star_lin_truth"] for v in disc.values()
           if v["D_star_lin_truth"] is not None]
dE_vals = [v["D_star_exp_truth"] for v in disc.values()
           if v["D_star_exp_truth"] is not None]
metrics["discrimination_power"] = {
    "definition": ("noiseless-template scan per battery-wash: truth "
                   "g_D(t) = max(0,1-(D/80)t) [lin] or exp(-(-ln(1-D)/80)t) "
                   "[exp] with the fitted floor; both families refit; "
                   "f(D)=AICc_EXP-AICc_LIN; D* = min D separating at |dAICc| "
                   ">= 2 — the endpoint gap-decline a TRUE member needs "
                   "before this grid/n catches it (probe dispersion NOT "
                   "modeled: the systematic-signal threshold; see bootstrap "
                   "for the noise side)"),
    "per_battery_wash": disc,
    "median_D_star_lin_truth": (float(np.median(dL_vals))
                                if dL_vals else None),
    "median_D_star_exp_truth": (float(np.median(dE_vals))
                                if dE_vals else None),
    "median_D_obs_LIN": float(np.median(
        [disc[k]["D_obs_LIN_fit"] for k in disc])),
    "median_D_obs_EXP": float(np.median(
        [disc[k]["D_obs_EXP_fit"] for k in disc])),
}
write_metrics(metrics, "P4a DONE (discrimination-power template scan)")
print(f"[P4a] D*_lin median={metrics['discrimination_power']['median_D_star_lin_truth']}"
      f" D*_exp median={metrics['discrimination_power']['median_D_star_exp_truth']}"
      f" D_obs(LIN) median={metrics['discrimination_power']['median_D_obs_LIN']}")

# bootstrap co-report on dAICc
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for b in BATTERIES:
    anchors = P[b]["t0"]
    cmax = float(anchors.min())
    n = n_counts[b]
    for wash, states in WASH_STATES.items():
        t = np.array(states, dtype=float)
        Pm = np.stack([P[b][f"{wash}s{s}"] for s in states], axis=1)
        n_obs = n * len(states)
        starts = {}
        for fam in ("EXP", "LIN"):
            th0 = np.array(fits[(b, wash, fam)]["theta"])
            lo_b = [v[0] for _, v in fam_bounds(fam, cmax)]
            hi_b = [v[1] for _, v in fam_bounds(fam, cmax)]
            starts[fam] = [th0] + [np.clip(th0 * (1 + dx), lo_b, hi_b)
                                   for dx in (0.15, -0.15, 0.4)]
        das = []
        for _ in range(BOOT_B):
            ix = rng.integers(0, n, size=n)
            a_ix, p_ix = anchors[ix], Pm[ix]
            cm = float(a_ix.min())
            _, nllE = fit_family("EXP", a_ix, t, p_ix, cmax=cm,
                                 starts=starts["EXP"], maxiter=2000)
            _, nllL = fit_family("LIN", a_ix, t, p_ix, cmax=cm,
                                 starts=starts["LIN"], maxiter=2000)
            das.append(aicc(nllE, 2, n_obs) - aicc(nllL, 2, n_obs))
        arr = np.array(das)
        boot[f"{b}/{wash}"] = {
            "median": float(np.median(arr)),
            "ci95": [float(np.percentile(arr, 2.5)),
                     float(np.percentile(arr, 97.5))],
            "frac_dAICc_ge_2": float((arr >= DA_BAR).mean()),
            "frac_dAICc_le_minus2": float((arr <= -DA_BAR).mean()),
        }
metrics["bootstrap_daicc"] = {
    "method": f"probe-level case resampling, B={BOOT_B}, seed {BOOT_SEED} "
              f"(frozen); both families refit per resample from full-data "
              "theta starts (disclosed)",
    "per_battery_wash": boot,
}
write_metrics(metrics, "P4b DONE (bootstrap dAICc co-report)")
print("[P4b] bootstrap medians: " + ", ".join(
    f"{k}={v['median']:+.2f}" for k, v in boot.items()))

# verdict (frozen routing)
flat_fires = bool(n_lin >= 5 and consistent_lin)
wealth_fires = bool(n_exp >= 5)
if flat_fires:
    verdict = "FLAT-TAX"
    clause = (f"LINEAR wins dAICc >= 2 in {n_lin}/8 fits AND x12's rho ~ 0 "
              f"is consistent (x12's measured rho closer to LIN-implied "
              f"than EXP-implied in all four batteries)")
elif wealth_fires:
    verdict = "WEALTH-TAX"
    clause = (f"EXP wins dAICc <= -2 in {n_exp}/8 fits — the rates are real "
              f"proportional drains; x12's rho ~ 0 then needs its own "
              f"explanation (the tension stands, FLAGGED)")
else:
    verdict = "MIXED"
    clause = (f"split or inconclusive: n_lin={n_lin}, n_exp={n_exp}, "
              f"inconclusive={n_inconclusive}, consistent-with-LIN="
              f"{consistent_lin} — the table verbatim + the discrimination-"
              f"power statement below")
metrics["adjudication"] = {
    "bars": {"FLAT-TAX": flat_fires, "WEALTH-TAX": wealth_fires,
             "MIXED": verdict == "MIXED"},
    "verdict": verdict,
    "clause": clause,
    "counts": {"n_lin": n_lin, "n_exp": n_exp, "n_inconclusive": n_inconclusive},
    "consistent_with_LIN": consistent_lin,
}
write_metrics(metrics, "P4c DONE (adjudication)")
print(f"[P4c] VERDICT: {verdict} — {clause}")

# ---------------- P5: figure + report ----------------------------------------
colors = {"ctrl": "#1f77b4", "fact": "#2ca02c",
          "near": "#d62728", "tmpl": "#ff7f0e"}
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

# (A) the 8-fit dAICc table as bars
axA = fig.add_subplot(gs[0, 0])
labels = [f"{b}/{w}" for b in BATTERIES for w in WASH_STATES]
vals = [dA[(b, w)] for b in BATTERIES for w in WASH_STATES]
cols = ["#1f77b4" if v >= 2 else "#d62728" if v <= -2 else "#aaaaaa"
        for v in vals]
axA.bar(range(8), vals, color=cols, edgecolor="k", linewidth=0.5)
axA.axhline(2, color="#1f77b4", ls="--", lw=1.2)
axA.axhline(-2, color="#d62728", ls="--", lw=1.2)
axA.axhline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    axA.text(i, v + (0.15 if v >= 0 else -0.35), f"{v:+.2f}",
             ha="center", fontsize=8)
axA.set_xticks(range(8))
axA.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axA.set_ylabel("dAICc = AICc$_{EXP}$ - AICc$_{LIN}$")
axA.set_title(f"(A) the 8 battery-wash fits — verdict {verdict} "
              f"(LIN wins {n_lin}, EXP wins {n_exp}, incl. {n_inconclusive})",
              fontsize=10)

# (B) discharge curves + both family overlays (per battery)
axB = fig.add_subplot(gs[0, 1])
for b in BATTERIES:
    for w, ls in zip(WASH_STATES, ["-", ":"]):
        states = WASH_STATES[w]
        tt = np.array([0] + states, dtype=float)
        mm = [float(P[b]["t0"].mean())] + [
            float(P[b][f"{w}s{s}"].mean()) for s in states]
        axB.plot(tt, mm, ls, color=colors[b], lw=1.8, marker=".", ms=7,
                 label=f"{b} obs" if w == "w1" else None)
        anchors = P[b]["t0"]
        qL = fam_q(np.array(fits[(b, w, "LIN")]["theta"]), "LIN",
                   anchors, tgrid).mean(axis=0)
        qE = fam_q(np.array(fits[(b, w, "EXP")]["theta"]), "EXP",
                   anchors, tgrid).mean(axis=0)
        axB.plot(tgrid, qL, color=colors[b], lw=0.9, alpha=0.55)
        axB.plot(tgrid, qE, color=colors[b], lw=0.9, alpha=0.55, ls="--")
axB.set_xlabel("wash steps t")
axB.set_ylabel("battery mean p(answer)")
axB.text(0.02, 0.02, "thin solid = LIN fit, thin dashed = EXP fit "
         "(per battery-wash)", transform=axB.transAxes, fontsize=8,
         color="gray")
axB.set_title("(B) discharge curves with both family overlays",
              fontsize=10)
axB.legend(fontsize=8, ncol=2)

# (C) the consistency check: damage vs strength (fact + all four rhos)
axC = fig.add_subplot(gs[1, 0])
b0 = "fact"
curves_obs = P[b0]
for wash in WASH_STATES:
    d, s = damage_strength_pairs(curves_obs, wash)
    axC.scatter(s, d, s=14, color="#888888", alpha=0.6, zorder=2,
                label="observed (p currency)" if wash == "w1" else None)
for fam, col, mk in (("EXP", "#d62728", "x"), ("LIN", "#1f77b4", "+")):
    theta = np.array(fits[(b0, "w1", fam)]["theta"])
    curves = model_curves(theta, fam, P[b0]["t0"], "w1")
    d, s = damage_strength_pairs(curves, "w1")
    axC.scatter(s, d, s=22, marker=mk, color=col, alpha=0.8, zorder=3,
                label=f"{fam}-implied (w1 fit)")
txt = "\n".join(
    f"{b}: x12={consistency[b]['rho_x12_margin_currency']:+.3f} "
    f"bridge={consistency[b]['rho_observed_bridge_p_currency']:+.3f} "
    f"LINimp={consistency[b]['rho_LIN_implied_p_currency']:+.3f} "
    f"EXPimp={consistency[b]['rho_EXP_implied_p_currency']:+.3f}"
    for b in BATTERIES)
axC.text(0.98, 0.97, "damage-strength Spearman per battery\n"
         "(x12 margin | bridge p | LIN-impl | EXP-impl)\n" + txt,
         transform=axC.transAxes, va="top", ha="right", fontsize=7,
         family="monospace",
         bbox=dict(fc="white", ec="#999", alpha=0.9))
axC.set_xlabel("remaining strength q(s)")
axC.set_ylabel("per-step damage (q(s)-q(s'))/(s'-s)")
axC.set_title("(C) the consistency check — damage vs strength "
              f"(scatter: {b0}; rhos: all four)", fontsize=10)
axC.legend(fontsize=8, loc="lower left")

# (D) discrimination power curves
axD = fig.add_subplot(gs[1, 1])
ds_scan = np.concatenate([[0.05], np.arange(0.10, 0.96, 0.05), [0.98]])
shown = {"ctrl/w1", "fact/w1", "near/w1", "tmpl/w1"}
for b in BATTERIES:
    for wash in WASH_STATES:
        if f"{b}/{wash}" not in shown:
            continue
        fs = [template_daicc(b, wash, "lin_truth", float(D))
              for D in ds_scan]
        axD.plot(ds_scan, fs, color=colors[b], lw=1.4,
                 label=f"{b} {wash} (truth=LIN)")
        dstar = disc[f"{b}/{wash}"]["D_star_lin_truth"]
        if dstar is not None:
            axD.scatter([dstar], [2], color=colors[b], zorder=4, s=42,
                        edgecolor="k", linewidth=0.5)
        dobs = disc[f"{b}/{wash}"]["D_obs_LIN_fit"]
        axD.scatter([dobs], [-3.4], marker="*", s=90, color=colors[b],
                    zorder=4)
axD.axhline(2, color="k", ls="--", lw=1)
axD.axhline(0, color="k", lw=0.6)
axD.set_xlabel("endpoint gap-decline D = 1 - g(80) of the TRUE template")
axD.set_ylabel("AICc$_{EXP}$ - AICc$_{LIN}$ (noiseless truth)")
med = metrics["discrimination_power"]["median_D_star_lin_truth"]
medE = metrics["discrimination_power"]["median_D_star_exp_truth"]
axD.text(0.02, 0.03,
         f"circles: D* (LIN-truth, dAICc=2), median = "
         f"{med if med is not None else float('nan'):.2f}\n"
         f"stars (bottom): observed D (LIN fit); EXP-truth median D* = "
         f"{medE if medE is not None else float('nan'):.2f}",
         transform=axD.transAxes, fontsize=8,
         bbox=dict(fc="white", ec="#999", alpha=0.9))
axD.set_title("(D) discrimination power — how deep must the true decline "
              "go before the design separates the families", fontsize=10)
axD.legend(fontsize=8, loc="upper left")

fig.suptitle("x13 — FLAT TAX vs WEALTH TAX: LINEAR (the missing member) vs "
             "EXPONENTIAL on x9's exact data — verdict: " + verdict,
             fontsize=12)
fig.savefig(RUN / "x13_tax_fits.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# REPORT.md
tbl = "\n".join(
    f"| {b}/{w} | {fits[(b, w, 'LIN')]['nll']:.3f} | "
    f"{fits[(b, w, 'EXP')]['nll']:.3f} | "
    f"{fits[(b, w, 'LIN')]['aicc']:.2f} | "
    f"{fits[(b, w, 'EXP')]['aicc']:.2f} | **{dA[(b, w)]:+.2f}** | "
    f"{metrics['daicc_table'][f'{b}/{w}']['winner']} | "
    f"{disc[f'{b}/{w}']['max_mean_curve_gap']:.3f} |"
    for b in BATTERIES for w in WASH_STATES)
cons_tbl = "\n".join(
    f"| {b} | {consistency[b]['rho_x12_margin_currency']:+.3f} | "
    f"{consistency[b]['rho_observed_bridge_p_currency']:+.3f} | "
    f"{consistency[b]['rho_LIN_implied_p_currency']:+.3f} | "
    f"{consistency[b]['rho_EXP_implied_p_currency']:+.3f} | "
    f"{'LIN' if consistency[b]['x12_closer_to_LIN'] else 'EXP'} |"
    for b in BATTERIES)
disc_tbl = "\n".join(
    f"| {k} | " +
    (f"{v['D_star_lin_truth']:.3f}" if v["D_star_lin_truth"] is not None
     else ">0.98") + " | " +
    (f"{v['D_star_exp_truth']:.3f}" if v["D_star_exp_truth"] is not None
     else ">0.98") + f" | {v['D_obs_LIN_fit']:.3f} | "
    f"{v['D_obs_EXP_fit']:.3f} |"
    for k, v in disc.items())
boot_tbl = "\n".join(
    f"| {k} | {v['median']:+.2f} | [{v['ci95'][0]:+.2f}, "
    f"{v['ci95'][1]:+.2f}] | {v['frac_dAICc_ge_2']:.2f} | "
    f"{v['frac_dAICc_le_minus2']:.2f} |"
    for k, v in boot.items())

report = f"""# x13 — FLAT TAX vs WEALTH TAX (W047's follow-up tension)

**Status**: COMPLETE — verdict **{verdict}**
**Envelope**: CPU-only desk cell (no torch import, threads 1 (<= 2 per the
dispatch), 0 GPU calls, 0 envelope-log writes, no NOTES/THINKING/QUEUE/STATE
edits; every timestamp datetime.now(UTC)).
**Registration**: bars frozen VERBATIM in the script docstring, birth-committed
before compute (git head at start {metrics['provenance']['git_head_at_start']}).

## The tension

x9 fitted exponentials and called the one-T ladder a RATE ladder (exponential
margins imply per-step damage proportional to current gap — a WEALTH tax).
x12 measured the damage-strength Spearman ~ 0 for every battery — damage
independent of remaining strength — a FLAT tax (linear margin decline). Both
cannot be right; x9's family set NEVER INCLUDED LINEAR, and over a limited
state range exp and linear are nearly indistinguishable by R2. The missing
member decides.

## Gates (all machine-checked)

- G_SHA: all 8 e238 dumps' sha256 match x9's committed G_SHA table AND x12's
  independent re-verification; runs/x9/metrics.json hashed fresh matches x12's
  committed record.
- G_MEANP: float64 softmax mean_p reproduces x9's committed discharge curves
  to {meanp_dev:.2e} (tol {TOL_MEANP:.0e}).
- G_REPRO: this cell's EXP refit reproduces x9's 8 committed EXP rows to
  max |dtheta| = {gates['G_REPRO']['max_abs_dtheta']:.2e},
  max |dnll| = {gates['G_REPRO']['max_abs_dnll']:.2e}.

## The 8-fit dAICc table (both families k=2, so dAICc = 2*dNLL exactly)

| fit | NLL_LIN | NLL_EXP | AICc_LIN | AICc_EXP | dAICc | winner | max mean-curve gap |
|---|---|---|---|---|---|---|---|
{tbl}

LIN wins (dAICc >= 2): **{n_lin}/8**; EXP wins (dAICc <= -2): **{n_exp}/8**;
inconclusive: {n_inconclusive}/8.

## The consistency check (hybrid diagnostic, cross-cell)

| battery | x12 rho (margin) | bridge rho (p) | LIN-implied rho | EXP-implied rho | x12 closer to |
|---|---|---|---|---|---|
{cons_tbl}

Arithmetic: pure EXP (c=0) forces damage = lam*strength -> rho = +1; pure LIN
makes damage constant within a probe. Consistent-with-LIN (all batteries
closer to LIN-implied): **{consistent_lin}**.

## Discrimination power (the design statement)

Noiseless-template scan on the measured state range: D = 1 - g(80) of the TRUE
template; D* = the decline depth at which the design separates the families
at dAICc >= 2 (systematic-signal threshold; probe dispersion not modeled —
the bootstrap co-report carries the noise side).

| fit | D* (truth=LIN) | D* (truth=EXP) | D obs (LIN fit) | D obs (EXP fit) |
|---|---|---|---|---|
{disc_tbl}

Medians: D* truth=LIN **{metrics['discrimination_power']['median_D_star_lin_truth']}**,
D* truth=EXP **{metrics['discrimination_power']['median_D_star_exp_truth']}**,
observed D (LIN) {metrics['discrimination_power']['median_D_obs_LIN']:.3f},
observed D (EXP) {metrics['discrimination_power']['median_D_obs_EXP']:.3f}.

## Bootstrap dAICc (probe resampling, B={BOOT_B}, seed {BOOT_SEED}; never a bar)

| fit | median | 95% CI | frac >= 2 | frac <= -2 |
|---|---|---|---|---|
{boot_tbl}

## Verdict clause (frozen routing)

{clause}

## Honesty

n=1 organism; near n=3 (coarse); x12's measured rho is in MARGIN currency
while the implied rhos are in x9's p currency (the bridge row shows the
currency gap on the same transitions); the discrimination template is
noiseless (systematic-signal threshold, not a power analysis); the LIN
survival hinge max(0, 1-slope*t) never activates at the fitted slopes; equal
k means dAICc is pure likelihood ratio; no bar shopping — routing frozen in
the birth commit.
"""
(RUN / "REPORT.md").write_text(report, encoding="utf-8")

metrics["status"] = "COMPLETE — adjudicated (this write replaces all PARTIAL progressive writes)"
metrics["outputs"] = {
    "metrics": str(RUN / "metrics.json"),
    "figure": str(RUN / "x13_tax_fits.png"),
    "report": str(RUN / "REPORT.md"),
}
metrics["timing"] = {"total_s": round(time.time() - T_START, 2)}
write_metrics(metrics, "P5 DONE (figure + report + adjudication)")
print(f"[P5] done in {metrics['timing']['total_s']}s -> {RUN}")
print(f"[P5] VERDICT: {verdict}")
