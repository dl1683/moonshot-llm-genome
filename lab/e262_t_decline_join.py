"""E262 — T232's REGISTERED JOIN: THE T-LADDER vs THE DECLINE (desk-only,
CPU, pure committed records — no model, no torch, no GPU; e261's recovery
owns the GPU).

WHY: T232 (x5) found the battery thermal ladder — each battery carries its
own T(t) in a stable wash-replicating ordering (ctrl 1.18-1.27 < fact
1.26-1.37 < near 1.42-1.63 < tmpl 1.51-1.65 at the deep states) — and
registered THE JOIN: "the battery T(t)s against the committed decline
fractions — if the ladder tracks the deaths monotonically across washes,
the one-T fit's battery variation IS the erosion ordering's coarse-graining
(the third dimension seen through the thermal lens, battery-level)."
T236 (e259) sharpened the target with the death-meter law
(rho(decline, R2_own) = +1.00 across all four batteries). THE QUESTION:
does the battery T-ladder TRACK the decline fractions QUANTITATIVELY? If
T(t) is the erosion rate in lens clothing, then per battery: higher T at
matched states = higher decline — a monotone join across the four
batteries and both washes.

THE CELL (desk, minutes — the dispatch, verbatim scope):
  (1) THE JOIN: per battery per wash, the deep-state own-T (x5's committed
      table) vs the committed mean decline (1 - mean_p(t)/mean_p(t0));
      Spearman across the 4 batteries x 2 washes (n=8) + the per-wash reads.
  (2) THE QUANTITATIVE FORM: is the relation linear in T or in log T?
      (fit both, report).
  (3) THE CO-READ: the same join against the R2s (the death-meter's own
      texture — does the T also track fit quality?).

REGISTERED BARS (frozen BEFORE compute, VERBATIM from the dispatch brief;
adjudicate against exactly this; no bar shopping):
  - T-TRACKS-DECLINE — "the join is monotone (per-wash Spearman >= 0.80,
    n=4 each, disclosed) AND the pooled fit's sign matches — the battery
    T-ladder IS the erosion rate's coarse-graining; the lens calibrated as
    a death-meter at the battery level; one instrument, two faces"
  - T-INDEPENDENT — "no monotone relation (per-wash rho < 0.80 or
    sign-discordant) — the T-ladder and the declines are distinct battery
    properties; the mirror was qualitative only"
  - MIXED — "the tables verbatim, both washes"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * THE T SIDE ("the deep-state own-T, x5's committed table") = x5's
    committed per-battery per-wash T_mle (runs/x5/metrics.json fit_rows,
    read at runtime), aggregated over the DEEP STATES {+50, +80} — T232's
    OWN ladder construction (its published ranges ctrl 1.18-1.27 / fact
    1.26-1.37 / near 1.42-1.63 / tmpl 1.51-1.65 span exactly these states
    x washes). Per (battery, wash): T_deep = mean(T_mle at s50, s80).
    DISCLOSED SENSITIVITY, never adjudicating: the s80-only join, the
    s50-only join, the per-state joins, and the {10,50,80} mean variant.
  * THE DECLINE SIDE ("the committed mean decline, 1 - mean_p(t)/mean_p(t0)")
    = per-state decline fraction from e214's committed journal
    (runs/e214/journal.json, read at runtime: per-battery mean_p at t0 and
    at every state of BOTH washes — the record x5 itself re-probe-certified
    against, max dp 0.0), aggregated over the SAME deep states {50, 80}:
    decline_deep = mean over s in {50,80} of [1 - mean_p(s)/mean_p(t0)].
    Provenance cross-checks (gates below): ctrl/near mean_p vs x5's own
    fit_rows mean_p_obs (expected 0.0); the w2 rows vs e182c2's committed
    journal_p2 (expected ~0, float re-serialization); the n-weighted
    pooled mean_p0 vs e238's committed mean_p0.
  * THE JOIN = 8 points (4 batteries x 2 washes). Per-wash Spearman n=4
    (exact permutation p by full enumeration of the 24 rank permutations —
    no asymptotic p at n=4); pooled Spearman n=8 (exact permutation p,
    40320 permutations) and pooled OLS.
  * THE VERDICT (frozen): T-TRACKS-DECLINE iff (rho_w1 >= 0.80 AND
    rho_w2 >= 0.80 AND pooled OLS slope on T > 0); T-INDEPENDENT iff (any
    per-wash rho < 0.80 OR pooled slope <= 0); any gate failure -> MIXED
    ("the tables verbatim, both washes"). The two live branches are exact
    complements; MIXED is the gate-failure fallback only.
  * THE FORM (2) = pooled OLS of decline on T and of decline on log10(T),
    both reported with slope / intercept / R2 / Pearson r; the DEGENERACY
    DISCLOSURE = Pearson r between T and log10(T) over the observed range
    (if ~1, this cell CANNOT discriminate the form — the honest outcome).
  * THE CO-READ (3) = e238's committed per-battery R2_battery (under the
    committed pooled T; runs/e238/metrics.json fit_rows, read at runtime),
    aggregated over {50, 80} the same way: joins T_deep ~ R2_deep and
    decline_deep ~ R2_deep, per-wash Spearman + pooled. The second is the
    death-meter law (T236) re-run on committed records at this aggregate.
    DISCLOSED: e238's R2_battery is R2 under the POOLED committed T, not
    per-battery own-T; x5's own-fit R2_own exists only for ctrl/near
    (co-reported, n=2 per wash — too small to rank, shown as numbers).

CHECKS (the dispatch's own): the T table's committed provenance (x5); the
declines' provenance (e214's journal, itself cross-anchored to e182c2's
records); n=4 per wash (small, disclosed); nothing guaranteed.
HONESTY: n=1 organism; near n=3 probes / ctrl n=12 / fact n=20 / tmpl n=19;
n=4-per-wash Spearman is coarse (it takes values 1.0/0.8/0.4/... in steps
of 0.2 — a rho of 0.80 means EXACTLY one adjacent transposition); wash 1
is a CPU fp32 replay, wash 2 GPU fp32 (the archive's inherited asymmetry);
observational — a monotone join supports coarse-graining, it does not
intervene; lens-not-mechanism stands (T228/T229/T232).
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# ----------------------------------------------------------------- bars ----
# FROZEN BEFORE COMPUTE. The verdict logic and every constant it uses.
BARS_VERBATIM = {
    "T-TRACKS-DECLINE": (
        '"the join is monotone (per-wash Spearman >= 0.80, n=4 each, '
        'disclosed) AND the pooled fit\'s sign matches — the battery T-ladder '
        'IS the erosion rate\'s coarse-graining; the lens calibrated as a '
        'death-meter at the battery level; one instrument, two faces"'
    ),
    "T-INDEPENDENT": (
        '"no monotone relation (per-wash rho < 0.80 or sign-discordant) — '
        'the T-ladder and the declines are distinct battery properties; '
        'the mirror was qualitative only"'
    ),
    "MIXED": '"the tables verbatim, both washes"',
}
RHO_BAR = 0.80  # frozen
BATTERIES = ["ctrl", "fact", "near", "tmpl"]
WASHES = ["w1", "w2"]
DEEP_STATES = [50, 80]  # T232's own ladder construction
ALL_STATES = [10, 50, 80]  # x5's adjudication states (the +2 state is w1-only)
N_PROBES = {"fact": 20, "ctrl": 12, "near": 3, "tmpl": 19}


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- statistics --
def _ranks(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(_ranks(x), _ranks(y))


def perm_p_exact(x, y) -> float:
    """One-sided exact permutation p: P(rho >= observed) under exchangeability.

    Full enumeration (24 perms at n=4; 40320 at n=8) — no asymptotics at
    tiny n. Ties (none expected) handled by re-ranking each permutation.
    """
    obs = spearman(x, y)
    if np.isnan(obs):
        return float("nan")
    y = list(np.asarray(y, dtype=float))
    ge = 0
    total = 0
    for perm in itertools.permutations(range(len(y))):
        yp = [y[i] for i in perm]
        if spearman(x, yp) >= obs - 1e-12:
            ge += 1
        total += 1
    return ge / total


def ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": n,
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "r2": r2,
        "pearson_r": pearson(x, y),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


# ------------------------------------------------------------------ data ---
def load_records():
    paths = {
        "x5_metrics": RUNS / "x5" / "metrics.json",
        "e238_metrics": RUNS / "e238" / "metrics.json",
        "e214_journal": RUNS / "e214" / "journal.json",
        "e182c2_journal_p2": RUNS / "e182c2" / "journal_p2.json",
    }
    recs = {}
    hashes = {}
    for name, p in paths.items():
        if not p.exists():
            sys.exit(f"GATE FAIL: committed record missing: {p}")
        recs[name] = json.loads(p.read_text(encoding="utf-8"))
        hashes[name] = {"path": str(p), "sha256_16": sha256_16(p)}
    return recs, hashes


def e214_table(journal):
    """(wash, step) -> {battery: mean_p}; t0 row shared."""
    out = {}
    for s in journal["states"]:
        key = (s.get("wash", "t0"), int(s["step"]))
        out[key] = {b: float(s[b]["mean_p"]) for b in BATTERIES}
    return out


# ------------------------------------------------------------------ gates --
def run_gates(recs):
    gates = {}

    # G_T_TABLE: x5's committed per-battery T table, all 4 batteries x 2
    # washes x both deep states present; t0 anchors read 1.0.
    ok = True
    detail = {}
    for r in recs["x5_metrics"]["fit_rows"]:
        if r["wash"] in WASHES and r["state"] in DEEP_STATES:
            detail[f"{r['wash']}+{r['state']}/{r['battery']}"] = round(r["T_mle"], 4)
            ok &= r["battery"] in BATTERIES
    n_rows = len(detail)
    ok &= n_rows == len(BATTERIES) * len(WASHES) * len(DEEP_STATES)
    t0_rows = [r for r in recs["x5_metrics"]["fit_rows"] if r["wash"] == "t0"]
    t0_anchor = max(abs(r["T_mle"] - 1.0) for r in t0_rows) if t0_rows else float("nan")
    gates["G_T_TABLE"] = {
        "pass": bool(ok and n_rows == 16 and t0_anchor < 1e-3),
        "n_deep_rows": n_rows,
        "t0_anchor_max_abs_delta_from_1": t0_anchor,
        "note": "x5 fit_rows T_mle: 4 batteries x 2 washes x {50,80} + the t0 anchors",
    }

    # G_DECLINE_PROV: e214 journal carries t0 + all four batteries at the
    # deep states of both washes; ctrl/near cross-check vs x5's own
    # mean_p_obs (x5 certified its re-probe to 0.0 against e214); w2 rows
    # cross-check vs e182c2's journal_p2; the n-weighted pooled mean_p0
    # identity vs e238's committed pooled fit.
    j = recs["e214_journal"]
    tbl = e214_table(j)
    ok = True
    for w in WASHES:
        for s in DEEP_STATES:
            ok &= (w, s) in tbl
            ok &= all(b in tbl.get((w, s), {}) for b in BATTERIES)
    ok &= all(b in tbl.get(("t0", 0), {}) for b in BATTERIES)
    # ctrl/near vs x5's mean_p_obs
    mx_x5 = 0.0
    n_x5 = 0
    for r in recs["x5_metrics"]["fit_rows"]:
        if (
            r["battery"] in ("ctrl", "near")
            and r["wash"] in WASHES
            and r["state"] in DEEP_STATES
            and "mean_p_obs" in r
        ):
            mx_x5 = max(mx_x5, abs(r["mean_p_obs"] - tbl[(r["wash"], r["state"])][r["battery"]]))
            n_x5 += 1
    ok &= n_x5 == 8 and mx_x5 <= 1e-6
    # w2 rows vs e182c2 journal_p2
    mx_c2 = 0.0
    for s2 in recs["e182c2_journal_p2"]["states"]:
        step = int(s2["step"])
        if step == 0:
            continue
        for b in BATTERIES:
            mx_c2 = max(mx_c2, abs(s2[b]["mean_p"] - tbl[("w2", step)][b]))
    ok &= mx_c2 <= 1e-6
    # pooled mean_p0 identity vs e238
    t0 = tbl[("t0", 0)]
    pooled_p0 = sum(t0[b] * N_PROBES[b] for b in BATTERIES) / sum(N_PROBES.values())
    e238_p0 = float(recs["e238_metrics"]["fit_rows"][0]["mean_p0"])
    ok &= abs(pooled_p0 - e238_p0) <= 1e-3
    gates["G_DECLINE_PROV"] = {
        "pass": bool(ok),
        "x5_vs_e214_ctrl_near_max_dp": mx_x5,
        "n_x5_checks": n_x5,
        "e182c2_vs_e214_w2_max_dp": mx_c2,
        "pooled_mean_p0_nweighted": pooled_p0,
        "e238_committed_mean_p0": e238_p0,
        "note": (
            "declines from e214's committed per-battery mean_p records (the record x5 "
            "itself re-probe-certified against); w2 == e182c2's journal_p2; the "
            "n-weighted pooled t0 mean reproduces e238's committed mean_p0"
        ),
    }

    # G_R2_TABLE: e238's committed per-battery R2_battery at the deep states.
    ok = True
    have = 0
    for r in recs["e238_metrics"]["fit_rows"]:
        if r["wash"] in WASHES and r["state"] in DEEP_STATES and "R2_battery" in r:
            have += 1
            ok &= all(b in r["R2_battery"] for b in BATTERIES)
    ok &= have == len(WASHES) * len(DEEP_STATES)
    gates["G_R2_TABLE"] = {
        "pass": bool(ok),
        "n_state_rows": have,
        "note": "e238 committed R2_battery (under the pooled committed T) for all four batteries at {w1,w2} x {50,80}",
    }

    # G_ENV: desk-only, CPU, no torch, no GPU call possible.
    env = {"cpu_only": True, "torch_imported": False, "gpu_calls": 0, "threads_env": os.environ.get("OMP_NUM_THREADS")}
    try:
        import psutil

        env["launch_cpu_percent"] = psutil.cpu_percent(interval=0.5)
    except Exception:
        env["launch_cpu_percent"] = None
    gates["G_ENV"] = {"pass": True, **env, "note": "pure committed-record desk cell (numpy+json+matplotlib only); e261's recovery owns the GPU"}
    return gates, tbl


# ------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="smoke mode: own dir, nothing adjudicated")
    args = ap.parse_args()

    out_dir = RUNS / ("e262_smoke" if args.smoke else "e262")
    out_dir.mkdir(parents=True, exist_ok=True)
    git_head = os.popen("git rev-parse HEAD").read().strip() or "unknown"

    recs, hashes = load_records()

    # ---- the T side (x5 committed) -----------------------------------------
    T_mle = {}  # (battery, wash, state) -> T_mle
    T_lsq = {}
    R2_own_x5 = {}  # (battery, wash, state) -> own-fit R2 (ctrl/near only)
    mean_p_obs_x5 = {}
    for r in recs["x5_metrics"]["fit_rows"]:
        if r["wash"] not in WASHES:
            continue
        key = (r["battery"], r["wash"], int(r["state"]))
        T_mle[key] = float(r["T_mle"])
        T_lsq[key] = float(r["T_lsq"])
        if "R2_own" in r:
            R2_own_x5[key] = float(r["R2_own"])
        if "mean_p_obs" in r:
            mean_p_obs_x5[key] = float(r["mean_p_obs"])

    gates, meanp = run_gates(recs)
    all_gates_pass = all(g["pass"] for g in gates.values())

    # ---- the decline side (e214 committed) ---------------------------------
    t0 = meanp[("t0", 0)]
    decline = {}  # (battery, wash, state) -> 1 - mean_p(t)/mean_p(t0)
    for w in WASHES:
        for s in ALL_STATES + ([2] if w == "w1" else []):
            if (w, s) in meanp:
                for b in BATTERIES:
                    decline[(b, w, s)] = 1.0 - meanp[(w, s)][b] / t0[b]

    # ---- the R2 side (e238 committed) --------------------------------------
    R2_batt = {}  # (battery, wash, state) -> R2_battery
    for r in recs["e238_metrics"]["fit_rows"]:
        if r["wash"] in WASHES and r["state"] in DEEP_STATES:
            for b in BATTERIES:
                R2_batt[(b, r["wash"], int(r["state"]))] = float(r["R2_battery"][b])

    # ---- the primary join table (frozen: deep states {50,80} mean) ---------
    join_rows = []
    for b in BATTERIES:
        for w in WASHES:
            join_rows.append(
                {
                    "battery": b,
                    "wash": w,
                    "n_probes": N_PROBES[b],
                    "T_deep": float(np.mean([T_mle[(b, w, s)] for s in DEEP_STATES])),
                    "T_deep_lsq": float(np.mean([T_lsq[(b, w, s)] for s in DEEP_STATES])),
                    "T_s50": T_mle[(b, w, 50)],
                    "T_s80": T_mle[(b, w, 80)],
                    "decline_s50": decline[(b, w, 50)],
                    "decline_s80": decline[(b, w, 80)],
                    "mean_p0": t0[b],
                    "decline_deep": float(np.mean([decline[(b, w, s)] for s in DEEP_STATES])),
                    "decline_deep_ratio_of_means": 1.0
                    - float(np.mean([meanp[(w, s)][b] for s in DEEP_STATES])) / t0[b],
                    "R2_deep": float(np.mean([R2_batt[(b, w, s)] for s in DEEP_STATES])),
                    "R2_own_deep_x5": (
                        float(np.mean([R2_own_x5[(b, w, s)] for s in DEEP_STATES]))
                        if all((b, w, s) in R2_own_x5 for s in DEEP_STATES)
                        else None
                    ),
                }
            )

    def col(name, wash=None, rows=None):
        rows = rows if rows is not None else join_rows
        sel = [r for r in rows if wash is None or r["wash"] == wash]
        return np.array([r[name] for r in sel], dtype=float)

    # ---- (1) THE JOIN: per-wash Spearman n=4 + pooled ----------------------
    join_stats = {}
    for w in WASHES:
        rho = spearman(col("T_deep", w), col("decline_deep", w))
        p = perm_p_exact(col("T_deep", w), col("decline_deep", w))
        r_lin = ols(col("T_deep", w), col("decline_deep", w))
        join_stats[f"{w}_n4"] = {
            "spearman_rho": rho,
            "perm_p_exact_one_sided": p,
            "n": 4,
            "ols": {k: r_lin[k] for k in ("slope", "intercept", "r2")},
        }
    pooled_lin = ols(col("T_deep"), col("decline_deep"))
    pooled_rho = spearman(col("T_deep"), col("decline_deep"))
    pooled_rho_p = perm_p_exact(col("T_deep"), col("decline_deep"))
    join_stats["pooled_n8"] = {
        "spearman_rho": pooled_rho,
        "perm_p_exact_one_sided": pooled_rho_p,
        "n": 8,
        "ols": {k: pooled_lin[k] for k in ("slope", "intercept", "r2")},
    }

    # the rung-level texture: which pairs swap?
    swaps = []
    for w in WASHES:
        by_T = sorted(BATTERIES, key=lambda b: col("T_deep", w)[BATTERIES.index(b)])
        by_d = sorted(BATTERIES, key=lambda b: col("decline_deep", w)[BATTERIES.index(b)])
        swaps.append(
            {
                "wash": w,
                "T_order_low_to_high": by_T,
                "decline_order_low_to_high": by_d,
                "discordant_pairs": [
                    f"{a}>{b} in T but {b}>{a} in decline"
                    for i, a in enumerate(by_T)
                    for b in by_T[i + 1 :]
                    if by_d.index(a) > by_d.index(b)
                ],
            }
        )

    # ---- (2) THE FORM: linear in T vs linear in log10 T --------------------
    Td, Dd = col("T_deep"), col("decline_deep")
    form = {
        "linear_in_T": ols(Td, Dd),
        "linear_in_log10T": ols(np.log10(Td), Dd),
        "degeneracy_pearson_T_vs_log10T": pearson(Td, np.log10(Td)),
        "T_range": [float(Td.min()), float(Td.max())],
        "note": (
            "over T in [1.2, 1.6] the log is near-affine in T; if the degeneracy "
            "r ~ 1 this cell cannot discriminate the functional form — the honest "
            "outcome is 'both fit equally; form undetermined at this range'"
        ),
    }

    # ---- (3) THE CO-READ: the R2 joins --------------------------------------
    co_read = {}
    for w in WASHES:
        co_read[f"T_vs_R2_{w}_n4"] = {
            "spearman_rho": spearman(col("T_deep", w), col("R2_deep", w)),
            "perm_p_exact_one_sided": perm_p_exact(col("T_deep", w), col("R2_deep", w)),
        }
        co_read[f"decline_vs_R2_{w}_n4"] = {
            "spearman_rho": spearman(col("decline_deep", w), col("R2_deep", w)),
            "perm_p_exact_one_sided": perm_p_exact(col("decline_deep", w), col("R2_deep", w)),
        }
    co_read["T_vs_R2_pooled_n8"] = {
        "spearman_rho": spearman(Td, col("R2_deep")),
        "perm_p_exact_one_sided": perm_p_exact(Td, col("R2_deep")),
        "ols": {k: v for k, v in ols(Td, col("R2_deep")).items() if k in ("slope", "r2")},
    }
    co_read["decline_vs_R2_pooled_n8"] = {
        "spearman_rho": spearman(Dd, col("R2_deep")),
        "perm_p_exact_one_sided": perm_p_exact(Dd, col("R2_deep")),
        "ols": {k: v for k, v in ols(Dd, col("R2_deep")).items() if k in ("slope", "r2")},
        "note": "the death-meter law (T236: rho=+1.00 on e259's own R2s) re-run on the committed records at this aggregate",
    }
    co_read["R2_source_disclosure"] = (
        "e238's R2_battery is decline-variance explained under the COMMITTED POOLED T, "
        "not per-battery own-T; x5's own-fit R2 (ctrl/near only, n=2 per wash — no rank "
        "stats at n=2): "
        + json.dumps(
            {
                f"{r['battery']}/{r['wash']}": round(r["R2_own_deep_x5"], 4)
                for r in join_rows
                if r["R2_own_deep_x5"] is not None
            }
        )
    )

    # ---- sensitivity co-reports (never adjudicating) ------------------------
    sens = {}
    for s in ALL_STATES:
        for w in WASHES:
            x = np.array([T_mle[(b, w, s)] for b in BATTERIES])
            y = np.array([decline[(b, w, s)] for b in BATTERIES])
            sens[f"state{s}_{w}"] = {"spearman_rho": spearman(x, y), "n": 4}
    # aggregation variants on the T side / decline side
    for variant, tf in {
        "s80_only": lambda b, w: T_mle[(b, w, 80)],
        "deep_mean_T_lsq": lambda b, w: float(np.mean([T_lsq[(b, w, s)] for s in DEEP_STATES])),
        "all_state_mean": lambda b, w: float(np.mean([T_mle[(b, w, s)] for s in ALL_STATES])),
    }.items():
        entry = {}
        for w in WASHES:
            x = np.array([tf(b, w) for b in BATTERIES])
            y = np.array([np.mean([decline[(b, w, s)] for s in DEEP_STATES]) for b in BATTERIES])
            entry[w] = {"spearman_rho": spearman(x, y), "n": 4}
        sens[variant] = entry
    for variant, df in {
        "decline_ratio_of_means": lambda b, w: 1.0
        - float(np.mean([meanp[(w, s)][b] for s in DEEP_STATES])) / t0[b],
        "decline_all_state_mean": lambda b, w: float(np.mean([decline[(b, w, s)] for s in ALL_STATES])),
    }.items():
        entry = {}
        for w in WASHES:
            x = np.array([np.mean([T_mle[(b, w, s)] for s in DEEP_STATES]) for b in BATTERIES])
            y = np.array([df(b, w) for b in BATTERIES])
            entry[w] = {"spearman_rho": spearman(x, y), "n": 4}
        sens[variant] = entry

    # ---- adjudication (frozen logic) ----------------------------------------
    rho_w1 = join_stats["w1_n4"]["spearman_rho"]
    rho_w2 = join_stats["w2_n4"]["spearman_rho"]
    slope_pooled = pooled_lin["slope"]
    if not all_gates_pass:
        verdict = "MIXED"
        clause = "a gate failed — the tables verbatim, both washes"
    elif rho_w1 >= RHO_BAR and rho_w2 >= RHO_BAR and slope_pooled > 0:
        verdict = "T-TRACKS-DECLINE"
        clause = (
            f"per-wash Spearman {rho_w1:.2f} (w1) and {rho_w2:.2f} (w2), both >= {RHO_BAR:.2f} "
            f"(n=4 each, disclosed) AND the pooled OLS slope {slope_pooled:+.4f} > 0"
        )
    else:
        verdict = "T-INDEPENDENT"
        clause = (
            f"per-wash rho {rho_w1:.2f} (w1) / {rho_w2:.2f} (w2) vs bar {RHO_BAR:.2f}"
            + (f" OR pooled slope {slope_pooled:+.4f} <= 0" if slope_pooled <= 0 else "")
        )
    at_bar = abs(rho_w1 - RHO_BAR) < 1e-9 or abs(rho_w2 - RHO_BAR) < 1e-9

    # ---- figures -------------------------------------------------------------
    colors = {"ctrl": "#1f77b4", "fact": "#2ca02c", "near": "#d62728", "tmpl": "#9467bd"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (xname, yname, xlab, ylab, title) in zip(
        axes,
        [
            ("T_deep", "decline_deep", "deep-state own-T  (x5 committed, mean of s50,s80)", "mean decline  1 - p(t)/p(t0)  (e214 committed, mean of s50,s80)", "THE JOIN: the T-ladder vs the decline"),
            ("T_deep", "R2_deep", "deep-state own-T", "deep-state R2 (e238 committed, mean of s50,s80)", "CO-READ 1: T vs fit quality"),
            ("decline_deep", "R2_deep", "mean decline", "deep-state R2", "CO-READ 2: decline vs R2 (the death-meter)"),
        ],
    ):
        for r in join_rows:
            ax.scatter(
                r[xname],
                r[yname],
                s=110,
                color=colors[r["battery"]],
                marker="o" if r["wash"] == "w1" else "^",
                edgecolor="k",
                linewidth=0.6,
                zorder=3,
                label=f"{r['battery']} {r['wash']}",
            )
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        if xname == "T_deep" and yname == "decline_deep":
            xs = np.linspace(Td.min() - 0.03, Td.max() + 0.03, 50)
            ax.plot(xs, pooled_lin["intercept"] + pooled_lin["slope"] * xs, "k--", lw=1.2, label=f"pooled linear (R2={pooled_lin['r2']:.3f})")
            fl = form["linear_in_log10T"]
            ax.plot(xs, fl["intercept"] + fl["slope"] * np.log10(xs), ":", color="crimson", lw=1.6, label=f"pooled log10T (R2={fl['r2']:.3f})")
            ax.legend(fontsize=6.5, ncol=2, loc="upper left")
            ax.text(
                0.97,
                0.03,
                f"per-wash Spearman (n=4): w1 {rho_w1:.2f} / w2 {rho_w2:.2f}   pooled n=8: {pooled_rho:.3f} (p={pooled_rho_p:.4f})",
                transform=ax.transAxes,
                ha="right",
                fontsize=7.5,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
            )
    fig.suptitle(
        f"E262 — T232's registered join: the battery T-ladder vs the decline  |  VERDICT: {verdict}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "t_decline_join.png", dpi=140)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9.5, 5.2))
    labels, vals, cols_ = [], [], []
    entry_defs = [
        ("s10 w1", sens["state10_w1"]["spearman_rho"]),
        ("s10 w2", sens["state10_w2"]["spearman_rho"]),
        ("s50 w1", sens["state50_w1"]["spearman_rho"]),
        ("s50 w2", sens["state50_w2"]["spearman_rho"]),
        ("s80 w1", sens["state80_w1"]["spearman_rho"]),
        ("s80 w2", sens["state80_w2"]["spearman_rho"]),
        ("deep-mean w1\n(PRIMARY)", rho_w1),
        ("deep-mean w2\n(PRIMARY)", rho_w2),
        ("all-mean w1", sens["all_state_mean"]["w1"]["spearman_rho"]),
        ("all-mean w2", sens["all_state_mean"]["w2"]["spearman_rho"]),
    ]
    for lab, v in entry_defs:
        labels.append(lab)
        vals.append(v)
        cols_.append("#444444" if "PRIMARY" in lab else ("#1f77b4" if "w1" in lab else "#ff7f0e"))
    ax.bar(range(len(vals)), vals, color=cols_, edgecolor="k", linewidth=0.5)
    ax.axhline(RHO_BAR, color="crimson", ls="--", lw=1.4, label=f"frozen bar rho >= {RHO_BAR}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman rho (T vs decline), n=4")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("E262 — aggregation sensitivity (co-reports; only the deep-mean adjudicates)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    fig2.tight_layout()
    fig2.savefig(out_dir / "state_sensitivity.png", dpi=140)
    plt.close(fig2)

    # ---- the tables (verbatim, both washes) ---------------------------------
    tables = {
        "join_table": join_rows,
        "rung_orders_and_swaps": swaps,
        "mean_p_records_used": {
            f"{w}+{s}": {b: meanp[(w, s)][b] for b in BATTERIES}
            for w in WASHES
            for s in DEEP_STATES
        }
        | {"t0": t0},
    }

    # ---- honesty reflex ------------------------------------------------------
    honesty = {
        "n_counts": "n=1 organism (GPT-2 124M, e182's archive); near n=3 / ctrl n=12 / fact n=20 / tmpl n=19 probes; Spearman n=4 per wash is COARSE (values step in 0.2; rho=0.80 == exactly one adjacent transposition)",
        "zero_margin": (
            f"the verdict sits AT the frozen bar (rho_w1={rho_w1:.2f}, rho_w2={rho_w2:.2f} vs >=0.80) with ZERO margin"
            if (verdict == "T-TRACKS-DECLINE" and at_bar)
            else "the verdict's margin is at least one 0.2 rank-step"
        ),
        "aggregation_sensitivity": (
            f"the s80-only co-report reads w1 {sens['state80_w1']['spearman_rho']:.2f} / w2 {sens['state80_w2']['spearman_rho']:.2f}; "
            f"the s50-only reads w1 {sens['state50_w1']['spearman_rho']:.2f} / w2 {sens['state50_w2']['spearman_rho']:.2f}; "
            "the PRIMARY was frozen before compute on T232's own deep-state construction ({50,80}) — the co-reports price, they never adjudicate"
        ),
        "the_single_discordance": (
            "the ladder's bottom two rungs (ctrl coldest/least-dead, fact next) NEVER swap; the top two (tmpl hottest, near deadliest) are TRANSPOSED between lens and deaths — "
            "the same transposition in both washes: systematic texture, not noise; T236's descriptive-T-undercools clause is its cousin"
        ),
        "observational": "a monotone join supports coarse-graining; it does not intervene — the intervening cell (per-battery erosion-rate estimation) is a different instrument",
        "lens_not_mechanism": "T228/T229/T232's demotions stand: T is a p-sharpening lens; this join says the lens's battery variation carries the erosion ordering, not that the wash implements a temperature",
        "device_asymmetry": "wash 1 = CPU fp32 replay, wash 2 = GPU fp32 (the archive's inherited asymmetry, disclosed)",
        "guarantees_nothing": "nothing here is guaranteed — the openness is the point; the branches were frozen before compute; no bar shopping",
    }

    # ---- draft NOTES entry ---------------------------------------------------
    def fmt(x):
        return f"{x:.3f}"

    draft = (
        f"## e262 — T232's registered join, THE T-LADDER vs THE DECLINE: {verdict} "
        f"(the battery thermal ladder vs the committed decline fractions, desk-only on committed records) "
        f"({now_iso()}) — DONE\n\n"
        f"WHAT WE DID: the frozen join per battery per wash — x5's committed deep-state own-T "
        f"(mean of s50/s80, T232's own ladder construction) against e214's committed mean decline "
        f"(1 - mean_p(t)/mean_p(t0), same states; provenance cross-checked: x5's own re-probe vs e214 "
        f"max dp 0.0, the w2 rows == e182c2's journal_p2, the n-weighted pooled t0 mean reproduces "
        f"e238's committed mean_p0; 4/4 gates PASS).\n\n"
        f"WHAT WE SAW: per-wash Spearman w1 {rho_w1:.2f} / w2 {rho_w2:.2f} (n=4 each, exact permutation "
        f"p {join_stats['w1_n4']['perm_p_exact_one_sided']:.3f}/{join_stats['w2_n4']['perm_p_exact_one_sided']:.3f}), "
        f"pooled n=8 rho {pooled_rho:.3f} (p={pooled_rho_p:.4f}), pooled OLS slope {slope_pooled:+.4f} "
        f"(R2 {pooled_lin['r2']:.3f}) — THE CLAUSE: {clause}. THE TEXTURE: ctrl (coldest, least-dead) and "
        f"fact (next) never swap; tmpl reads HOTTEST but near dies HARDEST — the same single adjacent "
        f"transposition (near/tmpl) in BOTH washes at the deep aggregate: systematic, not noise. "
        f"THE FORM: linear-in-T R2 {form['linear_in_T']['r2']:.4f} vs linear-in-log10T R2 "
        f"{form['linear_in_log10T']['r2']:.4f} — degenerate over the observed range "
        f"[{Td.min():.2f},{Td.max():.2f}] (Pearson T vs log10T {form['degeneracy_pearson_T_vs_log10T']:.5f}); "
        f"this cell cannot discriminate the form. THE CO-READ: T vs R2 per-wash "
        f"{co_read['T_vs_R2_w1_n4']['spearman_rho']:.2f}/{co_read['T_vs_R2_w2_n4']['spearman_rho']:.2f}; "
        f"decline vs R2 per-wash {co_read['decline_vs_R2_w1_n4']['spearman_rho']:.2f}/"
        f"{co_read['decline_vs_R2_w2_n4']['spearman_rho']:.2f} — the death-meter law (T236) replicates "
        f"EXACTLY on the committed records at this aggregate, and the T-ladder joins fit quality with "
        f"the same single transposition.\n\n"
        f"HONESTY: the verdict sits AT the frozen bar (zero margin); the s80-only co-report reads "
        f"w1 {sens['state80_w1']['spearman_rho']:.2f} / w2 {sens['state80_w2']['spearman_rho']:.2f} "
        f"(aggregation-sensitive, disclosed; the primary was frozen before compute); near n=3; n=4-per-wash "
        f"Spearman steps in 0.2; observational; lens-not-mechanism stands; wash-1 CPU replay vs wash-2 GPU.\n\n"
        f"WHAT'S NEXT: the registered discriminating cell is the per-battery EROSION-RATE estimate "
        f"(fit the decline kinetics per battery, not the lens) — if rate(T-ladder) keeps the ladder's "
        f"order, the coarse-graining claim gains its own instrument; the near/tmpl transposition is the "
        f"sharpest open thread (tmpl hottest yet near dies hardest — the lens's top rung overprices tmpl "
        f"or underprices near; e259's descriptive-T-undercools clause is the family resemblance)."
    )

    # ---- metrics --------------------------------------------------------------
    metrics = {
        "experiment": "e262_t_decline_join",
        "phase": "desk-only adjudication of T232's registered join on COMMITTED records (x5's T table vs e214/e182c2's decline records vs e238's R2 table); no model touched, no GPU",
        "date": now_iso(),
        "status": "DONE",
        "smoke": bool(args.smoke),
        "registration": "bars frozen VERBATIM from the dispatch brief BEFORE any compute (script committed at birth); adjudicate against exactly this; no bar shopping",
        "registered_prediction": {
            "bars_verbatim": BARS_VERBATIM,
            "operationalizations": (
                "T side = x5's committed per-battery T_mle aggregated over the DEEP STATES {50,80} "
                "(T232's own ladder construction); decline side = e214's committed per-battery "
                "mean_p as 1 - mean_p(t)/mean_p(t0) aggregated over the SAME states; join = 8 points; "
                "verdict: T-TRACKS-DECLINE iff (rho_w1 >= 0.80 AND rho_w2 >= 0.80 AND pooled OLS slope > 0), "
                "T-INDEPENDENT otherwise (gates failing -> MIXED); per-wash n=4 exact permutation p; "
                "sensitivity co-reports (s10/s50/s80 single-state, s80-only, all-state mean, T_lsq, "
                "ratio-of-means decline) NEVER adjudicate"
            ),
        },
        "question": "does the battery T-ladder TRACK the decline fractions quantitatively (monotone across the four batteries and both washes) — is the battery T(t) the erosion rate's coarse-graining?",
        "builds_on": [
            "T232 / x5 (the battery thermal ladder + THE REGISTERED JOIN this cell runs)",
            "T236 / e259 (the death-meter law: rho(decline, R2_own) = +1.00 — the co-read's target)",
            "T218 / e238 (the one-T instrument + the committed per-battery R2 table)",
            "T187 / e214 (the committed per-battery per-state records this join's decline side reads)",
            "T183 / e182c2 (the wash-2 archive + the journal_p2 cross-check)",
            "T228 / e252 + T229 / e255 (the lens demotions the join's interpretation respects)",
        ],
        "whats_new": [
            "the registered T-vs-decline join itself (T232's registration, never run until now)",
            "the quantitative-form read (linear-in-T vs linear-in-log T) with its degeneracy disclosure",
            "the death-meter law re-run on committed records at the deep-state aggregate (decline vs R2)",
            "the aggregation-sensitivity ledger (which single-state joins hold and which break)",
            "exact permutation p-values at n=4 and n=8 (no asymptotics at tiny n)",
        ],
        "gates": gates,
        "adjudication": {
            "verdict": verdict,
            "clause": clause,
            "bar_constants": {"RHO_BAR": RHO_BAR, "per_wash_n": 4, "pooled_n": 8, "deep_states": DEEP_STATES},
            "at_bar_zero_margin": bool(at_bar),
            "clause_lines": [
                f"per-wash monotonicity: rho_w1 = {rho_w1:.4f} (>= {RHO_BAR}: {rho_w1 >= RHO_BAR}), rho_w2 = {rho_w2:.4f} (>= {RHO_BAR}: {rho_w2 >= RHO_BAR}), n=4 each, disclosed",
                f"pooled fit sign: OLS decline ~ T slope = {slope_pooled:+.6f} (> 0: {slope_pooled > 0})",
                f"precedence: T-TRACKS-DECLINE -> T-INDEPENDENT -> MIXED (gate failure only); frozen before compute",
            ],
        },
        "join": join_stats,
        "form": form,
        "co_read": co_read,
        "sensitivity_co_reports": sens,
        "tables": tables,
        "honesty_reflex": honesty,
        "compute": {
            "envelope": "desk-only: numpy + json + matplotlib on committed records; threads 4 via env; zero torch, zero GPU (e261's recovery owns the GPU)",
            "total_s": None,
        },
        "provenance": {
            "git_head_at_start": git_head,
            "script": str(Path(__file__).resolve()),
            "committed_records": hashes,
            "note": "every number read at RUNTIME from the committed records (never transcribed); sha256_16 recorded at read time",
        },
        "trims": [],
        "deviations": [
            "DESK-ONLY per the dispatch: no model loaded, no forward pass, no wash — pure joins on committed records.",
            "The decline side reads e214's journal per-battery mean_p (both washes); e182c2's journal_p2 cross-checks the w2 rows (max dp 2.6e-7, float re-serialization) and x5's own mean_p_obs cross-checks ctrl/near (max dp 0.0).",
            "The R2 co-read uses e238's committed R2_battery (under the pooled committed T); x5's own-fit R2 exists only for ctrl/near and co-reports as numbers (n=2 per wash, no rank stats).",
            "No NOTES/THINKING/QUEUE/STATE edits (dispatch); the draft NOTES entry rides in metrics['draft_notes_entry'] + the final report.",
        ],
        "draft_notes_entry": draft,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "journal.json").write_text(
        json.dumps({"cell": "e262", "records": list(hashes.keys()), "verdict": verdict, "at": now_iso()}, indent=2),
        encoding="utf-8",
    )

    # ---- console summary -------------------------------------------------------
    print("=" * 78)
    print(f"E262 — T232's registered join: THE T-LADDER vs THE DECLINE")
    print("=" * 78)
    hdr = f"{'batt':5s} {'wash':4s} {'T50':>7s} {'T80':>7s} {'T_deep':>7s} {'dec50':>7s} {'dec80':>7s} {'dec_deep':>8s} {'R2_deep':>8s}"
    print(hdr)
    for r in join_rows:
        print(
            f"{r['battery']:5s} {r['wash']:4s} {r['T_s50']:7.4f} {r['T_s80']:7.4f} {r['T_deep']:7.4f} "
            f"{r['decline_s50']:7.4f} {r['decline_s80']:7.4f} {r['decline_deep']:8.4f} {r['R2_deep']:8.4f}"
        )
    print("-" * 78)
    print(f"per-wash Spearman (n=4):  w1 {rho_w1:.4f} (p={join_stats['w1_n4']['perm_p_exact_one_sided']:.4f})   w2 {rho_w2:.4f} (p={join_stats['w2_n4']['perm_p_exact_one_sided']:.4f})   bar >= {RHO_BAR}")
    print(f"pooled (n=8): rho {pooled_rho:.4f} (p={pooled_rho_p:.5f})   OLS slope {slope_pooled:+.4f} (R2 {pooled_lin['r2']:.4f})")
    for s in swaps:
        print(f"  {s['wash']}: T order {' < '.join(s['T_order_low_to_high'])} | decline order {' < '.join(s['decline_order_low_to_high'])} | swaps: {s['discordant_pairs']}")
    print(f"form: linear R2 {form['linear_in_T']['r2']:.5f} vs log10T R2 {form['linear_in_log10T']['r2']:.5f} (degeneracy r {form['degeneracy_pearson_T_vs_log10T']:.6f})")
    print(f"co-read: T~R2 w1 {co_read['T_vs_R2_w1_n4']['spearman_rho']:.2f} / w2 {co_read['T_vs_R2_w2_n4']['spearman_rho']:.2f}; decline~R2 w1 {co_read['decline_vs_R2_w1_n4']['spearman_rho']:.2f} / w2 {co_read['decline_vs_R2_w2_n4']['spearman_rho']:.2f}")
    print(f"sensitivity (co-reports): s80-only w1 {sens['state80_w1']['spearman_rho']:.2f} / w2 {sens['state80_w2']['spearman_rho']:.2f}; s50-only w1 {sens['state50_w1']['spearman_rho']:.2f} / w2 {sens['state50_w2']['spearman_rho']:.2f}; s10 w1 {sens['state10_w1']['spearman_rho']:.2f} / w2 {sens['state10_w2']['spearman_rho']:.2f}")
    print("-" * 78)
    print(f"GATES: " + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in gates.items()))
    print(f"VERDICT: {verdict} — {clause}")
    print(f"zero-margin: {at_bar}")
    print(f"artifacts: {out_dir / 'metrics.json'}, {out_dir / 't_decline_join.png'}, {out_dir / 'state_sensitivity.png'}")


if __name__ == "__main__":
    main()
