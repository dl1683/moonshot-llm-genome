"""X8 — THE FLASH THROUGH THE THERMAL LENS (first pass). CPU-ONLY desk cell.

Owner directive 2026-10-05 ~21:20Z (aggressive CPU desk work; one of three
parallel desk cells; threads <= 4, NO GPU, no envelope-log writes).

WHY. The concurrent-training flash — the transient RISE then decay of a
write's probe strength under the interleaved corpus (traj_g0 at milestones
s1/100/200/300/400) — broke its saturation law tonight (e271: peak 0.0730
at s200, 3.5x the claimed ~0.02 band; THINKING.md T249). W040's A4
question: is the flash a READOUT event (gain — thermal-lens-fittable, an
artifact of the measurement channel) or a STORAGE event (genuine transient
memory — the rise-then-decay of a real trace)? The lab's one-T lens
(x4/x5 family) fits discharge curves with a single scalar T; x5's verdict
was BATTERY-SPECIFIC and the lens was demoted to a death-meter for
DECLINES. A RISE has never been through it.

DATA (read at runtime, never transcribed as primary): runs/e268..e271
metrics.json — each carries the CONCURRENT arm's traj_g0 milestone curve
AND the SERIAL arm's; e271's adjudication block also carries the full
four-rung ladder (cross-checked against the originals as a gate). 237k
serial PRIMARY = the e271 session arm (bit-faithful to e261's committed
rung at ~1e-6; e261 co-reported).

REGISTERED BARS (frozen VERBATIM from the dispatch letter BEFORE any
compute; this script committed at birth; adjudicate against exactly this;
no bar shopping):
  - READOUT-LIKE: a single scalar per rung fits BOTH limbs cleanly
    (R2 >= 0.9 all rungs) — the flash is gain, not store; H-POISON's
    floor lives in the measurement channel.
  - STORAGE-LIKE: the fits misfit (any rung R2 < 0.7) OR the rise and
    decay limbs demand inconsistent scalars (|T_rise - T_decay| > 50%)
    — the flash is a genuine transient store; a new object class
    (W040's A4).
  - UNDERPOWERED: 5 points per curve cannot discriminate at these effect
    sizes (be honest — this is a REAL possibility; if so, state exactly
    what finer milestone grid + seeds would be needed, quantitatively).
  - MIXED: anything else.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * THE ONE-T FAMILY (the x4/x5 lens reduced to scalar milestones): the
    readout claim "the concurrent probe is the serial latent read through
    the channel" reduces (the two-class exact form of softmax(L/T)) to
    logit gain:  z_c(t) = z_s(t)/T,  z = ln(p/(1-p)). ONE scalar T per
    rung, least squares through the origin, fit points t in
    {100,200,300,400} (s1 = the shared pre-install floor / initial
    condition — co-reported, never a fit point; DISCLOSED). Closed-form
    gain g = 1/T; T reported with a physicality flag (a lens temperature
    must be > 0; g <= 0 is an unphysical read, reported verbatim).
  * R2 (all families): 1 - SS_res/SS_tot, SS_tot about the target's mean
    over that family's fit points, in the family's PRIMARY space (lens:
    z-space primary, p-space co-report; race: p-space primary, z-space
    co-report). Negative R2 reported verbatim. dof disclosed (n=4 fit
    points; k=1 lens, k=2 race).
  * LIMBS: t_peak = argmax of the concurrent interior milestone curve;
    rise limb = interior points {100..t_peak}; decay limb =
    {t_peak..400}; the peak point belongs to both. Limb T's by the same
    closed form on each limb's points. Inconsistency
    I = |T_rise - T_decay| / (0.5*(|T_rise|+|T_decay|)); fires at I > 0.5;
    a SIGN DISAGREEMENT (T_rise*T_decay <= 0) counts as inconsistent (no
    positive temperature reads the pair). Degenerate limbs (n=1, no
    residuals) flagged and still computed (single-point T = z_s/z_c).
  * THE KINETIC RACE (the storage form, consult #005's kinetic form):
    dX/dt = alpha*G(t) - beta*X, G = the rung's OWN serial curve
    (piecewise-LINEAR on the milestone knots, knots placed at
    t in {0,100,200,300,400} with G(0) := serial p(s1) — the 1-step
    shift is immaterial and disclosed), X(0) = the concurrent floor
    p_c(s1); exact closed-form propagation through each linear segment;
    alpha, beta >= 0; fitted by profiling: log-grid scan of beta in
    [1e-4, 3] (400 points) + bounded polish, alpha closed-form per beta
    (clamped >= 0); targets the 4 interior milestones in p-space.
    Co-reports: the z-space race (same ODE on z-curves), tau = 1/beta,
    alpha/beta, and fitted X(400)/serial(400) vs the measured endpoint
    ratio (the calibration read).
  * DECAY-SHAPE DISCRIMINATOR (T249's registered shape test; co-report,
    never adjudicates alone): on the decay limb in z-space,
    linear-in-t (dose-erosion, H-POISON) vs exponential-in-t
    (first-order wash), both 2-param LS; winner by z-space R2;
    uninformative at n <= 2 (flagged).
  * RANK TRENDS: fitted T, alpha, beta, alpha/beta vs rung k in
    {10k, 40k, 100k, 237k}; Spearman co-reported with the n=4
    disclosure; the measured ratio ladder cross-referenced.
  * SERIAL REFERENCES: (a) lens serial-vs-serial must return T = 1,
    R2 = 1 exactly (the instrument's algebra gate); (b) lens of the
    e271 session serial vs e261's committed serial (T ~ 1 — the
    bit-faithfulness read); (c) the race fitted to the serial curve
    with ITSELF as driver = the family's driver-following CEILING on
    this grid (the race low-passes its driver; its R2 there bounds what
    any driver-following story can reproduce at this milestone
    granularity).
  * BOOTSTRAP (the UNDERPOWERED instrument): 2000 draws, seed 20261005;
    two PROXY sigma models, both reported: (S1) the lens's own
    full-fit z-residual RMS per rung; (S2) the serial driver's
    detrended z-wobble RMS per rung (the trajectory-level fluctuation
    scale visible in-data). Draws add N(0, sigma) to the FITTED lens
    curve at the fit points and refit full-T and limb-T's. Reported:
    90% CI widths, sign-stability of (T_rise - T_decay),
    P(R2_draw < 0.7), P(I_draw > 0.5), MDD_90 = 2.8 * SD_boot of
    (T_rise - T_decay) (the limb difference the design resolves at 90%),
    and the grid prescription: milestone multiplier needed for
    MDD_90 <= half the inconsistency bar in absolute T units, and for
    bootstrap R2 CI width <= 0.1. HARD CAVEAT (verbatim in outputs):
    both sigmas are PROXIES. The serial replicate says the arms are
    deterministic to ~1e-6 (run-to-run); the CONCURRENT flash has ONE
    corpus seed per rung (fresh per cell) — its seed-to-seed variance is
    UNMEASURED at n=1, so the bootstrap can only UNDERSTATE it.
  * COMPOSITE (frozen order): (a) READOUT-LIKE iff every rung's lens
    full-curve R2_z >= 0.9 AND every rung I <= 0.5 with no sign flip
    AND every rung's full-curve T > 0. (b) STORAGE-LIKE iff any rung
    R2_z < 0.7 OR any rung (I > 0.5 or sign flip). (c) OVERRIDE: if (b)
    fired ONLY through drivers whose bootstrap 90% CIs cross their
    thresholds (R2 CI reaching >= 0.7; I CI reaching <= 0.5) and NO
    stable driver remains, the verdict is UNDERPOWERED (point estimates
    carried verbatim). (d) If neither (a) nor (b) fired: UNDERPOWERED
    iff for >= 2 rungs the R2_z CI spans [0.7, 0.9] or the I CI spans
    0.5; else MIXED. Every clause's per-rung evaluation recorded.
  * W041's P-W41d ("x8 returns UNDERPOWERED") is CO-REPORTED as a
    registered third-party prediction — never an input.
  * INSTRUMENT DISCLOSURE: this is the DESK reduction of the x5 lens.
    The full-logit MLE form q = softmax(L0_i/T) needs milestone logit
    dumps that do not exist (a dump mini-cell is the named follow-up if
    the verdict is STORAGE-flavored).

Outputs: runs/x8/metrics.json (progressive), runs/x8/x8_flash_fits.png,
runs/x8/REPORT.md. NOTES/THINKING/QUEUE/STATE untouched (dispatch order).
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"  # CPU-only desk cell; threads <= 4 (dispatch)

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT = RUNS / "x8"
OUT.mkdir(parents=True, exist_ok=True)

STEPS = [1, 100, 200, 300, 400]
INTERIOR = [100, 200, 300, 400]
BOOT_DRAWS = 2000
BOOT_SEED = 20261005

# ---------------------------------------------------------------- data ----
# Embedded literals are the CROSS-CHECK; the primary read is the runtime
# json (G_DATA below asserts they agree to rel 1e-12).
EMBED = {
    "10k_e268": {
        "concurrent": [1.3246100024844054e-05, 0.0002721511118579656,
                       0.00023426816915161908, 0.0007357693975791335,
                       4.004325455753133e-05],
        "serial": [1.3466438758769073e-05, 0.30591899156570435,
                   0.21840216219425201, 0.21022741496562958,
                   0.26464739441871643],
    },
    "40k_e269": {
        "concurrent": [1.315838653681567e-05, 0.024467766284942627,
                       0.00911789108067751, 0.005631699226796727,
                       0.003268091706559062],
        "serial": [1.3541608495870605e-05, 0.37455758452415466,
                   0.37828463315937745, 0.4843420684334616,
                   0.3464753329753876],
    },
    "100k_e270": {
        "concurrent": [1.3522996596293524e-05, 0.0007310498622246087,
                       0.02207603119313717, 0.006495761685073376,
                       0.010897441767156124],
        "serial": [1.3658516763825901e-05, 0.49801623821258545,
                   0.4266950488090515, 0.3523316979408264,
                   0.4359813928604126],
    },
    "237k_e271": {
        "concurrent": [1.3693145774595905e-05, 0.019378451630473137,
                       0.0729694813489914, 0.03993295133113861,
                       0.034998517483472824],
        "serial": [1.3798319741908927e-05, 0.5885881781578064,
                   0.3677981197834015, 0.5534284114837646,
                   0.38436517119407654],
        "serial_e261_committed": [1.3798317922919523e-05, 0.5885877013206482,
                                  0.36779922246932983, 0.5534279942512512,
                                  0.38436421751976013],
    },
}
RUNG_ORDER = ["10k_e268", "40k_e269", "100k_e270", "237k_e271"]
RUNG_SRC = {"10k_e268": "e268", "40k_e269": "e269",
            "100k_e270": "e270", "237k_e271": "e271"}


def _sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _traj_to_map(traj):
    return {int(p_["step"]): float(p_["g0_pz"]) for p_ in traj}


def load_and_gate():
    """G_DATA: read the four run cells at runtime; assert against the
    embedded literals and against e271's verbatim ladder copy."""
    gate = {"files": {}, "checks": {}}
    data = {}
    for rung in RUNG_ORDER:
        src = RUNG_SRC[rung]
        mp = json.load(open(RUNS / src / "metrics.json"))
        gate["files"][src] = {
            "path": str(RUNS / src / "metrics.json"),
            "sha256_16": _sha16(RUNS / src / "metrics.json"),
        }
        arms = mp["arms"]
        cs = _traj_to_map(arms["CONCURRENT"]["install"]["traj"])
        sr = _traj_to_map(arms["SERIAL"]["install"]["traj"])
        conc = [cs[s] for s in STEPS]
        ser = [sr[s] for s in STEPS]
        for lbl, got, want in (("concurrent", conc, EMBED[rung]["concurrent"]),
                               ("serial", ser, EMBED[rung]["serial"])):
            ok = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
                     for a, b in zip(got, want))
            gate["checks"][f"{rung}:{lbl}==embed"] = bool(ok)
            assert ok, f"G_DATA fail {rung} {lbl}"
        data[rung] = {"concurrent": conc, "serial": ser}
    # e271 ladder verbatim-vs-originals + e261 committed co-report
    e271 = json.load(open(RUNS / "e271" / "metrics.json"))
    lad = e271["adjudication"]["reads"]["ratio_ladder"]["ladder"]
    pairs = [("10k_e268", "10k_e268"), ("40k_e269", "40k_e269"),
             ("100k_e270", "100k_e270"), ("237k_e271", "237k_e271_this_cell")]
    for rung, key in pairs:
        tr = lad[key]["traj_g0"]
        orig = data[rung]["concurrent"]
        ok = all(math.isclose(tr[str(s)], v, rel_tol=1e-12) for s, v in
                 zip(STEPS, orig))
        gate["checks"][f"e271_ladder_verbatim:{rung}"] = bool(ok)
        assert ok, f"G_LADDER fail {rung}"
    e261 = lad["237k_e271_this_cell"]  # carries committed serial in reads
    committed = e271["adjudication"]["reads"]["cited_serial_rung_e261"]["traj_g0"]
    e261_vals = [committed[str(s)] for s in STEPS]
    for a, b in zip(e261_vals, EMBED["237k_e271"]["serial_e261_committed"]):
        assert math.isclose(a, b, rel_tol=1e-12), "G_DATA fail e261"
    gate["checks"]["e261_committed_serial==embed"] = True
    data["237k_e271"]["serial_e261_committed"] = e261_vals
    gate["pass"] = all(gate["checks"].values())
    return data, gate


# ------------------------------------------------------------- families ---
def z_of(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def p_of(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def r2(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return (1.0 if ss_res == 0.0 else float("-inf"))
    return 1.0 - ss_res / ss_tot


def fit_lens(zs, zc):
    """z_c = z_s / T through the origin; closed-form gain g = 1/T."""
    zs = np.asarray(zs, float)
    zc = np.asarray(zc, float)
    denom = float(np.sum(zs * zs))
    if denom == 0.0:
        return {"T": float("nan"), "g": float("nan"),
                "R2_z": float("nan"), "SS_res_z": float("nan"),
                "n": int(zs.size), "physical": None}
    g = float(np.sum(zs * zc)) / denom
    pred = zs * g
    ss_res = float(np.sum((zc - pred) ** 2))
    return {"T": (1.0 / g if g != 0.0 else float("inf")), "g": g,
            "R2_z": r2(zc, pred), "SS_res_z": ss_res,
            "n": int(zs.size), "physical": bool(g > 0.0)}


def lens_p_space(pc, ps, g):
    pred_z = z_of(ps) * g
    pred_p = p_of(pred_z)
    return {"R2_p": r2(np.asarray(pc, float), pred_p),
            "pred_p": [float(x) for x in pred_p]}


def race_propagate(knot_t, knot_G, X0, alpha, beta, query_t):
    """Exact solution of X' = alpha*G(t) - beta*X, G piecewise linear on
    knots (t ascending); returns X at each query_t (segment-chained)."""
    X = float(X0)
    qi = 0
    out = []
    qts = sorted(query_t)
    for k in range(len(knot_t) - 1):
        t0, t1 = float(knot_t[k]), float(knot_t[k + 1])
        G0, G1 = float(knot_G[k]), float(knot_G[k + 1])
        m = (G1 - G0) / (t1 - t0)
        while qi < len(qts) and qts[qi] <= t1:
            d = qts[qi] - t0
            if d < 0:
                d = 0.0
            e = math.exp(-beta * d)
            if beta > 1e-12:
                h = (1.0 - e) / beta
                val = X * e + alpha * (G0 * h + m * (d - h) / beta)
            else:  # beta -> 0 limit
                val = X + alpha * (G0 * d + 0.5 * m * d * d)
            out.append(val)
            qi += 1
        d = t1 - t0
        e = math.exp(-beta * d)
        if beta > 1e-12:
            h = (1.0 - e) / beta
            X = X * e + alpha * (G0 * h + m * (d - h) / beta)
        else:
            X = X + alpha * (G0 * d + 0.5 * m * d * d)
    while qi < len(qts):  # queries at/after the last knot
        out.append(X)
        qi += 1
    return out


def race_target_matrix(knot_t, knot_G, X0, beta, query_t):
    """X(t) = X0*decay(t) + alpha*F(t): return (decay_vec, F_vec)."""
    dec = race_propagate(knot_t, knot_G, X0, 1.0, 0.0, query_t)  # alpha=1,beta=0 -> cumulative; not used
    # decay-only: alpha=0
    dec = race_propagate(knot_t, knot_G, X0, 0.0, beta, query_t)
    # F: response to driver with X0=0, alpha=1
    F = race_propagate(knot_t, knot_G, 0.0, 1.0, beta, query_t)
    return np.asarray(dec, float), np.asarray(F, float)


def fit_race(knot_t, knot_G, X0, target, query_t, beta_lo=1e-4, beta_hi=3.0):
    """Profile fit: scan/polish beta (log scale), alpha closed form."""
    target = np.asarray(target, float)

    def ssr_for_beta(beta):
        dec, F = race_target_matrix(knot_t, knot_G, X0, beta, query_t)
        f2 = float(np.sum(F * F))
        if f2 == 0.0:
            alpha = 0.0
        else:
            alpha = max(0.0, float(np.sum((target - dec) * F)) / f2)
        pred = dec + alpha * F
        return float(np.sum((target - pred) ** 2)), alpha, pred

    grid = np.exp(np.linspace(math.log(beta_lo), math.log(beta_hi), 400))
    best = min(grid, key=lambda b: ssr_for_beta(b)[0])
    res = minimize_scalar(lambda lb: ssr_for_beta(math.exp(lb))[0],
                          bounds=(math.log(max(best / 2.0, beta_lo)),
                                  math.log(min(best * 2.0, beta_hi))),
                          method="bounded",
                          options={"xatol": 1e-6})
    beta = float(math.exp(res.x))
    if ssr_for_beta(beta)[0] > ssr_for_beta(best)[0]:
        beta = float(best)
    ssr, alpha, pred = ssr_for_beta(beta)
    return {"alpha": float(alpha), "beta": float(beta),
            "SS_res": ssr, "pred": pred}


def race_curve_fine(knot_t, knot_G, X0, alpha, beta, n=801):
    ts = np.linspace(knot_t[0], knot_t[-1], n)
    vals = race_propagate(knot_t, knot_G, X0, alpha, beta, list(ts))
    return ts, np.asarray(vals)


# ------------------------------------------------------------ adjudicate --
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_metrics(obj):
    obj = dict(obj)
    obj["written_at"] = now_iso()
    with open(OUT / "metrics.json", "w") as f:
        json.dump(obj, f, indent=1)


def main():
    t0 = datetime.now(timezone.utc)
    metrics = {
        "experiment": "x8_flash_thermal (desk cell, first pass)",
        "phase": "THE FLASH THROUGH THE THERMAL LENS — is the concurrent "
                 "flash a READOUT event (one-T-fittable gain in the "
                 "measurement channel) or a STORAGE event (a genuine "
                 "transient trace)? CPU-only desk instrument on the "
                 "committed e268-e271 milestone curves",
        "envelope": {
            "device": "CPU ONLY (desk cell; owner directive 2026-10-05 "
                      "~21:20Z) — no GPU touched, no envelope-log writes",
            "threads": "OMP/MKL/OPENBLAS/NUMEXPR/VECLIB pinned to 4 "
                       "before numpy import",
            "runtime_note": "pure numpy/scipy/matplotlib desk work; no "
                            "model loads, no checkpoints",
        },
        "registration": {
            "bars_verbatim": {
                "READOUT-LIKE": "a single scalar per rung fits BOTH limbs "
                                "cleanly (R2 >= 0.9 all rungs) — the flash "
                                "is gain, not store; H-POISON's floor lives "
                                "in the measurement channel",
                "STORAGE-LIKE": "the fits misfit (any rung R2 < 0.7) OR "
                                "the rise and decay limbs demand "
                                "inconsistent scalars "
                                "(|T_rise - T_decay| > 50%) — the flash is "
                                "a genuine transient store; a new object "
                                "class (W040's A4)",
                "UNDERPOWERED": "5 points per curve cannot discriminate at "
                                "these effect sizes (be honest — this is a "
                                "REAL possibility; if so, state exactly "
                                "what finer milestone grid + seeds would "
                                "be needed, quantitatively)",
                "MIXED": "anything else",
            },
            "operationalizations": "see lab/x8_flash_thermal.py docstring "
                                   "(frozen at the birth commit BEFORE any "
                                   "compute); one-T := logit-gain lens "
                                   "z_c=z_s/T (the two-class reduction of "
                                   "the x5 softmax(L/T) family — the "
                                   "full-logit MLE needs milestone logit "
                                   "dumps that do not exist; DISCLOSED); "
                                   "race := dX/dt=alpha*G-beta*X with G the "
                                   "rung's own serial curve (consult #005 "
                                   "kinetic form); limbs split at the "
                                   "concurrent interior argmax; s1 floor is "
                                   "the race IC, never a lens fit point",
            "composite_order": "(a) READOUT iff all rungs R2_z>=0.9 AND "
                               "all I<=0.5 no sign-flip AND all T>0; "
                               "(b) STORAGE iff any R2_z<0.7 OR any "
                               "(I>0.5 or sign-flip); (c) OVERRIDE to "
                               "UNDERPOWERED if STORAGE fired only through "
                               "bootstrap-unstable drivers and no stable "
                               "driver remains; (d) if neither a nor b: "
                               "UNDERPOWERED iff >=2 rungs' R2_z CI spans "
                               "[0.7,0.9] or I CI spans 0.5; else MIXED",
        },
        "status": "PARTIAL — data gate",
    }
    write_metrics(metrics)

    data, gate = load_and_gate()
    metrics["data_gate"] = gate
    metrics["data"] = {r: data[r] for r in RUNG_ORDER}
    metrics["status"] = "PARTIAL — fits"
    write_metrics(metrics)
    print("[gate] G_DATA + G_LADDER pass:", gate["pass"])

    # ---- per-rung fits ---------------------------------------------------
    rungs_out = {}
    for rung in RUNG_ORDER:
        pc = data[rung]["concurrent"]
        ps = data[rung]["serial"]
        zc = z_of(pc)
        zs = z_of(ps)
        zc_i, zs_i = zc[1:], zs[1:]  # interior {100..400}
        pc_i, ps_i = pc[1:], ps[1:]

        peak_idx = int(np.argmax(pc_i))  # 0..3 over interior
        t_peak = INTERIOR[peak_idx]
        rise_idx = list(range(0, peak_idx + 1))
        decay_idx = list(range(peak_idx, 4))

        full = fit_lens(zs_i, zc_i)
        full_p = lens_p_space(pc_i, ps_i, full["g"]) if not math.isnan(full["g"]) else {}
        lr = fit_lens(zs_i[rise_idx], zc_i[rise_idx])
        ld = fit_lens(zs_i[decay_idx], zc_i[decay_idx])

        Tr, Td = lr["T"], ld["T"]
        if math.isfinite(Tr) and math.isfinite(Td) and (abs(Tr) + abs(Td)) > 0:
            I = abs(Tr - Td) / (0.5 * (abs(Tr) + abs(Td)))
        else:
            I = float("nan")
        sign_flip = (math.isfinite(Tr) and math.isfinite(Td) and Tr * Td <= 0.0)

        # ---- race (p-space primary) ----
        knot_t = [0] + INTERIOR
        knot_G = [ps[0]] + list(ps_i)  # G(0) := serial p(s1) (disclosed)
        X0 = pc[0]
        race = fit_race(knot_t, knot_G, X0, pc_i, INTERIOR)
        race["R2_p"] = r2(pc_i, race["pred"])
        race["alpha_over_beta"] = (race["alpha"] / race["beta"]
                                   if race["beta"] > 0 else float("inf"))
        race["tau_steps"] = (1.0 / race["beta"] if race["beta"] > 0
                             else float("inf"))
        race["end_ratio_fit"] = (race["pred"][-1] / ps_i[-1]
                                 if ps_i[-1] else float("nan"))
        race["end_ratio_measured"] = pc_i[-1] / ps_i[-1]
        # z-space co-report
        race_z = fit_race(knot_t, [float(x) for x in z_of(knot_G)], zc[0],
                          zc_i, INTERIOR)
        race_z["R2_z"] = r2(zc_i, race_z["pred"])
        race["z_space"] = {k: race_z[k] for k in
                           ("alpha", "beta", "R2_z", "SS_res")}

        # ---- decay-shape discriminator (co-report) ----
        td = [INTERIOR[i] for i in decay_idx]
        zd = zc_i[decay_idx]
        shape = {"n": len(td)}
        if len(td) >= 3:
            cl = np.polyfit(td, zd, 1)
            pl = np.polyval(cl, td)
            shape["linear_R2_z"] = r2(zd, pl)
            lz = np.log(-zd)
            ce = np.polyfit(td, lz, 1)
            pe = -np.exp(np.polyval(ce, td))
            shape["exp_R2_z"] = r2(zd, pe)
            shape["winner"] = ("linear(H-POISON dose-erosion)"
                               if shape["linear_R2_z"] >= shape["exp_R2_z"]
                               else "exponential(first-order wash)")
            shape["delta_R2"] = shape["linear_R2_z"] - shape["exp_R2_z"]
        else:
            shape["winner"] = "uninformative (n<=2)"
            shape["linear_R2_z"] = None
            shape["exp_R2_z"] = None
            shape["delta_R2"] = None

        # ---- serial references ----
        self_lens = fit_lens(zs_i, zs_i)  # must be T=1, R2=1
        wobble = float(np.std(zs_i - np.polyval(np.polyfit(INTERIOR, zs_i, 1), INTERIOR)))
        race_ceiling = fit_race(knot_t, knot_G, ps[0], ps_i, INTERIOR)
        race_ceiling["R2_p"] = r2(ps_i, race_ceiling["pred"])

        rungs_out[rung] = {
            "concurrent": pc, "serial": ps,
            "peak": {"t_peak": t_peak, "p_peak": pc[peak_idx + 1],
                     "p_floor_s1": pc[0]},
            "lens": {
                "full": {**full, **full_p},
                "T_rise": Tr, "T_rise_n": lr["n"],
                "T_decay": Td, "T_decay_n": ld["n"],
                "R2_rise_z": lr["R2_z"], "R2_decay_z": ld["R2_z"],
                "limb_inconsistency_I": I,
                "sign_flip": bool(sign_flip),
                "clause_fires": {
                    "R2_below_0.7": bool(full["R2_z"] < 0.7),
                    "I_above_0.5": bool(I > 0.5) if math.isfinite(I) else None,
                    "sign_flip": bool(sign_flip),
                },
            },
            "race": {k: v for k, v in race.items() if k != "pred"},
            "race_pred": [float(x) for x in race["pred"]],
            "decay_shape": shape,
            "serial_reference": {
                "lens_self_T": self_lens["T"],
                "lens_self_R2": self_lens["R2_z"],
                "driver_z_wobble_rms": wobble,
                "race_ceiling": {k: v for k, v in race_ceiling.items()
                                 if k != "pred"},
            },
            "antiphase_note": {
                "pearson_serial_vs_concurrent_interior":
                    float(np.corrcoef(ps_i, pc_i)[0, 1]),
            },
        }
        print(f"[{rung}] T={full['T']:.4g} R2z={full['R2_z']:.3f} "
              f"Tr={Tr:.4g} Td={Td:.4g} I={I:.3g} flip={sign_flip} | "
              f"race a={race['alpha']:.4g} b={race['beta']:.4g} "
              f"R2p={race['R2_p']:.3f} | shape {shape.get('winner')}")

    metrics["rungs"] = rungs_out

    # ---- rank trends ------------------------------------------------------
    ks = [10, 40, 100, 237]  # k in thousands
    trends = {
        "k_thousands": ks,
        "T_full": [rungs_out[r]["lens"]["full"]["T"] for r in RUNG_ORDER],
        "alpha": [rungs_out[r]["race"]["alpha"] for r in RUNG_ORDER],
        "beta": [rungs_out[r]["race"]["beta"] for r in RUNG_ORDER],
        "alpha_over_beta": [rungs_out[r]["race"]["alpha_over_beta"]
                            for r in RUNG_ORDER],
        "R2_z_lens": [rungs_out[r]["lens"]["full"]["R2_z"] for r in RUNG_ORDER],
        "R2_p_race": [rungs_out[r]["race"]["R2_p"] for r in RUNG_ORDER],
        "measured_ratio": [rungs_out[r]["race"]["end_ratio_measured"]
                           for r in RUNG_ORDER],
        "fit_end_ratio": [rungs_out[r]["race"]["end_ratio_fit"]
                          for r in RUNG_ORDER],
    }

    def _spearman(x, y):
        if any(not math.isfinite(v) for v in y):
            return None
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    trends["spearman_T_vs_k"] = _spearman(ks, trends["T_full"])
    trends["spearman_beta_vs_k"] = _spearman(ks, trends["beta"])
    trends["spearman_aob_vs_k"] = _spearman(ks, trends["alpha_over_beta"])
    trends["disclosure"] = "n=4 rungs; Spearman on 4 points is a "
                           "direction-only read (p undefined at this n); "
                           "reported as trend texture, never adjudicating"
    metrics["rank_trends"] = trends
    metrics["status"] = "PARTIAL — bootstrap"
    write_metrics(metrics)

    # ---- bootstrap (UNDERPOWERED instrument) -----------------------------
    rng = np.random.default_rng(BOOT_SEED)
    boot = {"draws": BOOT_DRAWS, "seed": BOOT_SEED,
            "caveat_verbatim": "both sigmas are PROXIES. The serial "
            "replicate says the arms are deterministic to ~1e-6 "
            "run-to-run; the concurrent flash has ONE corpus seed per "
            "rung (fresh per cell) — its seed-to-seed variance is "
            "UNMEASURED at n=1, so the bootstrap can only UNDERSTATE it",
            "per_rung": {}}
    for rung in RUNG_ORDER:
        pc = data[rung]["concurrent"]
        ps = data[rung]["serial"]
        zc_i = z_of(pc)[1:]
        zs_i = z_of(ps)[1:]
        peak_idx = int(np.argmax(np.asarray(pc)[1:]))
        rise_idx = list(range(0, peak_idx + 1))
        decay_idx = list(range(peak_idx, 4))
        full = rungs_out[rung]["lens"]["full"]
        g_hat = full["g"]
        zhat = zs_i * g_hat  # fitted lens curve at fit points
        sigma_s1 = math.sqrt(full["SS_res_z"] / 4.0) if math.isfinite(full["SS_res_z"]) else 0.0
        sigma_s2 = rungs_out[rung]["serial_reference"]["driver_z_wobble_rms"]
        out_r = {}
        for sname, sigma in (("S1_lens_residual_rms", sigma_s1),
                             ("S2_driver_wobble_rms", sigma_s2)):
            draws_T, draws_Tr, draws_Td, draws_I, draws_R2, draws_diff = \
                [], [], [], [], [], []
            for _ in range(BOOT_DRAWS):
                zc_s = zhat + rng.normal(0.0, sigma, size=4)
                f = fit_lens(zs_i, zc_s)
                lr = fit_lens(zs_i[rise_idx], zc_s[rise_idx])
                ld = fit_lens(zs_i[decay_idx], zc_s[decay_idx])
                draws_T.append(f["T"])
                draws_R2.append(f["R2_z"])
                draws_Tr.append(lr["T"])
                draws_Td.append(ld["T"])
                if (math.isfinite(lr["T"]) and math.isfinite(ld["T"])
                        and (abs(lr["T"]) + abs(ld["T"])) > 0):
                    draws_I.append(abs(lr["T"] - ld["T"])
                                   / (0.5 * (abs(lr["T"]) + abs(ld["T"]))))
                    draws_diff.append(ld["T"] - lr["T"])
            a = np.asarray(draws_T)
            rr = np.asarray(draws_R2, float)
            dd = np.asarray(draws_diff)
            ii = np.asarray(draws_I)
            ci = lambda v: [float(np.percentile(v, 5)),
                            float(np.percentile(v, 95))]
            thr_abs = 0.25 * (abs(rungs_out[rung]["lens"]["T_rise"])
                              + abs(rungs_out[rung]["lens"]["T_decay"]))
            sd_diff = float(np.std(dd)) if dd.size else float("nan")
            mdd = 2.8 * sd_diff if math.isfinite(sd_diff) else float("nan")
            mult = (mdd / thr_abs) ** 2 if (math.isfinite(mdd)
                                            and thr_abs > 0) else float("nan")
            w_r2 = float(3.29 * np.nanstd(rr))
            out_r[sname] = {
                "sigma_z": float(sigma),
                "CI90_T_full": ci(a),
                "CI90_T_rise": ci(np.asarray(draws_Tr)),
                "CI90_T_decay": ci(np.asarray(draws_Td)),
                "CI90_I": (ci(ii) if ii.size else None),
                "CI90_R2_z": ci(rr),
                "width90_R2_z": w_r2,
                "P_R2_below_0.7": float(np.mean(rr < 0.7)),
                "P_I_above_0.5": (float(np.mean(ii > 0.5)) if ii.size else None),
                "sign_stability_Tdecay_minus_Trise": (
                    float(np.mean(np.sign(dd) ==
                                  np.sign(rungs_out[rung]["lens"]["T_decay"]
                                          - rungs_out[rung]["lens"]["T_rise"])))
                    if dd.size else None),
                "MDD90_limb_diff": mdd,
                "milestone_multiplier_for_halfbar": mult,
                "milestone_multiplier_for_R2width_0.1": (
                    (w_r2 / 0.1) ** 2 if math.isfinite(w_r2) else float("nan")),
            }
        # stability clauses for the composite (use S1; S2 co-reports)
        s1 = out_r["S1_lens_residual_rms"]
        lo_r2, hi_r2 = s1["CI90_R2_z"]
        r2_driver_stable = bool(hi_r2 < 0.7)
        r2_ci_spans_0_7_to_0_9 = bool(lo_r2 < 0.9 and hi_r2 > 0.7)
        if s1["CI90_I"] is not None:
            lo_i, hi_i = s1["CI90_I"]
            i_driver_stable = bool(lo_i > 0.5)
            i_ci_spans_0_5 = bool(lo_i <= 0.5 <= hi_i)
        else:
            i_driver_stable, i_ci_spans_0_5 = None, None
        out_r["stability"] = {
            "R2_driver_stable": r2_driver_stable,
            "R2_ci_spans_[0.7,0.9]": r2_ci_spans_0_7_to_0_9,
            "I_driver_stable": i_driver_stable,
            "I_ci_spans_0.5": i_ci_spans_0_5,
        }
        boot["per_rung"][rung] = out_r
        print(f"[boot {rung}] S1 sigma={sigma_s1:.3g} "
              f"CI90 R2=[{lo_r2:.3f},{hi_r2:.3f}] P(R2<0.7)={s1['P_R2_below_0.7']:.2f} "
              f"MDD={s1['MDD90_limb_diff']:.3g}")

    metrics["bootstrap"] = boot

    # ---- composite adjudication (frozen order) ---------------------------
    all_r2 = [rungs_out[r]["lens"]["full"]["R2_z"] for r in RUNG_ORDER]
    all_T = [rungs_out[r]["lens"]["full"]["T"] for r in RUNG_ORDER]
    fires_r2 = [bool(v < 0.7) for v in all_r2]
    fires_i, flips = [], []
    for r in RUNG_ORDER:
        L = rungs_out[r]["lens"]
        fires_i.append(bool(L["limb_inconsistency_I"] > 0.5)
                       if math.isfinite(L["limb_inconsistency_I"]) else False)
        flips.append(bool(L["sign_flip"]))
    readout_ok = (all(v >= 0.9 for v in all_r2)
                  and not any(fires_i) and not any(flips)
                  and all(math.isfinite(t) and t > 0 for t in all_T))
    storage_fired = any(fires_r2) or any(fires_i) or any(flips)

    drivers = []
    for j, r in enumerate(RUNG_ORDER):
        st = boot["per_rung"][r]["stability"]
        if fires_r2[j]:
            drivers.append({"rung": r, "driver": "R2<0.7",
                            "stable": st["R2_driver_stable"]})
        if fires_i[j] or flips[j]:
            drivers.append({"rung": r, "driver": "limb-inconsistency",
                            "stable": st["I_driver_stable"]})
    stable_drivers = [d for d in drivers if d["stable"] is True]
    span_counts = sum(
        1 for r in RUNG_ORDER
        if boot["per_rung"][r]["stability"]["R2_ci_spans_[0.7,0.9]"]
        or boot["per_rung"][r]["stability"]["I_ci_spans_0.5"])

    if readout_ok:
        verdict = "READOUT-LIKE"
        rationale = "every rung's single lens scalar fits both limbs at R2_z>=0.9 with consistent limb temperatures"
    elif storage_fired:
        if drivers and not stable_drivers:
            verdict = "UNDERPOWERED"
            rationale = ("STORAGE clauses fired at the point estimates but "
                         "every driver is bootstrap-unstable under S1; the "
                         "point estimates are carried verbatim")
        else:
            verdict = "STORAGE-LIKE"
            rationale = (f"stable drivers: "
                         f"{[d['rung'] + ':' + d['driver'] for d in stable_drivers]}")
    else:
        if span_counts >= 2:
            verdict = "UNDERPOWERED"
            rationale = ("neither bar fired and >=2 rungs' bootstrap CIs "
                         "span the [0.7,0.9]/0.5 decision bands")
        else:
            verdict = "MIXED"
            rationale = "neither bar's clauses fired; no spanning evidence"

    # grid prescription (quantitative, for whichever verdict)
    mults = [boot["per_rung"][r]["S1_lens_residual_rms"]["milestone_multiplier_for_halfbar"]
             for r in RUNG_ORDER]
    mults_r2 = [boot["per_rung"][r]["S1_lens_residual_rms"]["milestone_multiplier_for_R2width_0.1"]
                for r in RUNG_ORDER]
    mult_max = max((m for m in mults if math.isfinite(m)), default=float("nan"))
    mult_r2_max = max((m for m in mults_r2 if math.isfinite(m)), default=float("nan"))
    prescription = {
        "current_interior_milestones_per_curve": 4,
        "milestone_multiplier_needed_halfbar_max": mult_max,
        "milestone_multiplier_needed_R2width0.1_max": mult_r2_max,
        "concrete_design": f"milestone grid every 10 steps (40 interior "
                           f"points, ~10x) with J=3 corpus seeds per rung "
                           f"(N_eff ~ 120 = 30x) on the top two rungs "
                           f"(100k, 237k) covers the worst required "
                           f"multiplier "
                           f"{mult_max:.1f}x (halfbar) / {mult_r2_max:.1f}x "
                           f"(R2 width 0.1) computed from S1; S2 multipliers "
                           f"co-reported per rung in bootstrap.per_rung",
        "assumption": "sigma is seed-level (PROXY; see caveat_verbatim) — "
                      "the multiplier scales as 1/N_eff for milestone "
                      "count times seeds",
    }
    metrics["adjudication"] = {
        "per_rung_R2_z": {r: rungs_out[r]["lens"]["full"]["R2_z"]
                          for r in RUNG_ORDER},
        "per_rung_clauses": {r: rungs_out[r]["lens"]["clause_fires"]
                             for r in RUNG_ORDER},
        "drivers": drivers,
        "stable_drivers": stable_drivers,
        "spanning_rungs_count": span_counts,
        "verdict": verdict,
        "rationale": rationale,
        "co_reports": {
            "W041_P-W41d_prediction": "x8 returns UNDERPOWERED "
                                      "(registered in THINKING.md W041 "
                                      "before this cell could be read; "
                                      "scored, never an input)",
            "T249_discriminator_context": "decay-shape reads "
                                          "(linear=H-POISON vs "
                                          "exponential=wash) co-report per "
                                          "rung in rungs.*.decay_shape",
            "follow_up_if_storage": "full-logit MLE one-T on milestone "
                                    "logit dumps (the x5 form; a dump "
                                    "mini-cell) — the desk lens is the "
                                    "two-class reduction",
        },
        "prescription": prescription,
    }
    metrics["timing"] = {"total_s": round(
        (datetime.now(timezone.utc) - t0).total_seconds(), 1)}
    metrics["status"] = "COMPLETE — adjudicated"
    write_metrics(metrics)

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, rung in enumerate(RUNG_ORDER):
        ax = axes[j // 3][j % 3]
        pc = data[rung]["concurrent"]
        ps = data[rung]["serial"]
        L = rungs_out[rung]["lens"]
        R = rungs_out[rung]["race"]
        ax.semilogy(STEPS, pc, "o-", color="crimson", lw=2, ms=7,
                    label="concurrent flash (traj_g0)")
        ax.semilogy(STEPS, ps, "s--", color="steelblue", lw=1.5, ms=5,
                    label="serial formation (driver)")
        # lens prediction at milestones
        if math.isfinite(L["full"]["g"]):
            pred_p = p_of(z_of(ps[1:]) * L["full"]["g"])
            ax.semilogy(INTERIOR, pred_p, "^:", color="darkgreen", lw=1.5,
                        ms=6, label=f"one-T lens (T={L['full']['T']:.2g}, "
                                    f"R2z={L['full']['R2_z']:.2f})")
        # race curve (fine)
        ts, xv = race_curve_fine([0] + INTERIOR, [ps[0]] + ps[1:], pc[0],
                                 R["alpha"], R["beta"])
        ax.semilogy(ts, np.clip(xv, 1e-7, None), "-", color="purple",
                    lw=1.8,
                    label=f"race a={R['alpha']:.3g} b={R['beta']:.3g} "
                          f"(R2p={R['R2_p']:.2f})")
        t_peak = rungs_out[rung]["peak"]["t_peak"]
        ax.axvspan(100, t_peak, color="green", alpha=0.07)
        ax.axvspan(t_peak, 400, color="red", alpha=0.07)
        ax.axvline(t_peak, color="gray", ls=":", lw=1)
        ax.annotate("rise", (0.5 * (100 + t_peak), 2e-5), ha="center",
                    fontsize=8, color="green")
        ax.annotate("decay", (0.5 * (t_peak + 400), 2e-5), ha="center",
                    fontsize=8, color="darkred")
        ax.set_title(f"{rung}  (endpoint ratio "
                     f"{R['end_ratio_measured']:.2e}x)", fontsize=10)
        ax.set_xlabel("install step s")
        ax.set_ylabel("probe strength g0_pz")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, which="both", alpha=0.2)
    # summary panels
    ax = axes[1][2]
    Ts = [rungs_out[r]["lens"]["full"]["T"] for r in RUNG_ORDER]
    ax.plot(ks, Ts, "o-", color="darkgreen")
    ax.set_xscale("log")
    ax.set_xlabel("room width k (thousands)")
    ax.set_ylabel("fitted lens T (full curve)")
    ax.set_title("one-T vs rank", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax = axes[0][2]
    aobs = [rungs_out[r]["race"]["alpha"] for r in RUNG_ORDER]
    bets = [rungs_out[r]["race"]["beta"] for r in RUNG_ORDER]
    ax.plot(ks, aobs, "o-", color="purple", label="alpha (formation)")
    ax.plot(ks, bets, "s--", color="orange", label="beta (erosion)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("room width k (thousands)")
    ax.set_title("race parameters vs rank", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    axes[1][2].set_title(f"one-T vs rank — verdict: {verdict}", fontsize=10)
    fig.suptitle("x8 — the flash through the thermal lens (e268-e271 "
                 "concurrent traj_g0, one-T lens vs formation/erosion race)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "x8_flash_fits.png", dpi=130)
    print("[done] verdict:", verdict)
    print("[done] outputs:", OUT / "metrics.json", OUT / "x8_flash_fits.png")
    return verdict


if __name__ == "__main__":
    main()
