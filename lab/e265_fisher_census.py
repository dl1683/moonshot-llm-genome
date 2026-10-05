"""E265 — THE FISHER/NTK CENSUS (T240 cell (a); the owner's Jacobian /
natural-gradient directive, 2026-10-05 ~16:40Z).

DESK-ONLY on COMMITTED records. No model, no GPU, no corpus: every number
is closed-form linear algebra on the committed Gram/dmat/traj records of
runs/e234 (+ e231/e246/e240/e254/e261 metrics). The object: the empirical
Fisher's nonzero spectrum over each wash's distribution IS the eigenbasis
of the wash gradients' Gram (F_emp = (1/S) sum_t g_t g_t^T; its nonzero
eigenvalues = eigenvalues of the S x S Gram G[t,s] = <g_t, g_s>, to within
the 1/S scaling, which does not touch any ratio reported here).

REGISTRATION (frozen BEFORE any compute below is run; source: the e265
dispatch = T240's registered cell (a); bars VERBATIM):

  CLIFF-IS-THE-KNEEL — "the Fisher spectrum's knee (largest log-eigengap)
  lands inside the expression threshold's bracket [1k, 237k] (or within a
  factor 3 of its edges at both scales measured) — THE MEMORY CAPACITY LAW
  AND THE OPTIMIZATION GEOMETRY ARE ONE OBJECT: the network stores what
  the Fisher cannot normalize away"

  CLIFF-IS-SEPARATE — "the knee sits far from the bracket (> 3x either
  edge) — the capacity cliff and the Fisher's decay are distinct objects;
  the honest map"

  MIXED — "the spectra verbatim (both scales), no inflation"

Operationalization (frozen with the bars):
  knee := argmax_i log10(lambda_{i-1}/lambda_i), i = 1..S-1 (the split
  index k; largest log-eigengap in the head); "within a factor 3 of its
  edges" := knee in [1000/3, 3*237123]; "the bracket" := e261's committed
  threshold_bracket [1000, 237123] at the 2.74M g1c root (e246 G_ROOTSPAN
  n_params 2,739,072 — the scale where the cliff was measured).
  Scales measured := (i) the 2.74M root's own 20-step gradient history
  (e231 J1's committed Gram-SVD sv list = the small-model side; e246's
  late-half sv + L2 counts as the coarser co-report), and (ii) the 124M
  GPT-2 wash Grams (e234's committed 80x80 gram_scaled_{w1,w2,w3} in true
  units). Adjudication: KNEEL iff the knee is in the factor-3 band at BOTH
  scales; SEPARATE iff the knee is outside the band (> 3x from an edge) at
  BOTH scales; MIXED otherwise, or when a knee/erank is window-capped (see
  the caveat) so one scale cannot adjudicate.

THE SPECTRUM-ESTIMATE CAVEAT (disclosed prominently, per the dispatch):
80 gradient samples give AT MOST 80 nonzero eigenvalues of a
124,439,808-dimensional operator (the remaining 124,439,728 eigenvalues of
the sample sum are EXACTLY zero); likewise 20 steps bound the 2.74M side.
Every erank at a deep mass threshold that equals its window size is a
LOWER BOUND ONLY — the census measures the TOP of the Fisher spectrum and
the SHAPE of its decay within the window, not the tail's extent. The
Gram's fp16 row provenance (2^14-scaled unit rows) puts an absolute noise
floor under the small eigenvalues; the floor and its effect on the NTK
pseudo-inverse are measured and disclosed (G_FLOOR, pinv sensitivity).

Cells:
  (1) THE FISHER SPECTRUM — full eigen-decomposition (float64) of the
      three true-unit 80x80 wash Grams: decay e_i/e_1, erank at explained-
      mass thresholds {1e-2, 1e-3, 1e-6, 1e-10}, participation ratio,
      knee; cross-wash top-k eigenspace alignment via the committed cross
      Grams (is the top eigenspace a distribution property?).
  (2) THE CLIFF JOIN — the same statistics on the 2.74M side (e231 J1's
      committed sv spectrum; e246 late-half co-report); knee/erank vs the
      bracket [1000, 237123], absolute and fraction units; adjudication
      against the frozen bars.
  (3) THE NTK BLOCK STRUCTURE — the 54 supports' cross-probe Gram through
      the gradient basis: K_proj = D^T G^+ D (54x54), the supports'
      span-capture fractions, the killed-vs-living block asymmetry
      (fates recomputed from e234's committed trajectories under e240's
      frozen definition p(+80) < 0.5 p(0)), vs e240's committed
      full-space overlap numbers.
  (4) THE NATURAL-GRADIENT GAP (co-report) — exact coefficient-domain
      natural steps n_t = F_A^{+k} g_t (split-half: A = first 40 steps)
      and cos(g_t, n_t); the natural step's PC1 concentration; Adam's
      actual applied direction d_t = m/sqrt(v) (e240's committed per-step
      cos_d/cos_g vs the 54 supports) read through the same support
      window against the exact top-eigenvector support profiles — how
      much of the natural step's structure does the diagonal capture?
      FULL-SPACE angle(d_t, n_t) is NOT computable from committed
      records (no per-coordinate v is committed) — every Adam read here
      is window-restricted and labeled as such.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
RD = RUNS / "e265"
SMOKE = os.environ.get("E265_SMOKE", "") == "1"

CACHE_SCALE = 2 ** 14          # e226/e234 fp16 cache convention
SQ = float(CACHE_SCALE ** 2)   # 2^28 — the scaled-Gram divisor
WASHES = ("w1", "w2", "w3")
S80 = 80                       # committed wash steps
HALF = 40
N_SUP = 54
N_PARAMS_124M = 124_439_808    # e234 organism (GPT-2, gpt2 revision 607a30d)
N_PARAMS_2_74M = 2_739_072     # e246 G_ROOTSPAN (g1c root)
BRACKET = (1000, 237123)       # e261 threshold_bracket (verbatim)
MASS_TOL = (1e-2, 1e-3, 1e-6, 1e-10)
FATE_DEF = "killed := p(+80) < 0.5*p(0) (e240's frozen clause)"
E234_WIND = {"w1": 0.05528856465742009, "w2": 0.057170082712768175,
             "w3": 0.021459122287842882}   # e234 committed wind_share_median_A
E240_OVERLAP = {   # e240 committed fate_overlap_structure (full space)
    "w1": {"killed": 0.02421974898020478, "living": 0.008656070320444229,
           "cross": 0.012064452400965835},
    "w2": {"killed": 0.023233671820371046, "living": 0.010570070610344142,
           "cross": 0.012643927588368844},
    "w3": {"killed": 0.02468662500150929, "living": 0.0104504,
           "cross": None},     # w3 living/cross filled from the record at load
}

t0 = time.time()
LOG_LINES: list[str] = []


def log(msg: str):
    line = f"[e265 +{time.time()-t0:6.1f}s] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_metrics(metrics: dict, tag: str):
    metrics["status"] = tag
    RD.mkdir(parents=True, exist_ok=True)
    (RD / "metrics.json").write_text(
        json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    log(f"metrics written ({tag})")


# ------------------------------------------------------------ spectrum stats

def spectrum_stats(ev: np.ndarray) -> dict:
    """Frozen statistics of a descending eigenvalue vector (fp64).
    Negative numerical eigenvalues are clipped to 0 and COUNTED (never
    hidden); clipped values cannot inflate any erank below the clip."""
    ev = np.asarray(ev, dtype=np.float64)
    ev = np.clip(ev, 0.0, None)
    ev = np.sort(ev)[::-1]
    lam1 = float(ev[0])
    tot = float(ev.sum())
    pos = ev[ev > 0]
    pr = float(tot ** 2 / np.sum(pos ** 2)) if tot > 0 else 0.0
    cum = np.cumsum(ev) / tot if tot > 0 else np.zeros_like(ev)
    erank = {}
    for tol in MASS_TOL:
        k = int(np.searchsorted(cum, 1.0 - tol) + 1)
        k = min(k, len(ev))
        erank[str(tol)] = {"k": k, "window_capped": bool(k >= len(ev))}
    # knee: largest log-eigengap (frozen rule)
    gaps = np.log10(np.maximum(ev[:-1] / np.maximum(ev[1:], 1e-300), 1e-300))
    knee = int(np.argmax(gaps)) + 1
    top_gaps = sorted(
        [{"i": int(i) + 1, "log10_ratio": float(gaps[i])}
         for i in range(len(gaps))], key=lambda d: -d["log10_ratio"])[:5]
    return {
        "n": int(len(ev)),
        "lambda_1": lam1, "lambda_n_over_1": float(ev[-1] / lam1),
        "n_negative_clipped": int((np.asarray(ev) == 0).sum()),
        "participation_ratio": pr,
        "erank": erank,
        "cum_at": {str(k): float(cum[k - 1]) for k in (1, 2, 3, 5, 10, 20, 40, 80)
                   if k <= len(ev)},
        "decay_ei_over_e1": {str(i): float(ev[i - 1] / lam1)
                             for i in (1, 2, 3, 5, 10, 20, 40, 80) if i <= len(ev)},
        "knee_loggap": {"k": knee, "log10_ratio": float(gaps[knee - 1]),
                        "top5": top_gaps},
        "spectrum": ev.tolist(),
    }


# ------------------------------------------------------------ main

def main() -> int:
    metrics: dict = {
        "experiment": "e265_fisher_census",
        "phase": ("THE FISHER/NTK CENSUS — desk-only, committed records: "
                  "the wash Grams' full nonzero spectrum (the empirical "
                  "Fisher over each wash distribution), the knee-vs-cliff "
                  "join at both scales, the NTK blocks through the gradient "
                  "basis, the natural-gradient gap co-report"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registration": ("bars VERBATIM from the e265 dispatch (T240 cell "
                         "(a)), frozen before compute; see docstring; no bar "
                         "shopping"),
        "bars_verbatim": {
            "CLIFF-IS-THE-KNEEL": ("the Fisher spectrum's knee (largest "
                                   "log-eigengap) lands inside the expression "
                                   "threshold's bracket [1k, 237k] (or within "
                                   "a factor 3 of its edges at both scales "
                                   "measured) — THE MEMORY CAPACITY LAW AND "
                                   "THE OPTIMIZATION GEOMETRY ARE ONE OBJECT: "
                                   "the network stores what the Fisher cannot "
                                   "normalize away"),
            "CLIFF-IS-SEPARATE": ("the knee sits far from the bracket (> 3x "
                                  "either edge) — the capacity cliff and the "
                                  "Fisher's decay are distinct objects; the "
                                  "honest map"),
            "MIXED": "the spectra verbatim (both scales), no inflation",
        },
        "operationalization": {
            "knee": "argmax_i log10(lam_{i-1}/lam_i); the split index k",
            "factor3_band": "[1000/3, 3*237123]",
            "bracket": {"lo": BRACKET[0], "hi": BRACKET[1],
                        "source": "runs/e261/metrics.json adjudication.reads."
                                  "threshold_bracket (the 2.74M g1c root)"},
            "scales": ["2.74M: e231 J1 committed Gram-SVD sv (20 steps)",
                       "124M: e234 committed gram_scaled_{w} (80 steps)"],
            "adjudication": ("KNEEL iff knee in band at BOTH scales; SEPARATE "
                             "iff knee outside band at BOTH scales; MIXED "
                             "otherwise or when window caps prevent "
                             "adjudication at one scale"),
            "erank": ("smallest k with cumulative explained mass >= 1-tol; "
                      "k == window size means LOWER BOUND ONLY"),
            "fates": FATE_DEF,
        },
        "spectrum_estimate_caveat": (
            "80 samples give at most 80 nonzero eigenvalues of a "
            "124,439,808-dim operator (the rest of the SAMPLE sum is exactly "
            "zero); 20 steps bound the 2.74M side the same way. This census "
            "measures the TOP of the empirical Fisher's spectrum and its "
            "decay shape within the window — NOT the tail's extent. Deep-"
            "threshold eranks that hit the window cap are lower bounds."),
        "compute": {"device": "CPU desk only (threads: numpy default, tiny "
                              "matrices)", "gpu_touched": False},
        "gates": {},
    }

    # ---------------- load committed records + provenance
    src = {}
    prov = {}
    for tag, rel in [("e234_journal", "e234/journal.json"),
                     ("e234_metrics", "e234/metrics.json"),
                     ("e231_metrics", "e231/metrics.json"),
                     ("e240_journal", "e240/journal.json"),
                     ("e240_metrics", "e240/metrics.json"),
                     ("e246_metrics", "e246/metrics.json"),
                     ("e261_metrics", "e261/metrics.json"),
                     ("e254_metrics", "e254/metrics.json")]:
        p = RUNS / rel
        src[tag] = json.loads(p.read_text(encoding="utf-8"))
        prov[tag] = {"file": f"runs/{rel}", "md5": md5(p)}
    metrics["provenance_records"] = prov
    J = src["e234_journal"]
    log("committed records loaded (8 files, md5s recorded)")

    # fill e240's w3 overlap numbers verbatim
    fos = src["e240_metrics"]["reads"]["fate_overlap_structure"]
    for w in WASHES:
        E240_OVERLAP[w] = {"killed": fos[f"{w}_within_killed_mean_cos"],
                           "living": fos[f"{w}_within_living_mean_cos"],
                           "cross": fos[f"{w}_cross_mean_cos"]}

    # ---------------- CELL 1: the 124M wash Fisher spectra
    grams_true, gnorms_all = {}, {}
    diag_dev_max, sym_dev_max = 0.0, 0.0
    for w in WASHES:
        Gs = np.array(J[f"gram_scaled_{w}"], dtype=np.float64)
        gn = np.array(J[w]["gnorms"], dtype=np.float64)
        G = Gs * np.outer(gn, gn) / SQ          # true-unit Gram
        grams_true[w] = G
        gnorms_all[w] = gn
        diag_dev_max = max(diag_dev_max, float(np.max(np.abs(
            np.diag(G) - gn ** 2)) / np.max(gn ** 2)))
        sym_dev_max = max(sym_dev_max, float(np.max(np.abs(G - G.T))
                                             / np.max(np.abs(G))))
    metrics["gates"]["G_UNITS"] = {
        "check": "max |G[t,t] - gnorm_t^2| / max gnorm^2 (the unit conversion "
                 "identity; G_scaled rows are 2^14-scaled unit grads)",
        "max_dev": diag_dev_max, "tol": 1e-3, "pass": bool(diag_dev_max < 1e-3)}
    metrics["gates"]["G_SYMMETRY"] = {
        "max_dev": sym_dev_max, "tol": 1e-12,
        "pass": bool(sym_dev_max < 1e-12)}
    log(f"G_UNITS dev {diag_dev_max:.2e}; G_SYMMETRY dev {sym_dev_max:.2e}")

    fisher = {}
    for w in WASHES:
        ev = np.linalg.eigvalsh(grams_true[w])[::-1]
        # numerical stability cross-check: sum(eigenvalues) vs trace
        tr = float(np.trace(grams_true[w]))
        trace_dev = abs(float(ev.sum()) - tr) / abs(tr)
        st = spectrum_stats(ev)
        st["fisher_lambda_1"] = float(ev[0] / S80)   # F_emp = G/S
        st["trace_identity_dev"] = trace_dev
        st["n_negative_raw"] = int((np.linalg.eigvalsh(grams_true[w]) < 0).sum())
        fisher[w] = st
        log(f"{w}: lam1 {ev[0]:.3e} lam80/lam1 {ev[-1]/ev[0]:.2e} "
            f"PR {st['participation_ratio']:.2f} knee@{st['knee_loggap']['k']} "
            f"(gap {st['knee_loggap']['log10_ratio']:.2f}) "
            f"erank(1e-2)={st['erank']['0.01']['k']} "
            f"erank(1e-6)={st['erank']['1e-06']['k']}"
            f"{' CAPPED' if st['erank']['1e-06']['window_capped'] else ''}")
    metrics["fisher_spectra_124M"] = fisher

    # knee robustness co-report: the NORM-FREE flavor — the unit-cosine
    # Gram (each gradient direction weighted equally) removes the step-1
    # norm transient (gnorm 22.9 vs plateau ~4.2) that inflates lambda_1.
    # If the knee stays small here too, the CLIFF-IS-SEPARATE verdict is
    # robust to the trajectory's norm shape.
    knee_normfree = {}
    for w in WASHES:
        Gs = np.array(J[f"gram_scaled_{w}"], dtype=np.float64) / SQ
        evnf = np.linalg.eigvalsh(Gs)[::-1]
        stnf = spectrum_stats(evnf)
        knee_normfree[w] = {
            "knee": stnf["knee_loggap"]["k"],
            "log10_ratio": stnf["knee_loggap"]["log10_ratio"],
            "erank_1e-2": stnf["erank"]["0.01"]["k"],
            "participation_ratio": stnf["participation_ratio"],
            "note": "unit-direction Gram (norm-free): the wash's DIRECTION "
                    "spectrum; co-report only",
        }
        log(f"{w} norm-free knee@{stnf['knee_loggap']['k']} "
            f"(gap {stnf['knee_loggap']['log10_ratio']:.2f}) "
            f"erank(1e-2)={stnf['erank']['0.01']['k']} "
            f"PR {stnf['participation_ratio']:.2f}")
    metrics["knee_normfree_coreport"] = knee_normfree

    # cross-wash top-k eigenspace alignment (distribution property check)
    cross = {}
    for a, b in (("w1", "w2"), ("w1", "w3"), ("w2", "w3")):
        C = np.array(J[f"cross_{a}_{b}"], dtype=np.float64)
        C = C * np.outer(gnorms_all[a], gnorms_all[b]) / SQ
        eva, VVa = np.linalg.eigh(grams_true[a])
        evb, VVb = np.linalg.eigh(grams_true[b])
        ia, ib = np.argsort(eva)[::-1], np.argsort(evb)[::-1]
        eva, Va = eva[ia], VVa[:, ia]
        evb, Vb = evb[ib], VVb[:, ib]
        row = {}
        for k in (1, 2, 5, 10, 20):
            A = Va[:, :k] / np.sqrt(np.maximum(eva[:k], 1e-30))
            B = Vb[:, :k] / np.sqrt(np.maximum(evb[:k], 1e-30))
            M = A.T @ C @ B                    # principal-cosine matrix
            sv = np.linalg.svd(M, compute_uv=False)
            row[f"k{k}"] = float(np.mean(sv))
        cross[f"{a}|{b}"] = row
        log(f"top-eigenspace alignment {a}|{b}: k1 {row['k1']:.4f} "
            f"k2 {row['k2']:.4f} k10 {row['k10']:.4f}")
    metrics["cross_wash_eigenspace_alignment"] = cross
    write_metrics(metrics, "PARTIAL: cell 1 (the 124M Fisher spectra) done")

    # ---------------- CELL 2: the cliff join
    # 2.74M side: e231 J1's committed Gram-SVD sv (20 segments, seed-10902
    # stream, e225 machinery); e246's late-half sv (10) + 20-step L2 counts
    sv231 = np.array(
        src["e231_metrics"]["cells"]["J1"]["primary_stream"]["span"]["sv"],
        dtype=np.float64)
    lam_2m = sv231 ** 2
    st_2m = spectrum_stats(lam_2m)
    e231_knee_mass = src["e231_metrics"]["cells"]["J1"]["primary_stream"][
        "knee"]
    sv246 = np.array(
        src["e246_metrics"]["span"]["span"]["sv"], dtype=np.float64)
    st_2m_late = spectrum_stats(sv246 ** 2)
    hist246 = src["e246_metrics"]["span"]["history"]
    metrics["fisher_spectrum_2_74M"] = {
        "primary": {
            "source": "runs/e231/metrics.json cells.J1.primary_stream.span.sv "
                      "(the e131 consolidated root — g1b's own root; the B1 "
                      "bridge organism; seed-10902 20-step unwalled AdamW "
                      "history, e225's chunked fp64 Gram-SVD)",
            "n_params": None,   # 2.74M family — bridge organism
            "stats": st_2m,
            "committed_mass_plateau_knee": {
                "k": e231_knee_mass["k"], "rule": "cumulative mass >= 0.9 "
                "(e231's committed rule, co-reported)",
                "mass_at_k": e231_knee_mass["mass_at_k"]},
        },
        "co_report_g1c_root": {
            "source": "runs/e246/metrics.json span.span.sv (the g1c root — "
                      "the root where e261 measured the cliff; LATE half "
                      "segments 11-20 only; 10 SVs)",
            "stats": st_2m_late,
            "steps_L2_counts": [r["L2"] for r in hist246["rows"]],
            "disclosure": "coarser: 10 SVs of the late half; no full 20x20 "
                          "Gram committed at the g1c root",
        },
        "pr_cross_checks": {
            "e231_committed": 4.117363094870959,
            "recomputed": st_2m["participation_ratio"],
            "e246_committed": 2.0277648570019418,
            "recomputed_late": st_2m_late["participation_ratio"]},
    }
    metrics["gates"]["G_PR231"] = {
        "check": "PR(e231 J1 sv^2) vs the committed 4.117363094870959",
        "dev": abs(st_2m["participation_ratio"] - 4.117363094870959),
        "tol": 1e-6,
        "pass": bool(abs(st_2m["participation_ratio"] - 4.117363094870959)
                     < 1e-6)}
    metrics["gates"]["G_PR246"] = {
        "check": "PR(e246 late sv^2) vs the committed 2.0277648570019418",
        "dev": abs(st_2m_late["participation_ratio"] - 2.0277648570019418),
        "tol": 1e-6,
        "pass": bool(abs(st_2m_late["participation_ratio"]
                         - 2.0277648570019418) < 1e-6)}
    log(f"2.74M side: knee@{st_2m['knee_loggap']['k']} "
        f"(gap {st_2m['knee_loggap']['log10_ratio']:.2f}) "
        f"PR {st_2m['participation_ratio']:.3f} (committed 4.117) "
        f"erank(1e-2)={st_2m['erank']['0.01']['k']} "
        f"erank(1e-6)={st_2m['erank']['1e-06']['k']} CAPPED="
        f"{st_2m['erank']['1e-06']['window_capped']}")

    # the join table
    band = (BRACKET[0] / 3.0, 3.0 * BRACKET[1])
    join = {"band_factor3": {"lo": band[0], "hi": band[1]},
            "bracket": {"lo": BRACKET[0], "hi": BRACKET[1]},
            "sides": {}}

    def side_row(name, stats, n_params, window):
        knee = stats["knee_loggap"]["k"]
        er = {t: stats["erank"][t]["k"] for t in
              ("0.01", "0.001", "1e-06", "1e-10")}
        capped = {t: stats["erank"][t]["window_capped"] for t in er}
        return {
            "name": name, "n_params": n_params, "window": window,
            "knee": knee, "knee_in_band": bool(band[0] <= knee <= band[1]),
            "knee_over_bracket_lo": float(BRACKET[0] / max(knee, 1)),
            "erank": er, "erank_window_capped": capped,
            "erank_as_fraction": {t: (er[t] / n_params) for t in er},
            "erank_in_bracket": {t: bool(BRACKET[0] <= er[t] <= BRACKET[1])
                                 for t in er},
            "bracket_as_fraction": {"lo": BRACKET[0] / n_params,
                                    "hi": BRACKET[1] / n_params},
            "participation_ratio": stats["participation_ratio"],
            "cum_at_knee": stats["cum_at"].get(str(knee)),

        }

    join["sides"]["2.74M_e231J1"] = side_row(
        "2.74M (e231 J1, 20-step window)", st_2m, N_PARAMS_2_74M, 20)
    join["sides"]["124M_w1"] = side_row(
        "124M wash w1 (80-step window)", fisher["w1"], N_PARAMS_124M, S80)
    join["sides"]["124M_w2"] = side_row(
        "124M wash w2 (80-step window)", fisher["w2"], N_PARAMS_124M, S80)
    join["sides"]["124M_w3"] = side_row(
        "124M wash w3 (80-step window)", fisher["w3"], N_PARAMS_124M, S80)
    join["sides"]["2.74M_g1c_late_e246"] = side_row(
        "2.74M g1c root LATE half (e246, 10-SV window)", st_2m_late,
        N_PARAMS_2_74M, 10)

    knees_in = [join["sides"]["2.74M_e231J1"]["knee_in_band"],
                join["sides"]["124M_w1"]["knee_in_band"],
                join["sides"]["124M_w2"]["knee_in_band"],
                join["sides"]["124M_w3"]["knee_in_band"]]
    knee_capped = any(join["sides"][s]["knee"] >= join["sides"][s]["window"]
                      for s in ("2.74M_e231J1", "124M_w1", "124M_w2",
                                "124M_w3"))
    if all(knees_in):
        verdict = "CLIFF-IS-THE-KNEEL"
    elif not any(knees_in) and not knee_capped:
        verdict = "CLIFF-IS-SEPARATE"
    else:
        verdict = "MIXED"
    join["verdict"] = verdict
    join["verdict_logic"] = {
        "knees_in_band": {"2.74M": knees_in[0], "124M_w1": knees_in[1],
                          "124M_w2": knees_in[2], "124M_w3": knees_in[3]},
        "any_knee_window_capped": bool(knee_capped),
        "rule": "KNEEL iff all in band; SEPARATE iff none in band and no "
                "knee at its window cap; MIXED otherwise",
        "texture_disclosed": (
            "deep-threshold eranks are window-capped lower bounds: the "
            "bracket CANNOT be excluded as the Fisher tail's scale — the "
            "join adjudicates the KNEE (the head's decay), not the tail; "
            "this is the measurement limit, not a verdict input"),
    }
    metrics["cliff_join"] = join
    log(f"CLIFF JOIN verdict: {verdict} "
        f"(knees 2.74M@{join['sides']['2.74M_e231J1']['knee']}, "
        f"124M @{fisher['w1']['knee_loggap']['k']}/"
        f"{fisher['w2']['knee_loggap']['k']}/{fisher['w3']['knee_loggap']['k']}; "
        f"bracket [{BRACKET[0]}, {BRACKET[1]}])")
    write_metrics(metrics, "PARTIAL: cell 2 (the cliff join) done")

    # ---------------- CELL 3: the NTK blocks
    # fates from e234's committed trajectories (e240's frozen definition)
    fates = {}
    n_killed = {}
    for w in WASHES:
        t0r = J[w]["traj"]["0"]
        t80 = J[w]["traj"]["80"]
        kf = [r["fact"] for r0, r in zip(t0r, t80) if r["p"] < 0.5 * r0["p"]]
        fates[w] = {"killed_facts": kf,
                    "labels": [r["fact"] in set(kf) for r in t0r]}
        n_killed[w] = len(kf)
    e240_nk = {w: src["e240_metrics"]["fates"]["per_wash"][w]["n_killed"]
               for w in WASHES}
    metrics["gates"]["G_FATES"] = {
        "check": "fates recomputed from e234 traj under e240's definition",
        "recomputed": n_killed, "e240_committed": e240_nk,
        "pass": bool(all(n_killed[w] == e240_nk[w] for w in WASHES))}
    log(f"G_FATES: recomputed {n_killed} vs e240 {e240_nk}")

    ntk = {}
    for w in WASHES:
        G = grams_true[w]
        D = np.array(J[f"dmat_scaled_{w}"], dtype=np.float64) \
            * gnorms_all[w][:, None] / SQ          # <g_t, s_i> true units
        ew, VW = np.linalg.eigh(G)
        order = np.argsort(ew)[::-1]
        ew, VW = ew[order], VW[:, order]
        lam1 = float(ew[0])

        def pinv_at(fl):
            keepm = ew > fl * lam1
            inv = np.where(keepm, 1.0 / np.maximum(ew, 1e-300), 0.0)
            return (VW * inv) @ VW.T, int(keepm.sum())

        # pinv floor sensitivity: relative floors 1e-9 vs 1e-12
        Gp9, rank9 = pinv_at(1e-9)
        Gp12, rank12 = pinv_at(1e-12)
        K9 = D.T @ Gp9 @ D
        K = D.T @ Gp12 @ D                          # 54x54
        rel_dev = float(np.max(np.abs(K9 - K)) /
                        max(np.max(np.abs(K)), 1e-300))
        rank = rank12
        cap = np.diag(K).copy()                     # ||P s_i||^2
        cap = np.clip(cap, 0.0, None)
        denom = np.sqrt(np.outer(cap, cap))
        Kh = np.divide(K, denom, out=np.zeros_like(K),
                       where=denom > 1e-300)        # in-span cosine
        lab = fates[w]["labels"]
        ii_k = [i for i in range(N_SUP) if lab[i]]
        ii_l = [i for i in range(N_SUP) if not lab[i]]

        def mblock(ii, jj):
            vals = [Kh[i, j] for i in ii for j in jj if i != j]
            return float(np.mean(vals)) if vals else 0.0

        wk, wl, wc = mblock(ii_k, ii_k), mblock(ii_l, ii_l), mblock(ii_k, ii_l)
        ntk[w] = {
            "gram_rank_at_1e-12": rank,
            "lam_min_over_max_reported": float(ev[-1] / lam1),
            "pinv_floor_sensitivity_max_rel_dev_1e-9_vs_1e-12": rel_dev,
            "support_span_capture": {
                "mean": float(np.mean(cap)), "median": float(np.median(cap)),
                "min": float(np.min(cap)), "max": float(np.max(cap)),
                "per_fact": {J["supports_fd"][i]["fact"]: float(cap[i])
                             for i in range(N_SUP)}},
            "blocks_inspan_cos": {"within_killed": wk, "within_living": wl,
                                  "cross": wc, "killed_over_living":
                                  float(wk / wl) if wl > 0 else None},
            "blocks_fullspace_e240_committed": E240_OVERLAP[w],
            "block_ratio_fullspace": (
                E240_OVERLAP[w]["killed"] / E240_OVERLAP[w]["living"]
                if E240_OVERLAP[w]["living"] else None),
        }
        log(f"NTK {w}: rank {rank} span-capture mean {np.mean(cap):.4f} "
            f"in-span K/L {wk:.4f}/{wl:.4f} ({wk/wl:.2f}x) vs full-space "
            f"{E240_OVERLAP[w]['killed']:.4f}/{E240_OVERLAP[w]['living']:.4f}")
    metrics["ntk_blocks"] = ntk
    write_metrics(metrics, "PARTIAL: cell 3 (the NTK blocks) done")

    # ---------------- CELL 4: the natural-gradient gap (co-report)
    # (a) exact coefficient-domain natural steps (split-half)
    # algebra (frozen in a comment to make it checkable): with G_A = V L V^T
    # the first-half true-unit Gram and b_j = sum_s V[s,j] g_s / sqrt(L_j)
    # the orthonormal parameter-space basis, a new g_t has coefficients
    # alpha_j = <g_t, b_j> = (V^T c)_j / sqrt(L_j)  (c = <g_s, g_t> column);
    # ||P_A g_t||^2 = sum alpha_j^2 = sum wv_j^2/L_j;
    # the natural step n_k = sum_{j<=k} alpha_j/L_j b_j gives
    # <g_t, n_k> = sum wv_j^2/L_j^2, ||n_k||^2 = sum wv_j^2/L_j^3,
    # cos(g_t, n_k) = <g,n> / sqrt(||n||^2 ||g_t||^2).
    nat = {}
    for w in WASHES:
        G = grams_true[w]
        gn = gnorms_all[w]
        GA = G[:HALF, :HALF]
        ew, VW = np.linalg.eigh(GA)
        order = np.argsort(ew)[::-1]
        ew, VW = ew[order], VW[:, order]
        lam_safe = np.maximum(ew, 1e-30)
        cos_by_k = {str(k): [] for k in (1, 2, 5, 10, 20, 40)}
        pc1_mass = []
        wind_list = []
        wind2_top2 = []
        for t in range(HALF, S80):
            c = G[:HALF, t]                        # <g_s, g_t>, s in A
            gt2 = float(gn[t] ** 2)
            wv = VW.T @ c                          # eigvec coefficients
            wv2 = wv ** 2
            wind2 = float(np.sum(wv2 / lam_safe))  # ||P_A g_t||^2
            wind_list.append(wind2 / gt2)
            # e234's convention co-read: the TOP-2 truncated wind share
            # (e234's committed kA=2 knee, G_DECOMP) — the like-for-like
            wind2_top2.append(float(np.sum(wv2[:2] / lam_safe[:2])) / gt2)
            gn_n2 = wv2 / lam_safe ** 3            # per-j parts of ||n_k||^2
            gn_num = wv2 / lam_safe ** 2           # per-j parts of <g, n_k>
            cs_n2 = np.cumsum(gn_n2)
            cs_num = np.cumsum(gn_num)
            for k in (1, 2, 5, 10, 20, 40):
                cos = float(cs_num[k - 1] /
                            np.sqrt(max(cs_n2[k - 1] * gt2, 1e-300)))
                cos_by_k[str(k)].append(cos)
            pc1_mass.append(float(gn_n2[0] / max(cs_n2[-1], 1e-300)))
        wind_med = float(np.median(wind_list))
        nat[w] = {
            "natural_step_def": "n_k = F_A^{+k} g_t with F_A the first-half "
                                "empirical Fisher, top-k eigen-truncation; "
                                "cos(g_t, n_k) exact in the coefficient "
                                "domain (see the algebra comment)",
            "cos_g_nk_median": {k: float(np.median(v))
                                for k, v in cos_by_k.items()},
            "cos_g_nk_q1_q3": {k: [float(np.percentile(v, 25)),
                                   float(np.percentile(v, 75))]
                               for k, v in cos_by_k.items()},
            "natural_step_pc1_mass_median": float(np.median(pc1_mass)),
            "wind_share_median_fullspan": wind_med,
            "wind_share_median_top2": float(np.median(wind2_top2)),
            "e234_committed_wind_share_median_A": E234_WIND[w],
            "wind_def_note": "e234's committed wind_share_median_A is the "
                             "TOP-2 truncated share (its kA=2 knee); the "
                             "full-40-span share is the new co-read here",
        }
    metrics["gates"]["G_WIND"] = {
        "check": "split-half TOP-2 wind share median vs e234's committed "
                 "wind_share_median_A (its kA=2 knee convention; recompute "
                 "identity)",
        "recomputed": {w: nat[w]["wind_share_median_top2"] for w in WASHES},
        "committed": E234_WIND,
        "tol": 0.005,
        "pass": bool(all(abs(nat[w]["wind_share_median_top2"] - E234_WIND[w])
                         < 0.005 for w in WASHES))}
    log(f"G_WIND (top2): "
        f"{ {w: round(nat[w]['wind_share_median_top2'], 5) for w in WASHES} }"
        f" vs committed {E234_WIND}; full-span "
        f"{ {w: round(nat[w]['wind_share_median_fullspan'], 4) for w in WASHES} }")

    # (b) Adam's applied direction through the 54-support window
    # exact top-eigenvector support profiles: prof_k[i] = <b_k, s_i>
    win = {}
    for w in WASHES:
        D = np.array(J[f"dmat_scaled_{w}"], dtype=np.float64) \
            * gnorms_all[w][:, None] / SQ
        G = grams_true[w]
        ew, VW = np.linalg.eigh(G)
        order = np.argsort(ew)[::-1]
        ew, VW = ew[order], VW[:, order]
        prof1 = (VW[:, 0] / np.sqrt(max(ew[0], 1e-30))) @ D   # (54,)
        prof2 = (VW[:, 1] / np.sqrt(max(ew[1], 1e-30))) @ D
        steps = src["e240_journal"][w]["steps"]
        cos_dg, cos_d_b1, cos_g_b1 = [], [], []
        for s in range(1, S80 + 1):
            cd = np.array(steps[str(s)]["cos_d"], dtype=np.float64)
            cg = np.array(steps[str(s)]["cos_g"], dtype=np.float64)

            def wcos(a, b):
                d = np.sqrt((a @ a) * (b @ b))
                return float(a @ b / d) if d > 0 else float("nan")

            cos_dg.append(wcos(cd, cg))
            cos_d_b1.append(wcos(cd, prof1))
            cos_g_b1.append(wcos(cg, prof1))
        arr = lambda x: np.array([v for v in x if np.isfinite(v)])
        a_dg, a_db1, a_gb1 = arr(cos_dg), arr(cos_d_b1), arr(cos_g_b1)
        # cross-check cos_g vs the e234 dmat (per-step, all 80)
        dev = 0.0
        for s in range(1, S80 + 1):
            cg = np.array(steps[str(s)]["cos_g"], dtype=np.float64)
            dev = max(dev, float(np.max(np.abs(
                D[s - 1] / gnorms_all[w][s - 1] - cg))))
        win[w] = {
            "window_def": "54-support window: cos profiles <x, s_i>; the "
                          "full-space angle(d_t, n_t) is NOT computable from "
                          "committed records (no per-coordinate v); these "
                          "are window reads, labeled as such",
            "cos_d_g_window": {"median": float(np.median(a_dg)),
                               "mean": float(np.mean(a_dg)),
                               "q1_q3": [float(np.percentile(a_dg, 25)),
                                         float(np.percentile(a_dg, 75))]},
            "cos_d_top1_window": {"median": float(np.median(a_db1)),
                                  "mean": float(np.mean(a_db1))},
            "cos_g_top1_window": {"median": float(np.median(a_gb1)),
                                  "mean": float(np.mean(a_gb1))},
            "prof1_norm": float(np.linalg.norm(prof1)),
            "prof2_norm": float(np.linalg.norm(prof2)),
            "cos_g_vs_dmat_dev_max": dev,
        }
        log(f"NGAP {w}: window cos(d,g) med {np.median(a_dg):.4f}; "
            f"cos(d,b1) med {np.median(a_db1):.4f} vs cos(g,b1) med "
            f"{np.median(a_gb1):.4f}")
    metrics["gates"]["G_COSG"] = {
        "check": "e240 committed cos_g vs e234 dmat cos(g_t, s_i) "
                 "(unit-normalized), max dev over all steps x probes",
        "max_dev": max(win[w]["cos_g_vs_dmat_dev_max"] for w in WASHES),
        "tol": 1e-4,
        "pass": bool(max(win[w]["cos_g_vs_dmat_dev_max"]
                         for w in WASHES) < 1e-4)}

    # e254's committed v-mass context (the diagonal-capture statement)
    t254 = src["e254_metrics"]["reads"]["tables"]
    metrics["natural_gradient_gap"] = {
        "split_half_exact": nat,
        "adam_window_reads": win,
        "e254_v_mass_context": {
            "what": "Adam's own denominator v = E[g^2] (the diag Fisher): "
                    "committed squared-L2 span-mass fractions on the top-k "
                    "first-half basis (e254, certified vs random nulls)",
            "tables": {f"{r['wash']}_s{r['state']}_k{r['k']}": r["span_mass"]
                       for r in t254},
            "range_k2_all_states_washes": [
                float(min(r["span_mass"] for r in t254 if r["k"] == 2)),
                float(max(r["span_mass"] for r in t254 if r["k"] == 2))]},
        "honesty": "the diagonal captures the span's SCALE (v-mass 42-85% at "
                   "k=2, e254) but the window reads show the applied "
                   "direction's alignment with the top eigenvector vs the "
                   "gradient's own — the directionality gap is the number "
                   "above; full-space angle not recoverable from committed "
                   "records",
    }
    write_metrics(metrics, "PARTIAL: cell 4 (the natural-gradient gap) done")

    # ---------------- plots
    RD.mkdir(parents=True, exist_ok=True)

    # P1: the spectrum decay, both scales
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    ax = axes[0]
    cols = {"w1": "#1f77b4", "w2": "#d62728", "w3": "#2ca02c"}
    for w in WASHES:
        ev = np.array(fisher[w]["spectrum"])
        ax.semilogy(np.arange(1, len(ev) + 1), ev / ev[0], "-o", ms=3,
                    color=cols[w], label=f"124M {w} (80-step window)")
        kk = fisher[w]["knee_loggap"]["k"]
        ax.axvline(kk, color=cols[w], ls=":", lw=1, alpha=0.7)
        ax.annotate(f"knee {kk}", (kk, fisher[w]["spectrum"][kk - 1] /
                    fisher[w]["spectrum"][0]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8, color=cols[w])
    ev2 = np.array(st_2m["spectrum"])
    ax.semilogy(np.arange(1, len(ev2) + 1), ev2 / ev2[0], "-s", ms=4,
                color="#9467bd", label="2.74M e231-J1 (20-step window)")
    kk = st_2m["knee_loggap"]["k"]
    ax.axvline(kk, color="#9467bd", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("eigenvalue index i")
    ax.set_ylabel(r"$\lambda_i / \lambda_1$ (log)")
    ax.set_title("the empirical Fisher's nonzero spectrum (top window)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for w in WASHES:
        ev = np.array(fisher[w]["spectrum"])
        cum = np.cumsum(ev) / ev.sum()
        ax.plot(np.arange(1, len(ev) + 1), 1 - cum, "-o", ms=3,
                color=cols[w], label=f"124M {w}")
    ev2 = np.array(st_2m["spectrum"])
    cum2 = np.cumsum(ev2) / ev2.sum()
    ax.plot(np.arange(1, len(ev2) + 1), 1 - cum2, "-s", ms=4,
            color="#9467bd", label="2.74M e231-J1")
    for tol, c in zip(MASS_TOL, ("#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f")):
        ax.axhline(tol, color=c, ls="--", lw=0.8, alpha=0.8)
        ax.annotate(f"erank tol {tol:g}", (1, tol), fontsize=7, color=c,
                    xytext=(2, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("k (eigenvalue index)")
    ax.set_ylabel("unexplained mass 1 - cumsum (log)")
    ax.set_title("explained-mass erank curves (window = lower bound when "
                 "curve does not cross the tol)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("E265 cell 1 — the Fisher spectrum census (committed "
                 "Grams; fp64)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RD / "e265_spectrum.png", dpi=130)
    plt.close(fig)
    log("plot: e265_spectrum.png")

    # P2: the cliff join
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    ax = axes[0]
    for w, c in zip(WASHES, ("#1f77b4", "#d62728", "#2ca02c")):
        ev = np.array(fisher[w]["spectrum"])
        cum = np.cumsum(ev) / ev.sum()
        ax.plot(np.arange(1, len(ev) + 1), cum, "-o", ms=3, color=c,
                label=f"124M {w} (window 80)")
    ev2 = np.array(st_2m["spectrum"])
    ax.plot(np.arange(1, 21), np.cumsum(ev2) / ev2.sum(), "-s", ms=5,
            color="#9467bd", label="2.74M e231-J1 (window 20)")
    ax.axvspan(BRACKET[0], BRACKET[1], color="#ffbb33", alpha=0.25,
               label=f"expression cliff bracket [{BRACKET[0]:,}, "
                     f"{BRACKET[1]:,}]")
    ax.axvline(BRACKET[0] / 3, color="#aa6600", ls="--", lw=1,
               label="factor-3 lower edge (333)")
    ax.set_xscale("log")
    ax.set_xlabel("dimension k (log)")
    ax.set_ylabel("cumulative explained Fisher mass")
    ax.set_title("the join: does the spectrum's kneel reach the bracket?")
    ax.set_xlim(0.7, 4e5)
    ax.axvline(80, color="#555", ls=":", lw=1)
    ax.annotate("124M window cap (80)", (80, 0.02), rotation=90, fontsize=7,
                color="#555", xytext=(4, 0), textcoords="offset points")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    names, poss, cols_, ann = [], [], [], []
    rows_ = [("2.74M knee (e231J1)", join["sides"]["2.74M_e231J1"]["knee"],
              "#9467bd", "k"),
             ("2.74M erank 1e-2", join["sides"]["2.74M_e231J1"]["erank"]["0.01"],
              "#9467bd", "o"),
             ("2.74M erank 1e-6 (cap)", join["sides"]["2.74M_e231J1"]["erank"]["1e-06"],
              "#9467bd", "s"),
             ("124M w1 knee", join["sides"]["124M_w1"]["knee"], "#1f77b4", "k"),
             ("124M w1 erank 1e-2", join["sides"]["124M_w1"]["erank"]["0.01"],
              "#1f77b4", "o"),
             ("124M w1 erank 1e-6 (cap)", join["sides"]["124M_w1"]["erank"]["1e-06"],
              "#1f77b4", "s")]
    for name, pos, c, mk in rows_:
        names.append(name)
        poss.append(max(pos, 0.5))
        cols_.append(c)
        ann.append(f"{pos}")
    yv = np.arange(len(names))
    ax.barh(yv, poss, color=cols_, alpha=0.8)
    for y, p, a in zip(yv, poss, ann):
        ax.annotate(a, (p, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.axvline(BRACKET[0], color="#aa6600", lw=2)
    ax.axvline(BRACKET[1], color="#aa6600", lw=2)
    ax.axvspan(BRACKET[0], BRACKET[1], color="#ffbb33", alpha=0.25)
    ax.set_yticks(yv, names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1.2e6)
    ax.set_xlabel("dimensions (log)")
    ax.set_title(f"the verdict: {verdict} — the head's knee sits "
                 f"{join['sides']['2.74M_e231J1']['knee_over_bracket_lo']:.0f}x"
                 "(2.74M) / "
                 f"{BRACKET[0]/max(fisher['w1']['knee_loggap']['k'],1):.0f}x"
                 "(124M) below the bracket's lower edge;\ndeep eranks are "
                 "window caps (lower bounds) — the tail is unmeasured")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("E265 cell 2 — the knee-vs-cliff join (both scales)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RD / "e265_cliff_join.png", dpi=130)
    plt.close(fig)
    log("plot: e265_cliff_join.png")

    # P3: the NTK blocks
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    w = "w1"
    G = grams_true[w]
    D = np.array(J[f"dmat_scaled_{w}"], dtype=np.float64) \
        * gnorms_all[w][:, None] / SQ
    ew, VW = np.linalg.eigh(G)
    order = np.argsort(ew)[::-1]
    ew, VW = ew[order], VW[:, order]
    keepm = ew > 1e-12 * ew[0]
    inv = np.where(keepm, 1.0 / np.maximum(ew, 1e-300), 0.0)
    Gpi = (VW * inv) @ VW.T
    K = D.T @ Gpi @ D
    cap = np.clip(np.diag(K), 0, None)
    Kh = np.divide(K, np.sqrt(np.outer(cap, cap)),
                   out=np.zeros_like(K), where=np.outer(cap, cap) > 1e-300)
    lab = fates[w]["labels"]
    idx = sorted(range(N_SUP), key=lambda i: (not lab[i], i))
    facts = [J["supports_fd"][i]["fact"] for i in idx]
    M = Kh[np.ix_(idx, idx)]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    nk = sum(1 for i in idx if lab[i])
    ax.axhline(nk - 0.5, color="k", lw=1.2)
    ax.axvline(nk - 0.5, color="k", lw=1.2)
    ax.set_title(f"w1 in-span NTK cross-probe cosine K̂ (killed first, "
                 f"n={nk}; full window 80)")
    ax.set_xlabel("probe j (killed | living)")
    ax.set_ylabel("probe i (killed | living)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    labels, kl, lv, cr, kf, lf, cf = [], [], [], [], [], [], []
    for w in WASHES:
        labels.append(w)
        kl.append(ntk[w]["blocks_inspan_cos"]["within_killed"])
        lv.append(ntk[w]["blocks_inspan_cos"]["within_living"])
        cr.append(ntk[w]["blocks_inspan_cos"]["cross"])
        kf.append(ntk[w]["blocks_fullspace_e240_committed"]["killed"])
        lf.append(ntk[w]["blocks_fullspace_e240_committed"]["living"])
        cf.append(ntk[w]["blocks_fullspace_e240_committed"]["cross"])
    x = np.arange(len(labels))
    wd = 0.2
    ax.bar(x - 1.5 * wd, kl, wd, color="#d62728", label="in-span killed")
    ax.bar(x - 0.5 * wd, cr, wd, color="#999", label="in-span cross")
    ax.bar(x + 0.5 * wd, lv, wd, color="#1f77b4", label="in-span living")
    ax.bar(x + 1.5 * wd, kf, wd, color="#d62728", alpha=0.35,
           label="full-space killed (e240)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean pairwise cosine")
    ax.set_title("the killed-vs-living block asymmetry:\nin-span (NTK "
                 "through the gradient basis) vs full-space (e240)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    bp = ax.boxplot([[ntk[w]["support_span_capture"]["per_fact"][f]
                      for f in ntk[w]["support_span_capture"]["per_fact"]]
                     for w in WASHES], tick_labels=list(WASHES),
                    showfliers=True)
    ax.set_ylabel("||P s_i||^2 (span-capture of each support)")
    ax.set_title("how much of each probe's Jacobian column\nthe wash "
                 "gradient span can see")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("E265 cell 3 — the NTK block structure (the supports' "
                 "cross-probe Gram through the gradient basis)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RD / "e265_ntk_blocks.png", dpi=130)
    plt.close(fig)
    log("plot: e265_ntk_blocks.png")

    # ---------------- adjudication + honesty
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": metrics["bars_verbatim"],
        "logic": join["verdict_logic"],
        "the_numbers": {
            "knee_2_74M": join["sides"]["2.74M_e231J1"]["knee"],
            "knee_124M": {w: fisher[w]["knee_loggap"]["k"] for w in WASHES},
            "bracket": list(BRACKET),
            "knee_below_bracket_lo_by": {
                "2.74M_x": join["sides"]["2.74M_e231J1"]["knee_over_bracket_lo"],
                "124M_w1_x": join["sides"]["124M_w1"]["knee_over_bracket_lo"]},
            "eranks": {s: join["sides"][s]["erank"] for s in join["sides"]},
        },
    }
    metrics["honesty_reflex"] = {
        "previsible_head": "the head of these spectra was already committed "
                           "(e234's half-split metaA spectra; e231's knee "
                           "k=6): this cell's new content is the FULL-80 "
                           "decomposition, the both-scale join "
                           "formalization, the NTK blocks, and the natural-"
                           "gap reads — not the top-of-spectrum shape",
        "spectrum_estimate": metrics["spectrum_estimate_caveat"],
        "logits_predict": "the join is a geometry-vs-capacity identity "
                          "check, not a behavioral prediction; the "
                          "intervention links live elsewhere and stand: "
                          "e237 (cutting the aligned component flips "
                          "fates), e254 (the span carries 42-85% of v at "
                          "k=2), e261 (the rank installs)",
        "no_inflation": "deep eranks reported as window caps (lower "
                        "bounds); the tail's extent is unmeasured and the "
                        "bracket is NOT excluded as the tail's scale — "
                        "stated in the verdict texture, not adjudicated",
    }
    metrics["plot_outputs"] = ["runs/e265/e265_spectrum.png",
                               "runs/e265/e265_cliff_join.png",
                               "runs/e265/e265_ntk_blocks.png"]
    metrics["builds_on"] = [
        "T240 (the connection map: this cell's registration)",
        "T239/e261 (the expression cliff bracket [1k, 237k] — the join's "
        "other side)",
        "e234's committed journal (the 80x80 wash Grams + dmat + traj — "
        "the primary record)",
        "e231 J1 + e246 (the 2.74M side's committed gradient-history "
        "spectra)",
        "e240 (the applied-direction records + the fates definition + the "
        "full-space overlaps)",
        "e254 (the v-mass span certification — the diagonal-Fisher "
        "context)",
        "e226/e228 (the supports = the Jacobian columns; the margins)",
    ]
    metrics["whats_new"] = [
        "the FIRST full nonzero-spectrum measurement of the empirical "
        "Fisher in this organism lineage (the 80x80 wash Grams "
        "eigen-decomposed; decay, eranks, PR, knee, cross-wash eigenspace "
        "alignment)",
        "the knee-vs-cliff join at BOTH scales against e261's bracket — "
        "the capacity-vs-optimization-geometry identity question answered "
        "on its frozen bars",
        "the NTK block structure formalized from the committed dmat (the "
        "54x54 cross-probe kernel through the gradient basis; killed/"
        "living asymmetry vs e240's full-space numbers)",
        "the natural-gradient gap co-report: exact split-half natural "
        "steps in the coefficient domain + Adam's applied direction read "
        "through the support window against the exact top-eigenvector "
        "profiles",
    ]
    all_pass = all(g.get("pass", False) for g in metrics["gates"].values())
    metrics["all_gates_pass"] = bool(all_pass)
    write_metrics(metrics, f"DONE (verdict {verdict}; all gates "
                           f"{'PASS' if all_pass else 'FAIL'})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
