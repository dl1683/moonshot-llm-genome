"""E256 — FQ14, THE SUPPORT-SPAN RESIDENCY CENSUS (the R64 ideator's Rank 4;
the interactions gap at its cheapest rung; the counterfeit-self's aiming data;
2026-10-04).

THE QUESTION (the dispatch brief, verbatim in substance): across all 54
probes, does the t=0 ALIGNMENT of a probe's support to the wash-span
predict its fate — the death order (P-FIRST/NEITHER/MARGIN-FIRST/
TOGETHER), the flip mode (mis-dial/collapse/other), the thermal share
(e238's per-battery R2 or the committed T-explained fractions)? Do the
tide and the undertow couple, or is e246's install-side decoupling the
death-side truth too?

BARS VERBATIM (the dispatch brief IS the registration — frozen BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  - RESIDENCY-PREDICTS — "the t=0 span-residency separates the death
    classes (a rank test at p <= 0.05 on at least one join, pooled washes,
    disclosed n per cell) — the tide feeds the undertow; W038's layers
    gain their first coupling constant; the counterfeit-self gains its
    target rationale"
  - NO-COUPLING — "no join separates (all p > 0.05) — the seat's pull is
    DYNAMIC (acquired during the wash), not standing geometry; e226's
    reading narrowed to a trajectory property; the layers decoupled end
    to end — a law-shaped negative"
  - MIXED — "the tables verbatim, all three joins, no inflation"

THE MATERIAL (all committed, read-only; NO replay, NO organism load, NO
GPU — pure fp64 Gram algebra on the committed journals):
  * runs/e234/journal.json — gram_scaled_{w} (80x80, unit-cos x 2^28),
    dmat_scaled_{w} (80x54, <g_hat_t, s_i> x 2^28), w.gnorms (80, fp64):
    THE committed geometry of the 54 t=0 supports against every step's
    raw gradient of all three washes (the supports/span caches were
    deleted on DONE by design; the journal IS their committed form).
  * runs/e234/metrics.json — the committed span decomposition (kA/kB,
    metaA spectra, cross-wash basis angles): the certification targets.
  * runs/e226/metrics.json — the committed t=0 support records (probes,
    p0, support_geometry_t0) + curves.alignment.w1.0 (the per-probe
    cos(g_w1_draw1, s_i) e234's dmat must reproduce).
  * runs/e230/metrics.json — the death orders, w1/w2, all 54 per wash
    (read1_population.probe_level).
  * runs/e247/metrics.json — the w3 order census (all 54) + the 11-row
    flip-mode table (MIS-DIAL / FREQUENCY-COLLAPSE / OTHER).
  * runs/e238/metrics.json — the thermal records: fit_rows (per-battery
    R2_battery per (wash,state) cell) + residual_structure.cells
    (per-probe resid_z, within-battery z of the one-T model's residual —
    e238's z-join convention, canonical battery order).

THE INSTRUMENT (frozen here BEFORE compute; fixes clauses, moves no bars):
  1. THE SPAN BASES, rebuilt from the committed journal exactly as e234
     built them (module-verified algebra, not re-derived): Gn = G_scaled
     x outer(gnorms)/2^28 (true-unit Gram); basis A = top-k eigenvectors
     of Gn[:40,:40] at e234's scree knee (kA = 2 on all three washes),
     orthonormalized by 1/sqrt(lambda); MIRROR basis B = same on
     Gn[40:,40:]; FULL-WASH basis F = same on the whole Gn (co-report).
     dmat_true = dmat_scaled x gnorms / 2^14 = 2^14 x <g_t, s_i>;
     M = c^T @ dmat_true[:40,:] = 2^14 x <b_j, s_i>; cos_j,i = M[j,i]/2^14
     (the supports are unit rows; the fp16 cache floor ~5e-4 per cos,
     e226's disclosed floor, rides every value).
  2. RESIDENCY (the per-probe scalar): residency_i(w) = sum_j
     cos^2(b_j^w, s_i) over the kA=2 basis vectors — the fraction of the
     support's squared norm living inside the wash's span (e234's
     wind-share convention). PRIMARY basis = split A (e234's committed
     primary); mirror B, full-wash F, and the normalized-Gram flavor are
     co-reported robustness flavors, never adjudicated.
  3. THE THREE JOINS (the fates read at runtime from the committed
     records above; n per cell disclosed everywhere):
       JOIN 1 (death order): Kruskal-Wallis across the four classes,
         POOLED washes (162 rows = 54 probes x 3 washes; the wash is a
         repeated measure, disclosed). Co-reports: per-wash KW; pooled +
         per-wash Mann-Whitney P-FIRST vs NEITHER (the two populated
         classes).
       JOIN 2 (flip mode): Kruskal-Wallis across MIS-DIAL /
         FREQUENCY-COLLAPSE / OTHER on the 11 w3 dead-flipped probes,
         w3 residency; exact-label permutation p (20k shuffles, seed
         256) co-reported.
       JOIN 3 (thermal share): PRIMARY = per-probe Spearman(residency_w,
         |resid_z|) pooled over e238's 16 committed residual cells
         (54 x w1/w2 x {50,80} = 216 rows; |resid_z| = thermal
         DEVIATION — small = the one-T fit explains the probe).
         Co-reports: per-(wash,state) Spearman; the signed-resid_z
         flavor; the COARSE battery-level read — Spearman(battery-mean
         residency, R2_battery) over e238's 28 committed fit cells
         (4 batteries x 7 (wash,state); residency repeated across a
         wash's states, disclosed).
  4. THE ANCHORS: Gmail/iPhone residency per wash, z vs the product
     family's 5 non-anchor probes (e239's band convention), all flavors.
  5. nearrel vs product AT t=0 RESIDENCY (e234's inverted finding
     re-checked at the standing-geometry rung): medians + ratio per wash.
  6. SCALE REFERENCES: the analytic random null E[residency] = k/P
     (P = 124,439,808); the fp16 per-cos floor ~5e-4.

ADJUDICATION (frozen; the letter of the bars decides):
  RESIDENCY-PREDICTS := >= 1 of the three PRIMARY tests at p <= 0.05
    (join 1 pooled KW; join 2 KW; join 3 pooled Spearman).
  NO-COUPLING := all three primaries p > 0.05.
  MIXED (the registered contingency, "the tables verbatim, all three
    joins, no inflation") := no primary fires BUT a registered co-report
    flavor of a join fires at p <= 0.05, OR two primaries fire with
    discordant direction — direction defined as sign(median P-FIRST -
    median NEITHER) for join 1 vs sign(median MIS-DIAL - median
    FREQUENCY-COLLAPSE) for join 2 (both are commitment-first-vs-
    belief-first contrasts; opposite signs = discordant).
  MULTIPLICITY: three joins (plus their registered co-report flavors),
    NO correction claimed — disclosed in the verdict block.
  SCOPE: n = 1 organism (GPT-2 124M), 3 washes; nothing guaranteed.

GATES: G_ENV (desk-only: no organism load, no GPU, no replay — committed
JSONs only), G_RECORDS (all inputs present, shapes (80,80)/(80,54)/(80)),
G_ORDER (the canonical 54: e226's probes == e234's probes, sets and
order; the fate records cover 54x3 orders + 11 modes + 16 thermal cells),
G_REPRO_SPECTRUM (rebuilt split-A spectra == e234's committed metaA
spectra, rel tol 1e-9; kA == committed kA), G_REPRO_ANGLES (rebuilt
cross-wash principal angles == e234's committed basis_angles_cos, abs
tol 1e-9), G_REPRO_ALIGN (dmat_scaled_w1[0]/2^28 vs e226's committed
w1:s0 alignment, tol 5e-3 = e226's cache floor).

Envelope: DESK-ONLY (the dispatch: e248 owns the GPU, e254 is CPU —
load-polite; this cell is numpy/scipy on committed JSONs, single BLAS
thread-pool, minutes). Progressive PARTIAL metrics. No NOTES/THINKING/
QUEUE/STATE edits (dispatch). Smoke via E256_SMOKE=1 (certifications +
join 1 on w1 only; nothing adjudicated).

PROVENANCE (extend-don't-repeat): builds on T217/e234 (the committed
span bases + the journal's Gram/dmat records — the tide), T209/e230 +
T224/e247 (the fate taxonomy: death orders + flip modes), T218/e238
(the thermal ledger's per-battery R2 + per-probe resid_z — the z-join
convention), T216/e239 (the 54 t=0 supports' committed geometry), T226/
e246 (the anti-substrate; the install-side decoupling this census tests
on the death side), T229/e255 (the gap-is-the-ledger context), the R64
ideator's FQ14 (scratch/review_r64_ideator.md — the spec). NEW: the
t=0 residency scalar itself (never computed — e234 measured DURING-wash
wind-reach, e226 measured support-vs-step-gradient alignment, nobody
has measured the STANDING overlap of each support with the wash span);
the three fate joins; the anchors'/families' residency callouts.

Run:  cd lab && python e256_residency_census.py     (E256_SMOKE=1 for
      the shakedown; nothing adjudicated or gated in smoke)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                   # noqa: E402

import common                                        # noqa: E402
from common import now_iso, run_dir, save_json       # noqa: E402

from scipy.stats import kruskal, mannwhitneyu, spearmanr   # noqa: E402

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

try:
    import psutil                                    # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E256_SMOKE") == "1"
NAME = "e256_smoke" if SMOKE else "e256"
HALF = 40                    # e234's split-half boundary
KNEE_MAX_I, K_CAP = 20, 12   # e234's knee constants (verbatim)
SCALE = float(2 ** 14)       # e226's CACHE_SCALE
SCALE2 = float(2 ** 28)
N_PARAMS = 124_439_808       # the 124M organism (e226's committed count)
PERM_N, PERM_SEED = 20_000, 256
WASHES = ("w1",) if SMOKE else ("w1", "w2", "w3")

T0 = time.time()
_log_t0 = time.time()
def log(m: str) -> None:
    print(f"[{time.time() - _log_t0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------ committed inputs
R = common.REPO
E234_J = R / "runs" / "e234" / "journal.json"
E234_M = R / "runs" / "e234" / "metrics.json"
E226_M = R / "runs" / "e226" / "metrics.json"
E230_M = R / "runs" / "e230" / "metrics.json"
E247_M = R / "runs" / "e247" / "metrics.json"
E238_M = R / "runs" / "e238" / "metrics.json"

REGISTERED = {
    "bars_verbatim": {
        "RESIDENCY-PREDICTS": "the t=0 span-residency separates the death "
        "classes (a rank test at p <= 0.05 on at least one join, pooled "
        "washes, disclosed n per cell) — the tide feeds the undertow; "
        "W038's layers gain their first coupling constant; the "
        "counterfeit-self gains its target rationale",
        "NO-COUPLING": "no join separates (all p > 0.05) — the seat's pull "
        "is DYNAMIC (acquired during the wash), not standing geometry; "
        "e226's reading narrowed to a trajectory property; the layers "
        "decoupled end to end — a law-shaped negative",
        "MIXED": "the tables verbatim, all three joins, no inflation",
    },
    "registration": "bars frozen VERBATIM from the dispatch brief (the "
    "coordinator's E256 task = the R64 ideator's FQ14 spec) BEFORE any "
    "compute; the dispatch brief is the registration; adjudicate against "
    "exactly this; no bar shopping",
}

deviations: list[str] = [
    "The span bases and support dots are NOT recomputed by replay (e248 "
    "owns the GPU; the desk-only mandate): they are rebuilt by fp64 Gram "
    "algebra from e234's COMMITTED journal records (gram_scaled/dmat_scaled/"
    "gnorms) — the supports/span caches were deleted on DONE by design and "
    "the journal is their committed form; certified against e234's "
    "committed spectra + cross-wash angles and e226's committed w1:s0 "
    "alignments (G_REPRO_SPECTRUM / G_REPRO_ANGLES / G_REPRO_ALIGN).",
    "The per-probe cos values inherit e226's fp16 cache floor (~5e-4 per "
    "cos, both sides rounded) — disclosed on every residency read; the "
    "rank tests operate on ranks, the floor enters as ties/noise.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E256_SMOKE=1): certifications + join 1 on w1 only; "
    "nothing adjudicated or gated (SMOKE stamp).",
]

# ------------------------------------------------------------ e234's algebra


def scree_knee(evals: np.ndarray, max_i: int = KNEE_MAX_I,
               cap: int = K_CAP) -> tuple[int, list[float]]:
    """k at the biggest absolute spectral drop in the head (e234's rule,
    module-verified against its committed knees)."""
    ev = np.asarray(evals, dtype=np.float64)
    gaps = [(float(ev[i - 1] - ev[i]), i + 1)
            for i in range(1, min(max_i, len(ev) - 1) + 1)]
    k = max(gaps)[1] if gaps else 1
    if k > cap:
        k = cap
    return k, [g for g, _ in gaps]


def basis_of(M: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Top-k orthonormalized Gram-eigenvector coefficients (e234's
    basis_of: eigvecs scaled by 1/sqrt(lambda) — columns give the basis
    vectors' coefficients over the step-gradients)."""
    ev, evec = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    k, _ = scree_knee(ev)
    c = evec[:, :k] / np.sqrt(np.clip(ev[:k], 1e-12, None))
    return k, c, ev


def principal_angles(cA: np.ndarray, GB_block: np.ndarray,
                     cB: np.ndarray) -> np.ndarray:
    """e234's principal_angles (module-verified): svd of cA^T G cB."""
    s = np.linalg.svd(cA.T @ GB_block @ cB, compute_uv=False)
    return np.clip(s, -1.0, 1.0)


def load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


# ----------------------------------------------------------------------- main

def main() -> int:
    rd = run_dir(NAME)
    log(f"E256 — FQ14 THE SUPPORT-SPAN RESIDENCY CENSUS (smoke={SMOKE}) "
        f"-> {rd}")

    metrics = {
        "experiment": "e256_residency_census",
        "phase": "FQ14: desk-only fp64 Gram algebra on e234's committed "
                 "journal (the supports x span geometry) joined at runtime "
                 "to the committed fate records (e230/e247 orders+modes, "
                 "e238 thermal); the three registered joins",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered_prediction": REGISTERED,
        "question": "does the t=0 span-residency of a probe's support "
                    "predict its fate (death order / flip mode / thermal "
                    "share) — do the tide and the undertow couple?",
        "smoke": SMOKE,
    }

    def write_metrics(status: str):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    load_checks = [load_check("launch")]
    metrics["load_checks"] = load_checks

    # ------------------------------------------------ P0 the committed records
    for p in (E234_J, E234_M, E226_M, E230_M, E247_M, E238_M):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            write_metrics("PARTIAL: missing committed record — HALT")
            return 1
    j234 = json.loads(E234_J.read_text(encoding="utf-8"))
    m234 = json.loads(E234_M.read_text(encoding="utf-8"))
    m226 = json.loads(E226_M.read_text(encoding="utf-8"))
    m230 = json.loads(E230_M.read_text(encoding="utf-8"))
    m247 = json.loads(E247_M.read_text(encoding="utf-8"))
    m238 = json.loads(E238_M.read_text(encoding="utf-8"))
    metrics["provenance_records"] = {
        "span+support geometry (Gram/dmat/gnorms)": str(E234_J),
        "span decomposition cert targets": str(E234_M),
        "supports t0 + w1:s0 alignments": str(E226_M),
        "death orders w1/w2": str(E230_M),
        "w3 order census + flip modes": str(E247_M),
        "thermal (R2_battery + resid_z)": str(E238_M),
    }

    G_REC = {}
    grams, dmats, gnorms = {}, {}, {}
    for w in WASHES:
        G = np.array(j234[f"gram_scaled_{w}"], dtype=np.float64)
        D = np.array(j234[f"dmat_scaled_{w}"], dtype=np.float64)
        gn = np.array(j234[w]["gnorms"], dtype=np.float64)
        G_REC[w] = {"gram_shape": list(G.shape), "dmat_shape": list(D.shape),
                    "gnorms_n": int(gn.shape[0])}
        grams[w], dmats[w], gnorms[w] = G, D, gn
    G_REC["pass"] = bool(all(G_REC[w]["gram_shape"] == [80, 80]
                             and G_REC[w]["dmat_shape"] == [80, 54]
                             and G_REC[w]["gnorms_n"] == 80
                             for w in WASHES))
    metrics["gates"] = {"G_ENV": {
        "desk_only": True, "organism_loaded": False, "gpu_used": False,
        "replay": False, "note": "committed JSONs + fp64 numpy only; "
        "BLAS threads capped 4; e248 owns the GPU, e254 is CPU "
        "(load-polite dispatch)", "pass": True},
        "G_RECORDS": G_REC}
    log(f"G_RECORDS: {'PASS' if G_REC['pass'] else 'FAIL'}")
    if not G_REC["pass"]:
        write_metrics("PARTIAL: G_RECORDS FAILED — HALT")
        return 1

    # ------------------------------------------------------- P1 the canonical 54
    probes226 = [(p["fact"], p["battery"], p["family6"], p["p0"])
                 for p in m226["probes"]]
    probes234 = [(p["fact"], p["battery"], p["family6"])
                 for p in m234["probes"]]
    names = [f for f, *_ in probes226]
    fam6 = {f: fam for f, _b, fam, _p in probes226}
    battery_of = {f: b for f, b, _fam, _p in probes226}
    p0_of = {f: p for f, _b, _fam, p in probes226}
    G_ORDER = {
        "rule": "e226's canonical 54 == e234's canonical 54 (facts, order, "
                "batteries); the fate records key by fact name",
        "equal_e234": bool([n for n, *_ in probes234] == names
                           and all(bw == bo for (_f, bw, _g), (_f2, _bo, _g2)
                                   in zip(probes234, probes226))),
        "n": len(names),
        "pass": bool([n for n, *_ in probes234] == names and len(names) == 54),
    }
    metrics["gates"]["G_ORDER"] = G_ORDER
    log(f"G_ORDER: {'PASS' if G_ORDER['pass'] else 'FAIL'} (n={len(names)})")
    if not G_ORDER["pass"]:
        write_metrics("PARTIAL: G_ORDER FAILED — HALT")
        return 1
    idx_of = {f: i for i, f in enumerate(names)}

    # ---------------------------------------------- P2 the fate records (runtime)
    order_class: dict[tuple[str, str], str] = {}          # (wash, fact) -> class
    for b, per in m230["read1_population"]["probe_level"].items():
        for w, rows in per.items():
            for r in rows:
                order_class[(w, r["probe"])] = r["class"]
    for key, rec in m247["w3_order_census_all54"]["probe_level"].items():
        order_class[("w3", key.split("|", 1)[1])] = rec["class"]
    mode_rows = m247["table"]                              # the 11 flip modes
    thermal_cells = m238["residual_structure"]["cells"]    # 16 (wash,state,batt)
    fit_rows = m238["fit_rows"]                            # 7 R2_battery cells
    G_FATE = {
        "order_rows": len(order_class),
        "order_classes_vocab": sorted({v for v in order_class.values()}),
        "mode_rows": len(mode_rows),
        "mode_vocab": sorted({r["classification"] for r in mode_rows}),
        "thermal_resid_cells": len(thermal_cells),
        "thermal_fit_cells": len(fit_rows),
        "orders_cover_all54_x3": bool(all(
            sum(1 for (w, f), c in order_class.items() if f == f_) == 3
            for f_ in names)),
        "pass": bool(len(order_class) == 162 and len(mode_rows) == 11
                     and len(thermal_cells) == 16),
    }
    metrics["gates"]["G_FATE"] = G_FATE
    log(f"G_FATE: {'PASS' if G_FATE['pass'] else 'FAIL'} "
        f"(orders {G_FATE['order_rows']}; modes {G_FATE['mode_rows']}; "
        f"thermal cells {G_FATE['thermal_resid_cells']})")
    if not G_FATE["pass"]:
        write_metrics("PARTIAL: G_FATE FAILED — HALT")
        return 1
    write_metrics("PARTIAL: records + fates loaded and gated")

    # ------------------------------------- P3 the span bases + residency (fp64)
    load_checks.append(load_check("span algebra"))
    bases, res_flavors = {}, {}
    for w in WASHES:
        Gn = grams[w] * np.outer(gnorms[w], gnorms[w]) / SCALE2
        dmat_true = dmats[w] * gnorms[w][:, None] / SCALE   # 2^14 <g_t, s_i>
        # split A (PRIMARY): first half builds
        kA, cA, evA = basis_of(Gn[:HALF, :HALF])
        M_A = cA.T @ dmat_true[:HALF, :]                    # 2^14 <b_j, s_i>
        cos_A = M_A / SCALE
        res_A = np.sum(cos_A ** 2, axis=0)
        # mirror split B: second half builds
        kB, cB, evB = basis_of(Gn[HALF:, HALF:])
        M_B = cB.T @ dmat_true[HALF:, :]
        cos_B = M_B / SCALE
        res_B = np.sum(cos_B ** 2, axis=0)
        # full-wash basis F
        kF, cF, evF = basis_of(Gn)
        M_F = cF.T @ dmat_true
        cos_F = M_F / SCALE
        res_F = np.sum(cos_F ** 2, axis=0)
        # normalized-Gram flavor (co-report)
        dgn = np.sqrt(np.maximum(np.diag(Gn), 1e-30))
        Gn_norm = Gn / np.outer(dgn, dgn)
        kAn, cAn, _evAn = basis_of(Gn_norm[:HALF, :HALF])
        dm_unit = dmats[w] / SCALE2                          # <g_hat, s_hat>
        M_An = cAn.T @ dm_unit[:HALF, :]
        res_An = np.sum(M_An ** 2, axis=0)
        bases[w] = {"kA": kA, "kB": kB, "kF": kF, "kAn": kAn,
                    "cA": cA, "cB": cB, "cF": cF,
                    "evA": evA, "evB": evB, "evF": evF}
        res_flavors[w] = {"A_primary": res_A, "B_mirror": res_B,
                          "F_fullwash": res_F, "A_normgram": res_An,
                          "cos_A": cos_A, "cos_B": cos_B, "cos_F": cos_F}
        log(f"  {w}: kA={kA} kB={kB} kF={kF} kAn={kAn}; residency[A] "
            f"median {np.median(res_A):.3e} (IQR "
            f"{np.percentile(res_A, 25):.3e}..{np.percentile(res_A, 75):.3e})")

    # ------------------------------------------ P4 the G_REPRO certifications
    G_SPEC = {"tol_rel": 1e-9, "per_wash": {}}
    for w in WASHES:
        committed = m234["decomposition"][w]["metaA"]["spectrum"]
        mine = bases[w]["evA"][:len(committed)].tolist()
        rel = max(abs(a - b) / max(abs(b), 1e-30)
                  for a, b in zip(mine, committed))
        G_SPEC["per_wash"][w] = {
            "kA_rebuilt": bases[w]["kA"],
            "kA_committed": m234["decomposition"][w]["kA"],
            "max_rel_dev": rel}
    G_SPEC["pass"] = bool(all(
        r["kA_rebuilt"] == r["kA_committed"] and r["max_rel_dev"] <= 1e-9
        for r in G_SPEC["per_wash"].values()) and not SMOKE)
    metrics["gates"]["G_REPRO_SPECTRUM"] = G_SPEC
    log(f"G_REPRO_SPECTRUM: {'PASS' if G_SPEC['pass'] else 'FAIL'} "
        + " ".join(f"{w}: dev {r['max_rel_dev']:.1e}" for w, r
                   in G_SPEC["per_wash"].items()))

    G_ANG = {"tol_abs": 1e-9, "pairs": {}}
    if len(WASHES) == 3:
        for a, b in (("w1", "w2"), ("w1", "w3"), ("w2", "w3")):
            cross = np.array(j234[f"cross_{a}_{b}"], dtype=np.float64)
            ga, gb = gnorms[a][:HALF], gnorms[b][:HALF]
            X = cross[:HALF, :HALF] * np.outer(ga, gb) / SCALE2
            mine = principal_angles(bases[a]["cA"], X, bases[b]["cA"])
            committed = m234["cross_wash"]["basis_angles_cos"][f"{a}vs{b}"]
            dev = max(abs(x - c) for x, c in zip(mine, committed))
            G_ANG["pairs"][f"{a}vs{b}"] = {"max_abs_dev": dev,
                                           "rebuilt": mine[:2].tolist(),
                                           "committed": committed[:2]}
        G_ANG["pass"] = bool(all(r["max_abs_dev"] <= 1e-9
                                 for r in G_ANG["pairs"].values()))
    else:
        G_ANG["pass"] = True   # smoke: single wash, no pairs (SMOKE stamp)
    metrics["gates"]["G_REPRO_ANGLES"] = G_ANG
    log(f"G_REPRO_ANGLES: {'PASS' if G_ANG['pass'] else 'FAIL'}")

    a0_226 = m226["curves"]["alignment"]["w1"]["0"]
    d0 = dmats["w1"][0] / SCALE2
    dp_align = max(abs(d0[idx_of[f]] - v) for f, v in a0_226.items())
    G_AL = {"rule": "dmat_scaled_w1[0]/2^28 == e226's committed cos(g_w1s0, "
                    "s_i) per probe (certifies the journal's dmat IS the "
                    "committed supports' dots)",
            "max_dp": float(dp_align), "tol": 5e-3,
            "pass": bool(dp_align <= 5e-3)}
    metrics["gates"]["G_REPRO_ALIGN"] = G_AL
    log(f"G_REPRO_ALIGN: max dp {dp_align:.2e} -> "
        f"{'PASS' if G_AL['pass'] else 'FAIL'}")

    all_gates = [metrics["gates"][g]["pass"] for g in metrics["gates"]]
    metrics["all_gates_pass"] = bool(all(all_gates)) and not SMOKE
    if not all(all_gates) and not SMOKE:
        write_metrics("PARTIAL: a G_REPRO gate FAILED — HALT (no joins)")
        return 1
    write_metrics("PARTIAL: span bases rebuilt + certified")

    # ------------------------------------------------------- P5 the census rows
    load_checks.append(load_check("joins"))
    census = []       # per (wash, probe): residency flavors + fate
    for w in WASHES:
        for i, f in enumerate(names):
            census.append({
                "wash": w, "fact": f, "battery": battery_of[f],
                "family6": fam6[f], "i": i,
                "residency": float(res_flavors[w]["A_primary"][i]),
                "residency_B": float(res_flavors[w]["B_mirror"][i]),
                "residency_F": float(res_flavors[w]["F_fullwash"][i]),
                "residency_norm": float(res_flavors[w]["A_normgram"][i]),
                "cos_pc1": float(res_flavors[w]["cos_A"][0, i]),
                "cos_pc2": float(res_flavors[w]["cos_A"][1, i])
                if bases[w]["kA"] > 1 else None,
                "order_class": order_class.get((w, f)),
            })
    journal = {
        "census": census,
        "bases_meta": {w: {"kA": bases[w]["kA"], "kB": bases[w]["kB"],
                           "kF": bases[w]["kF"],
                           "evA_head": bases[w]["evA"][:6].tolist(),
                           "evF_head": bases[w]["evF"][:6].tolist()}
                       for w in WASHES},
    }

    # JOIN 1 — death order
    j1 = {"statistic": "Kruskal-Wallis across order classes, pooled washes",
          "classes": ["P-FIRST", "NEITHER", "MARGIN-FIRST", "TOGETHER"]}
    pooled = census if not SMOKE else [r for r in census if r["wash"] == "w1"]
    groups = [[r["residency"] for r in pooled if r["order_class"] == c]
              for c in j1["classes"]]
    j1["n_per_cell"] = {c: len(g) for c, g in zip(j1["classes"], groups)}
    j1["median_per_cell"] = {c: (float(np.median(g)) if g else None)
                             for c, g in zip(j1["classes"], groups)}
    if all(len(g) > 0 for g in groups):
        H, p = kruskal(*groups)
        j1["H"] = float(H); j1["p"] = float(p)
        n_tot = sum(len(g) for g in groups)
        j1["epsilon_sq"] = float((H - len(groups) + 1) / (n_tot - len(groups)))
    else:
        j1["H"] = j1["p"] = None
        j1["note"] = "empty class cell — KW undefined"
    j1["per_wash"] = {}
    for w in WASHES:
        rw = [r for r in census if r["wash"] == w]
        gw = [[r["residency"] for r in rw if r["order_class"] == c]
              for c in j1["classes"]]
        if all(len(g) > 0 for g in gw):
            Hw, pw = kruskal(*gw)
            j1["per_wash"][w] = {"H": float(Hw), "p": float(pw),
                                 "n": {c: len(g) for c, g
                                       in zip(j1["classes"], gw)}}
    # Mann-Whitney P-FIRST vs NEITHER (pooled + per wash)
    pf = [r["residency"] for r in pooled if r["order_class"] == "P-FIRST"]
    ne = [r["residency"] for r in pooled if r["order_class"] == "NEITHER"]
    if pf and ne:
        U, pmw = mannwhitneyu(pf, ne, alternative="two-sided")
        j1["mw_pfirst_vs_neither"] = {
            "U": float(U), "p": float(pmw),
            "median_pfirst": float(np.median(pf)),
            "median_neither": float(np.median(ne)),
            "direction": ("P-FIRST higher" if np.median(pf) > np.median(ne)
                          else "NEITHER higher")}
    j1["flavors_pooled_kw"] = {}
    for fl in ("residency_B", "residency_F", "residency_norm"):
        gfl = [[r[fl] for r in pooled if r["order_class"] == c]
               for c in j1["classes"]]
        if all(len(g) > 0 for g in gfl):
            Hf, pf_ = kruskal(*gfl)
            j1["flavors_pooled_kw"][fl] = {"H": float(Hf), "p": float(pf_)}
    metrics["join1_death_order"] = j1

    # JOIN 2 — flip mode (w3, n=11)
    j2 = {"statistic": "Kruskal-Wallis across flip-mode classes (e247's "
                       "committed 11-row table, w3)",
          "classes": ["MIS-DIAL", "FREQUENCY-COLLAPSE", "OTHER"]}
    mode_res = {c: [] for c in j2["classes"]}
    mode_facts = {c: [] for c in j2["classes"]}
    if "w3" in WASHES:
        for r in mode_rows:
            v = float(res_flavors["w3"]["A_primary"][idx_of[r["probe"]]])
            mode_res[r["classification"]].append(v)
            mode_facts[r["classification"]].append(r["probe"])
    j2["n_per_cell"] = {c: len(v) for c, v in mode_res.items()}
    j2["median_per_cell"] = {c: (float(np.median(v)) if v else None)
                             for c, v in mode_res.items()}
    j2["facts_per_cell"] = mode_facts
    groups2 = [v for v in mode_res.values() if v]
    if len(groups2) >= 2 and all(len(v) > 0 for v in groups2):
        H2, p2 = kruskal(*groups2)
        j2["H"] = float(H2); j2["p"] = float(p2)
        # exact-label permutation co-report (20k, seed 256): permuted H
        # counted against the OBSERVED H (the statistic, not the p)
        allv = np.array([x for v in mode_res.values() for x in v])
        labels = np.array([c for c, v in mode_res.items() for _ in v])
        rng = np.random.default_rng(PERM_SEED)
        cnt = 0
        uniq = sorted(set(labels))
        for _ in range(PERM_N):
            perm = rng.permutation(len(allv))
            gl = [allv[perm][labels[perm] == c] for c in uniq]
            try:
                Hh, _pp = kruskal(*gl)
            except ValueError:
                Hh = -np.inf
            if not np.isnan(Hh) and Hh >= H2:
                cnt += 1
        j2["perm_p"] = float((cnt + 1) / (PERM_N + 1))
    else:
        j2["H"] = j2["p"] = None
    if "w3" in WASHES:
        for fl, fl_tag in (("residency_B", "B_mirror"), ("residency_F",
                                                         "F_fullwash")):
            gfl = [np.array([float(res_flavors["w3"][fl_tag][idx_of[f_]])
                             for f_ in mode_facts[c]]) for c in j2["classes"]]
            if all(len(v) > 0 for v in gfl):
                Hf2, pf2 = kruskal(*gfl)
                j2.setdefault("flavors_kw", {})[fl] = {"H": float(Hf2),
                                                       "p": float(pf2)}
    metrics["join2_flip_mode"] = j2

    # JOIN 3 — thermal share
    j3 = {"statistic": "Spearman(residency_w, |resid_z|) pooled over e238's "
                       "16 committed residual cells (per-probe, w1/w2 x "
                       "{50,80}); |resid_z| = thermal deviation",
          "coarse_statistic": "Spearman(battery-mean residency, R2_battery) "
                              "over the 28 committed fit cells"}
    rows3 = []
    for cell in thermal_cells:
        w, s_ = cell["wash"], cell["state"]
        if w not in WASHES:
            continue
        b_names = [f for f in names if battery_of[f] == cell["battery"]]
        for f, rz in zip(b_names, cell["resid_z"]):
            rows3.append({"wash": w, "state": s_, "fact": f,
                          "residency": float(
                              res_flavors[w]["A_primary"][idx_of[f]]),
                          "abs_resid_z": abs(rz), "resid_z": rz})
    j3["n_rows"] = len(rows3)
    if len(rows3) > 3:
        rho3, p3 = spearmanr([r["residency"] for r in rows3],
                             [r["abs_resid_z"] for r in rows3])
        j3["rho"] = float(rho3); j3["p"] = float(p3)
        rho3s, p3s = spearmanr([r["residency"] for r in rows3],
                               [r["resid_z"] for r in rows3])
        j3["rho_signed"] = float(rho3s); j3["p_signed"] = float(p3s)
        j3["per_cell"] = {}
        for w in WASHES:
            for s_ in (50, 80):
                rc = [r for r in rows3 if r["wash"] == w and r["state"] == s_]
                if len(rc) > 3:
                    rr, pp = spearmanr([r["residency"] for r in rc],
                                       [r["abs_resid_z"] for r in rc])
                    j3["per_cell"][f"{w}+{s_}"] = {"rho": float(rr),
                                                    "p": float(pp),
                                                    "n": len(rc)}
        for fl_key, fl_tag in (("residency_B", "B_mirror"),
                               ("residency_F", "F_fullwash")):
            fl_rows = []
            for cell in thermal_cells:
                w = cell["wash"]
                if w not in WASHES:
                    continue
                b_names = [f for f in names if battery_of[f] == cell["battery"]]
                for f, rz in zip(b_names, cell["resid_z"]):
                    fl_rows.append((float(res_flavors[w][fl_tag][idx_of[f]]),
                                    abs(rz)))
            rr, pp = spearmanr([a for a, _ in fl_rows],
                               [b for _, b in fl_rows])
            j3.setdefault("flavors_pooled", {})[fl_key] = {
                "rho": float(rr), "p": float(pp)}
    # the coarse battery-level read
    coarse = []
    for fr in fit_rows:
        w = fr["wash"]
        if w not in WASHES:
            continue
        for b, r2 in fr["R2_battery"].items():
            bmean = float(np.median([res_flavors[w]["A_primary"][idx_of[f]]
                                     for f in names if battery_of[f] == b]))
            coarse.append({"wash": w, "state": fr["state"], "battery": b,
                           "R2": r2, "battery_median_residency": bmean})
    j3["coarse_n"] = len(coarse)
    if len(coarse) > 3:
        rc3, pc3 = spearmanr([c["battery_median_residency"] for c in coarse],
                             [c["R2"] for c in coarse])
        j3["coarse_rho"] = float(rc3); j3["coarse_p"] = float(pc3)
    metrics["join3_thermal"] = j3

    # ------------------------------------------------ P6 the anchors + families
    anchors = {}
    prod_band = [f for f in names if fam6[f] == "product"
                 and f not in ("The email service made by Google->Gmail",
                               "The phone made by Apple->iPhone")]
    for w in WASHES:
        band = np.array([res_flavors[w]["A_primary"][idx_of[f]]
                         for f in prod_band])
        mu, sd = float(band.mean()), float(band.std(ddof=1))
        anchors[w] = {}
        for tag, f in (("Gmail", "The email service made by Google->Gmail"),
                       ("iPhone", "The phone made by Apple->iPhone")):
            v = float(res_flavors[w]["A_primary"][idx_of[f]])
            anchors[w][tag] = {"residency": v,
                               "band_mean": mu, "band_sd": sd,
                               "z_vs_band": (v - mu) / sd if sd > 0 else None}
    metrics["anchors"] = anchors

    families = {}
    for w in WASHES:
        med = {}
        for fam in ("near-uscap", "product", "cap-cur", "lang",
                    "founder-anchor", "rev-capital"):
            v = [float(res_flavors[w]["A_primary"][idx_of[f]])
                 for f in names if fam6[f] == fam]
            med[fam] = {"n": len(v), "median": float(np.median(v))}
        fams = {
            "medians": med,
            "nearrel_over_product_ratio": (
                med["near-uscap"]["median"] / med["product"]["median"]
                if med["product"]["median"] > 0 else None),
            "e234_reference": {"wind_alignment_ratio": 0.7028083665778966,
                               "finding": "INVERTED at the during-wash rung "
                               "(the dying family the least wind-reached)"},
        }
        families[w] = fams
    metrics["families"] = families
    metrics["scale_references"] = {
        "analytic_random_null_residency": 2.0 / N_PARAMS,
        "fp16_per_cos_floor": 5e-4,
        "note": "E[sum cos^2] for a random unit vector in the 124M space "
                "against a k=2 basis; every measured value sits orders "
                "above it; per-cos instrument floor from e226's cache "
                "(both sides fp16-rounded)",
    }

    # ------------------------------------------------------- P7 adjudication
    prim = {
        "join1_kw_pooled": j1.get("p"),
        "join2_kw": j2.get("p"),
        "join3_spearman_pooled": j3.get("p"),
    }
    fired = [k for k, v in prim.items() if v is not None and v <= 0.05]
    co_flavors = []
    for k, v in j1.get("flavors_pooled_kw", {}).items():
        co_flavors.append((f"join1:{k}", v["p"]))
    for k, v in j2.get("flavors_kw", {}).items():
        co_flavors.append((f"join2:{k}", v["p"]))
    for k, v in j3.get("flavors_pooled", {}).items():
        co_flavors.append((f"join3:{k}", v["p"]))
    if j3.get("coarse_p") is not None:
        co_flavors.append(("join3:coarse_R2", j3["coarse_p"]))
    co_fired = [k for k, p_ in co_flavors if p_ is not None and p_ <= 0.05]
    d1 = None
    if j1.get("median_per_cell", {}).get("P-FIRST") is not None \
            and j1["median_per_cell"].get("NEITHER") is not None:
        d1 = (j1["median_per_cell"]["P-FIRST"]
              - j1["median_per_cell"]["NEITHER"])
    d2 = None
    if j2.get("median_per_cell", {}).get("MIS-DIAL") is not None \
            and j2["median_per_cell"].get("FREQUENCY-COLLAPSE") is not None:
        d2 = (j2["median_per_cell"]["MIS-DIAL"]
              - j2["median_per_cell"]["FREQUENCY-COLLAPSE"])
    discordant = bool(len(fired) >= 2 and d1 is not None and d2 is not None
                      and np.sign(d1) != np.sign(d2) and d1 != 0 and d2 != 0)
    if not SMOKE:
        if len(fired) >= 1 and not discordant:
            verdict = "RESIDENCY-PREDICTS"
        elif len(fired) == 0 and len(co_fired) >= 1:
            verdict = "MIXED"
        elif discordant:
            verdict = "MIXED"
        else:
            verdict = "NO-COUPLING"
    else:
        verdict = "SMOKE (nothing adjudicated)"
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "primaries": prim,
        "primaries_fired": fired,
        "coreport_flavors": dict(co_flavors),
        "coreport_flavors_fired": co_fired,
        "direction_join1_pfirst_minus_neither": d1,
        "direction_join2_misdial_minus_collapse": d2,
        "discordant": discordant,
        "rule_applied": {
            "RESIDENCY-PREDICTS": ">=1 primary at p<=0.05 and not "
                                  "discordant",
            "NO-COUPLING": "all primaries p>0.05 and no co-report flavor "
                           "fired",
            "MIXED": "no primary fired but a registered flavor did, OR two "
                     "primaries fired with discordant direction (both "
                     "commitment-vs-belief contrasts, opposite signs)",
        },
        "multiplicity": "three joins + registered co-report flavors, NO "
                        "correction claimed (disclosed)",
        "scope": "n=1 organism (GPT-2 124M), 3 washes; the wash is a "
                 "repeated measure in pooled tests (disclosed); nothing "
                 "guaranteed",
        "gated_on": "G_ENV/G_RECORDS/G_ORDER/G_FATE/G_REPRO_SPECTRUM/"
                    "G_REPRO_ANGLES/G_REPRO_ALIGN",
        "all_gates_pass": metrics["all_gates_pass"],
    }

    # ------------------------------------------------------------- P8 the plots
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    cls_order = ["P-FIRST", "NEITHER", "MARGIN-FIRST", "TOGETHER"]
    cls_col = {"P-FIRST": "#c0392b", "NEITHER": "#2471a3",
               "MARGIN-FIRST": "#8e44ad", "TOGETHER": "#7d6608"}
    ax = axes[0][0]
    data = [[r["residency"] for r in pooled if r["order_class"] == c]
            for c in cls_order]
    keep = [(c, d) for c, d in zip(cls_order, data) if d]
    bp = ax.boxplot([d for _c, d in keep], showfliers=False, widths=0.55,
                    patch_artist=True)
    for patch, (c, _d) in zip(bp["boxes"], keep):
        patch.set_facecolor(cls_col[c]); patch.set_alpha(0.45)
    for jk, (c, d) in enumerate(keep):
        ax.scatter(np.full(len(d), jk + 1) + np.random.default_rng(7 + jk)
                   .uniform(-0.13, 0.13, len(d)), d, s=14, alpha=0.6,
                   color=cls_col[c], zorder=3)
    ax.set_xticklabels([f"{c}\nn={len(d)}" for c, d in keep], fontsize=8)
    ax.set_ylabel("t=0 span residency  $\\Sigma_j \\cos^2(b_j, s_i)$")
    ax.set_title(f"JOIN 1 — death order, pooled washes (n={len(pooled)})\n"
                 f"Kruskal-Wallis H={j1.get('H'):.2f} p={j1.get('p'):.4f}"
                 if j1.get("H") is not None else "JOIN 1 (KW undefined)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0][1]
    mk = ([(float(res_flavors["w3"]["A_primary"][idx_of[r["probe"]]]),
            r["classification"], r["probe"]) for r in mode_rows]
          if "w3" in WASHES else [])
    mode_col = {"MIS-DIAL": "#c0392b", "FREQUENCY-COLLAPSE": "#1a6faf",
                "OTHER": "#7f8c8d"}
    for c, col in mode_col.items():
        v = [x for x, cl, _f in mk if cl == c]
        ax.scatter(np.full(len(v), list(mode_col).index(c) + 1)
                   + np.random.default_rng(11 + list(mode_col).index(c))
                   .uniform(-0.1, 0.1, len(v)), v, s=42, alpha=0.8,
                   color=col, label=f"{c} (n={len(v)})")
    ax.set_xticks([1, 2, 3], list(mode_col), fontsize=8)
    ax.legend(fontsize=7.5)
    ax.set_title(f"JOIN 2 — flip mode, w3 dead-flipped (n={len(mk)})\n"
                 f"KW H={j2.get('H'):.2f} p={j2.get('p'):.4f} "
                 f"(perm {j2.get('perm_p'):.3f})" if j2.get("H") is not None
                 else "JOIN 2 (w3 absent in smoke)", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0][2]
    if rows3:
        for b, col in zip(("fact", "ctrl", "near", "tmpl"),
                          ("#8e44ad", "#0d5c3f", "#1a6faf", "#b7950b")):
            xs = [r["residency"] for r in rows3 if battery_of[r["fact"]] == b]
            ys = [r["abs_resid_z"] for r in rows3 if battery_of[r["fact"]] == b]
            ax.scatter(xs, ys, s=12, alpha=0.55, color=col, label=b)
        ax.set_xlabel("t=0 span residency")
        ax.set_ylabel("|resid_z| (thermal deviation, e238)")
        ax.set_title(f"JOIN 3 — thermal deviation, pooled cells "
                     f"(n={len(rows3)})\nSpearman rho={j3.get('rho'):.3f} "
                     f"p={j3.get('p'):.4f}", fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)

    ax = axes[1][0]
    for w, mk_, col in zip(WASHES, ("o", "s", "^"),
                           ("#c0392b", "#2471a3", "#7f8c8d")):
        xs = [anchors[w]["Gmail"]["residency"],
              anchors[w]["iPhone"]["residency"]]
        ys = [anchors[w]["Gmail"]["z_vs_band"],
              anchors[w]["iPhone"]["z_vs_band"]]
        ax.scatter(xs, ys, marker=mk_, s=60, color=col, label=w)
        for x, y, t in zip(xs, ys, ("G", "i")):
            ax.annotate(t, (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=10, weight="bold")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("residency")
    ax.set_ylabel("z vs product band (5 non-anchor probes)")
    ax.set_title("THE ANCHORS — Gmail/iPhone vs the product band",
                 fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1][1]
    fams6 = ("cap-cur", "lang", "founder-anchor", "product", "near-uscap",
             "rev-capital")
    fam_col = {"cap-cur": "#8e44ad", "lang": "#5d6d7e", "product": "#c0392b",
               "founder-anchor": "#0d5c3f", "near-uscap": "#1a6faf",
               "rev-capital": "#b7950b"}
    width = 0.8 / len(WASHES)
    for wi_, w in enumerate(WASHES):
        vals = [families[w]["medians"][fam]["median"] for fam in fams6]
        ax.bar(np.arange(len(fams6)) + (wi_ - (len(WASHES) - 1) / 2) * width,
               vals, width=width, color=[fam_col[f] for f in fams6],
               alpha=0.45 + 0.25 * wi_, label=w)
    ax.set_xticks(range(len(fams6)), fams6, rotation=30, fontsize=7.5)
    ax.set_ylabel("median residency (A-primary)")
    ax.set_title("FAMILIES — nearrel vs product at t=0 "
                 f"(ratio w1 {families['w1']['nearrel_over_product_ratio']:.2f}"
                 if "w1" in WASHES else "", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    ax = axes[1][2]
    for w, mk_, col in zip(WASHES, ("o", "s", "^"),
                           ("#c0392b", "#2471a3", "#7f8c8d")):
        ax.scatter(res_flavors[w]["A_primary"], res_flavors[w]["F_fullwash"],
                   s=18, alpha=0.55, marker=mk_, color=col, label=f"{w}: A vs F")
    lims = [min(np.min(res_flavors[w]["A_primary"]) for w in WASHES),
            max(np.max(res_flavors[w]["A_primary"]) for w in WASHES)]
    ax.set_xlabel("residency (split-A primary)")
    ax.set_ylabel("residency (full-wash basis)")
    ax.set_title("BASIS-FLAVOR STABILITY (rank test robustness)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    if not SMOKE:
        rho_st = spearmanr(np.concatenate([res_flavors[w]["A_primary"]
                                           for w in WASHES]),
                           np.concatenate([res_flavors[w]["F_fullwash"]
                                           for w in WASHES])).statistic
        ax.set_title(f"BASIS-FLAVOR STABILITY  A-vs-F rho={rho_st:.3f}",
                     fontsize=9.5)

    fig.suptitle(f"E256 — FQ14 THE SUPPORT-SPAN RESIDENCY CENSUS  |  "
                 f"verdict: {verdict}", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    png1 = rd / "e256_residency_by_class.png"
    fig.savefig(png1, dpi=140)
    plt.close(fig)
    metrics["plot_outputs"] = [str(png1)]

    # ------------------------------------------------------------- P9 closeout
    (rd / "journal.json").write_text(
        json.dumps(journal, indent=1, default=float), encoding="utf-8")
    metrics["honesty_reflex"] = {
        "n": "3 washes x 54 probes; ONE organism (GPT-2 124M) — "
             "instance-relative scope; pooled tests treat the wash as a "
             "repeated measure (disclosed)",
        "provenance": "the span bases are REBUILT from e234's committed "
                      "journal Gram/dmat/gnorms by the module-verified "
                      "algebra (no replay — desk-only mandate); "
                      "G_REPRO_SPECTRUM/ANGLES/ALIGN certify the rebuild "
                      "against e234's and e226's committed numbers",
        "instrument_floor": "per-cos fp16 cache floor ~5e-4 (e226's "
                            "disclosed floor, both sides rounded) rides "
                            "every residency value; residency values near "
                            "the floor are noise-tier (disclosed)",
        "frozen_fates": "the death orders (e230 w1/w2 + e247 w3), the flip "
                        "modes (e247's 11-row table), and the thermal "
                        "records (e238) are read at runtime from the "
                        "committed metrics — no re-derivation",
        "multiplicity": "three joins + registered co-report flavors; NO "
                        "correction claimed",
        "circularity": "none by construction: residency is t=0-only "
                       "geometry; the fates are post-wash outcomes "
                       "measured by independent cells",
        "nothing_guaranteed": "MIXED is a real outcome (the registered "
                              "text carries it); the counterfeit-self "
                              "aiming data is descriptive regardless of "
                              "the verdict",
    }
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "desk (CPU, BLAS threads 4); no organism load, no GPU, "
                  "no replay",
        "load_checks": len(load_checks),
    }
    metrics["trims"] = []
    metrics["deviations"] = deviations
    metrics["status"] = "DONE" if not SMOKE else "SMOKE"
    write_metrics("DONE" if not SMOKE else "SMOKE (nothing adjudicated)")
    log(f"done -> {rd}  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
