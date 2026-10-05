"""E266 — THE INSTALL-SIDE FISHER CENSUS (T241/T242's registered next
Jacobian cell: the unification's second chance, now with a LOCATED cliff
at k ~ 10,000 of 2,739,072 = 0.37% of the space, e264's SHARP-THRESHOLD
with the 609x adjacent jump at 1k -> 10k).

DESK-ONLY on COMMITTED records (threads 4; e263 owns the GPU — never
touched, G_NOGPU). No model forward, no corpus, no training: closed-form
linear algebra on committed Gram/SV records, committed checkpoints
(CPU torch.load) and committed ledger scalars. Module-imports e265's
census machinery where possible (the convention): spectrum_stats comes
from lab/e265_fisher_census.py VERBATIM.

THE HONEST INVENTORY (stated up front, per the dispatch): the committed
caches hold NO install-time (teach-stream) Gram anywhere — runs/e246,
runs/e258 and runs/e260 have no journal.json, and their install arms
committed per-step SCALARS only (gn, kept_frac, in_span_frac, v_excess;
the ledgers), never gradient cross-products. The distributions actually
held are:
  (W1) the 124M wash Grams, 80x80, w1/w2/w3 (e234 journal) — the CORPUS
       wash distribution; window resolves k <= 80; 10k is 125x beyond it;
  (W2) the 2.74M e131-root wash 20-step Gram-SVD sv (e231 J1) — corpus
       wash; window resolves k <= 20; 10k is 500x beyond it;
  (W3) the 2.74M g1c-root (the cliff's own organism) wash LATE-half 10
       SVs + the 20-step L2 history (e246) — corpus wash; k <= 10;
  (W4) the full-resolution 2,739,072-dim wash-history diagonal
       second-moment v (runs/checkpoints/e258_vmap.pt, e258's committed
       v-map, fp64 recursion over the seed-10902 20-step wash) — Adam's
       standing ruler at the installs; WASH-derived (not teach); the
       ONLY held object in which k = 10k IS resolvable;
  (T1) the teach stream's per-step ledger scalars across the committed
       install arms (e246 ALIGNED/ORTHO/FREE, e258 VHEAVY/VLIGHT/FREE,
       e260 RANDOM/SPAN/FREE, e261 K1K/K237K/FREE, e264 K10K/K40K/
       K100K/FREE): kept_frac at five rungs (the teach gradient's
       capture by certified random rank-k rooms), in_span_frac /
       applied_in_span_frac (its capture by the corpus 10-span), and
       v_excess — the teach distribution's resolvable geometry;
  (T2) the teach stream's integrated OUTCOME at full resolution: the
       write displacements d = flat(root) - flat(e001 base) of the
       committed natural-install roots (g1c_root — the lineage's own;
       e264_FREE_root — this session's natural replicate). Their
       dimensional mass profiles (k50, eranks, cum-at-k) are teach-side
       subspace reads at 2.74M resolution;
  (T3) rung roots (e261 K1K/K237K, e264 K10K/K40K/K100K/K237KCONS2)
       — profiled and TABLED as outcome context ONLY: their ranks are
       set by the intervention (the rooms), so they are CIRCULAR as
       Fisher evidence and excluded from the verdict inputs (disclosed).

REGISTERED BARS (frozen VERBATIM from the e266 dispatch BEFORE any
compute below is run; no bar shopping):

  TEACH-KNEELS-NEAR-CLIFF — "any committed teach/install-side spectrum
  or subspace read shows structure (a knee, an erank, a dimension
  ratio) within a factor 3 of the cliff's ~10k at its resolvable scale
  — the unification's second chance lands: the cliff is the
  INSTALL-distribution Fisher's effective rank"

  WASH-LIKE — "the install-side top is as isotropic as the wash's
  (erank at the window's edge; no knee) — both distributions'
  observable tops are flat; the cliff's carrier is below both windows
  or non-spectral; the honest map"

  MIXED — "the tables verbatim, all distributions, no inflation"

OPERATIONALIZATION (frozen with the bars; fixes clauses, moves nothing):
  * the cliff := k* = 10,000 (e264's committed floor_cross_k; the
    609x jump 1k -> 10k); the factor-3 band := [10000/3, 3*10000] =
    [3333.3, 30000]; N := 2,739,072 (e246 G_ROOTSPAN); the band as a
    fraction := [0.00122, 0.01095] of the space.
  * TEACH-side structure locations (the bar's evidence set):
      L(T1) := any rung k in {1000, 10000, 40000, 100000, 237123} whose
               kept-frac isotropy ratio r_k = median_t(kept_frac) /
               sqrt(k/N) departs from [0.5, 2.0] (a >=2x departure is
               structure; its location is that rung's k);
      L(T2) := for each natural write (g1c_root, e264_FREE_root): the
               locations {k50, erank(0.1), erank(0.01)} of its squared-
               mass profile (erank(tol) := smallest k with cumulative
               mass >= 1 - tol; erank(0.5) == k50);
      L(T3) := the corpus-span coupling sits at the span's rank 10 —
               333x below the band floor; CANNOT fire; disclosed.
  * WASH-side flatness (the WASH-LIKE clause): the held Gram tops'
               knees at k <= 2 with erank(1e-2) in each window (re-
               derived by module-import; e265's committed facts
               re-verified), and the full-res diagonal's own locations
               {k50_v, erank_v(0.1), erank_v(0.01)} OUT of the band.
  * VERDICT: TEACH-KNEELS-NEAR-CLIFF iff any teach-side location
    (L(T1) or L(T2)) lands in the band. WASH-LIKE iff no teach-side
    location is in band AND no v-map location is in band AND every
    held wash top's knee is <= 2. MIXED otherwise (e.g. teach flat but
    the wash-derived full-res diagonal kneels at the band — the
    "carrier below the windows" clause of WASH-LIKE is then FALSE at
    the only scale where it is testable, and the map is mixed), or any
    table splits.
  * the v-map is WASH-derived and CANNOT fire the TEACH bar (the bar
    says teach/install-side); its reads enter the WASH-LIKE clause and
    the MIXED branch. Disclosed wherever it appears.
  * the rung roots are excluded from the evidence set (circularity);
    their profiles are tabled under rung_context.
  * Principal angles between the teach stream's gradient span and the
    corpus stream's span are NOT computable from committed records
    (no teach-span vectors were ever committed; the corpus side holds
    the 10-dim late span + Grams only) — disclosed, not estimated.

Cells:
  (1) THE TOP OF EVERY GRAM HELD — module-import spectrum_stats on W1
      (w1/w2/w3), W2, W3: decay, knee, eranks; the resolution limit vs
      10k stated per distribution.
  (2) THE FULL-RESOLUTION JOIN — the v-map's diagonal census (W4) and
      the natural writes' profiles (T2) at 2,739,072 resolution, with
      the k-grid cums and the band tests; the isotropic-linear
      reference drawn on both.
  (3) THE TEACH LEDGER CENSUS (T1) — kept-frac isotropy ratios per
      rung (medians + IQR over the committed 41-row ledgers), the
      in-span coupling table, the v-excess context.
  (4) THE CONDITIONING PROFILE (co-report) — the two streams' coupling
      numbers verbatim from committed records: the teach gradient's
      mass excess in the corpus 10-span (vs the random floor), the
      supports' span-capture excess in the 124M wash window (e265's
      committed NTK tables), the write's final cos-to-span (e260/e264
      displacement_loads), and the dimension-ratio ladder (write k50
      vs wash-window eranks vs the cliff).

Smoke mode (E266_SMOKE=1): the g1c write profile + the v-map census
only; no plots; quick plumbing verification.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e265_fisher_census as E265          # the census machinery, VERBATIM

spectrum_stats = E265.spectrum_stats        # module-import convention

REPO = HERE.parent
RUNS = REPO / "runs"
CKPT = RUNS / "checkpoints"
RD = RUNS / "e266"
SMOKE = os.environ.get("E266_SMOKE", "") == "1"

N = 2_739_072                               # e246 G_ROOTSPAN (gated)
CLIFF_K = 10_000                            # e264 floor_cross_k (committed)
BAND = (CLIFF_K / 3.0, 3.0 * CLIFF_K)       # the factor-3 band, frozen
BRACKET_OLD = (1000, 237123)                # e261's pre-location bracket
K_GRID = (1, 10, 100, 1000, 3333, 10000, 30000, 100000, 237123, 500000,
          1000000, 2739072)
PROFILE_TOLS = (0.5, 0.1, 0.01)             # erank mass tolerances (frozen)
CACHE_SCALE = E265.CACHE_SCALE
SQ = E265.SQ
WASHES = ("w1", "w2", "w3")
RUNG_KS = (1000, 10000, 40000, 100000, 237123)
# the committed medians this cell must reproduce (e264's kept_frac_curve)
E264_KEPT_CURVE = {1000: 0.016095496225535792,
                   10000: 0.060045162390265784,
                   40000: 0.12105602142254653,
                   100000: 0.19149141218938606,
                   237123: 0.2945979051104054}
E258_VSTATS = {"mean": 1.9735449455913735e-07,
               "median": 6.045241757117187e-08,
               "q25": 2.0388100097079587e-08,
               "q75": 1.3248366315110616e-07,
               "q95": 5.234346645011101e-07,
               "q99": 2.376100610490539e-06}
E258_K50 = 237123                           # the committed k_raw (gate)
E258_ROOT_DISP_L2 = 21.559598922729492      # the committed L2 (gate)
E265_PR_231 = 4.117363094870959             # committed PR cross-checks
E265_PR_124M = {"w1": 9.209354064251515, "w2": 7.1209812963002594,
                "w3": 11.526272945807355}
E265_SPANCAP_MEAN = 0.0018277482949078244   # e265 committed, w1 (cited)

t0 = time.time()
LOG_LINES: list[str] = []


def log(msg: str):
    line = f"[e266 +{time.time()-t0:6.1f}s] {msg}"
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


# ----------------------------------------------------- full-res profile stats

def profile_stats(sq: np.ndarray) -> dict:
    """Frozen statistics of a NONNEGATIVE full-resolution mass vector
    (squared coordinate magnitudes), fp64. Order-invariant. erank(tol)
    := smallest k with cumulative sorted mass >= 1 - tol (tol on the
    UNEXPLAINED mass; erank(0.5) == k50 by construction). No window
    caps exist at full resolution."""
    x = np.asarray(sq, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    tot = float(x.sum())
    xs = np.sort(x)[::-1]
    cum = np.cumsum(xs) / tot
    pr = float(tot ** 2 / float(np.sum(xs ** 2)))
    erank = {}
    for tol in PROFILE_TOLS:
        k = int(np.searchsorted(cum, 1.0 - tol) + 1)
        erank[str(tol)] = min(k, len(xs))
    k50 = int(np.searchsorted(cum, 0.5) + 1)
    return {
        "n": int(len(xs)),
        "total_mass": tot,
        "participation_ratio": pr,
        "k50": k50,
        "erank": erank,
        "erank_as_fraction": {t: erank[t] / len(xs) for t in erank},
        "cum_at_k": {str(k): float(cum[k - 1]) for k in K_GRID},
        "excess_over_linear_at_k": {str(k): float(cum[k - 1] / (k / len(xs)))
                                    for k in K_GRID},
    }


def flat_sd(sd: dict) -> np.ndarray:
    """Flatten a state dict in state_dict order (fp64). The k50 gate
    against e258's committed 237123 certifies the convention matches
    the lineage's net.parameters() order (k50/L2 are order-invariant;
    the gate still pins the same object)."""
    return torch.cat([sd[k].double().reshape(-1) for k in sd
                      if hasattr(sd[k], "numel") and sd[k].numel() > 0]) \
        .numpy()


def load_ckpt(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    return sd, meta


# ------------------------------------------------------------------- main

def main() -> int:
    metrics: dict = {
        "experiment": "e266_install_fisher",
        "phase": ("THE INSTALL-SIDE FISHER CENSUS — desk-only, committed "
                  "records: the honest inventory (no teach Gram exists), "
                  "the top of every held Gram re-derived, the "
                  "full-resolution join (the v-map diagonal + the natural "
                  "writes' dimension profiles) against the LOCATED cliff "
                  "at ~10k, the teach ledger census, the conditioning "
                  "profile co-report"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registration": ("bars VERBATIM from the e266 dispatch (T242's "
                         "registered cell), frozen before compute; see "
                         "docstring; no bar shopping"),
        "bars_verbatim": {
            "TEACH-KNEELS-NEAR-CLIFF": ("any committed teach/install-side "
                                        "spectrum or subspace read shows "
                                        "structure (a knee, an erank, a "
                                        "dimension ratio) within a factor 3 "
                                        "of the cliff's ~10k at its "
                                        "resolvable scale — the "
                                        "unification's second chance lands: "
                                        "the cliff is the INSTALL-"
                                        "distribution Fisher's effective "
                                        "rank"),
            "WASH-LIKE": ("the install-side top is as isotropic as the "
                          "wash's (erank at the window's edge; no knee) — "
                          "both distributions' observable tops are flat; "
                          "the cliff's carrier is below both windows or "
                          "non-spectral; the honest map"),
            "MIXED": "the tables verbatim, all distributions, no inflation",
        },
        "operationalization": {
            "cliff_k": CLIFF_K,
            "cliff_source": "runs/e264/metrics.json adjudication.reads."
                            "floor_cross_k (SHARP-THRESHOLD; the 609x "
                            "adjacent jump 1k -> 10k)",
            "factor3_band": {"lo": BAND[0], "hi": BAND[1],
                             "as_fraction": {"lo": BAND[0] / N,
                                             "hi": BAND[1] / N}},
            "n_params": N,
            "teach_structure_locations": {
                "T1_keptfrac": "any rung k in {1k,10k,40k,100k,237k} whose "
                               "kept-frac isotropy ratio median_t(kept)/"
                               "sqrt(k/N) leaves [0.5, 2.0]; location = k",
                "T2_write": "per natural write: {k50, erank(0.1), "
                            "erank(0.01)} of the squared-mass profile",
                "T3_span": "the corpus-span coupling sits at rank 10 — "
                           "333x below the band floor; cannot fire; "
                           "disclosed"},
            "wash_flatness": "held Gram tops' knees at k <= 2 (e265's "
                             "committed facts re-verified by module-"
                             "import) AND the v-map diagonal's locations "
                             "out of band",
            "verdict_rule": ("TEACH-KNEELS-NEAR-CLIFF iff any teach-side "
                             "location in band; WASH-LIKE iff no teach "
                             "location AND no v-map location in band AND "
                             "all held wash knees <= 2; MIXED otherwise"),
            "vmap_cannot_fire_teach": ("the v-map is WASH-derived (the "
                                       "seed-10902 wash history's second "
                                       "moment): it enters the WASH-LIKE "
                                       "clause and the MIXED branch, "
                                       "never the TEACH bar"),
            "rung_circularity": ("rung roots' ranks are set by the rooms "
                                 "(the intervention): profiled as outcome "
                                 "context, excluded from the verdict"),
            "principal_angles": "teach-span vs corpus-span principal "
                                "angles NOT computable (no teach-span "
                                "vectors committed) — disclosed",
        },
        "resolution_disclosure": (
            "the committed windows resolve k <= 80 (the 124M wash Grams), "
            "k <= 20 (e231 J1) and k <= 10 (e246's late half); the cliff "
            "at k = 10,000 of 2,739,072 (0.37%) lies 125x / 500x / 1000x "
            "beyond them and CANNOT be seen in any held Gram. k = 10k is "
            "resolvable ONLY in the full-resolution objects this census "
            "adds: the wash-derived v-map diagonal (2.74M coords) and the "
            "teach stream's write displacements (2.74M coords)."),
        "compute": {"device": "CPU desk only (torch threads 4; e263 owns "
                              "the GPU — untouched)", "gpu_touched": False},
        "gates": {},
    }

    # ---------------- load committed records + provenance
    src, prov = {}, {}
    files = [("e231_metrics", "runs/e231/metrics.json"),
             ("e234_journal", "runs/e234/journal.json"),
             ("e246_metrics", "runs/e246/metrics.json"),
             ("e258_metrics", "runs/e258/metrics.json"),
             ("e260_metrics", "runs/e260/metrics.json"),
             ("e261_metrics", "runs/e261/metrics.json"),
             ("e264_metrics", "runs/e264/metrics.json"),
             ("e265_metrics", "runs/e265/metrics.json")]
    ckpts = [("e258_vmap", "e258_vmap.pt"),
             ("e001_base", "e001.pt"),
             ("g1c_root", "g1c_root.pt"),
             ("e264_FREE_root", "e264_FREE_root.pt"),
             ("e264_K10K_root", "e264_K10K_root.pt"),
             ("e264_K40K_root", "e264_K40K_root.pt"),
             ("e264_K100K_root", "e264_K100K_root.pt"),
             ("e264_K237KCONS2_root", "e264_K237KCONS2_root.pt"),
             ("e261_K1K_root", "e261_K1K_root.pt"),
             ("e261_K237K_root", "e261_K237K_root.pt")]
    if SMOKE:
        ckpts = [c for c in ckpts if c[0] in ("e258_vmap", "e001_base",
                                              "g1c_root")]
    for tag, rel in files:
        p = RUNS / rel
        src[tag] = json.loads(p.read_text(encoding="utf-8"))
        prov[tag] = {"file": f"runs/{rel}", "md5": md5(p)}
    sd_cache, meta_cache = {}, {}
    for tag, name in ckpts:
        p = CKPT / name
        sd_cache[tag], meta_cache[tag] = load_ckpt(p)
        prov[f"ckpt_{tag}"] = {"file": f"runs/checkpoints/{name}",
                               "md5": md5(p)}
    metrics["provenance_records"] = prov
    log(f"committed records loaded ({len(files)} metrics/journal + "
        f"{len(ckpts)} checkpoints; md5s recorded)")

    # ---------------- CELL 1: the top of every Gram held
    # (W2) e231 J1 — the 2.74M e131-root wash 20-step spectrum
    sv231 = np.array(src["e231_metrics"]["cells"]["J1"]["primary_stream"]
                     ["span"]["sv"], dtype=np.float64)
    st_231 = spectrum_stats(sv231 ** 2)
    # (W3) e246 late half — the g1c root (the cliff's organism)
    sv246 = np.array(src["e246_metrics"]["span"]["span"]["sv"],
                     dtype=np.float64)
    st_246 = spectrum_stats(sv246 ** 2)
    hist246 = src["e246_metrics"]["span"]["history"]["rows"]
    # (W1) e234 — the 124M wash Grams, true-unit conversion (e265 math)
    J = src["e234_journal"]
    st_124m, diag_dev, sym_dev = {}, 0.0, 0.0
    for w in WASHES:
        Gs = np.array(J[f"gram_scaled_{w}"], dtype=np.float64)
        gn = np.array(J[w]["gnorms"], dtype=np.float64)
        G = Gs * np.outer(gn, gn) / SQ
        diag_dev = max(diag_dev, float(np.max(np.abs(np.diag(G) - gn ** 2))
                                       / np.max(gn ** 2)))
        sym_dev = max(sym_dev, float(np.max(np.abs(G - G.T))
                                     / np.max(np.abs(G))))
        ev = np.linalg.eigvalsh(G)[::-1]
        st_124m[w] = spectrum_stats(ev)
    metrics["gates"]["G_UNITS"] = {
        "check": "e265's G_UNITS identity re-run on the e234 Grams",
        "max_dev": diag_dev, "tol": 1e-3, "pass": bool(diag_dev < 1e-3)}
    metrics["gates"]["G_SYMMETRY"] = {
        "max_dev": sym_dev, "tol": 1e-12, "pass": bool(sym_dev < 1e-12)}
    metrics["gates"]["G_IMPORT"] = {
        "check": "module-import identity: PR(e231 sv^2) via e265's "
                 "spectrum_stats vs the committed 4.117363094870959",
        "recomputed": st_231["participation_ratio"],
        "committed": E265_PR_231,
        "dev": abs(st_231["participation_ratio"] - E265_PR_231),
        "tol": 1e-6,
        "pass": bool(abs(st_231["participation_ratio"] - E265_PR_231)
                     < 1e-6)}
    pr_ok = all(abs(st_124m[w]["participation_ratio"] - E265_PR_124M[w])
                < 1e-6 for w in WASHES)
    metrics["gates"]["G_PR265"] = {
        "check": "re-derived 124M wash PRs vs e265's committed w1/w2/w3",
        "recomputed": {w: st_124m[w]["participation_ratio"] for w in WASHES},
        "committed": E265_PR_124M, "tol": 1e-6, "pass": bool(pr_ok)}

    def gram_row(name, st, window, n_params, dist):
        knee = st["knee_loggap"]["k"]
        return {
            "name": name, "distribution": dist, "n_params": n_params,
            "window": window, "resolves_k_upto": window,
            "cliff_over_window": round(CLIFF_K / window, 1),
            "cliff_resolvable": bool(window >= CLIFF_K),
            "knee": knee, "knee_loggap": st["knee_loggap"]["log10_ratio"],
            "erank_1e-2": st["erank"]["0.01"]["k"],
            "erank_frac_of_window": st["erank"]["0.01"]["k"] / window,
            "participation_ratio": st["participation_ratio"],
            "lambda_n_over_1": st["lambda_n_over_1"],
            "decay_ei_over_e1": st["decay_ei_over_e1"],
            "spectrum": st["spectrum"],
        }

    grams = {
        "W2_e231J1": gram_row("2.74M e131-root wash (e231 J1, 20 steps)",
                              st_231, 20, N, "corpus wash"),
        "W3_e246late": gram_row("2.74M g1c-root wash LATE half (e246, "
                                "10 SVs)", st_246, 10, N, "corpus wash"),
    }
    for w in WASHES:
        grams[f"W1_e234_{w}"] = gram_row(
            f"124M wash {w} (e234, 80x80 Gram)", st_124m[w], 80,
            124_439_808, "corpus wash")
    grams["W3_e246late"]["steps_L2_counts"] = [r["L2"] for r in hist246]
    grams["W3_e246late"]["l2_disclosure"] = (
        "the g1c wash's 20-step L2 history (1.654 -> 0.418); no full "
        "20x20 Gram was ever committed at the g1c root")
    metrics["gram_spectra_reverified"] = grams
    wash_knees_ok = all(grams[k]["knee"] <= 2 for k in grams)
    knees = {k: grams[k]["knee"] for k in grams}
    eranks_124m = {w: st_124m[w]["erank"]["0.01"]["k"] for w in WASHES}
    gates1 = "PASS" if (pr_ok and diag_dev < 1e-3 and sym_dev < 1e-12
                        and metrics["gates"]["G_IMPORT"]["pass"]) else "FAIL"
    log(f"CELL 1: held Gram tops — knees {knees}; 124M eranks(1e-2) "
        f"{eranks_124m}/80; cell-1 gates {gates1}")
    write_metrics(metrics, "PARTIAL: cell 1 (the held Grams' tops) done")

    # ---------------- CELL 2a: the v-map diagonal census (full res)
    v = sd_cache["e258_vmap"]["v_flat_fp32"].double().numpy().astype(
        np.float64)
    vm_meta = meta_cache["e258_vmap"]
    qs = np.percentile(v, [25, 50, 75, 95, 99])
    dev_rel = lambda a, b: abs(float(a) - b) / b
    vmap_gate_devs = {
        "mean": dev_rel(v.mean(), E258_VSTATS["mean"]),
        "median": dev_rel(qs[1], E258_VSTATS["median"]),
        "q25": dev_rel(qs[0], E258_VSTATS["q25"]),
        "q75": dev_rel(qs[2], E258_VSTATS["q75"]),
        "q95": dev_rel(qs[3], E258_VSTATS["q95"]),
        "q99": dev_rel(qs[4], E258_VSTATS["q99"]),
    }
    metrics["gates"]["G_VMAP"] = {
        "check": "loaded v-map's stats vs e258's committed v_stats "
                 "(fp32 storage tolerance) + meta identity",
        "n_coords": int(v.shape[0]), "meta_experiment": vm_meta.get(
            "experiment"), "meta_k": vm_meta.get("k"),
        "rel_devs": vmap_gate_devs, "tol": 2e-4,
        "pass": bool(v.shape[0] == N
                     and vm_meta.get("experiment") == "e258"
                     and int(vm_meta.get("k", -1)) == E258_K50
                     and all(d < 2e-4 for d in vmap_gate_devs.values()))}
    vst = profile_stats(v)
    v_locations = {"k50": vst["k50"],
                   "erank(0.1)": vst["erank"]["0.1"],
                   "erank(0.01)": vst["erank"]["0.01"]}
    metrics["vmap_diag_census"] = {
        "what": ("the wash-history diagonal second-moment (Adam's "
                 "standing ruler at the installs) — the ONLY held object "
                 "with k = 10k resolvable; WASH-derived, cannot fire the "
                 "TEACH bar (disclosed)"),
        "convention": "e258's fp64 AdamW (0.9, 0.95) recursion over the "
                      "20-step seed-10902 unwalled wash history at the "
                      "g1c root (post-clip supply)",
        "stats": vst,
        "structure_locations": v_locations,
        "locations_in_band": {k: bool(BAND[0] <= x <= BAND[1])
                              for k, x in v_locations.items()},
        "cum_at_cliff": vst["cum_at_k"]["10000"],
        "excess_over_linear_at_cliff": vst["excess_over_linear_at_k"][
            "10000"],
        "context_quantiles_e258": E258_VSTATS,
    }
    log(f"CELL 2a: v-map diag — k50 {vst['k50']}, erank(0.1) "
        f"{vst['erank']['0.1']}, erank(0.01) {vst['erank']['0.01']}, "
        f"PR {vst['participation_ratio']:.0f}, cum@10k "
        f"{vst['cum_at_k']['10000']:.4f} "
        f"({vst['excess_over_linear_at_k']['10000']:.1f}x linear)")

    # ---------------- CELL 2b: the natural writes' profiles (full res)
    base_flat = flat_sd(sd_cache["e001_base"])
    write_set = [("g1c_root", "the committed lineage root "
                               "(install s400 + cons s300 — the "
                               "natural write the cliff's organism "
                               "carries)")]
    if "e264_FREE_root" in sd_cache:
        write_set.append(("e264_FREE_root", "this session's natural-"
                                            "install replicate "
                                            "(unprojected)"))
    writes = {}
    for tag, label in write_set:
        d = flat_sd(sd_cache[tag]) - base_flat
        st = profile_stats(d ** 2)
        locs = {"k50": st["k50"], "erank(0.1)": st["erank"]["0.1"],
                "erank(0.01)": st["erank"]["0.01"]}
        writes[tag] = {
            "label": label, "L2": float(np.linalg.norm(d)),
            "stats": st, "structure_locations": locs,
            "locations_in_band": {k: bool(BAND[0] <= x <= BAND[1])
                                  for k, x in locs.items()},
            "cum_at_cliff": st["cum_at_k"]["10000"],
        }
        log(f"CELL 2b: {tag} write — L2 {writes[tag]['L2']:.4f} k50 "
            f"{st['k50']} erank(0.1) {st['erank']['0.1']} erank(0.01) "
            f"{st['erank']['0.01']} PR {st['participation_ratio']:.0f} "
            f"cum@10k {st['cum_at_k']['10000']:.4f}")
    metrics["gates"]["G_K50"] = {
        "check": "g1c displacement k50 == e258's committed k_raw 237123 "
                 "(exact) and L2 vs the committed 21.5596 (fp32 storage "
                 "tolerance)",
        "k50_recomputed": writes["g1c_root"]["structure_locations"]["k50"],
        "k50_committed": E258_K50,
        "L2_recomputed": writes["g1c_root"]["L2"],
        "L2_committed": E258_ROOT_DISP_L2,
        "L2_rel_dev": dev_rel(writes["g1c_root"]["L2"], E258_ROOT_DISP_L2),
        "tol_L2": 1e-3,
        "pass": bool(writes["g1c_root"]["structure_locations"]["k50"]
                     == E258_K50
                     and dev_rel(writes["g1c_root"]["L2"],
                                 E258_ROOT_DISP_L2) < 1e-3)}
    metrics["teach_write_profiles"] = writes

    # the rung context (intervention-set; NOT verdict inputs)
    if not SMOKE:
        rung_ctx = {}
        for tag in ("e261_K1K_root", "e264_K10K_root", "e264_K40K_root",
                    "e264_K100K_root", "e264_K237KCONS2_root",
                    "e261_K237K_root"):
            d = flat_sd(sd_cache[tag]) - base_flat
            st = profile_stats(d ** 2)
            rung_ctx[tag] = {
                "L2": float(np.linalg.norm(d)), "k50": st["k50"],
                "erank(0.1)": st["erank"]["0.1"],
                "cum_at_10k": st["cum_at_k"]["10000"],
                "note": "intervention-set: the room forced this rank; "
                        "outcome context, not Fisher evidence",
            }
        metrics["rung_context"] = rung_ctx
        log("CELL 2b ctx: rung roots profiled (k50 "
            + str({k.split('_', 1)[1]: v["k50"]
                   for k, v in rung_ctx.items()}) + ") — context only")
    write_metrics(metrics, "PARTIAL: cell 2 (the full-resolution join) done")

    # ---------------- CELL 3: the teach ledger census
    ledgers = {1000: ("e261_metrics", "K1K"), 237123: ("e261_metrics",
                                                       "K237K"),
               10000: ("e264_metrics", "K10K"), 40000: ("e264_metrics",
                                                        "K40K"),
               100000: ("e264_metrics", "K100K")}
    kept_tab = {}
    for k, (mtag, arm) in ledgers.items():
        led = src[mtag]["arms"][arm]["install"]["ledger"]
        rows = [led[s]["kept_frac"] for s in led]
        inspan = [led[s]["in_span_frac"] for s in led]
        ainspan = [led[s]["applied_in_span_frac"] for s in led]
        iso = float(np.sqrt(k / N))
        ratio_med = float(np.median(rows) / iso)
        kept_tab[str(k)] = {
            "source": f"runs/{mtag.replace('_metrics', '')}/metrics.json "
                      f"arms.{arm}", "n_rows": len(rows),
            "kept_median": float(np.median(rows)),
            "kept_q1_q3": [float(np.percentile(rows, 25)),
                           float(np.percentile(rows, 75))],
            "isotropic_expectation": iso,
            "isotropy_ratio_median": ratio_med,
            "in_band_structure": bool(not (0.5 <= ratio_med <= 2.0)),
            "in_span_median": float(np.median(inspan)),
            "applied_in_span_median": float(np.median(ainspan)),
        }
        log(f"CELL 3: rung k={k}: kept median {np.median(rows):.4f} vs "
            f"iso {iso:.4f} -> ratio {ratio_med:.3f}"
            + ("  <-- STRUCTURE (>2x departure)" if not
               (0.5 <= ratio_med <= 2.0) else ""))
    metrics["gates"]["G_LEDGER"] = {
        "check": "kept-frac medians recomputed from the committed "
                 "ledgers vs e264's committed kept_frac_curve",
        "recomputed": {k: kept_tab[k]["kept_median"] for k in kept_tab},
        "committed": {str(k): E264_KEPT_CURVE[k] for k in RUNG_KS},
        "max_dev": max(abs(kept_tab[str(k)]["kept_median"]
                           - E264_KEPT_CURVE[k]) for k in RUNG_KS),
        "tol": 1e-9,
        "pass": bool(all(abs(kept_tab[str(k)]["kept_median"]
                             - E264_KEPT_CURVE[k]) < 1e-9
                         for k in RUNG_KS))}
    # context: e260's RANDOM/SPAN + e258's VHEAVY/VLIGHT + FREE arms
    ctx = {}
    for mtag, arms in [("e260_metrics", ("RANDOM", "SPAN", "FREE")),
                       ("e258_metrics", ("VHEAVY", "VLIGHT", "FREE")),
                       ("e246_metrics", ("ALIGNED", "ORTHO", "FREE"))]:
        for arm in arms:
            rec = src[mtag]["arms"].get(arm, {}).get("install", {})
            row = {}
            for f in ("ledger_kept_frac_median", "ledger_in_span_frac_median",
                      "ledger_applied_in_span_frac_median",
                      "ledger_v_excess_pre_median"):
                if f in rec:
                    row[f.replace("ledger_", "").replace("_median", "")] = \
                        rec[f]
            ctx[f"{mtag.split('_')[0]}_{arm}"] = row
    metrics["teach_ledger_census"] = {
        "kept_frac_table": kept_tab,
        "structure_def": "isotropy ratio outside [0.5, 2.0] at a rung",
        "any_teach_kept_structure": any(
            kept_tab[str(k)]["in_band_structure"] for k in RUNG_KS),
        "structure_rungs": [k for k in RUNG_KS
                            if kept_tab[str(k)]["in_band_structure"]],
        "arm_context": ctx,
        "span_coupling": {
            "corpus_span_rank": 10,
            "random_mass_expectation": 10 / N,
            "random_norm_expectation": float(np.sqrt(10 / N)),
            "teach_grad_mass_excess_vs_random": float(
                (kept_tab["10000"]["in_span_median"] ** 2) / (10 / N)),
            "note": "the teach gradient's mass in the corpus 10-span vs "
                    "the random floor (in_span medians ~0.08: ~1900x "
                    "excess); location = rank 10 — 333x below the band "
                    "floor; cannot fire the TEACH bar",
        },
    }
    write_metrics(metrics, "PARTIAL: cell 3 (the teach ledger census) done")

    # ---------------- CELL 4: the conditioning profile (co-report)
    disp_loads = {}
    for mtag, arms in [("e264_metrics", ("FREE", "K10K", "K40K", "K100K")),
                       ("e260_metrics", ("FREE", "RANDOM", "SPAN"))]:
        for arm in arms:
            rec = src[mtag]["arms"].get(arm, {})
            for phase in ("install", "root"):
                dl = rec.get(phase, {}).get("displacement_loads")
                if dl:
                    disp_loads[f"{mtag.split('_')[0]}_{arm}_{phase}"] = dl
    metrics["conditioning_profile"] = {
        "what": "the two streams' coupling, verbatim from committed "
                "records — the resolvable form of the dispatch's "
                "constructive alternative",
        "couplings": {
            "teach_grad_vs_corpus_span_mass_excess":
                metrics["teach_ledger_census"]["span_coupling"][
                    "teach_grad_mass_excess_vs_random"],
            "supports_vs_124m_wash_window_mass_excess": float(
                E265_SPANCAP_MEAN / (80 / 124_439_808)),
            "write_install_final_cos_to_span_e260_FREE":
                src["e260_metrics"]["arms"]["FREE"]["install"][
                    "displacement_loads"]["cos_to_span"],
            "write_root_cos_to_span_e260_FREE":
                src["e260_metrics"]["arms"]["FREE"]["root"][
                    "displacement_loads"]["cos_to_span"],
            "note": ("the teach GRADIENT couples to the corpus span "
                     "(~1900x mass excess at rank 10) but the "
                     "install-final displacement carries only ~7x the "
                     "random floor in that span (cos 5.1e-3; the root "
                     "after cons 3.6e-2): the per-step alignment is "
                     "transient, not cumulative"),
        },
        "dimension_ratio_ladder": {
            "corpus_wash_window_eranks": {"e231J1_20": st_231["erank"]
                                          ["0.01"]["k"],
                                          "e246late_10": st_246["erank"]
                                          ["0.01"]["k"],
                                          "e234_w1_80": st_124m["w1"]
                                          ["erank"]["0.01"]["k"]},
            "wash_diag_v_k50": vst["k50"],
            "teach_write_k50": {t: writes[t]["structure_locations"]["k50"]
                                for t in writes},
            "the_cliff": CLIFF_K,
        },
        "principal_angles": "NOT computable from committed records (no "
                            "teach-span vectors exist); disclosed, not "
                            "estimated",
        "displacement_loads_verbatim": disp_loads,
    }
    log("CELL 4: conditioning profile assembled (couplings "
        + json.dumps(metrics["conditioning_profile"]["couplings"])[:220]
        + ")")

    # ---------------- adjudication
    teach_locs = []
    for k in RUNG_KS:
        if kept_tab[str(k)]["in_band_structure"]:
            teach_locs.append(("T1_keptfrac_rung", k))
    for t in writes:
        for nm, x in writes[t]["structure_locations"].items():
            if BAND[0] <= x <= BAND[1]:
                teach_locs.append((f"T2_{t}_{nm}", x))
    v_locs_in = [k for k, x in v_locations.items() if BAND[0] <= x <= BAND[1]]
    verdict_logic = {
        "teach_locations_in_band": teach_locs,
        "vmap_locations_in_band": v_locs_in,
        "wash_knees_all_le_2": bool(wash_knees_ok),
        "t3_disclosed": "the span coupling sits at rank 10 — below the "
                        "band floor; cannot fire",
    }
    if teach_locs:
        verdict = "TEACH-KNEELS-NEAR-CLIFF"
    elif not v_locs_in and wash_knees_ok:
        verdict = "WASH-LIKE"
    else:
        verdict = "MIXED"
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": metrics["bars_verbatim"],
        "logic": verdict_logic,
        "the_numbers": {
            "cliff_k": CLIFF_K, "band": [BAND[0], BAND[1]],
            "held_gram_knees": {k: grams[k]["knee"] for k in grams},
            "vmap": v_locations,
            "teach_writes": {t: writes[t]["structure_locations"]
                             for t in writes},
            "kept_isotropy_ratios": {k: kept_tab[str(k)]
                                     ["isotropy_ratio_median"]
                                     for k in RUNG_KS},
            "vmap_cum_at_cliff": vst["cum_at_k"]["10000"],
            "write_cum_at_cliff": {t: writes[t]["cum_at_cliff"]
                                   for t in writes},
        },
    }
    log(f"ADJUDICATION: {verdict} (teach in band: {teach_locs}; v-map in "
        f"band: {v_locs_in}; wash knees ok: {wash_knees_ok})")
    write_metrics(metrics, f"PARTIAL: adjudication ({verdict}) — plots next")

    # ---------------- plots
    if not SMOKE:
        RD.mkdir(parents=True, exist_ok=True)

        # P1: the held Gram tops
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
        ax = axes[0]
        series = [("W2_e231J1", "#9467bd", "s", "2.74M e131 wash (e231 J1)"),
                  ("W3_e246late", "#8c564b", "D",
                   "2.74M g1c wash late (e246)")]
        for w, c in zip(WASHES, ("#1f77b4", "#d62728", "#2ca02c")):
            series.append((f"W1_e234_{w}", c, "o",
                           f"124M wash {w} (e234)"))
        for key, c, mk, lab in series:
            ev = np.array(grams[key]["spectrum"])
            ax.semilogy(np.arange(1, len(ev) + 1), ev / ev[0], marker=mk,
                        ms=3.5, lw=1.2, color=c, label=lab)
        ax.set_xlabel("eigenvalue index i (window ends at 80/20/10)")
        ax.set_ylabel(r"$\lambda_i/\lambda_1$ (log)")
        ax.set_title("every held Gram's TOP (re-derived, e265 "
                     "module-import)\nk = 10,000 lies 125-1000x beyond "
                     "every window")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for key, c, mk, lab in series:
            ev = np.array(grams[key]["spectrum"])
            cum = np.cumsum(ev) / ev.sum()
            ax.plot(np.arange(1, len(ev) + 1), cum, marker=mk, ms=3.5,
                    lw=1.2, color=c, label=lab)
        ax.set_xlabel("k (window)")
        ax.set_ylabel("cumulative explained Fisher mass")
        ax.set_title("the windows' mass curves — the cliff's 0.37% is "
                     "unreachable here")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.suptitle("E266 cell 1 — the distributions actually held: "
                     "NO teach-stream Gram exists (the honest inventory)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(RD / "e266_gram_tops.png", dpi=130)
        plt.close(fig)
        log("plot: e266_gram_tops.png")

        # P2: the full-resolution join
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
        ax = axes[0]
        kplot = np.unique(np.round(np.logspace(0, np.log10(N),
                                               1200)).astype(np.int64))

        def cum_on_grid(mass):
            xs = np.sort(mass)[::-1]
            cum = np.cumsum(xs) / xs.sum()
            return np.interp(kplot, np.arange(1, len(xs) + 1), cum)

        ax.plot(kplot, cum_on_grid(v), lw=1.6, color="#d62728",
                label="WASH diag v (e258 v-map; full 2.74M res)")
        ax.plot(kplot, cum_on_grid((flat_sd(sd_cache["g1c_root"])
                                    - base_flat) ** 2), lw=1.6,
                color="#1f77b4",
                label="TEACH write d(g1c root) — natural install")
        if "e264_FREE_root" in sd_cache:
            ax.plot(kplot, cum_on_grid((flat_sd(sd_cache["e264_FREE_root"])
                                        - base_flat) ** 2), lw=1.2,
                    ls="--", color="#1f77b4", alpha=0.7,
                    label="TEACH write d(e264 FREE) — replicate")
        if "e264_K10K_root" in sd_cache:
            ax.plot(kplot, cum_on_grid((flat_sd(sd_cache["e264_K10K_root"])
                                        - base_flat) ** 2), lw=1.0,
                    ls=":", color="#7f7f7f",
                    label="rung K10K root (context; room-set)")
        ax.plot(kplot, kplot / N, lw=1.0, color="k", alpha=0.5,
                label="isotropic reference k/N")
        ax.axvspan(BAND[0], BAND[1], color="#ffbb33", alpha=0.3,
                   label=f"factor-3 band around the cliff "
                         f"[{BAND[0]:.0f}, {BAND[1]:.0f}]")
        ax.axvline(CLIFF_K, color="#aa6600", lw=2)
        ax.annotate("the cliff k=10k", (CLIFF_K, 0.03), rotation=90,
                    fontsize=9, color="#aa6600", xytext=(4, 0),
                    textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("k coordinates (log; full resolution)")
        ax.set_ylabel("cumulative squared mass")
        ax.set_title("the only objects where k=10k is resolvable")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3, which="both")
        ax = axes[1]
        names, poss, cols_ = [], [], []
        for nm, x in v_locations.items():
            names.append(f"wash diag v — {nm}"); poss.append(x)
            cols_.append("#d62728")
        for t in writes:
            for nm, x in writes[t]["structure_locations"].items():
                names.append(f"{t.replace('_root','')} write — {nm}")
                poss.append(x); cols_.append("#1f77b4")
        for k in RUNG_KS:
            names.append(f"teach kept-frac rung k={k}")
            poss.append(k); cols_.append("#2ca02c")
        yv = np.arange(len(names))
        ax.barh(yv, poss, color=cols_, alpha=0.85)
        for y, p in zip(yv, poss):
            ax.annotate(f"{p:,}", (p, y), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=7)
        ax.axvspan(BAND[0], BAND[1], color="#ffbb33", alpha=0.3)
        ax.axvline(CLIFF_K, color="#aa6600", lw=2)
        ax.set_yticks(yv, names, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlim(300, 4e6)
        ax.set_xlabel("structure location (log dimensions)")
        ax.set_title(f"the join: {verdict} — no read kneels at the "
                     "cliff's scale\n(green = teach rungs; the cliff "
                     "band shaded)")
        ax.grid(alpha=0.3, axis="x")
        fig.suptitle("E266 cell 2 — the LOCATED-cliff join at full "
                     "resolution", fontsize=11)
        fig.tight_layout()
        fig.savefig(RD / "e266_full_res_join.png", dpi=130)
        plt.close(fig)
        log("plot: e266_full_res_join.png")

        # P3: the teach ledger reads
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
        ax = axes[0]
        ks = list(RUNG_KS)
        meds = [kept_tab[str(k)]["kept_median"] for k in ks]
        iso = [kept_tab[str(k)]["isotropic_expectation"] for k in ks]
        lo = [kept_tab[str(k)]["kept_q1_q3"][0] for k in ks]
        hi = [kept_tab[str(k)]["kept_q1_q3"][1] for k in ks]
        x = np.arange(len(ks))
        ax.errorbar(x, meds, yerr=[np.array(meds) - np.array(lo),
                                   np.array(hi) - np.array(meds)],
                    fmt="o-", ms=5, color="#2ca02c", capsize=3,
                    label="teach kept_frac (median, IQR)")
        ax.plot(x, iso, "s--", ms=5, color="k", alpha=0.6,
                label=r"isotropic $\sqrt{k/N}$")
        ax.set_xticks(x, [f"{k//1000}k" if k < 1e6 else "237k"
                          for k in ks])
        ax.set_xlabel("room rank k (the committed rungs)")
        ax.set_ylabel("||P_room g|| / ||g||")
        ax.set_title("the teach gradient vs random rooms:\n"
                     "indistinguishable from isotropic")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for k, c in zip(RUNG_KS, ("#2ca02c", "#1f77b4", "#d62728",
                                  "#9467bd", "#8c564b")):
            led = src[ledgers[k][0]]["arms"][ledgers[k][1]]["install"][
                "ledger"]
            steps = sorted(int(s) for s in led)
            ax.plot(steps, [led[str(s)]["in_span_frac"] for s in steps],
                    "-o", ms=2.5, lw=1, color=c,
                    label=f"teach grads, rung {k//1000}k"
                          if k < 1e6 else "237k")
        ax.axhline(float(np.sqrt(10 / N)), color="k", ls="--", lw=1,
                   label="random floor sqrt(10/N)=0.0019")
        ax.set_xlabel("install step (the committed ledger rows)")
        ax.set_ylabel("||V_10span g|| / ||g||")
        ax.set_title("teach-gradient coupling to the corpus 10-span:\n"
                     "~40x the random floor (mass ~1900x) — at rank 10, "
                     "not 10k")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[2]
        coup_names = ["teach grad in\ncorpus 10-span",
                      "supports in\n124M wash window",
                      "install-final write in\ncorpus 10-span"]
        coup_vals = [metrics["conditioning_profile"]["couplings"][
                         "teach_grad_vs_corpus_span_mass_excess"],
                     metrics["conditioning_profile"]["couplings"][
                         "supports_vs_124m_wash_window_mass_excess"],
                     float(src["e260_metrics"]["arms"]["FREE"]["install"][
                         "displacement_loads"]["cos_to_span"] ** 2
                         / (10 / N))]
        ax.bar(np.arange(3), np.log10(coup_vals),
               color=["#2ca02c", "#1f77b4", "#7f7f7f"], alpha=0.85)
        for i, v_ in enumerate(coup_vals):
            ax.annotate(f"{v_:.0f}x", (i, np.log10(v_)), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=9)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(np.arange(3), coup_names, fontsize=8)
        ax.set_ylabel("log10 mass excess over random")
        ax.set_title("the conditioning profile: couplings that ARE\n"
                     "resolvable (all at rank 10-80, none at 10k)")
        ax.grid(alpha=0.3, axis="y")
        fig.suptitle("E266 cells 3+4 — the teach-side reads (committed "
                     "ledgers; desk-only)", fontsize=11)
        fig.tight_layout()
        fig.savefig(RD / "e266_teach_reads.png", dpi=130)
        plt.close(fig)
        log("plot: e266_teach_reads.png")
        metrics["plot_outputs"] = ["runs/e266/e266_gram_tops.png",
                                   "runs/e266/e266_full_res_join.png",
                                   "runs/e266/e266_teach_reads.png"]

    # ---------------- honesty + close-out
    metrics["honesty_reflex"] = {
        "previsible_head": "the held Gram tops and their knees are "
                           "e265's committed facts (re-derived here by "
                           "module-import as gates, not as new content); "
                           "the NEW content is the full-resolution layer: "
                           "the v-map diagonal's census, the natural "
                           "writes' dimension profiles (the g1c write's "
                           "k50/erank published for the first time), the "
                           "kept-frac isotropy table, and the "
                           "no-teach-Gram inventory itself",
        "resolution": metrics["resolution_disclosure"],
        "logits_predict": "the join is a geometry-vs-capacity identity "
                          "check on committed records; the intervention "
                          "links live elsewhere and stand: e264 (the "
                          "rank installs at the located rungs), e254 "
                          "(the span carries 42-85% of v at k=2), e237 "
                          "(cutting the aligned component flips fates)",
        "no_inflation": "the rung roots' profiles are labeled "
                        "intervention-set and excluded from the verdict; "
                        "the v-map is labeled wash-derived and cannot "
                        "fire the teach bar; the teach side's only "
                        "full-res objects are OUTCOME displacements (the "
                        "integrated write), disclosed as such — the "
                        "teach INSTANTANEOUS Fisher remains unmeasured "
                        "anywhere in the lab",
    }
    metrics["builds_on"] = [
        "T240/T241/T242 (the Jacobian map; the wash-side verdict; the "
        "LOCATED cliff — this cell's registration chain)",
        "e265 (the census machinery — module-imported; the committed "
        "spectra, NTK span-captures and conventions this cell extends)",
        "e264/e261 (the located cliff k~10k + the committed rung ledgers "
        "and roots — the teach-side material)",
        "e258 (the committed v-map + k_rule k50=237123 — the full-res "
        "wash diagonal and the write gate)",
        "e246 (the g1c wash history + late span; the cliff's organism)",
        "e234/e231 (the 124M wash Grams; the 2.74M wash spectrum)",
        "e260 (the displacement_loads conventions; RANDOM/SPAN context)",
    ]
    metrics["whats_new"] = [
        "the FIRST full-resolution distribution reads in the Jacobian "
        "lane: the v-map diagonal's and the natural writes' dimension "
        "profiles at 2,739,072 coordinates — the scale where the cliff "
        "lives, which every Gram window misses by 125-1000x",
        "the honest inventory: no teach-stream Gram exists in any "
        "committed run — stated per distribution with resolution limits",
        "the teach ledger census: the kept-frac isotropy table across "
        "the five committed rungs (the teach gradient is "
        "indistinguishable from isotropic at every probed rank) and the "
        "span-coupling excesses (~1900x mass at rank 10)",
        "the conditioning profile co-report: the gradient-vs-write "
        "coupling asymmetry (the teach gradient couples to the corpus "
        "span; the integrated write escapes it, cos ~5e-3)",
    ]
    metrics["all_gates_pass"] = bool(
        all(g.get("pass", False) for g in metrics["gates"].values()))
    # G_NOGPU: assert nothing ever left the CPU
    cpu_ok = all(str(t.device) == "cpu"
                 for sd in sd_cache.values()
                 for t in (sd.values() if isinstance(sd, dict) else []))
    metrics["gates"]["G_NOGPU"] = {
        "check": "every loaded tensor on CPU (e263 owns the GPU)",
        "pass": bool(cpu_ok)}
    metrics["all_gates_pass"] = bool(
        all(g.get("pass", False) for g in metrics["gates"].values()))
    (RD / "run.log").write_text("\n".join(LOG_LINES) + "\n",
                                encoding="utf-8")
    write_metrics(metrics, f"DONE (verdict {verdict}; all gates "
                           f"{'PASS' if metrics['all_gates_pass'] else 'FAIL'})")
    return 0 if metrics["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
