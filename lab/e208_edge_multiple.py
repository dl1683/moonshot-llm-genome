"""E208 — THE EDGE-MULTIPLE CENSUS (T173's new scalar gets its table).

WHY. e205 minted a scalar and used it exactly once per organism: the
fact's consolidation EDGE (its own u0 static kill-D — the R2_SIGN ray
walked from the root until the fact's battery shuts) as a MULTIPLE of
its organism's IN-SPAN RANDOM BAND (the middle order statistic of the
organism's own in-span random-ray kill-Ds): org1 3.72x, MIRABEL 2.12x,
half 0.616x. Three numbers, one paragraph, no table. The scalar is "the
memory-vs-noise margin" — how many times wider than the organism's own
wash-noise band the fact's static edge sits. Nobody has asked the
census question yet: assemble EVERY (organism, fact) pair with
committed data and look at the margin's first structure.

WHAT BUILDS ON WHAT (directive 1):
  * e199/e200's committed u0 EDGES (org1 2.2699, MIRABEL 1.9471, half
    0.5252) and their committed natural WALKS (alive ledgers);
  * e_chart's fine-D in-span band (org1, seeds 11601-3 — e205's
    promoted instrument), e193b's inspan_range bands (seeds 11911-3,
    BOTH facts' rulers read on the SAME draws), e205's fresh half band
    (seeds 12001-3 — the only same-instrument band);
  * e193b's committed per-fact kills (ZEPHYRA's sign_kill under the
    FALLBACK g+0 ruler) and e198's phase0 walk (the ONE walk where two
    facts of the SAME organism were read side by side: ZEPHYRA died at
    t=1, MIRABEL survived to t=3);
  * e196's natural-walk kill at step 1 (the half lineage's protocol
    walk) and e197/e200's half-step counterfactual (the fork).
WHAT IS NEW: the cross-fact, cross-organism TABLE; the survival
comparison (margin class vs committed-walk survival — never asked
before); the g1bR survey (do the 2.74M wall roots have rays? no);
the first margin-vs-survival figure.

THE CELL: a DESK CENSUS on committed data — ZERO model compute, ZERO
torch. Every number is LOADED from a committed metrics.json with its
file+path recorded (G_LOAD); the arithmetic (multiples, medians, the
Spearman context) is recomputed and gated (G_ARITH); the walks'
alive/dead reads are re-checked against the 0.27 shut bar from the
committed rows (G_WALKS). e205's conventions carried verbatim: band =
middle order statistic, right-censored draws order above resolved;
first-dead GRID points are upper bounds (bias direction stated per
row); the half lineage's FRONTS belong to the counterfactual half-step
construction while its BAND is root-level.

REGISTERED BARS (frozen here, before assembly; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - MARGIN-PREDICTS: "fires if every fact with margin > 2 outlived
    (in its lineage's committed walk) every fact with margin < 1
    across the table — the noise margin is a survival predictor; the
    scalar earns object status."
  - MARGIN-DECORRELATED: "fires if survival and margin decorrelate
    across the table — the margin is a per-organism descriptor, not a
    predictor; the table stands as the first map."
  - GRADED: "any partial — the table verbatim with the caveats."

OPERATIONALIZATIONS (frozen before assembly; they fix the clauses,
they do not move the bars):
  * row = (organism, fact) with BOTH a committed u0 static kill-D
    (the fact's own ruler) AND a committed in-span band readable by
    that same ruler. Census rows: org1/ZEPHYRA, e193b/MIRABEL,
    e193b/ZEPHYRA, org2-f2("half")/ZEPHYRA.
  * edge(row) = D_kill(root, u0) under the row's own primary ruler,
    loaded from the PRIMARY committed source (e199 x2, e200, e193b).
  * band(row) = middle order statistic of the organism's in-span
    draws' kill-Ds AS READ BY THAT ROW'S OWN RULER (e193b's two facts
    share draws, differ in ruler — the ruler's contribution to the
    band is visible inside the table; T155's lesson).
  * multiple = edge / band. survival(row) = the number of NATURAL
    (protocol-step) wash steps of the lineage's committed walk after
    which the fact was still alive (ruler read > 0.27). The half
    lineage's half-step counterfactual survival is reported alongside
    as THE FORK, never adjudicated (e197's own framing: a
    counterfactual construction, not the lineage's protocol).
  * MARGIN-PREDICTS fires iff min(survival | margin > 2) >
    max(survival | margin < 1). MARGIN-DECORRELATED fires iff
    PREDICTS fails AND Spearman(margin, survival) over ALL census
    rows <= 0. GRADED = any partial. Composite order
    MARGIN-PREDICTS -> MARGIN-DECORRELATED -> GRADED (first two
    mutually exclusive on the primary reading). The gray zone
    (1 <= margin <= 2) is reported, never adjudicated by the letter.
  * DESK-FORCED DISCLOSURE (e205's convention): the census is
    desk-complete from committed data — the four rows' margins and
    survivals were readable at registration. The verdict is therefore
    DESK-FORCED and reported as forced; this cell's fresh work is the
    assembly, the gates, the g1bR survey, the figure and the caveats.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lab"))

RUNS = REPO / "runs"

REGISTERED_BARS = {
    "MARGIN-PREDICTS": "MARGIN-PREDICTS: \"fires if every fact with margin > 2 outlived (in its lineage's committed walk) every fact with margin < 1 across the table — the noise margin is a survival predictor; the scalar earns object status.\"",
    "MARGIN-DECORRELATED": "MARGIN-DECORRELATED: \"fires if survival and margin decorrelate across the table — the margin is a per-organism descriptor, not a predictor; the table stands as the first map.\"",
    "GRADED": "GRADED: \"any partial — the table verbatim with the caveats.\"",
    "operationalizations": (
        "row = (organism, fact) with committed u0 static kill-D AND a same-ruler "
        "in-span band; edge = D_kill(root,u0) on the row's own primary ruler "
        "(e199 x2 / e200 / e193b, loaded); band = middle order statistic of the "
        "organism's in-span draws' kill-Ds AS READ BY THAT ROW'S OWN RULER "
        "(e193b's two facts share draws, differ in ruler); multiple = edge/band; "
        "survival = natural (protocol-step) walk steps survived (ruler read > "
        "0.27); the half lineage's half-step counterfactual is THE FORK, "
        "reported never adjudicated; PREDICTS fires iff min(surv | margin>2) > "
        "max(surv | margin<1); DECORRELATED fires iff PREDICTS fails AND "
        "Spearman(margin, survival) over all rows <= 0; GRADED = any partial; "
        "composite PREDICTS -> DECORRELATED -> GRADED, first two mutually "
        "exclusive on the primary reading; the gray zone [1,2] is reported, "
        "never adjudicated by the letter."
    ),
    "registered_prediction": (
        "DESK-FORCED to MARGIN-PREDICTS on the natural-walk primary reading "
        "(org1 3.72x survived 1 step > half 0.616x survived 0; MIRABEL 2.12x "
        "survived 2 > half 0) — reported as forced, e205's convention. THE "
        "FORK: under the half-step counterfactual reading (half survival 4) "
        "PREDICTS fails and DECORRELATED's Spearman clause is the live "
        "outcome — the fork is the table's biggest caveat and is reported "
        "inside the verdict. The fresh compute owns only the gates, the "
        "g1bR survey, the table, and the figure."
    ),
    "registration": "the dispatch's registration IS the registration (the bars quoted verbatim in the module docstring and here, frozen before assembly). Adjudicate against exactly this; no bar shopping.",
}

DEVIATIONS = [
    "DESK CENSUS, ZERO MODEL COMPUTE: no torch import, no GPU, no training — every number loaded from committed metrics.json (the owner envelope's desk+census form; the 'tiny eval bursts' clause was not needed: NO band is missing — the fourth row's band (e193b/ZEPHYRA) was already committed in e193b's inspan_range, read by the ZEPHYRA ruler on the same draws as MIRABEL's).",
    "THE e193b PAIR SHARES ONE WALK AND ONE SET OF DRAWS: rows 2 and 3 are two facts of the SAME organism — the margins differ by ruler and edge, the walk is a single realization; the pair is the table's only within-organism contrast (and its only same-walk contrast).",
    "THE FALLBACK RULER ROW (e193b/ZEPHYRA): its ruler is g+0, the frozen mechanical fallback (g-12 read 0.1486 < 0.27 at the root) — a CROSS-GEOMETRY ruler vs the g-12 family and vs org2's g-4; its edge is a FIRST-DEAD GRID POINT (upper bound, e193b's 27-pt grid), its band likewise grid upper bounds — the multiple's bias direction is UNDETERMINED (both numerator and denominator upper-bounded).",
    "THE FORK (carried from e197/e200, load-bearing): the half lineage's PRIMARY walk is the NATURAL-step walk (e196: death AT t=1, gm 0.0068); its half-step counterfactual (e197/e200: alive t1-t4, death t5) is reported as the ALTERNATE reading — under it MARGIN-PREDICTS fails. The dispatcher's reading (natural walk) is the primary; both are in the table.",
    "INSTRUMENT HETEROGENEITY (e205's disclosures carried verbatim): org1's band = e_chart's FINE_GRID co-read (cap 1.5; 1-of-3 draws right-censored >1.5; the co-read PROMOTED to band instrument by e205, disclosed there); MIRABEL's + ZEPHYRA's bands = first-dead GRID points on e193b's 27-pt grid (upper bounds); the half band = e205's fresh onset-grid interpolated crossings (the ONLY same-instrument band). Grid class context: e193's own bracket interpolates 0.4% off the onset grid.",
    "RULER GEOMETRY ASYMMETRY (T153/T155, Rule 12): org1's ruler is the NOVEL geometry g-12 (its 0.27 shut bar sits on a battery the organism never trained); org2-f2's is TRAINED g-4 (max committed root read); e193b/ZEPHYRA's is the TRAINED g+0 fallback. Within a row, edge and band share the ruler (self-consistent); ACROSS rows the rulers differ — cross-row comparisons carry this flag.",
    "g1bR's 2.74M roots (C10907/C10908/W1_10907/W1_10908, s300): the rays DO NOT EXIST — no u0/static kill-D instrument, no in-span band in the committed metrics (verified by key-survey in P4) — OUT OF THE CENSUS, reported honestly, not silently dropped.",
    "the load-check is recorded, not gating (e204/e205's convention).",
    "Smoke mode: loads rows 1-2 only, stamps SMOKE, nothing adjudicated.",
]


def log(msg: str):
    print(f"[e208 {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cpu_load_probe():
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def load_metrics(name: str) -> dict:
    p = RUNS / name / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def band_median(kill_Ds: list) -> dict:
    """e205's convention verbatim: middle order statistic; None = right-
    censored (unresolved-high) and orders above all resolved."""
    resolved = sorted([d for d in kill_Ds if d is not None])
    n_cens = sum(1 for d in kill_Ds if d is None)
    n = len(kill_Ds)
    if n_cens >= 2:
        return {"median": None, "n": n, "n_resolved": len(resolved),
                "n_censored": n_cens, "defined": False,
                "note": "majority-censored band — UNDEFINED"}
    order = sorted([d if d is not None else float("inf") for d in kill_Ds])
    med = order[n // 2] if n % 2 == 1 else 0.5 * (order[n // 2 - 1] + order[n // 2])
    med = None if med == float("inf") else float(med)
    return {"median": med, "n": n, "n_resolved": len(resolved),
            "n_censored": n_cens, "defined": med is not None,
            "resolved_sorted": resolved}


def interp_crossing(d1, v1, d2, v2, bar=0.27):
    """Linear-in-D interpolation of the downcrossing (e199's convention)."""
    if not (v1 > bar >= v2):
        return None
    return d1 + (d2 - d1) * (v1 - bar) / (v1 - v2)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 1.0 + 0.5 * (i + j)
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


SHUT = 0.27


def build_rows(sources: dict) -> list:
    """Assemble the census rows from the loaded committed sources.
    Every value is extracted AT ITS COMMITTED PATH — no recompute."""
    e199, e200, e193b, e196, e198, e205, echart = (
        sources[k] for k in ["e199", "e200", "e193b", "e196", "e198", "e205", "e_chart"])

    # ---- row 1: org1 / ZEPHYRA --------------------------------------------
    org1_curve = e199["onset_curves"]["org1"]
    org1_ledger = e199["organisms"]["org1"]["alive_ledger"]
    org1_band_ds = echart["partB_subspace"]["e131"]["fine_D_summary"]["inspan_thr_D"]
    org1_band_list = [org1_band_ds["11601"], org1_band_ds["11602"], org1_band_ds["11603"]]

    # ---- row 2/3: e193b / MIRABEL + ZEPHYRA --------------------------------
    mir_curve = e199["onset_curves"]["mirabel"]
    mir_ledger = e199["organisms"]["mirabel"]["alive_ledger"]
    mir_band_list = e193b["inspan_range"]["MIRABEL"]["kill_Ds"]
    zeph_band_list = e193b["inspan_range"]["ZEPHYRA"]["kill_Ds"]
    zeph_sign_kill = e193b["adjudication"]["per_fact"]["ZEPHYRA"]["sign_kill"]
    zeph_ruler = e193b["rulers"]["primary"]["ZEPHYRA"]
    zeph_dial = e193b["rulers"]["root_dial_seven_geos"]["ZEPHYRA"]
    kz = e198["phase0_walk_rebuild"]["kill_zephyra_rederived"]
    kj1 = e198["phase0_walk_rebuild"]["walk_journal"][0]

    # ---- row 4: org2-f2 ("half") / ZEPHYRA ---------------------------------
    half_edge = e200["profiles"]["root_u0"]["D_kill"]
    half_band = e205["bands"]["half"]["kill_Ds_fresh"]
    half_nat_stop = e196["phase0_walk_rebuild"]["stop"]
    half_nat_s1 = e196["phase0_walk_rebuild"]["walk_journal"][0]["gm"]
    half_alt_ledger = e200["alive_ledger"]

    def surv_from_ledger(ledger):
        """Wash steps survived: count ALIVE entries with t >= 1 (the t=0
        root row, where present, is the pre-wash anchor — not a step)."""
        n = 0
        for e in ledger:
            if e.get("t", 1) == 0:
                continue
            if e.get("alive"):
                n += 1
            else:
                break
        return n

    rows = [
        {
            "id": "R1", "organism": "org1 (e131_consolidated_e113.pt)",
            "arch": "6L/6H/192d/256ctx, 2,739,072 params",
            "fact": "ZEPHYRA",
            "ruler": "ZEPHYRA install-60 g-12 (NOVEL geometry; root read 0.9156)",
            "ruler_class": "g-12 novel",
            "edge": org1_curve[0]["D_kill"],
            "edge_source": "runs/e199/metrics.json onset_curves.org1[0].D_kill (u0 = e192's R2, interpolated)",
            "edge_kind": "interpolated downcrossing (the onset instrument)",
            "band_kill_Ds": org1_band_list,
            "band_source": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D (seeds 11601-3; e205's promoted instrument)",
            "band_instrument": "e_chart FINE_GRID 11pts 0.05..1.5, interpolated linear-in-D; 1-of-3 right-censored >1.5",
            "walk_source": "e199's natural walk (STEP_L2 1.6543): alive_ledger t1 alive 0.6786 -> t2 death 9.83e-05",
            "death_t": 2, "survival": surv_from_ledger(org1_ledger),
            "survival_source": "runs/e199/metrics.json organisms.org1.alive_ledger",
            "alternate_walk": None,
            "caveats": [
                "band = the e_chart fine-D co-read PROMOTED to band instrument by e205 (disclosed there)",
                "n=3 draws with 1 right-censored (>1.5) — the median is censoring-robust",
                "ruler is the NOVEL g-12 geometry (T153's asymmetry: org1's shut bar sits on an untrained battery)",
            ],
        },
        {
            "id": "R2", "organism": "e193b (e193b_root.pt, fresh two-fact)",
            "arch": "6L/6H/192d/256ctx, 2,739,072 params (org1's EXACT architecture)",
            "fact": "MIRABEL",
            "ruler": "MIRABEL install-60 g-12 (root read 0.6236 — no fallback triggered)",
            "ruler_class": "g-12 novel",
            "edge": mir_curve[0]["D_kill"],
            "edge_source": "runs/e199/metrics.json onset_curves.mirabel[0].D_kill (u0 = e193b's R2_SIGN, interpolated)",
            "edge_kind": "interpolated downcrossing (the onset instrument)",
            "band_kill_Ds": mir_band_list,
            "band_source": "runs/e193b/metrics.json inspan_range.MIRABEL.kill_Ds (seeds 11911-3, e205's loaded band)",
            "band_instrument": "e193b D_GRID 27pts, FIRST-DEAD GRID POINTS (upper bounds)",
            "walk_source": "e198/e199's natural walk (STEP_L2 1.6544): t1 alive 0.3673, t2 alive 0.6911 -> t3 death 0.1230",
            "death_t": 3, "survival": surv_from_ledger(mir_ledger),
            "survival_source": "runs/e199/metrics.json organisms.mirabel.alive_ledger",
            "alternate_walk": None,
            "caveats": [
                "band kill-Ds are first-dead GRID points (upper bounds) => band median overstates => the multiple is a LOWER bound (biased toward arrival)",
                "shares its walk and its in-span DRAWS with R3 (one organism, two facts)",
                "ruler g-12 (novel geometry), same class as R1's",
            ],
        },
        {
            "id": "R3", "organism": "e193b (e193b_root.pt, fresh two-fact)",
            "arch": "6L/6H/192d/256ctx, 2,739,072 params (same organism as R2)",
            "fact": "ZEPHYRA",
            "ruler": f"ZEPHYRA install-60 {zeph_ruler['battery']} — THE FALLBACK RULER (g-12 read 0.1486 < 0.27 => frozen mechanical fallback to the max root read {zeph_ruler['root_read']:.4f})",
            "ruler_class": "g+0 trained (FALLBACK)",
            "edge": zeph_sign_kill,
            "edge_source": "runs/e193b/metrics.json adjudication.per_fact.ZEPHYRA.sign_kill (u0 = R2_SIGN, FIRST-DEAD GRID POINT)",
            "edge_kind": "first-dead grid point (upper bound; interpolated context recomputed in G_ARITH from e193b's committed rows)",
            "band_kill_Ds": zeph_band_list,
            "band_source": "runs/e193b/metrics.json inspan_range.ZEPHYRA.kill_Ds (the SAME draws as R2's — seeds 11911-3 — read by the ZEPHYRA ruler)",
            "band_instrument": "e193b D_GRID 27pts, FIRST-DEAD GRID POINTS (upper bounds)",
            "walk_source": "the SAME e198/e199 natural walk as R2, ZEPHYRA g+0 co-read: step-1 journal read 0.1629 <= 0.27 (the a_sign repro row 0.0526) -> death AT t=1 (e198 kill_zephyra_rederived: kind=kill, step=1)",
            "death_t": kz["step"], "survival": 0,
            "survival_source": "runs/e198/metrics.json phase0_walk_rebuild.kill_zephyra_rederived + walk_journal[0]['g+0']",
            "alternate_walk": None,
            "caveats": [
                "THE FALLBACK RULER: g+0 (trained geometry) — a CROSS-GEOMETRY row vs the g-12 family and R4's g-4; the fallback was frozen before compute in e193b, but the row's margin is not same-ruler-family comparable",
                "edge AND band are both grid upper bounds => the multiple's bias direction is UNDETERMINED",
                "the table's only within-organism, same-walk contrast (vs R2): margin 1.24x died at t=1 where 2.12x survived to t=3",
            ],
        },
        {
            "id": "R4", "organism": "org2-f2 'half' (e157_f2_consolidated.pt)",
            "arch": "4L/4H/128d/512ctx, 873,472 params (family 2)",
            "fact": "ZEPHYRA (f2)",
            "ruler": "install-60 g-4 (e193's frozen primary; root read 0.8872 — max committed)",
            "ruler_class": "g-4 trained",
            "edge": half_edge,
            "edge_source": "runs/e200/metrics.json profiles.root_u0.D_kill (u0 = e193's R2_SIGN, onset-grid interpolated; e193's own 27-pt bracket interpolates 0.5271 — a 0.4% grid class)",
            "edge_kind": "interpolated downcrossing (the onset instrument)",
            "band_kill_Ds": half_band,
            "band_source": "runs/e205/metrics.json bands.half.kill_Ds_fresh (seeds 12001-3 — e205's fresh band at this root)",
            "band_instrument": "onset grid 0.05..3.00, interpolated linear-in-D (the ONLY same-instrument band in the table)",
            "walk_source": "PRIMARY = the NATURAL-step walk (e196 phase0: step-1 gm 0.0068 <= 0.27 => death AT t=1); ALTERNATE = the half-step counterfactual (e197/e200: alive t1-t4, death t5 at 0.1648)",
            "death_t": half_nat_stop["step"], "survival": 0,
            "survival_source": "runs/e196/metrics.json phase0_walk_rebuild.stop (natural) + runs/e200/metrics.json alive_ledger (alternate)",
            "alternate_walk": {
                "kind": "half-step counterfactual (e197/e200 — the direction machinery at HALF the natural step)",
                "death_t": 5, "survival": surv_from_ledger(half_alt_ledger),
                "note": "THE FORK: under this reading R4 survives 4 steps and MARGIN-PREDICTS fails; reported, never adjudicated",
            },
            "caveats": [
                "THE FORK: the lineage's protocol walk dies at t=1; the half-step counterfactual survives to t=5 — both committed, the primary is the natural walk",
                "the band is root-level (step-size independent) while the fronts belong to the counterfactual construction (e197/e200's carried deviation)",
                "different architecture (873k vs 2.74M): absolute Ds are NOT cross-currency (e193's dual-currency note) — but the multiple is a RATIO and carries no unit",
                "ruler g-4 (trained geometry) — a third ruler class in the table",
            ],
        },
    ]
    # attach kj1 read for the G_WALKS gate
    rows[2]["journal_step1_g+0"] = kj1["g+0"]
    rows[2]["asign_repro_step1"] = e198["gates"]["G_REPRO"]["rows"]["s1_gm_ZEPHYRA"]["committed"]
    rows[0]["ledger_t1_read"] = org1_ledger[1]["ruler_read"]
    rows[0]["ledger_t2_read"] = org1_ledger[2]["ruler_read"]
    rows[1]["ledger_t1_read"] = mir_ledger[1]["ruler_read"]
    rows[1]["ledger_t2_read"] = mir_ledger[2]["ruler_read"]
    rows[1]["ledger_t3_read"] = mir_ledger[3]["ruler_read"]
    rows[3]["natural_step1_read"] = half_nat_s1
    rows[3]["alt_t1_read"] = half_alt_ledger[0]["ruler_read"]
    rows[3]["alt_t5_read"] = half_alt_ledger[4]["ruler_read"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rd = RUNS / ("e208_smoke" if args.smoke else "e208")
    rd.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": "e208_edge_multiple",
        "date": now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": bool(args.smoke),
        "envelope": {
            "device": "DESK CENSUS — CPU-only, NO torch, NO model compute (CUDA never touched)",
            "threads": "n/a (no compute threads; matplotlib Agg only)",
            "load_check_recorded_not_gating": True,
            "phases": "P0 registration -> P1 edges -> P2 bands -> P3 walks -> P4 g1bR survey -> P5 table+adjudication -> P6 figure; progressive writes",
        },
        "deviations": DEVIATIONS,
    }

    def write_partial(note):
        metrics["date"] = now_iso()
        metrics["phase"] = note
        (rd / "metrics.json").write_text(
            json.dumps(metrics, indent=1), encoding="utf-8")
        log(f"WROTE partial metrics ({note})")

    load0 = cpu_load_probe()
    metrics["envelope"]["cpu_load_pct_at_launch"] = load0
    log(f"E208 THE EDGE-MULTIPLE CENSUS (smoke={args.smoke}) -> {rd}")
    log(f"desk census: no torch, no GPU; load-check recorded (launch: {load0}%)")

    # ================= P1+P2+P3: load the committed parents (G_LOAD) =========
    src_names = ["e199", "e200", "e193b", "e196", "e198", "e205", "e_chart"]
    sources, prov = {}, {}
    for nm in src_names:
        p = RUNS / nm / "metrics.json"
        sources[nm] = load_metrics(nm)
        prov[nm] = {"file": str(p.relative_to(REPO)).replace("\\", "/"),
                    "md5": md5_file(p),
                    "experiment": sources[nm].get("experiment"),
                    "date": sources[nm].get("date")}
    metrics["parents_loaded"] = prov
    gates = {}

    rows = build_rows({k: sources.get(k) for k in
                       ["e199", "e200", "e193b", "e196", "e198", "e205", "e_chart"]})
    if args.smoke:
        rows = [r for r in rows if r["id"] in ("R1", "R2")]
    for r in rows:
        bm = band_median(r["band_kill_Ds"])
        r["band_median_stat"] = bm
        r["band_median"] = bm["median"]
        r["multiple"] = (r["edge"] / r["band_median"]
                         if (r["edge"] is not None and bm["median"]) else None)
    metrics["table_rows_loaded"] = [
        {k: r[k] for k in ["id", "organism", "fact", "ruler_class", "edge",
                           "band_kill_Ds", "band_median", "multiple",
                           "survival", "death_t"]} for r in rows]
    gates["G_LOAD"] = {
        "pass": True,
        "note": "every row's edge/band/walk loaded AT its committed path; parent file md5s recorded in parents_loaded",
        "rows": {r["id"]: {
            "edge_path": r["edge_source"], "band_path": r["band_source"],
            "walk_path": r["survival_source"]} for r in rows},
    }
    write_partial("P1-P3 loaded: edges + bands + walks")

    # ================= G_ARITH: the arithmetic gates =========================
    arith = {"pass": True, "checks": []}

    def check(name, ok, detail):
        arith["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            arith["pass"] = False

    # (a) e205's three committed multiples must reproduce from loaded parents
    e205_mult = sources["e205"]["normalization"]["edge_multiples"]
    e205_med = sources["e205"]["normalization"]["band_medians"]
    for rid, key in (("R1", "org1"), ("R2", "mirabel"), ("R4", "half")):
        cand = [x for x in rows if x["id"] == rid]
        if not cand:
            continue
        r = cand[0]
        if key in e205_mult:
            d = abs(r["multiple"] - e205_mult[key])
            check(f"{rid} multiple == e205 committed edge_multiples.{key}",
                  d < 1e-12, {"recomputed": r["multiple"],
                              "committed": e205_mult[key], "abs_diff": d})
            dm = abs(r["band_median"] - e205_med[key])
            check(f"{rid} band median == e205 committed band_medians.{key}",
                  dm < 1e-12, {"recomputed": r["band_median"],
                               "committed": e205_med[key], "abs_diff": dm})

    # (b) R3's median recomputed from the committed kill_Ds (fresh arithmetic)
    if any(r["id"] == "R3" for r in rows):
        r3 = next(x for x in rows if x["id"] == "R3")
        check("R3 band median == middle order statistic of [1.75, 0.74, 0.58]",
              r3["band_median"] == 0.74, {"recomputed": r3["band_median"]})
        check("R3 multiple == 0.92/0.74",
              abs(r3["multiple"] - 0.92 / 0.74) < 1e-12,
              {"recomputed": r3["multiple"]})
        # (c) R3's interpolated-edge context from e193b's committed R2_SIGN rows
        prof = sources["e193b"]["profiles"]["R2_SIGN"]
        b = [p for p in prof if p["D"] in (0.87, 0.92)]
        v87 = b[0]["gm_ZEPHYRA"]
        v92 = b[1]["gm_ZEPHYRA"]
        xi = interp_crossing(0.87, v87, 0.92, v92)
        r3["edge_interp_context"] = xi
        check("R3 edge bracket: 0.87 alive / 0.92 dead (e193b rows)",
              v87 > SHUT >= v92, {"gm_at_0.87": v87, "gm_at_0.92": v92,
                                  "interp_crossing": xi,
                                  "grid_gap_pct": (0.92 - xi) / xi * 100})
        r3["multiple_interp_context"] = xi / r3["band_median"]

    gates["G_ARITH"] = arith
    metrics["gates"] = gates
    write_partial("G_ARITH done")

    # ================= G_WALKS: the survival reads re-checked =================
    wk = {"pass": True, "checks": []}

    def wcheck(name, ok, detail):
        wk["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            wk["pass"] = False

    for r in rows:
        rid = r["id"]
        if rid == "R1":
            wcheck("R1 t1 alive (0.6786 > 0.27)", r["ledger_t1_read"] > SHUT,
                   {"read": r["ledger_t1_read"]})
            wcheck("R1 t2 death (9.83e-05 <= 0.27)", r["ledger_t2_read"] <= SHUT,
                   {"read": r["ledger_t2_read"]})
            wcheck("R1 survival == 1", r["survival"] == 1, {})
        if rid == "R2":
            wcheck("R2 t1,t2 alive; t3 death",
                   r["ledger_t1_read"] > SHUT and r["ledger_t2_read"] > SHUT
                   and r["ledger_t3_read"] <= SHUT,
                   {"t1": r["ledger_t1_read"], "t2": r["ledger_t2_read"],
                    "t3": r["ledger_t3_read"]})
            wcheck("R2 survival == 2", r["survival"] == 2, {})
        if rid == "R3":
            wcheck("R3 step-1 ZEPHYRA read dead (journal 0.1629 <= 0.27)",
                   r["journal_step1_g+0"] <= SHUT, {"journal": r["journal_step1_g+0"]})
            wcheck("R3 step-1 a_sign repro read dead (0.0526 <= 0.27)",
                   r["asign_repro_step1"] <= SHUT, {"asign": r["asign_repro_step1"]})
            wcheck("R3 e198 kill row: kind=kill, step=1",
                   r["death_t"] == 1, {"kill_zephyra_rederived.step": r["death_t"]})
            wcheck("R3 survival == 0", r["survival"] == 0, {})
        if rid == "R4":
            wcheck("R4 natural step-1 dead (0.0068 <= 0.27)",
                   r["natural_step1_read"] <= SHUT, {"read": r["natural_step1_read"]})
            wcheck("R4 natural survival == 0 (e196 stop step 1)",
                   r["survival"] == 0 and r["death_t"] == 1, {})
            wcheck("R4 half-step t1 alive (0.4192 > 0.27) [the fork]",
                   r["alt_t1_read"] > SHUT, {"read": r["alt_t1_read"]})
            wcheck("R4 half-step t5 death (0.1648 <= 0.27) [the fork]",
                   r["alt_t5_read"] <= SHUT, {"read": r["alt_t5_read"]})
    gates["G_WALKS"] = wk
    metrics["gates"] = gates
    write_partial("G_WALKS done")

    # ================= P4: the g1bR survey ===================================
    g1 = load_metrics("g1bR")
    blob = json.dumps(g1).lower()
    survey_keys = ["d_kill", "static", "ray", "inspan", "edge_multiple"]
    found = {k: (k in blob) for k in survey_keys}
    g1bR = {
        "file": "runs/g1bR/metrics.json",
        "md5": md5_file(RUNS / "g1bR" / "metrics.json"),
        "experiment": g1.get("experiment"),
        "verdict": g1.get("adjudication", {}).get("verdict"),
        "roots": ["g1bR_C10907_s300.pt", "g1bR_C10908_s300.pt",
                  "g1bR_W1_10907_s300.pt", "g1bR_W1_10908_s300.pt"],
        "instrument_key_survey": found,
        "status": "OUT OF CENSUS — the rays DO NOT EXIST: no u0/static kill-D "
                  "instrument and no in-span random band in the committed "
                  "metrics (batteries = the g1b ruler reads; the wash arms are "
                  "trained walks, not static-ray profiles). The four 2.74M "
                  "roots are real and checkpointed; a margin for them would "
                  "need a NEW instrument cell (e191/e192's machinery at those "
                  "roots) — named here as the census's named debt, not run.",
    }
    metrics["g1bR_survey"] = g1bR
    write_partial("P4 g1bR survey done")

    # ================= P5: the table + the adjudication ======================
    if not args.smoke:
        margins = [r["multiple"] for r in rows]
        survivals = [r["survival"] for r in rows]
        cls_gt2 = [r for r in rows if r["multiple"] > 2.0]
        cls_lt1 = [r for r in rows if r["multiple"] < 1.0]
        gray = [r for r in rows if 1.0 <= r["multiple"] <= 2.0]
        min_gt2 = min((r["survival"] for r in cls_gt2), default=None)
        max_lt1 = max((r["survival"] for r in cls_lt1), default=None)
        predicts_fires = (min_gt2 is not None and max_lt1 is not None
                          and min_gt2 > max_lt1)
        rho_primary = spearman(margins, survivals)

        # the fork: R4's alternate (half-step) survival
        r4 = next(r for r in rows if r["id"] == "R4")
        alt_surv = [r4["alternate_walk"]["survival"] if r["id"] == "R4"
                    else r["survival"] for r in rows]
        alt_min_gt2 = min((s for r, s in zip(rows, alt_surv) if r["multiple"] > 2.0),
                          default=None)
        alt_max_lt1 = max((s for r, s in zip(rows, alt_surv) if r["multiple"] < 1.0),
                          default=None)
        alt_predicts = (alt_min_gt2 is not None and alt_max_lt1 is not None
                        and alt_min_gt2 > alt_max_lt1)
        rho_alt = spearman(margins, alt_surv)

        # the 2x-line class observation (context, never adjudicated)
        first_wash = {r["id"]: (r["survival"] >= 1) for r in rows}
        two_line_consistent = all(
            (r["multiple"] > 2.0) == first_wash[r["id"]] for r in rows)

        decorr_fires = (not predicts_fires) and (rho_primary is not None
                                                 and rho_primary <= 0)
        if predicts_fires:
            verdict = "MARGIN-PREDICTS"
        elif decorr_fires:
            verdict = "MARGIN-DECORRELATED"
        else:
            verdict = "GRADED"

        metrics["census_table"] = [
            {"id": r["id"], "organism": r["organism"], "arch": r["arch"],
             "fact": r["fact"], "ruler": r["ruler"], "ruler_class": r["ruler_class"],
             "edge": r["edge"], "edge_kind": r["edge_kind"],
             "band_kill_Ds": r["band_kill_Ds"],
             "band_median": r["band_median"], "band_n": r["band_median_stat"]["n"],
             "band_censored": r["band_median_stat"]["n_censored"],
             "band_instrument": r["band_instrument"],
             "margin_multiple": r["multiple"],
             "survival_steps_natural": r["survival"], "death_t": r["death_t"],
             "alternate_walk": r["alternate_walk"],
             "class": (">2" if r["multiple"] > 2.0 else
                       ("<1" if r["multiple"] < 1.0 else "gray[1,2]")),
             "caveats": r["caveats"]} for r in rows]

        metrics["adjudication"] = {
            "bars": {
                "MARGIN-PREDICTS": {
                    "fires": predicts_fires,
                    "detail": {
                        "class_>2": {r["id"]: [r["multiple"], r["survival"]] for r in cls_gt2},
                        "class_<1": {r["id"]: [r["multiple"], r["survival"]] for r in cls_lt1},
                        "gray_[1,2]": {r["id"]: [r["multiple"], r["survival"]] for r in gray},
                        "min_survival_>2": min_gt2, "max_survival_<1": max_lt1,
                        "clause": "min(surv | margin>2) > max(surv | margin<1)",
                        "desk_forced": True,
                        "desk_forced_note": "the census is desk-complete from committed data; the firing was readable at registration and is reported as forced (e205's convention)",
                    },
                },
                "MARGIN-DECORRELATED": {
                    "fires": decorr_fires,
                    "detail": {
                        "spearman_primary": rho_primary,
                        "spearman_alternate_fork": rho_alt,
                        "clause": "fires iff PREDICTS fails AND Spearman(margin, survival) <= 0",
                    },
                },
                "GRADED": {"fires": verdict == "GRADED"},
            },
            "verdict": verdict,
            "composite_order": "MARGIN-PREDICTS -> MARGIN-DECORRELATED -> GRADED (first two mutually exclusive on the primary reading)",
            "the_fork": {
                "primary": "the half lineage's NATURAL-step walk (e196): death AT t=1 -> survival 0 -> PREDICTS fires",
                "alternate": f"the half-step counterfactual (e197/e200): survival 4 -> min(>2)={alt_min_gt2} vs max(<1)={alt_max_lt1} -> PREDICTS "
                             f"{'still fires' if alt_predicts else 'FAILS'}; Spearman {rho_alt:.3f}",
                "alternate_predicts_fires": alt_predicts,
                "note": "the dispatcher's registered reading (the natural walk, 'the half lineage's fact died at t=1') is the primary; the counterfactual is e197's own construction and is reported, never adjudicated",
            },
            "context_never_adjudicated": {
                "two_line_separation": {
                    "observation": "every margin > 2 row survived its first wash step; every margin < 2 row died AT its first wash step (4/4 rows, the gray-zone row included on the death side)",
                    "holds": two_line_consistent,
                    "note": "a sharper cut than the registered letter asked for; reported as context because it was not registered as a bar (no bar shopping)",
                },
                "not_an_ordering": "the margin does NOT order deaths inside the survivor class: R1 (3.72x) died at t=2 while R2 (2.12x) died at t=3 — class separation, not a graded survival law",
                "within_organism_pair": "R3 vs R2 (same organism, same walk, same draws): 1.24x died at t=1 where 2.12x survived to t=3 — the table's only same-walk margin contrast, margin-ordered",
            },
        }
        write_partial("P5 table + adjudication")

    # ================= P6: the figure ========================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.smoke:
        fig = plt.figure(figsize=(15.5, 8.0), dpi=130)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5], wspace=0.16)

        ax = fig.add_subplot(gs[0, 0])
        colors = {"R1": "#1f77b4", "R2": "#2ca02c", "R3": "#d62728", "R4": "#9467bd"}
        for r in rows:
            x, y = r["multiple"], r["survival"]
            ax.scatter(x, y, s=150, color=colors[r["id"]], zorder=5,
                       edgecolor="black", linewidth=0.8)
            dy = 0.13 if r["id"] != "R2" else -0.18
            ax.annotate(f"{r['id']} {r['fact']}\n{r['multiple']:.2f}x",
                        (x, y), xytext=(8, 14 if dy > 0 else -22),
                        textcoords="offset points", fontsize=8.5)
        # the fork: R4's alternate reading
        r4 = next(r for r in rows if r["id"] == "R4")
        ax.scatter([r4["multiple"]], [r4["alternate_walk"]["survival"]], s=120,
                   facecolor="none", edgecolor=colors["R4"], linewidth=1.8,
                   linestyle="--", zorder=5)
        ax.annotate("R4 ALT: half-step\nwalk, surv 4 (the fork)",
                    (r4["multiple"], r4["alternate_walk"]["survival"]),
                    xytext=(10, -6), textcoords="offset points", fontsize=8,
                    color="#9467bd")
        ax.plot([r4["multiple"], r4["multiple"]],
                [r4["survival"], r4["alternate_walk"]["survival"]],
                ls=":", color=colors["R4"], lw=1.2, zorder=4)
        ax.axvline(1.0, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(2.0, color="black", lw=1.2, ls="--", alpha=0.9)
        ax.text(2.0, 4.45, " 2x line", fontsize=9, va="top")
        ax.text(1.0, 4.45, " 1x line", fontsize=9, va="top", color="gray")
        ax.axvspan(2.0, 4.4, color="#2ca02c", alpha=0.06)
        ax.axvspan(0.4, 1.0, color="#d62728", alpha=0.06)
        ax.axvspan(1.0, 2.0, color="gray", alpha=0.07)
        ax.text(3.15, 4.32, "margin > 2", fontsize=8.5, ha="center", color="#2ca02c")
        ax.text(1.5, 4.32, "gray zone", fontsize=8.5, ha="center", color="dimgray")
        ax.text(0.7, 4.32, "margin < 1", fontsize=8.5, ha="center", color="#d62728")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.616, 1.0, 1.243, 2.0, 2.116, 3.72])
        ax.set_xticklabels(["0.5", "0.62", "1", "1.24", "2", "2.12", "3.72"],
                           fontsize=8.5)
        ax.set_ylim(-0.45, 4.75)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_xlabel("edge multiple  =  u0 static kill-D / in-span band median (log)",
                      fontsize=10)
        ax.set_ylabel("natural-walk steps survived", fontsize=10)
        ax.set_title("E208 the edge-multiple census: margin vs survival\n"
                     "(filled = natural/protocol walk; open = R4's half-step counterfactual — the fork)",
                     fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5)

        axt = fig.add_subplot(gs[0, 1])
        axt.axis("off")
        col_heads = ["row", "organism / fact", "ruler", "edge", "band med (n)",
                     "MARGIN", "walk death t (surv)", "class"]
        table_rows = []
        for r in rows:
            bm = f"{r['band_median']:.3g} (n={r['band_median_stat']['n']}"
            bm += f", {r['band_median_stat']['n_censored']} cens)" if r["band_median_stat"]["n_censored"] else ")"
            table_rows.append([
                r["id"],
                f"{r['organism'].split(' (')[0]}\n{r['fact']}",
                r["ruler_class"],
                f"{r['edge']:.4g}",
                bm,
                f"{r['multiple']:.3f}x",
                f"t={r['death_t']} (surv {r['survival']})"
                + (f"\nALT t=5 (surv {r['alternate_walk']['survival']})"
                   if r.get("alternate_walk") else ""),
                (">2" if r["multiple"] > 2 else
                 ("<1" if r["multiple"] < 1 else "gray")),
            ])
        verdict = metrics.get("adjudication", {}).get("verdict", "SMOKE")
        tbl = axt.table(cellText=table_rows, colLabels=col_heads,
                        cellLoc="center", loc="upper center",
                        colWidths=[0.05, 0.19, 0.13, 0.09, 0.15, 0.10, 0.19, 0.08])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.3)
        tbl.scale(1, 1.75)
        for (rowi, coli), cell in tbl.get_celld().items():
            cell.set_edgecolor("#bbbbbb")
            if rowi == 0:
                cell.set_facecolor("#e8e8e8")
            elif rowi % 2 == 0:
                cell.set_facecolor("#f7f7f7")
        axt.set_title(
            f"the table (verdict: {verdict} — desk-forced, disclosed; "
            f"MARGIN-PREDICTS letter: min surv of >2 = {min_gt2} > max surv of <1 = {max_lt1})\n"
            "caveats per row in metrics.json census_table; g1bR's 2.74M roots: NO committed rays — out of census",
            fontsize=9.5, pad=18)

        fig.suptitle("E208 — the memory-vs-noise margin census (T173's scalar; all values loaded committed)",
                     fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        out = rd / "margin_vs_survival.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        metrics["figure"] = str(out.relative_to(REPO)).replace("\\", "/")
        log(f"figure -> {out}")

    # ================= honesty + finalize ====================================
    metrics["honesty"] = {
        "bands_own_ns": "every band is n=3 draws of ONE 20-step wash-history realization; R1's carries 1-of-3 right-censored >1.5 (median censoring-robust); R2/R3's are first-dead GRID points (upper bounds — R2's multiple is a lower bound, biased toward survival); R3's edge is ALSO a grid upper bound so its multiple's bias is UNDETERMINED; R4's band is the only same-instrument (interpolated onset grid) band",
        "cross_ruler_rows": "three ruler classes in one table (g-12 novel x2, g+0 trained-fallback, g-4 trained) — within a row the edge and band share the ruler; ACROSS rows the rulers differ (T153/T155); R3 is the flagged fallback-ruler row; R2 vs R3 isolates the ruler's contribution (same draws: band 0.92 under g-12 vs 0.74 under g+0)",
        "single_realizations": "every survival number is ONE realized walk; rows R2/R3 share a single walk; margins are single-u0, single-span-realization numbers — nothing here is a distribution; the 2-3x draw lottery T155 committed would move any band median on redraw",
        "the_fork": "R4's two committed walks disagree by four survival steps (natural death t=1 vs counterfactual death t=5); the primary is the natural walk; the verdict flips to MARGIN-DECORRELATED under the alternate — this is the table's single biggest fragility and it is on the face of the adjudication",
        "logits_prediction_check": "the census is behavior reads (battery p(Z) shut-bar crossings) on committed data; NO intervening was done in this cell — the margin's causal claim is only as good as the walks it is compared against (single realizations)",
        "nothing_guaranteed": "the openness is the point: n=4 rows, 3 organisms, 1 shared walk, one gray-zone row; the 2x-line 4/4 separation is context, NOT a registered bar; g1bR's roots are the named debt (a new instrument cell would be needed to extend the census)",
        "desk_forced": "like e205, the verdict was desk-computable at registration; the cell's value is the gated assembly, the fork made explicit, the g1bR survey and the map itself",
    }
    metrics["provenance"] = {
        "edges": {"R1": "runs/e199/metrics.json onset_curves.org1[0]",
                  "R2": "runs/e199/metrics.json onset_curves.mirabel[0]",
                  "R3": "runs/e193b/metrics.json adjudication.per_fact.ZEPHYRA.sign_kill",
                  "R4": "runs/e200/metrics.json profiles.root_u0"},
        "bands": {"R1": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D (via e205's committed copy, cross-checked)",
                  "R2": "runs/e193b/metrics.json inspan_range.MIRABEL.kill_Ds",
                  "R3": "runs/e193b/metrics.json inspan_range.ZEPHYRA.kill_Ds",
                  "R4": "runs/e205/metrics.json bands.half.kill_Ds_fresh"},
        "walks": {"R1": "runs/e199/metrics.json organisms.org1.alive_ledger",
                  "R2": "runs/e199/metrics.json organisms.mirabel.alive_ledger",
                  "R3": "runs/e198/metrics.json phase0_walk_rebuild.kill_zephyra_rederived (+ walk_journal[0])",
                  "R4 primary": "runs/e196/metrics.json phase0_walk_rebuild.stop",
                  "R4 alternate": "runs/e200/metrics.json alive_ledger"},
        "g1bR": "runs/g1bR/metrics.json (key survey; OUT of census)",
        "parent_file_md5s": {k: v["md5"] for k, v in prov.items()},
    }
    metrics["status"] = ("SMOKE — nothing adjudicated" if args.smoke else
                         "COMPLETE — adjudicated (this write replaces all PARTIAL progressive writes)")
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=1),
                                     encoding="utf-8")
    ok = all(g.get("pass", True) for g in gates.values())
    log(f"DONE smoke={args.smoke} gates={'PASS' if ok else 'FAIL'} "
        f"verdict={metrics.get('adjudication', {}).get('verdict', 'SMOKE')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
