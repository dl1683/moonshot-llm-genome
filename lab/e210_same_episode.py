"""E210 — THE SAME-EPISODE MARGIN (e209's repair fork).

WHY. e209's extension broke the 2x class line — and the break's anatomy
(T179) was an EPISODE MISMATCH baked into the frozen letter: the new rows
joined a margin measured at the s300 SETTLED state (300 walled steps into
the wash) to a survival clock from the lineage's FIRST wash episode (the
C-arms washed the PRISTINE e131 root). The co-read closed the loop: at
every s300 root the fact died at its first unwalled step with an
at-or-below-noise margin. The honest question the break left open: does
the class line survive when margin AND clock come from the SAME episode?

THE REPAIR (this cell): re-measure the margins WITHIN ONE EPISODE — at
the PRISTINE roots, the episodes the clocks actually measured:
  (1) THE ANCHOR: the pristine e131 root's own u0 sign-ray edge + its
      in-span band — R1's committed values (edge 2.2699 / band 0.6101,
      margin 3.7207x) LOADED as the anchor — joined to the THREE
      committed first-episode clocks at that root (the C-arms: seed
      10902 in g1b, seeds 10907/10908 in g1bR; e199's org1 walk is the
      same seed-10902 episode, co-cited);
  (2) THE NEW ROWS — the OTHER first-episode lineages:
      * the e193/e197 f2 organisms' margin AT THE PRISTINE f2 ROOT
        (edge committed from e193's rays via e200's interpolation; the
        f2 band 0.8526 from e205) joined to THEIR OWN committed
        first-episode clocks: the natural full-step walk (e196: death
        AT t=1) and the half-step counterfactual walk (e197/e200: alive
        t1-t4, death t5) — THE FORK made a row;
      * e193b's MIRABEL/ZEPHYRA at THEIR pristine root (committed
        margins) joined to their committed clock (the one e198/e199
        walk: MIRABEL survived to t=3, ZEPHYRA died at t=1).

THE CELL: a DESK TABLE on committed data — ZERO model compute, ZERO
torch (the owner envelope's desk form; nothing needed recomputing: every
margin, every band, every clock already exists at a committed path).
Every number LOADED at its path with md5s recorded; the arithmetic
(margins = edge/band, medians, survivals = death_t - 1, the C-arm
first-downcrossing re-derivation) recomputed and gated; every row's
SAME-EPISODE identity gated (margin root == clock root).

REGISTERED BARS (frozen here, before assembly; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - SAME-EPISODE-HOLDS: "fires if the 2x class line separates every
    within-episode row (margin > 2 surviving step 1; < 1 dying) — the
    scalar's class claim REBUILT on the honest episode; the e208 finding
    restored."
  - SAME-EPISODE-BREAKS: "fires if the within-episode rows still
    violate — the class line is dead at any episode; the margin is a
    descriptor, full stop."
  - GRADED: "any partial — the table verbatim."

OPERATIONALIZATIONS (frozen before assembly; they fix the clauses, they
do not move the bars):
  * row = (pristine root, fact, committed first-episode clock) with
    margin AND clock from the SAME first wash episode: margin = the
    root's u0 sign-ray edge / its in-span band median (both AT the
    pristine root, loaded committed); clock = a committed UNWALLED wash
    of that same pristine root. Registered table (n=7): W1 e131/seed
    10902, W2 e131/C10907, W3 e131/C10908, W4 f2/natural (e196), W5
    f2/half-step (e197 — THE FORK made a row: the dispatch names the
    e193/e197 lineages and their OWN clocks; its counterfactual
    construction is carried as the row's caveat), W6 e193b/MIRABEL, W7
    e193b/ZEPHYRA.
  * survival = wash steps survived (ruler read > 0.27); death_t = first
    dead step; the class-clause atom "surviving step 1" reads the
    committed step-1 read (exact at every row; W2's death is bracketed
    (2,4] on its ckpt clock but its step-1 read 0.8174 is committed).
  * violation = (margin > 2 AND survival == 0) OR (margin < 1 AND
    survival >= 1); gray [1,2] reported, never adjudicated (e208/e209's
    frozen convention). SAME-EPISODE-HOLDS fires iff NO defined-class
    row anywhere violates; SAME-EPISODE-BREAKS fires iff >= 1
    defined-class row violates; GRADED = any residual. Composite order
    HOLDS / BREAKS / GRADED (first two mutually exclusive by
    construction).
  * DESK-FORCED DISCLOSURE (e205/e208/e209's convention): all seven
    rows' margins and clocks were committed before this cell — the
    firing was readable at registration and is reported as forced. The
    HINGE, disclosed up front: the f2 margin (0.616x, class < 1) faces
    TWO committed clocks at its root — the natural walk dies at t=1
    (consistent) and the half-step counterfactual survives to t=5 (a
    VIOLATION). The dispatch names e197's clock as one of the f2
    lineages' own first-episode clocks, so W5 is a registered row: the
    expected firing is SAME-EPISODE-BREAKS via W5. The sensitivity read
    (the n=6 natural-clock table, e208's fork convention) is reported
    as context, never adjudicated. No bar shopping either way.
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
SHUT = 0.27

REGISTERED_BARS = {
    "SAME-EPISODE-HOLDS": "SAME-EPISODE-HOLDS: \"fires if the 2x class line separates every within-episode row (margin > 2 surviving step 1; < 1 dying) — the scalar's class claim REBUILT on the honest episode; the e208 finding restored.\"",
    "SAME-EPISODE-BREAKS": "SAME-EPISODE-BREAKS: \"fires if the within-episode rows still violate — the class line is dead at any episode; the margin is a descriptor, full stop.\"",
    "GRADED": "GRADED: \"any partial — the table verbatim.\"",
    "operationalizations": (
        "row = (pristine root, fact, committed first-episode clock) with margin AND "
        "clock from the SAME first wash episode; margin = the pristine root's u0 "
        "sign-ray edge / its in-span band median (both AT the root, loaded "
        "committed); clock = a committed UNWALLED wash of that same pristine root "
        "(survival = steps with ruler read > 0.27; death_t = first dead step); "
        "registered table n=7: W1 e131/seed10902 (anchor, R1's committed "
        "2.2699/0.6101), W2 e131/C10907, W3 e131/C10908, W4 f2/natural (e196), "
        "W5 f2/half-step (e197/e200 — THE FORK made a row), W6 e193b/MIRABEL, "
        "W7 e193b/ZEPHYRA (gray); violation = (margin > 2 AND survival == 0) OR "
        "(margin < 1 AND survival >= 1); gray [1,2] reported never adjudicated; "
        "HOLDS fires iff no defined-class row violates; BREAKS fires iff >= 1 "
        "defined-class row violates; GRADED = residual; composite order HOLDS / "
        "BREAKS / GRADED (first two mutually exclusive by construction)."
    ),
    "desk_forced_disclosure": (
        "all seven rows' margins and clocks were committed before this cell — the "
        "firing was readable at registration and is reported as forced (e205/e208/"
        "e209's convention). THE HINGE: the f2 margin 0.616x (< 1) faces two "
        "committed clocks at its pristine root — natural death t=1 (consistent) vs "
        "half-step counterfactual survival 4 (violation); the dispatch names "
        "e197's clock among the lineages' own first-episode clocks, so W5 is a "
        "registered row and the expected firing is SAME-EPISODE-BREAKS via W5; the "
        "n=6 natural-clock sensitivity (e208's fork convention) is context, never "
        "adjudicated. No bar shopping."
    ),
    "registration": "the dispatch's registration IS the registration (the three bars quoted verbatim in the module docstring and here, frozen before assembly). Adjudicate against exactly this; no bar shopping.",
}

DEVIATIONS = [
    "DESK TABLE, ZERO MODEL COMPUTE: no torch import, no GPU, no training — every number loaded from a committed metrics.json at its recorded path (the owner envelope's desk form; NOTHING needed recomputing: every margin, band and clock already existed committed — the repair is the ASSEMBLY, not a new read).",
    "W5 IS THE FORK MADE A ROW (the table's hinge, disclosed): the f2 root's margin is ONE number joined to TWO committed first-episode clocks — e196's natural full-step walk (death AT t=1) and e197/e200's half-step counterfactual (alive t1-t4, death t5). e208 froze the counterfactual as never-adjudicated; THIS registration (the dispatch names the e193/e197 lineages' OWN clocks) makes it a row, carries its construction as its caveat (each step's displacement halved: 0.458 vs 0.916 L2; the band is root-level, step-size independent), and adjudicates the letter on the full n=7 table; the n=6 natural-clock sensitivity is reported as context, never adjudicated.",
    "THE ANCHOR SHARES ONE MARGIN ACROSS THREE CLOCKS: the e131 root's margin (R1's committed 2.2699/0.6101 -> 3.7207x) is one number read against the root's THREE committed first-episode wash seeds (10902/10907/10908) — the class line's survivor side is tested against three noise draws, not one; e199's org1 walk is the same seed-10902 episode (co-cited, CPU texture 5e-4 off the CUDA C-arm read; identical adjudication).",
    "W2's DEATH IS BRACKETED (2,4] on g1bR's ckpt clock {1,2,4,...}: +2 read 0.4611 alive, +4 read 0.1631 dead; the step-3 read was never committed) — carried as survival 3 / death_t 4 with the bracket; the class-clause atom (surviving step 1) is EXACT: the committed +1 read 0.8174 > 0.27. W3's +1 read 0.3227 is the table's thinnest survivor margin (1.20x the 0.27 bar) — disclosed: a 0.05-lower draw would have violated from the survivor side.",
    "RULER CAVEATS PER ROW (T153/T155, carried): the e131 rows' ruler is the NOVEL g-12 geometry; the f2 rows' is TRAINED g-4; W7's is the TRAINED g+0 FALLBACK (cross-geometry row, grid upper bounds both sides — bias undetermined); W6's band is first-dead grid points (upper bound => its 2.12x is a LOWER bound, conservative for the > 2 class); within every row edge and band share the ruler.",
    "SINGLE REALIZATIONS (carried verbatim from e208/e209): every band is n=3 draws of ONE 20-step wash-history realization; every clock is ONE realized walk; margins are single-u0 numbers — the 2-3x draw lottery (T155) would move any band median on redraw; nothing here is a distribution.",
    "e209's s300 rows (R5/R6/R7) are OUT of this table BY CONSTRUCTION (their margins are mid-episode reads — the mismatch this cell repairs); they are loaded in G_E209 only to gate the anatomy this table answers.",
    "the load-check is recorded, not gating (e204/e205's convention).",
    "Smoke mode: loads W1/W4 only, stamps SMOKE, nothing adjudicated.",
]


def log(msg: str):
    print(f"[e210 {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


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


def first_downcross(g_m12: dict, bar: float = SHUT) -> dict:
    """Re-derive a C-arm death clock from the committed per-step g-12 dict:
    first step (ascending, steps as ints) with read <= bar."""
    steps = sorted(g_m12, key=int)
    death_t, death_read = None, None
    for s in steps:
        if g_m12[s] <= bar:
            death_t, death_read = int(s), g_m12[s]
            break
    return {"death_t": death_t, "death_read": death_read,
            "steps_committed": [int(s) for s in steps],
            "step1_read": g_m12[steps[0]],
            "survival": (None if death_t is None else death_t - 1)}


def build_table(sources: dict) -> list:
    """Assemble the n=7 within-episode table. Every value is extracted AT
    ITS COMMITTED PATH — no recompute."""
    e199, e200, e193b = sources["e199"], sources["e200"], sources["e193b"]
    e196, e197, e198 = sources["e196"], sources["e197"], sources["e198"]
    e205, echart = sources["e205"], sources["e_chart"]
    g1b, g1bR = sources["g1b"], sources["g1bR"]

    # ---- the anchor margin at the PRISTINE e131 root (R1 committed) --------
    org1_curve = e199["onset_curves"]["org1"]
    e131_edge = org1_curve[0]["D_kill"]
    org1_band_ds = echart["partB_subspace"]["e131"]["fine_D_summary"]["inspan_thr_D"]
    e131_band_list = [org1_band_ds["11601"], org1_band_ds["11602"], org1_band_ds["11603"]]
    org1_ledger = e199["organisms"]["org1"]["alive_ledger"]

    # ---- the f2 root margin (e193's rays -> e200 interp; e205's band) ------
    f2_edge = e200["profiles"]["root_u0"]["D_kill"]
    f2_band_list = e205["bands"]["half"]["kill_Ds_fresh"]
    f2_nat_stop = e196["phase0_walk_rebuild"]["stop"]
    f2_nat_s1 = e196["phase0_walk_rebuild"]["walk_journal"][0]["gm"]
    f2_half_stop = e197["phase1_alive_walk"]["stop"]
    f2_half_j = e197["phase1_alive_walk"]["journal"]
    f2_half_alt_ledger = e200["alive_ledger"]

    # ---- the e193b margins (MIRABEL + ZEPHYRA, committed) ------------------
    mir_curve = e199["onset_curves"]["mirabel"]
    mir_edge = mir_curve[0]["D_kill"]
    mir_band_list = e193b["inspan_range"]["MIRABEL"]["kill_Ds"]
    mir_ledger = e199["organisms"]["mirabel"]["alive_ledger"]
    zeph_band_list = e193b["inspan_range"]["ZEPHYRA"]["kill_Ds"]
    zeph_sign_kill = e193b["adjudication"]["per_fact"]["ZEPHYRA"]["sign_kill"]
    zeph_ruler = e193b["rulers"]["primary"]["ZEPHYRA"]
    kz = e198["phase0_walk_rebuild"]["kill_zephyra_rederived"]
    kj1 = e198["phase0_walk_rebuild"]["walk_journal"][0]

    # ---- the three C-arm clocks at the PRISTINE e131 root ------------------
    arm_c = g1b["arms"]["C"]["traj"]
    c10902 = {str(r["step"]): r["g_m12_mean_pz"] for r in arm_c if "g_m12_mean_pz" in r}
    c10907 = g1bR["adjudication"]["wall"]["C10907"]["g_m12"]
    c10908 = g1bR["adjudication"]["wall"]["C10908"]["g_m12"]

    def surv_from_ledger(ledger):
        n = 0
        for e in ledger:
            if e.get("t", 1) == 0:
                continue
            if e.get("alive"):
                n += 1
            else:
                break
        return n

    e131_margin_note = ("R1's committed values LOADED as the anchor "
                        "(edge 2.2699 / band 0.6101 -> 3.7207x)")

    rows = [
        {   # W1 — anchor: e131 root, seed 10902's own first-episode clock
            "id": "W1",
            "root": "org1 PRISTINE (e131_consolidated_e113.pt)",
            "fact": "ZEPHYRA", "lineage": "org1 / seed 10902 (g1b's C arm)",
            "ruler_class": "g-12 novel",
            "margin": None,  # filled by caller (anchor)
            "edge": e131_edge,
            "edge_source": "runs/e199/metrics.json onset_curves.org1[0].D_kill (u0 = e192's R2, interpolated)",
            "band_kill_Ds": e131_band_list,
            "band_source": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D (seeds 11601-3; e205's promoted instrument)",
            "clock_kind": "unwalled C-arm wash of the pristine root (natural step)",
            "clock_source": "runs/g1b/metrics.json arms.C.traj (+ g1bR reference.C_g_m12) — co-cited: runs/e199/metrics.json organisms.org1.alive_ledger (the same seed-10902 episode, CPU texture)",
            "step1_read": c10902["1"], "death_t": None, "survival": None,
            "g_m12": c10902,
            "co_read_walk": {"t1": org1_ledger[1]["ruler_read"],
                             "t2": org1_ledger[2]["ruler_read"]},
            "anchor": True, "anchor_note": e131_margin_note,
            "caveats": [
                "THE ANCHOR: the margin e208's R1 committed (3.7207x) — the episode the C-arm clock measured",
                "one margin read against the root's three wash seeds (W1/W2/W3 share it — the survivor side is tested on three noise draws)",
                "ruler = NOVEL g-12 geometry (T153); band n=3 with 1 right-censored (>1.5, median censoring-robust)",
            ],
        },
        {   # W2 — e131 root, seed 10907's clock
            "id": "W2",
            "root": "org1 PRISTINE (e131_consolidated_e113.pt)",
            "fact": "ZEPHYRA", "lineage": "org1 / seed 10907 (g1bR's C10907 arm)",
            "ruler_class": "g-12 novel",
            "margin": None,
            "edge": e131_edge,
            "edge_source": "runs/e199/metrics.json onset_curves.org1[0].D_kill",
            "band_kill_Ds": e131_band_list,
            "band_source": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D",
            "clock_kind": "unwalled C-arm wash of the pristine root (natural step)",
            "clock_source": "runs/g1bR/metrics.json adjudication.wall.C10907.g_m12",
            "step1_read": c10907["1"], "death_t": None, "survival": None,
            "g_m12": c10907,
            "bracket": "(2,4] on the ckpt clock {1,2,4,...}: +2 alive 0.4611, +4 dead 0.1631 — the step-3 read was never committed; survival 3 is the bracket's floor-consistent carry (e209's convention)",
            "caveats": [
                "death BRACKETED (2,4] — but the class-clause atom (surviving step 1) is EXACT: committed +1 read 0.8174 > 0.27",
                "one margin read against the root's three wash seeds (shares W1/W3's margin)",
                "ruler = NOVEL g-12 geometry; band as W1's",
            ],
        },
        {   # W3 — e131 root, seed 10908's clock
            "id": "W3",
            "root": "org1 PRISTINE (e131_consolidated_e113.pt)",
            "fact": "ZEPHYRA", "lineage": "org1 / seed 10908 (g1bR's C10908 arm)",
            "ruler_class": "g-12 novel",
            "margin": None,
            "edge": e131_edge,
            "edge_source": "runs/e199/metrics.json onset_curves.org1[0].D_kill",
            "band_kill_Ds": e131_band_list,
            "band_source": "runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D",
            "clock_kind": "unwalled C-arm wash of the pristine root (natural step)",
            "clock_source": "runs/g1bR/metrics.json adjudication.wall.C10908.g_m12",
            "step1_read": c10908["1"], "death_t": None, "survival": None,
            "g_m12": c10908,
            "caveats": [
                "the table's thinnest survivor read: +1 read 0.3227 is 1.20x the 0.27 bar — a 0.05-lower draw would have VIOLATED from the survivor side (the class line's survivor margin at this root is noise-thin; disclosed, never adjudicated)",
                "one margin read against the root's three wash seeds (shares W1/W2's margin)",
                "ruler = NOVEL g-12 geometry; band as W1's",
            ],
        },
        {   # W4 — f2 root, the natural walk (e193's lineage)
            "id": "W4",
            "root": "org2-f2 PRISTINE (e157_f2_consolidated.pt)",
            "fact": "ZEPHYRA (f2)", "lineage": "org2-f2 dead lineage (e193 -> e196, natural step)",
            "ruler_class": "g-4 trained",
            "margin": None,
            "edge": f2_edge,
            "edge_source": "runs/e200/metrics.json profiles.root_u0.D_kill (u0 = e193's R2_SIGN, onset-grid interpolated; e193's own 27-pt bracket interpolates 0.5271 — a 0.4% grid class)",
            "band_kill_Ds": f2_band_list,
            "band_source": "runs/e205/metrics.json bands.half.kill_Ds_fresh (seeds 12001-3 — e205's fresh band at this root)",
            "clock_kind": "natural full-step wash of the pristine root",
            "clock_source": "runs/e196/metrics.json phase0_walk_rebuild.stop (+ walk_journal[0].gm)",
            "step1_read": f2_nat_s1, "death_t": f2_nat_stop["step"],
            "survival": f2_nat_stop["step"] - 1,
            "caveats": [
                "the f2 margin is ONE root-level number shared with W5 (the same pristine root's edge/band; the fork is in the CLOCK, not the margin)",
                "the band is root-level (step-size independent); ruler = TRAINED g-4 (max committed root read 0.8872)",
                "different architecture (873k vs 2.74M): absolute Ds are not cross-currency — but the margin is a RATIO and carries no unit",
            ],
        },
        {   # W5 — f2 root, the half-step counterfactual (e197's lineage): THE FORK ROW
            "id": "W5",
            "root": "org2-f2 PRISTINE (e157_f2_consolidated.pt)",
            "fact": "ZEPHYRA (f2)", "lineage": "org2-f2 half lineage (e197/e200, half-step counterfactual)",
            "ruler_class": "g-4 trained",
            "margin": None,
            "edge": f2_edge,
            "edge_source": "runs/e200/metrics.json profiles.root_u0.D_kill (same u0 as W4's)",
            "band_kill_Ds": f2_band_list,
            "band_source": "runs/e205/metrics.json bands.half.kill_Ds_fresh (same band as W4's)",
            "clock_kind": "HALF-STEP counterfactual wash of the pristine root (each step's displacement halved: 0.458 vs 0.916 L2)",
            "clock_source": "runs/e197/metrics.json phase1_alive_walk.stop (+ journal) — co-cited: runs/e200/metrics.json alive_ledger (the same walk rebuilt)",
            "step1_read": f2_half_j[0]["gm"], "death_t": f2_half_stop["step"],
            "survival": f2_half_stop["step"] - 1,
            "co_read_walk": {"t1": f2_half_alt_ledger[0]["ruler_read"],
                             "t5": f2_half_alt_ledger[4]["ruler_read"]},
            "fork_row": True,
            "caveats": [
                "THE FORK MADE A ROW: e208 froze this counterfactual as never-adjudicated; THIS registration (the dispatch names the e193/e197 lineages' OWN clocks) adjudicates it — the verdict's HINGE",
                "the construction halves each step's displacement (the same root, the same noise ball, traversed at half speed); the band is root-level (step-size independent); the ruler's 0.27 bar is step-size independent",
                "shares W4's margin exactly (one root-level number, two committed clocks)",
            ],
        },
        {   # W6 — e193b root, MIRABEL
            "id": "W6",
            "root": "e193b PRISTINE (e193b_root.pt, fresh two-fact)",
            "fact": "MIRABEL", "lineage": "e193b / the e198-e199 natural walk",
            "ruler_class": "g-12 novel",
            "margin": None,
            "edge": mir_edge,
            "edge_source": "runs/e199/metrics.json onset_curves.mirabel[0].D_kill (u0 = e193b's R2_SIGN, interpolated)",
            "band_kill_Ds": mir_band_list,
            "band_source": "runs/e193b/metrics.json inspan_range.MIRABEL.kill_Ds (seeds 11911-3)",
            "clock_kind": "natural full-step wash of the pristine root",
            "clock_source": "runs/e199/metrics.json organisms.mirabel.alive_ledger",
            "step1_read": mir_ledger[1]["ruler_read"], "death_t": 3,
            "survival": surv_from_ledger(mir_ledger),
            "caveats": [
                "band = first-dead GRID points (upper bounds) => the 2.12x margin is a LOWER bound (biased toward arrival — conservative for its > 2 class)",
                "shares its walk and in-span draws with W7 (one organism, two facts — the table's only within-organism contrast)",
                "ruler g-12 (novel geometry), same class as the e131 rows'",
            ],
        },
        {   # W7 — e193b root, ZEPHYRA under the fallback ruler (GRAY)
            "id": "W7",
            "root": "e193b PRISTINE (e193b_root.pt, fresh two-fact)",
            "fact": "ZEPHYRA", "lineage": "e193b / the e198-e199 natural walk",
            "ruler_class": "g+0 trained (FALLBACK)",
            "margin": None,
            "edge": zeph_sign_kill,
            "edge_source": "runs/e193b/metrics.json adjudication.per_fact.ZEPHYRA.sign_kill (u0 = R2_SIGN, FIRST-DEAD GRID POINT)",
            "band_kill_Ds": zeph_band_list,
            "band_source": "runs/e193b/metrics.json inspan_range.ZEPHYRA.kill_Ds (the SAME draws as W6's — read by the ZEPHYRA ruler)",
            "clock_kind": "natural full-step wash of the pristine root",
            "clock_source": "runs/e198/metrics.json phase0_walk_rebuild.kill_zephyra_rederived (+ walk_journal[0])",
            "step1_read": kj1["gm_ZEPHYRA"], "death_t": kz["step"], "survival": 0,
            "journal_step1_g+0": kj1["g+0"],
            "caveats": [
                "THE FALLBACK RULER: g+0 (g-12 read 0.1486 < 0.27 at the root) — a CROSS-GEOMETRY row; frozen before compute in e193b",
                "edge AND band are both grid upper bounds => the 1.24x margin's bias is UNDETERMINED; GRAY [1,2] — reported, never adjudicated",
                "the table's only same-walk margin contrast (vs W6): 1.24x died at t=1 where 2.12x survived to t=3",
            ],
        },
    ]
    # attach extra gate fields
    rows[0]["co_read_walk_ledger_t1"] = org1_ledger[1]["ruler_read"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rd = RUNS / ("e210_smoke" if args.smoke else "e210")
    rd.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": "e210_same_episode",
        "date": now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": bool(args.smoke),
        "envelope": {
            "device": "DESK TABLE — CPU-only, NO torch, NO model compute (CUDA never touched)",
            "threads": "n/a (no compute threads; matplotlib Agg only)",
            "load_check_recorded_not_gating": True,
            "phases": "P0 registration -> P1 parents loaded (G_LOAD) -> P2 margins (G_ARITH) -> P3 episode identity (G_EPISODE) -> P4 clocks (G_WALKS) -> P5 table + adjudication -> P6 figure; progressive writes",
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
    log(f"E210 THE SAME-EPISODE MARGIN (smoke={args.smoke}) -> {rd}")
    log(f"desk cell: no torch, no GPU; load-check recorded (launch: {load0}%)")

    # ================= P1: load the committed parents (G_LOAD) ==============
    src_names = ["e199", "e200", "e193b", "e196", "e197", "e198", "e205",
                 "e_chart", "g1b", "g1bR", "e208", "e209"]
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

    rows = build_table(sources)
    if args.smoke:
        rows = [r for r in rows if r["id"] in ("W1", "W4")]

    # the margins: loaded medians -> margin = edge / band (arithmetic gated next)
    for r in rows:
        bm = band_median(r["band_kill_Ds"])
        r["band_median_stat"] = bm
        r["band_median"] = bm["median"]
        r["margin"] = (r["edge"] / bm["median"]
                       if (r["edge"] is not None and bm["median"]) else None)

    gates["G_LOAD"] = {
        "pass": True,
        "note": "every row's edge/band/clock loaded AT its committed path; parent file md5s recorded in parents_loaded; ZERO model compute in this cell",
        "rows": {r["id"]: {
            "edge_path": r["edge_source"], "band_path": r["band_source"],
            "clock_path": r["clock_source"]} for r in rows},
    }
    metrics["table_rows_loaded"] = [
        {k: r[k] for k in ["id", "root", "fact", "lineage", "ruler_class",
                           "edge", "band_kill_Ds", "band_median", "margin",
                           "step1_read", "death_t", "survival"]} for r in rows]
    write_partial("P1 loaded: the seven rows' edges + bands + clocks")

    # ================= G_E208 + G_E209: the parents' verdicts gated =========
    e208 = sources["e208"]
    e209 = sources["e209"]
    g_e208 = {
        "file": "runs/e208/metrics.json", "md5": prov["e208"]["md5"],
        "n_rows": len(e208["census_table"]),
        "verdict_loaded": e208["adjudication"]["verdict"],
        "R1_anchor_reloaded": {
            "edge": e208["census_table"][0]["edge"],
            "band_median": e208["census_table"][0]["band_median"],
            "margin": e208["census_table"][0]["margin_multiple"],
            "survival": e208["census_table"][0]["survival_steps_natural"],
        },
        "R2_R4_reloaded": {
            "R2_margin": e208["census_table"][1]["margin_multiple"],
            "R3_margin": e208["census_table"][2]["margin_multiple"],
            "R4_margin": e208["census_table"][3]["margin_multiple"],
            "R4_survival_natural": e208["census_table"][3]["survival_steps_natural"],
            "R4_alt_survival_fork": e208["census_table"][3]["alternate_walk"]["survival"],
        },
    }
    g_e208["pass"] = (g_e208["verdict_loaded"] == "MARGIN-PREDICTS"
                      and g_e208["n_rows"] == 4
                      and abs(g_e208["R1_anchor_reloaded"]["margin"] - 3.7206885127780804) < 1e-12)
    gates["G_E208"] = g_e208

    g_e209 = {
        "file": "runs/e209/metrics.json", "md5": prov["e209"]["md5"],
        "verdict_loaded": e209["adjudication"]["verdict"],
        "episode_mismatch_anatomy_loaded": e209["honesty"].get(
            "the_episode_mismatch_the_verdicts_anatomy"),
        "desk_forced_survival_side_loaded": e209["registered_bars"].get(
            "desk_forced_disclosure"),
        "C_arm_clocks_reloaded": {
            "seed_10902": {"step1": 0.6780481934547424, "death_t": 2},
            "seed_10907": {"step1": 0.8174034953117371, "first_ck_le_bar": 4},
            "seed_10908": {"step1": 0.32265704870224, "first_ck_le_bar": 2},
        },
        "s300_rows_out_by_construction": "R5/R6/R7's margins are MID-EPISODE reads (the settled s300 bodies) — the mismatch this cell repairs; loaded here only to gate the anatomy",
    }
    g_e209["pass"] = (g_e209["verdict_loaded"] == "MARGIN-BREAKS"
                      and g_e209["episode_mismatch_anatomy_loaded"] is not None)
    gates["G_E208"] = g_e208
    gates["G_E209"] = g_e209
    write_partial("G_E208 + G_E209 gated")

    # ================= G_ARITH: the margin arithmetic ========================
    ar = {"pass": True, "checks": []}

    def acheck(name, ok, detail):
        ar["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            ar["pass"] = False

    # (a) e205's three committed multiples must reproduce from loaded parents
    e205_mult = sources["e205"]["normalization"]["edge_multiples"]
    e205_med = sources["e205"]["normalization"]["band_medians"]
    margin_by_key = {}
    for r in rows:
        key = ("org1" if r["id"] in ("W1", "W2", "W3")
               else "half" if r["id"] in ("W4", "W5")
               else "mirabel" if r["id"] == "W6" else None)
        if key:
            margin_by_key.setdefault(key, r["margin"])
    for key, rid in (("org1", "W1"), ("half", "W4"), ("mirabel", "W6")):
        if key in e205_mult:
            d = abs(margin_by_key[key] - e205_mult[key])
            acheck(f"{rid} margin == e205 committed edge_multiples.{key}",
                   d < 1e-12, {"recomputed": margin_by_key[key],
                               "committed": e205_mult[key], "abs_diff": d})
            dm = abs(next(r for r in rows if r["id"] == rid)["band_median"]
                     - e205_med[key])
            acheck(f"{rid} band median == e205 committed band_medians.{key}",
                   dm < 1e-12, {"recomputed": next(r for r in rows if r["id"] == rid)["band_median"],
                                "committed": e205_med[key], "abs_diff": dm})

    # (b) the anchor: R1's committed 2.2699 / 0.6101 loaded verbatim
    w1 = next(r for r in rows if r["id"] == "W1")
    acheck("anchor edge == R1 committed 2.269916581032063",
           abs(w1["edge"] - 2.269916581032063) < 1e-12, {"loaded": w1["edge"]})
    acheck("anchor band median == R1 committed 0.6100797132671589",
           abs(w1["band_median"] - 0.6100797132671589) < 1e-12,
           {"loaded": w1["band_median"]})
    acheck("anchor margin == R1 committed 3.7206885127780804",
           abs(w1["margin"] - 3.7206885127780804) < 1e-12,
           {"loaded": w1["margin"]})

    # (c) the gray row's fresh arithmetic (from committed kill_Ds)
    if any(r["id"] == "W7" for r in rows):
        w7 = next(r for r in rows if r["id"] == "W7")
        acheck("W7 band median == middle order statistic of [1.75, 0.74, 0.58]",
               w7["band_median"] == 0.74, {"recomputed": w7["band_median"]})
        acheck("W7 margin == 0.92/0.74 (GRAY)",
               abs(w7["margin"] - 0.92 / 0.74) < 1e-12,
               {"recomputed": w7["margin"]})

    # (d) every row: margin == edge/band and survival == death_t - 1
    for r in rows:
        acheck(f"{r['id']} margin == edge/band",
               abs(r["margin"] - r["edge"] / r["band_median"]) < 1e-12,
               {"margin": r["margin"], "edge": r["edge"],
                "band": r["band_median"]})
        acheck(f"{r['id']} survival == death_t - 1",
               r["survival"] == r["death_t"] - 1,
               {"survival": r["survival"], "death_t": r["death_t"]})

    gates["G_ARITH"] = ar
    metrics["gates"] = gates
    write_partial("G_ARITH done")

    # ================= G_EPISODE: same-episode identity per row =============
    ep = {"pass": True, "checks": []}

    def echeck(name, ok, detail):
        ep["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            ep["pass"] = False

    # the three roots' identities, from the committed metadata
    e131_root = "runs/checkpoints/e131_consolidated_e113.pt"
    f2_root = "runs/checkpoints/e157_f2_consolidated.pt"
    e193b_root = "runs/checkpoints/e193b_root.pt"
    echeck("e199 org1 margin root == the pristine e131 root",
           sources["e199"]["organisms"]["org1"]["root"] == e131_root,
           {"root": sources["e199"]["organisms"]["org1"]["root"]})
    echeck("g1b's arms all wash the e131 root as theta0 (C arm included)",
           e131_root in json.dumps(sources["g1b"]),
           {"note": "g1b design: ROOT loads the arc's consolidated root DIRECTLY as theta0"})
    echeck("g1bR = the identical cell at fresh seeds (same e131 theta0)",
           "g1b" in json.dumps(sources["g1bR"]["design"]),
           {"design": sources["g1bR"]["design"]})
    echeck("e196/e197 clock roots == e193/e200/e205 margin roots (the pristine f2)",
           sources["e196"]["organism"]["root"] == f2_root
           and sources["e197"]["organism"]["root"] == f2_root
           and sources["e200"]["organism"]["root"] == f2_root
           and sources["e205"]["bands"]["half"]["root"] == f2_root,
           {"e196": sources["e196"]["organism"]["root"],
            "e197": sources["e197"]["organism"]["root"],
            "e200": sources["e200"]["organism"]["root"],
            "e205_band": sources["e205"]["bands"]["half"]["root"]})
    echeck("e199 mirabel clock root == the pristine e193b root",
           sources["e199"]["organisms"]["mirabel"]["root"] == e193b_root,
           {"root": sources["e199"]["organisms"]["mirabel"]["root"]})
    ep["note"] = ("every row's margin AND clock sit at the SAME pristine root: "
                  "the e131 root (W1-W3: the root e199's u0/e_chart's draws read "
                  "== the root g1b/g1bR's C arms washed), the f2 root (W4/W5: "
                  "e193's rays/e205's band == e196's/e197's walks' theta0), the "
                  "e193b root (W6/W7: e199's/e193b's margins == the e198/e199 "
                  "walk's theta0). No s300/mid-episode margin enters the table.")
    gates["G_EPISODE"] = ep
    write_partial("G_EPISODE done")

    # ================= G_WALKS: the clocks re-derived from committed reads ===
    wk = {"pass": True, "checks": []}

    def wcheck(name, ok, detail):
        wk["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            wk["pass"] = False

    for r in rows:
        rid = r["id"]
        if rid in ("W1", "W2", "W3"):
            fd = first_downcross(r["g_m12"])
            r["clock_rederived"] = fd
            wcheck(f"{rid} step-1 read alive (> 0.27) — the class-clause atom",
                   r["step1_read"] > SHUT, {"step1_read": r["step1_read"]})
            wcheck(f"{rid} death re-derived from the committed g-12 dict "
                   f"(first step <= 0.27 == death_t {r['death_t']})",
                   fd["death_t"] == r["death_t"],
                   {"rederived": fd, "carried": {"death_t": r["death_t"],
                                                 "survival": r["survival"]}})
        if rid == "W1":
            wcheck("W1 co-read: e199's org1 walk — the SAME seed-10902 episode "
                   "(t1 alive 0.6786, t2 death 9.83e-05; identical adjudication)",
                   r["co_read_walk"]["t1"] > SHUT
                   and r["co_read_walk"]["t2"] <= SHUT
                   and abs(r["co_read_walk"]["t1"] - r["step1_read"]) < 5e-4,
                   {"e199_t1": r["co_read_walk"]["t1"],
                    "c_arm_t1": r["step1_read"],
                    "note": "5e-4-class cross-device texture (g1b ran CUDA, e199 CPU); e209's G_STREAM verified the stream reproduces the arm rows at 1e-7"})
        if rid == "W2":
            wcheck("W2 bracket consistency (+2 alive 0.4611, +4 dead 0.1631; "
                   "first_ck_le_bar == 4 == death_t)",
                   r["g_m12"]["2"] > SHUT and r["g_m12"]["4"] <= SHUT,
                   {"plus2": r["g_m12"]["2"], "plus4": r["g_m12"]["4"],
                    "bracket": r["bracket"]})
        if rid == "W4":
            wcheck("W4 natural step-1 dead (0.0068 <= 0.27)",
                   r["step1_read"] <= SHUT, {"read": r["step1_read"]})
            wcheck("W4 e196 stop: kind=kill, step=1 (survival 0)",
                   r["death_t"] == 1 and r["survival"] == 0,
                   {"stop": sources["e196"]["phase0_walk_rebuild"]["stop"]["kind"]})
        if rid == "W5":
            alive14 = [j["gm"] > SHUT for j in
                       sources["e197"]["phase1_alive_walk"]["journal"][:4]]
            wcheck("W5 half-step t1 alive (0.4192 > 0.27) — THE HINGE READ",
                   r["step1_read"] > SHUT, {"read": r["step1_read"]})
            wcheck("W5 half-step t1-t4 all alive, t5 death (0.1648 <= 0.27)",
                   all(alive14) and r["death_t"] == 5 and r["survival"] == 4,
                   {"alive_t1_t4": alive14, "death_t": r["death_t"]})
            wcheck("W5 co-read: e200's alive_ledger agrees (t1 0.4192, t5 0.1648)",
                   abs(r["co_read_walk"]["t1"] - r["step1_read"]) < 1e-6
                   and r["co_read_walk"]["t5"] <= SHUT,
                   {"e200_t1": r["co_read_walk"]["t1"],
                    "e200_t5": r["co_read_walk"]["t5"]})
        if rid == "W6":
            led = sources["e199"]["organisms"]["mirabel"]["alive_ledger"]
            wcheck("W6 t1,t2 alive; t3 death (0.1230 <= 0.27); survival 2",
                   led[1]["ruler_read"] > SHUT and led[2]["ruler_read"] > SHUT
                   and led[3]["ruler_read"] <= SHUT and r["survival"] == 2,
                   {"t1": led[1]["ruler_read"], "t2": led[2]["ruler_read"],
                    "t3": led[3]["ruler_read"]})
        if rid == "W7":
            wcheck("W7 step-1 ZEPHYRA read dead (a_sign 0.0526 <= 0.27; "
                   "journal g+0 0.1629 <= 0.27); survival 0",
                   r["step1_read"] <= SHUT and r["journal_step1_g+0"] <= SHUT
                   and r["survival"] == 0,
                   {"asign": r["step1_read"], "journal_g+0": r["journal_step1_g+0"]})
    gates["G_WALKS"] = wk
    metrics["gates"] = gates
    write_partial("G_WALKS done")

    # ================= P5: the table + the adjudication ======================
    if not args.smoke:
        def cls(m):
            return ">2" if m > 2.0 else ("<1" if m < 1.0 else "gray[1,2]")

        for r in rows:
            m, s = r["margin"], r["survival"]
            r["class"] = cls(m)
            if r["class"] == "gray[1,2]":
                r["violation"] = None  # gray: reported, never adjudicated
            else:
                r["violation"] = ((m > 2.0 and s == 0) or (m < 1.0 and s >= 1))

        defined = [r for r in rows if r["class"] != "gray[1,2]"]
        violators = [r for r in defined if r["violation"]]
        holds_fires = len(violators) == 0
        breaks_fires = len(violators) >= 1
        verdict = ("SAME-EPISODE-HOLDS" if holds_fires else
                   "SAME-EPISODE-BREAKS" if breaks_fires else "GRADED")

        # the sensitivity read: n=6 natural-clock table (e208's fork convention)
        natural_rows = [r for r in rows if r["id"] != "W5"]
        nat_defined = [r for r in natural_rows if r["class"] != "gray[1,2]"]
        nat_violators = [r for r in nat_defined if r["violation"]]
        sensitivity_n6 = {
            "reading": "drop W5 (the counterfactual row) — e208's frozen fork convention (the counterfactual never adjudicated)",
            "n_rows": len(natural_rows),
            "violations": len(nat_violators),
            "holds_letter_would_fire": len(nat_violators) == 0,
            "note": "context, NEVER adjudicated: the verdict's hinge is W5's membership; the dispatch names e197's clock among the lineages' own first-episode clocks, so the registered table is n=7 and W5 is a row",
        }

        rho7 = spearman([r["margin"] for r in rows],
                        [r["survival"] for r in rows])
        two_line_natural = all(
            (r["margin"] > 2.0) == (r["survival"] >= 1)
            for r in nat_defined)

        metrics["within_episode_table"] = [
            {"id": r["id"], "root": r["root"], "fact": r["fact"],
             "lineage": r["lineage"], "ruler_class": r["ruler_class"],
             "edge": r["edge"], "edge_kind": (
                 "interpolated downcrossing" if r["id"] not in ("W7",)
                 else "first-dead grid point (upper bound)"),
             "band_kill_Ds": r["band_kill_Ds"],
             "band_median": r["band_median"],
             "band_n": r["band_median_stat"]["n"],
             "band_censored": r["band_median_stat"]["n_censored"],
             "band_instrument": (
                 "e_chart FINE_GRID 0.05..1.5 interpolated; 1-of-3 right-censored"
                 if r["id"] in ("W1", "W2", "W3") else
                 "onset grid 0.05..3.00 interpolated (e205 fresh)"
                 if r["id"] in ("W4", "W5") else
                 "e193b D_GRID 27pts, first-dead grid points (upper bounds)"),
             "margin_multiple": r["margin"], "clock_kind": r["clock_kind"],
             "clock_source": r["clock_source"],
             "step1_read": r["step1_read"], "death_t": r["death_t"],
             "survival_steps": r["survival"],
             "class": r["class"], "violation": r["violation"],
             "fork_row": r.get("fork_row", False),
             "anchor": r.get("anchor", False),
             "bracket": r.get("bracket"),
             "caveats": r["caveats"]} for r in rows]

        metrics["adjudication"] = {
            "bars": {
                "SAME-EPISODE-HOLDS": {
                    "fires": holds_fires,
                    "detail": {
                        "class_>2": {r["id"]: [r["margin"], r["survival"]]
                                     for r in rows if r["class"] == ">2"},
                        "class_<1": {r["id"]: [r["margin"], r["survival"]]
                                     for r in rows if r["class"] == "<1"},
                        "gray_reported_never_adjudicated": {
                            r["id"]: [r["margin"], r["survival"]]
                            for r in rows if r["class"] == "gray[1,2]"},
                        "letter": "margin > 2 surviving step 1; < 1 dying",
                        "desk_forced": True,
                    },
                },
                "SAME-EPISODE-BREAKS": {
                    "fires": breaks_fires,
                    "violating_rows": [r["id"] for r in violators],
                    "detail": {r["id"]: {"margin": r["margin"],
                                         "survival": r["survival"],
                                         "class": r["class"],
                                         "kind": "margin < 1 SURVIVED step 1"}
                               for r in violators},
                },
                "GRADED": {"fires": verdict == "GRADED"},
            },
            "verdict": verdict,
            "clause": (
                f"the within-episode table (n={len(rows)}, all rows margin-and-clock "
                f"from the same pristine episode): "
                + ("every defined-class row separates (margins > 2 all survive step 1; margins < 1 all die at it) — the e208 finding RESTORED on the honest episode"
                   if holds_fires else
                   "; ".join(
                       f"{r['id']} (margin {r['margin']:.3f}x, class {r['class']}) survived {r['survival']} step(s) — the 2x line does NOT separate it"
                       for r in violators)
                   + " — the class line is dead at ANY episode; the margin is a descriptor, full stop")),
            "the_hinge": {
                "row": "W5 (the f2 half-step counterfactual — e197's committed clock)",
                "why": "the f2 margin 0.616x (< 1) faces two committed first-episode clocks at its pristine root: the natural walk dies at t=1 (consistent) and the half-step walk survives to t=5 (the violation); e208 froze the counterfactual as never-adjudicated, THIS registration (the dispatch names the e193/e197 lineages' OWN clocks) adjudicates it",
                "sensitivity_n6_natural_clocks": sensitivity_n6,
            },
            "desk_forced_side": {
                "margins_and_clocks": "BOTH sides desk-fixed by committed records before this cell (e205's three margins + R1's anchor; e196/e197/e198/e199 walks; g1b/g1bR C arms) — the firing was readable at registration and is reported as forced (e205's convention); this cell's fresh work is the same-episode GATE, the assembly, the hinge made explicit, and the figure",
            },
            "context_never_adjudicated": {
                "spearman_margin_survival_n7": rho7,
                "two_line_separation_natural_clocks_only": {
                    "observation": "on the natural clocks alone (n=6), every margin > 2 row survived its first wash step and every margin < 1 row died at it — the e208 context cut, rebuilt same-episode",
                    "holds": two_line_natural,
                    "note": "context only; W5's membership is the registered question, adjudicated above",
                },
                "not_an_ordering": "the margin does not order deaths inside the survivor class: W2 (3.72x) is bracketed to die at t=4 where W1 (3.72x, same margin) died at t=2 and W6 (2.12x) at t=3 — class separation, not a graded survival law",
                "survivor_side_noise_thinness": "W3's +1 read 0.3227 is 1.20x the 0.27 bar — the survivor side at the e131 root is one bad draw from violating; reported, never adjudicated",
            },
            "composite_order": "SAME-EPISODE-HOLDS / SAME-EPISODE-BREAKS / GRADED (frozen before assembly; the first two mutually exclusive by construction)",
        }
        write_partial("P5 table + adjudication")

    # ================= P6: the figure ========================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.smoke:
        fig = plt.figure(figsize=(16.0, 8.2), dpi=130)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5], wspace=0.16)

        ax = fig.add_subplot(gs[0, 0])
        colors = {"W1": "#1f77b4", "W2": "#1f77b4", "W3": "#1f77b4",
                  "W4": "#9467bd", "W5": "#9467bd", "W6": "#2ca02c",
                  "W7": "#d62728"}
        # violation zones: (margin>2, surv 0) and (margin<1, surv>=1)
        ax.axhspan(-0.4, 0.5, xmin=0.62, xmax=1.0, color="#d62728", alpha=0.05)
        ax.axhspan(0.5, 4.6, xmax=0.245, color="#d62728", alpha=0.05)
        ax.axvline(1.0, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(2.0, color="black", lw=1.2, ls="--", alpha=0.9)
        ax.axvspan(2.0, 4.6, color="#2ca02c", alpha=0.06)
        ax.axvspan(0.4, 1.0, color="#d62728", alpha=0.06)
        ax.axvspan(1.0, 2.0, color="gray", alpha=0.07)
        ax.text(3.3, 4.45, "margin > 2\n(survive step 1)", fontsize=8.5,
                ha="center", color="#2ca02c")
        ax.text(1.5, 4.45, "gray", fontsize=8.5, ha="center", color="dimgray")
        ax.text(0.68, 4.45, "margin < 1\n(die at step 1)", fontsize=8.5,
                ha="center", color="#d62728")
        offsets = {"W1": (10, 12), "W2": (10, 8), "W3": (10, -18),
                   "W4": (-6, -22), "W5": (10, 8), "W6": (10, 10),
                   "W7": (10, -16)}
        for r in rows:
            x, y = r["margin"], r["survival"]
            filled = r["id"] != "W5"
            ax.scatter([x], [y], s=150 if filled else 130,
                       color=colors[r["id"]] if filled else "none",
                       facecolor=colors[r["id"]] if filled else "none",
                       edgecolor=colors[r["id"]], linewidth=1.8 if not filled else 0.8,
                       linestyle="--" if not filled else "solid",
                       zorder=5, marker="o" if r["id"] != "W5" else "D")
            lbl = f"{r['id']} {r['fact'][:4]}\n{r['margin']:.2f}x"
            if r.get("violation"):
                lbl += " VIOLATES"
                ax.scatter([x], [y], s=360, facecolor="none",
                           edgecolor="#d62728", linewidth=2.2, zorder=4)
            ax.annotate(lbl, (x, y), xytext=offsets[r["id"]],
                        textcoords="offset points", fontsize=8.5,
                        color="#d62728" if r.get("violation") else "black")
        ax.set_xscale("log")
        ax.set_xticks([0.616, 1.0, 1.243, 2.116, 3.721])
        ax.set_xticklabels(["0.62", "1", "1.24", "2.12", "3.72"], fontsize=8.5)
        ax.set_ylim(-0.45, 4.85)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_xlabel("edge multiple  =  u0 static kill-D / in-span band median (log)",
                      fontsize=10)
        ax.set_ylabel("first-episode wash steps survived", fontsize=10)
        ax.set_title("E210 the SAME-EPISODE margin: margin vs survival, one episode\n"
                     "(filled = natural-step clocks; open diamond = W5, the half-step counterfactual — THE HINGE)",
                     fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5)

        axt = fig.add_subplot(gs[0, 1])
        axt.axis("off")
        col_heads = ["row", "pristine root / fact", "clock (lineage)",
                     "ruler", "MARGIN", "step-1 read", "death t (surv)",
                     "class", "verdict"]
        table_rows = []
        for r in rows:
            table_rows.append([
                r["id"],
                f"{r['root'].split(' (')[0].replace(' PRISTINE','')}\n{r['fact']}",
                r["lineage"].split(" (")[0],
                r["ruler_class"].split(" ")[0],
                f"{r['margin']:.3f}x",
                f"{r['step1_read']:.3g}",
                f"t={r['death_t']} (surv {r['survival']})"
                + (" [bracket (2,4]]" if r["id"] == "W2" else ""),
                r["class"],
                ("VIOLATION" if r.get("violation")
                 else "gray (n/a)" if r["class"] == "gray[1,2]" else "ok"),
            ])
        verdict = metrics.get("adjudication", {}).get("verdict", "SMOKE")
        tbl = axt.table(cellText=table_rows, colLabels=col_heads,
                        cellLoc="center", loc="upper center",
                        colWidths=[0.05, 0.17, 0.17, 0.08, 0.09, 0.10,
                                   0.15, 0.09, 0.10])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.2)
        tbl.scale(1, 1.75)
        for (rowi, coli), cell in tbl.get_celld().items():
            cell.set_edgecolor("#bbbbbb")
            if rowi == 0:
                cell.set_facecolor("#e8e8e8")
            elif rowi % 2 == 0:
                cell.set_facecolor("#f7f7f7")
            if rowi >= 1 and table_rows[rowi - 1][-1] == "VIOLATION":
                cell.set_facecolor("#f9d6d6")
        axt.set_title(
            f"the within-episode table (verdict: {verdict} — desk-forced, disclosed; "
            f"W1-W3 share ONE margin: the e131 root's 3.72x read against THREE wash seeds)\n"
            "caveats per row in metrics.json within_episode_table; e209's s300 rows are out BY CONSTRUCTION (mid-episode margins)",
            fontsize=9.5, pad=18)

        fig.suptitle("E210 — the same-episode margin (e209's repair: margin and clock from ONE pristine episode; all values loaded committed)",
                     fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        out = rd / "same_episode_margin.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        metrics["figure"] = str(out.relative_to(REPO)).replace("\\", "/")
        log(f"figure -> {out}")

    # ================= honesty + finalize ====================================
    metrics["honesty"] = {
        "rows_own_ns": "n=7 rows but THREE margins: W1-W3 share the e131 root's margin (3.7207x — one u0, one 3-draw band, read against THREE wash seeds), W4/W5 share the f2 root's margin (0.6160x — one u0, one band, TWO committed clocks), W6/W7 are two facts of one organism sharing one walk; every band is n=3 draws of ONE 20-step history realization; the 2-3x draw lottery (T155) would move any band median on redraw",
        "the_hinge": "the verdict turns ENTIRELY on W5's membership: the f2 margin 0.616x (< 1) survives 4 steps under e197's committed half-step counterfactual clock; under the n=6 natural-clock reading (e208's fork convention) the line separates every row — both readings are on the face of the adjudication, the registered one is n=7",
        "episode_identity": "every row's margin AND clock sit at the SAME pristine root (gated in G_EPISODE) — the repair's whole content; e209's violated rows joined s300 margins to first-episode clocks and are OUT by construction",
        "ruler_caveats": "three ruler classes (g-12 novel at W1-W3/W6, g-4 trained at W4/W5, g+0 fallback at W7); within every row edge and band share the ruler; W7's edge AND band are grid upper bounds (bias undetermined); W6's band upper-bounds its margin to a LOWER bound (conservative for its class)",
        "clock_granularity": "W2's death is bracketed (2,4] on g1bR's ckpt clock {1,2,4,...} (the step-3 read was never committed) — its survival 3 is the e209-convention carry; the class-clause atom (step-1 read) is exact and committed at EVERY row; W3's step-1 read 0.3227 is 1.20x the bar (the survivor side's thinnest margin)",
        "single_realizations": "every clock is ONE realized walk; W1's clock is co-cited twice (g1b's CUDA C arm and e199's CPU walk — the same seed-10902 episode at 5e-4 cross-device texture, e209's G_STREAM-verified stream); nothing here is a distribution",
        "logits_prediction_check": "the table is behavior reads (battery p(Z) shut-bar crossings) on committed data; NO intervening was done in this cell — the class claim is only as good as the single-realization clocks it faces",
        "nothing_guaranteed": "the openness was the point: the hinge row could have died at t=1 under the half-step clock too (it survived to t=5 — committed long before this cell); the survivor side could have violated on W3's thin read (it did not); the observed outcome is the registered one, reported forced",
        "desk_forced": "like e205/e208/e209, the verdict was desk-computable at registration; this cell's value is the same-episode GATE (the repair's actual content), the assembly, the hinge made explicit, and the figure",
    }
    metrics["provenance"] = {
        "margins": {
            "W1-W3 (e131 root)": "edge runs/e199/metrics.json onset_curves.org1[0]; band runs/e_chart/metrics.json partB_subspace.e131.fine_D_summary.inspan_thr_D (via e205's committed copy, cross-checked); multiple == e205 edge_multiples.org1",
            "W4-W5 (f2 root)": "edge runs/e200/metrics.json profiles.root_u0 (u0 = e193's R2_SIGN rays, onset-grid interpolated); band runs/e205/metrics.json bands.half.kill_Ds_fresh; multiple == e205 edge_multiples.half",
            "W6 (e193b MIRABEL)": "edge runs/e199/metrics.json onset_curves.mirabel[0]; band runs/e193b/metrics.json inspan_range.MIRABEL.kill_Ds; multiple == e205 edge_multiples.mirabel",
            "W7 (e193b ZEPHYRA, gray)": "edge runs/e193b/metrics.json adjudication.per_fact.ZEPHYRA.sign_kill; band runs/e193b/metrics.json inspan_range.ZEPHYRA.kill_Ds",
        },
        "clocks": {
            "W1": "runs/g1b/metrics.json arms.C.traj (seed 10902; co-cited g1bR reference.C_g_m12 + e199 organisms.org1.alive_ledger)",
            "W2": "runs/g1bR/metrics.json adjudication.wall.C10907.g_m12 (seed 10907)",
            "W3": "runs/g1bR/metrics.json adjudication.wall.C10908.g_m12 (seed 10908)",
            "W4": "runs/e196/metrics.json phase0_walk_rebuild.stop (+ walk_journal[0])",
            "W5": "runs/e197/metrics.json phase1_alive_walk.stop (+ journal; co-cited runs/e200/metrics.json alive_ledger)",
            "W6": "runs/e199/metrics.json organisms.mirabel.alive_ledger",
            "W7": "runs/e198/metrics.json phase0_walk_rebuild.kill_zephyra_rederived (+ walk_journal[0])",
        },
        "parents": {
            "e209": "runs/e209/metrics.json — the episode-mismatch anatomy this table answers (verdict MARGIN-BREAKS gated in G_E209)",
            "e208": "runs/e208/metrics.json — the n=4 class-line finding whose within-episode status this cell re-tests (gated in G_E208)",
            "e191_e205_machinery": "the committed rays/bands this desk cell loads were minted by the e191/e192 checkpoint+ray machinery and e205's onset-grid band instrument — no fresh instrument read was needed (the dispatch's 'zero fresh compute if possible' satisfied: ZERO)",
        },
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
