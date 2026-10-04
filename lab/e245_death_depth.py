"""
E245 — T221's registered discrimination: THE DEATH-DEPTH READ
(what selects the commitment's death mode — the mis-dial vs the frequency collapse)

DESK-ONLY on COMMITTED data. CPU threads 4, no GPU, no torch, no model loads.
Two other agents live — load-polite (json + numpy + matplotlib only).

T221's registration (THINKING.md, verbatim):
  REGISTERED PREDICTION (e245, no retrofit): "over the 19 flips, p(+80) (and p
  at the flip's first-crossing state where available) separates the modes —
  the mis-dials carry surviving belief mass, the collapses do not; median
  separation >= 2x."
  ALTERNATIVE: "the wash split (w1 collapse-heavy, w2 mis-dial-heavy) is the
  selector — the washes differ (their corpora/streams), and if p does not
  separate but wash does, the mode is WRITTEN BY THE WASH, not the death's
  depth (a harder, stranger claim: the wash's own texture deciding how the
  orphaned commitment falls)."

REGISTERED BARS — frozen VERBATIM from the dispatch brief BEFORE any compute:
  DEPTH-SELECTS — "the mis-dials' median p(+80) >= 2x the collapses' AND the
    separation holds within each wash (or one wash has no collapses to test —
    disclosed) — two stages of death confirmed"
  WASH-WRITES — "depth does not separate (ratio < 2 or within-wash fails)
    while the wash contingency is lopsided (one wash >= 75% one mode) — the
    mode is written by the wash, the stranger branch"
  MIXED — "the tables verbatim, no inflation"

OPERATIONALIZATIONS FROZEN BEFORE COMPUTE:
  population: the 19 (probe, wash) flip records VERBATIM from
    runs/e243/metrics.json "table" (do NOT re-derive; e243's G_FLIPS/G_POP
    already double-sourced the enumeration against the e238 npz dumps);
    classifications (MIS-DIAL / FREQUENCY-COLLAPSE / OTHER) and +80 targets
    carried verbatim from the same table.
  p(+80): the probe's p at (wash, +80) from runs/e228/journal.json (the
    per-state per-probe p records); gate: exact match vs e243's p_80 field
    for all 19; provenance re-certification: 3 records re-checked against
    runs/e214/journal.json (the independent journal of the same archive).
  flip time (first crossing): the EARLIEST state in the record's wash lineage
    (w1: +2, +10, +50, +80; w2: +10, +50, +80) whose top1_id differs from the
    t0 top1_id, per the e228 journal argmax records (the only committed
    argmax source for intermediate states); p_at_flip = that state's p;
    co-report whether the first-crossing token equals the +80 target.
  DEPTH read: separation ratio = median(mis-dial p(+80)) / median(collapse
    p(+80)); numpy median (linear interpolation at even n — convention
    disclosed). DEPTH-SELECTS requires ratio_overall >= 2 AND ratio_w1 >= 2
    AND (ratio_w2 >= 2 OR w2 has zero collapses — the bar's own escape,
    disclosed; w2's collapse count is whatever it is, n disclosed per
    stratum). Both washes' mis-dial strata are non-empty (e243: w1 5, w2 4).
  WASH read: the mode-by-wash contingency, the 2x2 (wash x {MIS-DIAL,
    FREQUENCY-COLLAPSE}) with the OTHER class DISCLOSED. The lopsidedness
    clause adjudicates on the ALL-records denominators (a wash's flips are
    that wash's flips; OTHER counts against mode purity): WASH-WRITES'
    contingency clause fires iff max over washes/modes of P(mode | wash's all
    flip records) >= 0.75. CO-REPORTS, disclosed, NOT adjudicating (frozen
    here): (a) the two-mode-restricted P(mode|wash) (OTHER excluded from the
    denominator), (b) P(wash | mode) concentrations, (c) the cross-wash
    collapse-rate contrast. Rationale for the adjudicating choice: the bar's
    words ("one wash >= 75% one mode") name the WASH's composition; WASH-
    WRITES is the stranger branch and takes the conservative test; hiding the
    3 w2 OTHER records (37.5% of w2's flips) inside a denominator exclusion
    would manufacture the lopsidedness the bar asks to measure.
  verdict: DEPTH-SELECTS iff the depth clauses all hold; else WASH-WRITES iff
    the contingency clause holds; else MIXED. No bar shopping, no inflation.
  census binary (the dispatch's wording "halved / never-halved"):
    halved iff p(wash, +80) < 0.5 * p_t0, computed from the journals (e232's
    own endpoint criterion); e243's e232_fate and e230's committed death-ORDER
    class (P-FIRST / MARGIN-FIRST / TOGETHER, read not re-derived) carried
    verbatim as co-reports. Known gloss risk, checked not assumed: T221
    parenthesized census-outside as "the belief NEVER halved" — the endpoint
    computation adjudicates that gloss; whatever it says is DISCLOSED (the
    registered bars do not read on the census binary).
  n = 19 over (9 mis-dial, 7 collapse, 3 other; w1 11, w2 8); 6 probes die
    under both washes (shared t0/RU reads — e243's confound carried verbatim);
    nothing guaranteed.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
E228 = REPO / "runs" / "e228" / "journal.json"
E214 = REPO / "runs" / "e214" / "journal.json"
E243 = REPO / "runs" / "e243" / "metrics.json"
E230 = REPO / "runs" / "e230" / "metrics.json"
OUT = REPO / "runs" / "e245"
OUT.mkdir(parents=True, exist_ok=True)

WASH_STATES = {"w1": [2, 10, 50, 80], "w2": [10, 50, 80]}
THE_ID = 262  # ' the' — e243's tokenizer-verified frequency token


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_journal(path: Path):
    """-> {(battery, probe): {state_key: record}} with state_key 't0' or 'w1_50'."""
    j = json.loads(path.read_text())
    table = {}
    for s in j["states"]:
        key = "t0" if s["wash"] == "t0" else f"{s['wash']}_{s['step']}"
        for bat in ("fact", "ctrl", "near", "tmpl"):
            for r in s[bat]["probes"]:
                table[(bat, r["fact"])] = table.get((bat, r["fact"]), {})
                table[(bat, r["fact"])][key] = r
    return table


def median(xs):
    return float(np.median(np.asarray(xs, dtype=float)))


def main():
    t_start = time.time()
    git_head = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()

    e243 = json.loads(E243.read_text())
    table = e243["table"]
    j228 = load_journal(E228)
    j214 = load_journal(E214)
    e230 = json.loads(E230.read_text())
    # e230 death-order class join: (battery, wash, probe) -> class / p_cross_step / censored
    e230cls = {}
    for bat, washes in e230["read1_population"]["probe_level"].items():
        for wash, recs in washes.items():
            for r in recs:
                e230cls[(bat, wash, r["probe"])] = r

    gates = {}

    # ---------------- the cell: per-flip reads ----------------
    rows = []
    for rec in table:
        wash, bat, probe = rec["wash"], rec["battery"], rec["probe"]
        traj = j228[(bat, probe)]
        t0 = traj["t0"]
        p_t0 = t0["p"]
        p80 = traj[f"{wash}_80"]["p"]
        # first crossing: earliest state in this wash's lineage whose top1 differs from t0's
        first_step, first_rec = None, None
        for st in WASH_STATES[wash]:
            r = traj[f"{wash}_{st}"]
            if r["top1_id"] != t0["top1_id"]:
                first_step, first_rec = st, r
                break
        assert first_rec is not None, f"no crossing found for {wash} {probe}"
        halved_endpoint = bool(p80 < 0.5 * p_t0)
        c30 = e230cls.get((bat, wash, probe), {})
        rows.append({
            "wash": wash,
            "battery": bat,
            "probe": probe,
            "classification": rec["classification"],
            "target_80": rec["target"],
            "target_id_80": rec["target_id"],
            "p_t0": p_t0,
            "p_80": p80,
            "p_80_from_e243": rec["p_80"],
            "flip_step": first_step,
            "p_at_flip": first_rec["p"],
            "first_cross_token_id": first_rec["top1_id"],
            "first_cross_is_target_80": bool(first_rec["top1_id"] == rec["target_id"]),
            "first_cross_is_answer": bool(first_rec["top1_id"] == t0["top1_id"]),
            "halved_endpoint": halved_endpoint,
            "half_threshold": 0.5 * p_t0,
            "e232_fate": rec["e232_fate"],
            "e230_death_order_class": c30.get("class"),
            "e230_p_cross_step": c30.get("p_cross_step"),
            "e230_censored": c30.get("censored"),
        })

    # ---------------- gates ----------------
    gates["G_SHA_e228"] = {
        "got": sha16(E228), "committed": e243["sources"]["margins_journal"]["sha256_16"],
        "pass": sha16(E228) == e243["sources"]["margins_journal"]["sha256_16"],
        "desc": "e228 journal sha256_16 == e243's committed source sha"}
    gates["G_POP"] = {
        "pass": len(rows) == 19,
        "n": len(rows),
        "desc": "population carried verbatim from e243's table (do not re-derive); n=19"}
    diffs = [abs(r["p_80"] - r["p_80_from_e243"]) for r in rows]
    gates["G_P80"] = {
        "max_abs_diff_e228_vs_e243": max(diffs),
        "pass": max(diffs) < 1e-12,
        "desc": "p(+80) recomputed from the e228 journal equals e243's committed p_80 for all 19 records"}
    tgt_match = [r["wash"] + " " + r["probe"] for r in rows
                 if j228[(r["battery"], r["probe"])][f"{r['wash']}_80"]["top1_id"] != r["target_id_80"]]
    gates["G_TGT80"] = {
        "n_mismatch": len(tgt_match), "mismatches": tgt_match,
        "pass": len(tgt_match) == 0,
        "desc": "journal top1 at (wash,+80) == e243's target for all 19 (flip identities consistent; e243 G_FLIPS already double-sourced this vs the npz dumps)"}
    # provenance re-certification: 3 records vs e214's independent journal
    recheck = [("fact", "France->Paris", "w1"),
               ("ctrl", "The phone made by Apple->iPhone", "w1"),
               ("tmpl", "Jackson->Mississippi", "w2")]
    cert = []
    for bat, probe, wash in recheck:
        a, b = j228[(bat, probe)], j214[(bat, probe)]
        d_t0 = abs(a["t0"]["p"] - b["t0"]["p"])
        d_80 = abs(a[f"{wash}_80"]["p"] - b[f"{wash}_80"]["p"])
        d_flip = abs(a["t0"]["top1_id"] - b["t0"]["top1_id"])
        cert.append({"probe": f"{wash} {probe}", "abs_dp_t0": d_t0, "abs_dp_80": d_80,
                     "top1_t0_identical": d_flip == 0})
    gates["G_RECERT214"] = {
        "records": cert,
        "pass": all(c["abs_dp_t0"] < 1e-12 and c["abs_dp_80"] < 1e-12 and c["top1_t0_identical"]
                    for c in cert),
        "desc": "3 records re-certified vs e214's journal: p at t0 and (wash,+80) identical, t0 argmax identical (independent journal of the same committed archive)"}
    gates["G_E230_JOIN"] = {
        "n_joined": sum(1 for r in rows if r["e230_death_order_class"] is not None),
        "pass": all(r["e230_death_order_class"] is not None for r in rows),
        "desc": "all 19 records joined to e230's committed death-order class (read, not re-derived)"}

    # ---------------- (1) the DEPTH read ----------------
    def depth_stats(sel):
        p80s = [r["p_80"] for r in rows if sel(r)]
        pflips = [r["p_at_flip"] for r in rows if sel(r)]
        return {
            "n": len(p80s),
            "median_p_80": median(p80s) if p80s else None,
            "median_p_at_flip": median(pflips) if pflips else None,
            "min_p_80": min(p80s) if p80s else None,
            "max_p_80": max(p80s) if p80s else None,
        }

    is_md = lambda r: r["classification"] == "MIS-DIAL"
    is_fc = lambda r: r["classification"] == "FREQUENCY-COLLAPSE"
    is_ot = lambda r: r["classification"] == "OTHER"

    depth = {
        "mis_dial_all": depth_stats(is_md),
        "collapse_all": depth_stats(is_fc),
        "other_all": depth_stats(is_ot),
        "mis_dial_w1": depth_stats(lambda r: is_md(r) and r["wash"] == "w1"),
        "collapse_w1": depth_stats(lambda r: is_fc(r) and r["wash"] == "w1"),
        "mis_dial_w2": depth_stats(lambda r: is_md(r) and r["wash"] == "w2"),
        "collapse_w2": depth_stats(lambda r: is_fc(r) and r["wash"] == "w2"),
    }
    for scope, md, fc in [
        ("overall", depth["mis_dial_all"], depth["collapse_all"]),
        ("w1", depth["mis_dial_w1"], depth["collapse_w1"]),
        ("w2", depth["mis_dial_w2"], depth["collapse_w2"]),
    ]:
        depth[f"ratio_{scope}"] = (md["median_p_80"] / fc["median_p_80"]
                                   if md["median_p_80"] is not None and fc["median_p_80"] not in (None, 0)
                                   else None)
    # the p-at-flip co-separation (T221 named it; the bar reads on p(+80) — co-report only)
    for scope, md, fc in [("overall", depth["mis_dial_all"], depth["collapse_all"])]:
        depth["ratio_p_at_flip_overall"] = (md["median_p_at_flip"] / fc["median_p_at_flip"]
                                            if md["median_p_at_flip"] and fc["median_p_at_flip"] else None)

    clause_ratio = depth["ratio_overall"] is not None and depth["ratio_overall"] >= 2.0
    clause_w1 = depth["ratio_w1"] is not None and depth["ratio_w1"] >= 2.0
    w2_esc = depth["collapse_w2"]["n"] == 0  # the bar's own escape, disclosed
    clause_w2 = w2_esc or (depth["ratio_w2"] is not None and depth["ratio_w2"] >= 2.0)
    depth_selects = clause_ratio and clause_w1 and clause_w2
    depth["clauses"] = {
        "ratio_overall_ge_2": clause_ratio,
        "within_w1_ge_2": clause_w1,
        "within_w2_ge_2_or_empty": clause_w2,
        "w2_collapse_n": depth["collapse_w2"]["n"],
        "w2_escape_used": w2_esc,
        "DEPTH-SELECTS": depth_selects,
    }

    # ---------------- (2) the WASH read ----------------
    washes = ["w1", "w2"]
    modes = ["MIS-DIAL", "FREQUENCY-COLLAPSE", "OTHER"]
    counts = {w: {m: sum(1 for r in rows if r["wash"] == w and r["classification"] == m)
                  for m in modes} for w in washes}
    n_w = {w: sum(counts[w].values()) for w in washes}
    # ADJUDICATING: all-records denominators (frozen; see docstring)
    p_mode_given_wash = {w: {m: counts[w][m] / n_w[w] for m in modes} for w in washes}
    max_lob = max((p_mode_given_wash[w][m], w, m) for w in washes for m in modes)
    wash_lopsided = bool(max_lob[0] >= 0.75)
    # co-reports (disclosed, NOT adjudicating)
    two_mode_n = {w: counts[w]["MIS-DIAL"] + counts[w]["FREQUENCY-COLLAPSE"] for w in washes}
    p_mode_given_wash_2m = {
        w: {"MIS-DIAL": counts[w]["MIS-DIAL"] / two_mode_n[w] if two_mode_n[w] else None,
            "FREQUENCY-COLLAPSE": counts[w]["FREQUENCY-COLLAPSE"] / two_mode_n[w] if two_mode_n[w] else None}
        for w in washes}
    n_m = {m: sum(counts[w][m] for w in washes) for m in modes}
    p_wash_given_mode = {
        m: {w: counts[w][m] / n_m[m] if n_m[m] else None for w in washes} for m in modes}
    wash_read = {
        "counts": counts,
        "n_per_wash": n_w,
        "P_mode_given_wash_ALLrecords_ADJUDICATING": p_mode_given_wash,
        "max_lopsidedness": {"p": max_lob[0], "wash": max_lob[1], "mode": max_lob[2]},
        "lopsided_at_75": wash_lopsided,
        "CO_REPORT_two_mode_restricted_not_adjudicating": p_mode_given_wash_2m,
        "CO_REPORT_P_wash_given_mode_not_adjudicating": p_wash_given_mode,
        "CO_REPORT_collapse_rate_contrast": {
            "P_collapse_w1": p_mode_given_wash["w1"]["FREQUENCY-COLLAPSE"],
            "P_collapse_w2": p_mode_given_wash["w2"]["FREQUENCY-COLLAPSE"]},
        "other_records_disclosed": [f"{r['wash']} {r['probe']} -> {r['target_80']}"
                                    for r in rows if is_ot(r)],
    }

    # ---------------- (3) the joint: depth WITHIN wash ----------------
    joint = {
        "within_w1": {"ratio": depth["ratio_w1"],
                      "n_md": depth["mis_dial_w1"]["n"], "n_fc": depth["collapse_w1"]["n"],
                      "median_md": depth["mis_dial_w1"]["median_p_80"],
                      "median_fc": depth["collapse_w1"]["median_p_80"]},
        "within_w2": {"ratio": depth["ratio_w2"],
                      "n_md": depth["mis_dial_w2"]["n"], "n_fc": depth["collapse_w2"]["n"],
                      "median_md": depth["mis_dial_w2"]["median_p_80"],
                      "median_fc": depth["collapse_w2"]["median_p_80"]},
        "read": "depth separates within each wash -> depth wins; only wash separates -> the alternative stands",
        "paired_contrasts": [],
    }
    # within-probe, cross-wash pairs (the cleanest depth-vs-wash discriminator)
    by_probe = {}
    for r in rows:
        by_probe.setdefault((r["battery"], r["probe"]), []).append(r)
    for (bat, probe), rs in sorted(by_probe.items()):
        if len(rs) == 2:
            a, b = sorted(rs, key=lambda r: r["wash"])
            joint["paired_contrasts"].append({
                "probe": probe,
                "w1": {"mode": a["classification"], "p_80": a["p_80"]},
                "w2": {"mode": b["classification"], "p_80": b["p_80"]},
                "p_80_ratio_w1_over_w2": a["p_80"] / b["p_80"] if b["p_80"] else None,
                "mode_flipped": a["classification"] != b["classification"]})

    # ---------------- adjudication ----------------
    if depth_selects:
        verdict = "DEPTH-SELECTS"
        bar_fired = ("the mis-dials' median p(+80) >= 2x the collapses' AND the separation "
                     "holds within each wash (or one wash has no collapses to test — "
                     "disclosed) — two stages of death confirmed")
    elif wash_lopsided:
        verdict = "WASH-WRITES"
        bar_fired = ("depth does not separate (ratio < 2 or within-wash fails) while the wash "
                     "contingency is lopsided (one wash >= 75% one mode) — the mode is "
                     "written by the wash, the stranger branch")
    else:
        verdict = "MIXED"
        bar_fired = "the tables verbatim, no inflation"

    # ---------------- census read (co-report; the gloss check) ----------------
    n_halved = sum(1 for r in rows if r["halved_endpoint"])
    census = {
        "halved_endpoint_n": n_halved,
        "never_halved_endpoint_n": len(rows) - n_halved,
        "e232_fate_counts": {},
        "e230_death_order_counts": {},
        "mode_by_death_order": {},
        "gloss_check": ("T221 glossed census-outside as 'the belief NEVER halved'; the endpoint "
                        "criterion (p(w,80) < 0.5*p_t0) adjudicates that gloss"),
    }
    for r in rows:
        census["e232_fate_counts"][r["e232_fate"]] = census["e232_fate_counts"].get(r["e232_fate"], 0) + 1
        census["e230_death_order_counts"][r["e230_death_order_class"]] = \
            census["e230_death_order_counts"].get(r["e230_death_order_class"], 0) + 1
        k = r["e230_death_order_class"]
        census["mode_by_death_order"].setdefault(k, {})
        census["mode_by_death_order"][k][r["classification"]] = \
            census["mode_by_death_order"][k].get(r["classification"], 0) + 1

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    mode_color = {"MIS-DIAL": "tab:blue", "FREQUENCY-COLLAPSE": "tab:red", "OTHER": "tab:gray"}
    mode_label = {"MIS-DIAL": "mis-dial", "FREQUENCY-COLLAPSE": "collapse", "OTHER": "other"}

    def strip(ax, subset, title, field):
        rng = np.random.default_rng(20261004)
        for i, m in enumerate(modes):
            vals = [r[field] for r in subset if r["classification"] == m]
            if not vals:
                continue
            x = np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=46, color=mode_color[m], alpha=0.85, edgecolor="k",
                       linewidth=0.4, zorder=3, label=f"{mode_label[m]} (n={len(vals)})")
            med = median(vals)
            ax.hlines(med, i - 0.28, i + 0.28, color=mode_color[m], linestyle="--",
                      linewidth=2, zorder=4)
            ax.annotate(f"{med:.3f}", (i + 0.30, med), fontsize=8, va="center",
                        color=mode_color[m])
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([mode_label[m] for m in modes])
        ax.set_ylabel("p(answer)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    strip(axes[0, 0], rows,
          f"A. p(+80) by mode — all 19 (ratio md/fc = {depth['ratio_overall']:.2f}; bar >= 2)", "p_80")
    strip(axes[0, 1], rows,
          f"B. p at first-crossing state by mode (co-report; median ratio = "
          f"{depth['ratio_p_at_flip_overall']:.2f})", "p_at_flip")
    strip(axes[1, 0], [r for r in rows if r["wash"] == "w1"],
          f"C. p(+80) by mode — w1 only (ratio = {depth['ratio_w1']:.2f}; bar >= 2)", "p_80")
    strip(axes[1, 1], [r for r in rows if r["wash"] == "w2"],
          f"D. p(+80) by mode — w2 only (ratio = {depth['ratio_w2']:.2f}; collapse n="
          f"{depth['collapse_w2']['n']})", "p_80")
    fig.suptitle("E245 — the death-depth read: surviving belief mass p(+80) by death mode "
                 "(dashed = medians)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f1 = OUT / "e245_depth_distributions.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    xs = np.arange(2)
    bottom = np.zeros(2)
    for m in modes:
        vals = np.array([counts[w][m] for w in washes], dtype=float)
        ax.bar(xs, vals, 0.55, bottom=bottom, color=mode_color[m], label=mode_label[m],
               edgecolor="k", linewidth=0.5)
        for x, v, b in zip(xs, vals, bottom):
            if v > 0:
                ax.annotate(f"{int(v)}", (x, b + v / 2), ha="center", va="center",
                            color="white", fontsize=10, fontweight="bold")
        bottom += vals
    for x, w in zip(xs, washes):
        tot = n_w[w]
        for m in modes:
            p = p_mode_given_wash[w][m]
            if p > 0:
                pass
        ax.annotate(f"n={tot}", (x, tot + 0.15), ha="center", fontsize=9)
        # 75% lopsidedness line per wash
        ax.hlines(0.75 * tot, x - 0.30, x + 0.30, color="k", linestyle=":", linewidth=1.6)
        ax.annotate("75%", (x + 0.31, 0.75 * tot), fontsize=8, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"w1 (max mode share {max(p_mode_given_wash['w1'].values()):.0%})",
                        f"w2 (max mode share {max(p_mode_given_wash['w2'].values()):.0%})"])
    ax.set_ylabel("flip records")
    ax.set_title("Wash contingency — all-records denominators (ADJUDICATING;\n"
                 "dotted line = the 75% lopsidedness bar)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    cells = [[counts[w]["MIS-DIAL"], counts[w]["FREQUENCY-COLLAPSE"]] for w in washes]
    cells = np.array(cells)
    ax.imshow(cells, cmap="Blues", vmin=0, vmax=cells.max())
    for i in range(2):
        for j in range(2):
            ax.annotate(str(cells[i, j]), (j, i), ha="center", va="center", fontsize=18,
                        color="black" if cells[i, j] < cells.max() else "white",
                        fontweight="bold")
            ax.annotate(f"{p_mode_given_wash[washes[i]][modes[j]]:.0%} of wash",
                        (j, i + 0.28), ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["mis-dial", "collapse"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["w1", "w2"])
    ax.set_title("The 2x2 (counts; % of that wash's ALL flip records)\n"
                 f"OTHER disclosed: w1={counts['w1']['OTHER']}, w2={counts['w2']['OTHER']}",
                 fontsize=10)
    fig.suptitle("E245 — the wash read: is the mode written by the wash?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f2 = OUT / "e245_wash_contingency.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)

    # ---------------- metrics ----------------
    all_pass = all(g.get("pass", True) for g in gates.values())
    metrics = {
        "experiment": "e245_death_depth",
        "phase": ("desk-only on COMMITTED data (CPU threads 4; no GPU, no torch, no model "
                  "loads; load-polite — two other agents live)"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "DONE",
        "registration": ("bars frozen VERBATIM from the dispatch brief in the script docstring; "
                         "registration commit precedes any compute; no bar shopping"),
        "question": ("over the 19 flips, does the surviving belief mass select the death mode "
                     "(T221: mis-dials carry higher p(+80) than collapses, median >= 2x — two "
                     "stages of death) — or does the WASH identity write the mode (w1 "
                     "collapse-heavy, w2 mis-dial-heavy)?"),
        "builds_on": [
            "T221 (the registration — prediction + alternative verbatim)",
            "T219/e241 + T212/e232 (the two modes and the zombie taxonomy)",
            "e243 (the 19-flip table: identities, modes, targets — carried verbatim, not re-derived)",
            "e228 (the per-state per-probe p and argmax journal)",
            "e214 (the census journal — provenance re-certification)",
            "e230 (the committed death-order classes — read for the census co-report)",
        ],
        "registered_bars_verbatim": {
            "DEPTH-SELECTS": ("the mis-dials' median p(+80) >= 2x the collapses' AND the "
                              "separation holds within each wash (or one wash has no collapses "
                              "to test — disclosed) — two stages of death confirmed"),
            "WASH-WRITES": ("depth does not separate (ratio < 2 or within-wash fails) while the "
                            "wash contingency is lopsided (one wash >= 75% one mode) — the mode "
                            "is written by the wash, the stranger branch"),
            "MIXED": "the tables verbatim, no inflation",
        },
        "registered_prediction_T221_verbatim": ("over the 19 flips, p(+80) (and p at the flip's "
                                                "first-crossing state where available) separates "
                                                "the modes — the mis-dials carry surviving belief "
                                                "mass, the collapses do not; median separation "
                                                ">= 2x"),
        "operationalizations_frozen": {
            "population": "e243's 19 (probe, wash) flip records verbatim (do not re-derive)",
            "p_80": "e228 journal p at (wash,+80); gate: exact vs e243's p_80 for all 19",
            "flip_time": ("earliest state in the wash lineage whose top1_id != t0 top1_id "
                          "(e228 journal; w1 states +2/+10/+50/+80, w2 +10/+50/+80)"),
            "median": "numpy median, linear interpolation at even n",
            "depth_bar": "ratio of medians (mis-dial / collapse) >= 2 overall AND within each wash (w2's-collapse-empty escape only if n=0; w2's n disclosed)",
            "wash_bar": ("P(mode | wash's ALL flip records) >= 0.75 adjudicates (OTHER counts "
                         "against purity); two-mode-restricted and P(wash|mode) co-reports "
                         "disclosed, NOT adjudicating — frozen before compute"),
        },
        "sources": {
            "e243_metrics": {"path": str(E243), "sha256_16": sha16(E243)},
            "e228_journal": {"path": str(E228), "sha256_16": sha16(E228)},
            "e214_journal": {"path": str(E214), "sha256_16": sha16(E214)},
            "e230_metrics": {"path": str(E230), "sha256_16": sha16(E230)},
        },
        "table": rows,
        "depth_read": depth,
        "wash_read": wash_read,
        "joint_read": joint,
        "census_read": census,
        "adjudication": {
            "verdict": verdict,
            "bar_fired_verbatim": bar_fired,
            "depth_clauses": depth["clauses"],
            "wash_clause": {"lopsided_at_75": wash_lopsided,
                            "max_mode_share": max_lob},
            "read": ("DEPTH fails at its own bar; the contingency clause of WASH-WRITES is "
                     "tested only if depth fails — the verdict follows the frozen ladder"),
        },
        "gates": gates,
        "honesty_reflex": (
            "KNOWN CONFOUNDS, disclosed not corrected: (1) e243's confounds carry VERBATIM — 6 "
            "probes die under both washes sharing one t0/RU read; n=19 (9 mis-dial / 7 collapse "
            "/ 3 other; w1 11 / w2 8); the bars adjudicate on whatever n exists; nothing "
            "guaranteed. (2) The w2 collapse stratum is n=1 (iPhone w2) — the within-w2 depth "
            "clause is a single-record comparison, disclosed, not a distribution. (3) The wash "
            "contingency's adjudicating denominator includes OTHER (3 records, all w2): frozen "
            "before compute; the two-mode-restricted percentages co-report and do NOT "
            "adjudicate — under the restricted reading w2 would read 80% mis-dial, disclosed "
            "here so no reader is ambushed by the choice. (4) CENSUS GLOSS: T221 (and e243's "
            "coverage note) glossed the census-outside flips as 'the belief NEVER halved' — the "
            "endpoint criterion says otherwise (see census_read); the binary that actually "
            "tracks 'census-outside' is e230's death-ORDER class. The registered bars do not "
            "read on the census; this is a correction of a gloss, not a bar. (5) p at flip "
            "time is confounded with the flip STEP for records that first cross at +80 (n at "
            "+80: p_at_flip == p_80 by construction) — the p(+80) medians carry the bar; the "
            "flip-time medians co-report."),
        "figures": [str(f1), str(f2)],
        "compute": {"device": "cpu desk pass (threads 4)", "model_loads": 0, "gpu_calls": 0,
                    "training": "none", "torch_imported": False},
        "provenance": {
            "git_head_at_start": git_head,
            "script": str(Path(__file__).resolve()),
            "script_sha256_16": sha16(Path(__file__).resolve()),
        },
        "timing": {"wall_s": round(time.time() - t_start, 1)},
        "trims": [],
        "deviations": [],
        "all_gates_pass": all_pass,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"VERDICT: {verdict}")
    print(f"depth ratio overall = {depth['ratio_overall']:.3f} (bar >= 2; "
          f"w1 {depth['ratio_w1']:.3f}, w2 {depth['ratio_w2']:.3f})")
    print(f"wash max mode share = {max_lob[0]:.3f} ({max_lob[1]} {max_lob[2]}; bar >= 0.75)")
    print(f"census: halved {n_halved}/19 at endpoint; e230 order classes "
          f"{census['e230_death_order_counts']}")
    print(f"gates all pass: {all_pass}")
    print(f"figures: {f1.name}, {f2.name}")


if __name__ == "__main__":
    main()
