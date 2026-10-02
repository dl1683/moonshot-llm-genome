"""E214 — THE CONSERVATION-OF-RANK CELL (W028's named question).

WHY: W028's wonder card asked the program law's direct question — every
object the lab has dissected splits into a SHAPE (robust, replicating;
lives in ORDERINGS and DESTINATIONS) and a HEIGHT (a lottery; lives in
DISTANCES and RATES): "is the split itself the deepest finding: that
RANK information is conserved under the wash while scale information
is destroyed?" — naming "the conservation-of-rank cell, if it ever
wants to run". T185 (e213) tightened the substrate: at 124M the
mid-dose erosion is a STATE FUNCTION (the same battery decline on two
independent washes to three decimals). THE CELL: take the two-wash
archive with the lab's richest per-probe records and split every
battery's probe vector into the RANK layer (the ORDER of the probes by
p(answer)) and the SCALE layer (their absolute levels) — then ask
where scale dies, does rank survive?

THE ARCHIVE (both washes on disk; this cell is EVAL-ONLY):
  * WASH 1 = e182's original stream (seed 18202) as replayed CPU fp32
    by e182c phase 1 — states runs/checkpoints/e182c_s{2,10,50,80}.pt.
  * WASH 2 = e182c2 phase 2's fresh stream (seed 20261002), GPU fp32
    (TF32 off) — states runs/checkpoints/e182c2_fresh_s{10,50,80}.pt.
  Per-probe p(answer first token) records exist for every battery at
  every state (fact n=20, ctrl n=12, near n=3, tmpl n=19); +2 is
  wash-1-only (loaded from the checkpoint and verified against the
  committed records; the only state without a wash-2 twin).

THE CELL (four readings):
  (1) THE RANK LAYER — within-wash: Spearman's rho of the battery's
      probe p-ranks between t=0 and each t (does the ORDER of the
      probes survive the wash?); cross-wash: rho between the two
      washes' p-vectors at each shared state (is the order the same
      FUNCTION OF STATE — T185's state function read at the rank
      layer?).
  (2) THE SCALE LAYER — the batteries' mean_p declines per wash per
      state (the known-to-erode absolute levels).
  (3) THE DISSOCIATION — the rank-preservation curve vs the
      scale-decay curve on one plot + one table: at the states where
      scale has decayed >= 50%, is rho still >= 0.8?
  (4) THE TINY-SCALE ECHO (load-only, if cheap) — the e185-family
      metrics scanned for committed per-probe records (the same two
      curves at 2.74M); if none exist, note and skip.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any
compute; commit eaed63f / QUEUE row 'DISPATCHED 14:37Z'; adjudicate
against exactly this; no bar shopping):
  - RANK-CONSERVED: "fires if the rank correlations (t=0 vs t, and
    cross-wash) stay high (rho >= 0.8) at states where the scale has
    decayed >= 50% — RANK INFORMATION IS CONSERVED UNDER THE WASH
    WHILE SCALE IS DESTROYED: W028's law licensed as a measurement;
    the shape/height split given an information-theoretic form."
  - RANK-DESTROYED: "fires if the ranks decorrelate with the scale
    (rho tracks the decline) — the shape layer is not rank; the law's
    form is elsewhere; reported honestly."
  - GRADED: "any partial — the curves verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * batteries: fact / ctrl / near are phase-1's VERBATIM (module
    import of lab/e182c_forgetting_control.py); tmpl is phase-2's
    VERBATIM (module import of lab/e182c2_template.py). Identity is
    GATED (G_BATT) against both committed records' t=0 readings.
  * p-vector of battery b at state (wash, s): the per-probe
    p(answer first token) values, probed FRESH here on the LOADED
    state (CPU fp32, the e182c probe path) and certified per-probe
    against the committed records (the re-probe convention; expected
    ~1e-6, tol 0.005).
  * SCALE layer: mean_p_b(wash, s); decline_b(w, s) = 1 -
    mean_p_b(w, s)/mean_p_b(0); t=0 is ONE read of the shared pristine
    organism (both washes start there; both committed t=0s verified).
  * RANK layer: Spearman's rho on average ranks (Pearson-on-ranks —
    exact under ties); within-wash rho_b(w, s) = rho(p_b(0), p_b(w,s));
    cross-wash xrho_b(s) = rho(p_b(w1,s), p_b(w2,s)) at shared states
    {10, 50, 80} (t=0 excluded — the washes share the pristine state
    there and the correlation is trivially 1).
  * ADJUDICATION CELL = (battery, wash, state) with decline >= 0.50
    (SCALE_BAR). A cell HOLDS iff rho_b(w, s) >= RHO_BAR (0.8) AND
    (xrho_b(s) >= RHO_BAR if s is a shared state — every >= 0.50
    decline in this archive sits at a shared state; if one ever did
    not, the cell would adjudicate on the within-wash rho alone,
    DISCLOSED). A cell BREAKS otherwise.
  * RANK-CONSERVED := at least one cell exists AND every cell HOLDS.
    RANK-DESTROYED := at least one cell exists AND every cell BREAKS.
    GRADED := otherwise (some hold, some break; curves verbatim).
  * "rho tracks the decline" (the RANK-DESTROYED texture): per
    battery-wash Spearman(decline, rho) across states + the pooled
    coupling, co-reported.
  * ORDER: RANK-CONSERVED -> RANK-DESTROYED -> GRADED; adjudication
    GATED on the verification gates; if any fails: VERIFICATION-FAILED,
    curves reported, no bar read.
  * CO-REPORTS (never adjudicated, disclosed): (a) the >= 0.40-decline
    echo — the frozen 0.50 line's just-under neighbors (fact/ctrl
    cluster 0.37-0.46 at +80) given the same table at the softer cut,
    labeled NON-ADJUDICATED; (b) Kendall tau-b beside every rho (the
    ties-robust echo); (c) the near battery's n=3 coarseness flagged
    wherever it adjudicates.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_STATES — the two state archives inventoried (sizes/mtimes; the
    wash-1 set cross-checked against e182c2's committed inventory) and
    RE-VERIFY: every battery probed on every LOADED state compared
    per-probe to the committed records (wash-1 fact/ctrl/near vs
    runs/e182c/metrics.json; wash-1 tmpl vs runs/e182c2/metrics.json
    part1_template; wash-2 all four vs runs/e182c2/journal_p2.json;
    the part-1 ctrl/near re-probes at +50/+80 a second witness) within
    0.005 (expected ~1e-6: same fp32 weights, same CPU fp32 probe
    path). THE re-probe convention of e182c2/e213.
  * G_BATT — the four batteries rebuilt VERBATIM reproduce BOTH
    committed t=0 records: same kept sets, per-probe |dp| <= 0.010.
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (feeds the batteries' contamination scans;
    INHERITED FROZEN, never re-filtered).
  * G_ENV — the owner envelope: CPU-only (zero GPU calls), torch
    threads <= 4, load checks before launch and between states, small
    eval bursts (one state = one burst), decisive.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. n=2 washes is texture, not law; wash 1 is a CPU fp32 replay
of e182's GPU original (e182c's G_REPLAY stamp) and wash 2 is GPU fp32
— a device-texture asymmetry between the arms, disclosed, inherited
from the archive; the >= 0.50-decline regime is reached by near (n=3)
and tmpl — the fact/ctrl batteries top out at 0.43/0.40, so the bar's
regime is carried by small/er batteries and the 0.40 echo carries the
n=20 texture (NON-ADJUDICATED); rho on n=3 is quantized to half-points
(and any tie breaks it further) — flagged in every near cell; rank
conservation here means conservation of the p-ORDER of THESE frozen
probe sets under THESE two washes — an instrument-level measurement of
W028's law, not a proof that the shape layer is rank in general; the
2-shot exemplar structure is part of the frozen instrument (orderings
could live partly in the exemplar prefix); cross-wash rho at deep
states partly re-expresses T185's state function (if erosion is
state-typed, both washes' deep states sit near the same attractor —
the within-wash rho is the independently informative half of the bar).

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and between states; small eval
bursts; decisive. ~486 CPU fp32 forwards total (54 prompts x 8
t0/state records + ~83 screening probes) + 7 checkpoint loads; no
wash, no bank ppl (inherited from the records), no NOTES/THINKING/
QUEUE/STATE edits (dispatch). Progressive PARTIAL metrics + resumable
journal after every state (the standing disruption rule).

PROVENANCE: the organism, the frozen corpus filter+verify, the fact/
ctrl/nearrel batteries and probe machinery are lab/e182c_forgetting_
control.py VERBATIM via module import (inheriting lab/e182_gpt2_wash.
py); the template battery is lab/e182c2_template.py VERBATIM via
module import; the two-wash archive and the census loop design are
lab/e213_path_independence.py's (this cell adds the +2 wash-1 state
and swaps the census for the rank/scale split). The committed records
are read at RUNTIME (never transcribed). Builds on: W028 (the named
question), T185/e213 (the state function + the archive), T183/e182c2
(the fresh draw + the two-wash design), T149/e182c (the saved-state
discipline), T123/e182 (the parent wash). NEW: the rank layer (within-
and cross-wash Spearman on the per-probe p-vectors), the scale layer
curves, the dissociation table/plot, the three-rank-bar adjudication,
the >= 0.40 echo, the Kendall echo, the e185 tiny-scale per-probe
scan.

Run:  cd lab && python e214_rank_conservation.py   (E214_SMOKE=1:
      t=0 + the +10 pair only, own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch                                            # noqa: E402

import common                                            # noqa: E402
from common import now_iso, run_dir, save_json            # noqa: E402

import e182c_forgetting_control as e1                    # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                             # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                 # the owner envelope (this dispatch: <= 4)

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import textwrap                                          # noqa: E402

SMOKE = os.environ.get("E214_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e214_smoke" if SMOKE else "e214"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants (registered BEFORE compute) ---------------------
SHARED_STATES: tuple[int, ...] = (10,) if SMOKE else (10, 50, 80)
W1_ONLY_STATES: tuple[int, ...] = () if SMOKE else (2,)   # wash-1-only curve point
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
W1_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c_s{s}.pt"
           for s in SHARED_STATES + W1_ONLY_STATES}
W2_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c2_fresh_s{s}.pt"
           for s in SHARED_STATES}

# ---- registered bar constants (frozen) ----------------------------------------
RHO_BAR = 0.8          # "rank correlations stay high (rho >= 0.8)"
SCALE_BAR = 0.50       # "states where the scale has decayed >= 50%"
ECHO_BAR = 0.40        # the NON-ADJUDICATED >= 0.40 echo (disclosed co-report)

# ---- registered verification tolerances (frozen) -------------------------------
TOL_PROBE_DP = 0.010   # per-probe t=0 dp vs committed records
TOL_STATE_DP = 0.005   # per-probe dp on LOADED states vs committed records

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "RANK-CONSERVED": "fires if the rank correlations (t=0 vs t, and "
            "cross-wash) stay high (rho >= 0.8) at states where the scale "
            "has decayed >= 50% — RANK INFORMATION IS CONSERVED UNDER THE "
            "WASH WHILE SCALE IS DESTROYED: W028's law licensed as a "
            "measurement; the shape/height split given an information-"
            "theoretic form.",
        "RANK-DESTROYED": "fires if the ranks decorrelate with the scale "
            "(rho tracks the decline) — the shape layer is not rank; the "
            "law's form is elsewhere; reported honestly.",
        "GRADED": "any partial — the curves verbatim.",
    },
    "operationalizations": (
        "cell = (battery, wash, state) with decline >= 0.50 where decline_b"
        "(w,s) = 1 - mean_p_b(w,s)/mean_p_b(0) on ONE shared pristine t=0 "
        "read; cell HOLDS iff within-wash rho_b(w,s) >= 0.8 AND (cross-wash "
        "xrho_b(s) >= 0.8 at shared states); rho = Spearman on average "
        "ranks (Pearson-on-ranks, exact under ties) over the battery's "
        "per-probe p(answer) vector, probed fresh on the loaded states and "
        "certified vs the committed records (re-probe tol 0.005); "
        "RANK-CONSERVED := cells exist and all HOLD; RANK-DESTROYED := "
        "cells exist and all BREAK; GRADED := otherwise; order CONSERVED -> "
        "DESTROYED -> GRADED, gated on G_STATES/G_BATT/G_CORPUS/G_ENV; "
        "co-reports (never adjudicated): the >= 0.40 echo, Kendall tau-b, "
        "the near n=3 flags"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (commit "
                     "eaed63f, QUEUE row DISPATCHED 14:37Z) BEFORE any "
                     "compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY on the two archived washes (no wash run here); the +2 "
    "wash-1-only state IS loaded from its checkpoint (e213 co-reported it "
    "from records only) and verified per-probe like every other state — "
    "it enters the curves, never the adjudication (no wash-2 twin).",
    "The rank layer uses the RUNTIME re-probes (from the loaded states), "
    "not the committed journals: same weights, same CPU fp32 probe path — "
    "the committed records are the verification reference (the re-probe "
    "convention), not the data source.",
    "The e185-family tiny-scale echo was scanned LOAD-ONLY: no per-probe "
    "records exist in runs/e185*/metrics.json (aggregates only) — noted "
    "and skipped per the dispatch's 'if the per-probe records don't "
    "exist, note and skip'.",
    "CLAUSE AMENDMENT (reporting only, post-first-run, pre-final-commit): "
    "the RANK-DESTROYED clause discloses the cross-wash half's status "
    "(computed from the cells) — the verdict RULE was frozen before "
    "compute (a cell HOLDS iff BOTH the t=0-vs-t and cross-wash rhos "
    ">= 0.8) and is unchanged; the amendment adds the honest split, it "
    "moves no bar.",
    "PLOT-ONLY FIX: the first full pass wrote DONE metrics + journal and "
    "crashed in the figure (adj keying); the figure and all downstream "
    "records were regenerated VERBATIM from the frozen journal — zero "
    "recompute of any state, zero metric change.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: t=0 + the +10 pair only, own smoke dir, nothing "
    "adjudicated or verified.",
]


# ------------------------------------------------------------------ helpers

def cpu_load_check(tag: str) -> dict:
    """The owner envelope's load check (CPU-only cell; nothing GPU)."""
    try:
        import psutil
        pct = float(psutil.cpu_percent(interval=1.0))
        rec = {"tag": tag, "cpu_percent": pct, "tool": "psutil"}
    except Exception as e:                                # noqa: BLE001
        pct, rec = None, {"tag": tag, "cpu_percent": None,
                          "tool": f"unavailable ({e})"}
    log(f"  [load:{tag}] cpu {pct if pct is None else round(pct, 1)}%")
    return rec


def avg_ranks(v: list[float]) -> list[float]:
    """Average ranks (1-based), exact under ties."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5


def spearman(xs: list[float], ys: list[float]) -> tuple[float | None, int]:
    """Spearman rho = Pearson on average ranks (exact under ties);
    returns (rho, n_tied values across both vectors)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None, 0
    ties = (len(xs) - len({round(x, 12) for x in xs})
            + len(ys) - len({round(y, 12) for y in ys}))
    return _pearson(avg_ranks(xs), avg_ranks(ys)), ties


def kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    """Kendall tau-b (tie-adjusted) — the robustness echo, never adjudicated."""
    n = len(xs)
    if n < 2:
        return None
    con = dis = 0
    xt = yt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            p = dx * dy
            if p > 0:
                con += 1
            elif p < 0:
                dis += 1
            if dx == 0:
                xt += 1
            if dy == 0:
                yt += 1
    denom = ((con + dis + xt) * (con + dis + yt)) ** 0.5
    return (con - dis) / denom if denom > 0 else None


def bat_rec(b: dict) -> dict:
    return {"mean_p": b["mean_p"], "frac_top1": b["frac_top1"],
            "frac_top5": b["frac_top5"],
            "probes": {r["fact"]: {"p": r["p"], "rank": r["rank"]}
                       for r in b["probes"]}}


def pvec(rec: dict, battery: str) -> tuple[list[str], list[float]]:
    """(probe names, p values) in the battery's frozen probe order."""
    pr = rec[battery]["probes"]
    return list(pr.keys()), [pr[f]["p"] for f in pr]


# ------------------------------------------------- the e185 tiny-scale scan

def e185_perprobe_scan() -> dict:
    """LOAD-ONLY scan of runs/e185*/metrics.json for committed per-probe
    records (the dispatch's part 4). A per-probe record = a dict/list whose
    leaves carry per-probe p/pz values (not aggregates)."""
    import glob as _glob
    out = {}
    for f in sorted(_glob.glob(str(common.REPO / "runs" / "e185*"
                                   / "metrics.json"))):
        found = []
        try:
            m = json.loads(Path(f).read_text(encoding="utf-8"))

            def walk(d, path=""):
                if isinstance(d, dict):
                    if (("p" in d) or ("pz" in d)) \
                            and set(d.keys()) <= {"p", "rank", "pz", "z"}:
                        found.append(path)
                    for k, v in d.items():
                        walk(v, path + "/" + str(k))
                elif isinstance(d, list) and d and isinstance(d[0], dict):
                    for i, v in enumerate(d):
                        walk(v, path + f"[{i}]")

            walk(m)
        except Exception as e:                            # noqa: BLE001
            out[str(Path(f).parent.name)] = {"error": str(e)}
            continue
        out[str(Path(f).parent.name)] = {
            "per_probe_records": len(found),
            "note": ("aggregates only (mean_pz/median/std/frac at dose "
                     "points) — no per-probe p vectors committed"
                     if not found else f"{len(found)} per-probe leaves"),
        }
    return out


# ------------------------------------------------------------------ plot

def make_plot(rd, curves, cells, adj, R0):
    """THE FIGURE: (a) THE dissociation scatter (rank vs scale, the bar
    quadrants); (b) the scale layer (mean_p curves, both washes); (c) the
    rank layer (rho curves within-wash + cross-wash); (d) the dissociation
    table + verdict + gates."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) THE DISSOCIATION: rho (y) vs decline (x)
    ax = axes[0, 0]
    ylo, yhi = -0.15, 1.06
    # the conserved-while-destroyed quadrant (decl >= 0.5, rho >= 0.8)
    frac = (RHO_BAR - ylo) / (yhi - ylo)
    ax.axvspan(SCALE_BAR, 0.85, ymin=frac, facecolor="gold", alpha=0.22)
    ax.axvline(SCALE_BAR, color="k", ls="--", lw=1.3)
    ax.axhline(RHO_BAR, color="k", ls="--", lw=1.3)
    ax.text(0.505, 0.022, "scale destroyed >= 50%", fontsize=7, rotation=90,
            va="bottom", family="monospace")
    ax.text(0.002, RHO_BAR + 0.008, "rho = 0.8 bar", fontsize=7,
            family="monospace")
    ax.text(SCALE_BAR + 0.008, RHO_BAR - 0.055,
            "CONSERVED-WHILE-DESTROYED\n(the bar's quadrant)",
            fontsize=7.4, family="monospace", weight="bold",
            color="darkgoldenrod", va="top")
    for b in BATTERIES:
        for wash, mk in (("w1", "o-"), ("w2", "s--")):
            steps_this = [s for s in (SHARED_STATES + W1_ONLY_STATES)
                          if str(s) in curves.get(wash, {})]
            if not steps_this:
                continue
            xs = [curves[wash][str(s)][b]["decl"] for s in steps_this]
            ys = [curves[wash][str(s)][b]["rho_t0"] for s in steps_this]
            ax.plot(xs, ys, mk, ms=8 if wash == "w1" else 7, lw=1.8,
                    color=cols[b], alpha=0.9 if wash == "w1" else 0.65,
                    label=f"{b} {wash} (n={curves['n'][b]})")
            for s, x, y in zip(steps_this, xs, ys):
                ax.annotate(f"+{s}", (x, y), textcoords="offset points",
                            xytext=(4, -8), fontsize=6.4, color=cols[b])
    ax.set_xlabel("SCALE DECAY: decline 1 - R(s)/R(0)")
    ax.set_ylabel("RANK PRESERVATION: Spearman rho vs t=0")
    ax.set_ylim(-0.15, 1.06)
    ax.set_xlim(-0.02, 0.85)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("THE DISSOCIATION — where scale dies (>= 50%), does rank "
                 "survive (rho >= 0.8)?", fontsize=10)

    # (0,1) the scale layer
    ax = axes[0, 1]
    steps_w1 = sorted({s for s in SHARED_STATES + W1_ONLY_STATES
                       if str(s) in curves["w1"]})
    steps_w2 = sorted({s for s in SHARED_STATES if str(s) in curves["w2"]})
    for b in BATTERIES:
        ax.plot(steps_w1, [curves["w1"][str(s)][b]["mean_p"]
                           for s in steps_w1], "o-", ms=7, lw=1.9,
                color=cols[b], label=f"{b} wash 1")
        ax.plot(steps_w2, [curves["w2"][str(s)][b]["mean_p"]
                           for s in steps_w2], "s--", ms=6, lw=1.6,
                color=cols[b], alpha=0.65, label=f"{b} wash 2")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("wash step (+2 = wash-1-only)")
    ax.set_ylabel("SCALE LAYER: mean p(answer) [R(0): "
                  + ", ".join(f"{b} {R0[b]:.2f}" for b in BATTERIES) + "]",
                  fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower left", ncol=2)
    ax.set_title("the scale layer — the absolute levels erode (the known "
                 "story)", fontsize=9.5)

    # (1,0) the rank layer
    ax = axes[1, 0]
    for b in BATTERIES:
        ax.plot(steps_w1, [curves["w1"][str(s)][b]["rho_t0"]
                           for s in steps_w1], "o-", ms=7, lw=1.9,
                color=cols[b], label=f"{b}: rho vs t=0 (both washes)")
        ax.plot(steps_w2, [curves["w2"][str(s)][b]["rho_t0"]
                           for s in steps_w2], "s--", ms=6, lw=1.6,
                color=cols[b], alpha=0.65)
        ax.plot(steps_w2, [curves["xwash"][str(s)][b] for s in steps_w2],
                "^:", ms=7, lw=1.6, color=cols[b], alpha=0.85,
                label=f"{b}: cross-wash rho" if b == "fact" else None)
    ax.axhline(RHO_BAR, color="k", ls="--", lw=1.2, label="rho = 0.8 bar")
    ax.set_xlabel("wash step")
    ax.set_ylabel("RANK LAYER: Spearman rho (circles w1, squares w2, "
                  "triangles cross-wash)")
    ax.set_ylim(-0.15, 1.06)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower left", ncol=2)
    ax.set_title("the rank layer — does the probes' ORDER survive? "
                 "(^ = cross-wash: same order as a function of state)",
                 fontsize=9.5)

    # (1,1) the dissociation table + verdict
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE DISSOCIATION TABLE (decl >= 0.50 cells, the bar's "
            "regime; the 0.40 echo beneath):", fontsize=8.8, va="top",
            family="monospace", weight="bold")
    y -= 0.024
    ax.text(0.02, y, "  battery   wash  state   decl   rho_t0  xrho   tau_b"
                     "   cell", fontsize=7.0, va="top", family="monospace")
    y -= 0.020
    for c in cells:
        flag = " [n=3 coarse]" if c["battery"] == "near" else ""
        ax.text(0.02, y, f"  {c['battery']:8s} {c['wash']:4s}  +{c['state']:<4d}"
                         f" {c['decl']:6.3f}  {c['rho_t0']:6.3f} "
                         f"{c['xrho'] if c['xrho'] is not None else float('nan'):6.3f}"
                         f" {c['tau_t0'] if c['tau_t0'] is not None else float('nan'):6.3f}"
                         f"   {c['cell_holds_flag']}{flag}",
                fontsize=7.0, va="top", family="monospace",
                color="black" if c["cell_holds_flag"] == "HOLDS" else "darkred")
        y -= 0.0188
        if y < 0.44:
            break
    y = 0.44
    echo_txt = ("  0.40 echo (NON-ADJUDICATED): "
                + "; ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                            f"decl {c['decl']:.3f} rho {c['rho_t0']:.3f}"
                            for c in adj["echo40_cells"]))
    for wd in textwrap.wrap(echo_txt, width=100,
                            break_long_words=False)[:4]:
        ax.text(0.02, y, wd, fontsize=6.2, va="top", family="monospace",
                color="dimgray")
        y -= 0.0165
    y = 0.36
    ax.text(0.02, y, f"E214 VERDICT: {adj['bars']['verdict']}", fontsize=9.6,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(adj["bars"]["clause"], width=94,
                            break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.020
    y -= 0.004
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in adj["gates_summary"].items()), fontsize=7.0, va="top",
        family="monospace")

    fig.suptitle("E214 — THE CONSERVATION-OF-RANK CELL (W028's named "
                 "question): rank vs scale under the two washes -> "
                 f"{adj['bars']['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "rank_conservation.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E214 — THE CONSERVATION-OF-RANK CELL (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e214_rank_conservation",
        "phase": "eval-only on the two archived washes (the rank/scale "
                 "split of the per-probe records)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("W028's named question: is RANK information (the "
                     "probes' p-ORDER) conserved under the wash while "
                     "SCALE information (their absolute levels) is "
                     "destroyed?"),
        "builds_on": ["W028 (the named question; the shape/height law)",
                      "T185 / e213 (the mid-dose state function + the "
                      "two-wash archive this cell reads)",
                      "T183 / e182c2 (the fresh draw, the two-wash design)",
                      "T149 / e182c (the saved-state discipline + the "
                      "batteries)",
                      "T123 / e182 (the parent wash)"],
        "whats_new": ["the rank layer: within-wash Spearman(p(0), p(t)) "
                      "per battery per wash per state",
                      "the cross-wash rho at each shared state (the order "
                      "as a function of state)",
                      "the scale layer curves; THE dissociation table + "
                      "plot (rho vs decline, the 0.8/0.5 quadrants)",
                      "the three-rank-bar adjudication + the 0.40 echo + "
                      "the Kendall tau-b echo",
                      "the e185 tiny-scale per-probe scan (load-only)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    p1_path = common.REPO / "runs" / "e182c" / "metrics.json"
    c2_path = common.REPO / "runs" / "e182c2" / "metrics.json"
    j2_path = common.REPO / "runs" / "e182c2" / "journal_p2.json"
    for p in (p1_path, c2_path, j2_path):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    p1m = json.loads(p1_path.read_text(encoding="utf-8"))
    c2m = json.loads(c2_path.read_text(encoding="utf-8"))
    j2 = json.loads(j2_path.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}          # fact/ctrl/near
    w1_tmpl_rec = {s["step"]: s                               # tmpl (part 1)
                   for s in c2m["part1_template"]["states"]}
    w2_rec = {s["step"]: s for s in j2["states"]}            # all four
    for s_ in SHARED_STATES + W1_ONLY_STATES:
        assert s_ in w1_rec and s_ in w1_tmpl_rec, \
            f"committed wash-1 records lack state +{s_}"
    for s_ in SHARED_STATES:
        assert s_ in w2_rec, f"committed wash-2 records lack state +{s_}"
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]

    # ------------------------------------------ P1 G_STATES part A: inventory
    def inv(p: Path):
        return {"path": str(p), "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                    p.stat().st_mtime)) if p.exists() else None}
    w1_inv = {str(s): inv(p) for s, p in W1_ARCH.items()}
    w2_inv = {str(s): inv(p) for s, p in W2_ARCH.items()}
    c2_inv = c2m.get("inventory", {}).get("files", {})
    w1_match = {str(s): (bool(c2_inv.get(str(s)))
                         and c2_inv[str(s)]["size_bytes"]
                         == w1_inv[str(s)]["size_bytes"]
                         and c2_inv[str(s)]["mtime"]
                         == w1_inv[str(s)]["mtime"])
                for s in W1_ARCH}
    G_STATES_A = {
        "wash1_files": w1_inv, "wash2_files": w2_inv,
        "wash1_crosscheck_vs_e182c2_inventory": w1_match,
        "note": "wash 1 = e182c_s* (seed 18202, CPU fp32 replay, incl. the "
                "+2 wash-1-only state); wash 2 = e182c2_fresh_s* (seed "
                "20261002, GPU fp32); sizes/mtimes must match e182c2's "
                "committed inventory for the wash-1 set",
    }
    all_exist = all(v["exists"] for v in
                    list(w1_inv.values()) + list(w2_inv.values()))
    log(f"G_STATES inventory: wash1 {sorted(w1_inv)} wash2 {sorted(w2_inv)} "
        f"all on disk = {all_exist}; wash-1 crosscheck = {w1_match}")
    metrics["inventory"] = G_STATES_A
    write_metrics("PARTIAL: records read; inventory taken")

    # the e185 tiny-scale echo scan (load-only, part 4)
    e185_scan = e185_perprobe_scan()
    metrics["tiny_scale_echo"] = {
        "scan": e185_scan,
        "verdict": "NOTED AND SKIPPED — no per-probe records committed in "
                   "any runs/e185*/metrics.json (aggregates only); the "
                   "2.74M rank/scale curves cannot be drawn load-only "
                   "(the dispatch's escape clause)",
    }
    log("e185 tiny-scale echo: " + metrics["tiny_scale_echo"]["verdict"])

    # ---------------------------------------------------- P2 the organism
    load_checks = [cpu_load_check("launch")]
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = org_meta
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"
    metrics["size_gate"] = G_SIZE

    # ------------------------------------------- P3 the frozen corpus (G_CORPUS)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in e1.POOLS
                     for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    base_cand = e1.probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    kept_facts, _sel = e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "lines_total": [G_STR["lines_total"], e_str["lines_total"]],
        "lines_dropped": [G_STR["lines_dropped"], e_str["lines_dropped"]],
        "chars_after": [corpus_stats["chars_after"], e_corp["chars_after"]],
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "bank_windows": [corpus_stats["bank_windows"],
                         e_corp["bank_windows"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN — it feeds the batteries' "
                "contamination scans only; no wash is run here",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"]
        and corpus_stats["bank_windows"] == e_corp["bank_windows"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens; e182 "
        f"{e_corp['tokens_after']})")
    assert G_CORPUS["pass"] or SMOKE, f"corpus rebuild diverged: {G_CORPUS}"
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS}

    # --------------------------------- P4 the four batteries, VERBATIM (G_BATT)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]

    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]

    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}

    # -------------------------------------------- P5 the state reads (the cell)
    states_journal = []
    if jp.exists():
        try:
            states_journal = json.loads(
                jp.read_text(encoding="utf-8"))["states"]
            log(f"journal: {len(states_journal)} records restored")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); recomputing")
            states_journal = []
    done = {(r["wash"], r["step"]) for r in states_journal}

    def probe_all(evl):
        return {b: bat_rec(e1.probe_battery(evl, bats[b]))
                for b in BATTERIES}

    def reprobe_dps(wash, step, cur):
        """Per-probe dp of my runtime probes vs the committed records (the
        re-probe convention). Includes the part-1 ctrl/near re-probe
        witnesses at +50/+80."""
        out = {}
        if wash == "w1":
            for b in ("fact", "ctrl", "near"):
                ref = w1_rec[step][b]["probes"]
                out[b] = max(abs(cur[b]["probes"][f]["p"] - v["p"])
                             for f, v in ref.items())
            ref = w1_tmpl_rec[step]["tmpl"]["probes"]
            out["tmpl"] = max(abs(cur["tmpl"]["probes"][f]["p"] - v["p"])
                              for f, v in ref.items())
            if step in (50, 80) and not SMOKE:
                for b, key in (("ctrl", "ctrl_reprobe"),
                               ("near", "near_reprobe")):
                    wit = w1_tmpl_rec[step].get(key)
                    if wit:
                        out[f"{b}_witness"] = max(
                            abs(cur[b]["probes"][f]["p"] - v["p"])
                            for f, v in wit["probes"].items())
        else:
            for b in BATTERIES:
                ref = w2_rec[step][b]["probes"]
                out[b] = max(abs(cur[b]["probes"][f]["p"] - v["p"])
                             for f, v in ref.items())
        return out

    # t=0 (the shared pristine baseline)
    if ("t0", 0) not in done:
        cur = probe_all(net0)
        rec = {"wash": "t0", "step": 0, **cur}
        states_journal.append(rec)
        done.add(("t0", 0))
        jp.write_text(json.dumps({"states": states_journal}, indent=1),
                      encoding="utf-8")
    t0_rec = [r for r in states_journal if r["wash"] == "t0"][0]

    # G_BATT: t=0 vs BOTH committed records (idempotent off the journal)
    cur0 = {b: t0_rec[b] for b in BATTERIES}
    gb = {}
    for b in ("fact", "ctrl", "near"):
        ref = w1_rec[0][b]["probes"]
        gb[b] = {"kept_set_equal": bool(
                     {r["fact"] for r in bats[b]} == set(ref)),
                 "n": len(ref),
                 "max_dp_vs_p1": max(abs(cur0[b]["probes"][f]["p"]
                                         - v["p"])
                                     for f, v in ref.items())}
    ref = w1_tmpl_rec[0]["tmpl"]["probes"]
    gb["tmpl"] = {"kept_set_equal": bool(
                     {r["fact"] for r in bats["tmpl"]} == set(ref)),
                  "n": len(ref),
                  "max_dp_vs_p1": max(abs(cur0["tmpl"]["probes"][f]["p"]
                                          - v["p"])
                                      for f, v in ref.items())}
    for b in BATTERIES:
        ref2 = w2_rec[0][b]["probes"]
        gb[b]["max_dp_vs_p2"] = max(abs(cur0[b]["probes"][f]["p"]
                                        - v["p"])
                                    for f, v in ref2.items())
        gb[b]["kept_set_equal_vs_p2"] = bool(
            {r["fact"] for r in bats[b]} == set(ref2))
    G_BATT = {
        **gb,
        "tol_per_probe_dp": TOL_PROBE_DP,
        "note": "the four batteries are the phase-1/phase-2 pools "
                "VERBATIM (module import); t=0 must reproduce BOTH "
                "committed records (same pristine organism, same "
                "prompts, same CPU fp32 probe path)",
    }
    G_BATT["pass"] = bool(
        all(G_BATT[b]["kept_set_equal"]
            and G_BATT[b]["max_dp_vs_p1"] <= TOL_PROBE_DP
            and G_BATT[b]["max_dp_vs_p2"] <= TOL_PROBE_DP
            for b in BATTERIES)) if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} "
        + " | ".join(f"{b}: dp1 {G_BATT[b]['max_dp_vs_p1']:.2e} "
                     f"dp2 {G_BATT[b]['max_dp_vs_p2']:.2e}"
                     for b in BATTERIES))
    R0 = {b: t0_rec[b]["mean_p"] for b in BATTERIES}
    log("t=0 shared baseline: " + " | ".join(
        f"{b} {R0[b]:.4f} (n={len(bats[b])})" for b in BATTERIES))

    # the state loop (wash x state; one state = one eval burst)
    ck_map = {"w1": W1_ARCH, "w2": W2_ARCH}
    state_list = ([("w1", s_) for s_ in SHARED_STATES + W1_ONLY_STATES]
                  + [("w2", s_) for s_ in SHARED_STATES])
    for wash, s_ in state_list:
        if (wash, s_) in done:
            continue
        load_checks.append(cpu_load_check(f"{wash}s{s_}"))
        f = ck_map[wash][s_]
        sd = torch.load(f, map_location=CPU,
                        weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        cur = probe_all(evl)
        dps = reprobe_dps(wash, s_, cur)
        del evl, sd
        rec = {"wash": wash, "step": s_, "reprobe_max_dp": dps, **cur}
        states_journal.append(rec)
        done.add((wash, s_))
        states_journal.sort(key=lambda r: (r["wash"], r["step"]))
        jp.write_text(json.dumps({"states": states_journal}, indent=1),
                      encoding="utf-8")
        log(f"  {wash.upper()} STATE +{s_}: " + " | ".join(
            f"{b} {cur[b]['mean_p']:.4f} (decl "
            f"{1 - cur[b]['mean_p'] / R0[b]:.4f}, dp "
            f"{dps[b]:.2e})" for b in BATTERIES))
        write_metrics(f"PARTIAL: states read through {wash} +{s_}")
    st = {(r["wash"], r["step"]): r for r in states_journal}

    # G_STATES part B: the re-probe max dp over every loaded state
    all_dps = [dp for r in states_journal
               if r["wash"] in ("w1", "w2")
               for k, dp in r["reprobe_max_dp"].items()
               if not k.endswith("_witness")]
    wit_dps = [dp for r in states_journal
               if r["wash"] in ("w1", "w2")
               for k, dp in r["reprobe_max_dp"].items()
               if k.endswith("_witness")]
    G_STATES = {**G_STATES_A, "all_exist": all_exist,
                "reprobe_max_dp": max(all_dps) if all_dps else None,
                "witness_reprobe_max_dp": (max(wit_dps)
                                           if wit_dps else None),
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_note": "every battery re-probed on every LOADED "
                                "state of BOTH archives (incl. the wash-1-"
                                "only +2) and compared per-probe to the "
                                "committed records (wash-1 fact/ctrl/near "
                                "vs e182c; wash-1 tmpl vs e182c2 part 1; "
                                "wash-2 all four vs e182c2 journal_p2; the "
                                "part-1 ctrl/near re-probes at +50/+80 a "
                                "second witness) — same fp32 weights + "
                                "same CPU fp32 probe path -> expected ~0.0"}
    G_STATES["pass"] = bool(all_exist and (SMOKE or (all_dps
                                                     and max(all_dps)
                                                     <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "load_checks": len(load_checks), "gpu_calls": 0,
             "burst": "one state = one eval burst",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_STATES re-probe: {len(all_dps)} battery-state checks, max dp "
        f"{max(all_dps) if all_dps else 0:.8f} (tol {TOL_STATE_DP})"
        + (f"; witnesses max {max(wit_dps):.2e}" if wit_dps else ""))

    # --------------------------------------------- P6 the rank/scale split
    # THE SCALE LAYER + within-wash RANK LAYER
    curves = {"w1": {}, "w2": {}, "xwash": {}, "n": {b: len(bats[b])
                                                     for b in BATTERIES}}
    for wash in ("w1", "w2"):
        steps = [s_ for s_ in (SHARED_STATES + W1_ONLY_STATES)
                 if (wash, s_) in st] if wash == "w1" else \
                [s_ for s_ in SHARED_STATES if (wash, s_) in st]
        for s_ in steps:
            entry = {}
            for b in BATTERIES:
                names0, p0 = pvec(t0_rec, b)
                names_s, ps = pvec(st[(wash, s_)], b)
                assert names0 == names_s, \
                    f"probe identity drift {b} {wash} +{s_}"
                rho, _ties = spearman(p0, ps)
                tau = kendall_tau_b(p0, ps)
                entry[b] = {
                    "mean_p": st[(wash, s_)][b]["mean_p"],
                    "retention": st[(wash, s_)][b]["mean_p"] / R0[b],
                    "decl": 1 - st[(wash, s_)][b]["mean_p"] / R0[b],
                    "rho_t0": rho, "tau_t0": tau,
                    "probes": dict(zip(names_s, ps)),
                }
            curves[wash][str(s_)] = entry
    # THE CROSS-WASH RANK LAYER (shared states; t=0 excluded — trivial)
    for s_ in SHARED_STATES:
        entry = {}
        for b in BATTERIES:
            _, pw1 = pvec(st[("w1", s_)], b)
            _, pw2 = pvec(st[("w2", s_)], b)
            rho, _t = spearman(pw1, pw2)
            entry[b] = rho
            entry[f"{b}_tau"] = kendall_tau_b(pw1, pw2)
        curves["xwash"][str(s_)] = entry
    metrics["curves"] = curves

    # --------------------------------------------- P7 the dissociation table
    diss_rows = []
    for wash in ("w1", "w2"):
        for s_ in sorted(curves[wash], key=int):
            for b in BATTERIES:
                e = curves[wash][s_][b]
                diss_rows.append({
                    "battery": b, "wash": wash, "state": int(s_),
                    "n": curves["n"][b],
                    "mean_p": e["mean_p"], "decl": e["decl"],
                    "rho_t0": e["rho_t0"], "tau_t0": e["tau_t0"],
                    "xrho": (curves["xwash"][s_][b]
                             if s_ in curves["xwash"] else None),
                    "xtau": (curves["xwash"][s_].get(f"{b}_tau")
                             if s_ in curves["xwash"] else None),
                })
    metrics["dissociation_rows"] = diss_rows

    # the RELATION-BLOCK texture (co-report): the multi-relation batteries'
    # per-relation mean p at t=0 and the deepest state, both washes — the
    # axis the wash actually orders along (computed, never asserted)
    rel_tex = {}
    rel_of = {b: {r["fact"]: r["relation"] for r in bats[b]}
              for b in ("fact", "ctrl")}
    for b in ("fact", "ctrl"):
        rel_tex[b] = {}
        for wash in ("t0", "w1", "w2"):
            src = t0_rec if wash == "t0" else st.get((wash, 80))
            if src is None:
                continue
            agg: dict[str, list[float]] = {}
            for f, d in src[b]["probes"].items():
                agg.setdefault(rel_of[b][f], []).append(d["p"])
            rel_tex[b][wash] = {r: round(sum(v) / len(v), 4)
                                for r, v in sorted(agg.items())}
    metrics["relation_texture"] = {
        **rel_tex,
        "note": "co-report: the fact battery's t=0 order interleaves "
                "relations while the wash sorts BY relation (lang holds, "
                "cap/cur collapse — regardless of baseline p; e.g. "
                "China->yuan p0 0.93 dies to 0.16 while China->Chinese "
                "p0 0.57 holds 0.57); ctrl splits identically "
                "(founders/unique-anchor hold, products collapse). The "
                "erosion order lives on the wash's own axes — which is "
                "why the t=0 order dies while the cross-wash order "
                "replicates",
    }

    # the adjudication cells: decline >= SCALE_BAR
    cells = []
    for r in diss_rows:
        if r["decl"] >= SCALE_BAR and r["state"] > 0:
            shared = str(r["state"]) in curves["xwash"]
            holds = bool(r["rho_t0"] is not None
                         and r["rho_t0"] >= RHO_BAR
                         and ((r["xrho"] is not None
                               and r["xrho"] >= RHO_BAR) if shared
                              else True))
            cells.append({**r, "shared_state": shared,
                          "cell_holds_flag": "HOLDS" if holds else "BREAKS",
                          "n3_coarse_flag": r["battery"] == "near"})
    # the 0.40 echo (NON-ADJUDICATED co-report)
    echo40 = [r for r in diss_rows
              if ECHO_BAR <= r["decl"] < SCALE_BAR and r["state"] > 0]

    # "rho tracks the decline" texture (per battery-wash + pooled)
    tracking = {}
    pooled_d, pooled_r = [], []
    for wash in ("w1", "w2"):
        for b in BATTERIES:
            ds = [curves[wash][s_][b]["decl"] for s_ in
                  sorted(curves[wash], key=int) if int(s_) > 0]
            rs = [curves[wash][s_][b]["rho_t0"] for s_ in
                  sorted(curves[wash], key=int) if int(s_) > 0]
            tracking[f"{wash}:{b}"] = spearman(ds, rs)[0]
            pooled_d += ds
            pooled_r += rs
    tracking["pooled"] = spearman(pooled_d, pooled_r)[0]

    # --------------------------------------------- P8 adjudication (frozen)
    gates_ok = bool(G_STATES["pass"] and G_BATT["pass"]
                    and G_CORPUS["pass"] and G_ENV["pass"])
    n_cells = len(cells)
    n_hold = sum(1 for c in cells if c["cell_holds_flag"] == "HOLDS")

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (curves reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_STATES", G_STATES["pass"]),
                           ("G_BATT", G_BATT["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif n_cells == 0:
        verdict = "GRADED"
        clause = ("no battery-wash-state reached decline >= 0.50 in this "
                  "archive — the bar's regime is empty; curves verbatim "
                  "(the 0.40 echo beneath)")
    elif n_hold == n_cells:
        verdict = "RANK-CONSERVED"
        clause = (f"the rank correlations (t=0 vs t, and cross-wash) stay "
                  f"high (rho >= {RHO_BAR}) at all {n_cells} states where "
                  f"the scale has decayed >= {int(SCALE_BAR*100)}% "
                  f"(declines "
                  + ", ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"{c['decl']:.3f} @ rho {c['rho_t0']:.3f}"
                              for c in cells)
                  + ") — RANK INFORMATION IS CONSERVED UNDER THE WASH "
                  "WHILE SCALE IS DESTROYED: W028's law licensed as a "
                  "measurement; the shape/height split given an "
                  "information-theoretic form")
    elif n_hold == 0:
        verdict = "RANK-DESTROYED"
        xhalf = sum(1 for c in cells
                    if c["xrho"] is not None and c["xrho"] >= RHO_BAR)
        clause = (f"the ranks decorrelate with the scale — at every one of "
                  f"the {n_cells} states with decline >= "
                  f"{int(SCALE_BAR*100)}%, the t=0-vs-t rho has fallen "
                  f"below {RHO_BAR} ("
                  + ", ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"decl {c['decl']:.3f} rho "
                              f"{c['rho_t0']:.3f}"
                              for c in cells)
                  + "); the decline-vs-rho coupling "
                  f"(pooled Spearman) {tracking['pooled']:.2f} — the "
                  "shape layer is not rank; the law's form is elsewhere; "
                  "reported honestly. DISCLOSED SPLIT: the bar's CROSS-"
                  f"WASH half HELD at {xhalf}/{n_cells} cells (xrho "
                  + ", ".join(f"{c['xrho']:.3f}" for c in cells)
                  + ") — the deep-state ORDER is wash-path-independent "
                  "(T185's state function read on the full p-vector); "
                  "what dies is the ORIGIN's order, not order per se")
    else:
        verdict = "GRADED"
        clause = (f"partial dissociation: {n_hold}/{n_cells} of the "
                  f"decline >= {int(SCALE_BAR*100)}% cells hold rho >= "
                  f"{RHO_BAR} ("
                  + ", ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"decl {c['decl']:.3f} rho "
                              f"{c['rho_t0']:.3f} "
                              f"{'HOLDS' if c['cell_holds_flag'] == 'HOLDS' else 'BREAKS'}"
                              for c in cells)
                  + "); the 0.40 echo ("
                  + "; ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"decl {c['decl']:.3f} rho "
                              f"{c['rho_t0']:.3f}" for c in echo40)
                  + ") — NON-ADJUDICATED; the curves verbatim")

    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {"RANK_CONSERVED": bool(n_cells and n_hold == n_cells),
                 "RANK_DESTROYED": bool(n_cells and n_hold == 0),
                 "verdict": verdict, "clause": clause,
                 "order": "RANK-CONSERVED -> RANK-DESTROYED -> GRADED "
                          "(gated on G_STATES/G_BATT/G_CORPUS/G_ENV)"},
        "bar_constants": {"RHO_BAR": RHO_BAR, "SCALE_BAR": SCALE_BAR,
                          "ECHO_BAR (non-adjudicated)": ECHO_BAR},
        "cells": cells, "n_cells": n_cells, "n_hold": n_hold,
        "echo40_cells": echo40,
        "echo40_note": "co-report at the softer >= 0.40 cut — the frozen "
                       "0.50 bar's just-under neighbors (fact/ctrl top out "
                       "0.43/0.40 at +80); NOT adjudicated, no bar shopping",
        "tracks_decline": tracking,
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E214 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")
    log(f"  cells: {n_cells}, holding {n_hold}; echo40 cells: "
        f"{len(echo40)}; tracks-decline pooled "
        f"{tracking['pooled'] if tracking['pooled'] is not None else 'n/a'}")

    # ---------------------------------------------------- P9 honesty + close
    metrics["honesty_reflex"] = {
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "a device-texture asymmetry between the arms, "
                    "inherited from the archive (disclosed)",
        "bar_regime_small_n": f"the >= 0.50-decline regime is reached by "
                    f"{sorted({c['battery'] for c in cells})} — near is "
                    "n=3 and tmpl n=19; fact (n=20) and ctrl (n=12) top "
                    "out below the 0.50 line (the 0.40 echo carries their "
                    "texture, NON-ADJUDICATED)",
        "n3_quantization": "Spearman on n=3 is quantized to half-points "
                           "({-1,-0.5,0.5,1} untied; any tie breaks it "
                           "further) — every near cell flagged coarse; its "
                           "weight in the adjudication is exactly its 3 "
                           "probes",
        "t0_shared": "both washes' t=0 is ONE read of the same pristine "
                     "organism (the committed wash-specific t=0s verified "
                     "identical to ~1e-6 via G_BATT); within-wash rho for "
                     "both washes is measured against that single read",
        "crosswash_state_function": "high cross-wash rho at deep states "
                    "partly re-expresses T185's state function (if erosion "
                    "is state-typed, both washes sit near the same "
                    "deep-state attractor); the WITHIN-wash rho is the "
                    "independently informative half of the bar",
        "what_rank_means_here": "conservation of the p-ORDER of these "
                    "frozen probe sets under these two washes — an "
                    "instrument-level measurement of W028's law on the "
                    "124M archive, not a proof that the shape layer is "
                    "rank in general; the 2-shot exemplar structure is "
                    "part of the frozen instrument (orderings could live "
                    "partly in the exemplar prefix)",
        "rho_floor_effect": "batteries saturate near t=0 (p 0.59-0.91) — "
                    "the p-vectors carry baseline heterogeneity that rho "
                    "inherits; Kendall tau-b co-reported as the "
                    "ties-robust echo (never adjudicated)",
        "relation_sorting": "the within-wash rho death is partly a BLOCK "
                    "SORT, not item noise: the wash re-orders the fact "
                    "battery BY RELATION (lang holds, cap/cur collapse, "
                    "baseline p irrelevant) and ctrl by anchor type "
                    "(see relation_texture) — 'rank destroyed' means the "
                    "ORIGIN's order is overwritten by the wash's own "
                    "ordering, which itself replicates (the cross-wash "
                    "half)",
        "plus2_w1_only": "+2 is a wash-1-only state (in the curves, never "
                         "the adjudication — no wash-2 twin)",
        "tiny_scale_echo": "the e185 family committed NO per-probe records "
                    "(aggregates only, scan recorded in metrics) — the "
                    "2.74M echo is noted and skipped per the dispatch's "
                    "escape clause; a future cell would have to re-probe "
                    "the archived e185 states",
        "guarantees_nothing": "rank holding at deep states does not prove "
                    "conservation in the mutual-information sense (the "
                    "probe sets are small and frozen); rank breaking does "
                    "not kill W028's law (the shape layer may live in "
                    "finer structure than battery-level order); the "
                    "openness is the point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, one state = one eval burst, "
                    "decisive; ~54 prompts x 8 records + screening "
                    "forwards, 7 checkpoint loads; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(p) for p in
                          list(W1_ARCH.values()) + list(W2_ARCH.values())],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, curves, cells, adj, R0)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
