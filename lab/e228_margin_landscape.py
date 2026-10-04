"""E228 — W031's MARGIN LANDSCAPE vs THE EROSION ORDER (eval-only, CPU).

WHY: W031 (the freshest wonder card) crossed T204's floor probe with
e214's erosion order and registered this exact cell as its
discriminating observation. T204 left the registered prediction that
argmax decisions flip under mere batch-shape re-rounding exactly where
their margins go thin (~0.05 sigma and below) — the MARGIN is the dial
that maps an organism's commitment spectrum. e214 found the EROSION
ORDER is the SHAPE layer — the conserved object under the wash,
relational, replicating at rho 0.94-1.00 across independent washes,
while the t=0 baseline order dies with the scale (coupling -0.88). THE
QUESTION: is the conserved erosion order the SHADOW of the margin
landscape — do the thin-margin probes die first under the wash? Or
does the erosion order outrun local decision fragility (the stranger
read: the order is real, replicated, and deeper than the organism's
thin spots)?

THE e208 DISTINCTION (owed, verbatim from W031): e208's margin object
was the FACT-EDGE over the wash band on the tiny-net organisms — the
2x noise-margin line as a cross-organism CLASS separator for
first-wash survival — an instrument that died its honest scope death
(W029: scoped to NATURAL-STEP organisms). THIS cell's object is the
NEXT-TOKEN ARGMAX MARGIN at the answer position in SIGMA units vs
arithmetic noise (top1 logit - top2 logit, divided by the std of the
full-vocab logits at that position), per PROBE, on the 124M archive —
a different ruler that shares only the beloved word "margin". Do not
conflate them. Not a resurrection of e208.

THE CELL (eval-only, CPU, committed checkpoints only):
  (1) PER-PROBE MARGINS AT t=0 — for every probe of the four frozen
      batteries (fact n=20, ctrl n=12, near n=3, tmpl n=19 — the e214
      conventions; batteries module-imported from
      lab/e182c_forgetting_control.py and lab/e182c2_template.py,
      VERBATIM, never retyped), the argmax margin at the answer
      position on the t=0 state of each wash lineage. Both wash
      lineages start from ONE pristine organism (e214's G_BATT
      verified the two committed t=0s identical to ~1e-6) — so t=0 is
      ONE read, shared, disclosed (the t0_shared convention).
  (2) THE JOIN — margin rank at t=0 vs e214's COMMITTED per-probe
      erosion records (runs/e214/metrics.json curves + journal t0,
      read at runtime, never transcribed): Spearman rho per battery
      per wash-lineage, with the wash-to-wash replication of the
      EROSION order itself (e214's rho 0.94-1.00 on the p-vectors)
      quoted as the reference floor, AND recomputed at the exact join
      cells on the erosion vectors.
  (3) THE SECOND, QUIETER READ (the law's version) — the margin
      landscape's OWN shape-conservation: margin ORDER at t=0 vs at
      +50/+80 within each wash (Spearman), cross-wash margin-order
      rho, and whether margin LEVELS fall while margin ORDER holds —
      W028's shape/height split applied to the commitment spectrum.
  (4) THE HONESTY GUARDS — the e208 distinction above; margins are
      deterministic in-session (T204: same code path, same device ->
      bit-exact) so n=1 evals suffice, with the eval's provenance
      recorded and every state re-probe-certified against e214's
      committed per-probe p (the re-probe convention, tol 0.005);
      nothing here is guaranteed.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any
compute; commit a58fa48 / QUEUE row 'DISPATCHED ~13:25Z Oct-4';
adjudicate against exactly this; no bar shopping):
  - MARGIN-ORDERS-EROSION — "margin rank at t=0 predicts the erosion
    order (rho >= the wash-to-wash replication floor's lower edge,
    ~0.9) — the conserved order acquires a mechanism candidate: local
    decision fragility" (fires only at the floor; a middling rho is
    NOT this bar)
  - ORDER-OUTRUNS-FRAGILITY — "margin rank does not predict the
    erosion order (rho clearly below the floor, e.g. < 0.6) while the
    erosion order still replicates — the order is deeper than local
    thinness; the STRANGER read"
  - PARTIAL — "anything between — the table verbatim, both batteries,
    no narrative inflation"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * margin_i(state) = (top1 logit - top2 logit) / std(logits over the
    vocab) at the prompt's last position (the answer position), on the
    SAME forward as the p re-probe (logits[0, -1]; torch std,
    unbiased). The object is the ORGANISM'S OWN DECISION margin
    (argmax vs runner-up), whether or not the argmax is the answer.
  * EROSION (the y-side, COMMITTED records): primary erosion_i(w,s) =
    1 - p_i(w,s)/p_i(0) — e214's battery-level decline formula applied
    per probe (p_i(0) = e214's committed journal t0). Echo: absolute
    drop p_i(0) - p_i(w,s), co-reported.
  * JOIN STATISTIC, SIGNED so POSITIVE = thin margins die first:
    join_rho(b,w,s) = Spearman(margin_i(0), retention_i(w,s)) with
    retention_i = p_i(w,s)/p_i(0) (identical to -Spearman(margin,
    erosion)). A strong NEGATIVE join (fat margins die first) is NOT
    the MARGIN-ORDERS-EROSION bar — the reverse read, reported in the
    table.
  * THE FLOOR: (a) the registered number ~0.9 = the lower edge of
    e214's wash-to-wash replication band (0.94-1.00); (b) computed at
    runtime on the y-side's own vectors: xrho_erosion(b,s) =
    Spearman(erosion_i(w1,s), erosion_i(w2,s)) per battery per shared
    state, from the committed records.
  * ADJUDICATION CELLS: (battery in {fact, ctrl, tmpl} — n >= 10;
    wash in {w1, w2}; state in {50, 80}) with e214's COMMITTED
    battery-level decline >= 0.40 (the e214 echo cut — the regime
    where per-probe erosion is substantial; the stricter 0.50 cut
    leaves the n=3 near battery plus ONE tmpl cell, powerless at a
    0.9 floor — DISCLOSED here before compute). near (n=3, Spearman
    quantized to half-points) = flagged coarse co-report everywhere,
    never adjudicates. +2 (wash-1-only) never adjudicates.
  * MARGIN-ORDERS-EROSION := adjudication cells exist AND EVERY cell
    join_rho >= 0.90 (fires only at the floor).
  * ORDER-OUTRUNS-FRAGILITY := cells exist AND EVERY cell join_rho <
    0.60 AND the erosion order still replicates there (xrho_erosion >=
    0.90 at every adjudication cell's battery+state).
  * PARTIAL := otherwise. Table verbatim (all four batteries, all
    states, both washes), no narrative inflation.
  * ORDER: MARGIN-ORDERS-EROSION -> ORDER-OUTRUNS-FRAGILITY ->
    PARTIAL, gated on G_STATES/G_BATT/G_CORPUS/G_ENV.
  * CO-REPORTS (never adjudicated): the full join table incl. near and
    +2; Kendall tau-b beside every rho; the absolute-drop erosion
    echo; the margin-vs-baseline-p coupling Spearman(margin(0), p(0))
    per battery (the honesty reflex: is the margin just p?); the
    margin relation-block texture; the t=0 landscape against T204's
    ~0.05 sigma flip threshold.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_STATES — the two state archives inventoried (sizes/mtimes;
    wash-1 cross-checked against e182c2's committed inventory) and
    RE-VERIFY: every battery's p re-probed on every LOADED state and
    compared per-probe to e214's committed records (its journal for
    t0, its metrics curves for w1/w2 states) within 0.005 (expected
    ~1e-6: same fp32 weights, same CPU fp32 probe path). The margin
    rides the same forward, so this certifies the margin states.
  * G_BATT — the four batteries rebuilt VERBATIM (module import)
    reproduce e214's journal t0 record: same kept sets, per-probe
    |dp| <= 0.010.
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (feeds the batteries' contamination scans;
    INHERITED FROZEN, never re-filtered).
  * G_ENV — CPU-only (zero GPU calls), torch threads <= 4, load
    checks before launch and between states, one state = one eval
    burst.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — n=2 washes is
texture, not law; wash 1 is a CPU fp32 replay of e182's GPU original,
wash 2 is GPU fp32 (the archive's device asymmetry, inherited,
disclosed); the margin landscape is ONE organism's (the shared
pristine 124M); the probe sets are small and frozen and the 2-shot
exemplar prefix is part of the instrument; Spearman on near (n=3) is
quantized to half-points; the join is rank-level on 12-20 probes per
battery — its floor-claim (rho >= 0.9) is a strong-association claim,
not an identity; margins deterministic in-session means n=1 evals
suffice (T204) but the arithmetic floor (~3e-7 on cosines; batch-shape
re-rounding ~85% of bits) sits far below every margin here — the
provenance records the exact path; nothing here is guaranteed.

COMPUTE ENVELOPE: CPU-only (desk+eval; two other agents live —
load-polite); threads 4; load-check before launch and between states;
one state = one eval burst; ~54 probes x 8 state reads + screening
forwards + 7 checkpoint loads; minutes. Progressive PARTIAL metrics +
resumable journal after every state (the standing disruption rule).

PROVENANCE: the organism, corpus filter+verify, fact/ctrl/nearrel
batteries and probe machinery are lab/e182c_forgetting_control.py
VERBATIM via module import (inheriting lab/e182_gpt2_wash.py); the
template battery is lab/e182c2_template.py VERBATIM via module import;
the two-wash archive, the census loop and the gate pattern are
lab/e214_rank_conservation.py's. The committed records (runs/e214/
metrics.json + journal.json) are read at RUNTIME (never transcribed).
Builds on: W031 (the registered prediction this cell discharges),
T204/x3 (the margin dial + the arithmetic floor), W028 + T187/e214
(the erosion order + the committed per-probe records), T185/e213 (the
two-wash archive), T183/e182c2 (the fresh draw + tmpl battery), T149/
e182c (the saved-state discipline), T123/e182 (the parent wash). NEW:
the per-probe argmax margin in sigma at every state of the archive;
the margin-rank-vs-erosion-order join with the y-side's own
replication floor; the margin landscape's shape/height split (order
conservation + level decline); the margin-vs-p coupling co-report.

Run:  cd lab && python e228_margin_landscape.py   (E228_SMOKE=1:
      t=0 + the w1 +10 pair only, own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

import common                                            # noqa: E402
from common import now_iso, run_dir, save_json            # noqa: E402

import e182c_forgetting_control as e1                    # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                             # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                 # the dispatch envelope (load-polite, <= 4)

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import textwrap                                          # noqa: E402

SMOKE = os.environ.get("E228_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e228_smoke" if SMOKE else "e228"

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
E214_METRICS = common.REPO / "runs" / "e214" / "metrics.json"
E214_JOURNAL = common.REPO / "runs" / "e214" / "journal.json"

# ---- registered bar constants (frozen) ----------------------------------------
FLOOR_BAR = 0.90       # "the wash-to-wash replication floor's lower edge, ~0.9"
BELOW_BAR = 0.60       # "rho clearly below the floor, e.g. < 0.6"
DECL_BAR = 0.40        # the e214 echo cut — the adjudication regime (disclosed)
COARSE_BATTERIES = ("near",)   # n=3, Spearman quantized — co-report only

# ---- registered verification tolerances (frozen) -------------------------------
TOL_PROBE_DP = 0.010   # per-probe t=0 dp vs e214's committed journal record
TOL_STATE_DP = 0.005   # per-probe dp on LOADED states vs e214's committed records

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "MARGIN-ORDERS-EROSION": 'margin rank at t=0 predicts the erosion '
            'order (rho >= the wash-to-wash replication floor\'s lower edge, '
            '~0.9) — the conserved order acquires a mechanism candidate: '
            'local decision fragility (fires only at the floor; a middling '
            'rho is NOT this bar)',
        "ORDER-OUTRUNS-FRAGILITY": 'margin rank does not predict the erosion '
            'order (rho clearly below the floor, e.g. < 0.6) while the '
            'erosion order still replicates — the order is deeper than local '
            'thinness; the STRANGER read',
        "PARTIAL": 'anything between — the table verbatim, both batteries, '
            'no narrative inflation',
    },
    "operationalizations": (
        "margin_i = (top1-top2 logit)/std(vocab logits) at the answer "
        "position, one forward per probe (the SAME forward as the p "
        "re-probe); t=0 is ONE read of the shared pristine organism (both "
        "wash lineages start there — e214's t0_shared); the y-side is "
        "e214's COMMITTED per-probe records read at runtime: primary "
        "erosion_i(w,s) = 1 - p_i(w,s)/p_i(0); join statistic SIGNED so "
        "POSITIVE = thin margins die first: join_rho = Spearman(margin(0), "
        "retention_i(w,s)) = -Spearman(margin(0), erosion); floor = e214's "
        "replication band lower edge 0.9 (and recomputed xrho_erosion on "
        "the y-side's own erosion vectors per battery per shared state); "
        "adjudication cells = (fact/ctrl/tmpl [n>=10], w1/w2, state in "
        "{50,80}) with e214's committed battery-level decline >= 0.40 "
        "(the e214 echo cut; the 0.50 cut leaves near-n3 + one tmpl cell — "
        "powerless at a 0.9 floor, disclosed); near = coarse co-report "
        "everywhere; +2 never adjudicates; MARGIN-ORDERS-EROSION := cells "
        "exist and ALL join_rho >= 0.90; ORDER-OUTRUNS-FRAGILITY := cells "
        "exist and ALL join_rho < 0.60 AND xrho_erosion >= 0.90 at every "
        "cell; PARTIAL := otherwise; order MARGIN-ORDERS-EROSION -> "
        "ORDER-OUTRUNS-FRAGILITY -> PARTIAL, gated on G_STATES/G_BATT/"
        "G_CORPUS/G_ENV; co-reports (never adjudicated): the full table "
        "incl. near and +2, Kendall tau-b, the absolute-drop erosion echo, "
        "the margin-vs-p coupling, the relation-block margin texture"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (commit "
                     "a58fa48, QUEUE row 'DISPATCHED ~13:25Z Oct-4') BEFORE "
                     "any compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY desk+eval on the committed checkpoints (no wash run here); "
    "CPU-only per the dispatch (two other agents live — threads 4, "
    "load-polite).",
    "The margin x-side is computed FRESH here (no committed margins exist); "
    "the erosion y-side is e214's COMMITTED records read at runtime — the "
    "join is my-margin-rank vs committed-erosion-rank, and every state my "
    "margins ride on is re-probe-certified against those same records "
    "(G_STATES, tol 0.005).",
    "The adjudication regime uses the e214 ECHO cut (decline >= 0.40), not "
    "the 0.50 bar cut: at 0.50 the regime is near (n=3, quantized) plus "
    "ONE tmpl cell — no power at a 0.9 floor. Disclosed in the "
    "operationalizations BEFORE compute; the 0.50-cut table is co-reported "
    "verbatim inside the full join table.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: t=0 + the w1 +10 state only, own smoke dir, nothing "
    "adjudicated or verified.",
]


# ------------------------------------------------------------------ helpers

def cpu_load_check(tag: str) -> dict:
    """The dispatch envelope's load check (CPU-only cell; nothing GPU)."""
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


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(common.REPO),
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unavailable"


# ------------------------------------------------- THE MARGIN PASS (the x-side)

@torch.no_grad()
def margin_pass(net, battery: list[dict]) -> dict:
    """One forward per probe: the e182/e182c/e214 p/rank path VERBATIM
    (logits[0, -1] -> softmax -> p[ans_id]; rank = (lg > lg[ans_id]).sum())
    plus THIS cell's object on the same logits: margin_sigma =
    (top1 - top2)/std(vocab logits) — the argmax decision margin in
    sigma units vs arithmetic noise (T204's dial)."""
    net.eval()
    rows = []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        pv = float(p[pr["ans_id"]])
        rank = int((lg > lg[pr["ans_id"]]).sum().item())
        top2v = torch.topk(lg, 2)
        top1_id, top2_id = int(top2v.indices[0]), int(top2v.indices[1])
        sigma = float(lg.std())                # torch default: unbiased
        margin_raw = float(top2v.values[0] - top2v.values[1])
        rows.append({
            "fact": pr["fact"], "relation": pr["relation"],
            "p": pv, "rank": rank, "top1": bool(rank == 0),
            "top5": bool(rank < 5),
            "margin_sigma": margin_raw / sigma if sigma > 0 else None,
            "margin_raw": margin_raw, "sigma": sigma,
            "top1_id": top1_id, "top2_id": top2_id,
            "argmax_is_answer": bool(top1_id == pr["ans_id"]),
        })
    ms = [r["margin_sigma"] for r in rows]
    ps = [r["p"] for r in rows]
    return {"probes": rows,
            "mean_p": float(sum(ps) / len(ps)),
            "mean_margin_sigma": float(sum(ms) / len(ms)),
            "median_margin_sigma": float(sorted(ms)[len(ms) // 2])
            if len(ms) % 2 else float((sorted(ms)[len(ms) // 2 - 1]
                                       + sorted(ms)[len(ms) // 2]) / 2),
            "min_margin_sigma": float(min(ms)),
            "frac_argmax_answer": float(sum(r["argmax_is_answer"]
                                            for r in rows) / len(rows))}


def mvec(rec: dict, battery: str) -> tuple[list[str], list[float]]:
    """(probe names, margin_sigma values) in the battery's frozen order."""
    pr = rec[battery]["probes"]
    return list(pr.keys()), [pr[f]["margin_sigma"] for f in pr]


def pvec_mine(rec: dict, battery: str) -> list[float]:
    return [r["p"] for r in rec[battery]["probes"]]


# ------------------------------------------------------------------ plots

def make_join_plot(rd, join_rows, cells, adj, margins_t0):
    """THE FIGURE: (a-d) the margin-rank vs erosion-rank scatter per
    battery (the join, at the deepest shared state, both washes);
    (e) the join rho vs state curves against the floor; (f) the verdict."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    deep = 80 if any(r["state"] == 80 for r in join_rows) else 10

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))

    # (a-d) rank-vs-rank scatters per battery at the deepest state
    for ax, b in zip(axes[0, :3].tolist() + axes[1, :1].tolist(), BATTERIES):
        n = next(r["n"] for r in join_rows if r["battery"] == b)
        for wash, mk in (("w1", "o"), ("w2", "s")):
            row = next((r for r in join_rows
                        if r["battery"] == b and r["wash"] == wash
                        and r["state"] == deep), None)
            if row is None:
                continue
            mr = avg_ranks(row["margin_vec"])
            er = avg_ranks(row["erosion_vec"])
            ax.scatter(mr, er, marker=mk, s=64, color=cols[b],
                       alpha=0.9 if wash == "w1" else 0.55,
                       edgecolor="k", linewidth=0.5,
                       label=f"{wash} +{deep}: rho {row['join_rho']:+.3f}")
            for i, f in enumerate(row["probes"]):
                if b == "fact" or wash == "w1":
                    short = f.split("->")[0][:9]
                    ax.annotate(short, (mr[i], er[i]),
                                textcoords="offset points", xytext=(3, 2),
                                fontsize=5.2, color=cols[b], alpha=0.8)
        ax.set_title(f"{b} (n={n}) — margin rank vs erosion rank "
                     f"[floor xrho_e {next((r['floor_xrho_erosion'] for r in join_rows if r['battery'] == b and r['wash'] == 'w1' and r['state'] == deep), float('nan')):.3f}]",
                     fontsize=9.5)
        ax.set_xlabel("t=0 margin rank (thin -> fat)")
        ax.set_ylabel(f"erosion rank at +{deep} (held -> eroded)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    # (e) join rho vs state, both washes, per battery
    ax = axes[1, 1]
    ax.axhspan(FLOOR_BAR, 1.02, color="gold", alpha=0.18)
    ax.axhline(FLOOR_BAR, color="k", ls="--", lw=1.2,
               label="floor lower edge 0.9")
    ax.axhline(BELOW_BAR, color="darkred", ls=":", lw=1.2,
               label="clearly-below 0.6")
    ax.axhline(0.0, color="gray", lw=0.8)
    for b in BATTERIES:
        for wash in ("w1", "w2"):
            rows = sorted((r for r in join_rows
                           if r["battery"] == b and r["wash"] == wash),
                          key=lambda r: r["state"])
            if not rows:
                continue
            ax.plot([r["state"] for r in rows], [r["join_rho"] for r in rows],
                    "o-" if wash == "w1" else "s--", ms=6, lw=1.7,
                    color=cols[b], alpha=0.9 if wash == "w1" else 0.6,
                    label=f"{b} {wash}" if wash == "w1" else None)
    ax.set_xlabel("wash step (+2 = wash-1-only)")
    ax.set_ylabel("JOIN rho: Spearman(margin(0), retention)\n"
                  "[+ = thin margins die first]")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower left", ncol=2)
    ax.set_title("the join — does margin rank predict the erosion order?",
                 fontsize=9.5)

    # (f) verdict + adjudication table
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "ADJUDICATION CELLS (decl >= 0.40, n >= 10, states "
            "{50,80}):", fontsize=8.6, va="top", family="monospace",
            weight="bold")
    y -= 0.026
    ax.text(0.02, y, "  battery  wash state  decl   join_rho  tau_b   floor"
                     "  cell", fontsize=6.9, va="top", family="monospace")
    y -= 0.019
    for c in cells:
        ax.text(0.02, y, f"  {c['battery']:7s} {c['wash']:4s} +{c['state']:<4d}"
                         f" {c['decl']:5.3f}  {c['join_rho']:+6.3f} "
                         f"{c['join_tau'] if c['join_tau'] is not None else float('nan'):+6.3f}"
                         f"  {c['floor_xrho_erosion']:5.3f}"
                         f"  {'AT-FLOOR' if c['join_rho'] >= FLOOR_BAR else ('BELOW' if c['join_rho'] < BELOW_BAR else 'mid')}",
                fontsize=6.9, va="top", family="monospace")
        y -= 0.0188
    y -= 0.012
    ax.text(0.02, y, f"E228 VERDICT: {adj['bars']['verdict']}", fontsize=10.2,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.034
    for wd in textwrap.wrap(adj["bars"]["clause"], width=88,
                            break_long_words=False)[:11]:
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.019
    y -= 0.006
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in adj["gates_summary"].items()), fontsize=6.9, va="top",
        family="monospace")

    fig.suptitle("E228 — W031's MARGIN LANDSCAPE vs THE EROSION ORDER: "
                 f"margin rank at t=0 vs e214's committed erosion records "
                 f"-> {adj['bars']['verdict']}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    png = rd / "margin_vs_erosion.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_conservation_plot(rd, mcons, margins_t0, rel_of):
    """THE SECOND READ'S PANEL: the margin landscape's own shape/height
    split. (a) margin ORDER conservation (within-wash rho + cross-wash);
    (b) margin LEVELS (battery means, both washes); (c) the shape/height
    dissociation scatter; (d) THE t=0 LANDSCAPE (sorted margins, by
    relation, vs T204's 0.05 sigma flip threshold)."""
    cols = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
            "tmpl": "darkorange"}
    relc = {"cap": "tab:red", "lang": "tab:purple", "cur": "tab:brown",
            "found": "tab:blue", "make": "tab:cyan", "near": "tab:green",
            "tmpl": "darkorange"}

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))

    # (a) margin-order conservation
    ax = axes[0, 0]
    steps_w1 = sorted(int(s) for s in mcons["w1"])
    steps_w2 = sorted(int(s) for s in mcons["w2"])
    for b in BATTERIES:
        ax.plot(steps_w1, [mcons["w1"][str(s)][b]["rho_m0"] for s in steps_w1],
                "o-", ms=6, lw=1.8, color=cols[b],
                label=f"{b} (both washes)" if b == "fact" else None)
        ax.plot(steps_w2, [mcons["w2"][str(s)][b]["rho_m0"] for s in steps_w2],
                "s--", ms=5, lw=1.5, color=cols[b], alpha=0.6)
        ax.plot(steps_w2, [mcons["xwash"][str(s)][b] for s in steps_w2],
                "^:", ms=6, lw=1.5, color=cols[b], alpha=0.85,
                label="cross-wash" if b == "fact" else None)
    ax.set_xlabel("wash step")
    ax.set_ylabel("margin-ORDER rho (o w1 / s w2 / ^ cross-wash)")
    ax.set_ylim(-0.2, 1.06)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(3a) the margin ORDER — conserved under the wash?",
                 fontsize=9.5)

    # (b) margin levels
    ax = axes[0, 1]
    for b in BATTERIES:
        ax.plot(steps_w1, [mcons["w1"][str(s)][b]["mean_margin_sigma"]
                           for s in steps_w1], "o-", ms=6, lw=1.8,
                color=cols[b], label=f"{b} w1")
        ax.plot(steps_w2, [mcons["w2"][str(s)][b]["mean_margin_sigma"]
                           for s in steps_w2], "s--", ms=5, lw=1.5,
                color=cols[b], alpha=0.6, label=f"{b} w2" if b == "fact"
                else None)
    ax.set_xlabel("wash step")
    ax.set_ylabel("margin LEVEL: battery mean margin (sigma)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="best", ncol=2)
    ax.set_title("(3b) the margin HEIGHT — do levels fall?", fontsize=9.5)

    # (c) shape/height dissociation
    ax = axes[1, 0]
    for b in BATTERIES:
        for wash in ("w1", "w2"):
            rows = sorted((mcons[wash][str(s)][b] for s in mcons[wash]),
                          key=lambda r: r["state"])
            xs = [1 - r["mean_margin_sigma"] / mcons["t0"][b]["mean_margin_sigma"]
                  for r in rows]
            ys = [r["rho_m0"] for r in rows]
            ax.plot(xs, ys, "o-" if wash == "w1" else "s--", ms=7, lw=1.7,
                    color=cols[b], alpha=0.9 if wash == "w1" else 0.6,
                    label=f"{b} {wash}" if wash == "w1" else None)
            for r, x, y in zip(rows, xs, ys):
                ax.annotate(f"+{r['state']}", (x, y),
                            textcoords="offset points", xytext=(4, -7),
                            fontsize=6.0, color=cols[b])
    ax.set_xlabel("margin LEVEL decline 1 - mean_margin(s)/mean_margin(0)")
    ax.set_ylabel("margin ORDER rho vs t=0")
    ax.set_ylim(-0.2, 1.06)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower left", ncol=2)
    ax.set_title("(3c) the shape/height split on the commitment spectrum",
                 fontsize=9.5)

    # (d) THE t=0 LANDSCAPE
    ax = axes[1, 1]
    y0 = 0.0
    for b in BATTERIES:
        pr = sorted(margins_t0[b]["probes"],
                    key=lambda r: r["margin_sigma"])
        for r in pr:
            rel = r["relation"]
            ax.plot([r["margin_sigma"]], [y0], "o", ms=5,
                    color=relc.get(rel, cols[b]))
            y0 -= 1.0
        y0 -= 3.0
    ax.axvline(0.05, color="darkred", ls="--", lw=1.3)
    ax.text(0.052, 0.5, "T204 ~0.05 sigma flip zone", fontsize=7,
            color="darkred", rotation=90, va="bottom", family="monospace")
    ax.set_xlabel("t=0 argmax margin (sigma)")
    ax.set_ylabel("probes (sorted per battery; color = relation)")
    ax.set_xlim(left=-0.02)
    ax.set_yticks([])
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("(1) THE t=0 MARGIN LANDSCAPE — the organism's thin spots",
                 fontsize=9.5)

    fig.suptitle("E228 — the second, quieter read: the margin landscape's "
                 "own shape-conservation under the two washes (W028's "
                 "shape/height split on the commitment spectrum)",
                 fontsize=10.8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = rd / "margin_conservation.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E228 — W031's MARGIN LANDSCAPE vs THE EROSION ORDER "
        f"(smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e228_margin_landscape",
        "phase": ("eval-only desk+eval on the committed two-wash 124M "
                  "archive (W031's registered discriminating observation)"),
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the conserved EROSION ORDER the shadow of the "
                     "MARGIN LANDSCAPE — do the thinnest-margin probes die "
                     "first under the wash? or does the erosion order "
                     "outrun local decision fragility?"),
        "builds_on": ["W031 (the registered prediction this cell discharges)",
                      "T204 / x3 (the margin dial; the arithmetic floor; "
                      "the ~0.05 sigma flip threshold)",
                      "W028 + T187 / e214 (the erosion order; the committed "
                      "per-probe records = the y-side)",
                      "T185 / e213 (the two-wash archive)",
                      "T183 / e182c2 (the fresh draw + the tmpl battery)",
                      "T149 / e182c (the saved-state discipline + batteries)",
                      "T123 / e182 (the parent wash)"],
        "whats_new": ["the per-probe argmax margin in sigma at every state "
                      "of the archive (the commitment spectrum instrument)",
                      "the margin-rank-vs-erosion-order join with the "
                      "y-side's own replication floor (xrho_erosion)",
                      "the margin landscape's shape/height split (order "
                      "conservation + level decline)",
                      "the margin-vs-p coupling co-report (is the margin "
                      "just p?)",
                      "the e208 distinction drawn in the docstring (argmax "
                      "margin vs arithmetic noise; NOT the fact-edge "
                      "object)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    for p in (E214_METRICS, E214_JOURNAL, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e214m = json.loads(E214_METRICS.read_text(encoding="utf-8"))
    e214j = json.loads(E214_JOURNAL.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    y_t0 = [r for r in e214j["states"] if r["wash"] == "t0"][0]
    y_st = {(r["wash"], r["step"]): r for r in e214j["states"]
            if r["wash"] in ("w1", "w2")}
    y_cur = e214m["curves"]                     # committed per-probe p's
    y_cells_decl = {(r["battery"], r["wash"], r["state"]): r["decl"]
                    for r in e214m["dissociation_rows"]}
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    log(f"y-side loaded: e214 curves {sorted(y_cur['w1'])}|w1 "
        f"{sorted(y_cur['w2'])}|w2; journal states "
        f"{sorted(k[0] + str(k[1]) for k in y_st)}")

    # ------------------------------------------ P1 G_STATES part A: inventory
    def inv(p: Path):
        return {"path": str(p), "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                    p.stat().st_mtime)) if p.exists() else None}
    w1_inv = {str(s): inv(p) for s, p in W1_ARCH.items()}
    w2_inv = {str(s): inv(p) for s, p in W2_ARCH.items()}
    c2_inv = json.loads((common.REPO / "runs" / "e182c2" / "metrics.json")
                        .read_text(encoding="utf-8")) \
        .get("inventory", {}).get("files", {})
    w1_match = {str(s): (bool(c2_inv.get(str(s)))
                         and c2_inv[str(s)]["size_bytes"]
                         == w1_inv[str(s)]["size_bytes"]
                         and c2_inv[str(s)]["mtime"]
                         == w1_inv[str(s)]["mtime"])
                for s in W1_ARCH}
    all_exist = all(v["exists"] for v in
                    list(w1_inv.values()) + list(w2_inv.values()))
    G_STATES_A = {
        "wash1_files": w1_inv, "wash2_files": w2_inv,
        "wash1_crosscheck_vs_e182c2_inventory": w1_match,
        "note": "wash 1 = e182c_s* (seed 18202, CPU fp32 replay, incl. the "
                "+2 wash-1-only state); wash 2 = e182c2_fresh_s* (seed "
                "20261002, GPU fp32); sizes/mtimes must match e182c2's "
                "committed inventory for the wash-1 set",
    }
    log(f"G_STATES inventory: wash1 {sorted(w1_inv)} wash2 {sorted(w2_inv)} "
        f"all on disk = {all_exist}; wash-1 crosscheck = {w1_match}")
    metrics["inventory"] = G_STATES_A
    write_metrics("PARTIAL: records read; inventory taken")

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
    log("batteries rebuilt VERBATIM: " + " | ".join(
        f"{b} n={len(bats[b])}" for b in BATTERIES))

    # --------------------------------- P5 the margin pass (t=0 + every state)
    mjour = []
    if jp.exists():
        try:
            mjour = json.loads(
                jp.read_text(encoding="utf-8"))["states"]
            log(f"journal: {len(mjour)} margin records restored")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); recomputing")
            mjour = []
    mdone = {(r["wash"], r["step"]) for r in mjour}

    def margin_all(evl):
        return {b: margin_pass(evl, bats[b]) for b in BATTERIES}

    # t=0 — ONE read of the shared pristine organism (both lineages start
    # here; e214's t0_shared convention, disclosed)
    if ("t0", 0) not in mdone:
        load_checks.append(cpu_load_check("t0"))
        cur = margin_all(net0)
        mjour.append({"wash": "t0", "step": 0, **cur})
        mdone.add(("t0", 0))
        mjour.sort(key=lambda r: (r["wash"], r["step"]))
        jp.write_text(json.dumps({"states": mjour}, indent=1),
                      encoding="utf-8")
    t0m = [r for r in mjour if r["wash"] == "t0"][0]

    # G_BATT: my t=0 p (margin pass, identical code path) vs e214's
    # committed journal t0 + kept-set identity
    gb = {}
    for b in BATTERIES:
        ref = y_t0[b]["probes"]
        mine = {r["fact"]: r for r in t0m[b]["probes"]}
        gb[b] = {
            "kept_set_equal": bool({r["fact"] for r in bats[b]}
                                   == set(ref)),
            "n": len(ref),
            "max_dp_vs_e214_t0": max(abs(mine[f]["p"] - v["p"])
                                     for f, v in ref.items()),
        }
    G_BATT = {
        **gb,
        "tol_per_probe_dp": TOL_PROBE_DP,
        "note": "the four batteries are the phase-1/phase-2 pools VERBATIM "
                "(module import); my t=0 margin-pass p must reproduce "
                "e214's committed journal t=0 (same pristine organism, "
                "same prompts, same CPU fp32 probe path) — certifying the "
                "forward the margins ride on",
    }
    G_BATT["pass"] = bool(
        all(G_BATT[b]["kept_set_equal"]
            and G_BATT[b]["max_dp_vs_e214_t0"] <= TOL_PROBE_DP
            for b in BATTERIES)) if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} " + " | ".join(
        f"{b}: dp {G_BATT[b]['max_dp_vs_e214_t0']:.2e}"
        for b in BATTERIES))

    # the state loop (wash x state; one state = one eval burst)
    ck_map = {"w1": W1_ARCH, "w2": W2_ARCH}
    state_list = ([("w1", s_) for s_ in SHARED_STATES + W1_ONLY_STATES]
                  + [("w2", s_) for s_ in SHARED_STATES])
    for wash, s_ in state_list:
        if (wash, s_) in mdone:
            continue
        load_checks.append(cpu_load_check(f"{wash}s{s_}"))
        f = ck_map[wash][s_]
        sd = torch.load(f, map_location=CPU,
                        weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        cur = margin_all(evl)
        del evl, sd
        mjour.append({"wash": wash, "step": s_, **cur})
        mdone.add((wash, s_))
        mjour.sort(key=lambda r: (r["wash"], r["step"]))
        jp.write_text(json.dumps({"states": mjour}, indent=1),
                      encoding="utf-8")
        log(f"  {wash.upper()} STATE +{s_}: " + " | ".join(
            f"{b} m {cur[b]['mean_margin_sigma']:.3f}s "
            f"(argmax-ans {cur[b]['frac_argmax_answer']:.2f})"
            for b in BATTERIES))
        write_metrics(f"PARTIAL: margins read through {wash} +{s_}")
    mst = {(r["wash"], r["step"]): r for r in mjour}

    # G_STATES part B: the p re-probe on every loaded state vs e214
    all_dps = []
    for (wash, s_), rec in sorted(mst.items()):
        if wash == "t0":
            continue
        for b in BATTERIES:
            ref = y_st[(wash, s_)][b]["probes"]
            mine = {r["fact"]: r for r in rec[b]["probes"]}
            all_dps.append(max(abs(mine[f]["p"] - v["p"])
                               for f, v in ref.items()))
    G_STATES = {**G_STATES_A, "all_exist": all_exist,
                "reprobe_max_dp": max(all_dps) if all_dps else None,
                "n_reprobe_checks": len(all_dps),
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_note": "every battery's p re-probed on every "
                                "LOADED state of BOTH archives (incl. the "
                                "wash-1-only +2) and compared per-probe to "
                                "e214's committed journal records — same "
                                "fp32 weights + same CPU fp32 probe path "
                                "-> expected ~0.0; the margins ride the "
                                "same forwards"}
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
        f"{max(all_dps) if all_dps else 0:.8f} (tol {TOL_STATE_DP})")

    # ------------------------------------------- P6 CELL 1: the t=0 landscape
    metrics["margins_t0"] = {
        b: {"mean_margin_sigma": t0m[b]["mean_margin_sigma"],
            "median_margin_sigma": t0m[b]["median_margin_sigma"],
            "min_margin_sigma": t0m[b]["min_margin_sigma"],
            "frac_argmax_answer": t0m[b]["frac_argmax_answer"],
            "probes": {r["fact"]: {"margin_sigma": r["margin_sigma"],
                                    "p": r["p"],
                                    "argmax_is_answer":
                                        r["argmax_is_answer"],
                                    "relation": r["relation"]}
                       for r in t0m[b]["probes"]}}
        for b in BATTERIES}
    # the margin-vs-p coupling (the honesty reflex: is the margin just p?)
    coupling = {}
    for b in BATTERIES:
        ms = [r["margin_sigma"] for r in t0m[b]["probes"]]
        ps = [r["p"] for r in t0m[b]["probes"]]
        coupling[b] = spearman(ms, ps)[0]
    metrics["margin_p_coupling_t0"] = {
        "spearman_margin_vs_p": coupling,
        "note": "the honesty reflex: the margin is a logits object "
                "correlated with p but not identical to it — this "
                "coupling prices how much the join re-asks e214's killed "
                "baseline-rank question; the join's x is the DECISION "
                "margin, not the recall probability",
    }
    log("t=0 landscape: " + " | ".join(
        f"{b} mean {t0m[b]['mean_margin_sigma']:.3f}s min "
        f"{t0m[b]['min_margin_sigma']:.3f}s (couple "
        f"{coupling[b]:+.3f})" for b in BATTERIES))
    write_metrics("PARTIAL: cell 1 (t=0 margins + coupling) done")

    # --------------------------------------------- P7 CELL 2: THE JOIN
    # the y-side committed records -> erosion + retention vectors
    join_rows = []
    for wash in ("w1", "w2"):
        for s_ in sorted(int(k) for k in y_cur[wash]):
            for b in BATTERIES:
                ref = y_cur[wash][str(s_)][b]["probes"]     # COMMITTED
                names = list(ref.keys())
                p0 = [y_t0[b]["probes"][f]["p"] for f in names]
                ps = [ref[f]["p"] for f in names]
                erosion = [1 - pi / p0i for pi, p0i in zip(ps, p0)]
                retention = [pi / p0i for pi, p0i in zip(ps, p0)]
                absdrop = [p0i - pi for p0i, pi in zip(p0, ps)]
                t0_names, t0_marg = mvec(t0m, b)
                assert t0_names == names, \
                    f"probe identity drift {b} {wash} +{s_}"
                rho, _t = spearman(t0_marg, retention)
                tau = kendall_tau_b(t0_marg, retention)
                rho_abs, _t2 = spearman(t0_marg,
                                        [-a for a in absdrop])
                join_rows.append({
                    "battery": b, "wash": wash, "state": s_,
                    "n": len(names),
                    "decl": y_cells_decl.get((b, wash, s_)),
                    "join_rho": rho, "join_tau": tau,
                    "join_rho_absdrop_echo": rho_abs,
                    "probes": names, "margin_vec": t0_marg,
                    "erosion_vec": erosion,
                })
    # the floor: the erosion order's own wash-to-wash replication, from
    # the committed records, per battery per shared state
    floor_x = {}
    for s_ in sorted(int(k) for k in y_cur.get("xwash", {})):
        for b in BATTERIES:
            e1v, e2v = [], []
            for wash, out in (("w1", e1v), ("w2", e2v)):
                ref = y_cur[wash][str(s_)][b]["probes"]
                names = list(ref.keys())
                p0 = [y_t0[b]["probes"][f]["p"] for f in names]
                ps = [ref[f]["p"] for f in names]
                out.extend(1 - pi / p0i for pi, p0i in zip(ps, p0))
            floor_x[(b, s_)] = spearman(e1v, e2v)[0]
    for r in join_rows:
        r["floor_xrho_erosion"] = floor_x.get((r["battery"], r["state"]))
        r["adjudicates"] = bool(
            r["battery"] not in COARSE_BATTERIES
            and r["state"] in (50, 80)
            and r["decl"] is not None and r["decl"] >= DECL_BAR)
        r["coarse_n3"] = bool(r["battery"] in COARSE_BATTERIES)
    metrics["join_rows"] = join_rows
    metrics["erosion_floor"] = {
        "e214_quoted_band": "the wash-to-wash replication of the order "
                            "itself (e214's xrho on the p-vectors): "
                            "0.94-1.00 — lower edge ~0.9 = the registered "
                            "floor",
        "xrho_erosion_computed": {f"{b}+{s}": v
                                  for (b, s), v in sorted(
                                      floor_x.items(),
                                      key=lambda kv: (kv[0][1], kv[0][0]))},
        "note": "the floor recomputed on the y-side's OWN erosion vectors "
                "(Spearman(erosion_w1, erosion_w2) per battery per shared "
                "state, committed records) — the replication the join is "
                "measured against at the exact cells",
    }
    write_metrics("PARTIAL: cell 2 (the join) computed — adjudication next")

    # ------------------------------------------ P8 CELL 3: the quieter read
    mcons = {"w1": {}, "w2": {}, "xwash": {}, "t0": {}}
    for b in BATTERIES:
        mcons["t0"][b] = {"mean_margin_sigma": t0m[b]["mean_margin_sigma"]}
    for wash in ("w1", "w2"):
        for (w_, s_) in sorted((k for k in mst if k[0] == wash),
                               key=lambda k: k[1]):
            entry = {}
            for b in BATTERIES:
                _, m0 = mvec(t0m, b)
                _, ms_ = mvec(mst[(w_, s_)], b)
                rho, _t = spearman(m0, ms_)
                entry[b] = {
                    "state": s_,
                    "rho_m0": rho,
                    "tau_m0": kendall_tau_b(m0, ms_),
                    "mean_margin_sigma":
                        mst[(w_, s_)][b]["mean_margin_sigma"],
                    "median_margin_sigma":
                        mst[(w_, s_)][b]["median_margin_sigma"],
                    "level_decline": 1 - mst[(w_, s_)][b]["mean_margin_sigma"]
                        / t0m[b]["mean_margin_sigma"],
                    "frac_argmax_answer":
                        mst[(w_, s_)][b]["frac_argmax_answer"],
                }
            mcons[wash][str(s_)] = entry
    for s_ in SHARED_STATES:
        entry = {}
        for b in BATTERIES:
            _, mw1 = mvec(mst[("w1", s_)], b)
            _, mw2 = mvec(mst[("w2", s_)], b)
            entry[b] = spearman(mw1, mw2)[0]
        mcons["xwash"][str(s_)] = entry
    metrics["margin_conservation"] = mcons
    write_metrics("PARTIAL: cell 3 (shape/height split) computed")

    # ------------------------------------------- P9 the adjudication (frozen)
    cells = [r for r in join_rows if r["adjudicates"]]
    gates_ok = bool(G_STATES["pass"] and G_BATT["pass"]
                    and G_CORPUS["pass"] and G_ENV["pass"])
    n_cells = len(cells)
    all_at_floor = bool(cells) and all(
        r["join_rho"] is not None and r["join_rho"] >= FLOOR_BAR
        for r in cells)
    all_below = bool(cells) and all(
        r["join_rho"] is not None and r["join_rho"] < BELOW_BAR
        for r in cells)
    erosion_replicates = bool(cells) and all(
        r["floor_xrho_erosion"] is not None
        and r["floor_xrho_erosion"] >= FLOOR_BAR for r in cells)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (table reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_STATES", G_STATES["pass"]),
                           ("G_BATT", G_BATT["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif n_cells == 0:
        verdict = "PARTIAL"
        clause = ("no adjudication cell exists (decline >= 0.40, n >= 10, "
                  "states {50,80}) — the table verbatim, no narrative "
                  "inflation")
    elif all_at_floor:
        verdict = "MARGIN-ORDERS-EROSION"
        clause = ("margin rank at t=0 predicts the erosion order at every "
                  f"adjudication cell (join rho >= {FLOOR_BAR} at the "
                  "wash-to-wash replication floor's lower edge; "
                  + "; ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"{c['join_rho']:+.3f} vs floor "
                              f"{c['floor_xrho_erosion']:.3f}"
                              for c in cells)
                  + ") — the conserved order acquires a mechanism "
                  "candidate: local decision fragility")
    elif all_below and erosion_replicates:
        verdict = "ORDER-OUTRUNS-FRAGILITY"
        clause = ("margin rank at t=0 does not predict the erosion order "
                  f"(every adjudication cell join rho < {BELOW_BAR}: "
                  + "; ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"{c['join_rho']:+.3f}" for c in cells)
                  + ") while the erosion order still replicates (its own "
                  "wash-to-wash xrho on the committed records: "
                  + ", ".join(f"{c['battery']}+{c['state']} "
                              f"{c['floor_xrho_erosion']:.3f}"
                              for c in cells)
                  + f", all >= {FLOOR_BAR}) — the order is deeper than "
                  "local thinness; the STRANGER read")
    else:
        verdict = "PARTIAL"
        clause = ("anything between — the table verbatim, no narrative "
                  "inflation (adjudication cells: "
                  + "; ".join(f"{c['battery']}/{c['wash']}+{c['state']} "
                              f"decl {c['decl']:.3f} join "
                              f"{c['join_rho']:+.3f} floor "
                              f"{c['floor_xrho_erosion']:.3f}"
                              for c in cells)
                  + (")" if cells else "; none)"))

    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {
            "MARGIN_ORDERS_EROSION": bool(all_at_floor and gates_ok),
            "ORDER_OUTRUNS_FRAGILITY": bool(all_below and erosion_replicates
                                            and gates_ok),
            "verdict": verdict, "clause": clause,
            "order": "MARGIN-ORDERS-EROSION -> ORDER-OUTRUNS-FRAGILITY -> "
                     "PARTIAL (gated on G_STATES/G_BATT/G_CORPUS/G_ENV)",
        },
        "bar_constants": {"FLOOR_BAR": FLOOR_BAR, "BELOW_BAR": BELOW_BAR,
                          "DECL_BAR (the e214 echo cut)": DECL_BAR},
        "cells": [{k: v for k, v in c.items()
                   if k not in ("margin_vec", "erosion_vec", "probes")}
                  for c in cells],
        "n_cells": n_cells,
        "n_at_floor": sum(1 for c in cells
                          if c["join_rho"] is not None
                          and c["join_rho"] >= FLOOR_BAR),
        "n_below": sum(1 for c in cells
                       if c["join_rho"] is not None
                       and c["join_rho"] < BELOW_BAR),
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E228 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}; cells {n_cells} "
        f"(at-floor {adj['n_at_floor']}, below {adj['n_below']})")

    # --------------------------------------------- P10 honesty + close
    metrics["honesty_reflex"] = {
        "e208_distinction": "e208's margin object was the FACT-EDGE over "
                            "the wash band on the tiny-net organisms (a "
                            "cross-organism class separator, scoped to "
                            "NATURAL-STEP organisms after e209/e210); THIS "
                            "cell's object is the NEXT-TOKEN ARGMAX MARGIN "
                            "in sigma units vs arithmetic noise, per probe, "
                            "on the 124M archive — a different ruler; do "
                            "not conflate; not a resurrection",
        "determinism": "margins are deterministic in-session (T204: same "
                       "code path, same device -> bit-exact), so n=1 "
                       "evals suffice; the arithmetic floor (~3e-7 on "
                       "cosines; batch-shape re-rounding flips only "
                       "~<=0.05-sigma decisions) sits far below every "
                       "margin measured here; provenance recorded",
        "margin_vs_p": f"the t=0 margin-vs-p Spearman coupling per battery "
                       f"(recorded in margin_p_coupling_t0) prices how "
                       "much of the join re-asks e214's killed "
                       "baseline-rank question — the margin is the "
                       "DECISION's commitment, not the recall probability, "
                       "but they are correlated objects on the same "
                       "forwards; the relation-block texture (the wash "
                       "sorts BY RELATION) is the known breaker of any "
                       "t=0-strength ordering",
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "the archive's device asymmetry, inherited, disclosed",
        "one_organism": "the margin landscape is ONE organism's (the "
                         "shared pristine 124M); nothing here speaks to "
                         "cross-organism margin structure",
        "t0_shared": "both wash lineages' t=0 is ONE read of the same "
                     "pristine organism (e214's G_BATT verified the "
                     "committed wash-specific t=0s identical to ~1e-6) — "
                     "the t=0 margin vector is shared, disclosed",
        "bar_regime": f"the adjudication regime is the e214 ECHO cut "
                      f"(decline >= {DECL_BAR}) — the 0.50 cut leaves "
                      "near (n=3) plus ONE tmpl cell, powerless at a 0.9 "
                      "floor; ctrl never enters the regime (its committed "
                      "decline tops out ~0.38); near is a flagged coarse "
                      "co-report everywhere (Spearman on n=3 is quantized "
                      "to half-points)",
        "join_is_rank_level": "the join is rank-level on 12-20 probes per "
                              "battery; a floor-claim (rho >= 0.9) is a "
                              "strong-association claim, not an identity; "
                              "the +2 wash-1-only state and all shallow "
                              "cells sit in the table but never "
                              "adjudicate",
        "guarantees_nothing": "nothing here is guaranteed — the openness "
                              "is the point (W031's two named branches and "
                              "the registered 'either' were written before "
                              "this cell ran; no retrofit)",
    }
    metrics["compute"] = {
        "envelope": "CPU-only desk+eval (dispatch): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, one state = one eval burst, "
                    "decisive; ~54 probes x 8 state reads + screening "
                    "forwards + 7 checkpoint loads; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(p) for p in
                          list(W1_ARCH.values()) + list(W2_ARCH.values())],
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "committed_y_side": {
            "e214_metrics": {"path": str(E214_METRICS),
                             "sha256_16": sha256_of(E214_METRICS),
                             "mtime": inv(E214_METRICS)["mtime"]},
            "e214_journal": {"path": str(E214_JOURNAL),
                             "sha256_16": sha256_of(E214_JOURNAL),
                             "mtime": inv(E214_JOURNAL)["mtime"]},
        },
        "batteries": "module import of lab/e182c_forgetting_control.py "
                     "(fact/ctrl/near) and lab/e182c2_template.py (tmpl), "
                     "VERBATIM — import, never retyped",
        "margin_definition": "(top1 logit - top2 logit)/std(vocab logits) "
                             "at the prompt's last position; torch std "
                             "unbiased; the same forward as the p re-probe",
        "versions": {"torch": torch.__version__,
                     "transformers": org_meta["transformers_version"],
                     "numpy": __import__("numpy").__version__,
                     "matplotlib": matplotlib.__version__},
        "threads": torch.get_num_threads(),
        "device": "cpu fp32",
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")

    if not SMOKE:
        rel_of = {b: {r["fact"]: r["relation"] for r in bats[b]}
                  for b in BATTERIES}
        png1 = make_join_plot(rd, join_rows, adj["cells"], adj, metrics[
            "margins_t0"])
        png2 = make_conservation_plot(rd, mcons, metrics["margins_t0"],
                                      rel_of)
        log(f"outputs: {rd / 'metrics.json'}, {png1}, {png2}")
    else:
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
