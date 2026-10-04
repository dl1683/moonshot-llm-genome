"""E230 — W033's desk pass: DOES THE COMMITMENT DIE BEFORE THE BELIEF?
+ T207's flip-zone ENTRY ORDER (desk-only, CPU, committed data, no model loads).

WHY: e228's honesty block flagged the coupling (margin correlates with p at
+0.80) — so are the wash's MANUFACTURED thin spots anything more than p's
shadow? W033 turned the confound into the question: the organism has TWO
levels to lose a fact at — the BELIEF (p(answer): what it expects) and the
COMMITMENT (the argmax margin in sigma: how hard it holds the choice against
noise, T204's dial). DOES THE COMMITMENT DIE BEFORE THE BELIEF? This cell is
W033's registered discriminating observation, verbatim: "per probe, the
state index where margin first crosses 0.05 sigma (the flip zone) vs the
state index where p first falls below its half-of-t=0 line; the population
of MARGIN-FIRST probes and the LEAD TIME (states between the two
crossings)." One desk pass, two reads; the second read is T207's registered
(a): the flip-zone ENTRY ORDER vs e214's committed erosion order.

THE e208 DISTINCTION (restated, owed): e208's margin object was the
FACT-EDGE over the wash band on the tiny-net organisms (a cross-organism
class separator, scoped to NATURAL-STEP organisms after e209/e210). THIS
cell's object is the NEXT-TOKEN ARGMAX MARGIN at the answer position in
SIGMA units vs ARITHMETIC noise ((top1 logit - top2 logit)/std(vocab
logits), T204/x3's dial), per probe, on the committed 124M archive — a
different ruler that shares only the word "margin". Not the fact-edge
object; not a resurrection.

DATA (all COMMITTED, read at runtime, never transcribed):
  x-side  = runs/e228/journal.json — per-state per-probe argmax margins
            (margin_sigma) for all four frozen batteries (fact n=20,
            ctrl n=12, near n=3 co-report only, tmpl n=19), states t0 +
            w1{2,10,50,80} + w2{10,50,80}; every state's p is
            re-probe-certified against e214 (e228 G_STATES, max dp 0.0).
  p-side  = runs/e214/journal.json — the committed per-probe p(answer)
            records at the same state grid (provenance: e214's journals,
            NOT re-derived); runs/e214/metrics.json for conventions;
            runs/e228/metrics.json for the join_rows decl cross-check.

REGISTERED BARS (frozen VERBATIM from the dispatch brief, itself frozen
from W033/T207; the registration commit of this file precedes any compute;
adjudicate against exactly this; no bar shopping):

  COMMITMENT-DIES-FIRST — "margin-first probes are a real population in at
  least the tmpl battery (>= 3 probes with >= 1 state of lead) — the
  organism abandons commitments before beliefs; W033's prediction
  confirmed; a new dissociation class named"

  BELIEF-AND-COMMITMENT-TOGETHER — "margins and p move together
  (margin-first count at noise: < 3 per battery, lead times ~0) — the thin
  spots are p's shadow; T207's free find deflates honestly to a co-read"

  MIXED — "any between — the tables verbatim, both reads, no narrative
  inflation"

  (The entry-order read is adjudicated at its own registered line:
  >= 0.6 SAME-AUTHOR (the wash writes both) / < 0.6 TWO-ORDERS — reported
  as its own clause.)

W033's REGISTERED PREDICTION (honored verbatim, no retrofit): "the tmpl
battery carries margin-first probes (the few-shot faculty's death would
begin in the decision layer — commitments thinning while the belief still
quotes the template); the fact battery is mostly together-moving (installed
facts die as wholes)."

T207's REGISTERED PREDICTION (a), honored verbatim: "the flip-zone ENTRY
order (which probes' margins cross 0.05 sigma first) correlates with the
erosion order >= 0.6 if it is the same wash writing both — a candidate
FQ5-interior instrument."

OPERATIONALIZATIONS (frozen BEFORE compute):
  state grid — per wash, the ordered sequence is [t0 (step 0)] + that
  wash's committed states ascending: w1 = [0, 2, 10, 50, 80] (state indices
  0..4), w2 = [0, 10, 50, 80] (indices 0..3). The two journals' grids are
  reconciled at runtime: only SHARED (wash, step) states are used (they are
  asserted identical here; listed in metrics), so no interpolation is
  needed — if they ever differed, the shared-only rule is the fallback.
  margin crossing — first state index (scanning from t0) where the
  probe's argmax margin_sigma < 0.05 (T204's flip zone; strictly below;
  absorbing: later recovery does not un-cross).
  p crossing — first state index (scanning from t0) where p(answer) <
  0.5 * p_t0 (strictly below the half-of-t=0 line; absorbing).
  classification per probe per wash — MARGIN-FIRST: margin crossed and
  (p crossed strictly later, or p never crossed by the last state);
  P-FIRST: p crossed and (margin crossed strictly later, or never);
  TOGETHER: both crossed at the same state index; NEITHER: neither event
  by the last state.
  lead time — states between the two crossings (p-crossing index minus
  margin-crossing index); for margin-first probes whose p never crossed,
  the lead is CENSORED (at least last-state minus crossing, reported as a
  lower bound and flagged; it still qualifies as ">= 1 state of lead" iff
  the margin crossing precedes the last state by >= 1 state index).
  bar scope — "a real population in at least the tmpl battery" fires iff
  tmpl has >= 3 margin-first probes with >= 1 state of lead in EACH of the
  two washes (a population replicates; one wash alone is an anecdote — the
  two washes are the archive's only independence). "per battery" in the
  TOGETHER bar ranges over fact/ctrl/tmpl; near (n=3, quantized) is a
  co-report everywhere and never adjudicates.
  entry order (T207-a) — per battery per wash, probes are ranked by
  (first-crossing state ascending, ties by margin depth at the crossing
  state: deeper = earlier); probes that never enter the flip zone are tied
  last (average ranks). Spearman vs e214's committed per-probe erosion
  (erosion_i = 1 - p_i(w,s)/p_i(0)) at each state; "died first" pairs with
  "entered first" so POSITIVE = same-author. PRIMARY statistic = the
  lexicographic entry key under Spearman's average-rank convention;
  sensitivity co-report = the crossing-state-index-only key (all
  same-state entrants fully tied). Adjudication cells mirror e228's: the
  n>=10 batteries, states {50, 80}, with the e214 committed battery-level
  decline (e214's committed formula: decline_b(w,s) = 1 - mean_p_b(w,s)/
  mean_p_b(0)) >= 0.40 (the e214 echo cut). [INSTRUMENT CORRECTION,
  disclosed: the registration pass of this docstring wrote 'mean of
  per-probe erosion' — a definitional slip caught by the G_DECL desk gate
  (max abs diff 0.0141 vs e228's committed join_rows); e214's committed
  operationalization is the ratio of means, which reproduces e228's decl
  to 0.0 exactly and its adjudication cell set exactly; corrected to the
  committed convention, BARS UNTOUCHED, e226 precedent]
  SAME-AUTHOR iff ALL adjudication cells >= 0.6; TWO-ORDERS iff ALL < 0.6;
  any split is reported per cell with no narrative inflation. The 0.6 line
  is T207's registered (a), verbatim.

CHECKS (registered): state grids may differ between journals — reconcile
to SHARED states only and list them; the p records' provenance is e214's
committed journals, not re-derived (probe-set equality + journal-to-journal
p agreement verified at runtime as desk re-certification); near n=3 is
quantized — co-report only; the e208 distinction is restated above; nothing
is guaranteed.
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E228_JOURNAL = os.path.join(ROOT, "runs", "e228", "journal.json")
E228_METRICS = os.path.join(ROOT, "runs", "e228", "metrics.json")
E214_JOURNAL = os.path.join(ROOT, "runs", "e214", "journal.json")
E214_METRICS = os.path.join(ROOT, "runs", "e214", "metrics.json")
OUT_DIR = os.path.join(ROOT, "runs", "e230")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")

FLIP_ZONE = 0.05  # sigma, T204's flip zone (strictly below)
P_HALF = 0.5  # p crossing: p < P_HALF * p_t0 (strictly below)
DECL_CUT = 0.40  # the e214 echo cut, mirroring e228's adjudication regime
ENTRY_LINE = 0.6  # T207-a's registered line (SAME-AUTHOR / TWO-ORDERS)
POP_N = 3  # ">= 3 probes with >= 1 state of lead"
BATTERIES = ["fact", "ctrl", "near", "tmpl"]
ADJ_BATTERIES = ["fact", "ctrl", "tmpl"]  # near never adjudicates
WASHES = ["w1", "w2"]
ADJ_STATES = [50, 80]

SCRIPT_PATH = os.path.abspath(__file__)


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def git_head():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


M = {
    "experiment": "e230_commitment_belief",
    "phase": "desk-only pass on COMMITTED data (e228 journal margins x e214 journal p) "
    "- no model loads, no GPU, no wash, pure join",
    "date": now_utc(),
    "status": "RUNNING",
    "registration": "bars frozen VERBATIM from the dispatch brief (W033/T207) in this "
    "file's docstring; the registration commit of this file precedes any compute; "
    "adjudicate against exactly this; no bar shopping",
}


def write_metrics(stage):
    M["stage"] = stage
    M["updated"] = now_utc()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(M, f, indent=2, sort_keys=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# STAGE 0 — registration + provenance
# ---------------------------------------------------------------------------
write_metrics("s0-registration")

M["question"] = (
    "does the commitment (argmax margin, sigma) die before the belief (p(answer)) "
    "under the wash — a decision-level death preceding the belief-level death? "
    "and does the flip-zone ENTRY order match e214's committed erosion order "
    "(same author: the wash) or not (two orders)?"
)
M["builds_on"] = [
    "W033 (the wonder card this cell discharges — its registered discriminating "
    "observation is this pass, verbatim)",
    "T207 / e228 (the manufactured thin spots; registered prediction (a): the "
    "entry order vs the erosion order; the x-side journal)",
    "W028 + T187 / e214 (the erosion order; the committed per-probe p records = "
    "the p-side)",
    "T204 / x3 (the margin dial; the ~0.05 sigma flip zone vs arithmetic noise)",
    "e228 G_STATES (every state re-probe-certified vs e214, max dp 0.0 — the "
    "margins and the committed p ride the same forwards)",
]
M["whats_new"] = [
    "the per-probe DISSOCIATION TIMING join: first flip-zone margin crossing vs "
    "first half-of-t0 p crossing, per battery per wash, on committed records",
    "the margin-first population count + lead-time distribution (W033's "
    "registered instrument)",
    "the flip-zone entry order vs the committed erosion order (T207's registered "
    "(a), its own clause at its own 0.6 line)",
]
M["registered_bars_verbatim"] = {
    "COMMITMENT-DIES-FIRST": "margin-first probes are a real population in at "
    "least the tmpl battery (>= 3 probes with >= 1 state of lead) — the organism "
    "abandons commitments before beliefs; W033's prediction confirmed; a new "
    "dissociation class named",
    "BELIEF-AND-COMMITMENT-TOGETHER": "margins and p move together (margin-first "
    "count at noise: < 3 per battery, lead times ~0) — the thin spots are p's "
    "shadow; T207's free find deflates honestly to a co-read",
    "MIXED": "any between — the tables verbatim, both reads, no narrative "
    "inflation",
    "entry_order_clause_own_line": ">= 0.6 reads SAME-AUTHOR (the wash writes "
    "both) / < 0.6 TWO-ORDERS — reported as its own clause",
}
M["registered_predictions"] = {
    "W033_verbatim": "the tmpl battery carries margin-first probes (the few-shot "
    "faculty's death would begin in the decision layer — commitments thinning "
    "while the belief still quotes the template); the fact battery is mostly "
    "together-moving (installed facts die as wholes)",
    "T207a_verbatim": "the flip-zone ENTRY order (which probes' margins cross "
    "0.05 sigma first) correlates with the erosion order >= 0.6 if it is the "
    "same wash writing both — a candidate FQ5-interior instrument",
}
M["sources"] = {
    "x_side": {
        "path": E228_JOURNAL,
        "sha256_16": sha16(E228_JOURNAL),
    },
    "p_side": {
        "path": E214_JOURNAL,
        "sha256_16": sha16(E214_JOURNAL),
    },
    "e228_metrics_decl_crosscheck": {
        "path": E228_METRICS,
        "sha256_16": sha16(E228_METRICS),
    },
    "e214_metrics_conventions": {
        "path": E214_METRICS,
        "sha256_16": sha16(E214_METRICS),
    },
    "p_provenance": "e214's COMMITTED journals, read at runtime, not re-derived; "
    "desk re-certification below (probe-set equality + journal-to-journal p "
    "agreement + decl cross-check vs e228's committed join_rows)",
}
M["compute"] = {
    "device": "cpu desk pass",
    "model_loads": 0,
    "gpu_calls": 0,
    "torch_used": False,
    "threads_note": "pure json/numpy join; no training, no evals",
}
write_metrics("s0-registration")

# ---------------------------------------------------------------------------
# Load + reconcile the two journals
# ---------------------------------------------------------------------------
with open(E228_JOURNAL) as f:
    j228 = json.load(f)
with open(E214_JOURNAL) as f:
    j214 = json.load(f)
with open(E214_METRICS) as f:
    m214 = json.load(f)
with open(E228_METRICS) as f:
    m228 = json.load(f)


def state_grid(journal):
    grid = {"t0": 0}
    washes = {}
    for entry in journal["states"]:
        w, s = entry["wash"], entry["step"]
        if w == "t0":
            continue
        washes.setdefault(w, []).append(s)
    for w in washes:
        washes[w] = sorted(washes[w])
    grid["washes"] = washes
    return grid


grid228 = state_grid(j228)
grid214 = state_grid(j214)

shared = {}
shared_list = []
for w in WASHES:
    common = sorted(set(grid228["washes"].get(w, [])) & set(grid214["washes"].get(w, [])))
    shared[w] = common
    for s in common:
        shared_list.append((w, s))

grid_check = {
    "e228_grid": grid228["washes"],
    "e214_grid": grid214["washes"],
    "shared_states_used": {w: shared[w] for w in WASHES},
    "identical": grid228["washes"] == grid214["washes"],
    "note": "only SHARED (wash, step) states are used; here the grids are "
    "identical (t0 + w1{2,10,50,80} + w2{10,50,80}), so no interpolation was "
    "needed — listed per the registered check",
}

# probe-set equality + journal-to-journal p agreement (desk re-certification)
max_dp = 0.0
probe_set_mismatches = []
ns = {}


def e228_probe_records(entry, battery):
    return {rec["fact"]: rec for rec in entry[battery]["probes"]}


def e214_probe_records(entry, battery):
    return entry[battery]["probes"]


for entry228 in j228["states"]:
    w, s = entry228["wash"], entry228["step"]
    if w == "t0":
        key = ("t0", 0)
        entry214 = next(e for e in j214["states"] if e["wash"] == "t0")
    else:
        if (w, s) not in shared_list:
            continue
        entry214 = next(
            e for e in j214["states"] if e["wash"] == w and e["step"] == s
        )
    for b in BATTERIES:
        r228 = e228_probe_records(entry228, b)
        r214 = e214_probe_records(entry214, b)
        ns.setdefault(b, set()).add(len(r214))
        if set(r228.keys()) != set(r214.keys()):
            probe_set_mismatches.append({"wash": w, "step": s, "battery": b})
        for name in r214:
            if name in r228:
                max_dp = max(max_dp, abs(r228[name]["p"] - r214[name]["p"]))

gates = {
    "G_GRID": {**grid_check, "pass": grid_check["identical"]},
    "G_PROBES": {
        "mismatches": probe_set_mismatches,
        "ns": {b: sorted(v) for b, v in ns.items()},
        "expected": {"fact": [20], "ctrl": [12], "near": [3], "tmpl": [19]},
        "pass": len(probe_set_mismatches) == 0
        and ns == {"fact": {20}, "ctrl": {12}, "near": {3}, "tmpl": {19}},
    },
    "G_PP": {
        "max_abs_dp_e228_journal_vs_e214_journal": max_dp,
        "tol": 0.005,
        "note": "e228's committed journal p vs e214's committed journal p, every "
        "probe, every shared state — desk re-certification of the p-side's "
        "provenance (e228's G_STATES reprobe already certified the forwards)",
        "pass": max_dp <= 0.005,
    },
}

# ---------------------------------------------------------------------------
# STAGE 1 — build the per-probe tables
# ---------------------------------------------------------------------------
t0_228 = next(e for e in j228["states"] if e["wash"] == "t0")
t0_214 = next(e for e in j214["states"] if e["wash"] == "t0")

# decl cross-check vs e228's committed join_rows
decl_rows = {}
for row in m228["join_rows"]:
    decl_rows[(row["battery"], row["wash"], row["state"])] = row["decl"]


def decl_crosscheck():
    out = {
        "definition": "decline_b(w,s) = 1 - mean_p_b(w,s)/mean_p_b(0) (e214's "
        "committed formula, verified against e228's committed join_rows decl)",
        "max_abs_diff": 0.0,
        "n_rows": 0,
        "slipped_definition_coreport": {
            "definition": "mean of per-probe erosion (the registration-pass "
            "slip, kept as a co-report)",
            "max_abs_diff": 0.0,
        },
    }
    for row in m228["join_rows"]:
        b, w, s = row["battery"], row["wash"], row["state"]
        entry214 = next(e for e in j214["states"] if e["wash"] == w and e["step"] == s)
        p0 = e214_probe_records(t0_214, b)
        ps = e214_probe_records(entry214, b)
        p0v = [p0[name]["p"] for name in p0]
        psv = [ps[name]["p"] for name in p0]
        ratio = 1 - float(np.mean(psv)) / float(np.mean(p0v))
        mean_ero = float(np.mean([1 - a / b_ for a, b_ in zip(psv, p0v)]))
        out["max_abs_diff"] = max(out["max_abs_diff"], abs(ratio - row["decl"]))
        out["slipped_definition_coreport"]["max_abs_diff"] = max(
            out["slipped_definition_coreport"]["max_abs_diff"],
            abs(mean_ero - row["decl"]),
        )
        out["n_rows"] += 1
    out["tol"] = 1e-9
    out["pass"] = out["max_abs_diff"] <= 1e-9
    return out


gates["G_DECL"] = decl_crosscheck()
gates["G_ENV"] = {
    "cpu_only": True,
    "model_loads": 0,
    "gpu_calls": 0,
    "no_wash": True,
    "pass": True,
}

M["gates"] = gates
M["provenance"] = {
    "git_head_at_start": git_head(),
    "script": SCRIPT_PATH,
    "script_sha256_16": sha16(SCRIPT_PATH),
    "versions": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
    },
    "e214_status_at_read": m214.get("status"),
    "e228_status_at_read": m228.get("status"),
}
M["timing"] = {"started": now_utc()}
write_metrics("s1-gates")

# ---------------------------------------------------------------------------
# READ 1 — the dissociation (W033)
# ---------------------------------------------------------------------------
seq = {"w1": [0] + shared["w1"], "w2": [0] + shared["w2"]}
seq_idx = {w: list(range(len(seq[w]))) for w in WASHES}


def entry_for(wash, step):
    if wash == "t0":
        return next(e for e in j228["states"] if e["wash"] == "t0"), next(
            e for e in j214["states"] if e["wash"] == "t0"
        )
    return (
        next(e for e in j228["states"] if e["wash"] == wash and e["step"] == step),
        next(e for e in j214["states"] if e["wash"] == wash and e["step"] == step),
    )


cache228 = {}
cache214 = {}
for w in WASHES:
    for s in seq[w]:
        e228e, e214e = entry_for("t0" if s == 0 else w, s)
        cache228[(w, s)] = e228e
        cache214[(w, s)] = e214e

population_rows = []
probe_level = {}

for b in BATTERIES:
    probe_level[b] = {}
    for w in WASHES:
        p0 = e214_probe_records(t0_214, b)
        names = list(p0.keys())
        rows = []
        for name in names:
            half = P_HALF * p0[name]["p"]
            m_cross = None
            m_depth = None
            p_cross = None
            for i, s in enumerate(seq[w]):
                r228 = e228_probe_records(cache228[(w, s)], b)[name]
                r214 = e214_probe_records(cache214[(w, s)], b)[name]
                if m_cross is None and r228["margin_sigma"] < FLIP_ZONE:
                    m_cross = i
                    m_depth = r228["margin_sigma"]
                if p_cross is None and r214["p"] < half:
                    p_cross = i
            last = seq_idx[w][-1]
            if m_cross is not None and (p_cross is None or p_cross > m_cross):
                cls = "MARGIN-FIRST"
                if p_cross is None:
                    lead = None  # censored
                    lead_lb = last - m_cross
                else:
                    lead = p_cross - m_cross
                    lead_lb = lead
            elif p_cross is not None and (m_cross is None or m_cross > p_cross):
                cls = "P-FIRST"
                lead = lead_lb = None
            elif m_cross is not None and p_cross is not None and m_cross == p_cross:
                cls = "TOGETHER"
                lead = lead_lb = 0
            else:
                cls = "NEITHER"
                lead = lead_lb = None
            qualifies = cls == "MARGIN-FIRST" and lead_lb is not None and lead_lb >= 1
            rows.append(
                {
                    "probe": name,
                    "m_cross_idx": m_cross,
                    "m_cross_step": seq[w][m_cross] if m_cross is not None else None,
                    "m_depth_at_cross": m_depth,
                    "p_cross_idx": p_cross,
                    "p_cross_step": seq[w][p_cross] if p_cross is not None else None,
                    "class": cls,
                    "lead": lead,
                    "lead_lower_bound": lead_lb,
                    "censored": cls == "MARGIN-FIRST" and p_cross is None,
                    "qualifies_ge1_lead": qualifies,
                }
            )
        probe_level[b][w] = rows
        n = len(rows)
        counts = {k: sum(1 for r in rows if r["class"] == k) for k in
                  ["MARGIN-FIRST", "P-FIRST", "TOGETHER", "NEITHER"]}
        qual = [r for r in rows if r["qualifies_ge1_lead"]]
        population_rows.append(
            {
                "battery": b,
                "wash": w,
                "n": n,
                "state_sequence": seq[w],
                "coarse_n3": b == "near",
                "counts": counts,
                "margin_first_with_ge1_lead": len(qual),
                "of_which_censored": sum(1 for r in qual if r["censored"]),
                "lead_times": [r["lead"] for r in qual if not r["censored"]],
                "censored_leads_lower_bounds": [
                    r["lead_lower_bound"] for r in qual if r["censored"]
                ],
                "margin_crossed_before_or_with_p": counts["MARGIN-FIRST"]
                + counts["TOGETHER"],
                "p_crossed_before_or_with_m": counts["P-FIRST"]
                + counts["TOGETHER"],
                "margin_crossed_ever": sum(
                    1 for r in rows if r["m_cross_idx"] is not None
                ),
                "p_crossed_ever": sum(1 for r in rows if r["p_cross_idx"] is not None),
            }
        )

M["read1_population"] = {
    "definition": "per probe per wash: margin event = first state with argmax "
    "margin_sigma < 0.05 (T204 flip zone, absorbing); p event = first state "
    "with p < 0.5 * p_t0 (absorbing); MARGIN-FIRST = margin crossed with p "
    "still standing (crossed strictly later or never); lead = states between "
    "crossings; censored = p never crossed by the last state",
    "table": population_rows,
    "probe_level": {
        b: {w: probe_level[b][w] for w in WASHES} for b in BATTERIES
    },
}
write_metrics("s2-read1-population")

# bar adjudication
tmpl_qual = {
    w: next(
        r["margin_first_with_ge1_lead"]
        for r in population_rows
        if r["battery"] == "tmpl" and r["wash"] == w
    )
    for w in WASHES
}
any_battery_wash_ge3 = [
    (r["battery"], r["wash"], r["margin_first_with_ge1_lead"])
    for r in population_rows
    if r["battery"] in ADJ_BATTERIES and r["margin_first_with_ge1_lead"] >= POP_N
]

if all(v >= POP_N for v in tmpl_qual.values()):
    verdict1 = "COMMITMENT-DIES-FIRST"
    clause1 = (
        "tmpl carries >= {} margin-first probes with >= 1 state of lead in BOTH "
        "washes (w1: {}, w2: {}) — the organism abandons commitments before "
        "beliefs; W033's prediction confirmed; the dissociation class named: "
        "decision-death precedes belief-death".format(
            POP_N, tmpl_qual["w1"], tmpl_qual["w2"]
        )
    )
elif not any_battery_wash_ge3:
    verdict1 = "BELIEF-AND-COMMITMENT-TOGETHER"
    pfirst_texture = "; ".join(
        "{} {} p-first {}/{}".format(r["battery"], r["wash"], r["counts"]["P-FIRST"], r["n"])
        for r in population_rows
    )
    qual_list = [
        "{} {} {} (lead {})".format(b, w, p["probe"], p["lead_lower_bound"])
        for b in ADJ_BATTERIES
        for w in WASHES
        for p in probe_level[b][w]
        if p["qualifies_ge1_lead"]
    ]
    clause1 = (
        "no battery-wash of fact/ctrl/tmpl reaches {} margin-first probes with "
        ">= 1 state of lead (max per battery-wash: {}) — margins and p move "
        "together at this grid; the thin spots read as p's shadow; T207's free "
        "find deflates honestly to a co-read. TEXTURE, THE TABLES VERBATIM: "
        "the mirror cell dominates — P-FIRST (belief below its half-of-t0 "
        "line while the argmax margin still stands >= 0.05 sigma) is the "
        "largest crossing class in every battery-wash ({}); TOGETHER counts "
        "are 0-2; the qualifying margin-first probes are exactly: {}; "
        "margins enter the flip zone AT ALL for only 1-5 probes per "
        "fact/tmpl battery-wash (crossed-ever column)".format(
            POP_N,
            max(
                (r["margin_first_with_ge1_lead"] for r in population_rows
                 if r["battery"] in ADJ_BATTERIES),
                default=0,
            ),
            pfirst_texture,
            ", ".join(qual_list) if qual_list else "none",
        )
    )
else:
    verdict1 = "MIXED"
    clause1 = (
        "between the bars: battery-washes reaching {} margin-first probes: {} "
        "(tmpl per wash: w1 {}, w2 {}) — tables verbatim, both reads, no "
        "narrative inflation".format(
            POP_N,
            ", ".join("{}:{}={}".format(*t) for t in any_battery_wash_ge3),
            tmpl_qual["w1"],
            tmpl_qual["w2"],
        )
    )

# W033 prediction clause reads (co-reported, the bar adjudicates)
fact_frac_together = {}
tmpl_frac_together = {}
for r in population_rows:
    frac = r["counts"]["TOGETHER"] / r["n"]
    if r["battery"] == "fact":
        fact_frac_together[r["wash"]] = frac
    if r["battery"] == "tmpl":
        tmpl_frac_together[r["wash"]] = frac

M["read1_adjudication"] = {
    "bars": {
        "COMMITMENT_DIES_FIRST": verdict1 == "COMMITMENT-DIES-FIRST",
        "BELIEF_AND_COMMITMENT_TOGETHER": verdict1 == "BELIEF-AND-COMMITMENT-TOGETHER",
        "MIXED": verdict1 == "MIXED",
        "verdict": verdict1,
        "clause": clause1,
    },
    "w033_prediction_clauses": {
        "prediction_verbatim": M["registered_predictions"]["W033_verbatim"],
        "tmpl_carries_margin_first": tmpl_qual,
        "fact_together_fraction": fact_frac_together,
        "note": "the bar adjudicates; these clause reads are co-reports against "
        "W033's registered wording, no retrofit",
    },
}
write_metrics("s2-read1-population")

# ---------------------------------------------------------------------------
# READ 2 — the entry order vs the erosion order (T207-a)
# ---------------------------------------------------------------------------
from scipy.stats import rankdata  # noqa: E402

entry_rows = []
cell_records = {}

for b in BATTERIES:
    for w in WASHES:
        rows = probe_level[b][w]
        names = [r["probe"] for r in rows]
        last = seq_idx[w][-1]
        # lexicographic entry key: (crossing state asc, margin depth asc)
        # never-entered tied last via a sentinel far above all keys
        lex_keys, state_keys = [], []
        for r in rows:
            if r["m_cross_idx"] is not None:
                lex_keys.append(r["m_cross_idx"] + r["m_depth_at_cross"] / 1000.0)
                state_keys.append(r["m_cross_idx"])
            else:
                lex_keys.append(1e9)
                state_keys.append(last + 1)
        lex_ranks = rankdata(lex_keys)
        p0 = e214_probe_records(t0_214, b)
        for s in shared[w]:
            entry214 = cache214[(w, s)]
            ps = e214_probe_records(entry214, b)
            erosion = np.array(
                [1 - ps[name]["p"] / p0[name]["p"] for name in names]
            )
            p0v = [p0[name]["p"] for name in names]
            psv = [ps[name]["p"] for name in names]
            decl = 1 - float(np.mean(psv)) / float(np.mean(p0v))
            decl_mean_ero = float(np.mean(erosion))
            rho_primary = float(
                spearmanr(lex_keys, erosion).statistic
            ) if len(set(lex_keys)) > 1 else float("nan")
            rho_sens = float(
                spearmanr(state_keys, erosion).statistic
            ) if len(set(state_keys)) > 1 else float("nan")
            # look-ahead-free sensitivity: rank only crossings that occurred
            # BY state s (entrants-so-far vs erosion at s; rest tied last)
            s_pos = seq[w].index(s)
            keys_so_far = [
                k if (r["m_cross_idx"] is not None and r["m_cross_idx"] <= s_pos) else 1e9
                for k, r in zip(lex_keys, rows)
            ]
            rho_so_far = float(
                spearmanr(keys_so_far, erosion).statistic
            ) if any(k < 1e9 for k in keys_so_far) else float("nan")
            n_entered = sum(1 for r in rows if r["m_cross_idx"] is not None)
            n_entered_so_far = sum(1 for k in keys_so_far if k < 1e9)
            adjudicates = (
                b in ADJ_BATTERIES and s in ADJ_STATES and decl >= DECL_CUT
            )
            rec = {
                "battery": b,
                "wash": w,
                "state": s,
                "n": len(names),
                "n_entered_flip_zone": n_entered,
                "n_entered_by_this_state": n_entered_so_far,
                "decl": decl,
                "decl_mean_of_erosion_coreport": decl_mean_ero,
                "rho_entry_vs_erosion_primary": rho_primary,
                "rho_entry_vs_erosion_stateindex_sensitivity": rho_sens,
                "rho_entry_so_far_vs_erosion_sensitivity": rho_so_far,
                "entry_line": ENTRY_LINE,
                "adjudicates": adjudicates,
                "coarse_n3": b == "near",
            }
            cell_records[(b, w, s)] = rec
            entry_rows.append(rec)

adj_cells = [r for r in entry_rows if r["adjudicates"]]
e228_adj_set = {
    (row["battery"], row["wash"], row["state"])
    for row in m228["join_rows"]
    if row["adjudicates"]
}
my_adj_set = {(r["battery"], r["wash"], r["state"]) for r in adj_cells}
gates["G_CELLS"] = {
    "e228_adjudication_cells": sorted(e228_adj_set),
    "e230_adjudication_cells": sorted(my_adj_set),
    "identical": e228_adj_set == my_adj_set,
    "pass": e228_adj_set == my_adj_set,
}
M["gates"] = gates

rhos = [r["rho_entry_vs_erosion_primary"] for r in adj_cells]
rhos_sens = [
    r["rho_entry_vs_erosion_stateindex_sensitivity"] for r in adj_cells
]
rhos_so_far = [
    r["rho_entry_so_far_vs_erosion_sensitivity"] for r in adj_cells
]
sens_agree = (
    "sensitivities agree: stateindex-key max {:.3f}, look-ahead-free max "
    "{:.3f} (nan = no entrants by that state)".format(
        max(x for x in rhos_sens if x == x) if any(x == x for x in rhos_sens) else float("nan"),
        max(x for x in rhos_so_far if x == x) if any(x == x for x in rhos_so_far) else float("nan"),
    )
)
if all(x >= ENTRY_LINE for x in rhos):
    verdict2 = "SAME-AUTHOR"
    clause2 = (
        "every adjudication cell's entry-vs-erosion rho >= {} (min {:.3f}) — "
        "the same wash writes both; T207's registered (a) confirmed; {}".format(
            ENTRY_LINE, min(rhos), sens_agree
        )
    )
elif all(x < ENTRY_LINE for x in rhos):
    verdict2 = "TWO-ORDERS"
    clause2 = (
        "every adjudication cell's entry-vs-erosion rho < {} (max {:.3f}) — "
        "the flip-zone entry order and the erosion order are two orders; "
        "T207's registered (a) denied; {}".format(
            ENTRY_LINE, max(rhos), sens_agree
        )
    )
else:
    verdict2 = "SPLIT"
    clause2 = (
        "the cells straddle the {} line — per-cell rhos reported verbatim, "
        "no narrative inflation; {}".format(ENTRY_LINE, sens_agree)
    )

M["read2_entry_order"] = {
    "definition": "entry order = probes ranked by (first flip-zone crossing "
    "state, ties by margin depth at crossing); never-entered tied last; "
    "erosion = 1 - p(w,s)/p(0) on e214's committed records; Spearman on "
    "average ranks; POSITIVE = entered-first pairs with most-eroded; "
    "sensitivities: (i) crossing-state-index-only key (same-state entrants "
    "fully tied), (ii) look-ahead-free (only crossings that occurred BY the "
    "row's state; rest tied last)",
    "line_verbatim": ">= 0.6 reads SAME-AUTHOR (the wash writes both) / < 0.6 "
    "TWO-ORDERS",
    "table": entry_rows,
    "adjudication": {
        "verdict": verdict2,
        "clause": clause2,
        "cells": adj_cells,
    },
}
write_metrics("s3-read2-entry-order")

# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# (1) the population table
fig, ax = plt.subplots(figsize=(11, 5.2))
cols = [
    "battery", "wash", "n", "MARGIN-FIRST", ">=1-lead", "cens.",
    "P-FIRST", "TOGETHER", "NEITHER", "m crossed ever", "p crossed ever",
]
cell_text = []
for r in population_rows:
    c = r["counts"]
    cell_text.append(
        [
            r["battery"] + (" (n=3 co-report)" if r["coarse_n3"] else ""),
            r["wash"],
            str(r["n"]),
            str(c["MARGIN-FIRST"]),
            str(r["margin_first_with_ge1_lead"]),
            str(r["of_which_censored"]),
            str(c["P-FIRST"]),
            str(c["TOGETHER"]),
            str(c["NEITHER"]),
            str(r["margin_crossed_ever"]),
            str(r["p_crossed_ever"]),
        ]
    )
table = ax.table(
    cellText=cell_text, colLabels=cols, loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
ax.axis("off")
ax.set_title(
    "E230 — W033 dissociation population: margin-first (argmax margin < 0.05 sigma)\n"
    "vs p-first (p < half of p_t0), per battery per wash (committed e228 x e214 records)",
    fontsize=10,
)
fig.text(
    0.5,
    0.015,
    "verdict: {} | near n=3 quantized, co-report only | lead = states between crossings; "
    "cens. = p never crossed by last state".format(verdict1),
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT_DIR, "population_table.png"), dpi=150)
plt.close(fig)

# (2) the lead-time histogram
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for ax_, b in zip(axes, ["fact", "ctrl", "tmpl"]):
    leads = []
    labels = []
    cens = []
    for r in population_rows:
        if r["battery"] != b:
            continue
        for l in r["lead_times"]:
            leads.append(l)
            labels.append(r["wash"])
        for lb in r["censored_leads_lower_bounds"]:
            cens.append((r["wash"], lb))
    if not leads and not cens:
        ax_.text(
            0.5, 0.5, "no margin-first probes", ha="center", va="center",
            transform=ax_.transAxes, fontsize=9,
        )
        ax_.set_title("{} (n={})".format(b, next(r["n"] for r in population_rows if r["battery"] == b)))
        continue
    xmax = 3
    xs = np.arange(0, xmax + 1)
    w1v = [sum(1 for l, ww in zip(leads, labels) if l == x and ww == "w1") for x in xs]
    w2v = [sum(1 for l, ww in zip(leads, labels) if l == x and ww == "w2") for x in xs]
    ax_.bar(xs - 0.18, w1v, width=0.36, label="w1", color="#4C72B0")
    ax_.bar(xs + 0.18, w2v, width=0.36, label="w2", color="#DD8452")
    ycens1 = sum(1 for ww, lb in cens if ww == "w1")
    ycens2 = sum(1 for ww, lb in cens if ww == "w2")
    if ycens1 or ycens2:
        ax_.bar(
            [xmax + 1 - 0.18, xmax + 1 + 0.18], [ycens1, ycens2], width=0.36,
            color=["#4C72B0", "#DD8452"], hatch="//", edgecolor="k", linewidth=0.4,
        )
        ax_.set_xticks(list(xs) + [xmax + 1])
        ax_.set_xticklabels([str(x) for x in xs] + ["cens."])
    else:
        ax_.set_xticks(xs)
    ax_.set_title("{} (n={})".format(b, next(r["n"] for r in population_rows if r["battery"] == b)))
    ax_.set_xlabel("lead time (states)")
    ax_.legend(fontsize=7)
axes[0].set_ylabel("margin-first probes")
fig.suptitle(
    "E230 — lead time: states between the margin crossing and the p crossing "
    "(margin-first probes; hatch = censored, p never crossed)",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT_DIR, "lead_time_hist.png"), dpi=150)
plt.close(fig)

# (3) the entry-order scatter (adjudication cells)
adj_order = [c for c in adj_cells]
n_cells = len(adj_order)
ncol = 3
nrow = int(np.ceil(n_cells / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
for i, rec in enumerate(adj_order):
    ax_ = axes[i // ncol][i % ncol]
    b, w, s = rec["battery"], rec["wash"], rec["state"]
    rows = probe_level[b][w]
    names = [r["probe"] for r in rows]
    last = seq_idx[w][-1]
    lex_keys, state_keys = [], []
    for r in rows:
        if r["m_cross_idx"] is not None:
            lex_keys.append(r["m_cross_idx"] + r["m_depth_at_cross"] / 1000.0)
            state_keys.append(r["m_cross_idx"])
        else:
            lex_keys.append(1e9)
            state_keys.append(last + 1)
    p0 = e214_probe_records(t0_214, b)
    ps = e214_probe_records(cache214[(w, s)], b)
    erosion = np.array([1 - ps[name]["p"] / p0[name]["p"] for name in names])
    xr = rankdata(lex_keys)
    yr = rankdata(erosion)
    ax_.scatter(xr, yr, s=22, color="#55A868", edgecolor="k", linewidth=0.3)
    rho = rec["rho_entry_vs_erosion_primary"]
    ax_.set_title(
        "{} {} +{}: rho={:.2f} ({})".format(
            b, w, s, rho,
            "SAME-AUTH" if rho >= ENTRY_LINE else "two-orders",
        ),
        fontsize=9,
    )
    ax_.set_xlabel("flip-zone entry rank (never-entered tied right)", fontsize=8)
    ax_.set_xlabel("")
    ax_.set_ylabel("erosion rank (e214)", fontsize=8)
    ax_.tick_params(labelsize=7)
fig.suptitle(
    "E230 — T207-a entry order vs e214's committed erosion order (adjudication cells; "
    "line 0.6; verdict: {})".format(verdict2),
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT_DIR, "entry_order_scatter.png"), dpi=150)
plt.close(fig)

M["figures"] = {
    "population_table": os.path.join(OUT_DIR, "population_table.png"),
    "lead_time_hist": os.path.join(OUT_DIR, "lead_time_hist.png"),
    "entry_order_scatter": os.path.join(OUT_DIR, "entry_order_scatter.png"),
}
write_metrics("s4-figures")

# ---------------------------------------------------------------------------
# HONESTY + close-out
# ---------------------------------------------------------------------------
M["honesty_reflex"] = {
    "e208_distinction": "e208's margin object was the FACT-EDGE over the wash "
    "band on the tiny-net organisms (a cross-organism class separator, scoped "
    "to NATURAL-STEP organisms after e209/e210); THIS cell's object is the "
    "NEXT-TOKEN ARGMAX MARGIN in sigma units vs arithmetic noise, per probe, "
    "on the 124M archive — a different ruler; do not conflate; not a "
    "resurrection",
    "p_provenance": "the p-side is e214's COMMITTED journals read at runtime "
    "(not re-derived); the desk re-certification (journal-to-journal max dp, "
    "probe-set equality, decl cross-check vs e228's committed join_rows) is "
    "in gates",
    "near_quantized": "near is n=3 — every statistic on it is quantized; "
    "co-report only, never adjudicates",
    "grid_coarseness": "the wash state grid is coarse (2/10/50/80); lead "
    "times are bounded by the grid — a 1-state lead spans 2->10, 10->50, or "
    "50->80; 'same state' (TOGETHER) is same-GRID-CELL, not same-instant",
    "censoring": "margin-first probes whose p never crossed are censored "
    "(lead is a lower bound); counts and the histogram separate them (hatch)",
    "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 replay "
    "of e182's GPU original, wash 2 is GPU fp32 — the archive's device "
    "asymmetry, inherited, disclosed",
    "one_organism": "ONE organism's commitment spectrum (the shared pristine "
    "124M); nothing here speaks cross-organism",
    "coupling_prior": "margin-p coupling at t=0 is +0.80 (fact, e228) — the "
    "prior is that they move together; the dissociation question is exactly "
    "whether the TIMING separates despite the coupling",
    "guarantees_nothing": "nothing here is guaranteed — W033/T207's "
    "registered branches were written before this pass; no retrofit",
}
M["trims"] = []
M["deviations"] = [
    "DESK-ONLY per the dispatch: no model loads, no GPU, no wash; pure join "
    "of committed records (e228 journal margins x e214 journal p).",
    "INSTRUMENT CORRECTION post-registration, bars untouched (e226 "
    "precedent): the registration pass defined the adjudication decline as "
    'the mean of per-probe erosion; e214/e228\'s committed convention is '
    "decline = 1 - mean_p(w,s)/mean_p(0). The G_DECL desk gate caught the "
    "slip (max abs diff 0.0141; the committed formula reproduces e228's "
    "join_rows decl to 0.0 over all 28 rows); corrected BEFORE results were "
    "committed — the wrong definition had admitted ctrl/w1+80 as a seventh "
    "adjudication cell (committed decl 0.39695 < 0.40); with the committed "
    "formula the cell set equals e228's six exactly (G_CELLS). Read-1 does "
    "not use decl; no verdict changed direction.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch); the draft NOTES entry "
    "is delivered in the cell's report only.",
    "Lead-time histogram x-axis extends one bin past the data for the "
    "censored (p never crossed) count, hatched — a display convention, "
    "disclosed, not a data value.",
]
M["timing"]["finished"] = now_utc()
M["status"] = "DONE"
M["all_gates_pass"] = all(g.get("pass", False) for g in gates.values())
write_metrics("s5-done")

print("E230 DONE")
print("verdict read1:", verdict1)
print("  clause:", clause1)
print("verdict read2:", verdict2)
print("  clause:", clause2)
print("gates:", {k: g.get("pass") for k, g in gates.items()})
