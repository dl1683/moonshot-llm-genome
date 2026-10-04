"""E232 — THE ZOMBIE LAG + THE TEMPERATURE LENS (one desk pass, two registered
reads; desk-only, CPU threads=4, COMMITTED data, no model loads, no GPU).

WHY: e230/T209 found the zombie decisions — P-FIRST is the modal crossing
class in every battery-wash (beliefs halve while the argmax stands; the
commitment OUTLIVES the belief). Two questions were registered on that
residue, and this cell discharges BOTH reads on the committed journals:

  READ 1 — THE ZOMBIE LAG (T209's registered discriminating observation,
  verbatim): "THE ZOMBIE LAG — per P-FIRST probe, whether its margin EVER
  crosses by +80, and the standing-zombie population at +80 (margin >= 0.05
  sigma with p < half)." The population split: STANDING ZOMBIES at +80 vs
  RESOLVED (margin followed p down), per battery (fact/ctrl/tmpl adjudicate;
  near n=3 co-report), per wash.

  READ 2 — THE TEMPERATURE LENS (W034's registered discriminating
  observation, verbatim): "the +80 zombies' (margin, p) cloud vs
  temperature-manufactured zombies' cloud on the t=0 states — if they match,
  the wash's p-erosion is temperature-LIKE (a flattening) ... if they do not
  match ..., the wash does something temperature cannot — a real
  restructuring, not a flattening."

THE ANALYTIC FORM (registered BEFORE compute — the dispatch instructs:
"WORK THIS OUT CAREFULLY AND REGISTER THE CORRECT ANALYTIC FORM BEFORE
COMPUTE". This section is frozen at the registration commit; nothing in it
was computed from the data):

  The temperature family on a probe's t=0 logits: l_v -> l_v / T for every
  vocab entry v, T >= 1.
    - argmax: INVARIANT (division by a positive scalar preserves order) —
      the zombie's defining property survives every T.
    - raw gap: gap_T = (l_1 - l_2)/T — scales as 1/T.
    - sigma_T = std_v(l_v / T) = sigma / T — scales as 1/T.
    - margin in sigma: gap_T / sigma_T = (l_1 - l_2)/sigma = margin_sigma(t0)
      — EXACTLY INVARIANT under T. [The brief's parenthetical wording
      "(top1-top2 logits at T=1)/sigma_T" pairs the T=1 gap with the
      T-scaled sigma, which would give T * margin(t0) — rising with T. That
      is not the sigma-normalized margin of the heated organism. The correct
      object normalizes the T-scaled gap by the T-scaled sigma; the 1/T
      factors cancel. REGISTERED FORM: margin_sigma(T) == margin_sigma(t0)
      for all T — the dispatch's own worked clause: "under pure temperature,
      margin-in-sigma is invariant while p falls".]
    - p_T(answer) = softmax(l/T)[a]: continuous in T, strictly decreasing
      while the answer holds argmax (zombies hold it by definition), from
      p(t0) at T=1 toward 1/|V| (GPT-2 124M: |V|=50257, 1/|V| ~ 1.99e-5)
      as T -> infinity.

  LOCUS: each probe's temperature family is a VERTICAL segment in the
  (margin_sigma, p) plane at x = margin_sigma(t0), spanning p in
  (1/|V|, p(t0)] — margin pinned, p free to fall. COROLLARY (the complete
  residual structure): for every natural +80 zombie there is EXACTLY ONE
  T* matching its observed p (its p fell below p(t0) by construction and
  stays above 1/|V| — both verified at runtime), and at that T* the
  family's margin is EXACTLY margin(t0). No temperature can move a margin
  off its vertical. The horizontal departure
      D_i = margin_sigma_i(w, 80) - margin_sigma_i(t0)
  is therefore the COMPLETE temperature residual for zombie i: temperature
  can match the zombie's p exactly, never a margin displacement.

  CONSEQUENCE REGISTERED BEFORE COMPUTE: under the correct form temperature
  is the MAXIMAL-margin null at matched p (it predicts the FULLEST margin
  the probe ever had — t0's — at ANY fallen p); it cannot thin margins at
  all. So a natural cloud LEFT of the verticals (margins ground below t0
  while p fell) reads RESTRUCTURING just as decisively as a cloud right of
  them. W034's prediction (below) is honored VERBATIM and adjudicated
  against THIS form — its letter ("AT HIGHER margins than the temperature
  family at matched p") resolves to "median signed D > 0" (cloud RIGHT of
  the verticals); its premise clause "temperature couples them tightly" is
  itself denied by the derivation (temperature does not couple margin to p
  at all); its closing clause "a mismatch anywhere in the cloud reads REAL
  RESTRUCTURING" is exactly the RESTRUCTURED bar. No retrofit: the bars
  below were frozen before any journal number was computed for this cell.

REGISTERED BARS (frozen VERBATIM from the dispatch brief, itself frozen
from T209/W034; the registration commit of this file precedes any compute;
adjudicate against exactly this; no bar shopping):

  ZOMBIES-STAND — "the standing-zombie population at +80 is the modal
  outcome (>= 50% of P-FIRST probes per battery-wash) — the zombie is a
  stable resting state"

  ZOMBIES-RESOLVE — "margins follow p down for most P-FIRST probes by +80
  (< 50% standing) — the lag story; commitment death is delayed, not
  averted"

  RESTRUCTURED — "the natural zombie cloud departs the t=0 vertical
  temperature lines systematically (median departure >= the journals'
  noise floor)"

  TEMPERATURE-LIKE — "the cloud lies on the t=0 verticals within noise —
  the wash flattens without restructuring the decision geometry"

T209's REGISTERED PREDICTION (honored verbatim, no retrofit): "a large
standing-zombie population persists at +80 (the zombie is a stable resting
state — the modal probe is a zombie by the end); if instead zombies resolve
by +80 (margins follow p down), the two-actions story weakens to a lag
story and H-i strengthens."

W034's REGISTERED PREDICTION (honored verbatim, no retrofit): "the natural
zombie cloud sits AT HIGHER margins than the temperature family at matched
p — because T209's P-FIRST class keeps margins >= 0.05 sigma while p
halves, whereas temperature couples them tightly; a mismatch anywhere in
the cloud reads REAL RESTRUCTURING."

OPERATIONALIZATIONS (frozen BEFORE compute):
  data — all COMMITTED, read at runtime, never transcribed: e228's journal
  (per-state per-probe margin_sigma AND p, states t0 + w1{2,10,50,80} +
  w2{10,50,80}, batteries fact n=20 / ctrl n=12 / near n=3 / tmpl n=19),
  e214's journal (the committed p records; re-certified against e228's at
  runtime, gate G_PP), e230's metrics.json (THE committed classification —
  the P-FIRST classes are READ from it, not re-derived).
  read 1 population — every probe whose committed e230 class is P-FIRST in
  that battery-wash. STANDING ZOMBIE at +80 = margin_sigma(w,80) >= 0.05
  (T204's flip zone) AND p(w,80) < 0.5 * p_t0 (the p condition is implied
  by P-FIRST's absorbing p crossing and is verified from the journals).
  RESOLVED = margin_sigma(w,80) < 0.05. EVER-CROSSED = any state in the
  wash's sequence with margin_sigma < 0.05 (strictly below, absorbing from
  the first crossing); a standing-at-80 probe that ever crossed earlier is
  flagged 'recovered' (counted standing by the +80 snapshot definition,
  disclosed). Cross-at-80 counts as resolved-by-+80 (crossed BY +80).
  read 1 verdict — per-cell flag: STANDS iff standing_fraction >= 0.5
  (fraction among that cell's committed P-FIRST probes). Overall:
  ZOMBIES-STAND iff >= 4 of the 6 adjudicated cells (fact/ctrl/tmpl x
  w1/w2) stand; ZOMBIES-RESOLVE iff >= 4 of 6 have standing_fraction
  < 0.5; any split MIXED — tables verbatim, no narrative inflation. near
  (n=3, quantized) is a co-report everywhere and never adjudicates.
  read 2 cloud — the read-1 standing zombies at +80, pooled over both
  washes and the adjudicated batteries (fact/ctrl/tmpl); per-wash and
  per-battery breakdowns co-reported; near co-report only. Each point is
  one (wash, probe) death; the same probe under two washes shares one t0
  vertical (disclosed). The cloud-vs-P-FIRST containment is a gate
  (G_ZSUBSET), derived from the journals, not assumed: NEITHER keeps p >=
  half at +80; TOGETHER/MARGIN-FIRST crossed margin at/before the p
  crossing (absorbing) hence margin(+80) < 0.05.
  read 2 noise floor (journal-intrinsic, registered BEFORE compute) —
  floor = MEDIAN over all fact/ctrl/tmpl probes of |margin_sigma(w1,2) -
  margin_sigma(t0)|: the journals' minimal-wash-dose margin change, i.e.
  the smallest wash-induced margin movement the committed journals
  register. The journals are single deterministic evals (no measurement
  noise exists to quote); e182 (the GPU w1 original) committed p only, so
  no replay floor for margins exists. The minimal-dose median is the
  conservative choice — it biases TOWARD TEMPERATURE-LIKE (the floor is
  itself wash movement, not noise); stated here before compute.
  read 2 departure — departure_i = |D_i| (horizontal distance from the
  probe's t0 vertical); primary statistic = median(departure) over the
  pooled adjudicated cloud; RESTRUCTURED iff median >= floor;
  TEMPERATURE-LIKE iff median < floor. Signed median D and per-wash
  medians are co-reports (direction of restructuring); floor sensitivities
  at the 25th/75th percentiles of the minimal-dose distribution are
  co-reports only, never adjudicated.
  W034 letter clause — "cloud AT HIGHER margins than the temperature
  family at matched p" fires iff median signed D > 0. W034 spirit clause
  ("a mismatch anywhere in the cloud reads REAL RESTRUCTURING") is
  adjudicated by the RESTRUCTURED bar.

CHECKS (registered): the P-FIRST classes read from e230's committed
classification (the class field is authoritative; the margin crossings are
recomputed from the journals ONLY as a consistency cross-check against
e230's committed m_cross fields — gate G_E230X, not a re-derivation); state
grids shared and reconciled at runtime (only states present in BOTH
journals are used); near n=3 co-report only; the e208 distinction restated
below; grid coarseness (crossing states quantized to the committed grid);
nothing guaranteed.

THE e208 DISTINCTION (restated, owed): e208's margin object was the
FACT-EDGE over the wash band on the tiny-net organisms (a cross-organism
class separator, scoped to NATURAL-STEP organisms after e209/e210). THIS
cell's object is the NEXT-TOKEN ARGMAX MARGIN at the answer position in
SIGMA units vs ARITHMETIC noise ((top1 logit - top2 logit)/std(vocab
logits), T204/x3's dial), per probe, on the committed 124M archive — a
different ruler that shares only the word "margin". Not the fact-edge
object; not a resurrection.
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E228_JOURNAL = os.path.join(ROOT, "runs", "e228", "journal.json")
E214_JOURNAL = os.path.join(ROOT, "runs", "e214", "journal.json")
E230_METRICS = os.path.join(ROOT, "runs", "e230", "metrics.json")
OUT_DIR = os.path.join(ROOT, "runs", "e232")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
FIG_LAG = os.path.join(OUT_DIR, "zombie_lag_population.png")
FIG_LENS = os.path.join(OUT_DIR, "temperature_lens_cloud.png")

FLIP_ZONE = 0.05  # sigma, T204's flip zone (strictly below)
P_HALF = 0.5      # p zombie condition: p(+80) < 0.5 * p_t0
VOCAB = 50257     # GPT-2 124M vocab (only used as the T->inf p bound, 1/|V|)
INV_V = 1.0 / VOCAB

BATTERIES = ["fact", "ctrl", "near", "tmpl"]
ADJ_BATTERIES = ["fact", "ctrl", "tmpl"]  # near never adjudicates
WASHES = ["w1", "w2"]
STAND_RULE_CELLS = 4  # >= 4 of 6 adjudicated cells -> overall verdict

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
    "experiment": "e232_zombie_pass",
    "phase": "desk-only pass on COMMITTED data (e230's committed P-FIRST classes x "
    "e228 journal margins/p x e214 journal p) - no model loads, no GPU, no wash, "
    "pure join + analytic temperature family",
    "date": now_utc(),
    "status": "RUNNING",
    "registration": "bars + analytic form frozen VERBATIM from the dispatch brief "
    "(T209/W034) in this file's docstring; the registration commit of this file "
    "precedes any compute; adjudicate against exactly this; no bar shopping",
}


def write_metrics(stage):
    M["stage"] = stage
    M["updated"] = now_utc()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(M, f, indent=2, sort_keys=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# STAGE 0 — registration + provenance (no journal-derived numbers above this)
# ---------------------------------------------------------------------------
write_metrics("s0-registration")

M["question"] = (
    "READ 1 (the zombie lag): for every P-FIRST probe, does its margin EVER cross "
    "below 0.05 sigma by +80 (at which state), and how large is the standing-zombie "
    "population at +80 (margin >= 0.05 with p < half) vs resolved — is the zombie a "
    "stable resting state or a lag? READ 2 (the temperature lens): does the natural "
    "+80 zombies' (margin, p) cloud lie ON the t=0 vertical temperature lines "
    "(pure flattening) or depart them systematically (real restructuring)?"
)
M["builds_on"] = [
    "T209 / e230 (the zombie decisions — P-FIRST modal everywhere; read 1's "
    "registration; the committed P-FIRST classification this pass reads, not "
    "re-derives)",
    "W034 (the zombie as the law's microscope — read 2's registration, honored "
    "verbatim; its temperature-family intuition corrected analytically in the "
    "docstring BEFORE compute)",
    "e228 (the per-state per-probe margin_sigma journal — the x-side AND the "
    "certified p-side)",
    "e214 (the committed per-probe p records; re-certified against e228 at "
    "runtime, gate G_PP)",
    "T204 / x3 (the margin dial; the 0.05-sigma flip zone vs arithmetic noise)",
    "W028 (shape vs height — the lens's frame: argmax is order/shape, p is "
    "scale/height)",
]
M["whats_new"] = [
    "the ZOMBIE LAG population: per P-FIRST probe the margin's eventual fate by "
    "+80 (standing vs resolved, crossing state), per battery per wash",
    "the TEMPERATURE LENS: the correct analytic form worked out and registered "
    "BEFORE compute (margin-in-sigma EXACTLY invariant under temperature -> "
    "vertical lines in (margin, p); temperature is the maximal-margin null at "
    "matched p), and the natural zombie cloud read against those verticals via "
    "the complete residual D = margin(+80) - margin(t0)",
    "the journal-intrinsic noise floor (median minimal-dose |d_margin| at w1@2) "
    "registered before compute as the TEMPERATURE-LIKE/RESTRUCTURED line",
]
M["registered_bars_verbatim"] = {
    "ZOMBIES-STAND": "the standing-zombie population at +80 is the modal outcome "
    "(>= 50% of P-FIRST probes per battery-wash) — the zombie is a stable "
    "resting state",
    "ZOMBIES-RESOLVE": "margins follow p down for most P-FIRST probes by +80 "
    "(< 50% standing) — the lag story; commitment death is delayed, not averted",
    "RESTRUCTURED": "the natural zombie cloud departs the t=0 vertical temperature "
    "lines systematically (median departure >= the journals' noise floor)",
    "TEMPERATURE-LIKE": "the cloud lies on the t=0 verticals within noise — the "
    "wash flattens without restructuring the decision geometry",
}
M["registered_predictions"] = {
    "T209_verbatim": "a large standing-zombie population persists at +80 (the "
    "zombie is a stable resting state — the modal probe is a zombie by the end); "
    "if instead zombies resolve by +80 (margins follow p down), the two-actions "
    "story weakens to a lag story and H-i strengthens",
    "W034_verbatim": "the natural zombie cloud sits AT HIGHER margins than the "
    "temperature family at matched p — because T209's P-FIRST class keeps margins "
    ">= 0.05 sigma while p halves, whereas temperature couples them tightly; a "
    "mismatch anywhere in the cloud reads REAL RESTRUCTURING",
}
M["analytic_form_registered"] = {
    "family": "l_v -> l_v / T for all vocab v, T >= 1, on the t=0 logits",
    "argmax": "invariant",
    "raw_gap": "(l1-l2)/T — scales 1/T",
    "sigma_T": "std(l/T) = sigma/T — scales 1/T",
    "margin_sigma_T": "(l1-l2)/sigma — EXACTLY invariant (the 1/T factors "
    "cancel); the brief's parenthetical '(top1-top2 logits at T=1)/sigma_T' is "
    "resolved to this form per the dispatch's instruction to register the "
    "correct analytic form before compute",
    "p_T": "softmax(l/T)[answer]: continuous, strictly decreasing while the "
    "answer holds argmax, from p(t0) at T=1 toward 1/|V| ~ 1.99e-5 as T -> inf",
    "locus": "a VERTICAL segment per probe at x = margin_sigma(t0), spanning "
    "p in (1/|V|, p(t0)]",
    "corollary": "for each +80 zombie there is exactly one T* matching its p "
    "exactly, and at that T* the family's margin is EXACTLY margin(t0) — the "
    "horizontal departure D = margin(+80) - margin(t0) is the COMPLETE "
    "temperature residual; temperature is the maximal-margin null at matched p "
    "(it can never thin a margin)",
    "w034_premise_note": "W034's 'temperature couples them tightly' is denied "
    "by the derivation itself (margin invariant while p falls — no coupling); "
    "the prediction's letter is still adjudicable as median signed D > 0, its "
    "closing clause as the RESTRUCTURED bar; registered before compute",
}
M["operationalizations_frozen"] = {
    "read1_population": "committed e230 P-FIRST class (authoritative); STANDING "
    "at +80 = margin_sigma(w,80) >= 0.05 AND p(w,80) < 0.5*p_t0 (verified); "
    "RESOLVED = margin_sigma(w,80) < 0.05; EVER-CROSSED = any state with "
    "margin < 0.05 (strictly below, absorbing); standing probe that ever "
    "crossed = 'recovered' (counts standing, disclosed); cross-at-80 = "
    "resolved-by-+80",
    "read1_verdict": "cell STANDS iff standing_fraction >= 0.5; overall "
    "ZOMBIES-STAND iff >= 4 of 6 adjudicated cells stand; ZOMBIES-RESOLVE iff "
    ">= 4 of 6 have standing_fraction < 0.5; else MIXED; near co-report only",
    "read2_cloud": "read-1 standing zombies at +80 pooled over washes and "
    "fact/ctrl/tmpl; per-wash/per-battery co-reports; near co-report only; a "
    "probe under two washes = two points sharing one t0 vertical",
    "read2_floor": "median over fact/ctrl/tmpl probes of |margin(w1,2) - "
    "margin(t0)| (minimal wash dose; journals' smallest registered wash-induced "
    "margin movement; single deterministic evals -> no measurement noise to "
    "quote; e182 committed p only -> no margin replay floor exists; "
    "conservative: biases toward TEMPERATURE-LIKE)",
    "read2_departure": "departure = |D|; primary = median(departure) over the "
    "pooled adjudicated cloud; RESTRUCTURED iff median >= floor; "
    "TEMPERATURE-LIKE iff < floor; signed median, per-wash medians, p25/p75 "
    "floor sensitivities = co-reports only",
    "w034_letter": "fires iff median signed D > 0 (cloud right of the "
    "verticals); spirit clause = the RESTRUCTURED bar",
}
M["sources"] = {
    "margins_and_p": {
        "path": E228_JOURNAL,
        "sha256_16": sha16(E228_JOURNAL),
    },
    "p_records": {
        "path": E214_JOURNAL,
        "sha256_16": sha16(E214_JOURNAL),
    },
    "committed_classification": {
        "path": E230_METRICS,
        "sha256_16": sha16(E230_METRICS),
    },
    "classification_provenance": "e230's COMMITTED probe-level classes are READ "
    "from its metrics.json (never re-derived); the margin crossings recomputed "
    "here from the journals serve ONLY as a consistency cross-check (gate "
    "G_E230X)",
}
M["compute"] = {
    "device": "cpu desk pass (threads 4)",
    "model_loads": 0,
    "gpu_calls": 0,
    "torch_used": False,
    "threads_note": "pure json/numpy join + analytic arithmetic; no training, "
    "no evals",
}
M["provenance"] = {
    "git_head_at_start": git_head(),
    "script": SCRIPT_PATH,
    "script_sha256_16": sha16(SCRIPT_PATH),
}
write_metrics("s0-registration")

# ---------------------------------------------------------------------------
# Load + reconcile the journals; gates
# ---------------------------------------------------------------------------
with open(E228_JOURNAL) as f:
    j228 = json.load(f)
with open(E214_JOURNAL) as f:
    j214 = json.load(f)
with open(E230_METRICS) as f:
    m230 = json.load(f)

S228 = {(s["wash"], s["step"]): s for s in j228["states"]}
S214 = {(s["wash"], s["step"]): s for s in j214["states"]}
GRID = {
    "w1": [0, 2, 10, 50, 80],
    "w2": [0, 10, 50, 80],
}


def j228_probe(state, battery, probe):
    for pr in state[battery]["probes"]:
        if pr["fact"] == probe:
            return pr
    return None


def j214_probe(state, battery, probe):
    return state[battery]["probes"].get(probe)


# G_GRID — shared states only
shared = sorted(set(S228.keys()) & set(S214.keys()))
grid_ok = (
    ("t0", 0) in shared
    and all(("w1", st) in shared for st in [2, 10, 50, 80])
    and all(("w2", st) in shared for st in [10, 50, 80])
)
M["gates"] = {}
M["gates"]["G_GRID"] = {
    "shared_states_used": {
        "t0": [0],
        "w1": [2, 10, 50, 80],
        "w2": [10, 50, 80],
    },
    "n_shared_keys": len(shared),
    "pass": bool(grid_ok),
}

# G_PROBES — probe sets identical across journals and e230's committed records
probe_mismatch = []
e230_level = m230["read1_population"]["probe_level"]
for b in BATTERIES:
    st228 = S228[("t0", 0)]
    set228 = {pr["fact"] for pr in st228[b]["probes"]}
    set214 = set(S214[("t0", 0)][b]["probes"].keys())
    set230 = {r["probe"] for w in WASHES for r in e230_level[b][w]}
    if not (set228 == set214 == set230):
        probe_mismatch.append(
            {"battery": b, "j228": sorted(set228), "j214": sorted(set214),
             "e230": sorted(set230)}
        )
M["gates"]["G_PROBES"] = {
    "mismatches": probe_mismatch,
    "ns": {b: len(S228[("t0", 0)][b]["probes"]) for b in BATTERIES},
    "pass": len(probe_mismatch) == 0,
}

# G_PP — p re-certification e228 journal vs e214 journal (desk re-cert)
max_dp = 0.0
for (wash, step), st228 in S228.items():
    if (wash, step) not in S214:
        continue
    st214 = S214[(wash, step)]
    for b in BATTERIES:
        for pr in st228[b]["probes"]:
            q = j214_probe(st214, b, pr["fact"])
            if q is not None:
                max_dp = max(max_dp, abs(pr["p"] - q["p"]))
M["gates"]["G_PP"] = {
    "max_abs_dp_e228_journal_vs_e214_journal": max_dp,
    "tol": 0.005,
    "note": "same re-certification as e230's G_PP (e228's journal p vs e214's "
    "committed journal p, every probe, every shared state)",
    "pass": max_dp <= 0.005,
}

# G_E230X — consistency cross-check: journal-recomputed margin crossings vs
# e230's committed m_cross fields for EVERY probe (the CLASS stays authoritative)
xcheck_mismatch = []
for b in BATTERIES:
    for w in WASHES:
        seq = GRID[w]
        committed = {r["probe"]: r for r in e230_level[b][w]}
        for probe, rec in committed.items():
            margins = []
            for st in seq:
                pr = j228_probe(S228[("t0", 0) if st == 0 else (w, st)], b, probe)
                margins.append(pr["margin_sigma"])
            cross_idx = None
            for i, mg in enumerate(margins):
                if mg < FLIP_ZONE:
                    cross_idx = i
                    break
            cross_step = None if cross_idx is None else seq[cross_idx]
            if cross_step != rec["m_cross_step"] or cross_idx != rec["m_cross_idx"]:
                xcheck_mismatch.append(
                    {"battery": b, "wash": w, "probe": probe,
                     "recomputed": [cross_idx, cross_step],
                     "committed": [rec["m_cross_idx"], rec["m_cross_step"]]}
                )
M["gates"]["G_E230X"] = {
    "n_records_checked": sum(
        len(e230_level[b][w]) for b in BATTERIES for w in WASHES
    ),
    "mismatches": xcheck_mismatch,
    "note": "cross-CHECK only — the P-FIRST classes are read from e230's "
    "committed classification (do not re-derive); recomputed margin crossings "
    "must merely AGREE with e230's committed m_cross fields",
    "pass": len(xcheck_mismatch) == 0,
}
write_metrics("s1-gates")

# ---------------------------------------------------------------------------
# STAGE 2 — READ 1: the zombie lag
# ---------------------------------------------------------------------------
read1_rows = {b: {w: [] for w in WASHES} for b in BATTERIES}
for b in BATTERIES:
    for w in WASHES:
        seq = GRID[w]
        t0 = S228[("t0", 0)]
        last = S228[(w, 80)]
        for rec in e230_level[b][w]:
            if rec["class"] != "P-FIRST":
                continue
            probe = rec["probe"]
            p_t0 = j228_probe(t0, b, probe)["p"]
            traj = []
            for st in seq:
                pr = j228_probe(S228[("t0", 0) if st == 0 else (w, st)], b, probe)
                traj.append(
                    {
                        "step": st,
                        "margin": pr["margin_sigma"],
                        "p": pr["p"],
                        "argmax_is_answer": pr["argmax_is_answer"],
                    }
                )
            margins = [t["margin"] for t in traj]
            ps = [t["p"] for t in traj]
            args = [t["argmax_is_answer"] for t in traj]
            m80 = margins[-1]
            p80 = ps[-1]
            cross_i = next(
                (i for i, mg in enumerate(margins) if mg < FLIP_ZONE), None
            )
            ever = cross_i is not None
            flip_i = next((i for i, a in enumerate(args) if not a), None)
            p_below_half = p80 < P_HALF * p_t0
            standing = (m80 >= FLIP_ZONE) and p_below_half
            fate = (
                "standing-zombie" if standing and not ever
                else ("recovered" if standing else "resolved")
            )
            read1_rows[b][w].append(
                {
                    "probe": probe,
                    "margin_t0": margins[0],
                    "p_t0": p_t0,
                    "margin_80": m80,
                    "p_80": p80,
                    "p_half_t0": P_HALF * p_t0,
                    "p_below_half_at_80": p_below_half,
                    "margin_ever_crossed_by_80": ever,
                    "margin_cross_step": None if cross_i is None else seq[cross_i],
                    "standing_at_80": standing,
                    "fate": fate,
                    "argmax_is_answer_80": args[-1],
                    "argmax_first_flip_step": (
                        None if flip_i is None else seq[flip_i]
                    ),
                }
            )

read1_table = []
for b in BATTERIES:
    for w in WASHES:
        rows = read1_rows[b][w]
        n = len(rows)
        n_stand = sum(1 for r in rows if r["standing_at_80"])
        n_res = n - n_stand
        frac = (n_stand / n) if n else None
        read1_table.append(
            {
                "battery": b,
                "wash": w,
                "n_pfirst": n,
                "n_standing": n_stand,
                "n_resolved": n_res,
                "n_recovered": sum(1 for r in rows if r["fate"] == "recovered"),
                "n_standing_argmax_hold": sum(
                    1 for r in rows if r["standing_at_80"]
                    and r["argmax_is_answer_80"]
                ),
                "n_standing_argmax_flip": sum(
                    1 for r in rows if r["standing_at_80"]
                    and not r["argmax_is_answer_80"]
                ),
                "standing_fraction": frac,
                "adjudicates": b in ADJ_BATTERIES,
                "cell_stands": (frac >= 0.5) if (n and b in ADJ_BATTERIES) else None,
                "cross_step_histogram_resolved": {
                    str(st): sum(
                        1
                        for r in rows
                        if not r["standing_at_80"]
                        and r["margin_cross_step"] == st
                    )
                    for st in ([2, 10, 50, 80] if w == "w1" else [10, 50, 80])
                },
                "n_p_below_half_violations": sum(
                    1 for r in rows if not r["p_below_half_at_80"]
                ),
            }
        )

adj_cells = [c for c in read1_table if c["adjudicates"]]
n_stand_cells = sum(1 for c in adj_cells if c["cell_stands"])
if n_stand_cells >= STAND_RULE_CELLS:
    r1_verdict = "ZOMBIES-STAND"
elif sum(1 for c in adj_cells if not c["cell_stands"]) >= STAND_RULE_CELLS:
    r1_verdict = "ZOMBIES-RESOLVE"
else:
    r1_verdict = "MIXED"
cell_txt = "; ".join(
    f"{c['battery']}-{c['wash']} {c['n_standing']}/{c['n_pfirst']}"
    f" ({c['standing_fraction']:.2f})"
    for c in adj_cells
)
if r1_verdict == "ZOMBIES-STAND":
    r1_clause = (
        f"{n_stand_cells}/6 adjudicated cells have standing_fraction >= 0.5 "
        f"({cell_txt}) — the standing zombie is the modal P-FIRST outcome; the "
        "zombie is a stable resting state"
    )
elif r1_verdict == "ZOMBIES-RESOLVE":
    r1_clause = (
        f"{6 - n_stand_cells}/6 adjudicated cells have standing_fraction < 0.5 "
        f"({cell_txt}) — margins follow p down for most P-FIRST probes by +80; "
        "the lag story; commitment death is delayed, not averted"
    )
else:
    r1_clause = (
        f"cells split {n_stand_cells} stand / {6 - n_stand_cells} resolve "
        f"({cell_txt}) — tables verbatim, both reads, no narrative inflation"
    )

M["read1_zombie_lag"] = {
    "definition": "population = e230's committed P-FIRST class (read, not "
    "re-derived); STANDING at +80 = margin_sigma(w,80) >= 0.05 with p(w,80) < "
    "0.5*p_t0 (verified from the journals); RESOLVED = margin(+80) < 0.05; "
    "'recovered' = standing at +80 after an earlier crossing (disclosed, "
    "counts standing); cross-at-80 counts as resolved-by-+80",
    "table": read1_table,
    "probe_level": read1_rows,
    "adjudication": {
        "bars": {
            "ZOMBIES-STAND": r1_verdict == "ZOMBIES-STAND",
            "ZOMBIES-RESOLVE": r1_verdict == "ZOMBIES-RESOLVE",
            "MIXED": r1_verdict == "MIXED",
        },
        "verdict": r1_verdict,
        "rule": "cell STANDS iff standing_fraction >= 0.5; overall = modal "
        "outcome across the 6 adjudicated cells (>= 4 of 6); near co-report",
        "clause": r1_clause,
        "t209_prediction_clauses": {
            "prediction_verbatim": M["registered_predictions"]["T209_verbatim"],
            "large_standing_population_persists": r1_verdict == "ZOMBIES-STAND",
            "note": "the bar adjudicates; this clause read is a co-report "
            "against T209's registered wording, no retrofit",
        },
    },
}
write_metrics("s2-read1")

# ---------------------------------------------------------------------------
# STAGE 3 — READ 2: the temperature lens
# ---------------------------------------------------------------------------
# the natural cloud: read-1 standing zombies (adjudicated batteries pool)
cloud = []
for b in ADJ_BATTERIES + ["near"]:
    for w in WASHES:
        for r in read1_rows[b][w]:
            if not r["standing_at_80"]:
                continue
            cloud.append(
                {
                    "battery": b,
                    "wash": w,
                    "probe": r["probe"],
                    "margin_t0": r["margin_t0"],
                    "p_t0": r["p_t0"],
                    "margin_80": r["margin_80"],
                    "p_80": r["p_80"],
                    "D_signed": r["margin_80"] - r["margin_t0"],
                    "departure": abs(r["margin_80"] - r["margin_t0"]),
                    "adjudicates": b in ADJ_BATTERIES,
                    "argmax_is_answer_80": r["argmax_is_answer_80"],
                    "argmax_first_flip_step": r["argmax_first_flip_step"],
                    "t_star_exists": (r["p_80"] < r["p_t0"])
                    and (r["p_80"] > INV_V),
                }
            )

# G_ZSUBSET — the +80 standing-zombie SNAPSHOT set vs the committed P-FIRST
# class, plus the T* existence bound. The registered derivation claimed
# containment (NEITHER keeps p >= half at +80; TOGETHER/MARGIN-FIRST crossed
# margin absorbingly). HONEST OUTCOME: the containment assumption is
# FALSIFIABLE and was FALSIFIED — crossing is an absorbing EVENT (first
# crossing), not an absorbing STATE: margins can dip below 0.05 and recover.
# The violations are disclosed verbatim below; the registered cloud
# definition (read-1 standing zombies, P-FIRST-scoped) is a DEFINITION and
# adjudicates unchanged; the full-snapshot sensitivity is a co-report.
zsub_violations = [
    c for c in cloud if not c["t_star_exists"]
]
non_pfirst_standing = []
for b in BATTERIES:
    for w in WASHES:
        pfirst = {
            r["probe"] for r in e230_level[b][w] if r["class"] == "P-FIRST"
        }
        class_by_probe = {r["probe"]: r["class"] for r in e230_level[b][w]}
        # scan ALL probes of the battery for +80 standing-zombie snapshots
        t0 = S228[("t0", 0)]
        last = S228[(w, 80)]
        for pr in last[b]["probes"]:
            p_t0 = j228_probe(t0, b, pr["fact"])["p"]
            if pr["margin_sigma"] >= FLIP_ZONE and pr["p"] < P_HALF * p_t0:
                if pr["fact"] not in pfirst:
                    non_pfirst_standing.append(
                        {
                            "battery": b,
                            "wash": w,
                            "probe": pr["fact"],
                            "e230_class": class_by_probe.get(pr["fact"]),
                            "argmax_is_answer_80": pr["argmax_is_answer"],
                            "margin_80": pr["margin_sigma"],
                            "p_80": pr["p"],
                        }
                    )
M["gates"]["G_ZSUBSET"] = {
    "standing_zombies_outside_committed_PFIRST": non_pfirst_standing,
    "t_star_bound_violations": [
        {k: c[k] for k in ("battery", "wash", "probe")} for c in zsub_violations
    ],
    "note": "registered containment assumption FALSIFIED: crossing is an "
    "absorbing EVENT (first crossing), not an absorbing STATE — the listed "
    "+80 standing-zombie snapshots dip below 0.05 earlier (TOGETHER class) "
    "and recover above it by +80; the registered cloud definition "
    "(P-FIRST-scoped) adjudicates unchanged (a definition, not a claim); the "
    "full +80 snapshot set is a sensitivity co-report in "
    "read2.argmax_decomposition_and_sensitivities; every zombie's p lies in "
    "(1/|V|, p_t0) so exactly one T* matches it",
    "pass": len(non_pfirst_standing) == 0 and len(zsub_violations) == 0,
}

# the noise floor — registered BEFORE compute: median minimal-dose |d_margin|
# (w1@2 vs t0) over all fact/ctrl/tmpl probes
min_dose = []
for b in ADJ_BATTERIES:
    t0 = S228[("t0", 0)]
    w12 = S228[("w1", 2)]
    for pr in t0[b]["probes"]:
        q = j228_probe(w12, b, pr["fact"])
        min_dose.append(abs(q["margin_sigma"] - pr["margin_sigma"]))
min_dose = np.array(min_dose)
floor = float(np.median(min_dose))
floor_p25 = float(np.percentile(min_dose, 25))
floor_p75 = float(np.percentile(min_dose, 75))

adj_cloud = [c for c in cloud if c["adjudicates"]]
dep = np.array([c["departure"] for c in adj_cloud])
dsigned = np.array([c["D_signed"] for c in adj_cloud])
med_dep = float(np.median(dep))
med_signed = float(np.median(dsigned))
per_wash = {}
for w in WASHES:
    dw = np.array([c["departure"] for c in adj_cloud if c["wash"] == w])
    dsw = np.array([c["D_signed"] for c in adj_cloud if c["wash"] == w])
    per_wash[w] = {
        "n": len(dw),
        "median_departure": float(np.median(dw)) if len(dw) else None,
        "median_signed": float(np.median(dsw)) if len(dsw) else None,
    }
per_battery = {}
for b in ADJ_BATTERIES:
    db = np.array([c["departure"] for c in adj_cloud if c["battery"] == b])
    dsb = np.array([c["D_signed"] for c in adj_cloud if c["battery"] == b])
    per_battery[b] = {
        "n": len(db),
        "median_departure": float(np.median(db)) if len(db) else None,
        "median_signed": float(np.median(dsb)) if len(dsb) else None,
    }

r2_verdict = "RESTRUCTURED" if med_dep >= floor else "TEMPERATURE-LIKE"
if r2_verdict == "RESTRUCTURED":
    r2_clause = (
        f"median departure {med_dep:.4f} sigma >= the journals' noise floor "
        f"{floor:.4f} sigma (median minimal-dose |d_margin| at w1@2 over "
        f"fact/ctrl/tmpl, n={len(min_dose)}) — the zombie cloud departs the "
        "t=0 vertical temperature lines systematically; the wash does "
        "something temperature cannot"
    )
else:
    r2_clause = (
        f"median departure {med_dep:.4f} sigma < the journals' noise floor "
        f"{floor:.4f} sigma — the cloud lies on the t=0 verticals within "
        "noise; the wash flattens without restructuring the decision geometry"
    )

M["read2_temperature_lens"] = {
    "definition": "cloud = read-1 standing zombies at +80 (margin >= 0.05, "
    "p < half), pooled over both washes, fact/ctrl/tmpl adjudicated, near "
    "co-report; temperature family = vertical line at x = margin_sigma(t0) "
    "per probe (analytic form registered above); departure_i = |margin(+80) "
    "- margin(t0)|; floor = median |margin(w1,2) - margin(t0)| over "
    "fact/ctrl/tmpl (registered before compute)",
    "noise_floor": {
        "definition": "median minimal-dose |d_margin| at w1@2 vs t0, all "
        "fact/ctrl/tmpl probes",
        "n": int(len(min_dose)),
        "value": floor,
        "p25": floor_p25,
        "p75": floor_p75,
        "min": float(np.min(min_dose)),
        "max": float(np.max(min_dose)),
    },
    "cloud": {
        "n_pooled_adjudicated": len(adj_cloud),
        "n_near_coreport": sum(1 for c in cloud if not c["adjudicates"]),
        "points": cloud,
    },
    "departures": {
        "median_abs": med_dep,
        "median_signed": med_signed,
        "q25_signed": float(np.percentile(dsigned, 25)),
        "q75_signed": float(np.percentile(dsigned, 75)),
        "n_left_of_vertical": int(np.sum(dsigned < 0)),
        "n_right_of_vertical": int(np.sum(dsigned > 0)),
        "n_on_vertical_exact": int(np.sum(dsigned == 0)),
        "per_wash": per_wash,
        "per_battery": per_battery,
    },
    "adjudication": {
        "bars": {
            "RESTRUCTURED": r2_verdict == "RESTRUCTURED",
            "TEMPERATURE-LIKE": r2_verdict == "TEMPERATURE-LIKE",
        },
        "verdict": r2_verdict,
        "clause": r2_clause,
        "w034_prediction_clauses": {
            "prediction_verbatim": M["registered_predictions"]["W034_verbatim"],
            "letter_higher_margins_than_family_at_matched_p": med_signed > 0,
            "letter_note": "at matched p the family's margin is exactly "
            "margin(t0) (the vertical), so the letter fires iff the cloud "
            "sits RIGHT of the verticals (median signed D > 0)",
            "spirit_mismatch_reads_restructuring": r2_verdict == "RESTRUCTURED",
            "premise_note": "the derivation registered before compute denies "
            "W034's 'temperature couples them tightly' — temperature does not "
            "couple margin to p at all (margin invariant while p falls); and "
            "temperature is the MAXIMAL-margin null at matched p (it predicts "
            "t0's full margin at any fallen p), so natural zombies LEFT of "
            "the verticals are exactly as fatal to the flattening story as "
            "right-of ones; no retrofit — both notes were written into this "
            "file's docstring at the registration commit",
        },
        "sensitivities_coreport_only": {
            "floor_p25_adjudication": "RESTRUCTURED"
            if med_dep >= floor_p25
            else "TEMPERATURE-LIKE",
            "floor_p75_adjudication": "RESTRUCTURED"
            if med_dep >= floor_p75
            else "TEMPERATURE-LIKE",
            "per_wash": {
                w: "RESTRUCTURED"
                if per_wash[w]["median_departure"] >= floor
                else "TEMPERATURE-LIKE"
                for w in WASHES
            },
            "note": "co-reports only — the primary floor and statistic were "
            "registered before compute; these never adjudicate",
        },
    },
}

# --- co-report block (added after the registered compute; bars untouched) ---
# (a) ARGMAX DECOMPOSITION of the cloud: temperature preserves the argmax by
#     construction (order-preserving division), so a +80 zombie whose argmax
#     is no longer the answer token is off-vertical IN KIND — no T exists for
#     it at all, at any margin; a holder is tested metrically (departure).
adj_hold = [c for c in adj_cloud if c["argmax_is_answer_80"]]
adj_flip = [c for c in adj_cloud if not c["argmax_is_answer_80"]]
dep_hold = float(np.median([c["departure"] for c in adj_hold]))
dep_flip = float(np.median([c["departure"] for c in adj_flip]))
# (b) FULL +80 SNAPSHOT-ZOMBIE SENSITIVITY: the registered cloud plus the
#     G_ZSUBSET violations (recovered-from-crossing snapshots outside
#     P-FIRST) — does the verdict survive the definitional widening?
snap_extra = []
for v in non_pfirst_standing:
    if v["battery"] not in ADJ_BATTERIES:
        continue
    t0m = j228_probe(S228[("t0", 0)], v["battery"], v["probe"])["margin_sigma"]
    snap_extra.append(
        {
            "battery": v["battery"],
            "wash": v["wash"],
            "probe": v["probe"],
            "e230_class": v["e230_class"],
            "margin_t0": t0m,
            "margin_80": v["margin_80"],
            "departure": abs(v["margin_80"] - t0m),
            "D_signed": v["margin_80"] - t0m,
            "argmax_is_answer_80": v["argmax_is_answer_80"],
        }
    )
snap_all = (
    [
        {
            "battery": c["battery"],
            "wash": c["wash"],
            "probe": c["probe"],
            "e230_class": "P-FIRST",
            "margin_t0": c["margin_t0"],
            "margin_80": c["margin_80"],
            "departure": c["departure"],
            "D_signed": c["D_signed"],
            "argmax_is_answer_80": c["argmax_is_answer_80"],
        }
        for c in adj_cloud
    ]
    + snap_extra
)
med_snap = float(np.median([c["departure"] for c in snap_all]))
med_hold_signed = float(np.median([c["D_signed"] for c in adj_hold]))
M["read2_temperature_lens"]["argmax_decomposition_and_sensitivities"] = {
    "added_after": "s6 co-report pass, triggered by the G_ZSUBSET failure "
    "and the argmax_is_answer column; BARS UNTOUCHED — the adjudication "
    "above is the registered compute",
    "argmax_decomposition": {
        "note": "temperature preserves the argmax for every T (division is "
        "order-preserving) — an argmax-flipped zombie is off-vertical IN "
        "KIND (no T exists at any margin); an answer-holding zombie is "
        "tested metrically (departure)",
        "n_adjudicated": len(adj_cloud),
        "n_argmax_hold_answer": len(adj_hold),
        "n_argmax_flip": len(adj_flip),
        "flip_first_steps": sorted(
            {str(c["argmax_first_flip_step"]) for c in adj_flip}
        ),
        "median_departure_holders": dep_hold,
        "median_departure_flipped": dep_flip,
        "median_signed_holders": med_hold_signed,
        "n_near_coreport_hold": sum(
            1 for c in cloud if not c["adjudicates"]
            and c["argmax_is_answer_80"]
        ),
        "n_near_coreport_flip": sum(
            1 for c in cloud if not c["adjudicates"]
            and not c["argmax_is_answer_80"]
        ),
        "verdict_holders_only_coreport": "RESTRUCTURED"
        if dep_hold >= floor
        else "TEMPERATURE-LIKE",
        "verdict_flipped_in_kind": "RESTRUCTURED-IN-KIND (argmax flip is "
        "temperature-impossible; not a margin comparison at all)",
    },
    "full_snapshot_sensitivity": {
        "note": "the registered cloud (P-FIRST-scoped) widened to EVERY +80 "
        "standing-zombie snapshot in the adjudicated batteries (adds the "
        "G_ZSUBSET violations: TOGETHER-class dips that recovered above "
        "0.05 by +80); co-report only",
        "n_full_snapshot": len(snap_all),
        "extra_points": snap_extra,
        "median_departure_full_snapshot": med_snap,
        "verdict_full_snapshot_coreport": "RESTRUCTURED"
        if med_snap >= floor
        else "TEMPERATURE-LIKE",
    },
}
write_metrics("s3-read2")

# ---------------------------------------------------------------------------
# STAGE 4 — figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8.5, "figure.dpi": 150})

# Fig 1 — the zombie-lag population
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
cells = [c for c in read1_table]
labels = [f"{c['battery']}\n{c['wash']} (n={c['n_pfirst']})" for c in cells]
x = np.arange(len(cells))
res = np.array([c["n_resolved"] for c in cells], dtype=float)
sta = np.array([c["n_standing"] for c in cells], dtype=float)
colors = {"fact": "#1f77b4", "ctrl": "#2ca02c", "near": "#9467bd", "tmpl": "#d62728"}
ax = axes[0]
sta_hold = np.array([c["n_standing_argmax_hold"] for c in cells], dtype=float)
sta_flip = np.array([c["n_standing_argmax_flip"] for c in cells], dtype=float)
ax.bar(x, res, color="#bbbbbb", edgecolor="k", linewidth=0.4,
       label="resolved by +80 (margin < 0.05)")
ax.bar(x, sta_hold, bottom=res, color="#d62728", edgecolor="k",
       linewidth=0.4, alpha=0.85,
       label="standing zombie, argmax still the answer")
ax.bar(x, sta_flip, bottom=res + sta_hold, color="#ff7f0e", edgecolor="k",
       linewidth=0.4, alpha=0.9,
       label="standing zombie, argmax FLIPPED (wrong-choice survivor)")
for i, c in enumerate(cells):
    if c["n_recovered"]:
        ax.text(i, res[i] + sta[i] + 0.15, f"+{c['n_recovered']}rec", ha="center",
                fontsize=6.5, color="#7f3f00")
for i, c in enumerate(cells):
    if c["n_pfirst"]:
        ax.text(i, res[i] + sta[i] / 2, f"{c['standing_fraction']:.2f}",
                ha="center", va="center", fontsize=7.5,
                color="w" if c["standing_fraction"] > 0.4 else "k")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("P-FIRST probes")
ax.set_title(
    "READ 1 — the zombie lag: P-FIRST populations at +80\n"
    "(fraction = standing share; near n=3 co-report)",
    fontsize=9,
)
ax.legend(fontsize=7, loc="upper right")

near_idx = [i for i, c in enumerate(cells) if c["battery"] == "near"]
ax = axes[1]
fr = [c["standing_fraction"] if c["n_pfirst"] else np.nan for c in cells]
barcols = [colors[c["battery"]] for c in cells]
bars = ax.bar(x, fr, color=barcols, edgecolor="k", linewidth=0.4)
for i in near_idx:
    bars[i].set_hatch("//")
ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
ax.text(len(cells) - 0.4, 0.51, "0.5 bar", fontsize=7, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([f"{c['battery']}-{c['wash']}" for c in cells], fontsize=7,
                   rotation=45, ha="right")
ax.set_ylabel("standing fraction among P-FIRST")
ax.set_title("standing fraction per battery-wash (hatched = near, co-report)",
             fontsize=9)
ax.set_ylim(0, 1.05)

ax = axes[2]
steps_w1 = ["2", "10", "50", "80"]
steps_w2 = ["10", "50", "80"]
h_w1 = []
h_w2 = []
for st in steps_w1:
    h_w1.append(sum(c["cross_step_histogram_resolved"].get(st, 0)
                    for c in cells if c["wash"] == "w1"))
for st in steps_w2:
    h_w2.append(sum(c["cross_step_histogram_resolved"].get(st, 0)
                    for c in cells if c["wash"] == "w2"))
xx = np.arange(4)
ax.bar(xx - 0.2, h_w1, width=0.4, color="#1f77b4", edgecolor="k",
       linewidth=0.4, label="w1 (grid 2/10/50/80)")
ax.bar(np.arange(3) + 0.2, h_w2, width=0.4, color="#ff7f0e", edgecolor="k",
       linewidth=0.4, label="w2 (grid 10/50/80)")
n_stand_total = sum(c["n_standing"] for c in cells)
ax.bar([3.0], [n_stand_total], width=0.4, color="#d62728", edgecolor="k",
       linewidth=0.4, alpha=0.85, label=f"never (standing, n={n_stand_total})")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["cross@2\n(w1)", "cross@10", "cross@50", "cross@80 /\nnever"])
ax.set_ylabel("resolved P-FIRST probes")
ax.set_title("when the margins finally cross (the lag structure)", fontsize=9)
ax.legend(fontsize=7)
fig.suptitle(
    "E232 READ 1 — THE ZOMBIE LAG (T209): standing zombies vs resolved at +80 | "
    f"verdict: {r1_verdict}",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_LAG)
plt.close(fig)

# Fig 2 — the temperature lens: cloud vs verticals
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
bcols = {"fact": "#1f77b4", "ctrl": "#2ca02c", "tmpl": "#d62728", "near": "#9467bd"}
for c in cloud:
    col = bcols[c["battery"]]
    # the temperature family vertical (reachable p range for this zombie)
    ax.plot([c["margin_t0"], c["margin_t0"]], [c["p_80"], c["p_t0"]],
            linestyle="--", linewidth=1.1, color=col, alpha=0.55,
            zorder=1 if c["adjudicates"] else 0.6)
    # the actual displacement t0 -> +80
    ax.plot([c["margin_t0"], c["margin_80"]], [c["p_t0"], c["p_80"]],
            linestyle=":", linewidth=0.8, color=col, alpha=0.5, zorder=1)
    ax.scatter([c["margin_t0"]], [c["p_t0"]], facecolors="none", edgecolors=col,
               s=18, linewidths=0.8, zorder=2)
    if c["argmax_is_answer_80"]:
        mk = "o" if c["wash"] == "w1" else "s"
        ax.scatter([c["margin_80"]], [c["p_80"]], marker=mk, color=col, s=24,
                   zorder=3, alpha=0.95, edgecolors="w", linewidths=0.4)
    else:
        ax.scatter([c["margin_80"]], [c["p_80"]], marker="x", color=col, s=30,
                   zorder=4, linewidths=1.3)
ax.axvline(FLIP_ZONE, color="k", linewidth=0.8, linestyle="-.")
ax.text(FLIP_ZONE + 0.004, 0.9, "0.05 flip zone", fontsize=7, rotation=90,
        va="top")
handles = [
    plt.Line2D([], [], linestyle="--", color="k", linewidth=1.1,
               label="t=0 temperature vertical (margin pinned)"),
    plt.Line2D([], [], linestyle=":", color="k", linewidth=0.8,
               label="actual wash path t0 -> +80"),
    plt.Line2D([], [], marker="o", color="w", markeredgecolor="k",
               label="t=0 (T=1)"),
]
for b in ["fact", "ctrl", "tmpl", "near"]:
    handles.append(
        plt.Line2D([], [], marker="o", color=bcols[b], linestyle="",
                   label=f"{b} zombie at +80 (o/s = w1/w2)")
    )
handles.append(
    plt.Line2D([], [], marker="x", color="k", linestyle="",
               label="argmax FLIPPED at +80 — off-vertical IN KIND")
)
ax.legend(handles=handles, fontsize=6.5, loc="upper right")
ax.set_xlabel("argmax margin (sigma)")
ax.set_ylabel("p(answer)")
ax.set_title(
    "READ 2 — natural +80 zombies vs the temperature family\n"
    "(dashed verticals: ALL temperatures available to each probe; "
    f"floor band ±{floor:.3f})",
    fontsize=9,
)
xmin = min(min(c["margin_t0"], c["margin_80"]) for c in cloud)
xmax = max(max(c["margin_t0"], c["margin_80"]) for c in cloud)
pad = 0.05 * (xmax - xmin + 0.1)
ax.set_xlim(max(0, xmin - pad), xmax + pad)
ax.set_ylim(0, 1.0)

ax = axes[1]
ax.hist(dsigned, bins=12, color="#1f77b4", edgecolor="k", linewidth=0.4,
        alpha=0.85)
ax.axvline(0, color="k", linewidth=1)
ax.axvline(-floor, color="#d62728", linestyle="--", linewidth=1.2,
           label=f"noise floor ±{floor:.4f}")
ax.axvline(+floor, color="#d62728", linestyle="--", linewidth=1.2)
ax.axvline(med_signed, color="#ff7f0e", linewidth=1.4,
           label=f"median D = {med_signed:+.4f}")
ax.set_xlabel("D = margin(+80) - margin(t0)  (sigma; left = thinner than t0)")
ax.set_ylabel("zombie points")
ax.set_title(
    f"departure from the t=0 verticals (n={len(adj_cloud)} adjudicated)\n"
    f"median |D| = {med_dep:.4f} vs floor {floor:.4f} -> {r2_verdict}",
    fontsize=9,
)
ax.legend(fontsize=7)
fig.suptitle(
    "E232 READ 2 — THE TEMPERATURE LENS (W034): can temperature write these "
    "zombies? | verdict: " + r2_verdict,
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG_LENS)
plt.close(fig)

M["figures"] = {
    "zombie_lag_population": FIG_LAG,
    "temperature_lens_cloud": FIG_LENS,
}
write_metrics("s4-figures")

# ---------------------------------------------------------------------------
# STAGE 5 — honesty, provenance, close
# ---------------------------------------------------------------------------
M["honesty_reflex"] = {
    "e208_distinction": "e208's margin object was the FACT-EDGE over the wash "
    "band on the tiny-net organisms (a cross-organism class separator, scoped "
    "to NATURAL-STEP organisms after e209/e210); THIS cell's object is the "
    "NEXT-TOKEN ARGMAX MARGIN at the answer position in sigma units vs "
    "arithmetic noise, per probe, on the committed 124M archive — a different "
    "ruler that shares only the word 'margin'",
    "classification_provenance": "the P-FIRST classes are e230's COMMITTED "
    "classification, read from its metrics.json, never re-derived here; the "
    "journal-recomputed margin crossings agree with e230's committed m_cross "
    "fields on every record (gate G_E230X)",
    "near_quantized": "near is n=3 — every statistic on it is quantized; "
    "co-report only, never adjudicates",
    "grid_coarseness": "the state grid is coarse (w1 2/10/50/80, w2 10/50/80); "
    "crossing states and lag lengths are quantized to it; 'resolved at 80' "
    "means crossed BY the last observed state — the true crossing could be "
    "anywhere in (50, 80]",
    "snapshot_definition": "standing/resolved is a +80 SNAPSHOT call (T209's "
    "registered definition); a probe that crossed earlier and recovered would "
    "count standing (flagged 'recovered', disclosed in the table)",
    "floor_conservatism": "the noise floor is itself wash-induced movement "
    "(w1@2 minimal dose), not measurement noise — the journals are single "
    "deterministic evals and e182 committed p only (no margin replay floor "
    "exists); this biases the read TOWARD TEMPERATURE-LIKE, i.e., against the "
    "RESTRUCTURED bar — the conservative direction",
    "maximal_margin_null": "under the registered analytic form, temperature "
    "predicts the probe's FULLEST margin (t0's) at every fallen p — the null "
    "is maximal-margin at matched p, so left-of-vertical departures (natural "
    "margins thinner than t0) are as fatal to the flattening story as "
    "right-of-vertical ones; registered before compute",
    "shared_verticals": "a probe zombie under both washes contributes two "
    "cloud points sharing one t0 vertical (each wash is a separate death); "
    "per-wash medians co-reported",
    "recovered_snapshot_zombies": "the +80 standing-zombie SNAPSHOT set is "
    "slightly larger than the registered P-FIRST-scoped cloud: 3 "
    "adjudicated probes (all TOGETHER-class: margin and p crossed together "
    "at 50, margin dipped below 0.05, then PARTIALLY RECOVERED above it by "
    "+80) — crossing is an absorbing EVENT, not an absorbing STATE; "
    "disclosed in gate G_ZSUBSET; the full-snapshot sensitivity leaves the "
    "verdict unchanged",
    "wrong_choice_survivors": "7 of the 33 adjudicated standing zombies (and "
    "2 of the 4 near co-report points) no longer hold the ANSWER as argmax "
    "at +80: a fat margin survives on a DIFFERENT token while p(answer) "
    "stays halved — the commitment outlives the belief but has CHANGED "
    "HANDS; for the temperature lens these are off-vertical IN KIND "
    "(temperature preserves the argmax for every T — no temperature can "
    "write them at any margin); the 26 answer-holders are off-vertical in "
    "degree (median departure reported separately); both populations "
    "co-reported, the registered bar pools them as registered",
    "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 replay "
    "of e182's GPU original, wash 2 is GPU fp32 — the archive's device "
    "asymmetry, inherited, disclosed",
    "one_organism": "ONE organism's commitment spectrum (the shared pristine "
    "124M); nothing here speaks cross-organism",
    "guarantees_nothing": "nothing here is guaranteed — T209/W034's registered "
    "branches were written before this pass; no retrofit",
}
M["provenance"]["versions"] = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
}
M["provenance"]["e230_status_at_read"] = m230.get("status")
M["timing"] = {"started": M["date"], "finished": now_utc()}
M["trims"] = []
M["deviations"] = [
    "DESK-ONLY per the dispatch: no model loads, no GPU, no wash; pure join "
    "of committed records + analytic arithmetic on the t=0 summaries",
    "the temperature family is evaluated through its REGISTERED analytic "
    "form (vertical lines; margin_sigma exactly invariant) rather than by "
    "materializing softmax(l/T) curves — the journals commit per-probe "
    "summaries (p, margin, sigma, top1/top2 ids), not full logits; the "
    "registered read (cloud position relative to the verticals) is complete "
    "without p(T) curves, and the exact-p T* exists for every zombie (gate "
    "G_ZSUBSET) — disclosed as a scope note, not a bar change",
    "s6 CO-REPORT PASS (post-registration, bars untouched): the registered "
    "compute falsified this file's own G_ZSUBSET containment assumption "
    "(crossing is an absorbing event, not state) and surfaced the "
    "argmax_is_answer column; the argmax hold/flip decomposition and the "
    "full-snapshot / holders-only sensitivities were added AFTER the "
    "adjudication, are labeled co-reports everywhere, and change no bar",
]
M["all_gates_pass"] = all(
    bool(M["gates"][g]["pass"]) for g in M["gates"]
)
M["status"] = "DONE"
write_metrics("s6-coreports-done")

print("e232 done. verdicts:", r1_verdict, "|", r2_verdict)
print("read1 cells:", cell_txt)
print(
    "read2: median|D|=%.4f signed=%.4f floor=%.4f (n=%d, left %d / right %d)"
    % (med_dep, med_signed, floor, len(adj_cloud),
       int(np.sum(dsigned < 0)), int(np.sum(dsigned > 0)))
)
