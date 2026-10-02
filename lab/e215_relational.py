"""E215 — THE RELATIONAL SIGNATURE (T187's named follow-up: the wash's own
sorting key).

WHY: T187/e214 found the wash re-orders the probes BY RELATION (lang holds,
cap/cur collapses; founders/unique-anchor hold, products collapse; baseline
p irrelevant) — but that was a TEXTURE read off four battery-level relation
blocks. THE OPEN QUESTION THIS CELL REGISTERS: WHICH relation-types hold vs
collapse, and WHAT IS THE WASH'S SORTING KEY — the family label itself (the
'relational' reading: the wash sorts by RELATION TYPE) or a cheap structural
statistic (frequency/length; the 'exposure' reading: the sorting key is
statistical, not semantic)? This is W028's law pushed one floor down: the
erosion ORDER was shown to replicate (e214's cross-wash half); the next
question is what that order is MADE OF.

THE ARCHIVE (committed; this cell is a DESK cell on per-probe records with
at most TINY re-probe bursts):
  * the per-probe p(answer) records of runs/e214/metrics.json (curves:
    wash-1 states +{2,10,50,80}, wash-2 states +{10,50,80}; every battery)
    and runs/e214/journal.json (the shared pristine t=0 record) — e214
    certified every one of them against the e182c/e182c2 originals at
    re-probe dp <= 3.3e-06.
  * the batteries themselves rebuilt VERBATIM by module import (the e214
    convention) for probe METADATA (subject, answer, prompt ids, relation).
  * the committed corpus file data/input.txt -> the frozen wash corpus
    (rebuild + assert, the e214 G_CORPUS chain) for the frequency features.

THE CELL (three parts):
  (1) THE TYPOLOGY — a hand-registered assignment of EVERY probe in the
      archive (~54) to a relation family, following the dispatch's named
      families: the fact battery's two known families (lang vs cap/cur),
      the controls (founders/unique-anchor vs products), the near-related
      (capital-of US states), the template (reversed capital). Documented
      probe-by-probe (metrics.typology.assignment). Extended with cheap
      structural features per probe: the answer's token length; the
      subject's token length; the relation-cue frequency in the wash
      corpus; one-hot family.
  (2) THE HOLD/COLLAPSE SPLIT — per probe the hold ratio p_t/p_0 at the
      deepest shared state (+80) on BOTH washes; per family the median
      hold ratio per wash (HOLD >= 0.5 both washes / COLLAPSE < 0.3 both
      washes / MIXED); the per-probe CROSS-WASH AGREEMENT (the sorting
      key's reliability: Spearman(hr_w1, hr_w2) + the hold-class
      confusion).
  (3) THE PREDICTOR LADDER — which feature predicts the hold ratio: the
      family label (between-family separation vs within-family spread),
      the relation's corpus frequency, the token structure? A rank test
      per feature (Spearman + the family-partial); the honest "what is
      the wash's sorting key".

REGISTERED BARS (frozen VERBATIM from the dispatch brief, QUEUE row
DISPATCHED 15:07Z / commit 06d38d8, BEFORE any compute; adjudicate against
exactly this; no bar shopping):
  - FAMILY-IS-THE-KEY: "fires if the family label predicts the hold ratio
    (between-family separation >= the within-family spread by 2x) and no
    structural feature adds materially — the wash sorts by RELATION TYPE;
    the relational signature is the law's object."
  - STRUCTURE-BEATS-FAMILY: "fires if a structural feature
    (frequency/length) outpredicts the family — the 'relational' reading
    reduces to exposure; the sorting key is statistical, not semantic."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * DATA = the committed per-probe records (e214 metrics curves + journal
    t=0); the runtime re-probes here CERTIFY provenance, they are not the
    data source. Deepest state = +80 (the deepest state present on BOTH
    washes). hold ratio hr_w(probe) = p_w(+80) / p_0. Primary outcome
    Y = hr_mean = (hr_w1 + hr_w2)/2; per-wash co-reports beside it.
  * TYPOLOGY (the bar's "family label", 6 families, hand-registered
    following the dispatch's naming): lang (fact/lang); cap-cur
    (fact/cap + fact/cur — e214's one collapse family); founder-anchor
    (ctrl/found, incl. the ONE unique-anchor probe Wikipedia, flagged);
    product (ctrl/make); near-uscap (near); rev-capital (tmpl). The
    finer 7-relation split (cap vs cur separate) co-reported.
  * FAMILY SEPARATION (the bar's clause "between-family separation >= the
    within-family spread by 2x"): one-way ANOVA decomposition of Y on the
    6 families; SEPARATION := SS_between / SS_within >= 2 (the plain
    variance-ratio reading of the bar's words: between-variance at least
    twice the within-variance). Co-reported, never adjudicated: the
    mean-square F, eta^2 = SS_b/SS_tot, the std ratio, the same table on
    hr_w1 / hr_w2 / the 7-relation split.
  * STRUCTURAL FEATURES (adjudicable — the dispatch's "frequency/length"
    set): (s1) relation-cue frequency = the count of the relation's
    frozen cue words in the FROZEN filtered wash corpus (capital;
    currency; speak+language; founded; made), per 1e6 chars, a
    relation-level feature (every probe of a family shares it);
    (s2) answer token length (len(tok.encode(" "+answer))); (s3) subject
    token length. COMPETITORS (co-reported, never adjudicated): prompt
    token length; p0 (the baseline-strength hypothesis e214 killed,
    quantified here); the answer-token count in the exact training
    stream (expected identically 0 by the contamination gates — the
    degenerate feature, disclosed).
  * RANK TEST per feature: rho = Spearman(feature, Y); rho2 = rho^2;
    PARTIAL = Spearman of the family-mean-residualized Y vs the
    family-mean-residualized feature (relation-level features have zero
    within-family variance -> PARTIAL undefined, structurally null,
    disclosed).
  * "ADDS MATERIALLY" := an adjudicable feature with |PARTIAL| >= 0.30.
  * "OUTPREDICTS THE FAMILY" := an adjudicable feature with rho2 > eta2
    (both are fractions of Y's variance, the monotone-invariant form for
    the rank side).
  * ADJUDICATION ORDER: gates ok -> FAMILY-IS-THE-KEY iff (SEPARATION)
    and (no material addition); else STRUCTURE-BEATS-FAMILY iff
    (OUTPREDICTS); else GRADED. Gated on G_STATES/G_BATT/G_CORPUS/G_ENV.
  * TYPE-LEVEL SPLIT (descriptive, co-report): family HOLDS iff median
    hr >= 0.5 on BOTH washes; COLLAPSES iff median hr < 0.3 on BOTH
    washes; else MIXED.
  * RELIABILITY (co-report): Spearman(hr_w1, hr_w2) pooled + per battery;
    hold-class confusion (HOLD hr >= 0.5 / COLLAPSE hr < 0.3 / MID).

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_STATES — the two deepest shared checkpoints inventoried (sizes/
    mtimes cross-checked against e214's committed inventory) and
    RE-PROBED: all four batteries probed on the LOADED w1 +80 and w2 +80,
    compared per-probe to e214's committed records within 0.005 (expected
    ~1e-6: same fp32 weights, same CPU fp32 probe path). THE TINY
    RE-PROBE BURST — two loads, nothing else re-run; the intermediate
    states (+2/+10/+50) enter the analysis from the COMMITTED records
    only (e214 already certified them against the e182c/e182c2
    originals).
  * G_BATT — the four batteries rebuilt VERBATIM (module import) must
    reproduce e214's committed t=0 journal per-probe (same kept sets,
    |dp| <= 0.010; expected ~1e-6).
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (the e214 chain; the filtered text is the
    substrate of the frequency features).
  * G_ENV — the owner envelope: CPU-only (zero GPU calls), torch threads
    <= 4, load checks before launch and between bursts, tiny bursts,
    decisive.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. n=2 washes is texture, not law; wash 1 is a CPU fp32 replay /
wash 2 GPU fp32 (inherited, disclosed). THE TYPOLOGY IS HAND-REGISTERED
and NOT BLIND to the outcome: the families were named by T187's reading
of this very archive's relation blocks — the family label's win is
partly built-in, and the honest defense is only that the GROUPING comes
from the instrument's design (the frozen battery pools), not from the
wash data; disclosed in every table. UNBALANCED n's: lang 10, cap-cur
10, founder-anchor 5, product 7, near-uscap 3 (n=3!), rev-capital 19.
THE FREQUENCY ARM IS NEAR-DEGENERATE IN THIS CORPUS: the wash corpus is
filtered Shakespeare — the relation cues occur 0-13 times (currency 0,
founded 0, capital 4, language 13) and EVERY probe's subject/answer
strings are ABSENT from the wash corpus by the contamination gates —
in-wash exposure is identically zero at the probe level (the e182c
design's own property); the exposure hypothesis is only testable at the
relation-cue level here, disclosed. THE ANSWER-TOKEN-LENGTH ARM IS
STRUCTURALLY NULL: the e182 gate kept only single-token answers — the
instrument froze the feature at 1 (a Rule-12 disclosure, not a finding).
hr is a ratio on p0 as low as 0.49; deep-state floors (p 0.05-0.2)
compress ratios; hr and p80 co-reported.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and between bursts; TINY bursts
(two checkpoint loads); ~230 CPU fp32 forwards total (battery screening
+ t0 + the two re-probe bursts); no wash, no bank ppl, no NOTES/
THINKING/QUEUE/STATE edits (dispatch). Progressive PARTIAL metrics after
every phase (the standing disruption rule).

PROVENANCE: the per-probe records, the two-wash archive and the census
convention are e214's (this cell reads its committed outputs and re-
certifies the two states its outcome reads at depth); the batteries,
corpus and organism are lab/e182c_forgetting_control.py + lab/
e182c2_template.py VERBATIM via module import. Builds on: T187/e214
(the named follow-up; the relation texture this cell interrogates), W028
(the shape/height law the erosion order serves), T185/e213 (the state
function), T183/e182c2 + T149/e182c + T123/e182 (the archive). NEW: the
hand-registered typology with the probe-by-probe assignment; the
per-probe hold ratios on both washes; the hold/collapse split table; the
cross-wash reliability of the sorting key; the predictor ladder (family
ANOVA vs frequency/length rank tests, with family-partials).

Run:  cd lab && python e215_relational.py   (E215_SMOKE=1: t=0 + the
      w2 +80 burst only, own smoke dir, nothing adjudicated)
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

SMOKE = os.environ.get("E215_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e215_smoke" if SMOKE else "e215"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants (registered BEFORE compute) ---------------------
DEEPEST = 80                    # the deepest state present on BOTH washes
DEPTH_COURSE = (10, 50, 80)     # co-report: the family medians' depth course
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
REPROBE_STATES = {"w1": [80] if not SMOKE else [],
                  "w2": [80]}    # THE tiny re-probe burst (both washes, +80)
W1_ARCH = {80: common.REPO / "runs" / "checkpoints" / "e182c_s80.pt"}
W2_ARCH = {80: common.REPO / "runs" / "checkpoints" / "e182c2_fresh_s80.pt"}

# ---- the hand-registered typology (frozen BEFORE compute) ----------------------
# family6 = the bar's "family label" (the dispatch's named families);
# relation7 = the finer module-relation split (cap vs cur separate), co-reported.
FAMILY6 = {                       # (battery, module relation) -> family
    ("fact", "lang"): "lang",
    ("fact", "cap"): "cap-cur",
    ("fact", "cur"): "cap-cur",
    ("ctrl", "found"): "founder-anchor",
    ("ctrl", "make"): "product",
    ("near", "near"): "near-uscap",
    ("tmpl", "tmpl"): "rev-capital",
}
FAMILY6_ORDER = ["lang", "cap-cur", "founder-anchor", "product",
                 "near-uscap", "rev-capital"]
FAMILY6_DEF = {
    "lang": "fact/lang — 'People in {c} speak {a}' (official-language cloze)",
    "cap-cur": "fact/cap + fact/cur — capital-of + currency-of cloze "
               "(e214's ONE collapse family)",
    "founder-anchor": "ctrl/found — '{c} is called {a}' founder/unique-"
                      "anchor company cloze (incl. ONE unique-anchor probe, "
                      "Wikipedia, flagged in the assignment)",
    "product": "ctrl/make — '{c} is called {a}' product-of-company cloze",
    "near-uscap": "near — the fact battery's capital template over US states "
                  "(disjoint entities; n=3)",
    "rev-capital": "tmpl — the REVERSED capital form 'The state whose capital "
                   "is {a} is {c}' (city->state)",
}
# the relation-cue words (the frequency feature's frozen surface forms; the
# arbitrary-choice disclosure lives in the honesty block)
TYPE_CUES = {
    "lang": ["speak", "language"],
    "cap-cur": ["capital", "currency"],
    "founder-anchor": ["founded"],
    "product": ["made"],
    "near-uscap": ["capital"],
    "rev-capital": ["capital"],
}
UNIQUE_ANCHOR_PROBES = {"The online encyclopedia that anyone can edit->Wikipedia"}

# ---- registered bar constants (frozen) ----------------------------------------
SEP_BAR = 2.0          # "between-family separation >= the within-family spread
                        #  by 2x" -> SS_between / SS_within >= 2 (variance ratio)
MATERIALITY_BAR = 0.30  # "adds materially" -> |family-partial Spearman| >= 0.30
HOLD_BAR, COLLAPSE_BAR = 0.5, 0.3   # the type-split descriptive bands

# ---- registered verification tolerances (frozen) -------------------------------
TOL_PROBE_DP = 0.010   # per-probe t=0 dp vs e214's committed journal
TOL_STATE_DP = 0.005   # per-probe dp on the LOADED +80 states vs e214 records

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "FAMILY-IS-THE-KEY": "fires if the family label predicts the hold "
            "ratio (between-family separation >= the within-family spread "
            "by 2x) and no structural feature adds materially — the wash "
            "sorts by RELATION TYPE; the relational signature is the law's "
            "object.",
        "STRUCTURE-BEATS-FAMILY": "fires if a structural feature "
            "(frequency/length) outpredicts the family — the 'relational' "
            "reading reduces to exposure; the sorting key is statistical, "
            "not semantic.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "operationalizations": (
        "outcome Y = per-probe hold ratio hr_mean = (p_w1(+80)/p_0 + "
        "p_w2(+80)/p_0)/2 on the committed e214 per-probe records (deepest "
        "SHARED state; the runtime re-probes certify provenance only); "
        "family = the hand-registered 6-family typology (lang / cap-cur / "
        "founder-anchor / product / near-uscap / rev-capital; the 7-relation "
        "split co-reported); SEPARATION := SS_between/SS_within >= 2 on the "
        "one-way ANOVA of Y on family (F, eta2, std-ratio co-reported); "
        "ADDS-MATERIALLY := |family-partial Spearman| >= 0.30 for an "
        "adjudicable structural feature (relation-cue frequency / answer "
        "token length / subject token length); OUTPREDICTS := rho2 > eta2 "
        "for an adjudicable structural feature; FAMILY-IS-THE-KEY iff "
        "SEPARATION and not ADDS-MATERIALLY; else STRUCTURE-BEATS-FAMILY "
        "iff OUTPREDICTS; else GRADED; adjudication gated on G_STATES/"
        "G_BATT/G_CORPUS/G_ENV; co-reports (never adjudicated): the "
        "type-split bands (HOLD median hr >= 0.5 both washes / COLLAPSE "
        "< 0.3 both washes / MIXED), the cross-wash reliability (Spearman"
        "(hr_w1, hr_w2) + hold-class confusion), the depth course (+10/"
        "+50/+80), the per-wash ladders, the competitor features (prompt "
        "length, p0, the identically-zero stream count)"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (QUEUE "
                     "row DISPATCHED 15:07Z, commit 06d38d8) BEFORE any "
                     "compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "DESK cell: the hold ratios are computed on e214's COMMITTED per-probe "
    "records (metrics curves + journal t=0), not on fresh probes — the "
    "runtime re-probes (two +80 bursts, one per wash) CERTIFY provenance "
    "(the e214 convention, tol 0.005); the intermediate states enter from "
    "the committed records only.",
    "The typology is HAND-REGISTERED and not blind to the outcome (the "
    "families were named by T187's reading of this archive's relation "
    "blocks); the grouping itself follows the instrument's design (the "
    "frozen battery pools + the dispatch's naming), not the wash data — "
    "disclosed in the honesty block and every table.",
    "The relation-cue frequency feature uses FROZEN cue words per family "
    "(capital; currency; speak+language; founded; made) — an arbitrary "
    "surface choice, disclosed; per-word counts reported so the table is "
    "transparent.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "PLOT-ONLY FIX: the first full pass wrote DONE metrics and crashed in "
    "the figure (a None-precedence bug in the ladder annotation line); the "
    "figure was regenerated by re-running the cell — deterministic (all "
    "re-probe gates read dp 0.0 both passes) and zero metric change.",
    "Smoke mode: t=0 + the w2 +80 burst only, own smoke dir, nothing "
    "adjudicated.",
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


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho = Pearson on average ranks (exact under ties)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(avg_ranks(xs), avg_ranks(ys))


def median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def one_way(y: list[float], groups: list[str]) -> dict:
    """One-way ANOVA decomposition (pure python). Returns the plain variance
    ratio SS_b/SS_w (the frozen SEPARATION statistic), F, eta2, std ratio."""
    n = len(y)
    fams = sorted(set(groups))
    k = len(fams)
    grand = sum(y) / n
    by: dict[str, list[float]] = {f: [] for f in fams}
    for yi, g in zip(y, groups):
        by[g].append(yi)
    means = {f: sum(v) / len(v) for f, v in by.items()}
    ss_b = sum(len(v) * (means[f] - grand) ** 2 for f, v in by.items())
    ss_w = sum((x - means[g]) ** 2 for x, g in zip(y, groups))
    ss_t = sum((x - grand) ** 2 for x in y)
    f_stat = (ss_b / (k - 1)) / (ss_w / (n - k)) if ss_w > 0 else None
    var_b, var_w = ss_b / n, ss_w / n
    return {
        "n": n, "k": k, "grand_mean": grand,
        "family_means": means, "family_ns": {f: len(by[f]) for f in fams},
        "family_medians": {f: median(by[f]) for f in fams},
        "family_vars": {f: (sum((x - means[f]) ** 2 for x in by[f])
                            / len(by[f])) for f in fams},
        "SS_between": ss_b, "SS_within": ss_w, "SS_total": ss_t,
        "var_ratio_SSb_SSw": (ss_b / ss_w) if ss_w > 0 else None,
        "F": f_stat,
        "eta2": (ss_b / ss_t) if ss_t > 0 else None,
        "std_ratio": ((var_b ** 0.5) / (var_w ** 0.5)) if var_w > 0 else None,
    }


def partial_spearman(feat: list[float], y: list[float],
                     groups: list[str]) -> tuple[float | None, str]:
    """Family-partial Spearman: residualize BOTH (y and the feature) on
    their own family means; a relation-level feature (zero within-family
    variance) returns None with the structural note."""
    keys = set(groups)
    ym: dict[str, list[float]] = {f: [] for f in keys}
    fm: dict[str, list[float]] = {f: [] for f in keys}
    for yi, fi, g in zip(y, feat, groups):
        ym[g].append(yi)
        fm[g].append(fi)
    ymean = {f: sum(v) / len(v) for f, v in ym.items()}
    fmean = {f: sum(v) / len(v) for f, v in fm.items()}
    ry = [yi - ymean[g] for yi, g in zip(y, groups)]
    rf = [fi - fmean[g] for fi, g in zip(feat, groups)]

    def _const_within(g: str) -> bool:
        vals = [fi for fi, gg in zip(feat, groups) if gg == g]
        return max(vals) - min(vals) <= 1e-12 * max(1.0, abs(vals[0]))

    if all(_const_within(g) for g in keys):
        return None, ("zero within-family variance (relation-level feature) "
                      "— structurally unidentifiable inside family")
    return spearman(rf, ry), "residualized on family means"


# ------------------------------------------------------------------ plot

FAM_COLS = {"lang": "tab:green", "cap-cur": "tab:red",
            "founder-anchor": "tab:blue", "product": "tab:purple",
            "near-uscap": "tab:olive", "rev-capital": "darkorange"}


def make_plot(rd, assign, fam_split, ladder, rel, adj, anova):
    """THE FIGURE: (a) the typology hold/collapse split (per-family hr
    strips, the 0.5/0.3 bands); (b) the predictor ladder (rho2 + |partial|
    per feature vs the family's eta2); (c) the cross-wash reliability
    scatter; (d) the typology table + verdict + gates."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) the hold/collapse split
    ax = axes[0, 0]
    ax.axhspan(HOLD_BAR, 1.05, facecolor="green", alpha=0.08)
    ax.axhspan(0.0, COLLAPSE_BAR, facecolor="red", alpha=0.08)
    ax.axhline(HOLD_BAR, color="k", ls="--", lw=1.2)
    ax.axhline(COLLAPSE_BAR, color="k", ls="--", lw=1.2)
    ax.text(5.52, HOLD_BAR + 0.012, "HOLD >= 0.5", fontsize=7,
            family="monospace")
    ax.text(5.52, COLLAPSE_BAR - 0.045, "COLLAPSE < 0.3", fontsize=7,
            family="monospace")
    for xi, fam in enumerate(FAMILY6_ORDER):
        pts1 = [r["hr_w1"] for r in assign if r["family6"] == fam]
        pts2 = [r["hr_w2"] for r in assign if r["family6"] == fam]
        ax.scatter([xi + 0.10] * len(pts1), pts1, s=26, marker="o",
                   color=FAM_COLS[fam], alpha=0.75,
                   label="wash 1" if xi == 0 else None)
        ax.scatter([xi - 0.10] * len(pts2), pts2, s=26, marker="s",
                   facecolors="none", edgecolors=FAM_COLS[fam], linewidths=1.4,
                   alpha=0.9, label="wash 2" if xi == 0 else None)
        m1, m2 = fam_split[fam]["median_hr_w1"], fam_split[fam]["median_hr_w2"]
        ax.plot([xi - 0.22, xi + 0.22], [m1, m1], color="k", lw=1.6)
        ax.plot([xi - 0.22, xi + 0.22], [m2, m2], color="k", lw=1.6,
                ls=":")
        ax.annotate(fam_split[fam]["verdict"], (xi, 1.012),
                    ha="center", fontsize=6.6, family="monospace",
                    color="darkgreen" if fam_split[fam]["verdict"] == "HOLDS"
                    else "darkred" if fam_split[fam]["verdict"] == "COLLAPSES"
                    else "dimgray")
    ax.set_xticks(range(len(FAMILY6_ORDER)))
    ax.set_xticklabels([f"{f}\n(n={fam_split[f]['n']})" for f in FAMILY6_ORDER],
                       fontsize=7.4)
    ax.set_ylabel("HOLD RATIO  p(+80)/p_0")
    ax.set_ylim(-0.04, 1.10)
    ax.legend(fontsize=7, loc="center left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(1) THE HOLD/COLLAPSE SPLIT — the wash sorts the archive "
                 "by relation (o w1 / [] w2; — median w1, : median w2)",
                 fontsize=9.5)

    # (0,1) the predictor ladder
    ax = axes[0, 1]
    rows = ([("family (eta2)", anova["eta2"], None, True)]
            + [(f["label"], f["rho2"],
                None if f["partial_rho"] is None else abs(f["partial_rho"]),
                f["adjudicable"]) for f in ladder["features"]])
    labels = [r[0] for r in rows]
    xs = range(len(rows))
    for i, (lab, r2, pr, adjc) in enumerate(rows):
        ax.bar(i - 0.17, 0 if r2 is None else r2, width=0.32,
               color="tab:blue" if adjc else "lightgray",
               edgecolor="k", lw=0.6,
               hatch="" if adjc else "//")
        if pr is not None:
            ax.bar(i + 0.17, pr, width=0.32, color="tab:orange",
                   edgecolor="k", lw=0.6)
        elif rows[i][3] and i > 0:
            ax.text(i + 0.17, 0.012, "n/a", rotation=90, fontsize=6,
                    ha="center", va="bottom", color="dimgray")
    ax.axhline(anova["eta2"], color="tab:blue", ls="--", lw=1.4,
               label=f"family eta2 = {anova['eta2']:.3f}")
    ax.axhline(MATERIALITY_BAR, color="tab:orange", ls=":", lw=1.4,
               label=f"materiality |partial| = {MATERIALITY_BAR}")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.2)
    ax.set_ylabel("PREDICTION STRENGTH on hr_mean")
    ax.grid(alpha=0.25, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
               plt.Rectangle((0, 0), 1, 1, color="lightgray", hatch="//")]
    ax.legend(handles + [plt.Line2D([0], [0], color="tab:blue", ls="--"),
                         plt.Line2D([0], [0], color="tab:orange", ls=":")],
              ["rho^2 raw (blue=adjudicable)", "|family-partial rho|",
               "competitors (co-reported)", "family eta2", "materiality bar"],
              fontsize=6.4, loc="upper right")
    ax.set_title("(3) THE PREDICTOR LADDER — what is the wash's sorting key?",
                 fontsize=9.5)

    # (1,0) the cross-wash reliability scatter
    ax = axes[1, 0]
    for fam in FAMILY6_ORDER:
        xs_ = [r["hr_w1"] for r in assign if r["family6"] == fam]
        ys_ = [r["hr_w2"] for r in assign if r["family6"] == fam]
        ax.scatter(xs_, ys_, s=28, color=FAM_COLS[fam], alpha=0.8,
                   label=f"{fam} (n={fam_split[fam]['n']})")
    lim = 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=0.9, ls="--", alpha=0.6)
    ax.axhline(HOLD_BAR, color="k", ls=":", lw=0.9)
    ax.axvline(HOLD_BAR, color="k", ls=":", lw=0.9)
    ax.set_xlabel("hold ratio, wash 1  p(+80)/p_0")
    ax.set_ylabel("hold ratio, wash 2  p(+80)/p_0")
    ax.set_xlim(-0.03, lim)
    ax.set_ylim(-0.03, lim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title(f"(2) THE SORTING KEY'S RELIABILITY — Spearman(hr_w1, hr_w2) "
                 f"= {rel['spearman_pooled']:.3f} (n=54); hold-class agree "
                 f"{rel['class_agreement']:.0%}", fontsize=9.5)

    # (1,1) the typology table + verdict
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE TYPOLOGY TABLE (median hold ratio p(+80)/p_0 per "
            "wash; HOLD >= 0.5 / COLLAPSE < 0.3 on both):", fontsize=8.8,
            va="top", family="monospace", weight="bold")
    y -= 0.030
    ax.text(0.02, y, "  family             n   p0_med  hr_w1  hr_w2   verdict"
                     "        cue/1M", fontsize=7.0, va="top",
            family="monospace")
    y -= 0.021
    for fam in FAMILY6_ORDER:
        s = fam_split[fam]
        cue = ladder["cue_freq_by_family"][fam]
        ax.text(0.02, y, f"  {fam:17s} {s['n']:3d}  {s['median_p0']:6.3f}"
                         f" {s['median_hr_w1']:6.3f} {s['median_hr_w2']:6.3f}"
                         f"   {s['verdict']:11s}"
                         f" {cue if cue is not None else float('nan'):8.2f}",
                fontsize=7.0, va="top", family="monospace",
                color="darkgreen" if s["verdict"] == "HOLDS"
                else "darkred" if s["verdict"] == "COLLAPSES" else "black")
        y -= 0.0195
    y -= 0.008
    for line in (
        f"LADDER: family eta2 {anova['eta2']:.3f} | SSb/SSw "
        f"{anova['var_ratio_SSb_SSw']:.2f} (bar {SEP_BAR:g}) | F "
        f"{anova['F']:.2f} | best structural rho2 "
        f"{max((f['rho2'] or 0) for f in ladder['features']):.3f} | best "
        f"|partial| "
        f"{max((0 if f['partial_rho'] is None else abs(f['partial_rho'])) for f in ladder['features']):.3f}"
        f" (bar {MATERIALITY_BAR})",
    ):
        ax.text(0.02, y, line, fontsize=7.0, va="top", family="monospace")
        y -= 0.020
    y -= 0.006
    ax.text(0.02, y, f"E215 VERDICT: {adj['bars']['verdict']}", fontsize=9.6,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(adj["bars"]["clause"], width=94,
                            break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.0195
    y -= 0.004
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in adj["gates_summary"].items()), fontsize=7.0, va="top",
        family="monospace")

    fig.suptitle("E215 — THE RELATIONAL SIGNATURE (T187's follow-up): the "
                 "wash's own sorting key -> " + adj["bars"]["verdict"],
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "relational_signature.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E215 — THE RELATIONAL SIGNATURE (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e215_relational_signature",
        "phase": "desk+eval on committed per-probe records (tiny re-probe "
                 "bursts for provenance)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T187's named follow-up: the wash sorts the probes BY "
                     "RELATION — which relation-types hold vs collapse, and "
                     "WHAT IS THE SORTING KEY: the family label (relational) "
                     "or a structural statistic (frequency/length)?"),
        "builds_on": ["T187 / e214 (the named follow-up; the relation texture"
                      " this cell interrogates + the committed per-probe "
                      "records it reads)",
                      "W028 (the shape/height law the erosion order serves)",
                      "T185 / e213 (the state function; the archive's design)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "two-wash 124M archive)"],
        "whats_new": ["the hand-registered typology: every probe assigned to "
                      "a relation family (documented probe-by-probe)",
                      "the per-probe hold ratios p(+80)/p_0 on BOTH washes; "
                      "the hold/collapse split table with the type verdicts",
                      "the cross-wash reliability of the sorting key "
                      "(Spearman(hr_w1, hr_w2) + hold-class confusion)",
                      "the predictor ladder: family ANOVA (between/within) "
                      "vs frequency/length rank tests with family-partials",
                      "the three-bar adjudication (FAMILY-IS-THE-KEY / "
                      "STRUCTURE-BEATS-FAMILY / GRADED)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    e214_path = common.REPO / "runs" / "e214" / "metrics.json"
    e214_jp = common.REPO / "runs" / "e214" / "journal.json"
    for p in (e214_path, e214_jp, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e214m = json.loads(e214_path.read_text(encoding="utf-8"))
    e214j = json.loads(e214_jp.read_text(encoding="utf-8"))["states"]
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    t0j = [r for r in e214j if r["wash"] == "t0"][0]
    w1j = {r["step"]: r for r in e214j if r["wash"] == "w1"}
    w2j = {r["step"]: r for r in e214j if r["wash"] == "w2"}
    cur_w1 = {int(s): e for s, e in e214m["curves"]["w1"].items()}
    cur_w2 = {int(s): e for s, e in e214m["curves"]["w2"].items()}
    for s_ in (10, 50, DEEPEST):
        assert s_ in cur_w1 and s_ in cur_w2, f"e214 curves lack +{s_}"
        assert s_ in w1j and s_ in w2j, f"e214 journal lacks +{s_}"
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    log("committed records read: e214 metrics+journal (54-probe archive), "
        "e182 corpus record")

    # ------------------------------------------ P1 G_STATES part A: inventory
    def inv(p: Path):
        return {"path": str(p), "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                    p.stat().st_mtime)) if p.exists() else None}
    files = {"w1": {80: W1_ARCH[80]}, "w2": {80: W2_ARCH[80]}}
    inventories = {w: {str(s): inv(p) for s, p in d.items()}
                   for w, d in files.items()}
    e214_inv = e214m["inventory"]
    xcheck = {
        f"{w}:{s}": bool(e214_inv[f"wash{w[1]}_files"][str(s)]
                         and e214_inv[f"wash{w[1]}_files"][str(s)]["size_bytes"]
                         == inventories[w][str(s)]["size_bytes"]
                         and e214_inv[f"wash{w[1]}_files"][str(s)]["mtime"]
                         == inventories[w][str(s)]["mtime"])
        for w in ("w1", "w2") for s in (80,)}
    all_exist = all(v["exists"] for w in inventories
                    for v in inventories[w].values())
    G_STATES_A = {"files": inventories,
                  "crosscheck_vs_e214_inventory": xcheck,
                  "note": "only the two +80 states (the outcome's states) "
                          "are loaded and re-probed — THE tiny burst; the "
                          "intermediate states enter from e214's committed "
                          "certified records"}
    log(f"G_STATES inventory: both +80 states on disk = {all_exist}; "
        f"crosscheck = {xcheck}")
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
        "note": "the corpus is INHERITED FROZEN (the e214 chain) — it feeds "
                "the batteries' contamination scans AND this cell's frequency "
                "features; no wash is run here",
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
    write_metrics("PARTIAL: corpus rebuilt + certified")

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

    # t=0 probe (the runtime G_BATT witness)
    t0_run = {b: {r["fact"]: r["p"] for r in e1.probe_battery(net0, bats[b])
                  ["probes"]} for b in BATTERIES}
    G_BATT = {"tol_per_probe_dp": TOL_PROBE_DP, "batteries": {}}
    for b in BATTERIES:
        ref = t0j[b]["probes"]
        kept_eq = bool({r["fact"] for r in bats[b]} == set(ref))
        dp = max(abs(t0_run[b][f] - v["p"]) for f, v in ref.items())
        G_BATT["batteries"][b] = {"kept_set_equal": kept_eq, "n": len(ref),
                                  "max_dp_vs_e214_journal_t0": dp}
    G_BATT["pass"] = bool(all(
        G_BATT["batteries"][b]["kept_set_equal"]
        and G_BATT["batteries"][b]["max_dp_vs_e214_journal_t0"]
        <= TOL_PROBE_DP for b in BATTERIES)) if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} "
        + " | ".join(f"{b}: dp "
                     f"{G_BATT['batteries'][b]['max_dp_vs_e214_journal_t0']:.2e}"
                     for b in BATTERIES))
    write_metrics("PARTIAL: batteries rebuilt, t=0 certified")

    # ----------------------------- P5 the tiny re-probe bursts (G_STATES part B)
    reprobe = {}
    ck_map = {"w1": W1_ARCH, "w2": W2_ARCH}
    for wash, states in REPROBE_STATES.items():
        for s_ in states:
            load_checks.append(cpu_load_check(f"{wash}s{s_}"))
            sd = torch.load(ck_map[wash][s_], map_location=CPU,
                            weights_only=False)["model"]
            evl = copy.deepcopy(net0)
            evl.load_state_dict(sd)
            cur = {b: {r["fact"]: r["p"]
                       for r in e1.probe_battery(evl, bats[b])["probes"]}
                   for b in BATTERIES}
            del evl, sd
            ref_j = {"w1": w1j, "w2": w2j}[wash][s_]
            reprobe[f"{wash}:{s_}"] = {
                b: max(abs(cur[b][f] - v["p"]) for f, v in
                       ref_j[b]["probes"].items()) for b in BATTERIES}
            log(f"  {wash.upper()} +{s_} re-probe burst: max dp "
                + " | ".join(f"{b} {reprobe[f'{wash}:{s_}'][b]:.2e}"
                             for b in BATTERIES))
    all_dps = [dp for d in reprobe.values() for dp in d.values()]
    G_STATES = {**G_STATES_A, "all_exist": all_exist,
                "reprobe_max_dp": max(all_dps) if all_dps else None,
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_detail": reprobe,
                "reprobe_note": "all four batteries probed on the LOADED "
                                "w1 +80 and w2 +80 states, compared per-probe "
                                "to e214's committed records (which e214 "
                                "itself certified vs the e182c/e182c2 "
                                "originals at <= 3.3e-06) — same fp32 weights "
                                "+ same CPU fp32 probe path -> expected ~0.0"}
    G_STATES["pass"] = bool(all_exist and xcheck[f"w1:80"] and
                            xcheck[f"w2:80"]
                            and (SMOKE or (all_dps
                                           and max(all_dps)
                                           <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "load_checks": len(load_checks), "gpu_calls": 0,
             "burst": "two re-probe bursts (+80 per wash) + battery screening",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_STATES re-probe: max dp "
        f"{max(all_dps) if all_dps else 0:.8f} (tol {TOL_STATE_DP})")
    write_metrics("PARTIAL: re-probe bursts done")

    # --------------------------------------------- P6 the typology + features
    n_filtered = len(filtered)
    raw_lower = text.lower()
    filt_lower = filtered.lower()
    cue_counts = {}
    for fam, cues in TYPE_CUES.items():
        per_word_f = {w: filt_lower.count(w) for w in cues}
        per_word_r = {w: raw_lower.count(w) for w in cues}
        cnt = sum(per_word_f.values())
        cue_counts[fam] = {
            "cues": cues, "counts_filtered": per_word_f,
            "counts_raw": per_word_r, "total_filtered": cnt,
            "freq_per_1m_chars": cnt / n_filtered * 1e6,
        }
    log("relation-cue counts (filtered corpus / 1M chars): "
        + " | ".join(f"{f}: {c['total_filtered']} ({c['freq_per_1m_chars']:.1f})"
                     for f, c in cue_counts.items()))

    assign = []
    for b in BATTERIES:
        for r in bats[b]:
            fam = FAMILY6[(b, r["relation"])]
            p0 = t0j[b]["probes"][r["fact"]]["p"]
            row = {
                "battery": b, "relation": r["relation"], "family6": fam,
                "subject": r["subject"], "answer": r["answer"],
                "fact": r["fact"],
                "unique_anchor": r["fact"] in UNIQUE_ANCHOR_PROBES,
                "ans_tok_len": len(r["answer_ids"]),
                "subj_tok_len": len(tok.encode(" " + r["subject"])),
                "prompt_tok_len": int(r["ids"].shape[1]),
                "rel_cue_freq_per_1m": cue_counts[fam]["freq_per_1m_chars"],
                "p0": p0,
                "p80_w1": cur_w1[DEEPEST][b]["probes"][r["fact"]],
                "p80_w2": cur_w2[DEEPEST][b]["probes"][r["fact"]],
            }
            row["hr_w1"] = row["p80_w1"] / p0
            row["hr_w2"] = row["p80_w2"] / p0
            row["hr_mean"] = (row["hr_w1"] + row["hr_w2"]) / 2.0
            for s_ in DEPTH_COURSE:
                row[f"hr_w1_s{s_}"] = cur_w1[s_][b]["probes"][r["fact"]] / p0
                row[f"hr_w2_s{s_}"] = cur_w2[s_][b]["probes"][r["fact"]] / p0
            # the identically-zero probe-level exposure (co-report, verified)
            row["ans_stream_count"] = int(
                (train_ids == r["ans_id"]).sum().item())
            assign.append(row)
    n_probes = len(assign)
    assert n_probes == 54 or SMOKE, f"expected ~54 probes, got {n_probes}"
    n_stream_nonzero = sum(1 for r in assign if r["ans_stream_count"])
    metrics["typology"] = {
        "n_probes": n_probes,
        "family6_def": FAMILY6_DEF,
        "family6_ns": {f: sum(1 for r in assign if r["family6"] == f)
                       for f in FAMILY6_ORDER},
        "relation7_ns": {f"{b}:{rel}": sum(1 for r in assign
                                           if r["battery"] == b
                                           and r["relation"] == rel)
                         for b in BATTERIES
                         for rel in sorted({r["relation"]
                                            for r in assign
                                            if r["battery"] == b})},
        "cue_counts": cue_counts,
        "assignment_rule": "family6 = FAMILY6[(battery, module relation)] — "
                           "the hand-registered map at the top of this file "
                           "(frozen before compute); the assignment is "
                           "mechanical from the frozen pools, the GROUPING is "
                           "the hand-registered part",
        "assignment": assign,
        "subjectivity": "the typology is HAND-REGISTERED and follows the "
                        "dispatch's named families (which themselves follow "
                        "T187's reading of this archive's relation blocks — "
                        "NOT blind to the outcome); cap+cur merged per "
                        "e214's texture; the one unique-anchor probe "
                        "(Wikipedia) grouped with the founders per the "
                        "dispatch's 'founders/unique-anchor' family, flagged "
                        "per-probe",
    }
    log(f"typology: {n_probes} probes assigned; "
        + " | ".join(f"{f} {metrics['typology']['family6_ns'][f]}"
                     for f in FAMILY6_ORDER))
    write_metrics("PARTIAL: typology + features")

    # ------------------------------------------ P7 the hold/collapse split
    def med_hr(fam, key):
        return median([r[key] for r in assign if r["family6"] == fam])

    fam_split = {}
    for fam in FAMILY6_ORDER:
        rows = [r for r in assign if r["family6"] == fam]
        m1, m2 = med_hr(fam, "hr_w1"), med_hr(fam, "hr_w2")
        verdict = ("HOLDS" if (m1 >= HOLD_BAR and m2 >= HOLD_BAR)
                   else "COLLAPSES" if (m1 < COLLAPSE_BAR
                                        and m2 < COLLAPSE_BAR) else "MIXED")
        fam_split[fam] = {
            "n": len(rows), "def": FAMILY6_DEF[fam],
            "median_p0": median([r["p0"] for r in rows]),
            "median_hr_w1": m1, "median_hr_w2": m2,
            "mean_hr_w1": sum(r["hr_w1"] for r in rows) / len(rows),
            "mean_hr_w2": sum(r["hr_w2"] for r in rows) / len(rows),
            "median_hr_mean": median([r["hr_mean"] for r in rows]),
            "verdict": verdict,
        }
    depth_course = {fam: {f"w{w}_s{s}": med_hr(fam, f"hr_w{w}_s{s}")
                          for s in DEPTH_COURSE for w in (1, 2)}
                    for fam in FAMILY6_ORDER}

    # the cross-wash reliability of the sorting key
    hr1 = [r["hr_w1"] for r in assign]
    hr2 = [r["hr_w2"] for r in assign]

    def hclass(x: float) -> str:
        return "HOLD" if x >= HOLD_BAR else (
            "COLLAPSE" if x < COLLAPSE_BAR else "MID")

    conf = {}
    n_agree = 0
    for r in assign:
        c1, c2 = hclass(r["hr_w1"]), hclass(r["hr_w2"])
        conf.setdefault(c1, {}).setdefault(c2, 0)
        conf[c1][c2] += 1
        n_agree += int(c1 == c2)
    reliability = {
        "spearman_pooled": spearman(hr1, hr2),
        "spearman_per_battery": {
            b: spearman([r["hr_w1"] for r in assign if r["battery"] == b],
                        [r["hr_w2"] for r in assign if r["battery"] == b])
            for b in BATTERIES},
        "max_abs_hr_gap": max(abs(a - b) for a, b in zip(hr1, hr2)),
        "hold_class_def": f"HOLD hr >= {HOLD_BAR} / COLLAPSE hr < "
                          f"{COLLAPSE_BAR} / MID else",
        "class_confusion_w1xw2": conf,
        "class_agreement": n_agree / n_probes,
        "discordant_probes": [r["fact"] for r in assign
                              if hclass(r["hr_w1"]) != hclass(r["hr_w2"])],
    }
    metrics["hold_collapse"] = {
        "definition": f"hr_w(probe) = p_w(+{DEEPEST})/p_0 on the committed "
                      f"e214 records; family statistic = MEDIAN hr; HOLDS "
                      f"iff >= {HOLD_BAR} on both washes, COLLAPSES iff "
                      f"< {COLLAPSE_BAR} on both, else MIXED",
        "family_split": fam_split,
        "depth_course": depth_course,
    }
    metrics["reliability"] = reliability
    log("hold/collapse split: "
        + " | ".join(f"{f} {fam_split[f]['verdict']} "
                     f"({fam_split[f]['median_hr_w1']:.2f}/"
                     f"{fam_split[f]['median_hr_w2']:.2f})"
                     for f in FAMILY6_ORDER))
    log(f"reliability: Spearman(hr_w1, hr_w2) = {reliability['spearman_pooled']}"
        f"; class agreement {reliability['class_agreement']:.0%}")
    write_metrics("PARTIAL: hold/collapse split + reliability")

    # --------------------------------------------- P8 the predictor ladder
    Y = [r["hr_mean"] for r in assign]
    G6 = [r["family6"] for r in assign]
    G7 = [f"{r['battery']}:{r['relation']}" for r in assign]
    anova6 = one_way(Y, G6)
    anova7 = one_way(Y, G7)
    anova_w1 = one_way([r["hr_w1"] for r in assign], G6)
    anova_w2 = one_way([r["hr_w2"] for r in assign], G6)

    feat_defs = [
        ("rel-cue freq", "rel_cue_freq_per_1m", True, "relation-level"),
        ("ans tok len", "ans_tok_len", True, "probe-level"),
        ("subj tok len", "subj_tok_len", True, "probe-level"),
        ("prompt tok len", "prompt_tok_len", False, "probe-level competitor"),
        ("p0 (height)", "p0", False, "the baseline-strength competitor "
                                     "(e214's negative, quantified)"),
        ("ans stream count", "ans_stream_count", False,
         "probe-level; identically 0 by the contamination gates (expected "
         "degenerate)"),
    ]
    features = []
    for label, key, adjudicable, note in feat_defs:
        fv = [r[key] for r in assign]
        rho = spearman(fv, Y)
        rho_w1 = spearman(fv, [r["hr_w1"] for r in assign])
        rho_w2 = spearman(fv, [r["hr_w2"] for r in assign])
        part, part_note = partial_spearman(fv, Y, G6)
        features.append({
            "label": label, "key": key, "adjudicable": adjudicable,
            "note": note, "value_range": [min(fv), max(fv)],
            "rho": rho, "rho2": None if rho is None else rho ** 2,
            "rho_w1": rho_w1, "rho_w2": rho_w2,
            "partial_rho": part, "partial_note": part_note,
        })
        log(f"ladder {label:16s}: rho {rho if rho is None else round(rho, 3)}"
            f" (rho2 {None if rho is None else round(rho ** 2, 3)})"
            f" partial {part if part is None else round(part, 3)}"
            f" [{note}]")

    # the cue value per family (for the table)
    cue_by_fam = {f: cue_counts[f]["freq_per_1m_chars"] for f in FAMILY6_ORDER}

    ladder = {
        "outcome": "hr_mean = (hr_w1 + hr_w2)/2, per probe, from the "
                   "committed e214 records",
        "family_anova": {**anova6,
                         "co_report_7rel": {k: anova7[k] for k in
                                            ("var_ratio_SSb_SSw", "F",
                                             "eta2")},
                         "co_report_w1": {k: anova_w1[k] for k in
                                          ("var_ratio_SSb_SSw", "F", "eta2")},
                         "co_report_w2": {k: anova_w2[k] for k in
                                          ("var_ratio_SSb_SSw", "F", "eta2")},
                         "separation_stat": "SS_between/SS_within",
                         "separation_bar": SEP_BAR},
        "features": features,
        "cue_freq_by_family": cue_by_fam,
        "materiality_bar": MATERIALITY_BAR,
        "constants": {
            "SEP_BAR (variance ratio)": SEP_BAR,
            "MATERIALITY_BAR (|partial rho|)": MATERIALITY_BAR,
            "HOLD_BAR / COLLAPSE_BAR (descriptive bands)":
                [HOLD_BAR, COLLAPSE_BAR],
        },
    }
    metrics["predictor_ladder"] = ladder
    write_metrics("PARTIAL: predictor ladder")

    # --------------------------------------------- P9 adjudication (frozen)
    separation = bool(anova6["var_ratio_SSb_SSw"] is not None
                      and anova6["var_ratio_SSb_SSw"] >= SEP_BAR)
    material = [f["label"] for f in features
                if f["adjudicable"] and f["partial_rho"] is not None
                and abs(f["partial_rho"]) >= MATERIALITY_BAR]
    outpred = [f["label"] for f in features
               if f["adjudicable"] and f["rho2"] is not None
               and f["rho2"] > anova6["eta2"]]
    gates_ok = bool(G_STATES["pass"] and G_BATT["pass"]
                    and G_CORPUS["pass"] and G_ENV["pass"])

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_STATES", G_STATES["pass"]),
                           ("G_BATT", G_BATT["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif separation and not material:
        best_part = max((abs(f["partial_rho"]) or 0) for f in features
                        if f["adjudicable"])
        verdict = "FAMILY-IS-THE-KEY"
        clause = (f"the family label predicts the hold ratio — between-family "
                  f"separation SSb/SSw {anova6['var_ratio_SSb_SSw']:.2f} >= "
                  f"{SEP_BAR:g}x the within-family spread (eta2 "
                  f"{anova6['eta2']:.3f}, F {anova6['F']:.2f}, n=54, k=6) and "
                  f"no structural feature adds materially (best |family-"
                  f"partial| {best_part:.3f} < "
                  f"{MATERIALITY_BAR}; the frequency and length arms "
                  f"near-degenerate in this archive: answer length frozen at "
                  f"1 token by the instrument's own gate, in-wash exposure "
                  f"identically 0 at the probe level, relation cues 0-13 per "
                  f"1M chars) — the wash sorts by RELATION TYPE; the "
                  f"relational signature is the law's object")
    elif outpred:
        out_txt = ", ".join(
            f"{lab} rho2 "
            f"{[x['rho2'] for x in features if x['label'] == lab][0]:.3f}"
            f" > eta2 {anova6['eta2']:.3f}" for lab in outpred)
        verdict = "STRUCTURE-BEATS-FAMILY"
        clause = (f"a structural feature outpredicts the family — {out_txt}"
                  + f" (family separation SSb/SSw "
                  f"{anova6['var_ratio_SSb_SSw']:.2f}) — the 'relational' "
                  "reading reduces to exposure; the sorting key is "
                  "statistical, not semantic")
    else:
        why = []
        if not separation:
            why.append(f"the separation clause failed (SSb/SSw "
                       f"{anova6['var_ratio_SSb_SSw']:.2f} < {SEP_BAR:g} — "
                       f"the within-family spread carries the outcome)")
        if material:
            part_txt = ", ".join(
                f"{lab} |partial| "
                f"{abs([x['partial_rho'] for x in features if x['label'] == lab][0]):.3f}"
                for lab in material)
            why.append(f"structural features add materially inside family "
                       f"({part_txt})")
        if not outpred:
            why.append(f"no structural feature outpredicts the family (best "
                       f"rho2 "
                       f"{max((f['rho2'] or 0) for f in features if f['adjudicable']):.3f}"
                       f" vs eta2 {anova6['eta2']:.3f})")
        verdict = "GRADED"
        clause = ("any partial — " + "; ".join(why)
                  + "; the tables verbatim (the typology split, the "
                  "reliability, the ladder)")

    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {"FAMILY_IS_THE_KEY": bool(separation and not material
                                           and gates_ok and not SMOKE),
                 "STRUCTURE_BEATS_FAMILY": bool(outpred and gates_ok
                                                and not SMOKE),
                 "verdict": verdict, "clause": clause,
                 "order": "FAMILY-IS-THE-KEY -> STRUCTURE-BEATS-FAMILY -> "
                          "GRADED (gated on G_STATES/G_BATT/G_CORPUS/G_ENV)"},
        "clauses": {
            "separation (SSb/SSw >= 2)": separation,
            "no_material_structural_addition (|partial| < 0.30)":
                not material,
            "material_features": material,
            "outpredicting_features (rho2 > eta2)": outpred,
        },
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E215 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    # ---------------------------------------------------- P10 honesty + close
    metrics["honesty_reflex"] = {
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "a device-texture asymmetry between the arms, inherited "
                    "from the archive (disclosed)",
        "typology_not_blind": "the typology is HAND-REGISTERED from the "
                              "dispatch's named families, which T187 read "
                              "off THIS archive's relation blocks — the "
                              "family label's win is partly built-in; the "
                              "defense is only that the GROUPING comes from "
                              "the instrument's design (the frozen battery "
                              "pools), not the wash data; subjectivity "
                              "disclosed in typology.subjectivity",
        "unbalanced_ns": "lang 10, cap-cur 10, founder-anchor 5, product 7, "
                         "near-uscap 3 (n=3!), rev-capital 19 — the "
                         "ANOVA's family means carry unequal precision",
        "frequency_arm_degenerate": "the wash corpus is filtered Shakespeare: "
                                    "relation cues occur 0-13 times (per-word "
                                    "counts in metrics); EVERY probe's "
                                    "subject/answer strings are ABSENT from "
                                    "the wash corpus by the contamination "
                                    "gates (verified: ans_stream_count "
                                    f"nonzero for {n_stream_nonzero}/54 "
                                    "probes) — in-wash exposure is "
                                    "identically zero at the probe level; "
                                    "the exposure hypothesis is testable "
                                    "here only at the relation-cue level",
        "length_arm_null": "answer token length is CONSTANT 1 by the e182 "
                           "gate (single-token answers only) — the "
                           "instrument froze the feature; a Rule-12 "
                           "disclosure, not a finding",
        "relation_level_features": "rel-cue frequency is family-level "
                                   "(constant within family): its raw rho "
                                   "tests whether the TYPE ORDER tracks "
                                   "corpus exposure; its family-partial is "
                                   "structurally undefined (zero within-"
                                   "family variance), disclosed per feature",
        "hr_ratio_floor": "hr divides by p0 as low as "
                          f"{min(r['p0'] for r in assign):.2f}; deep-state "
                          "floors (p 0.05-0.2) compress ratios; hr and p80 "
                          "co-reported per probe",
        "collinearity": "subject length and family are strongly collinear "
                        "(ctrl subjects long, fact subjects short, "
                        "near/tmpl single-word states) — the family-partial "
                        "is the honest residual test, and it cannot "
                        "attribute variance between collinear predictors",
        "sep_bar_reading": "the separation bar was operationalized as the "
                           "plain variance ratio SSb/SSw >= 2 (the literal "
                           "reading); the mean-square F ("
                           f"{anova6['F']:.2f}) is df-inflated (~9.6x the "
                           "ratio here) and co-reported, never adjudicated",
        "guarantees_nothing": "the ladder describes THESE 54 probes under "
                              "THESE two washes with a hand-registered "
                              "typology — 'the wash sorts by relation type' "
                              "is an instrument-level statement about this "
                              "archive; the openness is the point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, TINY bursts (two +80 re-probe "
                    "loads + battery screening), decisive; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(W1_ARCH[80]), str(W2_ARCH[80])],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, assign, fam_split, ladder, reliability, adj,
                        anova6)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
