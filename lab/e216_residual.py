"""E216 — THE WITHIN-FAMILY RESIDUAL (T189's named open thread).

WHY: e215/T189 resolved the sorting key as FAMILY-FIRST (eta2 0.577) with
the within-family residue carried by the baseline p0 (family-partial
Spearman 0.45 — "family x height is the model"). THE OPEN QUESTION THIS CELL
REGISTERS: is that model the WHOLE story — how much of the hold ratio does
family x height actually EXPLAIN (the R^2 of the joint regression, never
yet computed), and what lives in its RESIDUAL? The named splits to test
against the model: the product family's internal chasm (Gmail/PlayStation
HOLD at hr ~0.68-0.93 while iPhone COLLAPSES at 0.18-0.34) and the tmpl
family's width (rev-capital hr_mean spans 0.16-0.89). Do they SURVIVE the
model (structured residual = a third sorting dimension) or DISSOLVE into it
(family x height is the whole key)?

THE CELL (a PURE DESK cell on the committed 54-probe records — no model
loads, no re-probes; the tokenizer only, for the corpus/prompt rebuild the
vocab-overlap feature needs):
  (1) THE FULL MODEL — regress hr_w on family (5 dummies, reference = lang)
      + p0 (centered) per wash; the R^2 (and the ladder beneath it:
      family-only eta2, p0-only rho2-equivalent, the increments).
  (2) THE RESIDUAL — per-probe residuals from the full model; their
      structure: do the named splits survive (the product contrast in
      residual-SD units; the tmpl width as the per-family SD ratio) or
      dissolve?
  (3) RESIDUAL PREDICTORS — any feature left that correlates with the
      residual: the probe's token overlap with the wash vocabulary (the
      fraction of its prompt token ids present in the frozen wash TRAIN
      stream); the answer's entrenchment in the base model (p0 at t=0 —
      the PRE-INSTALL state in this no-install archive, computable from
      the committed records, hence computed, not skipped; DISCLOSED: p0 is
      already a linear regressor in the model, so its residual rho tests
      the NONLINEAR leftover of entrenchment); the probe's position in its
      battery (the instrument's own pool order).
  (4) THE CROSS-WASH RESIDUAL — Spearman(resid_w1, resid_w2): is the
      residual physics (replicating idiosyncrasy) or noise?

REGISTERED BARS (frozen VERBATIM from the dispatch brief, QUEUE row
DISPATCHED 15:47Z / commit 28b97dc, BEFORE any compute; adjudicate against
exactly this; no bar shopping):
  - MODEL-COMPLETE: "fires if the full model's R-squared >= 0.75 AND the
    residuals are unstructured (no residual predictor |rho| > 0.3; the
    cross-wash residual correlation < 0.3) — FAMILY x HEIGHT is the whole
    sorting key; the named splits dissolve into it."
  - RESIDUAL-STRUCTURED: "fires if the residuals carry structure (a
    predictor |rho| > 0.3 or the cross-wash residual rho > 0.5) — a third
    sorting dimension exists; named with the best candidate."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * DATA = runs/e215/metrics.json typology.assignment (the committed
    54-probe records; e215 certified them against e214's committed records
    by runtime re-probe at dp 0.0, and e214 had certified those against the
    e182c/e182c2 originals at <= 3.3e-06). Here the chain is re-certified
    DESK-ONLY: hr and p0 recomputed from e214's committed curves + journal
    t=0 and matched per probe (G_RECOMPUTE, tol 1e-9) + the git provenance
    of the records files (G_RECORDS).
  * OUTCOME = hr_w = p_w(+80)/p_0 per wash (the deepest shared state); the
    hr_mean co-report beside it.
  * FULL MODEL = OLS hr_w ~ 1 + 5 family dummies (treatment coding,
    reference family = lang) + p0 centered at the 54-probe mean; raw hr
    (no transform — disclosed). R^2 = 1 - SS_res/SS_tot. PRIMARY = the
    per-wash R^2; MODEL-COMPLETE requires BOTH washes >= 0.75. adj-R^2,
    the family-only R^2 (eta2), the p0-only R^2, and both increments
    co-reported.
  * RESIDUAL = per-probe e_i = hr_w,i - fitted_i (the same rows, both
    washes + hr_mean).
  * NAMED SPLITS (descriptive conventions — they feed the clause wording,
    they never flip a bar): the PRODUCT CONTRAST = mean resid{Gmail,
    PlayStation} - mean resid{iPhone}, per wash, in units of that wash's
    residual SD; "SURVIVES" iff >= 1.0 SD on BOTH washes. THE TMPL WIDTH =
    SD(resid)/SD(hr) within rev-capital per wash; "DISSOLVES" iff <= 0.5
    on both washes. The per-family SD table + the within-family
    Spearman(p0, hr) co-reported.
  * RESIDUAL PREDICTORS (the registered set, the bar's "a predictor"):
    (r1) vocab overlap = fraction of the probe's PROMPT token ids present
    in the set of the frozen wash TRAIN stream ids (rebuilt + certified by
    G_CORPUS; prompts rebuilt by module import and certified by G_PROMPT);
    (r2) entrenchment = p0 (t=0, the pre-wash/pre-install analogue; in the
    model linearly — its residual rho is the nonlinear-leftover test,
    disclosed); (r3) position = the 0-based index of the probe in its
    battery's kept order (the instrument's own candidate/pool order,
    certified equal to the committed assignment order by G_PROMPT).
    Statistic per predictor = Spearman(predictor, residual) per wash; the
    predictor FIRES iff max |rho| over the two washes > 0.30. Co-reported,
    never adjudicated: subj_tok_len, prompt_tok_len, rel-cue frequency
    (family-level: constant within family — its residual rho orders the
    families' mean residuals, disclosed), ans_tok_len (constant 1,
    degenerate).
  * CROSS-WASH RESIDUAL = Spearman(resid_w1, resid_w2) over the 54 probes
    (the e215 reliability convention; Pearson co-reported; per-family
    co-reported). MODEL-COMPLETE requires < 0.3; RESIDUAL-STRUCTURED fires
    at > 0.5.
  * ADJUDICATION ORDER: gates ok -> MODEL-COMPLETE iff (R2_w1 >= 0.75 AND
    R2_w2 >= 0.75 AND no registered predictor fires AND xwash Spearman
    < 0.3); else RESIDUAL-STRUCTURED iff (any registered predictor fires
    OR xwash Spearman > 0.5), named with the best candidate (the largest
    firing |rho|; the cross-wash term named when it is the leader); else
    GRADED.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_RECORDS — e215's committed record is DONE, n=54, its own four gates
    (G_STATES/G_BATT/G_CORPUS/G_ENV) all PASS; the git provenance recorded
    (HEAD + the last commit touching runs/e215/metrics.json + a
    clean-tree check on the records chain e214/e215 metrics+journal).
  * G_RECOMPUTE — p0, p80_w1, p80_w2, hr_w1, hr_w2 recomputed from e214's
    committed curves + journal t=0, matched per probe: max |dp| <= 1e-9.
  * G_CORPUS — the frozen wash corpus rebuilt (tokenizer + e1's filter,
    answer ids taken from the committed fact battery) and asserted EQUAL
    to e182's committed filter stats; the banned list identical.
  * G_PROMPT — the four batteries' probes rebuilt VERBATIM by module
    import (candidate construction, no model): the kept sets equal the
    committed 54 per battery, the ORDER equal, and every probe's prompt
    token length equal to the committed prompt_tok_len (tol 0).
  * G_ENV — the owner envelope: CPU-only (zero GPU calls), torch threads
    <= 4, load checks, ZERO model loads (the tokenizer only), decisive.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. n=2 washes is texture, not law (wash 1 a CPU fp32 replay, wash
2 GPU fp32 — inherited, disclosed). The typology is HAND-REGISTERED and
not blind to the outcome (disclosed in e215 and inherited here). The full
model's family term INHERITS that subjectivity: R^2 measured against a
hand-grouped factor. OLS on a raw bounded ratio (hr in ~[0.12, 1.22]) with
unbalanced family n's (3..19) and heteroskedastic family widths — the
plain R^2 is registered as the bar's letter ("the full model's
R-squared"), the adj-R^2 and the rank-side co-reports beside it. The
entrenchment predictor is collinear with the model's own p0 term BY
DESIGN (the dispatch's proxy is p0 itself); its firing means nonlinearity,
not an independent dimension — disclosed wherever it is read. The named
split conventions (1 SD, 0.5 ratio) are descriptive yardsticks frozen
before compute, not bars. A residual correlated across washes could be
probe physics OR shared pipeline texture (the same p0 denominator in both
arms) — the discrimination is disclosed, not assumed.

PROVENANCE: the per-probe records, the typology and the committed
certification chain are e215's (read here, never rewritten); the curves,
journal and checkpoint inventory are e214's; the batteries, corpus filter
and organism constants are lab/e182c_forgetting_control.py + lab/
e182c2_template.py VERBATIM via module import. Builds on: T189 / e215
(the named open thread; the family x height model this cell residuals),
W028 (the shape/height law), T187 / e214 (the archive), T183 / e182c2 +
T149 / e182c + T123 / e182 (the two-wash 124M archive). NEW: the joint
family x height regression and its R^2 ladder; the per-probe residual
table on both washes; the named-split survival tests against the model;
the residual-predictor set (vocab overlap / entrenchment-leftover /
battery position); the cross-wash residual correlation.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and after the corpus phase; ZERO
model loads (the GPT-2 tokenizer from the pinned local cache only); no
wash, no probes, no NOTES/THINKING/QUEUE/STATE edits (dispatch).
Progressive PARTIAL metrics after every phase (the standing disruption
rule).

Run:  cd lab && python e216_residual.py   (E216_SMOKE=1: same desk tables
      on the same committed records, own smoke dir, nothing adjudicated —
      the e215 smoke disclosure applies: a desk shakedown inevitably
      displays the committed tables)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import common                                               # noqa: E402
from common import now_iso, run_dir, save_json              # noqa: E402

import e182c_forgetting_control as e1                       # noqa: E402 — the archive machinery, VERBATIM
import e182c2_template as e2                                # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                    # the owner envelope (this dispatch: <= 4)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import textwrap                                             # noqa: E402

try:      # Windows console safety for the lab's typographic register
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                           # noqa: BLE001
    pass

SMOKE = os.environ.get("E216_SMOKE") == "1"
NAME = "e216_smoke" if SMOKE else "e216"

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
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
FAMILY6_ORDER = ["lang", "cap-cur", "founder-anchor", "product",
                 "near-uscap", "rev-capital"]
REF_FAMILY = "lang"             # the treatment-coding reference (largest n=10)

# the registered bar constants (frozen)
R2_BAR = 0.75          # "the full model's R-squared >= 0.75" -> BOTH washes
PRED_BAR = 0.30        # "no residual predictor |rho| > 0.3"
XWASH_LOW = 0.30       # MODEL-COMPLETE's "cross-wash residual correlation < 0.3"
XWASH_HIGH = 0.50      # RESIDUAL-STRUCTURED's "cross-wash residual rho > 0.5"

# the named-split descriptive yardsticks (frozen; never adjudicated)
PRODUCT_HOLD_GROUP = ["Gmail", "PlayStation"]   # the dispatch's named holds
PRODUCT_COLLAPSE_ANCHOR = "iPhone"              # the dispatch's named collapse
SPLIT_SD_BAR = 1.0     # product contrast SURVIVES iff >= 1.0 residual SD both washes
WIDTH_DISSOLVE_BAR = 0.5   # tmpl width DISSOLVES iff SD(resid)/SD(hr) <= 0.5 both washes

# the registered residual-predictor set (the bar's "a predictor")
RESID_PREDICTORS = [
    ("vocab overlap", "vocab_overlap",
     "fraction of the probe's prompt token ids present in the frozen wash "
     "TRAIN stream (rebuilt + G_CORPUS/G_PROMPT certified)"),
    ("entrenchment p0", "p0",
     "p0 at t=0 — the PRE-INSTALL analogue (computable from the committed "
     "records; in the model LINEARLY, so its residual rho tests the "
     "nonlinear leftover; disclosed)"),
    ("battery position", "battery_pos",
     "0-based index of the probe in its battery's kept order (the "
     "instrument's own candidate/pool order)"),
]
# co-report competitors (never adjudicated)
RESID_COMPETITORS = [
    ("subj tok len", "subj_tok_len"),
    ("prompt tok len", "prompt_tok_len"),
    ("rel-cue freq", "rel_cue_freq_per_1m"),
]

# registered verification tolerances (frozen)
TOL_RECOMPUTE_DP = 1e-9

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "MODEL-COMPLETE": "fires if the full model's R-squared >= 0.75 AND "
            "the residuals are unstructured (no residual predictor |rho| "
            "> 0.3; the cross-wash residual correlation < 0.3) — FAMILY x "
            "HEIGHT is the whole sorting key; the named splits dissolve "
            "into it.",
        "RESIDUAL-STRUCTURED": "fires if the residuals carry structure (a "
            "predictor |rho| > 0.3 or the cross-wash residual rho > 0.5) — "
            "a third sorting dimension exists; named with the best "
            "candidate.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "operationalizations": (
        "data = e215's committed 54-probe assignment (re-certified desk-only "
        "by G_RECOMPUTE against e214's curves+journal at 1e-9 + G_RECORDS "
        "git provenance); outcome = hr_w = p_w(+80)/p0 per wash; FULL MODEL "
        "= OLS hr_w ~ 1 + 5 family dummies (reference lang) + p0 centered; "
        "R2 = 1 - SSres/SStot, PRIMARY per wash (MODEL-COMPLETE needs BOTH "
        ">= 0.75), adj-R2/family-only/p0-only/increments co-reported; "
        "residual = hr - fitted per probe; named splits descriptive "
        "(product contrast mean resid{Gmail,PlayStation} - mean "
        "resid{iPhone} in residual-SD units, SURVIVES iff >= 1.0 SD both "
        "washes; tmpl width SD(resid)/SD(hr) within rev-capital, DISSOLVES "
        "iff <= 0.5 both washes); RESIDUAL PREDICTORS (registered set) = "
        "vocab overlap (prompt-token overlap with the frozen wash train "
        "stream), entrenchment p0 (t=0 pre-install analogue — already a "
        "linear regressor: its residual rho is the nonlinear leftover, "
        "disclosed), battery position (0-based index in the battery's kept "
        "order); statistic = Spearman(predictor, residual) per wash, FIRES "
        "iff max |rho| over washes > 0.30; CROSS-WASH RESIDUAL = "
        "Spearman(resid_w1, resid_w2) (Pearson co-reported); ADJUDICATION: "
        "gates ok -> MODEL-COMPLETE iff (R2 both washes >= 0.75 AND no "
        "predictor fires AND xwash Spearman < 0.3); else RESIDUAL-STRUCTURED "
        "iff (a predictor fires OR xwash Spearman > 0.5), named with the "
        "best candidate; else GRADED; gated on G_RECORDS/G_RECOMPUTE/"
        "G_CORPUS/G_PROMPT/G_ENV"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (QUEUE "
                     "row DISPATCHED 15:47Z, commit 28b97dc) BEFORE any "
                     "compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "PURE DESK cell: NO model loads, NO re-probes (the dispatch's 'pure desk "
    "on the committed records') — the provenance chain enters as e215's "
    "committed certification (its runtime re-probes read dp 0.0 vs e214's "
    "records, which e214 certified vs the e182c/e182c2 originals at "
    "<= 3.3e-06) PLUS this cell's desk recompute (G_RECOMPUTE) and git "
    "provenance (G_RECORDS).",
    "The entrenchment predictor IS the model's own p0 term (the dispatch's "
    "proxy — p0 at t=0, the pre-install analogue in this no-install "
    "archive): computable, hence computed; its residual Spearman can only "
    "fire on NONLINEAR entrenchment leftover, never on an independent "
    "dimension — disclosed at the feature and in the honesty block.",
    "The named-split conventions (1.0 residual SD for the product contrast; "
    "0.5 SD-ratio for the tmpl width) are DESCRIPTIVE yardsticks frozen "
    "before compute — they feed the clause wording, they never flip a bar.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: the same desk tables on the same committed records (the "
    "e215 smoke disclosure applies — a desk shakedown inevitably displays "
    "the committed tables), own smoke dir, nothing adjudicated.",
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


def _pearson(xs, ys) -> float | None:
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


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS via lstsq: betas, fitted, resid, R2, adj-R2, resid SD."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    dof = n - p
    # coefficient SEs (classical homoskedastic formula; co-report only)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        sigma2 = ss_res / dof
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return {"beta": beta, "fitted": fitted, "resid": resid, "r2": r2,
            "adj_r2": adj, "ss_res": ss_res, "ss_tot": ss_tot,
            "resid_sd": float(ss_res / dof) ** 0.5, "dof": dof,
            "beta_se": se}


def design(fams: list[str], p0: list[float], p0_mean: float) -> np.ndarray:
    """[1, 5 family dummies (ref = lang), p0 centered]."""
    cols = [np.ones(len(fams))]
    for f in FAMILY6_ORDER:
        if f == REF_FAMILY:
            continue
        cols.append(np.array([1.0 if g == f else 0.0 for g in fams]))
    cols.append(np.array(p0) - p0_mean)
    return np.column_stack(cols)


def coef_names() -> list[str]:
    return (["intercept(lang)"]
            + [f"D[{f}]" for f in FAMILY6_ORDER if f != REF_FAMILY]
            + ["p0_centered"])


def git_record() -> dict:
    """The records chain's git provenance (G_RECORDS)."""
    def _git(*args) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(common.REPO), capture_output=True,
                text=True, timeout=30).stdout.strip()
        except Exception as e:                            # noqa: BLE001
            return f"unavailable ({e})"
    chain = ["runs/e214/metrics.json", "runs/e214/journal.json",
             "runs/e215/metrics.json"]
    return {
        "head": _git("rev-parse", "HEAD"),
        "records_commits": {c: _git("log", "-1", "--format=%h %ad",
                                    "--date=short", "--", c) for c in chain},
        "dirty_records": [c for c in chain
                          if _git("status", "--porcelain", "--", c) != ""],
    }


# ------------------------------------------------------------------ plot

FAM_COLS = {"lang": "tab:green", "cap-cur": "tab:red",
            "founder-anchor": "tab:blue", "product": "tab:purple",
            "near-uscap": "tab:olive", "rev-capital": "darkorange"}


def make_plot(rd, rows, model, splits, preds, xw, adj):
    """THE FIGURE: (a) fitted vs hr (both washes, family colors, R^2);
  (b) the residuals by family (the named splits annotated);
  (c) the residual-predictor ladder (|rho| per wash, the bars);
  (d) the cross-wash residual scatter + verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) fitted vs observed
    ax = axes[0, 0]
    for fam in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([r["fit_w1"] for r in fr], [r["hr_w1"] for r in fr],
                   s=26, marker="o", color=FAM_COLS[fam], alpha=0.75,
                   label=f"{fam} (n={len(fr)})")
        ax.scatter([r["fit_w2"] for r in fr], [r["hr_w2"] for r in fr],
                   s=26, marker="s", facecolors="none",
                   edgecolors=FAM_COLS[fam], linewidths=1.3, alpha=0.9)
    lims = [0.0, 1.15]
    ax.plot(lims, lims, "k--", lw=0.9, alpha=0.6)
    ax.set_xlabel("fitted hr  (o = wash 1,  [] = wash 2)")
    ax.set_ylabel("observed hr  p(+80)/p_0")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, loc="upper left")
    ax.set_title(f"(1) THE FULL MODEL hr ~ family + p0 — R2 w1 "
                 f"{model['w1']['r2']:.3f} / w2 {model['w2']['r2']:.3f} "
                 f"(bar {R2_BAR:g}; family-only eta2 "
                 f"{model['ladder']['family_only_r2_by_wash']['w1']:.3f}"
                 f"/{model['ladder']['family_only_r2_by_wash']['w2']:.3f} "
                 f"+ p0 -> "
                 f"{model['ladder']['delta_p0_given_family']['w1']:+.3f})",
                 fontsize=9.0)

    # (0,1) residuals by family
    ax = axes[0, 1]
    ax.axhline(0.0, color="k", lw=1.0)
    for xi, fam in enumerate(FAMILY6_ORDER):
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([xi + 0.10] * len(fr), [r["resid_w1"] for r in fr],
                   s=26, marker="o", color=FAM_COLS[fam], alpha=0.75)
        ax.scatter([xi - 0.10] * len(fr), [r["resid_w2"] for r in fr],
                   s=26, marker="s", facecolors="none",
                   edgecolors=FAM_COLS[fam], linewidths=1.3, alpha=0.9)
    ax.set_xticks(range(len(FAMILY6_ORDER)))
    ax.set_xticklabels([f"{f}\n(n={sum(1 for r in rows if r['family6'] == f)})"
                        for f in FAMILY6_ORDER], fontsize=7.2)
    ax.set_ylabel("RESIDUAL  hr - (family x height fit)")
    # annotate the named product probes
    named = {a: a for a in PRODUCT_HOLD_GROUP + [PRODUCT_COLLAPSE_ANCHOR]}
    for r in rows:
        if r["answer"] in named:
            xi = FAMILY6_ORDER.index(r["family6"])
            for w, off in (("resid_w1", 0.10), ("resid_w2", -0.10)):
                ax.annotate(named[r["answer"]], (xi + off, r[w]),
                            textcoords="offset points", xytext=(6, 3),
                            fontsize=6.0, family="monospace",
                            color="black" if w == "resid_w1" else "dimgray")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(2) THE RESIDUAL BY FAMILY — product contrast "
                 f"{splits['product']['gap_sd_w1']:+.2f}/{splits['product']['gap_sd_w2']:+.2f} SD "
                 f"({splits['product']['verdict']}); tmpl width ratio "
                 f"{splits['tmpl_width']['ratio_w1']:.2f}/"
                 f"{splits['tmpl_width']['ratio_w2']:.2f} "
                 f"({splits['tmpl_width']['verdict']})", fontsize=9.0)

    # (1,0) the residual-predictor ladder
    ax = axes[1, 0]
    regs = preds["registered"]
    comps = preds["competitors"]
    labels = ([f["label"] for f in regs] + ["XWASH resid"] +
              [f["label"] + "*" for f in comps])
    vals1 = ([f["rho_w1"] for f in regs] + [xw["spearman"]]
             + [f["rho_w1"] for f in comps])
    vals2 = ([f["rho_w2"] for f in regs] + [xw["spearman"]]
             + [f["rho_w2"] for f in comps])
    xs = range(len(labels))
    for i in xs:
        for v, dx, mk, col in ((vals1[i], -0.17, "o", "tab:blue"),
                               (vals2[i], +0.17, "s", "tab:cyan")):
            if v is None:
                continue
            ax.bar(i + dx, v, width=0.32, color=col, edgecolor="k", lw=0.5)
    ax.axhline(PRED_BAR, color="tab:red", ls=":", lw=1.4,
               label=f"|rho| bar {PRED_BAR}")
    ax.axhline(-PRED_BAR, color="tab:red", ls=":", lw=1.4)
    ax.axhline(XWASH_HIGH, color="darkred", ls="--", lw=1.2,
               label=f"xwash structured {XWASH_HIGH}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=7.0)
    ax.set_ylabel("Spearman(feature, residual)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("(3) RESIDUAL PREDICTORS (blue w1 / cyan w2; * = co-report "
                 "competitors) + the cross-wash residual term", fontsize=9.0)

    # (1,1) cross-wash residual scatter + verdict
    ax = axes[1, 1]
    for fam in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([r["resid_w1"] for r in fr], [r["resid_w2"] for r in fr],
                   s=28, color=FAM_COLS[fam], alpha=0.8, label=fam)
    rmax = max(max(abs(r["resid_w1"]), abs(r["resid_w2"])) for r in rows)
    lim = rmax * 1.12
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.9, alpha=0.6)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("residual, wash 1")
    ax.set_ylabel("residual, wash 2")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, loc="lower right")
    ax.set_title(f"(4) THE CROSS-WASH RESIDUAL — Spearman "
                 f"{xw['spearman']:.3f} / Pearson {xw['pearson']:.3f} "
                 f"(noise < {XWASH_LOW}; physics > {XWASH_HIGH})",
                 fontsize=9.0)

    # the verdict strip under (4)
    y = -0.155
    ax.text(0.0, y - 0.045, f"E216 VERDICT: {adj['bars']['verdict']}",
            fontsize=10.5, family="monospace", weight="bold",
            color="darkred", transform=ax.transAxes)
    for k, wd in enumerate(textwrap.wrap(adj["bars"]["clause"], width=110,
                                         break_long_words=False)):
        ax.text(0.0, y - 0.095 - k * 0.040, wd, fontsize=6.8,
                family="monospace", transform=ax.transAxes)
    ax.text(0.0, y - 0.095 - (len(textwrap.wrap(adj["bars"]["clause"],
                                                width=110)) + 0.6) * 0.040,
            "  GATES: " + "  ".join(f"{g}={'PASS' if v else 'FAIL'}"
                                    for g, v in adj["gates_summary"].items()),
            fontsize=7.0, family="monospace", transform=ax.transAxes)

    fig.suptitle("E216 — THE WITHIN-FAMILY RESIDUAL (T189's open thread): "
                 "what is left of family x height? -> "
                 + adj["bars"]["verdict"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "residual.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E216 — THE WITHIN-FAMILY RESIDUAL (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e216_residual",
        "phase": "pure desk on the committed 54-probe records (no model "
                 "loads, no re-probes; the tokenizer only)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T189's named open thread: family x height is the "
                     "sorting-key model — WHAT IS LEFT? (1) the full model's "
                     "R^2; (2) do the named splits survive its residual; "
                     "(3) any residual predictor; (4) is the residual "
                     "physics (cross-wash) or noise"),
        "builds_on": ["T189 / e215 (the named open thread; the family x "
                      "height model this cell residuals + the committed "
                      "54-probe records it reads)",
                      "W028 (the shape/height law the two-layer model "
                      "serves)",
                      "T187 / e214 (the archive + the committed per-probe "
                      "records)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "two-wash 124M archive)"],
        "whats_new": ["the JOINT family x height regression (OLS family "
                      "dummies + p0) and its R^2 — never yet computed "
                      "(e215 reported the marginal eta2 and the partial "
                      "separately)",
                      "the per-probe RESIDUAL table on both washes",
                      "the named-split survival tests against the model "
                      "(product Gmail/PlayStation vs iPhone; the tmpl "
                      "width)",
                      "the residual-predictor set: prompt-token overlap "
                      "with the wash train stream, entrenchment p0 (the "
                      "nonlinear leftover), battery position",
                      "the cross-wash residual correlation (physics or "
                      "noise)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)
        return

    load_checks = [cpu_load_check("launch")]

    # ---------------------------------------- P1 the committed records + provenance
    e215p = common.REPO / "runs" / "e215" / "metrics.json"
    e214p = common.REPO / "runs" / "e214" / "metrics.json"
    e214j = common.REPO / "runs" / "e214" / "journal.json"
    for p in (e215p, e214p, e214j, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e215m = json.loads(e215p.read_text(encoding="utf-8"))
    e214m = json.loads(e214p.read_text(encoding="utf-8"))
    e214j = json.loads(e214j.read_text(encoding="utf-8"))["states"]
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    assign = e215m["typology"]["assignment"]
    prov = git_record()
    G_RECORDS = {
        "e215_status": e215m["status"], "n_probes": len(assign),
        "e215_gates": {g: e215m["adjudication"]["gates_summary"][g]
                       for g in ("G_STATES", "G_BATT", "G_CORPUS", "G_ENV")},
        "e215_reprobe_max_dp": e215m["gates"]["G_STATES"]["reprobe_max_dp"],
        "git": prov,
        "note": "the records chain's own certification is inherited (e215 "
                "re-probed both +80 states at dp 0.0 vs e214's records; "
                "e214 certified those vs the e182c/e182c2 originals at "
                "<= 3.3e-06); this cell adds the desk recompute (G_RECOMPUTE)"
                " + git provenance",
    }
    G_RECORDS["pass"] = bool(
        e215m["status"] == "DONE" and len(assign) == 54
        and all(G_RECORDS["e215_gates"].values())
        and not prov["dirty_records"])
    metrics["gates"] = {"G_RECORDS": G_RECORDS}
    log(f"G_RECORDS: {'PASS' if G_RECORDS['pass'] else 'FAIL'} "
        f"(e215 {e215m['status']}, {len(assign)} probes, "
        f"reprobe dp {e215m['gates']['G_STATES']['reprobe_max_dp']}, "
        f"HEAD {prov['head'][:9]})")
    write_metrics("PARTIAL: records read, provenance taken")

    # ------------------------------------------ P2 the desk recompute (G_RECOMPUTE)
    t0j = [r for r in e214j if r["wash"] == "t0"][0]
    cur_w1 = {int(s): e for s, e in e214m["curves"]["w1"].items()}
    cur_w2 = {int(s): e for s, e in e214m["curves"]["w2"].items()}
    assert DEEPEST in cur_w1 and DEEPEST in cur_w2
    max_dp = 0.0
    rows = []
    pos_in_battery: dict[str, int] = {}
    for r in assign:
        b = r["battery"]
        pos_in_battery[b] = pos_in_battery.get(b, -1) + 1
        p0_rc = t0j[b]["probes"][r["fact"]]["p"]
        p80w1_rc = cur_w1[DEEPEST][b]["probes"][r["fact"]]
        p80w2_rc = cur_w2[DEEPEST][b]["probes"][r["fact"]]
        for a, bb in ((p0_rc, r["p0"]), (p80w1_rc, r["p80_w1"]),
                      (p80w2_rc, r["p80_w2"]),
                      (p80w1_rc / p0_rc, r["hr_w1"]),
                      (p80w2_rc / p0_rc, r["hr_w2"])):
            max_dp = max(max_dp, abs(a - bb))
        rows.append(dict(r))                    # the committed row, verbatim
        rows[-1]["battery_pos"] = pos_in_battery[b]
    G_RECOMPUTE = {
        "recomputed_from": "e214 curves w1/w2 +{80} and journal t=0, matched "
                           "per probe by fact+battery",
        "max_dp_vs_e215_assignment": max_dp, "tol": TOL_RECOMPUTE_DP,
        "note": "e215's assignment re-derived from the same committed "
                "records it certified — expected exactly 0",
    }
    G_RECOMPUTE["pass"] = bool(max_dp <= TOL_RECOMPUTE_DP)
    metrics["gates"]["G_RECOMPUTE"] = G_RECOMPUTE
    log(f"G_RECOMPUTE: {'PASS' if G_RECOMPUTE['pass'] else 'FAIL'} "
        f"(max dp {max_dp:.2e} over p0/p80/hr on {len(rows)} probes)")
    write_metrics("PARTIAL: desk recompute certified")
    if not G_RECOMPUTE["pass"] and not SMOKE:
        log("FATAL: committed records failed the desk recompute — aborting")
        return 1

    # ------------------------- P3 the corpus + prompts rebuild (G_CORPUS, G_PROMPT)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(e1.MODEL_REPO,
                                            revision=e1.MODEL_REV)
    metrics["tokenizer"] = {"repo": e1.MODEL_REPO, "revision": e1.MODEL_REV,
                            "note": "the pinned local cache; the ONLY "
                                    "artifact loaded in this cell (no model)"}

    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in e1.POOLS
                     for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    banned_identical = bool(banned == e_banned)
    # the answer ids from the COMMITTED fact battery (the corpus filter's
    # own inputs — no model, no re-screening)
    fact_rows = [r for r in rows if r["battery"] == "fact"]
    answer_ids = {}
    for r in fact_rows:
        a_ids = tok.encode(" " + r["answer"])
        assert len(a_ids) == 1, f"{r['fact']} answer not single-token"
        answer_ids[r["fact"]] = a_ids[0]
    _tid, _bank, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": banned_identical,
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
        "note": "rebuilt TOKENIZER-ONLY (answer ids from the committed fact "
                "battery; e1's filter VERBATIM) — the substrate of the "
                "vocab-overlap predictor; no wash, no model",
    }
    G_CORPUS["pass"] = bool(
        banned_identical
        and G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"]
        and corpus_stats["bank_windows"] == e_corp["bank_windows"])
    metrics["gates"]["G_CORPUS"] = G_CORPUS
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens vs e182 "
        f"{e_corp['tokens_after']}; banned identical {banned_identical})")
    assert G_CORPUS["pass"] or SMOKE, f"corpus rebuild diverged: {G_CORPUS}"
    load_checks.append(cpu_load_check("post_corpus"))
    write_metrics("PARTIAL: corpus rebuilt + certified")

    train_set = set(int(t) for t in _tid.tolist())
    log(f"wash train-stream vocabulary: {len(train_set)} distinct token ids "
        f"over {corpus_stats['train_tokens']} tokens")

    # the four batteries rebuilt VERBATIM by module import (no model):
    # candidate construction only; kept membership from the committed rows.
    fcand, _fd = e1.build_candidates(tok)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), _tid, e_banned, e1.CTRL_POOLS, e1.CTRL_TMPL)
    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), _tid, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), _tid,
                                           e_banned)
    kept_by_battery = {b: [r["fact"] for r in rows if r["battery"] == b]
                       for b in BATTERIES}
    rebuilt = {"fact": [r for r in fcand if r["fact"] in kept_by_battery["fact"]],
               "ctrl": [r for r in ccand if r["fact"] in kept_by_battery["ctrl"]],
               "near": [r for r in ncand if r["fact"] in kept_by_battery["near"]],
               "tmpl": [r for r in tcand if r["fact"] in kept_by_battery["tmpl"]]}
    prompt_ids: dict[str, list[int]] = {}
    G_PROMPT = {"batteries": {}}
    for b in BATTERIES:
        order_equal = [r["fact"] for r in rebuilt[b]] == kept_by_battery[b]
        by_fact = {r["fact"]: r for r in rebuilt[b]}
        max_dp_len = 0
        for r in rows:
            if r["battery"] != b:
                continue
            ids_l = by_fact[r["fact"]]["ids"][0].tolist()
            prompt_ids[r["fact"]] = ids_l
            max_dp_len = max(max_dp_len,
                             abs(len(ids_l) - r["prompt_tok_len"]))
        G_PROMPT["batteries"][b] = {
            "n": len(kept_by_battery[b]), "order_equal_committed": order_equal,
            "max_prompt_toklen_dp": max_dp_len}
    G_PROMPT["note"] = ("candidate construction VERBATIM by module import "
                        "(no model); kept membership from the committed "
                        "rows; order + prompt token lengths cross-checked")
    G_PROMPT["pass"] = bool(all(
        G_PROMPT["batteries"][b]["order_equal_committed"]
        and G_PROMPT["batteries"][b]["max_prompt_toklen_dp"] == 0
        and G_PROMPT["batteries"][b]["n"] > 0 for b in BATTERIES))
    metrics["gates"]["G_PROMPT"] = G_PROMPT
    log(f"G_PROMPT: {'PASS' if G_PROMPT['pass'] else 'FAIL'} "
        + " | ".join(f"{b}: n {G_PROMPT['batteries'][b]['n']}, order "
                     f"{G_PROMPT['batteries'][b]['order_equal_committed']}"
                     for b in BATTERIES))
    write_metrics("PARTIAL: prompts rebuilt + certified")

    # the vocab-overlap predictor + the battery-position predictor
    for r in rows:
        ids_l = prompt_ids[r["fact"]]
        r["vocab_overlap"] = sum(1 for t in ids_l if t in train_set) / len(ids_l)
    n_pos_mismatch = sum(
        1 for b in BATTERIES
        if [r["fact"] for r in rebuilt[b]] != kept_by_battery[b])
    log("vocab overlap (mean per family): "
        + " | ".join(
            f"{f}: {sum(r['vocab_overlap'] for r in rows if r['family6'] == f) / max(1, sum(1 for r in rows if r['family6'] == f)):.3f}"
            for f in FAMILY6_ORDER))

    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "model_loads": 0, "gpu_calls": 0,
             "load_checks": len(load_checks),
             "burst": "tokenizer + corpus filter + candidate construction "
                      "(no model, no probes, no wash)",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV

    # ------------------------------------------------- P4 THE FULL MODEL (part 1)
    fams = [r["family6"] for r in rows]
    p0s = [r["p0"] for r in rows]
    p0_mean = sum(p0s) / len(p0s)
    X_full = design(fams, p0s, p0_mean)
    y_w1 = np.array([r["hr_w1"] for r in rows])
    y_w2 = np.array([r["hr_w2"] for r in rows])
    y_mean = np.array([r["hr_mean"] for r in rows])

    # the ladder beneath the full model
    X_fam = np.column_stack(
        [np.ones(len(fams))] + [np.array([1.0 if g == f else 0.0
                                          for g in fams])
                                for f in FAMILY6_ORDER if f != REF_FAMILY])
    X_p0 = np.column_stack([np.ones(len(fams)), np.array(p0s)])
    fam_r2 = {"w1": ols(X_fam, y_w1)["r2"], "w2": ols(X_fam, y_w2)["r2"],
              "mean": ols(X_fam, y_mean)["r2"]}
    p0_r2 = {"w1": ols(X_p0, y_w1)["r2"], "w2": ols(X_p0, y_w2)["r2"],
             "mean": ols(X_p0, y_mean)["r2"]}

    model = {}
    for tag, y in (("w1", y_w1), ("w2", y_w2), ("mean", y_mean)):
        fit = ols(X_full, y)
        model[tag] = {"r2": fit["r2"], "adj_r2": fit["adj_r2"],
                      "ss_res": fit["ss_res"], "ss_tot": fit["ss_tot"],
                      "resid_sd": fit["resid_sd"], "dof": fit["dof"],
                      "coefs": {n: [float(bv), float(se)] for n, bv, se in
                                zip(coef_names(), fit["beta"], fit["beta_se"])},
                      "fitted": fit["fitted"].tolist(),
                      "resid": fit["resid"].tolist()}
    # attach fitted/resid to the rows
    for i, r in enumerate(rows):
        r["fit_w1"] = model["w1"]["fitted"][i]
        r["fit_w2"] = model["w2"]["fitted"][i]
        r["fit_mean"] = model["mean"]["fitted"][i]
        r["resid_w1"] = model["w1"]["resid"][i]
        r["resid_w2"] = model["w2"]["resid"][i]
        r["resid_mean"] = model["mean"]["resid"][i]

    model["ladder"] = {
        "definition": "R^2 of nested OLS on hr (raw): family-only (5 "
                      "dummies), p0-only, full (family + p0); increments "
                      "= the added R^2 over the nested model",
        "family_only_r2_by_wash": fam_r2, "p0_only_r2_by_wash": p0_r2,
        "delta_family_given_p0": {k: model[k]["r2"] - p0_r2[k]
                                  for k in ("w1", "w2", "mean")},
        "delta_p0_given_family": {k: model[k]["r2"] - fam_r2[k]
                                  for k in ("w1", "w2", "mean")},
        "eta2_crosscheck_vs_e215": e215m["predictor_ladder"]["family_anova"]["eta2"],
    }
    model["ladder"]["family_only_r2"] = fam_r2["w1"]   # w1 headline (co-report)
    metrics["full_model"] = {
        "definition": f"OLS hr_w ~ 1 + 5 family dummies (reference "
                      f"{REF_FAMILY}) + p0 centered at the 54-probe mean "
                      f"{p0_mean:.4f}; raw hr; n=54",
        "primary": "R2 per wash (MODEL-COMPLETE requires BOTH >= 0.75)",
        "R2_bar": R2_BAR,
        "r2": {"w1": model["w1"]["r2"], "w2": model["w2"]["r2"],
               "mean": model["mean"]["r2"]},
        "adj_r2": {"w1": model["w1"]["adj_r2"], "w2": model["w2"]["adj_r2"],
                   "mean": model["mean"]["adj_r2"]},
        "resid_sd": {"w1": model["w1"]["resid_sd"],
                     "w2": model["w2"]["resid_sd"]},
        "dof": model["w1"]["dof"],
        "coefs_w1": model["w1"]["coefs"], "coefs_w2": model["w2"]["coefs"],
        "coefs_mean": model["mean"]["coefs"],
        "ladder": model["ladder"],
    }
    log("THE FULL MODEL (family dummies + p0): R2 w1 "
        f"{model['w1']['r2']:.3f} / w2 {model['w2']['r2']:.3f} / mean "
        f"{model['mean']['r2']:.3f} | family-only {fam_r2['w1']:.3f}/"
        f"{fam_r2['w2']:.3f} + p0 -> +{model['w1']['r2'] - fam_r2['w1']:.3f}/"
        f"+{model['w2']['r2'] - fam_r2['w2']:.3f} | beta_p0 w1 "
        f"{model['w1']['coefs']['p0_centered'][0]:+.3f} w2 "
        f"{model['w2']['coefs']['p0_centered'][0]:+.3f}")
    write_metrics("PARTIAL: the full model fit")

    # ------------------------------------------- P5 THE RESIDUAL STRUCTURE (part 2)
    resid_table = [{"fact": r["fact"], "battery": r["battery"],
                    "family6": r["family6"], "answer": r["answer"],
                    "p0": r["p0"], "hr_w1": r["hr_w1"], "hr_w2": r["hr_w2"],
                    "fit_w1": r["fit_w1"], "fit_w2": r["fit_w2"],
                    "resid_w1": r["resid_w1"], "resid_w2": r["resid_w2"],
                    "vocab_overlap": r["vocab_overlap"],
                    "battery_pos": r["battery_pos"]}
                   for r in rows]

    # the product contrast (the named split #1)
    def _mean(v):
        return sum(v) / len(v) if v else None
    prod = [r for r in rows if r["family6"] == "product"]
    hold_g = [r for r in prod if r["answer"] in PRODUCT_HOLD_GROUP]
    anchor = [r for r in prod if r["answer"] == PRODUCT_COLLAPSE_ANCHOR]
    others = [r for r in prod if r not in hold_g and r not in anchor]
    gap_sd = {}
    for w in ("w1", "w2"):
        gap = (_mean([r[f"resid_{w}"] for r in hold_g])
               - _mean([r[f"resid_{w}"] for r in anchor]))
        gap_sd[w] = gap / model[w]["resid_sd"]
    prod_survives = bool(gap_sd["w1"] >= SPLIT_SD_BAR
                         and gap_sd["w2"] >= SPLIT_SD_BAR)
    product_split = {
        "definition": f"contrast = mean resid{PRODUCT_HOLD_GROUP} - mean "
                      f"resid[{PRODUCT_COLLAPSE_ANCHOR}], in residual-SD "
                      f"units, per wash; SURVIVES iff >= {SPLIT_SD_BAR} SD "
                      f"on both washes (descriptive, never adjudicated)",
        "residuals": {r["answer"]: [r["resid_w1"], r["resid_w2"]]
                      for r in prod},
        "hr_for_reference": {r["answer"]: [r["hr_w1"], r["hr_w2"]]
                             for r in prod},
        "gap_sd_w1": gap_sd["w1"], "gap_sd_w2": gap_sd["w2"],
        "others_mean_resid": {w: _mean([r[f"resid_{w}"] for r in others])
                              for w in ("w1", "w2")},
        "verdict": "SURVIVES (structured)" if prod_survives else
                   "DISSOLVES (the model accounts for it)",
    }

    # the tmpl width + the per-family SD table (the named split #2)
    fam_width = {}
    for f in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == f]
        entry = {"n": len(fr)}
        for w in ("w1", "w2"):
            hrs = [r[f"hr_{w}"] for r in fr]
            rds = [r[f"resid_{w}"] for r in fr]
            sd_hr = (sum((x - _mean(hrs)) ** 2 for x in hrs) / len(hrs)) ** 0.5
            sd_rd = (sum((x - _mean(rds)) ** 2 for x in rds) / len(rds)) ** 0.5
            entry[f"sd_hr_{w}"] = sd_hr
            entry[f"sd_resid_{w}"] = sd_rd
            entry[f"ratio_{w}"] = sd_rd / sd_hr if sd_hr > 0 else None
            entry[f"spearman_p0_hr_{w}"] = spearman(
                [r["p0"] for r in fr], hrs)
        fam_width[f] = entry
    tw = fam_width["rev-capital"]
    tmpl_dissolves = bool(tw["ratio_w1"] is not None and tw["ratio_w2"] is not None
                          and tw["ratio_w1"] <= WIDTH_DISSOLVE_BAR
                          and tw["ratio_w2"] <= WIDTH_DISSOLVE_BAR)
    tmpl_width = {
        "definition": f"within rev-capital: SD(resid)/SD(hr) per wash; "
                      f"DISSOLVES iff <= {WIDTH_DISSOLVE_BAR} on both "
                      f"washes (descriptive; the only within-family "
                      f"regressor is p0, so this is the model's own account "
                      f"of the width under the COMMON p0 slope)",
        "ratio_w1": tw["ratio_w1"], "ratio_w2": tw["ratio_w2"],
        "within_family_spearman_p0_hr": {
            "w1": tw["spearman_p0_hr_w1"], "w2": tw["spearman_p0_hr_w2"]},
        "verdict": "DISSOLVES (p0 accounts for the width)"
                   if tmpl_dissolves else "SURVIVES (width unabsorbed)",
    }
    splits = {"product": product_split, "tmpl_width": tmpl_width,
              "family_width_table": fam_width}
    metrics["residual"] = {
        "table": resid_table,
        "named_splits": splits,
        "largest_abs_resid_w1": sorted(
            ((abs(r["resid_w1"]), r["fact"]) for r in rows),
            reverse=True)[:6],
        "largest_abs_resid_w2": sorted(
            ((abs(r["resid_w2"]), r["fact"]) for r in rows),
            reverse=True)[:6],
    }
    log(f"named splits: product contrast {gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD -> "
        f"{product_split['verdict']}; tmpl width ratio "
        f"{tw['ratio_w1']:.2f}/{tw['ratio_w2']:.2f} -> "
        f"{tmpl_width['verdict']}")
    write_metrics("PARTIAL: residual structure + named splits")

    # ------------------------------------------- P6 RESIDUAL PREDICTORS (part 3)
    preds = {"registered": [], "competitors": []}
    for label, key, note in RESID_PREDICTORS:
        fv = [r[key] for r in rows]
        r1 = spearman(fv, [r["resid_w1"] for r in rows])
        r2_ = spearman(fv, [r["resid_w2"] for r in rows])
        rm = spearman(fv, [r["resid_mean"] for r in rows])
        fires = bool(max(abs(x) for x in (r1, r2_) if x is not None)
                     > PRED_BAR) if (r1 is not None or r2_ is not None) else False
        preds["registered"].append({
            "label": label, "key": key, "note": note,
            "value_range": [min(fv), max(fv)],
            "rho_w1": r1, "rho_w2": r2_, "rho_mean": rm,
            "fires": fires,
            "firing_stat": "max |rho| over the two washes vs "
                           f"{PRED_BAR}"})
        log(f"resid predictor {label:18s}: rho w1 "
            f"{r1 if r1 is None else round(r1, 3)} / w2 "
            f"{r2_ if r2_ is None else round(r2_, 3)}"
            f"{'  FIRES' if fires else ''}")
    for label, key in RESID_COMPETITORS:
        fv = [r[key] for r in rows]
        preds["competitors"].append({
            "label": label, "key": key,
            "note": "co-report competitor (never adjudicated)"
                    + (" — family-level (constant within family): orders the "
                       "families' mean residuals" if key == "rel_cue_freq_per_1m"
                       else ""),
            "rho_w1": spearman(fv, [r["resid_w1"] for r in rows]),
            "rho_w2": spearman(fv, [r["resid_w2"] for r in rows]),
            "rho_mean": spearman(fv, [r["resid_mean"] for r in rows])})
    metrics["residual_predictors"] = preds
    write_metrics("PARTIAL: residual predictors")

    # --------------------------------------- P7 CROSS-WASH RESIDUAL (part 4)
    xw = {
        "spearman": spearman([r["resid_w1"] for r in rows],
                             [r["resid_w2"] for r in rows]),
        "pearson": _pearson([r["resid_w1"] for r in rows],
                            [r["resid_w2"] for r in rows]),
        "per_family": {f: spearman([r["resid_w1"] for r in rows
                                    if r["family6"] == f],
                                   [r["resid_w2"] for r in rows
                                    if r["family6"] == f])
                       for f in FAMILY6_ORDER},
        "for_reference_hr_level": e215m["reliability"]["spearman_pooled"],
        "note": "the e215 reliability convention (Spearman primary); a high "
                "rho means the model's per-probe misses REPLICATE across "
                "washes (physics or shared texture — see honesty)",
    }
    metrics["cross_wash_residual"] = xw
    log(f"cross-wash residual: Spearman {xw['spearman']:.3f} / Pearson "
        f"{xw['pearson']:.3f} (hr-level was "
        f"{e215m['reliability']['spearman_pooled']:.3f}); per family: "
        + " ".join(f"{f.split('-')[0]} "
                   f"{'--' if xw['per_family'][f] is None else format(xw['per_family'][f], '.2f')}"
                   for f in FAMILY6_ORDER))
    write_metrics("PARTIAL: cross-wash residual")

    # ----------------------------------------------- P8 ADJUDICATION (frozen)
    r2_ok = bool(model["w1"]["r2"] >= R2_BAR and model["w2"]["r2"] >= R2_BAR)
    firing = [f for f in preds["registered"] if f["fires"]]
    xw_rho = xw["spearman"]
    xw_low_ok = bool(xw_rho is not None and abs(xw_rho) < XWASH_LOW)
    xw_high = bool(xw_rho is not None and xw_rho > XWASH_HIGH)
    gates_ok = bool(G_RECORDS["pass"] and G_RECOMPUTE["pass"]
                    and G_CORPUS["pass"] and G_PROMPT["pass"]
                    and G_ENV["pass"])

    candidates = ([(abs(f["rho_w1"]) if f["rho_w1"] is not None else 0,
                     abs(f["rho_w2"]) if f["rho_w2"] is not None else 0,
                     f["label"], "predictor") for f in firing]
                  + ([(xw_rho, xw_rho, "the cross-wash residual rho",
                       "cross-wash")] if xw_high else []))

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_RECORDS", G_RECORDS["pass"]),
                           ("G_RECOMPUTE", G_RECOMPUTE["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_PROMPT", G_PROMPT["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif r2_ok and not firing and xw_low_ok:
        verdict = "MODEL-COMPLETE"
        clause = (f"the full model's R-squared clears the bar on both washes "
                  f"(w1 {model['w1']['r2']:.3f}, w2 {model['w2']['r2']:.3f} "
                  f">= {R2_BAR:g}; family-only eta2 "
                  f"{fam_r2['w1']:.3f}/{fam_r2['w2']:.3f}, the p0 term adds "
                  f"+{model['w1']['r2'] - fam_r2['w1']:.3f}/"
                  f"+{model['w2']['r2'] - fam_r2['w2']:.3f}) AND the "
                  f"residuals are unstructured (no registered predictor "
                  f"|rho| > {PRED_BAR} — best "
                  f"{max((max(abs(f['rho_w1']) if f['rho_w1'] is not None else 0, abs(f['rho_w2']) if f['rho_w2'] is not None else 0) for f in preds['registered'])):.3f}"
                  f"; cross-wash residual rho {xw_rho:.3f} < {XWASH_LOW}) — "
                  f"FAMILY x HEIGHT is the whole sorting key; the named "
                  f"splits dissolve into it")
    elif firing or xw_high:
        verdict = "RESIDUAL-STRUCTURED"
        best = max(candidates, key=lambda c: max(c[0], c[1]))
        if best[3] == "predictor":
            named = (f"the best candidate: {best[2]} "
                     f"(rho w1 {best[0]:.3f} / w2 {best[1]:.3f})")
        else:
            named = f"the best candidate: {best[2]} (rho {xw_rho:.3f})"
        r2_txt = (f"R-squared {'cleared' if r2_ok else 'FAILED'} the bar "
                  f"(w1 {model['w1']['r2']:.3f}, w2 {model['w2']['r2']:.3f} "
                  f"vs {R2_BAR:g}); ")
        struct_txt = []
        if firing:
            struct_txt.append(
                "predictors fire: " + ", ".join(
                    f"{f['label']} (w1 "
                    f"{f['rho_w1']:.3f} / w2 {f['rho_w2']:.3f})"
                    for f in firing))
        if xw_high:
            struct_txt.append(f"the cross-wash residual rho {xw_rho:.3f} > "
                              f"{XWASH_HIGH} — the model's per-probe misses "
                              f"replicate across washes")
        clause = (f"the residuals carry structure ({'; '.join(struct_txt)}) "
                  f"— a third sorting dimension exists; named with {named}; "
                  + r2_txt
                  + f"the named splits: product {product_split['verdict']}, "
                  f"tmpl width {tmpl_width['verdict']}")
    else:
        why = []
        if not r2_ok:
            why.append(f"the R-squared clause failed (w1 "
                       f"{model['w1']['r2']:.3f}, w2 {model['w2']['r2']:.3f} "
                       f"vs the {R2_BAR:g} bar — family x height leaves the "
                       f"registered share of variance unexplained)")
        if not firing:
            why.append(f"no registered residual predictor fires (best "
                       f"{max((max(abs(f['rho_w1']) if f['rho_w1'] is not None else 0, abs(f['rho_w2']) if f['rho_w2'] is not None else 0) for f in preds['registered'])):.3f} "
                       f"<= {PRED_BAR})")
        if not xw_high:
            why.append(f"the cross-wash residual rho {xw_rho:.3f} <= "
                       f"{XWASH_HIGH} (the residual is not wash-replicating "
                       f"at the structured level)")
        verdict = "GRADED"
        clause = ("any partial — " + "; ".join(why)
                  + f"; the named splits: product {product_split['verdict']} "
                  f"(contrast {gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD), "
                  f"tmpl width {tmpl_width['verdict']} (ratio "
                  f"{tw['ratio_w1']:.2f}/{tw['ratio_w2']:.2f}); the tables "
                  f"verbatim")

    gates_summary = {"G_RECORDS": G_RECORDS["pass"],
                     "G_RECOMPUTE": G_RECOMPUTE["pass"],
                     "G_CORPUS": G_CORPUS["pass"],
                     "G_PROMPT": G_PROMPT["pass"], "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {"MODEL_COMPLETE": bool(r2_ok and not firing and xw_low_ok
                                        and gates_ok and not SMOKE),
                 "RESIDUAL_STRUCTURED": bool((firing or xw_high)
                                             and gates_ok and not SMOKE
                                             and not (r2_ok and not firing
                                                      and xw_low_ok)),
                 "verdict": verdict, "clause": clause,
                 "order": "MODEL-COMPLETE -> RESIDUAL-STRUCTURED -> GRADED "
                          "(gated on G_RECORDS/G_RECOMPUTE/G_CORPUS/G_PROMPT/"
                          "G_ENV)"},
        "clauses": {
            "r2_both_washes >= 0.75": r2_ok,
            "no_predictor_fires (|rho| <= 0.30)": not firing,
            "crosswash_spearman < 0.30": xw_low_ok,
            "crosswash_spearman > 0.50": xw_high,
            "firing_predictors": [f["label"] for f in firing],
        },
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E216 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    # ------------------------------------------------ P9 honesty + close
    metrics["honesty_reflex"] = {
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "inherited from the archive (disclosed)",
        "typology_inherited": "the family factor is e215's HAND-REGISTERED "
                              "typology (not blind to the outcome — the "
                              "grouping follows the instrument's pools, "
                              "disclosed there); the full model's R^2 is "
                              "measured AGAINST that hand-grouped factor — "
                              "'family x height explains X%' inherits the "
                              "typology's subjectivity wholesale",
        "ols_conventions": "raw bounded hr (no transform), plain OLS, "
                           "homoskedastic SEs (co-reported, never "
                           "adjudicated); unbalanced family n's (3..19) and "
                           "heteroskedastic family widths make the plain R^2 "
                           "generous in the wide families — the adj-R^2 "
                           "beside it",
        "entrenchment_collinear": "the entrenchment predictor IS the model's "
                                  "own p0 term (the dispatch's proxy): its "
                                  "residual Spearman is a NONLINEARITY test, "
                                  "not an independent dimension — a firing "
                                  "means the linear height term mis-fits "
                                  "(e.g. floor compression), disclosed "
                                  "wherever read",
        "xwash_reading": "a cross-wash residual correlation could be probe "
                         "physics OR shared texture (both arms divide by "
                         "the same p0; both share the wash corpus and the "
                         "prompt strings) — the correlation replicates the "
                         "MISS, its MECHANISM stays open",
        "vocab_overlap_surface": "the overlap feature is a token-id surface "
                                 "statistic (fraction of prompt ids seen in "
                                 "the wash train stream) — it conflates "
                                 "syntax frequency with content words; "
                                 "co-reported per-family means so the table "
                                 "is transparent",
        "position_surface": "battery position is the instrument's own pool "
                            "order (fact: cap->lang->cur blocks; ctrl: "
                            "pool listing) — an ordering-artifact check, "
                            "not a cognitive variable",
        "hr_ratio_floor": "hr divides by p0 as low as 0.52; deep-state floors "
                          "compress ratios — hr, p80 and fitted are "
                          "co-reported per probe in the residual table",
        "guarantees_nothing": "the model and its residual describe THESE 54 "
                              "probes under THESE two washes with an "
                              "inherited hand-registered typology — the "
                              "openness is the point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, ZERO model loads (tokenizer "
                    "only), decisive; no GPU calls",
        "load_checks": load_checks,
        "records_chain": ["runs/e214/metrics.json", "runs/e214/journal.json",
                          "runs/e215/metrics.json"],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    if n_pos_mismatch:
        metrics["deviations"].append(
            f"battery-order mismatch count {n_pos_mismatch} (position "
            f"predictor read from the committed assignment order; G_PROMPT "
            f"flags it)")

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, rows, {**model, "ladder": model["ladder"]},
                        splits, preds, xw, adj)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
