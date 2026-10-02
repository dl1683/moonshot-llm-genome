"""E218 — THE NONLINEAR HEIGHT TEST (T190's named cut).

WHY: e216/T190 closed the within-family residual cell with a cut, not an
answer: family x height (p0 linear) explains only ~2/3 (R2 0.671/0.621)
and the per-probe misses REPLICATE across washes at Spearman 0.934 — a
THIRD SORTING DIMENSION, unnamed by every registered surface feature
(vocab overlap, entrenchment-leftover, battery position; all |rho| <<
0.30). T190's named cut: is that third dimension NONLINEAR WITHIN-FAMILY
HEIGHT (the p0 term in a nonlinear dress — rank or logit) or GENUINELY
NEW PROBE FEATURES (something no function of p0 carries)?

THE CELL (a pure desk cell on the committed 54-probe records — no model
loads, no re-probes, no tokenizer; JSON records only):
  (1) THE EXPANDED MODELS, per wash — hr ~ family + HEIGHT, with HEIGHT in
      four registered forms: p0-LINEAR (the e216 baseline, reproduced here
      and bit-checked against e216's committed fit by G_BASELINE); p0-RANK
      (the within-family rank of p0 — the height as ORDER, not magnitude);
      logit(p0) (the height stretched near the ceiling); p0 + p0^2 (the
      height with curvature). Each form's R2 compared, per wash.
  (2) THE DECISIVE TEST — the cross-wash RESIDUAL correlation under each
      expanded model: if some height form absorbs the 0.934 (drops it
      below ~0.5), THE THIRD DIMENSION IS NONLINEAR HEIGHT; if the
      residual correlation survives every height form, the dimension is
      genuinely beyond height.
  (3) THE NAMED SPLITS' FATE under the best model — the product contrast
      (Gmail/PlayStation hold vs iPhone collapses) and the tmpl width
      (rev-capital's spread): absorbed or surviving.

REGISTERED BARS (frozen VERBATIM from the dispatch brief, QUEUE row
DISPATCHED 16:17Z / commit 7493a55, BEFORE any compute; adjudicate
against exactly this; no bar shopping):
  - HEIGHT-NONLINEAR: "fires if some height form absorbs the cross-wash
    residual (rho < 0.5) with R2 >= 0.75 — the third dimension IS
    nonlinear height; the family x height model completed in its
    nonlinear form."
  - BEYOND-HEIGHT: "fires if the cross-wash residual rho stays >= 0.5
    under every height form — the third dimension is genuinely new; the
    probe-feature hunt owed (named candidates: the answer's base-model
    entrenchment measured independently, the probe's internal token
    structure, the relation's compositionality)."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * DATA = runs/e215/metrics.json typology.assignment (the committed
    54-probe records; the certification chain inherited and re-certified
    desk-only: G_RECORDS [e215 DONE + its 4 gates + e216 DONE + its 5
    gates + git provenance + clean records chain], G_RECOMPUTE [hr/p0
    recomputed from e214's curves + journal t=0 at 1e-9], and G_BASELINE
    [this cell's own p0-linear arm must reproduce e216's committed
    full-model fit — R2 / adj-R2 / resid SD / per-probe residuals / the
    cross-wash Spearman — at 1e-9; the baseline is e216 REPRODUCED, not
    repeated]).
  * OUTCOME = hr_w = p_w(+80)/p0 per wash (the e216 convention; hr_mean
    co-reported).
  * THE FOUR REGISTERED HEIGHT FORMS (the bar's "height form"):
      linear : OLS hr ~ 1 + 5 family dummies (reference lang) + p0
               centered at the committed 54-probe mean (e216 VERBATIM);
      rank   : + the WITHIN-FAMILY fractional rank of p0 (average ranks
               under ties, 1-based, divided by the family's n — the rank
               scale equalized across unbalanced families; with family
               dummies present any common affine transform is
               fit-equivalent, so only the /n_f scaling choice matters);
      logit  : + logit(p0) centered (p0 in 0.52..0.99 -> 0.08..4.66; the
               ceiling stretched, the floor compressed);
      quad   : + p0 centered AND (p0 centered)^2.
    R2 = 1 - SSres/SStot, PRIMARY per wash; adj-R2 / resid SD co-reported.
  * CROSS-WASH RESIDUAL = Spearman(resid_w1, resid_w2) over the 54 probes
    (the e215/e216 reliability convention; Pearson + per-family
    co-reported). ABSORPTION is read as |rho| < 0.5 (the bar's "(rho <
    0.5)": a negatively-replicating miss is not absorption either —
    e216's abs-convention on the low side, inherited, disclosed);
    SURVIVAL is read as rho >= 0.5 (positive replication).
  * ADJUDICATION ORDER: gates ok -> HEIGHT-NONLINEAR iff SOME registered
    form has (|xwash Spearman| < 0.5 AND R2_w1 >= 0.75 AND R2_w2 >= 0.75 —
    the R2 clause inherits e216's both-washes convention, on the SAME
    form); else BEYOND-HEIGHT iff xwash Spearman >= 0.5 under EVERY
    registered form; else GRADED.
  * BEST MODEL (for the named splits) = if HEIGHT-NONLINEAR fires, the
    firing form with the highest mean R2; else the form with the LOWEST
    cross-wash Spearman (ties: higher mean R2). Deterministic.
  * NAMED SPLITS under the best model (the e216 descriptive yardsticks,
    frozen, never adjudicated): the PRODUCT CONTRAST = mean
    resid{Gmail, PlayStation} - mean resid{iPhone}, per wash, in units of
    that wash's residual SD; "SURVIVES" iff >= 1.0 SD on BOTH washes. THE
    TMPL WIDTH = SD(resid)/SD(hr) within rev-capital per wash;
    "DISSOLVES" iff <= 0.5 on both washes.
  * CO-REPORT COMPETITORS (never adjudicated): the RAW-rank form (family
    + the unnormalized within-family rank 1..n_f) and the FAMILY-SLOPES
    form (family + family x p0 interactions — family-specific linear
    height slopes; the one height functional the registered set does
    NOT contain, co-reported so the BEYOND-HEIGHT reading knows its own
    boundary).

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_RECORDS — e215 DONE n=54 with its gates (G_STATES/G_BATT/G_CORPUS/
    G_ENV) PASS; e216 DONE with its gates (G_RECORDS/G_RECOMPUTE/G_CORPUS/
    G_PROMPT/G_ENV) PASS; git provenance recorded (HEAD + the last
    commits touching the records chain e214/e215/e216 metrics+journal +
    a clean-tree check on that chain).
  * G_RECOMPUTE — p0, p80_w1, p80_w2, hr_w1, hr_w2 recomputed from
    e214's committed curves + journal t=0, matched per probe by
    fact+battery: max |dp| <= 1e-9 (the e216 recompute, verbatim).
  * G_BASELINE — this cell's linear form vs e216's committed full model:
    max |dp| <= 1e-9 over R2/adj-R2/resid-SD (both washes + mean), the
    54 per-probe residuals (both washes), and the cross-wash Spearman +
    Pearson. The baseline is certified as a reproduction before any new
    form is read.
  * G_ENV — the owner envelope: CPU-only (zero GPU calls), torch threads
    <= 4, load checks, ZERO model loads AND zero tokenizer loads (JSON
    desk only), decisive.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. n=2 washes is texture, not law (wash 1 a CPU fp32 replay,
wash 2 GPU fp32 — inherited, disclosed). The typology is HAND-REGISTERED
and not blind to the outcome (inherited from e215, disclosed). The four
registered height forms are all FUNCTIONS OF p0: rank is the
within-family monotone of p0 (LESS information than p0 itself), logit
and quad are global transforms — a BEYOND-HEIGHT verdict means "no
registered function of p0 + family absorbs the replicating residual",
which is T190's cut as dispatched, NOT a proof that no height functional
exists (the family-slopes competitor sits beside it, never adjudicated).
Four forms compared on the same 54 probes is a multiple-comparison
surface; only the frozen clauses adjudicate. logit stretches the ceiling
(few probes near p0 ~ 0.99 dominate its variance). The named-split
conventions (1 SD, 0.5 ratio) are descriptive yardsticks frozen before
compute, not bars. A residual correlated across washes could be probe
physics OR shared pipeline texture (the same p0 denominator in both
arms) — the correlation replicates the MISS, its mechanism stays open.

PROVENANCE: the per-probe records, typology and the committed
certification chain are e215's, and the baseline model + its committed
fit are e216's (both read here, never rewritten; G_BASELINE bit-checks
the latter). The curves, journal and checkpoint inventory are e214's;
the two-wash archive descends from e182c/e182c2/e182 (T149/T183/T123).
Builds on: T190 / e216 (the named cut; the rho 0.934 this cell tries to
absorb), T189 / e215 (the family x height model + the records), T187 /
e214 (the archive), W028 (the shape/height law). NEW: the three
nonlinear height forms (within-family rank / logit / quadratic); the
cross-wash ABSORPTION test (the decisive statistic per form); the named
splits re-read under the best form; the raw-rank and family-slopes
competitors; the G_BASELINE reproduction gate.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and at the mid phase; ZERO model
loads, ZERO tokenizer loads (JSON desk only); no wash, no probes, no
NOTES/THINKING/QUEUE/STATE edits (dispatch). Progressive PARTIAL metrics
after every phase (the standing disruption rule).

Run:  cd lab && python e218_nonlinear.py   (E218_SMOKE=1: same desk
      tables on the same committed records, own smoke dir, nothing
      adjudicated — the e215/e216 smoke disclosure applies: a desk
      shakedown inevitably displays the committed tables)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import common                                               # noqa: E402
from common import now_iso, run_dir, save_json              # noqa: E402

torch.set_num_threads(4)                                    # the owner envelope (this dispatch: <= 4)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import textwrap                                             # noqa: E402

try:      # Windows console safety for the lab's typographic register
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                           # noqa: BLE001
    pass

SMOKE = os.environ.get("E218_SMOKE") == "1"
NAME = "e218_smoke" if SMOKE else "e218"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants (registered BEFORE compute) ---------------------
DEEPEST = 80                    # the deepest state present on BOTH washes (e216 convention)
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
FAMILY6_ORDER = ["lang", "cap-cur", "founder-anchor", "product",
                 "near-uscap", "rev-capital"]
REF_FAMILY = "lang"             # the treatment-coding reference (largest n=10; e216 convention)

# the registered height forms (the bar's "height form" set — frozen)
FORMS: tuple[str, ...] = ("linear", "rank", "logit", "quad")
FORM_LABELS = {
    "linear": "family + p0 (LINEAR — the e216 baseline)",
    "rank":   "family + within-family RANK of p0 (fractional, /n_f)",
    "logit":  "family + logit(p0)",
    "quad":   "family + p0 + p0^2",
}
# co-report competitor forms (never adjudicated)
COMPETITOR_FORMS: tuple[str, ...] = ("rawrank", "fslopes")
COMPETITOR_LABELS = {
    "rawrank": "family + RAW within-family rank (1..n_f, unnormalized)",
    "fslopes": "family + family x p0 (family-specific linear slopes)",
}

# the registered bar constants (frozen)
R2_BAR = 0.75          # "with R2 >= 0.75" -> BOTH washes, on the SAME form (e216 convention)
XWASH_ABSORB = 0.5     # "absorbs the cross-wash residual (rho < 0.5)" -> |rho| < 0.5 (disclosed)
XWASH_SURVIVE = 0.5    # "stays >= 0.5 under every height form"

# the named-split descriptive yardsticks (frozen; never adjudicated; e216 verbatim)
PRODUCT_HOLD_GROUP = ["Gmail", "PlayStation"]   # the dispatch's named holds
PRODUCT_COLLAPSE_ANCHOR = "iPhone"              # the dispatch's named collapse
SPLIT_SD_BAR = 1.0     # product contrast SURVIVES iff >= 1.0 residual SD both washes
WIDTH_DISSOLVE_BAR = 0.5   # tmpl width DISSOLVES iff SD(resid)/SD(hr) <= 0.5 both washes

# registered verification tolerances (frozen)
TOL_RECOMPUTE_DP = 1e-9
TOL_BASELINE_DP = 1e-9

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "HEIGHT-NONLINEAR": "fires if some height form absorbs the cross-wash "
            "residual (rho < 0.5) with R2 >= 0.75 — the third dimension IS "
            "nonlinear height; the family x height model completed in its "
            "nonlinear form.",
        "BEYOND-HEIGHT": "fires if the cross-wash residual rho stays >= 0.5 "
            "under every height form — the third dimension is genuinely new; "
            "the probe-feature hunt owed (named candidates: the answer's "
            "base-model entrenchment measured independently, the probe's "
            "internal token structure, the relation's compositionality).",
        "GRADED": "any partial — the tables verbatim.",
    },
    "operationalizations": (
        "data = e215's committed 54-probe assignment (re-certified desk-only "
        "by G_RECORDS [e215+e216 DONE, gates pass, git chain clean] + "
        "G_RECOMPUTE vs e214's curves+journal at 1e-9 + G_BASELINE: this "
        "cell's linear form reproduces e216's committed fit at 1e-9); "
        "outcome = hr_w = p_w(+80)/p0 per wash; the FOUR registered height "
        "forms = linear (e216 verbatim), rank (within-family fractional rank "
        "of p0, avg-ranks/n_f), logit(p0), quad (p0 + p0^2), each "
        "family-dummy OLS (reference lang), R2 per wash primary; CROSS-WASH "
        "RESIDUAL = Spearman(resid_w1, resid_w2) per form (Pearson + "
        "per-family co-reported); ABSORPTION read as |rho| < 0.5 (the bar's "
        "'(rho < 0.5)' — a negatively-replicating miss is not absorption "
        "either; e216's abs-convention inherited, disclosed), SURVIVAL as "
        "rho >= 0.5; ADJUDICATION: gates ok -> HEIGHT-NONLINEAR iff some "
        "registered form has (|xwash| < 0.5 AND R2 >= 0.75 on BOTH washes, "
        "same form); else BEYOND-HEIGHT iff xwash >= 0.5 under EVERY "
        "registered form; else GRADED; BEST = the firing form with highest "
        "mean R2 if HEIGHT-NONLINEAR fires, else the lowest-xwash form "
        "(ties: higher mean R2); named splits under BEST with the e216 "
        "descriptive yardsticks (product contrast >= 1.0 SD both washes "
        "SURVIVES; tmpl SD(resid)/SD(hr) <= 0.5 both washes DISSOLVES); "
        "competitors rawrank + family-slopes co-reported, never adjudicated; "
        "gated on G_RECORDS/G_RECOMPUTE/G_BASELINE/G_ENV"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (QUEUE "
                     "row DISPATCHED 16:17Z, commit 7493a55) BEFORE any "
                     "compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "PURE DESK cell, one step lighter than e216: NO model loads AND NO "
    "tokenizer (JSON records only) — the provenance chain enters as e215's "
    "and e216's committed certifications (e215 re-probed both +80 states at "
    "dp 0.0 vs e214's records, which e214 certified vs the e182c/e182c2 "
    "originals at <= 3.3e-06; e216 added its corpus/prompt rebuild "
    "certifications) PLUS this cell's desk recompute (G_RECOMPUTE), git "
    "provenance (G_RECORDS) and the bit-checked baseline reproduction "
    "(G_BASELINE).",
    "The rank form's /n_f scaling (fractional rank in (0,1]) is the "
    "registered choice — with family dummies present any COMMON affine "
    "transform is fit-equivalent, so the unnormalized raw-rank variant sits "
    "in the co-report competitors rather than the registered set.",
    "The absorption convention: the bar's '(rho < 0.5)' is read as |rho| "
    "< 0.5 (a negatively-replicating miss is not absorption either); "
    "survival is read as rho >= 0.5 — both disclosed before compute.",
    "The family-slopes form (family x p0 interactions) is the one height "
    "functional the registered set does NOT contain; it is co-reported as a "
    "competitor so the BEYOND-HEIGHT reading knows its own boundary — never "
    "adjudicated (no bar shopping).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: the same desk tables on the same committed records (the "
    "e215/e216 smoke disclosure applies — a desk shakedown inevitably "
    "displays the committed tables), own smoke dir, nothing adjudicated.",
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


def _dummy_cols(fams: list[str]) -> list[np.ndarray]:
    return [np.array([1.0 if g == f else 0.0 for g in fams])
            for f in FAMILY6_ORDER if f != REF_FAMILY]


def design(form: str, fams: list[str], p0s: list[float],
           rank_frac: list[float], raw_rank: list[float],
           p0_mean: float) -> np.ndarray:
    """The per-form design: [1, 5 family dummies (ref = lang), HEIGHT...]."""
    cols = [np.ones(len(fams))] + _dummy_cols(fams)
    p0c = np.array(p0s) - p0_mean
    if form == "linear":
        cols.append(p0c)
    elif form == "rank":
        cols.append(np.array(rank_frac) - np.mean(rank_frac))
    elif form == "logit":
        lg = np.log(np.clip(np.array(p0s), 1e-12, 1 - 1e-12)
                    / np.clip(1 - np.array(p0s), 1e-12, 1.0))
        cols.append(lg - lg.mean())
    elif form == "quad":
        cols.append(p0c)
        cols.append(p0c ** 2)
    elif form == "rawrank":
        cols.append(np.array(raw_rank) - np.mean(raw_rank))
    elif form == "fslopes":
        cols.append(p0c)
        for d in _dummy_cols(fams):
            cols.append(d * p0c)
    else:
        raise ValueError(form)
    return np.column_stack(cols)


def form_coef_names(form: str) -> list[str]:
    names = (["intercept(lang)"]
             + [f"D[{f}]" for f in FAMILY6_ORDER if f != REF_FAMILY])
    return {"linear": names + ["p0_centered"],
            "rank":   names + ["rank_frac_centered"],
            "logit":  names + ["logit_p0_centered"],
            "quad":   names + ["p0_centered", "p0_centered_sq"],
            "rawrank": names + ["raw_rank_centered"],
            "fslopes": names + ["p0_centered"]
                       + [f"D[{f}]:p0" for f in FAMILY6_ORDER
                          if f != REF_FAMILY]}[form]


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
             "runs/e215/metrics.json", "runs/e216/metrics.json"]
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


def make_plot(rd, rows, table, decisive, splits, best, adj, comp_table):
    """THE FIGURE: (a) the model comparison (R2 per form, the bar);
  (b) THE DECISIVE TEST (cross-wash residual rho per form, the two bars);
  (c) the residuals by family under the BEST form (named splits annotated);
  (d) the cross-wash residual scatter under the BEST form + verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) the model comparison: R2 per form per wash
    ax = axes[0, 0]
    all_forms = list(FORMS) + list(COMPETITOR_FORMS)
    xs = np.arange(len(FORMS))
    for k, (w, col) in enumerate((("w1", "tab:blue"), ("w2", "tab:cyan"),
                                  ("mean", "0.55"))):
        vals = [table[f]["r2"][w] for f in FORMS]
        bars = ax.bar(xs + (k - 1) * 0.26, vals, width=0.24, color=col,
                      edgecolor="k", lw=0.5,
                      label=f"R2 {w}" + (" (co-report)" if w == "mean" else ""))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=6.4)
    ax.axhline(R2_BAR, color="tab:red", ls="--", lw=1.4,
               label=f"R2 bar {R2_BAR:g} (both washes, same form)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}\n({table[f]['n_params']} params)"
                        for f in FORMS], fontsize=8.5)
    ax.set_ylabel("R^2 of hr ~ family + HEIGHT form")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("(1) THE EXPANDED MODELS — R2 per height form (linear = the "
                 "e216 baseline, bit-checked)", fontsize=9.0)

    # (0,1) THE DECISIVE TEST: xwash residual rho per form
    ax = axes[0, 1]
    xs = np.arange(len(FORMS))
    vals = [decisive[f]["spearman"] for f in FORMS]
    bars = ax.bar(xs, vals, width=0.5, color=["0.45" if f == "linear"
                                              else "tab:orange" for f in FORMS],
                  edgecolor="k", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=7.6)
    # the competitors as short reference whiskers (co-report, never adjudicated)
    for cf in COMPETITOR_FORMS:
        v = comp_table[cf]["xwash_spearman"]
        ax.hlines(v, -0.45, 1.9, color="0.35", ls="-.", lw=1.1)
        ax.text(-0.44, v + 0.015, f"{cf}: {v:.3f}", fontsize=6.6,
                color="0.25")
    ax.axhline(XWASH_SURVIVE, color="darkred", ls="--", lw=1.4,
               label=f"absorption |rho| < {XWASH_ABSORB:g} / survival rho >= {XWASH_SURVIVE:g}")
    ax.axhline(0.3, color="tab:red", ls=":", lw=1.1,
               label="e216 noise line 0.3 (reference)")
    ax.set_xticks(xs)
    ax.set_xticklabels(FORMS, fontsize=8.5)
    ax.set_ylabel("Spearman(resid_w1, resid_w2)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=6.6, loc="center right")
    ax.set_title("(2) THE DECISIVE TEST — does any height form absorb the "
                 "cross-wash residual? (diamonds = competitors, co-report)",
                 fontsize=9.0)

    # (1,0) the residuals by family under the BEST form
    ax = axes[1, 0]
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
    ax.set_ylabel(f"RESIDUAL under the BEST form ({best})")
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
    ax.set_title("(3) THE NAMED SPLITS under the best form — product contrast "
                 f"{splits['product']['gap_sd_w1']:+.2f}/"
                 f"{splits['product']['gap_sd_w2']:+.2f} SD "
                 f"({splits['product']['verdict']}); tmpl width ratio "
                 f"{splits['tmpl_width']['ratio_w1']:.2f}/"
                 f"{splits['tmpl_width']['ratio_w2']:.2f} "
                 f"({splits['tmpl_width']['verdict']})", fontsize=9.0)

    # (1,1) cross-wash residual scatter under the BEST form + verdict strip
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
    ax.set_xlabel(f"residual, wash 1 (form: {best})")
    ax.set_ylabel(f"residual, wash 2 (form: {best})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, loc="lower right")
    ax.set_title(f"(4) THE CROSS-WASH RESIDUAL under the best form — "
                 f"Spearman {decisive[best]['spearman']:.3f} / Pearson "
                 f"{decisive[best]['pearson']:.3f} "
                 f"(absorption |rho| < {XWASH_ABSORB:g})", fontsize=9.0)

    # the verdict strip under (4)
    y = -0.155
    ax.text(0.0, y - 0.045, f"E218 VERDICT: {adj['bars']['verdict']}",
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

    fig.suptitle("E218 — THE NONLINEAR HEIGHT TEST (T190's cut): nonlinear "
                 "within-family height or genuinely new probe features? -> "
                 + adj["bars"]["verdict"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "nonlinear.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E218 — THE NONLINEAR HEIGHT TEST (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e218_nonlinear",
        "phase": "pure desk on the committed 54-probe records (no model "
                 "loads, no re-probes, no tokenizer — JSON records only)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T190's named cut: is the third sorting dimension "
                     "NONLINEAR WITHIN-FAMILY HEIGHT (p0 as rank or logit) "
                     "or GENUINELY NEW probe features? (1) the expanded "
                     "models' R2 per wash; (2) the decisive test — the "
                     "cross-wash residual correlation under each form; "
                     "(3) the named splits' fate under the best form"),
        "builds_on": ["T190 / e216 (the named cut; the rho 0.934 this cell "
                      "tries to absorb; the baseline model reproduced via "
                      "G_BASELINE)",
                      "T189 / e215 (the family x height model + the "
                      "committed 54-probe records)",
                      "T187 / e214 (the archive + the committed curves)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "two-wash 124M archive)",
                      "W028 (the shape/height law)"],
        "whats_new": ["the three NONLINEAR height forms (within-family rank "
                      "/ logit / quadratic) beside the e216 linear baseline",
                      "the cross-wash ABSORPTION test — the decisive "
                      "statistic computed per form",
                      "the named splits re-read under the best form",
                      "the raw-rank and family-slopes competitor forms "
                      "(co-report, never adjudicated)",
                      "the G_BASELINE reproduction gate (the linear arm "
                      "bit-checked against e216's committed fit at 1e-9)"],
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
    e216p = common.REPO / "runs" / "e216" / "metrics.json"
    e214p = common.REPO / "runs" / "e214" / "metrics.json"
    e214j = common.REPO / "runs" / "e214" / "journal.json"
    for p in (e215p, e216p, e214p, e214j):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e215m = json.loads(e215p.read_text(encoding="utf-8"))
    e216m = json.loads(e216p.read_text(encoding="utf-8"))
    e214m = json.loads(e214p.read_text(encoding="utf-8"))
    e214j = json.loads(e214j.read_text(encoding="utf-8"))["states"]
    assign = e215m["typology"]["assignment"]
    prov = git_record()
    G_RECORDS = {
        "e215_status": e215m["status"], "n_probes": len(assign),
        "e215_gates": {g: e215m["adjudication"]["gates_summary"][g]
                       for g in ("G_STATES", "G_BATT", "G_CORPUS", "G_ENV")},
        "e216_status": e216m["status"],
        "e216_gates": dict(e216m["adjudication"]["gates_summary"]),
        "e215_reprobe_max_dp": e215m["gates"]["G_STATES"]["reprobe_max_dp"],
        "git": prov,
        "note": "the records chain's own certifications are inherited (e215 "
                "re-probed both +80 states at dp 0.0 vs e214's records; "
                "e214 certified those vs the e182c/e182c2 originals at "
                "<= 3.3e-06; e216 added the corpus/prompt rebuild "
                "certifications + its own baseline fit); this cell adds the "
                "desk recompute (G_RECOMPUTE), git provenance and the "
                "bit-checked baseline reproduction (G_BASELINE)",
    }
    G_RECORDS["pass"] = bool(
        e215m["status"] == "DONE" and len(assign) == 54
        and all(G_RECORDS["e215_gates"].values())
        and e216m["status"] == "DONE"
        and all(e216m["adjudication"]["gates_summary"].values())
        and not prov["dirty_records"])
    metrics["gates"] = {"G_RECORDS": G_RECORDS}
    log(f"G_RECORDS: {'PASS' if G_RECORDS['pass'] else 'FAIL'} "
        f"(e215 {e215m['status']}, {len(assign)} probes; e216 "
        f"{e216m['status']}, gates "
        f"{sum(bool(v) for v in e216m['adjudication']['gates_summary'].values())}"
        f"/{len(e216m['adjudication']['gates_summary'])}; HEAD "
        f"{prov['head'][:9]})")
    write_metrics("PARTIAL: records read, provenance taken")

    # ------------------------------------------ P2 the desk recompute (G_RECOMPUTE)
    t0j = [r for r in e214j if r["wash"] == "t0"][0]
    cur_w1 = {int(s): e for s, e in e214m["curves"]["w1"].items()}
    cur_w2 = {int(s): e for s, e in e214m["curves"]["w2"].items()}
    assert DEEPEST in cur_w1 and DEEPEST in cur_w2
    max_dp = 0.0
    rows = []
    for r in assign:
        b = r["battery"]
        p0_rc = t0j[b]["probes"][r["fact"]]["p"]
        p80w1_rc = cur_w1[DEEPEST][b]["probes"][r["fact"]]
        p80w2_rc = cur_w2[DEEPEST][b]["probes"][r["fact"]]
        for a, bb in ((p0_rc, r["p0"]), (p80w1_rc, r["p80_w1"]),
                      (p80w2_rc, r["p80_w2"]),
                      (p80w1_rc / p0_rc, r["hr_w1"]),
                      (p80w2_rc / p0_rc, r["hr_w2"])):
            max_dp = max(max_dp, abs(a - bb))
        rows.append(dict(r))                    # the committed row, verbatim
    G_RECOMPUTE = {
        "recomputed_from": "e214 curves w1/w2 +{80} and journal t=0, matched "
                           "per probe by fact+battery (the e216 recompute, "
                           "verbatim)",
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

    # ------------------------------------ P3 the four forms + competitors, fit
    fams = [r["family6"] for r in rows]
    p0s = [r["p0"] for r in rows]
    p0_mean = sum(p0s) / len(p0s)

    # the within-family ranks of p0 (average ranks under ties, 1-based)
    rank_frac: list[float] = [0.0] * len(rows)
    raw_rank: list[float] = [0.0] * len(rows)
    for f in FAMILY6_ORDER:
        idx = [i for i, g in enumerate(fams) if g == f]
        rk = avg_ranks([p0s[i] for i in idx])
        for j, i in enumerate(idx):
            raw_rank[i] = rk[j]
            rank_frac[i] = rk[j] / len(idx)
    for r, rf, rr in zip(rows, rank_frac, raw_rank):
        r["p0_rank_frac"] = rf
        r["p0_raw_rank"] = rr

    y_w1 = np.array([r["hr_w1"] for r in rows])
    y_w2 = np.array([r["hr_w2"] for r in rows])
    y_mean = np.array([r["hr_mean"] for r in rows])

    table: dict[str, dict] = {}
    comp_table: dict[str, dict] = {}
    all_entries: dict[str, dict] = {}
    for form in FORMS + COMPETITOR_FORMS:
        X = design(form, fams, p0s, rank_frac, raw_rank, p0_mean)
        entry = {"label": (FORM_LABELS if form in FORMS else COMPETITOR_LABELS)[form],
                 "n_params": X.shape[1],
                 "registered": form in FORMS}
        for tag, y in (("w1", y_w1), ("w2", y_w2), ("mean", y_mean)):
            fit = ols(X, y)
            entry[tag] = {"r2": fit["r2"], "adj_r2": fit["adj_r2"],
                          "resid_sd": fit["resid_sd"], "dof": fit["dof"],
                          "coefs": {n: [float(bv), float(se)] for n, bv, se
                                    in zip(form_coef_names(form), fit["beta"],
                                           fit["beta_se"])},
                          "resid": fit["resid"].tolist()}
        entry["r2"] = {t: entry[t]["r2"] for t in ("w1", "w2", "mean")}
        entry["adj_r2"] = {t: entry[t]["adj_r2"] for t in ("w1", "w2", "mean")}
        entry["resid_sd"] = {t: entry[t]["resid_sd"] for t in ("w1", "w2", "mean")}
        all_entries[form] = entry
        if form in FORMS:
            entry["delta_r2_vs_linear"] = {
                t: entry[t]["r2"] - table["linear"][t]["r2"]
                for t in ("w1", "w2", "mean")} if form != "linear" \
                else {t: 0.0 for t in ("w1", "w2", "mean")}
            table[form] = entry
        else:
            comp_table[form] = {
                "label": entry["label"], "n_params": entry["n_params"],
                "r2": entry["r2"], "adj_r2": entry["adj_r2"],
                "resid_sd": entry["resid_sd"],
                "note": "co-report competitor (never adjudicated)"}

    # G_BASELINE: the linear form vs e216's committed full model
    e216_res = {(t["fact"], t["battery"]): t for t in e216m["residual"]["table"]}
    max_db = 0.0
    for key, t in (("r2_w1", table["linear"]["w1"]["r2"]),
                   ("r2_w2", table["linear"]["w2"]["r2"]),
                   ("r2_mean", table["linear"]["mean"]["r2"]),
                   ("adj_w1", table["linear"]["w1"]["adj_r2"]),
                   ("adj_w2", table["linear"]["w2"]["adj_r2"]),
                   ("adj_mean", table["linear"]["mean"]["adj_r2"]),
                   ("resid_sd_w1", table["linear"]["w1"]["resid_sd"]),
                   ("resid_sd_w2", table["linear"]["w2"]["resid_sd"])):
        ref = {"r2_w1": e216m["full_model"]["r2"]["w1"],
               "r2_w2": e216m["full_model"]["r2"]["w2"],
               "r2_mean": e216m["full_model"]["r2"]["mean"],
               "adj_w1": e216m["full_model"]["adj_r2"]["w1"],
               "adj_w2": e216m["full_model"]["adj_r2"]["w2"],
               "adj_mean": e216m["full_model"]["adj_r2"]["mean"],
               "resid_sd_w1": e216m["full_model"]["resid_sd"]["w1"],
               "resid_sd_w2": e216m["full_model"]["resid_sd"]["w2"]}[key]
        max_db = max(max_db, abs(t - ref))
    for i, r in enumerate(rows):
        ref = e216_res[(r["fact"], r["battery"])]
        max_db = max(max_db,
                     abs(table["linear"]["w1"]["resid"][i] - ref["resid_w1"]),
                     abs(table["linear"]["w2"]["resid"][i] - ref["resid_w2"]))

    # P4 (part 2): the decisive statistic per form — first for the baseline
    def xw_of(entry) -> dict:
        r1 = entry["w1"]["resid"]
        r2_ = entry["w2"]["resid"]
        return {"spearman": spearman(r1, r2_), "pearson": _pearson(r1, r2_),
                "per_family": {f: spearman(
                    [entry["w1"]["resid"][i] for i, g in enumerate(fams) if g == f],
                    [entry["w2"]["resid"][i] for i, g in enumerate(fams) if g == f])
                    for f in FAMILY6_ORDER}}

    decisive = {f: xw_of(table[f]) for f in FORMS}
    decisive_comp = {f: xw_of(all_entries[f]) for f in COMPETITOR_FORMS}
    G_BASELINE = {
        "compared": "this cell's linear form vs e216's committed full_model "
                    "(r2/adj_r2/resid_sd both washes + mean; the 54 per-probe "
                    "residuals both washes)",
        "max_dp": max_db, "tol": TOL_BASELINE_DP,
        "xw_spearman_this_cell": decisive["linear"]["spearman"],
        "xw_spearman_e216": e216m["cross_wash_residual"]["spearman"],
        "note": "the baseline is e216 REPRODUCED (extend, don't repeat): the "
                "new forms are read only after this bit-check passes",
    }
    G_BASELINE["pass"] = bool(
        max_db <= TOL_BASELINE_DP
        and abs(decisive["linear"]["spearman"]
                - e216m["cross_wash_residual"]["spearman"]) <= TOL_BASELINE_DP
        and abs(decisive["linear"]["pearson"]
                - e216m["cross_wash_residual"]["pearson"]) <= TOL_BASELINE_DP)
    metrics["gates"]["G_BASELINE"] = G_BASELINE
    log(f"G_BASELINE: {'PASS' if G_BASELINE['pass'] else 'FAIL'} "
        f"(max dp {max_db:.2e} vs e216's committed fit; xw spearman "
        f"{decisive['linear']['spearman']:.6f} vs "
        f"{e216m['cross_wash_residual']['spearman']:.6f})")
    load_checks.append(cpu_load_check("mid_desk"))
    write_metrics("PARTIAL: the four forms fit; baseline bit-checked")
    if not G_BASELINE["pass"] and not SMOKE:
        log("FATAL: the linear arm failed to reproduce e216's committed fit "
            "— aborting before reading any new form")
        return 1

    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "model_loads": 0, "tokenizer_loads": 0, "gpu_calls": 0,
             "load_checks": len(load_checks),
             "burst": "pure JSON desk (OLS via numpy lstsq; no model, no "
                      "tokenizer, no corpus, no prompts)",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV

    # the model comparison table
    comparison = {
        "definition": "OLS hr ~ family (5 dummies, reference lang) + HEIGHT "
                      "form; R2 = 1 - SSres/SStot; PRIMARY per wash",
        "forms": {f: {"label": table[f]["label"],
                      "n_params": table[f]["n_params"],
                      "r2": table[f]["r2"], "adj_r2": table[f]["adj_r2"],
                      "resid_sd": table[f]["resid_sd"],
                      "delta_r2_vs_linear": table[f]["delta_r2_vs_linear"]}
                  for f in FORMS},
        "log": " | ".join(
            f"{f}: R2 w1 {table[f]['r2']['w1']:.3f} w2 {table[f]['r2']['w2']:.3f}"
            for f in FORMS),
    }
    metrics["model_comparison"] = comparison
    for f in FORMS:
        log(f"form {f:7s}: R2 w1 {table[f]['r2']['w1']:.4f} / w2 "
            f"{table[f]['r2']['w2']:.4f} / mean {table[f]['r2']['mean']:.4f} "
            f"({table[f]['n_params']} params; delta vs linear "
            f"{table[f]['delta_r2_vs_linear']['w1']:+.4f}/"
            f"{table[f]['delta_r2_vs_linear']['w2']:+.4f})")
    write_metrics("PARTIAL: the model comparison table")

    # --------------------------------------------- P4 THE DECISIVE TEST (part 2)
    for f in COMPETITOR_FORMS:
        comp_table[f]["xwash_spearman"] = decisive_comp[f]["spearman"]
        comp_table[f]["xwash_pearson"] = decisive_comp[f]["pearson"]
    metrics["decisive_test"] = {
        "definition": "Spearman(resid_w1, resid_w2) per form (the e215/e216 "
                      "reliability convention; Pearson + per-family "
                      "co-reported); ABSORPTION = |rho| < 0.5; SURVIVAL = "
                      "rho >= 0.5",
        "baseline_commit": e216m["cross_wash_residual"]["spearman"],
        "xwash_by_form": {f: {"spearman": decisive[f]["spearman"],
                              "pearson": decisive[f]["pearson"],
                              "per_family": decisive[f]["per_family"]}
                          for f in FORMS},
        "note": "the 0.934 baseline is the linear form's entry — the cell "
                "asks whether rank/logit/quad move it below the 0.5 "
                "absorption line",
    }
    for f in FORMS:
        log(f"DECISIVE {f:7s}: xwash Spearman {decisive[f]['spearman']:.4f} "
            f"/ Pearson {decisive[f]['pearson']:.4f}; per family: "
            + " ".join(f"{fam.split('-')[0]} "
                       f"{decisive[f]['per_family'][fam]:.2f}"
                       for fam in FAMILY6_ORDER))
    write_metrics("PARTIAL: the decisive test")

    # ------------------------------- P5 the BEST form + the named splits under it
    height_forms = [f for f in FORMS
                    if decisive[f]["spearman"] is not None
                    and abs(decisive[f]["spearman"]) < XWASH_ABSORB
                    and table[f]["r2"]["w1"] >= R2_BAR
                    and table[f]["r2"]["w2"] >= R2_BAR]
    if height_forms:
        best = max(height_forms, key=lambda f: table[f]["r2"]["mean"])
    else:
        best = min(FORMS, key=lambda f: (decisive[f]["spearman"],
                                         -table[f]["r2"]["mean"]))
    log(f"BEST form: {best} "
        f"({'firing absorber' if height_forms else 'lowest cross-wash rho'})")

    for i, r in enumerate(rows):
        r["resid_w1"] = table[best]["w1"]["resid"][i]
        r["resid_w2"] = table[best]["w2"]["resid"][i]

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
        gap_sd[w] = gap / table[best][w]["resid_sd"]
    prod_survives = bool(gap_sd["w1"] >= SPLIT_SD_BAR
                         and gap_sd["w2"] >= SPLIT_SD_BAR)
    product_split = {
        "definition": f"under the best form ({best}): contrast = mean "
                      f"resid{PRODUCT_HOLD_GROUP} - mean "
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
    tmpl_dissolves = bool(
        tw["ratio_w1"] is not None and tw["ratio_w2"] is not None
        and tw["ratio_w1"] <= WIDTH_DISSOLVE_BAR
        and tw["ratio_w2"] <= WIDTH_DISSOLVE_BAR)
    tmpl_width = {
        "definition": f"under the best form ({best}): within rev-capital, "
                      f"SD(resid)/SD(hr) per wash; DISSOLVES iff <= "
                      f"{WIDTH_DISSOLVE_BAR} on both washes (descriptive)",
        "ratio_w1": tw["ratio_w1"], "ratio_w2": tw["ratio_w2"],
        "within_family_spearman_p0_hr": {
            "w1": tw["spearman_p0_hr_w1"], "w2": tw["spearman_p0_hr_w2"]},
        "verdict": "DISSOLVES (the height form accounts for the width)"
                   if tmpl_dissolves else "SURVIVES (width unabsorbed)",
    }
    splits = {"best_form": best, "product": product_split,
              "tmpl_width": tmpl_width, "family_width_table": fam_width}
    metrics["named_splits_under_best"] = splits
    log(f"named splits under the best form ({best}): product contrast "
        f"{gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD -> "
        f"{product_split['verdict']}; tmpl width ratio "
        f"{tw['ratio_w1']:.2f}/{tw['ratio_w2']:.2f} -> "
        f"{tmpl_width['verdict']}")

    # the per-probe residual table under the best form + the height covariates
    resid_table = [{"fact": r["fact"], "battery": r["battery"],
                    "family6": r["family6"], "answer": r["answer"],
                    "p0": r["p0"], "p0_rank_frac": r["p0_rank_frac"],
                    "hr_w1": r["hr_w1"], "hr_w2": r["hr_w2"],
                    "resid_w1": r["resid_w1"], "resid_w2": r["resid_w2"]}
                   for r in rows]
    metrics["residual_table_best_form"] = {
        "form": best,
        "largest_abs_resid_w1": sorted(
            ((abs(r["resid_w1"]), r["fact"]) for r in rows),
            reverse=True)[:6],
        "largest_abs_resid_w2": sorted(
            ((abs(r["resid_w2"]), r["fact"]) for r in rows),
            reverse=True)[:6],
        "table": resid_table,
    }
    metrics["competitors"] = {
        "note": "co-report competitor forms (never adjudicated — no bar "
                "shopping): rawrank = the unnormalized within-family rank; "
                "fslopes = family-specific linear p0 slopes (the one height "
                "functional the registered set lacks)",
        "forms": comp_table,
    }
    write_metrics("PARTIAL: best form + named splits + competitors")

    # ----------------------------------------------- P6 ADJUDICATION (frozen)
    absorbed = [f for f in FORMS if decisive[f]["spearman"] is not None
                and abs(decisive[f]["spearman"]) < XWASH_ABSORB]
    all_survive = bool(all(decisive[f]["spearman"] is not None
                           and decisive[f]["spearman"] >= XWASH_SURVIVE
                           for f in FORMS))
    gates_ok = bool(G_RECORDS["pass"] and G_RECOMPUTE["pass"]
                    and G_BASELINE["pass"] and G_ENV["pass"])

    form_txt = "; ".join(
        f"{f}: rho {decisive[f]['spearman']:.3f}, R2 "
        f"{table[f]['r2']['w1']:.3f}/{table[f]['r2']['w2']:.3f}"
        for f in FORMS)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_RECORDS", G_RECORDS["pass"]),
                           ("G_RECOMPUTE", G_RECOMPUTE["pass"]),
                           ("G_BASELINE", G_BASELINE["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif height_forms:
        verdict = "HEIGHT-NONLINEAR"
        clause = (f"the {best} form absorbs the cross-wash residual "
                  f"(Spearman {decisive[best]['spearman']:.3f}, |rho| < "
                  f"{XWASH_ABSORB:g}) with R2 "
                  f"{table[best]['r2']['w1']:.3f}/{table[best]['r2']['w2']:.3f} "
                  f">= {R2_BAR:g} on both washes — the third dimension IS "
                  f"nonlinear height; the family x height model completed in "
                  f"its nonlinear form; the named splits under it: product "
                  f"{product_split['verdict']}, tmpl width "
                  f"{tmpl_width['verdict']}; the forms: {form_txt}")
    elif all_survive:
        worst = max(FORMS, key=lambda f: decisive[f]["spearman"])
        clause = (f"the cross-wash residual rho stays >= {XWASH_SURVIVE:g} "
                  f"under EVERY height form ({form_txt}) — the third "
                  f"dimension is genuinely new; the probe-feature hunt owed "
                  f"(named candidates: the answer's base-model entrenchment "
                  f"measured independently, the probe's internal token "
                  f"structure, the relation's compositionality); the named "
                  f"splits under the best form ({best}): product "
                  f"{product_split['verdict']} "
                  f"({gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD), tmpl width "
                  f"{tmpl_width['verdict']} "
                  f"({tw['ratio_w1']:.2f}/{tw['ratio_w2']:.2f})")
        verdict = "BEYOND-HEIGHT"
    else:
        why = []
        if absorbed:
            why.append("forms that absorb (|rho| < 0.5) but MISS the R2 "
                       "clause: " + ", ".join(
                           f"{f} (rho {decisive[f]['spearman']:.3f}, R2 "
                           f"{table[f]['r2']['w1']:.3f}/"
                           f"{table[f]['r2']['w2']:.3f})" for f in absorbed))
        else:
            why.append("no form absorbs the cross-wash residual "
                       "(all |rho| >= 0.5)")
        if not all_survive and not absorbed:
            weak = [f for f in FORMS if decisive[f]["spearman"] is None
                    or decisive[f]["spearman"] < XWASH_SURVIVE]
            why.append("but not every form stays >= 0.5 either: " + ", ".join(
                f"{f} (rho "
                f"{'None' if decisive[f]['spearman'] is None else format(decisive[f]['spearman'], '.3f')})"
                for f in weak))
        verdict = "GRADED"
        clause = ("any partial — " + "; ".join(why)
                  + f"; the named splits under the best form ({best}): "
                  f"product {product_split['verdict']} "
                  f"({gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD), tmpl width "
                  f"{tmpl_width['verdict']} "
                  f"({tw['ratio_w1']:.2f}/{tw['ratio_w2']:.2f}); the forms: "
                  f"{form_txt}; the tables verbatim")

    gates_summary = {"G_RECORDS": G_RECORDS["pass"],
                     "G_RECOMPUTE": G_RECOMPUTE["pass"],
                     "G_BASELINE": G_BASELINE["pass"],
                     "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {"HEIGHT_NONLINEAR": bool(height_forms and gates_ok
                                          and not SMOKE),
                 "BEYOND_HEIGHT": bool(all_survive and gates_ok and not SMOKE
                                       and not height_forms),
                 "verdict": verdict, "clause": clause,
                 "order": "HEIGHT-NONLINEAR -> BEYOND-HEIGHT -> GRADED "
                          "(gated on G_RECORDS/G_RECOMPUTE/G_BASELINE/G_ENV)"},
        "clauses": {
            "absorbing_forms (|xwash rho| < 0.5)": absorbed,
            "r2_both_washes >= 0.75 (per form)": {
                f: bool(table[f]["r2"]["w1"] >= R2_BAR
                        and table[f]["r2"]["w2"] >= R2_BAR) for f in FORMS},
            "all_forms_xwash >= 0.5": all_survive,
            "best_form": best,
        },
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E218 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    # ------------------------------------------------ P7 honesty + close
    metrics["honesty_reflex"] = {
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                    "replay of e182's GPU original, wash 2 is GPU fp32 — "
                    "inherited from the archive (disclosed)",
        "typology_inherited": "the family factor is e215's HAND-REGISTERED "
                              "typology (not blind to the outcome — the "
                              "grouping follows the instrument's pools, "
                              "disclosed there); every form's R^2 here is "
                              "measured AGAINST that hand-grouped factor — "
                              "'height explains X%' inherits the typology's "
                              "subjectivity wholesale",
        "ols_conventions": "raw bounded hr (no transform), plain OLS, "
                           "homoskedastic SEs (co-reported, never "
                           "adjudicated); unbalanced family n's (3..19) and "
                           "heteroskedastic family widths make the plain R^2 "
                           "generous in the wide families — the adj-R^2 "
                           "beside it (the e216 conventions, inherited)",
        "forms_are_functions_of_p0": "the four registered forms carry NO "
                                     "information beyond p0 itself (rank is "
                                     "p0's within-family monotone — strictly "
                                     "LESS information; logit and quad are "
                                     "global transforms): a BEYOND-HEIGHT "
                                     "verdict means no registered FUNCTION "
                                     "of p0+family absorbs the residual — "
                                     "T190's cut as dispatched, not a proof "
                                     "that no height functional exists (the "
                                     "family-slopes competitor sits beside "
                                     "it, never adjudicated)",
        "rank_scaling": "the rank form's /n_f fractional scaling is the "
                        "registered choice; with family dummies present any "
                        "COMMON affine transform is fit-equivalent — the "
                        "raw-rank competitor shows the unnormalized variant",
        "absorption_convention": "the bar's '(rho < 0.5)' is read as "
                                 "|rho| < 0.5 (a negatively-replicating miss "
                                 "is not absorption either; e216's "
                                 "abs-convention on the low side, inherited); "
                                 "survival is read as rho >= 0.5",
        "multiple_forms": "four forms compared on the same 54 probes is a "
                          "multiple-comparison surface — the comparison "
                          "table is descriptive; only the frozen clauses "
                          "adjudicate",
        "logit_ceiling": "logit(p0) stretches the top of p0's range "
                         "(0.52..0.99 -> 0.08..4.66): the few probes near "
                         "p0 ~ 0.99 dominate that regressor's variance",
        "xwash_reading": "a cross-wash residual correlation could be probe "
                         "physics OR shared texture (both arms divide by "
                         "the same p0; both share the wash corpus and the "
                         "prompt strings) — the correlation replicates the "
                         "MISS, its mechanism stays open",
        "named_split_yardsticks": "the product-contrast 1.0-SD and tmpl "
                                  "0.5-ratio conventions are DESCRIPTIVE "
                                  "yardsticks frozen before compute (the "
                                  "e216 conventions) — they feed the clause "
                                  "wording, they never flip a bar",
        "hr_ratio_floor": "hr divides by p0 as low as 0.52; deep-state floors "
                          "compress ratios — hr and the per-form residuals "
                          "co-reported per probe under the best form",
        "guarantees_nothing": "the forms and their residuals describe THESE "
                              "54 probes under THESE two washes with an "
                              "inherited hand-registered typology — the "
                              "openness is the point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, ZERO model loads AND zero "
                    "tokenizer loads (pure JSON desk), decisive; no GPU "
                    "calls",
        "load_checks": load_checks,
        "records_chain": ["runs/e214/metrics.json", "runs/e214/journal.json",
                          "runs/e215/metrics.json",
                          "runs/e216/metrics.json"],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, rows, table, decisive, splits, best, adj,
                        comp_table)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
