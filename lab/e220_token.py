"""E220 — THE TOKEN-STRUCTURE TEST (the hunt's second candidate).

WHY: T191/e218 proved the third sorting dimension is BEYOND HEIGHT (no
function of p0+family absorbs it; best form moves rho by 0.02).
T193/e219 broke the pipeline-texture confound and RETIRED the first
candidate: the paraphrase channel is independent (rho(IE,p0)=0.33),
correlates weakly with the residual (+0.234/+0.256 — the miss is real,
not instrument texture) but absorbs NOTHING (0.936 vs 0.934) — and
Gmail, the archive's biggest residual, sits mid-channel. THE HUNT'S
SECOND CANDIDATE (T191's own list): the probe's INTERNAL TOKEN
STRUCTURE — the tokenizer-level shape of the answer, the cue and the
prompt as the GPT-2 tokenizer sees them. Does token structure NAME the
third dimension, or does it retire cleanly like entrenchment did (the
last candidate, compositionality, then owed)?

THE CELL (a PURE DESK cell on the committed records — tokenizer-only
loads, no model, no wash state):
  (1) TOKEN FEATURES per probe, computed from the COMMITTED prompts and
      answers with the pinned GPT-2 tokenizer (prompts + cues rebuilt
      VERBATIM by module import and certified by G_PROMPT):
        t1 ans_tok_count      — the answer's token count (" "+answer);
        t2 ans_frag           — the answer's subword fragmentation
                                (tokens per char of " "+answer);
        t3 cue_tok_count      — the cue's token count (the prompt's
                                final query clause, template-certified);
        t4 prompt_tok_len     — the prompt's total token length (the
                                committed field, re-certified);
        t5 ans_first_tok_freq — the answer's FIRST-TOKEN identity as
                                commonness: the token's frequency (per
                                1M tokens) in the SOURCE corpus
                                (data/input.txt, tokenizer-only) —
                                "is it a common continuation token?";
                                token id + string recorded per probe;
        t6 ans_cue_overlap    — the token overlap between the answer
                                and the cue (|ids(ans) n ids(cue)| /
                                |ids(ans)|).
  (2) THE TEST: Spearman(feature, e216's residual) per wash — the
      residual re-derived through G_BASELINE (this cell's own family+p0
      OLS must reproduce e216's committed residual table at 1e-9
      first); the decisive ABSORPTION — family + each token feature
      (and a small token-feature composite: family + all varying
      features jointly) SUBSTITUTED as the height term in place of p0
      (e219's substitution convention): does the cross-wash residual
      rho drop below 0.5?
  (3) THE NAMED SPLITS' fate under the best token model (the
      substitution form with the lowest |cross-wash rho|): the product
      contrast (Gmail/PlayStation vs iPhone, residual-SD units) and
      the tmpl width (SD(resid)/SD(hr) within rev-capital) — e216's
      yardsticks verbatim.
  (4) THE GMAIL/IPHONE CASE STUDY: their token features vs their
      opposite residuals (the hunt's sharpest single anchor).

REGISTERED BARS (frozen VERBATIM from the dispatch brief, QUEUE row
DISPATCHED 17:58Z / commit e186b89, BEFORE any compute; adjudicate
against exactly this; no bar shopping):
  - TOKEN-NAMES-IT: "fires if a token feature (or composite) correlates
    |rho| >= 0.4 with the residual or absorbs the cross-wash rho below
    0.5 — the third dimension is TOKEN STRUCTURE; named."
  - TOKEN-SILENT: "fires if all token features sit |rho| < 0.2 with no
    absorption — token structure retired; the hunt's last candidate
    (compositionality) owed."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * DATA = e215's committed 54-probe assignment, re-certified desk-only
    by G_RECOMPUTE against e214's curves+journal at 1e-9 + G_RECORDS
    git provenance. Outcome = hr_w = p_w(+80)/p0 per wash.
  * BASELINE = e216's full model (family dummies + p0 centered),
    re-fit HERE; G_BASELINE requires this cell's per-probe residuals
    to match e216's committed residual table at 1e-9 and the cross-wash
    Spearman to match e216's committed 0.933981322660568. All feature
    tests read THIS cell's re-derived residuals.
  * CORRELATION = Spearman(feature, resid_w) per wash, per registered
    feature; FIRES iff max |rho| over the two washes >= 0.40.
  * COMPOSITE = the small token-feature composite is the SUBSTITUTION
    model itself: family + ALL VARYING registered features jointly, in
    place of p0 (constant/degenerate features are excluded from OLS
    forms, disclosed). Its correlation score = the feature-part of that
    model's fitted linear predictor on hr_mean (a SUPERVISED composite
    — its weights are fit to the outcome, disclosed as generous; the
    Bonferroni note covers it).
  * ABSORPTION = for each registered feature alone AND the composite:
    OLS hr_w ~ family + feature(s) centered (p0 SUBSTITUTED out, e219's
    convention); xwash_f = Spearman(resid_w1, resid_w2) of that model;
    ABSORBED iff ANY |xwash_f| < 0.5 (e218's abs-convention on the
    bar's 'rho < 0.5'). The additive co-report (family + p0 + feature)
    beside it, never adjudicated.
  * BEST TOKEN MODEL (for the named splits) = the substitution form
    with the LOWEST |xwash| (ties -> fewer features); the product
    contrast and tmpl width read from ITS residuals with e216's
    descriptive yardsticks (>= 1.0 SD survives; <= 0.5 ratio
    dissolves).
  * ADJUDICATION ORDER: gates ok -> TOKEN-NAMES-IT iff (any single
    feature or the composite max-wash |rho| >= 0.4 OR any substitution
    |xwash| < 0.5); else TOKEN-SILENT iff (every single feature AND
    the composite max-wash |rho| < 0.2 AND no absorption); else GRADED.

DESK TEXTURE DISCLOSED UP FRONT (the honesty core): the battery's own
construction rules PRE-CONSTRAIN this feature set — e182's rule drops
every multi-token answer (t1 expected constant 1, making t2 a
char-length proxy), the contamination scans remove every answer token
from the wash train stream, and the rotating-exemplar rule keeps the
answer out of its own prompt (t6 and the answer-prompt overlap
expected 0). A TOKEN-SILENT verdict is therefore partly BY
CONSTRUCTION; the genuinely varying features are t3/t4 (cue and prompt
lengths), t2 (as answer char-length) and t5 (source-corpus first-token
frequency). Nothing guaranteed — the openness is the point.

PROVENANCE: the residual records and the certification chain are
e216's (re-derived here, never rewritten); the assignment is e215's;
the curves/journal are e214's; the batteries, corpus filter and
organism constants are lab/e182c_forgetting_control.py +
lab/e182c2_template.py VERBATIM via module import. Builds on: T191 /
e218 (the named hunt + BEYOND-HEIGHT), T193 / e219 (the first
candidate retired; the substitution convention; the Gmail anchor),
T190 / e216 (the residual this cell hunts), T189 / e215, T187 / e214,
T183 / e182c2 + T149 / e182c + T123 / e182 (the two-wash 124M
archive). NEW: the six-feature token table on the committed prompts;
the correlation test against the residual; the per-feature and
composite substitution absorption battery; the named splits under the
best token model; the Gmail/iPhone case study; the Bonferroni note
across features.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and after the corpus phase; ZERO
model loads (the GPT-2 tokenizer from the pinned local cache only); no
wash, no probes, no NOTES/THINKING/QUEUE/STATE edits (dispatch).
Progressive PARTIAL metrics after every phase (the standing disruption
rule).

Run:  cd lab && python e220_token.py   (E220_SMOKE=1: the same desk
      tables on the same committed records, own smoke dir, nothing
      adjudicated — the e215/e216 smoke disclosure applies)
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

SMOKE = os.environ.get("E220_SMOKE") == "1"
NAME = "e220_smoke" if SMOKE else "e220"

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
REF_FAMILY = "lang"             # the treatment-coding reference (e216 verbatim)

# the registered bar constants (frozen)
CORR_BAR = 0.40         # TOKEN-NAMES-IT's "|rho| >= 0.4"
SILENT_BAR = 0.20       # TOKEN-SILENT's "|rho| < 0.2"
ABSORB_BAR = 0.50       # "absorbs the cross-wash rho below 0.5" -> |xwash| < 0.5

# the named-split descriptive yardsticks (e216 verbatim; never adjudicated)
PRODUCT_HOLD_GROUP = ["Gmail", "PlayStation"]   # the dispatch's named holds
PRODUCT_COLLAPSE_ANCHOR = "iPhone"              # the dispatch's named collapse
SPLIT_SD_BAR = 1.0     # product contrast SURVIVES iff >= 1.0 residual SD both washes
WIDTH_DISSOLVE_BAR = 0.5   # tmpl width DISSOLVES iff SD(resid)/SD(hr) <= 0.5 both washes

# the case-study pair (the hunt's sharpest single anchor)
CASE_PAIR = ("Gmail", "iPhone")

# the registered token-feature set (the dispatch's list, verbatim order)
TOKEN_FEATURES = [
    ("ans tok count", "ans_tok_count",
     "the answer's token count (encoding ' '+answer) — the battery's own "
     "rule drops multi-token answers, so this is expected CONSTANT (=1); "
     "computed, disclosed degenerate"),
    ("ans fragmentation", "ans_frag",
     "the answer's subword fragmentation: tokens per char of ' '+answer — "
     "under the single-token rule this is the answer's char-length inverse "
     "(a char-length proxy; disclosed)"),
    ("cue tok count", "cue_tok_count",
     "the cue's token count — the prompt's final query clause, rebuilt from "
     "the frozen templates and certified by prompt.endswith(cue)"),
    ("prompt tok len", "prompt_tok_len",
     "the prompt's total token length (the committed field, re-certified "
     "by the module-import rebuild, G_PROMPT)"),
    ("ans first-tok freq", "ans_first_tok_freq",
     "the answer's first-token commonness: occurrences of its token id in "
     "the SOURCE corpus (data/input.txt) per 1M tokens, tokenizer-only — "
     "'is it a common continuation token?'; the token id + string "
     "co-reported per probe"),
    ("ans-cue tok overlap", "ans_cue_overlap",
     "|ids(answer) n ids(cue)| / |ids(answer)| — under single-token answers "
     "this is binary; expected 0 by the exemplar-rotation rule (disclosed)"),
]
# co-report competitors (never adjudicated)
TOKEN_COMPETITORS = [
    ("ans-prompt tok overlap", "ans_prompt_overlap",
     "fraction of the answer's token ids present in the prompt's ids"),
    ("ans first-tok freq (filtered)", "ans_first_tok_freq_filtered",
     "the same first-token frequency measured on the FROZEN filtered "
     "corpus (construction-limited: the banned-string filter)"),
    ("ans char len", "ans_char_len", "len(' '+answer)"),
    ("cue char len", "cue_char_len", "len(cue)"),
]

# registered verification tolerances (frozen)
TOL_RECOMPUTE_DP = 1e-9
E216_XWASH_COMMITTED = 0.933981322660568   # e216's committed cross-wash Spearman

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "TOKEN-NAMES-IT": "fires if a token feature (or composite) "
            "correlates |rho| >= 0.4 with the residual or absorbs the "
            "cross-wash rho below 0.5 — the third dimension is TOKEN "
            "STRUCTURE; named.",
        "TOKEN-SILENT": "fires if all token features sit |rho| < 0.2 "
            "with no absorption — token structure retired; the hunt's "
            "last candidate (compositionality) owed.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "operationalizations": (
        "data = e215's committed 54-probe assignment (re-certified "
        "desk-only by G_RECOMPUTE against e214's curves+journal at 1e-9 + "
        "G_RECORDS git provenance); outcome = hr_w = p_w(+80)/p0 per wash; "
        "BASELINE = e216's family+p0 model re-fit here, G_BASELINE at 1e-9 "
        "vs e216's committed residual table + its xw Spearman "
        "0.933981322660568; TOKEN FEATURES t1..t6 from the committed "
        "prompts/answers with the pinned GPT-2 tokenizer (prompts+cues "
        "rebuilt by module import, G_PROMPT incl. prompt.endswith(cue)); "
        "CORRELATION = Spearman(feature, resid_w) per wash, FIRES iff max "
        "|rho| over washes >= 0.40 (single features + the supervised "
        "composite score); COMPOSITE = family + all VARYING registered "
        "features jointly SUBSTITUTED for p0 (constants excluded, "
        "disclosed; its correlation score = the feature-part of the "
        "hr_mean fit — supervised, generous, Bonferroni-noted); "
        "ABSORPTION = per feature alone AND the composite: OLS hr_w ~ "
        "family + feature(s) centered (p0 substituted out), xwash_f = "
        "Spearman(resid_w1, resid_w2), ABSORBED iff ANY |xwash_f| < 0.5 "
        "(e218's abs-convention); additive family+p0+feature co-reported, "
        "never adjudicated; BEST TOKEN MODEL = the substitution form with "
        "the lowest |xwash| (ties -> fewer features) — the named splits "
        "read from ITS residuals with e216's yardsticks (product contrast "
        ">= 1.0 SD; tmpl width <= 0.5 ratio); ADJUDICATION: gates ok -> "
        "TOKEN-NAMES-IT iff (correlation OR absorption); else TOKEN-SILENT "
        "iff (all features AND composite |rho| < 0.20 AND no absorption); "
        "else GRADED; gated on G_RECORDS/G_RECOMPUTE/G_BASELINE/G_CORPUS/"
        "G_PROMPT/G_ENV"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (QUEUE "
                     "row DISPATCHED 17:58Z, commit e186b89) BEFORE any "
                     "compute; adjudicate against exactly this; no bar "
                     "shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "PURE DESK cell: NO model loads, NO re-probes (the dispatch's 'pure desk "
    "on committed records; tokenizer-only loads permitted') — the provenance "
    "chain enters as e215/e216's committed certifications PLUS this cell's "
    "desk recompute (G_RECOMPUTE), baseline reproduction (G_BASELINE) and "
    "git provenance (G_RECORDS).",
    "The battery's construction rules pre-constrain the dispatch's feature "
    "list: single-token answers (t1 constant, t2 a char-length proxy), "
    "contamination scans (answer tokens absent from the wash train stream), "
    "exemplar rotation (the answer absent from its own prompt) — disclosed "
    "up front in the docstring and the honesty block.",
    "The composite's correlation score is SUPERVISED (its weights fit to "
    "hr_mean under family controls) — generous by construction; the "
    "Bonferroni note across the feature x wash test count covers it.",
    "The named-split conventions (1.0 residual SD; 0.5 SD-ratio) are "
    "DESCRIPTIVE yardsticks frozen before compute (e216 verbatim) — they "
    "feed the clause wording, they never flip a bar.",
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


def _mean(v):
    return sum(v) / len(v) if v else None


def _sd(v):
    if not v:
        return None
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def spearman_p(rho: float, n: int) -> float | None:
    """Two-sided p for Spearman rho via the t-approximation (co-report)."""
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return None
    import math
    t = abs(rho) * math.sqrt((n - 2) / (1.0 - rho * rho))
    # normal-approx of the t tail (n=54; co-report only, never adjudicated)
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(t / 2.0 ** 0.5)))
    return p


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
    return {"beta": beta, "fitted": fitted, "resid": resid, "r2": r2,
            "adj_r2": adj, "ss_res": ss_res, "ss_tot": ss_tot,
            "resid_sd": float(ss_res / dof) ** 0.5, "dof": dof}


def fam_design(fams: list[str]) -> np.ndarray:
    """[1, 5 family dummies (ref = lang)] — the family block alone."""
    cols = [np.ones(len(fams))]
    for f in FAMILY6_ORDER:
        if f == REF_FAMILY:
            continue
        cols.append(np.array([1.0 if g == f else 0.0 for g in fams]))
    return np.column_stack(cols)


def design(fams: list[str], height_cols: list[np.ndarray]) -> np.ndarray:
    """[1, 5 family dummies, *height_cols] — the e216 design with the
    height term(s) pluggable (p0 for the baseline; token features for
    the substitutions)."""
    X = fam_design(fams)
    for c in height_cols:
        X = np.column_stack([X, c])
    return X


def coef_names(extra: list[str]) -> list[str]:
    return (["intercept(lang)"]
            + [f"D[{f}]" for f in FAMILY6_ORDER if f != REF_FAMILY]
            + extra)


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


def make_plot(rd, rows, feats_corr, absorb, splits_best, case, adj, base_xw):
    """THE FIGURE: (a) the feature-correlation ladder (|rho| vs the bars);
  (b) the absorption ladder (|xwash| per substitution form vs 0.5);
  (c) the Gmail/iPhone case study (first-token freq vs residual, the
  product family annotated); (d) the named splits under the best token
  model + the verdict strip."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) the feature-correlation ladder
    ax = axes[0, 0]
    entries = (feats_corr["registered"]
               + [feats_corr["composite"]] * (1 if feats_corr.get("composite") else 0))
    labels = [f["label"] for f in entries]
    vals1 = [f["rho_w1"] for f in entries]
    vals2 = [f["rho_w2"] for f in entries]
    for i in range(len(labels)):
        for v, dx, col in ((vals1[i], -0.17, "tab:blue"), (vals2[i], +0.17, "tab:cyan")):
            if v is None:
                continue
            ax.bar(i + dx, v, width=0.32, color=col, edgecolor="k", lw=0.5)
    ax.axhline(CORR_BAR, color="tab:red", ls="--", lw=1.4,
               label=f"names-it {CORR_BAR}")
    ax.axhline(-CORR_BAR, color="tab:red", ls="--", lw=1.4)
    ax.axhline(SILENT_BAR, color="gray", ls=":", lw=1.2,
               label=f"silent {SILENT_BAR}")
    ax.axhline(-SILENT_BAR, color="gray", ls=":", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.0)
    ax.set_ylabel("Spearman(feature, e216 residual)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("(1) TOKEN FEATURES vs THE THIRD DIMENSION "
                 "(blue w1 / cyan w2; composite = supervised)", fontsize=9.0)

    # (0,1) the absorption ladder
    ax = axes[0, 1]
    forms = absorb["forms"]
    flabels = [f["label"] for f in forms]
    fv = [f["xwash_spearman"] for f in forms]
    cols = ["tab:red" if (f["absorbs"] if "absorbs" in f else False) else "tab:gray"
            for f in forms]
    ax.bar(range(len(flabels)), fv, color=cols, edgecolor="k", lw=0.5)
    ax.axhline(ABSORB_BAR, color="tab:red", ls="--", lw=1.4,
               label=f"absorption < {ABSORB_BAR}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(abs(base_xw), color="darkblue", ls=":", lw=1.6,
               label=f"e216 baseline {abs(base_xw):.3f}")
    ax.set_xticks(range(len(flabels)))
    ax.set_xticklabels(flabels, rotation=20, ha="right", fontsize=7.0)
    ax.set_ylabel("|Spearman(resid_w1, resid_w2)| under the substitution")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("(2) THE DECISIVE ABSORPTION — family + token feature(s) "
                 "as the height term", fontsize=9.0)

    # (1,0) the case study: first-token freq vs residual
    ax = axes[1, 0]
    for fam in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([r["ans_first_tok_freq"] for r in fr],
                   [r["resid_w1"] for r in fr], s=26, marker="o",
                   color=FAM_COLS[fam], alpha=0.7, label=fam)
        ax.scatter([r["ans_first_tok_freq"] for r in fr],
                   [r["resid_w2"] for r in fr], s=26, marker="s",
                   facecolors="none", edgecolors=FAM_COLS[fam],
                   linewidths=1.3, alpha=0.9)
    for r in rows:
        if r["answer"] in PRODUCT_HOLD_GROUP + [PRODUCT_COLLAPSE_ANCHOR]:
            ax.annotate(r["answer"], (r["ans_first_tok_freq"], r["resid_w1"]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7.0, family="monospace")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("answer first-token frequency in the source corpus (per 1M)")
    ax.set_ylabel("e216 residual (o = w1, [] = w2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, loc="lower right")
    cs = case["pair"]
    ax.set_title(f"(3) THE CASE STUDY — {cs[0]} (holds, resid "
                 f"+{cs[0].lower()}) vs {cs[1]} (dies): "
                 f"freq {case['gmail']['ans_first_tok_freq']:.0f} vs "
                 f"{case['iphone']['ans_first_tok_freq']:.0f} per 1M",
                 fontsize=9.0)

    # (1,1) the named splits under the best token model + verdict
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"BEST TOKEN MODEL: {absorb['best_form']['label']} "
        f"(|xwash| {absorb['best_form']['xwash_spearman']:.3f} vs baseline "
        f"{abs(base_xw):.3f})",
        f"product contrast (Gmail/PlayStation - iPhone): "
        f"{splits_best['product']['gap_sd_w1']:+.2f} / "
        f"{splits_best['product']['gap_sd_w2']:+.2f} SD -> "
        f"{splits_best['product']['verdict']}",
        f"tmpl width (SD resid/SD hr, rev-capital): "
        f"{splits_best['tmpl_width']['ratio_w1']:.2f} / "
        f"{splits_best['tmpl_width']['ratio_w2']:.2f} -> "
        f"{splits_best['tmpl_width']['verdict']}",
        "",
        f"GMAIL: frag {case['gmail']['ans_frag']:.3f}, cue "
        f"{case['gmail']['cue_tok_count']} tok, first tok "
        f"'{case['gmail']['first_tok_str']}' @ "
       	f"{case['gmail']['ans_first_tok_freq']:.0f}/1M, resid "
        f"{case['gmail']['resid_w1']:+.3f}/{case['gmail']['resid_w2']:+.3f}",
        f"IPHONE: frag {case['iphone']['ans_frag']:.3f}, cue "
        f"{case['iphone']['cue_tok_count']} tok, first tok "
        f"'{case['iphone']['first_tok_str']}' @ "
       	f"{case['iphone']['ans_first_tok_freq']:.0f}/1M, resid "
       	f"{case['iphone']['resid_w1']:+.3f}/{case['iphone']['resid_w2']:+.3f}",
    ]
    for k, ln in enumerate(lines):
        ax.text(0.02, 0.95 - k * 0.075, ln, fontsize=8.2,
                family="monospace", transform=ax.transAxes,
                color="black" if k < 3 else "dimgray")

    y = -0.10
    ax.text(0.02, y, f"E220 VERDICT: {adj['bars']['verdict']}",
            fontsize=10.5, family="monospace", weight="bold",
            color="darkred", transform=ax.transAxes)
    for k, wd in enumerate(textwrap.wrap(adj["bars"]["clause"], width=105,
                                         break_long_words=False)):
        ax.text(0.02, y - 0.055 - k * 0.038, wd, fontsize=6.8,
                family="monospace", transform=ax.transAxes)
    ax.text(0.02, y - 0.055 - (len(textwrap.wrap(adj["bars"]["clause"],
                                                 width=105)) + 0.6) * 0.038,
            "  GATES: " + "  ".join(f"{g}={'PASS' if v else 'FAIL'}"
                                    for g, v in adj["gates_summary"].items()),
            fontsize=7.0, family="monospace", transform=ax.transAxes)

    fig.suptitle("E220 — THE TOKEN-STRUCTURE TEST (the hunt's second "
                 "candidate): does the probe's internal token structure name "
                 "the third dimension? -> " + adj["bars"]["verdict"],
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "token_structure.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E220 — THE TOKEN-STRUCTURE TEST (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e220_token_structure",
        "phase": "pure desk on the committed records (tokenizer-only loads; "
                 "no model, no wash state)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T191's named hunt, second candidate: does the probe's "
                     "INTERNAL TOKEN STRUCTURE (answer token count / "
                     "fragmentation, cue length, prompt length, first-token "
                     "commonness, answer-cue overlap) correlate with e216's "
                     "replicating residual — and does family + the best "
                     "token feature (or a small composite) as the height "
                     "term ABSORB the cross-wash residual where every "
                     "height form and the independent entrenchment channel "
                     "failed? (Gmail vs iPhone: the sharpest anchor)"),
        "builds_on": [
            "T191 / e218 (the named hunt; the BEYOND-HEIGHT verdict)",
            "T193 / e219 (the first candidate retired: the substitution "
            "convention, the Gmail anchor, the paraphrase registry)",
            "T190 / e216 (the residual this cell hunts + the committed "
            "residual table G_BASELINE reproduces)",
            "T189 / e215 (the family x height model + the committed "
            "54-probe records)",
            "T187 / e214 (the archive + the committed curves/journal)",
            "T183 / e182c2 + T149 / e182c + T123 / e182 (the two-wash 124M "
            "archive)",
        ],
        "whats_new": [
            "the SIX-FEATURE TOKEN TABLE on the committed prompts/answers "
            "(the tokenizer's own view of the battery: answer tokens / "
            "fragmentation, cue length, prompt length, first-token "
            "commonness in the source corpus, answer-cue overlap)",
            "the CORRELATION TEST: Spearman per feature vs e216's "
            "re-derived residual, against the frozen 0.4/0.2 bars",
            "the ABSORPTION BATTERY: family + each feature (and the joint "
            "composite) SUBSTITUTED as the height term — the decisive "
            "cross-wash test against the 0.934 baseline",
            "the named splits re-read under the best token model "
            "(e216's yardsticks verbatim)",
            "the GMAIL/IPHONE case study: token features vs opposite "
            "residuals",
            "the Bonferroni note across the feature x wash test count",
        ],
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
    e216p = common.REPO / "runs" / "e216" / "metrics.json"
    for p in (e215p, e214p, e214j, e216p, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e215m = json.loads(e215p.read_text(encoding="utf-8"))
    e214m = json.loads(e214p.read_text(encoding="utf-8"))
    e214j = json.loads(e214j.read_text(encoding="utf-8"))["states"]
    e216m = json.loads(e216p.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    assign = e215m["typology"]["assignment"]
    e216_res_tab = {(r["fact"], r["battery"]): r
                    for r in e216m["residual"]["table"]}
    prov = git_record()
    G_RECORDS = {
        "e215_status": e215m["status"], "n_probes": len(assign),
        "e216_status": e216m["status"],
        "e215_gates": {g: e215m["adjudication"]["gates_summary"][g]
                       for g in ("G_STATES", "G_BATT", "G_CORPUS", "G_ENV")},
        "e216_gates": {g: e216m["adjudication"]["gates_summary"][g]
                       for g in ("G_RECORDS", "G_RECOMPUTE", "G_CORPUS",
                                 "G_PROMPT", "G_ENV")},
        "e215_reprobe_max_dp": e215m["gates"]["G_STATES"]["reprobe_max_dp"],
        "e216_xwash_committed": e216m["cross_wash_residual"]["spearman"],
        "git": prov,
        "note": "the records chain's own certifications are inherited (e215 "
                "re-probed both +80 states at dp 0.0 vs e214's records; "
                "e214 certified those vs the e182c/e182c2 originals at "
                "<= 3.3e-06; e216 added the committed residual table); this "
                "cell adds the desk recompute (G_RECOMPUTE), the baseline "
                "reproduction (G_BASELINE), git provenance and the "
                "token-feature table",
    }
    G_RECORDS["pass"] = bool(
        e215m["status"] == "DONE" and e216m["status"] == "DONE"
        and len(assign) == 54
        and all(G_RECORDS["e215_gates"].values())
        and all(G_RECORDS["e216_gates"].values())
        and not prov["dirty_records"])
    metrics["gates"] = {"G_RECORDS": G_RECORDS}
    log(f"G_RECORDS: {'PASS' if G_RECORDS['pass'] else 'FAIL'} "
        f"(e215 {e215m['status']}, e216 {e216m['status']}, {len(assign)} "
        f"probes, HEAD {prov['head'][:9]})")
    write_metrics("PARTIAL: records read, provenance taken")

    # ------------------------------------------ P2 the desk recompute (G_RECOMPUTE)
    t0j = [r for r in e214j if r["wash"] == "t0"][0]
    cur_w1 = {int(s): e for s, e in e214m["curves"]["w1"].items()}
    cur_w2 = {int(s): e for s, e in e214m["curves"]["w2"].items()}
    assert DEEPEST in cur_w1 and DEEPEST in cur_w2
    max_dp = 0.0
    rows = []
    for r in assign:
        p0_rc = t0j[r["battery"]]["probes"][r["fact"]]["p"]
        p80w1_rc = cur_w1[DEEPEST][r["battery"]]["probes"][r["fact"]]
        p80w2_rc = cur_w2[DEEPEST][r["battery"]]["probes"][r["fact"]]
        for a, bb in ((p0_rc, r["p0"]), (p80w1_rc, r["p80_w1"]),
                      (p80w2_rc, r["p80_w2"]),
                      (p80w1_rc / p0_rc, r["hr_w1"]),
                      (p80w2_rc / p0_rc, r["hr_w2"])):
            max_dp = max(max_dp, abs(a - bb))
        rows.append(dict(r))                    # the committed row, verbatim
    G_RECOMPUTE = {
        "recomputed_from": "e214 curves w1/w2 +{80} and journal t=0, matched "
                           "per probe by fact+battery (the e216/e218/e219 "
                           "recompute, verbatim)",
        "max_dp_vs_e215_assignment": max_dp, "tol": TOL_RECOMPUTE_DP,
        "n_rows": len(rows),
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

    # -------------------------- P3 the baseline reproduction (G_BASELINE, e216's model)
    fams = [r["family6"] for r in rows]
    p0s = [r["p0"] for r in rows]
    p0_mean = sum(p0s) / len(p0s)
    y_w1 = np.array([r["hr_w1"] for r in rows])
    y_w2 = np.array([r["hr_w2"] for r in rows])
    y_mean = np.array([r["hr_mean"] for r in rows])
    X_base = design(fams, [np.array(p0s) - p0_mean])
    base = {}
    for tag, y in (("w1", y_w1), ("w2", y_w2), ("mean", y_mean)):
        base[tag] = ols(X_base, y)
    for i, r in enumerate(rows):
        r["fit_w1"] = float(base["w1"]["fitted"][i])
        r["fit_w2"] = float(base["w2"]["fitted"][i])
        r["resid_w1"] = float(base["w1"]["resid"][i])
        r["resid_w2"] = float(base["w2"]["resid"][i])
    max_dp_base = 0.0
    for r in rows:
        er = e216_res_tab[(r["fact"], r["battery"])]
        for a, bb in ((r["resid_w1"], er["resid_w1"]),
                      (r["resid_w2"], er["resid_w2"]),
                      (r["fit_w1"], er["fit_w1"]),
                      (r["fit_w2"], er["fit_w2"])):
            max_dp_base = max(max_dp_base, abs(a - bb))
    xw_base = spearman([r["resid_w1"] for r in rows],
                       [r["resid_w2"] for r in rows])
    G_BASELINE = {
        "compared": "this cell's family+p0 OLS vs e216's committed per-probe "
                    "residual table (resid + fitted, both washes)",
        "max_dp": max_dp_base, "tol": TOL_RECOMPUTE_DP,
        "xw_spearman_this_cell": xw_base,
        "xw_spearman_e216": e216m["cross_wash_residual"]["spearman"],
        "base_r2": {"w1": base["w1"]["r2"], "w2": base["w2"]["r2"]},
        "note": "the e216 baseline REPRODUCED (extend, don't repeat): the "
                "token features are read only after this bit-check passes; "
                "every feature test below reads THIS cell's re-derived "
                "residuals",
    }
    G_BASELINE["pass"] = bool(
        max_dp_base <= TOL_RECOMPUTE_DP
        and xw_base is not None
        and abs(xw_base - e216m["cross_wash_residual"]["spearman"]) <= 1e-9)
    metrics["gates"]["G_BASELINE"] = G_BASELINE
    log(f"G_BASELINE: {'PASS' if G_BASELINE['pass'] else 'FAIL'} "
        f"(max dp {max_dp_base:.2e}; xw {xw_base:.9f} vs committed "
        f"{e216m['cross_wash_residual']['spearman']:.9f})")
    write_metrics("PARTIAL: the e216 baseline reproduced")
    if not G_BASELINE["pass"] and not SMOKE:
        log("FATAL: e216's committed residuals not reproduced — aborting")
        return 1

    # ------------------------- P4 the corpus + source-token counts (G_CORPUS)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(e1.MODEL_REPO,
                                            revision=e1.MODEL_REV)
    import transformers
    metrics["tokenizer"] = {"repo": e1.MODEL_REPO, "revision": e1.MODEL_REV,
                            "transformers_version": transformers.__version__,
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
        "note": "rebuilt TOKENIZER-ONLY (the e214/e215/e216/e219 chain) — "
                "the substrate of the first-token commonness feature and "
                "the cue/prompt rebuild; no wash, no model",
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

    # the SOURCE-corpus token counts (the commonness substrate)
    src_ids = np.array(tok.encode(text), dtype=np.int64)
    src_counts = np.bincount(src_ids, minlength=50257).astype(np.int64)
    src_total = int(src_ids.shape[0])
    filt_ids = np.array(tok.encode(filtered), dtype=np.int64)
    filt_counts = np.bincount(filt_ids, minlength=50257).astype(np.int64)
    filt_total = int(filt_ids.shape[0])
    log(f"source corpus: {src_total:,} tokens; filtered corpus: "
        f"{filt_total:,} tokens (first-token frequency substrate)")

    # ------------------------------- P5 the batteries + cues rebuilt (G_PROMPT)
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

    def cue_of(r) -> str:
        """The prompt's final query clause, rebuilt from the frozen
        templates (the module's own constants)."""
        b, rel = r["battery"], r["relation"]
        if b == "fact":
            _, query = e1.REL_TMPL[rel]
            return query.format(c=r["subject"])
        if b == "ctrl":
            _, query = e1.CTRL_TMPL[rel]
            return query.format(c=r["subject"])
        if b == "near":
            _, query = e1.NEAR_TMPL
            return query.format(c=r["subject"])
        assert b == "tmpl", b
        cap = r["fact"].split("->")[0]      # fact = f"{cap}->{state}"
        _, query = e2.TMPL2
        return query.format(a=cap, c=r["answer"])

    prompt_strs: dict[str, str] = {}
    prompt_ids: dict[str, list[int]] = {}
    G_PROMPT = {"batteries": {}}
    cue_ok_all = True
    for b in BATTERIES:
        order_equal = [r["fact"] for r in rebuilt[b]] == kept_by_battery[b]
        by_fact = {r["fact"]: r for r in rebuilt[b]}
        max_dp_len = 0
        cue_ok = 0
        for r in rows:
            if r["battery"] != b:
                continue
            cand = by_fact[r["fact"]]
            ids_l = cand["ids"][0].tolist()
            prompt_ids[r["fact"]] = ids_l
            prompt_strs[r["fact"]] = cand["prompt"]
            max_dp_len = max(max_dp_len,
                             abs(len(ids_l) - r["prompt_tok_len"]))
            if cand["prompt"].endswith(cue_of(r)):
                cue_ok += 1
            else:
                cue_ok_all = False
        n_b = len(kept_by_battery[b])
        G_PROMPT["batteries"][b] = {
            "n": n_b, "order_equal_committed": order_equal,
            "max_prompt_toklen_dp": max_dp_len,
            "cue_suffix_ok": f"{cue_ok}/{n_b}"}
    G_PROMPT["cue_rule"] = ("cue = the template query clause "
                            "(REL_TMPL/CTRL_TMPL/NEAR_TMPL/TMPL2 .format on "
                            "the committed subject/answer); certified by "
                            "prompt.endswith(cue) on every kept probe")
    G_PROMPT["note"] = ("candidate construction VERBATIM by module import "
                        "(no model); kept membership from the committed "
                        "rows; order + prompt token lengths cross-checked; "
                        "the cue is the certified final query clause")
    G_PROMPT["pass"] = bool(all(
        G_PROMPT["batteries"][b]["order_equal_committed"]
        and G_PROMPT["batteries"][b]["max_prompt_toklen_dp"] == 0
        and G_PROMPT["batteries"][b]["cue_suffix_ok"].split("/")[0]
        == str(G_PROMPT["batteries"][b]["n"])
        and G_PROMPT["batteries"][b]["n"] > 0 for b in BATTERIES))
    metrics["gates"]["G_PROMPT"] = G_PROMPT
    log(f"G_PROMPT: {'PASS' if G_PROMPT['pass'] else 'FAIL'} "
        + " | ".join(f"{b}: n {G_PROMPT['batteries'][b]['n']}, order "
                     f"{G_PROMPT['batteries'][b]['order_equal_committed']}, "
                     f"cue {G_PROMPT['batteries'][b]['cue_suffix_ok']}"
                     for b in BATTERIES))
    write_metrics("PARTIAL: prompts + cues rebuilt + certified")

    # ------------------------------------------ P6 THE TOKEN FEATURES (the table)
    for r in rows:
        a_str = " " + r["answer"]
        a_ids_l = tok.encode(a_str)
        cue = cue_of(r)
        c_ids_l = tok.encode(cue)
        p_ids_l = prompt_ids[r["fact"]]
        aid0 = a_ids_l[0]
        r["ans_tok_count"] = float(len(a_ids_l))
        r["ans_frag"] = len(a_ids_l) / len(a_str)
        r["cue_tok_count"] = float(len(c_ids_l))
        r["ans_first_tok_id"] = int(aid0)
        r["first_tok_str"] = tok.decode([aid0])
        r["ans_first_tok_count_src"] = int(src_counts[aid0])
        r["ans_first_tok_freq"] = (src_counts[aid0] / src_total) * 1e6
        a_set, c_set, p_set = set(a_ids_l), set(c_ids_l), set(p_ids_l)
        r["ans_cue_overlap"] = (len(a_set & c_set) / len(a_set)
                                if a_set else None)
        # competitors
        r["ans_prompt_overlap"] = (len(a_set & p_set) / len(a_set)
                                   if a_set else None)
        r["ans_first_tok_freq_filtered"] = (filt_counts[aid0] / filt_total) * 1e6
        r["ans_char_len"] = float(len(a_str))
        r["cue_char_len"] = float(len(cue))
        r["cue"] = cue

    # degeneracy audit (which features vary?)
    def _var(key):
        vals = [r[key] for r in rows]
        return len({round(v, 12) for v in vals}) > 1
    varying = [k for _, k, _ in TOKEN_FEATURES if _var(k)]
    constant = [k for _, k, _ in TOKEN_FEATURES if not _var(k)]
    log("token features: varying " + ", ".join(varying)
        + (" | CONSTANT (excluded from OLS forms, disclosed): "
           + ", ".join(constant) if constant else ""))

    feature_table = []
    for r in rows:
        feature_table.append({
            "fact": r["fact"], "battery": r["battery"],
            "family6": r["family6"], "answer": r["answer"],
            "p0": r["p0"], "hr_w1": r["hr_w1"], "hr_w2": r["hr_w2"],
            "resid_w1": r["resid_w1"], "resid_w2": r["resid_w2"],
            "ans_tok_count": r["ans_tok_count"], "ans_frag": r["ans_frag"],
            "cue_tok_count": r["cue_tok_count"],
            "prompt_tok_len": float(r["prompt_tok_len"]),
            "ans_first_tok_id": r["ans_first_tok_id"],
            "first_tok_str": r["first_tok_str"],
            "ans_first_tok_count_src": r["ans_first_tok_count_src"],
            "ans_first_tok_freq": r["ans_first_tok_freq"],
            "ans_cue_overlap": r["ans_cue_overlap"],
            "cue": r["cue"],
            "ans_prompt_overlap": r["ans_prompt_overlap"],
            "ans_first_tok_freq_filtered": r["ans_first_tok_freq_filtered"],
        })
    metrics["token_features"] = {
        "definition": "the six registered features computed on the committed "
                      "prompts/answers with the pinned GPT-2 tokenizer; "
                      "prompts + cues certified by G_PROMPT; the first-token "
                      "frequency is per 1M tokens of the SOURCE corpus "
                      "(data/input.txt, tokenizer-only)",
        "varying": varying, "constant_excluded_from_ols": constant,
        "per_family_medians": {
            f: {k: _mean([r[k] for r in rows if r["family6"] == f])
                for k in ("ans_frag", "cue_tok_count", "prompt_tok_len",
                          "ans_first_tok_freq")}
            for f in FAMILY6_ORDER},
        "table": feature_table,
    }
    write_metrics("PARTIAL: the token-feature table")
    log("first-token frequencies (per 1M, source corpus): "
        + " | ".join(
            f"{f}: {_mean([r['ans_first_tok_freq'] for r in rows if r['family6'] == f]):.0f}"
            for f in FAMILY6_ORDER))

    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "model_loads": 0, "gpu_calls": 0,
             "load_checks": len(load_checks),
             "burst": "tokenizer + corpus filter + candidate construction + "
                      "source-corpus token counts (no model, no probes, no "
                      "wash)",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV

    # --------------------------------------- P7 THE CORRELATION TEST (part 1)
    feats_corr = {"registered": [], "competitors": []}
    for label, key, note in TOKEN_FEATURES:
        fv = [r[key] for r in rows]
        r1 = spearman(fv, [r["resid_w1"] for r in rows])
        r2_ = spearman(fv, [r["resid_w2"] for r in rows])
        mx = max((abs(x) for x in (r1, r2_) if x is not None), default=0.0)
        feats_corr["registered"].append({
            "label": label, "key": key, "note": note,
            "value_range": [min(fv), max(fv)],
            "degenerate": key in constant,
            "rho_w1": r1, "rho_w2": r2_,
            "p_w1": spearman_p(r1, len(rows)) if r1 is not None else None,
            "p_w2": spearman_p(r2_, len(rows)) if r2_ is not None else None,
            "fires": bool(mx >= CORR_BAR and key in varying),
            "silent": bool(mx < SILENT_BAR),
            "firing_stat": f"max |rho| over the two washes vs {CORR_BAR}",
        })
        log(f"token feature {label:22s}: rho w1 "
            f"{r1 if r1 is None else round(r1, 3)} / w2 "
            f"{r2_ if r2_ is None else round(r2_, 3)}"
            f"{'  (CONSTANT)' if key in constant else ''}")
    for label, key, note in TOKEN_COMPETITORS:
        fv = [r[key] for r in rows]
        feats_corr["competitors"].append({
            "label": label, "key": key, "note": note,
            "value_range": [min(fv), max(fv)],
            "rho_w1": spearman(fv, [r["resid_w1"] for r in rows]),
            "rho_w2": spearman(fv, [r["resid_w2"] for r in rows])})
    metrics["correlation_test"] = {
        "definition": f"Spearman(feature, resid_w) per wash vs THIS cell's "
                      f"re-derived e216 residuals; fires at max |rho| >= "
                      f"{CORR_BAR}; silent at max |rho| < {SILENT_BAR}",
        **feats_corr,
    }
    write_metrics("PARTIAL: the correlation test")

    # --------------------------------------- P8 THE ABSORPTION BATTERY (part 2)
    absorb = {"forms": [], "definition":
              "OLS hr_w ~ family + feature(s) centered (p0 SUBSTITUTED out, "
              "e219's convention); xwash = Spearman(resid_w1, resid_w2) of "
              "the substituted model; ABSORBED iff |xwash| < "
              f"{ABSORB_BAR} for ANY form"}
    feat_cols = {}
    for key in varying:
        vals = np.array([r[key] for r in rows], dtype=float)
        feat_cols[key] = vals - vals.mean()

    def run_form(label, keys, additive_too=True):
        cols = [feat_cols[k] for k in keys]
        X = design(fams, cols)
        entry = {"label": label, "features": list(keys),
                 "r2": {}, "xwash_spearman": None, "xwash_pearson": None,
                 "absorbs": False}
        fits = {}
        for tag, y in (("w1", y_w1), ("w2", y_w2)):
            fits[tag] = ols(X, y)
            entry["r2"][tag] = fits[tag]["r2"]
        e1r = fits["w1"]["resid"]
        e2r = fits["w2"]["resid"]
        entry["xwash_spearman"] = spearman(e1r.tolist(), e2r.tolist())
        entry["xwash_pearson"] = _pearson(e1r.tolist(), e2r.tolist())
        entry["absorbs"] = bool(entry["xwash_spearman"] is not None
                                and abs(entry["xwash_spearman"]) < ABSORB_BAR)
        entry["resid_w1"] = e1r.tolist()
        entry["resid_w2"] = e2r.tolist()
        entry["resid_sd"] = {"w1": fits["w1"]["resid_sd"],
                             "w2": fits["w2"]["resid_sd"]}
        # additive co-report: family + p0 + feature(s)
        if additive_too:
            Xa = design(fams, [np.array(p0s) - p0_mean] + cols)
            fa1 = ols(Xa, y_w1)
            fa2 = ols(Xa, y_w2)
            entry["additive_xwash_spearman"] = spearman(
                fa1["resid"].tolist(), fa2["resid"].tolist())
            entry["additive_r2"] = {"w1": fa1["r2"], "w2": fa2["r2"]}
        absorb["forms"].append(entry)
        log(f"absorption form {label:34s}: |xwash| "
            f"{abs(entry['xwash_spearman']):.3f}"
            f"{'  ABSORBS' if entry['absorbs'] else ''}"
            f"  (R2 {entry['r2']['w1']:.3f}/{entry['r2']['w2']:.3f}; "
            f"additive |xwash| "
            f"{abs(entry.get('additive_xwash_spearman') or 0):.3f})")
        return entry

    for label, key, _ in TOKEN_FEATURES:
        if key in varying:
            run_form(f"family + {label}", [key])
    composite_entry = None
    if len(varying) >= 2:
        composite_entry = run_form(
            f"family + COMPOSITE ({len(varying)} feats)", list(varying))
    # the p0 baseline is G_BASELINE's own fit — recorded as a reference row
    absorb["forms"].append({
        "label": "family + p0 (the e216 baseline)", "features": ["p0"],
        "r2": {"w1": base["w1"]["r2"], "w2": base["w2"]["r2"]},
        "xwash_spearman": xw_base,
        "xwash_pearson": _pearson([r["resid_w1"] for r in rows],
                                  [r["resid_w2"] for r in rows]),
        "absorbs": bool(abs(xw_base) < ABSORB_BAR),
        "reference": True,
    })

    any_absorb = bool(any(f.get("absorbs") for f in absorb["forms"]
                          if not f.get("reference")))
    absorb["any_absorption"] = any_absorb
    absorb["baseline_xwash"] = xw_base

    # the supervised composite's correlation score (generous; Bonferroni-noted)
    if composite_entry is not None:
        Xc = design(fams, [feat_cols[k] for k in varying])
        betac = ols(Xc, y_mean)["beta"]
        score = np.column_stack([feat_cols[k] for k in varying]) @ \
            np.array(betac[6:])            # the feature part of the fit
        sc1 = spearman(score.tolist(), [r["resid_w1"] for r in rows])
        sc2 = spearman(score.tolist(), [r["resid_w2"] for r in rows])
        feats_corr["composite"] = {
            "label": "composite (supervised)", "key": "__composite__",
            "note": "the feature-part of the composite substitution model's "
                    "hr_mean fit — SUPERVISED (weights fit to the outcome); "
                    "generous by construction, covered by the Bonferroni note",
            "rho_w1": sc1, "rho_w2": sc2,
            "p_w1": spearman_p(sc1, len(rows)) if sc1 is not None else None,
            "p_w2": spearman_p(sc2, len(rows)) if sc2 is not None else None,
            "fires": bool(max(abs(x) for x in (sc1, sc2)) >= CORR_BAR),
            "silent": bool(max(abs(x) for x in (sc1, sc2)) < SILENT_BAR),
        }
        log(f"composite (supervised): rho w1 "
            f"{sc1 if sc1 is None else round(sc1, 3)} / w2 "
            f"{sc2 if sc2 is None else round(sc2, 3)}")
    metrics["correlation_test"] = {
        "definition": metrics["correlation_test"]["definition"],
        **feats_corr,
    }
    metrics["absorption_test"] = absorb
    write_metrics("PARTIAL: the absorption battery")

    # --------------- P8b the named splits under the BEST token model (part 3)
    subst_forms = [f for f in absorb["forms"] if not f.get("reference")]
    best_form = min(subst_forms,
                    key=lambda f: (abs(f["xwash_spearman"]),
                                   len(f["features"])))
    absorb["best_form"] = {"label": best_form["label"],
                           "features": best_form["features"],
                           "xwash_spearman": best_form["xwash_spearman"]}
    for i, r in enumerate(rows):
        r["bres_w1"] = best_form["resid_w1"][i]
        r["bres_w2"] = best_form["resid_w2"][i]
    prod = [r for r in rows if r["family6"] == "product"]
    hold_g = [r for r in prod if r["answer"] in PRODUCT_HOLD_GROUP]
    anchor = [r for r in prod if r["answer"] == PRODUCT_COLLAPSE_ANCHOR]
    others = [r for r in prod if r not in hold_g and r not in anchor]
    gap_sd = {}
    for w in ("w1", "w2"):
        gap = (_mean([r[f"bres_{w}"] for r in hold_g])
               - _mean([r[f"bres_{w}"] for r in anchor]))
        gap_sd[w] = gap / best_form["resid_sd"][w]
    prod_survives = bool(gap_sd["w1"] >= SPLIT_SD_BAR
                         and gap_sd["w2"] >= SPLIT_SD_BAR)
    rc = [r for r in rows if r["family6"] == "rev-capital"]
    ratios = {w: _sd([r[f"bres_{w}"] for r in rc]) / _sd([r[f"hr_{w}"] for r in rc])
              for w in ("w1", "w2")}
    tmpl_dissolves = bool(ratios["w1"] <= WIDTH_DISSOLVE_BAR
                          and ratios["w2"] <= WIDTH_DISSOLVE_BAR)
    splits_best = {
        "under_model": best_form["label"],
        "product": {
            "definition": f"contrast = mean resid{PRODUCT_HOLD_GROUP} - "
                          f"mean resid[{PRODUCT_COLLAPSE_ANCHOR}] under the "
                          f"best token model, residual-SD units; SURVIVES "
                          f"iff >= {SPLIT_SD_BAR} SD both washes "
                          f"(descriptive, never adjudicated)",
            "gap_sd_w1": gap_sd["w1"], "gap_sd_w2": gap_sd["w2"],
            "residuals": {r["answer"]: [r["bres_w1"], r["bres_w2"]]
                          for r in prod},
            "others_mean_resid": {w: _mean([r[f"bres_{w}"] for r in others])
                                  for w in ("w1", "w2")},
            "verdict": "SURVIVES (structured)" if prod_survives else
                       "DISSOLVES (the token model accounts for it)",
        },
        "tmpl_width": {
            "definition": f"within rev-capital under the best token model: "
                          f"SD(resid)/SD(hr) per wash; DISSOLVES iff <= "
                          f"{WIDTH_DISSOLVE_BAR} on both washes "
                          f"(descriptive)",
            "ratio_w1": ratios["w1"], "ratio_w2": ratios["w2"],
            "verdict": "DISSOLVES (the token model accounts for the width)"
                       if tmpl_dissolves else "SURVIVES (width unabsorbed)",
        },
    }

    metrics["named_splits_best_model"] = splits_best
    log(f"named splits under {best_form['label']}: product "
        f"{gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD -> "
        f"{splits_best['product']['verdict']}; tmpl width "
        f"{ratios['w1']:.2f}/{ratios['w2']:.2f} -> "
        f"{splits_best['tmpl_width']['verdict']}")
    write_metrics("PARTIAL: the named splits under the best token model")

    # --------------------------------------- P9 THE CASE STUDY (part 4)
    def case_row(ans):
        r = [x for x in rows if x["answer"] == ans][0]
        return {
            "answer": ans, "fact": r["fact"], "family6": r["family6"],
            "ans_tok_count": r["ans_tok_count"], "ans_frag": r["ans_frag"],
            "cue_tok_count": r["cue_tok_count"],
            "prompt_tok_len": r["prompt_tok_len"],
            "first_tok_str": r["first_tok_str"],
            "ans_first_tok_id": r["ans_first_tok_id"],
            "ans_first_tok_freq": r["ans_first_tok_freq"],
            "ans_first_tok_count_src": r["ans_first_tok_count_src"],
            "ans_cue_overlap": r["ans_cue_overlap"],
            "hr_w1": r["hr_w1"], "hr_w2": r["hr_w2"],
            "resid_w1": r["resid_w1"], "resid_w2": r["resid_w2"],
            "best_model_resid_w1": r["bres_w1"],
            "best_model_resid_w2": r["bres_w2"],
        }
    g_row, i_row = case_row(CASE_PAIR[0]), case_row(CASE_PAIR[1])
    case = {
        "pair": list(CASE_PAIR),
        "gmail": g_row, "iphone": i_row,
        "product_family": [case_row(a) for a in
                           ("Xbox", "Chrome", "iPhone", "iPad", "Gmail",
                            "iTunes", "PlayStation")],
        "reading": ("the hunt's sharpest single anchor: Gmail holds on both "
                    "washes (hr ~0.90/0.93, resid +0.43/+0.43) while iPhone "
                    "dies (hr 0.18/0.34, resid -0.21/-0.06) — do their token "
                    "features differ in any way that tracks the sign?"),
    }
    metrics["case_study"] = case
    log(f"case study: Gmail '{g_row['first_tok_str']}' @ "
        f"{g_row['ans_first_tok_freq']:.0f}/1M, frag {g_row['ans_frag']:.3f}, "
        f"cue {g_row['cue_tok_count']} tok vs iPhone "
        f"'{i_row['first_tok_str']}' @ {i_row['ans_first_tok_freq']:.0f}/1M, "
        f"frag {i_row['ans_frag']:.3f}, cue {i_row['cue_tok_count']} tok")
    write_metrics("PARTIAL: the case study")

    # --------------------------------------------- P10 ADJUDICATION (frozen)
    corr_entries = feats_corr["registered"] + (
        [feats_corr["composite"]] if "composite" in feats_corr else [])
    firing = [f for f in corr_entries if f.get("fires")]
    all_silent = bool(all(f.get("silent") for f in corr_entries))
    gates_ok = bool(G_RECORDS["pass"] and G_RECOMPUTE["pass"]
                    and G_BASELINE["pass"] and G_CORPUS["pass"]
                    and G_PROMPT["pass"] and G_ENV["pass"])
    m_tests = len(corr_entries) * 2          # the Bonferroni count
    best_rho = max((max(abs(f["rho_w1"] or 0), abs(f["rho_w2"] or 0))
                    for f in corr_entries), default=0.0)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_RECORDS", G_RECORDS["pass"]),
                           ("G_RECOMPUTE", G_RECOMPUTE["pass"]),
                           ("G_BASELINE", G_BASELINE["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_PROMPT", G_PROMPT["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only"
    elif firing or any_absorb:
        verdict = "TOKEN-NAMES-IT"
        named = []
        if firing:
            named.append("correlation: " + ", ".join(
                f"{f['label']} (rho w1 {f['rho_w1']:.3f} / w2 "
                f"{f['rho_w2']:.3f})" for f in firing))
        if any_absorb:
            absorbing = [f for f in subst_forms if f["absorbs"]]
            named.append("absorption: " + ", ".join(
                f"{f['label']} (|xwash| {abs(f['xwash_spearman']):.3f} < "
                f"{ABSORB_BAR})" for f in absorbing))
        clause = ("a token feature (or composite) correlates |rho| >= "
                  f"{CORR_BAR} with the residual or absorbs the cross-wash "
                  f"rho below {ABSORB_BAR} — the third dimension is TOKEN "
                  f"STRUCTURE; named — " + "; ".join(named)
                  + f"; the baseline xwas {xw_base:.3f}")
    elif all_silent and not any_absorb:
        verdict = "TOKEN-SILENT"
        clause = (f"all token features sit |rho| < {SILENT_BAR} with no "
                  f"absorption — token structure retired; the hunt's last "
                  f"candidate (compositionality) owed. Best |rho| "
                  f"{best_rho:.3f} over {len(corr_entries)} features x 2 "
                  f"washes; every substitution leaves |xwash| >= "
                  f"{min(abs(f['xwash_spearman']) for f in subst_forms):.3f} "
                  f"vs the {ABSORB_BAR} bar (baseline {xw_base:.3f}); the "
                  f"named splits under the best token model: product "
                  f"{splits_best['product']['verdict']}, tmpl width "
                  f"{splits_best['tmpl_width']['verdict']}")
    else:
        why = []
        mid = [f for f in corr_entries
               if not f.get("silent") and not f.get("fires")]
        if mid:
            why.append("features in the middle band (" + ", ".join(
                f"{f['label']}: w1 "
                f"{'--' if f['rho_w1'] is None else format(f['rho_w1'], '+.3f')}"
                f" / w2 "
                f"{'--' if f['rho_w2'] is None else format(f['rho_w2'], '+.3f')}"
                f", between {SILENT_BAR} and {CORR_BAR}") + ")")
        why.append(f"no absorption (best |xwash| "
                   f"{min(abs(f['xwash_spearman']) for f in subst_forms):.3f} "
                   f">= {ABSORB_BAR})")
        verdict = "GRADED"
        clause = ("any partial — " + "; ".join(why)
                  + f"; the named splits under the best token model "
                  f"({best_form['label']}): product "
                  f"{splits_best['product']['verdict']} (contrast "
                  f"{gap_sd['w1']:+.2f}/{gap_sd['w2']:+.2f} SD), tmpl width "
                  f"{splits_best['tmpl_width']['verdict']} (ratio "
                  f"{ratios['w1']:.2f}/{ratios['w2']:.2f}); the tables "
                  f"verbatim")

    gates_summary = {"G_RECORDS": G_RECORDS["pass"],
                     "G_RECOMPUTE": G_RECOMPUTE["pass"],
                     "G_BASELINE": G_BASELINE["pass"],
                     "G_CORPUS": G_CORPUS["pass"],
                     "G_PROMPT": G_PROMPT["pass"], "G_ENV": G_ENV["pass"]}
    adj = {
        "bars": {"TOKEN_NAMES_IT": bool((firing or any_absorb)
                                        and gates_ok and not SMOKE),
                 "TOKEN_SILENT": bool(all_silent and not any_absorb
                                      and gates_ok and not SMOKE
                                      and not (firing or any_absorb)),
                 "verdict": verdict, "clause": clause,
                 "order": "TOKEN-NAMES-IT -> TOKEN-SILENT -> GRADED (gated "
                          "on G_RECORDS/G_RECOMPUTE/G_BASELINE/G_CORPUS/"
                          "G_PROMPT/G_ENV)"},
        "clauses": {
            "any_feature_or_composite_maxwash_rho >= 0.40": bool(firing),
            "any_substitution_absorbs (|xwash| < 0.50)": any_absorb,
            "all_features_and_composite_rho < 0.20": all_silent,
            "firing_features": [f["label"] for f in firing],
            "absorbing_forms": [f["label"] for f in subst_forms
                                if f["absorbs"]],
        },
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E220 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    # ------------------------------------------------ P11 honesty + close
    metrics["honesty_reflex"] = {
        "construction_constrained_features": "the battery's own rules "
            "pre-constrain the dispatch's feature list: e182's construction "
            "drops every multi-token answer (t1 constant = 1, t2 a "
            "char-length proxy), the contamination scans remove answer "
            "tokens from the wash train stream, and the rotating-exemplar "
            "rule keeps the answer out of its own prompt (t6 and the "
            "answer-prompt overlap expected 0) — a TOKEN-SILENT verdict is "
            "partly BY CONSTRUCTION, disclosed wherever read",
        "multiple_comparisons": f"{m_tests} correlation tests "
            f"({len(corr_entries)} features/composite x 2 washes) plus "
            f"{len(subst_forms)} substitution forms — the Bonferroni note: "
            f"at alpha 0.05 family-wise, per-test alpha "
            f"{0.05 / max(1, m_tests):.4f}; with n=54 that demands roughly "
            f"|rho| > 0.44 for a single test to clear family-wise "
            f"significance — the {CORR_BAR} bar is near that line, and the "
            "SUPERVISED composite gets its fit for free; any marginal "
            "firing must be read with this note",
        "supervised_composite": "the composite's correlation score uses "
            "weights fit to hr_mean under family controls — outcome-coupled "
            "by construction (generous); its absorption form is the honest "
            "version (a plain OLS model, but still selected over "
            "alternatives)",
        "frequency_proxy": "first-token commonness is proxied by the SOURCE "
            "corpus (data/input.txt, Shakespeare, tokenizer-only) — NOT "
            "GPT-2's true pretraining distribution; the filtered-corpus "
            "frequency is construction-limited (the banned-string filter) "
            "and co-reported as a competitor",
        "cue_definition": "the cue is the template query clause certified by "
                          "prompt.endswith(cue); within a family it varies "
                          "only through subject length — it is heavily "
                          "family-collinear, and the substitution model "
                          "cannot separate that from the family dummies",
        "n_washes": "n=2 washes is texture, not law; wash 1 a CPU fp32 "
                    "replay, wash 2 GPU fp32 — inherited (disclosed)",
        "typology_inherited": "the family factor is e215's HAND-REGISTERED "
                              "typology (not blind to the outcome) — every "
                              "family+feature R^2 inherits it",
        "ols_conventions": "raw bounded hr, plain OLS, homoskedastic SEs "
                           "never adjudicated; unbalanced family n's "
                           "(3..19)",
        "xwash_reading": "absorption by substitution means the token feature "
                         "carries what p0 carried AND orders the per-probe "
                         "misses; it does not license causation — the residual "
                         "could still be shared pipeline texture on a "
                         "different axis (e219's open boundary)",
        "guarantees_nothing": "the token table and its tests describe THESE "
                              "54 probes under THESE two washes with an "
                              "inherited hand-registered typology and a "
                              "corpus-proxied frequency — the openness is "
                              "the point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, ZERO model loads (tokenizer "
                    "only), decisive; no GPU calls",
        "load_checks": load_checks,
        "records_chain": ["runs/e214/metrics.json", "runs/e214/journal.json",
                          "runs/e215/metrics.json",
                          "runs/e216/metrics.json"],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, rows,
                        metrics["correlation_test"], absorb, splits_best,
                        case, adj, xw_base)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
