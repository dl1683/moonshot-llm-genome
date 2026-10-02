"""E219 — THE INDEPENDENT-ENTRENCHMENT CELL (T191's named hunt, first target).

WHY: e216/T190 found the THIRD SORTING DIMENSION — the family x height
model's per-probe misses replicate across washes at rho 0.934 — and
e218/T191 proved NO function of p0+family absorbs it (linear/rank/logit/
quad/family-slopes all leave rho >= 0.91). THE CONFOUND (T191's own
honesty boundary): BOTH washes read the same probes through the SAME
battery — identical prompt strings, identical p0 denominators — so the
replicating residual could be SHARED INSTRUMENT TEXTURE rather than
probe physics. THE BREAK THIS CELL REGISTERS: measure each probe's
entrenchment INDEPENDENTLY — at t=0 (the pristine state), under 2-3
DIFFERENT hand-registered prompt forms (a paraphrase battery: the same
knowledge asked differently — paraphrased cloze forms AND reversed
cue<->answer forms for the fact batteries; alternative natural
completion forms for the controls; the U.S.-state and capital-of
paraphrases for near/tmpl) — a channel sharing NO machinery with the
wash battery's reads (different strings, different exemplar sentences,
its own denominators). THEN: (1) does the independent entrenchment IE
correlate with e216's committed residual? (2) THE DECISIVE SPLIT —
re-run the e216 regression with IE SUBSTITUTED as the height term:
does IT absorb the cross-wash residual where every same-channel form
failed?

THE ARCHIVE (committed; this cell is a desk+one-burst-eval cell):
  * e216's committed 54-probe residual table (runs/e216/metrics.json)
    — the residual this cell hunts; re-derived desk-only through the
    e215 assignment -> e214 curves/journal chain (G_RECOMPUTE) and the
    e216 OLS fit (G_BASELINE, bit-checked at 1e-9).
  * the pristine t=0 state: the pinned pretrained GPT-2 124M itself
    (e1.load_organism, offline cache, dropout off) — the SAME pristine
    state e214's journal t0 records were measured on; re-certified here
    by rebuilding all four batteries VERBATIM (module import + the full
    screening gate) and matching per-probe t=0 p (G_T0).
  * the batteries/corpus rebuilt VERBATIM (the e214/e215 convention)
    so the paraphrase battery's exemplar mechanics mirror the original
    construction exactly.

THE PARAPHRASE BATTERY (hand-registered, documented per probe in
metrics.paraphrase): every kept probe gets 2-3 forms; template forms
are filled with the SAME rotating first-two-others exemplar rule as
the original battery (K_SHOT=2), but every sentence string is new;
reversed forms swap cue and answer (the dispatch's own example: "X is
spoken in" for the lang items); hand-registered per-item forms cover
the founder-first / maker-first / made-by constructions the pools'
descriptor subjects need. IE(probe) = the MEAN p over its measured
forms (frozen); per-form values, per-direction means and every dropped
form (multi-token answers) are co-reported.

REGISTERED BARS (frozen VERBATIM from the dispatch brief, QUEUE row
DISPATCHED 17:07Z / commit 1db7f74, BEFORE any compute; adjudicate
against exactly this; no bar shopping):
  - ENTRENCHMENT-REAL: "fires if the independent-channel p0 correlates
    with the residual (|rho| >= 0.4) or absorbs the cross-wash residual
    (rho < 0.5 when substituted as the height term) — the third
    dimension is ENTRENCHMENT seen through a second channel: real probe
    physics, the confound broken."
  - INSTRUMENT-TEXTURE: "fires if the independent channel shows no
    relation (|rho| < 0.2, no absorption) — the residual's replication
    is same-channel texture; the third dimension retires as an
    instrument artifact; W021's family grows."
  - GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * residual = e216's committed per-probe residuals (resid_w1/w2),
    re-derived here through G_RECOMPUTE + G_BASELINE at 1e-9 and then
    read from THIS cell's own recompute (identical by the gate).
  * IE(probe) = mean over its measured paraphrase forms of
    p_t0(answer_form | paraphrase prompt) — one pristine-model eval
    burst; forward and reversed directions mixed under the frozen mean
    (per-direction means co-reported, never adjudicated).
  * CORRELATION TEST: rho = Spearman(IE, resid_w) per wash; FIRES iff
    max |rho| over the two washes >= 0.40 (the e216 max-over-washes
    convention; the bar's "|rho| >= 0.4").
  * ABSORPTION TEST: the e216 full model with the height term
    SUBSTITUTED — OLS hr_w ~ 1 + 5 family dummies (reference lang) +
    IE centered at its own 54-probe mean — residuals per probe; xwash
    = Spearman(resid_w1, resid_w2); ABSORBED iff |xwash| < 0.5 (the
    bar's "(rho < 0.5)" read with e218's abs-convention — a
    negatively-replicating miss is not absorption either; disclosed).
  * NO-RELATION (the texture clause): max |rho| < 0.20 AND no
    absorption.
  * ADJUDICATION: gates ok -> ENTRENCHMENT-REAL iff (correlation fires
    OR absorbed); else INSTRUMENT-TEXTURE iff no-relation; else GRADED.
  * CO-REPORTS (never adjudicated): rho(IE, p0) — the channel's
    statistical independence from the same-channel height; the additive
    form family + p0 + IE (xwash + R2); the substituted model's R2 per
    wash; IE_forward / IE_reversed correlations; per-family IE medians;
    the named splits under the substituted model (the e216 yardsticks:
    product contrast >= 1.0 residual SD both washes SURVIVES; tmpl
    SD(resid)/SD(hr) <= 0.5 DISSOLVES); the e216 committed xwash
    (0.934) as the reference line.
  * Gated on G_RECORDS/G_RECOMPUTE/G_BASELINE/G_CORPUS/G_T0/G_PARA/
    G_ENV.

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_RECORDS — e215/e216 metrics DONE with gates passing, git
    provenance of the records chain clean (HEAD, per-file last commit,
    no dirty records).
  * G_RECOMPUTE — p0/hr re-derived from e214's curves+journal per
    probe, vs e215's committed assignment at 1e-9 (the e216/e218
    convention).
  * G_BASELINE — this cell's linear-form OLS reproduces e216's
    committed per-probe residuals (both washes) at 1e-9 (extend,
    don't repeat: the new channel is read only after the baseline
    bit-checks).
  * G_CORPUS — the frozen wash corpus rebuilt TOKENIZER-ONLY and
    asserted equal to e182's recorded filter stats (the substrate the
    battery builders' contamination scans need).
  * G_T0 — the pristine provenance: the four batteries rebuilt
    VERBATIM (module import, full screening gate), kept sets equal to
    e214's committed t0 sets, per-probe t=0 dp <= 0.010 vs the journal
    (expected ~0.0: same fp32 weights, same CPU fp32 probe path — the
    e215 precedent read exactly 0.0).
  * G_PARA — form-matching documented: every measured form's full
    prompt string + answer + p committed per probe; every kept probe
    has >= 2 measured forms; every dropped form (multi-token answer)
    recorded with its reason; the template registry committed verbatim.
  * G_ENV — the owner envelope: CPU-only (zero GPU calls), torch
    threads <= 4, load checks before launch / before the eval burst /
    after, one small eval burst, decisive.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the openness is
the point. The paraphrase forms are HAND-REGISTERED and NOT BLIND to
the outcome (the residual table was readable when the forms were
chosen); the defense is only that the forms were fixed for
naturalness/coverage per family BEFORE any p was measured, and every
string is committed with the cell. The channel is independent of the
NAMED confound (same strings + shared p0 denominators), not of
everything: it shares the organism, the answer tokens (forward forms),
the k-shot structure and the probe pools with the wash battery —
rho(IE, p0) co-reports the residual statistical dependence. Reversed
forms measure the reverse association (a different trace — the lab's
own forward/backward asymmetry); IE mixes directions under the frozen
mean. The outcome hr still divides by the same-channel p0 (the
committed outcome; independence enters only through the predictor/
regressor). n=2 washes is texture, not law; wash 1 is a CPU fp32
replay of e182's GPU original, wash 2 GPU fp32 — inherited, disclosed.
Paraphrase p can sit near 0 for form-following failures — per-family
form medians reported; IE floor effects disclosed.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02 dispatch): CPU-only;
threads <= 4; load-check before launch and around the burst; ONE small
eval burst (battery screening ~90 forwards + t=0 + ~150 paraphrase
forwards, 124M CPU fp32); no wash, no bank ppl, no NOTES/THINKING/
QUEUE/STATE edits (dispatch). Progressive PARTIAL metrics after every
phase (the standing disruption rule).

PROVENANCE: the residual is e216's (re-derived through the committed
chain); the archive, batteries, corpus and organism are e182c/e182c2
VERBATIM via module import (the e214/e215 convention). Builds on:
T191/e218 + T190/e216 (the named hunt; the residual and the confound),
T189/e215 (the family x height model + the committed 54-probe
records), T187/e214 (the archive), T183/e182c2 + T149/e182c + T123/
e182 (the two-wash 124M archive), W028 (the shape/height law). NEW:
the hand-registered paraphrase battery (2-3 independent forms per
probe, every string committed); the independent-entrenchment IE table;
the correlation test IE-vs-residual; the SUBSTITUTION absorption test
(family + IE replacing p0 as the height term); the three-bar
adjudication.

Run:  cd lab && python e219_entrench.py   (E219_SMOKE=1: fact+ctrl
      batteries only, own smoke dir, nothing adjudicated)
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

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402

import common                                            # noqa: E402
from common import now_iso, run_dir, save_json            # noqa: E402

import e182c_forgetting_control as e1                    # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                             # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                 # the owner envelope (this dispatch: <= 4)

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import textwrap                                          # noqa: E402

SMOKE = os.environ.get("E219_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e219_smoke" if SMOKE else "e219"

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
SMOKE_BATTERIES: tuple[str, ...] = ("fact", "ctrl")   # the smoke subset
FAMILY6_ORDER = ["lang", "cap-cur", "founder-anchor", "product",
                 "near-uscap", "rev-capital"]
REF_FAMILY = "lang"            # the e216 dummy reference (verbatim)

# ---- registered bar constants (frozen) ----------------------------------------
CORR_BAR = 0.40         # ENTRENCHMENT-REAL: "|rho| >= 0.4" (max over washes)
NO_REL_BAR = 0.20       # INSTRUMENT-TEXTURE: "|rho| < 0.2"
ABSORB_BAR = 0.50       # both bars' "rho < 0.5" (read as |rho| < 0.5, e218)

# ---- registered verification tolerances (frozen) -------------------------------
TOL_RECOMPUTE_DP = 1e-9     # desk recompute vs e215 assignment (e216/e218)
TOL_BASELINE_DP = 1e-9      # the e216 OLS residual reproduction
TOL_T0_DP = 0.010           # per-probe t=0 dp vs e214's committed journal

# ---- the hand-registered paraphrase registry (frozen BEFORE any measurement) ----
# Template forms: (sentence, query) filled with the SAME rotating
# first-two-others exemplar rule as the original battery (K_SHOT=2).
# 'ans' says which pair member the form's answer is ("a" = the committed
# answer for fact/near pairs; "c" = the reversed cue<->answer swap).
# For tmpl pairs (state, capital) the COMMITTED answer is the state (= c):
# T1 answers the capital (direction-flipped), T2/T3 answer the state.
TMPL_FORMS: dict[str, list[dict]] = {
    "lang": [
        {"id": "L1 lang-passive",
         "tmpl": ("The language spoken in {c} is {a}. ",
                  "The language spoken in {c} is"), "ans": "a",
         "dir": "forward"},
        {"id": "L2 in-country",
         "tmpl": ("In {c}, people speak {a}. ", "In {c}, people speak"),
         "ans": "a", "dir": "forward"},
        {"id": "L3 reversed",
         "tmpl": ("{a} is spoken in {c}. ", "{a} is spoken in"),
         "ans": "c", "dir": "reversed"},
    ],
    "cap": [
        {"id": "C1 capital-city",
         "tmpl": ("The capital city of {c} is {a}. ",
                  "The capital city of {c} is"), "ans": "a",
         "dir": "forward"},
        {"id": "C2 possessive",
         "tmpl": ("{c}'s capital is {a}. ", "{c}'s capital is"),
         "ans": "a", "dir": "forward"},
        {"id": "C3 reversed",
         "tmpl": ("{a} is the capital of {c}. ", "{a} is the capital of"),
         "ans": "c", "dir": "reversed"},
    ],
    "cur": [
        {"id": "U1 official-currency",
         "tmpl": ("The official currency of {c} is the {a}. ",
                  "The official currency of {c} is the"), "ans": "a",
         "dir": "forward"},
        {"id": "U2 money",
         "tmpl": ("Money in {c} is the {a}. ", "Money in {c} is the"),
         "ans": "a", "dir": "forward"},
        {"id": "U3 reversed",
         "tmpl": ("The {a} is the currency of {c}. ",
                  "The {a} is the currency of"), "ans": "c",
         "dir": "reversed"},
    ],
    "found": [
        {"id": "F1 bare-is",
         "tmpl": ("{c} is {a}. ", "{c} is"), "ans": "a",
         "dir": "forward"},
        # F2 hand-registered per item below (anchor-first constructions)
        {"id": "F2 anchor-first", "hand": "FOUND_F2", "ans": "a",
         "dir": "forward"},
    ],
    "make": [
        {"id": "M1 is-the",
         "tmpl": ("{c} is the {a}. ", "{c} is the"), "ans": "a",
         "dir": "forward"},
        {"id": "M2 maker-first", "hand": "MAKE_M2", "ans": "a",
         "dir": "forward"},
        # M3 reversed: answer = the MAKER (cue<->answer swap)
        {"id": "M3 made-by-rev", "hand": "MAKE_M3", "ans": "map",
         "dir": "reversed"},
    ],
    "near": [
        {"id": "N1 us-state",
         "tmpl": ("The capital of the U.S. state of {c} is {a}. ",
                  "The capital of the U.S. state of {c} is"), "ans": "a",
         "dir": "forward"},
        {"id": "N2 reversed",
         "tmpl": ("{a} is the capital of the U.S. state of {c}. ",
                  "{a} is the capital of the U.S. state of"),
         "ans": "c", "dir": "reversed"},
    ],
    "tmpl": [
        {"id": "T1 forward",
         "tmpl": ("The capital of {c} is {a}. ", "The capital of {c} is"),
         "ans": "a", "dir": "forward-flipped"},
        {"id": "T2 capital-of",
         "tmpl": ("{a} is the capital of {c}. ", "{a} is the capital of"),
         "ans": "c", "dir": "same-as-committed"},
        {"id": "T3 us-rev",
         "tmpl": ("The U.S. state whose capital is {a} is {c}. ",
                  "The U.S. state whose capital is {a} is"), "ans": "c",
         "dir": "same-as-committed"},
    ],
}

# The hand-registered per-item forms (subject -> the natural completion
# construction its descriptor needs; sentences serve BOTH as exemplar
# fills for other probes and (minus the answer) as the query).
HAND_FORMS: dict[str, dict[tuple[str, str], tuple[str, str, str]]] = {
    # (sentence, query, answer)
    "FOUND_F2": {
        ("The software company founded by Bill Gates", "Microsoft"):
            ("Bill Gates founded the software company Microsoft. ",
             "Bill Gates founded the software company", "Microsoft"),
        ("The social network founded by Mark Zuckerberg", "Facebook"):
            ("Mark Zuckerberg founded the social network Facebook. ",
             "Mark Zuckerberg founded the social network", "Facebook"),
        ("The electric car company founded by Elon Musk", "Tesla"):
            ("Elon Musk founded the electric car company Tesla. ",
             "Elon Musk founded the electric car company", "Tesla"),
        ("The rocket company founded by Elon Musk", "SpaceX"):
            ("Elon Musk founded the rocket company SpaceX. ",
             "Elon Musk founded the rocket company", "SpaceX"),
        ("The search engine founded by Larry Page", "Google"):
            ("Larry Page founded the search engine Google. ",
             "Larry Page founded the search engine", "Google"),
        ("The social network founded by Jack Dorsey", "Twitter"):
            ("Jack Dorsey founded the social network Twitter. ",
             "Jack Dorsey founded the social network", "Twitter"),
        ("The shoe company founded by Phil Knight", "Nike"):
            ("Phil Knight founded the shoe company Nike. ",
             "Phil Knight founded the shoe company", "Nike"),
        ("The coffee chain founded in Seattle", "Starbucks"):
            ("Seattle is home to the coffee chain Starbucks. ",
             "Seattle is home to the coffee chain", "Starbucks"),
        ("The online encyclopedia that anyone can edit", "Wikipedia"):
            ("Anyone can edit the online encyclopedia Wikipedia. ",
             "Anyone can edit the online encyclopedia", "Wikipedia"),
    },
    "MAKE_M2": {
        ("The gaming console made by Microsoft", "Xbox"):
            ("Microsoft makes the gaming console Xbox. ",
             "Microsoft makes the gaming console", "Xbox"),
        ("The web browser made by Google", "Chrome"):
            ("Google makes the web browser Chrome. ",
             "Google makes the web browser", "Chrome"),
        ("The phone made by Apple", "iPhone"):
            ("Apple makes the phone iPhone. ", "Apple makes the phone",
             "iPhone"),
        ("The tablet made by Apple", "iPad"):
            ("Apple makes the tablet iPad. ", "Apple makes the tablet",
             "iPad"),
        ("The email service made by Google", "Gmail"):
            ("Google makes the email service Gmail. ",
             "Google makes the email service", "Gmail"),
        ("The music store made by Apple", "iTunes"):
            ("Apple makes the music store iTunes. ",
             "Apple makes the music store", "iTunes"),
        ("The game console made by Sony", "PlayStation"):
            ("Sony makes the game console PlayStation. ",
             "Sony makes the game console", "PlayStation"),
        ("The console made by Nintendo", "Wii"):
            ("Nintendo makes the console Wii. ", "Nintendo makes the console",
             "Wii"),
    },
    # M3 answers the MAKER (reversed direction)
    "MAKE_M3": {
        ("The gaming console made by Microsoft", "Xbox"):
            ("The Xbox is made by Microsoft. ", "The Xbox is made by",
             "Microsoft"),
        ("The web browser made by Google", "Chrome"):
            ("Chrome is made by Google. ", "Chrome is made by", "Google"),
        ("The phone made by Apple", "iPhone"):
            ("The iPhone is made by Apple. ", "The iPhone is made by",
             "Apple"),
        ("The tablet made by Apple", "iPad"):
            ("The iPad is made by Apple. ", "The iPad is made by", "Apple"),
        ("The email service made by Google", "Gmail"):
            ("Gmail is made by Google. ", "Gmail is made by", "Google"),
        ("The music store made by Apple", "iTunes"):
            ("iTunes is made by Apple. ", "iTunes is made by", "Apple"),
        ("The game console made by Sony", "PlayStation"):
            ("The PlayStation is made by Sony. ",
             "The PlayStation is made by", "Sony"),
        ("The console made by Nintendo", "Wii"):
            ("The Wii is made by Nintendo. ", "The Wii is made by",
             "Nintendo"),
    },
}

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "ENTRENCHMENT-REAL": "fires if the independent-channel p0 "
            "correlates with the residual (|rho| >= 0.4) or absorbs the "
            "cross-wash residual (rho < 0.5 when substituted as the height "
            "term) — the third dimension is ENTRENCHMENT seen through a "
            "second channel: real probe physics, the confound broken.",
        "INSTRUMENT-TEXTURE": "fires if the independent channel shows no "
            "relation (|rho| < 0.2, no absorption) — the residual's "
            "replication is same-channel texture; the third dimension "
            "retires as an instrument artifact; W021's family grows.",
        "GRADED": "any partial — the tables verbatim.",
    },
    "operationalizations": (
        "residual = e216's committed per-probe residuals (resid_w1/w2), "
        "re-derived through G_RECOMPUTE (p0/hr from e214 curves+journal vs "
        "the e215 assignment at 1e-9) + G_BASELINE (this cell's linear-form "
        "OLS reproduces e216's committed residuals at 1e-9) and then read "
        "from this cell's own recompute; IE(probe) = MEAN over its measured "
        "hand-registered paraphrase forms of p_t0(answer | paraphrase "
        "prompt) on the pristine state (one eval burst; forward+reversed "
        "mixed under the frozen mean, per-direction co-reports); "
        "CORRELATION: Spearman(IE, resid_w) per wash, FIRES iff max |rho| "
        "over washes >= 0.40; ABSORPTION: OLS hr_w ~ 1 + 5 family dummies "
        "(reference lang) + IE centered at its own mean (the e216 full "
        "model with p0 SUBSTITUTED by IE), xwash = Spearman(resid_w1, "
        "resid_w2), ABSORBED iff |xwash| < 0.5 (e218's abs-convention on "
        "the bar's 'rho < 0.5'); NO-RELATION iff max |rho| < 0.20 AND no "
        "absorption; ADJUDICATION: gates ok -> ENTRENCHMENT-REAL iff "
        "(correlation OR absorption); else INSTRUMENT-TEXTURE iff "
        "no-relation; else GRADED; co-reports never adjudicated: rho(IE, "
        "p0), the additive form family+p0+IE, substituted R2 per wash, "
        "IE_forward/IE_reversed rhos, per-family IE medians, the named "
        "splits under the substituted model (e216 yardsticks); gated on "
        "G_RECORDS/G_RECOMPUTE/G_BASELINE/G_CORPUS/G_T0/G_PARA/G_ENV"),
    "registration": ("bars frozen VERBATIM from the dispatch brief (QUEUE "
                     "row DISPATCHED 17:07Z, commit 1db7f74) BEFORE any "
                     "compute; the paraphrase registry above committed with "
                     "the cell BEFORE any p measurement; adjudicate against "
                     "exactly this; no bar shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "DESK + ONE-BURST-EVAL cell: the residual chain enters through the "
    "committed records (e216 re-derived via G_RECOMPUTE + G_BASELINE); the "
    "ONLY model compute is the pristine t=0 burst (battery screening + the "
    "paraphrase battery) — no wash state is loaded.",
    "The batteries' screening re-probe runs at t=0 to reproduce the kept "
    "sets (the frozen gate needs p); the kept probes' t=0 certification "
    "values are read from that same pass (no duplicate probe_battery call "
    "— same net, same prompts, same numbers; the e215 precedent probed "
    "twice and read dp 0.0).",
    "Reversed-direction forms swap cue and answer (the dispatch's own "
    "example, 'X is spoken in'): the form's answer is the pair's other "
    "member, a DIFFERENT token than the committed probe's answer; per-form "
    "direction recorded; IE mixes directions under the frozen mean "
    "(per-direction co-reports included).",
    "Forms whose answer tokenizes to >1 token are DROPPED per-probe with "
    "the reason recorded (trims + the per-probe table); the affected probes "
    "keep their remaining 2 forms (the registered minimum).",
    "The absorption clause reads the bar's '(rho < 0.5)' as |rho| < 0.5 "
    "(a negatively-replicating miss is not absorption either — e218's "
    "convention, inherited); disclosed before compute.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: fact+ctrl batteries only, own smoke dir, nothing "
    "adjudicated or gated (tables descriptive).",
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


def spearman(xs, ys) -> float | None:
    """Spearman rho = Pearson on average ranks (exact under ties)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(avg_ranks(xs), avg_ranks(ys))


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS via lstsq (the e216 convention, VERBATIM)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    dof = n - p
    return {"beta": [float(b) for b in beta],
            "r2": r2, "adj_r2": adj,
            "ss_res": ss_res, "ss_tot": ss_tot,
            "resid_sd": float(ss_res / dof) ** 0.5, "dof": dof,
            "fitted": fitted, "resid": resid}


def design(fams: list[str], height: list[float],
           height_mean: float) -> np.ndarray:
    """[1, 5 family dummies (ref = lang), height centered] (e216 VERBATIM,
    with the height column pluggable — p0 for the baseline, IE for the
    substitution)."""
    cols = [np.ones(len(fams))]
    for f in FAMILY6_ORDER:
        if f == REF_FAMILY:
            continue
        cols.append(np.array([1.0 if g == f else 0.0 for g in fams]))
    cols.append(np.array(height) - height_mean)
    return np.column_stack(cols)


def coef_names(height_label: str) -> list[str]:
    return (["intercept(lang)"]
            + [f"D[{f}]" for f in FAMILY6_ORDER if f != REF_FAMILY]
            + [f"{height_label}_centered"])


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


def sd(v: list[float]) -> float:
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


FAM_COLS = {"lang": "tab:green", "cap-cur": "tab:red",
            "founder-anchor": "tab:blue", "product": "tab:purple",
            "near-uscap": "tab:olive", "rev-capital": "darkorange"}


# ------------------------------------------------------------------ plot

def make_plot(rd, rows, corr, absorb, chan, fam_table, adj):
    """THE FIGURE: (a) IE vs the e216 residual (both washes, the bars);
    (b) the absorption test (xwash under baseline/substituted/additive/
    family-only vs the 0.5 line); (c) IE vs p0 (channel independence);
    (d) the family table + verdict + gates."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))

    # (0,0) IE vs residual
    ax = axes[0, 0]
    ax.axhspan(CORR_BAR, 1.2, facecolor="green", alpha=0.06)
    ax.axhspan(-1.2, -CORR_BAR, facecolor="green", alpha=0.06)
    ax.axhspan(-NO_REL_BAR, NO_REL_BAR, facecolor="red", alpha=0.07)
    for fam in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([r["IE"] for r in fr], [r["resid_w1"] for r in fr],
                   s=26, marker="o", color=FAM_COLS[fam], alpha=0.75,
                   label=f"{fam} (n={len(fr)})")
        ax.scatter([r["IE"] for r in fr], [r["resid_w2"] for r in fr],
                   s=26, marker="s", facecolors="none",
                   edgecolors=FAM_COLS[fam], linewidths=1.3, alpha=0.9)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("INDEPENDENT ENTRENCHMENT  IE = mean paraphrase p at t=0")
    ax.set_ylabel("e216 RESIDUAL  (o w1, [] w2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.4, loc="lower left")
    ax.set_title(f"(1) THE CORRELATION TEST — Spearman(IE, resid): "
                 f"w1 {corr['rho_w1']:+.3f} / w2 {corr['rho_w2']:+.3f} "
                 f"(REAL bar |rho|>={CORR_BAR:g}; TEXTURE bar "
                 f"|rho|<{NO_REL_BAR:g})", fontsize=9.3)

    # (0,1) absorption bars
    ax = axes[0, 1]
    entries = absorb["xwash_table"]
    labels = list(entries.keys())
    vals = [entries[k]["spearman"] for k in labels]
    cols = ["dimgray", "tab:red", "tab:orange", "lightgray"]
    bars = ax.barh(range(len(labels)), vals, color=cols, edgecolor="k",
                   lw=0.6)
    ax.axvline(ABSORB_BAR, color="darkgreen", ls="--", lw=1.6)
    ax.text(ABSORB_BAR + 0.006, len(labels) - 0.35,
            f"absorption line {ABSORB_BAR:g}", fontsize=7.5,
            color="darkgreen", family="monospace")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{k}\n{xwsl(entries[k])}" for k in labels],
                       fontsize=7.0, family="monospace")
    for i, v in enumerate(vals):
        ax.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.6,
                family="monospace")
    ax.set_xlim(0.0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("CROSS-WASH RESIDUAL Spearman (lower = absorbed)")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("(2) THE ABSORPTION TEST — IE substituted as the height "
                 "term (e216 committed 0.934)", fontsize=9.3)

    # (1,0) IE vs p0 — channel independence
    ax = axes[1, 0]
    for fam in FAMILY6_ORDER:
        fr = [r for r in rows if r["family6"] == fam]
        ax.scatter([r["p0"] for r in fr], [r["IE"] for r in fr], s=28,
                   color=FAM_COLS[fam], alpha=0.8)
    ax.set_xlabel("SAME-CHANNEL HEIGHT  p0 (the wash battery's own reading)")
    ax.set_ylabel("IE (the paraphrase channel)")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title(f"(3) CHANNEL INDEPENDENCE — Spearman(IE, p0) = "
                 f"{chan['spearman']:+.3f} / Pearson {chan['pearson']:+.3f} "
                 f"(n={len(rows)})", fontsize=9.3)

    # (1,1) the family table + verdict
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE PARAPHRASE TABLE (per family; IE = mean form p "
            "at t=0):", fontsize=8.8, va="top", family="monospace",
            weight="bold")
    y -= 0.030
    ax.text(0.02, y, "  family             n  forms  IE_med  IE_fwd  IE_rev "
                     "  p0_med", fontsize=7.0, va="top",
            family="monospace")
    y -= 0.021
    for fam in FAMILY6_ORDER:
        s = fam_table[fam]
        if s["n"] == 0:
            continue
        ax.text(0.02, y,
                f"  {fam:17s} {s['n']:3d}  {s['n_forms']:3d}   "
                f"{s['median_IE']:6.3f}  {s['median_IE_fwd']:6.3f}  "
                f"{s['median_IE_rev']:6.3f}  {s['median_p0']:6.3f}",
                fontsize=7.0, va="top", family="monospace",
                color=FAM_COLS[fam])
        y -= 0.0195
    y -= 0.008
    for line in (
        f"CORRELATION: rho(IE, resid) w1 {corr['rho_w1']:+.3f} / w2 "
        f"{corr['rho_w2']:+.3f} -> max |rho| {corr['max_abs_rho']:.3f} "
        f"(REAL >= {CORR_BAR:g}; TEXTURE < {NO_REL_BAR:g})",
        f"ABSORPTION: family+IE xwash {absorb['substituted']['spearman']:.3f} "
        f"(absorbed iff |rho| < {ABSORB_BAR:g}; R2 "
        f"{absorb['substituted']['r2']['w1']:.3f}/"
        f"{absorb['substituted']['r2']['w2']:.3f})",
        f"           family+p0+IE xwash {absorb['additive']['spearman']:.3f} "
        f"(additive co-report; e216 baseline "
       	f"{absorb['xwash_table']['family+p0 (e216 baseline)']['spearman']:.3f})",
        f"CHANNEL: rho(IE, p0) {chan['spearman']:+.3f}",
    ):
        ax.text(0.02, y, line, fontsize=7.0, va="top", family="monospace")
        y -= 0.020
    y -= 0.006
    ax.text(0.02, y, f"E219 VERDICT: {adj['bars']['verdict']}", fontsize=9.6,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.032
    for wd in textwrap.wrap(adj["bars"]["clause"], width=94,
                            break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        y -= 0.0195
    y -= 0.004
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in adj["gates_summary"].items()), fontsize=7.0, va="top",
        family="monospace")

    fig.suptitle("E219 — THE INDEPENDENT-ENTRENCHMENT CELL (T191's hunt, "
                 "first target): the paraphrase channel vs the third "
                 f"dimension -> {adj['bars']['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = rd / "entrenchment_cell.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def xwsl(entry: dict) -> str:
    """A one-line xwash annotation for the bar chart labels."""
    r2 = entry.get("r2") or {}
    bits = [f"R2 {r2['w1']:.2f}/{r2['w2']:.2f}"] if "w1" in r2 else []
    if entry.get("absorbed") is not None:
        bits.append("ABSORBED" if entry["absorbed"] else "not absorbed")
    return "  ".join(bits) if bits else ""


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E219 — THE INDEPENDENT-ENTRENCHMENT CELL (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e219_independent_entrenchment",
        "phase": "desk + one pristine-burst eval (the paraphrase channel; "
                 "no wash state loaded)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("T191's named hunt, first target: does the third "
                     "sorting dimension correlate with entrenchment "
                     "measured through a DIFFERENT channel (the "
                     "hand-registered paraphrase battery at t=0) — and "
                     "does the independent entrenchment ABSORB the "
                     "cross-wash residual where every same-channel height "
                     "form failed? (the pipeline-texture confound, broken "
                     "or confirmed)"),
        "builds_on": [
            "T191 / e218 (the named hunt + the pipeline-texture confound; "
            "the BEYOND-HEIGHT verdict this cell interrogates)",
            "T190 / e216 (the residual this cell hunts; the committed "
            "54-probe table + the OLS baseline reproduced via G_BASELINE)",
            "T189 / e215 (the family x height model + the committed "
            "54-probe records)",
            "T187 / e214 (the archive + the committed curves/journal)",
            "T183 / e182c2 + T149 / e182c + T123 / e182 (the two-wash "
            "124M archive)",
            "W028 (the shape/height law)",
        ],
        "whats_new": [
            "the HAND-REGISTERED PARAPHRASE BATTERY: 2-3 independent "
            "prompt forms per probe (paraphrased cloze + reversed "
            "cue<->answer + per-item hand forms), every string committed "
            "with the cell before any measurement",
            "the INDEPENDENT-ENTRENCHMENT table IE = mean paraphrase p at "
            "t=0 (the pristine state, one eval burst)",
            "the CORRELATION TEST: Spearman(IE, e216's committed residual) "
            "per wash against the frozen 0.4/0.2 bars",
            "the SUBSTITUTION absorption test: family + IE replacing p0 as "
            "the height term — the decisive split against the 0.934 "
            "cross-wash residual",
            "the channel-independence co-report rho(IE, p0) + the additive "
            "form family+p0+IE",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P1 the records
    e216p = common.REPO / "runs" / "e216" / "metrics.json"
    e215p = common.REPO / "runs" / "e215" / "metrics.json"
    e214p = common.REPO / "runs" / "e214" / "metrics.json"
    e214j = common.REPO / "runs" / "e214" / "journal.json"
    for p in (e216p, e215p, e214p, e214j, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e216m = json.loads(e216p.read_text(encoding="utf-8"))
    e215m = json.loads(e215p.read_text(encoding="utf-8"))
    e214m = json.loads(e214p.read_text(encoding="utf-8"))
    e214j = json.loads(e214j.read_text(encoding="utf-8"))["states"]
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    assign = e215m["typology"]["assignment"]
    e216_res = {(r["battery"], r["fact"]): r
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
                "<= 3.3e-06; e216 added the corpus/prompt rebuild "
                "certifications + its committed residual table); this cell "
                "adds the desk recompute (G_RECOMPUTE), the baseline "
                "reproduction (G_BASELINE), git provenance, and its own "
                "t=0 pristine re-certification (G_T0)",
    }
    G_RECORDS["pass"] = bool(
        e215m["status"] == "DONE" and e216m["status"] == "DONE"
        and len(assign) == 54 and len(e216_res) == 54
        and all(G_RECORDS["e215_gates"].values())
        and all(G_RECORDS["e216_gates"].values())
        and not prov["dirty_records"])
    metrics["gates"] = {"G_RECORDS": G_RECORDS}
    log(f"G_RECORDS: {'PASS' if G_RECORDS['pass'] else 'FAIL'} "
        f"(e215 {e215m['status']}, e216 {e216m['status']}, "
        f"{len(e216_res)} committed residual rows, HEAD "
        f"{prov['head'][:9]})")
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
        if SMOKE and b not in SMOKE_BATTERIES:
            continue
        pos_in_battery[b] = pos_in_battery.get(b, -1) + 1
        p0_rc = t0j[b]["probes"][r["fact"]]["p"]
        p80w1_rc = cur_w1[DEEPEST][b]["probes"][r["fact"]]
        p80w2_rc = cur_w2[DEEPEST][b]["probes"][r["fact"]]
        for a, bb in ((p0_rc, r["p0"]), (p80w1_rc, r["p80_w1"]),
                      (p80w2_rc, r["p80_w2"]),
                      (p80w1_rc / p0_rc, r["hr_w1"]),
                      (p80w2_rc / p0_rc, r["hr_w2"])):
            max_dp = max(max_dp, abs(a - bb))
        row = dict(r)
        row["battery_pos"] = pos_in_battery[b]
        row["p0"] = p0_rc
        row["hr_w1"] = p80w1_rc / p0_rc
        row["hr_w2"] = p80w2_rc / p0_rc
        rows.append(row)
    G_RECOMPUTE = {
        "recomputed_from": f"e214 curves w1/w2 +{DEEPEST} and journal t=0, "
                           "matched per probe by fact+battery (the "
                           "e216/e218 recompute, verbatim)",
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

    # --------------------------------- P3 the e216 baseline reproduction (G_BASELINE)
    fams = [r["family6"] for r in rows]
    p0v = [r["p0"] for r in rows]
    p0_mean = float(np.mean(p0v))
    Xb = design(fams, p0v, p0_mean)
    base_dp = 0.0
    base_fit = {}
    for w in ("w1", "w2"):
        y = np.array([r[f"hr_{w}"] for r in rows])
        fit = ols(Xb, y)
        base_fit[w] = fit
        for i, r in enumerate(rows):
            committed = e216_res[(r["battery"], r["fact"])][f"resid_{w}"]
            base_dp = max(base_dp, abs(float(fit["resid"][i]) - committed))
            r[f"resid_{w}"] = float(fit["resid"][i])
    G_BASELINE = {
        "compared": "this cell's linear form (family dummies + p0 "
                    "centered) vs e216's committed per-probe residuals, "
                    "both washes",
        "max_dp": base_dp, "tol": TOL_BASELINE_DP,
        "xw_spearman_this_cell": spearman([r["resid_w1"] for r in rows],
                                          [r["resid_w2"] for r in rows]),
        "xw_spearman_e216": e216m["cross_wash_residual"]["spearman"],
        "p0_mean": p0_mean,
        "note": "the e216 baseline REPRODUCED (extend, don't repeat): the "
                "independent channel is read only after this bit-check "
                "passes",
    }
    G_BASELINE["pass"] = bool(base_dp <= TOL_BASELINE_DP) if not SMOKE \
        else True
    if SMOKE:
        G_BASELINE["smoke_note"] = ("smoke fits the 32-probe subset, not "
                                    "the 54 — subset residuals differ from "
                                    "the committed full-fit by construction; "
                                    "the gate is deferred to the full run")
    metrics["gates"]["G_BASELINE"] = G_BASELINE
    log(f"G_BASELINE: {'PASS' if G_BASELINE['pass'] else 'FAIL'} "
        f"(residual dp {base_dp:.2e}; xwash "
        f"{G_BASELINE['xw_spearman_this_cell']:.6f} vs committed "
        f"{G_BASELINE['xw_spearman_e216']:.6f})")
    write_metrics("PARTIAL: e216 baseline reproduced")
    if not G_BASELINE["pass"] and not SMOKE:
        log("FATAL: baseline reproduction failed — aborting before any "
            "new channel is read")
        return 1

    # ------------------------------------------- P4 the corpus rebuild (G_CORPUS)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(e1.MODEL_REPO,
                                            revision=e1.MODEL_REV)
    metrics["tokenizer"] = {"repo": e1.MODEL_REPO, "revision": e1.MODEL_REV,
                            "note": "the pinned local cache; the "
                                    "instrument this cell shares with the "
                                    "archive (disclosed)"}
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
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
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
        "note": "rebuilt TOKENIZER-ONLY (the e214/e215/e216 chain) — the "
                "substrate the battery builders' contamination scans "
                "need; no wash, no model",
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
        f"({corpus_stats['tokens_after']} tokens; e182 "
        f"{e_corp['tokens_after']})")
    write_metrics("PARTIAL: corpus rebuilt + certified")

    # ---------------------- P5 the organism + batteries VERBATIM + t0 (G_T0)
    load_checks = [cpu_load_check("launch")]
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = org_meta
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"], f"size envelope exceeded: {G_SIZE}"
    metrics["size_gate"] = G_SIZE

    load_checks.append(cpu_load_check("pre_burst"))
    cand, _dropped = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]

    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    if SMOKE:
        nbattery, tbattery = [], []
    else:
        ncand, _nd = e1.build_control_candidates(
            tok, filtered.lower(), train_ids, e_banned,
            {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
        for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
            r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                      "top5": b["top5"]})
        e1.select_battery(ncand)
        nbattery = [r for r in ncand if r["kept"]]

        tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(),
                                              train_ids, e_banned)
        for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
            r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                      "top5": b["top5"]})
        e1.select_battery(tcand)
        tbattery = [r for r in tcand if r["kept"]]

    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}
    if SMOKE:
        bats = {b: v for b, v in bats.items() if b in SMOKE_BATTERIES}
    load_checks.append(cpu_load_check("post_screening"))

    # t=0 certification: kept sets + per-probe p vs e214's journal
    G_T0 = {"tol_per_probe_dp": TOL_T0_DP, "batteries": {}}
    for b, bat in bats.items():
        ref = t0j[b]["probes"]
        kept_eq = bool({r["fact"] for r in bat} == set(ref))
        dp = max(abs(r["p"] - v["p"])
                 for r in bat if r["fact"] in ref for v in [ref[r["fact"]]])
        G_T0["batteries"][b] = {"kept_set_equal": kept_eq, "n": len(ref),
                                "max_dp_vs_e214_journal_t0": dp}
    G_T0["pass"] = bool(all(
        G_T0["batteries"][b]["kept_set_equal"]
        and G_T0["batteries"][b]["max_dp_vs_e214_journal_t0"] <= TOL_T0_DP
        for b in bats)) if not SMOKE else True
    G_T0["note"] = ("the pristine t=0 state re-certified: the pinned "
                    "pretrained 124M is the state e214's journal t0 was "
                    "measured on (kept sets equal, per-probe p matched; "
                    "expected ~0.0 — the e215 precedent read exactly 0.0); "
                    "the screening pass doubles as the kept probes' t0 "
                    "witness (same net, same prompts — the deviation is "
                    "disclosed)")
    metrics["gates"]["G_T0"] = G_T0
    log(f"G_T0: {'PASS' if G_T0['pass'] else 'FAIL'} "
        + " | ".join(f"{b}: kept_eq "
                     f"{G_T0['batteries'][b]['kept_set_equal']}, dp "
                     f"{G_T0['batteries'][b]['max_dp_vs_e214_journal_t0']:.2e}"
                     for b in bats))
    write_metrics("PARTIAL: pristine t=0 re-certified (batteries verbatim)")

    # ---------------------------- P6 the paraphrase battery (G_PARA) + the burst
    pool_of = {}
    for rel, pool in e1.POOLS.items():
        for i, (c, a) in enumerate(pool):
            pool_of[("fact", rel, c, a)] = (pool, i, c, a)
    for rel, pool in e1.CTRL_POOLS.items():
        for i, (c, a) in enumerate(pool):
            pool_of[("ctrl", rel, c, a)] = (pool, i, c, a)
    for i, (c, a) in enumerate(e1.NEAR_POOL):
        pool_of[("near", "near", c, a)] = (e1.NEAR_POOL, i, c, a)
    for i, (c, a) in enumerate(e2.TMPL_POOL):
        pool_of[("tmpl", "tmpl", c, a)] = (e2.TMPL_POOL, i, c, a)

    para_recs = []
    para_by_probe: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        b, rel = r["battery"], r["relation"]
        key = (b, rel, r["subject"], r["answer"])
        if key not in pool_of:
            log(f"FATAL: probe {r['fact']} not found in its frozen pool")
            return 1
        pool, i, c0, a0 = pool_of[key]
        ex = [pool[j] for j in range(len(pool)) if j != i][:e1.K_SHOT]
        if rel not in TMPL_FORMS:
            log(f"FATAL: relation '{rel}' (probe {r['fact']}) has no "
                f"registered paraphrase forms")
            return 1
        forms = []
        for spec in TMPL_FORMS[rel]:
            if "tmpl" in spec:
                sent, query = spec["tmpl"]
                prefix = "".join(sent.format(c=ec, a=ea) for ec, ea in ex)
                prompt = prefix + query.format(c=c0, a=a0)
                ans_str = a0 if spec["ans"] == "a" else c0
            else:
                hmap = HAND_FORMS[spec["hand"]]
                if (r["subject"], r["answer"]) not in hmap:
                    continue
                prefix = "".join(hmap[(ec, ea)][0] for ec, ea in ex
                                 if (ec, ea) in hmap)
                prompt = prefix + hmap[(r["subject"], r["answer"])][1]
                ans_str = hmap[(r["subject"], r["answer"])][2]
            a_ids = tok.encode(" " + ans_str)
            if len(a_ids) != 1:
                trims.append(f"{r['fact']} form {spec['id']}: answer "
                             f"'{ans_str}' tokenizes to {len(a_ids)} "
                             f"tokens — form dropped (first-token "
                             f"measurement would not be exact)")
                continue
            ukey = f"{r['fact']}#{spec['id']}"
            rec = {"fact": ukey, "battery": b, "relation": rel,
                   "form": spec["id"],
                   "dir": spec["dir"], "prompt": prompt, "answer": ans_str,
                   "ans_id": a_ids[0],
                   "ids": torch.tensor([tok.encode(prompt)],
                                       dtype=torch.long),
                   "key": ukey}
            para_recs.append(rec)
            forms.append({"form": spec["id"], "dir": spec["dir"],
                          "prompt": prompt, "answer": ans_str,
                          "ans_id": a_ids[0], "key": rec["key"]})
        para_by_probe[(b, r["fact"])] = forms

    load_checks.append(cpu_load_check("pre_paraphrase_burst"))
    run = e1.probe_battery(net0, para_recs)
    pmap = {row["fact"]: row for row in run["probes"]}
    load_checks.append(cpu_load_check("post_paraphrase_burst"))
    del net0                                     # the only model load, done

    para_table = []
    for r in rows:
        forms = para_by_probe[(r["battery"], r["fact"])]
        measured = []
        for f in forms:
            pr = pmap[f["key"]]
            f["p"] = pr["p"]
            f["rank"] = pr["rank"]
            measured.append(f)
        ie = sum(f["p"] for f in measured) / len(measured)
        fwd = [f["p"] for f in measured if f["dir"].startswith("forward")]
        rev = [f["p"] for f in measured if f["dir"] == "reversed"]
        r["IE"] = ie
        r["IE_forms"] = measured
        r["IE_forward"] = (sum(fwd) / len(fwd)) if fwd else None
        r["IE_reversed"] = (sum(rev) / len(rev)) if rev else None
        para_table.append({
            "fact": r["fact"], "battery": r["battery"],
            "family6": r["family6"], "answer_committed": r["answer"],
            "p0": r["p0"], "IE": ie,
            "IE_forward": r["IE_forward"], "IE_reversed": r["IE_reversed"],
            "n_forms": len(measured),
            "forms": [{"form": f["form"], "dir": f["dir"],
                       "answer": f["answer"], "p": f["p"],
                       "rank": f["rank"], "prompt": f["prompt"]}
                      for f in measured],
        })
    n_forms_min = min(t["n_forms"] for t in para_table)
    G_PARA = {
        "registry_committed": True,
        "n_probes": len(para_table),
        "n_forms_total": len(para_recs),
        "min_forms_per_probe": n_forms_min,
        "registered_minimum": 2,
        "forms_dropped": len(trims),
        "exemplar_rule": "first-two-others per frozen pool (e1.K_SHOT=2) — "
                         "the original battery's rotating-exemplar rule, "
                         "applied to the paraphrase sentences",
        "registry": {rel: [
            {"id": s["id"], "dir": s["dir"],
             **({"template": list(s["tmpl"])} if "tmpl" in s else
                {"hand_map": s["hand"]})}
            for s in specs] for rel, specs in TMPL_FORMS.items()},
        "hand_maps": {k: {f"{c}->{a}": list(v)
                          for (c, a), v in m.items()}
                      for k, m in HAND_FORMS.items()},
        "note": "every measured form's full prompt string + answer + p is "
                "committed per probe in metrics.paraphrase.table; the "
                "registry + hand maps above are the frozen source; the "
                "forms were fixed BEFORE any p was measured (the honesty "
                "block discloses the not-blind boundary)",
    }
    G_PARA["pass"] = bool(n_forms_min >= 2 and len(para_recs) > 0)
    metrics["gates"]["G_PARA"] = G_PARA
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "model_loads": 1, "gpu_calls": 0,
             "load_checks": len(load_checks),
             "burst": "battery screening (~4 candidate sets) + ONE "
                      "paraphrase burst on the pristine t=0 state; no wash "
                      "state loaded",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    metrics["paraphrase"] = {
        "definition": "IE(probe) = MEAN p over its measured hand-registered "
                      "paraphrase forms at t=0 (the pristine state); "
                      "forward and reversed directions mixed under the "
                      "frozen mean; per-direction means co-reported",
        "table": para_table,
    }
    log(f"G_PARA: {'PASS' if G_PARA['pass'] else 'FAIL'} "
        f"({len(para_recs)} forms over {len(para_table)} probes; min "
        f"{n_forms_min}/probe; {len(trims)} dropped)")
    ie_all = [r["IE"] for r in rows]
    log(f"IE range [{min(ie_all):.3f}, {max(ie_all):.3f}] mean "
        f"{sum(ie_all) / len(ie_all):.3f}")
    write_metrics("PARTIAL: the paraphrase battery measured")

    # ------------------------------------------------ P7 the statistics
    resid1 = [r["resid_w1"] for r in rows]
    resid2 = [r["resid_w2"] for r in rows]
    ie_v = [r["IE"] for r in rows]
    ie_f = [r["IE_forward"] if r["IE_forward"] is not None else r["IE"]
            for r in rows]
    ie_r = [r["IE_reversed"] if r["IE_reversed"] is not None else r["IE"]
            for r in rows]

    corr = {
        "definition": f"Spearman(IE, resid_w) per wash; FIRES iff max "
                      f"|rho| >= {CORR_BAR} (the e216 max-over-washes "
                      f"convention)",
        "rho_w1": spearman(ie_v, resid1),
        "rho_w2": spearman(ie_v, resid2),
        "max_abs_rho": max(abs(spearman(ie_v, resid1)),
                           abs(spearman(ie_v, resid2))),
        "co_report": {
            "rho_w1_IE_forward": spearman(ie_f, resid1),
            "rho_w2_IE_forward": spearman(ie_f, resid2),
            "rho_w1_IE_reversed": spearman(ie_r, resid1),
            "rho_w2_IE_reversed": spearman(ie_r, resid2),
            "rho_IE_p0": spearman(ie_v, p0v),
            "pearson_IE_p0": _pearson(ie_v, p0v),
            "rho_IE_hr_w1": spearman(ie_v, [r["hr_w1"] for r in rows]),
            "rho_IE_hr_w2": spearman(ie_v, [r["hr_w2"] for r in rows]),
            "note": "per-direction means and raw-hr rhos co-reported, "
                    "never adjudicated; rho(IE, p0) is the channel-"
                    "independence read",
        },
    }
    corr["fires"] = bool(corr["max_abs_rho"] >= CORR_BAR)
    metrics["correlation_test"] = corr
    log(f"CORRELATION: rho(IE,resid) w1 {corr['rho_w1']:+.3f} / w2 "
        f"{corr['rho_w2']:+.3f} -> max |rho| {corr['max_abs_rho']:.3f} "
        f"(fires: {corr['fires']}); rho(IE,p0) "
        f"{corr['co_report']['rho_IE_p0']:+.3f}")
    write_metrics("PARTIAL: correlation test done")

    # the absorption test: family + IE (p0 substituted out)
    ie_mean = float(np.mean(ie_v))
    Xsub = design(fams, ie_v, ie_mean)
    Xadd = np.column_stack([Xb, np.array(ie_v) - ie_mean])
    Xfam = Xb[:, :6]
    sub = {}
    sub_res: dict[str, dict[str, list[float]]] = {"w1": {}, "w2": {}}
    for w in ("w1", "w2"):
        y = np.array([r[f"hr_{w}"] for r in rows])
        fsub = ols(Xsub, y)
        fadd = ols(Xadd, y)
        ffam = ols(Xfam, y)
        sub[w] = {"r2": fsub["r2"], "adj_r2": fsub["adj_r2"],
                  "resid_sd": fsub["resid_sd"], "dof": fsub["dof"],
                  "coefs": {n: [b, float(se)]
                            for n, b, se in zip(
                                coef_names("IE"), fsub["beta"],
                                _coef_se(Xsub, fsub))}}
        sub_res[w]["sub"] = [float(v) for v in fsub["resid"]]
        sub_res[w]["add"] = [float(v) for v in fadd["resid"]]
        sub_res[w]["fam"] = [float(v) for v in ffam["resid"]]
    xw_sub = spearman(sub_res["w1"]["sub"], sub_res["w2"]["sub"])
    xw_add = spearman(sub_res["w1"]["add"], sub_res["w2"]["add"])
    xw_fam = spearman(sub_res["w1"]["fam"], sub_res["w2"]["fam"])
    absorb = {
        "definition": f"the e216 full model with the height term "
                      f"SUBSTITUTED: OLS hr_w ~ 1 + 5 family dummies + IE "
                      f"centered at its own mean; ABSORBED iff |xwash| < "
                      f"{ABSORB_BAR} (the bar's '(rho < 0.5)' with e218's "
                      f"abs-convention)",
        "substituted": {
            "spearman": xw_sub,
            "pearson": _pearson(sub_res["w1"]["sub"], sub_res["w2"]["sub"]),
            "r2": {"w1": sub["w1"]["r2"], "w2": sub["w2"]["r2"]},
            "adj_r2": {"w1": sub["w1"]["adj_r2"], "w2": sub["w2"]["adj_r2"]},
            "coefs_w1": sub["w1"]["coefs"], "coefs_w2": sub["w2"]["coefs"],
            "absorbed": bool(abs(xw_sub) < ABSORB_BAR),
        },
        "xwash_table": {
            "family+p0 (e216 baseline)": {
                "spearman": G_BASELINE["xw_spearman_this_cell"],
                "pearson": e216m["cross_wash_residual"]["pearson"],
                "r2": {"w1": base_fit["w1"]["r2"], "w2": base_fit["w2"]["r2"]},
                "absorbed": bool(abs(G_BASELINE["xw_spearman_this_cell"])
                                 < ABSORB_BAR)},
            "family+IE (SUBSTITUTED)": {
                "spearman": xw_sub,
                "r2": {"w1": sub["w1"]["r2"], "w2": sub["w2"]["r2"]},
                "absorbed": bool(abs(xw_sub) < ABSORB_BAR)},
            "family+p0+IE (additive)": {
                "spearman": xw_add,
                "r2": {}, "absorbed": bool(abs(xw_add) < ABSORB_BAR)},
            "family-only": {
                "spearman": xw_fam, "r2": {}, "absorbed": None},
        },
        "ie_mean": ie_mean,
        "note": "the decisive split: IE substituted where every "
                "same-channel height form (linear/rank/logit/quad/"
                "family-slopes — e218) left rho >= 0.91; the additive and "
                "family-only forms are co-reports",
    }
    metrics["absorption_test"] = absorb
    log(f"ABSORPTION: family+IE xwash {xw_sub:.3f} (absorbed: "
        f"{absorb['substituted']['absorbed']}); R2 "
        f"{sub['w1']['r2']:.3f}/{sub['w2']['r2']:.3f}; additive "
        f"{xw_add:.3f}; family-only {xw_fam:.3f}")

    # per-family IE table (the paraphrase p0 table, family summary)
    fam_table = {}
    for fam in FAMILY6_ORDER:
        frs = [r for r in rows if r["family6"] == fam]
        if not frs:
            continue
        fwd_all = [f["p"] for r in frs for f in r["IE_forms"]
                   if f["dir"].startswith("forward")]
        rev_all = [f["p"] for r in frs for f in r["IE_forms"]
                   if f["dir"] == "reversed"]

        def med(v):
            s = sorted(v)
            return s[len(s) // 2] if len(s) % 2 else (
                (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2.0) if s else None
        fam_table[fam] = {
            "n": len(frs), "n_forms": sum(len(r["IE_forms"])
                                          for r in frs),
            "median_IE": med([r["IE"] for r in frs]),
            "median_IE_fwd": med([r["IE_forward"] for r in frs
                                  if r["IE_forward"] is not None]),
            "median_IE_rev": med([r["IE_reversed"] for r in frs
                                  if r["IE_reversed"] is not None]),
            "median_p0": med([r["p0"] for r in frs]),
            "median_resid_w1": med([r["resid_w1"] for r in frs]),
            "median_resid_w2": med([r["resid_w2"] for r in frs]),
            "median_form_p_forward": med(fwd_all),
            "median_form_p_reversed": med(rev_all),
        }
    metrics["paraphrase"]["family_table"] = fam_table

    # named splits under the substituted model (e216 yardsticks, co-report)
    def _splits():
        if not any(r["family6"] == "product" for r in rows) or not any(
                r["family6"] == "rev-capital" for r in rows):
            return {"note": "families absent in this (smoke) subset — "
                            "splits skipped"}
        prod_res = {r["answer"]: [a, b] for r, a, b in zip(
            rows, sub_res["w1"]["sub"], sub_res["w2"]["sub"])
            if r["family6"] == "product"}
        gp = [((prod_res["Gmail"][0] + prod_res["PlayStation"][0]) / 2
               - prod_res["iPhone"][0]) / sd(sub_res["w1"]["sub"]),
              ((prod_res["Gmail"][1] + prod_res["PlayStation"][1]) / 2
               - prod_res["iPhone"][1]) / sd(sub_res["w2"]["sub"])]
        tmpl_r = [r for r in tmpl_rows]
        out = {
            "definition": "the e216 descriptive yardsticks under the "
                          "SUBSTITUTED model (never adjudicated): product "
                          "contrast mean resid[Gmail,PlayStation]-"
                          "resid[iPhone] in residual SD; tmpl "
                          "SD(resid)/SD(hr)",
            "product_contrast_sd": {"w1": gp[0], "w2": gp[1],
                                    "verdict": "SURVIVES (structured)"
                                    if min(abs(gp[0]), abs(gp[1])) >= 1.0
                                    else "DISSOLVES"},
            "tmpl_width": {
                "ratio_w1": sd([a for r, a in zip(tmpl_r, tmpl_sub1)])
                / sd([r["hr_w1"] for r in tmpl_r]),
                "ratio_w2": sd([b for r, b in zip(tmpl_r, tmpl_sub2)])
                / sd([r["hr_w2"] for r in tmpl_r])},
        }
        out["tmpl_width"]["verdict"] = (
            "DISSOLVES" if max(out["tmpl_width"]["ratio_w1"],
                               out["tmpl_width"]["ratio_w2"]) <= 0.5
            else "SURVIVES (width unabsorbed)")
        return out

    tmpl_rows = [r for r, a, b in zip(rows, sub_res["w1"]["sub"],
                                      sub_res["w2"]["sub"])
                 if r["family6"] == "rev-capital"]
    tmpl_sub1 = [a for r, a, b in zip(rows, sub_res["w1"]["sub"],
                                      sub_res["w2"]["sub"])
                 if r["family6"] == "rev-capital"]
    tmpl_sub2 = [b for r, a, b in zip(rows, sub_res["w1"]["sub"],
                                      sub_res["w2"]["sub"])
                 if r["family6"] == "rev-capital"]
    splits = _splits()
    metrics["named_splits_substituted"] = splits
    write_metrics("PARTIAL: absorption test + tables done")

    # --------------------------------------------- P8 adjudication (frozen)
    gates_ok = bool(G_RECORDS["pass"] and G_RECOMPUTE["pass"]
                    and G_BASELINE["pass"] and G_CORPUS["pass"]
                    and G_T0["pass"] and G_PARA["pass"] and G_ENV["pass"])
    absorbed = bool(absorb["substituted"]["absorbed"])
    no_relation = bool(corr["max_abs_rho"] < NO_REL_BAR and not absorbed)

    if not gates_ok and not SMOKE:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            g for g, v in (("G_RECORDS", G_RECORDS["pass"]),
                           ("G_RECOMPUTE", G_RECOMPUTE["pass"]),
                           ("G_BASELINE", G_BASELINE["pass"]),
                           ("G_CORPUS", G_CORPUS["pass"]),
                           ("G_T0", G_T0["pass"]),
                           ("G_PARA", G_PARA["pass"]),
                           ("G_ENV", G_ENV["pass"])) if not v))
    elif SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke run: pipeline shakedown only (fact+ctrl subset)"
    elif corr["fires"] or absorbed:
        which = []
        if corr["fires"]:
            which.append(f"the independent-channel IE correlates with the "
                         f"residual (max |rho| {corr['max_abs_rho']:.3f} "
                         f">= {CORR_BAR}; w1 {corr['rho_w1']:+.3f} / w2 "
                         f"{corr['rho_w2']:+.3f})")
        if absorbed:
            which.append(f"IE substituted as the height term absorbs the "
                         f"cross-wash residual (rho {xw_sub:.3f} < "
                         f"{ABSORB_BAR}; every same-channel height form "
                         f"left rho >= 0.91 — e218)")
        verdict = "ENTRENCHMENT-REAL"
        clause = ("; ".join(which)
                  + " — the third dimension is ENTRENCHMENT seen through "
                  "a second channel: real probe physics, the confound "
                  "broken (rho(IE, p0) "
                  f"{corr['co_report']['rho_IE_p0']:+.3f}; substituted "
                  f"R2 {sub['w1']['r2']:.3f}/{sub['w2']['r2']:.3f})")
    elif no_relation:
        verdict = "INSTRUMENT-TEXTURE"
        clause = (f"the independent channel shows no relation (max |rho| "
                  f"{corr['max_abs_rho']:.3f} < {NO_REL_BAR}, w1 "
                  f"{corr['rho_w1']:+.3f} / w2 {corr['rho_w2']:+.3f}; no "
                  f"absorption — family+IE xwash {xw_sub:.3f}) — the "
                  f"residual's replication is same-channel texture; the "
                  f"third dimension retires as an instrument artifact; "
                  f"W021's family grows")
    else:
        verdict = "GRADED"
        clause = (f"any partial — max |rho| {corr['max_abs_rho']:.3f} "
                  f"(between the {NO_REL_BAR} texture bar and the "
                  f"{CORR_BAR} real bar; w1 {corr['rho_w1']:+.3f} / w2 "
                  f"{corr['rho_w2']:+.3f}) and no absorption (family+IE "
                  f"xwash {xw_sub:.3f}; additive {xw_add:.3f}; e216 "
                  f"baseline {G_BASELINE['xw_spearman_this_cell']:.3f}); "
                  f"the tables verbatim (the paraphrase table, the "
                  f"correlation co-reports, the absorption ladder)")

    gates_summary = {g: v for g, v in (
        ("G_RECORDS", G_RECORDS["pass"]), ("G_RECOMPUTE", G_RECOMPUTE["pass"]),
        ("G_BASELINE", G_BASELINE["pass"]), ("G_CORPUS", G_CORPUS["pass"]),
        ("G_T0", G_T0["pass"]), ("G_PARA", G_PARA["pass"]),
        ("G_ENV", G_ENV["pass"]))}
    adj = {
        "bars": {"ENTRENCHMENT_REAL": bool((corr["fires"] or absorbed)
                                           and gates_ok and not SMOKE),
                 "INSTRUMENT_TEXTURE": bool(no_relation and gates_ok
                                            and not SMOKE),
                 "verdict": verdict, "clause": clause,
                 "order": "ENTRENCHMENT-REAL -> INSTRUMENT-TEXTURE -> "
                          "GRADED (gated on G_RECORDS/G_RECOMPUTE/"
                          "G_BASELINE/G_CORPUS/G_T0/G_PARA/G_ENV)"},
        "clauses": {
            f"correlation (max |rho| >= {CORR_BAR})": corr["fires"],
            f"absorption (|xwash| < {ABSORB_BAR})": absorbed,
            f"no_relation (max |rho| < {NO_REL_BAR} and no absorption)":
                no_relation,
        },
        "gates_summary": gates_summary,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E219 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  gates: {gates_summary}")

    # ---------------------------------------------------- P9 honesty + close
    metrics["honesty_reflex"] = {
        "hand_registered_not_blind": "the paraphrase forms were chosen by "
            "this cell's author with e216's residual table readable "
            "(Gmail holds, iPhone dies, Topeka high) — NOT blind to the "
            "outcome; the defense is only that the forms were fixed for "
            "naturalness/coverage per family BEFORE any p was measured, "
            "and every string is committed with the cell (the registry + "
            "the per-probe table)",
        "channel_independence_scope": "the channel breaks the NAMED "
            "confound (identical prompt strings + shared p0 denominators "
            "between the two wash reads) — it does NOT break: the shared "
            "organism (the same 124M net, CPU fp32 texture), the shared "
            "answer tokens (forward forms), the k-shot exemplar "
            "structure, the probe pools, or the committed hr outcome "
            "(which still divides by same-channel p0; independence "
            "enters only through the predictor/regressor); rho(IE, p0) = "
            f"{corr['co_report']['rho_IE_p0']:+.3f} co-reports the "
            "residual statistical dependence",
        "reversed_direction_forms": "reversed forms measure the reverse "
            "association — a DIFFERENT trace (the lab's own forward/"
            "backward asymmetry literature); IE mixes directions under "
            "the frozen mean; the per-direction co-reports (IE_forward / "
            "IE_reversed) bound that mixing",
        "form_floor_effects": "paraphrase p can sit near 0 for "
            "form-following failures rather than knowledge absence "
            "(per-family form medians in the family table); IE's range "
            f"[{min(ie_all):.3f}, {max(ie_all):.3f}] against p0's "
            f"[{min(p0v):.3f}, {max(p0v):.3f}]",
        "n_washes": "n=2 washes is texture, not law; wash 1 is a CPU "
            "fp32 replay of e182's GPU original, wash 2 is GPU fp32 — "
            "inherited from the archive (disclosed)",
        "typology_inherited": "the family factor is e215's hand-"
            "registered typology (not blind to the outcome — disclosed "
            "there); every R2 measured against it inherits that "
            "subjectivity wholesale",
        "ols_conventions": "raw bounded hr (no transform), plain OLS, "
            "homoskedastic SEs (co-reported, never adjudicated); the "
            "e216 conventions, inherited",
        "absorption_convention": "the bar's '(rho < 0.5)' read as "
            "|rho| < 0.5 (a negatively-replicating miss is not absorption "
            "either — e218's convention); disclosed before compute",
        "guarantees_nothing": "the paraphrase channel describes THESE 54 "
            "probes under THESE two washes on THIS 124M CPU-fp32 "
            "organism with hand-registered forms — the openness is the "
            "point",
    }
    metrics["compute"] = {
        "envelope": "CPU-only (owner envelope 2026-10-02): threads "
                    f"{torch.get_num_threads()}, load checks "
                    f"{len(load_checks)}, ONE pristine-state eval burst "
                    "(battery screening + ~150 paraphrase forwards), "
                    "decisive; no GPU calls; no wash state loaded",
        "load_checks": load_checks,
        "records_chain": ["runs/e214/metrics.json",
                          "runs/e214/journal.json",
                          "runs/e215/metrics.json",
                          "runs/e216/metrics.json"],
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    if not SMOKE:
        png = make_plot(rd, rows, corr, absorb,
                        {"spearman": corr["co_report"]["rho_IE_p0"],
                         "pearson": corr["co_report"]["pearson_IE_p0"]},
                        fam_table, adj)
        log(f"outputs: {rd / 'metrics.json'}, {png}")
    else:
        png = None
        log(f"outputs: {rd / 'metrics.json'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


def _coef_se(X: np.ndarray, fit: dict) -> list[float]:
    """Classical homoskedastic coefficient SEs (co-report only)."""
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        sigma2 = fit["ss_res"] / fit["dof"]
        return [float(v) for v in np.sqrt(
            np.maximum(np.diag(xtx_inv) * sigma2, 0.0))]
    except np.linalg.LinAlgError:
        return [float("nan")] * X.shape[1]


if __name__ == "__main__":
    sys.exit(main())
