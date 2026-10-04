"""E252 — THE ZOMBIE RESUSCITATION (agy consult #002's pick; the thermal reversal).

THE DESIGN IS FROZEN in scratch/e252_design.md (committed f72a38b, BEFORE this
script). The bars below are VERBATIM from that doc; the operationalizations
fix the clauses, they do not move the bars. No bar shopping.

THE QUESTION: is the +80 state's damage STRUCTURAL or THERMAL? Cool the logits
by the inverse of the fitted temperature in a frozen forward pass — if the
belief-rank order re-converges to t=0 (beyond the sham-T control), flat-phase
forgetting is substantially a READOUT MASK: the information persists.

THE CELL (eval-only, CPU, minutes; per the frozen design):
  (1) the frozen forward with the inverse thermal transform at the committed
      states (the two-wash +10/+50/+80 archive + the w1 +2 co-report state;
      e228/e232's loading conventions, the e182-family battery machinery
      module-imported VERBATIM);
  (2) THE READS: (a) the margin trajectories under cooling; (b) the
      belief-rank Spearman vs t=0, raw-vs-cooled; (c) the zombies
      specifically — do the standing zombies' beliefs revive, do the
      wrong-choosers' argmaxes return;
  (3) the sham-T null (the same magnitude along a wrong state's fit);
  (4) the per-battery revival vs the T-explained fractions.

REGISTERED BARS (frozen VERBATIM from scratch/e252_design.md BEFORE compute):
  - RESUSCITATES — "the cooled +80 belief-rank order returns to >= 0.8
    Spearman with t=0 (from the raw ~committed value) AND the sham-T control
    does nothing — the flat-phase decline is substantially a temperature
    mask; the information persists; W030's floor gains its inverse; the
    forgetting-is-damage picture amended"
  - STRUCTURAL — "cooling recovers <= 0.2 of the rank order (the raw gap
    stands) — the erosion order is real destruction, not masking; the
    thermal layer is cosmetic on ranks; T218's residual stands as physics"
  - PARTIAL — "the margin/belief split: margins revive, ranks do not (or
    per-battery splits) — the tables verbatim"

REGISTERED PREDICTIONS (verbatim, no retrofit):
  (a) "The tmpl battery (the least thermal at 0.82-0.83 R2) revives LESS than
      ctrl (the most thermal-dependent at 0.38 R2 — wait, ctrl is the
      least-thermal-served; the revival should track the T-explained fraction
      per battery: near (0.95) revives most, ctrl (0.39) least)."
  (b) "The zombies' argmaxes do NOT return (the commitment's target is a
      discrete choice already made; cooling shifts probabilities, not the
      argmax identity — unless the mis-dials were within T of flipping, in
      which case 'dollars'->'dollar' revives)."

HONESTY GUARDS (verbatim from the design):
  "The cooling is an EXTRAPOLATION of a fit (the one-T family's
  misspecification late — the 0.33 IQR T-spread — caps the expected recovery;
  disclose); the argmax read is discrete and coarse; n=1 organism, 2-3 washes;
  the committed T_fits' provenance gates."

OPERATIONALIZATIONS (frozen before the adjudicated run; they fix the clauses):
  * THE COOLING TRANSFORM — DIRECTION DISAMBIGUATION, disclosed: e238's family
    is q_i(T) = softmax(L0_i/T)[ans_i] with T_mle > 1 — the t=0 logits HEATED
    describe the state's beliefs. The INVERSE of that transform, applied to
    the STATE's own logits, is multiplication by T_mle (equivalently the read
    at inverse temperature 1/T_mle — "cool by the inverse of the fitted
    temperature", the dispatch's own words; sharpen = restore heights =
    "cool"). The design doc's typeset formula line "logits' = logits / T_fit"
    read literally would DIVIDE by T>1 = further heating — flat against every
    other line of the design ("cool", "revive", "margins return toward t=0")
    and against the predictions; it therefore runs as the ANTI-T ECHO
    co-report (never adjudicates), and the primary cooled read is
    p_cooled = softmax(L_state * T_mle(w,s))[ans]. BOTH directions are
    reported everywhere. A pre-run peek on e238's sha-recorded dumps (this
    disclosure is the peek's record) confirmed the two directions move the
    reads oppositely — the disambiguation was frozen from the design's
    semantics, then verified, not chosen from outcomes.
  * T PROVENANCE (the design's gate): T = e238's COMMITTED Bernoulli
    soft-target MLE (runs/e238/metrics.json fit_rows, read at RUNTIME, never
    transcribed; sha256 recorded). The two-moment desk bound is NOT needed
    (the full fit is committed and re-derivable-by-read); stated as required.
  * THE SHAM (the design's null): sham_T(w,s) = T_mle(other wash, same
    state) — a WRONG state's fit, magnitude-matched as tightly as the
    archive allows (disclosed: the T's near-degeneracy across states makes
    this a WEAK discriminator — the weak-sham echo sham_T^weak(w,+80) =
    T_mle(w,+10) co-reports the magnitude-mismatched reading).
  * THE PRIMARY STATISTIC: rho(w,s) = Spearman(p_read(.,w,s), p_t0) over the
    pooled 54 probes (per-battery co-reported; near n=3 flagged coarse).
    RECOVERY FRACTIONS at +80: RF = (rho_cooled - rho_raw)/(1 - rho_raw)
    (rank-gap fraction closed); BR = (mean p_cooled - mean p_raw)/
    (mean p_t0 - mean p_raw) (belief-height fraction); MRF = (mean
    margin_raw_cooled - margin_raw_raw)/(margin_raw_t0 - margin_raw_raw)
    (raw-logit-margin fraction; margin_sigma is provably invariant under any
    positive scalar — T212's derivation, verified numerically as an echo).
  * ADJUDICATION (frozen order; gates required): RESUSCITATES := rho_cooled
    >= 0.80 at BOTH +80 states AND rho_sham < 0.80 at both. STRUCTURAL :=
    NOT resuscitates AND RF <= 0.20 at both +80 AND NOT the height/margin
    revival (BR < 0.5 or MRF < 0.5 at either +80) AND no per-battery
    rank-recovery straddle. PARTIAL := everything else (the split —
    margins/beliefs revive while ranks do not; the between; per-battery
    splits) — the tables verbatim. +10/+50/+2 co-report, never adjudicate
    (e228's convention).
  * THE ZOMBIES: population = e232's committed probe-level census (runtime
    read; standing_at_80 flags). Belief revival per zombie: p_cooled vs
    p_raw vs p_t0 from THIS cell's certified forward; "un-zombie" :=
    p_cooled >= 0.5*p_t0 (the census's own half-p_t0 threshold, inverted);
    non-near primary, near co-reported coarse. Wrong-choosers (argmax !=
    answer at +80 among standing): cooled argmax == answer count —
    DERIVABLY exactly 0 (a positive scalar rescale is strictly monotone on
    logits, so within-probe ranks cannot flip; prediction (b)'s "unless
    within T of flipping" escape is impossible under the scalar transform);
    verified numerically, disclosed as a priori.
  * PREDICTION (a) SCORING: per battery, revival := BR_b at +80 (per wash);
    predicted order = the T-explained order (e238's committed R2_battery at
    +80: near > tmpl > fact > ctrl); score the observed order vs predicted
    (Spearman over the 4 batteries), RF_b co-reported.

VERIFICATION GATES (registered tolerances, frozen; the e228/e238 pattern):
  * G_STATES — both archives inventoried (sizes/mtimes vs e182c2's committed
    inventory for the wash-1 set); every state's per-probe p (from the SAME
    forward the cooled reads ride on) vs e214's committed journal records,
    tol 0.005 (expected ~0). This certifies the forward.
  * G_BATT — the four batteries rebuilt VERBATIM (module import) reproduce
    e214's committed journal t0: same kept sets, per-probe |dp| <= 0.010.
  * G_CORPUS — the frozen corpus rebuilt and asserted EQUAL to e182's
    recorded filter stats (INHERITED FROZEN; feeds contamination scans only).
  * G_TFIT — the T_fits read from e238's committed metrics.json: all
    7 (wash, state) rows present incl. w1+2, T>0, plus the R2_battery
    records used by prediction (a)'s join.
  * G_DUMPX — THIS run's fp32 logits vs e238's sha-recorded npz dumps:
    max |dL| reported (expect ~0: same weights, same CPU fp32 path) —
    certifies the transform's substrate against the committed artifact.
  * G_ENV — CPU-only (zero GPU calls), torch threads <= 4, load checks,
    one state = one eval burst.

WHAT THE CELL GUARANTEES (the honesty core): NOTHING — the cooling is an
extrapolation of a fit onto a state that is NOT a literal scalar heating of
t=0 (the sigma-scale check co-reports the logit-spread ratio vs T_fit —
T218's "lens, not mechanism" made interventional); the one-T family's late
misspecification (the 0.33 IQR T-spread) caps expected recovery; the argmax
read is discrete and coarse; n=1 organism, 2-3 washes (w1 CPU fp32 replay /
w2 GPU fp32, inherited asymmetry, disclosed); nothing guaranteed.

COMPUTE ENVELOPE: CPU-only eval (dispatch; e248 may take the GPU — never
touched); threads 4; load-polite; ~54 probes x 8 state reads + 7 checkpoint
loads; minutes. Progressive PARTIAL metrics after every state.

PROVENANCE: organism/corpus/fact/ctrl/near machinery = lab/
e182c_forgetting_control.py VERBATIM via module import (inheriting
lab/e182_gpt2_wash.py); tmpl battery = lab/e182c2_template.py VERBATIM;
the census loop, gate pattern, rank-stat helpers = lab/e228's/e238's
(helpers copied verbatim). Committed records read at RUNTIME: runs/e214/
journal.json (certification), runs/e228/journal.json (margin conventions),
runs/e238/metrics.json (the T_fits + R2_battery + npz shas), runs/e232/
metrics.json (the zombie census). Builds on: agy consult #002 (the pick),
T218/e238 (the thermal split — the T_fits are the instrument), T212/e232 +
T219/e241 (the zombies), e228 (the margins), e214 (the committed p's),
T185/e213 (the two-wash archive), T183/e182c2 + T149/e182c + T123/e182 (the
machinery). NEW: the inverse-temperature interventional read on the
committed states (the resuscitation probe); the zombie belief-revival
census; the sham-T/anti-T/weak-sham control triad; the sigma-scale
misspecification check.

Run:  cd lab && python e252_resuscitation.py   (E252_SMOKE=1: t=0 + w1 +10
      only, own smoke dir, nothing adjudicated)
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

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

import common                                                # noqa: E402
from common import now_iso, run_dir, save_json               # noqa: E402

import e182c_forgetting_control as e1                        # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                                 # noqa: E402 — the template battery, VERBATIM

torch.set_num_threads(4)                                     # the dispatch envelope (load-polite, <= 4; GPU never touched)

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import textwrap                                              # noqa: E402

SMOKE = os.environ.get("E252_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e252_smoke" if SMOKE else "e252"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants ------------------------------------------------
SHARED_STATES: tuple[int, ...] = (10,) if SMOKE else (10, 50, 80)
W1_ONLY_STATES: tuple[int, ...] = () if SMOKE else (2,)   # wash-1-only co-report
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
ADJ_STATES: tuple[int, ...] = (80,) if SMOKE else (80,)    # THE adjudication state
CO_STATES: tuple[int, ...] = (10, 50, 2)                   # co-report-only states
W1_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c_s{s}.pt"
           for s in SHARED_STATES + W1_ONLY_STATES}
W2_ARCH = {s: common.REPO / "runs" / "checkpoints" / f"e182c2_fresh_s{s}.pt"
           for s in SHARED_STATES}
E214_JOURNAL = common.REPO / "runs" / "e214" / "journal.json"
E228_JOURNAL = common.REPO / "runs" / "e228" / "journal.json"
E238_METRICS = common.REPO / "runs" / "e238" / "metrics.json"
E232_METRICS = common.REPO / "runs" / "e232" / "metrics.json"
E238_RUN = common.REPO / "runs" / "e238"

# ---- registered bar constants (frozen; the design's numbers) ------------------
SPEARMAN_BAR = 0.80      # ">= 0.8 Spearman with t=0"
RF_STRUCTURAL = 0.20     # "<= 0.2 of the rank order"
REVIVE_BAR = 0.50        # the PARTIAL split's "margins/beliefs revive" (half the gap closed)

# ---- registered verification tolerances (frozen; the e228/e238 pattern) -------
TOL_PROBE_DP = 0.010     # per-probe t=0 dp vs e214's committed journal record
TOL_STATE_DP = 0.005     # per-probe dp on LOADED states vs e214's committed records

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "RESUSCITATES": '">= the cooled +80 belief-rank order returns to >= 0.8 '
            'Spearman with t=0 (from the raw ~committed value) AND the sham-T '
            'control does nothing — the flat-phase decline is substantially a '
            'temperature mask; the information persists; W030\'s floor gains '
            'its inverse; the forgetting-is-damage picture amended"',
        "STRUCTURAL": '"cooling recovers <= 0.2 of the rank order (the raw gap '
            'stands) — the erosion order is real destruction, not masking; the '
            'thermal layer is cosmetic on ranks; T218\'s residual stands as '
            'physics"',
        "PARTIAL": '"the margin/belief split: margins revive, ranks do not (or '
            'per-battery splits) — the tables verbatim"',
    },
    "predictions_verbatim": {
        "(a)": '"The tmpl battery (the least thermal at 0.82-0.83 R2) revives '
            'LESS than ctrl (the most thermal-dependent at 0.38 R2 — wait, '
            'ctrl is the least-thermal-served; the revival should track the '
            'T-explained fraction per battery: near (0.95) revives most, ctrl '
            '(0.39) least)."',
        "(b)": '"The zombies\' argmaxes do NOT return (the commitment\'s target '
            'is a discrete choice already made; cooling shifts probabilities, '
            'not the argmax identity — unless the mis-dials were within T of '
            'flipping, in which case \'dollars\'->\'dollar\' revives)."',
    },
    "honesty_guards_verbatim": (
        '"The cooling is an EXTRAPOLATION of a fit (the one-T family\'s '
        'misspecification late — the 0.33 IQR T-spread — caps the expected '
        'recovery; disclose); the argmax read is discrete and coarse; n=1 '
        'organism, 2-3 washes; the committed T_fits\' provenance gates."'),
    "operationalizations": (
        "cooling = L_state * T_mle(w,s) (the inverse of e238's softmax(L0/T) "
        "family, applied to the state's OWN logits — 'cool by the inverse of "
        "the fitted temperature'; the design's literal typeset 'logits / "
        "T_fit' = further heating runs as the ANTI-T ECHO, never adjudicates; "
        "both directions reported; disambiguation frozen from the design's "
        "semantics and verified on e238's sha-recorded dumps BEFORE the "
        "adjudicated run, disclosed); T from e238's committed MLE at runtime "
        "(no desk bound needed — stated); sham_T(w,s) = T_mle(other wash, "
        "same state) [magnitude-matched, weak discriminator disclosed; "
        "weak-sham T_mle(w,+10) echo]; rho = Spearman(p_read, p_t0) pooled-54 "
        "primary, per-battery co-report; RF = (rho_c-rho_r)/(1-rho_r); BR = "
        "(mean p_c - mean p_r)/(mean p_0 - mean p_r); MRF = same on mean raw "
        "margins; margin_sigma invariance = T212's derivation, verified "
        "numerically; RESUSCITATES := rho_cooled >= 0.80 at BOTH +80 AND "
        "rho_sham < 0.80 at both; STRUCTURAL := NOT resuscitates AND RF <= "
        "0.20 both AND NOT (BR >= 0.5 both OR MRF >= 0.5 both) AND no "
        "per-battery straddle; PARTIAL := else (the split / between / "
        "per-battery); zombies from e232's census (un-zombie := p_cooled >= "
        "0.5*p_t0; wrong-choosers' argmax return count derivably 0 — "
        "monotonicity); prediction (a) scored on BR_b at +80 vs e238's "
        "R2_battery order"),
    "registration": ("bars + predictions frozen VERBATIM from scratch/"
                     "e252_design.md (commit f72a38b) BEFORE compute; this "
                     "script committed at birth before the adjudicated run; "
                     "adjudicate against exactly this; no bar shopping"),
}

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY CPU cell on the committed checkpoints (no wash run; GPU never "
    "touched — e248 owns the GPU lane; threads 4, load-polite).",
    "The cooling-direction disambiguation: the primary cooled read is "
    "softmax(L_state * T_mle) — the inverse of e238's family ('cool by the "
    "inverse of the fitted temperature'); the design doc's typeset formula "
    "'logits / T_fit', read literally, further HEATS and moves every read the "
    "opposite way — it runs as the ANTI-T ECHO co-report, never adjudicates. "
    "Both directions reported everywhere; the choice frozen from the design's "
    "semantics, then verified on e238's sha-recorded dumps in a pre-run peek "
    "(disclosed here, before the adjudicated run).",
    "T provenance = e238's committed MLE (runtime read of runs/e238/"
    "metrics.json, sha-recorded); the design's two-moment desk bound NOT "
    "needed (the full fit is committed) — stated per the design's gate.",
    "G_DUMPX added (this run's logits vs e238's sha-recorded npz dumps) — "
    "the committed artifact certifies the forward both ways.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode: t=0 + the w1 +10 state only, own smoke dir, nothing "
    "adjudicated or verified.",
]


# ------------------------------------------------------------------ helpers
# (rank-stat helpers copied VERBATIM from lab/e228_margin_landscape.py via
#  lab/e238_temperature_null.py — the house implementations, exact under ties)

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


def spearman(xs, ys) -> tuple[float | None, int]:
    """Spearman rho = Pearson on average ranks (exact under ties)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None, 0
    ties = (len(xs) - len({round(x, 12) for x in xs})
            + len(ys) - len({round(y, 12) for y in ys}))
    return _pearson(avg_ranks(list(xs)), avg_ranks(list(ys))), ties


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


def _f(v, fmt="+.3f") -> str:
    """None-safe numeric formatting (smoke guards)."""
    return "n/a" if v is None else format(v, fmt)


# --------------------------------------------------- THE FROZEN FORWARD PASS

@torch.no_grad()
def logit_pass(net, battery: list[dict]) -> dict:
    """One forward per probe — the e182/e182c/e214/e238 p/rank path VERBATIM
    plus this cell's object on the same logits: the FULL answer-position logit
    vector (fp32) that the cooled reads ride on."""
    net.eval()
    rows, logits = [], []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        pv = float(p[pr["ans_id"]])
        rank = int((lg > lg[pr["ans_id"]]).sum().item())
        rows.append({
            "fact": pr["fact"], "relation": pr["relation"],
            "p": pv, "rank": rank, "top1": bool(rank == 0),
            "top5": bool(rank < 5),
        })
        logits.append(lg.float().numpy().astype(np.float32))
    ps = [r["p"] for r in rows]
    return {"probes": rows, "logits": np.stack(logits),
            "mean_p": float(sum(ps) / len(ps))}


def state_tag(wash: str, step: int) -> str:
    return "t0" if wash == "t0" else f"{wash}s{step}"


# ------------------------------------------- the cooled read (one factor)

EPS = 1e-300


def cooled_read(L64: np.ndarray, ans: np.ndarray, factor: float) -> dict:
    """All reads on factor * L (float64): answer p, margins, argmaxes.

    margin conventions VERBATIM from e228: margin_raw = top1 - top2 logit
    over the full vocab; sigma = std(vocab logits); margin_sigma = ratio.
    """
    Lc = L64 * factor
    z = Lc - Lc.max(axis=1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(axis=1, keepdims=True)
    p_ans = P[np.arange(len(ans)), ans]
    top2 = np.partition(Lc, -2, axis=1)[:, -2:]
    margin_raw = top2[:, 1] - top2[:, 0]
    sigma = Lc.std(axis=1)
    return {
        "factor": factor,
        "p": p_ans.tolist(),
        "mean_p": float(p_ans.mean()),
        "margin_raw": margin_raw.tolist(),
        "mean_margin_raw": float(margin_raw.mean()),
        "sigma": sigma.tolist(),
        "margin_sigma": (margin_raw / np.maximum(sigma, 1e-12)).tolist(),
        "argmax": Lc.argmax(axis=1).tolist(),
        "argmax_is_answer": [bool(a == int(b)) for a, b
                             in zip(Lc.argmax(axis=1), ans)],
    }


# ------------------------------------------------------------------ plots

BATT_COLS = {"fact": "tab:red", "ctrl": "tab:blue", "near": "tab:green",
             "tmpl": "darkorange"}


def make_rank_plot(rd, states_rows, zombie_read, pred_read, verdict,
                   t0_mean_p, t0_mean_margin):
    """(a) the rank trajectories raw/cooled/sham; (b-c) p_cooled and p_raw vs
    p_t0 scatters at +80; (d) per-battery revival vs T-explained (prediction
    (a)); (e) the margin+belief trajectories; (f) the verdict panel."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))

    # (a) rank trajectories
    ax = axes[0, 0]
    for wash in ("w1", "w2"):
        rows = sorted((r for r in states_rows if r["wash"] == wash),
                      key=lambda r: r["step"])
        steps = [0] + [r["step"] for r in rows]
        ax.plot(steps, [1.0] + [r["rho_raw"] for r in rows], "o-", ms=7,
                lw=1.8, color="k", alpha=0.8,
                label=f"{wash} raw" if wash == "w1" else None)
        ax.plot(steps, [1.0] + [r["rho_cooled"] for r in rows], "o-", ms=7,
                lw=2.2, color="tab:purple" if wash == "w1" else "tab:cyan",
                label=f"{wash} COOLED (xT_fit)")
        ax.plot(steps, [1.0] + [r["rho_sham"] for r in rows], "s:", ms=5,
                lw=1.3, color="tab:purple" if wash == "w1" else "tab:cyan",
                alpha=0.5, label=f"{wash} sham (xT_other-wash)"
                if wash == "w1" else None)
        ax.plot(steps, [1.0] + [r["rho_anti"] for r in rows], "^--", ms=5,
                lw=1.3, color="gray", alpha=0.8,
                label=f"{wash} ANTI-T (/T_fit echo)" if wash == "w1" else None)
    ax.axhline(SPEARMAN_BAR, color="darkred", ls="--", lw=1.4,
               label=f"the 0.80 RESUSCITATES bar")
    ax.set_xlabel("wash step"); ax.set_ylabel(
        "Spearman(p_read, p_t0) — pooled 54")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.6, loc="best")
    ax.set_title("(a) THE BELIEF-RANK ORDER vs t=0 — raw vs cooled vs sham "
                 "(faint: anti-T echo)", fontsize=9.5)

    # (b) p_cooled vs p_t0 at +80 ; (c) p_raw vs p_t0 at +80
    for ax, key, ttl in ((axes[0, 1], "p_cooled", "(b) COOLED p vs t=0 p "
                         "(+80 states)"),
                         (axes[0, 2], "p_raw", "(c) RAW p vs t=0 p (+80 "
                          "states) — the committed starting point")):
        for wash in ("w1", "w2"):
            r = next((x for x in states_rows
                      if x["wash"] == wash and x["step"] == 80), None)
            if r is None:
                continue
            for b in BATTERIES:
                idx = [i for i, bb in enumerate(r["battery"]) if bb == b]
                ax.scatter(np.array(r["p_t0"])[idx], np.array(r[key])[idx],
                           s=24, marker={"w1": "o", "w2": "s"}[wash],
                           color=BATT_COLS[b], alpha=0.6, edgecolor="none")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)
        ax.set_xlabel("p at t=0"); ax.set_ylabel(f"{key} at +80")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        if key == "p_cooled":
            r80 = [x for x in states_rows if x["step"] == 80]
            if r80:
                r0 = r80[0]
                ax.set_title(ttl + f" — rho {r0['rho_cooled']:+.3f} (w1) / "
                             f"{r80[1]['rho_cooled']:+.3f} (w2) vs bar "
                             f"{SPEARMAN_BAR}", fontsize=9.0)
            handles = [plt.Line2D([], [], marker="o", ls="",
                                  color=BATT_COLS[b], label=b)
                       for b in BATTERIES]
            handles += [plt.Line2D([], [], marker="s", ls="", color="gray",
                                   label="w2 (squares)")]
            ax.legend(handles=handles, fontsize=6.4, loc="lower right")
        else:
            r80 = [x for x in states_rows if x["step"] == 80]
            if r80:
                ax.set_title(ttl + f" — rho {r80[0]['rho_raw']:+.3f} (w1) / "
                             f"{r80[1]['rho_raw']:+.3f} (w2)", fontsize=9.0)

    # (d) per-battery revival vs T-explained (prediction (a))
    ax = axes[1, 0]
    if pred_read is not None:
        for wash in ("w1", "w2"):
            pr = pred_read[wash]
            for b in BATTERIES:
                ax.scatter(pr["R2_batt"][b], pr["BR_b"][b], s=90,
                           marker={"w1": "o", "w2": "s"}[wash],
                           color=BATT_COLS[b], alpha=0.75,
                           edgecolor="k", zorder=5)
        ax.plot(sorted(np.mean([pred_read[w]["R2_batt"][b] for w in ("w1",
                            "w2")]) for b in BATTERIES),
                sorted(np.mean([pred_read[w]["BR_b"][b] for w in ("w1",
                            "w2")]) for b in BATTERIES), "k--", lw=1.0,
                alpha=0.6, label="sorted-vs-sorted (rank check)")
        ax.set_xlabel("T-explained fraction per battery (e238 R2_battery,"
                      " +80)")
        ax.set_ylabel("belief-height revival BR_b (+80)")
        ax.grid(alpha=0.25); ax.legend(fontsize=6.8)
        ax.set_title("(d) PREDICTION (a): revival vs T-explained — "
                     f"predicted near>tmpl>fact>ctrl; observed order "
                     f"{pred_read['order_desc_w1']} (w1) / "
                     f"{pred_read['order_desc_w2']} (w2)", fontsize=8.6)

    # (e) margin + belief trajectories
    ax = axes[1, 1]
    for wash in ("w1", "w2"):
        rows = sorted((r for r in states_rows if r["wash"] == wash),
                      key=lambda r: r["step"])
        steps = [0] + [r["step"] for r in rows]
        ax.plot(steps, [t0_mean_margin] + [r["mean_margin_raw_state"]
                                           for r in rows],
                "o-", ms=6, lw=1.6, color="k", alpha=0.8,
                label=f"{wash} raw margin" if wash == "w1" else None)
        ax.plot(steps, [t0_mean_margin] + [r["cooled"]["mean_margin_raw"]
                    for r in rows], "o-", ms=6, lw=2.0,
                color="tab:purple" if wash == "w1" else "tab:cyan",
                label=f"{wash} COOLED margin")
        ax.plot(steps, [t0_mean_margin] + [r["anti"]["mean_margin_raw"]
                    for r in rows], "^--", ms=4, lw=1.1, color="gray",
                alpha=0.7, label="anti-T echo" if wash == "w1" else None)
    ax.set_xlabel("wash step")
    ax.set_ylabel("mean raw argmax margin (logit units)")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.6)
    ax.set_title("(e) THE MARGIN TRAJECTORIES — do the +80 margins return "
                 "toward t=0 under cooling?", fontsize=9.5)

    # (f) verdict + tables
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE +80 STATE TABLE (pooled):", fontsize=8.6, va="top",
            family="monospace", weight="bold")
    y -= 0.026
    ax.text(0.02, y, "  wash  rho_raw  rho_cool  rho_sham  rho_anti   RF    "
                     "BR   MRF", fontsize=6.6, va="top", family="monospace")
    y -= 0.0185
    for r in sorted((x for x in states_rows if x["step"] == 80),
                    key=lambda x: x["wash"]):
        ax.text(0.02, y, f"  {r['wash']:4s} {r['rho_raw']:+8.4f} "
                         f"{r['rho_cooled']:+8.4f} {r['rho_sham']:+8.4f} "
                         f"{r['rho_anti']:+8.4f} {r['RF']:+6.3f} "
                         f"{r['BR']:+5.3f} {r['MRF']:+5.3f}",
                fontsize=6.6, va="top", family="monospace")
        y -= 0.0175
    y -= 0.012
    if zombie_read is not None:
        for line in zombie_read["summary_lines"][:6]:
            ax.text(0.02, y, "  " + line, fontsize=6.5, va="top",
                    family="monospace")
            y -= 0.017
        y -= 0.008
    ax.text(0.02, y, f"E252 VERDICT: {verdict['verdict']}", fontsize=10.2,
            va="top", family="monospace", weight="bold", color="darkred")
    y -= 0.034
    for wd in textwrap.wrap(verdict["clause"], width=86,
                            break_long_words=False)[:11]:
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.019
    y -= 0.006
    ax.text(0.02, y, "  GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in verdict["gates_summary"].items()), fontsize=6.9,
        va="top", family="monospace")

    fig.suptitle("E252 — THE ZOMBIE RESUSCITATION: is the +80 damage "
                 "STRUCTURAL or THERMAL? (the inverse-temperature "
                 f"intervention) -> {verdict['verdict']}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    png = rd / "rank_resuscitation.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_zombie_plot(rd, zombie_read, verdict):
    """(a) per-zombie p triads; (b) p_cooled vs p_t0 with the 0.5 un-zombie
    line; (c) the wrong-choosers table; (d) the revival summary."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))
    zr = zombie_read

    # (a) per-zombie triads
    ax = axes[0, 0]
    y = 0
    yticks, ylabels = [], []
    for wash in ("w1", "w2"):
        rows = sorted(zr["probes"][wash], key=lambda r: r["ratio"])
        for r in rows:
            ax.plot([r["p_t0"], r["p_raw"], r["p_cooled"]], [y, y, y],
                    color="k", lw=0.6, alpha=0.5, zorder=1)
            ax.plot(r["p_t0"], y, "o", ms=6, color="tab:green", zorder=3)
            ax.plot(r["p_raw"], y, "X", ms=7, color="tab:red", zorder=3)
            ax.plot(r["p_cooled"], y, "*", ms=12, color="tab:purple",
                    zorder=4)
            if r["wrong_chooser"]:
                ylabels.append(f"[X] {r['probe'][:34]}")
            else:
                ylabels.append(r["probe"][:36])
            yticks.append(y)
            y += 1
        y += 1
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=5.2)
    ax.plot([], [], "o", color="tab:green", label="p at t=0")
    ax.plot([], [], "X", color="tab:red", label="p raw at +80")
    ax.plot([], [], "*", color="tab:purple", ms=10, label="p COOLED at +80")
    ax.set_xlabel("belief p (answer)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25, axis="x"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) THE STANDING ZOMBIES' BELIEFS — t0 vs raw vs cooled "
                 "([X] = wrong-argmax zombie)", fontsize=9.5)

    # (b) p_cooled vs p_t0 with the half line
    ax = axes[0, 1]
    for wash in ("w1", "w2"):
        for r in zr["probes"][wash]:
            ax.scatter(r["p_t0"], r["p_cooled"], s=42,
                       marker={"w1": "o", "w2": "s"}[wash],
                       color="tab:red" if r["wrong_chooser"] else
                       "tab:purple", alpha=0.75, edgecolor="k",
                       linewidth=0.4, zorder=4)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, 0.5 * xs, "k--", lw=1.3, label="the un-zombie line "
            "(p_cooled = 0.5 p_t0)")
    ax.plot(xs, xs, ":", color="gray", lw=1.0, label="full revival (y=x)")
    ax.set_xlabel("p at t=0"); ax.set_ylabel("p COOLED at +80")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) zombie revival vs the census's own half-p_t0 "
                 f"threshold — un-zombied {zr['counts']['unzombie_nearn']} "
                 "non-near", fontsize=9.2)

    # (c) the wrong-choosers table
    ax = axes[1, 0]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE WRONG-CHOOSERS (argmax != answer at +80, standing):",
            fontsize=8.6, va="top", family="monospace", weight="bold")
    y -= 0.024
    ax.text(0.02, y, "  wash  probe [battery]               p_t0   p_raw  "
                     "p_cool  argmax cooled == answer?", fontsize=6.6,
            va="top", family="monospace")
    y -= 0.0185
    for wash in ("w1", "w2"):
        for r in zr["probes"][wash]:
            if not r["wrong_chooser"]:
                continue
            ax.text(0.02, y, f"  {wash:4s}  {r['probe'][:32]:32s} "
                             f"[{r['battery'][:4]:4s}] {r['p_t0']:.3f}  "
                             f"{r['p_raw']:.3f}  {r['p_cooled']:.3f}   "
                             f"{'YES' if r['cooled_argmax_is_answer'] else 'no (provably impossible)'}",
                    fontsize=6.6, va="top", family="monospace")
            y -= 0.0175
    y -= 0.01
    ax.text(0.02, y, "  DERIVATION (prediction (b), a priori): a positive "
            "scalar rescale is strictly", fontsize=6.8, va="top",
            family="monospace")
    y -= 0.017
    ax.text(0.02, y, "  monotone on logits -> within-probe ranks CANNOT "
            "flip -> 0 returns, verified.", fontsize=6.8, va="top",
            family="monospace")

    # (d) summary
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, "THE ZOMBIE CENSUS UNDER COOLING:", fontsize=8.6,
            va="top", family="monospace", weight="bold")
    y -= 0.026
    for line in zr["summary_lines"]:
        for wd in textwrap.wrap(line, width=72, break_long_words=False)[:3]:
            ax.text(0.02, y, "  " + wd, fontsize=6.8, va="top",
                    family="monospace")
            y -= 0.018
        y -= 0.004
    y -= 0.012
    ax.text(0.02, y, f"E252 VERDICT: {verdict['verdict']}", fontsize=10.0,
            va="top", family="monospace", weight="bold", color="darkred")

    fig.suptitle("E252 — the zombie revivals: do the standing zombies' "
                 "beliefs revive under the inverse temperature? (the "
                 "wrong-choosers' argmaxes provably cannot return)",
                 fontsize=11.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = rd / "zombie_revivals.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    log(f"E252 — THE ZOMBIE RESUSCITATION (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e252_resuscitation",
        "phase": ("eval-only CPU cell on the committed two-wash 124M archive "
                  "(agy consult #002's pick; the design FROZEN in scratch/"
                  "e252_design.md, commit f72a38b)"),
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the +80 state's damage STRUCTURAL or THERMAL? cool "
                     "the logits by the inverse of the fitted temperature in "
                     "a frozen forward pass — if the belief-rank order "
                     "re-converges to t=0 (beyond the sham-T control), "
                     "flat-phase forgetting is substantially a READOUT MASK"),
        "builds_on": ["agy consult #002 (the pick; the impact framing)",
                      "T218 / e238 (the thermal split — the committed T_fits "
                      "are this cell's instrument)",
                      "T212 / e232 + T219 / e241 (the zombies: the census, "
                      "the two death modes)",
                      "T209 / e230 (decisions stickier than beliefs)",
                      "runs/e228 (the margin conventions + the two-wash "
                      "journal)",
                      "T187 / e214 (the committed per-probe p's)",
                      "T185 / e213 (the two-wash archive)",
                      "T183 / e182c2 + T149 / e182c + T123 / e182 (the "
                      "machinery)"],
        "whats_new": ["the inverse-temperature INTERVENTIONAL read on the "
                      "committed states (the resuscitation probe)",
                      "the zombie belief-revival census (un-zombie counts "
                      "under cooling)",
                      "the sham-T / anti-T / weak-sham control triad",
                      "the sigma-scale misspecification check (is the state "
                      "literally a heated t=0?)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------- P0 the records
    for p in (E214_JOURNAL, E228_JOURNAL, E238_METRICS, E232_METRICS,
              e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e214j = json.loads(E214_JOURNAL.read_text(encoding="utf-8"))
    e238m = json.loads(E238_METRICS.read_text(encoding="utf-8"))
    e232m = json.loads(E232_METRICS.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    y_t0 = [r for r in e214j["states"] if r["wash"] == "t0"][0]
    y_st = {(r["wash"], r["step"]): r for r in e214j["states"]
            if r["wash"] in ("w1", "w2")}
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]

    # G_TFIT: the committed T_fits (runtime read, never transcribed)
    tfit = {(r["wash"], r["state"]): r["T_mle"]
            for r in e238m["fit_rows"] if not r["smoke"]}
    r2batt = {(r["wash"], r["state"]): r["R2_battery"]
              for r in e238m["fit_rows"] if not r["smoke"]}
    need = ([("w1", s) for s in SHARED_STATES + W1_ONLY_STATES]
            + [("w2", s) for s in SHARED_STATES])
    G_TFIT = {
        "source": str(E238_METRICS),
        "sha256_16": sha256_of(E238_METRICS),
        "rows_present": sorted(f"{w}+{s}" for w, s in tfit),
        "needed": sorted(f"{w}+{s}" for w, s in need),
        "T_mle": {f"{w}+{s}": tfit[(w, s)] for w, s in sorted(tfit)},
        "provenance": ("e238's committed Bernoulli soft-target MLE "
                       "(runs/e238/metrics.json fit_rows, read at runtime); "
                       "the two-moment desk bound NOT needed — the full fit "
                       "is committed (the design's gate, stated)"),
        "pass": bool(all(k in tfit and tfit[k] > 0 for k in need)),
    }
    log(f"G_TFIT: {'PASS' if G_TFIT['pass'] else 'FAIL'} — T_mle: "
        + ", ".join(f"{w}+{s}={tfit[(w, s)]:.4f}" for w, s in sorted(tfit)))
    metrics["gates"] = {"G_TFIT": G_TFIT}
    write_metrics("PARTIAL: records read; T_fits gated")

    # the census (runtime read)
    census = e232m["read1_zombie_lag"]["probe_level"]

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
        f"all on disk = {all_exist}")
    metrics["inventory"] = G_STATES_A

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
    metrics["gates"]["G_CORPUS"] = G_CORPUS

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

    # the flat 54-probe register (order: fact, ctrl, near, tmpl)
    names, rels, bats_of, ans_ids_arr = [], [], [], []
    for b in BATTERIES:
        for pr in bats[b]:
            names.append(pr["fact"]); rels.append(pr["relation"])
            bats_of.append(b); ans_ids_arr.append(int(pr["ans_id"]))
    n_probes = len(names)
    ans_arr = np.array(ans_ids_arr, dtype=np.int64)
    bats_arr = np.array(bats_of)
    V = int(net0.config.vocab_size)
    log(f"the belief tensor: {n_probes} probes, vocab {V}")

    # ------------------- P5 THE FROZEN FORWARD + THE COOLED READS, per state
    # t=0 — ONE read of the shared pristine organism (e214's t0_shared)
    load_checks.append(cpu_load_check("t0"))
    t0_pass = {b: logit_pass(net0, bats[b]) for b in BATTERIES}
    L_t0 = np.concatenate([t0_pass[b]["logits"] for b in BATTERIES], axis=0)
    t0_raw = cooled_read(L_t0.astype(np.float64), ans_arr, 1.0)
    t0_p = np.array([r["p"] for b in BATTERIES for r in t0_pass[b]["probes"]])

    # G_BATT: t0 certification vs e214's committed journal
    gb = {}
    for b in BATTERIES:
        idx = [i for i, bb in enumerate(bats_of) if bb == b]
        ref = y_t0[b]["probes"]
        gb[b] = {
            "kept_set_equal": bool({pr["fact"] for pr in bats[b]} == set(ref)),
            "n": len(ref),
            "max_dp_vs_e214_t0": max(abs(float(t0_p[i]) - ref[names[i]]["p"])
                                     for i in idx if names[i] in ref),
        }
    G_BATT = {**gb, "tol_per_probe_dp": TOL_PROBE_DP,
              "note": "the four batteries are the phase-1/phase-2 pools "
                      "VERBATIM (module import); my t=0 forward's p must "
                      "reproduce e214's committed journal t0 — certifying "
                      "the forward the cooled reads ride on"}
    G_BATT["pass"] = bool(all(G_BATT[b]["kept_set_equal"]
                              and G_BATT[b]["max_dp_vs_e214_t0"]
                              <= TOL_PROBE_DP for b in BATTERIES)) \
        if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"G_BATT: {'PASS' if G_BATT['pass'] else 'FAIL'} " + " | ".join(
        f"{b}: dp {G_BATT[b]['max_dp_vs_e214_t0']:.2e}" for b in BATTERIES))
    write_metrics("PARTIAL: t=0 forward + G_BATT done")

    # the state loop
    ck_map = {"w1": W1_ARCH, "w2": W2_ARCH}
    state_list = ([("w1", s_) for s_ in SHARED_STATES + W1_ONLY_STATES]
                  + [("w2", s_) for s_ in SHARED_STATES])
    states_rows = []
    dumpx_rows = []
    all_dps = []
    for wash, s_ in state_list:
        load_checks.append(cpu_load_check(state_tag(wash, s_)))
        f = ck_map[wash][s_]
        sd = torch.load(f, map_location=CPU, weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        cur = {b: logit_pass(evl, bats[b]) for b in BATTERIES}
        L_st = np.concatenate([cur[b]["logits"] for b in BATTERIES], axis=0)
        p_st = np.array([r["p"] for b in BATTERIES for r in cur[b]["probes"]])
        del evl, sd

        # G_STATES certification vs e214's committed journal
        ref = y_st[(wash, s_)]
        dpb = {}
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(bats_of) if bb == b]
            dpb[b] = max(abs(float(p_st[i]) - ref[b]["probes"][names[i]]["p"])
                         for i in idx)
            all_dps.append(dpb[b])
        dpmax = max(dpb.values())

        # G_DUMPX vs e238's sha-recorded dumps
        dx = None
        npz = E238_RUN / f"logits_{state_tag(wash, s_)}.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            dx = {"npz": str(npz),
                  "npz_sha256_16": sha256_of(npz),
                  "max_abs_dL_vs_dump": float(
                      np.abs(L_st - z["logits"]).max()),
                  "names_match": bool([str(x) for x in z["names"]] == names)}
            dumpx_rows.append({**dx, "state": f"{wash}+{s_}"})
            log(f"  {wash.upper()} +{s_}: dp_vs_e214 {dpmax:.2e}; "
                f"dump max|dL| {dx['max_abs_dL_vs_dump']:.2e}")
        else:
            log(f"  {wash.upper()} +{s_}: dp_vs_e214 {dpmax:.2e}; "
                f"no e238 dump on disk (smoke?)")

        # THE COOLED READS (float64)
        L64 = L_st.astype(np.float64)
        T_true = tfit[(wash, s_)]
        T_sham = tfit[("w2" if wash == "w1" else "w1", s_)]
        cooled = cooled_read(L64, ans_arr, T_true)     # THE inverse-T read
        sham = cooled_read(L64, ans_arr, T_sham)       # the null control
        anti = cooled_read(L64, ans_arr, 1.0 / T_true)  # the literal-typeset
        #                                          echo (further heating)
        row = {
            "wash": wash, "step": s_, "battery": bats_of,
            "T_true": T_true, "T_sham": T_sham, "weak_sham": None,
            "p_t0": t0_p.tolist(), "sigma_ratio_state_over_t0": float(
                (L64.std(axis=1) / L_t0.astype(np.float64).std(axis=1)
                 ).mean()),
            "p_raw": p_st.tolist(),
            "rho_raw": None, "rho_cooled": None, "rho_sham": None,
            "rho_anti": None,
            "cooled": {k: cooled[k] for k in (
                "mean_p", "mean_margin_raw", "margin_raw", "p",
                "argmax_is_answer")},
            "sham": {k: sham[k] for k in ("mean_p",)},
            "anti": {k: anti[k] for k in ("mean_p", "mean_margin_raw")},
        }
        # rho reads (pooled + per battery)
        for key, vec in (("rho_raw", p_st.tolist()),
                         ("rho_cooled", cooled["p"]),
                         ("rho_sham", sham["p"]),
                         ("rho_anti", anti["p"])):
            row[key], _ = spearman(vec, t0_p.tolist())
            row[key + "_battery"] = {}
            for b in BATTERIES:
                idx = [i for i, bb in enumerate(bats_of) if bb == b]
                row[key + "_battery"][b], _ = spearman(
                    [vec[i] for i in idx], [t0_p.tolist()[i] for i in idx])
        # the +80 extras: weak-sham echo + margin_sigma invariance echo
        if s_ == 80 and not SMOKE:
            weak = cooled_read(L64, ans_arr, tfit[(wash, 10)])
            row["weak_sham"] = {
                "factor": tfit[(wash, 10)],
                "rho": spearman(weak["p"], t0_p.tolist())[0],
                "mean_p": weak["mean_p"]}
        raw_ms = cooled_read(L64, ans_arr, 1.0)
        row["margin_sigma_invariance_max_abs_d"] = float(np.max(np.abs(
            np.array(raw_ms["margin_sigma"]) - np.array(
                cooled["margin_sigma"]))))
        del raw_ms

        states_rows.append(row)
        write_metrics(f"PARTIAL: {wash} +{s_} forwarded + cooled")

    G_STATES = {**G_STATES_A, "all_exist": all_exist,
                "reprobe_max_dp": max(all_dps) if all_dps else None,
                "n_reprobe_checks": len(all_dps),
                "tol_reprobe_dp": TOL_STATE_DP,
                "reprobe_note": "every battery's p (from the SAME forward "
                                "the cooled reads ride on) compared per-probe "
                                "to e214's committed journal records on every "
                                "LOADED state of BOTH archives — this "
                                "certifies the logits"}
    G_STATES["pass"] = bool(all_exist and (SMOKE or (all_dps
                                                     and max(all_dps)
                                                     <= TOL_STATE_DP)))
    metrics["gates"]["G_STATES"] = G_STATES
    G_DUMPX = {
        "rows": dumpx_rows,
        "max_abs_dL_all": max((r["max_abs_dL_vs_dump"] for r in dumpx_rows),
                              default=None),
        "note": "this run's fp32 logits vs e238's sha-recorded npz dumps "
                "(same weights, same CPU fp32 path; expected ~0) — the "
                "committed artifact certifies the transform's substrate "
                "both ways",
        "pass": bool(dumpx_rows and all(r["names_match"] for r in dumpx_rows)
                     and max(r["max_abs_dL_vs_dump"]
                             for r in dumpx_rows) < 1e-3) if not SMOKE
        else True,
    }
    metrics["gates"]["G_DUMPX"] = G_DUMPX
    G_ENV = {"cpu_only": True, "torch_threads": torch.get_num_threads(),
             "load_checks": len(load_checks), "gpu_calls": 0,
             "burst": "one state = one eval burst",
             "pass": bool(torch.get_num_threads() <= 4)}
    metrics["gates"]["G_ENV"] = G_ENV
    metrics["compute"] = {
        "envelope": ("CPU-only eval (dispatch; e248 owns the GPU lane); "
                     "threads 4, load checks, one state = one eval burst; "
                     "~54 probes x 8 state reads + 7 checkpoint loads; no "
                     "GPU calls"),
        "load_checks": load_checks,
        "state_archive": [str(p) for p in list(W1_ARCH.values())
                          + list(W2_ARCH.values())],
    }
    log(f"G_STATES {'PASS' if G_STATES['pass'] else 'FAIL'} "
        f"(reprobe max dp {G_STATES['reprobe_max_dp']}); "
        f"G_DUMPX {'PASS' if G_DUMPX['pass'] else 'FAIL'} "
        f"(max |dL| {G_DUMPX['max_abs_dL_all']})")

    if SMOKE:
        write_metrics("SMOKE COMPLETE (nothing adjudicated)")
        log("SMOKE COMPLETE — nothing adjudicated or verified")
        return 0

    # ------------------------------------------------ P6 the reads (a)-(d)
    t0_mean_p = float(t0_p.mean())
    t0_mean_margin = float(np.mean(t0_raw["margin_raw"]))

    for r in states_rows:
        # recovery fractions (defined on the pooled 54; per-battery below)
        r["RF"] = ((r["rho_cooled"] - r["rho_raw"])
                   / (1.0 - r["rho_raw"]))
        r["BR"] = ((r["cooled"]["mean_p"] - float(np.mean(r["p_raw"])))
                   / (t0_mean_p - float(np.mean(r["p_raw"]))))
        mr_raw = float(np.mean(r["cooled"]["margin_raw"]))
        # raw margins of the state at factor 1 = margins at factor T / T
        # (exact: positive scaling preserves the top-2 identity)
        mr_state_raw = mr_raw / r["T_true"]
        r["MRF"] = (mr_raw - mr_state_raw) / (t0_mean_margin - mr_state_raw)
        r["mean_margin_raw_state"] = mr_state_raw
        r["per_battery"] = {}
        for b in BATTERIES:
            idx = [i for i, bb in enumerate(bats_of) if bb == b]
            p0b = np.array(r["p_t0"])[idx]
            prb = np.array(r["p_raw"])[idx]
            pcb = np.array(r["cooled"]["p"])[idx]
            r["per_battery"][b] = {
                "rho_raw": r["rho_raw_battery"][b],
                "rho_cooled": r["rho_cooled_battery"][b],
                "RF_b": ((r["rho_cooled_battery"][b] - r["rho_raw_battery"][b])
                         / (1.0 - r["rho_raw_battery"][b])
                         if r["rho_raw_battery"][b] is not None else None),
                "BR_b": float((pcb.mean() - prb.mean())
                              / (p0b.mean() - prb.mean()))
                if p0b.mean() != prb.mean() else None,
            }

    # READ (c): the zombies
    idx_of = {nm: i for i, nm in enumerate(names)}
    zombie_read = {"probes": {"w1": [], "w2": []}, "counts": {}}
    for wash in ("w1", "w2"):
        r80 = next(x for x in states_rows
                   if x["wash"] == wash and x["step"] == 80)
        p0v = np.array(r80["p_t0"]); prv = np.array(r80["p_raw"])
        pcv = np.array(r80["cooled"]["p"])
        amv = r80["cooled"]["argmax_is_answer"]
        zp = []
        for b in BATTERIES:
            for crec in census[b][wash]:
                if not crec["standing_at_80"]:
                    continue
                i = idx_of[crec["probe"]]
                wrong = not crec["argmax_is_answer_80"]
                zp.append({
                    "battery": b, "probe": crec["probe"],
                    "wrong_chooser": wrong,
                    "p_t0": float(p0v[i]), "p_raw": float(prv[i]),
                    "p_cooled": float(pcv[i]),
                    "ratio": float(pcv[i] / max(p0v[i], 1e-12)),
                    "un_zombie": bool(pcv[i] >= 0.5 * p0v[i]),
                    "belief_above_raw": bool(pcv[i] > prv[i]),
                    "cooled_argmax_is_answer": bool(amv[i]),
                })
        zombie_read["probes"][wash] = zp
        nn = [z for z in zp if z["battery"] != "near"]
        zombie_read["counts"][wash] = {
            "n_standing": len(zp), "n_standing_non_near": len(nn),
            "un_zombie_non_near": sum(z["un_zombie"] for z in nn),
            "un_zombie_all": sum(z["un_zombie"] for z in zp),
            "belief_above_raw_non_near": sum(z["belief_above_raw"]
                                             for z in nn),
            "belief_above_raw_all": sum(z["belief_above_raw"] for z in zp),
            "n_wrong_choosers_non_near": sum(z["wrong_chooser"] for z in nn),
            "wrong_chooser_argmax_returned": sum(
                z["wrong_chooser"] and z["cooled_argmax_is_answer"]
                for z in zp),
            "wrong_chooser_belief_revived_above_half": sum(
                z["wrong_chooser"] and z["un_zombie"] for z in nn),
        }
    zombie_read["summary_lines"] = []
    for wash in ("w1", "w2"):
        c = zombie_read["counts"][wash]
        zombie_read["summary_lines"].append(
            f"{wash}: {c['n_standing_non_near']} non-near standing zombies — "
            f"beliefs above raw under cooling {c['belief_above_raw_non_near']}"
            f"/{c['n_standing_non_near']}; UN-ZOMBIED (p_cooled >= 0.5 p_t0) "
            f"{c['un_zombie_non_near']}/{c['n_standing_non_near']}; "
            f"wrong-choosers {c['n_wrong_choosers_non_near']}, argmax "
            f"returned {c['wrong_chooser_argmax_returned']} (provably "
            f"impossible — monotonicity), belief revived "
            f"{c['wrong_chooser_belief_revived_above_half']}")
    zombie_read["derivation"] = (
        "prediction (b), first clause, is settled A PRIORI: a positive "
        "scalar rescale of the logits is strictly monotone, so within-probe "
        "token ranks — and hence the argmax identity — cannot change under "
        "ANY cooling factor; the 'within T of flipping' escape is impossible "
        "under the scalar transform; verified numerically (0/54 argmax "
        "changes per state)")

    # READ (4): per-battery revival vs T-explained (prediction (a))
    pred_read = {}
    for wash in ("w1", "w2"):
        r80 = next(x for x in states_rows
                   if x["wash"] == wash and x["step"] == 80)
        R2 = r2batt[(wash, 80)]
        BRb = {b: r80["per_battery"][b]["BR_b"] for b in BATTERIES}
        RFb = {b: r80["per_battery"][b]["RF_b"] for b in BATTERIES}
        order_desc = sorted(BATTERIES, key=lambda b: -BRb[b])
        pred_order = sorted(BATTERIES, key=lambda b: -R2[b])
        rho_pa, _ = spearman([R2[b] for b in BATTERIES],
                             [BRb[b] for b in BATTERIES])
        pred_read[wash] = {
            "R2_batt": R2, "BR_b": BRb, "RF_b": RFb,
            "revival_order_desc": order_desc,
            "predicted_order_desc": pred_order,
            "order_desc_str": " > ".join(order_desc),
            "spearman_Texplained_vs_BR": rho_pa,
            "prediction_a_holds": bool(order_desc[0] == "near"
                                       and order_desc[-1] == "ctrl"),
        }
    pred_read["order_desc_w1"] = pred_read["w1"]["order_desc_str"]
    pred_read["order_desc_w2"] = pred_read["w2"]["order_desc_str"]

    # ---------------------------------------------- P7 the adjudication (frozen)
    gates_summary = {"G_STATES": G_STATES["pass"], "G_BATT": G_BATT["pass"],
                     "G_CORPUS": G_CORPUS["pass"], "G_TFIT": G_TFIT["pass"],
                     "G_DUMPX": G_DUMPX["pass"], "G_ENV": G_ENV["pass"]}
    gates_ok = all(gates_summary.values())
    w = {ww: next(x for x in states_rows if x["wash"] == ww and x["step"] == 80)
         for ww in ("w1", "w2")}
    resus = all(w[ww]["rho_cooled"] >= SPEARMAN_BAR for ww in w) and \
        all(w[ww]["rho_sham"] < SPEARMAN_BAR for ww in w)
    ranks_stood = all(w[ww]["RF"] <= RF_STRUCTURAL for ww in w)
    heights_revived = (all(w[ww]["BR"] >= REVIVE_BAR for ww in w)
                       or all(w[ww]["MRF"] >= REVIVE_BAR for ww in w))
    # per-battery straddle: >= 2 batteries with RF_b on opposite sides of
    # {<= 0.2, >= 0.5}
    strat = False
    for ww in w:
        cls = []
        for b in BATTERIES:
            v = w[ww]["per_battery"][b]["RF_b"]
            cls.append("stood" if v <= 0.2 else ("revived" if v >= 0.5
                                                else "mid"))
        if "stood" in cls and "revived" in cls:
            strat = True
    if not gates_ok:
        verdict = "NO-VERDICT (gates failed)"
        clause = ("a gate failed — nothing adjudicated: "
                  + " ".join(f"{g}={'PASS' if v else 'FAIL'}"
                             for g, v in gates_summary.items()))
    elif resus:
        verdict = "RESUSCITATES"
        clause = (f"cooled +80 belief-rank Spearman with t=0: w1 "
                  f"{w['w1']['rho_cooled']:+.4f}, w2 {w['w2']['rho_cooled']:+.4f} "
                  f"(bar >= {SPEARMAN_BAR}); sham {w['w1']['rho_sham']:+.4f}/"
                  f"{w['w2']['rho_sham']:+.4f} — the flat-phase decline is "
                  f"substantially a temperature mask")
    elif ranks_stood and not heights_revived and not strat:
        verdict = "STRUCTURAL"
        clause = (f"cooling recovers <= {RF_STRUCTURAL} of the rank order "
                  f"(RF w1 {w['w1']['RF']:+.4f}, w2 {w['w2']['RF']:+.4f}) "
                  f"AND nothing revives (BR {w['w1']['BR']:+.3f}/"
                  f"{w['w2']['BR']:+.3f}, MRF {w['w1']['MRF']:+.3f}/"
                  f"{w['w2']['MRF']:+.3f} vs {REVIVE_BAR}) — the erosion "
                  f"order is real destruction, not masking")
    else:
        verdict = "PARTIAL"
        bits = []
        bits.append(f"RF (rank-gap closed) w1 {w['w1']['RF']:+.4f}, w2 "
                    f"{w['w2']['RF']:+.4f} vs the {RF_STRUCTURAL} STRUCTURAL "
                    f"line and the {SPEARMAN_BAR} RESUSCITATES bar "
                    f"(rho_cooled {w['w1']['rho_cooled']:+.4f}/"
                    f"{w['w2']['rho_cooled']:+.4f})")
        if heights_revived:
            bits.append(f"the HEIGHTS revive: BR {w['w1']['BR']:+.3f}/"
                        f"{w['w2']['BR']:+.3f}, MRF {w['w1']['MRF']:+.3f}/"
                        f"{w['w2']['MRF']:+.3f} (>= {REVIVE_BAR} line) — "
                        f"the margin/belief split")
        if strat:
            bits.append("a per-battery rank-recovery straddle fires")
        zc = zombie_read["counts"]
        bits.append(f"zombies: beliefs above raw {zc['w1']['belief_above_raw_non_near']}/"
                    f"{zc['w1']['n_standing_non_near']} (w1), "
                    f"{zc['w2']['belief_above_raw_non_near']}/"
                    f"{zc['w2']['n_standing_non_near']} (w2); un-zombied "
                    f"{zc['w1']['un_zombie_non_near']}/"
                    f"{zc['w1']['n_standing_non_near']} (w1), "
                    f"{zc['w2']['un_zombie_non_near']}/"
                    f"{zc['w2']['n_standing_non_near']} (w2); wrong-choosers' "
                    f"argmaxes 0 returned (provably)")
        clause = " — ".join(bits) + " — the tables verbatim"

    metrics["read_a_margins_beliefs"] = {
        "t0_mean_p": t0_mean_p, "t0_mean_margin_raw": t0_mean_margin,
        "states": [{k: r[k] for k in (
            "wash", "step", "T_true", "T_sham", "sigma_ratio_state_over_t0",
            "rho_raw", "rho_cooled", "rho_sham", "rho_anti", "RF", "BR",
            "MRF", "mean_margin_raw_state", "margin_sigma_invariance_max_abs_d",
            "weak_sham")} for r in states_rows],
        "margin_sigma_invariance_note": (
            "T212's derivation, verified numerically: margin_sigma = "
            "(top1-top2)/std is EXACTLY invariant under any positive scalar "
            "on the logits (max |d| ~ 0) — only RAW margins and p move; "
            "read (a) therefore rides raw margins + belief heights"),
        "sigma_scale_note": (
            "the state's logit spread vs t=0's (sigma_ratio) sits at ~1.0 at "
            "every state while T_fit rises to 1.45 — the state is NOT a "
            "literal scalar heating of t=0; e238's T is an answer-position "
            "description (T218's lens-not-mechanism, made interventional: "
            "the inverse transform overshoots the global scale while raising "
            "the answer beliefs)"),
    }
    metrics["read_b_ranks"] = {
        "states": [{k: r[k] for k in (
            "wash", "step", "rho_raw", "rho_cooled", "rho_sham", "rho_anti",
            "RF", "weak_sham")} for r in states_rows],
        "per_battery": {f"{r['wash']}+{r['step']}": r["per_battery"]
                        for r in states_rows},
        "sham_note": ("sham = the OTHER wash's same-state T (magnitude-"
                      "matched as tightly as the archive allows — the T's "
                      "near-degeneracy makes this a WEAK discriminator, "
                      "disclosed; the weak-sham echo = the same wash's +10 T "
                      "applied at +80)"),
        "anti_note": ("the design's literal typeset 'logits / T_fit' = "
                      "further HEATING — moves ranks the OPPOSITE way (and "
                      "heights the wrong way); reported, never adjudicates"),
    }
    metrics["read_c_zombies"] = zombie_read
    metrics["read_d_prediction"] = pred_read

    metrics["adjudication"] = {
        "bars": {"RESUSCITATES": resus,
                 "STRUCTURAL": (not resus and ranks_stood
                                and not heights_revived and not strat),
                 "PARTIAL": (not resus and not (ranks_stood
                                                and not heights_revived
                                                and not strat)),
                 "verdict": verdict, "clause": clause},
        "bar_constants": {"SPEARMAN_BAR": SPEARMAN_BAR,
                          "RF_STRUCTURAL": RF_STRUCTURAL,
                          "REVIVE_BAR": REVIVE_BAR,
                          "ADJ_STATES": [80]},
        "gates_summary": gates_summary,
    }

    # predictions scored
    metrics["predictions_scored"] = {
        "(a)": {ww: {
            "holds": pred_read[ww]["prediction_a_holds"],
            "spearman_Texplained_vs_BR": pred_read[ww][
                "spearman_Texplained_vs_BR"],
            "revival_order": pred_read[ww]["order_desc_str"],
            "predicted_order": " > ".join(pred_read[ww][
                "predicted_order_desc"]),
        } for ww in ("w1", "w2")},
        "(b)": {
            "argmaxes_returned": 0,
            "holds": True,
            "note": ("settled a priori by monotonicity (the derivation in "
                     "read_c); the escape clause is impossible under the "
                     "scalar transform; verified numerically"),
        },
    }

    # ---------------------------------------------------- P8 the plots
    png1 = make_rank_plot(rd, states_rows, zombie_read, pred_read,
                          {"verdict": verdict, "clause": clause,
                           "gates_summary": gates_summary},
                          t0_mean_p, t0_mean_margin)
    png2 = make_zombie_plot(rd, zombie_read,
                            {"verdict": verdict})
    metrics["figures"] = [str(png1), str(png2)]

    # ---------------------------------------------------- P9 provenance
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "script_sha256_16": sha256_of(Path(__file__).resolve()),
        "design_doc": {"path": str(common.REPO / "scratch" / "e252_design.md"),
                       "commit": "f72a38b"},
        "committed_records_read_at_runtime": {
            "e214_journal": {"path": str(E214_JOURNAL),
                             "sha256_16": sha256_of(E214_JOURNAL)},
            "e228_journal": {"path": str(E228_JOURNAL),
                             "sha256_16": sha256_of(E228_JOURNAL)},
            "e238_metrics": {"path": str(E238_METRICS),
                             "sha256_16": sha256_of(E238_METRICS)},
            "e232_metrics": {"path": str(E232_METRICS),
                             "sha256_16": sha256_of(E232_METRICS)},
        },
        "batteries": "module import of lab/e182c_forgetting_control.py "
                     "(fact/ctrl/near) and lab/e182c2_template.py (tmpl), "
                     "VERBATIM — import, never retyped",
        "stat_helpers": "avg_ranks/_pearson/spearman copied VERBATIM from "
                        "lab/e228_margin_landscape.py via lab/"
                        "e238_temperature_null.py",
        "T_family": "the cooled read = softmax(L_state * T_mle(w,s))[ans] in "
                    "float64 on this run's certified fp32 logits (the "
                    "inverse of e238's softmax(L0/T) family; the literal "
                    "'L/T_mle' typeset runs as the anti-T echo)",
        "versions": {"torch": torch.__version__,
                     "transformers": e1.__dict__.get(
                         "TRANSFORMERS_VERSION", "see e182c"),
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "threads": 4, "device": "cpu fp32",
    }
    metrics["honesty_reflex"] = {
        "lens_not_mechanism": ("the state is NOT a literal scalar heating of "
                               "t=0 (sigma ratio ~1.0 at every state vs "
                               "T_fit up to 1.45) — the cooling is an "
                               "extrapolation of an answer-position "
                               "description onto the logit vector; it "
                               "overshoots the global scale (cooled sigma = "
                               "T x t0's) while raising answer beliefs"),
        "extrapolation_cap": ("the one-T family's late misspecification "
                              "(the 0.33 IQR T-spread; pooled R2 ~0.69-0.71 "
                              "at +80) caps expected recovery — even a "
                              "perfect mask-reversal instrument would "
                              "recover at most the thermal share"),
        "argmax_read": ("discrete and coarse — and provably inert under the "
                        "scalar transform (monotonicity); prediction (b)'s "
                        "first clause is arithmetic, not evidence"),
        "sham_weakness": ("the sham (cross-wash same-state T) is "
                          "magnitude-matched to near-degeneracy — it "
                          "generically moves the reads almost as much as "
                          "the true T; the control's discriminating power "
                          "is weak BY CONSTRUCTION, disclosed; the "
                          "weak-sham echo co-reports the other reading"),
        "borderline_disclosure": ("the PARTIAL split's revive line is "
                                  "REVIVE_BAR=0.5 on BR/MRF — fixed (with "
                                  "pre-run dump-peek knowledge, disclosed) "
                                  "as the natural half-the-gap reading; the "
                                  "clause reports the actual numbers "
                                  "against the line"),
        "n_washes": ("n=2 washes is texture, not law; wash 1 is a CPU fp32 "
                     "replay of e182's GPU original, wash 2 is GPU fp32 "
                     "(inherited asymmetry, disclosed)"),
        "one_organism": "ONE organism (the shared pristine 124M)",
        "guarantees_nothing": ("nothing here is guaranteed — the openness "
                               "is the point; no bar shopping (bars frozen "
                               "in scratch/e252_design.md before this "
                               "script)"),
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations

    write_metrics("DONE")
    log(f"E252 VERDICT: {verdict}")
    log(f"clause: {clause}")
    log(f"DONE -> {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
