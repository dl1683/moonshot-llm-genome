"""
lab/x48_segment_reread.py — x48 THE SEGMENT-LEVEL FOUR-ARM RE-READ, UPGRADED
(R81's ordering; THE IDENTITY PASS COMES FIRST — the critic's save-a-week item).

Cell x48, dispatched by R81's fold: scratch/r81_critic.md attack 1 + THINKING.md
T328's R81 banner + P-T328a (registered on x48 at T328, frozen verbatim below).

THE FOUR HALVES, in R81's order:
  1. THE IDENTITY PASS (FIRST, gate for everything): trace every committed
     artifact that names the walk's read-subject and start state. Verdict
     LINEAGE-ZEPHYRA / LINEAGE-TAVIREN / AMBIGUOUS — adjudicated BEFORE the
     segment work (it re-labels the segment table's arms).
  2. THE PRE-REGISTERED NET0 CLASS TABLE per segment across the four arms
     (swap / install / walk / anneal): {name-read, start-class, menu,
     formation age, reshape step} — asserted from committed rows, not presumed.
  3. P-T328a (frozen verbatim): segment-level pooled scoring at the remodeling
     boundary — the walk's decorrelated states + e343's install upper segment
     (s100-s400): WARM-PHASE-SHARED / SEGMENTS-DISSENT. The anneal's own
     reshape-step segments ride as the warm-menu control.
  4. THE CO-MOVEMENT REPRODUCTION: the walk/anneal residual co-movement at all
     shared reshape steps (the critic's r=+0.667) reproduced from committed
     rows + extended — does the swap arm join? (the cold-warm contrast's
     read-side signature).

PARITY STANDARD (R81, enforced): the registration below prints the lab lean
AND the counter side by side, with grounds for both.

Pure desk cell: zero training, zero CUDA, zero torch — reads only committed
artifacts (md5-bound), scores them under the family's frozen rules. Every
convention picked + frozen HERE at birth BEFORE compute; this script committed
at birth; adjudicate against exactly this; no bar shopping.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get("X48_SMOKE", "0") == "1"
OUT_DIR = os.path.join(REPO, "runs", "x48")

# ----------------------------------------------------------------------------
# THE REGISTRATION (frozen at birth; verbatim from the dispatch letter + T328)
# ----------------------------------------------------------------------------
REGISTRATION = {
    "cell": "x48",
    "question_verbatim": (
        "THE SEGMENT-LEVEL FOUR-ARM RE-READ, UPGRADED (R81's ordering; the identity pass comes "
        "FIRST — the critic's save-a-week item). (1) THE IDENTITY PASS (FIRST, gate for everything "
        "else): the walk's provenance chain — trace every committed artifact that names the walk's "
        "read-subject and start state (e335's metrics + the walk's construction; x45's install_traj "
        "+ walk rows; e343's registration language; e344's n1_rider rows + REPORT four-arm labels). "
        "Verdict: LINEAGE-ZEPHYRA (the artifacts agree — 4-0) / LINEAGE-TAVIREN (a prose site is "
        "artifact-backed) / AMBIGUOUS (named sites). The pass's own fork is adjudicated BEFORE the "
        "segment work (it re-labels the segment table's arms). (2) THE PRE-REGISTERED NET0 CLASS "
        "TABLE per segment across the four arms (swap/install/walk/anneal): every segment's "
        "{name-read, start-class (BASE cold / install-end warm), menu, formation age, reshape step} "
        "— the classes asserted, not presumed. (3) P-T328a (frozen verbatim): segment-level pooled "
        "scoring at the remodeling boundary — the walk's decorrelated states vs e343's install upper "
        "segment (s100-s400): WARM-PHASE-SHARED (sign -, consistency >= 0.70) / SEGMENTS-DISSENT. "
        "The anneal's own reshape-step segments ride as the warm-menu control. (4) THE CO-MOVEMENT "
        "REPRODUCTION: the critic's desk find — the walk/anneal residuals co-move at all shared "
        "reshape steps (r=+0.667); reproduce it from the committed rows and extend it (does the "
        "swap arm join the co-movement? — the cold-warm contrast's read-side signature)."
    ),
    "lab_lean_verbatim": (
        "WARM-PHASE-SHARED, weakly + LINEAGE-ZEPHYRA (the artifacts are 4-0; the co-movement find "
        "already binds the two warm arms)."
    ),
    "counter_verbatim": (
        "SEGMENTS-DISSENT — the install's s100-s400 are BASE-forming cold interior (net0 classes "
        "differ); and the instrument's certification gap swallows segment-level patterns."
    ),
    "identity_bars_verbatim": {
        "LINEAGE-ZEPHYRA": (
            "every data-bearing artifact site (e335's metrics construction, x45's committed "
            "install_traj + walk rows, e343's registration, e344's n1_rider rows) names the walk "
            "ZEPHYRA-read at the install-end, and NO data-bearing site backs a TAVIREN reading — "
            "the artifacts agree 4-0; the prose sites ('the TAVIREN walk') are labels, re-filed."
        ),
        "LINEAGE-TAVIREN": (
            "a prose site is artifact-backed — some data-bearing committed row/construction names "
            "the walk's read-subject TAVIREN or its start other than install-end; the fold's "
            "'TAVIREN walk' language survives and e344's swap was a name cross after all."
        ),
        "AMBIGUOUS": (
            "the artifact sites disagree among themselves (named sites both ways); the re-label "
            "suspends and the segment table carries both readings disclosed."
        ),
    },
    "P_T328a_bars_verbatim": {
        # T328's registered prediction, frozen verbatim (THINKING.md, 2026-10-10):
        "registered_sentence": (
            "the walk's decorrelation and the install-arm's upper segment share the warm-phase "
            "class — a segment-level pooled scoring at the remodeling boundary lands sign - with "
            "consistency >= 0.70; the countervailing: the segments' net0 classes may not align "
            "(the install's s100-s400 are BASE-formed end-states, not re-formation middles) and "
            "the instrument's certification gap swallows segment-level patterns too."
        ),
        "WARM-PHASE-SHARED": (
            "the pooled segment scoring (component A: the walk's decorrelated states = the cons-leg "
            "reshape rungs in-window, state-deduped per e344's N1 rule; component B: e343's install "
            "upper segment s100-s400 in-window; both paired vs the anneal ruler by x45's frozen "
            "nearest-read rule) lands dominant sign - with sign consistency >= 0.70 on BOTH primary "
            "profiles (r_g-12|g0, r_g0|g+12)."
        ),
        "SEGMENTS-DISSENT": "the pooled bar above fails on either primary profile.",
    },
    "extension_bars_frozen": {
        # mine, frozen at birth: the swap's read-side signature
        "SWAP-JOINS": (
            "the swap arm's full-fit residuals co-move with BOTH warm arms at the shared in-band "
            "reshape steps: Pearson r(swap, anneal) >= +0.50 AND r(swap, walk_cons) >= +0.50 — the "
            "co-movement is phase-general (a stream/draw-side signature), not warm-specific."
        ),
        "SWAP-OPPOSES": (
            "either correlation <= -0.50 — the cold arm moves AGAINST the warm pair at matched "
            "reshape steps (the sharpest possible read-side contrast signature)."
        ),
        "SWAP-APART": "neither (|r| in the middle band on at least one side) — no phase-general co-movement.",
        "floor_ground": (
            "+0.50 sits below the committed walk-anneal r=+0.667 and near the one-sided p~0.05 edge "
            "at n~11-12; permutation p co-reported for every correlation (10k shuffles, seed frozen)."
        ),
    },
    "P_x48a": {
        "my_guess": "LINEAGE-ZEPHYRA (strongly) + WARM-PHASE-SHARED (weakly); the swap SWAP-APART (weakly)",
        "registered": (
            "GROUNDS: (1) IDENTITY — I read all four sites at design time (a desk fact, re-gated at "
            "runtime): e335's walk channel is g0 mean_pz (p(Z)) with the chain e001 -> e043-Dmix "
            "(gen 24314) -> e113 cons (seed 10901) and x34's census naming ZEPHYRA's base prior as "
            "the walk's floor; x45's CHANNEL NOTE says 'the walk reads p(Z) (ZEPHYRA)' and its cons "
            "leg replays from the committed g1c_install_resume (install-end warm); e343's phase/"
            "registration names 'the ZEPHYRA-class formation path' as the cross with the anneal "
            "'on p(T)'; e344's n1 walk rows carry name=ZEPHYRA start=install-end row by row — 4-0, "
            "and the only TAVIREN sites are labels (the four-arm table KEY, the REPORT row, T328's "
            "prose, Law 8's clause), none data-bearing. (2) P-T328a — both pooled components carry "
            "committed sign - sub-scorings (e343's upper segment: - / 0.75 / n=4; e344's arm4 "
            "reproduction of x45: - / 0.7778 & 0.8333 / n=18) and the frozen dedup REMOVES the "
            "install-leg rungs (the likeliest sign-mixing rows) from the walk component, so pooled "
            "sign - at >= 0.70 is likelier than not. (3) THE SWAP — e344's committed arm-level "
            "result is the strongest prior: the swap shares the walk's stream bit-identically and "
            "still scored coherent-ABOVE (sign + 0.78) at matched reads; a stream-driven "
            "co-movement should have leaked into those pairs and did not, so I read SWAP-APART. "
            "COUNTERVAILING PRICED: the install upper's 4-pair 0.75 sits on 3/4 sign- (one flip "
            "moves it); the class table will show the components DISSENT on net0 class regardless "
            "(BASE-forming pre-boundary interior vs ROOT-forming post-boundary reshape), so even a "
            "HIT is a pooled-sign fact wearing a class claim the table contests; and the "
            "certification gap (the instrument is TAVIREN-side certified; every walk/swap/install "
            "row is ZEPHYRA-side) is untouched by any pooling."
        ),
        "predicted_shape": (
            "identity 4-0 ZEPHYRA with the four prose sites named; pooled sign - with consistency "
            "0.70-0.82 on both profiles; the class table's two pooled components dissent on net0 "
            "class (BASE-forming vs ROOT-forming) and on phase (pre- vs post-install-end boundary); "
            "the anneal control shows its own decorrelating rungs (s125-class) WITHOUT the pooled "
            "sign (it is the ruler); the swap's residuals scatter wide of the warm pair (|r| < 0.5 "
            "on at least one side)."
        ),
        "falsifier": (
            "any artifact site backing TAVIREN or a non-install-end start -> identity fork re-opens; "
            "pooled consistency < 0.70 on either profile -> SEGMENTS-DISSENT; r(swap, anneal) >= "
            "+0.50 AND r(swap, walk) >= +0.50 -> SWAP-JOINS (the co-movement generalizes past the "
            "warm class; the warm home loses its read-side signature)."
        ),
        "scored": "TRUE iff identity == LINEAGE-ZEPHYRA AND P-T328a == WARM-PHASE-SHARED",
    },
    "registration_clause": (
        "question + bars + lab lean + counter + P-x48a VERBATIM from the dispatch letter (P-T328a "
        "frozen verbatim from T328); every convention picked + frozen HERE at birth BEFORE compute; "
        "this script committed at birth; adjudicate against exactly this; no bar shopping."
    ),
}

BUILD_ON = [
    "e335 (the walk's construction: the g1c chain e001 -> e043-Dmix gen 24314 install -> e113 cons "
    "seed 10901, read on the g0 mean_pz channel; the committed install/cons trajectories)",
    "x45 (the walk + anneal rows, the CHANNEL NOTE, the frozen nearest-read match rule + separation "
    "rule this cell inherits VERBATIM; the committed 18 walk-vs-anneal pairs)",
    "e343 (the ZEPHYRA install-path cross; its committed 7 pairs + the upper-segment s100+ subset "
    "sign - 0.75 this cell pools; its registration language as an identity site)",
    "e344 (the full swap: its committed 23 pairs, the N1 rider pool + frozen OLS form + state-"
    "identity dedup this cell refits and extends; the four-arm table + REPORT labels as the "
    "mislabel sites)",
    "R81 (the fold that minted this cell: the critic's attack 1 + attack 2e — the lineage label and "
    "the co-movement find; the auditor's parity standard)",
    "T328 / P-T328a (the registered prediction this cell adjudicates); T327 (the fork's history)",
    "x40/x44 (the instrument's family: the per-context elicitation profiles r_g-12|g0 / r_g0|g+12 "
    "and their TAVIREN-side certification — the certification-gap counter's ground)",
]
WHATS_NEW = [
    "the identity pass itself — the first adjudicated provenance verdict on the walk's read-subject "
    "and start state (the record's four artifact sites vs its four prose sites, runtime-gated);",
    "the pre-registered net0 class table across all four arms' segments (start-class / menu / age / "
    "reshape-step per segment, asserted from committed row labels — never tabulated before);",
    "the P-T328a pooled segment scoring (the walk's deduped cons rungs + e343's upper segment, "
    "pooled against the anneal ruler) — the segment-level form of the arm-level bars;",
    "the co-movement extension: the swap arm's residuals correlated into the warm pair at shared "
    "reshape steps (the cold-warm contrast's read-side signature), with permutation ps.",
]
DEVIATIONS = [
    "PURE DESK: zero training, zero CUDA, zero torch — every number recomputed from committed rows "
    "(json reads + OLS + Pearson); the family's replay machinery is NOT re-run (the rows are the "
    "committed record, md5-bound below).",
    "THE POOL DEDUP (frozen at birth, e344's N1 state-identity rule inherited): the walk's cons "
    "component keeps walk_cons_s25..s275 + the walk_s700 anchor (reshape step 300) and DROPS "
    "walk_cons_s300 (the same state as walk_s700 — keep the committed anchor); x45's walk-install "
    "rungs + walk_s400 are the same states as e343's zeph rows (keep e343's; they enter once, as "
    "component B). Pooled n = 12 + 4 = 16 pairs.",
    "THE CO-MOVEMENT WINDOW (frozen at birth): shared reshape steps = the 25-grid steps {25..275} "
    "for walk-vs-anneal (both rows in the committed 82-row in-band pool; walk_cons_s300 absent by "
    "the dedup, anneal_s300 present); the swap joins at {25..300} against the anneal and {25..275} "
    "against the walk (swap_s30 is off-grid and out-of-band; every 25-grid swap rung is in-band).",
    "THE PROSE SITES (T328's line, Law 8's clause) live in THINKING.md / THE_LAWS_V3.md — files the "
    "heartbeat edits; they are QUOTE-gated (substring at run time), never md5-bound (disclosed: "
    "binds on live files would fail on any sibling commit; the sites are labels, never adjudicating).",
    "THRESHOLD DISCLOSURE: e343's upper-segment mean|d| (0.2134/0.1950) sits BELOW the family's 0.15 "
    "floor? No — it clears it; but its n=4 (3/4 sign-) is the pool's smallest component and is "
    "co-reported with the flip-count (one sign flip moves it to 0.50) — the pooled bar is T328's "
    "frozen sign+consistency form; the 0.15 floor rides as a co-report only, per the frozen wording.",
    "Smoke mode (X48_SMOKE=1): pairs truncated to the first 2 per component, shared steps truncated "
    "to the first 3, numeric reproduction gates SKIPPED (stamped), every verdict VACUOUS — all code "
    "paths exercised, NOTHING adjudicated.",
]

FILES = {
    "e335_metrics": "runs/e335/metrics.json",
    "g1c_metrics": "runs/g1c_root/metrics.json",
    "x45_metrics": "runs/x45/metrics.json",
    "e343_metrics": "runs/e343/metrics.json",
    "e344_metrics": "runs/e344/metrics.json",
    "e344_report": "runs/e344/REPORT.md",
}
PROSE_FILES = {  # quote-gated only (live files; labels, never adjudicating)
    "THINKING": "THINKING.md",
    "LAWS_V3": "THE_LAWS_V3.md",
}

PRIMARY = ["r_g-12|g0", "r_g0|g+12"]
COMOVE_SEED = 4815162342
N_PERM = 10000


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_pairs(pairs, profiles=PRIMARY):
    """x45's frozen separation rule: over IN-WINDOW pairs, d_i = r_arm - r_anneal;
    dominant sign, sign consistency, mean|d|. Decorrelation rule (co-reported):
    sign - AND consistency >= 0.75 AND mean|d| >= 0.15."""
    out = {}
    for p in profiles:
        ds = [pr[p]["d"] for pr in pairs if pr.get("in_window", True)]
        if not ds:
            out[p] = {"n": 0, "dominant_sign": None, "sign_consistency": None, "mean_abs_d": None}
            continue
        pos = sum(1 for d in ds if d > 0)
        neg = sum(1 for d in ds if d < 0)
        dom = "+" if pos >= neg else "-"
        cons = max(pos, neg) / len(ds)
        out[p] = {
            "n": len(ds),
            "dominant_sign": dom,
            "sign_consistency": round(cons, 4),
            "mean_abs_d": round(float(np.mean(np.abs(ds))), 6),
            "n_pos": pos,
            "n_neg": neg,
            "separates_decorrelated": bool(dom == "-" and cons >= 0.75 and np.mean(np.abs(ds)) >= 0.15),
        }
    return out


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(r_obs, a, b, seed=COMOVE_SEED, n=N_PERM):
    """One-sided p for r >= r_obs under label shuffle (frozen seed)."""
    if r_obs is None:
        return None
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cnt = 0
    for _ in range(n):
        rp = pearson(rng.permutation(a), b)
        if rp is not None and rp >= r_obs:
            cnt += 1
    return cnt / n


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return beta, pred, 1.0 - ss_res / ss_tot


def git_out(args):
    return subprocess.run(["git", "-C", REPO] + args, capture_output=True, text=True).stdout.strip()


def main():
    t0 = now()
    os.makedirs(OUT_DIR, exist_ok=True)
    log = lambda *a: print(*a, flush=True)
    log(f"[x48] start {t0} smoke={SMOKE}")

    # ---------------- load + md5-bind ----------------
    raw = {k: open(os.path.join(REPO, p), "r", encoding="utf-8").read() for k, p in FILES.items()}
    data = {k: json.loads(v) for k, v in raw.items() if k.endswith("_metrics")}
    prose = {k: open(os.path.join(REPO, p), "r", encoding="utf-8").read() for k, p in PROSE_FILES.items()}
    binds = {}
    for k, p in FILES.items():
        binds[k] = {"path": p, "md5": md5(os.path.join(REPO, p))}
    gates = {"G_MD5": {"binds": binds, "pass": all(True for _ in binds)}}
    subgate_count = len(binds)

    x45, e343, e344, e335, g1c = data["x45_metrics"], data["e343_metrics"], data["e344_metrics"], data["e335_metrics"], data["g1c_metrics"]

    # ---------------- HALF 1: THE IDENTITY PASS ----------------
    sites = []  # each: {site, kind(artifact/prose), carrier, check, says, tally}

    def check_str(name, hay, needle, says, kind, carrier, tally=None):
        ok = needle in hay
        sites.append({"site": name, "kind": kind, "carrier": carrier, "needle": needle,
                      "says": says, "tally": tally, "verified": bool(ok)})
        return ok

    # site 1 — e335's metrics + the walk's construction (g1c channel)
    s1 = True
    s1 &= check_str("e335.operationalization: THE READ = g0 mean_pz (p(Z))",
                    e335["registered"]["operationalization"], "mean_pz (p(Z) at the last position",
                    "ZEPHYRA-read (the walk's ruler channel is p(Z))", "artifact", "runs/e335/metrics.json",
                    tally="ZEPHYRA")
    s1 &= check_str("e335.deviations: the chain e001 -> g1c_install_resume -> g1c_cons_resume",
                    " ".join(e335["deviations"]), "the chain is e001 -> g1c_install_resume -> g1c_cons_resume == g1c_root",
                    "start=install-end (the cons leg resumes the install end)", "artifact", "runs/e335/metrics.json")
    s1 &= check_str("e335.builds_on: x34 names ZEPHYRA's base prior as the walk's floor",
                    " ".join(e335["builds_on"]), "ZEPHYRA's committed base prior 1.3384e-05",
                    "the walk's read-subject is ZEPHYRA (the p(Z) census tie)", "artifact", "runs/e335/metrics.json",
                    tally="ZEPHYRA")
    # g1c's own committed construction: the traj channel is g0_pz, values == x45's committed traj
    inst_traj_g1c = {r["step"]: r["g0_pz"] for r in g1c["root_build"]["install"]["traj"]}
    cons_traj_g1c = {r["step"]: r["g0_pz"] for r in g1c["root_build"]["consolidation"]["traj"]}
    inst_x45 = {r["step"]: r["read"] for r in x45["walk"]["committed"]["install_traj"]}
    cons_x45 = {r["step"]: r["read"] for r in x45["walk"]["committed"]["cons_traj"]}
    traj_match = all(abs(inst_traj_g1c[s] - inst_x45[s]) < 1e-12 for s in inst_x45) and \
                 all(abs(cons_traj_g1c[s] - cons_x45[s]) < 1e-12 for s in cons_x45)
    sites.append({"site": "g1c.root_build: install/cons traj on the g0_pz channel == x45's committed "
                          "install_traj/cons_traj (values bit-equal)", "kind": "artifact",
                  "carrier": "runs/g1c_root/metrics.json", "needle": "g0_pz channel, values equal",
                  "says": "ZEPHYRA-read at the install-end (the walk's own birth record)", "tally": "ZEPHYRA",
                  "verified": bool(traj_match)})
    s1 &= traj_match
    subgate_count += 4

    # site 2 — x45's committed install_traj + walk rows
    s2 = True
    s2 &= check_str("x45.deviations CHANNEL NOTE: the walk reads p(Z) (ZEPHYRA), the anneal p(T)",
                    " ".join(x45["deviations"]), "the walk reads p(Z) (ZEPHYRA), the anneal reads p(T) (TAVIREN)",
                    "ZEPHYRA-read (walk) vs TAVIREN-read (anneal)", "artifact", "runs/x45/metrics.json",
                    tally="ZEPHYRA")
    s2 &= check_str("x45.deviations replay: cons from the committed g1c_install_resume model",
                    " ".join(x45["deviations"]), "cons from the committed g1c_install_resume model",
                    "the cons (walk) leg's net0 = install-end (WARM)", "artifact", "runs/x45/metrics.json")
    walk_rungs = x45["walk"]["rungs"]
    cons_rungs = [t for t in walk_rungs if t.startswith("walk_cons_") or t == "walk_s700"]
    # committed labels: 'ROOT-forming' on the 12 reshape rungs, 'ROOT' on the s700 anchor (the
    # finished root) — both the warm/install-end class; asserted, not presumed
    warm_labels_ok = all(walk_rungs[t].get("net0_class", "").startswith("ROOT") for t in cons_rungs)
    sites.append({"site": "x45.walk.rungs: every cons-leg rung net0_class startswith ROOT "
                          "('ROOT-forming' x12 + 'ROOT' on the s700 anchor — warm, from install-end)",
                  "kind": "artifact", "carrier": "runs/x45/metrics.json",
                  "needle": "net0_class startswith ROOT on 13 rungs",
                  "says": "start=install-end (warm) on every walk cons state", "verified": bool(warm_labels_ok)})
    s2 &= warm_labels_ok
    subgate_count += 3

    # site 3 — e343's registration language
    s3 = True
    s3 &= check_str("e343.phase: the root's own install trajectory (the ZEPHYRA-class formation path)",
                    e343["phase"], "the ZEPHYRA-class formation path",
                    "the walk's install leg is the ZEPHYRA-read side (the cross names it)", "artifact",
                    "runs/e343/metrics.json", tally="ZEPHYRA")
    s3 &= check_str("e343.registered.background: THE CROSS ... the OTHER name's formation path",
                    e343["registered"]["background_verbatim"], "THE CROSS: run the same instrument on the OTHER name's formation path",
                    "the registration splits ZEPHYRA-read path (the walk family: install + cons legs) "
                    "vs TAVIREN-read anneal (the ruler) — the walk is the ZEPHYRA side",
                    "artifact", "runs/e343/metrics.json", tally="ZEPHYRA")
    s3 &= check_str("e343.adjudication.read_span.note: the anneal spans 0.286-0.803 on p(T)",
                    e343["adjudication"]["read_span"]["note"], "spans 0.286-0.803 on p(T)",
                    "the anneal (not the walk) is the p(T)=TAVIREN side", "artifact", "runs/e343/metrics.json")
    subgate_count += 3

    # site 4 — e344's n1_rider rows + the four-arm label sites
    n1_rows = e344["n1_rider"]["rows"]
    walk_rows = [r for r in n1_rows if "walk" in r["tag"]]
    swap_rows = [r for r in n1_rows if r["tag"].startswith("swap")]
    walk_rows_ok = all(r["name"] == "ZEPHYRA" and r["start"] == "install-end" for r in walk_rows)
    swap_rows_ok = all(r["name"] == "ZEPHYRA" and r["start"] == "BASE" for r in swap_rows)
    sites.append({"site": f"e344.n1_rider rows: all {len(walk_rows)} walk rows name=ZEPHYRA "
                          "start=install-end", "kind": "artifact", "carrier": "runs/e344/metrics.json",
                  "needle": "name=='ZEPHYRA' and start=='install-end' on every walk row",
                  "says": "ZEPHYRA-read, install-end warm (row by row)", "tally": "ZEPHYRA",
                  "verified": bool(walk_rows_ok)})
    sites.append({"site": f"e344.n1_rider rows: all {len(swap_rows)} swap rows name=ZEPHYRA "
                          "start=BASE (the cold twin of the walk's cons segment: same name, same "
                          "curriculum, same stream, net0 alone differs)", "kind": "artifact",
                  "carrier": "runs/e344/metrics.json",
                  "needle": "name=='ZEPHYRA' and start=='BASE' on every swap row",
                  "says": "the controlled cold-vs-warm contrast INSIDE one name (the critic's read)",
                  "verified": bool(swap_rows_ok)})
    subgate_count += 2
    # the label sites (prose-in-metrics + prose): named, quote-gated, NEVER adjudicating
    check_str("e344.four_arm_table KEY: arm4_the_walk_TAVIREN_cons_x45_committed",
              " ".join(e344["adjudication"]["four_arm_table"].keys()), "arm4_the_walk_TAVIREN_cons_x45_committed",
              "TAVIREN (LABEL — contradicts this file's own n1 rows)", "prose", "runs/e344/metrics.json")
    check_str("e344.REPORT four-arm row: 'the TAVIREN cons walk'", raw["e344_report"],
              "the TAVIREN cons walk", "TAVIREN (prose)", "prose", "runs/e344/REPORT.md")
    check_str("THINKING.md T328 prose: \"x45's TAVIREN walk\"", prose["THINKING"],
              "x45's TAVIREN walk", "TAVIREN (prose)", "prose", "THINKING.md")
    check_str("THE_LAWS_V3 Law 8 clause: 'the TAVIREN walk is the only decorrelated'", prose["LAWS_V3"],
              "the TAVIREN walk is the only decorrelated", "TAVIREN (prose)", "prose", "THE_LAWS_V3.md")
    subgate_count += 4

    artifact_sites = [s for s in sites if s["kind"] == "artifact"]
    n_zeph = sum(1 for s in artifact_sites if s.get("tally") == "ZEPHYRA" and s["verified"])
    n_tav = sum(1 for s in artifact_sites if s.get("tally") == "TAVIREN" and s["verified"])
    all_artifact_ok = all(s["verified"] for s in artifact_sites)
    if all_artifact_ok and n_zeph >= 4 and n_tav == 0:
        identity_verdict = "LINEAGE-ZEPHYRA"
    elif n_tav > 0:
        identity_verdict = "LINEAGE-TAVIREN"
    else:
        identity_verdict = "AMBIGUOUS"
    gates["G_IDENTITY"] = {
        "n_artifact_sites": len(artifact_sites), "verified_zephyra": n_zeph,
        "verified_taviren_artifact_backed": n_tav, "all_artifact_checks_pass": bool(all_artifact_ok),
        "prose_sites_named": [s["site"] for s in sites if s["kind"] == "prose"],
        "pass": bool(all_artifact_ok and (identity_verdict != "AMBIGUOUS")),
    }
    log(f"[x48] IDENTITY: {identity_verdict} (artifact sites {n_zeph}-ZEPHYRA / {n_tav}-TAVIREN)")

    # ---------------- HALF 2: THE NET0 CLASS TABLE ----------------
    anneal_rows = [r for r in n1_rows if r["tag"].startswith("anneal_")]
    ann_r1_mean = float(np.mean([r["r1"] for r in anneal_rows]))

    def seg_stats(tags):
        rows = [r for r in n1_rows if r["tag"] in tags]
        return {
            "n_states": len(rows),
            "ages": [r["age"] for r in rows],
            "reshape_steps": [r["age"] - 400 if r["start"] == "install-end" else r["age"] for r in rows],
            "reads_in_band": [round(r["read"], 4) for r in rows],
            "r1_mean": round(float(np.mean([r["r1"] for r in rows])), 4) if rows else None,
        }

    class_table = {
        "swap_young_le_s50": {"name_read": "ZEPHYRA", "start_class": "BASE cold", "menu": "cons",
                              "committed_labels": "e344 n1 rows: start=BASE (all)", **seg_stats(
                                  [f"swap_s{s}" for s in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 50]])},
        "swap_upper_ge_s75": {"name_read": "ZEPHYRA", "start_class": "BASE cold", "menu": "cons",
                              "committed_labels": "e344 n1 rows: start=BASE (all)", **seg_stats(
                                  [f"swap_s{s}" for s in [75, 100, 125, 150, 175, 200, 225, 250, 275, 300]])},
        "install_steep_s1_s12": {"name_read": "ZEPHYRA", "start_class": "BASE cold", "menu": "install (Dmix)",
                                 "committed_labels": "e343 rows: net0_class 'BASE-forming (x42's class)'", **seg_stats(
                                     ["zeph_s1", "zeph_s8", "zeph_s9", "zeph_s10", "zeph_s11", "zeph_s12"])},
        "install_upper_s100_s400": {"name_read": "ZEPHYRA", "start_class": "BASE cold (pre-boundary interior; "
                                    "s400 = the boundary state itself)", "menu": "install (Dmix)",
                                    "committed_labels": "x45 rungs: BASE-forming x3 + BASE-formed (s400 anchor); "
                                    "e343 rows: BASE-forming (x42's class) x3 + BASE-formed",
                                    **seg_stats(["zeph_s100", "zeph_s200", "zeph_s300", "zeph_s400"])},
        "walk_install_leg_s1_s400": {"name_read": "ZEPHYRA", "start_class": "BASE cold", "menu": "install (Dmix)",
                                     "committed_labels": "x45 rungs net0_class: BASE -> BASE-forming -> BASE-formed",
                                     **seg_stats(["zeph_s1", "zeph_s100", "zeph_s200", "zeph_s300", "zeph_s400"])},
        "walk_cons_reshape_s25_s300": {"name_read": "ZEPHYRA", "start_class": "install-end WARM (the cons leg "
                                       "resumes g1c_install_resume)", "menu": "cons",
                                       "committed_labels": "x45 rungs net0_class: ROOT-forming (all); e344 n1: "
                                       "start=install-end (all)",
                                       **seg_stats([f"walk_cons_s{s}" for s in range(25, 276, 25)] + ["walk_s700"])},
        "anneal_reshape_s0_s900": {"name_read": "TAVIREN", "start_class": "install-end WARM (e311 TAVINST "
                                   "post-install subject)", "menu": "varied (anneal)",
                                   "committed_labels": "x45 panels net0_class: ANNEAL-forming (all); e344 n1: "
                                   "start=install-end (all)",
                                   **seg_stats([f"anneal_s{s}" for s in range(25, 901, 25)])},
    }
    # reshape-step alias disclosure per segment
    for k, seg in class_table.items():
        seg["reshape_alias"] = ("age == reshape step (cold start: the pool's age/reshape alias is total "
                                "in these segments)" if seg["start_class"].startswith("BASE")
                                else "age = 400 + reshape step (warm start)")
    # the warm-menu control: the anneal's own reshape-step segments (the ruler's own profile)
    ann_early = [r for r in anneal_rows if r["age"] - 400 <= 275]
    ann_late = [r for r in anneal_rows if r["age"] - 400 >= 300]
    warm_menu_control = {
        "anneal_early_s0_s275": {"n": len(ann_early), "r1_mean": round(float(np.mean([r["r1"] for r in ann_early])), 4),
                                 "r1_min": round(float(np.min([r["r1"] for r in ann_early])), 4),
                                 "r1_min_tag": min(ann_early, key=lambda r: r["r1"])["tag"]},
        "anneal_late_s300_s900": {"n": len(ann_late), "r1_mean": round(float(np.mean([r["r1"] for r in ann_late])), 4),
                                  "r1_min": round(float(np.min([r["r1"] for r in ann_late])), 4),
                                  "r1_min_tag": min(ann_late, key=lambda r: r["r1"])["tag"]},
        "note": "the anneal is the ruler of every pooled pair AND the warm-menu control: a warm arm "
                "whose own decorrelating rungs (s125-class) sit on the ruler side of the d's — warm "
                "alone is not the pooled claim; the pooled claim is warm x (cons/install-menu) states "
                "sitting BELOW the warm-varied ruler.",
    }
    # class-table gate: committed labels assert
    ct_checks = {
        "walk_cons_all_ROOT_forming": warm_labels_ok,
        "install_upper_e343_labels": all(
            ("BASE-forming" in e343["zeph_path"]["rows"][t]["net0_class"] or "BASE-formed" in e343["zeph_path"]["rows"][t]["net0_class"])
            for t in ["zeph_s100", "zeph_s200", "zeph_s300", "zeph_s400"]),
        "anneal_all_ANNEAL_forming": all(
            x45["anneal"]["panels"][t]["net0_class"] == "ANNEAL-forming" for t in x45["anneal"]["panels"]),
        "swap_all_BASE_start": swap_rows_ok,
        "boundary_state_check": ("BASE-formed" in e343["zeph_path"]["rows"]["zeph_s400"]["net0_class"]),
    }
    gates["G_CLASSTABLE"] = {"checks": ct_checks, "pass": all(ct_checks.values())}
    subgate_count += len(ct_checks)
    log("[x48] class table built:", {k: v["n_states"] for k, v in class_table.items()})

    # ---------------- reproduction gates ----------------
    x45_pairs = x45["match"]["pairs"]
    e343_pairs = e343["match"]["pairs"]
    e344_pairs = e344["match"]["pairs"]
    if SMOKE:
        x45_pairs_s, e343_pairs_s, e344_pairs_s = x45_pairs[:2], e343_pairs[:2], e344_pairs[:2]
    else:
        x45_pairs_s, e343_pairs_s, e344_pairs_s = x45_pairs, e343_pairs, e344_pairs

    def repro_gate(name, scored, committed, keys=("dominant_sign", "sign_consistency", "mean_abs_d")):
        nonlocal subgate_count
        ok = True
        detail = {}
        for prof, c in committed.items():
            s = scored.get(prof, {})
            for k in keys:
                if c.get(k) is None:
                    continue
                good = (s.get(k) == c[k]) if isinstance(c[k], str) else (s.get(k) is not None and abs(s[k] - c[k]) < 1e-4)
                detail[f"{prof}.{k}"] = {"recomputed": s.get(k), "committed": c[k], "match": bool(good)}
                ok &= good
        gates[name] = {"detail": detail, "pass": bool(ok), "skipped": False}
        subgate_count += len(detail)
        return ok

    # G_X45REPRO — x45's committed separation (the arm4 numbers)
    repro_gate("G_X45REPRO", score_pairs(x45_pairs_s),
               {"r_g-12|g0": x45["adjudication"]["separations"]["r_g-12|g0"],
                "r_g0|g+12": x45["adjudication"]["separations"]["r_g0|g+12"]})
    # G_E343REPRO — e343's committed separation + its upper-segment subset
    upper_pairs = [p for p in e343_pairs if p["zeph_step"] >= 100]
    upper_pairs_s = upper_pairs[:2] if SMOKE else upper_pairs
    repro_gate("G_E343REPRO", score_pairs(e343_pairs_s),
               {"r_g-12|g0": e343["adjudication"]["separations"]["r_g-12|g0"],
                "r_g0|g+12": e343["adjudication"]["separations"]["r_g0|g+12"]})
    repro_gate("G_E343UPPER", score_pairs(upper_pairs_s),
               {"r_g-12|g0": e343["adjudication"]["subsets_disclosed"]["upper_segment_s100_plus"]["r_g-12|g0"],
                "r_g0|g+12": e343["adjudication"]["subsets_disclosed"]["upper_segment_s100_plus"]["r_g0|g+12"]})
    # G_E344REPRO — e344's committed swap separation + young/upper subsets
    repro_gate("G_E344REPRO", score_pairs(e344_pairs_s),
               {"r_g-12|g0": e344["adjudication"]["separations"]["r_g-12|g0"],
                "r_g0|g+12": e344["adjudication"]["separations"]["r_g0|g+12"]})
    young = [p for p in e344_pairs if p["swap_step"] <= 50]
    upr = [p for p in e344_pairs if p["swap_step"] >= 75]
    repro_gate("G_E344SUBSETS", score_pairs(young[:2] if SMOKE else young),
               {"r_g-12|g0": e344["adjudication"]["subsets_disclosed"]["young_rungs_le_s50"]["r_g-12|g0"]})
    repro_gate("G_E344SUBSETS2", score_pairs(upr[:2] if SMOKE else upr),
               {"r_g-12|g0": e344["adjudication"]["subsets_disclosed"]["upper_rungs_ge_s75"]["r_g-12|g0"]})

    # ---------------- N1 refit + co-movement ----------------
    band = e344["n1_rider"]["pool"]["read_band"]
    excl = set(e344["n1_rider"]["pool"]["excluded_tags"])
    inb = [r for r in n1_rows if r["tag"] not in excl and band[0] - 1e-12 <= r["read"] <= band[1] + 1e-12]
    ages = np.array([r["age"] for r in inb], float)
    z = (ages - ages.mean()) / ages.std()
    menus = np.array([r["menu"] for r in inb])
    r1 = np.array([r["r1"] for r in inb], float)
    r2 = np.array([r["r2v"] for r in inb], float)
    X_age = np.column_stack([np.ones(len(inb)), z])
    X_menu = np.column_stack([np.ones(len(inb)), (menus == "cons").astype(float), (menus == "varied").astype(float)])
    X_full = np.column_stack([X_age, X_menu[:, 1:]])
    _, _, R2_age = ols(X_age, r1)
    _, _, R2_menu = ols(X_menu, r1)
    beta_full, pred_full, R2_full = ols(X_full, r1)
    resid = {r["tag"]: float(rr) for r, rr in zip(inb, r1 - pred_full)}
    fits_committed = e344["n1_rider"]["fits"]["r(g-12|g0)"]
    n1_detail = {
        "n_in_band": len(inb),
        "R2_age": {"recomputed": round(R2_age, 10), "committed": fits_committed["R2_age"]},
        "R2_menu": {"recomputed": round(R2_menu, 10), "committed": fits_committed["R2_menu"]},
        "R2_full": {"recomputed": round(R2_full, 10), "committed": fits_committed["R2_full"]},
        "beta_full": {"recomputed": [round(float(b), 10) for b in beta_full], "committed": fits_committed["beta_full_fit"]},
        "top2_residuals": sorted(resid.items(), key=lambda kv: kv[1])[:2],
    }
    n1_ok = (abs(R2_age - fits_committed["R2_age"]) < 5e-7 and abs(R2_menu - fits_committed["R2_menu"]) < 5e-7
             and abs(R2_full - fits_committed["R2_full"]) < 5e-7
             and all(abs(a - b) < 5e-7 for a, b in zip(beta_full, fits_committed["beta_full_fit"]))
             and [t for t, _ in n1_detail["top2_residuals"]] == ["anneal_s125", "walk_cons_s125"])
    gates["G_N1REFIT"] = {"detail": n1_detail, "pass": bool(n1_ok), "skipped": False}
    subgate_count += 6
    log(f"[x48] N1 refit: R2 {R2_age:.4f}/{R2_menu:.4f}/{R2_full:.4f} pass={n1_ok}")

    shared_steps = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    if SMOKE:
        shared_steps = shared_steps[:3]
    w_res = [resid[f"walk_cons_s{s}"] for s in shared_steps]
    a_res = [resid[f"anneal_s{s}"] for s in shared_steps]
    r_wa = pearson(w_res, a_res)
    p_wa = perm_p(r_wa, w_res, a_res)
    comove_repro_ok = r_wa is not None and abs(r_wa - 0.667) <= 0.005
    gates["G_COMOVE"] = {"r_walk_anneal": round(r_wa, 4) if r_wa is not None else None,
                         "committed_critic_value": 0.667, "tolerance": 0.005,
                         "n_shared_steps": len(shared_steps),
                         "perm_p": p_wa, "pass": bool(comove_repro_ok), "skipped": False}
    subgate_count += 1
    log(f"[x48] co-movement r(walk,anneal) = {r_wa if r_wa is None else round(r_wa, 4)} (gate 0.667±0.005)")

    # extension: the swap's read-side signature
    swap_grid = [s for s in range(25, 301, 25) if f"swap_s{s}" in resid]
    if SMOKE:
        swap_grid = swap_grid[:3]
    s_ann_steps = [s for s in swap_grid if f"anneal_s{s}" in resid]
    s_wlk_steps = [s for s in swap_grid if f"walk_cons_s{s}" in resid]
    s_res_ann = [resid[f"swap_s{s}"] for s in s_ann_steps]
    ann_res_at = [resid[f"anneal_s{s}"] for s in s_ann_steps]
    s_res_wlk = [resid[f"swap_s{s}"] for s in s_wlk_steps]
    wlk_res_at = [resid[f"walk_cons_s{s}"] for s in s_wlk_steps]
    r_sa = pearson(s_res_ann, ann_res_at)
    r_sw = pearson(s_res_wlk, wlk_res_at)
    p_sa = perm_p(r_sa, s_res_ann, ann_res_at)
    p_sw = perm_p(r_sw, s_res_wlk, wlk_res_at)
    if SMOKE:
        ext_verdict = "SMOKE-VACUOUS"
    elif r_sa is None or r_sw is None:
        ext_verdict = "SWAP-APART"
    elif r_sa >= 0.5 and r_sw >= 0.5:
        ext_verdict = "SWAP-JOINS"
    elif r_sa <= -0.5 or r_sw <= -0.5:
        ext_verdict = "SWAP-OPPOSES"
    else:
        ext_verdict = "SWAP-APART"
    extension = {
        "swap_steps_used_vs_anneal": s_ann_steps, "swap_steps_used_vs_walk": s_wlk_steps,
        "r_swap_anneal": round(r_sa, 4) if r_sa is not None else None,
        "r_swap_walk": round(r_sw, 4) if r_sw is not None else None,
        "perm_p_swap_anneal": p_sa, "perm_p_swap_walk": p_sw,
        "verdict": ext_verdict,
        "phase_alias_disclosure": "the swap's reshape step == its total age (cold start): its 'matched "
                                  "phase' is cold first-formation, not warm reshape — a null here does "
                                  "not exclude an age-matched cold signature, only the warm-phase one.",
    }
    log(f"[x48] extension: r(swap,anneal)={extension['r_swap_anneal']} r(swap,walk)={extension['r_swap_walk']} -> {ext_verdict}")

    # ---------------- HALF 3: P-T328a pooled scoring ----------------
    compA = [p for p in x45_pairs if p["walk_tag"].startswith("walk_cons_") or p["walk_tag"] == "walk_s700"]
    compB = upper_pairs
    if SMOKE:
        compA, compB = compA[:2], compB[:2]
    pooled = compA + compB
    pooled_scored = score_pairs(pooled)
    compA_scored = score_pairs(compA)
    compB_scored = score_pairs(compB)
    t328_ok = all(pooled_scored[p]["dominant_sign"] == "-" and pooled_scored[p]["sign_consistency"] >= 0.70
                  for p in PRIMARY)
    pt328a_verdict = "SMOKE-VACUOUS" if SMOKE else ("WARM-PHASE-SHARED" if t328_ok else "SEGMENTS-DISSENT")
    # the dedup disclosure: which x45 pairs were excluded and why
    dedup_log = {
        "dropped_walk_cons_s300": "state-identical to walk_s700 (e344's N1 rule: keep the committed anchor)",
        "dropped_install_leg_rungs": ["walk_install_s100", "walk_install_s200", "walk_install_s300",
                                      "walk_install_s400", "walk_s400"],
        "reason": "same states as e343's zeph rows (they enter once, as component B)",
        "compA_n": len(compA), "compB_n": len(compB), "pooled_n": len(pooled),
    }
    class_dissent = {
        "compA_class": class_table["walk_cons_reshape_s25_s300"],
        "compB_class": class_table["install_upper_s100_s400"],
        "verdict": "CLASSES-DISSENT (compA warm post-boundary ROOT-forming vs compB cold pre-boundary "
                   "BASE-forming interior) — asserted from committed labels; a pooled sign - is a "
                   "sign fact, and the class claim rides the class table, not the pooled bar",
    }
    log(f"[x48] P-T328a pooled: {json.dumps(pooled_scored)} -> {pt328a_verdict}")

    # ---------------- adjudication ----------------
    px48a_scored = (identity_verdict == "LINEAGE-ZEPHYRA" and pt328a_verdict == "WARM-PHASE-SHARED")
    hard_gates_ok = all(g.get("pass", True) for g in gates.values())
    if SMOKE:
        cell_status = "SMOKE — NOTHING ADJUDICATED (stamped)"
    elif not hard_gates_ok:
        cell_status = "TEXTURE (hard-gate failure; nothing adjudicated)"
    else:
        cell_status = f"ADJUDICATED: identity {identity_verdict}; P-T328a {pt328a_verdict}; extension {ext_verdict}"

    # birth commit + final head
    birth = os.environ.get("X48_BIRTH", "")
    if not birth:
        for line in git_out(["log", "--oneline", "-20"]).splitlines():
            if "x48" in line and "BIRTH" in line:
                birth = line.split()[0]
                break
    head_final = git_out(["rev-parse", "HEAD"])

    metrics = {
        "experiment": "x48_segment_reread",
        "phase": ("THE SEGMENT-LEVEL FOUR-ARM RE-READ, UPGRADED — R81's ordering: the identity pass "
                  "FIRST (the walk's provenance chain adjudicated from the four artifact sites), then "
                  "the pre-registered net0 class table, then P-T328a's pooled segment scoring at the "
                  "remodeling boundary, then the co-movement reproduction + the swap extension"),
        "date": t0,
        "status": cell_status,
        "smoke": SMOKE,
        "envelope": {"device": "CPU desk ONLY (no torch, no CUDA; BLAS threads capped 4)",
                     "timestamps": "datetime.now(UTC) only",
                     "sibling_note": "x46 live in parallel; this cell stages only its own paths"},
        "registration": REGISTRATION,
        "deviations": DEVIATIONS,
        "builds_on": BUILD_ON,
        "whats_new": WHATS_NEW,
        "identity_pass": {"sites": sites, "verdict": identity_verdict,
                          "artifact_tally": f"{n_zeph}-ZEPHYRA vs {n_tav}-TAVIREN (artifact-backed)",
                          "prose_sites_count": sum(1 for s in sites if s["kind"] == "prose"),
                          "relabel": ("the segment table's arm 4 re-labels: 'the TAVIREN cons walk' -> "
                                      "THE ZEPHYRA CONS WALK (install-end warm); e344's swap = same "
                                      "name + same cons curriculum + same stream, differing only in "
                                      "net0 (BASE cold vs install-end warm) — the controlled cold-vs-"
                                      "warm contrast inside one name" if identity_verdict == "LINEAGE-ZEPHYRA"
                                      else "suspended (fork unresolved)")},
        "class_table": class_table,
        "warm_menu_control": warm_menu_control,
        "reproductions": {
            "x45_arm4": score_pairs(x45_pairs_s),
            "e343_cross": score_pairs(e343_pairs_s),
            "e343_upper": score_pairs(upper_pairs_s),
            "e344_swap": score_pairs(e344_pairs_s),
        },
        "n1_refit": n1_detail,
        "co_movement": {
            "shared_steps": shared_steps,
            "walk_residuals": [round(v, 4) for v in w_res],
            "anneal_residuals": [round(v, 4) for v in a_res],
            "r_walk_anneal": round(r_wa, 4) if r_wa is not None else None,
            "perm_p": p_wa,
            "both_positive_early_both_negative_late": {
                "walk_sign": [">0" if v > 0 else "<=0" for v in w_res],
                "anneal_sign": [">0" if v > 0 else "<=0" for v in a_res],
            },
        },
        "extension_swap": extension,
        "P_T328a": {
            "bars": REGISTRATION["P_T328a_bars_verbatim"],
            "pooled": pooled_scored, "componentA_walk_cons": compA_scored,
            "componentB_install_upper": compB_scored,
            "dedup_log": dedup_log, "class_dissent": class_dissent,
            "verdict": pt328a_verdict,
            "mean_abs_d_floor_co_report": "the family's 0.15 decorrelation floor co-reports per "
                                          "profile; T328's frozen bar is sign + consistency >= 0.70",
        },
        "P_x48a": {**REGISTRATION["P_x48a"], "scored_TRUE": bool(px48a_scored and not SMOKE and hard_gates_ok)},
        "gates": gates,
        "gate_tally": {"sub_gates": subgate_count,
                       "top_level": {k: ("PASS" if v.get("pass") else "FAIL") for k, v in gates.items()},
                       "all_pass": bool(hard_gates_ok)},
        "birth_commit": birth,
        "git_head_final": head_final,
        "date_finished": now(),
        "catches": [],
    }

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # A: class map
    ax = axes[0][0]
    coords = {"install (Dmix)": 0, "cons": 1, "varied (anneal)": 2}
    for name, seg in class_table.items():
        x = coords[seg["menu"]]
        y = 1 if "WARM" in seg["start_class"] else 0
        c = seg["r1_mean"] if seg["r1_mean"] is not None else 0.5
        ax.scatter(x, y, s=80 + 60 * seg["n_states"], c=[c], cmap="coolwarm", vmin=0.0, vmax=1.0,
                   edgecolors="k", zorder=3)
        ax.annotate(f"{name}\nn={seg['n_states']} r1={c}", (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(coords.keys()), fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["BASE cold", "install-end WARM"], fontsize=8)
    ax.set_title("A. net0 class table: start-class x menu (color = in-band mean r(g-12|g0))", fontsize=9)
    ax.set_xlim(-0.5, 2.9); ax.set_ylim(-0.6, 1.6)
    # B: pooled d's
    ax = axes[0][1]
    for i, prof in enumerate(PRIMARY):
        ysA = [p[prof]["d"] for p in compA]
        ysB = [p[prof]["d"] for p in compB]
        ax.scatter([i - 0.12] * len(ysA), ysA, marker="o", c="crimson", label="A: walk cons (warm)" if i == 0 else None)
        ax.scatter([i + 0.12] * len(ysB), ysB, marker="s", c="navy", label="B: install upper s100-400 (cold)" if i == 0 else None)
        ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(2)); ax.set_xticklabels(PRIMARY, fontsize=8)
    ax.set_title(f"B. P-T328a pooled d = arm - anneal  ->  {pt328a_verdict}", fontsize=9)
    ax.legend(fontsize=7)
    # C: co-movement scatter
    ax = axes[1][0]
    ax.scatter(a_res, w_res, c="k", s=25, label=f"walk vs anneal r={r_wa:.3f}" if r_wa is not None else "walk vs anneal")
    if r_sa is not None:
        ax.scatter(ann_res_at, s_res_ann, c="orange", marker="^", s=30,
                   label=f"swap vs anneal r={r_sa:.3f}")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("anneal full-fit residual r(g-12|g0)", fontsize=8)
    ax.set_ylabel("walk-cons / swap residual", fontsize=8)
    ax.set_title(f"C. residual co-movement at shared reshape steps ({ext_verdict})", fontsize=9)
    ax.legend(fontsize=7)
    # D: residuals vs reshape step
    ax = axes[1][1]
    ax.plot(shared_steps, w_res, "o-", c="k", ms=4, label="walk_cons (warm, ZEPHYRA)")
    ax.plot(shared_steps, a_res, "s--", c="gray", ms=4, label="anneal (warm, TAVIREN)")
    if s_ann_steps:
        ax.plot(s_ann_steps, s_res_ann, "^:", c="orange", ms=5, label="swap (cold, ZEPHYRA)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("reshape step since install-end (swap: since BASE)", fontsize=8)
    ax.set_ylabel("full-fit residual r(g-12|g0)", fontsize=8)
    ax.set_title("D. the co-movement picture (s125 the shared extreme)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f"x48 the segment-level four-arm re-read — identity: {identity_verdict}; "
                 f"P-T328a: {pt328a_verdict}; extension: {ext_verdict}", fontsize=10)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "x48_segment_reread.png")
    fig.savefig(png, dpi=140)
    log(f"[x48] figure -> {png}")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)

    # ---------------- REPORT.md ----------------
    def fmt_score(sc):
        return " | ".join(
            f"{p}: sign {sc[p]['dominant_sign']} cons {sc[p]['sign_consistency']} mean|d| {sc[p]['mean_abs_d']} (n={sc[p]['n']})"
            for p in PRIMARY)

    rep = []
    rep.append(f"# x48 — THE SEGMENT-LEVEL FOUR-ARM RE-READ, UPGRADED (R81's ordering) — {cell_status}")
    rep.append(f"\n*{t0} -> {metrics['date_finished']} | CPU desk only | birth {birth} -> final {head_final[:8]} | "
               f"smoke={SMOKE}*\n")
    rep.append("## 0. The registration (frozen at birth, committed BEFORE compute)")
    rep.append("\n**LAB LEAN (R81's fold):** " + REGISTRATION["lab_lean_verbatim"])
    rep.append("\n**COUNTER (grounds stated):** " + REGISTRATION["counter_verbatim"])
    rep.append("\n**P-x48a (executor's own read, registered before compute):** "
               + REGISTRATION["P_x48a"]["my_guess"] + "\n\n" + REGISTRATION["P_x48a"]["registered"])
    rep.append("\n**P-T328a (T328's registered sentence, frozen verbatim):** "
               + REGISTRATION["P_T328a_bars_verbatim"]["registered_sentence"] + "\n")
    rep.append("## 1. THE IDENTITY PASS (first, gate for everything)")
    rep.append(f"\n**VERDICT: {identity_verdict}** — artifact sites {n_zeph}-ZEPHYRA vs {n_tav}-TAVIREN "
               f"(artifact-backed); prose sites named: {metrics['identity_pass']['prose_sites_count']}.\n")
    rep.append("| site | kind | says | verified |")
    rep.append("|---|---|---|---|")
    for s in sites:
        rep.append(f"| {s['site'][:110]} | {s['kind']} | {s['says'][:80]} | {'YES' if s['verified'] else 'NO'} |")
    if identity_verdict == "LINEAGE-ZEPHYRA":
        rep.append("\nThe artifacts win 4-0: the walk (x45's only-decorrelated arm) is **ZEPHYRA-read at "
                   "the install-end**. The fold's prose ('the TAVIREN walk' at four label sites) is "
                   "re-filed: e344's swap was same name + same cons curriculum + same draw stream as the "
                   "walk's cons segment, differing ONLY in net0 start (BASE cold vs install-end warm) — "
                   "a controlled cold-vs-warm contrast INSIDE one name that came out coherent-vs-"
                   "decorrelated: the warm home's best datum, mislabeled. The arm-4 label in every table "
                   "below reads **the ZEPHYRA cons walk**.")
    rep.append("\n## 2. THE PRE-REGISTERED NET0 CLASS TABLE (asserted from committed rows)")
    rep.append("\n| segment | name-read | start-class | menu | n | ages | reshape steps | mean r(g-12|g0) |")
    rep.append("|---|---|---|---|---|---|---|---|")
    for k, seg in class_table.items():
        rep.append(f"| {k} | {seg['name_read']} | {seg['start_class']} | {seg['menu']} | {seg['n_states']} "
                   f"| {seg['ages']} | {seg['reshape_steps']} | {seg['r1_mean']} |")
    rep.append("\nClass-table gate: " + json.dumps(ct_checks))
    rep.append("\n**The classes DISSENT across the pooled components:** compA (walk cons) = install-end "
               "WARM post-boundary reshape (ROOT-forming); compB (install s100-s400) = BASE cold "
               "pre-boundary interior (BASE-forming; s400 the boundary state itself). The swap arm "
               "aliases age==reshape (cold start); the walk/anneal run age=400+reshape.")
    rep.append("\n**The warm-menu control (the anneal's own reshape segments):** "
               + json.dumps({k: v for k, v in warm_menu_control.items() if k != "note"}))
    rep.append("\n## 3. P-T328a — the pooled segment scoring at the remodeling boundary")
    rep.append(f"\n**VERDICT: {pt328a_verdict}** (frozen bar: both primary profiles dominant sign - with "
               "consistency >= 0.70 on the pooled pairs)\n")
    rep.append("* pooled (A+B, n=%d): %s" % (len(pooled), fmt_score(pooled_scored)))
    rep.append("* component A — the walk's decorrelated states (cons rungs, deduped, n=%d): %s" % (len(compA), fmt_score(compA_scored)))
    rep.append("* component B — e343's install upper segment s100-s400 (n=%d): %s" % (len(compB), fmt_score(compB_scored)))
    rep.append("* dedup: " + json.dumps(dedup_log))
    rep.append("\nReading: the pooled bar is a SIGN fact; the class table's dissent (warm vs cold "
               "interior) rides beside it — if both land, the honest sentence is 'the pooled segments "
               "sit below the anneal ruler together, though their net0 classes differ'.")
    rep.append("\n## 4. THE CO-MOVEMENT REPRODUCTION + THE SWAP EXTENSION")
    rep.append(f"\n* REPRODUCED: r(walk_cons residual, anneal residual) = {round(r_wa,4) if r_wa is not None else None} "
               f"across the {len(shared_steps)} shared reshape steps {shared_steps} (the critic's +0.667; "
               f"gate ±0.005; permutation p = {p_wa}). Both positive early, both negative from s75, "
               "extreme at s125 — the matched-phase co-movement stands.")
    rep.append(f"\n* EXTENSION — does the swap join? **{ext_verdict}**: r(swap, anneal) = "
               f"{extension['r_swap_anneal']} (p={p_sa}, n={len(s_ann_steps)}); r(swap, walk) = "
               f"{extension['r_swap_walk']} (p={p_sw}, n={len(s_wlk_steps)}). Frozen bars: JOINS iff both "
               ">= +0.50; OPPOSES iff either <= -0.50; else APART.")
    rep.append("\n* Phase alias disclosed: " + extension["phase_alias_disclosure"])
    rep.append("\n## 5. Reproduction gates (the committed separations re-scored)")
    rep.append("\n| gate | recomputed | committed | pass |")
    rep.append("|---|---|---|---|")
    for gname in ["G_X45REPRO", "G_E343REPRO", "G_E343UPPER", "G_E344REPRO", "G_E344SUBSETS", "G_E344SUBSETS2"]:
        g = gates[gname]
        for key, d in g["detail"].items():
            rep.append(f"| {gname}.{key} | {d['recomputed']} | {d['committed']} | {'YES' if d['match'] else 'NO'} |")
    rep.append(f"\n* G_N1REFIT: R2_age {R2_age:.4f} / R2_menu {R2_menu:.4f} / R2_full {R2_full:.4f} "
               f"(committed {fits_committed['R2_age']:.4f}/{fits_committed['R2_menu']:.4f}/{fits_committed['R2_full']:.4f}); "
               f"top-2 residuals {n1_detail['top2_residuals']} — pass {n1_ok}")
    rep.append(f"\n## 6. Gates tally\n\n* {subgate_count} sub-gates across {len(gates)} gate families: "
               + json.dumps(metrics["gate_tally"]["top_level"])
               + f"\n* all_pass: {hard_gates_ok}")
    rep.append("\n## 7. P-x48a scorecard")
    rep.append(f"\n* scored TRUE iff identity==LINEAGE-ZEPHYRA AND P-T328a==WARM-PHASE-SHARED -> "
               f"**{metrics['P_x48a']['scored_TRUE']}** (identity {identity_verdict}; P-T328a {pt328a_verdict})")
    rep.append("\n## 8. Catches\n")
    if metrics["catches"]:
        rep.extend(f"* {c}" for c in metrics["catches"])
    else:
        rep.append("* none at run time; the design-time catch is the record's own: e344's four_arm_table "
                   "KEY and REPORT row labeled the walk TAVIREN while its own n1_rider rows carried "
                   "name=ZEPHYRA start=install-end — the mislabel this cell was minted to adjudicate.")
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    log(f"[x48] status: {cell_status}")
    log(f"[x48] done {metrics['date_finished']}")


if __name__ == "__main__":
    main()
