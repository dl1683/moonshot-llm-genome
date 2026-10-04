"""E227 — FQ1, THE FACULTY AT HOME (the cross-scale bridge; scratch/
fresh_questions.md FQ1, dispatched as the 2.74M faculty census).

WHY (FQ1's own words, scratch/fresh_questions.md (a)): "At 124M: the
erosion's locus is the FEW-SHOT-FOLLOWING FACULTY — template-general,
draw-replicating, graded ctrl < template < nearrel, perplexity improving
throughout (T183/e182c2 ...). At 2.74M: every forgetting verdict was read
through FACT rulers (the g-12 channel, the battery channel, row0) — a
few-shot/template instrument was never pointed at a tiny-scale wash."
And (e): "tiny nets may have no few-shot faculty at all (then the negative
is itself the scale finding: the 124M locus is an emergence, and the tiny
law stays a fact-law); the template battery may floor at 2.74M (weak
instruments manufacture nulls — W021's family)."

THE 124M REFERENCE (runtime-read from runs/e182c2/metrics.json, never
transcribed): declines at +80 on phase-1's saved states: ctrl 0.397 <
tmpl 0.561 < near 0.766 (fact 0.429); the fresh draw at +50: ctrl 0.275 <
tmpl 0.420 < near 0.591 — TEMPLATE-GENERAL + DRAW-REPLICATES fired; the
family split (runs/e215/metrics.json): family carries eta2 0.577 of the
hold-ratio variance, sorting lang/founder-anchor vs cap-cur/product
EXPOSURE-INDEPENDENTLY. THE QUESTION: is that faculty AT HOME in the
small models — do the same ordering and family structure exist at 2.74M
— or is it scale-emergent? And on the rhythm organ: does the wash tempo
structure read as a SPLINT (a scale-specific scaffold) or the
FACULTY-KEEPER (the same object at every scale)?

THE CELL (dispatch letter): (1) run the template batteries (ctrl /
template / nearrel + the family battery) on the consolidated 2.74M
checkpoint (runs/checkpoints/e131_consolidated_e113.pt) under matched
e182c2 conventions; (2) if any trace shows, exposure-matched fresh wash
draws to test draw-replication at the small scale; (3) a 10M scale point
only if the 2.74M answer is non-trivial (a separate decision, not in
this file). NO organ arm here: the splint-vs-keeper question is read
off the faculty's own presence/absence at 2.74M (the dispatch's
parenthetical labels, frozen below).

THE 2.74M BATTERY SET (the ZEPHYRA-world adaptation of e182c2's
conventions — R = battery mean p(answer char) at the last position, the
lineage's own battery_cell instrument, imported; decline = 1-R(s)/R(0)):
  * fact   — the committed ruler: install-60 g-12 host contexts (118
    chars), p('Z'). THE provenance anchor (t=0 must reproduce e_chart's
    committed rung-0 0.9155886173248291).
  * near   — NEARREL: held-30 host contexts at g-12, p('Z') — the same
    relation over held-out entities (g1b's committed held battery).
  * tmpl2  — TEMPLATE (form change): the SAME relation in the 2-SHOT
    EXEMPLAR form — [72-char host snippet + 'ZEPHYRA' + '\\n'] x2
    exemplars + [72-char query snippet], p('Z') at the query end. The
    e182c2 tmpl analog (its reversed-copular form <-> our exemplar
    form); FQ1: "the install protocol itself (masked/jitter replay) IS
    a few-shot surface".
  * mis2   — the exemplar-CONTENT intervention: identical windows with
    exemplar name 'MIRABEL' (7 chars, 0 corpus occurrences — same
    geometry as ZEPHYRA); TWO reads: p('Z') (knowledge under exemplar
    conflict) and p('M') (exemplar-following — the pure few-shot read,
    the zclass instrument).
  * snip0  — the length-matched 0-shot baseline (72-char snippet alone),
    reads p('Z') and p('M') — the few-shot LIFT denominator.
  * ctrl   — GENERIC: pooled val-split (wash-disjoint) carriers:
    colon (speaker-format 'NAME' -> ':'), word (frequent val words cut
    at half -> next char), cname (val names mid-text -> first char).
  * FAMILY battery — fact+near+tmpl2+mis2Z+snip0Z items (the
    OUT-OF-CORPUS expression surface, where the 124M comparison is
    honest) + ctrl items (in-corpus carriers; co-report only — the
    wash trains that distribution, so including them in the primary
    eta2 would manufacture the split; registered below).

THE WASH: the g1b C-arm machinery VERBATIM via import (G1.g1_wash:
neutral-bank 16 + random-train-window 16, full-token CE, AdamW (0.9,
0.95) wd 0.1 clip 1.0, the seed-10902 stream arithmetic) at THREE
pre-registered doses: LINEAGE lr 1e-3 (the scale's own kill clock;
draws 10902 locked + 22701 + 22702), GENTLE lr 1e-4 (draws 22711 +
22712), MID lr 3e-4 (draw 22721). The dose ladder exists because the
124M wash's ROLE (corpus-healthy, graded declines) maps to a lower lr
at 2.74M (the 1e-3 lineage wash damages corpus CE itself: g1b C+2
CE_R 2.03 vs root 1.64); both roles are covered, nothing post-hoc.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  - FACULTY-AT-HOME: "the ordering and/or family split replicate at
    2.74M — the faculty predates scale; rhythm organ = faculty-keeper
    candidate."
  - NOT-AT-HOME: "the batteries read flat or unstructured at 2.74M
    where 124M splits — the faculty is scale-emergent; the organ reads
    as splint."
  - PARTIAL-TRACE: "anything between — mapped honestly."
  - FQ1's channel-death branch (the spec's own third branch, VERBATIM
    from scratch/fresh_questions.md (c)): "the template battery erodes
    first or fastest -> the tiny-scale 'fact death' is partly CHANNEL
    death, and the entire tiny-scale forgetting law (C3-C5) acquires a
    second reading through the e182c2 lens at its own scale."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars; the dispatch bars are qualitative so EVERY
numeric form below is a REGISTERED ADDITION, flagged for the report):
  * O1 (the ordering test, per state): decl_ctrl < decl_tmpl2 <
    decl_near strictly AND decl_ctrl <= decl_near / 1.5 (the e182c2
    1.5x separation constant, inverted), all three declines > 0.005.
  * Qualifying arm (for ordering/family clauses): the arm's near
    battery declines into [0.5, 0.95] at its comparison state s* (the
    erosion must be real and non-saturated); s* = the DEEPEST state
    with decl_near < 0.95 (else the arm is disclosed as saturated and
    non-qualifying — the lineage arm's expected fate, its kill clock
    co-reported).
  * ORDERING-REPLICATES: O1 holds at >= 2 registered states on EVERY
    qualifying arm (>= 2 qualifying arms required across the dose
    ladder / draws).
  * FAMILY-REPLICATES: one-way ANOVA eta2(family) over the
    out-of-corpus families' item hold-ratios hr = R_item(s*)/R_item(0)
    at s* >= 0.50 (the 124M point 0.577, runs/e215/metrics.json) on
    EVERY qualifying arm.
  * FACULTY-AT-HOME := ORDERING-REPLICATES OR FAMILY-REPLICATES.
  * NOT-AT-HOME := (no qualifying arm exists at any dose — the
    instrument/dose floor) OR (every qualifying arm fails O1 at every
    state AND eta2 < 0.35 everywhere) AND the tmpl2/mis2 instrument
    did not floor for a reason the near battery shares.
  * PARTIAL-TRACE := everything else (e.g. one clause fires at one
    dose only; eta2 in [0.35, 0.50); the new-form batteries floor at
    t=0 while the committed channels read).
  * CHANNEL-DEATH (FQ1 verbatim clause): decl_tmpl2 >= decl_near AND
    decl_tmpl2 >= decl_fact at >= 2 states of a qualifying arm, OR
    tmpl2 crosses decline 0.5 earliest of the four batteries.
  * DRAW sub-verdict (per dose, e182c2's convention adapted):
    DRAW-REPLICATES if the battery-decline Spearman across draws at s*
    >= 0.8 over the non-trivial batteries (decl in (0.02, 0.98));
    DRAW-DIFFERS if < 0.8 for every draw pair; else GRADED.
  * Screening (e182c2's discipline VERBATIM in char form): new-form
    batteries (tmpl2/mis2/snip0 read as one item set) and ctrl
    candidates are probed at t=0 on the pristine root AFTER the bars
    are frozen and BEFORE any wash compute; gate (argmax == ans AND
    p >= 0.8) OR (rank < 5 AND p >= 0.5); tmpl2 keeps its gate-passers
    (cap 20 by p0, floor 6 flagged reduced); mis2/snip0 read the SAME
    kept item set (like-for-like across forms); ctrl pools capped 8
    per sub-family, pooled cap 24; fact/near are COMMITTED instruments
    read whole (no gate — their t=0 values are the committed reads).

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_ROOT — runs/checkpoints/e131_consolidated_e113.pt loads bit-exact
    (max|diff| 0.0 vs file); the t=0 ruler (install-60 g-12) reproduces
    the committed 0.9155886173248291 within 5e-6 (CPU fp32, 8 threads,
    the e223 hard-bound reference).
  * G_WASH — the LINEAGE arm (seed 10902) must reproduce g1b's
    committed C-arm reads at +2 within 0.02 (g-12 0.02708 / g0 0.11471,
    runtime-read from runs/g1b/metrics.json) and CE_R within 0.05
    (2.0321) — device-texture tolerance on the provenance tie.
  * G_CORPUS — ZEPH-free train text; install mix FLORIZEL 19 /
    ELIZABETH 41 after the SPLICE_RNG shuffle (the g1b gate).
  * G_BATT — every battery window is deterministic; ctrl windows must
    not occur as substrings of train_text (wash-disjoint carriers) and
    contain no 'ZEPH'.
  * G_HEALTH — CE_R read at every state (co-report; no pass/fail: the
    disclosed asymmetry — 124M's wash IMPROVED ppl while the 2.74M
    lineage wash damages CE_R; the gentle arm is the role-matched
    health reference).

HONESTY REFLEX (before believing any trace):
  * Does CE/generic quality alone predict the split? The IN-CORPUS
    families (ctrl carriers are wash-distributional) are excluded from
    the primary eta2 exactly because the wash trains that content;
    the out-of-corpus set is where 124M's comparison is honest.
  * Does intervening change it? The exemplar-content swap (tmpl2 vs
    mis2) is the in-prompt intervention; the lr ladder is the dose
    intervention; the draw seeds are the stochastic intervention.
  * Draw noise quantified: n=3 lineage draws, n=2 gentle draws, n=1
    mid draw; per-item hr tables committed for the family battery.
  * The form change bundles (context length 118->72 + exemplars +
    separators) — disclosed as e182c2's form+direction confound was;
    snip0 is the length-matched control.
  * CPU fp32 probes everywhere; washes GPU fp32 (the g1b C-arm device
    convention; patterns, never bits, across devices).

COMPUTE ENVELOPE (owner directive 2026-10-04, temporary MAXIMUM
priority): GPU used assertively but NO concurrent jobs; single bursts
<< 180 s (each wash is ~300 steps of a 2.74M net, seconds); 35 s
cooldowns between arms; still temp-aware via the imported pick_dev
double-poll. Probes CPU fp32, 8 threads.

PROVENANCE: the organism + wash + battery instruments are
lab/g1_anchored_ball.py VERBATIM via import (g1_wash / battery_cell /
evl_load / load_g1 / val_windows; the g1b config override to the
2.74M family), itself the e176n/e185/e068 lineage; the install/held
occurrence rebuild is g1b's VERBATIM arithmetic (E43.find_occ +
SPLICE_RNG shuffle); CE_R bank e065's (seed 26502). Builds on:
e182c2/T183 (the 124M faculty locus + battery conventions), e214-e221/
T187-T197 (the 124M relational signature + W028), e131/e113 (the
consolidated root), g1b (the 2.74M wash stream + C-arm record), FQ1
(scratch/fresh_questions.md). NEW: the 2.74M few-shot battery set
(tmpl2/mis2/snip0/ctrl families), the dose ladder, the ordering/
family/channel adjudications at 2.74M, the wash-draw census.

Outputs: runs/e227/{metrics.json, battery_curves.png, family_battery.png}
+ per-arm journals. No NOTES/THINKING/QUEUE/STATE edits (dispatch).

Run:  cd lab && python e227_faculty_at_home.py   (E227_SMOKE=1: 2-step
      wash, grid {1,2}, own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402
import torch.nn.functional as F                        # noqa: E402

torch.set_num_threads(8)                              # the lab convention

import common                                          # noqa: E402
from common import Cfg, CharCorpus, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402
import g1_anchored_ball as G1                          # noqa: E402 — the wash + battery instruments

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E227_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "E227_FACULTY_AT_HOME"
DIRNAME = "e227_smoke" if SMOKE else "e227"

# ---- the g1b config override: the 2.74M family (6/6/192/256) ---------------
E227_CFG = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
E227_PARAMS = 2_739_072
G1.G1_CFG = E227_CFG
G1.G1_PARAMS = E227_PARAMS

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (DIRNAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ------------------------------------------------------- frozen constants
ROOT_CK = "e131_consolidated_e113.pt"
RULER_REF = 0.9155886173248291        # e_chart's committed rung-0 read
G_ROOT_TOL = 5e-6
G_WASH_TOL_PZ = 0.02                  # device texture on the +2 tie
G_WASH_TOL_CE = 0.05

# the arms: (tag, lr, seed, ckpt grid) — registered BEFORE compute
ARMS: list[tuple[str, float, int, tuple[int, ...]]] = [
    ("L10902", 1e-3, 10902, (1, 2, 4, 10, 25, 50)),          # lineage, locked
    ("L22701", 1e-3, 22701, (1, 2, 4, 10, 25, 50)),          # lineage fresh 1
    ("L22702", 1e-3, 22702, (1, 2, 4, 10, 25, 50)),          # lineage fresh 2
    ("G22711", 1e-4, 22711, (4, 10, 25, 50, 100, 300)),      # gentle d1
    ("G22712", 1e-4, 22712, (4, 10, 25, 50, 100, 300)),      # gentle d2
    ("M22721", 3e-4, 22721, (2, 4, 10, 25, 50, 100, 300)),   # mid dose
]
if SMOKE:
    ARMS = [("smk", 1e-3, 10902, (1, 2))]

COOLDOWN_S = 35.0                     # the dispatch's 30-60 s band

# battery-form constants (frozen)
SNIP_LEN = 72                         # the 2-shot exemplar snippet length
SEP = "\n"                            # exemplar separator (corpus-natural)
ALT_NAME = "MIRABEL"                  # 7 chars, 0 corpus occurrences
CTRL_CTX = 118                        # ctrl context length (= the g-12 ruler)
CTRL_SUB_CAP = 8                      # per ctrl sub-family cap
CTRL_CAP = 24                         # pooled ctrl cap

# bar constants (frozen; registered additions — flagged in the docstring)
SEP_BAND = 1.5                        # decl_ctrl <= decl_near / SEP_BAND
ETA2_BAR = 0.50                       # the 124M family point (0.577)
ETA2_NULL = 0.35                      # the unstructured floor
NEAR_QUAL_LO, NEAR_QUAL_HI = 0.5, 0.95
DECL_TRIVIAL = 0.005
ORDER_STATES_MIN = 2
DRAW_RHO_BAR = 0.8
CAP_ITEMS = 20                        # e182's battery cap
FLOOR_ITEMS = 6                       # e182's floor

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "FACULTY-AT-HOME": "the ordering and/or family split replicate at "
            "2.74M — the faculty predates scale; rhythm organ = "
            "faculty-keeper candidate.",
        "NOT-AT-HOME": "the batteries read flat or unstructured at 2.74M "
            "where 124M splits — the faculty is scale-emergent; the organ "
            "reads as splint.",
        "PARTIAL-TRACE": "anything between — mapped honestly.",
        "CHANNEL-DEATH (FQ1 spec branch, verbatim)": "the template battery "
            "erodes first or fastest -> the tiny-scale 'fact death' is "
            "partly CHANNEL death, and the entire tiny-scale forgetting "
            "law (C3-C5) acquires a second reading through the e182c2 lens "
            "at its own scale.",
    },
    "lean": "the record's tiny-scale whispers (e164's storage-behind-the-"
        "killed-readout; the install protocol being itself a few-shot "
        "surface) lean PARTIAL-TRACE-to-FACULTY (some structure expected); "
        "but the 2-shot form has never been pointed at this organism and "
        "may floor (FQ1's own failure mode (e)) — nothing guaranteed; the "
        "openness is the point",
    "operationalizations": "see the script docstring (O1 / qualifying arm "
        "/ s* / ORDERING-REPLICATES / FAMILY-REPLICATES / eta2 >= 0.50 on "
        "the OUT-OF-CORPUS families / DRAW rho >= 0.8 / screening gate "
        "(argmax==ans & p>=0.8)|(rank<5 & p>=0.5), cap 20, floor 6) — all "
        "REGISTERED ADDITIONS (the dispatch bars are qualitative)",
    "registration": "bars + pools + arms + constants frozen in this file "
        "and committed BEFORE any wash compute; no bar shopping",
}

deviations: list[str] = [
    "SMOKE SHAKEDOWN (runs/e227_smoke; mechanics only, nothing "
    "adjudicated): G_ROOT verified at dp 6e-8; the seed-10902 2-step "
    "wash reproduced g1b's committed C-arm EXACTLY (g-12 0.0271 / CE_R "
    "2.0321 — the wash tie already bit-clean); the colon/cname ctrl "
    "sub-families FLOOR at t=0 (p(':') ~ 0.000 after speaker names — the "
    "2.74M does not carry the speaker-colon format; a genuine instrument "
    "floor, kept in screening as the honest record); in response the word "
    "pool was widened top-10 -> top-24 and the 'sent' (? -> newline) "
    "sub-family added — both BEFORE the committed registration/wash, "
    "t=0-information only (the e182c2 screening precedent).",
]


# --------------------------------------------------------------- utilities

@torch.no_grad()
def probe_items(net, ids_list: list[torch.Tensor], ans_id: int,
                bs: int = 16) -> list[dict]:
    """Per-item p/rank/argmax read — battery_cell's arithmetic itemwise
    (the instrument, applied per probe so item-level hold-ratios exist).
    Windows bucketed by length (ctrl mixes sub-families at one length by
    construction; this guards the general case)."""
    net.eval()
    out: list[dict] = [None] * len(ids_list)           # type: ignore[list-item]
    bylen: dict[int, list[int]] = {}
    for i, t in enumerate(ids_list):
        bylen.setdefault(int(t.shape[0]), []).append(i)
    for L, idxs in bylen.items():
        for c in range(0, len(idxs), bs):
            chunk = idxs[c:c + bs]
            w = torch.stack([ids_list[i] for i in chunk])
            lg, _ = net(w)
            pr = F.softmax(lg[:, -1], -1)
            for k, gi in enumerate(chunk):
                row = pr[k]
                out[gi] = {
                    "p": float(row[ans_id]),
                    "rank": int((row > row[ans_id]).sum().item()),
                    "argmax": int(row.argmax().item()),
                    "argmax_p": float(row.max().item()),
                }
    assert all(r is not None for r in out)
    return out


def bat_mean(recs: list[dict], ans_id: int) -> dict:
    ps = [r["p"] for r in recs]
    return {"mean_p": float(np.mean(ps)), "n": len(ps),
            "frac_argmax": float(np.mean(
                [r["argmax"] == ans_id for r in recs]))}


def gate_pass(rec: dict, ans_id: int) -> bool:
    """e182's gate VERBATIM in char form: (top1 and p>=0.8) | (top5 and
    p>=0.5)."""
    return ((rec["argmax"] == ans_id and rec["p"] >= 0.8)
            or (rec["rank"] < 5 and rec["p"] >= 0.5))


def eta2(groups: dict[str, list[float]]) -> dict:
    """One-way ANOVA eta2 (e215's readout): SS_between / SS_total."""
    allv = [v for g in groups.values() for v in g]
    if len(allv) < 8 or len(groups) < 2:
        return {"eta2": None, "n": len(allv), "k": len(groups)}
    gm = float(np.mean(allv))
    sst = float(np.sum((np.array(allv) - gm) ** 2))
    ssb = 0.0
    for g in groups.values():
        ssb += len(g) * (float(np.mean(g)) - gm) ** 2
    return {"eta2": (ssb / sst if sst > 1e-12 else None),
            "ssb": ssb, "sst": sst, "n": len(allv), "k": len(groups),
            "group_means": {k: float(np.mean(v)) for k, v in groups.items()},
            "group_ns": {k: len(v) for k, v in groups.items()}}


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------- battery windows

def build_batteries(corpus, train_text, val_text, install_occ, held_occ,
                    zid: int):
    """All battery windows, deterministic. Returns (batteries, meta).
    batteries: key -> {"ids": list[Tensor], "ans": int, "items": [...],
    "family": str, "committed": bool}"""
    stoi = corpus.stoi
    enc = lambda s: corpus.encode(s)

    def ctx_windows(occs, j):
        # g1b's battery construction VERBATIM: train_text[p-PRE-j : p]
        return [enc(train_text[p - G1.PRE - j: p]) for p, _ in occs]

    B: dict[str, dict] = {}

    # committed instruments (read whole, no gate)
    B["fact"] = {"ids": ctx_windows(install_occ, -12), "ans": zid,
                 "family": "fact", "committed": True,
                 "desc": "install-60 g-12 host contexts (THE ruler)"}
    B["near"] = {"ids": ctx_windows(held_occ, -12), "ans": zid,
                 "family": "near", "committed": True,
                 "desc": "held-30 g-12 host contexts (the committed held "
                         "battery; nearrel: same relation, held-out "
                         "entities)"}

    # the 2-shot exemplar forms over held-30 snippets
    snips = [enc(train_text[p - SNIP_LEN: p]) for p, _ in held_occ]
    alt = enc(ALT_NAME)

    def two_shot(i: int, exemplar_name_ids: torch.Tensor) -> torch.Tensor:
        ex = [snips[j] for j in range(len(snips)) if j != i][:2]  # e182's rule
        parts = []
        for e in ex:
            parts += [e, exemplar_name_ids, enc(SEP)]
        parts.append(snips[i])
        return torch.cat(parts)

    tmpl_ids = [two_shot(i, enc(G1.NAME)) for i in range(len(snips))]
    mis_ids = [two_shot(i, alt) for i in range(len(snips))]
    snp_ids = list(snips)
    B["tmpl2"] = {"ids": tmpl_ids, "ans": zid, "family": "tmpl",
                  "committed": False,
                  "desc": f"2-shot exemplar form ({SNIP_LEN}-char held "
                          f"snippets, exemplars {G1.NAME}), p(Z)"}
    B["mis2Z"] = {"ids": mis_ids, "ans": zid, "family": "misZ",
                  "committed": False,
                  "desc": f"same windows, exemplars {ALT_NAME}; p(Z) = "
                          "knowledge under exemplar conflict"}
    B["mis2M"] = {"ids": mis_ids, "ans": stoi["M"], "family": "misM",
                  "committed": False,
                  "desc": f"p(M) on the {ALT_NAME}-exemplar windows = the "
                          "exemplar-following (zclass) read"}
    B["snip0Z"] = {"ids": snp_ids, "ans": zid, "family": "snipZ",
                   "committed": False,
                   "desc": f"{SNIP_LEN}-char snippet alone, p(Z) — the "
                           "length-matched 0-shot baseline"}
    B["snip0M"] = {"ids": snp_ids, "ans": stoi["M"], "family": "snipM",
                   "committed": False,
                   "desc": "snippet alone, p(M) — the 0-shot M baseline "
                           "(the lift denominator)"}

    # ---- ctrl pools from the VAL split (wash-disjoint carriers) ----------
    ctrl_meta: dict[str, dict] = {}
    n_scan_skips = {"short_ctx": 0, "in_train": 0, "zeph": 0}

    def val_ctx(s_end: int) -> str | None:
        """118-char val context ending at s_end; None if unavailable or
        contaminated (the full window must not occur in train_text)."""
        if s_end < CTRL_CTX:
            n_scan_skips["short_ctx"] += 1
            return None
        ctxw = val_text[s_end - CTRL_CTX: s_end]
        if "ZEPH" in ctxw:
            n_scan_skips["zeph"] += 1
            return None
        if ctxw in train_text:
            n_scan_skips["in_train"] += 1
            return None
        return ctxw

    # names by val frequency (frozen rule: top-10 uppercase tokens len>=5)
    toks = re.findall(r"\b[A-Z][A-Za-z]{4,}\b", val_text)
    name_list = [w for w, _ in Counter(toks).most_common(10)]
    # words: top-24 alphabetic tokens len>=6 (deterministic corpus stat;
    # widened from 10 after the smoke shakedown showed n=8 passers thin —
    # registration refinement BEFORE the committed registration/wash)
    wtoks = re.findall(r"\b[a-z]{6,}\b", val_text)
    word_list = [w for w, _ in Counter(wtoks).most_common(24)]

    # colon: val occurrences of speaker-format names, answer ':'
    colon_c = []
    for nm in name_list:
        for m in re.finditer(re.escape(nm), val_text):
            s = m.start()
            pre2 = val_text[max(0, s - 2): s]
            post1 = val_text[m.end(): m.end() + 1]
            if pre2 == "\n\n" and post1 == ":":
                ctxw = val_ctx(s)
                if ctxw is None:
                    continue
                colon_c.append({"fact": f"colon:{nm}", "answer": ":",
                                "ids": enc(ctxw), "sub": "colon"})
    # word: occurrences cut at half, answer = next char; ONE carrier per
    # word (frozen: the first full-context occurrence)
    word_c = []
    for w in word_list:
        cut = -(-len(w) // 2)          # ceil(len/2), frozen
        for m in re.finditer(re.escape(w), val_text):
            s = m.start()
            ctxw = val_ctx(s + cut)
            if ctxw is None:
                continue
            word_c.append({"fact": f"word:{w}", "answer": w[cut],
                           "ids": enc(ctxw), "sub": "word"})
            break
    # cname: mid-text val names (not speaker-format), answer = first char
    cname_c = []
    for nm in name_list:
        for m in re.finditer(re.escape(nm), val_text):
            s = m.start()
            pre2 = val_text[max(0, s - 2): s]
            post1 = val_text[m.end(): m.end() + 1]
            if pre2 == "\n\n" or post1 == ":":
                continue
            ctxw = val_ctx(s)
            if ctxw is None:
                continue
            cname_c.append({"fact": f"cname:{nm}", "answer": nm[0],
                            "ids": enc(ctxw), "sub": "cname"})
    # sent: sentence-final '?' -> '\n' (the strong format family the smoke
    # shakedown found; first 30 document-order occurrences)
    sent_c = []
    for m in re.finditer(re.escape("?"), val_text):
        s = m.end()
        ctxw = val_ctx(s)
        if ctxw is None:
            continue
        sent_c.append({"fact": f"sent:{s}", "answer": "\n",
                       "ids": enc(ctxw), "sub": "sent"})
        if len(sent_c) >= 30:
            break
    ctrl_meta["name_list"] = name_list
    ctrl_meta["word_list"] = word_list
    ctrl_meta["candidates"] = {
        "colon": len(colon_c), "word": len(word_c), "cname": len(cname_c),
        "sent": len(sent_c)}
    ctrl_meta["scan_skips"] = n_scan_skips
    ctrl_c = colon_c + word_c + cname_c + sent_c
    for c in ctrl_c:
        assert len(c["ids"]) == CTRL_CTX and len(c["answer"]) == 1
        c["ans_id"] = stoi[c["answer"]]
    B["ctrl"] = {"cands": ctrl_c, "ans": None, "family": "ctrl",
                 "committed": False, "desc": "pooled val-split generic "
                 "carriers (colon/word/cname/sent), gated at t=0"}
    return B, ctrl_meta


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(DIRNAME)
    log(f"E227 THE FACULTY AT HOME (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": NAME,
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "question": ("is the 124M few-shot-following faculty (graded "
                     "ctrl<tmpl<near erosion; family-first sorting) AT "
                     "HOME at 2.74M on the consolidated root, or "
                     "scale-emergent? splint vs faculty-keeper for the "
                     "rhythm organ"),
        "registration": ("bars frozen VERBATIM from the dispatch + FQ1's "
                         "channel branch; all numeric operationalizations "
                         "are REGISTERED ADDITIONS (flagged); pools + arms "
                         "committed before any wash compute; no bar "
                         "shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "builds_on": ["e182c2/T183 (the 124M faculty locus + conventions)",
                      "e214-e221/T187-T197 + W028 (the 124M relational "
                      "signature)", "e131/e113 (the consolidated root)",
                      "g1b (the 2.74M wash stream + C-arm record)",
                      "FQ1 (scratch/fresh_questions.md)"],
        "whats_new": ["the 2.74M few-shot battery set (tmpl2/mis2/snip0/"
                      "ctrl)", "the wash dose ladder (1e-4/3e-4/1e-3)",
                      "the ordering/family/channel adjudications at 2.74M",
                      "the wash-draw census at 2.74M"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------- P0 runtime-bound references (never transcribed)
    p_e182c2 = common.REPO / "runs" / "e182c2" / "metrics.json"
    p_g1b = common.REPO / "runs" / "g1b" / "metrics.json"
    p_e215 = common.REPO / "runs" / "e215" / "metrics.json"
    m182 = json.loads(p_e182c2.read_text(encoding="utf-8"))
    mg1b = json.loads(p_g1b.read_text(encoding="utf-8"))
    me215 = json.loads(p_e215.read_text(encoding="utf-8"))
    ref124 = {
        "declines_at_80": m182["adjudication_part1_template"]
        ["declines_at_deepest"],
        "fresh_declines_at_50": m182["adjudication_part2_fresh"]["declines"],
        "family_eta2": me215["predictor_ladder"]["family_anova"]["eta2"],
        "source": "runs/e182c2/metrics.json + runs/e215/metrics.json",
    }
    ctrace = mg1b["traces"]["C"]
    c2 = next(r for r in ctrace if r["freeze_steps"] == 2)
    refC2 = {"gm12": c2["gm12"], "g0": c2["g0"], "ce_r": c2["ce_r"],
             "source": "runs/g1b/metrics.json traces.C[freeze_steps=2]"}
    metrics["reference_124M"] = ref124
    metrics["reference_g1b_C2"] = refC2
    log(f"124M ref: ctrl {ref124['declines_at_80']['ctrl']:.3f} < tmpl "
        f"{ref124['declines_at_80']['tmpl']:.3f} < near "
        f"{ref124['declines_at_80']['near']:.3f}; family eta2 "
        f"{ref124['family_eta2']:.3f} | g1b C+2 gm12 {refC2['gm12']:.4f}")

    # ---------------- P1 the organism + protocol rebuild (g1b VERBATIM)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    G_NAMEFREE = {"corpus_zeph_count": train_text.count("ZEPH"),
                  "pass": bool(train_text.count("ZEPH") == 0)}
    assert G_NAMEFREE["pass"]

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_CORP = {"corpus_zeph": G_NAMEFREE["corpus_zeph_count"],
              "install_mix": mix,
              "n_install": len(install_occ), "n_held": len(held_occ),
              "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                           and len(held_occ) == 30)}
    assert G_CORP["pass"], f"protocol drift {mix}"
    metrics["gates"] = {"G_CORP": G_CORP}
    log(f"G_CORP: install60 {mix}, held30: PASS")

    # ---------------- P2 the root (G_ROOT)
    ck_path = common.REPO / "runs" / "checkpoints" / ROOT_CK
    root_net = G1.load_g1(ck_path)
    assert root_net.num_params() == E227_PARAMS
    theta0 = {k: v.detach().clone() for k, v in root_net.state_dict().items()}
    raw = torch.load(ck_path, map_location="cpu", weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    md0 = max(float((theta0[k].float() - raw_sd[k].float()).abs().max())
              for k in raw_sd)
    import hashlib
    md5root = hashlib.md5(
        (ck_path.read_bytes())).hexdigest()
    G_ROOT = {"checkpoint": f"runs/checkpoints/{ROOT_CK}",
              "meta": raw.get("meta"), "md5": md5root,
              "max_abs_diff_vs_file": md0,
              "params": root_net.num_params(),
              "ruler_ref": RULER_REF,
              "tol": G_ROOT_TOL, "t0_ruler": None, "pass": None}
    metrics["gates"]["G_ROOT"] = G_ROOT
    metrics["organism"] = {
        "cfg": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block": 256,
                "vocab": 65},
        "params": E227_PARAMS,
        "lineage": "e001 base <- e043 install (e048_repro, seed 42) <- e113 "
                   "jitter-replay consolidation (seed 10901) = e131 "
                   "consolidated; ruler = install-60 g-12 p(Z)",
    }
    log(f"G_ROOT: loaded {ROOT_CK} bit-diff {md0:.1e} md5 {md5root[:12]}")

    # CE_R bank (e065 VERBATIM)
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    ce0 = G1.ce_fixed_cpu(root_net, *r_eval_xy)
    log(f"root CE_R {ce0:.4f}")
    G_ROOT["ce_r_t0"] = ce0

    # ---------------- P3 the batteries + screening (t=0 ONLY)
    B, ctrl_meta = build_batteries(corpus, train_text, val_text,
                                   install_occ, held_occ, zid)
    net0 = root_net

    # committed instruments first (the ruler verification — battery_cell,
    # the EXACT committed instrument, for the gate; probe_items co-reads)
    gm12_gate_ids = torch.stack(B["fact"]["ids"])
    ruler_cell = G1.battery_cell(net0, gm12_gate_ids, zid)
    fact0 = probe_items(net0, B["fact"]["ids"], zid)
    near0 = probe_items(net0, B["near"]["ids"], zid)
    t0_ruler = float(np.mean([r["p"] for r in fact0]))
    t0_ruler_cell = float(ruler_cell["mean_pz"])
    G_ROOT["t0_ruler_probe_items"] = t0_ruler
    G_ROOT["t0_ruler"] = t0_ruler_cell
    G_ROOT["pass"] = bool(md0 == 0.0 and abs(t0_ruler_cell - RULER_REF)
                          <= G_ROOT_TOL)
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"G_ROOT: t0 ruler battery_cell {t0_ruler_cell:.10f} "
        f"(probe_items {t0_ruler:.10f}) vs ref {RULER_REF} "
        f"(dp {abs(t0_ruler_cell - RULER_REF):.2e}) -> "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    assert G_ROOT["pass"] or SMOKE, "G_ROOT FAILED"

    # the new-form instruments: t=0 screen + gate (e182c2's discipline)
    screen = {"when": "after bars frozen+committed, before any wash compute",
              "information": "t=0 pristine root ONLY"}
    tmpl0 = probe_items(net0, B["tmpl2"]["ids"], zid)
    kept_idx = [i for i, r in enumerate(tmpl0) if gate_pass(r, zid)]
    if len(kept_idx) > CAP_ITEMS:                       # cap 20 by p0
        kept_idx = sorted(kept_idx, key=lambda i: -tmpl0[i]["p"])[:CAP_ITEMS]
    reduced = len(kept_idx) < FLOOR_ITEMS
    B["tmpl2"]["ids"] = [B["tmpl2"]["ids"][i] for i in kept_idx]
    B["mis2Z"]["ids"] = [B["mis2Z"]["ids"][i] for i in kept_idx]
    B["mis2M"]["ids"] = [B["mis2M"]["ids"][i] for i in kept_idx]
    B["snip0Z"]["ids"] = [B["snip0Z"]["ids"][i] for i in kept_idx]
    B["snip0M"]["ids"] = [B["snip0M"]["ids"][i] for i in kept_idx]
    B["tmpl2"]["kept_idx"] = kept_idx
    screen["tmpl2"] = {"n_probed": len(tmpl0), "n_passed": len(kept_idx),
                       "reduced_flag": bool(reduced),
                       "p0_range": [float(min(r["p"] for r in tmpl0)),
                                    float(max(r["p"] for r in tmpl0))]}
    log(f"screen tmpl2: {len(kept_idx)}/{len(tmpl0)} pass the gate"
        + (" — FLAGGED reduced (<6)" if reduced else ""))

    # ctrl gating (per-item against its OWN answer id)
    ctrl_c = B["ctrl"]["cands"]
    ctrl_recs = []
    for c in ctrl_c:
        rec2 = probe_items(net0, [c["ids"]], c["ans_id"])[0]
        ctrl_recs.append((c, rec2))
    kept_ctrl = [(c, r) for c, r in ctrl_recs if gate_pass(r, c["ans_id"])]
    # per-sub-family cap, then pooled cap (document order)
    bysub: dict[str, list] = {}
    for c, r in kept_ctrl:
        bysub.setdefault(c["sub"], []).append((c, r))
    kept_ctrl = []
    for sub in ("colon", "word", "cname", "sent"):
        kept_ctrl += bysub.get(sub, [])[:CTRL_SUB_CAP]
    if len(kept_ctrl) > CTRL_CAP:
        kept_ctrl = kept_ctrl[:CTRL_CAP]
    ctrl_reduced = len(kept_ctrl) < FLOOR_ITEMS
    B["ctrl"]["ids"] = [c["ids"] for c, _ in kept_ctrl]
    B["ctrl"]["ans_ids"] = [c["ans_id"] for c, _ in kept_ctrl]
    B["ctrl"]["items"] = [c["fact"] for c, _ in kept_ctrl]
    B["ctrl"]["subs"] = [c["sub"] for c, _ in kept_ctrl]
    screen["ctrl"] = {"candidates": ctrl_meta["candidates"],
                      "n_passed": len(kept_ctrl),
                      "reduced_flag": bool(ctrl_reduced),
                      "by_sub": {s: sum(1 for c, _ in kept_ctrl
                                        if c["sub"] == s)
                                 for s in ("colon", "word", "cname", "sent")},
                      "name_list": ctrl_meta["name_list"],
                      "word_list": ctrl_meta["word_list"]}
    log(f"screen ctrl: {len(kept_ctrl)} kept "
        f"({screen['ctrl']['by_sub']}); pools {ctrl_meta['candidates']}")
    metrics["screening"] = screen
    metrics["batteries"] = {k: {"desc": v.get("desc"),
                                "n": len(v.get("ids", [])),
                                "family": v.get("family"),
                                "committed": v.get("committed", False)}
                            for k, v in B.items() if "ids" in v}
    metrics["gates"]["G_BATT"] = {
        "deterministic": True,
        "ctrl_contamination_scans": ctrl_meta["scan_skips"],
        "ctrl_window_length": CTRL_CTX,
        "committed_instruments": "fact/near read whole (no gate); their "
                                 "t=0 values are the committed reads",
        "new_form_gate": "(argmax==ans & p>=0.8)|(rank<5 & p>=0.5), cap "
                         f"{CAP_ITEMS} by p0, floor {FLOOR_ITEMS}",
        "pass": bool(G_CORP["pass"] and G_ROOT["pass"]),
    }
    write_metrics("PARTIAL: batteries built + screened; washes pending")

    # battery read plan: per state, read every battery per-item
    BAT_KEYS = ["fact", "near", "tmpl2", "mis2Z", "mis2M", "snip0Z",
                "snip0M", "ctrl"]
    t0_reads: dict[str, dict] = {}
    for k in BAT_KEYS:
        if k == "ctrl":
            per = []
            for ids_, a_ in zip(B["ctrl"]["ids"], B["ctrl"]["ans_ids"]):
                r = probe_items(net0, [ids_], a_)[0]
                per.append({"p": r["p"], "rank": r["rank"]})
            t0_reads[k] = {"mean_p": float(np.mean([r["p"] for r in per])),
                           "per": per}
        else:
            per = probe_items(net0, B[k]["ids"], B[k]["ans"])
            t0_reads[k] = {"mean_p": float(np.mean([r["p"] for r in per])),
                           "per": [{"p": r["p"], "rank": r["rank"]}
                                   for r in per]}
    metrics["t0_reads"] = {k: {"mean_p": v["mean_p"], "n": len(v["per"])}
                           for k, v in t0_reads.items()}
    # the few-shot lift at t=0 (zclass)
    lift0 = (t0_reads["mis2M"]["mean_p"] - t0_reads["snip0M"]["mean_p"])
    metrics["t0_fewshot_lift"] = {
        "p_M_2shot": t0_reads["mis2M"]["mean_p"],
        "p_M_0shot": t0_reads["snip0M"]["mean_p"], "lift": lift0,
        "note": "the pure few-shot-following read at t=0 (the zclass "
                "instrument); a lift near zero = the faculty absent/weak "
                "at 2.74M — FQ1's failure mode (e)"}
    log(f"t=0 reads: fact {t0_reads['fact']['mean_p']:.4f} | near "
        f"{t0_reads['near']['mean_p']:.4f} | tmpl2 "
        f"{t0_reads['tmpl2']['mean_p']:.4f} | mis2Z "
        f"{t0_reads['mis2Z']['mean_p']:.4f} | mis2M "
        f"{t0_reads['mis2M']['mean_p']:.4f} | snip0Z "
        f"{t0_reads['snip0Z']['mean_p']:.4f} | ctrl "
        f"{t0_reads['ctrl']['mean_p']:.4f} | few-shot lift(M) {lift0:.4f}")
    write_metrics("PARTIAL: t=0 reads done; washes pending")

    # ---------------- P4 the wash arms
    gm12_ids = gm12_gate_ids
    g0_ids = torch.stack([corpus.encode(train_text[p - G1.PRE: p])
                          for p, _ in install_occ])
    # the neutral anchor bank (e170's construction VERBATIM, g1b's code)
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    assert len(n_starts) == 16
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc = sum(1 for s in n_starts
             if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    whost = sum(1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1]
                       for f in G1.HOSTS))
    G_ANCH = {"seed": G1.E170_ANCHOR_SEED, "n": 16, "starts": n_starts,
              "tries": tries, "rejections": rejections,
              "windows_with_host_content": whost, "junctions_covered": jc,
              "pass": bool(whost == 0 and jc == 0
                           and anchor_neutral.shape == (16, G1.BLOCK)),
              "note": "e170's neutral bank VERBATIM (g1b's own gate); the "
                      "G_WASH tie depends on this identity"}
    metrics["gates"]["G_ANCHOR"] = G_ANCH
    assert G_ANCH["pass"], f"G_ANCHOR FAILED: {G_ANCH}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} ({rejections} rejections/"
        f"{tries} tries), host 0/16, junctions 0/16: PASS")

    def read_state(sd: dict) -> dict:
        net = G1.evl_load(sd)
        out = {}
        for k in BAT_KEYS:
            if k == "ctrl":
                per = []
                for ids_, a_ in zip(B["ctrl"]["ids"], B["ctrl"]["ans_ids"]):
                    r = probe_items(net, [ids_], a_)[0]
                    per.append({"p": r["p"]})
                out[k] = {"mean_p": float(np.mean([r["p"] for r in per])),
                          "per": per}
            else:
                per = probe_items(net, B[k]["ids"], B[k]["ans"])
                out[k] = {"mean_p": float(np.mean([r["p"] for r in per])),
                          "per": [{"p": r["p"]} for r in per]}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        return out

    arms_out: dict[str, dict] = {}
    for ai, (tag, lr, seed, cks) in enumerate(ARMS):
        jp = rd / f"journal_{tag}.json"
        if jp.exists():
            try:
                arms_out[tag] = json.loads(jp.read_text(encoding="utf-8"))
                log(f"arm {tag}: journal restored "
                    f"({len(arms_out[tag]['states'])} states)")
                continue
            except Exception as e:                     # noqa: BLE001
                log(f"journal {tag} unreadable ({e}); recompute")
        net0_arm = G1.evl_load(theta0)
        log(f"ARM {tag}: lr {lr:g} seed {seed} ckpts {list(cks)} "
            f"(wash: g1_wash VERBATIM, batch 32, AdamW(0.9,0.95) wd 0.1 "
            f"clip 1.0)")
        res = G1.g1_wash(tag, net0_arm, anchor_neutral, train_ids, itos,
                         r_eval_xy, gm12_ids, g0_ids, zid,
                         target_mode="true", ckpt_steps=cks, lr=lr,
                         seed=seed)
        states = [{"step": 0, "reads": {k: {"mean_p": t0_reads[k]["mean_p"]}
                                        for k in BAT_KEYS},
                   "ce_r": ce0}]
        for s_ in sorted(res["sds"]):
            rd_ = read_state(res["sds"][s_])
            states.append({
                "step": s_,
                "reads": {k: {"mean_p": rd_[k]["mean_p"],
                              "per": [q["p"] for q in rd_[k]["per"]]}
                          for k in BAT_KEYS},
                "ce_r": rd_["ce_r"],
                "g1_wash_row": {kk: next((r[kk] for r in res["traj"]
                                          if r["step"] == s_), None)
                                for kk in ("g_m12_mean_pz", "g0_mean_pz",
                                           "ce_r", "cum_disp")}})
            log(f"  {tag} +{s_:4d}: " + " ".join(
                f"{k} {states[-1]['reads'][k]['mean_p']:.4f}"
                for k in ("fact", "near", "tmpl2", "mis2M", "ctrl"))
                + f" | CE_R {rd_['ce_r']:.4f}")
            jp.write_text(json.dumps({"tag": tag, "lr": lr, "seed": seed,
                                      "states": states}, indent=1),
                          encoding="utf-8")
            write_metrics(f"PARTIAL: arm {tag} through +{s_}")
        arms_out[tag] = {"tag": tag, "lr": lr, "seed": seed,
                         "states": states,
                         "traj_keys": {"device": res.get("device"),
                                       "steps_ran": res["steps_ran"],
                                       "zeph_violations":
                                           res["zeph_violations"]}}
        jp.write_text(json.dumps(arms_out[tag], indent=1), encoding="utf-8")
        metrics["arms"] = {t: {"lr": a["lr"], "seed": a["seed"],
                               "device": a["traj_keys"]["device"],
                               "states": [s["step"] for s in a["states"]],
                               "reads": {str(s["step"]): {
                                   k: s["reads"][k]["mean_p"]
                                   for k in BAT_KEYS}
                                   for s in a["states"]},
                               "ce_r": {str(s["step"]): s["ce_r"]
                                        for s in a["states"]}}
                          for t, a in arms_out.items()}
        write_metrics(f"PARTIAL: arm {tag} complete "
                      f"({len(ARMS) - ai - 1} arms left)")
        del res
        if ai < len(ARMS) - 1:
            log(f"[thermal] inter-arm cooldown {COOLDOWN_S:.0f}s")
            time.sleep(COOLDOWN_S)

    # G_WASH: the lineage arm's +2 tie vs g1b's committed C-arm
    lin = arms_out.get("L10902")
    if lin is not None and not SMOKE:
        s2 = next((s for s in lin["states"] if s["step"] == 2), None)
        G_WASH = {"arm": "L10902 (seed 10902 = the locked stream)",
                  "ref": refC2, "tol_pz": G_WASH_TOL_PZ,
                  "tol_ce": G_WASH_TOL_CE}
        if s2 is not None:
            G_WASH["got_gm12"] = s2["reads"]["fact"]["mean_p"]
            G_WASH["got_ce_r"] = s2["ce_r"]
            G_WASH["dp_gm12"] = abs(G_WASH["got_gm12"] - refC2["gm12"])
            G_WASH["dp_ce"] = abs(G_WASH["got_ce_r"] - refC2["ce_r"])
            G_WASH["pass"] = bool(G_WASH["dp_gm12"] <= G_WASH_TOL_PZ
                                  and G_WASH["dp_ce"] <= G_WASH_TOL_CE)
        metrics["gates"]["G_WASH"] = G_WASH
        log(f"G_WASH: +2 gm12 {G_WASH.get('got_gm12')} vs ref "
            f"{refC2['gm12']:.4f} -> {'PASS' if G_WASH.get('pass') else 'FAIL'}")
        assert G_WASH.get("pass"), f"G_WASH FAILED: {G_WASH}"

    # ---------------- P5 adjudication (frozen)
    def decls(arm: dict) -> dict:
        st = {s["step"]: s for s in arm["states"]}
        base = {k: st[0]["reads"][k]["mean_p"] for k in BAT_KEYS}
        out = {}
        for s_, rec in st.items():
            if s_ == 0:
                continue
            out[s_] = {k: 1 - rec["reads"][k]["mean_p"] / base[k]
                       for k in BAT_KEYS}
        return out, base

    adj = {"bars": REGISTERED_PREDICTION["bars_verbatim"],
           "bar_constants": {"SEP_BAND": SEP_BAND, "ETA2_BAR": ETA2_BAR,
                             "ETA2_NULL": ETA2_NULL,
                             "near_qual": [NEAR_QUAL_LO, NEAR_QUAL_HI],
                             "DRAW_RHO_BAR": DRAW_RHO_BAR},
           "arms": {}}
    qual: list[tuple[str, dict]] = []
    for tag, arm in arms_out.items():
        if SMOKE:
            continue
        d, base = decls(arm)
        s_star = None
        for s_ in sorted(d, reverse=True):
            if d[s_]["near"] < NEAR_QUAL_HI:
                s_star = s_
                break
        rec = {"lr": arm["lr"], "seed": arm["seed"], "declines":
               {str(s_): d[s_] for s_ in d}, "R0": base,
               "s_star": s_star,
               "near_decl_s_star": (d[s_star]["near"]
                                    if s_star is not None else None),
               "qualifies": bool(s_star is not None
                                 and d[s_star]["near"] >= NEAR_QUAL_LO)}
        # O1 per state
        rec["O1_per_state"] = {str(s_): bool(
            d[s_]["ctrl"] < d[s_]["tmpl2"] < d[s_]["near"]
            and d[s_]["ctrl"] <= d[s_]["near"] / SEP_BAND
            and min(d[s_]["ctrl"], d[s_]["tmpl2"], d[s_]["near"])
            > DECL_TRIVIAL) for s_ in d}
        rec["O1_n_states"] = sum(rec["O1_per_state"].values())
        rec["ordering_replicates_arm"] = bool(
            rec["qualifies"] and rec["O1_n_states"] >= ORDER_STATES_MIN)
        # family eta2 at s* (out-of-corpus families)
        if s_star is not None:
            st = {s["step"]: s for s in arm["states"]}
            fam_keys = {"fact": "fact", "near": "near", "tmpl2": "tmpl",
                        "mis2Z": "misZ", "snip0Z": "snipZ"}
            groups: dict[str, list[float]] = {}
            for bk, fam in fam_keys.items():
                r0 = [q["p"] for q in t0_reads[bk]["per"]]
                rs = st[s_star]["reads"][bk]["per"]   # floats
                groups[fam] = [float(rs[i]) / max(r0[i], 1e-9)
                               for i in range(len(rs))]
            e = eta2(groups)
            # full-set co-report (ctrl items pooled as one family)
            ctrl0_ = [q["p"] for q in t0_reads["ctrl"]["per"]]
            ctrls_ = st[s_star]["reads"]["ctrl"]["per"]
            e_full = eta2({**groups, "ctrl": [
                float(ctrls_[i]) / max(ctrl0_[i], 1e-9)
                for i in range(len(ctrls_))]})
            rec["family_eta2_out_of_corpus"] = e
            rec["family_eta2_full_cocorpus"] = e_full
            rec["family_replicates_arm"] = bool(
                rec["qualifies"] and e["eta2"] is not None
                and e["eta2"] >= ETA2_BAR)
        adj["arms"][tag] = rec
        if rec["qualifies"]:
            qual.append((tag, rec))
        log(f"ADJ {tag}: s* {s_star} near_decl "
            f"{rec['near_decl_s_star']} O1@{rec['O1_n_states']} states "
            f"eta2 {rec.get('family_eta2_out_of_corpus', {}).get('eta2')}")

    # cross-arm clauses
    n_qual = len(qual)
    ordering_all = bool(qual) and all(r["ordering_replicates_arm"]
                                      for _, r in qual) and n_qual >= 2
    family_all = bool(qual) and all(r.get("family_replicates_arm")
                                    for _, r in qual) and n_qual >= 2
    channel = None
    for tag, r in qual:
        ch_states = [s_ for s_ in r["declines"]
                     if r["declines"][s_]["tmpl2"] >= r["declines"][s_]["near"]
                     and r["declines"][s_]["tmpl2"] >= r["declines"][s_]["fact"]]
        if len(ch_states) >= ORDER_STATES_MIN:
            channel = tag
            break
    adj["cross_arm"] = {
        "qualifying_arms": [t for t, _ in qual],
        "ORDERING-REPLICATES": ordering_all,
        "FAMILY-REPLICATES": family_all,
        "CHANNEL-DEATH-arm": channel,
    }
    if SMOKE:
        verdict, clause = "SMOKE (nothing adjudicated)", "smoke run"
    elif ordering_all or family_all:
        verdict = "FACULTY-AT-HOME"
        clause = ("the ordering and/or family split replicate at 2.74M — "
                  "the faculty predates scale; rhythm organ = "
                  "faculty-keeper candidate. ORDERING="
                  f"{ordering_all} FAMILY={family_all} on qualifying arms "
                  f"{[t for t, _ in qual]}; the tables verbatim.")
    elif n_qual == 0:
        verdict = "NOT-AT-HOME"
        clause = ("no qualifying arm at any dose — the batteries read flat "
                  "(the instrument/dose floor): the faculty is "
                  "scale-emergent at this instrument class; the organ "
                  "reads as splint. FQ1's honest failure mode (e).")
    else:
        any_O1 = any(r["O1_n_states"] > 0 for _, r in qual)
        any_eta = any((r.get("family_eta2_out_of_corpus") or {})
                      .get("eta2") is not None
                      and r["family_eta2_out_of_corpus"]["eta2"] >= ETA2_NULL
                      for _, r in qual)
        if not any_O1 and not any_eta:
            verdict = "NOT-AT-HOME"
            clause = ("the batteries read flat or unstructured at 2.74M "
                      "where 124M splits — the faculty is scale-emergent; "
                      "the organ reads as splint. No O1 state, no eta2 "
                      f">= {ETA2_NULL} on any qualifying arm.")
        else:
            verdict = "PARTIAL-TRACE"
            clause = ("anything between — mapped honestly: qualifying arms "
                      f"{[t for t, _ in qual]} show O1 states "
                      f"{ {t: r['O1_n_states'] for t, r in qual} } and "
                      "eta2 "
                      f"{ {t: (r.get('family_eta2_out_of_corpus') or {})
                           .get('eta2') for t, r in qual} } "
                      "against the registered bars; the tables verbatim.")
    adj["verdict"] = verdict
    adj["clause"] = clause
    if channel is not None:
        adj["channel_death_clause"] = (
            f"FIRES on arm {channel}: the template battery erodes with the "
            "fastest (>= near AND >= fact at >= 2 states) — the tiny-scale "
            "'fact death' is partly CHANNEL death (FQ1's branch, verbatim "
            "reading).")
    else:
        adj["channel_death_clause"] = (
            "does not fire on any qualifying arm (tmpl2 never erodes "
            "fastest at >= 2 states).")

    # DRAW sub-verdicts (per dose, at each dose's first arm's s*)
    draws = {}
    for dose_lr, tags in ((1e-3, ["L10902", "L22701", "L22702"]),
                          (1e-4, ["G22711", "G22712"])):
        tags = [t for t in tags if t in adj["arms"]]
        if len(tags) < 2 or SMOKE:
            continue
        dvec = {}
        for t in tags:
            r = adj["arms"][t]
            if r["s_star"] is None:
                continue
            dvec[t] = {k: r["declines"][str(r["s_star"])][k]
                       for k in BAT_KEYS
                       if 0.02 < r["declines"][str(r["s_star"])][k] < 0.98}
        keys = (sorted(set.intersection(*[set(v) for v in dvec.values()]))
                if dvec else [])
        rhos = []
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                if t1 in dvec and t2 in dvec and len(keys) >= 3:
                    rhos.append(spearman([dvec[t1][k] for k in keys],
                                         [dvec[t2][k] for k in keys]))
        if rhos:
            draws[f"lr{dose_lr:g}"] = {
                "keys": keys, "rhos": [None if np.isnan(r) else round(r, 4)
                                       for r in rhos],
                "min_rho": (None if all(np.isnan(r) for r in rhos)
                            else float(np.nanmin(rhos))),
                "verdict": ("DRAW-REPLICATES"
                            if (not all(np.isnan(r) for r in rhos)
                                and float(np.nanmin(rhos)) >= DRAW_RHO_BAR)
                            else ("DRAW-DIFFERS"
                                  if all(r < DRAW_RHO_BAR
                                         for r in rhos if not np.isnan(r))
                                  else "GRADED"))}
    adj["draw_verdicts"] = draws
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E227 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  channel clause: {adj['channel_death_clause']}")
    log(f"  draws: {draws}")
    write_metrics("PARTIAL: adjudicated; plots pending")

    # ---------------- P6 plots
    pngs = []
    if not SMOKE:
        comp_tag = next((t for t, r in adj["arms"].items()
                         if r["qualifies"]), None) or \
            next(iter(adj["arms"]))
        arm = arms_out[comp_tag]
        st = arm["states"]
        steps = [s["step"] for s in st]
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))
        cols = {"fact": "tab:red", "near": "tab:green", "tmpl2": "tab:purple",
                "mis2Z": "violet", "mis2M": "tab:orange", "snip0Z": "gray",
                "snip0M": "pink", "ctrl": "tab:blue"}
        ax = axes[0, 0]
        for k in BAT_KEYS:
            r0 = st[0]["reads"][k]["mean_p"]
            ax.plot(steps, [s["reads"][k]["mean_p"] / r0 for s in st],
                    "o-", ms=5, lw=1.8, color=cols[k],
                    label=f"{k} (n={metrics['batteries'][k]['n']})")
        axr = ax.twinx()
        axr.plot(steps, [s["ce_r"] for s in st], "s:", ms=4, lw=1.2,
                 color="seagreen", alpha=0.8)
        axr.set_ylabel("CE_R (dotted, right)", fontsize=8, color="seagreen")
        ax.set_xlabel(f"wash steps (lr {arm['lr']:g}, seed {arm['seed']})")
        ax.set_ylabel("retention R(s)/R(0)")
        ax.set_ylim(-0.05, 1.12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"THE COMPARISON ARM {comp_tag} -> {verdict}",
                     fontsize=10)

        ax = axes[0, 1]
        r80 = ref124["declines_at_80"]
        r50f = ref124["fresh_declines_at_50"]
        ss = adj["arms"][comp_tag]["s_star"] or steps[-1]
        dd = adj["arms"][comp_tag]["declines"].get(str(ss), {})
        labels = ["ctrl", "tmpl", "near"]
        v124 = [r80["ctrl"], r80["tmpl"], r80["near"]]
        v124f = [r50f["ctrl"], r50f["tmpl"], r50f["near"]]
        v227 = [dd.get("ctrl"), dd.get("tmpl2"), dd.get("near")]
        x = range(3)
        w = 0.26
        ax.bar([i - w for i in x], v124, w, color="gray", alpha=0.5,
               label=f"124M phase-1 +80")
        ax.bar([i for i in x], v124f, w, color="tan", alpha=0.6,
               label="124M fresh +50")
        ax.bar([i + w for i in x], v227, w, color="tab:cyan", alpha=0.85,
               label=f"2.74M {comp_tag} +{ss}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("decline")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
        ax.set_title("the graded ordering ctrl < tmpl < near: 124M vs "
                     "2.74M", fontsize=9.5)

        ax = axes[1, 0]
        for tag, r in adj["arms"].items():
            ax.plot([r["declines"][str(s_)]["near"] for s_ in
                     sorted(r["declines"], key=int)],
                    [r["declines"][str(s_)]["tmpl2"] for s_ in
                     sorted(r["declines"], key=int)],
                    "o", ms=6, label=f"{tag} (lr {r['lr']:g})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("decl_near")
        ax.set_ylabel("decl_tmpl2")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_title("tmpl2 vs near declines across the dose ladder "
                     "(above the diagonal = CHANNEL-DEATH territory)",
                     fontsize=9)

        ax = axes[1, 1]
        ax.axis("off")
        y = 0.98
        ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=11, va="top",
                family="monospace", weight="bold", color="darkred")
        y -= 0.05
        for wd in textwrap.wrap(clause, width=96, break_long_words=False):
            ax.text(0.02, y, wd, fontsize=7.2, va="top", family="monospace")
            y -= 0.022
        y -= 0.015
        ax.text(0.02, y, "gates:", fontsize=8.5, va="top",
                family="monospace", weight="bold")
        y -= 0.024
        for gname, gv in metrics["gates"].items():
            pv = gv.get("pass")
            ax.text(0.02, y, f"  {gname:10s} "
                    f"{'PASS' if pv else ('FAIL' if pv is False else 'n/a')}",
                    fontsize=7.4, va="top", family="monospace")
            y -= 0.02
        y -= 0.01
        ax.text(0.02, y, "t=0 few-shot lift (M): "
                f"{metrics['t0_fewshot_lift']['lift']:.4f} "
                f"(2shot {metrics['t0_fewshot_lift']['p_M_2shot']:.4f} vs "
                f"0shot {metrics['t0_fewshot_lift']['p_M_0shot']:.4f})",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.026
        ax.text(0.02, y, "CHANNEL-DEATH clause:", fontsize=7.4, va="top",
                family="monospace", weight="bold")
        y -= 0.024
        for wd in textwrap.wrap(adj["channel_death_clause"], width=96,
                                break_long_words=False):
            ax.text(0.02, y, wd, fontsize=6.8, va="top", family="monospace")
            y -= 0.02

        fig.suptitle("E227 — THE FACULTY AT HOME: the e182c2 batteries "
                     "pointed at the 2.74M consolidated root", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p1 = rd / "battery_curves.png"
        fig.savefig(p1, dpi=130)
        plt.close(fig)
        pngs.append(p1)

        # plot 2: the family battery
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
        ax = axes[0]
        r = adj["arms"][comp_tag]
        fam_keys = {"fact": "fact", "near": "near", "tmpl2": "tmpl",
                    "mis2Z": "misZ", "snip0Z": "snipZ"}
        starm = {s["step"]: s for s in arm["states"]}
        for bk, fam in fam_keys.items():
            hrs = [float(a) / max(q["p"], 1e-9) for a, q in
                   zip(starm[r["s_star"] or steps[-1]]["reads"][bk]["per"],
                       t0_reads[bk]["per"])]
            ax.scatter([fam] * len(hrs), hrs, s=14, alpha=0.55)
            ax.hlines(np.mean(hrs), -0.3 + list(fam_keys.values()).index(fam),
                      0.3 + list(fam_keys.values()).index(fam),
                      lw=2.4, color="black")
        ax.axhline(0.5, color="gray", ls="--", lw=0.9)
        ax.set_ylabel(f"item hold-ratio hr at s*={r['s_star']}")
        ax.set_title("the FAMILY battery (out-of-corpus expression "
                     "surface)", fontsize=9.5)
        ax.grid(alpha=0.25, axis="y")
        e = r.get("family_eta2_out_of_corpus") or {}
        ax.text(0.02, 0.02, f"eta2 = {e.get('eta2')}", transform=ax.transAxes,
                fontsize=9, family="monospace",
                bbox=dict(facecolor="wheat", alpha=0.6))

        ax = axes[1]
        for tag, r2 in adj["arms"].items():
            if r2.get("family_eta2_out_of_corpus", {}).get("eta2") is None:
                continue
            ax.bar(tag, r2["family_eta2_out_of_corpus"]["eta2"],
                   color="tab:cyan" if r2["qualifies"] else "lightgray",
                   alpha=0.85)
        ax.axhline(ETA2_BAR, color="darkred", ls="--", lw=1.2,
                   label=f"the 124M family bar eta2>={ETA2_BAR} "
                         f"(124M point {ref124['family_eta2']:.3f})")
        ax.axhline(ETA2_NULL, color="gray", ls=":", lw=1.0,
                   label=f"unstructured floor {ETA2_NULL}")
        ax.set_ylabel("family eta2 (out-of-corpus)")
        ax.legend(fontsize=8)
        ax.set_title("family-first sorting across arms", fontsize=9.5)
        fig.tight_layout()
        p2 = rd / "family_battery.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        pngs.append(p2)

    metrics["honesty_reflex"] = {
        "n_counts": ("n=1 organism (the committed e131 root); wash draws: "
                     "lineage n=3 (10902 locked + 22701/22702), gentle "
                     "n=2, mid n=1; batteries: fact 60 / near 30 / new-form "
                     "30-probed-gated / ctrl val-carried"),
        "exposure_confound": ("the in-corpus ctrl families are excluded "
                              "from the primary eta2 BECAUSE the wash "
                              "trains that distribution — the 124M "
                              "family-split comparison is honest only on "
                              "the out-of-corpus surface"),
        "form_bundle": ("tmpl2 changes context length (118->72) + exemplars "
                        "+ separators together; snip0 is the length-matched "
                        "control; e182c2's form+direction confound "
                        "precedent"),
        "device_texture": ("probes CPU fp32 everywhere (the lineage "
                           "instrument); washes GPU fp32 (the g1b C-arm "
                           "device); patterns, never bits, across devices; "
                           "G_WASH ties the wash within "
                           f"{G_WASH_TOL_PZ}"),
        "ce_asymmetry": ("124M's wash IMPROVED ppl (71->35); the 2.74M "
                         "lineage wash DAMAGES CE_R (1.64->2.03 at +2) — "
                         "the gentle arm is the role-matched health "
                         "reference; disclosed"),
        "nothing_guaranteed": ("controls holding does not prove the faculty "
                               "special; controls eroding does not prove "
                               "generic; the openness is the point"),
        "zclass_floor": ("the mis2M/snip0M lift is the pure few-shot read; "
                         "if it floors at t=0 the negative is itself the "
                         "scale finding (FQ1 (e)) — reported verbatim"),
    }
    metrics["compute"] = {
        "washes": [f"{t}: lr {lr:g} seed {s} steps {max(ck)}"
                   for t, lr, s, ck in ARMS],
        "instrument": "G1.g1_wash + G1.battery_cell arithmetic per item, "
                      "imported VERBATIM (g1_anchored_ball)",
        "run_log": str(LOG_PATH),
    }
    metrics["trims"] = []
    metrics["deviations"] = deviations
    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    log(f"outputs: {rd / 'metrics.json'}, {pngs}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
