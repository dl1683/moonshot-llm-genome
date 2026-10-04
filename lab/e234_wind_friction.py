"""E234 — THE WIND/FRICTION DECOMPOSITION (W035's registered instrument;
design frozen in scratch/e234_design.md BEFORE this script; 2026-10-04).

THE QUESTION (W035, verbatim from the frozen design): is the wash one
field with two components — a DIRECTED wind (the span its pull lives in:
kills beliefs whose supports it reaches) and a DIFFUSE friction (the
orthogonal residual: grinds everyone's commitments, undirected)?

THE MATERIAL (all committed): the three 124M washes replayed bit-exactly
on the draw streams (e226's certified seeds 18202 / 20261002 / 21703; the
e182c-family replay machinery, module-imported), every step's RAW
pre-clip gradient collected; the 54 probes' t=0 supports (e226's
instrument, module-imported, rebuilt and certified against e226's
committed alignment records); per-probe belief p(t) and margin m(t)
trajectories (e214/e228's committed records at the shared states; the
+40 and w3 states measured HERE in-replay, gated against the committed
where they overlap).

THE INSTRUMENT (frozen in scratch/e234_design.md §THE INSTRUMENT):
  1. THE SPAN BASIS (circularity guard): split-half — the FIRST half of
     each wash's steps (1..40) builds the top-k span basis (k at the scree
     knee, recorded); the SECOND half's steps (41..80) are decomposed
     against it. The MIRROR split (second half builds, first half
     decomposed) co-reported. The leave-one-wash-out flavor co-reported
     (w's second half decomposed against the pooled first-half basis of
     the other two washes).
  2. THE DECOMPOSITION per step: wind-component = ||P_span g||,
     friction-component = ||g - P_span g||; per probe: wind-alignment(t) =
     |cos(P_span g(t), s_probe)| (the projected gradient's alignment with
     the probe's support).
  3. THE TWO JOINS across the 54 probes (both halves reported separately):
     (a) belief decline (1 - p(+80)/p(0)) vs cumulative wind-alignment;
     (b) margin decline (1 - m(+80)/m(0)) vs cumulative friction magnitude
     AND vs cumulative wind-alignment (the independence read), AND vs
     cumulative FULL step magnitude (the WIND-ONLY discriminator).

BARS VERBATIM (scratch/e234_design.md §Bars, frozen by the dispatch
BEFORE any compute; adjudicate against exactly this; no bar shopping):
  - WIND-AND-FRICTION — "join (a) clears rho >= 0.6 (the wind kills
    aligned beliefs, extended from the anchor pair to all 54 probes) AND
    margin-vs-alignment rho <= 0.2 while margin-vs-friction rho >= 0.4 —
    two components, two targets; the field named"
  - WIND-ONLY — "beliefs track alignment but margins track the FULL step
    magnitude (not specifically the residual) — one directed field; the
    friction is not separable"
  - NEITHER — "belief decline does not extend beyond the anchors at the
    line — the e226 seat was anchor-specific, not the battery's law; the
    honest bound"

REGISTERED PREDICTIONS (design §Registered predictions, verbatim; no
retrofit):
  (a) "The anchor asymmetry generalizes: nearrel-family probes (the dying
      family) carry higher wind-alignment than product-family probes,
      median separation >= 2x."
  (b) "Margin decline is alignment-independent (|rho| <= 0.2) — the
      friction is undirected."
  (c) "If WIND-AND-FRICTION fires, the flat phase's fourth currency
      (T208's open ledger) reads as THE WIND'S REACH — and e231's
      root-level overlap join should agree in sign; disagreement between
      them is itself informative (probe-level vs root-level span
      composition)."

THE RIDER (registered in T212, folded into this report): the 7
WRONG-CHOICE zombies of e232's 33 standing zombies (fact/ctrl batteries;
the 2 near co-report points folded with them) — identify their +80 argmax
tokens against the wash-corpus's modal continuation at those positions
(the committed frozen corpus + e232/e228's committed records). H-i
RE-TEACHING predicts >= 5/7 corpus-mode; H-ii DRIFT predicts scattered,
low-probability tokens.

OPERATIONALIZATIONS (frozen here BEFORE compute; they fix the clauses,
they do not move the bars):
  * REPLAY: per wash, CPU fp32, torch threads 8 (e182c's convention —
    w1's archived states are a threads-8 CPU product), AdamW (0.9,0.95)
    wd 0.1 constant lr 5e-5, clip 1.0, batch 8 x ctx 512, the VERBATIM
    e182c replay arithmetic; draw #t = torch.randint(hi, (8,),
    generator=gen) from the wash's archived seed. Step t's gradient g_t =
    the RAW PRE-CLIP gradient of the batch-CE at weights W_{t-1} on draw
    #t (the step's own gradient — the matched point). Certified: the
    generator state after 80 draws == the archived *_latest.pt gen state
    (bit-exact, G_DRAWS); the replayed weights vs the archived state
    checkpoints (drift recorded; w1 CPU-origin expected ~0, w2/w3
    GPU-fp32-origin expected TEXTURE-tier ~1e-6..1e-4, disclosed per the
    design's provenance clause); the in-replay probe reads vs the
    committed e214/e217/e228 records at every shared state (the
    scientific gate, e214's dp <= 0.01 convention).
  * GRADIENT STORE: per step the L2-normalized direction x 2^14 in fp16
    (e226's cache convention; measured instrument floor ~5e-4 per cos),
    on a scratch memmap (NOT runs/; deleted on success, kept on failure);
    the true norm gnorm_t journaled per step (fp64). All Grams and dots
    are computed from these rows with fp64-cast chunks (e226's cache
    math) and reweighted by the journaled norms to true units.
  * SPAN BASIS: top-k eigenvectors of the first-half RAW-gradient Gram
    (norm-weighted — the pull with its magnitude; the normalized-Gram
    flavor co-reported). k = the scree knee: argmax_{1<=i<=20}
    (lambda_i - lambda_{i+1}) of the descending spectrum, capped at 12
    (cap disclosed if hit); spectrum + variance fraction recorded. The
    basis vectors are orthonormalized by 1/sqrt(lambda_j) (Gram
    eigenvectors span an orthogonal, not orthonormal, set) before every
    projection, angle, and decomposition.
  * RECIPE-IDENTITY DISCLOSURE (coordinator advisory at dispatch; e231's
    instrument autopsy): AdamW's first displacement is
    -lr*(sign(g)+wd*W), so a span basis built on EARLY steps carries the
    sign-step's shadow — "reached by the span" can be recipe-pinned near
    1 rather than economical. CO-REPORTED: the top principal component's
    cosine with the MEAN SIGN-DIRECTION of the first-half steps (how
    identity-dominated the basis is). GUARD: a PINNED-INSTRUMENT check on
    join (a) — if the pooled wind_cum spread (max-min)/median < 0.05 the
    x-variable is degenerate (a pinned instrument's signature) and NO bar
    is adjudicated; the verdict says PINNED, the numbers stand (an
    instrument-validity guard registered before compute, not a moved
    bar). The readouts here are per-PROBE alignments (not u0-vs-span),
    so the pinning risk is lower than e231's — disclosed regardless.
  * SPAN-COVERAGE GUARD (coordinator advisory #2 at dispatch; disclosure
    + adjudication clause, bars untouched): co-reported is the fraction
    of each second-half step's L2 inside the first-half basis (mean +
    spread; the smoke read median ~0.07). If the pooled mean coverage
    exceeds 0.8, a WIND-AND-FRICTION outcome is labeled COVERAGE-VACUOUS
    rather than firing on the arithmetic split (the decomposition only
    means something if the basis is genuinely partial); a REDUCED-RANK
    sensitivity (r = floor(k/2), min 1) is co-reported for joins (a) and
    (b) regardless — the wind/friction distinction should survive the
    basis being deliberately partial.
  * JOIN VARIABLES per (probe i, wash w): cumulative over the decomposed
    half's steps t: wind_cum(i) = sum_t |cos(P g_t, s_i)|;
    fric_cum(i) = sum_t ||g_res,t|| * |cos(g_res,t, s_i)| (the residual's
    magnitude reaching the probe — the only per-probe sense "friction
    magnitude" varies; the GLOBAL residual magnitude is wash-level and
    cannot correlate across probes, disclosed); full_cum(i) = sum_t
    ||g_t|| * |cos(g_t, s_i)| (the WIND-ONLY discriminator);
    fric_cos_cum(i) = sum_t |cos(g_res,t, s_i)| co-reported (the
    AdamW-step-normalized flavor — the applied step is clip- then
    Adam-normalized, so the direction-only flavor is the applied-field
    read; both reported, the raw flavor primary per the design).
  * DECLINES: belief_decl = 1 - p(80)/p(0); margin_decl = 1 - m(80)/m(0)
    with m = e228's margin_sigma = (top1-top2 logit)/std(vocab logits) at
    the answer position, measured in-replay at 0/40/80 (all washes) and
    gated against e214/e217 (p) and e228 (margin) at the shared states.
  * JOIN STATISTIC: Spearman rho (pooled primary = 162 probe-wash pairs,
    54 x 3 washes; per-wash co-reported; Pearson co-reported).
    PRIMARY half = split A (first half builds, second half decomposed)
    against the full-wash declines (the design's verbatim quantities).
    Co-reports: second-half-window declines (1 - p(80)/p(40)) vs the
    second-half cumulatives (the strictly matched window), and the mirror
    split B (second half builds, first half decomposed) vs first-half
    declines (1 - p(40)/p(0)).
  * ADJUDICATION (order WIND-AND-FRICTION -> WIND-ONLY -> NEITHER;
    PRIMARY = pooled split A, full-wash declines):
      WIND-AND-FRICTION := rho_a >= 0.6 AND |rho_mw| <= 0.2 AND
                           rho_mf >= 0.4   (rho_mw's bar text "rho <= 0.2"
                           operationalized as |rho| <= 0.2 — independence
                           — matching prediction (b)'s |rho| <= 0.2);
      WIND-ONLY := rho_a >= 0.6 AND rho_mfull >= 0.4 AND rho_mf < 0.4;
      NEITHER := rho_a < 0.6;
      contingency (disclosed, no bar shopping): if rho_a >= 0.6 but no
      margin join clears 0.4, the verdict is WIND-PRESENT-MARGINS-
      UNTRACKED (numbers verbatim; the three bars do not cover it).
  * RIDER: for each wrong-chooser, the +80 argmax token is read from the
    COMMITTED archived +80 states (e182c_s80 / e182c2_fresh_s80 — the
    states e232's census read; the argmax re-verified against e228's
    committed top1_id). The wash-corpus's modal continuation at the
    probe's final position = argmax_v count(final-token -> v) over the
    frozen train_ids stream (bigram PRIMARY; trigram last-2-token context
    co-reported, sparse contexts disclosed). H-i := >= 5/7 argmax ==
    corpus bigram mode; H-ii := scattered (few mode matches) AND the
    argmax tokens' corpus continuation probabilities low.

GATES: G_ENV (CPU-only, threads 8, psutil load checks per phase; no GPU),
G_SIZE (inherited 124M-with-reason via e182c), G_CORPUS (rebuild ==
e182's record), G_BATT (the four batteries rebuilt VERBATIM reproduce
the committed t=0 records; the 54 == e216's assignment), G_SUPPORT (the
rebuilt t=0 supports: e226's FD convention + the direct certification
max|cos(g_w1_draw1, s_i) - e226's committed w1:s0 alignment| <= 0.005),
G_DRAWS (bit-exact gen states at step 80), G_REPLAY (weights drift +
probe dp vs committed at shared states), G_TRAJ (the in-replay +40/w3
extension consistent at every shared state), G_DECOMP (friction^2 >= 0
clamps counted; knee recorded; basis stability co-reported via subspace
principal angles B1-vs-B2 per wash and B1-vs-B1' across washes).

Envelope: CPU-only (the e182c precedent; e233 owns the GPU lane), torch
threads 8, progressive PARTIAL metrics + resumable journal after every
wash, load checks per phase. No NOTES/THINKING/QUEUE/STATE edits
(dispatch). Smoke via E234_SMOKE=1 (12 steps, 11 probes, w1 only;
nothing adjudicated).

PROVENANCE (extend-don't-repeat): builds on e226/T198 (the support bank,
the wash-gradient conventions, the anchor seat), W035 (the registered
synthesis this cell adjudicates), T212/e232 (the zombie taxonomy + the
rider), e214 (the belief trajectories + the re-probe convention), e228
(the margin instrument), e182c/e182c2/e217 (the three-wash archive + the
certified draw streams), T208 (the fourth-currency ledger), e231 (the
root-level overlap, the cross-check). NEW: the split-half span basis and
the wind/friction decomposition of every step of the committed washes;
the two joins across all 54 probes x 3 washes; the +40/w3 trajectory
extension; the 7 wrong-choosers' corpus-mode identification.

Run:  cd lab && python e234_wind_friction.py     (E234_SMOKE=1 for the
      shakedown; nothing adjudicated or gated in smoke)
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import common                                        # noqa: E402
from common import now_iso, run_dir, save_json       # noqa: E402

import e182c_forgetting_control as e1                # noqa: E402 — the replay machinery, VERBATIM
import e182c2_template as e2                         # noqa: E402 — the template battery, VERBATIM
import e226_interior as e226                         # noqa: E402 — the support bank + wash-grad conventions

THREADS = 8                    # e182c's replay convention (w1 = a threads-8 CPU product)
torch.set_num_threads(THREADS)  # AFTER the imports (e182c sets 8; e226 resets 4)

import torch.nn.functional as F                      # noqa: E402
from scipy.stats import spearmanr, pearsonr          # noqa: E402

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import textwrap                                      # noqa: E402

try:
    import psutil                                   # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E234_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e234_smoke" if SMOKE else "e234"

T0 = time.time()
_log_t0 = time.time()
def log(m: str) -> None:
    print(f"[{time.time() - _log_t0:8.1f}s] {m}", flush=True)

# ------------------------------------------------------------ the frozen cell
WASHES = ("w1",) if SMOKE else ("w1", "w2", "w3")
W_SEED = dict(e226.W_SEED)                     # {w1: 18202, w2: 20261002, w3: 21703}
W_ARCH = e226.W_ARCH                           # archived state checkpoints
W_LATEST = e226.W_LATEST                       # archived step-80 gen states
STEPS = 12 if SMOKE else 80
HALF = STEPS // 2                              # split-half boundary (40 full)
KNEE_MAX_I = 5 if SMOKE else 20                # knee search head
K_CAP = 4 if SMOKE else 12                     # span rank cap (disclosed if hit)
READ_STATES = (0, 2, 4, 6, 12) if SMOKE else (0, 2, 10, 40, 50, 80)
W1_READ_STATES = (0, 2, 4, 6, 12) if SMOKE else (0, 2, 10, 40, 50, 80)
WX_READ_STATES = (0, 4, 6, 12) if SMOKE else (0, 10, 40, 50, 80)
CACHE_SCALE = e226.CACHE_SCALE                 # 2**14 (e226's fp16 cache convention)
CHUNK = 2_097_152                              # 2**21 coords per chunk (fp64 transients)
RAM_FLOOR_GB = 14.0                            # below this the supports cache is unsafe
ANCHOR_G, ANCHOR_I = e226.ANCHOR_G, e226.ANCHOR_I
CK = common.REPO / "runs" / "checkpoints"
SCRATCH = common.REPO / "scratch" / "e234_cache"

# committed records (load-only)
E182C_M = common.REPO / "runs" / "e182c" / "metrics.json"
E182C2_M = common.REPO / "runs" / "e182c2" / "metrics.json"
E182C2_J = common.REPO / "runs" / "e182c2" / "journal_p2.json"
E217_M = common.REPO / "runs" / "e217" / "metrics.json"
E214_M = common.REPO / "runs" / "e214" / "metrics.json"
E216_M = common.REPO / "runs" / "e216" / "metrics.json"
E226_M = common.REPO / "runs" / "e226" / "metrics.json"
E228_J = common.REPO / "runs" / "e228" / "journal.json"
E232_M = common.REPO / "runs" / "e232" / "metrics.json"
E231_M = common.REPO / "runs" / "e231" / "metrics.json"

# the rider's 7 wrong-choosers (e232's committed census: standing zombies of
# the fact/ctrl batteries whose +80 argmax is NOT the answer) + the 2 near
# co-report points (T212: "the 7"; the near battery is the recorded co-report)
RIDER_7 = [
    ("fact", "w1", "Egypt->Cairo"),
    ("fact", "w1", "the United States->dollar"),
    ("fact", "w2", "the United States->dollar"),
    ("ctrl", "w1", "The phone made by Apple->iPhone"),
    ("ctrl", "w1", "The tablet made by Apple->iPad"),
    ("ctrl", "w2", "The phone made by Apple->iPhone"),
    ("ctrl", "w2", "The music store made by Apple->iTunes"),
]
RIDER_NEAR2 = [
    ("near", "w1", "Georgia->Atlanta"),
    ("near", "w2", "Georgia->Atlanta"),
]

REGISTERED = {
    "bars_verbatim": {
        "WIND-AND-FRICTION": "join (a) clears rho >= 0.6 (the wind kills "
        "aligned beliefs, extended from the anchor pair to all 54 probes) "
        "AND margin-vs-alignment rho <= 0.2 while margin-vs-friction rho >= "
        "0.4 — two components, two targets; the field named",
        "WIND-ONLY": "beliefs track alignment but margins track the FULL "
        "step magnitude (not specifically the residual) — one directed "
        "field; the friction is not separable",
        "NEITHER": "belief decline does not extend beyond the anchors at "
        "the line — the e226 seat was anchor-specific, not the battery's "
        "law; the honest bound",
    },
    "predictions_verbatim": {
        "(a) anchor asymmetry generalizes": "The anchor asymmetry "
        "generalizes: nearrel-family probes (the dying family) carry "
        "higher wind-alignment than product-family probes, median "
        "separation >= 2x.",
        "(b) margins alignment-independent": "Margin decline is "
        "alignment-independent (|rho| <= 0.2) — the friction is undirected.",
        "(c) e231 cross-check": "If WIND-AND-FRICTION fires, the flat "
        "phase's fourth currency (T208's open ledger) reads as THE WIND'S "
        "REACH — and e231's root-level overlap join should agree in sign; "
        "disagreement between them is itself informative (probe-level vs "
        "root-level span composition).",
        "rider H-i (T212)": ">= 5/7 of the wrong-choosers' +80 argmax "
        "tokens ARE the wash-corpus's modal continuation at those "
        "positions (re-teaching visible in decision space first)",
        "rider H-ii (T212)": "the flips are scattered, low-probability "
        "tokens (noise-level re-orderings among low logits — pure "
        "inertia)",
    },
    "registration": "bars + predictions frozen VERBATIM from "
        "scratch/e234_design.md (frozen at dispatch BEFORE this script or "
        "any compute); adjudicate against exactly this; no bar shopping",
}

trims: list[str] = []
deviations: list[str] = [
    "The per-step gradient store is the L2-NORMALIZED direction x 2^14 in "
    "fp16 with true norms journaled (e226's cache convention; its measured "
    "floor ~5e-4 per cos) — not the raw fp32 vectors (60 GB/wash was "
    "outside the shared box); all Grams/dots reweight to true units by the "
    "journaled norms, so every registered quantity is in true units.",
    "The span basis is built from the raw-gradient Gram (norm-weighted) as "
    "registered; the normalized-Gram flavor is co-reported (spectrum + "
    "knee + its own join (a) rho) — a robustness read, never the primary.",
    "The +40 states (all washes) and the w3 belief/margin trajectories are "
    "measured HERE on the replayed states (the committed journals stop at "
    "{2,10,50,80} for w1/w2 and record no margins for w3); the in-replay "
    "reads are gated against e214/e217 (p) and e228 (margin) at every "
    "shared state (G_TRAJ) — an extension, disclosed as such.",
    "w2/w3 archived weights are GPU-fp32 products; this cell's replays are "
    "CPU fp32 (the design's disclosed convention; T204's floor makes the "
    "cross-device drift irrelevant at these magnitudes — measured and "
    "recorded per checkpoint in G_REPLAY).",
    "The rider's corpus-mode uses the committed ARCHIVED +80 states (the "
    "very states e232's census read), not this cell's replays — the zombie "
    "census stays on its own record; argmax re-verified vs e228's top1_id.",
    "MID-RUN ADVISORIES (disclosed): coordinator advisory #1 (recipe-"
    "identity: sign-step shadow) was registered before any compute; "
    "advisory #2 (span-coverage guard + reduced-rank sensitivity) arrived "
    "while the first full-run replay was ~85% through wash 1 — the run "
    "was STOPPED, the coverage clause registered in the docstring, and "
    "the run relaunched (resuming from the saved step-70 state) BEFORE "
    "any Gram, decomposition, join, or adjudication compute had run; no "
    "bars were moved; the coverage numbers (mean fraction of each "
    "decomposed-half step's L2^2 in the basis) are smoke-anticipated low "
    "(~0.07 median).",
    "RAM-LIGHT LIFECYCLE (mid-run restructure, disclosed): the 13.4 GB "
    "supports cache is freed after its P2 certification and rebuilt "
    "deterministically at P5 (bit-identity certified by G_SUPPORT's "
    "probe-0 recompute + the e226 dp 0.0 match) — the box is shared and "
    "available RAM fell below 1 GB during the first full-run attempt "
    "with the cache held.",
    "scratch/e234_cache/ holds the fp16 gradient memmaps + resume states "
    "(scratch is workspace, NOT runs/ or data/); memmaps deleted only on "
    "DONE, kept on any failure for post-mortem.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E234_SMOKE=1): 12 steps, w1 only, 11 probes; nothing "
    "adjudicated or gated (SMOKE stamp).",
]


# ------------------------------------------------------------ envelope

def cpu_load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


def ram_wait(tag: str, floor_gb: float, max_wait_s: int = 900) -> dict:
    """Shared-box courtesy: wait (bounded) for available RAM before a
    heavy phase; proceed with disclosure if the floor never clears."""
    if psutil is None:
        return {"tag": tag, "waited_s": 0, "note": "psutil absent"}
    t0 = time.time()
    while psutil.virtual_memory().available / 2**30 < floor_gb:
        if time.time() - t0 > max_wait_s:
            log(f"  [ram-wait] {tag}: floor {floor_gb} GB NOT reached in "
                f"{max_wait_s}s — proceeding anyway (disclosed)")
            return {"tag": tag, "waited_s": round(time.time() - t0, 1),
                    "floor_gb": floor_gb, "cleared": False}
        time.sleep(20)
    log(f"  [ram-wait] {tag}: avail "
        f"{psutil.virtual_memory().available / 2**30:.1f} GB >= {floor_gb} GB")
    return {"tag": tag, "waited_s": round(time.time() - t0, 1),
            "floor_gb": floor_gb, "cleared": True}


def build_support_cache(net0, probes54, n_p, n_params) -> torch.Tensor:
    """The 54 t=0 support rows (e226's convention: L2-normalized fp32
    grads x 2^14 in fp16). Deterministic; rebuilt at P5 (RAM-light
    lifecycle: the 13.4 GB cache is NOT held through the replay phase)."""
    cache = torch.empty((n_p, n_params), dtype=torch.float16)
    for i, pr in enumerate(probes54):
        s_i, _p0 = e226.probe_support(net0, pr)
        cache[i] = (s_i * CACHE_SCALE).to(torch.float16)
    return cache


# ------------------------------------------------------------ e228's margin pass

@torch.no_grad()
def margin_pass(net, battery: list[dict]) -> list[dict]:
    """e228's margin_pass VERBATIM (module-import convention): one forward
    per probe — p/rank (e182's path) + margin_sigma = (top1-top2 logit)/
    std(vocab logits), top1_id/top2_id, argmax_is_answer."""
    net.eval()
    rows = []
    for pr in battery:
        lg = net(input_ids=pr["ids"]).logits[0, -1]
        p = F.softmax(lg, -1)
        pv = float(p[pr["ans_id"]])
        rank = int((lg > lg[pr["ans_id"]]).sum().item())
        top2v = torch.topk(lg, 2)
        top1_id, top2_id = int(top2v.indices[0]), int(top2v.indices[1])
        sigma = float(lg.std())
        margin_raw = float(top2v.values[0] - top2v.values[1])
        rows.append({"fact": pr["fact"], "p": pv, "rank": rank,
                     "top1": bool(rank == 0), "top5": bool(rank < 5),
                     "margin_sigma": margin_raw / sigma if sigma > 0 else None,
                     "margin_raw": margin_raw, "sigma": sigma,
                     "top1_id": top1_id, "top2_id": top2_id,
                     "argmax_is_answer": bool(top1_id == pr["ans_id"])})
    return rows


# ------------------------------------------------------------ Gram machinery
# All heavy algebra is CHUNKED fp64 over the fp16 rows (e226's cache math):
# one sweep over coords builds a wash's full Gram (S x S) AND its support
# dots (S x 54) — every later quantity (bases, projections, alignments,
# principal angles, the LOO pooled basis) is closed-form linear algebra on
# those two matrices + the journaled norms. No basis is ever materialized.

def gram_and_dots(rows: np.memmap, s: int, supports: torch.Tensor,
                  n_sup: int) -> tuple[np.ndarray, np.ndarray]:
    """rows: (S, N) fp16 scaled-unit rows; returns (Gram, D) in SCALED
    units: Gram[t,s] = <row_t, row_s> (unit-cos), D[t,i] = <row_t, s_i>
    (s_i rows scaled 2^14) — fp64 accumulation over chunks."""
    G = np.zeros((s, s), dtype=np.float64)
    D = np.zeros((s, n_sup), dtype=np.float64)
    N = rows.shape[1]
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        C = np.asarray(rows[:, a:b], dtype=np.float64)
        G += C @ C.T
        S = supports[:, a:b].to(torch.float64).numpy()
        D += C @ S.T
    return G, D


def cross_gram(rowsA: np.memmap, rowsB: np.memmap) -> np.ndarray:
    """<rowA_t, rowB_s> for all pairs, fp64 chunks (unit-cos units)."""
    sA, sB = rowsA.shape[0], rowsB.shape[0]
    G = np.zeros((sA, sB), dtype=np.float64)
    N = rowsA.shape[1]
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        A = np.asarray(rowsA[:, a:b], dtype=np.float64)
        B = np.asarray(rowsB[:, a:b], dtype=np.float64)
        G += A @ B.T
    return G


def sign_shadow(rows: np.memmap, half: int) -> tuple[np.ndarray, float]:
    """THE RECIPE-IDENTITY CO-REPORT (coordinator advisory): the mean
    SIGN-direction of the FIRST-HALF steps' gradients, and every step's
    dot with it. Returns (dots <g_t, m> for all t in unit-cos units,
    ||m||). Sign of the scaled fp16 rows == sign of the raw grads."""
    N = rows.shape[1]
    dots = np.zeros(rows.shape[0], dtype=np.float64)
    m2 = 0.0
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        Sg = np.sign(np.asarray(rows[:half, a:b], dtype=np.float32))
        m = Sg.mean(axis=0, dtype=np.float64)
        C = np.asarray(rows[:, a:b], dtype=np.float64)
        dots += C @ m
        m2 += float(m @ m)
    return dots, math.sqrt(max(m2, 0.0))


def scree_knee(evals: np.ndarray, max_i: int, cap: int) -> tuple[int, dict]:
    """k at the biggest absolute spectral drop in the head (registered)."""
    ev = np.asarray(evals, dtype=np.float64)
    gaps = [(float(ev[i - 1] - ev[i]), i + 1) for i in
            range(1, min(max_i, len(ev) - 1) + 1)]  # (gap, k=i+1)
    k = max(gaps)[1] if gaps else 1
    capped = False
    if k > cap:
        k, capped = cap, True
    tot = float(ev.sum())
    return k, {"spectrum": ev.tolist(), "gaps": [g for g, _ in gaps],
               "var_frac_at_k": float(ev[:k].sum() / tot) if tot > 0 else None,
               "k_capped": capped}


def principal_angles(cA: np.ndarray, GB_block: np.ndarray,
                     cB: np.ndarray) -> list[float]:
    """cosines of the principal angles between span(basis A) and
    span(basis B): svd(cA @ GB_block @ cB) with cA/cB orthonormal
    coefficient matrices (columns = eigvecs) — pure Gram algebra."""
    M = cA.T @ GB_block @ cB
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, -1.0, 1.0).tolist()


# ------------------------------------------------------------ replay + collect

def drift_vs_archived(net, path) -> dict:
    """max|dW| and max relative drift of the LIVE net vs an archived
    checkpoint (compared AT the checkpoint's own step — see the G_REPLAY
    block)."""
    sdc = torch.load(path, map_location=CPU, weights_only=False)["model"]
    md = mr = 0.0
    with torch.no_grad():
        for (_k, v), (_kb, va) in zip(net.state_dict().items(),
                                      sdc.items()):
            d = (v - va).abs()
            md = max(md, float(d.max().item()))
            mr = max(mr, float((d.norm() / va.norm().clamp(min=1e-12)
                                ).item()))
    del sdc
    return {"max_abs_dw": md, "max_rel_dw": mr}


def replay_and_collect(net0, train_ids, probes, rd, journal,
                       load_checks, metrics, write_metrics, wi: int,
                       w: str) -> dict:
    """One wash: the VERBATIM e182c replay arithmetic on CPU; every step's
    raw pre-clip gradient captured (normalized, fp16, scratch memmap) +
    gnorm/CE/clip journaled; trajectory reads at READ_STATES; certification
    vs the archived checkpoints + committed records."""
    S = STEPS
    N = sum(p.numel() for p in net0.parameters())
    SCRATCH.mkdir(parents=True, exist_ok=True)
    mm_path = SCRATCH / f"grads_w{wi + 1}.npy"
    resume_path = SCRATCH / f"resume_w{wi + 1}.pt"
    wj = journal.setdefault(w, {})
    steps_done = int(wj.get("steps_done", 0))

    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=e1.LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(W_SEED[w])
    hi = train_ids.shape[0] - e1.SEQ - 1

    mm = (np.lib.format.open_memmap(mm_path, mode="r+")
          if steps_done > 0 else
          np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float16,
                                    shape=(S, N)))
    if steps_done > 0 and resume_path.exists():
        rs = torch.load(resume_path, map_location=CPU, weights_only=False)
        rs_step = int(rs["step"])
        if rs_step != steps_done:
            # the resume file is the truth for weights+generator: redo the
            # steps past it (deterministic replay; rows rewritten
            # identically; journal overwritten identically)
            log(f"{w}: resume file at step {rs_step} vs journal "
                f"{steps_done} — rewinding to {rs_step} and redoing "
                f"(deterministic)")
            steps_done = rs_step
        net.load_state_dict(rs["model"])
        opt.load_state_dict(rs["opt"])
        gen.set_state(rs["gen"])
        # the stored rows are unit-scaled: spot-check the last stored row
        row = torch.from_numpy(np.asarray(mm[steps_done - 1],
                                          dtype=np.float32))
        nrm64 = float(row.norm(dtype=torch.float64).item()) / CACHE_SCALE
        assert abs(nrm64 - 1.0) < 0.02, f"stored row off-unit: {nrm64}"
        log(f"{w}: RESUMED at step {steps_done} (draws continue "
            f"bit-identically from the saved generator state)")
    elif steps_done > 0:
        raise RuntimeError(f"{w}: journal says {steps_done} steps but no "
                           f"resume state — restarting wash required")

    recs = wj.setdefault("steps", {})
    traj = wj.setdefault("traj", {})
    t_start = time.time()

    def read_traj(state_label: int):
        load_checks.append(cpu_load_check(f"{w} traj {state_label}"))
        rows = margin_pass(net, probes)
        net.train()                      # dropout is 0; restores e182c's mode
        traj[str(state_label)] = rows
        mp = sum(r["p"] for r in rows) / len(rows)
        mmg = sum(r["margin_sigma"] for r in rows) / len(rows)
        log(f"  {w} @{state_label}: mean_p {mp:.4f} mean_margin {mmg:.3f}")
        journal[w] = wj
        save_journal(rd, journal)

    if steps_done == 0:
        read_traj(0)

    step = steps_done
    while step < S:
        step += 1
        off = torch.randint(hi, (e1.BATCH,), generator=gen)
        x = torch.stack([train_ids[o: o + e1.SEQ] for o in off])
        y = torch.stack([train_ids[o + 1: o + 1 + e1.SEQ] for o in off])
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        ce = float(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # --- capture the RAW pre-clip gradient (the step's own field) ---
        g = e226.flat_grad(net)
        nrm = float(g.norm(dtype=torch.float64).item())
        if not (nrm > 0 and math.isfinite(nrm)):
            raise RuntimeError(f"{w} s{step}: degenerate gradient")
        mm[step - 1] = (g / nrm * CACHE_SCALE).to(torch.float16).numpy()
        del g
        clip_fac = min(1.0, 1.0 / nrm) if nrm > 0 else 1.0
        # NOTE: no zero_grad here — clip_grad_norm_ rescales .grad in place
        # and opt.step() consumes it (zeroing between capture and step
        # would void the update; opt.zero_grad at the loop top is the
        # e182c order).
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        recs[str(step)] = {"ce": ce, "gnorm": nrm, "clip_fac": clip_fac}
        # inline replay-cert: compare the LIVE net at the checkpoint's own
        # step vs the archived state (the final-net-vs-mid-run-checkpoint
        # comparison in the first attempt recorded DISPLACEMENT, not
        # drift — relabeled post-hoc, disclosed)
        if not SMOKE and step in W_ARCH[w]:
            wj.setdefault("replay_cert", {})[str(step)] = \
                drift_vs_archived(net, W_ARCH[w][step])
        if step % 10 == 0:
            log(f"  [{w} replay] s{step:3d}/{S} CE {ce:.4f} |g| {nrm:.2f} "
                f"({time.time() - t_start:.0f}s)")
        reads = W1_READ_STATES if w == "w1" else WX_READ_STATES
        if step in reads and str(step) not in traj:
            read_traj(step)
        if step % 10 == 0 or step == HALF or step == S:
            torch.save({"model": {k: v.detach().clone() for k, v in
                                  net.state_dict().items()},
                        "opt": opt.state_dict(), "gen": gen.get_state(),
                        "step": step}, resume_path)
            wj["steps_done"] = step
            journal[w] = wj
            save_journal(rd, journal)
            write_metrics(f"PARTIAL: {w} replay through s{step}")
    mm.flush()
    wj["steps_done"] = S
    wj["gnorms"] = [recs[str(t)]["gnorm"] for t in range(1, S + 1)]
    wj["ces"] = [recs[str(t)]["ce"] for t in range(1, S + 1)]

    # ---- G_DRAWS: bit-exact generator state vs the archived *_latest ----
    gen_ok = None
    if not SMOKE:
        archived = torch.load(W_LATEST[w], map_location=CPU,
                              weights_only=False)["gen"]
        gen_ok = bool(torch.equal(gen.get_state(), archived))

    # ---- G_REPLAY: the archived-checkpoint drifts were measured inline
    # AT each checkpoint's own step (see the loop); the end-state drift is
    # re-confirmed here (valid whether or not the loop ran this session);
    # intermediate checkpoints not measured this session carry G_TRAJ's
    # probe-level gate instead (disclosed).
    cert = wj.setdefault("replay_cert", {})
    if not SMOKE:
        sfin = max(W_ARCH[w])
        cert[str(sfin)] = drift_vs_archived(net, W_ARCH[w][sfin])
        for s_ in W_ARCH[w]:
            if str(s_) not in cert:
                cert[str(s_)] = {"note": "not measured this session — "
                                 "certified via G_TRAJ's probe-level gate"}
        cert["gen_state_identical_after_80"] = gen_ok
    else:
        cert["gen_state_identical_after_80"] = gen_ok
    wj["replay_cert"] = cert
    journal[w] = wj
    save_journal(rd, journal)
    _cs = {k: (round(v["max_abs_dw"], 8)
               if isinstance(v, dict) and "max_abs_dw" in v
               else str(v)[:40]) for k, v in cert.items()}
    log(f"{w}: replay done ({time.time() - t_start:.0f}s); gen_ok "
        f"{gen_ok}; cert {json.dumps(_cs)}")
    del net, opt
    return {"gen_ok": gen_ok, "cert": cert, "traj": traj}


def save_journal(rd: Path, journal: dict):
    (rd / "journal.json").write_text(
        json.dumps(journal, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------ decomposition

def decompose(G_scaled: np.ndarray, gnorms: np.ndarray,
              D_scaled: np.ndarray, row_norms: np.ndarray,
              half: int) -> dict:
    """The split-half decomposition in TRUE units from the scaled Gram/D.
    Returns basis coefficients, per-step wind/friction, per-(step,probe)
    alignments for the decomposed half."""
    S = G_scaled.shape[0]
    Gn = G_scaled.copy()
    for t in range(S):                      # true-unit dots: unit-rows x norms
        Gn[t] *= gnorms[t]
        Gn[:, t] *= gnorms[t]
    Gn /= float(CACHE_SCALE ** 2)           # rows are 2^14-scaled: /2^28
    # normalized-gram flavor (co-report): D^-1/2 G D^-1/2
    dgn = np.sqrt(np.maximum(np.diag(Gn), 1e-30))
    Gn_norm = Gn / np.outer(dgn, dgn)

    def basis_of(M, tag):
        ev, evec = np.linalg.eigh(M)
        order = np.argsort(ev)[::-1]
        ev, evec = ev[order], evec[:, order]
        k, meta = scree_knee(ev, KNEE_MAX_I, K_CAP)
        meta["tag"] = tag
        # orthonormalize: Gram eigvecs give orthogonal (norm sqrt(lambda))
        # basis vectors — scale column j by 1/sqrt(lambda_j) (clamped)
        c = evec[:, :k] / np.sqrt(np.clip(ev[:k], 1e-12, None))
        return k, c, ev, meta

    kA, cA, evA, metaA = basis_of(Gn[:half, :half], "A(first-half builds)")
    S_A = S - half
    # projections of the DECOMPOSED half against basis A (true units):
    # b_t = cA^T @ Gn[:half, t]
    B_A = cA.T @ Gn[:half, half:]                     # (kA, S_A)
    wind_A = np.linalg.norm(B_A, axis=0)              # ||P g_t||
    gnorm_dec = gnorms[half:]
    fric_A2 = np.clip(gnorm_dec ** 2 - wind_A ** 2, 0.0, None)
    fric_A = np.sqrt(fric_A2)
    n_clamped = int((gnorm_dec ** 2 - wind_A ** 2 < -1e-9 * gnorm_dec ** 2
                     ).sum())
    # M_A = cA^T @ dmat[:half]  (dmat true units: dot(g, cache_row))
    dmat = D_scaled * gnorms[:, None] / float(CACHE_SCALE)  # (S, 54)
    M_A = cA.T @ dmat[:half, :]                       # (kA, 54)
    pg_dot = B_A.T @ M_A                              # (S_A, 54) = <P g_t, s_i>
    cos_wind = np.abs(pg_dot) / (np.outer(wind_A, row_norms)
                                 ).clip(min=1e-30)
    res_dot = dmat[half:, :] - pg_dot                 # <g_res, s_i>
    cos_fric = np.abs(res_dot) / (np.outer(np.maximum(fric_A, 1e-12),
                                           row_norms)).clip(min=1e-30)
    cos_full = np.abs(dmat[half:, :]) / (np.outer(gnorm_dec, row_norms)
                                         ).clip(min=1e-30)
    # mirror split B (co-report): second half builds, first half decomposed
    kB, cB, evB, metaB = basis_of(Gn[half:, half:], "B(mirror)")
    B_B_of_first = cB.T @ Gn[half:, :half]            # (kB, half)
    wind_B = np.linalg.norm(B_B_of_first, axis=0)
    fric_B2 = np.clip(gnorms[:half] ** 2 - wind_B ** 2, 0.0, None)
    M_B = cB.T @ dmat[half:, :]
    pg_dot_B = B_B_of_first.T @ M_B                   # (half, 54)
    cos_wind_B = np.abs(pg_dot_B) / (np.outer(wind_B, row_norms)
                                     ).clip(min=1e-30)
    res_dot_B = dmat[:half, :] - pg_dot_B
    cos_fric_B = np.abs(res_dot_B) / (np.outer(
        np.maximum(np.sqrt(fric_B2), 1e-12), row_norms)).clip(min=1e-30)
    cos_full_B = np.abs(dmat[:half, :]) / (np.outer(gnorms[:half],
                                                    row_norms)).clip(min=1e-30)
    # basis stability: principal angles between span A and span B
    ang_AB = principal_angles(cA, Gn[:half, half:], cB)
    # normalized-flavor basis (co-report)
    kAn, cAn, evAn, metaAn = basis_of(Gn_norm[:half, :half], "A-norm")
    return {
        "kA": kA, "kB": kB, "cA": cA, "cB": cB,
        "metaA": metaA, "metaB": metaB, "metaA_norm": metaAn,
        "wind_A": wind_A, "fric_A": fric_A, "wind_B": wind_B,
        "fric_B": np.sqrt(fric_B2), "n_negative_fric2_clamped": n_clamped,
        "cos_wind_A": cos_wind, "cos_fric_A": cos_fric,
        "cos_full_A": cos_full, "cos_wind_B": cos_wind_B,
        "cos_fric_B": cos_fric_B, "cos_full_B": cos_full_B,
        "angles_AB_cos": ang_AB, "spectrum_A": evA.tolist(),
        "spectrum_B": evB.tolist(), "spectrum_A_norm": evAn.tolist(),
        "dmat": dmat, "Gn": Gn, "Gn_norm": Gn_norm,
    }


# ------------------------------------------------------------ plots

def plot_join_a(rd, rows, verdict_line):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    fam_c = {"cap-cur": "#8e44ad", "lang": "#5d6d7e", "product": "#c0392b",
             "founder-anchor": "#0d5c3f", "near-uscap": "#1a6faf",
             "rev-capital": "#b7950b"}
    for fam, c in fam_c.items():
        xs = [r["wind_cum"] for r in rows if r["family6"] == fam]
        ys = [r["belief_decl"] for r in rows if r["family6"] == fam]
        ax.scatter(xs, ys, s=26, alpha=0.75, color=c, label=fam)
    for r in rows:
        if r["fact"] in (ANCHOR_G, ANCHOR_I):
            ax.annotate("G" if r["fact"] == ANCHOR_G else "i",
                        (r["wind_cum"], r["belief_decl"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=10, weight="bold")
    ax.set_xlabel("cumulative wind-alignment (split A, 2nd half)")
    ax.set_ylabel("belief decline  1 - p(80)/p(0)")
    ax.set_title(f"JOIN (a) — pooled n={len(rows)}  "
                 f"Spearman rho={spearmanr([r['wind_cum'] for r in rows], [r['belief_decl'] for r in rows]).statistic:.3f}",
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    ax = axes[1]
    for wtag, mk in zip(("w1", "w2", "w3"), ("o", "s", "^")):
        if not any(r["wash"] == wtag for r in rows):
            continue
        xs = [r["wind_cum"] for r in rows if r["wash"] == wtag]
        ys = [r["belief_decl"] for r in rows if r["wash"] == wtag]
        rho = spearmanr(xs, ys).statistic
        ax.scatter(xs, ys, s=22, alpha=0.7, marker=mk,
                   label=f"{wtag} rho={rho:.2f}")
    ax.set_xlabel("cumulative wind-alignment")
    ax.set_ylabel("belief decline")
    ax.set_title("per-wash (replication read)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("E234 — JOIN (a): belief decline vs the WIND's cumulative "
                 "alignment\n" + textwrap.fill(verdict_line, 110), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = rd / f"{NAME}_join_a.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_join_b(rd, rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    panels = [("fric_cum", "cumulative FRICTION magnitude (residual "
               "reaching the probe)", "friction"),
              ("wind_cum", "cumulative WIND alignment", "wind"),
              ("full_cum", "cumulative FULL step magnitude", "full")]
    for ax, (xk, xlab, tag) in zip(axes, panels):
        xs = [r[xk] for r in rows]
        ys = [r["margin_decl"] for r in rows]
        rho = spearmanr(xs, ys).statistic
        rp = pearsonr(xs, ys).statistic
        for fam in sorted({r["family6"] for r in rows}):
            fx = [r[xk] for r in rows if r["family6"] == fam]
            fy = [r["margin_decl"] for r in rows if r["family6"] == fam]
            ax.scatter(fx, fy, s=22, alpha=0.7,
                       label=f"{fam} ({spearmanr(fx, fy).statistic:+.2f})")
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("margin decline  1 - m(80)/m(0)")
        ax.set_title(f"margins vs {tag}: rho={rho:.3f} (Pearson {rp:.3f})",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.4)
    fig.suptitle("E234 — JOIN (b): margin decline vs friction (the grind) / "
                 "wind (the independence read) / full magnitude "
                 "(the WIND-ONLY discriminator) — pooled n="
                 f"{len(rows)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = rd / f"{NAME}_join_b.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_basis(rd, dec, washes, cross_angles, loo):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    for w in washes:
        ev = np.asarray(dec[w]["metaA"]["spectrum"], dtype=float)
        ax.plot(range(1, len(ev) + 1), ev / max(ev[0], 1e-30), "o-",
                ms=4, label=f"{w} A (k={dec[w]['kA']})")
        evn = np.asarray(dec[w]["spectrum_A_norm"], dtype=float)
        ax.plot(range(1, len(evn) + 1), evn / max(evn[0], 1e-30), ":",
                alpha=0.6, label=f"{w} A-norm")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("lambda_i / lambda_1 (scree)")
    pcs = ", ".join(f"{w} {dec[w].get('cos_pc1_sign_shadow', float('nan')):+.2f}"
                    for w in washes)
    ax.set_title("THE SPAN'S SCREE (raw + norm flavors)\n"
                 f"cos(PC1, mean-sign): {pcs}", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax = axes[1]
    for w in washes:
        ac = dec[w]["angles_AB_cos"]
        ax.plot(range(1, len(ac) + 1), np.arccos(np.clip(ac, -1, 1))
                * 180 / math.pi, "s-", label=f"{w}: angle(B1,B2)")
    ax.set_xlabel("principal angle index")
    ax.set_ylabel("angle (deg)")
    ax.set_title("BASIS STABILITY: split A vs split B subspaces", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[2]
    if cross_angles:
        labs = list(cross_angles)
        vals = [np.degrees(np.arccos(np.clip(cross_angles[l][0], -1, 1)))
                for l in labs]
        ax.bar(labs, vals, color="#1a6faf", alpha=0.8)
        ax.set_ylabel("first principal angle (deg)")
        ax.set_title("CROSS-WASH: B1 vs B1' (wind span identity)", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    if loo:
        axt = ax.twinx()
        axt.plot(range(len(loo["wind_share"])), loo["wind_share"], "o--",
                 color="#c0392b", label="wind share under LOO basis")
        axt.set_ylabel("median wind share (LOO)", color="#c0392b", fontsize=8)
        axt.legend(fontsize=7)
    fig.suptitle("E234 — THE BASIS: scree knees, split-half stability "
                 "(principal angles), cross-wash identity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = rd / f"{NAME}_basis.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_anchor(rd, dec, rows, probes54, P_IDX):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    band = [i for i, r in enumerate(probes54)
            if r["family6"] == "product"
            and r["fact"] not in (ANCHOR_G, ANCHOR_I)]
    for ax, w in zip(axes[:3], ("w1", "w2", "w3")):
        if w not in dec:
            continue
        cw = dec[w]["cos_wind_A"]                     # (S_A, 54)
        xs = list(range(HALF + 1, HALF + 1 + cw.shape[0]))
        gi, ii = P_IDX[ANCHOR_G], P_IDX[ANCHOR_I]
        bm = np.mean(cw[:, band], axis=1)
        bs = np.std(cw[:, band], axis=1, ddof=1) if len(band) > 1 else 0
        ax.fill_between(xs, bm - 2 * bs, bm + 2 * bs, color="0.82",
                        label="product family ±2σ (n=5)")
        ax.plot(xs, cw[:, gi], "o-", color="#1a6faf", ms=4, label="Gmail")
        ax.plot(xs, cw[:, ii], "s--", color="#c0392b", ms=4, label="iPhone")
        ax.set_xlabel("wash step (decomposed half)")
        ax.set_ylabel("|cos(P_span g_t, s_probe)|")
        ax.set_title(f"{w}: the anchors' wind-alignment per step", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
    fig.suptitle("E234 — THE ANCHOR OVERLAY: e226's seat read through the "
                 "span projection (the dying anchor rides the wind; the "
                 "holding anchor sits in the family band)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = rd / f"{NAME}_anchor.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E234 — THE WIND/FRICTION DECOMPOSITION (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e234_wind_friction",
        "phase": "the three committed 124M washes replayed (CPU fp32, "
                 "certified draw streams) with every step's raw gradient "
                 "collected; split-half span decomposition; the two joins "
                 "across the 54 probes; the 7-zombie rider",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered_prediction": REGISTERED,
        "question": ("is the wash one field with two components — a "
                     "DIRECTED wind (kills beliefs whose supports its span "
                     "reaches) and a DIFFUSE friction (grinds everyone's "
                     "margins, undirected)?"),
        "builds_on": [
            "scratch/e234_design.md (the frozen design; W035's registered "
            "instrument)",
            "W035 (the wind/friction synthesis this cell adjudicates)",
            "T198 / e226 (the 54-probe support bank, the wash-gradient "
            "conventions, the anchor seat)",
            "T212 / e232 (the zombie taxonomy; the 7 wrong-choosers; the "
            "rider's registration)",
            "e214 (belief trajectories; the re-probe convention), e228 "
            "(the margin instrument), e182c/e182c2/e217 (the three-wash "
            "archive + certified draw streams)",
            "T208 (the fourth-currency ledger), e231 (the root-level "
            "overlap — prediction (c)'s cross-check)",
        ],
        "whats_new": [
            "the split-half SPAN BASIS of the wash (scree knee, stability "
            "via subspace principal angles, cross-wash identity, LOO "
            "flavor)",
            "every step's raw gradient decomposed into WIND (span "
            "component) and FRICTION (orthogonal residual) on all three "
            "committed washes",
            "the two joins across all 54 probes x 3 washes: belief "
            "decline vs cumulative wind-alignment; margin decline vs "
            "cumulative friction magnitude AND wind-alignment AND full "
            "magnitude",
            "the +40 / w3 trajectory extension of the committed journals "
            "(gated where they overlap)",
            "the 7 wrong-choice zombies' +80 argmaxes identified against "
            "the wash-corpus's modal continuations (H-i/H-ii)",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); starting fresh")
            journal = {}

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks

    # ------------------------------------------------ P0 the committed records
    paths = [E182C_M, E182C2_M, E182C2_J, E217_M, E214_M, E216_M, E226_M,
             E228_J, E232_M]
    for p in paths:
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    p1m = json.loads(E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(E182C2_M.read_text(encoding="utf-8"))
    j2 = json.loads(E182C2_J.read_text(encoding="utf-8"))
    u3m = json.loads(E217_M.read_text(encoding="utf-8"))
    e214 = json.loads(E214_M.read_text(encoding="utf-8"))
    e216 = json.loads(E216_M.read_text(encoding="utf-8"))
    e226m = json.loads(E226_M.read_text(encoding="utf-8"))
    e228j = json.loads(E228_J.read_text(encoding="utf-8"))
    e232m = json.loads(E232_M.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}
    w1_tmpl_rec = {s["step"]: s for s in c2m["part1_template"]["states"]}
    w2_rec = {s["step"]: s for s in j2["states"]}
    w3_rec = {s["step"]: s for s in u3m["wash3_states"]}
    e228_st = {(s["wash"], s["step"]): s for s in e228j["states"]}
    e216_tab = e216["residual"]["table"]
    e216_rows = {r["fact"]: r for r in e216_tab}
    assert len(e216_tab) == 54
    metrics["provenance_records"] = {
        "w1": str(E182C_M), "w1_tmpl": str(E182C2_M), "w2": str(E182C2_J),
        "w3": str(E217_M), "belief_traj": str(E214_M),
        "families": str(E216_M), "supports+alignments": str(E226_M),
        "margins": str(E228_J), "zombies": str(E232_M),
    }

    # ------------------------------------------- P1 organism + corpus + batteries
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": THREADS}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    metrics["size_gate"] = G_SIZE

    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    banned = sorted({s.lower() for rel in e1.POOLS for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_str["banned"]
    cand, _ = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {"tokens_after": [corpus_stats["tokens_after"],
                                 e_corp["tokens_after"]],
                "train_tokens": [corpus_stats["train_tokens"],
                                 e_corp["train_tokens"]],
                "banned_list_identical": True}
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'}")
    assert G_CORPUS["pass"] or SMOKE
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_CORPUS": G_CORPUS}
    write_metrics("PARTIAL: organism + corpus certified")

    ccand, _ = e1.build_control_candidates(tok, filtered.lower(), train_ids,
                                           e_str["banned"], e1.CTRL_POOLS,
                                           e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]
    ncand, _ = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_str["banned"],
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]
    tcand, _ = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                        e_str["banned"])
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]
    bats = {"fact": battery, "ctrl": cbattery, "near": nbattery,
            "tmpl": tbattery}

    def _probes(state_rec, batt):
        return {f: v["p"] for f, v in state_rec[batt]["probes"].items()}
    G_BATT = {}
    for b, bl in bats.items():
        mine_p = {r["fact"]: r["p"] for r in bl}
        ref1 = _probes(w1_rec[0], b) if b != "tmpl" \
            else _probes(w1_tmpl_rec[0], "tmpl")
        G_BATT[b] = {"n": len(bl),
                     "set_equal_committed": bool(set(mine_p) == set(ref1)),
                     "max_dp_w1": max(abs(mine_p[f] - v)
                                      for f, v in ref1.items())}
    G_BATT["pass"] = bool(all(G_BATT[b]["set_equal_committed"]
                              and G_BATT[b]["max_dp_w1"] <= 0.010
                              for b in bats)) or SMOKE
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL") + " | "
        + " | ".join(f"{b} n {G_BATT[b]['n']} dp "
                     f"{G_BATT[b]['max_dp_w1']:.2e}" for b in bats))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted")
        return 1

    probes54: list[dict] = []
    for b in ("fact", "ctrl", "near", "tmpl"):
        for r in bats[b]:
            probes54.append({**r, "battery": b,
                             "family6": e216_rows[r["fact"]]["family6"]})
    assert len(probes54) == 54
    if SMOKE:
        keep = {ANCHOR_G, ANCHOR_I,
                "The gaming console made by Microsoft->Xbox",
                "The web browser made by Google->Chrome",
                "The tablet made by Apple->iPad",
                "The music store made by Apple->iTunes",
                "The game console made by Sony->PlayStation",
                "The social network founded by Mark Zuckerberg->Facebook",
                "France->Paris", "Massachusetts->Boston",
                "Georgia->Atlanta"}
        probes54 = [r for r in probes54 if r["fact"] in keep]
        log(f"SMOKE: probe subset n={len(probes54)}")
    n_p = len(probes54)
    P_IDX = {r["fact"]: i for i, r in enumerate(probes54)}
    metrics["probes"] = [{"i": i, "fact": r["fact"], "battery": r["battery"],
                          "family6": r["family6"]} for i, r in
                         enumerate(probes54)]
    N_PARAMS = sum(p.numel() for p in net0.parameters())

    # ------------------------------------------------ P2 the t=0 supports cache
    load_checks.append(cpu_load_check("supports t=0"))
    if psutil is not None and \
            psutil.virtual_memory().available / 2**30 < RAM_FLOOR_GB:
        log("FATAL: RAM below floor for the supports cache")
        write_metrics("PARTIAL: RAM floor hit — HALT")
        return 1
    cache = torch.empty((n_p, N_PARAMS), dtype=torch.float16)
    fd_records = []
    netFD = copy.deepcopy(net0)
    base_sds = [p.detach().clone() for p in netFD.parameters()]
    for i, pr in enumerate(probes54):
        s_i, p0 = e226.probe_support(net0, pr)
        cache[i] = (s_i * CACHE_SCALE).to(torch.float16)
        fd = {}
        for eps in (0.02, 0.05):
            e226.add_flat(netFD, base_sds, s_i, eps)
            fd[str(eps)] = e226.probe_p(netFD, pr) - p0
            e226.restore_flat(netFD, base_sds)
        fd_records.append({"fact": pr["fact"], "p0": p0, "fd_dp": fd,
                           "fd_pass": bool(fd["0.02"] > 0)})
    del netFD, base_sds
    row_norms_t = e226.cache_row_norms(cache)
    row_norms = row_norms_t.tolist()
    del row_norms_t
    s_rep, _ = e226.probe_support(net0, probes54[0])
    d_rep = e226.dot_vec_rows(s_rep, cache, [0])[0] / row_norms[0]
    del s_rep

    # the DIRECT certification vs e226's committed support bank: recompute
    # the w1 draw-1 wash gradient at the pristine weights (the matched
    # point e226 read as w1:s0) and compare every probe's cos against
    # e226's committed alignment record — certifies the supports AND the
    # replay's gradient pipeline in one shot.
    e226_align0 = e226m["curves"]["alignment"]["w1"]["0"]
    gen1 = torch.Generator().manual_seed(W_SEED["w1"])
    hi1 = train_ids.shape[0] - e1.SEQ - 1
    off1 = torch.randint(hi1, (e1.BATCH,), generator=gen1)
    x1, y1 = e226.wash_batch(train_ids, off1)
    g1_hat, _ce1, _n1 = e226.wash_grad(net0, x1, y1)
    dots1 = e226.dot_vec_rows(g1_hat, cache, list(range(n_p)))
    cos1 = {probes54[i]["fact"]: dots1[i] / row_norms[i]
            for i in range(n_p)}
    dp_226 = max(abs(cos1[f] - e226_align0[f]) for f in cos1)
    del g1_hat, dots1, cos1
    G_SUPPORT = {"fd_pass_n": sum(1 for r in fd_records if r["fd_pass"]),
                 "n": n_p, "determinism_selfcos_probe0": d_rep,
                 "max_dp_vs_e226_committed_w1s0_alignment": dp_226,
                 "tol_dp_226": 0.005}
    G_SUPPORT["pass"] = bool(G_SUPPORT["fd_pass_n"] == n_p
                             and d_rep > 0.999
                             and dp_226 <= 0.005) if not SMOKE else True
    journal["supports_fd"] = fd_records
    save_journal(rd, journal)
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: fd {G_SUPPORT['fd_pass_n']}/{n_p}; probe-0 self-cos "
        f"{d_rep:.6f}; max dp vs e226 w1:s0 alignment {dp_226:.5f} -> "
        f"{'PASS' if G_SUPPORT['pass'] else 'FAIL'}")
    if not G_SUPPORT["pass"]:
        write_metrics("PARTIAL: G_SUPPORT FAILED — halted before replays")
        return 1
    write_metrics("PARTIAL: t=0 supports cached + certified vs e226")
    # RAM-LIGHT LIFECYCLE: the 13.4 GB cache is NOT held through the
    # ~1 h replay phase (the box is shared; it rebuilds deterministically
    # at P5 — bit-identity proven by the G_SUPPORT certification above)
    del cache
    import gc
    gc.collect()
    log("supports cache FREED for the replay phase (rebuilt at P5)")

    # ------------------------------------------------ P3 the replays
    replay_out = {}
    for wi, w in enumerate(WASHES):
        load_checks.append(cpu_load_check(f"replay {w}"))
        replay_out[w] = replay_and_collect(
            net0, train_ids, probes54, rd, journal, load_checks,
            metrics, write_metrics, wi, w)
        write_metrics(f"PARTIAL: replay {w} done "
                      f"(gen_ok {replay_out[w]['gen_ok']})")
    G_DRAWS = {"per_wash": {w: replay_out[w]["gen_ok"] for w in WASHES},
               "pass": bool(all(replay_out[w]["gen_ok"] is not False
                                for w in WASHES)) or SMOKE}
    metrics["gates"]["G_DRAWS"] = G_DRAWS
    log("G_DRAWS: " + ("PASS" if G_DRAWS["pass"] else "FAIL") + " | "
        + ", ".join(f"{w}: {replay_out[w]['gen_ok']}" for w in WASHES))

    # ------------------------------------- P4 G_TRAJ (vs committed p + margins)
    traj_rows_ok, traj_max_dp, margin_max_dm = [], 0.0, 0.0
    for w in WASHES:
        traj = journal[w]["traj"]
        recs = {"w1": (w1_rec, w1_tmpl_rec), "w2": (w2_rec, None),
                "w3": (w3_rec, None)}[w]
        for s_lab, rows in traj.items():
            s_ = int(s_lab)
            mine = {r["fact"]: r["p"] for r in rows}
            for b in ("fact", "ctrl", "near", "tmpl"):
                if w == "w1" and b == "tmpl":
                    ref = _probes(w1_tmpl_rec[s_], "tmpl") \
                        if s_ in w1_tmpl_rec else None
                else:
                    ref = _probes(recs[0][s_], b) if s_ in recs[0] else None
                if ref is None:
                    continue
                for f, v in ref.items():
                    if f in mine:
                        traj_max_dp = max(traj_max_dp, abs(mine[f] - v))
            mrec = e228_st.get((w, s_)) if s_ > 0 else e228_st.get(("t0", 0))
            if mrec is not None:
                mm = {r["fact"]: r["margin_sigma"]
                      for b2 in ("fact", "ctrl", "near", "tmpl")
                      for r in mrec[b2]["probes"]}
                for r in rows:
                    if r["fact"] in mm and r["margin_sigma"] is not None \
                            and mm[r["fact"]] is not None:
                        margin_max_dm = max(
                            margin_max_dm,
                            abs(r["margin_sigma"] - mm[r["fact"]]))
    G_TRAJ = {"max_dp_vs_committed_p": traj_max_dp,
              "tol_dp": 0.010,
              "max_dm_vs_e228_margins": margin_max_dm,
              "tol_dm": 0.05,
              "pass": bool(traj_max_dp <= 0.010 and margin_max_dm <= 0.05)
              or SMOKE}
    metrics["gates"]["G_TRAJ"] = G_TRAJ
    metrics["gates"]["G_REPLAY"] = {
        "per_wash_cert": {w: journal[w]["replay_cert"] for w in WASHES},
        "note": "weights drift vs the archived checkpoints (w1 CPU-origin "
                "~0 expected; w2/w3 GPU-fp32-origin TEXTURE-tier, "
                "disclosed); the scientific gate is G_TRAJ's probe-level "
                "agreement with the committed records",
        "pass": G_TRAJ["pass"],
    }
    log(f"G_TRAJ: {'PASS' if G_TRAJ['pass'] else 'FAIL'} "
        f"(max dp {traj_max_dp:.2e}; max dm {margin_max_dm:.3f})")
    write_metrics("PARTIAL: replays certified (G_DRAWS/G_REPLAY/G_TRAJ)")

    # ------------------------------------------------ P5 the sweeps (Grams + D)
    metrics["ram_waits"] = metrics.get("ram_waits", [])
    metrics["ram_waits"].append(ram_wait("P5 cache rebuild", 20.0))
    load_checks.append(cpu_load_check("gram sweeps"))
    if psutil is not None and \
            psutil.virtual_memory().available / 2**30 < RAM_FLOOR_GB:
        log("FATAL: RAM below floor for the P5 cache rebuild")
        write_metrics("PARTIAL: RAM floor hit at P5 — HALT (resumable)")
        return 1
    cache = build_support_cache(net0, probes54, n_p, N_PARAMS)
    _rn2 = e226.cache_row_norms(cache).tolist()
    _dp_rn = max(abs(a - b) for a, b in zip(_rn2, row_norms))
    del _rn2
    log(f"P5 support cache rebuilt (max row-norm drift vs P2: "
        f"{_dp_rn:.2e})")
    grams, dmats, shadows = {}, {}, {}, {}
    for wi, w in enumerate(WASHES):
        mm = np.lib.format.open_memmap(SCRATCH / f"grads_w{wi + 1}.npy",
                                       mode="r", dtype=np.float16)
        G_scaled, D_scaled = gram_and_dots(mm, STEPS, cache, n_p)
        grams[w] = G_scaled
        dmats[w] = D_scaled
        sd, smn = sign_shadow(mm, HALF)
        shadows[w] = {"dots_m": sd.tolist(), "m_norm": smn}
        journal[f"gram_scaled_{w}"] = G_scaled.tolist()
        journal[f"dmat_scaled_{w}"] = D_scaled.tolist()
        journal[f"sign_shadow_{w}"] = shadows[w]
        save_journal(rd, journal)
        log(f"  {w}: Gram (max off-diag cos "
            f"{np.max(G_scaled - np.diag(np.diag(G_scaled))) / float(CACHE_SCALE ** 2):.3f}) + "
            f"support dots + sign-shadow done")
        del mm
    cross = {}
    if len(WASHES) == 3:
        # the D sweeps are done: free the 13.4 GB cache before the pairs
        del cache
        gc.collect()
        metrics["ram_waits"].append(ram_wait("cross grams", 6.0))
        log("supports cache FREED again (cross grams need no supports)")
        for a, b in (("w1", "w2"), ("w1", "w3"), ("w2", "w3")):
            ma = np.lib.format.open_memmap(
                SCRATCH / f"grads_w{WASHES.index(a) + 1}.npy", mode="r",
                dtype=np.float16)
            mb = np.lib.format.open_memmap(
                SCRATCH / f"grads_w{WASHES.index(b) + 1}.npy", mode="r",
                dtype=np.float16)
            cross[f"{a}|{b}"] = cross_gram(ma, mb)
            journal[f"cross_{a}_{b}"] = cross[f"{a}|{b}"].tolist()
            del ma, mb
        save_journal(rd, journal)
    write_metrics("PARTIAL: Gram sweeps done")

    # ------------------------------------------------ P6 the decompositions
    dec = {}
    for w in WASHES:
        gnorms = np.array(journal[w]["gnorms"], dtype=np.float64)
        dec[w] = decompose(grams[w], gnorms, dmats[w], row_norms, HALF)
        # the recipe-identity co-report: cos(top PC of the span, the
        # first-half mean SIGN-direction). dots_m[t] = 2^14 <g_hat_t, m>;
        # true <g_t, m> = gnorm_t * dots_m[t] / 2^14; Q_1 unit-norm.
        sd_true = np.array(shadows[w]["dots_m"]) * gnorms / float(CACHE_SCALE)
        smn = shadows[w]["m_norm"]
        cos_pc1_sign = float(dec[w]["cA"][:, 0] @ sd_true[:HALF] / smn) \
            if smn > 0 else None
        dec[w]["cos_pc1_sign_shadow"] = cos_pc1_sign
        # SPAN-COVERAGE GUARD (coordinator advisory #2): the fraction of
        # each second-half step's L2^2 inside the first-half basis, and
        # the REDUCED-RANK sensitivity (r = floor(k/2), min 1)
        cov_t = (dec[w]["wind_A"] / gnorms[HALF:]) ** 2
        dec[w]["coverage_A"] = {
            "mean": float(np.mean(cov_t)), "median": float(np.median(cov_t)),
            "min": float(np.min(cov_t)), "max": float(np.max(cov_t)),
            "definition": "fraction of each decomposed-half step's squared "
                          "L2 inside the first-half basis (wind^2/gnorm^2)"}
        kA_red = max(1, dec[w]["kA"] // 2)
        cAr = dec[w]["cA"][:, :kA_red]
        B_Ar = cAr.T @ dec[w]["Gn"][:HALF, HALF:]
        wind_r = np.linalg.norm(B_Ar, axis=0)
        fric_r = np.sqrt(np.clip(gnorms[HALF:] ** 2 - wind_r ** 2, 0, None))
        M_Ar = cAr.T @ dec[w]["dmat"][:HALF, :]
        pg_r = B_Ar.T @ M_Ar
        dec[w].update({
            "kA_red": kA_red, "wind_red": wind_r, "fric_red": fric_r,
            "cos_wind_red": np.abs(pg_r) / np.outer(
                np.maximum(wind_r, 1e-12), row_norms).clip(min=1e-30),
            "cos_fric_red": np.abs(dec[w]["dmat"][HALF:, :] - pg_r)
            / np.outer(np.maximum(fric_r, 1e-12),
                       row_norms).clip(min=1e-30)})
        log(f"  {w}: kA={dec[w]['kA']} (var "
            f"{dec[w]['metaA']['var_frac_at_k']:.3f}) kB="
            f"{dec[w]['kB']}; coverage(mean/max) "
            f"{dec[w]['coverage_A']['mean']:.3f}/"
            f"{dec[w]['coverage_A']['max']:.3f}; angle(B1,B2) cos "
            f"{[round(a, 3) for a in dec[w]['angles_AB_cos'][:4]]}; "
            f"cos(PC1, mean-sign) {cos_pc1_sign:+.3f}; "
            f"wind share median "
            f"{np.median((dec[w]['wind_A'] / gnorms[HALF:])**2):.3f}")
    # cross-wash basis identity + LOO flavor (n=3 only). cross blocks are
    # 2^28-scaled unit-cos: reweight by the outer norms and /2^28.
    SCALE2 = float(CACHE_SCALE ** 2)
    cross_angles, loo = {}, None
    if len(WASHES) == 3:
        for a, b in (("w1", "w2"), ("w1", "w3"), ("w2", "w3")):
            ga = np.array(journal[a]["gnorms"])[:HALF]
            gb = np.array(journal[b]["gnorms"])[:HALF]
            X = cross[f"{a}|{b}"][:HALF, :HALF] * np.outer(ga, gb) / SCALE2
            cross_angles[f"{a}vs{b}"] = principal_angles(
                dec[a]["cA"], X, dec[b]["cA"])
        loo_rows, loo_windcum = [], {}
        for w in WASHES:
            others = [u for u in WASHES if u != w]
            ga = np.array(journal[others[0]]["gnorms"])[:HALF]
            gb = np.array(journal[others[1]]["gnorms"])[:HALF]
            gw = np.array(journal[w]["gnorms"])
            Block = np.zeros((2 * HALF, 2 * HALF))
            Block[:HALF, :HALF] = dec[others[0]]["Gn"][:HALF, :HALF]
            Block[HALF:, HALF:] = dec[others[1]]["Gn"][:HALF, :HALF]
            key_ab = f"{others[0]}|{others[1]}" if f"{others[0]}|{others[1]}" \
                in cross else f"{others[1]}|{others[0]}"
            XAB = cross[key_ab][:HALF, :HALF] * np.outer(ga, gb) / SCALE2
            if key_ab.startswith(others[1]):
                XAB = XAB.T
            Block[:HALF, HALF:] = XAB
            Block[HALF:, :HALF] = XAB.T
            ev, evec = np.linalg.eigh(Block)
            order = np.argsort(ev)[::-1]
            ev, evec = ev[order], evec[:, order]
            kL, _meta = scree_knee(ev, KNEE_MAX_I, K_CAP)
            CL = evec[:, :kL] / np.sqrt(np.clip(ev[:kL], 1e-12, None))
            def block_other(o, g):
                key = f"{min(o, w)}|{max(o, w)}"
                Xo = (cross[key][:HALF, :] if key.startswith(o)
                      else cross[key][:, :HALF].T) * np.outer(g, gw) / SCALE2
                return Xo                                   # (HALF, S)
            XA = block_other(others[0], ga)
            XB = block_other(others[1], gb)
            BL = CL.T @ np.vstack([XA, XB])                  # (kL, S)
            wind_L = np.linalg.norm(BL[:, HALF:], axis=0)
            share = float(np.median((wind_L / gw[HALF:]) ** 2))
            # per-probe wind_cum under the LOO basis (vs split A's);
            # dmats are 2^28-scaled: convert to <g, cache_row> first
            dm_o = [dmats[o][:HALF, :]
                    * np.array(journal[o]["gnorms"])[:HALF, None]
                    / float(CACHE_SCALE) for o in others]
            M_L = CL.T @ np.vstack(dm_o)
            pg_L = BL[:, HALF:].T @ M_L                      # (S_A, 54)
            cos_L = np.abs(pg_L) / np.outer(
                np.maximum(wind_L, 1e-12), row_norms).clip(min=1e-30)
            loo_windcum[w] = np.sum(cos_L, axis=0)
            loo_rows.append({"wash": w, "kL": kL,
                             "median_wind_share": share})
        all_A = np.concatenate([np.sum(dec[w]["cos_wind_A"], axis=0)
                                for w in WASHES])
        all_L = np.concatenate([loo_windcum[w] for w in WASHES])
        rho_loo = float(spearmanr(all_A, all_L).statistic)
        loo = {"wind_share": [r["median_wind_share"] for r in loo_rows],
               "rows": loo_rows, "rho_windcum_loo_vs_splitA": rho_loo,
               "wind_cum_loo": {w: loo_windcum[w].tolist()
                                for w in WASHES}}

    # ------------------------------------------------ P7 the joins
    join_rows = []
    for w in WASHES:
        traj = journal[w]["traj"]
        t0 = {r["fact"]: r for r in traj["0"]}
        t80 = {r["fact"]: r for r in traj[str(STEPS)]}
        tH = {r["fact"]: r for r in traj.get(str(HALF), [])} \
            if str(HALF) in traj else None
        cw = dec[w]["cos_wind_A"]; cf = dec[w]["cos_fric_A"]
        cful = dec[w]["cos_full_A"]; fr = dec[w]["fric_A"]
        gnorms_w = np.array(journal[w]["gnorms"])
        for i, pr in enumerate(probes54):
            f = pr["fact"]
            wind_cum = float(np.sum(cw[:, i]))
            fric_cum = float(np.sum(fr * cf[:, i]))
            full_cum = float(np.sum(gnorms_w[HALF:] * cful[:, i]))
            fric_cos_cum = float(np.sum(cf[:, i]))
            belief_decl = 1.0 - t80[f]["p"] / t0[f]["p"]
            m0, m8 = t0[f]["margin_sigma"], t80[f]["margin_sigma"]
            margin_decl = (1.0 - m8 / m0
                           if (m0 is not None and m8 is not None
                               and m0 > 0) else None)
            row = {"wash": w, "fact": f, "battery": pr["battery"],
                   "family6": pr["family6"],
                   "wind_cum": wind_cum, "fric_cum": fric_cum,
                   "full_cum": full_cum, "fric_cos_cum": fric_cos_cum,
                   "belief_decl": belief_decl, "margin_decl": margin_decl}
            if tH is not None and tH.get(f) is not None \
                    and tH[f]["p"] > 0:
                row["belief_decl_2nd_half"] = \
                    1.0 - t80[f]["p"] / tH[f]["p"]
                row["belief_decl_1st_half"] = \
                    1.0 - tH[f]["p"] / t0[f]["p"]
                if tH[f]["margin_sigma"] is not None \
                        and t0[f]["margin_sigma"] is not None \
                        and tH[f]["margin_sigma"] > 0:
                    row["margin_decl_1st_half"] = \
                        1.0 - tH[f]["margin_sigma"] / t0[f]["margin_sigma"]
            # mirror split B co-report
            row["wind_cum_mirror"] = float(
                np.sum(dec[w]["cos_wind_B"][:, i]))
            row["fric_cum_mirror"] = float(
                np.sum(dec[w]["fric_B"] * dec[w]["cos_fric_B"][:, i]))
            # reduced-rank sensitivity (coordinator advisory #2)
            row["wind_cum_red"] = float(
                np.sum(dec[w]["cos_wind_red"][:, i]))
            row["fric_cum_red"] = float(
                np.sum(dec[w]["fric_red"] * dec[w]["cos_fric_red"][:, i]))
            join_rows.append(row)
    journal["join_rows"] = join_rows
    save_journal(rd, journal)

    def col(rows, key):
        return [r[key] for r in rows if r.get(key) is not None]

    pooled = join_rows
    rho_a = float(spearmanr(col(pooled, "wind_cum"),
                            col(pooled, "belief_decl")).statistic)
    # the margin joins pair-filter (margin_decl can be None where m(0)<=0)
    jn = [r for r in pooled if r.get("margin_decl") is not None]
    rho_mw = float(spearmanr([r["wind_cum"] for r in jn],
                             [r["margin_decl"] for r in jn]).statistic)
    rho_mf = float(spearmanr([r["fric_cum"] for r in jn],
                             [r["margin_decl"] for r in jn]).statistic)
    rho_mfull = float(spearmanr([r["full_cum"] for r in jn],
                                [r["margin_decl"] for r in jn]).statistic)
    rho_mf_cos = float(spearmanr([r["fric_cos_cum"] for r in jn],
                                 [r["margin_decl"] for r in jn]).statistic)
    per_wash = {}
    for w in WASHES:
        rw = [r for r in pooled if r["wash"] == w]
        mw = [r for r in rw if r.get("margin_decl") is not None]
        per_wash[w] = {
            "rho_a": float(spearmanr(col(rw, "wind_cum"),
                                     col(rw, "belief_decl")).statistic),
            "rho_mw": float(spearmanr([r["wind_cum"] for r in mw],
                                      [r["margin_decl"] for r in mw]
                                      ).statistic) if len(mw) > 2 else None,
            "rho_mf": float(spearmanr([r["fric_cum"] for r in mw],
                                      [r["margin_decl"] for r in mw]
                                      ).statistic) if len(mw) > 2 else None,
            "rho_mfull": float(spearmanr([r["full_cum"] for r in mw],
                                         [r["margin_decl"] for r in mw]
                                         ).statistic) if len(mw) > 2
            else None}
    # co-reports: matched window + mirror split
    co = {}
    if all(r.get("belief_decl_2nd_half") is not None for r in pooled):
        co["rho_a_matched_window"] = float(spearmanr(
            col(pooled, "wind_cum"),
            col(pooled, "belief_decl_2nd_half")).statistic)
    if all(r.get("belief_decl_1st_half") is not None for r in pooled):
        co["rho_a_mirror"] = float(spearmanr(
            col(pooled, "wind_cum_mirror"),
            col(pooled, "belief_decl_1st_half")).statistic)
    if all(r.get("margin_decl_1st_half") is not None for r in pooled):
        co["rho_mf_mirror"] = float(spearmanr(
            col(pooled, "fric_cum_mirror"),
            col(pooled, "margin_decl_1st_half")).statistic)
        co["rho_mw_mirror"] = float(spearmanr(
            col(pooled, "wind_cum_mirror"),
            col(pooled, "margin_decl_1st_half")).statistic)
    # reduced-rank sensitivity (coordinator advisory #2): the wind/friction
    # distinction must survive the basis being deliberately partial
    co["rho_a_reduced_rank"] = float(spearmanr(
        col(pooled, "wind_cum_red"), col(pooled, "belief_decl")).statistic)
    if jn:
        co["rho_mf_reduced_rank"] = float(spearmanr(
            [r["fric_cum_red"] for r in jn],
            [r["margin_decl"] for r in jn]).statistic)
        co["rho_mw_reduced_rank"] = float(spearmanr(
            [r["wind_cum_red"] for r in jn],
            [r["margin_decl"] for r in jn]).statistic)
    co["rho_a_norm_basis"] = None

    # -------- the PINNED-INSTRUMENT guard (registered, coordinator
    # advisory): if join (a)'s x-variable is near-degenerate across the
    # probes, no bar is adjudicated (a pinned instrument's signature)
    wc = np.array(col(pooled, "wind_cum"))
    wc_spread = float((wc.max() - wc.min()) / wc.mean()) if wc.mean() > 0 \
        else None
    pinned = bool(wc_spread is not None and wc_spread < 0.05)
    cov_mean = float(np.mean([dec[w]["coverage_A"]["mean"]
                              for w in WASHES]))
    cov_max = float(np.max([dec[w]["coverage_A"]["max"]
                            for w in WASHES]))
    coverage_vacuous = bool(cov_mean > 0.8)

    # -------- the adjudication (bars verbatim, pooled split A primary)
    wa = (rho_a >= 0.6)
    if pinned:
        verdict = ("PINNED-INSTRUMENT (join (a)'s wind-alignment spread "
                   f"(max-min)/median = {wc_spread:.4f} < 0.05 — the span "
                   "reaches everyone equally; recipe-identity suspect per "
                   "e231's class; NO bar adjudicated; numbers verbatim)")
    elif wa and abs(rho_mw) <= 0.2 and rho_mf >= 0.4:
        verdict = ("WIND-AND-FRICTION (COVERAGE-VACUOUS — mean span "
                   f"coverage {cov_mean:.3f} > 0.8: the basis is not "
                   "genuinely partial, the split is arithmetic not "
                   "physics; disclosed per coordinator advisory #2; NOT "
                   "a clean fire)" if coverage_vacuous else
                   "WIND-AND-FRICTION")
    elif wa and rho_mfull >= 0.4 and rho_mf < 0.4:
        verdict = "WIND-ONLY"
    elif not wa:
        verdict = "NEITHER"
    else:
        verdict = "WIND-PRESENT-MARGINS-UNTRACKED (no bar; disclosed " \
                  "contingency — numbers verbatim)"
    if SMOKE:
        verdict += " (SMOKE — not adjudicated)"

    # -------- registered predictions
    near_i = [r for r in pooled if r["family6"] == "near-uscap"]
    prod_i = [r for r in pooled if r["family6"] == "product"]
    med_near = float(np.median([r["wind_cum"] for r in near_i])) if near_i \
        else None
    med_prod = float(np.median([r["wind_cum"] for r in prod_i])) if prod_i \
        else None
    pred = {
        "(a) nearrel >= 2x product median wind-alignment": {
            "median_nearrel": med_near, "median_product": med_prod,
            "ratio": (med_near / med_prod
                      if med_near is not None and med_prod else None),
            "clears_2x": bool(med_near is not None and med_prod
                              and med_near / med_prod >= 2.0)},
        "(b) margin alignment-independent |rho| <= 0.2": {
            "rho_mw": rho_mw, "clears": bool(abs(rho_mw) <= 0.2)},
        "(c) e231 cross-check": {"status": "read at compute (below)"},
    }
    if E231_M.exists():
        try:
            e231m = json.loads(E231_M.read_text(encoding="utf-8"))
            e231v = (e231m.get("adjudication") or {}).get("verdict")
            pred["(c) e231 cross-check"] = {
                "e231_status": e231m.get("status"),
                "e231_verdict": e231v,
                "note": "prediction (c) binds only if WIND-AND-FRICTION "
                        "fires: sign agreement expected; disagreement "
                        "(e.g. a probe-level fire against e231's "
                        f"'{e231v}' root-level null) is itself "
                        "informative — probe-level vs root-level span "
                        "composition",
            }
        except Exception:                                   # noqa: BLE001
            pass

    # ------------------------------------------------ P8 the rider
    rider = {"targets": [], "h_i_count_bigram": 0, "h_i_count_trigram": 0}
    if not SMOKE:
        e228_80 = {w: {r["fact"]: r for b2 in
                       ("fact", "ctrl", "near", "tmpl")
                       for r in e228_st[(w, 80)][b2]["probes"]}
                   for w in ("w1", "w2")}
        nets80 = {}
        for w in ("w1", "w2"):
            sd = torch.load(W_ARCH[w][80], map_location=CPU,
                            weights_only=False)["model"]
            nw = copy.deepcopy(net0)
            nw.load_state_dict(sd)
            nets80[w] = nw
            del sd
        probe_by_fact = {r["fact"]: r for r in probes54}
        for batt, w, f in RIDER_7 + RIDER_NEAR2:
            pr = probe_by_fact[f]
            c2 = int(pr["ids"][0, -1].item())
            c1 = int(pr["ids"][0, -2].item())
            # committed argmax (e228's top1_id at +80) vs re-read
            lg = nets80[w](input_ids=pr["ids"]).logits[0, -1]
            am = int(torch.argmax(lg).item())
            p_am = float(F.softmax(lg, -1)[am].item())
            com = e228_80[w][f]
            # corpus bigram continuation at the final position
            pos = (train_ids[:-1] == c2).nonzero().reshape(-1)
            nxt = train_ids[pos + 1]
            cnt = torch.bincount(nxt, minlength=50257)
            mode_bi = int(torch.argmax(cnt).item())
            n_ctx = int(pos.numel())
            # corpus trigram continuation (last-2-token context)
            m2 = (train_ids[:-2] == c1) & (train_ids[1:-1] == c2)
            nxt2 = train_ids[2:][m2]
            cnt2 = torch.bincount(nxt2, minlength=50257) if nxt2.numel() \
                else None
            mode_tri = int(torch.argmax(cnt2).item()) if cnt2 is not None \
                else None
            n_ctx2 = int(nxt2.numel())
            row = {"battery": batt, "wash": w, "fact": f,
                   "argmax_80_reread": am,
                   "argmax_80_committed": com["top1_id"],
                   "argmax_token": tok.decode([am]),
                   "bigram_mode_token": tok.decode([mode_bi]),
                   "is_bigram_mode": bool(am == mode_bi),
                   "p_model_argmax": p_am,
                   "corpus_bigram_prob_of_argmax":
                       float(cnt[am].item() / max(n_ctx, 1)),
                   "n_bigram_ctx": n_ctx,
                   "trigram_mode_token": tok.decode([mode_tri])
                   if mode_tri is not None else None,
                   "is_trigram_mode": bool(am == mode_tri)
                   if mode_tri is not None else None,
                   "n_trigram_ctx": n_ctx2,
                   "margin_80": com["margin_sigma"],
                   "p_ans_80": com["p"]}
            rider["targets"].append(row)
            rider["h_i_count_bigram"] += int(row["is_bigram_mode"])
            rider["h_i_count_trigram"] += int(bool(row["is_trigram_mode"]))
        del nets80
        rider["h_i_verdict"] = ("H-i RE-TEACHING" if
                                rider["h_i_count_bigram"] >= 5
                                else "H-ii DRIFT-leaning" if
                                rider["h_i_count_bigram"] <= 2 else
                                "MIXED (no hypothesis bar; numbers "
                                "verbatim)")
        rider["note"] = ("primary = bigram at the prompt's final token "
                         "(registered); trigram co-reported; the 2 near "
                         "co-report points counted separately from the 7")
        log(f"RIDER: bigram-mode {rider['h_i_count_bigram']}/7 "
            f"(trigram {rider['h_i_count_trigram']}/7)")

    # ------------------------------------------------ the outputs
    metrics["decomposition"] = {
        w: {"kA": dec[w]["kA"], "kB": dec[w]["kB"],
            "kA_reduced": dec[w]["kA_red"],
            "metaA": dec[w]["metaA"], "metaB": dec[w]["metaB"],
            "coverage_A": dec[w]["coverage_A"],
            "cos_pc1_mean_sign_direction":
                dec[w].get("cos_pc1_sign_shadow"),
            "wind_share_median_A": float(np.median(
                (dec[w]["wind_A"] / np.array(
                    journal[w]["gnorms"])[HALF:]) ** 2)),
            "fric_share_median_A": float(np.median(
                (dec[w]["fric_A"] / np.array(
                    journal[w]["gnorms"])[HALF:]) ** 2)),
            "angles_AB_cos": dec[w]["angles_AB_cos"],
            "n_negative_fric2_clamped": dec[w]["n_negative_fric2_clamped"],
            "spectrum_A_norm": dec[w]["spectrum_A_norm"]}
        for w in WASHES}
    metrics["cross_wash"] = {"basis_angles_cos": cross_angles, "loo": loo}
    metrics["joins"] = {
        "primary_pooled": {"rho_a": rho_a, "rho_mw": rho_mw,
                           "rho_mf": rho_mf, "rho_mfull": rho_mfull,
                           "rho_mf_cosflavor": rho_mf_cos, "n": len(pooled)},
        "per_wash": per_wash, "co_reports": co,
        "n_rows_with_margin": len(col(pooled, "margin_decl")),
        "wind_cum_spread_rel": wc_spread, "pinned_instrument": pinned,
        "span_coverage_mean": cov_mean, "span_coverage_max": cov_max,
        "coverage_vacuous": coverage_vacuous,
    }
    metrics["predictions"] = pred
    metrics["rider"] = rider
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "booleans": {"join_a_rho_ge_0.6": wa,
                     "margin_vs_wind_abs_le_0.2": bool(abs(rho_mw) <= 0.2),
                     "margin_vs_fric_ge_0.4": bool(rho_mf >= 0.4),
                     "margin_vs_full_ge_0.4": bool(rho_mfull >= 0.4)},
        "gated_on": "G_ENV/G_SIZE/G_CORPUS/G_BATT/G_SUPPORT/G_DRAWS/"
                    "G_REPLAY/G_TRAJ/G_DECOMP",
        "all_gates_pass": None,
    }
    G_DECOMP = {"knees": {w: {"kA": dec[w]["kA"], "kB": dec[w]["kB"],
                              "kA_reduced": dec[w]["kA_red"]}
                          for w in WASHES},
                "coverage": {w: dec[w]["coverage_A"] for w in WASHES},
                "clamped_negative_fric2":
                    {w: dec[w]["n_negative_fric2_clamped"] for w in WASHES},
                "pass": True}
    metrics["gates"]["G_DECOMP"] = G_DECOMP
    metrics["adjudication"]["all_gates_pass"] = bool(all(
        v.get("pass", True) for v in metrics["gates"].values()
        if isinstance(v, dict)))

    png_a = plot_join_a(rd, pooled, f"{verdict} — rho(a) {rho_a:.3f}; "
                       f"rho(m|wind) {rho_mw:.3f}; rho(m|fric) "
                       f"{rho_mf:.3f}; rho(m|full) {rho_mfull:.3f}")
    png_b = plot_join_b(rd, pooled)
    png_basis = plot_basis(rd, dec, WASHES, cross_angles, loo)
    png_anchor = plot_anchor(rd, dec, pooled, probes54, P_IDX)
    metrics["plot_outputs"] = [str(png_a), str(png_b), str(png_basis),
                               str(png_anchor)]

    metrics["honesty_reflex"] = {
        "n": "3 washes x 54 probes (pooled n=162 primary); ONE organism "
             "(GPT-2 124M) — instance-relative scope",
        "sign_step_shadow": "RECIPE-IDENTITY (coordinator advisory; e231's "
                            "instrument autopsy): AdamW's first "
                            "displacement is -lr*(sign(g)+wd*W), so early "
                            "steps are sign-ray-like and a span basis "
                            "built on them can carry the sign-step's "
                            "shadow — 'reached by the span' could be "
                            "recipe-pinned rather than economical. "
                            "Co-reported: cos(top PC, first-half mean "
                            "sign-direction) per wash; guarded by the "
                            "PINNED-INSTRUMENT check on join (a)'s "
                            "x-spread (verdict says PINNED if "
                            "(max-min)/median < 0.05; readouts here are "
                            "per-PROBE alignments, the lower-risk form)",
        "span_coverage": "COVERAGE (coordinator advisory #2): the mean "
                         "fraction of each decomposed-half step's squared "
                         "L2 inside the first-half basis is co-reported "
                         "per wash (joins.span_coverage_mean/max); if the "
                         "pooled mean exceeds 0.8 a WIND-AND-FRICTION "
                         "outcome is labeled COVERAGE-VACUOUS rather than "
                         "firing on the arithmetic split, and the "
                         "reduced-rank (r = floor(k/2)) sensitivity is "
                         "co-reported for joins (a) and (b) regardless "
                         "(joins.co_reports.rho_*_reduced_rank)",
        "first_order": "the decomposition is first-order (gradients at "
                       "states); the applied AdamW step is clip- then "
                       "Adam-normalized, so the RAW magnitudes weight the "
                       "cumulatives (registered) while the cos-only flavor "
                       "is co-reported as the applied-field read",
        "circularity": "the split-half guard is the registration's: the "
                       "basis never sees the steps it decomposes (mirror "
                       "split + LOO flavor + principal angles co-reported)",
        "supports_rotation": "supports are t=0-fixed (e226's rotation "
                             "sub-bar: supports rotate through the wash — "
                             "disclosed)",
        "correlational": "the joins are correlational across probes; the "
                         "causal read belongs to e233's intervention "
                         "(co-running)",
        "fric_definition": "the per-probe friction magnitude is the "
                           "residual's magnitude ALONG the probe's support "
                           "(the only per-probe sense it varies; the global "
                           "residual norm is wash-level, cannot correlate "
                           "across probes — disclosed)",
        "margin_coupling": "margin correlates with p at t=0 (+0.80, e228's "
                           "own honesty block) — margin_decl conditions on "
                           "m(0) > 0 and probes with m(0) <= 0 are dropped "
                           "from join (b) (count reported)",
    }
    metrics["compute"] = {"wall_s": round(time.time() - T0, 1),
                          "device": "CPU only", "threads": THREADS,
                          "backwards": f"3 washes x {STEPS} steps + "
                          f"{n_p} supports",
                          "load_checks": len(load_checks)}
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    status = ("SMOKE DONE" if SMOKE else
              ("DONE" if metrics["adjudication"]["all_gates_pass"]
               else "DONE (gate failures disclosed — see gates)"))
    write_metrics(status)
    log(f"VERDICT: {verdict}")
    log(f"  join (a) rho {rho_a:.3f} | margins: wind {rho_mw:.3f} "
        f"fric {rho_mf:.3f} full {rho_mfull:.3f}")
    log(f"outputs: {rd / 'metrics.json'} + 4 PNGs; total "
        f"{time.time() - T0:.1f}s")

    if not SMOKE and status.startswith("DONE"):
        for wi in range(len(WASHES)):
            for f in (SCRATCH / f"grads_w{wi + 1}.npy",
                      SCRATCH / f"resume_w{wi + 1}.pt"):
                if f.exists():
                    f.unlink()
        log("scratch gradient memmaps + resume states deleted (DONE)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
