"""G11 — THE CRUSH MECHANISM (T194's named question).

WHY. T194 closed the wall-saga's grid and named the one open mechanism
question: what does the jitter-replay consolidation leave in the weights
that costs the fact one extra projected step at the first wash gradient
— the wash/root draws don't pay it? THE +1 LEDGER (committed): at the
locked e131 cons-10901 root, g1b's W1 reads 0.9452 at +1; the two fresh
cons draws CRUSH — g1e (10912) 0.2719, g1f (10913) 0.2577 — while the
wash/root family reads 0.82-0.96 (g1bR 0.9623/0.9099, g1c 0.8214). The
first step's RAW AdamW displacement is 1.6543 L2 at ALL roots (lr*
sqrt(P) arithmetic, g1's own wall fuzz); the wall truncates it to R=0.7
(rescale ratio ~0.423) and the landing read splits 0.95 vs 0.26. The
C-arm co-read says the raw step kills the cons roots' fact outright
(+1 g-12 light: g1e 0.0022, g1f 0.0003 vs locked 0.6780) — the crush
is real per unit displacement, not a projection artifact.

BUILDS ON (directive 1): g1e/g1f (the two cons roots 10912/10913 and
their +1 crushes, committed; the wash machinery + the resume ckpts
carrying the COMMITTED first-step delta vectors), g1b (the locked
root's W1 +1 0.9452; the reference wall leg), e204 (THE
fact-sensitivity convention: s = the unit gradient of mean log p(Z)
over the g-12 install-60 battery at the state, FD-sign-gated,
matched-point; e194/e195's fact_grad arithmetic), e194/e195 (the
gradient/sign conventions and the dual-estimator matched-point lesson
T150), g1/g1b (the wall arithmetic: commit(R=0.7), the forward
projection, the settle+disarm eval twin). WHAT IS NEW: the first wash
gradient and the fact-sensitivity direction AT THE THREE ROOTS, their
alignments, the ball geometry of the first projected step, and the
coordinate composition of the first delta against each root's own
fact-carrier set — never measured at any 2.74M root.

THE CELL (eval-only; one CPU-rebuilt AdamW step per root — no training
semantics consumed beyond the committed first step itself):
  THREE ROOTS, provenance-gated: the LOCKED e131 cons-10901
  (e131_consolidated_e113.pt), g1e's cons-10912 (g1e_root.pt), g1f's
  cons-10913 (g1f_root.pt). The seed-10902 wash stream is drawn
  VERBATIM (the e170 neutral bank + the first batch, md5-gated vs the
  g1e/g1f resume ckpts' own x_hashes).
  (1) THE FIRST WASH STEP: g_0 = the first wash gradient at each root
      (the stream's actual step-driving gradient; cosine is
      clip-invariant — clip scales by a positive constant — the
      pre-clip norm co-reported); s_0 = the root's OWN fact
      sensitivity (the e204 convention); the PRIMARY read
      cos(g_0, -s_0) (the dispatch's formula; -s_0 = the
      fact-ERASING ray under the critic's convention). BOTH
      orientations ride in the table.
  (2) THE BALL GEOMETRY: theta_anchor = the root (commit(R=0.7)
      VERBATIM); the first raw delta (1.6543) vs R; the rescale ratio
      R/|delta|; the projected landing point theta_land; the landing
      read (the settled +1 g-12, the in-run ARMED-twin convention)
      gated vs the COMMITTED +1 values (0.9452 / 0.2719 / 0.2577); the
      same alignment AFTER projection (cos(delta_proj, +-s_0) —
      colinear with the raw delta, disclosed).
  (3) THE COORDINATE COMPOSITION: the first delta's top-|delta|
      coordinates vs each root's OWN fact-carrier set (top-|s_0| — the
      e204 sensitivity top-k as the CARRIER PROXY, disclosed): overlap
      at k=2000 primary (ladder 500/1000/5000/10000 co-reported), the
      L1 mass fraction on carriers, per-tensor mass tables, and the
      differing-weights co-read (overlap with top-|root_cons -
      root_locked|).

REGISTERED BARS (frozen here, before compute; the dispatch letter
VERBATIM; no bar shopping):
  ERASE-ALIGNED: "fires if the cons roots' first steps align more with
      their fact-erasing directions than the locked root's (cos
      difference >= 0.05 consistently) — the crush is DIRECTIONAL: the
      jitter-replay leaves the fact-support oriented toward the first
      wash gradient; named."
  CARRIER-CONCENTRATED: "fires if the cons roots' first deltas
      concentrate more on the fact-carrier coordinates (top-k overlap
      >= 1.5x the locked root's) — the crush is SPATIAL: the
      consolidation packed the fact where the first wash step lands;
      named."
  PROJECTION-NEUTRAL: "fires if the alignments and concentrations
      match across roots — the crush lives in the nonlinear
      interaction (not readable from first-order geometry); the honest
      limit; the nonlinear cell named."
  GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * cos(g_0, -s_0) is the PRIMARY alignment read (the letter's literal
    formula; g_0 = the post-clip first wash gradient at the root, the
    stream's own arithmetic; s_0 = the unit fact-sensitivity at the
    SAME root — matched-point, T150). Co-reports (never adjudicated):
    cos(g_0, +s_0) (the exact negation), cos(delta_raw, +-s_0) (what
    actually displaces; colinear with delta_proj since the projection
    rescales by a positive constant).
  * ERASE-ALIGNED fires iff BOTH cons roots' primary read exceeds the
    locked root's by >= 0.05 ("consistently" = the two fresh draws
    agree; a single-draw excess is a partial -> GRADED).
  * CARRIER-CONCENTRATED fires iff BOTH cons roots' overlap(k=2000)
    >= 1.5x the locked root's overlap(k=2000); the overlap =
    |top2000(|delta|) n top2000(|s_0|)| / 2000; the ratio's
    denominator floored at ONE coordinate (the random-collision
    baseline k/P ~ 0.0007 makes overlaps of exactly 0 improbable;
    disclosed). The k-ladder co-reported, never adjudicated.
  * PROJECTION-NEUTRAL fires iff NEITHER of the above fires AND the
    reads MATCH across roots: |cos_cons - cos_locked| < 0.05 at BOTH
    cons roots AND both overlap ratios strictly inside (1/1.5, 1.5).
  * GRADED fires otherwise (any partial: one-draw firing, anti-firing,
    mixed signatures, or gate texture).
  * composite order frozen: a full ERASE-ALIGNED and/or a full
    CARRIER-CONCENTRATED firing names the mechanism(s) (both may fire;
    the crush can be directional AND spatial); else PROJECTION-NEUTRAL
    if its match-clause holds; else GRADED. Hard-gate failure (a root
    provenance gate, the FD sign check, the first-step gate, or the
    landing gate) => the record completes with verdict TEXTURE (gate
    failure), nothing adjudicated.

PRE-DISPATCH CHECKS (Rule 12): the three roots' provenance gates
(load + dial vs the committed root cells + body md5 + meta; the W1
resume ckpts' anch__ buffers == the root for g1e/g1f — the anchor IS
the root); the stream gate (the first batch md5 vs the g1e/g1f resume
ckpts' x_hashes[1], with e185's stored hash as the family co-report);
the first-step gates (rebuilt disp vs 1.6543; rebuilt delta vs the
COMMITTED deltas[1] at cos > 0.999 + rel L2 < 5% — e193's G_S1CK
convention — where the committed vector exists: g1e/g1f; the locked
root's W1 predates the resume convention, so its rebuilt delta is
primary and gated by the disp + the landing read vs 0.9452); the FD
sign check per root (hard: read(theta0 + eps*s_0) > read(theta0) >
read(theta0 - eps*s_0) strictly at eps in {0.05, 0.02} — e204's
G_SENSDIR); the gradient/sensitivity conventions REGISTERED above
(the dual-estimator lesson stated); nothing guaranteed — the openness
is the point.

REGISTERED PREDICTION (frozen before compute): the honest prior is
TWO-SIDED. FOR a directional/spatial signature: the C-arm co-read
(the raw first step erases the cons roots' fact to ~0.00 while the
locked root holds 0.68) is too large and too replicated (0.2719 /
0.2577 to the second digit) to be first-order-orthogonal — SOME
geometry should separate the roots. AGAINST a clean firing: e204's
f2 lesson (first-order alignment reads were small and only partially
tracking; the support ROTATES) and T178's law (root strength softens
the shock — the roots' g-12 differ: 0.9156 / 0.8575 / 0.9682, a named
confound, co-noted). PREDICTED: the erase-alignment ORDERS locked <
cons at both draws; whether BOTH clear +0.05 is the open bit (the
most likely honest landing is GRADED). FALSIFIER: locked >= cons on
the primary read at either draw while the crush stands — the crush is
then not first-order-directional (toward PROJECTION-NEUTRAL / the
nonlinear cell). No bar shopping.

WHAT THIS CELL GUARANTEES: NOTHING — it is an alignment/composition
read (the class T157 taught the lab to distrust); n=1 per root (three
objects); the carrier set is a PROXY (top-|s_0|, not the directly
measured support — e204's own disclosure carried); no intervention
here, so no causality — a directional/spatial firing NAMES where to
intervene next, it does not establish the mechanism.

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never
claimed — the letter's "mostly eval-only; CPU fine"; the single AdamW
step per root is 2.74M CPU-viable), torch threads 4, the envelope
load-check poll recorded at start (recorded, not gating), tiny
sequential eval bursts, PROGRESSIVE metrics.json writes after every
phase, n=1 per root.

Outputs: runs/g11/{metrics.json (PROGRESSIVE), crush_mech.png}. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python g11_crush_mech.py    (G11_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e193/e204)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # THE OWNER ENVELOPE'S CAP

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (CharCorpus, run_dir, save_json)   # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
                                                      # (the 2.74M patch)
import g1_anchored_ball as G1                          # noqa: E402 — the
                                                      # machinery (patched to
                                                      # 2.74M by g1b's import)

torch.set_num_threads(4)           # shared machine (g1/g1b's import resets to 8)

import numpy as np                                     # noqa: E402 (plots)
import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("G11_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "g11 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ======================================================================
# THE CONFIG (the three roots + the frozen constants)
# ======================================================================
WASH_SEED = 10902                  # HELD (the licensed stream, verbatim)
R_CLAIM = 0.7                      # RAW L2 (the 2.74M convention)
LR_ADAMW = G1.FT_LR                # 1e-3 (the t=0 recipe)
CKPT_DIR = GB.CKPT_DIR
E185_X1_MD5 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's stored step-1 batch
                                                   # md5 (net-independent)

ROOTS = [
    {"tag": "locked_10901", "ckpt": "e131_consolidated_e113.pt",
     "cons_seed": 10901, "lineage": "the LOCKED draw (e109a/e113/g1b)",
     "ref_metrics": "runs/g1b/metrics.json",
     "committed_root_gm12": 0.9155886769294739,
     "committed_plus1": 0.9451885223388672,
     "committed_first_disp": 1.6542880535125732,
     "committed_delta": None,          # g1b predates the resume convention
     "resume_ck": None},
    {"tag": "g1e_10912", "ckpt": "g1e_root.pt",
     "cons_seed": 10912, "lineage": "the FIRST cons redraw (the +1 breach)",
     "ref_metrics": "runs/g1e/metrics.json",
     "committed_root_gm12": 0.857479989528656,
     "committed_plus1": 0.2719059884548187,
     "committed_first_disp": 1.6542880535125732,
     "committed_delta": "g1e_W1_resume.pt:deltas[1]",
     "resume_ck": "g1e_W1_resume.pt"},
    {"tag": "g1f_10913", "ckpt": "g1f_root.pt",
     "cons_seed": 10913, "lineage": "the SECOND cons redraw (CRUSH-IS-TEXTURE)",
     "ref_metrics": "runs/g1f/metrics.json",
     "committed_root_gm12": 0.9682114720344543,
     "committed_plus1": 0.25772199034690857,
     "committed_first_disp": 1.6542880535125732,
     "committed_delta": "g1f_W1_resume.pt:deltas[1]",
     "resume_ck": "g1f_W1_resume.pt"},
]
CONS_TAGS = ("g1e_10912", "g1f_10913")
LOCKED_TAG = "locked_10901"

# ---- the frozen adjudication constants --------------------------------------
K_PRIMARY = 2000                        # the carrier-proxy set size (frozen)
K_LADDER = (500, 1000, 2000, 5000, 10000) if not SMOKE else (2000,)
COS_DIFF_BAR = 0.05                     # ERASE-ALIGNED's ">= 0.05 consistently"
OVERLAP_RATIO_BAR = 1.5                 # CARRIER-CONCENTRATED's ">= 1.5x"
FD_EPS = (0.05, 0.02) if not SMOKE else (0.05,)   # e204's probe sizes (L2)
TOL_DIAL = 0.02                         # root dial vs committed (texture tier)
TOL_LANDING = 0.02                      # settled +1 read vs committed
TOL_DISP = 0.02                         # first-step disp vs committed 1.6543
S1CK_COS = 0.999                        # e193's G_S1CK convention
S1CK_REL = 0.05

REGISTERED_BARS = {
    "ERASE_ALIGNED": ("ERASE-ALIGNED: \"fires if the cons roots' first steps "
                      "align more with their fact-erasing directions than the "
                      "locked root's (cos difference >= 0.05 consistently) — "
                      "the crush is DIRECTIONAL: the jitter-replay leaves the "
                      "fact-support oriented toward the first wash gradient; "
                      "named.\""),
    "CARRIER_CONCENTRATED": ("CARRIER-CONCENTRATED: \"fires if the cons roots' "
                             "first deltas concentrate more on the "
                             "fact-carrier coordinates (top-k overlap >= 1.5x "
                             "the locked root's) — the crush is SPATIAL: the "
                             "consolidation packed the fact where the first "
                             "wash step lands; named.\""),
    "PROJECTION_NEUTRAL": ("PROJECTION-NEUTRAL: \"fires if the alignments and "
                           "concentrations match across roots — the crush "
                           "lives in the nonlinear interaction (not readable "
                           "from first-order geometry); the honest limit; the "
                           "nonlinear cell named.\""),
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": (
        "frozen BEFORE compute (they fix the clauses, they do not move the "
        "bars): the PRIMARY alignment read is the dispatch's literal "
        "cos(g_0, -s_0) with g_0 = the post-clip first wash gradient at the "
        "root (the stream's own arithmetic; cosine is clip-invariant — clip "
        "multiplies by a positive constant — the pre-clip norm co-reported) "
        "and s_0 = the unit fact-sensitivity at the SAME root (matched-point, "
        "T150); BOTH orientations ride in the table, the verdict reads the "
        "primary only. ERASE-ALIGNED fires iff BOTH cons roots' primary read "
        ">= locked's + 0.05 ('consistently' = the two fresh draws agree). "
        "CARRIER-CONCENTRATED fires iff BOTH cons roots' overlap at k=2000 "
        ">= 1.5x the locked root's, overlap = |top2000(|delta|) n "
        "top2000(|s_0|)|/2000, the ratio's denominator floored at one "
        "coordinate (disclosed); the k-ladder co-reported, never adjudicated. "
        "PROJECTION-NEUTRAL fires iff neither of the above fires AND the "
        "reads MATCH across roots: |cos_cons - cos_locked| < 0.05 at BOTH "
        "cons roots AND both overlap ratios strictly inside (1/1.5, 1.5). "
        "GRADED fires otherwise (any partial). Composite order frozen: a "
        "full firing names the mechanism(s) (both bars may fire — the crush "
        "can be directional AND spatial); hard-gate failure => the record "
        "completes, verdict TEXTURE (gate failure), nothing adjudicated."),
    "registered_prediction": (
        "TWO-SIDED, frozen before compute. FOR a directional/spatial "
        "signature: the C-arm co-read (the RAW first step erases the cons "
        "roots' fact to ~0.00 while the locked root holds 0.68) is too large "
        "and too replicated (0.2719/0.2577 to the second digit) to be "
        "first-order-orthogonal. AGAINST a clean firing: e204's f2 lesson "
        "(first-order alignments small, only partially tracking; the support "
        "ROTATES) and T178's law (root strength softens the shock — the "
        "roots' g-12 differ 0.9156/0.8575/0.9682, a named confound). "
        "PREDICTED: the erase-alignment ORDERS locked < cons at both draws; "
        "whether BOTH clear +0.05 is the open bit (the most likely honest "
        "landing is GRADED). FALSIFIER: locked >= cons on the primary read "
        "at either draw while the crush stands — the crush is then not "
        "first-order-directional. No bar shopping."),
    "registration": ("the dispatch's registration IS the registration (the "
                     "bars quoted verbatim here and in the module docstring, "
                     "frozen before compute). Adjudicate against exactly "
                     "this; no bar shopping."),
}

deviations: list[str] = [
    "CPU-ONLY, FORCED (e193/e204's convention; the letter's 'mostly "
    "eval-only; CPU fine'): CUDA_VISIBLE_DEVICES=-1 before torch; the GPU is "
    "never claimed; the single AdamW first step per root runs on CPU (2.74M, "
    "seconds); the envelope load-check poll is recorded at start (recorded, "
    "not gating).",
    "THE LOCKED ROOT'S FIRST-STEP DELTA IS REBUILT: g1b's W1 predates the "
    "resume-ckpt convention, so no committed delta vector exists for the "
    "locked root; its rebuilt delta is PRIMARY and gated by the disp-norm "
    "(1.6543) + the settled landing read vs the committed +1 (0.9452, "
    "texture tier 0.02). g1e/g1f's deltas are LOADED COMMITTED from the "
    "resume ckpts (deltas[1]) with the CPU rebuild as the gate (cos > 0.999 "
    "+ rel L2 < 5%, e193's G_S1CK convention).",
    "THE CARRIER SET IS A PROXY (e204's own disclosure carried): top-|s_0| "
    "at k=2000 primary (0.073% of 2,739,072 coords), ladder 500/1000/5000/"
    "10000 co-reported; the directly measured support does not exist at "
    "these roots; k is this cell's frozen choice (the letter fixes no k).",
    "THE SIGN CONVENTIONS (the critic's, carried): s_0 = grad of the fact "
    "readout, so -s_0 is the fact-ERASING ray; the PRIMARY read is the "
    "letter's literal cos(g_0, -s_0); the exact negation cos(g_0, +s_0) and "
    "the displacement reads cos(delta, +-s_0) ride verbatim (the projection "
    "rescales the delta by a positive constant, so the projected and raw "
    "displacement reads are IDENTICAL — disclosed, not recomputed).",
    "CLIP-INVARIANCE DISCLOSED: clip_grad_norm_ multiplies the gradient by "
    "a positive constant, so cos(g_0, s_0) is identical pre/post clip; the "
    "pre-clip norm is reported as texture (the roots' pre-clip norms differ "
    "— the gradient SCALE is its own read, outside the bars).",
    "THE ROOT-STRENGTH CONFOUND, CO-NOTED (T178): the three roots' g-12 "
    "differ (0.9156 / 0.8575 / 0.9682); T178's law (strength softens the "
    "shock) predicts the STRONGER roots crush LESS — g1f's root (0.9682) is "
    "the strongest yet crushes hardest, already against a pure-strength "
    "reading; the alignment/composition reads carry the question regardless.",
    "THE FD SIGN CHECK IS HARD per root (e204's G_SENSDIR): the direction "
    "must BE the fact's sensitivity (read(theta0+eps*s) strictly monotone at "
    "BOTH eps) before any alignment is believed; failure => TEXTURE, the "
    "record completes.",
    "n=1 PER ROOT (three objects); alignment/composition reads cannot "
    "establish causality (no intervention); a firing names where to "
    "intervene next, not a mechanism proven.",
    "torch threads 4 (shared machine; g1/g1b's import resets to 8 — reset "
    "after import).",
    "Smoke mode trims: FD eps {0.05}, the k-ladder to {2000}; nothing "
    "adjudicated.",
]

device_events: list[dict] = []
RD: Path = None                     # set in main (run_dir)
METRICS: dict = {}
_progressive = {"n": 0, "phases": []}

# the per-root unit-vector stash (module-level, never serialized)
_S0: dict = {}
_DELTA: dict = {}


# ------------------------------------------------------------------ instruments
def flat_params(net) -> torch.Tensor:
    """The fp32 flat parameter vector (net.parameters() order; 2,739,072)."""
    return torch.cat([p.detach().reshape(-1)
                      for p in net.parameters()]).clone()


def load_flat(net, flat: torch.Tensor) -> None:
    """Copy a flat vector back into parameters (the static-jump loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 cosine (the chart's estimator precision)."""
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def sd_md5_body(sd: dict) -> str:
    """Provenance hash of a model state dict's BODY (no anch__ buffers)."""
    h = hashlib.md5()
    for k in sorted(sd):
        if k.startswith("anch__"):
            continue
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def anch_md5_of(sd: dict) -> str:
    """md5 of a committed state dict's anchor stack (the theta_anchor id)."""
    return hashlib.md5(torch.cat(
        [sd[k].reshape(-1) for k in sorted(sd)
         if k.startswith("anch__") and k != "anch__R"]
    ).numpy().tobytes()).hexdigest()


def fact_readout(net, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e194/e195/e204's fact objective, VALUE ONLY: mean log p(Z) over the
    g-12 battery (the readout currency of the FD sign check)."""
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            lg, _ = net(ids[i:i + bs])
            tot += float(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
            n += ids[i:i + bs].shape[0]
    return tot / max(n, 1)


def fact_grad(net, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE SENSITIVITY INSTRUMENT (e194/e195's fact_grad VERBATIM in its
    arithmetic; e204's copy): gradient of the fact battery's mean log p(Z)
    readout at the net's CURRENT weights. Sign convention (the critic's,
    carried): -s = the fact-ERASING ray. Consumes no RNG; tiny eval burst."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    F_obj = torch.stack(sums).sum() / ids.shape[0]
    F_obj.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
    net.zero_grad(set_to_none=True)
    return g


def topk_idx(v: torch.Tensor, k: int) -> set:
    """Top-k coordinate INDICES by |v| (the carrier set / delta set)."""
    return set(torch.topk(v.abs(), k).indices.tolist())


def overlap_frac(a: set, b: set, k: int) -> float:
    return len(a & b) / k


def write_partial(phase: str) -> None:
    """PROGRESSIVE metrics (the outage lesson): metrics.json after every
    phase; bookkeeping must never kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    METRICS.update({
        "experiment": "g11_crush_mech", "date": common.now_iso(),
        "phase": phase, "write_n": _progressive["n"],
        "phases": list(_progressive["phases"]),
        "status": "PARTIAL" if phase != "final" else "COMPLETE",
        "timing_partial": {"total_s": round(time.time() - T0, 1)},
    })
    try:
        save_json(RD / "metrics.json", E43.jsonable(METRICS))
        log(f"[partial] metrics.json updated (phase '{phase}', write "
            f"#{_progressive['n']})")
    except Exception as e:
        log(f"[partial] WRITE FAILED at '{phase}' ({e}) — continuing")


def fail_texture(reason: str) -> None:
    """A hard gate failed: the record completes, verdict TEXTURE (the
    registered composite clause); nothing adjudicated."""
    METRICS["adjudication"] = {
        "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                          ("ERASE_ALIGNED", "CARRIER_CONCENTRATED",
                           "PROJECTION_NEUTRAL", "GRADED")},
        "verdict": "TEXTURE (gate failure)",
        "reason": reason,
        "note": "the registered composite: hard-gate failure => the record "
                "completes with verdict TEXTURE, nothing adjudicated",
    }
    log(f"[TEXTURE] hard gate failure: {reason}")
    write_partial("texture (gate failure)")


# ======================================================================
# MAIN
# ======================================================================

def main():
    global RD
    RD = run_dir("g11_smoke" if SMOKE else "g11")
    log(f"G11 THE CRUSH MECHANISM (T194's named question) (smoke={SMOKE}) "
        f"-> {RD}")
    # the envelope load-check (recorded, not gating; CPU-only policy)
    s0_poll = common.gpu_status()
    try:
        common._log_envelope_poll("g11:load-check(cpu-only-policy)",
                                  s0_poll["util"], s0_poll["temp"], True)
    except Exception:
        pass
    device_events.append({
        "tag": "g11", "event": "LOAD CHECK (CPU-only policy; GPU never "
        "claimed)", "status": s0_poll,
        "note": "the letter's envelope: eval-only cell; the single AdamW "
                "step per root runs CPU (2.74M); no GPU launch"})
    log(f"[envelope] load-check recorded: util {s0_poll['util']:.0f}% temp "
        f"{s0_poll['temp']:.0f}C — CPU-only policy (no GPU launch)")

    METRICS.update({
        "design": (
            "G11 THE CRUSH MECHANISM: at THREE roots (the locked e131 "
            "cons-10901; g1e's cons-10912; g1f's cons-10913 — all committed, "
            "provenance-gated): (1) the first wash gradient g_0 (the "
            "seed-10902 stream, VERBATIM) vs each root's OWN fact "
            "sensitivity s_0 (the e204 convention) — the primary read "
            "cos(g_0, -s_0); (2) the ball geometry: the first raw delta "
            "(1.6543) vs R=0.7, the rescale ratio, the projected landing "
            "point and the same alignment AFTER projection, the landing "
            "read gated vs the committed +1 crushes (0.9452 / 0.2719 / "
            "0.2577); (3) the coordinate composition: the first delta's "
            "top-|delta| overlap with each root's OWN fact-carrier set "
            "(top-|s_0|, the e204 sensitivity top-k as the carrier proxy), "
            "the mass fractions, per-tensor tables, the differing-weights "
            "co-read. T194's question: what does the jitter-replay "
            "consolidation leave that costs the fact one extra projected "
            "step at the first wash gradient — the wash/root draws don't "
            "pay it?"),
        "registered": REGISTERED_BARS,
        "deviations": deviations,
        "smoke": SMOKE,
        "owner_envelope": {
            "policy": "CPU-ONLY forced (CUDA_VISIBLE_DEVICES=-1); threads 4; "
                      "no GPU launch; load-check poll recorded",
            "device_events": device_events},
    })
    write_partial("start (bars + prediction registered before compute)")

    # ---------------- protocol rebuild (g1e/g1f VERBATIM) ---------------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    import random as _random
    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    # the fact batteries (the g-12 install-60 battery = the e194/e195/e204
    # sensitivity convention; g0 co-reported for context)
    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    G_BATTERY = {"shapes": {str(j): list(bat_ids[j].shape) for j in G1.GEOS},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]),
                 "note": "the g-12 install-60 battery (e194/e195's fact_grad "
                         "convention, e204's carrier; shapes 60x118)"}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    # the neutral anchor bank (e170's construction VERBATIM via g1e/g1f)
    arng = _random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in n_starts])

    # ---------------- the first wash batch (the seed-10902 stream) ------------
    def draw_step1_batch():
        g = torch.Generator().manual_seed(WASH_SEED)
        aj = torch.randint(anchor_neutral.shape[0], (G1.ANCH_BS,),
                           generator=g)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    x1, y1 = draw_step1_batch()
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()

    # load the resume ckpts' committed stream hashes + first deltas + anchors
    committed_stream: dict = {}
    committed_deltas: dict = {}
    committed_anchors: dict = {}
    for spec in ROOTS:
        if spec["resume_ck"] is None:
            continue
        rc = torch.load(CKPT_DIR / spec["resume_ck"], map_location="cpu",
                        weights_only=False)
        committed_stream[spec["tag"]] = rc["x_hashes"].get(
            1, rc["x_hashes"].get("1"))
        committed_deltas[spec["tag"]] = rc["deltas"][1].float()
        committed_anchors[spec["tag"]] = {
            "wall_R": float(rc["wall_R"]),
            "anch_md5": anch_md5_of(rc["sds"][1]),
        }
        del rc
    G_STREAM = {
        "x1_md5": x1_md5,
        "vs_g1e_resume": (committed_stream.get("g1e_10912") == x1_md5
                          if committed_stream.get("g1e_10912") else None),
        "vs_g1f_resume": (committed_stream.get("g1f_10913") == x1_md5
                          if committed_stream.get("g1f_10913") else None),
        "vs_e185_stored": bool(x1_md5 == E185_X1_MD5),
        "note": "the seed-10902 stream drawn VERBATIM (the e170 bank + the "
                "first batch); md5-gated vs the g1e/g1f resume ckpts' own "
                "x_hashes[1] (the exact batches the committed runs "
                "consumed) + e185's stored hash (the family co-report)",
    }
    G_STREAM["pass"] = bool(
        G_STREAM["vs_e185_stored"]
        and all(v is None or v for v in (G_STREAM["vs_g1e_resume"],
                                         G_STREAM["vs_g1f_resume"])))
    log(f"G_STREAM: x1 md5 {x1_md5[:10]}.. vs e185 "
        f"{G_STREAM['vs_e185_stored']} vs g1e {G_STREAM['vs_g1e_resume']} "
        f"vs g1f {G_STREAM['vs_g1f_resume']}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    if not G_STREAM["pass"]:
        METRICS["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                            "G_BATTERY": G_BATTERY, "G_STREAM": G_STREAM}
        fail_texture("G_STREAM failed — the stream diverged from the "
                     "committed runs' own x_hashes")
        return
    METRICS["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_STREAM": G_STREAM}
    write_partial("protocol + stream gates")

    # =====================================================================
    # THE CELL — one measurement bundle per root
    # =====================================================================
    cells: dict = {}
    theta_map: dict = {}

    for spec in ROOTS:
        tag = spec["tag"]
        log("=" * 78)
        log(f"ROOT {tag} — {spec['lineage']} ({spec['ckpt']})")

        # ---- load + the root provenance gate
        raw = torch.load(CKPT_DIR / spec["ckpt"], map_location="cpu",
                         weights_only=False)
        root_sd = {k: v.detach().clone() for k, v in
                   (raw["model"] if isinstance(raw, dict)
                    and "model" in raw else raw).items()}
        meta = E43.jsonable(raw.get("meta", {})) if isinstance(raw, dict) \
            else {}
        del raw
        net_root = G1.CommittedGPT(GB.G1B_CFG)
        net_root.load_state_dict(root_sd)
        net_root.eval()
        n_params = net_root.num_params()
        theta0 = flat_params(net_root)
        theta_map[tag] = theta0
        if n_params != GB.G1B_PARAMS:
            METRICS["gates"][f"{tag}__G_ROOT"] = {
                "params": n_params, "pass": False}
            fail_texture(f"{tag}: params {n_params} != {GB.G1B_PARAMS}")
            return

        root_gm12 = G1.battery_cell(net_root, gm12_ids, zid)["mean_pz"]
        root_g0 = G1.battery_cell(net_root, g0_ids, zid)["mean_pz"]
        meta_ok = True
        if tag != LOCKED_TAG:
            meta_ok = bool(meta.get("experiment") == ("g1e" if "g1e" in tag
                                                      else "g1f")
                           and meta.get("cons_seed") == spec["cons_seed"])
        G_ROOT = {
            "checkpoint": f"runs/checkpoints/{spec['ckpt']}",
            "meta": meta, "params": n_params,
            "body_md5": sd_md5_body(root_sd),
            "gm12": root_gm12, "gm12_committed": spec["committed_root_gm12"],
            "gm12_d": abs(root_gm12 - spec["committed_root_gm12"]),
            "g0": root_g0, "meta_gate": meta_ok,
            "tol": TOL_DIAL,
            "pass": bool(abs(root_gm12 - spec["committed_root_gm12"])
                         <= TOL_DIAL and meta_ok),
            "note": "the root loads into the 2,739,072 cfg clean; its "
                    "same-instrument g-12 reproduces the committed root "
                    "cell (texture tier 0.02); meta provenance for the "
                    "fresh draws (experiment + cons_seed)",
        }
        log(f"G_ROOT[{tag}]: gm12 {root_gm12:.4f} vs committed "
            f"{spec['committed_root_gm12']:.4f} (|d| "
            f"{G_ROOT['gm12_d']:.2e}), meta "
            f"{'OK' if meta_ok else 'DRIFT'}: "
            + ("PASS" if G_ROOT["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_ROOT"] = G_ROOT
        if not G_ROOT["pass"]:
            fail_texture(f"{tag}: root provenance gate FAILED")
            return

        # ---- G-ANCHOR (g1e/g1f): the resume sds' anchor IS the root ------
        if tag in committed_anchors:
            net_c = G1.CommittedGPT(GB.G1B_CFG)
            net_c.load_state_dict(root_sd)
            net_c.commit(R_CLAIM)
            anch_md5_root = anch_md5_of(net_c.state_dict())
            wall_r_ok = abs(committed_anchors[tag]["wall_R"]
                            - R_CLAIM) < 1e-6
            md5_ok = bool(anch_md5_root
                          == committed_anchors[tag]["anch_md5"])
            g_anchor = {
                "form": "the W1 arm's wall anchor (theta_anchor) IS the "
                        "root — the anch__ buffers in the committed resume "
                        "sds[1] must equal the root ckpt's commit(R)",
                "wall_R_resume": committed_anchors[tag]["wall_R"],
                "wall_R_expected": R_CLAIM,
                "wall_R_match": bool(wall_r_ok),
                "anch_md5_root_rebuilt": anch_md5_root,
                "anch_md5_resume": committed_anchors[tag]["anch_md5"],
                "anch_md5_match": md5_ok,
                "pass": bool(wall_r_ok and md5_ok),
            }
            del net_c
            log(f"G-ANCHOR[{tag}]: wall_R "
                f"{committed_anchors[tag]['wall_R']:.7f} "
                f"{'match' if wall_r_ok else 'DRIFT'}, anchor md5 "
                f"{'match' if md5_ok else 'DRIFT'}: "
                + ("PASS" if g_anchor["pass"] else "FAIL"))
            METRICS["gates"][f"{tag}__G_ANCHOR"] = g_anchor
            if not g_anchor["pass"]:
                fail_texture(f"{tag}: G-ANCHOR failed — the committed wall's "
                             "anchor is not this root")
                return

        # ---- s_0: the root's OWN fact sensitivity (the e204 convention) --
        s_raw = fact_grad(net_root, gm12_ids, zid)
        s0 = s_raw / torch.norm(s_raw)
        _S0[tag] = s0.clone()
        read0 = fact_readout(net_root, gm12_ids, zid)
        # the FD sign check (HARD, e204's G_SENSDIR)
        fd = {}
        twin = G1.CommittedGPT(GB.G1B_CFG)
        twin.load_state_dict(root_sd)
        twin.eval()
        for eps in FD_EPS:
            for sgn, nm in ((+1.0, "plus"), (-1.0, "minus")):
                load_flat(twin, theta0 + sgn * eps * s0)
                fd[f"eps{eps}_{nm}"] = fact_readout(twin, gm12_ids, zid)
            fd[f"eps{eps}_monotone"] = bool(
                fd[f"eps{eps}_plus"] > read0 > fd[f"eps{eps}_minus"])
        del twin
        G_SENSDIR = {
            "convention": "e194/e195's fact_grad VERBATIM (the g-12 "
                          "install-60 battery; matched-point at the root); "
                          "FD sign check HARD at eps in "
                          f"{tuple(FD_EPS)} (e204's G_SENSDIR)",
            "readout_theta0_mean_log_pz": read0,
            "probes": fd,
            "pass": bool(all(fd[f"eps{eps}_monotone"] for eps in FD_EPS)),
        }
        log(f"G_SENSDIR[{tag}]: read0 {read0:.4f}; FD monotone "
            + "/".join(f"eps{e}:{fd[f'eps{e}_monotone']}" for e in FD_EPS)
            + ": " + ("PASS" if G_SENSDIR["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_SENSDIR"] = G_SENSDIR
        if not G_SENSDIR["pass"]:
            fail_texture(f"{tag}: FD sign check FAILED — s_0 is not the "
                         "fact's sensitivity at this root")
            return

        # ---- the first wash step, REBUILT (the stream's arithmetic) -------
        net_w = G1.CommittedGPT(GB.G1B_CFG)
        net_w.load_state_dict(root_sd)
        net_w.commit(R_CLAIM)          # theta_anchor = the root (verbatim)
        net_w.train()
        opt = torch.optim.AdamW(net_w.parameters(), lr=LR_ADAMW,
                                betas=(0.9, 0.95), weight_decay=0.1)
        logits, _ = net_w(x1)          # the wall projects here (no-op at root)
        ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    y1.reshape(-1)).item())
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y1.reshape(-1))
        loss.backward()
        preclip_norm = float(torch.norm(torch.stack(
            [torch.norm(p.grad.detach()) for p in net_w.parameters()])))
        torch.nn.utils.clip_grad_norm_(net_w.parameters(), 1.0)
        g0 = torch.cat([p.grad.detach().reshape(-1)
                        for p in net_w.parameters()]).clone()  # post-clip
        clip_binds = bool(preclip_norm > 1.0)
        opt.step()
        theta1 = flat_params(net_w)                 # RAW post-step (outside)
        delta_rebuilt = theta1 - theta0
        d_raw = float(torch.norm(delta_rebuilt))
        del logits, loss

        # ---- the primary delta: COMMITTED where it exists ------------------
        delta_commit = committed_deltas.get(tag)
        if delta_commit is not None:
            cos_rb_c = cos64(delta_rebuilt, delta_commit)
            rel_l2 = abs(float(torch.norm(delta_commit)) - d_raw) / d_raw
            G_STEP1 = {"disp_rebuilt": d_raw,
                       "disp_committed": spec["committed_first_disp"],
                       "disp_d": abs(d_raw - spec["committed_first_disp"]),
                       "cos_rebuilt_vs_committed": cos_rb_c,
                       "rel_l2_vs_committed": rel_l2,
                       "primary": "COMMITTED deltas[1] (the run's own "
                                  "vector); the rebuild is the gate",
                       "pass": bool(abs(d_raw - spec["committed_first_disp"])
                                    <= TOL_DISP
                                    and cos_rb_c > S1CK_COS
                                    and rel_l2 < S1CK_REL)}
            delta = delta_commit
        else:
            G_STEP1 = {"disp_rebuilt": d_raw,
                       "disp_committed": spec["committed_first_disp"],
                       "disp_d": abs(d_raw - spec["committed_first_disp"]),
                       "primary": "REBUILT (g1b predates the resume "
                                  "convention; gated by disp + landing)",
                       "pass": bool(abs(d_raw - spec["committed_first_disp"])
                                    <= TOL_DISP)}
            delta = delta_rebuilt
        _DELTA[tag] = delta.clone()
        log(f"G_STEP1[{tag}]: |delta| {d_raw:.4f} vs committed "
            f"{spec['committed_first_disp']:.4f} (|d| "
            f"{G_STEP1['disp_d']:.2e})"
            + (f"; rebuild-vs-committed cos {cos_rb_c:.6f} rel {rel_l2:.2e}"
               if delta_commit is not None else "")
            + ": " + ("PASS" if G_STEP1["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_STEP1"] = G_STEP1
        if not G_STEP1["pass"]:
            fail_texture(f"{tag}: first-step gate FAILED")
            return

        # ---- (2) the ball geometry + the landing read ---------------------
        theta_raw1 = theta0 + delta          # the PRIMARY delta's endpoint
        d_primary = float(torch.norm(delta))
        rescale = R_CLAIM / d_primary
        theta_land = theta0 + rescale * delta
        # the settled +1 read: the in-run ARMED-twin convention (the twin's
        # own forward projects in place — the projection arithmetic gate)
        evl = copy.deepcopy(net_w)      # committed; anchors = the root
        load_flat(evl, theta_raw1)      # the raw (outside-ball) endpoint
        land_read = G1.battery_cell(evl, gm12_ids, zid)["mean_pz"]
        settled = flat_params(evl)
        proj_arith_d = float((settled - theta_land).abs().max())
        del evl
        # the RAW landing read (the C-arm convention, co-reported)
        evl_raw = G1.CommittedGPT(GB.G1B_CFG)
        evl_raw.load_state_dict(root_sd)
        load_flat(evl_raw, theta_raw1)
        raw_read = G1.battery_cell(evl_raw, gm12_ids, zid)["mean_pz"]
        del evl_raw
        G_LAND = {
            "committed_plus1": spec["committed_plus1"],
            "settled_read": land_read,
            "d_vs_committed": abs(land_read - spec["committed_plus1"]),
            "raw_read_coreported": raw_read,
            "projection_arithmetic_maxdiff": proj_arith_d,
            "tol": TOL_LANDING,
            "pass": bool(abs(land_read - spec["committed_plus1"])
                         <= TOL_LANDING and proj_arith_d < 1e-5),
            "note": "the settled read at the projected landing point "
                    "reproduces the committed +1 crush read (the in-run "
                    "ARMED-twin convention) + the in-place projection "
                    "matches theta0 + (R/|delta|)*delta to fp32",
        }
        log(f"G_LAND[{tag}]: settled +1 {land_read:.4f} vs committed "
            f"{spec['committed_plus1']:.4f} (|d| "
            f"{G_LAND['d_vs_committed']:.2e}); raw landing {raw_read:.4f}; "
            f"proj arith |d| {proj_arith_d:.1e}: "
            + ("PASS" if G_LAND["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_LAND"] = G_LAND
        if not G_LAND["pass"]:
            fail_texture(f"{tag}: landing gate FAILED — the rebuilt first "
                         "step does not reproduce the committed +1 read")
            return

        # ---- (1) the alignments -------------------------------------------
        align = {
            "primary_cos_g0_neg_s0": cos64(g0, -s0),
            "cos_g0_pos_s0": cos64(g0, s0),
            "cos_delta_neg_s0": cos64(delta, -s0),
            "cos_delta_pos_s0": cos64(delta, s0),
            "note": "the projection rescales delta by a positive constant, "
                    "so the projected-step alignment is IDENTICAL to "
                    "cos_delta_*; cos(g0, +-s0) is clip-invariant; both "
                    "orientations verbatim",
        }

        # ---- (3) the coordinate composition --------------------------------
        dset = {k: topk_idx(delta, k) for k in K_LADDER}
        sset = {k: topk_idx(s0, k) for k in K_LADDER}
        overlaps = {k: overlap_frac(dset[k], sset[k], k) for k in K_LADDER}
        carrier_coords = torch.tensor(sorted(sset[K_PRIMARY]),
                                      dtype=torch.long)
        d_mass_total = float(delta.abs().sum())
        mass_on_carriers = float(delta.abs()[carrier_coords].sum())
        mass_frac = mass_on_carriers / d_mass_total
        null_frac = K_PRIMARY / float(theta0.numel())
        per_tensor, i0 = {}, 0
        d_sq = float(torch.norm(delta) ** 2)
        for name, p in net_w.named_parameters():
            n = p.numel()
            per_tensor[name] = {
                "n": n,
                "delta_l2_frac": float(torch.norm(
                    delta[i0:i0 + n]) ** 2) / d_sq,
                "s_l2_frac": float(torch.norm(s0[i0:i0 + n]) ** 2),
            }
            i0 += n
        cells[tag] = {
            "spec": {k: spec[k] for k in ("tag", "ckpt", "cons_seed",
                                          "lineage", "ref_metrics")},
            "root": {"gm12": root_gm12, "g0": root_g0,
                     "body_md5": G_ROOT["body_md5"],
                     "readout_mean_log_pz": read0},
            "first_step": {"ce_batch": ce1, "preclip_gnorm": preclip_norm,
                           "clip_binds": clip_binds,
                           "d_raw_rebuilt": d_raw,
                           "d_primary": d_primary, "R": R_CLAIM,
                           "rescale_ratio": rescale,
                           "delta_primary": G_STEP1["primary"]},
            "landing": {"settled_plus1": land_read, "raw_plus1": raw_read,
                        "committed_plus1": spec["committed_plus1"],
                        "crush_delta_vs_root": land_read - root_gm12},
            "alignment": align,
            "composition": {"overlap_by_k": overlaps,
                            "k_primary": K_PRIMARY,
                            "mass_frac_on_carriers": mass_frac,
                            "null_frac_k_over_P": null_frac,
                            "mass_excess_ratio": mass_frac / null_frac},
            "per_tensor": per_tensor,
            "_dset_primary": dset[K_PRIMARY],
            "_sset_primary": sset[K_PRIMARY],
        }
        del net_w, net_root, opt, g0, s_raw
        write_partial(f"cell {tag} (gates + reads)")

    # ---- the differing-weights co-read + the mutual geometry ---------------
    log("=" * 78)
    log("THE CROSS-ROOT READS (differing weights + mutual geometry)")
    for tag in CONS_TAGS:
        diff_vec = (theta_map[tag] - theta_map[LOCKED_TAG]).abs()
        diff_set = topk_idx(diff_vec, K_PRIMARY)
        cells[tag]["composition"]["differing_overlap_primary"] = \
            overlap_frac(cells[tag]["_dset_primary"], diff_set, K_PRIMARY)
        cells[tag]["composition"]["sens_on_differing_primary"] = \
            overlap_frac(cells[tag]["_sset_primary"], diff_set, K_PRIMARY)
        log(f"  {tag}: delta-on-differing overlap "
            f"{cells[tag]['composition']['differing_overlap_primary']:.4f}; "
            f"sens-on-differing "
            f"{cells[tag]['composition']['sens_on_differing_primary']:.4f}")
    mutual = {}
    for a, b in ((LOCKED_TAG, "g1e_10912"), (LOCKED_TAG, "g1f_10913"),
                 ("g1e_10912", "g1f_10913")):
        sa, sb = cells[a]["_sset_primary"], cells[b]["_sset_primary"]
        da, db = cells[a]["_dset_primary"], cells[b]["_dset_primary"]
        mutual[f"{a}__{b}"] = {
            "root_l2": float(torch.norm(theta_map[a] - theta_map[b])),
            "cos_s0_s0": cos64(_S0[a], _S0[b]),
            "cos_delta_delta": cos64(_DELTA[a], _DELTA[b]),
            "sens_topk_overlap": overlap_frac(sa, sb, K_PRIMARY),
            "delta_topk_overlap": overlap_frac(da, db, K_PRIMARY),
        }
        m = mutual[f"{a}__{b}"]
        log(f"  mutual[{a} vs {b}]: root L2 {m['root_l2']:.3f}; "
            f"cos(s0,s0) {m['cos_s0_s0']:+.4f}; cos(d,d) "
            f"{m['cos_delta_delta']:+.4f}; sens topk ov "
            f"{m['sens_topk_overlap']:.4f}; delta topk ov "
            f"{m['delta_topk_overlap']:.4f}")

    for tag in list(cells):
        cells[tag].pop("_dset_primary", None)
        cells[tag].pop("_sset_primary", None)
    METRICS["cells"] = cells
    METRICS["mutual"] = mutual
    write_partial("cross-root reads complete")

    # =====================================================================
    # THE ADJUDICATION (frozen composite; no bar shopping)
    # =====================================================================
    cosE = {t: cells[t]["alignment"]["primary_cos_g0_neg_s0"]
            for t in cells}
    ovl = {t: cells[t]["composition"]["overlap_by_k"][K_PRIMARY]
           for t in cells}
    ovl_counts = {t: int(round(ovl[t] * K_PRIMARY)) for t in cells}
    erase_diff = {c: cosE[c] - cosE[LOCKED_TAG] for c in CONS_TAGS}
    ovl_locked_floor = max(ovl[LOCKED_TAG], 1.0 / K_PRIMARY)
    ov_ratio = {c: ovl[c] / ovl_locked_floor for c in CONS_TAGS}
    erase_fired = all(d >= COS_DIFF_BAR for d in erase_diff.values())
    carrier_fired = all(r >= OVERLAP_RATIO_BAR for r in ov_ratio.values())
    # generic boundary disclosure (Rule-12 honesty): flag any decisive read
    # within 2% of its bar so a boundary landing is never read as a margin
    boundary = {
        "erase_within_2pct_of_bar": {c: bool(abs(erase_diff[c] - COS_DIFF_BAR)
                                             <= 0.02 * COS_DIFF_BAR
                                             or abs(erase_diff[c])
                                             <= 0.02 * COS_DIFF_BAR)
                                     for c in CONS_TAGS},
        "carrier_within_2pct_of_bar": {c: bool(
            abs(ov_ratio[c] - OVERLAP_RATIO_BAR)
            <= 0.02 * OVERLAP_RATIO_BAR) for c in CONS_TAGS},
        "note": "a decisive read within 2% of its bar is flagged — the "
                "verdict stands on the frozen comparison (>= is >=), the "
                "flag carries the honesty",
    }
    neutral_match = (all(abs(d) < COS_DIFF_BAR for d in erase_diff.values())
                     and all(1.0 / OVERLAP_RATIO_BAR < r < OVERLAP_RATIO_BAR
                             for r in ov_ratio.values()))
    if erase_fired and carrier_fired:
        verdict = "ERASE-ALIGNED + CARRIER-CONCENTRATED"
    elif erase_fired:
        verdict = "ERASE-ALIGNED"
    elif carrier_fired:
        verdict = "CARRIER-CONCENTRATED"
    elif neutral_match:
        verdict = "PROJECTION-NEUTRAL"
    else:
        verdict = "GRADED"
    adjudication = {
        "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                          ("ERASE_ALIGNED", "CARRIER_CONCENTRATED",
                           "PROJECTION_NEUTRAL", "GRADED")},
        "operationalization": REGISTERED_BARS["operationalizations"],
        "constants": {"cos_diff_bar": COS_DIFF_BAR,
                      "overlap_ratio_bar": OVERLAP_RATIO_BAR,
                      "k_primary": K_PRIMARY,
                      "k_ladder": list(K_LADDER)},
        "erase_table": {"cos_g0_neg_s0": cosE,
                        "cons_minus_locked": erase_diff,
                        "fired": bool(erase_fired),
                        "clause": "BOTH cons roots >= locked + 0.05"},
        "carrier_table": {"overlap_k2000": ovl,
                          "overlap_counts_k2000": ovl_counts,
                          "cons_over_locked": ov_ratio,
                          "locked_floor": ovl_locked_floor,
                          "fired": bool(carrier_fired),
                          "clause": "BOTH cons roots' overlap >= 1.5x "
                                    "locked's (denominator floored at one "
                                    "coordinate)"},
        "boundary_disclosure": boundary,
        "co_read_texture": {
            "mass_excess_ratio": {t: cells[t]["composition"][
                "mass_excess_ratio"] for t in cells},
            "mass_note": "the delta's L1 mass ON the carrier coords vs the "
                         "k/P null — 1.0x = exactly null concentration (the "
                         "top-k overlap read's honest companion; never "
                         "adjudicated)",
            "delta_topk_mutual_overlap": {
                f"{a}__{b}": mutual[f"{a}__{b}"]["delta_topk_overlap"]
                for a, b in ((LOCKED_TAG, "g1e_10912"),
                             (LOCKED_TAG, "g1f_10913"),
                             ("g1e_10912", "g1f_10913"))},
            "sens_topk_mutual_overlap": {
                f"{a}__{b}": mutual[f"{a}__{b}"]["sens_topk_overlap"]
                for a, b in ((LOCKED_TAG, "g1e_10912"),
                             (LOCKED_TAG, "g1f_10913"),
                             ("g1e_10912", "g1f_10913"))},
        },
        "neutral_clause": {"matches": bool(neutral_match),
                           "fired": bool(not erase_fired
                                         and not carrier_fired
                                         and neutral_match)},
        "verdict": verdict,
    }
    METRICS["adjudication"] = adjudication
    log("=" * 78)
    log(f"THE VERDICT: {verdict}")
    log(f"  erase: cos(g0,-s0) locked {cosE[LOCKED_TAG]:+.4f} | "
        + " | ".join(f"{c} {cosE[c]:+.4f} (d {erase_diff[c]:+.4f})"
                     for c in CONS_TAGS))
    log(f"  carrier: overlap@{K_PRIMARY} locked {ovl[LOCKED_TAG]:.4f} | "
        + " | ".join(f"{c} {ovl[c]:.4f} ({ov_ratio[c]:.2f}x)"
                    for c in CONS_TAGS))
    write_partial("adjudication")

    # =====================================================================
    # THE CHART
    # =====================================================================
    tags = [LOCKED_TAG] + list(CONS_TAGS)
    short = {LOCKED_TAG: "locked 10901\n(+1 0.945)",
             "g1e_10912": "cons 10912\n(+1 0.272)",
             "g1f_10913": "cons 10913\n(+1 0.258)"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("G11 — the crush mechanism (T194): the first wash step at "
                 "three cons roots (CPU, eval-only)", fontsize=12)

    ax = axs[0, 0]
    xs = np.arange(len(tags))
    settled = [cells[t]["landing"]["settled_plus1"] for t in tags]
    committed = [cells[t]["landing"]["committed_plus1"] for t in tags]
    raws = [cells[t]["landing"]["raw_plus1"] for t in tags]
    roots = [cells[t]["root"]["gm12"] for t in tags]
    ax.bar(xs - 0.27, roots, 0.25, label="root g-12", color="#bbbbbb")
    ax.bar(xs, committed, 0.25, label="committed +1 (settled)",
           color="#2b6cb0")
    ax.bar(xs + 0.27, settled, 0.25, label="rebuilt settled +1",
           color="#63b3ed")
    ax.plot(xs, raws, "r_", ms=18, label="raw landing (C-arm read)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[t] for t in tags])
    ax.set_ylabel("light g-12")
    ax.set_title("(a) the crush, reproduced: settled +1 vs committed")
    ax.legend(fontsize=7)

    ax = axs[0, 1]
    pe = [cells[t]["alignment"]["primary_cos_g0_neg_s0"] for t in tags]
    de = [cells[t]["alignment"]["cos_delta_neg_s0"] for t in tags]
    dp = [cells[t]["alignment"]["cos_delta_pos_s0"] for t in tags]
    ax.bar(xs - 0.25, pe, 0.25, label="cos(g$_0$, -s$_0$) PRIMARY",
           color="#c05621")
    ax.bar(xs, de, 0.25, label="cos($\\Delta_1$, -s$_0$)", color="#f6ad55")
    ax.bar(xs + 0.25, dp, 0.25, label="cos($\\Delta_1$, +s$_0$)",
           color="#fbd38d")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[t] for t in tags])
    ax.set_ylabel("fp64 cosine")
    ax.set_title("(b) the erase-alignment reads (both orientations)")
    ax.legend(fontsize=7)

    ax = axs[1, 0]
    for t in tags:
        ks = sorted(cells[t]["composition"]["overlap_by_k"])
        ys = [cells[t]["composition"]["overlap_by_k"][k] for k in ks]
        ax.plot(ks, ys, "o-", label=short[t].replace("\n", " "))
    ax.axhline(K_PRIMARY / GB.G1B_PARAMS, color="r", ls=":",
               label="random null k/P (at k=2000)")
    ax.set_xscale("log")
    ax.set_xlabel("k (top-|delta| n top-|s$_0$| set size)")
    ax.set_ylabel("overlap fraction")
    ax.set_title("(c) the carrier overlap ladder")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    top_tensors = sorted(
        cells[LOCKED_TAG]["per_tensor"].items(),
        key=lambda kv: -kv[1]["delta_l2_frac"])[:8]
    names = [n.replace(".weight", "").replace("h.", "L")
             for n, _ in top_tensors]
    xs2 = np.arange(len(top_tensors))
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        ax.bar(xs2 + i * w - 0.4 + w / 2,
               [cells[t]["per_tensor"][n]["delta_l2_frac"]
                for n, _ in top_tensors], w,
               label=short[t].replace("\n", " "))
    ax.set_xticks(xs2)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("L2 mass fraction of $\\Delta_1$")
    ax.set_title("(d) the first delta's per-tensor mass")
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RD / "crush_mech.png", dpi=130)
    log(f"[chart] saved {RD / 'crush_mech.png'}")

    # ---- the final write -----------------------------------------------------
    METRICS["honesty_reflex"] = {
        "n1_per_root": "n=1 per root (three objects); every number is a "
                       "single draw's read; no seed replicates here",
        "carrier_proxy": "the carrier set is top-|s_0| (the e204 sensitivity "
                         "top-k as the carrier PROXY) — the directly "
                         "measured support does not exist at these roots; "
                         "k=2000 is this cell's frozen choice",
        "alignment_class": "alignment/composition reads cannot establish "
                           "causality (the T157 class): no intervention "
                           "here; a firing NAMES where to intervene next "
                           "(e.g. a g_0-minus-its-s_0-component step), it "
                           "does not prove the mechanism",
        "root_strength_confound": "the roots' g-12 differ (0.9156 / 0.8575 / "
                                  "0.9682); T178's strength-softening law is "
                                  "a live confound co-noted in deviations",
        "open_bits": "what the jitter-replay leaves: directional, spatial, "
                     "or nonlinear — the verdict names the readable part; "
                     "the intervention cell is the natural successor",
    }
    METRICS["reference"] = {
        "plus1_ledger": {"locked_g1b": 0.9451885223388672,
                         "g1e_10912": 0.2719059884548187,
                         "g1f_10913": 0.25772199034690857,
                         "wash_root_family": "0.82-0.96 (g1bR 0.9623/0.9099; "
                                             "g1c 0.8214; g1d base 0.4820 "
                                             "observed-unadjudicated)"},
        "sources": ["runs/g1b/metrics.json", "runs/g1e/metrics.json",
                    "runs/g1f/metrics.json", "runs/g1bR/metrics.json",
                    "THINKING T194/T181",
                    "lab/e204_support.py (the sensitivity convention)",
                    "lab/e194_sign_front.py (the fact_grad convention)",
                    "lab/g1_anchored_ball.py (the wall arithmetic)"],
        "first_step_disp_all_roots": 1.6542880535125732,
    }
    METRICS["owner_envelope"]["device_events"] = device_events
    METRICS["timing"] = {"total_s": round(time.time() - T0, 1)}
    METRICS["config"] = {"smoke": SMOKE, "torch": torch.__version__,
                         "threads": torch.get_num_threads(),
                         "params": GB.G1B_PARAMS,
                         "device": "cpu-forced (CUDA_VISIBLE_DEVICES=-1)",
                         "wash_seed": WASH_SEED, "R": R_CLAIM,
                         "lr": LR_ADAMW}
    write_partial("final")
    log(f"DONE — verdict {verdict} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
