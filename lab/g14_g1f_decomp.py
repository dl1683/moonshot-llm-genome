"""G14 — THE g1f CRUSH DECOMPOSITION (T200's named successor).

WHY. g13 (T200) closed the symmetry honestly: the crush is
DELTA-CARRIED in both directions (n=2), but the s_0-decomposition's
lethality is DRAW-SPECIFIC — at g1e the crush was s_0-carried (the
delta-form removal spares, 0.708; delta.s0 -0.0565) while at g1f it
is s_0-INDEPENDENT (the same removal reads 0.295, still a crush;
delta.s0 -0.0188, 3x smaller). T200's named successor asks the open
question: WHAT carries the g1f crush? This cell decomposes g1f's own
AdamW delta into nested components against the family's committed
wash-span and transplants/removes each at g1f's root.

BUILDS ON (directive 1): g13 (the six-cell table + the g1f replicate
frame, committed in runs/g13/metrics.json; the settled ARMED-twin +1
read), g12 (the four-cell intervention table + the delta-form removal
co-read), g11 (the three-root geometry + the G_LAND instrument),
g1e/g1f (the cons redraws + their resume ckpts carrying the COMMITTED
first-step deltas), e211 (THE WASH-SPAN: the locked e131 root's own
contiguous 20-step unwalled AdamW wash, seed-10902 stream = g1b's
C-arm, Gram-SVD fp64 top-20 — this run's "g11's span", the
digit-swap reading disclosed below), e212 (the span reproduction
gate: PR + all 20 SVs), e204 (the s_0 convention, FD-sign-gated),
e194/e195 (the fact_grad arithmetic), g1b (the C-arm traj rows 1..20,
the pristine history anchors), g1/g1b (the wall arithmetic:
commit(R=0.7), the ARMED-twin settled +1 read). WHAT IS NEW: the
component ladder (span / span-orthogonal / top-3-SV / the s_0-split
of the span projection) — never run before; every prior removal
removed exactly ONE unit direction (s_0); this ladder removes
SUBSPACES.

THE CELL (single projected steps at g1f's root, g12's machinery
verbatim; the component ladder): decompose g1f's own AdamW delta
(the COMMITTED deltas[1], the crush carrier) into nested components
and transplant each (each renormalized to the FULL step's L2,
projected at R=0.7, the standard +1 read):
  (1) the SPAN component — the delta projected onto the wash-span's
      top-20 SVD subspace (e211's committed pristine span);
  (2) the SPAN-ORTHOGONAL component — the complement;
  (3) the TOP-3-SV component — the largest three span directions
      alone;
  (4) the s_0-SPAN split — the delta's span projection further split:
      the s_0-direction part vs the rest.
THE LADDER'S PURPOSE: WHICH sub-component's removal spares g1f's
root the way s_0-removal spared g1e's?

REGISTERED BARS (frozen here, before compute; the dispatch letter
VERBATIM; no bar shopping):
  COMPONENT-NAMED: "fires if removing exactly one ladder component
      spares g1f (>= 0.5) while removing the others does not — the
      g1f crush's carrier named; the mechanism table complete at both
      draws with DIFFERENT carriers (the per-draw structure mapped)."
  MULTI-COMPONENT: "fires if several components each carry part
      (removing any one still crushes) — the g1f crush is
      distributed; the per-draw structure is dimensional, not
      componential; reported with the ladder verbatim."
  GRADED: "any partial — the tables verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses,
they do not move the bars):
  * THE +1 READ is the settled ARMED-twin read at the projected
    landing (g11's G_LAND instrument, g12/g13 verbatim: a
    CommittedGPT anchored AT THE ROOT with wall R=0.7, loaded with
    the raw outside-ball endpoint, the first forward settles onto the
    ball in place). CRUSH = settled +1 < 0.5; SPARE >= 0.5.
  * THE SPAN ("g11's span", the reading DISCLOSED here): no span
    object exists in runs/g11 — the lab's committed wash-span is
    e211's PRISTINE span (the g/e digit-swap reading, frozen), and a
    span built from g1f's OWN wash history would be DEGENERATE for
    this cell (the decomposed delta IS that history's first row —
    its span projection would be the identity; named and rejected).
    The span = the locked e131 root's own contiguous 20-step
    UNWALLED AdamW wash (lr 1e-3, betas (0.9, 0.95), wd 0.1, clip
    1.0, the seed-10902 stream = g1b's C-arm arithmetic, e211
    verbatim), Gram-SVD fp64 (e_chart/e193b/e205/e209/e211's
    svd_basis), top-20 right-singular subspace (rank 20); REBUILT
    in-run and GATED: per-step displacement L2 vs g1b's committed
    C-arm rows 1..20 (tol 5e-3, e211's G_HIST), x1 md5 vs e185's
    stored hash, span PR + all 20 SVs vs e211's committed pristine
    span (tol 1e-5 rel; e212's 1e-6 co-reported).
  * THE DELTA = g1f's COMMITTED resume deltas[1] (the run's own
    vector; the in-run CPU rebuild is the gate, cos > 0.999 + rel L2
    < 5%, e193's G_S1CK). s_0 = the unit fact-sensitivity at g1f's
    root (e204's convention, FD-sign-gated HARD at eps {0.05, 0.02}).
  * THE COMPONENTS (nested projections of the one delta): V = the
    span basis rows (unit, orthonormal); C_span = P_V(delta);
    C_spanorth = delta - C_span; C_top3 = P_V3(delta) (the largest
    three SV directions); C_s0part = (C_span . s_0) s_0; C_spanrest
    = C_span - C_s0part. Arithmetic gates: C_span + C_spanorth ==
    delta (1e-4 rel); C_top3 inside the span; C_s0part orthogonal to
    C_spanrest; the Pythagoras checks recorded.
  * THE ARMS: TRANSPLANT each component (renormalized to L2 =
    |delta| — the NORM-MATCHING ASSERTED, 1e-4 rel) and REMOVE each
    (delta - C_i, renormalized the same way). Removing the span
    component IS transplanting the span-orthogonal one (the vectors
    coincide exactly; asserted numerically, computed once) and vice
    versa. Disclosed: the renormalization and the wall projection
    are positive-constant rescalings — each landing is theta_root +
    R*unit(arm) exactly.
  * ADJUDICATION (the removal arms, five): R_span (= delta - C_span
    = C_spanorth), R_spanorth (= delta - C_spanorth = C_span),
    R_top3 = delta - C_top3, R_s0part = delta - C_s0part,
    R_spanrest = delta - C_spanrest. n_spared = #(settled +1 >= 0.5).
    COMPONENT-NAMED iff n_spared == 1 (the different-carriers clause
    holds by construction: g13 already read the whole-delta s_0
    removal at g1f as a crush, so any sparing removal is a carrier
    g1e did not have). MULTI-COMPONENT iff n_spared == 0 (removing
    any one still crushes — the bar's own parenthetical). GRADED iff
    n_spared >= 2 (any partial — the redundant-carriage reading;
    iff-exclusive, g12's convention). A degenerate removal arm (|delta
    - C_i| < 1e-6 * |delta|) is read CRUSH-BY-CONSTRUCTION (removing
    nothing = the own-delta reference 0.2577 < 0.5); a degenerate
    transplant arm is recorded DEGENERATE, never adjudicated.
  * REFERENCES (provenance, tol 0.02): g1f's own-delta landing must
    reproduce the committed +1 0.2577; the g13 delta-form removal
    (delta - (delta.s_0)s_0, renormalized) must reproduce 0.2951 —
    it anchors the removal family's arithmetic to the committed
    record. The g12 four-cell table and the g13 six-cell table are
    LOADED (gated COMPLETE/GRADED).
  * CO-READS (never adjudicated): the raw outside-ball endpoint read
    per arm (the C-arm convention); the first-order ledger (s_raw .
    displacement vs the actual settled dlog); the span-vs-delta
    geometry (|C_span|/|delta|, cos(delta, v_i) ladder, the s_0
    in-span fraction); the transplant arms (the bars speak of
    removals; the transplants of the two complementary rungs ARE the
    removals, asserted).
  * Hard-gate failure (a root/stream/anchor/FD/first-step/span/
    component-arithmetic/reference gate) => the record completes
    with verdict TEXTURE (gate failure), nothing adjudicated.

PRE-DISPATCH CHECKS (Rule 12): the delta's/root's provenance (g12/
g13's chain verbatim: root load + dial vs the committed 0.9682 +
meta experiment/cons_seed + the resume sds[1] anchor == the root's
commit(R)); the stream gate (x1 md5 vs the g1f resume x_hashes[1] +
e185's stored hash); the first-step gate (rebuild vs the COMMITTED
deltas[1]); the FD sign check HARD at g1f; the span's provenance
(e211's committed PR + 20-SV spectrum + g1b's C-arm rows 1..20); the
component arithmetic registered (each arm the same L2 as the full
step — the norm-matching asserted); nothing guaranteed — the
openness is the point.

REGISTERED PREDICTION (frozen before compute): the honest prior is
TWO-SIDED, modal landing MULTI-COMPONENT. FOR the dimensional
reading: g13's first-order ledger predicted neither sign nor scale
(the actual settled dlog -1.75 vs first-order +8e-6 at the g1f
removal), the raw ~1.65-L2 endpoint crushes everywhere it was read
(0.0003), and g1f's delta.s0 is 3x smaller than g1e's — the damage
looks spread across the step's directions rather than packed on one
subspace handle. AGAINST a clean COMPONENT-NAMED firing: the delta
class is 55%-shared and wall-organized; if the crush concentrates in
the family's wash-span (the locked root's own movement subspace),
R_spanorth (== transplanting C_span) could spare alone; the s_0-part
arm is predicted NOT to spare (g13's whole-delta s_0 removal already
crushed at 0.295). FALSIFIER of the dimensional prior: exactly one
removal spares. No bar shopping.

WHAT THIS CELL GUARANTEES: nothing — n=1 per arm (single draws,
single single-step interventions at one root); the NESTED-COMPONENT
CAVEAT: the ladder's components are nested projections of ONE delta,
each removal renormalizes a 2.74M-dim remainder to the full step's
L2 — a spared removal means "this remainder DIRECTION is non-lethal
at R=0.7", not that the removed piece uniquely carried the damage
(a distributed kill can masquerade as any-removal-crushes — the
MULTI-COMPONENT bar's own reading); the span is ANOTHER ROOT's wash
(the locked root's; the family's reference span — a g1f-own span is
degenerate by construction, disclosed); the read is one
behavior-level readout (the +1 g-12 light battery, 60 prompts); no
wash continuation (the +2 re-capture phase is not tested here).

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never
claimed; g13 measured the same bundle at ~30s CPU; the 20-step span
history adds ~1 min), threads 4; the double-polled load checks
recorded at start (recorded, not gating); PROGRESSIVE metrics.json
writes after every phase.

Outputs: runs/g14/{metrics.json (PROGRESSIVE), g1f_decomp.png}. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python g14_g1f_decomp.py    (G14_SMOKE=1 shakedown)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (g11/g12/g13)

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

SMOKE = os.environ.get("G14_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "g14 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ======================================================================
# THE CONFIG (the g1f root + the locked root for the span + constants)
# ======================================================================
WASH_SEED = 10902                  # HELD (the licensed stream, verbatim)
R_CLAIM = 0.7                      # RAW L2 (the 2.74M convention)
LR_ADAMW = G1.FT_LR                # 1e-3 (the t=0 recipe)
CKPT_DIR = GB.CKPT_DIR
E185_X1_MD5 = "1ea27bffde6c4a53be8badf5ab453d64"   # e185's stored step-1 batch
                                                   # md5 (net-independent)
G12_METRICS = CKPT_DIR.parent / "g12" / "metrics.json"
G13_METRICS = CKPT_DIR.parent / "g13" / "metrics.json"
G1B_METRICS = CKPT_DIR.parent / "g1b" / "metrics.json"
E211_METRICS = CKPT_DIR.parent / "e211" / "metrics.json"

# ---- the g1f root spec (g13's ROOTS[2] verbatim) -----------------------------
G1F = {"tag": "g1f_10913", "ckpt": "g1f_root.pt",
       "cons_seed": 10913, "lineage": "the SECOND cons redraw (the n=2 breach)",
       "committed_root_gm12": 0.9682114720344543,
       "committed_plus1": 0.25772199034690857,
       "committed_first_disp": 1.6543004512786865,
       "committed_delta": "g1f_W1_resume.pt:deltas[1]",
       "resume_ck": "g1f_W1_resume.pt"}
LOCKED_CK = "e131_consolidated_e113.pt"       # the span's root (e211's pristine)

# ---- the committed reference numbers this run gates against ------------------
G13_CELLH_DFORM = 0.29514944553375244         # g13's delta-form removal at g1f
G13_CELLH_DELTA_DOT_S0 = -0.018764491474803607
G1E_DFORM_COREAD = 0.7075417041778564         # g12's co-read (the contrast)
G1E_DELTA_DOT_S0 = -0.05652995055824213

# e211's committed pristine span (the reproduction target; runs/e211/metrics
# .json roots.P.span_primary)
E211_PRISTINE_PR = 4.117363094870968
E211_PRISTINE_SV = [
    1.9597355390295585, 1.4676190745815085, 1.0840231000069138,
    0.8516531991871176, 0.6754297556601792, 0.5446478048652162,
    0.4751903432789083, 0.37694177303028514, 0.3406220016987602,
    0.2792117231232169, 0.23479511170828862, 0.2120952460349488,
    0.1816930659234477, 0.16720944597868573, 0.15851014009334544,
    0.12365014301894246, 0.11241917367756023, 0.10215793762375433,
    0.09377184368376072, 0.08634923178214998]

# ---- the frozen adjudication constants --------------------------------------
CRUSH_BAR = 0.5                         # the letter's threshold (+1 < 0.5 = crush)
SPAN_STEPS = 4 if SMOKE else 20         # e211's contiguous wash history
FD_EPS = (0.05, 0.02) if not SMOKE else (0.05,)   # e204's probe sizes (L2)
TOL_DIAL = 0.02                         # root dial vs committed (texture tier)
TOL_LANDING = 0.02                      # settled +1 read vs committed
TOL_DISP = 0.02                         # first-step disp vs committed 1.6543
S1CK_COS = 0.999                        # e193's G_S1CK convention
S1CK_REL = 0.05
TOL_ORTH = 1e-4                         # the removals' orthogonality gate
TOL_PROJ = 1e-5                         # the projection-arithmetic gate
TOL_NORM = 1e-4                         # the norm-matching gate (rel)
TOL_HIST = 5e-3                         # pristine per-step L2 vs g1b CUDA rows
TOL_SPAN = 1e-5                         # span PR/SV reproduction (rel; e212 1e-6)
DEGEN_FLOOR = 1e-6                      # degenerate-arm floor (rel to |delta|)

REGISTERED_BARS = {
    "COMPONENT_NAMED": ("COMPONENT-NAMED: \"fires if removing exactly one "
                        "ladder component spares g1f (>= 0.5) while removing "
                        "the others does not — the g1f crush's carrier named; "
                        "the mechanism table complete at both draws with "
                        "DIFFERENT carriers (the per-draw structure mapped).\""),
    "MULTI_COMPONENT": ("MULTI-COMPONENT: \"fires if several components each "
                        "carry part (removing any one still crushes) — the "
                        "g1f crush is distributed; the per-draw structure is "
                        "dimensional, not componential; reported with the "
                        "ladder verbatim.\""),
    "GRADED": "GRADED: \"any partial — the tables verbatim.\"",
    "operationalizations": (
        "frozen BEFORE compute (they fix the clauses, they do not move the "
        "bars): the +1 read is the SETTLED ARMED-twin read at the projected "
        "landing (g11's G_LAND instrument, g12/g13 verbatim — a CommittedGPT "
        "anchored at the root with wall R=0.7, loaded with the raw "
        "outside-ball endpoint, the first forward settles onto the ball in "
        "place). CRUSH = settled +1 < 0.5; SPARE >= 0.5. THE SPAN ('g11's "
        "span', the reading DISCLOSED): no span object exists in runs/g11 — "
        "the lab's committed wash-span is e211's PRISTINE span (the g/e "
        "digit-swap reading, frozen), and a span built from g1f's OWN wash "
        "history would be DEGENERATE (the decomposed delta IS that history's "
        "first row — its span projection would be the identity; named and "
        "rejected); the span = the locked e131 root's own contiguous 20-step "
        "UNWALLED AdamW wash (lr 1e-3, betas (0.9,0.95), wd 0.1, clip 1.0, "
        "the seed-10902 stream = g1b's C-arm arithmetic, e211 verbatim), "
        "Gram-SVD fp64 top-20 right-singular subspace, REBUILT and GATED "
        "(per-step L2 vs g1b's committed C-arm rows 1..20 tol 5e-3; x1 md5 "
        "vs e185; PR + all 20 SVs vs e211's committed pristine span tol "
        "1e-5 rel). THE DELTA = g1f's COMMITTED resume deltas[1] (rebuild "
        "gate cos > 0.999 + rel L2 < 5%). s_0 = the unit fact-sensitivity at "
        "g1f (e204's convention, FD-sign-gated HARD at eps {0.05, 0.02}). "
        "THE COMPONENTS (nested projections of the one delta): C_span = "
        "P_V(delta); C_spanorth = delta - C_span; C_top3 = P_V3(delta); "
        "C_s0part = (C_span . s_0) s_0; C_spanrest = C_span - C_s0part; "
        "arithmetic gated (partition, orthogonality, containment). THE ARMS: "
        "TRANSPLANT each component and REMOVE each (delta - C_i), every arm "
        "renormalized to L2 = |delta| — the NORM-MATCHING ASSERTED (1e-4 "
        "rel); removing the span component IS transplanting the "
        "span-orthogonal one (the vectors coincide; asserted numerically). "
        "Disclosed: the renormalization and the wall projection are "
        "positive-constant rescalings — each landing is theta_root + "
        "R*unit(arm) exactly. ADJUDICATION over the five REMOVAL arms "
        "{R_span, R_spanorth, R_top3, R_s0part, R_spanrest}: n_spared = "
        "#(settled +1 >= 0.5); COMPONENT-NAMED iff n_spared == 1 (the "
        "different-carriers clause holds by construction — g13 already read "
        "the whole-delta s_0 removal at g1f as a crush); MULTI-COMPONENT iff "
        "n_spared == 0 (removing any one still crushes — the bar's own "
        "parenthetical); GRADED iff n_spared >= 2 (the redundant-carriage "
        "reading; iff-exclusive, g12's convention). A degenerate removal arm "
        "(|delta - C_i| < 1e-6*|delta|) is read CRUSH-BY-CONSTRUCTION "
        "(removing nothing = the own-delta reference 0.2577 < 0.5); a "
        "degenerate transplant arm is recorded DEGENERATE, never adjudicated. "
        "REFERENCES (tol 0.02): the own-delta landing vs committed 0.2577; "
        "the g13 delta-form removal re-run vs 0.2951; the g12/g13 tables "
        "LOADED (gated COMPLETE/GRADED). CO-READS (never adjudicated): the "
        "raw outside-ball endpoint read per arm; the first-order ledger; the "
        "span-vs-delta geometry; the transplant arms. Hard-gate failure => "
        "the record completes, verdict TEXTURE (gate failure), nothing "
        "adjudicated."),
    "registered_prediction": (
        "frozen before compute. THE HONEST PRIOR IS TWO-SIDED, THE MODAL "
        "LANDING MULTI-COMPONENT. FOR the dimensional reading: g13's "
        "first-order ledger predicted neither sign nor scale (actual settled "
        "dlog -1.75 vs first-order +8e-6 at the g1f removal), the raw "
        "~1.65-L2 endpoint crushes everywhere it was read (0.0003), and "
        "g1f's delta.s0 is 3x smaller than g1e's — the damage looks spread "
        "across the step's directions rather than packed on one subspace "
        "handle. AGAINST a clean COMPONENT-NAMED firing: the delta class is "
        "55%-shared and wall-organized; if the crush concentrates in the "
        "family's wash-span (the locked root's own movement subspace), "
        "R_spanorth (== transplanting C_span) could spare alone; the s_0-part "
        "arm is predicted NOT to spare (g13's whole-delta s_0 removal "
        "already crushed at 0.295). FALSIFIER of the dimensional prior: "
        "exactly one removal spares. No bar shopping."),
    "registration": ("the dispatch's registration IS the registration (the "
                     "bars quoted verbatim here and in the module docstring, "
                     "frozen before compute). Adjudicate against exactly "
                     "this; no bar shopping."),
}

deviations: list[str] = [
    "CPU-ONLY, FORCED (g11/g12/g13's convention): CUDA_VISIBLE_DEVICES=-1 "
    "before torch; the GPU is never claimed; the double-polled load checks "
    "recorded at start (recorded, not gating).",
    "'G11'S SPAN' READ AS E211'S PRISTINE SPAN (the g/e digit-swap reading, "
    "DISCLOSED + frozen): no span object exists anywhere in runs/g11 (the "
    "g11 record is first-step alignments/composition, no SVD); the lab's "
    "committed wash-span is e211's pristine span at the locked e131 root — "
    "REBUILT here under e211/e212's own gates. A g1f-own wash-history span "
    "is DEGENERATE for this cell (the delta is itself the history's first "
    "row; the span projection would be the identity) — named and rejected "
    "in the operationalization.",
    "THE SPAN IS ANOTHER ROOT'S WASH (the locked e131 root's, the family's "
    "reference span): the ladder therefore reads 'does the g1f crush live "
    "in the family's wash-span directions or orthogonal to them', not "
    "'inside g1f's own future movement' — the honest scope of this cell.",
    "THE g12 FOUR-CELL TABLE + THE g13 SIX-CELL TABLE ARE LOADED, NOT "
    "RECOMPUTED (the dispatch's read-first artifacts): the two own-object "
    "reference cells (own-delta landing 0.2577; the g13 delta-form removal "
    "0.2951) are RE-RUN solely as the provenance gates for the delta "
    "object and the removal arithmetic (tol 0.02, g13's own convention).",
    "THE DELTA IS THE COMMITTED deltas[1] (the run's own vector); the "
    "in-run CPU rebuild is the gate (cos > 0.999 + rel L2 < 5%, e193's "
    "G_S1CK) — the committed run trained on GPU, the rebuild on CPU; "
    "g13 measured the same pair at cos 0.99999993.",
    "NO WASH CONTINUATION: each intervention is a SINGLE first-step-"
    "geometry application + the standard +1 read (the letter's cell); the "
    "+2 re-capture phase (the wall's own) is not tested here.",
    "n=1 PER ARM (single draws, single single-step interventions at one "
    "root); the reads are single applications of single objects.",
    "torch threads 4 (shared machine; g1/g1b's import resets to 8 — reset "
    "after import).",
    "Smoke mode trims: 4-step span history (top-4), FD eps {0.05}; the SV "
    "gate vs e211's 20-SV commit is SKIPPED in smoke; nothing adjudicated.",
]

device_events: list[dict] = []
RD: Path = None                     # set in main (run_dir)
METRICS: dict = {}
_progressive = {"n": 0, "phases": []}


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


def dot64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 dot product."""
    return float(torch.dot(a.double(), b.double()))


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
    g-12 battery (the readout currency of the FD sign check + the
    first-order ledger)."""
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
    arithmetic; e204/g11/g12/g13's copy): gradient of the fact battery's
    mean log p(Z) readout at the net's CURRENT weights."""
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


def participation_ratio(sv) -> float:
    """e211's PR (fp64)."""
    s2 = sv if isinstance(sv, torch.Tensor) else torch.tensor(
        sv, dtype=torch.float64)
    s2 = (s2.double() ** 2)
    return float(s2.sum() ** 2 / (s2 @ s2 + 1e-30))


def svd_basis(H: torch.Tensor) -> dict:
    """e_chart/e193b/e205/e209/e211's Gram-based right-singular basis (fp64)."""
    H64 = H.to(torch.float64)
    G = (H64 @ H64.T)
    evals, evecs = torch.linalg.eigh(G)     # ascending
    evals = torch.flip(evals, dims=(0,)).clamp(min=0)
    evecs = torch.flip(evecs, dims=(1,))
    sv = torch.sqrt(evals)
    pr = participation_ratio(sv) if float(sv[0]) > 0 else 0.0
    rank_eff = int((sv > sv[0] * 1e-7).sum())
    rows = []
    for i in range(rank_eff):
        v = H64.T @ evecs[:, i]
        rows.append((v / sv[i].clamp(min=1e-30)).to(torch.float32))
    Vp = torch.stack(rows) if rows else torch.empty(0, H.shape[1])
    return {"sv": sv, "Vp": Vp, "pr": pr, "rank_eff": rank_eff,
            "cond": float(sv[0] / sv[-1].clamp(min=1e-30))}


def write_partial(phase: str) -> None:
    """PROGRESSIVE metrics (the outage lesson): metrics.json after every
    phase; bookkeeping must never kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    METRICS.update({
        "experiment": "g14_g1f_decomp", "date": common.now_iso(),
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
                          ("COMPONENT_NAMED", "MULTI_COMPONENT", "GRADED")},
        "verdict": "TEXTURE (gate failure)",
        "reason": reason,
        "note": "the registered composite: hard-gate failure => the record "
                "completes with verdict TEXTURE, nothing adjudicated",
    }
    log(f"[TEXTURE] hard gate failure: {reason}")
    write_partial("texture (gate failure)")


def settled_plus1(root_sd: dict, theta0: torch.Tensor, delta: torch.Tensor,
                  gm12_ids: torch.Tensor, zid: int) -> dict:
    """THE STANDARD +1 READ (g11's G_LAND instrument, g12/g13 verbatim): the
    ARMED twin — a CommittedGPT anchored AT THE ROOT (commit R=0.7), loaded
    with the raw outside-ball endpoint theta0 + delta; the first forward
    settles onto the ball surface in place; read the g-12 battery. The raw
    (uncommitted-twin) endpoint read rides as the C-arm co-read; the
    projection-arithmetic gate rides with it."""
    d_norm = float(torch.norm(delta))
    landing = theta0 + (R_CLAIM / d_norm) * delta
    evl = G1.CommittedGPT(GB.G1B_CFG)
    evl.load_state_dict(root_sd)
    evl.eval()
    evl.commit(R_CLAIM)
    load_flat(evl, theta0 + delta)
    cell = G1.battery_cell(evl, gm12_ids, zid)
    settled = flat_params(evl)
    proj_arith_d = float((settled - landing).abs().max())
    settled_logpz = fact_readout(evl, gm12_ids, zid)   # the ledger's log read
    del evl
    raw = G1.CommittedGPT(GB.G1B_CFG)
    raw.load_state_dict(root_sd)
    load_flat(raw, theta0 + delta)
    raw_read = G1.battery_cell(raw, gm12_ids, zid)["mean_pz"]
    del raw
    return {
        "settled_plus1": cell["mean_pz"],
        "settled_median_pz": cell["median_pz"],
        "settled_frac_ge_0p5": cell["frac_pz_ge_0.5"],
        "raw_plus1_coreported": raw_read,
        "settled_mean_log_pz": settled_logpz,
        "delta_l2": d_norm, "R": R_CLAIM,
        "rescale_ratio": R_CLAIM / d_norm,
        "proj_arithmetic_maxdiff": proj_arith_d,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    global RD
    RD = run_dir("g14_smoke" if SMOKE else "g14")
    log(f"G14 THE g1f CRUSH DECOMPOSITION (T200's successor) "
        f"(smoke={SMOKE}) -> {RD}")
    # the envelope load-check, DOUBLE-POLLED (recorded, not gating —
    # CPU-only policy, no GPU launch ever happens)
    polls = []
    for k in (1, 2):
        s = common.gpu_status()
        polls.append(s)
        try:
            common._log_envelope_poll(f"g14:load-check-{k}(cpu-only-policy)",
                                      s["util"], s["temp"], True)
        except Exception:
            pass
        if k == 1:
            time.sleep(2.0)             # the double poll's gap
    device_events.append({
        "tag": "g14", "event": "LOAD CHECK x2 (CPU-only policy; GPU never "
        "claimed)", "status_polls": polls,
        "note": "the letter's envelope: eval-only cells + one 20-step span "
                "history, all CPU (2.74M; g13 measured the same bundle "
                "~30s; the span adds ~1min); no GPU launch"})
    log("[envelope] double load-check recorded: "
        + " | ".join(f"poll{ i+1 } util {p['util']:.0f}% temp {p['temp']:.0f}C"
                     for i, p in enumerate(polls))
        + " — CPU-only policy (no GPU launch)")

    METRICS.update({
        "design": (
            "G14 THE g1f CRUSH DECOMPOSITION (T200's named successor): "
            "g1f's own AdamW delta (the COMMITTED deltas[1], the crush "
            "carrier) decomposed into nested components against e211's "
            "committed pristine wash-span (top-20 SVD subspace; 'g11's "
            "span' read as e211's — the digit-swap, disclosed) and each "
            "component TRANSPLANTED and REMOVED at g1f's root (each "
            "renormalized to the full step's L2, projected at R=0.7, the "
            "standard settled ARMED-twin +1 read, g12/g13's machinery "
            "verbatim). THE LADDER: (1) SPAN (the delta's span projection); "
            "(2) SPAN-ORTHOGONAL (the complement); (3) TOP-3-SV (the "
            "largest three span directions); (4) the s_0-SPAN split (the "
            "span projection's s_0-direction part vs the rest). THE "
            "QUESTION: WHICH sub-component's removal spares g1f's root the "
            "way s_0-removal spared g1e's (0.708)? — against: g13 already "
            "read the whole-delta s_0 removal at g1f as a CRUSH (0.295, "
            "delta.s0 -0.0188 vs g1e's -0.0565)."),
        "registered": REGISTERED_BARS,
        "deviations": deviations,
        "smoke": SMOKE,
        "owner_envelope": {
            "policy": "CPU-ONLY forced (CUDA_VISIBLE_DEVICES=-1); threads "
                      "4; no GPU launch; double-polled load checks recorded",
            "device_events": device_events},
    })
    write_partial("start (bars + arithmetic + prediction registered before "
                  "compute)")

    # ---------------- the loaded tables (the read-first artifacts) -----------
    g12m = json.loads(G12_METRICS.read_text())
    g13m = json.loads(G13_METRICS.read_text())
    g1bm = json.loads(G1B_METRICS.read_text())
    e211m = json.loads(E211_METRICS.read_text())
    e211_span_c = e211m["roots"]["P"]["span_primary"]
    g12_table = g12m["adjudication"]["four_cell_table"]
    g13_six = g13m["adjudication"]["six_cell_table"]
    G_LOADS = {
        "g12": {"source": "runs/g12/metrics.json",
                "status": g12m.get("status"),
                "verdict": g12m["adjudication"]["verdict"],
                "four_cell_table": g12_table,
                "pass": bool(g12m.get("status") == "COMPLETE"
                             and g12m["adjudication"]["verdict"] == "GRADED")},
        "g13": {"source": "runs/g13/metrics.json",
                "status": g13m.get("status"),
                "verdict": g13m["adjudication"]["verdict"],
                "six_cell_table": g13_six,
                "g1f_replicate_frame":
                    g13m["adjudication"]["g1f_replicate_frame"],
                "pass": bool(g13m.get("status") == "COMPLETE"
                             and g13m["adjudication"]["verdict"] == "GRADED")},
        "e211": {"source": "runs/e211/metrics.json",
                 "status": e211m.get("status"),
                 "pristine_span": {"pr": e211_span_c["participation_ratio"],
                                   "sv": e211_span_c["sv"]},
                 "pass": bool("COMPLETE" in str(e211m.get("status", "")))},
        "g1b": {"source": "runs/g1b/metrics.json",
                "c_arm_traj_rows": len(g1bm["arms"]["C"]["traj"]),
                "pass": bool(len(g1bm["arms"]["C"]["traj"]) >= SPAN_STEPS)},
    }
    METRICS["gates"] = {"G_LOADS": G_LOADS}
    ok_loads = all(v["pass"] for v in G_LOADS.values())
    log(f"G_LOADS: g12 {G_LOADS['g12']['verdict']}, g13 "
        f"{G_LOADS['g13']['verdict']}, e211 span PR "
        f"{G_LOADS['e211']['pristine_span']['pr']:.4f}, g1b C-arm rows "
        f"{G_LOADS['g1b']['c_arm_traj_rows']}: "
        + ("PASS" if ok_loads else "FAIL"))
    if not ok_loads:
        fail_texture("G_LOADS failed — a read-first artifact is not the "
                     "committed COMPLETE/GRADED record")
        return

    # ---------------- protocol rebuild (g1e/g11/g13 VERBATIM) ----------------
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

    # the fact battery (the g-12 install-60 battery)
    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids = bat_ids[-12]
    G_BATTERY = {"shapes": {str(j): list(bat_ids[j].shape) for j in G1.GEOS},
                 "pass": bool(list(gm12_ids.shape) == [60, G1.PRE - 12]),
                 "note": "the g-12 install-60 battery (e194/e195's fact_grad "
                         "convention, e204's carrier, g11/g12/g13's +1 "
                         "readout; shapes 60x118)"}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    # the neutral anchor bank (e170's construction VERBATIM)
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

    # ---------------- the first wash batch (the seed-10902 stream) -----------
    def draw_step1_batch(g: torch.Generator):
        aj = torch.randint(anchor_neutral.shape[0], (G1.ANCH_BS,),
                           generator=g)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    g1 = torch.Generator().manual_seed(WASH_SEED)
    x1, y1 = draw_step1_batch(g1)
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()

    # the g1f resume ckpt: committed stream hash + first delta + anchor
    rc = torch.load(CKPT_DIR / G1F["resume_ck"], map_location="cpu",
                    weights_only=False)
    committed = {
        "stream": rc["x_hashes"].get(1, rc["x_hashes"].get("1")),
        "delta": rc["deltas"][1].float(),
        "wall_R": float(rc["wall_R"]),
        "anch_md5": anch_md5_of(rc["sds"][1]),
    }
    del rc
    G_STREAM = {
        "x1_md5": x1_md5,
        "vs_g1f_resume": bool(committed["stream"] == x1_md5),
        "vs_e185_stored": bool(x1_md5 == E185_X1_MD5),
        "note": "the seed-10902 stream drawn VERBATIM (the e170 bank + the "
                "first batch); md5-gated vs the g1f resume ckpt's own "
                "x_hashes[1] + e185's stored hash",
    }
    G_STREAM["pass"] = bool(G_STREAM["vs_e185_stored"]
                            and G_STREAM["vs_g1f_resume"])
    log(f"G_STREAM: x1 md5 {x1_md5[:10]}.. vs g1f resume "
        f"{G_STREAM['vs_g1f_resume']} vs e185 {G_STREAM['vs_e185_stored']}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    METRICS["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY, "G_STREAM": G_STREAM})
    if not G_STREAM["pass"]:
        fail_texture("G_STREAM failed — the stream diverged from the "
                     "committed run's own x_hashes")
        return
    write_partial("protocol + stream gates")

    # =====================================================================
    # THE g1f ROOT — provenance + s_0 + the first-step rebuild (g13's chain)
    # =====================================================================
    tag = G1F["tag"]
    log("=" * 78)
    log(f"ROOT {tag} — {G1F['lineage']} ({G1F['ckpt']})")
    raw = torch.load(CKPT_DIR / G1F["ckpt"], map_location="cpu",
                     weights_only=False)
    root_sd = {k: v.detach().clone() for k, v in
               (raw["model"] if isinstance(raw, dict)
                and "model" in raw else raw).items()}
    meta = E43.jsonable(raw.get("meta", {})) if isinstance(raw, dict) else {}
    del raw
    net_root = G1.CommittedGPT(GB.G1B_CFG)
    net_root.load_state_dict(root_sd)
    net_root.eval()
    n_params = net_root.num_params()
    theta0 = flat_params(net_root)
    if n_params != GB.G1B_PARAMS:
        METRICS["gates"][f"{tag}__G_ROOT"] = {"params": n_params,
                                              "pass": False}
        fail_texture(f"{tag}: params {n_params} != {GB.G1B_PARAMS}")
        return

    root_gm12 = G1.battery_cell(net_root, gm12_ids, zid)["mean_pz"]
    root_logpz = fact_readout(net_root, gm12_ids, zid)
    meta_ok = bool(meta.get("experiment") == "g1f"
                   and meta.get("cons_seed") == G1F["cons_seed"])
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{G1F['ckpt']}",
        "meta": meta, "params": n_params,
        "body_md5": sd_md5_body(root_sd),
        "gm12": root_gm12, "gm12_committed": G1F["committed_root_gm12"],
        "gm12_d": abs(root_gm12 - G1F["committed_root_gm12"]),
        "root_mean_log_pz": root_logpz,
        "meta_gate": meta_ok,
        "tol": TOL_DIAL,
        "pass": bool(abs(root_gm12 - G1F["committed_root_gm12"]) <= TOL_DIAL
                     and meta_ok),
        "note": "the root loads into the 2,739,072 cfg clean; its "
                "same-instrument g-12 reproduces the committed root cell "
                "0.9682 (g13's G_ROOT verbatim); meta provenance "
                "(experiment + cons_seed)",
    }
    log(f"G_ROOT[{tag}]: gm12 {root_gm12:.4f} vs committed "
        f"{G1F['committed_root_gm12']:.4f} (|d| {G_ROOT['gm12_d']:.2e}), "
        f"meta {'OK' if meta_ok else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    METRICS["gates"][f"{tag}__G_ROOT"] = G_ROOT
    if not G_ROOT["pass"]:
        fail_texture(f"{tag}: root provenance gate FAILED")
        return

    # G-ANCHOR: the resume sds' anchor IS the root
    net_c = G1.CommittedGPT(GB.G1B_CFG)
    net_c.load_state_dict(root_sd)
    net_c.commit(R_CLAIM)
    anch_md5_root = anch_md5_of(net_c.state_dict())
    wall_r_ok = abs(committed["wall_R"] - R_CLAIM) < 1e-6
    md5_ok = bool(anch_md5_root == committed["anch_md5"])
    g_anchor = {
        "form": "the W1 arm's wall anchor (theta_anchor) IS the root — the "
                "anch__ buffers in the committed resume sds[1] must equal "
                "the root ckpt's commit(R)",
        "wall_R_resume": committed["wall_R"],
        "wall_R_expected": R_CLAIM,
        "wall_R_match": bool(wall_r_ok),
        "anch_md5_root_rebuilt": anch_md5_root,
        "anch_md5_resume": committed["anch_md5"],
        "anch_md5_match": md5_ok,
        "pass": bool(wall_r_ok and md5_ok),
    }
    del net_c
    log(f"G-ANCHOR[{tag}]: wall_R {committed['wall_R']:.7f} "
        f"{'match' if wall_r_ok else 'DRIFT'}, anchor md5 "
        f"{'match' if md5_ok else 'DRIFT'}: "
        + ("PASS" if g_anchor["pass"] else "FAIL"))
    METRICS["gates"][f"{tag}__G_ANCHOR"] = g_anchor
    if not g_anchor["pass"]:
        fail_texture(f"{tag}: G-ANCHOR failed — the committed wall's anchor "
                     "is not this root")
        return

    # s_0: the root's OWN fact sensitivity (the e204 convention, HARD)
    s_raw = fact_grad(net_root, gm12_ids, zid)
    s0 = s_raw / torch.norm(s_raw)
    read0 = root_logpz
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
        "convention": "e194/e195's fact_grad VERBATIM (the g-12 install-60 "
                      "battery; matched-point at the root); FD sign check at "
                      f"eps in {tuple(FD_EPS)} (e204's G_SENSDIR)",
        "hard_gate": True,
        "readout_theta0_mean_log_pz": read0,
        "probes": fd,
        "pass": bool(all(fd[f"eps{eps}_monotone"] for eps in FD_EPS)),
        "note": "HARD (this root's s_0 is adjudicated — g13's same gate "
                "PASSED at this exact root)",
    }
    log(f"G_SENSDIR[{tag}] (HARD): read0 {read0:.4f}; FD monotone "
        + "/".join(f"eps{e}:{fd[f'eps{e}_monotone']}" for e in FD_EPS)
        + ": " + ("PASS" if G_SENSDIR["pass"] else "FAIL"))
    METRICS["gates"][f"{tag}__G_SENSDIR"] = G_SENSDIR
    if not G_SENSDIR["pass"]:
        fail_texture(f"{tag}: FD sign check FAILED — s_0 is not the fact's "
                     "sensitivity at this root")
        return

    # the first wash step, REBUILT (g11/g12/g13's arithmetic verbatim)
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
    del logits, loss, net_w, opt

    cd = committed["delta"]
    cos_rb_c = cos64(delta_rebuilt, cd)
    rel_l2 = abs(float(torch.norm(cd)) - d_raw) / d_raw
    G_STEP1 = {"disp_rebuilt": d_raw,
               "disp_committed": G1F["committed_first_disp"],
               "disp_d": abs(d_raw - G1F["committed_first_disp"]),
               "cos_rebuilt_vs_committed": cos_rb_c,
               "rel_l2_vs_committed": rel_l2,
               "primary": "COMMITTED deltas[1] (the run's own vector); "
                          "the rebuild is the gate",
               "pass": bool(
                   abs(d_raw - G1F["committed_first_disp"]) <= TOL_DISP
                   and cos_rb_c > S1CK_COS and rel_l2 < S1CK_REL)}
    log(f"G_STEP1[{tag}]: |delta| {d_raw:.4f} vs committed "
        f"{G1F['committed_first_disp']:.4f} (|d| {G_STEP1['disp_d']:.2e}); "
        f"rebuild-vs-committed cos {cos_rb_c:.6f} rel {rel_l2:.2e}: "
        + ("PASS" if G_STEP1["pass"] else "FAIL"))
    METRICS["gates"][f"{tag}__G_STEP1"] = G_STEP1
    METRICS["gates"][f"{tag}__FIRSTSTEP_TEXTURE"] = {
        "ce_batch": ce1, "preclip_gnorm": preclip_norm,
        "clip_binds": clip_binds,
        "g0_l2_postclip": float(torch.norm(g0)),
        "note": "the first wash step's stream arithmetic (g11/g13's texture "
                "reads)",
    }
    if not G_STEP1["pass"]:
        fail_texture(f"{tag}: first-step gate FAILED")
        return
    delta = cd.clone()                       # THE PRIMARY OBJECT (committed)
    del net_root, delta_rebuilt, theta1
    write_partial(f"root {tag} (gates + s_0 + first-step rebuild)")

    # =====================================================================
    # THE SPAN — e211's pristine wash span, rebuilt + gated
    # =====================================================================
    log("=" * 78)
    log(f"THE SPAN — e211's pristine wash span at the locked root "
        f"({LOCKED_CK}), {SPAN_STEPS} contiguous unwalled steps, seed-10902")
    praw = torch.load(CKPT_DIR / LOCKED_CK, map_location="cpu",
                      weights_only=False)
    locked_sd = {k: v.detach().clone() for k, v in
                 (praw["model"] if isinstance(praw, dict)
                  and "model" in praw else praw).items()}
    del praw
    net_p = G1.CommittedGPT(GB.G1B_CFG)
    net_p.load_state_dict(locked_sd)          # UNCOMMITTED (no wall — e211)
    net_p.train()
    popt = torch.optim.AdamW(net_p.parameters(), lr=LR_ADAMW,
                             betas=(0.9, 0.95), weight_decay=0.1)
    ptheta0 = flat_params(net_p)
    g2 = torch.Generator().manual_seed(WASH_SEED)
    segs, hist_rows = [], []
    for s_wh in range(1, SPAN_STEPS + 1):
        x_, y_ = draw_step1_batch(g2)
        logits_p, _ = net_p(x_)
        lp = F.cross_entropy(logits_p.reshape(-1, logits_p.shape[-1]),
                             y_.reshape(-1))
        popt.zero_grad(set_to_none=True)
        lp.backward()
        torch.nn.utils.clip_grad_norm_(net_p.parameters(), 1.0)
        popt.step()
        th_new = flat_params(net_p)
        segs.append(th_new - ptheta0)
        hist_rows.append({"step": s_wh,
                          "L2": float(torch.norm(th_new - ptheta0)),
                          "ce": float(lp.item())})
        ptheta0 = th_new
    del net_p, popt
    H = torch.stack(segs)

    # the stream gate for the span: its step-1 batch must BE the licensed x1
    g3 = torch.Generator().manual_seed(WASH_SEED)
    sx1, _ = draw_step1_batch(g3)
    span_x1_ok = bool(hashlib.md5(
        sx1.contiguous().numpy().tobytes()).hexdigest() == x1_md5)

    # G_HIST: per-step L2 (+ step-1 CE) vs g1b's committed C-arm rows
    carm = {int(r["step"]): r for r in g1bm["arms"]["C"]["traj"]}
    hist_dev = [abs(hist_rows[s - 1]["L2"] - carm[s]["step_disp"])
                for s in range(1, SPAN_STEPS + 1)]
    ce1_dev = abs(hist_rows[0]["ce"] - carm[1]["ce_batch"])
    G_HIST = {
        "gate": "per-step displacement L2 vs g1b's committed C-arm rows "
                "1..N (tol 5e-3, e211's pristine G_HIST) + step-1 CE",
        "max_L2_dev": max(hist_dev), "ce1_dev": ce1_dev, "tol": TOL_HIST,
        "stream_x1_md5_ok": span_x1_ok,
        "pass": bool(max(hist_dev) < TOL_HIST and ce1_dev < TOL_HIST
                     and span_x1_ok),
        "note": "the pristine history IS g1b's own C-arm stream arithmetic "
                "(e211's gate, CPU rebuild vs the committed GPU rows)",
    }
    log(f"G_HIST: max per-step |dL2| {max(hist_dev):.2e}, ce1 |d| "
        f"{ce1_dev:.2e} (tol {TOL_HIST}): "
        + ("PASS" if G_HIST["pass"] else "FAIL"))
    METRICS["gates"]["G_HIST"] = G_HIST
    if not G_HIST["pass"]:
        fail_texture("G_HIST failed — the rebuilt pristine history diverged "
                     "from g1b's committed C-arm rows")
        return

    basis = svd_basis(H)
    G_SPAN = {
        "n_segments": SPAN_STEPS, "rank_eff": basis["rank_eff"],
        "pr": basis["pr"], "cond": basis["cond"],
        "sv": [float(s) for s in basis["sv"]],
        "e211_committed_pr": E211_PRISTINE_PR,
        "pr_rel_dev": abs(basis["pr"] - E211_PRISTINE_PR) / E211_PRISTINE_PR,
        "note": "the pristine wash-span rebuilt (e211's Gram-SVD, fp64); "
                "gated vs e211's committed PR + SV spectrum below (full run "
                "only)",
        "pass": True,
    }
    if not SMOKE:
        sv_dev = max(abs(float(a) - b) / b for a, b in
                     zip(basis["sv"], E211_PRISTINE_SV))
        G_SPAN["sv_max_rel_dev"] = sv_dev
        G_SPAN["sv_tol"] = TOL_SPAN
        G_SPAN["e212_convention_tol"] = 1e-6
        G_SPAN["pass"] = bool(G_SPAN["pr_rel_dev"] < TOL_SPAN
                              and sv_dev < TOL_SPAN)
        log(f"G_SPAN: PR {basis['pr']:.6f} vs e211 {E211_PRISTINE_PR:.6f} "
            f"(rel {G_SPAN['pr_rel_dev']:.1e}); max SV rel dev "
            f"{sv_dev:.1e} (tol {TOL_SPAN}): "
            + ("PASS" if G_SPAN["pass"] else "FAIL"))
    else:
        log(f"G_SPAN (smoke): PR {basis['pr']:.6f} (SV gate skipped)")
    METRICS["gates"]["G_SPAN"] = G_SPAN
    if not G_SPAN["pass"]:
        fail_texture("G_SPAN failed — the rebuilt span does not reproduce "
                     "e211's committed pristine span")
        return
    Vp = basis["Vp"]                       # (rank, P) unit orthonormal rows
    V3 = Vp[:3]
    del H
    write_partial("the span (e211's pristine wash-span rebuilt + gated)")

    # =====================================================================
    # THE COMPONENT LADDER (nested projections of the one committed delta)
    # =====================================================================
    log("=" * 78)
    log("THE COMPONENT LADDER (nested projections of g1f's committed delta)")
    d_l2 = float(torch.norm(delta))

    def proj(v_basis: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return (v_basis.T @ (v_basis @ vec))

    C_span = proj(Vp, delta)
    C_spanorth = delta - C_span
    C_top3 = proj(V3, delta)
    c1_s0 = dot64(C_span, s0)
    C_s0part = c1_s0 * s0
    C_spanrest = C_span - C_s0part

    comp_geom = {
        "delta_l2": d_l2,
        "C_span": {"l2": float(torch.norm(C_span)),
                   "l2_frac_of_delta": float(torch.norm(C_span)) / d_l2,
                   "cos_vs_delta": cos64(C_span, delta)},
        "C_spanorth": {"l2": float(torch.norm(C_spanorth)),
                       "l2_frac_of_delta":
                           float(torch.norm(C_spanorth)) / d_l2,
                       "cos_vs_delta": cos64(C_spanorth, delta)},
        "C_top3": {"l2": float(torch.norm(C_top3)),
                   "l2_frac_of_delta": float(torch.norm(C_top3)) / d_l2,
                   "l2_frac_of_C_span":
                       float(torch.norm(C_top3)) / float(torch.norm(C_span)),
                   "cos_vs_delta": cos64(C_top3, delta)},
        "C_s0part": {"l2": float(torch.norm(C_s0part)),
                     "c_span_dot_s0": c1_s0},
        "C_spanrest": {"l2": float(torch.norm(C_spanrest))},
        "s0_geometry": {
            "delta_dot_s0": dot64(delta, s0),
            "g13_committed_delta_dot_s0": G13_CELLH_DELTA_DOT_S0,
            "C_spanorth_dot_s0": dot64(C_spanorth, s0),
            "s0_l2_in_span_frac": float(torch.norm(proj(Vp, s0))),
            "cos_delta_top1_sv": cos64(delta, Vp[0]),
            "cos_delta_top2_sv": cos64(delta, Vp[1]),
            "cos_delta_top3_sv": cos64(delta, Vp[2]),
            "cos_s0_top1_sv": cos64(s0, Vp[0]),
            "note": "the span-vs-delta geometry (co-read): how much of the "
                    "crush carrier lives in the family's wash-span, and "
                    "whether the fact-sensitivity direction is in-span",
        },
    }
    # the component arithmetic gates (registered)
    part_d = float(torch.norm(C_span + C_spanorth - delta)) / d_l2
    pyth_d = abs(float(torch.norm(C_span)) ** 2
                 + float(torch.norm(C_spanorth)) ** 2 - d_l2 ** 2) / d_l2 ** 2
    orth_spanorth_vs_V = (float((Vp @ C_spanorth).abs().max())
                          / float(torch.norm(C_spanorth)))
    top3_in_span_d = float(torch.norm(proj(Vp, C_top3) - C_top3)) / d_l2
    s0split_orth = abs(dot64(C_s0part, C_spanrest)) / (
        float(torch.norm(C_s0part)) * float(torch.norm(C_spanrest)) + 1e-30)
    s0split_part_d = float(torch.norm(C_s0part + C_spanrest - C_span)) / d_l2
    G_COMPS = {
        "gate": "the registered component arithmetic: C_span + C_spanorth "
                "== delta; Pythagoras; C_spanorth orthogonal to the span "
                "basis; C_top3 inside the span; the s_0-split orthogonal "
                "and partitioning C_span",
        "partition_max_rel": part_d, "pythagoras_rel_dev": pyth_d,
        "spanorth_in_span_max_proj": orth_spanorth_vs_V,
        "top3_in_span_rel": top3_in_span_d,
        "s0split_orth_cos": s0split_orth,
        "s0split_partition_rel": s0split_part_d,
        "pass": bool(part_d < TOL_ORTH and pyth_d < TOL_ORTH
                     and orth_spanorth_vs_V < TOL_ORTH
                     and top3_in_span_d < TOL_ORTH and s0split_part_d < TOL_ORTH
                     and (s0split_orth < TOL_ORTH
                          or float(torch.norm(C_s0part)) < DEGEN_FLOOR * d_l2
                          or float(torch.norm(C_spanrest)) < DEGEN_FLOOR
                          * d_l2)),
        "geometry": comp_geom,
    }
    log(f"G_COMPS: partition {part_d:.1e}; pythagoras {pyth_d:.1e}; "
        f"spanorth-in-span {orth_spanorth_vs_V:.1e}; top3-in-span "
        f"{top3_in_span_d:.1e}; s0-split orth {s0split_orth:.1e} part "
        f"{s0split_part_d:.1e}; |C_span| {comp_geom['C_span']['l2']:.4f} "
        f"({comp_geom['C_span']['l2_frac_of_delta']:.3f} of |delta|): "
        + ("PASS" if G_COMPS["pass"] else "FAIL"))
    METRICS["gates"]["G_COMPS"] = G_COMPS
    if not G_COMPS["pass"]:
        fail_texture("G_COMPS failed — the component arithmetic broke")
        return
    write_partial("the component ladder (arithmetic gated)")

    # =====================================================================
    # THE ARMS (transplants + removals + the two references)
    # =====================================================================
    log("=" * 78)
    log("THE ARMS (each: renormalized to |delta| (asserted), projected at "
        "R=0.7, the settled +1 read)")

    def renorm(vec: torch.Tensor) -> torch.Tensor:
        """Renormalize to the FULL step's L2 (the registered norm-match)."""
        return vec * (d_l2 / float(torch.norm(vec)))

    cells: dict = {}

    def arm(name: str, role: str, vec: torch.Tensor, form: str) -> dict:
        v = renorm(vec)
        nm = abs(float(torch.norm(v)) - d_l2) / d_l2
        cell = settled_plus1(root_sd, theta0, v, gm12_ids, zid)
        cell.update({"role": role, "form": form,
                     "norm_match_rel_dev": nm,
                     "norm_matched": bool(nm < TOL_NORM),
                     "proj_ok": bool(cell["proj_arithmetic_maxdiff"] < TOL_PROJ)})
        cells[name] = cell
        log(f"  [{name:>22s}] settled +1 {cell['settled_plus1']:.4f} "
            f"(raw {cell['raw_plus1_coreported']:.4f}) | "
            f"{cell['delta_l2']:.4f} norm-match {nm:.1e} "
            + ("OK" if cell["norm_matched"] and cell["proj_ok"] else "FAIL"))
        return cell

    # ---- the two references first (the provenance anchors) ------------------
    refA = arm("REF_own_delta",
               "REFERENCE GATE (the delta object's provenance; committed "
               "0.2577)",
               delta.clone(), "the committed deltas[1] verbatim")
    gA = {"committed_plus1": G1F["committed_plus1"],
          "d_vs_committed": abs(refA["settled_plus1"]
                                - G1F["committed_plus1"]),
          "tol": TOL_LANDING,
          "pass": bool(abs(refA["settled_plus1"] - G1F["committed_plus1"])
                       <= TOL_LANDING
                       and refA["proj_arithmetic_maxdiff"] < TOL_PROJ),
          "note": "g1f under its own COMMITTED delta must reproduce the "
                  "committed +1 0.2577 (g13's gate-R1f convention)"}
    METRICS["gates"]["REF_own__G_REF"] = gA
    if not gA["pass"]:
        fail_texture("REF_own: the own-delta reference landing FAILED — it "
                     "does not reproduce 0.2577; the delta object is not "
                     "the crush carrier")
        return

    # g13's whole-delta s_0 removal (the removal family's arithmetic anchor)
    d_ds = dot64(delta, s0)
    d_orth = delta - d_ds * s0
    refB = arm("REF_g13_dform_removal",
               "REFERENCE GATE (the removal arithmetic's provenance; g13 "
               "cell H committed 0.2951)",
               d_orth, "delta - (delta.s_0)s_0, renormalized (g13's "
                       "registered delta form)")
    gB = {"committed_plus1": G13_CELLH_DFORM,
          "d_vs_committed": abs(refB["settled_plus1"] - G13_CELLH_DFORM),
          "delta_dot_s0": d_ds,
          "g13_committed_delta_dot_s0": G13_CELLH_DELTA_DOT_S0,
          "tol": TOL_LANDING,
          "pass": bool(abs(refB["settled_plus1"] - G13_CELLH_DFORM)
                       <= TOL_LANDING
                       and refB["proj_arithmetic_maxdiff"] < TOL_PROJ),
          "note": "the g13 delta-form removal re-run must reproduce 0.2951 "
                  "(the removal family's anchor; g13 cell H)"}
    METRICS["gates"]["REF_g13rm__G_REF"] = gB
    if not gB["pass"]:
        fail_texture("REF_g13rm: the delta-form removal reference FAILED — "
                     "it does not reproduce 0.2951")
        return
    write_partial("the reference gates (own-delta 0.2577 + g13 removal "
                  "0.2951)")

    # ---- the transplant arms ------------------------------------------------
    t_span = arm("T_span", "TRANSPLANT rung 1 — the SPAN component",
                 C_span, "P_V(delta), renormalized to |delta|")
    t_spanorth = arm("T_spanorth",
                     "TRANSPLANT rung 2 — the SPAN-ORTHOGONAL component "
                     "(== REMOVING the span component)",
                     C_spanorth, "delta - P_V(delta), renormalized")
    t_top3 = arm("T_top3", "TRANSPLANT rung 3 — the TOP-3-SV component",
                 C_top3, "P_V3(delta), renormalized (the largest three span "
                         "directions)")
    if float(torch.norm(C_s0part)) >= DEGEN_FLOOR * d_l2:
        t_s0part = arm("T_s0part",
                       "TRANSPLANT rung 4a — the s_0-direction part of the "
                       "span projection",
                       C_s0part, "(P_V(delta) . s_0) s_0, renormalized "
                                 "(the direction is +/- s_0)")
    else:
        t_s0part = {"role": "TRANSPLANT rung 4a — DEGENERATE (|C_s0part| "
                      "below the floor; never adjudicated)",
                      "settled_plus1": None, "degenerate": True}
        cells["T_s0part"] = t_s0part
        log("  [               T_s0part] DEGENERATE (below the floor)")
    if float(torch.norm(C_spanrest)) >= DEGEN_FLOOR * d_l2:
        t_spanrest = arm("T_spanrest",
                         "TRANSPLANT rung 4b — the rest of the span "
                         "projection",
                         C_spanrest, "P_V(delta) - (P_V(delta).s_0)s_0, "
                                     "renormalized")
    else:
        t_spanrest = {"role": "TRANSPLANT rung 4b — DEGENERATE "
                       "(|C_spanrest| below the floor; never adjudicated)",
                      "settled_plus1": None, "degenerate": True}
        cells["T_spanrest"] = t_spanrest
        log("  [            T_spanrest] DEGENERATE (below the floor)")
    write_partial("the transplant arms (rungs 1-4)")

    # ---- the removal arms ---------------------------------------------------
    # R_span == T_spanorth's vector and R_spanorth == T_span's vector exactly
    # (asserted, computed once — the complementary-rung equivalence)
    id1 = float(torch.norm((delta - C_span) - C_spanorth)) / d_l2
    id2 = float(torch.norm((delta - C_spanorth) - C_span)) / d_l2
    G_DEDUPE = {
        "gate": "removing the span component IS transplanting the "
                "span-orthogonal one (and vice versa) — the vectors "
                "coincide exactly; asserted, computed once",
        "R_span_minus_Cspanorth_rel": id1,
        "R_spanorth_minus_Cspan_rel": id2,
        "pass": bool(id1 < TOL_ORTH and id2 < TOL_ORTH),
    }
    METRICS["gates"]["G_DEDUPE"] = G_DEDUPE
    log(f"G_DEDUPE: |delta-C_span - C_spanorth| {id1:.1e}; |delta-C_spanorth"
        f" - C_span| {id2:.1e}: " + ("PASS" if G_DEDUPE["pass"] else "FAIL"))
    if not G_DEDUPE["pass"]:
        fail_texture("G_DEDUPE failed — the complementary-rung equivalence "
                     "broke (the partition arithmetic)")
        return

    r_top3 = arm("R_top3", "REMOVAL rung 3 — the delta minus its TOP-3-SV "
                           "projection",
                 delta - C_top3, "delta - P_V3(delta), renormalized")
    r_s0part = arm("R_s0part", "REMOVAL rung 4a — the delta minus the "
                               "s_0-direction part of its span projection",
                   delta - C_s0part, "delta - (P_V(delta).s_0)s_0, "
                                     "renormalized")
    r_spanrest = arm("R_spanrest", "REMOVAL rung 4b — the delta minus the "
                                   "rest of its span projection",
                     delta - C_spanrest, "delta - (P_V(delta) - "
                                         "(P_V(delta).s_0)s_0), renormalized")
    write_partial("the removal arms (rungs 3-4 + the complementary rungs)")

    # ---- the first-order ledger (co-read, never adjudicated) ----------------
    ledger = {}
    for key, cell, vec in (
            ("REF_own", refA, delta),
            ("REF_g13rm", refB, d_orth),
            ("T_span", t_span, C_span),
            ("T_spanorth", t_spanorth, C_spanorth),
            ("T_top3", t_top3, C_top3),
            ("T_s0part", t_s0part, C_s0part if not t_s0part.get(
                "degenerate") else None),
            ("T_spanrest", t_spanrest, C_spanrest if not t_spanrest.get(
                "degenerate") else None),
            ("R_top3", r_top3, delta - C_top3),
            ("R_s0part", r_s0part, delta - C_s0part),
            ("R_spanrest", r_spanrest, delta - C_spanrest)):
        if vec is None:
            continue
        disp = (R_CLAIM / float(torch.norm(vec))) * vec
        ledger[key] = {
            "first_order_dlogpz": dot64(s_raw, disp),
            "actual_dlogpz_settled":
                cell["settled_mean_log_pz"] - root_logpz,
            "note": "first-order (s_raw . displacement) vs the actual "
                    "settled change in mean log p(Z) — the honesty "
                    "companion: first order predicts the crush or its "
                    "absence not at all if these disagree in sign/scale",
        }
    METRICS["first_order_ledger"] = ledger
    METRICS["cells"] = cells
    write_partial("the arm table complete (ledger included)")

    # =====================================================================
    # THE ADJUDICATION (frozen composite; no bar shopping)
    # =====================================================================
    removal_arms = {
        "R_span (remove the span component)": t_spanorth["settled_plus1"],
        "R_spanorth (remove the span-orth component)": t_span["settled_plus1"],
        "R_top3 (remove the top-3-SV projection)":
            r_top3["settled_plus1"],
        "R_s0part (remove the s_0-part of the span projection)":
            r_s0part["settled_plus1"],
        "R_spanrest (remove the rest of the span projection)":
            r_spanrest["settled_plus1"],
    }
    spared = {k: bool(v >= CRUSH_BAR) for k, v in removal_arms.items()}
    n_spared = sum(spared.values())
    if n_spared == 1:
        verdict = "COMPONENT-NAMED"
        carrier = next(k for k, ok in spared.items() if ok)
    else:
        carrier = None
    if n_spared == 1:
        verdict = "COMPONENT-NAMED"
    elif n_spared == 0:
        verdict = "MULTI-COMPONENT"
    else:
        verdict = "GRADED"

    ladder_table = {
        "REF_own_delta": {"settled_plus1": refA["settled_plus1"],
                          "committed": G1F["committed_plus1"],
                          "read": "CRUSH (the crush carrier reference)"},
        "REF_g13_dform_removal": {
            "settled_plus1": refB["settled_plus1"],
            "committed": G13_CELLH_DFORM,
            "read": "CRUSH (g13 cell H reproduced; the s_0 removal does "
                    "NOT spare g1f)"},
        "R_span": {"settled_plus1": t_spanorth["settled_plus1"],
                   "source": "the T_spanorth cell (the vectors coincide, "
                             "G_DEDUPE)",
                   "read": ("SPARED (>= 0.5)" if spared[
                       "R_span (remove the span component)"]
                       else "CRUSH (< 0.5)")},
        "R_spanorth": {"settled_plus1": t_span["settled_plus1"],
                       "source": "the T_span cell (the vectors coincide, "
                                 "G_DEDUPE)",
                       "read": ("SPARED (>= 0.5)" if spared[
                           "R_spanorth (remove the span-orth component)"]
                           else "CRUSH (< 0.5)")},
        "R_top3": {"settled_plus1": r_top3["settled_plus1"],
                   "read": ("SPARED (>= 0.5)" if spared[
                       "R_top3 (remove the top-3-SV projection)"]
                       else "CRUSH (< 0.5)")},
        "R_s0part": {"settled_plus1": r_s0part["settled_plus1"],
                     "read": ("SPARED (>= 0.5)" if spared[
                         "R_s0part (remove the s_0-part of the span "
                         "projection)"] else "CRUSH (< 0.5)")},
        "R_spanrest": {"settled_plus1": r_spanrest["settled_plus1"],
                       "read": ("SPARED (>= 0.5)" if spared[
                           "R_spanrest (remove the rest of the span "
                           "projection)"] else "CRUSH (< 0.5)")},
        "T_span": {"settled_plus1": t_span["settled_plus1"],
                   "read": "co-read (the transplant; == R_spanorth)"},
        "T_spanorth": {"settled_plus1": t_spanorth["settled_plus1"],
                       "read": "co-read (the transplant; == R_span)"},
        "T_top3": {"settled_plus1": t_top3["settled_plus1"],
                   "read": "co-read (the transplant)"},
        "T_s0part": {"settled_plus1": t_s0part.get("settled_plus1"),
                     "read": ("co-read (the transplant; the +/- s_0 "
                              "direction at full norm)"
                              if not t_s0part.get("degenerate")
                              else "DEGENERATE (never adjudicated)")},
        "T_spanrest": {"settled_plus1": t_spanrest.get("settled_plus1"),
                       "read": ("co-read (the transplant)"
                                if not t_spanrest.get("degenerate")
                                else "DEGENERATE (never adjudicated)")},
    }
    METRICS["adjudication"] = {
        "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                          ("COMPONENT_NAMED", "MULTI_COMPONENT", "GRADED")},
        "operationalization": REGISTERED_BARS["operationalizations"],
        "constants": {"crush_bar": CRUSH_BAR},
        "removal_arm_reads": removal_arms,
        "n_removal_arms_spared": n_spared,
        "spared": spared,
        "carrier_named": carrier,
        "ladder_table": ladder_table,
        "clauses": {
            "exactly_one_removal_spares": bool(n_spared == 1),
            "every_removal_still_crushes": bool(n_spared == 0),
            "redundant_carriage": bool(n_spared >= 2),
        },
        "co_reads_never_adjudicated": {
            "raw_outside_ball_plus1": {
                k: v["raw_plus1_coreported"] for k, v in cells.items()
                if v.get("raw_plus1_coreported") is not None},
            "first_order_ledger": ledger,
            "span_vs_delta_geometry": comp_geom,
            "g12_loaded_four_cell_table": g12_table,
            "g13_loaded_six_cell_table": g13_six,
            "the_two_draws_contrast": {
                "g1e": {"delta_dot_s0": G1E_DELTA_DOT_S0,
                        "dform_removal_plus1": G1E_DFORM_COREAD,
                        "read": "SPARED (0.708) — the s_0-carried draw"},
                "g1f": {"delta_dot_s0": G13_CELLH_DELTA_DOT_S0,
                        "dform_removal_plus1": G13_CELLH_DFORM,
                        "read": "CRUSH (0.295) — the s_0-independent draw"},
                "note": "the mechanism table at both draws (T198/T200); "
                        "this cell asks what carries the g1f side"},
        },
        "verdict": verdict,
    }
    log("=" * 78)
    log(f"THE VERDICT: {verdict} (n_spared {n_spared}/5)")
    for k, v in removal_arms.items():
        log(f"  {k}: +1 {v:.4f} — "
            f"{'SPARES' if v >= CRUSH_BAR else 'crushes'}")
    write_partial("adjudication")

    # =====================================================================
    # THE CHART
    # =====================================================================
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle("G14 — the g1f crush decomposition (T200's successor): "
                 "the component ladder against e211's committed wash-span "
                 "(CPU, single projected steps)",
                 fontsize=12)

    # (a) the removal ladder (the adjudication)
    ax = axs[0, 0]
    rlabels = ["R_span\n(=T spanorth)", "R_spanorth\n(=T span)",
               "R_top3", "R_s0part", "R_spanrest"]
    rvals = [t_spanorth["settled_plus1"], t_span["settled_plus1"],
             r_top3["settled_plus1"], r_s0part["settled_plus1"],
             r_spanrest["settled_plus1"]]
    rcols = ["#2b6cb0", "#c05621", "#2f855a", "#9b2c2c", "#6b46c1"]
    xs = np.arange(len(rvals))
    ax.bar(xs, rvals, 0.6, color=rcols)
    ax.axhline(CRUSH_BAR, color="r", ls="--", lw=1.2,
               label="the crush bar 0.5")
    for x, v in zip(xs, rvals):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(rlabels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("settled +1 g-12 light")
    ax.set_title(f"(a) THE REMOVAL LADDER — n_spared {n_spared}/5 "
                 f"-> {verdict}")
    ax.legend(fontsize=7)

    # (b) the transplants + the references
    ax = axs[0, 1]
    tlabels = ["own delta\n(REF 0.2577)", "g13 s$_0$-removal\n(REF 0.2951)",
               "T span", "T spanorth", "T top3",
               "T s$_0$part", "T spanrest"]
    tvals = [refA["settled_plus1"], refB["settled_plus1"],
             t_span["settled_plus1"], t_spanorth["settled_plus1"],
             t_top3["settled_plus1"],
             t_s0part.get("settled_plus1") if not t_s0part.get(
                 "degenerate") else 0.0,
             t_spanrest.get("settled_plus1") if not t_spanrest.get(
                 "degenerate") else 0.0]
    tcols = ["#718096", "#718096", "#2b6cb0", "#c05621", "#2f855a",
             "#9b2c2c", "#6b46c1"]
    xs = np.arange(len(tvals))
    ax.bar(xs, tvals, 0.6, color=tcols)
    ax.axhline(CRUSH_BAR, color="r", ls="--", lw=1.2,
               label="the crush bar 0.5")
    for x, v in zip(xs, tvals):
        if v is None:
            continue
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(tlabels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("settled +1 g-12 light")
    ax.set_title("(b) THE TRANSPLANTS + THE REFERENCES (co-reads)")
    ax.legend(fontsize=7)

    # (c) the span + the component masses
    ax = axs[1, 0]
    sv_r = [float(s) for s in basis["sv"]]
    ax.semilogy(np.arange(1, len(sv_r) + 1), sv_r, "o-", ms=4,
                label=f"rebuilt span SVs ({SPAN_STEPS} steps)", color="#2c7a7b")
    if not SMOKE:
        ax.semilogy(np.arange(1, 21), E211_PRISTINE_SV, "x--", ms=4,
                    label="e211 committed", color="#b7791f")
    ax.set_xlabel("SV index")
    ax.set_ylabel("singular value (log)")
    leg = ax.legend(fontsize=7, loc="upper right")
    ax2 = ax.twinx()
    masses = [comp_geom["C_span"]["l2_frac_of_delta"],
              comp_geom["C_spanorth"]["l2_frac_of_delta"],
              comp_geom["C_top3"]["l2_frac_of_delta"],
              float(torch.norm(C_s0part)) / d_l2,
              float(torch.norm(C_spanrest)) / d_l2]
    ax2.bar(np.arange(1, 6) + 5.5, masses, 0.5, color="#ecc94b", alpha=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("|component| / |delta|", color="#b7791f")
    ax2.set_xticks(np.arange(1, 6) + 5.5)
    ax2.set_xticklabels(["span", "spanorth", "top3", "s0part",
                         "spanrest"], fontsize=7)
    ax.set_title(f"(c) THE SPAN (PR {basis['pr']:.3f}) + the component "
                 f"masses")

    # (d) the first-order ledger
    ax = axs[1, 1]
    lkeys = list(ledger.keys())
    fo = [ledger[k]["first_order_dlogpz"] for k in lkeys]
    ac = [ledger[k]["actual_dlogpz_settled"] for k in lkeys]
    xs2 = np.arange(len(lkeys))
    ax.bar(xs2 - 0.2, fo, 0.38, label="first-order prediction "
           "(s$_{raw}$ . displacement)", color="#805ad5")
    ax.bar(xs2 + 0.2, ac, 0.38, label="actual settled d(mean log pZ)",
           color="#4a5568")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs2)
    ax.set_xticklabels([k.replace("_", "\n") for k in lkeys], fontsize=6)
    ax.set_ylabel("change in mean log p(Z)")
    ax.set_title("(d) the first-order ledger — first order predicts the "
                 "crush not at all (co-read)")
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RD / "g1f_decomp.png", dpi=130)
    log(f"[chart] saved {RD / 'g1f_decomp.png'}")

    # ---- the final write -----------------------------------------------------
    METRICS["honesty_reflex"] = {
        "n1_per_arm": "n=1 per arm (single draws, single single-step "
                      "interventions at ONE root); every number is one "
                      "application's read",
        "nested_component_caveat": "the ladder's components are nested "
                                   "projections of ONE delta; each removal "
                                   "renormalizes a 2.74M-dim remainder to "
                                   "the full step's L2 — a spared removal "
                                   "means 'this remainder DIRECTION is "
                                   "non-lethal at R=0.7', not that the "
                                   "removed piece uniquely carried the "
                                   "damage (a distributed kill can masquerade "
                                   "as any-removal-crushes — the "
                                   "MULTI-COMPONENT bar's own reading)",
        "span_is_another_root": "the span is the LOCKED root's wash-span "
                                "(e211's committed pristine span, the "
                                "family's reference); a g1f-own-history span "
                                "is degenerate for this cell (the delta is "
                                "itself the history's first row) — the "
                                "ladder reads 'in the family's wash-span "
                                "directions or orthogonal to them'",
        "reproduction_not_replication": "the two reference cells reproduce "
                                        "committed reads (0.2577 / 0.2951) — "
                                        "provenance gates, not independent "
                                        "replicates; every ladder arm is a "
                                        "first read",
        "readout_single": "the read is one behavior-level readout (the +1 "
                          "g-12 light battery, 60 prompts); no wash "
                          "continuation — the +2 re-capture phase is not "
                          "tested here",
        "open_bits": "whether the g1f crush's carrier (if named) replicates "
                     "at a third cons draw; the +2 re-capture phase's "
                     "anatomy; the g1e side's ladder (the mirrored cell) — "
                     "the successor tier",
    }
    METRICS["reference"] = {
        "g13_loaded": {
            "source": "runs/g13/metrics.json",
            "g1f_own_delta": 0.25772199034690857,
            "g1f_removed_dform": G13_CELLH_DFORM,
            "g1f_delta_dot_s0": G13_CELLH_DELTA_DOT_S0,
            "g1e_removed_dform": G1E_DFORM_COREAD,
            "g1e_delta_dot_s0": G1E_DELTA_DOT_S0,
            "verdict": g13m["adjudication"]["verdict"]},
        "g12_loaded": {
            "source": "runs/g12/metrics.json",
            "four_cell": {k: v["settled_plus1"] for k, v in
                          g12_table.items()},
            "verdict": g12m["adjudication"]["verdict"]},
        "e211_loaded": {
            "source": "runs/e211/metrics.json",
            "pristine_span_pr": E211_PRISTINE_PR,
            "pristine_span_sv_top": E211_PRISTINE_SV[0],
            "root": f"runs/checkpoints/{LOCKED_CK} (the locked e131 root)"},
        "roots": {"g1f_10913": {"ckpt": G1F["ckpt"],
                                 "committed_root_gm12":
                                     G1F["committed_root_gm12"],
                                 "committed_plus1": G1F["committed_plus1"]}},
        "sources": ["runs/g12/metrics.json (the four-cell table, loaded)",
                    "runs/g13/metrics.json (the six-cell table + the g1f "
                    "replicate frame, loaded)",
                    "runs/g11/metrics.json (the three-root geometry)",
                    "runs/e211/metrics.json + runs/e212/metrics.json (the "
                    "committed pristine wash-span + its reproduction gate)",
                    "runs/g1b/metrics.json (the C-arm traj rows 1..20 — "
                    "the pristine history anchors)",
                    "THINKING T198/T200",
                    "lab/g13_symmetry.py (the machinery carried verbatim)",
                    "lab/e211_walled_band.py (the wash-span machinery "
                    "verbatim)",
                    "lab/e204_support.py (the sensitivity convention)",
                    "lab/g1_anchored_ball.py (the wall arithmetic)"],
    }
    METRICS["owner_envelope"]["device_events"] = device_events
    METRICS["timing"] = {"total_s": round(time.time() - T0, 1)}
    METRICS["config"] = {"smoke": SMOKE, "torch": torch.__version__,
                         "threads": torch.get_num_threads(),
                         "params": GB.G1B_PARAMS,
                         "device": "cpu-forced (CUDA_VISIBLE_DEVICES=-1)",
                         "wash_seed": WASH_SEED, "R": R_CLAIM,
                         "lr": LR_ADAMW, "crush_bar": CRUSH_BAR,
                         "span_steps": SPAN_STEPS}
    write_partial("final")
    log(f"DONE — verdict {verdict} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
