"""G12 — THE CRUSH INTERVENTION (T196's named decisive cell).

WHY. g11 (T196) read the three-root geometry and named the crush a
ROTATED-SUPPORT x SHARED-HOT-SET interaction: the wash's first gradient
is essentially the same object at every root (the deltas 55% shared top
coordinates) while each root's fact support sits ROTATED relative to it
(sens overlap 0.23-0.33); the cons roots' first gradients run 20-34%
larger (the clip binding differently); the first-order alignment reads
FAILED to order the crush (the locked root's step is MORE erase-aligned
yet survives). g11 was a read — this cell is the CAUSAL follow-up it
named: intervene the interaction, single first-step applications, and
read the standard +1.

BUILDS ON (directive 1): g11 (the three-root geometry, committed: the
roots' provenance gates, the stream gate, the +1 ledger 0.9452 / 0.2719
/ 0.2577 reproduced in-run, the alignment/composition tables), g1e (the
cons-10912 root + its resume ckpt carrying the COMMITTED first-step
delta), g1b (the locked e131 cons-10901 root + its W1 +1 0.9452), e204
(the fact-sensitivity convention s_0, FD-sign-gated), e194/e195 (the
fact_grad arithmetic), g1/g1b (the wall arithmetic: commit(R=0.7), the
settled +1 read, the ARMED-twin convention). WHAT IS NEW: the two
INTERVENTIONS — the cross-transplant and the s_0-component removal —
each a SINGLE first-step application followed by the standard +1 read;
never intervened at these roots.

THE CELL (the four-cell table; each cell one single-step application +
the standard +1 read):
  A  locked-own-delta  — the locked root under its OWN first delta,
     projected at R=0.7 (the g11-reproduced reference; committed 0.9452).
  B  locked-cons-delta — THE CROSS-TRANSPLANT: g1e's committed first
     delta applied AT THE LOCKED ROOT, projected at R=0.7 verbatim —
     does the LOCKED root now crush at +1 (the delta carries the crush),
     or hold (the crush is the root's response, not the delta)?
  C  cons-own-delta    — g1e's root under its OWN committed first delta
     (the g11-reproduced reference; committed 0.2719).
  D  cons-removed-delta — THE COMPONENT-REMOVAL: at g1e's root, the
     first step with its s_0-component removed,
     delta' = g_0 - (g_0.s_0)s_0, normalized to the same L2, projected
     verbatim — does the cons root now HOLD at +1 (the interaction was
     the s_0-component), or still crush (the damage is elsewhere in the
     step)?

REGISTERED BARS (frozen here, before compute; the dispatch letter
VERBATIM; no bar shopping):
  DELTA-CARRIES: "fires if the cross-transplant crushes the locked root
      (+1 < 0.5 where its own delta reads 0.945) AND the
      component-removal spares the cons root (+1 >= 0.5) — the crush is
      carried by the delta's s_0-component; the interaction causal; the
      mechanism CLOSED."
  ROOT-RESPONDS: "fires if the locked root holds under the cons delta
      AND the cons root still crushes without its s_0-component — the
      crush is the ROOT's nonlinear response (the same step, different
      neighborhoods); the interaction is state-side; the mechanism
      CLOSED the other way."
  GRADED: "any split — the four +1 reads verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * THE +1 READ is the settled read at the projected landing point (the
    in-run ARMED-twin convention, g11's G_LAND instrument verbatim: a
    CommittedGPT anchored AT THE ROOT with wall R=0.7, loaded with the
    raw outside-ball endpoint theta0 + delta, the first forward settles
    onto the ball surface in place; this instrument reproduced the
    committed +1 ledger to 2.4e-7 in g11).
  * THE TRANSPLANT ARITHMETIC: delta_B = g1e's COMMITTED deltas[1] (the
    run's own first-step vector, loaded from g1e_W1_resume.pt and gated
    vs the in-run CPU rebuild, cos > 0.999 + rel L2 < 5%, e193's
    G_S1CK); landing_B = theta_locked + (R/|delta_B|)*delta_B —
    projected at R=0.7, verbatim (the same positive-constant rescale
    the wall applies to any first step).
  * THE REMOVAL ARITHMETIC: s_0 = the unit fact-sensitivity at g1e's
    root (e204's convention, FD-sign-gated HARD at eps {0.05, 0.02});
    g_0 = the post-clip first wash gradient at g1e (the stream's own
    arithmetic, g11 verbatim); delta' = g_0 - (g_0.s_0)s_0 (s_0 unit),
    normalized to L2 = |delta_B| (the first step's raw displacement),
    projected verbatim: landing_D = theta_g1e + (R/|delta'|)*delta'.
    Disclosed: the L2 normalization and the wall projection are both
    positive-constant rescalings — the landing point is
    theta_root + R*unit(g_perp) exactly; the normalization is
    bookkeeping parity with the transplant cell.
  * CRUSH = settled +1 < 0.5; HOLD/SPARE = settled +1 >= 0.5 (the
    letter's own threshold). DELTA-CARRIES fires iff
    cell_B < 0.5 AND cell_D >= 0.5. ROOT-RESPONDS fires iff
    cell_B >= 0.5 AND cell_D < 0.5. GRADED fires otherwise (any split).
  * CO-READS (never adjudicated): the RAW (outside-ball) endpoint read
    per cell (the C-arm convention); the DELTA-FORM removal
    (delta - (delta.s_0)s_0, normalized, projected — the letter's
    formula is the g_0-form; the actual committed first step is AdamW's
    per-coordinate normalization of it; both ride); the first-order
    ledger (s_raw . displacement vs the actual dlog readout); the
    transplant class confound (cos(delta_locked, delta_g1e)).
  * Hard-gate failure (a root provenance gate, the stream gate, the
    anchor gate, the FD sign check, the first-step gates, the reference
    landing gates, or the intervention arithmetic gates) => the record
    completes with verdict TEXTURE (gate failure), nothing adjudicated.

PRE-DISPATCH CHECKS (Rule 12): the two roots' provenance gates (g11's
chain verbatim: load + dial vs the committed root cells + body md5 +
meta; g1e's resume sds[1] anch__ buffers == the root's commit(R) — the
anchor IS the root); the stream gate (the first batch md5 vs the g1e
resume ckpt's own x_hashes[1] + e185's stored hash); the first-step
gates (rebuilt disp vs 1.6543; the rebuild vs the COMMITTED deltas[1]
at cos > 0.999 + rel L2 < 5% where the committed vector exists); the
FD sign check HARD at g1e (the removal's s_0 must BE the fact's
sensitivity); the transplant/removal arithmetic REGISTERED above;
nothing guaranteed — the openness is the point.

REGISTERED PREDICTION (frozen before compute): PREDICTED ROOT-RESPONDS
is the modal landing (T196's geometry: the wash's hot set is
root-independent while the supports rotate — the damage is the step x
state interaction, so the same-class delta at the locked root should
HOLD and the cons root without its s_0-component should still CRUSH).
FOR the root-responds arm 2: the s_0-component of g1e's first gradient
is fact-HELPING to first order (cos(g_0, +s_0) = +0.096, g11) —
removing it cannot first-order spare the fact; a spare would be a
nonlinear, specifically-s_0-mediated effect. AGAINST a clean landing:
the transplant is a DIFFERENT draw of the delta class (cos(delta_locked,
delta_g1e) ~ 0.09, 55% shared hot set) — a hold at the locked root is
ambiguous between "the root's response" and "this delta is not the
crushing vector at this geometry" (the class-vs-vector confound, named;
the removal cell carries the discrimination); a crush under the
transplant with a still-crushing removal would be GRADED (the delta
carries something, but not via s_0). FALSIFIER of the T196 interaction
story: DELTA-CARRIES in full — the s_0-component alone carries the
crush, and the first-order story g11 falsified returns. No bar shopping.

WHAT THIS CELL GUARANTEES: nothing — n=1 per cell (single draws, single
single-step interventions); the transplant moves the WHOLE delta (its
55%-shared class identity is the treatment, not a located component);
the removal removes ONE unit direction (s_0) from the gradient form of
the step — "the damage is elsewhere in the step" covers the entire
orthogonal complement, not a located mechanism; the read is one
behavior-level readout (the +1 g-12 light battery); no wash continuation
(the +2 recovery is not tested here).

COMPUTE ENVELOPE (the owner envelope, 2026-10-02): CPU-ONLY
(CUDA_VISIBLE_DEVICES=-1 forced before torch; the GPU is never claimed
— the letter's "CPU-able at 2.74M but slow; claim short GPU bursts if
free, else CPU"; g11 measured the same bundle's whole compute at ~18s
CPU, so CPU is the decisive choice); the double-polled load checks
recorded at start (recorded, not gating); torch threads 4; PROGRESSIVE
metrics.json writes after every phase.

Outputs: runs/g12/{metrics.json (PROGRESSIVE), crush_intervention.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python g12_intervention.py    (G12_SMOKE=1 shakedown)
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (g11's convention)

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

SMOKE = os.environ.get("G12_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "g12 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ======================================================================
# THE CONFIG (two roots + the frozen constants)
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
]
LOCKED_TAG = "locked_10901"
CONS_TAG = "g1e_10912"

# ---- the frozen adjudication constants --------------------------------------
CRUSH_BAR = 0.5                         # the letter's threshold (+1 < 0.5 = crush)
FD_EPS = (0.05, 0.02) if not SMOKE else (0.05,)   # e204's probe sizes (L2)
TOL_DIAL = 0.02                         # root dial vs committed (texture tier)
TOL_LANDING = 0.02                      # settled +1 read vs committed
TOL_DISP = 0.02                         # first-step disp vs committed 1.6543
S1CK_COS = 0.999                        # e193's G_S1CK convention
S1CK_REL = 0.05
TOL_ORTH = 1e-4                         # the removal's orthogonality gate
TOL_PROJ = 1e-5                         # the projection-arithmetic gate

REGISTERED_BARS = {
    "DELTA_CARRIES": ("DELTA-CARRIES: \"fires if the cross-transplant crushes "
                      "the locked root (+1 < 0.5 where its own delta reads "
                      "0.945) AND the component-removal spares the cons root "
                      "(+1 >= 0.5) — the crush is carried by the delta's "
                      "s_0-component; the interaction causal; the mechanism "
                      "CLOSED.\""),
    "ROOT_RESPONDS": ("ROOT-RESPONDS: \"fires if the locked root holds under "
                      "the cons delta AND the cons root still crushes without "
                      "its s_0-component — the crush is the ROOT's nonlinear "
                      "response (the same step, different neighborhoods); the "
                      "interaction is state-side; the mechanism CLOSED the "
                      "other way.\""),
    "GRADED": "GRADED: \"any split — the four +1 reads verbatim.\"",
    "operationalizations": (
        "frozen BEFORE compute (they fix the clauses, they do not move the "
        "bars): the +1 read is the SETTLED read at the projected landing "
        "(the in-run ARMED-twin convention, g11's G_LAND instrument verbatim "
        "— a CommittedGPT anchored at the root with wall R=0.7, loaded with "
        "the raw outside-ball endpoint, the first forward settles onto the "
        "ball in place; g11 reproduced the committed ledger to 2.4e-7 with "
        "it). TRANSPLANT: delta_B = g1e's COMMITTED deltas[1] (gated vs the "
        "in-run rebuild, cos > 0.999 + rel L2 < 5%); landing_B = "
        "theta_locked + (R/|delta_B|)*delta_B (projected at R=0.7, "
        "verbatim). REMOVAL: s_0 = the unit fact-sensitivity at g1e (e204's "
        "convention, FD-sign-gated HARD at eps {0.05, 0.02}); g_0 = the "
        "post-clip first wash gradient at g1e (the stream's own arithmetic); "
        "delta' = g_0 - (g_0.s_0)s_0, normalized to L2 = |delta_B| (the "
        "first step's raw displacement), projected verbatim: landing_D = "
        "theta_g1e + (R/|delta'|)*delta' — disclosed: the normalization and "
        "the projection are positive-constant rescalings, so the landing is "
        "theta_root + R*unit(g_perp) exactly. CRUSH = settled +1 < 0.5; "
        "HOLD/SPARE >= 0.5 (the letter's threshold). DELTA-CARRIES iff "
        "cell_B < 0.5 AND cell_D >= 0.5. ROOT-RESPONDS iff cell_B >= 0.5 "
        "AND cell_D < 0.5. GRADED otherwise (any split). CO-READS (never "
        "adjudicated): the raw outside-ball endpoint read per cell (the "
        "C-arm convention); the delta-form removal (delta - "
        "(delta.s_0)s_0, normalized, projected); the first-order ledger; "
        "the transplant class confound (cos(delta_locked, delta_g1e)). "
        "Hard-gate failure => the record completes, verdict TEXTURE (gate "
        "failure), nothing adjudicated."),
    "registered_prediction": (
        "frozen before compute. PREDICTED ROOT-RESPONDS is the modal "
        "landing (T196's geometry: the wash's hot set is root-independent "
        "while the supports rotate — the damage is the step x state "
        "interaction, so the same-class delta at the locked root should "
        "HOLD and the cons root without its s_0-component should still "
        "CRUSH). FOR root-responds arm 2: the s_0-component of g1e's first "
        "gradient is fact-HELPING to first order (cos(g_0, +s_0) = +0.096, "
        "g11) — removing it cannot first-order spare the fact; a spare "
        "would be a nonlinear, specifically-s_0-mediated effect. AGAINST a "
        "clean landing: the transplant is a DIFFERENT draw of the delta "
        "class (cos(delta_locked, delta_g1e) ~ 0.09, 55% shared hot set) — "
        "a hold at the locked root is ambiguous between 'the root's "
        "response' and 'this delta is not the crushing vector at this "
        "geometry' (the class-vs-vector confound, named; the removal cell "
        "carries the discrimination); a crush under the transplant with a "
        "still-crushing removal would be GRADED. FALSIFIER of the T196 "
        "interaction story: DELTA-CARRIES in full — the s_0-component "
        "alone carries the crush, and the first-order story g11 falsified "
        "returns. No bar shopping."),
    "registration": ("the dispatch's registration IS the registration (the "
                     "bars quoted verbatim here and in the module docstring, "
                     "frozen before compute). Adjudicate against exactly "
                     "this; no bar shopping."),
}

deviations: list[str] = [
    "CPU-ONLY, FORCED (g11's convention; the letter's 'CPU-able at 2.74M "
    "but slow; claim short GPU bursts if free, else CPU' — g11 measured "
    "the same bundle's whole compute at ~18s CPU, so CPU is the decisive "
    "choice): CUDA_VISIBLE_DEVICES=-1 before torch; the GPU is never "
    "claimed; the double-polled load checks recorded at start (recorded, "
    "not gating).",
    "THE LOCKED ROOT'S FIRST-STEP DELTA IS REBUILT (g11's own deviation "
    "carried): g1b's W1 predates the resume-ckpt convention, so no "
    "committed delta vector exists for the locked root; its rebuilt delta "
    "is PRIMARY and gated by the disp-norm (1.6543) + the settled landing "
    "read vs the committed +1 (0.9452, texture tier 0.02). g1e's delta is "
    "LOADED COMMITTED from the resume ckpt (deltas[1]) with the CPU "
    "rebuild as the gate (cos > 0.999 + rel L2 < 5%, e193's G_S1CK).",
    "THE LOCKED ROOT'S s_0 IS TEXTURE (ungated): the first-order ledger "
    "needs s_raw at both roots, but the adjudication uses s_0 at g1e only; "
    "the locked FD probes are recorded as co-reads without a hard gate "
    "(g11's same-instrument G_SENSDIR PASSED at this exact root, "
    "eps {0.05, 0.02} monotone).",
    "THE REMOVAL IS THE g_0-FORM (the letter's literal formula delta' = "
    "g_0 - (g_0.s_0)s_0): the actual committed first step is AdamW's "
    "per-coordinate normalization of the gradient, NOT a parallel vector; "
    "the delta-form removal rides as a co-read (never adjudicated).",
    "NO WASH CONTINUATION: each intervention is a SINGLE first-step "
    "application + the standard +1 read (the letter's cell); the +2 "
    "re-capture phase (the wall's own) is not tested here.",
    "n=1 PER CELL (single draws, single single-step interventions); the "
    "reads are single applications of single objects.",
    "torch threads 4 (shared machine; g1/g1b's import resets to 8 — reset "
    "after import).",
    "Smoke mode trims: FD eps {0.05}; nothing adjudicated.",
]

device_events: list[dict] = []
RD: Path = None                     # set in main (run_dir)
METRICS: dict = {}
_progressive = {"n": 0, "phases": []}

# the per-root stashes (module-level, never serialized)
_THETA: dict = {}
_SRAW: dict = {}
_ROOT_GM12: dict = {}


def fact_readout_root(tag: str) -> float:
    """The root's mean log p(Z) (the first-order ledger's baseline)."""
    return METRICS["gates"][f"{tag}__G_ROOT"]["root_mean_log_pz"]


def _root_gm12(tag: str) -> float:
    """The root's g-12 read (the chart's reference lines)."""
    return _ROOT_GM12[tag]


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
    arithmetic; e204/g11's copy): gradient of the fact battery's mean
    log p(Z) readout at the net's CURRENT weights. Sign convention (the
    critic's, carried): -s = the fact-ERASING ray. Consumes no RNG; tiny
    eval burst."""
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


def write_partial(phase: str) -> None:
    """PROGRESSIVE metrics (the outage lesson): metrics.json after every
    phase; bookkeeping must never kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    METRICS.update({
        "experiment": "g12_intervention", "date": common.now_iso(),
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
                          ("DELTA_CARRIES", "ROOT_RESPONDS", "GRADED")},
        "verdict": "TEXTURE (gate failure)",
        "reason": reason,
        "note": "the registered composite: hard-gate failure => the record "
                "completes with verdict TEXTURE, nothing adjudicated",
    }
    log(f"[TEXTURE] hard gate failure: {reason}")
    write_partial("texture (gate failure)")


def settled_plus1(root_sd: dict, theta0: torch.Tensor, delta: torch.Tensor,
                  gm12_ids: torch.Tensor, zid: int) -> dict:
    """THE STANDARD +1 READ (g11's G_LAND instrument verbatim): the ARMED
    twin — a CommittedGPT anchored AT THE ROOT (commit R=0.7), loaded with
    the raw outside-ball endpoint theta0 + delta; the first forward
    settles onto the ball surface in place; read the g-12 battery. The
    raw (uncommitted-twin) endpoint read rides as the C-arm co-read; the
    projection-arithmetic gate and the first-order ledger ride with it."""
    d_norm = float(torch.norm(delta))
    landing = theta0 + (R_CLAIM / d_norm) * delta
    # the settled read (ARMED twin: the wall projects in place)
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
    # the raw endpoint read (the C-arm convention, co-reported)
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
    RD = run_dir("g12_smoke" if SMOKE else "g12")
    log(f"G12 THE CRUSH INTERVENTION (T196's named decisive cell) "
        f"(smoke={SMOKE}) -> {RD}")
    # the envelope load-check, DOUBLE-POLLED (the letter; recorded, not
    # gating — CPU-only policy, no GPU launch ever happens)
    polls = []
    for k in (1, 2):
        s = common.gpu_status()
        polls.append(s)
        try:
            common._log_envelope_poll(f"g12:load-check-{k}(cpu-only-policy)",
                                      s["util"], s["temp"], True)
        except Exception:
            pass
        if k == 1:
            time.sleep(2.0)             # the double poll's gap
    device_events.append({
        "tag": "g12", "event": "LOAD CHECK x2 (CPU-only policy; GPU never "
        "claimed)", "status_polls": polls,
        "note": "the letter's envelope: nearly eval-only cell; the single "
                "first-step applications run CPU (2.74M, seconds — g11 "
                "measured the same bundle ~18s); no GPU launch"})
    log("[envelope] double load-check recorded: "
        + " | ".join(f"poll{ i+1 } util {p['util']:.0f}% temp {p['temp']:.0f}C"
                     for i, p in enumerate(polls))
        + " — CPU-only policy (no GPU launch)")

    METRICS.update({
        "design": (
            "G12 THE CRUSH INTERVENTION (T196's named decisive cell): the "
            "two registered interventions, each a SINGLE first-step "
            "application followed by the standard +1 read (the settled "
            "ARMED-twin read at the R=0.7-projected landing). THE "
            "FOUR-CELL TABLE: (A) locked-own-delta — the locked e131 "
            "cons-10901 root under its OWN rebuilt first delta (the "
            "g11-reproduced reference, committed 0.9452); (B) "
            "locked-cons-delta — THE CROSS-TRANSPLANT: g1e's committed "
            "first delta applied AT THE LOCKED ROOT, projected at R=0.7 "
            "verbatim — does the locked root crush (the delta carries the "
            "crush) or hold (the crush is the root's response)?; (C) "
            "cons-own-delta — g1e's root under its OWN committed delta "
            "(the g11-reproduced reference, committed 0.2719); (D) "
            "cons-removed-delta — THE COMPONENT-REMOVAL: at g1e's root, "
            "delta' = g_0 - (g_0.s_0)s_0 normalized to the same L2, "
            "projected verbatim — does the cons root HOLD (the "
            "interaction was the s_0-component) or still crush (the "
            "damage is elsewhere in the step)? T196's geometry in: the "
            "deltas 55% shared, the supports rotated 0.23-0.33, the cons "
            "roots' first gradients 20-34% larger."),
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

    # ---------------- protocol rebuild (g1e/g11 VERBATIM) ---------------------
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

    # the fact battery (the g-12 install-60 battery = the e194/e195/e204
    # sensitivity convention; the +1 read's readout)
    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids = bat_ids[-12]
    G_BATTERY = {"shapes": {str(j): list(bat_ids[j].shape) for j in G1.GEOS},
                 "pass": bool(list(gm12_ids.shape) == [60, G1.PRE - 12]),
                 "note": "the g-12 install-60 battery (e194/e195's fact_grad "
                         "convention, e204's carrier, g11's +1 readout; "
                         "shapes 60x118)"}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    # the neutral anchor bank (e170's construction VERBATIM via g1e/g11)
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

    # load the g1e resume ckpt's committed stream hash + first delta + anchor
    rc = torch.load(CKPT_DIR / "g1e_W1_resume.pt", map_location="cpu",
                    weights_only=False)
    committed_stream = rc["x_hashes"].get(1, rc["x_hashes"].get("1"))
    committed_delta_g1e = rc["deltas"][1].float()
    committed_anchor_g1e = {"wall_R": float(rc["wall_R"]),
                            "anch_md5": anch_md5_of(rc["sds"][1])}
    del rc
    G_STREAM = {
        "x1_md5": x1_md5,
        "vs_g1e_resume": bool(committed_stream == x1_md5),
        "vs_e185_stored": bool(x1_md5 == E185_X1_MD5),
        "note": "the seed-10902 stream drawn VERBATIM (the e170 bank + the "
                "first batch); md5-gated vs the g1e resume ckpt's own "
                "x_hashes[1] (the exact batch the committed run consumed) "
                "+ e185's stored hash (the family co-report)",
    }
    G_STREAM["pass"] = bool(G_STREAM["vs_e185_stored"]
                            and G_STREAM["vs_g1e_resume"])
    log(f"G_STREAM: x1 md5 {x1_md5[:10]}.. vs g1e resume "
        f"{G_STREAM['vs_g1e_resume']} vs e185 {G_STREAM['vs_e185_stored']}: "
        + ("PASS" if G_STREAM["pass"] else "FAIL"))
    METRICS["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_STREAM": G_STREAM}
    if not G_STREAM["pass"]:
        fail_texture("G_STREAM failed — the stream diverged from the "
                     "committed run's own x_hashes")
        return
    write_partial("protocol + stream gates")

    # =====================================================================
    # THE TWO ROOTS — provenance + the first-step rebuild (g11's chain)
    # =====================================================================
    root_sd_map: dict = {}
    delta_map: dict = {}
    cells: dict = {}

    for spec in ROOTS:
        tag = spec["tag"]
        log("=" * 78)
        log(f"ROOT {tag} — {spec['lineage']} ({spec['ckpt']})")

        # ---- load + the root provenance gate (g11's G_ROOT verbatim)
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
        _THETA[tag] = theta0
        root_sd_map[tag] = root_sd
        if n_params != GB.G1B_PARAMS:
            METRICS["gates"][f"{tag}__G_ROOT"] = {
                "params": n_params, "pass": False}
            fail_texture(f"{tag}: params {n_params} != {GB.G1B_PARAMS}")
            return

        root_gm12 = G1.battery_cell(net_root, gm12_ids, zid)["mean_pz"]
        root_logpz = fact_readout(net_root, gm12_ids, zid)
        _ROOT_GM12[tag] = root_gm12
        meta_ok = True
        if tag == CONS_TAG:
            meta_ok = bool(meta.get("experiment") == "g1e"
                           and meta.get("cons_seed") == 10912)
        G_ROOT = {
            "checkpoint": f"runs/checkpoints/{spec['ckpt']}",
            "meta": meta, "params": n_params,
            "body_md5": sd_md5_body(root_sd),
            "gm12": root_gm12, "gm12_committed": spec["committed_root_gm12"],
            "gm12_d": abs(root_gm12 - spec["committed_root_gm12"]),
            "root_mean_log_pz": root_logpz,
            "meta_gate": meta_ok,
            "tol": TOL_DIAL,
            "pass": bool(abs(root_gm12 - spec["committed_root_gm12"])
                         <= TOL_DIAL and meta_ok),
            "note": "the root loads into the 2,739,072 cfg clean; its "
                    "same-instrument g-12 reproduces the committed root "
                    "cell (texture tier 0.02); meta provenance (experiment "
                    "+ cons_seed) at the cons draw",
        }
        log(f"G_ROOT[{tag}]: gm12 {root_gm12:.4f} vs committed "
            f"{spec['committed_root_gm12']:.4f} (|d| {G_ROOT['gm12_d']:.2e}),"
            f" meta {'OK' if meta_ok else 'DRIFT'}: "
            + ("PASS" if G_ROOT["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_ROOT"] = G_ROOT
        if not G_ROOT["pass"]:
            fail_texture(f"{tag}: root provenance gate FAILED")
            return

        # ---- G-ANCHOR (g1e): the resume sds' anchor IS the root
        if tag == CONS_TAG:
            net_c = G1.CommittedGPT(GB.G1B_CFG)
            net_c.load_state_dict(root_sd)
            net_c.commit(R_CLAIM)
            anch_md5_root = anch_md5_of(net_c.state_dict())
            wall_r_ok = abs(committed_anchor_g1e["wall_R"]
                            - R_CLAIM) < 1e-6
            md5_ok = bool(anch_md5_root == committed_anchor_g1e["anch_md5"])
            g_anchor = {
                "form": "the W1 arm's wall anchor (theta_anchor) IS the "
                        "root — the anch__ buffers in the committed resume "
                        "sds[1] must equal the root ckpt's commit(R)",
                "wall_R_resume": committed_anchor_g1e["wall_R"],
                "wall_R_expected": R_CLAIM,
                "wall_R_match": bool(wall_r_ok),
                "anch_md5_root_rebuilt": anch_md5_root,
                "anch_md5_resume": committed_anchor_g1e["anch_md5"],
                "anch_md5_match": md5_ok,
                "pass": bool(wall_r_ok and md5_ok),
            }
            del net_c
            log(f"G-ANCHOR[{tag}]: wall_R "
                f"{committed_anchor_g1e['wall_R']:.7f} "
                f"{'match' if wall_r_ok else 'DRIFT'}, anchor md5 "
                f"{'match' if md5_ok else 'DRIFT'}: "
                + ("PASS" if g_anchor["pass"] else "FAIL"))
            METRICS["gates"][f"{tag}__G_ANCHOR"] = g_anchor
            if not g_anchor["pass"]:
                fail_texture(f"{tag}: G-ANCHOR failed — the committed wall's "
                             "anchor is not this root")
                return

        # ---- s_0: the root's OWN fact sensitivity (the e204 convention)
        s_raw = fact_grad(net_root, gm12_ids, zid)
        s0 = s_raw / torch.norm(s_raw)
        _SRAW[tag] = s_raw.clone()
        read0 = root_logpz
        # the FD sign check (HARD at g1e — the removal's s_0 is adjudicated;
        # TEXTURE at the locked root — first-order ledger only, g11 passed)
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
        hard = (tag == CONS_TAG)
        G_SENSDIR = {
            "convention": "e194/e195's fact_grad VERBATIM (the g-12 "
                          "install-60 battery; matched-point at the root); "
                          "FD sign check at eps in "
                          f"{tuple(FD_EPS)} (e204's G_SENSDIR)",
            "hard_gate": hard,
            "readout_theta0_mean_log_pz": read0,
            "probes": fd,
            "pass": bool(all(fd[f"eps{eps}_monotone"] for eps in FD_EPS)),
            "note": ("HARD (the removal's s_0 is adjudicated)" if hard else
                     "TEXTURE co-read (the first-order ledger needs s_raw "
                     "at the locked root; g11's same-instrument check "
                     "PASSED at this exact root — not re-gated here)"),
        }
        log(f"G_SENSDIR[{tag}] ({'HARD' if hard else 'texture'}): read0 "
            f"{read0:.4f}; FD monotone "
            + "/".join(f"eps{e}:{fd[f'eps{e}_monotone']}" for e in FD_EPS)
            + ": " + ("PASS" if G_SENSDIR["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_SENSDIR"] = G_SENSDIR
        if hard and not G_SENSDIR["pass"]:
            fail_texture(f"{tag}: FD sign check FAILED — s_0 is not the "
                         "fact's sensitivity at this root")
            return

        # ---- the first wash step, REBUILT (g11's arithmetic verbatim) --
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

        # ---- the primary delta: COMMITTED where it exists (g11's rule)
        if tag == CONS_TAG:
            cos_rb_c = cos64(delta_rebuilt, committed_delta_g1e)
            rel_l2 = abs(float(torch.norm(committed_delta_g1e))
                         - d_raw) / d_raw
            G_STEP1 = {"disp_rebuilt": d_raw,
                       "disp_committed": spec["committed_first_disp"],
                       "disp_d": abs(d_raw - spec["committed_first_disp"]),
                       "cos_rebuilt_vs_committed": cos_rb_c,
                       "rel_l2_vs_committed": rel_l2,
                       "primary": "COMMITTED deltas[1] (the run's own "
                                  "vector); the rebuild is the gate",
                       "pass": bool(
                           abs(d_raw - spec["committed_first_disp"])
                           <= TOL_DISP and cos_rb_c > S1CK_COS
                           and rel_l2 < S1CK_REL)}
            delta_map[tag] = committed_delta_g1e.clone()
        else:
            G_STEP1 = {"disp_rebuilt": d_raw,
                       "disp_committed": spec["committed_first_disp"],
                       "disp_d": abs(d_raw - spec["committed_first_disp"]),
                       "primary": "REBUILT (g1b predates the resume "
                                  "convention; gated by disp + landing)",
                       "pass": bool(abs(d_raw - spec["committed_first_disp"])
                                    <= TOL_DISP)}
            delta_map[tag] = delta_rebuilt.clone()
        log(f"G_STEP1[{tag}]: |delta| {d_raw:.4f} vs committed "
            f"{spec['committed_first_disp']:.4f} (|d| "
            f"{G_STEP1['disp_d']:.2e})"
            + (f"; rebuild-vs-committed cos {cos_rb_c:.6f} rel {rel_l2:.2e}"
               if tag == CONS_TAG else "")
            + ": " + ("PASS" if G_STEP1["pass"] else "FAIL"))
        METRICS["gates"][f"{tag}__G_STEP1"] = G_STEP1
        if not G_STEP1["pass"]:
            fail_texture(f"{tag}: first-step gate FAILED")
            return
        METRICS["gates"][f"{tag}__FIRSTSTEP_TEXTURE"] = {
            "ce_batch": ce1, "preclip_gnorm": preclip_norm,
            "clip_binds": clip_binds,
            "g0_l2_postclip": float(torch.norm(g0)),
            "note": "the first wash step's stream arithmetic (g11's "
                    "texture reads; the cons roots' pre-clip norms run "
                    "20-34% larger, g11's norm-asymmetry read)",
        }

        # stash the post-clip gradient (the removal's g_0 at the cons root)
        if tag == CONS_TAG:
            g0_cons = g0.clone()
            s0_cons = s0.clone()
        del net_root, g0, s_raw, s0, theta1, delta_rebuilt
        write_partial(f"root {tag} (gates + first-step rebuild)")

    # =====================================================================
    # THE FOUR-CELL TABLE
    # =====================================================================
    log("=" * 78)
    log("THE FOUR-CELL TABLE (each cell: a single first-step application "
        "+ the standard +1 read)")

    # ---- cell A: locked-own-delta (the reference; gate vs 0.9452) ----------
    cellA = settled_plus1(root_sd_map[LOCKED_TAG], _THETA[LOCKED_TAG],
                          delta_map[LOCKED_TAG], gm12_ids, zid)
    gA = {"committed_plus1": ROOTS[0]["committed_plus1"],
          "d_vs_committed": abs(cellA["settled_plus1"]
                                - ROOTS[0]["committed_plus1"]),
          "tol": TOL_LANDING,
          "pass": bool(abs(cellA["settled_plus1"]
                           - ROOTS[0]["committed_plus1"]) <= TOL_LANDING
                       and cellA["proj_arithmetic_maxdiff"] < TOL_PROJ),
          "note": "the g11-reproduced reference cell: the locked root under "
                  "its own delta, settled +1 vs the committed W1 +1 0.9452"}
    cells["locked_10901__own_delta"] = {
        "role": "REFERENCE (the locked root's own step; committed 0.9452)",
        "delta": "REBUILT (the locked root's own first step)",
        **cellA}
    METRICS["gates"]["cellA__G_REF"] = gA
    log(f"  [A] locked-own-delta    : settled +1 "
        f"{cellA['settled_plus1']:.4f} vs committed 0.9452 (|d| "
        f"{gA['d_vs_committed']:.2e}); raw {cellA['raw_plus1_coreported']:.4f}"
        f": " + ("PASS" if gA["pass"] else "FAIL"))
    if not gA["pass"]:
        fail_texture("cell A: the reference landing gate FAILED — the "
                     "locked root's own step does not reproduce 0.9452")
        return

    # ---- cell C: cons-own-delta (the reference; gate vs 0.2719) ------------
    cellC = settled_plus1(root_sd_map[CONS_TAG], _THETA[CONS_TAG],
                          delta_map[CONS_TAG], gm12_ids, zid)
    gC = {"committed_plus1": ROOTS[1]["committed_plus1"],
          "d_vs_committed": abs(cellC["settled_plus1"]
                                - ROOTS[1]["committed_plus1"]),
          "tol": TOL_LANDING,
          "pass": bool(abs(cellC["settled_plus1"]
                           - ROOTS[1]["committed_plus1"]) <= TOL_LANDING
                       and cellC["proj_arithmetic_maxdiff"] < TOL_PROJ),
          "note": "the g11-reproduced reference cell: the cons root under "
                  "its own committed delta, settled +1 vs the committed "
                  "+1 0.2719"}
    cells["g1e_10912__own_delta"] = {
        "role": "REFERENCE (the cons root's own step; committed 0.2719)",
        "delta": "COMMITTED deltas[1] (the cons root's own first step)",
        **cellC}
    METRICS["gates"]["cellC__G_REF"] = gC
    log(f"  [C] cons-own-delta      : settled +1 "
        f"{cellC['settled_plus1']:.4f} vs committed 0.2719 (|d| "
        f"{gC['d_vs_committed']:.2e}); raw {cellC['raw_plus1_coreported']:.4f}"
        f": " + ("PASS" if gC["pass"] else "FAIL"))
    if not gC["pass"]:
        fail_texture("cell C: the reference landing gate FAILED — the cons "
                     "root's own committed step does not reproduce 0.2719")
        return
    write_partial("reference cells A + C (gates + reads)")

    # ---- cell B: THE CROSS-TRANSPLANT (the open read) ----------------------
    delta_B = committed_delta_g1e.clone()
    cellB = settled_plus1(root_sd_map[LOCKED_TAG], _THETA[LOCKED_TAG],
                          delta_B, gm12_ids, zid)
    gB = {
        "form": "delta_B = g1e's COMMITTED deltas[1]; landing_B = "
                "theta_locked + (R/|delta_B|)*delta_B (projected at R=0.7, "
                "verbatim)",
        "delta_l2": cellB["delta_l2"],
        "landing_on_ball": bool(cellB["proj_arithmetic_maxdiff"] < TOL_PROJ),
        "class_confound_cos_delta_locked_vs_cons":
            cos64(delta_map[LOCKED_TAG], delta_B),
        "pass": bool(cellB["proj_arithmetic_maxdiff"] < TOL_PROJ),
        "note": "the intervention read has NO committed reference — the "
                "openness is the point; the transplant moves the whole "
                "delta (its 55%-shared class identity is the treatment)",
    }
    cells["locked_10901__cons_delta"] = {
        "role": "INTERVENTION 1 — THE CROSS-TRANSPLANT (the locked root "
                "under the cons delta)",
        "delta": "COMMITTED deltas[1] of g1e (transplanted verbatim)",
        **cellB}
    METRICS["gates"]["cellB__G_TRANSPLANT"] = gB
    log(f"  [B] locked-cons-delta   : THE CROSS-TRANSPLANT — settled +1 "
        f"{cellB['settled_plus1']:.4f} (raw "
        f"{cellB['raw_plus1_coreported']:.4f}); "
        f"{'CRUSH (<0.5)' if cellB['settled_plus1'] < CRUSH_BAR else 'HOLD (>=0.5)'}"
        f"; cos(delta_locked, delta_cons) "
        f"{gB['class_confound_cos_delta_locked_vs_cons']:+.4f}: "
        + ("PASS" if gB["pass"] else "FAIL"))
    if not gB["pass"]:
        fail_texture("cell B: the transplant arithmetic gate FAILED")
        return
    write_partial("cell B — the cross-transplant")

    # ---- cell D: THE COMPONENT-REMOVAL (the open read) ---------------------
    # the registered arithmetic: delta' = g_0 - (g_0.s_0)s_0, normalized to
    # the same L2 as the first step (|delta_B|), projected verbatim
    TARGET_L2 = float(torch.norm(delta_B))
    c_gs = dot64(g0_cons, s0_cons)
    g_perp = g0_cons - c_gs * s0_cons
    g_perp_l2 = float(torch.norm(g_perp))
    delta_removed = g_perp * (TARGET_L2 / g_perp_l2)
    cellD = settled_plus1(root_sd_map[CONS_TAG], _THETA[CONS_TAG],
                          delta_removed, gm12_ids, zid)
    orth_cos = cos64(delta_removed, s0_cons)
    # the co-read: the delta-form removal (AdamW's actual step, not the
    # gradient form) — never adjudicated
    c_ds = dot64(delta_map[CONS_TAG], s0_cons)
    d_orth = delta_map[CONS_TAG] - c_ds * s0_cons
    delta_removed_dform = d_orth * (TARGET_L2 / float(torch.norm(d_orth)))
    cellD_coread = settled_plus1(root_sd_map[CONS_TAG], _THETA[CONS_TAG],
                                 delta_removed_dform, gm12_ids, zid)
    gD = {
        "form": "delta' = g_0 - (g_0.s_0)s_0 (s_0 unit, FD-gated), "
                "normalized to L2 = |delta_B| (the first step's raw "
                "displacement), projected verbatim: landing_D = theta_cons "
                "+ (R/|delta'|)*delta'",
        "g0_dot_s0": c_gs,
        "g0_l2_postclip": float(torch.norm(g0_cons)),
        "g_perp_l2_prenorm": g_perp_l2,
        "s0_component_removed_l2": abs(c_gs),
        "delta_prime_l2": float(torch.norm(delta_removed)),
        "target_l2": TARGET_L2,
        "orth_cos_delta_prime_vs_s0": orth_cos,
        "landing_on_ball": bool(cellD["proj_arithmetic_maxdiff"] < TOL_PROJ),
        "pass": bool(abs(orth_cos) <= TOL_ORTH
                     and abs(float(torch.norm(delta_removed)) - TARGET_L2)
                     <= 1e-4 * TARGET_L2
                     and cellD["proj_arithmetic_maxdiff"] < TOL_PROJ),
        "disclosure": "the L2 normalization and the wall projection are "
                      "positive-constant rescalings — the landing point is "
                      "theta_cons + R*unit(g_perp) exactly; disclosed",
        "coread_delta_form": {
            "form": "delta - (delta.s_0)s_0 normalized + projected (the "
                    "letter's formula is the g_0-form; the committed first "
                    "step is AdamW's — both ride; NEVER adjudicated)",
            "delta_dot_s0": c_ds,
            "orth_cos_vs_s0": cos64(delta_removed_dform, s0_cons),
            "settled_plus1": cellD_coread["settled_plus1"],
            "raw_plus1_coreported": cellD_coread["raw_plus1_coreported"],
        },
        "note": "the intervention read has NO committed reference — the "
                "openness is the point",
    }
    cells["g1e_10912__removed_delta"] = {
        "role": "INTERVENTION 2 — THE COMPONENT-REMOVAL (the cons root "
                "under its first step minus the s_0-component)",
        "delta": "g_0 - (g_0.s_0)s_0, normalized to the same L2 (the "
                 "letter's formula)",
        **cellD}
    METRICS["gates"]["cellD__G_REMOVAL"] = gD
    log(f"  [D] cons-removed-delta  : THE COMPONENT-REMOVAL — settled +1 "
        f"{cellD['settled_plus1']:.4f} (raw "
        f"{cellD['raw_plus1_coreported']:.4f}); "
        f"{'SPARED (>=0.5)' if cellD['settled_plus1'] >= CRUSH_BAR else 'CRUSH (<0.5)'}"
        f"; g0.s0 {c_gs:+.4f}; |delta'| {gD['delta_prime_l2']:.4f}; orth "
        f"{orth_cos:+.2e}; delta-form co-read "
        f"{cellD_coread['settled_plus1']:.4f}: "
        + ("PASS" if gD["pass"] else "FAIL"))
    if not gD["pass"]:
        fail_texture("cell D: the removal arithmetic gate FAILED")
        return
    write_partial("cell D — the component-removal")

    # ---- the first-order ledger (co-read, never adjudicated) ---------------
    ledger = {}
    for key, tag, cell, delta in (
            ("A_locked_own", LOCKED_TAG, cellA, delta_map[LOCKED_TAG]),
            ("B_locked_cons", LOCKED_TAG, cellB, delta_B),
            ("C_cons_own", CONS_TAG, cellC, delta_map[CONS_TAG]),
            ("D_cons_removed", CONS_TAG, cellD, delta_removed)):
        disp = (R_CLAIM / float(torch.norm(delta))) * delta
        first_order = dot64(_SRAW[tag], disp)
        actual = cell["settled_mean_log_pz"] - fact_readout_root(tag)
        ledger[key] = {
            "first_order_dlogpz": first_order,
            "actual_dlogpz_settled": actual,
            "note": "first-order (s_raw . displacement) vs the actual "
                    "settled change in mean log p(Z) — the honesty "
                    "companion: first order predicts the crush or its "
                    "absence not at all if these disagree in sign/scale",
        }
    METRICS["first_order_ledger"] = ledger
    for k, v in ledger.items():
        log(f"  first-order[{k}]: predicted {v['first_order_dlogpz']:+.4f} "
            f"vs actual {v['actual_dlogpz_settled']:+.4f} (mean log pZ)")

    METRICS["cells"] = cells
    write_partial("four-cell table complete (ledger included)")

    # =====================================================================
    # THE ADJUDICATION (frozen composite; no bar shopping)
    # =====================================================================
    readB = cellB["settled_plus1"]
    readD = cellD["settled_plus1"]
    transplant_crush = bool(readB < CRUSH_BAR)
    removal_spare = bool(readD >= CRUSH_BAR)
    if transplant_crush and removal_spare:
        verdict = "DELTA-CARRIES"
    elif (not transplant_crush) and (not removal_spare):
        verdict = "ROOT-RESPONDS"
    else:
        verdict = "GRADED"
    table = {
        "locked_own_delta": {"settled_plus1": cellA["settled_plus1"],
                             "committed": ROOTS[0]["committed_plus1"],
                             "read": "HOLD (reference)"},
        "locked_cons_delta": {"settled_plus1": readB,
                              "read": ("CRUSH (< 0.5)" if transplant_crush
                                       else "HOLD (>= 0.5)")},
        "cons_own_delta": {"settled_plus1": cellC["settled_plus1"],
                           "committed": ROOTS[1]["committed_plus1"],
                           "read": "CRUSH (reference)"},
        "cons_removed_delta": {"settled_plus1": readD,
                               "read": ("SPARED (>= 0.5)" if removal_spare
                                        else "CRUSH (< 0.5)")},
    }
    METRICS["adjudication"] = {
        "bars_verbatim": {k: REGISTERED_BARS[k] for k in
                          ("DELTA_CARRIES", "ROOT_RESPONDS", "GRADED")},
        "operationalization": REGISTERED_BARS["operationalizations"],
        "constants": {"crush_bar": CRUSH_BAR},
        "four_cell_table": table,
        "clauses": {
            "transplant_crushes_locked": transplant_crush,
            "removal_spare_cons": removal_spare,
            "delta_carries": bool(transplant_crush and removal_spare),
            "root_responds": bool((not transplant_crush)
                                  and (not removal_spare)),
        },
        "co_reads_never_adjudicated": {
            "raw_outside_ball_plus1": {
                "A_locked_own": cellA["raw_plus1_coreported"],
                "B_locked_cons": cellB["raw_plus1_coreported"],
                "C_cons_own": cellC["raw_plus1_coreported"],
                "D_cons_removed": cellD["raw_plus1_coreported"],
                "note": "the C-arm convention: the read at the raw 1.6543-L2 "
                        "endpoint BEFORE the wall settles it (g11's "
                        "raw-erasure co-read)",
            },
            "delta_form_removal": gD["coread_delta_form"],
            "first_order_ledger": ledger,
            "transplant_class_confound": {
                "cos_delta_locked_vs_cons":
                    gB["class_confound_cos_delta_locked_vs_cons"],
                "note": "the transplanted delta is a DIFFERENT draw of the "
                        "delta class (55% shared hot set, g11) — a hold at "
                        "the locked root is ambiguous between 'state-side' "
                        "and 'not the crushing vector at this geometry'; "
                        "named, the removal cell carries the discrimination",
            },
        },
        "verdict": verdict,
    }
    log("=" * 78)
    log(f"THE VERDICT: {verdict}")
    log(f"  [A] locked-own-delta   +1 {cellA['settled_plus1']:.4f} "
        f"(committed 0.9452) — the reference HOLD")
    log(f"  [B] locked-cons-delta  +1 {readB:.4f} — the transplant "
        f"{'CRUSHES' if transplant_crush else 'HOLDS'}")
    log(f"  [C] cons-own-delta     +1 {cellC['settled_plus1']:.4f} "
        f"(committed 0.2719) — the reference CRUSH")
    log(f"  [D] cons-removed-delta +1 {readD:.4f} — the removal "
        f"{'SPARES' if removal_spare else 'still CRUSHES'}")
    write_partial("adjudication")

    # =====================================================================
    # THE CHART
    # =====================================================================
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("G12 — the crush intervention (T196's decisive cell): the "
                 "cross-transplant + the component-removal (CPU, eval-only)",
                 fontsize=12)

    # (a) the four-cell table
    ax = axs[0, 0]
    keys = ["locked_own_delta", "locked_cons_delta", "cons_own_delta",
            "cons_removed_delta"]
    labels = ["[A] locked\nown delta", "[B] locked\nCONS delta\n(transplant)",
              "[C] cons\nown delta", "[D] cons\nREMOVED delta\n(s$_0$ out)"]
    vals = [table[k]["settled_plus1"] for k in keys]
    cols = ["#2b6cb0", "#c05621", "#2f855a", "#b7791f"]
    xs = np.arange(len(keys))
    ax.bar(xs, vals, 0.6, color=cols)
    ax.axhline(CRUSH_BAR, color="r", ls="--", lw=1.2,
               label="the crush bar 0.5")
    ax.axhline(_root_gm12(LOCKED_TAG), color="#2b6cb0", ls=":", lw=1,
               label="locked root g-12 0.916")
    ax.axhline(_root_gm12(CONS_TAG), color="#2f855a", ls=":", lw=1,
               label="cons root g-12 0.857")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("settled +1 g-12 light")
    ax.set_title(f"(a) THE FOUR-CELL TABLE — verdict {verdict}")
    ax.legend(fontsize=7)

    # (b) the cross-transplant
    ax = axs[0, 1]
    ax.bar([0], [_root_gm12(LOCKED_TAG)], 0.55, color="#bbbbbb",
           label="locked root")
    ax.bar([1], [cellA["settled_plus1"]], 0.55, color="#2b6cb0",
           label="[A] + own delta (0.945 committed)")
    ax.bar([2], [readB], 0.55, color="#c05621",
           label="[B] + CONS delta (the transplant)")
    ax.plot([3], [cellC["settled_plus1"]], "_", ms=22, color="#2f855a",
            label="[C] the same delta at its own root (0.272)")
    ax.axhline(CRUSH_BAR, color="r", ls="--", lw=1.2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["root", "own delta", "CONS delta",
                        "at its own root"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("settled +1 g-12 light")
    ax.set_title("(b) THE CROSS-TRANSPLANT — does the delta carry the crush?")
    ax.legend(fontsize=7)

    # (c) the component-removal
    ax = axs[1, 0]
    ax.bar([0], [_root_gm12(CONS_TAG)], 0.55, color="#bbbbbb",
           label="cons root (g1e)")
    ax.bar([1], [cellC["settled_plus1"]], 0.55, color="#2f855a",
           label="[C] + own delta (0.272 committed)")
    ax.bar([2], [readD], 0.55, color="#b7791f",
           label="[D] + delta' (g$_0$ minus s$_0$-component)")
    ax.bar([3], [cellD_coread["settled_plus1"]], 0.55, color="#d69e2e",
           alpha=0.6, label="co-read: delta minus s$_0$-component "
           "(never adjudicated)")
    ax.axhline(CRUSH_BAR, color="r", ls="--", lw=1.2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["root", "own delta", "REMOVED (g$_0$)",
                        "REMOVED (delta)"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("settled +1 g-12 light")
    ax.set_title("(c) THE COMPONENT-REMOVAL — was it the s$_0$-component?")
    ax.legend(fontsize=7)

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
    ax.set_xticklabels(["[A] locked\nown", "[B] locked\ncons",
                        "[C] cons\nown", "[D] cons\nremoved"], fontsize=8)
    ax.set_ylabel("change in mean log p(Z)")
    ax.set_title("(d) the first-order ledger — first order predicts "
                 "the crush not at all (co-read)")
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RD / "crush_intervention.png", dpi=130)
    log(f"[chart] saved {RD / 'crush_intervention.png'}")

    # ---- the final write -----------------------------------------------------
    METRICS["honesty_reflex"] = {
        "n1_per_cell": "n=1 per cell (single draws, single single-step "
                       "interventions); every number is one application's "
                       "read; no replicates here",
        "transplant_whole_delta": "the transplant moves the WHOLE committed "
                                  "delta — the treatment is the delta-class "
                                  "identity (55% shared hot set), not a "
                                  "located component; the class-vs-vector "
                                  "confound is named in the adjudication's "
                                  "co-reads",
        "removal_one_direction": "the removal removes ONE unit direction "
                                 "(s_0) from the gradient form of the step; "
                                 "'the damage is elsewhere in the step' "
                                 "covers the entire orthogonal complement, "
                                 "not a located mechanism; the delta-form "
                                 "co-read rides ungated",
        "readout_single": "the read is one behavior-level readout (the +1 "
                          "g-12 light battery, 60 prompts); no wash "
                          "continuation — the +2 re-capture phase is not "
                          "tested here",
        "open_bits": "whether the crush travels with the delta or with the "
                     "root's neighborhood, and whether the s_0-component "
                     "carries the interaction — the four reads answer at "
                     "the single-draw tier; replicates (g1f's root; a "
                     "second cons draw's delta) are the successor tier",
    }
    METRICS["reference"] = {
        "plus1_ledger": {"locked_g1b": 0.9451885223388672,
                         "g1e_10912": 0.2719059884548187,
                         "g1f_10913": 0.25772199034690857,
                         "wash_root_family": "0.82-0.96 (g1bR 0.9623/0.9099;"
                                             " g1c 0.8214)"},
        "g11_geometry": {"delta_topk_overlap_locked_g1e": 0.5535,
                         "sens_topk_overlap_locked_g1e": 0.231,
                         "cos_delta_delta_locked_g1e": 0.09367704712621619,
                         "cos_g0_pos_s0_g1e": 0.09619380847574797,
                         "first_step_disp_all_roots": 1.6542880535125732},
        "sources": ["runs/g11/metrics.json (the three-root geometry)",
                    "runs/g1e/metrics.json", "runs/g1b/metrics.json",
                    "THINKING T196/T194",
                    "lab/g11_crush_mech.py (the instruments carried)",
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
                         "lr": LR_ADAMW, "crush_bar": CRUSH_BAR}
    write_partial("final")
    log(f"DONE — verdict {verdict} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
