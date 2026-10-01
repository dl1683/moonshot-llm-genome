# THE CHART CELL (e189+e190 merged; design draft v1, 2026-10-01 ~13:18Z)

Status: DESIGN-DRAFT (the R59-ideator's merge executed: the census
and the subspace test share the gradient history, the organisms,
and the figure — one dispatch instead of two). Builds on: e189's
draft (scratch/e189_design.md), e190's draft (scratch/e190_design.md),
W024/W025, opt1's committed step data, e191/e192's ray machinery.
Supersedes both drafts at dispatch.

## Why merged

The census reads WHICH COORDINATES carry the pump and the erosion;
the subspace test reads WHICH DIRECTIONS carry death. Both load the
same wash-gradient history (the e180 snapshot diffs + opt1 step
deltas), the same fact gradients, and the same organisms — and the
paper needs them on ONE chart (census x subspace x profile = the
terrain's full chart; day7 skeleton). One run, one figure, six
bars.

## The cell (eval-only CPU)

PART A — THE CENSUS (e189's spec verbatim): at the e180 snapshots +
opt1's step states: the wash gradient g decomposed by magnitude
class (top-k |g| for k in {0.1%, 1%, 10%} + the continuous
percentile curve); per part: cos(part, grad fact) and ||part||
share; the ADAM-VIEW flip census (which classes change sign under
normalization).
PART A-B RIDER — THE SHUFFLED-SIGN RAY (e192's open question:
what do g and sign(g) share that isotropic lacks? The sign PATTERN.
One more static ray: sign(g_0) with its coordinate assignment
SHUFFLED WITHIN magnitude deciles — same sign census, same magnitude
profile per decile, wrong coordinate-sign pairing. If it pumps,
structure means magnitude-profile+sign-census; if it is inert, the
pump needs the EXACT coordinate-sign pairing (truly
gradient-structural). D grid {0.05..2.0}, both rulers, dual
currency — a ray, not a bar; report-only.)
PART B — THE SUBSPACE (e190's spec verbatim): the SVD basis of the
step history (r in {64, 256, 1024}); the in-span random arm vs the
out-span-projected wash arm at the rung ladder, both rulers; the
participation ratio vs kappa^-2 * d.
SHARED: all gradients computed ONCE (load e188's artifacts where
they exist — never recompute); the two parts' figures combined into
THE CHART (one canvas: census panel + subspace panel + the e191/e192
profile panel as context).

## Registered bars (frozen at dispatch; both drafts' letters)

  A1 STITCHES-AND-CUTS / A2 FLAT-POSITIVE / A3 NO-STRUCTURE (the census trio — W024's mechanism).
  B1 SUBSPACE-CARRIES / B2 PARTIAL-PROJECTION / B3 SUBSPACE-REFUTED (the subspace trio — W025's picture).
  No cross-part bar (the parts adjudicate independently; the chart
  is the synthesis, not a gate).

## Honest failure modes

The finite-span proxy (r << true d_eff possible — sparing on the
in-span arm is ambiguous, reported as such); snapshot quadrature;
organ n=1 per ruler; the census's alignment floor 0.02 declared;
e192 (running) may already license parts of B — the chart cell
LOADS e192's rays where they overlap rather than duplicating.

## Envelope

Eval-only CPU; queues behind the current wave (e192/e182c/opt1b3);
one dispatch, progressive PARTIAL writes.
