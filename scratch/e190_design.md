# e190 — THE EFFECTIVE-SUBSPACE TEST (design draft v1, 2026-09-30 ~12:18Z)

Status: DESIGN-DRAFT (W025's named cell, spec'd). Builds on: W025
(the subspace picture), e188 (RAW-WINS — displacement invariant
along gradient paths), g3K (the kappas 4-10x at the organism; the
tilt ladder), opt1 (the step history for the SVD basis), e189
(design'd; reads the same gradients). What is NEW: nobody has
tested WHERE the killing displacement lives — in an empirical
subspace or anywhere in parameter space.

## The question

W025: gradient paths lie inside a low-dim effective subspace
(d_eff ~ d/kappa^2 ~ 9-35k); random directions project at
sqrt(d_eff/d); the kappas measure the PROJECTION RATIO, not a
basin. FREE CONSISTENCY CHECK (existing data): g3K's 45-degree
tilt ladder killed at ~2x — a 45-degree tilt of a subspace vector
keeps ~0.7-0.9 of its in-subspace norm, so 2x raw D delivers
~1.4-1.8x effective D — killing. CONSISTENT. The new cell tests
the picture directly.

## The cell (eval-only CPU; perturb-and-eval like g3K)

SUBSPACE BASIS: SVD of the committed wash-step history (the e180
snapshot diffs + opt1's per-step deltas where stored; r in
{64, 256, 1024} — the span of the top-r right singular vectors is
the EMPIRICAL effective subspace; all three r's reported).
ARMS (matched raw per-coordinate RMS, g3K's convention, rung
ladder {1,2,4,8,16,32,64}x):
  A_INSPAN-RANDOM: a fresh random direction drawn INSIDE the span
     (Gaussian in the top-r coordinates, zero elsewhere, renormalized).
  B_OUTSPAN-WASH: the wash direction with its in-span component
     REMOVED (orthogonalized against the span, renormalized; report
     the removed fraction — if the wash is ~fully in-span, the arm's
     construction is documented and its D matched).
  C_WASH reference + D_FULL-RANDOM reference (both from g3K's
     committed data where possible — LOAD, do not rerun).
READS: both rulers (store g0, host battery p(Z)) per rung; the
effective-rank co-read: participation ratio of the wash-gradient
history's spectrum, compared to kappa^-2 * d.

## Draft bars (freeze at registration)

  SUBSPACE-CARRIES: "fires if A_INSPAN-RANDOM kills at rung <= 2
      while B_OUTSPAN-WASH fails to kill through 16 — the kappas
      are the projection ratio sqrt(d/d_eff); 'static basin'
      retires; the paper adopts the subspace sentence."
  PARTIAL-PROJECTION: "graded outcomes — report each arm's kill
      rung as the empirical projection profile; compare the two
      dimension estimates (kappa-derived vs SVD-derived)."
  SUBSPACE-REFUTED: "fires if A_INSPAN-RANDOM spares like the full
      random arm — the picture dies; the static/learned contrast
      needs a different mechanism (e189's census becomes the lead)."

## Honest failure modes

The span is a FINITE-sample proxy (r capped by available history;
the true d_eff ~ 9-35k may exceed any feasible r — if so, A_INSPAN
tests a conservative LOWER bound of the subspace and sparing is
ambiguous, reported as such); organ n=1 per ruler; the
orthogonalization's numerical rank documented; B_OUTSPAN at
matched raw D also has matched effective-D = 0 by construction —
if it kills anyway, something outside the span carries death
(refutes cleanly).

## Envelope

Eval-only CPU-light; queues behind opt1c/e189 on the CPU lane.
