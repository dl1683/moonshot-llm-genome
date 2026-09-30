# e189 — THE GRADIENT CENSUS OF THE PUMP (design draft v1, 2026-09-30 ~11:50Z)

Status: DESIGN-DRAFT (W024's named discriminator, now spec'd).
Builds on: W024 (the sign-flip mechanism candidate), opt1's committed
per-step gradient data (the pre-clip norms and batch provenance),
e188 (running; the currency question), opt2 (queued; the
intervention). What is NEW: nobody has DECOMPOSED the wash gradient
by magnitude class and asked which part carries the pump and which
the erosion. THE DIVISION OF LABOR: e189 READS the decomposition
(which part holds which alignment); opt2 INTERVENES (which part's
removal rescues or kills). Read first, cut after.

## The question

W024's candidate: the pump (+0.098 raw alignment) lives in a few BIG
coordinates; the erosion is a mass of tiny cuts; Adam flattens
magnitudes and the cuts outvote the stitches. THE CENSUS answers it
directly from disk.

## The cell (eval-only CPU; minutes)

At the e180 snapshots (and opt1's A0/A1 step states where loadable):
compute the wash-batch gradient g (bit-identical batch provenance,
opt1's convention) and the fact gradient grad(fact) at the same
point. Decompose g by magnitude class: top-k |g| for k in {0.1%,
1%, 10%} vs the complement; also a continuous curve (alignment of
g restricted to coordinates above percentile p, p swept). For each
part: cos(part, grad(fact)) and the part's share of ||g||. Also the
ADAM-VIEW: the same decomposition on the normalized step
(g/sqrt(v)) — where the flip happens, which magnitude classes
change sign. Report the "flip census": sign(part-alignment) before
vs after normalization, per magnitude class.

PREDICTION (W024's): top-0.1%/1% positive, complement negative;
after normalization the complement's negative alignment persists and
its effective weight (uniform lr) dominates. FALSIFIER: both parts
positive (the flip is denominator-structure, not magnitude class —
also informative, arguably better).

## Draft bars (freeze at registration)

  STITCHES-AND-CUTS: "fires if at some percentile cut the top part's
      alignment is positive while the complement's is negative (both
      |cos| >= 0.02) — W024's mechanism confirmed as structure."
  FLAT-POSITIVE: "fires if all magnitude classes align positively —
      the flip arises from the second-moment denominator's
      correlation structure; different mechanism, reported."
  NO-STRUCTURE: "alignments too small everywhere to partition — the
      flip is noise-scale; reported honestly."

## Honest failure modes

Snapshot quadrature (the census is at sampled steps, not a
trajectory); single fact family; the alignment threshold 0.02 is a
declared floor (below it, "no structure" is honest); e188 (running)
consumes some of the same gradients — e189 must LOAD, not recompute,
where e188's artifacts exist.

## Envelope

Eval-only CPU-light; sequential; runs beside the current fleet.
Priority: behind opt1c/e188 (it interprets their outputs); dispatch
when a CPU slot frees after opt1b.
