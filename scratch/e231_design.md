# E231 design — THE BALL-SIDE OVERLAP JOIN (T208's registered instrument, ripened)

Status: RIPE for dispatch when a lane frees (GPU-free desk+eval; CPU threads 4).
Serves: T208 (the flat phase's currency ledger — SNR dead, strength dead, fragility
dead; THE BALL remains) + T206's regime read (its named discriminator).

## The question

The wall's flat phase (what survives at 0.6–1.0x root strength through the wash) tracks
none of the root-side currencies. T178 said it inside e225's join: the flat phase runs on
the ball's own economics. Made quantitative: does each root's retention track how much of
its FACT lives inside the wash's ACTIVE SPAN — the subspace the wash's own gradients
occupy? A fact pointing along the wash's span should be continuously re-touched (and
recaptured); a fact orthogonal to it should be untouched (and either safe or lost,
depending on whether the wall's flat phase is recapture or indifference).

## The material (all committed)

- The seven e225 roots (loading conventions ported from lab/e225_one_currency.py; the
  roots reproduce committed reads — gates G_ROOT etc.).
- The fact direction per root: u0 (the t=0 post-clip sign-ray convention, e209/e225 —
  already computed per root inside e225's instrument; module-import, do not recompute).
- The wash's active span per root: the Gram-SVD basis of the root's OWN 20-step
  unwalled AdamW history (seed 10902), exactly e209's band instrument, e225's chunked
  fp64 fix (the 10M multi-chunk path; unit-checked). n=1 history per root — same flavor
  as every band read in the join family; disclosed.

## The instrument (two operationalizations, frozen now)

- PRIMARY: the in-span fraction ||P_span u0|| / ||u0||, P = the projection onto the
  top-k span PCs at the mass plateau (k chosen by the scree's knee, k recorded; the
  e209 family's spans are low-dimensional — expect k in 2..6).
- CO-REPORT: the max single-PC cosine max_j |<u0, pc_j>| (the "one door" flavor — is
  the overlap concentrated in one wash direction or spread?).

## The join and the bars (frozen)

Join: the seven (overlap, flat-phase retention) pairs — Spearman; the multiple's join
(+0.607) and the aggregate's (-0.179) quoted on the same page (three-currency ledger
completed on one figure). The cons pair and the g1d/take6 breakers ringed.

- BALL-OWNS-THE-FLAT-PHASE — "rho(primary overlap, retention) >= 0.714 at n=7 AND the
  named breakers (g1d, take6) sit ON the curve (their anomalies explained by overlap) —
  the fourth currency named: the flat phase reads the wash's books"
- REGIME-PROXY — "the overlap correlates with retention only through formation regime:
  within fresh-formation rows it holds, within exotic rows it flatlines or inverts —
  T206's regime read survives as the organizer; the overlap is its correlate"
- NEITHER — "no relation at the line (rho < 0.714 with the breakers unexplained) — the
  flat phase's currency stays unnamed; the honest bound recorded"

## Registered predictions (no retrofit)

1. T208's: overall rho > 0.714 with g1d HIGH overlap (the half-expressed fact lives
   where the wash already points — its 1.327 retention recaptured) and g1f's overlap
   ABOVE g1e's (the cons pair separating in the ball's favor, extending the pair's
   decision-order match into the ball's geometry).
2. The discriminating cell (ball-story vs regime-story): g1d vs g1f. Under BALL, both
   exotic rows' retentions are set by overlap (g1d 1.327 needs the table's top overlap);
   under REGIME, exotic rows ignore the overlap (g1d high-retention at ANY overlap).
3. The co-report's shape: if overlap concentrates in ONE wash PC for every root, the
   flat phase has a single-door geometry (echoes g12's "the normalizer sets the AIM").

## Honesty guards

- The span basis is n=1 per root (one 20-step history, seed 10902) — the 2–3x band
  lottery applies to the span too; co-report the overlap's sensitivity at a second
  seed if cheap, else disclose.
- u0 and the span share the seed-10902 stream (both from the same wash instrument
  family) — not independent objects; disclose, and run one control: overlap of u0 with
  a RANDOM orthonormal basis of the same k (the null overlap level).
- The e208/T207 distinctions restated: this is neither the fact-edge object nor the
  argmax margin; it is a GEOMETRY object (subspace membership).
- Nothing guaranteed; n=7; the GRADED discipline of the e225/e229 family applies.
