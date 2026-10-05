# agy consult #005 — at the e271 landing (the ratio curve's close + the flash surprise)

Run: 2026-10-05 ~21:15Z, `agy --effort high` (first attempt died on a wrong flag —
`--dangerously-disable-permissions` → the CLI wants `--dangerously-skip-permissions`).
Question set re-derived fresh for this landing per protocol (retention/formation split;
the creep law; the optimizer-vs-trajectory question; the missing dissection; the
self-deception probe). Stored verbatim in the session log; the full reply's five
answers mined below.

## The reply, mined DOWN (cells + kills)

**1. NORM-SCRUBBING — the third hypothesis T249's H-POISON/H-TRUE-FORMATION framing
missed (Q3). THE CONSULT'S BIGGEST CONTRIBUTION.** The organism gets HEALTHIER
(corpus CE 1.00→0.81-0.88) = corpus feature norms GROW; through LayerNorm and the
softmax denominator, the fact's RELATIVE activation mass is mathematically squeezed
on the forward pass EVEN IF THE FACT'S WEIGHTS ARE GEOGRAPHICALLY UNTOUCHED — death
by denominator, not death by overwriting. Cheap control: a PURE SGD-MOMENTUM arm
(kill AdamW entirely) — survives ⇒ Adam was the killer; dies ⇒ true spatial
interference. NOTE (our annotation): e273's separate-AdamW null does NOT kill this
(separate AdamW still grows corpus norms); the SGD-M arm is the missing control, and
a weight-vs-read dissociation read (fact-anchored coordinate movement vs probe
decline) is its second barrel.

**2. THE KINETIC RACE MODEL + THE MID-FLIGHT FREEZE (Q1).** The flash-then-decay
fits dFact/dt = α·Grad(fact) − β·Turbulence; at the 19% peak the fact's gradient
shrinks below the constant β and the scrub wins. Falsification: double the fact's
lr at the peak — a simple race resumes formation instantly. THE SHARPEST MISSING
DISSECTION: freeze the corpus exactly at the flash peak, resume pure quiet serial —
recovers to 100% ⇒ the corpus is an additive abrasive (removable); stalls/dies ⇒
the corpus poisoned the optimizer trajectory or the subsise itself. OUR NOTE: this
is the clean replacement for T244's never-run AFTER arm — and it is the K→∞ limit
of e275's K-ladder; name the freeze as e275's terminal arm rather than a new cell.

**3. THE NOISE-CORPUS ISOTOPE (Q2).** Replace the concurrent corpus with a maximum-
entropy stream (random tokens/labels). Creep exponent unchanged ⇒ the power law is
a pure statistical-mechanics artifact of the high-D walk; shifted ⇒ the law governs
semantic capacity (corpus-fact manifold alignment). Also noted: higher rank should
RAISE the collision cross-section under random interference, yet survival RISES —
"the fact escapes into orthogonal subspaces as rank grows" (an interpretation; the
isotope + the exponent measurement will adjudicate). OUR NOTE: merge with e278's
direction-null sham — ONE cell, TWO nulls (the isotope = distribution-null; the
sham = direction-null).

**4. THE ORTHOGONAL IMPLANT (Q4).** Take the quiet write's ΔW, SVD, truncate to
rank ~10, INJECT into a clean model, run the concurrent corpus. Survives ⇒
concurrent death is an OPTIMIZATION failure (gradients scramble before the circuit
forms); decays ⇒ REPRESENTATIONAL failure (the corpus actively reclaims the
parameters). OUR NOTE: this is the sharpest possible form of e279's occupancy-
retention question — MERGE: e279 becomes the implant cell.

**5. THE ADVERSARIAL BET on the capacity number (Q5).** "You are confusing
optimization volume with representational capacity" — the 609x cliff is almost
certainly a SEARCH bottleneck (the rank needed to route around saddles from THIS
init), not a storage limit; the fact itself should be low-rank. THE BET: SVD the
successful 10k quiet-write delta; top-50 singulars explain 99%. TESTABLE AT DESK:
the vehicle checkpoints + install deltas are committed (runs/checkpoints/ e260-era
resume points + e261's stored rooms). If the ΔW's effective rank is ~10-50, the
wording changes from "needs ~10k dims to express" to "needs ~10k dims of SEARCH
ROOM to form" — and the implant (item 4) closes the loop in the organism.

## Pushback (the dialogue rule — where we do NOT just take it)

- **On "e277 corpus-dose tripling is a WASTE":** ordering conceded, cell defended.
  The dose response's MEANING depends on which mechanism holds (under norm-scrubbing,
  dose scales the denominator growth; under poisoning, the v-supply), so it runs
  AFTER the SGD-M/separate-optimizer arms — but the tripling remains the
  intervention's body and the flash-invariance bar rides it either way. Demoted in
  order, not deleted.
- **On Q5's "almost certainly false":** overreach. The cliff could equally be a
  conditioning threshold of the room-restricted operator (P_room F_corpus P_room —
  the critic's named uncomputed spectrum). The SVD bet is cheap and falsifiable and
  we TAKE it — but the conditioning profile stays a live rival until measured.
- **On norm-scrubbing vs the record:** the family's displacement data shows the
  weights DO move (in-own-room 0.94→0.67; v-excess 1.01→0.24) — both things can be
  true (weights move AND the denominator squeezes). The e273 rider (weight-vs-read
  dissociation) is what separates them; the consult's control alone doesn't.
- **Instrument check (ours, this beat):** the A2 span-decomposition "zero-GPU desk"
  claim is REFUTED — the e268-e271 cells saved NO per-milestone vectors (metrics +
  logs only; checkpoints stop at the e260 era). The decomposition needs one
  instrumented re-run (~10-15 GPU min, vector journaling at s100-s400).

## The resulting queue surgery (applied to QUEUE.md this beat)

- e273 EXPANDED: separate-AdamW + SGD-M + the weight-vs-read dissociation rider.
- e275 GAINS the mid-flight freeze as its terminal arm (K→∞ = freeze-at-peak,
  resume quiet — T244's AFTER arm finally run, in its sharpest form).
- e278 BECOMES the two-null cell (isotope + direction sham).
- e279 BECOMES the orthogonal implant cell.
- NEW desk cell x6: THE SVD BET (the write delta's effective rank — takes/denies
  the adversarial wording change on the capacity number).
- e277 (corpus-dose tripling) demoted behind e273 per the pushback above.
