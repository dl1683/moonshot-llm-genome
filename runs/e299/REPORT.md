# E299 — THE EMBALMING CURVE, AGE-0 HALF — REPORT

**Verdict: ALWAYS-RESUSCITABLE + MASS-TRACKS.** every corpse above the write-mass floor (50% of t0 in-room mass 8.6663) — all 7 of 7 — resurrects: the factors run 3.1x-10391x over their dead reads and the recovered fractions 0.148-1.0479 of baseline; the surgery works wherever mass survives AT AGE 0 — the boundary is the aging continuations' to set (the GPU half); DISCLOSED: 2 corpse(s) (e290-0.004X, e288-NAME-FIXED-TWIN) had their multiplier SATURATED below the 10x token (survival too shallow for 10x even under perfect surgery) — counted by the registered full-restoration clause (fraction >= 90%), both readings in the table; the recovered fraction tracks the surviving in-room mass (Spearman rho +0.883 >= 0.8) — BUT the discrimination is BETWEEN-CLASS at age 0 (the mass axis is quasi-binary: 3 values at 3dp; the collinearity disclosure stands; the aging continuations de-confound)

THE QUESTION: how long does a dead memory stay resuscitable by x14's
subtraction surgery? This is the age-0 half: seven corpses across three
kill classes and 2.7 orders of kill depth
(0.0297 -> 14.4539), each operated on by x14's Arm A
verbatim (subtract the out-of-room drift component, read the g0 probe),
each value-bound to its run's committed literals, and the instrument
itself reproducing x14's committed Arm A on the anchor corpse
(|d| 0.0e+00; G_PORTX14).

## (a) The corpse inventory (all states on disk; ages all 0)

| corpse | checkpoint | class | committed t400 g0 | kill depth (drift) |
|---|---|---|---|---|
| e283-CONCURRENT | runs/checkpoints/…post.pt | free-adamw | 9.046e-06 | 14.4539 |
| e285-TWIN | runs/checkpoints/…post.pt | free-adamw | 3.896e-06 | 14.4322 |
| e285-SANCTUARY | runs/checkpoints/…post.pt | orthogonal-sgdm | 8.025e-04 | 1.8445 |
| e290-0.1X | runs/checkpoints/…post.pt | orthogonal-sgdm | 2.571e-03 | 0.5350 |
| e290-0.02X | runs/checkpoints/…post.pt | orthogonal-sgdm | 1.585e-02 | 0.1277 |
| e290-0.004X | runs/checkpoints/…post.pt | orthogonal-sgdm | 8.453e-02 | 0.0297 |
| e288-NAME-FIXED-TWIN | runs/checkpoints/…post.pt | orthogonal+in-room-maint | 4.296e-02 | 1.7946 |
| e290-0.0008X (ALIVE boundary, holds 0.767x) | …0.0008X_post.pt | orthogonal-sgdm | 2.031e-01 | 0.0065 |

The t100-300 milestone states were never saved (only the t400 finals) —
the age axis is degenerate at 0 in this half (disclosed); the kill-depth
axis is the deliverable. Out-of-scope states (e286/e287/e289's arms,
e280/e284's other-vehicle posts, the optimizer resume artifacts) are
inventoried in metrics.json and excluded (the dispatch named
e285 + e290 + e288 + x14's e283 anchor).

## (b) The resurrection table (the surgery per corpse)

| corpse | class | depth | drift in-room frac | corpse g0 | mass ratio | factor | max attainable | fraction of baseline | resurrected |
|---|---|---|---|---|---|---|---|---|---|
| e283-CONCURRENT | free-adamw | 14.4539 | 1.14e-01 | 9.046e-06 | 0.8435 | 4328.2x | 29255.3x | 0.1479 | YES |
| e285-TWIN | free-adamw | 14.4322 | 1.13e-01 | 3.896e-06 | 0.8441 | 10390.9x | 67929.9x | 0.1530 | YES |
| e285-SANCTUARY | orthogonal-sgdm | 1.8445 | 1.23e-06 | 8.025e-04 | 1.0000 | 329.8x | 329.8x | 1.0000 | YES |
| e290-0.1X | orthogonal-sgdm | 0.5350 | 4.21e-06 | 2.571e-03 | 1.0000 | 102.9x | 102.9x | 1.0000 | YES |
| e290-0.02X | orthogonal-sgdm | 0.1277 | 1.76e-05 | 1.585e-02 | 1.0000 | 16.7x | 16.7x | 1.0000 | YES |
| e290-0.004X | orthogonal-sgdm | 0.0297 | 1.67e-04 | 8.453e-02 | 1.0000 | 3.1x | 3.1x | 1.0000 | YES |
| e288-NAME-FIXED-TWIN | orthogonal+in-room-maint | 1.7946 | 5.87e-03 | 4.296e-02 | 1.0003 | 6.5x | 6.2x | 1.0479 | YES |
| e290-0.0008X (ALIVE) | orthogonal | 0.0065 | 2.40e-03 | 2.031e-01 | 1.0000 | 1.30x | 1.30x | 1.0000 | (boundary) |

The write-mass floor: 50% of the t0 in-room mass 8.6663
(x14's EROSION_BAR carried); every corpse passes it (mass ratios
0.8435-1.0003).

## (c) The curve + the mechanism statistic

Kill depth spans 2.69 orders
(0.0297 -> 14.4539). The recovered fraction vs the
surviving in-room mass: **Spearman rho +0.8829**
(bar >= 0.8; tie-coarsened +0.8847; Pearson
+0.9992); the factor-scale co-report **-0.5000**
(the registered inversion FIRED: the multiplier ANTI-correlates with mass — denominator saturation, disclosed at birth). The mass axis is quasi-binary at age 0
(3 distinct values at 3dp): the discrimination is
BETWEEN-CLASS (free ~0.844 vs orthogonal ~1.000) — the collinearity debt
is registered for the aging (GPU) half. The orthogonal corpses' own
survival vs depth reproduces T269's inverse law (log-log slope
-1.144
vs the committed refit -1.005); the free-class corpses sit ~3 orders
below the 1/drift line — different kill animals.

## (d) The null model — T269's clock vs the mass clock

T269/R69's read clock: survival ~ 2^(-t/1040) (fitted on the holding
rung). The free-class corpses' implied MASS clock: half-life
1629-
1636 steps
(per-corpse arithmetic in metrics.json) — **the read dies ~
1.57x faster than
the mass erodes** under the free stream, and the orthogonal classes'
mass clock is INFINITE (mass preserved by construction while the read
dies — T265's dissociation). At age 0 both clocks predict zero
additional decay — this cell measures the initial condition. The
discriminating prediction handed to the GPU half: under MASS-TRACKS the
resurrection fraction's aging half-life ~= the MASS clock (~1.6k steps
on the free class); under read-clock coupling it is ~1,040 steps.

## The gates (all PASS)

13 hard gates: the probe identity (namefree / splice 19+41 /
battery shapes); 8 parent records md5-bound + the fact (md5/size/step +
flat-md5) + the 7 corpses + the boundary (design-time md5s) + the room
file + the span; the base fact-free; the root read-bound; the room
certified (idem/kept^2/span) and BIT-BOUND to e264_rooms.pt; the fact
behaviorally bound (|d g0| 0.0e+00);
every corpse + the boundary value-bound behaviorally + geometrically
(max |d g0| 0.0e+00;
max |d drift| 6.3e-12);
and G_PORTX14 — the instrument reproduced x14's committed Arm A
(0.0391536690) on the anchor corpse.

## Disclosures

- THE SATURATION CLAUSE (registered at birth, before compute): the 10x
  token is unattainable where survival > 0.10 (full restoration caps the
  multiplier at 1/survival); those corpses count by full restoration
  (fraction >= 90%). Both readings are in the table.
- THE COLLINEARITY DEBT: at age 0, surviving mass and kill class are
  collinear (the orthogonal streams preserve the room BY CONSTRUCTION);
  MASS-TRACKS here is a between-class discrimination — the aging
  continuations de-confound.
- THE ADDITIVITY NULL (x14 verbatim): resurrection => the out-of-room
  context carried the kill; a sub-bar read never proves
  unresuscitability — every 'resuscitated' is a LOWER BOUND.
- CPU-ONLY desk cell: no training, no stream, no steps; reads + fp64
  projections; torch threads 4, pocketfft workers 2; NO envelope-log
  writes; NO cuda tensors.
- n=1 per corpse, one lineage, one session; nothing guaranteed.

## Provenance

Birth commit db5a4e0ec75e078b9a45a53089ad09a05ee7cfca (bars + conventions + operationalizations +
predictions, BEFORE any compute); full run this commit. Machinery:
e261's SRCT/LadderRooms ported whole by import (the committed file
untouched); the surgery is x14's Arm A arithmetic verbatim in fp64. No
NOTES/THINKING/QUEUE/STATE edits.
