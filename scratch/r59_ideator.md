# R59 IDEATOR — rank and trim: the queue against the terrain picture

2026-10-01, frontier review R59. Inputs read: AGENTS.md, SUPERVISOR.md
(C13 items), README, STATE.json (fleet: g1bS GPU base-chunk-5, opt1b2
CPU step-1116 journal 516 rows), NOTES.md (e191, opt1c, opt1b, e188,
g1bW, opt1, g3K), THINKING.md T141–T144 + W024–W026 + T137/T138/T139
context, scratch/day7_skeleton.md (the three-slices framing),
scratch/day6_paper_skeleton.md (R1–R6, figure plan, cut-list),
QUEUE.md, REVIEWS.md R56–R58, and the specs: e189_design.md,
e190_design.md, r58_ideator.md (opt2/g9), r56_ideator.md (g6/g7/g8),
g2g_design.md, g1bS_design.md, r58_critic.md 2d (g1bW2), plus live
partial metrics for both computing cells.

## The terrain picture the queue must answer to (one paragraph)

After T141–T144 the object is charted, not just sampled: death is a
raw-displacement cliff whose edge depends on the DIRECTION CLASS —
the raw-gradient ray dies at D 0.92 (pump ridge 0.94–0.96 across
0.05–0.50, edge [0.80,0.92], floor 2.0 — TERRAIN, not overshoot:
e191); the sign-normalized (Adam) class dies at ~2.5; random rays at
4–10x (g3K). Three trajectory classes at matched D 1.6543: guillotine
0.678 / annihilation 0.0007 / bleed 0.79 — and the bleed is ALIVE
(~0.83) at the D where its own straight ray is dead, so the candidate
protective principle is RE-ORIENTATION off the lethal ray (W026's
managed bleed; the rhythm as deliberate re-orientation; the wall as a
displacement cage). Two questions own everything else: (Q1) WHAT IS
THE TERRAIN MADE OF — the three slices (e189 census = which
coordinates, e190 subspace = which directions, e191 profile = which
magnitudes, DONE); (Q2) WHERE DOES THE RE-ORIENTING PATH ITSELF DIE —
opt1b2, running. Everything in the queue is now either a slice of Q1,
a protection-family cell (W026's caging/dodging/reshaping), a scope
debt on a licensed claim, or a zombie of a superseded framing.

## The ranked queue

### CPU lane (after opt1b2 lands; all eval-only unless noted)

1. **e189 — the gradient census (KEEP, dispatch first).** Pure disk
   read; answers the question the subspace cannot: the WITHIN-SPAN
   lethality ordering (raw g 0.92 vs sign(g) 2.5 is a 2.7x spread
   inside any candidate span — a projection ratio cannot explain it;
   the magnitude classes can). Also the only owner of the W024
   sign-flip mechanism (the normalizer flips fact-relevance +0.098 →
   −0.038 on the same batch). Its census pre-registers opt2's TOPK
   predictions — fold before opt2 dispatches.
2. **e190 — the effective-subspace test (KEEP, second).** W025's
   picture: kappas as sqrt(d/d_eff). Decides whether "static basin"
   retires, AND produces the SVD-basis machinery that three other
   cells now need (g3O's amended ruler, g9's G-BASIS, e192's in-span
   rider). Highest leverage-per-minute in the queue.
3. **e182c — the GPT-2 forgetting control (PROMOTE — run beside g1bS
   NOW, per its own C13-3 instruction).** Carried since check-in 12;
   g1bS holds the GPU and the CPU lane is between heavy cells; there
   is no remaining excuse. Discharges the T123 n=1 stamp and the
   paper's external-validity sentence.
4. **e192 — THE ALL-RAYS TERRAIN MAP (NEW — spec below).** The Fig-5
   license; minutes of CPU.
5. **opt2 — the sign carrier (TRIMMED — see cuts).** Survives as the
   DENSITY cell (SIGN stretch-gate + TOPK-10/50) only; its protective
   arm is dead on arrival.

### GPU lane (after g1bS)

6. **g2g — the rhythm's controls (KEEP, first GPU slot — C13-2 order
   preserved).** Now pointed by the terrain picture, not just the R56
   critic: if the rhythm is a managed bleed, the organ's event rate
   should track the wash's displacement speed — SELF-TIMED-THERMOSTAT
   is W026's registered prediction, and the fixed-period head-to-head
   decides whether "self-timed" earns its keep in R6(b).
7. **g1c-root — the wall's root redraw (KEEP, PROMOTED above
   g1bW2).** The wall is the paper's lead licensed positive (R6a,
   Fig 4i) carrying a one-root scope (auditor repair 3). Paper debt
   outranks new-line cells; g1bS covers the SCALE axis but not the
   same-scale root-draw axis.
8. **g1bW2 — the B-dose ladder (KEEP).** Cheap; unblocks g9's
   operating point; the onset-tax paragraph gains its
   contrast-licensed dose.
9. **g6 — the function-space wall (KEEP, mid).** The noise-wound
   rider (g1b's unwallable position-acting kill) is still the g-series'
   best open question and the wall's cheapest-denomination answer
   feeds the discussion's protection taxonomy.
10. **g9 — the admission ball (KEEP, inherits e190).** Gated on
    g1bW2 by construction; its G-BASIS preflight should REUSE e190's
    SVD machinery (merge named below).
11. **g7 — the composed organism (KEEP, last).** Not a drafting
    blocker (Fig 4 is evidence-complete without it — see assembly
    order); it is the discussion's forward-looking paragraph or paper 2.
12. **g3O — the cone organ redraw (KEEP but RE-SPECCED as gated on
    e190).** Running a second organ's kappa brackets BEFORE e190 lands
    would measure the superseded quantity (a "basin") instead of the
    live one (the projection profile / d_eff). Re-register after e190
    with its ruler amended; carry e191's replicate rider (below).

### Retired / parked

- **g2h — the gate-clearing root (RETIRE to PARKED).** The amplitude
  claim is already honestly scoped draw-bound (T131/T136: timing 3/3
  roots, amplitude the lottery); T132's conditional prediction is
  recorded as untested and stays recorded; a GPU slot to win a
  strength lottery is exactly the zombie the mandate describes.
  Revisit only if a reviewer round demands a passing root or g2g's
  thermostat fires AND the paper needs the amplitude mechanism.
- **g8 — the native organ (PARK, formally).** A wonder-class
  graft-vs-grown question; no paper claim depends on it; it was never
  really in the GPU order anyway.

## The cuts and merges, named

**CUT 1 — opt2's WARMV arm.** Its bar (CALIBRATED-DEFUSE: calibrated
v holds the fact through 2x the band) is inverted by opt1c's
ordering: magnitude-informative steps are MORE lethal per
displacement, not less (raw g kills at 0.92 vs sign(g)'s 2.5;
flattening LOSES lethality — W024 inverted at the top end, T143).
WARMV can only confirm the ordering; it cannot discriminate anything
after opt1c. Keep at most as a one-line co-read if the machinery is
already written. The SIGN arm likewise demotes from discrimination to
stretch-invariance gate (fresh Adam's step-1 rule forever — its kill
in the band is near-predicted). The surviving cell is the DENSITY
question (TOPK-10/50: does a sparse |g|-selected front carry the
kill, or does death need every-coordinate total displacement?), which
composes with e189's census (the census names the stitches; the TOPK
arms remove or isolate them) and with e190 (top-|g| coordinates vs
in-span directions — two different notions of concentration, worth
one co-read table).

**CUT 2 — g2h (above).**

**MERGE 1 — e189 + e190 fold as ONE card (the chart card).** Keep
both cells and their separate bars, but the interpretation entry is
one THINKING card: census × subspace × profile = the terrain's chart
(day7's three-slices framing made literal). They answer orthogonal
slices of Q1; neither subsumes the other (the subspace explains the
4–10x outer ring; the census explains the within-span 2.7x ordering
and the sign-flip mechanism).

**MERGE 2 — g3O's ruler ← e190's projection profile.** g3O becomes
"the second organism's d_eff / projection profile", kappa brackets
secondary. Dispatch gated on e190's verdict; its SVD machinery reused.
RIDER attached: e191's stated honest replicate axis (a second
organism's g-ray profile — is the pump-cliff terrain-shaped there
too?) rides g3O's organism as one perturb-and-eval arm, discharging
e191's replicate debt without a new cell.

**MERGE 3 — g9's G-BASIS preflight ← e190's SVD span machinery.** The
admission ball's anchored subspace (install-gradient PCs) and e190's
empirical wash span are the same instrument family; freezing one
implementation saves g9's redesign and makes the two spans directly
comparable (install-span vs wash-span overlap is itself the
ADMISSION-WORKS prior).

**MERGE 4 — e191 + g3K co-report → e192 (the new cell).** Fig 5's
rays currently come from two organisms and three conventions; e192 is
the one-organism one-ruler version that makes the stitch a figure.

## The one new cell: e192 — THE ALL-RAYS TERRAIN MAP

**Why this is the gap.** The paper's centerpiece figure (Fig 5, named
in the skeleton from e191/opt1c/opt1b/opt1b2) is today a STITCH: the
g-ray profile from the e185 organism under the absolute-D convention;
Adam's ~2.5 from opt1's dynamic arms (same organism, drift included);
the random 4–10x band from g3K's organism under the rung-x-own-wash-1x
convention. R58's auditor already caught one convention mix at
exactly this scale (opt1's D-band). And one live question hides inside
the stitch with no owner: **is the Adam kill TERRAIN in the sign
direction, or trajectory?** e191 proved terrain for the g-ray only;
the sign-ray has never been jumped statically (opt1's Adam kills were
drifting fronts; opt1c jumped the g-direction, not sign(g)). If the
static sign-ray's edge brackets the 2.0–3.2 band, the guillotine's
edge is a place; if it spares past 3.2, the drifting front carries
lethality the fixed direction lacks — and the wall's radius sentence
("caps raw displacement below every class's kill-D") must be written
against the DRIFTING front's envelope instead.

**(a) Builds on.** e191/T144 (the ray-mapping machinery: graded
static single jumps, perturb-and-eval, 35 s, progressive writes — the
g-ray profile is LOADED verbatim, never re-measured); opt1c/T143 (the
t=0 gradient convention, recomputed root wash-batch gradient, fp64
direction gate, the sign(g) definition, the lethality ordering this
cell tests at the static level); opt1/T139 (the 2.0–3.2 stretch band
— used as the pre-registered bracket for the sign-ray's edge, chosen
from the BAND convention and NOT from A0's 2.489, per R58-critic K2's
circularity lesson); g3K/T137 (the random-ray convention and the
4–10x kappa band — the cross-reported contrast); e190 (the in-span
ray's basis, if landed — rider arm).

**(b) What is new.** No cell has measured more than one ray's static
profile on one organism under one ruler. The lethality ordering
(raw > sign > random) exists only as a mixed static/dynamic/cross-
organism inference; here it becomes one measured ladder family.
What is new specifically: the STATIC sign-ray (never measured, any
organism); the random band on the e185 organism under the absolute
convention; the single-figure license.

**(c) Knob and arms.** Knob: the RAY, at matched convention (static
single jumps, graded D ladder on the shared per-coordinate-RMS/L2 D
axis, e191's reader: g-12 battery + g0 + CE_R). Arms: (1) g-ray —
LOADED from e191 (reference, zero compute); (2) sign-ray —
theta_0 − D·sign(g_0), D densely graded across [0.5, ~6] to bracket
the 2.0–3.2 band; (3) random-rays — 5 fresh Gaussian rays
(magnitude-uniform, g3K's shape note co-reported), same ladder;
(4) RIDER after e190: in-span random ray from e190's SVD basis, 2
draws — the subspace's slot in the map.

**(d) Registered bars (draft).**
- TERRAIN-ORDERS: "fires if the static kill-D per ray orders
  g < sign(g) < random on ONE organism under ONE ruler (predicted
  ~0.9 / in-band / >= 4x) — the lethality ordering on mapped ground;
  Fig 5's license sentence."
- SIGN-IS-TERRAIN: "fires if the sign-ray's static edge falls inside
  [2.0, 3.2] — the Adam kill is geometry in the sign direction too;
  the guillotine's edge is a place, not a path."
- SIGN-IS-TRAJECTORY: "fires if the static sign-ray spares (fact >=
  0.5) past D 3.2 while every dynamic Adam arm died in-band — the
  drifting front carries lethality the fixed sign direction lacks;
  W026's cage sentence is rewritten against the front's envelope."
- BAND-REPLICATES: "fires if this organism's random-ray edge band
  overlaps g3K's 4–10x after convention translation — the kappas'
  cross-organism read at matched units (the partial second leg C13-1
  wanted)." / BAND-NARROWS: "outside it — a convention artifact,
  named and measured."

**(e) Honest failure modes.** n=1 organism — the point is the
within-organism map; the organism-replicate axis rides g3O (MERGE 2),
stated on the card; CPU fp32 (e191's disclosed texture); the bracket
[2.0, 3.2] is the stretch band, not A0's kill (circularity disclosed;
the registration is written before the ladder runs); the random rays
are one magnitude-shape — the wash's own profile differs (g3K's
concentration note co-reported, not shopped).

**Lane.** CPU eval-only, ~5–10 min, runs beside anything; dispatch
after e190 if the rider is wanted in one pass, or immediately with
the rider as a later one-arm addition.

## The paper's assembly order and the three drafting debts

**Assembly order (critical path to a draftable manuscript):**
(1) fold the running cells (opt1b2, g1bS) and dispatch e189+e190 →
ONE chart card completes the mechanism section R2b and the
discussion's terrain paragraph; (2) e192 licenses Fig 5; (3) the
zero-evidence writing step the skeleton has owed since check-in 8:
move the amendment history to a claims ledger and write the
four-sentence abstract (the current abstract is still the nested-
bracket block no outside reader can parse — it blocks DRAFTING more
than any missing cell); (4) draft R1–R5 and R6 from licensed
evidence — Fig 1/2/3 are done and **Fig 4 is evidence-complete**
(wall g1bR, rhythm g2d, cone g3R+g3K — g6/g9/g7 are NOT drafting
blockers; they are the discussion's forward-looking paragraph or
paper 2); (5) the scope cells (g1c-root, g3O, e182c) land during
drafting and only touch stamps.

**The three evidence debts that block drafting, ranked by cost:**

1. **Fig 5's license — the terrain stitch (e192 + fold).** Cheapest:
   minutes of CPU. Until it lands the centerpiece figure stitches two
   organisms and three conventions, and the sign-ray terrain question
   (SIGN-IS-TERRAIN vs SIGN-IS-TRAJECTORY) decides how the wall's
   radius sentence is written. Blocks the figure and one mechanism
   sentence; costs almost nothing.
2. **The protective-principle paragraph (opt1b2 + g1bS folds — in
   flight).** Zero new dispatch; lands this session. Until it lands,
   the discussion's central clause (re-orientation as the candidate
   protective principle vs displacement-capping as the only one) and
   the abstract's mechanism sentence cannot be written — the bleed's
   own death-point is the discriminating number. Blocks the
   discussion; costs waiting only.
3. **The scope stamps on the licensed positives (g1c-root, g3O,
   e182c).** Dearest: two GPU cells + one CPU pass + folds. Blocks
   the abstract's replication clauses (wall: one-root → root-redraw;
   kappas: organ n=1 → n=2 with the amended ruler; T123: n=1 →
   controlled). Gates SUBMISSION more than drafting — the draft can
   carry the scope clauses that already exist — hence third despite
   the cost.

**The one-sentence program view:** the queue was built cell-by-cell
as the terrain emerged; ranked against the finished picture it reads
"finish the chart (e189/e190/e192), pay the three cheap debts
(e182c, g1c-root, g3O-rider), then spend GPU on the protection
families (g2g, g1bW2, g6, g9, g7)" — and the manuscript is
draftable the moment the chart card and the two in-flight folds
land.
