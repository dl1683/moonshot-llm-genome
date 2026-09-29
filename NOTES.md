# Lab Notebook

Append-only. Newest entries at the top. Format per experiment:

```
## E### — name (date)
WHAT WE DID / WHAT WE SAW / WHAT'S NEXT
```

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---



## E185b — the neutral type cells: TEXTURE — no survivor on any type; the honest cross lands at +100 (not +50); the dwell collapses FASTER on neutral (2026-09-28 ~20:15Z) — DONE

WHAT WE DID: e176N arm A verbatim on the site type (e151_twodoor)
and the dwell peak (e152_steps32); gates bit-reproducible; the
extinction priors loaded for the four-way overlay.

WHAT WE SAW (T115): TEXTURE — BOTH-DISSOLVE fails by 0.08 (the
site's +50 onset 0.3496 vs the 0.27 bar; it crosses at +100 —
the SAME checkpoint as its extinction run); EITHER-SURVIVES
never close (no type retains >= 0.5 past +50). NO MEMORY TYPE
SURVIVES NEUTRAL STREAMS — "all types" is false in no direction
that matters; the by-+50 form is not earned (the honest cross
lands at +100). THE DWELL COLLAPSES FASTER ON NEUTRAL (first
under bar at +1 vs its extinction run's +50 — the neutral
stream kills the dwell peak on the first gradient step, joining
the family-level first-step pattern). The site's fine texture:
onset holds >= 0.5 through +4 then falls; CE transients are the
family signature. Honesty: single seed (the site's 0.35@+50 is
0.08 over — a replicate could close or widen it); the dwell
peak is itself n=1 as an object; both nets lineage-1 (the site
cell bounded until a second family runs it).

---

## E185 — the noise-gradient wash: NOISE-KILLS — the mechanism noun DIES; the kill is generic optimizer fragility, not corpus-directed (2026-09-28 ~19:40Z) — DONE

WHAT WE DID: two noise arms (iid labels; permuted targets) at
displacement-match against the real wash (inputs bit-identical
md5-gated; control bit-exact vs e176N's stored cells).

WHAT WE SAW (T114): NOISE-KILLS — both noise arms dissolve the
fact at displacement-match (and a fortiori BELOW it: first under
bar at +1); the noise deltas are ORTHOGONAL to the wash's
(cosine -0.03..-0.10 at matched norms). THE KILL IS GENERIC
TWO-STEP OPTIMIZER FRAGILITY: any AdamW step of the wash's size
destroys the readout — content-free. THE TEXTURE (the
redemption): the noise kills are COLLATERAL (CE 3.2-5.7, every
dial flat — noise wrecks the whole net) while the corpus kill is
SURGICAL (CE 2.0 -> 1.8, only the fact dies); and at +1 with
identical displacement the corpus step leaves 0.678 while noise
leaves 0.0003 — the corpus direction has one step of partiality
(buys nothing: both dead by +2). THE MECHANISM NOUN DIES: "no
robustness basin" replaces "corpus gradient flow"; the clock
re-reads as a basin-width statement (2 steps @ 1e-3 ~ 2.5
displacement; 50 @ 1e-4 ~ 5). THE PAPER'S REWRITE: the lead
finding's mechanism sentence becomes "the consolidated readout
has no robustness basin — continued training of any kind (even
content-free noise at matched displacement) destroys it; the
corpus's surgical variant kills the fact while sparing the
organism". Honesty: single input stream (10902), one noise seed
per arm (replicates owed before the noun formally moves); the
displacement currency is unweighted L2 (the cosine co-report
quantifies what the norm hides); full run executed twice
bit-identical.

---

## E157 — the lineage replication: LEAD-FINDING-REPLICATES (with a recorded bound) — the wash is n=2 families; the 2x2's phase structure is LINEAGE-SPECIFIC at n=2 (2026-09-28 ~19:00Z) — DONE

WHAT WE DID: three stages on the e098 s4305 family (0.84M) —
(A) consolidate (jitter ported, RNG-matched), (B) e176N arm A
wash, (C) the 2x2 rider; all trainings cuda (35s max, no
parking); all gates vs stored cells bit-exact or verified.

WHAT WE SAW (T113): LEAD-FINDING-REPLICATES fires — family 2's
consolidated fact DISSOLVES on the first gradient step of the
neutral stream (g-12 0.198 -> 0.0011 at +1; the g0/g+12 co-
reports — both well-expressed at root (0.578/0.591) — dissolve
at +1 with family 1's CE transient shape). THE WASH IS n=2
FAMILIES. THE BOUND: stage A's geometry door gate FAILED on one
clause (the ±12 extrapolation asymmetric — g+12 0.591 open,
g-12 0.198 not; the trained span's jitter geos all high) — the
co-reports discharge the substance. THE RIDER (report-only):
LINEAGE-BOUND — 3/4 doors flipped vs family 1's committed 2x2
(every cell shuts on family 2); the variance/placement phase
structure does NOT replicate; texture: the rider arms re-learn
strongly wherever taught (site 0.9995) while untrained
geometries stay shut — THE FACT MOVES, IT DOES NOT COHABIT.
Honesty: recipe-port RNG-matched; no device mixing in-run
(family 1's references are themselves mixed — recorded); the
smaller fresh family could be capacity texture, not structure;
n=1 per family.

---

## E184 — the seed replicates: ALL-DISSOLVE (n=3) — the lead finding's evidence COMPLETES; the seed clause discharges (2026-09-28 ~18:10Z) — DONE

WHAT WE DID: e176N's neutral arm at seeds 10903/10904 (GPU-
gated, self-heated to 83C mid-run -> CPU-parked at s175/s275,
both finished; margins 6-350x from any bar).

WHAT WE SAW (T112): ALL-DISSOLVE — both new seeds cross under
the bar in the SAME (1,2] bracket; ANY-SURVIVOR never came
close (max 0.002 vs the 0.5 bar). THE LEAD FINDING'S EVIDENCE
COMPLETES AT n=3 ACROSS SEEDS: no seed of the consolidated
line retains expression under continued training without the
fact's windows — joining three stream compositions, two lrs,
every memory type. THE TWO-STEP CLOCK ITSELF REPLICATES (3/3
seeds in (1,2] — the timing lottery lives in the DEPTH/TAIL,
not the clock: seed 10904's tail runs 2-34x slower; seed
10903's read TRANSIENTLY STRENGTHENED at +1 (0.9415, above the
root!) before collapsing — the first wash step can PUMP the
read before killing it). W019's seeds clause CLEARS (the
>=3 rule met for the direction; the field-facing line may
enter the discussion in its bounded n=3 form). THE PAPER'S
LEAD FINDING IS NOW FULLY EVIDENCED: bounded only by lineage
(e157's replication remains the one outstanding axis).
Honesty: device mixing twice (recorded per cell; margins
preclude drift flips); RNG-independent samples; 14 checkpoints
on disk under disk pressure.

---

## E183 — the filtered stream: STILL-DISSOLVES — the last residue discharged; the activity-dependence noun goes UNBOUNDED (2026-09-28 ~17:45Z) — DONE

WHAT WE DID: e176N arm A with the random half FILTERED of
host-junction windows (grep at draw time; 162/4962 rejections =
3.26% realized vs the 3.84% expected; 0 post-hoc leaks); all
gates bit-level.

WHAT WE SAW (T111): STILL-DISSOLVES — the filtered stream kills
on the SAME two-step clock (g-12 0.916 -> 0.803 (+1) -> 0.040
(+2) -> floor), kinetics indistinguishable from the neutral
stream; the whole anatomy together; CE-at-dissolution honestly
reported (1.99). BACKGROUND-CARRIED did not fire — removing the
~184 junction windows changed nothing material. THE NOUN GOES
UNBOUNDED [seed clause stands]: no memory state tested retains
expression under continued training — on ANY stream composition
run (extinction / neutral / filtered), at any lr tested, with
the fact's windows absent. THE HONEST FORM (the paper's):
bounded by one seed (timing is trajectory-specific — but the
DISSOLUTION is invariant across three stream compositions, two
lrs, and every memory type: the robust fact). W019's bar eases
to [seeds owed] — the residue clause clears. THE EPITAPH's
parenthetical updates: "one seed" remains; "3.84% of a window
still unexamined" CLEARS (examined: it was not the killer).
Honesty: 7 left-edge straddle windows remain (host-tails, no
junction signal — counted); ordinary corpus pressure at lr 1e-3
is the mechanism under test (unfilterable without emptying the
stream); single seed; the filtered arm is a RNG sibling, not a
paired-draw twin (the anchor half gated bit-identical).

---

## g1 — the anchored ball: TEXTURE (GATE FAILURE G-ROOT) — the wall works mechanically; nothing adjudicates at the size-capped organism; the 2.74M continuity cell is the discharge (2026-09-29 ~14:45Z) — DONE (bounded)

WHAT WE DID: CommittedGPT implemented (commit(R) + hard L2
projection in forward); the 0.84M organism run per the spec's
size correction; all 7 arms; every gate but G-ROOT passed.

WHAT WE SAW (T124): G-ROOT FAILED — the 0.84M consolidated
root expresses g-12 at only 0.2536 (the ±8 jitter set does not
generalize to offset -12 at this scale; the 2.74M line's 0.9156
was never in reach). Per the spec's frozen abort clause, no
WALL clause adjudicates. THE TEXTURES (reported): the wall
WORKS MECHANICALLY (displacement pinned at R + one-step fuzz
in every arm; the control free-runs to 9.29); W1 (R=0.7) holds
a HALF-EXPRESSED fact FLAT at 0.09->0.34->0.29 through +300
(FLAT-AT-PIN true; the control dead at 0.001) — a partial
maintenance signature on a partial root; the NOISE KILL is NOT
SPARED by the wall (N1/N2 dead at pinned R with CE devastated
— the F3 direction: damage beyond net displacement);
WALL-TAXES-ADAPTATION fires (+0.52 nats in-batch). THE
DISCHARGE: the registered 2.74M continuity cell (one config
line; the e131 root exists) — the size-capped organism cannot
test the arc's own bars. Honesty: the projection semantics
(armed-twin evals; settle-then-disarm for probes) documented;
the install's mask deviation registered (unlikely the driver;
cannot be excluded); one thermal migration; single seed.

---

## e182 — the GPT-2 wash: TEXTURE — pretrained facts are NOT wash-proof; the moderate lr erodes to 0.66 by +50 and falling (2026-09-29 ~14:10Z) — DONE (third dispatch)

WHAT WE DID: 10 high-recall cloze probes on GPT-2 124M
(top-1 100% at baseline); a grep-verified fact-free corpus
fine-tune at lr 5e-6 and 5e-5; recall checkpoints with a
perplexity guard; the time cap trimmed the arms at 108/80
steps (recorded; the +50 cells complete, the +200 partial).

WHAT WE SAW (T123): TEXTURE — neither bar fired cleanly, but
the direction is decisive AGAINST resistance: at the gentle lr
(5e-6) retention holds 0.993 through +50 (top-1 100% — the
hyper-consolidated probe set is untouched at this rate); at
the moderate lr (5e-5) retention falls to 0.662 by +50 (mean_p
0.528, top-1 90%) with the perplexity HEALTHY throughout
(bank_ppl IMPROVING 53.9 -> 34.7 — the model is getting
better on the corpus while losing the facts: the small-net
surgical signature, at 124M). THE CROSS-SCALE READ: the
no-basin physics TRANSLATES — the kill is lr-scaled (the
gentle rate spares within its horizon, exactly as e180's t* ~
lr^-1.16 predicts), the organism stays healthy while the facts
erode (the corpus direction's surgicality, at scale), and the
decay is slower per unit lr than the tiny-nets' two-step clock
(the basin is WIDER at 124M — the exit gradual and lr-gated
(~5-10x slower on the lr axis; NOT the two-step clock, which
does not replicate: +10 retention 0.987-1.004 at both lrs)). NO RESISTANCE: pretrained
facts are not archives either; they are practiced harder.
Honesty: the probe set is hyper-consolidated (selected for
recall >= 0.8 — the floor of what a real model knows; harder
facts would fall faster); the +200 cells time-capped; single
seed; the third-dispatch recovery provenance in metrics.

---

## g2 — the rehearsal organ (the lab's first BUILT architecture): GATE-SILENT — the organ works, the detector didn't; the one fired event self-triggered a resurrection to 0.44 (2026-09-29 ~14:05Z) — DONE (bounded)

WHAT WE DID: the committed spec implemented verbatim (bars
frozen); a fresh family-2 root built (the e157 one failed the
spec's own strength gate; the rebuild cleared 0.7106 >= 0.7
first attempt); four cells under the e176N neutral wash; all
gates PASS; 934s.

WHAT WE SAW (T122): the primary bar FAILED — CELL-G2's ruler
0.024@+50, 0.001@+300 — but NOT the way any registered
falsifier predicted: the gate fired ONCE in 300 steps (the
5-25-event economy band missed by 5x) because the monitor (mean
p over all 7 name chars) is dominated by the name's SELF-
CORRELATION channel (6/7 positions Z->E, ZE->P... sit at
0.65-0.87 under wash while the ctx->Z onset is dead — the
pre-registered coupling clause caught it: monitor-ruler
divergence 0.80, CUE-OVERFIT). THE ORGAN'S ENGINE DEMONSTRATED:
the ONE fired event (step 76, monitor 0.482) RESURRECTED the
ruler 0.024 -> 0.4435 within 24 steps — e179's resurrection
signature SELF-TRIGGERED (0.44, just under the 0.5 sawtooth
bar) — then the gate never re-opened and the fact re-died.
THE GHOST starved identically (2 events; transient 0.34 then
eroded) — e121's verdict unadjudicated, both gate cells
under-fired. SCHED (r=1/32 external): 0.693@+300 — the
maintaining endpoint replicates on family 2 (the +50 horizon
misses; the registered family-2 schedule-fragility bound
applies). BASE reproduced e157's +1 death (contrast gate).
THE ATTRUTION: the DETECTOR, not the events — g2b's delta is
named (an onset-only monitor decoupling the gate from the
self-correlation channel); a re-registration, not this run's.
Honesty: implementation deltas pre-registered (the family-2
param count corrected: 873,472, block 512); one thermal
migration recorded; single seed per cell.

---

## E187 — the noise replicates: NOISE-KILLS-REPLICATES — the no-basin mechanism formally licensed; the orthogonal kill replicates 4/4 (2026-09-29 ~12:30Z) — DONE

WHAT WE DID: finished from the outage's 15 surviving checkpoints
(3 cells eval-only, bit-gated; 1 cell minimally re-run,
reproducing its survivors bit-identically); the input stream
replayed and matched e185's md5s 10/10; a latent v1 plot bug
found and fixed.

WHAT WE SAW (T121): NOISE-KILLS-REPLICATES — all four cells
(labels-iid x2, shuffled-target x2) dead by +2 (worst 8.3e-4 vs
the 0.27 bar; ANY-SPARES never close — max 2.5e-2 vs 0.50);
displacement-match co-adjudicates (M=+2, 2.65 >= D_kill 2.489).
BOTH E185 TEXTURES REPLICATE: the labels-vs-shuffled magnitude
split (1e-3..1e-2 vs 1e-5..1e-4) and the ORTHOGONAL kill
direction (cos ~ -0.09 at +1 -> ~ -0.04 after, both arms);
the collateral-devastation profile replicates (CE 3.7-5.8 vs
the corpus's surgical 1.66). THE NO-BASIN NOUN STANDS,
FORMALLY LICENSED at n=3 draws per arm: "these memories have
no basin; what keeps them is the dataloader's direction — and
even that kills, just neatly." Honesty: one input stream, one
root, four draws; eval-only cells lack per-step training
telemetry (marked null; the re-run cell anchors transitively);
the recovery provenance documented cell-by-cell.

---

## E185c — the CPU-only tail re-run: TAIL-REPRODUCES — the tail lottery is NOT a device artifact; the dissolution clock is device-robust (2026-09-28 ~20:55Z) — DONE

WHAT WE DID: e184's seeds 10903/10904 re-run CPU end-to-end (no
device events possible); the same checkpoint grid and dial set.

WHAT WE SAW (T117): TAIL-REPRODUCES — the CPU classes match
e184's mixed-device classes at every tail checkpoint (+100:
10904-SLOWER 10/10 dials, median 2.40x; +200: 8/10, 1.80x;
+300: NOT-slower — the tails converge by the end). SEED
10904'S SLOWER AFTER-DEATH DECAY IS NOT A DEVICE ARTIFACT;
T112's tail sentence stands. THE DISSOLUTION ITSELF IS DEVICE-
ROBUST: both seeds still dissolve on the same two-step clock
CPU-only. Honesty: one lineage; the tail's magnitudes differ
slightly from the mixed run (float path); the convergence at
+300 means the lottery lives in 100-200, not the asymptote.

---

## E179 — the rehearsal-frequency law: TEXTURE + NO-PUMP — NINE replay events suffice; one replay RESURRECTS the dead; the curve is a non-monotone sawtooth (2026-09-28 ~23:00Z) — DONE

WHAT WE DID: the neutral wash with F1-replay at r in {1/32,
1/8, 1/4} (the stored r=0 wash and r=1/2 endpoints bracket);
three mid-run thermal migrations recorded; the r=0 replicate
reproduced the stored trace to 4.4e-6.

WHAT WE SAW (T120): no threshold and no gradient — the ladder
is NON-MONOTONE (sparse r=1/32 MAINTAINS 0.699 at +300 with
the whole anatomy intact; the DENSER 1/8 and 1/4 miss by
0.074/0.025; the pump does not fit, R^2 0.015). MAINTENANCE
NEEDS <= 9 REPLAY BATCHES PER 300 WASH STEPS (r* <= 1/32 at
the registered resolution). THE HEADLINERS: (1) ONE REPLAY
EVENT RESURRECTS THE +2-DEAD FACT (0.033 -> 0.686 by +50,
18 wash steps AFTER the single replay at +32) — the re-taught
state is far more wash-resistant than the consolidated root;
(2) the resurrect-and-oscillate SAWTOOTH — the +300 endpoints
sit at different cycle phases (co-reported means, no phase
shopping). THE INTEGRATION WITH THE BASIN LAW: rehearsal does
not hold the net IN the basin — it re-enters it CHEAPLY after
each exit (one event's re-entry outlasts 18+ subsequent wash
steps). T107's capacity formula stays dead (this curve is its
counterexample); the maintenance budget stands at ~9 events —
astonishingly small, not a smooth dial. Honesty: single seed
per rate, unreplicated; the maintain-failures sit inside the
sawtooth's amplitude; one pre-main bars correction recorded
(the first draft was structurally unfireable); the 1/2
endpoint stream-mismatched (conservative floor).

---

## E180 — the wash-rate law: LR-SCALED — a power law t* ~ lr^-1.16; at lr 1e-5 the fact still lives at +300 (2026-09-28 ~21:55Z) — DONE

WHAT WE DID: the neutral wash at lr 3e-5 and 1e-5 (GPU-gated,
thermal-migrated to CPU mid-run per precedent); the five-point
curve with the stored 1e-3/1e-4 cells.

WHAT WE SAW (T119): LR-SCALED — at 1e-5 the fact sits at 0.948
at +50 and NEVER crosses by +300 (0.565 — 62% expressed,
right-censored); monotone at every point. THE CURVE: t* ~
7.5e-4 x lr^-1.16 (R^2 0.975) — a slightly super-linear power
law. THE BASIN-WIDTH READING PRICED: displacement products
lr x t* = 2.0/5.0/6.0/>3.0 e-3 across a 100x lr range —
survival is DISPLACEMENT-LIMITED to first order, with a mild
super-linear drift (the gentle regime kills slower per unit
displacement — alpha 1.14-1.33 > 1; the stored-cell biases pull
opposite ways). THE PAPER'S RATE QUALIFIER: activity-dependence
as a RATE LAW, not a fixed step count — "continued training
above a rate threshold" (W019's rhetoric was already demoted;
this is its quantitative replacement). TEXTURE: the +10 pump
above root at both gentle lrs (0.954/0.942 vs 0.916 — e184's
coin-flip pump echoing). Honesty: single seed per cell (the
gentle regime unreplicated); displacement is lr x steps to
FIRST order (AdamW's normalized updates make alpha>1 partly
optimizer geometry); mixed-device trajectories (recorded); the
3e-5 crossing sits 0.006 under bar (near-bar flagged; the
bracket robust).

---

## E163 — the saturation control: DIAL-VALID — the intro's first sentence stands; the dial discriminates carriage (2026-09-28 ~21:20Z) — DONE

WHAT WE DID: e142's row-0 dial on the known non-carrier (arm_b,
7% row-0 share, 0.725 survivor) vs the carrier control; the
zero-arm bit-reproduces e133 (2.9e-7).

WHAT WE SAW (T118): DIAL-VALID — arm_b reads rel 0.275 (the
drop face) / survivor 0.725 (inside the registered 0.7-0.8
window, reproducing its ground truth) vs the carrier's 0.932
and the census band's 0.878-1.000. THE DIAL SEPARATES CLEANLY:
on a net whose fact is 7% row-0-dependent it reads 0.275, not
~1.0 — the 13/13 readings are informative about carriage. THE
INTRO STANDS LICENSED. THE NUANCE (T083's caveat survives
weakened): the MEAN arm is the saturated face (mean-replacement
drops arm_b 74%! — the "every readout loads the sink" truism
lives in the perturbation arm); the min-convention's ZERO arm
is the dial's effective face, and IT discriminates. Honesty:
arm_b's ground truth is dial-family (the zero-arm reproduces;
the mean-arm, min-convention, and controls are the independent
content); battery asymmetry discharged by the symmetric
control (carrier 0.90 on arm_b's own battery); bars fixed
before compute on both faces.

---

## E175 — the savings triple: NO-SAVINGS — the washed net re-learns at the naive price; no fast recovery under the persistent clamp (2026-09-28 ~17:05Z) — DONE

WHAT WE DID: identical HOME-site locked re-teaches (grid 10/30/
100/300) on the KILLED (persistent N2 clamp, bit-exact to
e160), the WASHED (e176's endpoint, bit-exact), and a RUN
NAIVE control (e001, the true pre-install base).

WHAT WE SAW (T110): NO-SAVINGS — steps-to-0.78 = 100 for ALL
THREE states; the washed net re-learns at the naive price (the
archive is truly empty at the threshold); FAST-RECOVERY FAILS
(0.745@30 under the persistent clamp). TEXTURES (report-only):
the washed arm LEADS naive at every sub-threshold checkpoint
(0.650/0.735 vs 0.286/0.554) — a residue visible in EARLY
kinetics that never cashes at the threshold; the killed arm's
g-12 hit 0.778 at step 10 (novel geometry recovers fast while
g0 lags — the clamp hurts the trained readout more than the
generalizing one). CE healthy throughout. Honesty: the kill is
a PERSISTENT clamp (recovery routes around it; clamped qkv get
no gradient — FAST's failure may price the operationalization,
not storage); the naive base never saw the install (substrate
familiarity differs); single seed; grid-resolution bounds.

---

## E176N — the neutral wash: NEUTRAL-DISSOLVES — the extinction escape is dead; the wash story holds across streams and rates (2026-09-28 ~16:50Z) — DONE

WHAT WE DID: three arms — (A) e176's protocol with e170's
NEUTRAL anchors (G_ANCHOR 0/16 junctions, seed-matched); (B)
the original stream at lr 1e-4; (C) restore-into-+50 (eval).

WHAT WE SAW (T109): NEUTRAL-DISSOLVES — g-12 0.916 -> 0.678
(+1) -> 0.027 (+2) -> 0.004 (+300); g0, sink, D-all, held30 all
together; CE-at-dissolution (+2) honestly 2.03 (the transient
the recovered values hide). THE EXTINCTION ESCAPE IS DEAD: the
consolidated fact dissolves on BOTH streams at the same two-
step clock. THE LR RIDER: wash rate scales with the optimizer
(under-bar +2 at 1e-3; +50 at 1e-4) but dissolution does not
need the big steps. ARM C (restore-into-+50): 60% retention —
ABOVE e178's +300 band (42%) but below the wash-depth pole
(78%): the "half-fact" is real but partially wash-depth-
inflated (interface AND depth both contribute). THE HONEST
RESIDUE: the random-corpus half (shared across all streams)
carries a 3.84%/window host-junction background (~4800 draws) —
a fully-filtered stream is the NEXT cell before the noun goes
unbounded. e177's verdict UNBLOCKS (the stream distinction
does not rescue the root type); W019's bar eases (n=1 + the
background cell owed). Honesty: single seed/arm; arm A's fine
steps main-run measured; wash timing varies across seeds
(e152R).

---

## E177 — wash the site-endpoint: SCRATCH-MEMORY [bounded: verdict on the EXTINCTION-grade stream — e176N's neutral cell still gates the noun] — knife-proof is not wash-proof; a decay GRADIENT without resistance (2026-09-28 ~16:15Z) — DONE

WHAT WE DID: e176's protocol verbatim on e151_twodoor (the
buried endpoint); site-read onset as primary; gates PASS; 0
name violations; CE healthy.

WHAT WE SAW (T108, bounded per the wait-gate): SCRATCH-MEMORY
fires — site onset 0.998 -> 0.672 @+50 (67% retention) -> 0.195
@+100 (under bar) -> 0.012 @+300; the 183-census content decays
with it (onset-arm 0.512 -> 0.008); the surviving span (0.80)
is the wreckage pattern, not an archive. KNIFE-PROOF IS NOT
WASH-PROOF — e125a's endpoint washes. THE DECAY GRADIENT: the
site-type washes ~4x slower than the dwell peak and ~24x slower
than the consolidated root through +50 (retentions 0.673 /
0.155 / 0.028) — crossing the bar at +100 vs +50: a resistance
GRADIENT without resistance. MID-WASH STATES CHECKPOINTED: at
+50 the census held full strength while the read dipped —
"content present, read degraded" nets on disk (e177_site_freeze
s50/s100/s200) for any later re-excavation. THE BOUND (e176N
pending): this ran the extinction-grade stream — under neutral
anchors the verdict may un-bound (both types surviving neutral
streams would REOPEN the resistance question entire). Honesty:
single seed; one lineage; one stream; the span/onset
dissociation mid-wash documented.

---

## E174 — the dose ladder + rehearsal: ALL-OR-NOTHING + REHEARSAL-RESOLVES — capacity is a MAINTENANCE BUDGET, not storage; cohabitation doubles it at ~zero cost (2026-09-28 ~16:05Z) — DONE

WHAT WE DID: arm A = e170's neutral install bit-exact with
in-run dose checkpoints (25/50/75/150/300; G_REPRO 0.0); arm B
= 1:1 interleaved F2-install/F1-replay (600 steps, matched 300
F2 batches).

WHAT WE SAW (T107): the dose cliff — F1 <= 0.2 at EVERY
graft-forming dose (0.785 -> 0.023 at dose 25 already; 0.0000
at 300); WIN-WIN never fires; smoke texture: F1 dead at F2-
dose TWO (first-contact-fast — any-F2-gradient-triggered, not
formation-gated). REHEARSAL-RESOLVES CLEANLY: at dose 25 (arm
A's kill point), arm B holds F1 at 0.921 WITH the graft formed
(F2 onset 0.844, census positive) — GENUINE COHABITATION at
every matched dose, F2's rate unimpaired (0.971 vs 0.994). THE
ONE-FACT LIMIT IS A MAINTENANCE BUDGET, NOT STORAGE: interleaved
replay doubles capacity at ~zero cost to F2 — T101's
life-support channel confirmed and quantified. FOR THE PAPER:
the capacity paragraph = "one fact wide without rehearsal;
cohabitation IS rehearsal" (the reading map's predicted form,
now licensed); the e171 cell stays dead (F1 survives only
under interleaving, so no per-fact-door object exists at any
protocol). PRACTICAL: a dataloader that interleaves old facts
during new installs costs ~nothing and saves everything — the
catastrophic-forgetting mitigation the field already knows,
here given its mechanism (extinction-avoidance, not
storage-capacity). Honesty: dose = step-count proxy; arm B
doubles optimizer steps (its F2-onset column rules out
rescue-by-slowdown); single seed/lineage.

---

## E178 — the reverse class-restore: TEXTURE — ~42% of expression returns with ~90% of structure (site-span 93%, brake co-carried 91% — the GAIN-ATTENUATION candidate; e176N arm C discriminates); the wash's layer signature flat vs the conversion's banded (2026-09-28 ~15:40Z) — DONE (bounded)

WHAT WE DID: e173's instrument in reverse — the root's MLP+LN
class restored into e176's washed net; graded L2-L4 arm; all
gates bit-level (the washed checkpoint verified 2e-13).

WHAT WE SAW (T106): PARTIAL RECOVERY — g-12 0.001 -> 0.422
(46.1% of the root span), g0 41.6%, held30 ~40%, sink 44%,
D-all 43% — every dial to the same ~40-45% depth (a coherent
half-fact, not a fragment); dCE +0.061. THE BRAKE RETURNS 91%
(-0.121 vs root -0.132) — the most MLP+LN-localized object in
the lab. THE GRADED SURPRISE: L2-L4 (the band that carried
the CONVERSION's closure) restores NOTHING on the wash axis
(g-12 0.018) — the wash's per-layer MLP delta is FLAT
(2.4-5.8%) where the conversion's was banded (L3 peak 29.5%).
THE READING: the washout is PARTLY the located rewrite (half
the fact is class-recoverable, bidirectionally confirming
e173's mechanism in weakened form), but the wash and the
conversion are NOT the same process run in opposite
directions — different layer signatures, and the other half
of the fact lives in what the washed attention/wpe context
withholds (restoring ALL is bit-root and fully alive). The
two-step collapse (e176) + partial class recovery (e178):
the wash destroys the class-carried half fast, and the
context-carried half with it.

---

## E176 — freeze the root: [R50 CRITICAL BOUND: the stream's anchors were the install's own name-deleted windows — EXTINCTION, not disuse; 'two steps at healthy CE' splices clocks (CE 2.000 at step 2); e176N running] the fact dissolves under the name-deleted teaching stream (2026-09-28 ~15:20Z) — DONE (bounded)

WHAT WE DID: e161's protocol verbatim on the FULLY-consolidated
root (stream-matched: same seed, same draw sequence — only the
starting net differs); all gates PASS; zero name leakage.

WHAT WE SAW (T105): the consolidated fact DISSOLVES exactly
like the dwell peak — g-12 0.916 -> 0.088 after TWO steps ->
0.001 at +300; held30, row-0 sink (0.732 -> -0.0005), brake,
D-all all to floor TOGETHER; CE healthy throughout (1.61-1.66).
NOTHING SURVIVES. The classical story — consolidation =
acquiring gradient-resistance — DIES: the consolidated memory
has no more wash-resistance than the unconsolidated one. EVERY
PAST FINE-TUNE'S ANCHOR BANK WAS REHEARSING THE FACT: the lab's
own protocol was the memory's life-support, and this is the
first time it was ever turned off. THE READING MAP'S RIDER IS
MANDATORY (e178, queued): restore the root's MLP+LN class into
the washed net — rescue would confirm washout = the located
MLP+LN rewrite (e173) operating bidirectionally. THE REMAINING
RESISTANCE QUESTION: e177 (wash the site-endpoint) — if it too
washes, the resistance axis is EMPTY for every memory type and
"memory" in these nets means "currently-being-trained"; if it
survives, the deep site-store is the one true archive. Honesty:
single seed but stream-matched vs e161; one distribution (hot
AdamW lr 1e-3 — the 2-step collapse dominates, but gentler
regimes untested); one consolidation schedule; the root had
already survived supervised fine-tunes whose anchors were,
per this result, rehearsing it.

---

## E173 — the closure partition: MLP-LN-CARRIES — the closure is a LOCATED weight rewrite; the door is reopenable by class surgery (2026-09-28 ~15:25Z) — DONE

WHAT WE DID: class-by-class restores (MLP+LN / attention /
full-wpe / io; graded per-layer MLP+LN) with the DOOR LADDER
(nested wpe-row visibility 0-141/0-171/0-183/0-195 — the
tautology fix applied prospectively) and e166's exact surgery
re-run as the gated control; all-restored gate HELD BIT-EXACT.

WHAT WE SAW (T104): MLP+LN restore alone reopens the door —
g-12 0.102 -> 0.782 (83.6% of the root's 0.916; CE +0.073),
graft INTACT and still read (site 0.998, census site_pos TRUE),
brake returns (-0.255); the reopened door is GEOMETRY-GENERAL
on the ladder (0.879/0.998/0.826). Attention: nothing (-2.5%).
Full-wpe: nothing, kills the site read. e166's row surgery
re-run: +0.0000 confirmed AS THE TAUTOLOGY (the control gate).
GRADED LAYERS (functional): a mid-late band carries it (L2 0.21 / L3 0.295 peak / L4 0.26; no single layer reaches the bar). THE LADDER's OWN FINDING: the
twodoor net is POSITION-LOCKED at its site (long-12/+12 ~0.2
vs long0 0.998) while root and the restored net are geometry-
general — the ladder sees the same phase switch. THE STRONG
"UNREOPENABLE-BY-SURGERY" DIES: doors ARE reopenable — by
CLASS-level MLP+LN surgery (not rows [tautological null], not
heads [e153 sub-bar]). Honesty: single seed/lineage; class-in-
isolation cannot exclude context-incompatibility for the null
classes (they also carry nothing on the ladder); the reopen's
CE +0.073 is cheap not free; graded reads are diagnostics
without bars.

---

## E170 — the anchor-neutral install: OVERWRITE-REAL — the anchors did NOT do it; capacity is one fact wide (2026-09-28 ~15:10Z) — DONE

WHAT WE DID: e154's MIRABEL install verbatim with NEUTRAL
anchors (16 plain-corpus windows; 0/16 host junctions vs
e154's 16/16 — the contradiction channel removed by
construction, G_ANCHOR gate); identical budget/RNG.

WHAT WE SAW (T103): F1 annihilated ESSENTIALLY IDENTICALLY —
g0 0.785 -> 0.0001 (e154: 0.0017); every dial at floor; CE
improved. F2's graft formed indistinguishably (+0.0215 @r63;
onset 0.986). The anchor channel is a NON-EXPLANATION: the
demolition travels with the graft/capacity channel. W012's
bandwidth answered the hard way: ~ONE FACT WIDE at this
budget — installing a second locked fact costs the first
everything. The two-facts question does NOT reopen (no
per-fact doors at this protocol); the paper's "globally"
clause unblocks (F1's door — and fact — close under any
second-fact install). The N2 rider correctly SKIPPED (dead F1
fails its precondition; recorded, not shopped). Honesty:
single seed/lineage; a register shift (install-position drama
text) and a ~3.8%/window shared host background in BOTH cells
recorded; s278/300 time-cap immaterial (F2 saturated from s25;
the 8-step smoke had already killed F1).

---

## E152R — the dwell re-seeds: DWELL-SEED-DEPENDENT — the dwell dies at n=3; the brake overshoot survives 3/3; the straddle cell is unstable (2026-09-28 ~15:00Z) — DONE

WHAT WE DID: two full conversion traces (seeds 10903/10904) +
the 10/12/14/16 insert (folded into 10903's trajectory) + the
straddle settler (locked@band at seeds 10905/10906); 57.5 min
GPU (gate-checked per training, never parked); all gates
bit-exact.

WHAT WE SAW (T102): DWELL-REPLICATES does NOT fire — the cliff
brackets are 8->16 / 16->32 / 128->300 across seeds; shelf
minima 0.538/0.213/0.854; T094's "8-16-step cliff" and "~50-
step dwell" are TRAJECTORY-SPECIFIC TEXTURE. WHAT SURVIVES 3/3:
(1) the CONVERSION itself (final <= 0.27 at every seed — the
door always closes by s300); (2) the BRAKE OVERSHOOT (A(129)
deepens below -0.35 mid-conversion in 3/3, deepening further
with later cliffs; released by s300 — now a real n=3
phenomenon). THE INSERT: the door stays fully open through s16
on 10903 (1.036 at s12) — timing seed-dependence at fine
granularity. THE STRADDLE SETTLER (outside its fork):
locked@band = 0.261/0.317 at new seeds vs 0.546 CPU / 0.458 GPU
— the cell SPANS OPEN-TO-SHUT across seeds; home-locking alone
CAN shut the door at some seeds; T097's two-factor gate
WEAKENED on its home leg (the honest reading: an UNSTABLE
cell, not a SHUT cell). UNDER THE DISUSE FRAME (T101): the
conversion timing is a WASHOUT RACE — when ordinary gradient
flow happens to beat the re-teach on a given trajectory; the
robust brake overshoot is the surviving mechanism signal.
Honesty: no device mixing this run (all GPU) but the seed-10902
baseline was CPU — cross-experiment drift priced; sequential
trajectories (n=3 over paths); one lineage; texture cells not
re-run (bars only).

---

## E161 — the freeze-cell: DISUSE/GENERIC-PRESSURE — the door closes with NO fact teaching; the ENTIRE fact dissolves under ordinary gradient flow (2026-09-28 ~14:45Z) — DONE

WHAT WE DID: 300 steps of plain-corpus continuation from the s32
dwell peak (zero name leakage verified at draw time); full dial
trajectory; gates bit-near.

WHAT WE SAW (T101): DISUSE fires CLEAN — g-12: 0.513 -> 0.040 at
+50 -> 0.0036 at +300 (74x under the 0.27 bar [R50 arithmetic fix; 128x was vs the start]; no fact teaching, no
graft, no anchors). COMPETITIVE dead (g never near 0.7);
DWELL-PERSISTS dead. THE BIGGER TEXTURE: this is NOT selective
door closure — the ENTIRE fact expression dissolves (g0, g+12,
the functional site read 0.965 -> 0.012, the brake -0.373 ->
+0.001, the row-0 sink 0.101 -> 0.002) while CE stays healthy
(1.61-1.69). THE DWELL-PHASE MEMORY IS NOT YET INCORRIGIBLE:
plain gradient flow washes it out in <50 steps — where e125a's
300-step endpoint site-memory survived everything. CONSOLIDATION
= GRADIENT-RESISTANCE ACQUISITION (the s32->s300 axis), and the
e151 'conversion' re-reads as: F1 washing out (as any
unconsolidated memory does under continued training) WHILE the
re-taught fact builds its graft. The 'phase switch' was
substantially FORGETTING + NEW LEARNING. THE MISSING CONTROL
(e176, queued): freeze the FULLY-CONSOLIDATED ROOT on plain
corpus — if the root's fact survives, consolidation really is
resistance-acquisition (the CLS story, licensed); if it too
dissolves, even 'consolidated' is use-it-or-lose-it and the
edifice reframes again. Honesty: single seed/trajectory; the
collapse margin (128x) dwarfs the known scatter; one
distribution tested (the anchor+random stream); s32 starting
point is one point on one trajectory.

---

## E175 — the savings triple: NO-SAVINGS — all three states re-learn at the SAME threshold price (100 steps); no fast recovery, no residue advantage, no scar penalty (2026-09-28 ~17:05Z) — DONE

WHAT WE DID: identical short HOME-site locked re-teaches on
the KILLED (N2 clamp), the WASHED (e176 endpoint), and a RUN
NAIVE control (e001); grid {10,30,100,300}; e160's kill
reproduced bit-exact.

WHAT WE SAW (T116): NO-SAVINGS — steps-to-0.78 = 100 for ALL
THREE states (the killed 0.662/0.745/0.909; the washed and the
naive matching); FAST-RECOVERY FAILS (the clamp's persistence
prices the operationalization, not storage — recovery routes
around it at the same speed); POSITIVE fails (no residue
advantage); NEGATIVE fails (no scar penalty). THE ARCHIVE IS
EMPTY AT THE THRESHOLD — the wash left nothing that speeds or
slows re-learning. THE TEXTURE (co-reported): the washed arm
LEADS naive at every sub-threshold checkpoint (0.650/0.735 vs
0.286/0.554) — a residue visible in EARLY kinetics that never
cashes at the threshold [R52's bound: the naive control's
substrate differs (never saw the install); grid-limited null].
Honesty: the clamp convention; the naive reference's
provenance; single seed; grid resolution (30,100] bounds.

---

## E154 — two facts, one door: TEXTURE [noun OVERWRITE-NOT-SHARE struck per R49 — anchor-confounded]; F1 annihilated (not door-closed) under a protocol whose anchors contradict it (2026-09-28 ~14:00Z) — DONE

WHAT WE DID: 300-step locked install of a NONCE fact (MIRABEL,
zero corpus occurrences — the novelty confound avoided) at rows
63-69 into the consolidated net; full F1/F2 dials; N2 + W017
riders; 9 gates PASS (N2 bit-exact vs e160).

WHAT WE SAW (T100): the registered bars cannot fire — F1 was
not untouched-with-door-closed, it was ANNIHILATED: g-12 0.916
-> 0.0002, g0 0.785 -> 0.0017, held30 -> 0.0001, row-0 strength
0.732 -> 0.001, A(129) -> +0.001, D-all -> 0.0003. F2's graft
FORMED (site onset 0.124, site_pos TRUE; F2 expresses 0.993 at
site) and CE_R IMPROVED (1.664 -> 1.649) — the damage is
F1-specific. OVERWRITE, NOT SHARE: no second-fact capacity at
this budget. THE CRITICAL CONFOUND (the agent's catch): the
e151-protocol anchor bank uses INCUMBENT-CONTINUATION host
windows — for a second-fact install these actively CONTRADICT
F1's home expression (8 paired anti-F1 anchors x 300 steps):
F1's demolition may be substantially ANCHOR-DRIVEN
unlearning-by-contradiction, not graft-driven; this run cannot
separate them. RIDERS (both clean): N2 kills NEITHER on the
two-fact net (F2 -3.1% — the knife's circuit-selectivity
REPLICATES on a fresh site-stored fact; F1's killable readout
no longer exists to kill); W017's prediction CONFIRMS on a
fresh fact — F2's census is DIFFUSE (top-1 0.139, 28/36 heads,
entropy 2.958 — the locked-trained redundant signature; F1's
sink-coupled was 0.352/16). Honesty: single seed/lineage;
MIRABEL nonce-clean; site at rows 63-69 carries only 64 tokens
pre-context (NEAR-mirror confound, priced: F2's deletion
footprint on F1-battery negligible — F1 was already gone).

---

## E166 — the inverse event: INVALID-BY-INSTRUMENT (R49) — the row cells were a prompt-geometry tautology (the door battery never reads rows 183-189); site/head cells real (2026-09-28 ~13:50Z) — DONE

WHAT WE DID: 29 CE-priced cells — row restore (3 modes), head
ablations (reader3/N2, both modes), joints, root ceiling +
jitter specificity controls; the agent caught stale pass-1
metrics itself and re-gated on the committed pass-2 (the
FOLD-ON-NOTIFICATION lesson, applied prospectively by an agent).

WHAT WE SAW (T099): DOOR-STAYS-SHUT — the clean graft deletion
(wpe[183:189] := root) KILLED the graft (site onset 0.998 ->
0.599) and moved the door by +0.0000 (g-12 0.1021, dCE -0.0000;
zero/mean modes identical). THE GRAFT ROWS CARRY EXACTLY ZERO
OF THE DOOR'S CLOSURE. Head ablations push the door DOWN
(reader3-zero 0.0041; N2-zero 0.0554); joints == head singles
(rows contribute nothing once heads ablated — clean
sub-additivity). Controls: root ceiling behaves (the knife
works there: N2-zero 0.0378); jitter specificity confirms
no-restore where no graft formed. THE READING: the closure is
NOT active competitive inhibition by the wpe graft — the third
great asymmetry: doors open by training, are killed by surgery,
and cannot be REOPENED by surgery (T037's write-once core
extended to re-opening; T091 bounded on the removal side). THE
HONEST FORK-NARROWING: the conversion's delta is 66.5% MLPs/LNs
(e153) that this surgery cannot touch — STAYS-SHUT cannot
separate 'destructively rewritten' from 'inhibition carried in
the stream state'; the REWRITE-vs-DISUSE fork stays open for
e161's freeze-cell, and a gradient re-teach-the-restore cell
(discriminating MLP/LN carriage) is named. Single lineage/seed.

---

## E164 — the post-kill census: SUBSTANCE-SURVIVES + MLP-WRECK-ONLY — the knife severed ACCESS not STORAGE; the four-layer model's circularity is broken (2026-09-28 ~13:35Z) — DONE

WHAT WE DID: the N2-killed state rebuilt in-memory (bit-exact vs
e160's stored cell: 70.75% @ +0.245); the full organ census on
the killed net; the MLP-coordinate plane on the site-fact.

WHAT WE SAW (T098): PART A — SUBSTANCE-SURVIVES: behind the
dead readout, the fact's machinery is intact and organized as in
the root — row-0 dependence full strength (0.918-1.03), MLP body
still consumes ~100% of the killed net's headroom, ALL root
top-5 non-N2 heads still load-bearing (L0H3 0.218...), with
sham contrasts 2.9-3.9x noise. THE KNIFE SEVERED ACCESS, NOT
STORAGE — layers 1/2 are genuinely separable. No organ or third
head restores the readout (best: L2H5 partial re-open +0.0875 —
a weak suppressor texture). TEXTURE: the band-5 brake sign FLIPS
post-kill (root -0.120 HELPS -> killed +0.140 COSTS). PART B —
MLP-WRECK-ONLY: the head plane reproduces NO-SITE-KNIFE (best
flat 21.8%; ladder saturates 43.6% @ +1.15); the MLP plane HAS
an any-CE kill (mlp_l5 alone: 92.9% @ +0.793 — the cheapest
full kill of the site-fact) but NO flat-CE kill exists ({l1,l2}
39.5% @ +0.24). The MLP third IS the incorrigible substrate —
removing it is indistinguishable from wrecking the organism;
un-killability at flat CE holds on BOTH surfaces. Honesty:
in-memory kill bit-exact (5e-6) but stateless (no dynamics);
"unchanged weights" is a construction tautology — SUBSTANCE
rests on functional probes; floor compression on the killed
baseline (bars >= 50% headroom + shams + absolutes); mode/
bank conventions differ across planes (priced).

---

## E158 — the 2x2 completion: TEXTURE (two-pass disclosure) — closure requires the CONJUNCTION novel-site x zero-variance; committed pass: jitter@183 OPEN 0.789, locked@band MID 0.458 (straddling)
[SUPERSEDES the 12:40Z fold, which used pass-1 CPU metrics (0.505/0.546 -> SITE-INDEPENDENT) while the agent re-ran on the freed GPU; the COMMITTED pass-2 (e9c4855, device-homogeneous with e151) reads TEXTURE — no bar fired cleanly. PROCESS RULE BORN: fold on the agent's completion notification, not early on-disk metrics]
WHAT WE SAW (T097, committed pass): CLOSURE-BY-PLACEMENT dead
(jitter@183: 0.789 OPEN — new-site teaching WITH variance does
not close). PHASE-BY-VARIANCE's b-clause dead (locked@band:
0.458 MID — degrades but does not shut; the cell STRADDLES the
0.5 bar across passes (0.546 CPU / 0.458 GPU) — scatter beyond
e152's ~0.03 cross-device bound; second seed owed before
canonizing b's label). ROBUST ACROSS PASSES: neither factor
alone closes; both degrade to ~50% of root 0.916; the only SHUT
cell is locked@183 — CLOSURE REQUIRES THE CONJUNCTION. DWELL
ANSWER: NO — the mixed state does not persist under variance
(site-store ~100x weaker under jitter: +0.0006 vs +0.0731).
TEXTURE GEM — THE MEMORY EMIGRATES: under jitter@183 the
home-geometry readout COLLAPSES (g0 0.785 -> 0.145) while the
novel doors stay open — the readout abandons its home for the
new territory. The brake OVERSHOOTS under jitter (A -0.132 ->
-0.610). Arm b re-sites without closing (home graft +0.0105,
A flips +0.072, sink coupling retained 0.802).

---


## E125a — the inverted knife: NO-SITE-KNIFE — an asymmetry of EXISTENCE; the generalizing memory is the removable one (2026-09-28 ~12:30Z) — DONE

WHAT WE DID: arm_b's own census (bit-tight vs e133's unread
table: L1H2 0.215 mode-robust, L0H5, L1H4, L1H1, L0H1 — nearly
DISJOINT from the consolidated net's kill ladder) + the full
escalation (B-ladder, e160 sets re-pointed, L3H5-class, X-sets,
random scatter) with split-window selection; 92 cells, 6/6
gates; cross-check on e143_near (site at 5-13).

WHAT WE SAW (T096): NO-SITE-KNIFE fires — best flat-CE drop
25.7% < 30% bar; STRONGER THAN THE BAR: no cell reached 60% at
ANY CE (ceiling 32.8% @ +0.836; site onset never below ~0.63).
N2 spares the site fact (-0.8%) — e160's control replicated.
The B-ladder is SATURATING/SUB-ADDITIVE (S 21.5% -> B4 32.8%) —
redundant population coding, the OPPOSITE of e160's
superadditive complementary circuit. Cross-check: no kill at
the second site either (best flat 3.6%; N2 spares at 1.2%) —
the un-killability replicates. TWO MEMORY TYPES, TWO FACT-HEAD
POPULATIONS (disjoint except the weakest member L0H1). THE
ASYMMETRY OF EXISTENCE: the sink-coupled fact dies 79-95% at
CE +0.25; the site-stored fact has NO kill set at any price.
Honesty: split-window selection (disjoint prompts/fillers;
sel-vs-bar agree); families registered census-independent
(non-self-fulfilling null); both modes; single lineage (near =
site-geometry replication, not family); scope = head-coordinate
surgery only (33% of arm_b's load is MLP).

---

## E162 — the READ-vs-MASS fork: MIXED — the poison is a TWO-EDGED weapon, each channel independently sufficient (2026-09-28 ~12:15Z) — DONE

WHAT WE DID: the two discriminating cells + the starvation
measurement; 186s CPU; all gates bit-tight (incl. cross-run
reproduction of e159's mass profile).

WHAT WE SAW (T095): (i) VALUE-RESTORE-UNDER-POISON HEALS
COMPLETELY — poisoned key/query geometry kept, clean values at
key 0: retention x1.000 @ g-12, x1.002 @ g0, CE cost -0.0004
(restoring value content alone erases the ENTIRE organism
damage; power audit: the poison changes v0 by 1.13x its own
norm — the cell had full power). (ii) MASS-INFLATE-ON-HEALTHY
KILLS — healthy row 0, bias swept to the exact absorber dose
(1.6927): x0.036 @ g-12, x0.301 @ g0, CE +0.572, monotone
dose-response (b=3.0: x0.337; b=4.0: x0.005). (iii) Both modes
coexist in real poison: band queries' key-0 mass explodes
~60x while band keys lose ~23% relative mass. THE NOUN'S FULL
FORM: the memory depends on the sink's DUAL ROLE — what it
SUPPLIES (content read off the pivot) and what it SPARES (the
allocation the absorber would steal); either corruption alone
kills. Honesty: the value-transplant restores the full channel
(not a minimal patch); the bias matches total dose not the
per-layer profile (a sufficiency test); single net.

---

## E152 — the conversion time-trace: TRANSIENT-TWO-DOOR — the cliff has a DWELL TIME; e151's P-b was EARLY, not wrong (2026-09-28 ~11:25Z) — DONE

WHAT WE DID: six-checkpoint sequential trace (8/16/32/64/128/300)
of e151's locked re-teach; one trajectory, seed-fixed; root gates
bit-exact; ran CPU (park-to-CPU: user game held GPU at 86-87C —
thermal rule honored; CPU/CUDA float-path equivalence shown).

WHAT WE SAW (T094): TRANSIENT-TWO-DOOR fires — the cliff runs
between 8 and 16 steps (g-12: 0.990 -> 0.447), then DWELLS on a
~0.5 shelf through step 64 (checkpoints {8, 32, 64} hold BOTH
doors: site clears e151's content bar — genuine, row-183-local,
66.8x bar-mean across dwell peaks (89x at s8; derivation per R47 audit), site read p_Z 0.962 — AND g-12 >= 0.5) before the
final descent (300: 0.139). CLEAN failed (non-monotone bounce
s16->s32; Spearman -0.60); DELAYED failed. TEXTURES: the A(129)
brake OVERSHOOTS mid-conversion (-0.132 -> -0.466 @s128) before
dissolving (the negative posterior intensifies as the graft
grows, then releases); CE_R's early wobble (1.705 @s8) recovers
to 1.643 — conversion, not damage; the 300-step endpoint
reproduces e151 behaviorally (max diff 0.029; two instrument
cells exceed strict tol, reported verbatim). Honesty: ONE
trajectory (the s16->s32 bounce is this path's, not a law);
single seed/lineage — the dwell time is a point estimate; the
~0.5 shelf partially conflates route survival with sink health
(mask/ladder columns in metrics price it).

---

## E159 — coupled-or-organism: MASK-HEALS + SITE-SPARED — the double dissociation; READ-coupled replaces sink-coupled (2026-09-28 ~11:15Z) — DONE

WHAT WE DID: the two R46-critic probes — the mask+poison joint
cell on the consolidated net, and the first-ever norm ladder on
the site-stored control; gates to 1e-7; 86s.

WHAT WE SAW (T093): MASK-HEALS — joint 0.07-under-mask heals
COMPLETELY (g0 x1.030, g-12 x1.009 @ CE +0.036, numerically
identical to mask-alone; even COMPLETE wpe[0] removal under the
mask heals: the fact needs neither row-0 content nor its
value). Manipulation check: poisoning turns row 0 into a mass
absorber (pre-mask attention on key 0 explodes 0.157 -> 1.693,
10.8x) — the kill is delivered THROUGH attention reads, not
around them (query-side/global-softmax story FALSIFIED).
SITE-SPARED — arm_b's ladder: nothing dies (0.07: x0.922 @ CE
+0.825 — the SAME organism damage the consolidated net pays at
that bracket +0.843); even full removal costs its fact only
27.5%. THE DOUBLE DISSOCIATION: equal organism damage, only the
coupled memory dies. THE NOUN: READ-COUPLED — the consolidated
fact dies of what attention READS off a degraded row 0 (the
poisoned sink's reads corrupt downstream computation); the mask
is literally the health door; a memory must itself read the
sink to die of its poison. T086's organism-death bound is
itself bounded: organism damage is real but NOT SUFFICIENT.
Honesty: joint cell doubly off-distribution (CE prices generic
damage; anchored by gated single-intervention rebuilds); single
nets (e157 owes replication); per-net retentions (bracket
comparison is the like-for-like).

---

## E153 — phase-switch surgery: TEXTURE — the order parameter is DISTRIBUTED (MLP-heavy), not any small head set; the geometry door is transplant-RIGID (2026-09-28 ~11:05Z) — DONE

WHAT WE DID: wiring diff + K-by-K head transplants (fact /
reader / mover / random classes, K=1/3/6) both directions
between the two phase nets; every cell CE-priced (all |dCE| <=
0.024 — nothing is wreckage); gates bit-exact.

WHAT WE SAW (T091): PHASE-IN-HEADS no (best reopen g-12 0.151 vs
bar 0.45 — the reader_K3 arm {L3H5,L0H3,L1H0}, +47.5%; rand_K6
also +32% — the site-phase net is fragile, movement not
specific). PHASE-DISTRIBUTED no as written (>20% movers exist —
head identity matters, partially). THE STORY: the 300 locked
steps' delta lives in MLPs (66.5% of ||d||^2) and late heads
(all six L5 heads top movers) — but transplanting the biggest
movers moves the door <4%; the door-movers (the RE-GROWN
POSITIONAL READER L3H5 — the L3H4-class namesake, one slot from
the twin's own — plus L0H3/L1H0 content heads) are mid-ranked
deltas that reopen under half the distance and never cross the
bar. ASYMMETRIC RIGIDITY: the open geometry door is transplant-
immovable (nothing closes it, best -9.1%); the shut door is
nudgable to ~1/3 of the bar. The reader set partially carries
the BRAKE both ways (A +0.005 -> -0.027; -0.132 -> -0.068 at
K6). Honesty: parameter-swap path dependence (donor heads land
in foreign LN contexts, deltas 0.40-0.71 — nulls conflate
'phase not in heads' with 'head off-manifold'; mitigated by
no-op/random controls and near-zero CE); single lineage, one
conversion draw.

---

## E160 — the head-set escalation: FLAT-CE-FACT-KILL — the surgical surface EXISTS, and it is TYPE-SELECTIVE (2026-09-28 ~10:50Z) — DONE

WHAT WE DID: graded head-set escalation (singles through E4 +
no-L0H3 sets N2-N4, zero and mean-replace, random-4 scatter,
install + site-stored control columns); gates bit-exact incl.
L0H3-zero reproducing e150's anchor exactly.

WHAT WE SAW (T090): FLAT-CE-FACT-KILL fires. THE KILL IS N2
{L1H0, L0H0} — WITHOUT the 'fact-specific' L0H3 — in BOTH modes
at the lowest CE: 79.4%/70.7% g0 drop @ +0.245/+0.28 (NLL3
+0.26). L0H3 is not necessary (its singleton: 58.6% @ +0.21,
zero mode). Random-4 scatter: no kills at CE +0.07-0.13 — the
kills are COORDINATE-SPECIFIC. SUPERADDITIVE: E2-mean 81.3% vs
33.6% summed singles — complementary heads (one suppressor +
route suppliers). THE DISSOCIATION (the paper figure): the same
coordinates kill the INSTALL-PHASE fact too (N2-zero 66.9% @ +0.299;
the 67.3% @ +0.32 cell is E2-mean — attribution fixed per R47) — a
SHARED readout circuit, not a consolidation scar) while the
SITE-STORED fact survives same-coordinate surgery (<= 10.6% @
+0.25; only N4 reaches 30.8% at wreck-adjacent +0.89). HEAD
SURGERY DISSOCIATES THE MEMORY TYPES. Honesty: strongly
superadditive (no kill hidden behind sub-additivity); mean-mode
flips L0H3 alone 12->59% — every claim names its mode, kills
corroborated across modes; single lineage (e157 owes the
replication); g-12 fragile under random sets too (the
pre-registered primary is g0 where the random null is clean);
CE and NLL3 agree in sign on every cell.

---

## E146 — the dissociation matrix: INSTRUMENT-INVALID — the self/other battery does not transfer to this line (2026-09-28 ~10:25Z) — DONE (null)

WHAT WE DID: the full intervention x function matrix (mask /
poison ladder / perm / head ablation x fact / self / CE) on the
B43-line consolidated net with the e111/e112 battery ported.

WHAT WE SAW (T089): the battery FAILS AT BASELINE — the foreign
donor does not collapse (gap 0.018 vs the >= 1.0 bar; every cell
reads ACCEPTS-FOREIGN; the printed ROUTED/TENANT clauses are
VOID). A weak occupancy separation survives (k*=1, E_sib 0.235 vs
E_for 0.077, clears nulls) but it is not the e111 holographic
k*=7 signature. Interpretation: INSTRUMENT TRANSFER FAILURE —
either this line lacks the binary self/other step or the donor/
null conventions mismatch; self-recognition, where the lab has
found it, is lineage-particular (consistent with family-typed
anchor physics, e108). W015's question (does the self survive
losing its pivot) is UNADJUDICABLE on this rig; e156 BLOCKED
until a lineage-native battery exists (options: run the matrix
on the e111 home lineage instead — its nets and battery exist).

---

## E151 — the P-b cell: ROUTE-OVERWRITES — the cliff is PER-NET; locked re-teaching converts the memory to site-only and closes the geometry door GLOBALLY (2026-09-28 ~10:10Z) — DONE

WHAT WE DID: one GPU re-teach (e143's locked-replay protocol,
300 steps) of the consolidated net's fact at read rows 183-189;
full before/after battery; root gated bit-exact.

WHAT WE SAW (T088): the committed TWO-DOOR prediction FAILED;
T087's fork resolves PER-NET. Before -> after: the 183 SITE
GREW (row-183 strength -0.007 -> +0.512, ratio 0.96; D-183 now
kills half: 0.998 -> 0.486) while THE ROUTE DOOR CLOSED (g-12
0.916 -> 0.102; g+12 0.948 -> 0.121; D-all 0.905 -> 0.098; the
brake vanished: A -0.132 -> +0.005; row-0 S 0.732 -> 0.106 with
base collapsed alongside, rel ~0.99). NOT WRECKAGE: CE_R IMPROVED
(1.6635 -> 1.6490); mask spares both nets (x1.00/x1.06 @ +0.04 —
e150 replicated); ladder poisons both (0.07 kills @ +0.99). THE
TRANSIENCE CLUE: the 8-step smoke retained g-12 at 0.99 — the
conversion is budget-dependent, living somewhere in 8-300 steps;
a transient two-door state may exist before the cliff re-runs.
Honesty: single seed/lineage (n=1 fork verdict); 300-step budget
matches the original consolidation but the conversion point is
unlocated; FAR-class placement confound (184-token pre-context)
entangled with locked-variance exactly as e143's FAR; site bar
weak at this readout (controls ~0) — growth carried by absolute
census + D-183 necessity.

---

## E147 — the width ladder: TEXTURE (a CLIFF, not a dose) — any variance switches the memory type; the address key dies at w=1, never gradually (2026-09-28 ~09:50Z) — DONE

WHAT WE DID: six width arms (w in {1,2,4,16,32,64}) from the
e048_repro root + free endpoints (w0=L, w8=R + a replicate, NEAR,
FAR, root); co-measured A(w) (row-129 address-key delta) and
NR(w) (d_r0 drop at g-12 + g+12); gates bit-exact; 1003s.

WHAT WE SAW (T087): INVARIANCE-CAUSAL did NOT fire as registered
(monotone clause: Spearman -0.381 vs bar <= -0.8; sensitivity
ladders all >= -0.38). DEAD-AGAIN no (A range 0.460).
SEED-COVERAGE no (NR(32)/NR(64) above onset bar — no collapse;
T079-pure fails mildly at 0.67x without W010's cliff). THE
TEXTURE IS THE FINDING: A CLIFF, NOT A DOSE — A(w): L +0.327,
w1 -0.029, then a mildly negative plateau (-0.03..-0.13, no
width trend); NR(w): L 0.071, w1 0.696, plateau 0.6-0.9. ANY
position variance (even +-1, name confined to rows 128-130)
kills the address key and births novel-geometry expression
within 300 steps — co-onset at the ladder's resolution limit,
STEP-FUNCTION form. SECONDARY: locked replay erodes NR BELOW
root (0.071 vs 0.166) while strengthening A (+0.327) — the roads
diverge in OPPOSITE directions from the first rung of variance.
FREE CELL FIRES: FAR-ROUTED-TAIL — e143_far's 0.205 g-12 tail
collapses under d_r0 (x0.069): FAR's hybrid tail was row-0-
coupled, not the band field (T084 item 3 resolved). Honesty:
single lineage; NR bounded by each arm's g-12 expression
(existence-confound; ratio forms co-reported, x0.004-0.023 for
all w>=1); NR now carries e150's POISONING semantics (sink-health
dependence, not information routing); L's 150-step budget
bracketed by sensitivity ladders; e143_jitter vs e119_r300 the
one clean replicate (A -0.1324/-0.1325).

---

## E150 — the flat-CE route test: ALL-KILLS-WRECK — 'routed' was never information flow; the kill is POISONING, and the reframe activates (2026-09-28 ~09:35Z) — DONE

WHAT WE DID: five probes, CE on every cell, mask instrument gated
bit-exact vs the standard forward; 57s CPU.

WHAT WE SAW (T086): ALL-KILLS-WRECK fires — all 4 killing cells
cost CE >= +0.70 (norm0@g0 93.2% @ +1.40; norm0@g-12 98.8% @
+1.40; norm0.07@g-12 84.2% @ +0.84; perm@col12 95.8% @ +0.705).
FLAT-CE-ROUTE does not fire (0/23 bar-eligible cells). THE
MECHANISM UNDERNEATH: the forced-off-sink mask (all attentional
access to position 0 blocked) SPARES the fact at the flattest CE
of any row-0-plane intervention ever measured (retention x1.009-
1.030 @ +0.03, all three net types incl. controls) — d_r0's kill
was never information flow from row 0. Coherent reading: a
low-norm wpe[0] leaves key 0 as a VALUE-LESS MASS ABSORBER that
corrupts downstream reads (poisoning); masking refunds the mass.
The presence threshold lands in (0.07, 0.15) — far sharper than
e141's (0.066, 0.382); pre-mask sink attention mass at the read
position is only ~0.007/layer. PRESENCE-AT-NOVEL fails by 0.8pt:
perm@g-12 costs 16-21% (x0.79-0.84) — the fact MILDLY consults
row-0's direction at novel geometry (unlike +4.1% spare at g0);
and P4's col-12 control DIED under perm (x0.042) — short-horizon
reads consult row-0's direction, the 129-read does not (read-
horizon texture). DIRECTION-CONSULTED does not fire: fact-at-
position-0 scramble IMPROVES the fact (x1.589, weak 0.23 base) —
W014 survives its control. L0H3-zero: 58.6% drop at CE +0.21 —
NEAR-MISS, 1.4pts under the kill bar, report-only; top-3-mean
joint x0.068 @ +0.71 (post-hoc rider). Honesty: mask is off-
distribution but CE +0.03 prices generic damage ~0 and the
site-stored control survives (x0.995); single-seed cells; P4
power-limited (base 0.23).

---

## E142 — row-0 at birth: ROW-0-ALWAYS — there was never an address-only phase, and the address itself was protocol-made (2026-09-28 ~09:30Z) — DONE

WHAT WE DID: row-0 content census (e131 instrument, within-net
adjudication) across 13 install checkpoints spanning the e048
dose ladder (1x-4x), direct/natural-exposure installs, e044
re-installs, the e098 fresh family (5 seeds), e117, e082 B43.

WHAT WE SAW (T085): ROW-0-ALWAYS fires at every dose in all 13
nets — no net's row-0 strength comes within two orders of its
2x-control bar (most conservative: 2.6x over). HUB-FIRST fails
(row 0 dominates the decision dial in 13/13; no dose where the
decision row takes over). THE HISTORY REWRITE: the five-day
'address -> field' arc was row-0-co-carried throughout — W011's
savor (c) PROMOTED TO LAW (consolidation = share-growth of the
largest seed). SHARPER: the ADDRESS ITSELF is protocol-
contingent — direct/natural-exposure installs (e048_direct400/
800) are almost purely row-0-carried (rel 0.947/0.983) with row
129 NULL (-0.006/+0.007): the 'address' was a property of the
masked-replay protocol, not of birth. In the fresh 0.84M family
row 0 carries the ENTIRE install (rel exactly 1.000 at every
seed). DOSE moves share, not presence (e048 ladder: row-0 share
0.981->0.952 as decision-row share grows 0.432->0.522 — the
address grows INTO an already-row-0-carried memory, never
overtaking). Honesty: per-net instruments (families/batteries
differ — all bars within-net); trained-geometry dial caveat
stands (the at-birth question is well-posed on its own dial;
routing-level replication needs novel-geometry arms); one
unresolved gate (e098_s4307 is an earlier-trajectory state, not
the stored patience twin — census internally valid, exclusion
would not change the verdict).

---

## E143 — error-placement steering: COMPASS-CAUSAL — the committed prediction holds; invariance survives its observational death by intervention (2026-09-28 ~09:00Z) — DONE

WHAT WE DID: three 300-step arms from the e048_repro root —
NEAR (fact locked at positions 5-13, zero diversity), FAR
(locked ~137-143), JITTER (e113 recipe); full battery; gates
clean; 427s.

WHAT WE SAW (T084): COMPASS-CAUSAL fires — NEAR consolidates
SITE-STORED at 5-13 (final report: NEAR FULLY address-bound — novel g-12
0.002, D-all 0.003, no routed tail; FAR site-stored at r137 +0.238,
not routed: novel 0.205, D-all 0.156; JITTER reference reproduced
0.722/0.903/0.914; instrument hardened post-first-pass — onset-only
readout had masked non-onset site rows, bars unchanged) (site strength 0.278 final-hardened) with row-0 presence
at/below install baseline (0.232 vs midpoint 0.634; PIGGY bar
1.091 — proximity piggybacking DEAD). The error-placement
compass is now CAUSAL: choose the error's site, choose the
store's site. Invariance survived its strongest attack by
INTERVENTION despite T079's observational death (T083): zero
position-variance at a sink-adjacent site did NOT create a
route. TEXTURE: FAR is a HYBRID — novel-geometry expression
0.249 (vs NEAR 0.002, JITTER 0.915), row-0 strength 0.943 at
trained geometry (sink-load per T083's saturation lesson, not
necessarily routing); FAR's site overlaps the install band
121-137, so the band population may supply de facto support
diversity — the invariance law's width-0 boundary may have a
band-overlap loophole, OR FAR's 0.249 is the old band field
(content-carried). Discriminator queued (free, minutes):
d_r0@g-12 on e143_far/jitter — FAR-ROUTED-TAIL (collapses >=70%)
vs CONTENT-TAIL (drops <=30%).

---

## E140 — route-dependence trace: GRADIENT-VOLUME fires — T079 dies on its dial; T078's 'erasure digs in' RETIRED (cycle damage); the trained-geometry dial SATURATES (2026-09-28 ~08:40Z) — DONE

WHAT WE DID: row-0 presence-strength S (e131 instrument verbatim,
install-60 g0) across seven checkpoints + the L-CYCLED arm
(3x300 locked, no reset); all gates bit-exact (twin reproduces
e116's stored seed-42 values; R@300 vs e109 GPU ref 7.1e-4).

WHAT WE SAW (T083): FIRED GRADIENT-VOLUME ONLY — ratio
R@150/L@150 = 0.745 < 1.3 (locked replay MORE row-0-dependent
than jitter at matched steps); R-ROUTE-MONOTONE failed (R@150
0.4687 DIPS below twin 0.5455 before R@300 0.7226); E-NEVER-
ROUTES failed absolutely (E@c2 off 0.228) but passed relatively
(all E rel within 0.10 of twin's 0.981). THE CEILING: twin
starts at rel 0.98 — presence-dependence at the trained geometry
measures SINK-LOAD, which every readout has; it does not
isolate routing. L-CYCLED: D-all residue thins 0.1307 ->
0.0045 -> 0.0003 (fold 0.0022) vs E's 0.1902 -> 0.0132 -> 0.0010
(fold 0.0053) — monotone, same-or-faster, NO erasure; grown rows
explode (209) like E's ~190. ANTI-MIGRATION HAS NO
ERASURE-SPECIFIC EVIDENCE LEFT — 'erasure digs in' retired.
TEXTURE GOLD (row-129 address-key strength): L@150 0.327 >
E@c3 0.280 > twin 0.241, but R@150/R@300 NEGATIVE (-0.21/-0.13)
— deleting the address FEEDS only jittered facts (the brake as
census texture): locked and erased roads STRENGTHEN the address
key; only the position-varied road negates it. Fixed D-all:
twin 0.193, E@c2/c3 0.061, R@300 0.903. Honesty: single lineage
(all arms from one twin install); the e131 dial conflates
route-keying with sink-load (e141's lesson) — the ceiling effect
is WHY gradient-volume fired while no arm separates; the
route-isolating dial is presence-dependence at NOVEL geometry
(e141's g-12 design), which this run did not measure.

---

## E139 — row-0 universality: HYBRID (site-dominant two-door) — row 0 is NOT universal; row 0 routes, the site stores (2026-09-28 ~08:15Z) — DONE

WHAT WE DID: five probes on the splice arms + consolidated
reference; arm-c regenerated (the one permitted training, 300
steps); all gates bit-exact (|d|=0.0 vs e131 tables); 1059s CPU.

WHAT WE SAW (T082): both splice arms HYBRID with the site
dominant — D-row-0 -24.8%/-27.5% (below the -50% universal bar),
D-183 -54.4%/-30.9% (below the -80% site-locked bar), both doors
together -78%/-62% (super-additive). Row-183 CONTENT-POSITIVE
(strength 0.455/0.271, ~1000x control band — the only strong
content row in the census); row-0 content test NULL on both arms.
GENERALIZATION (e131's honesty note d closed): both arms
generalize — novel train contexts 0.597/0.708, val-split
0.591/0.699, held-out fact segments 0.620/0.683 — real facts,
not window memorization; training-geometry premium ~0.3 (e131's
0.989 overstated strength; the training read AND generalization
were both true). BRAKE ABSENT on splice arms (row-129 replacement
~0) vs the consolidated line's -0.11/-0.13 — the brake is a
re-keying scar of the jitter road and does not reach across
homes (corroborates T078). CONSOLIDATED REFERENCE (report-only):
the jitter-road net reads p(Z)=0.660 at the 183-geometry it
NEVER trained, row-0-keyed there (drops +0.657/+0.551, ratio
0.84) — row 0 is a position-invariant readout route for the fact
that re-keyed to it. ARM-C RIDER: RIDER-NULL — genuine decay,
not instrument blindness: the dreams' 34 ZEPHYRAs sit at
x-col 130 (33/34; read position 129 = the OLD address — dreams
are never position-diverse); p(Z)@onset 0.230 BELOW the base's
0.391 at the same positions (dream replay actively ERODED the
fact there); consolidated nothing anywhere readable. T076's
error-compass survives its last open edge; e120's arm-c verdict
stands un-revised. Honesty: exposure spans excluded exactly;
D-row-0's CE +1.28 bounded by row-1 control (negligible
scaffold); ~25% drop is row-0-specific but NOT content (the
content test says not) — processing gate, not store.

---

## E141 — sink-key mechanism battery: ROLE-ROUTED (presence-only) — "re-keyed to row 0" formally dead; the noun is row-0 sink-routed (2026-09-28 ~08:05Z) — DONE

WHAT WE DID: five probes (install-restore t-surgery, presence-
vs-content scramble, rows-2-7 + norm-matched scaffold hardening,
d_r0 at novel geometry g-12 on both R nets, gate-vs-source
interpolation) on gated nets; 87s CPU; all artifact gates
bit-exact.

WHAT WE SAW (T081): ROLE-ROUTED, 3 corroborating votes to 0.
(1) INSTALL-RESTORE: removing the ENTIRE consolidation delta
from wpe[0] costs nothing (t=1 x0.999, CE +0.0002); direction-
scramble (norm-preserving permutation) RAISES fact expression to
0.817 while costing +0.70 CE; only removal-class kills (zero
0.053; mean-replace 0.001 — mean-row norm 0.066 = near-removal);
half-norm survives, double-norm slightly helps. (2) PRESENCE-KEY:
front-window scramble HELPS (0.858 vs mid-control 0.804, base
0.785) — scrambling the sink-region content improves the fact
read. (3) SINK-UNIQUENESS hardened: rows 2-7 + norm-matched
random all cheap (max CE +0.017); only row 0 wrecks (+1.40/+2.00).
(4) NOVEL-GEOMETRY COLLAPSE: d_r0 at g-12 x0.014 on R@150 AND
R@300 — W011's missing cell filled: the generalization itself
routes through row 0's presence (content-keyed alternative dies).
(5) NO-COLLAPSE curve: neither SOURCE-GRADED nor GATE-THRESHOLD
can fire (nothing is lost on the path; the flat-curve R^2 trap
documented before adjudication). MECHANISM: row 0 is the net's
attention-sink pivot (2nd-largest row norm); the consolidated
fact's READOUT WEIGHTS route through row 0 EXISTING — presence,
not content, not direction. SAVOR: the corpus-CE dissociation
(direction-scramble: fact intact, CE +0.70) — the fact's route
is more presence-robust than the net's general LM function.
Honesty: single-seed line for probes 1-3 (probe 4 adds two
independently-trained R nets, agreeing); t-curve probes one path
but perm + norm riders make the surviving manifold >=2-parameter
wide; scramble leakage bounded by matched control + the sign of
the effect.

---

## E133 — field anatomy census: TEXTURE — the fact is BODY-stored everywhere; what differs is the READ ROUTE (2026-09-28 ~07:45Z) — DONE

WHAT WE DID: per-organ causal ablation sweep (MLPs by layer,
attention heads, wpe row sets; mean-replace mode picked by
pre-registered CE_R rule over zero by 0.022 nats; both recorded)
on three gated nets: graduated (e131_consolidated, bit-exact
0.7850374), address-phase twin (e119 twin start, 0.5563082),
site-locked (e131 arm_b, site onset bit-exact 0.9880021).

WHAT WE SAW (T080): verdict TEXTURE — SUBSTRATE-IN-BODY half-
fired (graduated body share 0.689 >= 0.60 TRUE; twin address
mirror 0.388 < 0.60 FALSE — general organs dominate the marginal
denominator even pre-consolidation); ROUTE-ONLY dead (body 0.810
>> 0.30); W005-ALT (address-spreading) KILLED (band-adjacent
heads 0.120 < 0.50). Instrument cross-validated: wpe_r0 drops
0.732/0.546 reproduce e131/e116 stored strengths exactly. TEXTURE
FINDINGS: (1) ALL THREE NETS store the fact mostly in the body —
even the 'site-locked' 183 net keeps only 7.9% at row 183 (MLP
64.4%); content substrate is shared, READ ROUTES differ (W013).
(2) LOCALITY FILTER (CE damage <= 0.75): graduated fact-specific
residue is HEAD-dominated (84.5% heads / 15.5% MLP / wpe ~0) with
L0H3 a genuinely specific body head (0.46 drop at 0.21 CE); the
unfiltered registered map is dominated by load-bearing-for-
everything machinery (mlp_l0/l5, 2.4-4.1 nats CE). (3) ADDITIVITY
FAILS: joint top-MLP+top-head+key-row drops 0.785 vs parts-sum
1.98 — ~2.5x redundancy; W009's population frame confirmed at
the ORGAN level; all shares are marginals, not a partition.
(4) The TWIN runs a genuine positional-address apparatus — L3H4
reads the band with 91% attention mass (0.319 drop at 0.008 CE
damage) — and the graduated net has DISMANTLED it (address-head
share 0.12; band5 deletion now RAISES p(Z) 0.120 — brake
replicates; twin band5 drop +0.364 at ~0 CE = fact-specific
address read). (5) NO head is sink-adjacent >= 0.25 on the
scored position: row 0's causal load flows through SMALL-
attention VALUE channels — the sink route is not a high-sink-
attention route at this readout. Honesty: ablation-mode near-tie;
additivity assumption stated as failed; expression-drop
conflates fact load with general wreckage (both views recorded);
address-head label coarse (26-27/36 clear bar; sensitivity table
shows the two kills unchanged).

---

## E119 — migration head-to-head: AMBIGUOUS (1 of 2), leaning DIFFERENT-STORES — jitter re-keys, erasure TIGHTENS the address (2026-09-28 ~07:20Z) — DONE

WHAT WE DID: twin installs, same fact, matched final expression
(R@150 0.5597 vs E@c1 0.5606, gap 0.0009; both dials the
pre-registered freedom); full comparative battery; locked-replay
free-rider arm; 7 phase nets saved runs/checkpoints/e119_*.pt.

WHAT WE SAW (T078): verdict AMBIGUOUS as registered — one clean
dissociation ((c) brake: R +0.210 [+0.153,+0.279] FEEDS vs E -0.267
[-0.297,-0.240] SUPPRESSES, both CI-separated; L brakes -0.509
like E), near-misses on the same side (D-all R 0.769 vs E 0.190 —
E sits 0.01 under the 0.20 bar; held-30-under-D-all 0.663 vs
0.071 at MATCHED g+0 (R44 audit correction — the first fold paired R's
cross-geometry max 0.709@g-8 against E's g+0; R@g-8 vs E@g-8 is
0.709 vs 0.021); novel geometry g-12 R 0.813 vs E 0.092; share E off-grid).
Census: R grows 10 decision-band rows, E grows 2 (+ generic
high-row drift 220-254 — overlapping e131's row-249 census find).
PRE-REGISTRATIONS ALL FIRED (registered ~06:48Z before the
battery): P1 R-beats-E on deletion survival (3/3 geometries);
P2 E-scatters/R-concentrates (2 vs 10 band rows); P3 store-thins-
while-expression-recovers (field-only residue 0.190->0.013->0.001
monotone across cycles while cycle-ENDs go 0.347->0.425) — MORE
ERASE CYCLES MADE THE FACT MORE ADDRESS-BOUND, NOT LESS, the
opposite of T073's migration reading. e083's protocol transferred
cleanly to the e048_repro line (no vacuous erases). Honesty: E@c1
= one 300-step relearn vs R@150 mixed-position steps (matched
expression, different training mass); L's brake shows the R/E
brake difference is confounded with position-diversity, not
erasure per se; E's dall cell is threshold-fragile (0.01 under
bar); census growth-rule fires on generic anchor drift (the
band-restricted view is informative); V-typing mild (~0.30-0.36),
report-only.

---
## E131 — the re-keying census: RE-KEYED — the key is ROW 0, and the e120 splice arms never failed (2026-09-28 ~07:05Z) — DONE

WHAT WE DID: R43 critic's discriminator, as dispatched. Nets
regenerated bit-exactly from gated roots (G_E120/G_E113 max diff
0.0 vs stored tables; e082_b43_install + e048_repro loaded, e044
scar line inspected and rejected); CPU-only, 1538s; phase nets
saved runs/checkpoints/e131_*.pt. Four probes.

WHAT WE SAW (T077): all three RE-KEYED conditions fire.
(1) ROW-0 CONTENT TEST: post-consolidation strength 0.732 — ABOVE
its install-phase 0.545, 380x the max control row (0.0019); the
only row ABOVE the control band (380x max control; two control-level
rows 118/119 flag content:true at 0.0019 — R44 audit precision);
the whole 121-137 band
including 129 sits at ~0, and row-129 replacement RAISES p(Z)
(the e115 brake, now with the new home identified). (2) BAND-
MINUS-ROW-0: D-all reproduces e113 exactly (0.906); +row-0
collapses expression to 0.024 (-97%); scaffold-matched D-all+ROW-1
survives untouched (0.908). (3) CENSUS (weakest): worst OOB row
249 merely matches row 129's own delta — conditions 1+2 carry the
verdict. PROBE 1 (183-geometry read): both splice arms express
the fact at address 183 — p(Z) 0.989 (self) / 0.988 (corpus),
frac p>=0.5 = 1.000 over 840 name-char reads — vs band 0.015-0.035
and base pre-ft 0.061/0.066. E120's SIGNAL-IN-CONTEXTS
INSUFFICIENT was INSTRUMENT BLINDNESS: the arms consolidated
exactly where their training error lived, at an address no battery
read. T076's error-location wins outright. Honesty bounds: row-0
collapse has a scaffold component (CE_R +1.40) but two independent
controls bound it (mean-replacement arm kills identically; row-1
control null); necessity != storage (row 0 may be the ROUTE, with
content in body weights — e133's question); probe 1 is a
training-geometry read (learning-at-183 proven, novel-context
generalization at 183 untested -> e139).

---

## E120 — fact in contexts: SIGNAL-IN-CONTEXTS INSUFFICIENT — position diversity is the operative ingredient (2026-09-28) — DONE

WHAT WE DID: the signal x context decomposition — fact-spliced
into self-dream contexts vs corpus contexts vs verbatim dreams vs
jitter replay, matched budgets, deletion batteries; e121's arm (c)
bit-reproduced (max diff 0.0 over 30 cells).

WHAT WE SAW (T075): bar 3 — neither splice arm reaches 0.20
anywhere (0.023/0.098); jitter clears 5/5 THROUGH deletion
(0.694/0.720 — e113's BODY-STORED replicates on the B43 line).
DIRECTION REVERSED vs W004: corpus-contexts BEAT self-contexts by
CI separation; self-spliced = verbatim dreams; arm (a) carried
MORE name signal and still lost. All three full-mask arms fell
BELOW the base's own floor — the fine-tune erases more field
expression than the signal repairs. wpe row 183 grew (+0.16) but
stayed address-locked (battery never reads 183).

---

## E121 — the dreams probe: NO DREAM CONSOLIDATION — verbatim dreams slow forgetting 500x but never graduate (2026-09-28) — DONE

WHAT WE DID: matched 300-step fine-tunes on own-dreams / twin-
dreams / no-exposure text; D129 + D-all-addresses deletion
batteries; G_INST bit-matches e082's reference; all deviations
pre-registered.

WHAT WE SAW (T074): bar 4 fires — own-dreams collapse post-
deletion (max 0.045 <= 0.05). TEXTURE 1: the dreams CARRIED the
fact (34 ZEPHYRA in 15,120 own-dream chars — this line dreams
the name, unlike e048's silent seed-42) and still did not
consolidate it. TEXTURE 2: dream replay slows forgetting ~500x
(0.04-0.07 vs corpus-only 0.0001) — name tokens in self-generated
contexts resist decay — but never approach the 0.20 bar and die
under address deletion. Own-vs-twin: point estimates favor own
but CIs overlap and the trend tracks name-rate, not
self-specificity. WPE 121-137 band inflated in ALL arms (generic
fine-tune effect).

---

## E083 — canalization cycle-3: MIXED with oscillation — the groove persists but does NOT deepen; the erase weakens (2026-09-28) — DONE

WHAT WE DID: three full erase->re-learn cycles on the B43 install
line (e044b machinery verbatim; instruments bit-exact); two
protocol findings fixed en route (blowup gate re-based; battery
chaining reverted) — all documented.

WHAT WE SAW (T073): steps 20 -> 10 -> 9 (c3/c2 = 0.90, between
bars); cos 0.599/0.525/0.518 (between bars); OSCILLATION FLAG
FIRES as registered (both successors faster). THE MECHANISM: the
re-learn does NOT accelerate — the ERASE damages less each cycle
(post-erase NLL 2.724 -> 1.441 -> 1.369): the completion
migrates off the erased rows onto position-keyed machinery. The
groove PERSISTS (regrows along the original direction, cos ~0.52-
0.60 through three erasures) but does not DEEPEN. Strike cost
constant +0.06-0.08 nats (no trend, no sign flip).

---

## E118 — standardization control: SURVIVES — rogue-dimension confound excluded (2026-09-28) — DONE

WHAT WE DID: per-dim z-scoring over pooled own+donor V-sets (both
readings: common-frame registered + Timkey strictest frame); the
e111 subspace instrument standardized; rogue-dim census.

WHAT WE SAW (T072): sibling/foreign 2.05x post-z (CIs disjoint);
energy 2.80x — the standardized separation fires at EVERY k=1..8
(6.09x at k=1; cleaner than raw e111's k=7). No rogue dim (max
share 0.095; log-variance profiles own~sibling 0.987 vs
own~foreign 0.028 — the mild anisotropy is FAMILY-SPECIFIC, not
shared confound). Caveats: Timkey-strictest 1.68x (clear, below
2x); a third of raw self-cos was scale-structure (0.403→0.315).

---

## E115 — graded ablation: COORDINATE-LOCAL STORY SAFE — the address is content when the field is weak (2026-09-28) — DONE

WHAT WE DID: address-intact vs no-address across graded field
scales (r 0.1-1.0); all gates pass (G_CONT matches e114's cells;
G_SCALE exact; base rebuild dev +0.009).

WHAT WE SAW (T071): the brake SIGN FLIPS — full field: +0.132
(address suppresses, CI-separated); r=0.56: −0.0051 (deleting the
address LOWERS expression, CI-separated — the address helps when
the field is weak); floor at r<=0.25. M3 gets 0/3 under every
reading. The fourth story (coordinate-local modulation) stands
SAFE: the address is content when the field is weak, suppressor
only when the field is strong.

---

## E117 — maturity discriminator: BETWEEN — the constant is a DISTRIBUTION, not a maturity point (2026-09-28) — DONE

WHAT WE DID: seed 4309 at exactly e053c's 3133-step recipe (base
val-CE 1.5378 — between the siblings' 1.63-1.65 and e053c's
1.5227); install + the e110 mini-grid, 12/12 identity gates.

WHAT WE SAW (T070): k=128 product 77.4 — ABOVE the maturity band
[40.5, 67.5]; k=64 off-grid-high. Neither bar fires. Plotting the
four nets: steps 2000→31/42, 3133→54 (e053c) AND 77 (s4309) —
the constant is non-monotone in BOTH steps and val-CE. W007's
clean maturity prediction (matched exposure ⇒ ~54) FAILS.

---

## E116 — re-barred census: STRUCTURE-STRONG, CONCENTRATION-MIXED — graduation denied, stays descriptive (2026-09-28) — DONE

WHAT WE DID: the re-barred statistic (content-tested rows, top
content >=2x next) across all 6 seeds from stored censuses.

WHAT WE SAW (T069): 3/6 concentrate — both 2.7M seeds (2.31x,
3.31x) + 4305 (2.22x); 4306/4307/4308 sit at 1.15/1.44/1.01.
The wrinkle the re-bar exposed: ROW 0 passes the content test in
every seed (it is BOTH scaffold and content — e071's window-key
duality), so the measured ratio is row0/decision-row; in the
0.84M family the install's mass genuinely SPLITS between the
window-start row and the decision-window row (4308: 0.384 vs
0.381). Address concentration is architecture-family-dependent;
the one-decision-window-content-row structure holds 6/6.

---

## E098 — seed-ladder dual mandate: structure 4/4, statistic re-barred; share-form replicates, value maturity-dependent (2026-09-28) — DONE

WHAT WE DID: 4 fresh-seed base nets (2000-step recipe) + installs +
censuses + the e110 mini-grid on two; all gates pass; M2
instrument identity 12/12 exact.

WHAT WE SAW (T068): M1 — literal bar 0/4 but STRUCTURE replicates
4/4: every seed grew a single content-carrying decision-window
row (127-129, mean-arm = zero-arm, magnitudes 0.24-0.33 = seed-
42's); the registered top1/top2 statistic was calibrated on the
2.7M family where row 1 is negligible — in the 0.84M family the
0-1 scaffolding band tops every census. M2 — SEED-DEPENDENT value
(products 29.9-46.0, mean 36.8, −32% vs 54) BUT the law's FORM
holds within each net (r*k max/min 1.10/1.20, under the 2.0
constancy bar); post-hoc base diagnostic shows the departure is a
net property, not install; and the ladder nets were under-trained
vs e053c (2000 vs 3133 steps — a tasking deviation) — the
constant appears to GROW WITH MATURITY at fixed architecture.

---

## E114 — brake signatures: all three mechanisms NULL — the brake is coordinate-local, a fourth story (2026-09-28) — DONE

WHAT WE DID: the three W006 signatures on the bit-exact rebuilt
consolidated net (brake replicates +0.132 [0.065,0.207]; 67%
positive); manual-vs-net attention dev 2e-6.

WHAT WE SAW (T067): S1 routing-mass r=-0.104 NULL (held-30
anti-signed); S2 field-strength r=+103 NULL (addr-mass/field-mass
are exact complements r=-1.000 — one discriminator, zero on the
primary); S3 no suppression-without-field (ADDR_ONLY 2.1e-5 =
NEITHER 1.9e-5). FOURTH-STORY raw material: the brake is
install-specific (held-30 shows ANTI-brake -0.079), geometry-
specific (+4/+8 geometries ~0), and lives where the address
coordinate IS the decision row's own residual state at its
trained geometry.

---

## E110 — per-field floor: THE SHARE LAW — r*(k)*k ~= 54, one constant (2026-09-28) — DONE

WHAT WE DID: retention r x field-size k crossed grid (4x4, e102
rig, G5 cell replicates e102 bitwise); continuous 0.3-crossing
thresholds with bootstrap CIs.

WHAT WE SAW (T066): SHARE-LAW SHIFT fires (r*(154) < r*(64),
P=0.999; r*(32) off-grid-high, P=0.998); flat per-entry null
fails (P=0.002). THE CONSTANT: r*(k)*k = 53.3 / 55.5 / 53.2 —
max/min 1.04 with CI overlap. Total retained field mass at the
collapse boundary is ~54 entries-equivalent across a 2.4x range of
k. W001's "same floor" is now a number. e102's floor refined to
continuous ~0.35. Honest riders: k=32 partly removal-limited; the
tasking's Bar-3 literal inequality was inverted vs its own lead
sentence (dispatcher slip — the agent registered the slip pre-
compute and adjudicated the W001-direction reading, which is what
fires).

---

## E113 — all-addresses deletion: BODY-STORED — the fact left the address system (2026-09-28) — DONE

WHAT WE DID: rebuilt e109's consolidated net bit-exactly (post-none
table matches to 0.009; grown addresses replicate to 3 decimals);
three deletion arms with confinement verified per row.

WHAT WE SAW (T065): (i) D-all-addresses {121,125,129,133,137}
survives 5/5 geometries (0.889-0.930, argmax-Z 0.92-0.97; held-30
0.64-0.66) — zeroing EVERY address the fact grew costs only
~0.03-0.10. D0129's collapse (e109) was window-scaffold loss, NOT
address loss — the T064 correction confirmed. Texture: deleting
addresses RAISES p(Z) at the original geometry (0.785→0.905): the
address row is mildly suppressive at its own coordinate.

---

## E109 — consolidation: CONFIRMED — jittered replay makes the fact survive row-deletion (2026-09-28) — DONE

WHAT WE DID: the W003 head-to-head — jittered replay (offsets
-8..+8, 300 steps) vs position-locked replay (matched budget) vs
baseline; then row-deletion (d129 registered; d0+129 secondary);
battery at all geometries. All gates pass (battery reproduces
0.5563 pre-training).

WHAT WE SAW (T064): the registered bars MISFIRED on calibration —
the collapse bar (c<=0.05) was set against the dead strong-form
one-row law; T043's partial necessity (~0.32) means c lands at
0.215 and the bar never fires (honest flag). THE DATA ARE
DECISIVE ANYWAY: post-d129 at original geometry (through the
deleted coordinate) — jittered 0.909 vs locked 0.436 vs baseline
0.215; at best trained geometry — 0.992 vs 0.436. Position
DIVERSITY doubles matched training-mass. Deleting row 0 AND 129
kills all arms (secondary: one-row-scarcity stands). The agent's
delta-branch read TRAINING_MASS_SIGNAL; the geom0 comparison says
diversity dominates — recorded both, interpretation below.

---

## E112 — signature forgery: NOT FORGEABLE — self is deeper than its 7-dim stamp (2026-09-28) — DONE

WHAT WE DID: forged keys (corpus/foreign V projected into the
recipient's top-7 stamp, norms kept) + the inverse (sibling V with
the stamp projected OFF); all 6 gates pass; every control
replicates (sibling -0.078 healthy; corpus +1.68 collapsed;
e111's basis bit-identical, k*=7 re-confirmed).

WHAT WE SAW (T062): FORGEABLE fails — both forged keys collapse
(+1.56, +3.09). SIGNATURE-NECESSARY fires — removing the stamp
from genuine self-content kills it (+1.51). THE CLEANEST
DISSOCIATION: forged-full sits at event-time V-cos 0.353 — inside
the near-sibling zone — yet collapses: the anchor reads neither
the pooled subspace nor the pooled V-cos. And corpus V already
carries 0.753 stamp energy (the stamp is largely the shared common
mode): residence never discriminated. Forging on foreign clay made
it WORSE (+3.09 vs +2.41 verbatim). Caveat: stamp-destruction
jointly removes 81% of sibling energy — necessity shown jointly
with energy; sufficiency cleanly refuted.

---

## E111 — V-manifold self-signature: k* = 7 — selfhood is a seven-dimension readout (2026-09-28) — DONE

WHAT WE DID: PCA of the recipient's own anchor-band V-manifold;
donor V-sets projected onto top-k; energy-fraction curves + null
(97 same-norm Gaussians); all determinism pins respected.

WHAT WE SAW (T061): LOW-DIM SELF-SIGNATURE fires — the 2x
sibling/foreign energy separation holds at k=7 with the null below
both (foreign sits at the 98th percentile of the null). Top-1 PC
alone: 11.7x ratio. The fixed-point stamp (W004) is essentially
SEVEN DIMENSIONS — arguably one dominant direction plus six
correctors — in a 32-d per-head space.

---

## E108 — distance-ladder anchor: SHARP FAMILY — the anchor runs binary self-recognition (2026-09-28) — DONE

WHAT WE DID: the W002-earned distance ladder (sibling / intended-
middle / copy-net), x-axis MEASURED (stream-cos per rung); all 10
gates pass; both controls replicate bit-identically.

WHAT WE SAW (T060): the intended ~0.53 middle never existed —
W002's 0.53 was e029's ΔW ladder, not stream-cos; every
differently-trained net sits at the stream-cos floor (0.016-0.033).
SHARP FAMILY: middle collapsed like foreign (+4.40 vs +5.09;
sibling +0.027 healthy). V-cos is a TWO-CLUSTER STEP: 0.403 (self)
vs ~0.14 (any other). THE SHARPEST DECOMPOSITION: the middle
donor's OUTPUT is nearly sibling-like (JS 0.041 vs foreign 0.138)
YET IT COLLAPSES — output-similarity does not save you; the anchor
reads INTERNAL V-geometry, not behavior.

---

## E107 — value-side probe: ROUTING-ONLY STANDS — and content is ORTHOGONAL to readout (2026-09-27) — DONE

WHAT WE DID: V-content match (entry's c_proj V-write vs the
decision's top-2 readout directions via ln_f) + age-weighted
variants, per stratum, vs the attention baseline; all identity
gates bit-exact.

WHAT WE SAW (T058 close): every content variant at CHANCE in every
stratum (0.44-0.58); paired content-minus-attention at failing
−0.33. ROUTING-ONLY decisive. TEXTURE: age ALONE matches full
attention at the failing stratum (0.779 vs 0.788, diff CI
[−0.03,+0.02]) — the read's predictability = attention mass + age
prior, nothing value-side. Content |cos| tiny (0.033): entry
V-writes near-ORTHOGONAL to readout directions — third independent
echo of T038 (relay not answer-shaped) and T057 (V-family
geometry).

---

## E102 — direction-vs-magnitude: DIRECTION CARRIES THE ANCHOR — the specification finds its carrier (2026-09-27) — DONE

WHAT WE DID: the decomposition arms at g=250 (unit-norm originals /
exact-norm random directions / 0.1x originals / eps=0.5 additive
control — bitwise replication of e096); all gates pass incl. the
new G5 rig-vs-shipped-artifact check.

WHAT WE SAW (T051 completion): (a) direction-only HEALTHY (+0.09,
CI includes 0 — unit-norm originals anchor fine at realized 0.559
retention) while (b) magnitude-only COLLAPSES (+1.80, cj-gap
+1.70). The run reads WHERE entries point, not how loud. Rider:
arm (c) true-0.1x IS damaged (+1.09) — direction necessary,
magnitude floor between 10% and ~56%: not entirely redundant, just
deeply secondary. e096's corruption-robustness = norm-redundancy
(additive noise keeps cos 0.89). The "pure statistical mass" model
falsified: mass without geometry anchors nothing.

---

## E101 + E106 — adversarial subsets: no kill but recency-selection is strong, readership-selection fails; second channel unconfirmed (2026-09-27) — DONE

E101 (adversarial): MIXED/TEXTURE — no CI-backed kill of the
mass-action law, but the law-stands clauses fail too: greedy-recency
at k=64 costs 7.27x random-at-same-k (arm 0.684 vs 0.094) and
k=128-recency (+1.585) exceeds the random k=128 band — taking the
newest entries buys roughly double mass-impact. TOP-READERSHIP (the
eviction literature's importance heuristic) is the WORST selector
(0.40x) — the field's own tool fails at what it is designed for.
Greedy oracle k=16: +0.066 (near-nothing).

E106 (second channel): MIXED — two-channel bar fails (early AUC at
the failing stratum is 0.779, not < 0.6; L3H0 best late feature,
CIs not separated); single-channel also fails (late deviates from
early in 5/6 strata). The residual stays open, honestly.

---

## E105 — cross-family anchor: FAMILY = THE TRAINED WEIGHTS — the anchor's specification completes (2026-09-27) — DONE

WHAT WE DID: the copy-net's own competent generated text as anchor
content vs sibling-run vs corpus controls; all gates pass (five
legacy arms bit-identical to e099; both controls replicate exactly).

WHAT WE SAW (T055 close): cross-family text COLLAPSES (+5.094
[4.62, 5.70]; terminal KL joins the collapsed cluster; sibling
+0.0275 healthy; corpus +5.33 collapsed). The donor was competent
(tail CE 0.86-0.89) — collapse tracks FAMILY not quality. Riders:
the collapsed tail drifts toward the donor's alphabet (uppercase
nonce letters; its own basin); MECHANISTIC TRACE — donor V vectors
near-orthogonal to the recipient's own (mean |cos| 0.138 vs 0.403
for sibling entries) and ~1.3x larger-normed: the family signal
lives in V-vector GEOMETRY.

WHAT'S NEXT: the anchor's full specification: MASS (nonzero,
threshold law; e089/e096) + FAMILY (trained-weights-typed geometry;
e099/e105) + RECENCY (2-3x modulation; e097).

---

## E092 + E104 — the gate is DISTRIBUTED; the read residual is REAL and localized (2026-09-27) — DONE

E092 (gate census): H-GATE-DISTRIBUTED — no single component passes
necessity+sufficiency while the full-swap brackets pass (instrument
valid). mlp-L5 is necessity-only (restoring it alone does not
reopen the gate). The RMU sealing is distributed across the retrain
— the gate is the net, not a bottleneck.

E104 (failing-DP census): LOCALIZED RESIDUE — 3 strata fire >= 3x
base (near-tie margins Q0 3.14x; MIXED opened-age profile 4.29x;
NO_YOUNG 9.8x; permutation p=0.0001). Threshold-blur REFUTED
(failing labels stable; median blur-frac 0.494 vs 0; only 2/10
recover on label-dropping). The ~10% read-prediction failures are
GENUINE misalignment — a second channel at near-tie decisions with
mixed-age opens.

---

## E099 — attractor identity: ONE broad off-manifold attractor; the anchor is NET-FAMILY-specific, not run-specific (2026-09-27) — DONE

WHAT WE DID: 5 arms (none/vzero/noise/promptcopy/randomize — the new
arm swaps in ANOTHER RUN's generated entries); terminal SKL grid +
within-arm nulls; collapse gate via clean-judge. All gates pass
(e080 drift 1.8e-6; token-identical prefixes).

WHAT WE SAW (T048 correction): MIXED-universal — the universal KL
clause PASSES with room (collapsed cross-arm SKL 0.10-0.31 vs
within-arm floor 0.43: collapsed arms are MORE similar to each
other than sequences within one arm; top-5 terminal tokens shared
4/5 with near-tied fifths) — ONE broad off-manifold attractor. And
it is NOT degenerate: entropy 3.51-3.61 > corpus 3.31, ~53/65
distinct tokens — fluent-but-wrong, matching self-score-fine/
clean-judge-6-nats. THE RIDER THAT CORRECTS T048: randomize (other
run's GENERATED entries) does NOT collapse (+0.027) while
promptcopy (corpus text) collapses (+5.33) — the anchor tracks
GENERATED-TEXT STATISTICS at the net-family level, not run identity.

WHAT'S NEXT: T048's final statement is revised: the run needs
entries that look like what THIS NET would generate — its own, or a
sibling's. The anchor is stylistic selfhood, family-typed.

---

## E100 — read prediction: THE READ IS ATTENTION-ADDRESSED — T037's unified question answers to the routing (2026-09-27) — DONE

WHAT WE DID: query-side features (attention mass, q-k cosine over
the 80 candidates) vs ground-truth opened sets (bit-identical to
e084 — G5 matched 100/100 per-DP opened-set sizes); pooled pair-
weighted AUCs, 1000-DP bootstrap.

WHAT WE SAW (T054): ATTENTION-ADDRESSED fires — attention-mass AUC
0.907 [0.886, 0.931]; q-k cosine 0.882 (same band). The opened
coordinates ARE the attended ones: precision@1 0.724 (10x chance);
46% of opened in the attention top-5 (chance 6%). Layer structure:
L1/L2 mass alone 0.897 (best single head L1H1 0.898); L0 near chance
(0.64) — the addressing signal is COMPLETE BY MID-DEPTH. Rank-sum
combo adds nothing: address = routing.

WHAT'S NEXT: the residual is the registered next object — the ~10%
AUC gap and the few per-DP failures (min 0.405): where and why does
attention mispredict the read?

---

## E097 — stratification: POSITION ASYMMETRY, but RECENCY-weighted not sink-weighted — the law gets its gradient (2026-09-27) — DONE

WHAT WE DID: position-stratified removal at k=128 (primary) and k=64
(secondary), thirds of the anchor band vs fresh uniform controls; all
gates pass; uniform replication consistent with e089 (+0.580 vs
+0.683).

WHAT WE SAW (T051 amendment): monotone-in-recency ordering at k=128 —
new third 2.84x, mid 2.02x, old 1.76x uniform; at k=64, old/mid cost
LESS than uniform (0.54x/0.62x) with only the new third elevated
(2.23x). The registered sink hypothesis INVERTED: the KV-eviction
sink prior points the wrong way — recent self-generated entries carry
the disproportionate mass. Exposure confound neutralized (the k=64
new arm has 3x LESS exposure yet 2.2x the cost — the gradient runs
AGAINST exposure). Rider: contiguous-third concentration itself costs
more than interleaved removal (local redundancy pools).

WHAT'S NEXT: the law refines to "mass dominates, recency modulates
(2-3x), sink-side cheapest"; registered follow-up (e103): contiguous-
interleaved decomposition — local-pool vs pure-recency.

---

## E096 — coherence-gap ladder: hypothesis killed at premise — the anchor is REMOVAL-fragile, CORRUPTION-robust (2026-09-27) — DONE

WHAT WE DID: single-shot additive corruption of all 154 extant
anchor-band V-entries at g=250, eps in {0.05..1.0} x median-norm,
matched-stream, clean-judged; all 4 gates pass (eps=0 bitwise;
band deltas exactly eps x norm).

WHAT WE SAW (T051 amendment): NO-DAMAGE at every dose — eps=0.05
fully absorbed (0/8 rows diverge, 197 identical steps); mid doses
slightly FACILITATIVE (3/5 rungs' CIs exclude 0 on the negative
side); the off-manifold clean-judge gap appears at NO dose (vs
e080-vzero calibration +5.02). The registered U-shape and kill
both fail vacuously; an honesty guard (added after pass 1 exposed
the case) prevented the false "coherent-basin re-entry" story.
Incidental: stream divergence is dose-LEAKY (2/8 rows never diverge
even at eps=0.5).

WHAT'S NEXT: the mass-action law sharpens — the anchor's mass is
NONZERO CONTENT-BEARING ENTRIES, not their exact values (additive
noise leaves them functional; deletion or direction-replacement —
e080's norm-matched swap — kills). Canalization-continuity killed.

---

## E095 — T050 texture probes: BOTH flags close as noise — the read-kernel card stands final (2026-09-27) — DONE

WHAT WE DID: bit-exact census re-run (e084 machinery verbatim, all
replication gates pass incl. 48/48 donor hits) + the two registered
probes: escape-destination concentration and donor-voice divergence
tracking.

WHAT WE SAW (T050 final): H-tail NOISE — top escape destination 't'
at 1.71x marginal (bar 3x); the Monte-Carlo guard shows iid draws
from the battery marginal concentrate to 15.9% max-share, so the
observed 11.3% is BELOW the null's own best case (p=0.997) — escape
destinations are just the battery's ordinary frequencies. H-donor-
voice NOISE — the 48 hits sit at the 33rd divergence percentile
(bar 70), slightly BELOW median on all three measures (opposite
direction, too small and post-hoc to promote).

WHAT'S NEXT: T050 stands FINAL — kernel=shadow with the
tail-misreport flag as a permanent caveat; no tail model.

---

## E082 — cross-seed transplant: BASIS-PRIVATE — the address row dies at the seed boundary; crossmatch instrument validated (2026-09-26) — DONE

WHAT WE DID: e043 install verbatim on B43 (GATE-0 pass, p(Z) 0.320,
protocol-identical to the prior donor artifact), seed-43 census
(GATE-1 pass — bimodal 0/129 replicates: row 0 0.318, row 129
0.096), then the 6-arm transplant battery with crossmatch overlay.

WHAT WE SAW (T043 amendment): A2 donor-overwrite 0.198 < 0.30 =
BASIS-PRIVATE — A2 ≈ own-destroyed (0.195) ≈ donor-mean (0.191);
A2−A5 = +0.007 (the donor row carries no donor-specific
information; it acts as any foreign row). T043's n=2 portability
was within-family luck. SECONDARY: crossmatch cos(d2,d3) 0.059 <<
rule 0.4459 → REJECT — and the graft indeed failed: the e062
instrument correctly screens even a one-row organ (row-cos
donor129~host129 +0.059: the two address codes are basis-
orthogonal). T043 free rider resolved: (b) sub-argmax residue
(plateau-with-rank-2, Z the runner-up). Honest: knife-edge soft at
seed 43 (39% drop; lower base 0.32); batch-64 kept for protocol
identity (documented deviation from the brief).

WHAT'S NEXT: P1 phase-2 closes — the address is real, universal in
structure (bimodal census at both seeds), but seed-private in code.

---

## E065 — RMU-vs-surgery head-to-head: classic inversion FAILED; RMU closes the rescue channel specifically (2026-09-26) — DONE

WHAT WE DID: the critic-hardened 5-arm design (RMU grid + surgery D2
+ ascent + retain-only + no-removal; R1 transplant sweeps with e055
bars; R2 probe-only per T015; R3 relearn; R4 collateral; R5 the
RMU-minus-retain contrast). Full run after 6 smoke passes.

WHAT WE SAW (T052): the registered H-obfuscation inversion FAILED —
no arm is rescuable-but-reverting. Actual: ALL gradient removals
revert fast (RMU 12 steps, surgery 16, retain-only 12 — cos stays in
the groove 0.998; ascent reverts SLOW 75 but wrecks the net CE+11.4).
R1: RMU unrescuable (0.007), surgery unrescuable (0.0003),
retain-only RESCUABLE at d5 (0.345). R5 NOT VOID: RMU-minus-retain
deltas NEGATIVE at d1-d3 (CIs excluding 0) — the RMU loss
specifically closes the transplant-rescuable channel beyond generic
fine-tune scar. R4: RMU entropy drift +9% (flag); surgery/ascent
bands per design.

WHAT'S NEXT: T052 registers the reframe (obfuscation as
rescue-channel closure, not probe-generation gap) + the follow-ups.

---

## E091 — reverse-transplant: H-READOUT-GATE — the RMU net refuses even good states (2026-09-26) — DONE

WHAT WE DID: bit-exact RMU/retain replicas (e065 recipe, all
curves reproduced); the reverse sweep (no-removal donors → RMU net
at net0 onset sites, d0-d6, shuffled controls) + two report-only
cells.

WHAT WE SAW (T052 close): reverse-rescue FAILS everywhere — every
d<=5 site-mean ~45x under the 0.30 bar (max 0.0067) while
retain-only sits at 0.345 on the same sites: the RMU net won't
RECEIVE even good states. H-READOUT-GATE fires. RIDERS: (Y) RMU
states → intact net rescue at d5 = 0.362 (half the self-level) —
partial carriage SURVIVES in the RMU net's own d5 state; the
closure is at reception, not state-erasure. (Fallback) the RMU
net's own onset geometry DIED (0 host onsets in 22,400 chars) —
the expression channel itself was killed by the loss.

WHAT'S NEXT: T052 closes: the RMU loss seals the receiving circuit
while the knowledge's d5 trace survives — obfuscation at the gate,
not the store.

---
## E089 — mass-response curve: MASS-ACTION confirmed — the anchor's dose-response law is threshold-shaped (2026-09-26) — DONE

WHAT WE DID: removal cost vs k (random subsets, 10 draws per k,
e088's matched-stream instrument + progressive age-97-crossing
adaptation). 58 s CPU, all gates pass (4,527 fired columns exact).

WHAT WE SAW (T051 final): both threshold clauses FIRE —
cost(32)/cost(200) = 0.034 < 0.25; cost(128)/cost(64) = 3.35 > 3.0;
LINEAR excluded (cost(64)/cost(200) CI entirely below the 0.32
window). Curve: near-flat (CIs straddle 0) through k=32, convex
break 64→128→200 (+0.20 → +0.68 → +2.15). Variance collapses in
CV/sign terms past the break (CV 7.2 → 0.23; all draws positive at
k>=128) — WHICH entries doesn't matter, only HOW MANY. Texture: the
K=1 continuous schedule is ~8x more destructive than e075's K=32
lumps (full off-manifold collapse at k=200); 13 unremovable
p=414 draws tallied; mild exposure confound (Spearman -0.30) below
the break region.

WHAT'S NEXT: T051 closes with its registered confirmation — P3's
arc complete: junk is self-generated, anchors the run as MASS, and
obeys a threshold dose-response law.

---

## E085 — anchor description: the anchor resists per-entry description (2026-09-26) — DONE

WHAT WE DID: 100 anchor + 50 young entries × 5 properties → jackknife
removal cost (static AND free-run clean-judged, matched-stream);
surprisal liar-control; all gates pass (protocol identity vs e080
1.2e-07). 177 s CPU.

WHAT WE SAW (T051): NO bar fires cleanly — readership +0.089 on the
dynamic witness (bar 0.35); surprisal liar-control passes on the
primary (a short-horizon hard-tokens texture appears at +8 window,
honestly recorded); the kill is blocked only by the static arm
(readership 0.341 all-150, PCA 0.281 anchors) — the dynamic arm
alone would kill (all |r| ≤ 0.10). Honest verdict: BETWEEN kill and
partial; B=8 CI width is the fragility flag. RIDERS: T048
replicated at entry level — r(static, dynamic) = −0.004, sign-agree
26%; drift-alignment lower for anchors (0.342) than young (0.495) —
anchor values are more run-specific (descriptive T048 support).
T037-a rider: position-level −0.62 is MECHANICALLY ANTI-COUPLED
(mean vs below-threshold fraction); the less-biased per-entry
reading +0.035 — null-positive, matching the registered prediction.

WHAT'S NEXT: T051 registers the sequence-level pivot (P3's next
step) — the anchor is not an entry property.

---

## E088 — LATE FOLD (R43 audit debt repair; 2026-09-26) — pair-anchor factorial: NO super-additivity — pair removal costs LESS than singles summed — DONE

Backfilled from runs/e088/metrics.json (fold dropped — found by
the R43 audit; started 2026-09-26T15:28:48Z, CPU-only). T051's
registered fork: super-additive pair cost => anchor lives in
entry-PAIR interactions (view b); additive => sequence-level
basin (view a). Result: 40 pairs, 11 qualifying (many with null
denominator — singles cost ~0); median pair/(s1+s2) ratio 0.464,
CI [0.024, 0.5]; Spearman(ratio, pair-distance) ~= 0. The
super-additivity fork does NOT fire — observed is SUB-additive:
single removals already carry most of the pair cost (overlapping
redundant supports). Leans view (a) or plain redundancy; no
T-card (window passed). Flag: 29/40 pairs unqualifying — the
instrument saturates on this install; any sequel must pick pairs
with nonzero single costs first.

## E087 — RIF adjudication: STRING-LEVEL INDUCTION ONLY — reads are pure at the fact level; T049 closes (2026-09-26) — DONE

WHAT WE DID: both rigs × both nets × B=96, arms {real ELIZABETH,
sham, anagram ZIBLETHEA, Z-control ZABMOTHIC}; meta-gap statistic
(real − anagram) sham-free by construction; all instrument gates
pass (Z-drift 1.9e-07/5.3e-08; batched-runner equivalence exact).

WHAT WE SAW (T049 final): meta-bar fires on 2 of 4 legs (both repro
legs: gap +0.078/+0.089, CIs excluding 0, 97% windows) but FAILS on
both dose legs (facilitation, real ≈ anagram) → STRING-LEVEL
INDUCTION ONLY. THE RIG CONFLICT DISSOLVED: at B=96 both rigs agree
everywhere (repro real +0.090/+0.090) — the e081-vs-e081b
discrepancy was B=32 resample luck. THE DECISIVE CONTROL: the
meaningless Z-bearing ZABMOTHIC suppresses FL as much as the real
name on repro (gaps +0.004/+0.013, CIs crossing 0) while the
anagram is dead — the apparent suppression is carried by fine
string statistics, not name/fact identity. Exactly e081's
character-level decomposition.

WHAT'S NEXT: T049 closes — reads do NOT write at the fact level.
The e086 frequency-flip is DEAD (its premise was the fact-level
reading). e084's non-reactivity stands restored at the rule level.

---

## E084 — READ-KERNEL census: KERNEL = SHADOW (r 0.918) — the rule opens what the CE curves measure (2026-09-26) — DONE

WHAT WE DID: 200 margin-stratified decision points × 80 entries × 3
intervention types (24,000 cells after the documented 200-to-100 cut; suffix-window speed path with
exactness gate); flip census + taxonomy + correlation vs stored
dCE-load; all gates pass.

WHAT WE SAW (T050): battery flip rate 6.4% (dynamic range; kill
dead). PRIMARY r(flip-rate, dCE-load) = 0.918 [0.867, 0.940] ≥ the
0.8 bar — KERNEL EQUALS SHADOW: the argmax rule opens what the
CE-shadow says; the dissociation branch did NOT fire (the paper's
shadow-based cache claims stand validated at the rule level). Band
rates: young 1-10 pooled 31% / mid 3.8% / old 0.6% — the kernel is
young-heavy like the spike. Taxonomy: flips go to the runner-up
~48-55%, top-5 ~21-25%, OUTSIDE top-5 ~24-27% — the rule can be
pushed far. V-swap donor-continuation hits: 48.

WHAT'S NEXT: T050 registers the two open textures (outside-top-5
escapes; donor-continuation hits); the read-policy program proceeds
to the rule's DYNAMICS (e087's RIF adjudication feeds it).

---

## E081b — RIF replication: REVERSED — RIF IS PRESENT, asymmetric, n=2 (2026-09-26) — DONE

WHAT WE DID: the T049 discriminator on e048_repro with a tight sham
(B=96 paired half-splits + 200 random-split robustness); gates clean
(Z-drift 5.3e-08).

WHAT WE SAW (T049 amendment): the e081 NULL DOES NOT REPLICATE —
cross EL→FL FIRES on repro (+0.083, t-CI [+0.035,+0.132], boot CI
excludes 0, 79% of 29 windows suppressed, placebo gate passes at
tight resolution); dose was +0.076 marginal in the SAME direction.
The reverse (FL→EL) is FACILITATORY in both nets (repro −0.026;
dose −0.043 significant). The ASYMMETRY (majority-name read
suppresses minority neighbor; minority read facilitates majority) is
present at n=2. Scramble texture dead on repro (+0.003 — dose's
+0.094 was sampling noise). Sham floor: EL 0.013 / FL 0.028-0.043 —
not flat, not cleanly >0.04.

WHAT'S NEXT: T049 closes INTERMEDIATE→finding: reads write
asymmetrically. e086 registered: frequency-flip install (41 FL / 19
EL) — predicts the asymmetry follows the frequency ratio.

---

## E081 — RIF probe: NULL — reads are pure (dose net; instrument caveat) (2026-09-26) — DONE

WHAT WE DID: proposal P-A (explorer harvest) — prompt-only elicitation
of one installed name, then the neighbor name's expression at its own
geometry; 4 arms + scramble control; dose net primary.

WHAT WE SAW (T049): no RIF — cross-name suppression did not fire the
registered bars. The placebo gate itself failed (sham moved 0.044 /
0.021 vs the <0.01 flat requirement) — the instrument is noisier
than designed, so the null is directional. Report-only texture: the
scramble control's EL→FL direction shows +0.094 [0.011, 0.178]
(unexplained, flag for the replication cell). Answer to P-A as
measured: reading does not write at this resolution.

WHAT'S NEXT: T049 registers the purity-vs-coarse-instrument split;
the e048_repro cell is the natural replication (unrun — this was the
dose-net pass). Positive bearing on e084: the read census is
non-reactive — measuring the rule won't contaminate it.

---

## E076 — cosine mechanism + critic fixes: alignment survives; instrument out-of-sample-robust; NOT cosine-specific (2026-09-26) — DONE

WHAT WE DID: three-part reanalysis on the 204 cached pairs, every
instrument reproduced bitwise (e062 cosines, e059 align, e050 dw_cos
to 5.6e-09). 22 s CPU.

WHAT WE SAW (T046 close-out): (1) H-basis-alignment SURVIVES —
partial r −0.982 after partialing on A + donor W_out/W_in row-norms +
host stream-norms (strengthens from −0.976); honest caveat: in-situ
donor write-mass correlates 0.902 with D and shares variance
(dropping partial to −0.86, still over the 0.8 bar). (2) INSTRUMENT
SURVIVES: out-of-sample AUC 0.885 (resplit median 0.919, 100% of 200
splits ≥ 0.85); leave-one-lineage-out partial r ∈ [−0.980, −0.969];
leave-one-donor-out [−0.979, −0.966]. (3) NOT COSINE-SPECIFIC
(honest negative): the e052/e059 dW-alignment axis is near-
equivalent (partial −0.942, AUC 0.940 — beats the cosine as a
classifier); the cosine's residual advantage is practicality — a
2-batch probe, no init-lineage knowledge.

WHAT'S NEXT: T046 closes. Claim D headline: the crossmatch signal is
real, out-of-sample robust, and measurable by either instrument —
the cosine probe is the cheap deployment. P2 proceeds with it.

---

## E079 — junk-split B=16 resample: claim C is NET-DEPENDENT (2026-09-26) — DONE

WHAT WE DID: B=16 regeneration on e053c (e072 seed-family extension)
+ propagated B=16-equivalent CIs for the e053 2.7M/10M stored
profiles; paired-bootstrap CIs; concentration census.

WHAT WE SAW (T045 addendum): bar 1 FAILS on e053c — gen-old junk
0.114 vs prompt 0.099 (ratio 1.14, diff CI straddles 0, p-hold 0.29;
the fresh-12 harder-battery confound from e072 recurs). bar 2 FIRES
on BOTH mid_2.7M and large_10M (propagated CIs hold the contrast —
the 10M carries the strongest original signal 0.367 vs 0.000).
Concentration flag CLEARS at B=16 (max single-seq share 18.6% <
40% bar). Honest close: the source split is real where the effect is
large (2.7M/10M), B-4-only on the ctx-512 net.

WHAT'S NEXT: paper 5.5 wording gains the net-dependence precision;
critic claim-C flags resolved (threshold flag downgraded to
net-dependence; concentration flag cleared).

---

## E080 — prune-vs-replace: honest MIXED — neither presence nor prompt-content rescues; the anchor is run-specific (2026-09-26) — DONE

WHAT WE DID: the T048 discriminator on the e075 rig — A-noise
(norm-matched, verified 7e-07) vs A-promptcopy vs the known A-vzero
(bit-for-bit drift check, dev 0.0). B=8, same prune events.

WHAT WE SAW (T048 close-out): H-statistics-scaffold DEAD (noise
+0.287 ≈ vzero +0.262 — presence is not load-bearing).
H-content-anchoring MISSES its bar (promptcopy +0.0688 > +0.05, CI
straddles) but recovers ~3/4 of the cost and beats noise by +0.218.
The attractor signature fires in ALL THREE arms (clean-judge gap
5.0-5.4 nats; onset lost everywhere) — even promptcopy, whose
self-scored fluency nearly recovers, free-runs off-manifold by the
clean judge. promptcopy's young spike GROWS 2.5x (ages1-5 4.4-6.0 vs
1.4-2.4). Junk inversion full under vzero/noise (0.375), partial
under promptcopy (0.125).

WHAT'S NEXT: T048 closes: neither presence nor generic content —
the generation anchor is RUN-SPECIFIC trajectory content. Free-run
honesty reconfirmed as the only reliable witness. Registered: the
natural completion is a self-copy arm (reinsert the run's OWN pruned
contents — trivially circular, so instead the P3 line pivots to
describing the anchor, not replacing it).

---

## E078 — LATE FOLD (R43 audit debt repair; 2026-09-26 era) — dose rebinding at 4x dose: T047's ROW129 claim REPLICATES clean — DONE

Backfilled from runs/e078/metrics.json (fold dropped — found by
the R43 audit). Question: does ROW129-ONLY ~= PAIR-COPY >>
ROW0-ONLY ~= NO-COPY replicate on the second (4x-dose) install?
All three registered bars fire at BOTH k=10 and k=20:
row129~pair (min/max = 0.97: 0.302 vs 0.294 at k10, 0.294 vs
0.285 at k20, install-60 shifted); gap (row0/nocopy 0.108-0.117
<= 0.5 x pair/row129); row0~nocopy (0.95). Held-30 carries
~40% (0.125-0.127). T047's "~70% rebind, partial necessity — the
single most load-bearing portable row" re-confirms at 4x dose:
the row-129 geometry alone carries the rebind; row-0 adds nothing.

## E077 — untrained-init profile: TRAINING-BUILDS — the template is a fast training-dynamics emergent (2026-09-26) — DONE

WHAT WE DID: A-profiles on 6 untrained inits (2.7M/6L seeds 42/43/777
primary + 0.84M/4L support) + a 2000-draw permutation null for
shape-r (shuffled site-orderings + matched-marginal gaussians);
e063 machinery verbatim.

WHAT WE SAW (T041 final stamp): untrained |A| ≤ 0.061 nats at every
site of every seed (vs trained 0.15-4.08) — NO organ load exists at
init; sites-1-5 r median −0.33, deep inside the null; the full-profile
~0.9s were the quantified L0-domination artifact (null 95th pct
+0.984). CALIBRATION: T041's trained-vs-trained shape-rs SURVIVE —
sites-1-5 recompute to +0.993/+0.983 at the 99.0-99.4th pct. Sanity:
untrained CEs within +0.067 of ln(65).

WHAT'S NEXT: T041 final phrasing: the universal allocation template
is a FAST TRAINING-DYNAMICS EMERGENT (present by ~10³ steps — the
e021 913-step control had it; not init-carried; invariant thereafter).
Variance-ladder story unchanged.

---

## E075 — source-aware pruning: KILL fires — static junk is dynamically load-bearing (2026-09-26) — DONE

WHAT WE DID: the T045 intervention test per frozen design — 4 arms
(none / self-prune V-zero age>96 / prompt-placebo / both), B=8,
permanent mid-run pruning every K=32 steps, all 7 gates pass
(battery-A identity vs e069; arms token-identical through pos 164;
prune counts exact).

WHAT WE SAW (T048): A-self-prune costs +0.2619 nats (CI [+0.10,
+0.45]; 7/8 sequences worse) — 5× the kill bar; R2 fluency fails
(entropy +29%); R3 violated: pruned arms LOSE the onset entirely
(live_frac 1.000, everything above threshold). KEY TEXTURE: the
clean-judge check shows pruned tails self-score as fine but cost 6.40
nats under the clean net — pruning drives generation into a
self-consistent off-manifold attractor. The junk census inverts after
pruned generation: the once-harmless prompt band turns junk-heavy
(0.375).

WHAT'S NEXT: T048 registers the mechanism (dynamic-load vs
content-anchoring; e080 prune-vs-replace discriminator). Paper 5.5
carries the boundary: eval-time lesion utility ≠ generation-time
prunability.

---

## E062 — crossmatch predictor: P2 stream-cosine WINS (partial r −0.976; AUC 0.919) (2026-09-26) — DONE

WHAT WE DID: 204 graft pairs (18 hosts, e040/e058/e059 machinery
reproduced bitwise), predictors tested in the R34-mandated order:
W_out rowmean first, pre-graft stream-cosine at graft-input depths
second; partial r given organ-load A; host-cluster CIs; ROC.

WHAT WE SAW (T046): P1 (W_out rowmean) FAILS as a decision rule —
partial r −0.204 (real signal) but AUC 0.533 (chance). P2
(stream-cosine) WINS decisively: partial r(D|A) = −0.976, host-cluster
CI [−0.984, −0.963]; decision rule "graft only if cos ≥ 0.4459" at
AUC 0.919, Youden J 0.657. Scale-B secondary: transfers at partial
r −0.997 (2.7M, n=6 directional, no bars). A itself predicts nothing
across pairs (r 0.080 — consistent with T041's variance-free A).

WHAT'S NEXT: T046 registers the mechanism question (basis-alignment
vs magnitude-proxy) + the tolerance-transfer prediction. P2
IMMUNOLOGY has its screening instrument; crossmatch table (v014)
buildable.

---

## E074 — shuffled-prompt junk control: H-SOURCE fires — the poison is self-generation (2026-09-26) — DONE

WHAT WE DID: replaced prompt entries with shuffled chars (93% slots
moved; corpus statistics destroyed, age/count/recency kept), re-ran
the V-zero sweep on the same sequences + generation-order secondary.

WHAT WE SAW (T045 close-out): shuffled-prompt-band junk 0.024 (CI
[0.008, 0.040]; max-over-seqs 0.048) — ≤ the 0.05 H-source bar;
destroying corpus statistics created NO new junk in the prompt band.
Generated band stays junky (0.102 vs baseline 0.0995). Secondary
(base run): generation-order gradient CONFIRMED — late-generation
junk 0.169 vs early 0.031 (5.5×). Age+count+recency alone do not
make an old band go negative. Manipulation note: clean CE slightly
FELL with shuffled prompts (−0.037, one seq −0.147) — flagged,
secondary.

WHAT'S NEXT: T045 closes strict — cache junk = accumulated
self-generation drift (exposure bias at entry level, now
confound-free). P3 program proceeds to pruning-by-source design.

---

## E072 — value-vs-threshold: BOTH fire — per-norm value efficiency + a* B-fragility (2026-09-26) — DONE

WHAT WE DID: V-vector norms + post-attention residual-write norms at
ages 4-17 across windows (same tokens); B=16 bootstrap of a* on
eval-256 (+ eval-512 secondary). Protocol replica bit-exact.

WHAT WE SAW (T044 close-out): V-norm ratio 0.887 [0.876,0.910] —
outside [0.9,1.1], H-value-side fires BY THE FROZEN ORDERING, but the
sign is DOWN: magnitudes shrank ~11% while lesion dCE grew (residual
write −23%) — load PER UNIT value-norm increased (specificity, not
magnitude). H-threshold ALSO fires: B=16 a* = 7 [4,13] (was 18 [7,30]
at B=4); the B=4 window contrast (6 vs 18) collapses to (6 vs 7);
resamples land 20% at 6, 24% at 12-24. Confound flagged: fresh-12
battery is harder (CE 1.499 vs 0.459); ages-6-17 haze at B=16 is dead
(−0.0013).

WHAT'S NEXT: T044 closes — surviving story: attention-invariant,
magnitude-invariant load redistribution inside the value pathway
(per-norm efficiency) + a standing B-fragility flag on the a*
statistic. Paper 5.5 qualifier updated accordingly.

---

## E073 — junk-split reanalysis: the sleeper hypothesis fires 4/4 — cache junk is self-generated (2026-09-26) — DONE

WHAT WE DID: zero-GPU stratification of negative-utility cache entries
by SOURCE (generated vs prompt) across the four e053-family profiles
(prompt = oldest 64 corpus entries; generated = the free run's own
tokens; threshold −0.01 nats; beyond-onset restriction on the
generated side).

WHAT WE SAW (T045): generated-old junk 0.101 / 0.085 / 0.367 / 0.100
(small/mid/10M/ctx-512) vs prompt junk 0.000 / 0.032 / 0.000 / 0.063.
H-sleeper FIRES 4/4; H-inverse (far-context poison) dead. The 10M is
extreme: 36.7% of its beyond-onset generated entries are
lesion-HELPFUL and their mean dCE is negative, while all 63 prompt
entries are positive-utility. In both ctx-512 and 10M, generated-old
mean dCE < 0.

WHAT'S NEXT: T045 registers the age-vs-source discrimination (entries'
age and source correlate by design; shuffled-prompt control) and the
paper bridge (entry-level exposure bias — the free run poisons its own
cache).

---

## E068 — pair-rebinding: MIXED — the portable unit is ROW 129 ALONE (2026-09-26) — DONE

WHAT WE DID: wpe-row surgery on shifted install windows: pair-copy
(k,k+129)←(0,129), single-row controls, no-copy shift, reverse-context;
k ∈ {10,20}; install-60 + held-30 batteries. Restore checks bitwise.

WHAT WE SAW (T043): PORTABLE-PAIR bar fires (0.35-0.38 install-60 /
0.24-0.27 held-30 at new geometry, ~70% of unshifted) — but
CONJUNCTION-UNIT bar FAILS: row-129-ALONE equals the pair (0.397 vs
0.384); row-0-only sits at the no-copy baseline (0.14 ≈ 0.145).
No-copy shift only partially collapses (0.13-0.15 = T032 plateau).
Reverse-context: anchors alone at old geometry are inert (Δ −0.003) —
expression needs (content, anchor) ALIGNMENT.

WHAT'S NEXT: T043 registers the dissociation reading (destruction
weight vs portability) and the KL-calibrated refinement of T042.

---

## E070 — attention-mass discriminator: NO CLAUSE FIRES — the registered space was jointly insufficient (2026-09-26) — DONE

WHAT WE DID: per-layer×head attention-received mass (final query) at
ages 1-20 / 4 / 5 / window-start / 21-255, eval-256 vs eval-512, same
sequences; K/V lesion probes; native-positions-0-255 slice a*.

WHAT WE SAW (T044): ages1-20 ratio 1.006 [0.97,1.03]; age4 0.98; age5
1.03 — young-age attention is WINDOW-INVARIANT. Window-start 0.76
[0.38,2.41] — not ≥2. Mid-far ages 21-255 gain 1.44 [1.35,1.54],
EXCEEDING pure softmax renorm (1.265): real restructuring, landing
mid-far. Native-slice a* = 13 [6,24] ≠ 6 — window length per se shifts
the onset. NONE of the three registered hypotheses fires cleanly: the
ages-4-17 tail liveness grew with UNCHANGED attention mass and
unchanged clean CE.

WHAT'S NEXT: T044 registers the value-side vs threshold hypotheses;
the T042 cross-prediction (window-start mass large) is REFUTED — the
cache-thread and address-thread window-start objects are NOT the same
attention phenomenon.

---

## E071 — row-0 generalization: H-WINDOW-KEY clean sweep (2026-09-26) — DONE

WHAT WE DID: row-0 intervention (mean/zero arms) × {install-60,
held-30, uniform-60} × {installed, base} + row-129 secondary + never-fed
null check (rows 160/200/240: exact 0.0).

WHAT WE SAW (T042 outcome): row-0 drop on held-30 = 0.428→0.0009 —
the window-key generalizes to UNTRAINED install-family windows;
uniform and base cells floor-limited (no name to erase). Secondary:
row-129 ALSO generalizes (held-30 drop 0.199/0.282 vs install-60
0.240/0.342 — ~83% strength). Honest nuance: the KL column shows row 0
is generically load-bearing too (base-net KL 3.3-4.9 nats) — generic
importance + specific key-role coexist; the registered binary was
coarse and the data gives the conjunction.

WHAT'S NEXT: e068 ungated with the pair-design ({0,129} anchor copy to
a new window position). Window-TYPE recognizer confirmed (not
exact-window memory).

---

## E067 — address census: ROW 0 is the top anchor — window-anchored conjunction, bimodal + carpet (2026-09-26) — DONE

WHAT WE DID: full 256-row wpe perturbation census (row←mean arm + zero
arm for top rows) × install-60 battery, replicated on the 4×-dose
install net; d5 Δstate vs relay for top-5 rows. 19 min CPU; protocol
rebuild bit-exact (0.55631).

WHAT WE SAW (T042): NOT SPARSE at the registered 80% rule (45-48 rows
needed) but the registered dense-cluster picture is ALSO refuted — the
mass is BIMODAL HOTSPOTS at rows 0 and 129 (two-row mass 53%) + local
smear 123-128 + a generic micro-carpet (~240 rows × ~0.002, which
breaks the 80% rule). ROW 0 is the single most load-bearing row:
perturbing it alone erases the install (p(Z) 0.556→0.0014; zero-arm
0.546; dose net replicates 0.493). Rows 130-255 exactly 0.0 (sanity).
Secondary: high-weight rows do NOT feed one relay (mean pairwise |cos|
0.216); row 0's Δ anti-aligns with relay_d5 at −0.724 (strongest).

WHAT'S NEXT: T042 registers e071 (row-0 generalization: held-30 +
uniform batteries × base-net control — window-key vs generic-start).
e070 (running) measures the same window-start object from the cache
side. e068 rebinding design must include row 0.

---

## E063b — task-swap discriminator: H-ii OPTIMIZER-ATTRACTOR (2026-09-26) — DONE

WHAT WE DID: A-profiles on the e021 family (copy-task net: far-retrieval
learned, +3.27 far-value, 99.96% copy acc; + its control net as bonus
third), own-corpus val, e063 machinery verbatim, B anchor bitwise.

WHAT WE SAW (T041 amendment): copy-net shape r = +0.998 vs B template
(control +0.998 too); H-i's registered r < 0.5 prediction decisively
failed. The L0-huge/trough/rise template is corpus-invariant — an
optimizer/architecture attractor, not task-pinned. Nuance: the task
re-weights MAGNITUDES (mean |ΔA| 0.27, uniform elevation, L1 trough
partially filled 0.42 vs 0.15 — Spearman dips to +0.83 barely at bar);
shape is preserved, scale is not.

WHAT'S NEXT: T041 closes (template = attractor). Remaining open: what
WOULD move the shape — e033's energy constraint is the only known
mover; depth/architecture sweep parked.

---

## E069 — T039 onset discriminators: H1 circuit-horizon DECISIVE; D1 surprise — eval-window-sensitive (2026-09-26) — DONE

WHAT WE DID: the two T039-registered eval-only discriminators on the
frozen e053c net (CPU, 282 s, gates G0b/G1/G2/G3 all pass; a*(512)
reproduces 6 [4,8] bitwise).

WHAT WE SAW (T039 amendment): **D2 = H1 CIRCUIT-HORIZON, decisive** —
shuffled-char contexts (n-gram statistics destroyed, recency kept) leave
the ages-1-2 spike at 128% of normal (bar: ≥30% H1 / ≤10% H2; CI
[0.98, 1.70], per-seq 0.68-1.95). The recent-token spike is circuit
structure, not corpus statistics. **D1 = EVAL-WINDOW-SENSITIVE
(surprise, recorded as-is)** — the SAME net + sequences truncated to
eval-256 give a\* = 18 (CI [7,30]), outside the registered [4,8];
absolute-position-indexed variant gives 12. The horizon is absolute
w.r.t. the TRAINED window (e053c's cross-window verdict stands) but
shifts under eval-window truncation.

WHAT'S NEXT: T039 amendment card registers the D1 interpretation
question (attention re-anchoring vs instrument reference-shift);
e053c's window-invariant truncation claim needs the eval-window
qualifier in the paper.

---

## E063 — load homeostasis: H-EMERGENT — organ-reliance is a universal template (2026-09-26) — DONE

WHAT WE DID: own-organ load A tracked across cohorts (2.7M/6L: B vs BDO
[same init, diff order] vs B43 [diff init] + exposure arms; 0.84M/4L:
all-11 e040 lineage + e005s + e033), bootstrap noise floor, profile
correlations. CPU-only. Premise corrected en route: e041_bdo/e048 are
2.7M trainstates.

WHAT WE SAW (T041): H-EMERGENT at the registered ladder. Ladder: noise
0.014 | ORDER 0.063 | INIT 0.054 | exposure max 0.162 (base-CE confound
flagged). Profile shape r = +1.000 across order AND init — the
allocation template (L0 ~4 nats, L1 trough ~0.15, monotone rise to
~0.6) is universal. H-setpoint DEAD (order moves A more than fresh
init, 1.17×; replicate rung 0.336 beats init 3×). e033's energy
constraint is the biggest single mover (and even it keeps the shape).

WHAT'S NEXT: mechanically explains T024's trickle and T040's H-nothing
— A has no heritable variance to select; what scatter exists regenerates
through training noise. T041 registers the task-swap discriminator
(e063b, zero-GPU) and the e060 prediction (A-residualized selection
should move the e059 interface family instead).

---

## E059 — winner differencing: H-NOTHING at the bars; interface family is a second predictor (2026-09-26) — DONE

WHAT WE DID: zero-GPU ΔW audit of e040 lineage winners ({g1a,g1c}+children
vs unselected sibs), 8 axes, bootstrap CIs, gates clean (repro 8.7e-08,
REF self-graft 0, organ-band verbatim). 8.4 min CPU.

WHAT WE SAW (T040): no consistent winner signature at the registered bar
(|r| ≥ r(D,A)=0.807). A did NOT move (H-load's directional claim fails).
What winners DID: LN→closer to donor REF (weak), W_out row-norms +2.5%
with shape→REF, W_in erank→AWAY from REF. Sharpest: D itself is
inconsistent as a winner property (E1 −0.05 vs E2 +0.10); the gen-2
trickle was carried by the children — single-lineage artifact at
parameter level. SECOND PREDICTOR FOUND: interface-scale family
(W_out row-norm mean partial r(D|A) = −0.654 p=.040) — genuine
damage axis beyond organ-reliance, but not what selection changed.

WHAT'S NEXT: e063 (running) discriminates WHY A didn't move — defended
setpoint (canalization) vs invisible-to-R; T040 registers the
prediction. Interface family → P2 IMMUNOLOGY crossmatch predictor
candidate (e062).

---

## E053c — ctx-512 onset decider: ABSOLUTE verdict (2026-09-26) — DONE

WHAT WE DID: trained 4L/4H/128d/wpe-512 = 873k params (seed 42,
tokens-per-step matched to the e005s comparator; 180s cap hit at
3133/4000 steps = 78% exposure), then the e053b fine-onset machinery
verbatim (same seeds/rules, 511-position V-zero sweep, 1000× bootstrap).
GPU envelope respected (88°C peak → cooldown + re-check before eval).

WHAT WE SAW (T039): **a\*(512) = 6, CI [4,8], naive=robust=6** —
overlaps the ABSOLUTE window [5,13], entirely below PROPORTIONAL
[10.0,26.1]. Onset fraction HALVED (0.012 vs 0.027). Spike shape
preserved (ages 1–5: +1.7/+4.5/+2.3/+0.9/+0.3 nats; ≤0.007 after age
6). Val CE 1.5227 (beats both anchors). Gates G0b/G0-dev pass
(1.8e-06 / 1.6e-05). Honest caveats: 78% exposure biases AGAINST this
verdict (e053b: less training → larger a*, yet 6 ≤ 7); B=4 spread 3–8;
identity audit stays broken and widened (9.3% of positions
lesion-helpful, scattered to age 499) — plateau ≠ pure recency noise.

WHAT'S NEXT: paper cache-truncation claims restated window-invariantly
(fixed-token horizon). P3 junk-split now ungated (GPU free after
cooldown). Registered T039 discriminator: eval-window truncation on the
SAME net (a* at eval-256 on the e053c net) to separate circuit-horizon
from statistical-horizon readings.

---

## E066b — in-place row-129 interventions: the address is GRADED, not row-pure (2026-09-26) — DONE

WHAT WE DID: T038's H1-vs-H2 discriminator — in-place wpe-row-129
interventions on the 6 donor contexts (swap←130 / zero / mean-row vs
normal), p(Z) readout + d5 Δstate alignment vs relay. 29 s CPU.
Predictions registered in-script before compute.

WHAT WE SAW (T038 update): MIXED/NEITHER at the registered bars —
swap 0.462, zero 0.272, mean-row 0.434 (from 0.715): the row is
PARTIALLY load-bearing in place. Δstates anti-align with relay_d5
(−0.21…−0.26): breaking the address REDUCES relay content (pro-circuit
flavor, moderate). Per-donor spread 0.68→0.23 under swap — contexts
differ in how self-sufficient their content is (conjunction view).

WHAT'S NEXT: e067 address census — full single-row perturbation sweep
maps this partial-weight structure; e066/e066b justify it (the address
is distributed but wpe-row-concentrated).

---

## E066 — close the loop: relay direction vs wpe address row = TWO OBJECTS (2026-09-26) — DONE

WHAT WE DID: P1-COORDINATE ramp step 1 (ideator-registered, T037-refined).
Exact e055 protocol rebuild on e048_repro; mean-donor relay per depth
d0–d6; cosine sweep against all 256 wpe rows; delta-relay (minus shuffled
family), wte sweep + adjacent-row controls. 14 s CPU.

WHAT WE SAW (T038): registered metric |cos(relay_d5, wpe[130])| = 0.094
→ TWO-OBJECTS verdict (bar: ≥0.4 closes, ≤0.15 = distinct; null sd 0.072,
mean |row~row130| baseline 0.237). Depth profile: d0 cos 0.507 @ row 129
(decision position; partly mechanical — shared wpe survives donor
averaging), decaying through the stack to noise at d5 (max anywhere 0.121
@ row 152; rank of row 130 = 19). Delta variant 0.113 — same verdict.
wte control max 0.15 @ ' '. The T037 construct-1 optimism is REFUTED at
depth: the deep relay that rescues p(Z) at 0.912 is NOT the position row
itself.

WHAT'S NEXT: T038 discriminator — is the relay ANSWER-shaped (cos vs
Z-unembedding row) or circuit-address-shaped? Registered in T038; then
e067 address census (is the input-side code sparse?).

---

## E056c — downstream check: LOUD LOGIT PASTE at non-onset (2026-09-25) — DONE

WHAT WE DID: the R14-registered discriminator — free-run continuations
after one-shot d4 writes at the 24 non-onset positions; persistence
curves; 480 continuations total.

WHAT WE SAW (T035): the rescue is TRANSIENT off-onset — p(Z) 0.509 at
+1, floor by +2, zero recurrent Z-words (only 6 offset-0 "ZEPHY:" tag
  completions). Knowledge-specific but not portable. Claim-split final:
position-bound knowledge; real suppression at the bound position;
durable rescue there (e055); logit artifact elsewhere. The YOPO
collision risk shrinks — our write fails to steer even 2 tokens ahead.

## E056b — the circularity killer: rescue is position-general (2026-09-25) — DONE

WHAT WE DID: the d4-rescue donors transplanted at 24 floor-prior
non-onset positions (trajectory-identity verified).

WHAT WE SAW (T034): the rescue fires ANYWHERE (0.324 site-mean, AUC
1.000, shuffled ~0) — circularity resolved moot; the claim SPLITS:
onset-specific sub-argmax knowledge + natural wpe-130 knife-edge;
position-general knowledge-specific d≥4 state injection. Plus e055
full-report: A-rev symmetric suppression; mean-donor relay direction
beats all individuals. Paper draft spine written; submission gate open.

## E055 — the suppression localizer: causal, state-carried, depth-4 (2026-09-25) — DONE

WHAT WE DID: 24-site depth-survival transplant (TF-state at the onset
position, one-shot + held; shuffled/A-rev/base-net/direct800/e001
controls; R1/R2/R3 readouts), all gates pass.

WHAT WE SAW (T033): P1 CONFIRMED (state-rescue d4 0.374 / d5 0.494 vs
shuffled 0.000); P3 CONFIRMED (d*=4 mid-stack; d1-peak/d2-crash
replicated 10.3x); 32 downstream Z-word rows — rescued states express.
The expression gap is causally localized: position-bound knowledge,
destroyed across blocks 1-2, restorable from depth 4. The interventional
study the literature lacks.

## E053b — fine-onset remediation: the invariance was quantization (2026-09-25) — DONE (2-cell partial, honest smoke flag)

WHAT WE DID: fine-grid onset fitting on the cache-timeline cells (no
bin-edge fallback), bootstrap CIs, identity test.

RECONCILED vs final e053 metrics: registered-threshold a* = 73/86/4 and 3→21→86 across exposure (GROWS 28.7x, the registered direction); the shrink claim was a sign-based statistic — OPEN CONFLICT documented, ctx-512 + more seqs to resolve. Shape claims robust: spike+plateau, sink dead 5/5, 32% negative-utility old positions at 10M.

Superseded partial read: fine a* = 7/25/35/182/32 — 0/5
near "63"; the curve is a sharp recent spike (~4-15 positions, dCE
0.3-6.7) + dead old end (240-255: dCE <= 0.015, sink included) + 13-20%
NEGATIVE-utility entries (lesion helps); exposure INVERTS the prediction
(training SHRINKS the live window: live-frac 0.68→0.17→0.13 across
400→800→4000 steps); identity broken with sign flips. Honest publishable
core: ~85-95% of the ctx-256 cache is dead weight; sink dead at
generation; training collapses the live window. ctx-512 = the registered
absolute-vs-proportional decider.

## E064 + E053 — the stress test kills the unification; the timeline lands (2026-09-25) — DONE

E064 (gate stress): **KILLED** — R43's ladder peaks L3 with its gate at
L5 (and R peaks L0 with its gate at L4): the mid-stack interference peak
is shared-method idiosyncrasy, not gate-tracking. T029's unification
dead; two-mid-stack-phenomena residue recorded.

E053 (cache utility timeline, 25.7 min, 5 cells): sink DEAD at
generation (0.007 dCE); utility NOT clean recency (25-50% non-monotone
positions); **live fraction scale-invariant ~25%; onset age ~63
invariant across scale AND exposure** (the grows-with-exposure
prediction refuted); sink-cost decays at 0.84M/10M, flat at 2.7M. First
causal per-position utility curve — prunable-cache numbers.

## E058 — geometry-site anatomy: the causal-gate region carries the anchor (2026-09-25) — DONE

WHAT WE DID: per-site r(damage, geometry) across 11 ckpts at 0.84M + 12
B/B43 grafts at 2.7M; R-ladders by site; all instrument gates bitwise.

WHAT WE SAW (T029): A-residualized geometry replicates at depth (partial
r 0.92-0.96); interference peaks SLIDE with each host's causal gate (B
L3, B43 L4); L0 shows complete criticality/basis-specificity dissociation
(self-ablation +4.08 yet cross-seed R=1.03); anti-alignment exactly at
the rejection zone. The anchor lives in the causal-gate region.

WHAT'S NEXT: e053 (cache timeline) still computing; then interpretation
block before new dispatches per the critic's ratio flag.

## E050 — directed-mutation lineage: VISIBILITY-LIMITED (2026-09-25) — DONE

WHAT WE DID: the e040 protocol with mutation restricted to stream-facing
matrices (W_in/W_out) — the reachability-vs-visibility discriminator
registered in T024.

WHAT WE SAW (T027): D fell only -3.6% (bar -25%), alignment flat, gates
clean. Even directed-at-the-basis mutation gives selection nothing to
work with through the graft-damage trait. FROZEN holds at its strongest;
e060 (A-residualized index) is the last escape hatch.

## E052 — LN/geometry reanalysis: damage tracks organ-reliance (2026-09-25) — DONE

WHAT WE DID: zero-GPU regression of e040's 11-checkpoint graft damage on
four predictors (own-ablation A, LN-distance, dW-alignment, W-space),
bitwise-exact reproduction.

WHAT WE SAW (T026): A dominates (r=0.807); LN excluded (ΔR²=0.066);
geometry aggregate weak but a real L2-localized signal (r=0.747).
FROZEN survives reinterpreted — the trait measured organ-reliance, and
future compatibility selection must use A-residualized damage.

WHAT'S NEXT: e050 (directed mutation) should read out A-residualized or
L2-weighted — method note registered before its results land.

## E040 — graft-evolution lineage: init-anchoring is FROZEN (2026-09-25) — DONE

WHAT WE DID: 11 step-matched 0.84M trainings (seed-42 family wildtype + 3
mutants; fixed REF donor organs; 2 selection events x 3 children;
selection on cross-seed MLP graft damage with parity + organ-band gates;
thermal blocks throughout).

WHAT WE SAW (T024): **P2-FROZEN.** Damage fell only 7.3% over two
generations (P1 bar: 25%); R fell 2.2%; REF-alignment stayed at the floor
(0.001 -> 0.006 vs a 0.53 ceiling). CIs exclude zero — a real trickle,
not noise — but selection cannot see the stream basis at this regime.
The degeneration-route did not fire. C3 complete: the basis is
init-anchored, partial, interface-specific, and not evolvable under
standard selection.

WHAT'S NEXT: day-two report's last slot fills; the day-two arc is
complete. The lab's three stories (anatomy, editing, evolution) all have
first data.

## V012 — portrait v2: the 48-hour self-portrait (2026-09-25) — DONE

WHAT WE DID: v010's four panels refreshed with post-audit numbers (C1
re-anchored to the causal census + its non-invariance) + two new panels:
the retrieval-threshold curve (T021 flip from interference) and the
four-box edit law (T015/T018/T019, n=1 flags boxed). CPU-only.

WHAT WE SAW: the lab's third flagship artifact — one figure that carries
the whole 48-hour story with its confound flags visible.

## E033 — write-equalizer: the energy schedule is decorative (2026-09-25) — DONE

WHAT WE DID: fresh 0.84M net with every MLP write renormalized to one
uniform norm (1.64; hooks train+eval); baseline lesion map + equalized
lesion map + calibrator KL. Envelope-compliant (batch 32, cooldown).

WHAT WE SAW (T023): P1 parity TRUE (1.5147 < baseline 1.537 — BETTER);
P2 energy-migrates FALSE (attention unchanged; MLP damage rose/flatten);
P3 calibrator survives TRUE (KL 1.176). The late-MLP energy carrier is
real but the growing schedule is an allocable habit — the net defends
the coarse allocation, not the write norms. 0.84M-scoped, single net.

WHAT'S NEXT: e040 lineage RUNNING (11 step-matched trainings, thermal
blocks). README updated with the day-one/two results summary.

## E049 — the retrieval threshold (2026-09-25) — DONE

WHAT WE DID: refrain corpora at p ∈ {0,5,20,60}% (24-32-char verbatim
refrains in Shakespeare filler), 4 fresh 2.7M nets + a 10M arm at p5;
far-value / accuracy / retrieval-head readouts at refrain AND ordinary
positions.

WHAT WE SAW (T021): threshold ≤5% (refuted-low), GRADED not sharp, ZERO
leak (compartmentalized), 10M 1.87x more sensitive (weak support). THE
FLIP: far context HURTS at p0 (−2.24 nats, T007's interference) and turns
net-positive by p5 — far-value is a tug-of-war flipped by ~200 refrain
events. Retrieval heads form discretely in the LATE-ATTENTION slot (L5 at
6 layers, L7 at 8) — preferred depth, not preferred density. Cross-talk
at 60%.

WHAT'S NEXT: L4's arc complete (no-retrieval → scale erosion → threshold
mapped). Overnight program: e040 graft-evolution re-scoped remains; then
session review.

## E005s — the scaling capstone (0.84M / 2.7M / 10M) (2026-09-25) — DONE

WHAT WE DID: two new nets (4L/4H/128d = 0.84M; 8L/8H/320d = 9.98M), same
corpus/seed, frozen readouts P1-P4 registered before training.

WHAT WE SAW (T020): gate structure universal with relative depth sliding
by architecture (1.0 -> 0.6 -> 0.29); front-loading INTENSIFIES (11x ->
72x -> 116x; at 10M the three deepest attention blocks cost <=0.04 nats);
16-token sufficiency erodes monotonically with scale (+0.021/-0.001/+0.035);
address surgery scale-invariant in shape (S_name 1293/332, class-exact,
+0.0005 corpus). Distributed decisions collapse with depth (26->15->7%).
Harness bug fixed (resume map_location).

WHAT'S NEXT: card v3 scale-stamped; e049 (far-retrieval threshold) gains
priority from L4's erosion.

## V011 — the edit film (2026-09-25) — DONE

WHAT WE DID: six-frame filmstrip of law L7 from saved metrics (VISUALIZER
agent, CPU-only). runs/v011/edit_film.png.

WHAT WE SAW: baseline → address burn (S_name 573, real burned-net
generation) → complete erasure (n=1 boxed) → cheap install beside the
expression collapse (p(Z) 0.556→1.7e-6 log bars; "0 × ZEPHYRA in 2,800
chars"; the installed net opens ELIZABETH) → the scar (groove cos 0.760
vs 0.278; anti-carrier flip; 44.5% key-resistance, n=1) → the four-box
law. The lab's second user-facing artifact.

## E048 — expression gap: teacher-forcing-bound, at every dose (2026-09-25) — DONE

WHAT WE DID: dose x3, seeding (induction route), temperature x3, greedy
diagnostic on the installed cell; battery + free-generation readouts at
every arm.

WHAT WE SAW (T019): expression = 0 in ALL arms while battery holds
0.92-0.96. P1 confirmed (teacher-forcing-bound); P2/P3 refuted (no
threshold, no dose response). C7 final: address / ability / expression /
history — four separable faculties; install-by-teacher-forcing is
constitutionally silent. Doctrine: continuation batteries are not evidence
of usable knowledge; free generation is the honesty check.

WHAT'S NEXT: night program continues (e049 retrieval dose-response;
e040/e032/e005s gated). Review due.

## E044 — scar tissue (REAL run): erasure burns the address, not the attractor (2026-09-25) — DONE

WHAT WE DID: full 5-arm battery (re-install vs fresh vs patch-controls,
400 steps each); root cause of the earlier shakedown = E044_SMOKE=1 env
leftover; new COS_MIN_NORM validity guard.

WHAT WE SAW (T018): re-learn is 2.08x SLOWER but the address re-grows
along its ORIGINAL direction (cos 0.760 vs fresh 0.278) — the attractor
survived erasure; the new route is new (atlas rho 0.21; L3H5 flips
carrier->ANTI-carrier, -2.03); the re-learned memory is ~3x more
surgical-RESISTANT (44.5% vs 0.13% under the same D2+patch). P3 failed at
bar (incumbents +0.135). JOHN improved (re-learning repaired J-class
collateral).

WHAT'S NEXT: C7 final: address/ability/expression/history. Night program:
e048 expression gap next.

## E047 — positive-claims replication sweep: card v3's gate (2026-09-24) — DONE

WHAT WE DID: 3 surviving positives × 5 nets (references near-bit-exact;
renorm liveness asserted; uniform-floor batteries).

WHAT WE SAW (T017 = card v3):
- **L5-calibrator SURVIVES 5/5** (KL 0.91-1.08, ablation ≤0.046) → first
  positive at H under the min-nets rule.
- **MLP-5 energy carrier SURVIVES 5/5** (zero/rotate 0.25-0.34, graceful
  α everywhere) → H.
- **Shared-L0 name machine DIES as stated** (2/5, seed-42 only) →
  distributional form: L0 BLOCK top-1 in 20/20 cells; sublayer allocation
  is a lineage lottery.

WHAT'S NEXT: card v3 declared (T017 preamble). Night program continues
(e048 expression gap next; e044 rerun in flight).

## E046 — C6 replication: two-factor erasure does NOT replicate (2026-09-24) — DONE

WHAT WE DID: the two-factor recipe on B43 + BDO (own-head and B's-recipe
cells, 4 total), full honesty battery (train + uniform-floor contexts,
J-census, incumbents, corpus CE).

WHAT WE SAW (T016): no Bar-2 anywhere — JULIET stays 13-17% after row-zero
+ top-head lesion on both nets; B's L3H5 transfers as predicted-NO. Row
surgery alone (the address half) replicates exactly (13-17% band at
~+0.001 CE, class-exact). C6 DEMOTED: general cheap DAMAGER; complete
erasure was B-specific luck.

WHAT'S NEXT: Review 6 (overdue) must weigh a replication sweep of the
card's positive claims vs new arcs — the pattern of
positive-claims-die/negative-claims-hold is now itself the biggest fact.

## E043 — install a name: ASYMMETRIC-CHEAP-REMOVE confirmed (2026-09-24) — DONE (audited from raw metrics)

WHAT WE DID: the registered install battery per scratch/e043_design.md —
rows-only arms (BDO same-init donor; wte/lm/both x copy/delta), exposure
ladder, L0-MLP block graft, direct-training ceiling; gates G0-G6.

WHAT WE SAW (T015):
- **No surgical install reaches Bar-I1 at the guard** (best: NLL 6.43 /
  acc 0.055 vs bar 4.17/0.5); Bar-I2 unreachable in every arm.
- **AMENDED by full report: anchored exposure installs CHEAPLY** — 7
  guarded cells reach Bar-I2; best 0.09 NLL / 0.974 acc at +0.05 CE,
  S_install 144-289 (the earlier decay read was a partial trajectory).
  New caveats: the EXPRESSION GAP (0 ZEPHYRA in 2,800 generated chars at
  97% battery acc) and PROTOCOL FRAGILITY (onset wall anchor-manufactured).
- **Shared machinery conserved** (L0H3 top-1, atlas 0.9997).
- Interpretation: address is concentrated (rows, removable); usage-ability
  is distributed (body, needs training). Edit asymmetry law (C7).

WHAT'S NEXT: e044 scar tissue tests the law's re-learning prediction.

## E012d — 4-net causal census: C1's strong form is dead (2026-09-24) — DONE

WHAT WE DID: the e018 causal-depth protocol on B43/R/R43 (B reproduced
exactly); cross-net histogram correlations + the lens=6 scoping check.

WHAT WE SAW (T014):
- **Causal depth is NOT cross-net invariant:** seed axis 0.735, regime
  axis 0.500 (B43-R43 collapse) vs the 0.8 bar; mode slides 3->4->4->5
  across B/B43/R/R43; renorm shifts mass deeper. The lens census's
  0.82-0.85 was the by-construction artifact.
- **T012's demotion total:** the lens=6 bin (52.7% of positions) selects
  causally indistinguishable positions.
- C1 final: qualitative mid-stack causal gate in every net; quantitative
  depth non-invariant. e005s readout re-scoped to the qualitative gate.

WHAT'S NEXT: card C1 updated. e043 still running; audit slot next when it
lands. Evening program continues (e044 scar gated).

## E042 — name-circuit atlas + two-factor erasure (2026-09-24) — DONE

WHAT WE DID: position-resolved lesion atlas (36 heads + 12 blocks) at name
positions for JULIET/JOHN/ROMEO/LUCIO; residual atlas under D2; two-factor
erasure cells. All 6 gates pass; e023 numbers reproduced exactly.

WHAT WE SAW (T013):
- **Two-factor erasure works:** D2 + L3H5@JULIET-prefix → acc 0.0013,
  NLL ≥ ln65, corpus +0.00083 nats, S_name 1,937. Complete selective
  forgetting achieved.
- **Shared name machinery:** L0H3 #1 head for ALL names; L0-MLP #1 block;
  JULIET~LUCIO atlas correlation 0.965. Collateral idiosyncrasy lives in
  row space, not circuits.
- **Dissociation:** post-D2 residual (13.6%, all at position 3) rides
  mid-network machinery (L3H5, L1-attn), NOT the healthy L0 circuit.

WHAT'S NEXT: C6 finalized. e043 (INSTALL a name) now cleanly defined:
rows + which body. Review ~15:50Z.

## E019 — LATE FOLD (R43 audit debt repair; original-era run) — mlp5 thermostat: ENERGY-CARRIER confirmed via the e011c-consistent rule, not as-written — DONE

Backfilled from runs/e019/metrics.json (run predates E020; fold
dropped in a quota-crunch period — found by the R43 audit).
Registered rule (verbatim): "CE(a=0.5) and CE(a=2) within +0.15
of baseline AND rotate(a=1) costs >2x zero-ablation ->
energy>direction". Verdicts: graceful-alpha WITHIN +0.15 at both
a=0.5 and a=2.0; rotate_over_zero = 0.25 (R1 as-written FAILS —
rotation does NOT cost >2x zero-ablation); R2 (e011c-consistent,
zero > 2x rotate) TRUE -> energy_carrier_confirmed (graceful AND
R2). Reading: MLP-5's fact carrier degrades gracefully under
magnitude scaling and is MORE damaged by zero-ablation than
rotation — energy-form per the e011c lineage, though the literal
registered inequality oriented the opposite way. No T-card
interpretation (window passed; number stands in the ledger).

## E018 — causal depth: the lens is UNCORRELATED with causal depth (2026-09-24) — DONE

WHAT WE DID: activation-patching causal depth over 1536 positions
(counterfactual last-position stream spliced at each depth; sanity gates:
self-patch exact, post-L5 patch flips 100%).

WHAT WE SAW (verdict (b), T012 written):
- **Spearman(causal, lens) = −0.009** — per-position UNCORRELATED. The
  lens's depth ordering carries no causal-decision information.
- Causal mode depth 3 (mean 2.95 vs lens 4.86); the lens's 53% "decided at
  L5" mass has no causal counterpart — L5 flips are RECALIBRATION
  (convergent with L5-the-calibrator from every other instrument).
- Genuine point-of-no-return exists mid-stack (monotone flip curve, 77%
  suffix-monotone); 15.4% distributed decisions; shallow patches → third
  tokens (73% at d0), deep patches → the counterfactual answer.

WHAT'S NEXT: C1 re-anchored (T012); e012d debt registered (causal census on
the other 3 nets — is CAUSAL depth the invariant?). e042 still building.

## E023 — surgical forgetting at entity granularity (2026-09-24) — DONE (confirmed by full report 15:12Z)

WHAT WE DID: the full registered battery per scratch/e023_design.md — D1/D2
granularity ladder, arms A/B/C, G0-G4 gates, frozen selectivity metrics.

WHAT WE SAW (T011 written):
- **The J-row scalpel: S_name = 573 (bar 5) at corpus cost +0.0008 nats**
  — the program's first selective instrument, ~4907× less collateral than
  entity-ascent at matched damage. Bar-2 erasure missed narrowly (acc
  13.6% > 10%): surgery damages near-completely, does not fully erase.
- D1 all-letters bomb confirmed (+0.38 CE). P2 refuted (collateral-vs-
  overlap ρ=0.61; idiosyncratic per-name collateral). P3 confirmed (no
  revive trigger; ascent catastrophic at the name bar: val +1.16, S_name
  1.06). lm_head-row zero: NLL 2.04/acc 0.83 (write-side partial).
- Card consequence: C6 drafted (entity knowledge in I/O row coordinates;
  damage-vs-erase boundary open).

WHAT'S NEXT: e018 (causal depth — instrument validation) + e042 (name-
circuit atlas: the 13.6% residual + collateral idiosyncrasy) dispatched.

## E041 — ΔW ceiling null: PARTIAL ANCHORING (card C3 debt paid) (2026-09-24) — DONE

WHAT WE DID: trained BDO = seed-42 init, different data order (corpus seed
7777 changes every batch; init bitwise-verified identical); computed the
ΔW-alignment ceiling cos(ΔW_B, ΔW_BDO) with the e029 protocol.

WHAT WE SAW:
- **The full ladder: 1.0 (same everything) → 0.534 (same init, diff data
  order = CEILING) → 0.141-0.152 (same init, diff regime, B↔R) → −0.002
  (diff init).** Verdict: PARTIAL ANCHORING — the regime change moves a net
  well beyond batch-order noise (0.15 is only 0.26× the ceiling), yet
  same-init anchoring remains far above the diff-init floor.
- Per-organ ceiling: early organs order-robust (L0-attn 0.73, L0-mlp 0.79),
  depth erodes alignment (L5 0.34-0.46) — deep layers are where both order
  noise AND regime pressure act.
- BDO val 1.595 (parity PASS; batch order alone shifts final CE by −0.03 —
  data order is a real training variable). Motion magnitudes identical
  (‖ΔW‖ ratio 0.98-1.01) — only directions differ.

WHAT'S NEXT: card C3 updated (ceiling paid). e023 design memo in progress.
Review ~14:35Z.

## E035 + E038 — task-net anatomy + causal head lesion (2026-09-24) — DONE

E035 (eval-only on the e021 task net):
- **Q1: allocation NOT reorganized; lesion maps are blind to task circuits.**
  Attn damage [3.03, 2.04, 0.64, 0.45, 0.24, 0.03] vs Shakespeare [2.40,
  1.74, 1.06, 0.38, 0.19, 0.03] — no L4 spike (COPY is ~5% of tokens, so
  the whole L4-attn block costs +0.24 corpus nats while ONE head inside it
  costs +1.67 at COPY positions). Position-resolved instruments are
  mandatory for task-elicited circuits. MLP-0 keystone even larger (4.90).
- **Q2: ONE net holds TWO stage profiles.** JS(filler, Shakespeare) = 0.010
  (filler pipeline ≈ Shakespeare's; L5 61.9%) vs JS(filler, COPY) = 0.272
  (COPY at L4, 88.3%). Profiles are selected per-position, not a global
  rewiring. Control net's filler census also Shakespeare-like.
- **Q3: init-anchoring task-independent — slightly STRONGER than the B↔R
  band** (W_in 0.622 vs 0.531; W_out 0.313 vs 0.261). Growing a retrieval
  circuit did not pull the net off the shared init trajectory.

E038 (causal lesion of retrieval head L4-H1):
- **Registered collapse verdict: NOT-CAUSAL (42% acc drop, bar >50%) — the
  dedicated-head reading dies; retrieval is a redundant cooperative fan.**
  Boundary reading: zeroing one head of 36 takes COPY CE 0.007→1.681
  (+1.674 = 51.3% of the distance to chance) and accuracy 99.96%→58.0%
  (15× chance; residual uniform across nonce positions) with PERFECT
  locality (filler CE −0.0006; whole-corpus +0.056). L4-H1 is causally the
  single largest retrieval channel — about half the nonce information —
  the other half in a distributed backup. E011b's cooperative-fan doctrine,
  now at the retrieval layer. Control head: nothing moves.

WHAT'S NEXT: card v2 updates (C4 language, C1 two-profile refinement, C3
task-independence) folded next edit. Review due ~14:35Z.

## E003c + E019 — dose-to-bar and the MLP-5 thermostat (2026-09-24) — DONE (record corrected per full report)

E003c (exact projection dosed to the 0.66-nat forgetting bar; 2 seeds +
step-norm-matched naive control verified at ratio 0.99):
- **DOWNGRADE FIRES.** r at the bar = 1.43/1.39 (needed ≥2.0); matched
  naive = 1.23 — projection's entire reproducible margin is 1.17×. r(dose)
  declines monotonically 1.52 → 1.41 → ~1.15 (the earlier 'flat ≈1.7'
  quick-read was wrong; apparent r>1.5 recovery at high dose is ratios of
  destroyed-model CEs).
- **THE KILLER: r vs train-B at the bar = 1.08 ≈ naive.** A and B trained
  memories are forgotten at IDENTICAL rates — zero content selectivity in
  the memorization channel; the val_B 'selectivity' was a measurement-axis
  artifact (held-out fluency text is more robust than any trained text).
- **e003b's r≈5-6 head-start DOES NOT REPRODUCE** (agent re-ran e003b's own
  code+seed: peak 1.455 vs recorded 6.128; s280 target matches bit-for-bit)
  — a chaotic trajectory event, not an Adam-anchor mechanism. The
  'accidental hybrid' story is dead too.

E019 (MLP-5 thermostat): **energy carrier CONFIRMED** — zero +0.60 vs
rotate60 +0.15 (4×); α=0.5 slightly IMPROVES CE (−0.0065 — MLP-5 marginally
over-writes); α=2 graceful (+0.12); zeroing spikes output entropy +0.62.
Magnitude keeps the distribution sharp; direction worth ~¼ of presence.

WHAT'S NEXT: claim 5 closes as 'first-order ascent cannot content-
selectively forget (r=1.08 memorization-symmetric)'. Next family: weight
surgery (e023 entity-level embedding+lm_head rows — the genome-era
inheritance) or second-order. Card C5 updated.
## E003b — targeted/projected ascent: registered instruments FAIL; an Adam-anchor accident soars (2026-09-24) — DONE (record corrected)

WHAT WE DID: 300-step ascent arms with corrected labels (target = train-A
memorization CE, baseline 1.018 vs val_B 1.681 — a 0.66-nat gap; collateral
= val_B): naive anchors, exact projection (unit B-direction, 4-batch mean,
refresh/10), top-10% masked, combined, plus an accidental variant
(unnormalized B-direction = 70%-strength projection, kept for the record).

WHAT WE SAW (registered verdicts):
- **Naive r = 1.22 (NOT 1.0)** — under corrected labels even plain ascent
  is mildly selective; the old anti-selectivity constant was partly the
  val_a mislabeling.
- **Exact projection FAILS the 1.5 bar (peak 1.455, decays to 1.09)** —
  removing the full first-order B-component converges to the naive
  signature. Masked fails 2.0 (1.80); combined 1.93 (gentlest: +0.034
  collateral at +0.064 target).
- **The ACCIDENTAL 70%-projection soars (r 3.12-6.13, target +0.28 at
  collateral +0.09).** Mechanism (agent's reading): through Adam's
  sign-like updates, the residual B-component acts as a weak implicit
  B-DESCENT anchor — projection-before-Adam ≠ projection-of-the-step.
  This is E002's explicit retain-anchor, rediscovered implicitly at the
  right dose.
- Mean-level gradient cosine cos(g_A, g_B) = 0.799 vs batch-level 0.345 —
  averaging collapses both onto the shared fluency direction.

WHAT'S NEXT: e003d REGISTERED — deliberate partial projection + explicit
small retain-descent term (the accidental winner made explicit), dose-to-
the-0.66-bar, step-norm-matched naive control, train-B second collateral.
NOTE: the running e003c uses the EXACT projection — interpret its
projected arm knowing it is the failing variant.
## E031 — stream-facing matrix grafts: W_in is the violent one (2026-09-24) — DONE

WHAT WE DID: host B received one matrix at a time from B43 (cross-seed,
same regime) at L3/L5: W_in, W_out, c_attn, c_proj + full-organ references.
Registered v2 predictions (v009-corrected stream-basis mechanism).

WHAT WE SAW (P1 CONFIRMED, P2 REFUTED):
- **P1 CONFIRMED: both stream-facing MLP matrices are violent.** L3: W_in
  +1.186 + W_out +0.795 ≈ full-mlp +1.907 (near-additive). L5: W_in +3.284
  — MORE violent alone than the whole organ (+1.46): donor W_out partially
  RESCUES donor W_in (the pair is internally coherent; the host punishes a
  foreign read more than a foreign read+matching-write).
- **P2 REFUTED:** c_attn is the mildest at both sites (L3 +0.397 vs c_proj
  +0.506; L5 +0.044 vs +0.068) — the weak-anchoring functional exception is
  the attention QUERY/KEY side, not c_proj. Attention portability is
  carried by both its matrices being mild.
- The seed-anchored object is confirmed as the STREAM-FACING interface, with
  the read half (W_in) dominant.

WHAT'S NEXT: T008 claim-3 mechanism now causally supported. e003b targeted
ascent remains the last open instrument (claim 5). Review 13:26Z.

## V009 — ΔW portability atlas: reads are seed-anchored, writes converge (2026-09-24) — DONE

WHAT WE DID (VISUALIZER agent, CPU-only): top-16 singular subspaces of every
organ's ΔW across the 4 nets; same-seed vs diff-seed subspace alignment
(random baseline 0.289); sanity vs e029 exact. runs/v009/dw_atlas.png.

WHAT WE SAW:
- **Candidate mechanism REFUTED in reverse:** the same/diff-seed alignment
  gap is largest on the READ side (W_in 0.260, c_attn 0.235) and smallest
  for MLP W_out (0.091). Diff-seed alignment ≈ random for all reads
  (excess ~0.003); W_out keeps a small positive excess (+0.036).
- **AMENDED after full report: the seed-anchored object is the residual-
  STREAM basis.** MLP stream-writes (W_out-LEFT gap +0.267) are 3.8x more
  init-anchored than attention's (c_proj-left +0.071) — matching e029's
  transplant rho. Diff-seed alignment is at the random floor EVERYWHERE
  (excess <= +0.04): attention portability = weak anchoring, not shared
  subspaces. e031 re-registered: W_in and W_out grafts each VIOLENT
  (stream-facing); c_proj mildest.

## E021 — task-swap: retrieval exists when required; new L4 decision mode (2026-09-24) — DONE

WHAT WE DID (T009 registered design, background agent): retrieval-required
corpus (10.7k docs, ID→COPY gap ≥37 chars) + shuffled-nonce control; two
fresh nets (val: task 1.432, control 1.552); copy accuracy, far-value at
COPY, depth census at COPY, attention ID-mass. runs/e021/*.png.

WHAT WE SAW (all four registered predictions resolved):
- **P1 CONFIRMED: 100% copy accuracy** (2500 held-out nonce chars; chance
  3.8%; control net 4.1%). CE at COPY positions 0.007 nats — noiseless.
- **P2 CONFIRMED: far-value at COPY +3.269 nats ≈ ln 26** (control
  −0.002). The full nonce information is retrieved from far context.
  **T008 claim 4 NARROWS: "no retrieval on natural char data at this
  scale" — not an architectural limit.**
- **P3 CONFIRMED: a dedicated retrieval head.** L4-H1 puts 95.1% of its
  attention mass on the 5 ID nonce chars (control same head 15.0%);
  layer-mean ID-mass peaks L4 (0.364 vs 0.061); local mass collapses
  (0.038/0.042 vs Shakespeare L5 0.106).
- **P4 = NEW-MODE:** 88.3% of COPY decisions at L4 vs 8.0% on Shakespeare
  (JS divergence 0.265). A sharply concentrated task-dependent decision
  mode — at L4, one layer EARLIER than Shakespeare's L5-centered profile:
  retrieval completes before final calibration. The stage picture gains a
  task-dependent member; stages remain the organism (claim 1 intact).

WHAT'S NEXT: e031 write/read-path split grafts (v009-flipped prediction);
e003b targeted ascent still queued. Review ~13:26Z gets this full ledger.

## E030 debt slot — claims 1+2 upgraded to H; e011c CIs clean (2026-09-24) — DONE

WHAT WE DID: one eval-only slot on existing checkpoints (background agent,
21.7s compute after setup): e012c 4-net depth-census table; e014b.1 R43
lesion-map replication; e011c bootstrap CIs.

WHAT WE SAW:
- **Claim 1 (stages) RESTORED at H:** cross-seed same-regime histogram
  correlation (0.849 mean) ≥ cross-regime same-seed (0.828); all six pairs
  in 0.822-0.855; invariants replicate in both seed-43 nets (L5-finalization
  1027/1088; depth-entropy Spearman; class ordering). Stages are
  init-independent AND regime-independent.
- **Claim 2 (plastic anatomy) REPLICATED on R43:** keystone dissolution
  (+0.21 vs B +4.08), late-heavy MLP flip (ρ +0.74), attn front-load, third
  independent rebuild of the declining write schedule.
- **e011c exceptions all real:** attn-L0 1.38±0.006, MLP-L1 3.03±0.044,
  MLP-L5 0.25±0.009 — 20-100× beyond sd.

WHAT'S NEXT: T009/e021 (retrieval-required task) REGISTERED and running in
background — P1-P4 break-conditions for T008 claims 1 and 4.

## E029 — seed×regime 2×2 transplant + ΔW alignment: mechanism CONFIRMED (2026-09-24) — DONE

WHAT WE DID: trained R43 (seed-43 renorm, parity PASS val 1.5596), then the
full 2×2 matrix (54 cells, 3 hosts × 6 organs, paired bootstrap CIs, C0
bitwise gates clean) + the ΔW-alignment observable: cos(ΔW_donor, ΔW_host),
ΔW = W_trained − W_init(seed). runs/e029/*.png.

WHAT WE SAW:
- **Mechanism CONFIRMED decisively: same-init pairs mean cos(ΔW) = +0.152;
  different-init pairs ≈ 0.000 (max |cos| = 0.017 across 24 pairs).**
  Training motion from different inits lives in almost perfectly ORTHOGONAL
  parameter subspaces — organs refine init-anchored directions.
- **Seed dominance is ORGAN-TYPE SPECIFIC:** MLP organs show strong seed
  dominance (ρ = dCE(seed)/dCE(regime) 2.0-3.6 at L3/L5 across all hosts;
  e028's violent cell replicates exactly: +1.990 vs +0.646); attention
  organs are axis-insensitive (ρ 0.54-1.29 — portable either way). The one
  regime-dominant organ: R-host MLP-L0 (keystone asymmetry pinned to the
  regime axis, R=7.95 vs 1.49).
- Both-axes changes are SUB-additive (median 0.49) — the two interference
  modes overlap.
- All-cell median ρ 1.007 (L0 cells saturate at the ablation ceiling;
  ratio-of-medians 2.48) — the registered "seed dominates everywhere"
  prediction refines to "MLP organs are seed-anchored; attention organs are
  portable."

WHAT'S NEXT: T008 claim 3 upgraded + refined. Open: why are attention organs
portable across inits while MLP organs are not? (candidate: attention reads
stream directions shared by all adequate solutions; MLP writes into
seed-specific subspaces.)

## E028 — cross-anatomy transplant: P3 REFUTED reversed — organs are portable; incompatibility follows SEED (2026-09-24) — DONE

WHAT WE DID: implemented scratch/e028_transplant_design.md (background agent):
MLP/attn organ swaps at L0/L2/L3/L5; hosts: baseline-B (seed 42) with
donors B43 (same anatomy, seed 43) and R (renorm anatomy, SAME seed 42);
C0 bitwise self-transplant gate (exactly 0.0 ✓); renorm liveness ✓; R =
ΔCE_transplant/ΔCE_ablation. runs/e028/*.png.

WHAT WE SAW:
- **P3 REFUTED in reverse: cross-anatomy swaps cost LESS than same-anatomy
  different-seed swaps** (median ρ = cross/within = 0.874; ρ≥2 in 0/8;
  cross ≤ within in 6/8; paired CI excludes 0 negatively in 8/8).
  L3-mlp: within +1.99 nats vs cross +0.65 (ρ=0.32). **Organ compatibility
  tracks initialization lineage (B and R share seed 42) more than training
  regime** — the two anatomies differ in scheduling/addresses, not organ
  mechanics. Supports T006/PL2 (stages) over PL3.
- **S1 keystone asymmetry confirmed both ways:** B's keystone MLP-0 → R host
  = worst interference anywhere (R=7.95); R's near-dead MLP-0 → B host ≈
  inert (+4.20 ≈ B's own ablation 4.08) — quietness transfers even when
  function doesn't.
- **S3 failed informatively:** trained foreign tissue misleads MORE than
  random tissue (lottery R=1.14 vs cross R=7.95 at L0-mlp) — interference
  is content-specific, not off-manifold energy.
- Prefix 0..3 cross strongly SUBadditive (+2.83 vs 8.92 summed) — host
  layers compensate for whole foreign prefixes.

WHAT'S NEXT: e029 — the clean 2×2: seed(42/43) × regime(base/renorm)
transplant matrix to isolate the compatibility axis (init lineage vs
regime); T006 P4 (v008 phylogeny) gains a new question: do lesion maps
cluster by seed or by regime?

## E013d — interference audit: both T007 stories dead; far context acts through bulk statistics (2026-09-24) — DONE

WHAT WE DID: divergent-continuation n-gram proximity for loser positions
(P1); shuffled-far context collapse test (P2). Same 2000 positions as
E013c.

WHAT WE SAW:
- **P1 REFUTED (effect +0.22σ < 0.5):** 91% of hurt positions have NO
  divergent repeat (≥4 chars) in far context at all — the
  repetition-interference story (H1) is dead in its simple form.
- **P2 REFUTED, informatively:** shuffled-far makes the hurt population
  WORSE (bottom decile −2.26 vs −1.68) while the gain tail survives nearly
  intact (+1.50 vs +1.60). Real far gains are shuffle-ROBUST (statistical,
  not informational); incoherent far text destabilizes MORE than real far
  text.
- **Net conclusion (T007 closed): this 2.7M char model shows no evidence of
  SPECIFIC far-context information retrieval — far context acts through
  bulk statistics (char mix / length) and can destabilize predictions when
  incoherent.** Consistent with E013's finding that L5's far attention is
  idle grazing.

WHAT'S NEXT: closed. If long-range retrieval is wanted, it must be tested
on a task that provably requires it (copy-span probes, e021 task-swap).

## E013c — far-value tail: far context is a double-edged sword (2026-09-24) — DONE

WHAT WE DID: per-position far-value = CE(16-ctx) − CE(256-ctx) over 2000
held-out positions; distribution, tails, correlation with local difficulty;
top/bottom context examples.

WHAT WE SAW:
- **Bimodal, not average-zero:** 30.6% of positions gain ≥+0.15 nats (top
  decile +1.60, p99 +2.96); 28.2% LOSE ≥0.15 (bottom decile −1.68).
  "16-token sufficiency" hid a tug-of-war.
- P1, P2 confirmed; P3 refuted (ρ=0.133 — far-value tracks the position,
  not its local difficulty).
- Top gainers = locally-ambiguous rare continuations resolved by far context
  ("the carp"→T, "ere "→s). Losers = far context actively misleading
  (candidate mechanism: interference from earlier similar n-grams with
  different continuations — T007/H1).

WHAT'S NEXT: T007 discriminators — divergent-continuation n-gram proximity
for losers; shuffled-far context collapse test. e028 running in background.

## E013 — context truncation: L5's calibration is LOCAL; far context is worth ~0 (2026-09-24) — DONE

WHAT WE DID: same 300 held-out windows at full-256 vs last-16 tokens; measured
KL(L5‖L4 readout), L4→L5 argmax-flip rate, and per-position CE.

WHAT WE SAW:
- **Registered prediction REFUTED:** KL 0.997 → 0.928 (−6.9%, predicted
  ≥50%). L5's distribution reshaping does NOT depend on far context; the
  census's diffuse far attention is idle grazing, not evidence gathering.
  "Re-globalization" is epiphenomenal attention shape.
- **16-token sufficiency (the bigger finding):** CE full-256 1.648 vs
  trunc-16 1.644 — far context adds ≈ NOTHING to next-char prediction on
  Shakespeare at this scale. Depth ≠ range: late decisions (T004) are deep
  lexical computation, not long-range integration; D2 is refuted.
- Flip rate 49.3% → 52.0% (unchanged): L5's argmax work is local too.
- BUG (found + fixed): double-softmax CE (probs fed to F.cross_entropy)
  inflated the first run's CE to 3.73; verified E012 unaffected (its CEs
  came from forward logits).
- BONUS (from the debug check): the depth-CE readout ladder is
  anti-informative mid-stack — [4.62, 4.70, **5.19**, 3.99, 3.45, 2.50,
  1.74]: depth-2 readouts are WORSE than unigram (4.17). Absolute mid-stream
  distributions are not lens-aligned (needs a tuned lens); argmax-stability
  claims are order-robust and unaffected.

WHAT'S NEXT: far-value TAIL distribution (per-position full−trunc ΔCE): is
the ≈0 average uniform, or do a few positions (after rare names?) carry all
the far-context value? e028 transplant running in background.

## E013a — attention census over 200 prompts (2026-09-24) — DONE

WHAT WE DID: all 36 heads' last-token attention across 200 held-out
prompts: local/far mass, attended-token surprisal, distant concentration.
Adjudicates T005 (Review 1 required this before any causal spend).

WHAT WE SAW:
- **The locality funnel replicates at scale:** far-mass U-shape
  [L0 0.80, L1 0.23, L2 0.09, L3 0.10, L4 0.23, L5 0.54]; local-mass peaks
  at L2 (0.42). L5 abandons the local window in 82.5% of prompts.
- **The rare-token story is DEAD:** attended surprisal flat (~5 bits) at
  every layer; ZERO heads with consistent distant-concentration. v002's
  'O'-head was a one-prompt artifact — Review 1's suspicion confirmed.
- Final T005 form: L5 = diffuse re-globalizer. e013 redesigned as context
  truncation (does L5's calibration KL depend on far context?).

WHAT'S NEXT: e013-redesigned (truncation, minutes); e028 transplant (design
memo ready at scratch/e028_transplant_design.md — T006 P3); e019 thermostat.

## E012b — renorm-anatomy census: the function persists (2026-09-24) — DONE

WHAT WE DID: reran the decision-depth census + angular-displacement profile
on the E014b renorm checkpoint (hooks active). T006 P1+P2 discriminators.

WHAT WE SAW:
- **P2 CONFIRMED (histogram corr 0.822):** two different anatomies, one
  functional profile — L5-finalization 1084/2000 vs 1082/2000, same L1 dip,
  same depth↔entropy Spearman (+0.344 vs +0.322). Mid-stack timing
  reshuffled (baseline spreads early; renorm concentrates L3-L4) but the
  pipeline shape held.
- **P1 near-miss (0.386 vs ≤0.33 threshold), direction strong:** block-0
  angular displacement 0.746→0.288; every layer's angular displacement
  roughly halved in the renorm anatomy.
- Adopted as lab doctrine: decision depth / locality / calibration KL are
  the primary anatomy instruments (invariants); lesion maps are the
  secondary "where do the stages live this time" instrument.

WHAT'S NEXT: e028 transplant (P3 — design memo in progress in background);
e013a census; e019 MLP-5 thermostat.

## V002 — attention atlas: L5 is a rare-token re-globalizer (2026-09-24) — DONE

WHAT WE DID: VISUALIZER subagent built lab/v002_attention_atlas.py: last-token
attention of all 36 heads on a 96-token dialogue window; per-head entropy,
distance profiles, attended-token surprisal; attention recomputation sanity-
checked against the model's own output (4.8e-7). runs/v002/*.png.

WHAT WE SAW:
- **Locality funnel across depth:** attention entropy L0 4.51 (near-uniform)
  → L3 1.64 (tightest; 50.7% mass at distance 4-16) → L4/L5 re-broaden.
  Matches E012's decision-depth structure (local completion peaks mid-stack).
- **L5 reads rare identity tokens far away:** local d1-3 mass collapses
  0.234→0.060 (L4→L5) while ancient d65+ jumps 0.001→0.091 (~76×);
  attended-token surprisal 4.28→4.81 bits at flat entropy. L5 head 1 puts
  0.499+0.146+0.085 ≈ 0.73 of its mass on exact 'O' character matches.
- This is the L5-calibrator mechanism: distant rare evidence reshapes the
  distribution tail (KL ~1 nat) while local argmax was already settled
  (ablation +0.03).

WHAT'S NEXT: causal test (e013): mask exactly the rare tokens L5 heads
attend to (positions recorded in runs/v002/metrics.json) — predict KL(L5‖L4)
collapses while mean CE barely moves. See THINKING T005.

## E014b — stream-renorm training: the decisive authority-schedule test (2026-09-24) — DONE

WHAT WE DID: implemented scratch/e014b_design.md exactly — renorm arm pins
every block-input stream to c=5.6 per token (hooks active train+eval), seed
42, same budget; baseline arm reuses E001 weights; eval-only-renorm control;
write/stream instrumentation; P2 criteria operationalized.

WHAT WE SAW (FINAL):
- **P2 REFUTED at full parity.** Renorm arm val 1.610 (BETTER than baseline
  1.622; gate 1.7224). Lesion map stayed front-loaded: attn damage
  [2.80, 1.69, 0.48, 0.81, 0.21, 0.03] (spread 2.77 vs baseline 2.37).
  Front-loading is functional allocation, NOT stream-geometry.
- **Anatomy is plastic:** baseline MLP-0 keystone (+4.08) dissolved (+0.10)
  under renorm; attn-L0 grew MORE critical (+2.80) with 2.9× larger write
  (7.8 vs 2.7); renorm MLP damage flipped late-heavy [0.10, 0.08, 0.23,
  0.45, 0.78, 0.70]. Multiple anatomies reach the same function.
- **Optimization declines late authority:** renorm caps the stream at
  consumption points, not write size — L4/L5 COULD write big but deflated
  (ρ 0.82, 0.70) while L1-L3 partially re-inflated (ρ 1.25-1.47; mean 1.10
  < 1.3 criterion → not met).
- **Norm profile is per-net load-bearing but task-optional:** eval-only
  renorm on baseline +3.56 nats, yet renorm-trained learning is unimpaired.

WHAT'S NEXT: T003 final resolution written (THINKING.md). Live questions:
why do early layers hold the coarse work? Why do late MLPs write big-but-
cheap (now LATE-heavy in renorm arm — the arrangement flipped)?

## V006 — decision-depth passage map (2026-09-24) — DONE

WHAT WE DID: colored 480 chars of held-out text by decision depth
(runs/v006/depth_passage.png); profiled depth by character class.

WHAT WE SAW:
- **Letters decided LATE (uppercase 5.32, lowercase 5.15); structural chars
  EARLY (newline 3.69, space 3.82); punct mid (4.52).** Depth tracks the
  TYPE of discrimination — structural/syntactic decisions finish early,
  lexical identity needs the deep pipeline (and L5's calibration).
- 51% of positions finalize only at the L5 readout — consistent with E012's
  L5-calibrator finding (it often settles the final argmax).
- Zero positions decided at emb in this passage; top-1 acc 0.61.

WHAT'S NEXT: depth clustering by position-in-line/speaker-turn; consider
depth as a manipulable surface (can we force a position to decide late/early
by context surgery?).

## E012 — decision-depth census (2026-09-24) — DONE

WHAT WE DID: over 2000 held-out positions, decoded the last-token stream
(ln_f+lm_head lens) at every depth; decision depth = shallowest depth whose
top-1 survives to the end. Correlated depth with final entropy (P1); split
positions early(≤2)/late(≥4) and measured per-position ΔCE under joint
attn-L4+L5 zero-ablation (P2); measured KL between L5 and L4 readout
distributions (P3). runs/e012/decision_depth.png.

WHAT WE SAW (2 confirmed, 1 refuted):
- **P2 CONFIRMED — decision depth predicts per-position vulnerability:**
  late-decided positions suffer 3.04× more ablation damage than early-decided
  (ΔCE 0.328 vs 0.108). Per-token anatomy is real; corpus-averaged lesion
  maps hide it.
- **P3 CONFIRMED strongly — L5 is a calibrator, not vestigial:** KL between
  L5 and L4 readout distributions averages 1.03 nats (median 0.70), yet
  zeroing L5 costs only +0.03 mean CE. L5 reshapes the output distribution
  massively in ways argmax and mean-CE cannot see. (Caveat: mid-network lens
  is heuristic.)
- **P1 REFUTED with sign flip:** Spearman(depth, entropy) = +0.32, not
  ≤ −0.4. Late decisions associate with HIGHER entropy — open contexts need
  deeper integration; constrained positions are decided at emb/L0. D1 was
  backwards.
- All 2000 positions eventually stable (no chronic wobble).

WHAT'S NEXT: viz — color a passage by per-token decision depth (v006): does
depth cluster structurally (names? line-ends? dialogue turns)? Think — if
L5 calibrates, what does it calibrate TOWARD (temperature? rare-token boost?
top-k shape?)? e014b design memo incoming from background agent.

## E011c — matched-perturbation control: geometry vs meaning (2026-09-24) — DONE

WHAT WE DID: replaced each attention/MLP write w with a 60°-rotated w′
(‖w′−w‖ = ‖w‖ exactly, verified 1.0000) — same perturbation energy as
zeroing, destroyed content. Three-rung ladder: zero vs rotate60 vs
same-norm-random (√2 energy). runs/e011c/geometry_ladder.png.

WHAT WE SAW:
- **P1 refuted both ways; energy is the dominant factor.** Damage ranks
  {zero ≈ rotate} < {random} across components — at matched energy, most
  blocks tolerate scrambled writes as well as or better than removal.
  The lesion map is mostly about HOW MUCH a block moves the stream
  (authority schedule), not what it says.
- **Exceptions:** MLP-L1 direction-sensitive (rotate ×2.95 zero); attn-L0
  mildly (×1.38); MLP-L5 is an ENERGY CARRIER (zero +0.59 vs rotate +0.14 —
  its magnitude matters, its direction barely).
- Caveat: single-run ratios 0.7–1.2 need bootstrap CIs; only the three
  exceptions look safely beyond noise.

WHAT'S NEXT: bootstrap CIs on rotate/zero ratios (cheap); e014b
stream-renorm training stays the decisive test of the authority schedule;
e012 decision-depth census queued.

## V001 — token journey visualization (2026-09-24) — DONE

WHAT WE DID: first artifact of the standing VISUALIZER thread: one forward
pass per prompt, three panels — logit lens through depth (final LN+head
applied to the last token's stream after emb and each block), PCA-2D token
trajectory with write arrows, per-layer angular authority (1−cos and
write/stream). runs/v001/token_journey.png.

WHAT WE SAW:
- **New observable: decision depth.** "…torches to burn " is decided at L3
  (top-1 `t`, p 0.69) and L5 HALVES its confidence (0.69→0.33); "To be, or
  not to " only surfaces the correct `b` at L4. Predictions form at
  different depths per token — T004 written with hypotheses D1-D3 and three
  registered predictions.
- Authority panel makes T003 visible: L0 write/stream ≈ 8.5 vs ≈ 1 later;
  angular displacement 0.7 (L0) vs 0.2-0.3 (later).

WHAT'S NEXT: e012 redesigned → measure decision depth over ~2000 val
positions; correlate with next-token entropy and L4/L5 ablation damage
(T004 discriminators). Viz polish backlog: arrowheads on trajectory, L0 bar
headroom in panel 3 (v001.1).

## E011b — L0 redundancy sweep + orthogonal-innovation control (2026-09-24) — DONE

WHAT WE DID: eval-only discriminators from the critique harvest: (1) all 63
L0 head-subset lesions; (2) same-norm random replacements of every attention/
MLP write (orthogonal-innovation control); (3) residual-stream norm profile at
block inputs. runs/e011b/redundancy_ortho.png.

WHAT WE SAW:
- **L0 heads are a cooperative ensemble with graceful degradation:** singles
  mean +0.05 nats (one slightly negative), yet all-6 = +2.40. Sum of singles
  0.317 vs joint 2.403 = 7.6× superadditivity. Keeping 1 of 6 heads still
  leaves 94% of full-ablation damage. Hook implementation validated exactly
  (all-6 head-zero 2.403 ≈ block-zero 2.401).
- **Same-norm random writes hurt more than zeroing** everywhere — attn L0
  +3.66 vs +2.40; MLP L1 0.82 vs 0.15 (5.4×). Ratios exceed the √2
  perturbation-scale prediction for MLPs and late attention → downstream is
  calibrated to write direction, not just magnitude.
- **Stream norm:** 0.67 at block 0 → 8.4× jump → plateau ~5.5. Write/stream
  ratio falls ~12× L0→L5.
- Emerging mechanism (see THINKING T003): the residual stream's norm growth
  may SCHEDULE each block's angular authority — front-loaded lesion maps
  could be partly architecture, not learning.

WHAT'S NEXT: e011c matched-perturbation control (60°-rotated writes) settles
geometry-vs-content per component; e014b stream-renorm training tests the
authority-schedule hypothesis directly. e003b (corrected ascent instruments)
still queued.

## E003 — forgetting selectivity frontier (2026-09-24) — DONE

WHAT WE DID: T002 discriminator suite: (1) gradient cosines (A↔B vs within-half
vs A↔French), (2) LR ascent sweep 1e-6…3e-5 with (ΔA,ΔB) trajectories, (3)
implant-French (400 steps, 5e-4) then ascend-on-French arm, (4) fluency-vs-
content CE probes (A-unique/B-unique lines vs generic) after mild ascent.
runs/e003/selectivity_frontier.png.

WHAT WE SAW (all three registered predictions resolved):
- **P1 REFUTED:** cos(A,B) = 0.345 ≈ within-half baselines (0.363/0.390);
  French is 2.4× lower (0.144). Same-corpus halves are NOT gradient-parallel.
- **P2 CONFIRMED:** no LR reaches ΔA ≥ 1 with ΔB ≤ 0.1. Even lr 1e-6 (the
  gentle walk along grad_A itself) gives ΔA +0.29 / ΔB +0.27 at step 200 —
  perfectly anti-selective.
- **P3 REFUTED:** unlearning French also destroyed Shakespeare
  (dissimilar_selective = false). Content distance does not rescue ascent.
- Fluency probes: after "mild" ascent (1e-5×200) everything ≈ 24 nats
  (random): A-unique 1.26→24.2, B-unique 1.35→24.0, generic 1.63→23.7.
- **Unified story forming:** first-order ascent — any dose, any content —
  destroys the shared fluency substrate first and content memories only with
  it. Gradient geometry (P1) shows content IS distinguishable in gradient
  space; the failure is that ascent trajectories do not follow it.

WHAT'S NEXT: T002 final resolution (thinking, not running): read the early
trajectory steps — was there ANY transient selectivity window (French rising
faster in the first 25 steps) before collapse? If yes: early-stopped ascent +
fluency-anchored objective is the repair hypothesis. If no: first-order
methods are structurally dead here; next family = weight-targeted surgery
(ascend only low-overlap weights) or second-order directions. Also reconcile
with the independent critique (scratch/critique_T001_T002.md) when it lands.

## E011a — write norms vs lesion damage (2026-09-24) — DONE

WHAT WE DID: measured mean residual write norms (per token) of every
attention/MLP block on val batches; compared to E001 lesion damage (T001
discriminator 1, zero training).

WHAT WE SAW:
- **H4 (write-norm confound) REFUTED.** Attention write norms are NOT
  monotone in depth ([2.72, 2.49, 3.31, 2.73, 2.39, 1.93] — layer 2 writes
  the most), yet damage still falls monotonically. Damage per unit write:
  attn [0.88, 0.70, 0.32, 0.14, 0.08, 0.02] — an 11× efficiency gradient.
  The front-loading is information architecture, not geometry.
- MLP write norms RISE with depth (1.82 → 5.64); MLP-5 writes the largest
  residual in the net yet ablation costs only +0.59 nats. Late MLPs write
  large, dispensable content — new open anomaly (for whom/what is it
  writing?). MLP-0 damage-per-write (0.95) is 5-10× any other MLP.
- Registered prediction "write norms will not decline monotonically" was
  CONFIRMED — first register-then-run success of the discipline.

WHAT'S NEXT: T001's remaining discriminators: mean-replace ablations and
LN-only recalibration (H2, off-manifold artifact) — e011 proper. And the new
anomaly: what does MLP-5 write? (logit-lens on its output direction space).

---

## 2026-09-24 — The pivot (context entry)

After three programs (neural genome transplants, LLM control surfaces,
HANDLE), we got too ambitious and drifted from the original spirit: da
Vinci-style dissection of neural networks for its own sake. Today the repo was
reset to a clean lab (tombstone commit `106aeff` preserves everything prior)
and the mission narrowed to the original one: small nets (1–10M params), many
experiments, curiosity first, graphs for everything, play.

The three questions opening the program: (1) what does a lesion map of a
freshly trained tiny transformer look like? (2) can we make a network forget
one part of its training data without wrecking the rest? (3) everything after
that is whatever the first two cuts turn up.

---

## E002 — forgetting pilot: naive vs anchored unlearning (2026-09-24) — DONE

WHAT WE DID: split the corpus positionally (A = first half of Shakespeare, B =
second half). From the E001 checkpoint, unlearned A with two arms: (1) naive
gradient ascent on A only; (2) anchored ascent (retain loss on B, weight 1.0).
AdamW lr 2e-5, 400 steps, batch 32×256. Tracked held-out CE on A and B;
generation probes before/after (runs/e002/probes.txt).

WHAT WE SAW:
- **Naive ascent is a bomb, not a scalpel.** ΔA +26.1, ΔB +25.9 nats — the
  entire model is destroyed (final CE ≈ random guessing over 65 chars).
- **Anti-selectivity:** to raise A by just +1 nat, B had ALREADY risen +1.51
  nats. Collateral damage runs ahead of the target damage.
- **The retain anchor only slows the destruction** (ΔA +6.4, ΔB +5.5 at equal
  steps; B damage at A+1 nat still +1.04). It does not create selectivity at
  this dose.
- Honesty reflex: the intervention changes behavior (massively) but with zero
  targeting value at this scale of step size.

WHAT'S NEXT: map the *selectivity frontier*: sweep ascent LR × steps × retain
weight; unlearn a small subset (one play) instead of half the corpus; try
Fisher/EWC-style parameter anchor instead of replay anchor; try weight-space
surgery (rank parameters by gradient overlap between A and B). → e003.

## E001 — first cut: lesion map (2026-09-24) — DONE

WHAT WE DID: trained a 2.74M-param char GPT (6 layers, 6 heads, 192 dim,
block 256) on Tiny Shakespeare (val loss 1.622 after ~2000 steps / 200 s; val
bottomed ~1.55 at step 1500 then overfit slightly). Then zeroed every
attention block, every MLP block, and each of the 36 heads individually;
measured deterministic val-loss delta per lesion (runs/e001/lesion_map.png).

WHAT WE SAW:
- **MLP-0 is the keystone organ:** zeroing it costs +4.08 nats — the single
  most damaging lesion, worse than any attention ablation.
- **Attention is strictly front-loaded:** L0 +2.40, L1 +1.74, L2 +1.06,
  L3 +0.38, L4 +0.19, **L5 +0.03 — the last attention layer is almost dead
  weight** in this model.
- **MLP damage grows with depth** after L0: L1 +0.15 → L4 +0.60, L5 +0.59.
- **16/48 components are ~dispensable** (Δ<0.02) — mostly late-layer heads.
- Net shape: early attention + layer-0 MLP carry the load; the top of the net
  is attention-light, MLP-heavy, and partly vestigial.

WHAT'S NEXT: what does MLP-0 actually store (char/unigram statistics? test by
probing / ablate-then-finetune recovery cost)? Why is attn-5 dispensable —
vestigial or quietly specialized (punctuation/newline)? Damage ≠ necessity:
how cheaply can the net re-learn around a lesion (zero-finetune recovery)?
Does this shape hold at 1M/10M/30M params (→ e004 ladder)?
