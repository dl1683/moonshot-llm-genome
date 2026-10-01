# Paper Skeleton 2 — the consolidation paper (STRATEGIST draft, 2026-09-28 ~09:18Z)

Working title: **"Consolidation follows error placement: routed and site-stored
memories in a tiny language model"**

Status: SKELETON. Claims 1-3 are evidence-complete (single-lineage caveat);
claim 4 carries W014's open caveat; the e147/e150 cells are marked and their
reading map is pre-registered (THINKING.md "READING MAP"). Blocked on:
replication seeds (e145-class), the width dose-response (e147), flat-CE
verdict (e150), dream-confound discharge (e148).

## Abstract (draft ~150 words)

We dissect memory consolidation in 0.84-2.7M-parameter char-LMs with
pre-registered interventions and deletion batteries. Four findings. (1)
A fact consolidates where its training error is placed — shown causally by
steering: error locked at positions 5-13 builds a site-store there, with no
routing despite sink adjacency. (2) Two memory types follow, switched by a BINARY CLIFF at zero-vs-any error-position variance — and the types are PHASES of one substrate, BOUNDED per R50 (e176N RESOLVED — NEUTRAL-DISSOLVES): under the install's own name-deleted windows (an extinction-grade stream), no memory state we tested retains expression (the consolidated fact dissolves; CE was shocked at the dissolution moment); e176N RESOLVED [n=3 seeds via e184; n=2 families via e157]: NEUTRAL-DISSOLVES — the fact dies on the neutral stream too (both streams, two-step clock; lr scales the rate not the outcome); e183 RESOLVED: STILL-DISSOLVES — the filtered stream (background removed) kills on the same clock; the lead finding EVIDENCED WITHIN THE REGISTERED GRID (R52: six cells + one n=3 column; e185b RESOLVED: no type survives neutral streams (the cross lands at +100; the dwell collapses at +1 — faster than extinction) — the grid CLOSED; the noise discriminator (e185) RESOLVED: no robustness basin): no memory state tested retains expression under continued training on any stream composition run, at every lr tested above ~1e-5 within its horizon (survival t* ~ lr^-1.1..-1.4; the 1e-5 cell right-censored at +300 with the fact alive), with the fact's windows absent (sparse union, not a cross: 3 streams on the root; 3 seeds on neutral/root; the second lr on extinction/root; the dwell and site types on their own single streams — R52 axis audit; n=2 families (e157: the wash replicates; the phase structure is lineage-1-scoped)); the mechanism per e185+e180 (+g3K, PROPOSED n=1): no robustness basin against LEARNED displacement (the trajectory hypothesis — static random displacement is 4-10x more forgivable, kappa ~5-6 at matched per-coordinate RMS; every e185 arm was a trajectory; replication owed per C13-1),  (~2.5-5 L2 over 2.7M params; per-coordinate RMS ~1.5e-3) — content-free noise at displacement-match exits it identically; survival is a rate law (t* ~ lr^-1.1..-1.4 (grid-legal band; -1.16 stored); at 1e-5 the fact lives); the corpus's addition is surgicality, not the exit [e185 n=1; e180's gentle regime unreplicated] [e177: the site-store also washes, with a 4-24x decay gradient; e175: no savings at threshold — the archive empty] [was: memories lie on a GRADIENT-RESISTANCE axis — unconsolidated (dwell-phase) memories wash out under ANY continued training (e161: plain corpus dissolves the whole fact in <50 steps); variance training builds geometry-general access (the cliff survives); the 'closing' direction was substantially forgetting — confirmed by the root-freeze (e176): BOTH memory types wash without rehearsal; e177 tests the one candidate archive] (e151: one locked re-teach converts sink-coupled to site-stored, g-12 0.916->0.102, at improved CE) (e147: ±1 suffices — address key +0.327→−0.029, novel-geometry expression 0.071→0.696, no width trend): SITE-STORED
(content concentrated at a row, context-general, geometry-bound) and ROUTED
(readout keyed to the omnipresent row's presence, geometry-general,
deletion-tolerant) — switched by the error's position-variance. (3) Content
never moves: all states store in body organs and fact-specific heads;
"migration" is read-policy re-routing, the destination row carrying no
written key (install-restore is a no-op; direction-scramble spares the fact
while costing the LM 0.70 nats). (4) sink-coupling and the
removable-to-irremovable movement (e150): no flat-CE fact-kill exists —
masking all attention to position 0 spares the fact at CE +0.03 while
sub-threshold row-0 norm poisons every read (threshold in (0.07, 0.15));
consolidation moves the
    memory's dependence into coupling with the sink (e159's double
    dissociation; RESOLVED MIXED by e162: healthy content fully rescues; total-dose absorption on a flattened profile kills (the per-layer distribution untested, e167 queued) — functional dependence on the sink's dual role, condition-qualified: equal organism damage, only the read-coupled memory dies;
    the mask heals a poisoned net completely). L0H3-zero (58.6% drop at CE +0.21)
COMPLETED by e125a: an ASYMMETRY OF EXISTENCE — the sink-coupled
    (generalizing) memory dies 70.7-97.4% at CE 0.245-0.280 (N2, both modes) via a superadditive
    complementary circuit; the site-stored (locked-in) memory has NO kill
    set at ANY CE (92 cells, two sites, both modes; disjoint fact-head
    populations; saturating redundant ladder) — the memory that
    generalizes is the memory you can remove. Preceded by e160: — {L1H0,L0H0} (no 'fact-specific' head needed) kills the fact at CE +0.25 in both ablation modes, superadditively, while SPARING the site-stored fact under the same coordinates — circuit-selective head surgery. Fig 2's killer point; the unlearning ordering (heads > route >> band) is demonstrated. All corrections in the arc
were caught by pre-registered adversarial review and are reported.

## Introduction (draft, ~11:10Z; provisional clauses marked)

How does a memory become permanent? In complementary-learning-systems
accounts, replay moves memories from a fast, specific store into a slow,
distributed one — but the mechanism of the move, and what "distributed"
means mechanistically, are usually asserted rather than dissected. We
dissect it in char-LMs of 0.84-2.7M parameters with pre-registered
interventions, deletion batteries, and adversarial review, and report a
story that inverts several intuitions.

The lab's organisms are born with ONE memory organ: the omnipresent
row (the attention-sink coordinate) carries every naturally-placed
association from the first exposure (13/13 install checkpoints, five
seeds of the fresh family at rel 1.000) [LICENSED by e163: the dial's removal face discriminates carriage (arm_b reads 0.275 vs the census's 0.878-1.000); the perturbation face saturates — the two-face footnote]. The "addresses" an earlier
arc of this lab dissected — positional keys with mass laws and family
typing — are PROTOCOL-MADE GRAFTS: they form when a masked-replay
protocol pins a fact's position, and do not form under natural
placement (row 129 at essentially null). What we had called migration
was share-redistribution around a constant sink-carried core.

Against that backdrop, three results. (1) A CAUSAL COMPASS: a fact
consolidates where its training error is placed — shown by steering
(error locked at positions 5-13 builds a site-store there, with no
read-coupling despite sink adjacency; proximity piggybacking dead).
(2) A CLIFF, NOT A DOSE: memory TYPE is decided by a binary switch at
zero-vs-any error-position variance (±1 suffices; no width trend);
the types are phases of one substrate, with conversions demonstrated in
both directions across the lineage (same-net reversibility is e155's
queued cell) — and conversion completes at ALL seeds within 300 steps
(e152R, n=3) with seed-dependent trajectories (a washout race); a
mid-conversion state holding both natures was OBSERVED (one seed: site-store
66.8x the 2x-control bar, geometry retention 0.56) but is not a timescale
law; the brake overshoot replicates 3/3 (e152R) [e158 COMMITTED PASS-2: TEXTURE — closure requires the CONJUNCTION novelty x zero-variance (jitter@novel OPEN 0.789, locked@home MID 0.458 straddling, locked@novel SHUT 0.102); the door's closure ACCOMPANIES novel-site teaching — graft-formation per se is not the closer (a home graft formed with the door open); mechanism: graft formation is NOT SUFFICIENT for closure (a home graft formed with the door open — n=1, the straddling cell); novelty is a candidate variable (e165); carriage LOCATED by e173: the MLP+LN class carries the closure — restoring it alone reopens the door to 84% of ceiling (CE +0.07, graft intact; graded: a mid-late L2-L4 band) — [e166's row-null was an instrument tautology, withdrawn; e173's ladder licenses it]; e154 landed TEXTURE with a confound: F1 ANNIHILATED under a protocol whose anchors contradict it — e170 RESOLVED: OVERWRITE-REAL — F1 annihilates identically under neutral anchors (the contradiction channel removed by construction); capacity ~one fact wide at this budget; 'globally' licensed as 'any second-fact install demolishes the first'; e179 prices the maintenance budget: NINE replay events per 300 wash steps suffice (whole anatomy intact); one replay RESURRECTS the killed fact with 18 steps of stickiness; non-monotone in r (a sawtooth, not a dial) [single seed] — rehearsal does not prevent forgetting; it makes it irrelevant; the dose-2 kill is over-determined (the wash channel alone suffices per e176N); a transient eviction hole at t~4-8 exists even under rehearsal; no capacity formula [the 1/rehearsal-fraction formula struck per R51]]. (3) SPLIT CUSTODY: the converted memory's
DEPENDENCE is READ-coupled to the sink (it dies of what attention
reads off a degraded pivot — the double dissociation: equal organism
damage, only readers die) while its READOUT consolidates into a small
head-set that surgery can remove (type-selectively) — and what
training built, surgery cannot create (transplant-rigid when open).

Every inversion in this arc was caught by the lab's own adversarial
machinery, pre-registered before the discriminating data existed, and
is reported: four headline verdicts fell in one session, each caught
within hours. We offer the correction chain as part of the method.

## Contributions (numbered)

C1. Error-placement compass (T076/T084; e120/e131/e143) — observational then
    causal. Key exhibit: the "failed" splice arms expressed at 0.989/0.988 at
    the address the battery never read (instrument-geometry blindness, Rule 12).
C2. The type taxonomy + its switch (T082/T085; e139/e142/e143/[e147 pending]).
    Key exhibit: the ROAD→TYPE plate (fig 1). History rewrite: row-0-always,
    addresses are protocol-made grafts (e142, 13/13 nets).
C3. Content-everywhere/routes-differ (T080/T081; e133/e141). Additivity fails
    at the organ level (0.785 vs 1.98) — populations at rows/organs/routes.
    Presence-not-content routing; probe-power honesty (no-op-by-norm lesson).
C4. The memory tenant + unlearning ordering (W014/[e150 pending]; e125 design
    pre-registered: heads > route >> band, brake-trap as the naive failure).
C5. Methodology: the correction chain itself (three headline verdicts inverted
    in one session; every inversion caught by the lab's adversarial-review
    machinery; pre-registrations git-verified) — reproducible honesty.

## Results section outline (sections -> runs; assembled ~11:58Z)

R1 The compass is causal (e120/e131/e143): instrument-blindness
   exhibit (0.989/0.988 at 183) -> the steering NEAR/FAR/JITTER
   plate -> the committed-prediction record.
R2 The cliff and the phases (e147/e151/e152/e158*): A(w)/NR(w)
   twin panels; the conversion before/after; the dwell trace
   (with mask-column overlay); the 2x2 [e158 DONE: TEXTURE-pass-2 — the conjunction].
   R2b THE WASH LAW'S CURRENT FORM (2026-09-30 amendments, gaps
   10-12): no state survives continued training under AdamW at every
   lr tested (the optimizer clause — opt1: the two-step clock is
   Adam's sign-normalization, 1683x/step at matched lr, and the
   normalizer flips the sign of fact-relevance, -0.0385 vs +0.0981);
   the small-displacement pump strengthens the fact under every
   optimizer; the kill is priced in RAW displacement (e188 DONE: RAW-WINS —
   D at death CV 0.6% across seeds at matched t*; the aligned
   currency lost, CV 0.56; alignment a passenger that flips
   positive post-kill); opt1b/opt1c adjudicate the gate's
   trajectory-class scope; the static/learned contrast (g3K kappas,
   n=1) keeps the replication debt; task-arithmetic vocabulary
   REJECTED (install-vs-wash cos in [-0.034,-0.018] on four
   organisms).
R3 Content everywhere, access differs (e133/e141/e142): the
   three-net maps; install-restore/perm/halfnorm riders; the
   origin census (13/13) [e163 pending for the dial license];
   the POST-KILL CENSUS (e164): behind the N2-killed readout,
   storage intact and organized — access severed, substance
   survives; the MLP plane answers the site-fact's remainder
   (any-CE kill at organism prices only).
R4 Split custody (e150/e159/e160/e162/e125a (DONE)): the flat-CE
   plane with the killer point; READ-vs-MASS cells [*]; the two
   knives' planes side by side [* e125a]; the double
   dissociation panel.
R5 The correction chain (methodology): the timeline figure.

## Figure plan

Fig 1 (THE plate): rows = {NEAR locked, FAR locked, jitter ±8, [ladder w=1..64
from e147]}, columns = {site content census, novel-geometry generalization,
D-all survival, brake sign}. e143's numbers fill the core 3x4 (brake column
CORRECTED per R46 audit — was transposed):
NEAR 0.278/0.002/0.003/~0; FAR 0.238/0.205/0.156/-0.233; JITTER 0.722/0.914/0.903/brake(-0.132).
Fig 2: the flat-CE plane (fact-drop vs CE-cost scatter, every intervention,
flat-CE region shaded) — from e150; the paper's honesty centerpiece.
Fig 3: the correction chain timeline (verdict → attack → discriminator →
inversion), the C5 exhibit.

## Evidence gaps (submission blockers)

1. Single lineage everywhere — need ≥3 seeds/families for C1-C2 (e145-class:
   the e098 ladder + B43 exist on disk).
2. Width dose-response for C2's switch (e147, in flight; INVARIANCE-CAUSAL
   vs SEED-COVERAGE changes the claim from binary to parametric or adds the
   seed-reach constant).
3. Flat-CE verdict for C4 (e150; ALL-KILLS-WRECK triggers the
   removable→irremovable reframe — abstract claim 4 rewrites, not retreats).
4. Dream claim stays OUT of the paper until e148 discharges the harvest
   confound.
5. GPT-2 external-validity probe (parked; now has a sharp first question:
   row-0 presence-keying under the e141/e150 instruments).

## Risks (reviewer-kill, pre-empted)

R1 "Toy scale" — C5's methodology + the GPT-2 probe as external validity.
R2 DISCHARGED by e151 (one lineage, both phases, bidirectional). NEW R2:
  "was the closure global or self-conversion?" — e154's two-facts cell decides;
  run before the abstract's "globally" survives.
R3 "Known phenomenon" (attention sinks; memory types) — the novelty is the
  CAUSAL compass + presence-typed routing + the failure-mode inversion, not
  the sink's existence. Position against sink/StreamLLM and
  complementary-learning-systems literature (scratch/massaction_key_lit.md,
  W003's CLS analogy, now narrowed to replay-only; full claim-by-claim
  mapping: scratch/lit_beat_20260930.md).
R3b (T137, the most dangerous overlap): the trajectory hypothesis's nearest
  prior is THEORY — Evron COLT'22 + Goldfarb & Hand AISTATS'23 state
  that forgetting follows task/gradient geometry, not displacement
  magnitude. MUST cite and lead with the controlled 3-way dissociation
  (the theory predicted; nobody ran the static control).
R3c: the wall reads as hard-constraint methods (Wolczyk ICML'22,
  Elsayed RLC'24) without minimality (one commit + one scalar ball,
  zero old-task statistics) + the survival assay foregrounded.
R3d: "wash = negative task vector" (Ilharco ICLR'23) — pre-empted by
  reporting the install-vs-wash cosine (e188 co-read) and adopting
  the vocabulary if it fits.
FULL-TEXT RE-CHECKS before submission: SFAO (OpenReview Feb 2026),
  Elsayed RLC 2024.
R4 "Instrument circularity" — Rule 12 + the probe-power amendment are the
  pre-emptive answers; lead with them.

## Cut-list for the 8-page form (R49 ideator, adopted ~14:15Z)

CUT: the self/identity arc entirely (e146/e146b/e156, W004/W015 — next
paper); dreams (already out, mechanism sentence too); W017's coding story
beyond one operative sentence; the e165 novelty ladder (one sentence:
"novelty may itself be binary — untested axis"); the P-A anchor physics
compressed to a background paragraph (the other paper); e149's brake
decomposition (keep T094's overshoot as the one poetic sentence); R4's
kill-set fine structure (two headline numbers, two panels). COMPRESS: R5
correction chain to a half-page box + the timeline figure. FINISH LINE:
one reversibility exhibit (e155R or e172); one of e171/e174 per e170's
branch; GPT-2 probe or an explicit scope sentence; e147R run-or-flag.


## g-series integration amendment (coordinator, 2026-09-29 ~21:10Z)

The generative turn's three law-grade positives enter the paper as the
arc's payoff section. Every dissected law above says memory dies; the
g-series says the death is an ENGINEERING TARGET — each claim was
designed FROM a dissected law, pre-registered, then replicated to the
lab's n>=3 standard.

DISCUSSION-MECH NOTE (e194, for the mechanism paragraph): the
   lethal subspace flees with the state; a re-computed front
   pursues it (one recomputation = the whole 23% bonus; k-ladder a
   step function) while a re-orienting walk rotates away and
   spares — dynamics cut both ways, measured in both directions.
R6 THE GENERATIVE TURN: memory made architectural (g1/g1b/g1bR, g2/g2d/
   g2e[/g2f in flight], g3/g3R): (a) THE WALL — commit-and-project L2
   ball (zero new params) holds the consolidated fact at ~0.9 through
   the +300 wash that kills the control in 2 steps, n=3 seeds, tight
   band (mins 0.746/0.777/0.803), at +0.53 nats organism tax; (b) THE
   RHYTHM (g2g-controlled) — a zero-parameter rehearsal organ (cue
   pool + onset monitor + replay gate) self-times resurrection
   events: threat-responsive within [0.5x, 2x] (a weak step,
   ceiling-saturated beyond), the organ's advantage over a matched fixed schedule +0.07 of cycle-median at n=3 seeds (median +0.033, cleared by 11%) — DECOMPOSED by the paired control: majority replay-batch composition (+0.056, the cue-pool selector), minority timing (+0.013, the thermostat)
   over a matched fixed schedule at operating threat (run-stable;
   FIXED-MATCHES-OR-WINS the safer letter beyond), refractory-
   tunable with shorter better (the 20-45 band partly a
   construction floor, disclosed); timing root-robust 3/3,
   amplitude root-draw-bound (T136; the seed ladder licensed); (c) THE CONE (licensed: n=3 wash-draw seeds, one organ) + THE
   TRAJECTORY HYPOTHESIS (PROPOSED — g3K n=1 per organism, kappa
   intervals overlap [2.8,6.6] vs [4.0,9.7]; replication owed: e188
   + two more organisms; supervisor C13-1): killing is
   DIRECTION-typed (the wash direction and its 45-degree tilts kill at
   ~2x displacement; g3R n=3 wash-draw seeds, one organ) AND
   TRAJECTORY-typed (g3K: static random displacement is 4-10x more
   forgivable at the organism — kappa_store 5.0, kappa_host 6.0 at
   matched mean per-coordinate RMS; the earlier 24-32x was the
   STORE-ISOLATED leg, so the wide cone is an ORGAN property; no
   memory state survives continued TRAINING on any learned path —
   corpus or noise-label — while static jumps show graded basins).
   Effect sizes stated ONLY as organism-level threshold ratios; the
   store-isolated organ reading cited as the design's property, never
   as the organism's.
   Framing sentence (REWRITTEN AFTER E188 — the aligned-training
   slogan died by measurement; RAW-WINS): the wash kill is a
   DISPLACEMENT THRESHOLD under training — raw D at death is the
   invariant (0.6% across seeds at matched t*) — while the
   static/learned contrast governs how fast the threshold is
   reached (learned paths at 1x, static jumps 4-10x; alignment a
   passenger, seed-lottery at death); the front can be walled (g1,
   battery-channel; sequential: A held
   through an active second install, dose-adequate contrast owed),
   detected (g2, with the
   R56 construction disclosures), and its anisotropy measured (g3).

Fig 5 (THE terrain figure — LICENSED at n=3 organisms / 2 facts / both replication axes (the ORDER draw-fact-architecture-robust; absolute kill-Ds = fact-strength biography, windows in the caption):
   e192 primary + e193 replicate; the order g < sign < random and
   the re-orientation rider replicate; the pump ridge does NOT
   cross lineages — one panel, two organisms): the
   fact-vs-displacement overlay, dual-currency
   — the g-ray profile (pump ridge, cliff 0.92), the STATIC sign
   ray (kill 2.5; A0's step-1 read on the curve), three Gaussian
   rays (alive/flat at 4.0), and the two walks at matched D (the
   pinned ray dead at the static edge, the re-orienting bleed
   alive): forgetting's terrain, and the one path that dances on
   it. Runs from runs/e192/e192_fig5_terrain.png.
Fig 4 (THE generative plate, 3 panels): (i) wall: fact p(Z) vs wash
   step, W1 flat ~0.9 vs C dead by +50, three seeds shaded; (ii) rhythm:
   the sawtooth trace with self-timed events marked, three seeds'
   waveform overlay; (iii) cone/trajectory: the
   kappa pair plot (organ-level store-isolated vs organism-level,
   wash vs isotropic rung ladders from g3K, with g3R's n=3 wash legs)
   — the trajectory-vs-static dissociation panel.

C6 (contribution, after C5): "Dissection-to-design closure: each
   architectural positive was designed from a law dissected in R2-R4,
   pre-registered, and replicated — the compass (R1) predicted WHERE to
   wall, the no-basin/rate-law results (R2) predicted WHAT to detect,
   and the direction-vs-energy split (g3R) predicted the basin's shape."

Abstract clause (4), draft (R56+g3K-corrected): "(4) The same laws
   are generative: a projection ball (channel-scoped protection at a
   standing tax; sequential: A held through an active second
   install, dose-adequate contrast owed), a self-timed
   rehearsal organ (timing replicates across seeds and roots;
   amplitude root-draw-bound), and a displacement-threshold
   forgetting law (raw displacement at death is the invariant,
   0.6% across seeds; learned paths reach the gate at 1x where
   static jumps need 4-10x; alignment demoted to passenger by the
   death-currency measurement) each convert a dissected failure law
   into an
   architectural positive, replicated at n>=3 on single roots/organs
   — memory in these nets is not fragile by necessity but by
   default."

Evidence gaps (additions):
6. Rhythm amplitude is root-draw-bound (T132) — g2f (base redraw,
   in flight) either licenses n=2 roots or scopes the claim; the paper
   ships the honest decomposition either way (clock = organ, floor =
   root).
7. Cone/trajectory is organ-draw n=1 (g3R+g3K's split robustness is
   over wash-draw seeds on ONE reused organ) — one organ redraw (g3O
   queued, now carrying g3K's organism-level ruler); the R56+g3K
   corrections are ADOPTED (organism threshold ratios; the 24-32x
   store-isolated reading cited only as the organ's design property).
8. g1bW DONE (T140 + R58-critic 2c): MUSEUM fired as registered,
   its rescope WITHHELD (the paired reference failed the same ruler —
   B installs nowhere at this dose; the museum question is OPEN
   pending g1bW2's dose LADDER). LEAD WITH THE ONSET-TAX — the one
   contrast-licensed read: the install's partial form halved inside
   the ball (B g0 peak 0.21 vs 0.53, bit-identical inputs). A held
   (min 0.65) through the 300-step attempt — real but a weak
   antagonist (it installed nothing); the paper says "the wall held
   A through a second-install attempt; the dose-adequate contrast
   owed (g1bW2)".
9. Rhythm's controls (g2g, READY): threat-level ladder (self-timed vs
   thermostat), refractory-widened band, and the REGISTERED fixed-period
   head-to-head — without them "self-timed" stays scoped to one threat
   level and the fixed-arm co-read (0.693 vs 0.587-0.615) is disclosed
   in R6(b).
12. Optimizer clause EVIDENCED, decomposition PROPOSED (opt1 DONE;
   R58): the two-step clock is Adam's sign-normalization (1683x/step
   at matched lr; the normalizer ATTENUATES the stream's
   fact-relevance ~2.5x at a matched point; the trajectory-level negative
   read is the post-step view — the chart cell's estimator correction); the kill gate reads
   displacement in a two-convention bracket (checkpoint 2.49 vs 3.64;
   interpolated 2.18 vs 2.85) — the small-displacement PUMP
   strengthens the fact under every optimizer (SGD lingers at
   1/1683rd speed); paragraph (2)'s "at every lr tested" becomes
   "under AdamW at every lr tested"; opt1b/opt1c/e188 adjudicate the
   gate's trajectory-class scope and the death currency.
11. Static/learned replication (C13-1, narrowed by e188): the
   integral layer is DEAD (RAW-WINS); the owed replication is the
   kappa contrast alone — >=2 more organisms (kappa pairs) before
   the static-vs-learned wording is law-graded.
13. GPT-2 clause (e182c DONE, phase 1): T123's erosion is GENERIC
   forgetting at 124M (controls erode with the fact, ratio 0.92,
   while ppl improves) — every GPT-2 sentence reads "ordinary
   forgetting with improving perplexity", never a no-basin
   signature; the template-locus hint and phase 2 (fresh corpus
   draws) noted in the discussion.
10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-
   STATIC clause — "no memory state survives continued TRAINING"
   (the e185 arms were displacement-matched trajectories, corpus and
   noise-label alike); STATIC random displacement shows a graded
   4-10x basin (kappa_host ~6, kappa_store ~5 at matched mean
   per-coordinate RMS). Never state the no-basin law against random
   displacement.

Risks (addition):
R5 "Circularity — the architectures fix a problem the paper itself
   shows exists" — pre-empted: the g-series claims are not post-hoc
   engineering; each was pre-registered with frozen bars BEFORE compute
   (git-verified), and two falsifiers fired honestly (g5's W_q
   falsifier; g2's GATE-SILENT) — the design loop is itself evidence
   the dissected laws are causal, not descriptive.

Cut-list interaction: R6 gets 0.75 pages; pay for it by compressing
R3's rider list (the maps' numbers to a table) and R5's box (already
half-page). g5/g4 (the two honestly-scoped cells) enter as one
sentence each in R6's closing ("two scoped negatives: the compass is
positional at the A-floor; single-site walls fail two-site fragility").
