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
routing despite sink adjacency. (2) Two memory types follow, switched by a BINARY CLIFF at zero-vs-any error-position variance — and the types are PHASES of one substrate, BOUNDED per R50 (e176N running — the neutral-stream control): under the install's own name-deleted windows (an extinction-grade stream), no memory state we tested retains expression (the consolidated fact dissolves; CE was shocked at the dissolution moment); e176N RESOLVED [n=1, one lineage — seeds owed]: NEUTRAL-DISSOLVES — the fact dies on the neutral stream too (both streams, two-step clock; lr scales the rate not the outcome); e183 RESOLVED: STILL-DISSOLVES — the filtered stream (background removed) kills on the same clock; the lead finding FULLY EVIDENCED (e184: n=3 seeds, all dissolving in the same bracket): no memory state tested retains expression under continued training on any stream composition run, at any lr tested, with the fact's windows absent (sparse union, not a cross: 3 streams on the root; 3 seeds on neutral/root; the second lr on extinction/root; the dwell and site types on their own single streams — R52 axis audit; bounded by one lineage); the mechanism candidate is bare corpus gradient flow [e177: the site-store also washes, with a 4-24x decay gradient; e175: no savings at threshold — the archive empty] [was: memories lie on a GRADIENT-RESISTANCE axis — unconsolidated (dwell-phase) memories wash out under ANY continued training (e161: plain corpus dissolves the whole fact in <50 steps); variance training builds geometry-general access (the cliff survives); the 'closing' direction was substantially forgetting — confirmed by the root-freeze (e176): BOTH memory types wash without rehearsal; e177 tests the one candidate archive] (e151: one locked re-teach converts sink-coupled to site-stored, g-12 0.916->0.102, at improved CE) (e147: ±1 suffices — address key +0.327→−0.029, novel-geometry expression 0.071→0.696, no width trend): SITE-STORED
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
seeds of the fresh family at rel 1.000) [PROVISIONAL per R47: sits on
the trained-geometry dial T083 declared saturating; the licensing cell
(e163: the same dial on arm_b, 7% row-0-share — reads ~1.0 => collapse
to truism; 0.7-0.8 => stands) is queued]. The "addresses" an earlier
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
law; the brake overshoot replicates 3/3 (e152R) [e158 COMMITTED PASS-2: TEXTURE — closure requires the CONJUNCTION novelty x zero-variance (jitter@novel OPEN 0.789, locked@home MID 0.458 straddling, locked@novel SHUT 0.102); the door's closure ACCOMPANIES novel-site teaching — graft-formation per se is not the closer (a home graft formed with the door open); mechanism: graft formation is NOT SUFFICIENT for closure (a home graft formed with the door open — n=1, the straddling cell); novelty is a candidate variable (e165); carriage LOCATED by e173: the MLP+LN class carries the closure — restoring it alone reopens the door to 84% of ceiling (CE +0.07, graft intact; graded: a mid-late L2-L4 band) — [e166's row-null was an instrument tautology, withdrawn; e173's ladder licenses it]; e154 landed TEXTURE with a confound: F1 ANNIHILATED under a protocol whose anchors contradict it — e170 RESOLVED: OVERWRITE-REAL — F1 annihilates identically under neutral anchors (the contradiction channel removed by construction); capacity ~one fact wide at this budget; 'globally' licensed as 'any second-fact install demolishes the first'; e174 (direction only, n=1): rehearsal MAINTAINS (1:1 interleaving holds F1 ~0.92+ while F2's graft forms — two independent protocols now support the direction); the dose-2 kill is over-determined (the wash channel alone suffices per e176N); a transient eviction hole at t~4-8 exists even under rehearsal; no capacity formula [the 1/rehearsal-fraction formula struck per R51]]. (3) SPLIT CUSTODY: the converted memory's
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
  W003's CLS analogy, now narrowed to replay-only).
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
