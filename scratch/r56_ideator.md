# R56 IDEATOR — the next three generative cells (g6 / g7 / g8)

2026-09-29, frontier review R56. Inputs read: AGENTS.md, THINKING.md
(T115–T135, W008–W020 especially W020 and T129), NOTES.md (g/g2b–g5,
g1b/g1bR, g3/g3R, g4/g4R, e179–e187), QUEUE.md, REVIEWS.md R55,
scratch/g1–g4_design.md, scratch/g6_design.md (the 21:05Z draft —
reconciled below), runs/g*/metrics.json (g5 read in full; g1b tax
verified: W1 1.150 vs C 0.624 in-batch CE at +300, delta +0.526).

## Program state (one paragraph, so the cells are legible)

Three law-grade claims stand: THE WALL (commit-and-project L2 ball,
R=0.7, holds the fact flat ~0.9 through the 300-step wash that kills
in 2; n=3 wash seeds; tax +0.53 nats), THE RHYTHM (zero-parameter
rehearsal organ; onset-only monitor; 10–11 self-timed events, 24–36
spacing, ~70% duty; n=3 wash draws on one root; timing root-robust,
amplitude root-draw-bound — g2f, the base redraw, DISPATCHED 21:04Z
and NOT re-proposed here), THE CONE (the store's basin is directional:
wash-direction kills at ≤2x where matched-L2 isotropic spares through
4x; n=3 draws). g5 decomposed the kill: TWO-SITE, both in the host —
the composed query path q = W_q·LN(h) dies because the wash moves h,
and the expression route dies independently (root store into washed
host restores g0 to 0.00004) — while the STORE itself survives walled
(STW, R=0.04: 0.75→0.88 flat through +300, tax −0.018). T129 and R55
both name the owed cell: "store + a stream-stabilizing host — can the
host's h be walled more cheaply than the whole net?" That is g6.

Collision check (2026-09-29 ~21:30Z): QUEUE holds no g6/g7/g8; the
only in-flight g-item is g2f (base replicate). scratch/g6_design.md
(21:05Z) is a PRIOR draft of g6 in regularizer ("tether") form on the
2.7M organism, self-marked "to be reconciled with R56's ideation
report before registration (anti-collision)". Reconciliation is in
the g6 section: the draft's genuinely new pieces are adopted as
amendments, its organism choice is parked as g6b.

---

## g6 — WALL THE STREAM: the function-space wall at the memory's interface

### (a) Builds on

- **g5/T129** (verbatim numbers): W_q walled at R=0.03 — fact still
  dead at +1; the census attribution — WALLSTATE-q (root W_q in the
  washed host) x root-K reads a_fact 0.0832 while WQDRIFT-q (washed
  W_q in the root host) reads 0.8914: the wash moves h, not W_q.
  The route: root store into washed host restores g0 to 0.00004
  (site 2). STW (whole-store wall, R=0.04): store-into-root-host
  0.75→0.88 through +300 — the organ survives, the organism cannot
  address or express it. Tax −0.0177.
- **g1b/g1bR/T125/T133**: the whole-net ball works (0.918 through
  +300; seed band mins 0.746/0.777/0.803) but WALL-TAXES-ADAPTATION
  fires: in-batch wash CE 1.150 vs control 0.624 (+0.526) — the
  wash's own learning IS the displacement the ball forbids.
- **g3/g3R/T126/T135**: the cone — the damaging displacement is a
  low-dimensional direction; isotropic spares the store through 4x.
- **g1's design lineage**: commit-and-project as forward semantics
  (settle-at-checkpoint, G-PIN, flat interior — zero force below R).
- The vocabulary note (pre-empting a false collision): e161/e176's
  "frozen stream" meant the DATA stream (teaching stopped); g6 walls
  the RESIDUAL stream (activations). Parameter-space confinement was
  g1; FUNCTION-space confinement at one depth has never been run.

### (b) What is new

1. **The wall's location becomes a design axis**: g1 confined the
   PARAMETERS (a point-neighborhood of one configuration); g6
   confines the FUNCTION at the memory's interface (a tube around the
   anchored function, per-input): every parameter configuration whose
   graft-site activation stays within R_h of the anchored host's is
   legal. The host wash-adapts freely everywhere the tube permits.
2. **The composed survivor cell**: STW (g5's store wall) + the stream
   wall — the minimal organism g5's data named (wallable organ +
   stream-stabilized host), predicted to hold at near-zero tax where
   g1b's whole ball taxed +0.53.
3. **Site-2 localization**: g5's bypass probe conflated the landing
   zone (h at the injection site) with the readout weights
   (ln_f + lm_head). g6's arm ladder separates them for the first
   time.
4. **The noise-wound rider**: g1b's deepest unexplained number — the
   noise kill is POSITION-ACTING (dead at pinned R=0.7; settled +1
   reads 0.004). T125 names it "a g-series question". The stream wall
   is the first instrument that can localize it: the anchor forward
   re-derives h* on any input, so upstream functional damage is
   projection-healed BY CONSTRUCTION; only downstream (readout)
   damage can persist. Three named outcomes (bars below).

### (c) The single knob and the discriminating observation

**Knob: R_h — the activation-tube radius at the store's graft site**
(the post-block-3 residual stream, all positions; zero new trainable
parameters; state = one frozen anchor host trunk, the same order of
state g1's anchor already spends). Mechanics (the draft's "vacuous
projection" objection, answered): the projection fires on EVERY
input, wash tokens included — h_walled(x) = h*(x) + (h(x) − h*(x)) ·
min(1, R_h/||h(x) − h*(x)||), where h*(x) is the frozen anchor host's
forward on the SAME input x. No fact context is needed at wash time;
the wall is a per-input functional tube, not a context-matched hook.
Gradient flows through the rescale (autograd), so the wall teaches
the upstream weights to stay in the tube rather than fighting them.

**Discriminating observation**: the tax readout separates the two
candidate explanations of g1b's +0.53 — (i) protection INHERENTLY
costs the wash's adaptation (any wall that holds must tax) vs (ii)
the cost was the ball's POINT-locality (forbidding motion the memory
never reads). If the stream-tube holds the fact at tax < +0.15,
(ii) wins and the wall's correct denomination is functional; if the
tax persists at the graft site alone, (i)-localized wins — the
wash's learning is concentrated where the memory reads (the poetic
flip: the stream learns at the memory's interface), and only the
rhythm avoids the tax. Second discriminator (site 2): arm A vs arm
B (readout rung) separates landing-zone vs readout-weight route-kill.

### Arms (one file; g3 organism = 890,880 params ≤ 0.9M; root =
runs/checkpoints/g3_gen.pt bit-exact reuse — g5's precedent; wash =
e176N arm A via e157 seed 10902, draws md5-identical across arms)

| arm | store wall | stream wall | readout wall (ln_f+lm_head, ~8.6k ball) | question |
|---|---|---|---|---|
| FREE | — | — | — | control; G-REPRO vs g3/g5 stored cells |
| SW | — | R_h | — | stream wall alone (store still erodes via V/W_o — g5's 0.76→0.58) |
| STW | R=0.04 | — | — | g5's survivor (organ holds, expression dead) |
| SW+STW | R=0.04 | R_h | — | THE COMPOSED CELL |
| SW+STW+RW | R=0.04 | R_h | R_r | conditional rung: only if SW+STW fails (site-2 closer) |
| NZ | R=0.04 | R_h | — | the noise rider: labels-noise 10 steps (seed 18501) then read |

R_h operationalized in a frozen preflight (g5's preflight precedent,
eval-only on stored checkpoints): measure ||h@t* − h@root|| at the
graft site on the fact battery from g3's stored +1 wash; rungs
R_h ∈ {0.5×, 1×} of that. No bar shopping; the ladder is the ladder.

### (d) Registered bars (draft, frozen at registration)

- GATES: G-ROOT (bit-exact g3_gen reuse, root g0 ≥ 0.50); G-ANCHORFUNC
  (the anchor forward reproduces the root's graft-site h to 0.0 on
  the root weights — the tube's center is the root function);
  G-PIN-ACT (per-checkpoint live ||h − h*|| ≤ R_h + one-step fuzz at
  every measured site); G-INPUTS (per-step md5 across arms);
  G-REPRO (FREE ≡ g3/g5 stored trajectories to g5's 7-figure standard).
- STREAM-WALL-HOLDS: "fires if SW+STW g0 ≥ 0.50 at EVERY checkpoint
  {1,2,4,10,50,100,200,300} of the 300-step neutral wash."
- STREAM-WALL-CHEAP: "fires if SW+STW maintains AND its in-batch wash
  CE at +300 is within +0.15 nats of FREE's (g1b's registered cost
  +0.53; CE_R co-reported) — protection without the splint."
- SITE-2-STREAM: "fires if SW+STW maintains with the readout FREE —
  g5's route-kill was the LANDING ZONE (stream-side), and one tube
  covers both of g5's sites at once."
- SITE-2-READOUT: "fires if SW+STW dies but SW+STW+RW maintains — the
  route died in ln_f/lm_head; the minimal organism needs three walls
  (store, stream, readout — ~26k params of anchor + one tube)."
- SPLINT-TUBE (adopted from the 21:05Z draft, renamed): "fires if the
  fact holds ONLY at R_h where FREE's own stream motion is suppressed
  (in-batch CE at +300 within 0.05 of pre-wash — learning stopped) —
  the tube is a ball in disguise; report the full R_h frontier."
- NOISE-RIDER (T125's open question, three named outcomes):
  NOISE-HEALED-BY-TUBE: "fires if NZ's settled state reads g0 ≥ 0.50
  at {1,10} — the function wall heals what the parameter wall could
  not (g1b's F3): the position-acting wound was upstream functional.";
  NOISE-LIVES-DOWNSTREAM: "fires if NZ's store retrieval is intact
  (a_fact ≥ 0.5, store-into-root-host ≥ 0.5) yet g0 ≤ 0.27 — the
  wound is readout-side; the RW rung then closes it."; NOISE-TOTAL:
  "fires if retrieval itself dies under NZ — the wound enters the
  tube despite the anchor (a mechanism nobody has named yet)."

### (e) Honest failure modes

- The tube fights LEGITIMATE context-adaptation at the graft site
  (partial tax); the R_h ladder measures exactly this frontier.
- The anchor host is a second frozen copy of the trunk (~0.87M floats
  + one no-grad forward per step, ~2x step cost — 300 steps ≈ 60–90 s
  GPU at 0.89M; inside the 180 s cap; park-once/CPU fallback per
  e184). The wall's information budget = g1's anchor budget (stated
  on every card; the anchor carries no fact content — G_STOREOFF at
  root, g3's gate).
- SW-only can die slowly through store erosion (V/W_o drift, g5's
  0.76→0.58) even with a perfect tube — the composed arm is the real
  cell; SW-only is the attribution leg.
- Gradients through the projection may distort Adam's moments near
  the tube boundary — grad norms co-reported; SPLINT-TUBE is the
  named catch.
- Single wash seed (10902), single lineage (family 2), one
  construction draw (g3's organ, n=1 by reuse — the same scope the
  cone claim carries); replicates owed before any noun moves
  (g1bR/g3R precedent).

### Reconciliation with scratch/g6_design.md (21:05Z draft)

ADOPTED into g6: the SPLINT framing (renamed SPLINT-TUBE), the
frontier-reporting discipline, and the pre-registered P1/P2
predictions (small-needed-force from the cone's low dimensionality;
site-scan showing one-or-two critical sites — the graft tube IS that
prediction made structural). SUPERSEDED: the tether-as-regularizer
mechanics and the "runtime h-projection is vacuous" objection (the
objection holds for context-matched hooks, not for per-input anchor
tubes — answered above); the replay-confound control is unnecessary
for the hard projection (the wall adds no loss term and no data —
the tube cannot be replay), but it IS retained as the g6b control
(below). PARKED as **g6b (the economics cell, 2.7M with stated
reason: direct comparability with the registered +0.53 tax)**: the
draft's tether regularizer on the g1b organism — lambda-calibrated
penalty on probe-h at fact sites, WITH its replay control leg (same
probe batch as ordinary replay data, matched compute; if replay alone
holds the fact, g6b reduces to g2 and is reported as such). g6b is
the cheaper-to-think, dearer-to-run sibling; it queues after g6's
verdict (if STREAM-WALL-CHEAP fires at 0.89M, g6b asks the same
economics question on the organism where the tax was minted).

---

## g7 — THE ORGANISM: wall + rhythm + cone composed — a memory with a fail-over architecture

### (a) Builds on

- **g6** (prerequisite): the licensed wall configuration for the
  composed organism (predicted SW+STW, or SW+STW+RW if SITE-2-READOUT
  fires). Branch table: if g6's composed cell fails entirely, g7's
  substrate falls back to the best g6 survivor + a whole-net ball at
  the 0.89M organism (R calibrated in g7's preflight from stored
  trajectories — g1's sqrt-scaling estimate; stated, never shopped).
- **g2/g2b/g2c/g2d/T128/T130/T131**: the rehearsal organ verbatim —
  cue pool (jitter windows, registered buffer), ONSET-ONLY monitor
  (T122's sensor lesson: the self-correlation channel is NOT the
  memory; the wash spares Z→E while killing ctx→Z), gate θ_OPEN=0.5,
  refractory 24, error replay (e174 arm B's masked name-char CE,
  e163's replay batch arithmetic). Timing licensed n=3; the waveform
  mapped (24–36 spacing, ~70% duty, slow-rise/plateau/late-fall).
- **e179/T120 + e174/T107**: one replay event resurrects the dead
  (0.033→0.686 by +50; sticky 18+ steps); 9 events/300 maintain;
  events REPLACE wash steps (step parity).
- **g1b/T125**: the noise kill pierces the parameter ball
  (position-acting; settled +1 reads 0.004) — THE WALL'S BLIND SPOT.
- **g3R/T135**: the cone — the re-entry direction question (rider).

### (b) What is new

No composition of two g-mechanisms has ever run. Every g-cell to
date isolates one law. g7 is the lab's first THREE-LAW ORGANISM: a
cone-lawful store, walled against drift, self-rehearsing against
lapses — with each layer covering the other's registered blind spot:
the wall cannot see noise wounds (g1b F3); the rhythm cannot prevent
exit, only re-enter (T120). The systemic prediction nobody has
tested: **the wall makes the rhythm SILENT** (if the read never dips
below θ_OPEN, the gate never opens — 0 events), and the rhythm makes
the wall's blind spot survivable (the noise breach is exactly the
resurrection economy's use case). Fail-over made architectural.

### (c) The single knob and the discriminating observation

**Knob: the wall radius R (the licensed value vs R=off)** — the organ
is always live; the 2×2 {wall on/off} × {organ live/inert} is the
organism matrix. **Discriminating observation**: EVENT COUNT per cell
(0 vs >0 under the wall) + the breach protocol (after the wall holds
through 300 neutral steps, inject the 10-step labels-noise kill —
g1b's unwallable wound — and continue the neutral wash under the
gate). The event count separates "the rhythm is redundant under
protection" from "the rhythm fires anyway (the wall leaks reads below
θ_OPEN — a second clock g1b's FLAT-AT-PIN missed at this organism)";
the breach outcome separates "resurrection is wall-independent"
(the economy holds where the ball failed) from "the noise wound is
also rehearsal-proof" (e179 never tested noise-killed states — only
wash-killed ones; NEW question, honestly live).

### Arms (0.89M organism; g6's substrate; ~5 runs ≤ 180 s + one 10-step breach per walled arm)

1. ORG (wall off, organ live): does the rhythm maintain a
   STORE-CARRIED fact? (g2's rhythm ran on a discriminative fact —
   generality of the organ to the store organism; replay gradients
   update store + host inside no ball at all.)
2. WALL (wall on, organ inert): g6's composed cell re-run — the
   silence control.
3. ORGANISM (wall on, organ live): THE cell. Neutral wash 300 steps.
4. BREACH: arm 3's endpoint → 10-step labels-noise (seed 18501) →
   neutral wash +300 with the gate live. The fail-over test.
5. BASE (wall off, organ inert): dies (control; G-REPRO).

### (d) Registered bars (draft)

- GATES: g6's gates inherited; G-ORGAN-INERT (a dial battery with the
  organ's buffers zeroed is bit-identical — g2's clause); G-POOL
  (the cue pool = this organism's install battery windows, jittered;
  zero fact content in the wash stream); G-STEP-PARITY (events
  replace wash steps, batch parity 32 = 32); G-MONITOR (the monitor
  reads the ONSET channel p(Z|ctx) through the FULL composed
  readout — T122/T128's lesson, non-negotiable).
- WALL-SILENCES-RHYTHM: "fires if ORGANISM opens ZERO gates through
  +300 of neutral wash while the ruler stays ≥ 0.50 at every
  checkpoint — the rhythm is redundant under a working wall." (Co-bar
  WALL-LEAKS: "fires if ≥ 1 gate opens under the wall — the wall
  holds checkpoints but dips below θ_OPEN intra-step; the event log
  is then the wall's finest instrument.")
- ORGAN-MAINTAINS-STORE-FACT: "fires if ORG (no wall) holds
  cycle-median ≥ 0.50 with 5–25 events, 100% of spacings in [20,45]
  — the rhythm generalizes to the store organism." (If it fails, the
  organ is fact-type-bound — also a finding; g2e's root-lottery
  clause applies.)
- RHYTHM-CATCHES-THE-UNWALLABLE: "fires if BREACH's first event
  occurs within ≤ 25 steps of the noise kill, the ruler returns
  ≥ 0.50 by the checkpoint after the event, and the late-grid mean
  over {100,200,300} ≥ 0.50 — the organism survives its wall's blind
  spot." (Co-bar MONITOR-BLIND: "fires if the gate never opens after
  the breach — the noise wound killed the monitor channel itself;
  the organism cannot feel this kind of death — T122's lesson at a
  new site, registered as a failure mode with a name.")
- ORGANISM-TAX: "fires if ORGANISM's CE_R at +300 ≤ root + 0.10 and
  BREACH's total events ≤ 5 — the wall does the daily work at near
  zero events; the rhythm is catastrophe insurance."
- CONE-REENTRY (rider, g3R): "fires if at BREACH's first event the
  store census shows a_fact restored ≥ 0.5 BEFORE the bypass probe
  (route) recovers — replay re-enters through the query cone first;
  the cone is not only the fragility's geometry but the re-entry's."

### (e) Honest failure modes

- The replay event under the wall: after the noise kill the weights
  sit at/near the tube and ball boundaries; replay must pull them
  back INSIDE the flat interior. If the wound direction opposes the
  replay gradient at the boundary, resurrection may need > 1 event —
  the economy bar (≤ 5 events) may honestly fail while the mechanism
  works slower; report the event count, never shop the bar.
- The monitor's channel after a noise kill is itself noise-damaged
  (MONITOR-BLIND is a real candidate, not a strawman — the noise
  kill is collateral-devastating per e185).
- The organ's cue pool indexes the INSTALL battery — the organism
  "knows" what it was taught (g2's trench-coat clause inherited and
  stated: the pool is a retrieval cue, not the fact; the dials
  adjudicate the BODY).
- Substrate dependence on g6's verdict (branch table above); if g6
  lands SITE-2-READOUT, the organism adds a third wall and the tax
  stack must be re-measured (three walls ≈ 26k anchored params —
  still ~3% of the organism; stated).
- Single seed per cell; single lineage; the standing convention.

---

## g8 — THE NATIVE ORGAN: is the two-site fragility structural or developmental? (graft vs grown)

### (a) Builds on

- **g5/T129**: the two-site fragility (query path via the host's
  stream; expression route) was measured on a GRAFTED organ — the
  Hopfield store store-only-trained (host frozen at construction,
  G_STOREOFF enforced: store-off g0 = 0.0012). T129's closing words:
  "the organ-versus-organism distinction born."
- **W016**: "born with one organ — the sink is the native memory
  substrate; addresses are protocol-grown grafts." The complementary
  wonder: what does a memory organ that GREW WITH its host look like?
  Never tested — every g-organ is protocol-installed post-hoc.
- **g4/T127 + g4R/T134**: the A-floor offered at training was never
  recruited (a NEGATIVE about offered channels — but the A-floor was
  a per-token table, which structurally cannot host conjunctions);
  g8's organ is a full retrieval block ON the stream — a different
  offer, and one the jitter conjunction can actually use.
- **e157/T113**: the host's own discriminative fact dies at +1
  (family 2) — the known fate of host-carriage; g8's G_STOREOFF gate
  keeps the comparison honest.
- **e173/T104 + g5's census machinery**: class-restore closure
  partition, transplant attribution cells — reused verbatim on the
  co-trained organism (the store tensors remain a clean class:
  W_q, K, V, W_o).
- **e043/e113 (install + jitter), e098 base (s4305 family)**: the
  co-training recipe's ingredients, all on disk.

### (b) What is new

The single untested variable in the entire g-series: WHEN the organ
meets the host. g3 grafted onto a frozen host; g8 CO-DEVELOPS — the
store is present from pre-training (or from install start; frozen
choice below), the fact's gradient flows through store AND host
jointly, and the host experiences the organ as load-bearing
machinery throughout learning. The W020-necessity question in its
developmental form: if co-development removes the two-site
fragility, then walls and rhythms are PROSTHETICS FOR GRAFTS and the
design spec is "grow the organ with the host" (the cheapest fix in
the whole program — zero parameters, zero walls, zero events); if
the fragility survives co-development, it is STRUCTURAL to
stream/organ composition and the wall/rhythm organs are necessary
parts of any store-bearing architecture, not patches for a
grafting artifact. Either branch rewrites a paragraph.

### (c) The single knob and the discriminating observation

**Knob: ORIGIN — grafted (g3's root, reused bit-exact) vs
co-developed (the identical store architecture, trained jointly
with the host on corpus + install).** One binary; everything else
matched (same store class, same host family, same install battery,
same wash seed, same census). **Discriminating observation**: g5's
attribution census on the CO-TRAINED washed organism — the two
transplant cells (co-trained-root W_q in the washed host vs washed
W_q in the co-trained root host) plus the graft-site stream-drift
measure ||Δh|| at t* vs g3's stored graft value. If the co-trained
host's stream stays in the store's query cone under wash (a_fact ≥
0.5 at +50), development bought stability; if the stream leaves the
cone exactly as the graft's did, the fragility is compositional, not
developmental. The drift RATIO (co-trained/grafted) is the
quantitative bridge either way.

### Arms (0.89M; ~3 new trainings ≤ 180 s + eval-only census)

| arm | construction | wash | question |
|---|---|---|---|
| GRAFT | g3_gen root reuse (bit-exact) | 300, seed 10902 | the reference (G-REPRO) |
| CO | store present from install step 1; joint AdamW on corpus + install (e043 mix) + jitter consolidation (e113), host NEVER frozen | 300, seed 10902 | THE cell |
| CO+WALL (optional rung) | CO + g6's licensed walls | 300 | does the wall compose with a native organ too? |

Construction gates (frozen): G-ROOT-EXPR (g0 ≥ 0.50); **G-STOREOFF
(store-off g0 ≤ 0.27 — the fact remains STORE-carried; joint training
tempted the host to bypass the organ, and the gate refuses the
confound — if it fails after one registered remediation rung (store
CE upweighted ×4, e043's deviation-3 lesson: what carries the loss
determines what installs), the cell reports TEXTURE with the carrier
census, not a verdict)**; G-CE (CE_R ≤ base + 0.10); G-FAIR (matched
optimizer steps and fact-exposure steps vs GRAFT, same seed family).

### (d) Registered bars (draft)

- TWO-SITE-IS-STRUCTURAL (committed): "fires if CO's g0 ≤ 0.27 by
  +50 AND the census attributes the death to the host again (live
  a_fact < 0.5 at t* with washed-W_q-in-root-host ≥ 0.5) AND the
  graft-site drift ratio ≥ 0.5 — development changed nothing
  structural; walls/rhythms are necessary organs."
- NATIVE-STABILITY (the program-reshaping falsifier): "fires if CO's
  g0 ≥ 0.50 at EVERY checkpoint through +300 — the co-developed
  organism survives the wash bare; the two-site fragility was a
  grafting artifact and ORIGIN is the cheapest wall in the program."
- PARTIAL-HARDENING (the honest middle): "fires if CO dies but its
  t* is ≥ 3x GRAFT's AND/OR the drift ratio ≤ 0.5 — development
  slows the kill without stopping it; the fragility is structural
  but developable; report the frontier."
- ROUTE-ALSO (co-report): "the bypass probe on CO's washed state —
  does the expression route die independently in the co-trained
  organism too (g5's site 2 replicated) or did co-training couple
  the route to the store's welfare?"
- CONE-NATIVE (rider): "the lambda/isotropic ladder on CO's store
  subspace — is the co-trained store's basin cone-shaped like the
  graft's (g3R), or did joint training widen/re-orient it? Bars:
  wash-direction kill ≤ 2x with isotropic spare ≥ 0.5 at 2x (cone
  replicates) vs any-direction kill at 1x (the native store is a
  pan)."

### (e) Honest failure modes

- Joint training fails G-STOREOFF (the host slurps the fact — the
  known e157 fate; one remediation rung registered, then TEXTURE).
- The co-trained store may be a DIFFERENT KIND of object (the host
  co-adapts around it — e.g., the store's gate keys drift to
  host-dependent prototypes); the census reads this directly (the
  attribution cells), but the comparison to GRAFT then needs the
  caveat that ORIGIN changed the organ, not just the host — stated.
- CO's consolidation may land in the root-strength lottery band
  (g2e's 0.591–0.711); the ROOT-STRENGTH gate (≥ 0.7 at the ruler
  geo, g2's convention) applies, with one re-draw registered.
- Survival could arrive by DEGENERATE coupling (the store riding a
  host pathway so heavily that store-off barely dents g0) — the
  G-STOREOFF margin (how far below 0.27) is co-reported as the
  coupling strength.
- Single seed/lineage; replicates owed before any noun (standing).

---

## Dispatch order (the ideator's recommendation)

**g6 first** — it is the cell TWO artifacts independently named (T129:
"a g6 question: can the host's h be walled more cheaply than the
whole net?"; R55's owed-debts list: "g6 unqueued (store +
stream-stabilizing host)"), it discharges three open questions at
once (the composition survivor, site-2 localization, T125's
noise-wound mechanism), it is the prerequisite for g7's substrate,
and its preflight builds the stream-drift instrument g8 reuses.
Then **g7** (needs g6's licensed wall; the organism matrix is the
paper's capstone figure candidate). **g8** is independent of both —
it can slot into any GPU gap (its census legs are CPU-eval-only) but
is ranked third because its question, while fresh, does not gate
anything else. g6b (the parked 2.7M tether variant) queues after
g6's verdict. All cells: 0.89M (≤0.9M default honored; g6b alone
argues 2.7M for tax comparability), ≤180 s per run, zero new
trainable parameters, one file per experiment, gpu_ok/cooldown
discipline, no concurrent GPU (g2f currently owns the lane).
