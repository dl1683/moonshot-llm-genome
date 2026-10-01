# Draft — R2b (the wash law) + R6 (the generative turn)

DRAFTING AGENT, 2026-10-01. Assembled from the claims ledger
(scratch/claims_ledger.md) + skeleton (scratch/day6_paper_skeleton.md) with
every number spot-checked against THINKING.md T139-T157 (with amendments),
NOTES.md's newest entries, and runs/*/metrics.json. Stamps travel with
claims. Every number's source is in the TRACE CHECK appendix.
NOTE: R2b runs ~1070 words / R6 ~850 (counts include the inline
citations, ~70/~55 words). Both sit over the 800/700 ceilings — the
mandated clusters (decomposition, terrain, biography pump, front,
flight, classes, static contrast / wall+rhythm+cone+two negatives+
framing, each with stamps and named corrections) set the floor at this
density; trim candidates for the page budget are the bracketed
parentheticals, not the stamped clauses.

---

## R2b. The wash law: a clock, a gate, and typed trajectories

No consolidated memory state we tested survives continued training under
AdamW, at every lr tested above ~1e-5 within its horizon — licensed at n=3
seeds and 2 families over a sparse union of streams [C3; e161-e187; e184;
e157; e180]. The optimizer controls decompose the kill into three
separable objects; the terrain beneath them is mapped (Fig-5).

**The clock is Adam's arithmetic, not the memory's.** Step-1 pre-clip
gradient norm is 0.9829 in every arm; a fresh AdamW moves 1.6543 L2 per
step (lr*sqrt(N), sign-normalized) where matched-lr SGD moves 0.00098 —
1683x at the same lr [opt1/T139]. **The gate is displacement.** Every Adam
variant kills at D 2.49–2.84; warmup stretched the step clock 10.08x and
the kill still arrived at the same displacement within ~15%
(two-convention-bracketed: checkpoint 2.49 vs 3.64, interpolated 2.18 vs
2.85; the registered bar was circular for the reference arm) [opt1/T139 +
R58-amendment]. Death is priced in raw displacement: same arm, same t*,
three wash seeds die within 0.6% of the same D while alignment spreads
3.7x; install-vs-wash cosines in [-0.034, -0.018] reject the
task-arithmetic vocabulary. Alignment is a passenger [e188/T141].

**The matched-lr SGD arm ran the same corpus wash with the fact rising**
(g-12 0.916 -> 0.940–0.955) — and our first reading ("the killer stream
teaches") was half wrong, correctively so: the strengthening is a
small-displacement PUMP, not an optimizer property (AdamW+warmup pumps to
0.9476 at D 0.368 before dying; SGD lingers in the pump at 1/1683rd
speed), and the claimed sign-flip of the stream's fact-relevance was an
estimator-point artifact — at a matched point flattening only attenuates
(+0.0986 -> +0.0396, ~2.5x); the -0.0385 read is real but post-step
[opt1/T139 amendments; e_chart/T150]. The pump has no cuts, the
shuffled-sign ray is inert, and the pump tracks gradient structure (g
+0.045, sign(g) +0.037, isotropic +0.0002) [e_chart/T150; e192/T146]. The
pump is FACT-LEVEL BIOGRAPHY, decided inside one organism: on a fresh root
at organism-1's exact architecture, ZEPHYRA's g-ray ridge is present
(+0.018) and MIRABEL's is absent (+0.0004) along the same rays — the
ridge consolidation-alignment biography (present, we hypothesize, where
consolidation left the fact gradient-aligned with the wash's useful
directions), the cliff physics [e193b/T155; T153]. PROPOSED at per-fact
n=1; present on 2 of 4 facts across 3 organisms (the ledger's "2 of 3"
predates e193b — corrected here; TRACE CHECK note 1).

**The terrain.** Static graded jumps along five ray families, one
organism, one ruler, dual currency, give g < sign < random: the
raw-gradient ray kills at 0.92 (per-coordinate RMS 5.6e-4), the static
sign(g0) ray at 2.5 (2.72x; the reference arm's step-1 read 0.678 sits ON
this static curve — the path-length stitch objection dies by measurement),
three Gaussian rays leave the fact flat and alive at 4.0 [e192/T146;
e191/T144]. The cliff needs no overshoot: the static profile matches the
dynamic kill (edge bracket [0.80, 0.92]; within [0, 1.6543] static =
dynamic by construction, disclosed) [e191/T144]. The order is draw-,
fact-, and architecture-robust at n=3 organisms / 2 facts / both
replication axes (organism 2: 0.20 < 0.58 (2.90x) < >4.0 [e193/T153];
fresh-root MIRABEL: 0.8 < 2.0 < >4.0 [e193b/T155]); the ABSOLUTE kill
distances are the fact's strength biography — ZEPHYRA, ruler-dead at D=0
on the imported ruler, holds the order at >2x-down distances, the
[0.5x, 2x] windows held only by gate-passing facts [e193/T153;
e193b/T155]. The g-ray is the most lethal of the five rays sampled — the
gradient-history span contains directions more lethal still (in-span
randoms kill at 0.56–0.61), and the isotropic band is unresolved-high
(>12): three rays are not a map [e_chart/T150, R60-amendment].

**The lethal front.** At matched per-step L2, magnitude information
orders the kill — raw 0.92 -> flattened sign 1.75 -> Adam's warmed path
2.49 — while density carries nothing (top-10% rung 0.9066; top-50%
0.9203 == raw): the lethal object is the |g|-weighted front at any
density >= 10% [opt2/T151]. The magnitude-shuffle breaker refines it:
permuting the front's magnitudes kills at 1.68x the top-10% rung —
support + signs suffice, magnitude-pairing contributes: support > signs >
magnitudes in necessity order [e193b/T155]. The sign edge is the
terrain's softest number, triply measured at 1.90/2.63/2.24; the clause
carries the range [e192; e193; e193b].

**Trajectory classes, and the flight direction.** Three ways to die, one
way to erode. The guillotine: Adam's ~2 sign-steps, dead at D ~2.49. The
annihilation: one raw-gradient step at Adam's size kills at 0.92 — at
matched D 1.6543 the fact reads 0.0007 with the organism devastated, vs
0.678 under Adam and 0.79 under the small-step walk [opt1c/T143;
opt1b/T142]. The grind: a diffusive walk (per-step 0.0072 buys 0.0012 of
D) that entered the kill window's margin alive (0.324 at D 2.12),
stalled, and through 150 further every-step reads never died (min 0.289;
final D 2.25), eroding at -2e-4/step — asymptote unmeasured by design:
three linear projections falsified (D~10; the s1214 crossing; the class
again) retire the instrument class, not the question [opt1b/opt1b2/
opt1b3; T142/T145/T147]. Re-orientation owns the sparing,
interventionally: pinned small steps down the frozen g-ray — orientation
denied, step size at the bleed's own scale — die inside [0.827, 0.993]
while the re-orienting bleed lives 0.79–0.86 at the same D [e192/T146].
Dynamics cut both ways, measured in both directions: the re-computed
sign front kills at 1.75, 23% below its own static edge 2.27, the
k-ladder a step function — one recomputation is the whole bonus; the
front's frozen-ray overlap collapses while its alignment with the fact's
own gradient rises: the lethal subspace flees with the state, the fresh
front pursues it [e194/T156]. And the flight direction itself is the
killer: the rotated ray sign(g1) — ANTI-aligned with the root's fact
gradient (cos -0.067) — kills at 0.39 from the root where the aligned
original kills at 2.27, an 83% concentration of lethality into the
direction the support fled toward; from the one-stepped state the panel
reverses (the terrain concentrates AND rotates), and the valley the
support flees into has a pumping near rim (short jumps improve the fact
0.68 -> 0.82 before the crash) [e195/T157]. e194/e195 are n=1
organism/fact/stream — PROPOSED, both.

One contrast scopes the law's letter: no state survives continued
TRAINING on any learned path tested, corpus or noise-label alike; static
random displacement is 4–10x more forgivable at the organism
(kappa_store 5.0 [2.8, 6.6]; kappa_host 6.0 [4.0, 9.7] at matched
per-coordinate RMS; intervals overlap) — n=1 per organism, replication
owed; the wash law is never stated against random displacement
[g3K/T137; C13-1].

---

## R6. The generative turn: memory made architectural

Every law above says memory dies. The g-series asked whether the death is
an engineering target — each architecture designed FROM a dissected law,
pre-registered with frozen bars before compute, then replicated.
Construction preceded explanation: the physics explains the architectures
after the fact, which is not the same as "the same physics is generative"
(Fig-4) [C6; the corrected abstract clause, claims_ledger.md line 37].

**(a) The wall.** Commit-and-project — one commit, one scalar L2 ball,
zero old-task statistics, zero new parameters — holds the consolidated
fact flat at ~0.9 through the +300-step wash that kills the control:
mins 0.746/0.777/0.803 at n=3 wash draws on one root (LICENSED; the root
replicate queued) [g1b/g1bR/T133]. The protection is channel-scoped —
architectural against displacement, not against damage: at pinned
radius, corpus damage reverses as displacement accrues (settled +1 read
0.945) while noise-label damage does not (0.0037) [g1b/T125]. The tax is
standing: +0.53 nats [g1b/T125]. Under an active second-install attempt
the wall held fact A at min 0.65 through 300 steps — protection survives
interference — though the antagonist installed nothing at this dose and
is a weak antagonist until the dose ladder says otherwise [g1bW/T140].
The museum rescope is WITHHELD, honestly: the unwalled reference failed
the same ruler (B 0.063; at 300 steps B installs nowhere), so B's
failure inside the wall is uninformative about the wall. The
contrast-licensed read is the ONSET TAX: B's partial form peaked 0.21
walled vs 0.53 unwalled on bit-identical inputs — inside-the-ball
protected, outside-the-ball resisted [g1bW/T140; R58-2c]. Scale is the
honest BLOCKED cell: the 10M host never reached the wall — the inherited
recipe overtrains it ~3x past its validation minimum (val 1.577 at
s~1113, rising to 2.851; the coherence-pass/val-fail split is the
memorization signature), and the registered hard stop fired before any
arm ran. Recipes are scale-bound — steps and lr re-register per host
size, anchored at the val minimum — and the wall's scale question is
open (one licensed GPU hour; g1bS2) [g1bS/T148].

**(b) The rhythm.** A zero-parameter organ — cue-pool selector, onset
monitor, replay gate — self-times its resurrection events into a
maintaining oscillation (cycle-median 0.615, duty 69%) [g2c/T130]. Its
autonomy is real, bounded, priced. Threat-responsive within [0.5x, 2x]
but weakly: 10 -> 12 events across 8x threat, and at 4x every spacing
pins to the refractory floor while maintenance collapses (0.619 ->
0.240 -> 0.002) — a thermostat on a leash [g2g/T152]. Worth +0.07 of
cycle-median over a count-matched fixed schedule at operating threat,
licensed at n=3 wash seeds on one root: per-seed +0.069/+0.033/+0.012,
median +0.033 — barely clearing its 0.03 bar, by 11% [g2g2/T154]. The
paired control splits the margin: majority
replay-batch composition (+0.056 — the selector chooses the right
batches), minority timing (+0.013 — the thermostat): the organ is a
better selector than scheduler [g2g2/T154]. The refractory is a
tunable, shorter better (0.671 vs 0.619); the advertised 20–45 band was
partly a construction floor, disclosed [g2g/T152]. Timing replicates on
a second root (2/2); the adjudicated amplitude is root-draw-bound — the
confound the root's own strength lottery (0.591/0.684/0.711), not the
organ [g2e/T132]. Beyond operating threat the safer letter is
FIXED-MATCHES-OR-WINS, the 1x win co-reported (the 2x leg rides one
float-fragile check) [g2g/T152].

**(c) The cone.** Forgetting is not distance; it is direction. At the
organ, the store's basin is a cone: the wash trajectory kills at +1
while matched-L2 isotropic noise spares through 4x — the store-isolated
leg reading 24–32x, an ORGAN property (n=1 organ; the design's property,
never the organism's) [g3/T126; g3K/T137]. At the organism the split
replicates at n=3 wash-draw seeds: wash kills at <=2x where isotropic
spares at >=2x (at 2x identical L2: wash 0.10–0.16 vs iso 0.85–0.89)
[g3R/T135]; both rulers show graded static basins of 4–10x (kappa_store
5.0, kappa_host 6.0) — the trajectory-vs-static hypothesis,
organism-level, replication owed [g3K/T137].

Two scoped negatives bound the program. The compass is positional at the
A-floor: a content-conditional address table was never recruited — the
site lands on the positional floor every time (rows 5–13, +0.288; the
offered channel never taken; substance 3/3, the row band draw-brittle)
— and the type cliff did not fire on either 0.86M root, matched control
failing identically: the switch is scale/lineage-bound [g4/T127;
g4R/T134]. And single-site walls fail two-site fragility: pinning W_q
does not save the fact (the wash moves the host's stream, not W_q's
weights; root-W_q-in-washed-host retrieves 0.083 vs
washed-W_q-in-root-host 0.891), the route dies independently (a perfect
root store into the washed host reads 0.00004), and the whole-store
wall preserves the organ flat through +300 — the g-series' first wash
survivor — while the organism cannot address it: a 17k wall holds the
store; only a whole-net wall holds the memory [g5/T129].

The three positives were designed from the dissected laws — the compass
predicted where to wall, the no-basin/rate-law results predicted what to
detect, the direction-vs-energy split predicted the basin's shape — and
two falsifiers fired honestly en route (g5's W_q wall; g2's gate-silent
detector). That loop is itself evidence the laws are causal, not
descriptive: memory in these nets is not fragile by necessity but by
default [C6; skeleton R5 pre-emption].

---

## TRACE CHECK — every number and its source

Format: number -> source path (run metrics verified = M; card/notes
only = C).

R2b:
- "no state survives ... under AdamW, every lr > ~1e-5; n=3 seeds, 2
  families, sparse-union streams" -> C3 scratch/claims_ledger.md;
  scratch/day6_paper_skeleton.md R2 abstract; NOTES e184/e157 (M per
  their runs).
- pre-clip grad norm 0.9829; AdamW 1.6543/step; SGD 0.00098; 1683x ->
  runs/opt1/metrics.json (M: a0.traj[0] step_disp 1.6542880535125732,
  preclip_gnorm 0.9829167; a1.traj[0] step_disp 0.000982907) + NOTES
  opt1; T139.
- Adam variants kill D 2.49–2.84; warmup 10.08x; same D within ~15%;
  two-convention bracket 2.49 vs 3.64 / interpolated 2.18 vs 2.85;
  circular bar -> runs/opt1/metrics.json (M: a0 D_at_kill 2.4893, a3
  3.6361, t_x_ratio 10.0774, a4 2.4846) + T139 R58-amendment (bracket
  + circularity are card-level: C).
- SGD fact rising 0.916 -> 0.940–0.955 -> runs/opt1/metrics.json (M:
  a1 step-1 gm12 0.9162) + NOTES opt1.
- pump not optimizer property; A3 pumps 0.9476 at D 0.368 ->
  runs/opt1/metrics.json (M: max gm12 0.94762 at cum_disp 0.36758) +
  T139 R58-amendment.
- estimator artifact; +0.0986 -> +0.0396 (~2.5x); -0.0385 post-step ->
  NOTES e_chart + T150; anchor pair reproduced in runs/opt2/metrics.json
  provenance (M-gated).
- pump no cuts; shuffled sign inert; in-span 0.56–0.61; random band >12
  -> NOTES e_chart; T150 + R60-repairs.
- pump structure: g +0.045 / sign +0.037 / iso +0.0002 -> runs/e192/
  metrics.json (M: small_D_max_rise 0.0449 / 0.0373 / <=0.00017) +
  T146.
- pump fact-level: ZEPHYRA +0.018, MIRABEL +0.0004, same rays ->
  runs/e193b/metrics.json (M: pump_g_max_rise 0.01835 / 0.000356) +
  T155. "2 of 4 facts / 3 organisms" = e192 f1 (+0.045), e193 f2
  (absent, -0.079 per NOTES e193), e193b ZEPHYRA (present), e193b
  MIRABEL (absent).
- terrain: g 0.92 (RMS 5.6e-4), sign 2.5 (2.72x), Gaussians alive flat
  at 4.0; A0 step-1 0.678 on static curve -> runs/e192/metrics.json (M:
  first_dead_D 0.92 / 2.5, first_dead_rms 5.56e-4 / 1.51e-3,
  unresolved-high; NOTES e192 for flat 0.896–0.905 and the 0.678
  stitch read) + T146.
- e191: edge bracket [0.80, 0.92]; static=dynamic-by-construction
  disclosure; (ridge peak 0.9605 at D 0.20; floor 0.0004 at 2.0 held in
  reserve) -> runs/e191/metrics.json (M) + NOTES e191/T144.
- order n=3/2-facts: org2 0.20 < 0.58 (2.90x) < >4.0 -> runs/e193/
  metrics.json (M: g_kill 0.2, sign_kill 0.58) + T153; e193b MIRABEL
  0.8 < 2.0 < >4.0, ZEPHYRA 0.4 / 0.92 order holds, windows held by
  gate-passing facts only -> runs/e193b/metrics.json (M) + T155.
- "most lethal of the five rays sampled; span 0.56–0.61; three rays are
  not a map" -> T143 R60-amendment + NOTES e_chart.
- front ladder: topk-10 0.9066 / topk-50 0.9203 == raw 0.9203 / sign
  1.75 / A0 2.4893 -> runs/opt2/metrics.json (M:
  density_ladder_ordered) + T151.
- magnitude-shuffle 1.68x top-10% rung; support > signs > magnitudes ->
  runs/e193b/metrics.json (M: a_magshuf ratio 1.6628 vs a_topk10
  0.9914 = 1.68x) + T155.
- sign edge triple 1.90 / 2.63 / 2.24 -> runs/opt2/metrics.json (M: f1
  anchor 1.9018), runs/e193/metrics.json (M: 2.6287), runs/e193b/
  metrics.json (M: 2.2424) + T155.
- annihilation: kill 0.92; matched D 1.6543 reads 0.0007 / 0.678 / 0.79
  -> runs/opt1c/metrics.json (M: D_kill_interp 0.9203) + NOTES opt1c/
  T143 + NOTES opt1b/T142.
- grind: 0.0072 -> 0.0012/step; alive 0.324 at D 2.12; 150 reads min
  0.289, final D 2.25; -2e-4/step; three falsified projections (D~10;
  s1214; class) -> NOTES opt1b2 / opt1b3 / opt1b + T145/T147/T142 (C;
  runs/opt1b2, opt1b3 metrics on disk).
- rider: pinned walk dies inside [0.827, 0.993]; bleed lives 0.79–0.86
  -> runs/e192/metrics.json (M: first_dead_D_cum 0.9926; step-150 read
  0.434 at 0.827) + NOTES e192/T146.
- e194: k=1 kills 1.7496; k>=2 at 2.2744; static edge 2.2699; 23%
  below; one recomputation; overlap +0.595 -> -0.162; front-fact
  alignment +0.040 -> +0.060 (post-kill +0.13) -> runs/e194/metrics.json
  (M) + T156.
- e195: rotated ray 0.3875 vs 2.2699 (83% / bar 15%); cos -0.067 vs
  +0.040; ~6x later; theta_1 reversal (u0 0.616 < u1 0.828); rim pump
  0.679 -> 0.824 at D 0.30 -> runs/e195/metrics.json (M: D_kills
  root_u0 2.2699, root_u1 0.3875; th1 0.616 / 0.828) + NOTES e195/T157
  (rim numbers C).
- g3K kappas 5.0 [2.8, 6.6] / 6.0 [4.0, 9.7]; 4–10x static basins; n=1
  replication owed -> runs/g3K/metrics.json (M: pair string) + NOTES
  g3K/T137; C13-1 scope per skeleton gap 11.

R6:
- wall mins 0.746/0.777/0.803; n=3 wash draws one root; controls die ->
  runs/g1bR/metrics.json (M: W1_10907 min 0.8028; W1_10908 min 0.7460;
  reference 0.777 per NOTES g1b/g1bR) + T133.
- channel scope: settled +1 W1 0.945 vs N1 0.0037; +0.53 nats tax ->
  NOTES g1b/T125 (C; runs/g1b metrics on disk).
- A held min 0.65 through second install; B 0.0736; reference 0.0628;
  onset tax 0.21 vs 0.53; row0 0.62–0.69 vs 0.006 -> runs/g1bW/
  metrics.json (M: A_min 0.6505; B_final 0.0736; ref B_g0 peak 0.5328
  @s100 vs walled 0.2101 @s100; walled row0 0.6241–0.6912) + T140/
  R58-2c.
- scale BLOCKED: val 1.577 min at s1113 -> 2.851 final; ~3x past
  val-min; coherence-pass/val-fail; g1bS2 licensed -> runs/g1bS/
  metrics.json (M: chunk vals 1.5766 @s1113 -> 2.8506 final) + T148.
- rhythm: cycle-median 0.615, duty 69% -> NOTES g2c/T130 (C; runs/g2c
  on disk).
- thermostat: 10 -> 12 events across 8x; collapse 0.619 -> 0.240 ->
  0.002; [0.5x, 2x] -> runs/g2g/metrics.json (M: ladder rates {0.0333,
  0.0333, 0.04, 0.04}; head-to-head organ medians 0.6186 / 0.1674 /
  0.0016) + NOTES g2g/T152.
- +0.07 at n=3: +0.069/+0.033/+0.012, median +0.0333, cleared by 11%;
  decomposition +0.056 / +0.013 -> runs/g2g2/metrics.json (M: per_seed
  deltas 0.06878 / 0.03334 / 0.01170; paired cread) + T154.
- refractory: 0.671 vs 0.619; 20–45 band partly floor (5/14 < 20) ->
  runs/g2g/metrics.json (M: refractory legs) + NOTES g2g/T152.
- root robustness: timing 2/2 roots; amplitude root-draw-bound; root
  lottery 0.591/0.684/0.711 -> NOTES g2e/T132 (C).
- FIXED-MATCHES-OR-WINS beyond 1x; 2x float-fragile -> runs/g2g/
  metrics.json (M: 2x delta +0.007) + T152.
- cone: store-isolated 24–32x; wash kills +1 / iso spares through 4x ->
  NOTES g3 + g3K/T137 (C); organism split n=3 wash seeds; wash <=2x /
  iso >=2x; at 2x 0.10–0.16 vs 0.85–0.89 -> NOTES g3R/T135 (C;
  runs/g3R on disk); kappa_store 5.0 / kappa_host 6.0 -> runs/g3K/
  metrics.json (M).
- g4: rows 5–13 +0.288; A-floor never recruited; cliff no-fire on
  0.86M, control identical -> NOTES g4/T127; substance 3/3, band slid
  -> NOTES g4R/T134 (C).
- g5: W_q pinned fact still dies; 0.083 vs 0.891 transplant; route
  0.00004; STW 0.75 -> 0.88 flat through +300 -> NOTES g5/T129 (C).
- framing (construction preceded explanation; corrected clause) ->
  scratch/claims_ledger.md line 37; skeleton g-series amendment + C6;
  R5 pre-emption (g5 W_q falsifier; g2 GATE-SILENT -> NOTES g2).

### TRACE CHECK notes (ledger gaps/contradictions found while drafting)

1. C5's "present on 2 of 3 facts" is STALE: it predates e193b. Measured
   pump status is 2 of 4 facts across 3 organisms (f1 present, e193-f2
   absent, ZEPHYRA present, MIRABEL absent). The draft says "2 of 4"
   and flags it.
2. runs/opt2/metrics.json gates read a_topk10 energy_frac 0.7497 (~75%),
   while NOTES opt2/T151 state "the top-10% front holds 86.6% of ||g||
   (the committed census read; the agent's 75% had no artifact —
   corrected per R60-audit)". The 86.6% does not appear in the machine
   record I can find; the two census numbers contradict or the
   amendment's direction is misworded. The draft OMITS the mass
   fraction (the kill ladder stands without it). Needs adjudication
   before the paper quotes any mass number.
3. runs/g1bW/metrics.json's adjudication clause calls the reference's
   0.0628 an "install" and asserts the museum rescope; the R58 amendment
   (the interpretive record) withholds it. The draft follows the
   amendment; the metrics file stays frozen as the machine record.
   Already documented in T140, restated here because R6 quotes this
   cell.
4. The skeleton's Fig-5 caption ("the pump ridge does NOT cross
   lineages") is superseded by e193b: the ridge crossed the root-draw
   axis (fresh root, organism-1 architecture, ZEPHYRA pumps) and tracks
   FACTS, not lineages. Caption should read "fact-level biography" per
   T155.
5. Minor: T144's card says "pump ridge 0.94–0.96 across D 0.05–0.50"
   but runs/e191/metrics.json reads 0.910 at D 0.50. The draft uses
   only the metrics-verified numbers (edge bracket [0.80, 0.92]); peak
   and floor held in reserve in the trace list.
