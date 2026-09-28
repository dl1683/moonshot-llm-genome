# Experiment Queue

Statuses: `READY` (next up), `RUNNING`, `DONE (see NOTES.md)`, `PARKED`
(idea only, no live-hypothesis discrimination), `GATED` (waiting on a
prerequisite). Rewritten at Review 1 (2026-09-24T11:20Z) to fix drift.

| id | experiment | status | one-liner |
|---|---|---|---|
| e021 | task-swap retrieval | DONE | all 4 predictions: 100% copy, far-value ln26, retrieval head L4-H1 95.1% ID-mass, new L4 decision mode (88.3%) — claim 4 narrowed |
| e013a | attention census | DONE | funnel replicates at scale (far-mass U 0.80→0.09→0.54); L5 abandons local in 82.5% of prompts; rare-token story DEAD (0 concentrated heads, flat surprisal) |
| e013 | context-truncation calibration test | DONE | REFUTED: L5 calibration is local (KL −6.9%); 16-token sufficiency — far context worth ≈0 nats at char level; mid-stack readouts anti-informative |
| e013c | far-value tail | DONE | bimodal: 30.6% gain (decile +1.60), 28.2% HURT (decile −1.68) — far context is a double-edged sword; T007 written |
| e013d | interference audit | DONE | P1+P2 refuted: no repeat interference (91% no divergent match); gains shuffle-robust, incoherent far hurts MORE — far context = bulk statistics, T007 closed |
| e029 | seed × regime transplant matrix | DONE | ΔW-alignment CONFIRMED (same-init +0.152 vs diff-init ≈0.000 — orthogonal training motion); seed dominance is MLP-specific (ρ 2-3.6), attention portable; R-host MLP-L0 regime-dominant |
| e028 | cross-anatomy transplant | DONE | P3 REFUTED REVERSED (ρ=0.874): organs portable; incompatibility follows seed/init lineage; keystone asymmetry both ways; trained-foreign > random interference |
| e019 | MLP-5 thermostat | DONE | scale MLP-5 write by α∈{0,.5,1,2} + rotate; entropy/top-k/CE response — direct causal test of the energy-carrier claim (eval-only, minutes) |
| e003b | corrected ascent instruments | DONE (superseded by e003c) | projected + masked (top-k A-specific) ascent, dense steps 0–30; target=train-A CE, collateral=val_B CE (labels fixed per critique) |
| e013 | rare-token causal mask | SUPERSEDED | census found no concentrated rare-token heads; replaced by context-truncation design |
| e014c | write-clamp training | PARKED (R2) | clamp ‖w‖ ≤ α·‖x_in‖ during training (or eval-time rescale L0/L5 writes ×{0.5,2,4}) — decisive test of "damage tracks write allocation" (P3 passed correlationally) |
| e018 | causal depth | DONE (T012) | activation-patching depth: shallowest d where splicing a counterfactual context switches the decision — upgrades T004 past the depth-6/L5 circularity |
| e012d | causal census × 4 nets | DONE (T014) | causal-depth census on B43/R/R43: is CAUSAL depth the cross-net invariant? (C1's remaining evidence) |
| e043 | install a name | DONE (T015 amended) | ASYMMETRIC-CHEAP-REMOVE; expression gap; protocol-fragile install |
| e044 | scar tissue | DONE (T018) | post-erasure re-exposure: does the row regrow or the name return via body routes? |
| e046 | C6 replication | DONE (T016: C6 demoted) | two-factor erasure does NOT replicate; address-half general | R5 missing observation: D2-analog + in-run top residual head + J-census collateral + uniform-floor battery |
| e014b.1 | replication seed | DONE (e030 slot) | second seed for the renorm-plasticity result (anatomy plasticity is single-seed) |
| e011c-ci | bootstrap CIs | DONE (e030 slot) | resample eval batches for e011c rotate/zero ratios (MLP-L1 ×2.95, MLP-L5 ×0.24 beyond noise?) |
| e001–e003, e011a/b/c, e012, e014b | — | DONE | see NOTES.md |
| e027 | predict-and-poke | PARKED | (was e013 collision) pick a direction that should flip a behavior, poke it, score prediction vs surprise |
| e004–e011, e015, e016 | — | PARKED | no live-hypothesis discrimination; e011 refolded into e019 |

## Visualization thread (`lab/vNNN_*`, standing — see README VISUALIZER)

| id | viz | status | one-liner |
|---|---|---|---|
| v001 | token journey | DONE | decision-depth observable discovered (T004); authority schedule visualized |
| v009 | dw-portability atlas | DONE |
| v010 | self-portrait | DONE (v010.1 labels verified) |
| v002 | attention atlas | DONE | locality funnel (L0→L3→L4/L5); L5 local-abandonment real, rarity = one head (Review 1) |
| v006 | decision-depth passage map | DONE | letters late / structural chars early; 51% finalize at L5 |
| v007 | funnel film | PARKED | animation through depth: distance histograms + rare-token spotlight on dialogue |
| v008 | anatomy phylogeny | PARKED | 8-12 seeds × regimes; embed lesion-map vectors; which organs are conserved homologs vs plastic |
| v003 | write-space geometry | PARKED | PCA/dimensionality of each block's writes |
| v004 | lesion atlas explorer | PARKED | composite anatomy poster |
| v005 | forgetting animation | PARKED | animate ascent trajectories + generation decay |

## Night program (R6, gated)

| id | item | status | what |
|---|---|---|---|
| e047 | replication sweep | DONE (T017) | 3 surviving positives (L5-calibrator, MLP-5 carrier, shared-L0 machine) × 4 nets, eval-only — card v3 gate |
| e044 | scar tissue | DONE (T018, n=1 flag) | the smoke file never ran; full battery now |
| e048 | expression-gap boundary | DONE (T019 + geometry refinement) | does installed-but-silent ever express? exposure dose, prompt-seeding, temperature |
| e049 | retrieval dose-response | DONE (T021) | refrain corpora at p∈{0,5,20,60}% — where does far-retrieval appear on naturalistic data? |
| e040 | graft-evolution | DONE (T024: P2-FROZEN — basis invisible to selection) | structure readouts only (alignment, rho drift) — no circuit claims |
| e033 | write-equalizer | DONE (T023, 0.84M n=1) | homeostasis at structure level; who absorbs the energy |
| e005s | minimal scaling capstone | DONE (T020, steps-confound flagged) | 0.7M/8M × 2 seeds; qualitative readouts only |
| v011 | edit film (re-cut) | DONE | six-frame L7 strip, all numbers from metrics, n=1 flags boxed | the asymmetry law, with the expression gap visible |

| e050 | directed-mutation lineage | DONE (T027) | VISIBILITY-LIMITED: random-vs-directed trickle identity (−3.5/−3.6%) — FROZEN at full strength |
| e052 | LN/geometry reanalysis | DONE (T026) | damage tracks organ-reliance r=0.807; LN excluded; L2 geometry residue real |

## Day-three frontier candidates (T025; all reuse existing tooling)

| id | experiment | what |
|---|---|---|
| e055 | suppression localizer | DONE (T033/T035) | causal d4 rescue at onsets; transient off-onset; claim-split final |
| e056b | circularity killer | DONE (T034) | rescue position-general (R1); knowledge-specific |
| e056c | downstream check | DONE (T035) | LOUD LOGIT PASTE off-onset; claim-split final |
| e064 | gate stress test | DONE (T030) | unification KILLED |
| e053 | cache utility timeline | DONE (T030: live-frac ~25% invariant, onset ~63 invariant, sink dead) | per-position K/V patch-lesion; when does non-sink cache become dead weight? (nobody has this curve) |
| e054 | context-rot anatomy | KV-recall trained at 512, swept to 2048; positional-vs-content patching of the retrieval path |
| e055 | suppression localizer | transplant teacher-forced residual states at the divergence token into free-running; localize where known answers die |

## Day-three wave 2 (ideator harvest, T026 openings; zero-GPU first)

| id | experiment | what |
|---|---|---|
| e058 | geometry-site anatomy | DONE (T029) | zero-GPU per-site r(align-dist, damage) x 11 ckpts + 2.7M replication — why L2 but not L3? |
| e059 | winner differencing | DONE (T040: H-nothing at bars; interface family = second damage predictor, partial r −0.654; trickle = single-lineage artifact) |
| e063 | load homeostasis | DONE (T041: H-EMERGENT — universal template, r=+1.000 across init+order; no heritable A-variance) |
| e063b | task-swap discriminator | DONE (T041 amendment: H-ii optimizer-attractor — copy-net shape r=+0.998; magnitudes task-weighted, shape corpus-invariant) |
| e070 | attention-mass discriminator | DONE (T044: NO CLAUSE — young-age mass window-invariant 1.006; mid-far gains 1.44>renorm; native a*=13; cross-thread window-start prediction REFUTED) |
| e072 | value-side vs threshold | DONE (T044 close-out: BOTH fire — per-norm value efficiency +11% load/unit with magnitudes down; a* B-fragility: 18→7 at B=16) |
| e056 | healed-host graft | ablate L3-MLP, heal to parity, graft donor — host-fragile-organ vs donor-basis-fit |
| e060 | residual-selection lineage | e040 rerun with A-residualized damage (T026's method note) |
| e062 | subspace-cosine predictor | DONE (T046: P2 WINS — partial r(D|A) −0.976, rule cos≥0.4459 AUC 0.919; W_out rowmean chance-grade; scale-B transfers −0.997) |
| e061 | calibration rescue | scalar/gain nudges at e055's suppression depth vs full transplant |

## Day-4 programs (ideator harvest 2026-09-26; memo: scratch/post_paper_programs.md)

P1 COORDINATE (top pick) | P2 IMMUNOLOGY | P3 CACHE WEATHER | P4 THE ERASER (wildcard, gated on P1 census). Numbering fixed: e065 = RMU-vs-surgery head-to-head (design: scratch/e065_rmu_headtohead_design.md, under critic review); ideator's P1 ramp renumbered below.

| id | experiment | status | one-liner |
|---|---|---|---|
| e066 | close the loop (P1) | DONE (T038: TWO-OBJECTS, cos 0.094) + e066b in-place rows (GRADED, swap 0.46/zero 0.27 from 0.72) | relay is a circuit-shaped third thing, not the row; address = distributed conjunction w/ wpe-row concentration |
| e067 | address census (P1) | DONE (T042: ROW 0 top anchor — window-anchored conjunction; bimodal rows 0+129 (53%) + micro-carpet; NOT sparse, dense-cluster refuted) |
| e071 | row-0 generalization (T042) | DONE (H-WINDOW-KEY sweep; both anchors generalize to held-30; generic+key coexist per KL) |
| e068 | rebinding surgery (P1) | DONE (T043: MIXED — portable unit is ROW 129 ALONE, pair ≈ 129-only; row-0 = scaffolding; destruction-vs-portability dissociate) |
| e065 | RMU-vs-surgery head-to-head | STALE (R43 audit: E065 already in NOTES — row kept for history) (design critic-hardened 5f11a79; GPU free) | obfuscation inversion: rescuable-but-reverts-fast vs unrescuable-but-scarred; 5 arms incl. retain-only + no-removal controls |
| — | P2 ramp e060/e062/e063 | RUNNING (agents) | then trained-tolerance + crossmatch grid at promotion |
| e073 | P3 junk split (source stratification) | DONE (T045: H-SLEEPER 4/4 — cache junk is self-generated; prompt entries ~never hurt; 10M extreme 0.367) |
| e074 | shuffled-prompt junk control | DONE (T045 close-out: H-SOURCE strict — no new junk from shuffled prompts 0.024; late-gen 0.169 vs early 0.031 = 5.5x drift gradient) |
| e076 | cosine mechanism + critic fixes | DONE (T046 close-out: alignment survives −0.982; OOS AUC 0.885/0.919 median, LOO robust; NOT cosine-specific — dW near-equivalent, cosine wins on practicality) |
| e075 | source-aware pruning (P3 step 1) | DONE (T048: KILL — +0.26 nats, off-manifold attractor; static junk dynamically load-bearing) |
| e078 | dose-net rebinding rerun (T047) | DONE (REPLICATED n=2 — row129-alone > pair again; rebind 60-71%; old-bar MIXED stable) |
| e079 | B=16 junk-split resample | DONE (claim C net-dependent: 10M robust anchor; 2.7M fires registered-mapping; e053c battery-difficulty-dependent; concentration clears 18.6%) |
| e080 | prune-vs-replace (T048) | DONE (close-out: honest MIXED — noise≈vzero (presence dead), promptcopy recovers 3/4 but misses bar; anchor is RUN-SPECIFIC trajectory content; attractor in all arms) |
| e053c | ctx-512 onset decider | DONE (T039) | ABSOLUTE: a*(512)=6 CI[4,8], onset fraction halved; window-invariant truncation claims |

## Day-5 candidates (explorer harvest 2026-09-26; memo: scratch/explorations_harvest_20260926.md)

16 table-grade ideas mined from Open Exploration + _meta; top proposals:
| id | proposal | status | one-liner |
|---|---|---|---|
| P-A | RIF at our scale (reading writes) | READY (rides e078 pass, eval-only) | prompt-only elicitation of fact-1 suppresses neighbor fact-2's expression (>=0.05, sham-flat) — the read policy has dynamical side effects |
| P-B | coherence-gap dose ladder | READY (e080 rig, eval-only) | mid-generation corruption at eps 0.05..1.0 — NON-monotone per-nat damage (small doses drift permanently, large re-enter a coherent basin): canalization made causal; first structural description of the trajectory anchor |
| P-E | consolidation cycle law | READY (3 erase/re-learn cycles, <=180s/arm) | discharges T037's registered third-cycle prediction (monotone closure) vs interleaved-replay consolidation — canalization stands or takes its second haircut |
| — | synergy: state-over-output | noted | _meta's principle (6 instantiations) = our free-run honesty; lab supplies the 7th + mechanism; e062 crossmatch is the cash-out of _meta's Forecast/Diagnostic pivot |
| e081 | RIF probe (P-A) | DONE (T049: NULL — reads are pure at this resolution; placebo gate noisy 0.044; repro cell unrun) |
| e081b | RIF replication cell | DONE (T049 amendment: NULL REVERSED — RIF present asymmetric n=2; scramble texture was noise) |
| e086 | frequency-flip install (T049) | DEAD (premise was fact-level RIF — killed by e087) |
| e087 | RIF two-rig adjudication | DONE (T049 FINAL: string-level induction only — rig conflict dissolved at B=96; ZABMOTHIC control kills name-identity; reads pure at fact level; e086 dead) |
| e082 | cross-seed row-129 transplant | DONE (T053: BASIS-PRIVATE — structure universal, code seed-specific; crossmatch validated on a one-row graft; T043 rider resolved) |
| e083 | canalization cycle-3 | DONE (T073: MIXED + oscillation — groove persists, erase weakens; strong canalization DEAD, weak form survives) [R44 marker: 'migrates to field' INVERTED by T078 — erase tightens address-binding] |
| e084 | READ-KERNEL census | DONE (T050: KERNEL=SHADOW r 0.918 — dissociation dead, shadow claims validated; young-band 31% flips; 24-27% escapes outside top-5; 48 donor-continuation hits) |
| e085 | anchor-description probe | DONE (T051: anchor not entry-describable — no property fires; T048 rider r(static,dyn)=-0.004; anchors more run-specific) |
| e088 | pair-level anchor probe | DONE (T051: SUB-ADDITIVE 0.464 — interactions killed; anchor is MASS-ACTION) |
| e089 | mass-response curve | DONE (T051 FINAL: MASS-ACTION confirmed — both threshold clauses fire; variance collapse; threshold dose-response law) |
| e090 | dose-vs-schedule decomposition | PARKED | K=1 continuous removal ~8x K=32 lumps — mass vs removal-schedule |
| e091 | reverse-transplant discriminator | DONE (T052 close: H-READOUT-GATE — refuses good states; d5 carriage survives 0.362; onset geometry dead) |

## Next wave (ideator 2026-09-26 ~16:30Z; memo: scratch/next_wave_programs.md)

| id | experiment | status | one-liner |
|---|---|---|---|
| e092 | READOUT-GATE CENSUS | DONE (T056: H-GATE-DISTRIBUTED — no component passes both bars; mlp-L5 necessity-only; the gate is the retrain) |
| e106 | second-channel census | DONE (T058: MIXED — two-channel unconfirmed, single-channel fails; residual open; e107 per-DP layer trajectories registered) |
| e107 | value-side probe | DONE (T059: ROUTING-ONLY decisive; content ⊥ readout (0.033) — architecture forces routing+age selection; read-residual closes as attention-vs-age disagreement) |
| e095 | T050 texture probes | DONE (both flags NOISE — escapes diffuse below MC-null; donor hits 33rd pctile vs 70 bar; T050 final) |
| e083 | canalization cycle 3 | STALE (R43 audit: E083 already in NOTES — row kept for history) | T037's registered debt: ratio >=1 AND cos >=0.6 or canalization falsified |
| e093 | threshold-law generality | READY (eval-only) | k-ladder on ctx-256 + 8M/10M — fraction-invariant vs absolute-count law |
| e094 | ONE-ROW TOLERANCE | READY (GPU, <=200 steps) | pin donor row-129 in seed-43 host, fine-tune rest — recoded lock vs routed-around + crossmatch rider |
| e096 | coherence-gap dose ladder | DONE (T051 amendment: REMOVAL-fragile, CORRUPTION-robust — no damage at any eps; anchor = nonzero content-bearing entries) |
| e097 | sink-asymmetry | DONE (T051 amendment: POSITION ASYMMETRY RECENCY-weighted — new third 2.84x, old cheapest; sink prior INVERTED; exposure neutralized) |
| e103 | contiguous-interleaved decomposition | READY (CPU) | is the stratification elevation recency or locality — matched-recency contiguous vs interleaved subsets |
| e099 | attractor identity | DONE (T055: ONE broad attractor, fluent-but-wrong; T048 REVISED — anchor is NET-FAMILY-specific: sibling-run entries healthy, corpus collapses) |
| e105 | cross-family anchor test | DONE (T057: FAMILY = TRAINED WEIGHTS — competent foreign text collapses; family signal in V-geometry 0.138 vs 0.403; anchor spec complete: mass+family+recency) |
| e100 | read prediction | DONE (T054: ATTENTION-ADDRESSED AUC 0.907 — opened=attended; L1/L2 sufficient; address= routing) |
| e104 | failing-DP census | DONE (T056: LOCALIZED REAL RESIDUE — near-tie/mixed-age strata 3-10x base, p=1e-4; blur refuted; second-channel candidate) |
| e101 | adversarial subsets | DONE (T058: no kill; recency-selection 2-7x — law's 3rd revision; top-readership WORST selector 0.40x — eviction heuristic falsified) |
| e102 | direction-vs-magnitude | DONE (W001/T051 completion: DIRECTION CARRIES THE ANCHOR — unit-norm anchors, norm-random collapses; magnitude floor 10-56%; the anchor is a directional field) |
| e098 | seed-ladder | DONE (T068: structure universal n=6; share-form n=3, value grows with maturity) |

## NEXT ARC (ideator harvest 2026-09-28; memo: scratch/next_arc_programs.md) — P5-P8, Rule-11 ranked

| id | experiment | status | one-liner |
|---|---|---|---|
| e121 | the dreams probe | DONE (T074: NO DREAM CONSOLIDATION — dreams carried the fact and slowed forgetting 500x but never graduated; W004's self = verification boundary not training gate) |
| e120 | fact-in-contexts | DONE (verdict RETIRED-PROVISIONAL per R44 critic — e131 probe 1 showed the arms learned at 183 (0.989/0.988), but graduation needs the D-183 survival cell (e139); corpus>self downgraded to suggestive) |
| e131 | RE-KEYING CENSUS | DONE 07:05Z (T077: RE-KEYED 3/3 — key is ROW 0, strength 0.732>install 0.545, D-all+row0 -97% vs row-1 control null; probe 1: splice arms 0.989/0.988 at 183 — E120 verdict was instrument blindness; T075 retired, W010 killed, W009 bounded) |
| e139 | row-0 universality + 183-robustness | DONE 08:15Z (T082: HYBRID site-dominant — row 0 NOT universal; row-183 content ~1000x (site stores); arms generalize 0.6-0.7 novel+val (note d closed); brake absent on splice (re-routing scar); consolidated net reads 0.660 at never-trained 183 via row 0; arm-c RIDER-NULL — dreams sit at the OLD address 33/34, dream replay ERODES below base) |
| e141 | sink-key mechanism battery | DONE 08:05Z (T081: ROLE-ROUTED presence-only — install-restore x0.999 CE-flat; direction-scramble SPARES fact (+4%) while +0.70 CE; removal-class only kills; rows 2-7 cheap; g-12 collapse x0.014 both R nets kills content-keyed; presence >= ~0.38 norm suffices) |
| e143 | error-placement steering | DONE 09:00Z (T084: COMPASS-CAUSAL — committed prediction HELD; NEAR site-stored at 5-13 (0.280) with row-0 at baseline (0.232, PIGGY bar 1.091 dead); FAR hybrid texture — novel-geom 0.249 vs NEAR 0.002; free d_r0@g-12 cell queued) |
| e142 | row-0 at birth census | DONE 09:30Z (T085: ROW-0-ALWAYS 13/13 every dose; address was protocol-made — direct installs row-129-NULL; fresh family rel 1.000; W011c promoted to LAW; dose moves share not presence) |
| e146 | dissociation matrix | DONE 10:25Z — INSTRUMENT-INVALID (T089: self-battery fails at baseline on B43 — foreign gap 0.018 vs 1.0; weak k*=1 occupancy survives; W015 unadjudicable; e156 BLOCKED; REPAIR queued: rerun matrix on the e111 home lineage) |
| e146b | SELF-BATTERY HOME-LINEAGE RERUN (the T089 discriminator) | READY (CPU eval-only; e111 nets + battery on disk) | same matrix on the e111 home nets. Bars: BATTERY-HOME-VALID = foreign gap >= 1.0 at baseline there (self-recognition is lineage-indexed — B43 genuinely lacks the margin); BATTERY-MISCALIBRATED = fails on home too under matched conventions (recalibrate; W015 stays parked) |
| e150 | flat-CE route test | DONE 09:35Z (T086: ALL-KILLS-WRECK — no flat-CE fact-kill; the kill is POISONING (mask spares fact at CE +0.03; threshold (0.07,0.15)); reframe activated: consolidation = removable->irremovable, riding organism-critical sink health; taxonomy renamed SINK-COUPLED; L0H3-zero 58.6%@+0.21 near-miss = leading fact-circuit candidate; W014 survives with read-horizon texture) |
| e147 | width ladder | DONE 09:50Z (T087: TEXTURE = CLIFF — any variance (±1) kills address key (A: +0.327→-0.029) and births novel-geometry expression (NR: 0.071→0.696), no width trend; INVARIANCE-CAUSAL failed as graded form, returns as step-function; FAR-ROUTED-TAIL fires (×0.069); locked replay CONTRACTS reach below root) |
| e152 | THE CONVERSION TIME-TRACE (T088's central open number: where in 8-300 steps does the phase convert? does a transient two-door state exist?) | DISPATCHED 10:10Z (GPU free; short trainings) | from e131_consolidated: locked re-teach at 183 for steps in {8, 16, 32, 64, 128, 300} (checkpoints each); measure g-12/g+12 retention, 183-site content, A(129), D-all per checkpoint. Bars: CLEAN-CONVERSION = g-12 monotone decay, no plateau (site growth anti-correlates, Spearman <= -0.8); TRANSIENT-TWO-DOOR = a checkpoint where BOTH site content clears the bar AND g-12 retention >= 0.5, followed by decay (the cliff has a dwell time; P-b was EARLY, not wrong); DELAYED-CONVERSION = g-12 flat until a step threshold then cliff (a critical mass of zero-variance steps) |
| e158 | THE 2x2 COMPLETION (R46 critic attack 1 — THE missing cell: variance x site) | TOP PRIORITY — next GPU slot (before e154) | (a) JITTER@183: e151's protocol verbatim with ±1-8 offsets around 183 on the consolidated root — phase claim predicts geometry door STAYS OPEN (g-12 >= 0.5); placement-closure predicts it shuts (~0.1). (b) LOCKED@BAND: locked replay at the ORIGINAL band on the consolidated root — isolates zero-variance from new-site. Bars: PHASE-BY-VARIANCE = (a) open + (b) shut; CLOSURE-BY-PLACEMENT = (a) shut; SITE-INDEPENDENT = (b) stays open (locking the home site is harmless — the door keys on novel-site teaching) |
| e159 | COUPLED-OR-ORGANISM (R46 critic attack 2 — the two cheap reconcilers) | TOP PRIORITY — CPU eval-only, minutes | (a) MASK+LADDER JOINT: norm 0.07 UNDER the mask — heals (poisoning carried by attention reads; information sneaks back through the health door) or still kills (query-side global-softmax collapse; the reframe hardens). (b) SITE-STORED LADDER: norm ladder on e131_arm_b — dies at the same bracket (sink-poisoning is organism death; 'COUPLED' is a misnomer) or survives (coupling is real) |
| e160 | THE HEAD-SET ESCALATION (R46 critic's final line — one cell from the paper's best figure or its retraction) | DISPATCHED 10:45Z (CPU eval-only) | L0H3-zero was single-head 58.6% @ CE +0.21 (1.4pts under bar). Escalate: L0H3 + top-2 and top-3 fact-specific heads (e133's list, CE<=0.35 class), zero and mean-replace, singly and jointly, graded; co-measure CE + base skills per cell. Bars: FLAT-CE-FACT-KILL = some head-set >=60% fact drop at CE <= +0.35 (the surgical surface EXISTS — paper's best figure; the removable->irremovable sentence inverts to 'attackable by head coordinates'); NEAR-MISS-CONFIRMED = best set 40-60% (the frontier stays open); WRECK-ONLY = every >=60% cell costs CE >= +0.70 (the reframe holds; noun architecture survives bounded) |
| e147R | E147 RE-SEEDS (the A-dial error bar; ~10 min GPU) | QUEUED | 2-3 ARM_SEED replicates at w1 and w64: put error bars on the w1 crossing and the plateau; W010's-ghost check on the NR falloff (0.903->0.605) |
| e153 | PHASE-SWITCH SURGERY (R46 ideator top pick — the order parameter's identity; T088's 'why global') | DISPATCHED ~10:26Z (CPU eval-only, low threads; stamp led dispatch by ~10 min — noted per R46 audit) | e151_twodoor vs e131_consolidated differ by 300 locked steps — diff heads both ways (param-delta rank + re-run e133 census on the e151 net), transplant top-K head-sets (K=1,3,6; L0H3-class + twin-L3H4-class named arms) consolidated->e151 and e151->consolidated, eval-only. Readout: phase dial vector (g-12/g+12, D-all, D-183, A(129)). Bars: PHASE-IN-HEADS = some K<=6 transplant reopens the geometry door (g-12 >= 0.45) and/or reverse closes it (>=70% drop), CE priced per cell; PHASE-DISTRIBUTED = no head-set moves g-12 > 20% (order parameter in LN/MLP-stream state -> aims e135). Rider: e152 dwell checkpoint if TRANSIENT fires |
| e154 | TWO FACTS, ONE DOOR (rewrites e134; decides the day's headline — was e151's closure GLOBAL or self-conversion?) | QUEUED — first GPU slot after e152 | root e131_consolidated (F1 sink-coupled); install NEW fact F2 locked at a fresh site (~rows 60-70), 300 steps; measure F1's phase dials + trained-geometry expression. Bars: GLOBAL-PHASE = F1 g-12 drops >=70% while F1's own expression >=0.5 (door closed on an untouched fact); PER-MEMORY = F1 g-12 within 15% (e151 was self-conversion — the biggest correction since E120; abstract rewrites); CROSSOVER = 30-70% (shared substrate with capacity -> W012 contended-bandwidth). Rider: natural-install F2 arm (W016b) |
| e155 | THE HYSTERESIS LOOP (branch 2 of the phase diagram) | QUEUED — GPU after e154 | on e151_twodoor: jitter F1 (e113 ±8) with the same six budgets as e152; overlay both traces. Bars: SYMMETRIC = crossings in the same step-bin; HYSTERESIS-HARD = opening lags >=2 bins (graft pins the policy — W006 echo); PRIMED = opening leads >=1 bin (W010's ghost) |
| e156 | THE SELF ACROSS THE PHASE FLIP (e146 follow-up fork, pre-registered) | QUEUED — rides e146's rig after its gates pass | e146's self-battery on e151_twodoor vs consolidated (nets differ only by the phase conversion). Fork: if SELF-CONSTITUTIONAL -> predict SELF-INVARIANT (identity upstream of the policy); if SELF-ROUTED -> SELF-MOVES-WITH-PHASE; if SELF-TENANT -> e153's transplant arms localize the self; if SELF-INDEPENDENT -> depth-resolved V-source census |
| e157 | CONVERSION REPLICATION, second lineage (paper debt #1; absorbs e145-narrowed) | QUEUED — GPU | e098 s4305 -> jitter (e113) -> locked re-teach (e151 protocol); same dials. CONVERSION-REPLICATES = g-12 >=70% drop + site growth + CE non-worse on family 2; NO-CONVERSION = family boundary |
| e148 | DREAM-TOPOLOGY CENSUS with randomized harvest (the dream-address confound must be discharged before the claim travels) | READY (CPU ~30-60 min; after e142) | regenerate matched dream samples from 7 saved nets (twin/L@150/R@150/R@300/consolidated/L-cycled/site-stored) with RANDOMIZED prompt lengths 60-200; census name-onset read positions. Bars: ADDRESS-SEEKING = install/locked nets concentrate onsets near 129 >=5x uniform (p<0.01); ROUTE-DISSOLVES-ADDRESS = routed nets flat <2x; ROUTE-KEEPS-AN-ADDRESS = significant mode in a routed net (new object) |
| e149 | SCAR SURGERY on row 129 (W006's brake mechanism; post-e150 reframe applied) | READY (CPU eval-only ~10-20 min) | restore install-phase wpe[129] into routed nets (R@150/R@300/consolidated), measure brake before/after; co-measure cos(dwp129, install fact direction) + attention mass at readout; E/L as brake-negative controls. [e150/e147 REFRAME APPLIED: (a) POWER PRE-CHECK — measure norm(consolidated wpe[129]) vs norm(install wpe[129]) FIRST; if within ~5%, the restore probe is a no-op-by-norm (e141's lesson) and the informative arms are the anti-alignment read + the mask; (b) NEGATIVE-POSTERIOR PREDICTION (T087 addendum): the brake is the address key's demotion to below-prior predictor — cos(Δwpe[129], install fact direction) should be ANTI-aligned (<= -0.3) in variance-trained nets ONLY (L/R), not in locked/erase nets (which STRENGTHEN the key, A +0.327/+0.280); (c) the 'route' noun is retired — this is now purely the brake-mechanism experiment]. Add e151_twodoor as the PHASE CONTROL (T088(4): anti-alignment predicted only in variance-phase nets — the site-phase net should show NONE). Bars: SCAR-IN-CONTENT = restore removes >=50% of brake AND anti-alignment cos<=-0.3 in variance-trained nets only; SCAR-IN-POLICY = brake persists under full content restore (suppression lives in readout weights/attention — the negative posterior realized in softmax, not in the row) |
| e144 | [PARKED per R46: bars written under the dead re-keyed noun; e141's no-op-by-norm predicts null — re-aim only at the surviving COMPENSATION bar if ever run] FROZEN-SINK INSTALL (interventional twin of e142; sequence after e142 reads) | QUEUED | install with wpe row 0 gradient-masked; COMPENSATION = decision-row content >=1.5x family norm; SINK-ATTRACTOR = post-jitter key lands on virgin row 0 anyway; ALTERNATE-HUB = key lands elsewhere |
| e145 | HUB ACROSS FAMILIES (after e139) | QUEUED | jitter-consolidate on e098 s4305/s4308 + e082 B43 (+ R43 optional); HUB-UNIVERSAL = >=3/3 row-0-keyed; FAMILY-MODULATED = strength order matches T069's install-time row-0 share (4308>4305>B43); DIVERGENT = any family keys elsewhere |
| e140 | route-dependence trace | DONE 08:40Z (T083: GRADIENT-VOLUME — T079 dead on its dial, ratio 0.745; trained-geometry presence dial SATURATES (twin rel 0.98 — measures sink-load not routing); L-CYCLED retires T078's digs-in (thinning = cycle damage); row-129 texture: L/E strengthen address key, R NEGATES it — T079's unregistered signature) |
| e132 | the wiring trace | DEMOTED-OPTIONAL (T078/R44: e140 answers its kernel question eval-only; next TRAINING slot goes to e143 error-steering, the causal test) | was: one 300-step jitter replay, ~10 dense checkpoints, three dials: kernel-motion cos (e107 instrument), brake delta (e115), k=7 self-acceptance of t0 field; D-all survival per checkpoint. Registered: kernel-fact cos >=2x step-0 by step 100, plateaued by 200; brake crosses +0.05 only post-plateau; D-all survival monotone (rho>=0.8). Falsified by brake-leading-kernel, D-all jump, or self rekeying at the wiring event. Also delivers the critic's mid-replay stage x depth cell |
| e133 | field anatomy census | DONE 07:45Z (T080: TEXTURE — all three nets body-stored (site-locked keeps 7.9% at 183); fact-specific residue HEAD-dominated 84.5% (L0H3); additivity FAILS 0.785 vs 1.98 (population confirmed organ-level); twin's L3H4 address reader dismantled by graduation; no sink-attention head — row-0 route is a VALUE channel; W005-alt killed, ROUTE-ONLY dead) |
| e134 | [STALE per R46 — rewritten as e154: the share-grid is a graft instrument (W012-2nd), phase dials primary] two facts | SUPERSEDED by e154 | ADDITIVE: F1's r*(k).k within per-net band (max/min<1.2) with F2 consolidated; SHARED: >=25% drop. Riders: F2-at-F1's-old-address interference; brake-tag fact-vs-address specificity (T067/T071) |
| e135 | LN-causality variant (W001/W007 deferred test) | QUEUED | norm-variant twin: share product destabilizes (max/min>2) or collapse boundary vanishes => LN CAUSES the field physics; within 1.2 => LN decorative |
| e136 | dream-protection decomposition (e121's sole positive self-effect) | QUEUED | surprisal-carried: matched lowest-surprisal corpus reproduces within 1.5x; generator-carried: dreams protect >=3x at matched loss. [R44 pre-registration: protection scales with fact-token SURPRISAL mass, not self-generation — the frame's extension: error-presence pins, error-absence frees; the 500x number is otherwise unreached by the row-0 frame] |
| e137 | RMU rewiring speed (bridges edit-law x consolidation) | QUEUED | restoration <=150 steps (falsified if >=250). [R44 re-scope, W008 language barred: measure whether RMU leaves the ROW-0 KEY intact and whether restoration regrows it — 'adapters dormant' framing retired] |
| e138 | adapter head-start | RETIRED (R44: premise false — e120's row-183 net expresses 0.99 at 183; the 'unconnected adapter' never existed) |
| e119 | migration head-to-head | DONE 07:20Z (T078: AMBIGUOUS-lean-DIFFERENT-STORES; brake dissociation clean, R +0.210 feeds vs E -0.267 suppresses; jitter migrates/erasure tightens; all 3 pre-regs fired; T073 erasure-migration inverted; brake = re-keying scar) |
| e122 | self-at-distance (P6) | READY | does the anchor accept the same net's field from another run/window — generator-self vs episode-self. [R43 bar pre-committed: same-run-different-window MUST stay healthy; if it also collapses, the self/episode framing dies — the anchor is content-specific — and the card must say so] |
| e123 | self-drift curve (P6) | READY | identity half-life across checkpoints. [R44: 'rekeying probe' clause moot — e119/e131 answered re-keying; drift-vs-output-similarity bar stands] [R43 bar pre-committed: anchor-acceptance(t, delta) must decay DIFFERENTLY (slower or different shape) than trivial output-similarity JS between checkpoint samples — rule 7a control; if they coincide the card reads 'identity drift = behavior drift'] |
| e125 | attack the moved fact (P7; three surfaces; W014 mechanism ordering PRE-REGISTERED ~08:00Z: heads > route >> band) | READY | surfaces: (a) row-0 mean-replacement (fact -97% known — price collateral CE/base-skill/held-30); (b) band deletion (expected ~null or BRAKE +0.210 — an 'unlearning' move that STRENGTHENS the routed memory); (c) fact-specific readout heads (e133's L0H3 class) — predicted to remove the fact with LOW collateral (0.46 drop at 0.21 CE was single-head; test combined fact-specific head set). Bars (R43 collateral-matched + W014): HEADS-WIN = head-set removes fact >=60% at collateral <=25% of route-level's; ROUTE-ONLY-EFFECTIVE = no head-set reaches 60% fact drop at any collateral; BRAKE-TRAP = band deletion raises fact expression >=+0.1. Removability ratio vs pre-consolidation fact at matched collateral (e043 asymmetry reference) || e128 | inversion census (P8) | READY | dp27 promoted to census — do read-rule inversions cluster into a second mode? per-net rate as a fingerprint. [R43 bar pre-committed: e095 Monte-Carlo null guard on apparent clustering — a diffuse distribution reads as 'a second mode' by eye] |

## Parking lot (raw ideas, unranked)

- e037 forget-then-graft: graft-suite + ΔW atlas on a projectedly-forgotten net — fluency substrate vs stream basis
- e036 retrieval-head transplant: does L4-H1 carry its decision mode into another net?
- v010 synthesis poster: three tasks × {lesion, depth, write schedule, funnel} — the lab's first self-portrait
- e039 reconsolidation on the task net (retrieval-gated labilization with a real circuit)
- e040 graft-evolution: select lineages by graft damage; is init-anchoring evolvable?
- e005s mini-ladder 0.7M/2.7M/8M × 2 seeds (GATED on mechanism card T010)
- e031-alternative control: spectrum-matched random W_out (LN-statistics rescue test)
- e023 entity-granularity forgetting; e024/e026 bio/evolution raw ideas

- e032 MLP-5 per-token census: write norm vs entropy/decision depth (energy pump or ballast?)
- e033 write-equalizer (bio/homeostasis): train with equal-norm MLP writes; who absorbs the energy?
- e034 graft-evolution: lineage selected by cross-seed graft damage; does selection erode init-anchoring?

- e021 task-swap: copy-task vs word-shuffled vs Shakespeare — is front-loading task-dependent?
- e023 entity-granularity forgetting: anchored ascent on one name's windows vs embedding+lm_head row surgery
- e024 reconsolidation window (bio-analogue): retrieval-gated labilization then matched-dose noise vs control
- e026 selection-on-depth (evolution): lineages selected under L5-ablated loss; is depth selectable?
- dual-culture nets (successor if e003b refutes selective forgetting at all granularities)
- replay buffers vs catastrophic forgetting; sleep replay; curriculum scars; lottery tickets; scar tissue (retrain after unlearning)

## Parking-lot promotions policy
Reviews promote at most 1-3 items to READY; ideas that discriminate live
hypotheses (THINKING.md) outrank new topics. Replication debt outranks new
lines when a load-bearing claim is single-seed.

| ID | experiment | status | notes |
|---|---|---|---|
| e108 | distance-ladder anchor | DONE (T060: SHARP FAMILY — binary self-recognition; V-cos two-cluster step 0.40/0.14; output-near-identical donor still collapses — internal geometry, not behavior) |
| e111 | V-manifold self-signature | DONE (T061: k*=7 — selfhood is a 7-dim readout; top-1 PC 11.7x; the fixed-point stamp found) |
| e112 | signature forgery | DONE (T062: NOT FORGEABLE — stamp necessary not sufficient; the anchor reads full joint V-structure holographically; identity is process not summary) |
| e109 | consolidation test | DONE (T064: CONFIRMED — jittered 0.909 through deleted coordinate; diversity 2x mass) |
| e113 | all-addresses deletion | DONE (T065: fact survives all-five-address zeroing; D0129 was scaffold loss) [R44 marker: 'BODY-STORED'/'developmental stage' INVERTED by T077 — re-keyed to row 0; survival because D-all never deletes row 0] |
| e114 | brake signatures | DONE (T067: all three NULL — brake is coordinate-local, install+geometry-specific, strongest where field weakest; fourth story named) |
| e118 | standardization control | DONE (T072: SURVIVES — 2.05x/2.80x post-z; anisotropy is family-typed; Timkey-strictest 1.68x recorded; claims hardened) |
| e115 | graded ablation | DONE (T071: SAFE — brake sign flips with field strength; address = content when weak, suppressor when strong; M3 0/3) |

| e116 | re-barred census | DONE (T069: 3/6 — graduation denied; row-0 duality exposed; final form: structure 6/6, concentration family-dependent) |
| e117 | maturity point | DONE (T070: BETWEEN — constant is a per-net idiosyncrasy, 31/42/54/77 non-monotone in steps and CE; the FORM is the law; maturity question closed negative) |
