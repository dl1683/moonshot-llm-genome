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
| e065 | RMU-vs-surgery head-to-head | READY-GATED (design critic-hardened 5f11a79; GPU free) | obfuscation inversion: rescuable-but-reverts-fast vs unrescuable-but-scarred; 5 arms incl. retain-only + no-removal controls |
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
| e082 | cross-seed row-129 transplant (ideator #1) | READY (needs one 100-step install on B43; then eval-only) | "a one-row organ" — seed-42 address row into seed-43 install; A2 p(Z)>=0.30 = portable organ; + crossmatch-cosine overlay + T043 Z-rank rider |
| e083 | canalization cycle 3 (ideator #2) | READY (~100 steps total) | T037's registered monotone-closure prediction; ratio >=1 AND cos >=0.6 or canalization falsified |
| e084 | READ-KERNEL census | DONE (T050: KERNEL=SHADOW r 0.918 — dissociation dead, shadow claims validated; young-band 31% flips; 24-27% escapes outside top-5; 48 donor-continuation hits) |
| e085 | anchor-description probe | DONE (T051: anchor not entry-describable — no property fires; T048 rider r(static,dyn)=-0.004; anchors more run-specific) |
| e088 | pair-level anchor probe | DONE (T051: SUB-ADDITIVE 0.464 — interactions killed; anchor is MASS-ACTION) |
| e089 | mass-response curve | DONE (T051 FINAL: MASS-ACTION confirmed — both threshold clauses fire; variance collapse; threshold dose-response law) |
| e090 | dose-vs-schedule decomposition | PARKED | K=1 continuous removal ~8x K=32 lumps — mass vs removal-schedule |

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
