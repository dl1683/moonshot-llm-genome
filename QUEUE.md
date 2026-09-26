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
| e059 | winner differencing | zero-GPU ΔW audit of e040 winners vs unselected sibs — what did compatibility-without-alignment exploit? |
| e063 | load homeostasis | zero-GPU A-setpoint tracking across the 11 ckpts — is organ-reliance a defended init-anchored trait? |
| e056 | healed-host graft | ablate L3-MLP, heal to parity, graft donor — host-fragile-organ vs donor-basis-fit |
| e060 | residual-selection lineage | e040 rerun with A-residualized damage (T026's method note) |
| e062 | subspace-cosine predictor | pre-graft stream-cosine as cheap compatibility predictor |
| e061 | calibration rescue | scalar/gain nudges at e055's suppression depth vs full transplant |

## Day-4 programs (ideator harvest 2026-09-26; memo: scratch/post_paper_programs.md)

P1 COORDINATE (top pick) | P2 IMMUNOLOGY | P3 CACHE WEATHER | P4 THE ERASER (wildcard, gated on P1 census). Numbering fixed: e065 = RMU-vs-surgery head-to-head (design: scratch/e065_rmu_headtohead_design.md, under critic review); ideator's P1 ramp renumbered below.

| id | experiment | status | one-liner |
|---|---|---|---|
| e066 | close the loop (P1) | READY (zero-GPU, minutes) | cos(e056b mean-donor relay dir at d5, wpe-130 row) from cached trajectories — registered: \|cos\| ≥ 0.4 closes position-row→state-direction loop; ≈0 = two distinct "addresses" |
| e067 | address census (P1) | READY (eval-only) | single-row wpe perturbation × install battery, 0.84M + 2.7M — sparse (≤5 rows carry ≥80%) vs dense code |
| e068 | rebinding surgery (P1) | READY (eval-only) | copy wpe-130 row to position k / two-row swaps — durable expression at k ⇒ portable row; failure ⇒ downstream distributed key |
| e065 | RMU-vs-surgery head-to-head | DESIGN (critic reviewing) | obfuscation inversion: rescuable-but-reverts-fast vs unrescuable-but-scarred; GPU-gated behind e053c |
| — | P2 ramp e060/e062/e063 | RUNNING (agents) | then trained-tolerance + crossmatch grid at promotion |
| — | P3 settle + junk split | GATED (e053c verdict) | mass-stealers vs poison; sleeper hypothesis: self-generated entries dominate hurt set |

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
