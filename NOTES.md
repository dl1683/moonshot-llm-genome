# Lab Notebook

Append-only. Newest entries at the top. Format per experiment:

```
## E### — name (date)
WHAT WE DID / WHAT WE SAW / WHAT'S NEXT
```

---

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
