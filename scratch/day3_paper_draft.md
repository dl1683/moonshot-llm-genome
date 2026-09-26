# Paper Skeleton — P-A submission draft, Day 3 (STRATEGIST, 2026-09-25)

Status: SKELETON for the actual submission-shaped draft, per novelty_inventory
P-A "double-down" verdict and the completed expression-gap arc (T028-T035).
Not the day report. Every quantitative claim below is traceable to a run
(e023/e042/e043/e044/e048/e053/e055/e056b/e056c) or an audit annotation; every
caveat is carried verbatim from the T033 audit into Limitations, with the
circularity item resolved by e056b and refined by e056c (T034/T035) into the
3c/3d claim-split. Target venue class: mech-interp workshop → main
conference. Citations pass applied 2026-09-25: the 29-entry verified
bibliography (scratch/paper_bibliography.md) is wired in-text and
reproduced in References; six camera-ready citation fixes applied.

---

# 1. Title + Abstract

## Title

**Elicitation Failure in Small Language Models: Causal Localization of
Knowledge That Cannot Surface**

(subtitle option: *The four faculties of installed knowledge in a 2.7M
character transformer*)

## Abstract (~175 words)

Language models can score near-perfectly on continuation tests of a fact
while never producing it in free generation. We dissect this elicitation
failure in a 2.7M-parameter character-level transformer into four faculties —
address, ability, expression, and history — using registered, causally
controlled experiments. Installed knowledge (92–97% continuation-battery
accuracy) never surfaces in free generation outside the install geometry (0 of 2,800 chars off-geometry; 49/60 full expressions from battery geometry) across
dose, temperature, and seeding arms. The installed address is bound to
training position, not content: a one-character context shift collapses
expression probability (0.556→0.12), and the correct token persists
sub-argmax throughout. At the bound position, transplanting the model's own
teacher-forced residual state durably rescues expression (p 0.374 vs
shuffled 0.000; one-shot rescue that propagates to generated text), with a
mid-stack threshold at depth 4 of 6 and a destruction–re-emergence
structure (block-0 peak, block-2 crash). At non-onset positions the same
write is a transient logit artifact — floor by +2 tokens, zero recurrent
Z-words in 288 continuations: the knowledge is position-bound, and so are
its suppression and rescue. Findings are from a single model family of
≤2.7M parameters; terminal sites are pseudo-replicated.

*(Word count: ~175. The honesty sentences are load-bearing; do not trim
them.)*

---

# 2. Introduction

## 2.1 The probe-vs-generation gap

- Opening fact, established at 7B scale by prior work: internal
  representations encode knowledge that generation does not express.
  - **Orgad et al. (ICLR 2025, arXiv:2410.02707)** — probes on Mistral-7B /
    Llama3-8B read correct-answer information off exact-answer-token
    activations (AUC 0.85–0.95) even when the model consistently generates
    wrong answers; their C2/D/E1 resample-selection cells win 30–40 points.
    This is the **probe-level prior**: the gap exists, is measurable, and —
    critically — is left causally untouched ("not proposed here as an error
    mitigation strategy but rather as a diagnostic tool"). Their error-type
    prediction is weakest precisely on the expression-gap populations
    (AUC 0.59–0.68 for consistently-wrong / two-competing-answers).
  - **Buckmann, Nguyen & Hill (arXiv:2505.08662)** — linear probes on hidden
    states impute economic facts better than the model's own text outputs;
    they coin the term **"elicitation failure,"** which we adopt (the term is
    unclaimed for this use).
  - **Luo, Chu, He, Wang, Qin, Wu & Chen, "You Only Pass Once: Answering
    and Abstaining Together in a Single Forward Pass of a Frozen Language
    Model" (arXiv:2608.14465)** [YOPO] — the closest prior in the adjacent
    domain: on Qwen2.5-1.5B/3B/7B sufficiency detection, the residual
    stream encodes "context is sufficient" while generation fabricates
    anyway; a learned difference-of-means direction separates 124/125, and
    "relay steering" pushes the internal verdict into speech (fidelity 1.0),
    with causal onset at layer 19 and overwrite interference at mid-stack
    (the layer-19 and 124/125 specifics are v1 numbers). **Domain:
    abstention, not factual recall; method: learned direction, not
    own-state transplant; goal: elicitation, not depth-localizing
    suppression.** We cite them proactively and scope accordingly.
- The unclaimed question: *where, in depth, does installed factual knowledge
  stop being expressible — and is it destroyed, or suppressed?* No
  interventional study of factual-recall expression exists (lit-scan
  verdict, softened from "zero interventional studies anywhere" per T028).
  Neighbors to cite-and-distinguish: Yan & Jia (EMNLP 2025 Main,
  arXiv:2502.20475; promote-then-suppress circuits exist for enumeration
  repetition); ITI (Li et al., NeurIPS 2023, arXiv:2306.03341), DoLa
  (Chuang et al., ICLR 2024, arXiv:2309.03883), and contrastive decoding
  (Li et al., ACL 2023, arXiv:2210.15097) — all *assume* late-layer
  pollution of early-layer facts; none measure where knowledge dies; DoLa's
  layer choice is per-token automatic, not anatomical. (The published
  contrastive-decoding critique — "The Mirage of Performance Gains,"
  arXiv:2504.10020 — targets *MLLM object hallucination* and attributes
  the gains to MCQ-format artifacts; it is adjacent to, not a rebuttal of,
  DoLa. The "assumes, never measures" point is ours.) Bürger, Hamprecht &
  Nadler (NeurIPS 2024, arXiv:2407.12831; "Truth is Universal" — a truth
  direction supports robust lie detection even in models instructed to
  lie: probe-level and read-only; we do not read it as a claim that RLHF
  widens the elicitation gap). Cundy & Gleave (FAR AI), "Preference
  Learning with Lie Detectors can Induce Honesty or Evasion"
  (arXiv:2505.13787; NeurIPS 2025 — training against probes yields
  evasion: probe-visible knowledge is driven underground, not deleted —
  matters for our "the knowledge is really in there" claim).

## 2.2 Why a 2.7M char-LM

- The gap is behavioral and reproduces at 1.5B–8B (Orgad; YOPO); what is missing
  is not scale but *controls*: shuffled donors, base-net twins, pad-shifted
  position cues, one-shot-vs-held write semantics, full depth-survival
  curves per site. At 2.7M/6 layers these are minutes-per-cell, so the
  experiment the 7B literature implies becomes executable with n=24 sites
  instead of a case study.
- Honest framing: this paper is the interventional protocol plus the first
  depth-localized causal account, at toy scale, with the scale-replication
  path stated explicitly (Limitations).

## 2.3 Paper roadmap (one sentence per results section)

Surgical removal (5.1) → cheap-but-silent installation (5.2) → positional
binding and the off-geometry artifact (5.3) → causal state-rescue and the
depth structure (5.4) → adjacent cache-utility context (5.5).

---

# 3. Contributions (numbered)

1. **The four-faculty decomposition of knowledge editing** — ADDRESS
   (concentrated in I/O row coordinates; surgically removable, ~500×
   selectivity, scale-invariant 0.84M/2.7M/10M) / ABILITY (distributed,
   train-only; no parametric install reaches the bar) / EXPRESSION
   (teacher-forcing-bound; battery 92–96% with zero free-generation
   occurrence across dose ×3, temperature ×3, seeding) / HISTORY (re-learned
   memory regrows the original address direction, cos 0.760 vs 0.278 fresh,
   replicated at a second seed on the B43 net — e044b: cos 0.728 vs 0.243,
   2.99×; re-learn 2.92× slower — and ~3× more surgical-resistant; the
   route-flip/key-resistance sub-readouts remain n=1, flagged). With
   surgical evidence from e023/e042/e043/e044/e044b/e048.
2. **Positional (wpe) binding of installed knowledge** — the installed
   address is bound to an absolute position (wpe-130), not content: one-char
   shift 129→131 collapses p(Z) 0.556→0.12 and kills argmax; left-padding at
   fixed content collapses identically. The previously reported
   "zero expression" cell was an **off-geometry artifact** (probed 10
   positions off; generating *from* battery geometry expresses greedily
   49/60) — the install worked; the probe was wrong. The correct token is
   sub-argmax everywhere (rank-2, p 0.167–0.234 at onset; rank-3,
   p 0.004–0.007 at deep sites vs floor 2e-8).
3. **Causal state-rescue at depth 4 — split by the e056b/e056c
   discriminator into two claims:**
   - **3c (position-specific durable rescue at onset sites):** transplanting
     the model's own teacher-forced residual state at the free-run divergence
     token (an install-geometry onset) rescues expression *durably*: p(Z)
     0.374 (d4) / 0.494 (d5) vs shuffled donors ≈ 0.000 everywhere; the
     rescue is ~60× its base-net-twin control (0.0062); pad-shifted donors
     cap at 0.133 (position-cue leak excluded); one-shot semantics genuine
     (25/32 vs base 6/32) — the write survives the model's own dynamics; 32
     downstream Z-word rows show rescued states propagate to *generated*
     text, not just next-token probability. The interventional
     factual-recall study Orgad et al. lack.
   - **3d (the transient-injection negative at non-onset positions — itself
     a finding):** the same d4 write at 24 random non-onset positions blips
     p(Z) to 0.509 (argmax flips 24/24) then reverts to the control floor by
     +2 tokens (median 3.3e-6 vs floor 6.8e-7); ZERO recurrent Z-words in
     288 donor continuations — the only ZEPHYRA-like outputs are offset-0
     speaker-tag completions that die at the colon. The write is
     knowledge-specific (first-Z 49/96 donor rows vs 0/96 shuffled and
     0/96 base) yet cannot steer even 2 tokens ahead: an address, not a
     general steering direction.
   - **3e (donor construction: own-state vs relay-direction — the YOPO
     contrast run in-house):** averaging the onset-site donor states into a
     single mean-donor "relay direction" beats every individual donor at
     the rescue peak (d5: 0.912 vs 0.494 individual-donor mean; best
     single donor 0.826) — averaging denoises toward the address
     direction. The intervention therefore has two arms: the own-state
     transplant (one specific state at one named token — the conservative
     arm, and our primary claim) and the mean-donor relay direction (the
     YOPO-adjacent arm, a difference-of-means-style direction in the
     spirit of Luo et al.'s relay steering, arXiv:2608.14465). e056c binds
     both: away from onset positions neither construction steers beyond
     +2 tokens — durable rescue is position-bound whichever donor you
     build.
4. **The d1-peak/d2-crash destruction–re-emergence structure** — at
   off-geometry sites the address survives block-0 output (d1 peak), is
   destroyed across blocks 1→2 (crash), and becomes re-injectable from
   depth 4: suppression, not erasure, with a mid-stack causal locus
   (d*=4 of 6, inside the registered {2,3,4} window; d1/d2 ratios 4.6–76×
   at 9/10 deep sites).
5. **The negative-utility cache findings as adjacent context** — the same
   free-run causal apparatus yields, to our knowledge, the first
   per-position causal KV-cache utility curve: spike+plateau shape (~85–95% of ctx-256 cache is dead
   weight), sink dead at generation 5/5, 13–20% (32% at 10M) of old entries
   have *negative* utility (lesion improves), onset instrument-dependent
   (registered-threshold a* = 3–86; open conflict between statistics
   documented). Positions the elicitation findings inside a broader
   "what the free-running stream actually uses" story.

---

# 4. Methods: the tiny-lab protocol

## 4.1 Model and corpus

6-layer character-level transformer, 2.7M parameters (B-family), trained on
Shakespeare; scale ladder 0.84M/2.7M/10M available for the invariance
claims. Every experimental step ≤30 min on one laptop GPU; harness with
registered gates.

## 4.2 Registered predictions

- Every experiment ships a design memo (scratch/) with pre-registered
  predictions and pass/fail gates (G0–G6 pattern) **before** the run;
  verdicts are read against the registered bars, never re-benched
  post-hoc. Example: e055 registered the d* window {2,3,4} before the
  depth-survival sweep; the result d*=4 lands inside it.
- Thinking-gate: every result gets an interpretation entry (≥2 alternative
  explanations, a discriminating observation, a registered prediction)
  before any follow-up experiment can launch.

## 4.3 Measured-probe designs

Design memos include *measured probes* — small diagnostics run before the
main experiment whose outputs are themselves findings (and which caught the
off-geometry artifact, T032). Pre-registration extends to the
discrimination metric (adopted after the lit scan flagged Orgad's
error-type AUCs of 0.59–0.68 as the reviewers' target).

## 4.4 The honesty battery

- Continuation battery (teacher-forced NLL/acc on the installed fact) vs
  **free-generation probe** (occurrences of the installed token/name in
  sampled text). Batteries alone overstate install by 3.4×; free
  generation is the check. (External control: probes themselves disagree —
  Zhao et al., "Do We Know What LLMs Don't Know? A Study of Consistency in
  Knowledge Probing," arXiv:2505.21701, preprint, no venue listed, report
  intra-method probe agreement ~40% and cross-method consistency as low as
  7% — our battery-vs-generation split is the instrument-disagreement
  check, moved upstream of any probe claim.)
- Sub-argmax rank/prior logging at every diagnostic position; completion
  given the first token measured separately (TF-completion given 'Z' ≈ 1.00
  — the entire gap lives at the onset choice).

## 4.5 Transplant and control suite (e055 core)

- Own-state transplant: teacher-forced residual stream at the onset
  (divergence) position written into the free run, swept over depth d1–d6,
  24 sites; one-shot and held write semantics (patching methodology and
  metrics per Heimersheim & Nanda, arXiv:2309.16042).
- Controls, all pre-registered: shuffled donors (distributional null),
  base-net donors (the uninstalled twin — separates "knowledge" from
  "generic state"), pad-shifted donors (position-cue leak), A-rev
  symmetry, direct800 and e001 reference curves; bootstrap CIs throughout.
- Readouts: R1 next-token p(Z); R2 downstream Z-word occurrences in
  generated text; R3 trajectory-level semantics.

## 4.6 Adversarial audit chain

Every headline claim passed through in-place audit annotations (REVIEWS.md
+ THINKING.md audit brackets): the e043 battery-vs-expression amendment, the
e046 replication demotion of the two-factor erasure, the T032 off-geometry
correction, the T033 22:30Z audit of e055, the registered e056b
circularity control (landed: resolved), and the e056c loud-logit-paste
discriminator that split the rescue claim (T034/T035). The paper reports
post-audit numbers with the caveats attached, not the pre-audit headlines.

---

# 5. Results (mapped to runs)

## 5.1 Removal is surgical: the address faculty (e023, e042; audit: e046)

- e023 — the J-row scalpel: zeroing one rare letter's embedding+lm_head
  rows (384 of 2.7M params) → S_name = 573 (bar 5) at corpus cost +0.0008
  nats; ~4,907× less collateral than entity-ascent at matched damage
  (ascent at the name bar: val +1.16, S_name 1.06 — catastrophic). Bar-2
  erasure missed narrowly (acc 13.6%): surgery damages near-completely, does
  not fully erase.
- e042 — two-factor erasure achieves complete selective forgetting once
  (D2 + L3H5@JULIET-prefix: acc 0.0013, corpus +0.00083 nats, S_name 1,937;
  shared name machinery L0H3 top-1 for all names, atlas ρ 0.965); the
  13.6% post-D2 residual rides mid-network machinery (L3H5, L1-attn), not
  the healthy L0 circuit.
- Audit chain (e046): the completion half does NOT replicate across nets
  (B43 L4H4 vs BDO L3H1 — a seed lottery); the address half survives 5/5
  nets and 3 scales. The paper claims the address half as the replicated
  faculty; the two-factor recipe is reported as n=1.
- Positioning (unlearning framing): representation-level unlearning — RMU
  ("Representation Misdirection," introduced with the WMDP benchmark, Li et
  al., arXiv:2403.03218) and mechanistic localization for robust
  unlearning/editing (Guo et al., ICML 2025, arXiv:2410.12949) — and the
  full-stack microscope evaluation of unlearning methods and metrics (Fan
  et al., arXiv:2510.07626: unlearning must be checked for
  latent-knowledge recovery, because representation-level methods
  obfuscate rather than delete) frame the exact choice this section
  instantiates at toy scale: hide in representation vs remove parameters.
  The address surgery above is the parametric leg; the
  battery-vs-free-generation split is a two-instrument latent-recovery
  check. Big-lab counterpart: knowledge localization for capability
  removal (Shilov et al., arXiv:2512.05648).

## 5.2 Installation is cheap — and constitutionally silent (e043, e048)

- e043 — no parametric install (rows-only arms, copy/delta, wte/lm/both;
  L0-MLP block graft) reaches Bar-I1 at the guard (best NLL 6.43 / acc
  0.055 vs bar 4.17 / 0.50). Amended by full report: **anchored exposure
  installs cheaply** — 7 guarded cells reach Bar-I2; best 0.09 NLL / 0.974
  acc at +0.05 corpus CE, S_install 144–289.
- The discovery inside e043: **0 occurrences of ZEPHYRA in 2,800 generated
  characters at 97% battery accuracy.**
- e048 — the expression zero is invariant: dose ×3, temperature ×3,
  induction-route seeding — expression = 0 in every arm while battery holds
  0.92–0.96. No threshold, no dose response (P2/P3 refuted). Doctrine:
  continuation batteries are not evidence of usable knowledge; batteries
  overstate install 3.4×.
- e044 + e044b (history faculty, supporting; address-direction core n=2):
  re-learning after erasure is 2.08× slower but regrows the original
  address direction (cos 0.760 vs 0.278 fresh); e044b replicates the core
  on the B43 net (seed 43, same frozen install set and Dmix exposure,
  D2-only, compact): cos 0.728 vs 0.243 fresh-from-zero (2.99×, registered
  prediction met) with re-learn 2.92× slower (35 vs 12 steps; e044: 25 vs
  12). The new-route (atlas ρ 0.21; old carrier head flips to
  anti-carrier, −2.03) and key-resistance (44.5% vs 0.13% under D2+patch)
  sub-readouts remain n=1.
- Positioning (editing side effects): RippleEdits (Cohen et al., TACL
  2024, arXiv:2307.12976) shows edits ripple into related facts; MQuAKE
  (Zhong et al., EMNLP 2023, arXiv:2305.14795; the ICLR 2025
  MQuAKE-Remastered is a separate follow-up) shows edits fail to
  propagate multi-hop. The expression zero is the limiting case on the
  propagation axis — an installed fact that fails to surface in free
  generation *at all* (0 of 2,800 chars at 97% battery accuracy) — and
  both benchmarks, being query-based, sit on the battery side of the
  battery-vs-generation split, the side e048 shows overstates.

## 5.3 The gap is positional; the zero was off-geometry (T032 design probes; audits of e048)

- One-char context shift (129→131) collapses p(Z) 0.556→0.12 and kills
  argmax; left-padding with content fixed collapses identically → **wpe-130
  positional binding, not content binding** (resolves T019's open
  content-vs-position edge).
- **Day-4 refinement (n=1, single install, flagged):** a full 256-row
  wpe census + rebinding surgery localizes the address to a SINGLE
  portable row — wpe-129, the decision-position code: copying it alone
  to a shifted window's decision position restores ~70% of expression
  (0.40 vs 0.56 unshifted; pair-with-row-0 adds nothing); its in-place
  perturbation is surgically name-targeted (KL 0.2–0.5) while the
  heavier-perturbing row 0 (window start) is generic scaffolding (KL
  2.6–7.2, distribution-wide; the name dies as collateral). The
  address is the single most load-bearing portable row: ~70% rebind
  by copying it alone (partial sufficiency; a 0.13 plateau remains
  without any copy), partial necessity (row ablation retains ~0.32 of
  the 0.556 expression — the knife-edge belongs to the shift, not the
  row), **n=2 installs** (e078 dose-net replication: same pattern; rebind 60-71%, install-dependent). **Post-scan perimeter (scratch/day4_claims_lit.md):** no
  per-row positional-embedding causal edit found 2023-2026 (nearest is
  TAPE's content-vs-position framing, build-only); scoped to learned
  absolute PEs; distinguished from ROME-family MLP edits, prompt tuning
  (runtime vectors, no necessity/sufficiency bars), and attention-sink
  cache tricks.
- **Closest prior and the perimeter (post-scan wording, per
  scratch/positional_binding_lit.md):** binding IDs (Feng & Steinhardt),
  position-index vectors (Smolensky et al.), and Ordering-ID subspaces
  (Dai et al.) establish content-independent positional indices for
  *in-context* binding — but at *learned-weights* granularity nobody has
  measured knowledge bound to one absolute position embedding, and no
  prior runs the matched-vs-shifted *state-transplant* assay (ROME anchors
  fact recall to the last subject token implicitly and never shifts it;
  Wu et al. treat the residual stream as addressable memory for ICL
  programs, not installed facts). Against the field's position-INVARIANT
  default — function vectors (Todd et al.) port across positions — our
  installed knowledge is position-BOUND: the inversion is the claim.
  Framing note vs long-context work: this is an absolute/associative-
  address failure, not a relative-distance decay (NoLiMa's
  needle-in-last-2K control already excludes distance binding as the
  explanation there; our one-char shift at fixed distance-to-end is the
  complementary dissociation).
- The e048 zero-expression was an artifact of probing 10 positions off the
  install geometry: generating FROM battery geometry expresses (greedy
  49/60 full ZEPHYRA; sampled 10/7,200 chars — the sampled-vs-greedy
  contrast is itself the sub-argmax prior made visible).
- The sub-argmax prior, explicit: Z at rank-2 (p 0.167–0.234 vs argmax E
  0.68–0.78) at the onset choice; rank-3 (p 0.004–0.007 vs floor 2e-8) at
  deep sites. Knowledge present, sub-argmax, everywhere.
- Completion is invulnerable (TF-completion given 'Z' ≈ 1.00): the gap
  lives entirely at the onset choice.
- Mini-transplant pre-result: battery-TF state written at the free-run
  onset lifts p(Z) 0.004→0.716 across depths (d3 0.105 / d4 0.238 /
  d5 0.398 / d6 0.716) with shuffled writes at ~0–0.03 (AUC 1.0) — the
  full run's job was the 24-site curve and the depth structure.

## 5.4 Causal state-rescue and the depth structure (e055; audit chain: T033 22:30Z; follow-ups e056b/e056c landed)

- **P1 confirmed — state-rescue:** 24-site depth-survival transplant;
  TF-state at the onset position rescues p(Z) to 0.374 (d4) and 0.494 (d5)
  vs shuffled ≈ 0.000 everywhere, bootstrap CIs excluding 0. The knowledge
  IS present in the residual stream during free run; the expression gap is
  a present-but-suppressed STATE phenomenon, causally demonstrated.
- **P3 confirmed — d* = 4 of 6, mid-stack:** rescue threshold inside the
  registered {2,3,4} window; d1-peak→d2-crash replicated at deep sites
  (r1_d1/r1_d2 ratio 10.3× at t=298). The address survives block-0, is
  destroyed across blocks 1→2, becomes re-injectable from depth 4.
- **Downstream expression (R2/R3):** 32 nonzero Z-word rows — rescued
  states propagate to actually generated ZEPHYRA words, not just
  next-token probability. [Audit caveat attached: downstream Z-words partly
  onset-flip + mechanical completion.]
- **Audit 22:30Z (report these numbers):** d4 rescue 0.374 is ~60× its
  base-net twin (0.0062); pad-shifted donors cap at 0.133 — position-cue
  leak excluded; one-shot semantics genuine (25/32 vs base 6/32); P3 held
  with margin (9/10 deep sites, ratios 4.6–76×).
- **The causal story, paper-grade:** installed knowledge exists as a
  position-bound (wpe-130) sub-argmax address; free-run destroys it across
  blocks 1→2; a teacher-forced state at the onset position from depth ≥4
  restores expression durably; shuffled states do nothing; and the same
  write at non-onset positions is a transient logit blip, not a steering
  direction (e056c). Elicitation failure is real, localized,
  position-specific, and state-carried.
- **Follow-up landed (e056b, T034): circularity RESOLVED.** The same depth
  curve at 24 random NON-onset floor-prior positions from cached
  trajectories: the d4 rescue fires anywhere (site-mean 0.324 ≥ 0.30,
  AUC 1.000; shuffled 1.2e-7; base-net twin 0.007) — d*=4 is not a
  property of gap-selected sites.
- **Discriminator landed (e056c, T035): off-position firing is a LOUD LOGIT
  PASTE.** Under the frozen R14 rule, the non-onset "rescue" is a one-token
  Z-logit crank: p(Z) 0.509 at +1 (argmax flips 24/24, flip-to-Z 0.46) →
  control floor by +2 (median 3.3e-6 vs floor 6.8e-7) → 5.1e-6 at +10;
  zero recurrent Z-words in 288 donor continuations (the only
  ZEPHYRA-like outputs are 6 offset-0 speaker-tag completions that die at
  the colon); knowledge-specific (first-Z 49/96 donor rows vs 0/96
  shuffled and 0/96 base). The durable rescue is position-specific to
  onset sites; the off-position write is transient injection, not
  steering.
- **Persistence curve (Fig 5, runs/e056c/persistence_curve.png):** mean
  p(Z) vs continuation offset per arm — the donor-write blip at +1
  collapsing onto the shuffled/base floor by +2 and flat through +10 — is
  the visual form of the 3d transient-injection negative.

## 5.5 Adjacent context: what the free-running stream actually uses (e053/e053b; audits T030/T031)

- Context: sink-emergence work asks where attention sinks live (Gu et al.,
  ICLR 2025, arXiv:2410.10781) and how to preserve them in quantized
  caches (KVSink, COLM 2025, arXiv:2508.04257); we ask the inverse,
  generation-time question — what does free generation actually *use* —
  and answer it causally, per position.
- Per-position causal KV-cache utility curve (to our knowledge the first:
   the COLM-25 KVSink line characterizes sinks, not causal per-position
   read utility during free generation): last-~7-token spike
  (+0.4 to +3.2 nats/position) + shoulder (17–32) + near-zero plateau;
  ~85–95% of the ctx-256 cache is dead weight at generation.
- Sink dead at generation in 5/5 cells (sink-lesion dCE 0.0069; decays,
  sometimes sign-flips).
- Negative-utility entries: 13–20% of positions (32% of old positions at
  10M) — lesion IMPROVES the model. **Day-4 source split (e073, 4/4
  nets, flagged n=1-family): the lesion-helpful entries concentrate in
  the model's OWN generated tokens (up to 36.7% beyond-onset at 10M,
  negative mean dCE) while corpus-prompt entries almost never hurt.**
  Positioning (post-scan): the behavioral phenomenon is established —
  hallucination snowballing (Zhang & Press, ICML 2024) and
  imitation-learning error accumulation (Arora et al., AAAI 2022) —
  but neither localizes where the poison sits; the entry-level lesion
  account and the prune-by-source implication are the new part: an
  ENTRY-LEVEL account of exposure bias. **Boundary (e075, registered kill): static lesion utility does NOT predict generation-time prunability — V-zeroing the self-generated dead band mid-run costs +0.26 nats and drives generation into a self-consistent off-manifold attractor (clean-net judged 6.4 nats; the once-harmless prompt band turns junk-heavy behind it). Description ≠ intervention license.** Post-scan positioning (scratch/trajectory_anchor_lit.md): the trajectory-anchor conjunction is unclaimed in 2018-2026 — the nearest intervention family (StreamingLLM) evicts earliest-not-middle tokens and concludes position-not-content (the opposite sign); the snowball line treats self-history as error-source only; attractor-dynamics work lacks the lesion and the self-vs-clean judge dissociation. Control-confirmed (e074):
  shuffling the prompt band destroys corpus statistics yet creates no
  new junk there (0.024) while the generated band stays junky; within
  it, late-generation entries junk 5.5x more than early (0.169 vs
  0.031) — drift accumulates over the run. Net-dependence (e079,
  B=16 resample): the 10M cell is the robust anchor (0.372 vs 0.000,
  diff CI excludes 0 even under worst-case bounds); 2.7M fires under
  the registered mapping; the ctx-512 cell's weaker contrast does not
  survive a mixed-difficulty B=16 battery (hard sequences flood both
  bands) — claim the split where the effect is large.
- Onset resolved (e053c + e069): the a\* statistic is ABSOLUTE, not
  proportional — doubling the window 256→512 (tokens-per-step matched)
  leaves a\* = 6 CI [4,8] (onset fraction halves); reindexed by training
  steps the old "conflict" dissolves (r(a\*, steps) = +0.82; same-net
  exposure axis monotone 3→21→86). Two invariances of the SPIKE itself
  (e069): window-invariant (ages 1–3 unchanged under eval truncation)
  and statistics-invariant (shuffled-char contexts: ages-1–2 retention
  128% — the circuit reads recent positions, not n-gram statistics).
  Qualifier (revised per e072): the a\* tail sensitivity is largely
  B-fragility — at B=16 the eval-256 a\* is 7 [4,13] vs 6 [4,14] at
  full window (the B=4 contrast of 18 was 2-sequence instrument
  noise); a real but small value-pathway reorganization accompanies
  truncation (V-norms −11% with lesion cost up: per-norm load rises).
  Claim the fixed ~6-token spike horizon; treat the a\* tail statistic
  as B-fragile.
- Framing link: suppression of installed facts and dead cache entries are
  two readings of the same instrument — causal per-position intervention in
  the free-running stream.

---

# 6. Limitations (audit caveats, verbatim where marked)

1. **Single-family char-LMs, ≤2.7M for the core arc** — all expression-gap
   and transplant results are one architecture family at 2.7M/6 layers on
   character Shakespeare; the cache context extends to 10M but with
   steps-confounds. No claim is made about 7B-class models beyond what
   Orgad/YOPO already established behaviorally.
2. **Quote d4, not d5 (verbatim from T033 audit):** "base-net d5 is 23% of
   installed — the shakier leg."
3. **Terminal sites are pseudo-replicated:** "t=120 recurs."
4. **d* = 4 is terminal-carried:** "deep-only stratum would give d*=5,
   outside {2,3,4}." The depth claim is scoped to the mixed-site curve.
5. **Selection circularity RESOLVED by e056b, refined by e056c:** the
   registered kill-test ran the d4 write at 24 random non-onset floor-prior
   positions (base p(Z) median 6.8e-8; trajectory-identity gate bit-exact)
   — the rescue fires anywhere (site-mean 0.324 ≥ 0.30, AUC 1.000;
   shuffled 1.2e-7; base-net twin 0.007), so d*=4 is not a property of
   gap-selected sites. e056c's frozen-rule discriminator then showed the
   off-onset firing is a transient logit blip (floor by +2; zero recurrent
   Z-words in 288 continuations). The original T033 caveat — "ALL 21 sites
   are gap-selected onsets" — remains true of e055's sites; the claims are
   now scoped: durable rescue at onset sites, transient injection at
   non-onset positions.
6. **Downstream Z-words "partly onset-flip + mechanical completion"** —
   R2/R3 expression evidence is supportive, not load-bearing; R1 is the
   primary readout.
7. **n=1 items carried as flagged context, not claims:** the history/scar
   clause's ADDRESS-DIRECTION core is now n=2 across seeds (e044 seed-e001
   net: cos 0.760 vs 0.278 fresh, re-learn 2.08× slower; e044b on the
   seed-43 B43 net, same frozen install set and Dmix protocol, D2-only:
   cos 0.728 vs 0.243 fresh-from-zero, re-learn 2.92× slower — registered
   prediction cos > 0.5 AND > 2× fresh met at both seeds), while its
   route-flip and key-resistance sub-readouts stay n=1; the two-factor
   complete erasure (e046 demotion) and the battery-overstatement
   quantification remain n=1.
8. **Scale-invariance of the address surgery** carries the steps-confound
   caveat (4000/2226/1086 steps anti-correlated with scale).
9. **Terminology scope:** "zero interventional studies" claims are scoped
   to *factual-recall expression* / *own-state transplant* (per T028;
   YOPO is interventional in the abstention domain). "First per-position
   causal KV-cache utility curve" is scoped to free generation at toy
   scale and hedged to our knowledge.

---

# 7. Reviewer-kill risks and pre-emptive answers

## Risk 1 — "Toy scale, single architecture; nothing transfers"

**The kill:** 2.7M char-LM on Shakespeare; reviewers reject external
validity; "where is the 7B experiment?"

**Pre-emptive answer:** (a) The *phenomenon* is not ours to claim at scale —
Orgad et al. and Buckmann et al. establish the probe-vs-generation gap at
7B–8B, and YOPO localizes an analogous suppression onset (layer 19 of 28 —
mid-stack, matching our d*=4 of 6 in relative terms). (b) Our contribution
is the *interventional protocol* — own-state transplant at divergence
tokens with shuffled / base-net / pad-shifted controls and depth-survival
curves — which is scale-portable and, at 7B, cheap to run on the exact
cells Orgad's probes already certify (their exact-answer-token cells are
pre-registered donor sites). (c) The 24-site depth curve with per-site
controls is only feasible at tiny scale; we trade scale for controls and
say so in the title, abstract, and Limitations. (d) The sub-argmax prior
(rank-2 at onset) is directly checkable on any released 7B logit archive —
we provide the check as a one-script contribution.

## Risk 2 — "Selection circularity and pseudo-replication: you chose the
sites that show the gap, then showed they show the gap"

**The kill:** all 21 transplant sites are gap-selected onsets; terminal
sites share t=120; the deep-only stratum moves d* to 5; the base-net d5
control is itself 23% of installed.

**Pre-emptive answer:** (a) e056b RAN and resolved it: the same depth
curve at 24 random non-onset positions from cached trajectories — the
rescue fires anywhere (site-mean 0.324, AUC 1.000, shuffled 1.2e-7),
converting the site-selection objection into a robustness result; e056c
then discriminated what firing-off-onset means (transient logit blip, not
address installation — floor by +2, zero recurrent Z-words in 288
continuations), which is why the durable-rescue claim is scoped to onset
sites in the claim-split. (b) The circularity structure is disclosed in
Limitations
verbatim, with the pseudo-replication and terminal-carried d* caveats —
we quote d4 (the strong leg, 60× its own base-net twin) and scope the
depth claim to the mixed-site curve. (c) The pad-shifted control already
excludes the position-cue leak (donors cap 0.133 vs 0.374), and the
shuffled-donor floor of 0.000 excludes generic-state injection; what the
circularity test adds is the last mile, and it is pre-committed rather
than post-hoc. (d) Precedent framing: Orgad's probe cells are also
selected on the gap — our design at least registers the selection and its
kill-test in advance.

## Risk 3 — "Known phenomenon, missed prior art: exposure bias is 2016
textbook; relay steering already intervened; probes already showed it"

**The kill:** "teacher-forced vs free-run mismatch is exposure bias";
"YOPO/ITI/DoLa are the interventions"; "this is just probe-vs-output gap,
Buckmann said it."

**Pre-emptive answer:** (a) The exposure-bias classics — Ranzato et al.
(ICLR 2016, arXiv:1511.06732; named the train/test mismatch), Bengio et
al.'s Scheduled Sampling (NeurIPS 2015, arXiv:1506.03099), and Professor
Forcing (Lamb et al., NIPS 2016, arXiv:1610.09038) — are behavioral or
training-side: they document the TF/free-run mismatch and train around
it. Professor Forcing is the closest state-level prior, and we
distinguish on two axes: they train an adversarial discriminator to match
TF/free-run *state distributions* in aggregate; we transplant one
*specific state at a named token* and show it carries one *specific
installed fact* (durable rescue at onset sites, transient elsewhere). No
prior work connects TF/free-run *state* differences to *knowledge
expression at a named token*, and none measures where in depth the
knowledge stops being expressible — we own that specific claim, scoped per
T028 to "factual-recall expression / own-state transplant." (b) YOPO,
**Instruments caveat (day-4 addition):** our own head-to-head found PROBE, TRANSPLANT, and GENERATION readouts measure three different things — surgery leaves the deep probe intact while killing all rescue; an RMU-analogue kills both while a plain retain-only fine-tune kills the probe but keeps rescue (e065/e091, n=1 flagged). Any elicitation-failure claim that relies on a single readout inherits this caveat, including Orgad's probe-only evidence.

Yan & Jia, ITI, DoLa are cited in the intro as the adjacent wall — and the
YOPO collision dissolves on our own negative result: YOPO injects a
*learned direction* in the *abstention* domain for *elicitation* — a
general steering vector. Our d4 write is NOT a general steering direction,
and we demonstrate that ourselves: at non-onset positions it fails to
steer even 2 tokens ahead (p(Z) 0.509 at +1 → control floor by +2; zero
recurrent Z-words in 288 continuations; e056c). What we demonstrate is
position-specific suppression and position-specific rescue at the
install-geometry onset — the model's *own teacher-forced states* in the
*factual-recall* domain for *suppression-depth localization*. DoLa assumes
late-layer pollution and never measures where knowledge dies — we provide
the measurement its layer choice lacks. (c) "Probes already showed it" is
exactly our point: the entire prior record is read-only (Orgad's probes
are explicitly diagnostic; their interventional-looking result is
post-hoc *selection among resamples*, not a write); the causal question —
present-but-suppressed vs absent — is undecidable by probes, and our
transplant decides it (state-rescue 0.374 vs shuffled 0.000). (d) The
four-faculty decomposition additionally reframes the gap as one faculty of
an editing law with its own surgical evidence base — a construct none of
the adjacent literatures carry.

---

# Figure/table plan (for the full draft)

- Fig 1: the gap itself — battery acc 92–97% vs 0 free-generation
  occurrences across e048 arms (the honesty-battery panel).
- Fig 2: positional binding — p(Z) vs position (129/130/131), left-pad
  collapse, sub-argmax rank trace.
- Fig 3 (headline): depth-survival curves at 24 sites — TF-transplant vs
  shuffled vs base-net vs pad-shifted; d1-peak/d2-crash/d4-rescue
  annotated; inset d*/6 across sites.
- Fig 4: the four-faculty schema with per-faculty evidence table
  (run → metric → n → caveat).
- Fig 5 (from runs/e056c/persistence_curve.png): the persistence curve —
  mean p(Z) vs continuation offset per arm; donor-write blip at +1
  collapsing onto the shuffled/base floor by +2, flat through +10 (the 3d
  transient-injection negative).
- Table 1: e055 controls summary (rescue ratios, CIs).
- Table 2: limitations ledger (claim → caveat → kill-test → status:
  e056b resolved, e056c refined).
- Appendix: registered predictions verbatim (design memos), audit
  annotations, reproducibility (all runs/, metrics.json + figures in git).

# Submission checklist (pre-flight)

- [x] e056b/e056c landed and folded into 5.4 + Limitations + abstract +
      contributions (circularity OPEN → resolved by e056b; refined by e056c
      into the 3c/3d claim-split; persistence curve = Fig 5).
- [x] Verify search-level citations — done 2026-09-25 via the 29-entry
      verified bibliography (scratch/paper_bibliography.md; every entry
      live-checked against arXiv/OpenReview/ACL Anthology). Bürger "Truth
      is Universal" → Bürger, Hamprecht & Nadler, NeurIPS 2024,
      arXiv:2407.12831 (attribution fixed; RLHF-widening gloss dropped);
      Afzal "Knowing Before Saying" → Findings of ACL 2025,
      arXiv:2505.24362; TruthPrInt → CVPR 2025, arXiv:2503.10602; Sarkar
      → EMNLP 2025 Main, arXiv:2505.16411; Yang → ICLR 2025, OpenReview
      Bjq4W7P2Us. Also applied: FAR AI title → "…can Induce Honesty or
      Evasion" (arXiv:2505.13787); YOPO full title + Qwen2.5-1.5B/3B/7B;
      "Do We Know" as arXiv-only (2505.21701); Contrastive Decoding ID
      2210.15097; MQuAKE = EMNLP 2023 original; Risk-3 exposure-bias
      classics (Ranzato / Scheduled Sampling / Professor Forcing)
      replacing the unconfirmable Bridge-Garden / EGz8InJz6F items.
      References section added at end.
- [x] Decide d4-vs-d5 primary framing stays (audit says quote d4).
- [x] Scope-soften every "first"/"zero" claim per T028 wording.
- [x] Scar clause resolved 2026-09-25: e044b replicated the
      address-direction core on B43 (cos 0.728 vs 0.243 fresh-from-zero,
      2.99×; re-learn 2.92× slower) → the scar clause is now n=2;
      route-flip/key-resistance sub-readouts stay n=1-flagged in 5.2 and
      Limitations.
- [x] d6 readout-dominated caveat: e056b confirms d6 = donor readout
      (0.715 = donor p_z mean) — keep d6 out of causal claims.

---

# References

Verified citation set (29 entries), live-checked against
arXiv/OpenReview/ACL Anthology on 2026-09-25; source of truth:
scratch/paper_bibliography.md. Bracketed notes flag metadata still to
pull at camera-ready.

1. Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov. "LLMs Know
   More Than They Show: On the Intrinsic Representation of LLM
   Hallucinations." ICLR 2025. arXiv:2410.02707.
2. Buckmann, Nguyen, Hill. "Revealing economic facts: LLMs know more than
   they say." arXiv:2505.08662 (2025). [preprint]
3. Luo, Chu, He, Wang, Qin, Wu, Chen. "You Only Pass Once: Answering and
   Abstaining Together in a Single Forward Pass of a Frozen Language
   Model." arXiv:2608.14465 (2026). [preprint; Luo and Chu equal
   contribution]
4. Tianyi Lorena Yan and Jia. "Promote, Suppress, Iterate: How Language
   Models Answer One-to-Many Factual Queries." EMNLP 2025 (Main).
   arXiv:2502.20475.
5. Gu, Pang, Du, Liu, Guo, Pai, Bai, Jiao. "When Attention Sink Emerges
   in Language Models: An Empirical View." ICLR 2025. arXiv:2410.10781.
6. "KVSink: Understanding and Enhancing the Preservation of Attention
   Sinks in KV Cache Quantization for LLMs." COLM 2025. arXiv:2508.04257.
   [author list to pull at camera-ready]
7. Guo, Syed, Sheshadri, Ewart, Dziugaite. "Mechanistic Unlearning:
   Robust Knowledge Unlearning and Editing via Mechanistic Localization."
   ICML 2025. arXiv:2410.12949.
8. Modarressi et al. "NoLiMa: Long-Context Evaluation Beyond Literal
   Matching." ICML 2025. arXiv:2502.05167.
9. Li, Patel, et al. "Inference-Time Intervention: Eliciting Truthful
   Answers from a Language Model." NeurIPS 2023. arXiv:2306.03341.
10. Chuang, Xie, Luo, Kim, Glass, He. "DoLa: Decoding by Contrasting
    Layers Improves Factuality in Large Language Models." ICLR 2024.
    arXiv:2309.03883.
11. Li, Holtzman, Fried, Liang, Eisner, Hashimoto, Zettlemoyer, Liang.
    "Contrastive Decoding: Open-ended Text Generation as Optimization."
    ACL 2023. arXiv:2210.15097.
12. Heimersheim and Nanda. "Towards Best Practices of Activation Patching
    in Language Models: Metrics and Methods." arXiv:2309.16042.
13. Bürger, Hamprecht, Nadler. "Truth is Universal: Robust Detection of
    Lies in LLMs." NeurIPS 2024. arXiv:2407.12831.
14. Afzal, Matthes, Chechik, Ziser. "Knowing Before Saying: LLM
    Representations Encode Information About Chain-of-Thought Success
    Before Completion." Findings of ACL 2025. arXiv:2505.24362.
15. Cundy and Gleave (FAR AI). "Preference Learning with Lie Detectors
    can Induce Honesty or Evasion." arXiv:2505.13787 (2025); NeurIPS
    2025.
16. Duan et al. "TruthPrInt: Mitigating Large Vision-Language Models
    Object Hallucination via Latent Truthful-Guided Pre-Intervention."
    CVPR 2025. arXiv:2503.10602.
17. Yang et al. "Understanding and Mitigating Hallucination in Large
    Vision-Language Models via Modular Attribution and Intervention."
    ICLR 2025. OpenReview Bjq4W7P2Us. [arXiv mirror to pull at
    camera-ready]
18. Sarkar, Che, Gavin, Beerel, Kundu. "Mitigating Hallucinations in
    Vision-Language Models through Image-Guided Head Suppression."
    EMNLP 2025 (Main). arXiv:2505.16411.
19. Zhao, Köksal, Modarressi, Hedderich, Schütze. "Do We Know What LLMs
    Don't Know? A Study of Consistency in Knowledge Probing."
    arXiv:2505.21701 (2025). [preprint; no venue listed]
20. Miao et al. "Correctness-Optimized Residual Activation Lens (CORAL):
    Transferrable and Calibration-Aware Inference-Time Steering."
    arXiv:2602.06022 (2026).
21. "The Mirage of Performance Gains: Why Contrastive Decoding Fails to
    Mitigate Object Hallucinations in MLLMs?" arXiv:2504.10020 (2025).
    [authors to pull at camera-ready; scope: MLLM object hallucination]
22. Li, Pan, Gopal, Yue, ... Hendrycks (56 authors). "The WMDP Benchmark:
    Measuring and Reducing Malicious Use With Unlearning."
    arXiv:2403.03218 (2024). [origin of RMU; venue not asserted — check
    proceedings at camera-ready]
23. Fan, Wang, Huang, Pal, Liu, et al. "LLM Unlearning Under the
    Microscope: A Full-Stack View on Methods and Metrics."
    arXiv:2510.07626 (2025).
24. Shilov, Cloud, Gema, Goldman-Wetzler, Panickssery, Sleight, et al.
    (Anthropic). "Beyond Data Filtering: Knowledge Localization for
    Capability Removal in LLMs." arXiv:2512.05648 (2025).
25. Cohen, Biran, Yoran, Globerson, Geva. "Evaluating the Ripple Effects
    of Knowledge Editing in Language Models." TACL 12:283–298 (2024).
    arXiv:2307.12976.
26. Zhong, Wu, Manning, Potts, Liang. "MQuAKE: Assessing Knowledge
    Editing in Language Models via Multi-Hop Questions." EMNLP 2023.
    arXiv:2305.14795. [MQuAKE-Remastered, ICLR 2025, is a separate
    follow-up paper]
27. Ranzato, Chopra, Auli, Zaremba. "Sequence Level Training with
    Recurrent Neural Networks." ICLR 2016. arXiv:1511.06732.
28. Bengio, Vinyals, Jaitly, Shazeer. "Scheduled Sampling for Sequence
    Prediction with Recurrent Neural Networks." NeurIPS 2015.
    arXiv:1506.03099.
29. Lamb, Goyal, Zhang, Zhang, Courville, Bengio. "Professor Forcing: A
    New Algorithm for Training Recurrent Networks." NIPS 2016.
    arXiv:1610.09038.
30. Feng, Steinhardt. "How Do Language Models Bind Entities in
    Context?" ICLR 2023. arXiv:2310.17191.
31. Smolensky, McCoy, Lin, Farnadi, Murty, Prabhumoye, et al. "Positional
    Description Matters: Solving Mathematical Word Problems via
    Positional Descriptions in Vector Space." arXiv:2410.17498 (2024).
32. Dai, Gutierrez, Yang, Peng, Li. "Ordering IDs: Ordering Vector as
    Circuits!" arXiv:2409.05448 (2024).
33. Todd, Li, Arnold, Rajeswaran, Zettlemoyer, Schmidt. "Function
    Vectors in Large Language Models." EMNLP 2024 Findings.
    arXiv:2310.15213.
34. Meng, Bau, Andonian, Belinkov. "Locating and Editing Factual
    Associations in GPT." NeurIPS 2022. arXiv:2202.05262 (ROME).
35. Wu, Geiger, Millière. "The Residual Stream as a Memory: Analyzing
    and Manipulating In-Context Symbolic Programs." arXiv:2505.20896
    (2025).
36. He, Dai, et al. "TAPE: Learning Position-Ready Token Embeddings."
    ICML 2025. arXiv:2501.00712. [content-vs-position addressing
    framing; no row-level causality]
37. Zhang, Press. "How Language Model Hallucinations Can Snowball."
    ICML 2024. arXiv:2305.13534. [behavioral prior for Claim 2]
38. Arora, Del Corro, et al. "Learning to Crowdsource Fallacies /
    imitation-learning error accumulation." AAAI 2022.
    arXiv:2110.05978. [behavioral prior for Claim 2]
- [x] Citation VERIFIED 2026-09-26: the rumored title maps to
      Giannou et al., "Looped Transformers are Universal Computers"
      (arXiv:2308.02852) — programs weights + uses positional
      embeddings as instruction pointers. Cited as ref 39 in the
      Claim-1 perimeter (distinguished: they PROGRAM a looped net;
      we LOCALIZE an installed memory's address row).
39. Giannou, Rajput, et al. "Looped Transformers are Universal
    Computers." arXiv:2308.02852 (2023). [positional embeddings as
    instruction pointers in programmed looped nets — build-side;
    distinguished from our row-level causal localization]
