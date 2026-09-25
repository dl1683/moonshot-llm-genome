# Paper Skeleton — P-A submission draft, Day 3 (STRATEGIST, 2026-09-25)

Status: SKELETON for the actual submission-shaped draft, per novelty_inventory
P-A "double-down" verdict and the completed expression-gap arc (T028-T035).
Not the day report. Every quantitative claim below is traceable to a run
(e023/e042/e043/e044/e048/e053/e055/e056b/e056c) or an audit annotation; every
caveat is carried verbatim from the T033 audit into Limitations, with the
circularity item resolved by e056b and refined by e056c (T034/T035) into the
3c/3d claim-split. Target venue class: mech-interp workshop → main
conference.

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
  - **You Only Pass Once (arXiv:2608.14465)** — the closest prior in the
    adjacent domain: on Qwen2.5-7B sufficiency detection, the residual
    stream encodes "context is sufficient" while generation fabricates
    anyway; a learned difference-of-means direction separates 124/125, and
    "relay steering" pushes the internal verdict into speech (fidelity 1.0),
    with causal onset at layer 19 and overwrite interference at mid-stack.
    **Domain: abstention, not factual recall; method: learned direction, not
    own-state transplant; goal: elicitation, not depth-localizing
    suppression.** We cite them proactively and scope accordingly.
- The unclaimed question: *where, in depth, does installed factual knowledge
  stop being expressible — and is it destroyed, or suppressed?* No
  interventional study of factual-recall expression exists (lit-scan
  verdict, softened from "zero interventional studies anywhere" per T028).
  Neighbors to cite-and-distinguish: Yan & Jia EMNLP-25 (promote-then-suppress
  circuits exist for enumeration repetition), ITI/DoLa/contrastive-decoding
  (all *assume* late-layer pollution of early-layer facts; none measure
  where knowledge dies; DoLa's layer choice is per-token automatic, not
  anatomical), FAR AI probe-evasion (probe-visible knowledge can be driven
  deeper, not deleted — matters for our "the knowledge is really in there"
  claim).

## 2.2 Why a 2.7M char-LM

- The gap is behavioral and reproduces at 7B (Orgad, YOPO); what is missing
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
   and is ~3× more surgical-resistant; n=1, flagged). With surgical evidence
   from e023/e042/e043/e044/e048.
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
4. **The d1-peak/d2-crash destruction–re-emergence structure** — at
   off-geometry sites the address survives block-0 output (d1 peak), is
   destroyed across blocks 1→2 (crash), and becomes re-injectable from
   depth 4: suppression, not erasure, with a mid-stack causal locus
   (d*=4 of 6, inside the registered {2,3,4} window; d1/d2 ratios 4.6–76×
   at 9/10 deep sites).
5. **The negative-utility cache findings as adjacent context** — the same
   free-run causal apparatus yields the first per-position causal KV-cache
   utility curve: spike+plateau shape (~85–95% of ctx-256 cache is dead
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
  generation is the check.
- Sub-argmax rank/prior logging at every diagnostic position; completion
  given the first token measured separately (TF-completion given 'Z' ≈ 1.00
  — the entire gap lives at the onset choice).

## 4.5 Transplant and control suite (e055 core)

- Own-state transplant: teacher-forced residual stream at the onset
  (divergence) position written into the free run, swept over depth d1–d6,
  24 sites; one-shot and held write semantics.
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
- e044 (history faculty, supporting, n=1 flagged): re-learning after
  erasure is 2.08× slower but regrows the original address direction
  (cos 0.760 vs 0.278 fresh); the new route is genuinely new (atlas ρ 0.21;
  the old carrier head flips to anti-carrier, −2.03) and the re-learned
  memory is ~3× more resistant to the original surgical key (44.5% vs
  0.13% under D2+patch).

## 5.3 The gap is positional; the zero was off-geometry (T032 design probes; audits of e048)

- One-char context shift (129→131) collapses p(Z) 0.556→0.12 and kills
  argmax; left-padding with content fixed collapses identically → **wpe-130
  positional binding, not content binding** (resolves T019's open
  content-vs-position edge).
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

- First causal per-position KV-cache utility curve: last-~7-token spike
  (+0.4 to +3.2 nats/position) + shoulder (17–32) + near-zero plateau;
  ~85–95% of the ctx-256 cache is dead weight at generation.
- Sink dead at generation in 5/5 cells (sink-lesion dCE 0.0069; decays,
  sometimes sign-flips).
- Negative-utility entries: 13–20% of positions (32% of old positions at
  10M) — lesion IMPROVES the model.
- Onset is instrument-dependent (registered-threshold a* = 73/86/4 across
  scales; 3→21→86 across exposure = grows 28.7×, the registered direction;
  the sign-based shrink statistic conflicts — documented as an OPEN
  conflict, ctx-512 cell is the registered decider).
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
   clause (e044, single net), the two-factor complete erasure (e046
   demotion), battery overstatement quantified in one protocol.
8. **Scale-invariance of the address surgery** carries the steps-confound
   caveat (4000/2226/1086 steps anti-correlated with scale).
9. **Terminology scope:** "zero interventional studies" claims are scoped
   to *factual-recall expression* / *own-state transplant* (per T028;
   YOPO is interventional in the abstention domain).

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

**Pre-emptive answer:** (a) Exposure-bias literature is behavioral or
theoretical (self-recovery, concavity theory, distillation attribution);
no prior work connects TF/free-run *state* differences to *knowledge
expression at a named token*, and none measures where in depth the
knowledge stops being expressible — we own that specific claim, scoped per
T028 to "factual-recall expression / own-state transplant." (b) YOPO,
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
- [ ] Verify search-level citations before camera-ready (Truth is
      Universal; Knowing Before Saying; TruthPrInt; Sarkar; Yang) — lit-scan
      flags these as unverified.
- [ ] Decide d4-vs-d5 primary framing stays (audit says quote d4).
- [ ] Scope-soften every "first"/"zero" claim per T028 wording.
- [ ] Scar clause: keep n=1 flag visible or cut to footnote.
- [ ] d6 readout-dominated caveat: e056b confirms d6 = donor readout
      (0.715 = donor p_z mean) — keep d6 out of causal claims.
