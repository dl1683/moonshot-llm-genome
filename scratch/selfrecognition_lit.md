# Lit scan 2026-09-25 — self-recognition (T060/T061/e111), anchor-as-fixed-point (W004), direction+LN-floor (e102/W001)

Scope: 2018–2026, web scan only, RESEARCHER angle. Read together with
`trajectory_anchor_lit.md` (T048) and `massaction_key_lit.md` (T051/T053) — those cover
the trajectory-anchor and mass-law claims; this scan covers the NEW self-recognition /
fixed-point / LN-arithmetic angles.

---

## CLAIM A — binary self-recognition on internal V-geometry (T060/T061/e111)

**Claim under scan:** a free-running char-LM verifies its own history every token via a
BINARY self/other test on internal V-geometry — two-cluster step (cos 0.40 self vs 0.14
any-other net); behaviorally near-identical donors (output JS 0.041) are rejected;
self-signature is a ~7-dim principal subspace; other nets' ACTUAL outputs are EXCLUDED
from it at/below isotropic null (locked out, not merely misaligned).

### Closest works (ranked)

1. **Asvin G. & Lindsey, "From Simulation to Enaction: Post-trained language models
   recognize and react to their own generations" (Anthropic, arXiv:2605.25459, May
   2026).** THE closest prior and the one a reviewer will cite. Post-trained (not
   pretrained) LLMs implicitly recognize on-policy contexts: output entropy 3–4x lower
   on own generations, traced to an internal "representation of input surprise" that
   causally modulates entropy; explicit verbal self/other distinction routes through a
   SEPARATE mechanism. Differs on all four lab specifics: signal is an entropy/surprise
   aggregate (not per-token binary geometry), post-hoc (not an in-run gate), no
   near-identical-donor controls (an entropy channel cannot separate behaviorally
   cloned donors), no subspace-exclusion/below-null result.
2. **Zhou et al., "From Implicit to Explicit: Enhancing Self-Recognition in LLMs"
   (arXiv:2508.14408, Aug 2025; + the related "Territorial Awareness"/cognitive-surgery
   line it cites).** Closest on the SUBSPACE idea: models encode authorship internally
   ("implicit self-recognition" = internal-representation/output gap); their CoSur
   pipeline — representation extraction, SUBSPACE CONSTRUCTION, authorship
   discrimination, cognitive editing — reaches 97–99% self/other accuracy. But the
   subspace + classifier are ENGINEERED post-hoc instruments (external probe), eval-time
   single-text paradigm, own-vs-human/other-model text; nothing about an intrinsic
   every-token test the network itself runs during free generation, nothing below-null.
3. **Panickssery, Bowman & Feng (arXiv:2404.13076, NeurIPS 2024)** — LLM evaluators
   recognize and favor their own generations; watermark manipulation gives causal
   evidence the self-preference rests on genuine self-recognition. Behavior-level,
   eval-time, judge paradigm. (Already in-lab from T048 scan.) Adjacent: Ackman et al.
   2410.02064 (inspection/control of self-generated-text recognition); "Extreme
   Self-Preference" 2509.26464 (minimal identity cues suffice).
4. **Self-knowledge probing line: Orgad et al. (arXiv:2410.02707, ICLR 2025); Kadavath
   et al. (arXiv:2207.05221); Azaria & Mitchell (2304.13734); "Looking Inward"
   (2410.13787).** Internal representations encode properties of the model's OWN
   answers (correctness, hallucination, future behavior) — the probing template the lab
   result inverts: they probe self-KNOWLEDGE (truth of own output), Claim A is
   self-IDENTITY (authorship of own history) read off run geometry.
5. **Anti-prior for the exclusion clause: Platonic Representation Hypothesis (Huh et
   al., arXiv:2405.07987)** — models' representations CONVERGE at scale. Claim A's
   below-isotropic-null exclusion (other nets' actual outputs locked OUT of the
   self-subspace) is a convergence FALSIFIER at the output-token-geometry level;
   critiques of PRH (Umwelt Representation Hypothesis; objective-driven alignment
   results) establish divergence exists in general but never as an exclusion-at-null.

### Verdict: PARTIALLY KNOWN — behaviorally anticipated, mechanistically claimable

- Known: LLMs behaviorally recognize their own generations (2605.25459, 2404.13076,
  2508.14408, 2410.02064); authorship info exists in internal representations and an
  engineered subspace can decode it (2508.14408).
- Claimable: (i) an INTRINSIC, per-token, binary self/other discriminator in the
  V-geometry of a free-running PRETRAINED (not post-trained) tiny char-LM; (ii) the
  two-cluster step with near-identical-donor rejection (decouples identity from
  behavior — no prior paradigm even runs this control); (iii) below-isotropic-null
  EXCLUSION of other nets' actual outputs (an active lock-out signature; anti-Platonic
  at the token-geometry level).
- **Sharpest reviewer-collapse:** "Asvin & Lindsey already showed models implicitly
  recognize on-policy context (their headline), so this is a toy-model rediscovery; and
  the two-cluster cosine structure is an anisotropy/rogue-dimension artifact (Timkey &
  van Schijndel 2021) — cos 0.40 vs 0.14 is dominated by a few high-variance dims, and
  the '7-dim self-signature' is just top-PCA of token-frequency differences."
- **Rebuttal:** (1) the entropy/surprise channel of 2605.25459 cannot separate donors
  with JS 0.041 — near-surprise-equal inputs give near-equal entropy; the cos-geometry
  channel separates them, so the lab's signal is provably not reducible to their
  mechanism; (2) their effect requires POST-TRAINING; the lab's appears in a pretrained
  ~2.7M char-LM with no instruction tuning — different object; (3) rogue-dimension
  collapse predicts other-nets cos pulled TOWARD the dominant dims (shared anisotropy),
  the opposite of the observed below-null exclusion — run the standardization control
  (Timkey's own fix) and show the step survives; (4) causal gap: currently
  logits/geometry-only evidence — intervene on the 7-dim subspace (project donor
  outputs onto it / delete it) and show the binary verdict and donor acceptance flip,
  or the claim stays correlational. Honesty reflex: before claiming "the model verifies
  itself," show the binary variable gates BEHAVIOR (accept/reject continuation), not
  just that it exists.

---

## CLAIM B — the anchor as self-consistency / fixed point (W004)

**Claim under scan:** "self" is defined by construction — X is self iff X looks like
what my weights generate; the anchor checks whether the run's history is a fixed point
of its own dynamics; collapse = self-inconsistency detection.

### Closest works (ranked)

1. **Asvin & Lindsey 2605.25459** (again — also Claim B's nearest). They own the
   "enaction" framing (pretraining = simulating the text distribution; the model
   reacting to ITS OWN generations is the enactive turn) and the input-surprise
   mechanism = a running self-vs-other likelihood check. What they do NOT do: state the
   constitutional definition (self ≡ fixed point / typical set of own dynamics) or
   connect it to run stability/collapse.
2. **DEQ / looped-transformer line: Bai et al. Deep Equilibrium Models (arXiv:1909.01377,
   NeurIPS 2019); Universal Transformers (1807.03819); "Fixed-Point Reasoners: Stable
   and Adaptive Deep Looped Transformers" (2026); Consistency DEQs (2602.03024).**
   Fixed points as the DEFINITION of the model's computation — but over DEPTH (one
   operator iterated to convergence), not over the autoregressive trajectory; used as an
   architecture, never as an online identity predicate on run history.
3. **Wang et al., "Unveiling Attractor Cycles in LLMs" (arXiv:2502.15208, ACL 2025)** —
   the only explicit dynamical-attractor treatment of LLM generation found (also nearest
   for T048): iterated paraphrasing settles into stable limit cycles. But the attractor
   is of an EXTERNAL iteration loop, entered unperturbed; no history-is-fixed-point
   check, no lesion/collapse semantics.
4. **Self-consistency-as-verification line: Wang et al. self-consistency decoding
   (2203.11171); SelfCheckGPT (2303.08896); consistency models (2303.01469).** All
   verify a candidate against the model's own re-samples — an ORCHESTRATED, external
   self-consistency check. Nobody found claiming the forward pass ITSELF performs one
   continuously on its own context as the substrate of run stability.
5. **Hopfield/energy view: Ramsauer et al., "Hopfield Networks is All You Need"
   (2008.02217); Sussillo & Barak fixed-point finding in RNNs (1309.7942).** Attention
   update ≈ Hopfield retrieval to fixed-point memories — fixed points as the objects
   stored/ retrieved, the ancestor vocabulary; retrieval ≠ validation of trajectory
   authorship. Also Orgad et al. 2410.02707: internal representations check
   self-consistency of OWN answers (factuality) — the check exists in the literature,
   but on truth, not identity/history.

### Verdict: PARTIALLY KNOWN — framing is borrowed territory, the dynamical use is claimable

- Known: models react differentially to own generations (2605.25459); self-consistency
  as an external verification procedure (2203.11171, 2303.08896); fixed points as the
  computation (DEQ) or as memories (Hopfield); attractor language for degeneration
  (2502.15208).
- Claimable: recasting generation as ONGOING self-consistency verification of run
  history — collapse = the predicate's failure mode — with the lab's interventional
  evidence (vzero/noise/borrowed arms, donor controls) as the dynamics, and the
  fixed-point formulation's quantitative prediction: perturbations that keep history a
  near-fixed-point of the model's conditional are tolerated; ones that move it off
  collapse. As a THEORETICAL framing it must be sold as a model (W-paper), not a
  discovery, because the philosophical content ("self = what my weights generate") is
  close to tautology once said aloud.
- **Sharpest reviewer-collapse:** "This is exposure bias / off-policy conditioning
  (Zhang & Press snowballing 2305.13534; Braverman 1906.05664): the model is trained on
  human text and fed its own (or others') text at inference; 'collapse = self-
  inconsistency detection' is just distribution shift under teacher forcing, re-labeled
  with enactive vocabulary from the 2026 Anthropic paper."
- **Rebuttal:** (1) sign inversion relative to the exposure-bias canon — their self-
  history is poison that snowballs, ours is the anchor whose REMOVAL collapses
  (T048/T051 evidence); a pure off-policy-shift story predicts borrowed REAL content
  (on-distribution, on-policy-ish for a similar net) should rescue the run — it recovers
  only ~3/4, and near-identical donors (JS 0.041) are still geometrically rejected
  (Claim A) — shift magnitude does not explain the cliff; (2) the fixed-point model
  generates falsifiable dose-response predictions (which-subset irrelevance below
  threshold, sub-additivity) that the exposure-bias framing does not; (3) scope hedge:
  single tiny char-LM; register the GPT-2-small replication before calling it general.

---

## CLAIM C — direction carries the anchor + LN floor as stream-SHARE (e102/W001)

**Claim under scan:** the anchor needs DIRECTION (unit-norm originals still anchor;
norm-correct random directions collapse); the magnitude floor is a post-LayerNorm
stream-SHARE (signal-to-noise) effect, which re-derives the count-threshold mass law
(T051) from normalization arithmetic.

### Closest works (ranked)

1. **Normalization-arithmetic-as-explanation canon: Barbero et al., "Why do LLMs attend
   to the first token?" (arXiv:2504.02732, COLM 2025); Gu et al., "When Attention Sink
   Emerges" (arXiv:2410.10781, ICLR 2025 Spotlight); Sun et al., Active-Dormant heads
   (arXiv:2410.13835); provable-sink results (2603.11487).** THE methodological
   ancestors: "softmax must allocate its unit mass somewhere + LN geometry ⇒ observed
   allocation phenomena." The lab's move (LN share arithmetic ⇒ threshold mass law) is
   the same proof style applied to a different target (anchor viability, not sink
   parking). This is prior art for the STYLE of argument, and the citation reviewers
   will demand.
2. **Norm-dominance line: Timkey & van Schijndel, rogue dimensions (arXiv:2109.04404,
   EMNLP 2021); Sun et al., Massive Activations (2402.17762); Heimersheim & Turner,
   residual stream norms grow exponentially (LessWrong 2023, confirmed in Anthropic
   circuit-tracing 2025).** Magnitude dominates raw cosine/norm measures; LN re-scales
   so downstream blocks effectively read direction/share. Supports the claim's
   mechanism but never states a count-threshold law.
3. **LN-removal/LN-theory: Heimersheim, "You can remove GPT2's LayerNorm by fine-
   tuning" (2024) — LN ≈ magnitude rescaling + small directional rotation.** Direct
   evidence that what survives LN is (mostly) direction — the cleanest existing
   statement of "LN floors are about scale-removal, direction persists."
4. **The empirical threshold curves (no arithmetic): StreamingLLM (2309.17453), H2O
   (2306.14048), LLMLingua (2310.05736) — flat-then-cliff cache/prompt budgets (from
   the T051 scan).** The count-threshold phenomenon exists; no prior work derives the
   threshold from normalization.
5. **Residual stream as a finite shared channel: Elhage et al., toy models of
   superposition (transformer-circuits 2022).** Features compete for channel capacity —
   the qualitative ancestor of "stream-SHARE" (anchor must exceed noise's share of the
   post-LN unit norm).

### Verdict: PARTIALLY KNOWN — pieces known, the derivation is claimable

- Known: softmax sum-to-one + LN geometry explains allocation phenomena (sinks);
  magnitude dominance and LN's scale-stripping are established; flat-then-cliff budget
  curves are established.
- Claimable: (i) the quantitative RE-DERIVATION of T051's count threshold (~64–128
  entries) from post-LN share arithmetic — no paper found deriving ANY context-mass
  threshold from normalization; (ii) the direction-vs-magnitude DISSOCIATION as a
  causal test (unit-norm originals anchor; norm-correct random directions collapse) —
  not found anywhere; it kills the trivial alternative "any sufficiently large
  perturbation-tolerant mass works."
- **Sharpest reviewer-collapse:** "LayerNorm discards magnitude, so 'direction carries
  it' is the textbook isomorphism — dimensional analysis any transformer-theory
  reviewer can write on a napkin; Barbero/Gu already own normalization-arithmetic
  explanations of attention allocation; the 'law' is a napkin derivation dressed as a
  result."
- **Rebuttal:** (1) if it were trivial, the threshold NUMBER would be predicted by the
  napkin — the deliverable is that share arithmetic predicts the observed k* (64–128 of
  ~350) and its parameter dependence (threshold should shift with d_model, stream-norm
  growth, and position/depth — three falsifiable registrations, none testable from the
  sink literature); (2) the sink canon explains WHERE attention parks; ours explains
  HOW MUCH on-manifold mass an anchor needs to survive the LN floor — different
  quantity, different falsifier (their story has no count threshold); (3) the
  norm-correct-random-direction collapse arm refutes the pure-magnitude reading of the
  napkin version. Hedge: state explicitly which parts are arithmetic (derivable) vs
  empirical (the threshold value); do not sell the derivation as more surprising than
  it is — its value is UNIFYING T051's empirical law, not novelty of LN math.

---

## VERDICTS (summary)

- **Claim A (binary self-recognition, V-geometry): PARTIALLY KNOWN → claimable as
  mechanism.** Behavioral self-recognition is published (2605.25459 is the must-cite;
  also 2404.13076, 2508.14408, 2410.02064). The intrinsic per-token binary geometry
  test, near-identical-donor rejection, and below-null exclusion are unclaimed. MUST
  DO: standardization/rogue-dim control; causal intervention on the 7-dim subspace;
  behavior-gating evidence.
- **Claim B (anchor = self-consistency fixed point): PARTIALLY KNOWN → claimable as a
  model (W-paper), not a discovery.** Ingredients exist (enaction framing 2605.25459;
  DEQ/Hopfield fixed points; external self-consistency checks; attractor cycles
  2502.15208). The ongoing in-run predicate + collapse-as-failure-mode with the lab's
  interventional sign inversion is unclaimed. Position against exposure-bias canon
  explicitly; register GPT-2-small replication.
- **Claim C (direction + LN stream-share floor): PARTIALLY KNOWN → claimable as a
  derivation.** Normalization-arithmetic argument style is owned by the sink canon
  (2504.02732, 2410.10781); LN-strips-magnitude is textbook; NO prior derives a
  count-threshold mass law from it, and the direction/magnitude dissociation
  experiment is unclaimed. Sell as unification of T051, cite Barbero/Gu as
  methodological ancestors, register the parameter-dependence predictions.

Cross-claim synergy for the write-up: C supplies the mechanism (post-LN share readout
in direction space), A supplies the identity predicate implemented in that space
(self-subspace membership), B supplies the dynamical semantics (history-as-fixed-point
whose violation = collapse). No single prior work contains any two of the three.

## SOURCES

Claim A:
- From Simulation to Enaction (Asvin & Lindsey, Anthropic, May 2026): https://arxiv.org/abs/2605.25459
- From Implicit to Explicit: Enhancing Self-Recognition in LLMs (CoSur): https://arxiv.org/abs/2508.14408
- LLM Evaluators Recognize and Favor Their Own Generations (Panickssery et al.): https://arxiv.org/abs/2404.13076
- Self-Generated-Text Recognition / inspection & control (Ackman et al.): https://arxiv.org/abs/2410.02064
- Extreme Self-Preference in LLMs: https://arxiv.org/html/2509.26464v1
- LLMs Know More Than They Show (Orgad et al., ICLR 2025): https://arxiv.org/abs/2410.02707
- Language Models (Mostly) Know What They Know (Kadavath et al.): https://arxiv.org/abs/2207.05221
- Looking Inward: Introspection (Chen et al.): https://arxiv.org/abs/2410.13787
- The Internal State of an LLM Knows When It's Lying (Azaria & Mitchell): https://arxiv.org/abs/2304.13734
- Platonic Representation Hypothesis (Huh et al.): https://arxiv.org/abs/2405.07987
- All Bark and No Bite: Rogue Dimensions (Timkey & van Schijndel): https://arxiv.org/abs/2109.04404 / https://aclanthology.org/2021.emnlp-main.372

Claim B:
- From Simulation to Enaction: https://arxiv.org/abs/2605.25459
- Deep Equilibrium Models (Bai et al.): https://arxiv.org/abs/1909.01377
- Universal Transformers (Dehghani et al.): https://arxiv.org/abs/1807.03819
- Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers (2026; arXiv ID unverified — located via OpenReview/arXiv search June 2026)
- Unveiling Attractor Cycles in LLMs (Wang et al., ACL 2025): https://arxiv.org/abs/2502.15208
- Self-Consistency Improves CoT (Wang et al.): https://arxiv.org/abs/2203.11171
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Consistency Models (Song et al.): https://arxiv.org/abs/2303.01469
- Hopfield Networks is All You Need (Ramsauer et al.): https://arxiv.org/abs/2008.02217
- Opening the Black Box: fixed points in RNNs (Sussillo & Barak): https://arxiv.org/abs/1309.7942
- How LM Hallucinations Can Snowball (Zhang & Press): https://arxiv.org/abs/2305.13534
- Calibration, Entropy Rates, and Memory (Braverman et al.): https://arxiv.org/abs/1906.05664

Claim C:
- Why do LLMs attend to the first token? (Barbero et al., COLM 2025): https://arxiv.org/abs/2504.02732
- When Attention Sink Emerges (Gu et al., ICLR 2025): https://arxiv.org/abs/2410.10781
- Active-Dormant Attention Heads / extreme-token phenomena (Sun et al.): https://arxiv.org/abs/2410.13835
- Attention Sinks Provably Necessary in Softmax Transformers: https://arxiv.org/html/2603.11487v4
- Massive Activations in LLMs (Sun et al.): https://arxiv.org/abs/2402.17762
- Residual stream norms grow exponentially (Heimersheim & Turner): https://www.lesswrong.com/posts/gEZPhXPhb7o4PtFqk/residual-stream-norms-grow-exponentially-over-the-forward-pass
- You can remove GPT2's LayerNorm by fine-tuning (Heimersheim): https://www.lesswrong.com/posts/nYkpkeofnik9BMpKa/you-can-remove-gpt2-s-layernorm-by-fine-tuning
- Toy Models of Superposition (Elhage et al.): https://transformer-circuits.pub/2022/toy_model/index.html
- StreamingLLM: https://arxiv.org/abs/2309.17453 ; H2O: https://arxiv.org/abs/2306.14048 ; LLMLingua: https://arxiv.org/abs/2310.05736
- A Geometric Approach on Attention Sink: https://arxiv.org/abs/2508.02546

Search-run notes: "Territorial Awareness" (Wang et al.) is cited inside 2508.14408 but
its own arXiv ID could not be confirmed via the arXiv API (title search returns 0) —
cite via the CoSur paper. Timkey standardization control and the 2605.25459
entropy-confound dissociation are the two pre-registered reviewer defenses for A.
