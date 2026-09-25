# Frontier research scan — 2026-09-25 (RESEARCH EXPLORER, CPU-only)

Scope: hot mechanistic mysteries 2025-2026 where tiny-model dissection
(0.84M-2.7M char models) could contribute; state of mechanistic unlearning /
editing limits; init-anchoring & LMC; 3 blocker candidates. All web-sourced.

---

## Per-topic findings

### 1a. KV-cache junk states / attention sinks
- **Gu et al., "When Attention Sink Emerges in Language Models: An Empirical
  View" (ICLR 2025, arXiv:2410.10781)** — the anchor paper. Sinks are
  universal (appear even in a **14M-param model** — directly our size class),
  emerge during pretraining in a temporal order (first token first, then 2nd,
  3rd...), are *encouraged by weight decay*, migrate to prefix tokens under
  prefix-LM training, and deleting the responsible neurons kills the sink but
  also hurts performance. So sinks are load-bearing, not vestigial.
- **KVSink (COLM 2025, arXiv:2508.04257)** — sinks co-occur with cross-layer
  extreme activation outliers; sinks exist *beyond* initial positions, which
  is why "preserve-first-N" quantization is insufficient. Mechanistic story
  is still called "insufficiently understood" by the authors themselves.
- **Softmax sum-to-one theory** (Gu et al. + follow-ups incl. a 2025/2026 "A
  Unifying View of Attention Sinks" framing sinks as *two algorithms*:
  attention dilution + attention concentration): attention rows that want to
  be "off" must dump mass somewhere; the first token is the only universally
  visible sacrificial victim in causal attention.
- 2026 continuation: "How Attention Sinks Emerge in Large Language Models"
  (Peng et al., arXiv 2026) — emergence mechanism; plus sink-free attention
  architectures (e.g., Threshold Differential Attention, 2026) that relax
  sum-to-one.
- **What is NOT understood**: whether non-sink cached entries are junk or
  latent load-bearing state; *when in generation* a cache entry becomes dead;
  causal utility per cached position; why mid-sequence sinks appear; how junk
  interacts with long-generation drift (agentic serving papers report edited
  caches accumulating ~2.6e-2 relative error, AgentKV/Leyline line, but only
  behaviorally). Nobody has published a per-position *causal* utility
  timeline of a KV cache — exactly what lesion/patch tooling gives us.

### 1b. Memory / retrieval heads / induction heads in small models
- Induction-head formation got a formal treatment: **Mușat et al., "On the
  Emergence of Induction Heads for In-Context Learning" (arXiv:2511.01033)**
  — interpretable low-rank structure in forming weights + proofs;
  **"Predicting the Formation of Induction Heads" (NeurIPS 2025)** — a simple
  batch-size x context-size equation predicts *when* they form;
  "In-Context Meta Learning Induces Multi-Phase Circuit Formation" (ICML
  2025, arXiv:2505.16694) — phase-wise circuit assembly.
- Retrieval heads (Wu et al. 2024) now have 2025-26 follow-ups: SEAL
  (emphasis at inference), **CompressKV** (argues the top-1 retrieval-head
  identification rule is too rigid — head identity is unstable), DeCoRe
  (decoding by contrasting retrieval heads to cut hallucination), OLR-Heads
  (online position-aware head selection), "Retrieval Heads Meet Vision"
  (2026). Recurring complaint: *head role identity drifts* — formation and
  stability of retrieval heads is an open problem, and small-model training
  ladders (ours) are the cheapest place to watch them form.

### 1c. Catastrophic forgetting, 2025-26 state
- **Imanov et al., "Mechanistic Analysis of Catastrophic Forgetting in LLMs"
  (arXiv:2601.18699, 2026)** — claims first comprehensive mechanistic account
  across ~20 models: forgetting localizes as (a) "entropic dispersion" in
  early-layer attention heads and (b) localized representation collapse in
  mid-deep FFN/expert blocks; cause framed as parameter overwriting of
  ancestral circuits; proposes Low-Rank Circuit Projection, mitigating up to
  94.2% of ancestral capability loss. Caveat: single-author, big-model
  observational (CKA + routing drift), not causal interventional — our
  lesion/transplant methodology on tiny models is complementary and can test
  the overwriting story causally.
- Mitigation literature is engineering (OSFT orthogonal-subspace fine-tuning
  2026, LoRA, replay). Several 2025-26 analyses argue much "forgetting" is
  *recoverable* (latent, not deleted) — mbrenndoerfer.com essay — which
  rhymes with our surgical-forgetting S_name recovery results.

### 1d. Positional vs content binding in long context
- **NoLiMa (ICML 2025, arXiv:2502.05167)** — beyond literal lexical matching,
  performance drops >50% even for 1M-context models: associative binding
  over distance is the real failure, not retrieval per se.
- **Context Rot (Chroma, 2025)** — degradation is task-dependent and
  non-linear in context length across GPT-4.1/Claude/Gemini.
- Zheng et al. 2025 (attention-head survey): retrieval specialization and
  positional behavior of heads correlate with long-context ability;
  "Attention Sorting" work targets recency bias; content-aware positional
  embeddings (long-context survey arXiv:2503.17407) as a fix.
- **Not understood mechanistically**: why performance often falls off a
  *cliff* just past training length; whether the position-addressing pathway
  or the content-matching pathway degrades first; lost-in-the-middle has no
  circuit-level account. No causal component-patching study separating
  positional vs content subspace failure exists at any scale.

### 1e. Expression gap — knowing vs saying (teacher-forced vs free-run)
- **Orgad et al., "LLMs Know More Than They Show" (ICLR 2025,
  arXiv:2410.02707)** — probes on internal states decode the correct answer
  even when generation hallucinates ("elicitation gap"). NOTE: a public
  reproduction (Feijiang Han, X) claims code bugs — treat effect size with
  care; the qualitative gap is still widely cited.
- Exposure-bias literature is alive but training-side (Professor Forcing
  lineage; "Beyond Multi-Token Prediction" 2026 explicitly names the
  train-inference mismatch). **No mechanistic-interpretability paper found
  that causally dissociates teacher-forced correctness from free-generation
  failure** (where between residual stream and sampling does the correct
  answer die? is it copy-bias / prior dominance / attention re-entry?).
  Our T015/T018/T019 expression-gap box is genuinely near the frontier here.

### 2. Mechanistic unlearning / editing limits — who else found first-order fails?
- **Guo et al., "Mechanistic Unlearning: Robust Knowledge Unlearning and
  Editing via Mechanistic Localization" (ICML 2025, arXiv:2410.12949)** —
  circuit localization + delete-and-retrain; documents that *both* localized
  and non-localized (first-order/gradient) methods fail under prompt
  variation, format change, and relearning attacks. Closest big-model
  counterpart to our row-surgery + ascent-negative result.
- **RMU-is-obfuscation thread (2025)**: "LLM Unlearning Under the
  Microscope"; "Beyond Data Filtering" (RMU reverts to baseline forget loss
  in ~50 retraining steps); AntiDote (bi-level adversarial training,
  Sept 2025); SAE-subspace methods (SSPU) motivated by RMU's shallow
  suppression. Consensus 2025: representation-level unlearning hides rather
  than deletes — parametric surgery (ours) is the minority alternative.
- **Editing limits**: WikiBigEdit (lifelong editing butterfly-effect side
  effects); "Revisiting Parameter-Based Knowledge Editing: Theoretical Limits
  and Empirical Evidence" (parameter-editing has provable capacity limits);
  "Is Model Editing Built on Sand?"; MQuAKE-Remastered (ICLR 2025 — the
  multi-hop evaluation itself was flawed); REACT (May 2025) — chained
  representation edits fail to propagate; "Popular Knowledge Propagates More
  Errors" (2026) — hub entities amplify ripple damage. Nobody separates
  address/ability/expression/history as *distinct* failure modes — our
  four-box edit law is a novel decomposition.

### 3. Init-anchoring / LMC / seed lotteries — has anyone shown basis-frozenness under selection?
- **No.** Closest neighbors, none of which make the claim:
  - Lubana et al., "Mechanistic Mode Connectivity" — permutation symmetry
    (not basis) explains LMC between independently trained nets.
  - Frankle et al. LMC + Lottery Ticket — noise-stability locks in *early in
    training* (timing, not frozen basis under selection). MoE-LMC (NeurIPS
    2025) extends observational side.
  - "A Toy Model of Universality" (weak universality: same algorithm,
    different implementation across seeds); Arditi & Gurnee: low-level
    features do NOT align across models while high-level ones do.
  - Anthropic Crosscoders (2024-25): shared features across differently
    initialized models exist, but via learned dictionary, not raw basis.
  - "Small Initialization Matters for Large Language Models" (2026): early
    within-layer weight-vector alignment (condensation) — basis *forms* early
    but selection/evolvability is not studied.
  - Developmental interpretability survey (arXiv:2508.15841): SAE-Track
    feature-formation tracking; no selection-pressure experiments.
- **Verdict: E040's P2-FROZEN result (selection cannot see the stream basis;
  REF-alignment stuck at floor 0.006 vs 0.53 ceiling) appears novel and
  publishable as-is** — it is the mechanistic-evolvability experiment the
  universality literature lacks.

---

## 3 blocker candidates (big blocker, poorly understood origin, tiny-model-ready)

### Candidate A — KV-cache junk: when does a cached position become dead weight?
- **Known**: sinks are universal (14M models included), weight-decay-driven,
  load-bearing (removal hurts); softmax sum-to-one is the forcing function;
  quantization must protect sinks (KVSink); streaming fixes are engineering.
- **NOT understood**: causal utility of *individual non-sink* cache entries
  over generation time; whether "junk" accumulation is dead weight or
  latent state; why sinks appear mid-sequence; per-position dose-response.
- **1-line experiment this week**: in the 0.84M char model, generate 4k
  tokens, then causally patch/lesion each cached K/V entry (or contiguous
  blocks) at every causal depth and plot the "cache utility timeline" —
  registered prediction: utility collapses onto sink + recency window, and
  mid-sequence entries go dead early, quantifying cache "junk" causally
  (nobody has published this curve at any scale).

### Candidate B — the long-context cliff: does position-addressing or content-matching fail first?
- **Known**: context rot and NoLiMa show associative (non-literal) use of
  context collapses with length; recency/primacy biases; RoPE-extension
  hacks; retrieval-head positional behavior correlates with capability.
- **NOT understood**: mechanism of the sharp degradation past training
  length; no causal separation of positional-subspace vs content-subspace
  failure; lost-in-the-middle has no circuit account.
- **1-line experiment this week**: train the 0.84M model on key-value recall
  at context 512, sweep eval 512→2048 to locate the cliff, then reuse T021
  retrieval-threshold curves plus component patching (rotate/zero the
  positional components vs content components of the retrieval path) —
  registered prediction: content-matching survives past the cliff while
  position-addressing degrades smoothly and its threshold flip precedes the
  cliff.

### Candidate C — expression gap: where does a teacher-known answer die in free generation?
- **Known**: probes decode correct answers the model won't say (elicitation
  gap, ICLR 2025 — with reproduction caveats); multi-hop edits fail to
  propagate (MQuAKE line); exposure bias is documented only as a training
  mismatch, never dissected.
- **NOT understood**: the causal locus of the teacher-forced/free-running
  dissociation — is the correct answer suppressed by prior/copy dominance at
  the residual stream, overwritten by self-generated context re-entry, or
  lost in attention aggregation? No interventional study exists.
- **1-line experiment this week**: on facts the model addresses (S_name
  suite), compare teacher-forced logit margin vs free-run success rate, then
  transplant the teacher-forced residual-stream activation at the divergence
  token into the free-running pass — if behavior flips, knowledge was
  present-but-suppressed, and the depth of the transplant that works
  localizes the suppression circuit (direct extension of our transplant +
  four-box expression results).

*Runner-up*: sycophancy circuits (Wang et al. arXiv:2508.02087, AAAI 2026 —
opinion-driven override; arXiv:2509.21305 — sycophancy is causally
separable into multiple circuits). Origin (preference-data reward) is known;
the override circuit's formation is not — but char-level tiny models have no
"opinion", so it fits us poorly this week.

---

## Sources
- Gu et al., When Attention Sink Emerges (ICLR 2025): https://arxiv.org/abs/2410.10781 (project: https://guxm2021.github.io)
- KVSink (COLM 2025): https://arxiv.org/abs/2508.04257
- Peng et al., How Attention Sinks Emerge in LLMs (2026): https://arxiv.org (via search; arXiv 2026 listing)
- Mechanistic Analysis of Catastrophic Forgetting (2026): https://arxiv.org/abs/2601.18699
- Guo et al., Mechanistic Unlearning (ICML 2025): https://arxiv.org/abs/2410.12949 ; https://openreview.net/forum?id=92oBV5HAGl
- RMU obfuscation line: https://catniplab.github.io (Professor Forcing lineage); AntiDote & SSPU via https://huggingface.co (Daily Papers)
- WikiBigEdit / lifelong editing limits: https://tldr.takara.ai ; parameter-editing theoretical limits: https://openreview.net
- MQuAKE-Remastered (ICLR 2025) & REACT (2025): via https://arxiv.org search results
- Orgad et al., LLMs Know More Than They Show (ICLR 2025): https://arxiv.org/abs/2410.02707
- NoLiMa (ICML 2025): https://arxiv.org/abs/2502.05167
- Context Rot (Chroma 2025): https://www.trychroma.com/research/context-rot
- Laban et al., A Challenge to Long-Context LLMs and RAG Systems: https://arxiv.org/abs/2407.01370
- Long-context survey: https://arxiv.org/abs/2503.17407
- On the Emergence of Induction Heads: https://www.alphaxiv.org/abs/2511.01033v1
- Predicting the Formation of Induction Heads (NeurIPS 2025): https://neurips.cc/virtual/2025/129690
- Multi-Phase Circuit Formation (ICML 2025): https://arxiv.org/html/2505.16694v2
- Retrieval-head follow-ups (CompressKV, DeCoRe, OLR-Heads, SEAL): https://openreview.net ; https://github.com (Awesome-Attention-Heads)
- Wang et al., When Truth Is Overridden (AAAI 2026): https://arxiv.org/html/2508.02087v1
- Sycophancy causal separation: https://arxiv.org/pdf/2509.21305
- Anthropic, Towards Understanding Sycophancy: https://arxiv.org/pdf/2310.13548
- Anthropic, attribution graphs: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Frankle et al., LMC & Lottery Ticket: https://arxiv.org/abs/1912.05671
- Lubana et al., Mechanistic Mode Connectivity: https://ntt-research.com/wp-content/uploads/2023/02/Mechanistic-Mode-Connectivity.pdf
- MoE Linear Mode Connectivity (NeurIPS 2025): https://neurips.cc/virtual/2025/poster/118035
- Small Initialization Matters (2026): https://arxiv.org (2026 listing)
- Developmental Interpretability review: https://arxiv.org/pdf/2508.15841
- Crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
- AgentKV / Leyline / agentic KV error accumulation: https://openreview.net ; https://www.researchgate.net
- OSFT (2026): https://developers.redhat.com/articles/2026/07/28/osft-explained-prevent-catastrophic-forgetting-llm-fine-tuning
- Forgetting-is-recoverable essay: https://mbrenndoerfer.com/writing/catastrophic-forgetting-fine-tuning-mitigation
