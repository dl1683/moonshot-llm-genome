# Positional binding of installed knowledge — literature scan (2026-09-25)

Scope: 2023–2026 work on positional vs content binding, causal position-matched
transplants, address-vs-feature framings, and the long-context connection.
Context for lab findings T032/T033/T035: knowledge installed by training is
bound to wpe-130 (one-char shift kills expression); d4 state-rescue works only
at the bound position (0.374 vs 0.000 shuffled); transplanted elsewhere it is a
one-token logit blip.

## Prior art

**In-context binding via position-like indices (the strongest thread).**
- Feng & Steinhardt, "How do Language Models Bind Entities in Context?"
  (arXiv:2310.17191). In-context entity-attribute binding is solved by
  "binding ID" vectors: content-independent vectors attached to entities and
  attributes, forming a continuous subspace. This is the canonical modern
  statement that binding is done by arbitrary, content-independent indices
  rather than by content similarity. Follow-on work (e.g., ICLR 2026
  "Decoupling Positional and Symbolic Attention") argues these binding IDs
  are effectively positional.
- Smolensky, Fernandez, Xue et al., "Mechanisms of Symbol Processing for
  In-Context Learning in Transformer Networks" (arXiv:2410.17498, NeurIPS
  2024-era). First mechanistic account of symbol manipulation in ICL: binding
  uses position index vectors (associated with attention positions/keys) to
  encode symbol-role combinations. Tensor-product-flavored.
- Dai, Heinzerling, Inui, "Representational Analysis of Binding in Language
  Models" (arXiv:2409.05448). Localizes an "Ordering ID" linear subspace that
  causally determines which attribute an entity is bound to in entity-tracking
  tasks; extended by "Cell-Based Representation of Relational Binding"
  (arXiv:2604.19052) with causal patching of entity-relation "cells".
- Kim et al., "Task Schema and Binding: A Double Dissociation Study of
  In-Context Learning" (arXiv:2512.17325, Dec 2025). Activation-patching
  separation of task-schema vs binding mechanisms across 9 models incl. Mamba.

**Positional shortcuts / position dominance.**
- Clark et al., "What Does BERT Look At?" (arXiv:1906.04341). Many attention
  heads are fully explained by position-only baselines (adjacent-token,
  [SEP]/[CLS] attendance) — the original "position dominates content" result.
- "Seeing to Generalize: How Visual Data Corrects Binding Shortcuts"
  (arXiv:2602.15183, ICML 2026-era). Text-only models bind entities using
  brittle WHERE-it-is shortcuts — a "positional circuit" that "transfers only
  position indices"; visual training disrupts it. Direct modern "positional
  shortcut for binding" evidence.
- Kazemnejad et al., "The Impact of Positional Encoding on Length
  Generalization in Transformers" (arXiv:2305.19466, NeurIPS 2023). NoPE
  (no positional encoding) generalizes to longer sequences better than
  learned/APE schemes. Flip side of our finding: when absolute position is
  unavailable the model is forced into content-based addressing; when wpe is
  available (our GPT-2-style char-LM), training exploits absolute position as
  a cheap address.
- Abacus Embeddings ("Transformers Can Do Arithmetic...", NeurIPS 2024) and
  Position Coupling (NeurIPS 2024): fixing position-number entanglement for
  length generalization in arithmetic.
- Knowledge editing: FiNE (arXiv:2503.01090, ICLR 2025) reports edited
  knowledge is often sensitive only to the subject entity (entity-bound, not
  position-bound, but same flavor of brittle addressing); RelEdit (ACL
  Findings 2025) evaluates relational generality of edits.

## Closest to our transplant

- **No direct precedent found.** Nobody (in what surfaced) transplants an
  activation state at a matched position vs a shifted position to test
  position-binding of installed knowledge. Our matched-vs-shifted + shuffled
  control design appears unclaimed.
- Meng et al., ROME "Locating and Editing Factual Associations in GPT"
  (arXiv:2202.05262). Implicit nearest prior: causal-tracing patching rescues
  fact recall only when injected at the last subject token — factual recall
  is position-anchored to "where the subject ends". But ROME never states a
  position-binding law, never shifts the transplant, and targets pretrained
  facts, not freshly installed knowledge.
- Todd et al., "Function Vectors in Large Language Models" (arXiv:2310.15213,
  ICLR 2024). The key CONTRAST case: task/function vectors are
  position-invariant — effective when injected at arbitrary positions and
  under token-order shuffling. So the field's default assumption for
  transplantable states is position-robustness; our installed-knowledge state
  is the opposite (position-bound, one-token blip when shifted). This
  contrast is itself publishable framing.
- Wu, Geiger, Millière, "How Do Transformers Learn Variable Binding in
  Symbolic Programs?" (arXiv:2505.20896, ICML 2025). Causal interventions
  show the residual stream is used as an "addressable memory space" with
  attention heads routing values across token positions; training passes
  through heuristic phases before genuine chain-dereferencing. Closest in
  spirit (addresses + routing by position) but for an ICL-trained symbolic
  program, not parametric knowledge, and no transplant-off-position test.
- Position-dependence of steering (supporting, weaker): "Steer Like the LLM"
  (arXiv:2605.03907) finds prompting induces position-varying steering
  strength, and estimates per-position coefficients; Conceptors steering
  (arXiv:2410.16314) gates steering softly across token positions; KV-cache
  steering (2025) intervenes per token position. These show steering effects
  are position-modulated but do not quantify binding of a knowledge state.

## Address-vs-feature framing

- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories"
  (arXiv:2012.14913) + "Transformer Feed-Forward Layers Build Predictions by
  Promoting Concepts in the Vocabulary Space" (2021): FFN keys = patterns,
  values = next-token promotions — the static KV-memory view of knowledge.
- Wu et al. 2505.20896 (above): residual stream as addressable memory — the
  most literal "address" claim in the mechanistic literature.
- Feng & Steinhardt 2310.17191: binding IDs = addresses in a learned
  subspace; Smolensky et al. 2410.17498: position index vectors = addresses.
- Fast-weight lineage: Schlag/Irie/Schmidhuber "Enhancing the Transformer
  with Explicit Memory" (motivated by attention's binding problem);
  Katharopoulos et al. linear attention as fast-weight programmers;
  linear-attention-as-iterated-Hopfield equivalence (2024) — associative
  memory (outer-product) views where keys are addresses and recall is
  content-addressable. In that theory language, our result says: the address
  key the network learned for the installed fact is (wpe-130, content) —
  a conjunction — so pure content probing finds the feature (logit blip) but
  only the matched address retrieves it into behavior.

## Long-context connection

- NoLiMa (arXiv:2502.05167, ICML 2025): performance collapses when literal
  matching is unavailable; attribution is behavioral ("attention struggles
  without lexical overlap"), mechanistic analysis explicitly deferred as
  out of scope. IMPORTANT for us: their needle-in-last-2K control holds
  relative question-needle distance constant across lengths and degradation
  persists — they use this to argue RoPE relative distance does not explain
  the drop, i.e., total-context/competition effects dominate. A
  relative-position-binding story is therefore NOT supported for NoLiMa; an
  absolute-binding/weak-associative-address story is untouched and untested.
- Context rot (Chroma technical report, July 2025, research.trychroma.com):
  monotonic degradation with input length across tasks, incl. mid-length
  ranges; cites no mechanism. Follow-ons (e.g., APCE, arXiv Oct 2025) treat
  the symptom, not the mechanism.
- Classic: "Lost in the Middle" (Liu et al., 2023/TACL 2024) — position of
  relevant info in context predicts recall (a positional-binding-of-context
  phenomenon, pretrained, not causal).
- Gap: we found NO paper that connects position-bound addressing of knowledge
  to NoLiMa/context-rot degradation. NoLiMa's own analysis section says the
  mechanism question is open. If our lab can show (in the char-LM) that
  associative (non-literal) retrieval specifically fails when the address
  must be recomputed at a shifted position, that is a candidate mechanistic
  primitive for the NoLiMa/context-rot phenomenon — with the caveat that
  NoLiMa's last-2K control already rules out simple distance-based binding.

## Novelty verdict

**Claimable as novel, with a sharply defined perimeter.**
1. NOT novel: that transformers use position-like, content-independent
   indices for in-context binding (Feng & Steinhardt 2310.17191; Smolensky
   et al. 2410.17498; Dai et al. 2409.05448); that attention is
   position-dominated (Clark 2019); that positional shortcuts cause brittle
   generalization (2602.15183; NoPE 2305.19466); residual stream as
   addressable memory in a trained symbolic task (2505.20896).
2. Novel (no prior found): (a) knowledge INSTALLED INTO WEIGHTS by training
   being bound to one absolute learned position embedding (wpe-130), with a
   one-character address shift abolishing expression; (b) the causal
   matched-position vs shifted-position STATE TRANSPLANT with shuffled
   control (0.374 vs 0.000) as the assay — ROME implies position-anchoring
   but never runs the shift; (c) the "logit blip vs behavioral rescue"
   dissociation showing the feature exists off-address but cannot drive
   behavior — this inverts the Function-Vector literature's position-
   invariance default.
3. Positioning advice: frame against Todd et al. 2310.15213 (task vectors
   are position-free) and Wu et al. 2505.20896 (addressable memory for ICL)
   as the two poles; our contribution is that TRAINING-written knowledge
   adopts the cheapest available address — the absolute positional embedding
   — when the task distribution permits, which NoPE results suggest is a
   real attractor of learned positional encodings.
4. Risk: reviewers may fold this into "shortcut learning" (Geirhos lineage)
   or "binding ID" work; the transplant assay and the free-run expression
   failure are the defensible specifics.

## sources

- Feng & Steinhardt, arXiv:2310.17191 — https://arxiv.org/abs/2310.17191
- Smolensky et al., arXiv:2410.17498 — https://www.arxiv.org/abs/2410.17498
- Dai, Heinzerling, Inui, arXiv:2409.05448 — https://arxiv.org/abs/2409.05448
- Cell-Based Relational Binding, arXiv:2604.19052
- Kim et al. double dissociation, arXiv:2512.17325
- Clark et al., arXiv:1906.04341 — https://ar5iv.labs.arxiv.org/html/1906.04341
- Seeing to Generalize, arXiv:2602.15183 — https://www.alphaxiv.org/abs/2602.15183
- Kazemnejad et al. (NoPE), arXiv:2305.19466 — https://arxiv.org/abs/2305.19466
- Meng et al. (ROME), arXiv:2202.05262
- Todd et al. Function Vectors, arXiv:2310.15213 — https://arxiv.org/abs/2310.15213
- Wu, Geiger, Millière, arXiv:2505.20896 — https://arxiv.org/abs/2505.20896
- Geva et al. FFN-KV, arXiv:2012.14913
- FiNE, arXiv:2503.01090 — https://arxiv.org/abs/2503.01090
- RelEdit, ACL Findings 2025 — https://aclanthology.org/2025.findings-acl.533.pdf
- NoLiMa, arXiv:2502.05167 (ICML 2025) — https://arxiv.org/abs/2502.05167
- Context rot, Chroma tech report July 2025 — https://research.trychroma.com
- Steer Like the LLM, arXiv:2605.03907 — https://arxiv.org/html/2605.03907v1
- Conceptors steering, arXiv:2410.16314
- Abacus embeddings, NeurIPS 2024 — proceedings.neurips.cc paper c35986bc...
- Position Coupling, NeurIPS 2024 — neurips.cc/virtual/2024/poster/96579
