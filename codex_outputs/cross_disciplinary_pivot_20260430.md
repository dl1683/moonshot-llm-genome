I read the requested files and checked the current result state. My core read: stop treating “geometry” as the transferable object. The transferable object is more likely an **interface codebook plus decoder**. Within Qwen3, donor and recipient share that codebook. Cross-architecture fails because the same geometric statistic is being decoded through a different tokenizer/architecture channel.

**1. Neuroscience**
- Principle: critical periods + gated plasticity. Early sensory interfaces lock first; slower internal pathways adapt around them.
- Connection: embed/lm_head is the model’s sensory/motor interface. g181b says that interface prior survives; g181a says transformer-block anchoring harms. That looks like “right interface, wrong slow-path constraint.”
- Experiment: **critical-period anchor sweep**. Same total anchor force, varied timing: early-only, late-only, continuous, release-after-500, replay-only, scratch. If early-only preserves most of the +0.513 nat gain, interface geometry is a critical-period prior.

**2. Fungal / Mycorrhizal Networks**
- Principle: resources transfer through adaptive conduits, not through globally identical topology. Useful paths are grown by flux and repaired by rerouting.
- Connection: cross-arch fails because there is no conduit between codebooks. Shared-token row matching is too brittle.
- Experiment: **tokenizer-flow bridge**. Tokenize the same C4 spans with Qwen and GPT-2/Falcon tokenizers, build a bipartite token co-occurrence graph, run OT/diffusion, initialize recipient embeddings as barycentric donor embeddings. Compare scratch, direct shared-token anchor, random-flow anchor, graph-flow anchor. A 20% recovery of the Qwen effect is already meaningful.

**3. Quantum Physics**
- Principle: logical information can be nonlocal and code-dependent. Error correction works only with the right decoder.
- Connection: within-family transfer may work because the “logical code” is shared; cross-family fails because we apply the wrong decoder to the same apparent correlations.
- Experiment: **interface redundancy assay**. Corrupt trained embed/lm_head anchors by row dropout, dimension dropout, low-rank projection, and row permutation. If benefit degrades gracefully under erasure but collapses under token permutation, the interface is a redundant codebook with token identity as the decoder key.

**4. Alternative Number Systems**
- Principle: Euclidean row geometry may be the wrong basis. Language/tokenization is hierarchical, closer to ultrametric or hyperbolic structure than flat vector space.
- Connection: g180b/g182 may fail because they compare leaf nodes across incompatible trees. Coarse hierarchy may transfer even when exact tokens do not.
- Experiment: **vocab Haar / ultrametric bridge**. Build a byte/BPE/corpus co-occurrence token tree, transform embeddings into hierarchical wavelet coefficients, transfer only coarse coefficients across tokenizers, then optionally refine. If coarse transfer works and leaf-shuffled transfer fails, the missing object is hierarchy, not raw geometry.

**5. Coding Theory**
- Principle: the trained interface is a codebook; cross-system transfer requires rate-compatible transcoding. Successive refinement says coarse structure should transfer before fine detail.
- Connection: g183 is the right rescue: can a corpus-derived codebook replace a trained donor? g186 failed because generic geometry predicted alpha, not final value.
- Experiment: **successive-refinement codebook ladder**: unigram frequency → PPMI SVD → PPMI+position → teacher-logit clusters → trained Qwen embed. Same seeds, same anchor lambda, same final NLL metric. Plot gain vs estimated code rate/distortion. This could become a real theory result.

**6. Combinatorial Headline Strategy**
Best headline is not “universal transfer works.” It is:

“Cheap interface codebooks recover useful training efficiency; cross-architecture transfer needs a topology-aware bridge.”

Package 2-3 findings:
- g183-style corpus codebook recovers ≥50% of the within-Qwen trained-anchor gain.
- tokenizer-flow or ultrametric bridge recovers ≥20% cross-tokenizer/cross-arch gain, about 0.10 nats against the +0.513 reference.
- late/cheap KD or simple dose law gives a compute-saving recipe without pretending geometry forecast passed.

**Top 3 By Expected Breakthrough Value**
1. **Successive-refinement codebook ladder**: P≈0.45, impact 8/10, E≈3.6. Best immediate fit with g183 and the manifesto.
2. **Tokenizer-flow / ultrametric bridge**: P≈0.30, impact 9/10, E≈2.7. Highest upside for cross-architecture resurrection.
3. **Critical-period anchor sweep**: P≈0.55, impact 4.5/10, E≈2.5. Most likely to explain why interface geometry matters, even if not a headline alone.

My recommendation: do not start with quantum or complex-valued networks. Start with **coding theory + ultrametric token topology**. That is the shortest path from “cross-arch fails” to “we found the missing decoder.”

