I checked the repo docs and current g188 artifact. g188 is still `INCOMPLETE`; only preprocessing is saved, with `43,423` sparse alignment edges and string match covering `84.1%` of GPT-2 tokens.

**1. Why trained geometry works but PPMI SVD doesn’t**

Because the useful object is not “token semantics.” It is an architecture-conditioned interface format.

PPMI SVD gives a corpus-statistical embedding: symmetric co-occurrence, arbitrary SVD basis, no decoder, no autoregressive loss, no input-output logit coupling. Trained embeddings are shaped by the transformer, RMSNorm, lm_head, optimizer, token frequency, and next-token gradients. They are a codebook that the decoder has learned how to read and write through.

So g183 is decisive: PPMI has some semantic content, but the wrong chart. Anchoring to it creates a gradient conflict. This matches [WIKI.md](</c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:43>) and [OPEN_MYSTERIES.md](</c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/OPEN_MYSTERIES.md:163>).

Also: C23 is not proven content transfer. The best interpretation is format transfer: “Qwen-trained interface prior held in place,” not donor-specific knowledge.

**2. Current SOTA / prior art**

Crowded, but not identical to g188.

- [ULD](https://arxiv.org/abs/2402.12030): OT-based logit distillation across tokenizers/architectures.
- [MultiLevelOT](https://arxiv.org/abs/2412.14528): token-level plus sequence-level OT/Sinkhorn; accepted AAAI 2025 oral.
- [CDM](https://aclanthology.org/2025.findings-acl.419/): contextual dynamic mapping for sequence and vocabulary mismatch.
- [EMO](https://aclanthology.org/2025.emnlp-main.385/): cross-tokenizer embedding-model distillation using MinED, CKA attention alignment, and OT hidden-state alignment.
- [Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083): NeurIPS 2025; strong cross-tokenizer distillation, including subword-to-byte transfer and embedding prediction hypernetworks.
- [Embedding relearning/swapping](https://aclanthology.org/2026.eacl-long.357/): EACL 2026; adapting decoder-only LLMs to new tokenizers by relearning embeddings.
- [SEDI](https://api.repo.agh.edu.pl/server/api/core/bitstreams/64862e22-25de-4269-8882-5ea302c9a6fa/content): AAAI 2026; bipartite graph alignment plus sliding-window re-encoding and entropy alignment.

Novel gap: these mostly transfer logits/behavior or adapt tokenizers with training. g188 tests whether a trained interface embedding itself can be transcoded into a new tokenizer as an anchor prior.

**3. If g188 passes**

Highest-leverage next step: not “bigger g188.” Do `g189: tokenizer-transcoding law`.

Pre-register a frozen predictor of recovery:
`recovered_nats ~ transport_mass + boundary_overlap + plan_entropy + source_training_geometry + tokenizer_distance`

Then test held-out tokenizer pairs and controls. The claim needed for §0.1 7+ is:

“Cross-tokenizer transfer succeeds exactly when a trained interface codebook can be mapped into the recipient tokenizer chart; corpus stats and naive KD fail because they lack decoder-format alignment.”

After that, cash out with a small compute-saving demo: same NLL in fewer steps using the bridge.

**4. If g188 fails**

Codebook+decoder is not dead. Static bridge is dead.

Pivot to decoder-conditioned relearning: freeze a trained decoder, swap tokenizer, relearn only embed/lm_head, then use that learned interface as the anchor prior. If that works, the bridge needs gradients, not static OT. If that fails, then the codebook and decoder are co-adapted too deeply for transplant, and the contribution becomes a negative law: architecture charts are not directly commensurable.

In parallel, g187 becomes more important: characterize the architecture-specific chart as a diagnostic, not a transfer mechanism.

**5. Rating**

Current Mystery-8/g188 direction:

- Novelty: 7/10. Stronger than generic cross-tokenizer KD because it targets interface geometry, but the neighborhood is active.
- Breakthrough probability: 4/10 now; 6/10 if g188 passes cleanly; 7+ only with a prospective law plus compute-saving demo.
- Manifesto alignment: 8/10. It is squarely about model surgery, architecture charts, and training efficiency. It drops to 5/10 if it becomes just another descriptive geometry paper.

Bottom line: Mystery 8 is the right gold mine. The honest thesis is no longer “universal geometry transfers.” It is “learned systems have architecture-specific charts; the breakthrough is discovering when and how charts can be translated.”

