# Email to Weka

To: Liran (and/or Nilesh)
Subject: want to loop you in on this before we open-source it

---

Hey Liran,

Quick one before we push this public.

Core thesis: geometric manipulation of representations produces capabilities that scale-alone cannot. Three of our recent AI Moonshots each prove this from a different angle:

- Latent Space Reasoning — activation-level intervention (2 random tokens, no training) jumping Qwen3-4B arithmetic accuracy 20 points. https://github.com/dl1683/Latent-Space-Reasoning
- CTI (Compute Thermodynamics of Intelligence) — a first-principles universal law predicting model performance from compute. Same constant across 12 NLP architectures AND mouse V1 Neuropixels (CV=1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- Fractal Embeddings — hierarchy-aligned progressive prefix supervision delivers +5.36% L0 / +6.47% L1 accuracy with 5-seed validation; random-hierarchy causal control drops to −0.10%. Geometric prior alone produces emergent hierarchical capability.

All three shipped with results nobody expected. Geometry beats scale, reproducibly, with causal controls. The project I'm writing you about is Neural Genome — cross-architecture atlas of representational geometry, testing whether the same mathematical coordinates describe the internal organization of transformers, Mambas, ViTs, and BERTs.

The last two weeks (one RTX 5090 laptop, pre-registered, Bonferroni-corrected):

1. Candidate-8 spectral bridge — c ≈ eff_rank(X) / d_rd(X) holds within 15% on 7 of 8 trained systems (CLM decoders, MLM encoders, contrastive, vision ViT, CLIP both branches). Median rel_err 8.7%. One equation, two independent geometric measurements, one universal identity.
2. Survives 4× quantization (FP16 → Q8) at even tighter tolerance — the structure doesn't live in bit-width.
3. Universal bulk width k_bulk ≈ h/22 (CV 4.2% across 5 text systems) — a structural invariant that tells us exactly how many "informational directions" a trained text model actually uses.
4. GenomeGuard — a 20-second training health monitor that uses the bridge to detect silent data corruption across 5 architectures with 6.9× – 144.9× signal (DeepSeek is most sensitive at 144.9×, BERT at 23.7×). Catastrophic weight-divergence at 8× signal. Ship-ready in under 300 lines.
5. 12-op null catalog: every forward geometric manipulation we tested for installing capability fails. Capability is joint weight configuration, not any separable linear or geometric target.
6. Cross-substrate (today): bridge holds on mouse V1 Neuropixels under Natural Movie One at rel_err 12.3% — the same identity observed on the 7 ML systems. Not a training-recipe artifact.

Why I think this matters for Weka. Your whole NeuralMesh / Augmented Memory Grid thesis is that reasoning workloads need more context than HBM can hold, so you stream KV-cache intelligently between GPU memory and the petabyte token warehouse. The hard question underneath that is which activations are compressible and which are essential — today that's a per-model guess. We think the bridge answers it from first principles.

Specifically: the same eff_rank / d_rd coordinate that tells Furiosa which layers tolerate INT4 also tells NeuralMesh which KV slices are safe to evict cheaply vs must stay hot. The universal bulk width k_bulk ≈ h/22 says there are ~48 principal directions per 1024-d layer doing the heavy lifting — a cache policy targeting those specific directions at higher precision, and everything else at aggressive compression, should be provably near-optimal rather than heuristic.

This is where I want to be honest: the KV-caching application is an instinct, not a proven result yet. I want to explore it with your team, not pitch it as a done deal. But if the instinct is right, NeuralMesh gets a geometry-aware caching policy that's provably near-optimal — and you're the only data-platform company in position to own that story.

This is a difficult project but we think we can pull it off. Three precursors delivered real, reproducible results that surprised the field, the Neural Genome bridge is already holding on 7/8 systems, GenomeGuard is shipping cross-arch, and the 12-op null is already in hand. The KV-caching application specifically is still a hypothesis I want to test with your team, not pitch as a done deal — but the underlying geometric structure is no longer speculative. If it lands the way we think it might, this is the DeepSeek moment for efficient intelligence — a mathematical argument that says "intelligence doesn't require a data center, it requires better geometry," and a commercial story that re-positions everything downstream. For Weka that means becoming the data platform for reasoning-scale AI, not just a storage vendor to NVIDIA's stack.

CMC is putting together a research syndicate — us + Martian (interpretability) + Furiosa (inference hardware) + Liquid AI (efficient architectures) + VERSES (active inference) + you. Before we push the atlas open-source in a few weeks, I'd rather have Weka in on the roadmap than read-afterward.

Would love to talk further if that's useful.

Dev
