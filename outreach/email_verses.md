# Email to VERSES AI

**To:** Gabriel René, Dan Mapes, Karl Friston
**Subject:** something you'll want to see before we open-source it

---

Hey Gabriel (and Dan, and Karl),

Sending this before we push it public because I think the overlap with VERSES is close to 1-to-1.

**Core thesis: geometric manipulation of representations produces capabilities that scale-alone cannot.** Three of our AI Moonshots projects converge on this — each a separate proof-of-existence that geometric structure, not parameter count, is where capability actually lives:

- **Latent Space Reasoning** — 2 random embedding-scale tokens prepended at inference, no training, jumps Qwen3-4B arithmetic 32% → 51.6%. An activation-level intervention on the inference state's geometry. https://github.com/dl1683/Latent-Space-Reasoning
- **CTI (Compute Thermodynamics of Intelligence)** — a universal performance law derived from extreme-value theory *before* any fitting. Same constant (ρ ≈ 0.45, CV=1.65%) across 12 NLP architectures AND mouse V1 Neuropixels — artificial and biological systems producing the same equicorrelation invariant under matched stimuli. https://github.com/dl1683/moonshot-cti-universal-law
- **Fractal Embeddings** — hierarchy-aligned progressive prefix supervision delivers +5.36% L0 / +6.47% L1 accuracy (5-seed validation, random-hierarchy causal control drops to −0.10%). Geometric prior alone produces emergent hierarchical capability.

Each one says the same thing from a different angle: **geometry beats scale**, reproducibly, with causal controls — and the CTI result says it explicitly reaches into biology.

The project I'm writing about is **Neural Genome**. It takes the thesis all the way: is there a single universal coordinate system for this geometric substrate across every trained network, artificial *or* biological? We found one. `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on **7 of 8 trained systems** spanning CLM decoders, MLM encoders, contrastive, vision ViT, and cross-modal alignment. Median 8.7% relative error, all pre-registered.

The result's texture is perfect for the FEP worldview:

1. **The same spectral bridge on 7 of 8 ANN architectures.** Training produces a specific covariance-decay signature (α ≈ 0.86 power-law, bulk width ≈ h/22 universal at CV 4.2%), and the bridge is one equation relating two independent geometric summaries of the same activation cloud.
2. **Stimulus-sensitivity characterizes the substrate.** The bridge breaks at 3-45× rel_err when the input distribution shifts out of the training distribution — we turned this into **GenomeGuard**, a 20-second detector for silent data corruption across 5 architectures. Stimulus-conditional is a feature, not a bug: inference depends on the environment the system is in, and the substrate signature reflects that.
3. **12-operation null catalog for forward capability transfer.** Every way we tried to *install* capability via geometric manipulation (covariance transfer, codebook, PCA basis, weight-subset transplants, aux-loss) fails. Capability is irreducibly joint weight configuration. In FEP language: the generative model cannot be reduced to a geometric summary — the full probabilistic machinery is load-bearing.

**Why VERSES specifically.** The Free Energy Principle is a universality claim about inference structure — same mathematical object describes brains and artificial systems. Our atlas is an empirical test of a closely-related claim on actual networks + actual biological recordings. If the bridge extends to Allen V1 Neuropixels (probe in flight today), it's the strongest empirical evidence FEP has ever had outside its native neuroscience domain — a geometric invariant observable in inference systems regardless of substrate. If it fails on biology, we've found exactly where the artificial-vs-biological universality breaks — also a publishable contribution to your framework.

More concretely: we want Genius in the bestiary. If the bridge passes on an active-inference / generative-world-model architecture the way it does on transformers and recurrent LMs, that's a signature of active inference that nobody else has — geometric positioning that says VERSES is *measurably* doing the same inference math biological cortex does, grounded in a coordinate system that the whole syndicate cites. If it *doesn't* pass, we've found the structural difference between active-inference systems and scale-based systems, and that's an even bigger paper for VERSES because it puts numbers on the thesis Gabriel has been making for years.

This is a difficult project but we think we can pull it off. The bridge is already holding on 7/8 trained systems, the shipping tool (GenomeGuard) is already working cross-architecture, and the 12-op null catalog is a hard negative claim that frontier labs can't write up. If the biology extension lands, this is the **DeepSeek moment for non-scale AI** — a mathematical case that intelligence is about geometric structure, not parameter count, backed by the first cross-class + cross-species empirical evidence. The fame side is real. The positioning side (VERSES as the commercial anchor of the post-scale paradigm, with geometric receipts rather than just a thesis) is much bigger.

CMC is putting together a research syndicate — us + Martian (mechanistic interp) + Furiosa (inference hardware) + Weka (context memory) + Liquid AI (continuous-time models) + you. Active inference + representational geometry + inference hardware + data substrate + efficient architecture = the full stack for proving the non-scale thesis. Before the open-source drop in a few weeks, we'd rather have VERSES in on the long-term roadmap than shipping a paper you read afterward.

Karl — if any of this sounds wrong-headed from an FEP perspective I really want to hear where. That's exactly the feedback that makes the research better.

Would love to talk further if that's useful.

Dev
