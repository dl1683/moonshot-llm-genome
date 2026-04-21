# Email to VERSES AI

**To:** Gabriel René, Dan Mapes, Karl Friston
**Subject:** something you'll want to see before we open-source it

---

Hey Gabriel (and Dan, and Karl),

Sending this before we push it public because I think the overlap with VERSES is close to 1-to-1.

At CMC we've been running a research program called AI Moonshots. Two of the projects lay out where we're coming from:

- **CTI (Compute Thermodynamics of Intelligence)** — a universal performance law for trained networks, derived from extreme-value theory *before* any fitting. The same constant (ρ ≈ 0.45, CV=1.65%) holds across 12 NLP architectures AND mouse V1 Neuropixels recordings — artificial and biological systems produce the same equicorrelation invariant under matched stimuli. https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — 2 random embedding-scale tokens, no training, jumps small-LLM arithmetic from 32% → 51.6%. An activation-level intervention on the geometry of the inference state. https://github.com/dl1683/Latent-Space-Reasoning

The bigger project we're working on now is **Neural Genome** — a cross-architecture atlas of representational geometry. The question is whether the same mathematical coordinates describe the internal organization of a transformer, a Mamba, a ViT, a BERT, *and* a biological cortical population. If the answer is yes, we've got an empirical geometric signature of inference itself — architecture-agnostic, provable, testable on biology.

What we've shown in the last two weeks (single RTX 5090 laptop, all pre-registered, Bonferroni-corrected):

1. **One geometric coordinate passes portability across 8 architecture classes + 5 distinct training objectives** — CLM, reasoning-distilled, linear-attention recurrent, hybrid Mamba, MLM, contrastive-text, self-sup-ViT, contrastive-vision. Same number within strict tolerance.
2. **Survives 4× weight quantization** — the structure doesn't live in the weights.
3. **Causally load-bearing** — ablating the subspace blows up model loss +7.8% to +443% across architectures; random ablations of identical dimensionality move loss <1%. Specificity 20-66×. A real geometric mechanism, not a descriptive correlation.
4. **4-7× above random-Gaussian baseline** — not an artifact of high dimensions.
5. **Theoretically grounded** — locked a first-principles derivation from Laplace-Beltrami convergence of kNN graphs *before* seeing the empirical fit. Prediction held.

**Why VERSES specifically.** The Free Energy Principle is a universality claim about inference structure — same mathematical object describes brains and artificial systems. Our atlas is an empirical test of a closely-related claim using actual networks + actual biological recordings. If it holds, it's the strongest empirical evidence FEP has ever had outside its native neuroscience domain — a geometric invariant observable in inference systems regardless of substrate. If it fails, we learn exactly where the universality breaks, which is also a publishable contribution to your framework.

More concretely: we want Genius in the bestiary. If kNN-k10 passes on an active-inference / generative-world-model architecture the way it does on transformers and recurrent LMs, that's a signature of active inference that nobody else has — geometric positioning that says VERSES is *measurably* doing the same inference math biological cortex does, grounded in a coordinate system that the whole syndicate cites. If it *doesn't* pass, we've found the structural difference between active-inference systems and scale-based systems, and that's an even bigger paper for VERSES because it puts numbers on the thesis Gabriel has been making for years.

To be honest — this is a moonshot. It might turn out the universality is narrower than we hope, or the instrument needs an iteration we haven't seen yet. That's the nature of ambitious research and we name it that way. But CTI and Latent Space Reasoning both delivered results the field didn't expect, and if Neural Genome lands similarly, this is the DeepSeek moment for non-scale AI — a mathematical case that intelligence is about geometric structure, not parameter count, backed by the first cross-class + cross-species empirical evidence. The fame side is real. The positioning side (VERSES as the commercial anchor of the post-scale paradigm, with geometric receipts rather than just a thesis) is much bigger.

CMC is putting together a research syndicate — us + Martian (mechanistic interp) + Furiosa (inference hardware) + Weka (context memory) + you. Active inference + representational geometry + inference hardware + data substrate = the full stack for proving the non-scale thesis. Before the open-source drop in a few weeks, we'd rather have VERSES in on the long-term roadmap than shipping a paper you read afterward.

Karl — if any of this sounds wrong-headed from an FEP perspective I really want to hear where. That's exactly the feedback that makes the research better.

30-min call this week?

Dev
