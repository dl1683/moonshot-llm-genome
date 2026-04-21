# Email to Martian

**To:** Shriyash, Etan
**Subject:** something you'll actually want to see

---

Hey guys,

Wanted to send this before I push it public. I've been working on a research program called AI Moonshots and two of the projects are directly in your wheelhouse:

- **CTI (Compute Thermodynamics of Intelligence)** — universal law derived from EVT *before fitting* any model, passes across 12 NLP architectures AND mouse V1 Neuropixels (same constant, CV 1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — prepending 2 random embedding-scale tokens (no training) takes Qwen3-4B arithmetic from 32% → 51.6%. A new axis orthogonal to scaling/FT/prompting. https://github.com/dl1683/Latent-Space-Reasoning

Both of those were the warm-up. The real thing I'm building now is called **Neural Genome** — a cross-architecture atlas of representational geometry. Basically: is there a shared coordinate system that describes what's happening inside a transformer, a Mamba, a ViT, and a BERT?

First real result from this week: one coordinate (a specific kNN-based geometric measure) passes portability testing at strict threshold across 8 architecture classes and 5 training objectives, survives 4× quantization, and when I ablate the subspace it identifies the model's loss blows up 50-400% while random ablations barely move it. Basically a causal interpretability handle that's architecture-agnostic.

I'm going to open-source this in a few weeks but I wanted to get you guys in first. You're doing model-mapping and router interpretability — this is literally the theoretical instrument that would make your "circuits in Model A are homologous to circuits in Model B" claims *provable* instead of empirical. And I think the long-term roadmap is much more interesting if we're building it together rather than me publishing and you reading it later.

Proposing a research syndicate — you + Furiosa (hardware) + Weka (context memory) + me. Co-author the paper, share infra, move faster than any of us could alone.

30-min call this week?

Devansh
