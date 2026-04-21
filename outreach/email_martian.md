# Email to Martian

**To:** Shriyash, Etan
**Subject:** something you'll actually want to see

---

Hey guys,

Wanted to send this before we push it public. At CMC we've been running a research program called AI Moonshots and two of the projects are directly in your wheelhouse:

- **CTI (Compute Thermodynamics of Intelligence)** — universal law for LLM performance derived from extreme-value theory *before* any fitting. Same constant holds across 12 NLP architectures (R²=0.955) AND mouse V1 Neuropixels recordings (CV=1.65% across all families + biology). https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — prepending 2 random embedding-scale tokens (no training, no optimization) unlocks reasoning in small LLMs. Qwen3-4B arithmetic: 32% → 51.6%. 10 random directions hit 100% oracle coverage. A new improvement axis orthogonal to scaling/FT/prompting/RAG. https://github.com/dl1683/Latent-Space-Reasoning

Both of those were the warm-up. The real project now is **Neural Genome** — we're building a cross-architecture atlas of representational geometry. The question: is there a shared coordinate system that describes what's happening inside a transformer, a Mamba, a ViT, and a BERT? If yes, that coordinate system *is* the next-generation interpretability tool — architecture-agnostic, mathematically grounded, causally testable.

Some of what we've found in the last two weeks (single RTX 5090 laptop, all pre-registered, Bonferroni-corrected):

1. **One geometric coordinate passes portability testing on 8 architecture classes and 5 distinct training objectives** — autoregressive, reasoning-distilled, linear-attention recurrent, hybrid Mamba, MLM, contrastive-text, self-sup-ViT, contrastive-vision. Same number (within a strict tolerance) on all of them.
2. **Survives 4× weight quantization** (FP16 → Q8) at even tighter tolerance. The geometry doesn't live in the weights.
3. **Causally load-bearing** — when we ablate the subspace it identifies, model loss explodes +7.8% to +443% depending on system and layer. Random ablations of the same dimensionality move loss <1%. Specificity ratios 20–66×. This isn't a descriptive correlation; it's a mechanism.
4. **Not a random-geometry artifact** — trained values are 4–7× above random-Gaussian baseline at the same sample size and dimension.
5. **Theoretically grounded** — we locked a first-principles derivation from Laplace-Beltrami convergence before seeing any empirical fit. The prediction matched.

**Why this matters for Martian specifically.** Your Model-Mapping framework is trying to answer the commercial version of exactly this question — "are circuits in Model A homologous to circuits in Model B so we can transfer routing decisions and alignment interventions?" Right now that claim is empirical; we think the atlas makes it *provable*. It becomes a shared coordinate system you can cite when you say "this circuit in Model A is the same structure as this circuit in Model B." That turns your router's interpretability story from "trust us" to "here's the mathematical invariant."

To be honest — this is a moonshot. We don't know if the atlas extends to everything we want it to (biology, RL agents, diffusion, world models) and it might ultimately tell us the universality is narrower than we hope. That's a finding worth publishing too. But the precursors (CTI + Latent Space Reasoning) have already delivered results that nobody saw coming, and if this one lands the way the first two have, it's the DeepSeek moment for interpretability — and for us. A paper that everyone has to cite, a coordinate system everyone uses, a positioning no competitor can match because the math is openly provable.

We want CMC + Martian + Furiosa (inference hardware) + Weka (context memory) as the founding research syndicate. Before we push this public in a few weeks, we'd rather have you in on the long-term roadmap than reading it afterward.

30-min call this week?

Dev
