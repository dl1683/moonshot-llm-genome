# Email to NVIDIA Research

**To:** Bryan Catanzaro, Bill Dally (alt: Anima Anandkumar if still on the research side, or whoever owns Nemotron research outreach)
**Subject:** small heads-up on open-science work you might want eyes on

---

Hey Bryan,

Short one. We've been watching NVIDIA Research's open-science push — Nemotron, Cosmos, the open-reasoning datasets — and we think what we're doing at CMC would fit nicely into that orbit as something you'd want to be aware of before it's public.

Two open-source projects that set up what we're working on now:

- **CTI (Compute Thermodynamics of Intelligence)** — a first-principles universal law for LLM performance derived from extreme-value theory *before* any fitting. Same constant across 12 NLP architectures (R²=0.955) and mouse V1 Neuropixels recordings (CV=1.65% across all families + biology). https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — 2 random embedding-scale tokens prepended at inference (no training) takes Qwen3-4B arithmetic from 32% → 51.6%. A new activation-level improvement axis. https://github.com/dl1683/Latent-Space-Reasoning

The current project is **Neural Genome** — building a cross-architecture atlas of representational geometry. We're testing whether a single mathematical coordinate describes the internal organization of transformers, Mambas, ViTs, BERTs, CLIP, RWKVs, and — eventually — biological cortex under matched stimuli.

Early results from the last two weeks (one RTX 5090 laptop, all pre-registered, Bonferroni-corrected):

1. One geometric coordinate passes portability testing across 8 architecture classes and 5 distinct training objectives (autoregressive, reasoning-distilled, linear-attention recurrent, hybrid Mamba, MLM, contrastive text, self-sup ViT, contrastive vision). Same number within strict tolerance.
2. Survives 4× weight quantization (FP16 → Q8) at even tighter tolerance.
3. Causally load-bearing — ablating the subspace blows up model loss 8-443%; random ablations of the same dimensionality move it <1%. Specificity 20-66×.
4. 4-7× above random-Gaussian baseline — not a high-dimensional artifact.
5. Theoretically grounded — we locked a Laplace-Beltrami-convergence derivation before fitting, prediction held empirically.

**Why I'm writing.** This is straightforwardly the kind of thing the Nemotron / open-dataset program cares about. We're going to open-source the atlas, the code, the prereg discipline, and the extensible bestiary — CMC runs a fairly large open-source community and this drops into that pipeline. If Neural Genome works at Level-1 universality, it's a public-good tool for every model release — you test your next Nemotron against the atlas and get a free mechanistic sanity check.

Honestly — this is a moonshot. The precursors (CTI + Latent Space Reasoning) delivered results the field didn't expect, but we don't yet know if Neural Genome extends all the way to biology + RL agents + world models. If it does, it's the first cross-substrate empirical evidence for what a "universal science of intelligence" would even look like — and NVIDIA Research is one of a handful of groups in the world that would take that seriously as science, not just as a paper to cite.

Not asking for anything specific. Planning to stick to the open-source community for distribution. Just wanted NVIDIA Research to have a heads-up and, if it's interesting, an open door for feedback or collaboration on the public release.

Dev
CMC / AI Moonshots — github.com/dl1683/ai-moonshots
