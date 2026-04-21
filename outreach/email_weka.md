# Email to Weka

**To:** Liran (and/or Nilesh)
**Subject:** want to loop you in on this before I open-source it

---

Hey Liran,

Quick one before I push this public. I've been running a research program called AI Moonshots and two of the projects are in territory Weka cares about:

- **CTI (Compute Thermodynamics of Intelligence)** — a first-principles universal law predicting model performance from compute, same constant across 12 NLP architectures AND mouse V1 neural recordings. https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — activation-level interventions (2 random tokens, no training) jumping small-LLM reasoning accuracy 20 points. https://github.com/dl1683/Latent-Space-Reasoning

Both of those were groundwork. The main project now is **Neural Genome** — an atlas of representational geometry across architectures, trying to show that the same mathematical coordinates describe the internal organization of very different models. This week's result: a specific geometric coordinate passes portability across 8 architecture classes and 5 training objectives, survives 4× quantization, and is causally load-bearing (ablating it blows up model loss 50-400% vs <1% for random ablations).

The Weka angle I'm still figuring out — but here's the instinct. The whole NeuralMesh / Augmented Memory Grid thesis is "reasoning needs more context than HBM holds, so stream KV-cache intelligently." The atlas gives you a principled answer to *which activations/layers are compressible vs essential* — that's a geometry-aware caching policy. Same coordinate that tells Furiosa which layers tolerate INT4 tells you which KV slices you can evict cheaply vs which you must keep hot. I'd love to dig into this with your team to see if there's a real fit.

I'm open-sourcing in a few weeks but wanted to loop you in first. Thinking: research syndicate with you + Martian (interp) + Furiosa (inference hardware) + me. Move faster together on the long-term roadmap than solo.

30-min call this week?

Devansh
