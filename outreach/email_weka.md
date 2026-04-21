# Email to Weka

**To:** Liran (and/or Nilesh)
**Subject:** want to loop you in on this before we open-source it

---

Hey Liran,

Quick one before we push this public. At CMC we've been running a research program called AI Moonshots, and two of the projects set up the one I'm writing about:

- **CTI (Compute Thermodynamics of Intelligence)** — a first-principles universal law predicting model performance from compute. Same constant across 12 NLP architectures AND mouse V1 Neuropixels (CV=1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — activation-level intervention (2 random tokens, no training) jumping Qwen3-4B arithmetic accuracy 20 points. https://github.com/dl1683/Latent-Space-Reasoning

Both shipped with results nobody expected. The project I'm writing you about is **Neural Genome** — we're building a cross-architecture atlas of representational geometry, testing whether the same mathematical coordinates describe the internal organization of transformers, Mambas, ViTs, and BERTs.

The last two weeks (one RTX 5090 laptop, pre-registered, Bonferroni-corrected):

1. **One geometric coordinate is statistically indistinguishable across 8 architecture classes and 5 training objectives** (CLM + MLM + contrastive + self-sup + hybrid-Mamba). Strict tolerance, 3 stimulus-resample seeds.
2. **Survives 4× quantization** (FP16 → Q8) at even tighter tolerance.
3. **Causally load-bearing** — ablate the subspace, loss explodes 50-400%; random-10-dim ablations move it <1%. This is the mechanism, not a correlation.
4. **4–7× above random-Gaussian baseline** — not a high-dim artifact. Real signal.

**Why I think this matters for Weka.** Your whole NeuralMesh / Augmented Memory Grid thesis is that reasoning workloads need more context than HBM can hold, so you stream KV-cache intelligently between GPU memory and the petabyte token warehouse. The hard question underneath that is *which activations are compressible and which are essential* — today that's a per-model guess. We think the atlas answers it from first principles. The same coordinate that tells Furiosa which layers tolerate INT4 also tells NeuralMesh which KV slices are safe to evict cheaply vs must stay hot.

This is where I want to be honest: the KV-caching application is an instinct, not a proven result yet. I want to explore it with your team, not pitch it as a done deal. But if the instinct is right, NeuralMesh gets a geometry-aware caching policy that's provably optimal rather than heuristic — and you're the only data-platform company in position to own that story.

Honestly — this is a moonshot. It might not work, or it might work only narrowly. That's the nature of the ambition and we name it that way on purpose. But the two precursors delivered real, reproducible results that surprised the field, and if Neural Genome hits the way we think it might, this is the DeepSeek moment for efficient intelligence — a mathematical argument that says "intelligence doesn't require a data center, it requires better geometry," and a commercial story that re-positions everything downstream. For Weka that means becoming *the* data platform for reasoning-scale AI, not just a storage vendor to NVIDIA's stack.

CMC is putting together a research syndicate — us + Martian (interpretability) + Furiosa (inference hardware) + you. Before we push the atlas open-source in a few weeks, I'd rather have Weka in on the roadmap than read-afterward.

30-min call this week?

Dev
