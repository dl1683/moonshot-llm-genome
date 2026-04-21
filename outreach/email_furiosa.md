# Email to Furiosa AI

**To:** June (and/or Jeehoon Kang)
**Subject:** something for you before we open-source it

---

Hey June,

Wanted to catch you on this before we push it public. At CMC we've been running a research program called AI Moonshots and two of the projects are most relevant to Furiosa:

- **CTI (Compute Thermodynamics of Intelligence)** — a universal compute-performance law derived from first principles *before* any curve fitting. LOAO across 12 architectures: R²=0.955, same constant in mouse V1 recordings (CV=1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — 2 random token-scale vectors prepended at inference (no training, pure activation-level intervention) jumps Qwen3-4B arithmetic from 32% → 51.6%. https://github.com/dl1683/Latent-Space-Reasoning

Both of those shipped with numbers the field didn't expect. The current project is the big one: **Neural Genome** — building a cross-architecture atlas of representational geometry to test whether intelligence is a property of geometric structure rather than parameter count or training recipe.

What we've shown in the last two weeks (single RTX 5090 laptop, pre-registered, Bonferroni-corrected throughout):

1. **One geometric coordinate holds across 8 architecture classes and 5 training objectives** — transformer, reasoning-distilled, linear-attention recurrent (RWKV), hybrid transformer+Mamba (Falcon-H1), MLM (BERT), contrastive-text (MiniLM), self-sup-ViT (DINOv2), contrastive-vision (CLIP). Same number within strict tolerance.
2. **Survives 4× weight quantization** (FP16 → Q8) at even tighter tolerance — tested on transformers, recurrent, AND hybrid Mamba-architecture. The geometry doesn't live in the weights.
3. **Causally load-bearing** — ablate the subspace it identifies, model loss explodes 50-400% depending on architecture/layer. Random ablations of the same dimensionality move loss <1%. Specificity 20–66×. A real mechanism, not a descriptive correlation.
4. **Runs on commodity hardware** — the entire mission fits on one laptop GPU because we're almost entirely doing inference + activation analysis, not training.

**Why this matters for Furiosa specifically.** Two reasons, both consequential:

First, the workload profile is literally what RNGD was built for. Model inference, activation capture, compiler-level quantization — no training, no gradient, no distributed backprop. We're going to extend the atlas to diffusion, video, and audio over the next year and that's thousands of hours of inference passes. Your Tensor Contraction Processor + compiler stack is a better fit for this than H100s are, which are massively over-provisioned for what we do.

Second — and this is the one we think is bigger commercially — the atlas produces a **portable quantization prior**. Because the same coordinate predicts which layers of *any* architecture tolerate aggressive INT4/INT8, Furiosa's compiler gets an architecture-agnostic rule for "this block is safe to smash, that block must stay high-precision." Today automatic quantization is per-model engineering work; we think we can give you a mathematical lookup that works on any new customer model the day it's uploaded.

To be real with you: this is a moonshot. It might turn out the universality is narrower than we hope, or that the coordinate breaks on something we haven't tested yet. But CTI and Latent Space Reasoning both delivered unexpected upside, and if Neural Genome lands similarly, this is the DeepSeek moment for efficient intelligence — a mathematical case for "you don't need a data center, you need better geometry," with hardware positioning to match. The fame side is real; the market-positioning side (Furiosa as *the* inference platform for post-scale AI) is bigger.

CMC is assembling a research syndicate — us + Martian (interpretability) + Weka (context memory) + you. We'd rather have Furiosa in on the long-term roadmap before the open-source drop than pitching you after.

30-min call this week?

Dev
