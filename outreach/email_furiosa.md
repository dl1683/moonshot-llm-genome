# Email to Furiosa AI

**To:** June Paik (and/or Jeehoon Kang)
**Subject:** something for you before I open-source it

---

Hey June,

Wanted to catch you on this before I push it public. I've been running a research program called AI Moonshots and the two projects most relevant to Furiosa are:

- **CTI (Compute Thermodynamics of Intelligence)** — a universal compute-performance law derived from first principles *before* any curve fitting. Passes LOAO across 12 architectures (R²=0.955) and the same constant shows up in mouse V1 recordings. https://github.com/dl1683/moonshot-cti-universal-law
- **Latent Space Reasoning** — adding 2 random token-scale vectors at inference (no training, pure activation-level intervention) jumps Qwen3-4B arithmetic from 32% → 51.6%. https://github.com/dl1683/Latent-Space-Reasoning

Both were warm-ups. The main project now is **Neural Genome** — building a cross-architecture atlas of representational geometry. The early result from this week: a specific geometric coordinate passes portability at strict threshold across 8 architecture classes, and — the part you'll care about — **survives 4× weight quantization (FP16 → Q8) on transformers, recurrent (RWKV), and hybrid (Falcon-H1) with the same tight tolerance**. The geometric structure is invariant under aggressive compression, which means the same coordinate tells you which layers of any architecture can tolerate INT4/INT8 and which can't — a portable quantization prior.

The whole thing runs on one RTX 5090 laptop. The mission is basically "prove intelligence = geometry, not scale" which means inference, activation analysis, tiny training — exactly the workload profile RNGD is built for. I'm going to be running a lot of model-inference passes to extend the atlas to more classes (diffusion, video, audio, world models) and the Tensor Contraction Processor architecture + your compiler's quantization layer are genuinely the best-fit hardware for this research — better than H100s which are over-provisioned for training.

I'm open-sourcing this in a few weeks but wanted to get Furiosa in first. Thinking: research syndicate with you + Martian (interpretability side) + Weka (context memory) + me. You get (a) a research partner running real workloads on RNGD, (b) a quantization-prior coming out of the atlas that your compiler can exploit, (c) co-authorship on a paper that says "the theoretical case for Intelligence=Geometry is supported and here's what it means for inference hardware economics."

Worth a 30-min call?

Devansh
