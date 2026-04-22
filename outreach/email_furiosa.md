# Email to Furiosa AI

**To:** June (and/or Jeehoon Kang)
**Subject:** something for you before we open-source it

---

Hey June,

Wanted to catch you on this before we push it public.

**Core thesis: geometric manipulation of representations produces capabilities that scale-alone cannot.** Three of our AI Moonshots projects each prove this from a different angle:

- **Latent Space Reasoning** — 2 random token-scale vectors prepended at inference (no training, pure activation-level intervention) jumps Qwen3-4B arithmetic from 32% → 51.6%. https://github.com/dl1683/Latent-Space-Reasoning
- **CTI (Compute Thermodynamics of Intelligence)** — universal compute-performance law derived from first principles *before* any curve fitting. LOAO across 12 architectures: R²=0.955, same constant in mouse V1 Neuropixels (CV=1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- **Fractal Embeddings** — hierarchy-aligned progressive prefix supervision produces +5.36% L0 / +6.47% L1 accuracy (5-seed validation, random-hierarchy causal control drops to −0.10%). Geometric prior alone, no additional parameters.

Each one says the same thing: **geometry beats scale**, reproducibly, with hard causal controls. They shipped with numbers the field didn't expect. The current project is the big one: **Neural Genome** — building a cross-architecture atlas of representational geometry to test whether intelligence is a property of geometric structure rather than parameter count or training recipe.

The headline result from the last two weeks (single RTX 5090 laptop, pre-registered throughout):

1. **Candidate-8 spectral bridge** — `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on **7 of 8 trained systems**: CLM decoders (Qwen3, DeepSeek), MLM encoders (BERT, RoBERTa), contrastive (MiniLM), vision ViT (DINOv2), cross-modal (CLIP both branches). Median rel_err 8.7%. All pre-registered, Bonferroni-corrected.
2. **Universal bulk width**: plateau-plus-power-law spectrum fit gives `k_bulk = 48 ± 2` across 5 text systems at h=1024, CV 4.2%. A real structural universal — every trained 1024-d text model has approximately h/22 effective informational dimensions with roughly equal magnitude.
3. **Trained-spectrum signature**: α ≈ 0.86 power-law decay on trained covariance vs α ≈ 0.65 on shuffled or iid-Gaussian baseline. 30% steeper decay is the training-specific fingerprint.
4. **GenomeGuard** — a 20-second training health monitor built on the bridge. On 5 text architectures it flags silent data corruption with **6.9× – 144.9× rel_err spike** (DeepSeek most sensitive at 144.9×). Catches catastrophic weight-divergence at 8× signal. Cross-architecture. Ship-ready in under 300 lines.
5. **12-operation null catalog**: every forward geometric manipulation we tested (covariance transfer, codebook, PCA basis, aux-regularizer, single-layer weight transplant, QK/V/O/attn-all/MLP subset transplants, Procrustes-aligned transplant, candidate-8-ratio aux-loss) is null for installing capability. Capability is irreducibly joint weight config.
6. **Cross-substrate (today)**: bridge holds on mouse V1 Neuropixels under Natural Movie One at rel_err 12.3% — the same identity we measured on Qwen3, DeepSeek, BERT, CLIP also applies to biological cortex. Not a backprop artifact.

**Why this matters for Furiosa specifically.** Two reasons, both consequential:

First, the workload profile is literally what RNGD was built for. Model inference, activation capture, spectral analysis, compiler-level quantization — no training, no gradient, no distributed backprop. GenomeGuard alone is an inference-only workload that every training team will want to run as a canary. We're going to extend the atlas to diffusion, video, and audio over the next year and that's thousands of hours of inference passes. Your Tensor Contraction Processor + compiler stack is a better fit for this than H100s are, which are massively over-provisioned for what we do.

Second — and this is the one we think is bigger commercially — the bridge is a **portable quantization prior**. Because the same geometric coordinate predicts which layers of *any* architecture tolerate aggressive INT4/INT8 (and we've already shown the bridge survives 4× weight quantization from FP16→Q8 at even tighter tolerance), Furiosa's compiler gets an architecture-agnostic rule for "this block is safe to smash, that block must stay high-precision." Today automatic quantization is per-model engineering work; we think we can give you a mathematical lookup that works on any new customer model the day it's uploaded.

This is a difficult project but we think we can pull it off. The bridge is already holding on 7/8 systems, GenomeGuard is already shipping cross-arch, the 12-op null is already in hand. The remaining extensions (biology, diffusion, video, audio) are genuine open questions, but every step so far has landed. If it all lands, this is the **DeepSeek moment for efficient intelligence** — a mathematical case for "you don't need a data center, you need better geometry," with hardware positioning to match. The fame side is real; the market-positioning side (Furiosa as *the* inference platform for post-scale AI) is bigger.

CMC is assembling a research syndicate — us + Martian (interpretability) + Weka (context memory) + Liquid AI (efficient architectures) + VERSES (active inference) + you. We'd rather have Furiosa in on the long-term roadmap before the open-source drop than pitching you after.

Would love to talk further if that's useful.

Dev
