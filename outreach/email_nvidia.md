# Email to NVIDIA Research

**To:** Bryan Catanzaro, Bill Dally (alt: Anima Anandkumar if still on the research side, or whoever owns Nemotron research outreach)
**Subject:** small heads-up on open-science work you might want eyes on

---

Hey Bryan,

Short one. We've been watching NVIDIA Research's open-science push — Nemotron, Cosmos, the open-reasoning datasets — and we think what we're doing at CMC would fit nicely into that orbit as something you'd want to be aware of before it's public.

**The thesis: geometric manipulation of representations produces capabilities that scale-alone cannot.** Three of our recent projects converge on this — each a separate proof-of-existence that geometric structure, not parameter count, is the substrate capability actually lives on:

- **Latent Space Reasoning** — 2 random embedding-scale tokens prepended at inference, no training, jumps Qwen3-4B arithmetic from 32% → 51.6%. An activation-level intervention that unlocks reasoning. https://github.com/dl1683/Latent-Space-Reasoning
- **CTI (Compute Thermodynamics of Intelligence)** — a first-principles universal law for LLM performance derived from extreme-value theory *before* any fitting. Same constant across 12 NLP architectures (R²=0.955) and mouse V1 Neuropixels recordings (CV=1.65% across all families + biology). https://github.com/dl1683/moonshot-cti-universal-law
- **Fractal Embeddings** — hierarchy-aligned progressive prefix supervision delivers +5.36% L0 / +6.47% L1 accuracy (5-seed validation, 20 Newsgroups causal control drops to −0.10% with random hierarchy). Same encoder, different geometric prior, emergent hierarchical capability.

Each one says the same thing from a different angle: **geometry beats scale**, reproducibly, with causal controls.

The current project — **Neural Genome** — takes that thesis all the way: is there a single universal coordinate system for this geometric substrate across every trained network? We found one. `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on **7 of 8 trained systems** spanning CLM decoders (Qwen3, DeepSeek), MLM encoders (BERT, RoBERTa), contrastive (MiniLM), vision ViT (DINOv2), and cross-modal alignment (CLIP both branches). Median relative error 8.7%. All pre-registered, locked before the runs fired.

Two concrete deliverables already shipping:

1. **GenomeGuard** — a 20-second training health monitor that uses the bridge as a training-free canary. On 5 text architectures it flags silent data corruption with **6.9× – 144.9× signal** (DeepSeek is most sensitive because its baseline bridge is tightest — 0.002 rel_err). Catches catastrophic weight-divergence at 8× signal. Ship-ready in under 300 lines of Python.

2. **12-operation null catalog** — we exhaustively tested whether you can *install* capability via forward geometric manipulation (covariance transfer, codebook, PCA basis, aux-regularizer, single-layer weight transplant, QK/V/O/attn-all/MLP subset transplants, Procrustes-aligned transplant, candidate-8-ratio aux-loss). **All 12 null.** Capability is irreducibly the joint weight configuration — a publishable negative claim that frontier labs are structurally disincentivized to write up because it undercuts the "more scale is the answer" narrative.

Plus: universal bulk width `k_bulk ≈ h/22` (CV 4.2% across 5 text systems); stimulus-sensitivity characterized (candidate-5 alignment-axis additivity: text ≈ 2, vision ≈ 3, +1 per cross-modal alignment, 11/12 at C4).

**Why I'm writing.** This is straightforwardly the kind of thing the Nemotron / open-dataset program cares about. We're going to open-source the atlas, the preregs, the extensible bestiary, and GenomeGuard — CMC runs a fairly large open-source community and this drops into that pipeline. If the Neural Genome extends all the way to biology + RL + world models, it's a public-good tool for every model release — you test your next Nemotron against the atlas and get a free mechanistic sanity check + a training-time corruption detector.

This is a difficult project but we think we can pull it off. The precursors delivered results the field didn't expect, the bridge is already holding on 7/8 trained systems, and we have a shipping tool plus a hard negative claim already in hand. If the biology + RL + world-model extensions land the way text and vision did, it's the first cross-substrate empirical evidence for what a "universal science of intelligence" would even look like — and NVIDIA Research is one of a handful of groups in the world that would take that seriously as science, not just as a paper to cite.

Not asking for anything specific. Planning to stick to the open-source community for distribution. Just wanted NVIDIA Research to have a heads-up and, if it's interesting, an open door for feedback or collaboration on the public release. Would love to talk further if that's useful.

Dev
CMC / AI Moonshots — github.com/dl1683/ai-moonshots
