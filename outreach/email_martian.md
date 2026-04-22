# Email to Martian

**To:** Shriyash, Etan
**Subject:** something you'll actually want to see

---

Hey guys,

Wanted to send this before we push it public.

**Core thesis: geometric manipulation of representations produces capabilities that scale-alone cannot.** Three of our recent AI Moonshots each prove it from a different angle:

- **Latent Space Reasoning** — prepending 2 random embedding-scale tokens (no training, no optimization) unlocks reasoning in small LLMs. Qwen3-4B arithmetic: 32% → 51.6%. 10 random directions hit 100% oracle coverage. A new improvement axis orthogonal to scaling / FT / prompting / RAG. https://github.com/dl1683/Latent-Space-Reasoning
- **CTI (Compute Thermodynamics of Intelligence)** — universal law for LLM performance derived from extreme-value theory *before* any fitting. Same constant across 12 NLP architectures (R²=0.955) AND mouse V1 Neuropixels recordings (CV=1.65% across all families + biology). https://github.com/dl1683/moonshot-cti-universal-law
- **Fractal Embeddings** — hierarchy-aligned progressive prefix supervision delivers +5.36% L0 / +6.47% L1 accuracy (5-seed validation). Random-hierarchy causal control drops to −0.10%. Geometric prior alone produces emergent hierarchical capability.

Each says the same thing from a different angle: **geometry beats scale**, reproducibly, with hard causal controls. Those three were the warm-up. The real project now is **Neural Genome** — cross-architecture atlas of representational geometry. The question: is there a shared coordinate system that describes what's happening inside a transformer, a Mamba, a ViT, and a BERT? If yes, that coordinate system *is* the next-generation interpretability tool — architecture-agnostic, mathematically grounded, causally testable.

What we've found in the last two weeks (single RTX 5090 laptop, all pre-registered, Bonferroni-corrected):

1. **Candidate-8 spectral bridge** — `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on **7 of 8 trained systems**: Qwen3 (9% rel_err), DeepSeek-R1-Distill (0.2%), BERT (14%), RoBERTa (4%), MiniLM (8%), CLIP-text (7%), CLIP-vision (12%), DINOv2 (20%, 1 miss). Median 8.7%. Locked pre-registration: `research/prereg/genome_svd_bridge_2026-04-22.md`.
2. **Universal bulk width** `k_bulk ≈ h/22` (CV 4.2% across 5 text systems) — every trained 1024-d text model has ~48 principal directions doing the informational work.
3. **Trained-spectrum fingerprint**: α ≈ 0.86 power-law decay on trained activations vs α ≈ 0.65 on shuffled / iid-Gaussian baseline. 30% steeper decay *is* the training-specific signature.
4. **Causally load-bearing** — PCA-compression sweeps on trained activations trace a clean monotone curve from `d_rd` collapse to NLL explosion. Specificity 20–66× vs random-direction ablations. Real mechanism.
5. **GenomeGuard** — 20-second training health monitor built on the bridge. On 5 text architectures, silent data corruption triggers **6.9× – 144.9× rel_err spike** (DeepSeek tightest baseline = most sensitive detector). Catastrophic weight-divergence at 8× signal. Ship-ready in <300 lines.
6. **12-operation null catalog** — every forward geometric manipulation we tested for installing capability is null: covariance transfer, codebook, PCA basis, aux-regularizer, single-layer weight transplant, QK-only / V-only / O-only / attention-all / MLP-only subset transplants, Procrustes-aligned transplant, candidate-8-ratio aux-loss. **Capability is irreducibly joint weight configuration** — a publishable negative claim that frontier labs are structurally disincentivized to write up.

**Why this matters for Martian specifically.** Your Model-Mapping framework is trying to answer the commercial version of exactly this question — "are circuits in Model A homologous to circuits in Model B so we can transfer routing decisions and alignment interventions?" Right now that claim is empirical; we think the bridge makes it *provable*. `eff_rank/d_rd` is a shared coordinate you can cite when you say "this subspace in Model A is the same structure as this subspace in Model B." That turns your router's interpretability story from "trust us" to "here's the mathematical invariant, compute it yourself in 20 seconds on your own model."

The **12-op null catalog** is the other direct hit: it says capability doesn't live in any separable weight-subset or geometric target, which is exactly the bar Model-Mapping has to clear if it's going to make routing decisions based on structural similarity. Our null result tells you *which* invariants to look for (joint-configuration fingerprints, not per-component matches). That sharpens your mechanistic-interp targets enormously.

To be honest — this is a moonshot. We don't know if the bridge extends to everything we want it to (biology probe is in flight today, RL agents / diffusion / world models are queued). But the precursors have already delivered results nobody saw coming, and the Neural Genome already shipped a working tool (GenomeGuard) plus a hard negative claim (12-op null). If the biology extension lands, this is the **DeepSeek moment for interpretability** — and for us. A paper that everyone has to cite, a coordinate system everyone uses, a positioning no competitor can match because the math is openly provable.

We want CMC + Martian + Furiosa (inference hardware) + Weka (context memory) + Liquid AI (efficient architectures) + VERSES (active inference) as the founding research syndicate. Before we push this public in a few weeks, we'd rather have you in on the long-term roadmap than reading it afterward.

30-min call this week?

Dev
