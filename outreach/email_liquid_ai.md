# Email to Liquid AI

**To:** Ramin Hasani, Mathias Lechner (cc: Daniela Rus if still advising, or whoever owns external research outreach)
**Subject:** heads-up on cross-architecture work that overlaps hard with LFMs

---

Hey Ramin,

Sending this before we push it public because the overlap between what Liquid AI is building and what we're finding is unusually close.

**Core thesis: geometric manipulation of representations produces capabilities that scale-alone cannot.** Three of our recent AI Moonshots prove this from independent angles:

- **Latent Space Reasoning** — 2 random embedding-scale tokens prepended at inference (no training, no optimization) jump Qwen3-4B arithmetic from 32% → 51.6%. An activation-level intervention orthogonal to scaling / FT / prompting. https://github.com/dl1683/Latent-Space-Reasoning
- **CTI (Compute Thermodynamics of Intelligence)** — first-principles universal law predicting model performance from compute, derived *before* fitting. Same constant across 12 NLP architectures AND mouse V1 Neuropixels (CV=1.65%). https://github.com/dl1683/moonshot-cti-universal-law
- **Fractal Embeddings** — hierarchy-aligned progressive prefix supervision produces +5.36% L0 / +6.47% L1 accuracy (5-seed validation, random-hierarchy causal control drops to −0.10%). Geometric prior, emergent hierarchical capability, no added parameters.

Each one says the same thing from a different angle: **geometry beats scale**, reproducibly, with causal controls. That's exactly the thesis Liquid has been making commercially with continuous-time networks and LFMs — smaller, more structured models outperform larger unstructured ones on the right workloads.

The current project — **Neural Genome** — takes the thesis all the way: is there a single universal coordinate system for this geometric substrate across every trained network?

Headline results from the last two weeks (single RTX 5090 laptop, pre-registered, Bonferroni-corrected):

1. **Candidate-8 spectral bridge** — `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on **7 of 8 trained systems** spanning CLM decoders (Qwen3, DeepSeek), MLM encoders (BERT, RoBERTa), contrastive (MiniLM), vision ViT (DINOv2), cross-modal (CLIP both branches). Median relative error 8.7%. Prereg: `genome_svd_bridge_2026-04-22.md` locked before any measurement.
2. **Universal bulk width** `k_bulk ≈ h/22` (CV 4.2% across 5 text systems) — every trained 1024-d text model uses ~48 principal informational dimensions. Everything beyond that is the tail.
3. **Trained-spectrum fingerprint**: α ≈ 0.86 power-law decay on trained activation covariance vs α ≈ 0.65 on shuffled / iid-Gaussian. 30% steeper decay *is* the signature.
4. **GenomeGuard** — a 20-second training health monitor. Cross-architecture (5 text systems) silent data corruption detection with **6.9× – 144.9× rel_err spike**. Catastrophic weight-divergence at 8× signal. Ship-ready in under 300 lines.
5. **12-operation null catalog** — every forward geometric manipulation we tested for *installing* capability is null: covariance transfer, codebook, PCA basis, aux-regularizer, single-layer weight transplant, QK/V/O/attn-all/MLP subset transplants, Procrustes-aligned transplant, candidate-8-ratio aux-loss. Capability is irreducibly joint weight configuration. A publishable negative claim that frontier labs structurally won't write up.
6. **Cross-substrate (today)**: bridge holds on **mouse V1 Neuropixels** under Natural Movie One at rel_err 12.3% (Allen Brain Observatory, session 0, 900 frames × 50 cortical units). Same identity, biological substrate. Not a backprop artifact.

**Why Liquid AI specifically.** Two reasons:

First — direct scientific overlap. LFM's pitch is "fewer, better-structured parameters outperform scale," and continuous-time / Liquid-S architectures are the mechanism. Our bridge gives that pitch a **mathematical receipt**: if the `eff_rank / d_rd` identity predicts how much informational capacity a given layer actually uses, it's the right target for principled compression and architecture search. An LFM that hits the same `c` at 1/10 the parameters is *provably* matching the information geometry of the bigger model, not just the loss. That's the kind of commercial claim nobody else in the efficient-architecture space can make.

Second — your continuous-time networks are the **hardest test** of the universality claim. Everything in the scorecard so far is discrete-time transformer / RNN / ViT. If the bridge also holds on Liquid-S networks at the same 15% tolerance, that's the strongest possible extension: the coordinate doesn't depend on discrete attention or tokenwise state. If it fails on Liquid-S, we've found *exactly* where the artificial-vs-biological-vs-continuous-time universality splits — a publishable contribution for both of us, since it puts numbers on the structural difference between continuous-time and discrete-token inference.

We'd love to add an LFM to the bestiary. Measurement is ~30 minutes of GPU time per model, and the script is already open-source in the repo. If you share a checkpoint (or point us at a public one), we'll run it this week and share the result before any announcement.

This is a difficult project but we think we can pull it off. The bridge is already holding on 7/8 discrete-time systems, GenomeGuard is already shipping cross-arch, and the 12-op null is already in hand. The continuous-time extension is a real open question — but the underlying geometric structure is no longer speculative, and every step so far has landed. If it lands, this is the **DeepSeek moment for efficient intelligence** — a mathematical case that geometric structure, not parameter count, is what produces capability. Liquid AI is one of the few companies that has been making that case commercially since day one; it would be a pity if the paper cites everyone except the team whose thesis it most directly validates.

CMC is putting together a research syndicate — us + Martian (mechanistic interp) + Furiosa (inference hardware) + Weka (context memory) + VERSES (active inference) + you. Efficient architectures + representational geometry + active inference + inference hardware + data substrate = the full stack for proving the non-scale thesis. Before the open-source drop, we'd rather have Liquid in on the long-term roadmap than pitching you after.

Would love to talk further if that's useful.

Dev
CMC / AI Moonshots — github.com/dl1683/ai-moonshots
