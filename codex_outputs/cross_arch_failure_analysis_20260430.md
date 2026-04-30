**Bottom Line**

Cross-architecture fails at the **interface-plus-dynamics layer**, not just one superficial mismatch.

- **Not hidden dim:** g173 used hidden=768, layers=8, heads=12, ffn=2048 for both Llama/Qwen-style students. Still failed.
- **Not only tokenizer:** g173 shared the Qwen tokenizer/vocab and still got transfer_ratio=0.99x, below the 1.5x fail floor.
- **Tokenizer/interface is still a major break:** g180b changed tokenizer inside a Qwen-style shell and geometry got worse by -39.4% MSE.
- **Internal transformer blocks are not the transferable payload:** g181a no-embed/lm-head anchoring harmed by -0.439 nats, while embed+lm_head gave +0.483.
- **Cross-arch geometry is non-comparable in the current coordinates:** g182 P3 had 0/8 feature overlap cross-arch; seven features had KS=1.0, p=7.4e-7. LOAO R2 was catastrophic.

**Failure Modes**

| Exp | Specific failure |
|---|---|
| g173 | Shared tokenizer + matched width did not cash out. Llama KD C3 39.87% to 42.16%, Qwen-arch 40.91% to 41.71%, but transfer_ratio=0.99x. Retention=101.1%, late/full=97.7%, yet no FLOP advantage. |
| g177v2 | Donor identity rejected. Alt donors at NLL ~5.70-5.74 recovered 95-96% of Qwen3 effect. Best alt +1.048 nats vs Qwen +1.087; Qwen-minus-best-alt only +0.038 nats CI [+0.018,+0.068]. |
| g180b | Tokenizer portability failed. Overall MSE reduction=-39.4%, CI [-0.183,-0.0089]. BERT -42.9%, T5 -96.4%, GPT-2 +44.0%. KD was harmful under all 3 tokenizers: full KD labels BERT -0.478, T5 -0.541, GPT-2 -0.369 nats. |
| g181a | Mechanism isolated to interface. full_anchor +0.999 nats; embed_lm_head_only +0.483; no_embed_lm_head -0.439. no_embed-minus-embed = -0.923 CI [-1.055,-0.835]. Transformer-block anchor actively harms. |
| g181b | Within-family interface prior persists. embed/lm_head anchor gap +0.513 nats at 5000 steps, CI [+0.486,+0.531], stable from +0.387 at step 500 to +0.513 at step 5000. |
| g182 | LOAO cross-arch predictor failed. Model A/B/C/C'/D/E all failed with R2 roughly -11 to -19. Manifold-only C': GPT-2 R2=-11.94, Qwen3 R2=-16.57. Arm-demeaned LOAO near zero or negative: -0.003, -0.100. Only survivor was within-arch pairwise delta: R2=0.518, corr=0.720, n=24. |
| g186 | g182 survivor did not generalize. 60 cells, 48 deltas. Pooled R2=0.022, arm_mean R2=0.936, alpha_quad R2=0.774. MSE reduction vs best baseline=-1416%. Permutation p=0.705, conditioned p=1.000. Per-arch R2: Qwen3=-0.098, GPT-2=-8.728. Held-out-arch stress: Qwen3=-6.18, GPT-2=-1455. |

**What Exactly Breaks**

The raw geometric features are not invariant coordinates. They are **architecture/interface fingerprints**. In g182, Qwen3 vs GPT-2 feature means were far apart: mid_spectral_alpha 0.666 vs 1.085, kNN10 0.645 vs 0.508, TwoNN ID 7.15 vs 11.18. In g186 the split was even sharper: kNN10 0.730 vs 0.332, spectral alpha 0.989 vs 0.278, TwoNN ID 11.95 vs 17.31.

So the break is: **same feature name does not mean same coordinate across architectures.** A ridge trained on one architecture reads the other architecture’s coordinate chart incorrectly.

**Alternative Spaces**

A p-adic, ultrametric, hyperbolic, or complex representation could help only if it converts these architecture-specific charts into a shared invariant. The current data argues against a simple metric swap:

- g182 z-scoring still failed.
- g182 arm-demeaning still failed.
- g182 P3 showed 0/8 cross-arch feature overlap.
- g186 showed response is mostly arm/dose/architecture: arm_mean R2=0.936.
- g181a says the useful signal is token/interface structure, not internal block geometry.

Best bet: use those spaces for **token/interface priors** - e.g. ultrametric or hyperbolic token co-occurrence structure - not as a magic bridge for transformer-block transfer.

**Successful Design**

A real cross-arch experiment should stop trying to transfer raw geometry and instead learn an **architecture-conditioned compatibility law**:

1. Use native tokenizers for each architecture.
2. Build interface priors per tokenizer: trained embed/lm_head, PPMI/SVD, frequency, random structured, covariance/spectral controls.
3. Normalize geometry within each architecture against scratch populations.
4. Include tokenizer distance/corpus statistics as explicit covariates.
5. Train on at least two families, test frozen on a third.
6. Require: LOAO R2 > 0.25 per arch, >25% MSE reduction over arm_mean/alpha/telemetry, permutation p <= 0.01, and post-normalization feature overlap >=6/8.

That tests a law of compatibility. It does not pretend Qwen geometry and GPT-2 geometry already live in one coordinate system.

**Verdict**

Strong-form cross-architecture transfer is **fundamentally false in the current framing**. Architecture/tokenizer pairs impose different representational charts and response surfaces.

But the broader project is not dead. The methodological error is measuring raw geometry as if it were coordinate-free. The surviving path is narrower: **interface priors and architecture-conditioned intervention response**, not universal raw hidden-state geometry.

Memory note: I used memory only for prior project context; all experiment numbers above came from the current repo JSONs.

