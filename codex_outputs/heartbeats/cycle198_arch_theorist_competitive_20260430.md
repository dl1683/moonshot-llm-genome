**Bottom Line**

The unified theory is: **the lm_head is an output spherical codebook that defines the Fisher/gradient geometry of training**. g191 says the useful object is exact token-row content; g194 says that content is almost entirely **directional** on the unit sphere; g195 says the dominant surface is the **output classifier**, not the input embedding. Mathematically, for cross-entropy:

```text
grad_h L = W_out^T (p - e_y) = sum_j p_j w_j - w_y
```

So the row direction `w_y / ||w_y||` is the target gradient prototype for token `y`. If the direction is correct, training gets the right “where to move” signal. If rows are shuffled, the model is taught wrong token targets. Norms mostly act like temperature/frequency scaling, not content.

That makes the right framework a hybrid of **spherical codebooks + Fisher/NTK geometry + rate-distortion water-filling**. Neural collapse/ETF is only a special balanced-class case; language modeling needs a frequency-weighted, anisotropic output codebook.

**g197 Interpretation**

If g197 passes, Route 2 gets real support: output-interface geometry is not just a post-hoc artifact; it is a controllable early boundary condition whose spectrum/angular coverage predicts final NLL beyond early loss. But it still only validates the empirical water-filling proxy. To claim theory, you need a derived operator like:

```text
M_W = C_h^{1/2} W_out^T (Diag(pi) - pi pi^T) W_out C_h^{1/2}
```

and show its eigenvalue/rate profile predicts outcome better than generic features.

If g197 fails with meaningful NLL spread, then either the current features miss the true operator, or early loss already absorbs the useful geometry by step 50. If it fails because final NLL range is small, the arena failed, not Route 2.

**Feature Set**

Mostly right, but incomplete. Spectral rank, angular uniformity, kNN geometry, scaffold distances, and norms are good proxies. Missing high-value additions:

1. **Frequency-weighted spectra/PR/effective rank** using C4 token distribution `pi`.
2. **Fisher/logit operator features** from `W_out^T (Diag(pi)-pi pi^T) W_out`.
3. **Codebook smoothness**: cosine similarity vs token co-occurrence/PMI/edit/semantic neighborhoods.
4. **Hidden-state/head coupling** at step 50: spectrum of `W_out C_h W_out^T`, logit entropy, gradient anisotropy.
5. **Ablation split**: no-reference geometry-only vs scaffold-distance/reference features, to prove the model is not just recognizing condition families.

One concrete pre-launch issue: in [code/genome_197_output_interface_canary_arena.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_197_output_interface_canary_arena.py:336>), `freq_bucket_purity` appears to use the random kNN row subset for neighbors but the first `knn_n` sampled frequencies for labels. It should likely use `sample_freqs[knn_idx]`. Fix or drop that feature before launch.

**Next Derivation**

Write the **Output-Interface Water-Filling Theorem**:

For fixed Frobenius norm and token distribution `pi`, expected early-to-mid training loss improvement is controlled by the eigenvalue profile of the output Fisher codebook operator. Healthy heads maximize useful rate:

```text
Phi(W) = sum_i log(1 + lambda_i / theta)
```

with `lambda_i` from the frequency-weighted output Fisher/hidden coupling matrix. This would directly unify g191/g194/g195 and make g197 a theorem test, not just a regression test.

Expected §0.1 impact: **+0.6 to +1.0** if the derivation predicts g197 feature rankings and survives a held-out canary.

**Competitive Read**

I found adjacent work, but not the exact claim “LLM output-head initialization geometry predicts final training health.” Adjacent threats:

- Dynamical isometry and NTK show initialization/Jacobian spectra affect trainability: Pennington et al. 2017, Xiao et al. 2018, Jacot et al. 2018.
- Neural collapse work studies last-layer classifier geometry and transferability, including Wang et al. ICCV 2023 and NC-informed initialization in 2026.
- In-training probes predict downstream LLM performance from hidden states, not output-head geometry: Liu et al. 2026.
- The Geometric Canary predicts steerability/drift from representational stability, not pretraining outcome from lm_head geometry.
- MoE Fisher geometry predicts training failure at 10%, but for router specialization, not vocabulary head geometry.

Versus Anthropic/DeepMind interpretability: they map mature model features/circuits. Your chain is earlier and more causal: “which output codebook makes training healthy?” Versus MIT CRH / Huh Platonic representations: your result is a boundary-condition counterweight. It says representation convergence is not enough; architecture/tokenizer/interface charts matter before convergence.

If g197 passes, “big labs cannot publish this” is false. They can. “Big labs may not publish it because it exposes training triage/initialization heuristics” is plausible but not a moat. Honesty rating: **6/10 will-not-publish, 2/10 cannot-publish**.

**Highest-Leverage Move**

After g197, run a **prospective g198 canary policy**: freeze the g197 predictor, create unseen output-head conditions, score at step 50, then only continue predicted-good vs predicted-bad arms. Show compute saved or bad runs killed before final training.

Expected §0.1 impact: **+0.8 to +1.3** if successful. This moves the project from “interesting predictor” to “operational training triage,” which is the competitive wedge.

Sources: repo files read include [CLAUDE.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\CLAUDE.md:9>), [WIKI.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\WIKI.md:29>), [OPEN_MYSTERIES.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\research\OPEN_MYSTERIES.md:158>), [CLAIM_EVIDENCE_MAP.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\research\CLAIM_EVIDENCE_MAP.md:192>), [EXPERIMENTS.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\experiments\EXPERIMENTS.md:7>), [early_geometry_predicts_training_health.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\research\derivations\early_geometry_predicts_training_health.md:34>), and [g197 prereg](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\research\prereg\genome_197_output_interface_canary_arena_2026-04-30.md:45>). External sources: [Dynamical Isometry](https://proceedings.neurips.cc/paper/2017/hash/d9fc0cdb67638d50f411432d0d41d0ba-Abstract.html), [NTK](https://papers.nips.cc/paper/8076-neural-tangent-kernel), [Neural Collapse](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html), [Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html), [Anthropic circuit tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/), [Geometric Canary](https://huggingface.co/papers/2604.17698), [In-training LLM probes](https://papers.cool/arxiv/2604.01025), [Rate-distortion efficient codes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952), [Coverage Principle](https://www.microsoft.com/en-us/research/publication/the-coverage-principle-how-pre-training-enables-post-training/).

