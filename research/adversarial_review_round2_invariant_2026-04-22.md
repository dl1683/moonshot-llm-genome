# Adversarial review ROUND 2 - new blind spots (2026-04-22)

New blind spots (not Round-1 #1-#6). Each could inflate the apparent "universal mid-depth attractor".

1) **Fractional depth may be the wrong axis.** Normalized depth = `layer_index/(n_layers-1)` + interpolation assumes "halfway through BERT-12" matches "halfway through Qwen-28". Test: re-index layers by functional progress (logit-lens dNLL, attention distance/entropy, or `||d logits / d h_l||`) and see if the tight band survives.

2) **`seq_mean` pooling may be the whole story.** Pooling=`seq_mean` averages tokens; scrambled word order preserving the effect screams "bag-of-words/unigram". Test token-level clouds (no averaging) plus `last_token` (CLM), attention-weighted pooling, position buckets, and surprisal buckets.

3) **C4 universality could be dataset adjacency/memorization.** C4_clean is plausibly in-distribution for all models; wikitext is still training-adjacent. Test harder OOD (non-English scripts, code/math, synthetic grammar, freshly-written text) and plot CV vs token-frequency divergence from C4.

4) **f(depth) might be "settling" + measurement-regime artifact.** With `n_stimuli=800`, spectra are rank-capped (`<=800`) and may concentrate generically once layers are LN-stabilized. Test n-sweep (800->5k->50k), pre-LN vs post-LN taps, and component spectra (attn-out vs mlp-out vs residual). If universality shifts, it's not an attractor.

5) **Cross-class meaning is undefined.** "Mid-depth 0.4-0.9" has no analog for diffusion, RL agents, world models, or recurrent policies. Either define a compute axis (diffusion timestep, unroll step, planning iteration) and replicate, or scope the claim to transformer text encoders.

6) **The null catalog implies missing variables; "direction overlap" doesn't pin content.** Matching eigenvalues (and even a shared top-k stimulus subspace) ignores rotations within the subspace, hidden-space structure, and conditional/higher-order stats (token-conditioned covariance, cross-token coupling, non-Gaussianity). Do Procrustes/CCA alignment, compare per-stimulus coefficients + higher moments, and localize which directions differ between teacher and spectrum-matched student.
