**Findings**

1. **SEV-10: geometry model leaks early validation loss.**  
   The script adds `early_loss_slope` from step 10/25/50 validation NLL into `geometry_features`, then excludes only `early_loss_50` from the geometry model. That violates the prereg’s “early loss excluded” rule and makes the geometry-vs-step-50 baseline unfair.  
   [code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:601>), [feature filter](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:656>), [prereg](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:59>)

2. **SEV-8: several lm_head conditions do not match the prereg.**  
   `covariance_scaffold` samples from trained mean/covariance, then normalizes every row to one mean norm, so it no longer preserves the sampled covariance/norm geometry. Trained-row unmatched rows are also filled at a single mean norm, not the matched-row norm distribution.  
   [covariance code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:160>), [unmatched fill](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:107>), [prereg](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:28>)

3. **SEV-7: matched-row-only features are missing.**  
   The prereg locks features on both the full sampled head and matched rows only. The implementation extracts one feature set from `row_sample_idx` only; no matched-row mask/path exists in `extract_geometry_features` or the step-0/50 calls.  
   [extractor](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:256>), [step-0 call](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:497>), [prereg](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:47>)

4. **SEV-5: `geom_wins` uses MSE, not absolute error.**  
   PASS requires geometry beating early loss in at least 8/10 held-out conditions “by absolute error.” The code increments wins by held-out-condition MSE. This can flip a pass/fail boundary.  
   [code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:716>), [prereg](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:81>)

5. **SEV-4: frequency-bucket kNN purity labels the wrong rows.**  
   `W_knn` uses random sample positions `knn_idx`, but `knn_sample_freqs` takes the first `knn_n` sampled rows instead of `row_sample_idx[knn_idx]`. That makes `freq_bucket_purity` incorrect.  
   [code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:337>), [bug line](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_197_output_interface_canary_arena.py:371>)

**Checks That Passed**

LOCO CV is implemented correctly: each held-out condition tests exactly 3 seeds and trains on the other 27 rows. Seed-stratified permutation is correct: geometry rows are permuted within seed groups, not globally, and the p-value formula is `(exceedance + 1) / (n_perm + 1)`. Ridge uses `RidgeCV` with the prereg alpha grid inside each outer train fold.

Performance looks in-envelope but tight: expected peak VRAM is roughly **5-7 GB**, not the prereg’s `<5 GB`, due to FP32 params/grads/Adam states plus logits/activations. CPU RAM should stay well below 56 GB; the 151936-vocab Qwen head is copied but geometry SVD/kNN runs on the GPT-2 remapped 50257-vocab head/sample, so no obvious OOM. kNN at 4096x1024 is fine. Wall clock is likely **~3.8-4.4 h** including feature/head-construction overhead, so fix the correctness blockers before spending the run.

Launch recommendation: **do not launch full g197 yet.** Fix at least findings 1-4, then rerun smoke.

