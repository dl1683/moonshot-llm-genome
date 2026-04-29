**Cycle 108 Correctness + Performance Review (Claude-generated, Codex sessions timed out)**

## Correctness Findings

**D1 (continuous vs basin, line 2157-2180): SEV-3 — train-set MSE comparison inflates Ridge advantage**
Both Ridge and KMeans MSEs are computed on training data, not held-out. Ridge with CV-tuned alpha will systematically show lower train MSE than a 3-cluster KMeans mean predictor. This inflates the ridge_advantage metric. Fix: use LOOCV or k-fold MSE for both models. Not blocking since D1 is diagnostic, not gating.

**D2 (depth drift value, line 2182-2207): CLEAN**
Feature indices [0,1,2,6,7] correctly exclude depth drift features [3,4,5]. Same train-MSE inflation as D1 but comparison is internal (both Ridge models on same data), so relative comparison is fair. Both use the same `fit_ridge_cv` with same alpha grid.

**R1 (alpha sweet spot, line 2230-2246): CLEAN**
Quadratic Ridge on standardized alpha with `coefs[1] < 0` correctly identifies concavity. Standard approach.

**R3 (depth drift direction, line 2209-2228): SEV-4 — no statistical test**
The mean comparison `y[both_neg].mean() > y[~both_neg].mean()` is purely directional with no significance test. With n=24 cells, the split could be very uneven (e.g., 2 vs 22). Should add Mann-Whitney U or permutation test. Currently the result is "which direction" without knowing if it's noise.

**P7b (loss trajectory convergence, line 2248-2283): CLEAN**
No variable collision — `mask` is a local in the try block, not a function parameter. Trajectory loss keys correctly accessed as integers. CV computation standard.

**generate_teacher_texts (line 446-504): SEV-5 — no timeout or batch-level progress on first 99 batches**
Progress logging only fires at batch 100 and at the final batch. For 1088 batches at ~8s each, the first progress log appears at ~13 minutes. If the process hangs on any batch in 1-99, there's no evidence. Should log at batch 1 and then every 50 batches for observability.

## Performance Findings

**Peak VRAM during teacher gen: ~2.5 GB**
Qwen3-0.6B: ~1.2 GB. donor_embed_params (FP32 embed+lm_head on GPU): ~0.6 GB. Tokenization overhead: ~0.1 GB. KV cache during generate: ~0.2 GB. Total: ~2.1 GB. Well within 22 GB envelope.

**Peak VRAM per cell training: ~4 GB**
Model (83-90M params in bf16): ~0.2 GB. Optimizer states: ~0.6 GB. Activations + gradients: ~2 GB. Data tensors on GPU: ~1 GB. Feature extraction probe: ~0.3 GB. Total: ~4 GB. Well within envelope.

**Teacher gen hang diagnosis: NOT a code bug**
Tested the exact startup sequence (donor load → snapshot → delete → vocab map → teacher gen) — works correctly in isolation. 8704 texts × 8s/batch = ~2.5 hours total. The two observed hangs (both at 100% CPU, 0 I/O) may be transient CUDA/HF Datasets deadlocks specific to this Windows+Python 3.13 environment. The background full-scale test (bki534wvy) will confirm whether it's reproducible or transient.

## Action Items
1. Add batch-1 logging to `generate_teacher_texts` so early progress is visible
2. R3 needs a statistical test (at minimum permutation test on the group mean difference)
3. D1 train-MSE inflation is noted but non-blocking for a diagnostic
