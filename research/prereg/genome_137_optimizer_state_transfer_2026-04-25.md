# Pre-registration: genome_137 optimizer-state transfer

**Date:** 2026-04-25  
**Status:** LOCKED at first commit adding this file.

## Hypothesis

AdamW moment state (`m`, `v`, step count) contains path-dependent training information not present in weights alone. If true, at the same weight snapshot, preserving the correct optimizer state should improve subsequent convergence versus resetting or mismatching the state.

## System

- Tiny Llama 30M stack from genome_133/135/136
- Same fixed `c4_clean_v1` token pool/order protocol as genome_136
- Seeds: `[42, 7, 13]`
- Donor checkpoint: `K = 1000`
- Continuation horizon: `1000 -> 4000`

## Arms

For each seed, train one donor to step 1000, then fork:

1. `resume_true`: donor weights + donor optimizer state  
2. `resume_reset`: donor weights + fresh AdamW state  
3. `resume_foreign`: donor weights + optimizer state from another seed's donor at step 1000  
4. `state_only`: fresh random weights + donor optimizer state

## Metrics

- Primary: post-K `CtQ_75`, where target is `donor_final + 0.25 * (NLL_K - donor_final)`
- Secondary: mean eval NLL over first 128 continuation steps
- Tertiary: final eval NLL at step 4000

## Pre-stated criteria

- **PASS:** `resume_true` beats `resume_reset` by `>=20%` on post-K `CtQ_75` and `>=0.05` mean-NLL over first 128 steps in `>=2/3` seeds; `resume_foreign` is worse than `resume_true`; `state_only` shows no speedup over scratch.
- **PARTIAL:** `10-20%` `CtQ_75` gain or consistent short-horizon NLL gain without full threshold.
- **KILL:** `resume_true` and `resume_reset` are within `+/-5%` `CtQ_75` and `+/-0.03` early-mean NLL. Optimizer state adds no material transferable signal beyond weights.

## Compute

Approx. `3*1000 + 9*3000 + 3*4000 = 42000` train-step equivalents plus evals: about **20-25 min on RTX 5090**.
