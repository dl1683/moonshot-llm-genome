# Pre-registration: g193 Token-Row Compiler

**Status:** DRAFT (not yet locked)
**Date:** 2026-04-30
**Depends on:** g191 PASS_CONTENT (gating condition)

## Hypothesis

A small MLP trained to predict Qwen3 trained embedding rows from token-level features (byte histogram, token length, log-frequency) using the 42k exact string-matched GPT-2/Qwen3 token pairs as supervision can generate useful initialization rows for ALL tokens, including the ~8k unmatched tokens. A GPT-2-tokenizer Qwen3-arch shell initialized with ONLY compiler-generated rows (no copied target rows) will outperform scratch training.

## Measurement

- **Primary metric:** NLL gap between compiled_init_anchor and scratch_ce (mean over 3 seeds)
- **Compiler quality:** Holdout MSE and cosine similarity on 20% held-out matched tokens
- **Control:** compiled_shuffled (row-permuted compiler output) must show <= +0.10 nats gain

## Systems

- Source: Qwen3-0.6B (trained embeddings as supervision targets)
- Target: GPT-2-tokenizer 8-layer Qwen3-arch shell (same as g188/g191)

## Arms (4 arms x 3 seeds = 12 cells)

1. **scratch_ce** — random init, no anchor (baseline)
2. **compiled_init_anchor** — init + anchor with compiler-generated rows
3. **compiled_init_only** — init only, no anchor
4. **compiled_shuffled** — row-permuted compiler output (content-destruction control)

## Compiler Architecture

- Input: 258-dim (256-dim byte histogram + token length + log-frequency)
- Model: 258 → 1024 → 1024 → 1024 (GELU + LayerNorm)
- Loss: MSE between predicted and actual Qwen3 embedding rows
- Train/holdout: 80%/20% split of 42,257 matched token pairs
- Early stopping: patience=30 on holdout MSE

## Pass/Fail Criteria

### PASS
- compiled_init_anchor mean gain >= +0.30 nats (vs scratch)
- compiled_init_anchor positive 3/3 seeds
- compiled_shuffled mean gain <= +0.10 nats
- compiled_mean - shuffled_mean >= +0.20 nats

### PARTIAL
- compiled_init_anchor mean gain >= +0.15 nats

### FAIL
- compiled_init_anchor mean gain < +0.15 nats

## What a null result means

If the compiler fails to produce useful rows, it means the mapping from token-level features to trained embedding geometry is NOT learnable from byte-level statistics + frequency alone. The trained embedding row content depends on higher-order contextual information that byte histograms cannot capture. This would suggest the +0.478 nats signal is purely about exact lexical identity (copying the right row), not about a learnable function from token form to embedding content.

## COMPUTE.md compliance

- [x] VRAM: 8-layer Qwen3-arch + compiler MLP << 22GB
- [x] RAM: feature matrices + embeddings << 56GB
- [x] Wall-clock: compiler trains in ~5 min; 12 shell cells x ~7 min = ~84 min total
- [x] Checkpointing: incremental JSON save per cell
