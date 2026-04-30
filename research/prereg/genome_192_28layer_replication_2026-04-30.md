# Pre-registration: g192 28-Layer String-Match Replication

**Status:** DRAFT (g194 PASS_DIRECTION confirmed cycle 180; gated on g195 per Codex §B cycle 168 — g192 inherits tied-head confound until g195 resolves)

## Motivation

g191 PASS_CONTENT and g194 (pending) establish that trained embedding row content — specifically directional content — carries the +0.465 nats signal. But all experiments so far use an 8-layer Qwen3-arch shell. Adversarial A16 #3 (SEV-8) flags this as a scope limit: the signal may be a shallow-init regime artifact that disappears at full depth.

This experiment tests whether the embedding init effect persists at full 28-layer Qwen3-0.6B depth.

## Hypothesis

**H1 (Persistence):** The matched_rows_only init + anchor effect is >= +0.20 nats at 28 layers (at least 43% of the 8-layer effect).

**H2 (Attenuation):** The effect attenuates to < +0.10 nats at 28 layers (deeper model washes out embedding signal).

## Design

### Arms (3 arms x 3 seeds = 9 cells)

| Arm | Init embedding | Anchor | Tests |
|-----|---------------|--------|-------|
| `scratch_ce` | Random (Qwen3 init) | None | Baseline |
| `matched_rows_only` | Matched trained rows at correct positions | Same, masked to matched rows | Reference (expects +0.465 at 8-layer; how much survives at 28?) |
| `row_shuffled` | Permuted matched rows | Same (permuted), masked | Negative control (expects harmful at both depths) |

All arms: **28-layer** Qwen3-arch with GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. Same training hyperparameters (lr, betas, weight_decay, batch_size, grad_clip).

### Model specification

`Qwen3Config(vocab_size=50257, hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072, max_position_embeddings=320, tie_word_embeddings=True, head_dim=128, rope_theta=1000000.0, use_cache=False)`

Matches actual Qwen3-0.6B config (except vocab_size=50257 for GPT-2 tokenizer). Estimated VRAM: ~4.0 GB (well within 22 GB envelope).

## Pass/Fail Criteria

**PASS_PERSISTENCE:** matched_rows_only mean gain >= +0.20 nats vs scratch AND 3/3 seeds AND row_shuffled mean gain <= 0.0 nats (still harmful).

**PASS_ATTENUATION:** matched_rows_only mean gain >= +0.10 but < +0.20 nats AND matched_mean > shuffled_mean + 0.05 nats. Effect exists but is weaker at depth.

**FAIL:** matched_rows_only mean gain < +0.10 nats. Embedding init effect is a shallow artifact.

## Universality Level

Level-3 (architecture-specific, within Qwen3 family). Scale robustness test.

## Compute Envelope

- 28-layer Qwen3-arch, GPT-2 tokenizer, ~3.2 GB VRAM per cell
- 9 cells x ~17 min = ~2.6 hours total
- Within COMPUTE.md 4h envelope

## What a null result means

If FAIL: the embedding init effect is only relevant for shallow models. The "interface codebook" finding (g191, g194) is architecturally interesting but does not scale to practical depths. Training-health diagnostics would need to account for depth-dependent attenuation.
