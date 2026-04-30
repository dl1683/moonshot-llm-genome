# Pre-registration: genome_191 — String-Match Decomposition

**Date:** 2026-04-30
**Status:** DRAFT (locks after smoke test)
**Design gate:** `codex_outputs/g191_string_match_decomposition_design_gate_20260430.md`

---

## Hypothesis

The g188 direct_string_match signal (+0.478 nats mean, 93% of g181b same-tokenizer effect) is carried by the CONTENT of matched trained embeddings (specific learned token-to-vector relationships at exact-string-matched positions), not by FORMAT properties (norm, spectrum, frequency structure) or the unmatched-row fill.

## Motivation

- g188 direct_string_match: +0.478 nats (2/3 seeds done). First positive cross-tokenizer bridge.
- g188 flow_bridge (OT): -0.119 nats (HARMS). Sinkhorn transport destroys the signal.
- g188 char_overlap: -0.041 nats (attenuates to negative).
- 84% of GPT-2 tokens exactly match Qwen3 token strings. Direct copy works; blending doesn't.
- SEV-10 adversarial (cycle 150): C23 is not proven content transfer. This experiment resolves that for the cross-tokenizer case.

## Arms (7 total, 8-layer Qwen3 shell with GPT-2 tokenizer)

| # | Arm | Init | Anchor | Tests |
|---|-----|------|--------|-------|
| 1 | `scratch_ce` | Random | None | Baseline |
| 2 | `direct_init_only` | Full string-match matrix | None | Warm start without anchor |
| 3 | `direct_anchor_only` | Random | Full string-match matrix | Anchor alone |
| 4 | `matched_rows_only` | Matched rows only | Matched rows only (row-masked) | Signal source |
| 5 | `unmatched_rows_only` | Random | Unmatched rows only (row-masked) | Inverse control |
| 6 | `row_shuffled_matched` | Shuffled matched rows | Same (row-masked) | Format vs content |
| 7 | `frequency_bucket_shuffle` | Freq-quintile shuffled | Same (row-masked) | Frequency structure |

## Protocol

- 3 seeds: [42, 7, 13]
- 5000 steps per cell (matching g188)
- Same C4 train/val with 13-gram dedup
- Same optimizer/LR/batch/seq as g188
- anchor_lambda = 0.01 (matching g188)
- Row-wise anchor masking: anchor gradient only applied to rows where mask is True
- All embeddings Fro-norm matched to trained Qwen3 embeddings
- 8-layer Qwen3 shell (NOT full 28-layer), matching g188 for comparability

## PASS Criteria

**PASS_CONTENT (all must hold):**
- P1: `matched_rows_only` mean gain >= +0.35 nats
- P2: `matched_rows_only` positive in 3/3 seeds
- P3: `matched_rows_only` >= 70% of g188 `direct_string_match` full effect
- P4: Both `row_shuffled_matched` and `frequency_bucket_shuffle` mean gain <= +0.10 nats
- P5: `matched_rows_only` beats both shuffled arms by >= +0.25 nats

**PASS_FORMAT (alternative):**
- `row_shuffled_matched` mean gain >= +0.25 nats

**PARTIAL:**
- `matched_rows_only` mean gain >= +0.15 nats but shuffled controls not clearly beaten

**FAIL:**
- `matched_rows_only` mean gain < +0.15 nats, OR format controls match content

## Interpretation Table

| If this arm is high... | Then the signal is... |
|------------------------|----------------------|
| `matched_rows_only` high, shuffled low | Trained content at exact-string positions |
| `row_shuffled_matched` matches `matched_rows_only` | Format (norms + spectrum), not content |
| `frequency_bucket_shuffle` high, row_shuffle low | Frequency structure is sufficient |
| `direct_init_only` high, `direct_anchor_only` low | Warm start dominates |
| `direct_anchor_only` high | Anchor itself is causal (geometry-as-regularizer) |
| Both init/anchor weak, full strong | Init-anchor interaction is the mechanism |
| `unmatched_rows_only` high | Mean-fill of unmatched rows carries signal |

## Compute Envelope (COMPUTE.md)

- [x] Max VRAM: ~3 GB (8-layer Qwen3 shell). Under 22 GB.
- [x] Max RAM: < 4 GB. Under 56 GB.
- [x] Wall-clock: 21 cells x ~6.8 min = ~2.4h + preprocessing. Total ~2.5-2.7h. Under 4h.
- [x] Disk: Result JSON ~2 MB. No large artifacts.
- [x] Quantization: BF16 (model <1B).
- [x] Save/resume: per-cell JSON checkpoint.

## Source Files

- Code: `code/genome_191_string_match_decomposition.py`
- Results: `results/genome_191_string_match_decomposition.json`
- Design gate: `codex_outputs/g191_string_match_decomposition_design_gate_20260430.md`
