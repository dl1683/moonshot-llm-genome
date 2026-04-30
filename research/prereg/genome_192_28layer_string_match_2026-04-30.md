# Pre-registration: genome_192 -- 28-Layer String-Match Replication

**Date:** 2026-04-30
**Status:** DRAFT (gated on g191 PASS_CONTENT)
**Design gate:** pending (fires when g191 verdict arrives)

---

## Hypothesis

The g191 PASS_CONTENT finding (matched-row content drives the +0.478 cross-tokenizer signal) persists at full model scale (28-layer Qwen3-0.6B architecture), not just the 8-layer shell. This resolves Codex adversarial A16 attack #3 (SEV-8: shallow-init regime).

## Motivation

- g191 uses 8-layer Qwen3 shell (matching g188 for comparability)
- Codex cycle 155 adversarial: "8-layer/5000-step may be a shallow-init regime. The result may vanish or shrink in the full 28-layer model where deeper dynamics dominate."
- g181b proves persistence for same-tokenizer 8-layer, not full-model cross-tokenizer
- If 28-layer PASS: §0.1 moves from ~4.3 to ~5.0+ (depth-robust cross-tokenizer bridge)
- If 28-layer FAIL: effect is shell-dependent, not architecture-general

## Arms (3 total, 28-layer Qwen3-0.6B with GPT-2 tokenizer)

Only the winning arm from g191 plus controls:

| # | Arm | Init | Anchor | Tests |
|---|-----|------|--------|-------|
| 1 | `scratch_ce` | Random | None | Baseline |
| 2 | `g191_winner` | As per g191 PASS_CONTENT winner | As per g191 | Depth generalization |
| 3 | `row_shuffled_control` | Shuffled matched rows | Shuffled (row-masked) | Format control |

## Protocol

- 3 seeds: [42, 7, 13]
- 5000 steps per cell (matching g188/g191)
- Same C4 train/val with 13-gram dedup
- Same optimizer/LR/batch/seq as g188
- anchor_lambda = 0.01 (matching g188)
- 28-layer Qwen3-0.6B config (full model, NOT 8-layer shell)
- Row-wise anchor masking (as in g191)

## PASS Criteria

**PASS_28L (all must hold):**
- P1: g191_winner mean gain >= +0.25 nats (lower than g191's +0.35 threshold, accommodating possible depth attenuation)
- P2: g191_winner positive in 3/3 seeds
- P3: row_shuffled_control mean gain <= +0.10 nats
- P4: g191_winner beats shuffled by >= +0.15 nats

**FAIL:**
- g191_winner mean gain < +0.15 nats, OR shuffled matches winner

## Compute Envelope (COMPUTE.md)

- [ ] Max VRAM: ~5-6 GB (full 28-layer Qwen3-0.6B). Under 22 GB.
- [ ] Max RAM: < 6 GB. Under 56 GB.
- [ ] Wall-clock: 9 cells x ~12 min = ~1.8h + preprocessing. Total ~2h. Under 4h.
- [ ] Disk: Result JSON ~0.5 MB.
- [ ] Quantization: BF16 (model <1B).
- [ ] Save/resume: per-cell JSON checkpoint.

## Source Files

- Code: `code/genome_192_28layer_string_match.py` (to be written after g191 verdict)
- Results: `results/genome_192_28layer_string_match.json`

## Gating

This experiment ONLY launches if g191 returns PASS_CONTENT. If g191 returns PARTIAL, PASS_FORMAT, or FAIL, this experiment is CANCELLED and the g192 design pivots.
