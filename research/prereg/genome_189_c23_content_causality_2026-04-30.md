# Pre-registration: genome_189 — C23 Content-Causality Controls

**Date:** 2026-04-30
**Status:** DRAFT (locks after Codex review + smoke test)
**Design gate:** `codex_outputs/g189_content_causality_design_gate_20260430.md`

---

## Hypothesis

The +0.513 nats gain from embed/lm_head trained anchor (g181b, C23) is caused by the CONTENT of the trained embeddings (specific learned token-to-vector relationships), not by FORMAT properties (norm, spectrum, token-frequency structure, or generic anchoring).

## Motivation

- Adversarial cycle 150 (SEV-10): C23 is not proven content transfer. Any matched embedding/head constraint with trained-like norm/spectrum/token-frequency structure might work equally well.
- g181b PASS (+0.513 nats) established the anchor works, but did not isolate WHAT about the anchor is beneficial.
- Five matched controls systematically isolate format vs content properties.

## Arms (7 total, Qwen3-0.6B, same tokenizer)

| # | Arm | What it preserves | What it destroys |
|---|-----|------------------|-----------------|
| 1 | `scratch_ce` | — | — (baseline) |
| 2 | `true_trained_anchor` | Everything (C23 candidate) | — |
| 3 | `row_shuffled_anchor` | Exact rows, norms, spectrum, covariance | Token identity |
| 4 | `freq_bucket_shuffle_anchor` | Frequency structure + row norms + spectrum | Exact content within buckets |
| 5 | `spectrum_preserving_random` | SVD spectrum, Frobenius norm | All row structure |
| 6 | `same_frobenius_gaussian` | Frobenius norm only | Everything else |
| 7 | `anchor_to_initial` | Plausible init geometry | Trained content |

## Protocol

- 6 seeds: [42, 7, 13, 101, 202, 303]
- 5000 steps per cell (same as g181b)
- Same C4 train/val with 13-gram dedup
- Same optimizer/LR/batch/seq as g181b
- Anchor gradient matched per seed: `lambda_arm = G_ref / (2 * d_arm)` where `G_ref = 2 * lambda_base * d_true`
- All targets Frobenius-norm matched to trained embeddings

## PASS criteria (all must hold)

**PASS_CONTENT:**
- P1: true_trained_anchor beats scratch by >= +0.20 nats mean (reproduces g181b)
- P2: true_trained_anchor beats EVERY control arm by >= +0.20 nats mean
- P3: true_trained_anchor beats EVERY control arm in >= 5/6 paired seeds

**FORMAT_FAIL:**
- Any control within 0.20 nats of true, OR
- Any control beats true in >= 2/6 seeds

**REPRO_FAIL:**
- true_trained does not beat scratch by >= +0.20 nats

## Interpretation table

| If this control matches true... | Then C23 is about... |
|--------------------------------|---------------------|
| row_shuffled | Format (norms + spectrum), not content |
| freq_bucket_shuffle | Frequency-codebook prior, not exact lexical content |
| spectrum_preserving | Global spectral format alone |
| same_frobenius_gaussian | Pure norm/regularization |
| anchor_to_initial | Generic anchoring, not trained content |

## Compute envelope (COMPUTE.md compliance)

- 7 arms x 6 seeds x 5000 steps = 42 cells
- Per cell: ~400s (from g181b timing). Total: ~16800s = 4.7h per seed-block
- Must chunk by seed (7 cells/seed ~= 47 min): six invocations
- VRAM: ~10-12 GB (Qwen3-0.6B training). Under 22 GB envelope.
- RAM: target construction (SVD of 151K x 1024 matrix) ~1.2 GB. Under 56 GB.
- Wall-clock: each invocation ~50 min with resume. Well under 4h per block.
- Save/resume: per-cell JSON checkpoint.

## Source files

- Code: `code/genome_189_c23_content_causality.py`
- Results: `results/genome_189_c23_content_causality.json`
- Design gate: `codex_outputs/g189_content_causality_design_gate_20260430.md`
