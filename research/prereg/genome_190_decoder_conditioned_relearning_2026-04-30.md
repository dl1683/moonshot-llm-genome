# Pre-registration: genome_190 — Decoder-Conditioned Relearning

**Date:** 2026-04-30
**Status:** DRAFT (locks after Codex review + smoke test)
**Design gate:** `codex_outputs/g190_decoder_conditioned_relearning_design_gate_20260430.md`

---

## Hypothesis

A decoder-conditioned embedding — GPT-2 vocab embeddings trained against a frozen, fully-trained Qwen3-0.6B decoder — produces anchor targets that transfer meaningfully to a fresh decoder, unlike static bridges (OT/PPMI/char-overlap) which all HARM or attenuate to zero.

## Motivation

- g183 (PPMI SVD bridge): HARM -0.291 nats.
- g188 (Sinkhorn OT bridge): HARM -0.119 nats. char_overlap: -0.041 nats.
- All static cross-tokenizer bridges fail. The common factor: bridges lack decoder-format alignment. They project geometry that the decoder never shaped.
- g181b established that same-tokenizer trained embeddings work as anchors (+0.513 nats). The open question: does the decoder's role in shaping embedding geometry survive re-tokenization?
- If decoder-conditioned embeddings work, it proves the interface is a decoder-compiled codebook, not just corpus statistics.

## Design

### Phase 1: Embedding Relearning (Decoder-Conditioned)

- Load trained Qwen3-0.6B (full 28 layers, all weights).
- Freeze ALL transformer blocks, RMSNorm, RoPE — nothing moves except embed/lm_head.
- Replace `embed_tokens` and `lm_head` with GPT-2-vocab-sized (50257 × 1024) randomly initialized weights, Frobenius-norm-scaled to match trained embedding statistics.
- Tied weights (`embed_tokens.weight == lm_head.weight`).
- Train on GPT-2-tokenized C4 with 13-gram train/val dedup for 2000 steps.
- Checkpoints at steps 250, 500, 1000, 1500, 2000.

**Phase 1 convergence criterion (informational, does not gate Phase 2):**
- Val NLL improves < 0.03 nats over last 500 steps, OR
- Normalized embedding movement `||E_t - E_{t-500}||_F / ||E_t||_F < 0.01`.
- If not converged by 2000, use step 2000 anyway; mark `phase1_converged=false`.

### Phase 2: Anchor Transfer Test

3 arms × 3 seeds × 5000 steps:

| # | Arm | Init | Anchor |
|---|-----|------|--------|
| 1 | `scratch_ce` | Random | None |
| 2 | `relearned_init_anchor` | Phase 1 embed | Phase 1 embed (λ=0.01) |
| 3 | `relearned_anchor_only` | Random | Phase 1 embed (λ=0.01) |

- Full 28-layer Qwen3-0.6B architecture (NOT 8-layer shell from g188).
- GPT-2 tokenizer throughout.
- Same C4 data, LR (5e-4), optimizer (AdamW, β=(0.9,0.95), wd=0.01), batch size, seq length as g181b/g183/g188.
- Anchor: manual `grad.add_` after backward (sum-equivalent, NOT F.mse_loss mean reduction). Matches g181a/g183 implementation.

## Seeds

[42, 7, 13] — same as g188/g183 lineage.

## PASS Criteria

**PASS:**
- `relearned_init_anchor` mean gain >= +0.15 nats vs scratch
- 3/3 seeds positive

**STRONG PASS:**
- init+anchor gain >= +0.257 nats (≥50% of g181b's +0.513)
- anchor-only gain >= +0.15 nats

**PARTIAL:**
- init+anchor gain >= +0.05, ≥2/3 seeds positive

**FAIL:**
- init+anchor gain < +0.05 nats, OR
- anchor-only harms (negative mean gain)

## Interpretation

| Outcome | Meaning |
|---------|---------|
| STRONG PASS | Decoder-conditioned codebooks are transferable training priors. §0.1 → ~5.0-5.7 |
| PASS | Decoder format alignment creates useful anchors. §0.1 → ~4.5-5.0 |
| PARTIAL (init helps, anchor hurts) | Useful warm start but weak anchor thesis |
| Phase 1 succeeds, Phase 2 fails | Adaptation works but does not transfer to fresh decoder |
| anchor-only passes | Strongest evidence: decoder-conditioned targets are basin-shaping objects |

## Compute Envelope (COMPUTE.md §9)

- [x] Max VRAM: Phase 1 ~3 GB (Qwen3-0.6B BF16 + GPT-2 embed/lm_head). Phase 2 ~3 GB (full 28L Qwen3 random init). Well under 22 GB.
- [x] Max RAM: < 4 GB (C4 windows in CPU). Under 56 GB.
- [x] Wall-clock: Phase 1 ~15-20 min. Phase 2: 3 arms × 3 seeds × ~7 min/cell = ~63 min. Total ~80-90 min. Under 4h.
- [x] Disk: Relearned embed cache ~200 MB. Result JSON ~2 MB.
- [x] Quantization: BF16 (model <1B, per quantization ladder).
- [x] Save/resume: per-cell JSON checkpoint, Phase 1 embed cached to disk.

## Source Files

- Code: `code/genome_190_decoder_conditioned_relearning.py`
- Results: `results/genome_190_decoder_conditioned_relearning.json`
- Design gate: `codex_outputs/g190_decoder_conditioned_relearning_design_gate_20260430.md`
