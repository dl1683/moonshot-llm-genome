# Pre-registration: g195 Untied Input/Output Factorial

**Status:** LOCKED (g194 PASS_DIRECTION confirmed cycle 180; resolves A18 SEV-10 #1 tied lm_head confound)

## Motivation

All g191/g194 experiments use `tie_word_embeddings=True`. With tying, injecting trained embeddings into `embed_tokens` simultaneously sets the output classifier basis (`lm_head`). The +0.465 nats signal may be OUTPUT-logit class-vector prior, not INPUT embedding/interface geometry. This experiment untie the weights and tests each side independently.

## Hypothesis

**H1 (Input dominates):** input_inject_anchor gain >= +0.30 nats AND output_inject_anchor < +0.15 nats. The signal is genuinely input embedding geometry.

**H2 (Output dominates):** output_inject_anchor gain >= +0.30 nats AND input_inject_anchor < +0.15 nats. The signal is output logit geometry. "Embedding interface" framing is wrong.

**H3 (Both needed):** both_inject_anchor >= +0.30 but neither alone > 80% of combined. The mechanism requires input-output coherence.

## Design

### Arms (5 arms x 3 seeds = 15 cells)

| Arm | embed_tokens init/anchor | lm_head init/anchor | tie_word_embeddings | Tests |
|-----|--------------------------|---------------------|---------------------|-------|
| `scratch_untied` | Random | Random | False | Baseline |
| `input_inject_anchor` | Trained matched rows | Random | False | Input-only |
| `output_inject_anchor` | Random | Trained matched rows | False | Output-only |
| `both_inject_anchor` | Trained matched rows | Trained matched rows | False | Both (untied) |
| `tied_reference` | Trained matched rows | (tied) | True | Tied comparison |

All arms: 8-layer Qwen3-arch, GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. String-matched embeddings normalized to trained_fro.

### Smoke test (50 steps, seed=42)

| Arm | NLL | Gap vs scratch |
|-----|-----|---------------|
| scratch_untied | 7.605 | — |
| input_inject_anchor | 7.458 | +0.147 |
| output_inject_anchor | 7.422 | +0.183 |
| both_inject_anchor | 7.198 | +0.407 |
| tied_reference | 7.200 | +0.405 |

both_inject_anchor matches tied_reference (design validation). Neither side alone reaches combined effect at 50 steps.

## Pass/Fail Criteria

**PASS_INPUT:** input_inject_anchor mean gain >= +0.30 AND output_inject_anchor < +0.15. Signal is input embedding geometry.

**PASS_OUTPUT:** output_inject_anchor mean gain >= +0.30 AND input_inject_anchor < +0.15. Signal is output logit geometry. Kills "embedding interface" framing.

**PASS_INPUT_DOMINANT:** input_inject_anchor >= +0.20 AND input > output. Input carries more.

**PASS_OUTPUT_DOMINANT:** output_inject_anchor >= +0.20 AND output > input. Output carries more.

**PASS_BOTH_NEEDED:** both_inject_anchor >= +0.30 AND neither alone > 80% of combined. Coherence required.

**FAIL:** No arm reaches +0.10 nats gain.

## Universality Level

Level-3 (architecture-specific, within Qwen3 family). Mechanism isolation test.

## Compute Envelope

- 8-layer Qwen3-arch, GPT-2 tokenizer, ~4.3 GB VRAM peak (untied both-anchor worst case)
- 15 cells x ~7 min = ~1.75 hours total
- Within COMPUTE.md 4h envelope

## What a null result means

If PASS_OUTPUT: the entire "embedding interface" narrative is wrong. What we've been calling "trained embedding content" is actually "trained logit class vectors." The signal is output geometry, not input geometry. The training-health diagnostic framing survives but the mechanism shifts.

If PASS_BOTH_NEEDED: the signal requires input-output coherence (consistent codebook). Tying is mechanistically important, not just a confound.
