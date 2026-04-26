# Pre-registration: genome_154 distillation smoke test

**Date:** 2026-04-26
**Status:** LOCKED at first commit adding this file.

## Hypothesis

A frozen-teacher logit-distillation signal (top-k KL) trained jointly with cross-entropy provides a measurably stronger learning signal than pure CE for our minimal_3L_30M MLP-free student architecture. If true, this validates the distillation pipeline mechanics and unlocks g155 production-scale distillation with stronger teachers — the Codex-identified high-leverage path toward an MLP-free edge inference product.

## End-goal alignment

This experiment serves Neural Genome's TIER-0 end goal (CLAUDE.md §0): efficient transfer of trained capabilities from a trained model into an untrained model. The teacher (Qwen3-0.6B) holds capabilities we want to transfer. The student (minimal_3L_30M, no MLP) is untrained. Distillation is the mechanism. If the smoke test passes, the production scale-up moves us materially toward the manifesto's electricity-grade efficiency demo.

## System

- **Teacher:** `Qwen/Qwen3-0.6B` (canonical registry), frozen, BF16, attn_implementation=eager
- **Student:** minimal_3L_30M Llama-3 architecture (3 layers, hidden=384, 6 heads, no MLP via ZeroMLP, RMSNorm, RoPE, tied embeddings)
- **Tokenizer:** Qwen3 tokenizer used for BOTH arms (teacher's vocab is required for KD; control arm uses same tokenizer for fair comparison)
- **Data:** `c4_clean_v1` stimulus bank, N_TRAIN=4096 sequences × SEQ_LEN=256
- **Eval:** N_C4_EVAL=200 sequences (in-distribution)
- **Seed:** 42 (single seed — this is a smoke test; multi-seed validation deferred to g155 if PASS)

## Arms

1. **scratch:** Student trained with cross-entropy only on the same N_TRAIN pool
2. **kd:** Student trained with mixed loss `(1 - γ) * CE + γ * T² * KL_topk(student/T || teacher/T)` where γ=0.5, T=2.0, top-k=64

Teacher logits are precomputed once for all N_TRAIN sequences and cached on disk to keep both arms reading from a single fixed reference distribution. This eliminates teacher-stochasticity confound.

## Hyperparameters

- LR: 3e-4 (winning LR for minimal arm from g151)
- Linear LR warmup: 200 steps (per g150 protocol)
- Total steps: 4000
- Batch size: 8
- Optimizer: AdamW, default betas

## Metrics

- **Primary:** C4 top-1 next-token accuracy on held-out eval set after 4000 steps
- **Secondary:** Final eval NLL on the same C4 eval set
- **Tertiary:** Top-5 accuracy

## Pre-stated criteria

- **PASS:** kd beats scratch by ≥0.30pp absolute on C4 top-1. Distillation pipeline works; ready to scale to bigger teacher and student in g155.
- **PARTIAL:** kd matches scratch within ±0.30pp (no clear lift but no regression). Indicates pipeline runs correctly but smoke-scale teacher signal is too weak to detect; g155 still warranted but with stronger teacher first.
- **KILL:** kd is worse than scratch by >0.30pp on C4 top-1. Pipeline mechanics broken — diagnose before any g155 work.

## Universality level claimed

**None at this stage.** This is a methods-validation pilot, not a Level-1 claim about universal phenomena. No primitive added to MEASUREMENT_PRIMITIVES.md regardless of outcome.

## What a null result means

A KILL or PARTIAL outcome means top-k=64 KL with γ=0.5 / T=2.0 at 4096-sample scale is not enough signal to lift a tiny MLP-free student. That does NOT kill the production-distillation thesis; it kills only this specific protocol. g155 redesign options (per Codex AA1 follow-up) would include: increase γ, increase top-k, switch to full-vocab KL, switch to hidden-state matching, or move teacher to a larger model where the logit gap to random init is bigger.

## Compute envelope (COMPUTE.md §9 compliance)

- VRAM: teacher BF16 (~1.2 GB) + student (~120 MB) + activations + KV cache. Peak <8 GB. ✓
- RAM: top-k cache for 4096 × 255 tokens × (64 idx + 64 logits) ≈ 130 MB. ✓
- Wall-clock: teacher cache pass ~3 min + 2 student arms × ~5 min ≈ 15 min total. ✓
- Disk: result JSON + topk cache ~150 MB. ✓
- Quantization: teacher BF16 (sub-1B per ladder), student BF16. ✓
- Checkpointing: not required at 15-min runtime.

## Artifacts

- `code/genome_154_distillation_smoke.py` (already committed b0bcdc0)
- `results/genome_154_distillation_smoke.json` (per-arm metrics)
- `results/genome_154_run.log` (stdout)

## Locked at commit

This file is LOCKED upon commit. Any change to hypothesis, arms, criteria, or thresholds invalidates the pre-registration. Re-run requires a new pre-reg dated for the new run.
