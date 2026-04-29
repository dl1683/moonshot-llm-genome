# genome_182 — Blinded Training Triage Arena (DRAFT)

**Status:** DRAFT for Codex design gate. Not locked.

## Motivation

Adversarial A9 (cycle 75) flagged: g180's n=9 test set is effectively n=3 independent
samples, the early-loss-only baseline is too weak, and g180b tests tokenizer-perturbation
within Qwen3-shell, not true cross-architecture generalization.

Codex direction review (cycle 75) scored current trajectory at 5.0/10 and proposed the
"Blinded Training Triage Arena" as the §0.1=9.0 experiment.

## Hypothesis

Early-training geometry features (at ≤3% of training) predict final C4 NLL gain better
than ALL of: scalar early loss, early-loss trajectory (slope + curvature), gradient stats,
and shuffled-geometry controls — across ≥2 genuinely distinct architecture families.

## Design

### Architecture families (truly distinct, not tokenizer-swaps)

1. **Qwen3-arch** (GQA, SwiGLU, RoPE, Qwen3 tokenizer) — existing from g180 train data
2. **GPT-2-arch** (MHA, GELU, learned pos emb, GPT-2 tokenizer) — NEW
3. Optional phase 2: **Llama-arch** (GQA, SwiGLU, RoPE, Llama tokenizer) — if GPT-2 token works

### Cell configuration (per architecture)

- Hidden=768, layers=8, heads=12, ffn=2048 (matched across architectures)
- 3 arms: scratch_ce, seq_kd_full, seq_kd_late_only
- 10 seeds per arm
- 3600 steps per cell, features at step 108 (3%)
- Trajectory losses logged at steps {10, 20, 40, 60, 80, 108, 200, 500}

### Cell count

- Qwen3: 3 arms × 10 seeds = 30 cells (use existing g180 train + new seeds)
- GPT-2: 3 arms × 10 seeds = 30 cells (genuinely new architecture)
- Total: 60 minimum, 90 with Llama phase 2

### Baselines to beat (ALL of these)

1. **Scalar early loss** — Ridge on early_loss_at_target_step only
2. **Early-loss trajectory** — Ridge on {loss@10, loss@20, loss@40, loss@60, loss@80, loss@108, slope, curvature}
3. **Gradient stats** — Ridge on gradient norm, grad noise ratio at step 108
4. **Arm/protocol labels** — Ridge on one-hot arm encoding only (tests if it's just arm identity)
5. **Shuffled geometry** — permute geometry features across rows, 1000 iterations

### Evaluation

- Leave-one-architecture-out cross-validation: train on Qwen3, test on GPT-2 (and vice versa)
- Block-bootstrap by seed AND architecture (addresses effective-n problem)
- Metrics: MSE, R², paired bootstrap CI, AUROC for bad-run detection
- Economic metric: simulated kill/continue budget allocation, compute savings

### Pass/fail criteria

- **PASS**: geometry beats ALL baselines on BOTH architecture folds with CI > 0
- **WEAK PASS**: geometry beats ALL baselines on one fold, CI > 0
- **FAIL**: any baseline ties or beats geometry on both folds

### Compute estimate

- 60 cells × 21 min/cell = 21h total
- With checkpointing and resume: ~5 sessions of 4h each
- VRAM: <5 GB per cell (small models), well within 22 GB envelope

### Implementation plan

- Reuse g180b infrastructure (tokenizer config, feature extraction, Ridge application)
- Add GPT-2 architecture config (transformers GPT2Config, not just GPT-2 tokenizer)
- Add trajectory feature extraction from logged intermediate losses
- Add leave-one-arch-out cross-validation to analysis

## Dependencies

- g180b completed (validates the infrastructure)
- g181b completed (informs whether anchor arm is worth including at 5000 steps)

## §0.1 scoring

- If PASS: **9.0/10** — first adversarially-validated cross-architecture geometry diagnostic
- If WEAK PASS: **7.0/10** — directional but not robust
- If FAIL: **4.0/10** — geometry diagnostic is dead, pivot to capability transfer applications
