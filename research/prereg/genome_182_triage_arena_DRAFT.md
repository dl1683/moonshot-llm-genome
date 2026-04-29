# genome_182 — Blinded Training Triage Arena (DRAFT v2)

**Status:** DRAFT v2 for Codex design gate. Not locked. Incorporates Codex design-gate feedback.

## Motivation

Adversarial A9 (cycle 75) flagged: g180's n=9 test set is effectively n=3 independent
samples, the early-loss-only baseline is too weak, and g180b tests tokenizer-perturbation
within Qwen3-shell, not true cross-architecture generalization.

Codex direction review (cycle 75) scored current trajectory at 5.0/10 and proposed the
"Blinded Training Triage Arena" as the §0.1=9.0 experiment.

## Hypothesis

Early-training geometry features (at ≤3% of training) predict final C4 NLL gain better
than ALL of: scalar early loss, early-loss trajectory (slope + curvature), gradient stats,
arm/protocol labels, combined telemetry baseline, and shuffled-geometry controls — across
≥2 genuinely distinct Transformer families.

## Design

### Architecture families (truly distinct, not tokenizer-swaps)

1. **Qwen3-arch** (GQA, SwiGLU, RoPE, Qwen3 tokenizer) — fresh cells, matched config
2. **GPT-2-arch** (MHA, GELU, learned pos emb, GPT-2 tokenizer) — genuinely different attention/FFN/pos-emb
3. Optional phase 2: **SSM/hybrid** (Mamba or Falcon-H1) — if phase 1 PASS, adds non-attention family for §0.1=9.0+

Note: GPT-2-arch is still Transformer-family, so a PASS is "cross-Transformer-family diagnostic." Full non-attention generality requires the phase-2 SSM.

### Arms (all treatments visible by step 108)

**Codex required:** `seq_kd_late_only` removed — its treatment starts at step 180, invisible at 3% checkpoint. Replaced with `embed_anchor` (visible immediately).

1. **scratch_ce** — train from scratch on C4 (baseline)
2. **seq_kd_full** — train on Qwen3 teacher text from step 0
3. **embed_anchor** — freeze-aligned embed/lm_head from Qwen3 donor with anchor loss (visible from step 0)

### Cell configuration (per architecture)

- Hidden=768, layers=8, heads=12, ffn=2048 (matched across architectures)
- 3 arms × 12 seeds = 36 cells per architecture (Codex: 10 seeds borderline → 12)
- 3600 steps per cell, features at step 108 (3%)
- Trajectory losses logged at steps {10, 20, 40, 60, 80, 108, 200, 500}

### Cell count

- Qwen3: 3 arms × 12 seeds = 36 cells (all fresh, no g180 row mixing)
- GPT-2: 3 arms × 12 seeds = 36 cells
- Total phase 1: **72 cells**
- Phase 2 SSM: +36 cells if phase 1 PASS

**Staged execution (Codex recommended):** Run 48 cells first (2 arch × 2 arms × 12 seeds, scratch_ce + seq_kd_full), analyze futility. Expand to 72 only if CI is promising. Do not claim PASS from 48-cell stage — only futility stop allowed.

### Labels (normalized)

**Codex required:** Normalized final C4 gain vs matched scratch within architecture/seed.

`label = (scratch_final_nll[arch,seed] - arm_final_nll[arch,seed,arm]) / scratch_final_nll[arch,seed]`

This makes gains comparable across architectures with different absolute NLL scales.

### Baselines to beat (ALL of these, plus combined)

1. **Scalar early loss** — Ridge on early_loss_at_target_step only
2. **Early-loss trajectory** — Ridge on {loss@10, loss@20, loss@40, loss@60, loss@80, loss@108, slope, curvature}
3. **Gradient stats** — Ridge on gradient norm, grad noise ratio at step 108
4. **Arm/protocol labels** — Ridge on one-hot arm encoding only (tests if it's just arm identity)
5. **Shuffled geometry** — permute geometry features across rows, 1000 iterations
6. **Combined telemetry** (Codex required) — Ridge on arm labels + trajectory + gradient stats + learning-curve extrapolation (the strongest possible non-geometry baseline)

### Evaluation

- Leave-one-architecture-out cross-validation: train on Qwen3, test on GPT-2 (and vice versa)
- Block-bootstrap by seed, preserving all arms per seed (Codex: not row-level CV)
- Ridge scaling/alpha selection: pre-locked on train folds only (Codex required)
- Metrics: MSE, R², paired bootstrap CI, AUROC for bad-run detection, simulated kill/continue

### Pass/fail criteria (Codex-specified, strict)

- **PASS**: both LOAO folds show geometry+telemetry beating best non-geometry baseline by ≥25% MSE reduction; paired seed-block bootstrap 95% CI lower bound > 0; held-out R² ≥ 0.20; shuffled-geometry permutation p ≤ 0.01; bad-run AUROC ≥ 0.75 and ≥ 0.05 above best baseline; simulated kill bottom 30% saves ≥ 20% compute while retaining ≥ 90% of final gain.
- **WEAK PASS**: both folds CI > 0, but MSE reduction is 10-25%, or only one fold clears ≥ 25%.
- **FAIL**: any best non-geometry combined baseline ties or beats geometry on both folds, CI crosses zero in both folds, or arm/protocol labels explain the signal.

### Compute estimate

- Phase 1: 72 cells × 21 min/cell = ~25h total
- Staged: 48 cells first (~17h), then +24 cells if promising (~8h)
- With checkpointing and resume: ~6 sessions of 4h each
- VRAM: <5 GB per cell (small models), well within 22 GB envelope

### Implementation plan

- Reuse g180b infrastructure (tokenizer config, feature extraction, Ridge application)
- Add GPT-2 architecture config (transformers GPT2Config, not just GPT-2 tokenizer)
- Add embed_anchor arm (reuse g181a/b anchor infrastructure)
- Add trajectory feature extraction from logged intermediate losses
- Add combined telemetry baseline to analysis
- Add leave-one-arch-out cross-validation to analysis
- Add normalized label computation

## Dependencies

- g180b completed (validates the cross-tokenizer infrastructure)
- g181b completed (informs whether embed_anchor arm is worth including at 5000 steps)

## §0.1 scoring (Codex-assessed)

- If PASS (phase 1 only): **8.6/10** — cross-Transformer-family geometry diagnostic with strict baselines
- If PASS + phase 2 SSM: **9.0/10** — first adversarially-validated cross-architecture geometry diagnostic
- If WEAK PASS: **6.5/10** — directional but not robust
- If FAIL: **4.0/10** — geometry diagnostic is dead, pivot to capability transfer applications
