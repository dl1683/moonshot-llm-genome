# genome_182 — Blinded Training Triage Arena (LOCKED)

**Status:** LOCKED 2026-04-29. Codex design gate APPROVED (v3). Minor advisories addressed: smoke-test includes interrupt/resume verification, Ridge alpha grid predeclared [0.01–1000], wall-clock gate corrected to 4.2h/12-cell batch.

## Motivation

Adversarial A9 (cycle 75) flagged: g180's n=9 test set is effectively n=3 independent
samples, the early-loss-only baseline is too weak, and g180b tests tokenizer-perturbation
within Qwen3-shell, not true cross-architecture generalization.

Codex direction review (cycle 75) scored current trajectory at 5.0/10 and proposed the
"Blinded Training Triage Arena" as the §0.1=9.0 experiment.

**Competitive context:** arxiv 2604.01025 (April 2026) demonstrates in-training LLM
performance prediction using internal representations on OLMo3-7B (AUROC >0.75). Our
distinctive angle: cross-architecture generalization + adversarial baseline suite +
pre-registered strict criteria + training triage focus (kill/continue decisions).

## Hypothesis

Early-training geometry features (at ≤3% of training) predict final C4 NLL gain better
than ALL of: scalar early loss, early-loss trajectory (slope + curvature), gradient stats,
arm/protocol labels, combined telemetry baseline, delayed-loss baselines, within-arm
residual, and shuffled-geometry controls — across ≥2 genuinely distinct Transformer families.

## Design

### Architecture families (truly distinct, not tokenizer-swaps)

1. **Qwen3-arch** (GQA, SwiGLU, RoPE, Qwen3 tokenizer) — fresh cells, matched config
2. **GPT-2-arch** (MHA, GELU, learned pos emb, GPT-2 tokenizer) — genuinely different attention/FFN/pos-emb
3. Optional phase 2: **SSM/hybrid** (Mamba or Falcon-H1) — if phase 1 PASS, adds non-attention family for §0.1=9.0+

Note: GPT-2-arch is still Transformer-family, so a PASS is "cross-Transformer-family diagnostic." Full non-attention generality requires the phase-2 SSM.

### Arms (all treatments visible by step 108)

1. **scratch_ce** — train from scratch on C4 (baseline)
2. **seq_kd_full** — train on Qwen3 teacher text from step 0
3. **embed_anchor** — freeze-aligned embed/lm_head from Qwen3 donor with anchor loss (visible from step 0)

### Cell configuration (per architecture)

- Hidden=768, layers=8, heads=12, ffn=2048 (matched across architectures)
- Qwen3-arch: ~90.5M params (GQA + SwiGLU); GPT-2-arch: ~83.5M params (MHA + GELU)
- 3 arms × 12 seeds = 36 cells per architecture
- 3600 steps per cell, features at step 108 (3%)
- Trajectory losses logged at steps {10, 20, 40, 60, 80, 108, 200, 500}

### Cell count

- Qwen3: 3 arms × 12 seeds = 36 cells (all fresh, no g180 row mixing)
- GPT-2: 3 arms × 12 seeds = 36 cells
- Total phase 1: **72 cells**
- Phase 2 SSM: +36 cells if phase 1 PASS

**Staged execution:** Run 48 cells first (2 arch × 2 arms × 12 seeds, scratch_ce + seq_kd_full), analyze futility. Expand to 72 only if CI is promising. Do not claim PASS from 48-cell stage — only futility stop allowed.

### Labels (normalized)

`label = (scratch_final_nll[arch,seed] - arm_final_nll[arch,seed,arm]) / scratch_final_nll[arch,seed]`

Gains comparable across architectures with different absolute NLL scales.

### Co-primary geometry models (Codex v2 required)

**Model A: Full geometry** — All g180 features including Qwen3-reference Procrustes/RSA features (`hidden_to_qwen_ref_*`, `embed_to_qwen_ref_*`, `lm_head_to_qwen_ref_*`, `*_reference_rows_used`).

**Model B: Reference-free geometry** — Drop all `*_to_qwen_ref_*` and `*_reference_rows_used` features. This model uses only architecture-agnostic spectral, ID, gradient, curvature, and norm features. This is the main anti-leakage guard: Qwen3-reference features could be informative for Qwen3-arch but meaningless for GPT-2-arch, biasing LOAO CV.

Both co-primary models must independently beat all baselines for PASS.

### Baselines to beat (ALL of these, plus combined)

1. **Scalar early loss** — Ridge on early_loss_at_target_step only
2. **Early-loss trajectory** — Ridge on {loss@10, loss@20, loss@40, loss@60, loss@80, loss@108, slope, curvature}
3. **Gradient stats** — Ridge on gradient norm, grad noise ratio at step 108
4. **Arm/protocol labels** — Ridge on one-hot arm encoding only
5. **Arm/protocol mean** — per-arm mean of training labels (no features, just protocol identity)
6. **Delayed-loss baselines** — Ridge on loss@200, loss@500 (post-feature-point losses still within short horizon)
7. **Within-arm residual** — Ridge on trajectory features conditioned on arm (tests if geometry adds beyond arm + trajectory)
8. **Shuffled geometry** — permute geometry features across rows, 1000 iterations
9. **Combined telemetry** — Ridge on arm labels + trajectory + gradient stats + delayed losses + learning-curve extrapolation (strongest possible non-geometry baseline)

### Evaluation

- Leave-one-architecture-out cross-validation: train on Qwen3, test on GPT-2 (and vice versa)
- Block-bootstrap by seed, preserving all arms per seed
- Ridge scaling/alpha selection: pre-locked on train folds only (5-fold CV within train fold for alpha); alpha grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
- Metrics: MSE, R², paired bootstrap CI, AUROC for bad-run detection, simulated kill/continue

### Pass/fail criteria (Codex-specified, strict)

- **PASS**: BOTH co-primary models on BOTH LOAO folds show geometry beating best non-geometry baseline by ≥25% MSE reduction; paired seed-block bootstrap 95% CI lower bound > 0; held-out R² ≥ 0.20; shuffled-geometry permutation p ≤ 0.01; bad-run AUROC ≥ 0.75 and ≥ 0.05 above best baseline; simulated kill bottom 30% saves ≥ 20% compute while retaining ≥ 90% of final gain.
- **WEAK PASS**: both folds CI > 0, but MSE reduction is 10-25%, or only one fold clears ≥ 25%, or only Model B (reference-free) passes while Model A fails.
- **FAIL**: any best non-geometry combined baseline ties or beats geometry on both folds, CI crosses zero in both folds, or arm/protocol labels explain the signal.

### COMPUTE.md §9 compliance checklist

- [x] **Max VRAM:** <5 GB per cell (83-90M param model + feature extraction), well within 22 GB
- [x] **Max RAM:** <8 GB per cell (tokenized data pools), well within 56 GB
- [x] **Disk footprint:** ~500 MB results + ~200 MB cache = <1 GB total
- [x] **Quantization:** BF16 training (models <100M, no quantization needed per CLAUDE.md rules)
- [x] **Per-session wall clock:** 48 cells × 21 min/cell = ~17h staged, broken into ≤4h sessions with per-cell checkpoint/resume
- [x] **Checkpoint/resume:** Incremental JSON writes after each cell; atomic file writes; resume from last completed cell
- [x] **Smoke test:** 1 cell per architecture × 20 steps; validates tokenization, training loop, feature extraction, Ridge prediction, AND interrupt/resume (kill after 10 steps, resume, verify no data loss); projected wall-clock gate at 4.2h per 12-cell batch (12 × 21 min)

### Compute estimate

- Phase 1: 72 cells × 21 min/cell = ~25h total
- Staged: 48 cells first (~17h), then +24 cells if promising (~8h)
- Per-session: ≤4h with per-cell resume (no lost compute on crash)
- VRAM: <5 GB per cell, well within 22 GB envelope

### Implementation plan

- Reuse g180b infrastructure (tokenizer config, feature extraction, Ridge application)
- Add GPT-2 architecture config (transformers GPT2Config/GPT2LMHeadModel)
- Add embed_anchor arm (reuse g181a/b anchor infrastructure)
- Add trajectory feature extraction from logged intermediate losses
- Add reference-free geometry model (drop Qwen3-ref features)
- Add all 9 baselines + combined to analysis
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
