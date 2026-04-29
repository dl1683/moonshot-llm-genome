# Pre-registration: genome_184 Falcon-H1 Frozen Geometry Generalization

**STATUS:** DRAFT pre-staged 2026-04-29 cycle 100. **LOCKS** after g182 stage 1 analysis completes and Model C'/C Ridge is trained. Gates on g182 not being futility-stopped.

- Date: 2026-04-29
- Trigger: A15 adversarial (cycle 100) — tokenizer-family recognition risk + Model C impurity resolved by frozen no-refit test on third architecture family

---

## Hypothesis

**H1 (geometry generalizes to hybrid-SSM family):** A Ridge model trained on g182 Qwen3+GPT-2 cells using MANIFOLD_ONLY features (8 features: spectral alpha, participation ratio, sqrt_pr_alpha, 3 depth drifts, TwoNN intrinsic dim, kNN-10 clustering coeff) predicts training outcome on held-out Falcon-H1-0.5B cells WITHOUT refitting, beating arm_mean and combined_telemetry baselines.

**H0 (geometry is tokenizer/family-specific):** Frozen Ridge fails on Falcon-H1 — geometry features encode Qwen3/GPT-2-specific structure, not universal training health signals.

## Universality level claim

Level-1 cross-architecture-family generalization (if PASS: 3 families tested — Qwen3, GPT-2, Falcon-H1).

## Systems tested

- **Falcon-H1-0.5B** (tiiuae/Falcon-H1-0.5B-Base): hybrid attention+SSM, 1024d/36L, native tokenizer, ~500M params. Verified Windows-compatible (cycle 94). Cell models: from-scratch 768d/8L (~157M params, matches g182 cell dimensions). Teacher for seq_kd: the full pre-trained 0.5B model.

## Protocol

### Phase 1: Train frozen Ridge on g182 cells

1. Load g182 result JSON (cells with features + normalized labels)
2. Exclude scratch cells (per g182 protocol)
3. Train Ridge on ALL g182 cells (both architectures) using MANIFOLD_ONLY_FEATURE_NAMES (8 features)
4. Save trained Ridge coefficients + alpha + scaler parameters to frozen artifact
5. Record train-set MSE, R2, feature weights for audit

### Phase 2: Run Falcon-H1-0.5B cells

- Arms: scratch_ce, seq_kd_full (teacher = pre-trained tiiuae/Falcon-H1-0.5B-Base from HuggingFace; native-tokenizer teacher to avoid cross-tokenizer KD artifacts)
- NOTE: embed_anchor arm EXCLUDED — no Falcon-H1 donor available for embed anchoring. 2 arms x 12 seeds = 24 cells.
- Seeds: 12 (same seed set as g182: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 — matches SEEDS=list(range(12)) in g182 code)
- Training: identical protocol to g182 (3600 steps, AdamW, same LR/warmup/batch, feature extraction at step 108 = 3%)
- Feature extraction: same 8 manifold features from mid-depth hidden states
- Total cells: 24 (2 arms x 12 seeds)
- Labels: normalized fractional gain vs own-seed scratch (same as g182)

### Phase 3: Frozen evaluation

1. Apply frozen Ridge (from Phase 1) to Falcon-H1 cells — NO refitting
2. Compute MSE, R2 against actual labels
3. Compute baseline MSEs: arm_mean (from Falcon cells), frozen Model D (6 pure-telemetry features from g182, no refit — apples-to-apples frozen comparison)
4. Seed-block bootstrap (B=2000, blocks of 4 seeds) for 95% CI on MSE reduction vs best baseline
5. Shuffled-feature permutation test (1000 iterations) for p-value

### Phase 4: Comparison models (exploratory, not gating)

- Model C (10 features, includes norm/var): frozen from g182, no refit
- Model D (6 pure telemetry): frozen from g182, no refit
- Model E (3 Shesha features): if shesha-geometry installed, compute on Falcon cells

## PASS criteria (all must hold)

1. Frozen C' (manifold-only) MSE < best_baseline_MSE (arm_mean OR frozen Model D telemetry, whichever is lower)
2. MSE reduction vs best baseline >= 15% (relaxed from g182's 25% since this is zero-shot generalization)
3. Bootstrap 95% CI on MSE(best_baseline) - MSE(frozen_C') excludes zero (geometry is significantly better)
4. Permutation p <= 0.05

## FAIL criteria

- Frozen C' ties or loses to any strong baseline on Falcon-H1 → geometry is family-specific
- R2 < 0 → geometry features are anti-predictive on new family

## Compute envelope (COMPUTE.md section 9)

- [x] Peak VRAM <= 22 GB (Falcon-H1-0.5B ~2 GB; teacher gen ~4 GB; training ~6 GB)
- [x] System RAM <= 56 GB
- [x] Wall-clock <= 4h (24 cells x ~15 min = ~6h; requires 2 batches or overnight run)
- [x] No cloud compute required
- NOTE: wall-clock may exceed 4h single-shot; use --max-cells=12 in two batches with checkpoint resume

## What a null result means

If H0 holds: geometry features are architecture-family-specific embeddings of tokenizer/interface structure, not universal training health signals. The g182 LOAO between Qwen3 and GPT-2 would be "nearby English-decoder-Transformer transfer," not genuine cross-architecture generalization. The project's §0.1 score drops to 6.5-7.0 (Model C/C' passes on 2 related architectures only).

## Connection to A15 resolver

This is the EXACT experiment specified by cycle 100 adversarial as the resolver for the strongest attack (S10: tokenizer/interface-family recognition). A clean PASS here plus g182 PASS would give 3-family geometry generalization — the strongest possible evidence for the training-triage diagnostic claim.
