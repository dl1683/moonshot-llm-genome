# Pre-registration: g197 Output-Interface Canary Arena

**Status:** DRAFT. Motivated by g195 `PASS_OUTPUT_DOMINANT`: the lm_head carries about 65% of the tied interface signal and is the output-gradient generator.

## Motivation

The live section 0.1 goal is: earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed. g195 moved the active surface from input embeddings to the output classifier. This experiment asks whether the geometry of the lm_head at step 0 or step 50 forecasts final 5000-step NLL better than early loss alone.

## Hypothesis

**H1 Canary:** step-0/50 lm_head geometry predicts final validation NLL across deliberately varied output-interface initializations, beating a scalar step-50 early-loss baseline.

**H0 Loss-only:** early loss already captures the useful signal; lm_head geometry adds no predictive value.

## Protocol

- Architecture: 8-layer Qwen3 shell, GPT-2 tokenizer, same training/eval setup as g191-g196.
- Tying: `tie_word_embeddings=False`.
- Intervention: lm_head initialization only. Input embeddings and all decoder weights use normal scratch init.
- Anchors: none. All lm_head rows are trainable under normal CE after initialization.
- Train: 5000 steps. Record final C4 validation NLL.
- Seeds: `[42, 7, 13]`.
- Cells: 10 conditions x 3 seeds = 30 cells.
- Feature snapshots: step 0 before optimizer update; step 50 after normal training.

## lm_head Conditions

All heads are Frobenius-rescaled to the scratch lm_head norm. Trained-row conditions use exact GPT-2/Qwen3 string-matched rows; unmatched rows are random Gaussian rows rescaled to the matched-row norm distribution.

| Condition | Expected regime | Construction |
|---|---|---|
| `trained_qwen3` | healthy | exact-string trained Qwen3 lm_head rows |
| `frequency_scaled` | healthy/wasteful | trained unit directions, norms set by GPT-2 C4 unigram frequency |
| `orthogonal_scaffold` | wasteful | `trained_qwen3 @ Q`, fixed random orthogonal `Q`; preserves row-row angles, destroys decoder coordinates |
| `covariance_scaffold` | wasteful | random rows sampled from trained lm_head mean/covariance |
| `identity_axis` | wasteful | high-frequency tokens assigned repeated signed coordinate axes |
| `neural_collapse_etf` | wasteful/healthy test | equal-norm simplex/ETF prototype codebook with deterministic jitter |
| `random_gaussian` | neutral baseline | standard lm_head random init |
| `trained_random_directions` | doomed | trained row norms with random unit directions |
| `trained_shuffled` | doomed | trained rows randomly permuted across token ids |
| `anti_frequency_scaled` | doomed | inverse-frequency norm schedule with random unit directions |

The regime labels are priors only. Final NLL defines the observed health ordering.

## Geometry Features

Features are extracted on the full sampled head and on matched rows only. Row sample is fixed before training: top 4096 frequent rows plus 4096 deterministic random rows.

Primary geometry-only feature set:

- Spectral: tail alpha, stable rank, centered participation ratio, effective-rank entropy, top-singular-value mass, condition number, `sqrt(PR) * alpha`.
- Row norms: mean/std/CV/skew, norm entropy, Gini, frequent/rare norm ratio, Spearman norm-vs-frequency.
- Angular: mean/std pairwise cosine, max coherence, Gram off-diagonal Frobenius, angular spread, nearest-neighbor cosine mean/p95.
- Local graph: kNN-10 clustering, mutual-nearest-neighbor fraction, frequency-bucket neighbor purity.
- Reference distance: Procrustes residual and RSA distance to trained Qwen3 lm_head, spectral Wasserstein distance to trained Qwen3.
- Scaffold distances: ETF distance, identity-axis distance, covariance distance, row-norm KL to trained Qwen3.
- Dynamics: each feature at step 0, step 50, and delta `step50 - step0`; lm_head update Frobenius norm and cosine from initial head.

Early loss is excluded from the primary geometry model.

## Prediction Model

Target: absolute final validation NLL at step 5000.

Model: Ridge regression. Features are z-scored using train-fold statistics only. Alpha grid is `[0.01, 0.1, 1, 10, 100, 1000]`, selected inside each train fold.

Evaluation: leave-one-condition-out CV. Each fold trains on 9 conditions / 27 cells and tests on the held-out condition / 3 seeds. Condition labels are never model inputs.

Baseline: Ridge on scalar step-50 validation NLL only, using the same leave-one-condition-out folds and alpha grid. Secondary diagnostic baseline: step-10/25/50 loss slope, reported but not the locked comparator.

Statistics: paired condition-block bootstrap over the 10 held-out-condition folds, preserving the three seeds per condition. Shuffled-geometry control permutes geometry feature rows within seed index 1000 times.

## Pass/Fail Criteria

**PASS_CANARY** requires all:

1. Geometry-only Ridge reduces leave-one-condition-out MSE by at least 25% vs scalar step-50 loss.
2. Bootstrap 95% CI lower bound on MSE reduction is greater than 0.
3. Held-out R2 is at least 0.35.
4. Shuffled-geometry permutation p <= 0.05.
5. Geometry beats early loss in at least 8 of 10 held-out conditions by absolute error.

**WEAK_PASS**: MSE reduction is 10-25%, CI lower bound is >= 0, and shuffled p <= 0.10.

**FAIL_LOSS_ONLY**: scalar step-50 loss ties or beats geometry, or geometry improvement CI crosses 0.

**FAIL_NO_SIGNAL**: final NLL range across conditions is <0.10 nats, meaning the arena failed to create health regimes.

## Sample Size

Primary design is 3 seeds x 10 conditions = 30 cells. This is the largest clean design inside the 4-hour envelope using current g195/g196 timing. Do not expand to 12 conditions unless smoke timing is <=6.3 min/cell or the run is split into a separate checkpointed second session; any 12-condition expansion is exploratory unless locked in a new prereg.

## Compute Envelope

- Max VRAM: expected <5 GB per cell, below 22 GB.
- Max RAM: expected <8 GB, below 56 GB.
- Runtime: 30 cells x about 7.3 min = about 3.65 h, plus small feature overhead.
- Windows/CUDA: `num_workers=0`, `pin_memory=False`, sklearn `n_jobs=1`.
- Quantization: none; BF16/FP32 mixed training as in g195/g196.
- Disk: <2 GB JSON/features/checkpoints.
- Save/resume: append one JSON result per completed cell; feature cache after step 0 and step 50; atomic writes.
- Smoke: two conditions (`trained_qwen3`, `trained_shuffled`) x seed 42 x 50 steps must verify feature extraction and resume before full launch.

## Section 0.1 Score

If `PASS_CANARY`: section 0.1 moves from 5.8 to about 6.9/10. The claim becomes: earliest output-interface geometry forecasts downstream training health beyond early loss.

If `WEAK_PASS`: section 0.1 moves to about 6.2/10; useful diagnostic lead, not a breakthrough.

If `FAIL_LOSS_ONLY` or `FAIL_NO_SIGNAL`: section 0.1 drops/caps near 5.4/10. g195 remains a mechanistic output-dominance result, but the forecast headline is not supported by lm_head geometry.
