# genome_186 - KD Dose-Response Delta Geometry Diagnostic

**Status:** LOCKED (2026-04-30). Smoke test passed; full run begins.

- Date: 2026-04-29
- Trigger: g182 failed all preregistered cross-architecture prediction tests, but the seed-matched pairwise delta analysis survived: delta geometry predicted delta final NLL with R2 = 0.518, corr = 0.720, n = 24.
- Purpose: create real within-arm label variance by sweeping KD dose, then test whether early geometry change predicts the final KD dose-response curve better than non-geometry baselines.

---

## Motivation

g182 killed the broad "early geometry predicts training health across architectures" claim. LOAO R2 was catastrophically negative, baselines failed or tied, and within-arm label variance was too small to support a robust residual predictor.

The one surviving signal was narrower and more causal: within a fixed architecture and seed, the geometry shift induced by the KD intervention appeared to predict whether that intervention helped or harmed final validation NLL.

g186 is the decisive follow-up:

> Does seed-matched early geometry change predict the dose-response curve of KD benefit/harm?

If PASS, the live claim becomes "early geometry of a causal training intervention predicts that intervention's final effect." If FAIL, the Forecast direction is retired or demoted to a minor empirical curiosity.

## Hypotheses

**H1 (dose-response geometry):** For a fixed architecture and seed, the early geometry delta from scratch to KD dose alpha predicts the final validation-NLL delta from scratch to the same KD dose. The geometry model beats alpha-only, early-loss-delta, telemetry-delta, and combined non-geometry baselines on held-out seeds.

**H0 (dose or telemetry explains it):** The apparent g182 pairwise signal was a lucky alpha=1.0 slice, an arm/dose identity effect, an early-loss effect, or architecture-specific telemetry. Geometry does not add held-out predictive value once alpha and non-geometry training signals are included.

## Design

### Architecture families

Use the same two g182 families unless the implementation gate finds a hard blocker:

1. **Qwen3-arch**: GQA, SwiGLU, RoPE, Qwen3 tokenizer.
2. **GPT-2-arch**: MHA, GELU, learned positional embeddings, GPT-2 tokenizer.

Cell sizes, optimizer, data split, token budget, feature-extraction step, and final-evaluation protocol should match g182 as closely as possible, except for the shortened per-cell runtime needed for the 60-cell dose sweep.

### KD dose arms

For each architecture and seed, run five alpha doses:

| Arm | KD alpha | Role |
|---|---:|---|
| scratch_ce | 0.0 | Seed-matched denominator |
| kd_weak | 0.3 | Low-dose response |
| kd_medium | 0.7 | Near-linear response check |
| kd_full | 1.0 | g182-like KD slice |
| kd_strong | 2.0 | High-dose / possible harm regime |

Preregistered KD loss form:

`loss = CE_on_corpus + alpha * KD_loss`

If the existing implementation uses a normalized weighted mixture instead of additive alpha, the code must log the exact formula and convert alpha labels to the effective KD/CE weight in the result JSON. The preregistered dose order remains `[0.0, 0.3, 0.7, 1.0, 2.0]`.

### Seeds and cells

- Seeds: `[0, 1, 2, 3, 4, 5]` for each architecture.
- Raw cells: 2 architectures x 6 seeds x 5 doses = 60 cells.
- Primary analysis rows: 2 architectures x 6 seeds x 4 nonzero KD doses = 48 seed-matched deltas.
- Scratch cells are newly run for this experiment, not borrowed from g182, so every delta has a same-run-protocol denominator.
- Expected runtime: about 90 seconds per raw cell, about 90 minutes total cell time; budget 2 hours including analysis and I/O.

## Measurements

### Primary target

For each architecture `a`, seed `s`, and nonzero dose `d`:

`delta_NLL(a,s,d) = final_NLL(alpha=0.0,a,s) - final_NLL(alpha=d,a,s)`

Positive `delta_NLL` means KD dose `d` helped relative to the seed-matched scratch run. Negative means KD harmed.

### Primary predictors

Compute geometry features at the early feature step for every cell. Then construct dose deltas:

`delta_geometry_j(a,s,d) = geometry_j(alpha=d,a,s) - geometry_j(alpha=0.0,a,s)`

The primary geometry feature set is the g182 reference-free / manifold-only subset, not Qwen3-reference features. Minimum expected features:

- spectral alpha
- participation ratio
- sqrt_pr_alpha or equivalent spectral interaction feature
- depth drift features
- intrinsic-dimension estimate
- kNN-10 clustering coefficient

If Shesha is available, Shesha deltas are a separate baseline, not part of the primary geometry model.

### Non-geometry baselines

All baselines use the same 48 delta rows and the same train/test folds:

1. **alpha-only:** Ridge or linear regression on scalar alpha, plus optional alpha^2 if predeclared in the implementation.
2. **delta_early_loss:** early loss at alpha=d minus early loss at alpha=0.0, plus early loss slope delta if available.
3. **delta_telemetry:** non-geometry training signals only: gradient norm delta, gradient noise delta, early validation NLL delta, loss slope/curvature delta, and wall-clock/token-throughput deltas if logged.
4. **delta_Shesha:** Shesha feature deltas only, if the dependency is installed and feature extraction succeeds for all cells.
5. **alpha_plus_arch:** alpha-only plus architecture indicator.
6. **arm_mean:** mean label per `(architecture, alpha)` group from the training fold.
7. **combined_non_geometry:** alpha-only + delta_early_loss + delta_telemetry.
8. **shuffled_geometry:** primary geometry deltas permuted within architecture 1000 times.

The primary geometry model must beat the strongest available non-geometry baseline, not just early loss.

## Evaluation

### Primary split: held-out seeds

Use leave-two-seeds-out cross-validation within each architecture:

- Each fold holds out the same two seed IDs for both architectures.
- Train on the remaining four seeds per architecture.
- Test on held-out seeds across both architectures and all four nonzero doses.
- Folds are `(0,1)`, `(2,3)`, and `(4,5)`.

This is the primary test because the claim is about predicting unseen random initializations, not memorizing seed idiosyncrasies.

### Secondary stress test: held-out doses

Run leave-one-dose-out evaluation across nonzero doses:

- Train on three nonzero alpha levels.
- Test on the held-out alpha level.
- Repeat for alpha in `{0.3, 0.7, 1.0, 2.0}`.

This is not the primary PASS gate because n is small, but it determines whether the geometry relationship is a smooth dose-response law or only interpolation around known doses.

### Tertiary stress test: held-out architecture

Run leave-one-architecture-out on the 48 delta rows. This is explicitly not required for PASS, because g182 already killed broad LOAO forecasting. If it works here, it upgrades the interpretation; if it fails, the primary claim can still be within-architecture causal sensitivity.

### Model fitting

- Use Ridge regression for all feature models.
- Standardize features using training-fold means/scales only.
- Select Ridge alpha on training folds only from `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`.
- Report R2, MSE, Pearson correlation, Spearman correlation, and signed calibration slope.
- Bootstrap confidence intervals by seed block with at least 2000 bootstrap resamples.

## PASS criteria

Primary PASS requires all of:

1. Held-out-seed geometry R2 >= 0.30 on the pooled 48 delta rows.
2. Geometry MSE is at least 20% lower than the best non-geometry baseline.
3. Seed-block bootstrap 95% CI for `MSE(best baseline) - MSE(geometry)` has lower bound > 0.
4. Shuffled-geometry permutation p <= 0.05.
5. Geometry beats alpha-only. If alpha-only ties or wins, the result is a dose curve, not a geometry result.
6. At least one architecture individually has held-out-seed R2 >= 0.25, and neither architecture has R2 < 0.

## WEAK PASS criteria

Classify as WEAK PASS if:

- pooled held-out-seed geometry R2 is in `[0.20, 0.30)`, and
- geometry beats best non-geometry baseline by at least 10% MSE, and
- geometry beats alpha-only, but
- the bootstrap CI touches zero or one architecture has near-zero R2.

WEAK PASS keeps the causal-intervention direction alive but does not promote it as a reliable diagnostic.

## FAIL criteria

Classify as FAIL if any of:

- held-out-seed geometry R2 < 0.20;
- best non-geometry baseline ties or beats geometry;
- alpha-only explains the curve as well as geometry;
- geometry works only for alpha=1.0 and fails the other doses;
- pooled PASS is driven by one architecture while the other architecture has negative R2;
- shuffled geometry is not rejected at p <= 0.05.

## Pre-registered diagnostics

### D1: Label variance check

Report standard deviation and range of `delta_NLL` overall, by architecture, and by dose. If pooled `std(delta_NLL) < 0.005` nats, the experiment is under-identified even if a model reports positive R2.

### D2: Dose monotonicity

Report mean `delta_NLL` by alpha and architecture. A monotonic curve is not required. A nonmonotonic or harmful high-dose regime is scientifically useful because it increases label variance and tests whether geometry tracks benefit versus harm.

### D3: Architecture dependence

Report separate fits for Qwen3-arch and GPT-2-arch. A PASS with strong architecture asymmetry is interpreted as "within-family causal sensitivity predictor," not universal training-health prediction.

### D4: Scratch denominator stability

For each architecture, report scratch final-NLL variance across seeds. If scratch variance dominates KD-dose variance, analyze normalized deltas:

`delta_NLL_norm = (final_NLL(alpha=0.0) - final_NLL(alpha=d)) / final_NLL(alpha=0.0)`

The unnormalized delta remains primary unless scratch variance causes a preregistered denominator-stability warning.

### D5: Alpha leakage

Fit a classifier/regressor from geometry deltas to alpha. If alpha is nearly perfectly decoded and alpha-only also performs well, the result is not geometry-specific. This diagnostic does not fail the run by itself; it conditions interpretation.

## COMPUTE.md compliance

- [x] **Max VRAM:** expected <6 GB per cell; hard ceiling 22 GB.
- [x] **Max system RAM:** expected <12 GB; hard ceiling 56 GB.
- [x] **Wall clock:** 60 cells x about 90 sec = about 1.5 h cell time; target <=2 h total and hard cap <=4 h.
- [x] **Disk footprint:** expected <1 GB for JSONL metrics, feature arrays, and plots; no large activation dumps unless compressed and explicitly logged.
- [x] **Quantization:** no quantization for small trained cells; teacher/model precision must be logged. If a pretrained teacher is loaded, use the same precision policy as g182 and record it.
- [x] **Save/resume:** write one result record per completed cell with atomic JSON/JSONL update. Resume must skip completed `(arch, seed, alpha)` cells.
- [x] **Windows/CUDA rules:** `num_workers=0`, `pin_memory=False`, `n_jobs=1` while CUDA is active, launch with `PYTHONUNBUFFERED=1` for long runs.
- [x] **Smoke test:** before full run, execute 2 architectures x 1 seed x 2 doses (`alpha=0.0`, `alpha=1.0`) for a tiny step count, verify feature extraction, delta construction, analysis script, and resume behavior.

## Implementation artifacts

Expected implementation artifacts:

- `code/genome_186_kd_dose_response.py`
- `results/genome_186_kd_dose_response.json`
- optional per-cell cache under `results/cache/genome_186/`
- final summary added to `experiments/ledger.jsonl`

Expected post-run documentation updates:

- `WIKI.md`
- `research/CLAIM_EVIDENCE_MAP.md`
- this prereg status line, changed from DRAFT to LOCKED only before launch and to PASS/WEAK_PASS/FAIL only after analysis.

## Interpretation

**PASS:** g182's surviving pairwise result generalizes into a causal dose-response diagnostic. The project score can move toward about 6.5/10: not universal geometry forecasting, but a real intervention-response signal.

**WEAK PASS:** The direction remains alive, but needs either more seeds or a cleaner dose schedule before public claims.

**FAIL:** The g182 pairwise delta was likely a lucky slice or alpha/telemetry artifact. Retire Forecast as a central direction and stop adding architectures before a new mechanism is found.

