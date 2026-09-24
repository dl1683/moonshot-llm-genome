# MC001G Gemma 2 2B Sparse Feature Discovery Status

Status: complete. Official Gemma Scope sparse-feature discovery did not promote.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_sparse_discovery.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_SPARSE_DISCOVERY.md`
- prior activation-patch status: `research/cards/MC001G_GEMMA2_2B_ACTIVATION_PATCH_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_gemma_scope_sparse_discovery_20260630T125743.json`
- result SHA256: `251bc8075a9ca74c763ecb0e59d81d321d47fd420d7d09c354cdf77837c5c4d5`
- sparse release: `gemma-scope-2b-pt-res-canonical`
- SAE IDs: `layer_14/width_16k/canonical`, `layer_20/width_16k/canonical`
- model: `google/gemma-2-2b`
- render mode: `raw`
- matched rows: 48 total, 34 discovery, 14 holdout
- label-shuffle nulls: 50 per layer

## Loader Correction

A direct `params.npz` encoding smoke was attempted first and produced invalid
activation sanity: layer 14 mean L0 was about 295 and reconstruction cosine was
about 0.60. That result is not evidence.

The official SAELens loader fixed the activation surface:

| Layer | Mean L0 | Mean Reconstruction Cosine | Relative MSE |
| --- | ---: | ---: | ---: |
| 14 | 86.77 | 0.907 | 0.178 |
| 20 | 52.17 | 0.935 | 0.125 |

This status uses only the official SAELens run.

## Question

Do Gemma Scope residual SAE features expose a sparse truth-versus-agreement
signature on the matched MC001G rows that is cleaner than the dense matched
signature and reliability baselines?

## Gate Verdict

No.

The SAE features are validly applied, but the sparse signal does not beat the
prior dense matched direction and is not clean against label-shuffle nulls.

The strongest layer 14 discovery-selected feature reached 0.714 holdout AUC,
below the prior dense layer 14 direction at 0.755. A fixed top-8 sparse
combination reached only 0.694 holdout AUC. Label-shuffle nulls found rank-1
features with holdout AUC up to 0.816.

Layer 20 was weaker: the rank-1 feature reached 0.633 holdout AUC and the top-8
sparse combination reached 0.673.

## Baselines

| Baseline | Discovery AUC | Holdout AUC | Interpretation |
| --- | ---: | ---: | --- |
| same-row margin | 0.990 | 1.000 | tautological behavior readout |
| no-hint margin | 0.500 | 0.418 | matched out |
| no-hint margin bin | 0.500 | 0.500 | matched out |
| correct-hint margin | 0.770 | 0.663 | below dense signature |
| correct letter | 0.768 | 0.704 | comparable to sparse rank-1 |
| wrong letter | 0.689 | 0.622 | below sparse rank-1 |
| condition + correct/wrong letters | 0.841 | 0.653 | below sparse rank-1 |
| no-hint bin + condition + letters | 0.917 | 0.673 | below sparse rank-1 |

The prior matched dense direction reached 0.755 holdout AUC. The prior dense
logistic probe reached 0.939 holdout AUC but was not used as an intervention
surface.

## Sparse Results

| Layer | Rank-1 Feature | Discovery AUC | Holdout AUC | Top-8 Holdout AUC | Null Max | Null p>=Observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 9727 | 0.824 | 0.714 | 0.694 | 0.816 | 0.16 |
| 20 | 13487 | 0.716 | 0.633 | 0.673 | 0.796 | 0.26 |

Layer 14 feature 9727 is agreement-associated under the discovery orientation:
its raw activation is higher on user-agreement-error rows than truth-following
rows in both discovery and holdout. It is active on all 48 matched rows, so it
is not a clean sparse on/off feature for this behavior.

## Reliability Notes

- Holdout remains small: 14 matched rows.
- The sparse features do not beat the simple dense direction.
- Correct-letter baseline holdout AUC, 0.704, is close to the layer 14 rank-1
  sparse feature at 0.714.
- Shuffle nulls can find higher holdout AUC than the observed rank-1 layer 14
  feature, so the result is not null-clean.
- The direct NPZ path is unsafe unless it reproduces SAELens L0 and
  reconstruction checks.
- This is still forced-choice logit scoring, not free generation.

## Decision

Do not promote to a sparse feature-level intervention.

Do not claim a Gemma Scope sparse mechanism for MC001G.

The current supported MC001G result is:

> behavior substrate repaired, matched dense signature found, additive dense
> steering failed, activation replacement failed controls, and canonical Gemma
> Scope sparse features failed promotion against dense and null baselines.

Next useful branches:

1. expand the repaired/matched item bank to produce a larger exact-bin holdout;
2. only then rerun sparse discovery and feature-level interventions;
3. treat final-token dense and sparse signatures as diagnostic until an
   intervention passes donor-label, locality, wrong-layer, and null controls.
