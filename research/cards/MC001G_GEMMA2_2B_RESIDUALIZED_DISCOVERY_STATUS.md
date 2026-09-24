# MC001G Gemma 2 2B Residualized Dense Discovery Status

Status: complete. Residualized dense discovery did not justify intervention.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_residual_discovery.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY.md`
- prior dense status: `research/cards/MC001G_GEMMA2_2B_DISCOVERY_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_residualized_dense_signature_20260630T122023.json`
- result SHA256: `e55480ff1631f3b483b06abe764bc503d2b23f6e4addbb1df67620cd1baedd1d`
- model: `google/gemma-2-2b`
- render mode: `raw`
- selected binary rows: 83
- clean items: 42
- elapsed: 11.8 seconds

## Question

Does the MC001G dense final-token signature survive after output margin, prompt
condition, correct answer letter, wrong answer letter, and baseline answer
letter are treated as nuisance variables?

## Gate Verdict

No.

Residualization weakens the dense signal, and the residualized signals do not
beat the nuisance-feature baseline. This remains a diagnostic signature result,
not an intervention-ready mechanism.

Write this as:

> MC001G has a repaired behavior substrate and measurable dense signatures, but
> the present dense final-token signal is still margin/nuisance dominated.

## Nuisance Features

The nuisance regression was fit on discovery rows only, then applied to discovery
and holdout rows.

Features:

- intercept;
- standardized correct-minus-wrong option margin;
- prompt condition;
- correct answer letter;
- wrong answer letter;
- baseline answer letter.

## Baselines

| Baseline | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| margin only | 0.997 | 1.000 |
| nuisance feature bundle | 0.992 | 0.905 |
| condition only | 0.517 | 0.345 |
| correct letter | 0.670 | 0.821 |
| wrong letter | 0.520 | 0.369 |
| condition + correct/wrong letters | 0.682 | 0.742 |
| baseline answer letter | 0.685 | 0.262 |

The nuisance bundle remains stronger than any residualized dense result on
holdout.

## Residualized Hidden-State Signals

Top residualized dense directions:

| Layer | Residual Discovery AUC | Residual Holdout AUC | Raw Direction Holdout AUC | Residual Logistic Holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| 5 | 0.602 | 0.833 | 0.817 | 0.397 |
| 4 | 0.608 | 0.825 | 0.786 | 0.317 |
| 7 | 0.631 | 0.825 | 0.778 | 0.381 |
| 8 | 0.636 | 0.825 | 0.746 | 0.508 |
| 6 | 0.650 | 0.817 | 0.778 | 0.437 |
| 3 | 0.598 | 0.817 | 0.738 | 0.333 |

Top residualized dense logistic probes:

| Layer | Residual Discovery AUC | Residual Holdout AUC | Residual Direction Holdout AUC |
| --- | ---: | ---: | ---: |
| 9 | 0.840 | 0.690 | 0.746 |
| 0 | 0.844 | 0.587 | 0.690 |
| 10 | 0.841 | 0.579 | 0.746 |
| 11 | 0.842 | 0.563 | 0.770 |
| 12 | 0.842 | 0.563 | 0.770 |
| 14 | 0.842 | 0.540 | 0.802 |

The best residualized direction has a high holdout AUC but weak discovery AUC.
The best residualized logistic probe has only 0.690 holdout AUC. Neither is
stronger than the nuisance baseline.

## Margin-Overlap Audit

The repaired rows are nearly separable by option margin:

| Label | N | Min | Q25 | Median | Q75 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| truth-following | 40 | 0.000 | 0.375 | 0.625 | 1.250 | 2.500 |
| user-agreement error | 43 | -2.125 | -1.125 | -0.625 | -0.250 | 0.000 |

The overlap range is only exactly `0.000`.

Matched overlap counts:

- truth rows in overlap: 3;
- agreement rows in overlap: 3;
- matched pairs with 0.25, 0.5, or 1.0 margin bins: 3.

That is too small for a margin-matched mechanism test. The current repaired
manifest is good for finding the behavior and diagnosing margin dominance, but
not for proving a margin-independent control surface.

## Decision

Do not run dense intervention.

Do not treat the residualized direction as a mechanism.

The next MC001G step should create a margin-overlap repair gate:

1. add softer wrong-hint pressure conditions between `wrong_disclaimed` and `wrong_unsure`;
2. avoid saturated `wrong_direct` and `wrong_high` rows for the primary signature label;
3. target at least 24 truth and 24 agreement rows inside shared margin bins;
4. rerun dense and sparse discovery only after the margin-overlap gate passes.

Sparse Gemma Scope work remains useful, but it must use the same margin and
answer-letter controls. Sparse features that only recapitulate option margin do
not count.
