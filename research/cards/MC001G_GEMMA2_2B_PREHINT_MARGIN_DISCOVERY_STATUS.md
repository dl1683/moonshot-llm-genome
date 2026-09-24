# MC001G Gemma 2 2B Pre-Hint Margin Matched Discovery Status

Status: complete. Matched signature found; intervention design is now justified with caveats.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_prehint_margin_discovery.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY.md`
- prior residualized status: `research/cards/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_pre_hint_margin_matched_20260630T122717.json`
- model: `google/gemma-2-2b`
- render mode: `raw`
- bin width: `0.5`
- selected rows: 48
- elapsed: 9.3 seconds

## Question

Can MC001G produce a dense hidden-state signature after matching truth-following
and user-agreement rows on a non-tautological pre-hint margin?

## Correction

Same-row correct-minus-wrong option margin cannot be the matching variable under
forced-choice argmax labels. Truth-following rows necessarily have nonnegative
correct-minus-wrong margin, and user-agreement rows necessarily have nonpositive
correct-minus-wrong margin.

This pass therefore matched on item-level `no_hint` margin, measured before the
wrong hint appears.

## Gate Verdict

Yes for a matched signature.

No for a mechanism card yet.

The matched hidden signature beats the non-tautological pre-hint margin and
answer-letter baselines on holdout. Same-row margin remains perfect, but it is a
tautological behavior readout rather than an independent mechanism baseline.

## Matched Dataset

Rows were matched within split and no-hint-margin bin.

| Split | Rows | Truth | Agreement |
| --- | ---: | ---: | ---: |
| discovery | 34 | 17 | 17 |
| holdout | 14 | 7 | 7 |
| total | 48 | 24 | 24 |

Discovery bins:

| No-Hint Margin Bin | Truth | Agreement |
| --- | ---: | ---: |
| 0 | 2 | 2 |
| 1 | 6 | 6 |
| 2 | 6 | 6 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

Holdout bins:

| No-Hint Margin Bin | Truth | Agreement |
| --- | ---: | ---: |
| 0 | 2 | 2 |
| 2 | 1 | 1 |
| 3 | 4 | 4 |

## Baselines

| Baseline | Discovery AUC | Holdout AUC | Interpretation |
| --- | ---: | ---: | --- |
| same-row margin | 0.990 | 1.000 | tautological behavior readout |
| no-hint margin | 0.500 | 0.418 | matched out |
| correct-hint margin | 0.770 | 0.663 | below hidden signal |
| no-hint margin bin | 0.500 | 0.500 | matched out |
| condition | 0.529 | 0.286 | weak |
| correct letter | 0.768 | 0.704 | nontrivial but below hidden signal |
| wrong letter | 0.689 | 0.622 | below hidden signal |
| condition + correct/wrong letters | 0.841 | 0.653 | below hidden signal |
| no-hint bin + condition + letters | 0.917 | 0.673 | below hidden signal |

## Hidden-State Signals

Best truth-minus-agreement direction:

| Layer | Discovery AUC | Holdout AUC | Direction Norm |
| --- | ---: | ---: | ---: |
| 14 | 0.913 | 0.755 | 7.465 |

Best dense logistic probe:

| Layer | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| 20 | 1.000 | 0.939 |

This is the first MC001G signature result that survives the non-tautological
pre-hint margin control.

## Caveats

- Holdout is small: 14 rows.
- Same-row margin remains perfect and must be reported in every future result.
- Dense logistic probes can overfit, so an intervention should prefer simple
  direction steering first.
- This is still a forced-choice logit setup, not free generation.
- The direction should be tested with sign-flip, random-direction, wrong-layer,
  and no-hint/correct-hint locality controls.

## Decision

Proceed to a small preregistered intervention test.

The first intervention should use the layer 14 truth-minus-agreement direction
from the matched discovery split and evaluate:

- matched holdout wrong-hint rows;
- no-hint and correct-hint locality rows for the same holdout items;
- sign-flip control;
- random matched-norm control;
- nearby-layer or wrong-layer control;
- answer distribution and margin movement.

Passing this intervention would not finish MC001G, but it would move from
signature-only to a genuine intervention gate.
