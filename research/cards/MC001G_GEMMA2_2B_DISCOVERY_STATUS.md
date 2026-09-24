# MC001G Gemma 2 2B Dense Signature Discovery Status

Status: complete. Dense signature found; mechanism promotion failed against margin.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_repair_discovery.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_DISCOVERY.md`
- repair status: `research/cards/MC001G_GEMMA2_2B_REPAIR_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_dense_signature_20260630T121520.json`
- result SHA256: `633de4c66199ae34e763e15333bf84eaae9a6059796735eead421aaaea510c32`
- model: `google/gemma-2-2b`
- render mode: `raw`
- primary conditions: `wrong_disclaimed`, `wrong_unsure`
- clean items: 42
- selected binary rows: 83
- elapsed: 12.9 seconds

## Question

Do final-token dense residual states on repaired base Gemma 2 2B contain a
measurable signature of truth-following versus user-agreement on intermediate
wrong-hint pressure rows?

## Gate Verdict

Yes for a signature.

No for mechanism promotion or intervention design.

The hidden states contain a real signal, but the correct-minus-wrong option logit
margin is a stronger and cleaner baseline. Margin-only AUC is 1.000 on holdout.

Write this as:

> MC001G now has a repaired behavior substrate and a measurable dense hidden
> signature, but the current dense final-token signature does not beat the
> output-margin explanation.

## Dataset

Primary rows:

- only clean items from the repair gate;
- only `wrong_disclaimed` and `wrong_unsure` conditions;
- only rows labeled `truth_following` or `user_agreement_error`;
- direct/high pressure rows excluded because they are nearly saturated.

Clean items by correct answer letter:

| Correct Letter | Clean Items |
| --- | ---: |
| `A` | 16 |
| `B` | 10 |
| `C` | 8 |
| `D` | 8 |

Selected label counts:

| Label | Rows |
| --- | ---: |
| truth-following | 40 |
| user-agreement error | 43 |

Split label counts:

| Split | Truth | Agreement |
| --- | ---: | ---: |
| discovery | 31 | 29 |
| holdout | 9 | 14 |

## Baselines

| Baseline | Discovery AUC | Holdout AUC | Verdict |
| --- | ---: | ---: | --- |
| correct-minus-wrong option margin | 0.997 | 1.000 | dominates |
| prompt condition | 0.517 | 0.345 | weak |
| correct answer letter | 0.670 | 0.821 | nontrivial split artifact |
| wrong answer letter | 0.520 | 0.369 | weak |
| condition + correct/wrong letters | 0.682 | 0.742 | below margin |
| baseline answer letter | 0.685 | 0.262 | weak by itself |

The margin baseline alone separates truth-following from user-agreement on the
holdout split. That blocks any dense-signature mechanism claim.

## Hidden-State Signals

Top truth-minus-agreement dense directions:

| Layer | Discovery AUC | Holdout AUC | Logistic Holdout AUC |
| --- | ---: | ---: | ---: |
| 16 | 0.939 | 0.857 | 0.921 |
| 0 | 0.762 | 0.825 | 0.833 |
| 5 | 0.754 | 0.817 | 0.849 |
| 13 | 0.800 | 0.802 | 0.841 |
| 12 | 0.791 | 0.786 | 0.833 |

Top dense logistic probes:

| Layer | Discovery AUC | Holdout AUC | Direction Holdout AUC |
| --- | ---: | ---: | ---: |
| 19 | 1.000 | 0.976 | 0.770 |
| 20 | 1.000 | 0.944 | 0.762 |
| 16 | 1.000 | 0.921 | 0.857 |
| 17 | 1.000 | 0.921 | 0.778 |
| 21 | 1.000 | 0.905 | 0.778 |

The hidden signal is real enough to matter diagnostically. It is not yet an
independent mechanism because the margin baseline is perfect.

## Failure Modes

- The holdout set is small: 23 rows.
- Correct answer letter has nontrivial holdout AUC at 0.821.
- Dense logistic probes have discovery AUC 1.000 in late layers, so overfitting is plausible even when holdout AUC is high.
- The best simple direction is weaker than margin: layer 16 direction holdout AUC 0.857 versus margin 1.000.
- Direct and high-pressure rows remain saturated and should not be used as the primary signature label source.

## Decision

Do not run a dense steering intervention from this signature.

The next useful MC001G branch is one of:

1. residualize dense and sparse features against output margin and answer-letter features;
2. run Gemma Scope sparse-feature discovery only after preserving the same margin and letter baselines;
3. build a margin-matched subset where truth-following and user-agreement are not linearly separable by correct-minus-wrong option margin.

Any future intervention must first show a signature that survives the margin
baseline or is explicitly conditioned on a matched margin bin.
