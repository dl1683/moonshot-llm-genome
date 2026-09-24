# MC001G Gemma 2 2B Matched Dense Intervention Status

Status: complete. First matched dense intervention failed.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_intervention.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_INTERVENTION.md`
- matched signature status: `research/cards/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_layer14_intervention_20260630T123128.json`
- result SHA256: `62f1cb028ecdb898176f93114b5955f3416e7599c90e50165f013d9c0e2fb1a7`
- model: `google/gemma-2-2b`
- layer: `14`
- wrong-layer control: `13`
- alphas: `0.25`, `0.5`, `1.0`
- direction norm: `7.465`
- evaluation rows: 14 matched holdout rows, 7 no-hint locality rows, 7 correct-hint locality rows

## Question

Does adding the layer 14 truth-minus-agreement direction change matched holdout
wrong-hint behavior in the predicted direction?

## Gate Verdict

No.

The positive layer 14 intervention did not improve matched-holdout
truth-following at any tested dose. Controls matched or beat it.

Write this as:

> MC001G has a repaired behavior substrate and a matched dense signature, but the
> first simple dense intervention did not work.

## Matched Holdout Results

| Arm | Matched Truth | Matched Agreement | Mean Margin | Verdict |
| --- | ---: | ---: | ---: | --- |
| baseline | 7/14 | 7/14 | 0.134 | baseline |
| layer 14 alpha 0.25 | 7/14 | 7/14 | 0.098 | no effect |
| layer 14 alpha 0.5 | 7/14 | 7/14 | 0.116 | no effect |
| layer 14 alpha 1.0 | 7/14 | 7/14 | 0.125 | no effect |
| layer 14 sign-flip alpha 1.0 | 8/14 | 6/14 | 0.134 | control beats intended direction |
| layer 14 random alpha 1.0 | 7/14 | 7/14 | 0.116 | no effect |
| wrong-layer 13 alpha 1.0 | 8/14 | 6/14 | 0.089 | control beats intended layer |

The intended direction fails the intervention gate.

## Locality

| Arm | No-Hint Locality | Correct-Hint Locality |
| --- | ---: | ---: |
| baseline | 7/7 truth | 7/7 truth |
| layer 14 alpha 0.25 | 7/7 truth | 7/7 truth |
| layer 14 alpha 0.5 | 7/7 truth | 7/7 truth |
| layer 14 alpha 1.0 | 6/7 truth, 1/7 other | 7/7 truth |
| sign-flip alpha 1.0 | 7/7 truth | 7/7 truth |
| random alpha 1.0 | 7/7 truth | 7/7 truth |
| wrong-layer 13 alpha 1.0 | 7/7 truth | 7/7 truth |

The highest intended dose causes one no-hint locality error without improving
matched holdout behavior.

## Reliability Notes

- Holdout is small, but the result is not ambiguous in the intended direction:
  positive steering changed zero matched-holdout labels.
- Sign-flip and wrong-layer controls each improved matched-holdout truth by one
  row, so any claim that layer 14 is causally privileged is unsupported.
- Random matched-norm steering did not change labels, which suggests the hook is
  not simply randomizing behavior.
- The intervention was evaluated in forced-choice logit space, not free
  generation.

## Decision

Do not promote MC001G to a mechanism card.

Do not claim layer 14 as a reliable control surface.

The current supported MC001G result is:

> behavior substrate repaired, matched dense signature found, first simple dense
> intervention failed controls.

Next useful branches:

1. sparse-feature discovery on the same pre-hint-margin matched rows, with the
   dense intervention failure as a baseline;
2. path/source localization rather than additive dense steering;
3. a larger matched holdout set before any further intervention claim.
