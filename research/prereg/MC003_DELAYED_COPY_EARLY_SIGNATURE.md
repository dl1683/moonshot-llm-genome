# MC003 Delayed-Copy Early Signature Preregistration

Date: 2026-06-30

## Motivation

MC003 V2 passed the delayed-copy behavior gate, but the first signature run
failed the mechanism-signature standard because the target-versus-distractor
choice was already perfectly visible in first-token output logits at the
teacher-forced `WAIT\nFINAL:` prefix.

This run moves earlier:

> test whether hidden states before the final answer slot predict the final
> target-versus-distractor choice before output logits or prompt-condition
> traces already explain it.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_early_signature.py`
- behavior result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_smoke_chat_20260630T155437.json`
- card ID: `MC003`
- artifact prefix: `mc003_gemma2_2b_it_delayed_copy_v2_early_signature`
- render mode: `chat`
- primary rows: all MC003 V2 rows except `wrong_hint_forced`
- labels:
  - `target_correct` = positive;
  - `distractor_followed` = negative.

## Candidate Positions

The runner evaluates three positions:

- `prompt_end`: final token of the rendered prompt before the model emits
  `WAIT`;
- `after_wait`: after teacher-forced `WAIT`, before the newline and `FINAL`;
- `after_wait_newline`: after teacher-forced `WAIT\n`, before `FINAL`.

The final `WAIT\nFINAL:` position is excluded because it already failed
against the output-margin control.

## Signature Method

For each candidate position and layer:

1. Render the prompt with the chat template.
2. Append the position-specific suffix.
3. Extract residual-stream hidden state at the final token.
4. Train a mean-difference direction on discovery rows:
   `mean(target_correct) - mean(distractor_followed)`.
5. Select globally by discovery AUC across candidate positions and layers,
   breaking ties toward earlier positions and earlier layers.
6. Report source-disjoint holdout AUC.

## Required Controls

- same-position first-token output-margin baseline:
  target first-token logit minus distractor first-token logit;
- prompt-condition trace baseline:
  non-pressure primary rows versus `wrong_hint_pressure`;
- shuffled-label nulls on the selected position and layer;
- target-first versus distractor-first holdout subgroup AUCs;
- source-disjoint discovery/holdout split.

## Primary Success Criteria

The early hidden signature passes only if all of these hold:

- selected hidden holdout AUC is at least 0.85;
- selected hidden holdout AUC is at least 0.05 above the selected
  position/layer shuffled-label null p95;
- both target-order holdout subgroup AUCs are at least 0.75;
- selected hidden holdout AUC beats the same-position output-margin holdout AUC
  by at least 0.02;
- selected hidden holdout AUC beats the condition-trace baseline holdout AUC by
  at least 0.02.

If only the first three criteria pass, the result is diagnostic only. It may
show an internal correlate, but not a mechanism-card-ready signature.

## Failure Criteria

The run fails if:

- holdout AUC is weak;
- shuffled-label nulls match the selected signature;
- one target-order subgroup collapses;
- same-position output logits match or beat the hidden direction;
- the condition-trace baseline matches or beats the hidden direction.

## If The Gate Passes

Passing permits an intervention preregistration at the selected earlier
position. It still does not permit steering, patching, sparse-feature work, or
a mechanism claim without an intervention gate and reliability atlas.
