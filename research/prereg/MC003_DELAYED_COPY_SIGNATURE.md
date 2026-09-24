# MC003 Delayed-Copy Signature Discovery Preregistration

Date: 2026-06-30

## Motivation

MC003 V2 passed the behavior substrate gate. It produced clean delayed-copy
target behavior under baseline and locality conditions, while
`wrong_hint_pressure` moved 35 of 40 baseline-clean sources to the distractor,
including 12 holdout sources.

This run asks whether a hidden-state signature at the delayed answer prefix
predicts the target-versus-distractor output choice in a source-disjoint
holdout.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_signature.py`
- behavior result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_smoke_chat_20260630T155437.json`
- card ID: `MC003`
- artifact prefix: `mc003_gemma2_2b_it_delayed_copy_v2_signature`
- render mode: `chat`
- answer prefix: `WAIT\nFINAL:`
- primary rows: all MC003 V2 rows except `wrong_hint_forced`
- labels:
  - `target_correct` = positive;
  - `distractor_followed` = negative.

## Signature Method

For every primary row:

1. Render the prompt with the chat template.
2. Append the teacher-forced answer prefix `WAIT\nFINAL:`.
3. Extract residual-stream hidden states at the final prefix token for every
   model layer.
4. Train a mean-difference direction on discovery rows:
   `mean(target_correct) - mean(distractor_followed)`.
5. Select the layer by discovery AUC.
6. Report holdout AUC on source-disjoint holdout rows.

## Required Controls

- first-token output-margin baseline:
  target first-token logit minus distractor first-token logit at the same
  answer prefix;
- shuffled-label nulls on the selected layer;
- target-first versus distractor-first holdout subgroup AUCs;
- source-disjoint discovery/holdout split.

## Primary Success Criteria

The hidden signature passes only if all of these hold:

- selected-layer holdout AUC is at least 0.85;
- selected-layer holdout AUC is at least 0.05 above the selected-layer
  shuffled-label null p95;
- both target-order holdout subgroup AUCs are at least 0.75;
- selected-layer holdout AUC beats the first-token output-margin holdout AUC by
  at least 0.02.

If the first three criteria pass but the output-margin criterion fails, the run
is diagnostic only: it has a hidden correlate of the behavior, but not a
mechanism-card-ready signature.

## Failure Criteria

The signature discovery fails if:

- holdout AUC is weak;
- shuffled labels match the selected signature;
- one target-order subgroup collapses;
- output logits explain the behavior as well as or better than the hidden
  direction.

## If The Gate Passes

Passing permits an intervention preregistration. It still does not permit a
mechanism claim. The intervention must include random/nearby directions,
wrong-layer and wrong-token controls, prompt-only baselines, dose response,
source-disjoint holdout, and locality rows.
