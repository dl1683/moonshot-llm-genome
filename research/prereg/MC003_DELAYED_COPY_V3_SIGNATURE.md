# MC003 Delayed-Copy V3 Condition-Balanced Signature Preregistration

Date: 2026-06-30

## Motivation

MC003 V3 found a condition-balanced wrong-hint table. The frozen condition is
`wrong_hint_balanced`, selected by preregistered behavior-only criteria before
reading hidden states:

- `target_correct`: 19/40
- `distractor_followed`: 21/40
- discovery split: 12 target, 15 distractor
- holdout split: 7 target, 6 distractor
- both holdout target-order subgroups contain both labels
- baseline and correct-hint locality are 40/40 target-correct

This signature run asks whether hidden states predict target-versus-distractor
outcomes inside that single frozen prompt condition.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_v3_signature.py`
- behavior preregistration:
  `research/prereg/MC003_DELAYED_COPY_V3_CONDITION_BALANCE.md`
- behavior result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v3_condition_balance_smoke_chat_20260630T161743.json`
- selected primary condition: `wrong_hint_balanced`
- card ID: `MC003`
- artifact prefix:
  `mc003_gemma2_2b_it_delayed_copy_v3_condition_balanced_signature`
- render mode: `chat`
- labels:
  - `target_correct` = positive;
  - `distractor_followed` = negative.

## Candidate Positions

The runner evaluates the same early positions as the failed V2 early-signature
run:

- `prompt_end`: final token of the rendered prompt before the model emits
  `WAIT`;
- `after_wait`: after teacher-forced `WAIT`, before newline and `FINAL`;
- `after_wait_newline`: after teacher-forced `WAIT\n`, before `FINAL`.

The final `WAIT\nFINAL:` position remains excluded because it already failed
against the output-margin control in V2.

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
- shuffled-label nulls on the selected position and layer;
- target-first versus distractor-first holdout subgroup AUCs;
- source-disjoint discovery/holdout split;
- single-condition primary-row assertion.

No pressure/non-pressure condition-trace control is used here because the
primary table is intentionally one frozen condition. The replacement control is
the explicit single-condition assertion plus the behavior status showing both
labels within that condition.

## Primary Success Criteria

The condition-balanced hidden signature passes only if all of these hold:

- selected hidden holdout AUC is at least 0.85;
- selected hidden holdout AUC is at least 0.05 above the selected
  position/layer shuffled-label null p95;
- both target-order holdout subgroup AUCs are at least 0.75;
- selected hidden holdout AUC beats the same-position output-margin holdout AUC
  by at least 0.02;
- all primary rows come from `wrong_hint_balanced`.

If only the first three criteria pass, the result is diagnostic only. It may
show an internal correlate, but not a mechanism-card-ready signature.

## If The Gate Passes

Passing permits an intervention preregistration at the selected earlier
position. It still does not permit a mechanism claim without intervention,
locality, fluency, robustness, side-effect, and null-control gates.
