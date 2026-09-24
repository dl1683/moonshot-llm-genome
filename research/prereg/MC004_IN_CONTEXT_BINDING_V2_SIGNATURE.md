# MC004 In-Context Binding V2 Signature Preregistration

Date: 2026-06-30

## Motivation

MC004 V2 created the first condition-balanced table for nonce in-context
binding. The frozen condition is `update_prefer_latest`, selected by behavior
criteria before hidden-state inspection.

This run asks whether early hidden states predict whether the model follows
the original reference note or the later update inside that single condition.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc004_in_context_binding_v2_signature.py`
- behavior preregistration: `research/prereg/MC004_IN_CONTEXT_BINDING_V2.md`
- behavior result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_smoke_chat_20260630T163531.json`
- selected primary condition: `update_prefer_latest`
- card ID: `MC004`
- artifact prefix: `mc004_gemma2_2b_it_in_context_binding_v2_signature`
- render mode: `chat`
- labels:
  - `target_correct` = original reference note;
  - `distractor_followed` = later update.

Rows with labels other than `target_correct` or `distractor_followed` are
excluded from the signature table.

## Candidate Positions

The runner evaluates three early positions:

- `prompt_end`: final token of the rendered prompt before the model answers;
- `after_answer_prefix`: after teacher-forced `Answer:`;
- `after_final_prefix`: after teacher-forced `FINAL:`.

The answer-prefix positions are included because this task does not require
the MC003 `WAIT` scaffold.

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
  original code token logit minus update code token logit;
- shuffled-label nulls on the selected position and layer;
- target-first versus distractor-first holdout subgroup AUCs;
- source-disjoint discovery/holdout split;
- single-condition primary-row assertion.

## Primary Success Criteria

The condition-balanced hidden signature passes only if all of these hold:

- selected hidden holdout AUC is at least 0.85;
- selected hidden holdout AUC is at least 0.05 above the selected
  position/layer shuffled-label null p95;
- both target-order holdout subgroup AUCs are at least 0.75;
- selected hidden holdout AUC beats the same-position output-margin holdout AUC
  by at least 0.02;
- all primary rows come from `update_prefer_latest`.

If this gate passes, the next step is an intervention preregistration. Passing
does not itself prove a mechanism.
