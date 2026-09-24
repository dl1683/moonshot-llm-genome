# MC004 In-Context Binding V2 Lead-Time Preregistration

Date: 2026-06-30

## Motivation

MC004 V2 produced a condition-balanced behavior table, but the first signature
run failed because the original-note versus later-update choice was already
perfectly visible in output margin at the full prompt end.

This audit asks a narrower early-warning question:

> before the final answer instruction, does a hidden residual signal predict
> whether the model will use the original reference note or the later update,
> and does it beat the same-stage output-margin baseline?

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc004_in_context_binding_v2_leadtime.py`
- behavior result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_smoke_chat_20260630T163531.json`
- selected condition: `update_prefer_latest`
- primary rows: selected-condition rows labeled `target_correct` or
  `distractor_followed`
- card ID: `MC004`
- artifact prefix: `mc004_gemma2_2b_it_in_context_binding_v2_leadtime`
- render mode: `chat`

## Candidate Stages

The runner reconstructs partial prompts and evaluates hidden states at four
stages:

- `after_notes`: after the original reference notes, before the question;
- `after_question`: after the question, before the later-update instruction;
- `after_update`: after the later-update conflict instruction, before the
  final answer-format instruction;
- `after_answer_instruction`: full prompt end after the answer-format
  instruction.

Only the first three stages can support an early-warning claim.

## Signature Method

For each stage and layer:

1. Render the partial prompt with the chat template.
2. Extract residual-stream hidden state at the final token.
3. Train a mean-difference direction on discovery rows:
   `mean(target_correct) - mean(distractor_followed)`.
4. Select globally by discovery AUC across stages and layers, breaking ties
   toward earlier stages and earlier layers.
5. Report source-disjoint holdout AUC.

## Required Controls

- same-stage first-token output-margin baseline:
  original-code first token logit minus update-code first token logit;
- shuffled-label nulls on the selected stage and layer;
- target-first versus distractor-first holdout subgroup AUCs;
- source-disjoint split;
- single-condition primary-row assertion.

## Primary Success Criteria

The lead-time hidden signature passes only if all of these hold:

- selected stage is before `after_answer_instruction`;
- selected hidden holdout AUC is at least 0.85;
- selected hidden holdout AUC is at least 0.05 above the selected stage/layer
  shuffled-label null p95;
- both target-order holdout subgroup AUCs are at least 0.75;
- selected hidden holdout AUC beats the same-stage output-margin holdout AUC by
  at least 0.02;
- all primary rows come from `update_prefer_latest`.

If the selected stage is `after_answer_instruction`, this audit is only a
late-output diagnostic even if AUC is high.

## If The Gate Passes

Passing would justify a separate intervention preregistration at the selected
pre-answer stage. It would not itself prove a mechanism.
