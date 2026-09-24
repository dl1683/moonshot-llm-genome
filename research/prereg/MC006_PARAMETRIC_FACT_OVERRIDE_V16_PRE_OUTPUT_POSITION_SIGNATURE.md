# MC006 Parametric Fact Override V16 Pre-Output Position Signature Preregistration

Date: 2026-07-01

## Purpose

MC006 V14 passed the matched generated-answer behavior substrate for the
`fake_mapping_warning` real-after-fiction prompt surface. MC006 V15 then found
a perfect final-prompt-token hidden-state separator, but candidate-score margin
and final next-token output margin also reached perfect holdout AUC.

V16 asks a narrower question:

> Does the V14 table contain an earlier, pre-output hidden-state signal before
> the final `Answer:` output interface, and does that signal beat same-position
> output-logit controls?

V16 is a diagnostic only. It does not steer, edit, mask, patch, or otherwise
intervene on the model.

## Source Artifact

V16 uses the frozen V14 artifact:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`

The source artifact must have:

- `run_type == parametric_fact_override_v14_parser_normalized`;
- `diagnostic_class == parser_normalized_generated_substrate_passed`;
- `passed == true`;
- selected template rows available;
- parser-delta controls passed.

## Primary Rows

Primary rows are the selected-template V14 rows whose normalized selected label
is:

- `true_answer`, encoded as binary label 1;
- `override_answer`, encoded as binary label 0.

Rows with `lure_answer`, `unknown`, or `unparsed` are side rows and excluded
from signature fitting, but counted in the structural report.

Expected source table from V14:

- selected template: `fake_mapping_warning`;
- 40 selected-template rows;
- 30 binary rows;
- 10 side rows;
- 21 true-answer rows;
- 9 override-answer rows;
- holdout: 6 true-answer rows and 2 override-answer rows.

## Candidate Positions

For each primary row, V16 extracts hidden states at line-boundary token
positions inside the full V14 prompt:

1. `after_mapping_line`: after the fake mapping line containing
   `{country} -> {override}`.
2. `after_instruction_line`: after the line saying not to use the fake mapping
   for real-world geography.
3. `after_question_line`: after the real-world capital question.
4. `after_return_line`: after `Return only the city name.`
5. `final_prompt_token`: after `Answer:`.

Only the first four positions are selectable. `final_prompt_token` is reported
as a reference because V15 already showed that this position is output-margin
confounded.

Token positions are computed from the full-prompt fast-tokenizer offset map
with no special tokens, and the full prompt is run once per row without
padding. This avoids left-padding, batch-position ambiguity, and blind
substring-to-token assumptions when punctuation or newline bytes are merged
into a token.

## Candidate Features

For each selectable pre-output position and each transformer layer:

1. fit a mean-difference direction on non-holdout rows;
2. orient it on non-holdout AUC;
3. report discovery/non-holdout AUC and source-disjoint holdout AUC;
4. select by highest discovery AUC, then highest holdout AUC, then earliest
   position, then earliest layer.

The final-prompt-token reference uses the same fitting procedure but is not
eligible for pre-output selection.

## Baselines

V16 must report at least:

- same-position next-token output margin for each position: true first-token
  logit minus override first-token logit at that exact token position;
- final next-token output margin;
- candidate-score margin: true-capital mean logprob minus override-city mean
  logprob under the full prompt;
- prompt length;
- selected-position prefix token count;
- selected-position token id;
- final prompt token id;
- generated first-token id as a downstream/posthoc diagnostic only;
- parser-normalization delta.

The primary lead-time control is the same-position next-token output margin at
the selected pre-output position. Candidate-score margin and final next-token
output margin remain global output-interface controls. If they match the early
hidden signal, V16 cannot support intervention even if the same-position
lead-time diagnostic is positive.

## Shuffled-Label Null

Run 100 shuffled-label discovery selections with seed `26016`. For each
shuffle, select the best pre-output hidden position/layer by shuffled discovery
AUC and report its holdout AUC. The selected hidden signature must beat the
shuffled holdout AUC p95 by at least 0.05 for any lead-time support claim.

## Success Criteria

V16 reports two levels.

### Lead-Time Diagnostic Support

V16 supports a pre-output lead-time hidden signal only if all criteria pass:

1. Source artifact and structural checks pass.
2. Holdout has at least 2 rows per binary label.
3. Selected pre-output hidden holdout AUC is at least 0.85.
4. Selected pre-output hidden holdout AUC beats same-position next-token output
   margin holdout AUC by at least 0.02.
5. Selected pre-output hidden holdout AUC beats selected-position prefix token
   count holdout AUC by at least 0.02.
6. Selected pre-output hidden holdout AUC beats selected-position token-id
   holdout AUC by at least 0.02.
7. Selected pre-output hidden holdout AUC beats shuffled-label-selection p95 by
   at least 0.05.

### Mechanism-Signature Support

V16 supports a mechanism-grade hidden signature only if all lead-time criteria
pass and:

8. selected pre-output hidden holdout AUC beats candidate-score-margin holdout
   AUC by at least 0.02;
9. selected pre-output hidden holdout AUC beats final next-token output-margin
   holdout AUC by at least 0.02.

## Diagnostic Labels

- `pre_output_hidden_signature_supported`: all mechanism-signature criteria
  pass.
- `leadtime_signal_supported_but_output_global_confounded`: lead-time criteria
  pass, but candidate-score or final next-token output margins match the
  selected hidden signal.
- `source_artifact_invalid`: source or structural checks fail.
- `holdout_balance_failed`: holdout lacks at least 2 rows per label.
- `internal_signal_failed`: selected pre-output hidden holdout AUC is below
  0.85.
- `same_position_output_confounded`: same-position next-token output margin
  matches the selected pre-output hidden signature.
- `prefix_length_confounded`: selected-position prefix token count matches the
  selected pre-output hidden signature.
- `position_token_confounded`: selected-position token id matches the selected
  pre-output hidden signature.
- `shuffle_null_confounded`: shuffled-label selection matches the selected
  pre-output hidden signature.
- `mixed_signature_failure`: any other mixed failure.

## Allowed Interpretation

If mechanism-signature support passes:

> MC006 has a pre-output hidden signature on the V14 parser-normalized matched
> generated behavior table that beats same-position and global output-interface
> controls. Intervention may be preregistered only after carrying forward
> parser-delta, side-row, candidate-score, final-output, same-position-output,
> prompt/token, and shuffled-label controls.

If only lead-time support passes:

> MC006 contains an earlier hidden signal that is not merely the same-position
> output logits, but it is still not intervention-ready because the full output
> interface already exposes the behavior labels.

If V16 fails:

> MC006 remains behavior-supported but signature-blocked for learned capital
> facts versus fictional-code contamination.
