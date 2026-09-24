# MC006 Parametric Fact Override V15 Parser-Normalized Signature Preregistration

Date: 2026-07-01

## Purpose

MC006 V14 passed a generated-answer matched `real_after_fiction` behavior
substrate under a narrow accent-normalized strict parser. V15 asks whether that
substrate contains a hidden-state signature that is stronger than easier
observables.

V15 is a signature diagnostic only. It does not steer, edit, mask, patch, or
perform intervention.

## Source Artifact

V15 uses the frozen V14 artifact:

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

## Candidate Features

For each primary row, collect Qwen3-1.7B final-prompt-token residual hidden
states from every transformer layer under the V14 prompt text.

For each layer:

1. fit a mean-difference direction on non-holdout rows;
2. orient it on non-holdout AUC;
3. report discovery/non-holdout AUC and source-disjoint holdout AUC;
4. select by highest discovery AUC, then highest holdout AUC, then earliest
   layer.

## Baselines

V15 must report at least:

- candidate-score margin: true-capital mean logprob minus override-city mean
  logprob under the same prompt;
- next-token output margin: true first-token logit minus override first-token
  logit at the final prompt token;
- prompt length;
- final prompt token id;
- true candidate token count;
- override candidate token count;
- generated first-token id as a downstream/posthoc diagnostic only.

Candidate-score and next-token output margins are the critical baselines.
Generated first-token id is not a promotion criterion because it is downstream
of the behavior label, but it must be reported to expose output-text leakage.

## Shuffled-Label Null

Run 100 shuffled-label discovery selections with seed `26015`. For each
shuffle, select the best hidden layer by shuffled discovery AUC and report its
holdout AUC. The selected hidden signature must beat the shuffled holdout AUC
p95 by at least 0.05.

## Success Criteria

V15 supports a hidden signature only if all criteria pass:

1. Source artifact and structural checks pass.
2. Holdout has at least 2 rows per binary label.
3. Selected hidden holdout AUC is at least 0.85.
4. Selected hidden holdout AUC beats candidate-score-margin holdout AUC by at
   least 0.02.
5. Selected hidden holdout AUC beats next-token-output-margin holdout AUC by at
   least 0.02.
6. Selected hidden holdout AUC beats prompt-length holdout AUC by at least
   0.02.
7. Selected hidden holdout AUC beats final-token-id holdout AUC by at least
   0.02.
8. Selected hidden holdout AUC beats shuffled-label-selection p95 by at least
   0.05.

## Diagnostic Labels

- `hidden_signature_supported`: all criteria pass.
- `source_artifact_invalid`: source or structural checks fail.
- `holdout_balance_failed`: holdout lacks at least 2 rows per label.
- `internal_signal_failed`: selected hidden holdout AUC is below 0.85.
- `candidate_score_confounded`: candidate-score margin matches the selected
  hidden signature.
- `output_margin_confounded`: next-token output margin matches the selected
  hidden signature.
- `prompt_length_confounded`: prompt length matches the selected hidden
  signature.
- `final_token_confounded`: final prompt token id matches the selected hidden
  signature.
- `shuffle_null_confounded`: shuffled-label selection matches the selected
  hidden signature.
- `mixed_signature_failure`: any other mixed failure.

## Allowed Interpretation

If V15 passes:

> MC006 has a hidden signature on the V14 parser-normalized matched generated
> behavior table. Intervention may be preregistered only after carrying forward
> candidate-score, next-token output, prompt/token, shuffled-label, parser-delta,
> and side-row controls.

If V15 fails:

> MC006 still lacks a mechanism-grade hidden signature for learned capital facts
> versus fictional-code contamination. The result should be treated as a
> behavior-supported, signature-blocked knowledge-like branch.
