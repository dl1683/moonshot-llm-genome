# MC006 Parametric Fact Override V12 Real-After-Fiction Signature Preregistration

Date: 2026-07-01

## Purpose

MC006 V11 found a perfect hidden separation on the V10 chat-generated behavior
table, but the separation was matched by output margin, explicit requested-mode
text, and shuffled-label selection. V12 moves to a matched prompt surface where
the critical label is not the requested mode.

V12 uses the older V2 `real_after_fiction` condition:

```text
A fictional codebook from a separate task says: <country> -> <fictional-city>.
That codebook is not real-world geography. Ignore it for this question.
Question: What is the current real-world national capital of <country>?
Only output the city name.
Answer:
```

Every primary row has the same instruction class: ignore the fictional codebook
and answer real-world geography. The model nevertheless sometimes follows the
true capital and sometimes follows the fictional codebook. This makes V12 a
matched-surface diagnostic for truth-following versus fictional-code following.

This is a signature-discovery diagnostic only. It does not perform intervention
and does not claim a mechanism card.

## Source Artifact

V12 uses the V2 behavior artifact:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`

Source artifact SHA256:

`7ae7ae5a1d1a9cc1a36d7ac9684432b913dfc2bd9ecfb1ec556de76f66ebca8b`

## Primary Rows

Primary rows are restricted to:

- `condition == real_after_fiction`;
- `selected_label in {true_answer, override_answer}`.

Rows where the model selected the lure answer are excluded from the primary
binary signature table and reported as side rows.

Expected primary shape from the source artifact:

- 30 binary rows;
- 9 `true_answer` rows;
- 21 `override_answer` rows;
- discovery: 6 true, 12 override;
- calibration: 2 true, 4 override;
- holdout: 1 true, 5 override.

The holdout true-answer count is intentionally recorded as weak. V12 can be a
diagnostic, but it cannot support promotion unless the class-balance floor
passes.

## Model

- `Qwen/Qwen3-1.7B`

## Label

Binary label:

- `1` = `true_answer`;
- `0` = `override_answer`.

Discovery and calibration rows are used for fitting. Holdout rows are
source-disjoint from fitting rows, inherited from the V2 source split.

## Hidden Features

For each primary row, collect residual hidden state at the final prompt token
before answer generation, using the exact V2 prompt.

Candidate layers:

`0, 1, ..., final transformer layer`

For each layer:

1. Fit a mean-difference direction on non-holdout rows:
   `mean(true_answer) - mean(override_answer)`.
2. Score discovery and holdout rows by dot product with that direction.
3. Select the layer with highest discovery AUC, breaking ties by holdout AUC
   and then shallower layer.

## Baselines

V12 must compare the selected hidden direction against:

1. `candidate_score_margin`: source artifact
   `true_minus_override_mean_logprob`;
2. `next_token_output_margin`: freshly recomputed first-token logit margin
   `true_capital_first_token - override_city_first_token`;
3. `prompt_length`: exact prompt token count;
4. `final_token_id`: integer ID of the final prompt token;
5. `override_token_count`: token count of the fictional-code answer.

The candidate-score margin is expected to be an extremely strong baseline
because the source artifact itself is candidate-scored. A hidden signature that
does not beat it is diagnostic only.

## Nulls

Run a shuffled-label selection null:

- shuffle non-holdout labels;
- repeat layer selection over the same hidden features;
- score selected shuffled directions on the real holdout labels;
- 100 iterations with fixed seed.

## Success Criteria

V12 supports a diagnostic hidden signature only if all criteria pass:

1. structural checks pass;
2. holdout has at least 2 `true_answer` rows and at least 2 `override_answer`
   rows;
3. selected hidden holdout AUC is at least 0.85;
4. selected hidden holdout AUC beats candidate-score margin holdout AUC by at
   least 0.02;
5. selected hidden holdout AUC beats next-token output-margin holdout AUC by at
   least 0.02;
6. selected hidden holdout AUC beats prompt-length holdout AUC by at least
   0.02;
7. selected hidden holdout AUC beats shuffled-selection p95 by at least 0.05.

The class-balance floor is a promotion blocker, not a reason to skip the
diagnostic. If the holdout true-answer count is only one row, V12 must be
reported as a weak-holdout diagnostic even if AUC is high.

## Diagnostic Labels

- `hidden_signature_supported`: all criteria pass.
- `structural_invalid`: primary row structure is invalid.
- `holdout_balance_failed`: holdout class balance is too weak for promotion.
- `internal_signal_failed`: hidden holdout AUC is too weak.
- `candidate_score_confounded`: hidden AUC does not beat candidate-score
  margin.
- `output_margin_confounded`: hidden AUC does not beat next-token output
  margin.
- `prompt_length_confounded`: hidden AUC does not beat prompt length.
- `shuffle_null_confounded`: hidden AUC does not beat shuffled-label selection.
- `mixed_signature_failure`: any other mixed failure.

## Allowed Interpretation

If V12 passes:

> MC006 has a matched-prompt-surface hidden signature for
> real-after-fiction truth-following versus fictional-code following. It
> generalizes to source-disjoint holdout rows and beats candidate-score,
> next-token output-margin, prompt-length, and shuffled-label baselines.
> Intervention testing may proceed only on the selected layer/position.

If V12 fails:

> MC006 still lacks a mechanism-grade hidden signature for learned capital facts
> versus task-local fictional overwrites. The matched-surface lead is either too
> weakly balanced, output-confounded, or null-confounded for intervention.
