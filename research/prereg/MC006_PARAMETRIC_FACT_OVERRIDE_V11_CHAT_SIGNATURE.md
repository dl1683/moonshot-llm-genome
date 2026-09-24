# MC006 Parametric Fact Override V11 Chat-Signature Preregistration

Date: 2026-07-01

## Purpose

MC006 V10 passed a chat-rendered generated-answer behavior substrate:
real-world capital mode and fictional-code mode both worked in one prompt
family without listing the true capital.

V11 asks whether that V10 behavior has a hidden-state signature that survives
obvious controls. This is a signature-discovery gate only. It does not perform
intervention and does not claim a mechanism card.

The main risk is explicit: V10 includes the requested mode string in the prompt.
A final-prompt-token hidden direction may therefore classify prompt mode rather
than a learned knowledge/control mechanism. V11 treats that as a confound, not
as promotion evidence.

## Source Artifact

V11 uses the passed V10 behavior artifact:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v10_chat_generated_20260630T232318.json`

Primary rows are restricted to V10 clean source-level contrasts:

- `mode_real` selected the true capital;
- `mode_fictional` selected the fictional city code.

Expected primary shape:

- 20 clean sources;
- 5 original holdout sources;
- 40 rows;
- 20 `mode_real` rows;
- 20 `mode_fictional` rows.

Brazil and Croatia are excluded from the primary signature table because their
V10 source-level contrasts were not clean.

## Model

- `Qwen/Qwen3-1.7B`

## Label

Binary label:

- `1` = `mode_real`;
- `0` = `mode_fictional`.

Discovery/holdout split is inherited from the V10 source split. Holdout is
source-disjoint. Calibration rows are used with discovery rows for fitting.

## Hidden Features

For each primary row, collect residual hidden state at the final prompt token
before answer generation, using the exact V10 rendered chat prompt.

Candidate layers:

`0, 1, ..., final transformer layer`

For each layer:

1. Fit a mean-difference direction on non-holdout rows:
   `mean(mode_real) - mean(mode_fictional)`.
2. Score discovery and holdout rows by dot product with that direction.
3. Select the layer with highest discovery AUC, breaking ties by holdout AUC
   and then shallower layer.

## Baselines

V11 must compare the selected hidden direction against:

1. `output_margin`: first generation-token logit margin
   `true_capital_first_token - override_city_first_token`;
2. `requested_mode`: direct binary prompt baseline,
   `requested_mode == REAL_WORLD_CAPITAL`;
3. `prompt_length`: rendered chat prompt token count;
4. `final_token_id`: integer ID of the final rendered prompt token.

The `requested_mode` baseline is intentionally included because V10's behavior
contract explicitly asks for one of two modes. If this baseline reaches perfect
holdout AUC, a hidden state that only matches it is not a mechanism-grade
signature.

## Nulls

Run a shuffled-label selection null:

- shuffle non-holdout labels;
- repeat layer selection over the same hidden features;
- score selected shuffled directions on the real holdout labels;
- 100 iterations with fixed seed.

## Success Criteria

V11 supports a diagnostic hidden signature only if all criteria pass:

1. structural checks pass;
2. selected hidden holdout AUC is at least 0.85;
3. selected hidden holdout AUC beats output-margin holdout AUC by at least
   0.02;
4. selected hidden holdout AUC beats requested-mode holdout AUC by at least
   0.02;
5. selected hidden holdout AUC beats prompt-length holdout AUC by at least
   0.02;
6. selected hidden holdout AUC beats shuffled-selection p95 by at least 0.05;
7. every holdout source has selected-hidden score for its `mode_real` row
   greater than its `mode_fictional` row.

## Diagnostic Labels

- `hidden_signature_supported`: all criteria pass.
- `structural_invalid`: primary row structure is invalid.
- `internal_signal_failed`: hidden holdout AUC is too weak.
- `requested_mode_confounded`: hidden AUC does not beat the explicit requested
  mode baseline.
- `output_margin_confounded`: hidden AUC does not beat the output-margin
  baseline.
- `prompt_length_confounded`: hidden AUC does not beat rendered prompt length.
- `shuffle_null_confounded`: hidden AUC does not beat shuffled-label selection.
- `source_pair_instability`: holdout source-level real-vs-fictional ordering is
  unstable.
- `mixed_signature_failure`: any other mixed failure.

## Allowed Interpretation

If V11 passes:

> MC006 V10 has a hidden-state mode signature that generalizes to
> source-disjoint holdout rows and beats output/logit, requested-mode,
> prompt-length, and shuffled-label baselines. Intervention testing may proceed
> on the selected layer/position.

If V11 fails:

> MC006 V10 remains a behavior substrate only. Hidden-state work should either
> repair the signature confound or change the task design before intervention.
