# MC006 Parametric Fact Override V5 Hidden Signature Preregistration

Date: 2026-07-01

## Purpose

MC006 V4 passed a narrow source-selected behavior substrate. V5 asks whether
that behavior has a hidden-state signature that is stronger than easier
baselines.

This is a signature-discovery gate only. It does not perform intervention and
does not claim a mechanism card.

## Source Artifact

V5 uses the passed V4 behavior artifact:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`

Primary rows are restricted to V4 clean source-level contrasts. This excludes
the single non-clean source, Japan, whose `false_claim_check` row selected the
lure answer.

Expected primary shape:

- 22 clean sources;
- 5 original holdout sources;
- 88 rows;
- 66 real-world rows;
- 22 fictional-code rows.

## Model

- `Qwen/Qwen3-1.7B`

## Label

Binary label:

- `1` = real-world answer mode:
  `direct_real_paraphrase`, `true_fact_paraphrase`, `false_claim_check`;
- `0` = fictional-code mode:
  `fictional_code_lookup`.

Discovery/holdout split is inherited from the V4 source split, so holdout is
source-disjoint from discovery.

## Hidden Features

For each primary row, collect residual hidden state at the final prompt token
before answer generation, using the exact V4 prompt.

Candidate layers:

`0, 1, ..., final transformer layer`

For each layer:

1. Fit a mean-difference direction on discovery rows:
   `mean(real_world) - mean(fictional_code)`.
2. Score discovery and holdout rows by dot product with that direction.
3. Select the layer with highest discovery AUC, breaking ties by holdout AUC
   and then shallower layer.

## Baselines

V5 must compare the selected hidden direction against:

1. `output_margin`: first-token logit margin
   `true_capital_first_token - override_city_first_token`;
2. `prompt_format`: direct binary prompt-family baseline,
   `condition != fictional_code_lookup`;
3. `prompt_length`: token count of the exact prompt;
4. `final_token_id`: integer ID of the final prompt token.

The `prompt_format` baseline is intentionally included because V4 is
prompt-bounded. If it reaches perfect holdout AUC, a hidden-state signature that
only matches it is not enough for mechanism promotion.

## Nulls

Run a shuffled-label selection null:

- shuffle discovery labels;
- repeat layer selection over the same hidden features;
- score selected shuffled directions on the real holdout labels;
- 100 iterations with fixed seed.

## Success Criteria

V5 supports a diagnostic hidden signature only if all criteria pass:

1. structural checks pass;
2. selected hidden holdout AUC is at least 0.85;
3. selected hidden holdout AUC beats output-margin holdout AUC by at least 0.02;
4. selected hidden holdout AUC beats prompt-format holdout AUC by at least 0.02;
5. selected hidden holdout AUC beats shuffled-selection p95 by at least 0.05;
6. every holdout source has mean selected-hidden score for its real-world rows
   greater than its fictional-code row.

## Diagnostic Labels

- `hidden_signature_supported`: all criteria pass.
- `structural_invalid`: primary row structure is invalid.
- `internal_signal_failed`: hidden holdout AUC is too weak.
- `output_margin_confounded`: hidden AUC does not beat the output-margin
  baseline.
- `prompt_format_confounded`: hidden AUC does not beat the prompt-format
  baseline.
- `shuffle_null_confounded`: hidden AUC does not beat shuffled-label selection.
- `source_pair_instability`: holdout source-level real-vs-fictional ordering is
  unstable.
- `mixed_signature_failure`: any other mixed failure.

## Allowed Interpretation

If V5 passes:

> MC006 V4 has a hidden-state mode signature that generalizes to source-disjoint
> holdout rows and beats output/logit and prompt-format baselines. Intervention
> testing may proceed on the selected layer/position.

If V5 fails:

> MC006 V4 remains a behavior substrate only. Hidden-state work should either
> repair the signature confound or change the task design before intervention.
