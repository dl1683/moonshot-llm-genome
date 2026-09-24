# MC005 Associative Lookup Response-Marker V26 Internal Row Signature Preregistration

Date: 2026-06-30

## Purpose

V22 showed partial row-level all-three structure inside the reliable
all-head layers-24-26 parent intervention surface. V23 found source-position
and margin heterogeneity. V24 showed source position was directional but not
sufficiently causal. V25 showed that a simple prompt-layout factorial account
was also too weak.

V26 asks the next narrower question:

> Can a pre-intervention internal activation signature predict which
> parent-effect lookup rows become all-three/pair-resistant rows?

This is a signature diagnostic only. It is not an intervention or
mechanism-card promotion.

## Source Artifact

V26 uses the completed V25 result:

`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`

The V25 artifact provides:

- lookup prompts and token positions;
- baseline margins;
- parent-effect labels;
- all-three row labels;
- target/distractor/random source positions;
- row family, seed, and layout metadata.

## Split

Discovery split:

- V25 lookup seed 233

Holdout split:

- V25 lookup seed 239

Only parent-effect rows are eligible. The prediction label is
`all_three_margin_row`.

The diagnostic requires both discovery and holdout to contain at least:

- 20 positive all-three rows;
- 40 negative non-all-three parent-effect rows.

## Internal Candidate Features

V26 collects pre-intervention residual-stream hidden states from the baseline
forward pass, before any source masking.

Candidate layers:

- 20
- 21
- 22
- 23
- 24
- 25
- 26

Candidate token positions:

- `target_value`
- `distractor_value`
- `random_value`
- `final_label`
- `final_colon`

For each layer/position candidate, V26 trains a discovery-only mean-difference
direction:

1. z-score hidden dimensions using discovery rows only;
2. compute positive mean minus negative mean;
3. L2-normalize the direction;
4. score discovery and holdout rows by dot product;
5. orient the score by discovery AUC.

The selected internal candidate is the layer/position pair with highest
discovery AUC, tie-broken by holdout AUC, then lower layer, then position name.

## Baselines

V26 compares the selected internal candidate against non-internal baselines
trained only on discovery rows:

- `baseline_margin`;
- `target_source_position`;
- `distractor_source_position`;
- `target_distractor_token_distance`;
- `abs_target_distractor_token_distance`;
- `target_slot`;
- `distractor_slot`;
- `layout_margin_linear`, a discovery-only mean-difference direction over
  baseline margin, target/distractor positions, distance, absolute distance,
  target slot, distractor slot, and target-before-distractor.

These baselines are treated as output/logit and prompt-layout shortcuts.

## Shuffled-Selection Null

V26 runs a shuffled-label selection null:

1. shuffle discovery labels;
2. select the best internal layer/position candidate by shuffled discovery AUC;
3. evaluate that shuffled-selected candidate on the true holdout labels;
4. repeat for at least 100 iterations.

The null statistic is the 95th percentile of shuffled-selected holdout AUCs.

## Success Criteria

V26 supports an internal row signature only if all criteria pass:

1. discovery and holdout label balance are valid;
2. selected internal holdout AUC is at least 0.70;
3. selected internal holdout AUC exceeds the best non-internal baseline holdout
   AUC by at least 0.05;
4. selected internal holdout AUC exceeds the shuffled-selection null p95 by at
   least 0.03;
5. selected internal holdout AUC is at least 0.60 in every eligible holdout
   subgroup among:
   - target position group: `early`, `mid_late`;
   - distractor relation: `near`, `far`.

## Diagnostic Labels

- `internal_row_signature_supported`: all criteria pass.
- `label_balance_failed`: discovery or holdout class balance is insufficient.
- `internal_holdout_signal_failed`: selected internal holdout AUC is below
  0.70.
- `baseline_confounded`: selected internal holdout AUC does not beat the best
  output/layout baseline by at least 0.05.
- `shuffle_null_confounded`: selected internal holdout AUC does not beat the
  shuffled-selection null p95 by at least 0.03.
- `subgroup_unstable`: global criteria pass, but an eligible holdout subgroup
  falls below 0.60 AUC.

## Allowed Interpretation

If V26 passes, the current claim becomes:

> MC005 has an internal activation signature that predicts row-level all-three
> structure inside the reliable layers-24-26 parent control surface, beyond
> output-margin and simple layout baselines.

If V26 fails, the current claim remains:

> MC005 has a reliable aggregate layers-24-26 control surface, but the
> row-level all-three subset is not yet explained by prompt-layout factors or
> by the tested internal signature.

