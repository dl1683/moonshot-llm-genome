# MC005 Associative Lookup Response-Marker V23 Row Heterogeneity Preregistration

Date: 2026-06-30

## Question

MC005 V22 failed the broad row-level all-three interaction gate. The parent
`slice_l24_26_all` control still replicated, source controls and nulls passed,
and 67 rows showed all-three margin structure, but those rows were only 26.48
percent of parent-effect rows.

V23 asks:

> Which row features explain why some V22 rows require the full layers-24-26
> parent while most parent-effect rows are substantially recovered by a
> leave-one-layer-out pair?

## Fixed Inputs

V23 is an offline diagnostic over the V22 artifact:

- `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v22_row_interaction_20260630T201924.json`

No new model scoring is allowed in V23. The diagnostic must use only V22 rows,
V22 row-level deltas, and V22 metadata.

## Required Strata

V23 must stratify parent-effect rows by:

1. query pair index;
2. target source-value token position;
3. distance from target source-value token to final marker;
4. baseline target-minus-distractor margin band;
5. target value token identity;
6. best leave-one-layer-out pair winner;
7. omitted layer of the best leave-one-layer-out pair.

For each stratum, report:

- row count;
- parent-effect row count;
- all-three margin row count;
- all-three fraction among parent-effect rows;
- parent-flip pair-resistant count;
- median best-pair share among parent-effect rows;
- median single-layer-sum residual among parent-effect rows.

## Failure-Reason Accounting

For every parent-effect row that is not an all-three margin row, classify the
failed condition:

- `pair_share_recovered`: at least one leave-one-layer-out pair recovers at
  least 60 percent of the parent negative-effect magnitude;
- `single_residual_weak`: the single-layer-sum residual is greater than -1.0;
- `pair_plus_single_residual_weak`: at least one pair-plus-omitted-layer
  residual is greater than -0.5.

Rows may have multiple failure reasons. V23 must report single-reason and
multi-reason counts.

## Required Outputs

V23 must write one JSON artifact containing:

- source V22 path and SHA256;
- V22 headline replication metrics;
- enriched per-row records for V22 lookup rows;
- required stratum tables;
- failure-reason counts;
- a short machine-readable diagnostic label.

## Diagnostic Labels

- `best_pair_position_heterogeneity`: best-pair winner, source position, or
  query index produces a strong concentration of all-three rows or failures.
- `margin_band_heterogeneity`: baseline margin bands dominate the row split.
- `token_identity_heterogeneity`: target token identity dominates the row
  split.
- `diffuse_heterogeneity`: no required stratum has a strong enough contrast to
  dominate the row split.
- `invalid_source_artifact`: the V22 artifact is missing required data or does
  not match the expected V22 diagnostic class.

## Contrast Rule

A stratum family is a strong contrast only if:

- it has at least two groups with at least 16 parent-effect rows each; and
- max all-three fraction minus min all-three fraction across eligible groups is
  at least 0.25.

The diagnostic label should choose the first matching family in this order:

1. best pair winner / omitted layer / source position / query index;
2. baseline margin band;
3. target token identity;
4. diffuse heterogeneity.

V23 is explanatory, not a mechanism promotion. A strong contrast proposes the
next experiment target; it does not rescue the failed V22 row-level promotion.
