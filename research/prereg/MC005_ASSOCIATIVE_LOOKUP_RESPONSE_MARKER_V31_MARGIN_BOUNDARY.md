# MC005 Associative Lookup Response-Marker V31 Margin Boundary Preregistration

Date: 2026-06-30

## Purpose

V29 showed exact lookup mediation by replacing only the final-query
self-attention output writes in layers 24-26. V30 then showed that the same
write-replacement operation fails strict answer-absent null locality on fresh
rows.

The V30 changed rows all sat close to the target-versus-distractor margin
boundary before replacement. V31 tests that as a narrow failure-mode
hypothesis:

> Are answer-absent write-replacement null failures concentrated in low-margin
> rows, while the lookup target effect remains a large-margin causal effect?

V31 is not a mechanism-card promotion attempt. It is a reliability-boundary
audit for the V29/V30 write surface.

## Source Artifacts

V31 uses:

- V25 rows:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V29 write-replacement result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
- V30 write-null result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json`

V25 and V29 validation follows V30. V30 must validate as
`associative_lookup_response_marker_v30_write_null_sweep` with diagnostic class
`fresh_write_null_failed`, reproduced V29 replay failure, fresh mean-delta
tolerance passing, and fresh target-win changes present.

## Panels

V31 has two scored panels.

1. `lookup_target_write`: V25 seed-239 parent-effect lookup rows. V31 scores
   baseline and target-value write replacement in layers 24-26, then records
   every target-win loss row and its baseline margin.
2. `fresh_null_margin_128`: fresh V25-compatible answer-absent null rows with
   `build_null_rows(tokenizer, seed, 128)` for seeds
   311, 313, 317, 331, 337, 347, 349, and 353. V31 scores the same three null
   write-replacement source keys used in V29/V30:
   `source_value`, `non_source_control_value`, and `final_colon`.

The V30 changed rows are imported from the V30 artifact and included in the
combined null-flip margin audit, but they are not rescored.

## Margin Bands

For every scored row, V31 records the baseline target-minus-distractor margin
and places it into one absolute-margin band:

- `le_0p25`
- `0p25_0p5`
- `0p5_1`
- `1_2`
- `gt_2`

For null rows, V31 records target-win flips by band for each source key.

For lookup rows, V31 records target-win losses by band for the target-value
write replacement.

## Criteria

V31 records these criteria:

1. V25, V29, and V30 source artifacts validate.
2. V30 imported null flips all have absolute baseline margin at most 0.5.
3. V31 fresh null flips, if any, all have absolute baseline margin at most 0.5.
4. V31 fresh null arms all remain within absolute mean-delta 0.25.
5. V31 lookup target write replacement reproduces a large target effect:
   mean delta at most -5.0 and target-win loss at least 8.
6. V31 lookup target-win loss rows all have baseline margin at least 2.0.

## Diagnostic Labels

- `margin_boundary_supported`: all criteria pass.
- `source_artifact_invalid`: V25, V29, or V30 validation fails.
- `null_boundary_broad`: any imported or fresh null flip has absolute baseline
  margin greater than 0.5, or any fresh null arm exceeds the mean-delta
  tolerance.
- `lookup_effect_not_reproduced`: target lookup write replacement no longer
  reproduces the expected large effect.
- `lookup_effect_near_margin_confounded`: lookup target-win losses are not all
  from rows with baseline margin at least 2.0.
- `mixed_margin_boundary`: any other mixed state.

## Allowed Interpretation

If V31 returns `margin_boundary_supported`:

> The V29/V30 write surface has a precise failure mode: answer-absent null
> flips are confined to low-margin target/distractor threshold rows, while the
> lookup target effect changes high-margin rows by a large causal write effect.
> This explains the reliability boundary but does not by itself promote the
> surface to a full mechanism card.

If V31 returns `null_boundary_broad`:

> The answer-absent reliability failure is not merely a low-margin threshold
> artifact. The write surface has broader locality problems.

If V31 returns `lookup_effect_near_margin_confounded`:

> The lookup target-win-loss evidence is itself partly near-threshold, so the
> write surface cannot be cleanly separated from the same row-margin boundary
> that breaks the nulls.
