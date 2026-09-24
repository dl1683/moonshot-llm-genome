# MC005 Associative Lookup Response-Marker V25 Factorial Row-Heterogeneity Preregistration

Date: 2026-06-30

## Purpose

V22 showed that the all-head layers-24-26 parent path is an aggregate
interaction surface, but broad row-level all-three structure did not reach the
promotion threshold. V23 found a strong source-position/query-index
heterogeneity lead. V24 causally moved source position while holding target
identity fixed and found a directional but sub-threshold effect.

V25 asks whether row-level all-three structure needs a stronger factorial
account:

- target source position;
- distractor source position;
- observed baseline margin;
- blocked target/distractor value identity.

This is still a diagnostic, not a mechanism-card promotion attempt.

## Fixed Surface

Model:

- `Qwen/Qwen3-1.7B`

Prompt marker:

- `Response:`

Parent intervention path:

- all heads in layers 24, 25, and 26: `slice_l24_26_all`

Component paths:

- `slice_l24_25_all`
- `slice_l24_26_all_pair`
- `slice_l25_26_all`
- `single_l24_all`
- `single_l25_all`
- `single_l26_all`

Primary source arm:

- `target_value`

Source controls:

- `distractor_value`
- `random_value`

Answer-absent null arms:

- `source_value`
- `non_source_control_value`
- `earlier_neutral_colon`
- `final_label`
- `final_colon`

## Row Construction

V25 uses pair-count 16 `Response:` lookup prompts.

For each base family, the target key/value pair, distractor pair, random pair,
and query identity are fixed across all variants. The target and distractor
pairs are then placed into explicit slots:

| Variant | Target slot | Target group | Distractor relation | Distractor slot |
| --- | ---: | --- | --- | ---: |
| `early_near` | 0 | `early` | `near` | 1 |
| `early_far` | 0 | `early` | `far` | 13 |
| `late_near` | 13 | `mid_late` | `near` | 14 |
| `late_far` | 13 | `mid_late` | `far` | 0 |

All remaining pairs keep their sampled order after removing the target and
distractor pairs. This makes target position and distractor relation causal
manipulations while treating value identity as a family block.

Lookup seeds:

- 233
- 239

Base families per seed:

- 32

Total lookup rows:

- 256

Answer-absent null seeds:

- 251
- 257

Answer-absent rows per null seed:

- 96

## Row-Level Definitions

For every lookup row, V25 records:

- baseline margin;
- parent delta;
- target/distractor/random source positions;
- target slot and distractor slot;
- target/distractor token IDs and value strings;
- leave-one-layer-out pair deltas;
- single-layer deltas.

A `parent_effect_row` is a row where:

- the baseline predicts the target over the distractor; and
- parent target-value masking changes target-minus-distractor margin by at
  most `-1.0`.

An `all_three_margin_row` is a parent-effect row where:

- every leave-one-layer-out pair recovers less than `0.60` of the parent
  negative-effect magnitude;
- the parent remains at least `1.0` margin point more negative than the sum of
  single-layer deltas;
- every pair-plus-held-out-single reconstruction leaves at least `0.5` margin
  point of negative residual.

Baseline-margin bands are assigned after scoring, using only parent-effect
rows:

- `low_margin`: lower half by baseline margin;
- `high_margin`: upper half by baseline margin.

Rows without parent effect are labeled `non_parent_effect` for margin-band
strata.

## Success Criteria

V25 supports a factorial row-heterogeneity account only if all criteria pass.

Core validity:

1. Overall lookup baseline target wins are at least 75 percent.
2. Every target-position/distractor-relation cell has baseline target wins of
   at least 75 percent.
3. Every target-position/distractor-relation cell has a parent target-value
   mean delta at most `-1.0` and target-win loss at least 3.
4. In every target-position/distractor-relation cell, the parent target-value
   mean delta is at least `0.50` more negative than both distractor-value and
   random-value controls.
5. Both answer-absent null seeds are `clean_null`.

Factorial evidence:

6. The best minus worst all-three fraction across the four
   target-position/distractor-relation cells is at least `0.25`, with every
   cell having at least 24 parent-effect rows.
7. Distractor relation modulates the source-position effect: the absolute
   difference between the late-minus-early all-three fraction in `near` rows
   and the late-minus-early all-three fraction in `far` rows is at least
   `0.15`.
8. Baseline margin contributes within the factorial layout: the high-margin
   all-three fraction exceeds the low-margin all-three fraction by at least
   `0.20` inside at least one target-position group.
9. The best factorial cell is not a single-family artifact: its all-three rows
   span at least 8 base families and both lookup seeds.

## Diagnostic Labels

- `factorial_row_heterogeneity_supported`: all criteria pass.
- `factorial_parent_failed`: any core baseline, parent, source-control, or
  null criterion fails.
- `factorial_cell_contrast_failed`: core criteria pass, but the four-cell
  all-three contrast is too small or underpowered.
- `distractor_modulation_failed`: cell contrast passes, but distractor
  relation does not modulate the source-position effect.
- `margin_factor_failed`: distractor modulation passes, but baseline margin
  does not add the required within-position contrast.
- `family_artifact_failed`: all other factorial criteria pass, but the best
  cell is concentrated in too few families or one seed.

## Allowed Interpretation

If V25 passes, the current claim becomes:

> MC005 row-level interaction structure is not explained by source position
> alone; it is supported by a factorial source-position, distractor-position,
> margin, and blocked-identity account under the fixed Qwen3-1.7B `Response:`
> associative lookup contract.

If V25 fails, the current claim remains:

> MC005 has a reliable aggregate all-head layers-24-26 control surface with
> partial row-level all-three structure, but the row heterogeneity explanation
> is still incomplete.

