# MC005 Associative Lookup Response-Marker V14 Pair16 Null Diagnostic Preregistration

Date: 2026-06-30

## Question

MC005 V13 extended the Qwen3-1.7B `Response:` late-band lookup surface through
pair count 16, but failed the strict longer-context reliability gate because
one seed-23 pair16 answer-absent null had a final-label mean delta of -0.5039,
just beyond the clean-null bound of 0.50.

V14 asks:

> Is the Qwen3-1.7B pair16 answer-absent final-label failure a sample-fragile
> 32-row boundary, a gradual pair-count effect, or a persistent reliability
> failure at longer context?

## Fixed Surface

V14 does not reselect model, marker, layer band, source arms, or thresholds.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- final answer marker: `Response:`.

## Scenarios

V14 tests only answer-absent nulls. It does not rerun lookup or off-target
scenarios because V13 already showed those were clean/working through pair
count 16.

| Scenario | Pair count | Arms | Expected label |
| --- | ---: | --- | --- |
| `answer_absent_pair12_response_null` | 12 | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |
| `answer_absent_pair14_response_null` | 14 | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |
| `answer_absent_pair16_response_null` | 16 | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |

## Seeds And Rows

Run seeds:

- 17;
- 23;
- 31;
- 37;
- 41.

Each seed builds 128 rows per pair count. Within every row, sampled reference
keys, reference values, required answers, distractor answers, and non-source
control values are kept disjoint under the existing MC005 V9 row-generation
contract.

## Metrics

For each seed and pair count:

- baseline target-minus-distractor margin;
- baseline clean rows;
- greedy next-token target/distractor/other counts;
- per-arm mean delta, min delta, max delta, and target-win change;
- row-level diagnostics for every arm;
- final-label row-delta quantiles and threshold exceedance counts.

## Labels

Null scenario label:

- `clean_null`: baseline clean rows at least 75 percent of rows, and every arm
  has absolute mean delta at most 0.50 and absolute target-win change at most
  one row.
- `weak_null`: baseline clean rows at least 75 percent of rows, and every arm
  has absolute mean delta at most 1.00 and absolute target-win change at most
  two rows.
- `side_effect`: baseline is valid but at least one arm exceeds the weak-null
  limits.
- `invalid_baseline`: fewer than 75 percent of baseline rows prefer the target
  answer.

These thresholds intentionally remain strict rather than scaling target-win
change with row count. V14 is a reliability diagnostic, not a power analysis
designed to relax the V13 boundary.

## Diagnostic Classes

V14 diagnostic class:

- `all_clean_sample_fragile`: all pair-count/seed scenarios are `clean_null`;
  the V13 pair16 failure should be treated as a 32-row sample-fragile boundary.
- `pair16_specific_final_label_boundary`: pair counts 12 and 14 are clean, and
  non-clean pair16 scenarios are caused only by the final-label arm.
- `length_gradient_final_label_boundary`: pair count 14 also shows non-clean
  final-label arms, suggesting the boundary appears gradually before pair16.
- `broader_answer_absent_boundary`: any source value, non-source control value,
  earlier neutral colon, or final-colon arm is non-clean, or pair count 12
  becomes non-clean.
- `invalid_baseline`: any scenario fails the baseline floor.

## Pass Boundary

V14 passes the null repair only if every pair-count/seed answer-absent scenario
is `clean_null`.

If V14 passes, MC005 may treat V13's pair16 failure as sample-fragile but must
still keep the claim scoped to answer-absent contexts up to pair16 under this
row-generation contract. If V14 fails, the failing pair count, seed, and arm
define the current Qwen3-1.7B longer-context answer-absent boundary.
