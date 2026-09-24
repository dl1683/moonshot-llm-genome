# MC005 Associative Lookup Response-Marker V12 0.6B Marker-Specificity Preregistration

Date: 2026-06-30

## Question

MC005 V11 showed that the Qwen3-0.6B `Response:` answer-absent null is a
persistent final-marker/control boundary. V12 asks:

> Is the smaller-model answer-absent boundary specific to `Response:`, or does
> it persist under the other V8 marker candidates?

## Fixed Scope

V12 is diagnostic. It does not promote MC005 by itself.

Fixed model:

- `Qwen/Qwen3-0.6B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position.

Fixed scenario:

- answer-absent null;
- pair count 5;
- 128 rows per marker/seed;
- seeds 17, 23, 31, 37, and 41;
- markers: `Response:`, `Output:`, `Answer:`, and `Result:`;
- arms: source value, non-source control value, earlier neutral colon, final
  label, final colon.

The `Response:` rows must match V11 row content for the same seed and row index
except for row id prefix. Marker variants change only the final marker text.

Within every row, sampled reference values and answer/control values are kept
disjoint from sampled reference keys.

## Metrics

For each marker and seed:

- full 128-row baseline clean rows;
- first-32 baseline clean rows;
- arm mean delta;
- arm min and max delta;
- target-win loss;
- target-to-distractor flips;
- distractor-to-target flips;
- per-row baseline margin, arm margin, delta, greedy labels, and flip type.

## Labels

V12 reuses the scaled V11 null labels:

- `clean_null`: baseline clean rows at least 96/128 and every arm has absolute
  mean delta at most 0.50 and absolute target-win change at most one row.
- `weak_null`: baseline is valid and every arm has absolute mean delta at most
  1.00 and absolute target-win change at most two rows, but at least one arm
  fails `clean_null`.
- `side_effect`: baseline is valid but at least one arm exceeds the weak-null
  limits.
- `invalid_baseline`: fewer than 96/128 baseline rows prefer the target answer.

First-32 slices use the original 24/32 baseline floor and one-row clean target
win-change limit.

## Diagnostic Boundary

V12 reports:

- marker label counts;
- clean seed counts per marker;
- first-32 parity for `Response:` seed 23;
- whether any marker is clean on all five full 128-row seeds.

Diagnostic classes:

- `output_repairs_boundary`: `Output:` is clean on all five full seeds and
  `Response:` is not.
- `response_and_output_clean`: both `Response:` and `Output:` are clean on all
  five full seeds.
- `alternate_marker_repairs_boundary`: at least one non-`Response:` marker is
  clean on all five full seeds, but `Output:` is not.
- `no_clean_marker`: no marker is clean on all five full seeds.

Only a clean all-seed marker can justify returning to a full Qwen3-0.6B
size-replication atlas under that marker.
