# MC005 Associative Lookup Response-Marker V11 Weak-Null Diagnostic Preregistration

Date: 2026-06-30

## Question

MC005 V10 showed that the `Response:` lookup intervention replicated on
`Qwen/Qwen3-0.6B`, but strict size replication failed because seed 23 in the
answer-absent null was `weak_null`. The non-source-control-value arm changed
two target-win rows despite only +0.039 mean margin delta.

V11 asks:

> Is the Qwen3-0.6B answer-absent weak null a persistent reliability boundary,
> or a small-row-count / low-margin flip artifact?

## Fixed Scope

V11 is diagnostic only. It does not promote MC005 even if the larger diagnostic
is clean.

Fixed model:

- `Qwen/Qwen3-0.6B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- final answer marker: `Response:`.

Fixed scenario:

- `answer_absent_response_null`;
- pair count 5;
- 128 rows per seed;
- seeds 17, 23, 31, 37, and 41;
- arms: source value, non-source control value, earlier neutral colon, final
  label, final colon.

The first 32 rows for seeds 17, 23, and 31 are intended to match the V10
answer-absent rows exactly except for row id prefix. This gives a parity check
for the V10 failure while also measuring a larger row bank.

Within every row, sampled reference values and answer/control values are kept
disjoint from sampled reference keys.

## Metrics

For each seed:

- full 128-row baseline clean rows;
- first-32 baseline clean rows;
- arm mean delta;
- arm min and max delta;
- target-win loss;
- target-to-distractor flips;
- distractor-to-target flips;
- per-row baseline margin, arm margin, delta, greedy labels, and flip type;
- flip-row baseline absolute margins;
- count of flip rows with baseline absolute margin at most 1.0.

## Labels

The same V9/V10 null labels are reused:

- `clean_null`: baseline clean rows at least 24/32-equivalent and every arm has
  absolute mean delta at most 0.50 and absolute target-win change at most one
  row.
- `weak_null`: baseline is valid and every arm has absolute mean delta at most
  1.00 and absolute target-win change at most two rows, but at least one arm
  fails `clean_null`.
- `side_effect`: baseline is valid but at least one arm exceeds the weak-null
  limits.
- `invalid_baseline`: baseline target wins are below the baseline floor.

For 128-row V11 summaries, the baseline floor scales to 96/128. The per-arm
target-win clean limit remains one row; this intentionally keeps the diagnostic
strict.

## Diagnostic Boundary

V11 reports one diagnostic class:

- `v10_not_reproduced`: the first-32 seed-23 non-source-control weak-null
  pattern does not reproduce.
- `sample_fragile_low_margin`: the first-32 seed-23 weak null reproduces, all
  five 128-row seed summaries are `clean_null`, and all reproduced flip rows
  have baseline absolute margin at most 1.0.
- `sample_fragile`: the first-32 seed-23 weak null reproduces and all five
  128-row seed summaries are `clean_null`, but not all reproduced flip rows are
  low-margin by the 1.0 threshold.
- `persistent_boundary`: any 128-row seed summary is not `clean_null`.

Only `sample_fragile_low_margin` would support treating V10's strict failure as
a small-sample diagnostic artifact. `persistent_boundary` keeps the 0.6B
answer-absent null as an active reliability blocker.
