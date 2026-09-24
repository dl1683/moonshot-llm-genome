# MC005 Associative Lookup Response-Marker V9 Preregistration

Date: 2026-06-30

## Question

MC005 V8 found two clean final markers, `Response:` and `Output:`, while
`Answer:` and `Result:` remained marker-specific boundaries. V9 asks:

> If we fix the clean `Response:` marker, do the lookup intervention and the
> repaired off-target/source-line nulls all pass in one compact atlas?

## Fixed Intervention

V9 does not reselect a layer band.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- final answer marker: `Response:`.

## Seeds

Run seeds:

- 17;
- 23;
- 31.

Each seed builds the same scenario set with 32 rows per scenario.

Within every row, sampled reference values and answer/control values are kept
disjoint from sampled reference keys. This prevents a value-side intervention
from also being a key-token intervention in the same prompt.

## Scenarios

| Scenario | Pair count | Mode | Arms | Expected label |
| --- | ---: | --- | --- | --- |
| `lookup_pair5_response` | 5 | lookup | target source value, distractor source value, random source value | works |
| `lookup_pair8_response` | 8 | lookup | target source value, distractor source value, random source value | works |
| `offtarget_pair5_response` | 5 | off-target lookup | irrelevant source value, source key, random source value | clean null |
| `answer_absent_response_null` | 5 | answer-absent null | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |

Lookup scenarios query a reference key and score the queried value against a
reference distractor value. Target source-value masking should reduce the
target-vs-distractor margin, while distractor and random source-value masks
should not match the target effect.

Off-target scenarios keep the lookup grammar but mask an unrelated source line
while scoring the queried value. The irrelevant source value, paired key, and
random source value should be clean nulls.

The answer-absent null keeps the reference-pair grammar and the `Response:`
marker, but the scored answer words are specified outside the reference values.
This repeats the V8 clean-marker null with source/control arms retained.

## Metrics

For each seed and scenario:

- baseline target-minus-distractor margin;
- baseline clean rows;
- greedy next-token target/distractor/other counts;
- arm mean delta;
- arm min and max delta;
- target-win loss;
- absolute target-win change.

## Labels

Lookup scenario label:

- `works`: baseline clean rows at least 16/32, target source-value mask mean
  delta at most -1.0, target source-value mask target-win loss at least three
  rows, and target source-value mask beats each non-target arm by at least 0.50
  mean delta.
- `weak`: target source-value mask moves in the predicted negative direction
  but misses one or more `works` criteria.
- `breaks`: baseline behavior fails or target source-value mask does not move
  in the predicted direction.

Null scenario label:

- `clean_null`: baseline clean rows at least 24/32 and every arm has absolute
  mean delta at most 0.50 and absolute target-win change at most one row.
- `weak_null`: baseline clean rows at least 24/32 and every arm has absolute
  mean delta at most 1.00 and absolute target-win change at most two rows.
- `side_effect`: baseline is valid but at least one arm exceeds the weak-null
  limits.
- `invalid_baseline`: fewer than 24/32 baseline rows prefer the target answer.

## Pass Boundary

V9 passes the compact `Response:` marker atlas only if every lookup scenario is
`works` and every null scenario is `clean_null` across all three seeds.

If V9 passes, it still does not promote MC005 to a complete mechanism card by
itself. It would justify the next stage: model-family or size replication under
the fixed `Response:` marker, while preserving `Answer:` and `Result:` as
known marker-specific negative controls from V8.
