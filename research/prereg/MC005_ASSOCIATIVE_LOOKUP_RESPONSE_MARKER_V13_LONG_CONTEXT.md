# MC005 Associative Lookup Response-Marker V13 Long-Context Preregistration

Date: 2026-06-30

## Question

MC005 V9 passed a compact `Response:` atlas on Qwen3-1.7B up to pair count 8.
V10-V12 showed that Qwen3-0.6B replicates lookup but fails strict
answer-absent null reliability. V13 returns to the Qwen3-1.7B positive surface
and asks:

> Does the fixed `Response:` late-band source-value control surface survive
> longer associative lookup contexts beyond pair count 8?

## Fixed Intervention

V13 does not reselect model, marker, layer band, source arms, or pass
thresholds.

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

Each seed builds 32 rows per scenario. Within every row, sampled reference
values and answer/control values are kept disjoint from sampled reference keys.

## Scenarios

| Scenario | Pair count | Mode | Arms | Expected label |
| --- | ---: | --- | --- | --- |
| `lookup_pair12_response` | 12 | lookup | target source value, distractor source value, random source value | works |
| `lookup_pair16_response` | 16 | lookup | target source value, distractor source value, random source value | works |
| `offtarget_pair12_response` | 12 | off-target lookup | irrelevant source value, source key, random source value | clean null |
| `offtarget_pair16_response` | 16 | off-target lookup | irrelevant source value, source key, random source value | clean null |
| `answer_absent_pair12_response_null` | 12 | answer-absent null | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |
| `answer_absent_pair16_response_null` | 16 | answer-absent null | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |

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

V13 passes longer-context reliability only if every lookup scenario is `works`
and every null scenario is `clean_null` across all three seeds.

If V13 passes, MC005 gains Qwen3-1.7B longer-context support through pair count
16, but still needs finer head/path decomposition before mechanism-card
promotion. If V13 fails, the failing pair count and scenario define the current
Qwen3-1.7B context-length boundary.
