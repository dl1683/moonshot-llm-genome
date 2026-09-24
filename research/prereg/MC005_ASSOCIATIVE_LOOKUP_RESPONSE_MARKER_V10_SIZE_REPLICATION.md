# MC005 Associative Lookup Response-Marker V10 Size-Replication Preregistration

Date: 2026-06-30

## Question

MC005 V9 passed a compact `Response:` reliability atlas on
`Qwen/Qwen3-1.7B`. V10 asks:

> Does the same `Response:` late-band source-value control surface replicate on
> the smaller `Qwen/Qwen3-0.6B` model, or was V9 a single-size result?

## Fixed Intervention

V10 does not reselect a layer band, marker, seed set, lexicon policy, prompt
grammar, or pass boundary.

Fixed model:

- `Qwen/Qwen3-0.6B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- final answer marker: `Response:`.

The Qwen3-0.6B config has 28 transformer layers, so the V9 `late_20_26` band is
well-defined for direct size replication.

## Seeds

Run seeds:

- 17;
- 23;
- 31.

Each seed builds the same scenario set with 32 rows per scenario.

Within every row, sampled reference values and answer/control values are kept
disjoint from sampled reference keys. This preserves the V9 row-hygiene rule.

## Scenarios

| Scenario | Pair count | Mode | Arms | Expected label |
| --- | ---: | --- | --- | --- |
| `lookup_pair5_response` | 5 | lookup | target source value, distractor source value, random source value | works |
| `lookup_pair8_response` | 8 | lookup | target source value, distractor source value, random source value | works |
| `offtarget_pair5_response` | 5 | off-target lookup | irrelevant source value, source key, random source value | clean null |
| `answer_absent_response_null` | 5 | answer-absent null | source value, non-source control value, earlier neutral colon, final label, final colon | clean null |

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

V10 passes size replication only if every lookup scenario is `works` and every
null scenario is `clean_null` across all three seeds on `Qwen/Qwen3-0.6B`.

If V10 fails, MC005 remains a Qwen3-1.7B-specific compact atlas until a
different replication path passes. If V10 passes, MC005 gains size-replication
support but still needs longer-context holdouts and finer head/path
decomposition before mechanism-card promotion.
