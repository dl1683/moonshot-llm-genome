# MC005 Associative Lookup Response-Marker V22 Row Interaction Preregistration

Date: 2026-06-30

## Question

MC005 V20 showed that `slice_l24_26_all` is superadditive in aggregate:
leave-one-layer-out pairs and single layers do not recover the parent mean
effect. MC005 V21 showed that the fixed parent block survives layout,
shifted-lexicon, pair-count, source-control, and answer-absent null stress.

V22 asks:

> Is the layers-24-26 interaction visible at the row level, or is the V20
> superadditivity only an aggregate mean artifact?

## Fixed Surface

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- pair count 16;
- final answer marker: `Response:`.

Fixed intervention:

- eager-attention source mask at the final answer position.

Parent path:

- `slice_l24_26_all`: layers 24-26, all heads.

Component paths:

- leave-one-layer-out pairs:
  - `slice_l24_25_all`;
  - `slice_l24_26_all_pair`;
  - `slice_l25_26_all`;
- single-layer components:
  - `single_l24_all`;
  - `single_l25_all`;
  - `single_l26_all`.

## Rows

Lookup diagnostic:

- scenario: `lookup_pair16_response`;
- seeds: 191 and 193;
- 128 rows per seed.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 197 and 199;
- 96 rows per seed.

These seeds are disjoint from V15-V21 lookup and null seeds.

## Metrics

Aggregate lookup metrics:

- baseline target-minus-distractor margin;
- baseline target-win count;
- parent target, distractor, and random source-value mask mean deltas;
- pair and single-layer target source-value mask mean deltas;
- target-win losses.

Row-level interaction metrics:

- per-row parent target source-value delta;
- per-row pair and single-layer target source-value deltas;
- per-row negative-effect share for each leave-one-layer-out pair relative to
  the parent;
- per-row residual for:
  - `single_l24_all + single_l25_all + single_l26_all`;
  - `slice_l24_25_all + single_l26_all`;
  - `slice_l24_26_all_pair + single_l25_all`;
  - `slice_l25_26_all + single_l24_all`;
- count and fraction of parent-effect rows that satisfy the all-three-row
  criterion below;
- count of rows where the parent flips a target win but all leave-one-layer-out
  pairs preserve the target win.

Null metrics:

- parent source value, non-source control value, earlier neutral colon, final
  label, and final colon mean deltas and target-win changes.

## Row Definitions

A parent-effect row is a lookup row where:

1. the baseline target wins over the distractor;
2. the parent target source-value mask delta is at most -1.0.

An all-three margin row is a parent-effect row where:

1. every leave-one-layer-out pair recovers less than 60 percent of the parent
   negative-effect magnitude;
2. the single-layer sum residual is at most -1.0;
3. every pair-plus-omitted-layer residual is at most -0.5.

A parent-flip pair-resistant row is a row where:

1. the baseline target wins over the distractor;
2. the parent target source-value mask makes the target lose;
3. all leave-one-layer-out pair masks preserve the target win.

## Pass Boundary

V22 supports row-level interaction structure only if all criteria pass:

1. lookup baseline target wins are at least 75 percent of rows;
2. parent target source-value mask has mean delta at most -1.0 and at least
   three target-win losses;
3. parent target source-value mask beats distractor and random source controls
   by at least 0.50 mean delta;
4. at least 24 rows are all-three margin rows;
5. at least 40 percent of parent-effect rows are all-three margin rows;
6. at least 8 rows are parent-flip pair-resistant rows;
7. median single-layer-sum residual on parent-effect rows is at most -1.0;
8. both answer-absent null holdout seeds are `clean_null` under the parent path.

## Diagnostic Classes

- `row_level_interaction_supported`: all pass-boundary criteria pass.
- `parent_not_replicated`: the parent effect or parent source controls fail.
- `mean_only_interaction`: aggregate interaction remains, but row-level
  all-three support fails.
- `flip_support_failed`: margin rows exist, but too few behavior-flip rows are
  pair-resistant.
- `null_failed`: answer-absent parent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V22 does not promote a smaller path. A pass says the current layers-24-26
surface has row-level all-three interaction evidence under the pair16
`Response:` lookup contract. A fail says the V20/V21 surface remains useful,
but the interaction interpretation should be treated as aggregate-only until a
stronger row-level account is found.
