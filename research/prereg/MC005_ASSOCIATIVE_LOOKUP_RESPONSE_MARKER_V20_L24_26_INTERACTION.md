# MC005 Associative Lookup Response-Marker V20 L24-26 Interaction Preregistration

Date: 2026-06-30

## Question

MC005 V18 promoted `slice_l24_26_all` as the current supported Qwen3-1.7B
localization surface. MC005 V19 tried to promote a smaller path inside that
block, but the best smaller path, `slice_l24_25_all`, recovered only 49.88
percent of the parent effect. That makes direct smaller-path promotion the
wrong next move.

V20 asks:

> Is the layers-24-26 parent effect better explained as a three-layer
> interaction where no leave-one-layer-out pair recovers the parent effect?

## Fixed Surface

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- pair count 16;
- final answer marker: `Response:`.

Fixed intervention:

- eager-attention source mask at the final answer position.

Parent benchmark:

- `slice_l24_26_all`: layers 24-26, all heads.

Context references:

- `slice_l23_26_all`: layers 23-26, all heads;
- `full_l20_26_all`: layers 20-26, all heads.

## Paths

Primary parent:

- `slice_l24_26_all`: layers 24-26, all heads.

Leave-one-layer-out pairs:

- `slice_l24_25_all`: layers 24-25, all heads, omits layer 26;
- `slice_l24_26_all_pair`: layers 24 and 26, all heads, omits layer 25;
- `slice_l25_26_all`: layers 25-26, all heads, omits layer 24.

Single-layer components:

- `single_l24_all`;
- `single_l25_all`;
- `single_l26_all`.

Source controls:

- parent target, distractor, and random source-value masks;
- pair target, distractor, and random source-value masks;
- single-layer target source-value masks.

## Rows

Lookup diagnostic:

- scenario: `lookup_pair16_response`;
- seeds: 113 and 127;
- 128 rows per seed.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 131 and 137;
- 96 rows per seed.

These seeds are disjoint from V15-V19 lookup and null seeds.

## Metrics

Lookup metrics:

- baseline target-minus-distractor margin;
- baseline target-win count;
- per-path target source-value mask mean delta and target-win loss;
- source-control mean deltas for parent and pair paths;
- effect share of each pair and single layer relative to the parent;
- residuals for:
  - `single_l24_all + single_l25_all + single_l26_all`;
  - `slice_l24_25_all + single_l26_all`;
  - `slice_l24_26_all_pair + single_l25_all`;
  - `slice_l25_26_all + single_l24_all`.

Null metrics:

- parent source value, non-source control value, earlier neutral colon, final
  label, and final colon mean deltas and target-win changes.

## Pass Boundary

V20 supports the three-layer-interaction interpretation only if all criteria
pass:

1. parent `slice_l24_26_all` target source-value mask has mean delta at most
   -1.0 and at least three target-win losses;
2. parent target source-value mask beats distractor and random source controls
   by at least 0.50 mean delta;
3. every leave-one-layer-out pair recovers less than 60 percent of the parent
   target-mask mean-delta magnitude;
4. the sum of single-layer mean deltas is superadditive relative to the parent,
   with residual fraction below -0.20;
5. both answer-absent null holdout seeds are `clean_null` under the parent path.

## Diagnostic Classes

- `l24_26_three_layer_interaction_supported`: all pass-boundary criteria pass.
- `parent_not_replicated`: the parent effect or parent source controls fail on
  the V20 lookup diagnostic.
- `pair_candidate_reopened`: at least one leave-one-layer-out pair recovers at
  least 60 percent of the parent effect.
- `not_superadditive`: pair effects are weak, but single-layer additivity does
  not show a superadditive parent residual.
- `null_failed`: answer-absent parent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V20 does not promote a smaller surface. A pass says the current layers-24-26
surface should be treated as an interaction block, and future work should
study interaction structure rather than directly selecting a smaller path on
the same controls.
