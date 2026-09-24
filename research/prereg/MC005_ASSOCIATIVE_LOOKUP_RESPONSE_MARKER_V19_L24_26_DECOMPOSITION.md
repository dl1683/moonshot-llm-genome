# MC005 Associative Lookup Response-Marker V19 L24-26 Decomposition Preregistration

Date: 2026-06-30

## Question

MC005 V18 confirmed `slice_l24_26_all` as the current supported Qwen3-1.7B
localization surface for the pair16 `Response:` associative-lookup behavior.
It recovered 92.4 percent of the parent layers-23-26 effect on disjoint
holdout rows, preserved target/distractor/random source specificity, and kept
answer-absent nulls clean.

V19 asks:

> Can a smaller preregistered layer block or layer-head intersection inside
> layers 24-26 recover the parent `slice_l24_26_all` effect on disjoint
> holdouts while preserving source controls and answer-absent nulls?

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

## Candidate Set

Selectable smaller layer blocks:

- `single_l24_all`;
- `single_l25_all`;
- `single_l26_all`;
- `slice_l24_25_all`;
- `slice_l25_26_all`.

Selectable head partitions across layers 24-26:

- `l24_26_lower_heads`: heads 0-7;
- `l24_26_upper_heads`: heads 8-15;
- `l24_26_even_heads`: even-numbered heads;
- `l24_26_odd_heads`: odd-numbered heads;
- `l24_26_heads_0_3`: heads 0-3;
- `l24_26_heads_4_7`: heads 4-7;
- `l24_26_heads_8_11`: heads 8-11;
- `l24_26_heads_12_15`: heads 12-15.

Selectable layer-head intersections:

- `single_l24_lower_heads`;
- `single_l24_upper_heads`;
- `single_l25_lower_heads`;
- `single_l25_upper_heads`;
- `single_l26_lower_heads`;
- `single_l26_upper_heads`;
- `slice_l24_25_lower_heads`;
- `slice_l24_25_upper_heads`;
- `slice_l25_26_lower_heads`;
- `slice_l25_26_upper_heads`.

Selection rule:

- score every selectable candidate on discovery lookup rows using only the
  target source-value mask;
- select the candidate with the most negative discovery mean delta, breaking
  ties by larger target-win loss and then lexical candidate name.

No candidate outside this set can be promoted by V19.

## Rows

Discovery lookup:

- scenario: `lookup_pair16_response`;
- seeds: 89 and 97;
- 64 rows per seed.

Lookup holdout:

- scenario: `lookup_pair16_response`;
- seeds: 101 and 103;
- 96 rows per seed.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 107 and 109;
- 96 rows per seed.

These seeds are disjoint from V15-V18 lookup and null seeds.

## Metrics

Discovery metrics:

- baseline target-minus-distractor margin;
- target source-value mask mean delta and target-win loss for every selectable
  candidate, the parent benchmark, and context references.

Lookup holdout metrics:

- baseline target-minus-distractor margin;
- target, distractor, and random source-value mask mean deltas for the parent
  benchmark and the selected candidate;
- target source-value mask mean delta and target-win loss for every selectable
  candidate;
- selected-candidate effect share relative to the parent benchmark;
- selected-candidate holdout rank stability against unselected selectable
  candidates.

Null metrics:

- baseline target-minus-distractor margin;
- source value, non-source control value, earlier neutral colon, final label,
  and final colon mean deltas and target-win changes under the selected
  candidate.

## Pass Boundary

V19 supports a smaller layers-24-26 decomposition only if all criteria pass:

1. the parent `slice_l24_26_all` holdout target source-value mask has mean
   delta at most -1.0 and at least three target-win losses;
2. the parent target mask beats its distractor and random source controls by at
   least 0.50 mean delta;
3. the selected smaller candidate holdout target source-value mask has mean
   delta at most -1.0 and at least three target-win losses;
4. the selected candidate target mask beats its distractor and random source
   controls by at least 0.50 mean delta;
5. the selected candidate recovers at least 60 percent of the parent
   `slice_l24_26_all` target-mask mean-delta magnitude on the holdout;
6. the selected candidate remains the strongest selectable target-mask path on
   the holdout within a 0.25 mean-delta tolerance;
7. both answer-absent null holdout seeds are `clean_null` under the selected
   candidate.

## Diagnostic Classes

- `compact_l24_26_decomposition_supported`: all pass-boundary criteria pass.
- `parent_not_replicated`: the parent effect or parent source controls fail on
  the V19 holdout.
- `no_smaller_effect`: the selected smaller candidate misses the basic target
  effect criteria.
- `selected_source_control_failed`: selected distractor or random source
  controls match the target effect.
- `l24_26_block_still_required`: the selected candidate is directional and
  source-specific but misses the 60 percent parent-effect share.
- `discovery_unstable`: the selected discovery winner does not remain the
  strongest selectable holdout path within tolerance.
- `null_failed`: answer-absent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V19 still does not prove a semantic circuit. A pass would only narrow the
supported Qwen3-1.7B source-value intervention surface below layers 24-26 under
this synthetic lookup contract and reliability suite. A fail keeps layers 24-26
as the current supported localization.
