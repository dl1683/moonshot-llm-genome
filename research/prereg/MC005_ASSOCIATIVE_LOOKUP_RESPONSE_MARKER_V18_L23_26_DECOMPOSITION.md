# MC005 Associative Lookup Response-Marker V18 L23-26 Decomposition Preregistration

Date: 2026-06-30

## Question

MC005 V17 confirmed `slice_l23_26_all` as a held-out Qwen3-1.7B
localization surface for the pair16 `Response:` associative-lookup behavior.
That result narrows the supported source-value intervention from layers 20-26
to layers 23-26, but it does not show whether the effect is carried by a
smaller layer block or head partition inside layers 23-26.

V18 asks:

> Can a smaller preregistered layer block or head partition inside layers 23-26
> recover the parent `slice_l23_26_all` effect on disjoint holdouts while
> preserving source controls and answer-absent nulls?

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

- `slice_l23_26_all`: layers 23-26, all heads.

Global reference:

- `full_l20_26_all`: layers 20-26, all heads.

## Candidate Set

Selectable smaller layer blocks:

- `single_l23_all`;
- `single_l24_all`;
- `single_l25_all`;
- `single_l26_all`;
- `slice_l23_24_all`;
- `slice_l24_25_all`;
- `slice_l25_26_all`;
- `slice_l23_25_all`;
- `slice_l24_26_all`.

Selectable head partitions across layers 23-26:

- `l23_26_lower_heads`: heads 0-7;
- `l23_26_upper_heads`: heads 8-15;
- `l23_26_even_heads`: even-numbered heads;
- `l23_26_odd_heads`: odd-numbered heads;
- `l23_26_heads_0_3`: heads 0-3;
- `l23_26_heads_4_7`: heads 4-7;
- `l23_26_heads_8_11`: heads 8-11;
- `l23_26_heads_12_15`: heads 12-15.

Selection rule:

- score every selectable candidate on discovery lookup rows using only the
  target source-value mask;
- select the candidate with the most negative discovery mean delta, breaking
  ties by larger target-win loss and then lexical candidate name.

No candidate outside this set can be promoted by V18.

## Rows

Discovery lookup:

- scenario: `lookup_pair16_response`;
- seeds: 61 and 67;
- 64 rows per seed.

Lookup holdout:

- scenario: `lookup_pair16_response`;
- seeds: 71 and 73;
- 96 rows per seed.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 79 and 83;
- 96 rows per seed.

These seeds are disjoint from V15-V17 lookup and null seeds.

## Metrics

Discovery metrics:

- baseline target-minus-distractor margin;
- target source-value mask mean delta and target-win loss for every selectable
  candidate, the parent benchmark, and the global reference.

Lookup holdout metrics:

- baseline target-minus-distractor margin;
- target, distractor, and random source-value mask mean deltas for the parent
  benchmark and the selected candidate;
- target source-value mask mean delta and target-win loss for every selectable
  candidate;
- selected-candidate effect share relative to the parent benchmark;
- selected-candidate holdout rank stability against unselected candidates.

Null metrics:

- baseline target-minus-distractor margin;
- source value, non-source control value, earlier neutral colon, final label,
  and final colon mean deltas and target-win changes under the selected
  candidate.

## Pass Boundary

V18 supports a smaller layers-23-26 decomposition only if all criteria pass:

1. the parent `slice_l23_26_all` holdout target source-value mask has mean
   delta at most -1.0 and at least three target-win losses;
2. the parent target mask beats its distractor and random source controls by at
   least 0.50 mean delta;
3. the selected smaller candidate holdout target source-value mask has mean
   delta at most -1.0 and at least three target-win losses;
4. the selected candidate target mask beats its distractor and random source
   controls by at least 0.50 mean delta;
5. the selected candidate recovers at least 60 percent of the parent
   `slice_l23_26_all` target-mask mean-delta magnitude on the holdout;
6. the selected candidate remains the strongest selectable target-mask path on
   the holdout within a 0.25 mean-delta tolerance;
7. both answer-absent null holdout seeds are `clean_null` under the selected
   candidate.

## Diagnostic Classes

- `compact_l23_26_decomposition_supported`: all pass-boundary criteria pass.
- `parent_not_replicated`: the parent effect or parent source controls fail on
  the V18 holdout.
- `no_smaller_effect`: the selected smaller candidate misses the basic target
  effect criteria.
- `selected_source_control_failed`: selected distractor or random source
  controls match the target effect.
- `l23_26_block_still_required`: the selected candidate is directional and
  source-specific but misses the 60 percent parent-effect share.
- `discovery_unstable`: the selected discovery winner does not remain the
  strongest selectable holdout path within tolerance.
- `null_failed`: answer-absent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V18 still does not prove a semantic circuit. A pass would only narrow the
supported Qwen3-1.7B source-value intervention surface below layers 23-26 under
this synthetic lookup contract and reliability suite. A fail keeps layers 23-26
as the current supported localization.
