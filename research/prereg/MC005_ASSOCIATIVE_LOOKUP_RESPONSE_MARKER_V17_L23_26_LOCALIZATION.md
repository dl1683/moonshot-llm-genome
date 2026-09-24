# MC005 Associative Lookup Response-Marker V17 L23-26 Localization Preregistration

Date: 2026-06-30

## Question

MC005 V16 found that the full layers-20-26 all-head effect is
mixed-superadditive across the tested partitions. The strongest new block was
`slice_l23_26_all`, with mean delta -6.2678 and 51 target-win losses on the V16
diagnostic bank. V16 did not promote that block because it did not include
fresh holdouts or answer-absent nulls for `slice_l23_26_all`.

V17 asks:

> Does `slice_l23_26_all` pass a held-out localization gate against the full
> layers-20-26 benchmark while preserving source controls and answer-absent
> nulls?

## Fixed Surface

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- pair count 16;
- final answer marker: `Response:`.

Fixed intervention:

- eager-attention source mask at the final answer position.

## Paths

Primary candidate:

- `slice_l23_26_all`: layers 23-26, all heads.

Benchmark:

- `full_l20_26_all`: layers 20-26, all heads.

Smaller-slice controls:

- `slice_l20_22_all`: layers 20-22, all heads;
- `slice_l23_24_all`: layers 23-24, all heads;
- `slice_l25_26_all`: layers 25-26, all heads.

Source controls:

- target source-value mask;
- distractor source-value mask;
- random source-value mask.

## Rows

Lookup holdout:

- scenario: `lookup_pair16_response`;
- seeds: 43 and 47;
- 96 rows per seed.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 53 and 59;
- 96 rows per seed.

These seeds are disjoint from V16's lookup seeds 17, 23, and 31.

## Metrics

Lookup metrics:

- baseline target-minus-distractor margin;
- baseline target-win count;
- per-path target, distractor, and random source mask mean deltas;
- target-win losses;
- candidate effect share relative to full benchmark;
- candidate margin over smaller-slice controls.

Null metrics:

- baseline target-minus-distractor margin;
- baseline target-win count;
- source value, non-source control value, earlier neutral colon, final label,
  and final colon mean deltas and target-win changes under `slice_l23_26_all`.

## Pass Boundary

V17 supports `slice_l23_26_all` localization only if all criteria pass:

1. full benchmark target source-value mask has mean delta at most -1.0 and at
   least three target-win losses;
2. `slice_l23_26_all` target source-value mask has mean delta at most -1.0 and
   at least three target-win losses;
3. `slice_l23_26_all` target source-value mask beats its distractor and random
   source controls by at least 0.50 mean delta;
4. `slice_l23_26_all` recovers at least 60 percent of the full benchmark
   target-mask mean-delta magnitude;
5. `slice_l23_26_all` target source-value mask beats every smaller-slice target
   mask by at least 0.50 mean delta;
6. both answer-absent null holdout seeds are `clean_null` under
   `slice_l23_26_all`.

## Diagnostic Classes

- `l23_26_localization_supported`: all pass-boundary criteria pass.
- `full_band_still_required`: candidate is directional and source-specific but
  misses the 60 percent full-effect share.
- `smaller_slice_not_separated`: candidate is directional but does not beat
  smaller-slice controls.
- `source_control_failed`: distractor or random source controls match the
  candidate target effect.
- `null_failed`: answer-absent null holdout fails.
- `no_l23_26_effect`: candidate misses the basic target-effect criteria.

V17 still does not prove a single-head or single-layer circuit. A pass would
only narrow the supported Qwen3-1.7B intervention surface from layers 20-26 to
layers 23-26 under this prompt family and reliability suite.
