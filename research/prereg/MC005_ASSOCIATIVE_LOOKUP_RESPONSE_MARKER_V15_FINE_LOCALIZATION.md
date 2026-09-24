# MC005 Associative Lookup Response-Marker V15 Fine Localization Preregistration

Date: 2026-06-30

## Question

MC005 now has a Qwen3-1.7B `Response:` late-band source-value control surface
with lookup/off-target support through pair count 16 and an expanded
answer-absent null diagnostic that passed through pair count 16. The remaining
mechanism-card blocker is locality: the effect is still broad over layers
20-26 and all attention heads.

V15 asks:

> Can a smaller preregistered layer slice or head group inside the Qwen3-1.7B
> late band recover the lookup intervention on held-out rows while preserving
> target/distractor/random source controls and the V14 pair16 answer-absent
> null?

## Fixed Surface

V15 does not reselect task, model, marker, answer tokens, or prompt family.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- pair count 16;
- final answer marker: `Response:`.

Fixed source-mask mechanism:

- eager-attention source mask at the final answer position;
- source key for lookup target arm: target source value;
- controls: distractor source value and random source value.

## Candidate Paths

The full benchmark is not eligible for compact localization selection:

- `full_l20_26_all`: layers 20-26, all heads.

Compact candidates eligible for selection:

| Candidate | Layers | Heads |
| --- | --- | --- |
| `single_l20_all` | 20 | all |
| `single_l21_all` | 21 | all |
| `single_l22_all` | 22 | all |
| `single_l23_all` | 23 | all |
| `single_l24_all` | 24 | all |
| `single_l25_all` | 25 | all |
| `single_l26_all` | 26 | all |
| `slice_l20_22_all` | 20-22 | all |
| `slice_l23_24_all` | 23-24 | all |
| `slice_l25_26_all` | 25-26 | all |
| `full_l20_26_lower_heads` | 20-26 | heads 0-7 |
| `full_l20_26_upper_heads` | 20-26 | heads 8-15 |
| `full_l20_26_even_heads` | 20-26 | even heads |
| `full_l20_26_odd_heads` | 20-26 | odd heads |

## Data Splits

Lookup discovery:

- scenario: `lookup_pair16_response`;
- seeds: 17 and 23;
- 64 rows per seed.

Lookup holdout:

- scenario: `lookup_pair16_response`;
- seed: 31;
- 64 rows.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 37 and 41;
- 64 rows per seed.

The null holdout reuses the V14 answer-absent row-generation contract but is
smaller because V14 already established the expanded null result at 128 rows.

## Selection Rule

For each compact candidate, V15 scores only the discovery lookup target source
value mask. The selected compact candidate is the candidate with the most
negative discovery mean target-vs-distractor margin delta, breaking ties by
larger target-win loss and then by lexicographic candidate name.

The full benchmark is scored on discovery and holdout but cannot be selected.

## Metrics

Lookup metrics:

- baseline target-minus-distractor margin;
- baseline clean rows;
- target, distractor, and random source mask mean deltas;
- target-win loss and target-win flips;
- selected compact target-effect share relative to full-band target effect on
  lookup holdout.

Null metrics:

- baseline target-minus-distractor margin;
- baseline clean rows;
- source value, non-source control value, earlier neutral colon, final label,
  and final colon mean deltas;
- target-win changes for every null arm.

## Pass Boundary

V15 supports a compact localization candidate only if all criteria pass:

1. the selected compact candidate's lookup-holdout target source-value mask has
   mean delta at most -1.0;
2. the selected compact candidate's lookup-holdout target source-value mask
   causes at least three target-win losses;
3. the selected compact candidate's lookup-holdout target source-value mask
   beats both its distractor and random source controls by at least 0.50 mean
   delta;
4. the selected compact candidate recovers at least 60 percent of the full
   `full_l20_26_all` target-mask mean-delta magnitude on lookup holdout;
5. both answer-absent null holdout seeds are `clean_null` under the selected
   compact candidate for all five null arms.

If any criterion fails, V15 remains a fine-localization diagnostic and MC005
stays broad over layers 20-26/all heads.

## Diagnostic Classes

- `compact_localization_supported`: all pass-boundary criteria pass.
- `weak_compact_localization`: selected compact candidate moves lookup in the
  predicted direction but misses effect-share, control, or null criteria.
- `full_band_required`: no compact candidate reaches 60 percent of full-band
  holdout target-effect magnitude.
- `control_failed`: selected compact target effect is matched by distractor or
  random source controls.
- `null_failed`: selected compact candidate fails the answer-absent null
  holdout.
- `no_compact_effect`: selected compact candidate misses the basic held-out
  mean-delta or target-win-loss criteria.
