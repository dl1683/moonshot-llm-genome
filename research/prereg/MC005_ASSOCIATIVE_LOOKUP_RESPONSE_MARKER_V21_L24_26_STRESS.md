# MC005 Associative Lookup Response-Marker V21 L24-26 Stress Preregistration

Date: 2026-06-30

## Question

MC005 V20 supports the all-head layers-24-26 source-value mask as a
three-layer interaction block on pair16 `Response:` associative lookup rows.
The next reliability question is whether that interaction survives harder
surface variation, not whether a smaller path can replace it.

V21 asks:

> Does the layers-24-26 interaction block survive layout, shifted-lexicon, and
> pair-count stress while preserving target/distractor/random source controls
> and answer-absent nulls?

## Fixed Surface

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- final answer marker: `Response:`.

Fixed intervention:

- eager-attention source mask at the final answer position;
- all-head parent path `slice_l24_26_all`.

## Lookup Stress Scenarios

V21 evaluates only the parent path:

- `pair16_dash_base`: pair count 16, dash/colon layout, base lexicon;
- `pair16_arrow_layout`: pair count 16, arrow layout, base lexicon;
- `pair16_sentence_layout`: pair count 16, sentence layout, base lexicon;
- `pair16_dash_lexicon_shift`: pair count 16, dash/colon layout, lexicon
  offset 24;
- `pair20_dash_pairstress`: pair count 20, dash/colon layout, base lexicon;
- `pair20_arrow_lexicon_stress`: pair count 20, arrow layout, lexicon offset
  24.

Each lookup scenario uses 64 rows and a fixed scenario seed:

- 149, 151, 157, 163, 167, and 173 respectively.

These seeds are disjoint from V15-V20 lookup and null seeds.

## Null Holdout

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 179 and 181;
- 96 rows per seed.

## Metrics

Lookup metrics per stress scenario:

- baseline target-minus-distractor margin;
- baseline target-win count;
- baseline greedy next-token label counts;
- parent target, distractor, and random source-value mask mean deltas;
- target-win losses;
- lookup label: `works`, `weak`, `breaks`, or `invalid_baseline`.

Null metrics:

- source value, non-source control value, earlier neutral colon, final label,
  and final colon mean deltas and target-win changes under `slice_l24_26_all`;
- null label: `clean_null`, `weak_null`, `side_effect`, or
  `invalid_baseline`.

## Pass Boundary

V21 supports stress reliability only if all criteria pass:

1. every lookup stress scenario has baseline target wins at least 75 percent of
   rows;
2. every lookup stress scenario has target source-value mean delta at most
   -1.0 and at least three target-win losses;
3. every lookup stress scenario target source-value mask beats distractor and
   random source controls by at least 0.50 mean delta;
4. both answer-absent null holdout seeds are `clean_null`.

## Diagnostic Classes

- `l24_26_stress_supported`: all pass-boundary criteria pass.
- `stress_baseline_failed`: at least one lookup stress scenario has an invalid
  baseline.
- `stress_effect_failed`: at least one valid-baseline lookup stress scenario
  lacks the parent target effect.
- `stress_source_control_failed`: at least one lookup stress scenario has
  distractor or random source controls matching the target effect.
- `stress_null_failed`: an answer-absent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V21 does not prove arbitrary-context reliability. A pass only extends the V20
layers-24-26 interaction block across the listed stress axes.
